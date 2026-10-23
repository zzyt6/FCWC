#!/usr/bin/env python3
"""Detect tag36h11 AprilTags using OpenCV in the CWC conda environment.

    python carmer/apriltag_detector.py
    python carmer/apriltag_detector.py --config carmer/apriltag_config.json
    python carmer/apriltag_detector.py --no-display --frames 600
    python carmer/apriltag_detector.py --quad-decimate 2.0
    python carmer/apriltag_detector.py --profile

The RealSense color stream is native 640x480@60. Capture, detection, and preview
use separate threads and latest-frame buffers; CPU/GIL contention can still affect their rates. Each result
stays paired with its source image. Pose uses the saved 640x480 intrinsics and
a 0.10 m tag edge by default. OpenCV's centered marker frame is rotated 180 deg
about local Z to match the printed AprilRobotics tag orientation: X right, Y up,
Z out of the front face. The frame rotates with the decoded marker orientation.
Colors: X red, Y green, Z blue. Pose maps tag coordinates into camera coordinates.
Roll/pitch/yaw describe this same tag-to-camera rotation (Rz(yaw) Ry(pitch)
Rx(roll)); they are not the cart's heading in the robot/world frame.
Cart yaw is reported in the gravity-leveled pelvis frame using nominal G1 camera extrinsics.
Settings load from apriltag_config.json beside this script; explicit CLI options override them.
The shipped config uses tags flat on the cart top, +X forward, +Y left, +Z up,
and live waist yaw plus pelvis IMU tilt. Visible valid tag headings are fused by a circular mean;
a single tag passes through, and no tags produce an unavailable yaw.
Set cart-to-tag-rpy-deg to null in the config to use the top/rear mounting presets.
The rear preset is for an upright tag on the cart rear, facing the robot.
Use --pelvis-attitude-source imu/kinematics/fused to select tilt compensation.
The legacy fixed mode uses --waist-yaw-deg and does not level against gravity.
Use --cart-tag-id for one tag or --cart-tag-ids for a group (the deployed cart uses 0 and 1).
Quad detection uses 2x internal downsampling by default. Camera frames and
returned coordinates stay at 640x480; use --quad-decimate 1 for small/distant tags.
With cart tag IDs, each frame is searched only in an 8 px-aligned window around the
cart tags of the previous frame (same corners as a full search, ~7 ms instead of
~20 ms, so detection keeps up with 60 FPS); any tracked tag missing from the window
re-searches that same frame in full. --no-tag-window always searches in full.
Pose axes use an adaptive low-pass filter by default to reduce small frame-to-frame
jitter while following larger motion; use --no-smoothing for raw pose output.
Camera X/Y translation follows motion faster, with independent filtering for Z
and rotation; --smooth-xy-beta controls the X/Y motion response.
Frames without tags are still displayed, with a "No AprilTag detected" notice.
Detection+pose and total processing times are reported in milliseconds.
Use --profile for rolling mean/p95 timings of individual processing and preview stages.
The native camera image is on the left; a separate panel lists timings and
per-tag angles. Text is drawn once to avoid font-weight-dependent ghosting.
Q/Esc quits. S saves the raw image, annotated image, and detection JSON.
"""

import argparse
from collections import deque
from contextlib import ExitStack
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import sys
import threading
import time

import cv2
import numpy as np

if __package__:
    from .cart_heading import CartHeadingEstimator, HeadingSmoother, fuse_cart_headings
    from .pelvis_attitude import DEFAULT_URDF, OPTICAL_TO_LINK, G1Kinematics, PelvisCompensator
    from .robot_feedback import RobotFeedback
    from .opencv_camera_viewer import CameraReader, FPS, HEIGHT, WIDTH, WARMUP, INTRINSICS_FILE, open_camera
else:
    from cart_heading import CartHeadingEstimator, HeadingSmoother, fuse_cart_headings
    from pelvis_attitude import DEFAULT_URDF, OPTICAL_TO_LINK, G1Kinematics, PelvisCompensator
    from robot_feedback import RobotFeedback
    from opencv_camera_viewer import CameraReader, FPS, HEIGHT, WIDTH, WARMUP, INTRINSICS_FILE, open_camera


FAMILY = "tag36h11"
WINDOW = "AprilTag tag36h11 - RealSense 640x480"
NO_TAG_MESSAGE = "No AprilTag detected - searching..."
SNAPSHOT_DIR = Path(__file__).resolve().parent / "apriltag_snapshots"
CONFIG_FILE = Path(__file__).resolve().with_name("apriltag_config.json")
DEFAULT_QUAD_DECIMATE = 2.0
DEFAULT_SMOOTH_XY_BETA = 40.0
DEFAULT_SMOOTH_XY_SPEED_CUTOFF_HZ = 3.0
DEFAULT_SMOOTH_XY_NOISE_ALLOWANCE_M = 0.002
# p_opencv = OPENCV_FROM_TAG @ p_tag. Reverse local X/Y, preserve local Z.
OPENCV_FROM_TAG = np.diag([-1.0, -1.0, 1.0])
POSE_CONVENTION = {
    "name": "opencv_aruco_centered_z180",
    "reference": "https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html",
    "transform": "p_camera = R_tag_to_camera @ p_tag + tvec_m",
    "tag_axes": "center origin; X right, Y up in AprilRobotics printed orientation; Z out of tag front face",
    "rotation_from_opencv": "R_tag_to_camera = R_opencv_to_camera @ diag(-1, -1, 1)",
    "corner_order": "OpenCV detectMarkers order; output-frame object points (+h,-h,0), (-h,-h,0), (-h,+h,0), (+h,+h,0)",
    "camera_axes": "optical center origin; X image-right, Y image-down, Z forward through lens",
    "axis_colors": {"X": "red", "Y": "green", "Z": "blue"},
    "rpy_order": ["roll", "pitch", "yaw"],
    "rpy_transform": "R_tag_to_camera = Rz(yaw) @ Ry(pitch) @ Rx(roll)",
    "rpy_reference_frame": "camera optical frame; not robot/world cart heading",
    "rpy_ranges_rad": "roll/yaw in [-pi, pi]; pitch in [-pi/2, pi/2]",
    "rpy_singular_convention": "At pitch +/-pi/2, roll/yaw are coupled; set yaw=0 and mark rpy_singular=true",
}
TIMING_CONVENTION = {
    "detection_ms": "Current frame: grayscale conversion, tag detection and raw pose/RPY estimation for all tags",
    "processing_ms": "Current frame: detection_ms plus pose smoothing, state alignment/URDF/pelvis tilt, per-tag cart headings and circular yaw fusion",
    "excluded": "Waiting for camera frames, annotation, image saving, display and X11/network transport",
    "stage_timings_ms": "Optional --profile wall times per frame; read_wait is waiting for a new captured frame, not camera read duration; tag includes quad search, corner refinement and decoding; pose includes PnP refinement, RPY and axis projection; cart_pose is the URDF joint-PnP cart pose (only with --publish-cart-pose)",
}


def load_intrinsics(path):
    calibration = json.loads(Path(path).read_text(encoding="utf-8"))
    stream = calibration["stream"]
    if stream.get("type") != "color" or (stream["width"], stream["height"]) != (WIDTH, HEIGHT):
        raise ValueError(f"Pose requires intrinsics for the native {WIDTH}x{HEIGHT} color stream")
    matrix = np.asarray(calibration["camera_matrix"], dtype=np.float64)
    distortion = np.asarray(calibration["distortion_coefficients"], dtype=np.float64).reshape(-1)
    if (matrix.shape != (3, 3) or not np.isfinite(matrix).all()
            or matrix[0, 0] <= 0 or matrix[1, 1] <= 0
            or not np.allclose(matrix[2], [0, 0, 1])):
        raise ValueError("Invalid camera intrinsic matrix")
    if distortion.size not in (4, 5, 8, 12, 14) or not np.isfinite(distortion).all():
        raise ValueError("Invalid camera distortion coefficients")
    # Current RealSense inverse-Brown coefficients are all zero. A nonzero inverse
    # model cannot be passed directly to OpenCV's forward Brown projection.
    model = calibration.get("distortion_model", "")
    if np.any(distortion != 0) and model not in ("distortion.brown_conrady", "brown_conrady", "plumb_bob"):
        raise ValueError(f"Nonzero distortion model {model!r} needs conversion before OpenCV pose estimation")
    return matrix, distortion


def rotation_to_rpy(rotation):
    """Return ([roll, pitch, yaw] in radians, singular) for Rz @ Ry @ Rx.

    This decomposes the supplied rotation; it does not change coordinate frames.
    At gimbal lock, choose yaw=0 while preserving the represented orientation.
    """
    rotation = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    if not np.isfinite(rotation).all():
        raise ValueError("Cannot convert a nonfinite rotation to roll/pitch/yaw")
    cos_pitch = float(np.hypot(rotation[0, 0], rotation[1, 0]))
    pitch = np.arctan2(-rotation[2, 0], cos_pitch)
    singular = cos_pitch < 1e-8
    if singular:
        roll = np.arctan2(-rotation[1, 2], rotation[1, 1])
        yaw = 0.0
    else:
        roll = np.arctan2(rotation[2, 1], rotation[2, 2])
        yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
    return np.array([roll, pitch, yaw], dtype=np.float64), singular


class TagPoseEstimator:
    def __init__(self, camera_matrix, distortion, tag_size=0.10, axis_length=None):
        self.camera_matrix = np.asarray(camera_matrix, dtype=np.float64)
        self.distortion = np.asarray(distortion, dtype=np.float64)
        self.tag_size = float(tag_size)
        self.axis_length = self.tag_size if axis_length is None else float(axis_length)
        if not all(np.isfinite(x) and x > 0 for x in (self.tag_size, self.axis_length)):
            raise ValueError("Tag size and axis length must be positive finite meters")
        half = self.tag_size / 2
        # OpenCV's centered ArUco frame: canonical TL, TR, BR, BL, with Y up.
        # This exact object-point order is also required by IPPE_SQUARE.
        self.object_points = np.array([
            [-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0],
        ], dtype=np.float64)
        length = self.axis_length
        self.axis_points = np.array([[0, 0, 0], [length, 0, 0], [0, length, 0], [0, 0, length]], dtype=np.float64)

    def estimate(self, opencv_corners):
        corners = np.asarray(opencv_corners, dtype=np.float64).reshape(4, 2)
        if not np.isfinite(corners).all() or abs(cv2.contourArea(corners.astype(np.float32))) < 1:
            return None
        # Use the decoded corner order directly, as in OpenCV's ArUco pose example.
        # Do not reorder by screen position or apply the AprilRobotics frame mapping.
        image_points = np.ascontiguousarray(corners)
        _, rotations, translations, _ = cv2.solvePnPGeneric(
            self.object_points, image_points, self.camera_matrix, self.distortion,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )
        best = None
        for rvec, tvec in zip(rotations, translations):
            if not np.isfinite(rvec).all() or not np.isfinite(tvec).all():
                continue
            # Refine both planar candidates against the four observed corners.
            # This also avoids IPPE's loss of precision near a 180-degree rotation.
            rvec, tvec = cv2.solvePnPRefineLM(
                self.object_points, image_points, self.camera_matrix, self.distortion,
                rvec.copy(), tvec.copy(),
            )
            if not np.isfinite(rvec).all() or not np.isfinite(tvec).all():
                continue
            rotation, _ = cv2.Rodrigues(rvec)
            points_camera = self.object_points @ rotation.T + tvec.reshape(3)
            if np.any(points_camera[:, 2] <= 0):
                continue
            projected, _ = cv2.projectPoints(self.object_points, rvec, tvec, self.camera_matrix, self.distortion)
            error = float(np.sqrt(np.mean(np.sum((projected.reshape(4, 2) - image_points) ** 2, axis=1))))
            if best is None or error < best[0]:
                best = error, rvec, tvec, rotation
        if best is None:
            return None
        error, rvec, tvec, rotation = best
        # Change the tag frame, not camera coordinates: reverse both in-plane
        # axes by a proper local 180-degree rotation. The tag center and Z remain.
        rotation = rotation @ OPENCV_FROM_TAG
        return self.pose_from_transform(rotation, tvec.reshape(3), error)

    def pose_from_transform(self, rotation, tvec, reprojection_rmse_px=0.0):
        """Build the drawable/output pose from a tag-to-camera transform."""
        rotation = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
        tvec = np.asarray(tvec, dtype=np.float64).reshape(3)
        if not np.isfinite(rotation).all() or not np.isfinite(tvec).all():
            return None
        rvec, _ = cv2.Rodrigues(rotation)
        axis_camera = self.axis_points @ rotation.T + tvec.reshape(3)
        if np.any(axis_camera[:, 2] <= 0):
            return None
        axes_px, _ = cv2.projectPoints(self.axis_points, rvec, tvec, self.camera_matrix, self.distortion)
        axes_px = axes_px.reshape(4, 2)
        if not np.isfinite(axes_px).all() or np.any(np.abs(axes_px) > 1e6):
            return None
        rpy, singular = rotation_to_rpy(rotation)
        return {
            "tag_size_m": self.tag_size, "axis_length_m": self.axis_length,
            "rvec": rvec.reshape(3).tolist(), "tvec_m": tvec.tolist(),
            "R_tag_to_camera": rotation.tolist(), "reprojection_rmse_px": float(reprojection_rmse_px),
            "rpy_rad": rpy.tolist(), "rpy_deg": np.rad2deg(rpy).tolist(), "rpy_singular": singular,
            "axes_px": axes_px.tolist(),  # origin, +X endpoint, +Y endpoint, +Z endpoint
        }


def _alpha(cutoff_hz, dt):
    cutoff_hz = max(float(cutoff_hz), 1e-6)
    return 1.0 / (1.0 + 1.0 / (2.0 * np.pi * cutoff_hz * max(float(dt), 1e-6)))


def _rotation_to_quaternion(rotation):
    """Convert a rotation matrix to [w, x, y, z] without scipy."""
    m = np.asarray(rotation, dtype=np.float64)
    trace = float(np.trace(m))
    if trace > 0:
        s = 2.0 * np.sqrt(trace + 1.0)
        q = np.array([0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s])
    else:
        i = int(np.argmax(np.diag(m)))
        if i == 0:
            s = 2.0 * np.sqrt(max(1.0 + m[0, 0] - m[1, 1] - m[2, 2], 1e-15))
            q = np.array([(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s])
        elif i == 1:
            s = 2.0 * np.sqrt(max(1.0 + m[1, 1] - m[0, 0] - m[2, 2], 1e-15))
            q = np.array([(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s])
        else:
            s = 2.0 * np.sqrt(max(1.0 + m[2, 2] - m[0, 0] - m[1, 1], 1e-15))
            q = np.array([(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s])
    return q / np.linalg.norm(q)


def _quaternion_to_rotation(q):
    w, x, y, z = np.asarray(q, dtype=np.float64) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _slerp(q0, q1, amount):
    q0 = np.asarray(q0, dtype=np.float64) / np.linalg.norm(q0)
    q1 = np.asarray(q1, dtype=np.float64) / np.linalg.norm(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0:
        q1, dot = -q1, -dot
    if dot > 0.9995:
        return ((1.0 - amount) * q0 + amount * q1) / np.linalg.norm((1.0 - amount) * q0 + amount * q1)
    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    sine = np.sin(theta)
    return (np.sin((1.0 - amount) * theta) * q0 + np.sin(amount * theta) * q1) / sine


class AdaptivePoseSmoother:
    """Per-tag One-Euro-style position and quaternion pose smoothing.

    Translation is filtered per camera axis. X/Y have a stronger motion response
    for lateral tracking; moving one axis does not relax filtering on the others
    or on rotation. Their unsigned motion estimate avoids cancellation at turns.
    A noise allowance affects only this estimate, never the position update.
    Each tag has independent state; a long gap resets the filter.
    """

    def __init__(self, pose_estimator, min_cutoff_hz=2.0, beta=0.35, derivative_cutoff_hz=1.0, reset_gap_s=0.5,
                 *, xy_beta=DEFAULT_SMOOTH_XY_BETA,
                 xy_speed_cutoff_hz=DEFAULT_SMOOTH_XY_SPEED_CUTOFF_HZ,
                 xy_noise_allowance_m=DEFAULT_SMOOTH_XY_NOISE_ALLOWANCE_M):
        self.pose_estimator = pose_estimator
        self.min_cutoff_hz = float(min_cutoff_hz)
        self.beta = float(beta)
        self.xy_beta = float(xy_beta)
        self.xy_speed_cutoff_hz = float(xy_speed_cutoff_hz)
        self.xy_noise_allowance_m = float(xy_noise_allowance_m)
        self.derivative_cutoff_hz = float(derivative_cutoff_hz)
        self.reset_gap_s = float(reset_gap_s)
        if (not all(np.isfinite(x) and x > 0 for x in (
                self.min_cutoff_hz, self.derivative_cutoff_hz, self.reset_gap_s, self.xy_speed_cutoff_hz))
                or not all(np.isfinite(x) and x >= 0 for x in (self.beta, self.xy_beta, self.xy_noise_allowance_m))):
            raise ValueError("Invalid pose smoothing parameters")
        self.states = {}

    @property
    def settings(self):
        return {
            "method": "adaptive_one_euro_axiswise_position_quaternion",
            "min_cutoff_hz": self.min_cutoff_hz,
            "beta": self.beta,
            "xy_beta": self.xy_beta,
            "xy_motion_estimator": "low_pass_absolute_innovation_above_noise_allowance",
            "xy_speed_cutoff_hz": self.xy_speed_cutoff_hz,
            "xy_noise_allowance_m": self.xy_noise_allowance_m,
            "position_axis_order": "camera_x_y_z",
            "derivative_cutoff_hz": self.derivative_cutoff_hz,
            "reset_gap_s": self.reset_gap_s,
        }

    def update(self, tag_id, raw_pose, timestamp):
        position = np.asarray(raw_pose["tvec_m"], dtype=np.float64)
        rotation = np.asarray(raw_pose["R_tag_to_camera"], dtype=np.float64)
        quaternion = _rotation_to_quaternion(rotation)
        state = self.states.get(int(tag_id))
        if state is None or timestamp - state["timestamp"] > self.reset_gap_s or timestamp <= state["timestamp"]:
            state = {"timestamp": timestamp, "position": position.copy(), "velocity": np.zeros(3),
                     "xy_speed": np.zeros(2),
                     "quaternion": quaternion.copy(), "angular_speed": 0.0}
            self.states[int(tag_id)] = state
            alpha_position = np.ones(3)
            alpha_rotation = 1.0
        else:
            dt = max(timestamp - state["timestamp"], 1e-4)
            innovation = position - state["position"]
            velocity = innovation / dt
            derivative_alpha = _alpha(self.derivative_cutoff_hz, dt)
            state["velocity"] = derivative_alpha * velocity + (1.0 - derivative_alpha) * state["velocity"]
            # Take magnitude BEFORE temporal smoothing, so opposite directions
            # cannot cancel and incorrectly increase lag at a turn. The noise
            # allowance reduces response to jitter without freezing small moves:
            # the full position innovation still feeds the position update below.
            xy_speed = np.maximum(np.abs(innovation[:2]) - self.xy_noise_allowance_m, 0.) / dt
            xy_speed_alpha = _alpha(self.xy_speed_cutoff_hz, dt)
            state["xy_speed"] = xy_speed_alpha * xy_speed + (1. - xy_speed_alpha) * state["xy_speed"]
            position_cutoff = self.min_cutoff_hz + self.beta * np.abs(state["velocity"])
            position_cutoff[:2] = self.min_cutoff_hz + self.xy_beta * state["xy_speed"]
            alpha_position = np.array([_alpha(cutoff, dt) for cutoff in position_cutoff])
            state["position"] = alpha_position * position + (1.0 - alpha_position) * state["position"]
            dot = abs(float(np.dot(state["quaternion"], quaternion)))
            angular_speed = 2.0 * np.arccos(np.clip(dot, -1.0, 1.0)) / dt
            state["angular_speed"] = derivative_alpha * angular_speed + (1.0 - derivative_alpha) * state["angular_speed"]
            rotation_cutoff = self.min_cutoff_hz + self.beta * state["angular_speed"]
            alpha_rotation = _alpha(rotation_cutoff, dt)
            state["quaternion"] = _slerp(state["quaternion"], quaternion, alpha_rotation)
            state["timestamp"] = timestamp
        smoothed_rotation = _quaternion_to_rotation(state["quaternion"])
        smoothed = self.pose_estimator.pose_from_transform(smoothed_rotation, state["position"], raw_pose.get("reprojection_rmse_px", 0.0))
        if smoothed is None:
            return raw_pose
        smoothed["smoothing"] = {
            **self.settings, "position_alpha_xyz": alpha_position.tolist(), "rotation_alpha": float(alpha_rotation),
            "source": "raw_pose",
        }
        return smoothed


def make_detector(quad_decimate=DEFAULT_QUAD_DECIMATE):
    quad_decimate = float(quad_decimate)
    if not np.isfinite(quad_decimate) or quad_decimate < 1:
        raise ValueError("--quad-decimate must be a finite number >= 1")
    if not hasattr(cv2, "aruco") or not hasattr(cv2.aruco, "ArucoDetector"):
        raise RuntimeError("This program needs cv2.aruco.ArucoDetector; use the CWC conda environment.")
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
    params.aprilTagQuadDecimate = quad_decimate
    return cv2.aruco.ArucoDetector(dictionary, params)


class TagWindow:
    """Search only around the cart tags found in the previous frame.

    The window snaps to an ALIGN-pixel grid so AprilTag quad decimation and
    threshold tiles sample the same pixels as a full-image search; recorded cart
    frames give bit-identical corners. If a tracked tag is missing, the caller
    searches the same frame in full, so the window never drops a detection.
    While only some cart tags are tracked, every FULL_EVERY-th frame is also
    searched in full to pick up the others.
    """
    ALIGN = 8
    FULL_EVERY = 30

    def __init__(self, tag_ids, margin_px=48, margin_scale=0.5):
        self.tag_ids = frozenset(tag_ids)
        self.margin_px, self.margin_scale = margin_px, margin_scale
        self.corners = {}
        self.partial_frames = 0

    def region(self, shape):
        if not self.corners:
            return None
        if self.corners.keys() != self.tag_ids:
            self.partial_frames += 1
            if self.partial_frames % self.FULL_EVERY == 0:
                return None
        points = np.vstack(list(self.corners.values()))
        low, high = points.min(axis=0), points.max(axis=0)
        tag_size = max(float(np.ptp(quad, axis=0).max()) for quad in self.corners.values())
        margin = max(self.margin_px, self.margin_scale * tag_size)
        x0, y0 = (np.maximum(low - margin, 0) // self.ALIGN * self.ALIGN).astype(int)
        x1, y1 = np.minimum(-(-(high + margin) // self.ALIGN) * self.ALIGN, (shape[1], shape[0])).astype(int)
        return int(x0), int(y0), int(x1), int(y1)

    def covers(self, tags):
        """True when every tracked tag was found again inside the window."""
        return self.corners.keys() <= {tag['id'] for tag in tags}

    def update(self, tags):
        self.corners = {tag['id']: np.asarray(tag['corners_px']) for tag in tags if tag['id'] in self.tag_ids}
        if self.corners.keys() == self.tag_ids:
            self.partial_frames = 0


def detect_tags(detector, frame, pose_estimator=None, *, stage_timings=None, region=None):
    """region=(x0, y0, x1, y1) searches only that window; coordinates stay full-image."""
    if stage_timings is not None:
        started = time.perf_counter()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    offset = np.zeros(2, dtype=np.float32)
    if region is not None:
        x0, y0, x1, y1 = region
        gray, offset = gray[y0:y1, x0:x1], np.array([x0, y0], dtype=np.float32)
    if stage_timings is not None:
        gray_done = time.perf_counter()
        stage_timings['gray'] = (gray_done - started) * 1000
    corners, ids, _ = detector.detectMarkers(gray)
    if stage_timings is not None:
        markers_done = time.perf_counter()
        stage_timings['tag'] = (markers_done - gray_done) * 1000
        stage_timings['pose'] = 0.0
    tags = []
    if ids is not None:
        for tag_id, quad in zip(ids.ravel(), corners):
            points = quad.reshape(4, 2).astype(np.float32) + offset
            # Project the square's center: averaging corners is biased under perspective.
            transform = cv2.getPerspectiveTransform(
                np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32), points
            )
            center = cv2.perspectiveTransform(np.array([[[0.5, 0.5]]], dtype=np.float32), transform)[0, 0]
            tag = {"id": int(tag_id), "center_px": center.tolist(), "corners_px": points.tolist()}
            if pose_estimator is not None:
                if stage_timings is not None:
                    pose_started = time.perf_counter()
                tag["pose"] = pose_estimator.estimate(points)
                if stage_timings is not None:
                    stage_timings['pose'] += (time.perf_counter() - pose_started) * 1000
            tags.append(tag)
    tags.sort(key=lambda tag: tag["id"])
    if stage_timings is not None:
        stage_timings['tag_output'] = max(0.0, (time.perf_counter() - markers_done) * 1000 - stage_timings['pose'])
    return tags


def format_profile(label, samples):
    """Summarize wall times; overlapping worker/preview stages must not be added."""
    if not samples:
        return None
    values = []
    for name in samples[0]:
        durations = [sample[name] for sample in samples]
        values.append(f"{name}={np.mean(durations):.2f}/{np.percentile(durations, 95):.2f}")
    return f"Profile {label} (last {len(samples)} frames, mean/p95 ms): " + " | ".join(values)


def reader_profile(reader):
    """Capture-thread cap.read() wall times (USB wait + color conversion)."""
    with reader.condition:
        samples = [{'cap_read': value} for value in reader.read_ms]
    return format_profile('capture', samples)


def format_fused_heading(heading):
    if not heading['valid']:
        return f"fused_cart_yaw_pelvis=unavailable({heading['reason']})"
    ids = ','.join(str(tag_id) for tag_id in heading['tag_ids'])
    return (f"fused_cart_yaw_pelvis_deg={heading['yaw_deg']:+.2f} "
            f"fused_cart_yaw_pelvis_rad={heading['yaw_rad']:+.5f} "
            f"yaw_source={heading['mode']} yaw_tag_ids=[{ids}]")


def format_compensation(state):
    if not state:
        return 'yaw_reference=pelvis (fixed waist, no gravity leveling)'
    prefix = f"yaw_reference=pelvis_level attitude={state['requested_source']}"
    if not state['valid']:
        return f"{prefix} unavailable({state['reason']})"
    return (f"{prefix}->{state['effective_source']} pelvis_rp_deg=({state['roll_deg']:+.2f},"
            f"{state['pitch_deg']:+.2f}) waist_yaw_deg={state['waist_yaw_deg']:+.2f} "
            f"sync_ms={state['sync_error_ms']:.1f} status={state['reason']}")


def _image_label(image, text, position, color, scale=0.5):
    """One text pass on a solid background, clipped to the camera viewport.

    OpenCV 5 can use different glyph advances for different thicknesses;
    drawing a thick black copy under a thin colored copy causes ghosting.
    """
    (width, height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    x = max(3, min(image.shape[1] - width - 4, int(position[0])))
    y = max(height + 3, min(image.shape[0] - baseline - 4, int(position[1])))
    cv2.rectangle(image, (x - 3, y - height - 3), (x + width + 3, y + baseline + 3), (25, 25, 25), -1)
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def _information_panel(image, tags, telemetry=None, preview_fps=None):
    """Append a panel; all image coordinates keep their original origin/scale."""
    telemetry = telemetry or {}
    white, muted, yellow = (235, 235, 235), (175, 180, 190), (90, 215, 255)
    green, cyan = (125, 235, 130), (245, 230, 100)
    width, padding = 420, 16
    rows = []

    def add(text, color=white, scale=0.48):
        # Fit using the same weight used for the final draw.
        measured = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0]
        while measured > width - 2 * padding:
            scale *= (width - 2 * padding - 2) / measured
            measured = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0]
        (_, height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        rows.append((text, color, scale, height, baseline))

    def gap():
        rows.append(None)

    def number(key, suffix=""):
        value = telemetry.get(key)
        return "--" if value is None else f"{value:.1f}{suffix}"

    add(f"AprilTag | {image.shape[1]}x{image.shape[0]} | Tags: {len(tags)}", scale=0.62)
    view_rate = "--" if preview_fps is None else f"{preview_fps:.1f}"
    add(f"FPS  Capture {number('capture_fps')} / Detect {number('detect_fps')} / View {view_rate}", muted)
    add(f"Detect+Pose {number('detection_ms', ' ms')} | Process {number('processing_ms', ' ms')}", yellow)
    settings = telemetry.get("cart_heading_settings", {})
    compensation = telemetry.get('pelvis_compensation')
    if compensation:
        if compensation['valid']:
            add(f"Attitude {compensation['requested_source']} -> {compensation['effective_source']} | Waist {compensation['waist_yaw_deg']:+.1f} deg", muted)
            add(f"Pelvis R/P: {compensation['roll_deg']:+.1f} / {compensation['pitch_deg']:+.1f} | Sync {compensation['sync_error_ms']:.1f} ms", muted)
            if compensation['reason'] != 'ok':
                add(compensation['reason'], yellow)
        else:
            add(f"Attitude unavailable: {compensation['reason']}", yellow)
    elif settings and settings.get('waist_yaw_deg') is not None:
        add(f"Mount {settings['tag_mount']} | Waist {settings['waist_yaw_deg']:+.1f} deg (fixed)", muted)
    add("Level pelvis yaw: +left / -right" if compensation else "Pelvis yaw: +left / -right", muted)
    gap()

    fused = telemetry.get('fused_cart_heading')
    if fused is None:
        fused = fuse_cart_headings(tags)
    if fused['valid']:
        add(f"Cart yaw (pelvis): {fused['yaw_deg']:+.2f} deg", cyan, 0.64)
        ids = ', '.join(str(tag_id) for tag_id in fused['tag_ids'])
        add(f"Fused tags: {ids}" if fused['mode'] == 'fused' else f"Single tag: {ids}", muted)
    else:
        add("Cart yaw (pelvis): unavailable", yellow, 0.6)
        if fused['reason'] == 'opposing_headings':
            add("Tag headings oppose each other", yellow)
        elif tags:
            add("No valid cart heading", yellow)
    if not tags:
        add(NO_TAG_MESSAGE, yellow)
    gap()
    target_id = settings.get("cart_tag_id")
    target_ids = settings.get('cart_tag_ids')
    if tags and target_id is not None and not any(tag['id'] == target_id for tag in tags):
        add(f"Cart tag ID {target_id} not detected", yellow)
        gap()
    for tag in sorted(tags, key=lambda item: item['id']):
        heading = tag.get("cart_heading")
        if heading is not None and heading["valid"]:
            add(f"ID {tag['id']} | yaw: {heading['yaw_deg']:+.2f} deg", green, 0.56)
        elif ((target_id is not None and tag['id'] != target_id)
              or (target_ids is not None and tag['id'] not in target_ids)):
            add(f"ID {tag['id']} | yaw: not selected", muted, 0.56)
        else:
            add(f"ID {tag['id']} | yaw: unavailable", yellow, 0.56)
        pose = tag.get("pose")
        if pose:
            roll, pitch, yaw = pose["rpy_deg"]
            add(f"Cam R/P/Y: {roll:+.1f} / {pitch:+.1f} / {yaw:+.1f} deg")
            if pose["rpy_singular"]:
                add("Camera RPY singular", yellow)
        else:
            add("Camera pose: unavailable", yellow)
        gap()

    pose = next((tag['pose'] for tag in tags if tag.get('pose')), None)
    if pose:
        add(f"Tag {pose['tag_size_m'] * 100:g} cm | Axes {pose['axis_length_m'] * 100:g} cm", muted)
    add("Axes: X red | Y green | Z blue", muted)
    add("Q/Esc: quit | S: save", muted)
    panel_height = 2 * padding + sum(8 if row is None else row[3] + row[4] + 8 for row in rows)
    # All three channels have the same background: scalar fill avoids broadcasting
    # an RGB tuple across every pixel in each preview frame.
    canvas = np.full((max(image.shape[0], panel_height), image.shape[1] + width, 3), 29, dtype=np.uint8)
    canvas[:image.shape[0], :image.shape[1]] = image
    x, y = image.shape[1] + padding, padding
    for row in rows:
        if row is None:
            y += 8
            continue
        text, color, scale, height, baseline = row
        y += height
        cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
        y += baseline + 8
    return canvas


def annotate(frame, tags, camera_matrix=None, distortion=None, *, telemetry=None, preview_fps=None):
    image = frame.copy()
    for tag in tags:
        corners = np.rint(tag["corners_px"]).astype(np.int32)
        center = tuple(np.rint(tag["center_px"]).astype(int))
        cv2.polylines(image, [corners], True, (0, 255, 0), 2, cv2.LINE_AA)
        pose = tag.get("pose")
        if pose:
            if camera_matrix is None or distortion is None:
                raise ValueError("Drawing pose axes requires the camera intrinsics")
            cv2.drawFrameAxes(
                image, np.asarray(camera_matrix, dtype=np.float64),
                np.asarray(distortion, dtype=np.float64),
                np.asarray(pose["rvec"], dtype=np.float64),
                np.asarray(pose["tvec_m"], dtype=np.float64),
                pose["axis_length_m"], 2,
            )
            origin, *endpoints = np.rint(pose["axes_px"]).astype(np.int32)
            for label, color, end in zip("XYZ", ((0, 0, 255), (0, 255, 0), (255, 0, 0)), endpoints):
                label_at = (max(2, min(image.shape[1] - 18, int(end[0]) + 5)), max(16, min(image.shape[0] - 4, int(end[1]) - 5)))
                _image_label(image, label, label_at, color, 0.5)
            cv2.circle(image, tuple(origin), 3, (255, 255, 255), -1, cv2.LINE_AA)
        else:
            cv2.circle(image, center, 4, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.circle(image, tuple(corners[0]), 4, (255, 0, 0), -1, cv2.LINE_AA)
        x = max(0, min(image.shape[1] - 100, int(corners[:, 0].min())))
        y = max(18, int(corners[:, 1].min()) - 8)
        _image_label(image, f"ID {tag['id']}", (x, y), (0, 255, 0), 0.6)
    return _information_panel(image, tags, telemetry, preview_fps)


class DetectionWorker:
    """Keep only the latest completed image/result pair instead of queuing stale frames."""

    def __init__(self, reader, detector, pose_estimator=None, pose_smoother=None, cart_heading_estimator=None, *, profile=False,
                 pelvis_compensator=None, heading_smoother=None, cart_pose_estimator=None, missing_tag_frames=None,
                 tag_window=False, detection_sink=None):
        self.reader = reader
        self.detector = detector
        # The window tracks cart tags only; without a cart ID set every frame is searched in full.
        cart_ids = getattr(cart_heading_estimator, 'cart_tag_ids', None)
        self.tag_window = TagWindow(cart_ids) if tag_window and cart_ids else None
        self.search_counts = dict(full=0, window=0, window_fallback=0)
        self.detector_settings = {
            "quad_decimate": float(detector.getDetectorParameters().aprilTagQuadDecimate),
            "opencv_threads": cv2.getNumThreads(),
            "tag_window": self.tag_window is not None,
        }
        self.pose_estimator = pose_estimator
        self.pose_smoother = pose_smoother
        self.cart_heading_estimator = cart_heading_estimator
        self.pelvis_compensator = pelvis_compensator
        self.heading_smoother = heading_smoother
        self.cart_pose_estimator = cart_pose_estimator
        self.missing_tag_frames = missing_tag_frames
        self.detection_sink = detection_sink
        self.profile_samples = deque(maxlen=120) if profile else None
        self.condition = threading.Condition()
        self.stop = threading.Event()
        self.result = None
        self.done = False
        self.error = None
        self.processed = 0
        self.tag_frames = 0
        self.seen_ids = set()
        self.first_time = None
        self.last_time = None
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        previous = 0
        recent = deque(maxlen=FPS + 1)
        try:
            while not self.stop.is_set():
                stages = {} if self.profile_samples is not None else None
                if stages is not None:
                    wait_started = time.perf_counter()
                source = self.reader.latest(previous, with_timestamp=True)
                if source is None:
                    break
                if stages is not None:
                    stages['read_wait'] = (time.perf_counter() - wait_started) * 1000
                previous, frame, capture_fps, frame_timestamp = source
                started = time.monotonic()
                region = self.tag_window.region(frame.shape) if self.tag_window is not None else None
                tags = detect_tags(self.detector, frame, self.pose_estimator, stage_timings=stages, region=region)
                search = 'full' if region is None else 'window'
                if region is not None and not self.tag_window.covers(tags):
                    window_stages = stages
                    stages = {} if stages is not None else None
                    tags = detect_tags(self.detector, frame, self.pose_estimator, stage_timings=stages)
                    if stages is not None:
                        # Keep every window stage (read_wait too): profile rows must share one key set.
                        stages = {name: value + stages.get(name, 0.) for name, value in window_stages.items()}
                    search = 'window_fallback'
                if self.tag_window is not None:
                    self.tag_window.update(tags)
                self.search_counts[search] += 1
                detection_finished = time.monotonic()
                if stages is not None:
                    smoothing_started = time.perf_counter()
                if self.pose_smoother is not None:
                    for tag in tags:
                        raw_pose = tag.get("pose")
                        if raw_pose is None:
                            continue
                        tag["raw_pose"] = raw_pose
                        tag["pose"] = self.pose_smoother.update(tag["id"], raw_pose, frame_timestamp)
                if stages is not None:
                    smoothing_done = time.perf_counter()
                    stages['smooth'] = (smoothing_done - smoothing_started) * 1000
                compensation = None
                if self.pelvis_compensator is not None:
                    compensation = self.pelvis_compensator.at_frame(frame_timestamp, time.monotonic())
                if stages is not None:
                    attitude_done = time.perf_counter()
                    stages['attitude'] = (attitude_done - smoothing_done) * 1000
                if self.cart_heading_estimator is not None:
                    if self.heading_smoother:
                        self.heading_smoother.retain({tag['id'] for tag in tags})
                    for tag in tags:
                        if self.cart_heading_estimator.applies_to(tag["id"]):
                            raw = self.cart_heading_estimator.estimate(tag.get("raw_pose", tag.get("pose")), compensation)
                            tag["raw_cart_heading"] = raw
                            if compensation is None:
                                tag["cart_heading"] = self.cart_heading_estimator.estimate(tag.get("pose"))
                            else:
                                # Match RAW camera pose and robot state in time before filtering.
                                tag["cart_heading"] = (self.heading_smoother.update(
                                    tag['id'], raw, frame_timestamp, compensation['effective_source'])
                                    if self.heading_smoother else dict(raw))
                    fused_heading = fuse_cart_headings(tags)
                    raw_fused_heading = fuse_cart_headings(tags, 'raw_cart_heading')
                    if compensation is not None:
                        for heading in (fused_heading, raw_fused_heading):
                            heading['reference_frame'] = 'pelvis_level'
                            if not compensation['valid'] and tags:
                                heading['reason'] = compensation['reason']
                if stages is not None:
                    stages['yaw'] = (time.perf_counter() - attitude_done) * 1000
                now = time.monotonic()
                recent.append(now)
                detect_fps = (len(recent) - 1) / (recent[-1] - recent[0]) if len(recent) > 1 else 0.0
                self.processed += 1
                self.tag_frames += bool(tags)
                self.seen_ids.update(tag["id"] for tag in tags)
                if self.first_time is None:
                    self.first_time = now
                self.last_time = now
                packet = {
                    "frame": frame, "frame_number": previous, "tags": tags,
                    "capture_fps": capture_fps, "detect_fps": detect_fps,
                    "detection_ms": (detection_finished - started) * 1000,
                    "detection_search": search, "detection_region": region,
                    "processing_ms": (now - started) * 1000,
                    "frame_timestamp_monotonic": frame_timestamp,
                    "frame_timestamp_basis": getattr(self.reader, "timestamp_basis", "unspecified"),
                    "detector_settings": self.detector_settings,
                }
                if self.pose_smoother is not None:
                    packet["smoothing_settings"] = self.pose_smoother.settings
                if self.cart_heading_estimator is not None:
                    packet["cart_heading_settings"] = self.cart_heading_estimator.settings
                    packet["fused_cart_heading"] = fused_heading
                    packet["raw_fused_cart_heading"] = raw_fused_heading
                if compensation is not None:
                    packet['pelvis_compensation'] = compensation
                if self.pose_estimator is not None:
                    packet["camera_matrix"] = self.pose_estimator.camera_matrix.tolist()
                    packet["distortion_coefficients"] = self.pose_estimator.distortion.tolist()
                if self.cart_pose_estimator is not None:
                    if __package__:
                        from .cart_pose import urdf_cart_observation
                    else:
                        from cart_pose import urdf_cart_observation
                    if stages is not None:
                        cart_pose_started = time.perf_counter()
                    packet['cart_pose_urdf'] = urdf_cart_observation(packet, self.cart_pose_estimator)
                    if stages is not None:
                        stages['cart_pose'] = (time.perf_counter() - cart_pose_started) * 1000
                    packet['processing_ms'] = (time.monotonic()-started)*1000.
                if stages is not None:
                    stages['process'] = packet['processing_ms']
                    packet['stage_timings_ms'] = stages
                if self.missing_tag_frames is not None:
                    self.missing_tag_frames.submit(packet)
                if self.detection_sink is not None:
                    self.detection_sink(packet)
                with self.condition:
                    if stages is not None:
                        self.profile_samples.append(stages)
                    self.result = packet
                    self.condition.notify_all()
        except Exception as exc:
            with self.condition:
                self.error = str(exc)
        finally:
            with self.condition:
                self.done = True
                self.condition.notify_all()

    def latest(self, previous):
        with self.condition:
            ready = self.condition.wait_for(
                lambda: (self.result is not None and self.result["frame_number"] > previous) or self.done,
                timeout=15,
            )
            if not ready:
                raise RuntimeError("No processed camera frame arrived within 15 seconds; check capture/detection")
            if self.error:
                raise RuntimeError(self.error)
            if self.result is None or self.result["frame_number"] == previous:
                return None
            return self.result

    def profile_report(self):
        with self.condition:
            samples = list(self.profile_samples) if self.profile_samples is not None else []
        return format_profile('detect', samples)

    def close(self):
        self.stop.set()
        try:
            self.reader.close()
        finally:
            self.thread.join(timeout=15)
        if self.thread.is_alive():
            raise RuntimeError("AprilTag detection thread did not stop")


def save_result(packet, directory):
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc)
    stem = f"tag36h11_{stamp:%Y%m%dT%H%M%S_%fZ}"
    detected = annotate(packet["frame"], packet["tags"], packet.get("camera_matrix"), packet.get("distortion_coefficients"), telemetry=packet)
    for suffix, image in (("raw", packet["frame"]), ("detected", detected)):
        path = directory / f"{stem}_{suffix}.png"
        if not cv2.imwrite(str(path), image):
            raise RuntimeError(f"Could not save {path}")
    metadata = {key: value for key, value in packet.items() if key != "frame"}
    metadata.update({
        "family": FAMILY, "saved_at_utc": stamp.isoformat(), "image_size": [WIDTH, HEIGHT],
        "annotated_image_size": [detected.shape[1], detected.shape[0]],
        "coordinate_units": {"center_px": "pixels", "corners_px": "pixels", "axes_px": "pixels", "tvec_m": "meters", "rvec": "radians",
                             "rpy_rad": "radians", "rpy_deg": "degrees", "yaw_rad": "radians", "yaw_deg": "degrees"},
        "pose_convention": POSE_CONVENTION,
        "timing_convention": TIMING_CONVENTION,
    })
    path = directory / f"{stem}.json"
    path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Saved raw/annotated images and detections: {path}", flush=True)


def _config_defaults(parser, path):
    """Read CLI-named JSON settings, validating types before argparse defaults."""
    try:
        values = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        parser.error(f"Cannot load config {path}: {exc}")
    if not isinstance(values, dict):
        parser.error(f"Config {path} must contain a JSON object")
    actions = {action.dest.replace('_', '-'): action for action in parser._actions
               if action.dest not in ('help', 'config')}
    unknown = sorted(set(values) - set(actions))
    if unknown:
        parser.error(f"Unknown config setting(s) in {path}: {', '.join(unknown)}")
    defaults = {}
    nullable = {'axis_length', 'cart_tag_id', 'cart_tag_ids', 'cart_to_tag_rpy_deg', 'missing_tag_dir'}
    booleans = {'no_display', 'profile', 'no_smoothing', 'missing_tag_control_stdin', 'no_tag_window'}
    for key, value in values.items():
        action = actions[key]
        dest = action.dest
        if value is None and dest in nullable:
            defaults[dest] = None
            continue
        if dest in booleans:
            if type(value) is not bool:
                parser.error(f"Config {key} must be true or false")
        elif dest == 'cart_tag_ids':
            if not isinstance(value, list) or not value or any(type(item) is not int or item < 0 for item in value):
                parser.error('Config cart-tag-ids must be a nonempty array of nonnegative integers')
        elif action.nargs == 3:
            if (not isinstance(value, list) or len(value) != 3
                    or any(type(item) not in (int, float) for item in value)):
                parser.error(f"Config {key} must be an array of three numbers, e.g. [0, 0, 0]")
            value = [float(item) for item in value]
        elif action.type is float:
            if type(value) not in (int, float):
                parser.error(f"Config {key} must be a number")
            value = float(value)
        elif action.type is int:
            if type(value) is not int:
                parser.error(f"Config {key} must be an integer")
        else:
            if not isinstance(value, str) or not value:
                parser.error(f"Config {key} must be a nonempty string")
            if action.type is Path:
                value = Path(value).expanduser()
                if not value.is_absolute():
                    value = path.parent / value
                value = value.resolve()
        if action.choices is not None and value not in action.choices:
            parser.error(f"Config {key} must be one of {', '.join(action.choices)}")
        defaults[dest] = value
    return defaults


def parse_args(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=CONFIG_FILE, help="JSON config (default: apriltag_config.json beside this script); CLI options override settings")
    parser.add_argument("--device", default="auto", help="RealSense color V4L2 device, or auto to probe all /dev/video* nodes")
    parser.add_argument('--heading-udp-port', type=int, default=0,
                        help='publish gravity-leveled cart yaw to loopback UDP; 0 disables publishing')
    parser.add_argument('--publish-cart-pose', action='store_true',
                        help='also estimate real cart pose using raw ID 0/1 corners and URDF extrinsics')
    parser.add_argument("--no-display", action="store_true", help="detect and report to the terminal without a window")
    parser.add_argument("--display", dest="no_display", action="store_false", help="enable the window even if disabled in config")
    parser.add_argument("--profile", action="store_true", help="report rolling mean/p95 milliseconds for detection, pose, smoothing, yaw, drawing and GUI calls")
    parser.add_argument("--no-profile", dest="profile", action="store_false", help="disable profiling even if enabled in config")
    parser.add_argument("--frames", type=int, default=0, help=f"stop after N captured frames plus {WARMUP} warm-up frames; 0 runs until quit")
    parser.add_argument("--snapshot-dir", type=Path, default=SNAPSHOT_DIR)
    parser.add_argument('--missing-tag-dir', type=Path,
                        help='save raw PNG and frame metadata when no selected cart tag is detected')
    parser.add_argument('--data-control-port', type=int, default=0,
                        help='record push-cart tag results (CSV only) controlled by this loopback UDP port; '
                             '0 disables (see Viser/core/data_episode.py)')
    parser.add_argument('--missing-tag-control-stdin', action='store_true',
                        help='save only frames captured while navigation enables recording through stdin')
    parser.add_argument("--intrinsics", type=Path, default=INTRINSICS_FILE, help="factory 640x480 color intrinsics JSON")
    parser.add_argument("--tag-size", type=float, default=0.10, help="black square edge in meters, excluding white margin (default: 0.10)")
    parser.add_argument("--axis-length", type=float, help="drawn axis length in meters (default: the tag edge, 0.10)")
    tag_selection = parser.add_mutually_exclusive_group()
    tag_selection.add_argument('--cart-tag-id', type=int, help='only use this tag for cart yaw')
    tag_selection.add_argument('--cart-tag-ids', type=int, nargs='+', help='only fuse these cart tags, e.g. 0 1')
    parser.add_argument("--tag-mount", choices=("rear", "top"), default="top",
                        help="top (default): tag flat on cart top, printed top points forward; rear: upright on cart rear, facing robot")
    parser.add_argument("--cart-to-tag-rpy-deg", type=float, nargs=3, metavar=("ROLL", "PITCH", "YAW"),
                        help="custom R_cart_to_tag = Rz(yaw)@Ry(pitch)@Rx(roll), in degrees; overrides --tag-mount")
    parser.add_argument("--waist-yaw-deg", type=float, default=0.,
                        help="waist yaw used ONLY in legacy fixed mode; live modes read motor 12")
    parser.add_argument('--pelvis-attitude-source', choices=('imu', 'kinematics', 'fused', 'fixed'), default='imu',
                        help='pelvis tilt source; fixed is legacy uncompensated/offline mode')
    parser.add_argument('--robot-urdf', type=Path, default=DEFAULT_URDF,
                        help='G1 URDF with fixed waist roll/pitch, pelvis IMU and foot mid links')
    parser.add_argument('--network-interface', default='eth0', help='DDS interface for read-only rt/lowstate')
    parser.add_argument('--robot-state-timeout-ms', type=float, default=100., help='maximum feedback age')
    parser.add_argument('--state-sync-tolerance-ms', type=float, default=30., help='maximum image/state time mismatch')
    parser.add_argument('--camera-latency-ms', type=float, default=0.,
                        help='subtract this measured delay from camera host receipt time; 0 means uncalibrated')
    parser.add_argument('--kinematics-support-foot', choices=('auto', 'left', 'right', 'both'), default='auto',
                        help='flat horizontal support-foot assumption; auto infers support from foot geometry')
    parser.add_argument('--support-height-tolerance-m', type=float, default=.015)
    parser.add_argument('--support-normal-tolerance-deg', type=float, default=8.)
    parser.add_argument('--fusion-imu-weight', type=float, default=.9,
                        help='IMU up-vector weight when kinematics agrees (strictly between 0 and 1)')
    parser.add_argument('--fusion-max-disagreement-deg', type=float, default=15.,
                        help='reject kinematic tilt beyond this angular difference from IMU')
    parser.add_argument("--quad-decimate", type=float, default=DEFAULT_QUAD_DECIMATE,
                        help="internal quad-search downsampling: 1=full detail, 1.5=balanced, 2=faster (default); camera stays 640x480")
    parser.add_argument('--no-tag-window', action='store_true',
                        help='search every frame in full instead of a window around the previous cart tags '
                             '(the window falls back to a full search whenever a tracked tag is missing)')
    parser.add_argument('--opencv-threads', type=int, default=1,
                        help='OpenCV worker threads for tag detection (default: 1); results are identical, only speed changes')
    parser.add_argument("--no-smoothing", action="store_true", help="draw and report raw per-frame pose instead of filtered pose")
    parser.add_argument("--smoothing", dest="no_smoothing", action="store_false", help="enable smoothing even if disabled in config")
    parser.add_argument("--smooth-min-cutoff-hz", type=float, default=2.0,
                        help="static-pose smoothing cutoff in Hz (default: 2.0; lower is smoother/slower)")
    parser.add_argument("--smooth-beta", type=float, default=0.35,
                        help="Z/rotation motion response (default: 0.35; higher follows motion faster)")
    parser.add_argument("--smooth-xy-beta", type=float, default=DEFAULT_SMOOTH_XY_BETA,
                        help="camera X/Y translation response (default: 40; higher follows lateral motion faster)")
    parser.add_argument("--smooth-xy-speed-cutoff-hz", type=float, default=DEFAULT_SMOOTH_XY_SPEED_CUTOFF_HZ,
                        help="X/Y motion-estimate cutoff in Hz (default: 3)")
    parser.add_argument("--smooth-xy-noise-mm", type=float, default=DEFAULT_SMOOTH_XY_NOISE_ALLOWANCE_M * 1000,
                        help="noise allowance for X/Y motion response (default: 2 mm; small positions still update)")
    parser.add_argument("--smooth-reset-gap", type=float, default=0.5,
                        help="reset a tag's filter after this many seconds without a detection (default: 0.5)")
    if '-h' in argv or '--help' in argv:
        parser.parse_args(argv)  # Help remains available if the config is missing/broken.
    locator = argparse.ArgumentParser(add_help=False)
    locator.add_argument('--config', type=Path, default=CONFIG_FILE)
    config_arg, _ = locator.parse_known_args(argv)
    config_path = config_arg.config.expanduser().resolve()
    parser.set_defaults(**_config_defaults(parser, config_path))
    args = parser.parse_args(argv)
    if any(arg.split('=')[0] == '--cart-tag-id' for arg in argv):
        args.cart_tag_ids = None
    elif any(arg.split('=')[0] == '--cart-tag-ids' for arg in argv):
        args.cart_tag_id = None
    if args.cart_tag_id is not None and args.cart_tag_ids is not None:
        parser.error('Choose either cart-tag-id or cart-tag-ids in the config')
    if args.cart_tag_ids is not None and any(tag < 0 for tag in args.cart_tag_ids):
        parser.error('--cart-tag-ids must contain nonnegative integers')
    args.config = config_path
    for name in ('snapshot_dir', 'intrinsics', 'robot_urdf'):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    if args.missing_tag_dir is not None:
        args.missing_tag_dir = args.missing_tag_dir.expanduser().resolve()
    if args.missing_tag_control_stdin and args.missing_tag_dir is None:
        parser.error('--missing-tag-control-stdin requires --missing-tag-dir')
    for action in parser._actions:
        value = getattr(args, action.dest, None)
        if action.type is float and value is not None and not np.isfinite(value).all():
            parser.error(f"--{action.dest.replace('_', '-')} must contain finite numbers")
    if args.frames < 0:
        parser.error("--frames must be nonnegative")
    if not 0 <= args.heading_udp_port <= 65535:
        parser.error('--heading-udp-port must be 0..65535')
    if args.cart_tag_id is not None and args.cart_tag_id < 0:
        parser.error("--cart-tag-id must be nonnegative")
    if args.quad_decimate < 1:
        parser.error("--quad-decimate must be >= 1")
    if args.opencv_threads < 1:
        parser.error('--opencv-threads must be >= 1')
    for name in ('tag_size', 'axis_length', 'smooth_min_cutoff_hz', 'smooth_xy_speed_cutoff_hz', 'smooth_reset_gap',
                 'robot_state_timeout_ms', 'state_sync_tolerance_ms', 'support_height_tolerance_m',
                 'support_normal_tolerance_deg', 'fusion_max_disagreement_deg'):
        value = getattr(args, name)
        if value is not None and value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    for name in ('smooth_beta', 'smooth_xy_beta', 'smooth_xy_noise_mm', 'camera_latency_ms'):
        if getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} must be nonnegative")
    if not 0 < args.fusion_imu_weight < 1:
        parser.error('--fusion-imu-weight must be strictly between 0 and 1')
    for name in ('support_normal_tolerance_deg', 'fusion_max_disagreement_deg'):
        if getattr(args, name) >= 90:
            parser.error(f"--{name.replace('_', '-')} must be less than 90 degrees")
    return args


def main():
    with ExitStack() as resources:
        _run(parse_args(), resources)


def _run(args, resources):
    publisher = None
    if args.heading_udp_port:
        try:
            from .heading_udp import HeadingPublisher
        except ImportError:
            from heading_udp import HeadingPublisher
        publisher = HeadingPublisher(args.heading_udp_port)
        resources.callback(publisher.close)
    cart_heading_estimator = CartHeadingEstimator(
        waist_yaw_deg=args.waist_yaw_deg, tag_mount=args.tag_mount,
        cart_tag_id=args.cart_tag_id, cart_to_tag_rpy_deg=args.cart_to_tag_rpy_deg,
        cart_tag_ids=args.cart_tag_ids,
    )
    if not args.no_display and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        raise RuntimeError("No desktop display. Run in an Ubuntu desktop/X11 terminal, or add --no-display --frames 600.")
    cv2.setNumThreads(args.opencv_threads)
    detector = make_detector(args.quad_decimate)
    camera_matrix, distortion = load_intrinsics(args.intrinsics)
    pose_estimator = TagPoseEstimator(camera_matrix, distortion, args.tag_size, args.axis_length)
    cart_pose_estimator = None
    if args.publish_cart_pose:
        if args.pelvis_attitude_source != 'imu':
            raise ValueError('--publish-cart-pose requires --pelvis-attitude-source imu')
        if __package__:
            from .cart_pose import CartPoseEstimator
        else:
            from cart_pose import CartPoseEstimator
        cart_pose_estimator = CartPoseEstimator(camera_matrix, distortion)
    pose_smoother = None if args.no_smoothing else AdaptivePoseSmoother(
        pose_estimator, min_cutoff_hz=args.smooth_min_cutoff_hz,
        beta=args.smooth_beta, xy_beta=args.smooth_xy_beta, reset_gap_s=args.smooth_reset_gap,
        xy_speed_cutoff_hz=args.smooth_xy_speed_cutoff_hz,
        xy_noise_allowance_m=args.smooth_xy_noise_mm / 1000.,
    )
    compensator = heading_smoother = None
    if args.pelvis_attitude_source != 'fixed':
        model = G1Kinematics(args.robot_urdf)
        feedback = RobotFeedback(args.network_interface)
        resources.callback(feedback.close)
        compensator = PelvisCompensator(
            feedback, model, source=args.pelvis_attitude_source,
            support=args.kinematics_support_foot, imu_weight=args.fusion_imu_weight,
            disagreement_deg=args.fusion_max_disagreement_deg,
            height_tolerance=args.support_height_tolerance_m,
            normal_tolerance=args.support_normal_tolerance_deg,
            timeout_s=args.robot_state_timeout_ms / 1000.,
            sync_tolerance_s=args.state_sync_tolerance_ms / 1000.,
            camera_latency_s=args.camera_latency_ms / 1000.,
        )
        if not args.no_smoothing:
            heading_smoother = HeadingSmoother(args.smooth_min_cutoff_hz, args.smooth_beta, args.smooth_reset_gap)
        cart_heading_estimator.settings.update(
            reference_frame='pelvis_level',
            projection='cart +X and pelvis +X projected into the gravity-horizontal plane',
            waist_source='rt/lowstate motor_state[12].q, matched to image timestamp',
            waist_yaw_deg=None, R_camera_to_pelvis=None,
            extrinsics_source=str(model.path) + ' + assumed standard optical axes; not calibrated',
            pelvis_attitude_source=args.pelvis_attitude_source,
            kinematics_support_foot=args.kinematics_support_foot,
            support_height_tolerance_m=args.support_height_tolerance_m,
            support_normal_tolerance_deg=args.support_normal_tolerance_deg,
            fusion_imu_weight=args.fusion_imu_weight,
            fusion_max_disagreement_deg=args.fusion_max_disagreement_deg,
            robot_state_timeout_ms=args.robot_state_timeout_ms,
            state_sync_tolerance_ms=args.state_sync_tolerance_ms,
            camera_latency_ms=args.camera_latency_ms,
            heading_smoothing='after compensation, before tag fusion' if heading_smoother else 'disabled',
        )
        cart_heading_estimator.settings.pop('R_camera_to_torso')
        cart_heading_estimator.settings['R_camera_to_pelvis_zero'] = (
            model.transform('d435_link', np.zeros(15))[0] @ OPTICAL_TO_LINK
        ).tolist()
    print(f"Config: {args.config}", flush=True)
    print(f"Python: {sys.executable} | OpenCV {cv2.__version__} | family: {FAMILY}", flush=True)
    print(f"Detection downsampling: {args.quad_decimate:g}x | native image/coordinate size: {WIDTH}x{HEIGHT} | "
          f"OpenCV threads: {cv2.getNumThreads()} | cart tag window: "
          f"{'off' if args.no_tag_window or args.cart_tag_ids is None else 'on (full-frame fallback)'}", flush=True)
    if pose_smoother is None:
        print("Pose smoothing: disabled (raw pose)", flush=True)
    else:
        print(f"Pose smoothing: adaptive low-pass | cutoff {pose_smoother.min_cutoff_hz:g} Hz | "
              f"camera X/Y beta {pose_smoother.xy_beta:g} | Z/rotation beta {pose_smoother.beta:g} | "
              f"X/Y noise allowance {pose_smoother.xy_noise_allowance_m * 1000:g} mm", flush=True)
    print(f"Tag edge: {pose_estimator.tag_size * 100:g} cm | axis length: {pose_estimator.axis_length * 100:g} cm | Tag axes: X red/right, Y green/up, Z blue/out (printed tag orientation)", flush=True)
    print("Camera RPY: tag relative to camera optical frame, R=Rz(yaw)@Ry(pitch)@Rx(roll), degrees", flush=True)
    target = "fuse all valid visible tags" if args.cart_tag_id is None else f"ID {args.cart_tag_id}"
    if args.cart_tag_ids is not None:
        target = 'IDs ' + ', '.join(map(str, args.cart_tag_ids))
    if compensator:
        print(f"Cart yaw: gravity-leveled pelvis | attitude: {args.pelvis_attitude_source} | "
              f"live waist yaw (motor 12) | DDS: {args.network_interface} | target: {target}", flush=True)
        print(f"URDF: {args.robot_urdf} | nominal optical extrinsics | waist roll/pitch fixed | "
              f"tag mount: {cart_heading_estimator.tag_mount}", flush=True)
        print(f"State/image alignment: host receipt timestamps; camera delay {args.camera_latency_ms:g} ms "
              "(not hardware synchronized); stale/mismatched feedback makes yaw unavailable", flush=True)
        if args.pelvis_attitude_source in ('kinematics', 'fused'):
            print(f"Kinematics: FLAT HORIZONTAL support-foot assumption, support={args.kinematics_support_foot}; "
                  "no measured foot contact. Fused mode uses IMU alone if kinematics is ambiguous/disagrees.", flush=True)
    else:
        print(f"Cart yaw: pelvis XY, NO gravity leveling | tag mount: {cart_heading_estimator.tag_mount} | "
              f"waist yaw: {args.waist_yaw_deg:+.2f} deg (FIXED) | target: {target}", flush=True)
    print("Cart yaw: 0 = pelvis forward; positive = left; range [-180, 180] deg", flush=True)
    print("Cart yaw fusion: equal-weight circular mean; one valid tag passes through; no valid tags = unavailable", flush=True)
    print("Timing: Detect+Pose = raw detection and pose/RPY; Process = includes smoothing and cart yaw; "
          "both exclude capture waiting, drawing and display/network transport", flush=True)
    if args.profile:
        print("Profile: last 120 frames, mean/p95 wall ms; tag=quad search/refinement/decoding; "
              "attitude=state alignment/URDF/tilt fusion; "
              "pose=PnP/RPY/projection; read_wait=wait for next captured frame. "
              "GUI timings measure local calls, not arrival at the Windows screen. "
              "Detect and view run concurrently; do not add their totals.", flush=True)
    missing_frames = None
    if args.missing_tag_dir is not None:
        if __package__:
            from .missing_tag_frames import MissingTagFrames
        else:
            from missing_tag_frames import MissingTagFrames
        selected_ids = args.cart_tag_ids if args.cart_tag_ids is not None else (
            [args.cart_tag_id] if args.cart_tag_id is not None else [0, 1])
        missing_frames = MissingTagFrames(args.missing_tag_dir, selected_ids,
                                         control_fd=sys.stdin.fileno() if args.missing_tag_control_stdin else None)
        resources.callback(missing_frames.close)
    detections = None
    if args.data_control_port:
        if __package__:
            from .episode_detections import EpisodeDetections
        else:
            from episode_detections import EpisodeDetections
        detections = EpisodeDetections(args.data_control_port)

        def terminate(*_):
            # The owner stops this process with SIGTERM. Unwind normally once
            # so an open episode flushes its detection CSV; ignore repeats.
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            raise KeyboardInterrupt
        signal.signal(signal.SIGTERM, terminate)
        # The ExitStack runs this after _run has stopped the detector, so the
        # last results are written before the detection CSV is closed.
        resources.callback(detections.close)
    reader = CameraReader(open_camera(args.device), args.frames)
    try:
        worker = DetectionWorker(reader, detector, pose_estimator, pose_smoother, cart_heading_estimator,
                                 profile=args.profile, pelvis_compensator=compensator, heading_smoother=heading_smoother,
                                 cart_pose_estimator=cart_pose_estimator, missing_tag_frames=missing_frames,
                                 tag_window=not args.no_tag_window,
                                 detection_sink=detections.detection if detections is not None else None)
    except Exception:
        reader.close()
        raise
    previous = 0
    preview_times = deque(maxlen=FPS + 1)
    preview_profiles = deque(maxlen=120)
    last_report = 0.0
    last_tag_presence = None
    last_compensation_status = None
    try:
        if not args.no_display:
            cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
            print("Q/Esc: quit | S: save raw image, detected image, and JSON", flush=True)
        while True:
            packet = worker.latest(previous)
            if packet is None:
                break
            previous = packet["frame_number"]
            if publisher is not None:
                publisher.publish(packet)
            now = time.monotonic()
            tags = packet["tags"]
            has_tags = bool(tags)
            compensation = packet.get('pelvis_compensation')
            status = (compensation['valid'], compensation['reason']) if compensation else None
            if now - last_report >= 1 or has_tags != last_tag_presence or status != last_compensation_status:
                locations = []
                for tag in tags:
                    location = f"ID={tag['id']} center=({tag['center_px'][0]:.1f},{tag['center_px'][1]:.1f})"
                    if tag.get("pose"):
                        x, y, z = np.asarray(tag["pose"]["tvec_m"]) * 100
                        location += f" camera_xyz_cm=({x:.1f},{y:.1f},{z:.1f})"
                        roll, pitch, yaw = tag["pose"]["rpy_deg"]
                        location += f" camera_rpy_deg=({roll:+.2f},{pitch:+.2f},{yaw:+.2f})"
                        if tag["pose"]["rpy_singular"]:
                            location += " rpy_singular=true"
                    heading = tag.get("cart_heading")
                    if heading is not None:
                        if heading["valid"]:
                            location += f" cart_yaw_pelvis_deg={heading['yaw_deg']:+.2f} cart_yaw_pelvis_rad={heading['yaw_rad']:+.5f}"
                        else:
                            location += f" cart_yaw_pelvis=unavailable({heading['reason']})"
                    locations.append(location)
                locations = "; ".join(locations) or NO_TAG_MESSAGE
                if args.cart_tag_id is not None and not any(tag["id"] == args.cart_tag_id for tag in tags):
                    locations += f" | Cart tag ID {args.cart_tag_id} not detected; cart yaw unavailable"
                if args.cart_tag_ids is not None and not any(tag['id'] in args.cart_tag_ids for tag in tags):
                    locations += f" | Cart tags {args.cart_tag_ids} not detected; cart yaw unavailable"
                counts = worker.search_counts
                print(f"Capture {packet['capture_fps']:.1f} FPS | Detect {packet['detect_fps']:.1f} FPS | "
                      f"Detect+Pose {packet['detection_ms']:.2f} ms | Process {packet['processing_ms']:.2f} ms | "
                      f"Search window {counts['window']} / fallback {counts['window_fallback']} / full {counts['full']} | "
                      f"{format_fused_heading(packet['fused_cart_heading'])} | "
                      f"{format_compensation(compensation)} | {locations}", flush=True)
                if args.profile:
                    for report in (worker.profile_report(), format_profile('view', preview_profiles),
                                   reader_profile(reader)):
                        if report:
                            print(report, flush=True)
                last_report = now
            last_tag_presence = has_tags
            last_compensation_status = status
            if args.no_display:
                continue
            preview_times.append(now)
            preview_fps = (len(preview_times) - 1) / (preview_times[-1] - preview_times[0]) if len(preview_times) > 1 else 0.0
            if args.profile:
                draw_started = time.perf_counter()
            preview = annotate(packet["frame"], tags, camera_matrix, distortion, telemetry=packet, preview_fps=preview_fps)
            if args.profile:
                draw_done = time.perf_counter()
            cv2.imshow(WINDOW, preview)
            if args.profile:
                imshow_done = time.perf_counter()
            key = cv2.waitKey(1) & 0xFF
            quit_requested = key in (ord("q"), ord("Q"), 27) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1
            if args.profile:
                preview_profiles.append({
                    'draw': (draw_done - draw_started) * 1000,
                    'imshow': (imshow_done - draw_done) * 1000,
                    'events': (time.perf_counter() - imshow_done) * 1000,
                })
            if quit_requested:
                break
            if key in (ord("s"), ord("S")):
                save_result(packet, args.snapshot_dir)
    finally:
        try:
            worker.close()
        finally:
            if not args.no_display:
                cv2.destroyAllWindows()
        capture_fps = (reader.measured_count - 1) / (reader.last_sample - reader.first_sample) if reader.measured_count > 1 else 0.0
        detect_fps = (worker.processed - 1) / (worker.last_time - worker.first_time) if worker.processed > 1 else 0.0
        if args.profile:
            for report in (worker.profile_report(), format_profile('view', preview_profiles)):
                if report:
                    print(report, flush=True)
        print(f"Finished: capture {capture_fps:.2f} FPS; detect {detect_fps:.2f} FPS; processed {worker.processed} frames; tag frames {worker.tag_frames}; IDs {sorted(worker.seen_ids)}; "
              f"search {worker.search_counts}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
