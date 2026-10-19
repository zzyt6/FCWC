"""Shared raw AprilTag cart geometry and frame transforms; no camera/robot I/O."""
import math

import cv2
import numpy as np


TAG_SIZE = .10
BOARD_HEIGHT = .775
TAG_CENTERS = {0: np.array([-.103, .149, .775]),
               1: np.array([-.103, -.141, .775])}
# OpenCV decoded corner order, expressed in the detector's OUTPUT tag frame.
TAG_CORNERS = np.array([[.05, -.05, 0.], [-.05, -.05, 0.],
                        [-.05, .05, 0.], [.05, .05, 0.]])
BOARD_CORNERS = np.array([[-.45, -.25, .775], [.45, -.25, .775],
                          [.45, .25, .775], [-.45, .25, .775]])


def transform(rotation, position):
    result = np.eye(4)
    result[:3, :3], result[:3, 3] = rotation, position
    return result


def rotation_distance(a, b):
    return math.degrees(math.acos(float(np.clip((np.trace(a.T @ b) - 1.) / 2., -1., 1.))))


def level_from_robot(up):
    """Gravity-level axes, at the pelvis origin (NOT at ground height)."""
    up = np.asarray(up, dtype=float)
    up = up / np.linalg.norm(up)
    forward = np.array([1., 0., 0.]) - up[0]*up
    if np.linalg.norm(forward) < 1e-6:
        raise ValueError('Pelvis forward is vertical')
    forward /= np.linalg.norm(forward)
    return transform(np.stack([forward, np.cross(up, forward), up]), np.zeros(3))


class CartPoseEstimator:
    """Joint planar PnP from raw corners; no grasp geometry or temporal filter."""
    def __init__(self, camera_matrix, distortion, tag_ids=(0, 1), max_rmse_px=2.):
        self.camera_matrix = np.asarray(camera_matrix, dtype=float)
        self.distortion = np.asarray(distortion, dtype=float)
        self.tag_ids = frozenset(tag_ids)
        self.max_rmse_px = max_rmse_px

    def _solve(self, tags):
        anchor = np.array([0., 0., BOARD_HEIGHT])
        object_points = np.concatenate([TAG_CORNERS + TAG_CENTERS[t['id']] - anchor for t in tags])
        image_points = np.concatenate([np.asarray(t['corners_px'], dtype=float) for t in tags])
        if image_points.shape != (len(tags)*4, 2) or not np.isfinite(image_points).all():
            raise ValueError('Invalid tag corners')
        _, rvecs, tvecs, _ = cv2.solvePnPGeneric(
            object_points, image_points, self.camera_matrix, self.distortion,
            flags=cv2.SOLVEPNP_IPPE)
        candidates = []
        for rvec, tvec in zip(rvecs, tvecs):
            rvec, tvec = cv2.solvePnPRefineLM(object_points, image_points,
                                            self.camera_matrix, self.distortion,
                                            rvec.copy(), tvec.copy())
            rotation = cv2.Rodrigues(rvec)[0]
            position = tvec.reshape(3)
            if (not np.isfinite(rotation).all() or not np.isfinite(position).all()
                    or np.any((object_points @ rotation.T + position)[:, 2] <= 0)
                    or rotation[:, 2] @ position >= 0):
                continue  # All corners in front of lens; printed faces toward camera.
            projected = cv2.projectPoints(object_points, rvec, tvec, self.camera_matrix, self.distortion)[0]
            errors = np.linalg.norm(projected.reshape(-1, 2)-image_points, axis=1)
            rmse = float(np.sqrt(np.mean(errors**2)))
            per_tag = np.sqrt(np.mean(errors.reshape(-1, 4)**2, axis=1))
            candidates.append((rmse, transform(rotation, position-rotation @ anchor), per_tag))
        if not candidates:
            return dict(valid=False, reason='no_front_facing_pnp_solution')
        candidates.sort(key=lambda candidate: candidate[0])
        rmse, pose, per_tag = candidates[0]
        result = dict(valid=True, reason='ok', T=pose, rmse_px=rmse,
                      per_tag_rmse_px={t['id']: float(e) for t, e in zip(tags, per_tag)},
                      ambiguity_deg=0., ambiguity_gap_px=None)
        if len(candidates) > 1:
            gap = candidates[1][0] - rmse
            angle = rotation_distance(pose[:3, :3], candidates[1][1][:3, :3])
            result.update(ambiguity_deg=angle, ambiguity_gap_px=gap)
            if gap < .1 and angle > 15.:
                result.update(valid=False, reason='planar_pose_ambiguous')
        if max(per_tag) > self.max_rmse_px:
            result.update(valid=False, reason='reprojection_error_too_large')
        return result

    def estimate(self, tags, *, diagnostics=True):
        selected = [tag for tag in tags if tag['id'] in self.tag_ids]
        ids = [tag['id'] for tag in selected]
        if not ids:
            return dict(valid=False, reason='no_cart_tag', tag_ids=[], singles={})
        if len(ids) != len(set(ids)):
            return dict(valid=False, reason='duplicate_tag_id', tag_ids=ids, singles={})
        try:
            result = self._solve(selected)
            singles = {tag['id']: self._solve([tag]) for tag in selected} if diagnostics else {}
        except (cv2.error, ValueError, KeyError) as exc:
            return dict(valid=False, reason='invalid_corners: '+str(exc), tag_ids=ids, singles={})
        result.update(tag_ids=ids, singles=singles)
        if len(singles) == 2 and all(item['valid'] for item in singles.values()):
            a, b = singles[0]['T'], singles[1]['T']
            # Use independent single-tag results: a joint fit forces the known baseline.
            centers = [singles[i]['T'][:3, :3] @ TAG_CENTERS[i] + singles[i]['T'][:3, 3] for i in (0, 1)]
            result.update(baseline_m=float(np.linalg.norm(centers[0]-centers[1])),
                          single_position_difference_m=float(np.linalg.norm(a[:3, 3]-b[:3, 3])),
                          single_rotation_difference_deg=rotation_distance(a[:3, :3], b[:3, :3]))
        return result


def frame_poses(estimate, compensation=None, map_from_robot=None, robot_from_camera=None):
    """Compose column-vector transforms; do not mix physical and leveled pelvis."""
    poses = {}
    if not estimate['valid']:
        return poses
    poses['camera'] = estimate['T']
    if compensation is not None and compensation.get('valid'):
        if robot_from_camera is None:
            robot_from_camera = transform(np.asarray(compensation['R_camera_to_pelvis']),
                                          np.asarray(compensation['camera_position_in_pelvis_m']))
        poses['robot'] = robot_from_camera @ poses['camera']
        poses['level'] = level_from_robot(compensation['up_in_pelvis']) @ poses['robot']
        if map_from_robot is not None and map_from_robot.get('valid'):
            poses['map'] = map_from_robot['T'] @ poses['robot']
    return poses


def urdf_cart_observation(packet, estimator):
    """Same raw PnP and URDF composition as cart_pose_measure --extrinsics urdf."""
    compensation = packet.get('pelvis_compensation') or {}
    latency_ms = (packet.get('cart_heading_settings') or {}).get('camera_latency_ms', 0.)
    result = dict(schema='cart_pose_urdf_v1', valid=False, extrinsics='urdf',
                  reference_frame='pelvis_level',
                  captured_monotonic=packet['frame_timestamp_monotonic']-latency_ms/1000.,
                  frame_number=packet['frame_number'], tag_ids=[], reason='robot_feedback_unavailable')
    estimate = estimator.estimate(packet['tags'], diagnostics=False)
    result.update(tag_ids=estimate['tag_ids'], rmse_px=estimate.get('rmse_px'))
    if not estimate['valid']:
        result['reason'] = estimate['reason']
        return result
    if not compensation.get('valid'):
        result['reason'] = compensation.get('reason', result['reason'])
        return result
    poses = frame_poses(estimate, compensation)
    pose = poses['level']
    result.update(valid=True, reason='ok', T_level_cart=pose.tolist(),
                  yaw_rad=math.atan2(pose[1, 0], pose[0, 0]))
    return result
