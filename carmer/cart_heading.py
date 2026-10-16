"""Cart heading using nominal camera mounting and optional live compensation.

No hardware access. Live state arrives as a timestamp-matched compensation dict.
Camera optical axes: right/down/forward. Robot/cart axes: forward/left/up.
"""

import numpy as np


# G1_URDF/g1.urdf d435_joint pitch; assume d435_link uses forward/left/up.
_PITCH = 0.8307767239493009
R_CAMERA_TO_TORSO = np.array([
    [0., -np.sin(_PITCH), np.cos(_PITCH)],
    [-1., 0., 0.],
    [0., -np.cos(_PITCH), -np.sin(_PITCH)],
])

# Columns are cart forward, left, up expressed in the printed tag frame.
# Rear: printed top points up, tag's front face points backward toward robot.
# Top: printed top points along cart forward, tag's front face points upward.
CART_TO_TAG = {
    "rear": np.array([[0., -1., 0.], [0., 0., 1.], [-1., 0., 0.]]),
    "top": np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]),
}


def rotation_from_rpy_degrees(angles):
    """Rz(yaw) @ Ry(pitch) @ Rx(roll), with angles in degrees."""
    angles = np.asarray(angles, dtype=float)
    if angles.shape != (3,) or not np.isfinite(angles).all():
        raise ValueError("Cart-to-tag RPY must contain three finite angles in degrees")
    roll, pitch, yaw = np.deg2rad(angles)
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def fuse_cart_headings(tags, heading_key="cart_heading"):
    """Equal-weight circular mean of valid headings visible in this frame only.

    A single valid observation passes through. Opposing directions have no
    defined circular mean and must not produce a plausible-looking zero yaw.
    Per-tag heading eligibility/mounting is handled by CartHeadingEstimator.
    """
    result = {
        "valid": False, "yaw_rad": None, "yaw_deg": None,
        "reference_frame": "pelvis", "mode": "unavailable",
        "method": "equal_weight_circular_mean", "tag_ids": [], "tag_count": 0,
        "resultant_length": 0.0,
        "reason": "no_tags_detected" if not tags else "no_valid_heading",
    }
    observations = []
    frames = set()
    for tag in sorted(tags, key=lambda item: item["id"]):
        heading = tag.get(heading_key)
        if not heading or not heading.get("valid"):
            continue
        yaw = heading.get("yaw_rad")
        if yaw is None or not np.isfinite(yaw):
            continue
        observations.append((int(tag["id"]), float(yaw)))
        frames.add(heading.get('reference_frame', 'pelvis'))
    if not observations:
        return result
    if len(frames) != 1:
        result['reason'] = 'inconsistent_reference_frames'
        return result
    result['reference_frame'] = frames.pop()
    result.update(tag_ids=[tag_id for tag_id, _ in observations], tag_count=len(observations))
    if len(observations) == 1:
        yaw = observations[0][1]
        length = 1.0
    else:
        angles = np.array([yaw for _, yaw in observations])
        x, y = float(np.mean(np.cos(angles))), float(np.mean(np.sin(angles)))
        length = float(np.hypot(x, y))
        result["resultant_length"] = length
        if length < 1e-6:
            result["reason"] = "opposing_headings"
            return result
        yaw = float(np.arctan2(y, x))
    result.update(valid=True, yaw_rad=yaw, yaw_deg=float(np.rad2deg(yaw)),
                  mode="single" if len(observations) == 1 else "fused",
                  resultant_length=length, reason="ok")
    return result


class CartHeadingEstimator:
    def __init__(self, waist_yaw_deg=0., tag_mount="top", cart_tag_id=None,
                 cart_to_tag_rpy_deg=None, cart_tag_ids=None):
        if not np.isfinite(waist_yaw_deg):
            raise ValueError("--waist-yaw-deg must be finite")
        if tag_mount not in CART_TO_TAG:
            raise ValueError("--tag-mount must be rear or top")
        if cart_tag_id is not None and (not isinstance(cart_tag_id, int) or cart_tag_id < 0):
            raise ValueError("--cart-tag-id must be a nonnegative integer")
        self.cart_tag_id = cart_tag_id
        if cart_tag_ids is not None and (not isinstance(cart_tag_ids, (list, tuple)) or not cart_tag_ids
                or any(type(tag) is not int or tag < 0 for tag in cart_tag_ids)):
            raise ValueError('--cart-tag-ids must contain nonnegative integers')
        if cart_tag_id is not None and cart_tag_ids is not None:
            raise ValueError('Choose either --cart-tag-id or --cart-tag-ids')
        self.cart_tag_ids = frozenset(cart_tag_ids) if cart_tag_ids is not None else None
        self.waist_yaw_deg = float(waist_yaw_deg)
        self.tag_mount = tag_mount if cart_to_tag_rpy_deg is None else "custom"
        self.cart_to_tag = (CART_TO_TAG[tag_mount].copy() if cart_to_tag_rpy_deg is None
                            else rotation_from_rpy_degrees(cart_to_tag_rpy_deg))
        self.camera_to_pelvis = rotation_from_rpy_degrees([0., 0., waist_yaw_deg]) @ R_CAMERA_TO_TORSO
        self.settings = {
            "reference_frame": "pelvis",
            "projection": "cart +X projected into pelvis XY; no gravity leveling",
            "positive_direction": "from pelvis +X toward +Y (left turn)",
            "range_rad": "[-pi, pi]",
            "extrinsics_source": "G1_URDF/g1.urdf d435_joint + assumed standard optical axes; not calibrated",
            "R_camera_to_torso": R_CAMERA_TO_TORSO.tolist(),
            "R_camera_to_pelvis": self.camera_to_pelvis.tolist(),
            "R_cart_to_tag": self.cart_to_tag.tolist(),
            "tag_mount": self.tag_mount,
            "cart_tag_id": cart_tag_id,
            "cart_tag_ids": sorted(self.cart_tag_ids) if self.cart_tag_ids is not None else None,
            "waist_yaw_deg": self.waist_yaw_deg,
            "waist_source": "fixed parameter; no live joint feedback",
            "transform": "R_cart_to_pelvis = R_camera_to_pelvis @ R_tag_to_camera @ R_cart_to_tag",
        }

    def applies_to(self, tag_id):
        if self.cart_tag_ids is not None:
            return tag_id in self.cart_tag_ids
        return self.cart_tag_id is None or tag_id == self.cart_tag_id

    def estimate(self, pose, compensation=None):
        result = {"valid": False, "yaw_rad": None, "yaw_deg": None,
                  "reference_frame": "pelvis", "reason": "pose_unavailable"}
        camera = self.camera_to_pelvis
        if compensation is not None:
            result['reference_frame'] = 'pelvis_level'
            if not compensation['valid']:
                result['reason'] = compensation['reason']
                return result
            camera = np.asarray(compensation['R_camera_to_pelvis'])
        if pose is None:
            return result
        rotation = np.asarray(pose["R_tag_to_camera"], dtype=float)
        if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
            result["reason"] = "invalid_rotation"
            return result
        cart_to_pelvis = camera @ rotation @ self.cart_to_tag
        forward = cart_to_pelvis[:, 0]
        x_axis, y_axis = np.array([1., 0., 0.]), np.array([0., 1., 0.])
        if compensation is not None:
            up = np.asarray(compensation['up_in_pelvis'], dtype=float)
            x_axis = x_axis - up[0] * up
            length = np.linalg.norm(x_axis)
            if length < 1e-6:
                result['reason'] = 'pelvis_forward_axis_vertical'
                return result
            x_axis /= length
            y_axis = np.cross(up, x_axis)
        x, y = float(forward @ x_axis), float(forward @ y_axis)
        horizontal_norm = float(np.hypot(x, y))
        result.update(R_cart_to_pelvis=cart_to_pelvis.tolist(),
                      forward_in_pelvis=forward.tolist(), horizontal_norm=horizontal_norm)
        if horizontal_norm < 1e-6:
            result["reason"] = "forward_axis_vertical"
            return result
        yaw = float(np.arctan2(y, x))
        result.update(valid=True, yaw_rad=yaw, yaw_deg=float(np.rad2deg(yaw)), reason="ok")
        return result


class HeadingSmoother:
    """Circular low-pass applied AFTER live frame compensation, before tag fusion.

    Camera pose filtering is independent; its lag must not be combined with a
    newer robot attitude. Any lost/invalid observation clears that tag's history.
    """
    def __init__(self, min_cutoff_hz=2., beta=.35, reset_gap_s=.5):
        self.cutoff, self.beta, self.gap = min_cutoff_hz, beta, reset_gap_s
        self.states = {}

    def retain(self, tag_ids):
        self.states = {k: v for k, v in self.states.items() if k in tag_ids}

    def update(self, tag_id, heading, timestamp, source):
        if not heading['valid']:
            self.states.pop(tag_id, None)
            return dict(heading)
        raw = heading['yaw_rad']
        previous = self.states.get(tag_id)
        yaw, speed = raw, 0.
        if previous and previous['source'] == source:
            dt = timestamp - previous['time']
            if 0 < dt <= self.gap:
                delta = np.arctan2(np.sin(raw - previous['raw']), np.cos(raw - previous['raw']))
                speed_alpha = 1 - np.exp(-2 * np.pi * dt)
                speed = previous['speed'] + speed_alpha * (abs(delta) / dt - previous['speed'])
                alpha = 1 - np.exp(-2 * np.pi * (self.cutoff + self.beta * speed) * dt)
                error = np.arctan2(np.sin(raw - previous['yaw']), np.cos(raw - previous['yaw']))
                yaw = previous['yaw'] + alpha * error
                yaw = float(np.arctan2(np.sin(yaw), np.cos(yaw)))
        self.states[tag_id] = dict(time=timestamp, raw=raw, yaw=yaw, speed=speed, source=source)
        return dict(heading, yaw_rad=yaw, yaw_deg=float(np.rad2deg(yaw)),
                    smoothing='circular_lowpass_after_compensation')
