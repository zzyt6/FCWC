"""Pelvis tilt from IMU or flat-support-foot kinematics, and guarded fusion.

Only gravity direction is observable/needed here, not absolute world yaw. FK
requires a flat horizontal support foot; automatic support is a geometric
hypothesis, not a contact sensor. URDF chains are parsed only at startup.
"""
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

if __package__:
    from .cart_heading import rotation_from_rpy_degrees
    from .robot_feedback import unit_quaternion
else:
    from cart_heading import rotation_from_rpy_degrees
    from robot_feedback import unit_quaternion

DEFAULT_URDF = Path(__file__).resolve().parents[1] / 'G1_URDF/g1.urdf'
JOINT_NAMES = [f'{side}_{part}_joint' for side in ('left', 'right')
               for part in ('hip_pitch', 'hip_roll', 'hip_yaw', 'knee', 'ankle_pitch', 'ankle_roll')]
JOINT_INDEX = {name: i for i, name in enumerate(JOINT_NAMES + ['waist_yaw_joint'])}
OPTICAL_TO_LINK = np.array([[0., 0., 1.], [-1., 0., 0.], [0., -1., 0.]])


def normalized(v):
    v = np.asarray(v, dtype=float)
    norm = np.linalg.norm(v)
    if v.shape != (3,) or not np.isfinite(v).all() or norm < 1e-8:
        raise ValueError('invalid_direction')
    return v / norm


def angle_degrees(a, b):
    return float(np.rad2deg(np.arccos(np.clip(a @ b, -1, 1))))


def imu_up_in_sensor(quaternion):
    q = unit_quaternion(quaternion)
    if q is None:
        raise ValueError('invalid_pelvis_imu')
    w, x, y, z = q
    # Last row of R_world_from_sensor, i.e. R_world_from_sensor.T @ world_Z.
    return np.array([2 * (x*z - w*y), 2 * (y*z + w*x), 1 - 2 * (x*x + y*y)])


class G1Kinematics:
    def __init__(self, path=DEFAULT_URDF):
        self.path = Path(path).resolve()
        tree = ET.parse(self.path)
        self.joints = {}
        for joint in tree.findall('joint'):
            origin = joint.find('origin')
            xyz = np.fromstring(origin.get('xyz', '0 0 0') if origin is not None else '0 0 0', sep=' ')
            rpy = np.fromstring(origin.get('rpy', '0 0 0') if origin is not None else '0 0 0', sep=' ')
            axis_element = joint.find('axis')
            axis = normalized(np.fromstring(axis_element.get('xyz', '1 0 0')
                              if axis_element is not None else '1 0 0', sep=' '))
            if xyz.shape != (3,) or rpy.shape != (3,) or not np.isfinite(xyz).all():
                raise ValueError('Invalid URDF joint origin')
            rotation = rotation_from_rpy_degrees(np.rad2deg(rpy))
            a, b, c = axis
            skew = np.array([[0, -c, b], [c, 0, -a], [-b, a, 0]])
            self.joints[joint.find('child').get('link')] = dict(
                name=joint.get('name'), type=joint.get('type'),
                parent=joint.find('parent').get('link'), rotation=rotation,
                xyz=xyz, skew=skew, skew2=skew @ skew)
        self.chains = {link: self._chain(link) for link in
                       ('d435_link', 'imu_in_pelvis', 'left_foot_mid_link', 'right_foot_mid_link')}
        self.imu_to_pelvis = self.transform('imu_in_pelvis', np.zeros(15))[0]

    def _chain(self, link):
        chain, seen = [], set()
        while link != 'pelvis':
            if link in seen or link not in self.joints:
                raise ValueError(f'URDF has no unambiguous pelvis -> {link} chain')
            seen.add(link)
            joint = self.joints[link]
            if joint['name'] in ('waist_roll_joint', 'waist_pitch_joint') and joint['type'] != 'fixed':
                raise ValueError('Use the URDF with locked (fixed) waist roll/pitch')
            if joint['type'] != 'fixed' and (joint['type'] not in ('revolute', 'continuous')
                                            or joint['name'] not in JOINT_INDEX):
                raise ValueError(f"Unsupported movable URDF joint: {joint['name']}")
            chain.append(joint)
            link = joint['parent']
        return chain[::-1]

    def transform(self, link, q):
        rotation, position = np.eye(3), np.zeros(3)
        for joint in self.chains[link]:
            position += rotation @ joint['xyz']
            rotation = rotation @ joint['rotation']
            if joint['type'] != 'fixed':
                theta = float(q[JOINT_INDEX[joint['name']]])
                if not np.isfinite(theta):
                    raise ValueError('invalid_joint_angle')
                rotation = rotation @ (np.eye(3) + np.sin(theta) * joint['skew']
                                       + (1 - np.cos(theta)) * joint['skew2'])
        return rotation, position

    def support_up(self, q, support, height_tolerance, normal_tolerance):
        left, lp = self.transform('left_foot_mid_link', q)
        right, rp = self.transform('right_foot_mid_link', q)
        ln, rn = left[:, 2], right[:, 2]
        disagreement = angle_degrees(ln, rn)
        # Under each foot's flat-floor hypothesis, height of the other foot.
        rh, lh = float(ln @ (rp-lp)), float(rn @ (lp-rp))
        used = support
        if support == 'auto':
            if abs(rh) <= height_tolerance and abs(lh) <= height_tolerance and disagreement <= normal_tolerance:
                used = 'both'
            elif rh > height_tolerance and lh < -height_tolerance:
                used = 'left'
            elif lh > height_tolerance and rh < -height_tolerance:
                used = 'right'
            else:
                raise ValueError('support_foot_ambiguous')
        if used == 'both':
            if max(abs(rh), abs(lh)) > height_tolerance or disagreement > normal_tolerance:
                raise ValueError('support_feet_disagree')
            up = normalized(ln + rn)
        else:
            up = ln if used == 'left' else rn
        return up, dict(support_foot=used, support_source='flat_floor_assumption',
                        foot_normal_difference_deg=disagreement)


class PelvisCompensator:
    def __init__(self, feedback, model, *, source='imu', support='auto', imu_weight=.9,
                 disagreement_deg=15., height_tolerance=.015, normal_tolerance=8.,
                 timeout_s=.1, sync_tolerance_s=.03, camera_latency_s=0.):
        self.feedback, self.model = feedback, model
        self.source, self.support = source, support
        self.imu_weight, self.disagreement_deg = imu_weight, disagreement_deg
        self.height_tolerance, self.normal_tolerance = height_tolerance, normal_tolerance
        self.timeout_s, self.sync_tolerance_s = timeout_s, sync_tolerance_s
        self.camera_latency_s = camera_latency_s

    def at_frame(self, timestamp, now):
        sample, timing = self.feedback.at(timestamp - self.camera_latency_s, now,
                                          self.timeout_s, self.sync_tolerance_s)
        result = dict(timing, requested_source=self.source, effective_source=None,
                      reference_frame='pelvis_level', waist_yaw_deg=None,
                      up_in_pelvis=None, roll_deg=None, pitch_deg=None)
        if sample is None:
            return result
        result['valid'] = False
        try:
            q = sample['q']
            camera_link, camera_position = self.model.transform('d435_link', q)
            camera = camera_link @ OPTICAL_TO_LINK
            result.update(waist_yaw_deg=float(np.rad2deg(q[12])), R_camera_to_pelvis=camera.tolist(),
                          camera_position_in_pelvis_m=camera_position.tolist())
            imu = kin = None
            if self.source in ('imu', 'fused'):
                imu = normalized(self.model.imu_to_pelvis @ imu_up_in_sensor(sample['quaternion']))
            if self.source in ('kinematics', 'fused'):
                try:
                    if sample['mode_pr'] != 0:
                        raise ValueError('joint_feedback_not_pitch_roll_mode')
                    kin, info = self.model.support_up(q, self.support, self.height_tolerance, self.normal_tolerance)
                    result.update(info)
                except ValueError as exc:
                    result['kinematics_reason'] = str(exc)
                    if self.source == 'kinematics':
                        raise
            reason, effective = 'ok', self.source
            if self.source == 'imu':
                up = imu
            elif self.source == 'kinematics':
                up = kin
            else:
                up = imu
                if kin is None:
                    reason, effective = 'imu_only_kinematics_unavailable', 'imu'
                else:
                    difference = angle_degrees(imu, kin)
                    result['imu_kinematics_difference_deg'] = difference
                    if difference > self.disagreement_deg:
                        reason, effective = 'imu_only_kinematics_disagreement', 'imu'
                    else:
                        up = normalized(self.imu_weight * imu + (1 - self.imu_weight) * kin)
            # Roll/pitch equivalent to Rz(yaw) Ry(pitch) Rx(roll). World yaw
            # intentionally remains unestimated: only the up vector is fused.
            result.update(valid=True, reason=reason, effective_source=effective,
                          up_in_pelvis=up.tolist(),
                          roll_deg=float(np.rad2deg(np.arctan2(up[1], up[2]))),
                          pitch_deg=float(np.rad2deg(np.arctan2(-up[0], np.hypot(up[1], up[2])))))
        except ValueError as exc:
            result.update(valid=False, reason=str(exc))
        return result
