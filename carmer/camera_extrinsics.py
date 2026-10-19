"""Explicit camera-optical extrinsics and live pelvis composition."""
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

if __package__:
    from .cart_heading import rotation_from_rpy_degrees
    from .pelvis_attitude import OPTICAL_TO_LINK
else:
    from cart_heading import rotation_from_rpy_degrees
    from pelvis_attitude import OPTICAL_TO_LINK


def rigid_transform(rotation, position):
    result = np.eye(4)
    result[:3, :3], result[:3, 3] = rotation, position
    return validate_transform(result)


def validate_transform(value):
    matrix = np.asarray(value, dtype=float)
    if (matrix.shape != (4, 4) or not np.isfinite(matrix).all()
            or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-8)
            or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(matrix[:3, :3]), 1., atol=1e-6)):
        raise ValueError('Expected a finite rigid 4x4 transform with a proper rotation')
    return matrix.copy()


def urdf_camera_to_torso(path):
    """T_torso_camera_optical, including the same optical-axis mapping as detection."""
    root = ET.parse(path).getroot()
    joints = {joint.find('child').get('link'): joint for joint in root.findall('joint')}
    link, result, visited = 'd435_link', np.eye(4), set()
    while link != 'torso_link':
        if link in visited or link not in joints:
            raise ValueError('No URDF torso_link -> d435_link chain')
        visited.add(link)
        joint = joints[link]
        if joint.get('type') != 'fixed':
            raise ValueError('Camera must be fixed relative to torso_link')
        origin = joint.find('origin')
        xyz = np.fromstring(origin.get('xyz', '0 0 0') if origin is not None else '0 0 0', sep=' ')
        rpy = np.fromstring(origin.get('rpy', '0 0 0') if origin is not None else '0 0 0', sep=' ')
        result = rigid_transform(rotation_from_rpy_degrees(np.rad2deg(rpy)), xyz) @ result
        link = joint.find('parent').get('link')
    return validate_transform(result @ rigid_transform(OPTICAL_TO_LINK, np.zeros(3)))


def load_camera_to_torso(path):
    """Reject pending/nominal results instead of silently using them as calibration."""
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    if (data.get('schema') != 'cwc_camera_extrinsics_v1' or data.get('status') != 'calibrated'
            or data.get('source_frame') != 'camera_optical'
            or data.get('target_frame') != 'torso_link' or data.get('translation_units') != 'm'):
        raise ValueError('File does not contain calibrated camera_optical -> torso_link extrinsics')
    return validate_transform(data['T_torso_camera_optical'])


class CalibratedCameraExtrinsics:
    """Load once, then call in_pelvis(q) with every fresh measured joint vector.

    model is the existing G1Kinematics instance. Its nominal camera chain is
    used only to recover the live pelvis -> torso transform; the nominal fixed
    camera mount cancels algebraically before applying the calibrated mount.
    """
    def __init__(self, model, path):
        self.model = model
        self.T_torso_camera_optical = load_camera_to_torso(path)
        nominal = urdf_camera_to_torso(model.path)
        self.correction = np.linalg.inv(nominal) @ self.T_torso_camera_optical

    def in_pelvis(self, q):
        rotation, position = self.model.transform('d435_link', q)
        nominal_live = rigid_transform(rotation @ OPTICAL_TO_LINK, position)
        return validate_transform(nominal_live @ self.correction)


class ZeroWaistCameraExtrinsics:
    """A supplied optical -> physical pelvis transform at zero waist yaw.

    Transfer the URDF's relative waist motion, without retaining its nominal
    fixed camera mount. Global pelvis tilt is deliberately not applied here.
    """
    def __init__(self, model, path):
        self.path = Path(path).resolve()
        data = json.loads(self.path.read_text(encoding='utf-8'))
        if (data.get('schema') != 'cwc_zero_waist_camera_extrinsics_v1'
                or data.get('source_frame') != 'camera_optical'
                or data.get('target_frame') != 'pelvis'
                or data.get('translation_units') != 'm'
                or data.get('reference_waist_yaw_deg') != 0):
            raise ValueError('Expected camera_optical -> physical pelvis extrinsics in meters at zero waist yaw')
        self.provided_zero = validate_transform(data['T_pelvis_camera_optical'])
        rotation, position = model.transform('d435_link', np.zeros(15))
        self.urdf_zero = rigid_transform(rotation @ OPTICAL_TO_LINK, position)
        self.correction = np.linalg.inv(self.urdf_zero) @ self.provided_zero

    def from_nominal(self, nominal_live):
        """Use the nominal transform from the SAME timestamped feedback sample."""
        return validate_transform(validate_transform(nominal_live) @ self.correction)

    def metadata(self):
        return dict(file=str(self.path), source_frame='camera_optical', target_frame='pelvis',
                    translation_units='m', reference_waist_yaw_deg=0,
                    T_pelvis_camera_optical_provided_zero=self.provided_zero.tolist(),
                    T_pelvis_camera_optical_urdf_zero=self.urdf_zero.tolist(),
                    live_composition='T_urdf(q) @ inverse(T_urdf(0)) @ T_provided(0)',
                    body_tilt='Applied only in the later pelvis -> level/map transform')
