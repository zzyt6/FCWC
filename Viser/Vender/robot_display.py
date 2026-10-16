"""Display-only articulation; never changes the deployment or its joint count."""
from io import BytesIO
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

from core.hand_grasp import HAND_JOINTS
from core.robot import JOINT_NAMES

HAND_NAMES = [f'{side}_hand_{name}_joint' for side in ('left', 'right') for name in HAND_JOINTS]


class UrdfDisplayUpdater:
    """Update changed local joint frames while keeping the URDF graph current.

    Construct after ViserUrdf.update_cfg has initialized every frame. A child's
    local transform stays constant when only its ancestors move; Viser handles
    that hierarchy. Avoid repeating graph queries and pose setters for it.
    """
    def __init__(self, model):
        self.robot = model._urdf
        self.scale = model._scale
        self.frames = tuple(zip(model._joint_map_values, model._joint_frames))
        self.indices = {name: indices[0] for name, indices in
                        zip(self.robot.actuated_joint_names, self.robot.actuated_dof_indices)}

    def update_cfg(self, configuration):
        changed = {name: value for name, value in configuration.items()
                   if self.robot.cfg[self.indices[name]] != value}
        if not changed:
            return
        # Also updates the palm transforms used to place the reference cart.
        self.robot.update_cfg(changed)
        affected = [(joint, frame) for joint, frame in self.frames
                    if joint.name in changed or joint.mimic is not None]
        # update_cfg already computed these parent-to-child transforms. Reuse
        # the graph edges instead of doing the same joint kinematics twice.
        transforms = [self.robot.get_transform(joint.child, joint.parent) for joint, _ in affected]
        if not transforms:
            return
        quaternions = Rotation.from_matrix(np.asarray(transforms)[:, :3, :3]).as_quat()[:, [3, 0, 1, 2]]
        for (joint, frame), transform, quaternion in zip(affected, transforms, quaternions):
            frame.wxyz = quaternion
            # Revolute joints keep their local origin; translations were set at startup.
            if joint.type == 'prismatic':
                frame.position = transform[:3, 3] * self.scale


def load_display_robot(path):
    import yourdfpy
    tree = ET.parse(path)
    for joint in tree.getroot().findall('joint'):
        if joint.get('name') in HAND_NAMES:
            joint.set('type', 'revolute')
    return yourdfpy.URDF.load(BytesIO(ET.tostring(tree.getroot())),
                             mesh_dir=str(Path(path).parent), load_collision_meshes=False)


def feedback_configuration(urdf, q, hands=None, grasp_reference=None):
    values = {name: float(q[i]) for i, name in enumerate(JOINT_NAMES) if name in urdf.actuated_joint_names}
    hands = hands or {}
    if grasp_reference is not None:
        reference = np.asarray(grasp_reference)
        if reference.size == 14 and np.isfinite(reference).all():
            values.update({name: float(value) for name, value in zip(HAND_NAMES, reference.ravel())})
            return values
    positions = np.asarray(hands.get('q', []))
    if positions.shape == (14,):
        for side, offset in (('left', 0), ('right', 7)):
            if hands.get(side + '_healthy') and np.isfinite(positions[offset:offset+7]).all():
                values.update({HAND_NAMES[i]: float(positions[i]) for i in range(offset, offset+7)})
    return values


def current_grasp_reference(config, hands):
    """Reuse live GUI/config angles, never a grasp baked into a training asset."""
    if not hands.get('active') or hands.get('pose') not in ('grasp', 'grasp_compensated'):
        return None
    values = np.asarray(hands.get('grasp', config.hand_grasp)).copy()
    if hands.get('compensation_enabled') and config.hand_zero is not None:
        values += config.hand_zero
    return values
