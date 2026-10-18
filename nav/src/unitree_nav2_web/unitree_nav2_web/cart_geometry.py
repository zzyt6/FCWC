"""Measured-heading / grasp-geometry cart reference; no ROS or motor I/O."""
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


def rotation_z(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


def quaternion_matrix(q):
    q = np.asarray(q, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all() or not .5 <= np.linalg.norm(q) <= 1.5:
        raise ValueError('Invalid body quaternion')
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def load_settings(path):
    path = Path(path).resolve()
    settings = json.loads(path.read_text())
    limits = np.asarray(settings['task_limits'], dtype=float)
    if (settings.get('schema') != 'pushcart_navigation_v1' or limits.shape != (3, 2)
            or not np.isfinite(limits).all() or np.any(limits[:, 0] >= 0)
            or np.any(limits[:, 1] <= 0)):
        raise ValueError('Invalid pushcart navigation limits')
    workspace = path.parents[2]
    settings['display_file'] = str(workspace / 'Viser/config/cart_display.json')
    settings['robot_urdf'] = str(workspace / 'G1_URDF/g1.urdf')
    exported = json.loads((workspace / 'G1_high_policy_deploy/robot/models/v82_config.json').read_text())
    task = exported['task_command_range']
    allowed = np.array([task['vx'], [-task['vy_max'], task['vy_max']], [-task['wz_max'], task['wz_max']]])
    if np.any(limits[:, 0] < allowed[:, 0]) or np.any(limits[:, 1] > allowed[:, 1]):
        raise ValueError('Navigation limits exceed the deployed cart model range')
    # Nav2 provides the normal obstacle inflation. The combined footprint
    # defaults to the model geometry without an additional tracking envelope.
    for key, default in (('footprint_padding_m', 0.), ('tracking_margin_m', 0.),
                         ('relative_yaw_margin_rad', 0.), ('self_filter_padding_m', .04)):
        value = settings.setdefault(key, default)
        if type(value) not in (float, int) or not math.isfinite(value) or not 0 <= value <= .5:
            raise ValueError('Invalid cart geometry margin: ' + key)
    return settings


def origin(element):
    transform = np.eye(4)
    if element is not None:
        transform[:3, 3] = np.fromstring(element.get('xyz', '0 0 0'), sep=' ')
        roll, pitch, yaw = np.fromstring(element.get('rpy', '0 0 0'), sep=' ')
        cr, sr, cp, sp = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch)
        transform[:3, :3] = rotation_z(yaw) @ np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]]) @ np.array(
            [[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return transform


def convex_hull(points):
    points = sorted(set(map(tuple, points)))
    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1])-(a[1]-o[1])*(b[0]-o[0])
    def half(items):
        result = []
        for point in items:
            while len(result) >= 2 and cross(result[-2], result[-1], point) <= 0:
                result.pop()
            result.append(point)
        return result
    return np.asarray(half(points)[:-1] + half(points[::-1])[:-1])


class GantryGeometry:
    """Explicit rear test regions attached to the pelvis, independent of the cart."""

    def __init__(self, settings):
        self.enabled = settings.get('enabled', False)
        self.frame = settings.get('frame', 'base_link')
        padding = settings.get('padding_m', .03)
        if (type(self.enabled) is not bool or self.frame != 'base_link'
                or type(padding) not in (float, int) or not math.isfinite(padding)
                or not 0 <= padding <= .1):
            raise ValueError('Invalid gantry filter frame or padding')
        self.boxes = []
        for box in settings.get('boxes', []):
            bounds = np.asarray([box['min_m'], box['max_m']], dtype=float)
            if (bounds.shape != (2, 3) or not np.isfinite(bounds).all()
                    or np.any(bounds[0] >= bounds[1]) or bounds[1, 0]+padding >= 0):
                raise ValueError('Gantry regions must be finite boxes behind the robot')
            self.boxes.append((bounds[0]-padding, bounds[1]+padding))
        if self.enabled and not self.boxes:
            raise ValueError('Enabled gantry filter requires explicit rear regions')
        if len(self.boxes) > 1:
            raise ValueError('Gantry clearing supports one rear rectangle')

    def clearing_rays(self, count=720):
        """Clearing rays start inside the ignored rectangle and stop at its edge."""
        low, high = self.boxes[0]
        center, half = (low+high)/2, (high-low)/2
        angles = -math.pi + np.arange(count)*(2*math.pi/count)
        directions = np.abs(np.column_stack((np.cos(angles), np.sin(angles))))
        distances = np.divide(half[:2], directions, out=np.full_like(directions, np.inf),
                              where=directions > 1e-10).min(axis=1)
        # Stay inside boundary cells even at exact right angles.
        return center, angles, distances*.999

    def self_mask(self, points):
        points = np.asarray(points)
        inside = np.zeros(len(points), dtype=bool)
        if self.enabled:
            for low, high in self.boxes:
                inside |= np.all((points >= low) & (points <= high), axis=1)
        return inside

class CartGeometry:
    def __init__(self, settings):
        self.settings = settings
        path = Path(settings['display_file'])
        self.display = json.loads(path.read_text())
        robot = ET.parse(settings['robot_urdf']).getroot()
        self.joints = {j.find('child').get('link'): j for j in robot.findall('joint')}
        self.chains = {}
        for side in ('left', 'right'):
            link, chain = side + '_hand_palm_link', []
            while link in self.joints:
                joint = self.joints[link]
                chain.append(joint)
                link = joint.find('parent').get('link')
            self.chains[side] = chain[::-1]
        cart = ET.parse((path.parent / self.display['urdf']).resolve()).getroot()
        self.boxes, self.spheres = [], []
        base = cart.find("link[@name='base_link']")
        # Primitive model parts, not the empty volume between shelves.
        for item in list(base.findall('collision')) + list(base.findall('visual')):
            box = item.find('geometry/box')
            if box is not None:
                self.boxes.append((origin(item.find('origin')), np.fromstring(box.get('size'), sep=' ')/2))
        # Include the full swivel sweep instead of assuming the wheels point forward.
        for joint in cart.findall('joint'):
            if 'swivel_joint' in joint.get('name', ''):
                center = origin(joint.find('origin'))[:3, 3].copy()
                center[2] -= .02
                self.spheres.append((center, .10))
        points = []
        for transform, half in self.boxes:
            for x in (-half[0], half[0]):
                for y in (-half[1], half[1]):
                    for z in (-half[2], half[2]):
                        points.append((transform @ [x, y, z, 1])[:2])
        body_outline = convex_hull(points).tolist()
        handle = base.find("visual[@name='bar_0']")
        handle_transform = origin(handle.find('origin'))
        handle_half = np.fromstring(handle.find('geometry/box').get('size'), sep=' ')/2
        handle_ends = [(handle_transform @ [0, y, 0, 1])[:2].tolist()
                       for y in (-handle_half[1], handle_half[1])]
        for center, radius in self.spheres:
            points.extend(center[:2] + radius*np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]]))
        self.outline = convex_hull(points)
        # One model supplies both the navigation outline and its map symbol.
        # Body, handle and casters remain separate from the combined footprint.
        self.top_view = dict(outline=self.outline.tolist(), body=body_outline, handle=handle_ends,
                             wheels=[dict(center=center[:2].tolist(), radius=radius)
                                     for center, radius in self.spheres])

    def estimate(self, joints, quaternion, yaw):
        if not math.isfinite(yaw):
            raise ValueError('Invalid cart heading')
        body_rotation = quaternion_matrix(quaternion)
        body_yaw = math.atan2(body_rotation[1, 0], body_rotation[0, 0])
        leveled = rotation_z(-body_yaw) @ body_rotation
        palms = []
        for side, chain in self.chains.items():
            transform = np.eye(4)
            for joint in chain:
                transform = transform @ origin(joint.find('origin'))
                if joint.get('type') in ('revolute', 'continuous'):
                    angle = float(joints[joint.get('name')])
                    if not math.isfinite(angle):
                        raise ValueError('Invalid arm feedback')
                    axis = np.fromstring(joint.find('axis').get('xyz'), sep=' ')
                    axis /= np.linalg.norm(axis)
                    x, y, z = axis
                    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
                    motion = np.eye(4)
                    motion[:3, :3] = np.eye(3)+math.sin(angle)*skew+(1-math.cos(angle))*(skew @ skew)
                    transform = transform @ motion
            point = transform @ np.r_[self.display['grasp_point_in_palm_m'][side], 1.]
            palms.append(leveled @ point[:3])
        position = np.mean(palms, axis=0) - rotation_z(yaw) @ self.display['handle_center_in_cart_m']
        return dict(position=position.tolist(), yaw=float(yaw), palms=np.asarray(palms).tolist(),
                    body_rotation=leveled.tolist(),
                    position_source='抓握几何估计', heading_source='相机实测')

    def footprint(self, pose, cart_mode):
        if not cart_mode:
            return [[-.30, -.25], [.32, -.25], [.32, .25], [-.30, .25]]
        yaw, position = pose['yaw'], np.asarray(pose['position'])
        rotation = rotation_z(-yaw)[:2, :2]
        center = -rotation @ position[:2]
        robot = np.array([[-.30, -.25], [.32, -.25], [.32, .25], [-.30, .25]])
        points = [*self.outline, *((np.asarray(pose['palms'])[:, :2]-position[:2]) @ rotation.T)]
        for delta in (-self.settings['relative_yaw_margin_rad'], 0., self.settings['relative_yaw_margin_rad']):
            points.extend(robot @ rotation_z(-yaw+delta)[:2, :2].T + center)
        padding = self.settings['footprint_padding_m'] + self.settings['tracking_margin_m']
        offsets = padding*np.array([[math.cos(t), math.sin(t)] for t in np.arange(8)*math.pi/4])
        return convex_hull((np.asarray(points)[:, None, :] + offsets).reshape(-1, 2)).tolist()

    def self_mask(self, points_in_base, pose):
        points = (np.asarray(points_in_base)-pose['position']) @ rotation_z(pose['yaw'])
        inside = np.zeros(len(points), dtype=bool)
        padding = self.settings['self_filter_padding_m']
        for transform, half in self.boxes:
            local = (points-transform[:3, 3]) @ transform[:3, :3]
            inside |= np.all(np.abs(local) <= half+padding, axis=1)
        for center, radius in self.spheres:
            inside |= np.sum((points-center)**2, axis=1) <= (radius+padding)**2
        return inside

    def occluded_ranges(self, angles, ranges, pose):
        """2-D cart occlusion: preserve nearer hits; never ray-clear behind the cart."""
        rotation = rotation_z(-pose['yaw'])[:2, :2]
        ray_origin = -rotation @ np.asarray(pose['position'])[:2]
        directions = np.column_stack((np.cos(angles), np.sin(angles))) @ rotation.T
        low = self.outline.min(axis=0)-self.settings['self_filter_padding_m']
        high = self.outline.max(axis=0)+self.settings['self_filter_padding_m']
        parallel = np.abs(directions) < 1e-10
        a = np.divide(low-ray_origin, directions, out=np.zeros_like(directions), where=~parallel)
        b = np.divide(high-ray_origin, directions, out=np.zeros_like(directions), where=~parallel)
        near = np.where(parallel, -np.inf, np.minimum(a, b))
        far = np.where(parallel, np.inf, np.maximum(a, b))
        outside = (ray_origin < low) | (ray_origin > high)
        far = np.where(parallel & outside, -np.inf, far)
        enter = np.maximum(near.max(axis=1), 0.)
        leave = far.min(axis=1)
        return (leave >= enter) & (np.asarray(ranges) >= enter)
