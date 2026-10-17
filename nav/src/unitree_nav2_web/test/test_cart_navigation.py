"""Cart frame, self masking and stale-data regressions without any robot I/O."""
import copy
import json
import math
from pathlib import Path
import socket
import time
from types import SimpleNamespace

import numpy as np
import pytest
from geometry_msgs.msg import Polygon, TransformStamped
from nav_msgs.msg import Odometry
from rclpy.time import Time
from rclpy.clock import ClockType
from sensor_msgs.msg import LaserScan, PointCloud2, PointField

from unitree_nav2_web.cart_geometry import CartGeometry, load_settings, rotation_z
from unitree_nav2_web.cart_navigation import CartNavigation, cloud_arrays

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def geometry():
    return CartGeometry(load_settings(ROOT / 'config/pushcart_navigation.json'))


def pose():
    return dict(position=[1., 0., -.7], yaw=0., palms=[[.5, .2, .2], [.5, -.2, .2]],
                body_rotation=np.eye(3).tolist(), valid=True, attached=True, epoch=3,
                captured_monotonic=time.monotonic())


class Publisher:
    def __init__(self):
        self.messages = []

    def publish(self, msg):
        self.messages.append(msg)

    sendTransform = publish


def harness(geometry):
    h = SimpleNamespace(geometry=geometry, cart=pose(), received=time.monotonic(),
                        previous=None, velocity=np.zeros(3), epoch=3, odom_epoch=None,
                        last_base=-math.inf, last_filtered=-math.inf, last_scan=-math.inf,
                        removed_points=0, filter_error='', raw_frame='livox_frame')
    h._fresh = lambda: CartNavigation._fresh(h)
    h.get_clock = lambda: SimpleNamespace(now=lambda: Time(nanoseconds=int(time.time()*1e9), clock_type=ClockType.ROS_TIME))
    h.cloud_pub = Publisher()
    h.odom_pub = Publisher()
    h.broadcaster = Publisher()
    h.scan_pubs = {'scan': Publisher(), 'scan_localization': Publisher()}
    h.mount = TransformStamped()
    h.mount.header.stamp = h.get_clock().now().to_msg()
    h.mount.transform.rotation.w = 1.
    return h


def cloud(points, organized=False):
    fields = [PointField(name=name, offset=i*4, datatype=PointField.FLOAT32, count=1)
              for i, name in enumerate(('x', 'y', 'z', 'time'))]
    records = np.zeros((len(points), 4), dtype='<f4')
    records[:, :3] = points
    records[:, 3] = np.arange(len(points)) + 1234
    msg = PointCloud2(height=1, width=len(points), fields=fields, point_step=16,
                      row_step=len(points)*16, data=records.tobytes())
    msg.header.frame_id = 'livox_frame'
    msg.header.stamp.sec = 17  # Sensor clock must never be rewritten.
    if organized:
        msg.height, msg.width, msg.row_step = 2, len(points)//2, len(points)*8+8
        split = len(points)*8
        msg.data = records.tobytes()[:split]+b'padding!'+records.tobytes()[split:]+b'padding!'
    return msg


def test_grasp_geometry_uses_body_roll_pitch_and_only_relative_yaw(geometry):
    joints = {j.get('name'): 0. for chain in geometry.chains.values() for j in chain}
    zero = geometry.estimate(joints, [1, 0, 0, 0], 0.)
    world_yaw = geometry.estimate(joints, [math.cos(.7), 0, 0, math.sin(.7)], 0.)
    np.testing.assert_allclose(zero['position'], world_yaw['position'], atol=1e-12)
    angled = geometry.estimate(joints, [1, 0, 0, 0], .4)
    handle = np.asarray(geometry.display['handle_center_in_cart_m'])
    np.testing.assert_allclose(np.asarray(angled['position']) + rotation_z(.4) @ handle,
                               np.mean(angled['palms'], axis=0))
    assert zero['position'][0] > 0.4
    assert np.asarray(geometry.footprint(zero, True))[:, 0].ptp() > 1.4


def test_self_mask_keeps_external_obstacles_and_empty_cart_volume(geometry):
    p = pose()
    transform, half = geometry.boxes[0]
    body = transform[:3, 3]+p['position']
    points = [body, body+[0, 2, 0], [1, 0, 3]]
    assert geometry.self_mask(points, p).tolist() == [True, False, False]
    # Transforming the entire arrangement must preserve the mask.
    angle = .6
    rotated = copy.deepcopy(p)
    rotated['yaw'] = angle
    rotated['position'] = (rotation_z(angle) @ p['position']).tolist()
    assert geometry.self_mask(np.asarray(points) @ rotation_z(angle).T, rotated).tolist() == [True, False, False]


def test_occlusion_never_clears_behind_cart_but_keeps_near_hits(geometry):
    angles = np.array([0, 0, 0, math.pi, math.pi/2])
    ranges = np.array([.1, 4, np.inf, np.inf, 2])
    assert geometry.occluded_ranges(angles, ranges, pose()).tolist() == [False, True, True, False, False]


def test_raw_filter_preserves_sensor_time_fields_and_recovers_without_odometry(geometry):
    h = harness(geometry)
    body = geometry.boxes[0][0][:3, 3]+h.cart['position']
    msg = cloud([body, [0, 2, 0], body, [0, -2, 0]], organized=True)
    xyz, raw = cloud_arrays(msg)
    assert raw.shape == (4,)
    CartNavigation._cloud(h, msg)
    assert h.cloud_pub.messages, h.filter_error
    out = h.cloud_pub.messages[-1]
    assert out.width == 2 and out.height == 1 and h.removed_points == 2
    assert out.header == msg.header and out.fields == msg.fields
    assert np.frombuffer(out.data, dtype='<f4').reshape(-1, 4)[:, 3].tolist() == [1235, 1237]
    h.cart['captured_monotonic'] -= 1
    CartNavigation._cloud(h, msg)
    assert len(h.cloud_pub.messages) == 1
    h.cart['captured_monotonic'] = time.monotonic()
    CartNavigation._cloud(h, msg)
    assert len(h.cloud_pub.messages) == 2  # No odom input needed to restart.
    h.cart['attached'] = False
    CartNavigation._cloud(h, msg)
    assert h.cloud_pub.messages[-1] is msg


def test_cart_odometry_and_tf_apply_robot_world_heading(geometry):
    h = harness(geometry)
    msg = Odometry()
    msg.header.frame_id = 'odom'
    msg.pose.pose.position.x = 10.
    msg.pose.pose.position.y = 20.
    msg.pose.pose.orientation.w = math.cos(math.pi/4)
    msg.pose.pose.orientation.z = math.sin(math.pi/4)
    h.cart['yaw'] = .3
    CartNavigation._odom(h, msg)
    out = h.odom_pub.messages[-1]
    assert out.child_frame_id == 'nav_control'
    assert out.pose.pose.position.x == pytest.approx(10)
    assert out.pose.pose.position.y == pytest.approx(21)
    assert out.pose.pose.orientation.z == pytest.approx(math.sin((math.pi/2+.3)/2))
    assert h.broadcaster.messages[0].header.frame_id == 'base_nav'
    h.received -= 1
    CartNavigation._odom(h, msg)
    assert len(h.odom_pub.messages) == 1


def test_final_scans_mask_occlusion_and_pause_on_stale_cart(geometry):
    h = harness(geometry)
    msg = LaserScan(angle_min=0., angle_increment=0., ranges=[.1, 4., float('inf')])
    CartNavigation._scan(h, msg, 'scan')
    out = h.scan_pubs['scan'].messages[-1]
    assert out.ranges[0] == pytest.approx(.1)
    assert all(math.isnan(v) for v in out.ranges[1:])
    assert msg.ranges[1] == 4.  # Input is not mutated.
    h.received -= 1
    CartNavigation._scan(h, msg, 'scan')
    assert len(h.scan_pubs['scan'].messages) == 1


def test_reference_epoch_waits_for_new_odom_and_filtered_scan(geometry):
    h = harness(geometry)
    h.received = -math.inf
    h.state = {}
    h.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    h.sock.bind(('127.0.0.1', 0))
    h.sock.setblocking(False)
    h.epoch_pub, h.ready_pub, h.status_pub = Publisher(), Publisher(), Publisher()
    h.footprint_pubs = [Publisher(), Publisher()]
    sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        data = dict(magic='G1LOCO1', sent_monotonic=time.monotonic(), cart=pose())
        data['cart']['epoch'] = 5
        sender.sendto(json.dumps(data).encode(), h.sock.getsockname())
        CartNavigation._tick(h)
        assert h.epoch == 5 and not h.ready_pub.messages[-1].data
        assert isinstance(h.footprint_pubs[0].messages[-1], Polygon)
        h.last_base = h.last_filtered = h.last_scan = time.monotonic()
        h.odom_epoch = 5
        CartNavigation._tick(h)
        assert h.ready_pub.messages[-1].data
        # A stale packet cannot switch the reference back to the preceding mode.
        data['cart']['epoch'] = 3
        data['sent_monotonic'] -= 1
        sender.sendto(json.dumps(data).encode(), h.sock.getsockname())
        CartNavigation._tick(h)
        assert h.epoch == 5
        h.cart['captured_monotonic'] -= 1
        CartNavigation._tick(h)
        assert not h.ready_pub.messages[-1].data
    finally:
        h.sock.close()
        sender.close()
