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

from unitree_nav2_web.cart_geometry import CartGeometry, GantryGeometry, load_settings, rotation_z
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


def harness(geometry, gantry=False):
    h = SimpleNamespace(geometry=geometry, cart=pose(), received=time.monotonic(),
                        previous=None, velocity=np.zeros(3), epoch=3, odom_epoch=None,
                        last_base=-math.inf, last_filtered=-math.inf, last_scan=-math.inf,
                        last_raw=-math.inf, last_localization_scan=-math.inf,
                        localizing_without_cart=False,
                        removed_points=0, filter_error='', raw_frame='livox_frame')
    h.gantry = GantryGeometry(geometry.settings['gantry_filter'] if gantry else {})
    h.last_gantry_filtered = -math.inf
    h.gantry_removed_points, h.gantry_error = 0, ''
    h.gantry_clear_active, h.last_gantry_clear = False, -math.inf
    h._fresh = lambda: CartNavigation._fresh(h)
    h._gantry_ready = lambda: CartNavigation._gantry_ready(h)
    h._points_in_body = lambda msg: CartNavigation._points_in_body(h, msg)
    h._expire_gantry_clear = lambda: CartNavigation._expire_gantry_clear(h)
    h._localization_only = lambda msg: CartNavigation._localization_only(h, msg)
    h.get_clock = lambda: SimpleNamespace(now=lambda: Time(nanoseconds=int(time.time()*1e9), clock_type=ClockType.ROS_TIME))
    h.cloud_pub = Publisher()
    h.odom_pub = Publisher()
    h.broadcaster = Publisher()
    h.scan_pubs = {'scan': Publisher(), 'scan_localization': Publisher()}
    h.gantry_clear_pub = Publisher()
    h.gantry_clear_scan = LaserScan(ranges=[.2])
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


def test_map_symbol_uses_cart_model_without_the_robot_or_tracking_envelope(geometry):
    view = geometry.top_view
    np.testing.assert_allclose(np.ptp(view['body'], axis=0), [1.01, .55])
    np.testing.assert_allclose(np.ptp(view['outline'], axis=0), [1.2, .7])
    np.testing.assert_allclose(view['handle'], [[-.49, -.27], [-.49, .27]])
    assert len(view['wheels']) == 4
    # Combined navigation geometry also covers the robot behind the handle;
    # it must not stretch the cart-only drawing into a second robot footprint.
    assert min(x for x, _ in geometry.footprint(pose(), True)) < -1.
    assert min(x for x, _ in view['outline']) == pytest.approx(-.6)


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
    assert len(h.cloud_pub.messages) == 2
    assert h.cloud_pub.messages[-1] is msg and h.localizing_without_cart
    assert h.last_filtered == -math.inf
    h.cart['captured_monotonic'] = time.monotonic()
    CartNavigation._cloud(h, msg)
    assert len(h.cloud_pub.messages) == 3 and not h.localizing_without_cart
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
    CartNavigation._scan(h, msg, 'scan_localization')
    assert h.scan_pubs['scan_localization'].messages[-1] is msg
    assert h.last_localization_scan > 0


def test_missing_mount_keeps_lidar_for_localization_but_blocks_obstacle_scan(geometry):
    h = harness(geometry)
    h.mount = None
    msg = cloud([[0, 2, 0], [1, 0, 0]])
    CartNavigation._cloud(h, msg)
    assert h.cloud_pub.messages[-1] is msg
    assert h.localizing_without_cart and h.last_filtered == -math.inf
    scan = LaserScan(ranges=[1., 2.])
    CartNavigation._scan(h, scan, 'scan')
    CartNavigation._scan(h, scan, 'scan_localization')
    assert not h.scan_pubs['scan'].messages
    assert h.scan_pubs['scan_localization'].messages[-1] is scan


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


@pytest.mark.parametrize('mode', ['walk', 'cart', 'stale_cart'])
def test_gantry_region_filters_before_localization_in_both_modes(geometry, mode):
    h = harness(geometry, gantry=True)
    h.cart['attached'] = mode != 'walk'
    if mode == 'stale_cart':
        h.cart['captured_monotonic'] -= 1.
    # The lidar moves with the waist; the exclusion remains in pelvis coordinates.
    angle = .7
    h.mount.transform.rotation.w = math.cos(angle/2)
    h.mount.transform.rotation.z = math.sin(angle/2)
    h.mount.transform.translation.z = .5
    cart_body = geometry.boxes[0][0][:3, 3]+h.cart['position']
    points = np.array([[-.7, .2, .3], [-.7, -.2, -.6], [-1.2, 0, .2], [-.7, .9, .3],
                       [.8, 2., .2], [.2, 0., .3], cart_body, [-.3, 0., .3]])
    msg = cloud((points-[0, 0, .5]) @ rotation_z(angle), organized=True)
    CartNavigation._cloud(h, msg)
    out = h.cloud_pub.messages[-1]
    times = np.frombuffer(out.data, dtype='<f4').reshape(-1, 4)[:, 3].tolist()
    assert times == ([1236, 1237, 1238, 1239, 1241] if mode == 'cart' else
                     [1236, 1237, 1238, 1239, 1240, 1241])
    assert out.header == msg.header and out.fields == msg.fields
    assert out.height == 1 and out.row_step == out.width*out.point_step
    assert h.gantry_removed_points == 2 and h.last_gantry_filtered > 0
    assert h.localizing_without_cart == (mode == 'stale_cart')
    assert not h._gantry_ready()  # A cloud alone does not prove usable obstacle scans.


@pytest.mark.parametrize('attached', [False, True])
def test_rear_clear_keeps_scans_flowing_without_scan_transform(geometry, attached):
    h = harness(geometry, gantry=True)
    h.cart['attached'] = attached
    CartNavigation._cloud(h, cloud([[-.7, 0., 0.], [-1.5, 0., 0.]]))
    # No scan/body TF is supplied. A rear exclusion must not add another
    # transform dependency or stop the entire navigation scan stream.
    scan = LaserScan(angle_min=math.pi, angle_increment=0.,
                     ranges=[float('inf'), 1.5, .3, float('nan')])
    scan.header.frame_id = 'base_nav'
    for topic in ('scan', 'scan_localization'):
        CartNavigation._scan(h, scan, topic)
        ranges = h.scan_pubs[topic].messages[-1].ranges
        assert math.isinf(ranges[0]) and math.isnan(ranges[3])
        assert ranges[1:3] == pytest.approx([1.5, .3])
    assert h._gantry_ready() and math.isinf(scan.ranges[0])
    assert len(h.gantry_clear_pub.messages) == 1


@pytest.mark.parametrize('failure', ['no_mount', 'stale_mount', 'invalid_mount', 'wrong_frame'])
def test_rear_clear_failure_falls_back_to_normal_obstacles_and_recovers(geometry, failure):
    h = harness(geometry, gantry=True)
    h.cart['attached'] = False
    saved_mount = copy.deepcopy(h.mount)
    msg = cloud([[-.7, 0., .1], [-1.5, 0., .1]])
    if failure == 'no_mount':
        h.mount = None
    elif failure == 'stale_mount':
        h.mount.header.stamp.sec -= 1
    elif failure == 'invalid_mount':
        h.mount.transform.translation.x = float('nan')
    elif failure == 'wrong_frame':
        msg.header.frame_id = 'unexpected_frame'
    CartNavigation._cloud(h, msg)
    assert h.cloud_pub.messages[-1] is msg  # Keep actual rear returns as obstacles.
    scan = LaserScan(angle_min=math.pi, ranges=[float('inf')])
    scan.header.frame_id = 'base_nav'
    CartNavigation._scan(h, scan, 'scan')
    CartNavigation._scan(h, scan, 'scan_localization')
    assert not h._gantry_ready() and h.gantry_error
    assert h.scan_pubs['scan'].messages[-1] is scan
    assert not h.gantry_clear_pub.messages
    assert h.scan_pubs['scan_localization'].messages
    h.mount = saved_mount
    msg.header.frame_id = 'livox_frame'
    CartNavigation._cloud(h, msg)
    CartNavigation._scan(h, scan, 'scan')
    assert h._gantry_ready() and not h.gantry_error


def test_unavailable_rear_clear_does_not_add_a_navigation_gate(geometry):
    h = harness(geometry, gantry=True)
    h.cart['attached'] = False
    h.last_base = h.last_scan = time.monotonic()
    h.odom_epoch = h.epoch
    h.gantry_error = '后方区域清空暂不可用'
    h.epoch_pub, h.ready_pub, h.status_pub = Publisher(), Publisher(), Publisher()
    h.footprint_pubs = [Publisher(), Publisher()]
    def no_packets(_):
        raise BlockingIOError
    h.sock = SimpleNamespace(recvfrom=no_packets)
    CartNavigation._tick(h)
    state = json.loads(h.status_pub.messages[-1].data)
    assert state['ready'] and state['reason'] == ''
    assert not state['gantry_filter']['ready']


@pytest.mark.parametrize('settings', [
    {'enabled': True}, {'enabled': 'false'}, {'frame': 'map'}, {'padding_m': float('nan')},
    {'boxes': [{'min_m': [-1, -1, -1], 'max_m': [1, 1, 1]}]},
    {'boxes': [{'min_m': [-.5, 0, 0], 'max_m': [-1, 1, 1]}]},
])
def test_gantry_invalid_regions_are_rejected(settings):
    with pytest.raises(ValueError):
        GantryGeometry(settings)


def test_gantry_clearing_rays_stay_inside_explicit_rectangle(geometry):
    gantry = GantryGeometry(geometry.settings['gantry_filter'])
    center, angles, distances = gantry.clearing_rays()
    endpoints = center + np.column_stack((distances*np.cos(angles), distances*np.sin(angles),
                                         np.zeros(len(angles))))
    assert gantry.self_mask(endpoints).all()
    assert gantry.self_mask([center]).all()
    assert len(angles) == 720 and np.isfinite(distances).all() and (distances > 0).all()
    # The origin is in the rear region, so even the near-facing rays cannot
    # clear the corridor from the robot origin to the exclusion boundary.
    assert endpoints[:, 0].max() < -.5 and endpoints[:, 0].min() > -1.


def test_gantry_failure_replaces_cached_clearing_rays_with_empty_observation(geometry):
    h = harness(geometry, gantry=True)
    h.cart['attached'] = False
    h.last_gantry_filtered = time.monotonic()
    scan = LaserScan(angle_min=math.pi, ranges=[float('inf')])
    CartNavigation._scan(h, scan, 'scan')
    assert h.gantry_clear_active and len(h.gantry_clear_pub.messages) == 1
    h.last_gantry_filtered -= 1.
    CartNavigation._scan(h, scan, 'scan')
    assert not h.gantry_clear_active and len(h.gantry_clear_pub.messages) == 2
    assert all(math.isnan(value) for value in h.gantry_clear_pub.messages[-1].ranges)
