"""Cart frame, self masking and stale-data regressions without any robot I/O."""
import copy
from array import array
from collections import deque
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
from rclpy.serialization import deserialize_message, serialize_message
from sensor_msgs.msg import LaserScan, PointCloud2, PointField, TimeReference

from unitree_nav2_web.cart_geometry import CartGeometry, CartPoseHistory, GantryGeometry, PelvisYawHistory, load_settings, rotation_z
from unitree_nav2_web.cart_navigation import CartNavigation, cloud_arrays, filtered_cloud

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def geometry():
    return CartGeometry(load_settings(ROOT / 'config/pushcart_navigation.json'))


def test_navigation_lateral_override_preserves_other_limits_and_model_metadata(tmp_path):
    settings = json.loads((ROOT / 'config/pushcart_navigation.json').read_text())
    config = tmp_path / 'nav/config/pushcart_navigation.json'
    model = tmp_path / 'G1_high_policy_deploy/robot/models/v82_config.json'
    config.parent.mkdir(parents=True)
    model.parent.mkdir(parents=True)
    exported = (ROOT.parent / 'G1_high_policy_deploy/robot/models/v82_config.json').read_text()
    model.write_text(exported)
    limits = [[-.2, .6], [-.3, .3], [-.4, .4]]
    settings['task_limits'] = limits
    config.write_text(json.dumps(settings))
    assert load_settings(config)['task_limits'] == limits
    assert model.read_text() == exported
    for axis, side, value in [(1, 0, -.31), (1, 1, .31), (0, 1, .91), (2, 1, .61)]:
        invalid = copy.deepcopy(settings)
        invalid['task_limits'][axis][side] = value
        config.write_text(json.dumps(invalid))
        with pytest.raises(ValueError, match='deployment range'):
            load_settings(config)


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
                        odom_clock_stamped=None, navigation_odometry={},
                        last_base=-math.inf, last_filtered=-math.inf, last_scan=-math.inf,
                        last_raw=-math.inf, last_localization_scan=-math.inf,
                        localizing_without_cart=False,
                        removed_points=0, filter_error='', scan_error='', raw_frame='livox_frame')
    h.gantry = GantryGeometry(geometry.settings['gantry_filter'] if gantry else {})
    h.cart_history, h.pending_odometry, h.odometry_error = CartPoseHistory(), deque(maxlen=10), ''
    h.mount_history = CartPoseHistory(max_gap=.05, max_samples=256)
    h.pending_clouds = deque()
    now = time.monotonic()
    h.lidar_clock = (now-17., now)
    h.cart_history.add(dict(h.cart, sample_monotonic=now))
    h._publish_pending_clouds = lambda: CartNavigation._publish_pending_clouds(h)
    h._filter_cloud = lambda msg, wait=False: CartNavigation._filter_cloud(h, msg, wait=wait)
    h._publish_pending_odometry = lambda: CartNavigation._publish_pending_odometry(h)
    h._odom = lambda msg, cart=None: CartNavigation._odom(h, msg, cart)
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
    h.mount.header.frame_id, h.mount.child_frame_id = 'base_link', 'livox_frame'
    h.mount.transform.rotation.w = 1.
    CartNavigation._mount(h, h.mount)
    return h


def expire_pending_clouds(h):
    h.pending_clouds = deque((received-1., epoch, msg) for received, epoch, msg in h.pending_clouds)
    h._publish_pending_clouds()


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


def test_visual_cart_center_is_independent_of_hand_slip(geometry):
    joints = {j.get('name'): 0. for chain in geometry.chains.values() for j in chain}
    matrix = np.eye(4)
    matrix[:3, :3] = rotation_z(.2)
    matrix[:3, 3] = [1.2, .03, -.8]
    observation = dict(valid=True, yaw_rad=.2, T_level_cart=matrix.tolist(), tag_ids=[0, 1], rmse_px=.3)
    first = geometry.estimate_visual(joints, [1, 0, 0, 0], observation)
    joints['left_elbow_joint'] = joints['right_elbow_joint'] = .8
    slipped = geometry.estimate_visual(joints, [1, 0, 0, 0], observation)
    np.testing.assert_array_equal(first['position'], slipped['position'])
    np.testing.assert_allclose(slipped['position'], [1.2, .03, -.8])
    assert not np.allclose(first['palms'], slipped['palms'])
    assert first['yaw'] == .2 and 'URDF' in first['position_source']
    with pytest.raises(ValueError):
        geometry.estimate_visual(joints, [1, 0, 0, 0], dict(valid=False, reason='lost tag'))


def test_delayed_heading_preserves_turning_and_handles_yaw_wrap():
    history = PelvisYawHistory()
    for index in range(60):
        stamp = 10. + index*.02
        pelvis_yaw = 3.1 + .4*(stamp-10.)
        current = history.add(dict(received=stamp, session=1,
            quat=[math.cos(pelvis_yaw/2), 0., 0., math.sin(pelvis_yaw/2)]))
        if index < 10:
            continue
        captured = stamp-.11  # Between two feedback samples, including the +/- pi crossing.
        cart_yaw = .2*(captured-10.)
        camera_yaw = cart_yaw-(3.1+.4*(captured-10.))
        aligned, _ = history.align(camera_yaw, captured, current)
        error = current+aligned-cart_yaw
        assert math.atan2(math.sin(error), math.cos(error)) == pytest.approx(0., abs=1e-10)
    with pytest.raises(ValueError, match='时间对齐'):
        history.align(0., stamp-.5-1., current)
    history.add(dict(received=stamp+.02, session=2, quat=[1., 0., 0., 0.]))
    with pytest.raises(ValueError, match='时间对齐'):
        history.align(0., stamp-.11, 0.)


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


@pytest.mark.parametrize('source', ['grasp', 'apriltag'])
@pytest.mark.parametrize('scan_ms', [100, 20])
def test_raw_scan_uses_each_point_time_during_body_and_cart_motion(
        geometry, monkeypatch, record_property, source, scan_ms):
    monkeypatch.setattr(time, 'monotonic', lambda: 100.2)
    h = harness(geometry)
    h.get_clock = lambda: SimpleNamespace(now=lambda: Time(seconds=200.2, clock_type=ClockType.ROS_TIME))
    h.cart_history.samples.clear()
    h.mount_history.samples.clear()
    reference = TimeReference(source='lidar_imu_clock')
    reference.header.stamp = Time(seconds=200.2).to_msg()
    reference.time_ref = Time(seconds=17.2).to_msg()  # Independent sensor clock.
    CartNavigation._lidar_clock(h, reference)

    def cart_at(t):
        angle = 1.5*t
        c, s = math.cos(angle), math.sin(angle)
        cart = dict(pose(), pose_source=source, position=[1.+.3*t, -.3*t, -.7],
                    yaw=-t, sample_monotonic=100.+t,
                    body_rotation=[[1., 0., 0.], [0., c, -s], [0., s, c]])
        if source == 'apriltag':
            cart['rotation'] = rotation_z(-t).tolist()
        return cart

    for t in np.arange(0., .201, .01):
        mount = TransformStamped()
        mount.header.frame_id, mount.child_frame_id = 'base_link', 'livox_frame'
        mount.header.stamp = Time(seconds=200.+t).to_msg()
        mount.transform.rotation.w = math.cos(t)
        mount.transform.rotation.z = math.sin(t)
        mount.transform.translation.z = .5
        CartNavigation._mount(h, mount)
        h.cart_history.add(cart_at(t))
    h.cart = cart_at(.2)
    center, half = geometry.boxes[0]
    local = np.array([center[:3, 3]+[0., y, 0.]
                      for y in (half[1]-.005, -half[1]+.005, half[1]+.065, -half[1]-.065)])
    raw, offsets = [], []
    for t in np.linspace(0., (scan_ms-1)*.001, 50):
        cart = cart_at(t)
        level = local @ rotation_z(cart['yaw']).T + cart['position']
        assert geometry.self_mask(level, cart).tolist() == [True, True, False, False]
        body = level @ cart['body_rotation']
        raw.extend((body-[0., 0., .5]) @ rotation_z(2*t))
        offsets.extend([t*1e9]*4)
    msg = cloud(raw)
    data = np.frombuffer(msg.data, dtype='<f4').reshape(-1, 4).copy()
    data[:, 3] = offsets
    msg.data = array('B', data.tobytes())
    # Reproduce the former whole-scan/latest-pose filter independently.
    latest_body = np.asarray(raw) @ rotation_z(.4).T+[0., 0., .5]
    old_keep = ~geometry.self_mask(latest_body @ np.asarray(h.cart['body_rotation']).T, h.cart)
    expected = np.tile([False, False, True, True], 50)
    assert np.count_nonzero(old_keep != expected) > 20
    # The measured data rates are ~20k returns/100 ms or ~4k/20 ms.
    data, expected = np.repeat(data, scan_ms, axis=0), np.repeat(expected, scan_ms)
    msg.width, msg.row_step = len(data), len(data)*16
    msg.data = array('B', data.tobytes())
    started = time.perf_counter()
    CartNavigation._cloud(h, msg)
    record_property('filter_seconds', time.perf_counter()-started)
    assert not h.pending_clouds, h.filter_error
    out = h.cloud_pub.messages[-1]
    assert out.header == msg.header and out.fields == msg.fields
    assert bytes(out.data) == data[expected].tobytes()


def test_raw_scan_waits_for_delayed_camera_then_recovers_without_lio(geometry, monkeypatch):
    clock = [100.]
    monkeypatch.setattr(time, 'monotonic', lambda: clock[0])
    h = harness(geometry)
    h.cart_history.samples.clear()
    h.cart_history.add(dict(h.cart, sample_monotonic=99.99))
    msg = cloud([[0., 2., 0.], [0., -2., 0.]])
    CartNavigation._cloud(h, msg)
    assert len(h.pending_clouds) == 1 and not h.cloud_pub.messages
    clock[0] = 100.08
    h.cart_history.add(dict(h.cart, sample_monotonic=100.01))
    h._publish_pending_clouds()
    assert not h.pending_clouds and h.cloud_pub.messages
    assert not h.localizing_without_cart and h.last_filtered == clock[0]
    # Loss of pose history releases actual raw measurements after the bound.
    h.cart_history.samples.clear()
    CartNavigation._cloud(h, msg)
    assert len(h.pending_clouds) == 1
    clock[0] = 100.29
    h._publish_pending_clouds()
    assert not h.pending_clouds and h.localizing_without_cart
    assert bytes(h.cloud_pub.messages[-1].data) == bytes(msg.data)
    assert h.last_filtered == -math.inf


@pytest.mark.parametrize('bigendian', [False, True])
@pytest.mark.parametrize('retained', [0, 1, 20000])
def test_filtered_livox_records_survive_ros_serialization(bigendian, retained):
    # Actual Livox layout includes an unaligned float64 timestamp at byte 18.
    # Filtering must copy complete records, including fields unused by XYZ.
    endian = '>' if bigendian else '<'
    dtype = np.dtype(dict(names=['x', 'y', 'z', 'intensity', 'tag', 'line', 'timestamp'],
                          formats=[endian+'f4']*4+['u1', 'u1', endian+'f8'],
                          offsets=[0, 4, 8, 12, 16, 17, 18], itemsize=26))
    records = np.zeros(20000, dtype=dtype)
    records['x'] = np.linspace(-2., 2., len(records))
    records['intensity'] = 32.5
    records['tag'], records['line'] = 255, 3
    records['timestamp'] = 1792342044.+np.arange(len(records))*.000005
    types = [PointField.FLOAT32]*4+[PointField.UINT8]*2+[PointField.FLOAT64]
    fields = [PointField(name=name, offset=dtype.fields[name][1], datatype=kind, count=1)
              for name, kind in zip(dtype.names, types)]
    msg = PointCloud2(height=1, width=len(records), fields=fields, point_step=26,
                      row_step=26*len(records), is_bigendian=bigendian,
                      is_dense=True, data=records.tobytes())
    msg.header.frame_id = 'livox_frame'
    msg.header.stamp.sec, msg.header.stamp.nanosec = 1792342044, 123456789
    original = copy.deepcopy(msg)
    _, raw = cloud_arrays(msg)
    keep = np.arange(len(records)) < retained
    result = deserialize_message(serialize_message(filtered_cloud(msg, raw, keep)), PointCloud2)
    assert result.header == msg.header and result.fields == msg.fields
    assert result.is_bigendian == bigendian and result.is_dense
    assert (result.height, result.width, result.point_step, result.row_step) == (1, retained, 26, retained*26)
    assert bytes(result.data) == records[:retained].tobytes()
    assert msg == original


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


@pytest.mark.parametrize('period', [.02, .05])
def test_cart_velocity_updates_each_frame_with_same_filter_response(geometry, monkeypatch, period):
    h = harness(geometry)
    clock = [time.monotonic()]
    monkeypatch.setattr(time, 'monotonic', lambda: clock[0])
    msg = Odometry()
    msg.pose.pose.orientation.w = 1.
    CartNavigation._odom(h, msg)
    previous = 0.
    for index in range(1, round(.2/period)+1):
        clock[0] += period
        h.received = h.cart['captured_monotonic'] = clock[0]
        msg.pose.pose.position.x = .3*index*period
        CartNavigation._odom(h, msg)
        velocity = h.odom_pub.messages[-1].twist.twist.linear.x
        assert previous < velocity < .3
        previous = velocity
    assert previous == pytest.approx(.3*(1-.65**4), abs=1e-7)


def test_cart_velocity_uses_odometry_sample_time_not_callback_spacing(geometry, monkeypatch):
    h = harness(geometry)
    clock = [time.monotonic()]
    monkeypatch.setattr(time, 'monotonic', lambda: clock[0])
    msg = Odometry()
    msg.pose.pose.orientation.w = 1.
    for index in range(51):
        # Samples arrive in alternating 6/34 ms bursts, but were taken at 50 Hz.
        clock[0] += .006 if index % 2 else .034
        h.received = h.cart['captured_monotonic'] = clock[0]
        msg.header.stamp = Time(seconds=10.+index*.02).to_msg()
        msg.pose.pose.position.x = .3*index*.02
        CartNavigation._odom(h, msg)
    assert h.odom_pub.messages[-1].twist.twist.linear.x == pytest.approx(.3, abs=.001)
    assert h.navigation_odometry['vx_m_s'] == h.odom_pub.messages[-1].twist.twist.linear.x


@pytest.mark.parametrize('turn_rate', [0., .15])
def test_delayed_base_pose_uses_matching_cart_geometry_through_body_sway(geometry, monkeypatch, turn_rate):
    h, old = harness(geometry), harness(geometry)
    clock = [10.]
    monkeypatch.setattr(time, 'monotonic', lambda: clock[0])
    h.get_clock = lambda: SimpleNamespace(now=lambda: Time(seconds=1000.+clock[0]))
    samples, old_samples = [], []
    next_feedback = 9.8
    for index in range(41):
        captured = 10.+index*.1
        clock[0] = captured + [.04, .10, .06][index % 3]
        while next_feedback <= clock[0]:
            t = next_feedback-10.
            cart_yaw = turn_rate*t
            body_yaw = cart_yaw+.15*math.sin(4*math.pi*t)
            p = pose()
            p.update(sample_monotonic=next_feedback, pose_source='grasp',
                     yaw=cart_yaw-body_yaw,
                     position=(rotation_z(cart_yaw-body_yaw) @ [.8, 0., 0.]).tolist(),
                     captured_monotonic=clock[0]-.08)
            h.cart_history.add(p)
            h.cart = old.cart = p
            next_feedback += .02
        h.received = old.received = clock[0]
        t = captured-10.
        cart_yaw = turn_rate*t
        body_yaw = cart_yaw+.15*math.sin(4*math.pi*t)
        cart_point = (np.array([.3*t, 0., 0.]) if not turn_rate else
                      .3/turn_rate*np.array([math.sin(cart_yaw), 1-math.cos(cart_yaw), 0.]))
        base_point = cart_point-rotation_z(cart_yaw) @ [.8, 0., 0.]
        msg = Odometry()
        msg.header.stamp = Time(seconds=1000.+captured).to_msg()
        msg.pose.pose.position.x, msg.pose.pose.position.y = map(float, base_point[:2])
        msg.pose.pose.orientation.w, msg.pose.pose.orientation.z = math.cos(body_yaw/2), math.sin(body_yaw/2)
        CartNavigation._receive_odom(h, msg)
        # The old composition differentiated old LiDAR poses + latest geometry.
        CartNavigation._odom(old, msg)
        assert len(h.odom_pub.messages) == index+1, h.odometry_error
        if index > 10:
            out = h.odom_pub.messages[-1].twist.twist
            samples.append([out.linear.x, out.linear.y, out.angular.z])
            out = old.odom_pub.messages[-1].twist.twist
            old_samples.append([out.linear.x, out.linear.y, out.angular.z])
    samples, old_samples = np.array(samples), np.array(old_samples)
    np.testing.assert_allclose(samples[:, 0], .3, atol=.01)
    np.testing.assert_allclose(samples[:, 1], 0., atol=.012)
    np.testing.assert_allclose(samples[:, 2], turn_rate, atol=.02)
    assert np.max(np.abs(old_samples[:, 1])) > .3
    assert np.max(np.abs(old_samples[:, 2]-turn_rate)) > .4


def test_cart_time_pairing_waits_for_feedback_and_rejects_epoch_change(geometry, monkeypatch):
    h = harness(geometry)
    h.cart_history.samples.clear()
    clock = [10.1]
    monkeypatch.setattr(time, 'monotonic', lambda: clock[0])
    h.get_clock = lambda: SimpleNamespace(now=lambda: Time(seconds=1000.+clock[0]))
    h.received = h.cart['captured_monotonic'] = clock[0]
    msg = Odometry()
    msg.header.stamp = Time(seconds=1010.).to_msg()
    msg.pose.pose.orientation.w = 1.
    CartNavigation._receive_odom(h, msg)
    assert not h.odom_pub.messages
    assert '测量时刻' in h.odometry_error
    for stamp in (9.99, 10.01):
        h.cart_history.add(dict(h.cart, sample_monotonic=stamp))
    h._publish_pending_odometry()
    assert len(h.odom_pub.messages) == 1
    assert h.navigation_odometry['cart_sample_monotonic'] == pytest.approx(10.)
    msg.header.stamp = Time(seconds=1010.05).to_msg()
    CartNavigation._receive_odom(h, msg)
    h.epoch += 2
    h._publish_pending_odometry()
    assert len(h.odom_pub.messages) == 1
    assert not h.pending_odometry


def test_cart_history_keeps_pose_sources_separate_and_handles_angle_wrap():
    history = CartPoseHistory()
    p = dict(pose(), pose_source='grasp', yaw=math.pi-.1)
    history.add(dict(p, sample_monotonic=10.))
    history.add(dict(p, sample_monotonic=10.02, yaw=-math.pi+.1))
    assert abs(history.at(10.01)['yaw']) == pytest.approx(math.pi)
    history.add(dict(p, sample_monotonic=10.16))
    # Missing 50 Hz measurements must not become a smooth, invented pose
    # across a 140 ms gap, especially while the pelvis is swaying.
    assert history.at(10.08) is None
    assert history.at(10.2) is None
    history.add(dict(p, pose_source='apriltag', sample_monotonic=10.18))
    assert history.at(10.01) is None


def test_duplicate_odometry_does_not_change_cart_velocity_and_walk_keeps_base_twist(geometry, monkeypatch):
    h = harness(geometry)
    clock = [time.monotonic()]
    monkeypatch.setattr(time, 'monotonic', lambda: clock[0])
    msg = Odometry()
    msg.pose.pose.orientation.w = 1.
    msg.header.stamp = Time(seconds=10.).to_msg()
    CartNavigation._odom(h, msg)
    clock[0] += .02
    msg.header.stamp = Time(seconds=10.02).to_msg()
    msg.pose.pose.position.x = .006
    CartNavigation._odom(h, msg)
    published = len(h.odom_pub.messages)
    expected = h.velocity.copy()
    for stamp in (10.02, 10.01):
        clock[0] += .02
        msg.header.stamp = Time(seconds=stamp).to_msg()
        msg.pose.pose.position.x = 100.
        CartNavigation._odom(h, msg)
        np.testing.assert_array_equal(h.velocity, expected)
        assert len(h.odom_pub.messages) == published
    h.cart['attached'] = False
    msg.header.stamp = Time(seconds=10.1).to_msg()
    msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.angular.z = .2, -.03, .1
    CartNavigation._odom(h, msg)
    assert h.odom_pub.messages[-1].twist == msg.twist


def test_final_scans_mask_occlusion_and_pause_on_stale_cart(geometry):
    h = harness(geometry)
    msg = LaserScan(angle_min=0., angle_increment=0., ranges=[.1, 4., float('inf')])
    msg.header.stamp = h.get_clock().now().to_msg()
    h.cart_history.add(dict(h.cart, sample_monotonic=time.monotonic()))
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


@pytest.mark.parametrize('source', ['grasp', 'apriltag'])
def test_delayed_scan_uses_cart_at_acquisition_and_preserves_external_returns(geometry, monkeypatch, source):
    h = harness(geometry)
    monkeypatch.setattr(time, 'monotonic', lambda: 100.2)
    h.get_clock = lambda: SimpleNamespace(now=lambda: Time(seconds=200.2))
    h.received = 100.2
    h.cart.update(pose_source=source, captured_monotonic=100.2)
    for stamp, angle in [(100.1, .15), (100.2, -.15)]:
        h.cart.update(sample_monotonic=stamp, yaw=angle,
                      position=[.8*math.cos(angle), .8*math.sin(angle), -.7])
        h.cart_history.add(h.cart)
    old = h.cart_history.at(100.1)
    scan = LaserScan(angle_min=-math.pi, angle_increment=math.pi/360, ranges=[1.]*721)
    scan.header.frame_id = 'base_nav'
    scan.header.stamp = Time(seconds=200.1).to_msg()
    angles = scan.angle_min + np.arange(len(scan.ranges))*scan.angle_increment
    expected = geometry.occluded_ranges(angles, scan.ranges, old)
    wrong = geometry.occluded_ranges(angles, scan.ranges, h.cart)
    assert np.any(wrong & ~expected), 'The regression must include external hits the latest pose would hide'
    assert np.any(expected & ~wrong), 'The regression must also include newly exposed cart returns'
    for topic in ('scan', 'scan_localization'):
        CartNavigation._scan(h, scan, topic)
        output = h.scan_pubs[topic].messages[-1]
        np.testing.assert_array_equal(np.isnan(output.ranges), expected)
        assert output.header == scan.header

    # A missing pose must not turn an unknown occlusion into clear space.
    h.cart_history.samples.clear()
    CartNavigation._scan(h, scan, 'scan')
    assert len(h.scan_pubs['scan'].messages) == 1
    CartNavigation._scan(h, scan, 'scan_localization')
    assert h.scan_pubs['scan_localization'].messages[-1] is scan


def test_missing_mount_keeps_lidar_for_localization_but_blocks_obstacle_scan(geometry):
    h = harness(geometry)
    h.mount = None
    h.mount_history.samples.clear()
    msg = cloud([[0, 2, 0], [1, 0, 0]])
    CartNavigation._cloud(h, msg)
    expire_pending_clouds(h)
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
    h.mount_history.samples.clear()
    CartNavigation._mount(h, h.mount)
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
    scan.header.stamp = h.get_clock().now().to_msg()
    h.cart_history.add(dict(h.cart, sample_monotonic=time.monotonic()))
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
    h.mount_history.samples.clear()
    msg = cloud([[-.7, 0., .1], [-1.5, 0., .1]])
    if failure == 'no_mount':
        h.mount = None
    elif failure == 'stale_mount':
        h.mount.header.stamp.sec -= 1
    elif failure == 'invalid_mount':
        h.mount.transform.translation.x = float('nan')
    elif failure == 'wrong_frame':
        msg.header.frame_id = 'unexpected_frame'
    if h.mount is not None:
        CartNavigation._mount(h, h.mount)
    CartNavigation._cloud(h, msg)
    expire_pending_clouds(h)
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
    CartNavigation._mount(h, h.mount)
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
