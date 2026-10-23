"""Navigation reference, cart self filtering and observation freshness.

The motor executor is the sole source of cart attachment/epoch and selected pose.
FAST-LIO and AMCL retain the robot's base_nav chain. Only navigation follows
nav_control, which coincides with base_nav when walking and cart_nav when pushing.
"""
from array import array
from collections import deque
import copy
import json
import math
from pathlib import Path
import socket
import threading
import time

import numpy as np
import rclpy
from geometry_msgs.msg import Point32, Polygon, TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import LaserScan, PointCloud2, TimeReference
from std_msgs.msg import Bool, String, UInt32
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster

from .cart_geometry import CartGeometry, CartPoseHistory, GantryGeometry, load_settings, quaternion_matrix, rotation_z
from .sample_timing import sample_time as acquisition_time
from .stage_timing import StageTimer

# Readiness = the stream keeps delivering new samples (receive gap) whose
# acquisition is not excessively old. Measuring the gap from acquisition time
# double-counts the scan duration, FAST-LIO processing and the 10 Hz period
# (0.1-0.48 s in normal operation) and toggled readiness on every jitter.
STREAM_GAP_S = .5
MAX_ACQUISITION_AGE_S = 1.
# Nearest-sample fallback when exact interpolation of the 50 Hz cart geometry
# is unavailable. The grasped cart's pose relative to the pelvis changes slowly.
CART_PAIRING_FALLBACK_S = .1
# Forward localization scans to AMCL at ~10 Hz (obstacle scans follow the LiDAR rate).
LOCALIZATION_SCAN_PERIOD_S = .095


def stamp_age_ms(node, message):
    """ROS-clock age of a host-aligned measurement stamp, for TIMING lines."""
    stamp = message.header.stamp.sec*1000000000+message.header.stamp.nanosec
    return (node.get_clock().now().nanoseconds-stamp)*1e-6


def live(now, received):
    return 0 <= now-received <= STREAM_GAP_S


class PosePending(ValueError):
    """A clock/pose sample can still arrive without depending on FAST-LIO."""


def cloud_arrays(message):
    endian = '>' if message.is_bigendian else '<'
    fields = {field.name: field for field in message.fields}
    formats, offsets = [], []
    for name in ('x', 'y', 'z'):
        field = fields[name]
        if field.datatype not in (7, 8) or field.count != 1:
            raise ValueError('Expected float32/float64 XYZ fields')
        formats.append(endian + ('f4' if field.datatype == 7 else 'f8'))
        offsets.append(field.offset)
    dtype = np.dtype(dict(names=['x', 'y', 'z'], formats=formats, offsets=offsets, itemsize=message.point_step))
    shape, strides = (message.height, message.width), (message.row_step, message.point_step)
    cloud = np.ndarray(shape, dtype=dtype, buffer=message.data, strides=strides)
    xyz = np.column_stack([cloud[name].ravel() for name in ('x', 'y', 'z')])
    records = np.ndarray(shape, dtype='V'+str(message.point_step), buffer=message.data, strides=strides).ravel()
    return xyz, records


def filtered_cloud(message, records, keep):
    output = copy.copy(message)
    output.height, output.width = 1, int(keep.sum())
    output.row_step = output.width*output.point_step
    # Foxy's bytes setter validates every byte in Python. At lidar rates the
    # two filtering passes can then starve this node's odometry/status callbacks.
    # Use PointCloud2's native uint8 buffer, preserving each retained record.
    output.data = array('B', records[keep].tobytes())
    return output


class CartNavigation(Node):
    def __init__(self):
        super().__init__('cart_navigation')
        self.declare_parameter('config_file', str(Path.cwd() / 'config/pushcart_navigation.json'))
        self.declare_parameter('geometry_port', 39006)
        self.declare_parameter('raw_cloud_topic', '/utlidar/cloud_livox_mid360')
        self.declare_parameter('raw_cloud_frame', 'livox_frame')
        self.settings = load_settings(self.get_parameter('config_file').value)
        self.geometry = CartGeometry(self.settings)
        self.gantry = GantryGeometry(self.settings.get('gantry_filter', {}))
        self.last_gantry_filtered = -math.inf
        self.gantry_removed_points = 0
        self.gantry_error = ''
        self.gantry_clear_active = False
        self.last_gantry_clear = -math.inf
        self.raw_frame = self.get_parameter('raw_cloud_frame').value
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(('127.0.0.1', self.get_parameter('geometry_port').value))
        self.sock.setblocking(False)
        self.state = {}
        self.cart = {}
        self.cart_history = CartPoseHistory()
        self.pending_odometry = deque(maxlen=10)
        self.odometry_error = ''
        self.received = self.last_base = self.last_filtered = self.last_scan = -math.inf
        self.last_raw = self.last_localization_scan = -math.inf
        # Local receive times of the newest accepted samples (readiness).
        self.base_received = self.filtered_received = self.scan_received = -math.inf
        self.gantry_received = -math.inf
        self.localizing_without_cart = False
        self.previous = None
        self.odom_clock_stamped = None
        self.navigation_odometry = {}
        self.velocity = np.zeros(3)
        self.epoch = 0
        self.odom_epoch = None
        self.removed_points = 0
        self.filter_error = ''
        self.scan_error = ''
        self.mount = None
        # Retain the full one-second time window even with an older producer
        # publishing >1 kHz. 256 samples covered only ~240 ms, less than a
        # 100 ms scan plus the allowed 200 ms wait for camera geometry.
        self.mount_history = CartPoseHistory(max_gap=.05, max_samples=None)
        self.lidar_clock = None
        self.pending_clouds = deque()
        # The producer caps this stream at 200 Hz. Queue enough real samples
        # to bridge a cloud callback without dropping the interpolation pairs.
        # Receive pose/clock history while a cloud is being filtered. A deep
        # queue alone cannot help if filtering starves the receive callbacks.
        self.pose_node = Node('navigation_pose_receiver', context=self.context,
                              use_global_arguments=False)
        self.pose_node.create_subscription(TransformStamped, '/g1/lidar_mount', self._mount, 100)
        self.broadcaster = TransformBroadcaster(self)
        sensor = QoSProfile(depth=3, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.pose_node.create_subscription(TimeReference, '/g1/lidar_clock', self._lidar_clock, sensor)
        self.gantry_clear_pub = self.create_publisher(LaserScan, '/navigation/rear_clear', sensor)
        self.gantry_clear_scan = None
        if self.gantry.enabled:
            center, angles, distances = self.gantry.clearing_rays()
            transform = TransformStamped()
            transform.header.frame_id = self.gantry.frame
            transform.child_frame_id = 'gantry_exclusion'
            transform.header.stamp = self.get_clock().now().to_msg()
            transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z = map(float, center)
            transform.transform.rotation.w = 1.
            self.gantry_broadcaster = StaticTransformBroadcaster(self)
            self.gantry_broadcaster.sendTransform(transform)
            self.gantry_clear_scan = LaserScan(
                angle_min=float(angles[0]), angle_max=float(angles[-1]),
                angle_increment=float(angles[1]-angles[0]), range_min=0.,
                range_max=float(distances.max()+.1), ranges=distances.astype(np.float32).tolist())
            self.gantry_clear_scan.header.frame_id = transform.child_frame_id
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.cloud_pub = self.create_publisher(PointCloud2, '/navigation/lidar', sensor)
        # Match Nav2's reliable odometry subscribers; reliable also serves
        # sensor-QoS readers without losing the measured velocity feedback.
        self.odom_pub = self.create_publisher(Odometry, '/Odometry_navigation', 10)
        self.status_pub = self.create_publisher(String, '/g1/navigation_reference', latched)
        self.epoch_pub = self.create_publisher(UInt32, '/g1/navigation_epoch', latched)
        self.ready_pub = self.create_publisher(Bool, '/g1/navigation_reference_ready', latched)
        self.footprint_pubs = [self.create_publisher(Polygon, '/'+name+'/footprint', latched)
                               for name in ('local_costmap', 'global_costmap')]
        self.scan_pubs = {}
        self.create_subscription(PointCloud2, self.get_parameter('raw_cloud_topic').value, self._cloud, sensor)
        self.create_subscription(Odometry, '/Odometry_base', self._receive_odom, sensor)
        for topic in ('scan', 'scan_localization'):
            self.scan_pubs[topic] = self.create_publisher(LaserScan, '/'+topic, sensor)
            self.create_subscription(LaserScan, '/'+topic+'_unmasked',
                                     lambda msg, name=topic: self._scan(msg, name), sensor)
        self.timer = self.create_timer(.02, self._tick)
        # Measurement only: callback wall times, logged as TIMING lines.
        self.stage_timer = StageTimer('cart_navigation')
        self.create_timer(2., self._log_timing)

    def _log_timing(self):
        line = self.stage_timer.report()
        if line:
            self.get_logger().info(line)

    def _fresh(self):
        now = time.monotonic()
        if not 0 <= now-self.received <= .3:
            return False
        if self.cart.get('attached'):
            stamp = self.cart.get('captured_monotonic', -math.inf)
            return self.cart.get('valid', False) and 0 <= now-stamp <= .3
        return True

    def _mount(self, message):
        if message.header.frame_id == 'base_link' and message.child_frame_id == self.raw_frame:
            try:
                q, t = message.transform.rotation, message.transform.translation
                rotation = quaternion_matrix([q.w, q.x, q.y, q.z])
                if not np.isfinite([t.x, t.y, t.z]).all():
                    return
                age = (self.get_clock().now()-Time.from_msg(message.header.stamp)).nanoseconds*1e-9
                if not -.01 <= age <= .5:
                    return
                self.mount_history.add(dict(valid=True, epoch=0, pose_source='mount', yaw=0.,
                    position=[t.x, t.y, t.z], body_rotation=rotation.tolist(),
                    sample_monotonic=time.monotonic()-age))
                self.mount = message
            except (ValueError, TypeError):
                return

    def _lidar_clock(self, message):
        if message.source != 'lidar_imu_clock':
            return
        now = time.monotonic()
        age = (self.get_clock().now()-Time.from_msg(message.header.stamp)).nanoseconds*1e-9
        if not -.03 <= age <= .3:
            return
        sensor = message.time_ref.sec+message.time_ref.nanosec*1e-9
        if (self.lidar_clock is not None and sensor <= getattr(self, '_clock_sensor', -math.inf)
                and now-self.lidar_clock[1] <= .3):
            return
        self._clock_sensor = sensor
        self.lidar_clock = (now-age-sensor, now-age)

    def _gantry_ready(self):
        now = time.monotonic()
        return bool(not self.gantry.enabled or (live(now, self.gantry_received)
                                               and live(now, self.scan_received)))

    def _tick(self):
        with self.stage_timer.measure('tick'):
            self._tick_body()

    def _tick_body(self):
        now = time.monotonic()
        for _ in range(64):
            try:
                data, sender = self.sock.recvfrom(65535)
            except BlockingIOError:
                break
            try:
                state = json.loads(data)
                stamp, cart = state['sent_monotonic'], state['cart']
                epoch = cart['epoch']
                if (sender[0] != '127.0.0.1' or state.get('magic') != 'G1LOCO1'
                        or not 0 <= time.monotonic()-stamp <= .3 or stamp <= self.received
                        or type(epoch) is not int or not 0 < epoch <= 0xffffffff
                        or (epoch & 1) != int(bool(cart['attached']))):
                    continue
                if epoch != self.epoch:
                    self.previous = None
                    self.velocity.fill(0)
                    self.last_filtered = self.last_scan = -math.inf
                    self.filtered_received = self.scan_received = -math.inf
                self.state, self.cart, self.received, self.epoch = state, cart, stamp, epoch
                self.cart_history.add(cart)
            except (ValueError, TypeError, KeyError):
                continue
        with self.stage_timer.measure('tick_pending_clouds'):
            self._publish_pending_clouds()
        with self.stage_timer.measure('tick_pending_odometry'):
            self._publish_pending_odometry()
        now = time.monotonic()
        attached = bool(self.cart.get('attached'))
        fresh = self._fresh()
        if self.gantry_clear_active and (not fresh or not self._gantry_ready()
                                        or now-self.last_gantry_clear > .2):
            self._expire_gantry_clear()
        footprint = self.geometry.footprint(self.cart, attached) if fresh else None
        ready = fresh and live(now, self.base_received) and self.odom_epoch == self.epoch
        if attached:
            ready = ready and live(now, self.filtered_received) and live(now, self.scan_received)
        if not fresh:
            reason = ((self.cart.get('reason') or '小车航向或运控反馈已过期') if attached else '等待运控反馈')
        elif not ready:
            reason = self.odometry_error or self.filter_error or self.scan_error or '等待导航里程计和车体过滤后的扫描'
        else:
            reason = ''
        if footprint is not None:
            # Foxy costmaps consume an unstamped Polygon in robot_base_frame.
            message = Polygon(points=[Point32(x=float(x), y=float(y), z=0.) for x, y in footprint])
            for publisher in self.footprint_pubs:
                publisher.publish(message)
        self.epoch_pub.publish(UInt32(data=self.epoch))
        self.ready_pub.publish(Bool(data=bool(ready)))
        def age(stamp):
            return round(now-stamp, 3) if math.isfinite(stamp) else None
        try:
            status = json.dumps(dict(
                ready=bool(ready), epoch=self.epoch, mode='cart' if attached else 'walk',
                reason=reason, footprint=footprint, cart=self.cart, cart_model=self.geometry.top_view,
                odometry=self.navigation_odometry,
                removed_points=self.removed_points,
                gantry_filter=dict(enabled=self.gantry.enabled, frame=self.gantry.frame,
                                   ready=self._gantry_ready(), removed_points=self.gantry_removed_points,
                                   reason=self.gantry_error,
                                   boxes=[dict(min_m=low.tolist(), max_m=high.tolist())
                                          for low, high in self.gantry.boxes]),
                observations=dict(
                    raw_age_s=age(self.last_raw),
                    filtered_age_s=age(self.last_filtered),
                    obstacle_scan_age_s=age(self.last_scan),
                    localization_scan_age_s=age(self.last_localization_scan),
                    base_gap_s=age(self.base_received),
                    filtered_gap_s=age(self.filtered_received),
                    obstacle_scan_gap_s=age(self.scan_received),
                    localization_without_cart_filter=self.localizing_without_cart),
                sent_monotonic=now), ensure_ascii=False,
                default=lambda value: value.tolist() if isinstance(value, np.ndarray) else value.item())
        except (TypeError, ValueError, AttributeError) as exc:
            # A diagnostic field must never take down the filter and odometry
            # chain. The safety signals (ready/epoch) were already published.
            status = json.dumps(dict(ready=False, epoch=self.epoch, mode='cart' if attached else 'walk',
                                     reason='导航参考状态序列化失败：' + str(exc)[:120],
                                     sent_monotonic=now), ensure_ascii=False)
        self.status_pub.publish(String(data=status))

    def _receive_odom(self, message):
        # /Odometry_base keeps the host-aligned scan-END time: FAST-LIO + base kinematics.
        self.stage_timer.record('latency_odom_in', stamp_age_ms(self, message))
        self.pending_odometry.append((time.monotonic(), self.epoch, message))
        self._publish_pending_odometry()

    def _publish_pending_odometry(self):
        while self.pending_odometry:
            received, epoch, message = self.pending_odometry[0]
            now = time.monotonic()
            if epoch != self.epoch or now-received > STREAM_GAP_S:
                self.pending_odometry.popleft()
                continue
            cart = self.cart
            if cart.get('attached'):
                stamp = message.header.stamp.sec+message.header.stamp.nanosec*1e-9
                sample_monotonic = now + stamp-self.get_clock().now().nanoseconds*1e-9
                if not 0 <= now-sample_monotonic <= MAX_ACQUISITION_AGE_S:
                    self.odometry_error = '导航里程计测量时刻已过期'
                    self.pending_odometry.popleft()
                    continue
                cart = self.cart_history.at(sample_monotonic)
                if cart is None and self.cart_history.can_wait(sample_monotonic) and now-received < .2:
                    # Camera/arm geometry commonly trails LiDAR odometry.
                    self.odometry_error = '等待里程计测量时刻的小车位姿'
                    break
                if cart is None:
                    # The hand-held cart moves slowly relative to the pelvis.
                    # Use the nearest real geometry sample rather than stalling
                    # the whole navigation odometry on one missing pairing.
                    cart = self.cart_history.nearest(sample_monotonic, CART_PAIRING_FALLBACK_S)
                if cart is None:
                    self.odometry_error = '缺少里程计测量时刻附近的小车位姿'
                    self.pending_odometry.popleft()
                    continue
            self.pending_odometry.popleft()
            # Including the wait for cart geometry at the scan time (pairing).
            self.stage_timer.record('latency_odom_out', stamp_age_ms(self, message))
            self.stage_timer.record('pairing_wait', (now-received)*1000)
            self._odom(message, cart)

    def _odom(self, message, cart=None):
        with self.stage_timer.measure('odometry'):
            self._odom_body(message, cart)

    def _odom_body(self, message, cart=None):
        sampled = acquisition_time(self, message, MAX_ACQUISITION_AGE_S)
        if sampled is None:
            self.odometry_error = '导航里程计测量时刻已过期或无效'
            return
        cart = self.cart if cart is None else cart
        attached = bool(self.cart.get('attached'))
        if attached and not self._fresh():
            return
        try:
            q = message.pose.pose.orientation
            rotation = quaternion_matrix([q.w, q.x, q.y, q.z])
            point = message.pose.pose.position
            if not np.isfinite([point.x, point.y, point.z]).all():
                return
            yaw = math.atan2(rotation[1, 0], rotation[0, 0])
            offset = np.asarray(cart['position']) if attached else np.zeros(3)
            relative_yaw = cart['yaw'] if attached else 0.
            position = np.array([point.x, point.y, point.z]) + rotation @ offset
            heading = yaw + relative_yaw
            now = time.monotonic()
            pose = np.array([position[0], position[1], heading])
            # The base and relative geometry now describe the same acquisition
            # time. Differentiate in the world frame, then rotate the filtered
            # velocity into the current cart frame for the controller.
            stamped = bool(message.header.stamp.sec or message.header.stamp.nanosec)
            sample_time = (message.header.stamp.sec+message.header.stamp.nanosec*1e-9
                           if stamped else now)
            if getattr(self, 'odom_clock_stamped', stamped) != stamped:
                self.previous = None
                self.velocity.fill(0)
            self.odom_clock_stamped = stamped
            if self.previous is not None:
                old_time, old_pose = self.previous
                dt = sample_time-old_time
                if stamped and dt <= 0:
                    return  # Duplicate/out-of-order samples cannot create motion.
                if .005 <= dt <= .5:
                    delta = pose-old_pose
                    delta[2] = math.atan2(math.sin(delta[2]), math.cos(delta[2]))
                    velocity = delta/dt
                    if np.linalg.norm(velocity[:2]) <= 2. and abs(velocity[2]) <= 3.:
                        # Preserve the time constant for both the vendor's
                        # 10 Hz LiDAR and direct Livox's 50 Hz odometry.
                        alpha = 1. - .65**(dt/.05)
                        self.velocity += alpha*(velocity-self.velocity)
                    else:
                        self.velocity.fill(0)
                    self.previous = sample_time, pose
                elif dt > .5 or dt < 0:
                    self.velocity.fill(0)
                    self.previous = sample_time, pose
            else:
                self.previous = sample_time, pose
            output = copy.deepcopy(message)
            output.child_frame_id = 'nav_control'
            output.pose.pose.position.x, output.pose.pose.position.y, output.pose.pose.position.z = map(float, position)
            output.pose.pose.orientation.x = output.pose.pose.orientation.y = 0.
            output.pose.pose.orientation.w = math.cos(heading/2)
            output.pose.pose.orientation.z = math.sin(heading/2)
            # Walk mode retains the existing filtered robot odometry exactly.
            if attached:
                body_velocity = rotation_z(-heading)[:2, :2] @ self.velocity[:2]
                output.twist.twist.linear.x, output.twist.twist.linear.y = map(float, body_velocity)
                output.twist.twist.angular.z = float(self.velocity[2])
            self.odom_pub.publish(output)
            self.navigation_odometry = dict(
                topic='/Odometry_navigation', frame=output.child_frame_id,
                vx_m_s=output.twist.twist.linear.x, vy_m_s=output.twist.twist.linear.y,
                wz_rad_s=output.twist.twist.angular.z,
                sample_stamp_s=sample_time if stamped else None, received_monotonic=now,
                pose_odom=pose.tolist(), base_yaw_rad=yaw, relative_yaw_rad=relative_yaw,
                cart_sample_monotonic=cart.get('sample_monotonic'),
                pose_source=cart.get('pose_source'))
            frames = [('nav_control', offset, relative_yaw)]
            if self._fresh() and self.cart.get('valid'):
                frames.append(('cart_nav', cart['position'], cart['yaw']))
            for name, xyz, angle in frames:
                transform = TransformStamped()
                transform.header = copy.deepcopy(message.header)
                transform.header.frame_id = 'base_nav'
                transform.child_frame_id = name
                transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z = map(float, xyz)
                transform.transform.rotation.w = math.cos(angle/2)
                transform.transform.rotation.z = math.sin(angle/2)
                self.broadcaster.sendTransform(transform)
            self.last_base, self.odom_epoch = sampled, self.epoch
            self.base_received = time.monotonic()
            self.odometry_error = ''
        except (ValueError, KeyError, TypeError):
            return

    def _cloud(self, message):
        received = time.monotonic()
        if self.lidar_clock is not None and 0 <= received-self.lidar_clock[1] <= .3:
            sampled = message.header.stamp.sec+message.header.stamp.nanosec*1e-9+self.lidar_clock[0]
            # Latency from the frame START (header) to arrival: frame span + driver + adapter.
            self.stage_timer.record('latency_raw_cloud', (received-sampled)*1000)
            if not -.03 <= received-sampled <= MAX_ACQUISITION_AGE_S:
                # Diagnose only. Dropping raw returns here would also starve
                # FAST-LIO; filtering below decides what it can still use.
                self.filter_error = '雷达采样时刻已过期或超前'
            else:
                self.last_raw = sampled
        self.pending_clouds.append((received, self.epoch, message))
        with self.stage_timer.measure('cloud_callback'):
            self._publish_pending_clouds()

    def _publish_pending_clouds(self):
        while self.pending_clouds:
            received, epoch, message = self.pending_clouds[0]
            if epoch != self.epoch:
                if self.cart.get('attached'):
                    self._localization_only(message)
                else:
                    self.cloud_pub.publish(message)
                self.pending_clouds.popleft()
                continue
            # Camera feedback can arrive after a raw scan. Wait briefly for
            # history, but always release raw returns on loss so LIO can recover.
            wait = time.monotonic()-received < .2 and len(self.pending_clouds) < 16
            if not self._filter_cloud(message, wait=wait):
                break
            # Queue time of this frame (cart geometry wait plus filtering).
            self.stage_timer.record('cloud_wait', (time.monotonic()-received)*1000)
            self.pending_clouds.popleft()

    def _filter_cloud(self, message, wait=False):
        with self.stage_timer.measure('cloud_filter'):
            return self._filter_cloud_body(message, wait)

    def _filter_cloud_body(self, message, wait=False):
        attached, fresh = self.cart.get('attached'), self._fresh()
        points = records = groups = None
        if wait and attached and fresh:
            # Camera geometry commonly trails the scan. Check it before
            # repeatedly transforming thousands of points on each retry.
            try:
                _, _, groups = self._cloud_samples(message)
                for _, stamp in groups:
                    if not self.cart_history.can_at(stamp):
                        if self.cart_history.can_wait(stamp):
                            return False
                        wait = False
                        break
            except PosePending:
                return False
            except (ValueError, KeyError, TypeError):
                pass  # The regular filtering path reports malformed clouds.
        if self.gantry.enabled or (attached and fresh):
            try:
                points, records, groups = self._points_in_body(message)
            except Exception as exc:
                if wait and isinstance(exc, PosePending):
                    return False
                if self.gantry.enabled:
                    self.gantry_error = '后方区域清空暂不可用：' + str(exc)
                    self.last_gantry_filtered = self.gantry_received = -math.inf
                    self._expire_gantry_clear()
                if attached:
                    self.filter_error = '车体点云过滤等待有效坐标：' + str(exc)
        cart_keep = None
        if attached and fresh and points is not None:
            try:
                cart_points = np.empty_like(points)
                for indices, stamp in groups:
                    cart = self.cart_history.at(stamp)
                    if cart is None and wait and self.cart_history.can_wait(stamp):
                        raise PosePending('等待雷达采样时刻的小车位姿')
                    if cart is None:
                        cart = self.cart_history.nearest(stamp, CART_PAIRING_FALLBACK_S)
                    if cart is None:
                        raise ValueError('缺少雷达采样时刻附近的小车位姿')
                    leveled = points[indices] @ np.asarray(cart['body_rotation']).T
                    rotation = np.asarray(cart['rotation']) if 'rotation' in cart else rotation_z(cart['yaw'])
                    cart_points[indices] = (leveled-cart['position']) @ rotation
                cart_keep = ~self.geometry.cart_mask(cart_points)
            except Exception as exc:
                if wait and isinstance(exc, PosePending):
                    return False
                cart_keep = None
                self.filter_error = '车体点云过滤等待有效坐标：' + str(exc)
        if points is not None:
            keep = np.ones(len(points), dtype=bool)
            newest = max((stamp for _, stamp in groups), default=-math.inf)
            if self.gantry.enabled:
                keep &= ~self.gantry.self_mask(points)
                self.gantry_removed_points = int((~keep).sum())
                self.last_gantry_filtered, self.gantry_error = newest, ''
                self.gantry_received = time.monotonic()
            if cart_keep is not None:
                self.removed_points = int((keep & ~cart_keep).sum())
                keep &= cart_keep
            message = filtered_cloud(message, records, keep)
        if not attached or cart_keep is not None:
            if attached:
                self.last_filtered, self.filter_error = newest, ''
                self.filtered_received = time.monotonic()
                # Scan END to FAST-LIO input, including any wait for cart geometry.
                self.stage_timer.record('latency_filtered_cloud', (self.filtered_received-newest)*1000)
            self.localizing_without_cart = False
            self.cloud_pub.publish(message)
        else:
            self._localization_only(message)
        return True

    def _cloud_samples(self, message):
        if message.header.frame_id != self.raw_frame:
            raise ValueError('雷达点云坐标系不匹配')
        clock = self.lidar_clock
        now = time.monotonic()
        if clock is None or not 0 <= now-clock[1] <= .3:
            raise PosePending('等待雷达采样时钟')
        cached = getattr(self, '_prepared_cloud', None)
        if cached is None or cached[0] is not message:
            xyz, records = cloud_arrays(message)
            field = next((f for f in message.fields if f.name == 'time'), None)
            if field is None or field.datatype not in (7, 8) or field.count != 1:
                raise ValueError('缺少雷达逐点采样时间')
            dtype = np.dtype(dict(names=['time'], formats=[('>' if message.is_bigendian else '<')+
                ('f4' if field.datatype == 7 else 'f8')], offsets=[field.offset], itemsize=message.point_step))
            offsets = np.ndarray((message.height, message.width), dtype=dtype, buffer=message.data,
                strides=(message.row_step, message.point_step))['time'].ravel().astype(float)*1e-9
            if not np.isfinite(offsets).all() or np.any(offsets < 0.) or np.any(offsets > .2):
                raise ValueError('雷达逐点采样时间超出有效范围')
            bins = np.floor(offsets/.005).astype(np.int32)
            groups = []
            for value in np.unique(bins):
                indices = np.flatnonzero(bins == value)
                # Timestamps cross into watchdogs and JSON status. Keep them
                # native scalars so an expired comparison cannot yield np.bool_.
                groups.append((indices, float((offsets[indices].min()+offsets[indices].max())*.5)))
            cached = (message, xyz, records, offsets, groups)
            self._prepared_cloud = cached
        _, xyz, records, offsets, relative_groups = cached
        start = message.header.stamp.sec+message.header.stamp.nanosec*1e-9+clock[0]
        if len(offsets) and (now-start-offsets.min() > MAX_ACQUISITION_AGE_S or start+offsets.max()-now > .03):
            raise ValueError('雷达采样时刻已过期或超前')
        return xyz, records, [(indices, start+offset) for indices, offset in relative_groups]

    def _points_in_body(self, message):
        xyz, records, groups = self._cloud_samples(message)
        mounts, points = [], np.empty_like(xyz)
        for _, stamp in groups:
            mount = self.mount_history.at(stamp)
            if mount is None and self.mount_history.can_wait(stamp):
                raise PosePending('等待雷达采样时刻的安装位姿')
            if mount is None:
                # Waist joints move slowly; a short feedback hole must not
                # discard the whole scan from filtering and avoidance.
                mount = self.mount_history.nearest(stamp, CART_PAIRING_FALLBACK_S)
            if mount is None:
                raise ValueError('缺少雷达采样时刻附近的安装位姿')
            mounts.append(mount)
        # Check the entire scan's history before doing point transforms. A
        # pending tail must not repeatedly transform all earlier points.
        for (indices, _), mount in zip(groups, mounts):
            points[indices] = xyz[indices] @ np.asarray(mount['body_rotation']).T + mount['position']
        return points, records, groups

    def _expire_gantry_clear(self):
        if not self.gantry_clear_active:
            return
        # Foxy retains the last observation even when the publisher falls
        # silent. Replace it explicitly with a cloud containing no rays.
        empty = LaserScan(angle_increment=1., range_min=0., range_max=4., ranges=[float('nan')])
        empty.header.frame_id = self.gantry.frame
        empty.header.stamp = self.get_clock().now().to_msg()
        self.gantry_clear_pub.publish(empty)
        self.gantry_clear_active = False

    def _localization_only(self, message):
        # Camera/arm tracking must not starve FAST-LIO of lidar measurements.
        # With unknown cart geometry retain the remaining returns for localization;
        # do not guess a mask or publish an empty/free-space obstacle scan.
        # This frame never refreshes obstacle readiness; the last valid
        # observation has its own freshness deadline.
        self.localizing_without_cart = True
        # This frame cannot supply obstacle observations. Keep the timestamp
        # of the last successful frame; _tick expires it after the existing
        # 500 ms limit. One lost frame must not invalidate still-fresh scans.
        self.cloud_pub.publish(message)

    def _scan(self, message, topic):
        with self.stage_timer.measure(topic + '_mask'):
            self._scan_body(message, topic)

    def _scan_body(self, message, topic):
        sampled = acquisition_time(self, message, MAX_ACQUISITION_AGE_S, stream=topic)
        if sampled is None:
            self.scan_error = '扫描测量时刻已过期、重复或无效'
            return
        localization = topic == 'scan_localization'
        if localization and sampled-self.last_localization_scan < LOCALIZATION_SCAN_PERIOD_S:
            # AMCL (5000 particles x 360 beams) cannot keep up with 50 Hz
            # direct-Livox scans; it fell 0.3-0.7 s behind and starved the
            # navigation cores. Each forwarded scan already merges 100 ms.
            return
        if self.cart.get('attached'):
            cart = None
            # An obstacle scan from an unfiltered raw frame may still contain
            # cart-body returns; localization scans keep all real returns.
            if self._fresh() and (localization or not self.localizing_without_cart):
                # Each scan masks the cart with the geometry at its own
                # acquisition time; nearest real sample if not interpolable.
                cart = (self.cart_history.at(sampled)
                        or self.cart_history.nearest(sampled, CART_PAIRING_FALLBACK_S))
            if cart is None:
                self.scan_error = '缺少扫描测量时刻附近的小车位姿'
                if not localization:
                    return
            else:
                angles = message.angle_min + np.arange(len(message.ranges))*message.angle_increment
                ranges = np.asarray(message.ranges, dtype=float).copy()
                # base_nav and the scan describe the acquisition time. Using
                # current relative geometry can erase external hits on one side
                # and leave cart returns on the other while the pelvis sways.
                ranges[self.geometry.occluded_ranges(angles, ranges, cart)] = np.nan
                message = copy.copy(message)
                message.ranges = ranges.astype(np.float32).tolist()
                self.scan_error = ''
        else:
            self.scan_error = ''
        self.scan_pubs[topic].publish(message)
        if localization:
            self.last_localization_scan = sampled
        else:
            self.last_scan = sampled
            self.scan_received = time.monotonic()
        if not localization and self.gantry.enabled and self._gantry_ready() and self._fresh():
            # This is an explicit ignore region, not a physical free-space
            # measurement. Stock Nav2 uses it for clearing only. The origin
            # is inside the rear rectangle so rays never cross its exterior.
            clear_scan = copy.deepcopy(self.gantry_clear_scan)
            clear_scan.header.stamp = copy.copy(message.header.stamp)
            self.gantry_clear_pub.publish(clear_scan)
            self.gantry_clear_active = True
            self.last_gantry_clear = time.monotonic()
        elif not localization:
            self._expire_gantry_clear()

    def destroy_node(self):
        self.sock.close()
        self.pose_node.destroy_node()
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CartNavigation()
    pose_executor = SingleThreadedExecutor(context=node.context)
    pose_executor.add_node(node.pose_node)
    pose_thread = threading.Thread(target=pose_executor.spin, name='navigation-pose', daemon=True)
    pose_thread.start()
    try:
        rclpy.spin(node)
    finally:
        pose_executor.shutdown()
        pose_thread.join()
        node.destroy_node()
        rclpy.shutdown()
