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
import time

import numpy as np
import rclpy
from geometry_msgs.msg import Point32, Polygon, TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import LaserScan, PointCloud2, TimeReference
from std_msgs.msg import Bool, String, UInt32
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster

from .cart_geometry import CartGeometry, CartPoseHistory, GantryGeometry, load_settings, quaternion_matrix, rotation_z


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
        self.mount_history = CartPoseHistory(max_gap=.05, max_samples=256)
        self.lidar_clock = None
        self.pending_clouds = deque()
        self.create_subscription(TransformStamped, '/g1/lidar_mount', self._mount, 20)
        self.broadcaster = TransformBroadcaster(self)
        sensor = QoSProfile(depth=3, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(TimeReference, '/g1/lidar_clock', self._lidar_clock, sensor)
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
        self.lidar_clock = (now-age-sensor, now)

    def _gantry_ready(self):
        now = time.monotonic()
        return not self.gantry.enabled or (now-self.last_gantry_filtered <= .5
                                           and now-self.last_scan <= .5)

    def _tick(self):
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
                        or not 0 <= now-stamp <= .3 or stamp <= self.received
                        or type(epoch) is not int or not 0 < epoch <= 0xffffffff
                        or (epoch & 1) != int(bool(cart['attached']))):
                    continue
                if epoch != self.epoch:
                    self.previous = None
                    self.velocity.fill(0)
                    self.last_filtered = self.last_scan = -math.inf
                self.state, self.cart, self.received, self.epoch = state, cart, stamp, epoch
                self.cart_history.add(cart)
            except (ValueError, TypeError, KeyError):
                continue
        self._publish_pending_clouds()
        self._publish_pending_odometry()
        attached = bool(self.cart.get('attached'))
        fresh = self._fresh()
        if self.gantry_clear_active and (not fresh or not self._gantry_ready()
                                        or now-self.last_gantry_clear > .2):
            self._expire_gantry_clear()
        footprint = self.geometry.footprint(self.cart, attached) if fresh else None
        ready = fresh and now-self.last_base <= .5 and self.odom_epoch == self.epoch
        if attached:
            ready = ready and now-self.last_filtered <= .5 and now-self.last_scan <= .5
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
        self.status_pub.publish(String(data=json.dumps(dict(
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
                raw_age_s=round(now-self.last_raw, 3) if math.isfinite(self.last_raw) else None,
                filtered_age_s=round(now-self.last_filtered, 3) if math.isfinite(self.last_filtered) else None,
                obstacle_scan_age_s=round(now-self.last_scan, 3) if math.isfinite(self.last_scan) else None,
                localization_scan_age_s=round(now-self.last_localization_scan, 3)
                if math.isfinite(self.last_localization_scan) else None,
                localization_without_cart_filter=self.localizing_without_cart),
            sent_monotonic=now), ensure_ascii=False)))

    def _receive_odom(self, message):
        self.pending_odometry.append((time.monotonic(), self.epoch, message))
        self._publish_pending_odometry()

    def _publish_pending_odometry(self):
        while self.pending_odometry:
            received, epoch, message = self.pending_odometry[0]
            now = time.monotonic()
            if epoch != self.epoch or now-received > .3:
                self.pending_odometry.popleft()
                continue
            cart = self.cart
            if cart.get('attached'):
                stamp = message.header.stamp.sec+message.header.stamp.nanosec*1e-9
                sample_monotonic = now + stamp-self.get_clock().now().nanoseconds*1e-9
                if not 0 <= now-sample_monotonic <= .5:
                    self.odometry_error = '导航里程计测量时刻已过期'
                    self.pending_odometry.popleft()
                    continue
                cart = self.cart_history.at(sample_monotonic)
                if cart is None:
                    self.odometry_error = '等待里程计测量时刻的小车位姿'
                    break
            self.pending_odometry.popleft()
            self._odom(message, cart)

    def _odom(self, message, cart=None):
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
            # velocity into the current cart frame for DWB.
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
                        # Rate-independent: the DDS LiDAR is normally 10 Hz;
                        # a 50 Hz camera does not make odometry 50 Hz.
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
            self.last_base, self.odom_epoch = now, self.epoch
            self.odometry_error = ''
        except (ValueError, KeyError, TypeError):
            return

    def _cloud(self, message):
        self.last_raw = time.monotonic()
        self.pending_clouds.append((self.last_raw, self.epoch, message))
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
            self.pending_clouds.popleft()

    def _filter_cloud(self, message, wait=False):
        attached, fresh = self.cart.get('attached'), self._fresh()
        points = records = groups = None
        if self.gantry.enabled or (attached and fresh):
            try:
                points, records, groups = self._points_in_body(message)
            except Exception as exc:
                if wait and isinstance(exc, PosePending):
                    return False
                if self.gantry.enabled:
                    self.gantry_error = '后方区域清空暂不可用：' + str(exc)
                    self.last_gantry_filtered = -math.inf
                    self._expire_gantry_clear()
                if attached:
                    self.filter_error = '车体点云过滤等待有效坐标：' + str(exc)
        cart_keep = None
        if attached and fresh and points is not None:
            try:
                cart_points = np.empty_like(points)
                for indices, stamp in groups:
                    cart = self.cart_history.at(stamp)
                    if cart is None:
                        raise PosePending('等待雷达采样时刻的小车位姿')
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
            if self.gantry.enabled:
                keep &= ~self.gantry.self_mask(points)
                self.gantry_removed_points = int((~keep).sum())
                self.last_gantry_filtered, self.gantry_error = time.monotonic(), ''
            if cart_keep is not None:
                self.removed_points = int((keep & ~cart_keep).sum())
                keep &= cart_keep
            message = filtered_cloud(message, records, keep)
        if not attached or cart_keep is not None:
            if attached:
                self.last_filtered, self.filter_error = time.monotonic(), ''
            self.localizing_without_cart = False
            self.cloud_pub.publish(message)
        else:
            self._localization_only(message)
        return True

    def _points_in_body(self, message):
        if message.header.frame_id != self.raw_frame:
            raise ValueError('雷达点云坐标系不匹配')
        xyz, records = cloud_arrays(message)
        clock = self.lidar_clock
        now = time.monotonic()
        if clock is None or not 0 <= now-clock[1] <= .3:
            raise PosePending('等待雷达采样时钟')
        field = next((f for f in message.fields if f.name == 'time'), None)
        if field is None or field.datatype not in (7, 8) or field.count != 1:
            raise ValueError('缺少雷达逐点采样时间')
        dtype = np.dtype(dict(names=['time'], formats=[('>' if message.is_bigendian else '<')+
            ('f4' if field.datatype == 7 else 'f8')], offsets=[field.offset], itemsize=message.point_step))
        offsets = np.ndarray((message.height, message.width), dtype=dtype, buffer=message.data,
            strides=(message.row_step, message.point_step))['time'].ravel().astype(float)*1e-9
        if not np.isfinite(offsets).all() or np.any(offsets < 0.) or np.any(offsets > .2):
            raise ValueError('雷达逐点采样时间超出有效范围')
        start = message.header.stamp.sec+message.header.stamp.nanosec*1e-9+clock[0]
        if len(offsets) and (now-start-offsets.min() > .5 or start+offsets.max()-now > .03):
            raise ValueError('雷达采样时刻已过期或超前')
        bins = np.floor(offsets/.005).astype(np.int32)
        groups, points = [], np.empty_like(xyz)
        for value in np.unique(bins):
            indices = np.flatnonzero(bins == value)
            stamp = start+(offsets[indices].min()+offsets[indices].max())*.5
            mount = self.mount_history.at(stamp)
            if mount is None:
                raise PosePending('等待雷达采样时刻的安装位姿')
            points[indices] = xyz[indices] @ np.asarray(mount['body_rotation']).T + mount['position']
            groups.append((indices, stamp))
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
        # The cart reference remains unready and navigation stays gated off.
        self.localizing_without_cart = True
        self.last_filtered = -math.inf
        self.cloud_pub.publish(message)

    def _scan(self, message, topic):
        localization = topic == 'scan_localization'
        if self.cart.get('attached'):
            cart = None
            if self._fresh() and not self.localizing_without_cart:
                now = time.monotonic()
                age = (self.get_clock().now().nanoseconds
                       - Time.from_msg(message.header.stamp).nanoseconds)*1e-9
                if 0. <= age <= .5:
                    cart = self.cart_history.at(now-age)
            if cart is None:
                self.scan_error = '等待扫描测量时刻的小车位姿'
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
            self.last_localization_scan = time.monotonic()
        else:
            self.last_scan = time.monotonic()
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
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CartNavigation()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
