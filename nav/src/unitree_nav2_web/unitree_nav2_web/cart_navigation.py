"""Navigation reference, cart self filtering and observation freshness.

The motor executor is the sole source of cart attachment/epoch and grasp pose.
FAST-LIO and AMCL retain the robot's base_nav chain. Only navigation follows
nav_control, which coincides with base_nav when walking and cart_nav when pushing.
"""
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
from sensor_msgs.msg import LaserScan, PointCloud2
from std_msgs.msg import Bool, String, UInt32
from tf2_ros import TransformBroadcaster

from .cart_geometry import CartGeometry, load_settings, quaternion_matrix, rotation_z


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


class CartNavigation(Node):
    def __init__(self):
        super().__init__('cart_navigation')
        self.declare_parameter('config_file', str(Path.cwd() / 'config/pushcart_navigation.json'))
        self.declare_parameter('geometry_port', 39006)
        self.declare_parameter('raw_cloud_topic', '/utlidar/cloud_livox_mid360')
        self.declare_parameter('raw_cloud_frame', 'livox_frame')
        self.settings = load_settings(self.get_parameter('config_file').value)
        self.geometry = CartGeometry(self.settings)
        self.raw_frame = self.get_parameter('raw_cloud_frame').value
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(('127.0.0.1', self.get_parameter('geometry_port').value))
        self.sock.setblocking(False)
        self.state = {}
        self.cart = {}
        self.received = self.last_base = self.last_filtered = self.last_scan = -math.inf
        self.previous = None
        self.velocity = np.zeros(3)
        self.epoch = 0
        self.odom_epoch = None
        self.removed_points = 0
        self.filter_error = ''
        self.mount = None
        self.create_subscription(TransformStamped, '/g1/lidar_mount', self._mount, 3)
        self.broadcaster = TransformBroadcaster(self)
        sensor = QoSProfile(depth=3, reliability=ReliabilityPolicy.BEST_EFFORT)
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
        self.create_subscription(Odometry, '/Odometry_base', self._odom, sensor)
        for topic in ('scan', 'scan_localization'):
            self.scan_pubs[topic] = self.create_publisher(LaserScan, '/'+topic, sensor)
            self.create_subscription(LaserScan, '/'+topic+'_unmasked',
                                     lambda msg, name=topic: self._scan(msg, name), sensor)
        self.timer = self.create_timer(.05, self._tick)

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
            self.mount = message

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
            except (ValueError, TypeError, KeyError):
                continue
        attached = bool(self.cart.get('attached'))
        fresh = self._fresh()
        footprint = self.geometry.footprint(self.cart, attached) if fresh else None
        ready = fresh and now-self.last_base <= .5 and self.odom_epoch == self.epoch
        if attached:
            ready = ready and now-self.last_filtered <= .5 and now-self.last_scan <= .5
        if not fresh:
            reason = '等待新鲜的小车航向和运控反馈' if attached else '等待运控反馈'
        elif not ready:
            reason = self.filter_error or '等待导航里程计和车体过滤后的扫描'
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
            reason=reason, footprint=footprint, cart=self.cart, removed_points=self.removed_points,
            sent_monotonic=now), ensure_ascii=False)))

    def _odom(self, message):
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
            offset = np.asarray(self.cart['position']) if attached else np.zeros(3)
            relative_yaw = self.cart['yaw'] if attached else 0.
            position = np.array([point.x, point.y, point.z]) + rotation @ offset
            heading = yaw + relative_yaw
            now = time.monotonic()
            pose = np.array([position[0], position[1], heading])
            if self.previous is not None:
                old_time, old_pose = self.previous
                dt = now-old_time
                if .05 <= dt <= .5:
                    delta = pose-old_pose
                    delta[2] = math.atan2(math.sin(delta[2]), math.cos(delta[2]))
                    velocity = delta/dt
                    velocity[:2] = rotation_z(-heading)[:2, :2] @ velocity[:2]
                    if np.linalg.norm(velocity[:2]) <= 2. and abs(velocity[2]) <= 3.:
                        self.velocity += .35*(velocity-self.velocity)
                    else:
                        self.velocity.fill(0)
                    self.previous = now, pose
                elif dt > .5 or dt < 0:
                    self.velocity.fill(0)
                    self.previous = now, pose
            else:
                self.previous = now, pose
            output = copy.deepcopy(message)
            output.child_frame_id = 'nav_control'
            output.pose.pose.position.x, output.pose.pose.position.y, output.pose.pose.position.z = map(float, position)
            output.pose.pose.orientation.x = output.pose.pose.orientation.y = 0.
            output.pose.pose.orientation.w = math.cos(heading/2)
            output.pose.pose.orientation.z = math.sin(heading/2)
            # Walk mode retains the existing filtered robot odometry exactly.
            if attached:
                output.twist.twist.linear.x, output.twist.twist.linear.y = map(float, self.velocity[:2])
                output.twist.twist.angular.z = float(self.velocity[2])
            self.odom_pub.publish(output)
            frames = [('nav_control', offset, relative_yaw)]
            if self._fresh() and self.cart.get('valid'):
                frames.append(('cart_nav', self.cart['position'], self.cart['yaw']))
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
        except (ValueError, KeyError, TypeError):
            return

    def _cloud(self, message):
        if not self.cart.get('attached'):
            self.cloud_pub.publish(message)
            return
        if not self._fresh():
            return  # No fresh geometry: pause observations, never publish an empty free-space scan.
        try:
            # Mount geometry is published directly from lowstate, independent
            # of FAST-LIO. Filtering can resume after camera loss without a
            # circular dependency on new filtered odometry/TF.
            transform = self.mount
            if transform is None:
                raise ValueError('等待雷达安装位姿')
            stamp = Time.from_msg(transform.header.stamp)
            if abs((self.get_clock().now()-stamp).nanoseconds)*1e-9 > .3:
                raise ValueError('雷达与机身坐标变换已过期')
            q, t = transform.transform.rotation, transform.transform.translation
            leveled = np.asarray(self.cart['body_rotation'])
            rotation = leveled @ quaternion_matrix([q.w, q.x, q.y, q.z])
            xyz, records = cloud_arrays(message)
            points = xyz @ rotation.T + leveled @ [t.x, t.y, t.z]
            keep = ~self.geometry.self_mask(points, self.cart)
            output = copy.copy(message)
            output.height, output.width = 1, int(keep.sum())
            output.row_step = output.width*output.point_step
            output.data = records[keep].tobytes()
            self.cloud_pub.publish(output)
            self.removed_points = int((~keep).sum())
            self.last_filtered, self.filter_error = time.monotonic(), ''
        except Exception as exc:
            self.filter_error = '车体点云过滤等待有效坐标：' + str(exc)

    def _scan(self, message, topic):
        if self.cart.get('attached'):
            if not self._fresh():
                return
            angles = message.angle_min + np.arange(len(message.ranges))*message.angle_increment
            ranges = np.asarray(message.ranges, dtype=float).copy()
            ranges[self.geometry.occluded_ranges(angles, ranges, self.cart)] = np.nan
            message = copy.copy(message)
            message.ranges = ranges.astype(np.float32).tolist()
        self.scan_pubs[topic].publish(message)
        if topic == 'scan':
            self.last_scan = time.monotonic()

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
