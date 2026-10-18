"""Exercise the installed AMCL with lidar/TF only, on an isolated ROS domain."""
import math
import os
from pathlib import Path
import subprocess
import tempfile
import threading
import time
from types import SimpleNamespace

import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped, TransformStamped
from lifecycle_msgs.srv import ChangeState
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.context import Context
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import DurabilityPolicy, QoSProfile
from sensor_msgs.msg import LaserScan
from std_srvs.srv import Empty
from tf2_ros import TransformBroadcaster

from unitree_nav2_web.nav_web_bridge import NavigationWebBridge


def test_real_amcl_sway_needs_scan_refresh_and_missing_scan_cannot_fake_a_pose(monkeypatch):
    monkeypatch.setenv('ROS_DOMAIN_ID', '89')
    monkeypatch.setenv('ROS_LOCALHOST_ONLY', '1')
    monkeypatch.setenv('RMW_IMPLEMENTATION', 'rmw_cyclonedds_cpp')
    monkeypatch.setenv('CYCLONEDDS_URI',
        '<CycloneDDS><Domain><General><NetworkInterfaceAddress>lo</NetworkInterfaceAddress>'
        '<AllowMulticast>false</AllowMulticast></General><Discovery><ParticipantIndex>auto</ParticipantIndex>'
        '<Peers><Peer Address="127.0.0.1"/></Peers></Discovery></Domain></CycloneDDS>')
    context = Context()
    node = executor = process = None
    with tempfile.TemporaryDirectory(prefix='amcl_refresh_') as directory:
        monkeypatch.setenv('ROS_LOG_DIR', directory)
        with open(Path(directory) / 'amcl.log', 'w+') as log:
            try:
                rclpy.init(context=context)
                node = rclpy.create_node('amcl_refresh_test', context=context)
                executor = SingleThreadedExecutor(context=context)
                executor.add_node(node)
                tf = TransformBroadcaster(node)
                latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
                map_pub = node.create_publisher(OccupancyGrid, '/map', latched)
                scan_pub = node.create_publisher(LaserScan, '/scan_localization', 10)
                initial_pub = node.create_publisher(PoseWithCovarianceStamped, '/initialpose', 10)
                client = node.create_client(ChangeState, '/amcl/change_state')
                poses, faults = [], []
                bridge = SimpleNamespace(
                    lock=threading.RLock(), phase='NAVIGATING', localization_refinement_active=False,
                    navigation_active=True, navigation_pose_received=False, navigation_motion_started=False,
                    navigation_last_motion_time=0., nomotion_update_inflight=False,
                    next_nomotion_update_time=0., amcl_generation=0, last_pose_time=0.,
                    last_cloud_time=0., last_odom_time=0., last_localization_scan_time=0.,
                    nomotion_update_client=node.create_client(Empty, '/request_nomotion_update'),
                    _handle_localization_fault=faults.append)

                def received_pose(message):
                    bridge.last_pose_time = time.monotonic()
                    bridge.navigation_pose_received = True
                    poses.append((bridge.last_pose_time, message))

                node.create_subscription(PoseWithCovarianceStamped, '/amcl_pose', received_pose, 10)
                params = Path(__file__).resolve().parents[1] / 'config/nav2_foxy_g1.yaml'
                process = subprocess.Popen([
                    '/opt/ros/foxy/lib/nav2_amcl/amcl', '--ros-args', '--params-file', str(params)],
                    stdout=log, stderr=subprocess.STDOUT, env=os.environ.copy())
                last_publish, origin = 0., time.monotonic()
                refresh = publish_scan = False

                def diagnostic():
                    log.flush()
                    log.seek(0)
                    return log.read()

                def tick():
                    nonlocal last_publish
                    now = time.monotonic()
                    if now-last_publish >= .08:
                        last_publish = now
                        # 4 mm sway gives > .03 m/s, but never reaches AMCL's
                        # 2 cm displacement threshold in either direction.
                        x = .004*math.sin(8*math.pi*(now-origin))
                        stamp = node.get_clock().now().to_msg()
                        transform = TransformStamped()
                        transform.header.frame_id, transform.child_frame_id = 'odom', 'base_nav'
                        transform.header.stamp = stamp
                        transform.transform.translation.x = x
                        transform.transform.rotation.w = 1.
                        tf.sendTransform(transform)
                        odom = Odometry()
                        odom.pose.pose.orientation.w = 1.
                        odom.pose.pose.position.x = x
                        odom.twist.twist.linear.x = .004*8*math.pi*math.cos(8*math.pi*(now-origin))
                        NavigationWebBridge._odom_callback(bridge, odom)
                        bridge.last_cloud_time = now
                        if publish_scan:
                            scan = LaserScan(angle_min=-math.pi, angle_max=math.pi,
                                             angle_increment=2*math.pi/720, range_min=.6, range_max=8.)
                            scan.header.frame_id, scan.header.stamp = 'base_nav', stamp
                            # An asymmetric rectangular room, in the same map as below.
                            scan.ranges = [float(min(
                                (4-x if math.cos(a) >= 0 else 3+x)/max(abs(math.cos(a)), 1e-9),
                                (3 if math.sin(a) >= 0 else 2)/max(abs(math.sin(a)), 1e-9)))
                                for a in (scan.angle_min+i*scan.angle_increment for i in range(720))]
                            scan_pub.publish(scan)
                            bridge.last_localization_scan_time = now
                    executor.spin_once(timeout_sec=.01)
                    if refresh:
                        NavigationWebBridge._nomotion_update_timer(bridge)

                def until(predicate, timeout=8.):
                    end = time.monotonic()+timeout
                    while time.monotonic() < end:
                        tick()
                        if predicate():
                            return
                    raise AssertionError(diagnostic())

                until(client.service_is_ready)
                for transition in (1, 3):
                    request = ChangeState.Request()
                    request.transition.id = transition
                    future = client.call_async(request)
                    until(future.done)
                    assert future.result().success, diagnostic()
                grid = OccupancyGrid()
                grid.header.frame_id = 'map'
                grid.info.resolution = .05
                grid.info.width = grid.info.height = 200
                grid.info.origin.position.x = grid.info.origin.position.y = -5.
                grid.info.origin.orientation.w = 1.
                grid.data = [0 if -3 < -5+(col+.5)*.05 < 4 and -2 < -5+(row+.5)*.05 < 3 else 100
                             for row in range(200) for col in range(200)]
                map_pub.publish(grid)
                until(lambda: initial_pub.get_subscription_count() > 0 and scan_pub.get_subscription_count() > 0)
                initial = PoseWithCovarianceStamped()
                initial.header.frame_id = 'map'
                initial.header.stamp = node.get_clock().now().to_msg()
                initial.pose.pose.orientation.w = 1.
                initial.pose.covariance[0] = initial.pose.covariance[7] = initial.pose.covariance[35] = .001
                initial_pub.publish(initial)
                publish_scan = True
                until(lambda: len(poses) >= 1)
                # Reproduce the old watchdog premise using the real AMCL:
                # fresh input and apparent motion, yet no new localization pose.
                until(lambda: time.monotonic()-bridge.last_pose_time > 1.7)
                assert bridge.navigation_motion_started
                assert time.monotonic()-bridge.last_localization_scan_time < .2
                count = len(poses)
                refresh = True
                until(lambda: len(poses) >= count+4)
                updates = [t for t, _ in poses[count:]]
                assert all(b-a < 1.5 for a, b in zip(updates, updates[1:])), updates
                assert not faults
                publish_scan = False
                # Drain the last input, then verify service calls alone cannot
                # renew the pose timestamp when lidar really stops.
                until(lambda: time.monotonic()-bridge.last_pose_time > 1.7)
                assert time.monotonic()-bridge.last_localization_scan_time > 1.5
            finally:
                if process is not None:
                    process.terminate()
                    try:
                        process.wait(timeout=5.)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=2.)
                if executor is not None:
                    executor.shutdown()
                if node is not None:
                    node.destroy_node()
                if context.ok():
                    rclpy.shutdown(context=context)
