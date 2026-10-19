"""Load the actual Foxy DWB plugins and dynamic footprint on an isolated ROS graph."""
import os
import math
import copy
from collections import deque
from pathlib import Path
import signal
import subprocess
import tempfile
import time

import numpy as np
import pytest
import rclpy
from ament_index_python.packages import get_package_prefix
from geometry_msgs.msg import Point32, Polygon, PolygonStamped, PoseStamped, TransformStamped, Twist
from lifecycle_msgs.srv import ChangeState
from nav_msgs.msg import OccupancyGrid, Odometry, Path as NavigationPath
from nav2_msgs.action import FollowPath
from nav2_msgs.srv import ClearEntireCostmap
from sensor_msgs.msg import LaserScan
from rclpy.context import Context
from rclpy.time import Time
from rclpy.executors import SingleThreadedExecutor
from rclpy.action import ActionClient
from rclpy.qos import DurabilityPolicy, QoSProfile
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster

from unitree_nav2_web.cart_geometry import CartGeometry, GantryGeometry, load_settings


@pytest.mark.parametrize('scope', ['local', 'global'])
def test_stock_nav2_footprint_and_mode_reset_preserve_external_obstacles(monkeypatch, scope):
    monkeypatch.setenv('ROS_DOMAIN_ID', '89')
    monkeypatch.setenv('ROS_LOCALHOST_ONLY', '1')
    monkeypatch.setenv('RMW_IMPLEMENTATION', 'rmw_cyclonedds_cpp')
    monkeypatch.setenv('CYCLONEDDS_URI',
        '<CycloneDDS><Domain><General><NetworkInterfaceAddress>lo</NetworkInterfaceAddress>'
        '<AllowMulticast>false</AllowMulticast></General><Discovery><ParticipantIndex>auto</ParticipantIndex>'
        '<Peers><Peer Address="127.0.0.1"/></Peers></Discovery></Domain></CycloneDDS>')
    context = Context()
    process = node = executor = None
    with tempfile.TemporaryDirectory(prefix='cart_stack_') as directory:
        monkeypatch.setenv('ROS_LOG_DIR', directory)
        with open(Path(directory) / 'controller.log', 'w+') as log:
            try:
                rclpy.init(context=context)
                node = rclpy.create_node('cart_stack_test', context=context)
                costmap = scope+'_costmap'
                server = 'controller' if scope == 'local' else 'planner'
                executor = SingleThreadedExecutor(context=context)
                executor.add_node(node)
                broadcaster = StaticTransformBroadcaster(node)
                transform = TransformStamped()
                transform.header.stamp = node.get_clock().now().to_msg()
                transform.header.frame_id = 'odom' if scope == 'local' else 'map'
                transform.child_frame_id = 'nav_control'
                angle = .4
                transform.transform.rotation.w = math.cos(angle/2)
                transform.transform.rotation.z = math.sin(angle/2)
                base_transform = copy.deepcopy(transform)
                base_transform.child_frame_id = 'base_nav'
                broadcaster.sendTransform([transform, base_transform])
                footprints = []
                node.create_subscription(PolygonStamped, '/'+costmap+'/published_footprint',
                                         lambda msg: footprints.append(msg), 10)
                qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
                publisher = node.create_publisher(Polygon, '/'+costmap+'/footprint', qos)
                scan_pub = node.create_publisher(LaserScan, '/scan', 10)
                grids = []
                node.create_subscription(OccupancyGrid, '/'+costmap+'/costmap', grids.append, qos)
                if scope == 'global':
                    map_pub = node.create_publisher(OccupancyGrid, '/map', qos)
                    static_map = OccupancyGrid()
                    static_map.header.frame_id = 'map'
                    static_map.info.width = static_map.info.height = 160
                    static_map.info.resolution = .05
                    static_map.info.origin.position.x = static_map.info.origin.position.y = -4.
                    static_map.info.origin.orientation.w = 1.
                    static_map.data = [0]*25600
                    map_pub.publish(static_map)
                package = 'nav2_'+server
                binary = Path(get_package_prefix(package)) / ('lib/'+package+'/'+server+'_server')
                params = Path(__file__).resolve().parents[1] / 'config/nav2_foxy_g1.yaml'
                process = subprocess.Popen([str(binary), '--ros-args', '-r', '__node:='+server+'_server',
                                            '--params-file', str(params)],
                                           stdout=log, stderr=subprocess.STDOUT, env=os.environ.copy())
                client = node.create_client(ChangeState, '/'+server+'_server/change_state')

                def diagnostic():
                    log.flush(); log.seek(0)
                    return log.read()

                def pump(predicate, timeout=10.):
                    deadline = time.monotonic()+timeout
                    while time.monotonic() < deadline:
                        executor.spin_once(timeout_sec=.02)
                        if predicate():
                            return
                    raise AssertionError(diagnostic())

                pump(client.service_is_ready)
                for transition in (1, 3):  # Configure, activate; no motion action is sent.
                    request = ChangeState.Request()
                    request.transition.id = transition
                    future = client.call_async(request)
                    pump(future.done)
                    assert future.result().success, diagnostic()
                polygon = Polygon(points=[Point32(x=x, y=y) for x, y in
                                          [(-1.4, -.6), (.7, -.6), (.7, .6), (-1.4, .6)]])
                publisher.publish(polygon)
                pump(lambda: any(min(p.x for p in f.polygon.points) < -1.3 for f in footprints))
                text = diagnostic()
                if scope == 'local':
                    assert 'CartPush' in text and 'PathAlign' in text, text
                # nav_control changes reference on a mode switch. The cart
                # polygon must remain attached to the stable robot frame.
                transform.transform.translation.x = 1.
                broadcaster.sendTransform([transform, base_transform])

                def publish_scan(ranges):
                    scan = LaserScan(angle_min=0., angle_max=(len(ranges)-1)*math.pi/2,
                                     angle_increment=math.pi/2, range_min=.05, range_max=4.,
                                     ranges=ranges)
                    scan.header.frame_id = 'base_nav'
                    scan.header.stamp = node.get_clock().now().to_msg()
                    scan_pub.publish(scan)

                clear_client = node.create_client(
                    ClearEntireCostmap, '/'+costmap+'/clear_entirely_'+costmap)
                pump(clear_client.service_is_ready)

                def clear():
                    future = clear_client.call_async(ClearEntireCostmap.Request())
                    pump(future.done)
                    future.result()

                def cost(x, y):
                    if not grids:
                        return None
                    grid = grids[-1]
                    wx, wy = math.cos(angle)*x-math.sin(angle)*y, math.sin(angle)*x+math.cos(angle)*y
                    col = int((wx-grid.info.origin.position.x)/grid.info.resolution)
                    row = int((wy-grid.info.origin.position.y)/grid.info.resolution)
                    return grid.data[row*grid.info.width+col]

                def until_cost(predicate, action, timeout=5.):
                    def step():
                        action()
                        return predicate()
                    pump(step, timeout)

                # Seed a cart return and a real obstacle outside the physical
                # cart but inside the much larger navigation footprint.
                until_cost(lambda: cost(.3, 0.) == 100 and cost(0., 1.) == 100,
                           lambda: publish_scan([.3, 1.]))
                # Replace the old enlarged footprint with the model outline.
                # Stock inflation should narrow its highest-cost region while
                # retaining the exact same measured obstacles, without a reset.
                until_cost(lambda: cost(0., .5) == 99, lambda: publish_scan([.3, 1.]))
                settings = load_settings(Path(__file__).resolve().parents[3] / 'config/pushcart_navigation.json')
                geometry = CartGeometry(settings)
                outline = geometry.footprint(dict(position=[.8, 0., -.7], yaw=0.,
                                                  palms=[[.5, .2, .1], [.5, -.2, .1]]), True)
                model_polygon = Polygon(points=[Point32(x=x, y=y) for x, y in outline])
                def publish_model_footprint():
                    publisher.publish(model_polygon)
                    publish_scan([.3, 1.])
                until_cost(lambda: 0 < cost(0., .5) < 99 and cost(.3, 0.) == 100
                           and cost(0., 1.) == 100, publish_model_footprint)
                # Masking the new cart returns alone cannot erase old cells.
                deadline = time.monotonic()+.7
                while time.monotonic() < deadline:
                    publish_scan([float('nan'), 1.])
                    executor.spin_once(timeout_sec=.03)
                assert cost(.3, 0.) == 100
                # Stock Nav2 clear service is used only with the motion gate
                # closed; scans then rebuild external obstacles before opening.
                clear()
                until_cost(lambda: cost(.3, 0.) != 100 and cost(0., 1.) == 100,
                           lambda: publish_scan([float('nan'), 1.]))
                deadline = time.monotonic()+.7
                while time.monotonic() < deadline:
                    publish_scan([float('nan'), 1.])
                    executor.spin_once(timeout_sec=.03)
                    assert cost(.3, 0.) != 100 and cost(0., 1.) == 100
                if scope == 'global':
                    # Resetting the stock layers must reapply static obstacles.
                    x, y = .3*math.cos(angle), .3*math.sin(angle)
                    col, row = int((x+4.)/.05), int((y+4.)/.05)
                    static_map.data[row*160+col] = 100
                    map_pub.publish(static_map)
                    until_cost(lambda: cost(.3, 0.) == 100,
                               lambda: publish_scan([float('nan'), 1.]))
                    clear()
                    until_cost(lambda: cost(.3, 0.) == 100 and cost(0., 1.) == 100,
                               lambda: publish_scan([float('nan'), 1.]))

                # A moving rear ignore region must also clear previously
                # cached returns, without clearing obstacles beyond it.
                settings = load_settings(Path(__file__).resolve().parents[3] / 'config/pushcart_navigation.json')
                gantry = GantryGeometry(settings['gantry_filter'])
                center, angles, distances = gantry.clearing_rays()
                body_tf = TransformStamped()
                body_tf.header.frame_id, body_tf.child_frame_id = 'base_nav', 'base_link'
                body_tf.transform.rotation.w = 1.
                rear_tf = TransformStamped()
                rear_tf.header.frame_id, rear_tf.child_frame_id = 'base_link', 'gantry_exclusion'
                rear_tf.transform.rotation.w = 1.
                rear_tf.transform.translation.x, rear_tf.transform.translation.y, rear_tf.transform.translation.z = map(float, center)
                broadcaster.sendTransform([transform, base_transform, body_tf, rear_tf])
                rear_pub = node.create_publisher(LaserScan, '/navigation/rear_clear', 10)
                rear_scan = LaserScan(angle_min=float(angles[0]), angle_max=float(angles[-1]),
                                      angle_increment=float(angles[1]-angles[0]), range_min=0.,
                                      range_max=4., ranges=distances.astype(np.float32).tolist())
                rear_scan.header.frame_id = 'gantry_exclusion'
                until_cost(lambda: cost(-1.5, 0.) == 100,
                           lambda: publish_scan([float('nan'), 1., 1.5]))
                until_cost(lambda: cost(-.75, 0.) == 100,
                           lambda: publish_scan([float('nan'), 1., .75]))

                def clear_rear():
                    publish_scan([float('nan'), 1., float('nan')])
                    rear_scan.header.stamp = node.get_clock().now().to_msg()
                    rear_pub.publish(rear_scan)

                until_cost(lambda: cost(-.75, 0.) != 100 and cost(-1.5, 0.) == 100
                           and cost(0., 1.) == 100, clear_rear)
                # Shift the robot-relative clearing frame. The old region must
                # expire; only the new rectangle may keep clearing cached hits.
                body_tf.transform.translation.x = .4
                broadcaster.sendTransform([transform, base_transform, body_tf, rear_tf])
                deadline = time.monotonic()+.4
                while time.monotonic() < deadline:
                    clear_rear()
                    executor.spin_once(timeout_sec=.03)
                until_cost(lambda: cost(-.75, 0.) == 100,
                           lambda: publish_scan([float('nan'), 1., .75]))
                until_cost(lambda: cost(-.25, 0.) == 100,
                           lambda: publish_scan([float('nan'), 1., .25]))
                until_cost(lambda: cost(-.25, 0.) != 100 and cost(-.75, 0.) == 100
                           and cost(-1.5, 0.) == 100 and cost(0., 1.) == 100, clear_rear)
            finally:
                if process is not None and process.poll() is None:
                    process.send_signal(signal.SIGINT)
                    try:
                        process.wait(timeout=5.)
                    except subprocess.TimeoutExpired:
                        process.kill(); process.wait(timeout=2.)
                if executor is not None:
                    executor.shutdown()
                if node is not None:
                    node.destroy_node()
                if context.ok():
                    context.shutdown()


@pytest.mark.parametrize('scenario', ['straight', 'diagonal', 'fast_diagonal', 'velocity_noise', 'curve', 'state_sway',
                                     'offset_left', 'offset_right', 'heading_left', 'heading_right',
                                     'delayed_response', 'fixed_offset', 'fixed_delayed_response'])
def test_cart_controller_tracks_cart_center_and_rejects_obstacle_at_robot_rear(monkeypatch, scenario,
                                                                            record_property):
    monkeypatch.setenv('ROS_DOMAIN_ID', '89')
    monkeypatch.setenv('ROS_LOCALHOST_ONLY', '1')
    monkeypatch.setenv('RMW_IMPLEMENTATION', 'rmw_cyclonedds_cpp')
    monkeypatch.setenv('CYCLONEDDS_URI',
        '<CycloneDDS><Domain><General><NetworkInterfaceAddress>lo</NetworkInterfaceAddress>'
        '<AllowMulticast>false</AllowMulticast></General><Discovery><ParticipantIndex>auto</ParticipantIndex>'
        '<Peers><Peer Address="127.0.0.1"/></Peers></Discovery></Domain></CycloneDDS>')
    context = Context()
    process = node = executor = action = None
    with tempfile.TemporaryDirectory(prefix='cart_follow_') as directory:
        monkeypatch.setenv('ROS_LOG_DIR', directory)
        with open(Path(directory) / 'controller.log', 'w+') as log:
            try:
                rclpy.init(context=context)
                node = rclpy.create_node('cart_follow_test', context=context)
                executor = SingleThreadedExecutor(context=context)
                executor.add_node(node)
                broadcaster = TransformBroadcaster(node)
                # The real global plan is in map and DWB predicts in odom.
                # A substantial map rotation exposes swapped axes/signs and
                # accidental comparisons between these two coordinate frames.
                map_broadcaster = StaticTransformBroadcaster(node)
                map_yaw = -2.84
                map_rotation = np.array([[math.cos(map_yaw), -math.sin(map_yaw)],
                                         [math.sin(map_yaw), math.cos(map_yaw)]])
                map_origin = np.array([-2.3, 1.7])
                map_tf = TransformStamped()
                map_tf.header.frame_id, map_tf.child_frame_id = 'map', 'odom'
                map_tf.header.stamp = node.get_clock().now().to_msg()
                map_tf.transform.translation.x, map_tf.transform.translation.y = map(float, map_origin)
                map_tf.transform.rotation.w, map_tf.transform.rotation.z = math.cos(map_yaw/2), math.sin(map_yaw/2)
                map_broadcaster.sendTransform(map_tf)
                qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
                footprint_pub = node.create_publisher(Polygon, '/local_costmap/footprint', qos)
                scan_pub = node.create_publisher(LaserScan, '/scan', 10)
                odom_pub = node.create_publisher(Odometry, '/Odometry_navigation', 10)
                grids, commands, footprints = [], [], []
                recorded_poses, local_plans, transformed_plans = [], [], []
                plan_inputs = {}
                def local_plan_received(message):
                    local_plans.append(message)
                    plan_inputs[id(message)] = dict(estimator.navigation_odometry)
                node.create_subscription(NavigationPath, '/local_plan', local_plan_received, 10)
                node.create_subscription(NavigationPath, '/transformed_global_plan', transformed_plans.append, 10)
                node.create_subscription(OccupancyGrid, '/local_costmap/costmap', grids.append, qos)
                node.create_subscription(Twist, '/cmd_vel',
                    lambda msg: commands.append((time.monotonic(), msg)), 10)
                node.create_subscription(PolygonStamped, '/local_costmap/published_footprint', footprints.append, 10)
                geometry = CartGeometry(load_settings(Path(__file__).resolve().parents[3] / 'config/pushcart_navigation.json'))
                from test_cart_navigation import harness
                from unitree_nav2_web.cart_navigation import CartNavigation
                estimator = harness(geometry)
                estimator._fresh = lambda: True
                estimator.get_clock = node.get_clock
                truth_history = deque(maxlen=256)
                outline = geometry.footprint(dict(position=[.8, 0., -.7], yaw=0.,
                                                  palms=[[.5, .2, .1], [.5, -.2, .1]]), True)
                footprint = Polygon(points=[Point32(x=x, y=y) for x, y in outline])
                params = Path(__file__).resolve().parents[1] / 'config/nav2_foxy_g1.yaml'
                process = subprocess.Popen([
                    '/opt/ros/foxy/lib/nav2_controller/controller_server', '--ros-args', '--params-file', str(params),
                    '-p', 'CartPush.navigation_speed_mode:='+('fixed' if scenario.startswith('fixed_') else 'auto'),
                    '-p', 'CartPush.max_speed_xy:='+('0.6' if scenario == 'fast_diagonal' else '0.35'),
                    '-p', 'CartPush.navigation_speed:='+('0.6' if scenario == 'fast_diagonal' else '0.35')],
                    stdout=log, stderr=subprocess.STDOUT, env=os.environ.copy())
                lifecycle = node.create_client(ChangeState, '/controller_server/change_state')
                action = ActionClient(node, FollowPath, '/follow_path')
                yaw = -1.457 if 'diagonal' in scenario else .4
                heading = np.array([math.cos(yaw), math.sin(yaw)])
                lateral = np.array([-heading[1], heading[0]])
                cart = np.r_[.8*heading, yaw]
                path_start = cart[:2].copy()
                displaced = scenario in ('offset_left', 'offset_right', 'fixed_offset')
                misaligned = scenario in ('heading_left', 'heading_right')
                delayed = scenario in ('delayed_response', 'fixed_delayed_response')
                if displaced:
                    cart[:2] += (-.25 if scenario == 'offset_right' else .25)*lateral
                if misaligned:
                    cart[2] += -.45 if scenario == 'heading_right' else .45
                velocity = np.zeros(3)
                previous = last_publish = time.monotonic()
                moving, obstacle = True, False
                started = time.monotonic()

                def tick():
                    nonlocal previous, last_publish
                    now = time.monotonic()
                    sample_ros_ns = node.get_clock().now().nanoseconds
                    dt = min(.05, now-previous)
                    previous = now
                    target = np.zeros(3)
                    command_delay = .12 if delayed else 0.
                    for sent, msg in reversed(commands):
                        if now-sent >= command_delay:
                            if now-sent < .25+command_delay:
                                target[:] = msg.linear.x, msg.linear.y, msg.angular.z
                            break
                    velocity[:] += min(1., dt/(.8 if delayed else .4))*(target-velocity)
                    if moving:
                        c, s = math.cos(cart[2]), math.sin(cart[2])
                        cart[:2] += dt*np.array([c*velocity[0]-s*velocity[1], s*velocity[0]+c*velocity[1]])
                        cart[2] += dt*velocity[2]
                    else:
                        velocity.fill(0.)  # A fixed test fixture has zero measured motion.
                    recorded_poses.append(cart.copy())
                    truth_history.append((now, cart.copy()))
                    if scenario == 'state_sway':
                        relative_yaw = -.15*math.sin(4*math.pi*(now-started))
                        estimator.cart = dict(estimator.cart, sample_monotonic=now,
                            pose_source='grasp', yaw=relative_yaw,
                            position=[.8*math.cos(relative_yaw), .8*math.sin(relative_yaw), 0.])
                        estimator.cart_history.add(estimator.cart)
                    if now-last_publish >= (.1 if delayed else .02):
                        last_publish = now
                        stamp = Time(nanoseconds=sample_ros_ns).to_msg()
                        transform = TransformStamped()
                        transform.header.frame_id, transform.child_frame_id = 'odom', 'nav_control'
                        transform.header.stamp = stamp
                        transform.transform.translation.x, transform.transform.translation.y = map(float, cart[:2])
                        transform.transform.rotation.w = math.cos(cart[2]/2)
                        transform.transform.rotation.z = math.sin(cart[2]/2)
                        base = copy.deepcopy(transform)
                        base.child_frame_id = 'base_nav'
                        base.transform.translation.x -= .8*math.cos(cart[2])
                        base.transform.translation.y -= .8*math.sin(cart[2])
                        odom = Odometry()
                        odom.header = transform.header
                        odom.child_frame_id = 'nav_control'
                        odom.pose.pose.position.x, odom.pose.pose.position.y = map(float, cart[:2])
                        odom.pose.pose.orientation = transform.transform.rotation
                        odom.twist.twist.linear.x, odom.twist.twist.linear.y, odom.twist.twist.angular.z = map(float, velocity)
                        if scenario == 'state_sway':
                            captured = now-.06
                            if truth_history[0][0] > captured:
                                executor.spin_once(timeout_sec=.01)
                                return
                            times = [t for t, _ in truth_history]
                            poses = np.array([p for _, p in truth_history])
                            measured = np.array([np.interp(captured, times, poses[:, i]) for i in range(3)])
                            body_yaw = measured[2]+.15*math.sin(4*math.pi*(captured-started))
                            sample = Odometry()
                            sample.header.frame_id = 'odom'
                            # Stamp the same instant as the synthetic measurement,
                            # not the end of Python interpolation/message setup.
                            sample.header.stamp = Time(nanoseconds=sample_ros_ns-60000000).to_msg()
                            sample.pose.pose.position.x = measured[0]-.8*math.cos(measured[2])
                            sample.pose.pose.position.y = measured[1]-.8*math.sin(measured[2])
                            sample.pose.pose.orientation.w = math.cos(body_yaw/2)
                            sample.pose.pose.orientation.z = math.sin(body_yaw/2)
                            CartNavigation._receive_odom(estimator, sample)
                            if not estimator.odom_pub.messages:
                                return
                            odom = estimator.odom_pub.messages[-1]
                            # DWB receives the matched cart pose and measured
                            # velocity, including real estimator dynamics.
                            transform.header = odom.header
                            transform.transform.translation.x = odom.pose.pose.position.x
                            transform.transform.translation.y = odom.pose.pose.position.y
                            transform.transform.rotation = odom.pose.pose.orientation
                        if scenario == 'velocity_noise' and not obstacle:
                            # Correct heading and position, noisy measured lateral
                            # speed: isolate the planner from camera heading errors.
                            odom.twist.twist.linear.y += .2*math.sin(4*math.pi*(now-started))
                        broadcaster.sendTransform([transform, base])
                        odom_pub.publish(odom)
                        footprint_pub.publish(footprint)
                        scan = LaserScan(angle_min=-math.pi, angle_max=math.pi,
                                         angle_increment=math.pi/180, range_min=.05, range_max=4.,
                                         ranges=[float('inf')]*361)
                        scan.header.frame_id, scan.header.stamp = 'base_nav', stamp
                        if obstacle:
                            # A rear obstacle spans the footprint boundary. One
                            # zero-width point exactly on the edge can rasterize
                            # to either side when the cart/grid angle changes.
                            rear = float(-min(x for x, _ in outline)-.8)
                            for index in range(361):
                                angle = -math.pi+index*scan.angle_increment
                                if math.cos(angle) < -1e-6:
                                    distance = -rear/math.cos(angle)
                                    if abs(distance*math.sin(angle)) < .4:
                                        scan.ranges[index] = distance
                        scan_pub.publish(scan)
                    executor.spin_once(timeout_sec=.01)

                def diagnostic():
                    log.flush()
                    log.seek(0)
                    return log.read()

                def until(predicate, timeout=12.):
                    end = time.monotonic()+timeout
                    while time.monotonic() < end:
                        tick()
                        if predicate():
                            return
                    raise AssertionError(diagnostic()[-16000:])

                until(lifecycle.service_is_ready)
                for transition in (1, 3):
                    req = ChangeState.Request()
                    req.transition.id = transition
                    future = lifecycle.call_async(req)
                    until(future.done)
                    assert future.result().success, diagnostic()
                until(lambda: action.server_is_ready() and footprints and grids)

                def goal(target, curve=False, start=None):
                    start = cart[:2].copy() if start is None else start
                    request = FollowPath.Goal()
                    request.controller_id = 'CartPush'
                    request.path = NavigationPath()
                    request.path.header.frame_id = 'map'
                    request.path.header.stamp = node.get_clock().now().to_msg()
                    for fraction in np.linspace(0., 1., 100):
                        pose = PoseStamped()
                        pose.header = request.path.header
                        angle = fraction*math.pi/2 if curve else 0.
                        if curve:
                            lateral = np.array([-heading[1], heading[0]])
                            point = start+.9*(math.sin(angle)*heading+(1-math.cos(angle))*lateral)
                        else:
                            point = start+fraction*(target-start)
                        point = map_origin + map_rotation @ point
                        pose.pose.position.x, pose.pose.position.y = map(float, point)
                        pose.pose.orientation.w, pose.pose.orientation.z = math.cos((yaw+angle+map_yaw)/2), math.sin((yaw+angle+map_yaw)/2)
                        request.path.poses.append(pose)
                    future = action.send_goal_async(request)
                    until(future.done)
                    handle = future.result()
                    assert handle.accepted, diagnostic()
                    return handle.get_result_async()

                target = path_start+(.65 if scenario == 'straight' else 2.0)*heading
                if scenario == 'curve':
                    target = cart[:2]+.9*(heading+np.array([-heading[1], heading[0]]))
                result = goal(target, curve=scenario == 'curve', start=path_start)
                until(result.done, 18.)
                assert result.result().status == 4, diagnostic()
                assert transformed_plans
                first_plan = transformed_plans[0]
                # The full 2 m test path fits inside the local costmap. A
                # default 1 m forward prune creates an artificial endpoint
                # inside the rollout + 0.6 m alignment lookahead at high speed.
                assert first_plan.header.frame_id == 'odom'
                np.testing.assert_allclose(
                    [first_plan.poses[-1].pose.position.x, first_plan.poses[-1].pose.position.y],
                    target, atol=.025)
                assert np.linalg.norm(cart[:2]-target) <= .18
                robot = cart[:2]-.8*np.array([math.cos(cart[2]), math.sin(cart[2])])
                assert np.linalg.norm(robot-target) > .65  # Arrival is at the CART center.
                assert any(msg.linear.x > .05 for _, msg in commands)
                assert all(-.2-1e-6 <= msg.linear.x <= .6+1e-6 and abs(msg.linear.y) <= .3+1e-6
                           and abs(msg.angular.z) <= .4+1e-6 for _, msg in commands)
                assert all(math.hypot(msg.linear.x, msg.linear.y) <=
                           (.6 if scenario == 'fast_diagonal' else .35)+1e-6 for _, msg in commands)
                record_property('local_costmap_resolution_m', grids[-1].info.resolution)
                intervals = np.diff([stamp for stamp, _ in commands])*1000.
                record_property('median_command_period_ms', float(np.median(intervals)))
                record_property('p95_command_period_ms', float(np.percentile(intervals, 95)))
                if scenario != 'curve':
                    # Test following the path, including the green prediction.
                    # A small command change alone says nothing about tracking.
                    cross_errors = [abs((p[:2]-path_start) @ lateral) for p in recorded_poses]
                    yaw_errors = [abs(math.atan2(math.sin(p[2]-yaw), math.cos(p[2]-yaw))) for p in recorded_poses]
                    record_property('max_cross_track_m', max(cross_errors))
                    record_property('final_cross_track_m', cross_errors[-1])
                    record_property('final_heading_error_rad', yaw_errors[-1])
                    assert max(cross_errors) < (.27 if displaced else .10 if misaligned else .08)
                    assert max(yaw_errors) < (.47 if misaligned else .3 if displaced else .15)
                    # The configured completion tolerance is 10 degrees.
                    # DWB legitimately completes before reaching 0.10 rad;
                    # retain the stricter heading bound over the entire route.
                    assert cross_errors[-1] < .04 and yaw_errors[-1] <= math.radians(10.)
                    # Recovery must happen along the route, before goal rotation
                    # can hide a controller that drove the entire path sideways.
                    settled = [p for p in recorded_poses if (p[:2]-path_start) @ heading > 1.]
                    if displaced or misaligned:
                        assert settled and max(abs((p[:2]-path_start) @ lateral) for p in settled) < .06
                        assert max(abs(p[2]-yaw) for p in settled) < .15
                    assert local_plans
                    predicted_cross = max(abs((np.array([pose.pose.position.x,
                                                        pose.pose.position.y])-path_start) @ lateral)
                                          for plan in local_plans for pose in plan.poses)
                    record_property('max_predicted_cross_track_m', predicted_cross)
                    worst_plan = max((p for p in local_plans if p.poses), key=lambda p:
                        max(abs((np.array([s.pose.position.x, s.pose.position.y])-path_start) @ lateral)
                            for s in p.poses))
                    first = worst_plan.poses[0].pose.position
                    record_property('worst_prediction_goal_distance_m',
                                    float(np.linalg.norm(np.array([first.x, first.y])-target)))
                    record_property('worst_prediction_input', plan_inputs[id(worst_plan)])
                    record_property('worst_prediction_points', [
                        [s.pose.position.x, s.pose.position.y,
                         2*math.atan2(s.pose.orientation.z, s.pose.orientation.w)]
                        for s in worst_plan.poses])
                    if scenario != 'velocity_noise':
                        assert predicted_cross < (.27 if displaced else .15 if misaligned else .08)
                assert 'Created controller : CartPush of type dwb_core::DWBLocalPlanner' in diagnostic()
                assert 'Using critic "PathAlign" (dwb_critics::PathAlignCritic)' in diagnostic()
                assert 'Using critic "CommandChange"' not in diagnostic()
                assert 'Using critic "CartLookahead"' not in diagnostic()
                moving, obstacle = False, True
                velocity.fill(0.)
                commands.clear()
                until(lambda: any(value == 100 for value in grids[-1].data))
                result = goal(cart[:2]+.5*heading)
                until(result.done)
                assert result.result().status == 6, diagnostic()
                assert 'ObstacleFootprint' in diagnostic() and 'No valid trajectories' in diagnostic()
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
                if action is not None:
                    action.destroy()
                if node is not None:
                    node.destroy_node()
                if context.ok():
                    context.shutdown()
