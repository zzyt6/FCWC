"""Exercise acquisition-time joints, odometry and cloud TF on an isolated graph."""
from array import array
import json
import math
import os
from pathlib import Path
import socket
import struct
import subprocess
import tempfile
import time

import numpy as np
import pytest
import rclpy
from rclpy.context import Context
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import Imu, LaserScan, PointCloud2, PointField, TimeReference
from std_msgs.msg import String

from unitree_nav2_web.cart_geometry import CartGeometry, load_settings, quaternion_matrix, rotation_z


@pytest.mark.parametrize('period, speed', [(.1, 0.), (.019, .3), (.020, .3)])
def test_delayed_lidar_uses_historical_waist_for_odometry_and_cloud(monkeypatch, period, speed):
    monkeypatch.setenv('ROS_DOMAIN_ID', '92')
    monkeypatch.setenv('ROS_LOCALHOST_ONLY', '1')
    monkeypatch.setenv('RMW_IMPLEMENTATION', 'rmw_cyclonedds_cpp')
    monkeypatch.setenv('CYCLONEDDS_URI',
        '<CycloneDDS><Domain><General><NetworkInterfaceAddress>lo</NetworkInterfaceAddress>'
        '<AllowMulticast>false</AllowMulticast></General><Discovery><ParticipantIndex>auto</ParticipantIndex>'
        '<Peers><Peer Address="127.0.0.1"/></Peers></Discovery></Domain></CycloneDDS>')
    root = Path(__file__).resolve().parents[3]
    context = Context()
    rclpy.init(context=context)
    node = rclpy.create_node('acquisition_time_test', context=context)
    executor = SingleThreadedExecutor(context=context)
    executor.add_node(node)
    udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp.bind(('127.0.0.1', 0))
    port = udp.getsockname()[1]
    udp.close()
    udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    imu_pub = node.create_publisher(Imu, '/utlidar/imu_livox_mid360', qos_profile_sensor_data)
    odom_pub = node.create_publisher(Odometry, '/Odometry', 10)
    cloud_pub = node.create_publisher(PointCloud2, '/cloud_registered_body', qos_profile_sensor_data)
    outputs, clouds, mounts, clocks = [], [], [], []
    node.create_subscription(Odometry, '/Odometry_base', lambda m: outputs.append((time.time(), m)), 20)
    node.create_subscription(PointCloud2, '/cloud_nav', clouds.append, qos_profile_sensor_data)
    node.create_subscription(TransformStamped, '/g1/lidar_mount', mounts.append, 20)
    node.create_subscription(TimeReference, '/g1/lidar_clock', clocks.append, qos_profile_sensor_data)
    processes = []
    with tempfile.TemporaryDirectory(prefix='state-timing-') as directory:
        monkeypatch.setenv('ROS_LOG_DIR', directory)
        with open(Path(directory)/'nodes.log', 'w+') as log:
            try:
                processes.append(subprocess.Popen([
                    str(root/'build/unitree_g1_kinematics/g1_base_kinematics_node'),
                    '--ros-args', '-p', 'lowstate_udp_port:='+str(port)], stdout=log, stderr=log))
                processes.append(subprocess.Popen([
                    str(root/'build/unitree_nav2_core/cloud_frame_transformer'),
                    '--ros-args', '-p', 'target_frame:=base_nav'], stdout=log, stderr=log))
                started, wall_started = time.monotonic(), time.time()
                next_odom, sequence = started+.8, 0
                world_point = np.array([1., 2., .1])
                # Original 3-DoF URDF; roll/pitch joints are zero. Pelvis stays
                # fixed while waist yaw moves, so aligned base motion is zero.
                pitch = .04014257279586953
                ry = np.array([[math.cos(pitch), 0., math.sin(pitch)], [0., 1., 0.],
                               [-math.sin(pitch), 0., math.cos(pitch)]])
                mount_rotation = ry @ np.diag([1., -1., -1.])
                mount_position = (np.array([-.0039635+.0002835, .00003, .035+.019+.40618])
                                  + mount_rotation @ [.011, .02329, -.04412])
                while time.monotonic()-started < 4.8:
                    now, wall = time.monotonic(), time.time()
                    q = [0.]*29
                    q[12] = .7*(now-.012-started)
                    udp.sendto(struct.pack('<IHBBQd29d', 0x47314c53, 2, 0, 0, sequence, now-.012, *q),
                               ('127.0.0.1', port))
                    sequence += 1
                    imu = Imu()
                    imu.header.stamp = Time(seconds=wall-62.).to_msg()
                    imu_pub.publish(imu)
                    if now >= next_odom:
                        sampled = next_odom-.085
                        next_odom += period
                        if 3.8 < now-started < 4.45:
                            executor.spin_once(timeout_sec=.005)
                            continue
                        elapsed = sampled-started
                        yaw = .7*elapsed
                        rz = np.array([[math.cos(yaw), -math.sin(yaw), 0.],
                                       [math.sin(yaw), math.cos(yaw), 0.], [0., 0., 1.]])
                        rotation, position = rz @ mount_rotation, rz @ mount_position
                        position[0] += speed*min(elapsed, 3.7)
                        # qz(yaw) * qy(pitch) * qx(pi).
                        cy, sy = math.cos(yaw/2), math.sin(yaw/2)
                        cp, sp = math.cos(pitch/2), math.sin(pitch/2)
                        odom = Odometry()
                        odom.header.frame_id, odom.child_frame_id = 'odom', 'livox_imu'
                        odom.header.stamp = Time(seconds=wall_started+elapsed-62.).to_msg()
                        odom.pose.pose.position.x, odom.pose.pose.position.y, odom.pose.pose.position.z = map(float, position)
                        quat = odom.pose.pose.orientation
                        quat.w, quat.x, quat.y, quat.z = sy*sp, cy*cp, sy*cp, -cy*sp
                        point = (rotation.T @ (world_point-position)).astype('<f4')
                        cloud = PointCloud2(height=1, width=1, point_step=12, row_step=12,
                            fields=[PointField(name=name, offset=i*4, datatype=PointField.FLOAT32, count=1)
                                    for i, name in enumerate(('x', 'y', 'z'))], data=array('B', point.tobytes()))
                        cloud.header.frame_id, cloud.header.stamp = 'livox_imu', odom.header.stamp
                        cloud_pub.publish(cloud)  # Deliberately before its time-reference/TF.
                        odom_pub.publish(odom)
                    executor.spin_once(timeout_sec=.005)
                log.flush(); log.seek(0); diagnostic = log.read()
                assert len(outputs) >= 15, diagnostic
                assert len(clouds) >= 15, diagnostic
                assert len(mounts) >= 100 and len(clocks) >= 100, diagnostic
                for mount in mounts:
                    stamp = mount.header.stamp.sec+mount.header.stamp.nanosec*1e-9
                    q = mount.transform.rotation
                    yaw = math.atan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z))
                    error = yaw-.7*(stamp-wall_started)
                    assert abs(math.atan2(math.sin(error), math.cos(error))) < .003
                for clock in clocks:
                    stamp = clock.header.stamp.sec+clock.header.stamp.nanosec*1e-9
                    sensor = clock.time_ref.sec+clock.time_ref.nanosec*1e-9
                    assert abs(stamp-sensor-62.) < .02
                for received, output in outputs[3:]:
                    q = output.pose.pose.orientation
                    assert abs(2*math.atan2(q.z, q.w)) < .015, diagnostic
                    stamp = output.header.stamp.sec+output.header.stamp.nanosec*1e-9
                    assert .025 < received-stamp < .2
                for cloud in clouds[3:]:
                    elapsed = cloud.header.stamp.sec+cloud.header.stamp.nanosec*1e-9-wall_started
                    expected = world_point-np.array([speed*min(elapsed, 3.7), 0., 0.])
                    np.testing.assert_allclose(np.frombuffer(cloud.data, dtype='<f4'), expected, atol=.025)
                    assert cloud.header.frame_id == 'base_nav'
                moving = [m.twist.twist.linear.x for _, m in outputs
                          if 1.2 < m.header.stamp.sec+m.header.stamp.nanosec*1e-9-wall_started < 3.5]
                np.testing.assert_allclose(moving, speed, atol=.02)
                for (_, before), (_, after) in zip(outputs, outputs[1:]):
                    a = before.header.stamp.sec+before.header.stamp.nanosec*1e-9
                    b = after.header.stamp.sec+after.header.stamp.nanosec*1e-9
                    if b-a > .5:
                        assert abs(after.twist.twist.linear.x) < .01, 'A stream gap must reset old velocity'
                # No current-waist substitution when an old measurement arrives.
                odom.header.stamp = Time(seconds=time.time()-63.).to_msg()
                before = len(outputs)
                odom_pub.publish(odom)
                until = time.monotonic()+.15
                while time.monotonic() < until:
                    executor.spin_once(timeout_sec=.01)
                assert len(outputs) <= before+1  # One previously in-flight frame is possible.
            finally:
                for process in processes:
                    process.terminate()
                    try:
                        process.wait(timeout=5.)
                    except subprocess.TimeoutExpired:
                        process.kill(); process.wait(timeout=2.)
                udp.close()
                executor.shutdown()
                node.destroy_node()
                context.shutdown()


@pytest.mark.parametrize('feedback_hz, point_groups, repeats, pose_delay', [
    (200, 2, 1, .12), (1200, 20, 500, .17),
])
def test_raw_filter_receives_clock_and_delayed_pose_without_lio(
        monkeypatch, feedback_hz, point_groups, repeats, pose_delay):
    monkeypatch.setenv('ROS_DOMAIN_ID', '93')
    monkeypatch.setenv('ROS_LOCALHOST_ONLY', '1')
    monkeypatch.setenv('RMW_IMPLEMENTATION', 'rmw_cyclonedds_cpp')
    monkeypatch.setenv('CYCLONEDDS_URI',
        '<CycloneDDS><Domain><General><NetworkInterfaceAddress>lo</NetworkInterfaceAddress>'
        '<AllowMulticast>false</AllowMulticast></General><Discovery><ParticipantIndex>auto</ParticipantIndex>'
        '<Peers><Peer Address="127.0.0.1"/></Peers></Discovery></Domain></CycloneDDS>')
    root = Path(__file__).resolve().parents[3]
    settings = root/'config/pushcart_navigation.json'
    geometry = CartGeometry(load_settings(settings))
    context = Context()
    rclpy.init(context=context)
    node = rclpy.create_node('raw_acquisition_test', context=context)
    executor = SingleThreadedExecutor(context=context)
    executor.add_node(node)
    sockets = [socket.socket(socket.AF_INET, socket.SOCK_DGRAM) for _ in range(2)]
    for sock in sockets:
        sock.bind(('127.0.0.1', 0))
    ports = [sock.getsockname()[1] for sock in sockets]
    for sock in sockets:
        sock.close()
    udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    imu_pub = node.create_publisher(Imu, '/utlidar/imu_livox_mid360', qos_profile_sensor_data)
    raw_pub = node.create_publisher(PointCloud2, '/utlidar/cloud_livox_mid360', qos_profile_sensor_data)
    scan_pub = node.create_publisher(LaserScan, '/scan_unmasked', qos_profile_sensor_data)
    outputs, statuses, expected, mounts = [], [], {}, []
    node.create_subscription(PointCloud2, '/navigation/lidar', outputs.append, qos_profile_sensor_data)
    node.create_subscription(String, '/g1/navigation_reference',
                             lambda m: statuses.append(json.loads(m.data)), 10)
    node.create_subscription(TransformStamped, '/g1/lidar_mount', mounts.append, 100)
    processes = []
    with tempfile.TemporaryDirectory(prefix='raw-timing-') as directory:
        monkeypatch.setenv('ROS_LOG_DIR', directory)
        with open(Path(directory)/'nodes.log', 'w+') as log:
            try:
                processes.append(subprocess.Popen([
                    str(root/'build/unitree_g1_kinematics/g1_base_kinematics_node'),
                    '--ros-args', '-p', 'lowstate_udp_port:='+str(ports[0])], stdout=log, stderr=log))
                processes.append(subprocess.Popen([
                    str(root/'install/unitree_nav2_web/lib/unitree_nav2_web/cart_navigation'),
                    '--ros-args', '-p', 'geometry_port:='+str(ports[1]),
                    '-p', 'config_file:='+str(settings)], stdout=log, stderr=log))
                started, last_cloud, sequence = time.monotonic(), 0., 0
                next_joint = started
                pitch = .04014257279586953
                ry = np.array([[math.cos(pitch), 0., math.sin(pitch)], [0., 1., 0.],
                               [-math.sin(pitch), 0., math.cos(pitch)]])
                mount_rotation = ry @ np.diag([1., -1., -1.])
                mount_position = np.array([-.0039635+.0002835, .00003, .035+.019+.40618])
                cart_position = np.array([.8, 0., -.7])
                body_point = geometry.boxes[0][0][:3, 3]+cart_position
                while time.monotonic()-started < 5.:
                    now, wall = time.monotonic(), time.time()
                    # Preserve acquisition timestamps while sending the real
                    # 1.2 kHz input in small batches between ROS callbacks.
                    while next_joint <= now:
                        q = [0.]*29
                        q[12] = .7*(next_joint-started)
                        udp.sendto(struct.pack('<IHBBQd29d', 0x47314c53, 2, 0, 0, sequence, next_joint, *q),
                                   ('127.0.0.1', ports[0]))
                        sequence += 1
                        next_joint += 1/feedback_hz
                    imu = Imu()
                    imu.header.stamp = Time(seconds=wall-62.).to_msg()
                    imu_pub.publish(imu)
                    captured = now-pose_delay
                    cart = dict(epoch=3, valid=True, attached=True, pose_source='apriltag',
                                captured_monotonic=captured, sample_monotonic=captured,
                                position=cart_position.tolist(), yaw=0., rotation=np.eye(3).tolist(),
                                body_rotation=np.eye(3).tolist(), palms=[[.3, .2, .1], [.3, -.2, .1]])
                    udp.sendto(json.dumps(dict(magic='G1LOCO1', cart=cart, sent_monotonic=now)).encode(),
                               ('127.0.0.1', ports[1]))
                    # Keep pose feedback alive across a real observation gap.
                    # Expiry used to crash _tick while encoding numpy.bool_.
                    if now-started > .8 and not 2.3 < now-started < 3.4 and now-last_cloud > .1:
                        last_cloud = now
                        records = []
                        for offset in np.linspace(0., .099, point_groups):
                            rz = rotation_z(.7*(now-.1+offset-started))
                            rotation, position = rz @ mount_rotation, rz @ mount_position
                            for point in (body_point, np.array([0., 2., 0.])):
                                records.append([*((point-position) @ rotation), offset*1e9])
                        records = np.repeat(np.asarray(records, dtype='<f4'), repeats, axis=0)
                        cloud = PointCloud2(height=1, width=len(records), point_step=16, row_step=len(records)*16,
                            fields=[PointField(name=name, offset=i*4, datatype=PointField.FLOAT32, count=1)
                                    for i, name in enumerate(('x', 'y', 'z', 'time'))],
                            data=array('B', records.tobytes()))
                        cloud.header.frame_id = 'livox_frame'
                        cloud.header.stamp = Time(seconds=wall-.1-62.).to_msg()
                        key = (cloud.header.stamp.sec, cloud.header.stamp.nanosec)
                        keep = np.repeat(np.tile([False, True], point_groups), repeats)
                        expected[key] = records[keep].tobytes()
                        raw_pub.publish(cloud)
                        scan = LaserScan(angle_increment=.1, range_min=.1, range_max=5., ranges=[2.])
                        scan.header.frame_id = 'base_nav'
                        scan.header.stamp = Time(seconds=wall-.25).to_msg()
                        scan_pub.publish(scan)
                    executor.spin_once(timeout_sec=.005)
                log.flush(); log.seek(0); diagnostic = log.read()
                assert len(outputs) >= 20 and statuses, diagnostic
                for output in outputs[3:]:
                    key = (output.header.stamp.sec, output.header.stamp.nanosec)
                    assert output.width == point_groups*repeats and bytes(output.data) == expected[key], diagnostic
                assert statuses[-1]['removed_points'] == point_groups*repeats, statuses[-1]
                assert not statuses[-1]['observations']['localization_without_cart_filter'], statuses[-1]
                assert all(process.poll() is None for process in processes), diagnostic
                expired = [s for s in statuses if (s['observations']['filtered_age_s'] or 0.) > .5]
                assert expired and all(not s['gantry_filter']['ready'] for s in expired), diagnostic
                assert statuses[-1]['observations']['filtered_age_s'] < .5, statuses[-1]
                assert statuses[-1]['gantry_filter']['ready'], statuses[-1]
                mount_stamps = [m.header.stamp.sec+m.header.stamp.nanosec*1e-9 for m in mounts]
                assert len(mount_stamps) > 200, diagnostic
                assert (len(mount_stamps)-1)/(mount_stamps[-1]-mount_stamps[0]) <= 205
                for mount, stamp in zip(mounts, mount_stamps):
                    # Throttling selects an actual sample; it must not stamp
                    # old waist feedback with the later publish time.
                    elapsed = stamp-(wall-now)-started
                    q = mount.transform.rotation
                    actual_rotation = quaternion_matrix([q.w, q.x, q.y, q.z])
                    expected_rotation = rotation_z(.7*elapsed) @ mount_rotation
                    np.testing.assert_allclose(actual_rotation, expected_rotation, atol=2e-4)
                    actual = np.array([mount.transform.translation.x, mount.transform.translation.y])
                    wanted = (rotation_z(.7*elapsed) @ mount_position)[:2]
                    np.testing.assert_allclose(actual, wanted, atol=1e-5)
            finally:
                for process in processes:
                    process.terminate()
                    try:
                        process.wait(timeout=5.)
                    except subprocess.TimeoutExpired:
                        process.kill(); process.wait(timeout=2.)
                udp.close()
                executor.shutdown()
                node.destroy_node()
                context.shutdown()
