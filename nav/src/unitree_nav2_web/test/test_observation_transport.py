"""Real cloud/TF transport with delays and invalid observations, no motor nodes."""
from array import array
import os
from pathlib import Path
import subprocess
import tempfile
import time

import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.context import Context
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import PointCloud2, PointField, TimeReference
from tf2_ros import StaticTransformBroadcaster


def test_cloud_queue_skips_unmatchable_head_and_failed_ground_never_clears_obstacles(monkeypatch):
    monkeypatch.setenv('ROS_DOMAIN_ID', '94')
    monkeypatch.setenv('ROS_LOCALHOST_ONLY', '1')
    monkeypatch.setenv('RMW_IMPLEMENTATION', 'rmw_cyclonedds_cpp')
    monkeypatch.setenv('CYCLONEDDS_URI',
        '<CycloneDDS><Domain><General><NetworkInterfaceAddress>lo</NetworkInterfaceAddress>'
        '<AllowMulticast>false</AllowMulticast></General><Discovery><ParticipantIndex>auto</ParticipantIndex>'
        '<Peers><Peer Address="127.0.0.1"/></Peers></Discovery></Domain></CycloneDDS>')
    root = Path(__file__).resolve().parents[3]
    context = Context()
    rclpy.init(context=context)
    node = rclpy.create_node('cloud_transport_test', context=context)
    executor = SingleThreadedExecutor(context=context)
    executor.add_node(node)
    clouds, obstacles = [], []
    source = node.create_publisher(PointCloud2, '/cloud_registered_body', qos_profile_sensor_data)
    timing = node.create_publisher(TimeReference, '/g1/lidar_time_reference', 20)
    node.create_subscription(PointCloud2, '/cloud_nav', clouds.append, qos_profile_sensor_data)
    node.create_subscription(PointCloud2, '/cloud_obstacles', obstacles.append, qos_profile_sensor_data)
    broadcaster = StaticTransformBroadcaster(node)
    transform = TransformStamped()
    transform.header.frame_id, transform.child_frame_id = 'base_nav', 'livox_imu'
    transform.transform.rotation.w = 1.
    broadcaster.sendTransform(transform)
    def pump(duration):
        deadline = time.monotonic()+duration
        while time.monotonic() < deadline:
            executor.spin_once(timeout_sec=.005)
    def cloud(index, points, age=.05, with_timing=True):
        data = np.asarray(points, dtype='<f4').reshape((-1, 4))
        message = PointCloud2(height=1, width=len(data), point_step=16, row_step=len(data)*16,
            fields=[PointField(name=name, offset=i*4, datatype=PointField.FLOAT32, count=1)
                    for i, name in enumerate(('x', 'y', 'z', 'intensity'))],
            data=array('B', data.tobytes()))
        message.header.frame_id = 'livox_imu'
        message.header.stamp = Time(seconds=1000.+index).to_msg()
        reference = TimeReference(time_ref=message.header.stamp)
        reference.header.stamp = Time(seconds=time.time()-age).to_msg()
        source.publish(message)
        if with_timing:
            timing.publish(reference)
        return reference.header.stamp
    processes = []
    with tempfile.TemporaryDirectory(prefix='cloud-transport-') as directory:
        monkeypatch.setenv('ROS_LOG_DIR', directory)
        with open(Path(directory)/'nodes.log', 'w+') as log:
            try:
                for binary in ('cloud_frame_transformer', 'cloud_obstacle_filter'):
                    command = [str(root/'build/unitree_nav2_core'/binary), '--ros-args']
                    if binary == 'cloud_frame_transformer':
                        command += ['-p', 'target_frame:=base_nav']
                    processes.append(subprocess.Popen(command, stdout=log, stderr=log, env=os.environ.copy()))
                deadline = time.monotonic()+5.
                while source.get_subscription_count() < 1 and time.monotonic() < deadline:
                    pump(.05)
                pump(.3)
                # Unknown ground cannot be published as empty/free space.
                cloud(1, [[1., 0., -.4, 1.]])
                pump(.15)
                assert clouds and not obstacles
                # An older cloud has no matching time reference. The next fully
                # paired cloud must go through without waiting out its 300 ms timeout.
                cloud(2, [[1., 0., -.4, 1.]], with_timing=False)
                pump(.03)
                ground = [[x, y, -.8, 1.] for x in np.linspace(.7, 2., 15)
                          for y in np.linspace(-1., 1., 15)]
                stamp = cloud(3, ground+[[1., 0., -.4, 1.]])
                deadline = time.monotonic()+.23
                while not obstacles and time.monotonic() < deadline:
                    pump(.01)
                assert obstacles and obstacles[-1].width == 1
                assert obstacles[-1].header.stamp == stamp
                count = len(clouds)
                cloud(4, ground, age=1.)
                pump(.15)
                assert len(clouds) == count, 'A stale cloud must not re-enter current observations'
                # A valid ground plane also expires by acquisition time. No
                # empty scan may erase the obstacle when estimation later fails.
                pump(2.05)
                count = len(obstacles)
                cloud(5, [[1., 0., -.4, 1.]])
                pump(.15)
                assert len(obstacles) == count
                assert all(message.width > 0 for message in obstacles)
            finally:
                for process in processes:
                    process.terminate()
                    try:
                        process.wait(timeout=5.)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=2.)
                executor.shutdown()
                node.destroy_node()
                context.shutdown()


def test_fastlio_startup_wait_stall_and_reordered_clouds_keep_current_measurements(monkeypatch):
    import signal
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import Imu
    monkeypatch.setenv('ROS_DOMAIN_ID', '96')
    monkeypatch.setenv('ROS_LOCALHOST_ONLY', '1')
    monkeypatch.setenv('RMW_IMPLEMENTATION', 'rmw_cyclonedds_cpp')
    monkeypatch.setenv('CYCLONEDDS_URI',
        '<CycloneDDS><Domain><General><NetworkInterfaceAddress>lo</NetworkInterfaceAddress>'
        '<AllowMulticast>false</AllowMulticast></General><Discovery><ParticipantIndex>auto</ParticipantIndex>'
        '<Peers><Peer Address="127.0.0.1"/></Peers></Discovery></Domain></CycloneDDS>')
    root = Path(__file__).resolve().parents[3]
    context = Context()
    rclpy.init(context=context)
    node = rclpy.create_node('lio_transport_test', context=context)
    executor = SingleThreadedExecutor(context=context)
    executor.add_node(node)
    imu_pub = node.create_publisher(Imu, '/utlidar/imu_livox_mid360', qos_profile_sensor_data)
    cloud_pub = node.create_publisher(PointCloud2, '/utlidar/cloud_livox_mid360', qos_profile_sensor_data)
    outputs = []
    node.create_subscription(Odometry, '/Odometry', lambda m: outputs.append((time.time(), m)), 20)
    xyz = []
    for a in np.linspace(-2., 2., 16):
        for b in np.linspace(-2., 2., 16):
            for sign in (-2., 2.):
                xyz.extend(((sign, a, b), (a, sign, b), (a, b, sign)))
    points = np.zeros(len(xyz), dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
                                     ('intensity', '<f4'), ('ring', '<u2'), ('time', '<f4')])
    for index, name in enumerate(('x', 'y', 'z')):
        points[name] = np.asarray(xyz)[:, index]
    points['intensity'] = 1
    points['ring'] = np.arange(len(points)) % 4
    points['time'] = np.linspace(0., 99000000., len(points))
    cloud = PointCloud2(height=1, width=len(points), point_step=22, row_step=len(points)*22,
        fields=[PointField(name=name, offset=points.dtype.fields[name][1],
                           datatype=PointField.UINT16 if name == 'ring' else PointField.FLOAT32, count=1)
                for name in points.dtype.names], data=array('B', points.tobytes()))
    cloud.header.frame_id = 'livox_frame'
    last_cloud = [0.]
    def pump(duration, clouds=True, old=False):
        deadline = time.monotonic()+duration
        while time.monotonic() < deadline:
            now = time.time()-62.
            imu = Imu()
            imu.header.stamp = Time(seconds=now).to_msg()
            imu.linear_acceleration.z = 9.81
            imu_pub.publish(imu)
            if clouds and time.monotonic()-last_cloud[0] >= .1:
                last_cloud[0] = time.monotonic()
                cloud.header.stamp = Time(seconds=now-(2. if old else .1)).to_msg()
                cloud_pub.publish(cloud)
            executor.spin_once(timeout_sec=.005)
    process = None
    with tempfile.TemporaryDirectory(prefix='lio-transport-') as directory:
        monkeypatch.setenv('ROS_LOG_DIR', directory)
        with open(Path(directory)/'lio.log', 'w+') as log:
            try:
                process = subprocess.Popen([
                    str(root/'build/fast_lio/fastlio_mapping'), '--ros-args', '--params-file',
                    str(root/'src/unitree_nav2_web/config/fast_lio_mid360_navigation.yaml')],
                    cwd=directory, stdout=log, stderr=log, env=os.environ.copy())
                deadline = time.monotonic()+5.
                while imu_pub.get_subscription_count() < 1 and time.monotonic() < deadline:
                    executor.spin_once(timeout_sec=.01)
                pump(4.3, clouds=False)  # Camera/lidar startup cannot poison IMU history.
                pump(3.)
                log.flush(); log.seek(0)
                assert len(outputs) > 10, log.read()
                before = len(outputs)
                process.send_signal(signal.SIGSTOP)
                pump(.35)
                process.send_signal(signal.SIGCONT)
                pump(1.4)
                assert len(outputs) > before+5
                # Flush any in-flight frame, then deliver only older/reordered clouds.
                pump(.25, clouds=False)
                before = len(outputs)
                pump(.4, old=True)
                assert len(outputs) == before
                pump(.6)
                assert len(outputs) > before+2
                stamps = [msg.header.stamp.sec+msg.header.stamp.nanosec*1e-9 for _, msg in outputs]
                assert all(b > a for a, b in zip(stamps, stamps[1:]))
                assert outputs[-1][0]-62.-stamps[-1] < .5
                assert process.poll() is None
            finally:
                if process is not None:
                    process.send_signal(signal.SIGCONT)
                    process.terminate()
                    try:
                        process.wait(timeout=5.)
                    except subprocess.TimeoutExpired:
                        process.kill(); process.wait(timeout=2.)
                executor.shutdown()
                node.destroy_node()
                context.shutdown()
