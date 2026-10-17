"""Load the actual Foxy DWB plugins and dynamic footprint on an isolated ROS graph."""
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time

import rclpy
from ament_index_python.packages import get_package_prefix
from geometry_msgs.msg import Point32, Polygon, PolygonStamped, TransformStamped
from lifecycle_msgs.srv import ChangeState
from rclpy.context import Context
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import DurabilityPolicy, QoSProfile
from tf2_ros import StaticTransformBroadcaster


def test_real_controller_loads_cart_plugins_and_uses_cart_footprint(monkeypatch):
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
                executor = SingleThreadedExecutor(context=context)
                executor.add_node(node)
                broadcaster = StaticTransformBroadcaster(node)
                transform = TransformStamped()
                transform.header.stamp = node.get_clock().now().to_msg()
                transform.header.frame_id = 'odom'
                transform.child_frame_id = 'nav_control'
                transform.transform.rotation.w = 1.
                broadcaster.sendTransform(transform)
                footprints = []
                node.create_subscription(PolygonStamped, '/local_costmap/published_footprint',
                                         lambda msg: footprints.append(msg), 10)
                qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
                publisher = node.create_publisher(Polygon, '/local_costmap/footprint', qos)
                binary = Path(get_package_prefix('nav2_controller')) / 'lib/nav2_controller/controller_server'
                params = Path(__file__).resolve().parents[1] / 'config/nav2_foxy_g1.yaml'
                process = subprocess.Popen([str(binary), '--ros-args', '--params-file', str(params)],
                                           stdout=log, stderr=subprocess.STDOUT, env=os.environ.copy())
                client = node.create_client(ChangeState, '/controller_server/change_state')

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
                assert 'CartPush' in text and 'CartLookahead' in text, text
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
