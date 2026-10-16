"""Real ROS velocity-gate regressions, isolated to loopback with no robot process."""
import os
from pathlib import Path
import signal
import socket
import struct
import subprocess
import tempfile
import time

import pytest
import rclpy
from ament_index_python.packages import get_package_prefix
from geometry_msgs.msg import Twist
from rclpy.context import Context
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool


PACKET = struct.Struct('<IHBBQfff4xQ')


@pytest.fixture
def gate(monkeypatch):
    monkeypatch.setenv('ROS_DOMAIN_ID', '88')
    monkeypatch.setenv('ROS_LOCALHOST_ONLY', '1')
    monkeypatch.setenv('RMW_IMPLEMENTATION', 'rmw_cyclonedds_cpp')
    monkeypatch.setenv('CYCLONEDDS_URI',
        '<CycloneDDS><Domain><General><NetworkInterfaceAddress>lo</NetworkInterfaceAddress>'
        '<AllowMulticast>false</AllowMulticast></General><Discovery><ParticipantIndex>auto</ParticipantIndex>'
        '<Peers><Peer Address="127.0.0.1"/></Peers></Discovery></Domain></CycloneDDS>')
    receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receiver.bind(('127.0.0.1', 0))
    receiver.setblocking(False)
    context = Context()
    process = node = executor = None
    with tempfile.TemporaryDirectory(prefix='nav_gate_test_') as directory:
        monkeypatch.setenv('ROS_LOG_DIR', directory)
        with open(Path(directory) / 'gate.log', 'w+') as log:
            try:
                rclpy.init(context=context)
                node = rclpy.create_node('isolated_gate_regression', context=context)
                executor = SingleThreadedExecutor(context=context)
                executor.add_node(node)
                qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                                 durability=DurabilityPolicy.TRANSIENT_LOCAL)
                enables = node.create_publisher(Bool, '/g1/motion_enable', qos)
                commands = node.create_publisher(Twist, '/cmd_vel', 1)
                acknowledged = []
                node.create_subscription(Bool, '/g1/motion_enabled',
                                         lambda msg: acknowledged.append(msg.data), qos)
                binary = Path(get_package_prefix('unitree_g1_kinematics')) / 'lib/unitree_g1_kinematics/g1_cmd_vel_udp_node'
                process = subprocess.Popen([
                    str(binary), '--ros-args', '-p', f'udp_port:={receiver.getsockname()[1]}',
                    '-p', 'max_vx:=0.6', '-p', 'motion_enable_timeout:=0.4',
                ], stdout=log, stderr=subprocess.STDOUT, env=os.environ.copy())

                class Gate:
                    def pump(self, duration):
                        packets = []
                        deadline = time.monotonic() + duration
                        while time.monotonic() < deadline:
                            executor.spin_once(timeout_sec=.005)
                            while True:
                                try:
                                    packets.append(PACKET.unpack(receiver.recv(1024)))
                                except BlockingIOError:
                                    break
                        return packets

                    def enable(self, enabled):
                        enables.publish(Bool(data=enabled))

                    def velocity(self, vx=.4):
                        command = Twist()
                        command.linear.x = vx
                        commands.publish(command)

                    def wait(self, predicate, timeout=3.):
                        deadline = time.monotonic() + timeout
                        while time.monotonic() < deadline:
                            packets = self.pump(.01)
                            if predicate(packets):
                                return packets
                        log.flush()
                        log.seek(0)
                        raise AssertionError('ROS gate condition timed out: ' + log.read())

                result = Gate()
                result.wait(lambda _: enables.get_subscription_count() > 0
                            and commands.get_subscription_count() > 0 and acknowledged)
                yield result
            finally:
                if process is not None and process.poll() is None:
                    process.send_signal(signal.SIGINT)
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
                    context.shutdown()
                receiver.close()


def test_reopen_discards_cached_velocity_but_renewal_preserves_fresh_command(gate):
    gate.enable(False)
    for _ in range(4):
        gate.velocity()
        assert all(not packet[2] for packet in gate.pump(.02))
    gate.enable(True)
    packets = gate.pump(.13)
    assert packets and all(not packet[2] for packet in packets), packets
    gate.velocity()
    packets = gate.wait(lambda items: any(packet[2] for packet in items))
    moving = [packet for packet in packets if packet[2]][-1]
    assert moving[5] == pytest.approx(.4)
    stamp = moving[8]
    for _ in range(3):
        gate.enable(True)
        assert all(packet[2] and packet[8] == stamp for packet in gate.pump(.03))
    # Authorizing motion again must not refresh the original command's age.
    gate.wait(lambda items: any(not packet[2] and packet[3] == 2 for packet in items))


def test_close_and_lost_authorization_block_continuing_velocity(gate):
    gate.enable(True)
    gate.pump(.03)
    gate.velocity()
    gate.wait(lambda items: any(packet[2] for packet in items))
    gate.enable(False)
    gate.wait(lambda items: any(not packet[2] and packet[3] == 1 for packet in items))
    for _ in range(4):
        gate.velocity()
        assert all(not packet[2] for packet in gate.pump(.03))
    gate.enable(True)
    gate.pump(.03)
    gate.velocity()
    gate.wait(lambda items: any(packet[2] for packet in items))
    deadline = time.monotonic() + .6
    expired = False
    while time.monotonic() < deadline:
        gate.velocity()
        for packet in gate.pump(.02):
            if not packet[2] and packet[3] == 1:
                expired = True
            if expired:
                assert not packet[2]
    assert expired
