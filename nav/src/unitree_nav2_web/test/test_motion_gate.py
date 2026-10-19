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
from std_msgs.msg import Bool, UInt32


PACKET = struct.Struct('<IHBBQfffIQ')


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
                epochs = node.create_publisher(UInt32, '/g1/navigation_epoch', qos)
                references = node.create_publisher(Bool, '/g1/navigation_reference_ready', qos)
                reference_state = {'epoch': 2, 'ready': True, 'publish': True}
                def reference_tick():
                    if reference_state['publish']:
                        epochs.publish(UInt32(data=reference_state['epoch']))
                        references.publish(Bool(data=reference_state['ready']))
                node.create_timer(.04, reference_tick)
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
                    acknowledged = None
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

                    def reference(self, **values):
                        reference_state.update(values)

                    def velocity(self, vx=.4, vy=0., wz=0.):
                        command = Twist()
                        command.linear.x = vx
                        command.linear.y = vy
                        command.angular.z = wz
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
                result.acknowledged = acknowledged
                result.wait(lambda _: enables.get_subscription_count() > 0
                            and commands.get_subscription_count() > 0 and acknowledged)
                result.pump(.2)
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
    stamp = moving[9]
    for _ in range(3):
        gate.enable(True)
        assert all(packet[2] and packet[9] == stamp for packet in gate.pump(.03))
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


def test_cart_epoch_clears_old_commands_and_clamps_all_three_axes(gate):
    gate.enable(True)
    gate.pump(.05)
    gate.velocity()
    gate.wait(lambda items: any(p[2] for p in items))
    gate.reference(epoch=3)
    gate.pump(.15)
    gate.enable(True)
    assert all(not p[2] for p in gate.pump(.10))
    gate.velocity(-.8, .5, -.8)
    packets = gate.wait(lambda items: any(p[2] for p in items))
    active = [p for p in packets if p[2]][-1]
    assert active[5:8] == pytest.approx((-.2, .3, -.4))
    assert active[8] == 3
    gate.velocity(.9, -.5, .8)
    packets = gate.wait(lambda items: any(p[2] and p[5] > 0 for p in items))
    active = [p for p in packets if p[2] and p[5] > 0][-1]
    assert active[5:8] == pytest.approx((.6, -.3, .4))
    gate.reference(publish=False)
    deadline = time.monotonic()+.5
    while time.monotonic() < deadline:
        gate.enable(True)
        gate.velocity()
        gate.pump(.03)
    assert all(not p[2] for p in gate.pump(.10))
    gate.reference(publish=True, ready=False)
    gate.enable(True)
    gate.velocity()
    assert all(not p[2] for p in gate.pump(.10))


def test_short_reference_gap_stops_output_then_requires_new_velocity_without_reopening(gate):
    gate.enable(True)
    gate.pump(.05)
    gate.velocity()
    gate.wait(lambda items: any(p[2] for p in items))
    gate.reference(ready=False)
    gate.wait(lambda items: any(not p[2] for p in items))
    for _ in range(3):
        gate.enable(True)  # Lease renewals must not revoke existing authorization.
        gate.velocity(.5)
        assert all(not p[2] for p in gate.pump(.04))
    assert gate.acknowledged[-1] is True
    gate.reference(ready=True)
    assert all(not p[2] for p in gate.pump(.1))  # Discard velocity received in the gap.
    gate.velocity(.2)
    packets = gate.wait(lambda items: any(p[2] for p in items))
    assert [p for p in packets if p[2]][-1][5] == pytest.approx(.2)


def test_sustained_reference_loss_latches_closed_despite_late_renewals(gate):
    gate.enable(True)
    gate.pump(.05)
    gate.velocity()
    gate.wait(lambda items: any(p[2] for p in items))
    gate.reference(ready=False)
    deadline = time.monotonic()+.8
    while time.monotonic() < deadline:
        gate.enable(True)
        gate.velocity()
        gate.pump(.03)
    assert gate.acknowledged[-1] is False
    gate.reference(ready=True)
    for _ in range(5):
        gate.enable(True)
        gate.velocity()
        assert all(not p[2] for p in gate.pump(.03))
    gate.enable(False)  # Bridge acknowledges cancellation; operator must reopen.
    gate.pump(.05)
    gate.enable(True)
    gate.pump(.05)
    gate.velocity()
    gate.wait(lambda items: any(p[2] for p in items))
