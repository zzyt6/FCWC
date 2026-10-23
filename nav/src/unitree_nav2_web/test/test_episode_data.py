"""Offline recording regressions: real messages and loopback episode control, no ROS graph."""
import csv
import json
from pathlib import Path
import socket
import tempfile
import threading
import time
from types import SimpleNamespace
from unittest.mock import patch

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

from unitree_nav2_web.episode_data import FILES, RosEpisodeRecorder
from unitree_nav2_web.nav_web_bridge import NavigationWebBridge
from core.data_episode import decode_ack, encode
from test_navigation_safety import _Harness


def wait_for(predicate):
    deadline = time.monotonic() + 5
    while not predicate() and time.monotonic() < deadline:
        time.sleep(.005)
    assert predicate()


def announce(sender, recorder, seq, directory=None, start=float('nan')):
    sender.sendto(encode(seq, directory, start), recorder.sock.getsockname())
    sender.settimeout(5)
    ack = decode_ack(sender.recv(512))
    assert ack['seq'] == seq
    assert ack['directory'] == (str(directory) if directory is not None else None)
    return ack


def odometry():
    msg = Odometry()
    msg.header.stamp.sec, msg.header.stamp.nanosec = 123, 250000000
    msg.header.frame_id, msg.child_frame_id = 'odom', 'base_nav'
    msg.pose.pose.position.x, msg.pose.pose.position.y = 1.25, -2.5
    msg.pose.pose.orientation.w = 1.
    msg.twist.twist.linear.x = .42
    msg.twist.twist.angular.z = -.17
    return msg


def harness(enabled=True, port=0):
    node = _Harness()
    node.data_recorder = None
    node.logger.info = lambda *_: None
    parameters = dict(pushcart_data=enabled, pushcart_data_port=port)
    node.get_parameter = lambda key: SimpleNamespace(value=parameters[key])
    subscriptions = {}
    node.create_subscription = lambda kind, topic, callback, qos: subscriptions.setdefault(topic, callback)
    return node, subscriptions


def test_existing_node_records_all_three_streams_with_original_callback_times(monkeypatch):
    with tempfile.TemporaryDirectory(prefix='nav_episode_data_') as directory, socket.socket(
            socket.AF_INET, socket.SOCK_DGRAM) as sender:
        node, callbacks = harness()
        with patch('subprocess.Popen', side_effect=AssertionError('Unexpected process')):
            NavigationWebBridge._start_data_recorder(node)
        recorder = node.data_recorder
        assert set(callbacks) == {'/Odometry', '/cmd_vel'}  # base odometry reuses the existing subscription
        try:
            # Old odometry is still useful for diagnostics even when the UI
            # rejects its timestamp. It must not refresh the navigation state.
            monkeypatch.setattr('unitree_nav2_web.nav_web_bridge.sample_time', lambda *a, **k: None)
            previous_odom_time = node.last_odom_time
            before = time.monotonic()
            NavigationWebBridge._odom_callback(node, odometry())
            after = time.monotonic()
            assert node.last_odom_time == previous_odom_time
            callbacks['/Odometry'](odometry())
            start = time.monotonic()
            announce(sender, recorder, 1, directory, start)
            command = Twist()
            command.linear.x, command.linear.y, command.angular.z = .37, -.08, .19
            callbacks['/cmd_vel'](command)
            # Invalid feedback is recorded as received, while the existing
            # navigation fault response still executes.
            invalid = odometry()
            invalid.pose.pose.position.x = float('nan')
            NavigationWebBridge._odom_callback(node, invalid)
            assert node.faults
            wait_for(lambda: sum(recorder.file_rows.values()) == 4)
            time.sleep(.06)
            assert sum(recorder.file_rows.values()) == 4  # no timer-driven duplicate samples
            announce(sender, recorder, 2)
        finally:
            recorder.close()
        assert not recorder.thread.is_alive()
        tables = {}
        for name, columns in FILES.items():
            with (Path(directory) / name).open() as stream:
                reader = csv.DictReader(stream)
                assert tuple(reader.fieldnames) == columns
                tables[name] = list(reader)
        base = tables['odometry_base.csv']
        assert len(base) == 2
        assert before - 1e-6 <= float(base[0]['t']) <= after + 1e-6
        assert float(base[0]['t_rel']) <= 0
        assert base[0]['stamp_t'] == '123.250000'
        assert (base[0]['frame_id'], base[0]['child_frame_id']) == ('odom', 'base_nav')
        assert (base[0]['x'], base[0]['vx'], base[0]['wz']) == ('1.25', '0.42', '-0.17')
        assert base[1]['x'] == 'nan'
        command, = tables['cmd_vel.csv']
        assert (command['vx'], command['vy'], command['wz']) == ('0.37', '-0.08', '0.19')
        summary = json.loads((Path(directory) / 'ros_summary.json').read_text())
        assert summary['source'] == 'navigation_web_bridge'
        assert summary['file_rows'] == {name: len(rows) for name, rows in tables.items()}
        assert summary['rows'] == 4 and summary['dropped_before_write'] == 0
        assert summary['writer_error'] == ''


def test_full_writer_queue_drops_samples_without_blocking_callbacks():
    with tempfile.TemporaryDirectory(prefix='nav_episode_queue_') as directory, socket.socket(
            socket.AF_INET, socket.SOCK_DGRAM) as sender:
        recorder = RosEpisodeRecorder(0)
        blocked, release = threading.Event(), threading.Event()
        writer_threads = set()
        original_write = recorder.write

        def delayed_write(t, sample):
            writer_threads.add(threading.get_ident())
            blocked.set()
            assert release.wait(5)
            original_write(t, sample)

        recorder.write = delayed_write
        try:
            announce(sender, recorder, 1, directory, time.monotonic())
            recorder.cmd_vel(Twist())
            assert blocked.wait(3)
            for _ in range(recorder.pending.maxsize + 5):
                recorder.cmd_vel(Twist())
            assert recorder.dropped == 5
            assert recorder.pending.qsize() == recorder.pending.maxsize
            release.set()
            wait_for(lambda: recorder.file_rows['cmd_vel.csv'] == recorder.pending.maxsize + 1)
            assert writer_threads == {recorder.thread.ident}
            assert threading.get_ident() not in writer_threads
        finally:
            release.set()
            recorder.close()
        summary = json.loads((Path(directory) / 'ros_summary.json').read_text())
        assert summary['dropped_before_write'] == 5


def test_disk_error_stays_in_writer_and_later_samples_can_be_recorded():
    with tempfile.TemporaryDirectory(prefix='nav_episode_error_') as directory, socket.socket(
            socket.AF_INET, socket.SOCK_DGRAM) as sender:
        recorder = RosEpisodeRecorder(0)
        original_write = recorder.write

        def failed_write(*_):
            raise OSError('test disk failure')

        try:
            announce(sender, recorder, 1, directory, time.monotonic())
            recorder.write = failed_write
            recorder.cmd_vel(Twist())
            wait_for(lambda: 'test disk failure' in recorder.error)
            recorder.write = original_write
            recorder.cmd_vel(Twist())
            wait_for(lambda: recorder.file_rows['cmd_vel.csv'] == 1)
        finally:
            recorder.close()
        summary = json.loads((Path(directory) / 'ros_summary.json').read_text())
        assert 'test disk failure' in summary['writer_error']


def test_disabled_or_unavailable_recorder_does_not_add_subscriptions():
    node, callbacks = harness(enabled=False)
    NavigationWebBridge._start_data_recorder(node)
    assert node.data_recorder is None and not callbacks
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as occupied:
        occupied.bind(('127.0.0.1', 0))
        node, callbacks = harness(port=occupied.getsockname()[1])
        NavigationWebBridge._start_data_recorder(node)
        assert node.data_recorder is None and not callbacks
        assert 'recorder unavailable' in node.logger.errors[-1]
