"""Deterministic UDP -> real Controller/ONNX regressions with synthetic feedback.

Run with the CWC interpreter; no DDS, motor process, or on-disk test output.
"""
import json
from itertools import count
import socket
import unittest
from unittest.mock import patch

import numpy as np

from g1_locomotion_executor import (
    COMMAND, COMMAND_EXPIRED, GATE_CLOSED, MOTION_CONFIG, VELOCITY_MAGIC,
    VELOCITY_PACKET, VELOCITY_VERSION, Executor,
    REQUEST_PACKET, REQUEST_MAGIC, REQUEST_VERSION,
)
from core.controller import Controller
from core.robot import RobotConfig
from hybrid_deploy import protocol as p


class Feedback:
    def __init__(self, config):
        self.armed = True
        self.state = p.State(
            42, 1, 0, config.token, p.POLICY, 3, 0, 1,
            config.default_q.copy(), np.zeros(29), np.zeros(3),
            np.array([1., 0, 0, 0]), np.zeros(3), np.zeros(4), config.default_q.copy())
        self.state.hand_flags = 35

    def read(self):
        return self.state

    def send(self, *args, **kwargs):
        pass


class ExecutorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = RobotConfig(motion_config=MOTION_CONFIG)

    def setUp(self):
        self.now = 100.
        self.backend = Feedback(self.config)
        self.control = Controller(self.config, self.backend, 'sim', clock=lambda: self.now)
        self.control.request = p.RUN
        self.sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.status = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.status.bind(('127.0.0.1', 0))
        self.status.setblocking(False)
        self.executor = Executor(self.control, 0, self.status.getsockname()[1], clock=lambda: self.now)
        self.port = self.executor.sock.getsockname()[1]
        self.sequence = 0
        for _ in range(60):
            self.tick()
        self.assertEqual(self.control.snapshot()['owner'], 1)

    def tearDown(self):
        self.executor.close()
        self.sender.close()
        self.status.close()

    def tick(self, velocity=None, *, enabled=True, reason=None, stamp=None):
        # Model the web link continuously consuming status; otherwise a long
        # synthetic run fills the kernel UDP queue and drops the newest report.
        while True:
            try:
                self.status.recv(65535)
            except BlockingIOError:
                break
        if velocity is not None:
            self.sequence += 1
            reason = (COMMAND if enabled else GATE_CLOSED) if reason is None else reason
            stamp = self.now if stamp is None else stamp
            packet = VELOCITY_PACKET.pack(
                VELOCITY_MAGIC, VELOCITY_VERSION, int(enabled), reason, self.sequence,
                *velocity, self.executor.navigation_epoch, round(stamp*1e9))
            self.sender.sendto(packet, ('127.0.0.1', self.port))
        self.executor.step()
        self.now += .02
        self.backend.state.sequence += 1
        self.backend.state.received_ns = round((self.now-.000001)*1e9)
        self.control.step()
        return self.control.snapshot()

    def moving(self, velocity=(.3, 0, 0)):
        for _ in range(300):
            data = self.tick(velocity)
        self.assertEqual(data['selected_model'], 'homie')
        self.assertGreater(data['velocity'][0], .29)
        return data['velocity'].copy()

    def send_request(self, code, payload, sequence):
        packet = REQUEST_PACKET.pack(REQUEST_MAGIC, REQUEST_VERSION, code, payload, sequence, round(self.now*1e9))
        self.sender.sendto(packet, ('127.0.0.1', self.port))

    def waiting_to_prepare(self):
        self.backend.state.stage = p.WAITING
        self.backend.state.hand_flags = 0  # Body preparation is independent of Dex3.
        self.control.request = p.WAIT
        return self.tick()

    def test_udp_pose_selection_changes_no_joint_command_before_preparing(self):
        self.assertTrue(self.waiting_to_prepare()['prepare_pose_selectable'])
        before = self.backend.state.applied_q.copy()
        for sequence, payload, pose in ((1, 1, 'arms_down'), (2, 0, 'original')):
            self.send_request(7, payload, sequence)
            data = self.tick()
            self.assertEqual(data['prepare_pose'], pose)
            self.assertEqual(self.control.request, p.WAIT)
            self.assertEqual(self.control.hand_pose, 0)
            np.testing.assert_array_equal(data['requested_target'], before)

    def test_prepare_locks_pose_even_for_another_selection_queued_in_same_tick(self):
        self.waiting_to_prepare()
        # Real monotonic timestamps distinguish events even within one 20 ms
        # tick; WebControl intentionally rejects duplicate timestamps.
        event_ticks = count()
        self.control.clock = lambda: self.now + next(event_ticks) * 1e-7
        self.send_request(7, 1, 1)
        self.send_request(2, 0, 2)
        self.send_request(7, 0, 3)
        data = self.tick()
        self.assertEqual(self.control.request, p.PREPARE)
        self.assertFalse(data['prepare_pose_selectable'])
        self.assertEqual(data['prepare_pose'], 'arms_down')
        self.assertEqual(self.control.hand_pose, 0)
        np.testing.assert_array_equal(data['requested_target'], self.config.prepare_poses['arms_down'])
        for sequence, stage in enumerate((p.RAMPING, p.HOLDING, p.POLICY), 4):
            self.backend.state.stage = stage
            self.send_request(7, 0, sequence)
            data = self.tick()
            self.assertEqual(data['prepare_pose'], 'arms_down')
            self.assertFalse(data['prepare_pose_selectable'])

    def test_invalid_and_replayed_pose_packets_cannot_change_selection(self):
        self.waiting_to_prepare()
        self.send_request(7, 255, 50)
        self.send_request(2, 1, 50)  # Non-selection requests still require zero payload.
        self.send_request(7, 1, 2)
        self.send_request(7, 0, 1)
        self.send_request(7, 0, 2)
        data = self.tick()
        self.assertEqual(data['prepare_pose'], 'arms_down')
        self.assertEqual(self.control.request, p.WAIT)
        self.assertEqual(self.executor.rejected, 4)

    def test_pose_status_uses_controller_selection_and_ownership(self):
        self.waiting_to_prepare()
        self.send_request(7, 1, 1)
        self.tick()
        self.executor.last_status = -float('inf')
        self.executor.step()
        while True:
            try:
                report = json.loads(self.status.recv(8192))
            except BlockingIOError:
                break
        self.assertEqual(report['prepare_pose'], 'arms_down')
        self.assertTrue(report['prepare_pose_selectable'])
        self.assertEqual({item['value'] for item in report['prepare_pose_options']},
                         set(self.config.prepare_poses))
        data = self.control.snapshot()
        data['owner'] = 99
        self.executor.last_status = -float('inf')
        self.executor._report(data, self.now, False)
        self.assertFalse(json.loads(self.status.recv(8192))['prepare_pose_selectable'])

    def check_stop(self, start, send, previous):
        zero_at = None
        while self.now-start < 1.54:
            data = send()
            velocity = data['velocity']
            self.assertTrue(np.all(abs(velocity) <= abs(previous)+1e-6), data)
            self.assertTrue(np.all(velocity*previous >= 0), data)
            if data['decelerating']:
                self.assertFalse(data['heading']['correction_enabled'])
            if not np.any(velocity) and zero_at is None:
                zero_at = self.now
                self.assertLessEqual(zero_at-start, 1.1)
                self.assertGreaterEqual(zero_at-start, .68)
            if zero_at is None:
                self.assertEqual(data['selected_model'], 'homie')
            else:
                self.assertEqual(data['selected_model'], 'balance')
            previous = velocity.copy()
        self.assertIsNotNone(zero_at)
        self.assertEqual(data['selected_model'], 'balance')
        np.testing.assert_array_equal(data['velocity'], 0)

    def test_packet_loss_filters_then_hands_over_without_dwell(self):
        previous = self.moving()
        self.check_stop(self.executor.last_motion, self.tick, previous)

    def test_retransmitting_cached_cmd_vel_cannot_renew_motion(self):
        previous = self.moving()
        stamp = self.executor.last_motion
        self.check_stop(stamp, lambda: self.tick((.3, 0, 0), stamp=stamp), previous)

    def test_gate_close_stops_smoothly_and_repeated_packets_do_not_extend_it(self):
        previous = self.moving((.3, .1, .2))
        self.check_stop(self.now, lambda: self.tick((0, 0, 0), enabled=False), previous)

    def test_upstream_timeout_uses_original_time_not_time_of_disabled_packet(self):
        previous = self.moving()
        stamp = self.executor.last_motion
        while self.now-stamp < .26:
            self.tick()
        self.check_stop(stamp, lambda: self.tick((0, 0, 0), enabled=False,
                                               reason=COMMAND_EXPIRED, stamp=stamp), previous)

    def test_zero_command_and_commands_queued_during_ramp_cannot_revive_motion(self):
        previous = self.moving()
        start = self.now

        def send():
            elapsed = self.now-start
            if elapsed < .3:
                return self.tick((0, 0, 0))
            if elapsed < .6:
                return self.tick((.35, .1, .2))
            return self.tick()

        self.check_stop(start, send, previous)
        for _ in range(25):
            data = self.tick((.3, 0, 0))
        self.assertEqual(data['selected_model'], 'homie')
        self.assertGreater(data['velocity'][0], 0)

    def test_new_nonzero_command_after_stop_crosses_threshold_to_walk(self):
        self.moving()
        self.tick((0, 0, 0))
        while self.control.snapshot()['decelerating']:
            self.tick()
        for _ in range(10):
            self.tick()
        for _ in range(30):
            data = self.tick((.3, 0, 0))
        self.assertEqual(data['selected_model'], 'homie')
        self.assertGreater(data['velocity'][0], .1)

    def test_delayed_enabled_packet_before_stop_cannot_restart_motion(self):
        self.moving()
        stale = self.now-.1
        self.executor.request('zero')
        for _ in range(90):
            data = self.tick((.3, 0, 0), stamp=stale)
        np.testing.assert_array_equal(data['velocity'], 0)
        self.assertEqual(data['selected_model'], 'balance')

    def test_unseen_preclosure_velocity_cannot_start_a_stationary_policy(self):
        # A command delayed in transit can be newer than the last accepted
        # command, yet older than the gate closure. It must still be discarded.
        delayed_stamp = self.now - .01
        self.tick((0, 0, 0), enabled=False)
        for _ in range(20):
            data = self.tick((.3, 0, 0), stamp=delayed_stamp)
            np.testing.assert_array_equal(data['velocity'], 0)
            self.assertEqual(data['selected_model'], 'balance')
        self.assertFalse(self.executor.enabled)
        for _ in range(30):
            data = self.tick((.3, 0, 0))
        self.assertGreater(data['velocity'][0], .1)

    def test_decelerating_is_healthy_policy_not_a_navigation_fault(self):
        self.moving()
        self.tick((0, 0, 0))
        self.executor.last_status = -float('inf')
        self.executor.step()
        received = []
        while True:
            try:
                received.append(json.loads(self.status.recv(8192)))
            except BlockingIOError:
                break
        self.assertTrue(received[-1]['decelerating'])
        self.assertTrue(received[-1]['ready'])

    def test_explicit_damping_overrides_smooth_stop(self):
        self.moving()
        self.tick((0, 0, 0))
        self.assertTrue(self.control.snapshot()['decelerating'])
        self.executor.request('stop')
        data = self.tick()
        self.assertEqual(data['stage'], 'DAMPED')
        np.testing.assert_array_equal(data['velocity'], 0)

    def test_heading_recovery_closes_navigation_readiness_but_keeps_policy_running(self):
        self.moving()
        self.now += .12
        data = self.tick()
        self.assertTrue(data['heading']['motion_blocked'])
        self.executor.step()
        reports = []
        while True:
            try:
                reports.append(json.loads(self.status.recv(65535)))
            except BlockingIOError:
                break
        self.assertEqual(reports[-1]['stage'], 'POLICY')
        self.assertFalse(reports[-1]['ready'])
        self.assertIn('站立策略继续', reports[-1]['reason'])
        self.assertIn('policy_hz', reports[-1]['timing'])
        previous = self.control.policy_steps
        for _ in range(80):data = self.tick()
        self.assertEqual(data['stage'], 'POLICY')
        self.assertEqual(data['selected_model'], 'balance')
        self.assertGreater(self.control.policy_steps, previous+50)
        np.testing.assert_array_equal(data['velocity'], 0)

    def test_udp_burst_uses_newest_command_in_one_tick(self):
        for index in range(180):
            packet = VELOCITY_PACKET.pack(VELOCITY_MAGIC, VELOCITY_VERSION, 1, COMMAND,
                                         index+1000, index*.001, 0., 0., self.executor.navigation_epoch,
                                         round((self.now-.1+index*.0005)*1e9))
            self.sender.sendto(packet, ('127.0.0.1', self.port))
        self.executor.step()
        self.assertAlmostEqual(float(self.executor.target[0]), .179, places=6)
        self.assertAlmostEqual(self.executor.last_motion, self.now-.0105)

    def test_reordered_gate_packet_cannot_erase_newer_velocity(self):
        self.tick((.2, 0., 0.))
        packet = VELOCITY_PACKET.pack(VELOCITY_MAGIC, VELOCITY_VERSION, 0, GATE_CLOSED,
                                     self.sequence-1, 0., 0., 0., self.executor.navigation_epoch,
                                     round((self.now-.03)*1e9))
        self.sender.sendto(packet, ('127.0.0.1', self.port))
        self.executor.step()
        self.assertTrue(self.executor.enabled)

    def test_expired_operator_request_cannot_change_prepare_pose(self):
        self.waiting_to_prepare()
        packet = REQUEST_PACKET.pack(REQUEST_MAGIC, REQUEST_VERSION, 7, 1, 100,
                                     round((self.now-.6)*1e9))
        self.sender.sendto(packet, ('127.0.0.1', self.port))
        data = self.tick()
        self.assertEqual(data['prepare_pose'], 'original')
        self.assertEqual(self.executor.rejected, 1)

    def test_frozen_controller_snapshot_cannot_report_navigation_ready(self):
        self.now += .3
        self.executor.step()
        while True:
            try:
                report = json.loads(self.status.recv(65535))
            except BlockingIOError:
                break
        self.assertFalse(report['ready'])
        self.assertGreater(report['feedback_age_s'], .25)

    def test_warning_formatter_failure_cannot_stop_the_body_controller(self):
        self.moving()
        with patch('g1_locomotion_executor.live_warnings', side_effect=ValueError('bad display data')):
            self.executor.last_status = -float('inf')
            data = self.tick((.3, 0, 0))
            warnings = self.executor._warnings(data, self.now)
        self.assertEqual(data['stage'], 'POLICY')
        self.assertTrue(data['healthy'])
        self.assertGreater(data['velocity'][0], .29)
        self.assertIn('告警显示暂时不可用', warnings[0]['message'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
