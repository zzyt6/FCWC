"""Migration regressions using real ONNX policies and synthetic feedback only."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import time
import unittest
from unittest.mock import Mock
import numpy as np

from core.robot import ACTIVE, RobotConfig
from core.controller import Controller
from core.backends import Bridge
from test_heading import quaternion
from core.pelvis_imu import ImuFault
from hybrid_deploy import protocol as p


class FakeBridge:
    def __init__(self, config):
        self.armed = False
        self.started = time.monotonic()
        self.state = p.State(42, 1, time.monotonic_ns(), config.token, p.WAITING, 1, 0, 1,
                             config.default_q.copy(), np.zeros(29), np.zeros(3), np.array([1.,0,0,0]),
                             np.zeros(3), np.zeros(4), config.default_q.copy())
        self.state.hand_flags = 35
        self.writes = []
        self.starts = []

    def read(self):
        return self.state

    def send(self, request, q, profile=0, hand_pose=0, hand_recover=-1, hand_grasp=None, hand_live=False,
             upper_mode=0, upper_target_active=False):
        if self.armed:
            self.writes.append((request, q.copy(), profile, hand_pose))
            self.recovery_request = hand_recover
            self.grasp_request = None if hand_grasp is None else hand_grasp.copy()
            self.live_request = hand_live
            self.state.upper_mode = upper_mode
            self.upper_target_active = upper_target_active
            self.state.upper_gain_blend = 1.

    def start(self, armed=False):
        self.armed = armed
        self.starts.append(armed)
        self.state.flags = 3 if armed else 1
        self.state.stage = p.WAITING
        self.state.received_ns = time.monotonic_ns()


class ControlTests(unittest.TestCase):
    def setUp(self):
        self.now = 100.
        self.config = RobotConfig(heading_mode='active')
        self.bridge = FakeBridge(self.config)
        self.control = Controller(self.config, self.bridge, 'sim', clock=lambda:self.now)
        for _ in range(20): self.step()

    def step(self):
        self.now += .02
        self.bridge.state.received_ns = int((self.now-.000001)*1e9)
        self.bridge.state.sequence += 1
        self.control.step()

    def event(self, kind, client=1, velocity=None, **payload):
        self.control.emit(kind, client, velocity, **payload)
        self.step()

    def finish_zero(self):
        for _ in range(250):
            if not self.control.zeroing:
                break
            self.step()
        self.assertFalse(self.control.zeroing)
        np.testing.assert_array_equal(self.control.snapshot()['velocity'], 0)

    def test_shadow_warmup_does_not_commit_unexecuted_actions(self):
        for model in self.control.observations.models.values():
            np.testing.assert_array_equal(model.previous_action,0)

    def test_command_latency_includes_feedback_age_inference_and_send(self):
        self.active()
        infer = self.control.policy.infer
        send = self.bridge.send

        def timed_infer(name, observations):
            self.now += .005
            return infer(name, observations)

        def timed_send(*args, **kwargs):
            self.now += .003
            return send(*args, **kwargs)

        self.control.policy.infer = timed_infer
        self.bridge.send = timed_send
        self.step()
        timing = self.control.snapshot()['timing']
        self.assertAlmostEqual(timing['inference_ms'], 10., places=5)
        self.assertAlmostEqual(timing['state_age_ms'], .001, places=5)
        self.assertAlmostEqual(timing['state_to_command_ms'], 13.001, places=5)

    def test_model_and_estimator_outputs_align_with_feedback_and_clear_on_stop(self):
        self.active()
        data = self.control.snapshot()
        self.assertEqual(data['model_actions']['balance'].shape, (12,))
        self.assertEqual(data['model_actions']['homie'].shape, (12,))
        self.assertEqual(data['homie_estimator']['velocity_raw'].shape, (3,))
        self.assertEqual(data['homie_estimator']['latent_normalized'].shape, (32,))
        self.assertEqual(data['policy_inference_monotonic'], self.now)
        self.assertEqual(data['state']['tick'], (self.bridge.state.session, self.bridge.state.sequence))
        for name in ('balance', 'homie'):
            history = self.control.observations.models[name].history.copy()
            history[-1,:3] = (0 if name == 'balance' else data['filtered_velocity'] * [2,2,.5])
            # Histories are the exact input of this tick; previous action is only
            # updated after inference, without rewriting the stored history.
            actual = self.control.policy.infer(name, history.reshape(1, -1))
            np.testing.assert_allclose(data['model_actions'][name], actual, atol=1e-6)
        self.bridge.state.stage = p.RAMPING
        self.control.request = p.PREPARE
        self.step()
        self.assertIsNotNone(self.control.snapshot()['model_actions'])
        self.assertIsNotNone(self.control.snapshot()['homie_estimator'])
        for model in self.control.observations.models.values():
            np.testing.assert_array_equal(model.previous_action, 0)
        self.event('stop')
        self.step()
        for key in ('model_actions', 'homie_estimator', 'policy_inference_monotonic'):
            self.assertIsNone(self.control.snapshot()[key])

    def test_bundle_and_single_waist_display(self):
        self.assertEqual(len(ACTIVE), 27)
        self.assertEqual(self.control.policy.selected, 'balance')
        self.assertTrue(all(path.is_file() for path in self.config.policy_paths.values()))
        self.assertEqual(set(self.control.policy.sessions), {'balance', 'homie'})

    def test_acquire_requires_owner_health_and_order(self):
        self.event('acquire')
        self.assertFalse(self.bridge.starts)
        self.event('claim')
        self.event('prepare')
        self.assertEqual(self.control.request, p.WAIT)
        self.event('acquire', client=2)
        self.assertFalse(self.bridge.starts)
        self.event('acquire')
        self.step()
        self.assertEqual(self.control.stage, 'ZERO')
        self.event('policy')
        self.assertEqual(self.control.request, p.WAIT)
        self.event('prepare')
        self.assertEqual(self.control.request, p.PREPARE)

    def test_prepare_sends_named_upper_targets_without_enabling_hands(self):
        import yaml
        from core.robot import UPPER, JOINT_NAMES
        pose = yaml.safe_load(self.config.upper_pose_path.read_text())
        self.event('claim'); self.event('acquire'); self.step()
        self.control.hand_pose = 2
        self.event('prepare')
        request, q, _, hand_pose = self.bridge.writes[-1]
        self.assertEqual((request, hand_pose), (p.PREPARE, 0))
        np.testing.assert_array_equal(q[:12], self.config.default_q[:12])
        np.testing.assert_array_equal(q[13:15], 0)
        for index in UPPER:
            self.assertEqual(q[index], pose['hold_target_q'][JOINT_NAMES[index]])
        self.assertNotEqual(q[20], self.bridge.state.applied_q[20])

    def test_prepare_allows_missing_hand_feedback(self):
        self.event('claim'); self.event('acquire'); self.step()
        self.bridge.state.hand_flags = 33
        self.event('prepare')
        self.assertEqual(self.control.request, p.PREPARE)
        self.assertEqual(self.control.reason, '')

    def test_pose_choice_requires_owner_and_waiting_and_does_not_start_motion(self):
        self.assertEqual(self.control.prepare_pose, 'original')
        self.event('prepare_pose', pose='arms_down')
        self.assertEqual(self.control.prepare_pose, 'original')
        self.event('claim')
        for client, pose in ((2, 'arms_down'), (1, '../other.yaml'), (1, ['arms_down'])):
            self.event('prepare_pose', client=client, pose=pose)
            self.assertEqual(self.control.prepare_pose, 'original')
        self.event('prepare_pose', pose='arms_down')
        self.assertEqual(self.control.prepare_pose, 'arms_down')
        self.assertEqual(self.control.request, p.WAIT)
        self.assertEqual(self.control.hand_pose, 0)
        self.assertFalse(self.bridge.starts)
        self.assertFalse(self.bridge.writes)
        self.event('acquire'); self.step()
        self.assertEqual(self.control.prepare_pose, 'arms_down')
        self.assertTrue(self.control.snapshot()['prepare_pose_selectable'])
        self.event('prepare')
        request, q, _, hand_pose = self.bridge.writes[-1]
        self.assertEqual((request, hand_pose), (p.PREPARE, 0))
        np.testing.assert_array_equal(q, self.config.prepare_poses['arms_down'])
        np.testing.assert_array_equal(q[:12], self.config.default_q[:12])
        # Even before the bridge acknowledges PREPARE, changing targets is locked.
        self.assertEqual(self.bridge.state.stage, p.WAITING)
        self.assertFalse(self.control.snapshot()['prepare_pose_selectable'])
        self.event('prepare_pose', pose='original')
        self.assertEqual(self.control.prepare_pose, 'arms_down')
        for stage in (p.RAMPING, p.HOLDING, p.POLICY):
            self.bridge.state.stage = stage
            self.event('prepare_pose', pose='original')
            self.assertEqual(self.control.prepare_pose, 'arms_down')
            self.assertFalse(self.control.snapshot()['prepare_pose_selectable'])

    def test_stale_pose_choice_is_rejected_and_stop_takes_priority(self):
        self.event('claim')
        self.control.emit('prepare_pose', 1, pose='arms_down', command_time=self.now - .5)
        self.step()
        self.assertEqual(self.control.prepare_pose, 'original')
        self.control.emit('prepare_pose', 1, pose='arms_down')
        self.control.emit('stop', 1)
        self.step()
        self.assertEqual(self.control.stage, 'DAMPED')
        self.assertEqual(self.control.prepare_pose, 'original')
        self.assertFalse(self.control.snapshot()['prepare_pose_selectable'])

    def test_pose_choice_is_disabled_in_observe_mode(self):
        self.control = Controller(self.config, self.bridge, 'observe', clock=lambda: self.now)
        self.step()
        self.event('claim')
        self.event('prepare_pose', pose='arms_down')
        self.assertEqual(self.control.prepare_pose, 'original')
        self.assertFalse(self.control.snapshot()['prepare_pose_selectable'])
        self.assertFalse(self.bridge.starts)

    def test_preparation_warning_uses_selected_pose(self):
        from core.robot import JOINT_NAMES
        self.event('claim')
        self.event('prepare_pose', pose='arms_down')
        state = self.bridge.state
        state.stage = p.RAMPING
        state.flags |= p.PREPARATION_OVERDUE
        state.q = self.config.prepare_poses['arms_down'].copy()
        state.applied_q = state.q.copy()
        state.q[18] -= .2
        warning = self.control._preparation_warning(state)
        self.assertIn('left_elbow_joint', warning)
        for name in JOINT_NAMES:
            if name != 'left_elbow_joint':
                self.assertNotIn(name, warning)

    def test_dex3_toggle_requires_owner_ready_health_and_finished_ramp(self):
        self.event('claim')
        self.event('hand_toggle')
        self.assertEqual(self.control.hand_pose, 0)
        self.bridge.armed = True
        self.bridge.state.stage = p.HOLDING
        self.bridge.state.hand_flags = 39
        self.step()
        self.event('hand_toggle', client=2)
        self.assertEqual(self.control.hand_pose, 0)
        self.event('hand_toggle')
        self.assertEqual(self.control.hand_pose, 2)
        self.assertEqual(self.bridge.writes[-1][-1], 2)
        self.event('hand_toggle')  # Awaiting bridge acknowledgment.
        self.assertEqual(self.control.hand_pose, 2)
        self.bridge.state.hand_flags = 63  # Grasp ramp in progress.
        self.event('hand_toggle')
        self.assertEqual(self.control.hand_pose, 2)
        self.bridge.state.hand_flags = 55
        self.bridge.state.buttons = p.B
        self.event('hand_toggle')
        self.assertEqual(self.control.hand_pose, 2)
        self.bridge.state.buttons = 0
        self.event('hand_toggle')
        self.assertEqual(self.control.hand_pose, 1)
        self.event('stop'); count = len(self.bridge.writes)
        self.event('hand_toggle')
        self.assertEqual(len(self.bridge.writes), count)

    def test_reclaim_after_stop_restarts_read_only_then_requires_acquire(self):
        for mode in ('control', 'sim'):
            with self.subTest(mode=mode):
                self.setUp()
                self.control.mode = mode
                self.active()
                self.event('stop')
                self.now += 2  # Old bridge feedback is stale after damping exits.
                self.control.emit('claim', 1)
                self.control.step()
                self.assertEqual(self.bridge.starts[-1], False)
                self.assertFalse(self.control.stop_requested.is_set())
                self.step()
                self.assertEqual(self.control.stage, 'WAITING')
                self.assertEqual(self.control.request, p.WAIT)
                np.testing.assert_array_equal(self.control.snapshot()['velocity'], 0)
                self.event('policy')
                self.assertEqual(self.control.request, p.WAIT)
                self.event('acquire')
                self.step()
                self.assertEqual(self.control.stage, 'ZERO')
                self.event('prepare')
                self.assertEqual(self.control.request, p.PREPARE)

    def test_compensation_checkbox_owner_busy_and_toggle_targets(self):
        self.event('claim')
        self.event('hand_compensation', enabled=True)
        self.assertFalse(self.control.hand_compensation)  # Must acquire first.
        self.event('acquire'); self.step()
        self.event('hand_compensation', client=2, enabled=True)
        self.assertFalse(self.control.hand_compensation)
        self.event('hand_compensation', enabled=True)
        self.assertTrue(self.control.hand_compensation)
        self.event('prepare')
        self.assertEqual(self.bridge.writes[-1][-1], 0)
        self.bridge.state.stage = p.HOLDING
        self.bridge.state.hand_flags = 103  # Compensated open, stationary.
        self.event('hand_toggle')
        self.assertEqual(self.bridge.writes[-1][-1], 4)
        self.event('hand_compensation', enabled=False)  # Pending acknowledgment.
        self.assertTrue(self.control.hand_compensation)
        self.bridge.state.hand_flags = 127  # Compensated grasp moving.
        self.event('hand_compensation', enabled=False)
        self.assertTrue(self.control.hand_compensation)
        self.bridge.state.hand_flags = 119
        self.event('hand_toggle')
        self.assertEqual(self.bridge.writes[-1][-1], 3)
        self.bridge.state.hand_flags = 103
        self.event('hand_compensation', enabled=False)
        self.assertFalse(self.control.hand_compensation)
        self.event('hand_toggle')
        self.assertEqual(self.bridge.writes[-1][-1], 2)
        self.bridge.state.hand_flags = 55
        self.event('hand_toggle')
        self.assertEqual(self.bridge.writes[-1][-1], 1)
        self.bridge.state.buttons = p.L2
        self.event('hand_compensation', enabled=True)
        self.assertEqual(self.control.stage, 'DAMPED')
        self.assertEqual(self.bridge.writes[-1][0], p.STOP)

    def test_invalid_compensation_file_and_old_buttons_cannot_move_hand(self):
        self.event('claim'); self.event('acquire'); self.step()
        self.config.hand_grasp_compensated = None
        self.config.hand_grasp_compensation_error = 'invalid grasp + offset'
        self.event('hand_compensation', enabled=True)
        self.assertFalse(self.control.hand_compensation)
        self.assertIn('invalid grasp', self.control.reason)
        self.bridge.state.stage = p.HOLDING
        self.bridge.state.hand_flags = 39
        for kind in ('hand_zero_compensated', 'hand_grasp_compensated'):
            self.event(kind)
            self.assertEqual(self.control.hand_pose, 0)
        self.event('hand_toggle')
        self.assertEqual(self.control.hand_pose, 2)

    def test_recovery_only_for_owner_before_prepare_and_selected_eligible_joint(self):
        self.event('claim'); self.event('acquire'); self.step()
        self.bridge.state.hand_flags = 33
        self.bridge.state.hand_recovery_mask = 1 << 8
        self.event('hand_recover', client=2, joint=8)
        self.assertEqual(self.control.hand_recovery, -1)
        self.event('hand_recover', joint=7)
        self.assertEqual(self.control.hand_recovery, -1)
        self.event('hand_recover', joint=8)
        self.assertEqual(self.bridge.recovery_request, 8)
        self.event('prepare')
        self.assertEqual(self.control.request, p.PREPARE)
        self.assertEqual(self.control.hand_recovery, -1)
        self.bridge.state.hand_flags = 33 | 128
        self.bridge.state.hand_recovery_mask = 0
        self.step()
        self.assertTrue(self.control.snapshot()['dex3']['recovering'])
        self.bridge.state.hand_flags = 35
        self.step()
        self.assertEqual(self.control.hand_recovery, -1)
        self.event('prepare')
        self.assertEqual(self.control.request, p.PREPARE)
        self.bridge.state.stage = p.HOLDING
        self.bridge.state.hand_recovery_mask = 1 << 8
        self.event('hand_recover', joint=8)
        self.assertEqual(self.control.hand_recovery, -1)

    def test_recovery_request_timeout_and_source_change_stop(self):
        self.event('claim'); self.event('acquire'); self.step()
        self.bridge.state.hand_flags = 33
        self.bridge.state.hand_recovery_mask = 1 << 8
        self.event('hand_recover', joint=8)
        for _ in range(60): self.step()
        self.assertEqual(self.control.hand_recovery, -1)
        self.assertIn('未被底层接受', self.control.reason)
        self.event('hand_recover', joint=8)
        self.event('remote')
        self.assertEqual(self.control.stage, 'DAMPED')
        self.assertEqual(self.bridge.writes[-1][0], p.STOP)

    def test_disconnect_releases_owner_for_another_browser(self):
        self.active()
        self.event('disconnect')
        self.assertEqual(self.control.stage, 'DAMPED')
        self.assertIsNone(self.control.snapshot()['owner'])
        self.now += 2
        self.control.emit('claim', 2)
        self.control.step()
        self.step()
        self.assertEqual(self.control.snapshot()['owner'], 2)
        self.assertEqual(self.control.stage, 'WAITING')
        self.assertFalse(self.bridge.armed)

    def test_stopped_recovery_rejects_non_owner_and_old_claims(self):
        self.active()
        self.control.emit('claim', 1)
        self.event('stop')
        self.assertEqual(self.control.stage, 'DAMPED')
        self.event('claim', 2)
        self.assertFalse(self.bridge.starts)
        self.assertEqual(self.control.stage, 'DAMPED')

    def test_reclaim_fault_with_missing_feedback_and_stop_during_restart(self):
        self.active()
        self.control.damp('bridge failed', fault=True)
        self.bridge.state = None
        with self.control.lock:
            self.control.latest['state'] = None
        self.control.emit('claim', 1)
        def start(armed=False):
            self.bridge.starts.append(armed)
            self.control.emit('stop', 2)
        self.bridge.start = start
        self.control.step()
        self.assertEqual(self.bridge.starts, [False])
        self.assertTrue(self.control.stop_requested.is_set())
        self.bridge.process = None
        self.control.step()
        self.assertEqual(self.control.stage, 'DAMPED')

    def test_observe_never_arms_or_sends(self):
        self.control = Controller(self.config, self.bridge, 'observe', clock=lambda:self.now)
        self.step()
        for kind in ('claim', 'acquire', 'prepare', 'policy', 'hand_toggle', 'hand_zero_compensated',
                     'hand_grasp_compensated', 'move', 'stop'):
            self.event(kind, velocity=[.3, 0, 0])
        self.assertEqual(self.control.stage, 'OBSERVE')
        self.assertFalse(self.bridge.starts)
        self.assertFalse(self.bridge.writes)
        backend = Bridge.__new__(Bridge)
        backend.mode, backend.armed = 'observe', True
        backend.sock = Mock()
        backend.send(p.RUN, np.zeros(29))
        backend.sock.sendto.assert_not_called()

    def active(self):
        self.event('claim')
        self.bridge.armed = True
        self.bridge.state.flags = 3
        self.bridge.state.stage = p.POLICY
        self.control.request = p.RUN
        self.step()

    def test_all_command_axes_use_training_filter_with_heading_enabled(self):
        self.active()
        self.event('move',velocity=[1.2,.5,.8])
        data=self.control.snapshot()
        np.testing.assert_allclose(data['filtered_velocity'],[.06,.025,.04],atol=1e-7)
        np.testing.assert_array_equal(data['velocity'],data['filtered_velocity'])
        self.assertEqual(data['selected_model'],'homie')

    def test_release_filters_then_hands_over_without_dwell(self):
        self.active()
        for _ in range(300):self.event('move',velocity=[.6,0,0])
        self.bridge.state.dq[3]=1.
        previous=self.control.snapshot()['velocity'].copy()
        released=self.now;started=None;decay_ticks=0
        while self.now-released<1.5:
            preceding=self.control.snapshot()
            self.step();data=self.control.snapshot();current=data['velocity']
            self.assertTrue(np.all(abs(current)<=abs(previous)+1e-7))
            self.assertTrue(np.all(current*previous>=0))
            if self.control.zeroing or started is not None:
                if started is None:started=self.now
                decay_ticks+=1
                if np.any(current):
                    np.testing.assert_allclose(current,previous*.95,atol=1e-6)
                    self.assertEqual(data['selected_model'],'homie')
                else:
                    self.assertEqual(data['selected_model'],'balance')
                    history=self.control.observations.models['balance'].history
                    np.testing.assert_allclose(history[-1,64:],np.clip(preceding['model_actions']['homie'],-100,100))
                    self.assertTrue(np.any(history[:-1,:3]))
                    np.testing.assert_array_equal(history[-1,:3],0)
                    break
            previous=current.copy()
        self.assertIsNotNone(started)
        self.assertEqual(decay_ticks,49)
        self.assertAlmostEqual(self.now-started+.02,.98,places=5)
        self.assertFalse(self.control.zeroing)

    def test_release_filter_handles_full_range_and_turn_correction(self):
        self.active()
        for _ in range(300):self.event('move',velocity=[1.2,-.5,.8])
        initial=self.control.snapshot()['velocity'].copy();start=self.now;saw_ramp=False
        while self.now-start<2.:
            self.step();data=self.control.snapshot()
            if data['decelerating']:
                saw_ramp=True
                self.assertFalse(data['heading']['correction_enabled'])
                np.testing.assert_array_equal(data['input_velocity'],0)
                ratio=data['velocity']/initial
                np.testing.assert_allclose(ratio,ratio[0],atol=1e-6)
            elif saw_ramp:break
        self.assertTrue(saw_ramp)
        np.testing.assert_array_equal(data['velocity'],0)
        self.assertEqual(data['selected_model'],'balance')

    def test_release_ramp_rejects_queued_moves_and_hard_stops_win(self):
        for trigger in ('b','stop','disconnect','fault','stale'):
            with self.subTest(trigger=trigger):
                self.setUp();self.active()
                for _ in range(70):self.event('move',velocity=[.3,.1,.2])
                for _ in range(14):self.step()
                self.assertTrue(self.control.zeroing)
                previous=self.control.snapshot()['velocity'].copy()
                self.event('move',velocity=[1.2,.5,.8])
                self.assertTrue(np.all(abs(self.control.snapshot()['velocity'])<=abs(previous)))
                if trigger=='b':self.bridge.state.buttons=p.B;self.step()
                elif trigger=='fault':self.bridge.state.stage=p.FAULT;self.step()
                elif trigger=='stale':
                    self.now+=.2
                    with self.assertRaises(RuntimeError):self.control.step()
                    self.control.damp('stale feedback',fault=True)
                else:self.event(trigger)
                self.assertFalse(self.control.zeroing)
                np.testing.assert_array_equal(self.control.snapshot()['velocity'],0)

    def test_navigation_timeout_filters_then_switches_without_zero_hold(self):
        self.active()
        for _ in range(100):self.event('move',velocity=[.3,0,0],command_source='NAVIGATION')
        stamp=self.control.web.last_motion
        while self.now-stamp<.4:self.step()
        self.assertTrue(self.control.zeroing)
        self.assertGreater(self.control.snapshot()['velocity'][0],0)
        self.finish_zero()
        self.assertLess(self.now-stamp,1.1)
        self.assertEqual(self.control.snapshot()['selected_model'],'balance')

    def test_external_timeout_still_zeroes_immediately(self):
        self.active()
        for _ in range(60):self.event('move',velocity=[.3,0,0],command_source='EXTERNAL')
        self.control.web.last_motion-=.3
        self.step()
        self.assertFalse(self.control.zeroing)
        np.testing.assert_array_equal(self.control.snapshot()['velocity'],0)

    def test_zero_from_other_client_and_latched_stop(self):
        self.active()
        self.event('move', velocity=[.3, 0, 0])
        self.event('zero', client=2)
        self.finish_zero()
        np.testing.assert_array_equal(self.control.snapshot()['velocity'], 0)
        self.event('stop', client=2)
        self.assertEqual(self.control.stage, 'DAMPED')
        self.assertEqual(self.bridge.writes[-1][0], p.STOP)
        self.event('policy')
        self.assertEqual(self.control.stage, 'DAMPED')

    def test_disconnect_stops_even_with_stale_feedback(self):
        self.active()
        self.event('disconnect', client=2)
        self.assertEqual(self.control.stage, 'POLICY')
        self.bridge.state.received_ns -= 1_000_000_000
        self.control.emit('disconnect', 1)
        self.control.step()
        self.assertEqual(self.control.stage, 'DAMPED')
        self.assertIn('断开', self.control.reason)

    def test_zero_on_policy_entry_with_cached_remote_sticks(self):
        self.bridge.armed = True
        self.bridge.state.flags = 3
        self.bridge.state.stage = p.POLICY
        self.bridge.state.buttons = p.L1
        self.bridge.state.sticks[1] = 1.
        self.control.request = p.RUN
        self.step()
        np.testing.assert_array_equal(self.control.snapshot()['velocity'], 0)

    def test_stale_motion_event_cannot_revive_motion(self):
        self.active()
        state = self.bridge.state
        self.control.events.put(dict(kind='move', client=1, velocity=[.3, 0, 0],
                                     session=state.session, feedback_ns=state.received_ns,
                                     time=self.now-1))
        self.step()
        np.testing.assert_array_equal(self.control.snapshot()['velocity'], 0)

    def test_body_diagonal_heading_reach_homie_while_balance_stays_zero(self):
        self.active()
        self.event('move', velocity=[.1,.05,0])
        for i in range(240):
            self.bridge.state.quaternion = quaternion(min(.3,i*.005))
            self.bridge.state.omega[:] = [0,0,.25 if i<60 else 0]
            self.event('move', velocity=[.1,.05,0])
        data = self.control.snapshot()
        np.testing.assert_allclose(data['velocity'][:2],[.1,.05],atol=1e-6)
        self.assertAlmostEqual(data['velocity'][2],-.15,places=6)
        self.assertLess(data['heading']['error_rad'],0)
        self.assertEqual(data['heading']['yaw_ref'],0)
        self.check_policy_inputs(data)
        self.event('zero', client=2)
        self.assertTrue(self.control.zeroing)
        self.finish_zero()
        self.assertEqual(self.control.snapshot()['selected_model'],'balance')

    def check_policy_inputs(self, data):
        for name, model in self.control.observations.models.items():
            spec=self.config.contracts[name]['input']
            scale=np.asarray(spec['command_scale'])[:3]
            expected=data['velocity']*scale
            np.testing.assert_allclose(model.history[-1,:3],expected,atol=1e-6)
            # Original unfiltered gyro observation must remain byte-for-byte equivalent.
            offset=4
            np.testing.assert_allclose(model.history[-1,offset:offset+3],self.bridge.state.omega*spec['observation_scales']['ang_vel'],atol=1e-6)

    def test_heading_correction_expires_with_web_motion(self):
        self.active();self.event('move',velocity=[.1,.1,0])
        for _ in range(30):
            self.bridge.state.quaternion=quaternion(.05)
            self.event('move',velocity=[.1,.1,0])
        self.assertLess(self.control.snapshot()['velocity'][2],0)
        self.control.web.last_motion-=1
        self.step()
        self.assertTrue(self.control.zeroing)
        self.finish_zero()
        np.testing.assert_array_equal(self.control.snapshot()['velocity'],0)
        self.finish_zero()
        self.assertFalse(self.control.snapshot()['heading']['active'])

    def test_heading_settings_require_owner_and_zero_speed(self):
        self.active()
        changes = dict(command_frame='body', heading_kp=.5, max_correction_rad_s=.1,
                       gyro_cutoff_hz=4.)
        self.control.emit('motion_settings', 2, settings=changes)
        self.step()
        self.assertEqual(self.control.heading.settings, self.config.motion_settings)
        self.event('move', velocity=[.1, 0, 0])
        self.control.emit('motion_settings', 1, settings=changes)
        self.step()
        self.assertEqual(self.control.heading.settings, self.config.motion_settings)
        self.event('zero')
        self.finish_zero()
        self.control.emit('motion_settings', 1, settings=changes)
        self.step()
        self.assertEqual(self.control.heading.settings, dict(self.config.motion_settings, **changes))
        self.control.emit('motion_settings', 1, settings=dict(changes, heading_kp=float('nan')))
        self.step()
        self.assertEqual(self.control.heading.settings['heading_kp'], .5)
        self.control.emit('motion_settings', 1, settings=dict(changes, command_frame='world'))
        self.step()
        self.assertEqual(self.control.heading.settings, dict(self.config.motion_settings, **changes))

    def test_forward_after_turn_stays_positive_in_actual_policy_inputs(self):
        self.active()
        for i in range(160):
            self.bridge.state.quaternion=quaternion(-i*.016)
            self.bridge.state.omega[:]=[0,0,-.8]
            self.event('move',velocity=[0,0,-.8])
        self.assertEqual(self.control.snapshot()['heading']['heading_state'],'TURNING')
        self.bridge.state.omega.fill(0)
        for _ in range(300):self.event('move',velocity=[.29,0,0])
        data=self.control.snapshot()
        np.testing.assert_allclose(data['velocity'],[.29,0,0],atol=1e-6)
        self.assertTrue(data['heading']['correction_enabled'])
        self.assertAlmostEqual(data['heading']['yaw_ref'],-159*.016)
        self.check_policy_inputs(data)

    def test_live_config_rejects_old_world_override(self):
        with self.assertRaisesRegex(ValueError, '机身坐标'):
            RobotConfig(command_frame='world')

    def test_full_velocity_ranges_reach_policies_and_ramp_to_zero(self):
        self.active()
        for request, expected in [([100, 100, 100], [1.2, .5, .8]),
                                  ([-100, -100, -100], [-.8, -.5, -.8])]:
            self.event('zero')
            self.finish_zero()
            for _ in range(300):
                self.event('move', velocity=request)
            data = self.control.snapshot()
            np.testing.assert_allclose(data['velocity'], expected, atol=2e-6)
            self.check_policy_inputs(data)
            self.event('zero')
            self.assertTrue(self.control.zeroing)
            self.finish_zero()
            np.testing.assert_array_equal(self.control.snapshot()['velocity'], 0)
            self.assertEqual(self.control.heading.correction, 0)
        self.event('move', velocity=[.1, 0, 0])
        self.bridge.state.quaternion = quaternion(.05)
        for _ in range(30):self.event('move', velocity=[.1, 0, 0])
        self.assertNotEqual(self.control.heading.correction, 0)
        self.event('stop')
        self.assertEqual(self.control.heading.correction, 0)
        self.assertEqual(self.control.snapshot()['heading']['correction_raw_rad_s'], 0)

    def test_heading_reference_rebase_only_when_stationary(self):
        self.active()
        self.bridge.state.quaternion = np.array([np.cos(.2), 0, 0, np.sin(.2)])
        self.event('move', velocity=[.1, 0, 0])
        self.event('rebase_heading')
        self.assertEqual(self.control.heading.origin, 0)
        self.event('zero')
        self.finish_zero()
        self.event('rebase_heading')
        self.assertAlmostEqual(self.control.heading.origin, .4)

    def test_remote_manual_body_axes_heading_feedback_and_deadman(self):
        self.bridge.armed=True;self.bridge.state.flags=3;self.bridge.state.stage=p.POLICY
        self.control.request=p.RUN;self.step()
        self.bridge.state.buttons=p.L1;self.bridge.state.sticks[1]=1
        self.step()
        for i in range(300):
            self.bridge.state.quaternion=quaternion(min(i*.002,.1))
            self.step()
        data=self.control.snapshot()
        np.testing.assert_allclose(data['velocity'][:2],[1.2,0],atol=2e-6)
        self.assertLess(data['velocity'][2],0)
        self.check_policy_inputs(data)
        self.bridge.state.buttons=0;self.step()
        np.testing.assert_array_equal(self.control.snapshot()['velocity'],0)

    def test_manual_zero_ramps_final_policy_commands_despite_timeout_and_queued_moves(self):
        self.active()
        self.bridge.state.quaternion = np.array([np.cos(.1), 0, 0, np.sin(.1)])
        for _ in range(90):
            self.event('move', velocity=[.3, -.15, .2])
        previous = self.control.snapshot()['velocity'].copy()
        started = previous.copy()
        self.event('zero', client=2)
        self.assertTrue(self.control.zeroing)
        steps = 0
        while self.control.zeroing:
            data = self.control.snapshot()
            current = data['velocity']
            self.assertTrue(np.all(abs(current) <= abs(previous)+1e-7))
            self.assertTrue(np.all(current*started >= -1e-7))
            np.testing.assert_allclose(current,previous*.95,atol=1e-6)
            np.testing.assert_array_equal(data['input_velocity'], 0)
            self.assertEqual(data['heading']['correction_rad_s'], 0)
            self.assertFalse(data['heading']['correction_enabled'])
            self.check_policy_inputs(data)
            previous = current.copy()
            # No hold heartbeat is required after an explicit zero, and queued
            # old motion cannot cancel that finite stop ramp.
            self.control.web.last_motion -= 1
            self.bridge.state.quaternion = np.array([np.cos(.3),0,0,np.sin(.3)])
            self.event('move', velocity=[1.2, .5, .8])
            steps += 1
            self.assertLess(steps, 250)
        self.assertGreater(steps, 12)
        np.testing.assert_array_equal(self.control.snapshot()['velocity'], 0)
        self.assertEqual(self.control.snapshot()['selected_model'], 'balance')
        for _ in range(20):self.event('move', velocity=[.1,0,0])
        self.assertGreater(self.control.snapshot()['velocity'][0], 0)

    def test_manual_zero_is_interrupted_by_b_stop_disconnect_and_fault(self):
        for trigger in ['b', 'stop', 'disconnect', 'fault']:
            with self.subTest(trigger=trigger):
                self.setUp()
                self.active()
                for _ in range(80): self.event('move', velocity=[.3, .1, .2])
                self.event('zero')
                self.assertTrue(self.control.zeroing)
                if trigger == 'b':
                    self.bridge.state.buttons = p.B
                    self.step()
                elif trigger == 'fault':
                    self.bridge.state.stage = p.FAULT
                    self.step()
                else:
                    self.event(trigger)
                self.assertFalse(self.control.zeroing)
                np.testing.assert_array_equal(self.control.snapshot()['velocity'], 0)

    def test_zero_from_remote_preserves_final_command_for_ramp(self):
        self.active()
        self.event('remote')
        self.bridge.state.buttons = p.L1
        self.bridge.state.sticks[1] = 1.
        for _ in range(100):self.step()
        previous = self.control.snapshot()['velocity'][0]
        self.event('zero')
        current = self.control.snapshot()['velocity'][0]
        self.assertTrue(0 < current < previous)
        self.assertTrue(self.control.zeroing)
        self.assertEqual(self.control.web.source, 'web')
        self.finish_zero()

    def test_navigation_reference_rebases_at_source_handoff(self):
        self.active();self.event('move',velocity=[.1,0,0])
        for _ in range(50):
            self.bridge.state.quaternion=quaternion(.05)
            self.control.emit('move',1,[.1,0,0],command_source='NAVIGATION')
            self.step()
        data=self.control.snapshot()
        self.assertEqual(data['heading']['heading_state'],'TRACKING')
        self.assertAlmostEqual(data['heading']['yaw_ref'],.05)
        self.assertEqual(data['velocity'][2],0)
        self.check_policy_inputs(data)
        self.event('move',velocity=[.1,0,0])
        self.assertAlmostEqual(self.control.snapshot()['heading']['yaw_ref'],.05)

    def test_disabled_shadow_preserve_baseline_through_policy_scaling(self):
        for mode in ('disabled','shadow'):
            self.setUp();self.active()
            self.control.heading.configure(dict(self.config.motion_settings,run_mode=mode),quaternion())
            for _ in range(20):self.step()
            self.event('move',velocity=[.1,.1,0])
            for _ in range(60):
                self.bridge.state.quaternion=quaternion(.05)
                self.event('move',velocity=[.1,.1,0])
            data=self.control.snapshot()
            np.testing.assert_array_equal(data['velocity'],data['input_velocity'])
            self.check_policy_inputs(data)
            if mode=='shadow':self.assertLess(data['heading']['wz_suggested'],0)

    def test_independent_imu_filter_does_not_modify_policy_observations(self):
        self.active();self.event('move',velocity=[.1,0,0])
        self.bridge.state.quaternion=quaternion(.01,.1,-.1).astype(np.float32)
        self.bridge.state.omega[:]=[.11,-.13,.21]
        original=self.bridge.state.quaternion.copy()
        self.event('move',velocity=[.1,0,0])
        data=self.control.snapshot();self.check_policy_inputs(data)
        np.testing.assert_array_equal(self.bridge.state.quaternion,original)
        self.assertNotAlmostEqual(data['heading']['filtered_yaw_rate_for_heading'],.21)
        gravity=[np.sin(-.1),-np.sin(.1)*np.cos(-.1),-np.cos(.1)*np.cos(-.1)]
        for name,model in self.control.observations.models.items():
            offset=7
            np.testing.assert_allclose(model.history[-1,offset:offset+3],gravity,atol=1e-6)

    def test_shared_imu_fault_stops_before_inference_or_joint_command(self):
        self.active();steps=self.control.policy_steps;writes=len(self.bridge.writes)
        self.bridge.state.quaternion.fill(0)
        with self.assertRaises(ImuFault):self.step()
        self.assertEqual(self.control.policy_steps,steps)
        self.assertEqual(len(self.bridge.writes),writes)
        # Production _run catches the same exception and calls this safety path.
        self.control.damp('Invalid pelvis quaternion norm',fault=True)
        self.assertEqual(self.bridge.writes[-1][0],p.STOP)
        self.assertEqual(self.control.stage,'FAULT')


if __name__ == '__main__':
    unittest.main(verbosity=2)
