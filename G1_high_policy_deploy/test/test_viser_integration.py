"""Offline contracts: real ONNX inference, fake robot feedback, loopback camera."""
from pathlib import Path
from dataclasses import replace
import gzip
import json
import socket
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
sys.path[:0] = [str(ROOT / 'robot'), str(WORKSPACE / 'Viser'), str(WORKSPACE / 'Viser/test'),
               str(WORKSPACE / 'carmer')]
from core.robot import RobotConfig
from core.controller import Controller
from core.recording import SessionRecorder
from hybrid_deploy import protocol as p
from test_control import FakeBridge
from heading_udp import heading_packet, HeadingPublisher
from pushcart_v81.carmer_io import CarmerYaw
from pushcart_v81.controller import V81Controller, OnnxModels, Sensors, load_config
from pushcart_v81.viser_policy import ViserPushPolicy, UPPER
from pushcart_v81.homie_adapter import PushcartHomie
from cart_heading import CartHeadingEstimator, fuse_cart_headings, rotation_from_rpy_degrees as rotation
from pelvis_attitude import G1Kinematics, OPTICAL_TO_LINK, PelvisCompensator
from robot_feedback import StateBuffer


def camera_result(stamp, yaw=.2, valid=True):
    return dict(frame_timestamp_monotonic=stamp, frame_number=1,
                pelvis_compensation=dict(valid=True),
                cart_heading_settings=dict(camera_latency_ms=40),
                raw_fused_cart_heading=dict(valid=valid, yaw_rad=yaw,
                                            reference_frame='pelvis_level', tag_ids=[0, 1]))


class CameraTests(unittest.TestCase):
    def setUp(self):
        self.camera = CarmerYaw(0)
        self.addCleanup(self.camera.close)

    def test_actual_udp_and_exposure_timestamp(self):
        pub = HeadingPublisher(self.camera.sock.getsockname()[1])
        self.addCleanup(pub.close)
        pub.publish(camera_result(10., -.3))
        self.assertTrue(self.camera.poll(10.01))
        self.assertAlmostEqual(self.camera.stamp, 9.96)
        self.assertAlmostEqual(self.camera.yaw, -.3)
        self.assertFalse(self.camera.poll(10.27))

    def test_invalid_replay_future_and_dropout_do_not_refresh_measurement(self):
        packet = heading_packet(camera_result(10.))
        self.assertTrue(self.camera.accept(json.dumps(packet), 10.))
        for patch in [dict(yaw_rad=float('nan')), dict(captured_monotonic=11.),
                      dict(reference_frame='pelvis'), dict(captured_monotonic=9.9),
                      dict(valid=1), {}]:
            self.assertFalse(self.camera.accept(json.dumps(dict(packet, **patch)), 10.))
        missing = heading_packet(camera_result(10.2, valid=False))
        self.assertTrue(self.camera.accept(json.dumps(missing), 10.2))
        self.assertFalse(self.camera.fresh(10.27))
        for data in ('null', '[]', '{}', 'bad json'):
            self.assertFalse(self.camera.accept(data, 10.3))

    def test_uncompensated_camera_never_supplies_valid_heading(self):
        result = camera_result(10.)
        result.pop('pelvis_compensation')
        self.assertFalse(heading_packet(result)['valid'])

    def test_camera_reports_tag_loss_late_frames_and_stream_timeout_separately(self):
        self.camera.accept(json.dumps(heading_packet(camera_result(10.))), 10.)
        missing = camera_result(10.29, valid=False)
        missing['raw_fused_cart_heading']['reason'] = 'no_tags_detected'
        self.camera.accept(json.dumps(heading_packet(missing)), 10.29)
        status = self.camera.snapshot(10.30)
        self.assertFalse(status['fresh'])
        self.assertEqual(status['status'], 'no_heading')
        self.assertIn('完整入镜', status['reason'])
        late = camera_result(10.4)
        late.update(processing_ms=320., capture_fps=60., detect_fps=5.)
        self.assertFalse(self.camera.accept(json.dumps(heading_packet(late)), 10.8))
        status = self.camera.snapshot(10.81)
        self.assertEqual(status['status'], 'late_frame')
        self.assertFalse(status['fresh'])
        self.assertEqual(status['timing']['processing_ms'], 320.)
        self.assertAlmostEqual(self.camera.stamp, 9.96)
        self.assertEqual(self.camera.snapshot(11.2)['status'], 'frame_timeout')
        self.camera.accept(json.dumps(heading_packet(camera_result(11.3))), 11.3)
        self.assertEqual(self.camera.snapshot(11.3)['status'], 'tracking')

    def test_attitude_failure_keeps_its_reason_in_camera_packet(self):
        result = camera_result(10.)
        result['pelvis_compensation'] = dict(valid=False, reason='state_alignment_timeout')
        packet = heading_packet(result)
        self.assertFalse(packet['valid'])
        self.assertEqual(packet['reason'], 'state_alignment_timeout')

    def test_policy_receives_current_compensated_yaw_without_display_filter_lag(self):
        result = camera_result(10., -.3)
        # The display still points the other way while its low-pass catches up.
        result['fused_cart_heading'] = dict(result['raw_fused_cart_heading'], yaw_rad=.4)
        packet = heading_packet(result)
        self.assertTrue(self.camera.accept(json.dumps(packet), 10.))
        self.assertAlmostEqual(self.camera.yaw, -.3)
        self.assertAlmostEqual(self.camera.stamp, 9.96)
        # Display output cannot rescue an invalid or missing current raw result.
        result['raw_fused_cart_heading']['valid'] = False
        self.assertFalse(heading_packet(result)['valid'])
        result.pop('raw_fused_cart_heading')
        self.assertFalse(heading_packet(result)['valid'])

    def test_only_cart_ids_zero_and_one_can_refresh_yaw(self):
        self.camera.allowed_ids = frozenset((0, 1))
        packet = heading_packet(camera_result(10.))
        for ids in ([2], [0, 2], [], [True]):
            self.assertFalse(self.camera.accept(json.dumps(dict(packet, tag_ids=ids)), 10.))
            self.assertFalse(self.camera.fresh(10.))
        for index, ids in enumerate(([0], [1], [0, 1])):
            packet = heading_packet(camera_result(10. + index * .02))
            self.assertTrue(self.camera.accept(json.dumps(dict(packet, tag_ids=ids)), 10. + index * .02))
            self.assertEqual(self.camera.tag_ids, ids)

    def test_receive_error_invalidates_yaw_without_escaping(self):
        self.camera.accept(json.dumps(heading_packet(camera_result(10.))), 10.)
        self.camera.sock.close()
        self.assertFalse(self.camera.poll(10.01))
        self.assertIn('接收异常', self.camera.reason)

    def test_zero_rebases_raw_yaw_without_refreshing_or_accumulating(self):
        self.camera.accept(json.dumps(heading_packet(camera_result(10., -.3))), 10.)
        stamp = self.camera.stamp
        self.camera.set_offset(-.3)
        self.camera.set_offset(-.3)
        self.assertAlmostEqual(self.camera.yaw, 0.)
        self.assertAlmostEqual(self.camera.raw_yaw, -.3)
        self.assertEqual(self.camera.stamp, stamp)
        self.assertFalse(self.camera.fresh(10.31))
        for invalid in (float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                self.camera.set_offset(invalid)
        self.assertAlmostEqual(self.camera.offset, -.3)
        self.camera.set_offset(np.pi - .02)
        self.camera.accept(json.dumps(heading_packet(camera_result(11., -np.pi + .02))), 11.)
        self.assertAlmostEqual(self.camera.yaw, .04)


class ExportCompatibilityTests(unittest.TestCase):
    def test_tuned_him_matches_paired_export_and_keeps_original_latent(self):
        import onnxruntime as ort
        from core.policy import Policy
        config = RobotConfig()
        path = ROOT / 'robot/models/homie_v81.onnx'
        adapter = PushcartHomie(path, config.policy_paths['homie'])
        original = Policy(config.deploy, config.deployment)
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        reference = ort.InferenceSession(str(path), options, providers=['CPUExecutionProvider'])
        rng = np.random.default_rng(81)
        action_difference = velocity_difference = 0.
        self.assertTrue(adapter.warmed)
        for _ in range(16):
            obs = rng.normal(0, .3, (1, 456)).astype(np.float32)
            expected, used, old = reference.run(None, dict(obs_history=obs, use_finetuned=np.ones((1, 1), np.float32)))
            action, estimate = adapter.infer(obs)
            base_action = original.infer('homie', obs)
            np.testing.assert_allclose(action, expected[0], atol=2e-5, rtol=1e-5)
            np.testing.assert_allclose(estimate['velocity_raw'], used[0], atol=2e-5, rtol=1e-5)
            np.testing.assert_allclose(estimate['velocity_original_raw'], old[0], atol=2e-5, rtol=1e-5)
            for key in ('latent_raw', 'latent_normalized'):
                np.testing.assert_allclose(estimate[key], original.homie_estimator[key], atol=2e-5, rtol=1e-5)
            action_difference = max(action_difference, float(np.max(abs(action - base_action))))
            velocity_difference = max(velocity_difference, float(np.max(abs(used - old))))
        self.assertGreater(action_difference, .01)
        self.assertGreater(velocity_difference, .01)
        with self.assertRaises(ValueError):
            adapter.infer(np.full((1, 456), np.nan, np.float32))

    def test_base_homie_matches_the_packaged_training_actor(self):
        # Two independently exported ONNX graphs; no original ZIP is needed.
        import onnx
        import onnxruntime as ort
        from onnx import numpy_helper
        path = WORKSPACE / 'G1Deploy_BalanceHomie/models/homie/policy.onnx'
        base = onnx.load(str(path))
        paired = onnx.load(str(ROOT / 'robot/models/homie_v81.onnx'))
        weights = {x.name: numpy_helper.to_array(x) for x in paired.graph.initializer}
        for tensor in base.graph.initializer:
            np.testing.assert_array_equal(numpy_helper.to_array(tensor), weights[tensor.name])
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        session = ort.InferenceSession(str(path), options, providers=['CPUExecutionProvider'])
        models = OnnxModels(ROOT / 'robot/models', threads=1)
        rng = np.random.default_rng(8)
        for _ in range(12):
            observation = rng.normal(0, .3, 456).astype(np.float32)
            expected = session.run(None, {session.get_inputs()[0].name: observation[None]})[0][0]
            actual, _, _ = models.run_homie(observation, use_finetuned=False)
            np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=1e-5)


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.now = 100.
        self.config = RobotConfig()
        self.camera = CarmerYaw(0)
        self.high = ViserPushPolicy(self.config, ROOT / 'robot/config/viser_v81.json', camera=self.camera)
        self.bridge = FakeBridge(self.config)
        self.bridge.close = Mock()
        self.control = Controller(self.config, self.bridge, 'sim', clock=lambda: self.now,
                                  high_policy=self.high)
        self.addCleanup(self.control.close)
        self.refresh = True
        for _ in range(8):
            self.tick()
        self.event('claim')
        self.bridge.armed = True
        self.bridge.state.flags = 3
        self.bridge.state.stage = p.POLICY
        self.bridge.state.applied_q = self.config.prepare_q.copy()
        self.bridge.state.q = self.config.prepare_q.copy()
        self.bridge.state.hand_flags = 55  # supported + healthy + active + grasp
        self.control.hand_pose = 2
        self.control.request = p.RUN
        self.tick()

    def tick(self, yaw=.2, valid=True, dt=.02):
        self.now += dt
        self.bridge.state.received_ns = int((self.now - .000001) * 1e9)
        self.bridge.state.sequence += 1
        if self.refresh:
            self.camera.accept(json.dumps(heading_packet(camera_result(self.now, yaw, valid))), self.now)
        self.control.step()
        return self.control.snapshot()

    def event(self, kind, velocity=None, client=1):
        self.control.emit(kind, client, velocity)
        return self.tick()

    def enable(self):
        for _ in range(40):
            if self.high.history_ready(self.now):
                break
            self.tick()
        self.event('pushcart_toggle')
        self.assertTrue(self.high.enabled, self.control.reason)

    def move(self, ticks=70):
        for _ in range(ticks):
            self.event('move', [.3, 0, 0])
        return self.control.snapshot()

    def test_start_is_zero_and_high_is_opt_in(self):
        self.assertFalse(self.high.enabled)
        np.testing.assert_array_equal(self.control.snapshot()['velocity'], 0)
        self.assertEqual(self.control.snapshot()['selected_model'], 'balance')

    def test_recorded_pushcart_command_chain_and_inputs_survive_stop(self):
        self.config.high_policy_metadata = self.high.metadata
        with tempfile.TemporaryDirectory(prefix='pushcart-recording-') as directory:
            recorder = SessionRecorder(self.config, {'mode': 'sim'}, directory)
            self.control.recorder = recorder
            try:
                shadow = self.tick()
                self.enable()
                self.move()
                # A distinct 200 Hz sample supplies the actual torque observation.
                state = self.bridge.state
                self.bridge.previous_state = replace(
                    state, received_ns=int((self.now + .02 - .000001) * 1e9) - 5_000_000,
                    applied_q=state.q + .02, applied_kp=np.full(29, 10.), applied_kd=np.zeros(29))
                state.quaternion = np.array([np.cos(.05/2), 0., 0., np.sin(.05/2)])
                moving = self.event('move', [.3, 0, 0])
                high = moving['high_policy']
                inputs = high['inference_inputs']
                self.assertEqual(inputs['input_state_tick'], moving['state']['tick'])
                self.assertEqual(inputs['input_state_received_ns'], moving['state']['received_ns'])
                np.testing.assert_allclose(inputs['upper_torque_pd_estimate'], .2)
                np.testing.assert_allclose(self.high.core.frames[-1, 79:94],
                                           inputs['upper_torque_pd_estimate'] * self.high.core.tau_scale)
                self.assertAlmostEqual(self.high.core.frames[-1, 94], inputs['camera_yaw_rad'], places=6)
                # Recorder is asynchronous; later mutation of diagnostics cannot
                # silently change this already-queued telemetry row.
                self.high.inference_inputs['upper_torque_pd_estimate'].fill(999.)
                zero = self.event('zero')
                for _ in range(120):
                    standing = self.tick()
                    if standing['selected_model'] == 'balance':
                        break
                self.refresh = False
                for _ in range(20):
                    lost_camera = self.tick()
                self.event('stop')
                self.tick()
            finally:
                self.control.recorder = None
                summary = recorder.close()
            self.assertEqual(summary['recorder_error'], '')
            self.assertEqual(summary['dropped_records'], 0)
            with gzip.open(recorder.path / 'telemetry.jsonl.gz', 'rt') as stream:
                rows = [json.loads(line) for line in stream]
            indexed = {tuple(row['state']['tick']): row for row in rows}
            moving = indexed[moving['state']['tick']]
            high = moving['high_policy']
            np.testing.assert_allclose(high['inference_inputs']['upper_torque_pd_estimate'], .2)
            self.assertAlmostEqual(high['torque_sample_age_ms'], 5.)
            np.testing.assert_allclose(high['raw_task_command'], [.3, 0., 0.], atol=1e-7)
            # The pelvis turned to 0.05 only at this tick, after the 40 ms old
            # camera frame: pelvis motion newer than the frame is not cart drift.
            self.assertAlmostEqual(high['heading_hold']['imu_yaw_at_camera_rad'], 0.)
            np.testing.assert_allclose(high['task_command'], [.3, 0., 0.], atol=1e-7)
            self.assertAlmostEqual(high['heading_hold']['err'], 0.)
            self.assertAlmostEqual(high['heading_hold']['corr'], 0.)
            np.testing.assert_allclose(high['homie_command'], moving['input_velocity'])
            np.testing.assert_allclose(high['homie_command'], moving['filtered_velocity'])
            np.testing.assert_allclose(high['homie_command'], moving['velocity'])
            np.testing.assert_allclose(high['leg_residual'], high['applied_leg_residual'])
            np.testing.assert_allclose(np.asarray(moving['requested_target'])[UPPER],
                                       self.high.core.hold + high['upper_delta_rad'], atol=1e-6)
            self.assertTrue(high['upper_target_selected'])
            self.assertGreater(high['gate'], 0.)
            self.assertEqual(moving['homie_estimator_source'], 'pushcart_finetuned')
            self.assertEqual(len(moving['homie_estimator']['latent_raw']), 32)
            self.assertEqual(len(high['estimator']), 19)
            self.assertEqual(len(high['raw_actions']), 30)
            shadow = indexed[shadow['state']['tick']]['high_policy']
            self.assertEqual(shadow['inference_role'], 'shadow')
            self.assertFalse(shadow['upper_target_selected'])
            np.testing.assert_array_equal(shadow['applied_leg_residual'], 0.)
            zero = indexed[zero['state']['tick']]
            self.assertTrue(np.any(zero['high_policy']['homie_command']))
            np.testing.assert_array_equal(zero['high_policy']['task_command'], 0.)
            self.assertIsNone(zero['high_policy']['heading_hold']['ref'])
            self.assertEqual(zero['high_policy']['heading_hold']['corr'], 0.)
            np.testing.assert_array_equal(zero['input_velocity'], 0.)
            np.testing.assert_array_equal(zero['high_policy']['applied_leg_residual'], 0.)
            self.assertFalse(zero['high_policy']['upper_target_selected'])
            self.assertTrue(zero['decelerating'])
            standing = indexed[standing['state']['tick']]
            self.assertEqual(standing['selected_model'], 'balance')
            np.testing.assert_array_equal(standing['velocity'], 0.)
            self.assertTrue(indexed[lost_camera['state']['tick']]['high_policy']['motion_blocked'])
            self.assertEqual(rows[-1]['stage'], 'DAMPED')
            inference_fields = ('raw_actions', 'estimator', 'inference_inputs', 'raw_task_command', 'heading_hold', 'task_command',
                                'task_reference', 'homie_command', 'gate', 'leg_residual',
                                'upper_delta_rad', 'upper_target_selected', 'applied_leg_residual',
                                'inference_started_monotonic', 'inference_finished_monotonic')
            for row in rows:
                high = row['high_policy']
                if high['inference_role'] is None:
                    for field in inference_fields:
                        self.assertIsNone(high[field], field)
                else:
                    self.assertEqual(high['inference_inputs']['input_state_tick'], row['state']['tick'])
                    self.assertEqual(high['inference_inputs']['camera_captured_monotonic'],
                                     high['camera']['captured_monotonic'])
                    self.assertLessEqual(high['inference_started_monotonic'], high['inference_finished_monotonic'])
            metadata = json.loads((recorder.path / 'metadata.json').read_text())
            self.assertEqual(metadata['schema'], 20)
            self.assertEqual(metadata['high_policy']['cart_heading_hold'],
                             dict(kp=1., max_corr=.15, max_err=.3, task_wz_limit=.6))
            self.assertEqual(metadata['high_policy']['upper_pd']['target_interpolation_s'], .02)
            self.assertEqual(metadata['high_policy']['telemetry']['upper_sdk_indices'], UPPER.tolist())
            with gzip.open(recorder.path / 'events.jsonl.gz', 'rt') as stream:
                events = {json.loads(line)['type'] for line in stream}
            self.assertTrue({'smooth_zero_started', 'smooth_zero_completed', 'damping'} <= events)

    def test_runtime_switches_only_homie_velocity_estimator_and_preserves_viser_history(self):
        adapter = self.high.homie_model
        with patch.object(adapter, 'infer', wraps=adapter.infer) as tuned:
            normal = self.tick()
            tuned.assert_not_called()
            self.assertEqual(normal['homie_estimator_source'], 'homie_original')
            self.enable()
            data = self.move()
            self.assertEqual(data['homie_estimator_source'], 'pushcart_finetuned')
            obs = tuned.call_args.args[0].copy()
            action, estimate = adapter.infer(obs)
            np.testing.assert_allclose(data['model_actions']['homie'], action, atol=1e-6)
            for key, value in estimate.items():
                np.testing.assert_allclose(data['homie_estimator'][key], value, atol=1e-6)
            # The tuned branch consumes the existing 6-frame execution history,
            # including the previous selected leg action, not a private history.
            committed = self.control.observations.models['homie'].history
            np.testing.assert_allclose(obs.reshape(6, 76), committed, atol=1e-6)
            self.event('zero')
            for _ in range(120):
                data = self.tick()
                if data['selected_model'] == 'balance': break
            np.testing.assert_array_equal(data['velocity'], 0.)
            np.testing.assert_array_equal(self.high.applied_residual, 0.)
            tuned.reset_mock()
            normal = self.event('pushcart_disable')
            tuned.assert_not_called()
            self.assertEqual(normal['homie_estimator_source'], 'homie_original')
            session = self.control.policy.sessions['homie']
            obs = self.control.observations.models['homie'].history.reshape(1, 456)
            expected = session.run(['actions'], {session.get_inputs()[0].name: obs})[0][0]
            np.testing.assert_allclose(normal['model_actions']['homie'], expected, atol=1e-6)

    def test_zero_button_rebuilds_history_and_repeat_does_not_accumulate(self):
        for _ in range(40): self.tick()
        self.assertTrue(self.high.history_ready(self.now))
        np.testing.assert_allclose(self.high.core.frames_long[:, 94], .2)
        quat = self.bridge.state.quaternion.copy()
        hand_pose = self.control.hand_pose
        for _ in range(2):
            self.event('cart_yaw_zero')
            self.event('pushcart_enable')
            self.assertFalse(self.high.enabled)
            self.assertIn('标定', self.control.reason)
            for _ in range(60):
                data = self.tick()
                if self.high.zero_calibration['status'] != 'collecting':
                    break
            self.assertEqual(self.high.zero_calibration['status'], 'done')
            self.assertAlmostEqual(self.camera.offset, .2)
            self.assertAlmostEqual(self.camera.yaw, 0.)
            self.assertEqual(self.high.history_frames, 1)
            self.assertFalse(self.high.history_ready(self.now))
            np.testing.assert_allclose(self.high.core.frames_long[:, 94], 0., atol=1e-7)
            np.testing.assert_array_equal(self.bridge.state.quaternion, quat)
            self.assertEqual(self.control.hand_pose, hand_pose)
            self.assertEqual(data['selected_model'], 'balance')
            np.testing.assert_array_equal(data['velocity'], 0.)
        for _ in range(31): self.tick()
        self.event('pushcart_enable')
        self.assertTrue(self.high.enabled)

    def test_zero_averages_angles_across_pi_boundary(self):
        self.event('cart_yaw_zero')
        for i in range(60):
            self.tick(yaw=np.pi - .01 if i % 2 else -np.pi + .01)
            if self.high.zero_calibration['status'] != 'collecting': break
        self.assertEqual(self.high.zero_calibration['status'], 'done')
        self.assertAlmostEqual(abs(self.camera.offset), np.pi)
        self.assertLess(abs(self.camera.yaw), .011)

    def test_zero_rejects_motion_and_retains_previous_offset(self):
        self.camera.set_offset(.1)
        self.event('cart_yaw_zero')
        for i in range(60):
            self.tick(yaw=.2 if i % 2 else .5)
            if self.high.zero_calibration['status'] != 'collecting': break
        self.assertEqual(self.high.zero_calibration['status'], 'failed')
        self.assertAlmostEqual(self.camera.offset, .1)
        self.event('cart_yaw_zero')
        self.event('move', [.3, 0., 0.])
        self.assertEqual(self.high.zero_calibration['status'], 'failed')
        self.assertAlmostEqual(self.camera.offset, .1)

    def test_zero_never_counts_held_or_invalid_camera_frames(self):
        self.event('cart_yaw_zero')
        self.refresh = False
        for _ in range(5): self.tick()
        self.assertEqual(self.high.zero_calibration['samples'], 0)
        self.refresh = True
        for _ in range(5): self.tick(valid=False)
        self.assertEqual(self.high.zero_calibration['samples'], 0)
        for _ in range(15): self.tick(valid=False)
        self.assertEqual(self.high.zero_calibration['status'], 'failed')
        self.assertAlmostEqual(self.camera.offset, 0.)

    def test_zero_requires_owner_and_disabled_pushcart(self):
        self.event('cart_yaw_zero', client=2)
        self.assertEqual(self.high.zero_calibration['status'], 'idle')
        self.enable()
        self.event('cart_yaw_zero')
        self.assertEqual(self.high.zero_calibration['status'], 'idle')
        self.assertTrue(self.high.enabled)
        self.assertIn('返回普通行走', self.control.reason)

    def test_zero_times_out_if_fresh_camera_supplies_too_few_distinct_frames(self):
        self.camera.set_offset(.1)
        self.event('cart_yaw_zero')
        for i in range(160):
            # 5 Hz stays within the freshness limit but cannot finish sampling.
            self.refresh = i % 10 == 0
            self.tick()
            if self.high.zero_calibration['status'] != 'collecting': break
        self.assertEqual(self.high.zero_calibration['status'], 'failed')
        self.assertIn('标定超时', self.high.zero_calibration['message'])
        self.assertLess(self.high.zero_calibration['samples'], 30)
        self.assertAlmostEqual(self.camera.offset, .1)

    def test_zero_cancels_on_disconnect_and_physical_stop(self):
        self.event('cart_yaw_zero')
        self.event('disconnect')
        self.assertEqual(self.high.zero_calibration['status'], 'failed')
        self.event('claim')
        self.event('cart_yaw_zero')
        self.bridge.state.buttons = p.B
        self.tick()
        self.assertEqual(self.high.zero_calibration['status'], 'failed')
        self.assertAlmostEqual(self.camera.offset, 0.)

    def test_camera_geometry_through_udp_matches_training_student_yaw(self):
        # CJME_Push_car 1b9574d: task_env.py uses wrap(cart_yaw - pelvis_yaw).
        # Exercise live waist/IMU compensation, UDP and the actual ONNX input.
        model, buffer = G1Kinematics(), StateBuffer()
        compensator = PelvisCompensator(buffer, model)
        estimator = CartHeadingEstimator(cart_to_tag_rpy_deg=[0, 0, 0])
        pub = HeadingPublisher(self.camera.sock.getsockname()[1])
        self.addCleanup(pub.close)
        for body_rpy in ((10, 20, 90), (-18, 25, -170)):
            for waist in (-35., 40.):
                for relative in (-179., 0., 35., 179.):
                    with self.subTest(body=body_rpy, waist=waist, relative=relative):
                        self.now += .02
                        state = self.bridge.state
                        state.received_ns = int(self.now * 1e9)
                        state.sequence += 1
                        state.q[12] = np.deg2rad(waist)
                        world_body = rotation(body_rpy)
                        cr, cp, cy = np.cos(np.deg2rad(body_rpy) / 2)
                        sr, sp, sy = np.sin(np.deg2rad(body_rpy) / 2)
                        state.quaternion = np.array([cr*cp*cy + sr*sp*sy, sr*cp*cy - cr*sp*sy,
                                                     cr*sp*cy + sr*cp*sy, cr*cp*sy - sr*sp*cy])
                        buffer.add(dict(received_ns=state.received_ns, tick=state.sequence,
                                        mode_pr=0, q=state.q[:15], quaternion=state.quaternion))
                        compensation = compensator.at_frame(self.now, self.now)
                        self.assertTrue(compensation['valid'])
                        body_camera = (rotation((0, 0, waist)) @
                                       rotation((0, np.rad2deg(.8307767239493009), 0)) @ OPTICAL_TO_LINK)
                        world_cart = rotation((4, -6, body_rpy[2] + relative))
                        camera_tag = body_camera.T @ world_body.T @ world_cart
                        measured = estimator.estimate({'R_tag_to_camera': camera_tag}, compensation)
                        self.assertTrue(measured['valid'])
                        result = camera_result(self.now)
                        result['cart_heading_settings']['camera_latency_ms'] = 0
                        result['pelvis_compensation'] = compensation
                        result['raw_fused_cart_heading'] = fuse_cart_headings(
                            [dict(id=i, cart_heading=measured) for i in (0, 1)])
                        result['fused_cart_heading'] = dict(result['raw_fused_cart_heading'], yaw_rad=.7)
                        pub.publish(result)
                        self.assertTrue(self.camera.poll(self.now))
                        with patch.object(self.high.model, 'run_student', wraps=self.high.model.run_student) as run:
                            self.high.warm(state, np.zeros(12), self.now)
                        self.assertEqual(self.high.warm_error, '')
                        observation, history = run.call_args.args
                        for frames in (observation.reshape(5, 126), history.reshape(32, 126)):
                            self.assertAlmostEqual(frames[-1, 94], np.deg2rad(relative), places=6)
                            np.testing.assert_allclose(frames[-1, 7:10], -world_body.T @ [0, 0, 1], atol=1e-7)

    def test_preparation_histories_survive_holding_policy_and_push_activation(self):
        self.bridge.state.stage = p.RAMPING
        self.control.request = p.PREPARE
        self.bridge.state.hand_flags = 35
        self.control.hand_pose = 0
        with patch('onnxruntime.InferenceSession', side_effect=AssertionError('late model load')):
            for index in range(40):
                self.bridge.state.q[0] = self.config.default_q[0] + index * .0001
                data = self.tick()
            self.assertTrue(data['model_warmup']['balance_homie'])
            self.assertTrue(data['high_policy']['model_warmed'])
            self.assertTrue(data['high_policy']['homie_model_warmed'])
            self.assertEqual(self.high.history_frames, 32)
            self.assertEqual(data['high_policy']['inference_role'], 'shadow')
            self.assertFalse(self.high.enabled)
            self.assertEqual(self.bridge.writes[-1][0], p.PREPARE)
            self.assertEqual(self.bridge.writes[-1][3], 0)
            for model in self.control.observations.models.values():
                np.testing.assert_array_equal(model.previous_action, 0)
            low = self.control.observations.models['homie'].history.copy()
            high = self.high.core.frames_long.copy()
            self.bridge.state.stage = p.HOLDING
            self.tick()
            np.testing.assert_array_equal(self.control.observations.models['homie'].history[:-1], low[1:])
            np.testing.assert_array_equal(self.high.core.frames_long[:-1], high[1:])
            self.event('policy')
            low = self.control.observations.models['homie'].history.copy()
            high = self.high.core.frames_long.copy()
            self.bridge.state.stage = p.POLICY
            self.tick()
            np.testing.assert_array_equal(self.control.observations.models['homie'].history[:-1], low[1:])
            np.testing.assert_array_equal(self.high.core.frames_long[:-1], high[1:])
            self.bridge.state.hand_flags = 55
            self.control.hand_pose = 2
            high = self.high.core.frames_long.copy()
            self.high.set_enabled(True, self.control, self.bridge.state, self.now)
            np.testing.assert_array_equal(self.high.core.frames_long, high)
            self.assertEqual(self.high.history_frames, 32)

    def test_shadow_actions_are_never_executed_or_written_as_previous_actions(self):
        self.bridge.state.stage = p.RAMPING
        self.control.request = p.PREPARE
        def student(*args):
            self.high.model.raw = np.full(30, .9, np.float32)
            self.high.model.inference_ms = 0.
            return self.high.model.raw.copy(), np.zeros(19, np.float32)
        self.high.model.run_student = student
        for _ in range(40): self.tick()
        np.testing.assert_array_equal(self.high.core.frames_long[:, 95:125], 0)
        np.testing.assert_array_equal(self.high.core.hl_actions, 0)
        np.testing.assert_array_equal(self.high.core.leg_residual, 0)
        np.testing.assert_array_equal(self.bridge.writes[-1][1], self.config.prepare_q)
        self.assertEqual(self.bridge.state.upper_mode, 0)

    def test_fresh_camera_and_full_history_are_required_before_push(self):
        self.refresh = False
        self.camera.stamp = -float('inf')
        for _ in range(40): self.tick()
        self.assertTrue(self.high.model.warmed)
        self.assertEqual(self.high.history_frames, 0)
        self.assertFalse(self.high.history_ready(self.now))
        self.refresh = True
        for _ in range(10): self.tick()
        self.event('pushcart_enable')
        self.assertFalse(self.high.enabled)
        self.assertIn('历史', self.control.reason)
        self.enable()
        self.assertTrue(self.high.history_ready(self.now))

    def test_shadow_failure_does_not_stop_ordinary_walking(self):
        original = self.high.model.run_student
        self.high.model.run_student = Mock(side_effect=RuntimeError('shadow failure'))
        data = self.move()
        self.assertEqual(data['stage'], 'POLICY')
        self.assertEqual(data['selected_model'], 'homie')
        self.assertEqual(self.high.history_frames, 0)
        self.assertEqual(data['high_policy']['warmup_error'], 'shadow failure')
        self.high.model.run_student = original
        for _ in range(32): self.tick()
        self.assertTrue(self.high.history_ready(self.now))

    def test_preparation_requests_read_only_camera_in_background(self):
        self.control.mode = 'control'  # FakeBridge still has no hardware transport.
        camera_process = Mock()
        camera_process.snapshot.return_value = {}
        self.high.camera_process = camera_process
        self.refresh = False
        self.camera.stamp = -float('inf')
        self.bridge.state.stage = p.WAITING
        self.control.request = p.WAIT
        self.tick()
        self.event('prepare')
        camera_process.request_start.assert_called_once_with()
        self.assertEqual(self.control.request, p.PREPARE)
        self.assertFalse(self.high.enabled)
        self.assertEqual(self.control.hand_pose, 0)

    def test_normal_walking_does_not_need_camera_or_grasp(self):
        self.refresh = False
        self.camera.stamp = -float('inf')
        self.bridge.state.hand_flags = 35
        data = self.move()
        self.assertEqual(data['selected_model'], 'homie')
        self.assertIsNone(data['high_policy']['raw_actions'])
        self.assertFalse(self.high.enabled)
        self.assertEqual(self.control.hand_pose, 0)
        np.testing.assert_array_equal(self.config.velocity_bounds, self.config.deployment['command_limits'])

    def test_camera_socket_failure_keeps_normal_body_control(self):
        self.camera.sock.close()
        data = self.move()
        self.assertEqual(data['stage'], 'POLICY')
        self.assertEqual(data['selected_model'], 'homie')
        self.assertFalse(data['high_policy']['camera']['fresh'])

    def test_camera_socket_failure_stops_push_smoothly(self):
        self.enable()
        self.move()
        self.camera.sock.close()
        stopping = self.tick()
        self.assertTrue(stopping['decelerating'])
        self.assertEqual(stopping['homie_estimator_source'], 'pushcart_finetuned')
        data = self.move(120)
        self.assertEqual(data['stage'], 'POLICY')
        self.assertTrue(self.high.motion_blocked)
        self.assertEqual(data['selected_model'], 'balance')
        self.assertEqual(data['homie_estimator_source'], 'pushcart_finetuned')
        normal = self.event('pushcart_disable')
        self.assertEqual(normal['homie_estimator_source'], 'homie_original')

    def test_duplicate_mode_requests_do_not_reset_history_or_velocity(self):
        self.enable()
        self.move()
        before = self.high.core.frames_long.copy()
        data = self.event('pushcart_enable')
        np.testing.assert_array_equal(self.high.core.frames_long[:-1], before[1:])
        self.assertTrue(np.any(data['velocity']))
        self.event('zero')
        for _ in range(120): self.tick()
        self.event('pushcart_disable')
        self.move()
        data = self.event('pushcart_disable')
        self.assertTrue(np.any(data['velocity']))

    def test_grasp_editor_is_locked_only_while_pushing_and_angles_survive(self):
        initial = self.config.hand_grasp.copy()
        self.enable()
        self.control.emit('hand_edit', 1, joint=0, angle_rad=.1)
        self.tick()
        np.testing.assert_array_equal(self.config.hand_grasp, initial)
        self.assertFalse(self.control.snapshot()['dex3']['editor_available'])
        self.event('pushcart_disable')
        self.assertFalse(self.high.enabled)
        np.testing.assert_array_equal(self.config.hand_grasp, initial)
        self.control.emit('hand_edit', 1, joint=0, angle_rad=.1)
        self.tick()
        self.assertEqual(self.config.hand_grasp.flat[0], .1)
        self.assertTrue(self.control.snapshot()['dex3']['editor_available'])

    def test_camera_pause_can_return_to_normal_without_camera(self):
        self.enable()
        self.move()
        self.refresh = False
        self.move(120)
        self.assertTrue(self.high.motion_blocked)
        self.event('pushcart_disable')
        self.assertFalse(self.high.motion_blocked)
        data = self.move()
        self.assertEqual(data['selected_model'], 'homie')
        self.assertIsNone(data['high_policy']['raw_actions'])

    def test_mode_switch_never_enables_or_rewrites_hands(self):
        before = self.config.hand_grasp.copy()
        self.enable()
        self.assertEqual(self.control.hand_pose, 2)
        self.event('pushcart_disable')
        self.assertEqual(self.control.hand_pose, 2)
        np.testing.assert_array_equal(self.config.hand_grasp, before)
        self.assertFalse(self.control.hand_live)

    def test_enable_requires_owner_grasp_camera_and_trained_pose(self):
        self.event('pushcart_toggle', client=2)
        self.assertFalse(self.high.enabled)
        self.bridge.state.hand_flags = 35
        self.event('pushcart_toggle')
        self.assertFalse(self.high.enabled)
        self.bridge.state.hand_flags = 55
        self.control.prepare_pose = 'arms_down'
        self.event('pushcart_toggle')
        self.assertFalse(self.high.enabled)
        self.control.prepare_pose = 'original'
        self.refresh = False
        self.camera.stamp = -float('inf')
        self.event('pushcart_toggle')
        self.assertFalse(self.high.enabled)

    def test_real_student_receives_yaw_and_commits_selected_residual(self):
        self.enable()
        data = self.move()
        self.assertEqual(data['selected_model'], 'homie')
        self.assertEqual(data['high_policy']['raw_actions'].shape, (30,))
        self.assertEqual(self.high.core.frames.shape, (5, 126))
        self.assertEqual(self.high.core.frames_long.shape, (32, 126))
        self.assertAlmostEqual(float(self.high.core.frames[-1, 94]), .2, places=6)
        action = np.clip(data['model_actions']['homie'] + self.high.applied_residual, -100, 100)
        for model in self.control.observations.models.values():
            np.testing.assert_allclose(model.previous_action, action, atol=1e-6)
        expected = self.config.default_q[:12] + .25 * action
        expected = np.clip(expected, self.config.deployment['joint_lower_sdk29'][:12],
                           self.config.deployment['joint_upper_sdk29'][:12])
        np.testing.assert_allclose(data['requested_target'][:12], expected, atol=1e-6)
        np.testing.assert_allclose(data['requested_target'][UPPER], self.high.output.q_upper, atol=1e-6)
        self.assertTrue(self.bridge.upper_target_active)
        self.assertGreaterEqual(data['timing']['inference_ms'], data['high_policy']['inference_ms'])

    def test_torque_uses_prior_servo_sample_and_reports_stale_fallback(self):
        self.enable()
        state = self.bridge.state
        previous = replace(state, received_ns=state.received_ns - 5_000_000,
                           applied_q=state.q + .1, applied_kp=np.full(29, 20.),
                           applied_kd=np.zeros(29))
        self.high.command(state, [.3, 0, 0], np.zeros(12), self.now,
                          moving=True, dt=.02, previous_state=previous)
        np.testing.assert_allclose(self.high.core.frames[-1, 79:94], .04, atol=1e-6)
        self.assertAlmostEqual(self.high.torque_sample_age_ms, 5.)
        stale = replace(previous, received_ns=state.received_ns - 50_000_000)
        self.high.command(state, [.3, 0, 0], np.zeros(12), self.now,
                          moving=True, dt=.02, previous_state=stale)
        self.assertEqual(self.high.torque_sample_age_ms, 0.)
        np.testing.assert_array_equal(self.high.core.frames[-1, 79:94], 0)

    def test_zero_decelerates_and_balance_has_no_residual(self):
        self.enable()
        self.move()
        previous = self.control.snapshot()['velocity']
        self.event('zero')
        for _ in range(100):
            data = self.tick()
            self.assertFalse(self.bridge.upper_target_active)
            self.assertTrue(np.all(np.abs(data['velocity']) <= np.abs(previous) + 1e-7))
            self.assertTrue(np.all(data['velocity'] * previous >= 0))
            previous = data['velocity']
        self.assertEqual(data['selected_model'], 'balance')
        np.testing.assert_array_equal(data['velocity'], 0)
        np.testing.assert_array_equal(self.high.applied_residual, 0)
        np.testing.assert_allclose(self.high.core.applied, data['model_actions']['balance'], atol=1e-6)

    def test_cart_heading_corrects_task_and_homie_passes_unchanged_into_histories(self):
        self.assertEqual(self.control.heading.settings['run_mode'], 'active')
        self.enable()
        previous = np.zeros(3)
        scale = np.asarray(self.config.contracts['homie']['input']['command_scale'][:3])
        for i in range(100):
            # Simulate leftward cart drift while the user requests straight
            # pushing. Only the input to the task-reference filter is corrected.
            yaw = min(i * .002, .1)
            self.bridge.state.quaternion = np.array([np.cos(yaw/2), 0., 0., np.sin(yaw/2)])
            self.bridge.state.omega = np.array([0., 0., .1 if i < 50 else 0.])
            data = self.event('move', [.3, 0., 0.])
            self.assertEqual(data['heading']['command_source'], 'PUSHCART')
            self.assertFalse(data['heading']['correction_enabled'])
            self.assertFalse(data['heading']['pushcart_tracking'])
            self.assertEqual(data['heading']['correction_rad_s'], 0.)
            high = data['high_policy']
            hold = high['heading_hold']
            self.assertAlmostEqual(hold['imu_yaw_rad'], data['heading']['imu_yaw_rad'])
            # Cart yaw pairs the camera yaw with the pelvis yaw at its capture
            # time (40 ms latency here), not the newer current pelvis yaw.
            captured = hold['imu_yaw_at_camera_rad']
            self.assertTrue(yaw - .0041 <= captured <= yaw + 1e-9, (yaw, captured))
            self.assertAlmostEqual(hold['cart_yaw_rad'], captured + high['camera']['yaw_rad'])
            self.assertAlmostEqual(hold['err'], -captured)
            self.assertAlmostEqual(hold['corr'], -captured)
            np.testing.assert_allclose(high['raw_task_command'], [.3, 0., 0.], atol=1e-7)
            np.testing.assert_allclose(high['task_command'], [.3, 0., -captured], atol=1e-7)
            old_ref = high['inference_inputs']['previous_task_reference']
            np.testing.assert_allclose(high['task_reference'],
                                       old_ref + self.high.core.ref_alpha * (high['task_command'] - old_ref))
            np.testing.assert_allclose(data['input_velocity'], self.high.output.cmd_homie)
            np.testing.assert_allclose(data['filtered_velocity'], self.high.output.cmd_homie, atol=1e-7)
            np.testing.assert_allclose(self.high.core.frames[-1, :3], previous * scale, atol=1e-7)
            np.testing.assert_allclose(high['inference_inputs']['previous_homie_command'], previous, atol=1e-7)
            if data['selected_model'] == 'homie':
                np.testing.assert_allclose(data['velocity'], self.high.output.cmd_homie, atol=1e-7)
                self.assertAlmostEqual(data['velocity'][2], data['heading']['wz_final_applied'], places=6)
                np.testing.assert_allclose(self.control.observations.models['homie'].history[-1, :3],
                                           data['velocity'] * scale, atol=1e-7)
            previous = data['velocity'].copy()
        self.assertEqual(data['selected_model'], 'homie')
        self.assertAlmostEqual(data['high_policy']['heading_hold']['corr'], -.1)

    def test_cart_reference_integrates_raw_task_wz_using_actual_control_dt(self):
        self.enable()
        self.camera.set_offset(.1)
        expected_ref = .1
        for index, dt in enumerate([.013, .037, .011, .029] * 5):
            wz = .3 if index < 10 else -.3
            self.control.emit('move', 1, [.3, 0., wz])
            data = self.tick(dt=dt)
            high = data['high_policy']
            expected_ref += wz * dt
            self.assertAlmostEqual(high['heading_hold']['dt_s'], dt)
            self.assertAlmostEqual(high['heading_hold']['ref'], expected_ref, places=7)
            self.assertAlmostEqual(high['heading_hold']['cart_yaw_rad'], .1)
            self.assertAlmostEqual(high['task_command'][2], wz + expected_ref - .1, places=7)

    def test_cart_hold_uses_held_camera_yaw_and_releases_on_timeout(self):
        self.enable()
        self.move()
        self.refresh = False
        for i in range(5):
            yaw = .01 * (i + 1)
            self.bridge.state.quaternion = np.array([np.cos(yaw/2), 0., 0., np.sin(yaw/2)])
            data = self.event('move', [.3, 0., 0.])
            self.assertTrue(data['high_policy']['camera']['fresh'])
            # The held frame keeps its capture time: the pelvis turning after it
            # (gait sway) is not cart drift and must not create a correction.
            self.assertAlmostEqual(data['high_policy']['heading_hold']['corr'], 0.)
        for _ in range(20):
            data = self.tick()
        self.assertTrue(data['high_policy']['motion_blocked'])
        self.assertIsNone(self.high.hold.ref)
        self.assertEqual(self.high.hold.corr, 0.)

    def test_pushcart_starts_at_zero_with_active_heading_feedback(self):
        self.assertEqual(self.control.heading.settings['run_mode'], 'active')
        self.enable()
        for i in range(80):
            yaw = .03 * np.sin(i * .05)
            self.bridge.state.quaternion = np.array([np.cos(yaw/2), 0., 0., np.sin(yaw/2)])
            data = self.tick()
            self.assertEqual(data['selected_model'], 'balance')
            for key in ('input_velocity', 'filtered_velocity', 'velocity'):
                np.testing.assert_array_equal(data[key], 0.)
            self.assertFalse(data['heading']['correction_enabled'])
            self.assertEqual(data['heading']['correction_rad_s'], 0.)
            self.assertIsNone(data['high_policy']['heading_hold']['ref'])
            self.assertEqual(data['high_policy']['heading_hold']['corr'], 0.)
            np.testing.assert_array_equal(data['high_policy']['applied_leg_residual'], 0.)
            self.assertFalse(data['high_policy']['upper_target_selected'])

    def test_held_cart_zero_uses_viser_stop_and_zero_balance_observation(self):
        self.control.heading.configure(dict(self.control.heading.settings, run_mode='active'),
                                       self.bridge.state.quaternion)
        self.enable()
        self.move()
        previous = self.control.snapshot()['velocity'].copy()
        self.assertGreater(np.linalg.norm(previous), .05)
        switched = False
        for index in range(100):
            data = self.event('move', [0., 0., 0.])
            if index == 0:
                # The native high-level filter retains a tail. The adapter must
                # request a stop instead of passing that tail to locomotion.
                self.assertGreater(np.linalg.norm(self.high.output.cmd_homie), 0.)
            np.testing.assert_array_equal(data['input_velocity'], 0.)
            expected = previous * .95  # 50 Hz, the existing 0.4 s stop filter.
            if np.linalg.norm(expected) < .05:
                expected.fill(0.)
            np.testing.assert_allclose(data['velocity'], expected, atol=1e-7)
            np.testing.assert_array_equal(self.high.applied_residual, 0.)
            np.testing.assert_allclose(data['requested_target'][UPPER], self.bridge.state.applied_q[UPPER], atol=1e-6)
            self.assertEqual(data['heading']['correction_rad_s'], 0.)
            name = data['selected_model']
            command = self.control.observations.models[name].history[-1, :3]
            scale = np.asarray(self.config.contracts[name]['input']['command_scale'][:3])
            np.testing.assert_allclose(command, data['velocity'] * scale, atol=1e-7)
            if name == 'balance':
                switched = True
                np.testing.assert_array_equal(command, 0.)
            elif switched:
                self.fail('A held zero must not restart Homie after Balance takes over')
            previous = data['velocity'].copy()
        self.assertTrue(switched)

    def test_lost_camera_cannot_resume_on_held_or_recovered_commands(self):
        self.enable()
        self.move()
        self.refresh = False
        self.move(100)
        self.assertFalse(self.high.enabled)
        self.assertFalse(self.bridge.upper_target_active)
        self.assertTrue(self.high.motion_blocked)
        self.assertEqual(self.control.snapshot()['selected_model'], 'balance')
        self.refresh = True
        data = self.move(40)
        np.testing.assert_array_equal(data['velocity'], 0)
        self.assertTrue(self.high.motion_blocked)
        self.assertIn('航向已恢复', data['high_policy']['reason'])
        self.assertNotIn('超时', data['high_policy']['reason'])
        self.assertEqual(data['stage'], 'POLICY')
        self.assertIn('航向已恢复', data['reason'])
        self.event('zero')
        self.enable()
        self.assertFalse(self.high.motion_blocked)

    def test_physical_b_immediately_zeros_pushcart_and_selects_balance(self):
        self.enable()
        before = self.move()
        self.assertEqual(before['selected_model'], 'homie')
        self.assertTrue(np.any(before['velocity']))
        self.bridge.state.buttons = p.B
        stopped = self.tick()
        self.assertFalse(self.bridge.upper_target_active)
        np.testing.assert_array_equal(stopped['velocity'], 0)
        np.testing.assert_array_equal(self.high.applied_residual, 0)
        self.assertEqual(stopped['selected_model'], 'balance')
        self.assertFalse(stopped['decelerating'])
        np.testing.assert_allclose(stopped['requested_target'][UPPER], self.bridge.state.applied_q[UPPER], atol=1e-6)
        self.bridge.state.buttons = 0
        np.testing.assert_array_equal(self.tick()['velocity'], 0)

    def test_hand_failure_only_stops_push_task_not_body_control(self):
        self.enable()
        self.move()
        self.bridge.state.hand_flags = 32
        data = self.move(100)
        self.assertEqual(data['stage'], 'POLICY')
        self.assertTrue(data['healthy'])
        self.assertTrue(self.high.motion_blocked)
        np.testing.assert_array_equal(data['velocity'], 0)

    def test_stop_clears_high_and_wins_over_queued_motion(self):
        self.enable()
        self.move()
        self.control.emit('move', 1, [.4, 0, 0])
        data = self.event('stop')
        self.assertEqual(data['stage'], 'DAMPED')
        self.assertFalse(self.high.enabled)
        self.assertIsNone(data['high_policy']['raw_actions'])
        self.assertEqual(self.bridge.writes[-1][0], p.STOP)


if __name__ == '__main__':
    unittest.main()
