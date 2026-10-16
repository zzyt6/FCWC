"""Offline geometry, timing, dropout and moving-camera regression checks."""
import contextlib
import io
import json
from pathlib import Path
import queue
import sys
import tempfile
import time
import unittest
from unittest import mock

import cv2
import numpy as np

from carmer import apriltag_detector as app
from carmer.cart_heading import CartHeadingEstimator, HeadingSmoother, rotation_from_rpy_degrees as rotation
from carmer.pelvis_attitude import DEFAULT_URDF, G1Kinematics, OPTICAL_TO_LINK, PelvisCompensator
from carmer.robot_feedback import RobotFeedback, StateBuffer


def packet(stamp=10., tick=1, angles=None, rpy=(0, 0, 0), quaternion=None):
    return dict(received_ns=int(stamp * 1e9), tick=tick, mode_pr=0,
                q=np.zeros(15) if angles is None else angles,
                quaternion=app._rotation_to_quaternion(rotation(rpy)) if quaternion is None else quaternion)


class AttitudeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = G1Kinematics()

    def compensate(self, record, source='imu', **kwargs):
        buffer = StateBuffer()
        self.assertTrue(buffer.add(record))
        return PelvisCompensator(buffer, self.model, source=source, **kwargs).at_frame(10., 10.01)

    def test_live_imu_and_waist_remove_pitch_roll_and_world_heading(self):
        estimator = CartHeadingEstimator(cart_to_tag_rpy_deg=[0, 0, 0])
        for rpy in ((10, 20, 0), (-18, 25, 90), (25, -15, -170)):
            for waist in (-35., 0., 40.):
                for cart_relative in (-179., 0., 35., 179.):
                    with self.subTest(rpy=rpy, waist=waist, cart=cart_relative):
                        joints = np.zeros(15)
                        joints[12] = np.deg2rad(waist)
                        # Locked waist channels must not affect the camera, even if garbage.
                        joints[13:] = [np.nan, 2.]
                        state = self.compensate(packet(angles=joints, rpy=rpy))
                        self.assertTrue(state['valid'])
                        world_body = rotation(rpy)
                        world_cart = rotation((0, 0, rpy[2] + cart_relative))
                        body_camera = rotation((0, 0, waist)) @ app.CartHeadingEstimator().settings['R_camera_to_torso']
                        observed = body_camera.T @ world_body.T @ world_cart
                        heading = estimator.estimate({'R_tag_to_camera': observed}, state)
                        self.assertTrue(heading['valid'])
                        error = np.arctan2(np.sin(heading['yaw_rad'] - np.deg2rad(cart_relative)),
                                           np.cos(heading['yaw_rad'] - np.deg2rad(cart_relative)))
                        self.assertAlmostEqual(error, 0., places=10)
                        self.assertEqual(heading['reference_frame'], 'pelvis_level')
        # Demonstrate the old local-XY result's tilt error for the user's case.
        observed = estimator.camera_to_pelvis.T @ rotation((10, 20, 0)).T
        self.assertAlmostEqual(estimator.estimate({'R_tag_to_camera': observed})['yaw_deg'], 3.616441573, places=6)

    def test_kinematics_uses_urdf_joint_origins_and_does_not_require_imu(self):
        joints = np.zeros(15)
        joints[4], joints[5], joints[12] = np.deg2rad([-20., -10., 30.])
        state = self.compensate(packet(angles=joints, quaternion=[0, 0, 0, 0]),
                                source='kinematics', support='left')
        self.assertTrue(state['valid'])
        expected_foot = rotation((0, -20, 0)) @ rotation((-10, 0, 0))
        np.testing.assert_allclose(state['up_in_pelvis'], expected_foot[:, 2], atol=1e-12)
        expected_camera = rotation((0, 0, 30)) @ rotation((0, np.rad2deg(.8307767239493009), 0)) @ OPTICAL_TO_LINK
        np.testing.assert_allclose(state['R_camera_to_pelvis'], expected_camera, atol=1e-12)
        for source in ('imu', 'fused'):
            self.assertEqual(self.compensate(packet(quaternion=[0, 0, 0, 0]), source=source)['reason'], 'invalid_pelvis_imu')

    def test_auto_support_recognizes_double_and_alternating_support(self):
        state = self.compensate(packet(), source='kinematics')
        self.assertEqual(state['support_foot'], 'both')
        for lifted, expected in ((6, 'left'), (0, 'right')):
            joints = np.zeros(15)
            joints[lifted], joints[lifted + 3], joints[lifted + 4] = -.3, .6, -.3
            state = self.compensate(packet(angles=joints), source='kinematics')
            self.assertTrue(state['valid'])
            self.assertEqual(state['support_foot'], expected)
            np.testing.assert_allclose(state['up_in_pelvis'], [0, 0, 1], atol=1e-12)

    def test_fusion_and_explicit_fallback_on_ambiguous_or_disagreeing_feet(self):
        state = self.compensate(packet(rpy=(0, 5, 120)), source='fused', imu_weight=.8)
        expected = .8 * (rotation((0, 5, 0)).T @ [0, 0, 1]) + .2 * np.array([0, 0, 1])
        expected /= np.linalg.norm(expected)
        np.testing.assert_allclose(state['up_in_pelvis'], expected, atol=1e-12)
        self.assertEqual(state['effective_source'], 'fused')
        state = self.compensate(packet(rpy=(0, 30, 0)), source='fused')
        self.assertEqual(state['effective_source'], 'imu')
        self.assertEqual(state['reason'], 'imu_only_kinematics_disagreement')
        joints = np.zeros(15)
        joints[5], joints[11] = .3, -.3
        state = self.compensate(packet(angles=joints), source='fused')
        self.assertEqual(state['effective_source'], 'imu')
        self.assertEqual(state['reason'], 'imu_only_kinematics_unavailable')
        self.assertFalse(self.compensate(packet(angles=joints), source='kinematics')['valid'])

    def test_bad_waist_and_ab_leg_feedback_invalidate_the_relevant_modes(self):
        joints = np.zeros(15)
        joints[12] = np.nan
        for source in ('imu', 'kinematics', 'fused'):
            self.assertFalse(self.compensate(packet(angles=joints), source=source)['valid'])
        record = packet()
        record['mode_pr'] = 1
        self.assertFalse(self.compensate(record, source='kinematics')['valid'])
        self.assertTrue(self.compensate(record, source='imu')['valid'])
        self.assertEqual(self.compensate(record, source='fused')['effective_source'], 'imu')

    def test_horizontal_heading_degeneracy_is_unavailable(self):
        estimator = CartHeadingEstimator(cart_to_tag_rpy_deg=[0, 0, 0])
        state = self.compensate(packet(rpy=(0, 90, 0)))
        self.assertFalse(estimator.estimate({'R_tag_to_camera': np.eye(3)}, state)['valid'])
        state = self.compensate(packet())
        observed = np.asarray(state['R_camera_to_pelvis']).T @ rotation((0, 90, 0))
        self.assertEqual(estimator.estimate({'R_tag_to_camera': observed}, state)['reason'], 'forward_axis_vertical')


class TimingTest(unittest.TestCase):
    def test_invalid_imu_in_interpolation_cannot_reuse_previous_attitude(self):
        buffer = StateBuffer()
        buffer.add(packet(9.99, 1))
        buffer.add(packet(10.01, 2, quaternion=[None, 0, 0, 0]))
        state = PelvisCompensator(buffer, G1Kinematics()).at_frame(10., 10.02)
        self.assertFalse(state['valid'])
        self.assertEqual(state['reason'], 'invalid_pelvis_imu')

    def test_interpolates_at_image_time_instead_of_using_latest(self):
        buffer = StateBuffer()
        qa, qb = np.zeros(15), np.zeros(15)
        qb[12] = .4
        buffer.add(packet(9.98, 1, qa, (0, 0, 170)))
        buffer.add(packet(10.02, 2, qb, (0, 0, -170)))
        sample, meta = buffer.at(10., 10.03, .1, .03)
        self.assertEqual(meta['alignment'], 'interpolated')
        self.assertAlmostEqual(sample['q'][12], .2)
        rotation_mid = app._quaternion_to_rotation(sample['quaternion'])
        np.testing.assert_allclose(rotation_mid, rotation((0, 0, 180)), atol=1e-10)
        self.assertEqual(meta['time_basis'], 'host_monotonic_receive_proxy')

    def test_missing_stale_mismatched_future_and_duplicate_samples(self):
        buffer = StateBuffer()
        self.assertEqual(buffer.at(10., 10., .1, .03)[1]['reason'], 'robot_state_missing')
        self.assertTrue(buffer.add(packet()))
        self.assertFalse(buffer.add(packet(10.05, 1)))
        self.assertFalse(buffer.add(packet(10.06, 0)))
        self.assertFalse(buffer.add(packet(9.99, 2)))
        self.assertEqual(buffer.at(10., 10.15, .1, .03)[1]['reason'], 'robot_state_stale')
        self.assertEqual(buffer.at(9.8, 10.01, .1, .03)[1]['reason'], 'image_state_time_mismatch')
        self.assertEqual(buffer.at(10., 9.9, .1, .03)[1]['reason'], 'robot_state_clock_error')
        # Duplicate data did not refresh the actual last receive time.
        self.assertEqual(len(buffer.samples), 1)
        rollover = StateBuffer()
        self.assertTrue(rollover.add(packet(10., 0xffffffff)))
        self.assertTrue(rollover.add(packet(10.01, 0)))

    def test_camera_latency_matches_older_state(self):
        buffer = StateBuffer()
        buffer.add(packet(9.9, 1, rpy=(0, 10, 0)))
        buffer.add(packet(10., 2, rpy=(0, 30, 0)))
        state = PelvisCompensator(buffer, G1Kinematics(), camera_latency_s=.1).at_frame(10., 10.01)
        self.assertAlmostEqual(state['pitch_deg'], 10.)
        self.assertAlmostEqual(state['target_time'], 9.9)


class PipelineTest(unittest.TestCase):
    def test_subscriber_pipe_reads_timestamps_and_process_is_closed(self):
        # Stand-in process exercises the actual pipe/thread/buffer lifecycle,
        # without requiring network access or publishing anything to DDS.
        with tempfile.TemporaryDirectory() as folder:
            executable = Path(folder) / 'feedback'
            executable.write_text(f'#!{sys.executable}\n' +
                'import json,time\n'
                'print("SDK diagnostic",flush=True)\n'
                'print("{malformed",flush=True)\n'
                'for i in range(500):\n'
                ' print(json.dumps(dict(received_ns=time.monotonic_ns(),tick=i,mode_pr=0,'
                'q=[0.0]*15,quaternion=[1,0,0,0])),flush=True)\n'
                ' time.sleep(.01)\n')
            executable.chmod(0o700)
            with mock.patch('carmer.robot_feedback.ensure_bridge', return_value=executable):
                feedback = RobotFeedback('unused')
            try:
                deadline = time.monotonic() + 2
                sample = None
                while time.monotonic() < deadline:
                    now = time.monotonic()
                    sample, meta = feedback.at(now, now, .1, .03)
                    if sample is not None:
                        break
                    time.sleep(.01)
                self.assertIsNotNone(sample)
                self.assertTrue(meta['valid'])
                self.assertEqual(sample['q'].shape, (15,))
            finally:
                feedback.close()
            self.assertIsNotNone(feedback.process.poll())
            self.assertFalse(feedback.thread.is_alive())

    def test_moving_camera_compensates_raw_pose_before_smoothing_and_clears_loss(self):
        incoming = queue.Queue()
        class Reader:
            def latest(self, previous, with_timestamp=False):
                return incoming.get(timeout=3)
            def close(self):
                incoming.put(None)
        class Compensator:
            state = None
            def at_frame(self, timestamp, now):
                return self.state
        comp = Compensator()
        matrix = np.array([[600., 0, 320], [0, 600., 240], [0, 0, 1.]])
        pose_estimator = app.TagPoseEstimator(matrix, np.zeros(5))
        heading = CartHeadingEstimator(cart_to_tag_rpy_deg=[0, 0, 0])
        smoother = HeadingSmoother()
        detection = mock.Mock()
        # Mock only marker extraction; exercise actual pose smoothing, compensation,
        # circular fusion, timestamps, worker packet, profiling and snapshot serialization.
        with mock.patch.object(app, 'detect_tags', detection):
            worker = app.DetectionWorker(Reader(), app.make_detector(), pose_estimator,
                        app.AdaptivePoseSmoother(pose_estimator), heading,
                        pelvis_compensator=comp, heading_smoother=smoother, profile=True)
            try:
                for n, (waist, ids, valid) in enumerate(((0, [0, 1], True), (40, [0], True),
                                                        (40, [0], False), (40, [], True), (0, [1], True)), 1):
                    buffer = StateBuffer()
                    joints = np.zeros(15)
                    joints[12] = np.deg2rad(waist)
                    buffer.add(packet(angles=joints, rpy=(10, 20, 0)))
                    comp.state = PelvisCompensator(buffer, G1Kinematics()).at_frame(10., 10.01)
                    if not valid:
                        comp.state.update(valid=False, reason='robot_state_stale')
                    camera = np.asarray(comp.state['R_camera_to_pelvis'])
                    observed = camera.T @ rotation((10, 20, 0)).T @ rotation((0, 0, 25))
                    pose = pose_estimator.pose_from_transform(observed, np.array([0., 0., .7]), 0.)
                    tags = [dict(id=i, center_px=[320., 240.], pose=pose,
                                 corners_px=[[280., 200.], [360., 200.], [360., 280.], [280., 280.]]) for i in ids]
                    detection.return_value = tags
                    incoming.put((n, np.zeros((480, 640, 3), np.uint8), 60., 10. + n * .02))
                    result = worker.latest(n-1)
                    if valid and ids:
                        self.assertAlmostEqual(result['fused_cart_heading']['yaw_deg'], 25., places=8)
                        self.assertEqual(result['fused_cart_heading']['tag_ids'], ids)
                    else:
                        self.assertFalse(result['fused_cart_heading']['valid'])
                        self.assertIsNone(result['fused_cart_heading']['yaw_rad'])
                        self.assertEqual(smoother.states, {})
                    self.assertIn('attitude', result['stage_timings_ms'])
                    self.assertEqual(result['fused_cart_heading']['reference_frame'], 'pelvis_level')
                    with tempfile.TemporaryDirectory() as folder:
                        app.save_result(result, Path(folder))
                        saved = json.loads(next(Path(folder).glob('*.json')).read_text())
                        self.assertEqual(saved['pelvis_compensation'], result['pelvis_compensation'])
            finally:
                worker.close()

    def test_config_selects_modes_and_validates_parameters(self):
        self.assertEqual(app.parse_args([]).pelvis_attitude_source, 'imu')
        self.assertEqual(app.parse_args([]).robot_urdf, DEFAULT_URDF)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'cfg.json'
            for source in ('imu', 'kinematics', 'fused', 'fixed'):
                path.write_text(json.dumps({'pelvis-attitude-source': source, 'robot-urdf': 'custom.urdf'}))
                args = app.parse_args(['--config', str(path)])
                self.assertEqual(args.pelvis_attitude_source, source)
                self.assertEqual(args.robot_urdf, Path(folder) / 'custom.urdf')
                self.assertEqual(app.parse_args(['--config', str(path), '--pelvis-attitude-source', 'imu']).pelvis_attitude_source, 'imu')
            for key, value in (('fusion-imu-weight', 1), ('camera-latency-ms', -1),
                               ('state-sync-tolerance-ms', 0), ('fusion-max-disagreement-deg', 90),
                               ('kinematics-support-foot', 'wrong'), ('robot-state-timeout-ms', '100')):
                path.write_text(json.dumps({key: value}))
                with self.subTest(key=key), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    app.parse_args(['--config', str(path)])

    def test_feedback_closes_on_camera_startup_failure(self):
        args = app.parse_args(['--no-display'])
        with mock.patch.object(app, 'parse_args', return_value=args), \
                mock.patch.object(app, 'RobotFeedback') as feedback, \
                mock.patch.object(app, 'open_camera', side_effect=RuntimeError('camera_busy')), \
                contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, 'camera_busy'):
                app.main()
            feedback.return_value.close.assert_called_once()


if __name__ == '__main__':
    unittest.main()
