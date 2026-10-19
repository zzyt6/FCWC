"""Nav UDP -> shared push policy -> real ONNX, using synthetic robot/camera only."""
import json
import copy
import math
import socket
import sys
import unittest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from g1_locomotion_executor import Executor, NAV_CART_CONFIG, load_settings, VELOCITY_PACKET, VELOCITY_MAGIC, VELOCITY_VERSION

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'G1_high_policy_deploy/test'))
import test_viser_integration as fixtures


class NavPushcartTests(unittest.TestCase):
    event = fixtures.IntegrationTests.event

    def test_missing_tag_recording_follows_pushcart_enable(self):
        camera = Mock()
        camera.snapshot.return_value = {}
        self.high.camera_process = camera
        self.tick()
        self.assertTrue(camera.record_missing_tags.call_args.args[0])
        self.operator('pushcart_disable')
        self.tick()
        self.assertFalse(camera.record_missing_tags.call_args.args[0])

    def setUp(self):
        fixtures.IntegrationTests.setUp(self)
        self.status = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.status.bind(('127.0.0.1', 0))
        self.status.setblocking(False)
        self.sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # The synthetic camera initially supplies heading only. Do not depend
        # on (or overwrite) the operator's persisted AprilTag/grasp selection.
        directory = tempfile.TemporaryDirectory(prefix='nav-pushcart-test-')
        self.addCleanup(directory.cleanup)
        config_path = Path(directory.name) / 'navigation.json'
        settings = load_settings(NAV_CART_CONFIG)
        settings['pose_source'] = 'grasp'
        config_path.write_text(json.dumps(settings))
        with patch('g1_locomotion_executor.load_settings', return_value=settings):
            self.executor = Executor(self.control, 0, self.status.getsockname()[1],
                                     clock=lambda: self.now, cart_config=config_path)
        self.addCleanup(self.executor.close)
        self.addCleanup(self.sender.close)
        self.addCleanup(self.status.close)
        self.sequence = 0
        for _ in range(45):
            self.tick()
        self.walk_epoch = self.executor.navigation_epoch
        self.executor.request('pushcart_enable')
        for _ in range(65):
            self.tick()
        self.assertTrue(self.high.enabled, self.control.reason)
        self.assertTrue(self.executor.cart_attached)
        self.assertNotEqual(self.walk_epoch, self.executor.navigation_epoch)

    def tick(self, yaw=.2, valid=True):
        data = fixtures.IntegrationTests.tick(self, yaw, valid)
        if hasattr(self, 'executor'):
            while True:
                try:
                    self.status.recv(65535)
                except BlockingIOError:
                    break
            self.executor.step()
        return data

    def test_grasp_navigation_aligns_delayed_heading_with_pelvis_history(self):
        data = copy.deepcopy(self.control.snapshot())
        start = self.now + 1.
        errors = []
        for index in range(100):
            now = start + index*.02
            captured = now-.12
            pelvis_yaw = .08*math.sin(2*math.pi*2*now)
            old_pelvis_yaw = .08*math.sin(2*math.pi*2*captured)
            data['state'].update(received=now, quat=np.array([
                math.cos(pelvis_yaw/2), 0., 0., math.sin(pelvis_yaw/2)]))
            camera = data['high_policy']['camera']
            camera.update(fresh=True, captured_monotonic=captured, yaw_rad=.3-old_pelvis_yaw)
            cart = self.executor._cart_status(data, now)
            if index >= 10:
                self.assertTrue(cart['valid'], cart)
                errors.append(cart['yaw']+pelvis_yaw-.3)
                self.assertEqual(cart['captured_monotonic'], captured)
                self.assertEqual(camera['yaw_rad'], .3-old_pelvis_yaw)
        self.assertLess(max(map(abs, errors)), .001,
                        'A straight cart must not inherit pelvis sway from camera delay')

    def test_grasp_geometry_matches_viser_with_identical_feedback(self):
        from scipy.spatial.transform import Rotation
        from core.robot import JOINT_NAMES
        from Vender.pushcart_panel import cart_pose
        from Vender.robot_display import load_display_robot, feedback_configuration
        from unitree_nav2_web.cart_geometry import rotation_z

        robot = load_display_robot(self.config.urdf)
        geometry = self.executor.cart_geometry
        rng = np.random.default_rng(42)
        for _ in range(12):
            joints = self.config.prepare_q + rng.uniform(-.05, .05, 29)
            robot.update_cfg(feedback_configuration(robot, joints))
            roll, pitch = rng.uniform(-.2, .2, 2)
            body_yaw, relative_yaw = rng.uniform(-math.pi, math.pi, 2)
            quat = Rotation.from_euler('xyz', [roll, pitch, body_yaw]).as_quat()[[3, 0, 1, 2]]
            base = rng.uniform(-1., 1., 3)
            expected_position, expected_rotation, _ = cart_pose(
                geometry.display, robot, base, quat, dict(fresh=True, yaw_rad=relative_yaw))
            actual = geometry.estimate(dict(zip(JOINT_NAMES, joints)), quat, relative_yaw)
            np.testing.assert_allclose(base + rotation_z(body_yaw) @ actual['position'],
                                       expected_position, atol=1e-10)
            np.testing.assert_allclose(rotation_z(body_yaw + actual['yaw']),
                                       expected_rotation, atol=1e-10)

    def test_visual_geometry_uses_body_and_arms_at_image_time(self):
        from core.robot import JOINT_NAMES
        data = copy.deepcopy(self.control.snapshot())
        self.executor.cart_pose_source = 'apriltag'
        start = self.now+1.
        baseline = np.array(data['state']['q'])
        for index in range(40):
            now, captured = start+index*.02, start+index*.02-.1
            def feedback(stamp):
                joints = baseline.copy()
                joints[15] += .3*(stamp-start)
                roll = .2*(stamp-start)
                return joints, np.array([math.cos(roll/2), math.sin(roll/2), 0., 0.])
            joints, quaternion = feedback(now)
            data['state'].update(received=now, q=joints, quat=quaternion)
            observation = dict(valid=True, yaw_rad=0., T_level_cart=np.eye(4).tolist(),
                               tag_ids=[1], captured_monotonic=captured)
            data['high_policy']['camera'].update(pose_source='apriltag', fresh=True,
                captured_monotonic=captured, cart_pose_urdf=observation)
            result = self.executor._cart_status(data, now)
            if index >= 6:
                old_joints, old_quaternion = feedback(captured)
                expected = self.executor.cart_geometry.estimate_visual(
                    dict(zip(JOINT_NAMES, old_joints)), old_quaternion, observation)
                self.assertTrue(result['valid'], result)
                self.assertEqual(result['sample_monotonic'], captured)
                np.testing.assert_allclose(result['palms'], expected['palms'], atol=1e-9)
                np.testing.assert_allclose(result['body_rotation'], expected['body_rotation'], atol=1e-9)
        # A valid tag measurement cannot make missing historical body state valid.
        self.executor.pelvis_yaw_history.samples.clear()
        self.assertFalse(self.executor._cart_status(data, now)['valid'])

    def send(self, command, epoch=None):
        self.sequence += 1
        self.sender.sendto(VELOCITY_PACKET.pack(
            VELOCITY_MAGIC, VELOCITY_VERSION, 1, 0, self.sequence, *command,
            self.executor.navigation_epoch if epoch is None else epoch, int(self.now*1e9)),
            self.executor.sock.getsockname())
        return self.tick()

    def operator(self, action, **payload):
        self.sequence += 1
        request = dict(magic='G1NAVCTL1', sequence=self.sequence, action=action,
                       payload=payload, sent_monotonic=self.now)
        self.sender.sendto(json.dumps(request).encode(), self.executor.sock.getsockname())
        for _ in range(5):
            self.tick()
        self.assertEqual(self.executor.cart_request['sequence'], self.sequence)
        return self.executor.cart_request

    def test_operator_buttons_and_ack_use_real_controller_and_temporary_grasp_file(self):
        original = self.config.hand_grasp.copy()
        reply = self.operator('hand_edit', joint=0, angle_rad=0.)
        self.assertEqual(reply['state'], 'rejected')
        np.testing.assert_array_equal(original, self.config.hand_grasp)
        reply = self.operator('pushcart_disable')
        self.assertEqual(reply['state'], 'handled')
        for _ in range(70):
            self.tick()
        self.assertFalse(self.high.enabled)
        # Never save to the operator's live YAML. Exercise the same file code on a copy.
        with tempfile.TemporaryDirectory(prefix='nav-grasp-buttons-') as directory:
            source = self.config.upper_pose_path
            target = Path(directory) / 'pose.yaml'
            target.write_bytes(source.read_bytes())
            self.config.upper_pose_path = target
            try:
                from core.hand_grasp import HAND_LOWER, HAND_UPPER, load_grasp
                angles = (HAND_LOWER + HAND_UPPER) / 2
                for index, angle in enumerate(angles.flat):
                    reply = self.operator('hand_edit', joint=index, angle_rad=float(angle))
                    self.assertEqual(reply['state'], 'handled')
                    self.assertAlmostEqual(self.config.hand_grasp.flat[index], angle)
                reply = self.operator('hand_save')
                if self.control.hand_file:
                    self.control.hand_file[0].result(timeout=5.)
                self.tick()
                self.assertIn('已保存', self.control.snapshot()['dex3']['editor_message'])
                np.testing.assert_allclose(load_grasp(target), angles)
                self.operator('hand_edit', joint=0, angle_rad=.1)
                self.operator('hand_load')
                if self.control.hand_file:
                    self.control.hand_file[0].result(timeout=5.)
                self.tick()
                np.testing.assert_allclose(self.config.hand_grasp, angles)
                self.operator('hand_toggle')
                self.assertEqual(self.control.hand_pose, 1)
                reply = self.operator('cart_detach')
                self.assertEqual(reply['state'], 'rejected')  # Hand motion not acknowledged yet.
                self.bridge.state.hand_flags = 39  # Simulated motor reports zero pose, healthy and active.
                self.tick()
                old_epoch = self.executor.navigation_epoch
                reply = self.operator('cart_detach')
                self.assertEqual(reply['state'], 'handled')
                self.assertFalse(self.executor.cart_attached)
                self.assertNotEqual(self.executor.navigation_epoch, old_epoch)
            finally:
                self.config.upper_pose_path = source

    def test_three_axis_task_limits_and_camera_loss_smooth_stop(self):
        for _ in range(80):
            data = self.send([.9, -.4, .6])
        high = data['high_policy']
        np.testing.assert_allclose(high['raw_task_command'], [.6, -.3, .4], atol=1e-6)
        # Fixed fake cart heading cannot follow the requested turn. The new
        # task heading loop saturates its correction before the student.
        self.assertAlmostEqual(high['heading_hold']['corr'], .15)
        np.testing.assert_allclose(high['task_command'], [.6, -.3, .55], atol=1e-6)
        np.testing.assert_allclose(data['velocity'], high['homie_command'], atol=1e-6)
        self.assertEqual(data['heading']['correction_rad_s'], 0.)
        self.assertEqual(data['selected_model'], 'homie')
        self.assertEqual(data['heading']['command_source'], 'PUSHCART')
        self.assertTrue(self.executor._cart_status(data, self.now)['valid'])
        self.refresh = False
        for _ in range(100):
            data = self.send([.3, .1, 0])
        np.testing.assert_array_equal(data['velocity'], 0)
        self.assertEqual(data['selected_model'], 'balance')
        self.assertFalse(self.executor._cart_status(data, self.now)['valid'])
        self.assertFalse(self.executor.motion_active)

    def test_visual_navigation_holds_pose_until_timeout_and_requires_reenable(self):
        self.operator('pushcart_disable')
        for _ in range(70):
            self.tick()
        self.assertEqual(self.operator('cart_pose_source', source='apriltag')['state'], 'handled')
        self.refresh = False

        def visual_tick(valid=True, moving=False):
            packet = fixtures.heading_packet(fixtures.camera_result(self.now, .9, valid))
            pose = np.eye(4)
            yaw = .12
            pose[:2, :2] = [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]]
            pose[:3, 3] = [1.1, .07, -.8]
            packet['cart_pose_urdf'] = dict(schema='cart_pose_urdf_v1', valid=valid, extrinsics='urdf',
                reference_frame='pelvis_level', captured_monotonic=packet['captured_monotonic'],
                tag_ids=[0, 1] if valid else [], T_level_cart=pose.tolist(), yaw_rad=yaw,
                reason='ok' if valid else 'no_cart_tag')
            self.camera.accept(json.dumps(packet), self.now)
            return self.send([.3, 0, 0]) if moving else self.tick()

        for _ in range(45):
            visual_tick()
        self.executor.request('pushcart_enable')
        for _ in range(65):
            visual_tick()
        self.assertTrue(self.high.enabled, self.control.reason)
        for _ in range(40):
            data = visual_tick(moving=True)
        self.assertTrue(np.any(data['velocity']))
        initial = self.camera.snapshot(self.now)
        for _ in range(5):
            data = visual_tick(valid=False, moving=True)
            self.assertTrue(self.high.enabled)
            self.assertFalse(self.high.motion_blocked)
            self.assertTrue(self.executor.motion_active)
            cart = self.executor._cart_status(data, self.now)
            self.assertTrue(cart['valid'], cart)
            self.assertEqual(cart['captured_monotonic'], initial['captured_monotonic'])
            np.testing.assert_allclose(cart['position'], [1.1, .07, -.8])
            self.assertAlmostEqual(cart['yaw'], .12)
        visual_tick(moving=True)
        self.assertGreater(self.camera.stamp, initial['captured_monotonic'])
        for _ in range(100):
            data = visual_tick(valid=False, moving=True)
        self.assertFalse(self.high.enabled)
        self.assertTrue(self.high.motion_blocked)
        self.assertFalse(self.executor._cart_status(data, self.now)['valid'])
        self.assertFalse(self.executor.motion_active)
        self.assertEqual(data['selected_model'], 'balance')
        np.testing.assert_array_equal(data['velocity'], 0)
        for _ in range(10):
            data = visual_tick(moving=True)
        self.assertTrue(self.camera.fresh(self.now))
        self.assertFalse(self.high.enabled)
        self.assertTrue(self.high.motion_blocked)
        np.testing.assert_array_equal(data['velocity'], 0)

    def test_old_mode_commands_cannot_be_reinterpreted_and_detach_requires_release(self):
        for _ in range(35):
            data = self.send([.4, .1, .1], epoch=self.walk_epoch)
        np.testing.assert_array_equal(data['velocity'], 0)

        self.assertFalse(self.executor.enabled)
        self.executor.request('cart_detach')
        self.assertTrue(self.executor.cart_attached)
        self.assertTrue(self.executor.cart_error)
        for _ in range(50):
            data = self.send([.3, .1, 0])
        self.assertGreater(abs(data['high_policy']['task_command'][1]), .09)
        self.control.emit('zero', 1)
        for _ in range(90):
            self.tick()
        self.executor.request('pushcart_disable')
        for _ in range(65):
            data = self.tick()
        self.assertFalse(self.high.enabled)
        self.assertTrue(self.executor.cart_attached)
        for _ in range(20):
            data = self.send([.4, .1, 0])
        np.testing.assert_array_equal(data['velocity'], 0)


    def test_visual_source_switch_is_confirmed_persisted_and_invalidates_old_epoch(self):
        old_epoch = self.executor.navigation_epoch
        self.assertEqual(self.operator('cart_pose_source', source='apriltag')['state'], 'rejected')
        self.assertEqual(self.executor.navigation_epoch, old_epoch)
        self.operator('pushcart_disable')
        for _ in range(70):
            self.tick()
        with tempfile.TemporaryDirectory(prefix='nav-pose-source-') as directory:
            path = Path(directory)/'settings.json'
            path.write_text(self.executor.cart_config_path.read_text())
            self.executor.cart_config_path = path
            reply = self.operator('cart_pose_source', source='apriltag')
            self.assertEqual(reply['state'], 'handled', reply)
            self.assertEqual(self.executor.cart_pose_source, 'apriltag')
            self.assertEqual(json.loads(path.read_text())['pose_source'], 'apriltag')
            self.assertEqual(self.camera.pose_source, 'apriltag')
            self.assertNotEqual(self.executor.navigation_epoch, old_epoch)
            self.assertFalse(self.executor._cart_status(self.control.snapshot(), self.now)['valid'])
            self.assertFalse(self.high.history_ready(self.now))
            self.refresh = False
            self.tick()
            for _ in range(45):
                packet = fixtures.heading_packet(fixtures.camera_result(self.now, .9))
                pose = np.eye(4)
                yaw = .12
                pose[:2, :2] = [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]]
                pose[:3, 3] = [1.1, .07, -.8]
                packet['cart_pose_urdf'] = dict(schema='cart_pose_urdf_v1', valid=True, extrinsics='urdf',
                    reference_frame='pelvis_level', captured_monotonic=packet['captured_monotonic'],
                    tag_ids=[0, 1], T_level_cart=pose.tolist(), yaw_rad=yaw, rmse_px=.3)
                self.camera.accept(json.dumps(packet), self.now)
                self.tick()
            cart = self.executor._cart_status(self.control.snapshot(), self.now)
            self.assertTrue(cart['valid'], cart)
            np.testing.assert_allclose(cart['position'], [1.1, .07, -.8])
            self.assertAlmostEqual(cart['yaw'], yaw)
            self.assertAlmostEqual(self.high.camera.yaw, yaw)
            self.assertIn('无需', self.high.zero_reason(self.control, self.bridge.state, self.now))
            self.assertTrue(self.high.history_ready(self.now))
            self.send([.3, 0, 0], epoch=old_epoch)
            self.assertFalse(self.executor.enabled)
            reply = self.operator('cart_pose_source', source='grasp')
            self.assertEqual(reply['state'], 'handled', reply)
            self.assertEqual(json.loads(path.read_text())['pose_source'], 'grasp')
            self.assertEqual(self.camera.pose_source, 'grasp')


if __name__ == '__main__':
    unittest.main(verbosity=2)
