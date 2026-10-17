"""Nav UDP -> shared push policy -> real ONNX, using synthetic robot/camera only."""
import json
import socket
import sys
import unittest
import tempfile
from pathlib import Path

import numpy as np

from g1_locomotion_executor import Executor, VELOCITY_PACKET, VELOCITY_MAGIC, VELOCITY_VERSION

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'G1_high_policy_deploy/test'))
import test_viser_integration as fixtures


class NavPushcartTests(unittest.TestCase):
    event = fixtures.IntegrationTests.event

    def setUp(self):
        fixtures.IntegrationTests.setUp(self)
        self.status = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.status.bind(('127.0.0.1', 0))
        self.status.setblocking(False)
        self.sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.executor = Executor(self.control, 0, self.status.getsockname()[1], clock=lambda: self.now)
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
        np.testing.assert_allclose(data['high_policy']['task_command'], [.6, -.15, .3], atol=1e-6)
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


if __name__ == '__main__':
    unittest.main(verbosity=2)
