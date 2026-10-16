"""Preparation selection, URDF geometry and actual UI-to-C++ offline integration."""
import socket
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import yaml
import yourdfpy

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import robot
from core.controller import Controller
from core.robot import ACTIVE, JOINT_NAMES, UPPER, UPPER_POSE_LABELS, RobotConfig
from hybrid_deploy import protocol as p
from Vender.app import Viewer
from test_pose_integration import HARNESS, PipeBridge
from test_web import Browser, until


class PreparationConfigTests(unittest.TestCase):
    def test_both_poses_use_deployment_legs_and_their_configured_upper_angles(self):
        original_read = robot.read_bundle
        def edited_leg_config(path):
            deployment, contracts, token = original_read(path)
            deployment['default_q_sdk29'][0] = -.15
            return deployment, contracts, token
        with patch.object(robot, 'read_bundle', side_effect=edited_leg_config):
            config = RobotConfig()
        self.assertEqual(config.default_q[0], -.15)
        self.assertEqual(set(config.prepare_poses), {'original', 'arms_down'})
        for name, q in config.prepare_poses.items():
            with self.subTest(pose=name):
                np.testing.assert_array_equal(q[:12], config.default_q[:12])
                np.testing.assert_array_equal(q[13:15], 0)
                # Named YAML targets are authoritative. The optional reference
                # list and degree comments may predate manual pose tuning.
                pose = yaml.safe_load(config.upper_pose_paths[name].read_text())
                expected = [pose['hold_target_q'][JOINT_NAMES[i]] for i in UPPER]
                np.testing.assert_array_equal(q[UPPER], expected)
        np.testing.assert_array_equal(config.prepare_q, config.prepare_poses['original'])
        self.assertFalse(np.array_equal(config.prepare_poses['original'][UPPER],
                                        config.prepare_poses['arms_down'][UPPER]))

    def test_urdf_distinguishes_grasp_pose_from_arms_down_pose(self):
        config = RobotConfig()
        model = yourdfpy.URDF.load(str(config.urdf), load_meshes=False, load_collision_meshes=False)
        for name, q in config.prepare_poses.items():
            model.update_cfg({JOINT_NAMES[i]: q[i] for i in ACTIVE})
            for side, sign in (('left', 1), ('right', -1)):
                with self.subTest(pose=name, side=side):
                    shoulder, elbow, wrist = [
                        model.get_transform(f'{side}_{joint}_link', 'torso_link')[:3, 3]
                        for joint in ('shoulder_roll', 'elbow', 'wrist_yaw')]
                    if name == 'original':
                        # Grasp pose: hands reach forwards and inwards from the
                        # elbows, below the shoulders but above the elbows.
                        self.assertGreater(elbow[0], shoulder[0])
                        self.assertGreater(wrist[0], elbow[0])
                        self.assertGreater(sign * wrist[1], 0)
                        self.assertLess(sign * wrist[1], sign * elbow[1])
                        self.assertLess(wrist[2], shoulder[2])
                        self.assertGreater(wrist[2], elbow[2])
                    else:
                        # Current tuned pose uses shoulder roll +/-0.35 rad
                        # and elbow 1.22 rad. Both segments point predominantly
                        # down and outwards; no obsolete 15-degree constraint.
                        for start, end in ((shoulder, elbow), (elbow, wrist)):
                            direction = end - start
                            self.assertGreater(-direction[2], np.linalg.norm(direction[:2]))
                            self.assertGreater(sign * direction[1], 0)

    def test_invalid_pose_is_rejected_before_starting_control(self):
        original = yaml.safe_load(robot.UPPER_POSE_PATHS['arms_down'].read_text())
        with tempfile.TemporaryDirectory(prefix='viser_pose_config_') as directory:
            path = Path(directory) / 'pose.yaml'
            for value in (float('nan'), float('inf'), 90., True, '90'):
                with self.subTest(value=value):
                    original['hold_target_q']['left_elbow_joint'] = value
                    path.write_text(yaml.safe_dump(original))
                    with patch.dict(robot.UPPER_POSE_PATHS, arms_down=path):
                        with self.assertRaisesRegex(ValueError, 'pose.yaml'):
                            RobotConfig()


class PreparationWebTests(unittest.TestCase):
    def test_web_choice_reaches_cpp_and_locks_during_motion(self):
        # No production bridge, DDS or shared robot UDP ports: only pipes plus
        # an ephemeral loopback port for actual Viser WebSocket clients.
        config = RobotConfig()
        with tempfile.TemporaryDirectory(prefix='viser_pose_web_') as directory:
            source, binary = Path(directory) / 'pose.cpp', Path(directory) / 'pose'
            source.write_text(HARNESS)
            subprocess.run([sys.executable, '-B', str(config.deploy / 'tools/generate_config.py'),
                            str(config.deploy / 'config/deploy.json'),
                            str(Path(directory) / 'generated_config.hpp')], check=True, capture_output=True)
            subprocess.run(['g++', '-std=c++17', '-O2', '-I' + str(config.deploy / 'cpp'),
                            '-I' + directory, str(source), '-o', str(binary)], check=True, capture_output=True)
            for pose_name in ('original', 'arms_down'):
                with self.subTest(pose=pose_name):
                    self.check_web_pose(config, binary, pose_name)

    def check_web_pose(self, config, binary, pose_name):
        other_pose = 'arms_down' if pose_name == 'original' else 'original'
        bridge = PipeBridge(binary, config)
        controller = Controller(config, bridge, 'sim', clock=bridge.now)
        viewer, clients, thread = None, [], None
        stopped, errors = threading.Event(), []
        ui_clock = patch('Vender.app.time', SimpleNamespace(monotonic=bridge.now))
        ui_clock.start()
        try:
            controller.step()
            with socket.socket() as probe:
                probe.bind(('127.0.0.1', 0))
                port = probe.getsockname()[1]
            viewer = Viewer(controller, '127.0.0.1', port)
            def run():
                try:
                    while not stopped.wait(.02):
                        controller.step()
                        viewer.update()
                except Exception as exc:
                    errors.append(exc)
            thread = threading.Thread(target=run, daemon=True)
            thread.start()
            first = Browser(port); clients.append(first)
            second = Browser(port); clients.append(second)
            label = '上半身准备姿态'
            selectors = [until(lambda client=client: client.find(label)) for client in clients]
            self.assertTrue(first.components[selectors[0]]['disabled'])
            self.assertEqual(first.components[selectors[0]]['value'], UPPER_POSE_LABELS['original'])
            first.click('接管网页操作')
            until(lambda: not first.components[selectors[0]]['disabled'])
            self.assertTrue(second.components[selectors[1]]['disabled'])
            first.set_slider(label, UPPER_POSE_LABELS[other_pose])
            until(lambda: controller.snapshot().get('prepare_pose') == other_pose)
            first.set_slider(label, UPPER_POSE_LABELS[pose_name])
            until(lambda: controller.snapshot().get('prepare_pose') == pose_name)
            until(lambda: second.components[selectors[1]]['value'] == UPPER_POSE_LABELS[pose_name])
            self.assertEqual(controller.request, p.WAIT)
            self.assertEqual(tuple(bridge.state.sent_ns), (0, 0, 0))
            first.click('② 移到准备姿态')
            until(lambda: controller.snapshot()['stage'] == 'PREPARING')
            until(lambda: first.components[selectors[0]]['disabled'])
            # A forged update of a disabled widget must not change the target.
            first.set_slider(label, UPPER_POSE_LABELS[other_pose])
            until(lambda: first.components[selectors[0]]['value'] == UPPER_POSE_LABELS[pose_name])
            until(lambda: controller.snapshot()['stage'] == 'HOLDING')
            data = controller.snapshot()
            self.assertEqual(data['prepare_pose'], pose_name)
            self.assertFalse(data['prepare_pose_selectable'])
            np.testing.assert_allclose(data['target'][ACTIVE], config.prepare_poses[pose_name][ACTIVE], atol=1e-4)
            np.testing.assert_allclose(data['target'][:12], config.default_q[:12], atol=1e-4)
            self.assertEqual(tuple(bridge.state.sent_ns[1:]), (0, 0))
            first.click('③ 启动混合策略')
            until(lambda: controller.snapshot()['stage'] == 'POLICY')
            np.testing.assert_allclose(controller.snapshot()['target'][UPPER],
                                       config.prepare_poses[pose_name][UPPER], atol=1e-4)
            self.assertTrue(first.components[selectors[0]]['disabled'])
            self.assertFalse(errors, errors)
            self.assertTrue(all(client.error is None for client in clients))
        finally:
            stopped.set()
            if thread is not None:
                thread.join(timeout=5)
            for client in clients:
                client.close()
            if viewer is not None:
                viewer.server.stop()
            controller.close()
            ui_clock.stop()


if __name__ == '__main__':
    unittest.main()
