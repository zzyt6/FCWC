"""Geometry, configured finger display and optional feature isolation; no DDS."""
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.robot import RobotConfig
from core.pushcart import load_feature
from Vender.robot_display import UrdfDisplayUpdater, load_display_robot, feedback_configuration, current_grasp_reference, HAND_NAMES
from Vender.pushcart_panel import CartDisplay, cart_pose, camera_fresh, DISPLAY_CONFIG


class SceneTests(unittest.TestCase):
    def test_ui_rechecks_age_after_control_snapshot_stops_updating(self):
        camera = dict(fresh=True, captured_monotonic=100., timeout_s=.3)
        self.assertTrue(camera_fresh(camera, now=100.1))
        self.assertFalse(camera_fresh(camera, now=100.31))

    @classmethod
    def setUpClass(cls):
        cls.config = RobotConfig()
        cls.original_digest = hashlib.sha256(cls.config.urdf.read_bytes()).hexdigest()
        cls.robot = load_display_robot(cls.config.urdf)
        cls.settings = json.loads(DISPLAY_CONFIG.read_text())

    def test_display_unlocks_fingers_without_changing_real_urdf_or_contract(self):
        self.assertEqual(self.robot.num_actuated_joints, 41)
        self.assertEqual(len(self.config.deployment['joint_names_sdk29']), 29)
        self.assertTrue(all(name in self.robot.actuated_joint_names for name in HAND_NAMES))
        self.assertEqual(hashlib.sha256(self.config.urdf.read_bytes()).hexdigest(), self.original_digest)

    def test_incremental_display_matches_full_urdf_frames_and_palm_positions(self):
        from viser.extras import ViserUrdf

        def model():
            # Exercise the real Viser update implementation without a web server.
            value = object.__new__(ViserUrdf)
            value._urdf = load_display_robot(self.config.urdf)
            value._joint_map_values = list(value._urdf.joint_map.values())
            value._joint_frames = [SimpleNamespace() for _ in value._joint_map_values]
            value._load_meshes, value._scale = True, 1.3
            value.update_cfg(dict.fromkeys(value._urdf.actuated_joint_names, 0.))
            return value

        expected, actual = model(), model()
        updater = UrdfDisplayUpdater(actual)
        rng = np.random.default_rng(12)
        values = feedback_configuration(actual._urdf, self.config.prepare_q,
                                        grasp_reference=self.config.hand_grasp)
        configurations = [values, {name: float(rng.uniform(-.5, .5)) for name in values},
                          {'waist_yaw_joint': .7}, {HAND_NAMES[0]: .35}, values]
        for configuration in configurations:
            expected.update_cfg(configuration)
            updater.update_cfg(configuration)
            for a, b in zip(expected._joint_frames, actual._joint_frames):
                np.testing.assert_allclose(a.position, b.position, atol=1e-12)
                ra = Rotation.from_quat(np.asarray(a.wxyz)[[1, 2, 3, 0]]).as_matrix()
                rb = Rotation.from_quat(np.asarray(b.wxyz)[[1, 2, 3, 0]]).as_matrix()
                np.testing.assert_allclose(ra, rb, atol=1e-12)
            for side in ('left', 'right'):
                link = side + '_hand_palm_link'
                np.testing.assert_allclose(actual._urdf.get_transform(link),
                                           expected._urdf.get_transform(link), atol=1e-12)
        with patch.object(actual._urdf, 'update_cfg', wraps=actual._urdf.update_cfg) as update:
            updater.update_cfg(values)
            update.assert_not_called()

    def test_grasp_reference_follows_live_settings_and_compensation(self):
        cfg = self.config
        angles = cfg.hand_grasp.copy()
        angles[0, 0] = .12
        hands = dict(active=True, pose='grasp', grasp=angles, compensation_enabled=False)
        reference = current_grasp_reference(cfg, hands)
        values = feedback_configuration(self.robot, cfg.prepare_q, hands, reference)
        self.assertAlmostEqual(values['left_hand_thumb_0_joint'], .12)
        angles[0, 0] = .2
        values = feedback_configuration(self.robot, cfg.prepare_q, hands, current_grasp_reference(cfg, hands))
        self.assertAlmostEqual(values['left_hand_thumb_0_joint'], .2)
        hands.update(pose='grasp_compensated', compensation_enabled=True)
        np.testing.assert_allclose(current_grasp_reference(cfg, hands), angles + cfg.hand_zero)
        hands.update(pose='open')
        self.assertIsNone(current_grasp_reference(cfg, hands))

    def test_reference_pose_is_anchored_at_palms_and_cart_stays_level(self):
        self.robot.update_cfg(feedback_configuration(self.robot, self.config.prepare_q))
        quat = Rotation.from_euler('xyz', [.1, -.08, .4]).as_quat()[[3, 0, 1, 2]]
        position, rotation, label = cart_pose(self.settings, self.robot, [0, 0, .76], quat,
                                               dict(fresh=True, yaw_rad=.2))
        self.assertTrue(np.isfinite(position).all())
        np.testing.assert_allclose(rotation[:, 2], [0, 0, 1], atol=1e-8)
        self.assertAlmostEqual(np.arctan2(rotation[1, 0], rotation[0, 0]), .6)
        self.assertIn('参考', label)

    def test_calibrated_tag_position_reconstructs_cart_origin(self):
        settings = dict(self.settings, tag_positions_in_cart_m={'2': [.1, .2, .75]})
        rotation = Rotation.from_euler('z', .3).as_matrix()
        expected = np.array([.8, -.1, 0.])
        base = np.array([0., 0., .76])
        tag = expected + rotation @ [.1, .2, .75] - base
        position, _, label = cart_pose(settings, self.robot, base, [1, 0, 0, 0],
                                       dict(fresh=True, yaw_rad=.3,
                                            tags_in_pelvis=[dict(id=2, position_m=tag)]))
        np.testing.assert_allclose(position, expected)
        self.assertEqual(label, '标签标定位置')

    def test_missing_high_policy_does_not_break_normal_configuration(self):
        cfg = RobotConfig()
        limits = cfg.velocity_bounds.copy()
        self.assertIsNone(load_feature(cfg, 'sim', 'eth0', '/missing/pushcart.json'))
        self.assertTrue(cfg.pushcart_error)
        np.testing.assert_array_equal(limits, cfg.velocity_bounds)

    def test_actual_cart_nodes_visibility_and_stale_pose(self):
        import viser
        server = viser.ViserServer(host='127.0.0.1', port=0)
        self.addCleanup(server.stop)
        display = CartDisplay(server, self.robot)
        self.assertFalse(display.error, display.error)
        data = dict(healthy=True, high_policy=dict(enabled=True, camera=dict(fresh=True, yaw_rad=.2)))
        display.update(data, [0, 0, .76], [1, 0, 0, 0])
        self.assertFalse(display.error, display.error)
        self.assertTrue(display.frame.visible)
        position = np.asarray(display.frame.position).copy()
        orientation = np.asarray(display.frame.wxyz).copy()
        data['high_policy']['camera']['fresh'] = False
        display.update(data, [1, 0, .76], [1, 0, 0, 0])
        np.testing.assert_array_equal(display.frame.position, position)
        np.testing.assert_array_equal(display.frame.wxyz, orientation)
        self.assertIn('过期', display.label.text)
        data['high_policy']['enabled'] = False
        display.update(data, [1, 0, .76], [1, 0, 0, 0])
        self.assertFalse(display.frame.visible)


if __name__ == '__main__':
    unittest.main()
