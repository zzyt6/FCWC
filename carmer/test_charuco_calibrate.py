"""Offline ChArUco calibration, capture semantics, and extrinsic-frame tests."""
import copy
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest import mock

import cv2
import numpy as np

from carmer.apriltag_detector import load_intrinsics, rotation_to_rpy
from carmer.camera_extrinsics import (CalibratedCameraExtrinsics, load_camera_to_torso,
                                      rigid_transform, urdf_camera_to_torso)
from carmer.charuco_calibrate import (CONFIG, BoardDetector, calibrate_intrinsics,
                                     calibrate_reference, calibration_quality, capture,
                                     intrinsics_document, load_config, make_board,
                                     reference_transform, write_json)
from carmer.pelvis_attitude import G1Kinematics


class CharucoCalibrationTest(unittest.TestCase):
    def setUp(self):
        cv2.setNumThreads(1)
        self.config = load_config(CONFIG)
        self.board = make_board(self.config)
        self.matrix = np.array([[610., 0., 321.], [0., 608., 239.], [0., 0., 1.]])
        self.distortion = np.array([.02, -.03, .001, -.001, .005])
        self.objects = self.board.getChessboardCorners()

    def observation(self, rvec, position, noise=0., index=0):
        images = cv2.projectPoints(self.objects, np.asarray(rvec, float), np.asarray(position, float),
                                    self.matrix, self.distortion)[0].reshape(-1, 2)
        images += np.random.default_rng(index).normal(0., noise, images.shape)
        return dict(filename=f'synthetic_{index}.png', object_points=self.objects.tolist(),
                    image_points=images.tolist(), charuco_ids=list(range(len(images))), coverage_fraction=.1)

    def diverse_views(self, noise=.02):
        rng = np.random.default_rng(73)
        views = []
        for i in range(25):
            rvec = rng.uniform([-.45, -.45, -.25], [.45, .45, .25])
            rotation = cv2.Rodrigues(rvec)[0]
            center = np.array([rng.uniform(-.13, .13), rng.uniform(-.08, .08), rng.uniform(.43, .85)])
            c = self.config['board']
            board_center = np.array([c['squares_x'], c['squares_y'], 0.])*c['square_length_m']/2
            position = center - rotation @ board_center
            views.append(self.observation(rvec, position, noise, i))
        return views

    def test_config_is_resolved_relative_to_its_file_and_missing_reference_rejected(self):
        self.assertTrue(Path(self.config['output']['session_dir']).is_absolute())
        self.assertTrue(Path(self.config['robot_urdf']).exists())
        with self.assertRaisesRegex(ValueError, 'independently measured'):
            reference_transform(self.config)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'bad.json'
            config = json.loads(CONFIG.read_text())
            config['board']['marker_length_m'] = config['board']['square_length_m']
            write_json(path, config)
            with self.assertRaises(ValueError):
                load_config(path)

    def test_rendered_charuco_detection_and_object_corner_correspondences(self):
        width = self.config['board']['squares_x']*100
        height = self.config['board']['squares_y']*100
        board_image = self.board.generateImage((width, height))
        source = np.array([[0., 0.], [width-1., 0.], [width-1., height-1.], [0., height-1.]], np.float32)
        target = np.array([[90., 70.], [545., 100.], [510., 408.], [115., 380.]], np.float32)
        warp = cv2.getPerspectiveTransform(source, target)
        image = cv2.warpPerspective(board_image, warp, (640, 480), borderValue=255)
        detector = BoardDetector(self.board, self.config)
        detection = detector.detect(image)
        self.assertTrue(detection['valid'], detection['reason'])
        observation = detector.observation(detection, 'test.png')
        expected = self.objects[detection['ids'].reshape(-1)]
        np.testing.assert_allclose(observation['object_points'], expected)
        self.assertFalse(detector.detect(np.full((480, 640), 255, np.uint8))['valid'])

    def test_intrinsics_recovery_and_existing_loader_compatibility(self):
        matrix, distortion, report = calibrate_intrinsics(self.diverse_views(), self.config)
        self.assertEqual(report['status'], 'calibrated', report)
        self.assertLess(report['rms_px'], .05)
        np.testing.assert_allclose(matrix[:2, :], self.matrix[:2, :], atol=1.5)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'intrinsics.json'
            write_json(path, intrinsics_document(matrix, distortion, report, directory))
            loaded_matrix, loaded_distortion = load_intrinsics(path)
            np.testing.assert_allclose(loaded_matrix, matrix)
            np.testing.assert_allclose(loaded_distortion, distortion.reshape(-1))

    def test_outlier_view_is_excluded_and_flat_pose_diversity_is_flagged(self):
        views = self.diverse_views()
        images = np.asarray(views[-1]['image_points'])
        images += np.random.default_rng(2).normal(0., 8., images.shape)
        views[-1]['image_points'] = images.tolist()
        _, _, report = calibrate_intrinsics(views, self.config)
        self.assertEqual(report['status'], 'calibrated', report)
        self.assertIn(24, [view['index'] for view in report['rejected_views']])
        failures, _ = calibration_quality(self.matrix, self.distortion, .01,
                                          [np.zeros(3)]*12, self.config['calibration'])
        self.assertIn('insufficient_board_tilt_diversity', failures)

    def reference_case(self):
        nominal = urdf_camera_to_torso(self.config['robot_urdf'])
        actual = nominal.copy()
        actual[:3, :3] = cv2.Rodrigues(np.array([.01, -.04, .025]))[0] @ nominal[:3, :3]
        actual[:3, 3] += np.array([.015, -.006, .012])
        rvec, position = np.array([.18, -.28, .05]), np.array([-.09, -.06, .65])
        camera_board = rigid_transform(cv2.Rodrigues(rvec)[0], position)
        torso_board = actual @ camera_board
        config = copy.deepcopy(self.config)
        config['reference']['position_in_torso_m'] = torso_board[:3, 3].tolist()
        config['reference']['rpy_in_torso_deg'] = np.rad2deg(rotation_to_rpy(torso_board[:3, :3])[0]).tolist()
        views = [self.observation(rvec, position, 0., i) for i in range(config['reference']['sample_count'])]
        return actual, config, views

    def test_known_reference_recovers_optical_to_torso_transform(self):
        actual, config, views = self.reference_case()
        report = calibrate_reference(views, self.matrix, self.distortion, config)
        np.testing.assert_allclose(report['T_torso_camera_optical'], actual, atol=2e-6)
        np.testing.assert_allclose(np.array(report['T_camera_optical_torso']) @ actual, np.eye(4), atol=2e-6)
        self.assertGreater(report['rotation_difference_from_urdf_deg'], 1.)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'camera_to_torso.json'
            write_json(path, report)
            np.testing.assert_allclose(load_camera_to_torso(path), actual, atol=2e-6)

    def test_reference_motion_disagreement_is_rejected(self):
        _, config, views = self.reference_case()
        views[-1] = self.observation([.18, -.28, .05], [-.03, -.06, .65], 0., 99)
        with self.assertRaisesRegex(ValueError, 'disagree'):
            calibrate_reference(views, self.matrix, self.distortion, config)

    def test_pending_or_improper_extrinsic_matrix_cannot_be_loaded(self):
        _, config, views = self.reference_case()
        report = calibrate_reference(views, self.matrix, self.distortion, config)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'extrinsics.json'
            write_json(path, dict(status='reference_pose_required', T_torso_camera_optical=None))
            with self.assertRaises(ValueError):
                load_camera_to_torso(path)
            report['T_torso_camera_optical'][0][0] *= -1
            write_json(path, report)
            with self.assertRaises(ValueError):
                load_camera_to_torso(path)

    def test_live_waist_angle_composition_uses_calibrated_mount(self):
        actual, config, views = self.reference_case()
        report = calibrate_reference(views, self.matrix, self.distortion, config)
        model = G1Kinematics(config['robot_urdf'])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'extrinsics.json'
            write_json(path, report)
            reader = CalibratedCameraExtrinsics(model, path)
            for yaw in (-.4, 0., .3):
                q = np.zeros(15)
                q[12] = yaw
                pelvis_waist = rigid_transform(cv2.Rodrigues(np.array([0., 0., yaw]))[0], np.zeros(3))
                waist_torso = rigid_transform(np.eye(3), [-.0039635, 0., .044])
                np.testing.assert_allclose(reader.in_pelvis(q), pelvis_waist @ waist_torso @ actual, atol=2e-6)

    def test_space_saves_displayed_raw_frames_and_stops_at_target(self):
        config = copy.deepcopy(self.config)
        config['capture']['sample_count'] = 2
        detector = mock.Mock()
        detector.detect.return_value = dict(valid=True, reason='ready', ids=np.arange(12), corners=None,
                                             marker_ids=None, marker_corners=None, coverage=.2)
        detector.observation.side_effect = lambda detection, filename: dict(filename=str(filename))
        reader = mock.Mock()
        frames = [np.full((480, 640, 3), value, np.uint8) for value in (20, 30, 40)]
        calls = iter(enumerate(frames, start=121))
        reader.latest.side_effect = lambda *a, **k: (lambda pair: (pair[0], pair[1], 60., time.monotonic()))(next(calls))
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.dict('os.environ', {'DISPLAY': ':test'}), \
                mock.patch('carmer.charuco_calibrate.open_camera'), \
                mock.patch('carmer.charuco_calibrate.CameraReader', return_value=reader), \
                mock.patch('carmer.charuco_calibrate.panel', side_effect=lambda frame, *a: frame), \
                mock.patch('carmer.charuco_calibrate.cv2.namedWindow'), \
                mock.patch('carmer.charuco_calibrate.cv2.imshow'), \
                mock.patch('carmer.charuco_calibrate.cv2.destroyAllWindows'), \
                mock.patch('carmer.charuco_calibrate.cv2.waitKey', side_effect=[32, ord('x'), 32]):
            observations, complete = capture(detector, config, Path(directory), 'intrinsics')
            self.assertTrue(complete)
            self.assertEqual(len(observations), 2)
            np.testing.assert_array_equal(cv2.imread(observations[0]['filename']), frames[0])
            np.testing.assert_array_equal(cv2.imread(observations[1]['filename']), frames[2])
            reader.close.assert_called_once()


if __name__ == '__main__':
    unittest.main()
