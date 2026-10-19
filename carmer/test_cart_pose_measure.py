"""Offline geometry/image/recording tests; never open a camera or DDS node."""
import csv
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest import mock

import cv2
import numpy as np

from carmer.apriltag_detector import detect_tags, load_intrinsics, make_detector
from carmer.cart_pose_measure import (BOARD_CORNERS, CSV_FIELDS, INTRINSICS_FILE, TAG_CENTERS,
                                      TAG_CORNERS, PROVIDED_EXTRINSICS_FILE, CartPoseEstimator, compare, csv_row,
                                      frame_poses, main as measure_main, measurement_poses, pose_values, quaternion_rotation,
                                      rotation_distance, summarize, transform)
from carmer.camera_extrinsics import ZeroWaistCameraExtrinsics
from carmer.pelvis_attitude import G1Kinematics, OPTICAL_TO_LINK


def camera_view(center=(-1.0, .02, 1.25), target=(-.10, 0., .775)):
    center, target = np.array(center), np.array(target)
    z = target-center
    z /= np.linalg.norm(z)
    x = np.cross(z, [0., 0., 1.])
    x /= np.linalg.norm(x)
    rotation = np.stack([x, np.cross(z, x), z])
    return transform(rotation, -rotation @ center)


class CartPoseMeasureTest(unittest.TestCase):
    def setUp(self):
        cv2.setNumThreads(1)
        self.matrix, self.distortion = load_intrinsics(INTRINSICS_FILE)
        self.estimator = CartPoseEstimator(self.matrix, self.distortion)

    def tags(self, pose, ids=(0, 1)):
        tags = []
        for tag_id in ids:
            corners = cv2.projectPoints(TAG_CORNERS+TAG_CENTERS[tag_id],
                                        cv2.Rodrigues(pose[:3, :3])[0], pose[:3, 3],
                                        self.matrix, self.distortion)[0].reshape(4, 2)
            tags.append(dict(id=tag_id, corners_px=corners.tolist()))
        return tags

    def test_joint_pose_recovers_ground_origin_across_views(self):
        for center in ((-1., .02, 1.25), (-.7, -.3, 1.15), (-1.5, .35, 1.5)):
            with self.subTest(center=center):
                truth = camera_view(center)
                result = self.estimator.estimate(self.tags(truth))
                self.assertTrue(result['valid'], result)
                np.testing.assert_allclose(result['T'], truth, atol=1e-7)
                self.assertAlmostEqual(result['baseline_m'], .29, places=7)
                self.assertLess(result['single_position_difference_m'], 1e-7)

    def test_single_tag_and_selection(self):
        truth = camera_view()
        for tag_id in (0, 1):
            with self.subTest(tag_id=tag_id):
                estimator = CartPoseEstimator(self.matrix, self.distortion, [tag_id])
                result = estimator.estimate(self.tags(truth))
                self.assertTrue(result['valid'], result)
                self.assertEqual(result['tag_ids'], [tag_id])
                np.testing.assert_allclose(result['T'], truth, atol=1e-7)

    def test_real_detector_corner_order_on_rendered_image(self):
        truth = camera_view(center=(-.7, .02, 1.25))
        canvas = np.full((480, 640), 255, np.uint8)
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        source = np.array([[0., 0.], [199., 0.], [199., 199.], [0., 199.]], np.float32)
        for tag in self.tags(truth):
            marker = cv2.aruco.generateImageMarker(dictionary, tag['id'], 200)
            warp = cv2.getPerspectiveTransform(source, np.array(tag['corners_px'], np.float32))
            canvas = np.minimum(canvas, cv2.warpPerspective(marker, warp, (640, 480), borderValue=255))
        tags = detect_tags(make_detector(1.), canvas)
        self.assertEqual([tag['id'] for tag in tags], [0, 1])
        result = self.estimator.estimate(tags)
        self.assertTrue(result['valid'], result)
        self.assertLess(np.linalg.norm(result['T'][:3, 3]-truth[:3, 3]), .015)
        self.assertLess(rotation_distance(result['T'][:3, :3], truth[:3, :3]), 1.)
        # Board edges are independently measured as 90 x 50 cm, not the 95 cm handle extent.
        self.assertAlmostEqual(np.linalg.norm(BOARD_CORNERS[1]-BOARD_CORNERS[0]), .9)
        self.assertAlmostEqual(TAG_CENTERS[0][0], -.103)

    def test_noise_stays_bounded_without_temporal_smoothing(self):
        truth = camera_view()
        rng = np.random.default_rng(42)
        errors = []
        for _ in range(30):
            tags = self.tags(truth)
            for tag in tags:
                tag['corners_px'] = np.asarray(tag['corners_px'])+rng.normal(0., .15, (4, 2))
            result = self.estimator.estimate(tags)
            self.assertTrue(result['valid'], result)
            errors.append(np.linalg.norm(result['T'][:3, 3]-truth[:3, 3]))
        self.assertLess(max(errors), .025)

    def test_missing_duplicate_and_inconsistent_tags_are_invalid(self):
        self.assertFalse(self.estimator.estimate([])['valid'])
        tags = self.tags(camera_view())
        self.assertEqual(self.estimator.estimate([tags[0], tags[0]])['reason'], 'duplicate_tag_id')
        tags[1]['corners_px'] = np.asarray(tags[1]['corners_px'])+np.array([35., 0.])
        result = self.estimator.estimate(tags)
        self.assertFalse(result['valid'])
        self.assertEqual(result['reason'], 'reprojection_error_too_large')
        tags[0]['corners_px'] = [[float('nan'), 0.]]*4
        self.assertFalse(self.estimator.estimate(tags)['valid'])

    def test_map_composition_preserves_body_tilt_and_lever_arm(self):
        map_robot = transform(cv2.Rodrigues(np.array([.18, -.12, .7]))[0], [3., -2., .85])
        robot_camera = transform(cv2.Rodrigues(np.array([.3, .6, -.2]))[0], [.05, .02, .43])
        truth = transform(cv2.Rodrigues(np.array([0., 0., .4]))[0], [3.8, -1.6, 0.])
        camera_cart = np.linalg.inv(robot_camera) @ np.linalg.inv(map_robot) @ truth
        compensation = dict(valid=True, R_camera_to_pelvis=robot_camera[:3, :3],
                            camera_position_in_pelvis_m=robot_camera[:3, 3],
                            up_in_pelvis=map_robot[:3, :3].T @ [0., 0., 1.])
        poses = frame_poses(dict(valid=True, T=camera_cart), compensation,
                            dict(valid=True, T=map_robot))
        np.testing.assert_allclose(poses['map'], truth, atol=1e-12)
        np.testing.assert_allclose(poses['level'][:3, 2], [0., 0., 1.], atol=1e-12)
        self.assertAlmostEqual(poses['level'][2, 3], -.85)
        self.assertNotIn('map', frame_poses(dict(valid=True, T=camera_cart), compensation,
                                          dict(valid=False, reason='missing TF')))
        self.assertEqual(frame_poses(dict(valid=False)), {})
        self.assertEqual(set(frame_poses(dict(valid=True, T=camera_cart))), {'camera'})

    def test_tf_quaternion_is_normalized_and_invalid_data_rejected(self):
        np.testing.assert_allclose(quaternion_rotation([0., 0., 0., 1.1]), np.eye(3))
        with self.assertRaises(ValueError):
            quaternion_rotation([0., 0., 0., 0.])
        with self.assertRaises(ValueError):
            quaternion_rotation([float('nan'), 0., 0., 1.])

    def comparison_case(self, yaw=0.):
        model = G1Kinematics()
        supplied = ZeroWaistCameraExtrinsics(model, PROVIDED_EXTRINSICS_FILE)
        q = np.zeros(15)
        q[12] = yaw
        rotation, position = model.transform('d435_link', q)
        nominal = transform(rotation @ OPTICAL_TO_LINK, position)
        # Independent expected motion: this URDF's waist axis goes through the
        # pelvis origin, so it rotates both the supplied orientation and lever arm.
        expected_live = transform(cv2.Rodrigues(np.array([0., 0., yaw]))[0], np.zeros(3)) @ supplied.provided_zero
        np.testing.assert_allclose(supplied.from_nominal(nominal), expected_live, atol=1e-11)
        map_robot = transform(cv2.Rodrigues(np.array([.18, -.12, .7]))[0], [3., -2., .8])
        compensation = dict(valid=True, R_camera_to_pelvis=nominal[:3, :3].tolist(),
                            camera_position_in_pelvis_m=nominal[:3, 3].tolist(),
                            up_in_pelvis=(map_robot[:3, :3].T @ [0., 0., 1.]).tolist(),
                            waist_yaw_deg=np.rad2deg(yaw), roll_deg=8., pitch_deg=-5.)
        truth = transform(cv2.Rodrigues(np.array([0., 0., .2]))[0], [1., .05, -.8])
        estimate = self.estimator.estimate(self.tags(np.linalg.inv(expected_live) @ truth))
        self.assertTrue(estimate['valid'], estimate)
        poses = measurement_poses(estimate, compensation, dict(valid=True, T=map_robot),
                                  extrinsics_mode='compare', provided_extrinsics=supplied)
        return supplied, compensation, estimate, poses, truth, map_robot

    def test_provided_extrinsics_recover_cart_and_follow_waist_with_body_tilt(self):
        for yaw in (0., .37, -.21):
            with self.subTest(yaw=yaw):
                supplied, compensation, estimate, poses, truth, map_robot = self.comparison_case(yaw)
                np.testing.assert_allclose(poses['robot_provided'], truth, atol=1e-7)
                np.testing.assert_allclose(poses['map_provided'], map_robot @ truth, atol=1e-7)
                np.testing.assert_allclose(poses['robot_urdf'], poses['robot'], atol=1e-12)
                np.testing.assert_array_equal(poses['camera'], estimate['T'])
                self.assertGreater(np.linalg.norm(poses['robot_provided'][:3, 3]-poses['robot_urdf'][:3, 3]), .005)
                if yaw == 0.:
                    np.testing.assert_allclose(supplied.provided_zero[:3, 3], [.04764571478, .01753, .46268178553], atol=1e-12)
                    self.assertAlmostEqual(rotation_distance(supplied.urdf_zero[:3, :3], supplied.provided_zero[:3, :3]), .4, places=4)

    def test_extrinsics_selection_and_unavailable_feedback_do_not_fallback(self):
        supplied, compensation, estimate, poses, truth, _ = self.comparison_case()
        original = json.dumps(compensation)
        selected = measurement_poses(estimate, compensation, extrinsics_mode='provided', provided_extrinsics=supplied)
        np.testing.assert_allclose(selected['robot'], truth, atol=1e-7)
        np.testing.assert_allclose(selected['level'], poses['level_provided'], atol=1e-7)
        nominal = measurement_poses(estimate, compensation)
        np.testing.assert_allclose(nominal['robot'], poses['robot_urdf'], atol=1e-12)
        self.assertEqual(json.dumps(compensation), original)
        for mode in ('provided', 'compare'):
            self.assertEqual(set(measurement_poses(estimate, dict(valid=False), extrinsics_mode=mode,
                                                  provided_extrinsics=supplied)), {'camera'})
            self.assertEqual(measurement_poses(dict(valid=False), compensation, extrinsics_mode=mode,
                                               provided_extrinsics=supplied), {})

    def test_comparison_csv_records_both_sources_deltas_and_timing_matched_waist(self):
        _, compensation, estimate, poses, _, _ = self.comparison_case(.37)
        row = csv_row(dict(frame_number=1, pelvis_compensation=compensation), estimate,
                      poses, 123., .02, 'ok', 'ok', extrinsics_mode='compare')
        self.assertEqual(row['extrinsics_mode'], 'compare')
        self.assertAlmostEqual(row['waist_yaw_deg'], np.rad2deg(.37))
        self.assertAlmostEqual(row['provided_minus_urdf_level_x_m'],
                               poses['level_provided'][0, 3]-poses['level_urdf'][0, 3])
        self.assertAlmostEqual(row['provided_vs_urdf_level_rotation_deg'], .4, places=4)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'comparison.csv'
            with path.open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
                writer.writeheader()
                writer.writerow(row)
            for name in ('level_urdf', 'level_provided'):
                report = summarize(path, name)
                self.assertEqual(report['valid_frames'], 1)
                self.assertAlmostEqual(report['mean']['x_m'], poses[name][0, 3])

    def test_supplied_matrix_requires_physical_pelvis_frame_and_rigid_transform(self):
        original = json.loads(PROVIDED_EXTRINSICS_FILE.read_text())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'invalid.json'
            for key, value in [('target_frame', 'pelvis_level'), ('translation_units', 'cm'),
                               ('reference_waist_yaw_deg', 10), ('T_pelvis_camera_optical', np.zeros((4, 4)).tolist())]:
                with self.subTest(key=key):
                    path.write_text(json.dumps(dict(original, **{key: value})))
                    with self.assertRaises(ValueError):
                        ZeroWaistCameraExtrinsics(G1Kinematics(), path)

    def test_cli_writes_selected_poses_and_comparison_metadata_without_hardware(self):
        supplied, compensation, estimate, poses, truth, _ = self.comparison_case(.25)
        for mode in ('urdf', 'provided', 'compare'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                output = Path(directory)/'measurement.csv'
                worker = mock.Mock()
                packet = dict(frame_number=121, frame_timestamp_monotonic=time.monotonic(),
                              tags=self.tags(estimate['T']), pelvis_compensation=compensation)
                worker.latest.side_effect = [packet, None]
                with mock.patch('carmer.cart_pose_measure.RobotFeedback'), \
                        mock.patch('carmer.cart_pose_measure.open_camera'), \
                        mock.patch('carmer.cart_pose_measure.CameraReader'), \
                        mock.patch('carmer.cart_pose_measure.DetectionWorker', return_value=worker), \
                        mock.patch('sys.argv', ['cart_pose_measure.py', '--no-display', '--extrinsics', mode,
                                                '--output', str(output)]), redirect_stdout(io.StringIO()):
                    measure_main()
                with output.open(newline='') as stream:
                    rows = list(csv.DictReader(stream))
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]['extrinsics_mode'], mode)
                expected = truth if mode == 'provided' else poses['robot_urdf']
                self.assertAlmostEqual(float(rows[0]['robot_x_m']), expected[0, 3], places=7)
                metadata = json.loads(output.with_suffix('.meta.json').read_text())
                self.assertEqual(metadata['primary_pose_source'], 'provided' if mode == 'provided' else 'urdf')
                if mode != 'urdf':
                    np.testing.assert_allclose(metadata['extrinsics']['T_pelvis_camera_optical_provided_zero'],
                                               supplied.provided_zero, atol=1e-12)
                if mode == 'compare':
                    self.assertAlmostEqual(float(rows[0]['robot_provided_x_m']), truth[0, 3], places=7)
                    summary = json.loads(output.with_suffix('.summary.json').read_text())
                    self.assertEqual(summary['level_provided']['valid_frames'], 1)
                    self.assertEqual(summary['level_urdf']['valid_frames'], 1)
                worker.close.assert_called_once()

    def test_csv_invalid_frames_and_comparison_wrap_yaw(self):
        with tempfile.TemporaryDirectory() as temp:
            paths = [Path(temp)/'before.csv', Path(temp)/'after.csv']
            for path, x, yaw in zip(paths, (1., 1.2), (179., -179.)):
                pose = transform(cv2.Rodrigues(np.array([0., 0., np.deg2rad(yaw)]))[0], [x, 2., 0.])
                with path.open('w', newline='') as stream:
                    writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
                    writer.writeheader()
                    writer.writerow(dict(camera_reason='no_cart_tag'))
                    for _ in range(4):
                        writer.writerow(csv_row(dict(frame_number=1),
                                                dict(valid=True, reason='ok', tag_ids=[0, 1]),
                                                dict(map=pose), 123., .02, 'ok', 'ok'))
            report = summarize(paths[0], 'map')
            self.assertEqual(report['total_frames'], 5)
            self.assertEqual(report['valid_frames'], 4)
            result = compare(*paths, 'map')
            np.testing.assert_allclose(result['delta_xyz_m'], [.2, 0., 0.])
            self.assertAlmostEqual(result['delta_yaw_deg'], 2.)
            self.assertAlmostEqual(result['delta_xy_in_initial_cart_heading_m'][0], -.199969539, places=7)
            self.assertEqual(summarize(paths[0], 'robot')['valid_frames'], 0)


if __name__ == '__main__':
    unittest.main()
