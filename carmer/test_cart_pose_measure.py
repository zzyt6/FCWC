"""Offline geometry/image/recording tests; never open a camera or DDS node."""
import csv
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from carmer.apriltag_detector import detect_tags, load_intrinsics, make_detector
from carmer.cart_pose_measure import (BOARD_CORNERS, CSV_FIELDS, INTRINSICS_FILE, TAG_CENTERS,
                                      TAG_CORNERS, CartPoseEstimator, compare, csv_row,
                                      frame_poses, pose_values, quaternion_rotation,
                                      rotation_distance, summarize, transform)


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
