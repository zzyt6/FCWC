"""Synthetic camera -> pose packet -> policy receiver, with no robot access."""
import json
import unittest
from unittest.mock import Mock

import cv2
import numpy as np

from carmer.cart_pose import (CartPoseEstimator, TAG_CENTERS, TAG_CORNERS, frame_poses,
                             level_from_robot, transform, urdf_cart_observation)
from carmer.heading_udp import heading_packet
from carmer.apriltag_detector import DetectionWorker, TagPoseEstimator, make_detector
from carmer.cart_heading import CartHeadingEstimator
from carmer.pelvis_attitude import G1Kinematics, OPTICAL_TO_LINK
from G1_high_policy_deploy.robot.pushcart_v81.carmer_io import CarmerYaw


class CartPoseStreamTest(unittest.TestCase):
    def setUp(self):
        self.camera = CarmerYaw(0, offset_rad=.7, tag_ids=[0, 1])
        self.addCleanup(self.camera.close)
        self.matrix = np.array([[610., 0., 320.], [0., 610., 240.], [0., 0., 1.]])
        self.estimator = CartPoseEstimator(self.matrix, np.zeros(5))

    def observation(self, stamp=10.):
        q = np.zeros(15)
        q[12] = .25
        r, t = G1Kinematics().transform('d435_link', q)
        up = cv2.Rodrigues(np.array([.1, -.08, 0.]))[0].T @ [0., 0., 1.]
        compensation = dict(valid=True, R_camera_to_pelvis=(r @ OPTICAL_TO_LINK).tolist(),
                            camera_position_in_pelvis_m=t.tolist(), up_in_pelvis=up.tolist())
        truth = transform(cv2.Rodrigues(np.array([0., 0., .12]))[0], [1., .05, -.8])
        camera_cart = np.linalg.inv(level_from_robot(up) @ transform(r @ OPTICAL_TO_LINK, t)) @ truth
        tags = []
        for tag_id in (0, 1):
            pixels = cv2.projectPoints(TAG_CORNERS+TAG_CENTERS[tag_id],
                                       cv2.Rodrigues(camera_cart[:3, :3])[0], camera_cart[:3, 3],
                                       self.matrix, np.zeros(5))[0].reshape(4, 2)
            tags.append(dict(id=tag_id, corners_px=pixels.tolist()))
        packet = dict(frame_timestamp_monotonic=stamp, frame_number=1, tags=tags,
                      pelvis_compensation=compensation, cart_heading_settings=dict(camera_latency_ms=40),
                      raw_fused_cart_heading=dict(valid=True, yaw_rad=.9, reference_frame='pelvis_level', tag_ids=[0, 1]))
        packet['cart_pose_urdf'] = urdf_cart_observation(packet, self.estimator)
        return packet, truth

    def test_visual_pose_matches_measurement_and_policy_uses_same_heading(self):
        packet, truth = self.observation()
        expected = frame_poses(self.estimator.estimate(packet['tags']), packet['pelvis_compensation'])['level']
        np.testing.assert_allclose(packet['cart_pose_urdf']['T_level_cart'], expected, atol=1e-12)
        np.testing.assert_allclose(expected, truth, atol=1e-7)
        self.camera.set_pose_source('apriltag', 9.)
        encoded = json.dumps(heading_packet(packet))
        self.assertLess(len(encoded.encode()), 4096)
        self.assertTrue(self.camera.accept(encoded, 10.))
        self.assertAlmostEqual(self.camera.yaw, .12, places=6)
        self.assertAlmostEqual(self.camera.stamp, 9.96)
        self.assertTrue(self.camera.snapshot(10.)['cart_pose_urdf']['valid'])
        self.assertEqual(self.camera.snapshot(10.)['yaw_offset_rad'], 0.)
        with self.assertRaises(ValueError):
            self.camera.set_offset(.2)

    def test_rendered_tags_through_detector_worker_and_policy_receiver(self):
        packet, truth = self.observation()
        frame = np.full((480, 640, 3), 255, dtype=np.uint8)
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        for tag in packet['tags']:
            marker = cv2.aruco.generateImageMarker(dictionary, tag['id'], 240)
            warp = cv2.getPerspectiveTransform(
                np.array([[0, 0], [239, 0], [239, 239], [0, 239]], np.float32),
                np.array(tag['corners_px'], np.float32))
            image = cv2.warpPerspective(marker, warp, (640, 480), borderValue=255)
            frame = np.minimum(frame, image[:, :, None])
        reader, compensator = Mock(), Mock()
        reader.latest.side_effect = [(1, frame, 30., 10.), None]
        compensator.at_frame.return_value = packet['pelvis_compensation']
        heading = CartHeadingEstimator(cart_tag_ids=[0, 1])
        heading.settings['camera_latency_ms'] = 40
        worker = DetectionWorker(reader, make_detector(1), TagPoseEstimator(self.matrix, np.zeros(5), .1),
                                 cart_heading_estimator=heading, pelvis_compensator=compensator,
                                 cart_pose_estimator=self.estimator)
        self.addCleanup(worker.close)
        result = worker.latest(0)
        self.assertEqual(set(result['cart_pose_urdf']['tag_ids']), {0, 1})
        self.camera.set_pose_source('apriltag', 9.)
        self.assertTrue(self.camera.accept(json.dumps(heading_packet(result)), 10.))
        self.assertTrue(self.camera.fresh(10.))
        measured = self.camera.cart_pose_urdf['T_level_cart']
        np.testing.assert_allclose(np.asarray(measured)[:3, 3], truth[:3, 3], atol=.01)
        self.assertAlmostEqual(self.camera.yaw, .12, delta=.01)
        self.assertAlmostEqual(self.camera.stamp, 9.96)

    def test_pose_loss_holds_until_timeout_without_grasp_fallback(self):
        self.camera.set_pose_source('apriltag', 9.)
        packet, _ = self.observation()
        self.camera.accept(json.dumps(heading_packet(packet)), 10.)
        initial = self.camera.snapshot(10.)
        for now in (10.02, 10.12, 10.24, 10.27, 10.32):
            with self.subTest(now=now):
                packet, _ = self.observation(now)
                packet['tags'] = []
                # A valid legacy heading must not replace the held visual pose.
                packet['cart_pose_urdf'] = urdf_cart_observation(packet, self.estimator)
                self.assertTrue(self.camera.accept(json.dumps(heading_packet(packet)), now))
                held = self.camera.snapshot(now)
                for key in ('captured_monotonic', 'yaw_rad', 'tag_ids', 'cart_pose_urdf'):
                    self.assertEqual(held[key], initial[key])
                self.assertEqual(held['fresh'], now < 10.26)
                self.assertIn('no_cart_tag', held['reason'])
        recovered, _ = self.observation(10.34)
        recovered['cart_pose_urdf']['T_level_cart'][0][3] += .1
        self.assertTrue(self.camera.accept(json.dumps(heading_packet(recovered)), 10.34))
        self.assertTrue(self.camera.fresh(10.34))
        self.assertEqual(self.camera.cart_pose_urdf, recovered['cart_pose_urdf'])
        self.assertAlmostEqual(self.camera.stamp, 10.3)
        self.assertEqual(self.camera.reason, 'ok')

    def test_rejected_pose_holds_without_extending_capture_time(self):
        self.camera.set_pose_source('apriltag', 9.)
        packet, _ = self.observation()
        self.camera.accept(json.dumps(heading_packet(packet)), 10.)
        initial = self.camera.snapshot(10.)
        failures = [dict(valid=False, reason=reason) for reason in (
            'reprojection_error_too_large', 'planar_pose_ambiguous', 'compensation_unavailable')]
        failures += [dict(T_level_cart=np.zeros((4, 4)).tolist()), dict(captured_monotonic=0.)]
        for index, changes in enumerate(failures):
            with self.subTest(changes=changes):
                now = 10.02 + index*.02
                packet, _ = self.observation(now)
                packet['cart_pose_urdf'].update(changes)
                self.assertTrue(self.camera.accept(json.dumps(heading_packet(packet)), now))
                held = self.camera.snapshot(now)
                self.assertTrue(held['fresh'])
                self.assertEqual(held['cart_pose_urdf'], initial['cart_pose_urdf'])
                self.assertEqual(held['captured_monotonic'], initial['captured_monotonic'])
                self.assertAlmostEqual(held['yaw_rad'], .12)
        self.assertTrue(self.camera.fresh(10.259))
        self.assertFalse(self.camera.fresh(10.261))

    def test_invalid_first_pose_and_source_switch_cannot_reuse_old_pose(self):
        packet, _ = self.observation()
        self.camera.accept(json.dumps(heading_packet(packet)), 10.)
        self.camera.set_pose_source('apriltag', 10.)
        packet, _ = self.observation(10.1)
        packet['tags'] = []
        packet['cart_pose_urdf'] = urdf_cart_observation(packet, self.estimator)
        self.assertTrue(self.camera.accept(json.dumps(heading_packet(packet)), 10.1))
        self.assertFalse(self.camera.fresh(10.1))
        self.assertFalse(self.camera.cart_pose_urdf['valid'])
        self.assertIn('no_cart_tag', self.camera.reason)
        packet, _ = self.observation(10.2)
        self.camera.accept(json.dumps(heading_packet(packet)), 10.2)
        self.assertTrue(self.camera.fresh(10.2))
        self.camera.set_pose_source('grasp', 10.21)
        self.camera.set_pose_source('apriltag', 10.22)
        packet, _ = self.observation(10.3)
        packet['cart_pose_urdf'] = None
        self.camera.accept(json.dumps(heading_packet(packet)), 10.3)
        self.assertFalse(self.camera.fresh(10.3))
        self.assertFalse(self.camera.cart_pose_urdf['valid'])

    def test_invalid_pose_time_frame_rotation_and_tag_are_rejected(self):
        packet, _ = self.observation()
        pose = packet['cart_pose_urdf']
        for patch in (dict(captured_monotonic=10.), dict(reference_frame='camera'), dict(extrinsics='provided'),
                      dict(tag_ids=[2]), dict(yaw_rad=.9), dict(T_level_cart=np.zeros((4, 4)).tolist())):
            with self.subTest(patch=patch):
                self.assertFalse(CarmerYaw._cart_pose(dict(pose, **patch), 9.96)['valid'])

    def test_legacy_heading_and_offset_are_retained_when_not_selected(self):
        packet, _ = self.observation()
        packet['cart_pose_urdf'] = None
        self.camera.accept(json.dumps(heading_packet(packet)), 10.)
        self.assertAlmostEqual(self.camera.yaw, .2)
        self.camera.set_pose_source('apriltag', 10.)
        self.assertFalse(self.camera.fresh(10.))
        self.assertFalse(self.camera.accept(json.dumps(heading_packet(packet)), 10.))
        self.camera.set_pose_source('grasp', 10.01)
        packet, _ = self.observation(10.1)
        self.camera.accept(json.dumps(heading_packet(packet)), 10.1)
        self.assertAlmostEqual(self.camera.yaw, .2)


if __name__ == '__main__':
    unittest.main()
