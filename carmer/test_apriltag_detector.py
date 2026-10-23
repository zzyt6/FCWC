"""Offline tag36h11 detection checks; run with the CWC conda environment."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import cv2
import numpy as np

from carmer.apriltag_detector import (
    AdaptivePoseSmoother, DetectionWorker, TagPoseEstimator, TagWindow, annotate, detect_tags,
    load_intrinsics, make_detector, rotation_to_rpy, save_result,
)
from carmer.cart_heading import CartHeadingEstimator, R_CAMERA_TO_TORSO


class AprilTagDetectionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cv2.setNumThreads(1)
        cls.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)

    def setUp(self):
        self.detector = make_detector()

    def marker(self, tag_id, size=144):
        return cv2.aruco.generateImageMarker(self.dictionary, tag_id, size)

    def test_multiple_ids_and_rotations(self):
        for rotation in range(4):
            with self.subTest(rotation=rotation):
                image = np.full((480, 640), 255, dtype=np.uint8)
                image[150:294, 70:214] = np.rot90(self.marker(7), rotation)
                image[190:334, 400:544] = np.rot90(self.marker(42), (rotation + 1) % 4)
                tags = detect_tags(self.detector, cv2.cvtColor(image, cv2.COLOR_GRAY2BGR))
                self.assertEqual([tag['id'] for tag in tags], [7, 42])
                for tag, center in zip(tags, ((142, 222), (472, 262))):
                    np.testing.assert_allclose(tag['center_px'], center, atol=1.5)
                    self.assertEqual(np.asarray(tag['corners_px']).shape, (4, 2))

    def test_perspective_center(self):
        source = np.array([[0, 0], [159, 0], [159, 159], [0, 159]], np.float32)
        quad = np.array([[170, 100], [430, 80], [390, 380], [210, 300]], np.float32)
        transform = cv2.getPerspectiveTransform(source, quad)
        image = cv2.warpPerspective(self.marker(123, 160), transform, (640, 480), borderValue=255)
        image = cv2.GaussianBlur(image, (3, 3), 0.6)
        tags = detect_tags(self.detector, image)
        self.assertEqual([tag['id'] for tag in tags], [123])
        along_diagonal = np.linalg.solve(np.column_stack((quad[2] - quad[0], quad[1] - quad[3])), quad[1] - quad[0])[0]
        expected = quad[0] + along_diagonal * (quad[2] - quad[0])
        np.testing.assert_allclose(tags[0]['center_px'], expected, atol=2.0)

    def test_blank_and_different_family(self):
        image = np.full((480, 640), 255, np.uint8)
        self.assertEqual(detect_tags(self.detector, image), [])
        other_dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_16h5)
        image[160:320, 240:400] = cv2.aruco.generateImageMarker(other_dictionary, 7, 160)
        self.assertEqual(detect_tags(self.detector, image), [])

    def cart_frame(self, positions, seed=0):
        """Textured background so the full-image search has real candidates to reject."""
        image = np.random.default_rng(seed).integers(60, 200, (480, 640)).astype(np.uint8)
        image = cv2.GaussianBlur(image, (0, 0), 3)
        for tag_id, (x, y) in positions.items():
            image[y - 8:y + 112, x - 8:x + 112] = 255
            image[y:y + 104, x:x + 104] = self.marker(tag_id, 104)
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    def test_tag_window_matches_full_search(self):
        window = TagWindow({0, 1})
        moves = ({0: (60, 200), 1: (300, 180)}, {0: (66, 204), 1: (305, 176)}, {0: (71, 199), 1: (311, 183)})
        for step, positions in enumerate(moves):
            frame = self.cart_frame(positions, step)
            full = detect_tags(self.detector, frame)
            region = window.region(frame.shape)
            self.assertEqual(region is None, step == 0)
            tags = detect_tags(self.detector, frame, region=region)
            self.assertTrue(window.covers(tags))
            window.update(tags)
            self.assertEqual([tag['id'] for tag in tags], [0, 1])
            if region is not None:
                self.assertTrue(all(value % TagWindow.ALIGN == 0 or value in (480, 640) for value in region))
                self.assertLess((region[2] - region[0]) * (region[3] - region[1]), 640 * 480 * .6)
            for windowed, reference in zip(tags, full):
                np.testing.assert_allclose(windowed['corners_px'], reference['corners_px'], atol=1e-4)

    def test_tag_window_falls_back_to_full_frame_without_losing_tags(self):
        class Reader:
            def __init__(self, frames):
                self.frames = frames

            def latest(self, previous, *, with_timestamp=False):
                if previous < len(self.frames):
                    return previous + 1, self.frames[previous], 60., previous / 60.
                return None

            def close(self):
                pass

        # Tag 1 jumps outside the window, then tag 0 disappears altogether.
        frames = [self.cart_frame({0: (60, 200), 1: (300, 180)}), self.cart_frame({0: (60, 200), 1: (300, 180)}),
                  self.cart_frame({0: (60, 200), 1: (500, 340)}), self.cart_frame({1: (500, 340)})]
        class Recorder:  # Sees every processed packet; latest() may skip ahead.
            def __init__(self):
                self.seen = []

            def submit(self, packet):
                self.seen.append(([tag['id'] for tag in packet['tags']], packet['detection_search']))

        recorder = Recorder()
        estimator = CartHeadingEstimator(cart_tag_ids=[0, 1])
        # Navigation runs the camera with --profile: a fallback frame must keep the
        # same stage keys, or the periodic profile report raises and stops the camera.
        worker = DetectionWorker(Reader(frames), self.detector, cart_heading_estimator=estimator,
                                 missing_tag_frames=recorder, tag_window=True, profile=True)
        worker.thread.join(timeout=5.)
        worker.close()
        seen = recorder.seen
        self.assertEqual(len({tuple(sample) for sample in worker.profile_samples}), 1)
        self.assertIn('read_wait=', worker.profile_report())
        self.assertEqual(seen, [([0, 1], 'full'), ([0, 1], 'window'), ([0, 1], 'window_fallback'),
                                ([1], 'window_fallback')])
        self.assertEqual(worker.search_counts, dict(full=1, window=1, window_fallback=2))
        # Without a cart ID set there is nothing to track: every frame is searched in full.
        self.assertIsNone(DetectionWorker(Reader([]), self.detector, tag_window=True).tag_window)

    def test_annotation_and_saved_results_keep_raw_frame(self):
        gray = np.full((480, 640), 255, np.uint8)
        gray[160:304, 240:384] = self.marker(0)
        raw = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        original = raw.copy()
        tags = detect_tags(self.detector, raw)
        self.assertEqual([tag['id'] for tag in tags], [0])
        annotated = annotate(raw, tags)
        np.testing.assert_array_equal(raw, original)
        self.assertFalse(np.array_equal(annotated, raw))
        packet = {'frame': raw, 'frame_number': 10, 'tags': tags, 'capture_fps': 60., 'detect_fps': 50., 'detection_ms': 5.}
        annotated = annotate(raw, tags, telemetry=packet)
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            save_result(packet, folder)
            record = json.loads(next(folder.glob('*.json')).read_text())
            self.assertEqual(record['family'], 'tag36h11')
            self.assertEqual(record['tags'], tags)
            self.assertEqual(record['image_size'], [640, 480])
            self.assertEqual(record['annotated_image_size'], [annotated.shape[1], annotated.shape[0]])
            np.testing.assert_array_equal(cv2.imread(str(next(folder.glob('*_raw.png')))), original)
            np.testing.assert_array_equal(cv2.imread(str(next(folder.glob('*_detected.png')))), annotated)


class OpenCVTagPoseTest(unittest.TestCase):
    def setUp(self):
        cv2.setNumThreads(1)
        self.camera_matrix = np.array([[600., 0, 320], [0, 600., 240], [0, 0, 1]])
        self.distortion = np.zeros(5)
        self.estimator = TagPoseEstimator(self.camera_matrix, self.distortion)
        self.detector = make_detector()

    def official_frame(self, tag_id=0, rotation=0):
        filename = Path(__file__).parent / 'testdata' / f'tag36_11_{tag_id:05d}.png'
        official = cv2.imread(str(filename), cv2.IMREAD_GRAYSCALE)
        self.assertIsNotNone(official)
        canvas = np.full((480, 640), 255, np.uint8)
        canvas[100:340, 200:440] = cv2.resize(np.rot90(official, rotation), (240, 240), interpolation=cv2.INTER_NEAREST)
        return cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)

    def test_printed_tag_axes_x_right_y_up_follow_all_four_rotations(self):
        # Keep exact synthetic-pixel pose checks at full detection resolution.
        # Downsampled corner quantization is covered separately below.
        detector = make_detector(quad_decimate=1.0)
        for tag_id in (0, 1):
            for turn in range(4):
                with self.subTest(tag_id=tag_id, rotation=turn):
                    tags = detect_tags(detector, self.official_frame(tag_id, turn), self.estimator)
                    self.assertEqual([tag['id'] for tag in tags], [tag_id])
                    pose = tags[0]['pose']
                    self.assertIsNotNone(pose)
                    angle = -turn * np.pi / 2
                    # Output axes follow printed official tags: X right, Y up,
                    # Z toward camera, including the requested local Z half-turn.
                    expected = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]) @ np.diag([1., -1., -1.])
                    np.testing.assert_allclose(pose['R_tag_to_camera'], expected, atol=0.002)
                    # 192 px black edge, 600 px focal length, 10 cm actual edge.
                    np.testing.assert_allclose(pose['tvec_m'], [0, -0.0104166667, 0.3125], atol=1e-5)
                    axes = np.asarray(pose['axes_px'])
                    np.testing.assert_allclose(axes[0], [320, 220], atol=0.1)
                    direction = axes[1:3] - axes[0]
                    np.testing.assert_allclose(direction / np.linalg.norm(direction, axis=1)[:, None], expected[:2, :2].T, atol=0.003)
                    # +Z points OUT of the front face: for this off-center view
                    # its endpoint is farther from the principal point.
                    self.assertLess(axes[3, 1], axes[0, 1])
                    self.assertLess(pose['R_tag_to_camera'][2][2], -.999)

    def test_downsampling_preserves_original_image_coordinates_and_pose_scale(self):
        for factor in (1.0, 1.5, 2.0):
            detector = make_detector(factor)
            for turn in range(4):
                with self.subTest(quad_decimate=factor, rotation=turn):
                    tags = detect_tags(detector, self.official_frame(rotation=turn), self.estimator)
                    self.assertEqual([tag['id'] for tag in tags], [0])
                    tag = tags[0]
                    # All coordinates must remain in 640x480 pixels, not in the
                    # smaller quad-search image. Allow interpolation quantization.
                    np.testing.assert_allclose(tag['center_px'], [320, 220], atol=1.5)
                    quad = np.array(tag['corners_px'])
                    np.testing.assert_allclose(quad.min(axis=0), [224, 124], atol=1.5)
                    np.testing.assert_allclose(quad.max(axis=0), [416, 316], atol=1.5)
                    self.assertIsNotNone(tag['pose'])
                    np.testing.assert_allclose(tag['pose']['tvec_m'], [0, -0.0104166667, 0.3125], atol=.002)
                    angle = -turn * np.pi / 2
                    expected = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]) @ np.diag([1., -1., -1.])
                    np.testing.assert_allclose(tag['pose']['R_tag_to_camera'], expected, atol=.03)

    def test_invalid_downsampling(self):
        for value in (0, .5, -1, float('nan'), float('inf')):
            with self.subTest(value=value), self.assertRaises(ValueError):
                make_detector(value)

    def test_opencv_generated_tags_xy_reversed_z_out(self):
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        for turn in range(4):
            with self.subTest(rotation=turn):
                image = np.full((480, 640), 255, np.uint8)
                marker = cv2.aruco.generateImageMarker(dictionary, 0, 192)
                image[124:316, 224:416] = np.rot90(marker, turn)
                tag = detect_tags(self.detector, image, self.estimator)[0]
                self.assertEqual(tag['id'], 0)
                angle = -turn * np.pi / 2
                expected = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]) @ np.diag([-1., 1., -1.])
                np.testing.assert_allclose(tag['pose']['R_tag_to_camera'], expected, atol=.002)
                axes = np.array(tag['pose']['axes_px'])
                if turn == 0:
                    self.assertLess(axes[1, 0], axes[0, 0])
                    self.assertGreater(axes[2, 1], axes[0, 1])
                    self.assertLess(axes[3, 1], axes[0, 1])

    def test_tilted_pose_metric_scale_and_projection(self):
        # OpenCV canonical TL, TR, BR, BL coordinates, Y up and Z outward.
        points = np.array([[-.05, .05, 0], [.05, .05, 0], [.05, -.05, 0], [-.05, -.05, 0]])
        tilt = cv2.Rodrigues(np.array([.35, -.45, .2]))[0]
        rvec = cv2.Rodrigues(tilt @ np.diag([1., -1., -1.]))[0]
        tvec = np.array([.04, -.03, .7])
        projected, _ = cv2.projectPoints(points, rvec, tvec, self.camera_matrix, self.distortion)
        detected_corners = projected.reshape(4, 2)
        pose = self.estimator.estimate(detected_corners)
        self.assertIsNotNone(pose)
        rotation, _ = cv2.Rodrigues(rvec)
        output_rotation = np.asarray(pose['R_tag_to_camera'])
        np.testing.assert_allclose(output_rotation[:, :2], -rotation[:, :2], atol=1e-8)
        np.testing.assert_allclose(output_rotation[:, 2], rotation[:, 2], atol=1e-8)
        self.assertAlmostEqual(np.linalg.det(output_rotation), 1.0, places=8)
        np.testing.assert_allclose(np.cross(output_rotation[:, 0], output_rotation[:, 1]), output_rotation[:, 2], atol=1e-8)
        np.testing.assert_allclose(cv2.Rodrigues(np.asarray(pose['rvec']))[0], output_rotation, atol=1e-8)
        np.testing.assert_allclose(pose['tvec_m'], tvec, atol=1e-8)
        self.assertLess(pose['reprojection_rmse_px'], 1e-8)
        # Project -X, -Y, +Z from the original pose to check the requested change.
        expected_axes, _ = cv2.projectPoints(np.array([[0.,0,0],[-.10,0,0],[0,-.10,0],[0,0,.10]]), rvec, tvec, self.camera_matrix, self.distortion)
        np.testing.assert_allclose(pose['axes_px'], expected_axes.reshape(4, 2), atol=1e-7)
        bigger = TagPoseEstimator(self.camera_matrix, self.distortion, tag_size=.20).estimate(detected_corners)
        np.testing.assert_allclose(bigger['tvec_m'], tvec * 2, atol=1e-8)
        np.testing.assert_allclose(bigger['axes_px'], pose['axes_px'], atol=1e-7)

    def test_invalid_size_and_degenerate_corners(self):
        for size in (0, -1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                TagPoseEstimator(self.camera_matrix, self.distortion, tag_size=size)
        self.assertIsNone(self.estimator.estimate(np.zeros((4,2))))
        self.assertIsNone(self.estimator.estimate(np.full((4,2), np.nan)))

    def test_calibration_dimensions_and_distortion(self):
        base = Path(__file__).parent / 'calibrate_camera' / 'camera_intrinsics.json'
        matrix, distortion = load_intrinsics(base)
        self.assertEqual(matrix.shape, (3,3))
        np.testing.assert_array_equal(distortion, np.zeros(5))
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'intrinsics.json'
            data = json.loads(base.read_text())
            data['stream']['width'] = 1280
            path.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                load_intrinsics(path)
            data['stream']['width'] = 640
            data['distortion_coefficients'][0] = .1
            path.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                load_intrinsics(path)

    def test_axes_annotation_and_snapshot(self):
        raw = self.official_frame()
        original = raw.copy()
        tags = detect_tags(self.detector, raw, self.estimator)
        annotated = annotate(raw, tags, self.camera_matrix, self.distortion)
        axes = np.array(tags[0]['pose']['axes_px'])
        for index, color in ((1, [0,0,255]), (2, [0,255,0])):
            x,y = np.rint((axes[0] + axes[index]) / 2).astype(int)
            np.testing.assert_array_equal(annotated[y,x], color)
        np.testing.assert_array_equal(raw, original)
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            packet = {'frame':raw,'tags':tags,'frame_number':1,
                      'camera_matrix':self.camera_matrix.tolist(),
                      'distortion_coefficients':self.distortion.tolist(),
                      'detection_ms':12.0, 'processing_ms':15.0}
            save_result(packet, folder)
            annotated = annotate(raw, tags, self.camera_matrix, self.distortion, telemetry=packet)
            record = json.loads(next(folder.glob('*.json')).read_text())
            self.assertEqual(record['tags'], tags)
            self.assertEqual(record['coordinate_units']['tvec_m'], 'meters')
            self.assertEqual(record['coordinate_units']['rpy_rad'], 'radians')
            self.assertEqual(record['coordinate_units']['rpy_deg'], 'degrees')
            self.assertEqual(record['pose_convention']['rpy_order'], ['roll', 'pitch', 'yaw'])
            self.assertEqual(record['tags'][0]['pose']['rpy_deg'], tags[0]['pose']['rpy_deg'])
            self.assertEqual(record['detection_ms'], 12.0)
            self.assertEqual(record['processing_ms'], 15.0)
            self.assertIn('detection_ms', record['timing_convention'])
            self.assertEqual(record['pose_convention']['name'], 'opencv_aruco_centered_z180')
            self.assertIn('out of tag', record['pose_convention']['tag_axes'])
            np.testing.assert_array_equal(cv2.imread(str(next(folder.glob('*_detected.png')))), annotated)


class RollPitchYawTest(unittest.TestCase):
    @staticmethod
    def rotation(angles):
        roll, pitch, yaw = angles
        # Compose independent axis-angle rotations using OpenCV, then recover RPY.
        return (cv2.Rodrigues(np.array([0., 0., yaw]))[0]
                @ cv2.Rodrigues(np.array([0., pitch, 0.]))[0]
                @ cv2.Rodrigues(np.array([roll, 0., 0.]))[0])

    def test_known_angles_and_near_singular_poses(self):
        for degrees in ([0., 0., 0.], [25., -30., 70.], [180., 0., 0.],
                        [-160., 89.99, -179.9], [120., -89.99, 179.9]):
            with self.subTest(degrees=degrees):
                expected = np.deg2rad(degrees)
                actual, singular = rotation_to_rpy(self.rotation(expected))
                self.assertFalse(singular)
                np.testing.assert_allclose(actual, expected, atol=1e-10)

    def test_gimbal_lock_preserves_rotation_and_marks_ambiguous_yaw(self):
        for pitch in (-90., 90.):
            for roll, yaw in ((30., 75.), (-130., -170.)):
                with self.subTest(pitch=pitch, roll=roll, yaw=yaw):
                    rotation = self.rotation(np.deg2rad([roll, pitch, yaw]))
                    actual, singular = rotation_to_rpy(rotation)
                    self.assertTrue(singular)
                    self.assertEqual(actual[2], 0.)
                    np.testing.assert_allclose(self.rotation(actual), rotation, atol=1e-8)

    def test_raw_and_smoothed_rpy_follow_their_rotation_across_angle_wrap(self):
        estimator = TagPoseEstimator(np.array([[600., 0., 320.], [0., 600., 240.], [0., 0., 1.]]), np.zeros(5))
        smoother = AdaptivePoseSmoother(estimator)
        outputs = []
        for i, yaw in enumerate((179., -179.)):
            raw = estimator.pose_from_transform(self.rotation(np.deg2rad([20., -15., yaw])), [0., 0., 1.])
            filtered = smoother.update(0, raw, i / 45.)
            np.testing.assert_allclose(raw['rpy_deg'], [20., -15., yaw], atol=1e-8)
            for pose in (raw, filtered):
                np.testing.assert_allclose(np.rad2deg(pose['rpy_rad']), pose['rpy_deg'], atol=1e-8)
                np.testing.assert_allclose(self.rotation(pose['rpy_rad']), pose['R_tag_to_camera'], atol=1e-8)
                self.assertFalse(pose['rpy_singular'])
            outputs.append(filtered['rpy_deg'][2])
        self.assertTrue(all(abs(yaw) > 170. for yaw in outputs))


class CartHeadingTest(unittest.TestCase):
    @staticmethod
    def rotation(degrees):
        return RollPitchYawTest.rotation(np.deg2rad(degrees))

    def test_nominal_optical_axes_and_urdf_pitch(self):
        import xml.etree.ElementTree as ET
        root = ET.parse(Path(__file__).resolve().parents[1] / 'G1_URDF/g1.urdf').getroot()
        joint = root.find("joint[@name='d435_joint']")
        pitch = float(joint.find('origin').attrib['rpy'].split()[1])
        mounting = cv2.Rodrigues(np.array([0., pitch, 0.]))[0]
        # Optical right maps to body right, down to body down, forward to body forward.
        np.testing.assert_allclose(R_CAMERA_TO_TORSO[:, 0], mounting @ [0., -1., 0.], atol=1e-12)
        np.testing.assert_allclose(R_CAMERA_TO_TORSO[:, 1], mounting @ [0., 0., -1.], atol=1e-12)
        np.testing.assert_allclose(R_CAMERA_TO_TORSO[:, 2], mounting @ [1., 0., 0.], atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(R_CAMERA_TO_TORSO), 1.)

    def test_rear_and_top_tag_axes_define_cart_forward_and_up(self):
        for mount, forward, up in (('rear', [0, 0, -1], [0, 1, 0]),
                                   ('top', [0, 1, 0], [0, 0, 1])):
            estimator = CartHeadingEstimator(tag_mount=mount)
            np.testing.assert_array_equal(estimator.cart_to_tag[:, 0], forward)
            np.testing.assert_array_equal(estimator.cart_to_tag[:, 2], up)
            np.testing.assert_array_equal(estimator.cart_to_tag[:, 1], np.cross(up, forward))
            self.assertAlmostEqual(np.linalg.det(estimator.cart_to_tag), 1.)

    def test_heading_recovers_cart_rotation_across_waist_and_wraparound(self):
        for mount in ('rear', 'top'):
            for waist in (-30., 0., 25.):
                estimator = CartHeadingEstimator(waist_yaw_deg=waist, tag_mount=mount)
                for yaw in (-179.9, -45., 0., 30., 179.9):
                    with self.subTest(mount=mount, waist=waist, yaw=yaw):
                        cart_rotation = self.rotation([7., -12., yaw])
                        # Build the observation from independent forward kinematics.
                        torso_rotation = self.rotation([0., 0., waist])
                        observation = R_CAMERA_TO_TORSO.T @ torso_rotation.T @ cart_rotation @ estimator.cart_to_tag.T
                        heading = estimator.estimate({'R_tag_to_camera': observation.tolist()})
                        self.assertTrue(heading['valid'])
                        self.assertAlmostEqual(heading['yaw_deg'], yaw)
                        self.assertAlmostEqual(heading['yaw_rad'], np.deg2rad(yaw))
                        np.testing.assert_allclose(heading['R_cart_to_pelvis'], cart_rotation, atol=1e-12)

    def test_projected_tag_corners_produce_correct_pelvis_heading(self):
        matrix = np.array([[600., 0., 320.], [0., 600., 240.], [0., 0., 1.]])
        pose_estimator = TagPoseEstimator(matrix, np.zeros(5))
        for mount in ('rear', 'top'):
            heading_estimator = CartHeadingEstimator(waist_yaw_deg=15., tag_mount=mount)
            for yaw in (-25., 0., 35.):
                with self.subTest(mount=mount, yaw=yaw):
                    cart_rotation = self.rotation([3., -5., yaw])
                    tag_in_camera = (R_CAMERA_TO_TORSO.T @ self.rotation([0., 0., -15.])
                                     @ cart_rotation @ heading_estimator.cart_to_tag.T)
                    # Pose estimator accepts OpenCV's unrotated marker corner order.
                    opencv_rotation = tag_in_camera @ np.diag([-1., -1., 1.])
                    corners, _ = cv2.projectPoints(pose_estimator.object_points,
                                                  cv2.Rodrigues(opencv_rotation)[0],
                                                  np.array([.02, -.03, .8]), matrix, np.zeros(5))
                    pose = pose_estimator.estimate(corners)
                    self.assertIsNotNone(pose)
                    heading = heading_estimator.estimate(pose)
                    self.assertTrue(heading['valid'])
                    self.assertAlmostEqual(heading['yaw_deg'], yaw, places=5)

    def test_unavailable_heading_and_custom_mount(self):
        estimator = CartHeadingEstimator(cart_tag_id=7, cart_to_tag_rpy_deg=[15., 30., -45.])
        np.testing.assert_allclose(estimator.cart_to_tag, self.rotation([15., 30., -45.]), atol=1e-12)
        self.assertTrue(estimator.applies_to(7))
        self.assertFalse(estimator.applies_to(8))
        self.assertEqual(estimator.estimate(None)['reason'], 'pose_unavailable')
        self.assertEqual(estimator.estimate({'R_tag_to_camera': np.full((3, 3), np.nan)})['reason'], 'invalid_rotation')
        vertical_cart = self.rotation([0., 90., 12.])
        observation = R_CAMERA_TO_TORSO.T @ vertical_cart @ estimator.cart_to_tag.T
        result = estimator.estimate({'R_tag_to_camera': observation})
        self.assertFalse(result['valid'])
        self.assertIsNone(result['yaw_deg'])
        self.assertIsNone(result['yaw_rad'])
        self.assertEqual(result['reason'], 'forward_axis_vertical')
        for kwargs in ({'waist_yaw_deg': float('nan')}, {'cart_tag_id': -1},
                       {'tag_mount': 'bad'}, {'cart_to_tag_rpy_deg': [0., 1., float('inf')]}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                CartHeadingEstimator(**kwargs)

    def test_worker_heading_selection_loss_and_snapshot(self):
        matrix = np.array([[600., 0., 320.], [0., 600., 240.], [0., 0., 1.]])
        pose_estimator = TagPoseEstimator(matrix, np.zeros(5))
        heading_estimator = CartHeadingEstimator(cart_tag_id=7, tag_mount='top', waist_yaw_deg=10.)
        marker = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11), 7, 144)
        frame = np.full((480, 640, 3), 255, np.uint8)
        frame[180:324, 240:384] = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
        blank = np.full_like(frame, 255)

        class Reader:
            def __init__(self, frames):
                self.frames = frames

            def latest(self, previous, *, with_timestamp=False):
                if previous < len(self.frames):
                    return previous + 1, self.frames[previous], 60., previous / 60.
                return None

            def close(self):
                pass

        for smoothing in (False, True):
            smoother = AdaptivePoseSmoother(pose_estimator) if smoothing else None
            worker = DetectionWorker(Reader([frame]), make_detector(), pose_estimator, smoother, heading_estimator)
            worker.thread.join(timeout=3.)
            try:
                packet = worker.latest(0)
                tag = packet['tags'][0]
                self.assertTrue(tag['cart_heading']['valid'])
                self.assertEqual(tag['cart_heading'], tag['raw_cart_heading'])
                self.assertGreaterEqual(packet['processing_ms'], packet['detection_ms'])
                self.assertEqual(packet['cart_heading_settings']['waist_yaw_deg'], 10.)
                with tempfile.TemporaryDirectory() as temp:
                    save_result(packet, Path(temp))
                    record = json.loads(next(Path(temp).glob('*.json')).read_text())
                    self.assertEqual(record['tags'][0]['cart_heading'], tag['cart_heading'])
                    self.assertEqual(record['cart_heading_settings']['reference_frame'], 'pelvis')
                    self.assertEqual(record['coordinate_units']['yaw_rad'], 'radians')
            finally:
                worker.close()
        # No previous heading survives a following empty frame.
        worker = DetectionWorker(Reader([frame, blank]), make_detector(), pose_estimator,
                                 AdaptivePoseSmoother(pose_estimator), heading_estimator)
        worker.thread.join(timeout=3.)
        try:
            self.assertEqual(worker.latest(0)['tags'], [])
            self.assertEqual(worker.processed, 2)
        finally:
            worker.close()
        # Other tags still get camera poses, without being labeled as the selected cart.
        worker = DetectionWorker(Reader([frame]), make_detector(), pose_estimator,
                                 cart_heading_estimator=CartHeadingEstimator(cart_tag_id=8))
        worker.thread.join(timeout=3.)
        try:
            tag = worker.latest(0)['tags'][0]
            self.assertIsNotNone(tag['pose'])
            self.assertNotIn('cart_heading', tag)
        finally:
            worker.close()


class DetectionTimingTest(unittest.TestCase):
    def test_profile_keeps_all_processed_frames_and_preserves_tag_loss(self):
        cv2.setNumThreads(1)
        estimator = TagPoseEstimator(np.array([[600., 0., 320.], [0., 600., 240.], [0., 0., 1.]]), np.zeros(5))
        frame = np.full((480, 640, 3), 255, np.uint8)
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        marker = cv2.aruco.generateImageMarker(dictionary, 0, 144)
        frame[160:304, 240:384] = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
        frames = [frame, frame, np.full_like(frame, 255)]

        class Reader:
            def latest(self, previous, *, with_timestamp=False):
                if previous < len(frames):
                    return previous + 1, frames[previous], 60., previous / 60.
                return None

            def close(self):
                pass

        for profile in (False, True):
            with self.subTest(profile=profile):
                worker = DetectionWorker(Reader(), make_detector(), estimator, AdaptivePoseSmoother(estimator),
                                         CartHeadingEstimator(), profile=profile)
                worker.thread.join(timeout=3.)
                try:
                    self.assertFalse(worker.thread.is_alive())
                    packet = worker.latest(0)  # Preview consumes only the last frame.
                    self.assertEqual(packet['tags'], [])
                    self.assertEqual(worker.tag_frames, 2)
                    self.assertEqual(worker.processed, 3)
                    if not profile:
                        self.assertNotIn('stage_timings_ms', packet)
                        self.assertIsNone(worker.profile_report())
                        continue
                    self.assertIn('last 3 frames', worker.profile_report())
                    self.assertIn('mean/p95 ms', worker.profile_report())
                    stages = packet['stage_timings_ms']
                    self.assertEqual(stages['pose'], 0.)
                    self.assertEqual(stages['process'], packet['processing_ms'])
                    self.assertGreater(worker.profile_samples[0]['pose'], 0.)
                    for name in ('read_wait', 'gray', 'tag', 'pose', 'tag_output', 'smooth', 'yaw', 'process'):
                        self.assertTrue(np.isfinite(stages[name]))
                        self.assertGreaterEqual(stages[name], 0.)
                    with tempfile.TemporaryDirectory() as temp:
                        save_result(packet, Path(temp))
                        record = json.loads(next(Path(temp).glob('*.json')).read_text())
                        self.assertEqual(record['stage_timings_ms'], stages)
                finally:
                    worker.close()

    def test_worker_measures_detection_and_total_processing_with_and_without_tags(self):
        cv2.setNumThreads(1)
        estimator = TagPoseEstimator(np.array([[600., 0., 320.], [0., 600., 240.], [0., 0., 1.]]), np.zeros(5))
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        for has_tag in (False, True):
            with self.subTest(has_tag=has_tag):
                frame = np.full((480, 640, 3), 255, np.uint8)
                if has_tag:
                    marker = cv2.aruco.generateImageMarker(dictionary, 0, 144)
                    frame[160:304, 240:384] = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)

                class Reader:
                    def latest(self, previous, *, with_timestamp=False):
                        return (1, frame, 60., .5) if previous == 0 else None

                    def close(self):
                        pass

                with mock.patch('carmer.apriltag_detector.time.monotonic', side_effect=[1., 1.012, 1.015]):
                    worker = DetectionWorker(Reader(), make_detector(), estimator, AdaptivePoseSmoother(estimator))
                    worker.thread.join(timeout=2.)
                try:
                    self.assertFalse(worker.thread.is_alive())
                    self.assertIsNone(worker.error)
                    packet = worker.latest(0)
                    self.assertIs(packet['frame'], frame)
                    self.assertAlmostEqual(packet['detection_ms'], 12.)
                    self.assertAlmostEqual(packet['processing_ms'], 15.)
                    self.assertEqual(bool(packet['tags']), has_tag)
                    if has_tag:
                        self.assertIn('rpy_rad', packet['tags'][0]['raw_pose'])
                        self.assertIn('rpy_rad', packet['tags'][0]['pose'])
                    self.assertIsNone(worker.latest(1))
                finally:
                    worker.close()


class PoseSmoothingTest(unittest.TestCase):
    def setUp(self):
        cv2.setNumThreads(1)
        self.camera_matrix = np.array([[600., 0, 320], [0, 600., 240], [0, 0, 1.]])
        self.estimator = TagPoseEstimator(self.camera_matrix, np.zeros(5), tag_size=.10, axis_length=.10)

    def pose(self, x, angle=0.0):
        rotation, _ = cv2.Rodrigues(np.array([0., angle, 0.]))
        value = self.estimator.pose_from_transform(rotation, np.array([x, 0., .7]), .1)
        self.assertIsNotNone(value)
        return value

    def test_static_noise_is_reduced_and_rotation_stays_right_handed(self):
        smoother = AdaptivePoseSmoother(self.estimator, min_cutoff_hz=2., beta=.35)
        conservative = AdaptivePoseSmoother(self.estimator, xy_beta=.35)
        raw_x = []
        filtered_x = []
        conservative_x = []
        filtered_rotations = []
        for i in range(300):
            x = .1 + .0015 * np.sin(i * 1.7) + .0008 * np.sin(i * 3.1)
            raw = self.pose(x, .08 + .003 * np.sin(i * 1.3))
            filtered = smoother.update(7, raw, i / 45.)
            raw_x.append(x)
            filtered_x.append(filtered['tvec_m'][0])
            conservative_x.append(conservative.update(7, raw, i / 45.)['tvec_m'][0])
            filtered_rotations.append(np.asarray(filtered['R_tag_to_camera']))
        self.assertLess(np.std(filtered_x[30:]), .6 * np.std(raw_x[30:]))
        self.assertLess(np.std(filtered_x[30:]), 1.2 * np.std(conservative_x[30:]))
        for rotation in filtered_rotations:
            np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-8)
            self.assertAlmostEqual(np.linalg.det(rotation), 1., places=8)

    def test_xy_translation_follows_step_and_continuous_motion_quickly(self):
        dt = 1. / 43.25
        for axis in (0, 1):
            with self.subTest(axis=axis):
                smoother = AdaptivePoseSmoother(self.estimator)
                initial = np.array([0., 0., .7])
                smoother.update(0, self.estimator.pose_from_transform(np.eye(3), initial), 0.)
                target = initial.copy()
                target[axis] = .10
                values = [smoother.update(0, self.estimator.pose_from_transform(np.eye(3), target), i * dt)
                          ['tvec_m'][axis] for i in range(1, 20)]
                self.assertGreaterEqual(values[3], .09)  # >=90% within 4 updates / 93 ms.
                self.assertTrue(np.all(np.diff(values) >= 0.))
                self.assertLessEqual(max(values), .10 + 1e-12)

                smoother = AdaptivePoseSmoother(self.estimator)
                errors = []
                for i in range(150):
                    point = initial.copy()
                    point[axis] = .3 * i * dt  # Continuous 30 cm/s translation.
                    filtered = smoother.update(0, self.estimator.pose_from_transform(np.eye(3), point), i * dt)
                    errors.append(point[axis] - filtered['tvec_m'][axis])
                self.assertLess(np.mean(errors[50:]), .008)  # <8 mm / 27 ms steady lag.

    def test_x_motion_does_not_relax_y_depth_or_rotation_smoothing(self):
        still_x = AdaptivePoseSmoother(self.estimator)
        moving_x = AdaptivePoseSmoother(self.estimator)
        for i in range(150):
            rotation, _ = cv2.Rodrigues(np.array([0., .003 * np.sin(i * 1.3), 0.]))
            point = np.array([0., .001 * np.sin(i * 1.7), .7 + .002 * np.sin(i * 2.1)])
            reference = still_x.update(0, self.estimator.pose_from_transform(rotation, point), i / 43.25)
            point[0] = .15 * np.sin(i / 20.)
            moving = moving_x.update(0, self.estimator.pose_from_transform(rotation, point), i / 43.25)
            np.testing.assert_allclose(moving['tvec_m'][1:], reference['tvec_m'][1:], atol=1e-12)
            np.testing.assert_allclose(moving['R_tag_to_camera'], reference['R_tag_to_camera'], atol=1e-12)

    def test_fast_xy_round_trips_follow_reversals_without_overshoot(self):
        for fps in (35., 43.25, 50.):
            with self.subTest(fps=fps):
                smoother = AdaptivePoseSmoother(self.estimator)
                times = np.arange(int(5 * fps)) / fps
                elapsed = np.clip(times - .5, 0., 4.)
                phase = elapsed % 2.
                target = np.where(phase <= 1., -.15 + .3 * phase, .15 - .3 * (phase - 1.))
                output = np.array([smoother.update(0, self.pose(x), t)['tvec_m'][0]
                                   for x, t in zip(target, times)])
                error = np.abs(output - target)
                for turn in (1.5, 2.5, 3.5):
                    self.assertLess(np.max(error[(times > turn) & (times < turn + .3)]), .005)
                stable = (times >= .8) & (times < 1.4)
                self.assertLess(np.mean(error[stable]), .004)
                self.assertGreaterEqual(output.min(), -.15 - 1e-12)
                self.assertLessEqual(output.max(), .15 + 1e-12)

    def test_noise_allowance_does_not_create_a_position_deadband(self):
        for displacement in (.0001, -.001, .01):
            with self.subTest(displacement=displacement):
                smoother = AdaptivePoseSmoother(self.estimator)
                smoother.update(0, self.pose(0.), 0.)
                positions = [smoother.update(0, self.pose(displacement), i / 43.25)['tvec_m'][0]
                             for i in range(1, 44)]
                self.assertGreater(positions[0] * displacement, 0.)
                self.assertAlmostEqual(positions[-1], displacement, delta=abs(displacement) * .01)

    def test_fast_xy_still_suppresses_stationary_white_noise(self):
        rng = np.random.default_rng(20261012)
        noise = rng.normal(size=900)
        for noise_std, output_limit in ((.001, .0005), (.002, .001)):
            with self.subTest(noise_std=noise_std):
                smoother = AdaptivePoseSmoother(self.estimator)
                output = [smoother.update(0, self.pose(x * noise_std), i / 43.25)['tvec_m'][0]
                          for i, x in enumerate(noise)]
                self.assertLess(np.std(output[200:]), output_limit)

    def test_smoothing_rejects_invalid_response_gains(self):
        for name in ('beta', 'xy_beta', 'xy_noise_allowance_m'):
            for value in (-1., float('nan'), float('inf')):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    AdaptivePoseSmoother(self.estimator, **{name: value})
        for value in (0., -1., float('nan'), float('inf')):
            with self.subTest(xy_speed_cutoff_hz=value), self.assertRaises(ValueError):
                AdaptivePoseSmoother(self.estimator, xy_speed_cutoff_hz=value)

    def test_filter_follows_step_and_resets_after_gap(self):
        smoother = AdaptivePoseSmoother(self.estimator, min_cutoff_hz=2., beta=.35, reset_gap_s=.5)
        first = smoother.update(3, self.pose(0.), 0.)
        self.assertAlmostEqual(first['tvec_m'][0], 0.)
        values = [smoother.update(3, self.pose(.10), i / 45.)['tvec_m'][0] for i in range(1, 15)]
        self.assertGreater(values[-1], values[0])
        self.assertLess(values[-1], .10)
        reset = smoother.update(3, self.pose(.30), 1.5)
        self.assertAlmostEqual(reset['tvec_m'][0], .30, places=8)

    def test_tags_keep_raw_and_filtered_pose_separate(self):
        detector = make_detector(1.5)
        smoother = AdaptivePoseSmoother(self.estimator)
        frame = np.full((480, 640), 255, dtype=np.uint8)
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        frame[124:316, 224:416] = cv2.aruco.generateImageMarker(dictionary, 0, 192)
        tags = detect_tags(detector, cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR), self.estimator)
        raw_pose = tags[0]['pose']
        tags[0]['raw_pose'] = raw_pose
        tags[0]['pose'] = smoother.update(tags[0]['id'], raw_pose, 0.)
        self.assertIn('raw_pose', tags[0])
        self.assertIn('smoothing', tags[0]['pose'])
        self.assertEqual(tags[0]['raw_pose']['tag_size_m'], tags[0]['pose']['tag_size_m'])


if __name__ == '__main__':
    unittest.main()
