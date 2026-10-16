"""Offline checks for configuration, circular fusion, and tag-loss reporting."""

import contextlib
import io
import json
import os
from pathlib import Path
import queue
import tempfile
import unittest
from unittest import mock

import cv2
import numpy as np

from carmer import apriltag_detector as app
from carmer.cart_heading import CartHeadingEstimator, fuse_cart_headings


def observation(tag_id, yaw_deg, *, valid=True):
    return {'id': tag_id, 'center_px': [320., 240.], 'cart_heading': {
        'valid': valid, 'yaw_deg': yaw_deg, 'yaw_rad': np.deg2rad(yaw_deg) if yaw_deg is not None else None,
        'reason': 'ok' if valid else 'pose_unavailable',
    }}


class ConfigurationTest(unittest.TestCase):
    def test_cart_ids_filter_and_cli_override(self):
        args = app.parse_args([])
        self.assertEqual(args.cart_tag_ids, [0, 1])
        estimator = CartHeadingEstimator(cart_tag_ids=args.cart_tag_ids)
        self.assertTrue(estimator.applies_to(0))
        self.assertTrue(estimator.applies_to(1))
        self.assertFalse(estimator.applies_to(2))
        single = app.parse_args(['--cart-tag-id', '1'])
        self.assertIsNone(single.cart_tag_ids)
        self.assertEqual(single.cart_tag_id, 1)
        multiple = app.parse_args(['--cart-tag-ids', '0', '1'])
        self.assertIsNone(multiple.cart_tag_id)
        self.assertEqual(multiple.cart_tag_ids, [0, 1])

    def test_shipped_config_loads_independently_of_working_directory(self):
        previous = Path.cwd()
        try:
            with tempfile.TemporaryDirectory() as temp:
                os.chdir(temp)
                args = app.parse_args([])
        finally:
            os.chdir(previous)
        self.assertEqual(args.config, app.CONFIG_FILE)
        self.assertEqual(args.quad_decimate, 2.)
        self.assertEqual(args.cart_to_tag_rpy_deg, [0., 0., 0.])
        self.assertIsNone(args.cart_tag_id)
        self.assertFalse(args.no_display)
        self.assertFalse(args.no_smoothing)
        self.assertEqual(args.intrinsics, app.INTRINSICS_FILE)
        self.assertEqual(args.snapshot_dir, app.SNAPSHOT_DIR)
        np.testing.assert_array_equal(CartHeadingEstimator(
            cart_to_tag_rpy_deg=args.cart_to_tag_rpy_deg).cart_to_tag, np.eye(3))

    def test_config_types_paths_and_explicit_cli_overrides(self):
        with tempfile.TemporaryDirectory() as temp:
            config = Path(temp) / 'settings.json'
            config.write_text(json.dumps({
                'quad-decimate': 1.5, 'cart-to-tag-rpy-deg': [0, 0, 0],
                'no-display': True, 'no-smoothing': True, 'profile': True,
                'intrinsics': 'intrinsics.json', 'snapshot-dir': 'captures', 'frames': 10,
            }))
            args = app.parse_args(['--config', str(config), '--quad-decimate', '2',
                                   '--display', '--smoothing', '--no-profile', '--frames', '25'])
            self.assertEqual(args.quad_decimate, 2.)
            self.assertEqual(args.frames, 25)
            self.assertFalse(args.no_display)
            self.assertFalse(args.no_smoothing)
            self.assertFalse(args.profile)
            self.assertEqual(args.intrinsics, Path(temp) / 'intrinsics.json')
            self.assertEqual(args.snapshot_dir, Path(temp) / 'captures')
            config.write_text(json.dumps({'cart-to-tag-rpy-deg': None, 'axis-length': None, 'cart-tag-id': None}))
            args = app.parse_args(['--config', str(config), '--tag-mount', 'rear'])
            self.assertIsNone(args.cart_to_tag_rpy_deg)
            self.assertEqual(args.tag_mount, 'rear')

    def test_bad_configuration_fails_before_camera_access(self):
        cases = [
            ('[]', 'JSON object'), ('{broken', 'Cannot load config'),
            ('{"quad-decimat": 2}', 'Unknown config'),
            ('{"no-display": "false"}', 'true or false'),
            ('{"frames": true}', 'integer'), ('{"frames": -1}', 'nonnegative'),
            ('{"quad-decimate": 0.5}', '>= 1'), ('{"quad-decimate": "2"}', 'number'),
            ('{"quad-decimate": NaN}', 'finite'), ('{"tag-size": 0}', 'positive'),
            ('{"cart-to-tag-rpy-deg": [0, 0]}', 'three numbers'),
            ('{"cart-to-tag-rpy-deg": [0, 0, Infinity]}', 'finite'),
            ('{"tag-mount": "wrong"}', 'one of'),
        ]
        with tempfile.TemporaryDirectory() as temp:
            config = Path(temp) / 'settings.json'
            for content, message in cases:
                with self.subTest(content=content), contextlib.redirect_stderr(io.StringIO()) as error:
                    config.write_text(content)
                    with self.assertRaises(SystemExit) as caught:
                        app.parse_args(['--config', str(config)])
                    self.assertEqual(caught.exception.code, 2)
                    self.assertIn(message, error.getvalue())
            config.unlink()
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                app.parse_args(['--config', str(config)])


class HeadingFusionTest(unittest.TestCase):
    def test_two_tags_and_wraparound_use_a_circular_mean(self):
        for angles, expected in (((10., 20.), 15.), ((179., -179.), 180.), ((-10., 10.), 0.)):
            with self.subTest(angles=angles):
                tags = [observation(1, angles[0]), observation(0, angles[1])]
                result = fuse_cart_headings(tags)
                self.assertTrue(result['valid'])
                self.assertEqual(result['mode'], 'fused')
                self.assertEqual(result['tag_ids'], [0, 1])
                self.assertEqual(result['tag_count'], 2)
                error = (result['yaw_deg'] - expected + 180) % 360 - 180
                self.assertAlmostEqual(error, 0.)

    def test_single_invalid_missing_and_opposing_headings(self):
        single = observation(7, 23.456)
        result = fuse_cart_headings([observation(8, None, valid=False), single])
        self.assertEqual(result['mode'], 'single')
        self.assertEqual(result['tag_ids'], [7])
        self.assertEqual(result['yaw_rad'], single['cart_heading']['yaw_rad'])
        for tags, reason in (([], 'no_tags_detected'),
                             ([{'id': 2}], 'no_valid_heading'),
                             ([observation(2, float('nan'))], 'no_valid_heading'),
                             ([observation(0, 0.), observation(1, 180.)], 'opposing_headings')):
            result = fuse_cart_headings(tags)
            self.assertFalse(result['valid'])
            self.assertIsNone(result['yaw_rad'])
            self.assertIsNone(result['yaw_deg'])
            self.assertEqual(result['reason'], reason)

    def test_worker_transitions_two_to_one_to_zero_without_stale_yaw(self):
        cv2.setNumThreads(1)
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        both = np.full((480, 640, 3), 255, np.uint8)
        for tag_id, x in ((0, 100), (1, 390)):
            both[160:304, x:x+144] = cv2.cvtColor(cv2.aruco.generateImageMarker(dictionary, tag_id, 144), cv2.COLOR_GRAY2BGR)
        one = both.copy()
        one[:, 350:] = 255
        empty = np.full_like(both, 255)

        class Reader:
            def __init__(self):
                self.items = queue.Queue()

            def latest(self, previous, *, with_timestamp=False):
                return self.items.get(timeout=3.)

            def close(self):
                self.items.put(None)

        reader = Reader()
        matrix, distortion = app.load_intrinsics(app.INTRINSICS_FILE)
        estimator = app.TagPoseEstimator(matrix, distortion)
        worker = app.DetectionWorker(reader, app.make_detector(2.), estimator, app.AdaptivePoseSmoother(estimator),
                                     CartHeadingEstimator(cart_to_tag_rpy_deg=[0, 0, 0]), profile=True)
        try:
            for number, (frame, mode, ids) in enumerate(((both, 'fused', [0, 1]), (one, 'single', [0]),
                                                         (empty, 'unavailable', [])), start=1):
                reader.items.put((number, frame, 60., number / 60.))
                packet = worker.latest(number - 1)
                heading = packet['fused_cart_heading']
                self.assertEqual(heading['mode'], mode)
                self.assertEqual(heading['tag_ids'], ids)
                self.assertEqual(packet['raw_fused_cart_heading']['tag_ids'], ids)
                if ids:
                    self.assertTrue(heading['valid'])
                    if len(ids) == 1:
                        self.assertEqual(heading['yaw_rad'], packet['tags'][0]['cart_heading']['yaw_rad'])
                else:
                    self.assertFalse(heading['valid'])
                    self.assertIsNone(heading['yaw_rad'])
                with tempfile.TemporaryDirectory() as temp:
                    app.save_result(packet, Path(temp))
                    saved = json.loads(next(Path(temp).glob('*.json')).read_text())
                    self.assertEqual(saved['fused_cart_heading'], heading)
                    self.assertEqual(saved['raw_fused_cart_heading'], packet['raw_fused_cart_heading'])
        finally:
            worker.close()

    def test_main_logs_fused_single_and_tag_loss_without_waiting_one_second(self):
        packets = []
        for tags in ([observation(0, 10.), observation(1, 20.)], [observation(0, 12.)], []):
            packets.append({'frame_number': len(packets) + 1, 'tags': tags,
                            'fused_cart_heading': fuse_cart_headings(tags),
                            'capture_fps': 60., 'detect_fps': 50., 'detection_ms': 18., 'processing_ms': 19.})
        worker = mock.Mock(processed=3, tag_frames=2, seen_ids={0, 1}, first_time=10., last_time=11.2)
        worker.latest.side_effect = packets + [None]
        reader = mock.Mock(measured_count=0)
        args = app.parse_args(['--no-display'])
        output = io.StringIO()
        with mock.patch.object(app, 'parse_args', return_value=args), \
             mock.patch.object(app, 'open_camera'), \
             mock.patch.object(app, 'CameraReader', return_value=reader), \
             mock.patch.object(app, 'RobotFeedback'), \
             mock.patch.object(app, 'DetectionWorker', return_value=worker), \
             mock.patch.object(app.time, 'monotonic', side_effect=[10., 11.1, 11.2]), \
             contextlib.redirect_stdout(output):
            app.main()
        text = output.getvalue()
        self.assertIn('fused_cart_yaw_pelvis_deg=+15.00', text)
        self.assertIn('yaw_source=fused yaw_tag_ids=[0,1]', text)
        self.assertIn('yaw_source=single yaw_tag_ids=[0]', text)
        self.assertIn(app.NO_TAG_MESSAGE, text)
        self.assertIn('fused_cart_yaw_pelvis=unavailable(no_tags_detected)', text)
        self.assertIn('Finished:', text)


if __name__ == '__main__':
    unittest.main()
