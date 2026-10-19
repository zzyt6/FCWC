"""Offline checks for capture-time gating and lossless, asynchronous evidence."""
import contextlib
import io
import json
import os
from pathlib import Path
import queue
import tempfile
import threading
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from carmer import apriltag_detector as app
from carmer.missing_tag_frames import MissingTagFrames


def packet(number, stamp, ids=()):
    return dict(frame_number=number, frame_timestamp_monotonic=stamp,
                frame=np.full((32, 40, 3), number, np.uint8),
                tags=[dict(id=tag) for tag in ids], detector_settings=dict(quad_decimate=2.))


class MissingTagFramesTest(unittest.TestCase):
    def test_capture_time_gate_single_tag_background_and_exact_image(self):
        read_fd, write_fd = os.pipe()
        try:
            with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
                recorder = MissingTagFrames(temp, control_fd=read_fd)
                try:
                    self.assertFalse(recorder.submit(packet(1, 9.)))
                    # A delayed frame from the enabled interval must still be saved
                    # even when the disabling message has already arrived.
                    os.write(write_fd, b'1 10\n0 12\n')
                    self.assertFalse(recorder.submit(packet(2, 9.5)))
                    self.assertFalse(recorder.submit(packet(3, 10.1, [0])))
                    self.assertFalse(recorder.submit(packet(4, 10.2, [1])))
                    source = packet(5, 11., [2])
                    expected = source['frame'].copy()
                    self.assertTrue(recorder.submit(source))
                    source['frame'].fill(99)
                    self.assertTrue(recorder.submit(packet(6, 11.5)))
                    self.assertFalse(recorder.submit(packet(7, 12.1)))
                    os.write(write_fd, b'1 13\n')
                    self.assertTrue(recorder.submit(packet(8, 13.1)))
                    os.close(write_fd)
                    write_fd = None
                    with patch('carmer.missing_tag_frames.time.monotonic', return_value=14.):
                        self.assertFalse(recorder.submit(packet(9, 14.1)))
                finally:
                    recorder.close()
                files = sorted(Path(temp).rglob('*.json'))
                metadata = [json.loads(p.read_text()) for p in files]
                self.assertEqual([p['frame_number'] for p in metadata], [5, 6, 8])
                self.assertEqual(metadata[0]['tags'], [{'id': 2}])
                self.assertEqual(metadata[0]['expected_tag_ids'], [0, 1])
                self.assertEqual(metadata[0]['detector_settings']['quad_decimate'], 2.)
                np.testing.assert_array_equal(cv2.imread(str(files[0].with_suffix('.png'))), expected)
                self.assertEqual((recorder.saved, recorder.failed, recorder.dropped), (3, 0, 0))
        finally:
            os.close(read_fd)
            if write_fd is not None:
                os.close(write_fd)

    def test_slow_or_failed_disk_does_not_block_detection(self):
        entered, release = threading.Event(), threading.Event()
        def slow_write(*args):
            entered.set()
            release.wait(3)
            return False
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()), \
                patch('carmer.missing_tag_frames.cv2.imwrite', side_effect=slow_write):
            recorder = MissingTagFrames(temp)
            try:
                recorder.submit(packet(1, 1.))
                self.assertTrue(entered.wait(1))
                submitter = threading.Thread(target=lambda: [recorder.submit(packet(n, float(n))) for n in range(2, 40)])
                submitter.start()
                submitter.join(timeout=1)
                self.assertFalse(submitter.is_alive())
                self.assertGreater(recorder.dropped, 0)
            finally:
                release.set()
                recorder.close()
            self.assertGreater(recorder.failed, 0)
            self.assertEqual(recorder.saved, 0)

    def test_detection_worker_saves_actual_missing_frame(self):
        class Reader:
            def __init__(self):
                self.items = queue.Queue()
            def latest(self, *args, **kwargs):
                return self.items.get(timeout=3)
            def close(self):
                self.items.put(None)
        cv2.setNumThreads(1)
        visible = np.full((480, 640, 3), 255, np.uint8)
        marker = cv2.aruco.generateImageMarker(
            cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11), 0, 144)
        visible[160:304, 200:344] = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
        missing = np.full_like(visible, 255)
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            reader = Reader()
            recorder = MissingTagFrames(temp)
            worker = app.DetectionWorker(reader, app.make_detector(2.), missing_tag_frames=recorder)
            try:
                for number, frame in enumerate((visible, missing, visible), start=1):
                    reader.items.put((number, frame, 60., float(number)))
                    worker.latest(number-1)
            finally:
                worker.close()
                recorder.close()
            files = list(Path(temp).rglob('*.json'))
            self.assertEqual(len(files), 1)
            self.assertEqual(json.loads(files[0].read_text())['frame_number'], 2)
            np.testing.assert_array_equal(cv2.imread(str(files[0].with_suffix('.png'))), missing)

    def test_cli_requires_directory_for_control_pipe(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            app.parse_args(['--missing-tag-control-stdin'])


if __name__ == '__main__':
    unittest.main()
