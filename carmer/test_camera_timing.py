"""Driver timestamps must survive read/USB delays at 50 Hz."""

import unittest
from unittest.mock import patch

import numpy as np

from carmer import opencv_camera_viewer as camera


class Capture:
    def __init__(self, timestamps):
        self.timestamps = iter(timestamps)
        self.stamp = None
        self.released = False

    def read(self):
        self.stamp = next(self.timestamps)
        return True, np.zeros((camera.HEIGHT, camera.WIDTH, 3), dtype=np.uint8)

    def get(self, _):
        return self.stamp * 1000.

    def release(self):
        self.released = True


class CameraTimingTest(unittest.TestCase):
    def test_driver_time_survives_variable_read_delay_at_50hz(self):
        capture = Capture([100., 100.020, 100.040])
        with patch.object(camera, 'WARMUP', 0), patch.object(
                camera.time, 'monotonic', side_effect=[100.017, 100.038, 100.087]):
            reader = camera.CameraReader(capture, 3)
            reader.thread.join(timeout=2.)
            self.assertFalse(reader.thread.is_alive())
            number, _, _, stamp = reader.latest(0, with_timestamp=True)
        self.assertEqual(number, 3)
        self.assertAlmostEqual(stamp, 100.040)
        self.assertTrue(capture.released)

    def test_bad_clock_does_not_turn_into_a_fresh_detection(self):
        for timestamps in ([0.], [1000.], [100., 100.]):
            with self.subTest(timestamps=timestamps):
                capture = Capture(timestamps)
                with patch.object(camera, 'WARMUP', 0), patch.object(
                        camera.time, 'monotonic', return_value=100.020):
                    reader = camera.CameraReader(capture, len(timestamps))
                    reader.thread.join(timeout=2.)
                    with self.assertRaisesRegex(RuntimeError, 'V4L2 frame timestamp'):
                        reader.latest(0, with_timestamp=True)
                self.assertTrue(capture.released)


if __name__ == '__main__':
    unittest.main()
