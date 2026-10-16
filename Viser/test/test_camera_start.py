"""A slow camera launch must not block body control or duplicate processes."""
from pathlib import Path
import sys
import threading
import unittest
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.pushcart import CameraProcess


class CameraStartTests(unittest.TestCase):
    def test_failed_launch_is_reported_without_raising_in_request(self):
        camera = CameraProcess(9870, 'control', 'eth0')
        camera.start = Mock(side_effect=RuntimeError('camera busy'))
        try:
            camera.request_start()
            with self.assertRaises(RuntimeError):
                camera.start_future.result(timeout=1)
            self.assertEqual(camera.snapshot()['error'], 'camera busy')
            self.assertFalse(camera.snapshot()['starting'])
        finally:
            camera.close()

    def test_launch_is_async_and_only_one_request_runs(self):
        camera = CameraProcess(9870, 'control', 'eth0')
        started, release = threading.Event(), threading.Event()
        def slow_start():
            started.set()
            release.wait(2)
        camera.start = Mock(side_effect=slow_start)
        try:
            camera.request_start()
            self.assertTrue(started.wait(1))
            self.assertTrue(camera.snapshot()['starting'])
            camera.request_start()
            self.assertEqual(camera.start.call_count, 1)
        finally:
            release.set()
            camera.close()


if __name__ == '__main__':
    unittest.main()
