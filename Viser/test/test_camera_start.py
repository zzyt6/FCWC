"""A slow camera launch must not block body control or duplicate processes."""
from pathlib import Path
import os
import sys
import threading
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.pushcart import CameraProcess


class CameraStartTests(unittest.TestCase):
    def test_navigation_reuses_camera_with_urdf_pose_and_real_tag_size(self):
        for publish in (False, True):
            with self.subTest(publish=publish):
                camera = CameraProcess(9870, 'control', 'eth0')
                camera.publish_cart_pose = publish
                camera.quad_decimate = 1.5
                camera.missing_tag_dir = Path('/tmp/navigation-missing-tags') if publish else None
                # Navigation pins the camera away from the policy/bridge cores.
                camera.cpus = {4, 5} if publish else None
                process = Mock()
                process.poll.return_value = None
                try:
                    with patch('core.pushcart.subprocess.Popen', return_value=process) as launch, \
                            patch('core.pushcart.threading.Thread'), patch('core.pushcart.os.set_blocking'):
                        camera.start()
                        camera.start()
                        launch.assert_called_once()
                        command = launch.call_args.args[0]
                        self.assertEqual(command[:3] == ['taskset', '-c', '4,5'], publish)
                        # One OpenCV detection thread per pinned core; Viser keeps the default.
                        self.assertEqual('--opencv-threads' in command, publish)
                        if publish:
                            self.assertEqual(command[command.index('--opencv-threads')+1], '2')
                        self.assertEqual('--publish-cart-pose' in command, publish)
                        self.assertEqual(command[command.index('--cart-tag-ids')+1:][:2], ['0', '1'])
                        if publish:
                            self.assertEqual(command[command.index('--pelvis-attitude-source')+1], 'imu')
                            self.assertEqual(command[command.index('--tag-size')+1], '0.10')
                            self.assertEqual(command[command.index('--quad-decimate')+1], '1.5')
                            self.assertTrue(command[command.index('--robot-urdf')+1].endswith('/G1_URDF/g1.urdf'))
                            self.assertEqual(command[command.index('--missing-tag-dir')+1], str(camera.missing_tag_dir))
                            self.assertIn('--missing-tag-control-stdin', command)
                        else:
                            self.assertNotIn('--missing-tag-dir', command)
                finally:
                    process.poll.return_value = 0
                    camera.close()

    def test_recording_pipe_transitions_restart_and_failure(self):
        camera = CameraProcess(9870, 'control', 'eth0')
        camera.missing_tag_dir = Path('/tmp/navigation-missing-tags')
        read_fd, write_fd = os.pipe()
        process = Mock()
        process.poll.return_value = None
        process.stdin = os.fdopen(write_fd, 'wb', buffering=0)
        camera.process = process
        try:
            camera.record_missing_tags(False, 10.)
            camera.record_missing_tags(False, 10.1)
            camera.record_missing_tags(True, 11.)
            camera.record_missing_tags(False, 12.)
            self.assertEqual(os.read(read_fd, 4096).decode().splitlines(),
                             ['0 10.000000000', '1 11.000000000', '0 12.000000000'])
            replacement = Mock()
            replacement.poll.return_value = None
            replacement.stdin = process.stdin
            camera.process = replacement
            camera.record_missing_tags(False, 13.)
            self.assertEqual(os.read(read_fd, 4096), b'0 13.000000000\n')
            with patch('core.pushcart.os.write', side_effect=BrokenPipeError('camera stopped')):
                camera.record_missing_tags(True, 14.)
            self.assertIn('camera stopped', camera.snapshot()['missing_tag_error'])
        finally:
            camera.process.poll.return_value = 0
            camera.close()
            os.close(read_fd)

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
