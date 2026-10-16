"""Offline checks for RGB node selection and actionable camera open failures."""

import errno
from pathlib import Path
import unittest
from unittest import mock

import cv2
import numpy as np

from carmer import opencv_camera_viewer as camera


class CameraOpenTest(unittest.TestCase):
    @staticmethod
    def sysfs_node(name):
        class Node:
            def __init__(self):
                self.name = name

            def __lt__(self, other):
                return self.name < other.name

            def resolve(self):
                return Path(f'/sys/devices/usb/2-3:1.3/video4linux/{self.name}')
        return Node()

    def test_query_uses_per_node_caps_instead_of_whole_camera_caps(self):
        def query(fd, request, buffer, mutate):
            self.assertEqual(fd, 17)
            self.assertEqual(request, camera._VIDIOC_QUERYCAP)
            self.assertEqual(len(buffer), 104)
            camera._V4L2_CAPABILITY.pack_into(buffer, 0, b'uvcvideo', b'RealSense', b'usb',
                                            0, 0x84a00001, 0x04a00000, 0, 0, 0)
        with mock.patch.object(camera.os, 'open', return_value=17), \
             mock.patch.object(camera.os, 'close') as close, \
             mock.patch.object(camera.fcntl, 'ioctl', side_effect=query):
            self.assertEqual(camera.video_node_capabilities('/dev/video5'), 0x04a00000)
            close.assert_called_once_with(17)

    def test_query_falls_back_to_legacy_caps_and_closes_on_failure(self):
        def query(fd, request, buffer, mutate):
            camera._V4L2_CAPABILITY.pack_into(buffer, 0, b'uvcvideo', b'RealSense', b'usb',
                                            0, 0x04000001, 0, 0, 0, 0)
        with mock.patch.object(camera.os, 'open', return_value=17), \
             mock.patch.object(camera.os, 'close') as close, \
             mock.patch.object(camera.fcntl, 'ioctl', side_effect=query):
            self.assertEqual(camera.video_node_capabilities('/dev/video4'), 0x04000001)
            close.assert_called_once_with(17)
        with mock.patch.object(camera.os, 'open', return_value=17), \
             mock.patch.object(camera.os, 'close') as close, \
             mock.patch.object(camera.fcntl, 'ioctl', side_effect=OSError(errno.ENODEV, 'No such device')):
            with self.assertRaises(OSError):
                camera.video_node_capabilities('/dev/video4')
            close.assert_called_once_with(17)

    def test_auto_skips_metadata_and_accepts_capture_after_renumbering(self):
        cap = mock.Mock()
        cap.isOpened.return_value = True
        cap.get.side_effect = [640., 480., 60.]
        cap.read.return_value = True, np.zeros((480, 640, 3), np.uint8)
        # Reversed numbering proves this doesn't assume an even-numbered image node.
        with mock.patch.object(Path, 'glob', return_value=[self.sysfs_node('video4'), self.sysfs_node('video5')]), \
             mock.patch.object(camera, 'video_node_capabilities', side_effect=[0x04a00000, 0x04200001]), \
             mock.patch.object(cv2, 'VideoCapture', return_value=cap) as capture:
            self.assertIs(camera.open_camera(), cap)
            capture.assert_called_once_with('/dev/video5', cv2.CAP_V4L2)
            cap.release.assert_not_called()

    def test_open_failure_reports_busy_and_skips_metadata(self):
        with mock.patch.object(Path, 'glob', return_value=[self.sysfs_node('video4'), self.sysfs_node('video5')]), \
             mock.patch.object(camera, 'video_node_capabilities',
                               side_effect=[OSError(errno.EBUSY, 'Device or resource busy'), 0x04a00000]), \
             mock.patch.object(cv2, 'VideoCapture') as capture:
            with self.assertRaises(RuntimeError) as raised:
                camera.open_camera()
            message = str(raised.exception)
            self.assertIn('EBUSY', message)
            self.assertIn('metadata', message)
            self.assertIn('sudo fuser -v /dev/video4 /dev/video5', message)
            capture.assert_not_called()

    def test_permission_and_absent_device_errors_are_not_disguised_as_busy(self):
        for code, name in ((errno.EACCES, 'EACCES'), (errno.ENOENT, 'ENOENT')):
            with self.subTest(code=code), \
                 mock.patch.object(camera, 'video_node_capabilities', side_effect=OSError(code, name)), \
                 mock.patch.object(cv2, 'VideoCapture') as capture:
                with self.assertRaisesRegex(RuntimeError, name):
                    camera.open_camera('/dev/video4')
                capture.assert_not_called()

    def test_opencv_initialization_failure_releases_capture_and_gives_debug_hint(self):
        cap = mock.Mock()
        cap.isOpened.return_value = False
        with mock.patch.object(camera, 'video_node_capabilities', return_value=0x04200001), \
             mock.patch.object(cv2, 'VideoCapture', return_value=cap):
            with self.assertRaises(RuntimeError) as raised:
                camera.open_camera('/dev/video4')
            self.assertIn('device is accessible', str(raised.exception))
            self.assertIn('OPENCV_VIDEOIO_DEBUG=1', str(raised.exception))
            cap.release.assert_called_once()


if __name__ == '__main__':
    unittest.main()
