"""Push-cart episodes retain timestamped tag results without recording images."""

import csv
import json
import math
from pathlib import Path
import socket
import tempfile
import time
import unittest
from carmer.episode_detections import EpisodeDetections
from core.data_episode import encode


class EpisodeDetectionsTest(unittest.TestCase):
    def test_episode_keeps_detection_window_without_video_files(self):
        with tempfile.TemporaryDirectory(prefix='episode_detections_') as directory:
            recorder = EpisodeDetections(0)
            port = recorder.sock.getsockname()[1]
            sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                base = time.monotonic()
                def push(first, last):
                    for count in range(first, last, 3):
                        captured = base + (count + .5) / 60  # off the exact window edges
                        recorder.detection(dict(frame_number=count, tags=[dict(id=0), dict(id=1)],
                                                frame_timestamp_monotonic=captured,
                                                raw_fused_cart_heading=dict(valid=True, yaw_rad=.1, reason='ok'),
                                                pelvis_compensation=dict(valid=True), detection_ms=2.,
                                                processing_ms=3.))
                        deadline = time.monotonic() + 2.
                        while not recorder.pending.empty() and time.monotonic() < deadline:
                            time.sleep(.001)
                        self.assertTrue(recorder.pending.empty())

                def control(seq, stop=math.nan, episode=True):
                    sender.sendto(encode(seq, directory if episode else None, base + 2, stop), ('127.0.0.1', port))
                    time.sleep(.3)

                push(0, 120)              # 0 .. 2 s: pre-roll buffer
                control(1)                # start = 2 s
                push(120, 240)            # 2 .. 4 s
                control(2, stop=base + 3)  # stop = 3 s, post-roll to 4 s
                push(240, 300)            # after the window: not written
                time.sleep(.3)
                control(3, episode=False)
            finally:
                recorder.close()
                sender.close()
            folder = Path(directory)
            self.assertEqual({path.name for path in folder.iterdir()},
                             {'camera_detections.csv', 'camera_summary.json'})
            with (folder / 'camera_detections.csv').open() as stream:
                detections = list(csv.DictReader(stream))
            counts = [int(row['camera_frame_count']) for row in detections]
            self.assertEqual(counts, list(range(60, 240, 3)))  # [start - 1 s, stop + 1 s]
            self.assertAlmostEqual(float(detections[0]['t_rel']), -1 + .5 / 60, places=5)
            self.assertEqual(detections[0]['tag_ids'], '0;1')
            self.assertTrue(all(row['cart_yaw_valid'] == '1' and row['cart_yaw_pelvis_rad'] == '0.1'
                                for row in detections))
            summary = json.loads((folder / 'camera_summary.json').read_text())
            self.assertEqual((summary['stream'], summary['close_reason'], summary['rows']),
                             ('camera', 'stopped', len(detections)))
            self.assertEqual(summary['dropped_before_write'], 0)


if __name__ == '__main__':
    unittest.main()
