"""Push-cart episode tag results, without retaining or encoding camera images.

The detection thread enqueues scalar results for camera_detections.csv.
Episodes follow Viser/core/data_episode.py.
"""
import csv
import math
from pathlib import Path
import queue
import sys
import threading

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'Viser'))
from core.data_episode import EpisodeRecorder  # noqa: E402


def _number(value, digits=7):
    return '' if value is None or not math.isfinite(value) else '%.*g' % (digits, value)


class EpisodeDetections(EpisodeRecorder):
    def __init__(self, port):
        super().__init__('camera', port)
        # Only scalar detection results enter this queue; images stay in the detector.
        self.pending = queue.Queue(maxsize=120)
        self.stream = self.detections = None
        self.stopping = threading.Event()
        self.error = ''
        self.thread = threading.Thread(target=self._run, daemon=True, name='episode-detections')
        self.thread.start()

    # Detection thread: one nonblocking put per processed frame.
    def detection(self, packet):
        heading = packet.get('raw_fused_cart_heading') or {}
        compensation = packet.get('pelvis_compensation') or {}
        yaw = heading.get('yaw_rad')
        row = (packet['frame_number'], ';'.join(str(tag['id']) for tag in packet['tags']),
               bool(heading.get('valid')) and compensation.get('valid', True) is True,
               yaw if isinstance(yaw, (int, float)) else None, heading.get('reason', ''),
               packet.get('detection_ms'), packet.get('processing_ms'))
        try:
            self.pending.put_nowait((packet['frame_timestamp_monotonic'], row))
        except queue.Full:
            self.dropped += 1

    def _run(self):
        while not self.stopping.is_set() or not self.pending.empty():
            try:
                t, sample = self.pending.get(timeout=.02)
                self.feed(t, sample)
            except queue.Empty:
                pass
            except Exception as exc:
                self._fail(exc)
            try:
                self.service()
            except Exception as exc:
                self._fail(exc)

    def _fail(self, exc):
        if repr(exc) != self.error:
            self.error = repr(exc)
            print('PUSHCART DATA camera detections warning: ' + self.error, flush=True)

    def open_files(self, directory):
        self.stream = (directory / 'camera_detections.csv').open('x', newline='', buffering=1 << 16)
        self.detections = csv.writer(self.stream)
        self.detections.writerow(('camera_frame_count', 't', 't_rel', 'tag_ids', 'cart_yaw_valid',
                                  'cart_yaw_pelvis_rad', 'reason', 'detection_ms', 'processing_ms'))
        print('PUSHCART DATA camera detections recording %s' % directory, flush=True)

    def write(self, t, sample):
        count, ids, valid, yaw, reason, detection_ms, processing_ms = sample
        start = self.episode['start']
        self.detections.writerow((count, '%.6f' % t, '%.6f' % (t - start), ids, int(valid),
                                  _number(yaw), reason, _number(detection_ms, 4), _number(processing_ms, 4)))

    def close_files(self):
        if self.stream is not None:
            self.stream.close()
        self.stream = self.detections = None

    def close(self):
        """Drain queued results and close the detection CSV."""
        self.stopping.set()
        self.thread.join(timeout=10)
        super().close()
