"""Push-cart ROS CSVs written by a thread in the existing navigation web bridge.

Callbacks enqueue their received messages and timestamps. Formatting, episode
coordination and disk I/O stay on the writer thread; no ROS node or process is
created here. The base odometry callback is shared with the navigation UI.
"""
import csv
import math
from pathlib import Path
import queue
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / 'Viser'))
from core.data_episode import CONTROL_PORTS, EpisodeRecorder  # noqa: E402

ODOMETRY = ('t', 't_rel', 'wall_t', 'stamp_t', 'frame_id', 'child_frame_id', 'x', 'y', 'z',
            'qw', 'qx', 'qy', 'qz', 'yaw', 'vx', 'vy', 'vz', 'wx', 'wy', 'wz')
TWIST = ('t', 't_rel', 'wall_t', 'vx', 'vy', 'vz', 'wx', 'wy', 'wz')
FILES = {'odometry_fastlio.csv': ODOMETRY, 'odometry_base.csv': ODOMETRY, 'cmd_vel.csv': TWIST}
TOPICS = {'/Odometry': 'odometry_fastlio.csv', '/Odometry_base': 'odometry_base.csv',
          '/cmd_vel': 'cmd_vel.csv'}


class RosEpisodeRecorder(EpisodeRecorder):
    def __init__(self, port=CONTROL_PORTS['ros']):
        super().__init__('ros', port)
        self.streams, self.writers, self.file_rows = {}, {}, {}
        self.pending = queue.Queue(maxsize=512)
        self.stopping = threading.Event()
        self.error = ''
        self.thread = threading.Thread(target=self._run, name='ros-episode-data', daemon=True)
        self.thread.start()

    def odometry(self, name, message):
        self._enqueue(name, message)

    def cmd_vel(self, message):
        self._enqueue('cmd_vel.csv', message)

    def _enqueue(self, name, message):
        # rclpy supplies a new message per callback. The UI only reads it;
        # retain that exact message and its callback time, not a later sample.
        if self.stopping.is_set():
            return
        sample = (time.monotonic(), (name, time.time(), message))
        try:
            self.pending.put_nowait(sample)
        except queue.Full:
            self.dropped += 1

    def _fail(self, exc):
        if repr(exc) != self.error:
            self.error = repr(exc)
            print('PUSHCART DATA ros warning: ' + self.error, flush=True)

    def _run(self):
        last_flush = time.monotonic()
        try:
            while not self.stopping.is_set() or not self.pending.empty():
                try:
                    item = self.pending.get(timeout=.02)
                except queue.Empty:
                    item = None
                try:
                    if item is not None:
                        self.feed(*item)
                    self.service()
                    now = time.monotonic()
                    if now - last_flush >= .5:
                        for stream in self.streams.values():
                            stream.flush()
                        last_flush = now
                except Exception as exc:
                    self._fail(exc)
        finally:
            super().close()

    def open_files(self, directory):
        self.file_rows = dict.fromkeys(FILES, 0)
        try:
            for name, header in FILES.items():
                self.streams[name] = (directory / name).open('x', newline='', buffering=1 << 16)
                self.writers[name] = csv.writer(self.streams[name])
                self.writers[name].writerow(header)
        except Exception:
            self.close_files()
            raise
        print('PUSHCART DATA ros recording %s (navigation_web_bridge thread)' % directory, flush=True)

    def write(self, t, sample):
        name, wall, message = sample
        if name == 'cmd_vel.csv':
            v, w = message.linear, message.angular
            row = ('%.6f' % wall, *('%.7g' % value for value in (v.x, v.y, v.z, w.x, w.y, w.z)))
        else:
            p, q = message.pose.pose.position, message.pose.pose.orientation
            v, w = message.twist.twist.linear, message.twist.twist.angular
            yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
            stamp = message.header.stamp.sec + message.header.stamp.nanosec * 1e-9
            row = ('%.6f' % wall, '%.6f' % stamp, message.header.frame_id, message.child_frame_id,
                   *('%.7g' % value for value in (p.x, p.y, p.z, q.w, q.x, q.y, q.z, yaw,
                                                 v.x, v.y, v.z, w.x, w.y, w.z)))
        self.writers[name].writerow(('%.6f' % t, '%.6f' % (t - self.episode['start'])) + row)
        self.file_rows[name] += 1

    def close_files(self):
        for stream in self.streams.values():
            stream.close()
        self.streams, self.writers = {}, {}
        return dict(source='navigation_web_bridge', files=list(FILES), file_rows=dict(self.file_rows),
                    topics=dict(TOPICS), writer_error=self.error,
                    note='/Odometry (FAST-LIO) twist is always zero; /Odometry_base carries body-frame planar velocity')

    def close(self):
        self.stopping.set()
        self.thread.join(timeout=10)
        if self.thread.is_alive():
            self._fail(RuntimeError('ROS data writer did not finish before shutdown timeout'))
