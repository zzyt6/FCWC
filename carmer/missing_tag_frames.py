"""Save the source images of missing cart tags without blocking detection."""
from collections import deque
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import queue
import threading
import time

import cv2


class MissingTagFrames:
    def __init__(self, directory, tag_ids=(0, 1), *, control_fd=None):
        session = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
        self.directory = Path(directory) / f'{session}_{os.getpid()}'
        self.tag_ids = frozenset(tag_ids)
        self.control_fd = control_fd
        self.control_buffer = b''
        self.transitions = deque(maxlen=64)
        self.control_ended = False
        if control_fd is not None:
            os.set_blocking(control_fd, False)
        self.pending = queue.Queue(maxsize=32)
        self.stopping = threading.Event()
        self.saved = self.dropped = self.failed = 0
        self.thread = threading.Thread(target=self._write, daemon=True, name='missing-tag-images')
        self.thread.start()

    def _active_at(self, stamp):
        if self.control_fd is None:
            return True
        if not self.control_ended:
            for _ in range(8):
                try:
                    data = os.read(self.control_fd, 4096)
                except BlockingIOError:
                    break
                except OSError:
                    data = b''
                if not data:
                    self.control_ended = True
                    self.transitions.append((time.monotonic(), False))
                    break
                self.control_buffer += data
                while b'\n' in self.control_buffer:
                    line, self.control_buffer = self.control_buffer.split(b'\n', 1)
                    try:
                        enabled, timestamp = line.split()
                        timestamp = float(timestamp)
                        if (enabled in (b'0', b'1') and math.isfinite(timestamp)
                                and (not self.transitions or timestamp >= self.transitions[-1][0])):
                            self.transitions.append((timestamp, enabled == b'1'))
                    except ValueError:
                        pass
                if len(self.control_buffer) > 16384:
                    self.control_buffer = b''
        # A frame captured before a stop still belongs to that push-cart run,
        # even if the stop reaches us while its detection is being processed.
        return next((enabled for timestamp, enabled in reversed(self.transitions)
                     if timestamp <= stamp), False)

    def submit(self, packet):
        active = self._active_at(packet['frame_timestamp_monotonic'])
        if not active or any(tag['id'] in self.tag_ids for tag in packet['tags']):
            return False
        metadata = {key: value for key, value in packet.items() if key != 'frame'}
        metadata.update(save_reason='no_cart_tag', expected_tag_ids=sorted(self.tag_ids),
                        saved_at_utc=datetime.now(timezone.utc).isoformat())
        try:
            self.pending.put_nowait((packet['frame'].copy(), metadata))
            return True
        except queue.Full:
            self.dropped += 1
            return False

    def _write(self):
        reported_drops = 0
        while not self.stopping.is_set() or not self.pending.empty():
            if self.dropped != reported_drops:
                reported_drops = self.dropped
                print(f'MISSING_TAG warning: image queue full; dropped={reported_drops}', flush=True)
            try:
                frame, metadata = self.pending.get(timeout=.1)
            except queue.Empty:
                continue
            try:
                self.directory.mkdir(parents=True, exist_ok=True)
                stamp_us = round(metadata['frame_timestamp_monotonic'] * 1e6)
                stem = f"frame_{metadata['frame_number']:09d}_mono_{stamp_us}"
                image_path = self.directory / f'{stem}.png'
                if not cv2.imwrite(str(image_path), frame, [cv2.IMWRITE_PNG_COMPRESSION, 1]):
                    raise OSError(f'Cannot write {image_path}')
                (self.directory / f'{stem}.json').write_text(
                    json.dumps(metadata, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
                self.saved += 1
                print(f'MISSING_TAG saved: {image_path}', flush=True)
            except Exception as exc:
                self.failed += 1
                print(f'MISSING_TAG warning: save failed: {exc}', flush=True)
            finally:
                self.pending.task_done()

    def close(self):
        self.stopping.set()
        self.thread.join(timeout=5)
        print(f'MISSING_TAG summary: saved={self.saved} dropped={self.dropped} '
              f'failed={self.failed} pending={self.pending.unfinished_tasks} directory={self.directory}', flush=True)
