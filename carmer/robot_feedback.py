"""Read-only G1 feedback, buffered on the same host monotonic clock as images.

Reception times are proxies for acquisition times. The configurable camera delay
does not constitute hardware synchronization. No robot command API is imported.
"""
from bisect import bisect_left
from collections import deque
import json
from pathlib import Path
import subprocess
import threading

import numpy as np


def unit_quaternion(value):
    q = np.asarray(value, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all():
        return None
    norm = np.linalg.norm(q)
    return q / norm if .8 < norm < 1.2 else None


def interpolate_quaternion(a, b, weight):
    a, b = unit_quaternion(a), unit_quaternion(b)
    if a is None or b is None:
        return None
    dot = float(a @ b)
    if dot < 0:
        b, dot = -b, -dot
    if dot > .9995:
        result = (1 - weight) * a + weight * b
        return result / np.linalg.norm(result)
    angle = np.arccos(np.clip(dot, -1, 1))
    return (np.sin((1 - weight) * angle) * a + np.sin(weight * angle) * b) / np.sin(angle)


class StateBuffer:
    def __init__(self):
        self.samples = deque(maxlen=1000)
        self.lock = threading.Lock()

    def add(self, packet):
        try:
            stamp = int(packet['received_ns']) * 1e-9
            tick = int(packet['tick'])
            q = np.asarray(packet['q'], dtype=float)
            if not np.isfinite(stamp) or stamp <= 0 or q.shape != (15,):
                return False
            sample = dict(time=stamp, tick=tick, q=q,
                          quaternion=unit_quaternion(packet['quaternion']),
                          mode_pr=int(packet['mode_pr']))
        except (KeyError, TypeError, ValueError, OverflowError):
            return False
        with self.lock:
            if self.samples:
                last = self.samples[-1]
                delta = (tick - last['tick']) & 0xffffffff
                if stamp <= last['time'] or not 0 < delta < 0x80000000:
                    return False
            self.samples.append(sample)
        return True

    def at(self, target, now, timeout, tolerance):
        meta = dict(valid=False, reason='robot_state_missing', target_time=target,
                    time_basis='host_monotonic_receive_proxy', state_age_ms=None,
                    sync_error_ms=None)
        with self.lock:
            samples = list(self.samples)
        if not samples:
            return None, meta
        age = now - samples[-1]['time']
        meta['state_age_ms'] = age * 1000
        if age < 0 or age > timeout:
            meta['reason'] = 'robot_state_stale' if age >= 0 else 'robot_state_clock_error'
            return None, meta
        i = bisect_left([sample['time'] for sample in samples], target)
        candidates = samples[max(0, i - 1):min(len(samples), i + 1)]
        closest = min(candidates, key=lambda sample: abs(sample['time'] - target))
        error = abs(closest['time'] - target)
        meta.update(sync_error_ms=error * 1000, tick=closest['tick'],
                    state_time=closest['time'], alignment='nearest')
        if error > tolerance:
            meta['reason'] = 'image_state_time_mismatch'
            return None, meta
        result = dict(closest)
        if len(candidates) == 2:
            a, b = candidates
            if (a['time'] < target < b['time']
                    and max(target - a['time'], b['time'] - target) <= tolerance
                    and a['mode_pr'] == b['mode_pr']):
                weight = (target - a['time']) / (b['time'] - a['time'])
                result['q'] = (1 - weight) * a['q'] + weight * b['q']
                result['quaternion'] = interpolate_quaternion(a['quaternion'], b['quaternion'], weight)
                result['time'] = target
                meta.update(alignment='interpolated', state_time=target,
                            bracket_times=[a['time'], b['time']])
        meta.update(valid=True, reason='ok')
        return result, meta


def ensure_bridge():
    folder = Path(__file__).resolve().parent
    build = folder / '.build'
    executable = build / 'robot_state_bridge'
    sources = [folder / 'robot_state_bridge.cpp', folder / 'CMakeLists.txt']
    if not executable.exists() or any(p.stat().st_mtime > executable.stat().st_mtime for p in sources):
        print('Building read-only robot feedback subscriber...', flush=True)
        try:
            subprocess.run(['cmake', '-S', str(folder), '-B', str(build),
                            '-DCMAKE_BUILD_TYPE=Release'], check=True)
            subprocess.run(['cmake', '--build', str(build), '-j2'], check=True)
        except (OSError, subprocess.CalledProcessError) as exc:
            raise RuntimeError('Cannot build robot feedback subscriber; install cmake/g++ and '
                               'keep the bundled G1Deploy_BalanceHomie/vendor/unitree_sdk2') from exc
    return executable


class RobotFeedback:
    """Own the subscription subprocess; closing it never changes robot state."""
    def __init__(self, interface):
        self.buffer = StateBuffer()
        self.error = None
        self.closed = False
        self.process = subprocess.Popen([str(ensure_bridge()), interface],
                                        stdout=subprocess.PIPE, text=True, bufsize=1)
        self.thread = threading.Thread(target=self._read, daemon=True, name='robot-feedback')
        self.thread.start()

    def _read(self):
        try:
            for line in self.process.stdout:
                if not line.startswith('{'):
                    continue  # SDK startup diagnostics are not feedback.
                try:
                    self.buffer.add(json.loads(line))
                except ValueError:
                    pass  # Never refresh timestamps with a malformed packet.
        finally:
            if not self.closed:
                self.error = 'robot_feedback_bridge_exited'

    def at(self, *args, **kwargs):
        if self.error:
            return None, dict(valid=False, reason=self.error)
        return self.buffer.at(*args, **kwargs)

    def close(self):
        self.closed = True
        if self.process.poll() is None:
            self.process.terminate()
        try:
            self.process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=2)
        self.thread.join(timeout=2)
        self.process.stdout.close()
