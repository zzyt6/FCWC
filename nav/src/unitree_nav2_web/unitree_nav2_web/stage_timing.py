"""Rolling wall-time statistics for periodic timing logs.

Only measures and formats durations; never changes the measured code path.
"""
from contextlib import contextmanager
import threading
import time

import numpy as np


class StageTimer:
    """Per-tick sequential stages (start/mark/finish) plus independent samples (record).

    Durations are perf_counter wall times in milliseconds. A report summarizes
    and clears the samples collected since the previous report.
    """

    def __init__(self, title):
        self.title = title
        self.lock = threading.Lock()
        self.samples = {}
        self.window_started = time.monotonic()
        self._tick = None
        self._last = None
        self._tick_started = None

    def start(self):
        self._tick = {}
        self._tick_started = self._last = time.perf_counter()

    def mark(self, stage):
        """Charge the time since the previous mark to this stage."""
        if self._tick is None:
            return
        now = time.perf_counter()
        self._tick[stage] = self._tick.get(stage, 0.) + (now - self._last) * 1000.
        self._last = now

    def value(self, name, ms):
        """Attach an already measured value (e.g. nested ONNX time, data age) to this tick."""
        if self._tick is not None and ms is not None:
            self._tick[name] = float(ms)

    def finish(self, total='step'):
        if self._tick is None:
            return
        self._tick[total] = (time.perf_counter() - self._tick_started) * 1000.
        tick, self._tick = self._tick, None
        with self.lock:
            for name, ms in tick.items():
                self.samples.setdefault(name, []).append(ms)

    def record(self, name, ms):
        with self.lock:
            self.samples.setdefault(name, []).append(float(ms))

    @contextmanager
    def measure(self, name):
        started = time.perf_counter()
        try:
            yield
        finally:
            self.record(name, (time.perf_counter() - started) * 1000.)

    def report(self):
        now = time.monotonic()
        with self.lock:
            samples, self.samples = self.samples, {}
            window, self.window_started = now - self.window_started, now
        if not samples:
            return None
        fields = []
        for name, values in samples.items():
            values = np.asarray(values)
            fields.append('%s=%d/%.2f/%.2f/%.2f' % (name, len(values), values.mean(),
                                                    np.percentile(values, 95), values.max()))
        return 'TIMING %s (%.1f s, n/mean/p95/max ms): %s' % (self.title, window, ' '.join(fields))
