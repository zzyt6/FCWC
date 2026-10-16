"""Optional push-cart feature loaded by the normal Viser entry point."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading

WORKSPACE = Path(__file__).resolve().parents[2]
CONFIG = WORKSPACE / 'G1_high_policy_deploy/robot/config/viser_v81.json'


class CameraProcess:
    """Own only the read-only camera process explicitly started from this UI."""
    def __init__(self, port, mode, interface, tag_ids=(0, 1)):
        self.port, self.mode, self.interface = port, mode, interface
        self.tag_ids = tuple(tag_ids)
        self.process = None
        self.reader = None
        self.lines = deque(maxlen=8)
        self.lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='cart-camera-start')
        self.start_future = None

    def request_start(self):
        """Launch away from the 50 Hz control thread, including first camera setup."""
        if self.mode == 'sim':
            raise ValueError('离线联调使用合成相机数据，不启动真机相机')
        if self.start_future is not None and not self.start_future.done():
            return
        if self.process is not None and self.process.poll() is None:
            return
        self.start_future = self.executor.submit(self.start)

    def start(self):
        if self.mode == 'sim':
            raise ValueError('离线联调使用合成相机数据，不启动真机相机')
        if self.process is not None and self.process.poll() is None:
            return
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
        for name in ('CYCLONEDDS_URI', 'RMW_IMPLEMENTATION', 'PYTHONPATH', 'AMENT_PREFIX_PATH', 'LD_LIBRARY_PATH'):
            env.pop(name, None)
        omp = Path(sys.prefix) / 'lib/libgomp.so.1'
        if omp.is_file():
            env['LD_PRELOAD'] = str(omp) + (':' + env['LD_PRELOAD'] if env.get('LD_PRELOAD') else '')
        with self.lock:
            self.lines.clear()
        self.process = subprocess.Popen(
            [sys.executable, '-B', str(WORKSPACE / 'carmer/apriltag_detector.py'),
             '--no-display', '--heading-udp-port', str(self.port), '--network-interface', self.interface,
             '--cart-tag-ids', *map(str, self.tag_ids)],
            # Explicit CLI IDs also override any broader standalone camera config.
            env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
        self.reader = threading.Thread(target=self._read, args=(self.process,), daemon=True)
        self.reader.start()

    def _read(self, process):
        with process.stdout:
            for line in process.stdout:
                with self.lock:
                    self.lines.append(line.strip()[:400])

    def snapshot(self):
        code = self.process.poll() if self.process is not None else None
        with self.lock:
            message = '\n'.join(self.lines) if code is not None else ''
        starting = self.start_future is not None and not self.start_future.done()
        if self.start_future is not None and self.start_future.done() and not self.start_future.cancelled():
            error = self.start_future.exception()
            if error is not None:
                message, code = str(error), -1
        return dict(running=self.process is not None and code is None, starting=starting,
                    exit_code=code, error=message if code else '')

    def close(self):
        self.executor.shutdown(wait=True, cancel_futures=True)
        if self.process is not None and self.process.poll() is None:
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(self.process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                self.process.wait(timeout=2)
        if self.reader is not None:
            self.reader.join(timeout=1)


def load_feature(config, mode, interface, path=CONFIG):
    """Feature failure must not remove normal Balance/Homie operation."""
    config.pushcart_error = ''
    try:
        sys.path.insert(0, str(WORKSPACE / 'G1_high_policy_deploy/robot'))
        from pushcart_v81.viser_policy import ViserPushPolicy
        high = ViserPushPolicy(config, path)
        high.camera_process = CameraProcess(high.settings['camera']['udp_port'], mode, interface,
                                            high.settings['camera']['tag_ids'])
        config.high_policy_metadata = high.metadata
        return high
    except Exception as exc:
        config.pushcart_error = str(exc)
        return None
