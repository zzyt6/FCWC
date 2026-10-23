"""Optional push-cart feature loaded by the normal Viser entry point."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

WORKSPACE = Path(__file__).resolve().parents[2]
CONFIG = WORKSPACE / 'G1_high_policy_deploy/robot/config/viser_v81.json'


class CameraProcess:
    """Own only the read-only camera process explicitly started from this UI."""
    def __init__(self, port, mode, interface, tag_ids=(0, 1)):
        self.port, self.mode, self.interface = port, mode, interface
        self.tag_ids = tuple(tag_ids)
        self.publish_cart_pose = False  # Navigation opts in before starting the shared camera.
        self.quad_decimate = 2.
        # Optional CPU set for the camera and its feedback subprocess. Navigation
        # keeps it off the cores of the 50 Hz policy and 200 Hz bridge; Viser
        # leaves it None (unchanged scheduling).
        self.cpus = None
        self.missing_tag_dir = None
        self.missing_tag_error = ''
        # Push-cart data episodes record scalar tag results in the camera
        # process; images are never sent to the data recorder.
        self.data_control_port = None
        self._recording_process = self._recording_enabled = None
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
        command = [sys.executable, '-B', str(WORKSPACE / 'carmer/apriltag_detector.py'),
                   '--no-display', '--heading-udp-port', str(self.port), '--network-interface', self.interface,
                   '--cart-tag-ids', *map(str, self.tag_ids)]
        if self.publish_cart_pose:
            # --profile only adds per-stage wall times (CAMERA TIMING lines).
            command += ['--publish-cart-pose', '--pelvis-attitude-source', 'imu', '--profile',
                        '--quad-decimate', f'{self.quad_decimate:g}',
                        '--tag-size', '0.10', '--robot-urdf', str(WORKSPACE / 'G1_URDF/g1.urdf')]
        if self.missing_tag_dir is not None:
            command += ['--missing-tag-dir', str(self.missing_tag_dir), '--missing-tag-control-stdin']
        if self.data_control_port:
            command += ['--data-control-port', str(self.data_control_port)]
        if self.cpus:
            # Children (the robot-state bridge) inherit this affinity. Detection
            # uses one OpenCV thread per pinned core (identical results).
            command += ['--opencv-threads', str(len(self.cpus))]
            command = ['taskset', '-c', ','.join(map(str, sorted(self.cpus)))] + command
        process = subprocess.Popen(
            command,
            # Explicit CLI IDs also override any broader standalone camera config.
            env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True,
            stdin=subprocess.PIPE if self.missing_tag_dir is not None else None)
        if self.missing_tag_dir is not None:
            os.set_blocking(process.stdin.fileno(), False)
        self.process = process
        self.reader = threading.Thread(target=self._read, args=(self.process,), daemon=True)
        self.reader.start()

    def record_missing_tags(self, enabled, now):
        """Nonblocking capture-time gate; called only by the navigation executor."""
        process = self.process
        if self.missing_tag_dir is None or process is None or process.poll() is not None:
            return
        enabled = bool(enabled)
        if process is self._recording_process and enabled == self._recording_enabled:
            return
        message = f'{int(enabled)} {now:.9f}\n'.encode('ascii')
        try:
            if os.write(process.stdin.fileno(), message) != len(message):
                raise OSError('Incomplete camera recording control message')
        except BlockingIOError:
            return  # Retry on the next executor tick; never block body control.
        except (OSError, ValueError) as exc:
            error = str(exc)
            if error != self.missing_tag_error:
                print('MISSING_TAG warning: recording control failed: ' + error, flush=True)
            self.missing_tag_error = error
            return
        self.missing_tag_error = ''
        self._recording_process, self._recording_enabled = process, enabled

    def _read(self, process):
        last_camera_report = last_profile_report = -float('inf')
        with process.stdout:
            for line in process.stdout:
                with self.lock:
                    self.lines.append(line.strip()[:400])
                if line.startswith('MISSING_TAG '):
                    print(line.rstrip(), flush=True)
                elif line.startswith('Capture ') and time.monotonic()-last_camera_report >= 1.:
                    # Keep the camera's throttled timing/tag-loss reports in
                    # the launch log, including frames without usable heading.
                    print('CAMERA ' + line.rstrip(), flush=True)
                    last_camera_report = time.monotonic()
                elif line.startswith('Profile detect') and time.monotonic()-last_profile_report >= 1.:
                    # Navigation runs the camera with --profile: per-stage wall times.
                    print('CAMERA TIMING ' + line.rstrip(), flush=True)
                    last_profile_report = time.monotonic()
                elif line.startswith('Profile capture') and time.monotonic()-last_profile_report < .1:
                    print('CAMERA TIMING ' + line.rstrip(), flush=True)
                elif line.startswith(('ERROR', 'Traceback')) or 'Error' in line:
                    # A camera exit otherwise only shows as growing frame age.
                    print('CAMERA ' + line.rstrip(), flush=True)
        try:
            code = process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            code = None
        print(f'CAMERA process exited code={code}', flush=True)

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
                    exit_code=code, error=message if code else '',
                    missing_tag_directory=str(self.missing_tag_dir) if self.missing_tag_dir is not None else None,
                    missing_tag_error=self.missing_tag_error)

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
        if self.process is not None and self.process.stdin is not None:
            self.process.stdin.close()


def load_feature(config, mode, interface, path=CONFIG, *, model_version='v81', camera=None,
                 camera_process=None):
    """Feature failure must not remove normal Balance/Homie operation."""
    config.pushcart_error = ''
    try:
        sys.path.insert(0, str(WORKSPACE / 'G1_high_policy_deploy/robot'))
        from pushcart_v81.viser_policy import ViserPushPolicy
        high = ViserPushPolicy(config, path, camera=camera, model_version=model_version)
        high.camera_process = camera_process if camera_process is not None else CameraProcess(
            high.settings['camera']['udp_port'], mode, interface, high.settings['camera']['tag_ids'])
        config.high_policy_metadata = high.metadata
        return high
    except Exception as exc:
        config.pushcart_error = str(exc)
        return None
