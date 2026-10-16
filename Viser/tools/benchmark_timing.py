"""Measure the Viser loop on C++ synthetic feedback; never opens robot DDS."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import tempfile
import argparse
import json
import socket
import threading
import time

import numpy as np

from core.backends import Bridge, build_bridge
from core.controller import Controller
from core.recording import SessionRecorder
from core.robot import ROOT, RobotConfig


def wait_for(controller, predicate, timeout=20):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        data = controller.snapshot()
        if data.get('stage') in ('FAULT', 'DAMPED'):
            raise RuntimeError(data['reason'])
        if predicate(data):
            return data
        time.sleep(.02)
    raise TimeoutError(controller.snapshot())


def stats(values):
    values = [value for value in values if value is not None]
    return dict(mean=float(np.mean(values)), p50=float(np.percentile(values, 50)),
                p99=float(np.percentile(values, 99)), maximum=float(np.max(values)),
                samples=len(values)) if values else None


class MeasuredController(Controller):
    """Collect every completed tick, without snapshot polling losing samples."""
    collecting = False

    def instrument(self):
        self.samples, self.model_ms = [], {}
        infer = self.policy.infer

        def timed_infer(name, observations):
            started = time.perf_counter()
            result = infer(name, observations)
            self.model_ms[name] = (time.perf_counter() - started) * 1000
            return result

        self.policy.infer = timed_infer

    def step(self):
        previous = self.policy_steps
        started = time.perf_counter()
        super().step()
        elapsed = (time.perf_counter() - started) * 1000
        if self.collecting and self.policy_steps != previous and self.stage == 'POLICY':
            data = self.snapshot()
            self.samples.append(dict(
                completed=data['policy_inference_monotonic'], step_ms=elapsed,
                models=dict(self.model_ms), **data['timing']))


class SyntheticCamera:
    def __init__(self, high):
        self.destination = high.camera.sock.getsockname()
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def run(self):
        while not self.stop.is_set():
            packet = dict(schema='carmer_cart_yaw_v1', captured_monotonic=time.monotonic()-.04,
                          valid=True, reference_frame='pelvis_level', yaw_rad=.1, tag_ids=[0, 1])
            self.sock.sendto(json.dumps(packet).encode(), self.destination)
            self.stop.wait(.02)

    def close(self):
        self.stop.set()
        self.thread.join(timeout=2)
        self.sock.close()


class ViewerLoad:
    """Actual Viser scene/UI serialization to one client, without browser rendering."""
    def __init__(self, controller):
        from Vender.app import Viewer
        import msgspec
        import viser
        from websockets.sync.client import connect
        with socket.socket() as probe:
            probe.bind(('127.0.0.1', 0))
            port = probe.getsockname()[1]
        self.viewer = Viewer(controller, '127.0.0.1', port)
        try:
            self.ws = connect('ws://127.0.0.1:%d' % port, max_size=96*1024*1024,
                              proxy=None, subprotocols=['viser-v' + viser.__version__])
            self.ws.send(msgspec.msgpack.encode(dict(
                type='ViewerCameraMessage', wxyz=[1, 0, 0, 0], position=[2, -2, 1.5],
                fov=1., near=.01, far=100., image_height=720, image_width=1280,
                look_at=[0, 0, .7], up_direction=[0, 0, 1])))
        except BaseException:
            self.viewer.close()
            raise
        self.error = ''
        self.thread = threading.Thread(target=self.receive, daemon=True)
        self.thread.start()

    def receive(self):
        try:
            for _ in self.ws:
                pass
        except Exception as exc:
            self.error = str(exc)

    def close(self):
        self.ws.close()
        self.thread.join(timeout=2)
        self.viewer.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds', type=float, default=20)
    parser.add_argument('--record', action='store_true')
    parser.add_argument('--pushcart', action='store_true', help='Run real student + both low-level models with synthetic cart yaw')
    parser.add_argument('--viewer', action='store_true', help='Include 30 Hz Viser updates and one receiving WebSocket client')
    parser.add_argument('--vx', type=float, default=0., help='Synthetic task/body forward command; never sent to a robot')
    args = parser.parse_args()
    if not np.isfinite(args.seconds) or args.seconds < 2:
        parser.error('--seconds must be finite and >= 2')
    if not np.isfinite(args.vx) or not -.2 <= args.vx <= .4:
        parser.error('--vx must be finite and within [-0.2, 0.4]')
    config = RobotConfig()
    build_bridge()
    temporary = tempfile.TemporaryDirectory(prefix='viser_benchmark_')
    bridge_output = tempfile.TemporaryFile(mode='w+')
    controller = backend = recorder = high = camera = viewer = None
    error = None
    try:
        if args.pushcart:
            from core.pushcart import CONFIG, WORKSPACE
            sys.path.insert(0, str(WORKSPACE / 'G1_high_policy_deploy/robot'))
            from pushcart_v81.carmer_io import CarmerYaw
            from pushcart_v81.viser_policy import ViserPushPolicy
            receiver = CarmerYaw(0, tag_ids=[0, 1])
            try:
                high = ViserPushPolicy(config, CONFIG, camera=receiver)
            except BaseException:
                receiver.close()
                raise
            config.high_policy_metadata = high.metadata
            camera = SyntheticCamera(high)
        if args.record:
            recorder = SessionRecorder(config, dict(mode='benchmark', **vars(args)),
                                       Path(temporary.name) / 'sessions')
        backend = Bridge(config, 'sim', output=bridge_output)
        controller = MeasuredController(config, backend, 'sim', recorder=recorder, high_policy=high)
        controller.instrument()
        if args.viewer:
            viewer = ViewerLoad(controller)
        controller.start()
        wait_for(controller, lambda d: d.get('healthy'))
        controller.emit('claim', 1)
        wait_for(controller, lambda d: d.get('owner') == 1)
        controller.emit('acquire', 1)
        wait_for(controller, lambda d: d.get('stage') == 'ZERO')
        controller.emit('prepare', 1)
        wait_for(controller, lambda d: d.get('stage') == 'HOLDING' and d.get('observation_frames', 0) >= 6)
        controller.emit('policy', 1)
        wait_for(controller, lambda d: d.get('stage') == 'POLICY')
        if high:
            controller.emit('hand_toggle', 1)
            wait_for(controller, lambda d: d['dex3']['active'] and not d['dex3']['moving']
                     and not d['dex3']['pending'] and d['high_policy']['history_ready'])
            controller.emit('pushcart_enable', 1)
            wait_for(controller, lambda d: d['high_policy']['enabled'] and not d['upper_pd_transitioning'])
        # Run the same viewer update rate as app.py, including a short settling period.
        started = time.monotonic()
        controller.collecting = False
        while time.monotonic() - started < args.seconds + 2:
            controller.collecting = time.monotonic() - started >= 2
            if args.vx:
                controller.emit('move', 1, [args.vx, 0., 0.])
            if viewer:
                viewer.viewer.update()
                if viewer.error:
                    raise RuntimeError(viewer.error)
            data = controller.snapshot()
            if data['stage'] != 'POLICY' or (high and not data['high_policy']['enabled']):
                raise RuntimeError(data['reason'])
            time.sleep(1/30)
        controller.collecting = False
        samples = list(controller.samples)
        if len(samples) < 2:
            raise RuntimeError('Not enough completed inference samples')
        completions = np.array([sample['completed'] for sample in samples])
        intervals = np.diff(completions) * 1000
        report = dict(
            scope='C++ synthetic feedback + real ONNX; no physics, robot DDS, camera image processing or browser rendering',
            pushcart=args.pushcart, automatic_recording=args.record, viser_client=args.viewer,
            seconds=float(completions[-1]-completions[0]),
            policy_hz=float((len(samples)-1)/(completions[-1]-completions[0])),
            per_model_ms={name: stats([s['models'][name] for s in samples]) for name in ('balance', 'homie')},
            student_ms=stats([s['high_policy_inference_ms'] for s in samples]),
            inference_ms=stats([s['inference_ms'] for s in samples]),
            full_step_ms=stats([s['step_ms'] for s in samples]),
            completion_interval_ms=stats(intervals),
            compute_over_20ms=sum(s['step_ms'] > 20 for s in samples),
            completion_gaps_over_25ms=int(np.count_nonzero(intervals > 25)),
            observation_to_action_ms=stats([s['observation_to_action_ms'] for s in samples]),
            state_to_command_ms=stats([s['state_to_command_ms'] for s in samples]),
            requested_vx=args.vx, velocity=data['velocity'].tolist(),
            recording_dropped=recorder.dropped if recorder else 0)
        if recorder and recorder.error:
            raise RuntimeError(recorder.error)
        print(json.dumps(report, indent=2))
    except BaseException as exc:
        error = repr(exc)
        bridge_output.seek(0)
        print('Bridge output:\n' + bridge_output.read()[-6000:], file=sys.stderr)
        raise
    finally:
        if camera:
            camera.close()
        if controller:
            controller.close()
        else:
            if backend:
                backend.close()
            if high:
                high.close()
        if viewer:
            viewer.close()
        if recorder:
            recorder.close(error)
        bridge_output.close()
        temporary.cleanup()


if __name__ == '__main__':
    main()
