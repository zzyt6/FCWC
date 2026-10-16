"""Real Viser WebSocket + student + UDP camera; synthetic body, no DDS/LCM.

Run directly with the CWC Python. Uses only temporary files and free local ports,
so it can run while the real robot's existing controller remains untouched.
"""
import gzip
import json
from pathlib import Path
import socket
import sys
import tempfile
import threading
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_viser_integration import (ROOT, RobotConfig, Controller, FakeBridge, CarmerYaw,
                                    HeadingPublisher, ViserPushPolicy, camera_result, p)
from core.recording import SessionRecorder
from Vender.app import Viewer
from test_web import Browser, until


class InteractiveBridge(FakeBridge):
    """Only models lifecycle/feedback for testing UI routing, not physics."""
    def __init__(self, config, mode="sim", interface="eth0", output=None):
        super().__init__(config)
        self.prepared_at = None

    def read(self):
        self.state.received_ns = time.monotonic_ns()
        self.state.sequence += 1
        if self.prepared_at is not None and time.monotonic() >= self.prepared_at:
            self.state.stage = p.HOLDING
            self.prepared_at = None
        return self.state

    def send(self, request, q, profile=0, hand_pose=0, **kwargs):
        super().send(request, q, profile, hand_pose, **kwargs)
        if not self.armed:
            return
        if request == p.PREPARE and self.state.stage == p.WAITING:
            self.state.stage = p.RAMPING
            self.prepared_at = time.monotonic() + .15
        elif request == p.RUN and self.state.stage == p.HOLDING:
            self.state.stage = p.POLICY
        elif request == p.STOP:
            self.state.stage = p.DAMPED
        self.state.applied_q = q.copy()
        self.state.q = q.copy()
        self.state.profile = profile
        if hand_pose:
            self.state.hand_flags = 39 | (16 if hand_pose in (2, 4) else 0)
            self.state.hand_target = np.asarray(kwargs.get('hand_grasp', np.zeros(14))).copy()
            self.state.hand_q = self.state.hand_target.copy()

    def close(self):
        pass


def main():
    import os
    import signal
    from unittest.mock import patch
    from core import backends
    from Vender import app
    with tempfile.TemporaryDirectory(prefix='pushcart-web-') as directory:
        directory = Path(directory)
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.bind(('127.0.0.1', 0))
            camera_port = probe.getsockname()[1]
        with socket.socket() as probe:
            probe.bind(('127.0.0.1', 0))
            port = probe.getsockname()[1]
        settings = json.loads((ROOT / 'robot/config/viser_v81.json').read_text())
        settings['models_dir'] = str(ROOT / 'robot/models')
        settings['camera']['udp_port'] = camera_port
        config_path = directory / 'pushcart.json'
        config_path.write_text(json.dumps(settings))
        publisher = HeadingPublisher(camera_port)
        stopped, camera_on = threading.Event(), threading.Event()
        shared, errors = {}, []
        original_viewer = app.Viewer

        def viewer_factory(control, host, server_port):
            viewer = original_viewer(control, host, server_port)
            shared.update(control=control, viewer=viewer)
            return viewer

        def camera_loop():
            while not stopped.is_set():
                if camera_on.is_set():
                    publisher.publish(camera_result(time.monotonic(), yaw=.1))
                stopped.wait(.02)

        def operate():
            browser = None
            try:
                until(lambda: shared.get('viewer'), timeout=30)
                control, viewer = shared['control'], shared['viewer']
                browser = Browser(port)
                browser.click('接管网页操作')
                browser.click('① 接管底层')
                browser.click('② 移到准备姿态')
                browser.click('③ 启动混合策略')
                until(lambda: control.snapshot().get('stage') == 'POLICY')
                assert not control.high_policy.enabled
                assert not viewer.cart_display.frame.visible
                move_uid = until(lambda: browser.find('按住发送速度'))
                until(lambda: not browser.components[move_uid]['disabled'])
                browser.set_slider('前后 vx (m/s)', .3)
                until(lambda: next(iter(viewer.clients.values()))[2][0].value == .3)
                browser.hold(.8)
                assert control.snapshot()['selected_model'] == 'homie', {
                    k: control.snapshot().get(k) for k in ('stage', 'reason', 'velocity', 'input_velocity')}
                assert control.snapshot()['high_policy']['raw_actions'] is None
                browser.click('速度归零')
                until(lambda: not np.any(control.snapshot()['velocity']))
                browser.click('Dex3 使能并抓握（3 秒）')
                until(lambda: control.snapshot().get('dex3', {}).get('active'))
                uid = until(lambda: browser.find('启用推车策略'))
                assert browser.components[uid]['disabled']
                camera_on.set()
                browser.click('标定航向零位（车身已对正）')
                until(lambda: control.snapshot()['high_policy']['zero_calibration']['status'] == 'done')
                assert abs(control.high_policy.camera.offset - .1) < 1e-6
                assert abs(control.high_policy.camera.yaw) < 1e-6
                assert json.loads(config_path.read_text())['camera']['yaw_offset_rad'] == settings['camera']['yaw_offset_rad']
                browser.click('启用推车策略')
                until(lambda: control.high_policy.enabled)
                until(lambda: control.snapshot().get('homie_estimator_source') == 'pushcart_finetuned')
                zero_uid = until(lambda: browser.find('标定航向零位（车身已对正）'))
                until(lambda: browser.components[zero_uid]['disabled'])
                until(lambda: viewer.cart_display.frame.visible)
                assert not viewer.cart_display.error, viewer.cart_display.error
                hand_uid = until(lambda: browser.find('左手拇指 0 (°)'))
                until(lambda: browser.components[hand_uid]['disabled'])
                reference = control.config.hand_grasp.copy()
                browser.set_slider('小车前后 vx (m/s)', .3)
                browser.hold(1.5)
                assert control.snapshot()['selected_model'] == 'homie'
                heading = control.snapshot()['heading']
                assert heading['run_mode'] == 'active'
                assert heading['command_source'] == 'PUSHCART'
                assert heading['correction_enabled'] and heading['pushcart_tracking']
                assert np.any(control.high_policy.applied_residual)
                browser.click('速度归零')
                until(lambda: not np.any(control.snapshot()['velocity']))
                assert control.snapshot()['selected_model'] == 'balance'
                assert control.snapshot()['heading']['correction_rad_s'] == 0.
                browser.click('返回普通行走 / 继续调抓握')
                until(lambda: not control.high_policy.enabled)
                until(lambda: control.snapshot().get('homie_estimator_source') == 'homie_original')
                np.testing.assert_array_equal(control.config.hand_grasp, reference)
                until(lambda: not browser.components[hand_uid]['disabled'])
                browser.set_slider('左手拇指 0 (°)', 5.73)
                until(lambda: abs(control.config.hand_grasp[0, 0] - np.deg2rad(5.73)) < 1e-6)
                index = viewer.display_robot.actuated_joint_names.index('left_hand_thumb_0_joint')
                until(lambda: abs(viewer.display_robot.cfg[index] - np.deg2rad(5.73)) < 1e-6)
                browser.click('启用推车策略')
                until(lambda: control.high_policy.enabled)
                browser.set_slider('小车前后 vx (m/s)', .3)
                camera_on.clear()
                browser.hold(1.2)
                until(lambda: control.high_policy.motion_blocked)
                until(lambda: not np.any(control.snapshot()['velocity']))
                assert viewer.cart_display.frame.visible
                browser.click('返回普通行走 / 继续调抓握')
                until(lambda: not control.high_policy.motion_blocked)
                browser.set_slider('前后 vx (m/s)', .3)
                browser.hold(.8)
                assert control.snapshot()['selected_model'] == 'homie'
                browser.click('阻尼停止')
                until(lambda: control.snapshot().get('stage') == 'DAMPED')
            except BaseException as exc:
                if shared.get('viewer'):
                    print('Workflow failure:', repr(exc), 'cart:', shared['viewer'].cart_display.error,
                          'controller:', shared['control'].snapshot().get('reason'), flush=True)
                errors.append(exc)
            finally:
                if browser:
                    browser.close()
                os.kill(os.getpid(), signal.SIGINT)

        camera_thread = threading.Thread(target=camera_loop, daemon=True)
        actor = threading.Thread(target=operate, daemon=True)
        args = ['app.py', '--mode', 'sim', '--host', '127.0.0.1', '--port', str(port),
                '--duration', '40', '--record-sim', '--record-dir', str(directory / 'sessions')]
        try:
            with patch.object(app, 'PUSHCART_CONFIG', config_path), patch.object(app, 'Viewer', viewer_factory), \
                 patch.object(backends, 'Bridge', InteractiveBridge), patch.object(backends, 'build_bridge'), \
                 patch.object(sys, 'argv', args):
                camera_thread.start()
                actor.start()
                app.main()  # Normal entry, no --pushcart-config or second frontend.
            actor.join(timeout=2)
            if errors:
                raise errors[0]
        finally:
            stopped.set()
            camera_thread.join(timeout=2)
            publisher.close()
        record = next((directory / 'sessions').iterdir())
        metadata = json.loads((record / 'metadata.json').read_text())
        assert 'high_policy' in metadata
        assert metadata['high_policy']['homie']['sha256']
        with gzip.open(record / 'telemetry.jsonl.gz', 'rt') as stream:
            rows = [json.loads(row) for row in stream]
        assert any((r.get('high_policy') or {}).get('raw_actions') is not None for r in rows)
        assert any((r.get('high_policy') or {}).get('inference_role') == 'shadow' for r in rows)
        first_push = next(r for r in rows if (r.get('high_policy') or {}).get('enabled'))
        assert first_push['high_policy']['history_frames'] == 32
        assert first_push['homie_estimator_source'] == 'pushcart_finetuned'
        for row in rows:
            high = row.get('high_policy')
            if high and high['inference_role'] is not None:
                assert high['inference_inputs']['input_state_tick'] == row['state']['tick']
                assert high['inference_inputs']['input_state_received_ns'] == row['state']['received_ns']
                assert row['state']['received'] <= high['inference_started_monotonic']
                assert high['inference_started_monotonic'] <= high['inference_finished_monotonic']
                assert high['inference_finished_monotonic'] <= row['policy_inference_monotonic']
        assert any(r['high_policy']['upper_target_selected'] for r in rows if r.get('high_policy'))
        first_policy = next(r for r in rows if r.get('stage') == 'POLICY')
        assert first_policy['observation_frames'] >= 6
        print('Synthetic feedback, first policy/push inference ms:',
              round(first_policy['timing']['inference_ms'], 3), round(first_push['timing']['inference_ms'], 3))
        print('PASS: original Viser entry; normal walk; grasp; yaw zero; opt-in push; cart display; live finger settings; '
              'return to normal; camera loss; restart normal without camera; stop; recording.')


if __name__ == '__main__':
    main()
