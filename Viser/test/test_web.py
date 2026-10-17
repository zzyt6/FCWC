"""Exercise the actual Viser WebSocket buttons against the hybrid C++ synthetic bridge; never opens DDS."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import tempfile
import numpy as np
import viser
import gzip
import socket
import subprocess
import threading
import time
import urllib.request

import msgspec
import zstandard
from websockets.sync.client import connect
from core.heading import load_settings

ROOT = Path(__file__).resolve().parents[1]


def until(predicate, timeout=12):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = predicate()
        if result:
            return result
        time.sleep(.025)
    raise AssertionError('Timed out: ' + str(predicate))


class Browser:
    def __init__(self, port):
        self.ws = connect('ws://127.0.0.1:%d' % port, max_size=96*1024*1024,
                          proxy=None, subprotocols=['viser-v' + viser.__version__])
        self.components = {}
        self.meshes = 0
        self.orientations = 0
        self.html = {}
        self.panel_edges = []
        self.images = 0
        self.error = None
        self.thread = threading.Thread(target=self.read, daemon=True)
        self.thread.start()
        self.send(type='ViewerCameraMessage', wxyz=[1, 0, 0, 0], position=[2, -2, 1.5],
                  fov=1., near=.01, far=100., image_height=720, image_width=1280,
                  look_at=[0, 0, .7], up_direction=[0, 0, 1])
        until(lambda: self.find('接管网页操作'))

    def send(self, **message):
        self.ws.send(msgspec.msgpack.encode(message))

    def read(self):
        try:
            for raw in self.ws:
                size = int.from_bytes(raw[8:16], 'little')
                payload = msgspec.msgpack.decode(zstandard.ZstdDecompressor().decompress(raw[16:16+size]))
                for message in payload['messages']:
                    kind = message['type']
                    if kind == 'GuiHtmlMessage':
                        self.html[message['uuid']] = message['props']['content']
                    elif kind == 'GuiUpdateMessage' and message['uuid'] in self.html:
                        self.html[message['uuid']] = message['updates'].get('content', self.html[message['uuid']])
                    elif kind == 'GuiImageMessage':
                        self.images += 1
                    elif kind == 'GuiSetPanelPositionMessage':
                        self.panel_edges.append(message['position'].get('edge'))
                    if kind in ('GuiButtonMessage', 'GuiSliderMessage', 'GuiDropdownMessage', 'GuiButtonGroupMessage', 'GuiCheckboxMessage'):
                        self.components[message['uuid']] = dict(message['props'])
                        self.components[message['uuid']]['value'] = message.get('value', message['props'].get('initial_value'))
                    elif kind == 'GuiUpdateMessage' and message['uuid'] in self.components:
                        self.components[message['uuid']].update(message['updates'])
                    elif kind in ('MeshMessage', 'GlbMessage'):
                        self.meshes += 1
                    elif kind == 'SetOrientationMessage':
                        self.orientations += 1
        except Exception as exc:
            self.error = str(exc)

    def find(self, label):
        return next((uid for uid, props in list(self.components.items()) if props.get('label') == label), None)

    def click(self, label, force=False):
        uid = until(lambda: self.find(label))
        if not force:
            until(lambda: not self.components[uid].get('disabled'))
        self.send(type='GuiUpdateMessage', uuid=uid, updates={'value': True})

    def hold(self, seconds):
        uid = until(lambda: self.find('按住发送速度'))
        until(lambda: not self.components[uid].get('disabled'))
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            self.send(type='GuiButtonHoldMessage', uuid=uid, frequency=20.)
            time.sleep(.05)

    def set_slider(self, label, value):
        uid = until(lambda: self.find(label))
        self.send(type='GuiUpdateMessage', uuid=uid, updates={'value': value})

    def close(self):
        self.ws.close()
        self.thread.join(timeout=2)


def main():
    defaults = load_settings(run_mode='active')
    default_kp = defaults['heading_kp']
    temporary = tempfile.TemporaryDirectory(prefix='viser_web_')
    directory = Path(temporary.name)
    log = directory / 'web_test.jsonl'
    log.unlink(missing_ok=True)
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0))
        port = probe.getsockname()[1]
    output = (directory / 'web_test.log').open('w')
    process = subprocess.Popen([sys.executable, str(ROOT / 'Vender/app.py'), '--mode', 'sim',
                                '--heading-mode', 'active', '--host', '127.0.0.1', '--port', str(port), '--duration', '100',
                                '--log', str(log), '--record-sim', '--record-dir', str(directory/'web_sessions')],
                               stdout=output, stderr=subprocess.STDOUT)
    clients = []
    def row():
        if not log.exists():
            return {}
        lines = log.read_text().splitlines()
        for line in reversed(lines):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                pass
        return {}
    def ready():
        if process.poll() is not None:
            raise AssertionError((directory / 'web_test.log').read_text())
        try:
            with urllib.request.urlopen('http://127.0.0.1:%d' % port, timeout=.2) as response:
                return response.status == 200
        except OSError:
            return False
    try:
        until(ready, 45)
        browser = Browser(port)
        clients.append(browser)
        until(lambda: row().get('healthy'))
        until(lambda: browser.meshes >= 40)
        # Left monitor is compact, fixed degrees, with no picture or unit switch.
        assert 'left' in browser.panel_edges, browser.panel_edges
        table = lambda: next((c for c in browser.html.values() if 'aria-label="全部关节角度"' in c), '')
        until(lambda: 'left_knee' in table() and '下发 °' in table())
        assert table().count('class="n ') == 41
        assert not browser.find('角度单位')
        assert not browser.find('Dex3 双手：补偿归零（3 秒）')
        assert not browser.find('Dex3 双手：补偿抓握（3 秒）')
        assert browser.find('Dex3 使用文件补偿角')
        assert browser.images == 0
        speed_labels = ['前后 vx (m/s)', '左右 vy (m/s)', '转向 yaw (rad/s)']
        for label, bounds in zip(speed_labels, [(-.8, 1.2), (-.5, .5), (-.8, .8)]):
            props = browser.components[browser.find(label)]
            assert props['value'] == 0., label
            assert (props['min'], props['max']) == bounds, (label, props)
        browser.click('接管网页操作')
        until(lambda: row().get('owner') is not None)
        owner = row()['owner']
        other = Browser(port)
        clients.append(other)
        other.click('接管网页操作', force=True)
        time.sleep(.15)
        assert row()['owner'] == owner
        until(lambda: row().get('heading', {}).get('effective_frame') == 'body')
        other.set_slider('航向 Kp (1/s)', .4)
        other.set_slider('角速度滤波截止频率 (Hz)', 2.)
        other.click('应用航向参数', force=True)
        time.sleep(.15)
        assert row()['heading']['heading_kp'] == default_kp
        assert row()['heading']['gyro_cutoff_hz'] == defaults['gyro_cutoff_hz']
        browser.set_slider('航向 Kp (1/s)', .65)
        browser.set_slider('角速度滤波截止频率 (Hz)', 4.)
        browser.click('应用航向参数')
        until(lambda: row().get('heading', {}).get('heading_kp') == .65)
        until(lambda: row().get('heading', {}).get('gyro_cutoff_hz') == 4.)
        assert row()['heading']['effective_frame'] == 'body'
        browser.set_slider('航向 Kp (1/s)', default_kp)
        browser.set_slider('角速度滤波截止频率 (Hz)', defaults['gyro_cutoff_hz'])
        browser.click('应用航向参数')
        until(lambda: row().get('heading', {}).get('heading_kp') == default_kp)
        browser.click('以当前朝向重设航向目标')
        browser.click('① 接管底层')
        until(lambda: row().get('stage') == 'ZERO')
        browser.click('Dex3 使能并抓握（3 秒）', force=True)
        time.sleep(.15)
        assert not row()['dex3']['active']
        browser.click('③ 启动混合策略', force=True)
        time.sleep(.15)
        assert row()['stage'] == 'ZERO'
        browser.click('② 移到准备姿态')
        until(lambda: row().get('stage') in ('HOLDING', 'FAULT'))
        assert row()['stage'] == 'HOLDING', row()
        from core.robot import RobotConfig, ACTIVE, UPPER
        config = RobotConfig()
        np.testing.assert_allclose(np.asarray(row()['target'])[ACTIVE], config.prepare_q[ACTIVE], atol=1e-4)
        np.testing.assert_array_equal(row()['dex3']['target'], np.zeros(14))
        until(lambda: not row()['dex3']['moving'])
        other.click('Dex3 使能并抓握（3 秒）', force=True)
        time.sleep(.15)
        assert row()['dex3']['pose'] == 'open'
        browser.click('Dex3 使能并抓握（3 秒）')
        until(lambda: row()['dex3']['moving'] and row()['dex3']['pose'] == 'grasp')
        browser.click('Dex3 双手：松开（3 秒）', force=True)
        time.sleep(.2)
        assert row()['dex3']['pose'] == 'grasp' and row()['dex3']['moving']
        until(lambda: not row()['dex3']['moving'])
        np.testing.assert_allclose(row()['dex3']['target'], config.hand_grasp.ravel(), atol=1e-6)
        np.testing.assert_allclose(np.asarray(row()['target'])[UPPER], config.prepare_q[UPPER], atol=1e-4)
        browser.click('③ 启动混合策略')
        until(lambda: row().get('stage') == 'POLICY')
        browser.click('Dex3 双手：松开（3 秒）')
        until(lambda: row()['dex3']['moving'] and row()['dex3']['pose'] == 'open')
        until(lambda: not row()['dex3']['moving'])
        np.testing.assert_array_equal(row()['dex3']['target'], np.zeros(14))
        browser.hold(.25)
        assert row()['velocity'] == [0., 0., 0.], row()
        browser.set_slider('前后 vx (m/s)', .1)
        time.sleep(.2)
        assert row()['velocity'] == [0., 0., 0.], row()
        browser.hold(1.2)
        browser.set_slider('航向 Kp (1/s)', .4)
        browser.set_slider('角速度滤波截止频率 (Hz)', 5.)
        browser.click('应用航向参数', force=True)
        assert row()['heading']['heading_kp'] == default_kp
        assert row()['heading']['gyro_cutoff_hz'] == defaults['gyro_cutoff_hz']
        assert row()['selected_model'] == 'homie', row()
        assert row()['pd_profile'] == 'homie', row()
        assert row()['pd_transition_blend'] == 1., row()
        from hybrid_deploy.configuration import load_config
        _, gains, _ = load_config(RobotConfig().deploy)
        np.testing.assert_allclose(np.asarray(row()['kp'])[ACTIVE], np.asarray(gains['homie']['kp_sdk29'])[ACTIVE])
        assert row()['kp'][13:15] == [0., 0.]
        until(lambda: row().get('velocity') == [0., 0., 0.])
        until(lambda: row().get('selected_model') == 'balance' and row().get('pd_profile') == 'balance'
              and row().get('pd_transition_blend') == 1.)
        np.testing.assert_allclose(np.asarray(row()['kp'])[ACTIVE], np.asarray(gains['balance']['kp_sdk29'])[ACTIVE])
        assert row()['stage'] == 'POLICY', row()
        steps = row()['policy_steps']
        assert steps >= 20
        assert abs(row()['state']['q'][0] + .1) > 1e-3
        browser.click('速度归零')
        until(lambda: browser.components[browser.find(speed_labels[0])]['value'] == 0.)
        browser.hold(.25)
        assert row()['velocity'] == [0., 0., 0.], row()
        browser.set_slider('前后 vx (m/s)', .2)
        browser.set_slider('左右 vy (m/s)', .1)
        browser.set_slider('转向 yaw (rad/s)', .2)
        browser.hold(1.2)
        heading = row()['heading']
        assert heading['active'] and not heading['correction_enabled'], row()
        assert heading['heading_state'] == 'TURNING', row()
        assert heading['effective_frame'] == 'body', row()
        assert heading['target_relative_rad'] is None, row()
        assert heading['correction_rad_s'] == 0, row()
        assert row()['velocity'][0] > 0 and row()['velocity'][1] > 0, row()
        assert .18 < row()['velocity'][2] <= .2, row()
        browser.click('速度归零')
        until(lambda: row().get('decelerating'))
        assert row()['velocity'] != [0.,0.,0.], row()
        assert row()['input_velocity'] == [0.,0.,0.], row()
        assert not row()['heading']['correction_enabled'], row()
        until(lambda: row().get('velocity') == [0., 0., 0.])
        assert not row()['decelerating'], row()
        assert row()['heading']['correction_rad_s'] == 0, row()
        assert row()['heading']['correction_raw_rad_s'] == 0, row()
        browser.click('速度归零')
        browser.click('阻尼停止')
        until(lambda: row().get('stage') == 'DAMPED')
        browser.click('③ 启动混合策略', force=True)
        time.sleep(.2)
        assert row()['stage'] == 'DAMPED'
        time.sleep(1.5)  # The stopped bridge has exited; its old feedback is stale.
        browser.click('接管网页操作')
        until(lambda: row().get('stage') == 'WAITING', 20)
        browser.click('① 接管底层')
        until(lambda: row().get('stage') == 'ZERO')
        browser.click('② 移到准备姿态')
        until(lambda: row().get('stage') == 'HOLDING')
        np.testing.assert_array_equal(row()['dex3']['target'], np.zeros(14))
        browser.click('Dex3 使能并抓握（3 秒）')
        until(lambda: row()['dex3']['moving'])
        browser.close()
        clients.remove(browser)
        until(lambda: row().get('stage') == 'DAMPED')
        assert '断开' in row()['reason']
        until(lambda: row().get('owner') is None)
        time.sleep(1.5)
        other.click('接管网页操作')
        until(lambda: row().get('stage') == 'WAITING' and row().get('healthy'), 20)
        other.click('① 接管底层')
        until(lambda: row().get('stage') == 'ZERO')
        report = dict(passed=True, mesh_messages=browser.meshes, orientation_messages=browser.orientations,
                      policy_steps=steps, tested=['ownership', 'ordered_stages', 'hybrid_bridge_27dof_display', 'balance_homie_switch', 'actual_pd_profiles',
                      'zero_default_sliders', 'zero_policy_start', 'slider_alone_no_command',
                      'speed_hold', 'release_to_zero', 'zero_clears_sliders',
                      'heading_settings_ownership', 'body_frame_with_heading', 'heading_reference_rebase',
                      'combined_xy_turn_releases_heading', 'heading_timeout_zero',
                      'manual_zero_ramp_final_commands',
                      'latched_stop', 'reclaim_after_stop', 'disconnect_damping', 'new_owner_reclaim',
                      'compact_degree_joint_panel', 'dex3_compensation_checkbox'])
        report['tested'] += ['yaml_full_body_prepare', 'dex3_3s_grasp_open', 'dex3_owner_and_busy_guard',
                             'dex3_disconnect_during_ramp', 'dex3_reset_open']
    finally:
        for browser in clients:
            browser.close()
        process.terminate()
        try:
            process.wait(timeout=12)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=3)
        output.close()
    sessions = [p.parent for p in (directory/'web_sessions').glob('*/metadata.json')
                if json.loads(p.read_text())['arguments']['port'] == port]
    assert len(sessions) == 1
    session = sessions[0]
    summary = json.loads((session/'summary.json').read_text())
    assert summary['dropped_records'] == 0 and summary['recorder_error'] == '', summary
    metadata = json.loads((session/'metadata.json').read_text())
    assert metadata['motion_settings'] == defaults
    assert not (session/'snapshot').exists()
    with gzip.open(session/'telemetry.jsonl.gz', 'rt') as stream:
        telemetry = [json.loads(line) for line in stream]
    assert len(telemetry) > 300
    assert all(not row['motor_command_written'] for row in telemetry)
    inference_rows = [row for row in telemetry if row.get('model_actions') is not None]
    assert len(inference_rows) > 100
    for row in inference_rows:
        assert len(row['model_actions']['balance']) == 12
        assert len(row['model_actions']['homie']) == 12
        assert len(row['homie_estimator']['velocity_raw']) == 3
        assert len(row['homie_estimator']['latent_raw']) == 32
        assert len(row['homie_estimator']['latent_normalized']) == 32
        assert row['policy_inference_monotonic'] <= row['monotonic']
    stopped = [row for row in telemetry if row['stage'] in ('DAMPED', 'FAULT')]
    assert stopped and all(row['model_actions'] is None and row['homie_estimator'] is None for row in stopped)
    with gzip.open(session/'events.jsonl.gz', 'rt') as stream:
        events = [json.loads(line) for line in stream]
    assert not any(e['type'] == 'policy_result' for e in events)
    assert any(e['type'] == 'motion_settings' and e['settings']['heading_kp'] == .65 for e in events)
    assert any(e['type'] == 'motion_settings_rejected' for e in events)
    assert any(e['type'] == 'smooth_zero_started' for e in events)
    assert any(e['type'] == 'smooth_zero_completed' for e in events)
    grasp_ramp = [r for r in telemetry if r.get('dex3', {}).get('moving')
                  and r['dex3']['pose'] == 'grasp' and r['stage'] == 'HOLDING']
    # The first complete transition precedes the later interrupted transition.
    first_ramp = []
    for r in grasp_ramp:
        if first_ramp and r['dex3']['progress'] < first_ramp[-1]['dex3']['progress']:
            break
        first_ramp.append(r)
    assert len(first_ramp) > 100
    assert 2.8 < first_ramp[-1]['monotonic']-first_ramp[0]['monotonic'] < 3.2
    hand_q = np.asarray([r['dex3']['target'] for r in first_ramp])
    assert np.all(np.diff(abs(hand_q), axis=0) >= -1e-6)
    # Recorded samples need not be evenly spaced. Bound the hand trajectory
    # by elapsed ramp progress (peak slope 1.875), not by row count.
    progress = np.asarray([r['dex3']['progress'] for r in first_ramp])
    max_change = 1.875 * np.diff(progress)[:, None] * abs(config.hand_grasp.ravel())
    assert np.all(abs(np.diff(hand_q, axis=0)) <= max_change + 1e-6)
    ramps = []
    current = []
    for r in telemetry:
        if r.get('decelerating'):
            current.append(r)
        elif current:
            ramps.append(current)
            current = []
            if r['stage'] == 'POLICY' and not np.any(r['velocity']):
                assert r['selected_model'] == 'balance'  # no zero-command dwell
    if current:
        ramps.append(current)
    assert sum(map(len, ramps)) > 12
    for ramp in ramps:
        ramp_velocity = np.asarray([r['velocity'] for r in ramp])
        assert np.all(np.diff(abs(ramp_velocity), axis=0) <= 1e-7)
    assert max(ramp[-1]['monotonic']-ramp[0]['monotonic'] for ramp in ramps) > .25
    report.update(compact_recording_verified=True)
    (directory / 'web_test_result.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
    print('Automatic archive verified:', session)
    temporary.cleanup()


if __name__ == '__main__':
    main()
