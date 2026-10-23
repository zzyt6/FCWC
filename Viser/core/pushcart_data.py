"""Push-cart data episodes in CWC/Data: one folder from push-cart enable to disable.

The control thread only enqueues each tick (update()). A background thread
detects the enable/disable edges and writes policy, motor and joint-command
CSVs from the same inference sample. It announces the episode window to
the tag detection recorder (carmer/episode_detections.py) and, in navigation, the ROS
odometry recording thread in the existing navigation web bridge. Format and time
alignment are documented in Data/README.md.
"""
from collections import deque
import csv
from datetime import datetime
import json
import math
from pathlib import Path
import queue
import socket
import subprocess
import threading
import time

import numpy as np

from core.data_episode import (CLOSE_MARGIN_S, CONTROL_PERIOD_S, CONTROL_PORTS, POST_S, PRE_S,
                               decode_ack, encode)
from core.hand_grasp import HAND_JOINTS
from core.joint_tracking import NAMES as SENT_NAMES
from core.robot import JOINT_NAMES, WORKSPACE
from hybrid_deploy.protocol import motor_telemetry

DATA_ROOT = WORKSPACE / 'Data'
HAND_NAMES = tuple(f'{side}_hand_{joint}_joint' for side in ('left', 'right') for joint in HAND_JOINTS)
MODES = {'web': 'Viser 网页滑条（Viser/run.sh）', 'nav': 'Nav2 导航（nav/run_navigation.sh）'}
# Recorders get this long to report their files closed before meta.json is final.
ACK_TIMEOUT_S = 3.0
XYZ = ('vx', 'vy', 'wz')

POLICY_COLUMNS = (
    't', 't_rel', 'state_received_t', 'inference_done_t', 'camera_captured_t', 'state_session', 'state_sequence',
    'stage', 'pushcart_enabled', 'motion_blocked', 'inference_role', 'selected_model', 'command_source',
    'decelerating',
    *('raw_task_' + a for a in XYZ), *('task_' + a for a in XYZ), *('task_ref_' + a for a in XYZ),
    *('homie_cmd_' + a for a in XYZ), *('policy_cmd_' + a for a in XYZ), *('filtered_cmd_' + a for a in XYZ),
    'camera_yaw_rad', 'camera_raw_yaw_rad', 'camera_yaw_offset_rad', 'camera_fresh', 'camera_status',
    'camera_age_s', 'camera_tag_ids', 'camera_pose_source',
    'cart_yaw_rad', 'imu_yaw_rad', 'imu_yaw_at_camera_rad', 'hold_ref_rad', 'hold_err_rad', 'hold_corr_rad_s',
    'inference_ms', 'high_policy_inference_ms', 'policy_hz', 'state_age_ms')
JOINT_COLUMNS = (
    't', 't_rel', 'state_received_t', 'pd_profile', 'pd_transition_blend', 'upper_pd_mode',
    *('target_req_' + n for n in JOINT_NAMES), *('target_' + n for n in JOINT_NAMES),
    *('kp_' + n for n in JOINT_NAMES), *('kd_' + n for n in JOINT_NAMES),
    *('sent_' + n for n in SENT_NAMES), 'sent_body_t', 'sent_hand_left_t', 'sent_hand_right_t')
MOTOR_COLUMNS = (
    't', 't_rel', 'state_received_t', 'inference_done_t', 'state_session', 'state_sequence',
    'tick', 'mode_pr', 'mode_machine',
    *(field + '_' + name for field in ('q', 'dq', 'ddq', 'tau_est', 'vol', 'mode') for name in JOINT_NAMES),
    'imu_qw', 'imu_qx', 'imu_qy', 'imu_qz',
    *('imu_' + field + '_' + axis for field in ('gyro', 'acc') for axis in ('x', 'y', 'z')),
    'imu_roll', 'imu_pitch', 'imu_yaw')
HAND_COLUMNS = {
    side: ('t', 't_rel', 'hand_received_t', 'inference_done_t', 'healthy', 'topic',
           *(field + '_' + name for field in ('q', 'dq', 'ddq', 'tau_est')
             for name in HAND_NAMES[index * 7:(index + 1) * 7]),
           'power_v', 'power_a', 'system_v', 'device_v', 'press_sensor_count',
           *('press_%d_%d' % (sensor, point) for sensor in range(9) for point in range(12)))
    for index, side in enumerate(('left', 'right'))}
LOCAL_COLUMNS = {'policy.csv': POLICY_COLUMNS, 'joint_command.csv': JOINT_COLUMNS,
                 'motor_state.csv': MOTOR_COLUMNS,
                 **{'hand_%s_state.csv' % side: columns for side, columns in HAND_COLUMNS.items()}}

FILES = {
    'meta.json': '本次记录的参数、时间窗口、时钟对照、关节顺序与各数据流状态',
    'policy.csv': '50 Hz 推车高层输入/输出：任务速度、小车 yaw、航向保持、相机时间戳',
    'joint_command.csv': '50 Hz 关节指令：策略目标、桥接实际目标、Kp/Kd、最近一次 DDS 下发值',
    'motor_state.csv': '与策略推理同频：本拍使用的 29 个电机 q/dq/ddq/tau_est/vol/mode 与 IMU',
    'hand_left_state.csv': '与策略推理同频：本拍桥接中的左手反馈，7 个电机与电源、9x12 压力',
    'hand_right_state.csv': '与策略推理同频：本拍桥接中的右手反馈，同上',
    'motor_summary.json': '与策略同拍的电机/手部记录行数、时间范围和丢弃计数',
    'camera_detections.csv': '标签检测结果：每个被检测帧的标签 ID 与小车相对骨盆 yaw',
    'odometry_fastlio.csv': '导航模式：FAST-LIO /Odometry 位姿（twist 恒为 0）',
    'odometry_base.csv': '导航模式：/Odometry_base 位姿与机身系平面速度',
    'cmd_vel.csv': '导航模式：Nav2 原始 /cmd_vel',
}


def _json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def _clock_pair():
    monotonic, realtime = time.monotonic(), time.time()
    return dict(monotonic=monotonic, realtime_unix=realtime,
                local=datetime.fromtimestamp(realtime).astimezone().isoformat())


def _git_state():
    try:
        commit = subprocess.run(['git', '-C', str(WORKSPACE), 'rev-parse', 'HEAD'], capture_output=True,
                                text=True, timeout=3).stdout.strip()
        dirty = subprocess.run(['git', '-C', str(WORKSPACE), 'status', '--porcelain', '--untracked-files=no'],
                               capture_output=True, text=True, timeout=5).stdout.strip()
        return dict(commit=commit or None, uncommitted_changes=bool(dirty))
    except (OSError, subprocess.SubprocessError):
        return dict(commit=None, uncommitted_changes=None)


def _time(value):
    return '' if value is None or not math.isfinite(value) else '%.6f' % value


def _number(value):
    if value is None:
        return ''
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    value = float(value)
    return '%.7g' % value if math.isfinite(value) else ('nan' if math.isnan(value) else '%sinf' % '-'[:value < 0])


def _vector(value, size=3):
    return [''] * size if value is None else [_number(v) for v in value]


class PushcartDataRecorder:
    def __init__(self, suffix, *, mode, interface, config=None, root=DATA_ROOT,
                 ros=False, ports=CONTROL_PORTS, clock=time.monotonic):
        if suffix not in MODES:
            raise ValueError('Push-cart data suffix must be web or nav')
        self.suffix, self.mode, self.interface, self.config = suffix, mode, interface, config
        self.root, self.ports, self.clock = Path(root), dict(ports), clock
        self.expected = ['ros'] if ros else []
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(('127.0.0.1', 0))
        self.sock.setblocking(False)
        self.queue = queue.Queue(maxsize=512)
        self.dropped_ticks = 0
        self.episode = None
        self.finishing = []
        self.preroll = deque()
        self.seq = 0
        self.last_send = -math.inf
        self.last_status = '推车启用后自动记录到 Data/'
        self.warning = ''
        self.closing = False
        self.thread = threading.Thread(target=self._run, daemon=True, name='pushcart-data')
        self.thread.start()

    def attach_camera(self, camera_process):
        """Record tag results in the camera process; set before it starts."""
        if camera_process is not None and self.mode != 'sim':
            camera_process.data_control_port = self.ports['camera']
            if 'camera' not in self.expected:
                self.expected.append('camera')

    # Control thread: one nonblocking put per tick.
    def update(self, data, command_time, feedback=None):
        try:
            # State.decode creates a new snapshot for every packet. Keep the
            # exact one used by this inference; never re-read live feedback.
            self.queue.put_nowait((command_time, data, feedback))
        except queue.Full:
            self.dropped_ticks += 1

    def status_text(self):
        episode = self.episode
        text = self.last_status
        if episode is not None and episode['stop'] is None:
            text = '正在记录 Data/%s（%.0f s）' % (episode['name'], self.clock() - episode['start'])
        return text + ('；' + self.warning if self.warning else '')

    def close(self):
        self.closing = True
        self.thread.join(timeout=ACK_TIMEOUT_S + 5)
        self.sock.close()

    # Background thread.
    def _run(self):
        while True:
            try:
                item = self.queue.get(timeout=.02)
            except queue.Empty:
                item = None
            now = self.clock()
            try:
                if item is not None:
                    self._tick(*item)
                if self.closing and self.queue.empty() and self.episode is not None and self.episode['stop'] is None:
                    self._stop(self.episode, now, '程序退出')
                self._advance(now)
                self._send(now)
                self._receive(now)
            except Exception as exc:
                self.warning = '数据记录异常：%r' % exc
                print('PUSHCART DATA warning: ' + self.warning, flush=True)
            if self.closing and self.queue.empty() and self.episode is None and not self.finishing:
                break

    def _tick(self, t, data, feedback):
        enabled = bool((data.get('high_policy') or {}).get('enabled'))
        episode = self.episode
        if enabled and (episode is None or episode['stop'] is not None) and not self.closing:
            if episode is not None:
                self._end(episode, self.clock())  # re-enabled during the post-roll
            self._start(t, data)
        elif not enabled and episode is not None and episode['stop'] is None:
            self._stop(episode, t, '推车已关闭')
        episode = self.episode
        if episode is not None and t >= episode['start'] - PRE_S and (
                episode['stop'] is None or t <= episode['stop'] + POST_S):
            self._write(episode, t, data, feedback)
            return
        self.preroll.append((t, data, feedback))
        while self.preroll and self.preroll[0][0] < t - PRE_S - .5:
            self.preroll.popleft()

    def _start(self, t, data):
        name = datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '_' + self.suffix
        path = self.root / name
        for index in range(2, 100):
            if not path.exists():
                break
            path = self.root / ('%s_%d' % (name, index))
        try:
            path.mkdir(parents=True)
        except OSError as exc:
            self.warning = '无法创建记录目录：%s' % exc
            print('PUSHCART DATA warning: ' + self.warning, flush=True)
            return
        files = {key: (path / key).open('x', newline='', buffering=1 << 16)
                 for key in LOCAL_COLUMNS}
        writers = {key: csv.writer(stream) for key, stream in files.items()}
        for key, columns in LOCAL_COLUMNS.items():
            writers[key].writerow(columns)
        episode = dict(path=path, name=path.name, start=t, stop=None, end=None, files=files, writers=writers,
                       rows=0, first_t=None, last_t=None, acks={}, closed=set(), stop_reason=None,
                       last_flush=self.clock(), dropped_at_open=self.dropped_ticks)
        episode['meta'] = self._metadata(episode, data)
        self.episode = episode
        self._write_meta(episode)
        buffered, self.preroll = self.preroll, deque()
        for sample_t, sample, sample_feedback in buffered:
            if sample_t >= t - PRE_S:
                self._write(episode, sample_t, sample, sample_feedback)
        self.last_send = -math.inf
        self.warning = ''
        print('PUSHCART DATA episode started %s' % path, flush=True)

    def _stop(self, episode, t, reason):
        episode['stop'], episode['stop_reason'] = t, reason
        episode['meta']['episode'].update(stop_monotonic=t, stop_reason=reason)
        self.last_send = -math.inf
        print('PUSHCART DATA episode stopped %s (%s); post-roll %.1f s' % (episode['path'], reason, POST_S),
              flush=True)

    def _advance(self, now):
        episode = self.episode
        if episode is not None:
            if now - episode['last_flush'] > .5:
                for stream in episode['files'].values():
                    stream.flush()
                episode['last_flush'] = now
            if episode['stop'] is not None and (
                    # Program exit: end once the recorders have been sent the stop time.
                    episode.get('stop_sent') if self.closing else now > episode['stop'] + POST_S + CLOSE_MARGIN_S):
                self._end(episode, now)
            elif episode['stop'] is None and now - episode['start'] > 2. and not episode.get('checked'):
                episode['checked'] = True
                missing = [name for name in self.expected if name not in episode['acks']]
                if missing:
                    self.warning = '未收到记录程序响应：' + ', '.join(missing)
                    print('PUSHCART DATA warning: %s (%s)' % (self.warning, episode['path']), flush=True)
        for episode in list(self.finishing):
            closed = all(name in episode['closed'] for name in self.expected)
            if closed or now - episode['end'] > ACK_TIMEOUT_S:
                self.finishing.remove(episode)
                self._finalize(episode)

    def _end(self, episode, now):
        for stream in episode['files'].values():
            stream.close()
        summary = dict(stream='motor', source='policy_feedback', close_reason=episode['stop_reason'],
                       rows=episode['rows'], first_t=episode['first_t'], last_t=episode['last_t'],
                       files={name: dict(rows=episode['rows'], first_t=episode['first_t'], last_t=episode['last_t'])
                              for name in ('motor_state.csv', 'hand_left_state.csv', 'hand_right_state.csv')},
                       dropped_before_write=self.dropped_ticks - episode['dropped_at_open'],
                       window=dict(start=episode['start'], stop=episode['stop'], pre=PRE_S, post=POST_S))
        (episode['path'] / 'motor_summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n')
        episode['end'] = now
        self.episode = None
        self.finishing.append(episode)
        self.last_send = -math.inf

    def _finalize(self, episode):
        meta = episode['meta']
        meta['status'] = 'complete' if all(name in episode['closed'] for name in self.expected) else 'incomplete'
        meta['clock']['pairs'].append(_clock_pair())
        meta['streams'] = {
            name: dict(acknowledged=name in episode['acks'], files_closed=name in episode['closed'],
                       rows_reported=episode['acks'].get(name, {}).get('rows'),
                       dropped_reported=episode['acks'].get(name, {}).get('dropped'))
            for name in self.expected}
        meta['streams']['policy'] = dict(rows=episode['rows'], dropped_ticks_total=self.dropped_ticks)
        meta['streams']['motor'] = dict(source='policy_feedback', files_closed=True, rows=episode['rows'],
                                        dropped_ticks_total=self.dropped_ticks)
        meta['files'] = {name: FILES[name] for name in FILES if (episode['path'] / name).exists()}
        self._write_meta(episode)
        self.last_status = '上次记录已保存：Data/%s（%s）' % (
            episode['name'], '完整' if meta['status'] == 'complete' else '部分数据流未确认，见 meta.json')
        print('PUSHCART DATA episode saved %s status=%s' % (episode['path'], meta['status']), flush=True)

    def _send(self, now):
        if not self.expected:
            if self.episode is not None:
                self.episode['stop_sent'] = self.episode['stop'] is not None
            return
        if now - self.last_send < CONTROL_PERIOD_S:
            return
        self.last_send = now
        self.seq += 1
        episode = self.episode
        if episode is None:
            message = encode(self.seq, None)
        else:
            message = encode(self.seq, episode['path'], episode['start'],
                             math.nan if episode['stop'] is None else episode['stop'],
                             PRE_S, 0. if self.closing else POST_S)
            episode['stop_sent'] = episode['stop'] is not None
        for name in self.expected:
            try:
                self.sock.sendto(message, ('127.0.0.1', self.ports[name]))
            except OSError:
                pass

    def _receive(self, now):
        for _ in range(64):
            try:
                data, _ = self.sock.recvfrom(512)
            except (BlockingIOError, OSError):
                break
            ack = decode_ack(data)
            if ack is None:
                continue
            for episode in ([self.episode] if self.episode else []) + self.finishing:
                if ack['directory'] == str(episode['path']):
                    episode['acks'][ack['name']] = ack
                elif episode['end'] is not None and ack['directory'] != str(episode['path']):
                    episode['closed'].add(ack['name'])

    def _metadata(self, episode, data):
        high = data.get('high_policy') or {}
        camera = high.get('camera') or {}
        artifacts = getattr(self.config, 'high_policy_metadata', None) or {}
        pair = _clock_pair()
        return dict(
            schema='pushcart_data_v2', status='recording', directory=episode['name'],
            input_mode=self.suffix, input_mode_description=MODES[self.suffix],
            control_mode=self.mode, network_interface=self.interface, host=socket.gethostname(),
            created_local=pair['local'],
            episode=dict(start_monotonic=episode['start'], stop_monotonic=None, stop_reason=None,
                         pre_s=PRE_S, post_s=POST_S,
                         definition='start = 推车策略启用的控制拍，stop = 推车关闭的控制拍；'
                                    '每个文件只保存时间戳在 [start - pre_s, stop + post_s] 内的样本'),
            clock=dict(time_base='所有 CSV 的 t 列均为本机 CLOCK_MONOTONIC 秒；t_rel = t - start_monotonic',
                       to_wall_time='realtime_unix ≈ t + (pair.realtime_unix - pair.monotonic)',
                       pairs=[pair]),
            motor_sampling=dict(source='本次策略推理所用的控制桥接反馈',
                                frequency='每次完成推理保存一行；无推理的控制拍不写入',
                                nominal_hz=getattr(self.config, 'deployment', {}).get('policy_hz', 50),
                                timestamp='t 与 policy.csv 同拍；state_received_t/hand_received_t 是原始接收时间',
                                missing='未收到的反馈或未提供的扩展字段留空，仿真不生成虚构的力矩、电压或压力'),
            model_set=data.get('model_set'),
            high_policy=dict(model_release=high.get('model_release'),
                             student_path=artifacts.get('student_path'),
                             student_sha256=artifacts.get('student_sha256'),
                             task_limits=high.get('task_limits'),
                             camera_pose_source=camera.get('pose_source'),
                             camera_yaw_offset_rad=camera.get('yaw_offset_rad'),
                             zero_calibration=high.get('zero_calibration')),
            joint_names=dict(body_sdk29=list(JOINT_NAMES), dex3=list(HAND_NAMES),
                             sent=list(SENT_NAMES)),
            units=dict(angle='rad', angular_velocity='rad/s', linear_velocity='m/s', torque='N·m (tau_est, 电机估计值)',
                       voltage='V', current='A (Dex3 power_a，整只手供电电流)', time='s'),
            not_recorded='相机视频与逐帧图像、电机电流（SDK 无此字段）、电机温度、电机错误码（按要求不记录）',
            streams=dict(expected=list(self.expected)),
            git=_git_state())

    @staticmethod
    def _write_meta(episode):
        path = episode['path'] / 'meta.json'
        temporary = path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(episode['meta'], indent=2, ensure_ascii=False, default=_json_value) + '\n')
        temporary.replace(path)

    def _write(self, episode, t, data, feedback):
        if data.get('policy_inference_monotonic') is None:
            return
        high = data.get('high_policy') or {}
        inputs = high.get('inference_inputs') or {}
        camera = high.get('camera') or {}
        hold = high.get('heading_hold') or {}
        state = data.get('state') or {}
        heading = data.get('heading') or {}
        timing = data.get('timing') or {}
        tick = state.get('tick') or (None, None)
        start = episode['start']
        captured = inputs.get('camera_captured_monotonic', camera.get('captured_monotonic'))
        episode['writers']['policy.csv'].writerow((
            _time(t), _time(t - start), _time(state.get('received')), _time(data.get('policy_inference_monotonic')),
            _time(captured), tick[0], tick[1],
            data.get('stage'), int(bool(high.get('enabled'))), int(bool(high.get('motion_blocked'))),
            high.get('inference_role') or '', data.get('selected_model'), heading.get('command_source', ''),
            int(bool(data.get('decelerating'))),
            *_vector(high.get('raw_task_command')), *_vector(high.get('task_command')),
            *_vector(high.get('task_reference')), *_vector(high.get('homie_command')),
            *_vector(data.get('velocity')), *_vector(data.get('filtered_velocity')),
            _number(inputs.get('camera_yaw_rad', camera.get('yaw_rad'))), _number(camera.get('raw_yaw_rad')),
            _number(camera.get('yaw_offset_rad')), int(bool(camera.get('fresh'))), camera.get('status', ''),
            _number(camera.get('age_s')), ';'.join(map(str, camera.get('tag_ids') or [])),
            camera.get('pose_source', ''),
            _number(hold.get('cart_yaw_rad')), _number(hold.get('imu_yaw_rad')),
            _number(hold.get('imu_yaw_at_camera_rad')), _number(hold.get('ref')), _number(hold.get('err')),
            _number(hold.get('corr')),
            _number(timing.get('inference_ms')), _number(timing.get('high_policy_inference_ms')),
            _number(timing.get('policy_hz')), _number(timing.get('state_age_ms'))))
        comparison = data.get('joint_comparison') or {}
        sent = comparison.get('command_rad') or [None] * len(SENT_NAMES)
        sent_ns = comparison.get('sent_ns') or [0, 0, 0]
        episode['writers']['joint_command.csv'].writerow((
            _time(t), _time(t - start), _time(state.get('received')),
            data.get('pd_profile', ''), _number(data.get('pd_transition_blend')), data.get('upper_pd_mode', ''),
            *_vector(data.get('requested_target'), 29), *_vector(data.get('target'), 29),
            *_vector(data.get('kp'), 29), *_vector(data.get('kd'), 29),
            *(_number(v) for v in sent), *(_time(ns * 1e-9) if ns else '' for ns in sent_ns)))
        self._write_feedback(episode, t, data, feedback)
        episode['rows'] += 1
        if episode['first_t'] is None:
            episode['first_t'] = t
        episode['last_t'] = t

    def _write_feedback(self, episode, t, data, feedback):
        state = data.get('state') or {}
        dex3 = data.get('dex3') or {}
        body, hands = motor_telemetry(getattr(feedback, 'motor_telemetry', b''))
        tick = state.get('tick') or (None, None)
        inference_t = _time(data['policy_inference_monotonic'])
        times = (_time(t), _time(t - episode['start']))
        episode['writers']['motor_state.csv'].writerow((
            *times, _time(state.get('received')), inference_t, *tick,
            body.get('tick'), body.get('mode_pr'), getattr(feedback, 'mode_machine', None),
            *_vector(state.get('q'), 29), *_vector(state.get('dq'), 29),
            *_vector(body.get('ddq'), 29), *_vector(body.get('tau_est'), 29),
            *_vector(body.get('vol'), 29), *_vector(body.get('mode'), 29),
            *_vector(state.get('quat'), 4), *_vector(state.get('omega')),
            *_vector(body.get('acc')), *_vector(getattr(feedback, 'rpy', None))))
        received_ns = getattr(feedback, 'hand_received_ns', (0, 0))
        for index, side in enumerate(('left', 'right')):
            hand = hands[index]
            received = received_ns[index] * 1e-9 if received_ns[index] else None
            source = hand.get('source')
            topic = ('rt/%sdex3/%s/state' % ('lf/' if source == 0 else '', side)
                     if source in (0, 1) else '')
            q = dex3.get('q')
            dq = dex3.get('dq')
            episode['writers']['hand_%s_state.csv' % side].writerow((
                *times, _time(received), inference_t, int(bool(dex3.get(side + '_healthy'))), topic,
                *_vector(q[index * 7:(index + 1) * 7] if received is not None and q is not None else None, 7),
                *_vector(dq[index * 7:(index + 1) * 7] if received is not None and dq is not None else None, 7),
                *_vector(hand.get('ddq'), 7), *_vector(hand.get('tau_est'), 7),
                *_vector(hand.get('power'), 4), hand.get('press_sensor_count'),
                *_vector(hand.get('pressure'), 108)))
