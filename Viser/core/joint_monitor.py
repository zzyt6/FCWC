"""Joint-limit flags and operator warnings for the left monitor panel.

Limits match the bridge's body and Dex3 position warnings
(deploy.json SDK29 bounds and the Dex3 command bounds, each +/-0.02 rad).
Motor temperature, motor error code and Dex3 error come from the official SDK
LowState_/HandState_ MotorState; the bridge reports them only in bridge.log
(BODY / HEALTH / DEX3 HEALTH / STAGE lines), which is tailed here read-only.
"""
import math
import json
import re
import threading
import time
from pathlib import Path

import numpy as np

from core.joint_tracking import INDICES, NAMES, joint_label
from core.robot import HAND_LOWER, HAND_UPPER, JOINT_NAMES

MARGIN = .02
WARNING_HOLD_S = 5.
MONITOR_CONFIG = Path(__file__).resolve().parents[1] / 'config/joint_monitor.json'


def load_monitor_settings(path=MONITOR_CONFIG):
    settings = json.loads(Path(path).read_text())
    value = settings.get('error_threshold_deg') if isinstance(settings, dict) else None
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError('关节误差阈值 error_threshold_deg 必须为正数，单位 °')
    return {'error_threshold_deg': float(value)}

BODY_REASONS = {
    'motor_temperature_high': '电机温度过高（超过 {limit:g} °C；日志未记录关节与实测温度）',
    'motor_status_nonzero': '电机报错（SDK MotorState.motorstate ≠ 0）',
    'body_tilt_high': '机身倾角过大（IMU roll/pitch 超限）',
    'lowstate_missing_or_stale': '本体 LowState 反馈超时',
    'lowstate_missing_or_invalid': '本体 LowState 反馈缺失或无效',
    'wirelesscontroller_missing_or_stale': '实体遥控器心跳超时',
    'imu_quaternion_nonfinite': 'IMU 四元数无效',
    'imu_quaternion_norm_invalid': 'IMU 四元数模长异常',
    'imu_angular_velocity_nonfinite': 'IMU 角速度无效',
    'imu_rpy_nonfinite': 'IMU 姿态角无效',
    'remote_sticks_invalid': '遥控摇杆数值无效',
}
HAND_REASONS = {
    'no_sample': '未收到反馈', 'stale_over_200ms': '反馈超过 200 ms 未更新',
    'timestamp_in_future': '反馈时间异常', 'invalid_state': '反馈被判为无效',
}
RECOVERY_REASONS = {
    'measured position did not recover in 3 s': '3 秒内未到位，恢复超时',
    'feedback invalid or stale': '手部反馈无效或过期',
    'nonfinite feedback': '手部反馈包含无效数值',
    'command heartbeat timeout': '恢复指令心跳超时',
    'position moved away from recovery target': '手指进一步偏离恢复目标',
    'recovery velocity too high': '手指恢复速度过大',
    'body preparation requested': '已切换到机身准备',
}


def limits(config):
    lower = np.r_[np.asarray(config.deployment['joint_lower_sdk29'], dtype=float), HAND_LOWER.ravel()]
    upper = np.r_[np.asarray(config.deployment['joint_upper_sdk29'], dtype=float), HAND_UPPER.ravel()]
    return lower[list(INDICES)], upper[list(INDICES)]


def out_of_range(comparison, lower, upper):
    """Indices (into NAMES) outside the reference range; never a stop condition."""
    if not comparison:
        return []
    return [i for i, q in enumerate(comparison['actual_rad'])
            if q is not None and (q < lower[i]-MARGIN or q > upper[i]+MARGIN)]


def _body_reason(text, max_temperature):
    text = text.strip()
    if text.split(' ')[0] == 'motor_temperature_high':
        joints = []
        for match in re.finditer(r'\bindex=(\d+) c=(\S+)', text):
            index, temperature = int(match.group(1)), float(match.group(2))
            if not math.isfinite(temperature):
                continue
            if 0 <= index < len(JOINT_NAMES):
                name = JOINT_NAMES[index]
                name = '%s关节（%s，SDK %d）' % (joint_label(name), name, index)
            else:
                name = 'SDK %d' % index
            joints.append('%s %g °C' % (name, temperature))
        if joints:
            limit = re.search(r'\blimit_c=(\S+)', text)
            threshold = float(limit.group(1)) if limit else max_temperature
            return '电机温度过高：%s（阈值 > %g °C）' % ('；'.join(joints), threshold)
    match = re.match(r'joint_position_out_of_bounds index=(\d+) q=(\S+)', text)
    if match:
        index = int(match.group(1))
        name = NAMES[INDICES.index(index)] if index in INDICES else 'SDK %d' % index
        return '%s 位置超限（实测 %.3f rad）' % (name, float(match.group(2)))
    match = re.match(r'joint_feedback_nonfinite index=(\d+)', text)
    if match:
        return 'SDK %s 关节反馈数值无效' % match.group(1)
    base = text.split(' ')[0]
    if base in BODY_REASONS:
        return BODY_REASONS[base].format(limit=max_temperature)
    return text


def _hand_reason(side, text):
    name = '左手' if side == 'left' else '右手'
    text = text.strip()
    match = re.match(r'joint_out_of_bounds index=(\d+) q=(\S+)', text)
    if match:
        joint = NAMES[27 + (side == 'right')*7 + int(match.group(1))]
        return '%s 位置超限（实测 %.3f rad）' % (joint, float(match.group(2)))
    match = re.match(r'motor_temperature index=(\d+) c=(\S+)', text)
    if match:
        joint = NAMES[27 + (side == 'right')*7 + int(match.group(1))]
        return '%s 温度过高（%s °C，SDK MotorState.temperature）' % (joint, match.group(2))
    match = re.match(r'motor_error index=(\d+) code=(\d+)', text)
    if match:
        joint = NAMES[27 + (side == 'right')*7 + int(match.group(1))]
        return '%s 电机报错（SDK motorstate=%s）' % (joint, match.group(2))
    match = re.match(r'hand_error=(\d+)', text)
    if match:
        return 'Dex3 %s报错（SDK HandState.error=%s）' % (name, match.group(1))
    return 'Dex3 %s%s' % (name, HAND_REASONS.get(text, '：' + text))


def parse_bridge_line(line, max_temperature):
    """Return (key, level, text) for warnings in one bridge.log line, else None."""
    if line.startswith('HEALTH false'):
        match = re.search(r'reason=(.*?) lowstate_age_ms=', line)
        if match:
            return 'body', 'error', _body_reason(match.group(1), max_temperature)
    elif line.startswith('HEALTH true'):
        return 'body', None, None
    elif line.startswith('BODY '):
        match = re.match(r'BODY (POSITION|TARGET|EFFORT) (warning|recovered) index=(\d+)(?: (.*))?$', line)
        if match and 0 <= int(match.group(3)) < 29:
            kind, event, index, values = match.groups()
            index = int(index)
            key = 'body_' + kind.lower() + '_' + str(index)
            if event == 'recovered':
                return key, None, None
            name = NAMES[INDICES.index(index)] if index in INDICES else 'SDK %d' % index
            if kind == 'EFFORT':
                return key, 'warning', name + ' PD 力矩限制：仅该关节降低增益，机身控制继续'
            q = re.search(r'(?:^| )q=(\S+)', values or '')
            angle = '（%+.2f°）' % math.degrees(float(q.group(1))) if q else ''
            text = ('位置超限' + angle + '，仅供参考，机身控制继续' if kind == 'POSITION' else
                    '目标角度超出参考范围' + angle + '，已限幅，机身控制继续')
            return key, 'warning', name + ' ' + text
    elif line.startswith('DEX3 HEALTH'):
        side = re.search(r'side=(\w+)', line)
        if side:
            reason = re.search(r'reason=(.*)$', line)
            if not reason:
                return 'dex3_' + side.group(1), None, None
            return 'dex3_' + side.group(1), 'warning', _hand_reason(side.group(1), reason.group(1))
    elif line.startswith('DEX3 CONTROL '):
        if line.startswith('DEX3 CONTROL resumed'):
            return 'dex3_control', None, None
        if line.startswith('DEX3 CONTROL warning'):
            return 'dex3_control', 'warning', 'Dex3 已暂停，机身控制继续；手部通信正常后可点击重新启用并归零'
    elif line.startswith('DEX3 TRANSPORT '):
        match = re.match(r'DEX3 TRANSPORT side=(left|right) operation=(\w+) (warning|recovered)(?: reason=(.*))?', line)
        if match:
            key = 'dex3_transport_' + match.group(1) + '_' + match.group(2)
            if match.group(3) == 'recovered':
                return key, None, None
            return key, 'warning', 'Dex3 %s通信异常（%s）：%s；机身控制继续' % (
                '左手' if match.group(1) == 'left' else '右手', match.group(2), match.group(4))
    elif line.startswith('DEX3 RECOVERY '):
        match = re.match(r'DEX3 RECOVERY (warning|started|completed) joint=(\d+)(?: reason=(.*))?', line)
        if match and 0 <= int(match.group(2)) < 14:
            key = 'dex3_recovery_' + match.group(2)
            if match.group(1) != 'warning':
                return key, None, None
            reason = match.group(3) or '恢复未完成'
            return key, 'warning', '%s 恢复已结束：%s；机身控制继续' % (
                NAMES[27 + int(match.group(2))], RECOVERY_REASONS.get(reason, reason))
    elif line.startswith('STAGE '):
        match = re.match(r'STAGE (\d+) (.*)', line)
        if match and match.group(1) == '5':
            reason = match.group(2)
            feedback = re.match(r'(?:feedback unhealthy|command rejected|Dex3 recovery: body feedback unhealthy): (.*)', reason)
            if feedback:
                reason = _body_reason(feedback.group(1), max_temperature)
            return 'stage', 'error', '底层故障停止：' + reason
        if match:
            return 'stage', None, None
    elif line.startswith('BRIDGE ERROR'):
        return 'bridge_error', 'error', '控制桥接错误：' + line.split(':', 1)[-1].strip()
    return None


class BridgeLogWatcher:
    """Tail the current bridge.log; never writes to it or blocks the bridge."""

    def __init__(self, path, max_temperature):
        self.path = Path(path) if path else None
        self.max_temperature = max_temperature
        self.lock = threading.Lock()
        self.active = {}
        self.offset = 0
        self.partial = ''
        self.identity = None

    def poll(self):
        if self.path is None:
            return
        try:
            stat = self.path.stat()
            identity = (stat.st_dev, stat.st_ino)
            if self.identity != identity or stat.st_size < self.offset:
                self.offset = 0
                self.partial = ''
                with self.lock:
                    self.active.clear()
            self.identity = identity
            with self.path.open('r', errors='replace') as stream:
                stream.seek(self.offset)
                chunk = stream.read(1 << 20)
                self.offset = stream.tell()
        except OSError:
            # A log rotation or unavailable recorder must not stop the viewer.
            return
        lines = (self.partial + chunk).split('\n')
        self.partial = lines.pop()
        with self.lock:
            for line in lines:
                if line.startswith('BRIDGE mode='):
                    # A restarted bridge reports its own health from scratch.
                    self.active.clear()
                    continue
                try:
                    parsed = parse_bridge_line(line, self.max_temperature)
                except (ValueError, IndexError):
                    continue
                if parsed is None:
                    continue
                key, level, text = parsed
                if level is None:
                    self.active.pop(key, None)
                else:
                    self.active[key] = (level, text)

    def warnings(self):
        with self.lock:
            return list(self.active.values())


def degrees(value):
    return math.degrees(value)


class WarningBoard:
    """Merge live checks with bridge.log warnings; keep recent ones briefly visible."""

    def __init__(self, hold_s=WARNING_HOLD_S, clock=time.monotonic):
        self.hold_s = hold_s
        self.clock = clock
        self.seen = {}

    def update(self, current):
        now = self.clock()
        live = set()
        for level, text in current:
            # Changing angles/latencies update one warning instead of flooding
            # the box with a new historical row on every telemetry sample.
            key = text.split(' 位置超限', 1)[0] if ' 位置超限' in text else text
            for prefix in ('本体反馈延迟', 'Dex3 未就绪：', '准备姿态超时',
                           '电机温度过高：', '底层故障停止：电机温度过高：'):
                if text.startswith(prefix):
                    key = prefix
            if key in live:
                continue  # Prefer live measurements over an older bridge log line.
            first = self.seen.get(key, (now,))[0]
            self.seen[key] = (first, now, level, text)
            live.add(key)
        for key, (_, last, _, _) in list(self.seen.items()):
            if now - last > self.hold_s:
                del self.seen[key]
        rows = [(level, text, key in live, first) for key, (first, last, level, text) in self.seen.items()]
        rows.sort(key=lambda row: (row[0] != 'error', not row[2], -row[3]))
        return rows


def live_warnings(data, comparison, lower, upper, stale_s, now=None):
    """Checks computed in Python from the bridge packet already in the snapshot."""
    result = []
    for i in out_of_range(comparison, lower, upper):
        q = comparison['actual_rad'][i]
        bound = lower[i] if q < lower[i] else upper[i]
        result.append(('warning', '%s 位置超限：实测 %+.3f rad（%+.1f°），参考边界 %+.3f rad'
                       % (NAMES[i], q, degrees(q), bound)))
    state = data.get('state')
    if state is not None:
        age = (time.monotonic() if now is None else now) - state['received']
        if not 0 <= age <= stale_s:
            result.append(('warning', '本体反馈延迟 %.0f ms' % (age*1000)))
    stage = data.get('stage')
    if stage == 'PREPARING' and data.get('preparation_warning'):
        result.append(('warning', data['preparation_warning']))
    if stage in ('DAMPED', 'FAULT') and data.get('reason'):
        result.append(('error' if stage == 'FAULT' else 'warning', '已停止：' + data['reason']))
    elif data.get('reason'):
        # Keep actionable controller feedback here after compacting the right panel.
        reason = data['reason']
        if reason not in ('航向参数已应用', '航向目标已设为当前朝向', '等待控制桥接反馈'):
            result.append(('warning', reason))
    hands = data.get('dex3', {})
    if hands.get('blocking_reason'):
        result.append(('warning', 'Dex3 未就绪：' + hands['blocking_reason']))
    if (stage in ('HOLDING', 'POLICY') and hands.get('ever_started')
            and not hands.get('active') and not hands.get('pending')):
        result.append(('warning', 'Dex3 已暂停，机身控制继续；通信正常后可重新启用并归零'))
    for field, label in (('zero_compensation_error', '补偿归零不可用'),
                         ('grasp_compensation_error', '补偿抓握不可用')):
        if hands.get(field):
            result.append(('warning', label + '：' + hands[field]))
    if data.get('recording_error'):
        result.append(('warning', '记录异常：' + data['recording_error']))
    return result
