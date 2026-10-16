"""Compare the last successful bridge write with the latest received feedback."""
import math

import numpy as np

from core.robot import ACTIVE, JOINT_NAMES, HAND_JOINTS

INDICES = tuple(map(int, ACTIVE)) + tuple(range(29, 43))
NAMES = tuple(JOINT_NAMES[i] for i in ACTIVE) + tuple(
    f'{side}_hand_{joint}_joint' for side in ('left', 'right') for joint in HAND_JOINTS)
GROUPS = (0,) * len(ACTIVE) + (1,) * 7 + (2,) * 7
OPTIONS = tuple(f'{i+1:02d} · {name}' for i, name in enumerate(NAMES))


def joint_label(name):
    """Chinese joint label shared by the monitor table and fault messages."""
    label = name.removesuffix('_joint').replace('left_', '左').replace('right_', '右')
    for source, target in (('hand_', ''), ('shoulder_', '肩'), ('ankle_', '踝'),
                           ('wrist_', '腕'), ('hip_', '髋'), ('waist_', '腰'),
                           ('elbow', '肘'), ('knee', '膝'), ('thumb_', '拇指 '),
                           ('middle_', '中指 '), ('index_', '食指 '),
                           ('pitch', '俯仰'), ('roll', '横滚'), ('yaw', '偏航')):
        label = label.replace(source, target)
    return label


def metadata():
    return dict(joint_names=NAMES, groups=('body', 'left_dex3', 'right_dex3'),
                joint_groups=GROUPS, angle_unit='rad', error='command_rad - actual_rad',
                command='Last successful local DDS Write, after interpolation/compensation; '
                        'not a motor acknowledgement. Synthetic in sim mode.',
                timestamps='Host monotonic nanoseconds; sent_ns and feedback_ns are per group. '
                           'Latest available pair, not synchronized motor execution times.',
                validity='valid requires fresh successful write, healthy fresh feedback and Kp > 0; '
                         'otherwise error_rad is null. Angles may retain old samples.',
                sampling='One snapshot per Python tick (~50 Hz), not every 200 Hz DDS write')


def status(comparison, index, now_ns):
    group = GROUPS[index]
    sent = comparison['sent_ns'][group]
    received = comparison['feedback_ns'][group]
    timeout = comparison['timeout_s'][group] * 1e9
    if not sent:
        return False, '本程序尚未下发该关节指令'
    if not received:
        return False, '尚未收到该关节反馈'
    if not 0 <= now_ns-sent <= timeout:
        return False, '指令记录已过期，显示最后一次下发值'
    if not 0 <= now_ns-received <= timeout:
        return False, '反馈已过期，显示最后一次实测值'
    if not comparison['feedback_healthy'][group]:
        return False, '反馈异常，暂停计算差值'
    if any(value is None or not math.isfinite(value) for value in
           (comparison['command_rad'][index], comparison['actual_rad'][index])):
        return False, '关节角度无效，暂停计算差值'
    if not comparison['position_control'][index]:
        return False, 'Kp=0，当前未做位置控制（如阻尼状态）'
    return True, '当前角度对比'


def snapshot(state, now, body_timeout):
    now_ns = int(now * 1e9)
    actual = np.concatenate((state.q, state.hand_q))
    received = (state.received_ns, *state.hand_received_ns)
    comparison = dict(
        sampled_ns=now_ns, bridge_ns=state.bridge_ns, simulation=bool(state.flags & 4),
        sent_ns=list(state.sent_ns), feedback_ns=list(received),
        timeout_s=[body_timeout, .2, .2],
        feedback_healthy=[bool(state.flags & 1), bool(state.hand_flags & 1), bool(state.hand_flags & 2)],
        position_control=[bool(state.position_mask & (1 << i)) for i in INDICES],
        command_rad=[float(state.sent_q[i]) if state.sent_ns[g] else None for i, g in zip(INDICES, GROUPS)],
        actual_rad=[float(actual[i]) if received[g] else None for i, g in zip(INDICES, GROUPS)])
    comparison['valid'] = [status(comparison, i, now_ns)[0] for i in range(len(NAMES))]
    comparison['error_rad'] = [c-a if valid else None for c, a, valid in zip(
        comparison['command_rad'], comparison['actual_rad'], comparison['valid'])]
    return comparison


def format_joint(comparison, index, now):
    if not comparison:
        return '等待关节指令与反馈…'
    valid, reason = status(comparison, index, int(now * 1e9))
    command, actual = comparison['command_rad'][index], comparison['actual_rad'][index]
    error = command-actual if valid else None
    lines = [f'**{NAMES[index]}**', '', '**合成反馈（离线联调）**' if comparison['simulation'] else '真机反馈',
             '', reason, '', '| 数值 | rad | ° |', '|---|---:|---:|']
    for label, value in [('下发角度', command), ('实际角度', actual),
                         ('差值（下发 − 实际）', error), ('绝对差值', abs(error) if error is not None else None)]:
        lines.append(f'| {label} | {value:+.5f} | {math.degrees(value):+.3f} |' if value is not None
                     else f'| {label} | — | — |')
    group = GROUPS[index]
    sent, received = comparison['sent_ns'][group], comparison['feedback_ns'][group]
    def age(stamp):
        return f'{(now-stamp*1e-9)*1000:.1f} ms' if stamp else '—'
    lines.extend(['', f'距上次下发：{age(sent)}；距上次反馈：{age(received)}。', '',
                  '下发角度包含平滑插值和所选姿态的补偿；两项取最新样本，时间未严格同步。'])
    return '\n'.join(lines)
