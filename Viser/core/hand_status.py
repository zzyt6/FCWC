"""Explain SDK/feedback readiness; position excursions are separate warnings."""
import numpy as np

def feedback_reasons(state, now):
    reasons = []
    for side, name in enumerate(('左手', '右手')):
        if not state.hand_flags & 32:
            reasons.append(f'{name} Dex3 控制未启用')
            continue
        if state.hand_flags & (1 << side):
            reasons.append('')
            continue
        received = state.hand_received_ns[side]
        age = now - received * 1e-9
        if not received:
            reasons.append(f'{name}尚未收到关节反馈')
        elif age < 0:
            reasons.append(f'{name}反馈时间异常')
        elif age > .2:
            reasons.append(f'{name}反馈已过期（{age*1000:.0f} ms，限值 200 ms）')
        else:
            q = state.hand_q[side*7:side*7+7]
            dq = state.hand_dq[side*7:side*7+7]
            if not np.isfinite(q).all() or not np.isfinite(dq).all():
                reasons.append(f'{name}关节反馈包含无效数值')
            else:
                # Motor/temperature faults and malformed DDS samples are kept
                # in bridge.log; their cause cannot be inferred from q alone.
                reasons.append(f'{name}反馈被底层判为异常，详见本次 bridge.log 的 DEX3 HEALTH')
    return reasons
