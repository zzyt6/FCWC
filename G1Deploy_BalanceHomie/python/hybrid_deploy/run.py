"""默认只读影子推理；--mode control 才向本机桥接发送请求。"""
import argparse
import json
from pathlib import Path
import signal
import socket
import time
import numpy as np
from . import protocol as p
from .policy import Policy, Observations, Commands, read_bundle, targets
from .configuration import load_config

ROOT = Path(__file__).resolve().parents[2]


def calibration_status(state, cfg, contract):
    goal = np.asarray(cfg['default_q_sdk29'][:cfg['controlled_dofs']], dtype=np.float32)
    position_error = np.abs(state.q[:cfg['controlled_dofs']] - goal)
    speed = np.abs(state.dq[:cfg['controlled_dofs']])
    target_error = np.abs(state.applied_q[:cfg['controlled_dofs']] - goal)
    worst = int(np.argmax(position_error))
    return {
        'max_position_error_rad': float(position_error[worst]),
        'worst_joint': contract['joint_names_sdk29'][worst],
        'max_velocity_rad_s': float(np.max(speed)),
        'max_target_error_rad': float(np.max(target_error)),
        'position_pending': np.flatnonzero(position_error > cfg['calibration_position_tolerance_rad']).tolist(),
        'velocity_pending': np.flatnonzero(speed > cfg['calibration_velocity_tolerance_rad_s']).tolist(),
        'target_pending': np.flatnonzero(target_error >= 1e-4).tolist(),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['shadow', 'control'], default='shadow')
    parser.add_argument('--policy-mode', choices=['auto', 'balance', 'homie'], default='auto',
                        help='auto: Homie at/above command threshold, immediate Balance below; or force one model')
    parser.add_argument('--check-only', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--duration', type=float, default=0)
    parser.add_argument('--log', type=Path, help='optional JSONL telemetry file')
    args = parser.parse_args(argv)
    if not np.isfinite(args.duration) or args.duration < 0:
        parser.error('Invalid duration')
    cfg, contract, token = read_bundle(ROOT)
    policy = Policy(ROOT, cfg, policy_mode=args.policy_mode)
    if args.check_only:
        return 0
    print('POLICY_PROFILE=%s MODEL_SHA256=%s' % (cfg['policy_profile'], policy.source_sha256), flush=True)
    print('POLICY_MODE=%s threshold=%g; independent Balance/Homie PD; transition=%gs' %
          (policy.policy_mode, policy.threshold, cfg['profile_transition_s']), flush=True)
    observations, commands = Observations(contract), Commands(cfg)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # 独占状态端口，同时保证控制请求来自这个进程，避免两个策略争用。
    destination = ('127.0.0.1', cfg['bridge_port'])
    stop = [False]
    def interrupted(signum, frame):
        stop[0] = True
    signal.signal(signal.SIGINT, interrupted)
    signal.signal(signal.SIGTERM, interrupted)
    state, session, sequence = None, None, 0
    request, buttons, previous_stage = p.WAIT, None, None
    started = time.monotonic()
    next_step = started
    last_log = 0.
    log = None
    if args.log:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        _, gains, _ = load_config(ROOT)
        args.log.with_suffix('.metadata.json').write_text(json.dumps({
            'config_token':token, 'config':cfg, 'pd_profiles':gains, 'contracts':contract,
            'policy_mode':policy.policy_mode, 'model_sha256':policy.source_sha256,
            'sampling':'50 Hz policy samples; not all 200 Hz servo commands',
            'unavailable':['base world position','base linear velocity','measured joint torque'],
        }, indent=2)+'\n')
        log = args.log.open('w')
    try:
        sock.bind(('127.0.0.1', cfg['state_port']))
        sock.setblocking(False)
        print('MODE=%s; R2: calibrate, R2 again: policy; L1+sticks: move; B: zero; L2: stop' % args.mode,
              flush=True)
        while not stop[0]:
            now = time.monotonic()
            if args.duration and now-started >= args.duration:
                break
            for _ in range(128):
                try:
                    data, sender = sock.recvfrom(p.STATE.size+1)
                except BlockingIOError:
                    break
                if sender != destination:
                    continue
                candidate = p.State.decode(data)
                if candidate.config_token != token:
                    raise RuntimeError('Config changed: rebuild bridge before running')
                if session is None:
                    session = candidate.session
                if candidate.session != session:
                    raise RuntimeError('Bridge restarted; explicit restart of policy is required')
                if state is None or candidate.sequence >= state.sequence:
                    state = candidate
            now_ns = time.monotonic_ns()
            if state is None or state.sequence == 0:
                if now-started > cfg['startup_timeout_s']:
                    raise RuntimeError('No valid LowState; check interface, DDS, and robot connection')
                time.sleep(.01)
                continue
            age = (now_ns-state.received_ns)*1e-9
            if age < 0 or age > cfg['state_timeout_s']:
                raise RuntimeError('Robot/bridge feedback is stale')
            if args.mode == 'control' and not state.flags & 2:
                raise RuntimeError('Bridge is observe-only; motor control is not armed')
            if state.stage in (p.FAULT, p.DAMPED):
                print('BRIDGE STOPPED: ' + p.STAGE_NAMES[state.stage], flush=True)
                return 2 if state.stage == p.FAULT else 0
            if previous_stage != state.stage:
                print('STAGE ' + p.STAGE_NAMES[state.stage], flush=True)
                if state.stage in (p.WAITING, p.RAMPING, p.HOLDING):
                    observations.reset()
                    policy.reset_selection()
                    commands.value.fill(0)
                previous_stage = state.stage
            if state.buttons & p.L2:
                print('Remote L2: stopping', flush=True)
                break
            edge = buttons is not None and bool(state.buttons & p.R2) and not bool(buttons & p.R2)
            buttons = state.buttons
            if not state.flags & 1:
                if args.mode == 'control' and state.stage != p.WAITING:
                    raise RuntimeError('Feedback health check failed')
                edge = False
            elif args.mode == 'control' and edge:
                if state.stage == p.WAITING:
                    request = p.PREPARE
                elif state.stage == p.HOLDING and observations.frames >= 6:
                    request = p.RUN
            if state.stage == p.HOLDING and request != p.RUN:
                request = p.HOLD
            velocity = commands.update(state, .02) if state.stage == p.POLICY else np.zeros(3)
            policy.select(velocity, state)
            filtered_velocity = velocity.copy()
            if policy.selected == 'balance':
                velocity = np.zeros(3, dtype=np.float32)
            action = np.zeros(12, dtype=np.float32)
            actions = None
            obs = None
            inference_ms = None
            if state.flags & 1 and state.stage != p.RAMPING:
                obs = observations.push(state, filtered_velocity, cfg['height_m'], cfg['torso_rpy_command'])
                # 待机时也可影子推理；只有 RUN 请求会把结果交给电机桥接。
                inference_started = time.monotonic_ns()
                actions = policy(obs, filtered_velocity, state)
                inference_ms = (time.monotonic_ns()-inference_started)*1e-6
                action = actions[policy.selected]
            q = targets(action, state, cfg, contract, policy.selected) if request == p.RUN else state.applied_q
            if args.mode == 'control':
                sequence += 1
                sock.sendto(p.command(state, sequence, request, q,
                                     profile=p.PROFILE_NAMES.index(policy.selected)), destination)
                if request == p.RUN and actions is not None:
                    observations.record_action(actions, policy.selected)
            row = {'elapsed_s': now-started, 'stage': p.STAGE_NAMES[state.stage],
                   'feedback_age_ms': age*1000, 'healthy': bool(state.flags & 1),
                   'remote_heartbeat_required': cfg['require_remote_heartbeat'],
                   'simulation': bool(state.flags & 4), 'velocity_command': velocity.tolist(),
                   'filtered_velocity_command': filtered_velocity.tolist(),
                   'command_source': 'remote', 'web_owner': None,
                   'policy_mode': policy.policy_mode,
                   'selected_model': policy.selected,
                   'pd_profile': p.PROFILE_NAMES[state.profile], 'pd_transition_blend': state.gain_blend,
                   'inference_ms': inference_ms,
                   'action_max_abs': float(np.max(np.abs(action)))}
            if state.stage == p.RAMPING:
                row['calibration'] = calibration_status(state, cfg, contract)
            if log:
                recorded = dict(row, controlled_q_rad=state.q[:cfg['controlled_dofs']].tolist(),
                                controlled_dq_rad_s=state.dq[:cfg['controlled_dofs']].tolist(),
                                controlled_commanded_q_rad=state.applied_q[:cfg['controlled_dofs']].tolist())
                # 记录策略实际使用的输入，便于区分指令偏差与零指令下的策略漂移。
                recorded.update(
                    sample_monotonic_ns=now_ns, lowstate_received_monotonic_ns=state.received_ns,
                    bridge_session=state.session, state_sequence=state.sequence, config_token=token,
                    applied_kp_sdk29=state.applied_kp.tolist(), applied_kd_sdk29=state.applied_kd.tolist(),
                    requested_q_sdk29_rad=q.tolist() if request == p.RUN else None,
                    q_sdk29_rad=state.q.tolist(), dq_sdk29_rad_s=state.dq.tolist(),
                    commanded_q_sdk29_rad=state.applied_q.tolist(),
                    imu_quaternion_wxyz=state.quaternion.tolist(), imu_rpy_rad=state.rpy.tolist(),
                    imu_angular_velocity_rad_s=state.omega.tolist(),
                    remote_buttons=state.buttons, remote_sticks_lx_ly_rx_ry=state.sticks.tolist(),
                    height_command_m=cfg['height_m'], torso_rpy_command_rad=cfg['torso_rpy_command'],
                    policy_action=action.tolist() if obs is not None else None,
                    policy_lower_body_target_rad=q[:cfg['controlled_dofs']].tolist() if request == p.RUN else None,
                    observation_frame=(obs[policy.selected][0, -76:].tolist()
                                       if obs is not None else None),
                    model_observations={name:value[0].tolist() for name,value in obs.items()} if obs is not None else None,
                    model_actions={name:value.tolist() for name,value in actions.items()} if actions is not None else None,
                    observation_frames=observations.frames,
                )
                log.write(json.dumps(recorded)+'\n')
                log.flush()
            if now-last_log >= 1:
                print(json.dumps(row), flush=True)
                last_log = now
            next_step += .02
            if next_step < time.monotonic():
                next_step = time.monotonic()
            time.sleep(max(0, next_step-time.monotonic()))
        return 0
    finally:
        if args.mode == 'control' and state is not None:
            # C++ 看门狗仍独立运行；正常退出时提前发送显式停止。
            try:
                for _ in range(3):
                    sequence += 1
                    sock.sendto(p.command(state, sequence, p.STOP, state.applied_q), destination)
                    time.sleep(.01)
            except (OSError, ValueError):
                pass
        if log:
            log.close()
        sock.close()


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        print('POLICY ERROR: %s' % exc, flush=True)
        raise SystemExit(1)
