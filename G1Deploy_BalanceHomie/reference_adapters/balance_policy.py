"""GR00T Decoupled WBC 的 516→15 双模型部署；运行时仅需 NumPy/ORT。"""
import hashlib
import json
from pathlib import Path
import numpy as np

PROFILE = 'decoupled_wbc_balance_walk_516_15'
SCHEMA = PROFILE + '_v1'


def read_bundle(root, config_path=None):
    root = Path(root)
    path = Path(config_path) if config_path else root / 'config/deploy.json'
    raw = path.read_bytes()
    cfg = json.loads(raw)
    contract = json.loads((root / cfg['contract_file']).read_text())
    if (cfg['schema'] != 1 or cfg['policy_profile'] != PROFILE or
            contract['schema'] != SCHEMA or cfg['controlled_dofs'] != 15 or
            contract['input']['shape'] != [1, 516] or contract['output']['shape'] != [1, 15]):
        raise ValueError('This package requires the Decoupled WBC 516→15 contract')
    if not isinstance(cfg.get('require_remote_heartbeat'), bool):
        raise ValueError('require_remote_heartbeat must be a boolean')
    if cfg['policy_hz'] != 50 or cfg['servo_hz'] != 200:
        raise ValueError('Unexpected control frequency')
    if cfg['model_switch_command_norm'] != contract['model_switch_command_norm']:
        raise ValueError('Model selection threshold does not match upstream contract')
    if cfg['upper_body_mode'] != 'hold_current' or cfg['waist_mode'] != 'policy':
        raise ValueError('Expected policy-controlled waist and held arms')
    if cfg['waist_pitch_kd_policy'] != 'positive_config_value' or min(cfg['kd_sdk29']) < 0:
        raise ValueError('Negative motor damping is not enabled in this package')
    if cfg['default_q_sdk29'] != contract['default_joint_positions']:
        raise ValueError('Calibration pose and model default pose differ')
    rpy = np.asarray(cfg['torso_rpy_command'], dtype=float)
    if rpy.shape != (3,) or not np.isfinite(rpy).all() or np.max(np.abs(rpy)) > .3:
        raise ValueError('Invalid torso RPY command')
    return cfg, contract, int(hashlib.sha256(raw).hexdigest()[:16], 16)


class Policy:
    def __init__(self, root, cfg, policy_mode='auto'):
        if policy_mode not in ('auto', 'walk'):
            raise ValueError('policy_mode must be auto or walk')
        self.policy_mode = policy_mode
        import onnxruntime as ort
        self.root = Path(root)
        self.manifest = json.loads((self.root / 'models/manifest.json').read_text())
        if self.manifest['policy_profile'] != PROFILE:
            raise ValueError('Model/config profile mismatch')
        self.source_sha256 = self.manifest['model_sha256']
        for name, digest in self.manifest['artifact_sha256'].items():
            if hashlib.sha256((self.root / name).read_bytes()).hexdigest() != digest:
                raise ValueError('Artifact hash mismatch: ' + name)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.sessions = {}
        for mode in ('balance', 'walk'):
            name = cfg['model_files'][mode]
            if name != self.manifest['model_files'][mode]:
                raise ValueError('Model path differs from verified manifest')
            session = ort.InferenceSession(str(self.root / name), sess_options=options,
                                           providers=['CPUExecutionProvider'])
            inp, out = session.get_inputs(), session.get_outputs()
            if (len(inp) != 1 or len(out) != 1 or inp[0].shape[-1] != 516 or
                    out[0].shape[-1] != 15 or inp[0].type != 'tensor(float)'):
                raise ValueError('ONNX interface mismatch: ' + mode)
            self.sessions[mode] = session
        self.threshold = cfg['model_switch_command_norm']
        self.selected = 'walk' if policy_mode == 'walk' else 'balance'

    def __call__(self, observations, velocity):
        obs = np.asarray(observations, dtype=np.float32).reshape(1, 516)
        velocity = np.asarray(velocity, dtype=np.float32)
        if velocity.shape != (3,) or not np.isfinite(velocity).all() or not np.isfinite(obs).all():
            raise ValueError('Non-finite or malformed policy input')
        # auto 遵循上游阈值；walk 用于所有速度（含零速度）的单模型对比。
        # 只选择网络，不修改速度命令或观测输入。
        self.selected = ('walk' if self.policy_mode == 'walk' else
                         'balance' if np.linalg.norm(velocity) < self.threshold else 'walk')
        session = self.sessions[self.selected]
        actions = session.run(None, {session.get_inputs()[0].name: obs})[0]
        if actions.shape != (1, 15) or not np.isfinite(actions).all():
            raise ValueError('Invalid policy output')
        return actions[0]


class Observations:
    def __init__(self, contract):
        self.contract = contract
        self.indices = np.asarray(contract['sdk29_indices'])
        self.default = np.asarray(contract['default_joint_positions'], dtype=np.float32)
        self.history = np.zeros((6, 86), dtype=np.float32)
        self.previous_action = np.zeros(15, dtype=np.float32)
        self.frames = 0

    def reset(self):
        self.history.fill(0)
        self.previous_action.fill(0)
        self.frames = 0

    def record_action(self, action):
        values = np.asarray(action, dtype=np.float32)
        if values.shape != (15,) or not np.isfinite(values).all():
            raise ValueError('Invalid previous action')
        # 上游使用原始网络动作，包括停用策略但继续推理的阶段；不反算限幅后的角度。
        self.previous_action[:] = values

    def push(self, state, velocity, height, torso_rpy):
        quat = np.asarray(state.quaternion, dtype=np.float64)
        norm = np.linalg.norm(quat)
        if not .8 < norm < 1.2:
            raise ValueError('Invalid IMU quaternion norm')
        w, x, y, z = quat / norm
        gravity = [2*(w*y-x*z), -2*(y*z+w*x), -(1-2*x*x-2*y*y)]
        spec = self.contract['input']
        scales = spec['observation_scales']
        frame = np.concatenate((
            np.asarray(velocity) * np.asarray(spec['command_scale']), [height],
            torso_rpy, state.omega * scales['ang_vel'], gravity,
            (state.q[self.indices] - self.default) * scales['dof_pos'],
            state.dq[self.indices] * scales['dof_vel'], self.previous_action,
        )).astype(np.float32)
        if frame.shape != (86,) or not np.isfinite(frame).all():
            raise ValueError('Invalid observation frame')
        self.history[:-1] = self.history[1:]
        self.history[-1] = frame
        self.frames += 1
        return self.history.reshape(1, 516)


class Commands:
    def __init__(self, cfg):
        self.cfg = cfg
        self.value = np.zeros(3, dtype=np.float32)

    def update(self, state, dt):
        from .protocol import L1, B
        limits = np.asarray(self.cfg['command_limits'])
        lx, ly, rx, _ = state.sticks
        sticks = np.asarray([ly, -lx, -rx])
        deadzone = self.cfg['stick_deadzone']
        sticks = np.where(np.abs(sticks) < deadzone, 0,
                          np.sign(sticks) * (np.abs(sticks)-deadzone)/(1-deadzone))
        target = np.where(sticks >= 0, sticks * limits[:, 1], -sticks * limits[:, 0])
        if not state.buttons & L1 or state.buttons & B:
            target.fill(0)
        if state.buttons & B:
            self.value.fill(0)
        maximum = np.asarray(self.cfg['command_slew_per_s']) * min(dt, .04)
        self.value += np.clip(target-self.value, -maximum, maximum)
        return self.value.copy()


def targets(actions, state, cfg, contract):
    action = np.asarray(actions, dtype=np.float32)
    if action.shape != (15,) or not np.isfinite(action).all():
        raise ValueError('Expected 15 finite policy actions')
    q = state.applied_q.copy()
    q[:15] = (np.asarray(contract['default_joint_positions'][:15]) +
              contract['output']['action_scale'] * action)
    return np.clip(q, cfg['joint_lower_sdk29'], cfg['joint_upper_sdk29']).astype(np.float32)
