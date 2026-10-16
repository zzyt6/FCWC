"""只依赖 NumPy/ONNX Runtime 的 456→12 推理与实机观测适配。"""
import hashlib
import json
from pathlib import Path
import numpy as np


def read_bundle(root, config_path=None):
    root = Path(root)
    path = Path(config_path) if config_path else root / 'config/deploy.json'
    raw = path.read_bytes()
    cfg = json.loads(raw)
    contract = json.loads((root / cfg['contract_file']).read_text())
    if cfg['schema'] != 1 or contract['input']['shape'] != [1, 456] or contract['output']['shape'] != [1, 12]:
        raise ValueError('Unsupported configuration/model contract')
    if cfg['policy_hz'] != 50 or cfg['servo_hz'] != 200:
        raise ValueError('Unexpected control frequency')
    if not isinstance(cfg.get('require_remote_heartbeat'), bool):
        raise ValueError('require_remote_heartbeat must be a boolean')
    return cfg, contract, int(hashlib.sha256(raw).hexdigest()[:16], 16)


class Policy:
    def __init__(self, root, cfg):
        import onnxruntime as ort
        self.root = Path(root)
        manifest = json.loads((self.root / 'models/manifest.json').read_text())
        for name, digest in manifest['artifact_sha256'].items():
            actual = hashlib.sha256((self.root / 'models' / name).read_bytes()).hexdigest()
            if actual != digest:
                raise ValueError('Model artifact hash mismatch: ' + name)
        if cfg['model_file'] != 'models/policy.onnx':
            raise ValueError('This package is pinned to its verified model')
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(self.root / cfg['model_file']),
                                           sess_options=options, providers=['CPUExecutionProvider'])
        inp, out = self.session.get_inputs(), self.session.get_outputs()
        if len(inp) != 1 or len(out) != 1 or inp[0].shape[-1] != 456 or out[0].shape[-1] != 12:
            raise ValueError('ONNX interface mismatch')
        self.input_name = inp[0].name

    def __call__(self, observations):
        obs = np.asarray(observations, dtype=np.float32).reshape(1, 456)
        if not np.isfinite(obs).all():
            raise ValueError('Non-finite observations')
        actions = self.session.run(None, {self.input_name: obs})[0]
        if actions.shape != (1, 12) or not np.isfinite(actions).all():
            raise ValueError('Invalid policy output')
        return actions[0]


class Observations:
    def __init__(self, contract):
        self.contract = contract
        self.indices = np.asarray(contract['sdk29_indices'])
        self.default = np.asarray(contract['default_joint_positions'], dtype=np.float32)
        self.history = np.zeros((6, 76), dtype=np.float32)
        self.frames = 0

    def reset(self):
        self.history.fill(0)
        self.frames = 0

    def push(self, state, velocity, height):
        q = np.asarray(state.quaternion, dtype=np.float64)
        norm = np.linalg.norm(q)
        if not .8 < norm < 1.2:
            raise ValueError('Invalid IMU quaternion norm')
        q /= norm
        w, x, y, z = q
        # Unitree 的四元数排列为 wxyz；计算 R^T [0,0,-1]，不是世界系线加速度。
        gravity = [2*(w*y-x*z), -2*(y*z+w*x), -(1-2*x*x-2*y*y)]
        spec = self.contract['input']
        scales = spec['observation_scales']
        frame = np.concatenate((
            np.r_[velocity, height] * np.asarray(spec['command_scale']),
            state.omega * scales['ang_vel'], gravity,
            (state.q[self.indices] - self.default) * scales['dof_pos'],
            state.dq[self.indices] * scales['dof_vel'],
            (state.applied_q[:12] - self.default[:12]) / self.contract['output']['action_scale'],
        )).astype(np.float32)
        if frame.shape != (76,) or not np.isfinite(frame).all():
            raise ValueError('Invalid observation frame')
        bound = spec['clip_observations']
        self.history[:-1] = self.history[1:]
        self.history[-1] = np.clip(frame, -bound, bound)
        self.frames += 1
        return self.history.reshape(1, 456)


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
        # L1 为运动命令使能；松开时仍平滑降到零。B 立即清零速度命令。
        if not state.buttons & L1 or state.buttons & B:
            target.fill(0)
        if state.buttons & B:
            self.value.fill(0)
        maximum = np.asarray(self.cfg['command_slew_per_s']) * min(dt, .04)
        self.value += np.clip(target-self.value, -maximum, maximum)
        return self.value.copy()


def targets(actions, state, cfg, contract):
    q = state.applied_q.copy()
    bounds = np.asarray([cfg['joint_lower_sdk29'], cfg['joint_upper_sdk29']])
    a = np.clip(actions, -contract['output']['training_clip_actions'],
                contract['output']['training_clip_actions'])
    q[:12] = np.asarray(contract['default_joint_positions'][:12]) + contract['output']['action_scale'] * a
    return np.clip(q, bounds[0], bounds[1]).astype(np.float32)
