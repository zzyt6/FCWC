"""Balance/Homie 456/12 inference on the selected policy's execution history."""
import hashlib
import json
from pathlib import Path
import numpy as np
from .configuration import load_config, PROFILES
from . import protocol as p
from .balance_observations import Observations as BalanceObservations
from .homie_observations import Observations as HomieObservations
from .command_filter import filter_command


def read_bundle(root):
    root = Path(root)
    cfg, _, token = load_config(root)
    contracts = {name: json.loads((root/cfg['contract_files'][name]).read_text()) for name in PROFILES}
    for name in PROFILES:
        spec = contracts[name]
        if spec['input']['shape'] != [1,456] or spec['output']['shape'] != [1,12]:
            raise ValueError('Wrong model contract: '+name)
        if spec['input'].get('history_frames',spec['input'].get('history_len')) != 6 or spec['output']['action_scale'] != .25:
            raise ValueError('Unsupported history or action scale')
        indices = list(range(13))+list(range(15,29))
        if spec['sdk29_indices'] != indices:
            raise ValueError('Wrong SDK mapping: '+name)
        defaults = np.asarray(cfg['default_q_sdk29'])[indices]
        if not np.array_equal(defaults, spec['default_joint_positions']):
            raise ValueError('Calibration/default mismatch: '+name)
    balance = contracts['balance']
    if (balance['schema'] != 'balance_g1_zero_command_456_to_12_v1' or
            balance['input']['fixed_velocity_command'] != [0,0,0] or
            balance['input']['fixed_height_command'] != .74 or
            balance['input']['previous_action_source'] != 'previous_selected_action_after_training_clip' or
            balance['output']['output_sdk29_indices'] != list(range(12))):
        raise ValueError('Unsupported Balance observation/action semantics')
    for spec in contracts.values():
        if (spec['input'].get('previous_action_source') != 'previous_selected_action_after_training_clip' or
                spec['input'].get('history_source') != 'selected_policy_execution'):
            raise ValueError('Unsupported execution history semantics')
        # Copying selected frames between candidates requires identical units.
        for key in ('command_scale', 'clip_observations'):
            if spec['input'][key] != balance['input'][key]:
                raise ValueError('Incompatible shared history scale: '+key)
        for key in ('ang_vel', 'dof_pos', 'dof_vel'):
            if spec['input']['observation_scales'][key] != balance['input']['observation_scales'][key]:
                raise ValueError('Incompatible shared observation scale: '+key)
        if spec['output']['training_clip_actions'] != balance['output']['training_clip_actions']:
            raise ValueError('Incompatible shared previous-action clipping')
    if cfg['height_m'] != balance['input']['fixed_height_command']:
        raise ValueError('Walking and Balance must use the same training height')
    # Existing web/calibration helpers need the common SDK names only.
    return cfg, dict(contracts, joint_names_sdk29=cfg['joint_names_sdk29']), token


class Policy:
    def __init__(self, root, cfg, policy_mode='auto'):
        if policy_mode not in ('auto','balance','homie'):
            raise ValueError('policy_mode must be auto, balance or homie')
        import onnxruntime as ort
        self.policy_mode = policy_mode
        self.root = Path(root)
        self.manifest = json.loads((self.root/'models/manifest.json').read_text())
        self.source_sha256 = self.manifest['model_sha256']
        for name, digest in self.manifest['artifact_sha256'].items():
            if hashlib.sha256((self.root/name).read_bytes()).hexdigest() != digest:
                raise ValueError('Artifact hash mismatch: '+name)
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        self.sessions = {}
        for name in PROFILES:
            if cfg['model_files'][name] != self.manifest['model_files'][name]:
                raise ValueError('Unexpected model path')
            session = ort.InferenceSession(str(self.root/cfg['model_files'][name]),
                                           sess_options=options, providers=['CPUExecutionProvider'])
            inp, out = session.get_inputs(), session.get_outputs()
            if (len(inp)!=1 or len(out)!=1 or len(inp[0].shape)!=2 or len(out[0].shape)!=2 or
                    inp[0].shape[-1]!=456 or out[0].shape[-1]!=12 or
                    inp[0].type!='tensor(float)' or out[0].type!='tensor(float)'):
                raise ValueError('Wrong ONNX interface: '+name)
            self.sessions[name] = session
        self.threshold = cfg['model_switch_command_norm']
        self.reset_selection()

    def reset_selection(self):
        self.selection_reason = 'initial'
        self.selected = 'homie' if self.policy_mode == 'homie' else 'balance'

    def select(self, velocity, state=None):
        value = np.asarray(velocity, dtype=np.float32)
        if value.shape != (3,) or not np.isfinite(value).all():
            raise ValueError('Invalid velocity command')
        norm = float(np.linalg.norm(value))
        if self.policy_mode != 'auto':
            self.selected, self.selection_reason = self.policy_mode, 'forced'
        elif state is not None and state.stage != p.POLICY:
            self.selected, self.selection_reason = 'balance', 'not_running'
        elif norm >= self.threshold:
            self.selected, self.selection_reason = 'homie', 'motion_command'
        else:
            # The switch bank hands over with residual motion below this
            # command threshold; no exact-zero dwell or motion gate is trained.
            self.selected, self.selection_reason = 'balance', 'below_command_threshold'
        return self.selected

    def infer(self, name, observations):
        session = self.sessions[name]
        obs = np.asarray(observations, dtype=np.float32)
        if obs.shape != (1,456) or not np.isfinite(obs).all():
            raise ValueError('Invalid observation: '+name)
        action = session.run(None, {session.get_inputs()[0].name: obs})[0][0]
        if action.shape != (12,) or not np.isfinite(action).all():
            raise ValueError('Invalid action: '+name)
        return action

    def __call__(self, observations, velocity, state=None):
        self.select(velocity, state)
        # Both candidates see the execution history; only the selected output
        # may become the next frame's previous action.
        return {name:self.infer(name, observations[name]) for name in PROFILES}


class Observations:
    def __init__(self, contracts):
        self.models = {'balance': BalanceObservations(contracts['balance']),
                       'homie': HomieObservations(contracts['homie'])}

    @property
    def frames(self):
        return min(model.frames for model in self.models.values())

    def reset(self):
        for model in self.models.values():
            model.reset()

    def push(self, state, velocity, height, torso_rpy):
        return {'balance':self.models['balance'].push(state, velocity, height, torso_rpy).copy(),
                'homie':self.models['homie'].push(state, velocity, height).copy()}

    def record_action(self, actions, selected):
        """Commit the action sent to the bridge and its command/history.

        Each candidate substitutes only its newest command. Past frames and
        previous actions must come from the selected policy, including across
        switches. Shadow predictions never enter that history.
        """
        history = self.models[selected].history.copy()
        for model in self.models.values():
            model.record_action(actions[selected])
            model.history[:] = history


def targets(action, state, cfg, contracts, selected):
    spec = contracts[selected]
    count = spec['output']['shape'][-1]
    action = np.asarray(action, dtype=np.float32)
    if action.shape != (count,) or not np.isfinite(action).all():
        raise ValueError('Invalid selected action')
    action = np.clip(action, -spec['output']['training_clip_actions'], spec['output']['training_clip_actions'])
    q = state.applied_q.copy()
    q[:count] = np.asarray(spec['default_joint_positions'][:count]) + spec['output']['action_scale']*action
    return np.clip(q, cfg['joint_lower_sdk29'],cfg['joint_upper_sdk29']).astype(np.float32)


class Commands:
    def __init__(self, cfg):
        self.cfg = cfg
        self.value = np.zeros(3, dtype=np.float32)
        self.target = self.value.copy()

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
            self.value.fill(0)
        self.target[:] = target
        self.value[:] = filter_command(self.value, target, dt, self.cfg['command_filter_tau_s'])
        return self.value.copy()
