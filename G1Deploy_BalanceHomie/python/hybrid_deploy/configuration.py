"""Shared build/runtime validation; token covers locomotion and push-cart PD."""
import hashlib
import json
import math
from pathlib import Path

PROFILE = 'hybrid_balance_homie_456_456'
PROFILES = ('balance', 'homie')


def load_config(root):
    root = Path(root)
    raw = (root/'config/deploy.json').read_bytes()
    cfg = json.loads(raw)
    digest = hashlib.sha256(b'config/deploy.json\0'+raw)
    if (cfg['schema'] != 1 or cfg['policy_profile'] != PROFILE or
            cfg['policy_hz'] != 50 or cfg['servo_hz'] != 200 or cfg['controlled_dofs'] != 12 or
            cfg['upper_body_mode'] != 'hold_current' or cfg['waist_mode'] != 'hold_current'):
        raise ValueError('Unsupported hybrid deployment configuration')
    if 'kp_sdk29' in cfg or 'kd_sdk29' in cfg:
        raise ValueError('Store gains in the independent PD profile files')

    def array(values, size, name):
        if len(values) != size or any(isinstance(v, bool) or not isinstance(v, (int, float)) or
                                     not math.isfinite(v) or abs(v) >= 1000 for v in values):
            raise ValueError('Invalid array: '+name)

    for key in ('joint_lower_sdk29','joint_upper_sdk29','default_q_sdk29','effort_sdk29'):
        array(cfg[key], 29, key)
    if any(not lo < hi or not lo <= q <= hi for lo, hi, q in
           zip(cfg['joint_lower_sdk29'],cfg['joint_upper_sdk29'],cfg['default_q_sdk29'])):
        raise ValueError('Invalid joint bounds/defaults')
    if any(v <= 0 for v in cfg['effort_sdk29']):
        raise ValueError('Invalid effort bounds')
    if len(cfg['joint_names_sdk29']) != 29 or len(set(cfg['joint_names_sdk29'])) != 29:
        raise ValueError('Expected 29 unique SDK joint names')
    for key in ('state_timeout_s','command_timeout_s','calibration_s','gain_ramp_s',
                'calibration_timeout_s','calibration_position_tolerance_rad','calibration_velocity_tolerance_rad_s',
                'calibration_speed_rad_s','damping_kd','shutdown_damping_s'):
        if not 0 < cfg[key] <= 20:
            raise ValueError('Invalid setting: '+key)
    if not 0 <= cfg['profile_transition_s'] <= 5:
        raise ValueError('profile_transition_s must be between 0 and 5 seconds (0: direct handoff)')
    # A late policy command first holds the last leg targets with Balance PD;
    # only a sustained gap removes support.
    cfg.setdefault('command_hold_timeout_s', 2.0)
    if not cfg['command_timeout_s'] < cfg['command_hold_timeout_s'] <= 10:
        raise ValueError('command_hold_timeout_s must exceed command_timeout_s and be at most 10 s')
    if not 0 <= cfg['policy_target_speed_rad_s'] <= 20:
        raise ValueError('Invalid policy target speed (0: direct policy targets)')
    if not .02 <= cfg['command_filter_tau_s'] <= 2:
        raise ValueError('Invalid command filter time constant')
    if cfg['calibration_timeout_s'] <= cfg['calibration_s']:
        raise ValueError('Calibration timeout must exceed ramp duration')
    if not 0 < cfg['model_switch_command_norm'] <= 1:
        raise ValueError('Invalid model switch threshold')
    if not isinstance(cfg['require_remote_heartbeat'], bool):
        raise ValueError('require_remote_heartbeat must be boolean')
    if not 0 < cfg['max_tilt_rad'] < 1.6 or not 0 < cfg['max_temperature_c'] <= 100:
        raise ValueError('Invalid health thresholds')
    if not 0 < cfg['startup_timeout_s'] <= 60 or not .45 <= cfg['height_m'] <= .85:
        raise ValueError('Invalid startup timeout/height')
    for key in ('bridge_port','state_port'):
        if not isinstance(cfg[key], int) or not 1024 <= cfg[key] <= 65535:
            raise ValueError('Invalid port')
    if cfg['bridge_port'] == cfg['state_port'] or not 0 <= cfg['stick_deadzone'] < 1:
        raise ValueError('Invalid ports/deadzone')
    if len(cfg['command_limits']) != 3 or any(len(v) != 2 or not -2 <= v[0] <= 0 <= v[1] <= 2 for v in cfg['command_limits']):
        raise ValueError('Invalid command limits')
    array(cfg['command_slew_per_s'], 3, 'command_slew_per_s')
    if any(not 0 < v <= 10 for v in cfg['command_slew_per_s']):
        raise ValueError('Invalid command slew')
    array(cfg['torso_rpy_command'], 3, 'torso_rpy_command')
    if any(abs(v) > .3 for v in cfg['torso_rpy_command']):
        raise ValueError('Invalid torso command')
    gains = {}
    for name in PROFILES:
        # Fixed, separate locations make dependencies unambiguous during CMake rebuilds.
        path = 'config/'+name+'_pd.json'
        if cfg['pd_files'][name] != path:
            raise ValueError('Unexpected PD file path')
        data = (root/path).read_bytes()
        digest.update(b'\0'+path.encode()+b'\0'+data)
        pd = json.loads(data)
        if pd['schema'] != 1 or pd['profile'] != name or pd['joint_names_sdk29'] != cfg['joint_names_sdk29']:
            raise ValueError('PD profile/joint order mismatch: '+name)
        for key in ('kp_sdk29','kd_sdk29'):
            array(pd[key], 29, name+'/'+key)
            if any(v < 0 for v in pd[key]):
                raise ValueError('Negative gains: '+name)
        if pd['kd_sdk29'][14] <= 0:
            raise ValueError('Positive waist pitch damping is required')
        gains[name] = pd
    path = 'config/pushcart_upper_pd.json'
    data = (root/path).read_bytes()
    digest.update(b'\0'+path.encode()+b'\0'+data)
    upper = json.loads(data)
    indices = [12] + list(range(15, 29))
    if (upper['schema'] != 1 or upper['profile'] != 'pushcart_upper'
            or upper['joint_names'] != [cfg['joint_names_sdk29'][i] for i in indices]):
        raise ValueError('Push-cart upper-body PD joint order mismatch')
    for key in ('kp', 'kd'):
        array(upper[key], 15, 'pushcart_upper/'+key)
        if any(v <= 0 for v in upper[key]):
            raise ValueError('Push-cart upper-body gains must be positive')
        values = [0.] * 29
        for i, value in zip(indices, upper[key]):
            values[i] = value
        cfg['pushcart_upper_'+key+'_sdk29'] = values
    transition = upper['transition_s']
    if type(transition) not in (int, float) or not .1 <= transition <= 2:
        raise ValueError('Push-cart upper-body PD transition must be 0.1..2 seconds')
    cfg['pushcart_upper_transition_s'] = transition
    interpolation = upper['target_interpolation_s']
    if type(interpolation) not in (int, float) or interpolation not in (0., 1 / cfg['policy_hz']):
        raise ValueError('Push-cart upper target interpolation must be 0 or one 50 Hz period (0.02 s)')
    cfg['pushcart_upper_target_interpolation_s'] = interpolation
    return cfg, gains, int(digest.hexdigest()[:16], 16)
