"""Display contract and the sibling Balance/Homie deployment bundle."""
from pathlib import Path
import json
import sys
import xml.etree.ElementTree as ET

import numpy as np
import yaml
from core.heading import CONFIG_PATH, load_settings
from core.hand_grasp import HAND_JOINTS, HAND_LOWER, HAND_UPPER, load_grasp, validate_grasp

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
DEPLOY = WORKSPACE / 'G1Deploy_BalanceHomie'
URDF_PATH = WORKSPACE / 'G1_URDF/g1.urdf'
sys.path.insert(0, str(DEPLOY / 'python'))

from hybrid_deploy.policy import read_bundle
from hybrid_deploy.web_control import joint_names

JOINT_NAMES = tuple(joint_names())
ACTIVE = np.array(list(range(13)) + list(range(15, 29)), dtype=int)
INACTIVE = (13, 14)
UPPER = np.array([12] + list(range(15, 29)), dtype=int)
UPPER_POSE_PATH = WORKSPACE / 'G1_Upper_Initpose/deploy_upper_hold.yaml'
UPPER_POSE_PATHS = {
    'original': UPPER_POSE_PATH,
    'arms_down': WORKSPACE / 'G1_Upper_Initpose/deploy_upper_arms_down.yaml',
}
UPPER_POSE_LABELS = {'original': '姿态 1 · 原准备姿态', 'arms_down': '姿态 2 · 双臂下垂'}
HAND_ZERO_PATH = ROOT / 'config/dex3_zero_calibration.json'
HAND_PD_PATH = ROOT / 'config/dex3_pd.json'


def load_hand_pd(path):
    data = json.loads(Path(path).read_text())
    if (not isinstance(data, dict) or set(data) != {'schema_version', 'kp', 'kd'}
            or type(data['schema_version']) is not int or data['schema_version'] != 1):
        raise ValueError('Dex3 PD 配置必须包含 schema_version=1、kp、kd')
    gains = {}
    for key in ('kp', 'kd'):
        value = data[key]
        if (type(value) not in (int, float) or not np.isfinite(value)
                or value < 0 or value > np.finfo(np.float32).max):
            raise ValueError(f'Dex3 {key} 必须为有限非负 float32 数值')
        gains[key] = float(np.float32(value))
    if gains['kp'] <= 0:
        raise ValueError('Dex3 正常位置控制 kp 必须大于 0')
    return gains


def load_hand_zero(path, lower, upper):
    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict) or data.get('schema_version') != 1:
        raise ValueError('Dex3 补偿文件必须为 schema_version=1')
    rows = []
    for side in ('left', 'right'):
        names = [f'{side}_hand_{joint}_joint' for joint in HAND_JOINTS]
        mapping = data.get(f'{side}_raw_zero_rad')
        if not isinstance(mapping, dict) or set(mapping) != set(names):
            raise ValueError(f'Dex3 {side} 补偿必须包含完整的 7 个关节名')
        if any(isinstance(mapping[n], bool) or not isinstance(mapping[n], (int, float)) for n in names):
            raise ValueError('Dex3 补偿角必须为数值')
        rows.append([mapping[n] for n in names])
    angles = np.asarray(rows, dtype=float)
    if not np.isfinite(angles).all() or (angles < lower).any() or (angles > upper).any():
        raise ValueError('Dex3 补偿归零姿态存在非有限值或超出关节限位')
    return angles


class RobotConfig:
    def __init__(self, urdf=URDF_PATH, motion_config=CONFIG_PATH, command_frame=None, heading_kp=None, heading_mode=None,
                 hand_zero_calibration=HAND_ZERO_PATH, hand_pd_config=HAND_PD_PATH):
        self.urdf = Path(urdf).resolve()
        self.motion_config_path = Path(motion_config).resolve()
        self.motion_settings = load_settings(self.motion_config_path, command_frame, heading_kp, heading_mode)
        self.hand_pd_path = Path(hand_pd_config).resolve()
        self.hand_pd = load_hand_pd(self.hand_pd_path)
        if self.motion_settings['command_frame'] != 'body':
            raise ValueError('网页速度已改为机身坐标 + 航向保持；请使用 --command-frame body，旧 world 配置仅供历史分析')
        self.deploy = DEPLOY
        self.deployment, self.contracts, self.token = read_bundle(self.deploy)
        joints = {j.get('name'): j for j in ET.parse(self.urdf).findall('joint')}
        actual = {n for n, j in joints.items() if j.get('type') in ('revolute', 'continuous')}
        if actual != {JOINT_NAMES[i] for i in ACTIVE}:
            raise ValueError('Expected the outer single-waist G1 URDF with 27 movable joints')
        if any(joints[JOINT_NAMES[i]].get('type') != 'fixed' for i in INACTIVE):
            raise ValueError('Display waist roll/pitch must remain fixed')
        self.default_q = np.asarray(self.deployment['default_q_sdk29'], dtype=float)
        self.upper_pose_path = UPPER_POSE_PATH
        self.upper_pose_paths = UPPER_POSE_PATHS.copy()
        self.prepare_poses = {}
        names = [JOINT_NAMES[i] for i in UPPER]
        lower = np.asarray(self.deployment['joint_lower_sdk29'])
        upper = np.asarray(self.deployment['joint_upper_sdk29'])
        for key, path in self.upper_pose_paths.items():
            pose = yaml.safe_load(path.read_text())
            angles = pose.get('hold_target_q') if isinstance(pose, dict) else None
            if not isinstance(angles, dict) or set(angles) != set(names):
                raise ValueError(f'{path.name} 上半身准备姿态必须包含腰 yaw 和双臂全部 15 个关节')
            if any(type(angles[name]) not in (int, float) for name in names):
                raise ValueError(f'{path.name} 上半身准备角度必须为数值')
            q = self.default_q.copy()
            q[UPPER] = [angles[name] for name in names]
            q[list(INACTIVE)] = 0.
            if not np.isfinite(q).all() or (q < lower).any() or (q > upper).any():
                raise ValueError(f'{path.name} 上半身准备姿态存在非有限值或超出关节限位')
            self.prepare_poses[key] = q
        # Preserve the default-pose API used by the navigation entry point.
        self.prepare_q = self.prepare_poses['original']
        # Choosing an upper-body posture never changes or enables Dex3.
        self.hand_grasp = load_grasp(self.upper_pose_path)
        hand_lower, hand_upper = HAND_LOWER, HAND_UPPER
        self.hand_zero_path = Path(hand_zero_calibration)
        self.hand_zero, self.hand_zero_error = None, ''
        try:
            self.hand_zero = load_hand_zero(self.hand_zero_path, hand_lower, hand_upper)
        except (OSError, ValueError) as exc:
            # Optional calibration never changes the original open/grasp targets.
            self.hand_zero_error = str(exc)
        self.set_hand_grasp(self.hand_grasp)
        self.velocity_bounds = np.asarray(self.deployment['command_limits'], dtype=float)
        self.policy_paths = {name: self.deploy / path
                             for name, path in self.deployment['model_files'].items()}

    def set_hand_grasp(self, angles):
        self.hand_grasp = validate_grasp(angles)
        self.hand_grasp_compensated = None
        self.hand_grasp_compensation_error = self.hand_zero_error
        if self.hand_zero is not None:
            compensated = self.hand_grasp + self.hand_zero
            if (not np.isfinite(compensated).all() or (compensated < HAND_LOWER).any()
                    or (compensated > HAND_UPPER).any()):
                self.hand_grasp_compensation_error = 'Dex3 抓握角 + 补偿角超出关节限位'
            else:
                self.hand_grasp_compensated = compensated
