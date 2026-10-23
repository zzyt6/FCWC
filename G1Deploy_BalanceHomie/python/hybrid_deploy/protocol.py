"""与 C++ 桥接共享的本机 UDP 格式；避免依赖 LCM/CUDA。"""
import struct
from dataclasses import dataclass, field
import numpy as np

STATE = struct.Struct('<IIQQQQIIII101fIf58fIf42f7Q43fIiIf')
COMMAND = struct.Struct('<IIQQQQQII29fIiI14fII')
STATE_MAGIC, COMMAND_MAGIC, VERSION = 0x48594253, 0x48594243, 9
PROFILE_NAMES = ('balance', 'homie')
UPPER_MODE_NAMES = ('hold', 'pushcart')
WAITING, RAMPING, HOLDING, POLICY, DAMPED, FAULT = range(6)
WAIT, PREPARE, HOLD, RUN, STOP = range(5)
R2, L2, B, L1 = 1 << 4, 1 << 5, 1 << 9, 1 << 1
STAGE_NAMES = ('WAITING', 'RAMPING', 'HOLDING', 'POLICY', 'DAMPED', 'FAULT')
# State flag independent of health; preparation keeps its targets and gains.
PREPARATION_OVERDUE = 1 << 3
# Policy command late: the bridge holds the last leg targets with Balance PD.
POLICY_HOLD = 1 << 4


@dataclass
class State:
    session: int
    sequence: int
    received_ns: int
    config_token: int
    stage: int
    flags: int
    buttons: int
    mode_machine: int
    q: np.ndarray
    dq: np.ndarray
    omega: np.ndarray
    quaternion: np.ndarray
    rpy: np.ndarray
    sticks: np.ndarray
    applied_q: np.ndarray
    profile: int = 0
    gain_blend: float = 1.
    applied_kp: np.ndarray = field(default_factory=lambda: np.zeros(29, dtype=np.float32))
    applied_kd: np.ndarray = field(default_factory=lambda: np.zeros(29, dtype=np.float32))
    # Health 1/2 excludes position warnings; active 4, moving 8, grasp 16,
    # supported 32, compensated 64, recovering 128, live grasp editing 256.
    hand_flags: int = 0
    hand_progress: float = 0.
    hand_q: np.ndarray = field(default_factory=lambda: np.zeros(14, dtype=np.float32))
    hand_dq: np.ndarray = field(default_factory=lambda: np.zeros(14, dtype=np.float32))
    hand_target: np.ndarray = field(default_factory=lambda: np.zeros(14, dtype=np.float32))
    bridge_ns: int = 0
    sent_ns: tuple = (0, 0, 0)
    hand_received_ns: tuple = (0, 0)
    position_mask: int = 0
    sent_q: np.ndarray = field(default_factory=lambda: np.zeros(43, dtype=np.float32))
    hand_recovery_mask: int = 0
    hand_recovery_joint: int = -1
    upper_mode: int = 0
    upper_gain_blend: float = 1.

    @classmethod
    def decode(cls, data):
        if len(data) != STATE.size:
            raise ValueError('Wrong state packet size')
        v = STATE.unpack(data)
        if v[:2] != (STATE_MAGIC, VERSION) or v[6] > FAULT:
            raise ValueError('Wrong state protocol')
        floats = np.asarray(v[10:111], dtype=np.float32)
        if not np.isfinite(floats).all():
            raise ValueError('Non-finite feedback')
        gains = np.asarray(v[112:171], dtype=np.float32)
        if v[111] > 1 or not np.isfinite(gains).all() or not 0 <= gains[0] <= 1 or (gains[1:] < 0).any():
            raise ValueError('Invalid profile/gain feedback')
        hands = np.asarray(v[172:215], dtype=np.float32)
        if not np.isfinite(hands).all() or not 0 <= hands[0] <= 1 or v[171] & ~511:
            raise ValueError('Invalid Dex3 feedback')
        sent_q = np.asarray(v[222:265], dtype=np.float32)
        if not np.isfinite(sent_q).all() or v[221] >> 43 or any(t > v[215] for t in v[216:221]):
            raise ValueError('Invalid command trace')
        if v[265] & ~0x3fff or not -1 <= v[266] < 14 or bool(v[171] & 128) != (v[266] >= 0):
            raise ValueError('Invalid Dex3 recovery state')
        if v[267] not in (0, 1) or not np.isfinite(v[268]) or not 0 <= v[268] <= 1:
            raise ValueError('Invalid upper-body PD state')
        return cls(*v[2:10], floats[:29], floats[29:58], floats[58:61],
                   floats[61:65], floats[65:68], floats[68:72], floats[72:101],
                   v[111], float(gains[0]), gains[1:30], gains[30:59],
                   v[171], float(hands[0]), hands[1:15], hands[15:29], hands[29:43],
                   v[215], v[216:219], v[219:221], v[221], sent_q, v[265], v[266], v[267], v[268])


def command(state, sequence, request, targets, profile=0, hand_pose=0, hand_recover=-1,
            hand_grasp=None, hand_live=False, upper_mode=0, upper_target_active=False):
    q = np.asarray(targets, dtype=np.float32)
    if q.shape != (29,) or not np.isfinite(q).all():
        raise ValueError('Command must have 29 finite targets')
    if profile not in (0,1):
        raise ValueError('Invalid control profile')
    if hand_pose not in (0, 1, 2, 3, 4):
        raise ValueError('Invalid Dex3 pose')
    if type(hand_recover) is not int or not -1 <= hand_recover < 14:
        raise ValueError('Invalid Dex3 recovery joint')
    if type(upper_mode) is not int or upper_mode not in (0, 1):
        raise ValueError('Invalid upper-body control mode')
    if type(upper_target_active) is not bool or (upper_target_active and upper_mode != 1):
        raise ValueError('Upper target motion requires push-cart mode and a boolean active flag')
    grasp = np.zeros(14, dtype=np.float32) if hand_grasp is None else np.asarray(hand_grasp, dtype=np.float32)
    if grasp.shape != (14,) or not np.isfinite(grasp).all() or (hand_live and hand_grasp is None):
        raise ValueError('Dex3 grasp must have 14 finite angles')
    grasp_mode = (2 if hand_live else 1) if hand_grasp is not None else 0
    return COMMAND.pack(COMMAND_MAGIC, VERSION, state.session, sequence,
                        state.sequence, state.received_ns, state.config_token, request, profile, *q,
                        hand_pose, hand_recover, grasp_mode, *grasp, upper_mode, int(upper_target_active))
