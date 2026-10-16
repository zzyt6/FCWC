"""Browser command arbitration. No network/Viser dependency in the policy loop."""
import time
import numpy as np
from . import protocol as p
from .command_filter import filter_command

WEB_TIMEOUT_S = 0.25


class WebControl:
    def __init__(self, cfg, enabled):
        self.enabled = enabled
        self.source = 'remote'
        self.command_source = 'MANUAL'
        self.owner = None
        self.limits = np.asarray(cfg['command_limits'])
        self.filter_tau = cfg['command_filter_tau_s']
        self.target = np.zeros(3, dtype=np.float32)
        self.value = self.target.copy()
        self.last_motion = -float('inf')
        self.last_event = -float('inf')

    def zero(self):
        self.target.fill(0)
        self.value.fill(0)
        self.last_motion = -float('inf')

    def accept(self, event, state, frames, now=None):
        """Private IPC events include the bridge session and original feedback time.

        Old, queued browser input must not revive motion after a stall. A browser
        disconnect keeps WEB selected at zero, never restoring cached sticks.
        """
        now = time.monotonic() if now is None else now
        if not self.enabled or not isinstance(event, dict):
            return None
        try:
            stamp = float(event['time'])
            feedback_time = int(event['feedback_ns']) * 1e-9
            if (event['session'] != state.session or not np.isfinite(stamp)
                    or not 0 <= now-stamp <= WEB_TIMEOUT_S
                    or not 0 <= now-feedback_time <= WEB_TIMEOUT_S
                    or stamp <= self.last_event):
                return None
            client, kind = int(event['client']), event['kind']
        except (KeyError, TypeError, ValueError, OverflowError):
            return None
        self.last_event = stamp
        if kind == 'stop':
            self.zero()
            return p.STOP
        if kind == 'zero':
            # Any viewer may clear motion; only the owner can set motion.
            self.zero()
            self.source = 'web'
            return None
        if kind == 'claim' and (self.owner is None or self.owner == client):
            self.source, self.owner = 'web', client
            self.zero()
            return None
        if kind == 'remote' and (self.owner is None or self.owner == client):
            self.source, self.owner = 'remote', None
            self.zero()
            return None
        if self.source != 'web' or self.owner != client:
            return None
        if kind == 'disconnect':
            self.owner = None
            self.zero()
            return None
        if not state.flags & 1 or state.buttons & (p.L2 | p.B):
            self.zero()
            return None
        if kind == 'prepare' and state.stage == p.WAITING:
            return p.PREPARE
        if kind == 'policy' and state.stage == p.HOLDING and frames >= 6:
            self.zero()
            return p.RUN
        if kind == 'move' and state.stage == p.POLICY:
            try:
                velocity = np.asarray(event['velocity'], dtype=np.float32)
            except (KeyError, TypeError, ValueError):
                return None
            if velocity.shape == (3,) and np.isfinite(velocity).all():
                self.target[:] = np.clip(velocity, self.limits[:, 0], self.limits[:, 1])
                source = event.get('command_source', 'MANUAL')
                self.command_source = source if source in ('MANUAL','NAVIGATION','EXTERNAL') else 'UNKNOWN'
                self.last_motion = stamp
        return None

    def update(self, state, dt, now=None):
        now = time.monotonic() if now is None else now
        if (state.stage != p.POLICY or not state.flags & 1
                or state.buttons & (p.B | p.L2)
                or not 0 <= now-self.last_motion <= WEB_TIMEOUT_S):
            self.zero()
        else:
            self.value[:] = filter_command(self.value, self.target, dt, self.filter_tau)
        return self.value.copy()


def joint_names():
    legs = [f'{side}_{joint}_joint' for side in ('left', 'right')
            for joint in ('hip_pitch', 'hip_roll', 'hip_yaw', 'knee', 'ankle_pitch', 'ankle_roll')]
    arms = [f'{side}_{joint}_joint' for side in ('left', 'right')
            for joint in ('shoulder_pitch', 'shoulder_roll', 'shoulder_yaw', 'elbow',
                          'wrist_roll', 'wrist_pitch', 'wrist_yaw')]
    return legs + ['waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint'] + arms
