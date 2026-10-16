#!/usr/bin/env python3
"""Headless executor: Nav2 velocity commands drive our own Balance/Homie locomotion.

Navigation is only the outer path follower. This process replaces Unitree's
LocoClient (the former g1_loco_udp_bridge) and runs the same controller as
/home/unitree/CWC/Viser: Balance/Homie ONNX policies from
G1Deploy_BalanceHomie, the pelvis-IMU heading PID (navigation settings in
nav/locomotion/motion_config.json), and the deployment's C++ 200 Hz servo bridge.
The Viser UI is not started; it remains the body-control test UI.

Data path:
  Nav2 /cmd_vel -> g1_cmd_vel_udp_node (limits, motion-gate lease, 250 ms timeout)
    -> UDP 127.0.0.1:39002 (VelocityPacket) -> this executor -> Viser core.Controller
    -> G1Deploy_BalanceHomie/build/g1_control -> rt/lowcmd
  Executor status -> UDP 127.0.0.1:39005 (JSON) -> nav web bridge (motion gate precondition)

Startup is an explicit, operator-attended sequence driven by the nav web page
(acquire -> prepare -> start policy). Motion commands are accepted only while
the policy runs. Navigation zero/gate closure use the training 0.4 s filter;
command expiry is detected from the original command timestamp. Physical L2, damping
and bridge faults still stop immediately. After damping, "reset" restarts
the bridge read-only.
"""
import argparse
from pathlib import Path
import json
import signal
import socket
import struct
import sys
import threading
import time

import numpy as np

WORKSPACE = Path(__file__).resolve().parents[2]
VISER = WORKSPACE / 'Viser'
sys.path.insert(0, str(VISER))

from core.robot import RobotConfig, URDF_PATH, HAND_PD_PATH, UPPER_POSE_LABELS  # noqa: E402
from core.controller import Controller, STAGES  # noqa: E402
from core.joint_monitor import limits, live_warnings, WarningBoard  # noqa: E402

# Must match nav/src/unitree_g1_kinematics/include/unitree_g1_kinematics/velocity_packet.hpp
# Version 2 includes the original /cmd_vel host monotonic timestamp.
VELOCITY_PACKET = struct.Struct('<IHBBQfff4xQ')
VELOCITY_MAGIC, VELOCITY_VERSION = 0x47315643, 2
COMMAND, GATE_CLOSED, COMMAND_EXPIRED = 0, 1, 2
MOTION_CONFIG = Path(__file__).with_name('motion_config.json')
# Lifecycle request from the nav web bridge (unitree_nav2_web/locomotion.py).
REQUEST_PACKET = struct.Struct('<IHBBQ')
REQUEST_MAGIC, REQUEST_VERSION = 0x47314C52, 1
REQUESTS = {1: 'acquire', 2: 'prepare', 3: 'policy', 4: 'zero', 5: 'stop', 6: 'reset',
            7: 'prepare_pose'}
PREPARE_POSES = {0: 'original', 1: 'arms_down'}
# Executor's single internal "browser" identity inside Viser's WebControl.
CLIENT = 1
COMMAND_TIMEOUT_S = 0.25
STATUS_INTERVAL_S = 0.1


class Executor:
    def __init__(self, controller, command_port, status_port, heading_source='NAVIGATION', clock=time.monotonic):
        self.control = controller
        self.clock = clock
        self.status_port = status_port
        if heading_source != 'NAVIGATION':
            raise ValueError('Navigation must use NAVIGATION; use heading_mode:=disabled to bypass feedback')
        self.heading_source = heading_source
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self.sock.bind(('127.0.0.1', command_port))
            self.sock.setblocking(False)
        except OSError as exc:
            self.sock.close()
            raise RuntimeError('UDP 127.0.0.1:%d 已被占用：请先停止旧的 g1_loco_udp_bridge '
                               '或其他运控执行进程' % command_port) from exc
        self.command_port = command_port
        self.target = np.zeros(3, dtype=np.float32)
        self.enabled = False
        self.last_packet = -float('inf')
        self.last_motion = -float('inf')
        self.last_command_stamp_ns = -1
        self.accept_after = -float('inf')
        self.new_motion = self.motion_active = False
        self.packets = self.rejected = 0
        self.last_status = -float('inf')
        self.last_request_sequence = -1
        self.last_claim = -float('inf')
        self.warning_lower, self.warning_upper = limits(controller.config)
        self.warning_board = WarningBoard(clock=clock)

    def _receive(self, now):
        for _ in range(64):
            try:
                data, sender = self.sock.recvfrom(64)
            except BlockingIOError:
                return
            now = self.clock()
            if sender[0] != '127.0.0.1':
                self.rejected += 1
                continue
            if len(data) == VELOCITY_PACKET.size:
                magic, version, enabled, reason, _, vx, vy, wz, stamp_ns = VELOCITY_PACKET.unpack(data)
                velocity = np.asarray([vx, vy, wz], dtype=np.float32)
                stamp = stamp_ns*1e-9
                if (magic != VELOCITY_MAGIC or version != VELOCITY_VERSION
                        or enabled not in (0, 1) or reason not in (COMMAND, GATE_CLOSED, COMMAND_EXPIRED)
                        or (bool(enabled) != (reason == COMMAND))
                        or stamp > now + 1e-6 or not np.isfinite(velocity).all()):
                    self.rejected += 1
                    continue
                self.packets += 1
                self.last_packet = now
                if enabled:
                    if stamp_ns > self.last_command_stamp_ns and stamp > self.accept_after:
                        self.enabled = True
                        self.last_command_stamp_ns = stamp_ns
                        self.target[:] = velocity
                        self.last_motion = min(stamp, now)  # nanosecond-to-float rounding
                        self.new_motion = True
                else:
                    self.enabled = False
                    # Discard unseen commands already in flight when the gate
                    # closed, including while standing still in Balance.
                    self.accept_after = max(self.accept_after, now)
                    self._stop_motion()
            elif len(data) == REQUEST_PACKET.size:
                magic, version, request, payload, sequence = REQUEST_PACKET.unpack(data)
                if (magic != REQUEST_MAGIC or version != REQUEST_VERSION or request not in REQUESTS
                        or (request == 7 and payload not in PREPARE_POSES)
                        or (request != 7 and payload != 0)
                        or sequence <= self.last_request_sequence):
                    self.rejected += 1
                    continue
                self.last_request_sequence = sequence
                self.request(REQUESTS[request], pose=PREPARE_POSES[payload] if request == 7 else None)
            else:
                self.rejected += 1

    def _stop_motion(self):
        self.target.fill(0)
        self.new_motion = False
        if self.motion_active:
            self.control.emit('zero', CLIENT)
        self.motion_active = False

    def request(self, kind, *, pose=None):
        """Operator lifecycle request from the nav web page."""
        if kind == 'prepare_pose':
            if not isinstance(pose, str) or pose not in self.control.config.prepare_poses:
                raise ValueError('请选择有效的准备姿态')
            # The shared Controller validates ownership and locks the choice
            # as soon as preparation is requested, including queued events.
            self.control.emit('prepare_pose', CLIENT, pose=pose)
        elif kind == 'stop':
            self.target.fill(0)
            self.control.emit('stop', CLIENT)
        elif kind == 'zero':
            self.target.fill(0)
            self.control.emit('zero', CLIENT)
        elif kind == 'reset':
            # After a damping stop: restart the C++ bridge read-only. Motors stay
            # released until acquire is requested again.
            self.target.fill(0)
            self.control.emit('claim', CLIENT)
        else:
            self.control.emit(kind, CLIENT)
        if kind in ('stop', 'zero', 'reset'):
            self.accept_after = self.clock()
            self.enabled = self.new_motion = self.motion_active = False

    def step(self):
        now = self.clock()
        self.new_motion = False
        self._receive(now)
        now = self.clock()
        data = self.control.snapshot()
        stopped = data.get('stage') in ('DAMPED', 'FAULT')
        if (data.get('state') is not None and not stopped
                and (data.get('owner') != CLIENT or data.get('source') != 'web')
                and now - self.last_claim >= 1.):
            # Own the command channel; claiming never arms motors.
            self.control.emit('claim', CLIENT)
            self.last_claim = now
        fresh = self.enabled and 0 <= now - self.last_motion <= COMMAND_TIMEOUT_S
        running = data.get('stage') == 'POLICY' and data.get('source') == 'web' and data.get('owner') == CLIENT
        if not fresh:
            self._stop_motion()
        elif running and self.new_motion and not data.get('decelerating'):
            # Forward each original command once; cached UDP retransmissions
            # cannot prolong the watchdog or revive motion after a stop ramp.
            self.control.emit('move', CLIENT, [float(v) for v in self.target],
                              command_source=self.heading_source, command_time=self.last_motion)
            self.motion_active = bool(np.any(self.target))
        self._report(data, now, fresh)

    def _warnings(self, data, now):
        # Use Viser's read-only checks. Warning rendering must never affect
        # controller health, arming, motion authorization or servo output.
        try:
            current = live_warnings(data, data.get('joint_comparison'),
                                    self.warning_lower, self.warning_upper,
                                    self.control.cfg['state_timeout_s'], now=now)
            return [dict(level=level, message=message, active=active)
                    for level, message, active, _ in self.warning_board.update(current)]
        except Exception as exc:
            # A display/formatting failure must not unwind main() and close the
            # body controller. Keep it visible without changing control state.
            return [dict(level='warning', active=True,
                         message='告警显示暂时不可用（机身控制继续）：' + str(exc)[:200])]

    def _report(self, data, now, fresh):
        if now - self.last_status < STATUS_INTERVAL_S:
            return
        self.last_status = now
        stage = data.get('stage', 'WAITING')
        dex3 = data.get('dex3') or {}
        owned = data.get('source') == 'web' and data.get('owner') == CLIENT
        ready = stage == 'POLICY' and bool(data.get('healthy')) and owned
        status = dict(
            magic='G1LOCO1', ready=ready, mode=self.control.mode, stage=stage,
            stage_text=STAGES.get(stage, stage), healthy=bool(data.get('healthy')),
            armed=bool(self.control.backend.armed), reason=data.get('reason') or '',
            source=data.get('source'), owned=data.get('owner') == CLIENT,
            prepare_pose=data.get('prepare_pose'),
            prepare_pose_options=[dict(value=key, label=UPPER_POSE_LABELS[key])
                                  for key in self.control.config.prepare_poses],
            prepare_pose_selectable=bool(owned and data.get('prepare_pose_selectable')),
            preparation_warning=data.get('preparation_warning') or '',
            warnings=self._warnings(data, now),
            observation_frames=int(data.get('observation_frames', 0)),
            dex3_ready=bool(dex3.get('healthy')), dex3_reason=dex3.get('blocking_reason') or '',
            selected_model=data.get('selected_model'),
            decelerating=bool(data.get('decelerating')),
            velocity=[float(v) for v in data.get('velocity', np.zeros(3))],
            nav_target=[float(v) for v in self.target], nav_command_fresh=fresh,
            nav_packet_age_s=None if self.last_packet < 0 else round(now - self.last_packet, 3),
            heading=(data.get('heading') or {}).get('heading_state'),
            simulation=bool(data.get('simulation')),
            sent_monotonic=now)
        try:
            self.sock.sendto(json.dumps(status, ensure_ascii=False).encode('utf-8'),
                             ('127.0.0.1', self.status_port))
        except OSError:
            pass

    def close(self):
        self.sock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--mode', choices=('sim', 'control'), default='control',
                        help='control: real robot; sim: synthetic C++ feedback, no DDS')
    parser.add_argument('--interface', default='eth0')
    parser.add_argument('--policy-mode', choices=('auto', 'balance', 'homie'), default='auto')
    parser.add_argument('--motion-config', default=str(MOTION_CONFIG),
                        help='Navigation heading PID; stop filter uses the shared deployment config')
    parser.add_argument('--heading-mode', choices=('config', 'disabled', 'shadow', 'active'), default='config',
                        help='Override run_mode from the motion config; "config" keeps the file value')
    parser.add_argument('--heading-kp', type=float, help='Override heading Kp (0..2 per second)')
    parser.add_argument('--heading-source', choices=('NAVIGATION',), default='NAVIGATION',
                        help='Nav2 moving heading reference; use --heading-mode disabled to bypass PID')
    parser.add_argument('--dex3-pd-config', default=str(HAND_PD_PATH))
    parser.add_argument('--command-port', type=int, default=39002)
    parser.add_argument('--status-port', type=int, default=39005)
    parser.add_argument('--record', action='store_true',
                        help='Archive this run in Viser/logs/sessions (shares its 5-session limit)')
    args = parser.parse_args()
    if (not 1024 <= args.command_port <= 65535 or not 1024 <= args.status_port <= 65535
            or args.command_port == args.status_port):
        parser.error('Invalid UDP ports')

    config = RobotConfig(URDF_PATH, args.motion_config, None, args.heading_kp,
                         None if args.heading_mode == 'config' else args.heading_mode,
                         hand_pd_config=args.dex3_pd_config)
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    recorder = controller = executor = backend = None
    try:
        if args.record and args.mode == 'control':
            from core.recording import SessionRecorder
            recorder = SessionRecorder(config, dict(vars(args), mode='control'))
            print('RECORDING ' + str(recorder.path), flush=True)
        from core.backends import Bridge, build_bridge
        build_bridge()
        backend = Bridge(config, args.mode, args.interface)
        controller = Controller(config, backend, args.mode, recorder=recorder, policy_mode=args.policy_mode)
        executor = Executor(controller, args.command_port, args.status_port, args.heading_source)
        controller.start()
        print('G1 LOCOMOTION EXECUTOR READY mode=%s interface=%s heading=%s Kp=%.2f source=%s | '
              'cmd udp://127.0.0.1:%d status udp://127.0.0.1:%d'
              % (args.mode, args.interface, config.motion_settings['run_mode'],
                 config.motion_settings['heading_kp'], args.heading_source,
                 args.command_port, args.status_port), flush=True)
        while not stop.is_set():
            executor.step()
            stop.wait(.02)
    finally:
        if controller:
            controller.close()
        elif backend:
            backend.close()
        if executor:
            executor.close()
        if recorder:
            recorder.close(error=controller.reason if controller and controller.stage == 'FAULT' else None)


if __name__ == '__main__':
    main()
