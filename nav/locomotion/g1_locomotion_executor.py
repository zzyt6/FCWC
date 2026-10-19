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
sys.path.insert(0, str(WORKSPACE / 'nav/src/unitree_nav2_web'))

from core.robot import RobotConfig, URDF_PATH, HAND_PD_PATH, UPPER_POSE_LABELS, JOINT_NAMES  # noqa: E402
from core.controller import Controller, STAGES  # noqa: E402
from core.joint_monitor import limits, live_warnings, WarningBoard  # noqa: E402
from core.model_sets import ModelSets, MODEL_SET_LABELS  # noqa: E402
from core.hand_grasp import HAND_LOWER, HAND_UPPER  # noqa: E402
from unitree_nav2_web.cart_geometry import CartGeometry, POSE_SOURCES, load_settings, save_pose_source, PelvisYawHistory  # noqa: E402
from unitree_nav2_web.locomotion import locomotion_actions  # noqa: E402

# Must match nav/src/unitree_g1_kinematics/include/unitree_g1_kinematics/velocity_packet.hpp
# Version 3 uses the former padding for the navigation reference epoch.
VELOCITY_PACKET = struct.Struct('<IHBBQfffIQ')
VELOCITY_MAGIC, VELOCITY_VERSION = 0x47315643, 3
COMMAND, GATE_CLOSED, COMMAND_EXPIRED = 0, 1, 2
MOTION_CONFIG = Path(__file__).with_name('motion_config.json')
# Lifecycle request from the nav web bridge (unitree_nav2_web/locomotion.py).
REQUEST_PACKET = struct.Struct('<IHBBQQ')
REQUEST_MAGIC, REQUEST_VERSION = 0x47314C52, 2
REQUESTS = {1: 'acquire', 2: 'prepare', 3: 'policy', 4: 'zero', 5: 'stop', 6: 'reset',
            7: 'prepare_pose', 8: 'model_set'}
PREPARE_POSES = {0: 'original', 1: 'arms_down'}
MODEL_SETS = {0: 'v81', 1: 'v82'}
# Executor's single internal "browser" identity inside Viser's WebControl.
CLIENT = 1
COMMAND_TIMEOUT_S = 0.25
EXECUTOR_PERIOD_S = 0.02
NAV_CART_CONFIG = WORKSPACE / 'nav/config/pushcart_navigation.json'
CART_ACTIONS = frozenset(('hand_toggle', 'hand_compensation', 'hand_edit', 'hand_load', 'hand_save',
                          'cart_camera_start', 'cart_yaw_zero', 'pushcart_enable', 'pushcart_disable',
                          'cart_detach', 'cart_pose_source'))


class Executor:
    def __init__(self, controller, command_port, status_port, heading_source='NAVIGATION', clock=time.monotonic,
                 cart_config=NAV_CART_CONFIG, geometry_port=None):
        self.control = controller
        self.clock = clock
        self.status_port = status_port
        self.geometry_port = geometry_port
        self.cart_settings = load_settings(cart_config)
        self.cart_config_path = Path(cart_config)
        self.cart_pose_source = self.cart_settings['pose_source']
        self.cart_geometry = CartGeometry(self.cart_settings)
        self.pelvis_yaw_history = PelvisYawHistory()
        self.cart_limits = np.asarray(self.cart_settings['task_limits'])
        highs = ([entry['high'] for entry in controller.model_sets.entries.values()]
                 if controller.model_sets is not None else [controller.high_policy])
        for high in highs:
            if high is not None:
                high.limits = self.cart_limits.copy()
                if high.camera_process is not None:
                    high.camera_process.publish_cart_pose = True
                    high.camera_process.quad_decimate = self.cart_settings['camera_quad_decimate']
                    high.camera_process.missing_tag_dir = WORKSPACE / 'nav/log/missing_tags'
                if self.cart_pose_source != 'grasp':
                    high.set_navigation_pose_source(self.cart_pose_source, clock())
        self.cart_attached = False
        self.navigation_epoch = (int(clock()*1e6) & 0xfffffffe) or 2
        self.cart_error = ''
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
        self.velocity_sequences = {}
        self.last_status = -float('inf')
        self.last_request_sequence = -1
        self.cart_request = {}
        self.cart_request_tick = None
        self.pending_pose_source = None
        self.last_claim = -float('inf')
        self.warning_lower, self.warning_upper = limits(controller.config)
        self.warning_board = WarningBoard(clock=clock)

    def _receive(self, now):
        for _ in range(512):
            try:
                data, sender = self.sock.recvfrom(4096)
            except BlockingIOError:
                return
            now = self.clock()
            if sender[0] != '127.0.0.1':
                self.rejected += 1
                continue
            if len(data) == VELOCITY_PACKET.size:
                magic, version, enabled, reason, sequence, vx, vy, wz, epoch, stamp_ns = VELOCITY_PACKET.unpack(data)
                velocity = np.asarray([vx, vy, wz], dtype=np.float32)
                stamp = stamp_ns*1e-9
                if (magic != VELOCITY_MAGIC or version != VELOCITY_VERSION
                        or enabled not in (0, 1) or reason not in (COMMAND, GATE_CLOSED, COMMAND_EXPIRED)
                        or (bool(enabled) != (reason == COMMAND))
                        or (enabled and epoch != self.navigation_epoch)
                        or stamp > now + 1e-6 or not np.isfinite(velocity).all()):
                    self.rejected += 1
                    continue
                stream = (sender, epoch)
                if sequence <= self.velocity_sequences.get(stream, -1):
                    self.rejected += 1
                    continue
                if stream not in self.velocity_sequences and len(self.velocity_sequences) >= 8:
                    del self.velocity_sequences[next(iter(self.velocity_sequences))]
                self.velocity_sequences[stream] = sequence
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
                magic, version, request, payload, sequence, stamp_ns = REQUEST_PACKET.unpack(data)
                if (magic != REQUEST_MAGIC or version != REQUEST_VERSION or request not in REQUESTS
                        or (request == 7 and payload not in PREPARE_POSES)
                        or (request == 8 and payload not in MODEL_SETS)
                        or (request not in (7, 8) and payload != 0)
                        or sequence <= self.last_request_sequence
                        or (request not in (4, 5) and not -1e-6 <= now-stamp_ns*1e-9 <= .5)):
                    self.rejected += 1
                    continue
                self.last_request_sequence = sequence
                if request == 8:
                    self.request('model_set', model_set=MODEL_SETS[payload])
                else:
                    self.request(REQUESTS[request], pose=PREPARE_POSES[payload] if request == 7 else None)
            elif data.startswith(b'{'):
                try:
                    request = json.loads(data)
                    sequence = request['sequence']
                    if (request.get('magic') != 'G1NAVCTL1' or request.get('action') not in CART_ACTIONS
                            or type(sequence) is not int or sequence <= self.last_request_sequence
                            or not 0 <= now-float(request['sent_monotonic']) <= .5
                            or not isinstance(request.get('payload', {}), dict)):
                        raise ValueError('Invalid cart control request')
                    self.last_request_sequence = sequence
                    self.cart_request_tick = (self.control.snapshot().get('state') or {}).get('tick')
                    accepted = self.request(request['action'], **request.get('payload', {}))
                    self.cart_request = dict(sequence=sequence, action=request['action'],
                                             state='queued' if accepted else 'rejected', error=self.cart_error)
                except (ValueError, KeyError, TypeError) as exc:
                    self.rejected += 1
                    self.cart_error = str(exc)
            else:
                self.rejected += 1
        # Never act on an arbitrary prefix of a saturated receive queue.
        # Continue draining next tick, with motion zero until the queue empties.
        self.enabled = False
        self._stop_motion()

    def _stop_motion(self):
        self.target.fill(0)
        self.new_motion = False
        if self.motion_active:
            self.control.emit('zero', CLIENT)
        self.motion_active = False

    def request(self, kind, *, pose=None, **payload):
        """Operator lifecycle request from the nav web page."""
        if kind in CART_ACTIONS:
            if self.pending_pose_source is not None:
                self.cart_error = '正在确认小车位姿切换，请稍候'
                return False
            data = self.control.snapshot()
            operator = dict(data, online=True, owned=data.get('owner') == CLIENT and data.get('source') == 'web',
                            armed=bool(self.control.backend.armed), mode=self.control.mode,
                            hands=data.get('dex3') or {}, cart=self._cart_status(data, self.clock()),
                            velocity=np.asarray(data.get('velocity', [0, 0, 0])).tolist())
            if not locomotion_actions(operator).get(kind):
                self.cart_error = '当前状态不允许此操作，请检查抓握、航向和策略状态'
                return False
            if self.motion_active or data.get('decelerating') or np.any(data.get('velocity', [0, 0, 0])):
                self.cart_error = '请先停止导航并等待速度归零'
                return False
            self.enabled = self.new_motion = False
            self.accept_after = self.clock()
            if kind == 'cart_pose_source':
                source = payload.get('source')
                if set(payload) != {'source'} or source not in POSE_SOURCES:
                    raise ValueError('请选择有效的小车位姿方式')
                camera_source = ((data.get('high_policy') or {}).get('camera') or {}).get('pose_source', 'grasp')
                if source != self.cart_pose_source or camera_source != source:
                    self.pending_pose_source = dict(source=source, request_id=self.last_request_sequence,
                                                    started=self.clock())
                    # Only the control thread may change camera/policy history.
                    self.control.emit('navigation_pose_source', CLIENT, source=source,
                                      request_id=self.last_request_sequence)
                    self._advance_navigation_epoch()
            elif kind == 'cart_detach':
                high = data.get('high_policy') or {}
                hands = data.get('dex3') or {}
                if high.get('enabled') or high.get('motion_blocked') or hands.get('pose') in ('grasp', 'grasp_compensated'):
                    self.cart_error = '先返回普通策略并松开抓握，再确认小车已移开'
                    return False
                self._set_attached(False)
            else:
                allowed = {'hand_edit': {'joint', 'angle_rad'}, 'hand_compensation': {'enabled'}}.get(kind, set())
                if set(payload) != allowed:
                    raise ValueError('Invalid cart action arguments')
                self.control.emit(kind, CLIENT, **payload)
            self.cart_error = ''
            return True
        elif kind == 'model_set':
            key = payload.get('model_set')
            if set(payload) != {'model_set'} or not isinstance(key, str) or key not in MODEL_SET_LABELS:
                raise ValueError('请选择有效的模型组合')
            if not self.cart_attached:
                self.control.emit('model_set', CLIENT, model_set=key)
        elif kind == 'prepare_pose':
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

    def _set_attached(self, attached):
        if self.cart_attached == attached:
            return
        self._stop_motion()
        self.cart_attached = attached
        self._advance_navigation_epoch()

    def _advance_navigation_epoch(self):
        self._stop_motion()
        self.navigation_epoch = ((((self.navigation_epoch & 0xfffffffe)+2) & 0xfffffffe) or 2) | int(self.cart_attached)
        self.accept_after = self.clock()
        self.enabled = self.new_motion = False

    def _cart_status(self, data, now):
        high = data.get('high_policy') or {}
        camera = high.get('camera') or {}
        state = data.get('state') or {}
        current_yaw = self.pelvis_yaw_history.add(state)
        result = dict(attached=self.cart_attached, mode='cart' if self.cart_attached else 'walk',
                      epoch=self.navigation_epoch, valid=False, error=self.cart_error,
                      pose_source=self.cart_pose_source,
                      pose_source_options=[dict(value=key, label=label) for key, label in POSE_SOURCES.items()],
                      task_limits=self.cart_settings['task_limits'])
        if self.pending_pose_source is not None:
            result['reason'] = '正在切换小车位姿方式'
            return result
        if camera.get('pose_source', 'grasp') != self.cart_pose_source:
            result['reason'] = '相机位姿方式与导航不一致，请重新选择位姿方式'
            return result
        stamp = camera.get('captured_monotonic')
        if not data.get('healthy'):
            result['reason'] = '机器人反馈不可用：' + (data.get('reason') or '反馈不新鲜或健康检查未通过')
            return result
        if (not camera.get('fresh') or stamp is None
                or not 0 <= now-stamp <= camera.get('timeout_s', .3)):
            result['reason'] = '小车 AprilTag 航向不可用：' + (camera.get('reason') or '相机观测已过期')
            return result
        try:
            joints = dict(zip(JOINT_NAMES, state['q']))
            if self.cart_pose_source == 'apriltag':
                observation = camera.get('cart_pose_urdf') or {}
                if (camera.get('pose_source') != 'apriltag' or observation.get('valid') is not True
                        or observation.get('captured_monotonic') != stamp):
                    raise ValueError(observation.get('reason', '等待同一帧的完整视觉位姿'))
                captured_state = self.pelvis_yaw_history.at(stamp)
                captured_joints = dict(zip(JOINT_NAMES, captured_state['q']))
                result.update(self.cart_geometry.estimate_visual(
                    captured_joints, captured_state['quat'], observation))
                result['sample_monotonic'] = stamp
            else:
                yaw, correction = self.pelvis_yaw_history.align(camera['yaw_rad'], stamp, current_yaw)
                result.update(self.cart_geometry.estimate(joints, state['quat'], yaw))
                result['sample_monotonic'] = state['received']
                result['heading_alignment'] = dict(camera_yaw_rad=camera['yaw_rad'],
                    correction_rad=correction, feedback_monotonic=state['received'],
                    captured_monotonic=stamp)
            result.update(valid=True, captured_monotonic=stamp)
        except (KeyError, ValueError, TypeError) as exc:
            result['error'] = '小车位置估计不可用：' + str(exc)
            result['reason'] = result['error']
        return result

    def _confirm_pose_source(self, data, now):
        pending = self.pending_pose_source
        if pending is None:
            return
        reply = data.get('navigation_pose_result') or {}
        if reply.get('request_id') == pending['request_id'] and reply.get('source') == pending['source']:
            if reply.get('accepted'):
                self.cart_pose_source = pending['source']
                self.cart_error = ''
                try:
                    save_pose_source(self.cart_config_path, self.cart_pose_source)
                except (OSError, ValueError) as exc:
                    self.cart_error = '位姿方式已应用，但保存失败：'+str(exc)
                self.cart_request.update(state='handled', error=self.cart_error)
            else:
                self.cart_error = reply.get('error', '位姿切换未执行')
                self.cart_request.update(state='rejected', error=self.cart_error)
            self.pending_pose_source = None
        elif now-pending['started'] > 1.:
            self.cart_error = '位姿切换未收到控制器确认，请重试'
            self.cart_request.update(state='rejected', error=self.cart_error)
            self.pending_pose_source = None

    def step(self):
        now = self.clock()
        self.new_motion = False
        self._receive(now)
        now = self.clock()
        data = self.control.snapshot()
        high = data.get('high_policy') or {}
        camera_process = self.control.high_policy.camera_process if self.control.high_policy is not None else None
        if camera_process is not None:
            camera_process.record_missing_tags(data.get('stage') == 'POLICY' and high.get('enabled'), now)
        if high.get('enabled') or high.get('motion_blocked'):
            self._set_attached(True)
        stopped = data.get('stage') in ('DAMPED', 'FAULT')
        if (data.get('state') is not None and not stopped
                and (data.get('owner') != CLIENT or data.get('source') != 'web')
                and now - self.last_claim >= 1.):
            # Own the command channel; claiming never arms motors.
            self.control.emit('claim', CLIENT)
            self.last_claim = now
        fresh = self.enabled and 0 <= now - self.last_motion <= COMMAND_TIMEOUT_S
        running = data.get('stage') == 'POLICY' and data.get('source') == 'web' and data.get('owner') == CLIENT
        if self.cart_attached:
            running = running and high.get('enabled') and not high.get('motion_blocked') and self._cart_status(data, now)['valid']
            if not running:
                self._stop_motion()
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
        # One report per executor tick also carries the latest cart geometry.
        # A second elapsed-time throttle would skip alternate ticks whenever
        # the 50 Hz scheduler wakes a little earlier than the previous tick.
        self.last_status = now
        self._confirm_pose_source(data, now)
        stage = data.get('stage', 'WAITING')
        dex3 = data.get('dex3') or {}
        owned = data.get('source') == 'web' and data.get('owner') == CLIENT
        feedback_age = now-(data.get('state') or {}).get('received', -float('inf'))
        feedback_fresh = 0 <= feedback_age <= self.control.cfg['state_timeout_s']
        ready = (stage == 'POLICY' and bool(data.get('healthy')) and feedback_fresh and owned
                 and not (data.get('heading') or {}).get('motion_blocked'))
        high = data.get('high_policy') or {}
        cart = self._cart_status(data, now)
        if self.cart_attached:
            ready = ready and bool(high.get('enabled')) and not high.get('motion_blocked') and cart['valid'] and not data.get('upper_pd_transitioning')
        # Keep the UDP status small; full inference outputs remain in the existing recorder.
        high_status = {key: high.get(key) for key in ('enabled', 'motion_blocked', 'reason', 'enable_reason', 'zero_reason',
                       'history_frames', 'history_required', 'warmup_error', 'zero_calibration', 'camera', 'camera_process',
                       'model_release')} if high else {}
        hand_status = {key: dex3.get(key) for key in ('active', 'healthy', 'pose', 'moving', 'pending',
                       'recovering', 'recovery_pending', 'blocking_reason', 'grasp', 'grasp_revision', 'grasp_dirty', 'editor_available', 'editor_message',
                       'compensation_enabled', 'compensation_available', 'file_busy', 'q', 'ever_started')}
        hand_status.update(lower=HAND_LOWER.tolist(), upper=HAND_UPPER.tolist())
        if (self.cart_request.get('state') == 'queued' and self.pending_pose_source is None
                and (data.get('state') or {}).get('tick') != self.cart_request_tick):
            self.cart_request['state'] = 'handled'
        # The controller may have switched since this snapshot was captured.
        # Report the hashes belonging to the snapshot's model, not a newer one.
        model_key = data.get('model_set', self.control.model_set)
        model_policy = (self.control.model_sets.entries[model_key]['policy']
                        if self.control.model_sets is not None else self.control.policy)
        status = dict(
            magic='G1LOCO1', ready=ready, mode=self.control.mode, stage=stage,
            stage_text=STAGES.get(stage, stage), healthy=bool(data.get('healthy')) and feedback_fresh,
            armed=bool(self.control.backend.armed), reason=data.get('reason') or '',
            source=data.get('source'), owned=data.get('owner') == CLIENT,
            model_set=model_key,
            model_set_options=[dict(value=key, label=label) for key, label in MODEL_SET_LABELS.items()],
            model_set_selectable=bool(owned and data.get('model_set_selectable') and not self.cart_attached),
            model_sha256=model_policy.source_sha256,
            model_warmup=data.get('model_warmup') or {},
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
            feedback_age_s=feedback_age if np.isfinite(feedback_age) else None,
            feedback_timeout_s=self.control.cfg['state_timeout_s'],
            heading=(data.get('heading') or {}).get('heading_state'),
            timing=data.get('timing') or {},
            high_policy=high_status, cart=cart, hands=hand_status,
            cart_request=dict(self.cart_request), upper_pd_transitioning=bool(data.get('upper_pd_transitioning')),
            pushcart_error=getattr(self.control.config, 'pushcart_error', ''),
            simulation=bool(data.get('simulation')),
            sent_monotonic=now)
        try:
            packet = json.dumps(status, ensure_ascii=False,
                                default=lambda value: value.tolist() if isinstance(value, np.ndarray) else value.item()).encode('utf-8')
            self.sock.sendto(packet, ('127.0.0.1', self.status_port))
            if self.geometry_port:
                self.sock.sendto(packet, ('127.0.0.1', self.geometry_port))
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
    parser.add_argument('--geometry-port', type=int, default=39006)
    parser.add_argument('--cart-config', default=str(NAV_CART_CONFIG))
    parser.add_argument('--record', action='store_true',
                        help='Archive this run in Viser/logs/sessions (shares its 5-session limit)')
    args = parser.parse_args()
    ports = (args.command_port, args.status_port, args.geometry_port)
    if any(not 1024 <= port <= 65535 for port in ports) or len(set(ports)) != 3:
        parser.error('Invalid UDP ports')

    config = RobotConfig(URDF_PATH, args.motion_config, None, args.heading_kp,
                         None if args.heading_mode == 'config' else args.heading_mode,
                         hand_pd_config=args.dex3_pd_config)
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    recorder = controller = executor = backend = model_sets = None
    try:
        model_sets = ModelSets(config, args.mode, args.interface, policy_mode=args.policy_mode)
        if args.record and args.mode == 'control':
            from core.recording import SessionRecorder
            recorder = SessionRecorder(config, dict(vars(args), mode='control'))
            print('RECORDING ' + str(recorder.path), flush=True)
        from core.backends import Bridge, build_bridge
        build_bridge()
        backend = Bridge(config, args.mode, args.interface)
        controller = Controller(config, backend, args.mode, recorder=recorder, policy_mode=args.policy_mode,
                                model_sets=model_sets)
        executor = Executor(controller, args.command_port, args.status_port, args.heading_source,
                            cart_config=args.cart_config, geometry_port=args.geometry_port)
        controller.start()
        print('G1 LOCOMOTION EXECUTOR READY mode=%s interface=%s heading=%s Kp=%.2f source=%s | '
              'models=%s homie_sha256=%s | '
              'cmd udp://127.0.0.1:%d status udp://127.0.0.1:%d'
              % (args.mode, args.interface, config.motion_settings['run_mode'],
                 config.motion_settings['heading_kp'], args.heading_source,
                 MODEL_SET_LABELS[controller.model_set], controller.policy.source_sha256['homie'],
                 args.command_port, args.status_port), flush=True)
        deadline = time.monotonic()
        while not stop.is_set():
            executor.step()
            # Include processing in the 20 ms period. Sleeping 20 ms after
            # every step makes the receiver slower than its 50 Hz input.
            deadline = max(deadline + EXECUTOR_PERIOD_S, time.monotonic())
            stop.wait(max(0., deadline - time.monotonic()))
    finally:
        if controller:
            controller.close()
        else:
            if backend:
                backend.close()
            if model_sets:
                model_sets.close()
        if executor:
            executor.close()
        if recorder:
            recorder.close(error=controller.reason if controller and controller.stage == 'FAULT' else None)


if __name__ == '__main__':
    main()
