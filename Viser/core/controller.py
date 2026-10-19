"""Outer Viser controls the existing hybrid policies and C++ servo bridge."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import queue
import threading
import time

import numpy as np

from core import robot  # Makes the sibling deployment package importable.
from hybrid_deploy import protocol as p
from hybrid_deploy.policy import Observations, Commands, targets
from core.policy import Policy
from hybrid_deploy.web_control import WebControl, WEB_TIMEOUT_S
from core.heading import HeadingControl
from core.joint_tracking import snapshot as joint_snapshot
from core.hand_status import feedback_reasons
from core.hand_grasp import load_grasp, save_grasp, validate_grasp

STAGES = dict(WAITING='等待接管', ACQUIRING='正在接管底层', ZERO='已接管，等待准备',
              PREPARING='移到准备姿态', HOLDING='准备完成', POLICY='Balance / Homie 策略运行',
              DAMPED='阻尼停止（可重新接管网页）', FAULT='故障停止（可重新连接检查）', OBSERVE='只读同步')
HAND_POSES = {1: 'open', 2: 'grasp', 3: 'open_compensated', 4: 'grasp_compensated'}


class Controller:
    def __init__(self, config, backend, mode, recorder=None, policy_mode='auto', clock=time.monotonic,
                 high_policy=None, model_sets=None):
        self.config, self.backend, self.mode, self.recorder = config, backend, mode, recorder
        self.clock = clock
        self.model_sets = model_sets
        self.model_set = getattr(config, 'model_set', 'v81')
        self.high_policy = high_policy
        self.cfg = config.deployment
        if model_sets is not None:
            self.policy, self.high_policy = model_sets.activate(config, self.model_set)
        else:
            self.policy = Policy(config.deploy, self.cfg, policy_mode=policy_mode,
                                 pushcart_homie=high_policy.homie_model if high_policy is not None else None)
        self.observations = Observations(config.contracts)
        self.commands = Commands(self.cfg)
        self.web = WebControl(self.cfg, enabled=mode != 'observe')
        self.heading = HeadingControl(config.motion_settings, self.cfg['command_limits'],
                                      self.cfg['command_slew_per_s'],
                                      command_filter_tau_s=self.cfg['command_filter_tau_s'],
                                      model_switch_command_norm=self.cfg['model_switch_command_norm'])
        self.motion_time = None
        self.motion_revision = 0
        self.zeroing = False
        self.events = queue.Queue(maxsize=128)
        self.stop_requested = threading.Event()
        self.stopped_at = -float('inf')
        self.shutdown = threading.Event()
        self.lock = threading.Lock()
        self.latest = {}
        self.thread = None
        self.stage = 'OBSERVE' if mode == 'observe' else 'WAITING'
        self.reason = ''
        self.navigation_pose_result = {}
        self.request = p.WAIT
        self.prepare_pose = 'original'
        self.hand_pose = 0
        self.hand_ever_started = False
        self.hand_resume_started = None
        self.hand_compensation = False
        self.hand_live = False
        self.hand_grasp_revision = 0
        self.hand_saved_grasp = config.hand_grasp.copy()
        self.hand_editor_message = ''
        self.hand_file = None
        self.hand_file_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='dex3-pose-file')
        self.hand_recovery = -1
        self.hand_recovery_seen = False
        self.hand_recovery_started = 0.
        self.previous_stage = None
        self.buttons = None
        self.policy_steps = 0
        self.ticks, self.results = deque(), deque()

    def record_event(self, **event):
        if self.recorder:
            self.recorder.record('events', event)

    def emit(self, kind, client, velocity=None, *, command_time=None, **payload):
        if self.mode == 'observe':
            return
        data = self.snapshot()
        if kind == 'stop' or (kind == 'disconnect' and data.get('owner') == client):
            self.reason = '网页持有者断开' if kind == 'disconnect' else '网页阻尼停止'
            self.stop_requested.set()
            if kind == 'stop':
                return
        state = data.get('state')
        if state is None and kind not in ('claim', 'disconnect', 'reset'):
            return
        # Navigation carries the original /cmd_vel time through loopback IPC.
        # Replaying a cached command must not restart the 250 ms watchdog.
        stamp = self.clock() if command_time is None else command_time
        event = dict(kind=kind, client=client, velocity=velocity, time=stamp,
                     session=state['session'] if state else None,
                     feedback_ns=state['received_ns'] if state else None, **payload)
        try:
            self.events.put_nowait(event)
        except queue.Full:
            # Motion expires independently. Emergency stop never uses this queue.
            pass

    def snapshot(self):
        with self.lock:
            return self.latest.copy()

    def _restart(self, armed):
        if self.high_policy is not None:
            self.high_policy.reset()
        if not armed:
            # Consume the old stop BEFORE starting; a new stop during startup
            # must remain set and win on the following controller tick.
            self.stop_requested.clear()
        while not self.events.empty():
            try:
                self.events.get_nowait()
            except queue.Empty:
                break
        self.zeroing = False
        self.web.zero()
        self.commands.value.fill(0)
        self.observations.reset()
        self.policy.reset_selection()
        self.policy.homie_estimator = None
        self.heading.clear()
        self.motion_time = None
        self.request, self.previous_stage, self.buttons = p.WAIT, None, None
        self.hand_pose = 0
        self.hand_ever_started = False
        self.hand_resume_started = None
        self.hand_compensation = False
        self.hand_live = False
        self.hand_recovery = -1
        self.hand_recovery_seen = False
        self.stage = 'ACQUIRING' if armed else 'WAITING'
        with self.lock:
            self.latest = dict(self.latest, stage=self.stage, healthy=False, state=None,
                               preparation_warning='',
                               prepare_pose=self.prepare_pose, prepare_pose_selectable=False,
                               model_set=self.model_set, model_set_selectable=False,
                               owner=self.web.owner, source=self.web.source, velocity=np.zeros(3), dex3={},
                               joint_comparison=None,
                               high_policy=self.high_policy.snapshot(self.clock()) if self.high_policy else None,
                               upper_pd_mode='hold', upper_pd_requested='hold',
                               upper_pd_transition_blend=1., upper_pd_transitioning=False,
                               model_actions=None, homie_estimator=None, homie_estimator_source=None,
                               policy_inference_monotonic=None)
        self.reason = ''
        self.backend.start(armed=armed)
        self.record_event(type='bridge_start', armed=armed, simulation=self.mode == 'sim')

    def damp(self, reason, fault=False):
        if self.stage in ('DAMPED', 'FAULT'):
            return
        self.reason = reason
        if self.high_policy is not None:
            self.high_policy.reset(reason)
        self.stopped_at = self.clock()
        self.stage = 'FAULT' if fault else 'DAMPED'
        self.zeroing = False
        self.web.zero()
        self.commands.value.fill(0)
        self.heading.stop(reason)
        if fault and self.heading.last:
            self.heading.last['imu_health'] = 'controller_fault: '+reason
        state = self.backend.state
        if state is not None:
            self.backend.send(p.STOP, state.applied_q)
        elif self.backend.process is not None and self.backend.process.poll() is None:
            # Cancel a pending real-robot preflight/mode release as well.
            self.backend.process.terminate()
        self.record_event(type='damping', reason=reason, stage=self.stage)
        self._publish_stopped()
        if fault:
            # Navigation normally runs without SessionRecorder. Preserve the
            # exact fault in launch.log as well as the transient web status.
            print('CONTROL FAULT: ' + reason, flush=True)

    def _publish_stopped(self):
        with self.lock:
            self.latest = dict(self.latest, stage=self.stage, reason=self.reason, healthy=False,
                               preparation_warning='',
                               prepare_pose=self.prepare_pose, prepare_pose_selectable=False,
                               model_set=self.model_set, model_set_selectable=False,
                               velocity=np.zeros(3), input_velocity=np.zeros(3),
                               decelerating=False,
                               model_actions=None, homie_estimator=None, homie_estimator_source=None,
                               policy_inference_monotonic=None,
                               heading=self.heading.last.copy(),
                               upper_pd_requested='hold', upper_pd_transitioning=False,
                               high_policy=self.high_policy.snapshot(self.clock()) if self.high_policy else None,
                               owner=self.web.owner, source=self.web.source)

    def _hand_status(self, state):
        flags = state.hand_flags
        pose = (2 if flags & 16 else 1) + (2 if flags & 64 else 0)
        reasons = feedback_reasons(state, self.clock())
        return dict(healthy=(flags & 35) == 35, left_healthy=bool(flags & 1),
                    left_reason=reasons[0], right_reason=reasons[1],
                    blocking_reason='；'.join(reason for reason in reasons if reason),
                    right_healthy=bool(flags & 2), active=bool(flags & 4),
                    ever_started=self.hand_ever_started or bool(flags & 4),
                    moving=bool(flags & 8), pose=HAND_POSES[pose],
                    editing=bool(flags & 256),
                    pending=(self.hand_pose not in (0, pose) or self.hand_resume_started is not None),
                    progress=state.hand_progress,
                    compensation_enabled=self.hand_compensation,
                    compensation_available=self.config.hand_grasp_compensated is not None,
                    recovering=bool(flags & 128), recovery_pending=self.hand_recovery >= 0,
                    recovery_mask=getattr(state, 'hand_recovery_mask', 0),
                    recovery_joint=getattr(state, 'hand_recovery_joint', -1),
                    zero_compensation_available=self.config.hand_zero is not None,
                    zero_compensation_error=self.config.hand_zero_error,
                    grasp_compensation_available=self.config.hand_grasp_compensated is not None,
                    grasp_compensation_error=self.config.hand_grasp_compensation_error,
                    grasp=self.config.hand_grasp.copy(), grasp_revision=self.hand_grasp_revision,
                    grasp_dirty=not np.array_equal(self.config.hand_grasp, self.hand_saved_grasp),
                    grasp_path=str(self.config.upper_pose_path),
                    editor_available=self._can_edit_hand(state),
                    file_busy=self.hand_file is not None, editor_message=self.hand_editor_message,
                    q=state.hand_q.copy(), dq=state.hand_dq.copy(), target=state.hand_target.copy())

    def _can_edit_hand(self, state):
        return (self.mode != 'observe' and self.backend.armed and state.flags & 1
                and not (self.high_policy and self.high_policy.enabled)
                and state.stage in (p.HOLDING, p.POLICY) and not state.buttons & (p.B | p.L2)
                and self.stage not in ('DAMPED', 'FAULT') and not self.stop_requested.is_set()
                and (state.hand_flags & 39) == 39 and not state.hand_flags & 128
                and self.hand_resume_started is None and self.hand_recovery < 0
                and (not state.hand_flags & 8 or bool(state.hand_flags & 256))
                and (self.hand_live or self.hand_pose in (0, (2 if state.hand_flags & 16 else 1)
                                                         + (2 if state.hand_flags & 64 else 0)))
                and self.hand_file is None)

    def _apply_hand_grasp(self, angles):
        zero = self.config.hand_zero if self.hand_compensation else None
        if self.hand_compensation and zero is None:
            raise ValueError('Dex3 补偿文件不可用')
        raw = validate_grasp(angles, zero)
        self.config.set_hand_grasp(raw)
        self.hand_pose = 4 if self.hand_compensation else 2
        self.hand_live = True
        self.hand_grasp_revision += 1
        self.hand_editor_message = '正在应用抓握角度（单关节最高 0.5 rad/s）'
        self.record_event(type='dex3_grasp_edited', raw_rad=raw.copy(), compensated=self.hand_compensation)

    def _poll_hand_file(self, state):
        if self.hand_file is None or not self.hand_file[0].done():
            return
        future, kind, context, angles = self.hand_file
        self.hand_file = None
        try:
            result = future.result()
            if kind == 'hand_save':
                self.hand_saved_grasp = angles
                self.hand_editor_message = '抓握角度已保存到 ' + str(self.config.upper_pose_path)
            elif (state is not None and context == (state.session, self.web.owner, self.hand_pose,
                                                    self.hand_compensation, self.hand_grasp_revision)
                  and self.web.source == 'web' and self._can_edit_hand(state)):
                self._apply_hand_grasp(result)
                self.hand_saved_grasp = result.copy()
                self.hand_editor_message = '已加载文件，正在应用抓握角度'
            else:
                self.hand_editor_message = '加载已取消：控制状态改变，请重新点击加载'
            self.record_event(type='dex3_grasp_file', operation=kind, message=self.hand_editor_message)
        except Exception as exc:
            self.hand_editor_message = '抓握文件操作失败：' + str(exc)
            self.record_event(type='dex3_grasp_file_error', operation=kind, reason=str(exc))

    def _prepare(self, state):
        if state.stage != p.WAITING or self.request != p.WAIT:
            return
        # Body preparation never requests hand motion or hand enablement.
        self.hand_recovery = -1
        self.hand_recovery_seen = False
        self.hand_resume_started = None
        self.hand_pose = 0
        self.request = p.PREPARE
        self.reason = ''
        if (self.high_policy is not None and self.high_policy.camera_process is not None
                and self.mode == 'control' and not self.high_policy.camera.fresh(self.clock())):
            try:
                self.high_policy.camera_process.request_start()
                self.record_event(type='cart_camera_preparation_start')
            except (OSError, ValueError, RuntimeError) as exc:
                # Camera availability is independent of body preparation.
                self.record_event(type='cart_camera_start_failed', reason=str(exc))
        self.record_event(type='preparation_started', pose=self.prepare_pose,
                          model_set=self.model_set,
                          config_path=str(self.config.upper_pose_paths[self.prepare_pose]))

    @property
    def prepare_q(self):
        return self.config.prepare_poses[self.prepare_pose]

    def _can_select_prepare_pose(self, state):
        return (self.mode != 'observe' and state.stage == p.WAITING and self.request == p.WAIT
                and bool(state.flags & 1) and not state.buttons & (p.B | p.L2)
                and self.stage not in ('DAMPED', 'FAULT') and not self.stop_requested.is_set())

    def _select_model_set(self, key):
        if self.high_policy is not None:
            self.high_policy.reset()
        self.policy, self.high_policy = self.model_sets.activate(self.config, key)
        self.model_set = key
        self.cfg = self.config.deployment
        self.policy.reset_selection()
        self.policy.pushcart_active = False
        self.policy.homie_estimator = self.policy.homie_estimator_source = None
        if self.high_policy is not None:
            self.high_policy.reset()
        self.observations = Observations(self.config.contracts)
        self.web.zero()
        self.commands.value.fill(0)
        self.heading.clear()
        self.motion_time = None
        self.reason = ''
        self.record_event(type='model_set_selected', model_set=key,
                          model_sha256=self.policy.source_sha256,
                          high_policy_release=self.high_policy.cfg['release'] if self.high_policy else None)

    def _preparation_warning(self, state):
        if state.stage != p.RAMPING or not state.flags & p.PREPARATION_OVERDUE:
            return ''
        pending = []
        for i in robot.ACTIVE:
            details = []
            error = self.prepare_q[i] - state.q[i]
            if abs(error) > self.cfg['calibration_position_tolerance_rad']:
                details.append('位置误差 %+.2f°' % np.rad2deg(error))
            if abs(state.dq[i]) > self.cfg['calibration_velocity_tolerance_rad_s']:
                details.append('速度 %+.3f rad/s' % state.dq[i])
            if abs(state.applied_q[i] - self.prepare_q[i]) >= 1e-4:
                details.append('目标仍在过渡')
            if details:
                pending.append('%s（%s）' % (robot.JOINT_NAMES[i], '，'.join(details)))
        return ('准备姿态超时（%g 秒）：继续保持位置控制，等待实测到位。未到位关节：%s'
                % (self.cfg['calibration_timeout_s'], '；'.join(pending) or '等待下一帧确认'))

    def _events(self, state):
        for _ in range(128):
            try:
                event = self.events.get_nowait()
            except queue.Empty:
                break
            self.record_event(type='web_event', event=event)
            if event['kind'] == 'disconnect' and self.web.owner == event['client']:
                self.web.owner = None
                self.web.zero()
                self._publish_stopped()
                continue
            # A stopped bridge has exited, so its feedback is necessarily stale.
            # Reclaim only restarts a read-only bridge. Motor acquire still needs
            # new healthy feedback and its own explicit button press.
            if (self.stage in ('DAMPED', 'FAULT') and event['kind'] in ('claim', 'reset')
                    and (event['kind'] == 'claim' or self.mode == 'sim')
                    and self.web.owner in (None, event['client'])
                    and self.stopped_at <= event['time']
                    and 0 <= self.clock() - event['time'] <= .25):
                self.web.owner, self.web.source = event['client'], 'web'
                self._restart(False)
                return True
            if self.stage in ('DAMPED', 'FAULT') or state is None:
                continue
            if self.zeroing and event['kind'] == 'move':
                # Queued hold events cannot interrupt an explicit stop ramp.
                continue
            if event['kind'] == 'move' and event.get('command_source') == 'NAVIGATION':
                velocity = np.asarray(event['velocity'], dtype=float)
                if velocity.shape == (3,) and np.isfinite(velocity).all() and not np.any(velocity):
                    event = dict(event, kind='zero')
            if (event['kind'] == 'reset' and self.mode == 'sim'
                    and self.stage in ('WAITING', 'DAMPED', 'FAULT')
                    and self.web.owner == event['client']
                    and 0 <= self.clock() - event['time'] <= .25):
                self._restart(False)
                return True
            previous_source = self.web.source
            result = self.web.accept(event, state, self.observations.frames, now=self.clock())
            accepted_zero = event['kind'] == 'zero' and event['time'] == self.web.last_event
            if accepted_zero:
                was_zeroing = self.zeroing
                self.zeroing = bool(self.mode != 'observe' and state.stage == p.POLICY
                                    and state.flags & 1 and not state.buttons & (p.B | p.L2)
                                    and np.any(self.heading.output))
                self.commands.value.fill(0)
                if self.zeroing and not was_zeroing:
                    self.record_event(type='smooth_zero_started', velocity=self.heading.output.copy(),
                                      filter_tau_s=self.heading.filter_tau, trigger='zero_button')
            if self.web.source != previous_source:
                if self.high_policy is not None:
                    self.high_policy.reset('控制来源已切换，高层已关闭')
                self.commands.value.fill(0)
                if self.hand_recovery >= 0:
                    self.damp('Dex3 恢复过程中控制来源改变')
                if not accepted_zero:
                    self.zeroing = False
                    self.heading.stop('command_source_changed')
            if result == p.STOP:
                self.damp('网页阻尼停止')
            if event['time'] != self.web.last_event:
                continue
            owned = self.web.source == 'web' and self.web.owner == event['client']
            if self.stage in ('DAMPED', 'FAULT'):
                continue
            if event['kind'] == 'navigation_pose_source':
                source = event.get('source')
                reply = dict(request_id=event.get('request_id'), source=source, accepted=False,
                             error='位姿切换需要保持静止、退出推车策略并持有控制权')
                highs = ([entry['high'] for entry in self.model_sets.entries.values()]
                         if self.model_sets is not None else [self.high_policy])
                if (owned and state.flags & 1 and not state.buttons & (p.B | p.L2)
                        and source in ('grasp', 'apriltag') and self.high_policy is not None
                        and not self.zeroing and not np.any(self.heading.output) and not np.any(self.web.target)
                        and all(high is None or (not high.enabled and not high.motion_blocked) for high in highs)):
                    for high in highs:
                        if high is not None:
                            high.set_navigation_pose_source(source, self.clock())
                    reply.update(accepted=True, error='')
                self.navigation_pose_result = reply
                self.record_event(type='navigation_pose_source', **reply)
                continue
            if event['kind'] == 'cart_camera_start':
                if owned and self.high_policy and self.high_policy.camera_process:
                    try:
                        if not self.high_policy.camera.fresh(self.clock()):
                            self.high_policy.camera_process.request_start()
                        self.reason = ''
                    except (OSError, ValueError) as exc:
                        self.reason = str(exc)
                continue
            if event['kind'] == 'cart_yaw_zero':
                if owned and state.flags & 1 and not state.buttons & (p.B | p.L2) and self.high_policy:
                    try:
                        self.high_policy.start_zero_calibration(self, state, self.clock())
                        self.reason = ''
                    except ValueError as exc:
                        self.reason = str(exc)
                    self.record_event(type='cart_yaw_zero_requested', reason=self.reason)
                continue
            if event['kind'] in ('pushcart_toggle', 'pushcart_enable', 'pushcart_disable'):
                if owned and state.flags & 1 and not state.buttons & (p.B | p.L2) and self.high_policy:
                    try:
                        enable = not self.high_policy.enabled if event['kind'] == 'pushcart_toggle' else event['kind'] == 'pushcart_enable'
                        changed = self.high_policy.set_enabled(enable, self, state, self.clock())
                        if changed:
                            self.web.zero()
                            self.commands.value.fill(0)
                            self.heading.rebase(state.quaternion)
                        self.reason = ''
                    except ValueError as exc:
                        self.reason = str(exc)
                    self.record_event(type='pushcart_toggle', enabled=self.high_policy.enabled,
                                      reason=self.reason)
                continue
            if (self.high_policy and self.high_policy.enabled
                    and event['kind'] in ('hand_toggle', 'hand_compensation', 'hand_edit', 'hand_load', 'hand_save')):
                self.reason = '先归零并返回普通行走，再调节或松开抓握'
                continue
            if event['kind'] == 'model_set':
                key = event.get('model_set')
                if (owned and self.model_sets is not None and self._can_select_prepare_pose(state)
                        and isinstance(key, str) and key in self.model_sets.entries and key != self.model_set):
                    self._select_model_set(key)
                continue
            if event['kind'] == 'prepare_pose':
                pose = event.get('pose')
                if (owned and self._can_select_prepare_pose(state)
                        and isinstance(pose, str) and pose in self.config.prepare_poses
                        and pose != self.prepare_pose):
                    self.prepare_pose = pose
                    self.reason = ''
                    self.record_event(type='preparation_pose_selected', pose=pose,
                                      config_path=str(self.config.upper_pose_paths[pose]))
                continue
            if event['kind'] == 'hand_compensation':
                hands = self._hand_status(state)
                if (owned and self.backend.armed and state.flags & 1
                        and state.stage in (p.WAITING, p.HOLDING, p.POLICY)
                        and not state.buttons & (p.B | p.L2) and not hands['moving']
                        and not hands['pending'] and not hands['recovering'] and self.hand_recovery < 0
                        and self.hand_file is None
                        and type(event.get('enabled')) is bool):
                    enabled = event['enabled']
                    if enabled and self.config.hand_grasp_compensated is None:
                        self.reason = 'Dex3 补偿不可用：' + self.config.hand_grasp_compensation_error
                    else:
                        self.hand_compensation = enabled
                        self.reason = ''
                        self.record_event(type='dex3_compensation', enabled=enabled)
                continue
            if event['kind'] in ('hand_edit', 'hand_save', 'hand_load'):
                if owned and self._can_edit_hand(state):
                    try:
                        if event['kind'] == 'hand_edit':
                            index, angle = event.get('joint'), event.get('angle_rad')
                            if (type(index) is not int or not 0 <= index < 14
                                    or type(angle) not in (int, float)):
                                raise ValueError('无效的 Dex3 关节角度')
                            angles = self.config.hand_grasp.copy()
                            angles.flat[index] = angle
                            self._apply_hand_grasp(angles)
                        else:
                            angles = self.config.hand_grasp.copy()
                            context = (state.session, self.web.owner, self.hand_pose,
                                       self.hand_compensation, self.hand_grasp_revision)
                            future = (self.hand_file_executor.submit(save_grasp, self.config.upper_pose_path, angles)
                                      if event['kind'] == 'hand_save' else
                                      self.hand_file_executor.submit(load_grasp, self.config.upper_pose_path))
                            self.hand_file = (future, event['kind'], context, angles)
                            self.hand_editor_message = '正在保存抓握角度…' if event['kind'] == 'hand_save' else '正在加载抓握角度…'
                    except (ValueError, TypeError, OverflowError) as exc:
                        self.hand_grasp_revision += 1
                        self.hand_editor_message = '抓握调节未应用：' + str(exc)
                else:
                    # Also restore a stale/unauthorized client's optimistic slider value.
                    self.hand_grasp_revision += 1
                continue
            if event['kind'] == 'hand_recover':
                index = event.get('joint')
                if (owned and self.backend.armed and state.flags & 1 and state.stage == p.WAITING
                        and not state.buttons & (p.B | p.L2) and not state.hand_flags & (4 | 8 | 128)
                        and self.hand_recovery < 0 and type(index) is int and 0 <= index < 14
                        and getattr(state, 'hand_recovery_mask', 0) & (1 << index)):
                    self.hand_recovery = index
                    self.hand_recovery_seen = False
                    self.hand_recovery_started = self.clock()
                    self.request = p.WAIT
                    self.reason = ''
                    self.record_event(type='dex3_recovery_requested', joint=index)
                continue
            if event['kind'] == 'hand_toggle':
                hands = self._hand_status(state)
                if (owned and self.backend.armed and state.flags & 1
                        and not state.buttons & (p.B | p.L2)
                        and state.stage in (p.HOLDING, p.POLICY) and hands['healthy']
                        and not hands['recovering'] and not hands['moving'] and not hands['pending']
                        and self.hand_file is None):
                    self.hand_live = False
                    if not hands['active']:
                        # First use performs the button's grasp action; an explicit
                        # restart after a local hand fault still returns to zero.
                        self.hand_pose = (1 if hands['ever_started'] else 2) + (2 if self.hand_compensation else 0)
                        self.hand_resume_started = self.clock()
                        self.reason = ''
                        self.record_event(type='dex3_resume_requested' if hands['ever_started'] else 'dex3_enable_requested')
                        self.hand_ever_started = True
                    else:
                        self.hand_pose = (1 if self.hand_pose in (2, 4) else 2) + (2 if self.hand_compensation else 0)
                    self.record_event(type='dex3_pose', pose=HAND_POSES[self.hand_pose],
                                      transition_s=3.)
                continue
            if event['kind'] in ('motion_settings', 'rebase_heading') and owned and state.flags & 1:
                if (np.any(self.web.target) or np.any(self.web.value) or np.any(self.heading.output)):
                    self.reason = '先点击速度归零，再调整航向参数或重设航向目标'
                    self.record_event(type='motion_settings_rejected', reason=self.reason)
                    continue
                try:
                    if event['kind'] == 'motion_settings':
                        patch = event.get('settings')
                        required = {'command_frame', 'heading_kp', 'max_correction_rad_s'}
                        if (not isinstance(patch, dict) or not required <= set(patch)
                                or set(patch) - required - {'gyro_cutoff_hz', 'run_mode'}):
                            raise ValueError('无效航向参数')
                        if patch['command_frame'] != 'body':
                            raise ValueError('速度指令必须使用机身坐标；IMU 仅修正转向')
                        self.heading.configure(dict(self.heading.settings, **patch), state.quaternion)
                    else:
                        self.heading.rebase(state.quaternion)
                    self.motion_revision += 1
                    self.reason = '航向参数已应用' if event['kind'] == 'motion_settings' else '航向目标已设为当前朝向'
                    self.record_event(type='motion_settings', kind=event['kind'],
                                      settings=self.heading.settings.copy(),
                                      heading_origin_yaw_rad=self.heading.origin,
                                      target_yaw_rad=self.heading.target)
                except (ValueError, TypeError) as exc:
                    self.reason = str(exc)
                    self.record_event(type='motion_settings_rejected', reason=self.reason)
                continue
            if (event['kind'] == 'acquire' and owned and self.stage == 'WAITING'
                    and state.flags & 1 and not state.buttons & p.L2):
                self._restart(True)
                return True
            if result == p.PREPARE and self.backend.armed:
                self._prepare(state)
            elif result == p.RUN and self.backend.armed:
                self.web.zero()
                self.commands.value.fill(0)
                self.request = result
        return False

    @staticmethod
    def _rate(samples, now):
        while samples and samples[0] < now - 2:
            samples.popleft()
        return (len(samples)-1)/(samples[-1]-samples[0]) if len(samples) > 1 else 0.

    def step(self):
        if self.high_policy is not None:
            self.high_policy.begin_tick(self.clock())
        self._poll_hand_file(self.backend.state)
        if self.stop_requested.is_set():
            self.damp(self.reason or '停止请求')
        if self.stage in ('DAMPED', 'FAULT'):
            # Record the bridge's real damping feedback until its shutdown completes.
            try:
                state = self.backend.read()
            except Exception:
                state = self.backend.state
            if state is not None:
                with self.lock:
                    data = self.latest.copy()
                    feedback = dict(data.get('state') or {})
                    feedback.update(q=state.q.copy(), dq=state.dq.copy(), quat=state.quaternion.copy(),
                                    omega=state.omega.copy(), tick=(state.session, state.sequence),
                                    session=state.session, received_ns=state.received_ns,
                                    received=state.received_ns * 1e-9)
                    data.update(state=feedback, target=state.applied_q.copy(), dex3=self._hand_status(state),
                                joint_comparison=joint_snapshot(state, self.clock(), self.cfg['state_timeout_s']),
                                kp=state.applied_kp.copy(), kd=state.applied_kd.copy(),
                                bridge_motor_control_active=self.mode == 'control' and bool(np.any(state.applied_kd)),
                                recording_error=self.recorder.error if self.recorder else '',
                                recording_dropped=self.recorder.dropped if self.recorder else 0,
                                timing=dict(control_hz=0., policy_hz=0., inference_ms=None,
                                            observation_to_action_ms=None))
                    self.latest = data
                if self.recorder:
                    self.recorder.record('telemetry', data)
            self._events(state)
            return
        state = self.backend.read()
        now = self.clock()
        if state is None or state.sequence == 0:
            if now - self.backend.started > self.cfg['startup_timeout_s']:
                raise RuntimeError('等待有效 LowState 超时')
            with self.lock:
                self.latest = dict(self.latest, stage=self.stage, reason='等待控制桥接反馈',
                                   healthy=False, velocity=np.zeros(3), owner=self.web.owner,
                                   source=self.web.source, state=None, joint_comparison=None)
            return
        age = now - state.received_ns * 1e-9
        if not 0 <= age <= self.cfg['state_timeout_s']:
            raise RuntimeError('状态反馈超过 %.0f ms 未更新' % (self.cfg['state_timeout_s'] * 1000))
        if state.stage in (p.DAMPED, p.FAULT):
            self.damp('控制桥接已停止：' + p.STAGE_NAMES[state.stage], state.stage == p.FAULT)
            return
        entered_policy = state.stage == p.POLICY and self.previous_stage != p.POLICY
        if state.stage != self.previous_stage:
            if state.stage in (p.WAITING, p.RAMPING, p.HOLDING, p.POLICY):
                self.zeroing = False
                self.web.zero()
                self.commands.value.fill(0)
            if state.stage in (p.WAITING, p.RAMPING):
                self.observations.reset()
                self.policy.reset_selection()
                if self.high_policy is not None:
                    self.high_policy.reset()
            self.previous_stage = state.stage
        self.stage = ('OBSERVE' if self.mode == 'observe' else
                      ('ZERO' if self.backend.armed else 'WAITING') if state.stage == p.WAITING else
                      'PREPARING' if state.stage == p.RAMPING else p.STAGE_NAMES[state.stage])
        if self.mode != 'observe' and state.buttons & p.L2:
            self.damp('实体遥控 L2 停止')
            return
        if state.stage in (p.RAMPING, p.HOLDING, p.POLICY):
            if state.hand_flags & 4:
                self.hand_ever_started = True
                self.hand_resume_started = None
            elif self.hand_resume_started is None:
                self.hand_pose = 0  # Acknowledge local hand suspension; never auto-resume.
                self.hand_live = False
            elif now - self.hand_resume_started > 1.:
                self.hand_resume_started = None
                self.hand_pose = 0
                self.reason = 'Dex3 重新启用未被底层接受，请检查手部通信后重试；机身控制继续'
        if self._events(state):
            return
        if self.stage in ('DAMPED', 'FAULT'):
            return
        if self.high_policy is not None:
            if state.buttons & (p.B | p.L2):
                self.high_policy.cancel_zero_calibration('实体遥控停止，标定取消；保留原零位')
            result = self.high_policy.update_zero_calibration(self, state, self.clock())
            if result is not None:
                self.record_event(type='cart_yaw_zero_result', **result)
        if self.hand_recovery >= 0:
            if state.hand_flags & 128:
                self.hand_recovery_seen = True
            elif self.hand_recovery_seen or not getattr(state, 'hand_recovery_mask', 0) & (1 << self.hand_recovery):
                self.record_event(type='dex3_recovery_finished', joint=self.hand_recovery)
                self.hand_recovery = -1
            elif self.clock() - self.hand_recovery_started > 1.:
                self.reason = 'Dex3 恢复请求未被底层接受，请检查反馈后重试'
                self.hand_recovery = -1
        edge = self.buttons is not None and state.buttons & p.R2 and not self.buttons & p.R2
        self.buttons = state.buttons
        if edge and self.web.source == 'remote' and state.flags & 1 and self.mode != 'observe':
            if not self.backend.armed:
                self._restart(True)
                return
            if state.stage == p.WAITING:
                self._prepare(state)
            elif state.stage == p.HOLDING and self.observations.frames >= 6:
                self.commands.value.fill(0)
                self.request = p.RUN
        if state.stage == p.HOLDING and self.request != p.RUN:
            self.request = p.HOLD
        if not state.flags & 1 and self.backend.armed and state.stage != p.WAITING:
            raise RuntimeError('控制桥接健康检查未通过')
        if state.stage != p.POLICY or not state.flags & 1 or state.buttons & (p.B | p.L2):
            self.zeroing = False
        now_motion = self.clock()
        dt = 1/self.cfg['policy_hz'] if self.motion_time is None else now_motion-self.motion_time
        self.motion_time = now_motion
        if (not self.zeroing and self.mode != 'observe' and state.stage == p.POLICY
                and state.flags & 1 and not state.buttons & (p.B | p.L2)
                and self.web.source == 'web' and self.web.owner is not None
                and self.web.command_source in ('MANUAL', 'NAVIGATION') and np.isfinite(self.web.last_motion)
                and now_motion-self.web.last_motion > WEB_TIMEOUT_S and np.any(self.heading.output)):
            self.zeroing = True
            self.record_event(type='smooth_zero_started',
                              trigger='navigation_timeout' if self.web.command_source == 'NAVIGATION' else 'hold_released',
                              velocity=self.heading.output.copy(), filter_tau_s=self.heading.filter_tau)
        if self.mode == 'observe':
            velocity = np.zeros(3, dtype=np.float32)
        elif self.web.source == 'web':
            velocity = self.web.update(state, dt, now=now_motion)
        else:
            # Do not adopt cached sticks on the first policy frame.
            velocity = self.commands.update(state, dt) if state.stage == p.POLICY else np.zeros(3)
        if entered_policy:
            self.web.zero()
            self.commands.value.fill(0)
            velocity = np.zeros(3, dtype=np.float32)
            self.heading.rebase(state.quaternion)
            self.record_event(type='heading_reference', heading_origin_yaw_rad=self.heading.origin,
                              target_yaw_rad=self.heading.target)
        # Keep physical remote sticks in their original body-frame convention.
        # Physical remote deadman/B still bypass the outer loop.
        if self.web.source == 'remote' and (not state.buttons & p.L1 or state.buttons & p.B):
            self.commands.value.fill(0)
            velocity = np.zeros(3, dtype=np.float32)
        input_velocity = np.asarray(velocity).copy()
        if self.zeroing:
            input_velocity.fill(0)
        raw_command = (self.web.target.copy() if self.web.source == 'web' else self.commands.target.copy())
        live_command = (0 <= now_motion-self.web.last_motion <= WEB_TIMEOUT_S if self.web.source == 'web'
                        else bool(state.buttons & p.L1 and not state.buttons & p.B))
        if entered_policy or self.zeroing or self.mode == 'observe':
            raw_command.fill(0)
        command_source = self.web.command_source if self.web.source == 'web' else 'MANUAL'
        high = self.high_policy
        high_sampled = None
        upper_transitioning = False
        if high is not None and self.mode != 'observe' and state.stage == p.POLICY:
            if high.enabled:
                hands = self._hand_status(state)
                if (not hands['healthy'] or not hands['active'] or hands['moving'] or hands['pending']
                        or hands['pose'] not in ('grasp', 'grasp_compensated')):
                    high.reset('抓握已中断；高层暂停，请归零并重新启用', blocked=True)
            upper_transitioning = (state.upper_mode != int(high.enabled) or state.upper_gain_blend < 1.)
            if upper_transitioning:
                raw_command.fill(0)
                input_velocity.fill(0)
                live_command = False
                self.web.zero()
                self.commands.value.fill(0)
            if high.enabled or high.motion_blocked:
                moving = bool(live_command and np.any(raw_command) and not self.zeroing)
                if high.enabled:
                    high_sampled = self.clock()
                    input_velocity, failed = high.command(
                        state, raw_command, self.observations.models['homie'].previous_action,
                        now_motion, moving=moving, dt=dt,
                        previous_state=getattr(self.backend, 'previous_state', None))
                    moving = moving and not failed
                else:
                    input_velocity = np.zeros(3)
                    moving = False
                # Cart heading feedback is applied before the student. Pass its
                # Homie command through unchanged, retaining Viser's stop path.
                command_source = 'PUSHCART'
                if not moving:
                    raw_command.fill(0)
                    input_velocity.fill(0)
                    # Physical B bypasses the smooth stop, also in push-cart
                    # mode. Do not re-arm deceleration after WebControl cleared
                    # the command for the remote's immediate stop request.
                    self.zeroing = bool(np.any(self.heading.output)) and not bool(state.buttons & (p.B | p.L2))
                    self.web.zero()
                if high.motion_blocked:
                    self.reason = high.reason
            # Camera failure can change the desired upper mode during command().
            upper_transitioning = (state.upper_mode != int(high.enabled) or state.upper_gain_blend < 1.)
        if (high is not None and not high.enabled and state.flags & 1
                and state.stage in (p.RAMPING, p.HOLDING, p.POLICY)):
            high_sampled = self.clock()
            high.warm(state, self.observations.models['homie'].previous_action, now_motion,
                      previous_state=getattr(self.backend, 'previous_state', None))
        velocity, heading = self.heading.update(
            state.quaternion, input_velocity, dt,
            active=self.mode != 'observe' and state.stage == p.POLICY and bool(state.flags & 1)
                   and (live_command or self.zeroing),
            omega=state.omega, sample_key=(state.session, state.sequence),
            received_time=state.received_ns*1e-9, control_time=now_motion,
            feedback_times=getattr(self.backend, 'feedback_times', ()),
            raw_command=raw_command,
            command_source=command_source,
            smooth_zero=self.zeroing)
        timing_reason = '航向反馈正在重新对齐；速度归零，站立策略继续'
        if heading['motion_blocked']:
            self.web.zero()
            self.commands.value.fill(0)
            raw_command.fill(0)
            live_command = False
            self.zeroing = heading['decelerating']
            self.reason = timing_reason
        elif self.reason == timing_reason:
            self.reason = ''
        if self.zeroing and not heading['decelerating']:
            self.zeroing = False
            self.record_event(type='smooth_zero_completed')
        previous_model = self.policy.selected
        self.policy.select(velocity, state)
        filtered_velocity = velocity.copy()
        if self.policy.selected == 'balance':
            velocity = np.zeros(3, dtype=np.float32)
        selection_reason = self.policy.selection_reason
        obs = actions = None
        inference_time = None
        inference_ms = latency_ms = None
        if state.flags & 1:
            sampled = high_sampled if high_sampled is not None else self.clock()
            obs = self.observations.push(state, filtered_velocity, self.cfg['height_m'], self.cfg['torso_rpy_command'])
            started = self.clock()
            # Keep the loaded pushcart estimator through a camera/grasp stop;
            # returning to ordinary walking explicitly restores original HIM.
            actions = self.policy(obs, filtered_velocity, state,
                                  pushcart=high is not None and (high.enabled or high.motion_blocked))
            finished = self.clock()
            inference_time = finished
            inference_ms = (finished - started) * 1000
            if high is not None and high.inferred:
                inference_ms += high.model.inference_ms
            latency_ms = (finished - sampled) * 1000
            self.policy_steps += 1
            self.results.append(finished)
            # Save outputs once beside their feedback, without full input histories.
        if self.policy.selected != previous_model:
            self.record_event(type='model_switch', previous_model=previous_model,
                              selected_model=self.policy.selected, reason=selection_reason,
                              velocity=velocity.copy(), max_leg_speed_rad_s=float(np.max(abs(state.dq[:12]))),
                              gyro_norm_rad_s=float(np.linalg.norm(state.omega)))
        executed_actions = actions
        high_stopping = self.zeroing or not np.any(raw_command) or not live_command or upper_transitioning
        if actions is not None and high is not None:
            executed_actions = high.compose(actions, self.policy.selected, velocity, stopping=high_stopping)
        q = (targets(executed_actions[self.policy.selected], state, self.cfg, self.config.contracts, self.policy.selected)
             if self.request == p.RUN and actions is not None else state.applied_q)
        if self.request == p.RUN and actions is not None and high is not None:
            q = high.upper_target(q, stopping=high_stopping)
        if self.request == p.PREPARE:
            q = self.prepare_q.copy()
        if self.stop_requested.is_set():
            self.damp(self.reason or '停止请求')
            return
        send_options = dict(hand_recover=self.hand_recovery, hand_grasp=self.config.hand_grasp.ravel(),
                            hand_live=self.hand_live)
        if high is not None:
            send_options['upper_mode'] = int(high.enabled)
            send_options['upper_target_active'] = bool(high.upper_target_selected)
        self.backend.send(self.request, q, p.PROFILE_NAMES.index(self.policy.selected), self.hand_pose,
                          **send_options)
        command_finished = self.clock()
        if self.request == p.RUN and actions is not None and self.mode != 'observe':
            self.observations.record_action(executed_actions, self.policy.selected)
            if high is not None:
                high.commit(velocity, executed_actions[self.policy.selected], q)
        self.ticks.append(self.clock())
        preparation_warning = self._preparation_warning(state)
        if bool(preparation_warning) != bool(self.latest.get('preparation_warning')):
            self.record_event(type='preparation_warning', active=bool(preparation_warning),
                              reason=preparation_warning or '准备姿态已到位')
        data = dict(stage=self.stage, reason=self.reason, healthy=bool(state.flags & 1),
                    model_set=self.model_set,
                    model_set_selectable=self.model_sets is not None and self._can_select_prepare_pose(state),
                    preparation_warning=preparation_warning,
                    prepare_pose=self.prepare_pose,
                    prepare_pose_selectable=self._can_select_prepare_pose(state),
                    source=self.web.source, owner=self.web.owner, velocity=velocity.copy(),
                    filtered_velocity=filtered_velocity,
                    input_velocity=input_velocity, heading=heading,
                    decelerating=self.zeroing,
                    dex3=self._hand_status(state),
                    joint_comparison=joint_snapshot(state, self.clock(), self.cfg['state_timeout_s']),
                    motion_revision=self.motion_revision,
                    target=state.applied_q.copy(), requested_target=q.copy(),
                    kp=state.applied_kp.copy(), kd=state.applied_kd.copy(),
                    selected_model=self.policy.selected, policy_mode=self.policy.policy_mode,
                    model_selection_reason=selection_reason,
                    pd_profile=p.PROFILE_NAMES[state.profile], pd_transition_blend=state.gain_blend,
                    upper_pd_mode=p.UPPER_MODE_NAMES[state.upper_mode],
                    upper_pd_requested='pushcart' if high is not None and high.enabled else 'hold',
                    upper_pd_transition_blend=state.upper_gain_blend,
                    upper_pd_transitioning=upper_transitioning,
                    policy_steps=self.policy_steps, observation_frames=self.observations.frames,
                    model_warmup=dict(balance_homie=self.policy.warmed,
                                      balance_homie_ms=self.policy.warmup_ms,
                                      pushcart=high.model.warmed if high is not None else False),
                    model_actions={name: value.copy() for name, value in actions.items()} if actions is not None else None,
                    high_policy=high.snapshot(self.clock()) if high is not None else None,
                    navigation_pose_result=dict(self.navigation_pose_result),
                    homie_estimator={name: value.copy() for name, value in self.policy.homie_estimator.items()}
                                    if actions is not None else None,
                    homie_estimator_source=self.policy.homie_estimator_source if actions is not None else None,
                    policy_inference_monotonic=inference_time,
                    simulation=bool(state.flags & 4), motor_command_written=False,
                    bridge_control_armed=self.backend.armed,
                    # Actual DDS writes belong to C++; its packet reports applied targets/gains.
                    bridge_motor_control_active=self.mode == 'control' and state.stage in (p.RAMPING, p.HOLDING, p.POLICY),
                    state=dict(q=state.q.copy(), dq=state.dq.copy(), quat=state.quaternion.copy(),
                               omega=state.omega.copy(), tick=(state.session, state.sequence),
                               received=state.received_ns * 1e-9, received_ns=state.received_ns,
                               session=state.session, buttons=state.buttons, sticks=state.sticks.copy()),
                    timing=dict(control_hz=self._rate(self.ticks, now),
                                policy_hz=self._rate(self.results, now), inference_ms=inference_ms,
                                high_policy_inference_ms=high.model.inference_ms if high is not None and high.inferred else None,
                                observation_to_action_ms=latency_ms,
                                state_age_ms=age * 1000,
                                state_to_command_ms=(command_finished - state.received_ns * 1e-9) * 1000
                                    if self.backend.armed and self.mode != 'observe' else None,
                                bridge_servo_hz=self.cfg['servo_hz']),
                    recording_error=self.recorder.error if self.recorder else '',
                    recording_dropped=self.recorder.dropped if self.recorder else 0)
        if high is not None:
            data['high_policy']['enable_reason'] = high.enable_reason(self, state, self.clock())
            data['high_policy']['zero_reason'] = high.zero_reason(self, state, self.clock())
        with self.lock:
            self.latest = data
        if self.recorder:
            self.recorder.record('telemetry', data)

    def start(self):
        self.backend.start()
        self.thread = threading.Thread(target=self._run, name='balance-homie-control', daemon=True)
        self.thread.start()

    def _run(self):
        deadline = self.clock()
        while not self.shutdown.is_set():
            try:
                self.step()
            except Exception as exc:
                self.record_event(type='exception', error=repr(exc))
                self.damp(str(exc), fault=True)
            deadline = max(deadline + 1/self.cfg['policy_hz'], self.clock())
            self.shutdown.wait(max(0., deadline - self.clock()))

    def close(self):
        self.stop_requested.set()
        self.shutdown.set()
        if self.thread:
            self.thread.join(timeout=5)
        self.record_event(type='shutdown', stage=self.stage, reason=self.reason or '程序退出')
        self.backend.close()
        if self.model_sets is not None:
            self.model_sets.close()
        elif self.high_policy is not None:
            self.high_policy.close()
        self.hand_file_executor.shutdown(wait=True)
