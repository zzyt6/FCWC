"""V8.1/V8.2 student extension for the existing Viser Controller.

Viser owns Balance/Homie selection, executed history, gains and all motor I/O.
This module loads the student and its paired Homie estimator/actor; no DDS or LCM.
"""
from collections import deque
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

from .carmer_io import CarmerYaw
from .controller import MODE_PUSH, Sensors, V81Controller, load_config
from .homie_adapter import PushcartHomie

ACTIVE = np.array(list(range(13)) + list(range(15, 29)))
UPPER = ACTIVE[12:]


class CartHeadingHold:
    """Cart yaw feedback on the task command, before the training reference filter."""
    def __init__(self, kp=1.0, max_corr=0.15, max_err=0.3):
        self.kp, self.max_corr, self.max_err = kp, max_corr, max_err
        self.reset()

    def reset(self):
        self.ref, self.err, self.corr = None, 0.0, 0.0

    def __call__(self, task_cmd, imu_yaw, cam_rel_yaw, dt):
        vx, vy, wz = task_cmd
        if not all(math.isfinite(x) for x in (vx, vy, wz, imu_yaw, cam_rel_yaw, dt)) or dt < 0:
            raise ValueError('Invalid cart heading input or control interval')
        if abs(vx) < 1e-6 and abs(vy) < 1e-6 and abs(wz) < 1e-6:
            self.reset()
            return task_cmd
        cart_yaw = imu_yaw + cam_rel_yaw
        if self.ref is None:
            self.ref = cart_yaw
        self.ref += wz * dt
        error = math.atan2(math.sin(self.ref - cart_yaw), math.cos(self.ref - cart_yaw))
        if abs(error) > self.max_err:
            error = math.copysign(self.max_err, error)
            self.ref = cart_yaw + error
        self.err = error
        self.corr = max(-self.max_corr, min(self.max_corr, self.kp * error))
        return (vx, vy, max(-0.6, min(0.6, wz + self.corr)))


class PelvisYawHistory:
    """Pelvis IMU yaw by state receive time, to pair with a camera capture.

    The camera measures cart yaw relative to the pelvis at its capture time,
    40-60 ms before control and up to 0.3 s during a held frame. The pelvis
    sways with every step (2026-10-21: up to ~2 rad/s), so adding the CURRENT
    IMU yaw turned that sway into a false cart heading error of 0.05-0.24 rad
    and saturated the heading correction. The cart itself turns slowly, so
    the cart yaw at capture time is used as the cart yaw now.
    """
    def __init__(self, seconds=1.):
        self.samples = deque(maxlen=max(2, int(seconds*50)+1))

    def add(self, time_s, yaw):
        if self.samples and time_s <= self.samples[-1][0]:
            return
        if self.samples:  # Unwrap so interpolation never crosses +-pi the long way.
            yaw = self.samples[-1][1] + math.atan2(math.sin(yaw-self.samples[-1][1]),
                                                   math.cos(yaw-self.samples[-1][1]))
        self.samples.append((time_s, yaw))

    def at(self, time_s):
        """Yaw at time_s, clamped to the stored span; None without samples."""
        if not self.samples or not math.isfinite(time_s):
            return None
        if time_s <= self.samples[0][0]:
            yaw = self.samples[0][1]
        elif time_s >= self.samples[-1][0]:
            yaw = self.samples[-1][1]
        else:
            for (t0, y0), (t1, y1) in zip(self.samples, list(self.samples)[1:]):
                if t0 <= time_s <= t1:
                    yaw = y0 + (time_s-t0)/(t1-t0)*(y1-y0)
                    break
        return math.atan2(math.sin(yaw), math.cos(yaw))


class StudentModel:
    def __init__(self, directory, threads=1, *, model_version='v81'):
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        if model_version not in ('v81', 'v82'):
            raise ValueError('Unsupported pushcart model version')
        self.path = Path(directory) / f'{model_version}_student.onnx'
        self.session = ort.InferenceSession(str(self.path), options, providers=['CPUExecutionProvider'])
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        if ([(x.name, x.shape, x.type) for x in inputs] !=
                [('policy', [1, 630], 'tensor(float)'), ('history', [1, 4032], 'tensor(float)')]
                or [(x.name, x.shape, x.type) for x in outputs] !=
                [('actions', [1, 30], 'tensor(float)'), ('velocity_estimate', [1, 19], 'tensor(float)')]):
            raise ValueError('Pushcart student ONNX interface mismatch')
        self.raw = None
        self.inference_ms = None
        self.started_monotonic = self.finished_monotonic = None
        started = time.perf_counter()
        for _ in range(2):
            self.run_student(np.zeros(630, np.float32), np.zeros(4032, np.float32))
        self.warmup_ms = (time.perf_counter() - started) * 1000
        self.warmed = True
        self.raw = None
        self.inference_ms = None
        self.started_monotonic = self.finished_monotonic = None

    def run_student(self, policy, history):
        if not np.isfinite(policy).all() or not np.isfinite(history).all():
            raise ValueError('Non-finite pushcart observation')
        started = time.perf_counter()
        self.started_monotonic = time.monotonic()
        action, estimate = self.session.run(None, dict(policy=policy[None], history=history[None]))
        self.finished_monotonic = time.monotonic()
        self.inference_ms = (time.perf_counter() - started) * 1000
        if not np.isfinite(action).all() or not np.isfinite(estimate).all():
            raise ValueError('Non-finite pushcart output')
        self.raw = action[0].copy()
        return action[0], estimate[0]


class ViserPushPolicy:
    def __init__(self, robot_config, path, *, camera=None, model_version='v81'):
        path = Path(path).resolve()
        self.settings = json.loads(path.read_text())
        settings = self.settings
        if settings.get('schema') != 'pushcart_viser_v1':
            raise ValueError('Invalid Viser pushcart configuration')
        directory = (path.parent / settings['models_dir']).resolve()
        cfg = load_config(directory, model_version)
        self.robot_config, self.cfg = robot_config, cfg
        self._validate_contract()
        self.limits = np.asarray(settings['task_limits'], dtype=float)
        training = cfg['task_command_range']
        allowed = np.array([training['vx'], [-training['vy_max'], training['vy_max']],
                            [-training['wz_max'], training['wz_max']]])
        if (self.limits.shape != (3, 2) or not np.isfinite(self.limits).all()
                or np.any(self.limits[:, 0] >= 0) or np.any(self.limits[:, 1] <= 0)
                or np.any(self.limits[:, 0] < allowed[:, 0]) or np.any(self.limits[:, 1] > allowed[:, 1])):
            raise ValueError('Task command limits must be inside the training range and include zero')
        cam = settings['camera']
        if (not isinstance(cam.get('tag_ids'), list) or not cam['tag_ids']
                or any(type(tag) is not int or tag < 0 for tag in cam['tag_ids'])):
            raise ValueError('Configure the cart AprilTag IDs explicitly')
        if (type(cam['udp_port']) is not int or not 1 <= cam['udp_port'] <= 65535
                or not 0 < float(cam['timeout_s']) <= .3 or not np.isfinite(cam['yaw_offset_rad'])
                or type(settings['onnx_threads']) is not int or not 1 <= settings['onnx_threads'] <= 8):
            raise ValueError('Invalid camera or inference settings')
        self.model = StudentModel(directory, settings['onnx_threads'], model_version=model_version)
        self.homie_model = PushcartHomie(directory / f'homie_{model_version}.onnx', robot_config.policy_paths['homie'],
                                         settings['onnx_threads'])
        self.homie_model.metadata.update(
            estimator_checkpoint=cfg['provenance']['homie_velocity_estimator'],
            estimator_checkpoint_sha256=cfg['provenance']['homie_velocity_estimator_sha256'],
            velocity_frame='pelvis body frame, x/y/z', velocity_scale=cfg['obs_scales']['lin_vel'])
        self.core = V81Controller(cfg, self.model)
        self.hold = CartHeadingHold()
        self.pelvis_yaw = PelvisYawHistory()
        self.camera = camera if camera is not None else CarmerYaw(
            cam['udp_port'], cam['timeout_s'], cam['yaw_offset_rad'], cam['tag_ids'])
        self.camera.allowed_ids = frozenset(cam['tag_ids'])
        self.camera_process = None
        self.zero_calibration = dict(status='idle', samples=0, required=30,
                                     message='本次运行尚未标零；使用配置中的航向偏置')
        self._zero_samples = []
        self.metadata = dict(student_path=str(self.model.path),
                             student_sha256=hashlib.sha256(self.model.path.read_bytes()).hexdigest(),
                             model_release=cfg['release'],
                             export_config_path=str(directory / f'{model_version}_config.json'),
                             export_config_sha256=hashlib.sha256((directory / f'{model_version}_config.json').read_bytes()).hexdigest(),
                             provenance=cfg['provenance'],
                             low_level='Viser auto Balance/Homie; paired tuned HIM velocity while pushcart is enabled or paused',
                             homie=self.homie_model.metadata,
                             upper_torque='PD estimate from preceding 2..8 ms bridge sample when available, otherwise latest; not measured torque',
                             heading_feedback='PUSHCART: integrate raw clipped task wz with actual control dt; '
                                              'measure IMU yaw + calibrated camera relative yaw; '
                                              'P correction before the task-reference filter/student; '
                                              'no pelvis heading correction on the student Homie command',
                             cart_heading_hold=dict(kp=self.hold.kp, max_corr=self.hold.max_corr,
                                                    max_err=self.hold.max_err, task_wz_limit=.6),
                             task_frame='cart forward/left; no extra XY rotation',
                             history='126 x 5 and 126 x 32; previous selected/executed leg action',
                             preparation='valid camera/body frames collected before activation; shadow predictions are not committed as actions',
                             settings_path=str(path))
        self.metadata['telemetry'] = dict(
            raw_task_command='clipped cart vx/vy/wz before cart heading feedback and motion gating',
            task_command='cart vx/vy/wz after cart heading feedback, entering the task-reference filter; zero when motion is blocked',
            heading_hold='ref/err in rad, corr in rad/s; imu_yaw_at_camera_rad (pelvis IMU yaw interpolated at '
                         'the camera capture time; imu_yaw_rad is the current one) + camera.yaw_rad = cart_yaw_rad; '
                         'dt_s is actual monotonic control interval; reset on zero/stop',
            task_reference='filtered cart reference after this tick update; inference_inputs.previous_task_reference is before update',
            homie_command='high-level robot vx/vy/wz candidate before Viser zero interception and command filtering',
            velocity_chain='raw_task_command -> cart heading hold -> task_command -> task_reference -> student Homie command; '
                           'high_policy.homie_command -> input_velocity -> filtered_velocity -> velocity, without heading correction; '
                           'zero/stop uses Viser deceleration and Balance receives zero; '
                           'velocity is the selected policy command, not measured robot speed',
            raw_actions=dict(shape=[30], slices=dict(command=[0, 3], upper=[3, 18], legs=[18, 30])),
            gate='dimensionless task-reference norm / gate_speed, clamped to [0, 1]',
            leg_residual='12 candidate residuals after training mapping/filter, in Homie action units',
            applied_leg_residual='residuals selected for addition to Homie before action clipping and joint/PD limits; zero for shadow/stop/Balance',
            upper_delta_rad='15 candidate joint offsets from hold_target_rad after training rate/position limits',
            upper_target_selected='candidate upper target was merged into Python requested_target; does not confirm motor execution',
            upper_sdk_indices=UPPER.tolist(), hold_target_rad=self.core.hold.tolist(),
            inference_inputs='same prediction: feedback tick/receipt time, used camera timestamp/yaw, '
                             'previous commands and upper PD estimate before torque_obs_scale',
            upper_torque_pd_estimate='15 PD torque estimates after training torque limits; not measured torque',
            torque_obs_scale=self.core.tau_scale,
            alignment='student ONNX start/finish use host monotonic seconds; input_state_tick matches '
                      'outer state.tick; inference fields are null without a new prediction')
        self.reset()

    def _validate_contract(self):
        c, rc = self.cfg, self.robot_config
        if (c['num_lower'], c['homie_frame'], c['homie_history'], c['frame_dim'],
                c['policy_history'], c['long_history'], c['action_dim'], c['hl_actions']) != (12, 76, 6, 126, 5, 32, 30, 18):
            raise ValueError('Unsupported pushcart observation dimensions')
        if not np.isclose(c['dt'], 1 / rc.deployment['policy_hz']):
            raise ValueError('Pushcart must run in the same 50 Hz Viser tick')
        if c['joints'] != [rc.deployment['joint_names_sdk29'][i] for i in ACTIVE]:
            raise ValueError('Pushcart joint order mismatch')
        for name in ('balance', 'homie'):
            spec = rc.contracts[name]
            pairs = [(c['default_dof_pos'], spec['default_joint_positions']),
                     (c['commands_scale'], spec['input']['command_scale'][:3]),
                     (c['height_command'], rc.deployment['height_m']),
                     (c['clip_obs'], spec['input']['clip_observations']),
                     (c['action_scale'], spec['output']['action_scale']),
                     (c['clip_actions'], spec['output']['training_clip_actions'])]
            pairs += [(c['obs_scales'][k], spec['input']['observation_scales'][k])
                      for k in ('ang_vel', 'dof_pos', 'dof_vel')]
            if not all(np.allclose(a, b, rtol=0, atol=1e-7) for a, b in pairs):
                raise ValueError('Pushcart and Viser observation/action units disagree: ' + name)

    def reset(self, reason='', *, blocked=False, preserve_history=False):
        self.cancel_zero_calibration('控制状态改变，标定已取消；保留原零位')
        self.enabled = False
        self.motion_blocked = blocked
        self.camera_timeout = False
        self.reason = reason
        self.output = None
        self.inferred = False
        self.shadow = False
        self.warm_error = ''
        self.torque_sample_age_ms = None
        self.inference_inputs = None
        self.hold.reset()
        self.heading_hold = None
        self.raw_task_command = None
        self.task_command = None
        self.upper_target_selected = False
        self.applied_residual = np.zeros(12)
        if not preserve_history:
            self.last_command = np.zeros(3)
            self.core.reset_homie()
            self.history_frames = 0
            self.history_time = -float('inf')
            self.history_key = None
        self.core.reset_task(preserve_history=preserve_history)
        self.core.cmd_homie.fill(0)

    def begin_tick(self, now):
        self.inferred = False
        self.shadow = False
        self.output = None
        self.upper_target_selected = False
        self.camera.poll(now)
        if self.camera_timeout and self.motion_blocked:
            self.reason = ('相机航向已恢复；推车仍暂停，请归零后重新启用' if self.camera.fresh(now)
                           else self.camera.snapshot(now)['reason'] + '；推车已暂停，站立策略继续')

    def history_ready(self, now):
        return (not self.warm_error and self.history_frames >= self.core.lh and self.camera.fresh(now)
                and 0 <= now - self.history_time <= self.robot_config.deployment['state_timeout_s'])

    def _forget_history(self):
        self.core.frames.fill(0)
        self.core.frames_long.fill(0)
        self.core.homie_hist.fill(0)
        self.core.fresh = True
        self.history_frames = 0
        self.history_time = -float('inf')
        self.history_key = None

    def set_navigation_pose_source(self, source, now):
        if self.enabled or self.motion_blocked:
            raise ValueError('请先退出推车策略再切换位姿方式')
        self.camera.set_pose_source(source, now)
        self.cancel_zero_calibration('位姿方式已切换')
        self._forget_history()
        self.core.reset_task(preserve_history=True)
        self.hold.reset()

    def zero_reason(self, control, state, now):
        if getattr(self.camera, 'pose_source', 'grasp') == 'apriltag':
            return '视觉位姿使用标签安装坐标，无需手动航向归零'
        if state is None or control.stage != 'POLICY' or not state.flags & 1:
            return '先完成准备并启动混合策略站立'
        if self.enabled or self.motion_blocked:
            return '先归零并返回普通行走，再标定航向'
        if control.web.source != 'web' or control.web.owner is None:
            return '先接管网页操作'
        if np.any(control.heading.output) or np.any(control.web.target) or control.zeroing:
            return '先速度归零并等待站稳'
        if not self.camera.fresh(now):
            return '等待相机识别到有效的小车航向'
        return ''

    def start_zero_calibration(self, control, state, now):
        if self.zero_calibration['status'] == 'collecting':
            raise ValueError('正在采集航向，请保持小车与骨盆对正')
        reason = self.zero_reason(control, state, now)
        if reason:
            raise ValueError(reason)
        self._zero_started, self._zero_stamp = now, now
        self._zero_owner = control.web.owner
        self._zero_samples = []
        self.zero_calibration = dict(status='collecting', samples=0, required=30,
                                     message='正在采集，请保持小车与骨盆朝向对正并静止')

    def cancel_zero_calibration(self, reason):
        if self.zero_calibration['status'] == 'collecting':
            self.zero_calibration.update(status='failed', message=reason)
            self._zero_samples = []

    def update_zero_calibration(self, control, state, now):
        """Sample distinct, post-click camera frames; never wait in the control loop."""
        if self.zero_calibration['status'] != 'collecting':
            return None
        reason = self.zero_reason(control, state, now)
        if control.web.owner != self._zero_owner:
            reason = '网页持有者已改变'
        if control.stop_requested.is_set():
            reason = '收到停止请求'
        if reason:
            self.cancel_zero_calibration(reason + '；标定取消，保留原零位')
        elif now - self._zero_started > 3.:
            self.cancel_zero_calibration('有效新帧不足，标定超时；保留原零位，请重试')
        else:
            camera = self.camera
            # A repeated/held observation is not an additional sample. An
            # invalid latest packet cannot contribute the previous valid yaw.
            if camera.stamp <= self._zero_stamp or camera.stamp != camera.last_packet_stamp:
                return None
            self._zero_stamp = camera.stamp
            self._zero_samples.append(camera.raw_yaw)
            self.zero_calibration['samples'] = len(self._zero_samples)
            if len(self._zero_samples) < self.zero_calibration['required']:
                return None
            angles = np.asarray(self._zero_samples)
            offset = float(np.arctan2(np.sin(angles).mean(), np.cos(angles).mean()))
            errors = np.arctan2(np.sin(angles - offset), np.cos(angles - offset))
            spread = float(np.max(np.abs(errors)))
            if spread > np.deg2rad(3.):
                self.cancel_zero_calibration('航向波动超过 3°，请对正并保持静止后重试；保留原零位')
            else:
                # Always use raw yaw, so a second click does not subtract the
                # previous zero twice. Camera freshness and IMU yaw are intact.
                camera.set_offset(offset)
                self.zero_calibration.update(status='done', offset_rad=offset,
                                             max_deviation_rad=spread,
                                             message='标定完成，本次运行生效；请查看归零后的航向')
                self._zero_samples = []
                self._forget_history()
                self.core.reset_task(preserve_history=True)
                self.warm_error = ''
        return self.zero_calibration.copy()

    def enable_reason(self, control, state, now):
        if self.zero_calibration['status'] == 'collecting':
            return '航向零位正在标定，请保持对正并静止'
        if state is None or control.stage != 'POLICY':
            return '先完成准备并启动混合策略'
        if np.any(control.heading.output) or np.any(control.web.target) or control.zeroing:
            return '先速度归零并等待 Balance 接管'
        if control.policy.policy_mode != 'auto':
            return '推车需要自动 Balance/Homie 模式'
        if not self.enabled and (state.upper_mode != 0 or state.upper_gain_blend < 1.):
            return '等待上肢增益切换完成'
        if control.prepare_pose != 'original':
            return '推车需要姿态 1；普通混合策略仍可使用其他姿态'
        if not np.allclose(self.cfg['hold_target_q'], control.prepare_q[UPPER], atol=1e-5, rtol=0):
            return '上半身准备姿态与推车训练姿态不同；普通行走不受影响'
        hands = control._hand_status(state)
        if (not hands['healthy'] or not hands['active'] or hands['moving'] or hands['pending']
                or hands['pose'] not in ('grasp', 'grasp_compensated') or control.hand_file is not None):
            return '摆好小车，完成抓握和角度调试后再启用'
        target = control.config.hand_grasp_compensated if control.hand_compensation else control.config.hand_grasp
        if control.hand_live and not np.allclose(state.hand_target, target.ravel(), atol=.002, rtol=0):
            return '等待最后一次抓握角度调节执行完成'
        if not self.camera.fresh(now):
            return '等待相机识别到有效的小车航向'
        if self.warm_error:
            return '推车历史预热异常：' + self.warm_error
        if not self.history_ready(now):
            return '推车历史准备中：%d / %d 帧' % (self.history_frames, self.core.lh)
        return ''

    def set_enabled(self, enabled, control, state, now):
        if enabled == self.enabled and not self.motion_blocked:
            return False
        if np.any(control.heading.output) or np.any(control.web.target) or control.zeroing:
            raise ValueError('先速度归零并等待 Balance 接管，再切换推车策略')
        if not enabled:
            self.reset('普通行走；可继续调整抓握', preserve_history=True)
            return True
        reason = self.enable_reason(control, state, now)
        if reason:
            raise ValueError(reason)
        self.reset(preserve_history=True)
        self.core.upper_delta = state.applied_q[UPPER].copy() - self.core.hold
        self.enabled = True
        return True

    def toggle(self, control, state, now):
        self.set_enabled(not self.enabled, control, state, now)

    def warm(self, state, previous_action, now, *, previous_state=None):
        """Populate real sensor history and run a shadow prediction only."""
        if not self.camera.fresh(now):
            self._forget_history()
            return
        if (state.session, state.received_ns) == self.history_key:
            return
        try:
            self._predict(state, np.zeros(3), previous_action, now, moving=False, previous_state=previous_state)
        except Exception as exc:
            # Optional shadow inference must not take down ordinary locomotion.
            # Independent body-health and low-level inference checks still run.
            self.warm_error = str(exc)
            self.inferred, self.output = False, None
            self._forget_history()
            self.core.reset_task(preserve_history=True)
            self.core.cmd_homie.fill(0)
            return
        self.warm_error = ''
        self.shadow = True
        # The shadow action was not executed. The next frame must still contain
        # zero high-level action/residual and the actual selected low-level action.
        self.core.reset_task(preserve_history=True)
        self.core.cmd_homie.fill(0)
        self.core.upper_delta = state.applied_q[UPPER].copy() - self.core.hold

    def command(self, state, task, previous_action, now, *, moving, dt, previous_state=None):
        if not self.camera.fresh(now):
            self.reset(self.camera.snapshot(now)['reason'] + '；推车已暂停，站立策略继续', blocked=True)
            self.camera_timeout = True
            return np.zeros(3), True
        self._predict(state, task, previous_action, now, moving=moving, dt=dt, previous_state=previous_state)
        return self.output.cmd_homie.copy() if moving else np.zeros(3), False

    def _predict(self, state, task, previous_action, now, *, moving, dt=0., previous_state=None):
        if now - self.history_time > self.robot_config.deployment['state_timeout_s']:
            self._forget_history()
        # Use the preceding 200 Hz bridge sample when it was received. Record
        # fallback explicitly: UDP gaps cannot be treated as a precise 5 ms delay.
        torque_state = state
        if previous_state is not None and previous_state.session == state.session:
            age = (state.received_ns - previous_state.received_ns) * 1e-9
            if .002 <= age <= .008:
                torque_state = previous_state
        self.torque_sample_age_ms = (state.received_ns - torque_state.received_ns) * 1e-6
        tau = np.clip((torque_state.applied_kp * (torque_state.applied_q - torque_state.q)
                       - torque_state.applied_kd * torque_state.dq)[ACTIVE],
                      -np.asarray(self.cfg['torque_limits']), np.asarray(self.cfg['torque_limits']))
        w, x, y, z = state.quaternion / np.linalg.norm(state.quaternion)
        gravity = np.array([2*(w*y-x*z), -2*(y*z+w*x), -(1-2*x*x-2*y*y)])
        sensors = Sensors(state.q[ACTIVE], state.dq[ACTIVE], state.omega, gravity)
        frame = self.core.homie_frame(sensors)
        frame[:3] = self.last_command * self.core.cmd_scale
        frame[-12:] = previous_action
        # Even a held zero command must use Viser's stop path, never a residual
        # generated from a previously nonzero task reference.
        self.raw_task_command = np.clip(task, self.limits[:, 0], self.limits[:, 1])
        command = self.raw_task_command if moving else np.zeros(3)
        # Same normalized SDK wxyz yaw as Viser heading.imu_yaw_rad. Camera yaw
        # is calibrated and held between frames; command() checks its timeout.
        imu_yaw = math.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
        self.pelvis_yaw.add(state.received_ns*1e-9, imu_yaw)
        # Cart yaw = pelvis yaw + relative yaw, both at the camera capture time.
        captured_yaw = self.pelvis_yaw.at(self.camera.stamp)
        hold_yaw = imu_yaw if captured_yaw is None else captured_yaw
        command = np.asarray(self.hold(command, hold_yaw, self.camera.yaw, dt))
        self.heading_hold = dict(ref=self.hold.ref, err=self.hold.err, corr=self.hold.corr, kp=self.hold.kp,
                                 dt_s=dt, imu_yaw_rad=imu_yaw, imu_yaw_at_camera_rad=hold_yaw,
                                 cart_yaw_rad=hold_yaw + self.camera.yaw)
        inputs = dict(input_state_tick=(state.session, state.sequence),
                      input_state_received_ns=state.received_ns,
                      camera_captured_monotonic=self.camera.stamp, camera_yaw_rad=self.camera.yaw,
                      previous_homie_command=self.last_command.copy(),
                      previous_task_reference=self.core.cmd_ref.copy(),
                      upper_torque_pd_estimate=tau[12:].copy(),
                      torque_state_tick=(torque_state.session, torque_state.sequence),
                      torque_state_received_ns=torque_state.received_ns)
        self.output = self.core.step_high(sensors, command, self.camera.yaw, MODE_PUSH,
                                         upper_tau=tau[12:], observed_homie_frame=frame)
        self.inference_inputs = inputs
        self.task_command = command.copy()
        self.inferred = True
        key = (state.session, state.received_ns)
        if key != self.history_key:
            self.history_frames = min(self.core.lh, self.history_frames + 1)
        self.history_key, self.history_time = key, now

    def compose(self, actions, selected, velocity, *, stopping=False):
        executed = {k: v.copy() for k, v in actions.items()}
        self.applied_residual.fill(0)
        if self.enabled and self.output is not None and selected == 'homie' and not stopping:
            self.applied_residual[:] = self.output.leg_residual
            executed[selected] += self.applied_residual
        # Balance always sees its own unmodified action.
        return executed

    def upper_target(self, q, *, stopping=False):
        if self.enabled and self.output is not None and not stopping:
            q[UPPER] = self.output.q_upper
            self.upper_target_selected = True
        return np.clip(q, self.robot_config.deployment['joint_lower_sdk29'],
                       self.robot_config.deployment['joint_upper_sdk29']).astype(np.float32)

    def commit(self, velocity, executed_action, q):
        self.last_command = np.asarray(velocity).copy()
        self.core.applied = np.clip(executed_action, -self.core.clip_actions, self.core.clip_actions)
        self.core.leg_residual[:] = self.applied_residual
        self.core.upper_delta = q[UPPER].copy() - self.core.hold

    def snapshot(self, now):
        return dict(enabled=self.enabled, motion_blocked=self.motion_blocked,
                    model_release=self.cfg['release'],
                    reason=self.reason, camera=self.camera.snapshot(now),
                    zero_calibration=self.zero_calibration.copy(),
                    model_warmed=self.model.warmed, startup_warmup_ms=self.model.warmup_ms,
                    homie_model_warmed=self.homie_model.warmed, homie_warmup_ms=self.homie_model.warmup_ms,
                    history_frames=self.history_frames, history_required=self.core.lh,
                    history_ready=self.history_ready(now),
                    warmup_error=self.warm_error,
                    inference_role=('shadow' if self.shadow else 'active') if self.inferred else None,
                    camera_process=self.camera_process.snapshot() if self.camera_process else {},
                    task_limits=self.limits.copy(),
                    raw_actions=self.model.raw.copy() if self.inferred else None,
                    estimator=self.output.est.copy() if self.inferred else None,
                    inference_ms=self.model.inference_ms if self.inferred else None,
                    inference_started_monotonic=self.model.started_monotonic if self.inferred else None,
                    inference_finished_monotonic=self.model.finished_monotonic if self.inferred else None,
                    inference_inputs={k: v.copy() if isinstance(v, np.ndarray) else v
                                      for k, v in self.inference_inputs.items()} if self.inferred else None,
                    torque_sample_age_ms=self.torque_sample_age_ms if self.inferred else None,
                    raw_task_command=self.raw_task_command.copy() if self.inferred else None,
                    heading_hold=self.heading_hold.copy() if self.inferred else None,
                    task_command=self.task_command.copy() if self.inferred else None,
                    task_reference=self.output.cmd_ref.copy() if self.inferred else None,
                    homie_command=self.output.cmd_homie.copy() if self.inferred else None,
                    gate=self.output.gate if self.inferred else None,
                    leg_residual=self.output.leg_residual.copy() if self.inferred else None,
                    upper_delta_rad=self.output.q_upper - self.core.hold if self.inferred else None,
                    upper_target_selected=self.upper_target_selected if self.inferred else None,
                    applied_leg_residual=self.applied_residual.copy() if self.inferred else None)

    def close(self):
        if self.camera_process:
            self.camera_process.close()
        self.camera.close()
