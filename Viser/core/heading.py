"""Pelvis-only, body-command heading feedback, applied before command scaling.

Kp [1/s], Ki [1/s^2], Kd [dimensionless]; integral output and correction [rad/s].
No XY transform, no changes to the policy's original IMU preprocessing.
"""
import json
from pathlib import Path
import numpy as np
from core.pelvis_imu import PelvisImu, ImuFault, yaw_wxyz, wrap_angle

CONFIG_PATH = Path(__file__).resolve().parents[1] / 'config/motion_config.json'
DEFAULTS = dict(command_frame='body', run_mode='disabled', heading_kp=1.,
                heading_ki=0., heading_kd=0., max_correction_rad_s=.15,
                integral_limit_rad_s=.04, heading_deadband_rad=np.deg2rad(.5),
                gyro_cutoff_hz=3., imu_timeout_s=.1, imu_recovery_s=.25,
                yaw_jump_tolerance_rad=.12, max_feedback_tilt_rad=.35,
                translation_enter_m_s=.025, translation_exit_m_s=.015,
                turn_enter_rad_s=.035, turn_exit_rad_s=.015,
                turn_release_s=.12, settle_rate_rad_s=.08, settle_max_s=.6,
                residual_stop_rad_s=.8, engage_ramp_s=.3, rate_control_enabled=False,
                navigation_max_heading_error_rad=.2,
                pushcart_max_heading_error_rad=.2)  # Accepted for old configs; PUSHCART bypasses this loop.


def validate_settings(settings):
    if set(settings)-set(DEFAULTS):
        raise ValueError('Unknown heading settings (legacy controller configs must be migrated)')
    result = dict(DEFAULTS, **settings)
    if result['command_frame'] != 'body':
        raise ValueError('速度指令必须使用机身坐标 body')
    if result['run_mode'] not in ('disabled', 'shadow', 'active'):
        raise ValueError('run_mode must be disabled, shadow or active')
    if result['rate_control_enabled'] is not False:
        raise ValueError('Rate tracking unavailable: exported policy wz training definition is unverified')
    bounds = dict(heading_kp=(0,2), heading_ki=(0,.5), heading_kd=(0,.5),
                  max_correction_rad_s=(0,.3), integral_limit_rad_s=(0,.1),
                  heading_deadband_rad=(0,.1), gyro_cutoff_hz=(.1,10),
                  imu_timeout_s=(.02,.1), imu_recovery_s=(.1,2),
                  yaw_jump_tolerance_rad=(.02,.3), max_feedback_tilt_rad=(.05,.35),
                  translation_enter_m_s=(.001,.1), translation_exit_m_s=(0,.1),
                  turn_enter_rad_s=(.001,.1), turn_exit_rad_s=(0,.1),
                  turn_release_s=(.02,.5), settle_rate_rad_s=(.01,.3),
                  settle_max_s=(.1,2), residual_stop_rad_s=(.3,1.5), engage_ramp_s=(.05,2),
                  navigation_max_heading_error_rad=(.02,.5), pushcart_max_heading_error_rad=(.02,.5))
    for key,(lo,hi) in bounds.items():
        value = float(result[key])
        if not np.isfinite(value) or not lo <= value <= hi:
            raise ValueError('%s must be within [%g, %g]' % (key,lo,hi))
        result[key] = value
    for stem,unit in [('translation','m_s'), ('turn','rad_s')]:
        if result[f'{stem}_exit_{unit}'] >= result[f'{stem}_enter_{unit}']:
            raise ValueError(stem+' exit threshold must be lower than enter')
    if result['turn_release_s'] > result['settle_max_s']:
        raise ValueError('Turn release debounce exceeds maximum wait')
    if result['settle_rate_rad_s'] >= result['residual_stop_rad_s']:
        raise ValueError('Residual stop threshold must exceed settling threshold')
    return result


def load_settings(path=CONFIG_PATH, command_frame=None, heading_kp=None, run_mode=None):
    settings = json.loads(Path(path).read_text())
    for key,value in [('command_frame',command_frame), ('heading_kp',heading_kp), ('run_mode',run_mode)]:
        if value is not None: settings[key] = value
    return validate_settings(settings)


def heading_to_policy_wz(request, roll, pitch, tilt_limit):
    """Small-tilt approximation only; no unverified body/world-rate conversion."""
    if max(abs(roll),abs(pitch)) > tilt_limit:
        raise ValueError('Outside small-tilt heading adapter range')
    return float(request)


def rate_execution_target(target, measured=None, enabled=False):
    """Reserved pass-through: preserve feedforward until training semantics verified."""
    if enabled:
        raise ValueError('Policy yaw-rate training definition is unverified')
    return float(target)


class HeadingControl:
    def __init__(self, settings, limits, slew, *, command_filter_tau_s=.4, model_switch_command_norm=.05):
        self.settings = validate_settings(settings)
        self.limits = np.asarray(limits, dtype=float)
        self.slew = np.asarray(slew, dtype=float)
        self.filter_tau = command_filter_tau_s
        self.switch_norm = model_switch_command_norm
        self.clear()

    def clear(self):
        self.imu = PelvisImu(self.settings)
        self.output = np.zeros(3)
        self.suggested_wz = 0.
        self.last = {}
        self.origin = None
        self.source = None
        self.needs_neutral = False
        self.timing_blocked = False
        self.translating = self.turning = False
        self.release_since = self.settle_since = None
        self.state, self.reason = 'STOPPED', 'initial'
        self._release()

    def _release(self):
        self.target = None
        self.integral = self.correction = 0.
        self.applied_correction = 0.
        self.engaged_at = None
        self.reference_wz = 0.

    def rebase(self, quaternion):
        # Re-enable while stationary; actual reference is captured on movement.
        self.origin = yaw_wxyz(quaternion)
        self._release()
        self.translating = self.turning = False
        self.release_since = self.settle_since = None
        self.needs_neutral = False
        self.output.fill(0)
        self.suggested_wz = 0.

    def configure(self, settings, quaternion):
        updated = validate_settings(settings)
        if np.any(self.output):
            raise ValueError('先速度归零，再修改航向参数')
        self.settings = updated
        self.clear()  # Shadow/old integral and filters cannot migrate into active.
        self.rebase(quaternion)

    def stop(self, reason):
        self._release()
        self.output.fill(0)
        self.suggested_wz = 0.
        self.state, self.reason = 'STOPPED', reason
        if self.last:
            self.last.update(heading_state=self.state, transition_reason=reason, yaw_ref=None,
                             target_yaw_rad=None, target_relative_rad=None,
                             error_rad=None, heading_error=None, active=False,
                             correction_enabled=False, decelerating=False, heading_P=None,
                             heading_I=None, heading_D=None, integral_frozen=True,
                             correction_saturated=False, slew_limit_flag=False, slew_limited=False,
                             saturation_flags=dict(heading=False, final=False),
                             correction_rad_s=0., correction_raw_rad_s=0.,
                             wz_feedforward=0., navigation_tracking=False, pushcart_tracking=False,
                             wz_target=0., wz_before_limits=0., wz_final_applied=0.,
                             wz_suggested=0., body_velocity=[0.,0.,0.])

    def update(self, quaternion, velocity, dt, active, *, omega, sample_key,
               received_time, control_time, raw_command, command_source='UNKNOWN', smooth_zero=False,
               feedback_times=()):
        cfg = self.settings
        command = np.array(velocity, dtype=float, copy=True)
        user = np.asarray(raw_command, dtype=float)
        if (command.shape != (3,) or user.shape != (3,) or not np.isfinite(command).all()
                or not np.isfinite(user).all() or not np.isfinite(dt) or dt < 0):
            raise ValueError('Invalid physical velocity command or control interval')
        m = self.imu.update(quaternion, omega, sample_key, received_time, control_time,
                            feedback_times=feedback_times)
        # A fresh valid body sample can still run Balance after a missed
        # heading interval. Drop motion/reference continuity, not motor support.
        control_gap = active and dt > cfg['imu_timeout_s']
        if control_gap or m['imu_gap_detected']:
            self.timing_blocked = True
            self.needs_neutral = True
            self.imu.good_since = received_time
            self._release()
        yaw = m['yaw_used']
        if self.origin is None: self.origin = yaw
        mode = cfg['run_mode']
        previous_state = self.state
        previous_source = self.source
        self.source = command_source
        command = np.clip(command, self.limits[:,0], self.limits[:,1])
        neutral = (np.linalg.norm(user[:2]) <= cfg['translation_exit_m_s'] and
                   abs(user[2]) <= cfg['turn_exit_rad_s'])
        if m['imu_health'] not in ('healthy','warming_up'):
            self.needs_neutral = True
        if m['imu_health'] == 'healthy' and neutral and not control_gap:
            self.needs_neutral = False
            self.timing_blocked = False
        health_ok = m['imu_health'] == 'healthy' and not self.needs_neutral
        if command_source != previous_source:
            self._release()
            self.turning = self.translating = False
            self.release_since = self.settle_since = None
        moving_threshold = cfg['translation_exit_m_s'] if self.translating else cfg['translation_enter_m_s']
        self.translating = np.linalg.norm(user[:2]) > moving_threshold
        error = effective_error = P = D = raw_correction = correction = 0.
        hold = False
        if self.timing_blocked:
            self.state, self.reason = 'RECOVERING', 'timing_gap_standing'
            self.turning = False
        elif not active or smooth_zero:
            self.state, self.reason = 'STOPPED', 'inactive_or_zero'
            self.turning = False
        elif command_source == 'PUSHCART':
            self.state, self.reason = 'PUSHCART', 'cart_heading_before_student'
        elif mode == 'disabled':
            self.state, self.reason = 'DISABLED', 'feedback_disabled'
        elif not health_ok:
            self.state, self.reason = 'RECOVERING', m['imu_health'] if not self.needs_neutral else 'await_healthy_and_neutral'
        elif command_source == 'NAVIGATION':
            source = command_source.lower()
            self.state, self.reason = ('TRACKING', source + '_reference') if np.any(user) else ('STOPPED', source + '_zero')
        elif command_source != 'MANUAL':
            self.state, self.reason = 'EXTERNAL' if command_source == 'EXTERNAL' else 'UNKNOWN', 'external_heading_owner'
        elif abs(user[2]) > (cfg['turn_exit_rad_s'] if self.turning else cfg['turn_enter_rad_s']):
            self.state, self.reason = 'TURNING', 'user_turn'
            self.turning = True
            self.release_since = self.settle_since = None
        elif self.turning:
            if self.release_since is None: self.release_since = control_time
            rate = abs(m['filtered_yaw_rate_for_heading'])
            if rate <= cfg['settle_rate_rad_s']:
                if self.settle_since is None: self.settle_since = control_time
            else:
                self.settle_since = None
            settled = self.settle_since is not None and control_time-self.settle_since >= cfg['turn_release_s']
            waited = control_time-self.release_since >= cfg['settle_max_s']
            if waited and rate > cfg['residual_stop_rad_s']:
                if mode == 'active': raise ImuFault('Excessive residual rotation after releasing turn')
                self.state, self.reason = 'RECOVERING', 'shadow_residual_stop_request'
            elif settled or waited:
                self.turning = False
                self.state, self.reason = ('HOLD','new_heading_after_turn') if self.translating else ('STOPPED','turn_finished')
            else:
                self.state, self.reason = 'REACQUIRE', 'wait_turn_settle'
        elif self.translating:
            self.state, self.reason = 'HOLD', 'manual_translation'
        else:
            self.state, self.reason = 'STOPPED', 'neutral_intent'
        tracking = self.state == 'TRACKING'
        hold = self.state in ('HOLD', 'TRACKING')
        # Use the filtered command as feedforward and
        # reference motion. Raw target integration would accumulate error while
        # the robot is still accelerating to that target.
        feedforward = float(command[2])
        if hold:
            if self.target is None:
                self.target, self.engaged_at = yaw, control_time
            elif tracking:
                # Feedback at this tick follows the previous tick's command.
                self.target = wrap_angle(self.target + self.reference_wz*dt)
            if tracking:
                error_limit = cfg['navigation_max_heading_error_rad']
                self.target = wrap_angle(yaw + np.clip(wrap_angle(self.target-yaw), -error_limit, error_limit))
                self.reference_wz = feedforward
            error = wrap_angle(self.target-yaw)
            effective_error = float(np.sign(error)*max(0., abs(error)-cfg['heading_deadband_rad']))
            blend = min(1., (control_time-self.engaged_at)/cfg['engage_ramp_s'])
            P = cfg['heading_kp']*effective_error
            # A moving reference uses rate ERROR. Damping the measured rate
            # alone would oppose an otherwise correctly tracked turn.
            D = cfg['heading_kd']*((feedforward if tracking else 0.) - m['filtered_yaw_rate_for_heading'])
            raw_correction = heading_to_policy_wz(blend*(P+self.integral+D), m['roll'], m['pitch'], cfg['max_feedback_tilt_rad'])
            correction = float(np.clip(raw_correction,-cfg['max_correction_rad_s'],cfg['max_correction_rad_s']))
        else:
            self._release()
        target = rate_execution_target(feedforward+correction)
        bounded = float(np.clip(target,*self.limits[2]))
        # The commanded yaw rate already passed the training filter. Only the
        # added heading correction is slew-limited; do not filter yaw twice.
        step = self.slew[2]*min(dt,.04)
        self.applied_correction += float(np.clip(bounded-feedforward-self.applied_correction,-step,step))
        proposed = float(np.clip(feedforward+self.applied_correction,*self.limits[2]))
        decelerating = False
        if (smooth_zero or self.timing_blocked) and active:
            # Same dt/tau filter as the training switch bank. Suppress heading
            # corrections throughout the stop, then hand over at the threshold.
            self.output *= max(0., 1. - dt/self.filter_tau)
            if np.linalg.norm(self.output) < self.switch_norm:
                self.output.fill(0)
            decelerating = bool(np.any(self.output))
            proposed = float(self.output[2])
        elif not active:
            self.output.fill(0)
            proposed = 0.
            self.turning = False
        elif mode == 'active':
            self.output[:] = command
            self.output[2] = proposed
        else:
            self.output[:] = command
        self.suggested_wz = proposed if active else 0.
        applied = float(self.output[2])
        integral_used = self.integral
        integral_frozen = True
        # Conditional integration uses the actual final command INCLUDING slew.
        if hold and mode == 'active' and blend >= 1.:
            delta_i = cfg['heading_ki']*effective_error*dt
            blocked = ((raw_correction-correction)*delta_i > 1e-12 or
                       (target-applied)*delta_i > 1e-12)
            if not blocked:
                self.integral = float(np.clip(self.integral+delta_i,-cfg['integral_limit_rad_s'],cfg['integral_limit_rad_s']))
                integral_frozen = False
        self.correction = correction if mode == 'active' else 0.
        saturated = abs(raw_correction-correction) > 1e-10
        slew_limited = abs(proposed-bounded) > 1e-8
        self.last = dict(cfg, **m, control_time=control_time, control_dt=dt, dt_s=dt,
                         command_source=command_source, heading_state=self.state,
                         transition_reason=self.reason, state_changed=self.state != previous_state,
                         vx_user=float(user[0]), vy_user=float(user[1]), wz_user=float(user[2]),
                         wz_feedforward=feedforward,
                         navigation_tracking=tracking and command_source == 'NAVIGATION',
                         pushcart_tracking=tracking and command_source == 'PUSHCART',
                         yaw_ref=self.target, heading_error=error if hold else None,
                         heading_P=P if hold else None, heading_I=integral_used if hold else None,
                         heading_D=D if hold else None, integral_frozen=integral_frozen,
                         wz_target=target, rate_correction=None, wz_before_limits=target,
                         wz_final_applied=applied, wz_suggested=proposed,
                         saturation_flags=dict(heading=saturated, final=target != bounded),
                         slew_limit_flag=slew_limited, needs_neutral=self.needs_neutral,
                         effective_frame='body', active=bool(active and np.any(self.output)),
                         correction_enabled=hold and mode == 'active', decelerating=decelerating,
                         motion_blocked=self.timing_blocked,
                         imu_yaw_rad=yaw, heading_origin_yaw_rad=self.origin,
                         yaw_relative_rad=wrap_angle(yaw-self.origin), target_yaw_rad=self.target,
                         target_relative_rad=None if self.target is None else wrap_angle(self.target-self.origin),
                         error_rad=error if hold else None, correction_rad_s=self.correction,
                         correction_raw_rad_s=raw_correction, correction_saturated=saturated,
                         input_velocity=command.tolist(), body_velocity=self.output.tolist(),
                         slew_limited=slew_limited)
        return self.output.astype(np.float32), self.last.copy()
