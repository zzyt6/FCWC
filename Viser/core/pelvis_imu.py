"""Independent pelvis feedback branch; never changes the RL observation arrays.

SDK wxyz is R_WB = Rz(yaw) Ry(pitch) Rx(roll). SI units throughout.
received_ns is a HOST steady-clock reception time, not a sensor acquisition time.
"""
import numpy as np


class ImuFault(ValueError):
    """Shared feedback is unusable: caller must use the existing safety stop."""


def wrap_angle(value):
    return float(np.arctan2(np.sin(value), np.cos(value)))


def attitude_wxyz(quaternion):
    q = np.array(quaternion, dtype=float, copy=True)
    if q.shape != (4,) or not np.isfinite(q).all():
        raise ImuFault('Invalid pelvis quaternion')
    norm = np.linalg.norm(q)
    if not .8 < norm < 1.2:
        raise ImuFault('Invalid pelvis quaternion norm')
    w, x, y, z = q / norm
    return (float(np.arctan2(2*(w*x+y*z), 1-2*(x*x+y*y))),
            float(np.arcsin(np.clip(2*(w*y-z*x), -1, 1))),
            float(np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z))))


def yaw_wxyz(quaternion):
    return attitude_wxyz(quaternion)[2]


class PelvisImu:
    def __init__(self, settings):
        self.settings = settings
        self.key = self.stamp = self.good_since = None
        self.previous_yaw = self.previous_rate = self.filtered = None
        self.last = {}

    def update(self, quaternion, omega, key, received_time, control_time):
        cfg = self.settings
        age = control_time - received_time
        if not np.isfinite([control_time, received_time]).all() or not 0 <= age <= cfg['imu_timeout_s']:
            raise ImuFault('Pelvis IMU stale or host clock invalid')
        if self.key is not None:
            if key[0] != self.key[0] or key[1] < self.key[1]:
                raise ImuFault('Pelvis IMU session changed or sample out of order')
            if key == self.key:
                if control_time-self.stamp > cfg['imu_timeout_s']:
                    raise ImuFault('Pelvis IMU cached sample expired')
                # A cached packet never advances filter/recovery time.
                return dict(self.last, imu_age=control_time-self.stamp, new_sample=False)
            if received_time <= self.stamp:
                raise ImuFault('Pelvis IMU receive time did not advance')
        roll, pitch, yaw = attitude_wxyz(quaternion)
        gyro = np.array(omega, dtype=float, copy=True)
        if gyro.shape != (3,) or not np.isfinite(gyro).all():
            raise ImuFault('Invalid pelvis gyroscope')
        dt = None if self.stamp is None else received_time-self.stamp
        if dt is not None and dt > cfg['imu_timeout_s']:
            raise ImuFault('Pelvis IMU sample gap exceeds timeout')
        p, q, r = gyro
        # Three different rates; do not use scaled policy observations here.
        world_z = -np.sin(pitch)*p + np.cos(pitch)*(np.sin(roll)*q+np.cos(roll)*r)
        rate = ((np.sin(roll)*q+np.cos(roll)*r)/np.cos(pitch)
                if abs(np.cos(pitch)) > .2 else None)
        reason = ''
        if max(abs(roll), abs(pitch)) > cfg['max_feedback_tilt_rad']:
            reason = 'tilt_outside_heading_approximation'
        elif dt is not None and cfg['gyro_cutoff_hz'] >= .5/dt:
            reason = 'sample_rate_below_filter_nyquist'
        elif dt is not None and self.previous_rate is not None:
            increment = wrap_angle(yaw-self.previous_yaw)
            predicted = .5*(rate+self.previous_rate)*dt
            if abs(wrap_angle(increment-predicted)) > cfg['yaw_jump_tolerance_rad']:
                reason = 'unexplained_yaw_jump'
        if reason or self.filtered is None or rate is None:
            self.filtered = None if reason else rate
            self.good_since = None
        else:
            alpha = np.exp(-2*np.pi*cfg['gyro_cutoff_hz']*dt)
            self.filtered = float(alpha*self.filtered + (1-alpha)*rate)
        if not reason:
            if self.good_since is None:
                self.good_since = received_time
            if received_time-self.good_since < cfg['imu_recovery_s']:
                reason = 'warming_up'
        self.key, self.stamp = key, received_time
        self.previous_yaw, self.previous_rate = yaw, rate
        self.last = dict(roll=roll, pitch=pitch, yaw_raw=yaw, yaw_used=yaw,
                         raw_gyro=gyro.tolist(), body_z_rate=float(r), world_z_rate=float(world_z),
                         yaw_rate_for_heading=None if rate is None else float(rate),
                         filtered_yaw_rate_for_heading=self.filtered,
                         measured_policy_yaw_rate=None, imu_sample_time=None,
                         imu_receive_time=received_time, imu_time_basis='host_receive_proxy',
                         imu_age=age, imu_dt=dt, new_sample=True,
                         consumed_sample_hz=None if dt is None else 1/dt,
                         imu_health=reason or 'healthy',
                         calibration_status='not_enabled_no_acceleration_stationarity_evidence')
        return self.last.copy()
