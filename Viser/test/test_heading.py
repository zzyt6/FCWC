"""Deterministic pelvis feedback tests; no bridge/network/hardware imports."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import unittest
import numpy as np
from core.heading import HeadingControl, DEFAULTS, validate_settings, heading_to_policy_wz, rate_execution_target
from core.pelvis_imu import PelvisImu, ImuFault, wrap_angle, attitude_wxyz


def quaternion(yaw=0., roll=0., pitch=0.):
    cr,sr=np.cos(roll/2),np.sin(roll/2)
    cp,sp=np.cos(pitch/2),np.sin(pitch/2)
    cy,sy=np.cos(yaw/2),np.sin(yaw/2)
    return np.array([cr*cp*cy+sr*sp*sy, sr*cp*cy-cr*sp*sy,
                     cr*sp*cy+sr*cp*sy, cr*cp*sy-sr*sp*cy])


class Rig:
    def __init__(self, **settings):
        self.control=HeadingControl(dict(DEFAULTS,run_mode='active',**settings),
                                    [[-.8,1.2],[-.5,.5],[-.8,.8]],[.3,.2,.4])
        self.t=10.; self.seq=0
        for _ in range(20): self.tick()

    def tick(self, command=(0,0,0), yaw=0., omega=(0,0,0), dt=.02, source='MANUAL', **kwargs):
        self.t+=dt; self.seq+=1
        return self.control.update(quaternion(yaw), command, dt, True, omega=omega,
                                   sample_key=(1,self.seq), received_time=self.t,
                                   control_time=self.t, raw_command=kwargs.pop('raw_command', command),
                                   command_source=source, **kwargs)


class ImuTests(unittest.TestCase):
    def make(self): return PelvisImu(validate_settings(DEFAULTS))

    def test_attitude_sign_order_wrap_and_equivalent_quaternions(self):
        for yaw in [-3.13,-1,0,1,3.13]:
            for scale in [1.,-1.,1.01]:
                np.testing.assert_allclose(attitude_wxyz(scale*quaternion(yaw,.2,-.25)), [.2,-.25,yaw], atol=1e-12)
        self.assertAlmostEqual(np.rad2deg(wrap_angle(np.deg2rad(-179-179))),2)
        self.assertAlmostEqual(np.rad2deg(wrap_angle(np.deg2rad(179+179))),-2)

    def test_feedback_normalization_and_filter_do_not_mutate_raw_arrays(self):
        q=1.01*quaternion(.3,.1,.2);omega=np.array([.1,.2,.3])
        original=q.copy();gyro=omega.copy()
        self.make().update(q,omega,(1,1),1.,1.)
        np.testing.assert_array_equal(q,original)
        np.testing.assert_array_equal(omega,gyro)

    def test_distinct_rates_tilt_and_adapter(self):
        m=self.make().update(quaternion(.7,.2,.3), [.4,.5,.6], (1,1),1.,1.)
        expected=(np.sin(.2)*.5+np.cos(.2)*.6)/np.cos(.3)
        self.assertAlmostEqual(m['yaw_rate_for_heading'],expected)
        self.assertNotAlmostEqual(expected,m['body_z_rate'])
        self.assertNotAlmostEqual(expected,m['world_z_rate'])
        self.assertIsNone(m['measured_policy_yaw_rate'])
        self.assertEqual(heading_to_policy_wz(.1,.2,.3,.35),.1)
        with self.assertRaises(ValueError): heading_to_policy_wz(.1,.4,0,.35)
        self.assertEqual(rate_execution_target(.2),.2)
        with self.assertRaises(ValueError): rate_execution_target(.2,enabled=True)

    def test_filter_first_variable_dt_and_duplicate(self):
        imu=self.make()
        m=imu.update(quaternion(),[0,0,.4],(1,1),1.,1.)
        self.assertEqual(m['filtered_yaw_rate_for_heading'],.4)
        for seq,dt in [(2,.02),(3,.035),(4,.015)]:
            previous=m['filtered_yaw_rate_for_heading']; t=imu.stamp+dt
            m=imu.update(quaternion(),[0,0,.2],(1,seq),t,t)
            expected=np.exp(-2*np.pi*3*dt)*previous+(1-np.exp(-2*np.pi*3*dt))*.2
            self.assertAlmostEqual(m['filtered_yaw_rate_for_heading'],expected)
            duplicate=imu.update(quaternion(),[0,0,99],(1,seq),t,t+.005)
            self.assertEqual(duplicate['filtered_yaw_rate_for_heading'],m['filtered_yaw_rate_for_heading'])
            self.assertFalse(duplicate['new_sample'])
        self.assertIsNone(m['imu_sample_time'])
        self.assertAlmostEqual(m['imu_dt'],.015)

    def test_bad_values_and_shared_timing_faults(self):
        for q in [[0]*4,[np.nan,0,0,0],[np.inf,0,0,0],[2,0,0,0]]:
            with self.assertRaises(ImuFault): self.make().update(q,[0]*3,(1,1),1.,1.)
        with self.assertRaises(ImuFault): self.make().update(quaternion(),[0,0,np.nan],(1,1),1.,1.)
        for key,t,now in [((1,0),1.02,1.02),((1,2),.99,1.),((2,2),1.02,1.02),
                          ((1,2),1.2,1.2),((1,1),1.,1.11),((1,2),1.02,1.01)]:
            imu=self.make(); imu.update(quaternion(),[0]*3,(1,1),1.,1.)
            with self.assertRaises(ImuFault): imu.update(quaternion(),[0]*3,key,t,now)

    def test_q_sign_wrap_real_turn_and_jump(self):
        imu=self.make()
        imu.update(quaternion(np.deg2rad(179)),[0,0,np.deg2rad(100)],(1,1),1.,1.)
        m=imu.update(-quaternion(np.deg2rad(-179)),[0,0,np.deg2rad(100)],(1,2),1.02,1.02)
        self.assertNotEqual(m['imu_health'],'unexplained_yaw_jump')
        m=imu.update(quaternion(0),[0,0,0],(1,3),1.04,1.04)
        self.assertEqual(m['imu_health'],'unexplained_yaw_jump')

    def test_filter_rejects_insufficient_sample_rate(self):
        imu=PelvisImu(dict(DEFAULTS,gyro_cutoff_hz=10.))
        imu.update(quaternion(),[0]*3,(1,1),1.,1.)
        self.assertEqual(imu.update(quaternion(),[0]*3,(1,2),1.06,1.06)['imu_health'],
                         'sample_rate_below_filter_nyquist')


class HeadingTests(unittest.TestCase):
    def test_every_translation_direction_uses_same_pelvis_feedback(self):
        for xy in [( .2,0),(-.2,0),(0,.2),(0,-.2),(.2,.1),(-.2,-.1)]:
            rig=Rig(); command=(*xy,0)
            rig.tick(command)
            for i in range(80): out,h=rig.tick(command,yaw=min(i*.002,.1),omega=(0,0,.1 if i<50 else 0))
            np.testing.assert_array_equal(out[:2],np.asarray(xy,dtype=np.float32))
            self.assertLess(out[2],0)
            self.assertEqual(h['yaw_ref'],0)
            self.assertEqual(h['heading_state'],'HOLD')
            self.assertLessEqual(abs(out[2]),.15)

    def test_actual_hold_feedback_across_pi_has_short_correct_sign(self):
        for sign in [-1,1]:
            rig=Rig();rig.control.clear()
            reference=sign*np.deg2rad(179)
            for _ in range(20):rig.tick(yaw=reference)
            rig.tick((.2,0,0),yaw=reference)
            for _ in range(25):out,h=rig.tick((.2,0,0),yaw=-reference)
            self.assertAlmostEqual(np.rad2deg(h['heading_error']),-sign*2)
            self.assertLess(out[2]*sign,0)
            self.assertEqual(h['imu_health'],'healthy')

    def test_zero_error_does_not_turn(self):
        rig=Rig()
        for _ in range(50):out,h=rig.tick((.2,0,0))
        self.assertEqual(out[2],0)
        self.assertEqual(h['heading_error'],0)

    def test_turn_releases_old_reference_then_holds_new_heading(self):
        rig=Rig();rig.tick((.2,.1,0))
        for i in range(100):out,h=rig.tick((.2,.1,.2),yaw=i*.004,omega=(0,0,.2))
        self.assertEqual(h['heading_state'],'TURNING')
        self.assertIsNone(h['yaw_ref']);self.assertEqual(h['correction_rad_s'],0)
        self.assertAlmostEqual(out[2],.2)
        for _ in range(70):out,h=rig.tick((.2,.1,0),yaw=.4)
        self.assertEqual(h['heading_state'],'HOLD')
        self.assertAlmostEqual(h['yaw_ref'],.4)
        self.assertEqual(out[2],0)

    def test_pure_rotation_is_not_stop_and_no_turn_damping(self):
        rig=Rig(heading_kd=.3)
        for _ in range(100):out,h=rig.tick((0,0,-.2),omega=(0,0,-.2))
        self.assertEqual(h['heading_state'],'TURNING')
        self.assertAlmostEqual(out[2],-.2)
        self.assertIsNone(h['heading_D'])

    def test_turn_release_hysteresis_and_maximum_wait(self):
        rig=Rig();rig.tick((.2,0,.1))
        for _ in range(20):out,h=rig.tick((.2,0,.02),omega=(0,0,.2))
        self.assertEqual(h['heading_state'],'TURNING')
        for i in range(45):out,h=rig.tick((.2,0,0),yaw=.004*i,omega=(0,0,.2))
        self.assertEqual(h['heading_state'],'HOLD')
        self.assertIsNotNone(h['yaw_ref'])

    def test_excess_residual_turn_requests_safety(self):
        rig=Rig();rig.tick((.2,0,.2))
        with self.assertRaisesRegex(ImuFault,'residual'):
            for i in range(60):rig.tick((.2,0,0),yaw=.024*i,omega=(0,0,1.2))

    def test_stop_restart_and_translation_hysteresis(self):
        rig=Rig();rig.tick((.1,0,0))
        _,h=rig.tick((.02,0,0));self.assertEqual(h['heading_state'],'HOLD')
        for _ in range(50):out,h=rig.tick((0,0,0),yaw=.05)
        np.testing.assert_array_equal(out,0);self.assertIsNone(h['yaw_ref'])
        _,h=rig.tick((.02,0,0),yaw=.05);self.assertEqual(h['heading_state'],'STOPPED')
        _,h=rig.tick((-.1,0,0),yaw=.05);self.assertAlmostEqual(h['yaw_ref'],.05)

    def test_external_source_handoff_without_heading_hold(self):
        for source in ['EXTERNAL','UNKNOWN']:
            rig=Rig();rig.tick((.2,0,0))
            for _ in range(30):out,h=rig.tick((.2,0,0),yaw=.05,source=source)
            self.assertIsNone(h['yaw_ref']);self.assertEqual(out[2],0)
            _,h=rig.tick((.2,0,0),yaw=.05)
            self.assertAlmostEqual(h['yaw_ref'],.05)

    def test_navigation_small_turn_tracks_moving_reference_without_opposition(self):
        for requested in (.02, -.02, .002, -.002):
            rig=Rig(heading_kd=.2)
            yaw=3.13
            rig.control.clear()
            for _ in range(20):rig.tick(yaw=yaw)
            previous=0.
            for _ in range(250):
                yaw=wrap_angle(yaw+previous*.02)
                out,h=rig.tick((.2,0,requested),yaw=yaw,omega=(0,0,previous),source='NAVIGATION')
                self.assertEqual(h['heading_state'],'TRACKING')
                self.assertGreater(out[2]*requested,0)
                self.assertLess(abs(h['heading_error']),.001)
                previous=requested
            self.assertAlmostEqual(float(out[2]),requested,places=5)
            self.assertAlmostEqual(h['heading_D'],0,places=5)

    def test_navigation_straight_drift_and_bounded_reference_then_stop(self):
        rig=Rig(heading_ki=.1)
        rig.tick((.2,0,0),source='NAVIGATION')
        for i in range(70):out,h=rig.tick((.2,0,0),yaw=min(i*.002,.1),source='NAVIGATION')
        self.assertLess(out[2],0)
        self.assertLessEqual(abs(out[2]),.15)
        for _ in range(500):out,h=rig.tick((.2,0,.2),yaw=.1,source='NAVIGATION')
        self.assertLessEqual(abs(h['heading_error']),.200001)
        self.assertLessEqual(abs(rig.control.integral),.04)
        for _ in range(150):out,h=rig.tick(source='NAVIGATION',smooth_zero=True)
        np.testing.assert_array_equal(out,0)
        self.assertIsNone(h['yaw_ref'])
        self.assertEqual(rig.control.integral,0)

    def test_navigation_source_change_rebases_and_disabled_shadow_preserve_commands(self):
        rig=Rig();rig.tick((.2,0,0))
        _,h=rig.tick((.2,0,.02),yaw=.05,source='NAVIGATION')
        self.assertAlmostEqual(h['yaw_ref'],.05)
        _,h=rig.tick((.2,0,0),yaw=.06)
        self.assertEqual(h['heading_state'],'HOLD')
        self.assertAlmostEqual(h['yaw_ref'],.06)
        for mode in ('disabled','shadow'):
            rig.control.stop('test')
            rig.control.configure(dict(DEFAULTS,run_mode=mode),quaternion())
            for _ in range(20):rig.tick()
            for _ in range(100):out,h=rig.tick((.2,.1,.02),source='NAVIGATION')
            np.testing.assert_array_equal(out,np.asarray([.2,.1,.02],dtype=np.float32))
            self.assertEqual(rig.control.integral,0)

    def test_pushcart_tracks_student_turn_even_when_cart_requests_straight(self):
        for requested in (.002, -.002, .2, -.2):
            with self.subTest(robot_wz=requested):
                rig=Rig(heading_kd=.2)
                rig.control.clear()
                yaw=3.13
                for _ in range(20):rig.tick(yaw=yaw)
                previous=0.
                for _ in range(250):
                    yaw=wrap_angle(yaw+previous*.02)
                    out,h=rig.tick((.2,.1,requested), yaw=yaw, omega=(0,0,previous),
                                   source='PUSHCART', raw_command=(.3,0,0))
                    self.assertTrue(h['pushcart_tracking'])
                    self.assertFalse(h['navigation_tracking'])
                    self.assertLess(abs(h['heading_error']),.001)
                    self.assertGreater(out[2]*requested,0)
                    previous=requested
                np.testing.assert_allclose(out,[.2,.1,requested],atol=1e-6)
                self.assertAlmostEqual(h['heading_D'],0,places=5)

    def test_pushcart_corrects_drift_bounds_reference_and_suppresses_stop_feedback(self):
        rig=Rig(heading_ki=.1)
        rig.tick((.2,0,0),source='PUSHCART')
        for i in range(70):
            out,h=rig.tick((.2,0,0),yaw=min(i*.002,.1),source='PUSHCART')
        self.assertLess(out[2],0)
        self.assertLessEqual(abs(out[2]),.15)
        for _ in range(500):out,h=rig.tick((0,0,.2),yaw=.1,source='PUSHCART')
        self.assertLessEqual(abs(h['heading_error']),.200001)
        self.assertLessEqual(abs(rig.control.integral),.04)
        previous=out.copy()
        for _ in range(150):
            out,h=rig.tick(yaw=.1,source='PUSHCART',smooth_zero=True)
            self.assertFalse(h['correction_enabled'])
            self.assertEqual(h['correction_rad_s'],0.)
            self.assertIsNone(h['yaw_ref'])
            self.assertTrue(np.all(abs(out)<=abs(previous)+1e-7))
            self.assertTrue(np.all(out*previous>=0.))
            previous=out.copy()
        np.testing.assert_array_equal(out,0.)
        self.assertEqual(rig.control.integral,0.)

    def test_pushcart_handoff_and_mode_controls(self):
        rig=Rig();rig.tick((.2,0,0))
        _,h=rig.tick((.2,0,.02),yaw=.05,source='PUSHCART')
        self.assertAlmostEqual(h['yaw_ref'],.05)
        _,h=rig.tick((.2,0,0),yaw=.06)
        self.assertEqual(h['heading_state'],'HOLD')
        self.assertAlmostEqual(h['yaw_ref'],.06)
        for mode in ('disabled','shadow'):
            rig.control.stop('test')
            rig.control.configure(dict(DEFAULTS,run_mode=mode,heading_ki=.1),quaternion())
            for _ in range(20):rig.tick()
            for _ in range(100):out,h=rig.tick((.2,.1,.02),source='PUSHCART')
            np.testing.assert_array_equal(out,np.asarray([.2,.1,.02],dtype=np.float32))
            self.assertFalse(h['correction_enabled'])
            self.assertEqual(rig.control.integral,0.)

    def test_defaults_disabled_shadow_exact_baseline_and_integral_frozen(self):
        self.assertEqual(DEFAULTS['run_mode'],'disabled')
        for mode in ['disabled','shadow']:
            rig=Rig(heading_ki=.2);rig.control.configure(dict(DEFAULTS,run_mode=mode,heading_ki=.2),quaternion())
            for _ in range(20):rig.tick()
            rig.tick((.2,.1,0))
            for i in range(80):out,h=rig.tick((.2,.1,0),yaw=min(.1,i*.002))
            np.testing.assert_array_equal(out,np.array([.2,.1,0],dtype=np.float32))
            self.assertEqual(rig.control.integral,0)
            if mode=='shadow': self.assertLess(h['wz_suggested'],0)
            rig.control.stop('mode_change_at_zero')
            rig.control.configure(dict(DEFAULTS,run_mode='active'),quaternion(.1))
            self.assertIsNone(rig.control.target);self.assertEqual(rig.control.integral,0)

    def test_fault_recovery_requires_healthy_window_and_neutral(self):
        rig=Rig();rig.tick((.2,0,0))
        _,h=rig.tick((.2,0,0),yaw=1.)
        self.assertEqual(h['imu_health'],'unexplained_yaw_jump')
        self.assertIsNone(h['yaw_ref'])
        for _ in range(80):out,h=rig.tick((.2,0,0),yaw=1.)
        self.assertEqual(h['heading_state'],'RECOVERING')
        self.assertEqual(out[2],0)
        rig.tick(yaw=1.)
        _,h=rig.tick((.2,0,0),yaw=1.)
        self.assertEqual(h['heading_state'],'HOLD');self.assertAlmostEqual(h['yaw_ref'],1.)

    def test_optional_d_is_filtered_heading_rate_not_error_difference(self):
        rig=Rig(heading_kd=.2);rig.tick((.2,0,0))
        for _ in range(50):_,h=rig.tick((.2,0,0),omega=(0,0,.1))
        self.assertAlmostEqual(h['heading_D'],-.02,places=5)
        self.assertEqual(h['heading_P'],0)

    def test_integral_antiwindup_final_slew_limits_and_exit(self):
        rig=Rig(heading_kp=0.,heading_ki=.4,integral_limit_rad_s=.03,max_correction_rad_s=.02)
        rig.tick((.2,0,0)); previous=0
        for i in range(500):
            out,h=rig.tick((.2,0,0),yaw=min(.1,i*.002))
            self.assertLessEqual(abs(out[2]-previous),.008001);previous=out[2]
            self.assertLessEqual(abs(out[2]),.020001)
            self.assertLessEqual(abs(rig.control.integral),.021)
        for i in range(100):out,h=rig.tick((.2,0,0),yaw=max(0,.1-i*.002))
        rig.tick((0,0,.2));self.assertEqual(rig.control.integral,0)
        rig=Rig(heading_kp=2.,heading_ki=.3)
        rig.tick((.2,0,0))
        for i in range(50):rig.tick((.2,0,0),yaw=-min(i*.01,.4))
        self.assertLessEqual(abs(rig.control.integral),.01)

    def test_final_bounds_and_zero_ramp_monotonic_then_hard_stop(self):
        rig=Rig()
        for _ in range(150):out,h=rig.tick((.2,-.1,.8))
        self.assertAlmostEqual(out[2],.8)
        previous=out.copy()
        for _ in range(120):
            out,h=rig.tick(smooth_zero=True)
            self.assertTrue(np.all(abs(out)<=abs(previous)+1e-7))
            self.assertTrue(np.all(out*previous>=0));previous=out
        np.testing.assert_array_equal(out,0)
        self.assertFalse(h['decelerating'])
        rig.control.stop('emergency');np.testing.assert_array_equal(rig.control.output,0)
        self.assertIsNone(rig.control.target)

    def test_control_gap_not_disguised_by_dt_clipping(self):
        rig=Rig()
        with self.assertRaisesRegex(ImuFault,'Control loop gap'):rig.tick((.2,0,0),dt=.2)

    def test_invalid_config_rejected(self):
        for patch in [dict(heading_kp=float('nan')),dict(command_frame='world'),dict(rate_control_enabled=True),
                      dict(turn_exit_rad_s=.04),dict(gyro_cutoff_hz=26.),dict(imu_timeout_s=1.)]:
            with self.assertRaises(ValueError):validate_settings(dict(DEFAULTS,**patch))

if __name__=='__main__':unittest.main(verbosity=2)
