from pathlib import Path
import sys
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'python'))
from hybrid_deploy import protocol as p
from hybrid_deploy.policy import read_bundle
from hybrid_deploy.web_control import WebControl, joint_names
from hybrid_deploy.run import calibration_status


class WebTests(unittest.TestCase):
    def setUp(self):
        self.cfg, _, _ = read_bundle(ROOT)
        self.state = p.State(42, 1, 10_000_000_000, 1, p.WAITING, 3, 0, 1,
                             np.zeros(29), np.zeros(29), np.zeros(3), np.array([1,0,0,0]),
                             np.zeros(3), np.zeros(4), np.zeros(29))
        self.web = WebControl(self.cfg, True)
        self.now = 10.

    def send(self, kind, client=1, **kwargs):
        self.now += .001
        self.state.received_ns = int((self.now-.0001)*1e9)
        event = dict(kind=kind, client=client, time=self.now, session=42,
                     feedback_ns=self.state.received_ns, **kwargs)
        return self.web.accept(event, self.state, 6, self.now)

    def move(self):
        self.send('claim')
        self.state.stage = p.POLICY
        self.send('move', velocity=[.3, .1, .2])
        return self.web.update(self.state, .02, self.now)

    def test_shadow_rejects_all_controls(self):
        self.web.enabled = False
        for kind in ('claim', 'prepare', 'policy', 'move', 'stop'):
            self.assertIsNone(self.send(kind, velocity=[1,1,1]))
        self.assertEqual(self.web.source, 'remote')

    def test_state_machine_requests_require_owner_and_stage(self):
        self.assertIsNone(self.send('prepare'))
        self.send('claim')
        self.assertEqual(self.send('prepare'), p.PREPARE)
        self.assertIsNone(self.send('policy'))
        self.state.stage = p.HOLDING
        self.assertEqual(self.send('policy'), p.RUN)
        self.state.flags = 2
        self.assertIsNone(self.send('policy'))

    def test_other_browser_cannot_take_over_or_move(self):
        self.send('claim')
        self.send('claim', client=2)
        self.assertEqual(self.web.owner, 1)
        self.state.stage = p.POLICY
        self.send('move', client=2, velocity=[.3,0,0])
        self.assertFalse(self.web.update(self.state, .02, self.now).any())
        self.send('remote', client=2)
        self.assertEqual(self.web.source, 'web')

    def test_clamp_and_training_filter(self):
        self.send('claim')
        self.state.stage = p.POLICY
        self.send('move', velocity=[100,-100,100])
        np.testing.assert_allclose(self.web.target, [1.2,-.5,.8])
        np.testing.assert_allclose(self.web.update(self.state,.02,self.now), [.06,-.025,.04])
        self.send('move', velocity=[-100,100,-100])
        np.testing.assert_allclose(self.web.target, [-.8,.5,-.8])

    def test_release_timeout_and_disconnect_do_not_restore_remote(self):
        self.assertTrue(self.move().any())
        self.assertFalse(self.web.update(self.state,.02,self.now+.31).any())
        self.send('move',velocity=[.3,0,0])
        self.send('disconnect')
        self.assertEqual(self.web.source, 'web')
        self.assertIsNone(self.web.owner)
        self.assertFalse(self.web.update(self.state,.02,self.now).any())

    def test_l2_b_health_and_nonpolicy_zero_motion(self):
        for buttons, flags, stage in [(p.L2,3,p.POLICY), (p.B,3,p.POLICY),
                                      (0,2,p.POLICY), (0,3,p.HOLDING)]:
            self.state.buttons, self.state.flags = 0, 3
            self.move()
            self.state.buttons, self.state.flags, self.state.stage = buttons, flags, stage
            self.assertFalse(self.web.update(self.state,.02,self.now).any())

    def test_stop_allowed_from_observer_even_unhealthy(self):
        self.move()
        self.state.flags = 2
        self.assertEqual(self.send('stop',client=99), p.STOP)
        self.assertFalse(self.web.value.any())

    def test_invalid_expired_replayed_and_wrong_session(self):
        self.send('claim')
        self.state.stage = p.POLICY
        event = dict(kind='move',client=1,time=self.now,session=42,
                     feedback_ns=self.state.received_ns,velocity=[.1,0,0])
        for changes in [dict(time=self.now-.5),dict(time=self.now+1),dict(session=999),
                        dict(feedback_ns=1),dict(time=float('nan')),dict()]:
            self.web.accept(dict(event, **changes),self.state,6,self.now)
            self.assertFalse(self.web.target.any())
        for velocity in ([float('nan'),0,0],[1,2],['bad',0,0]):
            self.send('move',velocity=velocity)
            self.assertFalse(self.web.target.any())

    def test_zero_and_explicit_handback(self):
        self.move()
        self.send('zero',client=2)
        self.assertFalse(self.web.value.any())
        self.send('remote')
        self.assertEqual(self.web.source,'remote')
        self.assertIsNone(self.web.owner)

    def test_policy_requires_six_history_frames(self):
        self.send('claim')
        self.state.stage = p.HOLDING
        self.now += .001
        event = dict(kind='policy', client=1, time=self.now, session=42,
                     feedback_ns=self.state.received_ns)
        self.assertIsNone(self.web.accept(event, self.state, 5, self.now))
        self.assertEqual(self.send('policy'), p.RUN)

    def test_calibration_checks_legs_but_not_held_upper_body(self):
        cfg, contract, _ = read_bundle(ROOT)
        self.state.q[:] = cfg['default_q_sdk29']
        self.state.applied_q[:] = self.state.q
        self.state.q[14] = .2
        self.state.q[15] = 1.
        self.state.q[11] = .2
        result = calibration_status(self.state, cfg, contract)
        self.assertEqual(result['position_pending'], [11])
        self.assertEqual(result['worst_joint'], 'right_ankle_roll_joint')


if __name__ == '__main__':
    unittest.main()
