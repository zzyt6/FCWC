"""Confirmed push-cart PD and waist/arm targets across Python -> real C++ core.

Uses the pipe simulator, temporary build files and no robot/DDS connection.
"""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT.parent / 'G1_high_policy_deploy/robot')]
from core.controller import Controller
from core.robot import RobotConfig, UPPER
from hybrid_deploy import protocol as p
from test_pose_integration import HARNESS, PipeBridge
from pushcart_v81.carmer_io import CarmerYaw
from pushcart_v81.viser_policy import ViserPushPolicy


class PushcartPdTests(unittest.TestCase):
    def test_mode_change_pd_targets_and_return_preserve_grasp(self):
        config = RobotConfig()
        with tempfile.TemporaryDirectory(prefix='g1_pushcart_pd_') as directory:
            source, binary = Path(directory)/'core.cpp', Path(directory)/'core'
            source.write_text(HARNESS)
            subprocess.run([sys.executable, '-B', str(config.deploy/'tools/generate_config.py'),
                            str(config.deploy/'config/deploy.json'), str(Path(directory)/'generated_config.hpp')], check=True)
            subprocess.run(['g++', '-std=c++17', '-O2', '-I'+str(config.deploy/'cpp'),
                            '-I'+directory, str(source), '-o', str(binary)], check=True)
            bridge = PipeBridge(binary, config)
            camera = CarmerYaw(0, tag_ids=[0, 1])
            high = ViserPushPolicy(config, ROOT.parent/'G1_high_policy_deploy/robot/config/viser_v81.json', camera=camera)
            control = Controller(config, bridge, 'sim', high_policy=high, clock=bridge.now)
            try:
                def tick(kind=None, velocity=None):
                    now = bridge.now()
                    camera.accept(json.dumps(dict(schema='carmer_cart_yaw_v1', captured_monotonic=now-.001,
                                                  valid=True, reference_frame='pelvis_level', yaw_rad=.1,
                                                  tag_ids=[0, 1])), now)
                    if kind:
                        control.emit(kind, 1, velocity)
                    control.step()
                    return control.snapshot()

                def advance_until(check, limit=400):
                    for _ in range(limit):
                        tick()
                        self.assertNotIn(control.stage, ('FAULT', 'DAMPED'), control.reason)
                        if check():
                            return
                    self.fail('State transition did not complete')

                tick();tick('claim');tick('prepare')
                advance_until(lambda: control.stage=='HOLDING' and control.observations.frames>=6)
                tick('policy')
                advance_until(lambda: control.stage=='POLICY')
                tick('hand_toggle')
                advance_until(lambda: bool(bridge.state.hand_flags&16) and not bridge.state.hand_flags&8)
                # Allow the Python hand-status snapshot to acknowledge completion.
                tick()
                grasp_before = config.hand_grasp.copy()
                hand_target = bridge.state.hand_target.copy()
                upper_before = bridge.state.applied_q[UPPER].copy()
                tick('pushcart_enable')
                self.assertTrue(high.enabled, control.reason)
                self.assertLess(bridge.state.upper_gain_blend, 1.)
                for _ in range(10):
                    data=tick('move', [.3, 0, 0])
                    np.testing.assert_array_equal(data['velocity'], 0)
                    np.testing.assert_allclose(bridge.state.applied_q[UPPER], upper_before, atol=1e-5)
                advance_until(lambda: bridge.state.upper_mode==1 and bridge.state.upper_gain_blend==1.)
                np.testing.assert_array_equal(bridge.state.applied_kp[UPPER], [300]+[100,100,50,50,20,20,20]*2)
                np.testing.assert_array_equal(bridge.state.applied_kd[UPPER], [3]+[2,2,2,2,1,1,1]*2)
                np.testing.assert_array_equal(bridge.state.applied_kp[[13,14]],0)

                def student(policy, history):
                    high.model.raw=np.full(30,.2,dtype=np.float32)
                    high.model.inference_ms=0.
                    return high.model.raw.copy(),np.zeros(19,dtype=np.float32)
                high.model.run_student=student
                for _ in range(60): tick('move',[.3,0,0])
                self.assertEqual(control.policy.selected,'homie')
                self.assertGreater(float(np.max(abs(bridge.state.applied_q[UPPER]-upper_before))),.03)
                np.testing.assert_array_equal(bridge.state.applied_kp[:12],[100,100,100,150,40,40]*2)
                tick('zero')
                advance_until(lambda: not control.zeroing and not np.any(control.snapshot()['velocity']))
                held=bridge.state.applied_q[UPPER].copy()
                tick('pushcart_disable')
                self.assertFalse(high.enabled)
                advance_until(lambda: bridge.state.upper_mode==0 and bridge.state.upper_gain_blend==1.)
                normal=json.loads((config.deploy/'config/balance_pd.json').read_text())
                np.testing.assert_array_equal(bridge.state.applied_kp[UPPER],np.asarray(normal['kp_sdk29'])[UPPER])
                np.testing.assert_array_equal(bridge.state.applied_kd[UPPER],np.asarray(normal['kd_sdk29'])[UPPER])
                np.testing.assert_allclose(bridge.state.applied_q[UPPER],held,atol=1e-5)
                np.testing.assert_array_equal(config.hand_grasp,grasp_before)
                np.testing.assert_array_equal(bridge.state.hand_target,hand_target)
            finally:
                control.close()


if __name__ == '__main__':
    unittest.main()
