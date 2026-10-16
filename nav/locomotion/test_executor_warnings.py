"""Fault injection: real nav UDP -> Viser Controller -> C++ control core.

Uses pipes and ephemeral loopback ports only: no DDS or physical motor writes.
Build outputs live in an automatically removed system temporary directory.
"""
from contextlib import contextmanager
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

import numpy as np

from g1_locomotion_executor import Executor, MOTION_CONFIG, WORKSPACE
from core.controller import Controller
from core.robot import ACTIVE, RobotConfig
from core.joint_tracking import NAMES
from hybrid_deploy import protocol as p

sys.path.insert(0, str(WORKSPACE / 'Viser/test'))
sys.path.insert(0, str(WORKSPACE / 'nav/src/unitree_nav2_web'))
from test_pose_integration import HARNESS, PipeBridge
from unitree_nav2_web.locomotion import LocomotionLink


class WarningIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = RobotConfig(motion_config=MOTION_CONFIG)
        cls.directory = tempfile.TemporaryDirectory(prefix='nav_warning_test_')
        cls.addClassCleanup(cls.directory.cleanup)
        root = Path(cls.directory.name)
        source = root / 'core.cpp'
        source.write_text(HARNESS)
        cls.binary = root / 'core'
        subprocess.run([sys.executable, '-B', str(cls.config.deploy / 'tools/generate_config.py'),
                        str(cls.config.deploy / 'config/deploy.json'),
                        str(root / 'generated_config.hpp')], check=True, capture_output=True)
        subprocess.run(['g++', '-std=c++17', '-O2', '-I' + str(cls.config.deploy / 'cpp'),
                        '-I' + str(root), str(source), '-o', str(cls.binary)],
                       check=True, capture_output=True)

    @contextmanager
    def chain(self, scenario):
        self.bridge = PipeBridge(self.binary, self.config, scenario)
        self.control = Controller(self.config, self.bridge, 'sim', clock=self.bridge.now)
        self.link = LocomotionLink(0, 0, clock=self.bridge.now)
        self.executor = Executor(self.control, 0, self.link.sock.getsockname()[1], clock=self.bridge.now)
        self.link.destination = self.executor.sock.getsockname()
        try:
            self.control.step()
            self.tick()
            self.tick()  # Executor's internal owner claim is now acknowledged.
            yield
        finally:
            self.executor.close()
            self.link.close()
            self.bridge.close()

    def tick(self):
        self.executor.step()
        self.control.step()
        now = self.bridge.now()
        self.executor.last_status = -float('inf')
        self.executor._report(self.control.snapshot(), now, False)
        deadline = time.monotonic() + 1.
        while time.monotonic() < deadline:
            with self.link.lock:
                acknowledged = self.link.status.get('sent_monotonic') == now
            if acknowledged:
                return self.link.snapshot()
            time.sleep(.001)
        self.fail('nav web link did not receive the current warning/status packet')

    def until(self, stage, count=750):
        for _ in range(count):
            status = self.tick()
            self.assertNotIn(status['stage'], ('DAMPED', 'FAULT'), status)
            if status['stage'] == stage:
                return status
        self.fail('did not reach ' + stage)

    def start_policy(self):
        self.link.request('prepare')
        self.until('HOLDING')
        self.start_ready_policy()

    def start_ready_policy(self):
        # Like the web button, wait for six real observation frames after
        # arriving at HOLDING; merely reaching the stage is not sufficient.
        for _ in range(10):
            if self.tick()['actions']['policy']:
                break
        self.assertTrue(self.link.snapshot()['actions']['policy'])
        self.link.request('policy')
        return self.until('POLICY')

    def assert_body_supported(self, status):
        self.assertTrue(status['healthy'] and status['ready'], status)
        self.assertEqual(self.bridge.state.stage, p.POLICY)
        self.assertTrue(np.all(self.bridge.state.applied_kp[ACTIVE] > 0))
        mask = sum(1 << int(index) for index in ACTIVE)
        self.assertEqual(self.bridge.state.position_mask & mask, mask)

    def test_every_body_joint_warning_reaches_nav_without_removing_support(self):
        with self.chain('body_limits'):
            self.assertTrue(self.link.snapshot()['actions']['prepare'])
            self.start_policy()
            seen = set()
            previous = self.bridge.state.sent_ns[0]
            for _ in range(180):
                status = self.tick()
                self.assert_body_supported(status)
                self.assertEqual(self.bridge.state.sent_ns[0] - previous, 20_000_000)
                previous = self.bridge.state.sent_ns[0]
                for warning in status['warnings']:
                    if ' 位置超限' in warning['message']:
                        self.assertEqual(warning['level'], 'warning')
                        seen.add(warning['message'].split(' 位置超限')[0])
            self.assertEqual(seen, set(NAMES[:27]))
            self.assertTrue(any(not w['active'] for w in status['warnings']))
            self.link.request('stop')
            self.tick()
            self.assertEqual(self.bridge.state.stage, p.DAMPED)
            self.assertEqual(self.bridge.state.position_mask, 0)

    def test_hand_limits_and_missing_feedback_warn_but_allow_body_policy(self):
        for scenario in ('stuck_hand', 'no_hand', 'sdk_prepare'):
            with self.subTest(scenario=scenario), self.chain(scenario):
                self.start_policy()
                saw_warning = False
                for _ in range(100):
                    status = self.tick()
                    self.assert_body_supported(status)
                    saw_warning |= any('Dex3' in w['message'] or 'hand_' in w['message']
                                       for w in status['warnings'])
                self.assertTrue(saw_warning, status)
                self.assertEqual(self.control.hand_pose, 0)
                self.assertEqual(self.bridge.state.position_mask >> 29, 0)

    def test_prepare_timeout_is_displayed_while_position_support_continues(self):
        with self.chain('prepare_timeout'):
            self.link.request('prepare')
            saw_warning = False
            for _ in range(750):
                status = self.tick()
                self.assertNotIn(status['stage'], ('DAMPED', 'FAULT'))
                if status['preparation_warning']:
                    saw_warning = True
                    self.assertEqual(status['stage'], 'PREPARING')
                    self.assertFalse(status['actions']['policy'])
                    self.assertTrue(np.all(self.bridge.state.applied_kp[ACTIVE] > 0))
                    self.assertTrue(any('准备姿态超时' in w['message'] for w in status['warnings']))
                if status['stage'] == 'HOLDING':
                    break
            self.assertTrue(saw_warning)
            self.assertEqual(status['stage'], 'HOLDING')
            self.assert_body_supported(self.start_ready_policy())


if __name__ == '__main__':
    unittest.main(verbosity=2)
