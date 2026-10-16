"""Dex3 position warnings leave body preparation available; SDK faults remain distinct."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from core.hand_status import feedback_reasons
from core.controller import Controller
from core.robot import RobotConfig
from core.joint_monitor import limits, live_warnings
from core.joint_tracking import snapshot as joint_snapshot
from hybrid_deploy import protocol as p
from Vender.app import Viewer
from test_control import FakeBridge


class HandStatusTests(unittest.TestCase):
    def setUp(self):
        self.now = 100.
        self.config = RobotConfig()
        self.bridge = FakeBridge(self.config)
        self.state = self.bridge.state
        self.state.hand_received_ns = (99_995_000_000, 99_995_000_000)
        self.state.hand_flags = 35  # Position alone no longer clears readiness.
        self.state.hand_q[8] = -1.118628

    def test_field_angle_is_warning_not_feedback_fault(self):
        self.assertEqual(feedback_reasons(self.state, self.now), ['', ''])
        comparison = joint_snapshot(self.state, self.now, .2)
        lower, upper = limits(self.config)
        warnings = live_warnings({}, comparison, lower, upper, .2, now=self.now)
        self.assertEqual(len(warnings), 1)
        self.assertEqual(warnings[0][0], 'warning')
        self.assertIn('right_hand_thumb_1_joint', warnings[0][1])
        self.assertIn('-1.119', warnings[0][1])

    def test_missing_stale_and_other_faults_are_distinct(self):
        self.state.hand_flags = 33
        self.state.hand_received_ns = (0, 0)
        self.assertIn('尚未收到', feedback_reasons(self.state, self.now)[1])
        self.state.hand_received_ns = (99_000_000_000, 99_000_000_000)
        self.assertIn('已过期', feedback_reasons(self.state, self.now)[1])
        self.state.hand_received_ns = (100_001_000_000, 100_001_000_000)
        self.assertIn('时间异常', feedback_reasons(self.state, self.now)[1])
        self.state.hand_received_ns = (99_995_000_000, 99_995_000_000)
        self.state.hand_q[8] = -.5  # Flags still report fault; q cannot reveal its cause.
        self.assertIn('bridge.log', feedback_reasons(self.state, self.now)[1])
        self.state.hand_flags = 35
        self.assertEqual(feedback_reasons(self.state, self.now), ['', ''])

    def test_controller_allows_body_preparation_despite_hand_sdk_fault(self):
        control = Controller(self.config, self.bridge, 'control', clock=lambda:self.now)
        control._prepare(self.state)
        self.assertEqual(control.request, p.PREPARE)
        self.assertEqual(control.reason, '')
        self.assertEqual(control._hand_status(self.state)['blocking_reason'], '')
        self.assertTrue(control._hand_status(self.state)['healthy'])
        control.request = p.WAIT
        self.state.hand_flags = 33
        control._prepare(self.state)
        self.assertEqual(control.request, p.PREPARE)
        self.assertIn('bridge.log', control._hand_status(self.state)['blocking_reason'])

    def test_web_keeps_prepare_enabled_with_hand_position_warning(self):
        control = Controller(self.config, self.bridge, 'control', clock=lambda:self.now)
        data = dict(stage='ZERO', healthy=True, owner=1, source='web', reason='', velocity=np.zeros(3),
                    state=dict(received=self.now, tick=(42,1), q=np.zeros(29), dq=np.zeros(29)),
                    dex3=control._hand_status(self.state))
        viewer = object.__new__(Viewer)
        viewer.control = SimpleNamespace(snapshot=lambda:data, mode='control', recorder=None,
                                          cfg={'state_timeout_s':.2})
        buttons = {key: Mock() for key in ('claim','acquire','prepare','prepare_pose','policy','hand_toggle',
                   'hand_compensation','hand_recovery_joint','hand_recover','move','remote','zero','stop')}
        buttons['hand_recovery_joint'].options = ('选择超限手指',)
        buttons['hand_recovery_joint'].value = '选择超限手指'
        viewer.clients = {1: (Mock(), buttons, [])}
        viewer.motion_panels = {}; viewer.hand_panels = {}; viewer.joint_monitor = Mock()
        viewer.last_stage = 'ZERO'; viewer.last_text = 0; viewer.last_tick = (42,1)
        viewer.last_push_enabled = False
        viewer.pushcart_panels = {}; viewer.cart_display = Mock()
        viewer.base = Mock()
        viewer.status = Mock(); viewer.diagnostics = Mock()
        with patch('Vender.app.time.monotonic', return_value=self.now):
            viewer.update()
            self.assertFalse(buttons['prepare'].disabled)
            self.assertNotIn('准备受阻', viewer.status.content)
            self.assertNotIn('-1.1186', viewer.status.content)
            lower, upper = limits(self.config)
            comparison = joint_snapshot(self.state, self.now, .2)
            warnings = live_warnings(data, comparison, lower, upper, .2, now=self.now)
            self.assertTrue(any(level == 'warning' and 'thumb_1' in text for level, text in warnings))
            self.state.hand_q[8] = -.5; self.state.hand_flags = 35
            data['dex3'] = control._hand_status(self.state)
            viewer.last_text = 0
            viewer.update()
            self.assertFalse(buttons['prepare'].disabled)
            self.assertNotIn('准备按钮被', viewer.status.content)
            self.assertNotIn('-1.1186', viewer.status.content)
            self.assertEqual(live_warnings(data, None, lower, upper, .2, now=self.now), [])
            self.state.hand_flags = 32
            data['dex3'] = control._hand_status(self.state)
            viewer.last_text = 0
            viewer.update()
            self.assertFalse(buttons['prepare'].disabled)
            self.assertNotIn('准备受阻', viewer.status.content)
            data['stage'] = 'POLICY'
            viewer.update()
            self.assertTrue(buttons['hand_toggle'].disabled)
            self.state.hand_flags = 35
            data['dex3'] = control._hand_status(self.state)
            viewer.update()
            self.assertFalse(buttons['hand_toggle'].disabled)
            self.assertIn('使能并抓握', buttons['hand_toggle'].label)
            self.assertFalse(any('已暂停' in text for _, text in live_warnings(data, None, lower, upper, .2, now=self.now)))
            control.hand_ever_started = True
            data['dex3'] = control._hand_status(self.state)
            viewer.last_text = 0
            viewer.update()
            self.assertIn('重新启用并归零', buttons['hand_toggle'].label)


if __name__ == '__main__':
    unittest.main()
