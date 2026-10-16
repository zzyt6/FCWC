"""Joint order, missing/stale command handling, UI selection and saved errors."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gzip
import json
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from core.joint_tracking import INDICES, NAMES, snapshot, format_joint
from core.joint_monitor import limits, out_of_range, live_warnings
from core.recording import SessionRecorder
from core.robot import RobotConfig
from Vender.app import Viewer
from Vender.joint_panel import joint_table


class JointTrackingTests(unittest.TestCase):
    def setUp(self):
        self.now = 10.005
        self.state = SimpleNamespace(
            q=np.arange(29)*.01, hand_q=np.arange(14)*.02, received_ns=10_000_000_000,
            hand_received_ns=(10_000_000_001, 10_000_000_002),
            bridge_ns=10_000_000_004, sent_ns=(10_000_000_001, 10_000_000_002, 10_000_000_003),
            position_mask=(1 << 43)-1, sent_q=np.arange(43)*.03, flags=3, hand_flags=35)

    def comparison(self):
        return snapshot(self.state, self.now, .2)

    def test_mapping_sign_and_sent_target_preserves_compensation(self):
        c = self.comparison()
        self.assertEqual(len(NAMES), 41)
        self.assertEqual(INDICES[13], 15)
        self.assertEqual(NAMES[34], 'right_hand_thumb_0_joint')
        self.assertTrue(all(c['valid']))
        np.testing.assert_allclose(c['command_rad'], self.state.sent_q[list(INDICES)])
        np.testing.assert_allclose(c['actual_rad'], np.r_[self.state.q, self.state.hand_q][list(INDICES)])
        np.testing.assert_allclose(c['error_rad'], np.array(c['command_rad'])-c['actual_rad'])
        self.state.sent_q[36] = .7
        c = self.comparison()
        self.assertEqual(c['command_rad'][34], .7)
        self.assertAlmostEqual(c['error_rad'][34], .56)

    def test_observe_and_independent_missing_hand(self):
        self.state.sent_ns = (0, 0, 0)
        c = self.comparison()
        self.assertEqual(c['command_rad'], [None]*41)
        self.assertEqual(c['error_rad'], [None]*41)
        self.assertIsNotNone(c['actual_rad'][0])
        self.assertIn('尚未下发', format_joint(c, 0, self.now))
        self.state.hand_received_ns = (0, 10_000_000_002)
        c = self.comparison()
        self.assertEqual(c['actual_rad'][27:34], [None]*7)
        self.assertIsNotNone(c['actual_rad'][34])

    def test_stale_feedback_publish_failure_and_damping(self):
        self.state.hand_received_ns = (9_000_000_000, 10_000_000_002)
        c = self.comparison()
        self.assertFalse(any(c['valid'][27:34]))
        self.assertTrue(all(c['valid'][34:]))
        self.assertIn('反馈已过期', format_joint(c, 27, self.now))
        self.state.sent_ns = (9_000_000_000, 10_000_000_002, 10_000_000_003)
        c = self.comparison()
        self.assertEqual(c['error_rad'][:27], [None]*27)
        self.assertIn('指令记录已过期', format_joint(c, 0, self.now))
        self.state.position_mask = 0
        c = self.comparison()
        self.assertFalse(any(c['valid']))
        self.assertIn('Kp=0', format_joint(c, 34, self.now))

    def test_display_expires_even_if_controller_stops_updating(self):
        c = self.comparison()
        text = format_joint(c, 34, self.now)
        self.assertIn('rad', text)
        self.assertIn('°', text)
        self.assertIn('+0.94000', text)
        text = format_joint(c, 34, self.now+1)
        self.assertIn('已过期', text)
        self.assertIn('| 差值（下发 − 实际） | — | — |', text)

    def test_compact_degree_table_warning_threshold_and_red_precedence(self):
        config = RobotConfig()
        lower, upper = limits(config)
        self.state.q = config.default_q.copy()
        self.state.hand_q = np.zeros(14)
        c = self.comparison()
        c['command_rad'] = list(c['actual_rad'])
        c['command_rad'][0] += np.deg2rad(6)
        c['command_rad'][1] -= np.deg2rad(4)
        c['actual_rad'][3] = upper[3] + .1
        bad = set(out_of_range(c, lower, upper))
        table = joint_table(c, bad, self.now)
        self.assertEqual(table.count('class="n warn"'), 1)
        self.assertEqual(table.count('class="n bad"'), 1)
        self.assertIn('class="n bad" title="left_knee_joint', table)
        self.assertIn('>+6.0</td>', table)
        self.assertNotIn('>实际</th>', table)
        self.assertEqual(table.count('scope="col"'), 6)
        self.assertEqual(table.count('<tr>'), 22)  # Header plus 21 paired rows.
        self.assertEqual(joint_table(c, bad, self.now, 7).count('class="n warn"'), 0)
        self.assertEqual(joint_table(c, bad, self.now, 3).count('class="n warn"'), 2)
        warnings = live_warnings(dict(stage='POLICY', state=None, reason=''), c, lower, upper, .2)
        self.assertEqual(len(warnings), 1)
        self.assertEqual(warnings[0][0], 'warning')
        self.assertIn('left_knee_joint 位置超限', warnings[0][1])

    def test_left_panel_update_shares_one_view(self):
        viewer = object.__new__(Viewer)
        data = dict(stage='OBSERVE', healthy=True, state=None, owner=None, source='web',
                    reason='', velocity=np.zeros(3), joint_comparison=self.comparison())
        viewer.control = SimpleNamespace(snapshot=lambda:data, mode='observe', recorder=None)
        viewer.clients = {}; viewer.motion_panels = {}
        viewer.last_stage = 'OBSERVE'; viewer.last_text = 0
        viewer.last_push_enabled = False
        viewer.pushcart_panels = {}; viewer.cart_display = Mock(); viewer.base = Mock()
        viewer.status = Mock(); viewer.diagnostics = Mock()
        viewer.joint_monitor = Mock()
        with patch('Vender.app.time.monotonic', return_value=self.now):
            viewer.update()
        viewer.joint_monitor.update.assert_called_once_with(data, self.now)

    def test_monitor_shows_all_joints_and_stale_errors_are_not_yellow(self):
        table = joint_table(None, set(), self.now)
        self.assertEqual(table.count('class="n '), 41)
        c = self.comparison()
        stale = joint_table(c, set(), self.now+1)
        self.assertNotIn('class="n warn"', stale)
        self.assertIn('class="stale">—</td>', stale)

    def test_nonfinite_angles_never_show_as_valid_errors(self):
        self.state.q[0] = float('nan')
        c = self.comparison()
        self.assertIsNone(c['error_rad'][0])
        row = joint_table(c, set(), self.now).split('left_hip_pitch_joint', 1)[1].split('</tr>')[0]
        self.assertNotIn('nan', row)

    def test_gzip_roundtrip_and_offline_error_statistics(self):
        config = RobotConfig()
        with tempfile.TemporaryDirectory() as directory:
            recorder = SessionRecorder(config, {'mode':'control'}, directory)
            c = self.comparison()
            row = dict(stage='HOLDING', velocity=[0,0,0], target=[0]*29, joint_comparison=c)
            recorder.record('telemetry', row)
            # This stale frame must not pollute valid error statistics.
            self.now += 1
            recorder.record('telemetry', dict(row, joint_comparison=self.comparison()))
            summary = recorder.close()
            self.assertFalse(summary['recorder_error'])
            with gzip.open(recorder.path/'telemetry.jsonl.gz', 'rt') as stream:
                saved = json.loads(stream.readline())['joint_comparison']
            self.assertEqual(saved, c)
            result = subprocess.run([sys.executable, '-B', str(Path(__file__).resolve().parents[1]/'tools/inspect_session.py'),
                                     str(recorder.path)], check=True, capture_output=True, text=True)
            stats = json.loads(result.stdout)['joint_angle_errors']['joints'][NAMES[34]]
            self.assertEqual(stats['samples'], 1)
            self.assertAlmostEqual(stats['mean_abs'], .94)
            self.assertAlmostEqual(stats['rms'], .94)
            self.assertAlmostEqual(stats['max_abs'], .94)


if __name__ == '__main__':
    unittest.main()
