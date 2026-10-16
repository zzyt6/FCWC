"""bridge.log warning parsing and the warning box hold time; no bridge or network."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import tempfile
import json
import unittest

from core.joint_monitor import BridgeLogWatcher, WarningBoard, parse_bridge_line, load_monitor_settings
from core.robot import JOINT_NAMES


class JointMonitorTests(unittest.TestCase):
    def test_preparation_warning_updates_without_flooding_and_expires(self):
        now = [0.]
        board = WarningBoard(clock=lambda: now[0])
        board.update([('warning', '准备姿态超时：left_wrist_roll_joint 误差 5.92°')])
        now[0] = 1.
        rows = board.update([('warning', '准备姿态超时：left_wrist_roll_joint 误差 5.91°')])
        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0][2])
        self.assertIn('5.91°', rows[0][1])
        self.assertFalse(board.update([])[0][2])
        now[0] = 7.
        self.assertEqual(board.update([]), [])

    def test_threshold_config_rejects_bad_values(self):
        self.assertEqual(load_monitor_settings()['error_threshold_deg'], 5.)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'monitor.json'
            for value in (0, -1, True, '5', float('nan')):
                path.write_text(json.dumps({'error_threshold_deg': value}))
                with self.assertRaises(ValueError):
                    load_monitor_settings(path)
            path.write_text('{"error_threshold_deg": 8.5}')
            self.assertEqual(load_monitor_settings(path)['error_threshold_deg'], 8.5)
    def test_sdk_health_lines(self):
        cases = {
            'HEALTH false reason=motor_temperature_high lowstate_age_ms=1 remote_age_ms=-1': '温度过高',
            'HEALTH false reason=motor_status_nonzero lowstate_age_ms=1 remote_age_ms=-1': 'motorstate',
            'HEALTH false reason=joint_position_out_of_bounds index=4 q=0.6 lower=-0.87 upper=0.52 '
            'lowstate_age_ms=1': 'left_ankle_pitch_joint 位置超限',
            'DEX3 HEALTH side=right healthy=false topic=t age_ms=1 reason=joint_out_of_bounds index=2 q=0.1':
                'right_hand_thumb_2_joint 位置超限',
            'DEX3 HEALTH side=left healthy=false topic=t age_ms=1 reason=motor_temperature index=1 c=82':
                'left_hand_thumb_1_joint 温度过高',
            'DEX3 HEALTH side=left healthy=false topic=t age_ms=1 reason=hand_error=5': 'HandState.error=5',
            'STAGE 5 feedback unhealthy: motor_temperature_high': '底层故障停止',
        }
        for line, text in cases.items():
            self.assertIn(text, parse_bridge_line(line, 80)[2], line)
        self.assertEqual(parse_bridge_line('HEALTH true lowstate_age_ms=1', 80), ('body', None, None))
        self.assertIsNone(parse_bridge_line('CALIBRATION elapsed_s=1', 80))

    def test_temperature_identifies_all_hot_joints_in_health_and_stop_messages(self):
        reason = ('motor_temperature_high limit_c=80.000000 '
                  'index=19 c=81 temperatures=45,81 index=26 c=83 temperatures=83,46')
        for prefix, suffix in (('HEALTH false reason=', ' lowstate_age_ms=1'),
                               ('STAGE 5 feedback unhealthy: ', ''),
                               ('STAGE 5 command rejected: ', ''),
                               ('STAGE 5 Dex3 recovery: body feedback unhealthy: ', '')):
            with self.subTest(prefix=prefix):
                parsed = parse_bridge_line(prefix + reason + suffix, 90)
                self.assertEqual(parsed[1], 'error')
                self.assertIn('左腕横滚关节（left_wrist_roll_joint，SDK 19） 81 °C', parsed[2])
                self.assertIn('右腕横滚关节（right_wrist_roll_joint，SDK 26） 83 °C', parsed[2])
                self.assertIn('阈值 > 80 °C', parsed[2])
                self.assertNotIn('90 °C', parsed[2])

    def test_temperature_sdk_indices_do_not_shift_after_fixed_waist_slots(self):
        for index, name in enumerate(JOINT_NAMES):
            with self.subTest(index=index):
                parsed = parse_bridge_line(
                    'HEALTH false reason=motor_temperature_high index=%d c=81 lowstate_age_ms=1' % index, 80)
                self.assertIn(name + '，SDK %d' % index, parsed[2])
        legacy = parse_bridge_line('HEALTH false reason=motor_temperature_high lowstate_age_ms=1', 80)
        self.assertIn('日志未记录关节与实测温度', legacy[2])
        unknown = parse_bridge_line('HEALTH false reason=motor_temperature_high index=99 c=82 lowstate_age_ms=1', 80)
        self.assertIn('SDK 99 82 °C', unknown[2])

    def test_temperature_fault_keeps_joint_details_after_feedback_recovers(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'bridge.log'
            watcher = BridgeLogWatcher(path, 80)
            reason = 'motor_temperature_high limit_c=80 index=3 c=81 temperatures=81,44'
            path.write_text('HEALTH false reason=' + reason + ' lowstate_age_ms=1\n'
                            'STAGE 5 feedback unhealthy: ' + reason + '\n')
            watcher.poll()
            self.assertEqual(len(watcher.warnings()), 2)
            with path.open('a') as stream:
                stream.write('HEALTH true lowstate_age_ms=1\n')
            watcher.poll()
            self.assertEqual(len(watcher.warnings()), 1)
            self.assertIn('底层故障停止：电机温度过高：左膝关节（left_knee_joint，SDK 3） 81 °C',
                          watcher.warnings()[0][1])
            with path.open('a') as stream:
                stream.write('BRIDGE mode=OBSERVE_ONLY config=1\n')
            watcher.poll()
            self.assertEqual(watcher.warnings(), [])

    def test_temperature_readings_update_without_flooding(self):
        board = WarningBoard()
        for temperature in (81, 82, 83):
            warning = parse_bridge_line(
                'HEALTH false reason=motor_temperature_high index=19 c=%d lowstate_age_ms=1' % temperature, 80)
            rows = board.update([warning[1:]])
            self.assertEqual(len(rows), 1)
            self.assertIn('%d °C' % temperature, rows[0][1])

    def test_healthy_hand_can_still_have_position_warning(self):
        parsed = parse_bridge_line('DEX3 HEALTH side=left healthy=true topic=t age_ms=1 '
                                   'reason=joint_out_of_bounds index=6 q=-1.79', 80)
        self.assertEqual(parsed[:2], ('dex3_left', 'warning'))
        self.assertIn('left_hand_index_1_joint', parsed[2])

    def test_body_reference_warnings_clear_without_a_global_fault(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'bridge.log'
            watcher = BridgeLogWatcher(path, 80)
            path.write_text('BODY POSITION warning index=11 q=0.286337 lower=-0.2618 upper=0.2618 margin=0.02\n'
                            'BODY TARGET warning index=15 q=3 lower=-3.0892 upper=2.6704 margin=0\n'
                            'BODY EFFORT warning index=20 kp=1 kd=0.1\n')
            watcher.poll()
            warnings = watcher.warnings()
            self.assertEqual(len(warnings), 3)
            self.assertTrue(all(level == 'warning' and '机身控制继续' in text for level, text in warnings))
            self.assertTrue(any('right_ankle_roll_joint 位置超限' in text for _, text in warnings))
            self.assertTrue(any('已限幅' in text for _, text in warnings))
            with path.open('a') as stream:
                stream.write('BODY POSITION recovered index=11 q=0.2\n'
                             'BODY TARGET recovered index=15 q=0\n'
                             'BODY EFFORT recovered index=20 kp=40 kd=1\n')
            watcher.poll()
            self.assertEqual(watcher.warnings(), [])

    def test_recovery_warning_is_local_and_clears_on_explicit_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'bridge.log'
            watcher = BridgeLogWatcher(path, 80)
            path.write_text('DEX3 RECOVERY warning joint=6 reason=measured position did not recover in 3 s\n')
            watcher.poll()
            warnings = watcher.warnings()
            self.assertEqual(len(warnings), 1)
            self.assertEqual(warnings[0][0], 'warning')
            self.assertIn('恢复超时', warnings[0][1])
            self.assertIn('机身控制继续', warnings[0][1])
            with path.open('a') as stream:
                stream.write('DEX3 RECOVERY started joint=6 initial_q=-1.79 target_q=-1.72 kp_min=0.5 kp_max=2 timeout_s=3\n')
            watcher.poll()
            self.assertEqual(watcher.warnings(), [])

    def test_tail_clears_recovered_and_restarted(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'bridge.log'
            watcher = BridgeLogWatcher(path, 80)
            watcher.poll()
            self.assertEqual(watcher.warnings(), [])

            with path.open('w') as stream:
                stream.write('DEX3 HEALTH side=left healthy=false topic=t age_ms=1 reason=no_sample\n'
                             'HEALTH false reason=body_tilt_high lowstate_age_ms=1')
            watcher.poll()
            self.assertEqual(len(watcher.warnings()), 1)  # Partial line waits for its newline.
            with path.open('a') as stream:
                stream.write(' remote_age_ms=1\n')
            watcher.poll()
            self.assertEqual(len(watcher.warnings()), 2)
            with path.open('a') as stream:
                stream.write('DEX3 HEALTH side=left healthy=true topic=t age_ms=1\n')
            watcher.poll()
            self.assertEqual([text for _, text in watcher.warnings()], ['机身倾角过大（IMU roll/pitch 超限）'])
            with path.open('a') as stream:
                stream.write('BRIDGE mode=OBSERVE_ONLY config=1\n')
            watcher.poll()
            self.assertEqual(watcher.warnings(), [])


    def test_hand_transport_warning_does_not_clear_pause_until_manual_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'bridge.log'
            watcher = BridgeLogWatcher(path, 80)
            path.write_text('DEX3 TRANSPORT side=left operation=publish warning reason=injected SDK exception\n'
                            'DEX3 CONTROL warning reason=hand command transport failed; body control continues\n')
            watcher.poll()
            self.assertEqual(len(watcher.warnings()), 2)
            self.assertTrue(all(level == 'warning' and '机身控制继续' in text
                                for level, text in watcher.warnings()))
            with path.open('a') as stream:
                stream.write('DEX3 TRANSPORT side=left operation=publish recovered\n')
            watcher.poll()
            self.assertEqual(len(watcher.warnings()), 1)
            self.assertIn('已暂停', watcher.warnings()[0][1])
            with path.open('a') as stream:
                stream.write('DEX3 CONTROL resumed\n')
            watcher.poll()
            self.assertEqual(watcher.warnings(), [])

    def test_board_holds_then_expires(self):
        now = [0.]
        board = WarningBoard(hold_s=5, clock=lambda: now[0])
        self.assertTrue(board.update([('error', 'a')])[0][2])
        now[0] = 2
        rows = board.update([])
        self.assertEqual(len(rows), 1)
        self.assertFalse(rows[0][2])
        now[0] = 6
        self.assertEqual(board.update([]), [])

    def test_changing_measurements_replace_same_joint_warning(self):
        now = [0.]
        board = WarningBoard(clock=lambda: now[0])
        board.update([('error', 'left_knee_joint 位置超限：1.000')])
        now[0] = .2
        rows = board.update([('error', 'left_knee_joint 位置超限：1.001'),
                             ('error', 'right_knee_joint 位置超限：2.000')])
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row[2] for row in rows))
        self.assertTrue(any('1.001' in row[1] for row in rows))
        self.assertFalse(any(row[2] for row in board.update([])))

    def test_log_replacement_and_malformed_line_do_not_break_monitor(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'bridge.log'
            watcher = BridgeLogWatcher(path, 80)
            path.write_text('HEALTH false reason=body_tilt_high lowstate_age_ms=1\n')
            watcher.poll()
            self.assertEqual(len(watcher.warnings()), 1)
            replacement = Path(directory) / 'new.log'
            replacement.write_text('DEX3 HEALTH side=left healthy=false reason=motor_temperature index=90 c=82\n'
                                   'HEALTH true lowstate_age_ms=1\n')
            replacement.replace(path)
            watcher.poll()
            self.assertEqual(watcher.warnings(), [])


if __name__ == '__main__':
    unittest.main()
