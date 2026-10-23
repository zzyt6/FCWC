"""Push-cart data episodes: recorder time window and Controller -> CWC/Data-style folders.

Uses temporary directories, the pipe simulator of test_pose_integration and no
robot, DDS or camera.
"""
import csv
import json
import math
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT.parent / 'G1_high_policy_deploy/robot')]
from core.controller import Controller
from core.data_episode import POST_S, PRE_S, EpisodeRecorder, decode, decode_ack, encode
from core.pushcart_data import HAND_COLUMNS, JOINT_COLUMNS, MOTOR_COLUMNS, POLICY_COLUMNS, PushcartDataRecorder
from core.robot import RobotConfig
from hybrid_deploy import protocol as p
from test_pose_integration import HARNESS, PipeBridge
from pushcart_v81.carmer_io import CarmerYaw
from pushcart_v81.viser_policy import ViserPushPolicy


class Collect(EpisodeRecorder):
    def open_files(self, directory):
        Path(directory).mkdir(parents=True, exist_ok=True)
        self.rows = []

    def write(self, t, sample):
        self.rows.append(t)

    def close_files(self):
        return dict(collected=len(self.rows))


def rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


class EpisodeWindowTests(unittest.TestCase):
    def test_messages_round_trip(self):
        message = decode(encode(7, '/tmp/x', 1.5, math.nan, .5, .25))
        self.assertEqual((message['seq'], message['directory'], message['start'], message['stop'],
                          message['pre'], message['post']), (7, '/tmp/x', 1.5, math.inf, .5, .25))
        self.assertIsNone(decode(encode(8, None))['directory'])
        self.assertIsNone(decode(b'PUSHCART_DATA 1 1 relative 0 nan 1 1'))
        with self.assertRaises(ValueError):
            encode(1, '/tmp/with space')

    def test_preroll_window_and_shortened_post_roll(self):
        with tempfile.TemporaryDirectory(prefix='pushcart_data_window_') as directory:
            recorder = Collect('test', 0)
            port = recorder.sock.getsockname()[1]
            sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sender.bind(('127.0.0.1', 0))

            def control(seq, now, **window):
                sender.sendto(encode(seq, window.get('directory'), window.get('start', math.nan),
                                     window.get('stop', math.nan), PRE_S, window.get('post', POST_S)),
                              ('127.0.0.1', port))
                time.sleep(.01)
                recorder.service(now)

            try:
                for t in range(0, 50):
                    recorder.feed(t / 10, None)
                control(1, 5., directory=directory, start=5.)
                ack = decode_ack(sender.recv(512))
                self.assertEqual((ack['name'], ack['directory']), ('test', directory))
                # Pre-roll: exactly the buffered samples from start - 1 s.
                self.assertEqual(recorder.rows[0], 4.)
                for t in range(50, 72):
                    recorder.feed(t / 10, None)
                control(2, 7.2, directory=directory, start=5., stop=7.)
                for t in range(72, 120):
                    recorder.feed(t / 10, None)
                self.assertEqual(recorder.rows[-1], 8.)  # stop + 1 s post-roll
                control(3, 12.)
                self.assertIsNone(recorder.episode)
                summary = json.loads((Path(directory) / 'test_summary.json').read_text())
                self.assertEqual((summary['close_reason'], summary['first_t'], summary['last_t']),
                                 ('stopped', 4., 8.))
                self.assertEqual(summary['rows'], summary['collected'])
                # A program exit announces stop with post = 0.
                control(4, 20., directory=directory + '/b', start=20.)
                recorder.feed(20.5, None)
                control(5, 21., directory=directory + '/b', start=20., stop=21., post=0.)
                recorder.feed(21.2, None)
                self.assertEqual(recorder.rows, [20.5])
            finally:
                recorder.close()
                sender.close()


class ControllerEpisodeTests(unittest.TestCase):
    def test_enable_disable_writes_one_episode_folder(self):
        config = RobotConfig()
        with tempfile.TemporaryDirectory(prefix='g1_pushcart_data_') as directory:
            source, binary = Path(directory)/'core.cpp', Path(directory)/'core'
            source.write_text(HARNESS)
            subprocess.run([sys.executable, '-B', str(config.deploy/'tools/generate_config.py'),
                            str(config.deploy/'config/deploy.json'), str(Path(directory)/'generated_config.hpp')],
                           check=True)
            subprocess.run(['g++', '-std=c++17', '-O2', '-I'+str(config.deploy/'cpp'),
                            '-I'+directory, str(source), '-o', str(binary)], check=True)
            bridge = PipeBridge(binary, config)
            camera = CarmerYaw(0, tag_ids=[0, 1])
            high = ViserPushPolicy(config, ROOT.parent/'G1_high_policy_deploy/robot/config/viser_v81.json', camera=camera)
            control = Controller(config, bridge, 'sim', high_policy=high, clock=bridge.now)
            data_root = Path(directory) / 'Data'
            control.data_recorder = PushcartDataRecorder('web', mode='sim', interface='lo', config=config,
                                                         root=data_root, clock=bridge.now)
            closed = False
            try:
                snapshots = {}
                def tick(kind=None, velocity=None):
                    now = bridge.now()
                    camera.accept(json.dumps(dict(schema='carmer_cart_yaw_v1', captured_monotonic=now-.001,
                                                  valid=True, reference_frame='pelvis_level', yaw_rad=.1,
                                                  tag_ids=[0, 1])), now)
                    if kind:
                        control.emit(kind, 1, velocity)
                    control.step()
                    data = control.snapshot()
                    if data.get('policy_inference_monotonic') is not None:
                        snapshots['%.6f' % data['policy_inference_monotonic']] = data
                    return data

                def advance_until(check, limit=400):
                    for _ in range(limit):
                        tick()
                        if check():
                            return
                    self.fail('State transition did not complete')

                tick();tick('claim');tick('prepare')
                advance_until(lambda: control.stage=='HOLDING' and control.observations.frames>=6)
                tick('policy')
                advance_until(lambda: control.stage=='POLICY')
                tick('hand_toggle')
                advance_until(lambda: bool(bridge.state.hand_flags&16) and not bridge.state.hand_flags&8)
                tick()
                tick('pushcart_enable')
                self.assertTrue(high.enabled, control.reason)
                start = bridge.now()
                advance_until(lambda: bridge.state.upper_mode==1 and bridge.state.upper_gain_blend==1.)
                for _ in range(40):
                    tick('move', [.2, 0, .1])
                tick('zero')
                advance_until(lambda: not control.zeroing and not any(control.snapshot()['velocity']))
                tick('pushcart_disable')
                self.assertFalse(high.enabled)
                stop = bridge.now()
                advance_until(lambda: bridge.now() > stop + POST_S + 1.2, limit=1000)
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline and not control.data_recorder.last_status.startswith('上次记录'):
                    tick()
                    time.sleep(.01)
                control.close()
                closed = True
                episodes = list(data_root.iterdir())
                self.assertEqual(len(episodes), 1)
                episode = episodes[0]
                self.assertTrue(episode.name.endswith('_web'))
                meta = json.loads((episode / 'meta.json').read_text())
                self.assertEqual(meta['status'], 'complete')
                self.assertEqual(meta['input_mode'], 'web')
                t0, t1 = meta['episode']['start_monotonic'], meta['episode']['stop_monotonic']
                self.assertAlmostEqual(t0, start, delta=.05)
                self.assertAlmostEqual(t1, stop, delta=.05)
                policy = rows(episode / 'policy.csv')
                self.assertEqual(tuple(policy[0]), POLICY_COLUMNS)
                times = [float(row['t']) for row in policy]
                self.assertEqual(times, sorted(times))
                self.assertGreaterEqual(times[0], t0 - PRE_S - 1e-6)
                self.assertLess(times[0], t0 - PRE_S + .05)
                self.assertLessEqual(times[-1], t1 + POST_S + 1e-6)
                self.assertGreater(times[-1], t1 + POST_S - .05)
                enabled = [row['pushcart_enabled'] for row in policy]
                self.assertEqual(enabled[0], '0')
                self.assertEqual(enabled[-1], '0')
                self.assertIn('1', enabled)
                active = [row for row in policy if row['pushcart_enabled'] == '1']
                # Times are written with microsecond resolution.
                self.assertTrue(all(t0 - 1e-6 <= float(row['t']) < t1 for row in active))
                self.assertTrue(any(row['raw_task_vx'] == '0.2' for row in active))
                self.assertTrue(all(row['camera_yaw_rad'] for row in active))
                self.assertTrue(all(row['state_received_t'] and float(row['state_received_t']) <= float(row['t'])
                                    for row in active))
                joints = rows(episode / 'joint_command.csv')
                self.assertEqual(tuple(joints[0]), JOINT_COLUMNS)
                self.assertEqual([row['t'] for row in joints], [row['t'] for row in policy])
                motor = rows(episode / 'motor_state.csv')
                self.assertEqual(tuple(motor[0]), MOTOR_COLUMNS)
                self.assertEqual([row['t'] for row in motor], [row['t'] for row in policy])
                for row, policy_row in zip(motor, policy):
                    self.assertEqual(row['state_received_t'], policy_row['state_received_t'])
                    sample = snapshots[row['inference_done_t']]
                    np.testing.assert_allclose([float(row[key]) for key in MOTOR_COLUMNS if key.startswith('q_')],
                                               sample['state']['q'], rtol=1e-6, atol=1e-7)
                    self.assertEqual(row['tau_est_left_hip_pitch_joint'], '')
                for index, side in enumerate(('left', 'right')):
                    hand = rows(episode / ('hand_%s_state.csv' % side))
                    self.assertEqual(tuple(hand[0]), HAND_COLUMNS[side])
                    self.assertEqual([row['t'] for row in hand], [row['t'] for row in policy])
                    sample = snapshots[hand[-1]['inference_done_t']]
                    np.testing.assert_allclose([float(hand[-1][key]) for key in HAND_COLUMNS[side]
                                                if key.startswith('q_')],
                                               sample['dex3']['q'][index * 7:(index + 1) * 7], rtol=1e-6, atol=1e-7)
                    self.assertEqual(hand[-1]['power_a'], '')
                self.assertEqual(meta['schema'], 'pushcart_data_v2')
                self.assertEqual(meta['streams']['motor']['rows'], len(policy))
                self.assertTrue(meta['streams']['motor']['files_closed'])
                summary = json.loads((episode / 'motor_summary.json').read_text())
                self.assertTrue(all(item['rows'] == len(policy) for item in summary['files'].values()))
            finally:
                if not closed:
                    control.close()

    def test_only_inference_ticks_are_written_without_starting_a_motor_process(self):
        with tempfile.TemporaryDirectory(prefix='inference_motor_data_') as directory:
            with patch('core.pushcart_data.subprocess.Popen', side_effect=AssertionError('Unexpected child process')):
                recorder = PushcartDataRecorder('nav', mode='control', interface='lo', root=directory)
            # Distinct values make field order and optional feedback visible.
            telemetry = p.MOTOR_TELEMETRY.pack(
                p.MOTOR_TELEMETRY_MAGIC, p.MOTOR_TELEMETRY_VERSION, 1, 6789, 0,
                *([1] * 29), *([2.] * 29), *([3.] * 29), *([48.] * 29), 0., 0., 9.8,
                1, 0, 9, *([4.] * 7), *([5.] * 7), 24., 1.5, 5., 3.3, *([6.] * 108),
                0, -1, 0, *([0.] * 126))
            feedback = SimpleNamespace(hand_received_ns=(123000000000, 0), mode_machine=1,
                                       rpy=np.zeros(3), motor_telemetry=telemetry)
            now = time.monotonic()
            data = dict(high_policy=dict(enabled=True), policy_inference_monotonic=now,
                        state=dict(received=123., tick=(42, 10), q=np.arange(29), dq=np.zeros(29)),
                        dex3=dict(q=np.arange(14), dq=np.zeros(14), left_healthy=True, right_healthy=False))
            try:
                recorder.update(data, now, feedback)
                deadline = time.monotonic() + 3
                while recorder.episode is None and time.monotonic() < deadline:
                    time.sleep(.01)
                self.assertIsNotNone(recorder.episode)
                # A tick without inference must not invent a new motor sample.
                recorder.update(dict(data, policy_inference_monotonic=None), now + .02, feedback)
                recorder.update(dict(data, policy_inference_monotonic=now + .04), now + .04, feedback)
                deadline = time.monotonic() + 3
                while recorder.episode['rows'] < 2 and time.monotonic() < deadline:
                    time.sleep(.01)
                self.assertEqual(recorder.episode['rows'], 2)
            finally:
                recorder.close()
            episode, = Path(directory).iterdir()
            motor = rows(episode / 'motor_state.csv')
            self.assertEqual([row['state_received_t'] for row in motor], ['123.000000'] * 2)
            self.assertEqual([row['t'] for row in motor], ['%.6f' % now, '%.6f' % (now + .04)])
            self.assertEqual(motor[0]['tick'], '6789')
            self.assertEqual(motor[0]['ddq_left_hip_pitch_joint'], '2')
            self.assertEqual(motor[0]['tau_est_left_hip_pitch_joint'], '3')
            self.assertEqual(motor[0]['vol_left_hip_pitch_joint'], '48')
            left, right = (rows(episode / ('hand_%s_state.csv' % side)) for side in ('left', 'right'))
            self.assertEqual(left[0]['hand_received_t'], '123.000000')
            self.assertEqual(left[0]['topic'], 'rt/lf/dex3/left/state')
            self.assertEqual(left[0]['tau_est_left_hand_thumb_0_joint'], '5')
            self.assertEqual(left[0]['power_a'], '1.5')
            self.assertEqual(left[0]['press_8_11'], '6')
            self.assertEqual(right[0]['hand_received_t'], '')
            self.assertEqual(right[0]['q_right_hand_thumb_0_joint'], '')
            self.assertEqual(json.loads((episode / 'meta.json').read_text())['status'], 'complete')


if __name__ == '__main__':
    unittest.main()
