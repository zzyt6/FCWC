"""Dex3 JSON -> bridge startup arguments, without opening network or DDS."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import tempfile
import unittest
from unittest.mock import Mock, patch

from core.robot import HAND_PD_PATH, RobotConfig, load_hand_pd
from core.backends import Bridge


class HandPdTests(unittest.TestCase):
    def test_default_and_custom_config_are_loaded_once_per_run(self):
        gains = load_hand_pd(HAND_PD_PATH)
        self.assertEqual(gains['kp'], 1.5)
        self.assertAlmostEqual(gains['kd'], .2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'pd.json'
            path.write_text(json.dumps(dict(schema_version=1, kp=1.2, kd=.15)))
            config = RobotConfig(hand_pd_config=path)
            self.assertAlmostEqual(config.hand_pd['kp'], 1.2)
            self.assertAlmostEqual(config.hand_pd['kd'], .15)
            path.write_text(json.dumps(dict(schema_version=1, kp=1.4, kd=.18)))
            self.assertAlmostEqual(config.hand_pd['kp'], 1.2)
            self.assertAlmostEqual(RobotConfig(hand_pd_config=path).hand_pd['kp'], 1.4)

    def test_invalid_or_missing_config_does_not_fall_back(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'pd.json'
            for key, value in [('kp', 0), ('kp', -1), ('kd', -.1), ('kp', True), ('kd', '.2'),
                               ('kp', float('nan')), ('kd', float('inf')), ('kp', 1e40),
                               ('kp', 1e-100), ('schema_version', True), ('schema_version', 2)]:
                data = dict(schema_version=1, kp=1.5, kd=.2); data[key] = value
                path.write_text(json.dumps(data))
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    load_hand_pd(path)
            for data in [[], {}, dict(schema_version=1, kp=1.5),
                         dict(schema_version=1, kp=1.5, kd=.2, kdd=.3)]:
                path.write_text(json.dumps(data))
                with self.assertRaises(ValueError): load_hand_pd(path)
            path.write_text('{')
            with self.assertRaises(ValueError): RobotConfig(hand_pd_config=path)
            with self.assertRaises(FileNotFoundError): RobotConfig(hand_pd_config=path.with_name('missing.json'))

    def test_bridge_passes_loaded_gains_on_acquire_and_reacquire(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'pd.json'
            path.write_text(json.dumps(dict(schema_version=1, kp=1.25, kd=.125)))
            config = RobotConfig(hand_pd_config=path)
            with patch('core.backends.socket.socket') as socket_factory, patch('core.backends.subprocess.Popen') as spawn:
                socket_factory.return_value.recvfrom.side_effect = BlockingIOError
                spawn.return_value.poll.return_value = 0
                bridge = Bridge(config, 'control')
                for armed in (False, True, False, True):
                    bridge.start(armed=armed)
                    args = spawn.call_args.args[0]
                    self.assertEqual(args[args.index('--dex3-pd')+1], '1.25,0.125')
                    self.assertEqual('--control' in args, armed)
                    self.assertEqual('--release-mode' in args, armed)
                bridge.close()


if __name__ == '__main__':
    unittest.main()
