"""Real model bundles and synthetic feedback; no robot connections."""
import json
from pathlib import Path
import socket
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.controller import Controller
from core.model_sets import ModelSets
from core.pushcart import CONFIG
from core.recording import SessionRecorder
from core.robot import RobotConfig
from hybrid_deploy import protocol as p
from test_control import FakeBridge


class ModelSetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='viser_model_sets_')
        self.addCleanup(self.temp.cleanup)
        settings = json.loads(CONFIG.read_text())
        settings['models_dir'] = str((CONFIG.parent / settings['models_dir']).resolve())
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.bind(('127.0.0.1', 0))
            settings['camera']['udp_port'] = probe.getsockname()[1]
        path = Path(self.temp.name) / 'pushcart.json'
        path.write_text(json.dumps(settings))
        self.config = RobotConfig()
        self.bank = ModelSets(self.config, 'sim', 'lo', path)
        self.addCleanup(self.bank.close)
        self.now = 100.
        self.bridge = FakeBridge(self.config)
        self.bridge.close = Mock()
        self.control = Controller(self.config, self.bridge, 'sim', clock=lambda: self.now,
                                  model_sets=self.bank)
        self.addCleanup(self.control.close)
        self.step()

    def step(self):
        self.now += .02
        self.bridge.state.sequence += 1
        self.bridge.state.received_ns = int((self.now - .000001) * 1e9)
        self.control.step()

    def event(self, kind, client=1, **payload):
        self.control.emit(kind, client, **payload)
        self.step()

    def test_both_complete_sets_are_warm_and_match_export_vectors(self):
        expected = {
            'v81': '676765c20e3df7c838cb3d1cdf054cd0376158dab41e176949e9a4cdfd049f0c',
            'v82': '301cabafabc9d0a1fb3eb6bfd467da97e1b2be225300d2f3ff65107982022de1',
        }
        for key, entry in self.bank.entries.items():
            with self.subTest(key=key):
                policy, high = entry['policy'], entry['high']
                self.assertEqual(policy.source_sha256['homie'], expected[key])
                self.assertEqual(high.cfg['release'], {'v81': 'V8.1', 'v82': 'V8.2'}[key])
                self.assertTrue(policy.warmed and high.model.warmed and high.homie_model.warmed)
                self.assertIs(policy.pushcart_homie, high.homie_model)
                self.assertFalse(high.enabled)
                file = entry['config'].policy_paths['homie'].parent / 'verification_vectors.npz'
                with np.load(file, allow_pickle=False) as data:
                    actual = [policy.infer('homie', obs.reshape(1, -1)) for obs in data['observations']]
                    np.testing.assert_allclose(actual, data['actions'], atol=1e-5, rtol=1e-5)
        self.assertIs(self.bank.entries['v81']['high'].camera, self.bank.entries['v82']['high'].camera)
        self.assertEqual(self.config.motion_settings['run_mode'], 'active')

    def test_selection_switches_entire_set_without_acquiring_or_moving(self):
        self.assertEqual(self.control.model_set, 'v81')
        self.event('claim')
        for key in ('v82', 'v81', 'v82'):
            self.event('model_set', model_set=key)
            entry = self.bank.entries[key]
            self.assertIs(self.control.policy, entry['policy'])
            self.assertIs(self.control.high_policy, entry['high'])
            self.assertIs(self.config.contracts, entry['config'].contracts)
            self.assertEqual(self.control.snapshot()['model_set'], key)
            self.assertFalse(self.control.high_policy.enabled)
            np.testing.assert_array_equal(self.control.snapshot()['velocity'], 0)
        self.assertFalse(self.bridge.armed)
        self.assertFalse(self.bridge.writes)
        self.assertFalse(self.bridge.starts)
        self.assertFalse(self.control.hand_live)

    def test_owner_validation_and_prepare_request_lock(self):
        self.event('claim')
        for client, key in ((2, 'v82'), (1, 'unknown'), (1, None)):
            self.event('model_set', client=client, model_set=key)
            self.assertEqual(self.control.model_set, 'v81')
        self.event('acquire')
        self.step()
        self.event('model_set', model_set='v82')
        self.assertEqual(self.control.model_set, 'v82')
        # Both queued before the next feedback tick: the prepare request must lock immediately.
        self.control.emit('prepare', 1)
        self.now += .001
        self.control.emit('model_set', 1, model_set='v81')
        self.step()
        self.assertEqual(self.control.request, p.PREPARE)
        self.assertEqual(self.control.model_set, 'v82')
        self.assertFalse(self.control.snapshot()['model_set_selectable'])
        for stage in (p.RAMPING, p.HOLDING, p.POLICY):
            self.bridge.state.stage = stage
            self.event('model_set', model_set='v81')
            self.assertEqual(self.control.model_set, 'v82')

    def test_stale_and_stopped_selections_are_rejected(self):
        self.event('claim')
        self.control.emit('model_set', 1, model_set='v82', command_time=self.now - .5)
        self.step()
        self.assertEqual(self.control.model_set, 'v81')
        self.control.emit('model_set', 1, model_set='v82')
        self.control.emit('stop', 1)
        self.step()
        self.assertEqual(self.control.model_set, 'v81')
        self.assertFalse(self.control.snapshot()['model_set_selectable'])

    def test_recording_keeps_both_hashes_and_initial_selection(self):
        recorder = SessionRecorder(self.config, {'mode': 'sim'}, directory=self.temp.name)
        try:
            self.event('claim')
            self.event('model_set', model_set='v82')
            metadata = recorder.metadata
            self.assertEqual(metadata['initial_model_set'], 'v81')
            self.assertEqual(metadata['high_policy']['model_release'], 'V8.1')
            self.assertEqual(metadata['model_sets']['v82']['high_policy']['model_release'], 'V8.2')
            self.assertEqual(metadata['model_sets']['v82']['model_sha256'], self.control.policy.source_sha256)
        finally:
            recorder.close()

    def test_mixed_pair_is_rejected(self):
        from pushcart_v81.homie_adapter import PushcartHomie
        with self.assertRaisesRegex(ValueError, 'base weight mismatch'):
            PushcartHomie(self.bank.entries['v82']['high'].homie_model.path,
                          self.bank.entries['v81']['config'].policy_paths['homie'])

    def test_unavailable_students_still_allow_normal_walking(self):
        from contextlib import redirect_stdout
        from io import StringIO
        with patch('pushcart_v81.viser_policy.ViserPushPolicy', side_effect=ValueError('student unavailable')):
            with redirect_stdout(StringIO()):
                bank = ModelSets(self.config, 'sim', 'lo')
        try:
            for key in bank.entries:
                policy, high = bank.activate(self.config, key)
                self.assertTrue(policy.warmed)
                self.assertIsNone(high)
                self.assertIn('student unavailable', self.config.pushcart_error)
                self.assertIsNone(self.config.high_policy_metadata)
        finally:
            bank.close()


if __name__ == '__main__':
    unittest.main()
