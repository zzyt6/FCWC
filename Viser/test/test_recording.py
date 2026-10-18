"""Recording retention and compact robot data regressions."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gzip
import json
import tempfile
import unittest
import numpy as np
from core.recording import SessionRecorder
from core.robot import RobotConfig

class RecordingTests(unittest.TestCase):
    def test_compact_archives_keep_latest_five_runs(self):
        config = RobotConfig()
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for i in range(7):
                recorder = SessionRecorder(config, {'mode': 'control' if i % 2 else 'observe'}, directory)
                recorder.record('telemetry', {'state': {'q': np.arange(29)}, 'run': i, 'model_actions': {'balance': np.arange(12), 'homie': np.arange(12)},
                                              'homie_estimator_source': 'pushcart_finetuned',
                                              'homie_estimator': {'velocity_raw': np.array([.1, -.2, .3]),
                                                                  'velocity_original_raw': np.array([.2, -.1, .4]),
                                                                  'latent_raw': np.ones(32),
                                                                  'latent_normalized': np.ones(32)/np.sqrt(32)}})
                recorder.record('events', {'type': 'damping'})
                summary = recorder.close()
                paths.append(recorder.path)
                self.assertEqual(summary['records'], {'telemetry': 1, 'events': 1})
                self.assertEqual(summary['recorder_error'], '')
            self.assertEqual(sorted(Path(directory).iterdir()), paths[-5:])
            for i, path in enumerate(paths[-5:], start=2):
                self.assertEqual({f.name for f in path.iterdir()},
                                 {'metadata.json', 'telemetry.jsonl.gz', 'events.jsonl.gz', 'summary.json'})
                with gzip.open(path / 'telemetry.jsonl.gz', 'rt') as stream:
                    row = json.loads(stream.readline())
                self.assertEqual(row['run'], i)
                self.assertEqual(row['model_actions']['balance'], list(range(12)))
                self.assertEqual(row['model_actions']['homie'], list(range(12)))
                self.assertEqual(row['homie_estimator']['velocity_raw'], [.1, -.2, .3])
                self.assertEqual(row['homie_estimator']['velocity_original_raw'], [.2, -.1, .4])
                self.assertEqual(row['homie_estimator_source'], 'pushcart_finetuned')
                self.assertEqual(len(row['homie_estimator']['latent_normalized']), 32)
                self.assertEqual(row['state']['q'], list(range(29)))
                metadata = json.loads((path / 'metadata.json').read_text())
                self.assertEqual(metadata['motion_settings'], config.motion_settings)
                self.assertNotIn('snapshot_sha256', metadata)
                self.assertEqual(metadata['schema'], 19)
                self.assertEqual(metadata['preparation_poses'],
                                 {key: str(path) for key, path in config.upper_pose_paths.items()})
                self.assertEqual(metadata['dex3_pd']['active'], config.hand_pd)
                self.assertEqual(metadata['dex3_pd']['stopped'], {'kp': 0., 'kd': .1})
                self.assertEqual(metadata['dex3_pd']['config_path'], str(config.hand_pd_path))
                self.assertEqual(len(metadata['joint_comparison']['joint_names']), 41)
                self.assertEqual(metadata['policy_outputs']['model_actions']['balance']['shape'], [12])
                self.assertEqual(len(metadata['policy_outputs']['model_sha256']['homie']), 64)
            empty = SessionRecorder(config, {'mode': 'control'}, directory)
            empty.close()
            self.assertFalse(empty.path.exists())
            self.assertEqual(sorted(Path(directory).iterdir()), paths[-5:])

if __name__ == '__main__':
    unittest.main(verbosity=2)
