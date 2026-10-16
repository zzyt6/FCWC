"""Validate additional estimator outputs against the graph and original actions."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import unittest

import numpy as np
import onnx
from onnx import numpy_helper

from core.policy import Policy
from core.robot import RobotConfig
from hybrid_deploy.policy import Policy as OriginalPolicy


class PolicyOutputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = RobotConfig()
        cls.original = OriginalPolicy(cls.config.deploy, cls.config.deployment)
        cls.policy = Policy(cls.config.deploy, cls.config.deployment)

    def test_actions_match_original_sessions_and_export_vectors(self):
        for name, obs_key, key in [('balance', 'inputs', 'outputs'), ('homie', 'observations', 'actions')]:
            with np.load(self.config.deploy / 'models' / name / 'verification_vectors.npz') as vectors:
                observations, expected_actions = vectors[obs_key], vectors[key]
            for obs, expected in zip(observations, expected_actions):
                actual = self.policy.infer(name, obs[None])
                baseline = self.original.infer(name, obs[None])
                np.testing.assert_allclose(actual, baseline, atol=1e-6, rtol=1e-6)
                np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)

    def test_estimator_matches_independent_encoder_and_normalization(self):
        model = onnx.load(str(self.config.policy_paths['homie']))
        weights = {v.name: numpy_helper.to_array(v) for v in model.graph.initializer}
        vectors = np.load(self.config.deploy / 'models/homie/verification_vectors.npz')
        for obs in vectors['observations'][:16]:
            self.policy.infer('homie', obs[None])
            x = obs.copy()
            for layer in (0, 2, 4):
                x = x @ weights[f'encoder.{layer}.weight'].T + weights[f'encoder.{layer}.bias']
                if layer != 4:
                    negative = x < 0
                    x[negative] = np.expm1(x[negative])
            estimate = self.policy.homie_estimator
            np.testing.assert_allclose(estimate['velocity_raw'], x[:3], atol=2e-5, rtol=2e-5)
            np.testing.assert_allclose(estimate['latent_raw'], x[3:], atol=2e-5, rtol=2e-5)
            expected = x[3:] / max(np.linalg.norm(x[3:]), 1e-12)
            np.testing.assert_allclose(estimate['latent_normalized'], expected, atol=2e-5, rtol=2e-5)
        with self.assertRaises(ValueError):
            self.policy.infer('homie', np.full((1, 456), np.nan, dtype=np.float32))
        self.assertIsNone(self.policy.homie_estimator)


if __name__ == '__main__':
    unittest.main(verbosity=2)
