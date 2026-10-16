"""Named calibration mapping and rejection of unsafe optional zero targets."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.robot import HAND_JOINTS, HAND_ZERO_PATH, RobotConfig, load_hand_zero


class HandCalibrationTests(unittest.TestCase):
    def test_source_calibration_and_mapping_ignore_json_key_order(self):
        config = RobotConfig()
        self.assertEqual(config.hand_zero_error, '')
        data = json.loads(HAND_ZERO_PATH.read_text())
        expected = np.asarray([[data[f'{side}_raw_zero_rad'][f'{side}_hand_{joint}_joint']
                                for joint in HAND_JOINTS] for side in ('left', 'right')])
        np.testing.assert_array_equal(config.hand_zero, expected)
        np.testing.assert_allclose(config.hand_grasp_compensated, config.hand_grasp + expected)
        self.assertEqual(config.hand_grasp_compensated[1, 0], expected[1, 0])
        self.assertEqual(config.hand_grasp[1, 0], 0.)
        data = json.loads(HAND_ZERO_PATH.read_text())
        # Deliberately distinct values and reversed insertion order on both hands.
        for side in ('left', 'right'):
            data[f'{side}_raw_zero_rad'] = {
                f'{side}_hand_{joint}_joint': .01*(index+1)
                for index, joint in reversed(list(enumerate(HAND_JOINTS)))}
        with tempfile.TemporaryDirectory(prefix='viser_hand_zero_') as directory:
            path = Path(directory)/'calibration.json'
            path.write_text(json.dumps(data))
            result = load_hand_zero(path, np.full((2, 7), -1.), np.ones((2, 7)))
            np.testing.assert_allclose(result, np.tile(np.arange(1, 8)*.01, (2, 1)))

    def test_invalid_calibration_never_becomes_a_motor_target(self):
        data = json.loads(HAND_ZERO_PATH.read_text())
        with tempfile.TemporaryDirectory(prefix='viser_hand_zero_') as directory:
            path = Path(directory)/'calibration.json'
            for value in (float('nan'), float('inf'), 2., True, '0.2'):
                data['right_raw_zero_rad']['right_hand_thumb_0_joint'] = value
                path.write_text(json.dumps(data))
                with self.subTest(value=value), self.assertRaises(ValueError):
                    load_hand_zero(path, np.full((2, 7), -1.05), np.full((2, 7), 1.05))
            del data['left_raw_zero_rad']['left_hand_index_0_joint']
            path.write_text(json.dumps(data))
            config = RobotConfig(hand_zero_calibration=path)
            self.assertIsNone(config.hand_zero)
            self.assertIn('7 个关节', config.hand_zero_error)
            self.assertEqual(config.hand_grasp.shape, (2, 7))
            config = RobotConfig(hand_zero_calibration=path.with_name('missing.json'))
            self.assertIsNone(config.hand_zero)
            self.assertTrue(config.hand_zero_error)
            self.assertIsNone(config.hand_grasp_compensated)

    def test_compensated_grasp_sum_is_validated_separately(self):
        data = json.loads(HAND_ZERO_PATH.read_text())
        # Valid raw zero, but grasp + offset exceeds this finger's limit.
        data['right_raw_zero_rad']['right_hand_middle_0_joint'] = 1.
        with tempfile.TemporaryDirectory(prefix='viser_hand_zero_') as directory:
            path = Path(directory)/'calibration.json'
            path.write_text(json.dumps(data))
            config = RobotConfig(hand_zero_calibration=path)
            self.assertIsNotNone(config.hand_zero)
            self.assertEqual(config.hand_zero_error, '')
            self.assertIsNone(config.hand_grasp_compensated)
            self.assertIn('限位', config.hand_grasp_compensation_error)


if __name__ == '__main__':
    unittest.main()
