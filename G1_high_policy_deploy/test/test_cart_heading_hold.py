"""Numerical contracts for the cart heading loop before student inference."""
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'robot'))
from pushcart_v81.viser_policy import CartHeadingHold


class CartHeadingHoldTests(unittest.TestCase):
    def test_translations_hold_cart_heading_and_preserve_xy(self):
        for vx, vy in ((.3, 0.), (-.3, 0.), (0., .2), (0., -.2)):
            with self.subTest(vx=vx, vy=vy):
                hold = CartHeadingHold()
                self.assertEqual(hold((vx, vy, 0.), .2, .1, .02), (vx, vy, 0.))
                # Pelvis turns but the cart stays fixed: no correction.
                out = hold((vx, vy, 0.), .25, .05, .02)
                self.assertAlmostEqual(out[2], 0.)
                # Then the cart drifts left: request rightward correction.
                out = hold((vx, vy, 0.), .25, .1, .02)
                self.assertEqual(out[:2], (vx, vy))
                self.assertAlmostEqual(out[2], -.05)

    def test_turn_reference_integrates_input_wz_at_variable_intervals(self):
        hold = CartHeadingHold()
        expected = .2
        for wz, dt in ((.3, .013), (.3, .037), (-.3, .009), (-.3, .041)):
            expected += wz * dt
            out = hold((0., 0., wz), .1, .1, dt)
            self.assertAlmostEqual(hold.ref, expected)
            self.assertAlmostEqual(hold.err, expected - .2)
            self.assertAlmostEqual(out[2], wz + expected - .2)

    def test_wrap_crossing_uses_shortest_error(self):
        for sign in (-1, 1):
            hold = CartHeadingHold()
            hold((.3, 0., 0.), sign * (math.pi - .02), 0., .02)
            out = hold((.3, 0., 0.), -sign * (math.pi - .02), 0., .02)
            self.assertAlmostEqual(out[2], -sign * .04)

    def test_error_drags_reference_and_correction_and_task_wz_are_bounded(self):
        for sign in (-1, 1):
            hold = CartHeadingHold()
            hold((.3, 0., 0.), 0., 0., .02)
            out = hold((.3, 0., 0.), sign * .7, 0., .02)
            self.assertAlmostEqual(hold.err, -sign * .3)
            self.assertAlmostEqual(hold.ref, sign * .4)
            self.assertAlmostEqual(out[2], -sign * .15)
            out = hold((0., 0., -sign * .6), sign * .7, 0., .02)
            self.assertEqual(out[2], -sign * .6)

    def test_zero_resets_and_next_motion_locks_current_cart_heading(self):
        hold = CartHeadingHold()
        hold((.3, 0., 0.), 0., 0., .02)
        hold((.3, 0., 0.), .1, 0., .02)
        for zero in ((0., 0., 0.), (1e-8, -1e-8, 1e-8)):
            self.assertEqual(hold(zero, .2, .3, .02), zero)
            self.assertIsNone(hold.ref)
            self.assertEqual((hold.err, hold.corr), (0., 0.))
        self.assertEqual(hold((.3, 0., 0.), .2, .3, .02), (.3, 0., 0.))
        self.assertAlmostEqual(hold.ref, .5)

    def test_invalid_measurement_or_time_is_rejected(self):
        for imu, cam, dt in ((float('nan'), 0., .02), (0., float('inf'), .02),
                             (0., 0., -.01), (0., 0., float('nan'))):
            with self.assertRaises(ValueError):
                CartHeadingHold()((.3, 0., 0.), imu, cam, dt)


if __name__ == '__main__':
    unittest.main()
