"""Numerical contracts for the cart heading loop before student inference."""
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'robot'))
from pushcart_v81.viser_policy import CartHeadingHold, PelvisYawHistory


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



class PelvisYawHistoryTests(unittest.TestCase):
    def test_interpolates_clamps_and_unwraps(self):
        history = PelvisYawHistory()
        self.assertIsNone(history.at(1.))
        for index in range(10):
            history.add(1. + .02*index, .1*index)
        self.assertAlmostEqual(history.at(1.03), .15)
        self.assertAlmostEqual(history.at(.5), 0.)   # Older than the span: oldest sample.
        self.assertAlmostEqual(history.at(9.), .9)   # Newer: latest sample.
        wrap = PelvisYawHistory()
        wrap.add(0., math.pi - .02)
        wrap.add(.02, -math.pi + .02)
        self.assertAlmostEqual(abs(wrap.at(.01)), math.pi)

    def test_gait_sway_with_camera_delay_is_not_a_heading_error(self):
        """Fixed cart, swaying pelvis, camera 60 ms old: no correction when paired at capture."""
        cart = .3
        history, paired, current = PelvisYawHistory(), CartHeadingHold(), CartHeadingHold()
        worst_paired = worst_current = 0.
        for tick in range(200):
            now = tick*.02
            pelvis = .15*math.sin(2*math.pi*1.7*now)
            history.add(now, pelvis)
            captured = max(0., now-.06)
            relative = cart - .15*math.sin(2*math.pi*1.7*captured)
            out_paired = paired((.3, 0., 0.), history.at(captured), relative, .02)
            out_current = current((.3, 0., 0.), pelvis, relative, .02)
            if tick > 10:
                worst_paired = max(worst_paired, abs(out_paired[2]))
                worst_current = max(worst_current, abs(out_current[2]))
        self.assertLess(worst_paired, 1e-9)
        self.assertGreater(worst_current, .09)  # The former pairing: ~0.1 rad/s of false turning.

if __name__ == '__main__':
    unittest.main()
