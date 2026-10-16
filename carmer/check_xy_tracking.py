#!/usr/bin/env python3
"""Measure the current pose filter on synthetic 30 cm/s lateral round trips.

Run in CWC: python -m carmer.check_xy_tracking
No camera or motor is opened. All delays exclude capture/detection/display.
"""
import argparse
import csv
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

if __package__:
    from .apriltag_detector import AdaptivePoseSmoother, TagPoseEstimator, load_intrinsics, INTRINSICS_FILE
else:
    from apriltag_detector import AdaptivePoseSmoother, TagPoseEstimator, load_intrinsics, INTRINSICS_FILE


SPEED = .30
HALF_SPAN = .15
START = .5
LEG_TIME = 2 * HALF_SPAN / SPEED
LEGS = 6
STOP = START + LEGS * LEG_TIME
OUTPUT = Path(__file__).resolve().parent / 'xy_tracking_check'


def trajectory(t):
    elapsed = np.clip(t - START, 0., LEGS * LEG_TIME)
    phase = np.mod(elapsed, 2 * LEG_TIME)
    return np.where(phase <= LEG_TIME, -HALF_SPAN + SPEED * phase,
                    HALF_SPAN - SPEED * (phase - LEG_TIME))


def run_case(estimator, fps, noise_std_m=0., seed=20261012):
    times = np.arange(int(np.ceil((STOP + 1.5) * fps)) + 1) / fps
    target = trajectory(times)
    rng = np.random.default_rng(seed)
    measured = target + rng.normal(0., noise_std_m, size=len(times))
    smoother = AdaptivePoseSmoother(estimator)
    filtered = []
    for t, x in zip(times, measured):
        raw = estimator.pose_from_transform(np.diag([1., -1., -1.]), [x, 0., .76])
        pose = smoother.update(0, raw, float(t))
        if pose is None:
            raise RuntimeError('Unexpected invalid pose')
        filtered.append(pose['tvec_m'][0])
    filtered = np.asarray(filtered)
    error = target - filtered
    motion = (times > START) & (times < STOP)
    legs = []
    for leg in range(LEGS):
        begin = START + leg * LEG_TIME
        direction = 1 if leg % 2 == 0 else -1
        # Exclude the first 0.3 s after starting/reversing and the last 0.1 s.
        stable = (times >= begin + .3) & (times <= begin + LEG_TIME - .1)
        signed_lag = error[stable] * direction
        legs.append({'direction': 'right' if direction == 1 else 'left',
                     'start_s': begin, 'steady_mean_lag_mm': float(np.mean(signed_lag) * 1000),
                     'steady_abs_error_p95_mm': float(np.percentile(np.abs(error[stable]), 95) * 1000),
                     'steady_equivalent_delay_ms': float(np.mean(signed_lag) / SPEED * 1000)})
    turns = []
    delta = np.diff(filtered, prepend=filtered[0])
    for leg in range(1, LEGS):
        turn_time = START + leg * LEG_TIME
        direction = 1 if leg % 2 == 0 else -1
        after = (times > turn_time) & (times < turn_time + .30)
        matching = np.flatnonzero(after & (delta * direction > 1e-10))
        turns.append({'at_s': turn_time, 'new_direction': 'right' if direction == 1 else 'left',
                      'first_output_sample_moving_new_direction_ms': float((times[matching[0]] - turn_time) * 1000) if len(matching) else None,
                      'max_abs_error_next_300ms_mm': float(np.max(np.abs(error[after])) * 1000)})
    stopped = np.flatnonzero(times >= STOP)
    remaining = np.maximum.accumulate(np.abs(error[stopped])[::-1])[::-1]
    settled = stopped[remaining <= .001]
    metrics = {'fps': fps, 'input_white_noise_std_mm': noise_std_m * 1000,
               'max_abs_error_while_moving_mm': float(np.max(np.abs(error[motion])) * 1000),
               'legs': legs, 'reversals': turns,
               'stop': {'at_s': STOP,
                        'first_post_stop_error_mm': float(abs(error[stopped[0]]) * 1000),
                        'within_1mm_and_stays_there_ms': float((times[settled[0]] - STOP) * 1000) if len(settled) else None,
                        'overshoot_beyond_stop_mm': float(max(0., np.max(-HALF_SPAN - filtered[stopped])) * 1000),
                        'last_half_second_output_std_mm': float(np.std(filtered[times >= STOP + 1.]) * 1000)},
               'settings': smoother.settings}
    return metrics, (times, target, measured, filtered)


def main():
    cv2.setNumThreads(1)
    K, D = load_intrinsics(INTRINSICS_FILE)
    estimator = TagPoseEstimator(K, D)
    cases = {}
    traces = {}
    for label, fps, noise in [('nominal', 43.25, 0.), ('nominal_with_1mm_noise', 43.25, .001),
                              ('35fps', 35., 0.), ('50fps', 50., 0.)]:
        cases[label], traces[label] = run_case(estimator, fps, noise)
    # Check the prescribed speed and filter bounds, not an assumed latency value.
    times, target, _, filtered = traces['nominal']
    interior = (times[:-1] > START) & (times[1:] < STOP)
    crossings = np.floor((times[:-1] - START) / LEG_TIME) != np.floor((times[1:] - START) / LEG_TIME)
    velocity = np.diff(target) / np.diff(times)
    np.testing.assert_allclose(np.abs(velocity[interior & ~crossings]), SPEED, atol=1e-12)
    assert np.all(filtered >= -HALF_SPAN - 1e-12) and np.all(filtered <= HALF_SPAN + 1e-12)
    assert cases['nominal']['stop']['within_1mm_and_stays_there_ms'] is not None
    OUTPUT.mkdir(exist_ok=True)
    report = {'created_at_utc': datetime.now(timezone.utc).isoformat(),
              'type': 'synthetic_pose_filter_check', 'speed_m_s': SPEED,
              'endpoints_m': [-HALF_SPAN, HALF_SPAN], 'round_trips': LEGS // 2,
              'notes': ['Uses current AdaptivePoseSmoother defaults and real camera intrinsics.',
                        'No physical motion, image detection, exposure blur or missed detections are simulated.',
                        'All delay/error metrics are filter-only, not end-to-end camera or preview latency.',
                        'Reversal timing includes discrete output sampling and depends on the turn phase.',
                        'Noise case uses independent Gaussian X noise with 1 mm standard deviation; not measured sensor noise.'],
              'cases': cases}
    (OUTPUT / 'results.json').write_text(json.dumps(report, indent=2) + '\n')
    with (OUTPUT / 'trajectory.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['time_s', 'target_x_m', 'filtered_x_m', 'noisy_measurement_x_m', 'noisy_filtered_x_m'])
        noisy = traces['nominal_with_1mm_noise']
        writer.writerows(zip(times, target, filtered, noisy[2], noisy[3]))

    print(json.dumps(cases['nominal'], indent=2))
    print('Artifacts:', OUTPUT)
    if importlib.util.find_spec('matplotlib'):
        plot_existing()
    else:
        print('Plot with: /usr/bin/python3 -m carmer.check_xy_tracking --plot-only')


def plot_existing():
    data = np.genfromtxt(OUTPUT / 'trajectory.csv', delimiter=',', names=True)
    times, target, filtered = (data[name] for name in ('time_s', 'target_x_m', 'filtered_x_m'))
    noisy_filtered = data['noisy_filtered_x_m']
    previous = None
    previous_path = OUTPUT / 'before_fast_xy' / 'trajectory.csv'
    if previous_path.is_file():
        old_data = np.genfromtxt(previous_path, delimiter=',', names=True)
        if len(old_data) == len(data) and np.allclose(old_data['time_s'], times):
            previous = old_data['filtered_x_m']
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    ax = axes[0, 0]
    ax.plot(times, target * 100, label='Input trajectory', color='#333333', linestyle='--')
    ax.plot(times, filtered * 100, label='Filtered output', color='#0072B2')
    ax.set(title='Three round trips at 30 cm/s', xlabel='Time (s)', ylabel='X position (cm)')
    ax.legend()
    ax = axes[0, 1]
    if previous is not None:
        ax.plot(times, (target - previous) * 1000, label='Previous filter', color='#CC3311', alpha=.8)
    ax.plot(times, (target - filtered) * 1000, label='Noiseless', color='#0072B2')
    ax.plot(times, (target - noisy_filtered) * 1000, label='With 1 mm input noise', color='#888888', alpha=.55)
    ax.axhline(0, color='gray', linewidth=.8)
    ax.set(title='Signed following error', xlabel='Time (s)', ylabel='Input minus output (mm)')
    ax.legend()
    for ax, event, title in [(axes[1, 0], START + LEG_TIME, 'First reversal: right to left'),
                            (axes[1, 1], STOP, 'Stop at the left endpoint')]:
        relative = (times - event) * 1000
        mask = (relative >= -150) & (relative <= 350)
        ax.plot(relative[mask], target[mask] * 100, label='Input', color='#333333', linestyle='--', marker='.')
        if previous is not None:
            ax.plot(relative[mask], previous[mask] * 100, label='Previous filter', color='#CC3311', alpha=.8)
        ax.plot(relative[mask], filtered[mask] * 100, label='Filtered', color='#0072B2', marker='.')
        ax.axvline(0, color='gray', linewidth=.8)
        ax.set(title=title, xlabel='Time relative to event (ms)', ylabel='X position (cm)')
        ax.legend()
    for ax in axes.flat:
        ax.grid(alpha=.2)
    fig.suptitle('Current X/Y smoothing | 43.25 updates/s | Simulation: filter latency only')
    fig.savefig(OUTPUT / 'tracking.png', dpi=160)
    plt.close(fig)
    print('Plot:', OUTPUT / 'tracking.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plot-only', action='store_true', help='plot saved CSV using an environment with matplotlib')
    args = parser.parse_args()
    if args.plot_only:
        plot_existing()
    else:
        main()
