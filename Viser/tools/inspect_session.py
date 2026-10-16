"""Summarize a saved session without connecting to the robot."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import gzip
import json

import numpy as np

from core.robot import ROOT, JOINT_NAMES


def rows(path):
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        while True:
            try:
                line = stream.readline()
            except EOFError:
                print('Archive ended without a gzip footer; reporting all complete recovered records.')
                return
            if not line:
                return
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                print('Skipping incomplete JSON record.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('session', nargs='?', type=Path, help='Default: latest automatic archive')
    args = parser.parse_args()
    if args.session is None:
        sessions = sorted((ROOT / 'logs/sessions').glob('*/metadata.json'))
        if not sessions:
            parser.error('No saved sessions found')
        args.session = sessions[-1].parent
    metadata = json.loads((args.session / 'metadata.json').read_text())
    summary_path = args.session / 'summary.json'
    report = dict(session=str(args.session), mode=metadata['arguments']['mode'],
                  started_utc=metadata['started_utc'],
                  completion=json.loads(summary_path.read_text()) if summary_path.exists() else 'Still running or interrupted')
    stages = []
    times, policy_times, inference_ms, roundtrip_ms = [], [], [], []
    previous_target = None
    max_jump, jump_index = 0., None
    max_velocity = np.zeros(3)
    motor_writes = 0
    heading_errors, heading_corrections = [], []
    heading_modes, heading_gains, heading_filters = set(), set(), set()
    heading_limited = 0
    run_modes, gyro_cutoffs = set(), set()
    joint_names = metadata.get('joint_comparison', {}).get('joint_names', [])
    counts = np.zeros(len(joint_names), dtype=int)
    sums, squares, maxima = (np.zeros(len(joint_names)) for _ in range(3))
    for row in rows(args.session / 'telemetry.jsonl.gz'):
        times.append(row['monotonic'])
        comparison = row.get('joint_comparison')
        if comparison and joint_names:
            errors = np.asarray(comparison['error_rad'], dtype=float)
            valid = np.asarray(comparison['valid'], dtype=bool) & np.isfinite(errors)
            counts += valid
            absolute = np.where(valid, np.abs(errors), 0.)
            sums += absolute
            squares += absolute ** 2
            maxima = np.maximum(maxima, absolute)
        timing = row.get('timing', {})
        if timing.get('inference_ms') is not None:
            policy_times.append(row['monotonic'])
            inference_ms.append(timing['inference_ms'])
            roundtrip_ms.append(timing.get('observation_to_action_ms'))
        if not stages or stages[-1]['stage'] != row['stage']:
            stages.append(dict(stage=row['stage'], reason=row.get('reason'),
                               elapsed_s=row['monotonic']-metadata['monotonic_origin']))
        max_velocity = np.maximum(max_velocity, np.abs(row['velocity']))
        motor_writes += bool(row.get('bridge_motor_control_active', row.get('motor_command_written')))
        target = np.asarray(row['target'])
        if row['stage'] == 'POLICY' and previous_target is not None:
            jumps = np.abs(target-previous_target)
            index = int(np.argmax(jumps))
            if jumps[index] > max_jump:
                max_jump, jump_index = float(jumps[index]), index
        previous_target = target if row['stage'] == 'POLICY' else None
        heading = row.get('heading')
        if heading:
            heading_modes.add(heading['effective_frame'])
            run_modes.add(heading.get('run_mode', 'legacy'))
            if 'gyro_cutoff_hz' in heading: gyro_cutoffs.add(heading['gyro_cutoff_hz'])
            heading_gains.add(heading['heading_kp'])
            if 'run_mode' not in heading:
                heading_filters.add(heading.get('correction_filter_tau_s', 0.))
            if row['stage'] == 'POLICY' and heading['correction_enabled']:
                heading_errors.append(heading['error_rad'])
                heading_corrections.append(heading['correction_rad_s'])
                limit = heading['max_correction_rad_s']
                raw = heading.get('correction_raw_rad_s', heading['correction_rad_s'])
                heading_limited += limit > 0 and abs(raw) >= limit - 1e-8
    faults = []
    for row in rows(args.session / 'events.jsonl.gz'):
        if row['type'] in ('exception', 'damping'):
            faults.append(row)
    def rate(values):
        return (len(values)-1)/(values[-1]-values[0]) if len(values)>1 else None
    def stats(values):
        values = [v for v in values if v is not None]
        return dict(mean=float(np.mean(values)), p99=float(np.percentile(values, 99)),
                    max=float(np.max(values))) if values else None
    report.update(telemetry_records=len(times), policy_results=len(policy_times),
                  motor_command_records=motor_writes, stages=stages,
                  average_control_hz=rate(times), average_policy_hz=rate(policy_times),
                  inference_ms=stats(inference_ms), observation_to_action_ms=stats(roundtrip_ms),
                  max_abs_velocity_command=max_velocity.tolist(), max_policy_target_jump_rad=max_jump,
                  largest_jump_joint=JOINT_NAMES[jump_index] if jump_index is not None else None,
                  stop_and_fault_events=faults)
    if joint_names:
        report['joint_angle_errors'] = dict(
            unit='rad', definition='last sent command - latest feedback; valid position-control samples only',
            joints={name: dict(samples=int(counts[i]), mean_abs=float(sums[i]/counts[i]),
                               rms=float(np.sqrt(squares[i]/counts[i])), max_abs=float(maxima[i]))
                    for i, name in enumerate(joint_names) if counts[i]})
    if heading_modes:
        report['heading_control'] = dict(
            frames=sorted(heading_modes), gains_used=sorted(heading_gains),
            run_modes=sorted(run_modes), gyro_cutoff_hz=sorted(gyro_cutoffs),
            correction_filter_tau_s=sorted(heading_filters),
            active_samples=len(heading_errors),
            absolute_error_deg=stats(np.abs(np.rad2deg(heading_errors))),
            absolute_correction_rad_s=stats(np.abs(heading_corrections)),
            correction_saturated_fraction=heading_limited/len(heading_errors) if heading_errors else None,
            limitation='IMU heading feedback only; no measured XY trajectory or linear velocity')
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
