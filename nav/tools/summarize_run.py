#!/usr/bin/env python3
"""Summarize one navigation run: camera, body control, navigation and CPU rates.

    python3 nav/tools/summarize_run.py nav/log/runs/<run>        # archived by record_run.sh
    python3 nav/tools/summarize_run.py ~/.ros/log/<run>/launch.log

Reads the periodic TIMING lines (2 s windows, n/mean/p95/max ms), CAMERA lines
and fault markers from launch.log, plus topics.csv and cpu.csv when present.
Only the standard library is used.
"""
import csv
from pathlib import Path
import re
import statistics
import sys

TIMING = re.compile(r'TIMING (\S+) \(([\d.]+) s, n/mean/p95/max ms\): (.*)')
FIELD = re.compile(r'(\w+)=(\d+)/([-\d.]+)/([-\d.]+)/([-\d.]+)')
DETECT = re.compile(r'CAMERA Capture ([\d.]+) FPS \| Detect ([\d.]+) FPS(?:.*?Search window (\d+) / fallback (\d+) / full (\d+))?')
MARKERS = ('CONTROL TICK late', 'POLICY HOLD started', 'policy heartbeat timeout', 'CAMERA ERROR',
           'CAMERA process exited', 'CAMERA STATUS', 'Traceback', 'FAULT')


def pct(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, int(q * len(values)))] if values else float('nan')


def parse_launch(path):
    windows = {}  # title -> stage -> list of (rate_hz, mean, p95, max)
    detect, search, markers = [], None, {name: 0 for name in MARKERS}
    for line in path.read_text(errors='replace').splitlines():
        match = TIMING.search(line)
        if match:
            title, span = match.group(1), float(match.group(2))
            for name, n, mean, p95, peak in FIELD.findall(match.group(3)):
                windows.setdefault(title, {}).setdefault(name, []).append(
                    (int(n) / span, float(mean), float(p95), float(peak)))
            continue
        match = DETECT.search(line)
        if match:
            detect.append(float(match.group(2)))
            if match.group(3):
                search = dict(window=int(match.group(3)), fallback=int(match.group(4)), full=int(match.group(5)))
        for name in MARKERS:
            markers[name] += name in line
    return windows, detect, search, markers


def stage(windows, title, name, label, rate=True, value='mean'):
    samples = windows.get(title, {}).get(name)
    if not samples:
        return f'  {label:34s} (no data)'
    rates = [s[0] for s in samples]
    means, p95s, peaks = [s[1] for s in samples], [s[2] for s in samples], [s[3] for s in samples]
    text = f'  {label:34s} '
    if rate:
        text += f'rate median {statistics.median(rates):5.1f} Hz (worst window {min(rates):5.1f}) | '
    return text + (f'mean {statistics.median(means):6.1f} ms, p95 {statistics.median(p95s):6.1f} ms, '
                   f'max {max(peaks):6.1f} ms  [{len(samples)} windows]')


def topic_rates(path):
    rows = list(csv.DictReader(path.open()))
    if not rows:
        return []
    lines = []
    for topic in [key for key in rows[0] if key not in ('time', 'window_s')]:
        rates = [float(row[topic]) / float(row['window_s']) for row in rows if row[topic] != '']
        active = [rate for rate in rates if rate > 0]
        if active:
            lines.append(f'  {topic:34s} median {statistics.median(active):5.1f} Hz, p10 {pct(active, .1):5.1f} Hz '
                         f'({len(active)}/{len(rates)} active windows)')
        else:
            lines.append(f'  {topic:34s} no messages')
    return lines


def cpu_usage(path):
    rows = list(csv.DictReader(path.open()))
    if not rows:
        return []
    cpus = sorted({key[:-5] for key in rows[0] if key.endswith('_busy')}, key=lambda name: int(name[3:]))
    lines = []
    for cpu in cpus:
        busy = [float(row[cpu + '_busy']) for row in rows]
        freq = [float(row[cpu + '_mhz']) for row in rows if row.get(cpu + '_mhz')]
        lines.append(f'  {cpu:5s} busy median {statistics.median(busy):5.1f} %, p95 {pct(busy, .95):5.1f} %'
                     + (f' | freq median {statistics.median(freq):6.0f} MHz' if freq else ''))
    return lines


def process_usage(path):
    totals, windows = {}, 0
    for line in path.open(errors='replace'):
        if line.startswith('---'):
            windows += 'over the last' in line  # Older logs held ps lifetime averages.
            continue
        parts = line.split(None, 3)
        if windows and len(parts) == 4:
            key = re.sub(r' --ros-args.*| --config.*', '', parts[3].strip())[:90]
            totals[key] = totals.get(key, 0.) + float(parts[0])
    if not windows:
        return ['  (older ps-based procs.log; per-window usage unavailable)']
    return [f'  {total/windows:6.1f} %  {name}' for name, total in sorted(totals.items(), key=lambda item: -item[1])[:14]]


def main():
    target = Path(sys.argv[1] if len(sys.argv) > 1 else '.').expanduser()
    launch = target / 'launch.log' if target.is_dir() else target
    folder = launch.parent
    windows, detect, search, markers = parse_launch(launch)
    out = [f'Run: {launch}']
    out += [f'Executor telemetry: {line.split("RECORDING ", 1)[1].strip()}'
            for line in launch.read_text(errors='replace').splitlines() if 'RECORDING /' in line]
    out += ['', 'Camera (cart AprilTag; target >= 50 Hz, frame age ~40 ms):']
    out.append(stage(windows, 'camera_seen_by_control', 'capture_to_receive', 'new frames seen by control'))
    out.append(stage(windows, 'camera_seen_by_control', 'frame_age', 'frame age at each control tick', rate=False))
    out.append(stage(windows, 'camera_seen_by_control', 'processing', 'camera processing per frame', rate=False))
    if detect:
        out.append(f'  {"camera process detect FPS":34s} median {statistics.median(detect):5.1f}, '
                   f'p10 {pct(detect, .1):5.1f} ({len(detect)} reports)')
    if search:
        out.append(f'  {"tag search (cumulative)":34s} window {search["window"]}, fallback {search["fallback"]}, '
                   f'full {search["full"]}')
    out += ['', 'Body control (target 50 Hz, tick 20 ms):']
    out.append(stage(windows, 'control', 'tick_interval', 'control tick interval'))
    out.append(stage(windows, 'control', 'step', 'control step', rate=False))
    out.append(stage(windows, 'control', 'balance_homie_onnx', 'Balance/Homie inference', rate=False))
    out.append(stage(windows, 'executor', 'step', 'executor UDP loop'))
    out += ['', 'Navigation inputs (LiDAR default 20 Hz):']
    for name, label in (('odometry', 'FAST-LIO odometry received'), ('cloud_filter', 'cart cloud filter'),
                        ('scan_mask', 'obstacle scan mask'), ('scan_localization_mask', 'localization scan mask')):
        out.append(stage(windows, 'cart_navigation', name, label))
    out += ['', 'Pose latency chain (from LiDAR measurement time; idle 20 Hz baseline: raw 55 ms incl. 50 ms scan, odometry 35 ms):']
    for name, label in (('latency_raw_cloud', 'raw cloud arrival (from scan START)'),
                        ('cloud_wait', 'cloud queue (cart wait + filter)'),
                        ('latency_filtered_cloud', 'filtered cloud to FAST-LIO (scan END)'),
                        ('latency_odom_in', '/Odometry_base arrival (scan END)'),
                        ('pairing_wait', 'odometry wait for cart geometry'),
                        ('latency_odom_out', '/Odometry_navigation out (scan END)')):
        out.append(stage(windows, 'cart_navigation', name, label, rate=False))
    for name, label in (('topics.csv', 'Topic rates (record_run):'), ('cpu.csv', 'CPU (record_run):')):
        if (folder / name).is_file():
            out += ['', label] + (topic_rates if name == 'topics.csv' else cpu_usage)(folder / name)
    if (folder / 'procs.log').is_file():
        out += ['', 'Top processes (mean %CPU of one core over 5 s windows; record_run):'] + process_usage(folder / 'procs.log')
    out += ['', 'Fault / stall markers (count of lines):']
    out += [f'  {name:34s} {count}' for name, count in markers.items()]
    text = '\n'.join(out)
    print(text)
    if target.is_dir():
        (folder / 'summary.txt').write_text(text + '\n')


if __name__ == '__main__':
    main()
