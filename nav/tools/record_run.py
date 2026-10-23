#!/usr/bin/env python3
"""Record one real navigation run for later analysis; Ctrl-C archives and summarizes it.

Start with nav/tools/record_run.sh in a second terminal while run_navigation.sh is
running. It only subscribes to small topics (no point clouds), never publishes,
and runs on the navigation CPUs at low priority, away from the camera (4-5) and
the body policy (6-7). Output: nav/log/runs/<start time>[_<label>]/
  topics.csv   message counts per 2 s window (cmd_vel = RPP/DWB output rate, AMCL pose)
  cpu.csv      per-CPU busy % and clock every second
  procs.log    top processes by CPU used in each 5 s window (not ps lifetime averages)
  launch.log   the navigation launch log of this run (TIMING/CAMERA lines)
  ros_nodes/   per-node ROS logs written during the run
  git.txt      commit, changed files and the rate settings used
  summary.txt  nav/tools/summarize_run.py output
"""
import argparse
import csv
from datetime import datetime
import os
from pathlib import Path
import shutil
import signal
import subprocess
import threading
import time

NAV = Path(__file__).resolve().parents[1]
WORKSPACE = NAV.parent
ROS_LOG = Path(os.environ.get('ROS_LOG_DIR', Path.home() / '.ros/log'))
WINDOW_S = 2.
# Only rates launch.log lacks. Foxy rclpy costs ~1 ms per message: with odometry
# and scans also counted the recorder used ~23 % of a navigation core, so those
# rates come from the TIMING cart_navigation lines instead.
TOPICS = (('/cmd_vel', 'geometry_msgs.msg', 'Twist'),
          ('/amcl_pose', 'geometry_msgs.msg', 'PoseWithCovarianceStamped'))


def cpu_times():
    stats = {}
    for line in Path('/proc/stat').read_text().splitlines():
        name, *values = line.split()
        if name.startswith('cpu') and name != 'cpu':
            values = list(map(int, values))
            stats[name] = (values[3] + values[4], sum(values))  # idle + iowait, total
    return stats


def cpu_mhz(cpu):
    try:
        return int(Path(f'/sys/devices/system/cpu/{cpu}/cpufreq/scaling_cur_freq').read_text()) / 1000
    except (OSError, ValueError):
        return ''


TICKS = os.sysconf('SC_CLK_TCK')


def process_times():
    """pid -> (utime+stime ticks, last CPU, command line) from /proc."""
    result = {}
    for entry in os.scandir('/proc'):
        if not entry.name.isdigit():
            continue
        try:
            stat = Path(entry.path, 'stat').read_text()
            fields = stat[stat.rindex(')')+2:].split()
            command = Path(entry.path, 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace').strip()
            result[int(entry.name)] = (int(fields[11])+int(fields[12]), fields[36],
                                       command or stat[stat.index('(')+1:stat.rindex(')')])
        except (OSError, ValueError, IndexError):
            continue
    return result


def sample_system(folder, stop):
    previous = cpu_times()
    names = sorted(previous, key=lambda name: int(name[3:]))
    with (folder / 'cpu.csv').open('w', newline='') as cpu_file, (folder / 'procs.log').open('w') as procs:
        writer = csv.writer(cpu_file)
        writer.writerow(['time'] + [f'{name}_{kind}' for name in names for kind in ('busy', 'mhz')])
        tick = 0
        before, before_time = process_times(), time.monotonic()
        while not stop.wait(1.):
            current = cpu_times()
            row = [f'{time.time():.1f}']
            for name in names:
                idle = current[name][0] - previous[name][0]
                total = current[name][1] - previous[name][1]
                row += [f'{100. * (1 - idle / total):.1f}' if total else '', cpu_mhz(name)]
            writer.writerow(row)
            cpu_file.flush()
            previous = current
            tick += 1
            if tick % 5 == 0:
                after, after_time = process_times(), time.monotonic()
                span = (after_time-before_time)*TICKS
                usage = sorted(((100.*(ticks-before[pid][0])/span, pid, last, command)
                                for pid, (ticks, last, command) in after.items() if pid in before),
                               reverse=True)[:16]
                procs.write(f'--- {time.time():.1f} {datetime.now():%H:%M:%S} (%CPU over the last {after_time-before_time:.1f} s, PID, last CPU, command)\n'
                            + ''.join(f'{pct:6.1f} {pid:>7} {last:>2} {command[:160]}\n' for pct, pid, last, command in usage))
                procs.flush()
                before, before_time = after, after_time


def count_topics(folder, stop):
    try:
        import importlib
        import rclpy
        from rclpy.qos import qos_profile_sensor_data
    except ImportError as exc:
        print(f'rclpy unavailable ({exc}); topic rates are not recorded', flush=True)
        return
    rclpy.init()
    # rclpy replaces the SIGINT handler; Ctrl-C must only end the recording loop.
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    node = rclpy.create_node('cwc_run_recorder')
    counts = {topic: 0 for topic, _, _ in TOPICS}
    lock = threading.Lock()

    def counter(topic):
        def callback(_):
            with lock:
                counts[topic] += 1
        return callback

    for topic, module, name in TOPICS:
        message = getattr(importlib.import_module(module), name)
        # Best effort receives from both reliable and best-effort publishers.
        node.create_subscription(message, topic, counter(topic), qos_profile_sensor_data)
    with (folder / 'topics.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['time', 'window_s'] + [topic for topic, _, _ in TOPICS])
        started = time.monotonic()
        while not stop.is_set():
            rclpy.spin_once(node, timeout_sec=.1)
            now = time.monotonic()
            if now - started >= WINDOW_S:
                with lock:
                    row = [counts[topic] for topic, _, _ in TOPICS]
                    counts.update({topic: 0 for topic in counts})
                writer.writerow([f'{time.time():.1f}', f'{now - started:.3f}'] + row)
                stream.flush()
                started = now
    node.destroy_node()
    rclpy.shutdown()


def archive(folder, started_wall):
    runs = [path for path in ROS_LOG.glob('*/launch.log') if path.stat().st_mtime >= started_wall]
    if runs:
        launch = max(runs, key=lambda path: path.stat().st_mtime)
        shutil.copy2(launch, folder / 'launch.log')
        (folder / 'ros_nodes').mkdir(exist_ok=True)
        for log in ROS_LOG.glob('*.log'):
            if log.stat().st_mtime >= started_wall:
                shutil.copy2(log, folder / 'ros_nodes' / log.name)
    else:
        print(f'No launch.log in {ROS_LOG} was written during this recording', flush=True)
    git = lambda *args: subprocess.run(['git', '-C', str(WORKSPACE), *args], capture_output=True, text=True).stdout
    settings = subprocess.run(
        ['grep', '-n', '-E', 'controller_frequency|update_frequency|direct_livox_hz|default_value="20"',
         str(NAV / 'src/unitree_nav2_web/config/nav2_foxy_g1.yaml'),
         str(NAV / 'src/unitree_nav2_web/launch/g1_navigation.launch.py')], capture_output=True, text=True).stdout
    (folder / 'git.txt').write_text('commit ' + git('rev-parse', 'HEAD') + '\n' + git('status', '--short')
                                    + '\n--- rate settings\n' + settings)
    missing = NAV / 'log/missing_tags'
    if missing.is_dir():
        new = [path for path in missing.rglob('*') if path.is_file() and path.stat().st_mtime >= started_wall]
        (folder / 'missing_tags.txt').write_text(f'{len(new)} files in {missing}\n' + '\n'.join(map(str, new[:200])))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('label', nargs='?', default='', help='optional short run name, e.g. cart_left90')
    args = parser.parse_args()
    os.sched_setaffinity(0, {int(cpu) for cpu in os.environ.get('G1_RECORD_CPUS', '0,1,2,3').split(',')})
    os.nice(10)
    name = datetime.now().strftime('%Y%m%d_%H%M%S') + (f'_{args.label}' if args.label else '')
    folder = NAV / 'log/runs' / name
    folder.mkdir(parents=True)
    # Logs still being written during the recording (mtime) belong to this run.
    started_wall = time.time()
    stop = threading.Event()
    threads = [threading.Thread(target=sample_system, args=(folder, stop), daemon=True)]
    threads[0].start()
    for number in (signal.SIGINT, signal.SIGTERM):
        signal.signal(number, lambda *_: stop.set())
    print(f'Recording to {folder}; press Ctrl-C after the run (navigation may keep running).', flush=True)
    count_topics(folder, stop)
    while not stop.wait(1.):
        pass  # Only reached without rclpy.
    for thread in threads:
        thread.join(timeout=3)
    archive(folder, started_wall)
    if (folder / 'launch.log').is_file():
        subprocess.run(['python3', str(NAV / 'tools/summarize_run.py'), str(folder)])
    print(f'Saved {folder}', flush=True)


if __name__ == '__main__':
    main()
