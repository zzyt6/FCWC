"""Offline end-to-end check: nav bridge link -> executor -> Viser Controller -> C++ --simulate.

Run:  python3 nav/locomotion/test_executor_sim.py
No DDS and no robot: the deployment bridge produces synthetic feedback.
"""
import os
from pathlib import Path
import signal
import socket
import struct
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src/unitree_nav2_web'))
from unitree_nav2_web.locomotion import LocomotionLink  # noqa: E402

VELOCITY = struct.Struct('<IHBBQfff4xQ')
COMMAND_PORT, STATUS_PORT = 39102, 39105


def wait(link, predicate, timeout, what):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = link.snapshot()
        if predicate(status):
            return status
        time.sleep(.05)
    raise AssertionError('timeout waiting for %s: %s' % (what, link.snapshot()))


def main():
    process = subprocess.Popen(
        ['bash', str(ROOT / 'locomotion/run_executor.sh'), '--mode', 'sim',
         '--command-port', str(COMMAND_PORT), '--status-port', str(STATUS_PORT)],
        start_new_session=True)
    link = LocomotionLink(COMMAND_PORT, STATUS_PORT)
    sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        wait(link, lambda s: s['online'] and s['stage'] == 'WAITING' and s['healthy'], 60, 'executor online')
        assert not link.snapshot()['actions']['policy']
        wait(link, lambda s: s['actions']['prepare_pose'], 3, 'posture choice available')
        assert link.snapshot()['prepare_pose'] == 'original'
        assert {item['value'] for item in link.snapshot()['prepare_pose_options']} == {'original', 'arms_down'}
        link.request('prepare_pose', pose='arms_down')
        status = wait(link, lambda s: s['prepare_pose'] == 'arms_down', 3, 'arms-down selected')
        assert status['stage'] == 'WAITING' and not status['armed']
        link.request('acquire')
        wait(link, lambda s: s['stage'] == 'ZERO' and s['actions']['prepare'], 20, 'acquired')
        assert link.snapshot()['prepare_pose'] == 'arms_down'
        link.request('prepare')
        status = wait(link, lambda s: s['stage'] == 'HOLDING' and s['actions']['policy'], 20, 'prepared')
        assert not status['prepare_pose_selectable'] and status['prepare_pose'] == 'arms_down'
        link.request('prepare_pose', pose='original')  # Forged request after lock.
        link.request('policy')
        status = wait(link, lambda s: s['ready'], 10, 'policy running')
        assert status['prepare_pose'] == 'arms_down' and not status['actions']['prepare_pose']
        assert status['velocity'] == [0.0, 0.0, 0.0], status
        # Disabled packets (motion gate closed) must not move the robot.
        for sequence in range(20):
            sender.sendto(VELOCITY.pack(0x47315643, 2, 0, 1, sequence, .3, 0, 0,
                                       time.monotonic_ns()), ('127.0.0.1', COMMAND_PORT))
            time.sleep(.05)
        assert link.snapshot()['velocity'][0] == 0.0, link.snapshot()
        # Enabled packets use the training 0.4 s command filter through Viser.
        for sequence in range(40):
            command_time = time.monotonic_ns()
            sender.sendto(VELOCITY.pack(0x47315643, 2, 1, 0, sequence, .3, 0, 0,
                                       command_time), ('127.0.0.1', COMMAND_PORT))
            time.sleep(.05)
        status = link.snapshot()
        assert status['nav_command_fresh'] and .2 < status['velocity'][0] <= .3 + 1e-6, status
        assert status['selected_model'] == 'homie', status
        print('moving:', status['velocity'], status['selected_model'], status['heading'])
        # Expire the ORIGINAL command, then filter down to the switch threshold.
        # Exact zero and Balance selection occur together, without a dwell.
        status = wait(link, lambda s: s['decelerating'], .6, 'smooth stop')
        assert 0 < status['velocity'][0] < .3 and status['selected_model'] == 'homie', status
        assert status['ready'], status  # a normal stop must not look like a fault
        status = wait(link, lambda s: s['velocity'] == [0.0, 0.0, 0.0], 1., 'exact zero')
        zero_seen = time.monotonic()
        assert zero_seen-command_time*1e-9 <= 1.2, status  # includes 100 ms status publication
        assert status['selected_model'] == 'balance', status
        print('stop: zero and Balance observed after %.3f s without dwell'
              % (zero_seen-command_time*1e-9))
        link.request('stop')
        status = wait(link, lambda s: s['stage'] == 'DAMPED', 5, 'damped')
        assert not status['ready'] and status['actions']['reset']
        link.request('reset')
        wait(link, lambda s: s['stage'] == 'WAITING' and s['healthy'], 20, 'reset to read-only')
        wait(link, lambda s: s['actions']['prepare_pose'], 3, 'choice unlocked after reset')
        link.request('prepare_pose', pose='original')
        wait(link, lambda s: s['prepare_pose'] == 'original', 3, 'original selected')
        link.request('acquire')
        wait(link, lambda s: s['actions']['prepare'], 20, 'reacquired')
        link.request('prepare')
        wait(link, lambda s: s['actions']['policy'], 20, 'original prepared')
        link.request('policy')
        status = wait(link, lambda s: s['ready'], 10, 'original policy running')
        assert status['prepare_pose'] == 'original' and not status['prepare_pose_selectable']
        link.request('stop')
        wait(link, lambda s: s['stage'] == 'DAMPED', 5, 'final stop')
        print('EXECUTOR SIM OK')
    finally:
        link.close()
        sender.close()
        os.killpg(process.pid, signal.SIGINT)
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)


if __name__ == '__main__':
    main()
