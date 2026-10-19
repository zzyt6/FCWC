"""Own one direct Livox driver and restore the original G1 lidar on exit.

No motion APIs are used. The recovery journal is kept outside the workspace so
an interrupted switch can be recovered with --restore before another run.
"""
import argparse
import binascii
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import struct
import subprocess
import sys
import tempfile
import time
import zlib


# Parameters the Livox driver changes: format, pattern, data destinations,
# installation transform, work mode and IMU enable. Never change the lidar IP.
PARAMETERS = (0, 1, 5, 6, 7, 18, 26, 28)


class LidarParameters:
    def __init__(self, host, lidar):
        self.host, self.lidar = host, lidar
        self.sequence = int(time.monotonic_ns()) & 0xffff

    def request(self, command, payload):
        if command not in (0x0100, 0x0101):
            raise ValueError('Only lidar parameter get/set is supported')
        self.sequence = (self.sequence + 1) & 0xffff
        head = struct.pack('<BBHIHBB6x', 0xaa, 0, 24 + len(payload),
                           self.sequence, command, 0, 0)
        packet = head + struct.pack('<HI', binascii.crc_hqx(head, 0xffff),
                                    zlib.crc32(payload)) + payload
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.bind((self.host, 0))
            sock.settimeout(2.0)
            for _ in range(3):
                sock.sendto(packet, (self.lidar, 56100))
                try:
                    data, sender = sock.recvfrom(4096)
                except socket.timeout:
                    continue
                if sender != (self.lidar, 56100) or len(data) < 25:
                    continue
                sof, version, size, seq, cmd, kind, _ = struct.unpack_from('<BBHIHBB', data)
                if (sof, version, size, seq, cmd, kind) != (0xaa, 0, len(data), self.sequence, command, 1):
                    continue
                if struct.unpack_from('<HI', data, 18) != (binascii.crc_hqx(data[:18], 0xffff), zlib.crc32(data[24:])):
                    continue
                if data[24] != 0:
                    raise RuntimeError('Lidar parameter request failed: ' + str(data[24]))
                return data[24:]
        raise RuntimeError('Lidar parameter request timed out')

    def read(self):
        payload = struct.pack('<HH', len(PARAMETERS), 0) + struct.pack('<8H', *PARAMETERS)
        data = self.request(0x0101, payload)
        count, = struct.unpack_from('<H', data, 1)
        offset, values = 3, {}
        for _ in range(count):
            key, size = struct.unpack_from('<HH', data, offset)
            offset += 4
            value = data[offset:offset + size]
            if len(value) != size:
                raise RuntimeError('Truncated lidar parameter response')
            values[str(key)] = value.hex()
            offset += size
        if set(values) != {str(key) for key in PARAMETERS}:
            raise RuntimeError('Incomplete lidar parameter snapshot')
        return values

    def restore(self, values):
        if set(values) != {str(key) for key in PARAMETERS}:
            raise ValueError('Invalid lidar recovery snapshot')
        payload = struct.pack('<HH', len(PARAMETERS), 0)
        for key in PARAMETERS:
            value = bytes.fromhex(values[str(key)])
            payload += struct.pack('<HH', key, len(value)) + value
        self.request(0x0100, payload)
        if self.read() != values:
            raise RuntimeError('Lidar parameters did not restore exactly')


class RobotLidarService:
    def __init__(self, interface, sdk_dir):
        if sdk_dir:
            sys.path.insert(0, sdk_dir)
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize
        from unitree_sdk2py.rpc.client import Client
        ChannelFactoryInitialize(0, interface)
        self.client = Client('robot_state', False)
        self.client.SetTimeout(3.0)
        self.client._RegistApi(1001, 0)
        self.client._RegistApi(1003, 0)

    def state(self):
        code, data = self.client._Call(1003, '{}')
        if code != 0:
            raise RuntimeError('Cannot read robot services: ' + str(code))
        services = {item['name']: item for item in json.loads(data)}
        state = services['lidar_driver']
        if state['status'] not in (0, 1):
            raise RuntimeError('Unknown lidar service state')
        return state

    def switch(self, enabled):
        code, data = self.client._Call(1001, json.dumps({'name': 'lidar_driver', 'switch': int(enabled)}))
        if code != 0:
            raise RuntimeError('Cannot switch lidar_driver: ' + str(code))
        if json.loads(data).get('status') != (0 if enabled else 1):
            raise RuntimeError('lidar_driver rejected the switch: ' + data)
        # On the G1, ServiceList can lag the switch acknowledgement by 7-10 s.
        # Wait for confirmation before allowing a second SDK to own the lidar.
        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline:
            if self.state()['status'] == (0 if enabled else 1):
                return
            time.sleep(.1)
        raise RuntimeError('lidar_driver did not reach the requested state')


def driver_config(host, lidar, values):
    roll, pitch, yaw, x, y, z = struct.unpack('<fffiii', bytes.fromhex(values['18']))
    return {
        'lidar_summary_info': {'lidar_type': 8},
        # The installed SDK disables lidar logging when the option is omitted.
        # Supplying only lidar_log_enable (even false) requires two extra keys
        # on older SDK2 releases and makes their parser reject the whole file.
        'MID360': {
            'lidar_net_info': dict(cmd_data_port=56100, push_msg_port=56200,
                                   point_data_port=56300, imu_data_port=56400, log_data_port=56500),
            'host_net_info': dict(cmd_data_ip=host, cmd_data_port=56101,
                                  push_msg_ip=host, push_msg_port=56201,
                                  point_data_ip=host, point_data_port=56301,
                                  imu_data_ip=host, imu_data_port=56401,
                                  log_data_ip='', log_data_port=56501),
        },
        'lidar_configs': [dict(ip=lidar, pcl_data_type=1, pattern_mode=0,
                               extrinsic_parameter=dict(roll=roll, pitch=pitch, yaw=yaw, x=x, y=y, z=z))],
    }


def restore_source(service, hardware, snapshot):
    # Try both operations even if one fails; retain the journal on any error.
    errors = []
    try:
        hardware.restore(snapshot['parameters'])
    except Exception as exc:
        errors.append(str(exc))
    try:
        service.switch(snapshot['service_enabled'])
    except Exception as exc:
        errors.append(str(exc))
    if errors:
        raise RuntimeError('; '.join(errors))


def stop_driver(child):
    if child is None or child.poll() is not None:
        return
    child.send_signal(signal.SIGINT)
    try:
        child.wait(timeout=8)
    except subprocess.TimeoutExpired:
        child.kill()
        child.wait(timeout=3)


def save_journal(path, snapshot):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(snapshot))
    temporary.replace(path)


def main(args=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--interface', default='eth0')
    parser.add_argument('--host-ip', default='192.168.123.164')
    parser.add_argument('--lidar-ip', default='192.168.123.120')
    parser.add_argument('--sdk-dir', default=os.environ.get('UNITREE_SDK2_PYTHON', str(Path.home() / 'unitree_sdk2_python')))
    parser.add_argument('--frequency', type=float, default=50.0, choices=(10.0, 20.0, 50.0))
    parser.add_argument('--duration', type=float, default=0.0, help='Stop and restore after this many seconds; 0 runs until stopped')
    parser.add_argument('--state-file', default=str(Path.home() / '.cache/cwc_navigation/livox_source.json'))
    parser.add_argument('--restore', action='store_true')
    options = parser.parse_args(args)
    if options.duration < 0:
        parser.error('--duration must be nonnegative')
    state_file = Path(options.state_file)
    state_file.parent.mkdir(parents=True, exist_ok=True)
    with state_file.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        service = RobotLidarService(options.interface, options.sdk_dir)
        if options.restore:
            snapshot = json.loads(state_file.read_text())
            pid = snapshot.get('driver_pid')
            if pid and Path('/proc/' + str(pid)).exists():
                raise RuntimeError('Stop the existing Livox driver before recovery: PID ' + str(pid))
            hardware = LidarParameters(snapshot['host'], snapshot['lidar'])
            restore_source(service, hardware, snapshot)
            state_file.unlink()
            print('Original lidar parameters and service restored.', flush=True)
            return
        if state_file.exists():
            raise RuntimeError('Unfinished lidar switch; run livox_source --restore first: ' + str(state_file))
        from ament_index_python.packages import get_package_prefix
        driver = Path(get_package_prefix('livox_ros_driver2')) / 'lib/livox_ros_driver2/livox_ros_driver2_safe_node'
        if not driver.is_file():
            raise RuntimeError('Livox driver is not built: ' + str(driver))
        state = service.state()
        if state['protect']:
            raise RuntimeError('The official lidar service is protected')
        hardware = LidarParameters(options.host_ip, options.lidar_ip)
        snapshot = dict(host=options.host_ip, lidar=options.lidar_ip,
                        service_enabled=state['status'] == 0, parameters=hardware.read())
        save_journal(state_file, snapshot)
        stopping = False

        def stop(signum, frame):
            nonlocal stopping
            stopping = True

        old_handlers = {sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM)}
        child = None
        try:
            service.switch(False)
            with tempfile.TemporaryDirectory(prefix='cwc_livox_') as directory:
                config = Path(directory) / 'MID360.json'
                config.write_text(json.dumps(driver_config(options.host_ip, options.lidar_ip, snapshot['parameters'])))
                if stopping:
                    return
                child = subprocess.Popen([
                    str(driver), '--ros-args', '-r', '__node:=navigation_livox_driver',
                    '-p', 'xfer_format:=0', '-p', 'multi_topic:=0', '-p', 'data_src:=0',
                    '-p', 'publish_freq:=' + str(options.frequency), '-p', 'output_data_type:=0',
                    '-p', 'frame_id:=livox_frame', '-p', 'user_config_path:=' + str(config),
                ], cwd=directory, start_new_session=True)
                snapshot['driver_pid'] = child.pid
                save_journal(state_file, snapshot)
                print('Direct Livox source requested at {} Hz; recovery journal: {}'.format(options.frequency, state_file), flush=True)
                started = time.monotonic()
                while not stopping and child.poll() is None:
                    if options.duration > 0 and time.monotonic() - started >= options.duration:
                        break
                    time.sleep(.1)
                if child.poll() is not None:
                    raise RuntimeError('Livox driver exited unexpectedly: ' + str(child.returncode))
                stop_driver(child)
        finally:
            stop_driver(child)
            try:
                restore_source(service, hardware, snapshot)
                state_file.unlink()
                print('Original lidar parameters and service restored.', flush=True)
            finally:
                for sig, handler in old_handlers.items():
                    signal.signal(sig, handler)


if __name__ == '__main__':
    main()
