"""Direct lidar format/recovery checks, with no robot communication."""
from array import array
import copy
import json
import struct
from types import SimpleNamespace

import numpy as np
import pytest
from rclpy.serialization import deserialize_message, serialize_message
from sensor_msgs.msg import PointCloud2, PointField

from unitree_nav2_web.livox_adapter import OUTPUT_DTYPE, adapt_cloud
from unitree_nav2_web import livox_source
from unitree_nav2_web.livox_source import PARAMETERS, LidarParameters, driver_config, restore_source


def livox_cloud(size=4000, bigendian=False):
    endian = '>' if bigendian else '<'
    data = np.zeros(size, dtype=np.dtype([
        ('x', endian+'f4'), ('y', endian+'f4'), ('z', endian+'f4'),
        ('intensity', endian+'f4'), ('tag', 'u1'), ('line', 'u1'), ('timestamp', endian+'f8')]))
    for i, name in enumerate(('x', 'y', 'z', 'intensity')):
        data[name] = np.arange(size)*.01+i
    data['line'] = np.arange(size) % 4
    base = 1792343631473193216
    data['timestamp'] = base + np.arange(size, dtype=np.int64)*5000
    msg = PointCloud2(height=1, width=size, point_step=26, row_step=size*26,
                      is_bigendian=bigendian, data=array('B', data.tobytes()),
                      fields=[PointField(name=name, offset=data.dtype.fields[name][1],
                                         datatype=datatype, count=1)
                              for name, datatype in zip(data.dtype.names, [7, 7, 7, 7, 2, 2, 8])])
    msg.header.frame_id = 'livox_frame'
    msg.header.stamp.sec, msg.header.stamp.nanosec = divmod(base, 1000000000)
    return msg, data


@pytest.mark.parametrize('size', [0, 1, 4000])
@pytest.mark.parametrize('bigendian', [False, True])
def test_real_epoch_timestamps_fields_and_ros_serialization(size, bigendian):
    msg, data = livox_cloud(size, bigendian)
    original = copy.deepcopy(msg)
    result = deserialize_message(serialize_message(adapt_cloud(msg)), PointCloud2)
    converted = np.frombuffer(result.data, dtype=OUTPUT_DTYPE)
    for name in ('x', 'y', 'z', 'intensity'):
        np.testing.assert_array_equal(converted[name], data[name])
    np.testing.assert_array_equal(converted['ring'], data['line'])
    np.testing.assert_allclose(converted['time'], np.arange(size)*5000, atol=256, rtol=0)
    assert result.header == msg.header and result.row_step == size*22
    assert msg == original


@pytest.mark.parametrize('bad_time', ['seconds', 'relative', 'before_header', 'nan', 'too_long'])
def test_reject_wrong_timestamp_units_or_corrupt_frame(bad_time):
    msg, data = livox_cloud()
    if bad_time == 'seconds':
        data['timestamp'] /= 1e9
    elif bad_time == 'relative':
        data['timestamp'] -= data['timestamp'][0]
    elif bad_time == 'before_header':
        data['timestamp'][20] = data['timestamp'][0] - 1000000
    elif bad_time == 'nan':
        data['timestamp'][0] = np.nan
    else:
        data['timestamp'][-1] += 1e9
    msg.data = array('B', data.tobytes())
    with pytest.raises(ValueError, match='timestamps'):
        adapt_cloud(msg)


def test_overlapping_packet_times_are_sorted_with_their_points():
    msg, data = livox_cloud(4000)
    data['timestamp'][96:192] -= 350000
    msg.data = array('B', data.tobytes())
    result = np.frombuffer(adapt_cloud(msg).data, dtype=OUTPUT_DTYPE)
    order = np.argsort(data['timestamp'], kind='stable')
    assert (np.diff(result['time']) >= 0).all()
    for name in ('x', 'y', 'z', 'intensity'):
        np.testing.assert_array_equal(result[name], data[name][order])
    np.testing.assert_array_equal(result['ring'], data['line'][order])
    np.testing.assert_allclose(result['time'], data['timestamp'][order] - data['timestamp'][0])


def test_organized_cloud_row_padding_is_not_a_point():
    msg, data = livox_cloud(4000)
    split = 2000*26
    msg.height, msg.width, msg.row_step = 2, 2000, split+8
    msg.data = array('B', msg.data[:split])+array('B', b'padding!')+array('B', msg.data[split:])+array('B', b'padding!')
    result = np.frombuffer(adapt_cloud(msg).data, dtype=OUTPUT_DTYPE)
    np.testing.assert_array_equal(result['x'], data['x'])


def test_config_preserves_calibration_and_targets_only_the_selected_lidar():
    values = {'18': struct.pack('<fffiii', 1.5, -.2, .7, 10, -20, 30).hex()}
    config = driver_config('192.168.123.164', '192.168.123.120', values)
    assert len(config['lidar_configs']) == 1
    assert config['lidar_configs'][0]['ip'] == '192.168.123.120'
    transform = config['lidar_configs'][0]['extrinsic_parameter']
    assert transform['roll'] == 1.5 and transform['x'] == 10
    assert config['MID360']['host_net_info']['point_data_ip'] == '192.168.123.164'


@pytest.mark.parametrize('enabled', [False, True])
@pytest.mark.parametrize('hardware_failure', [False, True])
def test_recovery_always_attempts_original_service_state(enabled, hardware_failure):
    calls = []

    def restore(values):
        calls.append(('parameters', values))
        if hardware_failure:
            raise RuntimeError('hardware unreachable')

    hardware = SimpleNamespace(restore=restore)
    service = SimpleNamespace(switch=lambda value: calls.append(('service', value)))
    snapshot = dict(service_enabled=enabled, parameters={'original': 'values'})
    if hardware_failure:
        with pytest.raises(RuntimeError, match='unreachable'):
            restore_source(service, hardware, snapshot)
    else:
        restore_source(service, hardware, snapshot)
    assert calls == [('parameters', snapshot['parameters']), ('service', enabled)]


def test_parameter_restore_refuses_partial_or_mismatched_snapshot():
    hardware = LidarParameters('127.0.0.1', '127.0.0.1')
    with pytest.raises(ValueError):
        hardware.restore({'6': '00'})
    values = {str(k): '01' for k in PARAMETERS}
    hardware.request = lambda *args: b'\x00'
    hardware.read = lambda: {**values, '6': '02'}
    with pytest.raises(RuntimeError, match='exactly'):
        hardware.restore(values)


@pytest.mark.parametrize('failure', ['service_stop', 'driver_exit'])
def test_start_failure_restores_source_and_clears_journal(monkeypatch, tmp_path, failure):
    calls = []
    values = {str(k): '01' for k in PARAMETERS}
    values['18'] = struct.pack('<fffiii', 0, 0, 0, 0, 0, 0).hex()

    def switch(enabled):
        calls.append(('service', enabled))
        if not enabled and failure == 'service_stop':
            raise RuntimeError('test startup failure')

    service = SimpleNamespace(state=lambda: {'status': 0, 'protect': 0}, switch=switch)
    hardware = SimpleNamespace(read=lambda: values, restore=lambda v: calls.append(('restore', v)))
    monkeypatch.setattr(livox_source, 'RobotLidarService', lambda *args: service)
    monkeypatch.setattr(livox_source, 'LidarParameters', lambda *args: hardware)
    child = SimpleNamespace(pid=123, returncode=2, poll=lambda: 2)
    monkeypatch.setattr(livox_source.subprocess, 'Popen', lambda *args, **kwargs: child)
    prefix = tmp_path/'install'
    driver = prefix/'lib/livox_ros_driver2/livox_ros_driver2_safe_node'
    driver.parent.mkdir(parents=True)
    driver.touch()
    import ament_index_python.packages
    monkeypatch.setattr(ament_index_python.packages, 'get_package_prefix', lambda name: str(prefix))
    journal = tmp_path/'recovery.json'
    with pytest.raises(RuntimeError):
        livox_source.main(['--state-file', str(journal)])
    assert calls == [('service', False), ('restore', values), ('service', True)]
    assert not journal.exists()


def test_service_waits_for_delayed_state_confirmation(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(livox_source.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(livox_source.time, 'sleep', lambda dt: clock.__setitem__(0, clock[0]+dt))
    service = object.__new__(livox_source.RobotLidarService)
    service.client = SimpleNamespace(_Call=lambda *args: (0, '{"status":1}'))
    service.state = lambda: {'status': 1 if clock[0] >= 7 else 0}
    service.switch(False)
    assert clock[0] >= 7
