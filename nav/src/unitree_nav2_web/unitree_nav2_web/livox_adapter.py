"""Adapt Livox Driver 2's native point timestamps to the G1 FAST-LIO input."""
from array import array
import copy

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2, PointField


OUTPUT_DTYPE = np.dtype([
    ('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('intensity', '<f4'),
    ('ring', '<u2'), ('time', '<f4'),
])


def adapt_cloud(message):
    """Keep sensor time/coordinates; convert absolute ns to relative ns.

    The bundled Livox driver writes PointXyzlt.offset_time (absolute integer
    nanoseconds) into its FLOAT64 timestamp field, not seconds. At Unix epoch
    magnitudes conversion to double introduces at most 128 ns rounding.
    """
    fields = {field.name: field for field in message.fields}
    names = ['x', 'y', 'z', 'intensity', 'line', 'timestamp']
    types = [PointField.FLOAT32] * 4 + [PointField.UINT8, PointField.FLOAT64]
    for name, datatype in zip(names, types):
        field = fields.get(name)
        if field is None or field.datatype != datatype or field.count != 1:
            raise ValueError('Unexpected Livox field: ' + name)
    endian = '>' if message.is_bigendian else '<'
    dtype = np.dtype(dict(names=names,
                          formats=[endian + t for t in ['f4'] * 4 + ['u1', 'f8']],
                          offsets=[fields[name].offset for name in names],
                          itemsize=message.point_step))
    points = np.ndarray((message.height, message.width), dtype=dtype,
                        buffer=message.data,
                        strides=(message.row_step, message.point_step)).ravel()
    base_ns = message.header.stamp.sec * 1000000000 + message.header.stamp.nanosec
    offsets = points['timestamp'] - float(base_ns)
    if (not np.isfinite(offsets).all() or (offsets < -256).any()
            or (offsets > 200000000).any()):
        raise ValueError('Livox timestamps must be absolute nanoseconds within 200 ms of header')
    if (points['line'] >= 4).any():
        raise ValueError('Expected MID-360 line IDs 0..3')
    output = np.empty(points.size, dtype=OUTPUT_DTYPE)
    for name in names[:4]:
        output[name] = points[name]
    output['ring'] = points['line']
    output['time'] = np.maximum(offsets, 0)
    # MID-360 packets overlap slightly in acquisition time. Keep every point's
    # coordinates and timestamp together, with the final point at scan end as
    # expected by FAST-LIO's IMU synchronization (before its internal sort).
    if (np.diff(offsets) < 0).any():
        output = output[np.argsort(offsets, kind='stable')]
    return PointCloud2(
        header=copy.deepcopy(message.header), height=1, width=points.size,
        fields=[PointField(name=name, offset=OUTPUT_DTYPE.fields[name][1],
                           datatype=PointField.UINT16 if name == 'ring' else PointField.FLOAT32,
                           count=1) for name in OUTPUT_DTYPE.names],
        is_bigendian=False, point_step=22, row_step=points.size * 22,
        data=array('B', output.tobytes()), is_dense=message.is_dense)


class LivoxAdapter(Node):
    def __init__(self):
        super().__init__('livox_adapter')
        self.publisher = self.create_publisher(
            PointCloud2, '/navigation/livox_points', qos_profile_sensor_data)
        self.create_subscription(PointCloud2, '/livox/lidar', self.cloud, qos_profile_sensor_data)

    def cloud(self, message):
        try:
            self.publisher.publish(adapt_cloud(message))
        except (ValueError, TypeError, KeyError) as exc:
            self.get_logger().error(str(exc), throttle_duration_sec=2.0)


def main(args=None):
    rclpy.init(args=args)
    node = LivoxAdapter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
