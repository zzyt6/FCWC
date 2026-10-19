"""Keep ROS acquisition timestamps across local monotonic watchdogs."""
import time


def sample_time(node, message, max_age, *, stream=None):
    stamp = message.header.stamp.sec*1000000000 + message.header.stamp.nanosec
    now = time.monotonic()
    age = (node.get_clock().now().nanoseconds-stamp)*1e-9
    if stamp <= 0 or not -.03 <= age <= max_age:
        return None
    if stream is not None:
        stamps = getattr(node, '_sample_stamps', None)
        if stamps is None:
            stamps = node._sample_stamps = {}
        if stamp <= stamps.get(stream, 0):
            return None
        stamps[stream] = stamp
    return now-age
