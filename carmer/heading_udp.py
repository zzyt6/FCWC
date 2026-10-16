"""Publish gravity-leveled cart heading without display filtering on loopback."""
import json
import math
import socket
import numpy as np


def heading_packet(result):
    # V8.1 trained with delayed/noisy, sample-and-held yaw, not a low-pass.
    # This is already pelvis-compensated and fused across eligible tags.
    # Never fall back to display history if this frame has no raw heading.
    heading = result.get('raw_fused_cart_heading') or {}
    compensation = result.get('pelvis_compensation') or {}
    latency_ms = (result.get('cart_heading_settings') or {}).get('camera_latency_ms', 0.)
    stamp = float(result['frame_timestamp_monotonic']) - float(latency_ms) / 1000.
    yaw = heading.get('yaw_rad')
    valid = (heading.get('valid') is True and compensation.get('valid') is True
             and heading.get('reference_frame') == 'pelvis_level'
             and isinstance(yaw, (int, float)) and math.isfinite(yaw))
    positions = []
    if valid and 'camera_position_in_pelvis_m' in compensation:
        rotation = np.asarray(compensation['R_camera_to_pelvis'])
        origin = np.asarray(compensation['camera_position_in_pelvis_m'])
        for tag in result.get('tags', []):
            if tag['id'] not in heading.get('tag_ids', []):
                continue
            pose = tag.get('raw_pose', tag.get('pose'))
            if pose is not None:
                position = origin + rotation @ np.asarray(pose['tvec_m'])
                if position.shape == (3,) and np.isfinite(position).all():
                    positions.append(dict(id=int(tag['id']), position_m=position.tolist()))
    return dict(schema='carmer_cart_yaw_v1', captured_monotonic=stamp,
                frame_number=int(result['frame_number']), valid=bool(valid),
                reference_frame='pelvis_level', yaw_rad=float(yaw) if valid else None,
                tag_ids=heading.get('tag_ids', []),
                tags_in_pelvis=positions,
                reason='ok' if valid else heading.get('reason', 'compensation_unavailable'))


class HeadingPublisher:
    def __init__(self, port):
        self.destination = ('127.0.0.1', port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setblocking(False)

    def publish(self, result):
        try:
            self.sock.sendto(json.dumps(heading_packet(result), allow_nan=False).encode(), self.destination)
        except (BlockingIOError, OSError):
            # Never block image processing; the consumer owns the freshness watchdog.
            pass

    def close(self):
        self.sock.close()
