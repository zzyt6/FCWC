"""Timestamp-preserving receiver for ../carmer/heading_udp.py (no camera/DDS)."""
import json
import math
import socket


class CarmerYaw:
    def __init__(self, port, timeout_s=.3, offset_rad=0., tag_ids=None):
        self.timeout = float(timeout_s)
        self.offset = float(offset_rad)
        self.allowed_ids = frozenset(tag_ids) if tag_ids is not None else None
        self.tag_ids = []
        self.raw_yaw = 0.
        self.yaw = 0.
        self.stamp = self.last_packet_stamp = -float('inf')
        self.reason = '等待 carmer 航向'
        self.tags_in_pelvis = []
        self.rejected = 0
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self.sock.bind(('127.0.0.1', int(port)))
            self.sock.setblocking(False)
        except BaseException:
            self.sock.close()
            raise

    def accept(self, data, now):
        try:
            value = json.loads(data)
            stamp = value['captured_monotonic']
            valid = value['valid']
            if (value['schema'] != 'carmer_cart_yaw_v1' or type(stamp) not in (int, float)
                    or not math.isfinite(stamp) or not 0 <= now - stamp <= self.timeout
                    or stamp <= self.last_packet_stamp or type(valid) is not bool
                    or value['reference_frame'] != 'pelvis_level'):
                raise ValueError('invalid camera packet')
            yaw = value['yaw_rad']
            if valid and (type(yaw) not in (int, float) or not math.isfinite(yaw)
                          or abs(yaw) > math.pi + 1e-6):
                raise ValueError('invalid yaw')
            tag_ids = value.get('tag_ids', [])
            if valid and self.allowed_ids is not None:
                if (not isinstance(tag_ids, list) or not tag_ids
                        or any(type(tag) is not int or tag not in self.allowed_ids for tag in tag_ids)):
                    self.reason = '忽略非小车标签；仅接受 ID ' + ', '.join(map(str, sorted(self.allowed_ids)))
                    raise ValueError('unexpected cart tag')
            self.last_packet_stamp = stamp
            if valid:
                self.raw_yaw = float(yaw)
                self.yaw = math.atan2(math.sin(yaw - self.offset), math.cos(yaw - self.offset))
                self.stamp, self.reason = stamp, 'ok'
                self.tag_ids = list(tag_ids)
                self.tags_in_pelvis = []
                # Optional display geometry never affects policy yaw validity.
                for tag in value.get('tags_in_pelvis', []) if isinstance(value.get('tags_in_pelvis', []), list) else []:
                    if not isinstance(tag, dict):
                        continue
                    pos = tag.get('position_m')
                    if (type(tag.get('id')) is int and tag['id'] in self.tag_ids
                            and isinstance(pos, list) and len(pos) == 3
                            and all(type(x) in (int, float) and math.isfinite(x) and abs(x) < 20 for x in pos)):
                        self.tags_in_pelvis.append(dict(id=tag['id'], position_m=pos))
            else:
                self.reason = 'carmer 暂无有效标签航向'
            return True
        except (ValueError, TypeError, KeyError, OverflowError, UnicodeDecodeError):
            self.rejected += 1
            return False

    def poll(self, now):
        for _ in range(64):
            try:
                data, sender = self.sock.recvfrom(4096)
            except BlockingIOError:
                break
            except OSError as exc:
                self.stamp = -float('inf')
                self.reason = '相机航向接收异常：' + str(exc)
                break
            if sender[0] == '127.0.0.1':
                self.accept(data, now)
        return self.fresh(now)

    def fresh(self, now):
        return 0 <= now - self.stamp <= self.timeout

    def set_offset(self, offset):
        """Rebase the latest raw measurement without making its timestamp newer."""
        offset = float(offset)
        if not math.isfinite(offset):
            raise ValueError('航向零位必须是有限数值')
        self.offset = math.atan2(math.sin(offset), math.cos(offset))
        self.yaw = math.atan2(math.sin(self.raw_yaw - self.offset),
                              math.cos(self.raw_yaw - self.offset))

    def snapshot(self, now):
        return dict(fresh=self.fresh(now), yaw_rad=self.yaw if math.isfinite(self.stamp) else None,
                    raw_yaw_rad=self.raw_yaw if math.isfinite(self.stamp) else None,
                    yaw_offset_rad=self.offset,
                    captured_monotonic=self.stamp if math.isfinite(self.stamp) else None,
                    timeout_s=self.timeout, tag_ids=list(self.tag_ids),
                    age_s=now - self.stamp if math.isfinite(self.stamp) else None,
                    tags_in_pelvis=list(self.tags_in_pelvis),
                    reason=self.reason, rejected=self.rejected)

    def close(self):
        self.sock.close()
