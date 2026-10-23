"""Timestamp-preserving receiver for ../carmer/heading_udp.py (no camera/DDS)."""
import json
import math
import socket

import numpy as np


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
        self.pose_source = 'grasp'
        self.cart_pose_urdf = dict(valid=False, reason='等待完整小车视觉位姿')
        self.rejected = 0
        self.last_frame = {}
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
                    or not math.isfinite(stamp) or now < stamp
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
            # Late frames explain processing/transport lag, but never renew yaw.
            if stamp > self.last_frame.get('captured_monotonic', -float('inf')):
                timing = value.get('timing') or {}
                self.last_frame = dict(captured_monotonic=stamp, received_monotonic=now,
                    valid=valid, reason=str(value.get('reason', 'no_valid_heading')),
                    timing={key: timing[key] for key in ('processing_ms', 'detection_ms', 'capture_fps', 'detect_fps')
                            if isinstance(timing, dict) and type(timing.get(key)) in (int, float)
                            and math.isfinite(timing[key]) and timing[key] >= 0.})
            if now-stamp > self.timeout:
                raise ValueError('late camera frame')
            self.last_packet_stamp = stamp
            pose = self._cart_pose(value.get('cart_pose_urdf'), stamp)
            if self.pose_source == 'apriltag':
                if pose['valid']:
                    self.cart_pose_urdf = pose
                    self.raw_yaw = self.yaw = pose['yaw_rad']
                    self.stamp, self.reason = stamp, 'ok'
                    self.tag_ids = list(pose['tag_ids'])
                else:
                    # Match heading mode: hold the last valid pose and its
                    # capture time. Invalid frames never renew the timeout.
                    if not self.cart_pose_urdf['valid']:
                        self.cart_pose_urdf = pose
                    self.reason = '小车视觉位姿不可用：' + pose['reason']
                return True
            self.cart_pose_urdf = pose
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
                self.reason = self._missing_reason(value.get('reason'))
            return True
        except (ValueError, TypeError, KeyError, OverflowError, UnicodeDecodeError):
            self.rejected += 1
            return False

    @staticmethod
    def _cart_pose(pose, stamp):
        if not isinstance(pose, dict):
            return dict(valid=False, reason='相机未提供完整位姿；请使用导航启动的相机')
        if pose.get('valid') is not True:
            return dict(valid=False, reason=str(pose.get('reason', '无有效小车标签')))
        try:
            matrix = np.asarray(pose['T_level_cart'], dtype=float)
            ids = pose['tag_ids']
            yaw = pose['yaw_rad']
            if (pose.get('schema') != 'cart_pose_urdf_v1' or pose.get('extrinsics') != 'urdf'
                    or pose.get('reference_frame') != 'pelvis_level'
                    or pose.get('captured_monotonic') != stamp
                    or not isinstance(ids, list) or not ids or len(set(ids)) != len(ids)
                    or any(type(tag) is not int or tag not in (0, 1) for tag in ids)
                    or type(yaw) not in (int, float) or not math.isfinite(yaw)
                    or matrix.shape != (4, 4) or not np.isfinite(matrix).all()
                    or np.max(np.abs(matrix[:3, 3])) > 20
                    or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-8)
                    or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-6)
                    or not np.isclose(np.linalg.det(matrix[:3, :3]), 1., atol=1e-6)
                    or abs(math.atan2(math.sin(yaw-math.atan2(matrix[1, 0], matrix[0, 0])),
                                      math.cos(yaw-math.atan2(matrix[1, 0], matrix[0, 0])))) > 1e-6):
                raise ValueError('invalid pose')
            return dict(pose)
        except (ValueError, KeyError, TypeError, OverflowError):
            return dict(valid=False, reason='视觉位姿格式、坐标系或时间戳不匹配')

    def set_pose_source(self, source, now):
        if source not in ('grasp', 'apriltag'):
            raise ValueError('Invalid cart pose source')
        if source == self.pose_source:
            return
        self.pose_source = source
        self.stamp = -float('inf')
        self.last_frame = {}
        self.last_packet_stamp = max(self.last_packet_stamp, now)
        self.cart_pose_urdf = dict(valid=False, reason='位姿方式已切换，等待新图像')
        self.reason = self.cart_pose_urdf['reason']

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

    @staticmethod
    def _missing_reason(reason):
        if reason == 'no_tags_detected':
            return '相机未识别到小车标签，请检查 ID 0/1 是否完整入镜'
        return '相机航向无效：' + str(reason or '没有有效的小车标签航向')

    def observation_status(self, now):
        frame = self.last_frame
        if self.fresh(now):
            return ('tracking', 'ok') if self.stamp == self.last_packet_stamp else ('holding', self.reason)
        if not frame:
            return 'waiting', self.reason
        if now-frame['received_monotonic'] > self.timeout:
            return 'frame_timeout', '相机新帧更新超时'
        if now-frame['captured_monotonic'] > self.timeout:
            return 'late_frame', '相机帧处理或传输延迟超过有效期限'
        return 'no_heading', self.reason

    def set_offset(self, offset):
        """Rebase the latest raw measurement without making its timestamp newer."""
        if self.pose_source == 'apriltag':
            raise ValueError('视觉位姿使用标签安装坐标，不使用手动航向零位')
        offset = float(offset)
        if not math.isfinite(offset):
            raise ValueError('航向零位必须是有限数值')
        self.offset = math.atan2(math.sin(offset), math.cos(offset))
        self.yaw = math.atan2(math.sin(self.raw_yaw - self.offset),
                              math.cos(self.raw_yaw - self.offset))

    def snapshot(self, now):
        status, reason = self.observation_status(now)
        return dict(fresh=self.fresh(now), yaw_rad=self.yaw if math.isfinite(self.stamp) else None,
                    raw_yaw_rad=self.raw_yaw if math.isfinite(self.stamp) else None,
                    yaw_offset_rad=0. if self.pose_source == 'apriltag' else self.offset,
                    captured_monotonic=self.stamp if math.isfinite(self.stamp) else None,
                    timeout_s=self.timeout, tag_ids=list(self.tag_ids),
                    age_s=now - self.stamp if math.isfinite(self.stamp) else None,
                    tags_in_pelvis=list(self.tags_in_pelvis),
                    pose_source=self.pose_source, cart_pose_urdf=dict(self.cart_pose_urdf),
                    reason=reason, status=status, rejected=self.rejected,
                    frame_age_s=now-self.last_frame['captured_monotonic'] if self.last_frame else None,
                    receive_age_s=now-self.last_frame['received_monotonic'] if self.last_frame else None,
                    timing=dict(self.last_frame.get('timing', {})))

    def close(self):
        self.sock.close()
