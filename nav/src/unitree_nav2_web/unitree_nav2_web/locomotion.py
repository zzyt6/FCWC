"""Loopback link to nav/locomotion/g1_locomotion_executor.py.

The executor runs our own Balance/Homie locomotion (the controller tested in
/home/unitree/CWC/Viser) and executes /cmd_vel from g1_cmd_vel_udp_node.  It
reports its stage about 10 times per second on ``status_port``; operator
lifecycle requests (acquire / prepare / start policy / zero / stop / reset)
are sent to the same loopback port that receives velocity packets.
"""
import json
import socket
import struct
import threading
import time

# Must match REQUEST_PACKET in nav/locomotion/g1_locomotion_executor.py.
REQUEST_PACKET = struct.Struct("<IHBBQ")
REQUEST_MAGIC = 0x47314C52
REQUEST_VERSION = 1
REQUEST_CODES = {
    "acquire": 1,
    "prepare": 2,
    "policy": 3,
    "zero": 4,
    "stop": 5,
    "reset": 6,
    "prepare_pose": 7,
}
CART_ACTIONS = frozenset(('hand_toggle', 'hand_compensation', 'hand_edit', 'hand_load', 'hand_save',
                          'cart_camera_start', 'cart_yaw_zero', 'pushcart_enable', 'pushcart_disable', 'cart_detach'))
# The request's payload byte selects a Viser pose; joint targets live in Viser.
PREPARE_POSE_CODES = {"original": 0, "arms_down": 1}
STATUS_MAGIC = "G1LOCO1"
STATUS_MAX_AGE_S = 0.5
# All 27 body and 14 hand joints may warn together, plus preparation details.
# Do not truncate their JSON and accidentally mark a healthy controller offline.
STATUS_MAX_PACKET_BYTES = 65535

STAGE_TEXT = {
    "WAITING": "等待接管底层",
    "ACQUIRING": "正在接管底层",
    "ZERO": "已接管，等待准备姿态",
    "PREPARING": "正在移到准备姿态",
    "HOLDING": "准备完成，等待启动策略",
    "POLICY": "Balance / Homie 策略运行",
    "DAMPED": "阻尼停止",
    "FAULT": "故障停止",
    "OBSERVE": "只读同步",
}


def locomotion_actions(status):
    """Return which lifecycle buttons are meaningful for this executor status."""
    if not status.get("online"):
        return {name: False for name in set(REQUEST_CODES) | CART_ACTIONS}
    stage = status.get("stage")
    healthy = bool(status.get("healthy"))
    stopped = stage in ("DAMPED", "FAULT")
    actions = {
        "acquire": healthy and stage == "WAITING",
        "prepare": healthy and stage == "ZERO",
        "prepare_pose": healthy and bool(status.get("prepare_pose_selectable")),
        "policy": healthy
        and stage == "HOLDING"
        and int(status.get("observation_frames", 0)) >= 6,
        "zero": stage == "POLICY",
        "stop": not stopped,
        "reset": stopped,
    }
    high, hands, cart = status.get('high_policy') or {}, status.get('hands') or {}, status.get('cart') or {}
    stationary = not status.get('decelerating') and not any(status.get('velocity') or [0, 0, 0])
    available = healthy and stationary and not stopped and bool(status.get('owned'))
    hand_idle = not any(hands.get(key) for key in ('moving', 'pending', 'recovering', 'recovery_pending', 'file_busy'))
    editing = available and hands.get('editor_available') and not high.get('enabled') and not hands.get('file_busy')
    actions.update({name: bool(editing) for name in ('hand_edit', 'hand_load', 'hand_save')})
    actions.update(
        hand_toggle=bool(available and stage in ('HOLDING', 'POLICY') and hands.get('healthy')
                         and status.get('armed') and not high.get('enabled') and hand_idle),
        hand_compensation=bool(available and status.get('armed') and stage in ('ZERO', 'HOLDING', 'POLICY')
                               and not high.get('enabled') and hand_idle
                               and (hands.get('compensation_enabled') or hands.get('compensation_available'))),
        cart_camera_start=bool(available and high and status.get('mode') != 'sim'
                               and not (high.get('camera') or {}).get('fresh')
                               and not (high.get('camera_process') or {}).get('running')
                               and not (high.get('camera_process') or {}).get('starting')),
        cart_yaw_zero=bool(available and high and not high.get('zero_reason')
                           and (high.get('camera') or {}).get('fresh')
                           and (high.get('zero_calibration') or {}).get('status') != 'collecting'),
        pushcart_enable=bool(available and high and not high.get('enabled') and not high.get('enable_reason') and cart.get('valid')),
        pushcart_disable=bool(available and (high.get('enabled') or high.get('motion_blocked'))),
        cart_detach=bool(available and cart.get('attached') and not high.get('enabled') and not high.get('motion_blocked')
                         and hand_idle and hands.get('healthy')
                         and hands.get('pose') not in ('grasp', 'grasp_compensated')))
    return actions


class LocomotionLink:
    def __init__(self, command_port, status_port, clock=time.monotonic):
        self.clock = clock
        self.destination = ("127.0.0.1", int(command_port))
        self.lock = threading.Lock()
        self.status = {}
        self.received = -float("inf")
        self.sequence = int(time.time() * 1000)
        self.sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self.sock.bind(("127.0.0.1", int(status_port)))
        except OSError:
            self.sock.close()
            self.sender.close()
            raise
        self.sock.settimeout(0.2)
        self.running = True
        self.thread = threading.Thread(
            target=self._receive, name="g1-locomotion-status", daemon=True
        )
        self.thread.start()

    def _receive(self):
        while self.running:
            try:
                data, sender = self.sock.recvfrom(STATUS_MAX_PACKET_BYTES)
            except socket.timeout:
                continue
            except OSError:
                return
            if sender[0] != "127.0.0.1":
                continue
            try:
                status = json.loads(data.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            if not isinstance(status, dict) or status.get("magic") != STATUS_MAGIC:
                continue
            with self.lock:
                self.status = status
                self.received = self.clock()

    def snapshot(self):
        now = self.clock()
        with self.lock:
            status = dict(self.status)
            age = now - self.received
        online = 0.0 <= age <= STATUS_MAX_AGE_S
        stage = status.get("stage") if online else None
        result = {
            "online": online,
            "age_s": None if self.received < 0 else round(age, 3),
            "ready": bool(online and status.get("ready")),
            "stage": stage,
            "stage_text": STAGE_TEXT.get(stage, "运控执行进程未连接")
            if online
            else "运控执行进程未连接",
            "healthy": bool(online and status.get("healthy")),
            "armed": bool(online and status.get("armed")),
            "owned": bool(online and status.get("owned")),
            "mode": status.get("mode") if online else None,
            "simulation": bool(online and status.get("simulation")),
            "reason": status.get("reason", "") if online else "",
            "prepare_pose": status.get("prepare_pose") if online else None,
            "prepare_pose_options": status.get("prepare_pose_options", []) if online else [],
            "prepare_pose_selectable": bool(online and status.get("prepare_pose_selectable")),
            "preparation_warning": status.get("preparation_warning", "") if online else "",
            "warnings": status.get("warnings", []) if online else [],
            "dex3_ready": bool(online and status.get("dex3_ready")),
            "dex3_reason": status.get("dex3_reason", "") if online else "",
            "observation_frames": int(status.get("observation_frames", 0))
            if online
            else 0,
            "selected_model": status.get("selected_model") if online else None,
            "decelerating": bool(online and status.get("decelerating")),
            "velocity": status.get("velocity") if online else None,
            "nav_target": status.get("nav_target") if online else None,
            "nav_command_fresh": bool(online and status.get("nav_command_fresh")),
            "heading": status.get("heading") if online else None,
            "high_policy": status.get("high_policy", {}) if online else {},
            "hands": status.get("hands", {}) if online else {},
            "cart": status.get("cart", {}) if online else {},
            "pushcart_error": status.get("pushcart_error", "") if online else "",
            "cart_request": status.get("cart_request", {}) if online else {},
            "upper_pd_transitioning": bool(online and status.get("upper_pd_transitioning")),
        }
        result["actions"] = locomotion_actions(result)
        return result

    def request(self, kind, *, pose=None, **payload):
        if kind in CART_ACTIONS:
            with self.lock:
                self.sequence += 1
                packet = json.dumps(dict(magic='G1NAVCTL1', sequence=self.sequence, action=kind,
                                         payload=payload, sent_monotonic=self.clock()), allow_nan=False).encode()
                self.sender.sendto(packet, self.destination)
                return self.sequence
        if kind not in REQUEST_CODES:
            raise ValueError(f"未知运控请求：{kind}")
        payload = 0
        if kind == "prepare_pose":
            if not isinstance(pose, str) or pose not in PREPARE_POSE_CODES:
                raise ValueError("请选择有效的准备姿态")
            payload = PREPARE_POSE_CODES[pose]
        elif pose is not None:
            raise ValueError("该运控请求不接受姿态参数")
        with self.lock:
            self.sequence += 1
            sequence = self.sequence
            packet = REQUEST_PACKET.pack(
                REQUEST_MAGIC, REQUEST_VERSION, REQUEST_CODES[kind], payload, sequence
            )
            self.sender.sendto(packet, self.destination)

    def close(self):
        self.running = False
        self.sock.close()
        self.sender.close()
        self.thread.join(timeout=1.0)
