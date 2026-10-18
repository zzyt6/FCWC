"""Navigation safety regressions for the normal-navigation bridge.

These tests exercise bridge methods on small harnesses without a ROS graph,
and the loopback locomotion link with real UDP sockets on free ports.
"""
import json
import socket
import threading
import time
from concurrent.futures import Future
from types import SimpleNamespace

import pytest
import yaml
from pathlib import Path
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry

from unitree_nav2_web.locomotion import (
    REQUEST_CODES,
    REQUEST_MAGIC,
    REQUEST_PACKET,
    PREPARE_POSE_CODES,
    MODEL_SET_CODES,
    LocomotionLink,
    locomotion_actions,
)
from unitree_nav2_web.nav_web_bridge import (
    LOCOMOTION_READY_GRACE_S,
    REFERENCE_READY_GRACE_S,
    NavigationWebBridge,
)

PACKAGE = Path(__file__).resolve().parents[1]


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class _Logger:
    def __init__(self):
        self.errors = []

    def error(self, message):
        self.errors.append(str(message))

    def warning(self, message):
        pass


class _Publisher:
    def __init__(self):
        self.messages = []

    def publish(self, message):
        self.messages.append(message)


class _GoalHandle:
    def __init__(self):
        self.cancel_count = 0

    def cancel_goal_async(self):
        self.cancel_count += 1


class _AliveProcess:
    def poll(self):
        return None


class _Locomotion:
    def __init__(self, ready=True, stage="POLICY", online=True):
        self.value = {
            "online": online,
            "ready": ready,
            "stage": stage,
            "stage_text": stage,
            "healthy": True,
            "cart": {"epoch": 2},
        }
        self.value["actions"] = locomotion_actions(self.value)
        self.requests = []

    def snapshot(self):
        return dict(self.value)

    def request(self, kind, *, pose=None, model_set=None):
        selection = model_set if model_set is not None else pose
        self.requests.append((kind, selection) if selection is not None else kind)


class _Harness:
    _cancel_goal_safely = NavigationWebBridge._cancel_goal_safely
    _validate_motion_enable = NavigationWebBridge._validate_motion_enable
    _validate_speed_ready = NavigationWebBridge._validate_speed_ready
    _abort_navigation_for_motion_gate = (
        NavigationWebBridge._abort_navigation_for_motion_gate
    )
    _reference_blockers = NavigationWebBridge._reference_blockers
    _check_reference_gate = NavigationWebBridge._check_reference_gate
    _locomotion_blockers = NavigationWebBridge._locomotion_blockers
    _status_timer = NavigationWebBridge._status_timer
    _validate_navigation_start = NavigationWebBridge._validate_navigation_start
    _motion_callback = NavigationWebBridge._motion_callback
    _stop_for_navigation_reference = NavigationWebBridge._stop_for_navigation_reference
    _navigation_diagnostic = NavigationWebBridge._navigation_diagnostic
    _record_navigation_failure = NavigationWebBridge._record_navigation_failure
    _nomotion_update_timer = NavigationWebBridge._nomotion_update_timer
    _set_motion_transaction = NavigationWebBridge._set_motion_transaction
    request_locomotion = NavigationWebBridge.request_locomotion

    def __init__(self, locomotion=None):
        now = time.monotonic()
        self.navigation_reference = {"ready": True, "epoch": 2}
        self.reference_received = now
        self.reference_not_ready_since = None
        self.lock = threading.RLock()
        self.navigation_start_lock = threading.Lock()
        self.logger = _Logger()
        self.locomotion = locomotion or _Locomotion()
        self.navigation_speed = type("Speed", (), {
            "snapshot": lambda _: {"ready": True, "pending": False}
        })()
        self.motion_enable_pub = _Publisher()
        self.map_ready = True
        self.localized = True
        self.localization_fault = ""
        self.localization_refinement_active = False
        self.motion_enabled = True
        self.motion_requested = True
        self.motion_request_generation = 0
        self.navigation_active = True
        self.navigation_generation = 3
        self.navigation_distance = 1.0
        self.navigation_started_time = now
        self.navigation_pose_received = True
        self.navigation_motion_started = True
        self.navigation_last_motion_time = now
        self.navigation_task_state = "ACTIVE"
        self.navigation_result_code = None
        self.navigation_completed_count = 0
        self.last_cloud_time = now
        self.last_odom_time = now
        self.last_localization_scan_time = now
        self.last_odom_speed = self.last_odom_angular_speed = 0.0
        self.last_pose_time = now
        self.nomotion_update_inflight = False
        self.next_nomotion_update_time = now
        self.amcl_generation = 0
        self.last_navigation_diagnostic = None
        self.costmap_clear_pending = {}
        self.footprint_confirmed = False
        self.footprint_preparing = False
        self.footprint_error = ""
        self.next_footprint_attempt = 0.0
        self.footprint_thread = None
        self.processing = False
        self.initial_pose_time = 0.0
        self.amcl_process = _AliveProcess()
        self.phase = "NAVIGATING"
        self.message = ""
        self.error = ""
        self.last_safety_event = ""
        self.goal_handle = _GoalHandle()
        self.locomotion_not_ready_since = 0.0
        self.motion_commands = []
        self.status_publishes = 0
        self.faults = []

    def get_logger(self):
        return self.logger

    def _set_motion(self, enabled):
        self.motion_commands.append(bool(enabled))
        self.motion_enabled = bool(enabled)
        self.motion_requested = bool(enabled)

    def _publish_status(self):
        self.status_publishes += 1

    def _handle_localization_fault(self, reason):
        self.faults.append(reason)


def test_locomotion_actions_follow_viser_lifecycle():
    offline = locomotion_actions({"online": False})
    assert not any(offline.values())
    waiting = locomotion_actions({"online": True, "stage": "WAITING", "healthy": True})
    assert waiting["acquire"] and not waiting["prepare"] and not waiting["policy"]
    zero = locomotion_actions(
        {"online": True, "stage": "ZERO", "healthy": True, "dex3_ready": False}
    )
    assert zero["prepare"]  # Hand faults must not block body preparation.
    assert not zero["prepare_pose"]
    selectable = locomotion_actions({"online": True, "stage": "ZERO", "healthy": True,
                                     "prepare_pose_selectable": True})
    assert selectable["prepare_pose"]
    holding = locomotion_actions(
        {"online": True, "stage": "HOLDING", "healthy": True, "observation_frames": 6}
    )
    assert holding["policy"] and holding["stop"] and not holding["reset"]
    damped = locomotion_actions({"online": True, "stage": "DAMPED", "healthy": False})
    assert damped["reset"] and not damped["stop"] and not damped["acquire"]


def test_locomotion_link_reports_freshness_and_sends_requests():
    command_port = _free_port()
    status_port = _free_port()
    receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receiver.bind(("127.0.0.1", command_port))
    receiver.settimeout(1.0)
    link = LocomotionLink(command_port, status_port)
    sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        assert not link.snapshot()["online"]
        sender.sendto(b"not json", ("127.0.0.1", status_port))
        sender.sendto(
            json.dumps({"magic": "WRONG", "ready": True}).encode(),
            ("127.0.0.1", status_port),
        )
        status = {"magic": "G1LOCO1", "ready": True, "stage": "POLICY", "healthy": True,
                  "prepare_pose": "arms_down", "prepare_pose_selectable": False,
                  "prepare_pose_options": [{"value": "arms_down", "label": "双臂下垂"}],
                  "preparation_warning": "等待关节到位"}
        sender.sendto(json.dumps(status).encode(), ("127.0.0.1", status_port))
        deadline = time.monotonic() + 1.0
        while not link.snapshot()["online"] and time.monotonic() < deadline:
            time.sleep(0.01)
        snapshot = link.snapshot()
        assert snapshot["online"] and snapshot["ready"] and snapshot["stage"] == "POLICY"
        assert snapshot["prepare_pose"] == "arms_down"
        assert snapshot["prepare_pose_options"] == status["prepare_pose_options"]
        assert not snapshot["actions"]["prepare_pose"]
        assert snapshot["preparation_warning"] == "等待关节到位"

        link.request("policy")
        link.request("stop")
        packets = [REQUEST_PACKET.unpack(receiver.recv(64)) for _ in range(2)]
        assert [packet[0] for packet in packets] == [REQUEST_MAGIC, REQUEST_MAGIC]
        assert [packet[2] for packet in packets] == [
            REQUEST_CODES["policy"],
            REQUEST_CODES["stop"],
        ]
        assert packets[1][4] > packets[0][4]
        for pose, code in PREPARE_POSE_CODES.items():
            link.request("prepare_pose", pose=pose)
            packet = REQUEST_PACKET.unpack(receiver.recv(64))
            assert packet[2:4] == (REQUEST_CODES["prepare_pose"], code)
            assert packet[4] > packets[-1][4]
            packets.append(packet)
        for pose in (None, "unknown", 1):
            with pytest.raises(ValueError):
                link.request("prepare_pose", pose=pose)
        for key, code in MODEL_SET_CODES.items():
            link.request('model_set', model_set=key)
            packet = REQUEST_PACKET.unpack(receiver.recv(64))
            assert packet[2:4] == (REQUEST_CODES['model_set'], code)
        for key in (None, 'unknown', 1):
            with pytest.raises(ValueError):
                link.request('model_set', model_set=key)
        with pytest.raises(ValueError):
            link.request("prepare", pose="arms_down")
        with pytest.raises(ValueError):
            link.request("dance")

        link.received = time.monotonic() - 1.0
        stale = link.snapshot()
        assert not stale["online"] and not stale["ready"]
        assert stale["prepare_pose"] is None and not stale["prepare_pose_selectable"]
    finally:
        link.close()
        sender.close()
        receiver.close()


def test_motion_gate_blocks_until_our_locomotion_runs():
    harness = _Harness(_Locomotion(online=False, ready=False, stage=None))
    assert harness._locomotion_blockers(harness.locomotion.snapshot())
    harness.locomotion = _Locomotion(ready=False, stage="HOLDING")
    assert "策略运行" in harness._locomotion_blockers(harness.locomotion.snapshot())[0]
    harness.locomotion = _Locomotion(ready=True)
    assert harness._locomotion_blockers(harness.locomotion.snapshot()) == []


def test_policy_stage_reports_grasp_failure_instead_of_wrong_body_stage():
    h = _Harness(_Locomotion(ready=False))
    h.locomotion.value['high_policy'] = dict(motion_blocked=True, reason='抓握已中断；高层暂停')
    assert h._locomotion_blockers(h.locomotion.snapshot()) == ['抓握已中断；高层暂停']
    h.locomotion.value['high_policy'] = dict(enabled=False)
    h.locomotion.value['cart']['attached'] = True
    assert '小车仍处于连接状态' in h._locomotion_blockers(h.locomotion.snapshot())[0]


def test_goal_rejected_when_locomotion_not_running():
    harness = _Harness(_Locomotion(ready=False, stage="DAMPED"))
    harness.navigation_active = False
    with pytest.raises(ValueError, match="策略运行"):
        harness._validate_navigation_start()


def test_normal_deceleration_does_not_abort_active_navigation():
    harness = _Harness()
    harness.locomotion.value["decelerating"] = True
    harness.locomotion_not_ready_since = time.monotonic() - 2.0
    harness._status_timer()
    assert harness.navigation_active
    assert harness.goal_handle.cancel_count == 0
    assert harness.motion_enable_pub.messages[-1].data is True
    assert harness._locomotion_blockers(harness.locomotion.snapshot())


def test_joint_warnings_do_not_close_motion_gate_or_cancel_navigation():
    harness = _Harness()
    harness.locomotion.value["warnings"] = [
        {"level": "warning", "message": "right_ankle_roll_joint 位置超限", "active": True},
        {"level": "warning", "message": "左手 Dex3 未就绪", "active": True},
    ]
    for _ in range(3):
        harness._status_timer()
    assert harness.navigation_active and harness.motion_enabled
    assert harness.goal_handle.cancel_count == 0
    assert not harness.locomotion_not_ready_since
    assert harness._locomotion_blockers(harness.locomotion.snapshot()) == []
    assert harness.locomotion.requests == []


@pytest.mark.parametrize("invalid", ["position", "quaternion", "covariance_y", "covariance_yaw"])
def test_invalid_localization_closes_gate_without_damping(invalid):
    harness = _Harness()
    harness._handle_localization_fault = NavigationWebBridge._handle_localization_fault.__get__(harness)
    harness.pose_sequence = harness.navigation_pose_sequence_start = 0
    harness.last_localization_sample = harness.initial_pose_request = None
    harness.bad_localization_updates = 0
    harness.pose_history = []
    harness._map_match_for_pose = lambda *args: {"good": True}
    original_time = harness.last_pose_time
    message = PoseWithCovarianceStamped()
    message.pose.pose.orientation.w = 1.
    if invalid == "position":
        message.pose.pose.position.x = float('nan')
    elif invalid == "quaternion":
        message.pose.pose.orientation.w = 0.
    elif invalid == "covariance_y":
        message.pose.covariance[7] = float('nan')
    else:
        message.pose.covariance[35] = float('inf')
    NavigationWebBridge._localization_callback(harness, message)
    assert not harness.motion_enabled and not harness.navigation_active
    assert harness.locomotion.requests == []
    assert harness.last_pose_time == original_time
    assert harness.pose_sequence == 0


@pytest.mark.parametrize("invalid", ["position", "quaternion", "linear", "angular"])
def test_invalid_navigation_odometry_cannot_renew_sensor_health(invalid):
    harness = _Harness()
    harness._handle_localization_fault = NavigationWebBridge._handle_localization_fault.__get__(harness)
    original_time = harness.last_odom_time
    message = Odometry()
    message.pose.pose.orientation.w = 1.
    if invalid == "position":
        message.pose.pose.position.y = float('nan')
    elif invalid == "quaternion":
        message.pose.pose.orientation.w = 0.
    elif invalid == "linear":
        message.twist.twist.linear.x = float('nan')
    else:
        message.twist.twist.angular.z = float('inf')
    NavigationWebBridge._odom_callback(harness, message)
    assert harness.last_odom_time == original_time
    assert not harness.motion_enabled and not harness.navigation_active
    assert harness.locomotion.requests == []


def test_large_warning_status_remains_online_and_keeps_all_joint_details():
    link = LocomotionLink(_free_port(), 0)
    sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    warnings = [{"level": "warning", "active": True,
                 "message": f"joint_{i} 位置超限：实测超出参考范围，机身控制继续。" * 3}
                for i in range(41)]
    try:
        packet = json.dumps({"magic": "G1LOCO1", "ready": True, "healthy": True,
                             "stage": "POLICY", "warnings": warnings}, ensure_ascii=False).encode()
        assert len(packet) > 8192
        sender.sendto(packet, link.sock.getsockname())
        deadline = time.monotonic() + 1.
        while not link.snapshot()["online"] and time.monotonic() < deadline:
            time.sleep(.01)
        snapshot = link.snapshot()
        assert snapshot["online"] and snapshot["ready"]
        assert snapshot["warnings"] == warnings
    finally:
        link.close()
        sender.close()


def test_new_goal_waits_until_deceleration_finishes():
    harness = _Harness()
    harness.navigation_active = False
    harness.locomotion.value["decelerating"] = True
    with pytest.raises(ValueError, match="平滑归零"):
        harness._validate_navigation_start()
    harness.locomotion.value["decelerating"] = False
    harness._validate_navigation_start()


def test_successful_arrival_closes_gate_and_requests_smooth_zero_once():
    harness = _Harness()
    generation = harness.navigation_generation
    NavigationWebBridge._navigation_finished(
        harness, "done", "", generation, task_state="SUCCEEDED", result_code=4)
    assert not harness.navigation_active and not harness.motion_enabled
    assert not harness.motion_requested
    assert harness.navigation_task_state == "SUCCEEDED"
    assert harness.navigation_completed_count == 1
    assert harness.locomotion.requests == ["zero"]
    assert harness.locomotion.snapshot()["ready"]  # No stop/reset/damping.
    with pytest.raises(ValueError, match="安全门尚未启用"):
        harness._validate_navigation_start()
    NavigationWebBridge._navigation_finished(
        harness, "late duplicate", "", generation, task_state="SUCCEEDED", result_code=4)
    assert harness.navigation_completed_count == 1
    assert harness.locomotion.requests == ["zero"]


def test_old_arrival_callback_cannot_stop_a_new_goal():
    harness = _Harness()
    NavigationWebBridge._navigation_finished(
        harness, "old arrival", "", harness.navigation_generation - 1,
        task_state="SUCCEEDED", result_code=4)
    assert harness.navigation_active and harness.motion_enabled
    assert harness.locomotion.requests == []


def test_arrival_gate_stays_closed_if_explicit_zero_packet_fails():
    harness = _Harness()

    def unavailable(_):
        raise OSError("test socket unavailable")

    harness.locomotion.request = unavailable
    NavigationWebBridge._navigation_finished(
        harness, "done", "", harness.navigation_generation,
        task_state="SUCCEEDED", result_code=4)
    assert not harness.motion_enabled and not harness.motion_requested
    assert harness.navigation_task_state == "SUCCEEDED"


def test_late_gate_acknowledgement_cannot_reopen_without_locomotion():
    harness = _Harness(_Locomotion(ready=False, stage="DAMPED"))
    harness.navigation_active = False

    class Message:
        data = True

    harness._motion_callback(Message())
    assert not harness.motion_enabled
    assert harness.motion_enable_pub.messages[-1].data is False


def test_late_enable_acknowledgement_cannot_undo_a_user_stop():
    harness = _Harness()
    harness.navigation_active = False
    harness.motion_requested = harness.motion_enabled = False
    harness._motion_callback(type("Message", (), {"data": True})())
    assert not harness.motion_enabled and not harness.motion_requested
    assert harness.motion_enable_pub.messages[-1].data is False


@pytest.mark.parametrize("reason", ["localization", "gate", "user"])
def test_cancel_failure_cannot_skip_gate_close_or_stop_body_controller(reason):
    harness = _Harness()

    class FailedGoal:
        def cancel_goal_async(self):
            assert not harness.motion_enabled and not harness.motion_requested
            raise RuntimeError("action server unavailable")

    harness.goal_handle = FailedGoal()
    if reason == "localization":
        NavigationWebBridge._handle_localization_fault(harness, "定位丢失")
    elif reason == "gate":
        harness._abort_navigation_for_motion_gate("运动门关闭")
    else:
        NavigationWebBridge.cancel_navigation(harness)
    assert not harness.navigation_active
    assert harness.motion_commands[-1] is False
    assert harness.locomotion.requests == []  # Never request damping or reset.
    assert any("取消请求失败" in error for error in harness.logger.errors)


def test_enable_authorization_is_committed_before_dds_acknowledgement():
    harness = _Harness()
    harness.navigation_active = False
    harness.motion_requested = harness.motion_enabled = False
    harness._confirm_footprint = lambda: None

    class ImmediateAck:
        def publish(self, message):
            assert harness.motion_requested is bool(message.data)
            harness._motion_callback(message)

    harness.motion_enable_pub = ImmediateAck()
    NavigationWebBridge._set_motion(harness, True)
    assert harness.motion_enabled and harness.motion_requested
    NavigationWebBridge._set_motion(harness, False)
    assert not harness.motion_enabled and not harness.motion_requested


def test_stop_during_footprint_confirmation_cannot_be_undone_by_pending_enable():
    harness = _Harness()
    harness.navigation_active = False
    harness._confirm_footprint = lambda: NavigationWebBridge._set_motion(harness, False)
    with pytest.raises(ValueError, match="本次请求已取消"):
        NavigationWebBridge._set_motion(harness, True)
    assert not harness.motion_enabled and not harness.motion_requested
    assert [msg.data for msg in harness.motion_enable_pub.messages] == [False]


def test_rejected_concurrent_enable_cannot_cancel_a_stop():
    harness = _Harness()
    harness.navigation_active = False
    harness.last_cloud_time -= 3.
    released = threading.Event()
    resume = threading.Event()
    errors = []

    class InterleavedLock:
        """Schedule another HTTP request after the stop's first lock release."""
        def __init__(self):
            self.lock = threading.RLock()

        def __enter__(self):
            self.lock.acquire()

        def __exit__(self, *args):
            self.lock.release()
            if threading.current_thread() is worker and not released.is_set():
                released.set()
                assert resume.wait(2.)

    def stop():
        try:
            NavigationWebBridge._set_motion(harness, False)
        except Exception as error:
            errors.append(error)

    harness.lock = InterleavedLock()
    worker = threading.Thread(target=stop)
    worker.start()
    try:
        assert released.wait(2.)
        with pytest.raises(ValueError, match="点云或里程计"):
            NavigationWebBridge._set_motion(harness, True)
    finally:
        resume.set()
        worker.join(2.)
    assert not worker.is_alive()
    assert errors == []
    assert not harness.motion_enabled and not harness.motion_requested
    assert [msg.data for msg in harness.motion_enable_pub.messages] == [False]


@pytest.mark.parametrize("lost", ["sensor", "locomotion"])
def test_gate_conditions_are_rechecked_after_footprint_confirmation(lost):
    harness = _Harness()
    harness.navigation_active = False
    harness.motion_requested = harness.motion_enabled = False

    def confirmation():
        if lost == "sensor":
            harness.last_cloud_time -= 2.
        else:
            harness.locomotion.value["ready"] = False

    harness._confirm_footprint = confirmation
    with pytest.raises(ValueError):
        NavigationWebBridge._set_motion(harness, True)
    assert harness.motion_enable_pub.messages == []
    assert not harness.motion_enabled and not harness.motion_requested


def test_status_timer_aborts_navigation_after_locomotion_grace():
    harness = _Harness(_Locomotion(ready=False, stage="DAMPED"))
    harness._status_timer()
    assert harness.navigation_active
    assert harness.goal_handle.cancel_count == 0
    harness.locomotion_not_ready_since -= LOCOMOTION_READY_GRACE_S + 0.05
    handle = harness.goal_handle
    harness._status_timer()
    assert not harness.navigation_active
    assert handle.cancel_count == 1
    assert harness.motion_commands[-1] is False
    assert "DAMPED" in harness.last_safety_event


def test_stop_request_cancels_navigation_before_forwarding():
    harness = _Harness()
    handle = harness.goal_handle
    harness.request_locomotion({"action": ["stop"]})
    assert not harness.navigation_active
    assert handle.cancel_count == 1
    assert harness.locomotion.requests == ["stop"]


def test_lifecycle_request_respects_executor_stage():
    harness = _Harness(_Locomotion(ready=False, stage="WAITING"))
    harness.navigation_active = False
    with pytest.raises(ValueError):
        harness.request_locomotion({"action": ["policy"]})
    harness.request_locomotion({"action": ["acquire"]})
    assert harness.locomotion.requests == ["acquire"]


def test_preparation_selection_requires_executor_permission_and_known_pose():
    harness = _Harness(_Locomotion(ready=False, stage="ZERO"))
    harness.navigation_active = False
    data = harness.locomotion.value
    data.update(prepare_pose_selectable=True,
                prepare_pose_options=[{"value": key} for key in PREPARE_POSE_CODES])
    data["actions"] = locomotion_actions(data)
    for pose in PREPARE_POSE_CODES:
        harness.request_locomotion({"action": ["prepare_pose"], "pose": [pose]})
    assert harness.locomotion.requests == [("prepare_pose", pose) for pose in PREPARE_POSE_CODES]
    assert harness.motion_commands == []
    with pytest.raises(ValueError, match="有效的准备姿态"):
        harness.request_locomotion({"action": ["prepare_pose"], "pose": ["unknown"]})
    with pytest.raises(ValueError, match="有效的准备姿态"):
        harness.request_locomotion({"action": ["prepare_pose"]})
    data.update(stage="PREPARING", prepare_pose_selectable=False)
    data["actions"] = locomotion_actions(data)
    with pytest.raises(ValueError, match="不允许"):
        harness.request_locomotion({"action": ["prepare_pose"], "pose": ["original"]})
    data["online"] = False
    with pytest.raises(ValueError, match="未连接"):
        harness.request_locomotion({"action": ["prepare_pose"], "pose": ["original"]})


def test_model_selection_validates_stage_and_model_name():
    harness = _Harness(_Locomotion(ready=False, stage='WAITING'))
    harness.navigation_active = False
    data = harness.locomotion.value
    data.update(model_set_selectable=True, model_set_options=[{'value': key} for key in MODEL_SET_CODES])
    data['actions'] = locomotion_actions(data)
    for key in MODEL_SET_CODES:
        harness.request_locomotion({'action': ['model_set'], 'model_set': [key]})
    assert harness.locomotion.requests == [('model_set', key) for key in MODEL_SET_CODES]
    for key in ('', 'unknown'):
        with pytest.raises(ValueError, match='模型组合'):
            harness.request_locomotion({'action': ['model_set'], 'model_set': [key]})
    for stage in ('PREPARING', 'HOLDING', 'POLICY', 'DAMPED'):
        data['stage'] = stage
        data['actions'] = locomotion_actions(data)
        with pytest.raises(ValueError, match='不允许'):
            harness.request_locomotion({'action': ['model_set'], 'model_set': ['v81']})


def test_nav2_parameters_keep_acceleration_bounds_and_allow_web_speed_range():
    params = yaml.safe_load((PACKAGE / "config" / "nav2_foxy_g1.yaml").read_text())
    controller = params["controller_server"]["ros__parameters"]
    assert controller["controller_plugins"] == ["FollowPath", "CartPush"]
    follow = controller["FollowPath"]
    # DWB prediction keeps its existing bounds; locomotion uses a 0.4 s filter.
    assert follow["acc_lim_x"] <= 0.3 and follow["acc_lim_y"] <= 0.2
    assert follow["acc_lim_theta"] <= 0.4 and follow["max_angular_accel"] <= 0.4
    # The web range is 0.20–0.60 m/s; the launch's UDP cap must permit it.
    assert follow["max_vel_x"] == 0.60 and follow["max_vel_y"] <= 0.20
    assert follow["max_speed_xy"] == 0.60
    assert follow["max_vel_theta"] <= 0.50
    assert controller["goal_checker"]["xy_goal_tolerance"] == follow["xy_goal_tolerance"] == 0.15
    assert controller["goal_checker"]["yaw_goal_tolerance"] == pytest.approx(0.17453292519943295)
    assert follow["primary_controller"] == "unitree_nav2_core::ApproachDWBLocalPlanner"
    assert follow["approach_distance"] == 0.6 and follow["approach_speed"] == 0.1
    planner = params["planner_server"]["ros__parameters"]
    assert planner["planner_plugins"] == ["GridBased"]


@pytest.mark.parametrize('failure', ['epoch', 'stale', 'unready'])
def test_cart_reference_must_match_executor_before_opening_gate(failure):
    harness = _Harness()
    harness.navigation_active = False
    if failure == 'epoch':
        harness.navigation_reference['epoch'] += 2
    elif failure == 'stale':
        harness.reference_received -= 1.
    else:
        harness.navigation_reference['ready'] = False
    with pytest.raises(ValueError, match='导航参考点'):
        harness._validate_motion_enable()


def test_cart_mode_change_closes_gate_and_invalidates_footprint_confirmation():
    harness = _Harness()
    harness._set_motion = NavigationWebBridge._set_motion.__get__(harness)
    harness.footprint_confirmed = True
    message = type('Message', (), {'data': json.dumps(dict(
        epoch=3, ready=True, sent_monotonic=time.monotonic()))})()
    NavigationWebBridge._reference_callback(harness, message)
    assert not harness.footprint_confirmed
    assert not harness.motion_enabled and not harness.navigation_active


def reference_preparation_harness():
    h = _Harness()
    h.motion_enabled = h.motion_requested = h.navigation_active = False
    h.navigation_reference['footprint'] = [[-.3, -.25], [.3, -.25], [.3, .25], [-.3, .25]]
    return h


def test_ready_reference_rebuilds_costmaps_before_any_motion_request():
    h = reference_preparation_harness()
    calls = []
    def confirm():
        calls.append('confirm_and_rebuild')
        assert not h.motion_enabled and not h.motion_requested
        h.footprint_confirmed = True
    h._confirm_footprint = confirm
    NavigationWebBridge._prepare_reference_timer(h)
    h.footprint_thread.join(timeout=1.)
    assert calls == ['confirm_and_rebuild']
    assert not h.footprint_preparing and not h.footprint_error
    assert not h.motion_commands and not h.motion_enable_pub.messages
    NavigationWebBridge._prepare_reference_timer(h)
    assert len(calls) == 1  # Rebuild once per reference, not every timer tick.


@pytest.mark.parametrize('blocked', ['motion_enabled', 'motion_requested', 'navigation_active',
                                     'processing', 'not_localized', 'no_map', 'stale', 'not_ready'])
def test_reference_preparation_waits_for_stationary_ready_navigation(blocked):
    h = reference_preparation_harness()
    if blocked == 'stale':
        h.reference_received -= 1.
    elif blocked == 'not_ready':
        h.navigation_reference['ready'] = False
    elif blocked == 'not_localized':
        h.localized = False
    elif blocked == 'no_map':
        h.map_ready = False
    else:
        setattr(h, blocked, True)
    NavigationWebBridge._prepare_reference_timer(h)
    assert h.footprint_thread is None and not h.footprint_preparing


def test_stop_cancels_queued_reference_preparation():
    h = reference_preparation_harness()
    calls = []
    h._confirm_footprint = lambda: calls.append('unexpected clear')
    with h.navigation_start_lock:
        NavigationWebBridge._prepare_reference_timer(h)
        NavigationWebBridge._set_motion_transaction(h, False)
    h.footprint_thread.join(timeout=1.)
    assert not calls and not h.footprint_preparing
    assert all(not m.data for m in h.motion_enable_pub.messages)


def test_failed_reference_preparation_keeps_motion_closed_and_retries():
    h = reference_preparation_harness()
    calls = []
    def confirm():
        calls.append(1)
        raise ValueError('等待新扫描')
    h._confirm_footprint = confirm
    NavigationWebBridge._prepare_reference_timer(h)
    h.footprint_thread.join(timeout=1.)
    assert h.footprint_error == '等待新扫描' and not h.footprint_confirmed
    assert not h.motion_requested and not h.motion_enable_pub.messages
    NavigationWebBridge._prepare_reference_timer(h)
    assert len(calls) == 1
    h.next_footprint_attempt = 0.
    NavigationWebBridge._prepare_reference_timer(h)
    h.footprint_thread.join(timeout=1.)
    assert len(calls) == 2


def test_navigation_refreshes_quiet_amcl_without_faking_pose_freshness(monkeypatch):
    harness = _Harness()
    now = time.monotonic()
    monkeypatch.setattr(time, 'monotonic', lambda: now)
    harness.last_pose_time = now-.7
    requests = []
    pending = Future()
    harness.nomotion_update_client = SimpleNamespace(
        service_is_ready=lambda: True,
        call_async=lambda req: requests.append(req) or pending)
    harness._nomotion_update_timer()
    assert len(requests) == 1 and harness.nomotion_update_inflight
    assert harness.last_pose_time == now-.7
    harness._nomotion_update_timer()
    assert len(requests) == 1  # No overlapping requests.
    pending.set_result(SimpleNamespace())
    harness._nomotion_update_timer()
    assert len(requests) == 1  # Respect the existing update interval.
    harness.next_nomotion_update_time = now-1.
    harness.last_pose_time = now  # A real AMCL update makes another request unnecessary.
    harness._nomotion_update_timer()
    assert len(requests) == 1


def test_amcl_timeout_remains_active_while_update_request_is_pending():
    harness = _Harness()
    harness.last_pose_time -= 2.
    harness.nomotion_update_inflight = True
    harness._status_timer()
    assert harness.faults == ['机器人运动中 AMCL 位姿连续超时 1.5 秒']


def test_missing_amcl_scan_is_reported_before_missing_pose():
    harness = _Harness()
    harness.last_localization_scan_time -= 2.
    harness.last_pose_time -= 2.
    harness._status_timer()
    assert harness.faults == ['AMCL 输入扫描 /scan_localization 连续超时 1.5 秒']


def test_cart_tracking_loss_cancels_task_preserves_localization_and_cannot_auto_resume(monkeypatch):
    harness = _Harness()
    harness._set_motion = NavigationWebBridge._set_motion.__get__(harness)
    handle = harness.goal_handle
    stamp = time.monotonic()
    now = [stamp]
    monkeypatch.setattr(time, 'monotonic', lambda: now[0])
    message = SimpleNamespace(data=json.dumps(dict(
        epoch=2, mode='cart', ready=False, reason='小车 AprilTag 航向不可用', sent_monotonic=stamp)))
    NavigationWebBridge._reference_callback(harness, message)
    assert harness.motion_enabled and harness.navigation_active
    now[0] += REFERENCE_READY_GRACE_S + .01
    message.data = json.dumps(dict(epoch=2, mode='cart', ready=False,
                                   reason='小车 AprilTag 航向不可用', sent_monotonic=now[0]))
    NavigationWebBridge._reference_callback(harness, message)
    assert not harness.motion_enabled and not harness.navigation_active
    assert harness.localized and not harness.localization_fault
    assert handle.cancel_count == 1
    assert 'AprilTag' in harness.last_safety_event
    assert harness.last_navigation_diagnostic['mode'] == 'cart'
    now[0] += .001
    message.data = json.dumps(dict(epoch=2, mode='cart', ready=True, sent_monotonic=now[0]))
    NavigationWebBridge._reference_callback(harness, message)
    assert not harness.motion_enabled and not harness.navigation_active
    assert handle.cancel_count == 1


def test_brief_reference_gaps_preserve_task_and_reset_loss_timer(monkeypatch):
    now = [time.monotonic()]
    monkeypatch.setattr(time, 'monotonic', lambda: now[0])
    h = _Harness()
    for _ in range(3):
        for ready in (False, True):
            now[0] += .1
            NavigationWebBridge._reference_callback(h, SimpleNamespace(data=json.dumps(dict(
                epoch=2, ready=ready, sent_monotonic=now[0], reason='' if ready else '等待扫描'))))
            h._status_timer()
            assert h.navigation_active and h.motion_enabled
            assert not h.goal_handle.cancel_count
        assert h.reference_not_ready_since is None


def test_lease_ack_during_brief_feedback_gap_does_not_cancel_running_task():
    h = _Harness(_Locomotion(ready=False))
    h._motion_callback(SimpleNamespace(data=True))
    assert h.motion_enabled and h.navigation_active
    h._status_timer()
    h.locomotion_not_ready_since -= LOCOMOTION_READY_GRACE_S + .01
    h._status_timer()
    assert not h.motion_enabled and not h.navigation_active


def test_downstream_reference_fault_is_acknowledged_even_without_a_goal():
    h = _Harness()
    h.navigation_active = False
    h.navigation_reference.update(ready=False, reason='等待扫描')
    h._motion_callback(SimpleNamespace(data=False))
    assert not h.motion_requested and not h.motion_enabled
    assert [message.data for message in h.motion_enable_pub.messages] == [False]
    assert '等待扫描' in h.last_safety_event


@pytest.mark.parametrize('failure', [None, 'zero_stamps', 'old_stamps', 'no_scan', 'no_maps', 'mode_change', 'gate_open', 'pending', 'reference_gap', 'reference_lost'])
def test_mode_reset_waits_for_fresh_scan_and_rebuilt_maps_with_gate_closed(monkeypatch, failure):
    now = [100.]
    monkeypatch.setattr(time, 'monotonic', lambda: now[0])
    h = _Harness()
    h.motion_enabled = h.motion_requested = failure == 'gate_open'
    h.navigation_active = False
    h.navigation_reference = dict(epoch=3, ready=True, sent_monotonic=99.,
                                  observations={'obstacle_scan_age_s': 0.})
    h.costmap_updates = {'local': 0, 'global': 0}
    h.costmap_stamps = {'local': 0, 'global': 0}
    if failure == 'pending':
        h.costmap_clear_pending['local'] = Future()
    h.get_clock = lambda: SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=int(now[0]*1e9)))
    calls = []

    def clear(name):
        calls.append(name)
        future = Future()
        future.set_result(SimpleNamespace())
        return future

    h.costmap_clear_clients = {name: SimpleNamespace(wait_for_service=lambda **_: True,
        call_async=lambda _, name=name: clear(name)) for name in ('local', 'global')}

    class Event:
        def __init__(self):
            self.set_flag = False

        def set(self):
            self.set_flag = True

        def wait(self, timeout):
            if not self.set_flag:
                now[0] += timeout
                if failure == 'mode_change':
                    h.navigation_reference['epoch'] = 5
                if failure in ('reference_gap', 'reference_lost'):
                    h.navigation_reference['ready'] = failure == 'reference_gap' and now[0] >= 100.19
                if now[0] >= 100.1 and failure != 'no_scan':
                    h.navigation_reference['sent_monotonic'] = now[0]
                if failure != 'no_maps':
                    for name in h.costmap_updates:
                        h.costmap_updates[name] += 1
                        h.costmap_stamps[name] = (0 if failure == 'zero_stamps' else
                                                 1 if failure == 'old_stamps' else int(now[0]*1e9))
            return self.set_flag

    monkeypatch.setattr(threading, 'Event', Event)
    if failure not in (None, 'zero_stamps', 'reference_gap'):
        with pytest.raises(ValueError):
            NavigationWebBridge._reset_navigation_costmaps(h, 3, 100.5)
    else:
        NavigationWebBridge._reset_navigation_costmaps(h, 3, 100.5)
        assert now[0] == pytest.approx(100.3 if failure == 'reference_gap' else 100.2)
    assert calls == ([] if failure in ('gate_open', 'pending') else ['local', 'global'])
