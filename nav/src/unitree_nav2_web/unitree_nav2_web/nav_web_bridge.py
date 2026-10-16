import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
import json
import math
from pathlib import Path
import subprocess
import threading
import time
from urllib.parse import parse_qs, unquote, urlparse

import numpy as np
from PIL import Image

import rclpy
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from lifecycle_msgs.msg import Transition
from lifecycle_msgs.srv import ChangeState
from nav2_msgs.action import NavigateToPose
from nav2_msgs.srv import ManageLifecycleNodes
from nav_msgs.msg import OccupancyGrid, Odometry, Path as NavigationPath
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from rcl_interfaces.srv import SetParameters, SetParametersAtomically
from sensor_msgs.msg import LaserScan, PointCloud2
from std_msgs.msg import Bool, String
from std_srvs.srv import Empty
from tf2_ros import Buffer, TransformListener

from unitree_nav2_web.locomotion import LocomotionLink
from unitree_nav2_web.navigation_speed import NavigationSpeedSettings

FOOTPRINT = "[[-0.30, -0.25], [-0.30, 0.25], [0.32, 0.25], [0.32, -0.25]]"
# The motion gate refuses to open unless our own locomotion executor reports
# a running policy within this age.  It is also re-checked every status tick.
LOCOMOTION_READY_GRACE_S = 0.6


def _valid_navigation_pose(pose):
    position, q = pose.position, pose.orientation
    components = (q.w, q.x, q.y, q.z)
    return (all(math.isfinite(value) for value in
                (position.x, position.y, position.z, *components))
            and 0.5 <= math.hypot(*components) <= 1.5)


class NavigationWebBridge(Node):
    def __init__(self):
        super().__init__("navigation_web_bridge")
        self.declare_parameter("maps_dir", str(Path.cwd() / "maps"))
        self.declare_parameter(
            "generated_maps_dir", str(Path.cwd() / "maps" / "generated")
        )
        self.declare_parameter("web_root", "")
        self.declare_parameter("http_host", "0.0.0.0")
        self.declare_parameter("http_port", 8080)
        self.declare_parameter("converter_executable", "pcd_to_nav_map")
        self.declare_parameter("amcl_executable", "/opt/ros/foxy/lib/nav2_amcl/amcl")
        self.declare_parameter("nav2_params_file", "")
        self.declare_parameter("footprint", FOOTPRINT)
        self.declare_parameter("locomotion_command_port", 39002)
        self.declare_parameter("locomotion_status_port", 39005)
        self.declare_parameter(
            "speed_settings_file", str(Path.cwd() / "config" / "navigation_speed.json")
        )

        self.maps_dir = Path(self.get_parameter("maps_dir").value).resolve()
        self.generated_maps_dir = Path(
            self.get_parameter("generated_maps_dir").value
        ).resolve()
        self.web_root = Path(self.get_parameter("web_root").value).resolve()
        self.converter_executable = self.get_parameter("converter_executable").value
        self.amcl_executable = self.get_parameter("amcl_executable").value
        self.nav2_params_file = self.get_parameter("nav2_params_file").value
        self.footprint = str(self.get_parameter("footprint").value)
        self.generated_maps_dir.mkdir(parents=True, exist_ok=True)

        self.lock = threading.RLock()
        self.navigation_start_lock = threading.Lock()
        self.phase = "SELECT_MAP"
        self.message = "请选择一个 PCD 地图并开始处理"
        self.error = ""
        self.processing = False
        self.map_ready = False
        self.selected_map = ""
        self.map_metadata = {}
        self.pending_map = None
        self.applying_map = False
        self.initial_pose_time = 0.0
        self.localization_updates = 0
        self.localized = False
        self.motion_enabled = False
        self.motion_requested = False
        self.motion_request_generation = 0
        self.navigation_active = False
        # A bridge-only restart cannot infer the live costmap footprint left
        # by a previous process; it is re-acknowledged before motion.
        self.footprint_confirmed = False
        self.navigation_task_state = "IDLE"
        self.navigation_result_code = None
        self.navigation_completed_count = 0
        self.navigation_distance = None
        self.navigation_generation = 0
        self.navigation_started_time = 0.0
        self.navigation_pose_sequence_start = 0
        self.navigation_pose_received = False
        self.navigation_global_path_received = False
        self.navigation_motion_started = False
        self.navigation_last_motion_time = 0.0
        self.last_cloud_time = 0.0
        self.last_odom_time = 0.0
        self.last_odom_speed = 0.0
        self.last_odom_angular_speed = 0.0
        self.last_pose_time = 0.0
        self.pose_sequence = 0
        self.last_localization_scan_time = 0.0
        self.localization_scan_points = 0
        self.localization_scan_bins = 0
        self.localization_scan_sectors = 0
        self.localization_scan_xy = np.empty((0, 2), dtype=np.float64)
        self.localization_map = None
        self.localization_map_match = None
        self.initial_pose_request = None
        self.current_pose = None
        self.goal_pose = None
        self.pose_history = []
        self.last_localization_sample = None
        self.bad_localization_updates = 0
        self.localization_fault = ""
        self.localization_quality = None
        self.last_safety_event = ""
        self.last_navigation_guard = None
        self.goal_handle = None
        self.amcl_process = None
        self.pending_amcl_map = None
        self.amcl_start_stage = ""
        self.amcl_start_deadline = 0.0
        self.amcl_start_not_before = 0.0
        self.amcl_generation = 0
        self.amcl_transition_inflight = False
        self.localization_refinement_active = False
        self.localization_refinement_attempts = 0
        self.nomotion_update_inflight = False
        self.next_nomotion_update_time = 0.0
        self.locomotion_not_ready_since = 0.0
        self.bitmap_last_publish = {}
        self.path_stats = {
            "global": {"points": 0, "source_frame": ""},
            "local": {"points": 0, "source_frame": ""},
        }
        self.path_last_time = {"global": 0.0, "local": 0.0}
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.locomotion = LocomotionLink(
            int(self.get_parameter("locomotion_command_port").value),
            int(self.get_parameter("locomotion_status_port").value),
        )

        latched = QoSProfile(depth=1)
        latched.reliability = ReliabilityPolicy.RELIABLE
        latched.durability = DurabilityPolicy.TRANSIENT_LOCAL
        sensor_qos = QoSProfile(depth=5)
        sensor_qos.reliability = ReliabilityPolicy.BEST_EFFORT
        pose_qos = QoSProfile(depth=10)
        pose_qos.reliability = ReliabilityPolicy.RELIABLE

        self.map_bitmap_pub = self.create_publisher(String, "/web/map_bitmap", latched)
        self.global_bitmap_pub = self.create_publisher(
            String, "/web/global_costmap_bitmap", latched
        )
        self.local_bitmap_pub = self.create_publisher(
            String, "/web/local_costmap_bitmap", latched
        )
        self.status_pub = self.create_publisher(String, "/web/nav_status", latched)
        self.global_path_pub = self.create_publisher(
            String, "/web/global_path", latched
        )
        self.local_path_pub = self.create_publisher(
            String, "/web/local_path", latched
        )
        self.initial_pose_pub = self.create_publisher(
            PoseWithCovarianceStamped, "/initialpose", 10
        )
        self.motion_enable_pub = self.create_publisher(
            Bool, "/g1/motion_enable", latched
        )

        self.create_subscription(
            OccupancyGrid,
            "/map",
            lambda msg: self._grid_callback("map", msg),
            latched,
        )
        self.create_subscription(
            OccupancyGrid,
            "/global_costmap/costmap",
            lambda msg: self._grid_callback("global", msg),
            latched,
        )
        self.create_subscription(
            OccupancyGrid,
            "/local_costmap/costmap",
            lambda msg: self._grid_callback("local", msg),
            latched,
        )
        self.create_subscription(
            PoseWithCovarianceStamped,
            "/amcl_pose",
            self._localization_callback,
            pose_qos,
        )
        self.create_subscription(
            PointCloud2,
            "/cloud_nav",
            lambda _msg: setattr(self, "last_cloud_time", time.monotonic()),
            sensor_qos,
        )
        self.create_subscription(
            Odometry,
            "/Odometry_base",
            self._odom_callback,
            sensor_qos,
        )
        self.create_subscription(
            LaserScan,
            "/scan_localization",
            self._localization_scan_callback,
            sensor_qos,
        )
        self.create_subscription(Bool, "/g1/motion_enabled", self._motion_callback, latched)
        self.create_subscription(
            NavigationPath,
            "/plan",
            lambda msg: self._path_callback("global", msg),
            10,
        )
        self.create_subscription(
            NavigationPath,
            "/local_plan",
            lambda msg: self._path_callback("local", msg),
            10,
        )

        self.map_parameter_client = self.create_client(
            SetParameters, "/map_server/set_parameters"
        )
        self.local_costmap_parameter_client = self.create_client(
            SetParameters, "/local_costmap/local_costmap/set_parameters"
        )
        self.global_costmap_parameter_client = self.create_client(
            SetParameters, "/global_costmap/global_costmap/set_parameters"
        )
        self.localization_manager_client = self.create_client(
            ManageLifecycleNodes, "/lifecycle_manager_localization/manage_nodes"
        )
        self.amcl_change_state_client = None
        self.nomotion_update_client = None
        self._recreate_amcl_service_clients()
        self.navigate_client = ActionClient(self, NavigateToPose, "navigate_to_pose")
        self.navigation_speed = NavigationSpeedSettings(
            self.create_client(SetParametersAtomically, "/controller_server/set_parameters_atomically"),
            self.get_parameter("speed_settings_file").value,
        )

        self.create_timer(0.5, self.navigation_speed.tick)
        self.create_timer(0.1, self._pending_map_timer)
        self.create_timer(0.2, self._amcl_supervisor_timer)
        self.create_timer(0.2, self._nomotion_update_timer)
        self.create_timer(0.5, self._status_timer)
        self.http_server = self._start_http_server()
        self._set_motion(False)

    def _set_state(self, phase=None, message=None, error=None):
        with self.lock:
            if phase is not None:
                self.phase = phase
            if message is not None:
                self.message = message
            if error is not None:
                self.error = error
        self._publish_status()

    def _locomotion_blockers(self, locomotion):
        """Why our own locomotion cannot execute navigation commands yet."""
        if not locomotion["online"]:
            return ["自研运控执行进程未连接（nav/locomotion）"]
        if locomotion.get("decelerating"):
            return ["运控正在平滑归零，等待减速完成"]
        if locomotion["ready"]:
            return []
        stage_text = locomotion["stage_text"]
        if locomotion["stage"] in ("DAMPED", "FAULT"):
            return [f"运控已{stage_text}，请先复位运控"]
        return [f"运控尚未进入策略运行：{stage_text}"]

    def status(self):
        now = time.monotonic()
        locomotion = self.locomotion.snapshot()
        with self.lock:
            sensors_ready = (
                now - self.last_cloud_time < 1.5
                and now - self.last_odom_time < 1.5
            )
            amcl_alive = (
                self.amcl_process is not None
                and self.amcl_process.poll() is None
            )
            gate_blockers = []
            if not self.map_ready:
                gate_blockers.append("地图未装载")
            if not self.localized:
                gate_blockers.append("AMCL 未收敛")
            if self.localization_fault:
                gate_blockers.append(self.localization_fault)
            if not sensors_ready:
                gate_blockers.append("点云或里程计未就绪")
            if not amcl_alive:
                gate_blockers.append("AMCL 进程未运行")
            if self.navigation_active:
                gate_blockers.append("导航任务正在执行")
            gate_blockers.extend(self._locomotion_blockers(locomotion))
            speed = self.navigation_speed.snapshot()
            if not speed["ready"] or speed["pending"]:
                gate_blockers.append("导航速度设置尚未确认")
            return {
                "phase": self.phase,
                "message": self.message,
                "error": self.error,
                "processing": self.processing,
                "map_ready": self.map_ready,
                "selected_map": self.selected_map,
                "metadata": self.map_metadata,
                "sensors_ready": sensors_ready,
                "localized": self.localized,
                "localization_updates": self.localization_updates,
                "localization_refinement_attempts": self.localization_refinement_attempts,
                "localization_fault": self.localization_fault,
                "localization_quality": self.localization_quality,
                "last_safety_event": self.last_safety_event,
                "last_navigation_guard": self.last_navigation_guard,
                "localization_scan": {
                    "points": self.localization_scan_points,
                    "bins": self.localization_scan_bins,
                    "occupied_sectors": self.localization_scan_sectors,
                    "fresh": now - self.last_localization_scan_time < 1.5,
                    "good": now - self.last_localization_scan_time < 1.5
                    and self.localization_scan_points >= 80
                    and self.localization_scan_sectors >= 4,
                    "map_match": self.localization_map_match,
                },
                "motion_enabled": self.motion_enabled,
                "motion_requested": self.motion_requested,
                "motion_gate": {
                    "can_enable": not gate_blockers,
                    "blockers": gate_blockers,
                },
                "locomotion": locomotion,
                "navigation_active": self.navigation_active,
                "navigation_distance": self.navigation_distance,
                "navigation_speed": speed,
                "navigation_task": {
                    "state": self.navigation_task_state,
                    "result_code": self.navigation_result_code,
                    "completed_count": self.navigation_completed_count,
                },
                "goal": self.goal_pose,
                "navigation_monitor": {
                    "generation": self.navigation_generation,
                    "pose_received_after_goal": self.navigation_pose_received,
                    "pose_age": None
                    if self.last_pose_time <= 0.0
                    else round(now - self.last_pose_time, 3),
                    "motion_started": self.navigation_motion_started,
                    "odom_speed": round(self.last_odom_speed, 3),
                    "odom_angular_speed": round(
                        self.last_odom_angular_speed, 3
                    ),
                },
                "paths": {
                    layer: {
                        **stats,
                        "fresh": now - self.path_last_time[layer]
                        < (2.0 if layer == "local" else 5.0),
                    }
                    for layer, stats in self.path_stats.items()
                },
                "pose": self.current_pose,
            }

    def list_maps(self):
        maps = []
        if self.maps_dir.exists():
            for path in sorted(self.maps_dir.glob("*.pcd")):
                maps.append(
                    {
                        "name": path.name,
                        "size_mb": round(path.stat().st_size / 1048576.0, 2),
                    }
                )
        return maps

    def start_map_processing(self, query):
        with self.lock:
            if self.processing or self.applying_map or self.navigation_active:
                raise ValueError("当前状态不能切换地图")
            replace_confirmed = query.get("replace", ["false"])[0].lower() in (
                "1",
                "true",
                "yes",
            )
            if self.map_ready and not replace_confirmed:
                raise ValueError(
                    "当前地图会话仍然有效；重新处理 PCD 会清除定位和路径，"
                    "请在前端确认切换地图"
                )
            name = query.get("name", [""])[0]
            if Path(name).name != name or not name.lower().endswith(".pcd"):
                raise ValueError("非法地图名称")
            source = (self.maps_dir / name).resolve()
            if source.parent != self.maps_dir or not source.is_file():
                raise ValueError("PCD 地图不存在")

            def number(key, default, low, high):
                value = float(query.get(key, [str(default)])[0])
                if not math.isfinite(value) or value < low or value > high:
                    raise ValueError(f"{key} 超出允许范围")
                return value

            options = {
                "resolution": number("resolution", 0.03, 0.02, 0.30),
                "obstacle_min_height": number(
                    "obstacle_min_height", 0.15, 0.02, 1.00
                ),
                "obstacle_max_height": number(
                    "obstacle_max_height", 1.80, 0.30, 3.00
                ),
                "ground_band": number("ground_band", 0.15, 0.03, 0.40),
                "obstacle_radius": number("obstacle_radius", 0.04, 0.0, 0.40),
                "free_radius": number("free_radius", 0.15, 0.0, 0.80),
            }
            if options["obstacle_max_height"] <= options["obstacle_min_height"]:
                raise ValueError("障碍物高度上限必须大于下限")
            ground_z = query.get("ground_z", ["auto"])[0].strip() or "auto"
            if ground_z != "auto":
                value = float(ground_z)
                if not math.isfinite(value) or value < -20.0 or value > 20.0:
                    raise ValueError("ground_z 超出允许范围")
                ground_z = str(value)

            self.processing = True
            self.map_ready = False
            self.localized = False
            self.navigation_active = False
            self.navigation_task_state = "IDLE"
            self.navigation_result_code = None
            self.navigation_generation += 1
            self.navigation_started_time = 0.0
            self.navigation_pose_received = False
            self.navigation_motion_started = False
            self.navigation_last_motion_time = 0.0
            self.current_pose = None
            self.goal_pose = None
            self.last_pose_time = 0.0
            self.localization_updates = 0
            self.pose_history = []
            self.last_localization_sample = None
            self.bad_localization_updates = 0
            self.localization_fault = ""
            self.localization_quality = None
            self.initial_pose_request = None
            self.localization_map_match = None
            self.localization_refinement_active = False
            self.localization_refinement_attempts = 0
            self.nomotion_update_inflight = False
            self.selected_map = name
            self.phase = "PROCESSING_MAP"
            self.message = "正在用 PCL + OctoMap 生成二维占据地图"
            self.error = ""
            self._set_motion(False)

        thread = threading.Thread(
            target=self._map_worker,
            args=(source, options, ground_z),
            daemon=True,
        )
        thread.start()

    def _map_worker(self, source, options, ground_z):
        output_dir = self.generated_maps_dir / source.stem
        command = [
            self.converter_executable,
            "--input",
            str(source),
            "--output-dir",
            str(output_dir),
            "--resolution",
            str(options["resolution"]),
            "--obstacle-min-height",
            str(options["obstacle_min_height"]),
            "--obstacle-max-height",
            str(options["obstacle_max_height"]),
            "--ground-band",
            str(options["ground_band"]),
            "--obstacle-radius",
            str(options["obstacle_radius"]),
            "--free-radius",
            str(options["free_radius"]),
            "--ground-z",
            ground_z,
        ]
        try:
            result = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                timeout=240,
            )
            metadata = json.loads((output_dir / "metadata.json").read_text())
            with self.lock:
                self.pending_map = {
                    "pcd": str(source),
                    "yaml": str(output_dir / "map.yaml"),
                    "metadata": metadata,
                    "converter": result.stdout.strip(),
                }
                self.processing = False
                self.phase = "LOADING_MAP"
                self.message = "二维地图已生成，正在装载 Nav2/AMCL 地图"
        except Exception as error:
            with self.lock:
                self.processing = False
                self.phase = "ERROR"
                self.error = str(error)
                self.message = "地图处理失败"
            self.get_logger().error(f"Map conversion failed: {error}")

    def _pending_map_timer(self):
        with self.lock:
            if self.pending_map is None or self.applying_map:
                return
            pending = self.pending_map
            self.pending_map = None
            self.applying_map = True
        self._load_navigation_map(pending)

    def _load_navigation_map(self, pending):
        if not self.map_parameter_client.service_is_ready():
            self._map_apply_failed("map_server 参数服务未就绪")
            return
        if not self.localization_manager_client.service_is_ready():
            self._map_apply_failed("定位 lifecycle manager 服务未就绪")
            return

        # Keep the upstream Foxy AMCL binary untouched.  A fresh process per
        # selected map avoids its unsafe cleanup/configure path while also
        # guaranteeing that no particles or transient pose from the previous
        # PCD world can leak into the new one.
        self._stop_amcl_process()

        def manage(command, next_step):
            request = ManageLifecycleNodes.Request()
            request.command = command
            future = self.localization_manager_client.call_async(request)

            def completed(done):
                try:
                    if not done.result().success:
                        raise RuntimeError(f"定位 lifecycle 命令 {command} 失败")
                    next_step()
                except Exception as error:
                    self._map_apply_failed(str(error))

            future.add_done_callback(completed)

        request = SetParameters.Request()
        request.parameters = [
            Parameter("yaml_filename", value=pending["yaml"]).to_parameter_msg()
        ]
        future = self.map_parameter_client.call_async(request)

        def parameter_set(done):
            try:
                results = done.result().results
                if not results or not all(item.successful for item in results):
                    raise RuntimeError("无法设置 map_server yaml_filename")
                manage(
                    ManageLifecycleNodes.Request.RESET,
                    lambda: manage(
                        ManageLifecycleNodes.Request.STARTUP,
                        lambda: self._start_amcl_process(pending),
                    ),
                )
            except Exception as error:
                self._map_apply_failed(str(error))

        future.add_done_callback(parameter_set)

    def _stop_amcl_process(self):
        self.amcl_generation += 1
        process = self.amcl_process
        self.amcl_process = None
        self.pending_amcl_map = None
        self.amcl_start_stage = ""
        self.amcl_transition_inflight = False
        self.localization_refinement_active = False
        self.localization_refinement_attempts = 0
        self.nomotion_update_inflight = False
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=3.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=1.0)

    def _recreate_amcl_service_clients(self):
        if self.amcl_change_state_client is not None:
            self.destroy_client(self.amcl_change_state_client)
        if self.nomotion_update_client is not None:
            self.destroy_client(self.nomotion_update_client)
        self.amcl_change_state_client = self.create_client(
            ChangeState, "/amcl/change_state"
        )
        self.nomotion_update_client = self.create_client(
            Empty, "/request_nomotion_update"
        )

    def _start_amcl_process(self, pending):
        try:
            if not Path(self.amcl_executable).is_file():
                raise RuntimeError(f"官方 AMCL 可执行文件不存在：{self.amcl_executable}")
            if not Path(self.nav2_params_file).is_file():
                raise RuntimeError(f"Nav2 参数文件不存在：{self.nav2_params_file}")
            self.amcl_generation += 1
            self._recreate_amcl_service_clients()
            self.amcl_process = subprocess.Popen(
                [
                    self.amcl_executable,
                    "--ros-args",
                    "-r",
                    "__node:=amcl",
                    "--params-file",
                    self.nav2_params_file,
                ]
            )
            self.pending_amcl_map = pending
            self.amcl_start_stage = "configure"
            self.amcl_start_deadline = time.monotonic() + 12.0
            # Allow DDS discovery from the previous same-name process to age
            # out before addressing the fresh lifecycle service.
            self.amcl_start_not_before = time.monotonic() + 1.0
            self.amcl_transition_inflight = False
            self._set_state(
                phase="STARTING_LOCALIZATION",
                message="地图已装载，正在启动全新的官方 AMCL 定位进程",
                error="",
            )
        except Exception as error:
            self._map_apply_failed(str(error))

    def _amcl_supervisor_timer(self):
        process = self.amcl_process
        if process is not None and process.poll() is not None:
            exit_code = process.returncode
            self.amcl_process = None
            if self.pending_amcl_map is not None or self.applying_map:
                self._map_apply_failed(f"AMCL 进程启动失败，退出码 {exit_code}")
            else:
                self._handle_localization_fault(f"AMCL 定位进程异常退出，退出码 {exit_code}")
            return

        if self.pending_amcl_map is None:
            return
        if time.monotonic() > self.amcl_start_deadline:
            self._stop_amcl_process()
            self._map_apply_failed("AMCL lifecycle 启动超时")
            return
        if self.amcl_transition_inflight:
            return
        if time.monotonic() < self.amcl_start_not_before:
            return
        if not self.amcl_change_state_client.service_is_ready():
            return

        request = ChangeState.Request()
        request.transition.id = (
            Transition.TRANSITION_CONFIGURE
            if self.amcl_start_stage == "configure"
            else Transition.TRANSITION_ACTIVATE
        )
        stage = self.amcl_start_stage
        generation = self.amcl_generation
        self.amcl_transition_inflight = True
        future = self.amcl_change_state_client.call_async(request)

        def transitioned(done):
            if generation != self.amcl_generation:
                return
            self.amcl_transition_inflight = False
            try:
                if not done.result().success:
                    raise RuntimeError(f"AMCL lifecycle {stage} 失败")
                if stage == "configure":
                    self.amcl_start_stage = "activate"
                else:
                    pending = self.pending_amcl_map
                    self.pending_amcl_map = None
                    self.amcl_start_stage = ""
                    self._map_apply_succeeded(pending)
            except Exception as error:
                self._stop_amcl_process()
                self._map_apply_failed(str(error))

        future.add_done_callback(transitioned)

    def _map_apply_succeeded(self, pending):
        with self.lock:
            self.applying_map = False
            self.map_ready = True
            self.localized = False
            self.navigation_active = False
            self.navigation_task_state = "IDLE"
            self.navigation_result_code = None
            self.navigation_generation += 1
            self.navigation_started_time = 0.0
            self.navigation_pose_received = False
            self.navigation_motion_started = False
            self.navigation_last_motion_time = 0.0
            self.initial_pose_time = 0.0
            self.last_pose_time = 0.0
            self.localization_updates = 0
            self.pose_history = []
            self.last_localization_sample = None
            self.bad_localization_updates = 0
            self.localization_fault = ""
            self.localization_quality = None
            self.initial_pose_request = None
            self.localization_map_match = None
            self.current_pose = None
            self.localization_refinement_active = False
            self.localization_refinement_attempts = 0
            self.nomotion_update_inflight = False
            self.map_metadata = pending["metadata"]
            self.phase = "WAITING_INITIAL_POSE"
            self.message = "AMCL 地图已装载；请拖拽指定机器人初始位置和朝向"
            self.error = ""
        self._set_motion(False)
        self._publish_status()

    def _map_apply_failed(self, reason):
        with self.lock:
            self.applying_map = False
            self.pending_amcl_map = None
            self.amcl_start_stage = ""
            self.amcl_transition_inflight = False
            self.localization_refinement_active = False
            self.localization_refinement_attempts = 0
            self.nomotion_update_inflight = False
            self.map_ready = False
            self.phase = "ERROR"
            self.message = "地图装载失败"
            self.error = reason
        self._set_motion(False)
        self._publish_status()

    def set_initial_pose(self, query):
        with self.lock:
            if not self.map_ready:
                raise ValueError("AMCL 二维地图尚未装载")
            if self.navigation_active:
                raise ValueError("导航执行中不能重新设置初始位姿，请先停止导航")
            replace_confirmed = query.get("replace", ["false"])[0].lower() in (
                "1",
                "true",
                "yes",
            )
            if self.localized and not replace_confirmed:
                raise ValueError(
                    "当前 AMCL 定位仍然有效；覆盖初始位姿会关闭运动门并重新定位，"
                    "请在前端确认"
                )
            if self.amcl_process is None or self.amcl_process.poll() is not None:
                raise ValueError("AMCL 定位进程未运行，请重新装载地图")
            x = float(query.get("x", [""])[0])
            y = float(query.get("y", [""])[0])
            yaw = float(query.get("yaw", [""])[0])
            if not all(math.isfinite(value) for value in (x, y, yaw)):
                raise ValueError("初始位姿包含非法数值")

            message = PoseWithCovarianceStamped()
            # Navigation TF is published at about 10 Hz.  Dating the initial
            # pose slightly in the past lets Foxy AMCL integrate the short
            # odometric interval to now instead of rejecting a future lookup.
            message.header.stamp = (
                self.get_clock().now() - Duration(seconds=0.25)
            ).to_msg()
            message.header.frame_id = "map"
            message.pose.pose.position.x = x
            message.pose.pose.position.y = y
            # AMCL is planar; pelvis height is deliberately not part of its
            # initial state and remains owned by odom->base_link.
            message.pose.pose.position.z = 0.0
            message.pose.pose.orientation.z = math.sin(yaw * 0.5)
            message.pose.pose.orientation.w = math.cos(yaw * 0.5)
            message.pose.covariance[0] = 0.25
            message.pose.covariance[7] = 0.25
            message.pose.covariance[14] = 0.16
            message.pose.covariance[35] = 0.12
            self.initial_pose_pub.publish(message)
            self.initial_pose_time = time.monotonic()
            self.initial_pose_request = {"x": x, "y": y, "yaw": yaw}
            self.localization_map_match = None
            self.last_pose_time = 0.0
            self.localization_updates = 0
            self.pose_history = []
            self.last_localization_sample = None
            self.bad_localization_updates = 0
            self.localization_fault = ""
            self.localization_quality = None
            self.localized = False
            # Foxy AMCL updates once immediately after initialisation but does
            # not refine again while a stationary robot stays below its motion
            # thresholds.  Request several official no-motion updates so the
            # scan/map likelihood can actually converge before opening the
            # safety gate.
            self.localization_refinement_active = True
            self.localization_refinement_attempts = 0
            self.nomotion_update_inflight = False
            self.next_nomotion_update_time = self.initial_pose_time + 0.7
            self.phase = "LOCALIZING"
            self.message = "AMCL 正在用实时 /scan 进行多轮静止地图匹配与修正"
            self.error = ""
        self._set_motion(False)
        self._publish_status()
        return {"x": x, "y": y, "z": 0.0, "yaw": yaw}

    def _odom_callback(self, message):
        twist = message.twist.twist
        if (not _valid_navigation_pose(message.pose.pose)
                or not all(math.isfinite(value) for value in
                           (twist.linear.x, twist.linear.y, twist.linear.z,
                            twist.angular.x, twist.angular.y, twist.angular.z))):
            self._handle_localization_fault("导航里程计含无效位姿或速度，已请求平滑停车")
            return
        now = time.monotonic()
        speed = math.hypot(
            float(message.twist.twist.linear.x),
            float(message.twist.twist.linear.y),
        )
        angular_speed = abs(float(message.twist.twist.angular.z))
        with self.lock:
            self.last_odom_time = now
            self.last_odom_speed = speed
            self.last_odom_angular_speed = angular_speed
            if self.navigation_active and (speed >= 0.03 or angular_speed >= 0.08):
                self.navigation_motion_started = True
                self.navigation_last_motion_time = now

    def _nomotion_update_timer(self):
        with self.lock:
            localization_request = (
                self.phase == "LOCALIZING"
                and self.localization_refinement_active
            )
            # A stationary, already-converged Foxy AMCL does not periodically
            # publish /amcl_pose.  At goal start request one official scan
            # update and judge freshness from a post-goal sequence number,
            # never from the old wall-clock age of the converged pose.
            navigation_request = (
                self.navigation_active and not self.navigation_pose_received
            )
            should_request = (
                (localization_request or navigation_request)
                and not self.nomotion_update_inflight
                and time.monotonic() >= self.next_nomotion_update_time
            )
        if not should_request or not self.nomotion_update_client.service_is_ready():
            return

        with self.lock:
            self.nomotion_update_inflight = True
            generation = self.amcl_generation
        future = self.nomotion_update_client.call_async(Empty.Request())

        def completed(done):
            if generation != self.amcl_generation:
                return
            try:
                done.result()
                with self.lock:
                    self.next_nomotion_update_time = time.monotonic() + 0.65
                    if self.localization_refinement_active:
                        self.localization_refinement_attempts += 1
                        attempt = self.localization_refinement_attempts
                        self.message = (
                            f"AMCL 持续静止地图匹配修正中，第 {attempt} 轮"
                        )
            except Exception as error:
                self._handle_localization_fault(f"AMCL 静止更新失败：{error}")
            finally:
                with self.lock:
                    self.nomotion_update_inflight = False

        future.add_done_callback(completed)

    def _localization_scan_callback(self, message):
        finite = []
        points = []
        sectors = set()
        count = max(1, len(message.ranges))
        for index, raw_range in enumerate(message.ranges):
            value = float(raw_range)
            if (
                math.isfinite(value)
                and message.range_min <= value <= message.range_max
            ):
                finite.append(value)
                angle = float(message.angle_min) + index * float(
                    message.angle_increment
                )
                points.append((value * math.cos(angle), value * math.sin(angle)))
                sectors.add(min(7, index * 8 // count))
        with self.lock:
            self.last_localization_scan_time = time.monotonic()
            self.localization_scan_points = len(finite)
            self.localization_scan_bins = len(message.ranges)
            self.localization_scan_sectors = len(sectors)
            self.localization_scan_xy = np.asarray(points, dtype=np.float64).reshape(
                (-1, 2)
            )

    def _map_match_for_pose(self, x, y, yaw):
        grid = self.localization_map
        scan = self.localization_scan_xy
        if grid is None or scan.shape[0] < 80:
            return None
        cosine = math.cos(yaw)
        sine = math.sin(yaw)
        world_x = x + cosine * scan[:, 0] - sine * scan[:, 1]
        world_y = y + sine * scan[:, 0] + cosine * scan[:, 1]
        origin_cosine = math.cos(-grid["yaw"])
        origin_sine = math.sin(-grid["yaw"])
        relative_x = world_x - grid["origin_x"]
        relative_y = world_y - grid["origin_y"]
        map_x = origin_cosine * relative_x - origin_sine * relative_y
        map_y = origin_sine * relative_x + origin_cosine * relative_y
        columns = np.floor(map_x / grid["resolution"]).astype(np.int32)
        rows = np.floor(map_y / grid["resolution"]).astype(np.int32)
        inside = (
            (columns >= 0)
            & (columns < grid["width"])
            & (rows >= 0)
            & (rows < grid["height"])
        )
        near = np.zeros(scan.shape[0], dtype=bool)
        exact = np.zeros(scan.shape[0], dtype=bool)
        near[inside] = grid["near_20cm"][rows[inside], columns[inside]]
        exact[inside] = grid["occupied"][rows[inside], columns[inside]]
        inside_ratio = float(np.mean(inside))
        near_ratio = float(np.mean(near))
        exact_ratio = float(np.mean(exact))
        return {
            "points": int(scan.shape[0]),
            "inside_ratio": round(inside_ratio, 4),
            "occupied_ratio": round(exact_ratio, 4),
            "within_20cm_ratio": round(near_ratio, 4),
            "good": inside_ratio >= 0.95 and near_ratio >= 0.70,
        }

    def _localization_callback(self, message):
        if (not _valid_navigation_pose(message.pose.pose)
                or not all(math.isfinite(value) for value in message.pose.covariance)):
            self._handle_localization_fault("AMCL 含无效位姿或协方差，已请求平滑停车")
            return
        now = time.monotonic()
        fault = None
        with self.lock:
            self.last_pose_time = now
            self.pose_sequence += 1
            if (
                self.navigation_active
                and self.pose_sequence > self.navigation_pose_sequence_start
            ):
                self.navigation_pose_received = True
            q = message.pose.pose.orientation
            yaw = math.atan2(
                2.0 * (q.w * q.z + q.x * q.y),
                1.0 - 2.0 * (q.y * q.y + q.z * q.z),
            )
            x = float(message.pose.pose.position.x)
            y = float(message.pose.pose.position.y)
            covariance = message.pose.covariance
            covariance_xy = max(float(covariance[0]), float(covariance[7]))
            covariance_yaw = float(covariance[35])
            sample_dt = 0.0
            sample_distance = 0.0
            sample_yaw_delta = 0.0
            linear_rate = 0.0
            angular_rate = 0.0
            hard_jump = False
            rate_spike = False
            if self.last_localization_sample is not None:
                previous_time, previous_x, previous_y, previous_yaw = (
                    self.last_localization_sample
                )
                sample_dt = now - previous_time
                if sample_dt > 0.02:
                    sample_distance = math.hypot(x - previous_x, y - previous_y)
                    sample_yaw_delta = abs(
                        math.atan2(
                            math.sin(yaw - previous_yaw),
                            math.cos(yaw - previous_yaw),
                        )
                    )
                    linear_rate = sample_distance / sample_dt
                    angular_rate = sample_yaw_delta / sample_dt
                    # A large single discontinuity is unsafe immediately.
                    # Rate estimates, however, can spike for one AMCL update
                    # during a humanoid step or particle correction and must
                    # persist for three consecutive updates before stopping.
                    hard_jump = sample_distance > 0.45 or sample_yaw_delta > 0.60
                    rate_spike = linear_rate > 1.0 or angular_rate > 1.5
            self.last_localization_sample = (now, x, y, yaw)
            covariance_good = (
                math.isfinite(covariance_xy)
                and math.isfinite(covariance_yaw)
                and 0.0 <= covariance_xy <= 0.25
                and 0.0 <= covariance_yaw <= 0.25
            )
            map_match = self._map_match_for_pose(x, y, yaw)
            self.localization_map_match = map_match
            map_match_good = map_match is not None and bool(map_match["good"])
            correction_distance = None
            correction_yaw = None
            if self.initial_pose_request is not None:
                correction_distance = math.hypot(
                    x - self.initial_pose_request["x"],
                    y - self.initial_pose_request["y"],
                )
                correction_yaw = abs(
                    math.atan2(
                        math.sin(yaw - self.initial_pose_request["yaw"]),
                        math.cos(yaw - self.initial_pose_request["yaw"]),
                    )
                )
            self.localization_quality = {
                "covariance_xy": round(covariance_xy, 6),
                "covariance_yaw": round(covariance_yaw, 6),
                "linear_rate": round(linear_rate, 4),
                "angular_rate": round(angular_rate, 4),
                "sample_dt": round(sample_dt, 4),
                "sample_distance": round(sample_distance, 4),
                "sample_yaw_delta": round(sample_yaw_delta, 4),
                "hard_jump": hard_jump,
                "rate_spike": rate_spike,
                "map_match": map_match,
                "correction_distance": None
                if correction_distance is None
                else round(correction_distance, 4),
                "correction_yaw": None
                if correction_yaw is None
                else round(correction_yaw, 4),
                "good": covariance_good
                and map_match_good
                and not hard_jump
                and not rate_spike,
            }
            # Dynamic obstacles are expected during navigation.  They lower
            # scan-vs-static-map hit ratio precisely when Nav2 must route
            # around a chair/person, so map_match_good is an initial-pose
            # acceptance/diagnostic signal, not a runtime localization-fault
            # trigger.  Runtime safety still checks discontinuity, covariance,
            # implausible rate and pose/data freshness.
            if hard_jump or rate_spike or not covariance_good:
                self.bad_localization_updates += 1
            else:
                self.bad_localization_updates = 0
            self.current_pose = {
                "x": x,
                "y": y,
                "z": message.pose.pose.position.z,
                "yaw": yaw,
            }
            self.pose_history.append((x, y, yaw))
            self.pose_history = self.pose_history[-10:]
            if (
                self.localization_refinement_active
                and self.initial_pose_time > 0.0
                and now - self.initial_pose_time > 0.5
            ):
                self.localization_updates += 1
                sensors_fresh = (
                    now - self.last_cloud_time < 1.5 and now - self.last_odom_time < 1.5
                )
                scan_good = (
                    now - self.last_localization_scan_time < 1.5
                    and self.localization_scan_points >= 80
                    and self.localization_scan_sectors >= 4
                )
                stable = False
                if len(self.pose_history) >= 3:
                    recent = self.pose_history[-5:]
                    xs = [item[0] for item in recent]
                    ys = [item[1] for item in recent]
                    mean_yaw = math.atan2(
                        sum(math.sin(item[2]) for item in recent),
                        sum(math.cos(item[2]) for item in recent),
                    )
                    yaw_error = max(
                        abs(
                            math.atan2(
                                math.sin(item[2] - mean_yaw),
                                math.cos(item[2] - mean_yaw),
                            )
                        )
                        for item in recent
                    )
                    stable = (
                        max(xs) - min(xs) < 0.20
                        and max(ys) - min(ys) < 0.20
                        and yaw_error < 0.15
                    )
                if (
                    # One immediate AMCL publication is not convergence.  Wait
                    # for all eight scheduled stationary scan/map corrections
                    # before allowing the G1 motion gate to become available.
                    self.localization_updates >= 8
                    and self.localization_refinement_attempts >= 8
                    and sensors_fresh
                    and scan_good
                    and stable
                    and covariance_good
                    and map_match_good
                    and not hard_jump
                    and not rate_spike
                ):
                    self.localized = True
                    self.localization_refinement_active = False
                    self.localization_fault = ""
                    if not self.navigation_active:
                        self.phase = "READY"
                        self.message = "AMCL 定位稳定；运控进入策略运行后可启用运动门"
            if self.localized and (hard_jump or self.bad_localization_updates >= 3):
                if hard_jump:
                    fault = (
                        "AMCL 位姿发生硬跳变："
                        f"Δd={sample_distance:.3f} m，"
                        f"Δyaw={math.degrees(sample_yaw_delta):.1f}°"
                    )
                elif rate_spike:
                    fault = (
                        "AMCL 位姿速度连续 3 帧异常："
                        f"v={linear_rate:.2f} m/s，"
                        f"ω={angular_rate:.2f} rad/s"
                    )
                else:
                    fault = "AMCL 协方差连续 3 帧超出安全阈值"
        if fault is not None:
            self._handle_localization_fault(fault)

    def _handle_localization_fault(self, reason):
        with self.lock:
            handle = self.goal_handle
            if self.navigation_active:
                self.navigation_task_state = "FAILED"
                self.navigation_result_code = None
            self.navigation_generation += 1
            self.localized = False
            self.localization_refinement_active = False
            self.localization_fault = reason
            self.navigation_active = False
            self.navigation_distance = None
            self.navigation_started_time = 0.0
            self.navigation_pose_received = False
            self.navigation_motion_started = False
            self.navigation_last_motion_time = 0.0
            self.goal_handle = None
            self.phase = "LOCALIZATION_FAULT"
            self.message = f"{reason}；已取消导航并关闭 G1 运动安全门"
            self.last_safety_event = reason
        self.get_logger().error(f"Navigation safety fault: {reason}")
        self._set_motion(False)
        self._cancel_goal_safely(handle)
        self._publish_status()

    def _cancel_goal_safely(self, handle):
        """Cancellation is best effort; the velocity gate is closed first."""
        if handle is not None:
            try:
                handle.cancel_goal_async()
            except Exception as error:
                self.get_logger().error(f"Nav2 取消请求失败，运动门保持关闭：{error}")

    def _set_costmap_footprint(self, name, client, footprint, timeout):
        if not client.wait_for_service(timeout_sec=min(1.0, timeout)):
            raise ValueError(f"{name} footprint 参数服务未就绪")
        request = SetParameters.Request()
        request.parameters = [
            Parameter("footprint", value=footprint).to_parameter_msg()
        ]
        future = client.call_async(request)
        completed = threading.Event()
        outcome = {}

        def parameter_set(done):
            try:
                response = done.result()
                results = response.results
                if not results or not all(item.successful for item in results):
                    reasons = [
                        item.reason
                        for item in results
                        if not item.successful and item.reason
                    ]
                    raise RuntimeError(
                        "; ".join(reasons) or f"{name} 拒绝 footprint 参数"
                    )
            except Exception as error:
                outcome["error"] = str(error)
            finally:
                completed.set()

        future.add_done_callback(parameter_set)
        if not completed.wait(timeout):
            raise ValueError(f"等待 {name} footprint 确认超时")
        if outcome.get("error"):
            raise ValueError(f"{name} footprint 切换失败：{outcome['error']}")

    def _confirm_footprint(self, timeout=3.0):
        """Acknowledge the robot footprint on both costmaps once per session."""
        with self.lock:
            if self.footprint_confirmed:
                return
        deadline = time.monotonic() + timeout
        for name, client in (
            ("local_costmap", self.local_costmap_parameter_client),
            ("global_costmap", self.global_costmap_parameter_client),
        ):
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                raise ValueError("等待两张 costmap footprint 确认超时")
            self._set_costmap_footprint(name, client, self.footprint, remaining)
        with self.lock:
            self.footprint_confirmed = True

    def _motion_callback(self, message):
        force_disable = False
        abort_reason = None
        locomotion_ready = self.locomotion.snapshot()["ready"]
        with self.lock:
            acknowledged = bool(message.data)
            if acknowledged and (
                not self.motion_requested
                or not self.map_ready
                or not self.localized
                or bool(self.localization_fault)
                or not locomotion_ready
            ):
                # Never allow a delayed enable acknowledgement to reopen the
                # independent UDP gate after a localization fault/map reset.
                self.motion_enabled = False
                self.motion_requested = False
                force_disable = True
            else:
                self.motion_enabled = acknowledged
                self.motion_requested = acknowledged
            if not self.motion_enabled and self.navigation_active:
                abort_reason = "G1 运动安全门在导航执行中关闭"
        if force_disable:
            command = Bool()
            command.data = False
            self.motion_enable_pub.publish(command)
        if abort_reason is not None:
            self._abort_navigation_for_motion_gate(abort_reason)

    def _abort_navigation_for_motion_gate(self, reason):
        with self.lock:
            if not self.navigation_active:
                return
            handle = self.goal_handle
            self.navigation_generation += 1
            self.navigation_active = False
            self.navigation_distance = None
            self.navigation_started_time = 0.0
            self.navigation_pose_received = False
            self.navigation_motion_started = False
            self.navigation_last_motion_time = 0.0
            self.goal_handle = None
            self.navigation_task_state = "FAILED"
            self.navigation_result_code = None
            self.phase = "READY" if self.localized else "WAITING_INITIAL_POSE"
            self.message = f"{reason}；当前导航已取消，已请求运控归零"
            self.error = reason
            self.last_safety_event = reason
        self.get_logger().error(reason)
        self._set_motion(False)
        self._cancel_goal_safely(handle)
        self._publish_status()

    def _validate_motion_enable(self):
        self._validate_speed_ready()
        with self.lock:
            if not self.localized or not self.map_ready:
                raise ValueError("只有地图和重定位完成后才能启用 G1 运动")
            if self.localization_fault:
                raise ValueError(f"定位安全检查未通过：{self.localization_fault}")
            if self.navigation_active:
                raise ValueError("导航执行中不能重复开启运动安全门")
            if (
                time.monotonic() - self.last_cloud_time >= 1.5
                or time.monotonic() - self.last_odom_time >= 1.5
            ):
                raise ValueError("点云或里程计数据未就绪，不能启用运动安全门")
            if self.amcl_process is None or self.amcl_process.poll() is not None:
                raise ValueError("AMCL 定位进程未运行，不能启用运动安全门")
        blockers = self._locomotion_blockers(self.locomotion.snapshot())
        if blockers:
            raise ValueError("；".join(blockers))

    def _set_motion(self, enabled):
        message = Bool()
        message.data = bool(enabled)
        abort_reason = None
        with self.lock:
            self.motion_request_generation += 1
            generation = self.motion_request_generation
            if not enabled:
                # Closing cannot be invalidated by another (possibly invalid)
                # open request between generation allocation and publication.
                self.motion_requested = False
                self.motion_enabled = False
                if self.navigation_active:
                    abort_reason = "导航执行中收到关闭 G1 运动安全门的请求"
                self.motion_enable_pub.publish(message)
        if enabled:
            self._validate_motion_enable()
            self._confirm_footprint()
            with self.lock:
                if generation != self.motion_request_generation:
                    raise ValueError("已有更新的开关门请求，本次请求已取消")
                # Footprint confirmation may take seconds. A stop, lost sensor
                # or locomotion fault during that wait must win over this open.
                self._validate_motion_enable()
                self.motion_requested = True
                # Actual enable still waits for /g1/motion_enabled. Commit
                # authorization before a potentially immediate acknowledgement.
                self.motion_enable_pub.publish(message)
        if abort_reason is not None:
            self._abort_navigation_for_motion_gate(abort_reason)

    def set_motion_from_http(self, query):
        enabled = query.get("enabled", ["false"])[0].lower() in ("1", "true", "yes")
        self._set_motion(enabled)
        self._publish_status()

    def request_locomotion(self, query):
        """Forward one operator lifecycle request to our own locomotion."""
        kind = str(query.get("action", [""])[0])
        status = self.locomotion.snapshot()
        if not status["online"]:
            raise ValueError("自研运控执行进程未连接（nav/locomotion）")
        if kind in ("zero", "stop", "reset"):
            # Leaving the running policy always closes navigation first.
            with self.lock:
                navigation_active = self.navigation_active
            if navigation_active:
                self._abort_navigation_for_motion_gate("运控请求停止，导航已取消")
            else:
                self._set_motion(False)
        elif not status["actions"].get(kind):
            raise ValueError(f"运控当前状态（{status['stage_text']}）不允许该操作")
        if kind == "prepare_pose":
            pose = str(query.get("pose", [""])[0])
            if pose not in {item["value"] for item in status.get("prepare_pose_options", [])}:
                raise ValueError("请选择有效的准备姿态")
            self.locomotion.request(kind, pose=pose)
        else:
            self.locomotion.request(kind)
        self._publish_status()
        return {"action": kind}

    def _validate_navigation_start(self):
        self._validate_speed_ready()
        with self.lock:
            if not self.map_ready or not self.localized:
                raise ValueError("地图或重定位尚未完成")
            if not self.motion_enabled:
                raise ValueError("G1 运动安全门尚未启用")
            if self.navigation_active:
                raise ValueError("已有导航任务正在执行")
            now = time.monotonic()
            if now - self.last_cloud_time >= 1.5 or now - self.last_odom_time >= 1.5:
                raise ValueError("点云或里程计数据已超时，拒绝导航目标")
            if self.amcl_process is None or self.amcl_process.poll() is not None:
                raise ValueError("AMCL 定位进程未运行，拒绝导航目标")
        locomotion = self.locomotion.snapshot()
        if not locomotion["ready"]:
            raise ValueError("自研运控未处于策略运行，拒绝导航目标")
        if locomotion.get("decelerating"):
            raise ValueError("运控正在平滑归零，请等待减速完成后下发新目标")

    def _validate_speed_ready(self):
        speed = self.navigation_speed.snapshot()
        if not speed["ready"] or speed["pending"]:
            raise ValueError("导航速度设置尚未由 Nav2 确认，请稍候")

    def send_goal(self, query):
        x = float(query.get("x", [""])[0])
        y = float(query.get("y", [""])[0])
        yaw = float(query.get("yaw", [""])[0])
        if not all(math.isfinite(value) for value in (x, y, yaw)):
            raise ValueError("目标位姿包含非法数值")
        # ThreadingHTTPServer can dispatch two goal requests concurrently.
        # Serialize validation, state commit and action submission so only one
        # request can cross that boundary.
        with self.navigation_start_lock:
            self._start_navigation_goal(x, y, yaw)

    def _start_navigation_goal(self, x, y, yaw):
        self._validate_navigation_start()
        if not self.navigate_client.wait_for_server(timeout_sec=1.0):
            raise ValueError("Nav2 NavigateToPose action 未就绪")

        pose = PoseStamped()
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.header.frame_id = "map"
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.orientation.z = math.sin(yaw * 0.5)
        pose.pose.orientation.w = math.cos(yaw * 0.5)
        goal = NavigateToPose.Goal()
        goal.pose = pose

        with self.lock:
            # Map, localization, sensor freshness or the independent motion
            # gate may have changed while waiting for the action server.
            # Keep the outer RLock through the state commit so a concurrent
            # stop/map request cannot slip between this check and ACTIVE.
            self._validate_navigation_start()

            # Commit the guarded navigation state before sending the
            # asynchronous action request. Foxy can complete the goal-
            # acceptance future from a DDS executor thread immediately;
            # committing afterwards allowed _goal_response() to observe
            # False and cancel a valid goal after motion had begun.
            self.navigation_generation += 1
            generation = self.navigation_generation
            self.navigation_active = True
            self.navigation_task_state = "ACTIVE"
            self.navigation_result_code = None
            self.navigation_distance = None
            self.navigation_started_time = time.monotonic()
            self.navigation_pose_sequence_start = self.pose_sequence
            self.navigation_pose_received = False
            self.navigation_global_path_received = False
            self.navigation_motion_started = False
            self.navigation_last_motion_time = 0.0
            self.next_nomotion_update_time = self.navigation_started_time
            self.goal_pose = {"x": x, "y": y, "yaw": yaw}
            self.phase = "NAVIGATING"
            self.message = "Nav2 目标已下发，正在确认目标后的 AMCL 位姿并规划"
            self.error = ""
        # /plan and /local_plan are not latched by Nav2.  Without an explicit
        # generation boundary, the web layer kept drawing the previous task's
        # path while a new goal was still planning (or had already failed to
        # plan), which made a recovery look like path following.
        self._clear_navigation_paths()
        self._publish_status()
        try:
            future = self.navigate_client.send_goal_async(
                goal,
                feedback_callback=lambda feedback: self._goal_feedback(
                    feedback, generation
                ),
            )
            future.add_done_callback(
                lambda done: self._goal_response(done, generation)
            )
        except Exception as error:
            self._navigation_finished("导航目标发送失败", str(error), generation)
            raise

    def _goal_response(self, future, generation):
        try:
            handle = future.result()
            if not handle.accepted:
                raise RuntimeError("Nav2 拒绝了目标")
            with self.lock:
                current_generation = generation == self.navigation_generation
                if current_generation:
                    self.goal_handle = handle
                localized = self.localized
                motion_enabled = self.motion_enabled
                navigation_active = self.navigation_active
                self.last_navigation_guard = {
                    "localized": localized,
                    "motion_enabled": motion_enabled,
                    "navigation_active": navigation_active,
                    "generation": generation,
                    "accepted_at": round(time.monotonic(), 3),
                }
            # The downstream UDP motion gate independently blocks every
            # velocity packet while disabled.  A genuine localization fault or
            # user cancellation clears navigation_active and is still
            # cancelled immediately.
            if (
                not current_generation
                or not localized
                or not motion_enabled
                or not navigation_active
            ):
                with self.lock:
                    self.last_safety_event = (
                        "过期目标回调已隔离"
                        if not current_generation
                        else "目标接受时定位已失效"
                        if not localized
                        else "目标接受时运动安全门已关闭"
                        if not motion_enabled
                        else "目标接受前导航已被取消"
                    )
                handle.cancel_goal_async()
            result = handle.get_result_async()
            result.add_done_callback(
                lambda done: self._goal_result(done, generation)
            )
        except Exception as error:
            self._navigation_finished("导航目标失败", str(error), generation)

    def _goal_feedback(self, feedback, generation):
        with self.lock:
            if generation != self.navigation_generation or not self.navigation_active:
                return
            distance = getattr(feedback.feedback, "distance_remaining", None)
            if distance is None:
                return
            self.navigation_distance = round(float(distance), 3)
            if self.navigation_pose_received:
                self.message = "Nav2 正在执行全局规划与局部实时避障"

    def _goal_result(self, future, generation):
        try:
            status = future.result().status
            if status == 4:
                self._navigation_finished(
                    "导航任务完成",
                    "",
                    generation,
                    task_state="SUCCEEDED",
                    result_code=status,
                )
            elif status == 5:
                self._navigation_finished(
                    "导航任务已取消",
                    "",
                    generation,
                    task_state="CANCELED",
                    result_code=status,
                )
            else:
                self._navigation_finished(
                    f"导航结束，状态码 {status}",
                    "",
                    generation,
                    task_state="FAILED",
                    result_code=status,
                )
        except Exception as error:
            self._navigation_finished(
                "读取导航结果失败",
                str(error),
                generation,
                task_state="FAILED",
            )

    def _navigation_finished(
        self,
        message,
        error,
        generation=None,
        task_state="FAILED",
        result_code=None,
    ):
        self.get_logger().warning(f"Navigation action finished: {message}; {error}")
        with self.lock:
            if generation is not None and (
                generation != self.navigation_generation or not self.navigation_active
            ):
                self.get_logger().warning(
                    f"Ignoring stale navigation result for generation {generation}"
                )
                return
            self.navigation_active = False
            self.navigation_distance = None
            self.navigation_started_time = 0.0
            self.navigation_pose_received = False
            self.navigation_global_path_received = False
            self.navigation_motion_started = False
            self.navigation_last_motion_time = 0.0
            self.goal_handle = None
            self.navigation_task_state = task_state
            self.navigation_result_code = result_code
            if task_state == "SUCCEEDED":
                self.navigation_completed_count += 1
            if self.localization_fault:
                # A cancellation result is secondary to the safety fault that
                # caused it.  Preserve the first actionable diagnosis.
                self.phase = "LOCALIZATION_FAULT"
            elif self.localized:
                self.phase = "READY"
                if task_state == "SUCCEEDED":
                    self.message = (
                        "导航任务完成；已关闭导航速度门，平滑归零并切回 Balance。"
                        "归零后重新启用运动门，可下发下一目标"
                    )
                    self.error = ""
                else:
                    self.message = message
                    self.error = error
            else:
                # LOCALIZING is reserved for an explicitly requested AMCL
                # refinement session.  A Nav2 result must never manufacture a
                # localization session or revive an old convergence timeout.
                self.phase = (
                    "LOCALIZING"
                    if self.localization_refinement_active
                    else "WAITING_INITIAL_POSE"
                )
                self.message = message
                self.error = error
            # Finish and close atomically: a new goal/open request must never
            # slip between marking this goal inactive and stopping its motion.
            self._set_motion(False)
            if task_state == "SUCCEEDED":
                try:
                    # Explicitly use Viser's smooth-zero event as well as the
                    # persistent gate closure. No damping or policy shutdown.
                    self.locomotion.request("zero")
                except OSError as error:
                    self.get_logger().warning(
                        f"到点归零请求发送失败，导航速度门已关闭：{error}"
                    )
        self._publish_status()

    def cancel_navigation(self):
        self.get_logger().warning("Navigation action cancelled from HTTP stop endpoint")
        with self.lock:
            handle = self.goal_handle
            was_active = self.navigation_active
            self.navigation_generation += 1
            self.navigation_active = False
            self.navigation_distance = None
            self.navigation_started_time = 0.0
            self.navigation_pose_received = False
            self.navigation_motion_started = False
            self.navigation_last_motion_time = 0.0
            self.goal_handle = None
            if was_active:
                self.navigation_task_state = "CANCELED"
                self.navigation_result_code = 5
            if self.localization_fault:
                self.phase = "LOCALIZATION_FAULT"
            elif self.localized:
                self.phase = "READY"
                self.message = (
                    "导航已取消，G1 运动安全门已关闭；"
                    "重新启用后可设置下一个目标"
                )
            elif self.map_ready and self.localization_refinement_active:
                self.phase = "LOCALIZING"
            elif self.map_ready:
                self.phase = "WAITING_INITIAL_POSE"
            else:
                self.phase = "SELECT_MAP"
                self.message = "请选择一个 PCD 地图并开始处理"
        self._set_motion(False)
        self._cancel_goal_safely(handle)
        self._publish_status()

    def _grid_callback(self, layer, message):
        interval = 0.0 if layer == "map" else (1.0 if layer == "global" else 0.10)
        now = time.monotonic()
        if now - self.bitmap_last_publish.get(layer, 0.0) < interval:
            return
        self.bitmap_last_publish[layer] = now
        try:
            source_frame = (message.header.frame_id or "map").lstrip("/")
            origin_x = float(message.info.origin.position.x)
            origin_y = float(message.info.origin.position.y)
            origin_q = message.info.origin.orientation
            origin_yaw = math.atan2(
                2.0 * (origin_q.w * origin_q.z + origin_q.x * origin_q.y),
                1.0 - 2.0 * (origin_q.y * origin_q.y + origin_q.z * origin_q.z),
            )
            if source_frame != "map":
                try:
                    transform = self.tf_buffer.lookup_transform("map", source_frame, Time())
                except Exception:
                    # A local costmap has no meaningful place on the static map
                    # until relocalization establishes map->odom.
                    return
                tf_q = transform.transform.rotation
                tf_yaw = math.atan2(
                    2.0 * (tf_q.w * tf_q.z + tf_q.x * tf_q.y),
                    1.0 - 2.0 * (tf_q.y * tf_q.y + tf_q.z * tf_q.z),
                )
                cosine = math.cos(tf_yaw)
                sine = math.sin(tf_yaw)
                transformed_x = (
                    transform.transform.translation.x
                    + cosine * origin_x
                    - sine * origin_y
                )
                transformed_y = (
                    transform.transform.translation.y
                    + sine * origin_x
                    + cosine * origin_y
                )
                origin_x, origin_y = transformed_x, transformed_y
                origin_yaw += tf_yaw
            data = np.asarray(message.data, dtype=np.int16).reshape(
                (message.info.height, message.info.width)
            )
            if layer == "map":
                occupied = data >= 65
                radius = max(1, int(math.ceil(0.20 / message.info.resolution)))
                near = occupied.copy()
                for row_offset in range(-radius, radius + 1):
                    column_radius = int(
                        math.floor(math.sqrt(radius * radius - row_offset * row_offset))
                    )
                    source_row_start = max(0, -row_offset)
                    source_row_end = min(message.info.height, message.info.height - row_offset)
                    target_row_start = source_row_start + row_offset
                    target_row_end = source_row_end + row_offset
                    for column_offset in range(-column_radius, column_radius + 1):
                        source_column_start = max(0, -column_offset)
                        source_column_end = min(
                            message.info.width, message.info.width - column_offset
                        )
                        target_column_start = source_column_start + column_offset
                        target_column_end = source_column_end + column_offset
                        near[
                            target_row_start:target_row_end,
                            target_column_start:target_column_end,
                        ] |= occupied[
                            source_row_start:source_row_end,
                            source_column_start:source_column_end,
                        ]
                with self.lock:
                    self.localization_map = {
                        "origin_x": origin_x,
                        "origin_y": origin_y,
                        "yaw": origin_yaw,
                        "resolution": float(message.info.resolution),
                        "width": int(message.info.width),
                        "height": int(message.info.height),
                        "occupied": occupied,
                        "near_20cm": near,
                    }
            rgba = np.zeros((message.info.height, message.info.width, 4), dtype=np.uint8)
            unknown = data < 0
            free = data == 0
            cost = data > 0
            if layer == "map":
                rgba[unknown] = (90, 99, 112, 255)
                rgba[free] = (237, 241, 245, 255)
                rgba[cost] = (24, 28, 34, 255)
            elif layer == "global":
                rgba[unknown] = (0, 0, 0, 0)
                rgba[free] = (0, 0, 0, 0)
                rgba[cost, 0] = 238
                rgba[cost, 1] = np.clip(190 - data[cost] * 0.55, 90, 190)
                rgba[cost, 2] = 28
                rgba[cost, 3] = np.clip(45 + data[cost] * 1.35, 55, 185)
            else:
                rgba[unknown] = (0, 0, 0, 0)
                rgba[free] = (0, 0, 0, 0)
                rgba[cost, 0] = np.clip(80 + data[cost] * 1.75, 0, 255)
                rgba[cost, 1] = np.clip(210 - data[cost] * 1.5, 20, 210)
                rgba[cost, 2] = 35
                rgba[cost, 3] = np.clip(60 + data[cost] * 1.8, 80, 240)
            image = Image.fromarray(np.flipud(rgba), mode="RGBA")
            buffer = BytesIO()
            image.save(buffer, format="PNG", optimize=False)
            payload = {
                "layer": layer,
                "width": int(message.info.width),
                "height": int(message.info.height),
                "resolution": float(message.info.resolution),
                "origin": [
                    origin_x,
                    origin_y,
                ],
                "yaw": origin_yaw,
                "source_frame": source_frame,
                "frame_id": "map",
                "png": base64.b64encode(buffer.getvalue()).decode("ascii"),
            }
            output = String()
            output.data = json.dumps(payload, separators=(",", ":"))
            {
                "map": self.map_bitmap_pub,
                "global": self.global_bitmap_pub,
                "local": self.local_bitmap_pub,
            }[layer].publish(output)
        except Exception as error:
            self.get_logger().warning(f"Cannot serialize {layer} costmap: {error}")

    def _clear_navigation_paths(self):
        """Publish an empty path generation before submitting a new goal."""
        outputs = []
        with self.lock:
            publishers = {
                "global": self.global_path_pub,
                "local": self.local_path_pub,
            }
            for layer, publisher in publishers.items():
                self.path_last_time[layer] = 0.0
                self.path_stats[layer] = {"points": 0, "source_frame": ""}
                output = String()
                output.data = json.dumps(
                    {
                        "layer": layer,
                        "source_frame": "",
                        "frame_id": "map",
                        "points": [],
                    },
                    separators=(",", ":"),
                )
                outputs.append((publisher, output))
        for publisher, output in outputs:
            publisher.publish(output)

    def _path_callback(self, layer, message):
        try:
            source_frame = (message.header.frame_id or "map").lstrip("/")
            points = [
                (float(item.pose.position.x), float(item.pose.position.y))
                for item in message.poses
            ]
            if source_frame != "map":
                transform = self.tf_buffer.lookup_transform(
                    "map", source_frame, Time()
                )
                rotation = transform.transform.rotation
                yaw = math.atan2(
                    2.0
                    * (rotation.w * rotation.z + rotation.x * rotation.y),
                    1.0
                    - 2.0
                    * (rotation.y * rotation.y + rotation.z * rotation.z),
                )
                cosine = math.cos(yaw)
                sine = math.sin(yaw)
                translation = transform.transform.translation
                points = [
                    (
                        float(translation.x) + cosine * x - sine * y,
                        float(translation.y) + sine * x + cosine * y,
                    )
                    for x, y in points
                ]
            now = time.monotonic()
            with self.lock:
                self.path_last_time[layer] = now
                self.path_stats[layer] = {
                    "points": len(points),
                    "source_frame": source_frame,
                }
                if layer == "global" and self.navigation_active and points:
                    self.navigation_global_path_received = True
            output = String()
            output.data = json.dumps(
                {
                    "layer": layer,
                    "source_frame": source_frame,
                    "frame_id": "map",
                    "points": points,
                },
                separators=(",", ":"),
            )
            {
                "global": self.global_path_pub,
                "local": self.local_path_pub,
            }[layer].publish(output)
        except Exception as error:
            self.get_logger().warning(
                f"Cannot transform {layer} path into map frame: {error}"
            )

    def _status_timer(self):
        now = time.monotonic()
        locomotion = self.locomotion.snapshot()
        with self.lock:
            sensor_lost = self.motion_enabled and (
                now - self.last_cloud_time >= 1.5
                or now - self.last_odom_time >= 1.5
            )
            navigation_waiting_for_first_pose = (
                self.navigation_active
                and not self.navigation_pose_received
                and now - self.navigation_started_time >= 3.0
            )
            navigation_is_moving = (
                self.navigation_active
                and self.navigation_motion_started
                and now - self.navigation_last_motion_time < 0.75
            )
            pose_lost_while_moving = (
                navigation_is_moving
                and self.navigation_pose_received
                and now - self.last_pose_time >= 1.5
            )
            localization_start_lost = (
                self.phase == "LOCALIZING"
                and self.localization_refinement_active
                and self.initial_pose_time > 0.0
                and now - self.initial_pose_time >= 3.0
                and (
                    self.last_pose_time < self.initial_pose_time
                    or now - self.last_pose_time >= 2.0
                )
            )
            localization_not_converged = (
                self.phase == "LOCALIZING"
                and self.localization_refinement_active
                and self.initial_pose_time > 0.0
                and now - self.initial_pose_time >= 30.0
            )
            gate_open = self.motion_enabled or self.motion_requested
            if locomotion["ready"] or not gate_open:
                self.locomotion_not_ready_since = 0.0
            elif self.locomotion_not_ready_since <= 0.0:
                self.locomotion_not_ready_since = now
            # One stale status sample must not drop a running task; a stopped
            # or disconnected executor closes the gate within the grace time.
            locomotion_lost = (
                gate_open
                and self.locomotion_not_ready_since > 0.0
                and now - self.locomotion_not_ready_since
                >= LOCOMOTION_READY_GRACE_S
            )
        if sensor_lost:
            self._handle_localization_fault("点云或里程计数据超时")
        elif navigation_waiting_for_first_pose:
            self._handle_localization_fault(
                "导航开始 3 秒仍未收到目标下发后的 AMCL 位姿"
            )
        elif pose_lost_while_moving:
            self._handle_localization_fault("机器人运动中 AMCL 位姿连续超时 1.5 秒")
        elif localization_start_lost:
            self._handle_localization_fault("AMCL 未持续输出地图匹配位姿")
        elif localization_not_converged:
            self._handle_localization_fault(
                "AMCL 连续修正 30 秒仍未收敛，请重新指定更准确的初始位姿"
            )
        elif locomotion_lost:
            reason = "自研运控已离开策略运行（{}），运动门已关闭".format(
                locomotion["stage_text"]
            )
            with self.lock:
                navigation_active = self.navigation_active
                self.last_safety_event = reason
            if navigation_active:
                self._abort_navigation_for_motion_gate(reason)
            else:
                self.get_logger().error(reason)
                self._set_motion(False)
        # Motion authorization is a lease rather than a one-shot latch.  If
        # this bridge stops running, the downstream node independently closes
        # its gate and keeps publishing zero velocity.
        with self.lock:
            renew_motion = (
                self.motion_requested
                and self.motion_enabled
                and self.map_ready
                and self.localized
                and not self.localization_fault
                and locomotion["ready"]
            )
            # A concurrent stop must not be followed by an older renewal.
            if renew_motion:
                message = Bool()
                message.data = True
                self.motion_enable_pub.publish(message)
        self._publish_status()

    def _publish_status(self):
        message = String()
        message.data = json.dumps(self.status(), ensure_ascii=False, separators=(",", ":"))
        self.status_pub.publish(message)

    def _start_http_server(self):
        bridge = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):
                bridge.get_logger().info("HTTP " + (fmt % args))

            def _headers(self, status, content_type="application/json; charset=utf-8"):
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()

            def _json(self, value, status=200):
                body = json.dumps(value, ensure_ascii=False).encode("utf-8")
                self._headers(status)
                self.wfile.write(body)

            def _static(self, relative):
                relative_path = Path(relative)
                if relative_path.is_absolute() or ".." in relative_path.parts:
                    self._json({"ok": False, "error": "invalid path"}, 404)
                    return
                # Keep the installed path unresolved: --symlink-install makes
                # each web asset a symlink back into the source tree.
                path = bridge.web_root / relative_path
                if not path.is_file():
                    self._json({"ok": False, "error": "not found"}, 404)
                    return
                content_type = {
                    ".html": "text/html; charset=utf-8",
                    ".js": "application/javascript; charset=utf-8",
                    ".css": "text/css; charset=utf-8",
                }.get(path.suffix, "application/octet-stream")
                self._headers(200, content_type)
                self.wfile.write(path.read_bytes())

            def do_GET(self):
                parsed = urlparse(unquote(self.path))
                query = parse_qs(parsed.query)
                try:
                    if parsed.path == "/api/status":
                        self._json({"ok": True, "status": bridge.status()})
                    elif parsed.path == "/api/maps":
                        self._json({"ok": True, "maps": bridge.list_maps()})
                    elif parsed.path == "/api/map/process":
                        bridge.start_map_processing(query)
                        self._json({"ok": True, "accepted": True})
                    elif parsed.path == "/api/initial_pose":
                        pose = bridge.set_initial_pose(query)
                        self._json({"ok": True, "pose": pose})
                    elif parsed.path == "/api/motion":
                        bridge.set_motion_from_http(query)
                        self._json({"ok": True})
                    elif parsed.path == "/api/locomotion":
                        result = bridge.request_locomotion(query)
                        self._json({"ok": True, **result})
                    elif parsed.path == "/api/navigation/speed":
                        result = bridge.navigation_speed.request(
                            query.get("mode", [""])[0], query.get("speed", [""])[0]
                        )
                        self._json({"ok": True, "navigation_speed": result})
                    elif parsed.path == "/api/goal":
                        bridge.send_goal(query)
                        self._json({"ok": True, "accepted": True})
                    elif parsed.path in ("/api/cancel", "/api/stop"):
                        bridge.cancel_navigation()
                        self._json({"ok": True})
                    elif parsed.path == "/":
                        self._static("index.html")
                    else:
                        self._static(parsed.path.lstrip("/"))
                except Exception as error:
                    bridge.get_logger().warning(f"HTTP request rejected: {error}")
                    self._json({"ok": False, "error": str(error)}, 409)

        server = ThreadingHTTPServer(
            (
                self.get_parameter("http_host").value,
                int(self.get_parameter("http_port").value),
            ),
            Handler,
        )
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.get_logger().info(
            f"Canvas navigation UI: http://{self.get_parameter('http_host').value}:"
            f"{self.get_parameter('http_port').value}"
        )
        return server

    def destroy_node(self):
        self._set_motion(False)
        self._stop_amcl_process()
        self.http_server.shutdown()
        self.http_server.server_close()
        self.navigate_client.destroy()
        self.locomotion.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = NavigationWebBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
