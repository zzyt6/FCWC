#!/usr/bin/env python3
"""RViz-like FAST-LIO visualization served to Wi-Fi clients with Viser."""

import fcntl
import os
import socket
import struct
import threading
import time
from typing import List, Optional, Tuple

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2, PointField
from std_srvs.srv import Trigger

try:
    import viser
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "Viser is not installed. Run: python3 -m pip install --user viser==1.0.30"
    ) from exc


_POINT_DTYPES = {
    PointField.INT8: "i1",
    PointField.UINT8: "u1",
    PointField.INT16: "i2",
    PointField.UINT16: "u2",
    PointField.INT32: "i4",
    PointField.UINT32: "u4",
    PointField.FLOAT32: "f4",
    PointField.FLOAT64: "f8",
}


class ViserBridge(Node):
    """Display FAST-LIO outputs without taking part in registration or mapping."""

    def __init__(self) -> None:
        super().__init__("viser_bridge")

        self.declare_parameter("host", "0.0.0.0")
        self.declare_parameter("port", 8080)
        self.declare_parameter("wifi_interface", "wlan0")
        self.declare_parameter("cloud_topic", "/cloud_registered")
        self.declare_parameter("map_topic", "/Laser_map")
        self.declare_parameter("odom_topic", "/Odometry_base")
        self.declare_parameter("map_save_service", "/map_save")
        self.declare_parameter("live_voxel_size", 0.10)
        self.declare_parameter("live_max_points", 250000)
        self.declare_parameter("current_scan_max_points", 15000)
        self.declare_parameter("sensor_range", 10.0)
        self.declare_parameter("update_rate_hz", 2.0)
        self.declare_parameter("point_size", 0.035)
        self.declare_parameter("color_mode", "height")
        self.declare_parameter("color_min_z", -1.5)
        self.declare_parameter("color_max_z", 2.5)
        self.declare_parameter("trajectory_min_distance", 0.02)
        self.declare_parameter("pcd_path", "./maps/fastlio_map.pcd")
        self.declare_parameter("full_map_chunk_points", 50000)
        self.declare_parameter("full_map_chunk_period", 0.15)

        self.host = str(self.get_parameter("host").value)
        self.port = int(self.get_parameter("port").value)
        self.wifi_interface = str(self.get_parameter("wifi_interface").value)
        self.cloud_topic = str(self.get_parameter("cloud_topic").value)
        self.map_topic = str(self.get_parameter("map_topic").value)
        self.odom_topic = str(self.get_parameter("odom_topic").value)
        self.map_save_service = str(self.get_parameter("map_save_service").value)
        self.live_voxel_size = max(
            float(self.get_parameter("live_voxel_size").value), 0.01
        )
        self.live_max_points = max(
            int(self.get_parameter("live_max_points").value), 1000
        )
        self.current_scan_max_points = max(
            int(self.get_parameter("current_scan_max_points").value), 1000
        )
        self.sensor_range = float(self.get_parameter("sensor_range").value)
        self.update_rate_hz = max(
            float(self.get_parameter("update_rate_hz").value), 0.2
        )
        self.point_size = float(self.get_parameter("point_size").value)
        self.color_mode = str(self.get_parameter("color_mode").value)
        self.color_min_z = float(self.get_parameter("color_min_z").value)
        self.color_max_z = float(self.get_parameter("color_max_z").value)
        self.trajectory_min_distance = float(
            self.get_parameter("trajectory_min_distance").value
        )
        self.pcd_path = os.path.abspath(
            os.path.expanduser(str(self.get_parameter("pcd_path").value))
        )
        self.full_map_chunk_points = max(
            int(self.get_parameter("full_map_chunk_points").value), 1000
        )
        self.full_map_chunk_period = max(
            float(self.get_parameter("full_map_chunk_period").value), 0.02
        )

        self._lock = threading.Lock()
        self._live_points = np.empty((0, 3), dtype=np.float32)
        self._live_intensity = np.empty((0,), dtype=np.float32)
        self._scan_points = np.empty((0, 3), dtype=np.float32)
        self._trajectory = np.empty((0, 3), dtype=np.float32)
        self._robot_position: Optional[np.ndarray] = None
        self._robot_wxyz = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self._map_dirty = True
        self._scan_dirty = True
        self._pose_dirty = True
        self._map_messages = 0
        self._scan_messages = 0
        self._odom_messages = 0
        self._last_map_time: Optional[float] = None
        self._last_scan_time: Optional[float] = None
        self._last_odom_time: Optional[float] = None
        self._frame_id = ""
        self._status_message = "正在等待 FAST-LIO 数据"
        self._destroyed = False

        self._full_mode = False
        self._full_points: Optional[np.ndarray] = None
        self._full_intensity: Optional[np.ndarray] = None
        self._full_index = 0
        self._full_handles: List[object] = []
        self._clear_full_requested = False
        self._full_load_in_progress = False
        self._next_full_chunk_time = 0.0

        self._server = viser.ViserServer(
            host=self.host,
            port=self.port,
            label="Unitree G1 · FAST-LIO 3D Map",
        )
        self._server.scene.set_up_direction("+z")
        self._server.scene.add_grid(
            "/world/grid",
            width=20.0,
            height=20.0,
            plane="xy",
            cell_size=1.0,
            section_size=5.0,
            cell_color=(90, 90, 90),
            cell_thickness=0.5,
            section_color=(140, 140, 140),
            section_thickness=1.0,
        )
        self._live_handle = self._server.scene.add_point_cloud(
            "/map/live",
            points=np.empty((0, 3), dtype=np.float32),
            colors=np.empty((0, 3), dtype=np.uint8),
            point_size=self.point_size,
            point_shape="circle",
            precision="float32",
        )
        self._scan_handle = self._server.scene.add_point_cloud(
            "/scan/current",
            points=np.empty((0, 3), dtype=np.float32),
            colors=(255, 220, 80),
            point_size=self.point_size * 1.25,
            point_shape="circle",
            precision="float32",
        )
        self._server.scene.add_frame(
            "/robot/base_link", axes_length=0.35, axes_radius=0.015
        )

        with self._server.gui.add_folder("FAST-LIO 状态"):
            self._status = self._server.gui.add_markdown(self._status_message)
            self._color_dropdown = self._server.gui.add_dropdown(
                "全局点云颜色",
                ("height", "intensity", "solid"),
                initial_value=(
                    self.color_mode
                    if self.color_mode in ("height", "intensity", "solid")
                    else "height"
                ),
            )
            self._point_size_slider = self._server.gui.add_slider(
                "点大小",
                min=0.01,
                max=0.10,
                step=0.005,
                initial_value=self.point_size,
            )
            self._full_button = self._server.gui.add_button(
                "保存实时建图层并渐进显示最终 PCD", color="green"
            )
            self._live_button = self._server.gui.add_button(
                "返回实时全局地图", color="blue"
            )

        @self._color_dropdown.on_update
        def _on_color_change(_) -> None:
            with self._lock:
                self.color_mode = str(self._color_dropdown.value)
                self._map_dirty = True

        @self._point_size_slider.on_update
        def _on_point_size_change(_) -> None:
            with self._lock:
                self.point_size = float(self._point_size_slider.value)
                self._map_dirty = True
                self._scan_dirty = True

        @self._full_button.on_click
        def _on_full_map(_) -> None:
            self._request_save_and_load()

        @self._live_button.on_click
        def _on_live_map(_) -> None:
            with self._lock:
                self._full_mode = False
                self._full_points = None
                self._full_intensity = None
                self._full_index = 0
                self._clear_full_requested = True
                self._map_dirty = True
                self._status_message = "已返回实时地图模式"

        sensor_qos = QoSProfile(
            depth=2,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self._map_sub = self.create_subscription(
            PointCloud2, self.map_topic, self._map_callback, sensor_qos
        )
        self._cloud_sub = self.create_subscription(
            PointCloud2, self.cloud_topic, self._cloud_callback, sensor_qos
        )
        self._odom_sub = self.create_subscription(
            Odometry, self.odom_topic, self._odom_callback, 20
        )
        self._map_save_client = self.create_client(Trigger, self.map_save_service)
        self._render_timer = self.create_timer(1.0 / self.update_rate_hz, self._render)

        wifi_ip = self._interface_ipv4(self.wifi_interface)
        wifi_url = (
            "http://{}:{}".format(wifi_ip, self.port)
            if wifi_ip
            else "http://<机器人 Wi-Fi IP>:{}".format(self.port)
        )
        self.get_logger().info(
            "Viser listens on {}:{}; Wi-Fi interface {} URL: {}".format(
                self.host, self.port, self.wifi_interface, wifi_url
            )
        )
        self.get_logger().info(
            "RViz-equivalent layers: current={}, global={}, odometry={}".format(
                self.cloud_topic, self.map_topic, self.odom_topic
            )
        )

    @staticmethod
    def _interface_ipv4(interface: str) -> Optional[str]:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            packed = struct.pack("256s", interface[:15].encode("ascii"))
            return socket.inet_ntoa(fcntl.ioctl(sock.fileno(), 0x8915, packed)[20:24])
        except (OSError, UnicodeEncodeError):
            return None
        finally:
            sock.close()

    @staticmethod
    def _field_view(msg: PointCloud2, name: str) -> Optional[np.ndarray]:
        field = next((item for item in msg.fields if item.name == name), None)
        if field is None or field.datatype not in _POINT_DTYPES:
            return None
        endian = ">" if msg.is_bigendian else "<"
        dtype = np.dtype(endian + _POINT_DTYPES[field.datatype])
        try:
            return np.ndarray(
                shape=(msg.height, msg.width),
                dtype=dtype,
                buffer=msg.data,
                offset=field.offset,
                strides=(msg.row_step, msg.point_step),
            ).reshape(-1)
        except (TypeError, ValueError):
            return None

    @classmethod
    def _pointcloud_to_numpy(
        cls, msg: PointCloud2
    ) -> Tuple[np.ndarray, np.ndarray]:
        x = cls._field_view(msg, "x")
        y = cls._field_view(msg, "y")
        z = cls._field_view(msg, "z")
        if x is None or y is None or z is None:
            return np.empty((0, 3), np.float32), np.empty((0,), np.float32)
        points = np.column_stack((x, y, z)).astype(np.float32, copy=False)
        intensity = cls._field_view(msg, "intensity")
        intensity_out = (
            np.zeros(points.shape[0], dtype=np.float32)
            if intensity is None
            else intensity.astype(np.float32, copy=False)
        )
        finite = np.isfinite(points).all(axis=1) & np.isfinite(intensity_out)
        return points[finite], intensity_out[finite]

    def _map_callback(self, msg: PointCloud2) -> None:
        points, intensity = self._pointcloud_to_numpy(msg)
        if points.shape[0] == 0:
            return

        voxels = np.floor(points / self.live_voxel_size).astype(np.int64)
        _, indices = np.unique(voxels, axis=0, return_index=True)
        indices.sort()
        points = points[indices]
        intensity = intensity[indices]
        if points.shape[0] > self.live_max_points:
            step = int(np.ceil(points.shape[0] / self.live_max_points))
            points = points[::step]
            intensity = intensity[::step]

        with self._lock:
            self._live_points = points
            self._live_intensity = intensity
            self._map_messages += 1
            self._last_map_time = time.monotonic()
            self._frame_id = msg.header.frame_id
            if not self._full_mode:
                self._map_dirty = True

    def _cloud_callback(self, msg: PointCloud2) -> None:
        points, _ = self._pointcloud_to_numpy(msg)
        if points.shape[0] == 0:
            return
        if points.shape[0] > self.current_scan_max_points:
            step = int(np.ceil(points.shape[0] / self.current_scan_max_points))
            points = points[::step]
        with self._lock:
            self._scan_points = points
            self._scan_messages += 1
            self._last_scan_time = time.monotonic()
            self._scan_dirty = True

    def _odom_callback(self, msg: Odometry) -> None:
        pose = msg.pose.pose
        position = np.array(
            [pose.position.x, pose.position.y, pose.position.z], dtype=np.float32
        )
        wxyz = np.array(
            [
                pose.orientation.w,
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
            ],
            dtype=np.float32,
        )
        norm = float(np.linalg.norm(wxyz))
        wxyz = wxyz / norm if norm > 1e-6 else np.array([1, 0, 0, 0], np.float32)
        with self._lock:
            self._robot_position = position
            self._robot_wxyz = wxyz
            if (
                self._trajectory.shape[0] == 0
                or np.linalg.norm(position - self._trajectory[-1])
                >= self.trajectory_min_distance
            ):
                self._trajectory = np.concatenate(
                    (self._trajectory, position.reshape(1, 3))
                )
                if self._trajectory.shape[0] > 20000:
                    self._trajectory = self._trajectory[::2]
            self._odom_messages += 1
            self._last_odom_time = time.monotonic()
            self._pose_dirty = True

    def _colors(
        self, points: np.ndarray, intensity: np.ndarray, mode: Optional[str] = None
    ) -> np.ndarray:
        selected = self.color_mode if mode is None else mode
        if selected == "solid":
            return np.broadcast_to(
                np.array([80, 190, 255], dtype=np.uint8), (points.shape[0], 3)
            ).copy()
        if selected == "intensity":
            clipped = np.clip(intensity, 0.0, 255.0).astype(np.uint8)
            return np.column_stack((clipped, clipped, clipped))
        denominator = max(self.color_max_z - self.color_min_z, 1e-6)
        value = np.clip((points[:, 2] - self.color_min_z) / denominator, 0.0, 1.0)
        red = np.clip(1.5 - np.abs(4.0 * value - 3.0), 0.0, 1.0)
        green = np.clip(1.5 - np.abs(4.0 * value - 2.0), 0.0, 1.0)
        blue = np.clip(1.5 - np.abs(4.0 * value - 1.0), 0.0, 1.0)
        return (np.column_stack((red, green, blue)) * 255.0).astype(np.uint8)

    def _request_save_and_load(self) -> None:
        if self._full_load_in_progress:
            return
        if not self._map_save_client.service_is_ready():
            self._status_message = "FAST-LIO /map_save 服务尚未就绪"
            self.get_logger().warning(self._status_message)
            return
        self._full_load_in_progress = True
        self._status_message = "正在请求 FAST-LIO 保存完整 PCD……"
        future = self._map_save_client.call_async(Trigger.Request())
        future.add_done_callback(self._save_finished)

    def _save_finished(self, future) -> None:
        try:
            response = future.result()
        except Exception as exc:  # pylint: disable=broad-except
            self._full_load_in_progress = False
            self._status_message = "地图保存调用失败：{}".format(exc)
            self.get_logger().error(self._status_message)
            return
        if not response.success:
            self._full_load_in_progress = False
            self._status_message = "FAST-LIO 保存失败：{}".format(response.message)
            self.get_logger().error(self._status_message)
            return
        self._status_message = "PCD 已保存，正在读取完整地图……"
        threading.Thread(target=self._load_pcd_worker, daemon=True).start()

    def _load_pcd_worker(self) -> None:
        try:
            points, intensity = self._read_pcd(self.pcd_path)
        except (OSError, ValueError) as exc:
            with self._lock:
                self._full_load_in_progress = False
                self._status_message = "读取 PCD 失败：{}".format(exc)
            self.get_logger().error(self._status_message)
            return
        with self._lock:
            self._full_points = points
            self._full_intensity = intensity
            self._full_index = 0
            self._full_mode = True
            self._clear_full_requested = True
            self._full_load_in_progress = False
            self._next_full_chunk_time = 0.0
            self._status_message = "开始按块显示完整 PCD（{:,} 点）".format(
                points.shape[0]
            )

    @staticmethod
    def _read_pcd(path: str) -> Tuple[np.ndarray, np.ndarray]:
        metadata = {}
        with open(path, "rb") as pcd:
            while True:
                raw = pcd.readline()
                if not raw:
                    raise ValueError("PCD 头部不完整")
                line = raw.decode("ascii", errors="strict").strip()
                if not line or line.startswith("#"):
                    continue
                key, *values = line.split()
                metadata[key.upper()] = values
                if key.upper() == "DATA":
                    break

            fields = metadata.get("FIELDS", metadata.get("FIELD", []))
            sizes = [int(item) for item in metadata.get("SIZE", [])]
            types = metadata.get("TYPE", [])
            counts = [int(item) for item in metadata.get("COUNT", ["1"] * len(fields))]
            point_count = int(metadata.get("POINTS", ["0"])[0])
            if not (fields and len(fields) == len(sizes) == len(types) == len(counts)):
                raise ValueError("PCD 字段描述无效")
            if point_count <= 0:
                point_count = int(metadata["WIDTH"][0]) * int(metadata["HEIGHT"][0])

            dtype_fields = []
            type_map = {
                ("F", 4): "<f4",
                ("F", 8): "<f8",
                ("I", 1): "i1",
                ("I", 2): "<i2",
                ("I", 4): "<i4",
                ("U", 1): "u1",
                ("U", 2): "<u2",
                ("U", 4): "<u4",
            }
            for name, size, kind, count in zip(fields, sizes, types, counts):
                base = type_map.get((kind.upper(), size))
                if base is None:
                    raise ValueError("不支持的 PCD 类型 {}{}".format(kind, size))
                dtype_fields.append((name, base) if count == 1 else (name, base, (count,)))

            data_format = metadata["DATA"][0].lower()
            if data_format == "binary":
                records = np.fromfile(pcd, dtype=np.dtype(dtype_fields), count=point_count)
            elif data_format == "ascii":
                matrix = np.loadtxt(pcd, max_rows=point_count, ndmin=2)
                records = np.empty(matrix.shape[0], dtype=np.dtype(dtype_fields))
                column = 0
                for name, count in zip(fields, counts):
                    if count == 1:
                        records[name] = matrix[:, column]
                    else:
                        records[name] = matrix[:, column : column + count]
                    column += count
            else:
                raise ValueError("暂不支持 DATA {}".format(data_format))

        if not all(name in records.dtype.names for name in ("x", "y", "z")):
            raise ValueError("PCD 缺少 x/y/z")
        points = np.column_stack((records["x"], records["y"], records["z"])).astype(
            np.float32, copy=False
        )
        intensity = (
            records["intensity"].astype(np.float32, copy=False)
            if "intensity" in records.dtype.names
            else np.zeros(points.shape[0], dtype=np.float32)
        )
        finite = np.isfinite(points).all(axis=1) & np.isfinite(intensity)
        return points[finite], intensity[finite]

    def _render(self) -> None:
        with self._lock:
            clear_full = self._clear_full_requested
            self._clear_full_requested = False
            map_dirty = self._map_dirty
            scan_dirty = self._scan_dirty
            pose_dirty = self._pose_dirty
            self._map_dirty = False
            self._scan_dirty = False
            self._pose_dirty = False
            full_mode = self._full_mode
            live_points = self._live_points.copy() if map_dirty else None
            live_intensity = self._live_intensity.copy() if map_dirty else None
            scan_points = self._scan_points.copy() if scan_dirty else None
            trajectory = self._trajectory.copy() if pose_dirty else None
            robot_position = (
                None if self._robot_position is None else self._robot_position.copy()
            )
            robot_wxyz = self._robot_wxyz.copy()
            color_mode = self.color_mode
            status_message = self._status_message
            full_load_in_progress = self._full_load_in_progress
            map_messages = self._map_messages
            scan_messages = self._scan_messages
            odom_messages = self._odom_messages
            last_map_time = self._last_map_time
            last_scan_time = self._last_scan_time
            last_odom_time = self._last_odom_time
            frame_id = self._frame_id
            live_count = self._live_points.shape[0]
            full_total = 0 if self._full_points is None else self._full_points.shape[0]
            full_index = self._full_index

        if clear_full:
            for handle in self._full_handles:
                handle.remove()
            self._full_handles.clear()

        self._live_handle.visible = not full_mode
        if map_dirty and not full_mode and live_points is not None:
            self._live_handle = self._server.scene.add_point_cloud(
                "/map/live",
                points=live_points,
                colors=self._colors(live_points, live_intensity, color_mode),
                point_size=self.point_size,
                point_shape="circle",
                precision="float32",
            )

        if scan_dirty and scan_points is not None:
            self._scan_handle = self._server.scene.add_point_cloud(
                "/scan/current",
                points=scan_points,
                colors=(255, 220, 80),
                point_size=self.point_size * 1.25,
                point_shape="circle",
                precision="float32",
            )

        if pose_dirty:
            if robot_position is not None:
                self._server.scene.add_frame(
                    "/robot/base_link",
                    axes_length=0.35,
                    axes_radius=0.015,
                    position=robot_position,
                    wxyz=robot_wxyz,
                )
            if trajectory is not None and trajectory.shape[0] >= 2:
                segments = np.stack((trajectory[:-1], trajectory[1:]), axis=1)
                self._server.scene.add_line_segments(
                    "/robot/trajectory",
                    points=segments,
                    colors=(255, 120, 20),
                    line_width=3.0,
                )

        now = time.monotonic()
        if full_mode and now >= self._next_full_chunk_time:
            with self._lock:
                if self._full_points is not None and self._full_index < self._full_points.shape[0]:
                    start = self._full_index
                    end = min(start + self.full_map_chunk_points, self._full_points.shape[0])
                    chunk_points = self._full_points[start:end]
                    chunk_intensity = self._full_intensity[start:end]
                    self._full_index = end
                    full_index = end
                else:
                    chunk_points = None
            if chunk_points is not None:
                handle = self._server.scene.add_point_cloud(
                    "/map/full/chunk_{:06d}".format(len(self._full_handles)),
                    points=chunk_points,
                    colors=self._colors(chunk_points, chunk_intensity, color_mode),
                    point_size=self.point_size,
                    point_shape="circle",
                    precision="float32",
                )
                self._full_handles.append(handle)
                self._next_full_chunk_time = now + self.full_map_chunk_period

        mode_text = (
            "完整 PCD {:,}/{:,}".format(full_index, full_total)
            if full_mode
            else "实时全局预览 {:,} 点".format(live_count)
        )
        if full_mode:
            display_status = (
                "正在渐进加载完整 PCD"
                if full_index < full_total
                else "完整 PCD 已全部显示"
            )
        elif full_load_in_progress:
            display_status = status_message
        else:
            scan_age = float("inf") if last_scan_time is None else now - last_scan_time
            odom_age = float("inf") if last_odom_time is None else now - last_odom_time
            map_age = float("inf") if last_map_time is None else now - last_map_time
            if scan_age < 3.0 and odom_age < 3.0:
                display_status = (
                    "FAST-LIO 正在实时建图（三路数据正常）"
                    if map_age < 5.0
                    else "FAST-LIO 正常；当前帧与腰部里程计实时更新，等待全局地图刷新"
                )
            elif scan_messages == 0 and odom_messages == 0:
                display_status = "正在等待 FAST-LIO 数据"
            else:
                display_status = "FAST-LIO 数据已超过 3 秒未更新，请检查输入或进程"
        self._status.content = (
            "**显示模式：** {}  \n"
            "**状态：** {}  \n"
            "**全局/当前帧/里程计消息：** {:,} / {:,} / {:,}  \n"
            "**坐标系：** `{}`  \n"
            "**输入限距：** {:.1f} m · **实时体素：** {:.2f} m  \n"
            "**网络：** DDS `eth0` · Viser Wi-Fi `{}`"
        ).format(
            mode_text,
            display_status,
            map_messages,
            scan_messages,
            odom_messages,
            frame_id or "等待中",
            self.sensor_range,
            self.live_voxel_size,
            self.wifi_interface,
        )

    def destroy_node(self) -> bool:
        if self._destroyed:
            return True
        self._destroyed = True
        self._server.stop()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ViserBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
