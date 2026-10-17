"""Acknowledged Nav2 speed settings, persisted independently of locomotion."""

import json
import math
import os
from pathlib import Path
import tempfile
import threading
import time

from rcl_interfaces.srv import SetParametersAtomically
from rclpy.parameter import Parameter


MIN_SPEED = 0.20
MAX_SPEED = 0.60


def validate_speed(mode, speed):
    if mode not in ("auto", "fixed"):
        raise ValueError("速度模式必须为自动调速或固定巡航")
    if isinstance(speed, bool):
        raise ValueError("速度必须是数字")
    speed = float(speed)
    if not math.isfinite(speed) or not MIN_SPEED <= speed <= MAX_SPEED:
        raise ValueError("导航速度范围为 0.20–0.60 m/s")
    return mode, speed


class NavigationSpeedSettings:
    def __init__(self, client, path):
        self.client = client
        self.path = Path(path)
        self.lock = threading.RLock()
        self.mode, self.speed = "auto", 0.35
        self.ready = False
        self.pending = False
        self.error = ""
        self.save_warning = ""
        self.next_retry = 0.0
        self.started = 0.0
        self.requested = None
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text())
                self.mode, self.speed = validate_speed(data["mode"], data["speed"])
            except (OSError, ValueError, KeyError, TypeError) as error:
                self.save_warning = f"保存的速度设置不可用，使用默认值：{error}"

    def snapshot(self):
        with self.lock:
            error = self.error
            if self.pending and time.monotonic() - self.started > 3.0:
                error = "等待 Nav2 确认超时；尚未确认生效"
            return {
                "mode": self.mode, "speed": self.speed,
                "min": MIN_SPEED, "max": MAX_SPEED,
                "ready": self.ready, "pending": self.pending,
                "requested": self.requested,
                "error": error, "save_warning": self.save_warning,
            }

    def tick(self):
        with self.lock:
            if self.ready or self.pending or time.monotonic() < self.next_retry:
                return
            try:
                self._submit(self.mode, self.speed, persist=False)
            except (ValueError, RuntimeError) as error:
                self.error = str(error)
                self.next_retry = time.monotonic() + 2.0

    def request(self, mode, speed):
        mode, speed = validate_speed(mode, speed)
        with self.lock:
            self._submit(mode, speed, persist=True)
            return self.snapshot()

    def _submit(self, mode, speed, persist):
        if self.pending:
            raise ValueError("上一组速度设置正在等待 Nav2 确认")
        if not self.client.service_is_ready():
            raise ValueError("Nav2 速度参数服务尚未就绪")
        request = SetParametersAtomically.Request()
        request.parameters = [
            Parameter("FollowPath.navigation_speed_mode", value=mode).to_parameter_msg(),
            Parameter("FollowPath.navigation_speed", value=speed).to_parameter_msg(),
            Parameter("CartPush.navigation_speed_mode", value=mode).to_parameter_msg(),
            Parameter("CartPush.navigation_speed", value=speed).to_parameter_msg(),
        ]
        self.pending = True
        self.error = ""
        self.started = time.monotonic()
        self.requested = {"mode": mode, "speed": speed}
        try:
            future = self.client.call_async(request)
        except Exception:
            self.pending = False
            self.requested = None
            raise

        def completed(done):
            with self.lock:
                try:
                    result = done.result().result
                    if not result.successful:
                        raise ValueError(result.reason or "Nav2 拒绝速度设置")
                    self.mode, self.speed = mode, speed
                    self.ready = True
                    if persist:
                        try:
                            self._save()
                            self.save_warning = ""
                        except OSError as error:
                            self.save_warning = f"本次已生效，但未保存到下次启动：{error}"
                except Exception as error:
                    self.error = f"速度设置未生效：{error}"
                finally:
                    self.pending = False
                    self.requested = None
                    self.next_retry = time.monotonic() + 2.0

        future.add_done_callback(completed)

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=str(self.path.parent), delete=False
            ) as stream:
                temporary = stream.name
                json.dump({"mode": self.mode, "speed": self.speed}, stream, indent=2)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, str(self.path))
        finally:
            if temporary is not None and os.path.exists(temporary):
                os.unlink(temporary)
