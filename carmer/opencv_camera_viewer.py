#!/usr/bin/env python3
"""Preview RealSense RGB with OpenCV/V4L2 at native 640x480, 60 FPS.

Run in the CWC conda environment:
    python carmer/opencv_camera_viewer.py
    python carmer/opencv_camera_viewer.py --no-display --frames 600

Q/Esc closes the window. S saves an unmodified BGR frame. Capture runs in a
separate thread so a slow preview does not block acquisition. K and D are loaded
from calibrate_camera/camera_intrinsics.json; images are not undistorted.
"""

import argparse
from collections import deque
from datetime import datetime
import errno
import fcntl
import json
import math
import os
from pathlib import Path
import struct
import sys
import threading
import time


WIDTH, HEIGHT, FPS = 640, 480, 60
WARMUP = FPS * 2
SCRIPT_DIR = Path(__file__).resolve().parent
INTRINSICS_FILE = SCRIPT_DIR / "calibrate_camera" / "camera_intrinsics.json"
WINDOW = "RealSense - 640x480 @ 60 FPS"

# linux/videodev2.h: query the capabilities of this node, not the whole camera.
_V4L2_CAPABILITY = struct.Struct("=16s32s32s6I")
_VIDIOC_QUERYCAP = 0x80685600
_V4L2_CAP_VIDEO_CAPTURE = 0x00000001
_V4L2_CAP_DEVICE_CAPS = 0x80000000


def video_node_capabilities(device):
    """Read node capabilities; os.open retains meaningful errno on failure."""
    fd = os.open(device, os.O_RDWR | os.O_NONBLOCK)
    try:
        buffer = bytearray(_V4L2_CAPABILITY.size)
        fcntl.ioctl(fd, _VIDIOC_QUERYCAP, buffer, True)
        _, _, _, _, capabilities, device_caps, *_ = _V4L2_CAPABILITY.unpack(buffer)
        return device_caps if capabilities & _V4L2_CAP_DEVICE_CAPS else capabilities
    finally:
        os.close(fd)


def camera_os_error(error):
    hints = {
        errno.EACCES: "permission denied; check device permissions and your login groups",
        errno.EPERM: "access denied; check device permissions or container/sandbox access",
        errno.EBUSY: "device is busy; close the process using this camera",
        errno.ENOENT: "device node is absent; check camera connection or container/sandbox device access",
        errno.ENODEV: "device disconnected; reconnect the camera",
        errno.EIO: "device I/O error; check USB connection and kernel log",
    }
    return f"{errno.errorcode.get(error.errno, 'OS_ERROR')} ({error.strerror}): {hints.get(error.errno, 'V4L2 device access failed')}"


class CameraReader:
    def __init__(self, cap, frames):
        self.cap = cap
        self.limit = WARMUP + frames if frames else 0
        self.condition = threading.Condition()
        self.stop = threading.Event()
        self.frame = None
        self.count = 0
        self.done = False
        self.error = None
        self.measured_count = 0
        self.first_sample = None
        self.last_sample = None
        self.recent = deque(maxlen=FPS + 1)
        self.fps = 0.0
        self.frame_timestamp = None
        self.timestamp_basis = "v4l2_monotonic"
        self.thread = threading.Thread(target=self._capture, daemon=True)
        self.thread.start()

    def _capture(self):
        try:
            import cv2

            while not self.stop.is_set():
                ok, frame = self.cap.read()
                now = time.monotonic()
                if not ok or frame is None:
                    raise RuntimeError("Camera read failed; check USB connection and camera ownership.")
                if frame.shape != (HEIGHT, WIDTH, 3):
                    raise RuntimeError(f"Expected {WIDTH}x{HEIGHT} BGR, got shape {frame.shape}")
                # The V4L2 backend exposes the dequeued buffer's timestamp.
                # cap.read() completion includes USB/queue/conversion delay and
                # must not be used as the measurement time for robot feedback.
                captured = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.
                if (not math.isfinite(captured) or captured <= 0.
                        or not -.002 <= now-captured <= 2.
                        or (self.frame_timestamp is not None and captured <= self.frame_timestamp)):
                    raise RuntimeError("Camera driver did not provide an increasing host-monotonic "
                                       "V4L2 frame timestamp; cannot align image and robot feedback.")
                with self.condition:
                    self.frame = frame
                    self.frame_timestamp = captured
                    self.count += 1
                    self.recent.append(now)
                    if len(self.recent) > 1:
                        self.fps = (len(self.recent) - 1) / (self.recent[-1] - self.recent[0])
                    if self.count > WARMUP:
                        if self.first_sample is None:
                            self.first_sample = now
                        self.last_sample = now
                        self.measured_count += 1
                    self.condition.notify_all()
                if self.limit and self.count >= self.limit:
                    break
        except Exception as exc:
            with self.condition:
                self.error = str(exc)
        finally:
            self.cap.release()
            with self.condition:
                self.done = True
                self.condition.notify_all()

    def latest(self, previous, *, with_timestamp=False):
        with self.condition:
            ready = self.condition.wait_for(lambda: self.count > previous or self.done, timeout=12)
            if not ready:
                raise RuntimeError("No camera frame arrived within 12 seconds")
            if self.error:
                raise RuntimeError(self.error)
            if self.count == previous:
                return None
            result = (self.count, self.frame, self.fps)
            return result + (self.frame_timestamp,) if with_timestamp else result

    def close(self):
        self.stop.set()
        self.thread.join(timeout=12)
        if self.thread.is_alive():
            raise RuntimeError("Camera read did not stop; the capture thread is still blocked")


def open_camera(device="auto"):
    """Open and verify the native 640x480@60 color stream for both viewers.

    RealSense exposes depth, infrared, metadata, and color as several V4L2
    nodes. The node numbers can change after a stream reset, so ``auto`` probes
    RGB interface's VIDEO_CAPTURE nodes (excluding metadata) and verifies a
    640x480 three-channel color frame.
    """
    import cv2
    if device in (None, "auto"):
        # D435i's USB interface 1.3 is the RGB color sensor. Interface 1.0
        # exposes depth/infrared streams; some of those also decode to 3-channel
        # BGR, so checking frame.shape alone can silently select grayscale IR.
        # The managed Ubuntu image does not always create /dev/v4l/by-path for
        # this user, but sysfs is available and stable across video-node renumbering.
        color_paths = []
        for sysfs in sorted(Path("/sys/class/video4linux").glob("video*")):
            try:
                if ":1.3/" in str(sysfs.resolve()):
                    color_paths.append(Path("/dev") / sysfs.name)
            except OSError:
                continue
        candidates = [str(path) for path in color_paths]
        if not candidates:
            raise RuntimeError("RealSense RGB interface (USB 1.3) is not present; reconnect the camera and retry")
    else:
        candidates = [str(device)]
    failures = []
    for candidate in candidates:
        try:
            capabilities = video_node_capabilities(candidate)
        except OSError as exc:
            failures.append(f"{candidate}: {camera_os_error(exc)}")
            continue
        if not capabilities & _V4L2_CAP_VIDEO_CAPTURE:
            failures.append(f"{candidate}: not a video capture node (e.g. UVC metadata); skipped")
            continue
        cap = cv2.VideoCapture(candidate, cv2.CAP_V4L2)
        if not cap.isOpened():
            failures.append(f"{candidate}: device is accessible, but OpenCV/V4L2 initialization failed "
                            "(camera may be busy or stream setup failed)")
            cap.release()
            continue
        try:
            for prop, value in (
                (cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV")),
                (cv2.CAP_PROP_FRAME_WIDTH, WIDTH),
                (cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT),
                (cv2.CAP_PROP_FPS, FPS),
                (cv2.CAP_PROP_CONVERT_RGB, 1),
            ):
                cap.set(prop, value)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 4)
            actual = tuple(cap.get(p) for p in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
            if any(abs(a - b) > 0.1 for a, b in zip(actual, (WIDTH, HEIGHT, FPS))):
                failures.append(f"{candidate}: negotiated {actual}")
                cap.release()
                continue
            ok, frame = cap.read()
            if not ok or frame is None or frame.shape != (HEIGHT, WIDTH, 3):
                failures.append(f"{candidate}: not BGR color ({None if frame is None else frame.shape})")
                cap.release()
                continue
            pixels = frame.reshape(-1, 3).astype("float32")
            channel_std = pixels.std(axis=0)
            chroma_difference = float((abs(pixels[:, 0] - pixels[:, 1]).mean() + abs(pixels[:, 1] - pixels[:, 2]).mean()))
            print(
                f"Selected RGB color node: {candidate} | BGR channel std="
                f"({channel_std[0]:.1f},{channel_std[1]:.1f},{channel_std[2]:.1f}) | "
                f"mean chroma difference={chroma_difference:.1f}",
                flush=True,
            )
            return cap
        except Exception as exc:
            failures.append(f"{candidate}: {exc}")
            cap.release()

    raise RuntimeError(
        f"Cannot open a {WIDTH}x{HEIGHT}@{FPS} BGR color stream:\n  " + "\n  ".join(failures)
        + "\nCheck camera owners in the robot's terminal: sudo fuser -v " + " ".join(candidates)
        + "\nFor the underlying OpenCV error, rerun with OPENCV_VIDEOIO_DEBUG=1 OPENCV_LOG_LEVEL=DEBUG."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="auto", help="RealSense color V4L2 device, or auto to probe all /dev/video* nodes")
    parser.add_argument("--no-display", action="store_true", help="measure capture FPS without a window")
    parser.add_argument("--frames", type=int, default=0, help=f"stop after N measured frames, excluding {WARMUP} warm-up frames; 0 runs until quit")
    parser.add_argument("--intrinsics", type=Path, default=INTRINSICS_FILE)
    args = parser.parse_args()
    if args.frames < 0:
        parser.error("--frames must be nonnegative")
    if not args.no_display and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        raise RuntimeError("No desktop display. Run in an Ubuntu desktop/X11 terminal, or use --no-display --frames 600.")

    import cv2

    # Small VGA frames do not benefit from an eight-thread conversion pool.
    cv2.setNumThreads(1)

    calibration = json.loads(args.intrinsics.read_text(encoding="utf-8"))
    stream = calibration["stream"]
    if (stream["width"], stream["height"], stream["fps"]) != (WIDTH, HEIGHT, FPS):
        raise RuntimeError(f"Intrinsics file must describe color {WIDTH}x{HEIGHT}@{FPS}")
    camera_matrix = calibration["camera_matrix"]
    distortion_coefficients = calibration["distortion_coefficients"]
    print(f"Python: {sys.executable} | OpenCV: {cv2.__version__}", flush=True)
    print(f"Intrinsics: {args.intrinsics}\nK = {camera_matrix}\nD = {distortion_coefficients}", flush=True)

    cap = open_camera(args.device)

    reader = CameraReader(cap, args.frames)
    last_count = 0
    last_report = time.monotonic()
    preview_times = deque(maxlen=FPS + 1)
    try:
        if not args.no_display:
            cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
            print("Q/Esc: quit | S: save raw image to carmer/snapshots/", flush=True)
        while True:
            latest = reader.latest(last_count)
            if latest is None:
                break
            last_count, raw, capture_fps = latest
            now = time.monotonic()
            if now - last_report >= 5:
                print(f"Read {last_count} frames | capture {capture_fps:.2f} FPS", flush=True)
                last_report = now
            if args.no_display:
                continue
            preview_times.append(now)
            preview_fps = (len(preview_times) - 1) / (preview_times[-1] - preview_times[0]) if len(preview_times) > 1 else 0.0
            preview = raw.copy()
            for y, text in (
                (24, f"640x480 @ 60 | Capture: {capture_fps:.1f} FPS"),
                (48, f"Preview: {preview_fps:.1f} FPS | Q/Esc: quit | S: save"),
            ):
                cv2.putText(preview, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(preview, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.imshow(WINDOW, preview)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key in (ord("s"), ord("S")):
                directory = SCRIPT_DIR / "snapshots"
                directory.mkdir(exist_ok=True)
                path = directory / f"color_{datetime.now():%Y%m%d_%H%M%S_%f}.png"
                if not cv2.imwrite(str(path), raw):
                    raise RuntimeError(f"Could not save {path}")
                print(f"Saved raw {WIDTH}x{HEIGHT} image: {path}", flush=True)
    finally:
        try:
            reader.close()
        finally:
            if not args.no_display:
                cv2.destroyAllWindows()
        if reader.measured_count > 1:
            duration = reader.last_sample - reader.first_sample
            measured = (reader.measured_count - 1) / duration
            print(f"Measured {reader.measured_count} frames over {duration:.3f} s: {measured:.3f} FPS (first {WARMUP} frames excluded)", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
