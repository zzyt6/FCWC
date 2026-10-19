#!/usr/bin/env python3
"""Measure the real cart from AprilTags 0/1, without commanding the robot.

Run in CWC; see CART_POSE_MEASUREMENT.md for setup and accuracy tests.
The cart origin is the ground projection of the 90 x 50 cm top-board center.
X is forward, Y left, Z up. Both printed tag frames have these same axes.
"""
import argparse
from contextlib import ExitStack
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time


def run_tf_worker():
    """ROS Foxy runs in system Python, independently of CWC's Python/OpenCV.

    stdin/stdout carry timestamped read requests only. There are no publishers,
    service calls, action clients, or fallback lookups at the latest TF time.
    """
    import rclpy
    from rclpy.duration import Duration
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.time import Time
    from tf2_ros import Buffer, TransformListener

    rclpy.init(args=[])
    node = rclpy.create_node('cart_pose_measure_tf_reader')
    buffer = Buffer(cache_time=Duration(seconds=10.))
    listener = TransformListener(buffer, node)
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()
    print(json.dumps({'ready': True}), flush=True)
    try:
        for line in sys.stdin:
            request = json.loads(line)
            response = {'id': request['id'], 'valid': False}
            try:
                stamp = float(request['stamp'])
                # Live measurements use host Unix time; simulation time is unsupported.
                if not math.isfinite(stamp) or abs(time.time() - stamp) > 2.:
                    raise ValueError('image timestamp is not recent host Unix time')
                transform = buffer.lookup_transform(
                    'map', 'base_link', Time(nanoseconds=round(stamp * 1e9)),
                    timeout=Duration(seconds=.08))
                t, q = transform.transform.translation, transform.transform.rotation
                response.update(valid=True, position=[t.x, t.y, t.z],
                                quaternion_xyzw=[q.x, q.y, q.z, q.w],
                                stamp=stamp)
            except Exception as exc:
                response['reason'] = str(exc)
            print(json.dumps(response, allow_nan=False), flush=True)
    finally:
        executor.shutdown()
        thread.join(timeout=2.)
        del listener
        node.destroy_node()
        rclpy.shutdown()


# This branch must precede numpy/OpenCV imports: Foxy uses Python 3.8, CWC 3.10.
if __name__ == '__main__' and sys.argv[1:] == ['--tf-worker']:
    run_tf_worker()
    raise SystemExit(0)

import cv2
import numpy as np

if __package__:
    from .apriltag_detector import (DetectionWorker, TagPoseEstimator, load_intrinsics,
                                   make_detector, rotation_to_rpy)
    from .camera_extrinsics import ZeroWaistCameraExtrinsics
    from .cart_pose import (TAG_SIZE, BOARD_HEIGHT, TAG_CENTERS, TAG_CORNERS, BOARD_CORNERS,
                            CartPoseEstimator, transform, rotation_distance, level_from_robot, frame_poses)
    from .opencv_camera_viewer import CameraReader, INTRINSICS_FILE, WARMUP, open_camera
    from .pelvis_attitude import DEFAULT_URDF, G1Kinematics, PelvisCompensator
    from .robot_feedback import RobotFeedback
else:
    from apriltag_detector import (DetectionWorker, TagPoseEstimator, load_intrinsics,
                                  make_detector, rotation_to_rpy)
    from camera_extrinsics import ZeroWaistCameraExtrinsics
    from cart_pose import (TAG_SIZE, BOARD_HEIGHT, TAG_CENTERS, TAG_CORNERS, BOARD_CORNERS,
                            CartPoseEstimator, transform, rotation_distance, level_from_robot, frame_poses)
    from opencv_camera_viewer import CameraReader, INTRINSICS_FILE, WARMUP, open_camera
    from pelvis_attitude import DEFAULT_URDF, G1Kinematics, PelvisCompensator
    from robot_feedback import RobotFeedback


FRAMES = ('camera', 'robot', 'level', 'map')
COMPARISON_FRAMES = tuple(f'{frame}_{source}' for frame in FRAMES[1:] for source in ('urdf', 'provided'))
ALL_FRAMES = FRAMES + COMPARISON_FRAMES
WINDOW = 'Cart pose measurement - Q/Esc quits'
LOG_DIR = Path(__file__).resolve().parent / 'log'
PROVIDED_EXTRINSICS_FILE = Path(__file__).with_name('cart_camera_extrinsics.json')


def quaternion_rotation(xyzw):
    x, y, z, w = np.asarray(xyzw, dtype=float)
    norm = math.sqrt(x*x + y*y + z*z + w*w)
    if not math.isfinite(norm) or not .5 <= norm <= 1.5:
        raise ValueError('Invalid TF quaternion')
    x, y, z, w = np.array([x, y, z, w]) / norm
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


class MapTransformReader:
    def __init__(self, setup):
        env = dict(os.environ)
        for key in ('PYTHONPATH', 'PYTHONHOME', 'LD_LIBRARY_PATH', 'LD_PRELOAD', 'AMENT_PREFIX_PATH',
                    'CMAKE_PREFIX_PATH', 'COLCON_PREFIX_PATH'):
            env.pop(key, None)
        self.responses = queue.Queue()
        self.sequence = 0
        self.process = subprocess.Popen(
            ['/bin/bash', '-c', 'source "$1" >/dev/null && exec /usr/bin/python3 -B "$2" --tf-worker',
             'cart-pose-tf', str(setup), str(Path(__file__).resolve())],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1, env=env)
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()
        try:
            ready = self.responses.get(timeout=10.)
            if not ready.get('ready'):
                raise RuntimeError('ROS TF reader failed to start; see its stderr')
        except Exception:
            self.close()
            raise

    def _read(self):
        for line in self.process.stdout:
            try:
                self.responses.put(json.loads(line))
            except ValueError:
                pass
        self.responses.put({'valid': False, 'reason': 'ROS TF reader exited'})

    def lookup(self, stamp):
        if self.process.poll() is not None:
            return dict(valid=False, reason='ROS TF reader exited')
        self.sequence += 1
        try:
            self.process.stdin.write(json.dumps({'id': self.sequence, 'stamp': stamp})+'\n')
            self.process.stdin.flush()
            deadline = time.monotonic()+.3
            while time.monotonic() < deadline:
                response = self.responses.get(timeout=max(.001, deadline-time.monotonic()))
                if response.get('id') != self.sequence:
                    continue
                if response['valid']:
                    position = np.asarray(response['position'], dtype=float)
                    if not np.isfinite(position).all():
                        raise ValueError('Invalid TF position')
                    response['T'] = transform(quaternion_rotation(response['quaternion_xyzw']), position)
                return response
        except (queue.Empty, OSError, ValueError) as exc:
            return dict(valid=False, reason='TF read failed: '+str(exc))
        return dict(valid=False, reason='TF response timeout')

    def close(self):
        if self.process.stdin:
            try:
                self.process.stdin.close()
            except OSError:
                pass
        try:
            self.process.wait(timeout=2.)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            try:
                self.process.wait(timeout=2.)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=2.)
        self.thread.join(timeout=1.)
        self.process.stdout.close()


def measurement_poses(estimate, compensation=None, map_from_robot=None, *,
                      extrinsics_mode='urdf', provided_extrinsics=None):
    """Reuse one PnP, feedback sample and map lookup for both extrinsic choices.

    In compare mode the original unsuffixed fields remain URDF-based. Explicit
    *_urdf and *_provided fields make same-frame comparisons unambiguous.
    """
    if extrinsics_mode not in ('urdf', 'provided', 'compare'):
        raise ValueError('Unknown extrinsics mode: '+extrinsics_mode)
    poses = frame_poses(estimate, compensation, map_from_robot)
    if extrinsics_mode == 'urdf' or 'robot' not in poses:
        return poses
    if provided_extrinsics is None:
        raise ValueError('Provided camera extrinsics are required for this mode')
    nominal = transform(np.asarray(compensation['R_camera_to_pelvis']),
                        np.asarray(compensation['camera_position_in_pelvis_m']))
    supplied = frame_poses(estimate, compensation, map_from_robot,
                           provided_extrinsics.from_nominal(nominal))
    if extrinsics_mode == 'provided':
        return supplied
    for frame in FRAMES[1:]:
        if frame in poses:
            poses[frame+'_urdf'] = poses[frame]
            poses[frame+'_provided'] = supplied[frame]
    return poses


def displayed_frames(extrinsics_mode):
    return ('camera', *COMPARISON_FRAMES) if extrinsics_mode == 'compare' else FRAMES


def extrinsics_differences(poses):
    differences = {}
    for frame in FRAMES[1:]:
        a, b = poses.get(frame+'_urdf'), poses.get(frame+'_provided')
        if a is not None and b is not None:
            differences.update(zip((f'provided_minus_urdf_{frame}_{axis}' for axis in AXES[:3]),
                                   (b[:3, 3]-a[:3, 3]).tolist()))
            differences[f'provided_vs_urdf_{frame}_rotation_deg'] = rotation_distance(a[:3, :3], b[:3, :3])
    return differences


def pose_values(pose):
    rpy, _ = rotation_to_rpy(pose[:3, :3])
    return [*pose[:3, 3], *np.rad2deg(rpy)]


BASE_FIELDS = ['timestamp_unix', 'frame', 'age_ms', 'tag_ids', 'camera_reason', 'robot_reason',
               'map_reason', 'rmse_px', 'baseline_m', 'single_position_difference_m',
               'single_rotation_difference_deg', 'ambiguity_deg', 'ambiguity_gap_px']
AXES = ('x_m', 'y_m', 'z_m', 'roll_deg', 'pitch_deg', 'yaw_deg')
EXTRINSICS_FIELDS = ['extrinsics_mode', 'waist_yaw_deg', 'pelvis_roll_deg', 'pelvis_pitch_deg']
DIFFERENCE_FIELDS = [name for frame in FRAMES[1:] for name in
                     (*[f'provided_minus_urdf_{frame}_{axis}' for axis in AXES[:3]],
                      f'provided_vs_urdf_{frame}_rotation_deg')]
CSV_FIELDS = (BASE_FIELDS + EXTRINSICS_FIELDS
              + [f'{name}_{axis}' for name in (*ALL_FRAMES, 'tag0_camera', 'tag1_camera') for axis in AXES]
              + DIFFERENCE_FIELDS)


def csv_row(packet, estimate, poses, stamp, age, robot_reason, map_reason, *, extrinsics_mode='urdf'):
    row = dict(timestamp_unix=stamp, frame=packet['frame_number'], age_ms=age*1000,
               tag_ids=' '.join(map(str, estimate['tag_ids'])), camera_reason=estimate['reason'],
               robot_reason=robot_reason, map_reason=map_reason, extrinsics_mode=extrinsics_mode)
    compensation = packet.get('pelvis_compensation') or {}
    for field, key in [('waist_yaw_deg', 'waist_yaw_deg'), ('pelvis_roll_deg', 'roll_deg'),
                       ('pelvis_pitch_deg', 'pitch_deg')]:
        row[field] = compensation.get(key)
    for field in BASE_FIELDS[7:]:
        row[field] = estimate.get(field)
    all_poses = dict(poses)
    for tag_id, single in estimate.get('singles', {}).items():
        if single['valid']:
            all_poses[f'tag{tag_id}_camera'] = single['T']
    for name, pose in all_poses.items():
        row.update(zip([f'{name}_{axis}' for axis in AXES], pose_values(pose)))
    row.update(extrinsics_differences(poses))
    return row


def summarize(path, reference):
    with Path(path).open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    selected = [row for row in rows if row.get(reference+'_x_m', '')]
    result = dict(file=str(path), reference=reference, total_frames=len(rows), valid_frames=len(selected))
    if not selected:
        return result
    values = np.array([[float(row[reference+'_'+axis]) for axis in AXES] for row in selected])
    mean = values.mean(axis=0)
    for index in (3, 4, 5):
        radians = np.deg2rad(values[:, index])
        mean[index] = math.degrees(math.atan2(np.sin(radians).mean(), np.cos(radians).mean()))
    error = values-mean
    error[:, 3:] = (error[:, 3:]+180.) % 360.-180.
    result.update(mean=dict(zip(AXES, mean.tolist())),
                  std=dict(zip(AXES, error.std(axis=0).tolist())),
                  p95_position_deviation_m=float(np.percentile(np.linalg.norm(error[:, :3], axis=1), 95)))
    return result


def compare(first, second, reference):
    a, b = summarize(first, reference), summarize(second, reference)
    if not a.get('mean') or not b.get('mean'):
        raise ValueError('Both recordings need valid poses in '+reference)
    delta = np.array([b['mean'][axis]-a['mean'][axis] for axis in AXES[:3]])
    yaw = math.radians(a['mean']['yaw_deg'])
    local = np.array([[math.cos(yaw), math.sin(yaw)], [-math.sin(yaw), math.cos(yaw)]]) @ delta[:2]
    return dict(reference=reference, delta_xyz_m=delta.tolist(),
                delta_xy_in_initial_cart_heading_m=local.tolist(),
                delta_yaw_deg=(b['mean']['yaw_deg']-a['mean']['yaw_deg']+180.) % 360.-180.,
                first=a, second=b)


def draw(packet, estimate, poses, matrix, distortion, robot_reason, map_reason, *, extrinsics_mode='urdf'):
    image = packet['frame'].copy()
    for tag in packet['tags']:
        if tag['id'] not in TAG_CENTERS:
            continue
        corners = np.rint(tag['corners_px']).astype(np.int32)
        cv2.polylines(image, [corners], True, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(image, str(tag['id']), tuple(corners[0]), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 0), 1)
    if 'camera' in poses:
        pose = poses['camera']
        if np.all((BOARD_CORNERS @ pose[:3, :3].T + pose[:3, 3])[:, 2] > .01):
            pixels = cv2.projectPoints(BOARD_CORNERS, cv2.Rodrigues(pose[:3, :3])[0],
                                       pose[:3, 3], matrix, distortion)[0]
            if np.isfinite(pixels).all() and np.max(np.abs(pixels)) < 1e6:
                cv2.polylines(image, [np.rint(pixels).astype(np.int32)], True, (255, 255, 0), 2, cv2.LINE_AA)
    lines = ['CART ORIGIN: board center projected to ground',
             'X forward | Y left | Z up', 'Cyan: measured 90 x 50 cm top board',
             'Camera extrinsics: '+extrinsics_mode,
             'Tags: '+str(estimate['tag_ids'])+' | '+estimate['reason'],
             'Reprojection RMS: '+str(round(estimate.get('rmse_px', -1), 3))+' px']
    for name in displayed_frames(extrinsics_mode):
        if name in poses:
            x, y, z, roll, pitch, yaw = pose_values(poses[name])
            lines += [f'{name}: XYZ {x:+.3f} {y:+.3f} {z:+.3f} m',
                      f'  RPY {roll:+.1f} {pitch:+.1f} {yaw:+.1f} deg']
        else:
            lines += [name+': unavailable']
    lines += ['robot: '+robot_reason[:58], 'map: '+map_reason[:60]]
    if 'baseline_m' in estimate:
        lines += [f"Single-tag baseline: {estimate['baseline_m']*100:.2f} cm (29.00)",
                  f"Single-tag cart gap: {estimate['single_position_difference_m']*100:.2f} cm"]
    if 'level_provided' in poses:
        delta = poses['level_provided'][:3, 3]-poses['level_urdf'][:3, 3]
        lines += [f'Level delta (provided - URDF):',
                  f'  XYZ {delta[0]:+.4f} {delta[1]:+.4f} {delta[2]:+.4f} m']
    canvas = np.full((max(image.shape[0], 24*len(lines)+20), image.shape[1]+480, 3), 28, np.uint8)
    canvas[:image.shape[0], :image.shape[1]] = image
    for index, line in enumerate(lines):
        cv2.putText(canvas, line, (image.shape[1]+12, 26+index*24),
                    cv2.FONT_HERSHEY_SIMPLEX, .47, (225, 225, 225), 1, cv2.LINE_AA)
    return canvas


def print_extrinsics(provided):
    print('Zero waist yaw; camera optical -> physical pelvis; translation in meters.')
    print('URDF:\n'+np.array2string(provided.urdf_zero, precision=11, suppress_small=True))
    print('Provided:\n'+np.array2string(provided.provided_zero, precision=11, suppress_small=True))
    print('Translation difference (provided - URDF), m:',
          provided.provided_zero[:3, 3]-provided.urdf_zero[:3, 3])
    print('Rotation difference, deg:', rotation_distance(provided.urdf_zero[:3, :3],
                                                        provided.provided_zero[:3, :3]))


def run(args):
    cv2.setNumThreads(1)
    matrix, distortion = load_intrinsics(args.intrinsics)
    estimator = CartPoseEstimator(matrix, distortion, args.tag_ids, args.max_rmse_px)
    if not args.no_display and not (os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')):
        raise RuntimeError('No desktop display; add --no-display')
    model = None if args.camera_only else G1Kinematics(args.robot_urdf)
    provided = (ZeroWaistCameraExtrinsics(model, args.extrinsics_file)
                if args.extrinsics != 'urdf' else None)
    output = args.output or LOG_DIR / ('cart_pose_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'.csv')
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                    tag_centers_m={str(k): v.tolist() for k, v in TAG_CENTERS.items()},
                    tag_size_m=TAG_SIZE, board_size_m=[.90, .50], board_height_m=BOARD_HEIGHT,
                    cart_axes='X forward, Y left, Z up; origin = board center ground projection',
                    camera_axes='X image right, Y image down, Z optical forward',
                    robot_frame='physical pelvis base_link', level_frame='gravity-level axes at pelvis origin',
                    map_frame='map, via TF map <- robot base_link at image receive time minus latency',
                    extrinsics_mode=args.extrinsics,
                    primary_pose_source='provided' if args.extrinsics == 'provided' else 'urdf',
                    extrinsics=(provided.metadata() if provided else
                                'nominal robot URDF + assumed optical axes, not separately calibrated'),
                    comparison_fields='*_urdf and *_provided use the same image, feedback and map transform; '
                                      'unsuffixed robot/level/map are URDF in compare mode',
                    smoothing='none', intrinsics=str(args.intrinsics), camera_matrix=matrix.tolist(),
                    distortion=distortion.tolist(),
                    arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
    # Exclusive creation prevents accidentally overwriting a ground-truth comparison run.
    with ExitStack() as resources:
        stream = resources.enter_context(output.open('x', newline=''))
        output.with_suffix('.meta.json').write_text(json.dumps(metadata, indent=2)+'\n')
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        mapper = None
        if args.map:
            mapper = MapTransformReader(args.ros_setup)
            resources.callback(mapper.close)
        compensator = None
        if not args.camera_only:
            feedback = RobotFeedback(args.network_interface)
            resources.callback(feedback.close)
            compensator = PelvisCompensator(feedback, model, source='imu',
                                           timeout_s=.1, sync_tolerance_s=.03,
                                           camera_latency_s=args.camera_latency_ms/1000.)
        reader = CameraReader(open_camera(args.device), 0)
        try:
            worker = DetectionWorker(reader, make_detector(args.quad_decimate),
                                     TagPoseEstimator(matrix, distortion, TAG_SIZE),
                                     pelvis_compensator=compensator)
        except Exception:
            reader.close()
            raise
        resources.callback(worker.close)
        if not args.no_display:
            cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
            resources.callback(cv2.destroyAllWindows)
        print('Read-only cart measurement. No pose smoothing. CSV: '+str(output), flush=True)
        print('Camera extrinsics: '+args.extrinsics, flush=True)
        if provided is not None:
            print_extrinsics(provided)
        print('Map: '+('enabled, timestamped TF lookup' if mapper else 'disabled; use --map'), flush=True)
        previous, started, last_report = 0, None, 0.
        try:
            while True:
                packet = worker.latest(previous)
                if packet is None:
                    break
                previous = packet['frame_number']
                if previous <= WARMUP:
                    continue
                now = time.monotonic()
                if started is None:
                    started = now
                if args.duration and now-started >= args.duration:
                    break
                captured = packet['frame_timestamp_monotonic']-args.camera_latency_ms/1000.
                # These clocks are sampled together on this host. Neither is exposure time.
                stamp = time.time() - (time.monotonic()-captured)
                estimate = estimator.estimate(packet['tags'])
                compensation = packet.get('pelvis_compensation')
                robot_reason = compensation.get('reason', 'invalid') if compensation else 'disabled'
                map_result = None
                map_reason = 'disabled (use --map)'
                if mapper:
                    map_reason = 'camera or robot pose unavailable'
                    if estimate['valid'] and compensation and compensation.get('valid'):
                        map_result = mapper.lookup(stamp)
                        map_reason = 'ok' if map_result['valid'] else map_result['reason']
                age = time.monotonic()-captured
                if not 0 <= age <= args.max_age:
                    estimate.update(valid=False, reason='image_stale')
                    robot_reason = map_reason = 'image_stale'
                poses = measurement_poses(estimate, compensation, map_result,
                                          extrinsics_mode=args.extrinsics, provided_extrinsics=provided)
                writer.writerow(csv_row(packet, estimate, poses, stamp, age, robot_reason, map_reason,
                                        extrinsics_mode=args.extrinsics))
                if now-last_report >= 1.:
                    stream.flush()
                    status = ' | '.join(f'{name}: '+np.array2string(np.array(pose_values(poses[name])), precision=4)
                                        for name in displayed_frames(args.extrinsics) if name in poses)
                    print(f"tags={estimate['tag_ids']} {estimate['reason']} | extrinsics={args.extrinsics} | "
                          f"XYZ(m), RPY(deg) | {status} | map={map_reason}", flush=True)
                    if 'level_provided' in poses:
                        delta = poses['level_provided'][:3, 3]-poses['level_urdf'][:3, 3]
                        print('level provided - URDF XYZ(m): '+np.array2string(delta, precision=5), flush=True)
                    last_report = now
                if not args.no_display:
                    cv2.imshow(WINDOW, draw(packet, estimate, poses, matrix, distortion, robot_reason, map_reason,
                                           extrinsics_mode=args.extrinsics))
                    if cv2.waitKey(1) & 0xff in (27, ord('q'), ord('Q')):
                        break
        except KeyboardInterrupt:
            pass
    report = {name: summarize(output, name) for name in
              (ALL_FRAMES if args.extrinsics == 'compare' else FRAMES)}
    output.with_suffix('.summary.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--intrinsics', type=Path, default=INTRINSICS_FILE)
    parser.add_argument('--robot-urdf', type=Path, default=DEFAULT_URDF)
    parser.add_argument('--extrinsics', choices=('urdf', 'provided', 'compare'), default='urdf',
                        help='camera -> pelvis transform; compare uses the same observations for both')
    parser.add_argument('--extrinsics-file', type=Path, default=PROVIDED_EXTRINSICS_FILE,
                        help='provided camera optical -> pelvis matrix at zero waist yaw, in meters')
    parser.add_argument('--network-interface', default='eth0')
    parser.add_argument('--camera-only', action='store_true', help='do not subscribe to robot feedback')
    parser.add_argument('--map', action='store_true', help='read map <- robot base_link from existing ROS TF')
    parser.add_argument('--ros-setup', type=Path, default=Path('/opt/ros/foxy/setup.bash'))
    parser.add_argument('--no-display', action='store_true')
    parser.add_argument('--duration', type=float, default=0., help='record seconds after camera warmup; 0 = until Q/Ctrl-C')
    parser.add_argument('--output', type=Path,
                        help='override CSV path; default carmer/log/cart_pose_YYYYmmdd_HHMMSS_microseconds.csv (local time)')
    parser.add_argument('--tag-ids', nargs='+', type=int, choices=(0, 1), default=[0, 1])
    parser.add_argument('--quad-decimate', type=float, default=1., help='1 = full resolution for accuracy measurements')
    parser.add_argument('--max-rmse-px', type=float, default=2., help='reject if either tag has larger reprojection RMS')
    parser.add_argument('--max-age', type=float, default=.3, help='maximum age in seconds, including configured latency')
    parser.add_argument('--camera-latency-ms', type=float, default=0.)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--summarize', type=Path, help='summarize an existing CSV; no camera/robot access')
    mode.add_argument('--compare', type=Path, nargs=2, metavar=('BEFORE', 'AFTER'), help='compare two static CSV recordings')
    mode.add_argument('--show-extrinsics', action='store_true', help='print both zero-waist matrices; no camera/robot access')
    parser.add_argument('--reference', choices=ALL_FRAMES, default='map', help='frame for summary/comparison')
    args = parser.parse_args()
    for name in ('duration', 'camera_latency_ms'):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            parser.error('--'+name.replace('_', '-')+' must be finite and nonnegative')
    for name in ('quad_decimate', 'max_rmse_px', 'max_age'):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error('--'+name.replace('_', '-')+' must be finite and positive')
    if args.quad_decimate < 1:
        parser.error('--quad-decimate must be at least 1')
    if args.map and args.camera_only:
        parser.error('--map requires robot feedback; remove --camera-only')
    if args.camera_only and args.extrinsics != 'urdf':
        parser.error('--extrinsics provided/compare requires robot feedback; remove --camera-only')
    if args.summarize:
        print(json.dumps(summarize(args.summarize, args.reference), indent=2))
    elif args.compare:
        print(json.dumps(compare(*args.compare, args.reference), indent=2))
    elif args.show_extrinsics:
        print_extrinsics(ZeroWaistCameraExtrinsics(G1Kinematics(args.robot_urdf), args.extrinsics_file))
    else:
        run(args)


if __name__ == '__main__':
    main()
