#!/usr/bin/env python3
"""Space-to-capture ChArUco calibration and known-board-pose extrinsics.

Default: move the board through diverse poses; SPACE saves a valid raw frame.
At sample_count, calibrate intrinsics automatically. --reference measures the
fixed camera-to-torso transform only when the board's torso pose is supplied.
No robot command, DDS subscription, or motor-control interface is used.
"""
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np

if __package__:
    from .apriltag_detector import load_intrinsics
    from .camera_extrinsics import rigid_transform, urdf_camera_to_torso
    from .cart_heading import rotation_from_rpy_degrees
    from .opencv_camera_viewer import CameraReader, HEIGHT, WIDTH, WARMUP, open_camera
else:
    from apriltag_detector import load_intrinsics
    from camera_extrinsics import rigid_transform, urdf_camera_to_torso
    from cart_heading import rotation_from_rpy_degrees
    from opencv_camera_viewer import CameraReader, HEIGHT, WIDTH, WARMUP, open_camera


CONFIG = Path(__file__).with_name('charuco_config.json')
WINDOW = 'ChArUco calibration | SPACE capture | Q / ESC quit'
BOARD_FRAME = ('OpenCV ChArUco board: origin at top-left OUTER chessboard corner, '
               'X right, Y down on printed face, Z into the board; white paper margin excluded')


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    temporary.replace(path)


def load_config(path):
    path = Path(path).resolve()
    config = json.loads(path.read_text(encoding='utf-8'))
    board = config['board']
    for key in ('squares_x', 'squares_y'):
        if type(board[key]) is not int or board[key] < 3:
            raise ValueError('board.'+key+' must be an integer >= 3')
    lengths = [board['marker_length_m'], board['square_length_m']]
    if not all(type(v) in (int, float) and math.isfinite(v) for v in lengths) or not 0 < lengths[0] < lengths[1]:
        raise ValueError('Require 0 < marker_length_m < square_length_m')
    if type(board['legacy_pattern']) is not bool:
        raise ValueError('legacy_pattern must be boolean')
    if not isinstance(board['dictionary'], str) or not board['dictionary'].startswith('DICT_'):
        raise ValueError('Expected an OpenCV DICT_* dictionary name')
    if not hasattr(cv2.aruco, board['dictionary']):
        raise ValueError('Unknown dictionary: '+board['dictionary'])
    if (config['camera']['width'], config['camera']['height']) != (WIDTH, HEIGHT):
        raise ValueError('This camera pipeline requires native 640x480 color images')
    for section, key in [('capture', 'sample_count'), ('capture', 'min_charuco_corners'),
                         ('calibration', 'min_views'), ('reference', 'sample_count')]:
        if type(config[section][key]) is not int or config[section][key] < 1:
            raise ValueError(section+'.'+key+' must be a positive integer')
    count = (board['squares_x']-1)*(board['squares_y']-1)
    if not 6 <= config['capture']['min_charuco_corners'] <= count:
        raise ValueError('min_charuco_corners must be between 6 and the board corner count')
    if not 8 <= config['calibration']['min_views'] <= config['capture']['sample_count']:
        raise ValueError('Require 8 <= min_views <= sample_count')
    for section, keys in [('capture', ('min_coverage_fraction', 'max_frame_age_s')),
                          ('calibration', ('max_view_rms_px', 'max_global_rms_px', 'min_normal_span_deg')),
                          ('reference', ('max_view_rms_px', 'max_translation_spread_m', 'max_rotation_spread_deg'))]:
        for key in keys:
            value = config[section][key]
            if type(value) not in (float, int) or not math.isfinite(value) or value <= 0:
                raise ValueError(section+'.'+key+' must be finite and positive')
    if config['capture']['min_coverage_fraction'] >= 1:
        raise ValueError('min_coverage_fraction must be < 1')
    if type(config['calibration']['save_undistorted_images']) is not bool:
        raise ValueError('save_undistorted_images must be boolean')
    for key in ('position_in_torso_m', 'rpy_in_torso_deg'):
        value = config['reference'][key]
        if value is not None and (np.asarray(value).shape != (3,) or not np.isfinite(np.asarray(value, float)).all()):
            raise ValueError('reference.'+key+' must be null or three finite numbers')
    for key, value in config['output'].items():
        config['output'][key] = str((path.parent / Path(value).expanduser()).resolve())
    config['robot_urdf'] = str((path.parent / Path(config['robot_urdf']).expanduser()).resolve())
    config['config_file'] = str(path)
    return config


def make_board(config):
    c = config['board']
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, c['dictionary']))
    board = cv2.aruco.CharucoBoard((c['squares_x'], c['squares_y']), c['square_length_m'],
                                  c['marker_length_m'], dictionary)
    board.setLegacyPattern(c['legacy_pattern'])
    return board


class BoardDetector:
    def __init__(self, board, config, intrinsics=None):
        self.board, self.config = board, config
        parameters = cv2.aruco.CharucoParameters()
        if intrinsics is not None:
            parameters.cameraMatrix, parameters.distCoeffs = intrinsics
        self.detector = cv2.aruco.CharucoDetector(board, parameters)

    def detect(self, image):
        if image.shape[:2] != (HEIGHT, WIDTH):
            raise ValueError('Input must be a native 640x480 image, without resizing/cropping')
        corners, ids, marker_corners, marker_ids = self.detector.detectBoard(image)
        result = dict(valid=False, reason='no_charuco_corners', corners=corners, ids=ids,
                      marker_corners=marker_corners, marker_ids=marker_ids, coverage=0.)
        if ids is None or len(ids) < self.config['capture']['min_charuco_corners']:
            result['reason'] = 'too_few_charuco_corners'
            return result
        if self.board.checkCharucoCornersCollinear(ids):
            result['reason'] = 'collinear_corners'
            return result
        coverage = cv2.contourArea(cv2.convexHull(corners.astype(np.float32))) / (WIDTH*HEIGHT)
        result['coverage'] = float(coverage)
        if coverage < self.config['capture']['min_coverage_fraction']:
            result['reason'] = 'board_too_small_in_image'
            return result
        result.update(valid=True, reason='ready')
        return result

    def observation(self, detection, filename):
        objects, images = self.board.matchImagePoints(detection['corners'], detection['ids'])
        return dict(filename=str(filename), object_points=objects.reshape(-1, 3).tolist(),
                    image_points=images.reshape(-1, 2).tolist(),
                    charuco_ids=detection['ids'].reshape(-1).tolist(),
                    coverage_fraction=detection['coverage'])


def image_points(observations):
    objects = [np.asarray(item['object_points'], dtype=np.float32) for item in observations]
    images = [np.asarray(item['image_points'], dtype=np.float32) for item in observations]
    return objects, images


def calibration_quality(matrix, distortion, rms, rvecs, settings):
    failures = []
    if not np.isfinite(matrix).all() or not np.isfinite(distortion).all() or not math.isfinite(rms):
        return ['nonfinite_calibration'], 0.
    if not (.2*WIDTH < matrix[0, 0] < 5*WIDTH and .2*HEIGHT < matrix[1, 1] < 5*WIDTH
            and 0 <= matrix[0, 2] < WIDTH and 0 <= matrix[1, 2] < HEIGHT):
        failures.append('implausible_intrinsics')
    normals = np.array([cv2.Rodrigues(rvec)[0][:, 2] for rvec in rvecs])
    span = float(np.rad2deg(np.arccos(np.clip(normals @ normals.T, -1, 1))).max())
    if span < settings['min_normal_span_deg']:
        failures.append('insufficient_board_tilt_diversity')
    if rms > settings['max_global_rms_px']:
        failures.append('global_reprojection_error_too_large')
    return failures, span


def calibrate_intrinsics(observations, config):
    settings = config['calibration']
    if len(observations) < settings['min_views']:
        raise ValueError('Not enough usable calibration views')
    used = list(range(len(observations)))
    rejected = []
    for _ in range(3):
        objects, images = image_points([observations[i] for i in used])
        result = cv2.calibrateCameraExtended(objects, images, (WIDTH, HEIGHT), None, None,
                                             criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-9))
        rms, matrix, distortion, rvecs, tvecs, std_intrinsics, _, errors = result
        errors = errors.reshape(-1)
        bad = np.flatnonzero(errors > settings['max_view_rms_px']).tolist()
        if not bad:
            break
        if len(used)-len(bad) < settings['min_views']:
            raise ValueError('Too few views remain after rejecting large reprojection errors')
        rejected.extend(dict(index=used[i], filename=observations[used[i]]['filename'], rms_px=float(errors[i])) for i in bad)
        used = [index for i, index in enumerate(used) if i not in bad]
    else:
        raise ValueError('Reprojection outlier removal did not converge; inspect saved images')
    failures, normal_span = calibration_quality(matrix, distortion, rms, rvecs, settings)
    views = []
    for index, rvec, tvec, error in zip(used, rvecs, tvecs, errors):
        views.append(dict(filename=observations[index]['filename'], rms_px=float(error),
                          T_camera_optical_board=rigid_transform(cv2.Rodrigues(rvec)[0], tvec.reshape(3)).tolist()))
    report = dict(status='calibrated' if not failures else 'quality_failed', failures=failures,
                  rms_px=float(rms), usable_views=len(views), rejected_views=rejected,
                  board_normal_span_deg=normal_span, camera_matrix=matrix.tolist(),
                  distortion_coefficients=distortion.reshape(-1).tolist(),
                  intrinsics_standard_deviations=std_intrinsics.reshape(-1).tolist(), views=views)
    return matrix, distortion, report


def reference_transform(config):
    reference = config['reference']
    xyz, rpy = reference['position_in_torso_m'], reference['rpy_in_torso_deg']
    if xyz is None or rpy is None:
        raise ValueError('Camera-to-torso calibration needs independently measured reference.'
                         'position_in_torso_m AND reference.rpy_in_torso_deg; '
                         'moving an untracked board cannot determine these extrinsics')
    return rigid_transform(rotation_from_rpy_degrees(rpy), xyz)


def rotation_angle(rotation):
    return float(np.rad2deg(np.arccos(np.clip((np.trace(rotation)-1)/2, -1, 1))))


def solve_board_pose(observation, matrix, distortion, max_rms):
    objects, images = image_points([observation])
    objects, images = objects[0].astype(float), images[0].astype(float)
    _, rvecs, tvecs, _ = cv2.solvePnPGeneric(objects, images, matrix, distortion, flags=cv2.SOLVEPNP_IPPE)
    candidates = []
    for rvec, tvec in zip(rvecs, tvecs):
        rvec, tvec = cv2.solvePnPRefineLM(objects, images, matrix, distortion, rvec.copy(), tvec.copy())
        rotation = cv2.Rodrigues(rvec)[0]
        position = tvec.reshape(3)
        # ChArUco +Z points into the printed board, opposite to this project's AprilTag +Z.
        if (not np.isfinite(position).all() or np.any((objects @ rotation.T + position)[:, 2] <= 0)
                or rotation[:, 2] @ position <= 0):
            continue
        projected = cv2.projectPoints(objects, rvec, tvec, matrix, distortion)[0].reshape(-1, 2)
        rms = float(np.sqrt(np.mean(np.sum((projected-images)**2, axis=1))))
        candidates.append((rms, rigid_transform(rotation, position)))
    candidates.sort(key=lambda item: item[0])
    if not candidates or candidates[0][0] > max_rms:
        raise ValueError('Reference board pose has no acceptable front-facing solution')
    if len(candidates) > 1:
        gap = candidates[1][0]-candidates[0][0]
        angle = rotation_angle(candidates[0][1][:3, :3].T @ candidates[1][1][:3, :3])
        if gap < .1 and angle > 10.:
            raise ValueError('Ambiguous planar reference pose; use a clearer, oblique view')
    return candidates[0][1], candidates[0][0]


def calibrate_reference(observations, matrix, distortion, config):
    torso_board = reference_transform(config)
    settings = config['reference']
    if len(observations) < settings['sample_count']:
        raise ValueError('Not enough stationary reference photographs')
    solutions = [solve_board_pose(view, matrix, distortion, settings['max_view_rms_px']) for view in observations]
    transforms = [torso_board @ np.linalg.inv(solution[0]) for solution in solutions]
    u, _, vt = np.linalg.svd(sum(pose[:3, :3] for pose in transforms))
    rotation = u @ np.diag([1., 1., np.linalg.det(u @ vt)]) @ vt
    position = np.mean([pose[:3, 3] for pose in transforms], axis=0)
    calibrated = rigid_transform(rotation, position)
    translations = [float(np.linalg.norm(pose[:3, 3]-position)) for pose in transforms]
    angles = [rotation_angle(rotation.T @ pose[:3, :3]) for pose in transforms]
    if max(translations) > settings['max_translation_spread_m'] or max(angles) > settings['max_rotation_spread_deg']:
        raise ValueError('Reference estimates disagree: keep both board and torso stationary and verify sharp images')
    nominal = urdf_camera_to_torso(config['robot_urdf'])
    return dict(schema='cwc_camera_extrinsics_v1', status='calibrated',
                source_frame='camera_optical', target_frame='torso_link', translation_units='m',
                convention='p_torso = T_torso_camera_optical @ p_camera_optical (homogeneous column vectors)',
                board_frame=BOARD_FRAME, T_torso_board=torso_board.tolist(),
                T_torso_camera_optical=calibrated.tolist(),
                T_camera_optical_torso=np.linalg.inv(calibrated).tolist(),
                T_torso_camera_optical_urdf=nominal.tolist(),
                created_utc=datetime.now(timezone.utc).isoformat(),
                camera_matrix=matrix.tolist(), distortion_coefficients=distortion.reshape(-1).tolist(),
                translation_difference_from_urdf_m=(position-nominal[:3, 3]).tolist(),
                rotation_difference_from_urdf_deg=rotation_angle(nominal[:3, :3].T @ rotation),
                max_translation_spread_m=max(translations), max_rotation_spread_deg=max(angles),
                reference_views=[dict(filename=view['filename'], rms_px=rms,
                                      T_camera_optical_board=pose.tolist())
                                 for view, (pose, rms) in zip(observations, solutions)])


def panel(frame, detection, saved, target, mode, message):
    image = frame.copy()
    if detection['marker_ids'] is not None:
        cv2.aruco.drawDetectedMarkers(image, detection['marker_corners'], detection['marker_ids'])
    if detection['ids'] is not None:
        cv2.aruco.drawDetectedCornersCharuco(image, detection['corners'], detection['ids'])
    canvas = np.full((HEIGHT, WIDTH+440, 3), 28, np.uint8)
    canvas[:, :WIDTH] = image
    count = len(detection['ids']) if detection['ids'] is not None else 0
    lines = [('KNOWN-POSE REFERENCE' if mode == 'reference' else 'CAMERA INTRINSICS'),
             f'Saved: {saved} / {target}', f'ChArUco corners: {count}',
             f'Corner hull coverage: {detection["coverage"]*100:.1f}%',
             detection['reason'], 'SPACE: capture this raw image', 'Q / ESC: save partial session and quit',
             ('KEEP board and torso fixed at reference pose' if mode == 'reference'
              else 'Move / tilt board between captures'),
             'Use near/far, left/right, top/bottom views', message]
    for i, line in enumerate(lines):
        cv2.putText(canvas, line[:58], (WIDTH+12, 30+32*i), cv2.FONT_HERSHEY_SIMPLEX,
                    .46, (80, 225, 100) if detection['valid'] else (80, 190, 240), 1, cv2.LINE_AA)
    return canvas


def capture(detector, config, session, mode):
    if not (os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')):
        raise RuntimeError('SPACE capture requires a desktop/X11 display; use --images for offline calibration')
    target = config['reference']['sample_count'] if mode == 'reference' else config['capture']['sample_count']
    image_dir = session/'raw'
    image_dir.mkdir()
    observations, previous, message = [], 0, ''
    reader = CameraReader(open_camera(config['camera']['device']), 0)
    try:
        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
        while len(observations) < target:
            item = reader.latest(previous, with_timestamp=True)
            if item is None:
                break
            previous, frame, _, stamp = item
            detection = detector.detect(frame)
            if previous <= WARMUP:
                detection.update(valid=False, reason='camera_warming_up')
            cv2.imshow(WINDOW, panel(frame, detection, len(observations), target, mode, message))
            key = cv2.waitKey(1) & 0xff
            if key in (27, ord('q'), ord('Q')):
                break
            if key != 32:
                continue
            if not detection['valid']:
                message = 'Not saved: '+detection['reason']
                print(message, flush=True)
                continue
            if time.monotonic()-stamp > config['capture']['max_frame_age_s']:
                message = 'Not saved: image is stale'
                continue
            filename = image_dir/f'{len(observations)+1:03d}.png'
            if not cv2.imwrite(str(filename), frame):
                raise RuntimeError('Cannot write image: '+str(filename))
            observation = detector.observation(detection, filename)
            observation.update(captured_monotonic=stamp, saved_utc=datetime.now(timezone.utc).isoformat())
            observations.append(observation)
            write_json(session/'observations.json', observations)
            message = f'Saved {len(observations)} / {target}'
            print(message+' -> '+str(filename), flush=True)
    finally:
        reader.close()
        cv2.destroyAllWindows()
    return observations, len(observations) == target


def read_images(directory, detector):
    observations = []
    files = sorted(path for path in Path(directory).iterdir() if path.suffix.lower() in ('.png', '.jpg', '.jpeg', '.bmp'))
    for path in files:
        image = cv2.imread(str(path))
        if image is None:
            print('Skipped unreadable image: '+str(path))
            continue
        detection = detector.detect(image)
        if detection['valid']:
            observations.append(detector.observation(detection, path.resolve()))
        else:
            print('Skipped '+str(path)+': '+detection['reason'])
    return observations


def intrinsics_document(matrix, distortion, report, session):
    return dict(schema_version=1, source='ChArUco calibration',
                created_utc=datetime.now(timezone.utc).isoformat(), session=str(session),
                stream=dict(type='color', width=WIDTH, height=HEIGHT, format='BGR8',
                            image_processing='Native raw frames; no crop, resize, or pre-undistortion'),
                camera_model='pinhole', distortion_model='plumb_bob',
                distortion_order=['k1', 'k2', 'p1', 'p2', 'k3'],
                camera_matrix=matrix.tolist(), distortion_coefficients=distortion.reshape(-1).tolist(),
                rms_px=report['rms_px'], usable_views=report['usable_views'])


def print_transforms(nominal, calibrated=None):
    print('\n矩阵约定：p_torso = T_torso_camera_optical @ p_camera_optical；平移单位：米。')
    print('URDF 的相机光学坐标系 -> torso_link：\n'+np.array2string(nominal, precision=7, suppress_small=True))
    if calibrated is None:
        print('实测相机 -> torso 外参：尚未标定。需要独立测量的标定板 torso 位姿。')
    else:
        print('标定所得相机光学坐标系 -> torso_link：\n'+np.array2string(calibrated, precision=7, suppress_small=True))
        print('平移差（标定 - URDF，米）：', calibrated[:3, 3]-nominal[:3, 3])
        print('旋转差（度）：', rotation_angle(nominal[:3, :3].T @ calibrated[:3, :3]))


def generate_board(board, config, path):
    from PIL import Image
    c = config['board']
    dpi = 300
    width_m, height_m = c['squares_x']*c['square_length_m'], c['squares_y']*c['square_length_m']
    size = (round(width_m/.0254*dpi), round(height_m/.0254*dpi))
    raw = board.generateImage(size, marginSize=0, borderBits=1)
    # Paper margin is not part of the ChArUco frame or its square measurements.
    margin = round(.005/.0254*dpi)
    image = cv2.copyMakeBorder(raw, margin, margin, margin, margin, cv2.BORDER_CONSTANT, value=255)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path, dpi=(dpi, dpi))
    print(f'Board saved: {path}; chessboard extent {width_m*1000:g} x {height_m*1000:g} mm; '
          f'square {c["square_length_m"]*1000:g} mm. Print at 100% and VERIFY with a ruler.')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--reference', action='store_true', help='measure extrinsics with a known stationary board torso pose')
    parser.add_argument('--intrinsics', type=Path, help='intrinsics for --reference; default from config output.intrinsics_file')
    parser.add_argument('--images', type=Path, help='process saved RAW images instead of opening camera')
    parser.add_argument('--generate-board', type=Path, help='save configured board as a printable 300 DPI PNG and exit')
    args = parser.parse_args()
    cv2.setNumThreads(1)
    config = load_config(args.config)
    board = make_board(config)
    if args.generate_board:
        generate_board(board, config, args.generate_board)
        return
    nominal = urdf_camera_to_torso(config['robot_urdf'])
    intrinsics = None
    if args.reference:
        reference_transform(config)  # Fail before touching the camera if no independently known pose exists.
        intrinsics = load_intrinsics(args.intrinsics or config['output']['intrinsics_file'])
    detector = BoardDetector(board, config, intrinsics)
    mode = 'reference' if args.reference else 'intrinsics'
    session = Path(config['output']['session_dir'])/(datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'_'+mode)
    session.mkdir(parents=True)
    write_json(session/'config.json', config)
    pending = dict(status='reference_pose_required', T_torso_camera_optical=None,
                   T_torso_camera_optical_urdf=nominal.tolist(), board_frame=BOARD_FRAME)
    write_json(session/'extrinsics_status.json', pending)
    print('Session: '+str(session), flush=True)
    print('Board configuration: '+json.dumps(config['board'], ensure_ascii=False), flush=True)
    print_transforms(nominal)
    try:
        if args.images:
            observations = read_images(args.images, detector)
            complete = True
            write_json(session/'observations.json', observations)
        else:
            observations, complete = capture(detector, config, session, mode)
        if not complete:
            print('拍摄提前结束，照片已保存。可使用 --images '+str(session/'raw')+' 离线计算。')
            return
        print('拍摄完成，开始计算……', flush=True)
        if args.reference:
            report = calibrate_reference(observations, *intrinsics, config)
            report['session'] = str(session)
            write_json(session/'camera_to_torso.json', report)
            write_json(config['output']['extrinsics_file'], report)
            write_json(session/'extrinsics_status.json', report)
            print_transforms(nominal, np.array(report['T_torso_camera_optical']))
            print('可供其他程序加载的外参文件：'+config['output']['extrinsics_file'])
        else:
            matrix, distortion, report = calibrate_intrinsics(observations, config)
            write_json(session/'intrinsics_report.json', report)
            if report['status'] != 'calibrated':
                raise ValueError('标定质量检查未通过：'+', '.join(report['failures']))
            document = intrinsics_document(matrix, distortion, report, session)
            write_json(session/'camera_intrinsics.json', document)
            write_json(config['output']['intrinsics_file'], document)
            if config['calibration']['save_undistorted_images']:
                folder = session/'undistorted'
                folder.mkdir()
                for i, view in enumerate(report['views']):
                    image = cv2.imread(view['filename'])
                    if image is not None:
                        corrected = cv2.undistort(image, matrix, distortion, None, matrix)
                        if not cv2.imwrite(str(folder/f'{i+1:03d}.png'), corrected):
                            raise RuntimeError('Failed saving undistorted preview')
            print('内参 K：\n'+str(matrix)+'\n畸变参数 [k1,k2,p1,p2,k3]：\n'+str(distortion.reshape(-1)))
            print(f'RMS: {report["rms_px"]:.4f} px; usable views: {report["usable_views"]}')
            print('可供其他程序加载的内参文件：'+config['output']['intrinsics_file'])
            print('相机到 torso 的实测外参仍待已知参考位姿；本次未生成或覆盖有效外参文件。')
        print('完整报告和原始数据：'+str(session))
    except Exception as exc:
        write_json(session/'failure.json', dict(status='failed', reason=str(exc)))
        raise


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\n操作中断，已拍照片保留。', file=sys.stderr)
        raise SystemExit(130)
    except (ValueError, RuntimeError, OSError, KeyError, cv2.error) as exc:
        print('标定失败：'+str(exc), file=sys.stderr)
        raise SystemExit(1)
