"""Optional push-cart workflow and measured-heading/reference-position scene."""
import json
from io import BytesIO
from pathlib import Path
import xml.etree.ElementTree as ET
import time

import numpy as np
from scipy.spatial.transform import Rotation

from core.pelvis_imu import yaw_wxyz

DISPLAY_CONFIG = Path(__file__).resolve().parents[1] / 'config/cart_display.json'


def camera_fresh(camera, now=None):
    """Recheck freshness even when the control thread has stopped updating snapshots."""
    if not camera.get('fresh'):
        return False
    if 'captured_monotonic' not in camera:
        return True  # Older snapshots and offline geometry fixtures.
    stamp = camera['captured_monotonic']
    now = time.monotonic() if now is None else now
    return stamp is not None and 0 <= now - stamp <= camera.get('timeout_s', .3)


def cart_pose(settings, robot, base_position, base_quaternion, camera):
    """Use calibrated tag offsets when supplied; otherwise the training grasp reference."""
    base_q = np.asarray(base_quaternion)
    base_rotation = Rotation.from_quat(base_q[[1, 2, 3, 0]]).as_matrix()
    heading = yaw_wxyz(base_q) + (camera.get('yaw_rad') or 0.)
    rotation = Rotation.from_euler('z', heading).as_matrix()
    measured = []
    offsets = settings.get('tag_positions_in_cart_m', {})
    if camera.get('fresh'):
        for tag in camera.get('tags_in_pelvis', []):
            offset = offsets.get(str(tag['id']))
            if offset is not None:
                measured.append(np.asarray(base_position) + base_rotation @ tag['position_m'] - rotation @ offset)
    if measured:
        return np.mean(measured, axis=0), rotation, '标签标定位置'
    points = []
    for side in ('left', 'right'):
        transform = robot.get_transform(side + '_hand_palm_link')
        point = transform[:3, 3] + transform[:3, :3] @ settings['grasp_point_in_palm_m'][side]
        points.append(np.asarray(base_position) + base_rotation @ point)
    position = np.mean(points, axis=0) - rotation @ settings['handle_center_in_cart_m']
    return position, rotation, '抓握参考位置（未标定）'


class CartDisplay:
    def __init__(self, server, robot, config_path=DISPLAY_CONFIG):
        self.server, self.robot = server, robot
        self.previews = set()
        self.error = ''
        self.position_source = '抓握参考位置（未标定）'
        self.have_observation = False
        self.frame = self.model = self.label = None
        try:
            import yourdfpy
            from viser.extras import ViserUrdf
            path = Path(config_path)
            self.settings = json.loads(path.read_text())
            for name in ('handle_center_in_cart_m',):
                self._vector(self.settings[name])
            for value in self.settings['grasp_point_in_palm_m'].values():
                self._vector(value)
            for value in self.settings['tag_positions_in_cart_m'].values():
                self._vector(value)
            asset = (path.parent / self.settings['urdf']).resolve()
            tree = ET.parse(asset)
            # Upstream wheel visuals share names with their links. Keep Viser's
            # geometry nodes distinct from joint frames, without editing assets.
            for i, visual in enumerate(tree.findall('.//visual')):
                visual.set('name', 'cart_visual_' + str(i))
            cart = yourdfpy.URDF.load(BytesIO(ET.tostring(tree.getroot())), mesh_dir=str(asset.parent),
                                     load_collision_meshes=False)
            self.frame = server.scene.add_frame('/pushcart', show_axes=False, visible=False)
            self.model = ViserUrdf(server, cart, root_node_name='/pushcart')
            self.model.update_cfg(np.zeros(cart.num_actuated_joints))
            self.label = server.scene.add_label('/pushcart/status', '', position=(0., 0., 1.1))
        except Exception as exc:
            self.error = str(exc)
            if self.frame:
                self.frame.visible = False

    @staticmethod
    def _vector(value):
        array = np.asarray(value, dtype=float)
        if array.shape != (3,) or not np.isfinite(array).all():
            raise ValueError('小车显示配置坐标必须是三个有限数值')

    def update(self, data, base_position, base_quaternion):
        if self.error:
            return
        try:
            self._update(data, base_position, base_quaternion)
        except Exception as exc:
            # A display failure must not unwind the live body controller.
            self.error = '小车显示不可用：' + str(exc)
            self.frame.visible = False

    def _update(self, data, base_position, base_quaternion):
        high = data.get('high_policy') or {}
        visible = bool(self.previews or high.get('enabled') or high.get('motion_blocked'))
        self.frame.visible = visible
        if not visible:
            return
        camera = high.get('camera') or {}
        fresh = camera_fresh(camera)
        body_fresh = bool(data.get('healthy'))
        fresh = fresh and body_fresh
        if fresh or not self.have_observation:
            position, rotation, self.position_source = cart_pose(
                self.settings, self.robot, base_position, base_quaternion, camera)
            quat = Rotation.from_matrix(rotation).as_quat()
            with self.server.atomic():
                self.frame.position = tuple(position)
                self.frame.wxyz = tuple(quat[[3, 0, 1, 2]])
            self.have_observation = bool(fresh)
        heading = '航向：相机实测' if fresh else '航向过期，保留最后观测' if self.have_observation else '等待相机，参考示意'
        self.label.text = self.position_source + ' · ' + heading


class PushcartPanel:
    def __init__(self, client, control, display, clear_sliders):
        self.control, self.display = control, display
        with client.gui.add_folder('④ 可选：推车策略'):
            client.gui.add_markdown('先用混合策略站稳 → 把车移到手边 → 抓握并调好手指 → 再启用推车。'
                                    '不启用时，保持原来的普通行走。')
            self.camera = client.gui.add_button('连接 / 重试小车相机', disabled=True)
            client.gui.add_markdown('航向标零：速度归零，将小车前进方向与骨盆正前方对齐并保持静止，'
                                    '然后点击下方按钮。标定只对本次运行生效。')
            self.zero = client.gui.add_button('标定航向零位（车身已对正）', disabled=True)
            self.zero_status = client.gui.add_markdown('等待航向状态…')
            self.preview = client.gui.add_checkbox('预览参考小车', initial_value=False)
            self.status = client.gui.add_markdown('等待状态…')
            self.enable = client.gui.add_button('启用推车策略', disabled=True, color='teal')
            self.disable = client.gui.add_button('返回普通行走 / 继续调抓握', disabled=True)

        async def connect(event):
            control.emit('cart_camera_start', event.client_id)

        async def zero(event):
            control.emit('cart_yaw_zero', event.client_id)

        async def enable(event):
            clear_sliders()
            control.emit('pushcart_enable', event.client_id)

        async def disable(event):
            clear_sliders()
            control.emit('pushcart_disable', event.client_id)

        async def preview(event):
            if event.target.value:
                display.previews.add(client.client_id)
            else:
                display.previews.discard(client.client_id)

        self.camera.on_click(connect)
        self.zero.on_click(zero)
        self.enable.on_click(enable)
        self.disable.on_click(disable)
        self.preview.on_update(preview)

    def update(self, data, owned, healthy):
        high = data.get('high_policy') or {}
        enabled, blocked = high.get('enabled', False), high.get('motion_blocked', False)
        camera, process = high.get('camera') or {}, high.get('camera_process') or {}
        fresh = camera_fresh(camera)
        stationary = not np.any(data.get('velocity', 0)) and not data.get('decelerating')
        reason = high.get('enable_reason', '推车模块不可用')
        calibration = high.get('zero_calibration') or {}
        collecting = calibration.get('status') == 'collecting'
        zero_reason = high.get('zero_reason', '推车模块不可用')
        self.zero.disabled = not (owned and healthy and fresh and not collecting and not zero_reason)
        zero_text = calibration.get('message', '等待航向状态…')
        if collecting:
            zero_text += '（%d / %d 帧）' % (calibration.get('samples', 0), calibration.get('required', 30))
        elif zero_reason:
            zero_text += '。' + zero_reason
        if camera.get('yaw_offset_rad') is not None:
            zero_text += '\n\n当前零位偏置 %+.2f°' % np.rad2deg(camera['yaw_offset_rad'])
        if calibration.get('status') == 'done':
            zero_text += ' · 采集最大波动 %.2f°' % np.rad2deg(calibration['max_deviation_rad'])
        self.zero_status.content = zero_text
        self.enable.disabled = not (owned and healthy and not enabled and not reason and fresh)
        self.disable.disabled = not (owned and healthy and stationary and (enabled or blocked))
        self.camera.disabled = not (owned and self.control.high_policy and self.control.mode != 'sim'
                                    and not process.get('running') and not process.get('starting') and not fresh)
        mode = '推车策略运行' if enabled else '推车已暂停，等待处理' if blocked else '普通混合策略'
        yaw = camera.get('yaw_rad')
        angle = '—' if yaw is None else '%+.1f°' % np.rad2deg(yaw)
        text = '**%s** · 小车航向 %s（%s）' % (mode, angle, '实时' if fresh else '无有效新数据')
        ids = camera.get('tag_ids', [])
        if fresh and ids:
            text += ' · 标签 ' + ', '.join(map(str, ids))
        text += '\n\n' + (high.get('reason', '') if blocked else
                           '先归零，再返回普通行走即可继续调抓握。' if enabled else reason or '抓握和相机已就绪，可选择启用推车。')
        text += '\n\n小车：' + (self.display.error or self.display.position_source)
        text += '\n\n模型%s · 推车历史 %d / %d 帧' % (
            '已预热' if high.get('model_warmed') else '未就绪',
            high.get('history_frames', 0), high.get('history_required', 32))
        if high.get('homie_model_warmed'):
            text += '\n\n推车微调速度估计器：' + ('使用中' if data.get('homie_estimator_source') == 'pushcart_finetuned'
                                               else '已预热，启用推车时使用')
        if process.get('starting'):
            text += ' · 正在连接相机'
        upper_mode = '推车增益' if data.get('upper_pd_mode') == 'pushcart' else '原 Viser 增益'
        if data.get('stage') in ('DAMPED', 'FAULT'):
            text += '\n\n上肢：已停止，执行底层阻尼退出。'
        elif data.get('upper_pd_transitioning'):
            text += '\n\n上肢增益正在切换，完成后可发送速度。'
        else:
            text += '\n\n上肢：' + upper_mode
        error = getattr(self.control.config, 'pushcart_error', '') or high.get('warmup_error') or process.get('error')
        if error:
            text += '\n\n' + error[-600:]
        self.status.content = text
