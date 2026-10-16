#!/usr/bin/env python3
"""Outer G1 Viser console backed by G1Deploy_BalanceHomie."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import json
import signal
import time

import numpy as np

from core.robot import JOINT_NAMES, URDF_PATH, HAND_PD_PATH, UPPER_POSE_LABELS, RobotConfig
from core.controller import Controller, STAGES
from core.heading import CONFIG_PATH
from core.pushcart import CONFIG as PUSHCART_CONFIG, load_feature
from Vender.joint_panel import JointMonitorPanel, joint_label
from Vender.hand_panel import HandGraspPanel
from Vender.robot_display import UrdfDisplayUpdater, load_display_robot, feedback_configuration, current_grasp_reference
from Vender.pushcart_panel import CartDisplay, PushcartPanel
from core.joint_tracking import NAMES

RECOVERY_OPTIONS = {joint_label(name): i for i, name in enumerate(NAMES[-14:])}
RECOVERY_PLACEHOLDER = '选择超限手指'


class Viewer:
    def __init__(self, controller, host, port):
        import viser
        import yourdfpy
        from viser.extras import ViserUrdf
        self.control = controller
        self.clients = {}
        self.motion_panels = {}
        self.hand_panels = {}
        self.pushcart_panels = {}
        config = controller.config
        urdf = self.display_robot = load_display_robot(config.urdf)
        self.server = server = viser.ViserServer(host=host, port=port, label='G1 · Balance / Homie')
        if server.get_port() != port:
            server.stop()
            raise RuntimeError('端口 %s 被占用，请指定其他 --port' % port)
        server.gui.configure_theme(show_share_button=False, brand_color=(26, 137, 150))
        server.scene.set_up_direction('+z')
        server.scene.add_grid('/ground', width=20, height=20, plane='xy')
        self.base = server.scene.add_frame('/g1', show_axes=False, position=(0, 0, .793))
        self.model = ViserUrdf(server, urdf, root_node_name='/g1')
        self.model.update_cfg(feedback_configuration(urdf, config.default_q, grasp_reference=config.hand_grasp))
        self.robot_display = UrdfDisplayUpdater(self.model)
        self.cart_display = CartDisplay(server, urdf)
        labels = dict(sim='离线联调（合成反馈）', observe='真机只读同步', control='真机控制')
        server.gui.add_markdown('**G1 · %s**' % labels[controller.mode])
        self.status = server.gui.add_markdown('等待状态…')
        with server.gui.add_folder('视图设置', expand_by_default=False):
            self.base_height = server.gui.add_slider('显示基座高度 m', .3, 1.5, .01, .793,
                                                    disabled=False)
            self.follow = server.gui.add_checkbox('镜头跟随', initial_value=False)
        with server.gui.add_folder('运行记录与频率', expand_by_default=False):
            self.diagnostics = server.gui.add_markdown('等待统计…')
            server.gui.add_markdown('Dex3 Kp=%g · Kd=%g（`%s`，重启生效）' %
                                    (config.hand_pd['kp'], config.hand_pd['kd'], config.hand_pd_path.name))
            if controller.recorder:
                server.gui.add_markdown('存档：`%s`' % controller.recorder.path.name)
        with server.gui.add_folder('航向反馈', expand_by_default=False):
            self.heading_status = server.gui.add_markdown('等待航向数据…')
        self.joint_monitor = JointMonitorPanel(server, controller)

        @server.on_client_connect
        async def connect(client):
            center = np.asarray(self.base.position)
            client.camera.position = tuple(center + [2.3, -2.5, .9])
            client.camera.look_at = tuple(center + [0, 0, -.1])
            client.camera.up_direction = (0, 0, 1)
            buttons = {}
            with client.gui.add_folder('状态操作'):
                for kind, label in [('claim', '接管网页操作'), ('acquire', '① 接管底层'),
                                    ('prepare', '② 移到准备姿态'), ('policy', '③ 启动混合策略'),
                                    ('hand_toggle', 'Dex3 使能并抓握（3 秒）'),
                                    ('zero', '速度归零'), ('stop', '阻尼停止'),
                                    ('remote', '交给实体遥控器')]:
                    button = client.gui.add_button(label, disabled=True,
                                                   color='red' if kind == 'stop' else None)
                    async def clicked(event, kind=kind):
                        if kind in ('zero', 'stop', 'policy', 'claim') and controller.mode != 'observe':
                            self.clear_speed_sliders()
                        controller.emit(kind, event.client_id)
                    button.on_click(clicked)
                    buttons[kind] = button
                    if kind == 'acquire':
                        pose_selector = client.gui.add_dropdown(
                            '上半身准备姿态', tuple(UPPER_POSE_LABELS.values()),
                            initial_value=UPPER_POSE_LABELS[controller.prepare_pose], disabled=True)
                        async def change_prepare_pose(event):
                            if event.client_id is not None:
                                pose = next((key for key, label in UPPER_POSE_LABELS.items()
                                             if label == event.target.value), None)
                                if pose is not None:
                                    controller.emit('prepare_pose', event.client_id, pose=pose)
                        pose_selector.on_update(change_prepare_pose)
                        buttons['prepare_pose'] = pose_selector
                        compensation = client.gui.add_checkbox('Dex3 使用文件补偿角', initial_value=False,
                                                               disabled=True, visible=False)
                        async def change_compensation(event):
                            if event.client_id is not None:
                                controller.emit('hand_compensation', event.client_id, enabled=event.target.value)
                        compensation.on_update(change_compensation)
                        buttons['hand_compensation'] = compensation
                if controller.mode == 'sim':
                    button = client.gui.add_button('重置离线联调', disabled=True)
                    async def reset(event):
                        controller.emit('reset', event.client_id)
                    button.on_click(reset)
                    buttons['reset'] = button
            self.hand_panels[client.client_id] = HandGraspPanel(client, controller)
            self.pushcart_panels[client.client_id] = PushcartPanel(
                client, controller, self.cart_display, self.clear_speed_sliders)
            with client.gui.add_folder('Dex3 超限恢复', expand_by_default=False):
                recovery_joint = client.gui.add_dropdown('恢复关节', (RECOVERY_PLACEHOLDER,), disabled=True)
                recovery = client.gui.add_button('恢复所选手指到限位内', disabled=True)
                async def recover(event):
                    controller.emit('hand_recover', event.client_id, joint=RECOVERY_OPTIONS.get(recovery_joint.value))
                recovery.on_click(recover)
                buttons['hand_recovery_joint'] = recovery_joint
                buttons['hand_recover'] = recovery
                client.gui.add_markdown('仅在接管后、准备前恢复轻微越界的手指。完成后再点击准备。')
            with client.gui.add_folder('速度指令'):
                running_push = (controller.snapshot().get('high_policy') or {}).get('enabled', False)
                velocity_bounds = controller.high_policy.limits if running_push else config.velocity_bounds
                velocity_labels = (['小车前后 vx (m/s)', '小车左右 vy (m/s)', '小车转向 wz (rad/s)'] if running_push else
                                   ['前后 vx (m/s)', '左右 vy (m/s)', '转向 yaw (rad/s)'])
                sliders = [client.gui.add_slider(label, float(lower), float(upper), .01, 0.,
                                                 disabled=controller.mode == 'observe')
                           for label, (lower, upper) in zip(
                               velocity_labels, velocity_bounds)]
                button = client.gui.add_button('按住发送速度', disabled=True, color='teal')
                async def held(event):
                    controller.emit('move', event.client_id, [s.value for s in sliders])
                button.on_hold(held, callback_hz=20.)
                buttons['move'] = button
                client.gui.add_markdown('+vx 前 · +vy 左 · +yaw 左转；松开后平滑减速，由 Balance 接管停稳。')
            with client.gui.add_folder('航向参数（先归零再应用）', expand_by_default=False):
                settings = controller.heading.settings
                client.gui.add_markdown('disabled 关闭 · shadow 仅记录 · active 应用纠偏')
                client.gui.add_markdown('行走与推车共用这些参数；推车时跟随策略要求的转向修正骨盆航向。')
                gain = client.gui.add_slider('航向 Kp (1/s)', 0., 2., .05, settings['heading_kp'])
                limit = client.gui.add_slider('纠偏上限 (rad/s)', 0., .3, .01, settings['max_correction_rad_s'])
                smoothing = client.gui.add_slider('角速度滤波截止频率 (Hz)', .1, 10., .1,
                                                  settings['gyro_cutoff_hz'])
                mode = client.gui.add_dropdown('骨盆纠偏模式', ('disabled','shadow','active'),
                                               initial_value=settings['run_mode'])
                apply_button = client.gui.add_button('应用航向参数', disabled=True)
                rebase_button = client.gui.add_button('以当前朝向重设航向目标', disabled=True)

                async def apply_motion(event):
                    controller.emit('motion_settings', event.client_id, settings=dict(
                        command_frame='body', heading_kp=gain.value,
                        max_correction_rad_s=limit.value, gyro_cutoff_hz=smoothing.value, run_mode=mode.value))

                async def rebase_motion(event):
                    controller.emit('rebase_heading', event.client_id)

                apply_button.on_click(apply_motion)
                rebase_button.on_click(rebase_motion)
                self.motion_panels[client.client_id] = (gain, limit, smoothing, mode, apply_button, rebase_button)
            self.clients[client.client_id] = (client, buttons, sliders)

        @server.on_client_disconnect
        async def disconnect(client):
            controller.emit('disconnect', client.client_id)
            self.clients.pop(client.client_id, None)
            self.motion_panels.pop(client.client_id, None)
            self.hand_panels.pop(client.client_id, None)
            self.pushcart_panels.pop(client.client_id, None)
            self.cart_display.previews.discard(client.client_id)

        self.last_text = 0.
        self.last_tick = None
        self.last_stage = None
        self.last_push_enabled = None
        self.last_position = np.array([0., 0., .793])

    def clear_speed_sliders(self):
        for _, _, sliders in list(self.clients.values()):
            for slider in sliders:
                slider.value = 0.

    def update(self):
        data = self.control.snapshot()
        if not data:
            return
        state = data.get('state')
        stage = data['stage']
        if stage != self.last_stage:
            if stage in ('WAITING', 'ZERO', 'POLICY', 'DAMPED', 'FAULT'):
                self.clear_speed_sliders()
            self.last_stage = stage
        healthy = data['healthy'] and state is not None and 0 <= time.monotonic()-state['received'] <= self.control.cfg['state_timeout_s']
        now = time.monotonic()
        high = data.get('high_policy') or {}
        push_enabled = high.get('enabled', False)
        if push_enabled != self.last_push_enabled:
            bounds = self.control.high_policy.limits if push_enabled else self.control.config.velocity_bounds
            labels = (['小车前后 vx (m/s)', '小车左右 vy (m/s)', '小车转向 wz (rad/s)'] if push_enabled else
                      ['前后 vx (m/s)', '左右 vy (m/s)', '转向 yaw (rad/s)'])
            self.clear_speed_sliders()
            for _, _, sliders in list(self.clients.values()):
                for slider, label, (lower, upper) in zip(sliders, labels, bounds):
                    slider.min, slider.max, slider.label = float(lower), float(upper), label
            self.last_push_enabled = push_enabled
        if state is not None and state['tick'] != self.last_tick:
            if np.isfinite(state['q']).all():
                with self.server.atomic():
                    hands = data.get('dex3') or {}
                    self.robot_display.update_cfg(feedback_configuration(
                        self.display_robot, state['q'], hands, current_grasp_reference(self.control.config, hands)))
                    norm = np.linalg.norm(state['quat'])
                    if .8 < norm < 1.2:
                        self.base.wxyz = tuple(state['quat']/norm)
                    position = np.array([0, 0, self.base_height.value])
                    self.base.position = tuple(position)
                    if self.follow.value:
                        delta = position - self.last_position
                        for client, _, _ in list(self.clients.values()):
                            client.camera.position = tuple(np.asarray(client.camera.position) + delta)
                            client.camera.look_at = tuple(np.asarray(client.camera.look_at) + delta)
                    self.last_position = position.copy()
            self.last_tick = state['tick']
        self.cart_display.update(data, self.base.position, self.base.wxyz)
        for client_id, (_, buttons, sliders) in list(self.clients.items()):
            allowed = self.control.mode != 'observe'
            owned = allowed and data['owner'] == client_id and data['source'] == 'web'
            buttons['claim'].disabled = not (allowed and data['owner'] in (None, client_id))
            buttons['acquire'].disabled = not (owned and healthy and stage == 'WAITING')
            hands = data.get('dex3', {})
            recovering = hands.get('recovering') or hands.get('recovery_pending')
            buttons['prepare'].disabled = not (owned and healthy and stage == 'ZERO')
            pose_selector = buttons['prepare_pose']
            pose_selector.disabled = not (owned and healthy and stage in ('WAITING', 'ZERO')
                                          and data.get('prepare_pose_selectable', False))
            pose_selector.value = UPPER_POSE_LABELS[data.get('prepare_pose', 'original')]
            buttons['policy'].disabled = not (owned and healthy and stage == 'HOLDING' and data.get('observation_frames', 0) >= 6)
            buttons['hand_toggle'].label = ('Dex3 使能并抓握（3 秒）' if not hands.get('ever_started') else
                                            'Dex3 重新启用并归零（3 秒）' if not hands.get('active') else
                                            'Dex3 双手：松开（3 秒）' if hands.get('pose') in ('grasp', 'grasp_compensated')
                                            else 'Dex3 双手：抓握（3 秒）')
            buttons['hand_toggle'].disabled = not (owned and healthy and stage in ('HOLDING', 'POLICY')
                                                   and not push_enabled
                                                   and hands.get('healthy') and not recovering
                                                   and not hands.get('moving') and not hands.get('pending')
                                                   and not hands.get('file_busy'))
            compensation = buttons['hand_compensation']
            compensation.visible = stage in ('ZERO', 'PREPARING', 'HOLDING', 'POLICY')
            compensation.disabled = not (owned and healthy and stage in ('ZERO', 'HOLDING', 'POLICY')
                                        and not push_enabled
                                        and not hands.get('moving') and not hands.get('pending') and not recovering
                                        and not hands.get('file_busy'))
            compensation.value = hands.get('compensation_enabled', False)
            if client_id in self.hand_panels:
                self.hand_panels[client_id].update(hands, owned, healthy)
            recovery_joint = buttons['hand_recovery_joint']
            options = (RECOVERY_PLACEHOLDER,) + tuple(name for name, index in RECOVERY_OPTIONS.items()
                                                     if hands.get('recovery_mask', 0) & (1 << index))
            if tuple(recovery_joint.options) != options:
                recovery_joint.options = options
            can_recover = owned and healthy and stage == 'ZERO' and not recovering
            recovery_joint.disabled = not can_recover
            buttons['hand_recover'].disabled = not (can_recover and recovery_joint.value in options[1:])
            buttons['move'].disabled = not (owned and healthy and stage == 'POLICY' and not data.get('decelerating')
                                            and not high.get('motion_blocked') and not data.get('upper_pd_transitioning'))
            buttons['remote'].disabled = not (owned and self.control.mode == 'control')
            buttons['zero'].disabled = not allowed
            buttons['stop'].disabled = not allowed
            if client_id in self.pushcart_panels:
                self.pushcart_panels[client_id].update(data, owned, healthy)
            if 'reset' in buttons:
                buttons['reset'].disabled = not (owned and stage in ('WAITING', 'DAMPED', 'FAULT'))
            if client_id in self.motion_panels:
                gain, limit, smoothing, mode, apply_button, rebase_button = self.motion_panels[client_id]
                idle = not np.any(data['velocity']) and not np.any(data.get('input_velocity', 0))
                editable = owned and healthy and idle and stage not in ('ACQUIRING', 'DAMPED', 'FAULT')
                for handle in (gain, limit, smoothing, mode, apply_button, rebase_button):
                    handle.disabled = not editable
        self.joint_monitor.update(data, now)
        if now-self.last_text >= .2:
            self.last_text = now
            velocity = data['velocity']
            age = '—' if state is None else '%.0f ms' % ((now-state['received'])*1000)
            status = (
                '**%s** · 反馈%s\n\n'
                '模型 **%s** · %s\n\n'
                '运动命令 vx / vy / wz：**%.2f / %.2f / %.2f**'
                % (STAGES[stage], '正常' if healthy else '异常',
                   data.get('selected_model', '—'),
                   '只读同步' if self.control.mode == 'observe' else
                   ('实体遥控' if data['source'] != 'web' else ('网页已接管' if data['owner'] is not None else '等待网页接管')),
                   *velocity))
            if data.get('selected_model') == 'balance':
                status += '\n\nBalance：零速站立'
            warmup = data.get('model_warmup') or {}
            if warmup.get('balance_homie'):
                status += '\n\n模型已预热 · 底层历史 %d / 6 帧' % min(data.get('observation_frames', 0), 6)
            if data.get('decelerating'):
                status += '\n\n**正在平滑减速至零**'
            if data.get('preparation_warning'):
                status += '\n\n**准备超时，继续保持姿态；未到位关节见左侧告警。**'
            status += '\n\n上半身：' + UPPER_POSE_LABELS[data.get('prepare_pose', 'original')]
            hands = data.get('dex3', {})
            hand_text = {'grasp': '抓握', 'open_compensated': '补偿归零（松开）',
                         'grasp_compensated': '补偿抓握'}.get(hands.get('pose'), '松开')
            if stage in ('DAMPED', 'FAULT'):
                hand_text = '已停止'
            elif hands.get('recovering') or hands.get('recovery_pending'):
                hand_text = '正在恢复超限手指'
            elif not hands.get('healthy'):
                hand_text = '未就绪，详见左侧告警'
            elif not hands.get('active'):
                hand_text = ('等待使能确认' if hands.get('pending') else
                             '已暂停，可重新启用' if hands.get('ever_started') else '未使能，点击手部按钮启动')
            elif hands.get('moving'):
                hand_text = ('正在跟随抓握调节' if hands.get('editing') else
                             '正在%s · %.0f%%' % (hand_text, 100*hands.get('progress', 0)))
            status += '\n\nDex3 **%s**' % hand_text
            status += '\n\n' + ('**推车策略已接入**' if push_enabled else
                                  '**推车暂停，先处理相机/抓握或返回普通行走**' if high.get('motion_blocked') else
                                  '**普通混合策略 · 推车未启用**')
            self.status.content = status
            timing = data.get('timing', {})
            heading = data.get('heading', {})
            if heading:
                def angle(value):
                    return '—' if value is None else '%+.1f°' % np.rad2deg(value)
                self.heading_status.content = (
                    '**%s** · %s · IMU %s\n\n'
                    'Kp %.2f · 上限 %.2f rad/s · 低通 %.1f Hz\n\n'
                    '目标 %s · 当前 %s · 误差 %s\n\n'
                    '修正 %+.3f · 建议 wz %+.3f rad/s'
                    % (heading['run_mode'], heading['heading_state'], heading['imu_health'],
                       heading['heading_kp'], heading['max_correction_rad_s'], heading['gyro_cutoff_hz'],
                       angle(heading['target_relative_rad']), angle(heading['yaw_relative_rad']),
                       angle(heading['error_rad']), heading['correction_rad_s'], heading['wz_suggested']))
            def milliseconds(value):
                return '—' if value is None else '%.2f ms' % value
            self.diagnostics.content = (
                '控制循环 **%.1f Hz** · 策略 **%.1f Hz**\n\n'
                '推理 %s · 观测至结果 %s\n\n'
                '反馈接收至指令发出 %s\n\n'
                '反馈 %s · 浏览器 %s\n\n'
                '选择 %s · PD %s %.0f%%\n\n%s'
                % (timing.get('control_hz', 0), timing.get('policy_hz', 0),
                   milliseconds(timing.get('inference_ms')),
                   milliseconds(timing.get('observation_to_action_ms')),
                   milliseconds(timing.get('state_to_command_ms')),
                   age, data['owner'], data.get('policy_mode', 'auto'),
                   data.get('pd_profile', '—'), 100 * data.get('pd_transition_blend', 1),
                   ('记录异常：%s；丢失 %d 条' % (data['recording_error'], data['recording_dropped']))
                   if data.get('recording_error') or data.get('recording_dropped') else
                   ('正在自动存档' if self.control.recorder else '未存档')))

    def close(self):
        self.server.stop()


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('sim', 'observe', 'control'), default='sim')
    parser.add_argument('--interface', default='eth0')
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8086)
    parser.add_argument('--urdf', default=str(URDF_PATH))
    parser.add_argument('--duration', type=float, default=0, help='Exit after N seconds; 0 means unlimited')
    parser.add_argument('--log', help='JSONL telemetry output')
    parser.add_argument('--record-dir', help='Parent directory for automatic session archives')
    parser.add_argument('--record-sim', action='store_true', help='Also archive synthetic offline sessions')
    parser.add_argument('--policy-mode', choices=('auto', 'balance', 'homie'), default='auto')
    parser.add_argument('--pushcart-config', default=str(PUSHCART_CONFIG),
                        help='Push-cart feature configuration; always starts disabled')
    parser.add_argument('--motion-config', default=str(CONFIG_PATH), help='Body-frame heading configuration JSON')
    parser.add_argument('--dex3-pd-config', default=str(HAND_PD_PATH), help='Dex3 Kp/Kd JSON, loaded once at startup')
    parser.add_argument('--command-frame', choices=('body',), help='Body axes for vx/vy; IMU corrects only wz')
    parser.add_argument('--heading-kp', type=float, help='Override heading Kp (0..2 per second)')
    parser.add_argument('--heading-mode', choices=('disabled','shadow','active'),
                        help='Pelvis heading feedback; default from config (active)')
    args = parser.parse_args()
    if not 1 <= args.port <= 65535 or not np.isfinite(args.duration) or args.duration < 0:
        parser.error('Invalid port or duration')
    if args.record_sim and not args.record_dir:
        parser.error('--record-sim requires --record-dir outside the robot logs directory')
    if args.mode == 'sim' and args.record_dir:
        from core.robot import ROOT
        if ROOT / 'logs' in (Path(args.record_dir).resolve(), *Path(args.record_dir).resolve().parents):
            parser.error('Synthetic recordings must be outside Viser/logs')
    config = RobotConfig(args.urdf, args.motion_config, args.command_frame, args.heading_kp, args.heading_mode,
                         hand_pd_config=args.dex3_pd_config)
    backend = controller = viewer = log = recorder = bridge_log = high_policy = None
    exit_error = None
    stop = False
    def request_stop(*_):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    try:
        high_policy = load_feature(config, args.mode, args.interface, args.pushcart_config)
        if config.pushcart_error:
            print('推车功能未加载，普通混合策略继续可用：' + config.pushcart_error, flush=True)
        if args.mode in ('observe', 'control') or args.record_sim:
            from core.recording import SessionRecorder
            recorder = SessionRecorder(config, vars(args), args.record_dir)
            print('RECORDING ' + str(recorder.path), flush=True)
        if args.log:
            path = Path(args.log)
            path.parent.mkdir(parents=True, exist_ok=True)
            # An explicitly requested export must not truncate earlier robot data either.
            log = path.open('x', buffering=1)
        from core.backends import Bridge, build_bridge
        build_bridge()
        if recorder:
            bridge_log = (recorder.path / 'bridge.log').open('x', buffering=1)
        backend = Bridge(config, args.mode, args.interface, output=bridge_log)
        controller = Controller(config, backend, args.mode, recorder=recorder, policy_mode=args.policy_mode,
                                high_policy=high_policy)
        viewer = Viewer(controller, args.host, args.port)
        controller.start()
        print('READY mode=%s | http://127.0.0.1:%d | 局域网用本机 IP:%d' %
              (args.mode, args.port, args.port), flush=True)
        start, last_log = time.monotonic(), 0.
        while not stop and (args.duration == 0 or time.monotonic()-start < args.duration):
            viewer.update()
            if log and time.monotonic()-last_log >= .1:
                last_log = time.monotonic()
                log.write(json.dumps(controller.snapshot(), default=json_value, ensure_ascii=False) + '\n')
            time.sleep(1/30)
    except BaseException as exc:
        exit_error = repr(exc)
        raise
    finally:
        if controller:
            controller.close()
        if viewer:
            viewer.close()
        if backend and not controller:
            backend.close()
        if high_policy and not controller:
            high_policy.close()
        if bridge_log:
            bridge_log.close()
        if log:
            log.close()
        if recorder:
            recorder.close(error=exit_error or (controller.reason if controller and controller.stage == 'FAULT' else None))


if __name__ == '__main__':
    main()
