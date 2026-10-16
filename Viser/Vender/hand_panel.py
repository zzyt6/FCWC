"""Per-client Dex3 grasp editor; the controller owns all angles and file writes."""
import numpy as np

from core.hand_grasp import HAND_JOINTS, HAND_LOWER, HAND_UPPER

FINGER_LABELS = {'thumb': '拇指', 'middle': '中指', 'index': '食指'}


class HandGraspPanel:
    def __init__(self, client, controller):
        self.controller = controller
        self.revision = None
        self.compensated = None
        self.sliders = []
        self.folder = client.gui.add_folder('Dex3 抓握姿态调节', visible=False)
        with self.folder:
            client.gui.add_markdown('调节即执行抓握，单关节最高约 **28.6°/s**。滑条为原始抓握角（°）；'
                                    '勾选文件补偿时，下发角度会叠加补偿。保存文件使用弧度。'
                                    '网页手指参考模型同步当前抓握设定；实测角度见左侧关节表。')
            for side, name in enumerate(('左手', '右手')):
                with client.gui.add_folder(name + '抓握关节'):
                    for j, joint in enumerate(HAND_JOINTS):
                        finger, number = joint.split('_')
                        index = side * 7 + j
                        minimum = float(np.ceil(np.rad2deg(HAND_LOWER[side, j]) * 100) / 100)
                        maximum = float(np.floor(np.rad2deg(HAND_UPPER[side, j]) * 100) / 100)
                        slider = client.gui.add_slider(
                            f'{name}{FINGER_LABELS[finger]} {number} (°)',
                            min=minimum, max=maximum, step=.01,
                            initial_value=float(np.clip(np.round(np.rad2deg(controller.config.hand_grasp[side, j]), 2),
                                                        minimum, maximum)),
                            disabled=True)

                        async def changed(event, index=index):
                            if event.client_id is not None:
                                controller.emit('hand_edit', event.client_id, joint=index,
                                                angle_rad=float(np.deg2rad(event.target.value)))

                        slider.on_update(changed)
                        self.sliders.append(slider)
            self.save = client.gui.add_button('保存抓握姿态', disabled=True)
            self.load = client.gui.add_button('加载并应用抓握姿态', disabled=True)
            self.status = client.gui.add_markdown('')
            client.gui.add_markdown('文件：`' + str(controller.config.upper_pose_path) + '`')

            async def save(event):
                controller.emit('hand_save', event.client_id)

            async def load(event):
                controller.emit('hand_load', event.client_id)

            self.save.on_click(save)
            self.load.on_click(load)

    def update(self, hands, owned, healthy):
        self.folder.visible = bool(hands.get('ever_started'))
        editable = bool(owned and healthy and hands.get('editor_available'))
        for handle in (*self.sliders, self.save, self.load):
            handle.disabled = not editable
        revision = hands.get('grasp_revision')
        compensated = hands.get('compensation_enabled', False)
        if revision != self.revision or compensated != self.compensated:
            lower, upper = HAND_LOWER.copy(), HAND_UPPER.copy()
            zero = self.controller.config.hand_zero
            if compensated and zero is not None:
                lower, upper = np.maximum(lower, lower-zero), np.minimum(upper, upper-zero)
            values = np.rad2deg(hands.get('grasp', self.controller.config.hand_grasp)).ravel()
            for i, slider in enumerate(self.sliders):
                slider.min = float(np.ceil(np.rad2deg(lower.flat[i]) * 100) / 100)
                slider.max = float(np.floor(np.rad2deg(upper.flat[i]) * 100) / 100)
                slider.value = float(np.clip(np.round(values[i], 2), slider.min, slider.max))
            self.revision, self.compensated = revision, compensated
        text = '**有未保存的修改**' if hands.get('grasp_dirty') else '抓握角度与已加载/保存文件一致'
        if hands.get('editor_message'):
            text += '\n\n' + hands['editor_message']
        if not editable and not hands.get('file_busy'):
            if self.controller.high_policy and self.controller.high_policy.enabled:
                text += '\n\n推车期间锁定抓握调节；先归零并返回普通行走，再继续调试。'
            else:
                text += '\n\n需持有网页控制权，且 Dex3 已使能、反馈正常、三秒姿态切换已完成后调节。'
        self.status.content = text
