"""Grasp editing through the Python/C++ boundary; files stay in /tmp."""
from pathlib import Path
from concurrent.futures import Future
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import Mock, patch
from types import SimpleNamespace

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.controller import Controller
from core.hand_grasp import load_grasp, save_grasp
from core.robot import RobotConfig, UPPER_POSE_PATH
from hybrid_deploy import protocol as p
from test_control import FakeBridge
from test_pose_integration import HARNESS, PipeBridge
from Vender.app import Viewer
from Vender.hand_panel import HandGraspPanel
from test_web import Browser, until


class GraspFileTests(unittest.TestCase):
    def test_save_preserves_body_settings_comments_and_loads_named_joints(self):
        original = UPPER_POSE_PATH.read_text()
        with tempfile.TemporaryDirectory(prefix='viser_grasp_') as directory:
            path = Path(directory) / 'pose.yaml'
            path.write_text(original)
            angles = load_grasp(path)
            angles[0, 0] = .12345
            angles[1, 6] = .98765
            save_grasp(path, angles)
            np.testing.assert_array_equal(load_grasp(path), angles)
            with patch('core.robot.UPPER_POSE_PATH', path), patch.dict('core.robot.UPPER_POSE_PATHS', original=path):
                np.testing.assert_array_equal(RobotConfig().hand_grasp, angles)
            before, after = yaml.safe_load(original), yaml.safe_load(path.read_text())
            before.pop('hand_grasp'); after.pop('hand_grasp')
            self.assertEqual(before, after)
            self.assertEqual(original.split('hand_grasp:\n')[0], path.read_text().split('hand_grasp:\n')[0])
            self.assertEqual(list(Path(directory).iterdir()), [path])
            saved = path.read_bytes()
            with patch('core.hand_grasp.os.replace', side_effect=OSError('disk error')):
                with self.assertRaises(OSError): save_grasp(path, angles * .9)
            self.assertEqual(saved, path.read_bytes())
            self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_bad_files_and_nonfinite_or_out_of_range_angles_are_rejected(self):
        original = UPPER_POSE_PATH.read_text()
        with tempfile.TemporaryDirectory(prefix='viser_grasp_invalid_') as directory:
            path = Path(directory) / 'pose.yaml'
            for bad in ('true', '.nan', '.inf', '9', 'null', '[0]', '"0.1"'):
                path.write_text(original.replace('left_hand_thumb_0_joint: 0.0', 'left_hand_thumb_0_joint: ' + bad))
                with self.subTest(bad=bad), self.assertRaises(ValueError): load_grasp(path)
            path.write_text(original.replace('left_hand_thumb_0_joint: 0.0',
                                             'left_hand_thumb_0_joint: 0.0\n    left_hand_thumb_0_joint: 0.1'))
            with self.assertRaises(ValueError): load_grasp(path)
            path.write_text(original)
            for bad in (float('nan'), 2.):
                angles = load_grasp(path); angles[0, 0] = bad
                with self.assertRaises(ValueError): save_grasp(path, angles)
                self.assertEqual(path.read_text(), original)


class GraspControllerTests(unittest.TestCase):
    def test_completed_load_cannot_apply_after_hand_fault_or_owner_loss(self):
        for fault in ('hand', 'owner', 'stop'):
            config = RobotConfig()
            bridge = FakeBridge(config)
            bridge.armed = True; bridge.state.stage = p.HOLDING; bridge.state.hand_flags = 55
            control = Controller(config, bridge, 'sim')
            self.addCleanup(control.hand_file_executor.shutdown)
            control.stage = 'HOLDING'; control.hand_pose = 2
            control.web.owner = 1; control.web.source = 'web'
            angles = config.hand_grasp.copy(); angles[0, 0] = .2
            context = (bridge.state.session, 1, 2, False, 0)
            future = Future(); future.set_result(angles)
            control.hand_file = (future, 'hand_load', context, None)
            if fault == 'hand': bridge.state.hand_flags = 35
            elif fault == 'owner': control.web.owner = None
            else: control.stop_requested.set()
            control._poll_hand_file(bridge.state)
            self.assertEqual(config.hand_grasp[0, 0], 0.)
            self.assertFalse(control.hand_live)
            self.assertIn('取消', control.hand_editor_message)

    def test_editor_guards_and_compensation_validation(self):
        config = RobotConfig()
        bridge = FakeBridge(config)
        now = 100.
        control = Controller(config, bridge, 'sim', clock=lambda: now)
        self.addCleanup(control.hand_file_executor.shutdown)

        def event(kind='hand_edit', client=1, **payload):
            nonlocal now
            now += .02
            bridge.state.received_ns = int((now-.000001) * 1e9)
            bridge.state.sequence += 1
            control.step()
            control.emit(kind, client, **payload)
            now += .001
            control.step()

        event('claim')
        original = config.hand_grasp.copy()
        for flags, stage, mode, client in [(35, p.HOLDING, 'sim', 1), (63, p.HOLDING, 'sim', 1),
                                           (55, p.RAMPING, 'sim', 1), (55, p.HOLDING, 'observe', 1),
                                           (55, p.HOLDING, 'sim', 2), (53, p.HOLDING, 'sim', 1)]:
            bridge.armed = True
            bridge.state.hand_flags = flags; bridge.state.stage = stage; control.mode = mode
            event(joint=0, angle_rad=.2, client=client)
            np.testing.assert_array_equal(config.hand_grasp, original)
        control.mode = 'sim'; bridge.state.hand_flags = 55
        event(joint=0, angle_rad=.2)
        self.assertTrue(bridge.live_request)
        self.assertEqual(config.hand_grasp[0, 0], .2)
        for invalid in (float('nan'), float('inf'), 2., True, '0.1'):
            event(joint=0, angle_rad=invalid)
            self.assertEqual(config.hand_grasp[0, 0], .2)
        event('hand_compensation', enabled=True)
        event(joint=7, angle_rad=.5)  # Raw angle fits; adding +.73 exceeds 1.05.
        self.assertEqual(config.hand_grasp[1, 0], original[1, 0])
        event(joint=7, angle_rad=.1)
        self.assertEqual(control.hand_pose, 4)
        self.assertAlmostEqual(config.hand_grasp_compensated[1, 0], .1 + config.hand_zero[1, 0])
        bridge.state.hand_flags = 35
        event(joint=0, angle_rad=.3)
        self.assertFalse(control.hand_live)
        self.assertEqual(control.hand_pose, 0)
        self.assertEqual(config.hand_grasp[0, 0], .2)


class GraspGuiTests(unittest.TestCase):
    def test_real_viser_widgets_and_callbacks_without_network(self):
        # Use Viser's actual widget implementation and message dispatcher with
        # an in-memory transport. No event loop, sockets, DDS or robot needed.
        import viser
        from viser._messages import GuiUpdateMessage

        def complete(coroutine, *_):
            future = Future()
            try:
                coroutine.send(None)
            except StopIteration as result:
                future.set_result(result.value)
            else:
                coroutine.close()
                self.fail('Unexpected asynchronous I/O in a widget callback')
            return future

        config = RobotConfig()
        bridge = FakeBridge(config)
        control = Controller(config, bridge, 'sim')
        self.addCleanup(control.hand_file_executor.shutdown)
        server = object.__new__(viser.ViserServer)
        server._websock_server = Mock()
        client = SimpleNamespace(client_id=1)
        server._connected_clients = {1: client}
        client.gui = viser.GuiApi(server, Mock(), Mock())
        server.gui = client.gui
        with patch('viser._gui_handles.asyncio.run_coroutine_threadsafe', side_effect=complete):
            panel = HandGraspPanel(client, control)
            self.assertEqual(len(panel.sliders), 14)
            self.assertFalse(panel.folder.visible)
            bridge.armed = True; bridge.state.stage = p.HOLDING; bridge.state.hand_flags = 55
            control.stage = 'HOLDING'; control.hand_pose = 2
            control.web.owner = 1; control.web.source = 'web'
            bridge.state.received_ns = time.monotonic_ns()
            control.step()
            panel.update(control.snapshot()['dex3'], True, True)
            self.assertTrue(panel.folder.visible)
            self.assertTrue(all(not slider.disabled for slider in panel.sliders))
            for index, slider in enumerate(panel.sliders):
                degrees = round(float(np.rad2deg(config.hand_grasp.flat[index]) * .9), 2)
                if degrees == 0.: degrees = 5.
                complete(client.gui._handle_gui_updates(1, GuiUpdateMessage(slider._impl.uuid, {'value': degrees})))
                bridge.state.received_ns = time.monotonic_ns(); bridge.state.sequence += 1
                control.step()
                self.assertAlmostEqual(config.hand_grasp.flat[index], np.deg2rad(degrees))
                panel.update(control.snapshot()['dex3'], True, True)
                self.assertTrue(control.events.empty())  # Programmatic synchronization never emits commands.
            control.hand_compensation = True
            panel.update(control._hand_status(bridge.state), True, True)
            self.assertLessEqual(np.deg2rad(panel.sliders[7].max)+config.hand_zero[1, 0], 1.05)
            panel.update(control._hand_status(bridge.state), False, True)
            self.assertTrue(all(slider.disabled for slider in panel.sliders))
            self.assertTrue(panel.save.disabled and panel.load.disabled)
            bridge.state.hand_flags = 35
            panel.update(control._hand_status(bridge.state), True, True)
            self.assertTrue(all(slider.disabled for slider in panel.sliders))


class GraspIntegrationTests(unittest.TestCase):
    def test_actual_browser_slider_save_load_and_other_viewer_guards(self):
        config = RobotConfig()
        with tempfile.TemporaryDirectory(prefix='viser_grasp_web_') as directory:
            directory = Path(directory)
            source, binary = directory/'pose.cpp', directory/'pose'
            source.write_text(HARNESS)
            subprocess.run([sys.executable, '-B', str(config.deploy/'tools/generate_config.py'),
                            str(config.deploy/'config/deploy.json'), str(directory/'generated_config.hpp')], check=True)
            subprocess.run(['g++', '-std=c++17', '-O2', '-I'+str(config.deploy/'cpp'),
                            '-I'+str(directory), str(source), '-o', str(binary)], check=True)
            config.upper_pose_path = directory/'pose.yaml'
            config.upper_pose_path.write_text(UPPER_POSE_PATH.read_text())
            bridge = PipeBridge(binary, config)
            control = Controller(config, bridge, 'sim', clock=bridge.now)
            viewer, browsers = None, []
            ui_clock = patch('Vender.app.time', SimpleNamespace(monotonic=bridge.now))
            ui_clock.start()
            try:
                control.step()
                with socket.socket() as probe:
                    probe.bind(('127.0.0.1', 0)); port = probe.getsockname()[1]
                viewer = Viewer(control, '127.0.0.1', port)
                owner = Browser(port); browsers.append(owner)
                other = Browser(port); browsers.append(other)

                def pump(predicate, timeout=12):
                    deadline = time.monotonic()+timeout
                    while time.monotonic()<deadline:
                        control.step(); viewer.update()
                        if predicate(): return
                        time.sleep(.01)
                    self.fail('Browser edit did not reach controller')

                viewer.update()
                owner.click('接管网页操作')
                pump(lambda: control.web.owner is not None)
                # PipeBridge is already armed; keep the normal explicit prepare and hand-enable buttons.
                self.assertTrue(all(not panel.folder.visible for panel in viewer.hand_panels.values()))
                owner.click('② 移到准备姿态')
                pump(lambda: control.stage == 'HOLDING')
                owner.click('Dex3 使能并抓握（3 秒）')
                pump(lambda: control.snapshot()['dex3']['active'] and not control.snapshot()['dex3']['moving'])
                label = '左手拇指 0 (°)'
                uid = until(lambda: owner.find(label))
                until(lambda: not owner.components[uid]['disabled'])
                self.assertTrue(other.components[other.find(label)]['disabled'])
                self.assertTrue(all(panel.folder.visible for panel in viewer.hand_panels.values()))
                other.set_slider(label, 30.)
                for _ in range(4): control.step(); viewer.update(); time.sleep(.02)
                self.assertEqual(config.hand_grasp[0, 0], 0.)
                owner.set_slider(label, 10.)
                pump(lambda: abs(config.hand_grasp[0, 0]-np.deg2rad(10))<1e-8)
                owner.set_slider('右手食指 1 (°)', 65.)
                pump(lambda: abs(config.hand_grasp[1, 6]-np.deg2rad(65))<1e-8)
                pump(lambda: not control.snapshot()['dex3']['moving'])
                expected = config.hand_grasp.copy()
                owner.click('保存抓握姿态')
                pump(lambda: '已保存' in control.hand_editor_message)
                np.testing.assert_array_equal(load_grasp(config.upper_pose_path), expected)
                owner.set_slider(label, 5.)
                pump(lambda: abs(config.hand_grasp[0, 0]-np.deg2rad(5))<1e-8)
                owner.click('加载并应用抓握姿态')
                pump(lambda: '已加载文件' in control.hand_editor_message)
                pump(lambda: not control.snapshot()['dex3']['moving'])
                np.testing.assert_array_equal(config.hand_grasp, expected)
                np.testing.assert_allclose(bridge.state.hand_target, expected.ravel(), atol=1e-6)
                until(lambda: abs(owner.components[uid]['value']-10.)<1e-6)
                until(lambda: abs(other.components[other.find(label)]['value']-10.)<1e-6)
                owner.click('阻尼停止')
                pump(lambda: control.stage == 'DAMPED')
                self.assertTrue(all(panel.save.disabled for panel in viewer.hand_panels.values()))
            finally:
                for browser in browsers: browser.close()
                if viewer: viewer.close()
                bridge.close(); control.hand_file_executor.shutdown(); ui_clock.stop()

    def test_stream_save_reload_and_toggle_use_edited_grasp(self):
        config = RobotConfig()
        with tempfile.TemporaryDirectory(prefix='viser_grasp_cpp_') as directory:
            directory = Path(directory)
            source, binary = directory/'pose.cpp', directory/'pose'
            source.write_text(HARNESS)
            subprocess.run([sys.executable, '-B', str(config.deploy/'tools/generate_config.py'),
                            str(config.deploy/'config/deploy.json'), str(directory/'generated_config.hpp')], check=True)
            subprocess.run(['g++', '-std=c++17', '-O2', '-I'+str(config.deploy/'cpp'),
                            '-I'+str(directory), str(source), '-o', str(binary)], check=True)
            path = directory/'pose.yaml'
            path.write_text(UPPER_POSE_PATH.read_text())
            config.upper_pose_path = path
            bridge = PipeBridge(binary, config)
            control = Controller(config, bridge, 'sim', clock=bridge.now)

            def event(kind, **payload):
                control.emit(kind, 1, **payload)
                control.step()

            def settle():
                for _ in range(450):
                    old = bridge.state.hand_target.copy()
                    was_editing = bool(bridge.state.hand_flags & 256)
                    control.step()
                    self.assertNotIn(control.stage, ('DAMPED', 'FAULT'))
                    if was_editing:
                        self.assertLessEqual(float(np.max(abs(bridge.state.hand_target-old))), .010001)
                    if not bridge.state.hand_flags & 8:
                        control.step()
                        return
                self.fail('Hand did not settle')

            def file_action(kind):
                event(kind)
                for _ in range(100):
                    control.step()
                    if control.hand_file is None: return
                    time.sleep(.005)
                self.fail('File operation did not complete')

            try:
                control.step(); event('claim'); event('prepare')
                for _ in range(300):
                    control.step()
                    if bridge.state.stage == p.HOLDING: break
                control.step()
                event('hand_toggle'); settle()
                body = bridge.state.applied_q.copy()
                event('hand_edit', joint=0, angle_rad=.3)
                self.assertTrue(bridge.state.hand_flags & 256)
                self.assertTrue(bridge.state.hand_flags & 8)
                event('hand_edit', joint=13, angle_rad=.9)  # Accept new targets while live following.
                event('hand_edit', joint=0, angle_rad=-.15)
                settle()
                np.testing.assert_allclose(bridge.state.hand_target, config.hand_grasp.ravel(), atol=1e-6)
                np.testing.assert_array_equal(bridge.state.applied_q, body)
                expected = config.hand_grasp.copy()
                file_action('hand_save')
                np.testing.assert_array_equal(load_grasp(path), expected)
                self.assertFalse(control.snapshot()['dex3']['grasp_dirty'])
                event('hand_edit', joint=0, angle_rad=.1); settle()
                file_action('hand_load'); settle()
                np.testing.assert_array_equal(config.hand_grasp, expected)
                np.testing.assert_allclose(bridge.state.hand_target, expected.ravel(), atol=1e-6)
                event('hand_toggle'); settle()
                np.testing.assert_array_equal(bridge.state.hand_target, 0)
                event('hand_toggle'); settle()
                np.testing.assert_allclose(bridge.state.hand_target, expected.ravel(), atol=1e-6)
                event('hand_compensation', enabled=True)
                event('hand_edit', joint=7, angle_rad=.1); settle()
                np.testing.assert_allclose(bridge.state.hand_target,
                                           (config.hand_grasp+config.hand_zero).ravel(), atol=1e-6)
                # File I/O errors must retain the current target and body support.
                expected = config.hand_grasp.copy()
                path.write_text('hand_grasp: invalid')
                file_action('hand_load')
                np.testing.assert_array_equal(config.hand_grasp, expected)
                self.assertIn('失败', control.hand_editor_message)
                self.assertEqual(bridge.state.stage, p.HOLDING)
                event('disconnect')
                self.assertEqual(bridge.state.stage, p.DAMPED)
                self.assertFalse(bridge.state.hand_flags & (4 | 256))
            finally:
                bridge.close()
                control.hand_file_executor.shutdown()


if __name__ == '__main__':
    unittest.main()
