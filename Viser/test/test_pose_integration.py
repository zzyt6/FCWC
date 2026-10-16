"""Python controller + real C++ control cores and wire structs, over offline pipes."""
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.robot import RobotConfig, ACTIVE, UPPER
from core.controller import Controller
from hybrid_deploy import protocol as p
from hybrid_deploy.configuration import load_config
from core.joint_tracking import snapshot as joint_snapshot, INDICES
from core.joint_monitor import limits, out_of_range, live_warnings


HARNESS = r'''
#include "control_core.hpp"
#include "hand_core.hpp"
#include "command_trace.hpp"
#include <iostream>
using namespace decoupled;
struct Motor {
  float angle=0,gain=0;
  float q() const { return angle; }
  float kp() const { return gain; }
};
int main(int argc,char** argv) {
  if(argc!=30) return 1;
  const std::string scenario=argv[29];
  const bool stuck_hand=scenario=="stuck_hand";
  std::array<float,14> grasp,compensated_zero;
  for(int i=0;i<14;++i) {
    grasp[i]=std::stof(argv[i+1]); compensated_zero[i]=std::stof(argv[i+15]);
  }
  ControlCore core(42,false,true,true);
  HandCore hands(true,grasp,compensated_zero);
  Feedback f; f.valid=true; f.packet.quaternion[0]=1;
  f.packet.session=42; f.packet.config_token=config::token;
  f.packet.sequence=1; f.packet.received_ns=1000000000;
  for(int i=0;i<29;++i) f.packet.q[i]=config::default_q_sdk29[i];
  if(scenario=="body_limits") f.packet.q[11]=.286337f; // The logged right ankle roll excursion.
  HandFeedback hf; hf.valid.fill(true); hf.q=grasp;
  for(auto& q:hf.q) q*=.4f; // Start between open and grasp to exercise the first motion.
  if(stuck_hand) hf.q[6]=hand_lower[6]-.0438f;
  Output out;
  CommandTrace trace;
  int policy_ticks=0,prepare_ticks=0;
  auto update_body_feedback=[&] {
    // Reproduce the logged left-wrist residual until 12 seconds, then recover.
    if(scenario=="prepare_timeout" && prepare_ticks>=500 && prepare_ticks<2400)
      f.packet.q[19]=1.118835f;
    if(scenario!="body_limits" || policy_ticks<40 || policy_ticks>=40+54*8) return;
    const int event=(policy_ticks-40)/8, active_index=event%27;
    const int joint=active_index<13?active_index:active_index+2;
    f.packet.q[joint]=event<27?config::joint_upper_sdk29[joint]+.03f:
                              config::joint_lower_sdk29[joint]-.03f;
  };
  auto update_hand_feedback=[&] {
    const bool fault=policy_ticks>=60 && policy_ticks<160;
    for(int side=0;side<2;++side) {
      const std::string suffix=side==0?"left":"right";
      hf.valid[side]=!(fault && scenario=="sdk_"+suffix) &&
        !(scenario=="sdk_prepare" && prepare_ticks>=60 && prepare_ticks<160);
      if(scenario=="no_hand" && policy_ticks<160) { hf.received_ns[side]=0; hf.valid[side]=false; }
      else if(!(fault && scenario=="lost_"+suffix)) hf.received_ns[side]=f.packet.received_ns;
    }
  };
  update_hand_feedback();
  auto packet=[&] {
    StatePacket s=f.packet; s.flags=6U|(core.healthy(f,f.packet.received_ns)?1U:0U); s.stage=uint32_t(core.stage);
    if(core.preparation_overdue(s.received_ns)) s.flags|=kPreparationOverdue;
    s.profile=out.profile; s.gain_blend=out.gain_blend;
    s.upper_mode=out.upper_mode; s.upper_gain_blend=out.upper_gain_blend;
    for(int i=0;i<29;++i) {
      s.applied_q[i]=out.publish?out.q[i]:f.packet.q[i];
      s.applied_kp[i]=(i==13 || i==14)?0:out.kp[i];
      s.applied_kd[i]=(i==13 || i==14)?0:out.kd[i];
    }
    hands.telemetry(s,hf,s.received_ns);
    std::copy(hf.received_ns.begin(),hf.received_ns.end(),s.hand_received_ns);
    trace.telemetry(s,s.received_ns);
    std::cout.write(reinterpret_cast<const char*>(&s),sizeof(s)); std::cout.flush();
  };
  packet();
  CommandPacket c;
  while(std::cin.read(reinterpret_cast<char*>(&c),sizeof(c))) {
    auto before=core.stage;
    hands.check_feedback(hf,f.packet.received_ns);
    if(core.command(c,f,f.packet.received_ns)) {
      if(before==Stage::Waiting && core.stage==Stage::Ramping) {
        if(hands.recovering) hands.cancel_recovery("body preparation requested");
        hands.suspend();
      }
      hands.command(c,core.stage,f.packet.received_ns,hf);
      if(c.request==uint32_t(Request::Wait))
        hands.request_recovery(c.hand_recover,core.stage,hf,f.packet.received_ns);
    }
    for(int tick=0;tick<4;++tick) {
      f.packet.received_ns+=5000000; ++f.packet.sequence;
      if(core.stage==Stage::Policy) ++policy_ticks;
      if(core.stage==Stage::Ramping) ++prepare_ticks;
      update_body_feedback();
      update_hand_feedback();
      hands.check_feedback(hf,f.packet.received_ns);
      // Recovery warnings end the hand attempt without changing the body core.
      hands.step_recovery(hf,f.packet.received_ns);
      out=core.step(f,f.packet.received_ns);
      hands.step(f.packet.received_ns,core.stage==Stage::Damped || core.stage==Stage::Fault);
      std::array<Motor,29> body;
      for(int i=0;i<29;++i) body[i]={out.q[i],(i==13 || i==14)?0:out.kp[i]};
      if(out.publish) trace.published(0,body,f.packet.received_ns);
      if(hands.ever_enabled) for(int side=0;side<2;++side) {
        std::array<Motor,7> motors;
        for(int j=0;j<7;++j) motors[j]={hands.q[side*7+j],hands.recovering?
          (side*7+j==hands.recovery_joint?hands.recovery_kp:0.f):(hands.active?.5f:0.f)};
        trace.published(side+1,motors,f.packet.received_ns);
      }
      for(int i=0;i<29;++i) if(out.publish && out.kp[i]>0)
        f.packet.q[i]+=.15f*(out.q[i]-f.packet.q[i]);
      update_body_feedback();
      if(hands.ever_enabled && !stuck_hand) { hf.q=hands.q; hf.dq=hands.dq; }
    }
    packet();
  }
}
'''


class PipeBridge:
    def __init__(self, executable, config, scenario='normal'):
        self.process = subprocess.Popen([str(executable), *map(str, config.hand_grasp.flat),
                                         *map(str, config.hand_zero.flat), scenario],
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        self.state = p.State.decode(self.process.stdout.read(p.STATE.size))
        self.sequence = 0
        self.armed = True
        self.started = self.now()

    def now(self):
        return self.state.received_ns*1e-9 + .000001

    def read(self):
        return self.state

    def send(self, request, q, profile=0, hand_pose=0, hand_recover=-1, hand_grasp=None, hand_live=False,
             upper_mode=0, upper_target_active=False):
        self.sequence += 1
        self.process.stdin.write(p.command(self.state, self.sequence, request, q, profile, hand_pose, hand_recover,
                                          hand_grasp, hand_live, upper_mode, upper_target_active))
        self.process.stdin.flush()
        self.state = p.State.decode(self.process.stdout.read(p.STATE.size))

    def close(self):
        self.process.stdin.close()
        try:
            self.process.wait(timeout=5)
        finally:
            if self.process.poll() is None:
                self.process.kill()
                self.process.wait()
            self.process.stdout.close()


class PoseIntegrationTests(unittest.TestCase):
    def test_preparation_timeout_warns_and_keeps_support_until_measured_arrival(self):
        config = RobotConfig()
        _, gains, _ = load_config(config.deploy)
        lower, upper = limits(config)
        with tempfile.TemporaryDirectory(prefix='viser_prepare_warning_') as directory:
            source, binary = Path(directory)/'pose.cpp', Path(directory)/'pose'
            source.write_text(HARNESS)
            subprocess.run([sys.executable, '-B', str(config.deploy/'tools/generate_config.py'),
                            str(config.deploy/'config/deploy.json'),
                            str(Path(directory)/'generated_config.hpp')], check=True)
            subprocess.run(['g++', '-std=c++17', '-O2', '-I'+str(config.deploy/'cpp'),
                            '-I'+directory, str(source), '-o', str(binary)], check=True)
            bridge = PipeBridge(binary, config, 'prepare_timeout')
            recorder = Mock(error='', dropped=0)
            control = Controller(config, bridge, 'sim', recorder=recorder, clock=bridge.now)
            try:
                control.step(); control.emit('claim', 1); control.step()
                control.emit('prepare', 1); control.step()
                start = bridge.now()
                warned = False
                warning_samples = 0
                body_mask = sum(1 << int(index) for index in ACTIVE)
                for _ in range(700):
                    control.step()
                    data = control.snapshot()
                    self.assertNotIn(data['stage'], ('DAMPED', 'FAULT'))
                    self.assertTrue(data['healthy'])
                    self.assertTrue(np.all(bridge.state.applied_kp[ACTIVE] > 0))
                    self.assertEqual(bridge.state.position_mask & body_mask, body_mask)
                    self.assertEqual(tuple(bridge.state.sent_ns[1:]), (0, 0))
                    if data['preparation_warning']:
                        warning_samples += 1
                        self.assertGreater(bridge.now() - start, 9.9)
                        self.assertEqual(data['stage'], 'PREPARING')
                        self.assertIn('继续保持位置控制', data['preparation_warning'])
                        warnings = live_warnings(data, data['joint_comparison'], lower, upper,
                                                 .1, bridge.now())
                        self.assertIn(('warning', data['preparation_warning']), warnings)
                        np.testing.assert_allclose(bridge.state.applied_q[ACTIVE],
                                                   config.prepare_q[ACTIVE], atol=1e-4)
                        self.assertEqual(bridge.state.applied_kp[19], gains['balance']['kp_sdk29'][19])
                        self.assertEqual(bridge.state.applied_kd[19], gains['balance']['kd_sdk29'][19])
                        if not warned:
                            self.assertIn('left_wrist_roll_joint', data['preparation_warning'])
                            self.assertAlmostEqual(config.prepare_q[19] - data['state']['q'][19],
                                                   .103083, places=5)
                            control.emit('policy', 1)  # Cannot bypass measured readiness.
                            warned = True
                        self.assertNotEqual(control.request, p.RUN)
                    if data['stage'] == 'HOLDING':
                        break
                self.assertGreater(warning_samples, 50)
                self.assertEqual(control.stage, 'HOLDING')
                self.assertEqual(control.snapshot()['preparation_warning'], '')
                self.assertFalse(bridge.state.flags & p.PREPARATION_OVERDUE)
                transitions = [call.args[1] for call in recorder.record.call_args_list
                               if call.args[0] == 'events' and
                               call.args[1].get('type') == 'preparation_warning']
                self.assertEqual([event['active'] for event in transitions], [True, False])
                for _ in range(8): control.step()
                control.emit('policy', 1); control.step(); control.step()
                self.assertEqual(control.stage, 'POLICY')
                control.emit('stop', 1); control.step()
                self.assertEqual(bridge.state.stage, p.DAMPED)
            finally:
                bridge.close()

    def test_first_hand_click_enables_grasp_only_after_body_is_ready(self):
        config = RobotConfig()
        with tempfile.TemporaryDirectory(prefix='viser_deferred_hand_') as directory:
            source, binary = Path(directory)/'pose.cpp', Path(directory)/'pose'
            source.write_text(HARNESS)
            subprocess.run([sys.executable, '-B', str(config.deploy/'tools/generate_config.py'),
                            str(config.deploy/'config/deploy.json'),
                            str(Path(directory)/'generated_config.hpp')], check=True)
            subprocess.run(['g++', '-std=c++17', '-O2', '-I'+str(config.deploy/'cpp'),
                            '-I'+directory, str(source), '-o', str(binary)], check=True)
            for compensated in (False, True):
                with self.subTest(compensated=compensated):
                    bridge = PipeBridge(binary, config)
                    control = Controller(config, bridge, 'sim', clock=bridge.now)
                    def event(kind, client=1, **payload):
                        control.emit(kind, client, **payload); control.step()
                    def assert_hand_idle():
                        self.assertFalse(bridge.state.hand_flags & (4 | 8 | 128))
                        self.assertEqual(tuple(bridge.state.sent_ns[1:]), (0, 0))
                        self.assertEqual(bridge.state.position_mask >> 29, 0)
                        np.testing.assert_array_equal(bridge.state.hand_q, initial)
                    try:
                        initial = bridge.state.hand_q.copy()
                        control.step(); event('claim')
                        self.assertEqual(control.hand_pose, 0)
                        assert_hand_idle()
                        event('hand_toggle')  # Rejected before preparation.
                        assert_hand_idle()
                        event('hand_compensation', enabled=compensated)
                        assert_hand_idle()
                        # Even a legacy PREPARE packet carrying a hand pose must
                        # never enable hands, including after body preparation.
                        for _ in range(450):
                            bridge.send(p.PREPARE, config.prepare_q, hand_pose=3 if compensated else 1)
                            assert_hand_idle()
                            if bridge.state.stage == p.HOLDING: break
                            event('hand_toggle')  # Rejected while arms are moving.
                            assert_hand_idle()
                        self.assertEqual(bridge.state.stage, p.HOLDING)
                        bridge.send(p.PREPARE, config.prepare_q, hand_pose=3 if compensated else 1)
                        assert_hand_idle()
                        for _ in range(8): control.step(); assert_hand_idle()
                        if compensated:
                            event('policy')
                            for _ in range(20): control.step(); assert_hand_idle()
                            self.assertEqual(bridge.state.stage, p.POLICY)
                        event('hand_toggle', client=2)
                        assert_hand_idle()
                        event('hand_toggle')
                        self.assertEqual(control.hand_pose, 4 if compensated else 2)
                        self.assertTrue(bridge.state.hand_flags & 4)
                        self.assertTrue(bridge.state.hand_flags & 8)
                        self.assertTrue(all(bridge.state.sent_ns[1:]))
                        np.testing.assert_allclose(bridge.state.hand_target, initial, atol=1e-4)
                        for _ in range(155): control.step()
                        expected = config.hand_grasp_compensated if compensated else config.hand_grasp
                        np.testing.assert_allclose(bridge.state.hand_target, expected.ravel(), atol=1e-6)
                        event('hand_toggle')
                        for _ in range(155): control.step()
                        expected = config.hand_zero.ravel() if compensated else np.zeros(14)
                        np.testing.assert_allclose(bridge.state.hand_target, expected, atol=1e-6)
                    finally:
                        bridge.close()

    def test_all_body_joints_outside_reference_keep_policy_and_position_control(self):
        config = RobotConfig()
        lower, upper = limits(config)
        with tempfile.TemporaryDirectory(prefix='viser_body_warning_') as directory:
            source, binary = Path(directory)/'pose.cpp', Path(directory)/'pose'
            source.write_text(HARNESS)
            subprocess.run([sys.executable, '-B', str(config.deploy/'tools/generate_config.py'),
                            str(config.deploy/'config/deploy.json'),
                            str(Path(directory)/'generated_config.hpp')], check=True)
            subprocess.run(['g++', '-std=c++17', '-O2', '-I'+str(config.deploy/'cpp'),
                            '-I'+directory, str(source), '-o', str(binary)], check=True)
            bridge = PipeBridge(binary, config, 'body_limits')
            control = Controller(config, bridge, 'sim', clock=bridge.now)
            def event(kind):
                control.emit(kind, 1); control.step()
            try:
                control.step(); event('claim')
                self.assertTrue(control.snapshot()['healthy'])
                self.assertIn(11, out_of_range(control.snapshot()['joint_comparison'], lower, upper))
                event('prepare')
                self.assertEqual(bridge.state.stage, p.RAMPING)
                for _ in range(450):
                    control.step()
                    self.assertNotIn(bridge.state.stage, (p.DAMPED, p.FAULT))
                    if bridge.state.stage == p.HOLDING: break
                self.assertEqual(bridge.state.stage, p.HOLDING)
                for _ in range(8): control.step()
                event('policy')
                seen_upper, seen_lower = set(), set()
                last_sent = bridge.state.sent_ns[0]
                body_mask = sum(1 << index for index in ACTIVE)
                for _ in range(180):
                    control.step()
                    self.assertEqual(bridge.state.stage, p.POLICY)
                    self.assertTrue(bridge.state.flags & 1)
                    self.assertEqual(bridge.state.position_mask & body_mask, body_mask)
                    self.assertTrue(np.all(bridge.state.applied_kp[list(ACTIVE)] > 0))
                    self.assertEqual(bridge.state.sent_ns[0] - last_sent, 20_000_000)
                    last_sent = bridge.state.sent_ns[0]
                    data = control.snapshot()
                    c = data['joint_comparison']
                    for index in out_of_range(c, lower, upper):
                        if index < 27:
                            (seen_upper if c['actual_rad'][index] > upper[index] else seen_lower).add(index)
                    warnings = live_warnings(data, c, lower, upper, .2, now=bridge.now())
                    self.assertTrue(all(level == 'warning' for level, text in warnings if '位置超限' in text))
                self.assertEqual(seen_upper, set(range(27)))
                self.assertEqual(seen_lower, set(range(27)))
                self.assertFalse(out_of_range(control.snapshot()['joint_comparison'], lower, upper))
                self.assertTrue(bridge.armed)
                event('stop')
                self.assertEqual(bridge.state.stage, p.DAMPED)
                self.assertEqual(bridge.state.position_mask, 0)
            finally:
                bridge.close()

    def test_hand_sdk_and_feedback_loss_never_stop_body_and_require_manual_resume(self):
        config = RobotConfig()
        with tempfile.TemporaryDirectory(prefix='viser_hand_fault_') as directory:
            source, binary = Path(directory)/'pose.cpp', Path(directory)/'pose'
            source.write_text(HARNESS)
            subprocess.run([sys.executable, '-B', str(config.deploy/'tools/generate_config.py'),
                            str(config.deploy/'config/deploy.json'),
                            str(Path(directory)/'generated_config.hpp')], check=True)
            subprocess.run(['g++', '-std=c++17', '-O2', '-I'+str(config.deploy/'cpp'),
                            '-I'+directory, str(source), '-o', str(binary)], check=True)
            for scenario in ('sdk_left', 'sdk_right', 'lost_left', 'lost_right', 'no_hand', 'sdk_prepare'):
                with self.subTest(scenario=scenario):
                    bridge = PipeBridge(binary, config, scenario)
                    control = Controller(config, bridge, 'sim', clock=bridge.now)
                    def event(kind):
                        control.emit(kind, 1)
                        control.step()
                    try:
                        control.step(); event('claim'); event('prepare')
                        self.assertEqual(bridge.state.stage, p.RAMPING)
                        for _ in range(400):
                            control.step()
                            self.assertNotIn(bridge.state.stage, (p.DAMPED, p.FAULT))
                            if bridge.state.stage == p.HOLDING: break
                        self.assertEqual(bridge.state.stage, p.HOLDING)
                        for _ in range(8): control.step()
                        event('policy')
                        if scenario not in ('no_hand', 'sdk_prepare'):
                            event('hand_toggle')  # Inject the fault while a grasp is moving.
                            self.assertTrue(bridge.state.hand_flags & 8)
                        saw_pause = False
                        body_mask = sum(1 << index for index in ACTIVE)
                        for _ in range(100):
                            control.step()
                            self.assertEqual(bridge.state.stage, p.POLICY)
                            self.assertTrue(np.all(bridge.state.applied_kp[list(ACTIVE)] > 0))
                            self.assertEqual(bridge.state.position_mask & body_mask, body_mask)
                            if not bridge.state.hand_flags & 4:
                                saw_pause = True
                                self.assertEqual(bridge.state.position_mask >> 29, 0)
                        self.assertTrue(saw_pause)
                        self.assertEqual(bridge.state.hand_flags & 35, 35)  # Reconnected.
                        self.assertFalse(bridge.state.hand_flags & (4 | 8))  # Still paused.
                        self.assertEqual(control.hand_pose, 0)
                        self.assertTrue(bridge.armed)
                        control.hand_compensation = True
                        event('hand_toggle')
                        self.assertTrue(bridge.state.hand_flags & 4)
                        self.assertTrue(bridge.state.hand_flags & 8)
                        for _ in range(155):
                            control.step()
                            self.assertEqual(bridge.state.stage, p.POLICY)
                        expected = (config.hand_grasp_compensated if scenario in ('no_hand', 'sdk_prepare')
                                    else config.hand_zero)
                        np.testing.assert_allclose(bridge.state.hand_target, expected.ravel(), atol=1e-6)
                        self.assertFalse(bridge.state.hand_flags & 8)
                        event('stop')
                        self.assertEqual(bridge.state.stage, p.DAMPED)
                        self.assertEqual(bridge.state.position_mask, 0)
                    finally:
                        bridge.close()

    def test_dex3_limit_and_recovery_timeout_keep_body_available(self):
        config = RobotConfig()
        with tempfile.TemporaryDirectory(prefix='viser_hand_warning_') as directory:
            source, binary = Path(directory)/'pose.cpp', Path(directory)/'pose'
            source.write_text(HARNESS)
            subprocess.run([sys.executable, '-B', str(config.deploy/'tools/generate_config.py'),
                            str(config.deploy/'config/deploy.json'),
                            str(Path(directory)/'generated_config.hpp')], check=True)
            subprocess.run(['g++', '-std=c++17', '-O2', '-I'+str(config.deploy/'cpp'),
                            '-I'+directory, str(source), '-o', str(binary)], check=True)
            for recover_first in (False, True):
                with self.subTest(recover_first=recover_first):
                    bridge = PipeBridge(binary, config, 'stuck_hand')
                    controller = Controller(config, bridge, 'sim', clock=bridge.now)
                    def event(kind, **payload):
                        controller.emit(kind, 1, **payload)
                        controller.step()
                    try:
                        controller.step(); event('claim')
                        self.assertTrue(controller.snapshot()['dex3']['healthy'])
                        if recover_first:
                            event('hand_recover', joint=6)
                            self.assertTrue(bridge.state.hand_flags & 128)
                            for _ in range(145):
                                controller.step()
                                self.assertTrue(bridge.state.hand_flags & 128)
                            for _ in range(8): controller.step()
                            self.assertFalse(bridge.state.hand_flags & 128)
                            self.assertEqual(controller.hand_recovery, -1)
                            self.assertEqual(controller.stage, 'ZERO')
                            self.assertEqual(bridge.state.position_mask, 0)
                            self.assertTrue(bridge.state.hand_recovery_mask & (1 << 6))
                        event('prepare')
                        self.assertEqual(bridge.state.stage, p.RAMPING)
                        for _ in range(400):
                            controller.step()
                            self.assertNotIn(bridge.state.stage, (p.DAMPED, p.FAULT))
                            if bridge.state.stage == p.HOLDING: break
                        self.assertEqual(bridge.state.stage, p.HOLDING)
                        for _ in range(8): controller.step()
                        event('policy')
                        for _ in range(200):
                            controller.step()
                            self.assertEqual(bridge.state.stage, p.POLICY)
                            self.assertTrue(np.all(bridge.state.applied_kp[:12] > 0))
                            self.assertLess(bridge.state.hand_q[6], -1.77)
                        self.assertTrue(controller.snapshot()['healthy'])
                    finally:
                        bridge.close()

    def test_yaml_preparation_and_dex3_roundtrip_through_cpp(self):
        config = RobotConfig()
        with tempfile.TemporaryDirectory(prefix='viser_pose_') as directory:
            source, binary = Path(directory)/'pose.cpp', Path(directory)/'pose'
            source.write_text(HARNESS)
            subprocess.run([sys.executable, '-B', str(config.deploy/'tools/generate_config.py'),
                            str(config.deploy/'config/deploy.json'),
                            str(Path(directory)/'generated_config.hpp')], check=True)
            subprocess.run(['g++', '-std=c++17', '-O2', '-I'+str(config.deploy/'cpp'),
                            '-I'+directory, str(source), '-o', str(binary)], check=True)
            bridge = PipeBridge(binary, config)
            controller = Controller(config, bridge, 'sim', clock=bridge.now)
            def event(kind, client=1, **payload):
                controller.emit(kind, client, **payload)
                controller.step()
            def advance_until(predicate, steps=600):
                for _ in range(steps):
                    if predicate():
                        return
                    controller.step()
                    self.assertNotIn(controller.stage, ('DAMPED', 'FAULT'))
                self.fail('C++ pose transition did not finish')
            try:
                controller.step()
                event('claim')
                self.assertTrue(all(v is None for v in controller.snapshot()['joint_comparison']['command_rad']))
                initial = bridge.state.hand_q.copy()
                event('prepare')
                self.assertEqual(bridge.state.stage, p.RAMPING)
                self.assertEqual(tuple(bridge.state.sent_ns[1:]), (0, 0))
                self.assertFalse(bridge.state.hand_flags & (4 | 8))
                advance_until(lambda: bridge.state.stage == p.HOLDING)
                np.testing.assert_allclose(bridge.state.applied_q[ACTIVE], config.prepare_q[ACTIVE], atol=1e-4)
                np.testing.assert_array_equal(bridge.state.hand_target, 0)
                np.testing.assert_array_equal(bridge.state.hand_q, initial)
                self.assertEqual(tuple(bridge.state.sent_ns[1:]), (0, 0))
                np.testing.assert_array_equal(bridge.state.applied_kp[13:15], 0)
                controller.step()
                event('hand_toggle', client=2)
                self.assertFalse(bridge.state.hand_flags & 16)
                event('hand_toggle')
                self.assertTrue(bridge.state.hand_flags & 8)
                self.assertTrue(bridge.state.hand_flags & 4)
                np.testing.assert_allclose(bridge.state.hand_target, initial, atol=1e-4)
                trajectory = []
                while bridge.state.hand_flags & 8:
                    trajectory.append((bridge.now(), bridge.state.hand_target.copy()))
                    comparison = joint_snapshot(bridge.state, bridge.now(), .2)
                    self.assertTrue(all(comparison['valid']))
                    np.testing.assert_allclose(comparison['command_rad'][-14:], bridge.state.hand_target)
                    np.testing.assert_allclose(comparison['command_rad'][:27], bridge.state.applied_q[list(INDICES[:27])])
                    event('hand_toggle')  # Repeated clicks cannot reverse the in-flight ramp.
                    self.assertTrue(bridge.state.hand_flags & 16)
                    self.assertLess(len(trajectory), 160)
                self.assertAlmostEqual(trajectory[-1][0]-trajectory[0][0], 2.96, delta=.03)
                samples = np.asarray([q for _, q in trajectory])
                self.assertTrue(np.all(np.diff(abs(samples), axis=0) >= -1e-6))
                self.assertLess(np.max(abs(np.diff(samples, axis=0))), .025)
                np.testing.assert_allclose(bridge.state.hand_target, config.hand_grasp.ravel(), atol=1e-6)
                np.testing.assert_allclose(bridge.state.applied_q[UPPER], config.prepare_q[UPPER], atol=1e-4)
                controller.step()
                event('hand_compensation', client=2, enabled=True)
                self.assertFalse(controller.hand_compensation)
                event('hand_compensation', enabled=True)
                self.assertTrue(controller.hand_compensation)
                # Same toggle now releases to compensated zero.
                event('hand_toggle')
                self.assertTrue(bridge.state.hand_flags & 64)
                trajectory = []
                while bridge.state.hand_flags & 8:
                    trajectory.append((bridge.now(), bridge.state.hand_target.copy()))
                    event('hand_compensation', enabled=False)  # Busy selection cannot change.
                    self.assertTrue(controller.hand_compensation)
                    self.assertLess(len(trajectory), 160)
                self.assertAlmostEqual(trajectory[-1][0]-trajectory[0][0], 2.96, delta=.06)
                np.testing.assert_allclose(bridge.state.hand_target, config.hand_zero.ravel(), atol=1e-6)
                controller.step()
                event('hand_toggle')
                advance_until(lambda: not bridge.state.hand_flags & 8)
                np.testing.assert_allclose(bridge.state.hand_target, config.hand_grasp_compensated.ravel(), atol=1e-6)
                controller.step()
                self.assertEqual(controller.snapshot()['dex3']['pose'], 'grasp_compensated')
                self.assertAlmostEqual(controller.snapshot()['joint_comparison']['command_rad'][34], config.hand_zero[1,0], places=6)
                event('hand_compensation', enabled=False)
                event('hand_toggle')
                advance_until(lambda: not bridge.state.hand_flags & 8)
                np.testing.assert_array_equal(bridge.state.hand_target, 0)
                controller.step()
                event('hand_toggle')
                advance_until(lambda: not bridge.state.hand_flags & 8)
                np.testing.assert_allclose(bridge.state.hand_target, config.hand_grasp.ravel(), atol=1e-6)
                for _ in range(8): controller.step()
                event('policy')
                self.assertEqual(bridge.state.stage, p.POLICY)
                controller.step()
                event('hand_toggle')
                advance_until(lambda: not bridge.state.hand_flags & 8)
                np.testing.assert_array_equal(bridge.state.hand_target, 0)
                np.testing.assert_allclose(bridge.state.applied_q[15:], config.prepare_q[15:], atol=1e-4)
                controller.step()
                event('hand_compensation', enabled=True)
                event('hand_toggle')
                self.assertTrue(bridge.state.hand_flags & 8)
                event('disconnect')
                self.assertEqual(bridge.state.stage, p.DAMPED)
                self.assertEqual(bridge.state.hand_flags & 12, 0)
                comparison = joint_snapshot(bridge.state, bridge.now(), .2)
                self.assertFalse(any(comparison['valid']))
                self.assertTrue(all(v is None for v in comparison['error_rad']))
                stopped = bridge.state.hand_target.copy()
                event('hand_toggle')
                event('hand_zero_compensated')
                event('hand_grasp_compensated')
                np.testing.assert_array_equal(bridge.state.hand_target, stopped)
            finally:
                bridge.close()


if __name__ == '__main__':
    unittest.main(verbosity=2)
