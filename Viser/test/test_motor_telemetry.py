"""C++ feedback extraction and optional wire data, without a robot or DDS."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'G1Deploy_BalanceHomie/python'))
from hybrid_deploy import protocol as p

SOURCE = r'''
#include "hand_core.hpp"
#include <iostream>
#include <vector>
using namespace decoupled;
struct Motor {
  int i;
  int mode() const { return i%3; }
  float ddq() const { return i+.25f; }
  float tau_est() const { return i+.5f; }
  float vol() const { return i+48.f; }
};
struct Imu {
  std::array<float,3> acc{1,2,3};
  const auto& accelerometer() const { return acc; }
};
struct Pressure {
  std::array<float,12> values{1,2,3,4,5,6,7,8,9,10,11,12};
  const auto& pressure() const { return values; }
};
struct Message {
  std::vector<Motor> motors;
  std::vector<Pressure> sensors{Pressure{}};
  Imu imu;
  const auto& motor_state() const { return motors; }
  const auto& imu_state() const { return imu; }
  const auto& press_sensor_state() const { return sensors; }
  unsigned tick() const { return 12345; }
  int mode_pr() const { return 2; }
  float power_v() const { return 24; }
  float power_a() const { return 1.5; }
  float system_v() const { return 5; }
  float device_v() const { return 3.3; }
};
int main() {
  Message msg;
  for(int i=0;i<29;++i) msg.motors.push_back({i});
  MotorTelemetry telemetry;
  telemetry.body=body_telemetry(msg);
  msg.motors.resize(7);
  HandFeedbackSources sources;
  sources.samples[1].received_ns[0]=1000;
  sources.samples[1].valid[0]=true;
  sources.samples[1].motor_telemetry[0]=hand_telemetry(msg,1);
  auto selected=sources.snapshot(2000);
  telemetry.hands[0]=selected.motor_telemetry[0];
  msg.motors.resize(6);
  telemetry.hands[1]=hand_telemetry(msg,0); // incomplete message stays unavailable
  StatePacket state;
  state.flags=1; state.received_ns=2000; state.bridge_ns=2000;
  for(int i=0;i<29;++i) state.q[i]=i;
  std::cout.write(reinterpret_cast<const char*>(&state),sizeof(state));
  std::cout.write(reinterpret_cast<const char*>(&telemetry),sizeof(telemetry));
}
'''


class MotorTelemetryTests(unittest.TestCase):
    def test_cpp_fields_and_optional_tail_preserve_base_feedback(self):
        with tempfile.TemporaryDirectory(prefix='motor_telemetry_wire_') as directory:
            binary = Path(directory) / 'wire'
            subprocess.run(['g++', '-std=c++17', '-O2', '-x', 'c++', '-',
                            '-I' + str(ROOT / 'G1Deploy_BalanceHomie/cpp'), '-o', str(binary)],
                           input=SOURCE, text=True, check=True)
            wire = subprocess.check_output([str(binary)])
        self.assertEqual(len(wire), p.MAX_STATE_SIZE)
        state = p.State.decode(wire)
        np.testing.assert_equal(state.q, np.arange(29))
        body, hands = p.motor_telemetry(state.motor_telemetry)
        self.assertEqual((body['tick'], body['mode_pr']), (12345, 2))
        np.testing.assert_equal(body['mode'], np.arange(29) % 3)
        np.testing.assert_equal(body['ddq'], np.arange(29) + .25)
        np.testing.assert_equal(body['tau_est'], np.arange(29) + .5)
        np.testing.assert_equal(body['vol'], np.arange(29) + 48.)
        self.assertEqual(body['acc'], (1., 2., 3.))
        self.assertEqual(hands[0]['source'], 1)
        np.testing.assert_equal(hands[0]['tau_est'], np.arange(7) + .5)
        np.testing.assert_allclose(hands[0]['power'], [24., 1.5, 5., 3.3])
        self.assertEqual(hands[0]['press_sensor_count'], 1)
        np.testing.assert_equal(hands[0]['pressure'][:12], np.arange(1, 13))
        self.assertTrue(np.isnan(hands[0]['pressure'][12:]).all())
        self.assertEqual(hands[1], {})
        legacy = p.State.decode(wire[:p.STATE.size])
        np.testing.assert_equal(legacy.q, state.q)
        self.assertEqual(p.motor_telemetry(legacy.motor_telemetry), ({}, ({}, {})))
        # Optional recording fields must not turn unavailable/NaN data into a
        # control fault. Existing control-field validation still runs.
        damaged = wire[:p.STATE.size] + bytes(p.MOTOR_TELEMETRY.size)
        np.testing.assert_equal(p.State.decode(damaged).q, state.q)
        self.assertEqual(p.motor_telemetry(p.State.decode(damaged).motor_telemetry), ({}, ({}, {})))


if __name__ == '__main__':
    unittest.main()
