#include "gtest/gtest.h"
#include "unitree_g1_kinematics/state_timing.hpp"

using unitree_g1_kinematics::JointHistory;
using unitree_g1_kinematics::LowStatePacket;
using unitree_g1_kinematics::SensorClock;

TEST(StateTiming, AcquisitionClockDoesNotFollowMappingCallbackJitter)
{
  SensorClock clock;
  // Real deployment has an approximately 62 second clock offset. IMU transport
  // occasionally stalls, while mapping adds a separate variable 30-180 ms.
  for (int i = 0; i < 200; ++i) {
    const double sensor = 100. + i * .005;
    clock.observe(sensor, sensor + 62. + (i % 7 == 0 ? .02 : 0.));
  }
  for (const double delay : {.03, .05, .18}) {
    const auto stamp = clock.hostTime(100.9, 162. + .9 + delay);
    ASSERT_TRUE(stamp);
    EXPECT_NEAR(*stamp, 162.9, 1e-9);
  }
  EXPECT_FALSE(clock.hostTime(100.9, 164.));
  clock.observe(1., 164.);  // Device clock restarted.
  ASSERT_TRUE(clock.hostTime(1., 164.01));
  EXPECT_NEAR(*clock.hostTime(1., 164.01), 164., 1e-9);
}

TEST(StateTiming, WaistAtAcquisitionTimeCancelsTorsoRotation)
{
  JointHistory history;
  for (int i = 0; i <= 100; ++i) {
    LowStatePacket packet;
    packet.sample_monotonic = 10. + .002 * i;
    packet.q[12] = .8 * (packet.sample_monotonic - 10.);
    history.add(packet);
  }
  const double captured = 10.131;
  const double torso_yaw = .8 * (captured - 10.);
  auto matched = history.at(captured);
  ASSERT_TRUE(matched);
  EXPECT_NEAR(torso_yaw - matched->q[12], 0., 1e-12);
  EXPECT_GT(std::abs(torso_yaw - history.at(10.2)->q[12]), .05);
  EXPECT_FALSE(history.at(9.9));
  EXPECT_FALSE(history.at(10.3));
}

TEST(StateTiming, JointHistoryRejectsGapsAndModeChanges)
{
  JointHistory history;
  LowStatePacket packet;
  packet.sample_monotonic = 1.; history.add(packet);
  packet.sample_monotonic = 1.2; history.add(packet);
  EXPECT_FALSE(history.at(1.1));
  packet.sample_monotonic = 1.21; packet.mode_machine = 4; history.add(packet);
  EXPECT_FALSE(history.at(1.2));
  ASSERT_TRUE(history.at(1.21));
}

TEST(StateTiming, ReorderedImuCannotMakeOldDataFresh)
{
  SensorClock clock;
  clock.observe(10., 100.);
  clock.observe(10.1, 100.1);
  clock.observe(9.95, 100.12);  // Late delivery must not move the offset by 170 ms.
  EXPECT_NEAR(*clock.hostTime(10.05, 100.13), 100.05, 1e-9);
  clock.observe(10.1, 100.35);  // Replayed data cannot renew the clock watchdog.
  EXPECT_FALSE(clock.hostTime(10.1, 100.41));
  clock.observe(10.1, 102.);  // Nor can replay be mistaken for a clock restart.
  clock.observe(10.2, 102.);  // A newer but two-second-delayed packet is still old.
  EXPECT_FALSE(clock.hostTime(10.2, 102.));
  clock.observe(12.01, 102.01);
  ASSERT_TRUE(clock.hostTime(12.01, 102.02));
  EXPECT_NEAR(*clock.hostTime(12.01, 102.02), 102.01, 1e-9);
}
