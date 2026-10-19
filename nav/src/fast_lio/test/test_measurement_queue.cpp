#include <gtest/gtest.h>
#include "measurement_queue.hpp"

TEST(MeasurementQueue, BurstSkipsObsoleteCloudsWithoutMismatchingTimestamps)
{
  MeasurementQueue<int> scans(3, .25);
  for (int frame = 0; frame < 1000; ++frame) scans.push(frame*.02, frame);
  ASSERT_EQ(scans.size(), 3U);
  for (int frame = 997; frame < 1000; ++frame) {
    EXPECT_EQ(scans.front().value, frame);
    EXPECT_DOUBLE_EQ(scans.front().stamp, frame*.02);
    scans.pop_front();
  }
  EXPECT_TRUE(scans.empty());
  EXPECT_FALSE(scans.push(19., -1).accepted);  // Still reject old data after drain.
  EXPECT_TRUE(scans.push(20., 1000).accepted);
}

TEST(MeasurementQueue, DuplicateOrReorderedDataCannotRewindEstimator)
{
  MeasurementQueue<int> scans(3, .25);
  scans.push(10., 1);
  scans.push(10.1, 2);
  EXPECT_FALSE(scans.push(10.1, 3).accepted);
  EXPECT_FALSE(scans.push(1., 4).accepted);
  EXPECT_FALSE(scans.push(std::numeric_limits<double>::quiet_NaN(), 5).accepted);
  EXPECT_EQ(scans.front().value, 1);
  scans.pop_front();
  EXPECT_DOUBLE_EQ(scans.front().stamp, 10.1);
  EXPECT_EQ(scans.front().value, 2);
}

TEST(MeasurementQueue, ScanSkippingRetainsIntermediateImuUntilExplicitOverflow)
{
  MeasurementQueue<int> imu(4000, 4.);
  for (int sample = 0; sample < 400; ++sample)
    EXPECT_EQ(imu.push(sample*.005, sample).dropped, 0U);
  // A two-second LiDAR interruption retains the whole integration interval.
  ASSERT_EQ(imu.size(), 400U);
  EXPECT_EQ(imu.front().value, 0);
  EXPECT_GT(imu.push(4.1, 820).dropped, 0U);
  EXPECT_LE(imu.size(), 4000U);
  EXPECT_GE(imu.front().stamp, .1);
}
