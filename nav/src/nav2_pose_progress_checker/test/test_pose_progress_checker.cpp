// Copyright (c) 2023 Dexory
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <gtest/gtest.h>

#include "geometry_msgs/msg/pose2_d.hpp"
#include "nav2_controller/plugins/pose_progress_checker.hpp"

namespace
{

class TestablePoseProgressChecker : public nav2_controller::PoseProgressChecker
{
public:
  void setThresholds(const double distance, const double angle)
  {
    radius_ = distance;
    required_movement_angle_ = angle;
  }

  void setBaseline(const geometry_msgs::msg::Pose2D & pose)
  {
    baseline_pose_ = pose;
  }

  bool movedEnough(const geometry_msgs::msg::Pose2D & pose)
  {
    return isRobotMovedEnough(pose);
  }
};

TEST(PoseProgressChecker, CountsTranslationOrRotationAsProgress)
{
  TestablePoseProgressChecker checker;
  checker.setThresholds(0.20, 0.20);

  geometry_msgs::msg::Pose2D baseline;
  checker.setBaseline(baseline);

  geometry_msgs::msg::Pose2D insufficient;
  insufficient.x = 0.10;
  insufficient.theta = 0.10;
  EXPECT_FALSE(checker.movedEnough(insufficient));

  geometry_msgs::msg::Pose2D translated;
  translated.x = 0.21;
  EXPECT_TRUE(checker.movedEnough(translated));

  geometry_msgs::msg::Pose2D rotated;
  rotated.theta = 0.21;
  EXPECT_TRUE(checker.movedEnough(rotated));
}

TEST(PoseProgressChecker, UsesShortestAngularDistance)
{
  TestablePoseProgressChecker checker;
  checker.setThresholds(0.20, 0.20);

  geometry_msgs::msg::Pose2D baseline;
  baseline.theta = 3.10;
  checker.setBaseline(baseline);

  geometry_msgs::msg::Pose2D wrapped;
  wrapped.theta = -3.10;
  EXPECT_FALSE(checker.movedEnough(wrapped));
}

}  // namespace
