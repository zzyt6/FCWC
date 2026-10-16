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

#ifndef NAV2_CONTROLLER__PLUGINS__POSE_PROGRESS_CHECKER_HPP_
#define NAV2_CONTROLLER__PLUGINS__POSE_PROGRESS_CHECKER_HPP_

#include <string>
#include <vector>

#include "geometry_msgs/msg/pose2_d.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "nav2_controller/plugins/simple_progress_checker.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_lifecycle/lifecycle_node.hpp"

namespace nav2_controller
{

/**
 * @class nav2_controller::PoseProgressChecker
 * @brief Check translation and rotation to determine whether the robot is progressing.
 */
class PoseProgressChecker : public SimpleProgressChecker
{
public:
  void initialize(
    const rclcpp_lifecycle::LifecycleNode::SharedPtr & parent,
    const std::string & plugin_name) override;

  bool check(geometry_msgs::msg::PoseStamped & current_pose) override;

protected:
  bool isRobotMovedEnough(const geometry_msgs::msg::Pose2D & pose);

  static double poseAngleDistance(
    const geometry_msgs::msg::Pose2D & pose1,
    const geometry_msgs::msg::Pose2D & pose2);

  double required_movement_angle_{0.5};
  rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr dyn_params_handler_;
  std::string plugin_name_;

  rcl_interfaces::msg::SetParametersResult dynamicParametersCallback(
    std::vector<rclcpp::Parameter> parameters);
};

}  // namespace nav2_controller

#endif  // NAV2_CONTROLLER__PLUGINS__POSE_PROGRESS_CHECKER_HPP_
