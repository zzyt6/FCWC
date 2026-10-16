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

#include "nav2_controller/plugins/pose_progress_checker.hpp"

#include <cmath>
#include <functional>
#include <string>
#include <vector>

#include "angles/angles.h"
#include "nav_2d_utils/conversions.hpp"
#include "nav2_core/progress_checker.hpp"
#include "nav2_util/node_utils.hpp"
#include "pluginlib/class_list_macros.hpp"

using rcl_interfaces::msg::ParameterType;
using std::placeholders::_1;

namespace nav2_controller
{

void PoseProgressChecker::initialize(
  const rclcpp_lifecycle::LifecycleNode::SharedPtr & parent,
  const std::string & plugin_name)
{
  plugin_name_ = plugin_name;
  SimpleProgressChecker::initialize(parent, plugin_name);

  nav2_util::declare_parameter_if_not_declared(
    parent, plugin_name + ".required_movement_angle", rclcpp::ParameterValue(0.5));
  parent->get_parameter_or(
    plugin_name + ".required_movement_angle", required_movement_angle_, 0.5);

  dyn_params_handler_ = parent->add_on_set_parameters_callback(
    std::bind(&PoseProgressChecker::dynamicParametersCallback, this, _1));
}

bool PoseProgressChecker::check(geometry_msgs::msg::PoseStamped & current_pose)
{
  const geometry_msgs::msg::Pose2D current_pose2d =
    nav_2d_utils::poseToPose2D(current_pose.pose);
  if (!baseline_pose_set_ || isRobotMovedEnough(current_pose2d)) {
    reset_baseline_pose(current_pose2d);
    return true;
  }
  return !((nh_->now() - baseline_time_) > time_allowance_);
}

bool PoseProgressChecker::isRobotMovedEnough(const geometry_msgs::msg::Pose2D & pose)
{
  const double dx = pose.x - baseline_pose_.x;
  const double dy = pose.y - baseline_pose_.y;
  return std::hypot(dx, dy) > radius_ ||
         poseAngleDistance(pose, baseline_pose_) > required_movement_angle_;
}

double PoseProgressChecker::poseAngleDistance(
  const geometry_msgs::msg::Pose2D & pose1,
  const geometry_msgs::msg::Pose2D & pose2)
{
  return std::abs(angles::shortest_angular_distance(pose1.theta, pose2.theta));
}

rcl_interfaces::msg::SetParametersResult PoseProgressChecker::dynamicParametersCallback(
  std::vector<rclcpp::Parameter> parameters)
{
  rcl_interfaces::msg::SetParametersResult result;
  result.successful = true;
  for (const auto & parameter : parameters) {
    if (parameter.get_name() != plugin_name_ + ".required_movement_angle") {
      continue;
    }
    if (parameter.get_type() != ParameterType::PARAMETER_DOUBLE ||
      !std::isfinite(parameter.as_double()) || parameter.as_double() <= 0.0)
    {
      result.successful = false;
      result.reason = parameter.get_name() + " must be a positive double";
      return result;
    }
    required_movement_angle_ = parameter.as_double();
  }
  return result;
}

}  // namespace nav2_controller

PLUGINLIB_EXPORT_CLASS(nav2_controller::PoseProgressChecker, nav2_core::ProgressChecker)
