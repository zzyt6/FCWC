// Copyright (c) 2021 Samsung Research America
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

#ifndef NAV2_ROTATION_SHIM_CONTROLLER__NAV2_ROTATION_SHIM_CONTROLLER_HPP_
#define NAV2_ROTATION_SHIM_CONTROLLER__NAV2_ROTATION_SHIM_CONTROLLER_HPP_

#include <algorithm>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "angles/angles.h"
#include "nav2_core/controller.hpp"
#include "nav2_core/exceptions.hpp"
#include "nav2_costmap_2d/footprint_collision_checker.hpp"
#include "nav2_util/geometry_utils.hpp"
#include "nav2_util/node_utils.hpp"
#include "nav2_util/robot_utils.hpp"
#include "pluginlib/class_list_macros.hpp"
#include "pluginlib/class_loader.hpp"
#include "rclcpp/rclcpp.hpp"

namespace nav2_rotation_shim_controller
{

/**
 * @class nav2_rotation_shim_controller::RotationShimController
 * @brief Rotate to rough path heading controller shim plugin
 */
class RotationShimController : public nav2_core::Controller
{
public:
  /** @brief Constructor. */
  RotationShimController();

  /** @brief Destructor. */
  ~RotationShimController() override = default;

  /** @brief Configure controller state machine. */
  void configure(
    const rclcpp_lifecycle::LifecycleNode::SharedPtr & parent,
    std::string name, const std::shared_ptr<tf2_ros::Buffer> & tf,
    const std::shared_ptr<nav2_costmap_2d::Costmap2DROS> & costmap_ros) override;

  void cleanup() override;
  void activate() override;
  void deactivate() override;

  /** @brief Compute the best command given the current pose and velocity. */
  geometry_msgs::msg::TwistStamped computeVelocityCommands(
    const geometry_msgs::msg::PoseStamped & pose,
    const geometry_msgs::msg::Twist & velocity) override;

  /** @brief Set the global plan. */
  void setPlan(const nav_msgs::msg::Path & path) override;

protected:
  geometry_msgs::msg::PoseStamped getSampledPathPt();

  geometry_msgs::msg::Pose transformPoseToBaseFrame(
    const geometry_msgs::msg::PoseStamped & pt);

  geometry_msgs::msg::TwistStamped computeRotateToHeadingCommand(
    const double & angular_distance,
    const geometry_msgs::msg::PoseStamped & pose,
    const geometry_msgs::msg::Twist & velocity);

  void isCollisionFree(
    const geometry_msgs::msg::TwistStamped & cmd_vel,
    const double & angular_distance_to_heading,
    const geometry_msgs::msg::PoseStamped & pose);

  rcl_interfaces::msg::SetParametersResult dynamicParametersCallback(
    std::vector<rclcpp::Parameter> parameters);

  rclcpp_lifecycle::LifecycleNode::WeakPtr node_;
  std::shared_ptr<tf2_ros::Buffer> tf_;
  std::string plugin_name_;
  rclcpp::Logger logger_ {rclcpp::get_logger("RotationShimController")};
  rclcpp::Clock::SharedPtr clock_;
  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros_;
  std::unique_ptr<
    nav2_costmap_2d::FootprintCollisionChecker<nav2_costmap_2d::Costmap2D *>>
  collision_checker_;

  pluginlib::ClassLoader<nav2_core::Controller> lp_loader_;
  nav2_core::Controller::Ptr primary_controller_;
  bool path_updated_;
  nav_msgs::msg::Path current_path_;
  double forward_sampling_distance_, angular_dist_threshold_;
  double rotate_to_heading_angular_vel_, max_angular_accel_;
  double control_duration_, simulate_ahead_time_;
  bool closed_loop_;
  double last_angular_vel_;

  std::mutex mutex_;
  rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr dyn_params_handler_;
};

}  // namespace nav2_rotation_shim_controller

#endif  // NAV2_ROTATION_SHIM_CONTROLLER__NAV2_ROTATION_SHIM_CONTROLLER_HPP_
