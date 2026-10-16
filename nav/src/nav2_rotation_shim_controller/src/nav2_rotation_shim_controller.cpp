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

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "nav2_rotation_shim_controller/nav2_rotation_shim_controller.hpp"

using rcl_interfaces::msg::ParameterType;

namespace nav2_rotation_shim_controller
{

RotationShimController::RotationShimController()
: lp_loader_("nav2_core", "nav2_core::Controller"),
  primary_controller_(nullptr),
  path_updated_(false),
  closed_loop_(true),
  last_angular_vel_(std::numeric_limits<double>::max())
{
}

void RotationShimController::configure(
  const rclcpp_lifecycle::LifecycleNode::SharedPtr & parent,
  std::string name, const std::shared_ptr<tf2_ros::Buffer> & tf,
  const std::shared_ptr<nav2_costmap_2d::Costmap2DROS> & costmap_ros)
{
  plugin_name_ = name;
  node_ = parent;
  auto node = parent;

  tf_ = tf;
  costmap_ros_ = costmap_ros;
  logger_ = node->get_logger();
  clock_ = node->get_clock();

  std::string primary_controller;
  double control_frequency;
  nav2_util::declare_parameter_if_not_declared(
    node, plugin_name_ + ".angular_dist_threshold", rclcpp::ParameterValue(0.785));
  nav2_util::declare_parameter_if_not_declared(
    node, plugin_name_ + ".forward_sampling_distance", rclcpp::ParameterValue(0.5));
  nav2_util::declare_parameter_if_not_declared(
    node, plugin_name_ + ".rotate_to_heading_angular_vel", rclcpp::ParameterValue(1.8));
  nav2_util::declare_parameter_if_not_declared(
    node, plugin_name_ + ".max_angular_accel", rclcpp::ParameterValue(3.2));
  nav2_util::declare_parameter_if_not_declared(
    node, plugin_name_ + ".simulate_ahead_time", rclcpp::ParameterValue(1.0));
  nav2_util::declare_parameter_if_not_declared(
    node, plugin_name_ + ".closed_loop", rclcpp::ParameterValue(true));
  nav2_util::declare_parameter_if_not_declared(
    node, plugin_name_ + ".primary_controller", rclcpp::ParameterValue(std::string("")));

  node->get_parameter(plugin_name_ + ".angular_dist_threshold", angular_dist_threshold_);
  node->get_parameter(plugin_name_ + ".forward_sampling_distance", forward_sampling_distance_);
  node->get_parameter(
    plugin_name_ + ".rotate_to_heading_angular_vel", rotate_to_heading_angular_vel_);
  node->get_parameter(plugin_name_ + ".max_angular_accel", max_angular_accel_);
  node->get_parameter(plugin_name_ + ".simulate_ahead_time", simulate_ahead_time_);
  node->get_parameter(plugin_name_ + ".closed_loop", closed_loop_);

  primary_controller = node->get_parameter(plugin_name_ + ".primary_controller").as_string();
  node->get_parameter("controller_frequency", control_frequency);
  control_duration_ = 1.0 / control_frequency;

  try {
    primary_controller_ = lp_loader_.createUniqueInstance(primary_controller);
    RCLCPP_INFO(
      logger_, "Created internal controller for rotation shimming: %s of type %s",
      plugin_name_.c_str(), primary_controller.c_str());
  } catch (const pluginlib::PluginlibException & ex) {
    RCLCPP_FATAL(
      logger_,
      "Failed to create internal controller for rotation shimming. Exception: %s", ex.what());
    return;
  }

  primary_controller_->configure(parent, name, tf, costmap_ros);

  RCLCPP_INFO(
    logger_, "Rotation shim acceleration feedback: %s",
    closed_loop_ ? "odometry (closed loop)" : "last command (open loop)");

  collision_checker_ = std::make_unique<nav2_costmap_2d::
      FootprintCollisionChecker<nav2_costmap_2d::Costmap2D *>>(costmap_ros->getCostmap());
}

void RotationShimController::activate()
{
  RCLCPP_INFO(
    logger_,
    "Activating controller: %s of type nav2_rotation_shim_controller::RotationShimController",
    plugin_name_.c_str());

  primary_controller_->activate();
  last_angular_vel_ = std::numeric_limits<double>::max();

  auto node = node_.lock();
  dyn_params_handler_ = node->add_on_set_parameters_callback(
    std::bind(
      &RotationShimController::dynamicParametersCallback,
      this, std::placeholders::_1));
}

void RotationShimController::deactivate()
{
  RCLCPP_INFO(
    logger_,
    "Deactivating controller: %s of type nav2_rotation_shim_controller::RotationShimController",
    plugin_name_.c_str());

  primary_controller_->deactivate();
  dyn_params_handler_.reset();
}

void RotationShimController::cleanup()
{
  RCLCPP_INFO(
    logger_,
    "Cleaning up controller: %s of type nav2_rotation_shim_controller::RotationShimController",
    plugin_name_.c_str());

  primary_controller_->cleanup();
  primary_controller_.reset();
}

geometry_msgs::msg::TwistStamped RotationShimController::computeVelocityCommands(
  const geometry_msgs::msg::PoseStamped & pose,
  const geometry_msgs::msg::Twist & velocity)
{
  {
    std::lock_guard<std::mutex> lock_reinit(mutex_);
    if (path_updated_) {
      try {
        geometry_msgs::msg::Pose sampled_pt_base = transformPoseToBaseFrame(getSampledPathPt());

        double angular_distance_to_heading =
          std::atan2(sampled_pt_base.position.y, sampled_pt_base.position.x);
        if (std::fabs(angular_distance_to_heading) > angular_dist_threshold_) {
          RCLCPP_DEBUG(
            logger_, "Robot is not within the new path's rough heading, rotating to heading...");
          auto cmd_vel = computeRotateToHeadingCommand(
            angular_distance_to_heading, pose, velocity);
          last_angular_vel_ = cmd_vel.twist.angular.z;
          return cmd_vel;
        } else {
          RCLCPP_DEBUG(logger_, "Robot is at the new path's rough heading, passing to controller");
          path_updated_ = false;
        }
      } catch (const std::runtime_error & e) {
        RCLCPP_DEBUG(
          logger_,
          "Rotation Shim Controller was unable to find a sampling point,"
          " a rotational collision was detected, or TF failed to transform"
          " into base frame! what(): %s", e.what());
        path_updated_ = false;
      }
    }
  }

  auto cmd_vel = primary_controller_->computeVelocityCommands(pose, velocity);
  last_angular_vel_ = cmd_vel.twist.angular.z;
  return cmd_vel;
}

geometry_msgs::msg::PoseStamped RotationShimController::getSampledPathPt()
{
  if (current_path_.poses.size() < 2) {
    throw nav2_core::PlannerException(
            "Path is too short to find a valid sampled path point for rotation.");
  }

  geometry_msgs::msg::Pose start = current_path_.poses.front().pose;
  double dx, dy;

  for (unsigned int i = 1; i != current_path_.poses.size(); i++) {
    dx = current_path_.poses[i].pose.position.x - start.position.x;
    dy = current_path_.poses[i].pose.position.y - start.position.y;
    if (std::hypot(dx, dy) >= forward_sampling_distance_) {
      current_path_.poses[i].header.frame_id = current_path_.header.frame_id;
      // A zero timestamp asks TF for the latest complete transform chain. The
      // navigation TF is scan-driven and can be slower than the controller
      // loop, so stamping this point with "now" would wait for a future TF on
      // every cycle and make the controller miss its desired rate.
      current_path_.poses[i].header.stamp.sec = 0;
      current_path_.poses[i].header.stamp.nanosec = 0;
      return current_path_.poses[i];
    }
  }

  throw nav2_core::PlannerException(
          std::string(
            "Unable to find a sampling point at least %0.2f from the robot,"
            "passing off to primary controller plugin.", forward_sampling_distance_));
}

geometry_msgs::msg::Pose RotationShimController::transformPoseToBaseFrame(
  const geometry_msgs::msg::PoseStamped & pt)
{
  try {
    return tf_->transform(
      pt, costmap_ros_->getBaseFrameID(),
      tf2::durationFromSec(costmap_ros_->getTransformTolerance())).pose;
  } catch (const tf2::TransformException & ex) {
    throw nav2_core::PlannerException(
            std::string("Failed to transform pose to base frame: ") + ex.what());
  }
}

geometry_msgs::msg::TwistStamped RotationShimController::computeRotateToHeadingCommand(
  const double & angular_distance_to_heading,
  const geometry_msgs::msg::PoseStamped & pose,
  const geometry_msgs::msg::Twist & velocity)
{
  double current_angular_vel = closed_loop_ ? velocity.angular.z : last_angular_vel_;
  if (current_angular_vel == std::numeric_limits<double>::max()) {
    current_angular_vel = 0.0;
  }

  geometry_msgs::msg::TwistStamped cmd_vel;
  cmd_vel.header = pose.header;
  const double sign = angular_distance_to_heading > 0.0 ? 1.0 : -1.0;
  const double angular_vel = sign * rotate_to_heading_angular_vel_;
  const double & dt = control_duration_;
  const double min_feasible_angular_speed = current_angular_vel - max_angular_accel_ * dt;
  const double max_feasible_angular_speed = current_angular_vel + max_angular_accel_ * dt;
  cmd_vel.twist.angular.z =
    std::clamp(angular_vel, min_feasible_angular_speed, max_feasible_angular_speed);

  // Official newer Nav2 behavior: reduce angular velocity when required to
  // stop without overshooting the requested heading.
  const double max_vel_to_stop = std::sqrt(
    2.0 * max_angular_accel_ * std::fabs(angular_distance_to_heading));
  if (std::fabs(cmd_vel.twist.angular.z) > max_vel_to_stop) {
    cmd_vel.twist.angular.z = sign * max_vel_to_stop;
  }

  isCollisionFree(cmd_vel, angular_distance_to_heading, pose);
  return cmd_vel;
}

void RotationShimController::isCollisionFree(
  const geometry_msgs::msg::TwistStamped & cmd_vel,
  const double & angular_distance_to_heading,
  const geometry_msgs::msg::PoseStamped & pose)
{
  double simulated_time = 0.0;
  double initial_yaw = tf2::getYaw(pose.pose.orientation);
  double yaw = 0.0;
  double footprint_cost = 0.0;
  double remaining_rotation_before_thresh =
    std::fabs(angular_distance_to_heading) - angular_dist_threshold_;

  while (simulated_time < simulate_ahead_time_) {
    simulated_time += control_duration_;
    yaw = initial_yaw + cmd_vel.twist.angular.z * simulated_time;

    if (angles::shortest_angular_distance(yaw, initial_yaw) >=
      remaining_rotation_before_thresh)
    {
      break;
    }

    using namespace nav2_costmap_2d;  // NOLINT
    footprint_cost = collision_checker_->footprintCostAtPose(
      pose.pose.position.x, pose.pose.position.y,
      yaw, costmap_ros_->getRobotFootprint());

    if (footprint_cost == static_cast<double>(NO_INFORMATION) &&
      costmap_ros_->getLayeredCostmap()->isTrackingUnknown())
    {
      throw std::runtime_error("RotationShimController detected a potential collision ahead!");
    }

    if (footprint_cost >= static_cast<double>(LETHAL_OBSTACLE)) {
      throw std::runtime_error("RotationShimController detected collision ahead!");
    }
  }
}

void RotationShimController::setPlan(const nav_msgs::msg::Path & path)
{
  {
    std::lock_guard<std::mutex> lock_reinit(mutex_);
    path_updated_ = true;
    current_path_ = path;
  }
  primary_controller_->setPlan(path);
}

rcl_interfaces::msg::SetParametersResult RotationShimController::dynamicParametersCallback(
  std::vector<rclcpp::Parameter> parameters)
{
  rcl_interfaces::msg::SetParametersResult result;
  std::lock_guard<std::mutex> lock_reinit(mutex_);

  for (auto parameter : parameters) {
    const auto & type = parameter.get_type();
    const auto & name = parameter.get_name();

    if (type == ParameterType::PARAMETER_DOUBLE) {
      if (name == plugin_name_ + ".angular_dist_threshold") {
        angular_dist_threshold_ = parameter.as_double();
      } else if (name == plugin_name_ + ".forward_sampling_distance") {
        forward_sampling_distance_ = parameter.as_double();
      } else if (name == plugin_name_ + ".rotate_to_heading_angular_vel") {
        rotate_to_heading_angular_vel_ = parameter.as_double();
      } else if (name == plugin_name_ + ".max_angular_accel") {
        max_angular_accel_ = parameter.as_double();
      } else if (name == plugin_name_ + ".simulate_ahead_time") {
        simulate_ahead_time_ = parameter.as_double();
      }
    } else if (type == ParameterType::PARAMETER_BOOL) {
      if (name == plugin_name_ + ".closed_loop") {
        closed_loop_ = parameter.as_bool();
      }
    }
  }

  result.successful = true;
  return result;
}

}  // namespace nav2_rotation_shim_controller

PLUGINLIB_EXPORT_CLASS(
  nav2_rotation_shim_controller::RotationShimController,
  nav2_core::Controller)
