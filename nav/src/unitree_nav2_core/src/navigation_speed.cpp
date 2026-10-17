#include "unitree_nav2_core/navigation_speed.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <stdexcept>
#include <tuple>

#include "nav2_util/node_utils.hpp"
#include "nav2_core/exceptions.hpp"
#include "nav_2d_utils/tf_help.hpp"
#include "pluginlib/class_list_macros.hpp"

namespace unitree_nav2_core
{
namespace
{
void declareProfile(const nav2_util::LifecycleNode::SharedPtr & node, const std::string & ns)
{
  nav2_util::declare_parameter_if_not_declared(
    node, ns + ".navigation_speed_mode", rclcpp::ParameterValue(std::string("auto")));
  nav2_util::declare_parameter_if_not_declared(
    node, ns + ".navigation_speed", rclcpp::ParameterValue(0.35));
}

rcl_interfaces::msg::SetParametersResult validateProfile(
  const std::vector<rclcpp::Parameter> & parameters, const std::string & ns)
{
  rcl_interfaces::msg::SetParametersResult result;
  result.successful = true;
  for (const auto & parameter : parameters) {
    if (parameter.get_name() == ns + ".navigation_speed") {
      if (parameter.get_type() != rclcpp::ParameterType::PARAMETER_DOUBLE ||
        !std::isfinite(parameter.as_double()) || parameter.as_double() < 0.20 ||
        parameter.as_double() > 0.60)
      {
        result.successful = false;
        result.reason = "Navigation speed must be between 0.20 and 0.60 m/s";
      }
    } else if (parameter.get_name() == ns + ".navigation_speed_mode") {
      if (parameter.get_type() != rclcpp::ParameterType::PARAMETER_STRING ||
        (parameter.as_string() != "auto" && parameter.as_string() != "fixed"))
      {
        result.successful = false;
        result.reason = "Navigation speed mode must be auto or fixed";
      }
    }
  }
  return result;
}
}  // namespace

void SpeedLimitedGenerator::initialize(
  const nav2_util::LifecycleNode::SharedPtr & node, const std::string & plugin_name)
{
  StandardTrajectoryGenerator::initialize(node, plugin_name);
  node_ = node;
  declareProfile(node, plugin_name);
  nav2_util::declare_parameter_if_not_declared(
    node, plugin_name + ".approach_distance", rclcpp::ParameterValue(0.6));
  nav2_util::declare_parameter_if_not_declared(
    node, plugin_name + ".approach_speed", rclcpp::ParameterValue(0.1));
  nav2_util::declare_parameter_if_not_declared(
    node, plugin_name + ".xy_goal_tolerance", rclcpp::ParameterValue(0.15));
  approach_distance_ = node->get_parameter(plugin_name + ".approach_distance").as_double();
  approach_speed_ = node->get_parameter(plugin_name + ".approach_speed").as_double();
  goal_tolerance_ = node->get_parameter(plugin_name + ".xy_goal_tolerance").as_double();
  if (!std::isfinite(approach_distance_) || !std::isfinite(approach_speed_) ||
    !std::isfinite(goal_tolerance_) || goal_tolerance_ <= 0.0 ||
    approach_distance_ <= goal_tolerance_ || approach_speed_ <= 0.05 || approach_speed_ > 0.2)
  {
    throw std::invalid_argument("Invalid goal approach distance, speed or tolerance");
  }
  auto initial = validateProfile(node->get_parameters({
    plugin_name + ".navigation_speed", plugin_name + ".navigation_speed_mode"}), plugin_name);
  if (!initial.successful) {throw std::invalid_argument(initial.reason);}
  parameter_callback_ = node->add_on_set_parameters_callback(
    [plugin_name](const std::vector<rclcpp::Parameter> & parameters) {
      return validateProfile(parameters, plugin_name);
    });
}

void SpeedLimitedGenerator::startNewIteration(const nav_2d_msgs::msg::Twist2D & velocity)
{
  const auto parameters = node_->get_parameters({
    plugin_name_ + ".navigation_speed", plugin_name_ + ".navigation_speed_mode"});
  const double navigation_speed = parameters[0].as_double();
  const double fraction = std::clamp(
    (goal_distance_ - goal_tolerance_) / (approach_distance_ - goal_tolerance_), 0.0, 1.0);
  const double speed = std::min(navigation_speed,
    approach_speed_ + fraction * (navigation_speed - approach_speed_));
  const bool fixed = parameters[1].as_string() == "fixed";
  StandardTrajectoryGenerator::startNewIteration(velocity);
  std::vector<nav_2d_msgs::msg::Twist2D> original;
  double min_x = std::numeric_limits<double>::max(), min_y = min_x;
  double max_x = -min_x, max_y = max_x;
  while (StandardTrajectoryGenerator::hasMoreTwists()) {
    const auto twist = StandardTrajectoryGenerator::nextTwist();
    original.push_back(twist);
    min_x = std::min(min_x, twist.x);
    max_x = std::max(max_x, twist.x);
    min_y = std::min(min_y, twist.y);
    max_y = std::max(max_y, twist.y);
  }
  candidates_.clear();
  index_ = 0;
  std::set<std::tuple<long long, long long, long long>> seen;
  const auto add = [&](const nav_2d_msgs::msg::Twist2D & twist) {
      const auto key = std::make_tuple(
        std::llround(twist.x * 1e7), std::llround(twist.y * 1e7),
        std::llround(twist.theta * 1e7));
      if (seen.insert(key).second) {candidates_.push_back(twist);}
    };
  for (auto twist : original) {
    const double norm = std::hypot(twist.x, twist.y);
    if (norm > 1e-9 && fixed) {
      auto cruise = twist;
      cruise.x *= speed / norm;
      cruise.y *= speed / norm;
      // Keep exact-speed samples inside the original reachable sampling window,
      // including reverse/lateral limits. DWB still predicts their acceleration.
      if (cruise.x >= min_x - 1e-9 && cruise.x <= max_x + 1e-9 &&
        cruise.y >= min_y - 1e-9 && cruise.y <= max_y + 1e-9)
      {
        add(cruise);
      }
    }
    if (norm > speed) {
      twist.x *= speed / norm;
      twist.y *= speed / norm;
    }
    // Preserve slower candidates and pure rotations for obstacles and arrival.
    add(twist);
  }
}

void SpeedLimitedGenerator::setGoalDistance(double distance)
{
  if (!std::isfinite(distance) || distance < 0.0) {
    throw nav2_core::PlannerException("Invalid goal distance for approach control");
  }
  goal_distance_ = distance;
}

geometry_msgs::msg::TwistStamped ApproachDWBLocalPlanner::computeVelocityCommands(
  const geometry_msgs::msg::PoseStamped & pose, const geometry_msgs::msg::Twist & velocity)
{
  auto * generator = dynamic_cast<SpeedLimitedGenerator *>(traj_generator_.get());
  if (!generator || global_plan_.poses.empty()) {
    throw nav2_core::PlannerException("Goal approach requires SpeedLimitedGenerator and a nonempty plan");
  }
  nav_2d_msgs::msg::Pose2DStamped goal, transformed_goal;
  goal.header = global_plan_.header;
  goal.header.stamp = pose.header.stamp;
  goal.pose = global_plan_.poses.back();
  if (!nav_2d_utils::transformPose(tf_, pose.header.frame_id, goal, transformed_goal,
    transform_tolerance_))
  {
    throw nav2_core::PlannerException("Cannot transform final goal for approach control");
  }
  generator->setGoalDistance(std::hypot(
    transformed_goal.pose.x - pose.pose.position.x,
    transformed_goal.pose.y - pose.pose.position.y));
  return DWBLocalPlanner::computeVelocityCommands(pose, velocity);
}

bool SpeedLimitedGenerator::hasMoreTwists() {return index_ < candidates_.size();}
nav_2d_msgs::msg::Twist2D SpeedLimitedGenerator::nextTwist()
{
  return candidates_.at(index_++);
}

void CruiseSpeedCritic::onInit() {declareProfile(nh_, dwb_plugin_name_);}

bool CruiseSpeedCritic::prepare(
  const geometry_msgs::msg::Pose2D & pose, const nav_2d_msgs::msg::Twist2D &,
  const geometry_msgs::msg::Pose2D & goal, const nav_2d_msgs::msg::Path2D &)
{
  const auto parameters = nh_->get_parameters({
    dwb_plugin_name_ + ".navigation_speed", dwb_plugin_name_ + ".navigation_speed_mode"});
  speed_ = parameters[0].as_double();
  // Release the cruise preference smoothly over the final 0.6 m. The regular
  // Goal/RotateToGoal critics retain control of braking and final orientation.
  const double distance = std::hypot(goal.x - pose.x, goal.y - pose.y);
  weight_ = parameters[1].as_string() == "fixed" ?
    std::clamp((distance - 0.15) / 0.45, 0.0, 1.0) : 0.0;
  return true;
}

double CruiseSpeedCritic::scoreTrajectory(const dwb_msgs::msg::Trajectory2D & trajectory)
{
  return weight_ * std::max(0.0, 1.0 -
         std::hypot(trajectory.velocity.x, trajectory.velocity.y) / speed_);
}

void CartLookaheadCritic::onInit()
{
  const auto prefix = dwb_plugin_name_ + "." + name_ + ".";
  for (const auto & item : std::vector<std::pair<std::string, double>>{
      {"lookahead_base", 0.4}, {"lookahead_gain", 0.6}, {"lookahead_max", 1.0}, {"heading_weight", 0.4}})
  {
    nav2_util::declare_parameter_if_not_declared(nh_, prefix + item.first, rclcpp::ParameterValue(item.second));
  }
  base_ = nh_->get_parameter(prefix + "lookahead_base").as_double();
  gain_ = nh_->get_parameter(prefix + "lookahead_gain").as_double();
  maximum_ = nh_->get_parameter(prefix + "lookahead_max").as_double();
  heading_weight_ = nh_->get_parameter(prefix + "heading_weight").as_double();
  if (!(base_ > 0 && gain_ >= 0 && maximum_ >= base_ && heading_weight_ >= 0)) {
    throw std::runtime_error("Invalid cart lookahead parameters");
  }
}

bool CartLookaheadCritic::prepare(
  const geometry_msgs::msg::Pose2D & pose, const nav_2d_msgs::msg::Twist2D & velocity,
  const geometry_msgs::msg::Pose2D & goal, const nav_2d_msgs::msg::Path2D & plan)
{
  if (plan.poses.empty()) {return false;}
  size_t nearest = 0;
  double best = std::numeric_limits<double>::infinity();
  for (size_t i = 0; i < plan.poses.size(); ++i) {
    const double distance = std::hypot(plan.poses[i].x-pose.x, plan.poses[i].y-pose.y);
    if (distance < best) {best = distance; nearest = i;}
  }
  double remaining = std::min(maximum_, base_+gain_*std::hypot(velocity.x, velocity.y));
  target_ = plan.poses.back();
  for (size_t i = nearest; i+1 < plan.poses.size(); ++i) {
    const auto & a = plan.poses[i];
    const auto & b = plan.poses[i+1];
    const double length = std::hypot(b.x-a.x, b.y-a.y);
    if (length > 1e-6 && length >= remaining) {
      target_.x = a.x+(b.x-a.x)*remaining/length;
      target_.y = a.y+(b.y-a.y)*remaining/length;
      target_.theta = std::atan2(b.y-a.y, b.x-a.x);
      return true;
    }
    remaining -= length;
  }
  target_.theta = goal.theta;
  return true;
}

double CartLookaheadCritic::scoreTrajectory(const dwb_msgs::msg::Trajectory2D & trajectory)
{
  if (trajectory.poses.empty()) {return 0.;}
  const auto & end = trajectory.poses.back();
  const double angle = std::atan2(std::sin(end.theta-target_.theta), std::cos(end.theta-target_.theta));
  return std::hypot(end.x-target_.x, end.y-target_.y)+heading_weight_*std::abs(angle);
}
}  // namespace unitree_nav2_core

PLUGINLIB_EXPORT_CLASS(unitree_nav2_core::SpeedLimitedGenerator, dwb_core::TrajectoryGenerator)
PLUGINLIB_EXPORT_CLASS(unitree_nav2_core::CruiseSpeedCritic, dwb_core::TrajectoryCritic)
PLUGINLIB_EXPORT_CLASS(unitree_nav2_core::ApproachDWBLocalPlanner, nav2_core::Controller)
PLUGINLIB_EXPORT_CLASS(unitree_nav2_core::CartLookaheadCritic, dwb_core::TrajectoryCritic)
