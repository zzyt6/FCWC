#include "unitree_nav2_core/navigation_speed.hpp"
#include "unitree_nav2_core/navigation_profile.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <tuple>

#include "nav2_util/node_utils.hpp"
#include "nav2_core/exceptions.hpp"
#include "nav_2d_utils/tf_help.hpp"
#include "pluginlib/class_list_macros.hpp"

namespace unitree_nav2_core
{
void SpeedLimitedGenerator::initialize(
  const nav2_util::LifecycleNode::SharedPtr & node, const std::string & plugin_name)
{
  StandardTrajectoryGenerator::initialize(node, plugin_name);
  node_ = node;
  declareNavigationProfile(node, plugin_name);
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
  auto initial = validateNavigationProfile(node->get_parameters({
    plugin_name + ".navigation_speed", plugin_name + ".navigation_speed_mode"}), plugin_name);
  if (!initial.successful) {throw std::invalid_argument(initial.reason);}
  parameter_callback_ = node->add_on_set_parameters_callback(
    [plugin_name](const std::vector<rclcpp::Parameter> & parameters) {
      return validateNavigationProfile(parameters, plugin_name);
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
  const auto add = [&](nav_2d_msgs::msg::Twist2D twist) {
      // The stock iterator can return +/- 1e-17 for its zero sample. Those
      // residual signs must not arm OscillationCritic's direction locks.
      if (std::abs(twist.x) < 1e-7) {twist.x = 0.;}
      if (std::abs(twist.y) < 1e-7) {twist.y = 0.;}
      if (std::abs(twist.theta) < 1e-7) {twist.theta = 0.;}
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

dwb_msgs::msg::Trajectory2D SpeedLimitedGenerator::generateTrajectory(
  const geometry_msgs::msg::Pose2D & start_pose,
  const nav_2d_msgs::msg::Twist2D & start_vel,
  const nav_2d_msgs::msg::Twist2D & cmd_vel)
{
  rollout_start_velocity_ = start_vel;
  return StandardTrajectoryGenerator::generateTrajectory(start_pose, start_vel, cmd_vel);
}

std::vector<double> SpeedLimitedGenerator::getTimeSteps(
  const nav_2d_msgs::msg::Twist2D & cmd_vel)
{
  // Stock spatial sampling only uses the target speed. A stop target can then
  // put the whole braking interval into one integration step, skipping travel
  // and collisions before the vehicle stops. Bound spacing by both velocities.
  auto sampling_velocity = cmd_vel;
  sampling_velocity.x = std::max(std::abs(cmd_vel.x), std::abs(rollout_start_velocity_.x));
  sampling_velocity.y = std::max(std::abs(cmd_vel.y), std::abs(rollout_start_velocity_.y));
  sampling_velocity.theta = std::max(std::abs(cmd_vel.theta), std::abs(rollout_start_velocity_.theta));
  return StandardTrajectoryGenerator::getTimeSteps(sampling_velocity);
}

void ApproachDWBLocalPlanner::configure(
  const rclcpp_lifecycle::LifecycleNode::SharedPtr & node, std::string name,
  const std::shared_ptr<tf2_ros::Buffer> & tf,
  const std::shared_ptr<nav2_costmap_2d::Costmap2DROS> & costmap_ros)
{
  DWBLocalPlanner::configure(node, name, tf, costmap_ros);
  nav2_util::declare_parameter_if_not_declared(
    node, name + ".decision_log_period", rclcpp::ParameterValue(0.));
  decision_log_period_ = node->get_parameter(name + ".decision_log_period").as_double();
  if (!std::isfinite(decision_log_period_) || decision_log_period_ < 0.) {
    throw std::invalid_argument("decision_log_period must be finite and nonnegative");
  }
}

dwb_msgs::msg::TrajectoryScore ApproachDWBLocalPlanner::coreScoringAlgorithm(
  const geometry_msgs::msg::Pose2D & pose, const nav_2d_msgs::msg::Twist2D velocity,
  std::shared_ptr<dwb_msgs::msg::LocalPlanEvaluation> & results)
{
  const auto started = std::chrono::steady_clock::now();
  auto best = DWBLocalPlanner::coreScoringAlgorithm(pose, velocity, results);
  const auto finished = std::chrono::steady_clock::now();
  if (decision_log_period_ > 0. &&
    std::chrono::duration<double>(finished - decision_logged_).count() >= decision_log_period_)
  {
    decision_logged_ = finished;
    const auto & command = best.traj.velocity;
    const auto & end = best.traj.poses.empty() ? pose : best.traj.poses.back();
    std::ostringstream trace;
    trace << "DWB decision: {\"controller\":\"" << dwb_plugin_name_ << "\",\"pose\":["
          << pose.x << ',' << pose.y << ',' << pose.theta << "],\"velocity\":["
          << velocity.x << ',' << velocity.y << ',' << velocity.theta << "],\"command\":["
          << command.x << ',' << command.y << ',' << command.theta << "],\"endpoint\":["
          << end.x << ',' << end.y << ',' << end.theta << "],\"scoring_ms\":"
          << std::chrono::duration<double, std::milli>(finished - started).count()
          << ",\"total\":" << best.total << ",\"costs\":{";
    for (size_t i = 0; i < best.scores.size(); ++i) {
      if (i) {trace << ',';}
      const auto & score = best.scores[i];
      trace << '"' << score.name << "\":" << score.raw_score * score.scale;
    }
    trace << "}}";
    // Only the winning trajectory is serialized, at a bounded rate. Full DWB
    // evaluation remains disabled so diagnostics do not allocate every rollout.
    RCLCPP_INFO(node_->get_logger(), "%s", trace.str().c_str());
  }
  return best;
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

void CruiseSpeedCritic::onInit() {declareNavigationProfile(nh_, dwb_plugin_name_);}

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

}  // namespace unitree_nav2_core

PLUGINLIB_EXPORT_CLASS(unitree_nav2_core::SpeedLimitedGenerator, dwb_core::TrajectoryGenerator)
PLUGINLIB_EXPORT_CLASS(unitree_nav2_core::CruiseSpeedCritic, dwb_core::TrajectoryCritic)
PLUGINLIB_EXPORT_CLASS(unitree_nav2_core::ApproachDWBLocalPlanner, nav2_core::Controller)
