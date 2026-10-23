#ifndef UNITREE_NAV2_CORE__NAVIGATION_SPEED_HPP_
#define UNITREE_NAV2_CORE__NAVIGATION_SPEED_HPP_

#include <string>
#include <vector>
#include <limits>
#include <chrono>

#include "dwb_core/dwb_local_planner.hpp"
#include "dwb_core/trajectory_critic.hpp"
#include "dwb_plugins/standard_traj_generator.hpp"

namespace unitree_nav2_core
{

// Change candidate commands before DWB predicts and collision-checks them.
class SpeedLimitedGenerator : public dwb_plugins::StandardTrajectoryGenerator
{
public:
  void initialize(
    const nav2_util::LifecycleNode::SharedPtr & node,
    const std::string & plugin_name) override;
  void startNewIteration(const nav_2d_msgs::msg::Twist2D & velocity) override;
  bool hasMoreTwists() override;
  nav_2d_msgs::msg::Twist2D nextTwist() override;
  void setGoalDistance(double distance);
  dwb_msgs::msg::Trajectory2D generateTrajectory(
    const geometry_msgs::msg::Pose2D & start_pose,
    const nav_2d_msgs::msg::Twist2D & start_vel,
    const nav_2d_msgs::msg::Twist2D & cmd_vel) override;

protected:
  std::vector<double> getTimeSteps(const nav_2d_msgs::msg::Twist2D & cmd_vel) override;

private:
  nav2_util::LifecycleNode::SharedPtr node_;
  rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr parameter_callback_;
  std::vector<nav_2d_msgs::msg::Twist2D> candidates_;
  size_t index_{0};
  double goal_distance_{std::numeric_limits<double>::infinity()};
  double approach_distance_{0.6}, approach_speed_{0.1}, goal_tolerance_{0.15};
  nav_2d_msgs::msg::Twist2D rollout_start_velocity_;
};

// Supplies distance to the actual final goal (not a pruned local-plan endpoint)
// before DWB samples, predicts and collision-checks its candidate trajectories.
class ApproachDWBLocalPlanner : public dwb_core::DWBLocalPlanner
{
public:
  using dwb_core::DWBLocalPlanner::computeVelocityCommands;
  void configure(const rclcpp_lifecycle::LifecycleNode::SharedPtr & node,
    std::string name, const std::shared_ptr<tf2_ros::Buffer> & tf,
    const std::shared_ptr<nav2_costmap_2d::Costmap2DROS> & costmap_ros) override;
  geometry_msgs::msg::TwistStamped computeVelocityCommands(
    const geometry_msgs::msg::PoseStamped & pose,
    const geometry_msgs::msg::Twist & velocity) override;
protected:
  dwb_msgs::msg::TrajectoryScore coreScoringAlgorithm(
    const geometry_msgs::msg::Pose2D & pose, const nav_2d_msgs::msg::Twist2D velocity,
    std::shared_ptr<dwb_msgs::msg::LocalPlanEvaluation> & results) override;
private:
  double decision_log_period_{0.};
  std::chrono::steady_clock::time_point decision_logged_{};
};

// A preference, not a minimum velocity: slow/rotating candidates remain legal.
class CruiseSpeedCritic : public dwb_core::TrajectoryCritic
{
public:
  void onInit() override;
  bool prepare(
    const geometry_msgs::msg::Pose2D & pose, const nav_2d_msgs::msg::Twist2D & velocity,
    const geometry_msgs::msg::Pose2D & goal,
    const nav_2d_msgs::msg::Path2D & plan) override;
  double scoreTrajectory(const dwb_msgs::msg::Trajectory2D & trajectory) override;

private:
  double speed_{0.35};
  double weight_{0.0};
};

}  // namespace unitree_nav2_core
#endif
