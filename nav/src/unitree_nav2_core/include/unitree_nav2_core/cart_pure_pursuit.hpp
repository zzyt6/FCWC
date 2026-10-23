#ifndef UNITREE_NAV2_CORE__CART_PURE_PURSUIT_HPP_
#define UNITREE_NAV2_CORE__CART_PURE_PURSUIT_HPP_

#include <chrono>
#include <memory>
#include <string>

#include "nav2_regulated_pure_pursuit_controller/regulated_pure_pursuit_controller.hpp"

namespace unitree_nav2_core
{

// NavFn paths have uneven spacing, one-cell zigzags and no intermediate
// headings. Resample uniformly, smooth with a symmetric moving average that
// keeps both endpoints (two passes = triangular kernel of +/- 2 x smoothing
// distance), resample again and set every heading to the path tangent. The
// final pose keeps the goal orientation.
nav_msgs::msg::Path processCartPath(
  const nav_msgs::msg::Path & path, double resolution, double smoothing_distance);

// Largest distance from the part of candidate between arc length `from` and
// `from + length` (all of it if shorter) to the reference polyline.
double pathDeviation(
  const nav_msgs::msg::Path & candidate, const nav_msgs::msg::Path & reference,
  double from, double length);

// Foxy's official Regulated Pure Pursuit for the cart, which only has to track
// the global path. The pursuit law, curvature/cost regulation and rotate-to-
// heading are unchanged; this adapter only adds what 0.4.7 lacks:
//  - path conditioning (above) and replan hysteresis: a new plan to the same
//    goal that stays within path_keep_tolerance of the current one is ignored,
//    so the 1 Hz replanning does not make the carrot jump,
//  - live web speed (Foxy RPP reads desired_linear_vel once at configure),
//  - the shared final-approach cap measured to the real final goal,
//  - open-loop ramping from the last command (closed_loop=false), because the
//    legged executor filters every command and does not answer tiny steps,
//  - an angular limit that keeps the pursued curvature, a vx/wz rate limit in
//    every mode and a stopping profile when turning to the goal heading,
//  - a full-footprint check along the commanded arc (the stock check only
//    tests the cart-center point, ~0.6 m behind the cart's front edge).
class CartPurePursuitController
  : public nav2_regulated_pure_pursuit_controller::RegulatedPurePursuitController
{
public:
  void configure(
    const rclcpp_lifecycle::LifecycleNode::SharedPtr & node,
    std::string name, const std::shared_ptr<tf2_ros::Buffer> & tf,
    const std::shared_ptr<nav2_costmap_2d::Costmap2DROS> & costmap_ros) override;
  void activate() override;
  void deactivate() override;
  void setPlan(const nav_msgs::msg::Path & path) override;
  geometry_msgs::msg::TwistStamped computeVelocityCommands(
    const geometry_msgs::msg::PoseStamped & pose,
    const geometry_msgs::msg::Twist & velocity) override;

  // Translational cap before regulation, for the given distance to the goal.
  double speedLimit(double goal_distance) const;
  // Plans accepted by setPlan (the rest were within the keep tolerance).
  size_t acceptedPlans() const {return accepted_plans_;}

private:
  void checkFootprintAlongArc(
    const geometry_msgs::msg::PoseStamped & pose, double linear, double angular);

  rclcpp_lifecycle::LifecycleNode::SharedPtr node_;
  rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr parameter_callback_;
  double approach_distance_{0.8}, approach_speed_{0.1}, goal_tolerance_{0.15};
  double max_vel_x_{0.6}, max_vel_theta_{0.4}, collision_time_{1.5};
  double min_rotate_vel_{0.1};
  double path_resolution_{0.025}, path_smoothing_distance_{0.1};
  double path_keep_tolerance_{0.05}, path_compare_start_{0.5}, path_compare_length_{3.0};
  bool closed_loop_{false};
  size_t accepted_plans_{0};
  geometry_msgs::msg::Twist last_command_;
  std::chrono::steady_clock::time_point last_command_time_{};
};

}  // namespace unitree_nav2_core
#endif
