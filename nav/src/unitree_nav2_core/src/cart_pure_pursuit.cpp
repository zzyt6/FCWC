#include "unitree_nav2_core/cart_pure_pursuit.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <utility>
#include <vector>

#include "nav2_core/exceptions.hpp"
#include "nav2_costmap_2d/cost_values.hpp"
#include "nav2_costmap_2d/footprint_collision_checker.hpp"
#include "nav2_util/node_utils.hpp"
#include "pluginlib/class_list_macros.hpp"
#include "tf2/utils.h"
#include "unitree_nav2_core/navigation_profile.hpp"

namespace unitree_nav2_core
{
namespace
{
struct Point {double x, y;};

std::vector<Point> resample(const std::vector<Point> & points, double spacing)
{
  std::vector<double> s{0.0};
  for (size_t i = 1; i < points.size(); ++i) {
    s.push_back(s.back() + std::hypot(points[i].x - points[i - 1].x, points[i].y - points[i - 1].y));
  }
  const double total = s.back();
  if (total < 1e-9) {return {points.front()};}
  const int count = std::max(1, static_cast<int>(std::ceil(total / spacing)));
  std::vector<Point> out;
  size_t segment = 1;
  for (int k = 0; k <= count; ++k) {
    const double target = total * k / count;
    while (segment + 1 < points.size() && s[segment] < target) {++segment;}
    const double length = s[segment] - s[segment - 1];
    const double t = length > 1e-12 ? std::clamp((target - s[segment - 1]) / length, 0.0, 1.0) : 1.0;
    out.push_back({points[segment - 1].x + t * (points[segment].x - points[segment - 1].x),
        points[segment - 1].y + t * (points[segment].y - points[segment - 1].y)});
  }
  return out;
}

// The window shrinks symmetrically near the ends, so both endpoints stay put
// and the start/goal are not pulled toward the interior.
std::vector<Point> movingAverage(const std::vector<Point> & points, int half_window)
{
  std::vector<Point> out(points.size());
  const int n = static_cast<int>(points.size());
  for (int i = 0; i < n; ++i) {
    const int k = std::min({half_window, i, n - 1 - i});
    Point sum{0.0, 0.0};
    for (int j = i - k; j <= i + k; ++j) {
      sum.x += points[j].x;
      sum.y += points[j].y;
    }
    out[i] = {sum.x / (2 * k + 1), sum.y / (2 * k + 1)};
  }
  return out;
}

double segmentDistance(const Point & p, const Point & a, const Point & b)
{
  const double dx = b.x - a.x, dy = b.y - a.y, d2 = dx * dx + dy * dy;
  const double t = d2 > 1e-12 ? std::clamp(((p.x - a.x) * dx + (p.y - a.y) * dy) / d2, 0.0, 1.0) : 0.0;
  return std::hypot(p.x - a.x - t * dx, p.y - a.y - t * dy);
}

std::vector<Point> points(const nav_msgs::msg::Path & path)
{
  std::vector<Point> out;
  for (const auto & pose : path.poses) {out.push_back({pose.pose.position.x, pose.pose.position.y});}
  return out;
}

geometry_msgs::msg::Quaternion yawToQuaternion(double yaw)
{
  geometry_msgs::msg::Quaternion q;
  q.z = std::sin(yaw / 2);
  q.w = std::cos(yaw / 2);
  return q;
}
}  // namespace

nav_msgs::msg::Path processCartPath(
  const nav_msgs::msg::Path & path, double resolution, double smoothing_distance)
{
  if (path.poses.size() < 2) {return path;}
  auto curve = resample(points(path), resolution);
  if (curve.size() >= 3) {
    const int half_window = static_cast<int>(std::lround(smoothing_distance / resolution));
    for (int pass = 0; pass < 2; ++pass) {curve = movingAverage(curve, half_window);}
    // Averaging shortens spacing around bends; make it uniform again.
    curve = resample(curve, resolution);
  }
  nav_msgs::msg::Path out;
  out.header = path.header;
  for (size_t i = 0; i < curve.size(); ++i) {
    geometry_msgs::msg::PoseStamped pose;
    pose.header = path.header;
    pose.pose.position.x = curve[i].x;
    pose.pose.position.y = curve[i].y;
    if (i + 1 == curve.size()) {
      pose.pose.orientation = path.poses.back().pose.orientation;
    } else {
      const auto & a = curve[i == 0 ? 0 : i - 1];
      const auto & b = curve[i + 1];
      pose.pose.orientation = yawToQuaternion(std::atan2(b.y - a.y, b.x - a.x));
    }
    out.poses.push_back(pose);
  }
  return out;
}

double pathDeviation(
  const nav_msgs::msg::Path & candidate, const nav_msgs::msg::Path & reference,
  double from, double length)
{
  const auto check = points(candidate), route = points(reference);
  if (check.empty()) {return 0.0;}
  if (route.empty()) {return std::numeric_limits<double>::infinity();}
  double total = 0.0;
  for (size_t i = 1; i < check.size(); ++i) {
    total += std::hypot(check[i].x - check[i - 1].x, check[i].y - check[i - 1].y);
  }
  if (total < from) {from = 0.0;}
  double s = 0.0, worst = 0.0;
  for (size_t i = 0; i < check.size(); ++i) {
    if (i > 0) {s += std::hypot(check[i].x - check[i - 1].x, check[i].y - check[i - 1].y);}
    if (s < from) {continue;}
    if (s > from + length) {break;}
    double best = std::hypot(check[i].x - route[0].x, check[i].y - route[0].y);
    for (size_t j = 1; j < route.size(); ++j) {
      best = std::min(best, segmentDistance(check[i], route[j - 1], route[j]));
    }
    worst = std::max(worst, best);
  }
  return worst;
}

void CartPurePursuitController::configure(
  const rclcpp_lifecycle::LifecycleNode::SharedPtr & node, std::string name,
  const std::shared_ptr<tf2_ros::Buffer> & tf,
  const std::shared_ptr<nav2_costmap_2d::Costmap2DROS> & costmap_ros)
{
  RegulatedPurePursuitController::configure(node, name, tf, costmap_ros);
  node_ = node;
  declareNavigationProfile(node, name);
  for (const auto & item : std::vector<std::pair<std::string, double>>{
      {"approach_distance", 0.8}, {"approach_speed", 0.1}, {"xy_goal_tolerance", 0.15},
      {"max_vel_x", 0.6}, {"max_vel_theta", 0.4}, {"min_rotate_vel", 0.1},
      {"path_resolution", 0.025}, {"path_smoothing_distance", 0.1},
      {"path_keep_tolerance", 0.05}, {"path_compare_start", 0.5}, {"path_compare_length", 3.0}})
  {
    nav2_util::declare_parameter_if_not_declared(
      node, name + "." + item.first, rclcpp::ParameterValue(item.second));
  }
  nav2_util::declare_parameter_if_not_declared(
    node, name + ".closed_loop", rclcpp::ParameterValue(false));
  approach_distance_ = node->get_parameter(name + ".approach_distance").as_double();
  approach_speed_ = node->get_parameter(name + ".approach_speed").as_double();
  goal_tolerance_ = node->get_parameter(name + ".xy_goal_tolerance").as_double();
  max_vel_x_ = node->get_parameter(name + ".max_vel_x").as_double();
  max_vel_theta_ = node->get_parameter(name + ".max_vel_theta").as_double();
  min_rotate_vel_ = node->get_parameter(name + ".min_rotate_vel").as_double();
  path_resolution_ = node->get_parameter(name + ".path_resolution").as_double();
  path_smoothing_distance_ = node->get_parameter(name + ".path_smoothing_distance").as_double();
  path_keep_tolerance_ = node->get_parameter(name + ".path_keep_tolerance").as_double();
  path_compare_start_ = node->get_parameter(name + ".path_compare_start").as_double();
  path_compare_length_ = node->get_parameter(name + ".path_compare_length").as_double();
  closed_loop_ = node->get_parameter(name + ".closed_loop").as_bool();
  collision_time_ = max_allowed_time_to_collision_;
  if (!std::isfinite(approach_distance_) || !std::isfinite(approach_speed_) ||
    !std::isfinite(goal_tolerance_) || goal_tolerance_ <= 0.0 ||
    approach_distance_ <= goal_tolerance_ || approach_speed_ <= 0.05 || approach_speed_ > 0.2)
  {
    throw std::invalid_argument("Invalid goal approach distance, speed or tolerance");
  }
  if (!(max_vel_x_ > approach_speed_) || !(max_vel_theta_ > 0.0) ||
    !(rotate_to_heading_angular_vel_ > 0.0) || rotate_to_heading_angular_vel_ > max_vel_theta_ ||
    !(min_rotate_vel_ > 0.0) || min_rotate_vel_ > rotate_to_heading_angular_vel_ ||
    !(collision_time_ > 0.0) || !(max_angular_accel_ > 0.0) ||
    !(max_linear_accel_ > 0.0) || !(max_linear_decel_ > 0.0))
  {
    throw std::invalid_argument("Invalid cart velocity limits or collision time");
  }
  if (!(path_resolution_ > 0.0) || !(path_smoothing_distance_ >= 0.0) ||
    !(path_keep_tolerance_ >= 0.0) || !(path_compare_start_ >= 0.0) ||
    !(path_compare_length_ > 0.0))
  {
    throw std::invalid_argument("Invalid cart path conditioning parameters");
  }
  // RPP rotates in place once the goal is within goal_dist_tol and never
  // translates again. Beyond the goal checker's radius that would spin forever.
  if (!(goal_dist_tol_ > 0.0) || goal_dist_tol_ > goal_tolerance_ + 1e-9) {
    throw std::invalid_argument("goal_dist_tol must be positive and within xy_goal_tolerance");
  }
  const auto initial = validateNavigationProfile(node->get_parameters({
    name + ".navigation_speed", name + ".navigation_speed_mode"}), name);
  if (!initial.successful) {throw std::invalid_argument(initial.reason);}
  parameter_callback_ = node->add_on_set_parameters_callback(
    [name](const std::vector<rclcpp::Parameter> & parameters) {
      return validateNavigationProfile(parameters, name);
    });
}

void CartPurePursuitController::activate()
{
  RegulatedPurePursuitController::activate();
  last_command_ = geometry_msgs::msg::Twist();
}

void CartPurePursuitController::deactivate()
{
  RegulatedPurePursuitController::deactivate();
  last_command_ = geometry_msgs::msg::Twist();
  // The next goal must not be compared with the previous one's path.
  global_plan_.poses.clear();
}

void CartPurePursuitController::setPlan(const nav_msgs::msg::Path & path)
{
  auto processed = processCartPath(path, path_resolution_, path_smoothing_distance_);
  if (!processed.poses.empty() && !global_plan_.poses.empty() &&
    processed.header.frame_id == global_plan_.header.frame_id)
  {
    // Same goal (endpoint and heading are preserved by the conditioning) and
    // the same route ahead: keep pursuing the current plan.
    const auto & goal = processed.poses.back().pose;
    const auto & current = global_plan_.poses.back().pose;
    const bool same_goal =
      std::hypot(goal.position.x - current.position.x, goal.position.y - current.position.y) < 1e-3 &&
      std::abs(std::remainder(tf2::getYaw(goal.orientation) - tf2::getYaw(current.orientation),
        2 * M_PI)) < 1e-3;
    if (same_goal && pathDeviation(processed, global_plan_, path_compare_start_,
      path_compare_length_) <= path_keep_tolerance_)
    {
      return;
    }
  }
  ++accepted_plans_;
  RegulatedPurePursuitController::setPlan(processed);
}

double CartPurePursuitController::speedLimit(double goal_distance) const
{
  // Fixed and automatic modes are the same here: pure pursuit always aims at
  // this speed, and only curvature/obstacle regulation slows it down.
  const double requested = node_->get_parameter(plugin_name_ + ".navigation_speed").as_double();
  // On the ramp, braking needs v * dv/dd; keep that within max_linear_decel
  // so the rate-limited command can follow the cap. This only starts the
  // ramp earlier above ~0.45 m/s (1.15 m instead of 0.80 m at 0.60 m/s).
  const double start = std::max(approach_distance_,
      goal_tolerance_ + requested * (requested - approach_speed_) / max_linear_decel_);
  const double fraction = std::clamp(
    (goal_distance - goal_tolerance_) / (start - goal_tolerance_), 0.0, 1.0);
  return std::min({max_vel_x_, requested,
      approach_speed_ + fraction * (requested - approach_speed_)});
}

geometry_msgs::msg::TwistStamped CartPurePursuitController::computeVelocityCommands(
  const geometry_msgs::msg::PoseStamped & pose, const geometry_msgs::msg::Twist & velocity)
{
  if (global_plan_.poses.empty()) {
    throw nav2_core::PlannerException("CartPush received an empty plan");
  }
  // RPP prunes passed poses, but the back of the plan is the real final goal.
  geometry_msgs::msg::PoseStamped goal = global_plan_.poses.back(), local_goal;
  goal.header.frame_id = global_plan_.header.frame_id;
  goal.header.stamp = pose.header.stamp;
  if (!transformPose(pose.header.frame_id, goal, local_goal)) {
    throw nav2_core::PlannerException("Cannot transform final goal for approach control");
  }
  const double goal_distance = std::hypot(
    local_goal.pose.position.x - pose.pose.position.x,
    local_goal.pose.position.y - pose.pose.position.y);
  desired_linear_vel_ = speedLimit(goal_distance);

  const auto now = std::chrono::steady_clock::now();
  // After a pause (failed plan, BT wait, new goal) ramp up again from rest.
  if (now - last_command_time_ > std::chrono::milliseconds(250)) {
    last_command_ = geometry_msgs::msg::Twist();
  }
  std::lock_guard<nav2_costmap_2d::Costmap2D::mutex_t> lock(*costmap_->getMutex());
  auto command = RegulatedPurePursuitController::computeVelocityCommands(
    pose, closed_loop_ ? velocity : last_command_);
  auto & twist = command.twist;
  if (std::abs(twist.angular.z) > max_vel_theta_) {
    // Scale both axes: the arc (curvature = wz / vx) is what pure pursuit chose.
    const double scale = max_vel_theta_ / std::abs(twist.angular.z);
    twist.linear.x *= scale;
    twist.angular.z *= scale;
  }
  if (twist.linear.x == 0.0 && twist.angular.z != 0.0 && goal_distance < goal_dist_tol_) {
    // Final in-place turn: brake so the turn can stop at the goal heading.
    const double remaining = std::abs(std::remainder(
        tf2::getYaw(local_goal.pose.orientation) - tf2::getYaw(pose.pose.orientation), 2 * M_PI));
    const double limit = std::max(min_rotate_vel_, std::sqrt(2.0 * max_angular_accel_ * remaining));
    twist.angular.z = std::clamp(twist.angular.z, -limit, limit);
  }
  // Pure pursuit sets wz = vx * curvature with no rate limit, and RPP drops vx
  // to zero when it switches to an in-place turn. Bound both changes per cycle
  // so a carrot or mode switch cannot step the command sent to the policy.
  const double dt = control_duration_;
  twist.linear.x = std::clamp(twist.linear.x,
      last_command_.linear.x - max_linear_decel_ * dt, last_command_.linear.x + max_linear_accel_ * dt);
  twist.linear.x = std::max(twist.linear.x, 0.0);
  twist.angular.z = std::clamp(twist.angular.z,
      last_command_.angular.z - max_angular_accel_ * dt, last_command_.angular.z + max_angular_accel_ * dt);
  checkFootprintAlongArc(pose, twist.linear.x, twist.angular.z);
  last_command_ = twist;
  last_command_time_ = now;
  return command;
}

void CartPurePursuitController::checkFootprintAlongArc(
  const geometry_msgs::msg::PoseStamped & pose, double linear, double angular)
{
  const auto footprint = costmap_ros_->getRobotFootprint();
  nav2_costmap_2d::FootprintCollisionChecker<nav2_costmap_2d::Costmap2D *> checker(costmap_);
  double x = pose.pose.position.x, y = pose.pose.position.y;
  double theta = tf2::getYaw(pose.pose.orientation);
  // At most one cell of travel or 0.025 rad of rotation between checks.
  const int steps = std::max(1, static_cast<int>(std::ceil(std::max(
      std::abs(linear) * collision_time_ / costmap_->getResolution(),
      std::abs(angular) * collision_time_ / 0.025))));
  const double dt = collision_time_ / steps;
  for (int i = 0; i <= steps; ++i) {
    if (i > 0) {
      const double heading = theta + 0.5 * angular * dt;
      x += linear * dt * std::cos(heading);
      y += linear * dt * std::sin(heading);
      theta += angular * dt;
    }
    if (checker.footprintCostAtPose(x, y, theta, footprint) >=
      static_cast<double>(nav2_costmap_2d::LETHAL_OBSTACLE))
    {
      throw nav2_core::PlannerException("CartPush footprint collision ahead");
    }
  }
}

}  // namespace unitree_nav2_core

PLUGINLIB_EXPORT_CLASS(unitree_nav2_core::CartPurePursuitController, nav2_core::Controller)
