// Closed-loop regression driver. All planning, sampling and scoring run in the
// installed Nav2 DWB; only the obstacle-free environment and vehicle are fake.
// No cmd_vel publisher, actuator, or real-time motion is used.
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "dwb_core/dwb_local_planner.hpp"
#include "nav2_costmap_2d/costmap_2d_ros.hpp"
#include "nav2_util/lifecycle_node.hpp"
#include "nav_2d_utils/conversions.hpp"
#include "rclcpp/rclcpp.hpp"
#include "tf2_ros/buffer.h"

using Pose = geometry_msgs::msg::Pose2D;
using Twist = nav_2d_msgs::msg::Twist2D;

double wrap(double angle) {return std::atan2(std::sin(angle), std::cos(angle));}

class RecordedDWB : public dwb_core::DWBLocalPlanner
{
public:
  dwb_msgs::msg::TrajectoryScore selected;
  nav_2d_msgs::msg::Path2D local_path;

  void describeCandidate(const Pose & pose, const Twist & velocity,
    double forward, double lateral = 0.0, double rotation = 0.0)
  {
    Twist command;
    command.x = forward;
    command.y = lateral;
    command.theta = rotation;
    try {
      auto score = scoreTrajectory(traj_generator_->generateTrajectory(pose, velocity, command));
      std::cout << "CART_DWB_CANDIDATE vx=" << forward << " vy=" << lateral
                << " wz=" << rotation << " total=" << score.total;
      for (const auto & critic : score.scores) {
        std::cout << ' ' << critic.name << '=' << critic.raw_score * critic.scale;
      }
      std::cout << std::endl;
    } catch (const std::exception & e) {
      std::cout << "CART_DWB_CANDIDATE vx=" << forward << " vy=" << lateral
                << " wz=" << rotation << " rejected=" << e.what() << std::endl;
    }
  }

protected:
  nav_2d_msgs::msg::Path2D transformGlobalPlan(
    const nav_2d_msgs::msg::Pose2DStamped & pose) override
  {
    local_path = DWBLocalPlanner::transformGlobalPlan(pose);
    return local_path;
  }

  dwb_msgs::msg::TrajectoryScore coreScoringAlgorithm(
    const Pose & pose, const Twist velocity,
    std::shared_ptr<dwb_msgs::msg::LocalPlanEvaluation> & results) override
  {
    selected = DWBLocalPlanner::coreScoringAlgorithm(pose, velocity, results);
    return selected;
  }
};

struct Projection {double distance, progress;};

Projection project(const Pose & p, const std::vector<Pose> & route)
{
  Projection best{std::numeric_limits<double>::infinity(), 0.0};
  double length = 0.0;
  for (size_t i = 1; i < route.size(); ++i) {
    const auto & a = route[i - 1];
    const auto & b = route[i];
    double dx = b.x - a.x, dy = b.y - a.y, d2 = dx * dx + dy * dy;
    if (d2 == 0.0) {continue;}
    double t = std::clamp(((p.x - a.x) * dx + (p.y - a.y) * dy) / d2, 0.0, 1.0);
    double distance = std::hypot(p.x - a.x - t * dx, p.y - a.y - t * dy);
    if (distance < best.distance) {best = {distance, length + t * std::sqrt(d2)};}
    length += std::sqrt(d2);
  }
  return best;
}

std::vector<Pose> makeRoute(const std::string & shape, double yaw)
{
  std::vector<Pose> route;
  Pose p;
  auto append = [&]() {
      Pose rotated;
      rotated.x = 0.73 + std::cos(yaw) * p.x - std::sin(yaw) * p.y;
      rotated.y = -0.41 + std::sin(yaw) * p.x + std::cos(yaw) * p.y;
      rotated.theta = wrap(yaw + p.theta);
      route.push_back(rotated);
    };
  auto line = [&](double length) {
      const Pose start = p;
      int count = static_cast<int>(std::ceil(length / 0.025));
      for (int i = 1; i <= count; ++i) {
        p.x = start.x + length * i / count * std::cos(p.theta);
        p.y = start.y + length * i / count * std::sin(p.theta);
        append();
      }
    };
  append();
  line(1.4);  // The cart must approach a corner that is initially far ahead.
  if (shape == "arc90") {
    const Pose start = p;
    for (int i = 1; i <= 48; ++i) {
      p.theta = M_PI_2 * i / 48;
      p.x = start.x + 0.6 * std::sin(p.theta);
      p.y = start.y + 0.6 * (1.0 - std::cos(p.theta));
      append();
    }
  } else if (shape == "left90") {
    p.theta = M_PI_2;
  } else if (shape == "right90") {
    p.theta = -M_PI_2;
  } else if (shape == "left60") {
    p.theta = M_PI / 3;
  } else {
    throw std::runtime_error("Unknown turn shape: " + shape);
  }
  line(3.0);
  return route;
}

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<nav2_util::LifecycleNode>("controller_server", "");
  const auto shape = node->declare_parameter<std::string>("test_shape", "left90");
  const double yaw = node->declare_parameter<double>("test_yaw", 0.785);
  const auto trace_path = node->declare_parameter<std::string>("test_trace", "");
  const double min_x = node->declare_parameter<double>("min_x_velocity_threshold", 0.0001);
  const double min_y = node->declare_parameter<double>("min_y_velocity_threshold", 0.0001);
  const double min_theta = node->declare_parameter<double>("min_theta_velocity_threshold", 0.0001);
  const auto route = makeRoute(shape, yaw);
  auto costmap = std::make_shared<nav2_costmap_2d::Costmap2DROS>("local_costmap");
  costmap->configure();
  auto * grid = costmap->getCostmap();
  auto tf = std::make_shared<tf2_ros::Buffer>(node->get_clock());
  RecordedDWB planner;
  planner.configure(node, "CartPush", tf, costmap);
  planner.activate();
  nav_2d_msgs::msg::Path2D path;
  path.header.frame_id = "odom";
  path.poses = route;
  planner.setPlan(nav_2d_utils::pathToPath(path));

  std::ofstream trace;
  if (!trace_path.empty()) {
    trace.open(trace_path);
    trace << "t,x,y,yaw,vx,vy,wz,progress,cross,local_goal_x,local_goal_y,score\n";
  }
  Pose cart = route.front();
  Twist velocity;
  double max_cross = 0.0, max_predicted_cross = 0.0, progress = 0.0;
  double last_progress_time = 0.0, high_progress = 0.0, t = 0.0;
  const double dt = 0.02;
  bool reached = false;
  std::string failure;
  for (; t < 55.0; t += dt) {
    // Match a rolling, empty local map. Keep production dimensions/resolution.
    grid->updateOrigin(cart.x - grid->getSizeInMetersX() / 2,
      cart.y - grid->getSizeInMetersY() / 2);
    nav_2d_msgs::msg::Pose2DStamped pose;
    pose.header.frame_id = "odom";
    pose.header.stamp = node->now();
    pose.pose = cart;
    std::shared_ptr<dwb_msgs::msg::LocalPlanEvaluation> evaluation;
    Twist command;
    try {
      // Match ControllerServer::getThresholdedTwist before calling DWB. Raw
      // exponentially decaying plant velocities otherwise create tiny signed
      // samples which the real server would never pass into the generator.
      Twist measured = velocity;
      if (std::abs(measured.x) <= min_x) {measured.x = 0.0;}
      if (std::abs(measured.y) <= min_y) {measured.y = 0.0;}
      if (std::abs(measured.theta) <= min_theta) {measured.theta = 0.0;}
      command = planner.computeVelocityCommands(pose, measured, evaluation).velocity;
    } catch (const std::exception & e) {
      failure = "planner_exception";
      std::cerr << e.what() << std::endl;
      break;
    }
    for (const auto & predicted : planner.selected.traj.poses) {
      max_predicted_cross = std::max(max_predicted_cross, project(predicted, route).distance);
    }
    // A lagged holonomic velocity response, not an ideal command integrator.
    double alpha = 1.0 - std::exp(-dt / 0.4);
    velocity.x += alpha * (command.x - velocity.x);
    velocity.y += alpha * (command.y - velocity.y);
    velocity.theta += alpha * (command.theta - velocity.theta);
    cart.x += dt * (std::cos(cart.theta) * velocity.x - std::sin(cart.theta) * velocity.y);
    cart.y += dt * (std::sin(cart.theta) * velocity.x + std::cos(cart.theta) * velocity.y);
    cart.theta = wrap(cart.theta + dt * velocity.theta);
    const auto projected = project(cart, route);
    max_cross = std::max(max_cross, projected.distance);
    progress = projected.progress;
    if (progress > high_progress + 0.025) {
      high_progress = progress;
      last_progress_time = t;
    }
    if (trace) {
      const auto & end = planner.local_path.poses.back();
      trace << t << ',' << cart.x << ',' << cart.y << ',' << cart.theta << ','
            << command.x << ',' << command.y << ',' << command.theta << ','
            << progress << ',' << projected.distance << ',' << end.x << ',' << end.y
            << ',' << planner.selected.total << '\n';
    }
    if (std::hypot(cart.x - route.back().x, cart.y - route.back().y) < 0.15 &&
      std::abs(wrap(cart.theta - route.back().theta)) < 0.174533)
    {
      reached = true;
      break;
    }
    // Allows several seconds of rotation at a sharp corner, but catches a
    // stationary local minimum well before the full simulated deadline.
    if (t - last_progress_time > 12.0) {failure = "stalled"; break;}
  }
  if (!reached && failure.empty()) {failure = "timeout";}
  if (!reached) {
    Twist measured = velocity;
    if (std::abs(measured.x) <= min_x) {measured.x = 0.0;}
    if (std::abs(measured.y) <= min_y) {measured.y = 0.0;}
    if (std::abs(measured.theta) <= min_theta) {measured.theta = 0.0;}
    planner.describeCandidate(cart, measured, 0.0);
    planner.describeCandidate(cart, measured, 0.1125);
    planner.describeCandidate(cart, measured, -0.1);
    for (double turn : {-0.4, -0.2, 0.2, 0.4}) {
      planner.describeCandidate(cart, measured, 0.0, 0.0, turn);
      planner.describeCandidate(cart, measured, 0.1125, 0.0, turn);
    }
  }
  std::cout << "CART_DWB_RESULT {\"shape\":\"" << shape << "\",\"yaw\":" << yaw
            << ",\"reached\":" << (reached ? "true" : "false")
            << ",\"failure\":\"" << failure << "\",\"sim_seconds\":" << t
            << ",\"progress_m\":" << progress << ",\"max_cross_m\":" << max_cross
            << ",\"max_predicted_cross_m\":" << max_predicted_cross
            << ",\"goal_distance_m\":" << std::hypot(cart.x - route.back().x, cart.y - route.back().y)
            << ",\"goal_heading_error_rad\":" << std::abs(wrap(cart.theta - route.back().theta))
            << ",\"shorten\":" << (node->get_parameter("CartPush.shorten_transformed_plan").as_bool() ? "true" : "false")
            << ",\"prune_distance\":" << node->get_parameter("CartPush.prune_distance").as_double()
            << ",\"twirling_scale\":" << node->get_parameter("CartPush.Twirling.scale").as_double()
            << ",\"speed\":" << node->get_parameter("CartPush.max_speed_xy").as_double()
            << ",\"sim_time\":" << node->get_parameter("CartPush.sim_time").as_double()
            << "}" << std::endl;
  planner.deactivate();
  planner.cleanup();
  costmap->cleanup();
  rclcpp::shutdown();
  return 0;  // Test assertions and the archived metrics are handled by pytest.
}
