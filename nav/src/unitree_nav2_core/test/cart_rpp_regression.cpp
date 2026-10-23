// Closed-loop regression driver. Pursuit, regulation and rotate-to-heading run
// in the installed Nav2 Foxy RPP through the production CartPush adapter; only
// the local map and vehicle are fake. No cmd_vel publisher or actuator is used.
#include <algorithm>
#include <cmath>
#include <deque>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "nav2_costmap_2d/cost_values.hpp"
#include "nav2_costmap_2d/costmap_2d_ros.hpp"
#include "nav2_costmap_2d/footprint_collision_checker.hpp"
#include "nav2_util/lifecycle_node.hpp"
#include "rclcpp/rclcpp.hpp"
#include "tf2/LinearMath/Quaternion.h"
#include "tf2_ros/buffer.h"
#include "unitree_nav2_core/cart_pure_pursuit.hpp"

using Pose = geometry_msgs::msg::Pose2D;

double wrap(double angle) {return std::atan2(std::sin(angle), std::cos(angle));}

geometry_msgs::msg::Quaternion quaternion(double yaw)
{
  tf2::Quaternion q;
  q.setRPY(0.0, 0.0, yaw);
  geometry_msgs::msg::Quaternion out;
  out.x = q.x(); out.y = q.y(); out.z = q.z(); out.w = q.w();
  return out;
}

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
  } else if (shape != "straight") {
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
  const double initial_offset = node->declare_parameter<double>("test_lateral_offset", 0.0);
  // > 0: a lethal 0.2 m square on the route this far from its start.
  const double obstacle_at = node->declare_parameter<double>("test_obstacle_at", -1.0);
  const auto trace_path = node->declare_parameter<std::string>("test_trace", "");
  // NavFn-like defects: +/- this zigzag normal to the route on alternate
  // points, and ~1/3 of the points dropped (uneven spacing).
  const double path_noise = node->declare_parameter<double>("test_path_noise", 0.0);
  // > 0: replan like the BT, from the cart's position with fresh noise.
  const double replan_period = node->declare_parameter<double>("test_replan_period", 0.0);
  // Plant: first-order response plus a pure dead time. The defaults are the
  // original 0.4 s response; the real push chain (executor filter, cart
  // student, body, caster cart) followed RPP wz with ~0.8 s total lag.
  const double response_s = node->declare_parameter<double>("test_response_s", 0.4);
  const double dead_time_s = node->declare_parameter<double>("test_dead_time_s", 0.0);
  node->declare_parameter<double>("controller_frequency", 20.0);
  // Arrival uses the production goal checker (controller_server parameters).
  const double goal_xy = node->declare_parameter<double>("goal_checker.xy_goal_tolerance", 0.25);
  const double goal_yaw = node->declare_parameter<double>("goal_checker.yaw_goal_tolerance", 0.3490658503988659);
  const double dt = 1.0 / node->get_parameter("controller_frequency").as_double();
  const auto route = makeRoute(shape, yaw);
  auto costmap = std::make_shared<nav2_costmap_2d::Costmap2DROS>("local_costmap");
  costmap->configure();
  auto * grid = costmap->getCostmap();
  auto tf = std::make_shared<tf2_ros::Buffer>(node->get_clock());
  tf->setUsingDedicatedThread(true);
  const auto global_frame = costmap->getGlobalFrameID();
  const auto base_frame = costmap->getBaseFrameID();
  unitree_nav2_core::CartPurePursuitController planner;
  planner.configure(node, "CartPush", tf, costmap);
  planner.activate();
  std::mt19937 random(7);
  // Raw plan from `start` through route[from...]; the goal pose stays exact.
  auto rawPlan = [&](const Pose & start, size_t from) {
      nav_msgs::msg::Path path;
      path.header.frame_id = global_frame;
      std::bernoulli_distribution drop(path_noise > 0.0 ? 1.0 / 3.0 : 0.0);
      const int phase = static_cast<int>(random() % 2);
      auto add = [&](const Pose & point, double normal) {
          geometry_msgs::msg::PoseStamped stamped;
          stamped.header.frame_id = global_frame;
          stamped.pose.position.x = point.x - normal * std::sin(point.theta);
          stamped.pose.position.y = point.y + normal * std::cos(point.theta);
          // NavFn leaves intermediate orientations unset.
          stamped.pose.orientation = quaternion(path_noise > 0.0 ? 0.0 : point.theta);
          path.poses.push_back(stamped);
        };
      add(start, 0.0);
      for (size_t i = from; i + 1 < route.size(); ++i) {
        if (!drop(random)) {add(route[i], ((i + phase) % 2 ? 1.0 : -1.0) * path_noise);}
      }
      add(route.back(), 0.0);
      path.poses.back().pose.orientation = quaternion(route.back().theta);
      return path;
    };
  planner.setPlan(rawPlan(route.front(), 1));

  Pose obstacle;
  if (obstacle_at > 0.0) {
    obstacle = route.front();
    obstacle.x += obstacle_at * std::cos(route.front().theta);
    obstacle.y += obstacle_at * std::sin(route.front().theta);
  }
  nav2_costmap_2d::FootprintCollisionChecker<nav2_costmap_2d::Costmap2D *> checker(grid);
  const auto footprint = costmap->getRobotFootprint();
  auto refreshMap = [&](const Pose & cart) {
      // Match a rolling local map. Keep production dimensions/resolution.
      grid->updateOrigin(cart.x - grid->getSizeInMetersX() / 2,
        cart.y - grid->getSizeInMetersY() / 2);
      if (obstacle_at <= 0.0) {return;}
      for (double ox = -0.1; ox <= 0.1; ox += grid->getResolution() / 2) {
        for (double oy = -0.1; oy <= 0.1; oy += grid->getResolution() / 2) {
          unsigned int mx, my;
          if (grid->worldToMap(obstacle.x + ox, obstacle.y + oy, mx, my)) {
            grid->setCost(mx, my, nav2_costmap_2d::LETHAL_OBSTACLE);
          }
        }
      }
    };

  std::ofstream trace;
  if (!trace_path.empty()) {
    trace.open(trace_path);
    trace << "t,x,y,yaw,vx,wz,cap,progress,cross\n";
  }
  Pose cart = route.front();
  cart.x -= initial_offset * std::sin(cart.theta);
  cart.y += initial_offset * std::cos(cart.theta);
  geometry_msgs::msg::Twist velocity;
  double max_cross = 0.0, late_cross = 0.0, progress = 0.0, max_vx = 0.0, max_wz = 0.0;
  int wz_reversals = 0, last_wz_sign = 0;
  std::deque<geometry_msgs::msg::Twist> delayed(
    static_cast<size_t>(std::lround(dead_time_s * node->get_parameter("controller_frequency").as_double())));
  double speed_excess = -std::numeric_limits<double>::infinity(), min_vx = 0.0, max_vy = 0.0;
  double last_progress_time = 0.0, high_progress = 0.0, t = 0.0;
  double max_footprint_cost = 0.0, max_vx_step = 0.0, max_wz_step = 0.0, wz_variation = 0.0, next_replan = replan_period;
  geometry_msgs::msg::Twist previous_command;
  bool reached = false;
  std::string failure, error;
  auto step = [&](geometry_msgs::msg::Twist command) {
      if (!delayed.empty()) {
        delayed.push_back(command);
        command = delayed.front();
        delayed.pop_front();
      }
      // A lagged velocity response, not an ideal command integrator.
      const double alpha = 1.0 - std::exp(-dt / response_s);
      velocity.linear.x += alpha * (command.linear.x - velocity.linear.x);
      velocity.linear.y += alpha * (command.linear.y - velocity.linear.y);
      velocity.angular.z += alpha * (command.angular.z - velocity.angular.z);
      cart.x += dt * (std::cos(cart.theta) * velocity.linear.x - std::sin(cart.theta) * velocity.linear.y);
      cart.y += dt * (std::sin(cart.theta) * velocity.linear.x + std::cos(cart.theta) * velocity.linear.y);
      cart.theta = wrap(cart.theta + dt * velocity.angular.z);
      refreshMap(cart);
      max_footprint_cost = std::max(max_footprint_cost,
        checker.footprintCostAtPose(cart.x, cart.y, cart.theta, footprint));
    };
  refreshMap(cart);
  for (; t < 60.0; t += dt) {
    geometry_msgs::msg::PoseStamped pose;
    pose.header.frame_id = global_frame;
    pose.header.stamp = node->now();
    pose.pose.position.x = cart.x;
    pose.pose.position.y = cart.y;
    pose.pose.orientation = quaternion(cart.theta);
    geometry_msgs::msg::TransformStamped transform;
    transform.header = pose.header;
    transform.child_frame_id = base_frame;
    transform.transform.translation.x = cart.x;
    transform.transform.translation.y = cart.y;
    transform.transform.rotation = pose.pose.orientation;
    tf->setTransform(transform, "cart_rpp_regression", true);
    const double cap = planner.speedLimit(
      std::hypot(route.back().x - cart.x, route.back().y - cart.y));
    geometry_msgs::msg::Twist command;
    try {
      command = planner.computeVelocityCommands(pose, velocity).twist;
    } catch (const std::exception & e) {
      failure = "planner_exception";
      error = e.what();
      break;
    }
    max_vx_step = std::max(max_vx_step, std::abs(command.linear.x - previous_command.linear.x));
    max_wz_step = std::max(max_wz_step, std::abs(command.angular.z - previous_command.angular.z));
    // Total variation: chatter adds up even when every single step is bounded.
    wz_variation += std::abs(command.angular.z - previous_command.angular.z);
    previous_command = command;
    max_vx = std::max(max_vx, command.linear.x);
    min_vx = std::min(min_vx, command.linear.x);
    max_vy = std::max(max_vy, std::abs(command.linear.y));
    max_wz = std::max(max_wz, std::abs(command.angular.z));
    // Weaving: count sign changes of a clear turn command.
    if (std::abs(command.angular.z) > 0.1) {
      const int sign = command.angular.z > 0.0 ? 1 : -1;
      wz_reversals += last_wz_sign != 0 && sign != last_wz_sign;
      last_wz_sign = sign;
    }
    speed_excess = std::max(speed_excess, command.linear.x - cap);
    step(command);
    const auto projected = project(cart, route);
    max_cross = std::max(max_cross, projected.distance);
    progress = projected.progress;
    // After the first 2 m the start transient is over; a stable tracker has
    // settled onto the path here.
    if (progress > 2.0) {late_cross = std::max(late_cross, projected.distance);}
    if (progress > high_progress + 0.025) {
      high_progress = progress;
      last_progress_time = t;
    }
    if (trace) {
      trace << t << ',' << cart.x << ',' << cart.y << ',' << cart.theta << ','
            << command.linear.x << ',' << command.angular.z << ',' << cap << ','
            << progress << ',' << projected.distance << '\n';
    }
    if (std::hypot(cart.x - route.back().x, cart.y - route.back().y) < goal_xy &&
      std::abs(wrap(cart.theta - route.back().theta)) < goal_yaw)
    {
      reached = true;
      break;
    }
    if (replan_period > 0.0 && t >= next_replan) {
      next_replan += replan_period;
      size_t nearest = 0;
      for (size_t i = 1; i < route.size(); ++i) {
        if (std::hypot(route[i].x - cart.x, route[i].y - cart.y) <
          std::hypot(route[nearest].x - cart.x, route[nearest].y - cart.y)) {nearest = i;}
      }
      Pose start = cart;
      planner.setPlan(rawPlan(start, std::min(nearest + 1, route.size() - 1)));
    }
    // Allows several seconds of rotation at a sharp corner, but catches a
    // stationary local minimum well before the full simulated deadline.
    if (t - last_progress_time > 12.0) {failure = "stalled"; break;}
  }
  if (!reached && failure.empty()) {failure = "timeout";}
  if (failure == "planner_exception") {
    // controller_server publishes zero on failure; let the lagged plant stop.
    for (int i = 0; i < static_cast<int>(3.0 / dt); ++i) {step(geometry_msgs::msg::Twist());}
  }
  for (auto & c : error) {if (c == '"' || c == '\\') {c = '\'';}}
  std::cout << "CART_RPP_RESULT {\"shape\":\"" << shape << "\",\"yaw\":" << yaw
            << ",\"reached\":" << (reached ? "true" : "false")
            << ",\"failure\":\"" << failure << "\",\"error\":\"" << error
            << "\",\"sim_seconds\":" << t
            << ",\"progress_m\":" << progress << ",\"max_cross_m\":" << max_cross
            << ",\"late_cross_m\":" << late_cross << ",\"wz_reversals\":" << wz_reversals
            << ",\"goal_distance_m\":" << std::hypot(cart.x - route.back().x, cart.y - route.back().y)
            << ",\"goal_heading_error_rad\":" << std::abs(wrap(cart.theta - route.back().theta))
            << ",\"max_vx\":" << max_vx << ",\"min_vx\":" << min_vx << ",\"max_abs_vy\":" << max_vy
            << ",\"max_abs_wz\":" << max_wz << ",\"speed_excess\":" << speed_excess
            << ",\"max_footprint_cost\":" << max_footprint_cost
            << ",\"max_vx_step\":" << max_vx_step << ",\"max_wz_step\":" << max_wz_step << ",\"wz_variation\":" << wz_variation
            << ",\"accepted_plans\":" << planner.acceptedPlans()
            << ",\"speed\":" << node->get_parameter("CartPush.navigation_speed").as_double()
            << ",\"dt\":" << dt
            << "}" << std::endl;
  planner.deactivate();
  planner.cleanup();
  costmap->cleanup();
  rclcpp::shutdown();
  return 0;  // Test assertions and the archived metrics are handled by pytest.
}
