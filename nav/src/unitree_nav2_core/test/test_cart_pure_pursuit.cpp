#include <cmath>
#include <vector>

#include "gtest/gtest.h"
#include "tf2/utils.h"
#include "unitree_nav2_core/cart_pure_pursuit.hpp"

namespace
{
nav_msgs::msg::Path makePath(const std::vector<std::pair<double, double>> & points, double goal_yaw)
{
  nav_msgs::msg::Path path;
  path.header.frame_id = "map";
  for (const auto & point : points) {
    geometry_msgs::msg::PoseStamped pose;
    pose.pose.position.x = point.first;
    pose.pose.position.y = point.second;
    pose.pose.orientation.w = 1.0;  // NavFn leaves intermediate headings unset.
    path.poses.push_back(pose);
  }
  path.poses.back().pose.orientation.z = std::sin(goal_yaw / 2);
  path.poses.back().pose.orientation.w = std::cos(goal_yaw / 2);
  return path;
}

double spacing(const nav_msgs::msg::Path & path, size_t i)
{
  return std::hypot(path.poses[i].pose.position.x - path.poses[i - 1].pose.position.x,
           path.poses[i].pose.position.y - path.poses[i - 1].pose.position.y);
}
}  // namespace

TEST(CartPathConditioning, UnevenZigzagBecomesUniformStraightAndKeepsEndpoints)
{
  // 4 m line, +/-2.5 cm one-cell zigzag, spacing alternating 2.5 / 7.5 cm.
  std::vector<std::pair<double, double>> points{{0.0, 0.0}};
  double x = 0.0;
  for (int i = 1; x < 3.9; ++i) {
    x += i % 2 ? 0.025 : 0.075;
    points.emplace_back(x, i % 2 ? 0.025 : -0.025);
  }
  points.emplace_back(4.0, 0.0);
  const auto out = unitree_nav2_core::processCartPath(makePath(points, 0.3), 0.025, 0.10);
  ASSERT_GT(out.poses.size(), 150u);
  EXPECT_EQ(out.header.frame_id, "map");
  EXPECT_DOUBLE_EQ(out.poses.front().pose.position.x, 0.0);
  EXPECT_DOUBLE_EQ(out.poses.front().pose.position.y, 0.0);
  EXPECT_NEAR(out.poses.back().pose.position.x, 4.0, 1e-9);
  EXPECT_NEAR(out.poses.back().pose.position.y, 0.0, 1e-9);
  EXPECT_NEAR(tf2::getYaw(out.poses.back().pose.orientation), 0.3, 1e-9);  // Goal heading kept.
  const double first = spacing(out, 1);
  EXPECT_LE(first, 0.025 + 1e-9);
  for (size_t i = 1; i < out.poses.size(); ++i) {
    EXPECT_NEAR(spacing(out, i), first, 1e-3);
  }
  for (size_t i = 0; i + 1 < out.poses.size(); ++i) {
    const auto & p = out.poses[i].pose;
    if (p.position.x > 0.4 && p.position.x < 3.6) {
      EXPECT_LT(std::abs(p.position.y), 0.005) << p.position.x;
      EXPECT_LT(std::abs(tf2::getYaw(p.orientation)), 0.05) << p.position.x;
    }
  }
}

TEST(CartPathConditioning, CornerIsRoundedSlightlyWithContinuousTangents)
{
  std::vector<std::pair<double, double>> points;
  for (int i = 0; i <= 80; ++i) {points.emplace_back(0.025 * i, 0.0);}
  for (int i = 1; i <= 80; ++i) {points.emplace_back(2.0, 0.025 * i);}
  const auto out = unitree_nav2_core::processCartPath(makePath(points, M_PI_2), 0.025, 0.10);
  double nearest = 1e9, max_turn = 0.0, turning_length = 0.0;
  for (size_t i = 0; i < out.poses.size(); ++i) {
    const auto & p = out.poses[i].pose;
    nearest = std::min(nearest, std::hypot(p.position.x - 2.0, p.position.y));
    if (i > 0 && i + 1 < out.poses.size()) {
      const double turn = std::remainder(
        tf2::getYaw(p.orientation) - tf2::getYaw(out.poses[i - 1].pose.orientation), 2 * M_PI);
      max_turn = std::max(max_turn, std::abs(turn));
      if (std::abs(turn) > 0.01) {turning_length += spacing(out, i);}
    }
  }
  // Rounded inward by a few centimetres, not cut across.
  EXPECT_GT(nearest, 0.02);
  EXPECT_LT(nearest, 0.07);
  // The 90 degree step is spread over the +/-0.2 m triangular kernel. A
  // larger fillet would cut the corner further; pure pursuit already rounds
  // it with its 0.5 m carrot and never reads intermediate headings.
  EXPECT_GT(turning_length, 0.3);
  EXPECT_LT(max_turn, 0.3);
}

TEST(CartPathConditioning, ShortPathsAreLeftUsable)
{
  const auto single = unitree_nav2_core::processCartPath(makePath({{1.0, 2.0}}, 0.5), 0.025, 0.1);
  ASSERT_EQ(single.poses.size(), 1u);
  const auto two = unitree_nav2_core::processCartPath(
    makePath({{0.0, 0.0}, {0.01, 0.0}}, 0.5), 0.025, 0.1);
  ASSERT_EQ(two.poses.size(), 2u);
  EXPECT_NEAR(two.poses.back().pose.position.x, 0.01, 1e-12);
  EXPECT_NEAR(tf2::getYaw(two.poses.back().pose.orientation), 0.5, 1e-9);
}

TEST(CartPathConditioning, DeviationIgnoresTheStartAndDetectsRouteChanges)
{
  std::vector<std::pair<double, double>> straight, rejoin, detour;
  for (int i = 0; i <= 200; ++i) {
    const double x = 0.025 * i;
    straight.emplace_back(x, 0.0);
    // A replan from a cart 0.2 m off the path that rejoins within 0.4 m.
    rejoin.emplace_back(x, x < 0.4 ? 0.2 * (1.0 - x / 0.4) : 0.0);
    // A new obstacle 2 m ahead forces a 0.3 m detour.
    detour.emplace_back(x, x > 1.5 && x < 3.0 ? 0.3 : 0.0);
  }
  const auto reference = makePath(straight, 0.0);
  EXPECT_DOUBLE_EQ(unitree_nav2_core::pathDeviation(reference, reference, 0.5, 3.0), 0.0);
  EXPECT_LT(unitree_nav2_core::pathDeviation(makePath(rejoin, 0.0), reference, 0.5, 3.0), 1e-9);
  EXPECT_NEAR(unitree_nav2_core::pathDeviation(makePath(rejoin, 0.0), reference, 0.0, 3.0), 0.2, 1e-9);
  EXPECT_GT(unitree_nav2_core::pathDeviation(makePath(detour, 0.0), reference, 0.5, 3.0), 0.29);
  // Beyond the compared window the detour is not seen.
  EXPECT_LT(unitree_nav2_core::pathDeviation(makePath(detour, 0.0), reference, 0.0, 1.4), 1e-9);
}
