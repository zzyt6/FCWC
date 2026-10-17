#include <cmath>
#include <limits>
#include <memory>
#include <string>

#include "gtest/gtest.h"
#include "unitree_nav2_core/navigation_speed.hpp"

class NavigationSpeedTest : public ::testing::Test
{
protected:
  static void SetUpTestCase() {rclcpp::init(0, nullptr);}
  static void TearDownTestCase() {rclcpp::shutdown();}
  void SetUp() override
  {
    rclcpp::NodeOptions options;
    options.parameter_overrides({
      {"FollowPath.min_vel_x", -0.12}, {"FollowPath.max_vel_x", 0.60},
      {"FollowPath.min_vel_y", -0.18}, {"FollowPath.max_vel_y", 0.18},
      {"FollowPath.max_vel_theta", 0.45}, {"FollowPath.max_speed_xy", 0.60},
      {"FollowPath.min_speed_xy", 0.0}, {"FollowPath.min_speed_theta", 0.0},
      {"FollowPath.acc_lim_x", 0.30}, {"FollowPath.acc_lim_y", 0.20},
      {"FollowPath.acc_lim_theta", 0.40}, {"FollowPath.decel_lim_x", -0.30},
      {"FollowPath.decel_lim_y", -0.20}, {"FollowPath.decel_lim_theta", -0.40},
      {"FollowPath.vx_samples", 15}, {"FollowPath.vy_samples", 11},
      {"FollowPath.vtheta_samples", 17}, {"FollowPath.sim_time", 1.0}});
    node = std::make_shared<nav2_util::LifecycleNode>("speed_test", "", false, options);
    generator = std::make_unique<unitree_nav2_core::SpeedLimitedGenerator>();
    generator->initialize(node, "FollowPath");
  }
  void profile(const std::string & mode, double speed)
  {
    ASSERT_TRUE(node->set_parameters_atomically({
      {"FollowPath.navigation_speed_mode", mode},
      {"FollowPath.navigation_speed", speed}}).successful);
  }
  std::shared_ptr<nav2_util::LifecycleNode> node;
  std::unique_ptr<unitree_nav2_core::SpeedLimitedGenerator> generator;
};

TEST_F(NavigationSpeedTest, BothModesLimitPlanarNormAndRetainRotationAndSlowCandidates)
{
  for (const std::string mode : {"auto", "fixed"}) {
    profile(mode, 0.2);
    const auto commands = generator->getTwists(nav_2d_msgs::msg::Twist2D());
    ASSERT_FALSE(commands.empty());
    bool rotation = false, slow = false;
    for (const auto & command : commands) {
      const double norm = std::hypot(command.x, command.y);
      EXPECT_LE(norm, 0.2 + 1e-9);
      EXPECT_GE(command.x, -0.12 - 1e-9);
      EXPECT_LE(std::fabs(command.y), 0.18 + 1e-9);
      EXPECT_LE(std::fabs(command.theta), 0.45 + 1e-9);
      rotation |= norm < 1e-9 && std::fabs(command.theta) > 0.01;
      slow |= norm > 1e-9 && norm < 0.19;
    }
    EXPECT_TRUE(rotation);
    EXPECT_TRUE(slow);
  }
}

TEST_F(NavigationSpeedTest, ExactCruiseSamplesAndLiveReduction)
{
  nav_2d_msgs::msg::Twist2D velocity;
  velocity.x = 0.5;
  profile("fixed", 0.6);
  auto commands = generator->getTwists(velocity);
  bool cruise = false;
  for (const auto & command : commands) {
    cruise |= std::fabs(command.x - 0.6) < 1e-9 && std::fabs(command.y) < 1e-9;
    EXPECT_LE(std::hypot(command.x, command.y), 0.6 + 1e-9);
  }
  EXPECT_TRUE(cruise);
  profile("auto", 0.2);
  commands = generator->getTwists(velocity);
  ASSERT_FALSE(commands.empty());
  for (const auto & command : commands) {
    EXPECT_LE(std::hypot(command.x, command.y), 0.2 + 1e-9);
  }
}

TEST_F(NavigationSpeedTest, InvalidBatchDoesNotPartiallyChangeMode)
{
  for (double speed : {0.19, 0.61, std::numeric_limits<double>::quiet_NaN()}) {
    EXPECT_FALSE(node->set_parameters_atomically({
      {"FollowPath.navigation_speed_mode", "fixed"},
      {"FollowPath.navigation_speed", speed}}).successful);
    EXPECT_EQ(node->get_parameter("FollowPath.navigation_speed_mode").as_string(), "auto");
  }
  EXPECT_FALSE(node->set_parameters_atomically({
    {"FollowPath.navigation_speed_mode", "invalid"}}).successful);
}

TEST_F(NavigationSpeedTest, BothModesApproachAtBoundedSpeedWithoutLosingLegalStopsOrTurns)
{
  nav_2d_msgs::msg::Twist2D current;
  current.x = 0.3;  // The 1 s acceleration window includes both zero and 0.6.
  for (const std::string mode : {"auto", "fixed"}) {
    for (double requested : {0.2, 0.6}) {
      profile(mode, requested);
      for (const auto sample : {std::make_pair(1.0, requested),
        std::make_pair(0.6, requested), std::make_pair(0.375, (requested + 0.1) / 2.0),
        std::make_pair(0.15, 0.1), std::make_pair(0.02, 0.1)})
      {
        generator->setGoalDistance(sample.first);
        const auto commands = generator->getTwists(current);
        bool stop = false, turn = false, walking = false;
        double fastest = 0.0;
        for (const auto & command : commands) {
          const double speed = std::hypot(command.x, command.y);
          EXPECT_LE(speed, sample.second + 1e-9);
          fastest = std::max(fastest, speed);
          stop |= speed < 1e-9 && std::abs(command.theta) < 1e-9;
          turn |= speed < 1e-9 && std::abs(command.theta) > 0.05;
          walking |= speed > 0.05;
        }
        EXPECT_TRUE(stop);
        EXPECT_TRUE(turn);
        EXPECT_TRUE(walking);
        EXPECT_NEAR(fastest, sample.second, 1e-9);
        EXPECT_DOUBLE_EQ(node->get_parameter("FollowPath.navigation_speed").as_double(), requested);
      }
      // Moving back away from a goal / assigning another goal restores the cap.
      generator->setGoalDistance(2.0);
      double fastest = 0.0;
      for (const auto & command : generator->getTwists(current)) {
        fastest = std::max(fastest, std::hypot(command.x, command.y));
      }
      EXPECT_NEAR(fastest, requested, 1e-9);
    }
  }
}

TEST_F(NavigationSpeedTest, ApproachPredictionUsesBoundedCommandsAndRetainsDecelerationTravel)
{
  profile("fixed", 0.6);
  generator->setGoalDistance(0.15);
  nav_2d_msgs::msg::Twist2D moving;
  moving.x = 0.6;
  geometry_msgs::msg::Pose2D origin;
  const auto commands = generator->getTwists(moving);
  ASSERT_FALSE(commands.empty());
  for (const auto & command : commands) {
    const auto trajectory = generator->generateTrajectory(origin, moving, command);
    EXPECT_LE(std::hypot(trajectory.velocity.x, trajectory.velocity.y), 0.1 + 1e-9);
    ASSERT_FALSE(trajectory.poses.empty());
    // The predicted robot must still travel while braking from its measured
    // speed: do not pretend that the new 0.1 m/s target applies instantaneously.
    EXPECT_GT(trajectory.poses.back().x, 0.2);
  }
}

TEST_F(NavigationSpeedTest, CruisePreferenceReleasesForArrivalAndAutomaticMode)
{
  unitree_nav2_core::CruiseSpeedCritic critic;
  std::string name = "CruiseSpeed", ns = "FollowPath";
  critic.initialize(node, name, ns, nullptr);
  geometry_msgs::msg::Pose2D pose, goal;
  nav_2d_msgs::msg::Twist2D velocity;
  nav_2d_msgs::msg::Path2D path;
  dwb_msgs::msg::Trajectory2D slow, cruise;
  slow.velocity.x = 0.1;
  cruise.velocity.x = 0.4;
  profile("fixed", 0.4);
  goal.x = 2.0;
  critic.prepare(pose, velocity, goal, path);
  EXPECT_GT(critic.scoreTrajectory(slow), critic.scoreTrajectory(cruise));
  EXPECT_DOUBLE_EQ(critic.scoreTrajectory(cruise), 0.0);
  goal.x = 0.10;
  critic.prepare(pose, velocity, goal, path);
  EXPECT_DOUBLE_EQ(critic.scoreTrajectory(slow), 0.0);
  profile("auto", 0.4);
  goal.x = 2.0;
  critic.prepare(pose, velocity, goal, path);
  EXPECT_DOUBLE_EQ(critic.scoreTrajectory(slow), 0.0);
}

TEST_F(NavigationSpeedTest, CartLookaheadRewardsLateralCorrectionAndIncreasesWithSpeed)
{
  unitree_nav2_core::CartLookaheadCritic critic;
  std::string name = "CartLookahead", ns = "CartPush";
  critic.initialize(node, name, ns, nullptr);
  nav_2d_msgs::msg::Path2D path;
  for (double x : {0., 1., 2.}) {
    geometry_msgs::msg::Pose2D point;
    point.x = x;
    point.y = 0.3;
    path.poses.push_back(point);
  }
  geometry_msgs::msg::Pose2D pose, goal;
  goal.x = 2.; goal.y = .3;
  nav_2d_msgs::msg::Twist2D velocity;
  dwb_msgs::msg::Trajectory2D corrected, uncorrected, farther;
  geometry_msgs::msg::Pose2D end;
  end.x = .4; end.y = .3;
  corrected.poses.push_back(end);
  corrected.velocity.y = .15;
  end.y = 0.; uncorrected.poses.push_back(end);
  end.x = .76; end.y = .3; farther.poses.push_back(end);
  ASSERT_TRUE(critic.prepare(pose, velocity, goal, path));
  EXPECT_LT(critic.scoreTrajectory(corrected), critic.scoreTrajectory(uncorrected));
  EXPECT_LT(critic.scoreTrajectory(corrected), critic.scoreTrajectory(farther));
  velocity.x = .6;
  ASSERT_TRUE(critic.prepare(pose, velocity, goal, path));
  EXPECT_LT(critic.scoreTrajectory(farther), critic.scoreTrajectory(corrected));
  EXPECT_DOUBLE_EQ(corrected.velocity.y, .15);  // The critic never rewrites DWB velocity.
  path.poses.clear();
  EXPECT_FALSE(critic.prepare(pose, velocity, goal, path));
}
