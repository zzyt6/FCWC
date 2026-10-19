#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <thread>

#include "gtest/gtest.h"
#include "unitree_nav2_core/navigation_speed.hpp"
#include "dwb_core/exceptions.hpp"

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
      {"FollowPath.vtheta_samples", 17}, {"FollowPath.sim_time", 1.0},
      {"FollowPath.linear_granularity", 0.04}, {"FollowPath.angular_granularity", 0.025}});
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

TEST_F(NavigationSpeedTest, StopPredictionIncludesTravelWhileBraking)
{
  geometry_msgs::msg::Pose2D origin;
  nav_2d_msgs::msg::Twist2D moving, stop;
  moving.x = 0.3;
  auto trajectory = generator->generateTrajectory(origin, moving, stop);
  ASSERT_GT(trajectory.poses.size(), 2u);
  // With 0.3 m/s^2 braking, stopping from 0.3 m/s takes 1 s and about 0.15 m.
  EXPECT_GT(trajectory.poses.back().x, 0.12);
  EXPECT_LE(trajectory.poses.back().x, 0.15 + 1e-9);
  for (size_t i = 1; i < trajectory.poses.size(); ++i) {
    EXPECT_LE(std::hypot(trajectory.poses[i].x-trajectory.poses[i-1].x,
      trajectory.poses[i].y-trajectory.poses[i-1].y), 0.04 + 1e-9);
  }
  moving.x = 0.;
  moving.theta = 0.3;
  trajectory = generator->generateTrajectory(origin, moving, stop);
  EXPECT_GT(trajectory.poses.back().theta, 0.09);
  EXPECT_LE(trajectory.poses.back().theta, 0.1125 + 1e-9);
}

TEST_F(NavigationSpeedTest, OfficialSamplesObeyLiveSpeedAndTrueGoalApproach)
{
  unitree_nav2_core::NavigationSpeedLimitCritic limit;
  std::string name = "NavigationSpeedLimit", ns = "FollowPath";
  limit.initialize(node, name, ns, nullptr);
  dwb_plugins::StandardTrajectoryGenerator official;
  // The fixture already initialized the spatial generator and declared these.
  node->set_parameter(rclcpp::Parameter("FollowPath.discretize_by_time", true));
  node->set_parameter(rclcpp::Parameter("FollowPath.time_granularity", 0.05));
  official.initialize(node, ns);
  geometry_msgs::msg::Pose2D pose, goal;
  nav_2d_msgs::msg::Twist2D current;
  nav_2d_msgs::msg::Path2D shortened;
  shortened.poses.push_back(pose);  // Must not mistake this for the final goal.
  goal.x = 2.;
  for (const auto & mode : {"auto", "fixed"}) {
    for (const double cap : {0.2, 0.6}) {
      profile(mode, cap);
      limit.prepare(pose, current, goal, shortened);
      bool moving = false, turn = false, rejected = false;
      for (const auto & command : official.getTwists(current)) {
        auto trajectory = official.generateTrajectory(pose, current, command);
        const double speed = std::hypot(command.x, command.y);
        bool numerical_zero = false;
        for (const double axis : {command.x, command.y, command.theta}) {
          numerical_zero |= axis != 0. && std::abs(axis) < 1e-7;
        }
        if (speed <= cap+1e-9 && !numerical_zero) {
          EXPECT_DOUBLE_EQ(limit.scoreTrajectory(trajectory), 0.);
          moving |= speed > 0.1;
          turn |= speed < 1e-9 && std::abs(command.theta) > 0.1;
        } else {
          EXPECT_THROW(limit.scoreTrajectory(trajectory), dwb_core::IllegalTrajectoryException);
          rejected = true;
        }
      }
      EXPECT_TRUE(moving);
      EXPECT_TRUE(turn);
      if (cap == 0.2) {EXPECT_TRUE(rejected);}
    }
  }
  goal.x = 0.15;
  limit.prepare(pose, current, goal, shortened);
  dwb_msgs::msg::Trajectory2D slow;
  slow.velocity.x = 0.1;
  EXPECT_DOUBLE_EQ(limit.scoreTrajectory(slow), 0.);
  slow.velocity.x = 0.11;
  EXPECT_THROW(limit.scoreTrajectory(slow), dwb_core::IllegalTrajectoryException);
  // Official time discretization must retain the motion while braking.
  current.x = 0.3;
  nav_2d_msgs::msg::Twist2D stop;
  auto braking = official.generateTrajectory(pose, current, stop);
  EXPECT_DOUBLE_EQ(limit.scoreTrajectory(braking), 0.);
  ASSERT_GT(braking.poses.size(), 2u);
  EXPECT_GT(braking.poses.back().x, 0.12);
  EXPECT_LE(braking.poses.back().x, 0.15);
  for (int axis = 0; axis < 3; ++axis) {
    for (const double residue : {-2.8e-17, 2.8e-17}) {
      auto duplicate = braking;
      if (axis == 0) {duplicate.velocity.x = residue;}
      if (axis == 1) {duplicate.velocity.y = residue;}
      if (axis == 2) {duplicate.velocity.theta = residue;}
      EXPECT_THROW(limit.scoreTrajectory(duplicate), dwb_core::IllegalTrajectoryException);
    }
  }
}

TEST_F(NavigationSpeedTest, CartLookaheadDoesNotJumpWhenTheNearestWaypointChanges)
{
  unitree_nav2_core::CartLookaheadCritic critic;
  std::string name = "CartLookahead", ns = "CartPush";
  critic.initialize(node, name, ns, nullptr);
  nav_2d_msgs::msg::Path2D path;
  for (double x : {0., 1., 2.}) {
    geometry_msgs::msg::Pose2D point;
    point.x = x;
    path.poses.push_back(point);
  }
  geometry_msgs::msg::Pose2D pose, goal;
  goal.x = 2.;
  nav_2d_msgs::msg::Twist2D velocity;
  for (double x : {0.49, 0.51, 1.49, 1.51}) {
    pose.x = x;
    dwb_msgs::msg::Trajectory2D trajectory;
    geometry_msgs::msg::Pose2D end;
    end.x = x + 0.4;
    trajectory.poses.push_back(end);
    ASSERT_TRUE(critic.prepare(pose, velocity, goal, path));
    EXPECT_NEAR(critic.scoreTrajectory(trajectory), 0.0, 1e-9);
  }
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

TEST_F(NavigationSpeedTest, ZeroSamplesCannotAcquireSpuriousDirectionSigns)
{
  nav_2d_msgs::msg::Twist2D current;
  current.x = .3;
  for (const auto & command : generator->getTwists(current)) {
    for (double axis : {command.x, command.y, command.theta}) {
      EXPECT_TRUE(axis == 0. || std::abs(axis) >= 1e-7);
    }
  }
}

TEST_F(NavigationSpeedTest, CommandContinuityUsesSelectedCommandsAndSurvivesReplans)
{
  unitree_nav2_core::CommandChangeCritic critic;
  std::string name = "CommandChange", ns = "FollowPath";
  critic.initialize(node, name, ns, nullptr);
  geometry_msgs::msg::Pose2D pose, goal;
  nav_2d_msgs::msg::Path2D path;
  nav_2d_msgs::msg::Twist2D measured, selected;
  measured.y = -.18;  // A fluctuating velocity estimate is not command history.
  selected.y = .1;
  selected.theta = .15;
  dwb_msgs::msg::Trajectory2D keep, reverse, brake;
  keep.velocity = selected;
  reverse.velocity.y = -.1;
  reverse.velocity.theta = -.15;
  critic.prepare(pose, measured, goal, path);
  EXPECT_DOUBLE_EQ(critic.scoreTrajectory(reverse), 0.);
  critic.debrief(selected);
  critic.reset();  // Periodic BT global replans must not discard continuity.
  critic.prepare(pose, measured, goal, path);
  EXPECT_DOUBLE_EQ(critic.scoreTrajectory(keep), 0.);
  EXPECT_GT(critic.scoreTrajectory(reverse), critic.scoreTrajectory(brake));
  EXPECT_GT(critic.scoreTrajectory(brake), 0.);
  EXPECT_TRUE(std::isfinite(critic.scoreTrajectory(reverse)));  // A soft preference.
  // Scoring another candidate must not replace the actually selected command.
  EXPECT_DOUBLE_EQ(critic.scoreTrajectory(keep), 0.);
  std::this_thread::sleep_for(std::chrono::milliseconds(320));
  critic.prepare(pose, measured, goal, path);
  EXPECT_DOUBLE_EQ(critic.scoreTrajectory(reverse), 0.);
}
