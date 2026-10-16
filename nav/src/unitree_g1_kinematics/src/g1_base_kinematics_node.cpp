#include <arpa/inet.h>
#include <fcntl.h>
#include <sys/socket.h>
#include <unistd.h>

#include <array>
#include <chrono>
#include <cmath>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Geometry>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/header.hpp>
#include <tf2_ros/transform_broadcaster.h>

#include "unitree_g1_kinematics/lowstate_packet.hpp"

namespace
{
constexpr double kPi = 3.14159265358979323846;
constexpr double kOriginalMid360Pitch = 0.04014257279586953;
constexpr double kModernMid360Pitch = 0.05112069379091391;

Eigen::Isometry3d translation(double x, double y, double z)
{
  Eigen::Isometry3d transform = Eigen::Isometry3d::Identity();
  transform.translation() = Eigen::Vector3d(x, y, z);
  return transform;
}

Eigen::Isometry3d rotation(const Eigen::Vector3d &axis, double angle)
{
  Eigen::Isometry3d transform = Eigen::Isometry3d::Identity();
  transform.linear() = Eigen::AngleAxisd(angle, axis).toRotationMatrix();
  return transform;
}

geometry_msgs::msg::TransformStamped toTransform(
    const Eigen::Isometry3d &pose,
    const std_msgs::msg::Header &header,
    const std::string &parent,
    const std::string &child)
{
  geometry_msgs::msg::TransformStamped output;
  output.header = header;
  output.header.frame_id = parent;
  output.child_frame_id = child;
  output.transform.translation.x = pose.translation().x();
  output.transform.translation.y = pose.translation().y();
  output.transform.translation.z = pose.translation().z();
  Eigen::Quaterniond quaternion(pose.linear());
  quaternion.normalize();
  output.transform.rotation.w = quaternion.w();
  output.transform.rotation.x = quaternion.x();
  output.transform.rotation.y = quaternion.y();
  output.transform.rotation.z = quaternion.z();
  return output;
}

const std::array<std::string, 29> kJointNames = {
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint",
    "left_wrist_yaw_joint", "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint",
    "right_wrist_pitch_joint", "right_wrist_yaw_joint"};
}  // namespace

class G1BaseKinematicsNode : public rclcpp::Node
{
public:
  G1BaseKinematicsNode() : Node("g1_base_kinematics")
  {
    declare_parameter<int>("lowstate_udp_port", 39001);
    declare_parameter<std::string>("fastlio_odom_topic", "/Odometry");
    declare_parameter<std::string>("base_odom_topic", "/Odometry_base");
    declare_parameter<std::string>("base_frame", "base_link");
    declare_parameter<std::string>("navigation_frame", "base_nav");
    declare_parameter<std::string>("model_variant", "auto");
    declare_parameter<double>("twist_smoothing_alpha", 0.35);

    base_frame_ = get_parameter("base_frame").as_string();
    navigation_frame_ = get_parameter("navigation_frame").as_string();
    model_variant_ = get_parameter("model_variant").as_string();
    twist_smoothing_alpha_ = std::max(
      0.0, std::min(1.0, get_parameter("twist_smoothing_alpha").as_double()));

    odom_pub_ = create_publisher<nav_msgs::msg::Odometry>(
        get_parameter("base_odom_topic").as_string(), 20);
    joint_pub_ = create_publisher<sensor_msgs::msg::JointState>("/joint_states", 20);
    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
    odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
        get_parameter("fastlio_odom_topic").as_string(), 20,
        std::bind(&G1BaseKinematicsNode::odomCallback, this, std::placeholders::_1));

    lowstate_socket_ = ::socket(AF_INET, SOCK_DGRAM, 0);
    if (lowstate_socket_ < 0) throw std::runtime_error("Cannot create lowstate UDP socket");
    const int flags = ::fcntl(lowstate_socket_, F_GETFL, 0);
    ::fcntl(lowstate_socket_, F_SETFL, flags | O_NONBLOCK);
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_port = htons(static_cast<std::uint16_t>(
        get_parameter("lowstate_udp_port").as_int()));
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (::bind(lowstate_socket_, reinterpret_cast<const sockaddr *>(&address),
               sizeof(address)) != 0)
      throw std::runtime_error("Cannot bind lowstate UDP socket");
    lowstate_timer_ = create_wall_timer(
        std::chrono::milliseconds(2),
        std::bind(&G1BaseKinematicsNode::receiveLowState, this));

    RCLCPP_INFO(
        get_logger(),
        "Unitree G1 URDF kinematics: lowstate UDP port=%ld, body=%s, planar navigation=%s",
        get_parameter("lowstate_udp_port").as_int(), base_frame_.c_str(),
        navigation_frame_.c_str());
  }

  ~G1BaseKinematicsNode() override
  {
    if (lowstate_socket_ >= 0) ::close(lowstate_socket_);
  }

private:
  struct KinematicState
  {
    std::array<double, 29> q{};
    uint8_t mode_machine = 0;
    bool received = false;
    std::chrono::steady_clock::time_point receipt_time;
  };

  void receiveLowState()
  {
    unitree_g1_kinematics::LowStatePacket packet;
    while (::recv(lowstate_socket_, &packet, sizeof(packet), 0) == sizeof(packet))
    {
      if (packet.magic != unitree_g1_kinematics::kLowStatePacketMagic ||
          packet.version != unitree_g1_kinematics::kLowStatePacketVersion)
        continue;
      std::lock_guard<std::mutex> lock(state_mutex_);
      state_.q = packet.q;
      state_.mode_machine = packet.mode_machine;
      state_.received = true;
      state_.receipt_time = std::chrono::steady_clock::now();
    }
  }

  bool isRevisionModel(uint8_t mode) const
  {
    if (model_variant_ == "rev_1_0") return true;
    if (model_variant_ == "original") return false;
    switch (mode)
    {
      case 4:
      case 5:
      case 6:
      case 10:
      case 11:
      case 12:
      case 13:
      case 14:
      case 15:
      case 16:
        return true;
      default:
        return false;
    }
  }

  bool hasThreeDofWaist(uint8_t mode) const
  {
    return mode != 1 && mode != 4 && mode != 10;
  }

  bool isWaistLocked(uint8_t mode) const
  {
    return mode == 3 || mode == 6 || mode == 12 || mode == 14 || mode == 16;
  }

  std::vector<Eigen::Isometry3d> localUrdfChain(
      const KinematicState &state, bool revision, bool three_dof,
      bool waist_locked) const
  {
    const double yaw = state.q[12];
    const double roll = three_dof && !waist_locked ? state.q[13] : 0.0;
    const double pitch = three_dof && !waist_locked ? state.q[14] : 0.0;
    std::vector<Eigen::Isometry3d> chain;

    if (three_dof)
    {
      chain.push_back(rotation(Eigen::Vector3d::UnitZ(), yaw));
      chain.push_back(
          translation(-0.0039635, 0.0, revision ? 0.044 : 0.035) *
          rotation(Eigen::Vector3d::UnitX(), roll));
      chain.push_back(
          translation(0.0, 0.0, revision ? 0.0 : 0.019) *
          rotation(Eigen::Vector3d::UnitY(), pitch));
    }
    else
    {
      chain.push_back(
          translation(-0.0039635, 0.0, revision ? 0.044 : 0.054) *
          rotation(Eigen::Vector3d::UnitZ(), yaw));
    }

    // torso_link -> mid360_link exactly follows Unitree's current official
    // mode-specific URDF. Modern/rev_1_0 files include the upside-down roll
    // directly in mid360_joint; deprecated original files do not.
    if (revision)
    {
      chain.push_back(
          translation(0.0002835, 0.00003, 0.428434) *
          rotation(Eigen::Vector3d::UnitY(), kModernMid360Pitch) *
          rotation(Eigen::Vector3d::UnitX(), kPi));
      chain.push_back(Eigen::Isometry3d::Identity());
    }
    else
    {
      chain.push_back(
          translation(0.0002835, 0.00003, 0.40618) *
          rotation(Eigen::Vector3d::UnitY(), kOriginalMid360Pitch));
      chain.push_back(rotation(Eigen::Vector3d::UnitX(), kPi));
    }
    // Official lidar page: pose of the internal IMU in the lidar frame.
    chain.push_back(translation(0.011, 0.02329, -0.04412));
    return chain;
  }

  void odomCallback(const nav_msgs::msg::Odometry::SharedPtr odom)
  {
    const auto receipt_time = std::chrono::steady_clock::now();
    std_msgs::msg::Header navigation_header = odom->header;
    // Unitree's MID-360 DDS publisher uses the sensor clock, which is several
    // seconds behind this ROS host. FAST-LIO must retain that source time for
    // IMU/LiDAR fusion, but Nav2 requires its public odom/TF chain on the ROS
    // clock. Re-stamp only this navigation boundary at receipt time.
    navigation_header.stamp = now();

    KinematicState state;
    {
      std::lock_guard<std::mutex> lock(state_mutex_);
      state = state_;
    }
    if (!state.received)
    {
      RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 5000,
          "Waiting for Unitree rt/lowstate; planar navigation odometry is not published yet.");
      return;
    }
    if (std::chrono::steady_clock::now() - state.receipt_time >
        std::chrono::milliseconds(500))
    {
      RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 5000,
          "Unitree rt/lowstate is stale; planar navigation odometry is paused.");
      return;
    }

    const bool revision = isRevisionModel(state.mode_machine);
    const bool three_dof = hasThreeDofWaist(state.mode_machine);
    const bool waist_locked = isWaistLocked(state.mode_machine);
    const auto chain = localUrdfChain(state, revision, three_dof, waist_locked);
    Eigen::Isometry3d pelvis_to_imu = Eigen::Isometry3d::Identity();
    for (const auto &transform : chain) pelvis_to_imu = pelvis_to_imu * transform;

    Eigen::Isometry3d world_to_imu = Eigen::Isometry3d::Identity();
    const auto &pose = odom->pose.pose;
    Eigen::Quaterniond imu_quaternion(
        pose.orientation.w, pose.orientation.x,
        pose.orientation.y, pose.orientation.z);
    if (imu_quaternion.norm() < 1e-8) imu_quaternion = Eigen::Quaterniond::Identity();
    world_to_imu.linear() = imu_quaternion.normalized().toRotationMatrix();
    world_to_imu.translation() =
        Eigen::Vector3d(pose.position.x, pose.position.y, pose.position.z);
    const Eigen::Isometry3d world_to_pelvis = world_to_imu * pelvis_to_imu.inverse();

    // Nav2 and AMCL are strictly planar.  A humanoid pelvis carries live
    // roll/pitch while standing and walking, so using pelvis base_link as the
    // 2-D navigation base makes AMCL inject the opposite roll/pitch into
    // map->odom.  Keep the real pelvis frame, but expose a yaw-only frame at
    // the same origin for localization, costmaps and velocity control.
    const double navigation_yaw = std::atan2(
      world_to_pelvis.linear()(1, 0), world_to_pelvis.linear()(0, 0));
    Eigen::Isometry3d world_to_navigation = Eigen::Isometry3d::Identity();
    world_to_navigation.translation() = world_to_pelvis.translation();
    world_to_navigation.linear() =
      Eigen::AngleAxisd(navigation_yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix();
    const Eigen::Isometry3d navigation_to_pelvis =
      world_to_navigation.inverse() * world_to_pelvis;

    nav_msgs::msg::Odometry base_odom = *odom;
    base_odom.header = navigation_header;
    base_odom.child_frame_id = navigation_frame_;
    base_odom.pose.pose.position.x = world_to_navigation.translation().x();
    base_odom.pose.pose.position.y = world_to_navigation.translation().y();
    base_odom.pose.pose.position.z = world_to_navigation.translation().z();
    Eigen::Quaterniond base_quaternion(world_to_navigation.linear());
    base_quaternion.normalize();
    base_odom.pose.pose.orientation.w = base_quaternion.w();
    base_odom.pose.pose.orientation.x = base_quaternion.x();
    base_odom.pose.pose.orientation.y = base_quaternion.y();
    base_odom.pose.pose.orientation.z = base_quaternion.z();

    // FAST-LIO's ROS 2 Odometry message leaves twist at zero even though its
    // pose changes. DWB uses this twist for its dynamic window, so derive the
    // planar base-frame velocity from consecutive stabilized poses and reject
    // physically impossible discontinuities instead of feeding them forward.
    base_odom.twist.twist = geometry_msgs::msg::Twist();
    if (previous_pose_valid_)
    {
      const double dt = std::chrono::duration<double>(receipt_time - previous_pose_time_).count();
      if (dt >= 0.02 && dt <= 0.50)
      {
        const Eigen::Vector3d world_delta =
          world_to_navigation.translation() - previous_world_to_navigation_.translation();
        const Eigen::Vector3d body_linear =
          world_to_navigation.linear().transpose() * world_delta / dt;
        const Eigen::Matrix3d relative_rotation =
          previous_world_to_navigation_.linear().transpose() * world_to_navigation.linear();
        const Eigen::AngleAxisd angle_axis(relative_rotation);
        const Eigen::Vector3d body_angular = angle_axis.axis() * angle_axis.angle() / dt;
        const bool plausible =
          std::isfinite(body_linear.x()) && std::isfinite(body_linear.y()) &&
          std::isfinite(body_angular.z()) && body_linear.head<2>().norm() <= 1.0 &&
          std::abs(body_angular.z()) <= 2.0;
        if (plausible)
        {
          const double alpha = twist_initialized_ ? twist_smoothing_alpha_ : 1.0;
          filtered_vx_ += alpha * (body_linear.x() - filtered_vx_);
          filtered_vy_ += alpha * (body_linear.y() - filtered_vy_);
          filtered_wz_ += alpha * (body_angular.z() - filtered_wz_);
          twist_initialized_ = true;
        }
        else
        {
          filtered_vx_ = 0.0;
          filtered_vy_ = 0.0;
          filtered_wz_ = 0.0;
          twist_initialized_ = false;
        }
      }
    }
    previous_world_to_navigation_ = world_to_navigation;
    previous_pose_time_ = receipt_time;
    previous_pose_valid_ = true;
    base_odom.twist.twist.linear.x = filtered_vx_;
    base_odom.twist.twist.linear.y = filtered_vy_;
    base_odom.twist.twist.angular.z = filtered_wz_;
    odom_pub_->publish(base_odom);

    sensor_msgs::msg::JointState joints;
    joints.header = navigation_header;
    joints.name.assign(kJointNames.begin(), kJointNames.end());
    joints.position.assign(state.q.begin(), state.q.end());
    joint_pub_->publish(joints);

    std::vector<geometry_msgs::msg::TransformStamped> transforms;
    transforms.push_back(toTransform(
        world_to_navigation, navigation_header, odom->header.frame_id, navigation_frame_));
    transforms.push_back(toTransform(
        navigation_to_pelvis, navigation_header, navigation_frame_, base_frame_));
    std::size_t index = 0;
    if (three_dof)
    {
      transforms.push_back(toTransform(
          chain[index++], navigation_header, base_frame_, "waist_yaw_link"));
      transforms.push_back(toTransform(
          chain[index++], navigation_header, "waist_yaw_link", "waist_roll_link"));
      transforms.push_back(toTransform(
          chain[index++], navigation_header, "waist_roll_link", "torso_link"));
    }
    else
    {
      transforms.push_back(toTransform(
          chain[index++], navigation_header, base_frame_, "torso_link"));
    }
    transforms.push_back(toTransform(
        chain[index++], navigation_header, "torso_link", "mid360_link"));
    transforms.push_back(toTransform(
        chain[index++], navigation_header, "mid360_link", "livox_frame"));
    transforms.push_back(toTransform(
        chain[index++], navigation_header, "livox_frame", "livox_imu"));
    tf_broadcaster_->sendTransform(transforms);

    if (!model_logged_)
    {
      model_logged_ = true;
      RCLCPP_INFO(
          get_logger(), "G1 mode_machine=%u: %s, %s waist%s",
          static_cast<unsigned>(state.mode_machine),
          revision ? "rev_1_0 URDF" : "original URDF",
          three_dof ? "3-link" : "1-DoF",
          waist_locked ? " (roll/pitch locked)" : "");
    }
  }

  std::string base_frame_;
  std::string navigation_frame_;
  std::string model_variant_;
  std::mutex state_mutex_;
  KinematicState state_;
  bool model_logged_ = false;
  double twist_smoothing_alpha_ = 0.35;
  bool previous_pose_valid_ = false;
  bool twist_initialized_ = false;
  double filtered_vx_ = 0.0;
  double filtered_vy_ = 0.0;
  double filtered_wz_ = 0.0;
  Eigen::Isometry3d previous_world_to_navigation_ = Eigen::Isometry3d::Identity();
  std::chrono::steady_clock::time_point previous_pose_time_;

  int lowstate_socket_ = -1;
  rclcpp::TimerBase::SharedPtr lowstate_timer_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr joint_pub_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
};

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<G1BaseKinematicsNode>());
  rclcpp::shutdown();
  return 0;
}
