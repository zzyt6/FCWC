#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>

#include <geometry_msgs/msg/twist.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/bool.hpp>

#include "unitree_g1_kinematics/velocity_packet.hpp"

class G1CmdVelUdpNode : public rclcpp::Node
{
public:
  G1CmdVelUdpNode() : Node("g1_cmd_vel_udp")
  {
    declare_parameter<std::string>("cmd_vel_topic", "/cmd_vel");
    declare_parameter<std::string>("udp_host", "127.0.0.1");
    declare_parameter<int>("udp_port", 39002);
    declare_parameter<double>("max_vx", 0.35);
    declare_parameter<double>("max_vy", 0.20);
    declare_parameter<double>("max_vyaw", 0.50);
    declare_parameter<double>("command_timeout", 0.25);
    declare_parameter<double>("motion_enable_timeout", 1.50);
    max_vx_ = get_parameter("max_vx").as_double();
    max_vy_ = get_parameter("max_vy").as_double();
    max_vyaw_ = get_parameter("max_vyaw").as_double();
    command_timeout_ = get_parameter("command_timeout").as_double();
    motion_enable_timeout_ = get_parameter("motion_enable_timeout").as_double();
    if (motion_enable_timeout_ <= 0.0)
      throw std::runtime_error("motion_enable_timeout must be positive");

    socket_ = ::socket(AF_INET, SOCK_DGRAM, 0);
    if (socket_ < 0) throw std::runtime_error("Cannot create G1 cmd_vel UDP socket");
    destination_.sin_family = AF_INET;
    destination_.sin_port = htons(
      static_cast<std::uint16_t>(get_parameter("udp_port").as_int()));
    if (::inet_pton(
        AF_INET, get_parameter("udp_host").as_string().c_str(),
        &destination_.sin_addr) != 1)
      throw std::runtime_error("Invalid G1 cmd_vel UDP host");

    enabled_status_publisher_ = create_publisher<std_msgs::msg::Bool>(
      "/g1/motion_enabled", rclcpp::QoS(1).transient_local().reliable());
    enable_subscription_ = create_subscription<std_msgs::msg::Bool>(
      "/g1/motion_enable", rclcpp::QoS(1).transient_local().reliable(),
      [this](const std_msgs::msg::Bool::SharedPtr message) {
        last_enable_command_ = std::chrono::steady_clock::now();
        enable_command_received_ = true;
        const bool changed = enabled_ != message->data;
        enabled_ = message->data;
        if (changed || !enabled_)
        {
          command_received_ = false;
          command_ = geometry_msgs::msg::Twist();
        }
        // Every close reaches the executor even if a reopen arrives before
        // the next 50 ms timer. Reopening requires a new cmd_vel callback.
        if (!enabled_) sendPacket(false, 0.0F, 0.0F, 0.0F);
        publishEnabled();
        if (changed)
          RCLCPP_WARN(
            get_logger(), "G1 motion gate is now %s", enabled_ ? "ENABLED" : "DISABLED");
      });
    cmd_subscription_ = create_subscription<geometry_msgs::msg::Twist>(
      get_parameter("cmd_vel_topic").as_string(), 1,
      [this](const geometry_msgs::msg::Twist::SharedPtr message) {
        if (!enabled_) return;
        command_ = *message;
        last_command_ = std::chrono::steady_clock::now();
        command_received_ = true;
      });
    timer_ = create_wall_timer(
      std::chrono::milliseconds(50), std::bind(&G1CmdVelUdpNode::sendCommand, this));
    publishEnabled();
    RCLCPP_WARN(
      get_logger(), "G1 cmd_vel safety gate starts DISABLED; enable only after localization is valid.");
  }

  ~G1CmdVelUdpNode() override
  {
    enabled_ = false;
    command_ = geometry_msgs::msg::Twist();
    for (int count = 0; count < 3; ++count) sendPacket(false, 0.0F, 0.0F, 0.0F);
    if (socket_ >= 0) ::close(socket_);
  }

private:
  static float clamp(double value, double limit)
  {
    if (!std::isfinite(value)) return 0.0F;
    return static_cast<float>(std::max(-limit, std::min(limit, value)));
  }

  void publishEnabled()
  {
    std_msgs::msg::Bool status;
    status.data = enabled_;
    enabled_status_publisher_->publish(status);
  }

  void sendPacket(
    bool enabled, float vx, float vy, float vyaw,
    std::uint8_t reason = unitree_g1_kinematics::kMotionGateClosed)
  {
    unitree_g1_kinematics::VelocityPacket packet;
    packet.sequence = sequence_++;
    packet.enabled = enabled ? 1 : 0;
    packet.reason = enabled ? unitree_g1_kinematics::kVelocityCommand : reason;
    packet.vx = vx;
    packet.vy = vy;
    packet.vyaw = vyaw;
    const auto stamp = command_received_ ? last_command_ : std::chrono::steady_clock::now();
    packet.command_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
      stamp.time_since_epoch()).count();
    ::sendto(
      socket_, &packet, sizeof(packet), MSG_DONTWAIT,
      reinterpret_cast<const sockaddr *>(&destination_), sizeof(destination_));
  }

  void sendCommand()
  {
    const auto now = std::chrono::steady_clock::now();
    const bool authorization_fresh = enable_command_received_ &&
      std::chrono::duration<double>(now - last_enable_command_).count() <=
      motion_enable_timeout_;
    if (enabled_ && !authorization_fresh)
    {
      enabled_ = false;
      publishEnabled();
      RCLCPP_ERROR(
        get_logger(),
        "G1 motion authorization lease timed out; the UDP gate is now DISABLED.");
    }
    const bool fresh = command_received_ &&
      std::chrono::duration<double>(now - last_command_).count() <=
      command_timeout_;
    if (!enabled_ || !fresh)
    {
      sendPacket(
        false, 0.0F, 0.0F, 0.0F,
        enabled_ ? unitree_g1_kinematics::kVelocityTimeout :
        unitree_g1_kinematics::kMotionGateClosed);
      if (enabled_ && !fresh)
        RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 3000, "cmd_vel timed out; G1 stop command is active.");
      return;
    }
    sendPacket(
      true,
      clamp(command_.linear.x, max_vx_),
      clamp(command_.linear.y, max_vy_),
      clamp(command_.angular.z, max_vyaw_));
  }

  int socket_ = -1;
  sockaddr_in destination_{};
  std::uint64_t sequence_ = 0;
  bool enabled_ = false;
  bool enable_command_received_ = false;
  bool command_received_ = false;
  double max_vx_ = 0.35;
  double max_vy_ = 0.20;
  double max_vyaw_ = 0.50;
  double command_timeout_ = 0.25;
  double motion_enable_timeout_ = 1.50;
  geometry_msgs::msg::Twist command_;
  std::chrono::steady_clock::time_point last_command_;
  std::chrono::steady_clock::time_point last_enable_command_;
  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr enabled_status_publisher_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr enable_subscription_;
  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr cmd_subscription_;
};

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<G1CmdVelUdpNode>());
  rclcpp::shutdown();
  return 0;
}
