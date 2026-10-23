#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>

#include <geometry_msgs/msg/twist.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/u_int32.hpp>

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
    declare_parameter<double>("motion_enable_timeout", 3.0);
    declare_parameter<bool>("require_navigation_reference", true);
    // A brief reference outage only pauses velocity (zero is sent at once).
    // Only a sustained outage revokes the operator's motion authorization.
    declare_parameter<double>("reference_fresh_timeout", 0.5);
    declare_parameter<double>("reference_fault_timeout", 3.0);
    declare_parameter<double>("cart_min_vx", -0.2);
    declare_parameter<double>("cart_max_vx", 0.6);
    declare_parameter<double>("cart_max_vy", 0.3);
    declare_parameter<double>("cart_max_vyaw", 0.4);
    require_reference_ = get_parameter("require_navigation_reference").as_bool();
    reference_fresh_timeout_ = get_parameter("reference_fresh_timeout").as_double();
    reference_fault_timeout_ = get_parameter("reference_fault_timeout").as_double();
    if (!(reference_fresh_timeout_ > 0.0) || !(reference_fault_timeout_ > 0.0))
      throw std::runtime_error("Navigation reference timeouts must be positive");
    cart_min_vx_ = get_parameter("cart_min_vx").as_double();
    cart_max_vx_ = get_parameter("cart_max_vx").as_double();
    cart_max_vy_ = get_parameter("cart_max_vy").as_double();
    cart_max_vyaw_ = get_parameter("cart_max_vyaw").as_double();
    if (!std::isfinite(cart_min_vx_) || !std::isfinite(cart_max_vx_) ||
      !std::isfinite(cart_max_vy_) || !std::isfinite(cart_max_vyaw_) ||
      cart_min_vx_ >= 0.0 || cart_max_vx_ <= 0.0 || cart_max_vy_ <= 0.0 || cart_max_vyaw_ <= 0.0)
      throw std::runtime_error("Invalid cart velocity limits");
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
    epoch_subscription_ = create_subscription<std_msgs::msg::UInt32>(
      "/g1/navigation_epoch", rclcpp::QoS(1).transient_local().reliable(),
      [this](const std_msgs::msg::UInt32::SharedPtr message) {
        if (message->data != reference_epoch_) {
          if (enabled_)
            RCLCPP_WARN(get_logger(), "Navigation epoch changed %u -> %u; motion gate DISABLED.",
              reference_epoch_, message->data);
          reference_epoch_ = message->data;
          enabled_ = command_received_ = false;
          sendPacket(false, 0.0F, 0.0F, 0.0F);
          publishEnabled();
        }
      });
    reference_subscription_ = create_subscription<std_msgs::msg::Bool>(
      "/g1/navigation_reference_ready", rclcpp::QoS(1).transient_local().reliable(),
      [this](const std_msgs::msg::Bool::SharedPtr message, const rclcpp::MessageInfo & info) {
        const auto sampled = sourceTime(info, reference_fresh_timeout_);
        if (message->data && !sampled) return;
        checkReference();
        reference_ready_ = message->data;
        if (sampled) last_reference_ = *sampled;
        checkReference();
        if (!reference_ready_ && require_reference_) {
          command_received_ = false;
          sendPacket(false, 0.0F, 0.0F, 0.0F,
            enabled_ ? unitree_g1_kinematics::kVelocityTimeout : unitree_g1_kinematics::kMotionGateClosed);
        }
      });
    enable_subscription_ = create_subscription<std_msgs::msg::Bool>(
      "/g1/motion_enable", rclcpp::QoS(1).transient_local().reliable(),
      [this](const std_msgs::msg::Bool::SharedPtr message, const rclcpp::MessageInfo & info) {
        const auto sampled = sourceTime(info, motion_enable_timeout_);
        if (message->data && !sampled) return;
        if (sampled) last_enable_command_ = *sampled;
        enable_command_received_ = true;
        checkReference();
        if (!message->data) reference_fault_latched_ = false;
        const bool was_enabled = enabled_;
        // Renew existing authorization through a brief sensor interruption;
        // opening a closed gate still requires a valid current reference.
        enabled_ = message->data && !reference_fault_latched_ && (enabled_ || referenceFresh());
        const bool changed = was_enabled != enabled_;
        if (changed && enabled_) accept_commands_after_ = std::chrono::steady_clock::now();
        if (!enabled_) reference_unavailable_ = false;
        if (changed || !enabled_)
        {
          command_received_ = false;
          command_ = geometry_msgs::msg::Twist();
        }
        // Every close reaches the executor even if a reopen arrives before
        // the next 20 ms timer. Reopening requires a new cmd_vel callback.
        if (!enabled_) sendPacket(false, 0.0F, 0.0F, 0.0F);
        publishEnabled();
        if (changed) {
          // Name the cause of every close so a gate drop is never anonymous.
          const auto & gid = info.get_rmw_message_info().publisher_gid.data;
          char publisher[17];
          std::snprintf(publisher, sizeof(publisher), "%02x%02x%02x%02x%02x%02x%02x%02x",
            gid[0], gid[1], gid[2], gid[3], gid[4], gid[5], gid[6], gid[7]);
          RCLCPP_WARN(
            get_logger(), "G1 motion gate is now %s (request=%s latched=%d reference_fresh=%d publisher=%s)",
            enabled_ ? "ENABLED" : "DISABLED", message->data ? "enable" : "disable",
            reference_fault_latched_, referenceFresh(), publisher);
        }
      });
    cmd_subscription_ = create_subscription<geometry_msgs::msg::Twist>(
      get_parameter("cmd_vel_topic").as_string(), 1,
      [this](const geometry_msgs::msg::Twist::SharedPtr message, const rclcpp::MessageInfo & info) {
        checkReference();
        if (!enabled_ || !referenceFresh()) return;
        const auto sampled = sourceTime(info, command_timeout_);
        if (!sampled || *sampled <= accept_commands_after_ ||
          (command_received_ && *sampled <= last_command_)) return;
        command_ = *message;
        last_command_ = *sampled;
        command_received_ = true;
      });
    timer_ = create_wall_timer(
      std::chrono::milliseconds(20), std::bind(&G1CmdVelUdpNode::sendCommand, this));
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
  static std::optional<std::chrono::steady_clock::time_point> sourceTime(
    const rclcpp::MessageInfo & info, double max_age)
  {
    // Twist/Bool have no Header. DDS publication time prevents a queued
    // message from acquiring a fresh lease when its callback finally runs.
    const auto source = info.get_rmw_message_info().source_timestamp;
    const auto wall = std::chrono::system_clock::now().time_since_epoch();
    const double age = std::chrono::duration<double>(wall).count() - double(source)*1e-9;
    if (source <= 0 || age < -.001 || age > max_age) return std::nullopt;
    return std::chrono::steady_clock::now() -
      std::chrono::duration_cast<std::chrono::steady_clock::duration>(
        std::chrono::duration<double>(std::max(0., age)));
  }

  void checkReference()
  {
    const auto now = std::chrono::steady_clock::now();
    if (!enabled_ || referenceFresh()) {
      reference_unavailable_ = false;
      return;
    }
    command_received_ = false;
    if (!reference_unavailable_) {
      reference_unavailable_ = true;
      reference_unavailable_since_ = now;
      RCLCPP_WARN(get_logger(), "Navigation reference unavailable; velocity paused at zero.");
    }
    if (std::chrono::duration<double>(now-reference_unavailable_since_).count() >=
      reference_fault_timeout_)
    {
      enabled_ = false;
      reference_fault_latched_ = true;
      publishEnabled();
      RCLCPP_ERROR(get_logger(), "Navigation reference unavailable for %.1f s; motion gate DISABLED.",
        reference_fault_timeout_);
    }
  }

  bool referenceFresh() const
  {
    return !require_reference_ || (reference_epoch_ != 0 && reference_ready_ &&
      std::chrono::duration<double>(std::chrono::steady_clock::now()-last_reference_).count() <= reference_fresh_timeout_);
  }

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
    packet.navigation_epoch = reference_epoch_;
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
    checkReference();
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
    if (!enabled_ || !fresh || !referenceFresh())
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
    const bool cart = (reference_epoch_ & 1U) != 0;
    const double vx = std::isfinite(command_.linear.x) ? command_.linear.x : 0.0;
    sendPacket(
      true,
      cart ? static_cast<float>(std::max(cart_min_vx_, std::min(cart_max_vx_, vx))) : clamp(vx, max_vx_),
      clamp(command_.linear.y, cart ? cart_max_vy_ : max_vy_),
      clamp(command_.angular.z, cart ? cart_max_vyaw_ : max_vyaw_));
  }

  int socket_ = -1;
  sockaddr_in destination_{};
  std::uint64_t sequence_ = 0;
  bool enabled_ = false;
  bool enable_command_received_ = false;
  bool command_received_ = false;
  bool require_reference_ = true;
  bool reference_ready_ = false;
  bool reference_unavailable_ = false;
  bool reference_fault_latched_ = false;
  std::chrono::steady_clock::time_point reference_unavailable_since_;
  std::uint32_t reference_epoch_ = 0;
  double cart_min_vx_ = -0.2, cart_max_vx_ = 0.6, cart_max_vy_ = 0.3, cart_max_vyaw_ = 0.4;
  std::chrono::steady_clock::time_point last_reference_;
  double max_vx_ = 0.35;
  double max_vy_ = 0.20;
  double max_vyaw_ = 0.50;
  double command_timeout_ = 0.25;
  double motion_enable_timeout_ = 3.0;
  double reference_fresh_timeout_ = 0.5;
  double reference_fault_timeout_ = 3.0;
  geometry_msgs::msg::Twist command_;
  std::chrono::steady_clock::time_point last_command_;
  std::chrono::steady_clock::time_point accept_commands_after_;
  std::chrono::steady_clock::time_point last_enable_command_;
  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr enabled_status_publisher_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr enable_subscription_;
  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr cmd_subscription_;
  rclcpp::Subscription<std_msgs::msg::UInt32>::SharedPtr epoch_subscription_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr reference_subscription_;
};

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<G1CmdVelUdpNode>());
  rclcpp::shutdown();
  return 0;
}
