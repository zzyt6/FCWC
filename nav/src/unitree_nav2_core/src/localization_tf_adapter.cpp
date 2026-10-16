#include <functional>
#include <memory>
#include <string>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <rclcpp/rclcpp.hpp>
#include <tf2/LinearMath/Transform.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>

class LocalizationTfAdapter : public rclcpp::Node
{
public:
  LocalizationTfAdapter()
  : Node("localization_tf_adapter"), buffer_(get_clock()), listener_(buffer_), broadcaster_(*this)
  {
    declare_parameter<std::string>("global_frame", "map");
    declare_parameter<std::string>("odom_frame", "odom");
    declare_parameter<std::string>("base_frame", "base_link");
    global_frame_ = get_parameter("global_frame").as_string();
    odom_frame_ = get_parameter("odom_frame").as_string();
    base_frame_ = get_parameter("base_frame").as_string();

    pcl_initial_pose_publisher_ = create_publisher<geometry_msgs::msg::PoseStamped>(
      "/pcl_initialpose", rclcpp::SystemDefaultsQoS());
    localization_pose_publisher_ =
      create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>(
        "/localization_pose", rclcpp::QoS(1).transient_local().reliable());
    initial_pose_subscription_ =
      create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
        "/initialpose", rclcpp::SystemDefaultsQoS(),
        std::bind(&LocalizationTfAdapter::initialPoseCallback, this, std::placeholders::_1));
    pose_subscription_ = create_subscription<geometry_msgs::msg::PoseStamped>(
      "/pcl_pose", rclcpp::QoS(1).transient_local().reliable(),
      std::bind(&LocalizationTfAdapter::poseCallback, this, std::placeholders::_1));
  }

private:
  void initialPoseCallback(const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr message)
  {
    if (message->header.frame_id != global_frame_)
    {
      RCLCPP_WARN(get_logger(), "Ignoring initial pose outside %s", global_frame_.c_str());
      return;
    }
    geometry_msgs::msg::PoseStamped output;
    output.header = message->header;
    output.pose = message->pose.pose;
    pcl_initial_pose_publisher_->publish(output);
  }

  void poseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr message)
  {
    try
    {
      const auto odom_to_base = buffer_.lookupTransform(
        odom_frame_, base_frame_, message->header.stamp,
        rclcpp::Duration(0, 150000000));
      tf2::Transform map_base;
      tf2::Transform odom_base;
      tf2::fromMsg(message->pose, map_base);
      tf2::fromMsg(odom_to_base.transform, odom_base);
      const tf2::Transform map_odom = map_base * odom_base.inverse();

      geometry_msgs::msg::TransformStamped transform;
      transform.header = message->header;
      transform.header.frame_id = global_frame_;
      transform.child_frame_id = odom_frame_;
      transform.transform = tf2::toMsg(map_odom);
      broadcaster_.sendTransform(transform);

      geometry_msgs::msg::PoseWithCovarianceStamped pose;
      pose.header = message->header;
      pose.pose.pose = message->pose;
      pose.pose.covariance[0] = 0.04;
      pose.pose.covariance[7] = 0.04;
      pose.pose.covariance[14] = 0.09;
      pose.pose.covariance[21] = 0.03;
      pose.pose.covariance[28] = 0.03;
      pose.pose.covariance[35] = 0.04;
      localization_pose_publisher_->publish(pose);
    }
    catch (const tf2::TransformException &error)
    {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 3000, "Cannot form map->odom: %s", error.what());
    }
  }

  std::string global_frame_;
  std::string odom_frame_;
  std::string base_frame_;
  tf2_ros::Buffer buffer_;
  tf2_ros::TransformListener listener_;
  tf2_ros::TransformBroadcaster broadcaster_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pcl_initial_pose_publisher_;
  rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr
    localization_pose_publisher_;
  rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr
    initial_pose_subscription_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr pose_subscription_;
};

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<LocalizationTfAdapter>());
  rclcpp::shutdown();
  return 0;
}
