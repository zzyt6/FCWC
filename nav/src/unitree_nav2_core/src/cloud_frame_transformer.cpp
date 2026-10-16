#include <chrono>
#include <functional>
#include <memory>
#include <string>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_sensor_msgs/tf2_sensor_msgs.h>

class CloudFrameTransformer : public rclcpp::Node
{
public:
  CloudFrameTransformer()
  : Node("cloud_frame_transformer"), buffer_(get_clock()), listener_(buffer_)
  {
    declare_parameter<std::string>("input_topic", "/cloud_registered_body");
    declare_parameter<std::string>("output_topic", "/cloud_nav");
    declare_parameter<std::string>("target_frame", "base_link");
    target_frame_ = get_parameter("target_frame").as_string();
    publisher_ = create_publisher<sensor_msgs::msg::PointCloud2>(
      get_parameter("output_topic").as_string(), rclcpp::SensorDataQoS());
    subscription_ = create_subscription<sensor_msgs::msg::PointCloud2>(
      get_parameter("input_topic").as_string(), rclcpp::SensorDataQoS(),
      std::bind(&CloudFrameTransformer::cloudCallback, this, std::placeholders::_1));
  }

private:
  void cloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr message)
  {
    try
    {
      const auto transform = buffer_.lookupTransform(
        target_frame_, message->header.frame_id, rclcpp::Time(0),
        rclcpp::Duration(0, 100000000));
      sensor_msgs::msg::PointCloud2 output;
      tf2::doTransform(*message, output, transform);
      output.header.frame_id = target_frame_;
      // Keep the original sensor time inside FAST-LIO, but put the current
      // body-frame scan onto the ROS clock before AMCL, costmaps, and Nav2.
      output.header.stamp = now();
      publisher_->publish(output);
    }
    catch (const tf2::TransformException &error)
    {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 3000, "Point cloud TF unavailable: %s", error.what());
    }
  }

  std::string target_frame_;
  tf2_ros::Buffer buffer_;
  tf2_ros::TransformListener listener_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr publisher_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription_;
};

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<CloudFrameTransformer>());
  rclcpp::shutdown();
  return 0;
}
