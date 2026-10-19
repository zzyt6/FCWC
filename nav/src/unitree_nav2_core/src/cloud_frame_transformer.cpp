#include <chrono>
#include <functional>
#include <deque>
#include <memory>
#include <string>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/time_reference.hpp>
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
      get_parameter("input_topic").as_string(), rclcpp::SensorDataQoS().keep_last(1),
      std::bind(&CloudFrameTransformer::cloudCallback, this, std::placeholders::_1));
    timing_subscription_ = create_subscription<sensor_msgs::msg::TimeReference>(
      "/g1/lidar_time_reference", 20,
      [this](const sensor_msgs::msg::TimeReference::SharedPtr timing) {
        timings_.push_back(*timing);
        while (timings_.size() > 30) {timings_.pop_front();}
        publishPending();
      });
    timer_ = create_wall_timer(std::chrono::milliseconds(5), [this]() {publishPending();});
  }

private:
  void cloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr message)
  {
    pending_.emplace_back(std::chrono::steady_clock::now(), message);
    while (pending_.size() > 5) {pending_.pop_front();}
    publishPending();
  }

  void publishPending()
  {
    // The cloud and odometry come from the same FAST-LIO frame. Use that
    // frame's converted acquisition stamp and waist transform, even if DDS
    // delivers the cloud before its TF/time-reference messages.
    for (auto it = pending_.begin(); it != pending_.end();) {
      const auto & message = it->second;
      bool published = false;
      bool obsolete = false;
      for (const auto & timing : timings_) {
        if (timing.time_ref != message->header.stamp) {continue;}
        const auto stamp = rclcpp::Time(timing.header.stamp);
        const double age = (now()-stamp).seconds();
        if (age > .5 || age < -.03 || stamp.nanoseconds() <= last_published_stamp_) {
          obsolete = true;
          break;
        }
        try {
          const auto transform = buffer_.lookupTransform(
            target_frame_, message->header.frame_id, rclcpp::Time(timing.header.stamp));
          sensor_msgs::msg::PointCloud2 output;
          tf2::doTransform(*message, output, transform);
          output.header.frame_id = target_frame_;
          output.header.stamp = timing.header.stamp;
          publisher_->publish(output);
          last_published_stamp_ = stamp.nanoseconds();
          published = true;
        } catch (const tf2::TransformException &) {
          // Wait without blocking reception of the matching TF.
        }
        break;
      }
      const bool expired = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - it->first).count() > .3;
      if (!published && !expired && !obsolete) {++it; continue;}
      if (expired && !published) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 3000,
          "Point cloud dropped: missing transform/time reference at acquisition time");
      }
      it = pending_.erase(it);
    }
  }

  std::string target_frame_;
  int64_t last_published_stamp_ = 0;
  tf2_ros::Buffer buffer_;
  tf2_ros::TransformListener listener_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr publisher_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription_;
  rclcpp::Subscription<sensor_msgs::msg::TimeReference>::SharedPtr timing_subscription_;
  rclcpp::TimerBase::SharedPtr timer_;
  std::deque<sensor_msgs::msg::TimeReference> timings_;
  std::deque<std::pair<std::chrono::steady_clock::time_point,
    sensor_msgs::msg::PointCloud2::SharedPtr>> pending_;
};

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<CloudFrameTransformer>());
  rclcpp::shutdown();
  return 0;
}
