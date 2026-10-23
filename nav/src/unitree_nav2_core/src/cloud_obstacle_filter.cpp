#include <chrono>
#include <cmath>
#include <deque>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>

#include <Eigen/Core>
#include <pcl/common/point_tests.h>
#include <pcl/common/transforms.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl_conversions/pcl_conversions.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <tf2_eigen/tf2_eigen.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

class CloudObstacleFilter : public rclcpp::Node
{
public:
  CloudObstacleFilter()
  : Node("cloud_obstacle_filter"), buffer_(get_clock()), listener_(buffer_)
  {
    declare_parameter<std::string>("input_topic", "/cloud_nav");
    declare_parameter<std::string>("output_topic", "/cloud_obstacles");
    declare_parameter<double>("ground_distance_threshold", 0.06);
    declare_parameter<double>("min_obstacle_height", 0.15);
    declare_parameter<double>("max_obstacle_height", 1.80);
    declare_parameter<double>("self_filter_radius", 0.60);
    declare_parameter<double>("max_range", 4.0);
    // A 50 Hz MID-360 frame (~4000 points) covers only part of the
    // non-repetitive pattern; one frame leaves most scan bins empty and too
    // few ground points for RANSAC. Merge the frames acquired in this window,
    // each placed in the odometry frame at its own acquisition time. Output
    // keeps the newest frame's stamp and rate. 10 Hz input = one frame.
    declare_parameter<double>("accumulation_window", 0.1);
    declare_parameter<std::string>("odom_frame", "odom");
    ground_distance_threshold_ = get_parameter("ground_distance_threshold").as_double();
    min_obstacle_height_ = get_parameter("min_obstacle_height").as_double();
    max_obstacle_height_ = get_parameter("max_obstacle_height").as_double();
    self_filter_radius_ = get_parameter("self_filter_radius").as_double();
    max_range_ = get_parameter("max_range").as_double();
    accumulation_window_ = get_parameter("accumulation_window").as_double();
    odom_frame_ = get_parameter("odom_frame").as_string();
    if (self_filter_radius_ < 0.0 || max_range_ <= self_filter_radius_ ||
        min_obstacle_height_ >= max_obstacle_height_ || !(accumulation_window_ >= 0.0) ||
        accumulation_window_ > 0.3)
    {
      throw std::invalid_argument("Invalid obstacle filter range, height or window bounds");
    }

    publisher_ = create_publisher<sensor_msgs::msg::PointCloud2>(
      get_parameter("output_topic").as_string(), rclcpp::SensorDataQoS());
    subscription_ = create_subscription<sensor_msgs::msg::PointCloud2>(
      get_parameter("input_topic").as_string(), rclcpp::SensorDataQoS().keep_last(1),
      std::bind(&CloudObstacleFilter::callback, this, std::placeholders::_1));
  }

private:
  struct Frame
  {
    rclcpp::Time stamp;
    pcl::PointCloud<pcl::PointXYZI>::Ptr odom_points;
  };

  // Returns the newest frame plus earlier frames in the window, all expressed
  // in the newest frame's coordinates at its acquisition time. Without a
  // valid odometry transform, fall back to the single current frame.
  pcl::PointCloud<pcl::PointXYZI>::Ptr accumulate(
    const pcl::PointCloud<pcl::PointXYZI>::Ptr & current,
    const std::string & frame_id, const rclcpp::Time & stamp)
  {
    if (accumulation_window_ <= 0.0) return current;
    Eigen::Isometry3d odom_from_frame;
    try {
      odom_from_frame = tf2::transformToEigen(
        buffer_.lookupTransform(odom_frame_, frame_id, stamp));
    } catch (const tf2::TransformException &) {
      frames_.clear();
      return current;
    }
    if (frame_id != frame_id_) frames_.clear();
    frame_id_ = frame_id;
    pcl::PointCloud<pcl::PointXYZI>::Ptr in_odom(new pcl::PointCloud<pcl::PointXYZI>());
    pcl::transformPointCloud(*current, *in_odom, odom_from_frame.matrix().cast<float>());
    frames_.push_back({stamp, in_odom});
    while (!frames_.empty() && (stamp-frames_.front().stamp).seconds() > accumulation_window_ + 1e-3)
      frames_.pop_front();
    if (frames_.size() == 1) return current;
    pcl::PointCloud<pcl::PointXYZI> merged;
    for (const auto & frame : frames_) merged += *frame.odom_points;
    pcl::PointCloud<pcl::PointXYZI>::Ptr output(new pcl::PointCloud<pcl::PointXYZI>());
    pcl::transformPointCloud(merged, *output, odom_from_frame.inverse().matrix().cast<float>());
    return output;
  }
  bool estimateGround(
    const pcl::PointCloud<pcl::PointXYZI>::ConstPtr &cloud,
    Eigen::Vector4f &plane)
  {
    pcl::PointCloud<pcl::PointXYZI>::Ptr candidates(
      new pcl::PointCloud<pcl::PointXYZI>());
    candidates->reserve(cloud->size());
    for (const auto &point : cloud->points)
    {
      if (!pcl::isFinite(point)) continue;
      const double range = std::hypot(point.x, point.y);
      if (range >= self_filter_radius_ && range <= max_range_ && point.z < -0.25F)
        candidates->push_back(point);
    }
    if (candidates->size() < 80) return false;

    pcl::SACSegmentation<pcl::PointXYZI> segmentation;
    segmentation.setOptimizeCoefficients(true);
    segmentation.setModelType(pcl::SACMODEL_PERPENDICULAR_PLANE);
    segmentation.setMethodType(pcl::SAC_RANSAC);
    segmentation.setAxis(Eigen::Vector3f::UnitZ());
    segmentation.setEpsAngle(0.35);
    segmentation.setDistanceThreshold(ground_distance_threshold_);
    segmentation.setMaxIterations(500);
    segmentation.setInputCloud(candidates);
    pcl::PointIndices inliers;
    pcl::ModelCoefficients coefficients;
    segmentation.segment(inliers, coefficients);
    if (coefficients.values.size() != 4 || inliers.indices.size() < 80) return false;

    plane = Eigen::Vector4f(
      coefficients.values[0], coefficients.values[1],
      coefficients.values[2], coefficients.values[3]);
    const float norm = plane.head<3>().norm();
    if (norm < 1e-6F) return false;
    plane /= norm;
    if (plane.z() < 0.0F) plane = -plane;
    const double floor_z_at_base = -plane.w() / plane.z();
    if (plane.z() < std::cos(0.35) || floor_z_at_base < -1.30 || floor_z_at_base > -0.40)
      return false;
    return true;
  }

  void callback(const sensor_msgs::msg::PointCloud2::SharedPtr message)
  {
    const auto stamp = rclcpp::Time(message->header.stamp);
    const double age = (now()-stamp).seconds();
    if (stamp.nanoseconds() <= last_stamp_ || age < -.03 || age > .5) return;
    last_stamp_ = stamp.nanoseconds();
    pcl::PointCloud<pcl::PointXYZI>::Ptr input(
      new pcl::PointCloud<pcl::PointXYZI>());
    pcl::fromROSMsg(*message, *input);
    input = accumulate(input, message->header.frame_id, stamp);

    Eigen::Vector4f plane;
    if (estimateGround(input, plane))
    {
      last_plane_ = plane;
      last_plane_time_ = std::chrono::steady_clock::now() -
        std::chrono::duration_cast<std::chrono::steady_clock::duration>(
          std::chrono::duration<double>(age));
      have_plane_ = true;
    }
    else if (!have_plane_ ||
             std::chrono::steady_clock::now() - last_plane_time_ > std::chrono::seconds(2))
    {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 3000,
        "No reliable live ground plane; obstacle observations paused.");
      // An empty cloud becomes an all-infinity scan and would falsely clear
      // obstacles. Let the observation deadline expire instead.
      return;
    }
    plane = last_plane_;

    pcl::PointCloud<pcl::PointXYZI>::Ptr obstacles(
      new pcl::PointCloud<pcl::PointXYZI>());
    obstacles->reserve(input->size());
    for (const auto &point : input->points)
    {
      if (!pcl::isFinite(point)) continue;
      const double range = std::hypot(point.x, point.y);
      if (range < self_filter_radius_ || range > max_range_) continue;
      const double height =
        plane.x() * point.x + plane.y() * point.y + plane.z() * point.z + plane.w();
      if (height >= min_obstacle_height_ && height <= max_obstacle_height_)
        obstacles->push_back(point);
    }
    obstacles->width = static_cast<std::uint32_t>(obstacles->size());
    obstacles->height = 1;
    obstacles->is_dense = false;
    sensor_msgs::msg::PointCloud2 output;
    pcl::toROSMsg(*obstacles, output);
    output.header = message->header;
    if ((now()-stamp).seconds() <= .5) publisher_->publish(output);

    RCLCPP_INFO_THROTTLE(
      get_logger(), *get_clock(), 3000,
      "Live ground z@base=%.3f m; obstacle cloud %zu/%zu points.",
      -plane.w() / plane.z(), obstacles->size(), input->size());
  }

  double ground_distance_threshold_ = 0.06;
  double min_obstacle_height_ = 0.15;
  double max_obstacle_height_ = 1.80;
  double self_filter_radius_ = 0.60;
  double max_range_ = 4.0;
  double accumulation_window_ = 0.1;
  std::string odom_frame_ = "odom";
  std::string frame_id_;
  std::deque<Frame> frames_;
  tf2_ros::Buffer buffer_;
  tf2_ros::TransformListener listener_;
  bool have_plane_ = false;
  int64_t last_stamp_ = 0;
  Eigen::Vector4f last_plane_ = Eigen::Vector4f::Zero();
  std::chrono::steady_clock::time_point last_plane_time_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr publisher_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription_;
};

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<CloudObstacleFilter>());
  rclcpp::shutdown();
  return 0;
}
