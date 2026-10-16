#include <cstdio>
#include <cstdlib>
#include <memory>

#include <rclcpp/rclcpp.hpp>

#include "driver_node.h"


int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<livox_ros::DriverNode>(rclcpp::NodeOptions());
  rclcpp::spin(node);
  if (rclcpp::ok()) {
    rclcpp::shutdown();
  }

  // Destroy the node, stop SDK threads, and release the lidar explicitly.
  node.reset();
  std::fflush(nullptr);

  // The SDK2 copy preinstalled on the G1 links its bundled spdlog into a ROS 2
  // process that also loads the system spdlog.  Its static logger destructor
  // crosses those ABIs and crashes after all useful cleanup has completed.
  // Avoid only the unsafe process-global destructors; node/SDK cleanup above
  // has already run.
  std::_Exit(EXIT_SUCCESS);
}
