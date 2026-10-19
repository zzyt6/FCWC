cmake_minimum_required(VERSION 3.8)
project(unitree_g1_kinematics)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

find_package(ament_cmake REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(unitree_sdk2 REQUIRED)

add_executable(g1_base_kinematics_node src/g1_base_kinematics_node.cpp)
target_include_directories(g1_base_kinematics_node PRIVATE include)
ament_target_dependencies(g1_base_kinematics_node
  geometry_msgs nav_msgs rclcpp sensor_msgs std_msgs tf2_ros)
target_link_libraries(g1_base_kinematics_node Eigen3::Eigen)

# Unitree SDK2 and ROS 2 Foxy ship ABI-incompatible CycloneDDS versions. Keep
# the SDK reader in its own process and forward only joint data over loopback.
add_executable(g1_lowstate_udp_bridge src/g1_lowstate_udp_bridge.cpp)
target_include_directories(g1_lowstate_udp_bridge PRIVATE include)
target_link_libraries(g1_lowstate_udp_bridge unitree_sdk2)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_state_timing test/test_state_timing.cpp)
  target_include_directories(test_state_timing PRIVATE include)
endif()

add_executable(g1_cmd_vel_udp_node src/g1_cmd_vel_udp_node.cpp)
target_include_directories(g1_cmd_vel_udp_node PRIVATE include)
ament_target_dependencies(g1_cmd_vel_udp_node geometry_msgs rclcpp std_msgs)

# /cmd_vel is executed by nav/locomotion/g1_locomotion_executor.py (our own
# Balance/Homie locomotion with the pelvis heading outer loop), not by
# Unitree's high-level LocoClient.

install(TARGETS
  g1_base_kinematics_node
  g1_lowstate_udp_bridge
  g1_cmd_vel_udp_node
  DESTINATION lib/${PROJECT_NAME})

ament_package()
