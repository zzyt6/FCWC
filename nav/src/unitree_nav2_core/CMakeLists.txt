cmake_minimum_required(VERSION 3.8)
project(unitree_nav2_core)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

find_package(ament_cmake REQUIRED)
find_package(dwb_core REQUIRED)
find_package(dwb_plugins REQUIRED)
find_package(nav2_core REQUIRED)
find_package(nav2_costmap_2d REQUIRED)
find_package(nav2_regulated_pure_pursuit_controller REQUIRED)
find_package(nav_2d_utils REQUIRED)
find_package(nav2_util REQUIRED)
find_package(pluginlib REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(octomap REQUIRED)
find_package(PCL REQUIRED COMPONENTS common filters io sample_consensus segmentation)
find_package(pcl_conversions REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_sensor_msgs REQUIRED)

add_library(navigation_speed SHARED src/navigation_speed.cpp)
target_include_directories(navigation_speed PUBLIC include)
ament_target_dependencies(navigation_speed dwb_core dwb_plugins nav2_core nav_2d_utils nav2_util rclcpp pluginlib)
pluginlib_export_plugin_description_file(dwb_core navigation_speed_plugins.xml)
pluginlib_export_plugin_description_file(nav2_core navigation_controller_plugins.xml)
# The cart follows the global path with Foxy's official Regulated Pure Pursuit.
add_library(cart_pure_pursuit SHARED src/cart_pure_pursuit.cpp)
target_include_directories(cart_pure_pursuit PUBLIC include)
ament_target_dependencies(cart_pure_pursuit nav2_core nav2_costmap_2d
  nav2_regulated_pure_pursuit_controller nav2_util pluginlib rclcpp tf2)
install(TARGETS navigation_speed cart_pure_pursuit LIBRARY DESTINATION lib)
install(DIRECTORY include/ DESTINATION include)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_navigation_speed test/test_navigation_speed.cpp)
  target_include_directories(test_navigation_speed PRIVATE include)
  target_link_libraries(test_navigation_speed navigation_speed)
  ament_target_dependencies(test_navigation_speed dwb_core dwb_plugins)
  ament_add_gtest(test_cart_pure_pursuit test/test_cart_pure_pursuit.cpp)
  target_link_libraries(test_cart_pure_pursuit cart_pure_pursuit)
  ament_target_dependencies(test_cart_pure_pursuit tf2)
  # Invoked by test_cart_rpp_turns.py with the production CartPush parameters.
  add_executable(cart_rpp_regression test/cart_rpp_regression.cpp)
  target_link_libraries(cart_rpp_regression cart_pure_pursuit)
  ament_target_dependencies(cart_rpp_regression
    nav2_costmap_2d nav2_util rclcpp tf2 tf2_ros)
endif()

add_executable(pcd_to_nav_map src/pcd_to_nav_map.cpp)
target_include_directories(pcd_to_nav_map PRIVATE ${PCL_INCLUDE_DIRS} ${OCTOMAP_INCLUDE_DIRS})
target_link_directories(pcd_to_nav_map PRIVATE ${PCL_LIBRARY_DIRS})
target_compile_definitions(pcd_to_nav_map PRIVATE ${PCL_DEFINITIONS})
target_link_libraries(pcd_to_nav_map ${PCL_LIBRARIES} ${OCTOMAP_LIBRARIES})

add_executable(cloud_frame_transformer src/cloud_frame_transformer.cpp)
ament_target_dependencies(cloud_frame_transformer
  rclcpp sensor_msgs tf2_ros tf2_sensor_msgs)

add_executable(cloud_obstacle_filter src/cloud_obstacle_filter.cpp)
target_include_directories(cloud_obstacle_filter PRIVATE ${PCL_INCLUDE_DIRS})
target_link_directories(cloud_obstacle_filter PRIVATE ${PCL_LIBRARY_DIRS})
target_compile_definitions(cloud_obstacle_filter PRIVATE ${PCL_DEFINITIONS})
target_link_libraries(cloud_obstacle_filter ${PCL_LIBRARIES})
ament_target_dependencies(cloud_obstacle_filter pcl_conversions rclcpp sensor_msgs tf2_eigen tf2_ros)

add_executable(localization_tf_adapter src/localization_tf_adapter.cpp)
ament_target_dependencies(localization_tf_adapter
  geometry_msgs rclcpp tf2 tf2_geometry_msgs tf2_ros)

install(TARGETS
  pcd_to_nav_map
  cloud_frame_transformer
  cloud_obstacle_filter
  localization_tf_adapter
  DESTINATION lib/${PROJECT_NAME})

ament_package()
