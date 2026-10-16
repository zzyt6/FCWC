cmake_minimum_required(VERSION 3.5)
project(nav2_pose_progress_checker)

find_package(ament_cmake REQUIRED)
find_package(nav2_common REQUIRED)
find_package(nav2_controller REQUIRED)
find_package(nav2_core REQUIRED)
find_package(nav2_util REQUIRED)
find_package(nav_2d_utils REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(pluginlib REQUIRED)
find_package(angles REQUIRED)

nav2_package()
set(CMAKE_CXX_STANDARD 17)

include_directories(include)

set(dependencies
  angles
  geometry_msgs
  nav_2d_utils
  nav2_controller
  nav2_core
  nav2_util
  pluginlib
  rclcpp
  rclcpp_lifecycle
)

set(library_name pose_progress_checker)

add_library(${library_name} SHARED
  src/pose_progress_checker.cpp)
ament_target_dependencies(${library_name}
  ${dependencies})

install(TARGETS ${library_name}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin)

install(DIRECTORY include/
  DESTINATION include/)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  # Foxy's cppcheck 1.90 cannot parse GoogleTest's TEST macro, although the
  # test target compiles and runs normally. Keep source/header analysis enabled.
  set(ament_cmake_cppcheck_ADDITIONAL_EXCLUDE
    "${CMAKE_CURRENT_SOURCE_DIR}/test")
  ament_lint_auto_find_test_dependencies()

  ament_add_gtest(test_pose_progress_checker
    test/test_pose_progress_checker.cpp)
  target_link_libraries(test_pose_progress_checker ${library_name})
  ament_target_dependencies(test_pose_progress_checker
    ${dependencies})
endif()

ament_export_include_directories(include)
ament_export_libraries(${library_name})
ament_export_dependencies(${dependencies})

pluginlib_export_plugin_description_file(nav2_core pose_progress_checker.xml)

ament_package()
