# Upstream source snapshot

- `src/fast_lio`: `Ericsii/FAST_LIO_ROS2`, branch `ros2`, downloaded on
  2026-07-19. Large paper/demo assets under `doc/` and `PCD/` were omitted;
  all build, launch, message, configuration, and source files are present.
- `src/fast_lio/include/ikd-Tree`: `hku-mars/ikd-Tree`, branch `fast_lio`.
- `src/livox_ros_driver2`: `Livox-SDK/livox_ros_driver2`, branch `master`,
  downloaded on 2026-07-19. It is retained only because the ROS 2 FAST-LIO
  source has a compile-time dependency on its `CustomMsg`; no Livox driver node
  is started by this baseline.
- G1 kinematic constants were transcribed from Unitree's current official
  `unitreerobotics/unitree_ros/robots/g1_description` URDFs. The tested robot
  reports `mode_machine=6`, selecting `g1_29dof_lock_waist_rev_1_0.urdf`.
- The Mid360-to-internal-IMU transform and native LiDAR DDS topic definitions
  come from Unitree's official G1 LiDAR service interface.
- `src/pointcloud_to_laserscan`: unchanged official
  `ros-perception/pointcloud_to_laserscan` tag `2.0.0`, commit
  `024e70ae2e0e845bf6525c5fcc540f7b4cc9d08c`.
- `src/rosbridge_suite`: unchanged `RobotWebTools/rosbridge_suite` tag `1.3.1`,
  commit `875710dbef037f09e3caf62128481852c8fe5aea`.
- Navigation uses the official ROS 2 Foxy Navigation2 `0.4.7` Debian packages.
- Offline PCD conversion links Ubuntu 20.04's upstream PCL `1.10.0` and OctoMap
  `1.9.3`. The local `unitree_nav2_core` package only supplies the integration,
  file conversion, projection policy, and frame adapters; upstream libraries
  are not patched.

Local integration changes are intentionally scoped: ROS 2 Foxy build fixes,
RViz disabled by default, Unitree's native Mid-360 DDS PointCloud2 adapter,
gravity-aligned `map`, official G1 waist-to-lidar kinematics, 10 m indoor
parameters, `/Laser_map`-consistent final PCD saving, and the Viser bridge
package, plus the local PCD-to-Nav2/Canvas/G1 safety integration packages.
Duplicate-heavy registered-frame PCD archival remains opt-in.
