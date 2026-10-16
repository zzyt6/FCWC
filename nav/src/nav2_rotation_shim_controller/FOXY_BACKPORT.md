# Nav2 Rotation Shim Controller: Foxy backport

This package is based on the official Navigation2 `nav2_rotation_shim_controller`
from release tag `1.0.12`:

https://github.com/ros-navigation/navigation2/tree/1.0.12/nav2_rotation_shim_controller

Nav2 did not ship this plugin in Foxy 0.4.7. The controller behavior is kept from
the official implementation; only the following API compatibility changes were
made:

- `configure()` accepts Foxy's lifecycle-node `SharedPtr`.
- `computeVelocityCommands()` uses Foxy's two-argument controller interface.
- `setSpeedLimit()` is omitted because that interface does not exist in Foxy.
- The primary-controller parameter uses an empty-string default because Foxy's
  parameter helper cannot declare a type-only default.
- The later `nav2_util::transformPoseInTargetFrame()` helper is replaced by the
  equivalent Foxy `tf2_ros::Buffer::transform()` call.
- The sampled path point requests TF's latest complete transform chain. This
  avoids waiting for a future scan-driven transform on every controller cycle
  when the controller runs faster than the navigation TF publishers.
- The official newer `closed_loop` option and last-command acceleration state
  are backported. This permits open-loop command ramping for bases whose
  locomotion controller does not respond to the first very small velocity step.
- The official stopping-velocity limit is backported to avoid overshooting a
  requested heading while using last-command acceleration state.
- Upstream Galactic-only tests are not vendored; this package uses the available
  Foxy linters and is integration-tested by loading it through controller_server.

The implementation remains under its original Apache-2.0 license and copyright.
