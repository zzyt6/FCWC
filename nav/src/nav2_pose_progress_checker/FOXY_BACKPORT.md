# Nav2 Pose Progress Checker: Foxy backport

This package is based on Navigation2's official Humble implementation:

https://github.com/ros-navigation/navigation2/tree/humble/nav2_controller

Foxy 0.4.7 ships only `SimpleProgressChecker`, which treats translation as
progress but ignores in-place rotation. The official `PoseProgressChecker`
implementation is built as a separate overlay plugin so the system Nav2
installation under `/opt/ros/foxy` remains unchanged.

The behavior is kept from the official implementation. API adaptations are
limited to Foxy's lifecycle-node `SharedPtr`, Foxy's snake-case protected base
methods, and Foxy's lifecycle node clock access.
