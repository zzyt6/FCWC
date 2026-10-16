These unmodified tag36h11 PNGs are the official AprilRobotics examples:

- https://github.com/AprilRobotics/apriltag-imgs/blob/master/tag36h11/tag36_11_00000.png
- https://github.com/AprilRobotics/apriltag-imgs/blob/master/tag36h11/tag36_11_00001.png

Retrieved 2026-10-11. They test canonical tag orientation independently of
OpenCV's AprilTag generator, whose bitmaps are rotated 180 degrees relative to
these official examples. Pose is computed using the [OpenCV ArUco convention](https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html),
then rotated 180 degrees about local Z as requested by the user. For these
AprilRobotics PNGs in their stored orientation, the output is +X right, +Y up,
and +Z out of the printed face. OpenCV-generated canonical images consequently
have +X left and +Y down in this output frame.
