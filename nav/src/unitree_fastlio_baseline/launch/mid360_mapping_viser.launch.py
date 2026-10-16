import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    share_dir = get_package_share_directory("unitree_fastlio_baseline")

    fastlio_config = LaunchConfiguration("fastlio_config")
    viser_config = LaunchConfiguration("viser_config")
    viser_host = LaunchConfiguration("viser_host")
    viser_port = LaunchConfiguration("viser_port")
    map_output = LaunchConfiguration("map_output")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "fastlio_config",
                default_value=os.path.join(
                    share_dir, "config", "fast_lio_mid360_indoor.yaml"
                ),
            ),
            DeclareLaunchArgument(
                "viser_config",
                default_value=os.path.join(share_dir, "config", "viser.yaml"),
            ),
            DeclareLaunchArgument("viser_host", default_value="0.0.0.0"),
            DeclareLaunchArgument("viser_port", default_value="8080"),
            DeclareLaunchArgument(
                "map_output",
                default_value=os.path.join(os.getcwd(), "maps", "fastlio_map.pcd"),
            ),
            Node(
                package="unitree_g1_kinematics",
                executable="g1_lowstate_udp_bridge",
                name="g1_lowstate_udp_bridge",
                output="screen",
                additional_env={"LD_LIBRARY_PATH": ""},
            ),
            Node(
                package="unitree_g1_kinematics",
                executable="g1_base_kinematics_node",
                name="g1_base_kinematics",
                output="screen",
                parameters=[
                    {
                        "lowstate_udp_port": 39001,
                        "fastlio_odom_topic": "/Odometry",
                        "base_odom_topic": "/Odometry_base",
                        "base_frame": "base_link",
                        "model_variant": "auto",
                    }
                ],
            ),
            Node(
                package="fast_lio",
                executable="fastlio_mapping",
                name="laser_mapping",
                output="screen",
                parameters=[fastlio_config, {"map_file_path": map_output}],
            ),
            Node(
                package="unitree_fastlio_baseline",
                executable="viser_bridge",
                name="viser_bridge",
                output="screen",
                parameters=[
                    viser_config,
                    {
                        "host": viser_host,
                        "port": ParameterValue(viser_port, value_type=int),
                        "pcd_path": map_output,
                    },
                ],
            ),
        ]
    )
