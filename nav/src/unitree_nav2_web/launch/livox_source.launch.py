"""Direct MID-360 source; the supervisor restores Unitree's source on exit."""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    source = ExecuteProcess(
        cmd=['/usr/bin/python3', '-m', 'unitree_nav2_web.livox_source',
             '--interface', LaunchConfiguration('lidar_interface'),
             '--host-ip', LaunchConfiguration('lidar_host_ip'),
             '--lidar-ip', LaunchConfiguration('lidar_ip'),
             '--frequency', '50.0'],
        name='livox_source', output='both', sigterm_timeout='40', sigkill_timeout='5')
    return LaunchDescription([
        DeclareLaunchArgument('lidar_interface', default_value='eth0'),
        DeclareLaunchArgument('lidar_host_ip', default_value='192.168.123.164'),
        DeclareLaunchArgument('lidar_ip', default_value='192.168.123.120'),
        # Source loss must let navigation freshness gates stop commands while
        # the locomotion process keeps supporting the robot. Do not shut down
        # the entire parent launch (and its motor process) on a sensor exit.
        source,
        Node(package='unitree_nav2_web', executable='livox_adapter', output='screen'),
    ])
