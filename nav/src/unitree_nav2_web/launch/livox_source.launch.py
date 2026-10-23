"""Direct MID-360 source; the supervisor restores Unitree's source on exit."""
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

# Same navigation CPU partition as g1_navigation.launch.py.
NAV_PREFIX = 'taskset -c {} nice -n {}'.format(os.environ.get('G1_NAV_CPUS', '0-3'),
                                              os.environ.get('G1_NAV_NICE', '5'))


def generate_launch_description():
    source = ExecuteProcess(
        cmd=['/usr/bin/python3', '-m', 'unitree_nav2_web.livox_source',
             '--interface', LaunchConfiguration('lidar_interface'),
             '--host-ip', LaunchConfiguration('lidar_host_ip'),
             '--lidar-ip', LaunchConfiguration('lidar_ip'),
             '--frequency', LaunchConfiguration('frequency')],
        prefix=NAV_PREFIX,
        name='livox_source', output='both', sigterm_timeout='40', sigkill_timeout='5')
    return LaunchDescription([
        DeclareLaunchArgument('lidar_interface', default_value='eth0'),
        DeclareLaunchArgument('lidar_host_ip', default_value='192.168.123.164'),
        DeclareLaunchArgument('lidar_ip', default_value='192.168.123.120'),
        DeclareLaunchArgument('frequency', default_value='20.0'),
        DeclareLaunchArgument('point_stride', default_value='1'),
        # Source loss must let navigation freshness gates stop commands while
        # the locomotion process keeps supporting the robot. Do not shut down
        # the entire parent launch (and its motor process) on a sensor exit.
        source,
        # Upstream of FAST-LIO like cart_navigation: normal priority on the navigation CPUs.
        Node(package='unitree_nav2_web', executable='livox_adapter', output='screen',
             prefix='taskset -c {}'.format(os.environ.get('G1_NAV_CPUS', '0-3')),
             parameters=[{'point_stride': ParameterValue(LaunchConfiguration('point_stride'), value_type=int)}]),
    ])
