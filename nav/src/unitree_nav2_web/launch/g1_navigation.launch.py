import os
import json

from ament_index_python.packages import get_package_prefix, get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

# CPU partition. Navigation (LiDAR, FAST-LIO, Nav2, Python filters, web UI) is
# pinned to CPUs 0-3 at lower priority; run_executor.sh pins the body policy
# and its 200 Hz C++ bridge to CPUs 6-7, and the executor runs the cart camera
# on CPUs 4-5. Under Viser alone the camera detects at ~48 Hz; sharing all
# cores with navigation cut it to ~26 Hz
# and delayed every observation. Lowering priority/affinity needs no root.
NAV_CPUS = os.environ.get("G1_NAV_CPUS", "0-3")
NAV_NICE = os.environ.get("G1_NAV_NICE", "5")
NAV_PREFIX = f"taskset -c {NAV_CPUS} nice -n {NAV_NICE}"
# Tiny safety nodes on the navigation cores, without lowered priority.
GATE_PREFIX = f"taskset -c {NAV_CPUS}"


def nav_node(prefix=NAV_PREFIX, **kwargs):
    return Node(prefix=prefix, **kwargs)


def generate_launch_description():
    share = get_package_share_directory("unitree_nav2_web")
    core_prefix = get_package_prefix("unitree_nav2_core")
    root = os.getcwd()
    cart_config = os.path.join(root, "config", "pushcart_navigation.json")
    with open(cart_config) as stream:
        cart_limits = json.load(stream)["task_limits"]
    # Foxy's stock replanning tree with server_timeout="1000" on every action
    # node; the 10 ms default aborts goals whenever the planner is busy.
    bt_xml = os.path.join(share, "behavior_trees", "walk_replanning.xml")
    footprint = "[[-0.30, -0.25], [-0.30, 0.25], [0.32, 0.25], [0.32, -0.25]]"

    map_yaml = LaunchConfiguration("map_yaml")
    http_port = LaunchConfiguration("http_port")
    rosbridge_port = LaunchConfiguration("rosbridge_port")
    fastlio_config = LaunchConfiguration("fastlio_config")
    nav2_params = LaunchConfiguration("nav2_params")
    start_locomotion = LaunchConfiguration("start_locomotion")
    locomotion_mode = LaunchConfiguration("locomotion_mode")
    locomotion_policy = LaunchConfiguration("locomotion_policy")
    heading_source = LaunchConfiguration("heading_source")
    heading_mode = LaunchConfiguration("heading_mode")
    command_port = LaunchConfiguration("locomotion_command_port")
    status_port = LaunchConfiguration("locomotion_status_port")
    network_interface = os.environ.get("G1_NETWORK_INTERFACE", "eth0")
    direct_livox = LaunchConfiguration("direct_livox")
    direct_enabled = ["'", direct_livox, "'.lower() in ('true', '1')"]
    raw_lidar_topic = PythonExpression([
        "'/navigation/livox_points' if ", *direct_enabled,
        " else '/utlidar/cloud_livox_mid360'"])
    imu_topic = PythonExpression([
        "'/livox/imu' if ", *direct_enabled, " else '/utlidar/imu_livox_mid360'"])
    # Direct MID-360 rate (10/20/50 Hz). 20 Hz keeps a 50 ms frame period for
    # obstacles while FAST-LIO, the cloud filter and both scan converters do
    # 60 % less per-frame work than at 50 Hz; Nav2 follows at 20 Hz anyway.
    lidar_hz = LaunchConfiguration("direct_livox_hz")
    scan_time = ParameterValue(PythonExpression(
        ["1.0 / float('", lidar_hz, "') if ", *direct_enabled, " else 0.10"]), value_type=float)

    nav_nodes = [
        nav_node(
            package="nav2_map_server",
            executable="map_server",
            name="map_server",
            output="screen",
            parameters=[nav2_params, {"yaml_filename": map_yaml}],
        ),
        # The goal path (BT -> planner -> 20 Hz controller) keeps normal
        # priority on the navigation cores; only heavy point-cloud, LIO and UI
        # work is lowered. A starved planner cannot acknowledge the BT in time.
        nav_node(
            prefix=GATE_PREFIX,
            package="nav2_controller",
            executable="controller_server",
            name="controller_server",
            output="both",
            parameters=[nav2_params, {
                # Pure pursuit only drives forward and turns: vx max and |wz|.
                "CartPush.max_vel_x": cart_limits[0][1],
                "CartPush.max_vel_theta": max(abs(v) for v in cart_limits[2]),
            }],
        ),
        nav_node(
            prefix=GATE_PREFIX,
            package="nav2_planner",
            executable="planner_server",
            name="planner_server",
            output="screen",
            parameters=[nav2_params],
        ),
        nav_node(
            package="nav2_recoveries",
            executable="recoveries_server",
            name="recoveries_server",
            output="screen",
            parameters=[nav2_params],
        ),
        nav_node(
            prefix=GATE_PREFIX,
            package="nav2_bt_navigator",
            executable="bt_navigator",
            name="bt_navigator",
            # Keep BT failures (send_goal timeouts, XML errors) in launch.log.
            output="both",
            parameters=[nav2_params, {"default_bt_xml_filename": bt_xml}],
        ),
        nav_node(
            package="nav2_waypoint_follower",
            executable="waypoint_follower",
            name="waypoint_follower",
            output="screen",
            parameters=[nav2_params],
        ),
        nav_node(
            package="nav2_lifecycle_manager",
            executable="lifecycle_manager",
            name="lifecycle_manager_localization",
            output="screen",
            parameters=[
                {
                    "autostart": True,
                    # Foxy AMCL 0.4.7 does not safely survive repeated
                    # cleanup/configure cycles when switching maps.  The web
                    # bridge therefore starts a fresh, unmodified official
                    # AMCL process for each selected map.  This manager owns
                    # only map_server.
                    "node_names": ["map_server"],
                }
            ],
        ),
        nav_node(
            package="nav2_lifecycle_manager",
            executable="lifecycle_manager",
            name="lifecycle_manager_navigation",
            output="screen",
            parameters=[
                {
                    "autostart": True,
                    "node_names": [
                        "controller_server",
                        "planner_server",
                        "recoveries_server",
                        "bt_navigator",
                        "waypoint_follower",
                    ],
                }
            ],
        ),
    ]

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "map_yaml",
                default_value=os.path.join(root, "maps", "generated", "cf", "map.yaml"),
            ),
            DeclareLaunchArgument("http_port", default_value="8080"),
            DeclareLaunchArgument("rosbridge_port", default_value="9090"),
            DeclareLaunchArgument("direct_livox", default_value="true",
                                  description="Directly acquire MID-360 at direct_livox_hz; "
                                              "restore Unitree lidar on exit. false = vendor 10 Hz"),
            DeclareLaunchArgument("direct_livox_hz", default_value="20", choices=["10", "20", "50"],
                                  description="Direct MID-360 frame rate (20 Hz: ~10000 points per frame)"),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(os.path.join(share, "launch", "livox_source.launch.py")),
                condition=IfCondition(direct_livox),
                # FAST-LIO's 1-in-3 point decimation happens in the adapter, before the
                # cart self-filter: 2026-10-21 that filter saturated cart_navigation
                # (11-21 Hz tick, cart poses 0.2-0.4 s old; web "观测已过期").
                launch_arguments={"lidar_interface": network_interface, "frequency": lidar_hz,
                                  "point_stride": "3"}.items()),
            DeclareLaunchArgument(
                "speed_settings_file",
                default_value=os.path.join(root, "config", "navigation_speed.json"),
            ),
            DeclareLaunchArgument(
                "fastlio_config",
                default_value=os.path.join(share, "config", "fast_lio_mid360_navigation.yaml"),
            ),
            DeclareLaunchArgument(
                "nav2_params",
                default_value=os.path.join(share, "config", "nav2_foxy_g1.yaml"),
            ),
            # Our own Balance/Homie locomotion (nav/locomotion). It owns the
            # Unitree low-level bridge, so Viser must not run at the same time.
            DeclareLaunchArgument("start_locomotion", default_value="true"),
            DeclareLaunchArgument("locomotion_mode", default_value="control"),
            DeclareLaunchArgument("locomotion_record", default_value="on", choices=["on", "off"]),
            DeclareLaunchArgument("locomotion_policy", default_value="auto"),
            DeclareLaunchArgument("heading_source", default_value="NAVIGATION"),
            # "config" keeps run_mode from nav/locomotion/motion_config.json;
            # disabled / shadow / active override it for this run only.
            DeclareLaunchArgument("heading_mode", default_value="config"),
            DeclareLaunchArgument("locomotion_command_port", default_value="39002"),
            DeclareLaunchArgument("locomotion_status_port", default_value="39005"),
            nav_node(
                prefix=GATE_PREFIX,
                package="unitree_g1_kinematics",
                executable="g1_lowstate_udp_bridge",
                name="g1_lowstate_udp_bridge",
                output="screen",
                # This SDK-only executable resolves DDS through its own RUNPATH.
                additional_env={"LD_LIBRARY_PATH": ""},
            ),
            ExecuteProcess(
                condition=IfCondition(start_locomotion),
                cmd=[
                    "bash",
                    os.path.join(root, "locomotion", "run_executor.sh"),
                    "--mode", locomotion_mode,
                    "--interface", network_interface,
                    "--policy-mode", locomotion_policy,
                    "--heading-source", heading_source,
                    "--heading-mode", heading_mode,
                    "--command-port", command_port,
                    "--status-port", status_port,
                    "--cart-config", cart_config,
                    # Per-tick task/student/camera telemetry, comparable with
                    # Viser push sessions; stored under nav/log/sessions.
                    "--recording", LaunchConfiguration("locomotion_record"),
                ],
                name="g1_locomotion_executor",
                # Keep model-loading/bridge tracebacks after the terminal closes.
                output="both",
                # The executor needs SIGINT for its own one-second damping stop.
                sigterm_timeout="12",
            ),
            nav_node(
                package="unitree_g1_kinematics",
                executable="g1_base_kinematics_node",
                name="g1_base_kinematics",
                output="screen",
                parameters=[
                    {
                        "lowstate_udp_port": 39001,
                        "fastlio_odom_topic": "/Odometry",
                        "base_odom_topic": "/Odometry_base",
                        "imu_topic": imu_topic,
                        "base_frame": "base_link",
                        "navigation_frame": "base_nav",
                        "model_variant": "auto",
                    }
                ],
            ),
            nav_node(
                prefix=GATE_PREFIX,
                package="unitree_g1_kinematics",
                executable="g1_cmd_vel_udp_node",
                name="g1_cmd_vel_udp",
                # Gate open/close causes belong in launch.log.
                output="both",
                parameters=[
                    {
                        "cmd_vel_topic": "/cmd_vel",
                        "udp_port": ParameterValue(command_port, value_type=int),
                        "max_vx": 0.60,
                        "max_vy": 0.20,
                        "max_vyaw": 0.50,
                        "cart_min_vx": cart_limits[0][0],
                        "cart_max_vx": cart_limits[0][1],
                        "cart_max_vy": max(abs(v) for v in cart_limits[1]),
                        "cart_max_vyaw": max(abs(v) for v in cart_limits[2]),
                        "command_timeout": 0.25,
                        # The web bridge renews motion authorization every
                        # 0.5 s. A stalled/closed web page must not stop the
                        # body; it only revokes navigation velocity after 3 s.
                        "motion_enable_timeout": 3.0,
                        # Reference loss pauses velocity immediately and only
                        # revokes the gate when it persists for 3 s.
                        "reference_fresh_timeout": 0.5,
                        "reference_fault_timeout": 3.0,
                    }
                ],
            ),
            nav_node(
                # Every LiDAR frame passes this node before FAST-LIO, and every
                # cart odometry after it. At nice 5 during a 2026-10-20 push its
                # 50 Hz tick ran at 11-21 Hz and poses reached the controller
                # ~0.35 s after measurement (35 ms with the cart detached).
                prefix=GATE_PREFIX,
                package="unitree_nav2_web",
                executable="cart_navigation",
                name="cart_navigation",
                output="both",
                parameters=[{"config_file": cart_config, "raw_cloud_topic": raw_lidar_topic}],
            ),
            nav_node(
                # Localization source for odometry, cart TF and AMCL: normal
                # priority so it keeps up with every LiDAR frame.
                prefix=GATE_PREFIX,
                package="fast_lio",
                executable="fastlio_mapping",
                name="laser_mapping",
                output="screen",
                parameters=[fastlio_config, {"common.lid_topic": "/navigation/lidar",
                                             "common.imu_topic": imu_topic,
                                             # The direct adapter already keeps 1 point in 3.
                                             "point_filter_num": ParameterValue(PythonExpression(
                                                 ["1 if ", *direct_enabled, " else 3"]), value_type=int),
                                             "preprocess.scan_rate": ParameterValue(PythonExpression(
                                                 ["int(float('", lidar_hz, "')) if ", *direct_enabled, " else 10"]),
                                                 value_type=int)}],
            ),
            nav_node(
                package="unitree_nav2_core",
                executable="cloud_frame_transformer",
                name="cloud_frame_transformer",
                output="screen",
                parameters=[{"target_frame": "base_nav"}],
            ),
            nav_node(
                package="unitree_nav2_core",
                executable="cloud_obstacle_filter",
                name="cloud_obstacle_filter",
                output="screen",
                parameters=[
                    {
                        "input_topic": "/cloud_nav",
                        "output_topic": "/cloud_obstacles",
                        "ground_distance_threshold": 0.06,
                        "min_obstacle_height": 0.15,
                        "max_obstacle_height": 1.80,
                        "self_filter_radius": 0.60,
                        # Produce one plane-relative obstacle cloud out to 8 m.
                        # The local LaserScan below still caps avoidance at
                        # 4 m, while AMCL can use farther structural returns.
                        "max_range": 8.0,
                    }
                ],
            ),
            nav_node(
                package="pointcloud_to_laserscan",
                executable="pointcloud_to_laserscan_node",
                name="pointcloud_to_laserscan",
                output="screen",
                remappings=[("cloud_in", "/cloud_obstacles"), ("scan", "/scan_unmasked")],
                parameters=[
                    {
                        "target_frame": "base_nav",
                        "transform_tolerance": 0.10,
                        # Ground and self points have already been removed in
                        # plane-relative coordinates by cloud_obstacle_filter.
                        "min_height": -2.0,
                        "max_height": 2.0,
                        "angle_min": -3.141592653589793,
                        "angle_max": 3.141592653589793,
                        "angle_increment": 0.008726646,
                        "scan_time": scan_time,
                        "range_min": 0.60,
                        "range_max": 4.0,
                        "use_inf": True,
                    }
                ],
            ),
            nav_node(
                package="pointcloud_to_laserscan",
                executable="pointcloud_to_laserscan_node",
                name="pointcloud_to_localization_scan",
                output="screen",
                remappings=[
                    ("cloud_in", "/cloud_obstacles"),
                    ("scan", "/scan_localization_unmasked"),
                ],
                parameters=[
                    {
                        "target_frame": "base_nav",
                        "transform_tolerance": 0.10,
                        # cloud_obstacles has already removed the live RANSAC
                        # ground plane and robot self points in plane-relative
                        # coordinates.  Do not reintroduce a fixed-Z floor
                        # slice here: it misclassifies sloped floor as walls.
                        "min_height": -2.0,
                        "max_height": 2.0,
                        "angle_min": -3.141592653589793,
                        "angle_max": 3.141592653589793,
                        "angle_increment": 0.008726646,
                        "scan_time": scan_time,
                        "range_min": 0.60,
                        "range_max": 8.0,
                        "use_inf": True,
                    }
                ],
            ),
            *nav_nodes,
            nav_node(
                package="rosbridge_server",
                executable="rosbridge_websocket",
                name="rosbridge_websocket",
                output="screen",
                parameters=[
                    {
                        "address": "0.0.0.0",
                        "port": ParameterValue(rosbridge_port, value_type=int),
                        "max_message_size": 10000000,
                    }
                ],
            ),
            nav_node(
                package="unitree_nav2_web",
                executable="nav_web_bridge",
                name="navigation_web_bridge",
                output="both",
                parameters=[
                    {
                        "maps_dir": os.path.join(root, "maps"),
                        "generated_maps_dir": os.path.join(root, "maps", "generated"),
                        "web_root": os.path.join(share, "web"),
                        "http_host": "0.0.0.0",
                        "http_port": ParameterValue(http_port, value_type=int),
                        "converter_executable": os.path.join(
                            core_prefix, "lib", "unitree_nav2_core", "pcd_to_nav_map"
                        ),
                        "amcl_executable": os.path.join(
                            get_package_prefix("nav2_amcl"), "lib", "nav2_amcl", "amcl"
                        ),
                        "nav2_params_file": nav2_params,
                        "cart_bt_xml": os.path.join(share, "behavior_trees", "cart_push.xml"),
                        "speed_settings_file": LaunchConfiguration("speed_settings_file"),
                        "footprint": footprint,
                        "initial_pose_requires_policy": ParameterValue(PythonExpression([
                            "'", start_locomotion, "'.lower() in ('true', '1', 'yes') and '",
                            locomotion_mode, "' != 'observe'",
                        ]), value_type=bool),
                        "locomotion_command_port": ParameterValue(
                            command_port, value_type=int
                        ),
                        "locomotion_status_port": ParameterValue(
                            status_port, value_type=int
                        ),
                    }
                ],
            ),
        ]
    )
