"""Long-path sharp turns using the installed Foxy sampler, critics and planner.

Build unitree_nav2_core with BUILD_TESTING=ON first. The native driver advances
a simulated 50 Hz feedback clock without sleeping or publishing motion commands.
The existing test_cart_stack.py separately exercises the live ROS action/TF path.
"""
import json
import os
from pathlib import Path
import subprocess

import pytest
import yaml

from unitree_nav2_web.cart_geometry import CartGeometry, load_settings


NAV = Path(__file__).resolve().parents[3]
CONFIG = NAV / 'src/unitree_nav2_web/config/nav2_foxy_g1.yaml'
DRIVER = NAV / 'build/unitree_nav2_core/cart_dwb_regression'


def run_turn(directory, shape, yaw, overrides=None):
    """Keep production controller settings; only mock the environment/plant."""
    assert DRIVER.is_file(), 'Build the cart_dwb_regression test target first'
    directory.mkdir(parents=True, exist_ok=True)
    settings = yaml.safe_load(CONFIG.read_text())
    server = settings['controller_server']['ros__parameters']
    server.update(test_shape=shape, test_yaw=float(yaw), test_trace=str(directory / 'trace.csv'))
    controller = server['CartPush']
    controller.update(navigation_speed_mode='auto', navigation_speed=.3, max_speed_xy=.3)
    if overrides:
        controller.update(overrides)
    geometry = CartGeometry(load_settings(NAV / 'config/pushcart_navigation.json'))
    outline = geometry.footprint(dict(position=[.8, 0., -.7], yaw=0.,
                                     palms=[[.5, .2, .1], [.5, -.2, .1]]), True)
    costmap = settings['local_costmap']['local_costmap']['ros__parameters']
    # Foxy's parameter parser cannot infer a type for an empty string array.
    # The inflation layer alone leaves this obstacle-free map free of costs.
    costmap.update(plugins=['inflation_layer'], track_unknown_space=False,
                   footprint=json.dumps([[float(x), float(y)] for x, y in outline]))
    parameters = directory / 'params.yaml'
    parameters.write_text(yaml.safe_dump(settings))
    env = dict(os.environ, ROS_DOMAIN_ID='91', ROS_LOCALHOST_ONLY='1',
               ROS_LOG_DIR=str(directory / 'ros_log'), RMW_IMPLEMENTATION='rmw_cyclonedds_cpp',
               CYCLONEDDS_URI='<CycloneDDS><Domain><General>'
               '<NetworkInterfaceAddress>lo</NetworkInterfaceAddress><AllowMulticast>false</AllowMulticast>'
               '</General><Discovery><ParticipantIndex>auto</ParticipantIndex>'
               '<Peers><Peer Address="127.0.0.1"/></Peers></Discovery></Domain></CycloneDDS>')
    run = subprocess.run([str(DRIVER), '--ros-args', '--params-file', str(parameters),
                          '--log-level', 'error'], env=env, text=True,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=180)
    (directory / 'controller.log').write_text(run.stdout)
    assert run.returncode == 0, run.stdout
    result = next(line[len('CART_DWB_RESULT '):] for line in run.stdout.splitlines()
                  if line.startswith('CART_DWB_RESULT '))
    data = json.loads(result)
    (directory / 'result.json').write_text(json.dumps(data, indent=2)+'\n')
    assert data['speed'] == .3 and data['sim_time'] == 1.0, data
    assert data['shorten'] == controller['shorten_transformed_plan'], data
    assert data['prune_distance'] == controller['prune_distance'], data
    assert data['twirling_scale'] == controller['Twirling.scale'], data
    return data


@pytest.mark.parametrize('shape', ['arc90', 'left90', 'right90', 'left60'])
@pytest.mark.parametrize('yaw', [0., .2, .4, .785, 1.2],
                         ids=['yaw0000', 'yaw0200', 'yaw0400', 'yaw0785', 'yaw1200'])
def test_cart_follows_long_path_through_sharp_turn(tmp_path, shape, yaw, record_property):
    data = run_turn(tmp_path, shape, yaw)
    for key, value in data.items():
        record_property(key, value)
    assert data['shorten'] is True, data
    assert data['reached'], data
    # Require following the actual polyline, not merely reaching its far end.
    # Sharp polygonal corners allow bounded corner cutting, unlike the
    # straight-path tests (8 cm). Keep the actual and predicted maxima in XML.
    assert data['max_cross_m'] < .20, data
    assert data['max_predicted_cross_m'] < .25, data
    assert data['goal_distance_m'] < .15, data
    assert data['goal_heading_error_rad'] < .174533, data
