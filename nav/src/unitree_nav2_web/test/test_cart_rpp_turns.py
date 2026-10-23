"""Cart path following with the installed Foxy RPP and the production CartPush adapter.

Build unitree_nav2_core with BUILD_TESTING=ON first. The native driver advances
a simulated feedback clock without sleeping or publishing motion commands.
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
DRIVER = NAV / 'build/unitree_nav2_core/cart_rpp_regression'
# Measured push chain: on 2026-10-20 23:04 the cart yaw followed RPP wz like a
# 0.4 s first-order response after 0.4 s dead time (least-squares fit to the
# run). IDEAL_PLANT is the earlier 0.4 s response without dead time.
ROBOT_PLANT = dict(response_s=.4, dead_time_s=.4)
IDEAL_PLANT = dict(response_s=.4, dead_time_s=0.)


def run_route(directory, shape, yaw=0., speed=.35, plant=ROBOT_PLANT, **scenario):
    """Keep production controller settings; only mock the environment/plant."""
    scenario = dict(plant, **scenario)
    assert DRIVER.is_file(), 'Build the cart_rpp_regression test target first'
    directory.mkdir(parents=True, exist_ok=True)
    settings = yaml.safe_load(CONFIG.read_text())
    server = settings['controller_server']['ros__parameters']
    server.update(test_shape=shape, test_yaw=float(yaw), test_trace=str(directory / 'trace.csv'),
                  **{'test_' + key: float(value) for key, value in scenario.items()})
    controller = server['CartPush']
    controller.update(navigation_speed_mode='auto', navigation_speed=speed)
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
    result = next(line[len('CART_RPP_RESULT '):] for line in run.stdout.splitlines()
                  if line.startswith('CART_RPP_RESULT '))
    data = json.loads(result)
    (directory / 'result.json').write_text(json.dumps(data, indent=2)+'\n')
    assert data['speed'] == speed and data['dt'] == pytest.approx(1/server['controller_frequency'])
    # Pure pursuit: forward and turning only, within the cart task limits.
    assert data['min_vx'] >= 0. and data['max_abs_vy'] == 0., data
    assert data['max_vx'] <= speed + 1e-9, data
    assert data['max_abs_wz'] <= controller['max_vel_theta'] + 1e-9, data
    # Never above the final-approach cap measured to the real goal.
    assert data['speed_excess'] <= 1e-9, data
    # No command steps: pursuit, in-place turns and the switch between them.
    dt = data['dt']
    assert data['max_wz_step'] <= controller['max_angular_accel']*dt + 1e-9, data
    assert data['max_vx_step'] <= max(controller['max_linear_accel'],
                                      controller['max_linear_decel'])*dt + 1e-9, data
    return data


GOAL = yaml.safe_load(CONFIG.read_text())['controller_server']['ros__parameters']['goal_checker']


def check_arrival(data):
    assert data['reached'], data
    assert data['goal_distance_m'] < GOAL['xy_goal_tolerance'], data
    assert data['goal_heading_error_rad'] < GOAL['yaw_goal_tolerance'], data
    assert data['max_footprint_cost'] < 254, data


@pytest.mark.parametrize('shape', ['arc90', 'left90', 'right90', 'left60'])
@pytest.mark.parametrize('yaw', [0., .2, .4, .785, 1.2],
                         ids=['yaw0000', 'yaw0200', 'yaw0400', 'yaw0785', 'yaw1200'])
def test_cart_follows_long_path_through_sharp_turn(tmp_path, shape, yaw, record_property):
    data = run_route(tmp_path, shape, yaw, speed=.3)
    for key, value in data.items():
        record_property(key, value)
    check_arrival(data)
    # Follow the actual polyline, not merely reach its far end. A sharp
    # polygonal corner allows bounded cutting; the R0.6 arc is smooth.
    assert data['max_cross_m'] < (.10 if shape == 'arc90' else .17), data
    # No weaving: the former 1.5 s carrot reversed the turn 3 times on this plant.
    assert data['wz_reversals'] <= 1, data


@pytest.mark.parametrize('shape', ['arc90', 'left90', 'left60'])
def test_ideal_plant_bounds_corner_cutting_of_the_longer_carrot(tmp_path, shape):
    """A fast cart cuts corners a little more with the 2.5 s carrot; keep that bounded."""
    data = run_route(tmp_path, shape, .4, speed=.3, plant=IDEAL_PLANT)
    check_arrival(data)
    assert data['max_cross_m'] < (.10 if shape == 'arc90' else .18), data


@pytest.mark.parametrize('offset', [-.25, .25])
def test_cart_settles_on_a_straight_path_without_weaving(tmp_path, offset, record_property):
    """Real-robot regression: the former carrot kept reversing the turn on a straight leg."""
    data = run_route(tmp_path, 'straight', .4, speed=.3, lateral_offset=offset)
    for key, value in data.items():
        record_property(key, value)
    check_arrival(data)
    # The former 1.5 s carrot: 4 reversals, 2.8 cm late offset, sum |dwz| 2.7.
    assert data['late_cross_m'] < .02, data
    assert data['wz_reversals'] <= 1, data
    assert data['wz_variation'] < 1.5, data


@pytest.mark.parametrize('speed', [.2, .35, .6])
@pytest.mark.parametrize('offset', [-.25, 0., .25])
def test_cart_converges_to_straight_path_at_every_web_speed(tmp_path, speed, offset, record_property):
    data = run_route(tmp_path, 'straight', .4, speed=speed, lateral_offset=offset)
    for key, value in data.items():
        record_property(key, value)
    check_arrival(data)
    assert data['max_cross_m'] < abs(offset) + .03, data
    assert data['max_vx'] == pytest.approx(speed), data  # Cruises at the web speed.


def test_cart_stops_before_its_footprint_reaches_an_obstacle_on_the_path(tmp_path):
    data = run_route(tmp_path, 'straight', .4, obstacle_at=2.)
    assert not data['reached'], data
    assert data['failure'] == 'planner_exception', data
    assert 'footprint collision' in data['error'], data
    # Includes the 0.4 s plant lag after controller_server publishes zero.
    assert data['max_footprint_cost'] < 254, data


@pytest.mark.parametrize('shape', ['straight', 'arc90', 'left90', 'right90'])
def test_jagged_uneven_replanned_paths_do_not_shake_the_cart(tmp_path, shape, record_property):
    """NavFn-like +/-2.5 cm zigzag, uneven spacing, unset headings, 1 Hz replans."""
    data = run_route(tmp_path, shape, .4, path_noise=.025, replan_period=1.)
    for key, value in data.items():
        record_property(key, value)
    check_arrival(data)
    assert data['max_cross_m'] < (.08 if shape in ('straight', 'arc90') else .15), data
    # Replans to the same goal along the same route are not taken over.
    assert data['accepted_plans'] == 1, data
    # Total |dwz| over the run. Without conditioning and hysteresis this was
    # 3.6 (straight) to 4.8 (arc); a clean polyline gives 0 / 1.1 / 1.5.
    assert data['wz_variation'] < (1.0 if shape == 'straight' else 2.0), data
