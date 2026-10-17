"""Button availability, direct HTTP validation and request encoding."""
import json
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlencode
from urllib.request import urlopen

import pytest

from unitree_nav2_web.locomotion import CART_ACTIONS, locomotion_actions
from unitree_nav2_web.nav_web_bridge import NavigationWebBridge
from test_navigation_safety import _Harness, _Locomotion, _Logger


def operator():
    return dict(online=True, owned=True, armed=True, healthy=True, stage='POLICY', stage_text='POLICY',
                velocity=[0, 0, 0], mode='control',
                high_policy=dict(enabled=False, enable_reason='', zero_reason='', camera=dict(fresh=True)),
                hands=dict(healthy=True, active=True, pose='zero', editor_available=True,
                           compensation_available=True), cart=dict(attached=True, valid=True))


@pytest.mark.parametrize('field,value', [('owned',False),('healthy',False),('decelerating',True),('velocity',[.1,0,0])])
def test_no_cart_buttons_when_operator_cannot_act(field, value):
    data = operator()
    data[field] = value
    assert not any(locomotion_actions(data)[key] for key in CART_ACTIONS)


@pytest.mark.parametrize('flag', ['moving', 'pending', 'recovering', 'recovery_pending', 'file_busy'])
def test_hand_transitions_block_toggle_compensation_and_detach(flag):
    data = operator()
    data['hands'][flag] = True
    actions = locomotion_actions(data)
    assert not actions['hand_toggle'] and not actions['hand_compensation'] and not actions['cart_detach']


def test_camera_buttons_and_compensation_match_real_capabilities():
    data = operator()
    assert not locomotion_actions(data)['cart_camera_start']
    data['high_policy']['camera']['fresh'] = False
    assert locomotion_actions(data)['cart_camera_start']
    data['high_policy']['camera_process'] = {'starting': True}
    assert not locomotion_actions(data)['cart_camera_start']
    data['high_policy']['camera_process'] = {}
    data['mode'] = 'sim'
    assert not locomotion_actions(data)['cart_camera_start']
    data['hands']['compensation_available'] = False
    assert not locomotion_actions(data)['hand_compensation']
    data['hands']['compensation_enabled'] = True
    assert locomotion_actions(data)['hand_compensation']  # Can always turn unavailable compensation off.
    data['stage'] = 'PREPARING'
    assert not locomotion_actions(data)['hand_compensation']


@pytest.mark.parametrize('action', sorted(CART_ACTIONS))
def test_http_cart_action_never_interrupts_active_navigation(action):
    h = _Harness(_Locomotion())
    h.locomotion.value['actions'] = {action: True}
    with pytest.raises(ValueError, match='先停止导航'):
        h.request_locomotion({'action':[action]})
    assert h.locomotion.requests == []
    assert h.motion_enabled and h.navigation_active


def test_reset_during_running_policy_is_rejected_without_closing_gate():
    h = _Harness()
    with pytest.raises(ValueError, match='复位'):
        h.request_locomotion({'action':['reset']})
    assert h.motion_enabled and h.locomotion.requests == []


@pytest.mark.parametrize('query', [dict(action=['hand_edit'],joint=['14'],angle_rad=['.1']),
                                  dict(action=['hand_edit'],joint=['0'],angle_rad=['nan']),
                                  dict(action=['hand_compensation'],enabled=['typo'])])
def test_invalid_hand_input_does_not_close_gate_or_reach_executor(query):
    h = _Harness()
    h.navigation_active = False
    h.locomotion.value['actions'] = {query['action'][0]: True}
    with pytest.raises(ValueError):
        h.request_locomotion(query)
    assert h.motion_enabled and h.locomotion.requests == []


def test_http_preserves_special_characters_in_pcd_filename():
    seen = []
    h = SimpleNamespace(web_root=Path(__file__).resolve().parents[1] / 'web',
                        get_logger=lambda: _Logger(),
                        get_parameter=lambda key: SimpleNamespace(value='127.0.0.1' if key=='http_host' else 0),
                        start_map_processing=lambda query: seen.append(query))
    logger = h.get_logger()
    logger.info = lambda *args: None
    h.get_logger = lambda: logger
    server = NavigationWebBridge._start_http_server(h)
    try:
        name = '大厅 + 20% & "x".pcd'
        query = urlencode({'name':name, 'ground_z':'auto'})
        with urlopen('http://127.0.0.1:%d/api/map/process?%s' % (server.server_port,query)) as response:
            assert json.load(response)['ok']
        assert seen == [{'name':[name], 'ground_z':['auto']}]
    finally:
        server.shutdown()
        server.server_close()
