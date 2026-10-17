"""Mapping controls with fake save service and in-memory point clouds; no ROS I/O."""
from concurrent.futures import Future
from types import SimpleNamespace
import threading
import time
import socket

import numpy as np
import pytest

from unitree_fastlio_baseline.viser_bridge import ViserBridge


@pytest.fixture
def view():
    h = SimpleNamespace(_lock=threading.Lock(), _full_mode=False, _full_load_in_progress=False,
                        _display_generation=0, _operation_error='', _status_message='', _destroyed=False,
                        _full_points=None, _full_intensity=None, _full_index=0, _clear_full_requested=False,
                        _map_dirty=False, _next_full_chunk_time=0., pcd_path='unused')
    for name in ('_request_save_and_load','_save_finished','_save_failed','_load_pcd_worker','_return_live','_restart_full_display'):
        setattr(h,name,getattr(ViserBridge,name).__get__(h))
    h.future = Future()
    h.requests = []
    def request(_):
        h.requests.append(True)
        return h.future
    h._map_save_client = SimpleNamespace(service_is_ready=lambda:True,call_async=request)
    h.get_logger = lambda: SimpleNamespace(error=lambda *args:None)
    h._read_pcd = lambda _: (np.ones((3,3),np.float32), np.zeros(3,np.float32))
    return h


def test_service_unavailable_and_send_error_leave_save_button_reusable(view):
    view._map_save_client.service_is_ready = lambda: False
    view._request_save_and_load()
    assert '未就绪' in view._operation_error and not view._full_load_in_progress
    view._map_save_client.service_is_ready = lambda: True
    def fail(_):
        raise RuntimeError('service vanished')
    view._map_save_client.call_async = fail
    view._request_save_and_load()
    assert 'service vanished' in view._operation_error and not view._full_load_in_progress


def test_repeated_save_is_single_request_and_return_live_ignores_late_result(view):
    view._request_save_and_load()
    view._request_save_and_load()
    assert len(view.requests) == 1
    view._return_live()
    view.future.set_result(SimpleNamespace(success=True, message='ok'))
    assert not view._full_mode and view._full_points is None and not view._full_load_in_progress


def test_return_live_during_pcd_read_cannot_be_overwritten(view):
    view._request_save_and_load()
    def read(_):
        view._return_live()
        return np.ones((3,3)),np.zeros(3)
    view._read_pcd = read
    generation = view._display_generation
    view._load_pcd_worker(generation)
    assert not view._full_mode and view._full_points is None


def test_saved_pcd_can_be_displayed_and_style_changes_restart_all_chunks(view):
    view._request_save_and_load()
    view._load_pcd_worker(view._display_generation)
    assert view._full_mode and len(view._full_points) == 3
    assert not view._full_load_in_progress
    view._full_index = 3
    view._clear_full_requested = False
    view._restart_full_display()
    assert view._full_index == 0 and view._clear_full_requested
    view._return_live()
    assert view._map_dirty and view._full_points is None


def test_service_failure_is_retained_for_the_page(view):
    view._request_save_and_load()
    view.future.set_result(SimpleNamespace(success=False,message='disk full'))
    assert 'disk full' in view._operation_error and not view._full_load_in_progress


def test_actual_mapping_widgets_over_websocket(monkeypatch):
    """Run the real Viser page and callbacks, with ROS and map saving replaced."""
    import msgspec
    import viser
    from rclpy.node import Node
    from websockets.sync.client import connect

    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0))
        port = probe.getsockname()[1]
    parameters = {'host':'127.0.0.1', 'port':port}
    pending, requests = [], []
    def request(_):
        future = Future()
        pending.append(future)
        requests.append(True)
        return future
    client = SimpleNamespace(service_is_ready=lambda:True, call_async=request)
    monkeypatch.setattr(Node, '__init__', lambda *args, **kwargs: None)
    monkeypatch.setattr(Node, 'declare_parameter', lambda self, name, value:parameters.setdefault(name,value))
    monkeypatch.setattr(Node, 'get_parameter', lambda self, name:SimpleNamespace(value=parameters[name]))
    monkeypatch.setattr(Node, 'get_logger', lambda self:SimpleNamespace(info=lambda *args:None,error=lambda *args:None))
    monkeypatch.setattr(Node, 'create_subscription', lambda *args:None)
    monkeypatch.setattr(Node, 'create_timer', lambda *args:None)
    monkeypatch.setattr(Node, 'create_client', lambda *args:client)
    monkeypatch.setattr(Node, 'destroy_node', lambda *args:True)
    node = ViserBridge()
    node._read_pcd = lambda _: (np.array([[0.,0.,0.],[1.,1.,1.]],dtype=np.float32),np.array([40.,80.]))
    def until(predicate):
        deadline = time.monotonic()+5
        while not predicate():
            assert time.monotonic()<deadline, 'Mapping widget did not complete'
            time.sleep(.01)
    try:
        node._render()
        assert node._live_button.disabled and not node._full_button.disabled
        with connect('ws://127.0.0.1:%d' % node._server.get_port(),max_size=8*1024*1024,
                     subprotocols=['viser-v'+viser.__version__]) as ws:
            ws.send(msgspec.msgpack.encode(dict(type='ViewerCameraMessage',wxyz=[1,0,0,0],
                position=[2,-2,1.5],fov=1.,near=.01,far=100.,image_height=720,image_width=1280,
                look_at=[0,0,.7],up_direction=[0,0,1])))
            until(lambda:bool(node._server.get_clients()))
            def update(handle, value):
                ws.send(msgspec.msgpack.encode(dict(type='GuiUpdateMessage',uuid=handle._impl.uuid,updates={'value':value})))
            update(node._full_button, True)
            until(lambda:len(requests)==1)
            node._render()
            assert node._full_button.disabled and not node._live_button.disabled
            update(node._live_button,True)
            until(lambda:node._display_generation==2)
            pending.pop().set_result(SimpleNamespace(success=True,message='saved'))
            assert not node._full_mode
            node._render()
            update(node._full_button,True)
            until(lambda:len(requests)==2)
            pending.pop().set_result(SimpleNamespace(success=True,message='saved'))
            until(lambda:node._full_mode)
            node._render()
            assert len(node._full_handles)==1
            for color in ('solid','intensity','height'):
                update(node._color_dropdown,color)
                until(lambda:node.color_mode==color and node._clear_full_requested)
                node._render()
                assert len(node._full_handles)==1
                np.testing.assert_array_equal(node._full_handles[0].colors,
                                              node._colors(node._full_points,node._full_intensity,color))
            update(node._point_size_slider,.06)
            until(lambda:node.point_size==.06 and node._clear_full_requested)
            node._render()
            assert len(node._full_handles)==1 and node._full_handles[0].point_size==.06
            update(node._live_button,True)
            until(lambda:not node._full_mode)
            node._render()
            assert node._full_handles==[] and node._live_handle.visible
    finally:
        node.destroy_node()
