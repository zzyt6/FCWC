import json
from concurrent.futures import Future
from types import SimpleNamespace

import pytest

from unitree_nav2_web.navigation_speed import NavigationSpeedSettings, validate_speed


class Client:
    def __init__(self):
        self.ready = True
        self.requests = []
        self.future = None

    def service_is_ready(self):
        return self.ready

    def call_async(self, request):
        self.requests.append(request)
        self.future = Future()
        return self.future

    def complete(self, success=True):
        self.future.set_result(SimpleNamespace(result=SimpleNamespace(
            successful=success, reason="" if success else "controller rejected profile"
        )))


@pytest.mark.parametrize("speed", [0, 0.19, 0.61, -1, "nan", "inf", True])
def test_rejects_invalid_speed(speed):
    with pytest.raises(ValueError):
        validate_speed("fixed", speed)


def test_profile_changes_only_after_ack_and_survives_restart(tmp_path):
    client = Client()
    path = tmp_path / "navigation_speed.json"
    settings = NavigationSpeedSettings(client, path)
    settings.tick()
    assert settings.snapshot()["pending"] and not settings.snapshot()["ready"]
    client.complete()
    assert settings.snapshot()["ready"]
    settings.request("fixed", 0.6)
    assert settings.snapshot()["mode"] == "auto"
    assert not path.exists()
    assert len(client.requests[-1].parameters) == 2
    client.complete()
    assert settings.snapshot()["mode"] == "fixed"
    assert json.loads(path.read_text()) == {"mode": "fixed", "speed": 0.6}
    restored = NavigationSpeedSettings(Client(), path)
    assert restored.snapshot()["speed"] == 0.6
    assert not restored.snapshot()["ready"]  # new process must obtain a new ack


def test_rejection_keeps_last_applied_profile_and_file(tmp_path):
    client = Client()
    settings = NavigationSpeedSettings(client, tmp_path / "speed.json")
    settings.request("fixed", 0.2)
    client.complete()
    before = settings.path.read_text()
    settings.request("auto", 0.5)
    client.complete(False)
    state = settings.snapshot()
    assert state["ready"] and not state["pending"]
    assert state["mode"] == "fixed" and state["speed"] == 0.2
    assert "未生效" in state["error"]
    assert settings.path.read_text() == before


def test_pending_updates_are_serialized_and_service_absence_is_visible(tmp_path):
    client = Client()
    settings = NavigationSpeedSettings(client, tmp_path / "speed.json")
    client.ready = False
    settings.tick()
    assert not settings.snapshot()["ready"]
    assert "未就绪" in settings.snapshot()["error"]
    client.ready = True
    settings.request("fixed", 0.4)
    with pytest.raises(ValueError, match="等待"):
        settings.request("auto", 0.3)
    settings.started -= 4.0
    assert "超时" in settings.snapshot()["error"]
    client.complete()
    assert settings.snapshot()["speed"] == 0.4


def test_bad_saved_profile_uses_valid_default(tmp_path):
    path = tmp_path / "speed.json"
    path.write_text('{"mode":"fixed","speed":20}')
    settings = NavigationSpeedSettings(Client(), path)
    assert settings.snapshot()["speed"] == 0.35
    assert settings.snapshot()["save_warning"]
