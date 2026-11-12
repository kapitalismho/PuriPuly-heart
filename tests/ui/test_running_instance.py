from __future__ import annotations

import pytest

from puripuly_heart.ui import running_instance
from puripuly_heart.ui.i18n import t_for_locale


class FakeUser32:
    def __init__(self, *, window: int | None, visible: bool = True, iconic: bool = False):
        self.window = window
        self.visible = visible
        self.iconic = iconic
        self.calls: list[tuple] = []

    def FindWindowW(self, class_name, title):
        self.calls.append(("find", title))
        return self.window

    def IsWindowVisible(self, window):
        return self.visible

    def IsIconic(self, window):
        return self.iconic

    def ShowWindow(self, window, command):
        self.calls.append(("show", command))
        return True

    def SetForegroundWindow(self, window):
        self.calls.append(("foreground", window))
        return True

    def MessageBoxW(self, owner, text, title, flags):
        self.calls.append(("message", text, title))
        return 1


@pytest.fixture(autouse=True)
def _korean_system(monkeypatch):
    monkeypatch.setattr(running_instance, "detect_system_locale", lambda: "Korean_Korea")


def test_visible_running_window_is_restored_and_focused_without_a_message() -> None:
    user32 = FakeUser32(window=42, iconic=True)

    assert running_instance.notify_running_instance(user32) == "focused"

    assert ("find", "PuriPuly <3") in user32.calls
    assert ("show", 9) in user32.calls
    assert ("foreground", 42) in user32.calls
    assert not [call for call in user32.calls if call[0] == "message"]


@pytest.mark.parametrize("window, visible", [(None, False), (42, False)])
def test_missing_or_hidden_window_shows_localized_already_running_message(window, visible) -> None:
    user32 = FakeUser32(window=window, visible=visible)

    assert running_instance.notify_running_instance(user32) == "notified"

    message = t_for_locale("ko", "app.already_running")
    assert message != "app.already_running"
    assert ("message", message, "PuriPuly <3") in user32.calls
    assert not [call for call in user32.calls if call[0] == "foreground"]
