import asyncio
from types import SimpleNamespace
from uuid import uuid4

from puripuly_heart.app.services.quick_translate import QuickTranslateOwner
from puripuly_heart.ui.quick_translate_window import (
    QuickWindow,
    clamp_window_bounds,
    quick_child_environment,
)


def test_quick_child_is_loopback_only_and_does_not_inherit_credentials():
    parent = {
        "FLET_SERVER_IP": "0.0.0.0",
        "FLET_SERVER_PORT": "12345",
        "FLET_FORCE_WEB_SERVER": "true",
        "FLET_DART_BRIDGE_PORT": "54321",
        "OPENROUTER_API_KEY": "never-inherit",
        "HF_TOKEN": "never-inherit",
        "PYTHONPATH": "required-source-path",
    }
    child = quick_child_environment(parent)
    assert child["FLET_SERVER_IP"] == "127.0.0.1"
    assert child["PYTHONPATH"] == "required-source-path"
    assert all(
        key not in child
        for key in (
            "FLET_SERVER_PORT",
            "FLET_FORCE_WEB_SERVER",
            "FLET_DART_BRIDGE_PORT",
            "OPENROUTER_API_KEY",
            "HF_TOKEN",
        )
    )
    assert parent["FLET_SERVER_IP"] == "0.0.0.0" and parent["HF_TOKEN"] == "never-inherit"


class Page:
    def __init__(self):
        self.window = SimpleNamespace()
        self.controls = []

    def add(self, control):
        self.controls.append(control)

    def run_task(self, callback):
        return None

    def update(self):
        return None


async def test_blank_duplicate_retry_and_identical_new_text():
    window = QuickWindow(Page())
    messages = []
    window.emit = messages.append
    window.input.value = "  "
    await window.send()
    assert not messages
    window.input.value = "相同文字"
    await window.send()
    await window.send()
    assert len(messages) == 1
    identity = messages[0]["id"]
    window.accept_result(dict(id=identity, code="translation_failed"))
    assert window.input.value == "相同文字"
    await window.send()
    assert messages[-1]["id"] != identity
    window.accept_result(dict(id=messages[-1]["id"], translation="Same text"))
    assert window.input.value == ""
    window.input.value = "相同文字"
    await window.send()
    assert len(messages) == 3
    assert window.input.multiline and window.input.shift_enter


def test_out_of_order_results_keep_originals_and_unknown_results_ignored():
    window = QuickWindow(Page())
    window.accept_chat(dict(id="a", role="self", original="first", translation="", state="pending"))
    window.accept_chat(
        dict(id="b", role="peer", original="second", translation="", state="pending")
    )
    window.accept_chat(
        dict(id="b", role="peer", original="second", translation="二", state="complete")
    )
    window.accept_chat(
        dict(id="a", role="self", original="first", translation="一", state="complete")
    )
    window.accept_result(dict(id="unknown", translation="wrong"))
    assert window.order == ["a", "b"]
    assert window.history["a"][1].value == "first\n→ 一"
    assert window.history["b"][1].value == "second\n→ 二"


def test_disconnected_monitor_and_scaling_bounds():
    values = clamp_window_bounds(
        dict(width=9000, height=9000, left=-20000, top=20000), work_area=(0, 0, 1280, 720)
    )
    assert values == dict(width=1280, height=720, left=0, top=0)


async def test_owner_results_are_request_correlated_and_failures_are_safe(caplog):
    blocker = asyncio.Event()

    async def submit(text):
        if text == "first":
            await blocker.wait()
        if text == "secret-input":
            raise RuntimeError("secret-input sk-test-token")
        return str(uuid4()), text.upper()

    async def noop(*args):
        return None

    owner = QuickTranslateOwner(
        spawn=noop, submit=submit, preferences=lambda: {}, save_preferences=noop
    )
    process = object()
    owner.process = process
    results = []

    async def send(result):
        results.append(result)

    owner.send = send
    first = asyncio.create_task(owner.translate("id1", "first", process))
    await owner.translate("id2", "second", process)
    blocker.set()
    await first
    await owner.translate("id3", "secret-input", process)
    assert [(r["id"], r.get("translation")) for r in results] == [
        ("id2", "SECOND"),
        ("id1", "FIRST"),
        ("id3", None),
    ]
    assert "code=translation_failed" in caplog.text
    assert "secret-input" not in caplog.text and "sk-test-token" not in caplog.text
