from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

pytest.importorskip("flet")

from puripuly_heart.core.clock import FakeClock
from puripuly_heart.core.osc.chatbox_paginator import ChatboxPaginator
from puripuly_heart.ui import controller as controller_module
from puripuly_heart.ui.controller import GuiController
from tests.helpers.fakes import FakeSender


def _controller_with_osc(clock: FakeClock, sender: FakeSender) -> GuiController:
    controller = GuiController(
        page=SimpleNamespace(run_task=lambda _task: None),
        app=SimpleNamespace(),
        config_path=Path("settings.json"),
    )
    controller.clock = clock
    controller.osc = ChatboxPaginator(sender=sender, clock=clock)
    return controller


@pytest.mark.asyncio
async def test_manual_input_activity_uses_idle_timeout_without_repeated_osc_sends(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(controller_module, "MANUAL_TYPING_IDLE_POLL_S", 0.001)
    clock = FakeClock()
    sender = FakeSender()
    controller = _controller_with_osc(clock, sender)

    controller.note_manual_input_activity(True)
    controller.note_manual_input_activity(True)

    assert sender.typing == [True]

    clock.advance(controller_module.MANUAL_TYPING_IDLE_TIMEOUT_S)
    await asyncio.sleep(0.05)

    assert sender.typing == [True, False]
    await controller._reset_manual_typing_state()


@pytest.mark.asyncio
async def test_manual_submit_keeps_typing_visible_until_submit_finishes() -> None:
    clock = FakeClock()
    sender = FakeSender()
    controller = _controller_with_osc(clock, sender)
    submitted: list[tuple[str, str]] = []

    class Hub:
        async def submit_text(self, text: str, *, source: str) -> None:
            submitted.append((text, source))
            await asyncio.sleep(0)

    controller.hub = Hub()

    controller.note_manual_input_activity(True)
    await controller.submit_text("hello")

    assert submitted == [("hello", "You")]
    assert sender.typing == [True, False]
    await controller._reset_manual_typing_state()


@pytest.mark.asyncio
async def test_manual_submit_waits_for_deferred_translation_task_before_clearing() -> None:
    clock = FakeClock()
    sender = FakeSender()
    controller = _controller_with_osc(clock, sender)
    release_translation = asyncio.Event()
    output_done: list[str] = []

    class Runtime:
        def __init__(self) -> None:
            self.translation_tasks: dict[object, asyncio.Task[None]] = {}

    class Hub:
        def __init__(self) -> None:
            self.self_runtime = Runtime()

        async def submit_text(self, text: str, *, source: str) -> object:
            _ = (text, source)
            utterance_id = uuid4()

            async def _translate() -> None:
                await release_translation.wait()
                output_done.append("done")

            task = asyncio.create_task(_translate())
            self.self_runtime.translation_tasks[utterance_id] = task
            task.add_done_callback(
                lambda _task: self.self_runtime.translation_tasks.pop(utterance_id, None)
            )
            return utterance_id

    controller.hub = Hub()

    submit_task = asyncio.create_task(controller.submit_text("hello"))
    await asyncio.sleep(0)

    assert sender.typing == [True]
    assert output_done == []

    release_translation.set()
    await submit_task

    assert output_done == ["done"]
    assert sender.typing == [True, False]
    await controller._reset_manual_typing_state()


@pytest.mark.asyncio
async def test_overlapping_manual_submits_keep_typing_until_all_finish() -> None:
    clock = FakeClock()
    sender = FakeSender()
    controller = _controller_with_osc(clock, sender)
    release_by_text = {"one": asyncio.Event(), "two": asyncio.Event()}
    output_done: list[str] = []

    class Runtime:
        def __init__(self) -> None:
            self.translation_tasks: dict[object, asyncio.Task[None]] = {}

    class Hub:
        def __init__(self) -> None:
            self.self_runtime = Runtime()

        async def submit_text(self, text: str, *, source: str) -> object:
            _ = source
            utterance_id = uuid4()

            async def _translate() -> None:
                await release_by_text[text].wait()
                output_done.append(text)

            task = asyncio.create_task(_translate())
            self.self_runtime.translation_tasks[utterance_id] = task
            task.add_done_callback(
                lambda _task: self.self_runtime.translation_tasks.pop(utterance_id, None)
            )
            return utterance_id

    controller.hub = Hub()

    first = asyncio.create_task(controller.submit_text("one"))
    second = asyncio.create_task(controller.submit_text("two"))
    await asyncio.sleep(0)

    assert sender.typing == [True]

    release_by_text["one"].set()
    await first

    assert sender.typing == [True]

    release_by_text["two"].set()
    await second

    assert output_done == ["one", "two"]
    assert sender.typing == [True, False]
    await controller._reset_manual_typing_state()
