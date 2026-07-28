from __future__ import annotations

import asyncio

import pytest

from puripuly_heart.app.ports.manual_typing import SelfChatboxTypingPort
from puripuly_heart.app.services.manual_typing import ManualTypingOwner


class RecordingSelfTypingOutput:
    def __init__(self) -> None:
        self.reasons: set[str] = set()
        self.calls: list[tuple[str, bool] | tuple[str, None]] = []

    def set_self_chatbox_typing_reason(self, reason: str, active: bool) -> None:
        self.calls.append((reason, active))
        if active:
            self.reasons.add(reason)
        else:
            self.reasons.discard(reason)

    def clear_self_chatbox_typing_reasons(self) -> None:
        self.calls.append(("clear", None))
        self.reasons.clear()


def _owner(
    output: RecordingSelfTypingOutput,
    *,
    completion_provider=lambda _utterance_id: None,
) -> tuple[ManualTypingOwner, list[str], list[str]]:
    detailed: list[str] = []
    errors: list[str] = []
    owner = ManualTypingOwner(
        output_provider=lambda: output,
        completion_provider=completion_provider,
        log_detailed=lambda message: detailed.append(message),
        log_error=lambda message: errors.append(message),
        idle_timeout_seconds=0.01,
        submit_timeout_seconds=0.1,
    )
    return owner, detailed, errors


@pytest.mark.asyncio
async def test_owner_clears_input_typing_on_empty_idle_and_release() -> None:
    output = RecordingSelfTypingOutput()
    owner, detailed, _ = _owner(output)

    owner.set_input_activity(True)
    assert output.reasons == {"manual_input"}
    owner.set_input_activity(False)
    assert output.reasons == set()

    owner.set_input_activity(True)
    await asyncio.sleep(0.02)
    assert output.reasons == set()

    owner.set_input_activity(True)
    idle_task = owner.idle_task
    assert idle_task is not None
    await owner.release()

    assert owner.idle_task is None
    assert idle_task.done()
    assert idle_task.cancelled()
    assert output.reasons == set()
    assert ("clear", None) in output.calls
    assert detailed[-1] == "[ManualTyping] release status=cleared"


@pytest.mark.asyncio
async def test_owner_keeps_overlapping_submit_generations_independent() -> None:
    output = RecordingSelfTypingOutput()
    release = asyncio.Event()
    completion = asyncio.create_task(release.wait())
    owner, _, _ = _owner(
        output,
        completion_provider=lambda _utterance_id: completion,
    )

    async def submit() -> str:
        return "utterance"

    first = asyncio.create_task(owner.submit(submit))
    await asyncio.sleep(0)
    assert output.reasons == {"manual_submit:1"}

    assert owner.begin_submit() == "manual_submit:2"
    release.set()
    await first

    assert output.reasons == {"manual_submit:2"}
    assert owner.submit_generation == 2


@pytest.mark.asyncio
async def test_owner_clears_submit_reason_after_submission_failure() -> None:
    output = RecordingSelfTypingOutput()
    owner, _, errors = _owner(output)

    async def fail() -> object:
        raise RuntimeError("boom")

    await owner.submit(fail)

    assert output.reasons == set()
    assert errors == ["Submit failed: boom"]


def test_typing_output_port_exposes_only_self_channel_operations() -> None:
    members = {name for name in SelfChatboxTypingPort.__dict__ if not name.startswith("_")}

    assert members == {
        "set_self_chatbox_typing_reason",
        "clear_self_chatbox_typing_reasons",
    }
