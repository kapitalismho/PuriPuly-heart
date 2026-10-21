from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest
from puripuly_heart.app.wiring_composition import create_manual_typing_owner

from puripuly_heart.core.orchestrator.self_translation_channel import TextSubmissionAdmission


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


@pytest.mark.asyncio
async def test_composed_manual_typing_owner_uses_injected_self_output_and_completion() -> None:
    output = RecordingSelfTypingOutput()
    release = asyncio.Event()
    completion = asyncio.create_task(release.wait())
    admission = TextSubmissionAdmission("admitted", uuid4(), ())
    detailed: list[str] = []
    errors: list[str] = []
    owner = create_manual_typing_owner(
        output_provider=lambda: output,
        completion_provider=lambda utterance_id: (
            completion if utterance_id == admission.turn_id else None
        ),
        log_diagnostic=detailed.append,
        idle_timeout_seconds=0.01,
        submit_timeout_seconds=0.1,
    )

    owner.set_input_activity(True)
    assert output.reasons == {"manual_input"}

    async def submit() -> TextSubmissionAdmission:
        return admission

    submit_task = asyncio.create_task(owner.submit(submit))
    await asyncio.sleep(0)

    assert output.reasons == {"manual_submit:1"}
    release.set()
    assert await submit_task is admission
    assert output.reasons == set()

    owner.set_input_activity(True)
    await owner.release()

    assert owner.idle_task is None
    assert output.reasons == set()
    assert ("clear", None) in output.calls
    assert detailed[-1] == "[ManualTyping] release status=cleared"
    assert errors == []
