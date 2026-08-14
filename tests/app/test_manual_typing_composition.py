from __future__ import annotations

import asyncio

import pytest
from puripuly_heart.app.wiring_composition import create_manual_typing_owner


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
    detailed: list[str] = []
    errors: list[str] = []
    owner = create_manual_typing_owner(
        output_provider=lambda: output,
        completion_provider=lambda utterance_id: (
            completion if utterance_id == "utterance" else None
        ),
        log_detailed=detailed.append,
        log_error=errors.append,
        idle_timeout_seconds=0.01,
        submit_timeout_seconds=0.1,
    )

    owner.set_input_activity(True)
    assert output.reasons == {"manual_input"}

    async def submit() -> str:
        return "utterance"

    submit_task = asyncio.create_task(owner.submit(submit))
    await asyncio.sleep(0)

    assert output.reasons == {"manual_submit:1"}
    release.set()
    await submit_task
    assert output.reasons == set()

    owner.set_input_activity(True)
    await owner.release()

    assert owner.idle_task is None
    assert output.reasons == set()
    assert ("clear", None) in output.calls
    assert detailed[-1] == "[ManualTyping] release status=cleared"
    assert errors == []
