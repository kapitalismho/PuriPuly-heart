from __future__ import annotations

from dataclasses import dataclass, field
from uuid import uuid4

import pytest

from puripuly_heart.core.orchestrator.configuration import TranslationRuntimeConfig
from puripuly_heart.core.runtime.output import OutputRuntime
from puripuly_heart.domain.events import (
    SelfTurnPresentation,
    SelfTurnTargetPresentation,
)
from puripuly_heart.domain.models import OSCMessage, Translation


@dataclass(slots=True)
class RecordingChatbox:
    messages: list[OSCMessage] = field(default_factory=list)
    typing_reasons: set[str] = field(default_factory=set)

    def enqueue(self, message: OSCMessage) -> None:
        self.messages.append(message)

    def set_typing_reason(self, reason: str, active: bool) -> None:
        if active:
            self.typing_reasons.add(reason)
        else:
            self.typing_reasons.discard(reason)

    def clear_typing_reasons(self) -> None:
        self.typing_reasons.clear()

    def drop_pending(self) -> None:
        self.messages.clear()


@pytest.mark.asyncio
async def test_late_active_self_turn_after_close_owns_no_new_batch() -> None:
    chatbox = RecordingChatbox()
    owner = OutputRuntime(chatbox=chatbox)
    utterance_id = uuid4()
    snapshot = SelfTurnPresentation(
        utterance_id=utterance_id,
        turn_generation=1,
        turn_order=0,
        revision=1,
        source_revision=1,
        source_text="late source",
        source_language="en",
        source_closed=True,
        status="active",
        targets=(
            SelfTurnTargetPresentation(
                target_language="ko",
                source_revision=1,
                status="ready",
                translation=Translation(utterance_id=utterance_id, text="late translation"),
            ),
        ),
        created_at=1.0,
    )

    await owner.start()
    await owner.close()
    assert not owner.has_resources

    try:
        await owner.publish_self_turn(snapshot, TranslationRuntimeConfig())

        assert chatbox.messages == []
        admission = owner.overlay_admission_snapshot()
        assert (admission["batches"], owner.has_resources) == (0, False)
    finally:
        await owner.close()
