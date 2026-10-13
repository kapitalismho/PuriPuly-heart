from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from uuid import UUID

from puripuly_heart.core.clock import FakeClock
from puripuly_heart.core.orchestrator.configuration import (
    TranslationRuntimeConfig,
    TranslationRuntimeConfigSnapshot,
)
from puripuly_heart.core.orchestrator.translation_turn import SourceChild
from puripuly_heart.core.overlay.presenter import OverlayPresenter
from puripuly_heart.core.overlay.sink import OverlayEventAdapter, OverlayPublicationScope
from puripuly_heart.domain.models import SpeakerAssignment, SpeakerAttribution, SpeakerKey
from puripuly_heart.ui.overlay_calibration import OverlayCalibration


def source_child(order: int, text: str, *, group: UUID | None, language: str = "en") -> SourceChild:
    return SourceChild(
        source_id=UUID(int=217000 + order),
        revision=f"source-{order}",
        channel="peer",
        order=(1, order, 0),
        raw_text=text,
        language=language,
        text_range=(0, len(text)),
        origins=(),
        speaker=None,
        settings=TranslationRuntimeConfigSnapshot(1, TranslationRuntimeConfig()),
        display_group_id=group,
    )


def assignment(
    order: int,
    speaker: str | None,
    *,
    scope: str = "epoch",
    index: int | None = 0,
    overflow: bool = False,
) -> SpeakerAssignment:
    return SpeakerAssignment(
        SpeakerAttribution(
            "identified" if speaker is not None else "missing",
            SpeakerKey("nemotron", scope, speaker) if speaker is not None else None,
            "nemotron",
            scope,
        ),
        (1, order, 0),
        index,
        palette_overflow=overflow,
    )


def translated_event(adapter, source, text, speaker_assignment, *, target="ko", attempt=None):
    event = adapter.translation_final(
        utterance_id=attempt or UUID(int=218000 + source.order[1]),
        channel="peer",
        text=text,
        source_text=source.effective_text,
        source_language=source.language or "",
        target_language=target,
        applied_context_mode="integrated",
        speaker_assignment=speaker_assignment,
        output_scope=OverlayPublicationScope(
            "peer", UUID(int=219000 + source.order[1]), 1, source.order[1]
        ),
    )
    return replace(event, source_child=source)


async def projected_source_cards() -> list[dict[str, object]]:
    clock = FakeClock(_now=10.0)
    presenter = OverlayPresenter(
        calibration=OverlayCalibration(anchor="spatial_locked"),
        clock=clock,
        native_retry_enabled=True,
    )
    adapter = OverlayEventAdapter(clock=clock)
    group = UUID(int=217)
    snapshots = []
    try:
        for order, source_text, translated, speaker, index in (
            (1, "same phrase", "First A", "A", 0),
            (2, "B words", "B", "B", 1),
            (3, "same phrase", "Next A", "A", 0),
        ):
            source = source_child(order, source_text, group=group)
            await presenter.emit(
                translated_event(
                    adapter, source, translated, assignment(order, speaker, index=index)
                )
            )
            snapshots.append(presenter.snapshot().to_dict())
            clock.advance(1.0)
        await presenter.begin_native_retry_epoch(enabled=True)
        snapshots.append(presenter.snapshot().to_dict())
        return snapshots
    finally:
        await presenter.close()


if __name__ == "__main__":
    print(json.dumps(asyncio.run(projected_source_cards())))
