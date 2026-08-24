from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from uuid import UUID, uuid4

import numpy as np
import pytest

from puripuly_heart.core import messages
from puripuly_heart.core.clock import FakeClock
from puripuly_heart.core.llm.provider import LLMProvider
from puripuly_heart.core.orchestrator.channel_runtime import _MergeBuffer
from puripuly_heart.core.orchestrator.peer_translation_channel import (
    PeerTranslationChannelOwner,
)
from puripuly_heart.core.orchestrator.translation_output_projection import (
    TranslationOverlayProjection,
)
from puripuly_heart.core.orchestrator.translation_request import DirectTranslationRequest
from puripuly_heart.core.overlay.diagnostics import OverlayDiagnosticsRecorder
from puripuly_heart.core.overlay.presenter import OverlayPresenter
from puripuly_heart.core.overlay.sink import OverlayEventAdapter
from puripuly_heart.core.overlay.state import ActiveSelfOverlayMetadata
from puripuly_heart.core.runtime_logging import (
    LATENCY_TRACE_POINT_CONTRACTS,
    SessionLoggingMode,
)
from puripuly_heart.core.translation_backend import LlmTranslationBackend
from puripuly_heart.core.vad.gating import SpeechChunk, SpeechEnd, SpeechStart
from puripuly_heart.domain.events import STTFinalEvent, STTPartialEvent, UIEventType
from puripuly_heart.domain.models import FinalLanguageRun, Transcript, Translation
from puripuly_heart.ui.overlay_calibration import OverlayCalibration
from tests.core.test_translation_owner_branch_coverage import (
    _make_runtime_logging_capture,
    _runtime_log_messages,
)
from tests.helpers.fakes import RecordingOscQueue
from tests.helpers.translation_owners import (
    compose_translation_test_harness,
    make_speculative_attempt,
)

_TRANSLATION_ACTIVE_SELF_MIRROR_FIELDS = {
    "_overlay_active_self_text",
    "_overlay_active_self_secondary_text",
    "_overlay_active_self_utterance_id",
    "_overlay_active_self_occupant_key",
    "_overlay_active_self_update_id",
    "_overlay_active_self_origin_wall_clock_ms",
    "_overlay_active_self_session_scope",
    "_overlay_active_self_source_text_hash",
    "_overlay_active_self_source_text_len",
    "_overlay_active_self_logical_turn_key",
}


def test_translation_does_not_declare_active_self_overlay_mirror_fields() -> None:
    assert _TRANSLATION_ACTIVE_SELF_MIRROR_FIELDS.isdisjoint(
        PeerTranslationChannelOwner.__dataclass_fields__
    )


@dataclass(slots=True)
class RecordingOverlaySink:
    events: list[object] = field(default_factory=list)
    active_self_metadata: ActiveSelfOverlayMetadata | None = None

    async def emit(self, event: object) -> None:
        self.events.append(event)
        event_type = getattr(event, "type", None)
        if event_type == "self_active_update":
            utterance_id = getattr(event, "utterance_id", None)
            if not isinstance(utterance_id, UUID):
                return
            self.active_self_metadata = ActiveSelfOverlayMetadata(
                text=getattr(event, "text", ""),
                secondary_text=getattr(event, "secondary_text", ""),
                utterance_id=utterance_id,
                occupant_key=getattr(event, "occupant_key", ""),
                update_id=getattr(event, "update_id", None),
                origin_wall_clock_ms=getattr(event, "origin_wall_clock_ms", None),
                session_scope=getattr(event, "session_scope", None),
                source_text_hash=getattr(event, "source_text_hash", None),
                source_text_len=getattr(event, "source_text_len", None),
                logical_turn_key=getattr(event, "logical_turn_key", None),
                primary_language=(str(getattr(event, "source_language", "") or "").strip() or None),
                secondary_language=(
                    str(getattr(event, "target_language", "") or "").strip() or None
                    if getattr(event, "secondary_text", "").strip()
                    else None
                ),
            )
        elif event_type == "self_active_clear":
            self.active_self_metadata = None
        elif event_type == "self_transcript_final" and self.active_self_metadata is not None:
            if self.active_self_metadata.utterance_id == getattr(event, "utterance_id", None):
                self.active_self_metadata = None

    def active_self_overlay_metadata(self) -> ActiveSelfOverlayMetadata | None:
        return self.active_self_metadata


def _active_self_metadata_for_buffer(
    buffer: _MergeBuffer,
    *,
    text: str,
    secondary_text: str,
    update_id: str | None = None,
    origin_wall_clock_ms: int | None = None,
    session_scope: str | None = None,
    source_text_hash: str | None = None,
    source_text_len: int | None = None,
    logical_turn_key: str | None = None,
) -> ActiveSelfOverlayMetadata:
    return ActiveSelfOverlayMetadata(
        text=text,
        secondary_text=secondary_text,
        utterance_id=buffer.merge_id,
        occupant_key=f"self:{buffer.merge_id}",
        update_id=update_id,
        origin_wall_clock_ms=origin_wall_clock_ms,
        session_scope=session_scope,
        source_text_hash=source_text_hash,
        source_text_len=source_text_len,
        logical_turn_key=logical_turn_key,
    )


@dataclass(slots=True)
class RecordingTranslationDiagnostics:
    translation_events: list[dict[str, object]] = field(default_factory=list)
    stt_events: list[dict[str, object]] = field(default_factory=list)

    def record_translation(self, event: str, **fields: object) -> dict[str, object]:
        payload = {"event": event, **fields}
        self.translation_events.append(payload)
        return payload

    def record_stt(self, event: str, **fields: object) -> dict[str, object]:
        payload = {"event": event, **fields}
        self.stt_events.append(payload)
        return payload


@dataclass(slots=True)
class RecordingPresentationBridge:
    snapshots: list[object] = field(default_factory=list)

    async def replace_snapshot(self, snapshot: object) -> None:
        self.snapshots.append(snapshot)

    async def broadcast_shutdown(self) -> None:
        return


@dataclass(slots=True)
class FailingOverlaySink:
    async def emit(self, event: object) -> None:
        _ = event
        raise RuntimeError("overlay boom")


@dataclass(slots=True)
class ImmediateFailingTranslateLLMProvider(LLMProvider):
    error: Exception

    async def translate(
        self,
        *,
        utterance_id: UUID,
        text: str,
        system_prompt: str,
        source_language: str,
        target_language: str,
        context: str = "",
    ):
        _ = (utterance_id, text, system_prompt, source_language, target_language, context)
        raise self.error

    async def close(self) -> None:
        return


@dataclass(slots=True)
class StubTranslateLLMProvider(LLMProvider):
    text: str

    async def translate(
        self,
        *,
        utterance_id: UUID,
        text: str,
        system_prompt: str,
        source_language: str,
        target_language: str,
        context: str = "",
    ) -> Translation:
        _ = (system_prompt, context)
        return Translation(
            utterance_id=utterance_id,
            text=self.text,
            source_text=text,
            source_language=source_language,
            target_language=target_language,
        )

    async def close(self) -> None:
        return


@dataclass(slots=True)
class BlockingTranslateLLMProvider(LLMProvider):
    started: asyncio.Event = field(default_factory=asyncio.Event)
    release: asyncio.Future[None] | None = None

    async def translate(
        self,
        *,
        utterance_id: UUID,
        text: str,
        system_prompt: str,
        source_language: str,
        target_language: str,
        context: str = "",
    ):
        _ = (utterance_id, text, system_prompt, source_language, target_language, context)
        self.started.set()
        if self.release is None:
            self.release = asyncio.get_running_loop().create_future()
        await self.release
        raise AssertionError("blocking provider should be cancelled before release")

    async def close(self) -> None:
        return


@dataclass(slots=True)
class CancelSuppressingTranslateLLMProvider(LLMProvider):
    started: asyncio.Event = field(default_factory=asyncio.Event)

    async def translate(
        self,
        *,
        utterance_id: UUID,
        text: str,
        system_prompt: str,
        source_language: str,
        target_language: str,
        context: str = "",
    ) -> Translation:
        _ = (text, system_prompt, source_language, target_language, context)
        self.started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            return Translation(utterance_id=utterance_id, text="cancelled success")
        raise AssertionError("suppressed cancellation provider should be cancelled")

    async def close(self) -> None:
        return


@dataclass(slots=True)
class ReleasableTranslateLLMProvider(LLMProvider):
    response_text: str
    response_source_language: str | None = None
    response_target_language: str | None = None
    started: asyncio.Event = field(default_factory=asyncio.Event)
    release: asyncio.Future[None] | None = None
    calls: list[str] = field(default_factory=list)
    requested_source_language: str | None = None
    requested_target_language: str | None = None

    async def translate(
        self,
        *,
        utterance_id: UUID,
        text: str,
        system_prompt: str,
        source_language: str,
        target_language: str,
        context: str = "",
    ):
        _ = (system_prompt, context)
        self.requested_source_language = source_language
        self.requested_target_language = target_language
        self.calls.append(text)
        self.started.set()
        if self.release is None:
            self.release = asyncio.get_running_loop().create_future()
        await self.release
        return Translation(
            utterance_id=utterance_id,
            text=self.response_text,
            source_language=self.response_source_language,
            target_language=self.response_target_language,
        )

    async def close(self) -> None:
        return


@dataclass(slots=True)
class ClockedTranslateLLMProvider(LLMProvider):
    clock: FakeClock
    responses: list[tuple[float, str]]

    async def translate(
        self,
        *,
        utterance_id: UUID,
        text: str,
        system_prompt: str,
        source_language: str,
        target_language: str,
        context: str = "",
    ):
        _ = (utterance_id, text, system_prompt, source_language, target_language, context)
        if not self.responses:
            raise AssertionError("no translate response configured")
        delay_s, response_text = self.responses.pop(0)
        self.clock.advance(delay_s)
        return Translation(utterance_id=utterance_id, text=response_text)

    async def close(self) -> None:
        return


@dataclass(slots=True)
class SequencedTranslateLLMProvider(LLMProvider):
    responses: list[str]
    delay_s: float = 0.01
    calls: list[str] = field(default_factory=list)

    async def translate(
        self,
        *,
        utterance_id: UUID,
        text: str,
        system_prompt: str,
        source_language: str,
        target_language: str,
        context: str = "",
    ):
        _ = (utterance_id, system_prompt, source_language, target_language, context)
        self.calls.append(text)
        await asyncio.sleep(self.delay_s)
        if not self.responses:
            raise AssertionError("no translate response configured")
        return Translation(utterance_id=utterance_id, text=self.responses.pop(0))

    async def close(self) -> None:
        return


@dataclass(slots=True)
class RecordingSequencedTranslateLLMProvider(LLMProvider):
    responses: list[str]
    delay_s: float = 0.01
    calls: list[tuple[UUID, str]] = field(default_factory=list)

    async def translate(
        self,
        *,
        utterance_id: UUID,
        text: str,
        system_prompt: str,
        source_language: str,
        target_language: str,
        context: str = "",
    ):
        _ = (system_prompt, source_language, target_language, context)
        self.calls.append((utterance_id, text))
        await asyncio.sleep(self.delay_s)
        if not self.responses:
            raise AssertionError("no translate response configured")
        return Translation(utterance_id=utterance_id, text=self.responses.pop(0))

    async def close(self) -> None:
        return


@dataclass(slots=True)
class GatedRecordingTranslateLLMProvider(LLMProvider):
    responses: list[str]
    start_target: int
    calls: list[tuple[UUID, str, str]] = field(default_factory=list)
    all_started: asyncio.Event = field(default_factory=asyncio.Event)
    release: asyncio.Future[None] | None = None

    async def translate(
        self,
        *,
        utterance_id: UUID,
        text: str,
        system_prompt: str,
        source_language: str,
        target_language: str,
        context: str = "",
    ):
        _ = (system_prompt, source_language, target_language)
        response_index = len(self.calls)
        self.calls.append((utterance_id, text, context))
        if len(self.calls) >= self.start_target:
            self.all_started.set()
        if self.release is None:
            self.release = asyncio.get_running_loop().create_future()
        await self.release
        if response_index >= len(self.responses):
            raise AssertionError("no translate response configured")
        return Translation(
            utterance_id=utterance_id,
            text=self.responses[response_index],
        )

    async def close(self) -> None:
        return


@pytest.mark.asyncio
async def test_translation_emits_self_and_peer_finals_to_overlay_sink() -> None:
    sink = RecordingOverlaySink()
    harness = compose_translation_test_harness(
        stt=None, llm=None, osc=RecordingOscQueue(), overlay_sink=sink
    )

    await harness.self_owner.submit_text("self text", source="You")
    await harness.handle_peer_transcript_final_for_test(text="peer text")

    assert [event.type for event in sink.events] == [
        "self_transcript_final",
        "utterance_closed",
        "peer_transcript_final",
        "utterance_closed",
    ]
    assert [event.channel for event in sink.events] == ["self", "self", "peer", "peer"]


@pytest.mark.asyncio
async def test_translation_active_self_overlay_snapshot_uses_spec_translation_languages_not_current_settings() -> (
    None
):
    presenter = OverlayPresenter(
        calibration=OverlayCalibration(),
        peer_presentation_refresh_burst=False,
    )
    harness = compose_translation_test_harness(
        stt=None,
        llm=None,
        osc=RecordingOscQueue(),
        overlay_sink=presenter,
        source_language="ko",
        target_language="en",
    )
    merge_id = uuid4()
    source_utterance_id = uuid4()
    buffer = _MergeBuffer(
        merge_id=merge_id,
        parts=["こんにちは"],
        utterance_ids=[source_utterance_id],
        speculative_attempt=make_speculative_attempt(
            source_text="こんにちは",
            result=Translation(
                utterance_id=merge_id,
                text="hello",
                source_text="こんにちは",
                source_language="ja",
                target_language="zh-TW",
            ),
        ),
    )

    await harness.self_owner._sync_overlay_active_self(buffer)

    block = presenter.snapshot().blocks[0]
    assert block.channel == "self"
    assert block.block_variant == "active_self"
    assert block.primary_text == "こんにちは"
    assert block.secondary_text == "hello"
    assert block.primary_language == "ja"
    assert block.secondary_language == "zh-TW"


@pytest.mark.asyncio
async def test_translation_blank_spec_translation_active_update_keeps_spec_language_metadata() -> (
    None
):
    sink = RecordingOverlaySink()
    harness = compose_translation_test_harness(
        stt=None,
        llm=None,
        osc=RecordingOscQueue(),
        overlay_sink=sink,
        source_language="ko",
        target_language="en",
    )
    merge_id = uuid4()
    buffer = _MergeBuffer(
        merge_id=merge_id,
        parts=["こんにちは"],
        utterance_ids=[uuid4()],
        speculative_attempt=make_speculative_attempt(
            source_text="こんにちは",
            result=Translation(
                utterance_id=merge_id,
                text="   ",
                source_text="こんにちは",
                source_language="ja",
                target_language="zh-TW",
            ),
        ),
    )

    await harness.self_owner._sync_overlay_active_self(buffer)

    assert len(sink.events) == 1
    event = sink.events[0]
    assert getattr(event, "type", None) == "self_active_update"
    assert event.secondary_text == ""
    assert event.source_language == "ja"
    assert event.target_language == "zh-TW"


@pytest.mark.asyncio
async def test_translation_same_text_blank_spec_language_update_feeds_final_transcript_language() -> (
    None
):
    presenter = OverlayPresenter(
        calibration=OverlayCalibration(),
        peer_presentation_refresh_burst=False,
    )
    adapter = OverlayEventAdapter(clock=FakeClock(_now=10.0))
    merge_id = uuid4()
    await presenter.emit(
        adapter.self_active_update(
            text="こんにちは",
            secondary_text="",
            utterance_id=merge_id,
            occupant_key=f"self:{merge_id}",
            source_language="ko",
            target_language="en",
            created_at=10.0,
        )
    )
    harness = compose_translation_test_harness(
        stt=None,
        llm=None,
        osc=RecordingOscQueue(),
        overlay_sink=presenter,
        source_language="ko",
        target_language="en",
    )
    buffer = _MergeBuffer(
        merge_id=merge_id,
        parts=["こんにちは"],
        utterance_ids=[uuid4()],
        speculative_attempt=make_speculative_attempt(
            source_text="こんにちは",
            result=Translation(
                utterance_id=merge_id,
                text="   ",
                source_text="こんにちは",
                source_language="ja",
                target_language="zh-TW",
            ),
        ),
    )

    await harness.self_owner._sync_overlay_active_self(buffer)
    await harness.output_projection.project_self_final_transcript(
        transcript=Transcript(
            utterance_id=merge_id,
            text="こんにちは",
            is_final=True,
            created_at=10.1,
            channel="self",
        ),
        source_language=harness.configuration.snapshot().value.source_language,
        target_language=harness.configuration.snapshot().value.target_language,
        translation_will_follow=True,
    )

    block = presenter.snapshot().blocks[0]
    assert block.block_variant == "finalized"
    assert block.primary_text == "こんにちは"
    assert block.primary_language == "ja"


@pytest.mark.asyncio
async def test_translation_self_translation_overlay_uses_translation_languages_not_current_settings() -> (
    None
):
    presenter = OverlayPresenter(
        calibration=OverlayCalibration(),
        peer_presentation_refresh_burst=False,
    )
    harness = compose_translation_test_harness(
        stt=None,
        llm=None,
        osc=RecordingOscQueue(),
        overlay_sink=presenter,
        source_language="ko",
        target_language="en",
    )
    utterance_id = uuid4()
    await harness.output_projection.publish_overlay_event(
        harness.output_projection.overlay_event_adapter.transcript_final(
            Transcript(
                utterance_id=utterance_id,
                text="こんにちは",
                is_final=True,
                created_at=10.0,
                channel="self",
            ),
            source_language="ja",
            target_language="zh-TW",
        )
    )

    await harness.output_projection.emit_translation(
        TranslationOverlayProjection(
            translation=Translation(
                utterance_id=utterance_id,
                text="你好",
                source_text="こんにちは",
                source_language="ja",
                target_language="zh-TW",
                channel="self",
                created_at=10.1,
            ),
            source_language="ja",
            target_language="zh-TW",
            applied_context_mode=None,
        )
    )

    block = presenter.snapshot().blocks[0]
    assert block.primary_text == "こんにちは"
    assert block.secondary_text == "你好"
    assert block.primary_language == "ja"
    assert block.secondary_language == "zh-TW"


@pytest.mark.asyncio
async def test_translation_translate_and_enqueue_overlay_uses_request_language_after_settings_change() -> (
    None
):
    llm = ReleasableTranslateLLMProvider(response_text="你好")
    sink = RecordingOverlaySink()
    harness = compose_translation_test_harness(
        stt=None,
        llm=llm,
        osc=RecordingOscQueue(),
        overlay_sink=sink,
        source_language="ja",
        target_language="zh-TW",
    )
    utterance_id = uuid4()

    task = asyncio.create_task(
        harness.process_translation(
            utterance_id,
            "こんにちは",
            runtime=harness.self_runtime,
        )
    )
    await asyncio.wait_for(llm.started.wait(), timeout=1.0)
    harness.replace_configuration(source_language="ko")
    harness.replace_configuration(target_language="en")
    assert llm.requested_source_language == "ja"
    assert llm.requested_target_language == "zh-TW"
    assert llm.release is not None
    llm.release.set_result(None)

    await task

    translation_events = [
        event for event in sink.events if getattr(event, "type", None) == "translation_final"
    ]
    assert len(translation_events) == 1
    assert translation_events[0].source_language == "ja"
    assert translation_events[0].target_language == "zh-TW"


@pytest.mark.asyncio
async def test_translation_translate_text_preserves_provider_language_after_settings_change() -> (
    None
):
    llm = ReleasableTranslateLLMProvider(
        response_text="你好",
        response_source_language="ja",
        response_target_language="zh-TW",
    )
    harness = compose_translation_test_harness(
        stt=None,
        llm=llm,
        osc=RecordingOscQueue(),
        source_language="es",
        target_language="fr",
    )
    utterance_id = uuid4()

    task = asyncio.create_task(
        harness.translation_requests.translate(
            DirectTranslationRequest(
                utterance_id=utterance_id,
                text="こんにちは",
                record_latency=False,
            )
        )
    )
    await asyncio.wait_for(llm.started.wait(), timeout=1.0)
    harness.replace_configuration(source_language="ko")
    harness.replace_configuration(target_language="en")
    assert llm.requested_source_language == "es"
    assert llm.requested_target_language == "fr"
    assert llm.release is not None
    llm.release.set_result(None)

    translation = await task

    assert translation.source_language == "ja"
    assert translation.target_language == "zh-TW"


@pytest.mark.asyncio
async def test_translation_peer_translation_overlay_uses_translation_languages_not_current_settings() -> (
    None
):
    presenter = OverlayPresenter(
        calibration=OverlayCalibration(),
        peer_presentation_refresh_burst=False,
    )
    harness = compose_translation_test_harness(
        stt=None,
        llm=None,
        osc=RecordingOscQueue(),
        overlay_sink=presenter,
        source_language="ko",
        target_language="en",
        peer_source_language="en",
        peer_target_language="ko",
    )
    utterance_id = uuid4()

    await harness.output_projection.emit_translation(
        TranslationOverlayProjection(
            translation=Translation(
                utterance_id=utterance_id,
                text="你好",
                source_text="こんにちは",
                source_language="ja",
                target_language="zh-TW",
                channel="peer",
                created_at=10.0,
            ),
            source_text="こんにちは",
            source_language="ja",
            target_language="zh-TW",
            applied_context_mode=None,
            record_peer_first_emit=True,
        )
    )

    block = presenter.snapshot().blocks[0]
    assert block.primary_text == "你好"
    assert block.secondary_text == "こんにちは"
    assert block.primary_language == "zh-TW"
    assert block.secondary_language == "ja"


@pytest.mark.asyncio
async def test_translation_active_self_sticky_secondary_preserves_cached_secondary_language() -> (
    None
):
    presenter = OverlayPresenter(
        calibration=OverlayCalibration(),
        peer_presentation_refresh_burst=False,
    )
    adapter = OverlayEventAdapter(clock=FakeClock(_now=10.0))
    merge_id = uuid4()
    await presenter.emit(
        adapter.self_active_update(
            text="こんにちは",
            secondary_text="你好",
            utterance_id=merge_id,
            occupant_key=f"self:{merge_id}",
            source_language="ja",
            target_language="zh-TW",
            created_at=10.0,
        )
    )
    harness = compose_translation_test_harness(
        stt=None,
        llm=None,
        osc=RecordingOscQueue(),
        overlay_sink=presenter,
        source_language="ko",
        target_language="en",
    )
    buffer = _MergeBuffer(
        merge_id=merge_id,
        parts=["こんにちは続き"],
        utterance_ids=[uuid4()],
    )

    await harness.self_owner._sync_overlay_active_self(buffer)

    block = presenter.snapshot().blocks[0]
    assert block.primary_text == "こんにちは続き"
    assert block.secondary_text == "你好"
    assert block.secondary_language == "zh-TW"


@pytest.mark.asyncio
async def test_translation_active_self_blank_secondary_preserves_cached_primary_language() -> None:
    presenter = OverlayPresenter(
        calibration=OverlayCalibration(),
        peer_presentation_refresh_burst=False,
    )
    adapter = OverlayEventAdapter(clock=FakeClock(_now=10.0))
    merge_id = uuid4()
    await presenter.emit(
        adapter.self_active_update(
            text="こんにちは",
            secondary_text="",
            utterance_id=merge_id,
            occupant_key=f"self:{merge_id}",
            source_language="ja",
            target_language="zh-TW",
            created_at=10.0,
        )
    )
    harness = compose_translation_test_harness(
        stt=None,
        llm=None,
        osc=RecordingOscQueue(),
        overlay_sink=presenter,
        source_language="ko",
        target_language="en",
    )
    buffer = _MergeBuffer(
        merge_id=merge_id,
        parts=["こんにちは続き"],
        utterance_ids=[uuid4()],
    )

    await harness.self_owner._sync_overlay_active_self(buffer)

    block = presenter.snapshot().blocks[0]
    assert block.primary_text == "こんにちは続き"
    assert block.secondary_text == ""
    assert block.primary_language == "ja"


@pytest.mark.asyncio
async def test_translation_self_final_transcript_preserves_active_display_language_metadata() -> (
    None
):
    presenter = OverlayPresenter(
        calibration=OverlayCalibration(),
        peer_presentation_refresh_burst=False,
    )
    harness = compose_translation_test_harness(
        stt=None,
        llm=None,
        osc=RecordingOscQueue(),
        overlay_sink=presenter,
        source_language="ko",
        target_language="en",
    )
    merge_id = uuid4()
    buffer = _MergeBuffer(
        merge_id=merge_id,
        parts=["こんにちは"],
        utterance_ids=[uuid4()],
        speculative_attempt=make_speculative_attempt(
            source_text="こんにちは",
            result=Translation(
                utterance_id=merge_id,
                text="你好",
                source_text="こんにちは",
                source_language="ja",
                target_language="zh-TW",
            ),
        ),
    )
    await harness.self_owner._sync_overlay_active_self(buffer)

    await harness.output_projection.project_self_final_transcript(
        transcript=Transcript(
            utterance_id=merge_id,
            text="こんにちは",
            is_final=True,
            created_at=10.0,
            channel="self",
        ),
        source_language=harness.configuration.snapshot().value.source_language,
        target_language=harness.configuration.snapshot().value.target_language,
        translation_will_follow=True,
    )

    block = presenter.snapshot().blocks[0]
    assert block.block_variant == "finalized"
    assert block.primary_text == "こんにちは"
    assert block.secondary_text == "你好"
    assert block.primary_language == "ja"
    assert block.secondary_language == "zh-TW"


@pytest.mark.asyncio
async def test_translation_stale_secondary_blanking_preserves_active_primary_language() -> None:
    bridge = RecordingPresentationBridge()
    presenter = OverlayPresenter(
        bridge=bridge,
        calibration=OverlayCalibration(),
        peer_presentation_refresh_burst=False,
    )
    adapter = OverlayEventAdapter(clock=FakeClock(_now=10.0))
    merge_id = uuid4()
    await presenter.emit(
        adapter.self_active_update(
            text="こんにちは",
            secondary_text="你好",
            utterance_id=merge_id,
            occupant_key=f"self:{merge_id}",
            source_language="ja",
            target_language="zh-TW",
            created_at=10.0,
        )
    )
    first_new_snapshot_index = len(bridge.snapshots)
    harness = compose_translation_test_harness(
        stt=None,
        llm=None,
        osc=RecordingOscQueue(),
        overlay_sink=presenter,
        source_language="ko",
        target_language="en",
    )
    buffer = _MergeBuffer(
        merge_id=merge_id,
        parts=["こんにちは"],
        utterance_ids=[uuid4()],
    )
    harness.self_owner.merge_buffer = buffer

    await harness.self_owner._commit_merge(buffer, reason="test_stale_secondary")

    blank_active_blocks = [
        snapshot.blocks[0]
        for snapshot in bridge.snapshots[first_new_snapshot_index:]
        if snapshot.blocks
        and snapshot.blocks[0].block_variant == "active_self"
        and snapshot.blocks[0].secondary_text == ""
    ]
    assert blank_active_blocks
    assert blank_active_blocks[0].primary_text == "こんにちは"
    assert blank_active_blocks[0].primary_language == "ja"


@pytest.mark.asyncio
async def test_translation_peer_overlay_snapshot_uses_peer_specific_source_and_target_languages() -> (
    None
):
    presenter = OverlayPresenter(
        calibration=OverlayCalibration(),
        peer_presentation_refresh_burst=False,
    )
    harness = compose_translation_test_harness(
        stt=None,
        llm=SequencedTranslateLLMProvider(responses=["你好"], delay_s=0.0),
        osc=RecordingOscQueue(),
        overlay_sink=presenter,
        source_language="ko",
        target_language="en",
        peer_source_language="ja",
        peer_target_language="zh-TW",
        peer_translation_enabled=True,
    )

    await harness.translate_peer_text_for_test("こんにちは")

    block = presenter.snapshot().blocks[0]
    assert block.channel == "peer"
    assert block.primary_text == "你好"
    assert block.secondary_text == "こんにちは"
    assert block.primary_language == "zh-TW"
    assert block.secondary_language == "ja"


@pytest.mark.asyncio
async def test_peer_source_only_overlay_emit_records_source_as_secondary_len() -> None:
    diagnostics = RecordingTranslationDiagnostics()
    sink = RecordingOverlaySink()
    harness = compose_translation_test_harness(
        stt=None,
        llm=None,
        osc=RecordingOscQueue(),
        overlay_sink=sink,
        overlay_diagnostics=diagnostics,  # type: ignore[arg-type]
    )

    await harness.handle_peer_transcript_final_for_test(text="peer source")

    source_only_events = [
        event
        for event in diagnostics.translation_events
        if event["event"] == "overlay_emit" and event["event_kind"] == "peer_transcript_final"
    ]
    assert len(source_only_events) == 1
    assert source_only_events[0]["secondary_len"] == len("peer source")


@pytest.mark.asyncio
async def test_peer_final_runs_in_one_parent_are_serial_and_close_after_last_child() -> None:
    parent_vad_id = uuid4()
    sink = RecordingOverlaySink()
    llm = RecordingSequencedTranslateLLMProvider(
        responses=["첫 번째 번역", "두 번째 번역"],
        delay_s=0.05,
    )
    harness = compose_translation_test_harness(
        stt=None,
        llm=llm,
        osc=RecordingOscQueue(),
        overlay_sink=sink,
        peer_translation_enabled=True,
    )
    await harness.peer_owner.handle_peer_vad_event(SpeechEnd(parent_vad_id))

    await harness.dispatch_stt_event(
        STTFinalEvent(
            utterance_id=parent_vad_id,
            transcript=Transcript(
                utterance_id=parent_vad_id,
                text="What about now? Can you hear me?",
                is_final=True,
                created_at=11.0,
                channel="peer",
                final_language_runs=(
                    FinalLanguageRun(text="What about now?", language="en"),
                    FinalLanguageRun(text="Can you hear me?", language="en"),
                ),
            ),
        )
    )

    assert not any(event.type == "peer_active_update" for event in sink.events)
    assert len(harness.peer_runtime.translation_tasks) == 1
    await harness.translation_turns.wait_for_idle()

    translation_events = [event for event in sink.events if event.type == "translation_final"]
    close_events = [event for event in sink.events if event.type == "utterance_closed"]
    assert [event.text for event in translation_events] == ["첫 번째 번역", "두 번째 번역"]
    assert [event.source_text for event in translation_events] == [
        "What about now?",
        "Can you hear me?",
    ]
    assert [event.utterance_id for event in close_events] == [
        event.utterance_id for event in translation_events
    ]
    ui_events = [harness.ui_events.get_nowait() for _ in range(harness.ui_events.qsize())]
    translation_done_ids = [
        event.utterance_id for event in ui_events if event.type == UIEventType.TRANSLATION_DONE
    ]
    assert translation_done_ids == [event.utterance_id for event in translation_events]
    assert parent_vad_id not in translation_done_ids
    assert [text for _utterance_id, text in llm.calls] == [
        "What about now?",
        "Can you hear me?",
    ]


@pytest.mark.asyncio
async def test_back_to_back_peer_parents_publish_in_submission_order() -> None:
    first_parent_vad_id = uuid4()
    second_parent_vad_id = uuid4()
    parent_vad_ids = [first_parent_vad_id, second_parent_vad_id]
    osc = RecordingOscQueue()
    sink = RecordingOverlaySink()
    llm = GatedRecordingTranslateLLMProvider(
        responses=["first translation", "second translation"],
        start_target=1,
    )
    harness = compose_translation_test_harness(
        stt=None,
        llm=llm,
        osc=osc,
        overlay_sink=sink,
        peer_translation_enabled=True,
        clock=FakeClock(_now=10.0),
    )

    try:
        await harness.peer_owner.handle_peer_vad_event(
            SpeechEnd(first_parent_vad_id, trailing_silence_ms=0, reason="max_duration")
        )
        await harness.dispatch_stt_event(
            STTFinalEvent(
                utterance_id=first_parent_vad_id,
                transcript=Transcript(
                    utterance_id=first_parent_vad_id,
                    text="first forced segment",
                    is_final=True,
                    created_at=11.0,
                    channel="peer",
                ),
            )
        )
        harness.clock.advance(0.1)
        await harness.peer_owner.handle_peer_vad_event(
            SpeechEnd(second_parent_vad_id, trailing_silence_ms=0, reason="max_duration")
        )
        await harness.dispatch_stt_event(
            STTFinalEvent(
                utterance_id=second_parent_vad_id,
                transcript=Transcript(
                    utterance_id=second_parent_vad_id,
                    text="second forced segment",
                    is_final=True,
                    created_at=12.0,
                    channel="peer",
                ),
            )
        )

        await asyncio.wait_for(llm.all_started.wait(), timeout=0.5)
        assert [text for _utterance_id, text, _context in llm.calls] == ["first forced segment"]
        assert len(harness.peer_runtime.translation_tasks) == 1

        assert llm.release is not None
        llm.release.set_result(None)
        await harness.translation_turns.wait_for_idle()

        translation_events = [event for event in sink.events if event.type == "translation_final"]
        close_events = [event for event in sink.events if event.type == "utterance_closed"]
        ui_events = [harness.ui_events.get_nowait() for _ in range(harness.ui_events.qsize())]
        transcript_ui_events = [
            event for event in ui_events if event.type == UIEventType.TRANSCRIPT_FINAL
        ]
        translation_done_events = [
            event for event in ui_events if event.type == UIEventType.TRANSLATION_DONE
        ]
        osc_sent_events = [event for event in ui_events if event.type == UIEventType.OSC_SENT]

        assert [event.source_text for event in translation_events] == [
            "first forced segment",
            "second forced segment",
        ]
        assert [event.logical_turn_key for event in translation_events] == [
            f"peer:{event.utterance_id}" for event in translation_events
        ]
        peer_turn_ids = [event.utterance_id for event in translation_events]
        peer_turn_id_set = set(peer_turn_ids)
        parent_vad_id_set = set(parent_vad_ids)
        assert [event.utterance_id for event in close_events] == peer_turn_ids
        assert [event.utterance_id for event in transcript_ui_events] == peer_turn_ids
        assert [event.payload.utterance_id for event in transcript_ui_events] == peer_turn_ids
        assert [event.utterance_id for event in translation_done_events] == peer_turn_ids
        assert [event.payload.utterance_id for event in translation_done_events] == peer_turn_ids
        assert osc_sent_events == []
        assert osc.messages == []

        exposed_output_ids = {
            event.utterance_id
            for event in [*sink.events, *ui_events, *osc.messages]
            if getattr(event, "utterance_id", None) is not None
        }
        assert exposed_output_ids == peer_turn_id_set
        assert parent_vad_id_set.isdisjoint(harness.peer_runtime.utterances)
        assert harness.peer_runtime.utterance_start_times == {}
        assert harness.peer_runtime.speech_ended_ids == set()
        assert not harness.translation_diagnostics.snapshot().timeline_keys
        assert harness.peer_owner._peer_parent_turn_ids == {}
        assert harness.peer_owner._peer_turn_parent_ids == {}
        assert harness.peer_owner._peer_completed_turn_ids == set()
    finally:
        await harness.stop()


@pytest.mark.asyncio
async def test_peer_partial_stt_event_remains_ignored_without_outputs_or_tasks() -> None:
    parent_vad_id = uuid4()
    sink = RecordingOverlaySink()
    osc = RecordingOscQueue()
    llm = RecordingSequencedTranslateLLMProvider(responses=["unused"], delay_s=0.0)
    harness = compose_translation_test_harness(
        stt=None,
        llm=llm,
        osc=osc,
        overlay_sink=sink,
        peer_translation_enabled=True,
    )

    await harness.dispatch_stt_event(
        STTPartialEvent(
            utterance_id=parent_vad_id,
            transcript=Transcript(
                utterance_id=parent_vad_id,
                text="partial peer speech",
                is_final=False,
                created_at=11.0,
                channel="peer",
            ),
        )
    )
    await asyncio.sleep(0)

    assert sink.events == []
    assert osc.messages == []
    assert harness.peer_runtime.utterances == {}
    assert harness.peer_runtime.translation_tasks == {}
    assert harness.peer_runtime.utterance_start_times == {}
    assert harness.peer_runtime.speech_ended_ids == set()
    assert harness.peer_owner._peer_parent_turn_ids == {}
    assert harness.peer_owner._peer_turn_parent_ids == {}
    assert harness.ui_events.empty()
    assert llm.calls == []


@pytest.mark.asyncio
async def test_identical_inflight_peer_finals_reject_the_second_final() -> None:
    parent_vad_id = uuid4()
    sink = RecordingOverlaySink()
    llm = RecordingSequencedTranslateLLMProvider(
        responses=["반복 번역 1"],
        delay_s=0.05,
    )
    harness = compose_translation_test_harness(
        stt=None,
        llm=llm,
        osc=RecordingOscQueue(),
        overlay_sink=sink,
        peer_translation_enabled=True,
    )

    for created_at in (11.0, 12.0):
        await harness.dispatch_stt_event(
            STTFinalEvent(
                utterance_id=parent_vad_id,
                transcript=Transcript(
                    utterance_id=parent_vad_id,
                    text="repeat this",
                    is_final=True,
                    created_at=created_at,
                    channel="peer",
                ),
            )
        )
    await asyncio.sleep(0)

    assert not any(event.type == "peer_active_update" for event in sink.events)
    assert len(harness.peer_runtime.translation_tasks) == 1
    await harness.translation_turns.wait_for_idle()

    close_events = [event for event in sink.events if event.type == "utterance_closed"]
    translation_events = [event for event in sink.events if event.type == "translation_final"]
    peer_turn_ids = [event.utterance_id for event in translation_events]
    assert [event.utterance_id for event in close_events] == peer_turn_ids
    assert [event.source_text for event in translation_events] == ["repeat this"]
    assert llm.calls == [(peer_turn_ids[0], "repeat this")]
    assert harness.output_runtime.routing_decisions[-1].reason == "peer_chatbox_denied"


@pytest.mark.asyncio
async def test_peer_overlay_first_emit_latency_summary_and_detailed_trace() -> None:
    basic_runtime_logging, basic_stream = _make_runtime_logging_capture()
    detailed_runtime_logging, detailed_stream = _make_runtime_logging_capture()
    detailed_runtime_logging.set_mode(SessionLoggingMode.DETAILED)

    basic_clock = FakeClock(_now=10.0)
    detailed_clock = FakeClock(_now=20.0)
    basic_harness = compose_translation_test_harness(
        stt=None,
        llm=ClockedTranslateLLMProvider(
            clock=basic_clock,
            responses=[(0.15, "hello")],
        ),
        osc=RecordingOscQueue(),
        overlay_sink=RecordingOverlaySink(),
        peer_translation_enabled=True,
        runtime_logging=basic_runtime_logging,
        clock=basic_clock,
        peer_hangover_s=0.95,
    )
    detailed_harness = compose_translation_test_harness(
        stt=None,
        llm=ClockedTranslateLLMProvider(
            clock=detailed_clock,
            responses=[(0.15, "hello")],
        ),
        osc=RecordingOscQueue(),
        overlay_sink=RecordingOverlaySink(),
        peer_translation_enabled=True,
        runtime_logging=detailed_runtime_logging,
        clock=detailed_clock,
        peer_hangover_s=0.95,
    )

    try:
        basic_utterance_id = uuid4()
        await basic_harness.peer_owner.handle_peer_vad_event(SpeechEnd(basic_utterance_id))
        basic_clock.advance(0.03)
        await basic_harness.dispatch_stt_event(
            STTFinalEvent(
                utterance_id=basic_utterance_id,
                transcript=Transcript(
                    utterance_id=basic_utterance_id,
                    text="안녕",
                    is_final=True,
                    created_at=basic_clock.now(),
                    channel="peer",
                ),
            )
        )
        await asyncio.gather(
            *basic_harness.peer_runtime.translation_tasks.values(), return_exceptions=True
        )

        detailed_utterance_id = uuid4()
        await detailed_harness.peer_owner.handle_peer_vad_event(SpeechEnd(detailed_utterance_id))
        detailed_clock.advance(0.03)
        await detailed_harness.dispatch_stt_event(
            STTFinalEvent(
                utterance_id=detailed_utterance_id,
                transcript=Transcript(
                    utterance_id=detailed_utterance_id,
                    text="안녕",
                    is_final=True,
                    created_at=detailed_clock.now(),
                    channel="peer",
                ),
            )
        )
        await asyncio.gather(
            *detailed_harness.peer_runtime.translation_tasks.values(), return_exceptions=True
        )

        basic_messages = _runtime_log_messages(basic_stream)
        detailed_messages = _runtime_log_messages(detailed_stream)
        basic_latency_message = next(
            message for message in basic_messages if "[Basic][Latency]" in message
        )

        assert "channel=peer" in basic_latency_message
        assert "e2e_ms=1130" in basic_latency_message
        assert "final_output_stage=" not in basic_latency_message
        assert not any("[Detailed][Latency]" in message for message in basic_messages)
        assert not any("[Detailed][LatencyBreakdown]" in message for message in basic_messages)

        detailed_peer_turn_id = detailed_harness.output_projection.overlay_sink.events[
            0
        ].utterance_id
        detailed_trace_messages = [
            message
            for message in detailed_messages
            if "[Detailed][Latency]" in message
            and f"utterance_id={str(detailed_peer_turn_id)[:8]}" in message
        ]
        detailed_trace_stages = [
            message.split("stage=")[1].split()[0] for message in detailed_trace_messages
        ]

        assert detailed_trace_stages == [
            "speech_end",
            "stt_final",
            "llm_request_start",
            "llm_done",
            "peer_overlay_first_emit",
        ]
        assert any(
            "[Detailed][LatencyBreakdown]" in message
            and "channel=peer" in message
            and "e2e_ms=1130" in message
            and "speech_end_to_stt_final_ms=30" in message
            and "stt_final_to_final_output_ms=150" in message
            and "final_output_stage=" not in message
            for message in detailed_messages
        )
        assert not any(
            "[Detailed][Latency]" in message and "stage=llm_first_chunk" in message
            for message in detailed_messages
        )
        assert not any(
            "[Detailed][Latency]" in message and "stage=peer_overlay_first_render" in message
            for message in detailed_messages
        )
    finally:
        basic_runtime_logging.close()
        detailed_runtime_logging.close()
        await basic_harness.stop()
        await detailed_harness.stop()


@pytest.mark.asyncio
async def test_peer_overlay_first_emit_waits_for_llm_done() -> None:
    runtime_logging, stream = _make_runtime_logging_capture()
    clock = FakeClock(_now=100.0)
    llm = ReleasableTranslateLLMProvider(response_text="hello")
    sink = RecordingOverlaySink()
    harness = compose_translation_test_harness(
        stt=None,
        llm=llm,
        osc=RecordingOscQueue(),
        overlay_sink=sink,
        peer_translation_enabled=True,
        runtime_logging=runtime_logging,
        clock=clock,
    )
    parent_vad_id = uuid4()

    try:
        await harness.peer_owner.handle_peer_vad_event(SpeechEnd(parent_vad_id))
        clock.advance(0.03)
        await harness.dispatch_stt_event(
            STTFinalEvent(
                utterance_id=parent_vad_id,
                transcript=Transcript(
                    utterance_id=parent_vad_id,
                    text="안녕",
                    is_final=True,
                    created_at=clock.now(),
                    channel="peer",
                ),
            )
        )
        await llm.started.wait()

        assert sink.events == []
        assert not any("[Basic][Latency]" in message for message in _runtime_log_messages(stream))

        assert llm.release is not None
        llm.release.set_result(None)
        await asyncio.gather(
            *harness.peer_runtime.translation_tasks.values(), return_exceptions=True
        )
        assert [event.type for event in sink.events] == ["translation_final", "utterance_closed"]
        assert any("[Basic][Latency]" in message for message in _runtime_log_messages(stream))
    finally:
        runtime_logging.close()
        await harness.stop()


@pytest.mark.asyncio
async def test_peer_detailed_latency_trace_survives_basic_to_detailed_mode_switch() -> None:
    runtime_logging, log_stream = _make_runtime_logging_capture()
    clock = FakeClock(_now=10.0)
    harness = compose_translation_test_harness(
        stt=None,
        llm=None,
        osc=RecordingOscQueue(),
        runtime_logging=runtime_logging,
        clock=clock,
    )
    utterance_id = uuid4()

    try:
        await harness.peer_owner.handle_peer_vad_event(SpeechEnd(utterance_id))
        runtime_logging.set_mode(SessionLoggingMode.DETAILED)
        clock.advance(0.05)

        await harness.dispatch_stt_event(
            STTFinalEvent(
                utterance_id=utterance_id,
                transcript=Transcript(
                    utterance_id=utterance_id,
                    text="안녕",
                    is_final=True,
                    created_at=clock.now(),
                    channel="peer",
                ),
            )
        )

        messages = _runtime_log_messages(log_stream)
        assert any(
            "[Detailed][Latency]" in message and "stage=speech_end" in message
            for message in messages
        )
        assert any(
            "[Detailed][Latency]" in message and "stage=stt_final" in message
            for message in messages
        )
    finally:
        runtime_logging.close()
        await harness.stop()


@pytest.mark.asyncio
async def test_peer_overlay_success_clears_latency_timeline() -> None:
    utterance_id = uuid4()
    harness = compose_translation_test_harness(
        stt=None,
        llm=SequencedTranslateLLMProvider(responses=["hello"], delay_s=0.0),
        osc=RecordingOscQueue(),
        overlay_sink=RecordingOverlaySink(),
        peer_translation_enabled=True,
        clock=FakeClock(_now=10.0),
    )

    await harness.peer_owner.handle_peer_vad_event(SpeechEnd(utterance_id))
    await harness.dispatch_stt_event(
        STTFinalEvent(
            utterance_id=utterance_id,
            transcript=Transcript(
                utterance_id=utterance_id,
                text="안녕",
                is_final=True,
                created_at=harness.clock.now(),
                channel="peer",
            ),
        )
    )
    await asyncio.gather(*harness.peer_runtime.translation_tasks.values(), return_exceptions=True)

    assert not harness.translation_diagnostics.snapshot().timeline_keys
    assert harness.peer_runtime.utterance_start_times == {}
    assert harness.peer_runtime.speech_ended_ids == set()
    assert harness.peer_owner._peer_turn_parent_ids == {}
    assert harness.peer_owner._peer_parent_turn_ids == {}
    assert harness.peer_owner._peer_completed_turn_ids == set()
    assert harness.peer_owner._peer_parent_speech_end_times == {}


@pytest.mark.asyncio
async def test_peer_overlay_translation_denies_chatbox_and_cleans_bookkeeping() -> None:
    utterance_id = uuid4()
    osc = RecordingOscQueue()
    harness = compose_translation_test_harness(
        stt=None,
        llm=SequencedTranslateLLMProvider(responses=["hello"], delay_s=0.0),
        osc=osc,
        overlay_sink=RecordingOverlaySink(),
        peer_translation_enabled=True,
        clock=FakeClock(_now=10.0),
    )

    await harness.peer_owner.handle_peer_vad_event(SpeechEnd(utterance_id))
    await harness.dispatch_stt_event(
        STTFinalEvent(
            utterance_id=utterance_id,
            transcript=Transcript(
                utterance_id=utterance_id,
                text="안녕",
                is_final=True,
                created_at=harness.clock.now(),
                channel="peer",
            ),
        )
    )
    await harness.translation_turns.wait_for_idle()
    results = await asyncio.gather(
        *harness.peer_runtime.translation_tasks.values(),
        return_exceptions=True,
    )

    assert results == []
    assert osc.messages == []
    decision = harness.output_runtime.routing_decisions[-1]
    assert decision.decision == "denied"
    assert decision.reason == "peer_chatbox_denied"
    assert "안녕" not in repr(decision)
    assert "hello" not in repr(decision)
    assert not harness.translation_diagnostics.snapshot().timeline_keys
    assert harness.peer_runtime.utterance_start_times == {}
    assert harness.peer_runtime.speech_ended_ids == set()
    assert harness.peer_owner._peer_turn_parent_ids == {}
    assert harness.peer_owner._peer_parent_turn_ids == {}
    assert harness.peer_owner._peer_completed_turn_ids == set()
    assert harness.peer_owner._peer_parent_speech_end_times == {}


@pytest.mark.asyncio
async def test_peer_overlay_failure_clears_latency_timeline() -> None:
    utterance_id = uuid4()
    harness = compose_translation_test_harness(
        stt=None,
        llm=ImmediateFailingTranslateLLMProvider(error=RuntimeError("boom")),
        osc=RecordingOscQueue(),
        overlay_sink=RecordingOverlaySink(),
        peer_translation_enabled=True,
        clock=FakeClock(_now=10.0),
    )

    await harness.peer_owner.handle_peer_vad_event(SpeechEnd(utterance_id))
    await harness.dispatch_stt_event(
        STTFinalEvent(
            utterance_id=utterance_id,
            transcript=Transcript(
                utterance_id=utterance_id,
                text="안녕",
                is_final=True,
                created_at=harness.clock.now(),
                channel="peer",
            ),
        )
    )
    await asyncio.gather(*harness.peer_runtime.translation_tasks.values(), return_exceptions=True)

    assert not harness.translation_diagnostics.snapshot().timeline_keys
    assert harness.peer_runtime.utterance_start_times == {}
    assert harness.peer_runtime.speech_ended_ids == set()


@pytest.mark.asyncio
async def test_peer_no_chatbox_terminal_path_clears_latency_bookkeeping() -> None:
    utterance_id = uuid4()
    harness = compose_translation_test_harness(
        stt=None,
        llm=None,
        osc=RecordingOscQueue(),
        clock=FakeClock(_now=10.0),
    )

    await harness.peer_owner.handle_peer_vad_event(SpeechEnd(utterance_id))
    await harness.dispatch_stt_event(
        STTFinalEvent(
            utterance_id=utterance_id,
            transcript=Transcript(
                utterance_id=utterance_id,
                text="안녕",
                is_final=True,
                created_at=harness.clock.now(),
                channel="peer",
            ),
        )
    )

    assert not harness.translation_diagnostics.snapshot().timeline_keys
    assert harness.peer_runtime.utterance_start_times == {}
    assert harness.peer_runtime.speech_ended_ids == set()


@pytest.mark.asyncio
async def test_late_peer_speech_end_after_completed_turn_does_not_resurrect_bookkeeping() -> None:
    parent_vad_id = uuid4()
    harness = compose_translation_test_harness(
        stt=None,
        llm=None,
        osc=RecordingOscQueue(),
        clock=FakeClock(_now=10.0),
    )

    await harness.dispatch_stt_event(
        STTFinalEvent(
            utterance_id=parent_vad_id,
            transcript=Transcript(
                utterance_id=parent_vad_id,
                text="안녕",
                is_final=True,
                created_at=harness.clock.now(),
                channel="peer",
            ),
        )
    )
    peer_turn_id = next(iter(harness.peer_runtime.utterances))

    assert peer_turn_id != parent_vad_id
    assert parent_vad_id not in harness.peer_owner._peer_parent_turn_ids
    assert peer_turn_id not in harness.peer_owner._peer_turn_parent_ids
    assert harness.peer_runtime.utterance_start_times == {}
    assert harness.peer_runtime.speech_ended_ids == set()

    harness.clock.advance(0.1)
    await harness.peer_owner.handle_peer_vad_event(SpeechEnd(parent_vad_id))

    assert parent_vad_id not in harness.peer_owner._peer_parent_turn_ids
    assert peer_turn_id not in harness.peer_owner._peer_turn_parent_ids
    assert harness.peer_runtime.utterance_start_times == {}
    assert harness.peer_runtime.speech_ended_ids == set()
    assert not harness.translation_diagnostics.snapshot().timeline_keys


@pytest.mark.asyncio
async def test_closed_parent_rejects_late_duplicate_final_without_child_output() -> None:
    parent_vad_id = uuid4()
    llm = ReleasableTranslateLLMProvider(response_text="hello")
    harness = compose_translation_test_harness(
        stt=None,
        llm=None,
        osc=RecordingOscQueue(),
        peer_translation_enabled=True,
        clock=FakeClock(_now=10.0),
    )

    await harness.peer_owner.handle_peer_vad_event(SpeechEnd(parent_vad_id))
    harness.clock.advance(0.01)
    await harness.dispatch_stt_event(
        STTFinalEvent(
            utterance_id=parent_vad_id,
            transcript=Transcript(
                utterance_id=parent_vad_id,
                text="first",
                is_final=True,
                created_at=harness.clock.now(),
                channel="peer",
            ),
        )
    )
    first_peer_turn_id = next(iter(harness.peer_runtime.utterances))

    harness.llm_runtime.attach_provider_reference(LlmTranslationBackend(llm))
    harness.clock.advance(0.01)
    await harness.dispatch_stt_event(
        STTFinalEvent(
            utterance_id=parent_vad_id,
            transcript=Transcript(
                utterance_id=parent_vad_id,
                text="second",
                is_final=True,
                created_at=harness.clock.now(),
                channel="peer",
            ),
        )
    )
    assert first_peer_turn_id != parent_vad_id
    assert set(harness.peer_runtime.utterances) == {first_peer_turn_id}
    assert harness.peer_runtime.translation_tasks == {}
    assert llm.calls == []
    assert harness.peer_owner._peer_turn_parent_ids == {}
    assert harness.peer_owner._peer_parent_turn_ids == {}
    decision = harness.output_runtime.routing_decisions[-1]
    assert decision.decision == "denied"
    assert decision.reason == "peer_chatbox_denied"


@pytest.mark.asyncio
async def test_inflight_parent_rejects_duplicate_final_without_second_child_or_output() -> None:
    parent_utterance_id = uuid4()
    sink = RecordingOverlaySink()
    llm = ReleasableTranslateLLMProvider(response_text="translated")
    harness = compose_translation_test_harness(
        stt=None,
        llm=llm,
        osc=RecordingOscQueue(),
        overlay_sink=sink,
        peer_translation_enabled=True,
    )

    await harness.peer_owner.handle_peer_vad_event(SpeechEnd(parent_utterance_id))
    await harness.dispatch_stt_event(
        STTFinalEvent(
            utterance_id=parent_utterance_id,
            transcript=Transcript(
                utterance_id=parent_utterance_id,
                text="first peer final",
                is_final=True,
                channel="peer",
            ),
        )
    )
    await asyncio.wait_for(llm.started.wait(), timeout=0.5)
    child_ids = tuple(harness.peer_runtime.utterances)

    await harness.dispatch_stt_event(
        STTFinalEvent(
            utterance_id=parent_utterance_id,
            transcript=Transcript(
                utterance_id=parent_utterance_id,
                text="duplicate peer final",
                is_final=True,
                channel="peer",
            ),
        )
    )

    assert tuple(harness.peer_runtime.utterances) == child_ids
    assert llm.calls == ["first peer final"]
    decision = harness.output_runtime.routing_decisions[-1]
    assert decision.decision == "denied"
    assert decision.reason == "peer_chatbox_denied"
    assert "duplicate peer final" not in repr(decision)

    assert llm.release is not None
    llm.release.set_result(None)
    await harness.translation_turns.wait_for_idle()

    translations = [event for event in sink.events if event.type == "translation_final"]
    assert [event.source_text for event in translations] == ["first peer final"]
    assert "duplicate peer final" not in repr(sink.events)


@pytest.mark.asyncio
async def test_peer_no_overlay_translation_path_keeps_latency_bookkeeping_until_translation_finishes() -> (
    None
):
    utterance_id = uuid4()
    llm = ReleasableTranslateLLMProvider(response_text="hello")
    harness = compose_translation_test_harness(
        stt=None,
        llm=llm,
        osc=RecordingOscQueue(),
        peer_translation_enabled=True,
        clock=FakeClock(_now=10.0),
    )

    await harness.peer_owner.handle_peer_vad_event(SpeechEnd(utterance_id))
    await harness.dispatch_stt_event(
        STTFinalEvent(
            utterance_id=utterance_id,
            transcript=Transcript(
                utterance_id=utterance_id,
                text="안녕",
                is_final=True,
                created_at=harness.clock.now(),
                channel="peer",
            ),
        )
    )
    await llm.started.wait()
    peer_turn_id = next(iter(harness.peer_runtime.translation_tasks))

    assert peer_turn_id != utterance_id
    assert peer_turn_id in harness.peer_runtime.utterance_start_times
    assert peer_turn_id in harness.peer_runtime.speech_ended_ids
    assert (
        "peer",
        peer_turn_id,
    ) in harness.translation_diagnostics.snapshot().timeline_keys
    assert llm.calls == ["안녕"]

    assert llm.release is not None
    llm.release.set_result(None)
    await asyncio.gather(*harness.peer_runtime.translation_tasks.values(), return_exceptions=True)

    assert not harness.translation_diagnostics.snapshot().timeline_keys
    assert harness.peer_runtime.utterance_start_times == {}
    assert harness.peer_runtime.speech_ended_ids == set()


@pytest.mark.asyncio
async def test_peer_without_overlay_sink_succeeds_via_translate() -> None:
    llm = SequencedTranslateLLMProvider(responses=["hello"], delay_s=0.0)
    harness = compose_translation_test_harness(
        stt=None,
        llm=llm,
        osc=RecordingOscQueue(),
        peer_translation_enabled=True,
    )

    utterance_id = await harness.translate_peer_text_for_test("안녕")
    events = [await harness.ui_events.get(), await harness.ui_events.get()]

    assert llm.calls == ["안녕"]
    assert [event.type for event in events] == [
        UIEventType.TRANSCRIPT_FINAL,
        UIEventType.TRANSLATION_DONE,
    ]
    assert events[-1].utterance_id == utterance_id
    assert events[-1].payload.text == "hello"
    assert harness.ui_events.empty()


@pytest.mark.asyncio
async def test_peer_test_helper_returns_new_logical_turn_for_identical_text_without_overlay_sink() -> (
    None
):
    harness = compose_translation_test_harness(
        stt=None,
        llm=None,
        osc=RecordingOscQueue(),
    )

    first_peer_turn_id = await harness.handle_peer_transcript_final_for_test(text="repeat")
    second_peer_turn_id = await harness.handle_peer_transcript_final_for_test(text="repeat")

    assert second_peer_turn_id != first_peer_turn_id
    assert second_peer_turn_id in harness.peer_runtime.utterances
    assert harness.peer_runtime.utterances[second_peer_turn_id].final.text == "repeat"


def test_peer_overlay_first_render_latency_contract_is_explicit() -> None:
    first_emit = LATENCY_TRACE_POINT_CONTRACTS["peer_overlay_first_emit"]
    first_render = LATENCY_TRACE_POINT_CONTRACTS["peer_overlay_first_render"]

    assert "paired source+translation when translation succeeds" in first_emit.timing_semantics
    assert "source-only fallback" in first_emit.timing_semantics
    assert "overlay_sink.emit" in first_emit.acceptance_expectation
    assert "wait for the paired source+translation overlay output" in (
        first_emit.acceptance_expectation
    )
    assert "first local visible peer source or translation overlay output" in (
        first_render.timing_semantics
    )
    assert "after peer_overlay_first_emit" in first_render.acceptance_expectation
    assert "once per peer logical turn" in first_render.acceptance_expectation
    assert "do not wait for lifecycle completion" in first_render.acceptance_expectation


@pytest.mark.asyncio
async def test_chatbox_stays_self_final_only_while_overlay_sink_receives_peer_finals() -> None:
    osc = RecordingOscQueue()
    sink = RecordingOverlaySink()
    harness = compose_translation_test_harness(stt=None, llm=None, osc=osc, overlay_sink=sink)

    await harness.self_owner.submit_text("self text", source="You")
    await harness.handle_peer_transcript_final_for_test(text="peer text")

    assert len(osc.messages) == 1
    assert osc.messages[0].text == "self text"
    assert sink.events[-1].channel == "peer"


@pytest.mark.asyncio
async def test_peer_no_translation_source_only_overlay_close_remains_final() -> None:
    sink = RecordingOverlaySink()
    harness = compose_translation_test_harness(
        stt=None, llm=None, osc=RecordingOscQueue(), overlay_sink=sink
    )

    utterance_id = await harness.handle_peer_transcript_final_for_test(text="안녕")

    assert [event.type for event in sink.events] == [
        "peer_transcript_final",
        "utterance_closed",
    ]
    assert sink.events[-1].utterance_id == utterance_id
    assert sink.events[-1].is_final is True


@pytest.mark.asyncio
async def test_peer_translation_disabled_finalizes_source_only_turn() -> None:
    sink = RecordingOverlaySink()
    harness = compose_translation_test_harness(
        stt=None,
        llm=None,
        osc=RecordingOscQueue(),
        overlay_sink=sink,
        peer_translation_enabled=False,
    )
    parent_vad_id = uuid4()

    await harness.dispatch_stt_event(
        STTFinalEvent(
            utterance_id=parent_vad_id,
            transcript=Transcript(
                utterance_id=parent_vad_id,
                text="source only",
                is_final=True,
                created_at=11.0,
                channel="peer",
            ),
        )
    )
    await harness.translation_turns.wait_for_idle()

    assert [event.type for event in sink.events] == [
        "peer_transcript_final",
        "utterance_closed",
    ]
    assert sink.events[0].utterance_id == sink.events[1].utterance_id
    assert sink.events[0].text == "source only"


@pytest.mark.asyncio
async def test_peer_translation_failure_finalizes_source_only_turn_and_emits_error() -> None:
    sink = RecordingOverlaySink()
    harness = compose_translation_test_harness(
        stt=None,
        llm=ImmediateFailingTranslateLLMProvider(RuntimeError("llm boom")),
        osc=RecordingOscQueue(),
        overlay_sink=sink,
        peer_translation_enabled=True,
    )
    parent_vad_id = uuid4()

    await harness.dispatch_stt_event(
        STTFinalEvent(
            utterance_id=parent_vad_id,
            transcript=Transcript(
                utterance_id=parent_vad_id,
                text="source after failure",
                is_final=True,
                created_at=11.0,
                channel="peer",
            ),
        )
    )
    await asyncio.gather(*harness.peer_runtime.translation_tasks.values(), return_exceptions=True)

    assert [event.type for event in sink.events] == [
        "peer_transcript_final",
        "utterance_closed",
    ]
    ui_events = [harness.ui_events.get_nowait() for _ in range(harness.ui_events.qsize())]
    assert [event.type for event in ui_events] == [
        UIEventType.TRANSCRIPT_FINAL,
        UIEventType.ERROR,
    ]
    assert ui_events[1].channel == "peer"


@pytest.mark.asyncio
async def test_translation_provider_failure_uses_message_ref_and_safe_runtime_log() -> None:
    raw_detail = "provider raw detail token=translation-secret-789"
    runtime_logging, log_stream = _make_runtime_logging_capture()
    harness = compose_translation_test_harness(
        stt=None,
        llm=ImmediateFailingTranslateLLMProvider(RuntimeError(raw_detail)),
        osc=RecordingOscQueue(),
        runtime_logging=runtime_logging,
    )

    try:
        await harness.self_owner.submit_text("source text", source="You")
        await asyncio.gather(
            *harness.self_runtime.translation_tasks.values(), return_exceptions=True
        )

        ui_events = [harness.ui_events.get_nowait() for _ in range(harness.ui_events.qsize())]
        error_event = next(event for event in ui_events if event.type == UIEventType.ERROR)
        error_report_type = getattr(messages, "UserErrorReport", None)
        assert error_report_type is not None, "UserErrorReport DTO is missing"
        assert isinstance(error_event.payload, error_report_type)
        assert error_event.payload.message.key == "provider.failure"
        assert error_event.payload.diagnostics.category == messages.DIAGNOSTIC_CATEGORY_UNKNOWN
        assert raw_detail not in repr(error_event.payload)

        runtime_log = "\n".join(_runtime_log_messages(log_stream))
        assert raw_detail not in runtime_log
        assert "category=unknown" in runtime_log
        assert "code=provider.unknown" in runtime_log
    finally:
        runtime_logging.close()


@pytest.mark.asyncio
async def test_legacy_peer_handle_transcript_gates_overlay_until_translation() -> None:
    sink = RecordingOverlaySink()
    llm = ReleasableTranslateLLMProvider(response_text="hello")
    harness = compose_translation_test_harness(
        stt=None,
        llm=llm,
        osc=RecordingOscQueue(),
        overlay_sink=sink,
        peer_translation_enabled=True,
    )
    utterance_id = uuid4()
    transcript = Transcript(
        utterance_id=utterance_id,
        text="안녕",
        is_final=True,
        created_at=11.0,
        channel="peer",
    )

    try:
        await harness.dispatch_transcript(transcript, is_final=True, source="Peer")
        await llm.started.wait()

        assert sink.events == []

        assert llm.release is not None
        llm.release.set_result(None)
        await asyncio.gather(
            *harness.peer_runtime.translation_tasks.values(), return_exceptions=True
        )

        assert [event.type for event in sink.events] == [
            "translation_final",
            "utterance_closed",
        ]
        assert sink.events[0].source_text == "안녕"
    finally:
        await harness.stop()


@pytest.mark.asyncio
async def test_peer_translation_overlay_waits_for_translation_and_includes_source_text() -> None:
    sink = RecordingOverlaySink()
    llm = ReleasableTranslateLLMProvider(response_text="hello")
    harness = compose_translation_test_harness(
        stt=None,
        llm=llm,
        osc=RecordingOscQueue(),
        overlay_sink=sink,
        peer_translation_enabled=True,
    )
    parent_vad_id = uuid4()

    try:
        await harness.dispatch_stt_event(
            STTFinalEvent(
                utterance_id=parent_vad_id,
                transcript=Transcript(
                    utterance_id=parent_vad_id,
                    text="안녕",
                    is_final=True,
                    created_at=11.0,
                    channel="peer",
                ),
            )
        )
        await llm.started.wait()

        assert sink.events == []

        assert llm.release is not None
        llm.release.set_result(None)
        await asyncio.gather(
            *harness.peer_runtime.translation_tasks.values(), return_exceptions=True
        )

        assert [event.type for event in sink.events] == [
            "translation_final",
            "utterance_closed",
        ]
        assert sink.events[0].channel == "peer"
        assert sink.events[0].text == "hello"
        assert sink.events[0].source_text == "안녕"
    finally:
        await harness.stop()


@pytest.mark.asyncio
async def test_peer_translation_emits_final_only_overlay_events() -> None:
    sink = RecordingOverlaySink()
    harness = compose_translation_test_harness(
        stt=None,
        llm=SequencedTranslateLLMProvider(responses=["hello"], delay_s=0.0),
        osc=RecordingOscQueue(),
        overlay_sink=sink,
        peer_translation_enabled=True,
    )

    await harness.translate_peer_text_for_test("안녕")

    assert [event.type for event in sink.events] == [
        "translation_final",
        "utterance_closed",
    ]
    assert not any(event.type == "translation_stream_update" for event in sink.events)
    assert sink.events[0].channel == "peer"
    assert sink.events[0].text == "hello"
    assert sink.events[0].source_text == "안녕"
    assert sink.events[1].channel == "peer"
    assert sink.events[1].is_final is True


@pytest.mark.asyncio
async def test_peer_overlay_events_arrive_before_translation_done_and_preserve_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class OrderingOverlaySink:
        def __init__(self, order: list[str]) -> None:
            self.events: list[object] = []
            self._order = order

        async def emit(self, event: object) -> None:
            self._order.append(f"overlay:{event.type}")
            self.events.append(event)

    call_order: list[str] = []
    sink = OrderingOverlaySink(call_order)
    harness = compose_translation_test_harness(
        stt=None,
        llm=SequencedTranslateLLMProvider(responses=["hello"], delay_s=0.0),
        osc=RecordingOscQueue(),
        overlay_sink=sink,
        peer_translation_enabled=True,
    )
    original_put = harness.ui_events.put

    async def recording_put(event) -> None:
        call_order.append(f"ui:{event.type.value}")
        await original_put(event)

    monkeypatch.setattr(harness.ui_events, "put", recording_put)

    await harness.translate_peer_text_for_test("안녕")

    events = [harness.ui_events.get_nowait() for _ in range(harness.ui_events.qsize())]

    assert [event.type for event in events] == [
        UIEventType.TRANSCRIPT_FINAL,
        UIEventType.TRANSLATION_DONE,
    ]
    assert events[1].payload.text == "hello"
    translation_event_order = [event.type for event in sink.events]
    assert translation_event_order == [
        "translation_final",
        "utterance_closed",
    ]
    assert sink.events[0].source_text == "안녕"
    assert call_order == [
        "ui:TRANSCRIPT_FINAL",
        "overlay:translation_final",
        "overlay:utterance_closed",
        "ui:TRANSLATION_DONE",
    ]
    decision = harness.output_runtime.routing_decisions[-1]
    assert decision.decision == "denied"
    assert decision.reason == "peer_chatbox_denied"
    assert "안녕" not in repr(decision)
    assert "hello" not in repr(decision)
    assert harness.ui_events.empty()


@pytest.mark.asyncio
async def test_self_osc_sent_channel_uses_utterance_runtime_when_peer_chatbox_active() -> None:
    harness = compose_translation_test_harness(
        stt=None,
        llm=None,
        osc=RecordingOscQueue(),
        overlay_sink=RecordingOverlaySink(),
    )

    utterance_id = await harness.self_owner.submit_text("self text", source="You")
    events = [harness.ui_events.get_nowait() for _ in range(harness.ui_events.qsize())]

    assert [event.type for event in events] == [
        UIEventType.TRANSCRIPT_FINAL,
        UIEventType.OSC_SENT,
    ]
    assert events[1].utterance_id == utterance_id
    assert events[1].payload.text == "self text"
    assert events[1].channel == "self"
    assert harness.ui_events.empty()


@pytest.mark.asyncio
async def test_self_stt_final_uses_self_chatbox_when_legacy_peer_chatbox_active() -> None:
    osc = RecordingOscQueue()
    harness = compose_translation_test_harness(
        stt=None,
        llm=None,
        osc=osc,
    )
    utterance_id = uuid4()

    await harness.dispatch_stt_event(
        STTFinalEvent(
            utterance_id=utterance_id,
            transcript=Transcript(
                utterance_id=utterance_id,
                text="self final",
                is_final=True,
                created_at=10.0,
                channel="self",
            ),
        )
    )
    events = [harness.ui_events.get_nowait() for _ in range(harness.ui_events.qsize())]

    assert [event.type for event in events] == [
        UIEventType.TRANSCRIPT_FINAL,
        UIEventType.OSC_SENT,
    ]
    assert events[1].utterance_id == utterance_id
    assert events[1].payload.text == "self final"
    assert events[1].channel == "self"
    assert [message.text for message in osc.messages] == ["self final"]
    assert harness.ui_events.empty()


@pytest.mark.asyncio
async def test_peer_overlay_emit_failures_still_emit_translation_done_and_deny_chatbox() -> None:
    class RecordingFailingOverlaySink:
        def __init__(self, order: list[str]) -> None:
            self.attempted_types: list[str] = []
            self._order = order

        async def emit(self, event: object) -> None:
            self._order.append(f"overlay:{event.type}")
            self.attempted_types.append(event.type)
            raise RuntimeError(f"overlay boom: {event.type}")

    call_order: list[str] = []
    sink = RecordingFailingOverlaySink(call_order)
    osc = RecordingOscQueue()
    harness = compose_translation_test_harness(
        stt=None,
        llm=SequencedTranslateLLMProvider(responses=["hello"], delay_s=0.0),
        osc=osc,
        overlay_sink=sink,
        peer_translation_enabled=True,
    )
    original_put = harness.ui_events.put

    async def recording_put(event) -> None:
        call_order.append(f"ui:{event.type.value}")
        await original_put(event)

    harness.ui_events.put = recording_put  # type: ignore[method-assign]

    utterance_id = await harness.translate_peer_text_for_test("안녕")
    events = [await harness.ui_events.get() for _ in range(2)]

    assert call_order == [
        "ui:TRANSCRIPT_FINAL",
        "overlay:translation_final",
        "overlay:utterance_closed",
        "ui:TRANSLATION_DONE",
    ]
    assert sink.attempted_types == [
        "translation_final",
        "utterance_closed",
    ]
    assert [event.type for event in events] == [
        UIEventType.TRANSCRIPT_FINAL,
        UIEventType.TRANSLATION_DONE,
    ]
    assert events[1].utterance_id == utterance_id
    assert events[1].payload.text == "hello"
    assert osc.messages == []
    decision = harness.output_runtime.routing_decisions[-1]
    assert decision.decision == "denied"
    assert decision.reason == "peer_chatbox_denied"
    assert "안녕" not in repr(decision)
    assert "hello" not in repr(decision)
    assert harness.translation_diagnostics.snapshot().last_error_source == "overlay_sink"
    assert harness.ui_events.empty()


@pytest.mark.asyncio
async def test_overlay_sink_failures_do_not_break_chatbox_or_translation_completion() -> None:
    sink = FailingOverlaySink()
    osc = RecordingOscQueue()
    harness = compose_translation_test_harness(
        stt=None,
        llm=StubTranslateLLMProvider(text="hello"),
        osc=osc,
        overlay_sink=sink,
    )

    await harness.self_owner.submit_text("self text", source="You")
    await asyncio.gather(*harness.self_runtime.translation_tasks.values(), return_exceptions=True)

    assert osc.messages[0].text == "self text (hello)"
    assert harness.translation_diagnostics.snapshot().last_error_source == "overlay_sink"


@pytest.mark.asyncio
async def test_translation_emits_self_translation_to_overlay_after_translation_completion() -> None:
    sink = RecordingOverlaySink()
    osc = RecordingOscQueue()
    harness = compose_translation_test_harness(
        stt=None,
        llm=StubTranslateLLMProvider(text="hello"),
        osc=osc,
        overlay_sink=sink,
    )

    await harness.self_owner.submit_text("self text", source="You")
    await asyncio.gather(*harness.self_runtime.translation_tasks.values(), return_exceptions=True)

    translation_events = [
        event
        for event in sink.events
        if event.type == "translation_final" and event.channel == "self"
    ]

    assert osc.messages[0].text == "self text (hello)"
    assert [event.type for event in sink.events[:2]] == [
        "self_transcript_final",
        "translation_final",
    ]
    assert translation_events[-1].text == "hello"
    assert translation_events[-1].text != osc.messages[0].text


@pytest.mark.asyncio
async def test_translation_newer_self_row_replaces_older_translated_self_row_without_protection_boost() -> (
    None
):
    bridge = RecordingPresentationBridge()
    clock = FakeClock(_now=10.0)
    presenter = OverlayPresenter(
        bridge=bridge,
        calibration=OverlayCalibration(),
        clock=clock,
        visible_window_target_blocks=1,
    )
    harness = compose_translation_test_harness(
        stt=None,
        llm=SequencedTranslateLLMProvider(
            responses=["translated first", "translated second"],
            delay_s=0.05,
        ),
        osc=RecordingOscQueue(),
        overlay_sink=presenter,
        clock=clock,
    )

    first_id = await harness.self_owner.submit_text("first", source="You")
    await asyncio.gather(*harness.self_runtime.translation_tasks.values(), return_exceptions=True)

    assert [block.id for block in presenter.snapshot().blocks] == [f"self:{first_id}"]
    assert presenter.snapshot().blocks[0].secondary_text == "translated first"

    second_id = await harness.self_owner.submit_text("second", source="You")

    assert [block.id for block in presenter.snapshot().blocks] == [f"self:{second_id}"]
    assert presenter.snapshot().blocks[0].secondary_text == ""

    await asyncio.gather(*harness.self_runtime.translation_tasks.values(), return_exceptions=True)

    assert [block.id for block in presenter.snapshot().blocks] == [f"self:{second_id}"]
    assert presenter.snapshot().blocks[0].secondary_text == "translated second"


@pytest.mark.asyncio
async def test_translation_closes_self_overlay_line_after_translation_completion() -> None:
    sink = RecordingOverlaySink()
    harness = compose_translation_test_harness(
        stt=None,
        llm=StubTranslateLLMProvider(text="hello"),
        osc=RecordingOscQueue(),
        overlay_sink=sink,
    )

    await harness.self_owner.submit_text("self text", source="You")
    await asyncio.gather(*harness.self_runtime.translation_tasks.values(), return_exceptions=True)

    assert [event.type for event in sink.events] == [
        "self_transcript_final",
        "translation_final",
        "utterance_closed",
    ]
    assert sink.events[-1].channel == "self"
    assert sink.events[-1].is_final is True


@pytest.mark.asyncio
async def test_self_translation_failure_closes_overlay_line_as_incomplete() -> None:
    sink = RecordingOverlaySink()
    harness = compose_translation_test_harness(
        stt=None,
        llm=ImmediateFailingTranslateLLMProvider(error=RuntimeError("boom")),
        osc=RecordingOscQueue(),
        overlay_sink=sink,
    )

    utterance_id = await harness.self_owner.submit_text("self text", source="You")
    await asyncio.gather(*harness.self_runtime.translation_tasks.values(), return_exceptions=True)

    assert [event.type for event in sink.events] == [
        "self_transcript_final",
        "utterance_closed",
    ]
    assert sink.events[-1].channel == "self"
    assert sink.events[-1].utterance_id == utterance_id
    assert sink.events[-1].is_final is False


@pytest.mark.asyncio
async def test_peer_translation_failure_closes_line_as_incomplete() -> None:
    sink = RecordingOverlaySink()
    harness = compose_translation_test_harness(
        stt=None,
        llm=ImmediateFailingTranslateLLMProvider(error=RuntimeError("boom")),
        osc=RecordingOscQueue(),
        overlay_sink=sink,
        peer_translation_enabled=True,
    )

    utterance_id = await harness.translate_peer_text_for_test("안녕")

    assert [event.type for event in sink.events] == [
        "peer_transcript_final",
        "utterance_closed",
    ]
    assert sink.events[-1].channel == "peer"
    assert sink.events[-1].utterance_id == utterance_id
    assert sink.events[-1].is_final is False


@pytest.mark.asyncio
async def test_peer_translation_failure_hard_denies_active_peer_chatbox_fallback() -> None:
    sink = RecordingOverlaySink()
    osc = RecordingOscQueue()
    harness = compose_translation_test_harness(
        stt=None,
        llm=ImmediateFailingTranslateLLMProvider(error=RuntimeError("boom")),
        osc=osc,
        overlay_sink=sink,
        peer_translation_enabled=True,
        fallback_transcript_only=True,
    )

    utterance_id = await harness.translate_peer_text_for_test("안녕")
    events = [await harness.ui_events.get() for _ in range(2)]

    assert [event.type for event in sink.events] == [
        "peer_transcript_final",
        "utterance_closed",
    ]
    assert sink.events[-1].channel == "peer"
    assert sink.events[-1].utterance_id == utterance_id
    assert sink.events[-1].is_final is False
    assert osc.messages == []
    assert [event.type for event in events] == [
        UIEventType.TRANSCRIPT_FINAL,
        UIEventType.ERROR,
    ]
    decision = harness.output_runtime.routing_decisions[-1]
    assert decision.decision == "denied"
    assert decision.reason == "peer_chatbox_denied"
    assert "안녕" not in repr(decision)
    assert harness.ui_events.empty()


@pytest.mark.asyncio
async def test_peer_translation_failure_hard_denies_active_peer_chatbox_without_fallback() -> None:
    sink = RecordingOverlaySink()
    osc = RecordingOscQueue()
    harness = compose_translation_test_harness(
        stt=None,
        llm=ImmediateFailingTranslateLLMProvider(error=RuntimeError("provider raw detail")),
        osc=osc,
        overlay_sink=sink,
        peer_translation_enabled=True,
        fallback_transcript_only=False,
    )

    utterance_id = await harness.translate_peer_text_for_test("peer failure text")
    events = [harness.ui_events.get_nowait() for _ in range(harness.ui_events.qsize())]

    assert osc.messages == []
    assert not any(event.type == UIEventType.OSC_SENT for event in events)
    decision = harness.output_runtime.routing_decisions[-1]
    assert decision.decision == "denied"
    assert decision.route == "self_chatbox"
    assert decision.publication_id == str(utterance_id)
    assert decision.publication_kind == "peer_subtitle"
    assert decision.reason == "peer_chatbox_denied"
    assert "peer failure text" not in repr(decision)
    assert "provider raw detail" not in repr(decision)


@pytest.mark.asyncio
async def test_legacy_peer_active_chatbox_route_is_hard_denied_without_user_text() -> None:
    osc = RecordingOscQueue()
    harness = compose_translation_test_harness(
        stt=None,
        llm=None,
        osc=osc,
        peer_translation_enabled=True,
    )

    utterance_id = await harness.handle_peer_transcript_final_for_test("secret peer line")
    events = [harness.ui_events.get_nowait() for _ in range(harness.ui_events.qsize())]

    assert osc.messages == []
    assert not any(event.type == UIEventType.OSC_SENT for event in events)
    decision = harness.output_runtime.routing_decisions[-1]
    assert decision.decision == "denied"
    assert decision.route == "self_chatbox"
    assert decision.publication_id == str(utterance_id)
    assert decision.publication_kind == "peer_subtitle"
    assert decision.reason == "peer_chatbox_denied"
    assert "secret peer line" not in repr(decision)


@pytest.mark.asyncio
async def test_peer_translation_cancellation_closes_line_as_incomplete() -> None:
    sink = RecordingOverlaySink()
    llm = BlockingTranslateLLMProvider()
    harness = compose_translation_test_harness(
        stt=None,
        llm=llm,
        osc=RecordingOscQueue(),
        overlay_sink=sink,
        peer_translation_enabled=True,
    )

    utterance_id = await harness.handle_peer_transcript_final_for_test(text="안녕")
    await asyncio.wait_for(llm.started.wait(), timeout=0.5)
    assert (
        "peer",
        utterance_id,
    ) in harness.translation_diagnostics.snapshot().timeline_keys
    await harness.peer_runtime.reset_runtime_state()
    await harness.translation_turns.wait_for_idle()

    assert [event.type for event in sink.events] == [
        "peer_transcript_final",
        "utterance_closed",
    ]
    assert sink.events[-1].channel == "peer"
    assert sink.events[-1].utterance_id == utterance_id
    assert sink.events[-1].is_final is False
    assert not harness.translation_diagnostics.snapshot().timeline_keys


@pytest.mark.asyncio
async def test_peer_translation_cancellation_hard_denies_active_peer_chatbox_without_user_text() -> (
    None
):
    sink = RecordingOverlaySink()
    osc = RecordingOscQueue()
    llm = BlockingTranslateLLMProvider()
    harness = compose_translation_test_harness(
        stt=None,
        llm=llm,
        osc=osc,
        overlay_sink=sink,
        peer_translation_enabled=True,
    )

    utterance_id = await harness.handle_peer_transcript_final_for_test(text="secret cancel line")
    await asyncio.wait_for(llm.started.wait(), timeout=0.5)
    await harness.peer_runtime.reset_runtime_state()
    await harness.translation_turns.wait_for_idle()
    events = [harness.ui_events.get_nowait() for _ in range(harness.ui_events.qsize())]

    assert osc.messages == []
    assert not any(event.type == UIEventType.OSC_SENT for event in events)
    decision = harness.output_runtime.routing_decisions[-1]
    assert decision.decision == "denied"
    assert decision.route == "self_chatbox"
    assert decision.publication_id == str(utterance_id)
    assert decision.publication_kind == "peer_subtitle"
    assert decision.reason == "peer_chatbox_denied"
    assert "secret cancel line" not in repr(decision)


@pytest.mark.asyncio
async def test_peer_final_runs_owner_shutdown_cancels_and_awaits_child() -> None:
    sink = RecordingOverlaySink()
    llm = BlockingTranslateLLMProvider()
    harness = compose_translation_test_harness(
        stt=None,
        llm=llm,
        osc=RecordingOscQueue(),
        overlay_sink=sink,
        peer_translation_enabled=True,
    )

    utterance_id = await harness.handle_peer_transcript_final_for_test(text="shutdown peer")
    await asyncio.wait_for(llm.started.wait(), timeout=0.5)
    await harness.stop()

    assert harness.translation_turns.has_resources is False
    assert [event.type for event in sink.events] == [
        "peer_transcript_final",
        "utterance_closed",
    ]
    assert sink.events[-1].utterance_id == utterance_id
    assert sink.events[-1].is_final is False


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["clear", "stop"])
async def test_peer_final_runs_cancellation_suppression_cannot_publish_success_or_leak_tasks(
    operation: str,
) -> None:
    sink = RecordingOverlaySink()
    llm = CancelSuppressingTranslateLLMProvider()
    harness = compose_translation_test_harness(
        stt=None,
        llm=llm,
        osc=RecordingOscQueue(),
        overlay_sink=sink,
        peer_translation_enabled=True,
    )

    utterance_id = await harness.handle_peer_transcript_final_for_test(text="cancel suppression")
    await asyncio.wait_for(llm.started.wait(), timeout=0.5)
    if operation == "clear":
        await asyncio.wait_for(harness.clear_channel_language_state(channel="peer"), timeout=0.5)
    else:
        await asyncio.wait_for(harness.stop(), timeout=0.5)

    assert not any(event.type == "translation_final" for event in sink.events)
    assert [event.type for event in sink.events] == [
        "peer_transcript_final",
        "utterance_closed",
    ]
    assert sink.events[-1].utterance_id == utterance_id
    assert sink.events[-1].is_final is False
    assert harness.translation_turns.has_resources is False
    assert harness.peer_runtime.translation_tasks == {}
    assert harness.peer_owner._peer_turn_parent_ids == {}
    assert harness.peer_owner._peer_parent_turn_ids == {}


@pytest.mark.asyncio
async def test_self_translation_cancellation_closes_overlay_line_as_incomplete() -> None:
    sink = RecordingOverlaySink()
    llm = BlockingTranslateLLMProvider()
    harness = compose_translation_test_harness(
        stt=None,
        llm=llm,
        osc=RecordingOscQueue(),
        overlay_sink=sink,
    )

    utterance_id = await harness.self_owner.submit_text("self text", source="You")
    await llm.started.wait()
    await harness.self_runtime.reset_runtime_state()

    assert [event.type for event in sink.events] == [
        "self_transcript_final",
        "utterance_closed",
    ]
    assert sink.events[-1].channel == "self"
    assert sink.events[-1].utterance_id == utterance_id
    assert sink.events[-1].is_final is False


@pytest.mark.asyncio
async def test_low_latency_self_partial_no_longer_emits_overlay_event() -> None:
    sink = RecordingOverlaySink()
    harness = compose_translation_test_harness(
        stt=None,
        llm=None,
        osc=RecordingOscQueue(),
        overlay_sink=sink,
        clock=FakeClock(_now=10.0),
        low_latency_mode=True,
    )
    utterance_id = uuid4()
    partial = Transcript(
        utterance_id=utterance_id, text="hello live", is_final=False, created_at=11.0
    )

    await harness.dispatch_stt_event(STTPartialEvent(utterance_id=utterance_id, transcript=partial))
    await asyncio.sleep(0.02)

    assert sink.events == []
    assert harness.ui_events.empty()


@pytest.mark.asyncio
async def test_low_latency_self_final_emits_active_update_with_merge_occupant_key() -> None:
    sink = RecordingOverlaySink()
    harness = compose_translation_test_harness(
        stt=None,
        llm=None,
        osc=RecordingOscQueue(),
        overlay_sink=sink,
        clock=FakeClock(_now=10.0),
        low_latency_mode=True,
    )
    utterance_id = uuid4()

    await harness.dispatch_stt_event(
        STTFinalEvent(
            utterance_id=utterance_id,
            transcript=Transcript(
                utterance_id=utterance_id,
                text="hello live",
                is_final=True,
                created_at=11.0,
            ),
        )
    )

    assert [event.type for event in sink.events] == ["self_active_update"]
    assert sink.events[0].text == "hello live"
    assert sink.events[0].occupant_key == f"self:{harness.self_owner.merge_buffer.merge_id}"
    assert harness.ui_events.empty()


@pytest.mark.asyncio
async def test_low_latency_self_active_updates_only_when_merged_text_changes() -> None:
    sink = RecordingOverlaySink()
    harness = compose_translation_test_harness(
        stt=None,
        llm=None,
        osc=RecordingOscQueue(),
        overlay_sink=sink,
        clock=FakeClock(_now=10.0),
        low_latency_mode=True,
    )
    utterance_id = uuid4()

    await harness.dispatch_stt_event(
        STTFinalEvent(
            utterance_id=utterance_id,
            transcript=Transcript(
                utterance_id=utterance_id,
                text="hello",
                is_final=True,
                created_at=12.0,
            ),
        )
    )
    await harness.dispatch_stt_event(
        STTFinalEvent(
            utterance_id=utterance_id,
            transcript=Transcript(
                utterance_id=utterance_id,
                text="hello",
                is_final=True,
                created_at=13.0,
            ),
        )
    )
    await harness.dispatch_stt_event(
        STTFinalEvent(
            utterance_id=utterance_id,
            transcript=Transcript(
                utterance_id=utterance_id,
                text="hello world",
                is_final=True,
                created_at=14.0,
            ),
        )
    )

    assert [event.type for event in sink.events] == [
        "self_active_update",
        "self_active_update",
    ]
    assert [event.text for event in sink.events] == ["hello", "hello world"]
    assert harness.self_owner.merge_buffer is not None
    assert [event.utterance_id for event in sink.events] == [
        harness.self_owner.merge_buffer.merge_id,
        harness.self_owner.merge_buffer.merge_id,
    ]


@pytest.mark.asyncio
async def test_low_latency_self_spec_translation_re_emits_active_update_with_secondary_only() -> (
    None
):
    sink = RecordingOverlaySink()
    osc = RecordingOscQueue()
    harness = compose_translation_test_harness(
        stt=None,
        llm=SequencedTranslateLLMProvider(responses=["translated live"]),
        osc=osc,
        overlay_sink=sink,
        clock=FakeClock(_now=10.0),
        low_latency_mode=True,
        low_latency_awaiting_vad_timeout_s=10.0,
    )
    utterance_id = uuid4()

    await harness.dispatch_stt_event(
        STTFinalEvent(
            utterance_id=utterance_id,
            transcript=Transcript(
                utterance_id=utterance_id,
                text="hello live",
                is_final=True,
                created_at=11.0,
            ),
        )
    )
    buffer = harness.self_owner.merge_buffer
    assert buffer is not None
    assert buffer.speculative_attempt is not None
    assert buffer.speculative_attempt.task is not None
    await asyncio.gather(buffer.speculative_attempt.task, return_exceptions=True)

    assert [event.type for event in sink.events] == [
        "self_active_update",
        "self_active_update",
    ]
    assert sink.events[0].text == "hello live"
    assert sink.events[0].secondary_text == ""
    assert sink.events[1].occupant_key == sink.events[0].occupant_key
    assert sink.events[1].text == "hello live"
    assert sink.events[1].secondary_text == "translated live"
    assert [event.type for event in sink.events if event.type != "self_active_update"] == []
    assert harness.self_owner.merge_buffer is buffer
    assert osc.messages == []


@pytest.mark.asyncio
async def test_low_latency_self_active_secondary_stays_sticky_on_soft_reuse_mismatch_then_recovers() -> (
    None
):
    sink = RecordingOverlaySink()
    harness = compose_translation_test_harness(
        stt=None,
        llm=SequencedTranslateLLMProvider(responses=["translated one", "translated two"]),
        osc=RecordingOscQueue(),
        overlay_sink=sink,
        clock=FakeClock(_now=10.0),
        low_latency_mode=True,
        low_latency_awaiting_vad_timeout_s=10.0,
    )
    utterance_id = uuid4()

    await harness.dispatch_stt_event(
        STTFinalEvent(
            utterance_id=utterance_id,
            transcript=Transcript(
                utterance_id=utterance_id,
                text="hello live",
                is_final=True,
                created_at=11.0,
            ),
        )
    )
    buffer = harness.self_owner.merge_buffer
    assert buffer is not None
    assert buffer.speculative_attempt is not None
    assert buffer.speculative_attempt.task is not None
    await asyncio.gather(buffer.speculative_attempt.task, return_exceptions=True)

    await harness.dispatch_stt_event(
        STTFinalEvent(
            utterance_id=utterance_id,
            transcript=Transcript(
                utterance_id=utterance_id,
                text="bye now",
                is_final=True,
                created_at=12.0,
            ),
        )
    )
    assert buffer.speculative_attempt is not None
    assert buffer.speculative_attempt.task is not None
    await asyncio.gather(buffer.speculative_attempt.task, return_exceptions=True)

    active_events = [event for event in sink.events if event.type == "self_active_update"]
    assert [event.secondary_text for event in active_events] == [
        "",
        "translated one",
        "translated one",
        "translated two",
    ]
    assert [event.text for event in active_events] == [
        "hello live",
        "hello live",
        "hello live bye now",
        "hello live bye now",
    ]
    assert [event.type for event in sink.events if event.type != "self_active_update"] == []


@pytest.mark.asyncio
async def test_low_latency_self_active_secondary_diagnostics_record_blank_sticky_and_spec_sources(
    tmp_path,
) -> None:
    sink = RecordingOverlaySink()
    diagnostics = OverlayDiagnosticsRecorder(
        overlay_instance_id="overlay-test",
        diagnostics_dir=tmp_path,
    )
    harness = compose_translation_test_harness(
        stt=None,
        llm=SequencedTranslateLLMProvider(responses=["translated one", "translated two"]),
        osc=RecordingOscQueue(),
        overlay_sink=sink,
        overlay_diagnostics=diagnostics,
        clock=FakeClock(_now=10.0),
        low_latency_mode=True,
        low_latency_awaiting_vad_timeout_s=10.0,
    )
    utterance_id = uuid4()

    await harness.dispatch_stt_event(
        STTFinalEvent(
            utterance_id=utterance_id,
            transcript=Transcript(
                utterance_id=utterance_id,
                text="hello live",
                is_final=True,
                created_at=11.0,
            ),
        )
    )
    buffer = harness.self_owner.merge_buffer
    assert buffer is not None
    assert buffer.speculative_attempt is not None
    assert buffer.speculative_attempt.task is not None
    await asyncio.gather(buffer.speculative_attempt.task, return_exceptions=True)

    await harness.dispatch_stt_event(
        STTFinalEvent(
            utterance_id=utterance_id,
            transcript=Transcript(
                utterance_id=utterance_id,
                text="bye now",
                is_final=True,
                created_at=12.0,
            ),
        )
    )

    assert buffer.speculative_attempt is not None
    assert buffer.speculative_attempt.task is not None
    await asyncio.gather(buffer.speculative_attempt.task, return_exceptions=True)
    assert list(diagnostics.translation_events) == []


@pytest.mark.asyncio
async def test_translation_active_self_metadata_flows_through_presenter_accessor() -> None:
    bridge = RecordingPresentationBridge()
    clock = FakeClock(_now=10.0)
    presenter = OverlayPresenter(
        bridge=bridge,
        calibration=OverlayCalibration(),
        clock=clock,
    )
    harness = compose_translation_test_harness(
        stt=None,
        llm=None,
        osc=RecordingOscQueue(),
        overlay_sink=presenter,
        clock=clock,
        low_latency_mode=True,
    )
    merge_id = uuid4()
    logical_turn_key = f"self:{merge_id}"
    expected_metadata = {
        "update_id": "self-active-update-1",
        "origin_wall_clock_ms": 123456789,
        "session_scope": "self-active-session",
        "source_text_hash": "0123456789abcdef",
        "source_text_len": len("hello live"),
        "logical_turn_key": logical_turn_key,
    }
    buffer = _MergeBuffer(
        merge_id=merge_id,
        parts=["hello live"],
        utterance_ids=[uuid4()],
        speculative_attempt=make_speculative_attempt(
            source_text="hello live",
            result=Translation(
                utterance_id=merge_id,
                text="translated live",
                source_text="hello live",
                source_language="ko",
                target_language="en",
                channel="self",
                created_at=10.0,
                **expected_metadata,
            ),
        ),
    )
    harness.self_owner.merge_buffer = buffer

    await harness.self_owner._sync_overlay_active_self(buffer, created_at=harness.clock.now())

    metadata = presenter.active_self_overlay_metadata()
    assert metadata == ActiveSelfOverlayMetadata(
        text="hello live",
        secondary_text="translated live",
        utterance_id=merge_id,
        occupant_key=f"self:{merge_id}",
        update_id="self-active-update-1",
        origin_wall_clock_ms=123456789,
        session_scope="self-active-session",
        source_text_hash="0123456789abcdef",
        source_text_len=len("hello live"),
        logical_turn_key=logical_turn_key,
        primary_language="ko",
        secondary_language="en",
    )
    active_block = presenter.snapshot().blocks[0]
    assert active_block.id == f"self:{merge_id}"
    assert active_block.primary_text == "hello live"
    assert active_block.secondary_text == "translated live"
    assert active_block.primary_language == "ko"
    assert active_block.secondary_language == "en"
    assert active_block.occupant_key == f"self:{merge_id}"
    assert {
        "update_id": active_block.update_id,
        "origin_wall_clock_ms": active_block.origin_wall_clock_ms,
        "session_scope": active_block.session_scope,
        "source_text_hash": active_block.source_text_hash,
        "source_text_len": active_block.source_text_len,
        "logical_turn_key": active_block.logical_turn_key,
    } == expected_metadata


@pytest.mark.asyncio
async def test_self_overlay_secondary_decision_logs_only_to_detailed_runtime_log() -> None:
    basic_runtime_logging, basic_stream = _make_runtime_logging_capture()
    detailed_runtime_logging, detailed_stream = _make_runtime_logging_capture()
    detailed_runtime_logging.set_mode(SessionLoggingMode.DETAILED)

    # Contract under test: runtime detailed logging must emit the
    # active_self_secondary token even when overlay_diagnostics is absent.
    basic_sink = RecordingOverlaySink()
    detailed_sink = RecordingOverlaySink()
    basic_harness = compose_translation_test_harness(
        stt=None,
        llm=None,
        osc=RecordingOscQueue(),
        overlay_sink=basic_sink,
        overlay_diagnostics=None,
        runtime_logging=basic_runtime_logging,
        clock=FakeClock(_now=10.0),
        low_latency_mode=True,
    )
    detailed_harness = compose_translation_test_harness(
        stt=None,
        llm=None,
        osc=RecordingOscQueue(),
        overlay_sink=detailed_sink,
        overlay_diagnostics=None,
        runtime_logging=detailed_runtime_logging,
        clock=FakeClock(_now=20.0),
        low_latency_mode=True,
    )

    basic_buffer = _MergeBuffer(
        merge_id=uuid4(),
        parts=["hello live"],
        utterance_ids=[uuid4()],
    )
    detailed_buffer = _MergeBuffer(
        merge_id=uuid4(),
        parts=["hello live"],
        utterance_ids=[uuid4()],
    )
    basic_harness.self_owner.merge_buffer = basic_buffer
    detailed_harness.self_owner.merge_buffer = detailed_buffer
    basic_sink.active_self_metadata = _active_self_metadata_for_buffer(
        basic_buffer,
        text="hello live",
        secondary_text="translated live",
    )
    detailed_sink.active_self_metadata = _active_self_metadata_for_buffer(
        detailed_buffer,
        text="hello live",
        secondary_text="translated live",
    )

    try:
        assert basic_harness.translation_diagnostics.overlay_diagnostics is None
        assert detailed_harness.translation_diagnostics.overlay_diagnostics is None

        await basic_harness.self_owner._sync_overlay_active_self(
            basic_buffer, created_at=basic_harness.clock.now()
        )
        await detailed_harness.self_owner._sync_overlay_active_self(
            detailed_buffer,
            created_at=detailed_harness.clock.now(),
        )

        basic_messages = _runtime_log_messages(basic_stream)
        detailed_messages = _runtime_log_messages(detailed_stream)
        basic_decision_messages = [
            message for message in basic_messages if "active_self_secondary" in message
        ]
        detailed_decision_messages = [
            message for message in detailed_messages if "active_self_secondary" in message
        ]

        assert basic_decision_messages == []
        assert detailed_decision_messages != [], (
            "expected runtime detailed logging to emit active_self_secondary "
            "without overlay_diagnostics"
        )
    finally:
        basic_runtime_logging.close()
        detailed_runtime_logging.close()
        await basic_harness.stop()
        await detailed_harness.stop()


@pytest.mark.asyncio
async def test_self_overlay_secondary_decision_emits_after_basic_to_detailed_mode_switch() -> None:
    runtime_logging, log_stream = _make_runtime_logging_capture()
    sink = RecordingOverlaySink()
    harness = compose_translation_test_harness(
        stt=None,
        llm=None,
        osc=RecordingOscQueue(),
        overlay_sink=sink,
        overlay_diagnostics=None,
        runtime_logging=runtime_logging,
        clock=FakeClock(_now=10.0),
        low_latency_mode=True,
    )
    buffer = _MergeBuffer(
        merge_id=uuid4(),
        parts=["hello live"],
        utterance_ids=[uuid4()],
    )
    harness.self_owner.merge_buffer = buffer
    sink.active_self_metadata = _active_self_metadata_for_buffer(
        buffer,
        text="hello live",
        secondary_text="translated live",
    )

    try:
        await harness.self_owner._sync_overlay_active_self(buffer, created_at=harness.clock.now())
        assert not any(
            "active_self_secondary" in message for message in _runtime_log_messages(log_stream)
        )

        runtime_logging.set_mode(SessionLoggingMode.DETAILED)
        await harness.self_owner._sync_overlay_active_self(buffer, created_at=harness.clock.now())

        assert any(
            "active_self_secondary" in message for message in _runtime_log_messages(log_stream)
        )
    finally:
        runtime_logging.close()
        await harness.stop()


@pytest.mark.asyncio
async def test_low_latency_self_active_secondary_stays_sticky_through_resume_continuation() -> None:
    sink = RecordingOverlaySink()
    harness = compose_translation_test_harness(
        stt=None,
        llm=SequencedTranslateLLMProvider(responses=["translated live", "translated continued"]),
        osc=RecordingOscQueue(),
        overlay_sink=sink,
        clock=FakeClock(_now=10.0),
        low_latency_mode=True,
        low_latency_awaiting_vad_timeout_s=10.0,
    )
    first_utterance_id = uuid4()
    resumed_utterance_id = uuid4()

    await harness.dispatch_stt_event(
        STTFinalEvent(
            utterance_id=first_utterance_id,
            transcript=Transcript(
                utterance_id=first_utterance_id,
                text="hello live",
                is_final=True,
                created_at=11.0,
            ),
        )
    )
    buffer = harness.self_owner.merge_buffer
    assert buffer is not None
    assert buffer.speculative_attempt is not None
    assert buffer.speculative_attempt.task is not None
    await asyncio.gather(buffer.speculative_attempt.task, return_exceptions=True)

    await harness.self_owner.handle_vad_event(
        SpeechStart(
            resumed_utterance_id,
            pre_roll=np.zeros((0,), dtype=np.float32),
            chunk=np.zeros((1,), dtype=np.float32),
        )
    )
    for _ in range(3):
        await harness.self_owner.handle_vad_event(
            SpeechChunk(
                resumed_utterance_id,
                chunk=np.zeros((1,), dtype=np.float32),
            )
        )

    await harness.dispatch_stt_event(
        STTFinalEvent(
            utterance_id=resumed_utterance_id,
            transcript=Transcript(
                utterance_id=resumed_utterance_id,
                text="again",
                is_final=True,
                created_at=12.0,
            ),
        )
    )
    assert buffer.speculative_attempt is not None
    assert buffer.speculative_attempt.task is not None
    await asyncio.gather(buffer.speculative_attempt.task, return_exceptions=True)

    active_events = [event for event in sink.events if event.type == "self_active_update"]
    assert [event.secondary_text for event in active_events] == [
        "",
        "translated live",
        "translated live",
        "translated continued",
    ]
    assert [event.text for event in active_events] == [
        "hello live",
        "hello live",
        "hello live again",
        "hello live again",
    ]
    assert [event.type for event in sink.events if event.type != "self_active_update"] == []
    assert harness.self_owner.merge_buffer is buffer


@pytest.mark.asyncio
async def test_low_latency_merge_commit_reuses_merge_identity_without_emitting_clear() -> None:
    sink = RecordingOverlaySink()
    harness = compose_translation_test_harness(
        stt=None,
        llm=None,
        osc=RecordingOscQueue(),
        overlay_sink=sink,
        clock=FakeClock(_now=10.0),
        low_latency_mode=True,
        low_latency_finalize_wait_ms=0,
    )
    utterance_id = uuid4()

    await harness.dispatch_stt_event(
        STTFinalEvent(
            utterance_id=utterance_id,
            transcript=Transcript(
                utterance_id=utterance_id,
                text="hello live",
                is_final=True,
                created_at=11.0,
            ),
        )
    )
    active_event = sink.events[-1]
    await harness.self_owner.handle_vad_event(SpeechEnd(utterance_id))
    await harness.dispatch_stt_event(
        STTFinalEvent(
            utterance_id=utterance_id,
            transcript=Transcript(
                utterance_id=utterance_id,
                text="hello live",
                is_final=True,
                created_at=12.0,
            ),
        )
    )

    assert [event.type for event in sink.events] == [
        "self_active_update",
        "self_transcript_final",
        "utterance_closed",
    ]
    final_event = next(event for event in sink.events if event.type == "self_transcript_final")
    assert active_event.utterance_id == final_event.utterance_id
    assert active_event.occupant_key == f"self:{final_event.utterance_id}"


@pytest.mark.asyncio
async def test_low_latency_self_active_update_failures_do_not_break_harness() -> None:
    sink = FailingOverlaySink()
    harness = compose_translation_test_harness(
        stt=None,
        llm=None,
        osc=RecordingOscQueue(),
        overlay_sink=sink,
        clock=FakeClock(_now=10.0),
        low_latency_mode=True,
    )
    utterance_id = uuid4()

    await harness.dispatch_stt_event(
        STTFinalEvent(
            utterance_id=utterance_id,
            transcript=Transcript(
                utterance_id=utterance_id,
                text="hello live",
                is_final=True,
                created_at=11.0,
            ),
        )
    )

    assert harness.translation_diagnostics.snapshot().last_error_source == "overlay_sink"
    assert harness.ui_events.empty()
