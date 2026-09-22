from __future__ import annotations

import json
from dataclasses import dataclass, field
from uuid import uuid4

import pytest

from puripuly_heart.core.audio.ownership import (
    AudioSegmentIdentity,
    AudioSegmentSettingsSnapshot,
    AudioSegmentSnapshot,
    AudioSegmentTerminalReceipt,
)
from puripuly_heart.core.clock import FakeClock
from puripuly_heart.core.overlay.presenter import OverlayPresenter
from puripuly_heart.core.stt.backend import STTProviderTurnIdentity, STTProviderTurnTerminal
from puripuly_heart.domain.events import STTFinalEvent
from puripuly_heart.domain.models import Transcript, Translation
from puripuly_heart.providers.stt.soniox import _FinalizeRequest, _SonioxSession
from puripuly_heart.ui.overlay_calibration import OverlayCalibration
from tests.helpers.fakes import RecordingOscQueue
from tests.helpers.translation_owners import compose_translation_test_harness


@dataclass(slots=True)
class _EchoTranslationProvider:
    calls: list[str] = field(default_factory=list)

    async def translate(
        self,
        *,
        utterance_id,
        text: str,
        system_prompt: str,
        source_language: str,
        target_language: str,
        context: str = "",
        scene_participant_count: int | None = None,
    ) -> Translation:
        _ = (
            system_prompt,
            source_language,
            target_language,
            context,
            scene_participant_count,
        )
        self.calls.append(text)
        return Translation(
            utterance_id=utterance_id,
            text=f"translated:{text}",
            source_text=text,
            source_language=source_language,
            target_language=target_language,
            channel="self" if text == "self" else "peer",
        )

    async def close(self) -> None:
        return None


@dataclass(slots=True)
class _RecordingOverlaySink:
    presenter: OverlayPresenter
    events: list[object] = field(default_factory=list)

    async def emit(self, event: object) -> None:
        self.events.append(event)
        await self.presenter.emit(event)  # type: ignore[arg-type]


def _soniox_session() -> _SonioxSession:
    return _SonioxSession(
        api_key="test",
        model="stt-rt-v5",
        endpoint="wss://example.invalid",
        sample_rate_hz=16000,
        language_hints=["en"],
        context_terms=[],
        keepalive_interval_s=10.0,
        trailing_silence_ms=100,
        connect_timeout_s=1.0,
        enable_language_identification=True,
        enable_speaker_diarization=True,
    )


def _settings() -> AudioSegmentSettingsSnapshot:
    return AudioSegmentSettingsSnapshot(
        provider_id="soniox",
        provider_signature=("soniox",),
        runtime_signature=("soniox",),
        source_mode="desktop",
        source_language="en",
        expected_languages=("en",),
        target_sample_rate_hz=16000,
        vad_speech_threshold=0.4,
        vad_hangover_ms=800,
        vad_pre_roll_ms=500,
    )


async def _terminal_from_soniox_tokens(
    session: _SonioxSession,
    *,
    speaker: str,
    text: str,
    start_ms: int,
    tokens: list[dict[str, object]] | None = None,
    end_ms: int,
    generation: int,
    order: int,
) -> tuple[AudioSegmentTerminalReceipt, STTProviderTurnTerminal]:
    await session.on_speech_end(trailing_silence_ms=500)
    assert isinstance(await session._audio_q.get(), _FinalizeRequest)
    message_tokens = (
        tokens
        if tokens is not None
        else [
            {
                "text": text,
                "language": "en",
                "speaker": speaker,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "confidence": 0.97,
                "is_final": True,
            }
        ]
    )
    session._handle_message(
        json.dumps(
            {
                "tokens": message_tokens + [{"text": "<fin>", "is_final": True}],
            }
        )
    )
    event = session._event_projection._legacy_events.get_nowait()
    segment_id = uuid4()
    segment_identity = AudioSegmentIdentity(generation, order, segment_id, 1)
    provider_identity = STTProviderTurnIdentity(
        segment=segment_identity,
        provider_epoch_id=f"epoch-{generation}",
        provider_turn_id=f"turn-{generation}-{order}",
    )
    snapshot = AudioSegmentSnapshot(
        identity=segment_identity,
        settings=_settings(),
        content_ranges=(),
        context_ranges=(),
        failed_ranges=(),
        content_sample_count=0,
        context_sample_count=0,
        failed_normalized_sample_count=0,
        failed_source_sample_count=0,
        prefix_context_sample_count=0,
        synthetic_context_sample_count=0,
        genuine_onset=True,
        state="terminal",
        opened_at_monotonic_s=0.0,
        sealed_at_monotonic_s=0.0,
        seal_reason="silence",
    )
    receipt = AudioSegmentTerminalReceipt(
        identity=segment_identity,
        outcome="final",
        segment=snapshot,
        terminal_at_monotonic_s=0.0,
        provider_epoch_id=provider_identity.provider_epoch_id,
        provider_turn_id=provider_identity.provider_turn_id,
        text_authority="authoritative",
    )
    terminal = STTProviderTurnTerminal(
        identity=provider_identity,
        outcome="final",
        text=event.text,
        final_language_runs=event.final_language_runs,
        final_speaker_runs=event.final_speaker_runs,
        text_authority="authoritative",
    )
    return receipt, terminal


@pytest.mark.asyncio
async def test_soniox_tokens_replay_through_translation_publication_and_presenter() -> None:
    clock = FakeClock(_now=100.0)
    presenter = OverlayPresenter(
        calibration=OverlayCalibration(),
        clock=clock,
        speaker_transition_mode="C",
    )
    provider = _EchoTranslationProvider()
    overlay = _RecordingOverlaySink(presenter)
    harness = compose_translation_test_harness(
        stt=None,
        llm=provider,
        osc=RecordingOscQueue(),
        peer_translation_enabled=True,
        overlay_sink=overlay,
        clock=clock,
    )
    session = _soniox_session()
    observed: list[tuple[str, str, bool]] = []

    async def publish_peer(speaker: str, order: int, start_ms: int) -> None:
        receipt, terminal = await _terminal_from_soniox_tokens(
            session,
            speaker=speaker,
            text=f"{speaker}-{order}",
            start_ms=start_ms,
            end_ms=start_ms + 90,
            generation=1,
            order=order,
        )
        harness.record_peer_speech_end_for_test(receipt.identity.segment_id)
        await harness.peer_owner.handle_provider_turn_terminal(receipt, terminal)
        await harness.peer_owner.translation_turns.wait_for_idle()
        await harness.output_runtime.wait_for_peer_output_idle()
        assert provider.calls, (
            presenter.snapshot(),
            harness.output_runtime.routing_decisions,
            overlay.events,
        )
        assert presenter.snapshot().blocks, [
            (decision.route, decision.reason, decision.metadata)
            for decision in harness.output_runtime.routing_decisions
        ]
        block = max(
            (block for block in presenter.snapshot().blocks if block.channel == "peer"),
            key=lambda block: block.appearance_seq,
        )
        observed.append((speaker, block.speaker_style or "", block.speaker_boundary))

    harness.output_runtime.activate_peer_generation(1)

    await harness.start()
    try:
        await publish_peer("A", 1, 100)
        await publish_peer("A", 2, 200)
        await publish_peer("B", 3, 300)
        await publish_peer("B", 4, 400)
        self_id = uuid4()
        await harness.dispatch_stt_event(
            STTFinalEvent(
                utterance_id=self_id,
                transcript=Transcript(
                    self_id,
                    "self",
                    True,
                    channel="self",
                ),
            )
        )
        await harness.peer_owner.translation_turns.wait_for_idle()
        self_block = max(
            (block for block in presenter.snapshot().blocks if block.channel == "self"),
            key=lambda block: block.appearance_seq,
        )
        assert self_block.speaker_style is None
        await publish_peer("B", 5, 500)
        await publish_peer("C", 6, 600)
        await publish_peer("A", 7, 700)

        assert observed == [
            ("A", "gold", False),
            ("A", "gold", False),
            ("B", "cyan", False),
            ("B", "cyan", False),
            ("B", "cyan", False),
            ("C", "gold", False),
            ("A", "cyan", False),
        ]
        assert provider.calls == [
            "A-1",
            "A-2",
            "B-3",
            "B-4",
            "self",
            "B-5",
            "C-6",
            "A-7",
        ]

        overlap_receipt, overlap_terminal = await _terminal_from_soniox_tokens(
            session,
            speaker="unused",
            text="unused",
            start_ms=800,
            end_ms=1000,
            generation=1,
            order=8,
            tokens=[
                {
                    "text": "A-overlap ",
                    "language": "en",
                    "speaker": "A",
                    "start_ms": 800,
                    "end_ms": 950,
                    "confidence": 0.96,
                    "is_final": True,
                },
                {
                    "text": "B-overlap",
                    "language": "en",
                    "speaker": "B",
                    "start_ms": 900,
                    "end_ms": 1000,
                    "confidence": 0.94,
                    "is_final": True,
                },
            ],
        )
        harness.record_peer_speech_end_for_test(overlap_receipt.identity.segment_id)
        await harness.peer_owner.handle_provider_turn_terminal(
            overlap_receipt,
            overlap_terminal,
        )
        await harness.peer_owner.translation_turns.wait_for_idle()
        await harness.output_runtime.wait_for_peer_output_idle()
        overlap_translations = [
            event for event in overlay.events if getattr(event, "type", None) == "translation_final"
        ][-2:]
        assert [event.source_text for event in overlap_translations] == [
            "A-overlap ",
            "B-overlap",
        ]
        assert [event.speaker_transition for event in overlap_translations] == [
            "continuity",
            "unavailable",
        ]
        overlap_blocks = {
            block.id: block
            for block in presenter.snapshot().blocks
            if block.id in {f"peer:{event.utterance_id}" for event in overlap_translations}
        }
        assert [
            overlap_blocks[f"peer:{event.utterance_id}"].speaker_style
            for event in overlap_translations
        ] == [
            "cyan",
            "cyan",
        ]

        reset_receipt, reset_terminal = await _terminal_from_soniox_tokens(
            session,
            speaker="B",
            text="generation reset",
            start_ms=800,
            end_ms=890,
            generation=2,
            order=1,
        )
        harness.output_runtime.activate_peer_generation(2)
        harness.record_peer_speech_end_for_test(reset_receipt.identity.segment_id)
        await harness.peer_owner.handle_provider_turn_terminal(reset_receipt, reset_terminal)
        await harness.peer_owner.translation_turns.wait_for_idle()
        await harness.output_runtime.wait_for_peer_output_idle()
        reset_block = max(
            (block for block in presenter.snapshot().blocks if block.channel == "peer"),
            key=lambda block: block.appearance_seq,
        )
        assert reset_block.speaker_style == "cyan"

        reconnect = _soniox_session()
        reconnect_receipt, reconnect_terminal = await _terminal_from_soniox_tokens(
            reconnect,
            speaker="C",
            text="scope reset",
            start_ms=900,
            end_ms=990,
            generation=2,
            order=2,
        )
        harness.record_peer_speech_end_for_test(reconnect_receipt.identity.segment_id)
        await harness.peer_owner.handle_provider_turn_terminal(
            reconnect_receipt, reconnect_terminal
        )
        await harness.peer_owner.translation_turns.wait_for_idle()
        await harness.output_runtime.wait_for_peer_output_idle()
        reconnect_block = max(
            (block for block in presenter.snapshot().blocks if block.channel == "peer"),
            key=lambda block: block.appearance_seq,
        )
        assert reconnect_block.speaker_style == "cyan"
    finally:
        await harness.stop()
        await presenter.close()
