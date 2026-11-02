from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field, replace
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
from puripuly_heart.core.stt.backend import (
    STTProviderTurnIdentity,
    STTProviderTurnTerminal,
    STTProviderTurnUpdate,
)
from puripuly_heart.core.stt.scoped_normalizer import STTScopedTurnNormalizer
from puripuly_heart.domain.models import Translation
from puripuly_heart.providers.stt.soniox import _FinalizeRequest, _SonioxSession
from puripuly_heart.ui.overlay_calibration import OverlayCalibration
from tests.helpers.fakes import RecordingOscQueue
from tests.helpers.translation_owners import compose_translation_test_harness


@dataclass(slots=True)
class _SegmentTranslationProvider:
    calls: list[str] = field(default_factory=list)
    active: int = 0
    peak: int = 0
    release: dict[str, asyncio.Event] = field(default_factory=dict)
    entered: dict[str, asyncio.Event] = field(default_factory=dict)
    cancelled: set[str] = field(default_factory=set)
    failures: set[str] = field(default_factory=set)

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
        self.calls.append(text)
        self.active += 1
        self.peak = max(self.peak, self.active)
        self.entered.setdefault(text, asyncio.Event()).set()
        try:
            gate = self.release.get(text)
            if gate is not None:
                await gate.wait()
            if text in self.failures:
                raise RuntimeError("selected translation failure")
            translations = {
                "Alice ": "Hello",
                "Beto ": "World",
                "Beto": "World",
                "Alice": "Again",
                "alpha ": "First",
                "beta": "Second",
                "Cora": "Later",
                "alpha,": "First,",
                "alpha, beta": "First, Second",
                "alpha, beta Alice": "First, Second Again",
            }
            return Translation(
                utterance_id=utterance_id,
                text=translations[text],
                source_text=text,
                source_language=source_language,
                target_language=target_language,
                channel="peer",
            )
        except asyncio.CancelledError:
            self.cancelled.add(text)
            raise
        finally:
            self.active -= 1

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
    malformed_run_text: str | None = None,
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
    speaker_runs = event.final_speaker_runs
    if malformed_run_text is not None:
        speaker_runs = (replace(speaker_runs[0], text=malformed_run_text),)
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
    terminal = STTScopedTurnNormalizer(provider_identity).apply_terminal(
        STTProviderTurnTerminal(
            identity=provider_identity,
            outcome="final",
            text=event.text,
            final_language_runs=event.final_language_runs,
            final_speaker_runs=speaker_runs,
            text_authority="authoritative",
        )
    )
    return receipt, terminal


async def _wait_for_peer_idle(harness) -> None:
    await harness.peer_owner.cloud_captions.wait_for_idle()
    await harness.peer_owner.translation_turns.wait_for_idle()
    await harness.output_runtime.wait_for_peer_output_idle()


async def _partial_before_final(harness, terminal: STTProviderTurnTerminal) -> None:
    await harness.peer_owner.handle_provider_update(
        STTProviderTurnUpdate(
            identity=replace(terminal.identity, settings_scope=("soniox",)),
            sequence=1,
            stability="provisional",
            assembly="replace",
            text=terminal.text,
            observed_at=harness.peer_owner.clock.now(),
        )
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("translate", [True, False], ids=["translated", "source-only"])
@pytest.mark.parametrize(
    ("mode", "style"),
    [
        ("off", "gold"),
        ("unsupported", "gold"),
        ("missing", "gray"),
        ("malformed", "gray"),
        ("invalid-runs", "gray"),
    ],
)
async def test_provider_speaker_availability_is_visible_on_peer_output(
    mode: str, style: str, translate: bool
) -> None:
    clock = FakeClock(_now=100.0)
    presenter = OverlayPresenter(calibration=OverlayCalibration(), clock=clock)
    overlay = _RecordingOverlaySink(presenter)
    harness = compose_translation_test_harness(
        stt=None,
        llm=_SegmentTranslationProvider() if translate else None,
        osc=RecordingOscQueue(),
        peer_translation_enabled=translate,
        overlay_sink=overlay,
        clock=clock,
    )
    session = _soniox_session()
    if mode in ("off", "unsupported"):
        session.enable_speaker_diarization = False
    harness.output_runtime.activate_peer_generation(1)
    await harness.start()
    try:
        token: dict[str, object] = {
            "text": "Alice",
            "language": "en",
            "start_ms": 100,
            "end_ms": 190,
            "is_final": True,
        }
        if mode in ("invalid-runs", "off", "unsupported"):
            token["speaker"] = "A"
        elif mode == "malformed":
            token["speaker"] = []
        receipt, terminal = await _terminal_from_soniox_tokens(
            session,
            speaker="A",
            text="Alice",
            start_ms=100,
            end_ms=190,
            tokens=[token],
            generation=1,
            order=1,
            malformed_run_text="mismatch" if mode == "invalid-runs" else None,
        )
        if mode == "unsupported":
            receipt = replace(
                receipt,
                segment=replace(
                    receipt.segment,
                    settings=replace(
                        receipt.segment.settings,
                        provider_id="qwen3_asr",
                        provider_signature=("qwen3_asr",),
                        runtime_signature=("qwen3_asr",),
                    ),
                ),
            )
        harness.record_peer_speech_end_for_test(receipt.identity.segment_id)
        await harness.peer_owner.handle_provider_turn_terminal(receipt, terminal)
        await _wait_for_peer_idle(harness)
        event_type = "translation_final" if translate else "peer_transcript_final"
        matching = [event for event in overlay.events if getattr(event, "type", None) == event_type]
        assert len(matching) == 1
        assert (matching[0].source_text if translate else matching[0].text) == "Alice"
        blocks = presenter.snapshot().blocks
        assert len(blocks) == 1
        assert blocks[0].speaker_style == style
        assert blocks[0].channel == "peer"
    finally:
        await harness.stop()
        await presenter.close()


@pytest.mark.asyncio
async def test_soniox_segments_preserve_attribution_through_output_and_presenter() -> None:
    clock = FakeClock(_now=100.0)

    async def advance(seconds: float) -> None:
        await asyncio.sleep(seconds)
        clock.advance(seconds)

    presenter = OverlayPresenter(calibration=OverlayCalibration(), clock=clock, sleep=advance)
    provider = _SegmentTranslationProvider()
    provider.release["Alice "] = asyncio.Event()
    provider.release["Beto "] = asyncio.Event()
    provider.release["Alice"] = asyncio.Event()
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
    harness.output_runtime.activate_peer_generation(1)
    await harness.start()
    try:
        tokens = [
            {
                "text": word,
                "language": language,
                "speaker": speaker,
                "start_ms": index * 100,
                "end_ms": index * 100 + 90,
                "confidence": 0.01,
                "is_final": True,
            }
            for index, (word, language, speaker) in enumerate(
                [("Alice ", "en", "A"), ("Beto ", "es", "B"), ("Alice", "en", "A")],
                start=1,
            )
        ]
        receipt, terminal = await _terminal_from_soniox_tokens(
            session,
            speaker="A",
            text="",
            start_ms=100,
            end_ms=390,
            tokens=tokens,
            generation=1,
            order=1,
        )
        assert terminal.text == "Alice Beto Alice"
        assert [run.attribution.state for run in terminal.final_speaker_runs] == [
            "identified",
            "identified",
            "identified",
        ]
        harness.record_peer_speech_end_for_test(receipt.identity.segment_id)
        await harness.peer_owner.handle_provider_turn_terminal(receipt, terminal)
        async with asyncio.timeout(2):
            while len(provider.entered) != 3:
                await asyncio.sleep(0)
        assert provider.peak == 3
        provider.release["Alice "].set()
        async with asyncio.timeout(2):
            while not any(
                getattr(event, "type", None) == "translation_final"
                and event.source_text == "Alice "
                for event in overlay.events
            ):
                await asyncio.sleep(0)
        assert [
            event.source_text
            for event in overlay.events
            if getattr(event, "type", None) == "translation_final"
        ] == ["Alice "]
        provider.release["Alice"].set()
        await asyncio.sleep(0)
        assert [
            event.source_text
            for event in overlay.events
            if getattr(event, "type", None) == "translation_final"
        ] == ["Alice "]
        provider.release["Beto "].set()
        await _wait_for_peer_idle(harness)
        assert provider.calls == ["Alice ", "Beto ", "Alice"]
        events = [
            event for event in overlay.events if getattr(event, "type", None) == "translation_final"
        ]
        assert [event.source_text for event in events] == ["Alice ", "Beto ", "Alice"]
        assert [event.text for event in events] == ["Hello", "World", "Again"]
        assert [event.speaker_assignment.palette_index for event in events] == [0, 1, 0]
        assert [event.speaker_assignment.source_text_range for event in events] == [
            (0, 6),
            (6, 11),
            (11, 16),
        ]
        revisions = {event.speaker_assignment.source_text_revision for event in events}
        assert len(revisions) == 1 and None not in revisions
        latest = presenter.snapshot().blocks
        assert latest[-1].speaker_style == "gold"
        assert latest[-1].primary_text == "Again"
        assert latest[-1].secondary_text == "Alice"
    finally:
        provider.release["Alice "].set()
        provider.release["Alice"].set()
        provider.release["Beto "].set()
        await harness.stop()
        await presenter.close()


@pytest.mark.asyncio
async def test_failed_segment_keeps_sibling_translation_and_every_source_fragment() -> None:
    clock = FakeClock(_now=100.0)
    presenter = OverlayPresenter(calibration=OverlayCalibration(), clock=clock)
    provider = _SegmentTranslationProvider(failures={"alpha "})
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
    harness.output_runtime.activate_peer_generation(1)
    await harness.start()
    try:
        receipt, terminal = await _terminal_from_soniox_tokens(
            session,
            speaker="A",
            text="",
            start_ms=100,
            end_ms=290,
            tokens=[
                {
                    "text": word,
                    "speaker": speaker,
                    "language": "en",
                    "start_ms": index * 100,
                    "end_ms": index * 100 + 90,
                    "is_final": True,
                }
                for index, (word, speaker) in enumerate(
                    [("alpha ", "A"), ("beta", "B")],
                    start=1,
                )
            ],
            generation=1,
            order=1,
        )
        harness.record_peer_speech_end_for_test(receipt.identity.segment_id)
        await harness.peer_owner.handle_provider_turn_terminal(receipt, terminal)
        await _wait_for_peer_idle(harness)
        assert provider.calls == ["alpha ", "beta"]
        original = [
            event.source_text
            for event in overlay.events
            if getattr(event, "type", None) == "translation_final" and not event.text
        ]
        assert original == ["alpha "]
        translated = [
            event
            for event in overlay.events
            if getattr(event, "type", None) == "translation_final" and event.text
        ]
        assert [
            (event.source_text, event.text, event.speaker_assignment.palette_index)
            for event in translated
        ] == [("beta", "Second", 1)]
    finally:
        await harness.stop()
        await presenter.close()


@pytest.mark.asyncio
async def test_parent_admission_allocates_every_speaker_before_child_ui_yields() -> None:
    harness = compose_translation_test_harness(
        stt=None,
        llm=None,
        osc=RecordingOscQueue(),
        peer_translation_enabled=False,
    )
    session = _soniox_session()
    entered = asyncio.Event()
    second_created = asyncio.Event()
    release = asyncio.Event()
    original_child_created = harness.translation_turns.on_child_created
    observed: dict[str, int | None] = {}

    async def block_after_first_child(child):
        if child.transcript.text == "Cora":
            assignment = harness.peer_owner._speaker_identities.assignment_for(child.utterance_id)
            observed[child.transcript.text] = (
                None if assignment is None else assignment.palette_index
            )
            second_created.set()
        await original_child_created(child)
        assignment = harness.peer_owner._speaker_identities.assignment_for(child.utterance_id)
        observed[child.transcript.text] = None if assignment is None else assignment.palette_index
        if child.transcript.text == "Alice ":
            entered.set()
            await release.wait()

    harness.translation_turns.on_child_created = block_after_first_child
    harness.output_runtime.activate_peer_generation(1)
    await harness.start()
    try:
        first_receipt, first_terminal = await _terminal_from_soniox_tokens(
            session,
            speaker="A",
            text="",
            start_ms=100,
            end_ms=290,
            generation=1,
            order=1,
            tokens=[
                {
                    "text": text,
                    "speaker": speaker,
                    "language": "en",
                    "is_final": True,
                    "start_ms": index * 100,
                    "end_ms": index * 100 + 90,
                }
                for index, (text, speaker) in enumerate((("Alice ", "A"), ("Beto", "B")), start=1)
            ],
        )
        second_receipt, second_terminal = await _terminal_from_soniox_tokens(
            session,
            speaker="C",
            text="Cora",
            start_ms=300,
            end_ms=390,
            generation=1,
            order=2,
        )
        harness.record_peer_speech_end_for_test(first_receipt.identity.segment_id)
        harness.record_peer_speech_end_for_test(second_receipt.identity.segment_id)
        first = asyncio.create_task(
            harness.peer_owner.handle_provider_turn_terminal(first_receipt, first_terminal)
        )
        await asyncio.wait_for(entered.wait(), 2)
        second = asyncio.create_task(
            harness.peer_owner.handle_provider_turn_terminal(second_receipt, second_terminal)
        )
        await asyncio.wait_for(second_created.wait(), 2)
        assert observed == {"Alice ": 0, "Cora": 2}
        release.set()
        await asyncio.wait_for(first, 2)
        await asyncio.wait_for(second, 2)
        await asyncio.wait_for(harness.translation_turns.wait_for_idle(), 4)
        assert observed["Beto"] == 1
    finally:
        release.set()
        await harness.stop()


@pytest.mark.asyncio
async def test_stalled_peer_child_times_out_without_losing_siblings_or_successor() -> None:
    clock = FakeClock(_now=100.0)
    presenter = OverlayPresenter(calibration=OverlayCalibration(), clock=clock)
    provider = _SegmentTranslationProvider(release={"alpha ": asyncio.Event()})
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
    harness.peer_owner.cloud_captions.request_timeout_s = 0.08
    harness.output_runtime.activate_peer_generation(1)
    await harness.start()
    try:
        first_receipt, first_terminal = await _terminal_from_soniox_tokens(
            session,
            speaker="A",
            text="",
            start_ms=100,
            end_ms=290,
            generation=1,
            order=1,
            tokens=[
                {
                    "text": text,
                    "speaker": speaker,
                    "language": "en",
                    "is_final": True,
                    "start_ms": index * 100,
                    "end_ms": index * 100 + 90,
                }
                for index, (text, speaker) in enumerate((("alpha ", "A"), ("beta", "B")), start=1)
            ],
        )
        second_receipt, second_terminal = await _terminal_from_soniox_tokens(
            session,
            speaker="C",
            text="Cora",
            start_ms=300,
            end_ms=390,
            generation=1,
            order=2,
        )
        harness.record_peer_speech_end_for_test(first_receipt.identity.segment_id)
        harness.record_peer_speech_end_for_test(second_receipt.identity.segment_id)
        await harness.peer_owner.handle_provider_turn_terminal(first_receipt, first_terminal)
        async with asyncio.timeout(2):
            while "alpha " not in provider.entered:
                await asyncio.sleep(0)
            await provider.entered["alpha "].wait()
        await harness.peer_owner.handle_provider_turn_terminal(second_receipt, second_terminal)
        await asyncio.wait_for(_wait_for_peer_idle(harness), 2)
        assert provider.cancelled == {"alpha "}
        assert provider.calls == ["alpha ", "beta", "Cora"]
        events = [
            ("source_only", event.source_text) if not event.text else (event.type, event.text)
            for event in overlay.events
            if getattr(event, "type", None) == "translation_final"
        ]
        assert events == [
            ("source_only", "alpha "),
            ("translation_final", "Second"),
            ("translation_final", "Later"),
        ]
    finally:
        provider.release["alpha "].set()
        await harness.stop()
        await presenter.close()


@pytest.mark.asyncio
async def test_multispeaker_final_settles_preview_and_keeps_every_attributed_span() -> None:
    clock = FakeClock(_now=100.0)
    presenter = OverlayPresenter(calibration=OverlayCalibration(), clock=clock)
    provider = _SegmentTranslationProvider()
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
    await harness.start()
    try:
        receipt, terminal = await _terminal_from_soniox_tokens(
            session,
            speaker="A",
            text="",
            start_ms=100,
            end_ms=290,
            generation=1,
            order=1,
            tokens=[
                {
                    "text": text,
                    "speaker": speaker,
                    "language": "en",
                    "is_final": True,
                }
                for text, speaker in (("Alice ", "A"), ("Beto", "B"))
            ],
        )
        await _partial_before_final(harness, replace(terminal, text="Alice "))
        clock.advance(5)
        await harness.peer_owner.cloud_captions.check_previews()
        await _wait_for_peer_idle(harness)
        assert [event.type for event in overlay.events] == ["translation_stream_update"]
        preview = overlay.events[0]
        assert harness.peer_runtime.translation_history == []
        await harness.peer_owner.handle_provider_turn_terminal(receipt, terminal)
        await _wait_for_peer_idle(harness)
        events = [
            event for event in overlay.events if getattr(event, "type", None) == "translation_final"
        ]
        assert provider.calls == ["Alice", "Alice ", "Beto"]
        assert [event.source_text for event in events] == ["Alice ", "Beto"]
        assert events[0].utterance_id == preview.utterance_id == receipt.identity.segment_id
        assert events[0].peer_caption_revision == preview.peer_caption_revision + 1
        assert events[1].utterance_id != events[0].utterance_id
        assert [event.speaker_assignment.palette_index for event in events] == [0, 1]
        assert [event.speaker_assignment.source_text_range for event in events] == [(0, 6), (6, 10)]
        assert [entry.text for entry in harness.peer_runtime.translation_history] == [
            "Alice ",
            "Beto",
        ]
        assert harness.osc.messages == []
    finally:
        await harness.stop()
        await presenter.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("incoming_speaker", ["A", "B", None], ids=["same", "different", "missing"])
async def test_cross_final_merge_retains_honest_attribution_and_cumulative_source(
    incoming_speaker: str | None, monkeypatch
) -> None:
    monkeypatch.setattr(
        "puripuly_heart.core.orchestrator.peer_caption.estimate_peer_translation_lines",
        lambda *_, **__: 2.0,
    )
    clock = FakeClock(_now=100.0)
    presenter = OverlayPresenter(calibration=OverlayCalibration(), clock=clock)
    provider = _SegmentTranslationProvider()
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
    await harness.start()
    try:
        first_receipt, first_terminal = await _terminal_from_soniox_tokens(
            session, speaker="A", text="alpha,", start_ms=100, end_ms=190, generation=1, order=1
        )
        await harness.peer_owner.handle_provider_turn_terminal(first_receipt, first_terminal)
        await _wait_for_peer_idle(harness)
        first = overlay.events[-1]
        assert first.speaker_assignment.attribution.state == "identified"
        first_style = presenter.snapshot().blocks[0].speaker_style
        for order, text, speaker in ((2, "beta", incoming_speaker), (3, "Alice", "A")):
            clock.advance(0.3)
            token = {"text": text, "language": "en", "is_final": True}
            if speaker is not None:
                token["speaker"] = speaker
            receipt, terminal = await _terminal_from_soniox_tokens(
                session,
                speaker="A",
                text=text,
                tokens=[token],
                start_ms=order * 100,
                end_ms=order * 100 + 90,
                generation=1,
                order=order,
            )
            await _partial_before_final(harness, terminal)
            await harness.peer_owner.handle_provider_turn_terminal(receipt, terminal)
            await _wait_for_peer_idle(harness)
            repair = overlay.events[-1]
            assignment = repair.speaker_assignment
            assert repair.utterance_id == first.utterance_id
            assert repair.peer_caption_revision == order
            assert assignment.source_text_range is None
            assert assignment.source_text_revision is None
            if incoming_speaker == "A":
                assert assignment.attribution == first.speaker_assignment.attribution
            else:
                assert assignment.attribution.state == "mixed"
                assert assignment.attribution.key is None
            assert presenter.snapshot().blocks[0].speaker_style == first_style
        assert provider.calls == ["alpha,", "alpha, beta", "alpha, beta Alice"]
        assert [entry.text for entry in harness.peer_runtime.translation_history] == [
            "alpha, beta Alice"
        ]
        assert len(overlay.events[-1].peer_superseded_unit_ids) == 2
        assert first.speaker_assignment.source_text_range == (0, 6)
    finally:
        await harness.stop()
        await presenter.close()
