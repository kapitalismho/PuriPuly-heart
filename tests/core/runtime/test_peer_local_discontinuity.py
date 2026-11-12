from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import pytest

from puripuly_heart.app.wiring import wiring_local_asr_provider_runtime as runtime_wiring
from puripuly_heart.config.overlay_calibration import OverlayCalibration
from puripuly_heart.core.audio.format import AudioCaptureDiscontinuity, AudioFrameF32
from puripuly_heart.core.audio.ownership import PeerAudioSegmentLedger
from puripuly_heart.core.clock import SystemClock
from puripuly_heart.core.local_asr.local_asr_provider_runtime import ProviderRuntimeBuildRequest
from puripuly_heart.core.orchestrator.translation_channel_callbacks import (
    TranslationChannelOwnerCallbacks,
)
from puripuly_heart.core.overlay.peer_caption_layout import peer_translation_fragments
from puripuly_heart.core.overlay.presenter import OverlayPresenter
from puripuly_heart.core.overlay.sink import TranslationFinal, UtteranceClosed
from puripuly_heart.core.peer_capture import PeerCaptureLanguageFacts, PeerCaptureSessionState
from puripuly_heart.core.runtime.audio_vad_loop import run_audio_vad_loop
from puripuly_heart.core.runtime.peer_channel import _CaptureGeneration, _GenerationGuardedVadSink
from puripuly_heart.core.runtime.stt_session_projection import SttSessionStateProjection
from puripuly_heart.core.stt.backend import STTProviderTurnTerminal
from puripuly_heart.core.stt.scoped_engine import ScopedRecognitionEngine
from puripuly_heart.core.vad.gating import SpeechChunk, SpeechEnd, SpeechStart, VadGating
from tests.core.runtime.test_local_asr_provider_runtime import _resolved_config
from tests.core.runtime.test_peer_capture_session import FakeProvider, make_config, make_owner
from tests.core.test_peer_caption_lifecycle import setup
from tests.core.test_peer_overlay_policy import Surface, Timeline
from tests.core.test_stt_scoped_engine import (
    ControlledScopedSession,
    settings,
    span,
    wait_until,
    watchdogs,
)
from tests.helpers.vad import SequenceVadEngine


class RetiringSession(ControlledScopedSession):
    def __init__(self) -> None:
        super().__init__()
        self.abort_entered = asyncio.Event()
        self.abort_release = asyncio.Event()

    async def abort_turn(self, identity, *, reason: str) -> None:
        await super().abort_turn(identity, reason=reason)
        self.abort_entered.set()
        await self.abort_release.wait()
        self.calls.append(("abort_done", identity))
        self.emit(
            STTProviderTurnTerminal(
                identity=identity,
                outcome="final",
                text="Late failed input.",
                text_authority="authoritative",
            )
        )


@pytest.mark.parametrize("provider_id", ["local_granite", "local_qwen_gpu"])
@pytest.mark.parametrize("started", [True, False], ids=["started-write", "wholly-unsent"])
async def test_audio_loop_gap_settles_local_scope_and_recovers_through_display(
    monkeypatch, provider_id, started
):
    old = RetiringSession()
    recovered = ControlledScopedSession()
    recovered.terminal_on_seal = ("final", "Recovered local final.")
    sessions = []
    source_finished = asyncio.Event()
    successor_queued = asyncio.Event()
    guards = []
    observed = []
    terminals = []

    class Backend:
        async def open_session(self, *, projection):
            assert projection.mode == "scoped"
            session = old if started and not sessions else recovered
            sessions.append(session)
            return session

        async def close(self):
            return None

    monkeypatch.setattr(runtime_wiring, "prepare_local_stt_backend_dependencies", lambda _: None)
    monkeypatch.setattr(
        runtime_wiring,
        "create_stt_backend_from_resolved_config",
        lambda *_args, **_kwargs: Backend(),
    )
    config = make_config(
        provider_id=provider_id,
        language=PeerCaptureLanguageFacts(source_mode="manual", source_language="en"),
    )
    factory = runtime_wiring.SharedSTTProviderFactory(
        secrets=object(),
        clock=SystemClock(),
        reset_deadline_s=300,
        gpu_model_path=Path("synthetic"),
    )
    recognition = await factory.create(
        ProviderRuntimeBuildRequest(
            config=_resolved_config("peer", provider_id),
            provider_signature=config.provider_signature,
            runtime_signature=config.runtime_signature,
            recognition_projection="scoped",
        ),
        gpu_runtime=object(),
    )

    class Source:
        terminal_reason = None

        async def frames(self):
            yield AudioFrameF32(
                samples=np.full(512, 0.2, dtype=np.float32),
                sample_rate_hz=16000,
                capture=span(0, 0, 512),
            )
            if started:
                await wait_until(lambda: any(call[0] == "send_done" for call in old.calls))
                old.send_gate.clear()
            yield AudioFrameF32(
                samples=np.full(512, 0.2, dtype=np.float32),
                sample_rate_hz=16000,
                capture=span(1, 512, 1024),
            )
            if started:
                await wait_until(lambda: sum(call[0] == "send" for call in old.calls) == 2)
            yield AudioFrameF32(
                samples=np.full(640, 0.2, dtype=np.float32),
                sample_rate_hz=16000,
                capture=span(2, 1024, 1664),
            )
            yield AudioFrameF32(
                samples=np.full(512, 0.5, dtype=np.float32),
                sample_rate_hz=16000,
                capture=span(4, 2176, 2688),
                discontinuity_before=AudioCaptureDiscontinuity(
                    kind="known_loss",
                    observed_at_monotonic_s=2176 / 16000,
                    lost_source_samples=512,
                ),
            )
            successor_queued.set()
            await source_finished.wait()

        async def close(self):
            source_finished.set()

    async def audio_loop(**kwargs):
        guards.append(kwargs["sink"])
        await run_audio_vad_loop(**kwargs)

    provider = FakeProvider()
    provider.release_gate = asyncio.Event()
    if not started:
        provider.start_gate = asyncio.Event()
    owner, *_ = make_owner(
        provider=provider,
        source_factory=lambda *_: Source(),
        vad_factory=lambda _: VadGating(
            SequenceVadEngine(probs=[0.9] * 4),
            sample_rate_hz=16000,
            chunk_samples=512,
            ring_buffer_ms=1,
            hangover_ms=900,
        ),
        run_audio_loop=audio_loop,
        sink=recognition,
        clock=SystemClock(),
        smart_turn_owner=SimpleNamespace(
            snapshot=SimpleNamespace(availability="unavailable"),
            request_prepare=lambda: None,
            close=lambda: asyncio.sleep(0),
        ),
    )
    timeline, surface = Timeline(), Surface()
    presenter = OverlayPresenter(
        calibration=OverlayCalibration(),
        clock=timeline.clock,
        sleep=timeline.sleep,
        bridge=surface,
        native_retry_enabled=True,
    )
    translation, backend, _, _ = setup(overlay_sink=presenter, clock=timeline.clock)
    monkeypatch.setattr(type(translation.peer_owner), "_current_provider_id", lambda _: provider_id)
    text = "short recovered" if provider_id == "local_granite" else "가" * 56 + "나" * 20
    backend.translations["Recovered local final."] = text
    original_emit = OverlayPresenter.emit

    async def observe(presenter, event):
        observed.append(event)
        await original_emit(presenter, event)

    monkeypatch.setattr(OverlayPresenter, "emit", observe)
    callbacks = TranslationChannelOwnerCallbacks(SttSessionStateProjection())
    callbacks.bind_peer_capture(owner)
    callbacks.bind_peer(translation.peer_owner)

    async def on_event(event):
        if isinstance(event, STTProviderTurnTerminal):
            terminals.append(event)
        await callbacks.peer_event_handler(event)

    recognition.bind_event_sink(on_event)
    owner.bind_publication_generation_observer(
        activated=translation.output_runtime.activate_peer_generation,
        retired=translation.output_runtime.retire_peer_generation,
    )
    start_task = None
    try:
        await translation.start()
        start_task = asyncio.create_task(owner.apply_intent(config, enabled=True))
        await asyncio.wait_for(successor_queued.wait(), 2)
        ledger = owner.segment_ledgers[-1]
        failed = ledger.terminal_receipts[0]
        successor = ledger.snapshots[0]
        assert failed.outcome == "failed"
        assert failed.text_authority == "none"
        assert failed.segment.seal_reason == "source_discontinuity"
        assert failed.segment.content_sample_count == 1536
        assert failed.segment.failed_normalized_sample_count == 128
        assert successor.content_sample_count == 512
        assert successor.context_sample_count == 0
        assert successor.content_ranges[0].source_start_sample == 2176
        assert backend.calls == []
        assert not presenter.snapshot().blocks
        if started:
            assert len(old.requests) == 1
            assert not any(call[0] in {"abort", "stop", "close"} for call in old.calls)
            old.send_gate.set()
            await asyncio.wait_for(old.abort_entered.wait(), 1)
            assert sessions == [old]
            assert not any(call[0] in {"stop", "close"} for call in old.calls)
            source_finished.set()
            old.abort_release.set()
        else:
            assert sessions == []
            provider.start_gate.set()
            source_finished.set()
        await asyncio.wait_for(start_task, 2)
        await wait_until(lambda: len(ledger.terminal_receipts) == 2)
        await recognition.wait_for_event_ingress_drain()
        await translation.translation_turns.wait_for_idle()
        await translation.output_runtime.wait_for_peer_output_idle()
        await timeline.advance(0)
        assert [receipt.outcome for receipt in ledger.terminal_receipts] == ["failed", "final"]
        assert ledger.terminal_receipts[1].segment.seal_reason == "source_eof"
        assert len(recovered.requests) == 1
        assert recovered.requests[0].identity.segment == successor.identity
        assert [(call[0], call[3]) for call in recovered.calls if call[0] == "send"] == [
            ("send", 1024)
        ]
        if started:
            assert sum(call[0] == "send" for call in old.calls) == 2
            order = [call[0] for call in old.calls]
            assert order.index("send_done", order.index("send_done") + 1) < order.index("abort")
            assert order.index("abort_done") < order.index("stop") < order.index("close")
            assert "seal" not in order
            assert [event.outcome for event in terminals] == ["cancelled", "final"]
            await callbacks.peer_event_handler(
                STTProviderTurnTerminal(
                    identity=old.requests[0].identity,
                    outcome="final",
                    text="Late failed input.",
                    text_authority="authoritative",
                )
            )
        else:
            assert old.calls == []
            assert [event.outcome for event in terminals] == ["final"]
        await translation.translation_turns.wait_for_idle()
        await translation.output_runtime.wait_for_peer_output_idle()
        assert [request.text for request in backend.calls] == ["Recovered local final."]
        assert [entry.text for entry in translation.peer_runtime.translation_history] == [
            "Recovered local final."
        ]
        child = backend.calls[0].utterance_id
        assert child != successor.identity.segment_id
        publications = [event for event in observed if event.utterance_id == child]
        assert [type(event) for event in publications] == [TranslationFinal, UtteranceClosed]
        assert publications[0].parent_utterance_id == successor.identity.segment_id
        assert publications[0].peer_caption_revision == 0
        assert publications[1].is_final
        blocks = presenter.snapshot().blocks
        assert [block.primary_text for block in blocks] == list(peer_translation_fragments(text))
        assert presenter.peer_fragment_state(child, 0).id == f"peer:{child}"
        assert presenter.snapshot().native_fresh_render_targets.peer in {
            block.id for block in blocks
        }
        assert translation.peer_owner.cloud_captions.request_counts == dict.fromkeys(
            ("preview", "final", "repair", "continuation"), 0
        )
        assert translation.osc.messages == []
        provider.release_gate.set()
        await wait_until(lambda: owner.snapshot.state is PeerCaptureSessionState.STOPPED)
        assert recognition.is_at_utterance_boundary
        assert recognition.cleanup_debt == 0
        assert guards[0]._queued_pcm_samples == 0
        assert ledger.snapshots == ()
    finally:
        old.send_gate.set()
        old.abort_release.set()
        source_finished.set()
        provider.release_gate.set()
        if provider.start_gate is not None:
            provider.start_gate.set()
        if start_task is not None:
            await asyncio.gather(start_task, return_exceptions=True)
        await owner.close()
        await recognition.close_backend()
        await translation.stop()
        await presenter.close()


@pytest.mark.parametrize("provider_id", ["local_granite", "local_qwen_gpu"])
async def test_consecutive_queued_gaps_do_not_exhaust_provider_recovery(provider_id):
    ledger = PeerAudioSegmentLedger(activation_generation=1, settings=settings(provider_id))
    sessions = []
    emitted = []

    async def open_session(_settings, _epoch):
        session = ControlledScopedSession()
        session.terminal_on_seal = ("final", "Completed after repeated gaps.")
        sessions.append(session)
        return session

    engine = ScopedRecognitionEngine(
        session_factory=open_session,
        event_sink=emitted.append,
        watchdog_resolver=lambda _: watchdogs(write_timeout_s=1),
    )
    ready = asyncio.Event()
    ready.set()
    guard = _GenerationGuardedVadSink(
        sink=engine,
        runtime=SimpleNamespace(
            segment_ledger=ledger, is_current_generation=lambda generation: generation == 1
        ),
        capture_generation=_CaptureGeneration(1),
        provider_ingress_ready=ready,
    )
    try:
        for index in range(4):
            segment_id = uuid4()
            first_sample = index * 2048
            start = ledger.observe_vad_event(
                SpeechStart(
                    segment_id,
                    np.empty(0, dtype=np.float32),
                    np.full(512, 0.2, dtype=np.float32),
                    chunk_capture=(span(index * 3, first_sample, first_sample + 512),),
                ),
                now_monotonic_s=index,
            )
            await guard.handle_owned_vad_event(start)
            await wait_until(lambda: len(sessions) == index + 1)
            session = sessions[-1]
            await wait_until(lambda: any(call[0] == "send_done" for call in session.calls))
            if index < 3:
                session.emit(
                    STTProviderTurnTerminal(
                        identity=session.requests[0].identity,
                        outcome="final",
                        text="Early text from input that will fail.",
                        text_authority="authoritative",
                    )
                )
                await wait_until(lambda: engine._turn.terminal_ready.done())
                session.send_gate.clear()
                chunk = ledger.observe_vad_event(
                    SpeechChunk(
                        segment_id,
                        np.full(512, 0.2, dtype=np.float32),
                        chunk_capture=(
                            span(index * 3 + 1, first_sample + 512, first_sample + 1024),
                        ),
                    ),
                    now_monotonic_s=index + 0.1,
                )
                await guard.handle_owned_vad_event(chunk)
                await wait_until(lambda: sum(call[0] == "send" for call in session.calls) == 2)
            end = ledger.observe_vad_event(
                SpeechEnd(
                    segment_id,
                    reason="source_discontinuity" if index < 3 else "source_eof",
                    trailing_silence_ms=0,
                ),
                now_monotonic_s=index + 0.2,
            )
            await guard.handle_owned_vad_event(end)
            if index < 3:
                ledger.terminalize(segment_id, outcome="failed", now_monotonic_s=index + 0.2)
                session.send_gate.set()
            await wait_until(lambda: len(emitted) == index + 1)
        await asyncio.wait_for(guard.finish(), 1)
        assert [event.outcome for event in emitted] == ["cancelled"] * 3 + ["final"]
        assert [event.text for event in emitted] == [""] * 3 + ["Completed after repeated gaps."]
        assert emitted[-1].identity.segment == start.segment.identity
        assert all(
            sum(call[0] == "abort" for call in session.calls) == 1 for session in sessions[:3]
        )
        assert not any(call[0] == "abort" for call in sessions[-1].calls)
    finally:
        for session in sessions:
            session.send_gate.set()
        await guard.abort()
        await engine.close_backend()
