from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import replace
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import pytest

from puripuly_heart.app.adapters.self_capture.self_capture_provider import (
    SelfCaptureProviderAdapter,
)
from puripuly_heart.app.adapters.self_capture.self_capture_vad_sink import SelfCaptureVadSinkAdapter
from puripuly_heart.app.wiring.wiring_stt_factory import (
    build_self_capture_session_config_from_vnext,
    build_self_stt_provider_request_from_vnext,
)
from puripuly_heart.config.settings_vnext.schema import AppSettingsVNext
from puripuly_heart.core.audio.format import AudioCaptureDiscontinuity, AudioFrameF32
from puripuly_heart.core.orchestrator.self_translation_channel import SelfTranslationChannelOwner
from puripuly_heart.core.orchestrator.translation_channel_callbacks import (
    TranslationChannelOwnerCallbacks,
)
from puripuly_heart.core.runtime.audio_vad_loop import run_audio_vad_loop
from puripuly_heart.core.runtime.local_asr_provider_runtime import LocalASRProviderRuntimeOwner
from puripuly_heart.core.stt.backend import (
    STTContributionConsumptionLedger,
    STTProviderTurnTerminal,
    STTProviderTurnUpdate,
)
from puripuly_heart.core.stt.scoped_normalizer import STTScopedTurnNormalizer
from puripuly_heart.core.vad.gating import SpeechChunk, SpeechEnd, SpeechStart
from puripuly_heart.domain.events import UIEventType
from tests.core.runtime.test_local_asr_provider_runtime import (
    FakeGpuRuntimeFactory,
    FakeProvisioningPort,
)
from tests.core.runtime.test_self_capture_session import LoopHarness, build_owner
from tests.core.test_self_translation_low_latency import FakeLLMProvider
from tests.core.test_stt_scoped_engine import span, wait_until
from tests.helpers.audio import FakeAudioSource
from tests.helpers.fakes import RecordingOscQueue
from tests.helpers.translation_owners import compose_translation_test_harness
from tests.providers.test_qwen_audio import ScriptedSocket, control, native, request
from tests.providers.test_qwen_audio_recovery_boundaries import (
    FaultTransport,
    RecoveryFactory,
    capture_runtime,
    finish,
    finish_sent,
    terminals,
    utterance,
)


class BoundRuntimeFactory:
    def __init__(self, provider_factory):
        self.provider_factory = provider_factory
        self.events = []

    def create(self, callbacks):
        self.callbacks = callbacks.self_event_handler.__self__

        async def receive(event):
            self.events.append(event)
            await callbacks.self_event_handler(event)

        self.runtime = LocalASRProviderRuntimeOwner(
            provider_factory=self.provider_factory,
            gpu_runtime_factory=FakeGpuRuntimeFactory(),
            provisioning=FakeProvisioningPort(),
            self_event_handler=receive,
            peer_event_handler=self.callbacks.peer_event_handler,
            retired_event_handler=callbacks.retired_event_handler,
        )
        return self.runtime


class HeldManualProvider(FakeLLMProvider):
    def __init__(self):
        super().__init__(delay_s=0)
        self.manual_entered = asyncio.Event()
        self.manual_release = asyncio.Event()

    async def translate(self, **kwargs):
        if kwargs["text"] == "Manual independent marker.":
            self.manual_entered.set()
            await self.manual_release.wait()
        return await super().translate(**kwargs)


@asynccontextmanager
async def self_capture_pipeline(
    transport, llm=None, *, provider_factory=None, peer_translation_enabled=False, settings=None
):
    if settings is None:
        base = AppSettingsVNext()
        settings = replace(
            base,
            intent=replace(
                base.intent,
                stt=replace(base.intent.stt, provider="qwen_audio"),
                languages=replace(
                    base.intent.languages, source_language="en", target_language="ko"
                ),
            ),
        )
    factory = provider_factory or RecoveryFactory(transport)
    settings_box = [settings]
    bound = BoundRuntimeFactory(factory)
    llm = llm or FakeLLMProvider(delay_s=0)
    harness = compose_translation_test_harness(
        stt=None,
        llm=llm,
        osc=RecordingOscQueue(),
        low_latency_mode=True,
        source_language="en",
        target_language="ko",
        peer_translation_enabled=peer_translation_enabled,
        local_asr_provider_runtime_factory=bound,
    )
    sink = SelfCaptureVadSinkAdapter(runtime_provider=lambda: harness.self_owner)
    loop = LoopHarness()
    adapter = SelfCaptureProviderAdapter(bound.runtime, harness)
    owner, *_ = build_owner(provider=adapter, sink=sink, loop=loop)
    owner._provider_request_factory = (
        lambda _config, warmup: build_self_stt_provider_request_from_vnext(
            settings_box[0], warmup=warmup
        )
    )
    bound.callbacks.bind_self_capture(owner)
    env = SimpleNamespace(
        channel="self",
        owner=owner,
        factory=factory,
        transport=transport,
        harness=harness,
        llm=llm,
        events=bound.events,
        callbacks=bound.callbacks,
        settings_box=settings_box,
        cursors={},
        sequence=0,
    )
    try:
        await harness.start()
        await owner.apply_intent(
            build_self_capture_session_config_from_vnext(settings), enabled=True
        )
        await wait_until(lambda: bool(loop.calls))
        env.guarded = loop.calls[-1]["sink"]
        yield env
    finally:
        if isinstance(llm, HeldManualProvider):
            llm.manual_release.set()
        await owner.close()
        await harness.stop()


async def spoken_turn(env, value=0.1, epoch=1):
    identity = uuid4()
    for index in range(4):
        cursor = env.cursors.get(epoch, 0)
        now = asyncio.get_running_loop().time()
        capture = replace(
            span(env.sequence, cursor, cursor + 160),
            capture_epoch=epoch,
            source_start_monotonic_s=now - 0.01,
            source_end_monotonic_s=now,
        )
        env.sequence += 1
        env.cursors[epoch] = cursor + 160
        pcm = np.full(160, value, dtype=np.float32)
        event = (
            SpeechStart(identity, np.empty(0, dtype=np.float32), pcm, chunk_capture=(capture,))
            if index == 0
            else SpeechChunk(identity, pcm, chunk_capture=(capture,))
        )
        await env.guarded.handle_vad_event(event)
    await env.guarded.handle_vad_event(
        SpeechEnd(identity, trailing_silence_ms=224, reason="silence")
    )
    return identity


async def source_gap(env):
    class NoOnsetVad:
        chunk_samples = 8

        def process_chunk(self, _chunk):
            return []

    now = asyncio.get_running_loop().time()
    frames = [
        AudioFrameF32(
            np.full(8, 0.4, dtype=np.float32),
            16000,
            capture=replace(
                span(index, 0, 8),
                capture_epoch=index + 1,
                source_start_monotonic_s=now,
                source_end_monotonic_s=now + 8 / 16000,
                discontinuity_before=(
                    AudioCaptureDiscontinuity("known_loss", now, 8) if index else None
                ),
            ),
        )
        for index in range(2)
    ]
    await asyncio.wait_for(
        run_audio_vad_loop(
            source=FakeAudioSource(frames),
            vad=NoOnsetVad(),
            sink=env.guarded,
            target_sample_rate_hz=16000,
        ),
        timeout=1,
    )


def final_sources(env):
    return [
        event.payload
        for event in env.harness.ui_events._queue
        if event.type is UIEventType.TRANSCRIPT_FINAL
    ]


async def require_final_source(env, text):
    await wait_until(lambda: any(item.text == text for item in final_sources(env)), timeout=2)
    await env.harness.translation_turns.wait_for_idle()
    assert sum(item.text == text for item in final_sources(env)) == 1
    assert any(call["text"] == text for call in env.llm.calls)
    return next(item for item in final_sources(env) if item.text == text)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["connect", "task_started"])
async def test_self_initial_native_admission_uses_short_retries_then_real_consumer_progress(
    failure,
):
    failed = (
        [TimeoutError(), TimeoutError()]
        if failure == "connect"
        else [ScriptedSocket(ready=False), ScriptedSocket(ready=False)]
    )
    transport = FaultTransport([*failed, ScriptedSocket()])
    async with self_capture_pipeline(transport) as env:
        assert transport.attempts == 0
        await spoken_turn(env)
        await wait_until(lambda: transport.attempts == 3 and finish_sent(transport.sockets[-1]))
        await finish(transport.sockets[-1], "Self initial recovered.")
        transcript = await require_final_source(env, "Self initial recovered.")
        assert transcript.recognition_origins
        assert env.factory.delays == [0.8, 1.6]
        assert env.factory.engines[0]._connection_retry_at_s is None
        assert env.owner.snapshot.effective_active
        assert [event.outcome for event in terminals(env)] == ["final"]
        assert (
            sum(
                len(value)
                for socket in transport.sockets
                for value in socket.sent
                if isinstance(value, bytes)
            )
            == 1280
        )
        if failure == "task_started":
            assert all(socket.closed for socket in failed)
            assert len({socket.task_ids[0] for socket in transport.sockets}) == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["connect", "task_started"])
async def test_self_initial_exhaustion_is_three_attempts_and_no_peer_cooldown(failure):
    failed = (
        [TimeoutError() for _ in range(3)]
        if failure == "connect"
        else [ScriptedSocket(ready=False) for _ in range(3)]
    )
    transport = FaultTransport(failed)
    async with self_capture_pipeline(transport) as env:
        await spoken_turn(env)
        await wait_until(lambda: not env.owner.snapshot.desired_active)
        assert transport.attempts == 3
        assert env.factory.delays == [0.8, 1.6]
        assert env.factory.engines[0]._connection_retry_at_s is None
        assert not env.owner.snapshot.effective_active
        assert len(terminals(env)) == 1
        assert terminals(env)[0].outcome == "failed"
        assert not terminals(env)[0].recovery_pending
        assert not final_sources(env)
        assert not any(
            isinstance(value, bytes) for socket in transport.sockets for value in socket.sent
        )


@pytest.mark.asyncio
async def test_self_later_successor_failure_keeps_nonrecovery_policy_without_new_short_retries():
    first = ScriptedSocket()
    unused = ScriptedSocket()
    transport = FaultTransport([first, TimeoutError(), unused])
    async with self_capture_pipeline(transport) as env:
        await spoken_turn(env)
        await wait_until(lambda: finish_sent(first))
        await finish(first, "Self admitted initial.")
        await require_final_source(env, "Self admitted initial.")
        await spoken_turn(env, 0.2)
        await wait_until(lambda: not env.owner.snapshot.desired_active)
        assert transport.attempts == 2
        assert env.factory.delays == []
        assert transport.outcomes == [unused]
        assert not terminals(env)[-1].recovery_pending
        assert env.factory.engines[0]._connection_retry_at_s is None
        assert not env.owner.snapshot.effective_active


@pytest.mark.asyncio
async def test_genuinely_new_self_provider_scope_gets_initial_retry_budget_without_cooldown():
    first, second = ScriptedSocket(), ScriptedSocket()
    transport = FaultTransport([first, TimeoutError(), second])
    async with self_capture_pipeline(transport) as env:
        await spoken_turn(env)
        await wait_until(lambda: finish_sent(first))
        await finish(first, "First scope completed.")
        await require_final_source(env, "First scope completed.")
        old_epoch = terminals(env)[0].identity.provider_epoch_id
        await source_gap(env)
        await spoken_turn(env, 0.3, epoch=2)
        await wait_until(lambda: finish_sent(second))
        await finish(second, "Fresh scope recovered.")
        await require_final_source(env, "Fresh scope recovered.")
        assert terminals(env)[-1].identity.provider_epoch_id != old_epoch
        assert env.factory.delays == [0.8]
        assert transport.attempts == 3
        assert env.factory.engines[0]._connection_retry_at_s is None
        assert env.owner.snapshot.effective_active


class UnclosedSocket(ScriptedSocket):
    async def close(self):
        raise TimeoutError("controlled attempt close timeout")


@pytest.mark.asyncio
async def test_failed_attempt_close_quarantines_socket_and_preserves_predecessor_before_safe_recovery():
    first, unsafe = ScriptedSocket(), UnclosedSocket(ready=False)
    transport = FaultTransport([first, unsafe])
    async with capture_runtime("peer", transport) as env:
        try:
            await utterance(env)
            await wait_until(lambda: finish_sent(first))
            await utterance(env, 0.2)
            engine = env.factory.engines[0]
            await wait_until(lambda: engine._quarantined_session is not None)
            assert transport.attempts == 2
            assert len(unsafe.task_ids) == 1
            assert env.factory.delays == []
            assert not first.closed and not unsafe.closed
            assert engine.cleanup_debt > 0
            await finish(first, "Healthy predecessor retained.")
            await wait_until(lambda: len(terminals(env)) == 2)
            assert terminals(env)[0].text == "Healthy predecessor retained."
            assert terminals(env)[1].recovery_pending
            assert env.owner.snapshot.effective_active
            await wait_until(lambda: not any(not task.done() for task in engine._cleanup_tasks))
            assert engine.cleanup_debt > 0
            unsafe.close = ScriptedSocket.close.__get__(unsafe, type(unsafe))

            async def require_closed_before_connect(attempt):
                if attempt == 3:
                    assert unsafe.closed and first.closed

            transport.on_attempt = require_closed_before_connect
            await env.factory.clock.advance_to(30)
            await utterance(env, 0.3)
            await wait_until(
                lambda: len(transport.sockets) == 3 and finish_sent(transport.sockets[-1])
            )
            await finish(transport.sockets[-1], "Recovered after owned retirement.")
            await wait_until(lambda: len(terminals(env)) == 3)
            assert terminals(env)[-1].text == "Recovered after owned retirement."
            assert engine.cleanup_debt == 0
            assert len(unsafe.task_ids) == 1
            assert transport.attempts == 3
            assert not any(isinstance(value, bytes) for value in unsafe.sent)
        finally:
            unsafe.close = ScriptedSocket.close.__get__(unsafe, type(unsafe))


@pytest.mark.asyncio
@pytest.mark.parametrize("channel", ["self", "peer"])
async def test_failed_close_uses_existing_transport_abort_before_any_retry(channel):
    unsafe = UnclosedSocket(ready=False)
    closed = asyncio.Event()
    aborts = []

    def abort():
        aborts.append(True)
        unsafe.closed = True
        closed.set()

    unsafe.transport = SimpleNamespace(abort=abort)
    unsafe.wait_closed = closed.wait
    transport = FaultTransport([unsafe, ScriptedSocket()])
    async with capture_runtime(channel, transport) as env:
        await utterance(env)
        await wait_until(lambda: len(transport.sockets) == 2 and finish_sent(transport.sockets[-1]))
        await finish(transport.sockets[-1], "Safely retired attempt recovered.")
        await wait_until(lambda: bool(terminals(env)))
        assert terminals(env)[0].outcome == "final"
        assert terminals(env)[0].text == "Safely retired attempt recovered."
        assert aborts == [True]
        assert len(unsafe.task_ids) == 1
        assert unsafe.task_ids[0] != transport.sockets[-1].task_ids[0]
        assert env.factory.delays == [0.8]
        assert not any(isinstance(value, bytes) for value in unsafe.sent)


@pytest.mark.parametrize("outcome", ["cancelled", "suppressed", "expired"])
@pytest.mark.parametrize("already_consumed", [False, True])
def test_empty_terminal_removes_text_ranges_without_weakening_consumption(
    outcome, already_consumed
):
    identity = request("self").identity
    normalizer = STTScopedTurnNormalizer(identity, preserve_text=True)
    ledger = STTContributionConsumptionLedger()
    stable = normalizer.apply_update(
        STTProviderTurnUpdate(
            identity,
            1,
            "stable",
            "replace",
            " exact stable ",
        )
    )
    assert stable.contribution.text_end == len(" exact stable ")
    if already_consumed:
        assert ledger.consume(stable) == " exact stable "
    terminal = normalizer.apply_terminal(STTProviderTurnTerminal(identity, outcome))
    assert terminal.text == ""
    assert terminal.included_contributions == ()
    assert terminal.recognition_origins == ()
    assert terminal.final_language_runs == () and terminal.final_speaker_runs == ()
    assert ledger.consume_with_origins(terminal) == ("", ())
    ledger.retire(identity)
    malformed = replace(terminal, included_contributions=(stable.contribution,))
    with pytest.raises(ValueError, match="invalid assembled contribution range"):
        ledger.consume(malformed)


@pytest.mark.asyncio
async def test_real_self_vad_gap_settles_admitted_stable_and_fresh_tail_reaches_source_and_translation():
    llm = HeldManualProvider()
    transport = FaultTransport()
    async with self_capture_pipeline(transport, llm) as env:
        first_id = await spoken_turn(env)
        await wait_until(lambda: len(transport.sockets) == 1 and finish_sent(transport.sockets[0]))
        first = transport.sockets[0]
        await first.push(native(first.task_ids[0], text="Already admitted A."))
        await wait_until(lambda: env.harness.self_owner.merge_buffer is not None)
        assert env.harness.self_owner.merge_buffer.parts == ["Already admitted A."]
        assert first_id in env.harness.self_owner.runtime.speech_ended_ids
        second_id = await spoken_turn(env, 0.2)
        await wait_until(lambda: len(transport.sockets) == 2 and finish_sent(transport.sockets[1]))
        second = transport.sockets[1]
        await second.push(native(second.task_ids[0], text="UNADMITTED_B"))
        engine = env.factory.engines[0]
        await wait_until(
            lambda: any(turn.pending_update is not None for turn in engine._turns.values())
        )
        await spoken_turn(env, 0.25)
        manual = await env.harness.self_owner.submit_text("Manual independent marker.")
        await asyncio.wait_for(llm.manual_entered.wait(), timeout=1)
        await source_gap(env)
        await wait_until(lambda: all(socket.closed for socket in transport.sockets))
        assert not engine._event_dispatch_task.done()
        await wait_until(lambda: len(terminals(env)) >= 2)
        cancelled = [event for event in terminals(env) if event.outcome == "cancelled"]
        assert {event.identity.segment.segment_id for event in cancelled} == {first_id, second_id}
        assert all(
            not event.included_contributions and not event.recognition_origins
            for event in cancelled
        )
        assert env.guarded.retention_budget.used_bytes == 0
        llm.manual_release.set()
        fresh_id = await spoken_turn(env, 0.3, epoch=2)
        await wait_until(lambda: len(transport.sockets) == 3 and finish_sent(transport.sockets[-1]))
        fresh = transport.sockets[-1]
        await fresh.push(native(fresh.task_ids[0], 1, " Fresh C."))
        await fresh.push(native(fresh.task_ids[0], 2, " Exact tail."))
        await fresh.push(control(fresh.task_ids[0], "task-finished"))
        expected = "Already admitted A. Fresh C. Exact tail."
        result = await require_final_source(env, expected)
        assert len(result.recognition_origins) == 3
        assert {origin.identity.stream.capture_epoch for origin in result.recognition_origins} == {
            1,
            2,
        }
        assert any(
            event.type is UIEventType.TRANSLATION_DONE and event.utterance_id == manual.turn_id
            for event in env.harness.ui_events._queue
        )
        assert sum(call["text"] == "Manual independent marker." for call in llm.calls) == 1
        assert not any("UNADMITTED_B" in call["text"] for call in llm.calls)
        assert all("UNADMITTED_B" not in item.text for item in final_sources(env))
        assert not engine._event_dispatch_task.done()
        assert engine._event_buffer.depth == 0
        assert env.owner.snapshot.effective_active
        assert any(
            event.identity.segment.segment_id == fresh_id and event.outcome == "final"
            for event in terminals(env)
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("held_boundary", ["provider", "callback", "consumer", "overflow"])
async def test_deferred_pre_gap_update_is_rejected_at_real_consumption_and_fresh_source_progresses(
    monkeypatch, held_boundary
):
    transport = FaultTransport()
    async with self_capture_pipeline(transport) as env:
        await spoken_turn(env)
        await wait_until(lambda: len(transport.sockets) == 1 and finish_sent(transport.sockets[0]))
        old = transport.sockets[0]
        engine = env.factory.engines[0]
        entered, release = asyncio.Event(), asyncio.Event()
        if held_boundary in ("provider", "overflow"):
            original = engine._deferred_event_sink
            if held_boundary == "overflow":
                original_capacity = engine._event_buffer.max_events
                engine._event_buffer._max_events = 1

            async def hold(event):
                if isinstance(event, STTProviderTurnUpdate):
                    entered.set()
                    await release.wait()
                await original(event)

            engine.bind_event_sink(hold)
        elif held_boundary == "callback":
            original = TranslationChannelOwnerCallbacks._publish_self_session_state

            async def hold(self, state):
                if self is env.callbacks and not entered.is_set():
                    entered.set()
                    await release.wait()
                await original(self, state)

            monkeypatch.setattr(
                TranslationChannelOwnerCallbacks, "_publish_self_session_state", hold
            )
        else:
            original = SelfTranslationChannelOwner.handle_stt_event

            async def hold(self, event):
                if self is env.harness.self_owner and isinstance(event, STTProviderTurnUpdate):
                    entered.set()
                    await release.wait()
                await original(self, event)

            monkeypatch.setattr(SelfTranslationChannelOwner, "handle_stt_event", hold)
        try:
            await old.push(native(old.task_ids[0], text="PRE_GAP_NOT_ADMITTED"))
            await asyncio.wait_for(entered.wait(), timeout=1)
            assert env.harness.self_owner.merge_buffer is None
            if held_boundary == "overflow":
                await old.push(native(old.task_ids[0], 2, " extra one."))
                await old.push(native(old.task_ids[0], 3, " extra two."))
                await wait_until(lambda: bool(engine._event_buffer._retired_epochs))
            before = engine._authority_generation
            await source_gap(env)
            assert engine._authority_generation > before
        finally:
            release.set()
        await engine.wait_for_event_ingress_drain()
        assert env.harness.self_owner.merge_buffer is None
        assert not any("PRE_GAP_NOT_ADMITTED" in call["text"] for call in env.llm.calls)
        assert env.owner.snapshot.effective_active
        if held_boundary == "overflow":
            engine._event_buffer._max_events = original_capacity
        await spoken_turn(env, 0.3, epoch=2)
        await wait_until(lambda: len(transport.sockets) == 2 and finish_sent(transport.sockets[-1]))
        await finish(transport.sockets[-1], "Only fresh admitted source.")
        result = await require_final_source(env, "Only fresh admitted source.")
        assert result.recognition_origins
        assert not any("PRE_GAP_NOT_ADMITTED" in item.text for item in final_sources(env))
        assert not engine._event_dispatch_task.done()
        assert env.owner.snapshot.effective_active


@pytest.mark.asyncio
async def test_source_gap_does_not_cancel_work_already_admitted_by_consumption(monkeypatch):
    transport = FaultTransport()
    async with self_capture_pipeline(transport) as env:
        await spoken_turn(env)
        await wait_until(lambda: len(transport.sockets) == 1 and finish_sent(transport.sockets[0]))
        old = transport.sockets[0]
        engine = env.factory.engines[0]
        entered, release = asyncio.Event(), asyncio.Event()
        original = SelfTranslationChannelOwner._apply_scoped_contribution

        async def hold(self, identity, text, origins=()):
            if self is env.harness.self_owner and text == "Consumed before gap.":
                entered.set()
                await release.wait()
            await original(self, identity, text, origins)

        monkeypatch.setattr(SelfTranslationChannelOwner, "_apply_scoped_contribution", hold)
        try:
            await old.push(native(old.task_ids[0], text="Consumed before gap."))
            await asyncio.wait_for(entered.wait(), timeout=1)
            assert env.harness.self_owner._stt_consumption.consumed_contribution_ids
            assert env.harness.self_owner.merge_buffer is None
            await source_gap(env)
            assert not engine._event_dispatch_task.cancelling()
        finally:
            release.set()
        await engine.wait_for_event_ingress_drain()
        assert env.harness.self_owner.merge_buffer.parts == ["Consumed before gap."]
        await spoken_turn(env, 0.3, epoch=2)
        await wait_until(lambda: len(transport.sockets) == 2 and finish_sent(transport.sockets[-1]))
        await finish(transport.sockets[-1], " Fresh after admitted work.")
        result = await require_final_source(env, "Consumed before gap. Fresh after admitted work.")
        assert len(result.recognition_origins) == 2
        assert not engine._event_dispatch_task.done()
        assert env.owner.snapshot.effective_active


@pytest.mark.asyncio
@pytest.mark.parametrize("retirement", ["close", "idle"])
async def test_orderly_retirement_preserves_deferred_terminal_and_consumed_source(retirement):
    transport = FaultTransport()
    async with self_capture_pipeline(transport) as env:
        await spoken_turn(env)
        await wait_until(lambda: len(transport.sockets) == 1 and finish_sent(transport.sockets[0]))
        socket = transport.sockets[0]
        engine = env.factory.engines[0]
        entered, release = asyncio.Event(), asyncio.Event()
        original = engine._deferred_event_sink

        async def hold(event):
            if isinstance(event, STTProviderTurnTerminal):
                entered.set()
                await release.wait()
            await original(event)

        engine.bind_event_sink(hold)
        closing = None
        try:
            await finish(socket, "Orderly source conserved.")
            await asyncio.wait_for(entered.wait(), timeout=1)
            if retirement == "close":
                closing = asyncio.create_task(engine.close())
                await wait_until(lambda: engine._closing)
                assert not closing.done()
            else:
                await env.factory.clock.advance_to(60)
                await wait_until(lambda: socket.closed)
        finally:
            release.set()
            if closing is not None:
                await closing
        await require_final_source(env, "Orderly source conserved.")
        assert [event.outcome for event in terminals(env)] == ["final"]
        assert env.owner.snapshot.desired_active
        assert transport.attempts == 1
