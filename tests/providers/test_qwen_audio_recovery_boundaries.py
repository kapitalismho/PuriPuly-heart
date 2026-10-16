from __future__ import annotations

import asyncio
import json
import ssl
from contextlib import asynccontextmanager
from dataclasses import replace
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import pytest

from puripuly_heart.app.adapters.peer_capture.peer_capture_provider import (
    PeerCaptureProviderAdapter,
)
from puripuly_heart.app.adapters.self_capture.self_capture_provider import (
    SelfCaptureProviderAdapter,
)
from puripuly_heart.app.wiring.wiring_stt_factory import (
    build_peer_capture_session_config_from_vnext,
    build_peer_stt_provider_request,
    build_self_capture_session_config_from_vnext,
    build_self_stt_provider_request_from_vnext,
)
from puripuly_heart.config.settings_vnext.schema import AppSettingsVNext
from puripuly_heart.core.audio.format import AudioCaptureDiscontinuity, AudioFrameF32
from puripuly_heart.core.audio.ownership import CaptureStreamInput
from puripuly_heart.core.orchestrator.translation_channel_callbacks import (
    TranslationChannelOwnerCallbacks,
)
from puripuly_heart.core.runtime.audio_vad_loop import run_audio_vad_loop
from puripuly_heart.core.runtime.local_asr_provider_runtime import LocalASRProviderRuntimeOwner
from puripuly_heart.core.stt.backend import (
    STTProviderTurnTerminal,
    STTProviderTurnUpdate,
    STTSessionProjection,
)
from puripuly_heart.core.stt.scoped_engine import ScopedRecognitionEngine
from puripuly_heart.core.vad.gating import SpeechEnd, SpeechStart
from puripuly_heart.domain.events import UIEventType
from puripuly_heart.providers.stt.qwen_audio import QwenAudioStreamingSTTBackend
from tests.core.runtime.test_local_asr_provider_runtime import (
    FakeGpuRuntimeFactory,
    FakeProvisioningPort,
)
from tests.core.runtime.test_peer_capture_session import make_owner
from tests.core.runtime.test_self_capture_session import LoopHarness, build_owner
from tests.core.test_stt_scoped_engine import ControlledMonotonicClock, span, wait_until, watchdogs
from tests.helpers.audio import FakeAudioSource
from tests.helpers.fakes import RecordingOscQueue
from tests.helpers.translation_owners import compose_translation_test_harness
from tests.providers.test_qwen_audio import ScriptedSocket, control, native
from tests.providers.test_qwen_audio_configuration import ControlledSmartTurn, RuntimeSink


class StatusError(Exception):
    def __init__(self, status):
        self.status_code = status
        super().__init__(f"controlled upgrade {status}")


class FaultTransport:
    def __init__(self, outcomes=()):
        self.outcomes = list(outcomes)
        self.attempts = 0
        self.sockets = []
        self.on_attempt = None

    async def __call__(self, *_args, **_kwargs):
        self.attempts += 1
        if self.on_attempt is not None:
            await self.on_attempt(self.attempts)
        outcome = self.outcomes.pop(0) if self.outcomes else ScriptedSocket()
        if isinstance(outcome, Exception):
            raise outcome
        self.sockets.append(outcome)
        return outcome


class BoundaryRuntimeSink(RuntimeSink):
    def __init__(self, runtime, channel):
        super().__init__(runtime, channel)
        self.boundaries = []
        self.cancelled_dispatch = asyncio.Event()
        self.cancelled_dispatch_gate = None

    async def handle_vad_event(self, event):
        await self.handle_owned_vad_event(event)

    async def handle_owned_vad_event(self, event):
        try:
            await super().handle_owned_vad_event(event)
        except asyncio.CancelledError:
            self.cancelled_dispatch.set()
            if self.cancelled_dispatch_gate is not None:
                await self.cancelled_dispatch_gate.wait()
            raise

    async def handle_stream_input(self, event):
        self.boundaries.append(event.event)
        await self.runtime.handle_stream_input(self.channel, event)


class RecoveryFactory:
    def __init__(self, transport, *, task_start_timeout_s=0.03):
        self.transport = transport
        self.task_start_timeout_s = task_start_timeout_s
        self.clock = ControlledMonotonicClock()
        self.engines = []
        self.delays = []

    async def sleep(self, delay):
        if delay >= 30:
            await self.clock.sleep(delay)
        else:
            self.delays.append(delay)
            await asyncio.sleep(0)

    async def create(self, request, *, gpu_runtime, on_terminal_failure=None):
        backend = QwenAudioStreamingSTTBackend(
            api_key="offline",
            channel=request.channel,
            language_hints=("en",),
            websocket_factory=self.transport,
            task_start_timeout_s=self.task_start_timeout_s,
            task_finish_timeout_s=2,
        )
        engine = ScopedRecognitionEngine(
            channel=request.channel,
            session_factory=lambda _settings, epoch: backend.open_session(
                projection=STTSessionProjection("scoped", epoch)
            ),
            watchdog_resolver=lambda _: watchdogs(
                readiness_timeout_s=30,
                write_timeout_s=0.2,
                final_timeout_s=2,
                connect_retry_base_s=0.8,
                connect_retry_max_s=1.6,
                connect_recovery_cooldown_s=30 if request.channel == "peer" else None,
            ),
            accepted_settings_scope=(
                request.provider_id,
                request.provider_signature,
                request.runtime_signature,
            ),
            terminal_failure_sink=on_terminal_failure,
            monotonic_clock=self.clock.now,
            sleep=self.sleep,
        )
        self.engines.append(engine)
        return engine


@asynccontextmanager
async def capture_runtime(channel, transport):
    base = AppSettingsVNext()
    settings = replace(
        base,
        intent=replace(
            base.intent,
            stt=replace(base.intent.stt, provider="qwen_audio"),
            peer_stt=replace(base.intent.peer_stt, provider="qwen_audio"),
            languages=replace(
                base.intent.languages, source_language="en", peer_source_language="en"
            ),
        ),
    )
    harness = compose_translation_test_harness(
        stt=None,
        llm=None,
        osc=RecordingOscQueue(),
        low_latency_mode=True,
    )
    callbacks = TranslationChannelOwnerCallbacks(harness.stt_sessions)
    callbacks.bind_self(harness.self_owner)
    callbacks.bind_peer(harness.peer_owner)
    factory = RecoveryFactory(transport)
    events = []

    async def receive(event):
        events.append(event)
        if channel == "peer":
            await callbacks.peer_event_handler(event)
        else:
            await callbacks.self_event_handler(event)

    runtime = LocalASRProviderRuntimeOwner(
        provider_factory=factory,
        gpu_runtime_factory=FakeGpuRuntimeFactory(),
        provisioning=FakeProvisioningPort(),
        self_event_handler=receive,
        peer_event_handler=receive,
        retired_event_handler=receive,
    )
    loop = LoopHarness()
    sink = BoundaryRuntimeSink(runtime, channel)
    if channel == "self":
        adapter = SelfCaptureProviderAdapter(runtime, harness)
        owner, *_ = build_owner(provider=adapter, sink=sink, loop=loop)
        owner._provider_request_factory = (
            lambda _config, warmup: build_self_stt_provider_request_from_vnext(
                settings, warmup=warmup
            )
        )
        config = build_self_capture_session_config_from_vnext(settings)
        callbacks.bind_self_capture(owner)
    else:
        adapter = PeerCaptureProviderAdapter(runtime, harness)
        owner, *_ = make_owner(
            provider=adapter,
            sink=sink,
            run_audio_loop=loop.run,
            smart_turn_owner=ControlledSmartTurn(),
        )
        owner._provider_request_factory = lambda config, warmup: build_peer_stt_provider_request(
            config, gpu_device_id="auto", warmup=warmup
        )
        config = build_peer_capture_session_config_from_vnext(settings)
        callbacks.bind_peer_capture(owner)
    env = SimpleNamespace(
        channel=channel,
        owner=owner,
        factory=factory,
        transport=transport,
        events=events,
        sink=sink,
        loop=loop,
        harness=harness,
    )
    try:
        await harness.start()
        await runtime.start()
        harness.output_runtime.activate_peer_generation(1)
        await owner.apply_intent(config, enabled=True)
        await wait_until(lambda: bool(loop.calls))
        env.guarded = loop.calls[-1]["sink"]
        yield env
    finally:
        await owner.close()
        await runtime.close()
        await harness.stop()


async def utterance(env, value=0.1, epoch=1):
    identity = uuid4()
    now = asyncio.get_running_loop().time()
    capture = replace(
        span(1, 0, 160),
        capture_epoch=epoch,
        source_start_monotonic_s=now,
        source_end_monotonic_s=now + 0.01,
    )
    start = SpeechStart(
        identity,
        np.empty(0, dtype=np.float32),
        np.full(160, value, dtype=np.float32),
        chunk_capture=(capture,),
    )
    end = SpeechEnd(identity, trailing_silence_ms=224, reason="silence")
    if env.channel == "self":
        await env.guarded.handle_vad_event(start)
        await env.guarded.handle_vad_event(end)
    else:
        ledger = env.owner.segment_ledger
        await env.guarded.handle_owned_vad_event(
            ledger.observe_vad_event(start, now_monotonic_s=now)
        )
        await env.guarded.handle_owned_vad_event(
            ledger.observe_vad_event(end, now_monotonic_s=now + 0.01)
        )
    return identity


def terminals(env):
    return [event for event in env.events if isinstance(event, STTProviderTurnTerminal)]


def finish_sent(socket):
    return any(
        isinstance(value, str) and json.loads(value)["header"]["action"] == "finish-task"
        for value in socket.sent
    )


async def finish(socket, text):
    await wait_until(lambda: finish_sent(socket))
    await socket.push(native(socket.task_ids[-1], text=text))
    await socket.push(control(socket.task_ids[-1], "task-finished"))


async def assert_consumer_sources(env, expected):
    if env.channel == "self":
        await wait_until(lambda: env.harness.self_owner.merge_buffer is not None)
        assert env.harness.self_owner.merge_buffer.parts == expected
    else:
        await wait_until(
            lambda: sum(
                event.type is UIEventType.TRANSCRIPT_FINAL for event in env.harness.ui_events._queue
            )
            >= len(expected)
        )
        assert [
            event.payload.text
            for event in env.harness.ui_events._queue
            if event.type is UIEventType.TRANSCRIPT_FINAL
        ] == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure", [TimeoutError, ConnectionError, lambda: StatusError(503), lambda: StatusError(429)]
)
async def test_actual_peer_readiness_budget_cooldown_keeps_capture_and_later_source_recovers(
    failure,
):
    transport = FaultTransport([failure(), failure(), failure()])
    async with capture_runtime("peer", transport) as env:
        assert transport.attempts == 0
        await utterance(env, 0.1)
        await wait_until(lambda: len(terminals(env)) == 1)
        assert [(event.outcome, event.recovery_pending) for event in terminals(env)] == [
            ("failed", True)
        ]
        assert transport.attempts == 3
        assert env.factory.delays == [0.8, 1.6]
        assert env.owner.snapshot.effective_active
        await env.factory.clock.advance_to(29.999)
        await utterance(env, 0.2)
        await wait_until(lambda: len(terminals(env)) == 2)
        assert transport.attempts == 3
        assert all(event.recovery_pending for event in terminals(env))
        await env.factory.clock.advance_to(30)
        await asyncio.sleep(0.01)
        assert transport.attempts == 3
        await utterance(env, 0.3)
        await wait_until(lambda: bool(transport.sockets) and finish_sent(transport.sockets[0]))
        socket = transport.sockets[0]
        assert len([value for value in socket.sent if isinstance(value, bytes)]) == 1
        assert sum(len(value) for value in socket.sent if isinstance(value, bytes)) == 320
        assert (
            np.frombuffer(
                next(value for value in socket.sent if isinstance(value, bytes)), dtype="<i2"
            ).tolist()
            == [9830] * 160
        )
        await finish(socket, "recovered fresh speech")
        await wait_until(lambda: len(terminals(env)) == 3)
        assert [(event.outcome, event.text) for event in terminals(env)][-1] == (
            "final",
            "recovered fresh speech",
        )
        assert transport.attempts == 4
        assert env.owner.snapshot.effective_active
        assert env.factory.engines[0]._connection_retry_at_s is None
        await assert_consumer_sources(env, ["recovered fresh speech"])
        print(
            "QWEN_READINESS_RECOVERY "
            + json.dumps(
                {
                    "cause": type(failure()).__name__,
                    "connects": transport.attempts,
                    "backoffs_s": env.factory.delays,
                    "cooldown_s": 30,
                    "source_times_s": [0, 29.999, 30],
                    "terminals": [
                        (event.outcome, event.recovery_pending) for event in terminals(env)
                    ],
                    "successful_pcm_bytes": sum(
                        len(value) for value in socket.sent if isinstance(value, bytes)
                    ),
                    "consumer_source": "recovered fresh speech",
                    "capture_active": env.owner.snapshot.effective_active,
                },
                sort_keys=True,
            )
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("predecessor_finish", ["during_retry", "after_cooldown"])
async def test_successor_readiness_failure_does_not_revoke_predecessor_or_clear_its_cooldown(
    predecessor_finish,
):
    first = ScriptedSocket()
    transport = FaultTransport([first, TimeoutError(), TimeoutError(), TimeoutError()])
    async with capture_runtime("peer", transport) as env:

        async def on_attempt(attempt):
            if predecessor_finish == "during_retry" and attempt == 3:
                await finish(first, "legitimate predecessor tail")
                await wait_until(lambda: len(terminals(env)) == 1)

        transport.on_attempt = on_attempt
        await utterance(env)
        await wait_until(lambda: finish_sent(first))
        await utterance(env, 0.2)
        await wait_until(lambda: env.factory.engines[0]._connection_retry_at_s == 30)
        assert not first.closed
        if predecessor_finish == "after_cooldown":
            assert terminals(env) == []
            await finish(first, "legitimate predecessor tail")
        await wait_until(lambda: len(terminals(env)) == 2)
        assert [
            (event.outcome, event.text, event.recovery_pending) for event in terminals(env)
        ] == [
            ("final", "legitimate predecessor tail", False),
            ("failed", "", True),
        ]
        assert env.factory.engines[0]._connection_retry_at_s == 30
        assert env.owner.snapshot.effective_active
        assert transport.attempts == 4
        await utterance(env, 0.25)
        await wait_until(lambda: len(terminals(env)) == 3)
        assert transport.attempts == 4
        await env.factory.clock.advance_to(30)
        await utterance(env, 0.3)
        await wait_until(lambda: len(transport.sockets) == 2 and finish_sent(transport.sockets[-1]))
        await finish(transport.sockets[-1], "fresh successor recovered")
        await wait_until(lambda: len(terminals(env)) == 4)
        assert terminals(env)[-1].text == "fresh successor recovered"
        assert env.owner.snapshot.effective_active
        await assert_consumer_sources(
            env, ["legitimate predecessor tail", "fresh successor recovered"]
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    [
        lambda: StatusError(401),
        lambda: StatusError(403),
        lambda: ssl.SSLCertVerificationError("controlled certificate"),
        lambda: FileNotFoundError("controlled config"),
        lambda: ValueError("controlled configuration"),
    ],
)
async def test_actual_native_nonrecoverable_readiness_does_not_retry_or_enter_cooldown(failure):
    transport = FaultTransport([failure()])
    async with capture_runtime("peer", transport) as env:
        await utterance(env)
        await wait_until(lambda: bool(terminals(env)))
        assert transport.attempts == 1
        assert env.factory.delays == []
        assert not terminals(env)[0].recovery_pending
        assert env.factory.engines[0]._connection_retry_at_s is None


@pytest.mark.asyncio
async def test_actual_task_started_timeout_retries_without_pcm_or_duplicate_public_terminals():
    sockets = [ScriptedSocket(ready=False) for _ in range(3)]
    transport = FaultTransport(sockets)
    async with capture_runtime("peer", transport) as env:
        await utterance(env)
        await wait_until(lambda: len(terminals(env)) == 1)
        assert transport.attempts == 3
        assert all(socket.closed for socket in sockets)
        assert all(not any(isinstance(value, bytes) for value in socket.sent) for socket in sockets)
        assert len({socket.task_ids[0] for socket in sockets}) == 3
        assert terminals(env)[0].recovery_pending
        assert env.owner.snapshot.effective_active


@pytest.mark.asyncio
async def test_uncertain_pcm_write_is_not_retried_as_readiness_or_replayed():
    socket = ScriptedSocket()
    socket.fail_audio = True
    transport = FaultTransport([socket])
    async with capture_runtime("peer", transport) as env:
        await utterance(env)
        await wait_until(lambda: bool(terminals(env)))
        assert transport.attempts == 1
        assert not terminals(env)[0].recovery_pending
        assert not any(isinstance(value, bytes) for value in socket.sent)
        assert env.factory.delays == []


@pytest.mark.asyncio
@pytest.mark.parametrize("channel", ["self", "peer"])
@pytest.mark.parametrize("gap", ["known_loss", "epoch"])
async def test_real_capture_gap_bypasses_blocked_fifo_retires_both_slots_and_new_source_progresses(
    channel, gap
):
    transport = FaultTransport()
    async with capture_runtime(channel, transport) as env:
        for value in (0.1, 0.2):
            await utterance(env, value)
        await wait_until(
            lambda: len(transport.sockets) == 2
            and all(finish_sent(socket) for socket in transport.sockets)
        )
        old_sockets = list(transport.sockets)
        engine = env.factory.engines[0]
        old_authority = engine._authority_generation
        if channel == "self":
            await old_sockets[1].push(native(old_sockets[1].task_ids[0], text="queued old update"))
            await wait_until(
                lambda: any(turn.pending_update is not None for turn in engine._turns.values())
            )
        env.sink.cancelled_dispatch_gate = asyncio.Event()
        await utterance(env, 0.25)
        await asyncio.sleep(0.01)
        assert transport.attempts == 2

        class NoOnsetVad:
            chunk_samples = 8

            def process_chunk(self, _chunk):
                return []

        now = asyncio.get_running_loop().time()
        frames = []
        for index in range(2):
            capture = replace(
                span(index, index * 8, (index + 1) * 8),
                capture_epoch=1 if index == 0 or gap == "known_loss" else 2,
                source_start_monotonic_s=now + index * 0.001,
                source_end_monotonic_s=now + index * 0.001 + 8 / 16000,
                discontinuity_before=(
                    AudioCaptureDiscontinuity("known_loss", now, 8)
                    if index and gap == "known_loss"
                    else None
                ),
            )
            frames.append(AudioFrameF32(np.full(8, 0.4, dtype=np.float32), 16000, capture=capture))
        gap_task = asyncio.create_task(
            run_audio_vad_loop(
                source=FakeAudioSource(frames),
                vad=NoOnsetVad(),
                sink=env.guarded,
                target_sample_rate_hz=16000,
                channel_label=channel,
            )
        )
        try:
            await wait_until(lambda: engine._authority_generation > old_authority)
            await asyncio.wait_for(env.sink.cancelled_dispatch.wait(), timeout=0.5)
            assert not gap_task.done()
            for socket in old_sockets:
                await socket.push(native(socket.task_ids[0], 99, "OLD_AFTER_DISCONTINUITY"))
                await socket.push(control(socket.task_ids[0], "task-finished"))
        finally:
            env.sink.cancelled_dispatch_gate.set()
            await asyncio.wait_for(gap_task, timeout=0.5)
        assert [event.boundary_reason for event in env.sink.boundaries] == ["source_discontinuity"]
        assert all(event.chunk.size == 0 and event.capture == () for event in env.sink.boundaries)
        await wait_until(lambda: all(socket.closed for socket in old_sockets))
        assert not env.guarded._queue
        retained_after_abort = env.guarded.retention_budget.used_bytes
        assert retained_after_abort == 0
        if channel == "peer":
            transport.outcomes.extend([TimeoutError(), TimeoutError()])
        await utterance(env, 0.3, epoch=2)
        await wait_until(lambda: len(transport.sockets) == 3 and finish_sent(transport.sockets[-1]))
        await finish(transport.sockets[-1], "valid postgap source")
        await wait_until(
            lambda: any(event.text == "valid postgap source" for event in terminals(env))
        )
        assert not any(
            "OLD_AFTER_DISCONTINUITY" in event.text or "queued old update" in event.text
            for event in env.events
            if isinstance(event, (STTProviderTurnTerminal, STTProviderTurnUpdate))
        )
        assert [event.text for event in terminals(env) if event.outcome == "final"] == [
            "valid postgap source"
        ]
        assert env.owner.snapshot.effective_active
        assert transport.attempts == (5 if channel == "peer" else 3)
        assert (
            sum(
                len(value)
                for socket in transport.sockets
                for value in socket.sent
                if isinstance(value, bytes)
            )
            == 960
        )
        await assert_consumer_sources(env, ["valid postgap source"])
        print(
            "QWEN_CAPTURE_GAP_RECOVERY "
            + json.dumps(
                {
                    "channel": channel,
                    "gap": gap,
                    "controls": [event.boundary_reason for event in env.sink.boundaries],
                    "old_authority": old_authority,
                    "new_authority": engine._authority_generation,
                    "old_sockets_closed": sum(socket.closed for socket in old_sockets),
                    "connect_attempts": transport.attempts,
                    "source_pcm_bytes": sum(
                        len(value)
                        for socket in transport.sockets
                        for value in socket.sent
                        if isinstance(value, bytes)
                    ),
                    "final_sources": [
                        event.text for event in terminals(env) if event.outcome == "final"
                    ],
                    "fifo_retained_bytes_after_abort": retained_after_abort,
                    "capture_active": env.owner.snapshot.effective_active,
                },
                sort_keys=True,
            )
        )


@pytest.mark.asyncio
async def test_successor_native_readiness_retries_preserve_predecessor_and_ordered_consumer_text():
    first, unready, second = ScriptedSocket(), ScriptedSocket(ready=False), ScriptedSocket()
    transport = FaultTransport([first, StatusError(503), unready, second])
    async with capture_runtime("peer", transport) as env:
        await utterance(env, 0.1)
        await wait_until(lambda: finish_sent(first))
        await utterance(env, 0.2)
        await wait_until(lambda: finish_sent(second))
        assert transport.attempts == 4
        assert unready.closed and not first.closed and not second.closed
        assert not any(isinstance(value, bytes) for value in unready.sent)
        assert sum(len(value) for value in second.sent if isinstance(value, bytes)) == 320
        assert env.factory.delays == [0.8, 1.6]
        await finish(second, " B real successor ")
        await asyncio.sleep(0.01)
        assert terminals(env) == []
        assert not any(
            event.type is UIEventType.TRANSCRIPT_FINAL for event in env.harness.ui_events._queue
        )
        await finish(first, " A valid predecessor ")
        await wait_until(lambda: len(terminals(env)) == 2)
        assert [event.outcome for event in terminals(env)] == ["final", "final"]
        await assert_consumer_sources(env, [" A valid predecessor ", " B real successor "])
        assert env.owner.snapshot.effective_active


@pytest.mark.asyncio
@pytest.mark.parametrize("channel", ["self", "peer"])
async def test_capture_control_cancels_actual_native_readiness_without_replaying_source(channel):
    unready = ScriptedSocket(ready=False)
    transport = FaultTransport([unready])
    async with capture_runtime(channel, transport) as env:
        await utterance(env, 0.1)
        await wait_until(lambda: bool(unready.task_ids))
        await env.guarded.handle_stream_input(
            CaptureStreamInput(
                np.empty(0, dtype=np.float32),
                (),
                boundary_reason="source_discontinuity",
            )
        )
        await wait_until(lambda: unready.closed)
        assert not any(isinstance(value, bytes) for value in unready.sent)
        assert not env.guarded._queue
        assert env.guarded.retention_budget.used_bytes == 0
        await utterance(env, 0.3, epoch=2)
        await wait_until(lambda: len(transport.sockets) == 2 and finish_sent(transport.sockets[-1]))
        await finish(transport.sockets[-1], "new source after readiness abort")
        await wait_until(lambda: any(event.outcome == "final" for event in terminals(env)))
        assert [event.outcome for event in terminals(env)] == ["cancelled", "final"]
        assert (
            sum(
                len(value)
                for socket in transport.sockets
                for value in socket.sent
                if isinstance(value, bytes)
            )
            == 320
        )
        await assert_consumer_sources(env, ["new source after readiness abort"])
        assert env.owner.snapshot.effective_active
