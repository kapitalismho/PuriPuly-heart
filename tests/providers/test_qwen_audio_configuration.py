from __future__ import annotations

import asyncio
import json
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
from puripuly_heart.core.audio.ownership import PeerAudioSegmentLedger
from puripuly_heart.core.runtime.local_asr_provider_runtime import LocalASRProviderRuntimeOwner
from puripuly_heart.core.stt.backend import (
    STTProviderTurnTerminal,
    STTProviderTurnUpdate,
    STTSessionProjection,
)
from puripuly_heart.core.stt.scoped_engine import ScopedRecognitionEngine
from puripuly_heart.core.vad.gating import SpeechEnd, SpeechStart
from puripuly_heart.providers.stt.qwen_audio import QwenAudioStreamingSTTBackend
from tests.core.runtime.test_local_asr_provider_runtime import (
    FakeGpuRuntimeFactory,
    FakeProvisioningPort,
)
from tests.core.runtime.test_peer_capture_session import make_owner
from tests.core.runtime.test_self_capture_session import LoopHarness, build_owner
from tests.core.test_stt_scoped_engine import segment_events, span, wait_until, watchdogs
from tests.helpers.fakes import RecordingOscQueue
from tests.helpers.translation_owners import compose_translation_test_harness
from tests.providers.test_qwen_audio import SocketFactory, control, native


class QwenFactory:
    def __init__(self):
        self.transport = SocketFactory()
        self.requests = []
        self.engines = []

    async def create(self, request, *, gpu_runtime, on_terminal_failure=None):
        self.requests.append(request)
        backend = QwenAudioStreamingSTTBackend(
            api_key="offline",
            channel=request.channel,
            endpoint=request.config.endpoint,
            language_hints=(request.config.source_language,),
            websocket_factory=self.transport,
            task_finish_timeout_s=1,
        )
        engine = ScopedRecognitionEngine(
            channel=request.channel,
            session_factory=lambda _settings, epoch: backend.open_session(
                projection=STTSessionProjection("scoped", epoch)
            ),
            watchdog_resolver=lambda _: watchdogs(final_timeout_s=1),
            accepted_settings_scope=(
                request.provider_id,
                request.provider_signature,
                request.runtime_signature,
            ),
        )
        self.engines.append(engine)
        return engine


class ControlledSmartTurn:
    def __init__(self):
        self.snapshot = SimpleNamespace(availability="unavailable")
        self.prepare_calls = 0

    def request_prepare(self):
        self.prepare_calls += 1

    async def close(self):
        return None


class RuntimeSink:
    def __init__(self, runtime, channel):
        self.runtime = runtime
        self.channel = channel

    async def handle_vad_event(self, event):
        await self.runtime.handle_owned_vad_event(self.channel, event)

    async def handle_owned_vad_event(self, event):
        await self.runtime.handle_owned_vad_event(self.channel, event)


@pytest.mark.asyncio
@pytest.mark.parametrize("channel", ["self", "peer"])
@pytest.mark.parametrize("entry", ["active_reconcile", "inactive_prepare"])
@pytest.mark.parametrize("change", ["region", "source"])
async def test_actual_capture_configuration_retires_old_qwen_authority_and_new_scope_progresses(
    channel, entry, change
):
    base = AppSettingsVNext()
    initial = replace(
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
    if change == "region":
        updated = replace(
            initial,
            intent=replace(
                initial.intent,
                translation=replace(
                    initial.intent.translation,
                    qwen=replace(initial.intent.translation.qwen, region="singapore"),
                ),
            ),
        )
    else:
        updated = replace(
            initial,
            intent=replace(
                initial.intent,
                languages=replace(
                    initial.intent.languages, source_language="ko", peer_source_language="ko"
                ),
            ),
        )
    harness = compose_translation_test_harness(
        stt=None, llm=None, osc=RecordingOscQueue(), low_latency_mode=True
    )
    factory = QwenFactory()
    received = []
    admitted = []
    ledgers = {}

    async def receive(event):
        received.append(event)
        if channel == "self":
            await harness.self_owner.handle_stt_event(event)
        elif isinstance(event, STTProviderTurnTerminal) and event.outcome in ("final", "degraded"):
            ledger = ledgers[event.identity.segment.segment_id]
            receipt = ledger.terminalize(
                event.identity.segment.segment_id,
                outcome=event.outcome,
                now_monotonic_s=10,
                provider_epoch_id=event.identity.provider_epoch_id,
                provider_turn_id=event.identity.provider_turn_id,
                text_authority=event.text_authority,
            )
            result = await harness.peer_owner.handle_provider_turn_terminal(receipt, event)
            admitted.append(result.transcript)

    runtime = LocalASRProviderRuntimeOwner(
        provider_factory=factory,
        gpu_runtime_factory=FakeGpuRuntimeFactory(),
        provisioning=FakeProvisioningPort(),
        self_event_handler=receive,
        peer_event_handler=receive,
        retired_event_handler=receive,
    )
    loop = LoopHarness()
    sink = RuntimeSink(runtime, channel)
    current_settings = [initial]
    if channel == "self":
        adapter = SelfCaptureProviderAdapter(runtime, harness)
        owner, _, _, _, _, _ = build_owner(provider=adapter, sink=sink, loop=loop)
        make_config = build_self_capture_session_config_from_vnext
        owner._provider_request_factory = (
            lambda _config, warmup: build_self_stt_provider_request_from_vnext(
                current_settings[0], warmup=warmup
            )
        )
    else:
        adapter = PeerCaptureProviderAdapter(runtime, harness)
        owner, _, _, _, _, _ = make_owner(
            provider=adapter,
            sink=sink,
            run_audio_loop=loop.run,
            smart_turn_owner=ControlledSmartTurn(),
        )
        make_config = build_peer_capture_session_config_from_vnext
        owner._provider_request_factory = lambda config, warmup: build_peer_stt_provider_request(
            config, gpu_device_id="auto", warmup=warmup
        )
    old_config, next_config = make_config(initial), make_config(updated)
    assert old_config.runtime_signature != next_config.runtime_signature

    async def enqueue_pair(turns=2):
        guarded = loop.calls[-1]["sink"]
        ids = []
        for index in range(turns):
            identity = uuid4()
            ids.append(identity)
            now = asyncio.get_running_loop().time()
            capture = replace(
                span(index, index * 160, (index + 1) * 160),
                source_start_monotonic_s=now,
                source_end_monotonic_s=now + 0.01,
            )
            event = SpeechStart(
                identity,
                np.empty(0, dtype=np.float32),
                np.full(160, 0.1, dtype=np.float32),
                chunk_capture=(capture,),
            )
            end = SpeechEnd(identity, trailing_silence_ms=224, reason="silence")
            if channel == "self":
                await guarded.handle_vad_event(event)
                await guarded.handle_vad_event(end)
                ledgers[identity] = guarded.ledger
            else:
                ledger = owner.segment_ledger
                ledgers[identity] = ledger
                await guarded.handle_owned_vad_event(
                    ledger.observe_vad_event(event, now_monotonic_s=index)
                )
                await guarded.handle_owned_vad_event(
                    ledger.observe_vad_event(end, now_monotonic_s=index + 0.1)
                )
        return ids

    try:
        await harness.start()
        await runtime.start()
        harness.output_runtime.activate_peer_generation(1)
        if entry == "active_reconcile":
            await owner.apply_intent(old_config, enabled=True)
            await wait_until(lambda: bool(loop.calls))
            await enqueue_pair(turns=3)
        else:
            await owner.prepare_provider(old_config)
            engine = factory.engines[0]
            from puripuly_heart.core.audio.ownership import AudioSegmentSettingsSnapshot

            ledger = PeerAudioSegmentLedger(
                activation_generation=1,
                settings=AudioSegmentSettingsSnapshot(
                    provider_id="qwen_audio",
                    provider_signature=old_config.provider_signature,
                    runtime_signature=old_config.runtime_signature,
                    source_mode="manual",
                    source_language="en",
                    expected_languages=("en",),
                    target_sample_rate_hz=16000,
                    vad_speech_threshold=0.3,
                    vad_hangover_ms=500,
                    vad_pre_roll_ms=500,
                ),
            )
            for index in range(2):
                start, _, end = segment_events(ledger, start_sample=index * 100, now=index)
                ledgers[start.event.utterance_id] = ledger
                await engine.handle_owned_vad_event(start)
                await engine.handle_owned_vad_event(end)
        await wait_until(lambda: len(factory.transport.sockets) == 2)
        old_sockets = list(factory.transport.sockets)
        old_engine = factory.engines[0]
        if entry == "active_reconcile":
            await owner.apply_intent(old_config, enabled=True)
        else:
            await owner.prepare_provider(old_config)
        assert not any(socket.closed for socket in old_sockets)
        assert len(factory.engines) == 1
        if channel == "self":
            await old_sockets[1].push(native(old_sockets[1].task_ids[0], text="queued old update"))
            await wait_until(
                lambda: any(turn.pending_update is not None for turn in old_engine._turns.values())
            )
        current_settings[0] = updated
        if entry == "active_reconcile":
            applied = await owner.apply_intent(next_config, enabled=True)
        else:
            applied = await owner.prepare_provider(next_config)
        assert applied.runtime_signature == next_config.runtime_signature
        assert all(socket.closed for socket in old_sockets)
        assert old_engine._authority_generation > 0
        assert runtime._handles[channel].retained_scoped_providers == ()
        for socket in old_sockets:
            await socket.push(native(socket.task_ids[0], 99, "RETIRED MUST NOT PUBLISH"))
            await socket.push(control(socket.task_ids[0], "task-finished"))
        if entry == "inactive_prepare":
            await owner.apply_intent(next_config, enabled=True)
        await wait_until(lambda: len(loop.calls) == (2 if entry == "active_reconcile" else 1))
        count = len(factory.transport.sockets)
        await enqueue_pair()
        await wait_until(lambda: len(factory.transport.sockets) == count + 2)
        new_sockets = factory.transport.sockets[-2:]
        await wait_until(
            lambda: all(
                any(
                    isinstance(value, str)
                    and json.loads(value)["header"]["action"] == "finish-task"
                    for value in socket.sent
                )
                for socket in new_sockets
            )
        )
        for socket in reversed(new_sockets):
            await socket.push(native(socket.task_ids[0], 1, "valid new scope"))
            await socket.push(control(socket.task_ids[0], "task-finished"))
        await wait_until(
            lambda: sum(
                isinstance(event, STTProviderTurnTerminal) and event.text == "valid new scope"
                for event in received
            )
            == 2
        )
        assert not any(
            "RETIRED" in event.text or "queued old update" in event.text
            for event in received
            if isinstance(event, (STTProviderTurnUpdate, STTProviderTurnTerminal))
        )
        assert factory.requests[-1].config.region == (
            "singapore" if change == "region" else initial.intent.translation.qwen.region
        )
        assert factory.requests[-1].config.source_language == ("ko" if change == "source" else "en")
        if channel == "peer":
            await wait_until(lambda: len(admitted) == 2)
            assert [transcript.text for transcript in admitted] == ["valid new scope"] * 2
    finally:
        await owner.close()
        await runtime.close()
        await harness.stop()
