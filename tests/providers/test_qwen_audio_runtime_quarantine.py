from __future__ import annotations

import asyncio
import hashlib
import json
from contextlib import asynccontextmanager
from dataclasses import replace
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import pytest
from websockets.asyncio.client import connect
from websockets.asyncio.server import serve

from puripuly_heart.app.adapters.peer_capture.peer_capture_provider import (
    PeerCaptureProviderAdapter,
)
from puripuly_heart.app.wiring.wiring_local_asr_provider_runtime import SharedSTTProviderFactory
from puripuly_heart.app.wiring.wiring_stt_factory import (
    build_peer_capture_session_config_from_vnext,
    build_peer_stt_provider_request,
    build_self_capture_session_config_from_vnext,
    build_self_stt_provider_request_from_vnext,
)
from puripuly_heart.config.settings_vnext.schema import AppSettingsVNext
from puripuly_heart.core import network_clients
from puripuly_heart.core.audio.format import float32_to_pcm16le_bytes
from puripuly_heart.core.audio.ownership import PeerAudioSegmentLedger
from puripuly_heart.core.clock import SystemClock
from puripuly_heart.core.runtime.provider_handle import ProviderResourceQuarantinedError
from puripuly_heart.core.storage.secrets import InMemorySecretStore
from puripuly_heart.core.stt.scoped_engine import ScopedRecognitionEngine
from puripuly_heart.core.vad.gating import SpeechChunk, SpeechEnd, SpeechStart
from puripuly_heart.domain.events import UIEventType
from tests.core.runtime.test_peer_capture_session import make_owner
from tests.core.runtime.test_self_capture_session import LoopHarness
from tests.core.test_self_translation_low_latency import FakeLLMProvider
from tests.core.test_stt_scoped_engine import (
    ControlledScopedSession,
    span,
    wait_until,
    watchdogs,
)
from tests.core.test_stt_scoped_engine import (
    settings as segment_settings,
)
from tests.helpers.fakes import RecordingOscQueue
from tests.helpers.translation_owners import compose_translation_test_harness
from tests.providers.test_qwen_audio import ScriptedSocket, control, native
from tests.providers.test_qwen_audio_configuration import ControlledSmartTurn, RuntimeSink
from tests.providers.test_qwen_audio_delivery_recovery import (
    BoundRuntimeFactory,
    HeldManualProvider,
    final_sources,
    require_final_source,
    self_capture_pipeline,
    spoken_turn,
)
from tests.providers.test_qwen_audio_recovery_boundaries import (
    FaultTransport,
    RecoveryFactory,
    finish,
    finish_sent,
    utterance,
)


class ReleasableSocket(ScriptedSocket):
    def __init__(self):
        super().__init__(ready=False)
        self.allow_close = False
        self.close_attempts = 0

    async def close(self):
        self.close_attempts += 1
        if not self.allow_close:
            raise TimeoutError("owned runtime retirement remains unconfirmed")
        await super().close()


class RequestFactory(RecoveryFactory):
    def __init__(self, transport):
        super().__init__(transport, task_start_timeout_s=0.1)
        self.requests = []
        self.build_entered = asyncio.Event()
        self.build_release = asyncio.Event()
        self.hold_next_build = False

    async def create(self, request, *, gpu_runtime, on_terminal_failure=None):
        self.requests.append(request)
        if self.hold_next_build:
            self.hold_next_build = False
            self.build_entered.set()
            await self.build_release.wait()
        if request.provider_id == "qwen_audio":
            return await super().create(
                request, gpu_runtime=gpu_runtime, on_terminal_failure=on_terminal_failure
            )
        session = ControlledScopedSession()
        session.terminal_on_seal = ("final", "Other provider permitted.")

        async def open_session(_settings, _epoch):
            return session

        engine = ScopedRecognitionEngine(
            channel=request.channel,
            session_factory=open_session,
            watchdog_resolver=lambda _: watchdogs(),
            accepted_settings_scope=(
                request.provider_id,
                request.provider_signature,
                request.runtime_signature,
            ),
        )
        self.engines.append(engine)
        return engine


async def require_retired_self(env, unsafe):
    await spoken_turn(env)
    runtime = env.harness.self_owner.local_asr_runtime
    handle = runtime._handles["self"]
    await wait_until(
        lambda: bool(handle._retired_providers) and not env.owner.snapshot.desired_active
    )
    old = env.factory.engines[0]
    assert old in handle._retired_providers
    assert old.cleanup_debt == 1
    assert not unsafe.closed
    assert len(unsafe.task_ids) == 1
    assert not any(isinstance(value, bytes) for value in unsafe.sent)
    return runtime, handle, old


async def enable_self(env, config):
    result = await env.owner.apply_intent(config, enabled=True)
    if result.effective_active:
        env.guarded = env.owner._vad_dispatch
    return result


def failed_reclaims(runtime, channel):
    return [
        item
        for item in runtime.diagnostics
        if item.channel == channel and item.failure_code == "provider_resource_quarantined"
    ]


@asynccontextmanager
async def peer_pipeline(env):
    base = AppSettingsVNext()
    settings = replace(
        base,
        intent=replace(
            base.intent,
            peer_stt=replace(base.intent.peer_stt, provider="qwen_audio"),
            languages=replace(base.intent.languages, peer_source_language="en"),
        ),
    )
    runtime = env.harness.self_owner.local_asr_runtime
    loop = LoopHarness()
    owner, *_ = make_owner(
        provider=PeerCaptureProviderAdapter(runtime, env.harness),
        sink=RuntimeSink(runtime, "peer"),
        run_audio_loop=loop.run,
        smart_turn_owner=ControlledSmartTurn(),
    )
    owner._provider_request_factory = lambda config, warmup: build_peer_stt_provider_request(
        config, gpu_device_id="auto", warmup=warmup
    )
    env.callbacks.bind_peer_capture(owner)
    owner.bind_publication_generation_observer(
        activated=env.harness.output_runtime.activate_peer_generation,
        retired=env.harness.output_runtime.retire_peer_generation,
    )
    config = build_peer_capture_session_config_from_vnext(settings)
    try:
        await owner.apply_intent(config, enabled=True)
        await wait_until(lambda: bool(loop.calls))
        yield SimpleNamespace(
            channel="peer",
            owner=owner,
            guarded=loop.calls[-1]["sink"],
            settings=settings,
            config=config,
            loop=loop,
        )
    finally:
        await owner.close()


def peer_final_sources(env):
    return [
        event.payload.text
        for event in env.harness.ui_events._queue
        if event.channel == "peer" and event.type is UIEventType.TRANSCRIPT_FINAL
    ]


@pytest.mark.asyncio
async def test_self_reenable_blocks_retired_qwen_before_build_then_reclaims_before_two_fresh_turns():
    unsafe = ReleasableSocket()
    transport = FaultTransport([unsafe])
    factory = RequestFactory(transport)
    llm = HeldManualProvider()
    async with self_capture_pipeline(
        transport, llm, provider_factory=factory, peer_translation_enabled=True
    ) as env:
        try:
            runtime, handle, old = await require_retired_self(env, unsafe)
            config = env.owner.current_config
            manual = await env.harness.self_owner.submit_text("Manual independent marker.")
            await asyncio.wait_for(llm.manual_entered.wait(), timeout=1)
            blocked = await enable_self(env, config)
            assert not blocked.effective_active
            assert transport.attempts == 1
            assert len(factory.requests) == 1 and len(factory.engines) == 1
            assert handle.provider is None
            assert handle._retired_providers == [old]
            assert old.cleanup_debt == 1 and not unsafe.closed
            assert failed_reclaims(runtime, "self")[-1].event == "provider_reclaim"
            with pytest.raises(ProviderResourceQuarantinedError) as captured:
                await handle.reclaim_qwen_resources()
            assert captured.value.cleanup_debt == 1 and captured.value.owner_count == 1
            assert captured.value.__cause__ is not None
            assert str(captured.value.__cause__) == "provider_resource_quarantined"
            async with peer_pipeline(env) as peer:
                await utterance(peer, 0.15)
                await wait_until(
                    lambda: len(transport.sockets) == 2 and finish_sent(transport.sockets[-1])
                )
                await finish(transport.sockets[-1], "Independent Peer source.")
                await wait_until(lambda: peer_final_sources(env) == ["Independent Peer source."])
                await wait_until(
                    lambda: any(call["text"] == "Independent Peer source." for call in llm.calls)
                )
                assert peer.owner.snapshot.effective_active
                assert not unsafe.closed
                llm.manual_release.set()
                await env.harness.translation_turns.wait_for_idle()
                assert any(
                    event.type is UIEventType.TRANSLATION_DONE
                    and event.utterance_id == manual.turn_id
                    for event in env.harness.ui_events._queue
                )
                unsafe.allow_close = True
                resumed = await enable_self(env, config)
                assert resumed.effective_active and unsafe.closed
                assert old.cleanup_debt == 0 and old not in handle._retired_providers
                before = len(transport.sockets)
                await spoken_turn(env, 0.2, epoch=2)
                await wait_until(
                    lambda: len(transport.sockets) == before + 1
                    and finish_sent(transport.sockets[-1])
                )
                first = transport.sockets[-1]
                await spoken_turn(env, 0.3, epoch=2)
                await wait_until(
                    lambda: len(transport.sockets) == before + 2
                    and finish_sent(transport.sockets[-1])
                )
                second = transport.sockets[-1]
                assert sum(not socket.closed for socket in (unsafe, first, second)) == 2
                for socket, prefix, tail in (
                    (second, " Runtime B.", " Tail B."),
                    (first, "Runtime A.", " Tail A."),
                ):
                    await socket.push(native(socket.task_ids[0], 1, prefix))
                    await socket.push(native(socket.task_ids[0], 2, tail))
                    await socket.push(control(socket.task_ids[0], "task-finished"))
                result = await require_final_source(env, "Runtime A. Tail A. Runtime B. Tail B.")
                assert len(result.recognition_origins) == 4
                assert env.owner.snapshot.effective_active and peer.owner.snapshot.effective_active
                assert len(unsafe.task_ids) == 1
        finally:
            unsafe.allow_close = True
            llm.manual_release.set()


@pytest.mark.asyncio
async def test_qwen_to_other_provider_then_qwen_does_not_hide_retired_qwen_ownership():
    unsafe = ReleasableSocket()
    transport = FaultTransport([unsafe])
    factory = RequestFactory(transport)
    async with self_capture_pipeline(transport, provider_factory=factory) as env:
        try:
            runtime, handle, old = await require_retired_self(env, unsafe)
            qwen = env.settings_box[0]
            other = replace(
                qwen, intent=replace(qwen.intent, stt=replace(qwen.intent.stt, provider="soniox"))
            )
            env.settings_box[0] = other
            other_config = build_self_capture_session_config_from_vnext(other)
            await enable_self(env, other_config)
            assert env.owner.snapshot.effective_active
            assert runtime.snapshot.channel_for("self").provider_id == "soniox"
            await spoken_turn(env, 0.2)
            await require_final_source(env, "Other provider permitted.")
            assert not unsafe.closed and old in handle._retired_providers
            assert transport.attempts == 1
            target = replace(
                qwen,
                intent=replace(
                    qwen.intent,
                    translation=replace(
                        qwen.intent.translation,
                        qwen=replace(qwen.intent.translation.qwen, region="singapore"),
                    ),
                ),
            )
            env.settings_box[0] = target
            target_config = build_self_capture_session_config_from_vnext(target)
            built = len(factory.engines)
            blocked = await enable_self(env, target_config)
            assert blocked.effective_active
            assert runtime.snapshot.channel_for("self").provider_id == "soniox"
            assert len(factory.engines) == built and transport.attempts == 1
            assert old in handle._retired_providers and old.cleanup_debt == 1
            assert failed_reclaims(runtime, "self")
            unsafe.allow_close = True
            resumed = await enable_self(env, target_config)
            assert resumed.effective_active and unsafe.closed
            assert old not in handle._retired_providers and old.cleanup_debt == 0
            await spoken_turn(env, 0.3, epoch=2)
            await wait_until(
                lambda: len(transport.sockets) == 2 and finish_sent(transport.sockets[-1])
            )
            await finish(transport.sockets[-1], "New Qwen scope reclaimed.")
            await require_final_source(env, "New Qwen scope reclaimed.")
            assert factory.requests[-1].provider_id == "qwen_audio"
            assert factory.requests[-1].config.region == "singapore"
        finally:
            unsafe.allow_close = True


@pytest.mark.asyncio
async def test_peer_configuration_replacement_cannot_bypass_its_retired_qwen_owner():
    unsafe = ReleasableSocket()
    transport = FaultTransport([unsafe])
    factory = RequestFactory(transport)
    async with self_capture_pipeline(
        transport, provider_factory=factory, peer_translation_enabled=True
    ) as env:
        try:
            async with peer_pipeline(env) as peer:
                await utterance(peer)
                old = factory.engines[-1]
                await wait_until(lambda: old._quarantined_session is not None)
                assert peer.owner.snapshot.effective_active
                runtime = env.harness.self_owner.local_asr_runtime
                handle = runtime._handles["peer"]
                updated = replace(
                    peer.settings,
                    intent=replace(
                        peer.settings.intent,
                        translation=replace(
                            peer.settings.intent.translation,
                            qwen=replace(peer.settings.intent.translation.qwen, region="singapore"),
                        ),
                    ),
                )
                config = build_peer_capture_session_config_from_vnext(updated)
                before = len(factory.engines)
                with pytest.raises(RuntimeError, match="provider_resource_quarantined"):
                    await peer.owner.apply_intent(config, enabled=True)
                assert old in handle._retired_providers and old.cleanup_debt == 1
                assert len(factory.engines) == before and transport.attempts == 1
                blocked = await peer.owner.apply_intent(config, enabled=True)
                assert not blocked.effective_active
                assert old in handle._retired_providers and old.cleanup_debt == 1
                assert len(factory.engines) == before and transport.attempts == 1
                assert failed_reclaims(runtime, "peer")
                assert env.owner.snapshot.effective_active
                unsafe.allow_close = True
                resumed = await peer.owner.apply_intent(config, enabled=True)
                assert resumed.effective_active and unsafe.closed
                await wait_until(lambda: peer.loop.calls[-1]["sink"] is not peer.guarded)
                peer.guarded = peer.loop.calls[-1]["sink"]
                await utterance(peer, 0.3)
                await wait_until(
                    lambda: len(transport.sockets) == 2 and finish_sent(transport.sockets[-1])
                )
                await finish(transport.sockets[-1], "Peer scope reclaimed.")
                await wait_until(lambda: "Peer scope reclaimed." in peer_final_sources(env))
                await wait_until(
                    lambda: any(call["text"] == "Peer scope reclaimed." for call in env.llm.calls)
                )
                assert old.cleanup_debt == 0 and old not in handle._retired_providers
        finally:
            unsafe.allow_close = True


@pytest.mark.asyncio
async def test_qwen_publication_rechecks_cleanup_that_appears_while_factory_is_in_flight():
    unsafe = ReleasableSocket()
    transport = FaultTransport([unsafe])
    factory = RequestFactory(transport)
    async with self_capture_pipeline(transport, provider_factory=factory) as env:
        replacing = None
        try:
            runtime = env.harness.self_owner.local_asr_runtime
            factory.hold_next_build = True
            request = build_self_stt_provider_request_from_vnext(env.settings_box[0], warmup=False)
            replacing = asyncio.create_task(runtime.replace_provider(request, start=True))
            await asyncio.wait_for(factory.build_entered.wait(), timeout=1)
            _, handle, old = await require_retired_self(env, unsafe)
            factory.build_release.set()
            result = await replacing
            assert result.status == "failed"
            assert result.failure_code == "provider_resource_quarantined"
            assert result.failure_stage == "provider_replace"
            assert handle.provider is None and old in handle._retired_providers
            assert old.cleanup_debt == 1
            assert transport.attempts == 1
            assert len(factory.engines) == 2 and factory.engines[1]._closed
            unsafe.allow_close = True
            await enable_self(env, env.owner.current_config)
            assert unsafe.closed and old.cleanup_debt == 0
            await spoken_turn(env, 0.3, epoch=2)
            await wait_until(
                lambda: len(transport.sockets) == 2 and finish_sent(transport.sockets[-1])
            )
            await finish(transport.sockets[-1], "Atomic admission recovered.")
            await require_final_source(env, "Atomic admission recovered.")
        finally:
            unsafe.allow_close = True
            factory.build_release.set()
            if replacing is not None:
                await replacing


@pytest.mark.asyncio
async def test_pending_qwen_handoff_cannot_report_admission_after_actual_fault_retirement():
    unsafe = ReleasableSocket()
    transport = FaultTransport([unsafe])
    factory = RequestFactory(transport)
    async with self_capture_pipeline(transport, provider_factory=factory) as env:
        handing_off = None
        try:
            await spoken_turn(env)
            await wait_until(lambda: bool(unsafe.task_ids))
            runtime = env.harness.self_owner.local_asr_runtime
            handle = runtime._handles["self"]
            request = build_self_stt_provider_request_from_vnext(env.settings_box[0], warmup=False)
            handing_off = asyncio.create_task(runtime.handoff_provider(request, start=True))
            await wait_until(lambda: handle._pending_handoff is not None)
            await wait_until(
                lambda: bool(handle._retired_providers) and not env.owner.snapshot.desired_active
            )
            result = await handing_off
            assert result.status == "failed"
            assert result.failure_code == "provider_resource_quarantined"
            assert handle.provider is None
            assert runtime.snapshot.channel_for("self").provider_id is None
            assert transport.attempts == 1 and not unsafe.closed
            unsafe.allow_close = True
            await enable_self(env, env.owner.current_config)
            assert unsafe.closed
            await spoken_turn(env, 0.3, epoch=2)
            await wait_until(
                lambda: len(transport.sockets) == 2 and finish_sent(transport.sockets[-1])
            )
            await finish(transport.sockets[-1], "Handoff admission recovered.")
            await require_final_source(env, "Handoff admission recovered.")
        finally:
            unsafe.allow_close = True
            if handing_off is not None:
                await handing_off


@pytest.mark.asyncio
async def test_qwen_return_blocks_while_its_previous_owned_retirement_is_still_in_flight():
    class HeldRetirementSocket(ReleasableSocket):
        def __init__(self):
            super().__init__()
            self.hold = False
            self.entered = asyncio.Event()
            self.release = asyncio.Event()

        async def close(self):
            if self.hold:
                self.entered.set()
                await self.release.wait()
            await super().close()

    unsafe = HeldRetirementSocket()
    transport = FaultTransport([unsafe])
    factory = RequestFactory(transport)
    async with self_capture_pipeline(
        transport, provider_factory=factory, peer_translation_enabled=True
    ) as env:
        try:
            async with peer_pipeline(env) as peer:
                await utterance(peer)
                old = factory.engines[-1]
                await wait_until(lambda: old._quarantined_session is not None)
                runtime = env.harness.self_owner.local_asr_runtime
                handle = runtime._handles["peer"]
                with pytest.raises(RuntimeError, match="provider_resource_quarantined"):
                    await runtime.release_channel("peer", mode="drain")
                assert old._closed and old.cleanup_debt == 1
                assert handle.provider is old
                other = replace(
                    peer.settings,
                    intent=replace(
                        peer.settings.intent,
                        peer_stt=replace(peer.settings.intent.peer_stt, provider="soniox"),
                    ),
                )
                request = build_peer_stt_provider_request(
                    build_peer_capture_session_config_from_vnext(other),
                    gpu_device_id="auto",
                    warmup=False,
                )
                unsafe.hold = True
                applied = await runtime.handoff_provider(request, start=True)
                assert applied.status == "applied"
                await asyncio.wait_for(unsafe.entered.wait(), timeout=1)
                assert any(not task.done() for task in handle._retirement_tasks)
                assert old in handle._retired_providers
                assert runtime.snapshot.channel_for("peer").provider_id == "soniox"
                before = len(factory.engines)
                qwen_request = build_peer_stt_provider_request(
                    peer.config, gpu_device_id="auto", warmup=False
                )
                blocked = await runtime.replace_provider(qwen_request, start=True)
                assert (
                    blocked.status == "failed"
                    and blocked.failure_code == "provider_resource_quarantined"
                )
                assert blocked.failure_stage == "provider_reclaim"
                assert len(factory.engines) == before and transport.attempts == 1
                assert old in handle._retired_providers and not unsafe.closed
                assert runtime.snapshot.channel_for("peer").provider_id == "soniox"
                unsafe.allow_close = True
                unsafe.hold = False
                unsafe.release.set()
                recovered = await runtime.replace_provider(qwen_request, start=True)
                assert recovered.status == "applied" and unsafe.closed
                assert old.cleanup_debt == 0 and old not in handle._retired_providers
                await utterance(peer, 0.3)
                await wait_until(
                    lambda: len(transport.sockets) == 2 and finish_sent(transport.sockets[-1])
                )
                await finish(transport.sockets[-1], "Retirement window recovered.")
                await wait_until(
                    lambda: peer_final_sources(env) == ["Retirement window recovered."]
                )
                await wait_until(
                    lambda: any(
                        call["text"] == "Retirement window recovered." for call in env.llm.calls
                    )
                )
        finally:
            unsafe.allow_close = True
            unsafe.hold = False
            unsafe.release.set()


async def expire_healthy_qwen_owner(env):
    runtime = env.harness.self_owner.local_asr_runtime
    handle = runtime._handles["self"]
    old = env.factory.engines[0]
    qwen = env.settings_box[0]
    other = replace(
        qwen,
        intent=replace(qwen.intent, stt=replace(qwen.intent.stt, provider="soniox")),
    )
    request = build_self_stt_provider_request_from_vnext(other, warmup=False)
    switching = asyncio.create_task(runtime.handoff_provider(request, start=True))
    await wait_until(lambda: handle._pending_handoff is not None)
    await runtime.commit_handoff("self")
    assert (await switching).status == "applied"
    for _ in range(handle.max_retained_scoped_providers):
        assert (await runtime.handoff_provider(request, start=True)).status == "applied"
    await wait_until(lambda: old._closing)
    assert old not in handle.retained_scoped_providers
    assert old.cleanup_debt == 0
    assert old in handle._retired_providers
    assert any(not task.done() for task in handle._retirement_tasks)
    return runtime, handle, old, build_self_stt_provider_request_from_vnext(qwen, warmup=False)


@pytest.mark.asyncio
async def test_healthy_qwen_retained_expiry_stays_owned_until_native_drain_then_two_fresh_turns():
    transport = FaultTransport([])
    factory = RequestFactory(transport)
    async with self_capture_pipeline(transport, provider_factory=factory) as env:
        returning = None
        completed = set()
        try:
            await spoken_turn(env)
            await wait_until(
                lambda: len(transport.sockets) == 1 and finish_sent(transport.sockets[0])
            )
            old_socket = transport.sockets[0]
            await old_socket.push(native(old_socket.task_ids[0], 1, "Already admitted old."))
            await wait_until(
                lambda: env.harness.self_owner.merge_buffer is not None
                and env.harness.self_owner.merge_buffer.parts == ["Already admitted old."]
            )
            runtime, handle, old, request = await expire_healthy_qwen_owner(env)
            assert not old_socket.closed
            assert handle._event_handler_for(provider=old, generation=0) is None
            before = len(factory.requests)
            returning = asyncio.create_task(runtime.handoff_provider(request, start=True))
            await wait_until(lambda: runtime.snapshot.channel_for("self").phase == "building")
            assert not returning.done()
            assert len(factory.requests) == before and transport.attempts == 1
            assert old in handle._retired_providers and old.cleanup_debt == 0
            assert not old_socket.closed
            await old_socket.push(native(old_socket.task_ids[0], 2, " EXPIRED OLD TAIL."))
            await old_socket.push(control(old_socket.task_ids[0], "task-finished"))
            completed.add(old_socket)
            assert (await returning).status == "applied"
            assert old_socket.closed and old.cleanup_debt == 0
            assert old not in handle._retired_providers
            await spoken_turn(env, 0.2)
            await wait_until(
                lambda: len(transport.sockets) == 2 and finish_sent(transport.sockets[-1])
            )
            first = transport.sockets[-1]
            await spoken_turn(env, 0.3)
            await wait_until(
                lambda: len(transport.sockets) == 3 and finish_sent(transport.sockets[-1])
            )
            second = transport.sockets[-1]
            assert sum(not socket.closed for socket in transport.sockets) == 2
            for socket, text, tail in (
                (second, " Fresh B.", " Tail B."),
                (first, " Fresh A.", " Tail A."),
            ):
                await socket.push(native(socket.task_ids[0], 1, text))
                await socket.push(native(socket.task_ids[0], 2, tail))
                await socket.push(control(socket.task_ids[0], "task-finished"))
                completed.add(socket)
            source = await require_final_source(
                env, "Already admitted old. Fresh A. Tail A. Fresh B. Tail B."
            )
            assert len(source.recognition_origins) == 5
            assert not any("EXPIRED" in call["text"] for call in env.llm.calls)
            assert all("EXPIRED" not in source.text for source in final_sources(env))
            assert env.owner.snapshot.effective_active
        finally:
            for socket in transport.sockets:
                if socket not in completed and not socket.closed:
                    await finish(socket, "Owned fixture cleanup.")
            if returning is not None:
                await returning


@pytest.mark.asyncio
async def test_healthy_retained_qwen_changed_scope_blocks_on_unconfirmed_close_then_recovers():
    unsafe = ReleasableSocket()
    transport = FaultTransport([unsafe])
    factory = RequestFactory(transport)
    async with self_capture_pipeline(transport, provider_factory=factory) as env:
        try:
            await spoken_turn(env)
            await wait_until(lambda: bool(unsafe.task_ids))
            await unsafe.push(control(unsafe.task_ids[0], "task-started"))
            await wait_until(lambda: finish_sent(unsafe))
            await finish(unsafe, "Healthy admitted source.")
            await require_final_source(env, "Healthy admitted source.")
            runtime = env.harness.self_owner.local_asr_runtime
            old = runtime.current_provider("self")
            assert old.cleanup_debt == 0 and not unsafe.closed
            initial = env.settings_box[0]
            other = replace(
                initial,
                intent=replace(initial.intent, stt=replace(initial.intent.stt, provider="soniox")),
            )
            result = await runtime.handoff_provider(
                build_self_stt_provider_request_from_vnext(other, warmup=False), start=True
            )
            assert result.status == "applied"
            changed = replace(
                initial,
                intent=replace(
                    initial.intent,
                    languages=replace(initial.intent.languages, source_language="ja"),
                ),
            )
            request = build_self_stt_provider_request_from_vnext(changed, warmup=False)
            before = len(factory.requests)
            blocked = await runtime.handoff_provider(request, start=True)
            assert blocked.status == "failed"
            assert blocked.failure_code == "provider_resource_quarantined"
            assert blocked.failure_stage == "provider_reclaim"
            assert len(factory.requests) == before and transport.attempts == 1
            assert not unsafe.closed
            assert old in runtime._handles["self"].retained_scoped_providers
            unsafe.allow_close = True
            resumed = await runtime.handoff_provider(request, start=True)
            assert resumed.status == "applied" and unsafe.closed
            env.settings_box[0] = changed
            applied = await env.owner.apply_intent(
                build_self_capture_session_config_from_vnext(changed), enabled=True
            )
            assert applied.effective_active
            env.guarded = env.owner._vad_dispatch
            await spoken_turn(env, 0.3)
            await wait_until(
                lambda: len(transport.sockets) == 2 and finish_sent(transport.sockets[-1])
            )
            await finish(transport.sockets[-1], "Fresh Japanese source.")
            await require_final_source(env, "Fresh Japanese source.")
            assert old.cleanup_debt == 0
        finally:
            unsafe.allow_close = True


@pytest.mark.asyncio
async def test_healthy_retained_qwen_reclaim_drains_authorized_tail_before_fresh_scope():
    transport = FaultTransport([])
    factory = RequestFactory(transport)
    async with self_capture_pipeline(transport, provider_factory=factory) as env:
        returning = None
        old_socket = None
        try:
            await spoken_turn(env)
            await wait_until(lambda: bool(transport.sockets) and finish_sent(transport.sockets[0]))
            old_socket = transport.sockets[0]
            await old_socket.push(native(old_socket.task_ids[0], 1, "Already admitted old."))
            await wait_until(
                lambda: env.harness.self_owner.merge_buffer is not None
                and env.harness.self_owner.merge_buffer.parts == ["Already admitted old."]
            )
            runtime = env.harness.self_owner.local_asr_runtime
            handle = runtime._handles["self"]
            old = runtime.current_provider("self")
            initial = env.settings_box[0]
            other = replace(
                initial,
                intent=replace(initial.intent, stt=replace(initial.intent.stt, provider="soniox")),
            )
            switching = asyncio.create_task(
                runtime.handoff_provider(
                    build_self_stt_provider_request_from_vnext(other, warmup=False), start=True
                )
            )
            await wait_until(lambda: handle._pending_handoff is not None)
            await runtime.commit_handoff("self")
            assert (await switching).status == "applied"
            changed = replace(
                initial,
                intent=replace(
                    initial.intent,
                    languages=replace(initial.intent.languages, source_language="ja"),
                ),
            )
            before = len(factory.requests)
            returning = asyncio.create_task(
                runtime.handoff_provider(
                    build_self_stt_provider_request_from_vnext(changed, warmup=False), start=True
                )
            )
            await wait_until(lambda: old._closing)
            assert old.cleanup_debt == 0 and old in handle.retained_scoped_providers
            assert not returning.done() and len(factory.requests) == before
            assert not old_socket.closed and transport.attempts == 1
            await old_socket.push(native(old_socket.task_ids[0], 2, " Legitimate tail."))
            await old_socket.push(control(old_socket.task_ids[0], "task-finished"))
            assert (await returning).status == "applied" and old_socket.closed
            source = await require_final_source(env, "Already admitted old. Legitimate tail.")
            assert len(source.recognition_origins) == 2
            env.settings_box[0] = changed
            applied = await env.owner.apply_intent(
                build_self_capture_session_config_from_vnext(changed), enabled=True
            )
            assert applied.effective_active
            env.guarded = env.owner._vad_dispatch
            await spoken_turn(env, 0.3)
            await wait_until(
                lambda: len(transport.sockets) == 2 and finish_sent(transport.sockets[-1])
            )
            await finish(transport.sockets[-1], "Fresh scope source.")
            await require_final_source(env, "Fresh scope source.")
        finally:
            if old_socket is not None and not old_socket.closed:
                await finish(old_socket, "Owned fixture cleanup.")
            if returning is not None:
                await returning


async def run_retained_qwen_loopback(monkeypatch, tmp_path, scenario):
    connections = []
    tasks = []
    trace = []
    failures = []
    peak = 0

    def physical_count():
        return sum(
            socket.transport.get_extra_info("socket") is not None
            and socket.transport.get_extra_info("socket").fileno() >= 0
            for socket in connections
        )

    def mark(event, connection, task=None):
        nonlocal peak
        live = physical_count()
        peak = max(peak, live)
        trace.append(
            {
                "event": event,
                "connection": connection,
                "task": None if task is None else task["order"],
                "live": live,
            }
        )

    async def handler(socket):
        connections.append(socket)
        connection = len(connections)
        task = None
        mark("connect", connection)
        try:
            async for message in socket:
                if isinstance(message, bytes):
                    assert task is not None
                    task["pcm"].extend(message)
                    mark("pcm", connection, task)
                    continue
                packet = json.loads(message)
                if packet["header"]["action"] == "run-task":
                    task = {
                        "id": packet["header"]["task_id"],
                        "order": len(tasks) + 1,
                        "socket": socket,
                        "connection": connection,
                        "parameters": packet["payload"]["parameters"],
                        "pcm": bytearray(),
                        "finished": False,
                        "completed": False,
                    }
                    tasks.append(task)
                    mark("run", connection, task)
                    await socket.send(json.dumps(control(task["id"], "task-started")))
                else:
                    assert packet["header"]["action"] == "finish-task"
                    task["finished"] = True
                    mark("finish", connection, task)
        except Exception as exc:
            failures.append(repr(exc))
            raise
        finally:
            mark("closed", connection, task)

    initial = AppSettingsVNext()
    initial = replace(
        initial,
        intent=replace(
            initial.intent,
            stt=replace(initial.intent.stt, provider="qwen_audio"),
            languages=replace(initial.intent.languages, source_language="en"),
        ),
    )
    changed = replace(
        initial,
        intent=replace(
            initial.intent,
            stt=replace(
                initial.intent.stt,
                custom_vocabulary_enabled=True,
                custom_terms={"ja": ["Puripuly"]},
            ),
            languages=replace(initial.intent.languages, source_language="ja"),
        ),
    )
    other = replace(
        initial,
        intent=replace(initial.intent, stt=replace(initial.intent.stt, provider="soniox")),
    )
    secrets = InMemorySecretStore()
    for key in ("alibaba_api_key_beijing", "soniox_api_key"):
        secrets.set(key, "offline")
    factory = SharedSTTProviderFactory(
        secrets=secrets, clock=SystemClock(), reset_deadline_s=1, gpu_model_path=tmp_path
    )
    bound = BoundRuntimeFactory(factory)
    llm = FakeLLMProvider(delay_s=0)
    harness = compose_translation_test_harness(
        stt=None,
        llm=llm,
        osc=RecordingOscQueue(),
        source_language="en",
        target_language="ko",
        low_latency_mode=False,
        local_asr_provider_runtime_factory=bound,
    )
    runtime = bound.runtime
    ledger = PeerAudioSegmentLedger(activation_generation=1, settings=segment_settings())
    cursor = 0
    expected_pcm = []
    expected_sources = []
    cycles = []

    async def handoff(settings):
        request = build_self_stt_provider_request_from_vnext(settings, warmup=False)
        result = await runtime.handoff_provider(request, start=True)
        assert result.status == "applied"
        return request

    def sources():
        return [
            event.payload
            for event in harness.ui_events._queue
            if event.type is UIEventType.TRANSCRIPT_FINAL
        ]

    async def complete(task, text, tail):
        await task["socket"].send(json.dumps(native(task["id"], 1, text)))
        await task["socket"].send(json.dumps(native(task["id"], 2, tail)))
        await task["socket"].send(json.dumps(control(task["id"], "task-finished")))
        task["completed"] = True
        mark("complete", task["connection"], task)

    async def pair(request, label, values):
        nonlocal cursor
        settings = replace(
            segment_settings("qwen_audio"),
            provider_signature=request.provider_signature,
            runtime_signature=request.runtime_signature,
            source_language=request.config.source_language,
            expected_languages=(request.config.source_language,),
        )
        ledger.rebind(activation_generation=1, settings=settings)
        offset = len(tasks)
        for value in values:
            identity = uuid4()
            pcm = np.full(160, value, dtype=np.float32)
            for index in range(3):
                now = asyncio.get_running_loop().time()
                capture = replace(
                    span(cursor // 160, cursor, cursor + 160),
                    source_start_monotonic_s=now - 0.01,
                    source_end_monotonic_s=now,
                )
                event = (
                    SpeechStart(
                        identity, np.empty(0, dtype=np.float32), pcm, chunk_capture=(capture,)
                    )
                    if index == 0
                    else SpeechChunk(identity, pcm, chunk_capture=(capture,))
                )
                cursor += 160
                owned = ledger.observe_vad_event(event, now_monotonic_s=now)
                await runtime.handle_owned_vad_event("self", owned)
            owned = ledger.observe_vad_event(
                SpeechEnd(identity, trailing_silence_ms=224, reason="silence"),
                now_monotonic_s=asyncio.get_running_loop().time(),
            )
            await runtime.handle_owned_vad_event("self", owned)
            expected_pcm.append(float32_to_pcm16le_bytes(np.full(480, value, dtype=np.float32)))
        await wait_until(
            lambda: len(tasks) == offset + 2 and all(t["finished"] for t in tasks[offset:])
        )
        first, second = tasks[offset:]
        await complete(second, f"{label} B.", " Tail B.")
        await asyncio.sleep(0)
        assert [item.text for item in sources()] == expected_sources
        await complete(first, f"{label} A.", " Tail A.")
        expected_sources.extend([f"{label} A. Tail A.", f"{label} B. Tail B."])
        await wait_until(lambda: [item.text for item in sources()] == expected_sources)
        await harness.translation_turns.wait_for_idle()
        await wait_until(
            lambda: all(
                any(
                    event.type is UIEventType.TRANSLATION_DONE
                    and event.utterance_id == source.utterance_id
                    and event.payload.text == llm.response_text
                    for event in harness.ui_events._queue
                )
                for source in sources()
            )
        )
        assert [call["text"] for call in llm.calls] == expected_sources
        assert all(len(source.recognition_origins) == 2 for source in sources())
        assert [bytes(task["pcm"]) for task in tasks] == expected_pcm
        assert peak <= 2
        return first, second

    async with serve(handler, "127.0.0.1", 0) as server:
        endpoint = f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}"

        def loopback_connect(_uri, **kwargs):
            return connect(endpoint, **{**kwargs, "proxy": None})

        monkeypatch.setattr(network_clients, "external_websocket_connect", loopback_connect)
        try:
            await harness.start()
            request = await handoff(initial)
            assert not tasks and not connections
            original = await pair(request, "Initial", (0.1, 0.2))
            original_sockets = [task["socket"] for task in original]
            original_engine = runtime.current_provider("self")
            assert original_engine.cleanup_debt == 0 and physical_count() == 2
            if scenario == "cached_scope":
                await handoff(other)
                request = await handoff(changed)
                await handoff(other)
                request = await handoff(changed)
                assert len(tasks) == 2
                fresh = await pair(request, "Japanese", (0.3, 0.4))
                assert all(socket.close_code is not None for socket in original_sockets)
                assert all(
                    task["parameters"]["language_hints"] == ["ja"]
                    and task["parameters"]["vocabulary"] == {"Puripuly": 4}
                    for task in fresh
                )
                fresh_connect = next(
                    index
                    for index, event in enumerate(trace)
                    if event["event"] == "connect" and event["connection"] == fresh[0]["connection"]
                )
                assert all(
                    any(
                        index < fresh_connect
                        and event["event"] == "closed"
                        and event["connection"] == task["connection"]
                        for index, event in enumerate(trace)
                    )
                    for task in original
                )
            else:
                for cycle in range(1, 7):
                    await handoff(other)
                    request = await handoff(initial)
                    pair_tasks = await pair(request, f"Return {cycle}", (0.25, 0.35))
                    if cycle <= 4:
                        assert [task["socket"] for task in pair_tasks] == original_sockets
                        assert physical_count() == 2 and len(connections) == 2
                    cycles.append(
                        {
                            "cycle": cycle,
                            "connections": len(connections),
                            "physical": physical_count(),
                            "peak": peak,
                        }
                    )
                assert len(connections) > 2
            assert all(task["parameters"]["max_sentence_silence"] == 6000 for task in tasks)
        finally:
            for task in tasks:
                if not task["completed"] and task["socket"].close_code is None:
                    await complete(task, "Fixture cleanup.", "")
            await harness.stop()
            for socket in connections:
                await asyncio.wait_for(socket.wait_closed(), timeout=1)
        assert physical_count() == 0 and not failures
        assert peak == 2
        return {
            "scenario": scenario,
            "peak_physical": peak,
            "final_physical": physical_count(),
            "connections": len(connections),
            "tasks": len(tasks),
            "pcm_sha256": [hashlib.sha256(pcm).hexdigest() for pcm in expected_pcm],
            "native_parameters": [task["parameters"] for task in tasks],
            "sources": expected_sources,
            "completed_translations": len(expected_sources),
            "cycles": cycles,
            "trace": trace,
            "server_failures": failures,
        }


@pytest.mark.asyncio
@pytest.mark.parametrize("scenario", ["cached_scope", "commit_cap_crossing"])
async def test_healthy_retained_qwen_readmission_bounds_physical_sockets_and_completes_consumers(
    monkeypatch, tmp_path, scenario
):
    report = await run_retained_qwen_loopback(monkeypatch, tmp_path, scenario)
    print("QWEN_RETAINED_RESOURCE_BOUND " + json.dumps(report, sort_keys=True))
