from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import replace
from types import SimpleNamespace

import pytest

from puripuly_heart.app.adapters.peer_capture.peer_capture_provider import (
    PeerCaptureProviderAdapter,
)
from puripuly_heart.app.wiring.wiring_stt_factory import (
    build_peer_capture_session_config_from_vnext,
    build_peer_stt_provider_request,
    build_self_capture_session_config_from_vnext,
    build_self_stt_provider_request_from_vnext,
)
from puripuly_heart.config.settings_vnext.schema import AppSettingsVNext
from puripuly_heart.core.runtime.provider_handle import ProviderResourceQuarantinedError
from puripuly_heart.core.stt.scoped_engine import ScopedRecognitionEngine
from puripuly_heart.domain.events import UIEventType
from tests.core.runtime.test_peer_capture_session import make_owner
from tests.core.runtime.test_self_capture_session import LoopHarness
from tests.core.test_stt_scoped_engine import ControlledScopedSession, wait_until, watchdogs
from tests.providers.test_qwen_audio import ScriptedSocket, control, native
from tests.providers.test_qwen_audio_configuration import ControlledSmartTurn, RuntimeSink
from tests.providers.test_qwen_audio_delivery_recovery import (
    HeldManualProvider,
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
