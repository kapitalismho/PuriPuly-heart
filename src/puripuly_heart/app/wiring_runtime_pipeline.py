from __future__ import annotations

import contextlib
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path

from puripuly_heart.app.services.peer_application import PeerApplicationOwner
from puripuly_heart.app.wiring_llm_factory import create_llm_provider
from puripuly_heart.app.wiring_managed_account import ManagedOpenRouterReleaseRuntime
from puripuly_heart.app.wiring_provider_runtime import (
    project_translation_runtime_settings,
)
from puripuly_heart.app.wiring_secrets_factory import create_secret_store
from puripuly_heart.app.wiring_stt_factory import (
    build_self_capture_session_config,
    build_self_stt_provider_request,
)
from puripuly_heart.app.wiring_translation_runtime_configuration import (
    build_translation_runtime_config,
)
from puripuly_heart.config.settings import AppSettings, STTProviderName
from puripuly_heart.core.audio.gate import VrcMicAudioGate
from puripuly_heart.core.clock import Clock
from puripuly_heart.core.llm.provider import LLMProvider
from puripuly_heart.core.local_asr_provider_runtime import (
    LocalASRProviderRuntimeFactoryPort,
)
from puripuly_heart.core.orchestrator.configuration import (
    TranslationRuntimeConfigurationOwner,
)
from puripuly_heart.core.orchestrator.hub import ClientHub
from puripuly_heart.core.osc.chatbox_paginator import ChatboxPaginator
from puripuly_heart.core.osc.receiver import VrcMicState
from puripuly_heart.core.osc.udp_sender import VrchatOscUdpSender
from puripuly_heart.core.runtime.peer_channel import PeerCaptureSessionOwner
from puripuly_heart.core.runtime.self_capture import SelfCaptureSessionOwner


@dataclass(slots=True)
class RuntimePipelineResources:
    llm: LLMProvider | None = None
    sender: VrchatOscUdpSender | None = None
    hub: ClientHub | None = None
    self_capture: SelfCaptureSessionOwner | None = None
    peer_capture: PeerCaptureSessionOwner | None = None

    @property
    def has_resources(self) -> bool:
        return any(
            resource is not None
            for resource in (
                self.llm,
                self.sender,
                self.hub,
                self.self_capture,
                self.peer_capture,
            )
        )

    async def close(self) -> None:
        failures: list[BaseException] = []
        for name in ("self_capture", "peer_capture"):
            resource = getattr(self, name)
            if resource is None:
                continue
            try:
                await resource.close()
            except BaseException as exc:
                failures.append(exc)
            else:
                setattr(self, name, None)
        hub = self.hub
        if hub is not None:
            try:
                await hub.stop()
            except BaseException as exc:
                failures.append(exc)
            else:
                self.hub = None
        llm = self.llm
        if llm is not None:
            try:
                await llm.close()
            except BaseException as exc:
                failures.append(exc)
            else:
                self.llm = None
        sender = self.sender
        if sender is not None:
            try:
                sender.close()
            except BaseException as exc:
                failures.append(exc)
            else:
                self.sender = None
        if failures:
            raise BaseExceptionGroup("runtime pipeline cleanup failed", failures)


@dataclass(frozen=True, slots=True)
class RuntimePipelineComponents:
    sender: VrchatOscUdpSender
    osc: ChatboxPaginator
    hub: ClientHub
    self_capture: SelfCaptureSessionOwner
    peer_capture: PeerCaptureSessionOwner
    vrc_mic_state: VrcMicState
    vrc_mic_audio_gate: VrcMicAudioGate
    prepare_self_provider: bool
    translation_runtime_configuration: TranslationRuntimeConfigurationOwner
    resources: RuntimePipelineResources = field(repr=False)

    async def close(self) -> None:
        await self.resources.close()


@dataclass(slots=True)
class RuntimePipelineHandle:
    current: RuntimePipelineComponents | None = field(
        init=False,
        default=None,
        repr=False,
    )
    sender: VrchatOscUdpSender | None = field(init=False, default=None)
    osc: ChatboxPaginator | None = field(init=False, default=None)
    translation_runtime_configuration: TranslationRuntimeConfigurationOwner | None = field(
        init=False,
        default=None,
    )
    hub: ClientHub | None = field(init=False, default=None)
    self_capture: SelfCaptureSessionOwner | None = field(init=False, default=None)
    vrc_mic_state: VrcMicState | None = field(init=False, default=None)
    vrc_mic_audio_gate: VrcMicAudioGate | None = field(init=False, default=None)

    def install(self, components: RuntimePipelineComponents) -> None:
        self.current = components
        self.sender = components.sender
        self.osc = components.osc
        self.translation_runtime_configuration = components.translation_runtime_configuration
        self.hub = components.hub
        self.self_capture = components.self_capture
        self.vrc_mic_state = components.vrc_mic_state
        self.vrc_mic_audio_gate = components.vrc_mic_audio_gate

    def clear(self, components: RuntimePipelineComponents | None = None) -> None:
        if components is None or self.current is components:
            self.current = None
            self.sender = None
            self.osc = None
            self.translation_runtime_configuration = None
            self.hub = None
            self.self_capture = None
            self.vrc_mic_state = None
            self.vrc_mic_audio_gate = None


@dataclass(slots=True)
class RuntimePipelineLauncher:
    config_path: Path
    clock: Clock
    runtime_logging: object
    managed_release: ManagedOpenRouterReleaseRuntime
    managed_delegate_ready: Callable[[], None]
    local_asr_factory: Callable[
        [object],
        LocalASRProviderRuntimeFactoryPort,
    ]
    self_capture_factory: Callable[
        [ClientHub, VrcMicAudioGate],
        SelfCaptureSessionOwner,
    ]
    peer_capture_factory: Callable[[ClientHub], PeerCaptureSessionOwner]
    previous_self_capture: Callable[[], SelfCaptureSessionOwner | None]
    component_sink: Callable[[RuntimePipelineComponents], None]
    peer_application: Callable[[], PeerApplicationOwner]
    configure_vrc_mic: Callable[..., Awaitable[None]]
    stt_failure_sink: Callable[[str], None]
    cleanup_failure_sink: Callable[[str, BaseException], None]
    failed_resources: RuntimePipelineResources | None = field(
        init=False,
        default=None,
        repr=False,
    )

    async def retry_failed_cleanup(self) -> None:
        resources = self.failed_resources
        if resources is None:
            return
        try:
            await resources.close()
        except BaseException as exc:
            with contextlib.suppress(Exception):
                self.cleanup_failure_sink(
                    "Runtime pipeline cleanup retry failed",
                    exc,
                )
            raise
        self.failed_resources = None

    async def close(self) -> None:
        await self.retry_failed_cleanup()

    async def launch(
        self,
        settings: AppSettings,
        *,
        vrc_mic_state: VrcMicState | None,
        vrc_mic_audio_gate: VrcMicAudioGate | None,
        receiver_active: bool,
    ) -> RuntimePipelineComponents:
        await self.retry_failed_cleanup()
        resources = RuntimePipelineResources()
        try:
            pipeline = await compose_runtime_pipeline(
                settings=settings,
                config_path=self.config_path,
                clock=self.clock,
                runtime_logging=self.runtime_logging,
                managed_release=self.managed_release,
                managed_delegate_ready=self.managed_delegate_ready,
                local_asr_factory=self.local_asr_factory,
                self_capture_factory=self.self_capture_factory,
                peer_capture_factory=self.peer_capture_factory,
                vrc_mic_state=vrc_mic_state,
                vrc_mic_audio_gate=vrc_mic_audio_gate,
                receiver_active=receiver_active,
                stt_failure_sink=self.stt_failure_sink,
                resources=resources,
            )
            if pipeline.prepare_self_provider:
                snapshot = await pipeline.self_capture.prepare_provider(
                    build_self_capture_session_config(settings)
                )
                if snapshot.provider_status.value != "ready":
                    self.stt_failure_sink("STT backend not available")
            previous = self.previous_self_capture()
            if previous is not None:
                await previous.close()
            self.component_sink(pipeline)
            peer = self.peer_application()
            await peer.replace_runtime(pipeline.peer_capture)
            peer.last_intent_enabled = settings.ui.peer_translation_enabled
            await self.configure_vrc_mic(enabled=settings.osc.vrc_mic_intercept)
            return pipeline
        except BaseException as exc:
            try:
                await resources.close()
            except BaseException as cleanup_exc:
                self.failed_resources = resources
                with contextlib.suppress(Exception):
                    self.cleanup_failure_sink(
                        "Runtime pipeline launch cleanup failed",
                        cleanup_exc,
                    )
                raise BaseExceptionGroup(
                    "runtime pipeline launch and cleanup failed",
                    [exc, cleanup_exc],
                ) from exc
            raise


async def compose_runtime_pipeline(
    *,
    settings: AppSettings,
    config_path: Path,
    clock: Clock,
    runtime_logging: object,
    managed_release: ManagedOpenRouterReleaseRuntime,
    managed_delegate_ready: Callable[[], None],
    local_asr_factory: Callable[
        [object],
        LocalASRProviderRuntimeFactoryPort,
    ],
    self_capture_factory: Callable[
        [ClientHub, VrcMicAudioGate],
        SelfCaptureSessionOwner,
    ],
    peer_capture_factory: Callable[[ClientHub], PeerCaptureSessionOwner],
    vrc_mic_state: VrcMicState | None,
    vrc_mic_audio_gate: VrcMicAudioGate | None,
    receiver_active: bool,
    stt_failure_sink: Callable[[str], None],
    resources: RuntimePipelineResources | None = None,
) -> RuntimePipelineComponents:
    owned_resources = resources is None
    pipeline_resources = resources or RuntimePipelineResources()
    try:
        return await _compose_runtime_pipeline(
            settings=settings,
            config_path=config_path,
            clock=clock,
            runtime_logging=runtime_logging,
            managed_release=managed_release,
            managed_delegate_ready=managed_delegate_ready,
            local_asr_factory=local_asr_factory,
            self_capture_factory=self_capture_factory,
            peer_capture_factory=peer_capture_factory,
            vrc_mic_state=vrc_mic_state,
            vrc_mic_audio_gate=vrc_mic_audio_gate,
            receiver_active=receiver_active,
            stt_failure_sink=stt_failure_sink,
            resources=pipeline_resources,
        )
    except BaseException as exc:
        if not owned_resources:
            raise
        try:
            await pipeline_resources.close()
        except BaseException as cleanup_exc:
            raise BaseExceptionGroup(
                "runtime pipeline composition and cleanup failed",
                [exc, cleanup_exc],
            ) from exc
        raise


async def _compose_runtime_pipeline(
    *,
    settings: AppSettings,
    config_path: Path,
    clock: Clock,
    runtime_logging: object,
    managed_release: ManagedOpenRouterReleaseRuntime,
    managed_delegate_ready: Callable[[], None],
    local_asr_factory: Callable[[object], LocalASRProviderRuntimeFactoryPort],
    self_capture_factory: Callable[
        [ClientHub, VrcMicAudioGate],
        SelfCaptureSessionOwner,
    ],
    peer_capture_factory: Callable[[ClientHub], PeerCaptureSessionOwner],
    vrc_mic_state: VrcMicState | None,
    vrc_mic_audio_gate: VrcMicAudioGate | None,
    receiver_active: bool,
    stt_failure_sink: Callable[[str], None],
    resources: RuntimePipelineResources,
) -> RuntimePipelineComponents:
    secrets = create_secret_store(settings.secrets, config_path=config_path)
    await managed_release.rebuild(secrets=secrets)

    llm = None
    with contextlib.suppress(Exception):
        llm = create_llm_provider(
            settings,
            secrets=secrets,
            managed_release_service=managed_release.service,
            managed_delegate_ready=managed_delegate_ready,
            runtime_logging=runtime_logging,
        )
        resources.llm = llm

    prepare_self_provider = settings.provider.stt != STTProviderName.LOCAL_QWEN_GPU
    if prepare_self_provider:
        try:
            build_self_stt_provider_request(settings)
        except Exception:
            prepare_self_provider = False
            stt_failure_sink("STT backend not available")

    sender = VrchatOscUdpSender(
        host=settings.osc.host,
        port=settings.osc.port,
        chatbox_address=settings.osc.chatbox_address,
        chatbox_send=settings.osc.chatbox_send,
        chatbox_clear=settings.osc.chatbox_clear,
    )
    resources.sender = sender
    osc = ChatboxPaginator(
        sender=sender,
        clock=clock,
        max_chars=settings.osc.chatbox_max_chars,
        runtime_logging=runtime_logging,
    )
    translation_runtime_configuration = TranslationRuntimeConfigurationOwner(
        build_translation_runtime_config(
            project_translation_runtime_settings(settings),
            fallback_transcript_only=True,
            translation_enabled=True,
            peer_translation_enabled=False,
            integrated_context_enabled=True,
        )
    )
    hub = ClientHub(
        stt=None,
        llm=llm,
        osc=osc,
        peer_stt=None,
        clock=clock,
        runtime_logging=runtime_logging,
        local_asr_provider_runtime_factory=local_asr_factory(secrets),
        translation_runtime_configuration=translation_runtime_configuration,
    )
    resources.hub = hub
    resources.llm = None
    state = vrc_mic_state or VrcMicState()
    gate = vrc_mic_audio_gate
    if gate is None:
        gate = VrcMicAudioGate(
            state=state,
            enabled=settings.osc.vrc_mic_intercept,
        )
    else:
        gate.state = state
        gate.set_enabled(settings.osc.vrc_mic_intercept)
    gate.set_receiver_active(receiver_active)
    gate.reset()

    self_capture = self_capture_factory(hub, gate)
    resources.self_capture = self_capture
    peer_capture = peer_capture_factory(hub)
    resources.peer_capture = peer_capture
    return RuntimePipelineComponents(
        sender=sender,
        osc=osc,
        hub=hub,
        self_capture=self_capture,
        peer_capture=peer_capture,
        vrc_mic_state=state,
        vrc_mic_audio_gate=gate,
        prepare_self_provider=prepare_self_provider,
        translation_runtime_configuration=translation_runtime_configuration,
        resources=resources,
    )


__all__ = [
    "RuntimePipelineComponents",
    "RuntimePipelineHandle",
    "RuntimePipelineLauncher",
    "RuntimePipelineResources",
    "compose_runtime_pipeline",
]
