from __future__ import annotations

import asyncio
import contextlib
import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path

from puripuly_heart.app.ports.capture_vad_runtime import (
    PeerCaptureVadEventRuntime,
    SelfCaptureVadEventRuntime,
)
from puripuly_heart.app.ports.provider_channel_runtime import ProviderChannelResetPort
from puripuly_heart.app.ports.runtime_pipeline_lifecycle import (
    RuntimePipelineCloseCallbacks,
    RuntimePipelineStartCallbacks,
)
from puripuly_heart.app.services.managed_gemma_translation import ManagedGemmaTranslationOwner
from puripuly_heart.app.services.peer_application import PeerApplicationOwner
from puripuly_heart.config.paths import default_http_extensions_dir
from puripuly_heart.config.settings import AppSettings, STTProviderName, TranslationModel
from puripuly_heart.core.audio.gate import VrcMicAudioGate
from puripuly_heart.core.clock import Clock
from puripuly_heart.core.http_extensions import HttpExtensionRegistry
from puripuly_heart.core.local_asr_provider_runtime import (
    LocalASRProviderRuntimeCallbacks,
    LocalASRProviderRuntimeFactoryPort,
    LocalASRProviderRuntimePort,
    ProviderRuntimeChannel,
)
from puripuly_heart.core.orchestrator.channel_runtime import ChannelRuntime
from puripuly_heart.core.orchestrator.configuration import (
    TranslationRuntimeConfigurationOwner,
)
from puripuly_heart.core.orchestrator.context import ContextResolver
from puripuly_heart.core.orchestrator.peer_translation_channel import (
    PeerTranslationChannelOwner,
)
from puripuly_heart.core.orchestrator.self_translation_channel import (
    SelfTranslationChannelOwner,
)
from puripuly_heart.core.orchestrator.translation_channel_callbacks import (
    TranslationChannelOwnerCallbacks,
)
from puripuly_heart.core.orchestrator.translation_diagnostics import (
    TranslationLatencyDiagnosticsOwner,
)
from puripuly_heart.core.orchestrator.translation_output_projection import (
    TranslationOutputProjectionOwner,
    TranslationUiMessageQueue,
)
from puripuly_heart.core.orchestrator.translation_request import TranslationRequestOwner
from puripuly_heart.core.orchestrator.translation_turn import (
    TranslationTurnLifecycleOwner,
)
from puripuly_heart.core.osc.chatbox_paginator import ChatboxPaginator
from puripuly_heart.core.osc.receiver import VrcMicState
from puripuly_heart.core.osc.udp_sender import VrchatOscUdpSender
from puripuly_heart.core.runtime.output import OutputRuntime
from puripuly_heart.core.runtime.peer_channel import PeerCaptureSessionOwner
from puripuly_heart.core.runtime.provider_handle import ProviderRuntimeHandle
from puripuly_heart.core.runtime.self_capture import SelfCaptureSessionOwner
from puripuly_heart.core.runtime.stt_session_projection import SttSessionStateProjection
from puripuly_heart.domain.events import UIEvent

from .wiring_managed_account import ManagedOpenRouterReleaseRuntime
from .wiring_managed_gemma import noop_managed_gemma_release
from .wiring_provider_runtime import (
    project_translation_runtime_settings,
)
from .wiring_secrets_factory import create_secret_store
from .wiring_stt_factory import (
    build_self_capture_session_config,
    build_self_stt_provider_request,
)
from .wiring_translation_backend import create_translation_backend
from .wiring_translation_runtime_configuration import (
    build_translation_runtime_config,
)


@dataclass(slots=True)
class RuntimePipelineChannelResetRouter:
    self_owner: SelfTranslationChannelOwner = field(repr=False)
    peer_owner: ProviderChannelResetPort = field(repr=False)

    async def reset_provider_channel(self, channel: ProviderRuntimeChannel) -> None:
        if channel == "self":
            await self.self_owner.reset_provider_channel(channel)
            return
        if channel == "peer":
            await self.peer_owner.reset_provider_channel(channel)
            return
        raise ValueError(f"invalid channel: {channel!r}")


@dataclass(slots=True)
class RuntimePipelineResourceOwner:
    pending_llm: object | None = None
    sender: VrchatOscUdpSender | None = None
    output_runtime: OutputRuntime | None = None
    self_runtime: ChannelRuntime | None = None
    peer_runtime: ChannelRuntime | None = None
    self_translation_channel: SelfTranslationChannelOwner | None = None
    peer_translation_channel: PeerTranslationChannelOwner | None = None
    translation_turns: TranslationTurnLifecycleOwner | None = None
    local_asr_runtime: LocalASRProviderRuntimePort | None = None
    llm_runtime: ProviderRuntimeHandle | None = None
    self_capture: SelfCaptureSessionOwner | None = None
    peer_capture: PeerCaptureSessionOwner | None = None
    self_ingress_open: bool = False
    peer_ingress_open: bool = False
    start_callbacks: RuntimePipelineStartCallbacks = field(init=False)
    close_callbacks: RuntimePipelineCloseCallbacks = field(init=False)

    def __post_init__(self) -> None:
        self.start_callbacks = RuntimePipelineStartCallbacks(
            start_output=self.start_output,
            open_self_ingress=self.open_self_ingress,
            open_peer_ingress=self.open_peer_ingress,
            start_translation_turns=self.start_translation_turns,
            start_local_asr=self.start_local_asr,
        )
        self.close_callbacks = RuntimePipelineCloseCallbacks(
            close_self_capture=self.close_self_capture,
            close_peer_capture=self.close_peer_capture,
            close_self_ingress=self.close_self_ingress,
            close_peer_ingress=self.close_peer_ingress,
            close_translation_turns=self.close_translation_turns,
            close_output=self.close_output,
            close_self_channel=self.close_self_channel,
            close_peer_channel=self.close_peer_channel,
            close_local_asr=self.close_local_asr,
            close_llm=self.close_llm,
            close_sender=self.close_sender,
        )

    @property
    def has_resources(self) -> bool:
        return (
            any(
                resource is not None
                for resource in (
                    self.pending_llm,
                    self.sender,
                    self.output_runtime,
                    self.self_runtime,
                    self.peer_runtime,
                    self.self_translation_channel,
                    self.peer_translation_channel,
                    self.translation_turns,
                    self.local_asr_runtime,
                    self.llm_runtime,
                    self.self_capture,
                    self.peer_capture,
                )
            )
            or self.self_ingress_open
            or self.peer_ingress_open
        )

    async def start_output(self, auto_flush_chatbox: bool) -> None:
        runtime = self.output_runtime
        if runtime is None:
            raise RuntimeError("runtime pipeline output owner is unavailable")
        await runtime.start(auto_flush_chatbox=auto_flush_chatbox)

    async def open_self_ingress(self) -> None:
        channel_owner = self.self_translation_channel
        turns = self.translation_turns
        if channel_owner is None:
            raise RuntimeError("runtime pipeline Self translation owner is unavailable")
        if turns is None:
            raise RuntimeError("runtime pipeline translation-turn owner is unavailable")
        await channel_owner.open_ingress()
        try:
            await turns.open_channel_ingress("self")
        except BaseException:
            await channel_owner.close_ingress()
            raise
        self.self_ingress_open = True

    async def open_peer_ingress(self) -> None:
        channel_owner = self.peer_translation_channel
        turns = self.translation_turns
        if channel_owner is None:
            raise RuntimeError("runtime pipeline Peer translation owner is unavailable")
        if turns is None:
            raise RuntimeError("runtime pipeline translation-turn owner is unavailable")
        await channel_owner.open_ingress()
        try:
            await turns.open_channel_ingress("peer")
        except BaseException:
            await channel_owner.close_ingress()
            raise
        self.peer_ingress_open = True

    async def start_translation_turns(self) -> None:
        owner = self.translation_turns
        if owner is None:
            raise RuntimeError("runtime pipeline translation-turn owner is unavailable")
        await owner.start()

    async def start_local_asr(self) -> None:
        owner = self.local_asr_runtime
        if owner is None:
            raise RuntimeError("runtime pipeline Local ASR owner is unavailable")
        await owner.start()

    async def close_self_capture(self) -> None:
        owner = self.self_capture
        if owner is None:
            return
        await owner.close()
        if self.self_capture is owner:
            self.self_capture = None

    async def close_peer_capture(self) -> None:
        owner = self.peer_capture
        if owner is None:
            return
        await owner.close()
        if self.peer_capture is owner:
            self.peer_capture = None

    async def close_self_ingress(self) -> None:
        if not self.self_ingress_open:
            return
        channel_owner = self.self_translation_channel
        turns = self.translation_turns
        if channel_owner is not None:
            await channel_owner.close_ingress()
        if turns is not None:
            await turns.close_channel_ingress("self")
        self.self_ingress_open = False

    async def close_peer_ingress(self) -> None:
        if not self.peer_ingress_open:
            return
        channel_owner = self.peer_translation_channel
        turns = self.translation_turns
        if channel_owner is not None:
            await channel_owner.close_ingress()
        if turns is not None:
            await turns.close_channel_ingress("peer")
        self.peer_ingress_open = False

    async def close_translation_turns(self) -> None:
        owner = self.translation_turns
        if owner is None:
            return
        await owner.close()
        if self.translation_turns is owner:
            self.translation_turns = None

    async def close_output(self) -> None:
        owner = self.output_runtime
        if owner is None:
            return
        await owner.close()
        if self.output_runtime is owner:
            self.output_runtime = None

    async def close_self_channel(self) -> None:
        owner = self.self_translation_channel
        if owner is None:
            runtime = self.self_runtime
            if runtime is not None:
                await runtime.reset_runtime_state()
                if self.self_runtime is runtime:
                    self.self_runtime = None
            return
        await owner.close()
        if self.self_translation_channel is owner:
            self.self_translation_channel = None
        if self.self_runtime is owner.runtime:
            self.self_runtime = None

    async def close_peer_channel(self) -> None:
        owner = self.peer_translation_channel
        if owner is None:
            runtime = self.peer_runtime
            if runtime is not None:
                await runtime.reset_runtime_state()
                if self.peer_runtime is runtime:
                    self.peer_runtime = None
            return
        await owner.close()
        if self.peer_translation_channel is owner:
            self.peer_translation_channel = None
        if self.peer_runtime is owner.runtime:
            self.peer_runtime = None

    async def close_local_asr(self) -> None:
        owner = self.local_asr_runtime
        if owner is None:
            return
        await owner.close()
        if self.local_asr_runtime is owner:
            self.local_asr_runtime = None

    async def close_llm(self) -> None:
        owner = self.llm_runtime
        if owner is not None:
            await owner.close()
            if self.llm_runtime is owner:
                self.llm_runtime = None
            self.pending_llm = None
            return
        provider = self.pending_llm
        if provider is None:
            return
        await provider.close()
        if self.pending_llm is provider:
            self.pending_llm = None

    def close_sender(self) -> None:
        owner = self.sender
        if owner is None:
            return
        owner.close()
        if self.sender is owner:
            self.sender = None

    async def close(self) -> None:
        failures: list[BaseException] = []
        callbacks = self.close_callbacks
        for callback in (
            callbacks.close_self_capture,
            callbacks.close_peer_capture,
            callbacks.close_self_ingress,
            callbacks.close_peer_ingress,
            callbacks.close_translation_turns,
            callbacks.close_output,
            callbacks.close_self_channel,
            callbacks.close_peer_channel,
            callbacks.close_local_asr,
            callbacks.close_llm,
            callbacks.close_sender,
        ):
            try:
                result = callback()
                if inspect.isawaitable(result):
                    await result
            except BaseException as exc:
                failures.append(exc)
        if failures:
            raise BaseExceptionGroup("runtime pipeline cleanup failed", failures)


@dataclass(frozen=True, slots=True)
class RuntimePipelineComponents:
    sender: VrchatOscUdpSender
    osc: ChatboxPaginator
    self_capture: SelfCaptureSessionOwner
    peer_capture: PeerCaptureSessionOwner
    vrc_mic_state: VrcMicState
    vrc_mic_audio_gate: VrcMicAudioGate
    prepare_self_provider: bool
    translation_runtime_configuration: TranslationRuntimeConfigurationOwner
    output_runtime: OutputRuntime
    self_runtime: ChannelRuntime
    peer_runtime: ChannelRuntime
    translation_turns: TranslationTurnLifecycleOwner
    local_asr_runtime: LocalASRProviderRuntimePort
    llm_runtime: ProviderRuntimeHandle
    context_resolver: ContextResolver
    translation_diagnostics: TranslationLatencyDiagnosticsOwner
    translation_output_projection: TranslationOutputProjectionOwner
    translation_requests: TranslationRequestOwner
    self_translation_channel: SelfTranslationChannelOwner
    peer_translation_channel: PeerTranslationChannelOwner
    channel_reset: RuntimePipelineChannelResetRouter
    stt_sessions: SttSessionStateProjection
    ui_events: asyncio.Queue[UIEvent]
    resource_owner: RuntimePipelineResourceOwner = field(repr=False)
    start_callbacks: RuntimePipelineStartCallbacks
    close_callbacks: RuntimePipelineCloseCallbacks


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
    output_runtime: OutputRuntime | None = field(init=False, default=None)
    self_runtime: ChannelRuntime | None = field(init=False, default=None)
    peer_runtime: ChannelRuntime | None = field(init=False, default=None)
    translation_turns: TranslationTurnLifecycleOwner | None = field(init=False, default=None)
    local_asr_runtime: LocalASRProviderRuntimePort | None = field(init=False, default=None)
    llm_runtime: ProviderRuntimeHandle | None = field(init=False, default=None)
    context_resolver: ContextResolver | None = field(init=False, default=None)
    translation_diagnostics: TranslationLatencyDiagnosticsOwner | None = field(
        init=False,
        default=None,
    )
    translation_output_projection: TranslationOutputProjectionOwner | None = field(
        init=False,
        default=None,
    )
    translation_requests: TranslationRequestOwner | None = field(init=False, default=None)
    self_translation_channel: SelfTranslationChannelOwner | None = field(
        init=False,
        default=None,
    )
    peer_translation_channel: PeerTranslationChannelOwner | None = field(
        init=False,
        default=None,
    )
    channel_reset: RuntimePipelineChannelResetRouter | None = field(
        init=False,
        default=None,
    )
    stt_sessions: SttSessionStateProjection | None = field(init=False, default=None)
    ui_events: asyncio.Queue[UIEvent] | None = field(init=False, default=None)
    self_capture: SelfCaptureSessionOwner | None = field(init=False, default=None)
    vrc_mic_state: VrcMicState | None = field(init=False, default=None)
    vrc_mic_audio_gate: VrcMicAudioGate | None = field(init=False, default=None)

    def install(self, components: RuntimePipelineComponents) -> None:
        self.current = components
        self.sender = components.sender
        self.osc = components.osc
        self.translation_runtime_configuration = components.translation_runtime_configuration
        self.output_runtime = components.output_runtime
        self.self_runtime = components.self_runtime
        self.peer_runtime = components.peer_runtime
        self.translation_turns = components.translation_turns
        self.local_asr_runtime = components.local_asr_runtime
        self.llm_runtime = components.llm_runtime
        self.context_resolver = components.context_resolver
        self.translation_diagnostics = components.translation_diagnostics
        self.translation_output_projection = components.translation_output_projection
        self.translation_requests = components.translation_requests
        self.self_translation_channel = components.self_translation_channel
        self.peer_translation_channel = components.peer_translation_channel
        self.channel_reset = components.channel_reset
        self.stt_sessions = components.stt_sessions
        self.ui_events = components.ui_events
        self.self_capture = components.self_capture
        self.vrc_mic_state = components.vrc_mic_state
        self.vrc_mic_audio_gate = components.vrc_mic_audio_gate

    def clear(self, components: RuntimePipelineComponents | None = None) -> None:
        if components is None or self.current is components:
            self.current = None
            self.sender = None
            self.osc = None
            self.translation_runtime_configuration = None
            self.output_runtime = None
            self.self_runtime = None
            self.peer_runtime = None
            self.translation_turns = None
            self.local_asr_runtime = None
            self.llm_runtime = None
            self.context_resolver = None
            self.translation_diagnostics = None
            self.translation_output_projection = None
            self.translation_requests = None
            self.self_translation_channel = None
            self.peer_translation_channel = None
            self.channel_reset = None
            self.stt_sessions = None
            self.ui_events = None
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
        [
            SelfCaptureVadEventRuntime,
            LocalASRProviderRuntimePort,
            ProviderChannelResetPort,
            VrcMicAudioGate,
        ],
        SelfCaptureSessionOwner,
    ]
    peer_capture_factory: Callable[
        [
            PeerCaptureVadEventRuntime,
            LocalASRProviderRuntimePort,
            ProviderChannelResetPort,
        ],
        PeerCaptureSessionOwner,
    ]
    previous_self_capture: Callable[[], SelfCaptureSessionOwner | None]
    component_sink: Callable[[RuntimePipelineComponents], None]
    peer_application: Callable[[], PeerApplicationOwner]
    configure_vrc_mic: Callable[..., Awaitable[None]]
    stt_failure_sink: Callable[[str], None]
    cleanup_failure_sink: Callable[[str, BaseException], None]
    managed_gemma: ManagedGemmaTranslationOwner | None = None
    http_extensions: HttpExtensionRegistry | None = None
    failed_resources: RuntimePipelineResourceOwner | None = field(
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
        resources = RuntimePipelineResourceOwner()
        try:
            pipeline = await compose_runtime_pipeline(
                settings=settings,
                config_path=self.config_path,
                clock=self.clock,
                runtime_logging=self.runtime_logging,
                managed_release=self.managed_release,
                managed_delegate_ready=self.managed_delegate_ready,
                managed_gemma=self.managed_gemma,
                local_asr_factory=self.local_asr_factory,
                self_capture_factory=self.self_capture_factory,
                peer_capture_factory=self.peer_capture_factory,
                vrc_mic_state=vrc_mic_state,
                vrc_mic_audio_gate=vrc_mic_audio_gate,
                receiver_active=receiver_active,
                stt_failure_sink=self.stt_failure_sink,
                http_extensions=self.http_extensions,
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
        [
            SelfCaptureVadEventRuntime,
            LocalASRProviderRuntimePort,
            ProviderChannelResetPort,
            VrcMicAudioGate,
        ],
        SelfCaptureSessionOwner,
    ],
    peer_capture_factory: Callable[
        [
            PeerCaptureVadEventRuntime,
            LocalASRProviderRuntimePort,
            ProviderChannelResetPort,
        ],
        PeerCaptureSessionOwner,
    ],
    vrc_mic_state: VrcMicState | None,
    vrc_mic_audio_gate: VrcMicAudioGate | None,
    receiver_active: bool,
    stt_failure_sink: Callable[[str], None],
    managed_gemma: ManagedGemmaTranslationOwner | None = None,
    http_extensions: HttpExtensionRegistry | None = None,
    resources: RuntimePipelineResourceOwner | None = None,
) -> RuntimePipelineComponents:
    owned_resources = resources is None
    pipeline_resources = resources or RuntimePipelineResourceOwner()
    try:
        return await _compose_runtime_pipeline(
            settings=settings,
            config_path=config_path,
            clock=clock,
            runtime_logging=runtime_logging,
            managed_release=managed_release,
            managed_delegate_ready=managed_delegate_ready,
            managed_gemma=managed_gemma,
            local_asr_factory=local_asr_factory,
            self_capture_factory=self_capture_factory,
            peer_capture_factory=peer_capture_factory,
            vrc_mic_state=vrc_mic_state,
            vrc_mic_audio_gate=vrc_mic_audio_gate,
            receiver_active=receiver_active,
            stt_failure_sink=stt_failure_sink,
            http_extensions=http_extensions,
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
    managed_gemma: ManagedGemmaTranslationOwner | None,
    local_asr_factory: Callable[[object], LocalASRProviderRuntimeFactoryPort],
    self_capture_factory: Callable[
        [
            SelfCaptureVadEventRuntime,
            LocalASRProviderRuntimePort,
            ProviderChannelResetPort,
            VrcMicAudioGate,
        ],
        SelfCaptureSessionOwner,
    ],
    peer_capture_factory: Callable[
        [
            PeerCaptureVadEventRuntime,
            LocalASRProviderRuntimePort,
            ProviderChannelResetPort,
        ],
        PeerCaptureSessionOwner,
    ],
    vrc_mic_state: VrcMicState | None,
    vrc_mic_audio_gate: VrcMicAudioGate | None,
    receiver_active: bool,
    stt_failure_sink: Callable[[str], None],
    http_extensions: HttpExtensionRegistry | None,
    resources: RuntimePipelineResourceOwner,
) -> RuntimePipelineComponents:
    secrets = create_secret_store(settings.secrets, config_path=config_path)
    if http_extensions is None and settings.translation.model == TranslationModel.CUSTOM_HTTP:
        http_extensions = HttpExtensionRegistry(default_http_extensions_dir())
        http_extensions.reload()
    if settings.translation.model != TranslationModel.MANAGED_GEMMA and managed_gemma is not None:
        await managed_gemma.deactivate()
    if settings.translation.model not in {
        TranslationModel.CUSTOM_HTTP,
        TranslationModel.MANAGED_GEMMA,
    }:
        await managed_release.rebuild(secrets=secrets)

    llm = None
    with contextlib.suppress(Exception):
        gemma_runtime = None
        gemma_release = None
        if settings.translation.model == TranslationModel.MANAGED_GEMMA:
            if managed_gemma is None:
                raise RuntimeError("managed Gemma translation runtime is unavailable")
            gemma_runtime = managed_gemma.runtime
            gemma_release = noop_managed_gemma_release
        llm = create_translation_backend(
            settings,
            secrets=secrets,
            http_extensions=(
                http_extensions or HttpExtensionRegistry(default_http_extensions_dir())
            ),
            managed_release_service=managed_release.service,
            managed_delegate_ready=managed_delegate_ready,
            runtime_logging=runtime_logging,
            managed_gemma_runtime=gemma_runtime,
            managed_gemma_release=gemma_release,
        )
        resources.pending_llm = llm

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
    ui_events: asyncio.Queue[UIEvent] = asyncio.Queue()
    stt_sessions = SttSessionStateProjection()
    callbacks = TranslationChannelOwnerCallbacks(stt_sessions)
    output_runtime = OutputRuntime(
        chatbox=osc,
        clock=clock,
    )
    resources.output_runtime = output_runtime
    self_runtime = ChannelRuntime(channel="self")
    resources.self_runtime = self_runtime
    peer_runtime = ChannelRuntime(channel="peer")
    resources.peer_runtime = peer_runtime
    context_resolver = ContextResolver(
        clock=clock,
        config_snapshot=translation_runtime_configuration.snapshot,
    )
    translation_diagnostics = TranslationLatencyDiagnosticsOwner(
        clock=clock,
        config_snapshot=translation_runtime_configuration.snapshot,
        runtime_logging=runtime_logging,
    )
    translation_output_projection = TranslationOutputProjectionOwner(
        output_runtime=output_runtime,
        ui_messages=TranslationUiMessageQueue(ui_events),
        diagnostics=translation_diagnostics,
        clock=clock,
    )
    llm_runtime = ProviderRuntimeHandle(
        name="llm",
        provider=llm,
    )
    resources.llm_runtime = llm_runtime
    resources.pending_llm = None
    translation_requests = TranslationRequestOwner(
        config_snapshot=translation_runtime_configuration.snapshot,
        self_runtime=self_runtime,
        peer_runtime=peer_runtime,
        context_resolver=context_resolver,
        provider_runtime=llm_runtime,
        diagnostics=translation_diagnostics,
        presentation=translation_output_projection,
        clock=clock,
    )
    translation_turns = TranslationTurnLifecycleOwner(
        on_child_created=callbacks.child_created,
        on_child_started=callbacks.child_started,
        process_child=callbacks.process_child,
        on_child_terminal=callbacks.child_terminal,
        on_parent_closed=callbacks.parent_closed,
        on_parent_rejected=callbacks.parent_rejected,
        output=callbacks,
        config_snapshot=translation_runtime_configuration.snapshot,
    )
    resources.translation_turns = translation_turns
    await translation_turns.close_channel_ingress("self")
    await translation_turns.close_channel_ingress("peer")
    local_asr_runtime = local_asr_factory(secrets).create(
        LocalASRProviderRuntimeCallbacks(
            self_event_handler=callbacks.self_event_handler,
            peer_event_handler=callbacks.peer_event_handler,
            retired_event_handler=callbacks.retired_event_handler,
            self_exception_handler=callbacks.self_exception_handler,
            peer_exception_handler=callbacks.peer_exception_handler,
        )
    )
    resources.local_asr_runtime = local_asr_runtime
    self_translation_channel = SelfTranslationChannelOwner(
        runtime=self_runtime,
        config_snapshot=translation_runtime_configuration.snapshot,
        translation_turns=translation_turns,
        local_asr_runtime=local_asr_runtime,
        translation_requests=translation_requests,
        output_projection=translation_output_projection,
        diagnostics=translation_diagnostics,
        clock=clock,
    )
    await self_translation_channel.close_ingress()
    resources.self_translation_channel = self_translation_channel
    callbacks.bind_self(self_translation_channel)
    peer_translation_channel = PeerTranslationChannelOwner(
        runtime=peer_runtime,
        config_snapshot=translation_runtime_configuration.snapshot,
        translation_turns=translation_turns,
        local_asr_runtime=local_asr_runtime,
        translation_requests=translation_requests,
        output_projection=translation_output_projection,
        diagnostics=translation_diagnostics,
        clock=clock,
    )
    await peer_translation_channel.close_ingress()
    resources.peer_translation_channel = peer_translation_channel
    callbacks.bind_peer(peer_translation_channel)
    channel_reset = RuntimePipelineChannelResetRouter(
        self_owner=self_translation_channel,
        peer_owner=peer_translation_channel,
    )
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

    self_capture = self_capture_factory(
        self_translation_channel,
        local_asr_runtime,
        self_translation_channel,
        gate,
    )
    resources.self_capture = self_capture
    peer_capture = peer_capture_factory(
        peer_translation_channel,
        local_asr_runtime,
        peer_translation_channel,
    )
    resources.peer_capture = peer_capture
    return RuntimePipelineComponents(
        sender=sender,
        osc=osc,
        self_capture=self_capture,
        peer_capture=peer_capture,
        vrc_mic_state=state,
        vrc_mic_audio_gate=gate,
        prepare_self_provider=prepare_self_provider,
        translation_runtime_configuration=translation_runtime_configuration,
        output_runtime=output_runtime,
        self_runtime=self_runtime,
        peer_runtime=peer_runtime,
        translation_turns=translation_turns,
        local_asr_runtime=local_asr_runtime,
        llm_runtime=llm_runtime,
        context_resolver=context_resolver,
        translation_diagnostics=translation_diagnostics,
        translation_output_projection=translation_output_projection,
        translation_requests=translation_requests,
        self_translation_channel=self_translation_channel,
        peer_translation_channel=peer_translation_channel,
        channel_reset=channel_reset,
        stt_sessions=stt_sessions,
        ui_events=ui_events,
        resource_owner=resources,
        start_callbacks=resources.start_callbacks,
        close_callbacks=resources.close_callbacks,
    )


__all__ = [
    "RuntimePipelineComponents",
    "RuntimePipelineChannelResetRouter",
    "RuntimePipelineHandle",
    "RuntimePipelineLauncher",
    "RuntimePipelineResourceOwner",
    "compose_runtime_pipeline",
]
