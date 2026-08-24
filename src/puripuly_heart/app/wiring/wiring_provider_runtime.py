from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path

from puripuly_heart.app.ports.translation_runtime_configuration import (
    TranslationRuntimeSettingsValues,
)
from puripuly_heart.app.services.canonical_settings_persistence import SettingsOwner
from puripuly_heart.app.services.managed_gemma_translation import ManagedGemmaTranslationOwner
from puripuly_heart.app.services.peer_application import PeerApplicationOwner
from puripuly_heart.app.services.provider_runtime_apply import (
    LlmProviderRebuildContext,
    LlmProviderRebuildOwner,
    ProviderRuntimeApplyPlan,
    ProviderRuntimeOwner,
    ProviderRuntimeState,
)
from puripuly_heart.config.paths import default_http_extensions_dir
from puripuly_heart.config.settings import (
    AppSettings,
    LLMProviderName,
    OpenRouterCredentialSource,
    TranslationModel,
)
from puripuly_heart.config.settings_vnext.schema import AppSettingsVNext
from puripuly_heart.config.vad_defaults import DEFAULT_STABLE_VAD_HANGOVER_MS
from puripuly_heart.core.http_extensions import HttpExtensionRegistry
from puripuly_heart.core.local_asr_provider_runtime import LocalASRProviderRuntimePort
from puripuly_heart.core.orchestrator.configuration import (
    TranslationRuntimeConfigurationPort,
)
from puripuly_heart.core.runtime.provider_handle import ProviderRuntimeHandle
from puripuly_heart.core.runtime.self_capture import SelfCaptureSessionOwner
from puripuly_heart.core.self_capture import SelfCaptureSessionSnapshot
from puripuly_heart.core.translation_policy import FIXED_TRANSLATION_POLICY

from .wiring_managed_account import ManagedOpenRouterReleaseRuntime
from .wiring_managed_gemma import (
    managed_gemma_selection,
    managed_gemma_translation_desired,
    noop_managed_gemma_release,
)
from .wiring_provider_runtime_policy import (
    build_llm_provider_signature,
    provider_runtime_requires_gpu_restart,
)
from .wiring_secrets_factory import create_secret_store
from .wiring_stt_factory import (
    build_peer_stt_provider_signature_from_vnext,
    build_peer_stt_runtime_signature,
    build_self_capture_session_config,
    build_self_stt_provider_signature,
    build_self_stt_runtime_signature,
)
from .wiring_translation_backend import create_translation_backend
from .wiring_translation_runtime_configuration import (
    replace_translation_runtime_settings,
)


def project_translation_runtime_settings(
    settings: AppSettings,
) -> TranslationRuntimeSettingsValues:
    return TranslationRuntimeSettingsValues(
        source_language=settings.languages.source_language,
        target_language=settings.languages.target_language,
        peer_source_language=settings.languages.peer_source_language,
        peer_target_language=settings.languages.peer_target_language,
        system_prompt=settings.system_prompt,
        chatbox_include_source=settings.osc.chatbox_include_source,
        hangover_s=(
            settings.stt.low_latency_vad_hangover_ms / 1000.0
            if FIXED_TRANSLATION_POLICY.fast_translation_enabled
            else DEFAULT_STABLE_VAD_HANGOVER_MS / 1000.0
        ),
        peer_hangover_s=settings.desktop_audio.vad_hangover_ms / 1000.0,
        low_latency_mode=FIXED_TRANSLATION_POLICY.fast_translation_enabled,
        low_latency_merge_gap_ms=settings.stt.low_latency_merge_gap_ms,
        low_latency_spec_retry_max=settings.stt.low_latency_spec_retry_max,
    )


@dataclass(slots=True)
class ProviderRuntimeSignatures:
    http_extensions: HttpExtensionRegistry | None = None
    last_self_runtime: tuple[object, ...] | None = None
    last_self_provider: tuple[object, ...] | None = None
    last_llm_provider: tuple[object, ...] | None = None
    superseded_settings_ids: set[int] = field(default_factory=set)

    def sync(
        self,
        settings: AppSettings,
        *,
        canonical: AppSettingsVNext,
        peer: PeerApplicationOwner,
    ) -> None:
        self.last_self_runtime = build_self_stt_runtime_signature(settings)
        self.last_self_provider = build_self_stt_provider_signature(settings)
        self.last_llm_provider = build_llm_provider_signature(
            settings,
            http_extensions=self.http_extensions,
        )
        peer.last_runtime_signature = build_peer_stt_runtime_signature(
            settings,
            canonical_settings=canonical,
        )
        peer.last_provider_signature = build_peer_stt_provider_signature_from_vnext(canonical)
        peer.last_intent_enabled = settings.ui.peer_translation_enabled
        peer.last_activation_requested = peer.activation_requested(
            intent_enabled=settings.ui.peer_translation_enabled,
            eula_accepted=settings.ui.peer_translation_eula_accepted,
        )

    def capture_peer_before_canonical_mutation(
        self,
        settings: AppSettings,
        *,
        canonical: AppSettingsVNext,
        peer: PeerApplicationOwner,
    ) -> None:
        if peer.last_provider_signature is None:
            peer.last_provider_signature = build_peer_stt_provider_signature_from_vnext(canonical)
        if peer.last_runtime_signature is None:
            peer.last_runtime_signature = build_peer_stt_runtime_signature(
                settings,
                canonical_settings=canonical,
            )

    def cache(
        self,
        peer: PeerApplicationOwner,
    ) -> tuple[object | None, object | None, object | None]:
        return (
            self.last_self_provider,
            peer.last_provider_signature,
            self.last_llm_provider,
        )

    def mark_llm_retry(self) -> None:
        self.last_llm_provider = ()

    def mark_superseded(self, settings: AppSettings) -> None:
        self.superseded_settings_ids.add(id(settings))

    def consume_superseded(self, settings: AppSettings) -> bool:
        settings_id = id(settings)
        if settings_id not in self.superseded_settings_ids:
            return False
        self.superseded_settings_ids.discard(settings_id)
        return True


@dataclass(slots=True)
class ProviderRuntimeEffects:
    settings: SettingsOwner
    llm_runtime_provider: Callable[[], ProviderRuntimeHandle | None]
    local_asr_runtime_provider: Callable[[], LocalASRProviderRuntimePort | None]
    translation_runtime_configuration_provider: Callable[
        [],
        TranslationRuntimeConfigurationPort | None,
    ]
    self_capture_provider: Callable[[], SelfCaptureSessionOwner | None]
    self_capture_owner: Callable[[], SelfCaptureSessionOwner]
    peer: Callable[[], PeerApplicationOwner]
    peer_desired: Callable[[AppSettings], bool]
    clear_local_pending: Callable[[], None]
    sync_local_notice: Callable[[], None]
    managed_pending_sink: Callable[[bool], None]
    managed_pending_provider: Callable[[], bool]
    dashboard_managed_pending_sink: Callable[[bool], None]
    sync_effective_flags: Callable[[AppSettings], None]
    refresh_overlay: Callable[[], None]
    refresh_peer_runtime: Callable[[], Awaitable[None]]
    replace_self_stt: Callable[[bool], Awaitable[None]]
    self_state_sink: Callable[[SelfCaptureSessionSnapshot], None]
    self_availability: Callable[[SelfCaptureSessionSnapshot], bool]
    gpu_recovery: Callable[[AppSettings, ProviderRuntimeApplyPlan], Awaitable[None]]
    failure_sink: Callable[[str], None]
    success_sink: Callable[[str], None]

    def state(self, settings: object) -> ProviderRuntimeState:
        llm_runtime = self.llm_runtime_provider()
        local_asr_runtime = self.local_asr_runtime_provider()
        self_owner = self.self_capture_provider()
        return ProviderRuntimeState(
            runtime_available=llm_runtime is not None and local_asr_runtime is not None,
            llm_available=llm_runtime is not None and llm_runtime.provider is not None,
            self_stt_available=(
                local_asr_runtime is not None
                and local_asr_runtime.snapshot.channel_for("self").provider_id is not None
            ),
            peer_stt_available=(
                local_asr_runtime is not None
                and local_asr_runtime.snapshot.channel_for("peer").provider_id is not None
            ),
            self_stt_desired=bool(self_owner is not None and self_owner.snapshot.desired_active),
            peer_stt_desired=isinstance(settings, AppSettings) and self.peer_desired(settings),
        )

    def apply_common(self, settings: object) -> None:
        if not isinstance(settings, AppSettings):
            raise TypeError("provider runtime settings must be AppSettings")
        self.settings.current = settings
        self.clear_local_pending()
        self.sync_local_notice()
        if (
            settings.translation.model == TranslationModel.CUSTOM_HTTP
            or settings.provider.llm != LLMProviderName.OPENROUTER
            or settings.openrouter.selected_source != OpenRouterCredentialSource.MANAGED
        ):
            self.managed_pending_sink(False)
        else:
            self.dashboard_managed_pending_sink(self.managed_pending_provider())
        config_owner = self.translation_runtime_configuration_provider()
        if config_owner is None:
            return
        peer_enabled = self.peer().effective_enabled()
        replace_translation_runtime_settings(
            config_owner,
            project_translation_runtime_settings(settings),
            peer_translation_enabled=peer_enabled,
            integrated_context_enabled=peer_enabled,
        )

    async def refresh_peer(self) -> None:
        await self.refresh_peer_runtime()
        current = self.settings.current
        if current is not None:
            self.sync_effective_flags(current)
        self.refresh_overlay()

    async def refresh_self_stt(self) -> None:
        owner = self.self_capture_provider()
        if owner is not None and owner.snapshot.desired_active:
            await self.replace_self_stt(True)
            return
        await self.rebuild_self_stt()

    async def rebuild_self_stt(self) -> None:
        current = self.settings.current
        if current is None or self.local_asr_runtime_provider() is None:
            return
        owner = self.self_capture_owner()
        config = build_self_capture_session_config(current)
        if owner.snapshot.desired_active:
            snapshot = await owner.apply_intent(config, enabled=True)
        else:
            snapshot = await owner.prepare_provider(config)
        self.self_state_sink(snapshot)
        if not self.self_availability(snapshot):
            self.failure_sink("STT backend not available")
            return
        self.success_sink("[Settings] STT provider replacement completed successfully")


@dataclass(frozen=True, slots=True)
class ProviderRuntimeComponents:
    runtime: ProviderRuntimeOwner
    llm_rebuild: LlmProviderRebuildOwner
    effects: ProviderRuntimeEffects
    signatures: ProviderRuntimeSignatures
    sync_signatures: Callable[[AppSettings], None]
    capture_signatures_before_canonical_mutation: Callable[[], None]


def compose_provider_runtime(
    *,
    config_path: Path,
    settings: SettingsOwner,
    llm_runtime_provider: Callable[[], ProviderRuntimeHandle | None],
    http_extensions: HttpExtensionRegistry | None = None,
    local_asr_runtime_provider: Callable[[], LocalASRProviderRuntimePort | None],
    translation_runtime_configuration_provider: Callable[
        [],
        TranslationRuntimeConfigurationPort | None,
    ],
    self_capture_provider: Callable[[], SelfCaptureSessionOwner | None],
    self_capture_owner: Callable[[], SelfCaptureSessionOwner],
    peer: Callable[[], PeerApplicationOwner],
    peer_desired: Callable[[AppSettings], bool],
    canonical_settings: Callable[[AppSettings], AppSettingsVNext],
    clear_local_pending: Callable[[], None],
    sync_local_notice: Callable[[], None],
    managed_pending_sink: Callable[[bool], None],
    managed_pending_provider: Callable[[], bool],
    dashboard_managed_pending_sink: Callable[[bool], None],
    sync_effective_flags: Callable[[AppSettings], None],
    refresh_overlay: Callable[[], None],
    refresh_peer_runtime: Callable[[], Awaitable[None]],
    replace_self_stt: Callable[[bool], Awaitable[None]],
    self_state_sink: Callable[[SelfCaptureSessionSnapshot], None],
    self_availability: Callable[[SelfCaptureSessionSnapshot], bool],
    gpu_recovery: Callable[[AppSettings, ProviderRuntimeApplyPlan], Awaitable[None]],
    managed_release: Callable[[], ManagedOpenRouterReleaseRuntime],
    managed_delegate_ready: Callable[[], None],
    runtime_logging: object,
    translation_needs_key_sink: Callable[[bool], None],
    usage_refresh: Callable[[], Awaitable[None]],
    failure_sink: Callable[[str], None],
    success_sink: Callable[[str], None],
    additional_signature_sink: Callable[[AppSettings], None],
    managed_gemma: ManagedGemmaTranslationOwner | None = None,
    signatures: ProviderRuntimeSignatures | None = None,
) -> ProviderRuntimeComponents:
    effective_http_extensions = http_extensions
    if effective_http_extensions is None:
        effective_http_extensions = HttpExtensionRegistry(default_http_extensions_dir())
        effective_http_extensions.reload()
    signature_state = signatures or ProviderRuntimeSignatures()
    if signature_state.http_extensions is None:
        signature_state.http_extensions = effective_http_extensions
    effects = ProviderRuntimeEffects(
        settings=settings,
        llm_runtime_provider=llm_runtime_provider,
        local_asr_runtime_provider=local_asr_runtime_provider,
        translation_runtime_configuration_provider=(translation_runtime_configuration_provider),
        self_capture_provider=self_capture_provider,
        self_capture_owner=self_capture_owner,
        peer=peer,
        peer_desired=peer_desired,
        clear_local_pending=clear_local_pending,
        sync_local_notice=sync_local_notice,
        managed_pending_sink=managed_pending_sink,
        managed_pending_provider=managed_pending_provider,
        dashboard_managed_pending_sink=dashboard_managed_pending_sink,
        sync_effective_flags=sync_effective_flags,
        refresh_overlay=refresh_overlay,
        refresh_peer_runtime=refresh_peer_runtime,
        replace_self_stt=replace_self_stt,
        self_state_sink=self_state_sink,
        self_availability=self_availability,
        gpu_recovery=gpu_recovery,
        failure_sink=failure_sink,
        success_sink=success_sink,
    )

    def llm_context() -> LlmProviderRebuildContext | None:
        current = settings.current
        runtime = llm_runtime_provider()
        if current is None or runtime is None:
            return None

        async def replace_provider(provider: object | None) -> object | None:
            return await runtime.replace_provider(provider, start=False)

        return LlmProviderRebuildContext(
            settings=current,
            replace_provider=replace_provider,
            requires_secret=(
                current.translation.model != TranslationModel.CUSTOM_HTTP
                and current.provider.llm
                in {
                    LLMProviderName.GEMINI,
                    LLMProviderName.OPENROUTER,
                    LLMProviderName.QWEN,
                    LLMProviderName.DEEPSEEK,
                }
            ),
            resource_label=(
                "Translation backend"
                if current.translation.model == TranslationModel.CUSTOM_HTTP
                else "LLM provider"
            ),
        )

    async def create_llm(settings_value: object) -> object | None:
        if not isinstance(settings_value, AppSettings):
            raise TypeError("LLM provider rebuild settings must be AppSettings")
        secrets = create_secret_store(settings_value.secrets, config_path=config_path)
        if settings_value.translation.model in (
            TranslationModel.MANAGED_GEMMA,
            TranslationModel.MANAGED_GEMMA_12B,
        ):
            if managed_gemma is None:
                raise RuntimeError("managed Gemma translation runtime is unavailable")
            config = translation_runtime_configuration_provider()
            translation_on = managed_gemma_translation_desired(
                translation_enabled=bool(
                    config is not None and config.snapshot().value.translation_enabled
                ),
                peer_translation_enabled=bool(settings_value.ui.peer_translation_enabled),
            )
            if not translation_on:
                return create_translation_backend(
                    settings_value,
                    secrets=secrets,
                    http_extensions=effective_http_extensions,
                    runtime_logging=runtime_logging,
                    managed_gemma_runtime=managed_gemma.runtime,
                    managed_gemma_release=noop_managed_gemma_release,
                )
            activation = await managed_gemma.prepare(managed_gemma_selection(settings_value))
            return create_translation_backend(
                settings_value,
                secrets=secrets,
                http_extensions=effective_http_extensions,
                runtime_logging=runtime_logging,
                managed_gemma_runtime=activation.runtime,
                managed_gemma_release=noop_managed_gemma_release,
            )
        if managed_gemma is not None:
            await managed_gemma.deactivate()
        if settings_value.translation.model == TranslationModel.CUSTOM_HTTP:
            return create_translation_backend(
                settings_value,
                secrets=secrets,
                http_extensions=effective_http_extensions,
            )
        release = managed_release()
        await release.rebuild(secrets=secrets)
        return create_translation_backend(
            settings_value,
            secrets=secrets,
            managed_release_service=release.service,
            managed_delegate_ready=managed_delegate_ready,
            runtime_logging=runtime_logging,
            http_extensions=effective_http_extensions,
        )

    llm_rebuild = LlmProviderRebuildOwner(
        context_provider=llm_context,
        provider_factory=create_llm,
        availability_sink=translation_needs_key_sink,
        usage_refresh=usage_refresh,
        failure_sink=failure_sink,
        success_sink=success_sink,
    )

    def sync_signatures(current: AppSettings) -> None:
        signature_state.sync(
            current,
            canonical=canonical_settings(current),
            peer=peer(),
        )
        additional_signature_sink(current)

    def capture_signatures_before_canonical_mutation() -> None:
        current = settings.current
        if current is None:
            return
        signature_state.capture_peer_before_canonical_mutation(
            current,
            canonical=canonical_settings(current),
            peer=peer(),
        )

    runtime = ProviderRuntimeOwner(
        state_provider=effects.state,
        common_effect=effects.apply_common,
        rebuild_llm=llm_rebuild.rebuild,
        recover_gpu=effects.gpu_recovery,
        refresh_peer=effects.refresh_peer,
        refresh_self_stt=effects.refresh_self_stt,
        signature_sink=sync_signatures,
        llm_retry_sink=signature_state.mark_llm_retry,
        current_settings_provider=lambda: settings.current,
        signature_cache_provider=lambda: signature_state.cache(peer()),
        self_signature_builder=build_self_stt_provider_signature,
        peer_signature_builder=lambda current, canonical: (
            build_peer_stt_provider_signature_from_vnext(canonical or canonical_settings(current))
        ),
        llm_signature_builder=lambda current: build_llm_provider_signature(
            current,
            http_extensions=effective_http_extensions,
        ),
        gpu_restart_decision=provider_runtime_requires_gpu_restart,
    )
    return ProviderRuntimeComponents(
        runtime=runtime,
        llm_rebuild=llm_rebuild,
        effects=effects,
        signatures=signature_state,
        sync_signatures=sync_signatures,
        capture_signatures_before_canonical_mutation=(capture_signatures_before_canonical_mutation),
    )


__all__ = [
    "ProviderRuntimeComponents",
    "ProviderRuntimeEffects",
    "ProviderRuntimeSignatures",
    "compose_provider_runtime",
]
