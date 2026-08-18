from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from puripuly_heart.app import wiring_llm_factory as _llm_factory
from puripuly_heart.app.adapters.peer_capture_provider import PeerCaptureProviderAdapter
from puripuly_heart.app.adapters.self_capture_provider import SelfCaptureProviderAdapter
from puripuly_heart.app.adapters.sync_secret_store import (
    SyncSecretStore,
    SyncSecretStoreAdapter,
)
from puripuly_heart.app.ports.provider_channel_runtime import ProviderChannelResetPort
from puripuly_heart.app.ports.secret_store import SecretStorePort
from puripuly_heart.config.runtime_resolution import resolve_llm_config
from puripuly_heart.core.local_asr_provider_runtime import LocalASRProviderRuntimePort
from puripuly_heart.core.local_stt_huggingface_xet_adapter import (
    HuggingFaceXetDownloadAdapter,
)
from puripuly_heart.core.openrouter_credentials import load_managed_openrouter_user_identifier
from puripuly_heart.core.peer_capture import (
    PeerCaptureAdmissionPort,
    PeerCaptureTargetResolverPort,
)
from puripuly_heart.core.runtime.local_asr_provisioning import (
    LocalASRProvisioningOwner,
    ProvisioningDiagnosticSink,
    ProvisioningStateChanged,
)
from puripuly_heart.core.runtime.peer_channel import PeerCaptureSessionOwner
from puripuly_heart.core.runtime.self_capture import (
    SelfCaptureAudioLoop,
    SelfCaptureDiagnosticSink,
    SelfCaptureProviderRequestFactory,
    SelfCaptureSessionOwner,
    SelfCaptureSourceFactory,
    SelfCaptureStateChanged,
    SelfCaptureVadFactory,
)
from puripuly_heart.core.self_capture import SelfCaptureAdmissionPort
from puripuly_heart.core.translation_policy import FIXED_TRANSLATION_POLICY

from .wiring_composition import (
    create_microphone_test_capture_adapter,
    create_peer_capture_admission_adapter,
    create_peer_capture_audio_loop_adapter,
    create_peer_capture_source_adapter,
    create_peer_capture_target_resolver_adapter,
    create_peer_capture_vad_adapter,
    create_peer_capture_vad_sink_adapter,
    create_provider_verifier,
    create_self_capture_admission_adapter,
    create_self_capture_audio_loop_adapter,
    create_self_capture_source_adapter,
    create_self_capture_vad_adapter,
    create_self_capture_vad_sink_adapter,
)
from .wiring_llm_factory import (
    MANAGED_OPENROUTER_RELEASE_SERVICE_REQUIRED_ERROR,
    _LazyFactoryLLMProvider,
)
from .wiring_local_asr_provider_runtime import (
    LocalASRProviderRuntimeFactory,
    ManagedSTTProviderFactory,
)
from .wiring_managed_auth_factory import (
    DiscordManagedBrokerClientAdapter,
    DiscordOAuthAuthAdapter,
    ManagedIdentityPreflightAdapter,
    ManagedIdentityStateAdapter,
    apply_discord_issue_result_to_managed_state,
    build_managed_identity_state_port,
    build_openrouter_credential_runtime_config,
    build_openrouter_release_runtime_config,
)
from .wiring_overlay_factory import resolve_overlay_config
from .wiring_secrets_factory import (
    SECRETS_PASSPHRASE_ENV,
    copy_stable_secrets_to_vnext_namespace,
    create_secret_store,
    require_secret,
    require_secret_any,
)
from .wiring_stt_factory import (
    ResolvedPeerSTTConfig,
    build_custom_vocabulary_runtime_config,
    build_local_asr_session_options,
    build_peer_capture_session_config,
    build_peer_capture_session_config_from_vnext,
    build_peer_stt_provider_request,
    build_peer_stt_provider_signature,
    build_peer_stt_provider_signature_from_vnext,
    build_peer_stt_runtime_signature,
    build_self_capture_session_config,
    build_self_capture_vad_signature,
    build_self_local_asr_transition_request,
    build_self_stt_provider_request,
    build_self_stt_provider_signature,
    build_self_stt_runtime_signature,
    create_peer_stt_backend,
    create_peer_stt_backend_from_resolved_config,
    create_stt_backend,
    create_stt_backend_from_resolved_config,
    resolve_peer_stt_config,
    resolve_peer_stt_runtime_config,
    resolve_peer_stt_runtime_config_from_vnext,
    resolve_self_stt_runtime_config,
)

_WIRING_SECRET_KEYS_FOR_COMPATIBILITY_GUARD = (
    "google_api_key",
    "deepseek_api_key",
    "deepgram_api_key",
    "soniox_api_key",
    "alibaba_api_key_beijing",
    "alibaba_api_key_singapore",
    "alibaba_api_key",
    "local_llm_api_key",
    "cerebras_api_key",
    "custom_stt_api_key",
)


def compose_self_capture_session_owner(
    *,
    provider_runtime: LocalASRProviderRuntimePort | None,
    channel_reset: ProviderChannelResetPort | None,
    admission: SelfCaptureAdmissionPort,
    provider_request_factory: SelfCaptureProviderRequestFactory,
    source_factory: SelfCaptureSourceFactory,
    vad_factory: SelfCaptureVadFactory,
    run_audio_loop: SelfCaptureAudioLoop,
    vad_sink: object,
    state_changed: SelfCaptureStateChanged | None = None,
    diagnostic_sink: SelfCaptureDiagnosticSink | None = None,
    audio_gate_reset: Callable[[], object] | None = None,
) -> SelfCaptureSessionOwner:
    return SelfCaptureSessionOwner(
        admission=admission,
        provider=SelfCaptureProviderAdapter(provider_runtime, channel_reset),
        provider_request_factory=provider_request_factory,
        source_factory=source_factory,
        vad_factory=vad_factory,
        run_audio_loop=run_audio_loop,
        vad_sink=vad_sink,
        state_changed=state_changed,
        diagnostic_sink=diagnostic_sink,
        audio_gate_reset=audio_gate_reset,
    )


def compose_peer_capture_session_owner(
    *,
    provider_runtime: LocalASRProviderRuntimePort | None,
    channel_reset: ProviderChannelResetPort | None,
    admission: PeerCaptureAdmissionPort,
    target_resolver: PeerCaptureTargetResolverPort,
    clock,
    provider_request_factory,
    source_factory,
    vad_factory,
    run_audio_loop,
    vad_sink: object,
    state_changed=None,
    diagnostic_sink=None,
    local_asr_diagnostic_sink=None,
) -> PeerCaptureSessionOwner:
    return PeerCaptureSessionOwner(
        admission=admission,
        target_resolver=target_resolver,
        provider=PeerCaptureProviderAdapter(provider_runtime, channel_reset),
        clock=clock,
        provider_request_factory=provider_request_factory,
        source_factory=source_factory,
        vad_factory=vad_factory,
        run_audio_loop=run_audio_loop,
        vad_sink=vad_sink,
        state_changed=state_changed,
        diagnostic_sink=diagnostic_sink,
        local_asr_diagnostic_sink=local_asr_diagnostic_sink,
    )


_base_llm_provider_from_resolved_config = _llm_factory._base_llm_provider_from_resolved_config
_openrouter_provider_from_resolved_config = _llm_factory._openrouter_provider_from_resolved_config
_openrouter_provider_from_resolved_fields = _llm_factory._openrouter_provider_from_resolved_fields
_cerebras_api_key_for_resolved_credential = _llm_factory._cerebras_api_key_for_resolved_credential
_qwen_api_key_for_resolved_credential = _llm_factory._qwen_api_key_for_resolved_credential


def create_llm_provider_from_resolved_config(*args, **kwargs):
    _llm_factory.load_managed_openrouter_user_identifier = load_managed_openrouter_user_identifier
    return _llm_factory.create_llm_provider_from_resolved_config(*args, **kwargs)


def create_llm_provider(settings, **kwargs):
    runtime_input = _llm_factory._runtime_resolution_input_from_compatibility_settings(settings)
    resolved = resolve_llm_config(runtime_input)
    return create_llm_provider_from_resolved_config(
        resolved,
        compatibility_settings=settings,
        qwen_low_latency_mode=FIXED_TRANSLATION_POLICY.fast_translation_enabled,
        **kwargs,
    )


def create_local_asr_provisioning_owner(
    *,
    model_root: Path | None = None,
    state_changed: ProvisioningStateChanged | None = None,
    diagnostic_sink: ProvisioningDiagnosticSink | None = None,
) -> LocalASRProvisioningOwner:
    return LocalASRProvisioningOwner(
        model_root=model_root,
        state_changed=state_changed,
        diagnostic_sink=diagnostic_sink,
        huggingface_downloader=HuggingFaceXetDownloadAdapter(),
    )


def create_sync_secret_store_adapter(store: SyncSecretStore) -> SecretStorePort:
    return SyncSecretStoreAdapter(store)


__all__ = (
    "SECRETS_PASSPHRASE_ENV",
    "MANAGED_OPENROUTER_RELEASE_SERVICE_REQUIRED_ERROR",
    "LocalASRProviderRuntimeFactory",
    "ManagedSTTProviderFactory",
    "ResolvedPeerSTTConfig",
    "build_local_asr_session_options",
    "build_peer_capture_session_config",
    "build_peer_capture_session_config_from_vnext",
    "build_peer_stt_provider_signature",
    "build_peer_stt_provider_signature_from_vnext",
    "build_peer_stt_provider_request",
    "build_peer_stt_runtime_signature",
    "build_self_capture_session_config",
    "build_self_capture_vad_signature",
    "build_self_local_asr_transition_request",
    "build_self_stt_provider_request",
    "build_self_stt_provider_signature",
    "build_self_stt_runtime_signature",
    "create_llm_provider",
    "create_llm_provider_from_resolved_config",
    "create_local_asr_provisioning_owner",
    "create_microphone_test_capture_adapter",
    "create_peer_capture_admission_adapter",
    "create_peer_capture_audio_loop_adapter",
    "create_peer_capture_source_adapter",
    "create_peer_capture_target_resolver_adapter",
    "create_peer_capture_vad_adapter",
    "create_peer_capture_vad_sink_adapter",
    "create_peer_stt_backend",
    "create_peer_stt_backend_from_resolved_config",
    "create_provider_verifier",
    "create_secret_store",
    "create_self_capture_admission_adapter",
    "create_self_capture_audio_loop_adapter",
    "create_self_capture_source_adapter",
    "create_self_capture_vad_adapter",
    "create_self_capture_vad_sink_adapter",
    "create_sync_secret_store_adapter",
    "copy_stable_secrets_to_vnext_namespace",
    "create_stt_backend",
    "create_stt_backend_from_resolved_config",
    "require_secret",
    "require_secret_any",
    "resolve_overlay_config",
    "resolve_peer_stt_config",
    "resolve_peer_stt_runtime_config",
    "resolve_peer_stt_runtime_config_from_vnext",
    "resolve_self_stt_runtime_config",
    "DiscordManagedBrokerClientAdapter",
    "DiscordOAuthAuthAdapter",
    "ManagedIdentityPreflightAdapter",
    "ManagedIdentityStateAdapter",
    "apply_discord_issue_result_to_managed_state",
    "build_managed_identity_state_port",
    "build_openrouter_credential_runtime_config",
    "build_openrouter_release_runtime_config",
    "build_custom_vocabulary_runtime_config",
    "_LazyFactoryLLMProvider",
)
