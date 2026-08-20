from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from puripuly_heart.app.services.local_asr_selection import (
    LOCAL_CPU_PROVIDERS,
    resolve_local_asr_selection,
)
from puripuly_heart.config.capture_target_resolution import (
    resolve_desktop_audio_capture_target,
)
from puripuly_heart.config.resolved import ResolvedCredentialRequirement, ResolvedSTTConfig
from puripuly_heart.config.runtime_resolution import (
    CREDENTIAL_REF_CUSTOM_STT,
    CREDENTIAL_REF_DEEPGRAM_STT,
    CREDENTIAL_REF_QWEN_SINGAPORE,
    CREDENTIAL_REF_SONIOX_STT,
    SONIOX_STT_DEFAULT_KEEPALIVE_INTERVAL_S,
    SONIOX_STT_DEFAULT_TRAILING_SILENCE_MS,
    SONIOX_STT_MODEL_RT_V5,
    STT_CUSTOM_PROVIDERS,
    STT_PROVIDER_DEEPGRAM,
    STT_PROVIDER_LOCAL_CPU_AUTO,
    STT_PROVIDER_LOCAL_PARAKEET_JAPANESE,
    STT_PROVIDER_LOCAL_PARAKEET_V3,
    STT_PROVIDER_LOCAL_QWEN,
    STT_PROVIDER_LOCAL_QWEN_GPU,
    STT_PROVIDER_QWEN_ASR,
    STT_PROVIDER_SONIOX,
    STTRuntimeIntent,
)
from puripuly_heart.config.runtime_resolution import (
    resolve_stt_config as resolve_stt_runtime_config,
)
from puripuly_heart.config.settings import (
    STT_INTERNAL_SAMPLE_RATE_HZ,
    AppSettings,
    QwenRegion,
    STTProviderName,
    custom_stt_selection_for_provider,
    is_custom_stt_provider,
)
from puripuly_heart.config.settings_vnext.schema import AppSettingsVNext
from puripuly_heart.core.language import get_local_qwen_language_hint
from puripuly_heart.core.local_asr_provider_runtime import ProviderRuntimeBuildRequest
from puripuly_heart.core.local_stt_assets import (
    LOCAL_QWEN_GPU_MODEL_ID,
    LOCAL_STT_MODEL_ID,
    PARAKEET_JAPANESE_MODEL_ID,
    PARAKEET_V3_MODEL_ID,
    default_local_stt_model_dir,
)
from puripuly_heart.core.peer_capture import (
    PeerCaptureLanguageFacts,
    PeerCaptureSessionConfig,
    PeerCaptureTargetIntent,
)
from puripuly_heart.core.runtime.gpu_asr import SharedGpuASRRuntime
from puripuly_heart.core.runtime.local_asr_transition import (
    LocalASRSessionOptions,
    LocalASRTransitionRequest,
)
from puripuly_heart.core.runtime.local_qwen_lifecycle import (
    LOCAL_QWEN_IDLE_RELEASE_SECONDS,
)
from puripuly_heart.core.self_capture import SelfCaptureSessionConfig
from puripuly_heart.core.storage.secrets import SecretStore
from puripuly_heart.core.stt.backend import STTBackend
from puripuly_heart.core.stt.custom import custom_stt_secret_generation
from puripuly_heart.core.stt.custom_vocab import (
    CustomVocabularyRuntimeConfig,
    get_effective_custom_terms,
    get_effective_local_qwen_hotwords,
)
from puripuly_heart.core.translation_policy import FIXED_TRANSLATION_POLICY

from .wiring_llm_factory import _qwen_api_key_for_resolved_credential
from .wiring_secrets_factory import require_secret


def build_custom_vocabulary_runtime_config(
    settings: AppSettings,
) -> CustomVocabularyRuntimeConfig:
    """Build a narrow custom-vocabulary runtime DTO from legacy settings."""

    return CustomVocabularyRuntimeConfig(
        enabled=settings.stt.custom_vocabulary_enabled,
        terms=settings.stt.custom_terms,
    )


@dataclass(frozen=True, slots=True)
class ResolvedPeerSTTConfig:
    provider: STTProviderName
    source_language: str
    sample_rate_hz: int
    keyterms: tuple[str, ...]
    deepgram_model: str | None = None
    qwen_model: str | None = None
    qwen_region: QwenRegion | None = None
    soniox_model: str | None = None
    soniox_endpoint: str | None = None
    soniox_keepalive_interval_s: float | None = None
    soniox_trailing_silence_ms: int | None = None
    soniox_enable_language_identification: bool = False
    soniox_language_hints: tuple[str, ...] | None = None
    soniox_language_hints_strict: bool = False

    @property
    def model(self) -> str | None:
        if self.provider == STTProviderName.DEEPGRAM:
            return self.deepgram_model
        if self.provider == STTProviderName.QWEN_ASR:
            return self.qwen_model
        if self.provider == STTProviderName.SONIOX:
            return self.soniox_model
        return None

    @property
    def endpoint(self) -> str | None:
        if self.provider == STTProviderName.SONIOX:
            return self.soniox_endpoint
        return None

    @property
    def region(self) -> QwenRegion | None:
        if self.provider == STTProviderName.QWEN_ASR:
            return self.qwen_region
        return None

    @property
    def provider_options(self) -> Mapping[str, object]:
        if self.provider == STTProviderName.SONIOX:
            return {
                "keepalive_interval_s": self.soniox_keepalive_interval_s,
                "trailing_silence_ms": self.soniox_trailing_silence_ms,
                "enable_language_identification": self.soniox_enable_language_identification,
                "language_hints": self.soniox_language_hints,
                "language_hints_strict": self.soniox_language_hints_strict,
            }
        return {}


def _stt_provider_name_or_raise(
    provider: STTProviderName | str,
    *,
    peer: bool,
) -> STTProviderName:
    if isinstance(provider, STTProviderName):
        return provider
    try:
        return STTProviderName(str(provider))
    except ValueError as exc:
        label = "peer STT" if peer else "STT"
        raise ValueError(f"Unsupported {label} provider: {provider}") from exc


def _stt_provider_value_or_raise(
    provider: STTProviderName | str,
    *,
    peer: bool,
) -> str:
    return _stt_provider_name_or_raise(provider, peer=peer).value


def _effective_custom_terms_for_resolved_config(
    settings: AppSettings,
    source_language: str,
) -> Mapping[str, tuple[str, ...]]:
    terms = tuple(
        get_effective_custom_terms(
            build_custom_vocabulary_runtime_config(settings), source_language
        )
    )
    if not terms:
        return {}
    return {source_language: terms}


def _self_stt_runtime_intent_from_compatibility_settings(settings: AppSettings) -> STTRuntimeIntent:
    source_language = settings.languages.source_language
    provider = _stt_provider_value_or_raise(settings.provider.stt, peer=False)
    soniox_language_hints = None
    soniox_language_hints_strict = False
    if provider == STT_PROVIDER_SONIOX:
        from puripuly_heart.core.language import get_soniox_language_hints

        soniox_language_hints = tuple(get_soniox_language_hints(source_language))
        soniox_language_hints_strict = True
    custom_mode, custom_compatibility = custom_stt_selection_for_provider(
        provider,
        stored_mode=settings.custom_stt.mode,
        stored_compatibility=settings.custom_stt.compatibility,
    )
    return STTRuntimeIntent(
        channel="self",
        provider=provider,
        source_language=source_language,
        input_host_api=settings.audio.input_host_api,
        input_device=settings.audio.input_device,
        output_device=None,
        sample_rate_hz=STT_INTERNAL_SAMPLE_RATE_HZ,
        channels=settings.audio.internal_channels,
        ring_buffer_ms=settings.audio.ring_buffer_ms,
        drain_timeout_s=settings.stt.drain_timeout_s,
        vad_speech_threshold=settings.stt.vad_speech_threshold,
        vad_hangover_ms=settings.stt.low_latency_vad_hangover_ms,
        vad_pre_roll_ms=500,
        low_latency_enabled=FIXED_TRANSLATION_POLICY.fast_translation_enabled,
        low_latency_merge_gap_ms=settings.stt.low_latency_merge_gap_ms,
        low_latency_spec_retry_max=settings.stt.low_latency_spec_retry_max,
        custom_vocabulary_enabled=(
            settings.stt.custom_vocabulary_enabled and not is_custom_stt_provider(provider)
        ),
        custom_terms=_effective_custom_terms_for_resolved_config(settings, source_language),
        deepgram_model=settings.deepgram_stt.model,
        qwen_asr_model=settings.qwen_asr_stt.model,
        qwen_region=settings.qwen.region.value,
        soniox_model=settings.soniox_stt.model,
        soniox_endpoint=settings.soniox_stt.endpoint,
        soniox_keepalive_interval_s=settings.soniox_stt.keepalive_interval_s,
        soniox_trailing_silence_ms=settings.soniox_stt.trailing_silence_ms,
        soniox_language_hints=soniox_language_hints,
        soniox_language_hints_strict=soniox_language_hints_strict,
        custom_stt_mode=custom_mode,
        custom_stt_compatibility=custom_compatibility,
        custom_stt_endpoint=settings.custom_stt.endpoint,
        custom_stt_model=settings.custom_stt.model,
        custom_stt_extra=dict(settings.custom_stt.extra),
    )


def _peer_stt_runtime_intent_from_compatibility_settings(settings: AppSettings) -> STTRuntimeIntent:
    from puripuly_heart.config.settings_vnext.migration import from_legacy_app_settings

    _stt_provider_value_or_raise(settings.provider.peer_stt, peer=True)
    return peer_stt_runtime_intent_from_vnext(from_legacy_app_settings(settings))


def peer_stt_runtime_intent_from_vnext(settings: AppSettingsVNext) -> STTRuntimeIntent:
    intent = settings.intent
    provider = intent.peer_stt.provider
    automatic = intent.languages.peer_source_mode == "auto"
    automatic_soniox = provider == STT_PROVIDER_SONIOX and automatic
    source_language = intent.languages.peer_source_language or intent.languages.source_language
    language_hints = None
    language_hints_strict = False
    if provider == STT_PROVIDER_SONIOX:
        from puripuly_heart.core.language import get_soniox_language_hints

        if automatic_soniox:
            mapped_language_hints = tuple(
                dict.fromkeys(
                    hint
                    for language in intent.languages.peer_expected_languages
                    for hint in get_soniox_language_hints(language)
                )
            )
            language_hints = mapped_language_hints or None
        else:
            language_hints = tuple(get_soniox_language_hints(source_language))
    custom_mode, custom_compatibility = custom_stt_selection_for_provider(
        provider,
        stored_mode=intent.stt.custom.mode,
        stored_compatibility=intent.stt.custom.compatibility,
    )
    return STTRuntimeIntent(
        channel="peer",
        provider=provider,
        source_language=source_language,
        source_mode="auto" if automatic else "manual",
        input_host_api=None,
        input_device=None,
        output_device=intent.desktop_audio.output_device,
        sample_rate_hz=STT_INTERNAL_SAMPLE_RATE_HZ,
        channels=1,
        ring_buffer_ms=intent.audio.ring_buffer_ms,
        drain_timeout_s=intent.stt.drain_timeout_s,
        vad_speech_threshold=intent.desktop_audio.vad_speech_threshold,
        vad_hangover_ms=intent.desktop_audio.vad_hangover_ms,
        vad_pre_roll_ms=intent.desktop_audio.vad_pre_roll_ms,
        low_latency_enabled=FIXED_TRANSLATION_POLICY.fast_translation_enabled,
        low_latency_merge_gap_ms=intent.stt.low_latency_merge_gap_ms,
        low_latency_spec_retry_max=intent.stt.low_latency_spec_retry_max,
        custom_vocabulary_enabled=False,
        custom_terms={},
        deepgram_model=intent.stt.deepgram.model,
        qwen_asr_model=intent.stt.qwen_asr.model,
        qwen_region=intent.translation.qwen.region,
        soniox_model=intent.stt.soniox.model,
        soniox_endpoint=intent.stt.soniox.endpoint,
        soniox_keepalive_interval_s=intent.stt.soniox.keepalive_interval_s,
        soniox_trailing_silence_ms=intent.stt.soniox.trailing_silence_ms,
        soniox_enable_language_identification=automatic_soniox,
        soniox_language_hints=language_hints,
        soniox_language_hints_strict=language_hints_strict,
        custom_stt_mode=custom_mode,
        custom_stt_compatibility=custom_compatibility,
        custom_stt_endpoint=intent.stt.custom.endpoint,
        custom_stt_model=intent.stt.custom.model,
        custom_stt_extra=dict(intent.stt.custom.extra),
    )


def create_stt_backend(
    settings: AppSettings,
    *,
    secrets: SecretStore,
    diagnostics_enabled: Callable[[], bool] | None = None,
    gpu_runtime: SharedGpuASRRuntime | None = None,
    gpu_model_path: Path | None = None,
) -> STTBackend:
    resolved = resolve_stt_runtime_config(
        _self_stt_runtime_intent_from_compatibility_settings(settings)
    )
    return create_stt_backend_from_resolved_config(
        resolved,
        secrets=secrets,
        diagnostics_enabled=diagnostics_enabled,
        gpu_runtime=gpu_runtime,
        gpu_model_path=gpu_model_path,
        gpu_device_id=settings.stt.gpu_device_id,
    )


def resolve_self_stt_runtime_config(settings: AppSettings) -> ResolvedSTTConfig:
    return resolve_stt_runtime_config(
        _self_stt_runtime_intent_from_compatibility_settings(settings)
    )


def _self_stt_custom_vocabulary_signature(
    settings: AppSettings,
) -> tuple[bool, tuple[str, ...]]:
    if settings.provider.stt not in {
        STTProviderName.DEEPGRAM,
        STTProviderName.LOCAL_QWEN,
        STTProviderName.SONIOX,
    }:
        return False, ()
    if settings.provider.stt == STTProviderName.LOCAL_QWEN:
        return (
            settings.stt.custom_vocabulary_enabled,
            tuple(
                get_effective_local_qwen_hotwords(
                    build_custom_vocabulary_runtime_config(settings),
                    settings.languages.source_language,
                )
            ),
        )
    return (
        settings.stt.custom_vocabulary_enabled,
        tuple(
            get_effective_custom_terms(
                build_custom_vocabulary_runtime_config(settings),
                settings.languages.source_language,
            )
        ),
    )


def build_self_stt_runtime_signature(settings: AppSettings) -> tuple[object, ...]:
    custom_vocab_enabled, custom_terms = _self_stt_custom_vocabulary_signature(settings)
    return (
        settings.languages.source_language,
        settings.audio.input_host_api,
        settings.audio.input_device,
        settings.provider.stt,
        settings.stt.vad_speech_threshold,
        FIXED_TRANSLATION_POLICY.fast_translation_enabled,
        settings.stt.low_latency_merge_gap_ms,
        settings.stt.low_latency_spec_retry_max,
        settings.stt.low_latency_vad_hangover_ms,
        settings.stt.drain_timeout_s,
        settings.audio.ring_buffer_ms,
        settings.audio.internal_sample_rate_hz,
        settings.audio.internal_channels,
        (
            settings.stt.gpu_device_id
            if settings.provider.stt == STTProviderName.LOCAL_QWEN_GPU
            else None
        ),
        (
            settings.deepgram_stt.model
            if settings.provider.stt == STTProviderName.DEEPGRAM
            else None
        ),
        settings.qwen.region if settings.provider.stt == STTProviderName.QWEN_ASR else None,
        (
            settings.qwen_asr_stt.model
            if settings.provider.stt == STTProviderName.QWEN_ASR
            else None
        ),
        (
            settings.qwen_asr_stt.endpoint
            if settings.provider.stt == STTProviderName.QWEN_ASR
            else None
        ),
        (settings.soniox_stt.model if settings.provider.stt == STTProviderName.SONIOX else None),
        (settings.soniox_stt.endpoint if settings.provider.stt == STTProviderName.SONIOX else None),
        (
            settings.soniox_stt.keepalive_interval_s
            if settings.provider.stt == STTProviderName.SONIOX
            else None
        ),
        (
            settings.soniox_stt.trailing_silence_ms
            if settings.provider.stt == STTProviderName.SONIOX
            else None
        ),
        (settings.custom_stt.mode if is_custom_stt_provider(settings.provider.stt) else None),
        (
            settings.custom_stt.compatibility
            if is_custom_stt_provider(settings.provider.stt)
            else None
        ),
        (settings.custom_stt.endpoint if is_custom_stt_provider(settings.provider.stt) else None),
        (settings.custom_stt.model if is_custom_stt_provider(settings.provider.stt) else None),
        custom_vocab_enabled,
        custom_terms,
    )


def build_self_stt_provider_signature(settings: AppSettings) -> tuple[object, ...]:
    return (
        settings.provider.stt,
        (
            settings.deepgram_stt.model
            if settings.provider.stt == STTProviderName.DEEPGRAM
            else None
        ),
        settings.qwen.region if settings.provider.stt == STTProviderName.QWEN_ASR else None,
        (
            settings.qwen_asr_stt.model
            if settings.provider.stt == STTProviderName.QWEN_ASR
            else None
        ),
        (settings.soniox_stt.model if settings.provider.stt == STTProviderName.SONIOX else None),
        (settings.soniox_stt.endpoint if settings.provider.stt == STTProviderName.SONIOX else None),
        (
            settings.soniox_stt.keepalive_interval_s
            if settings.provider.stt == STTProviderName.SONIOX
            else None
        ),
        (
            settings.soniox_stt.trailing_silence_ms
            if settings.provider.stt == STTProviderName.SONIOX
            else None
        ),
        (settings.custom_stt.mode if is_custom_stt_provider(settings.provider.stt) else None),
        (
            settings.custom_stt.compatibility
            if is_custom_stt_provider(settings.provider.stt)
            else None
        ),
        (settings.custom_stt.endpoint if is_custom_stt_provider(settings.provider.stt) else None),
        (settings.custom_stt.model if is_custom_stt_provider(settings.provider.stt) else None),
        (custom_stt_secret_generation() if is_custom_stt_provider(settings.provider.stt) else None),
        (
            str(default_local_stt_model_dir())
            if settings.provider.stt == STTProviderName.LOCAL_QWEN
            else None
        ),
        (
            settings.stt.gpu_device_id
            if settings.provider.stt == STTProviderName.LOCAL_QWEN_GPU
            else None
        ),
    )


def build_self_capture_vad_signature(settings: AppSettings) -> tuple[object, ...]:
    return (
        settings.audio.input_host_api,
        settings.audio.input_device,
        settings.stt.vad_speech_threshold,
        settings.stt.low_latency_vad_hangover_ms,
        settings.audio.ring_buffer_ms,
        settings.audio.internal_sample_rate_hz,
        settings.audio.internal_channels,
        settings.stt.gpu_device_id,
    )


def build_local_asr_session_options(
    *,
    source_language: str,
    source_mode: str = "manual",
) -> LocalASRSessionOptions:
    return LocalASRSessionOptions(
        source_language=source_language,
        source_mode=source_mode,
        language_hint=(
            None if source_mode == "auto" else get_local_qwen_language_hint(source_language)
        ),
    )


def build_self_local_asr_transition_request(
    settings: AppSettings,
    *,
    trigger: str,
) -> LocalASRTransitionRequest | None:
    provider = settings.provider.stt.value
    if provider == STTProviderName.LOCAL_QWEN_GPU.value:
        model_id = LOCAL_QWEN_GPU_MODEL_ID
        actual_provider = provider
    elif provider in LOCAL_CPU_PROVIDERS:
        decision = resolve_local_asr_selection(
            provider,
            settings.languages.source_language,
        )
        if not decision.supported:
            return None
        model_id = decision.model_id
        actual_provider = decision.effective_provider
    else:
        return None
    return LocalASRTransitionRequest(
        channel="self",
        requested_provider=provider,
        actual_provider=actual_provider,
        model_id=model_id,
        session_options=build_local_asr_session_options(
            source_language=settings.languages.source_language,
        ),
        trigger=trigger,
    )


def build_self_stt_provider_request(
    settings: AppSettings,
    *,
    warmup: bool = False,
) -> ProviderRuntimeBuildRequest:
    config = resolve_self_stt_runtime_config(settings)
    transition = build_self_local_asr_transition_request(settings, trigger="runtime")
    return ProviderRuntimeBuildRequest(
        config=config,
        gpu_device_id=settings.stt.gpu_device_id,
        warmup=warmup,
        model_id=transition.model_id if transition is not None else config.model,
        session_options=(
            transition.session_options
            if transition is not None
            else build_local_asr_session_options(
                source_language=config.source_language,
                source_mode=config.source_mode,
            )
        ),
    )


def build_self_capture_session_config(settings: AppSettings) -> SelfCaptureSessionConfig:
    provider = settings.provider.stt.value
    transition = build_self_local_asr_transition_request(settings, trigger="runtime")
    return SelfCaptureSessionConfig(
        provider_id=provider,
        provider_signature=build_self_stt_provider_signature(settings),
        runtime_signature=build_self_stt_runtime_signature(settings),
        capture_signature=build_self_capture_vad_signature(settings),
        target_sample_rate_hz=settings.audio.internal_sample_rate_hz,
        input_host_api=settings.audio.input_host_api,
        input_device=settings.audio.input_device,
        internal_channels=settings.audio.internal_channels,
        ring_buffer_ms=settings.audio.ring_buffer_ms,
        vad_speech_threshold=settings.stt.vad_speech_threshold,
        vad_hangover_ms=(
            settings.stt.low_latency_vad_hangover_ms
            if FIXED_TRANSLATION_POLICY.fast_translation_enabled
            else 1100
        ),
        session_options=transition.session_options if transition is not None else None,
        local_cpu=provider in LOCAL_CPU_PROVIDERS,
        local_gpu=provider == STTProviderName.LOCAL_QWEN_GPU.value,
        release_backend_after=(
            LOCAL_QWEN_IDLE_RELEASE_SECONDS if provider in LOCAL_CPU_PROVIDERS else None
        ),
        warmup=provider != STTProviderName.LOCAL_QWEN.value,
    )


def _resolved_stt_keyterms(config: ResolvedSTTConfig) -> tuple[str, ...]:
    if not config.custom_vocabulary_enabled:
        return ()
    exact_terms = config.custom_terms.get(config.source_language)
    if exact_terms is not None:
        return tuple(exact_terms)
    base_language = config.source_language.split("-")[0].lower()
    return tuple(config.custom_terms.get(base_language, ()))


def _resolved_float_option(
    options: Mapping[str, object],
    key: str,
    *,
    default: float,
) -> float:
    value = options.get(key)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return default


def _resolved_int_option(
    options: Mapping[str, object],
    key: str,
    *,
    default: int,
) -> int:
    value = options.get(key)
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return default


def _resolved_bool_option(
    options: Mapping[str, object],
    key: str,
    *,
    default: bool,
) -> bool:
    value = options.get(key)
    return value if isinstance(value, bool) else default


def _resolved_soniox_language_hints(config: ResolvedSTTConfig) -> list[str]:
    value = config.provider_options.get("language_hints")
    if isinstance(value, tuple) and all(isinstance(language, str) for language in value):
        return list(value)
    return []


def _deepgram_api_key_for_resolved_credential(
    credential: ResolvedCredentialRequirement,
    *,
    secrets: SecretStore,
) -> str:
    if credential.reference not in (CREDENTIAL_REF_DEEPGRAM_STT, None):
        raise ValueError("Unsupported Deepgram resolved credential reference")
    return require_secret(secrets, key="deepgram_api_key", env_var="DEEPGRAM_API_KEY")


def _soniox_api_key_for_resolved_credential(
    credential: ResolvedCredentialRequirement,
    *,
    secrets: SecretStore,
) -> str:
    if credential.reference not in (CREDENTIAL_REF_SONIOX_STT, None):
        raise ValueError("Unsupported Soniox resolved credential reference")
    return require_secret(secrets, key="soniox_api_key", env_var="SONIOX_API_KEY")


def _qwen_asr_endpoint_for_resolved_config(config: ResolvedSTTConfig) -> str:
    if config.endpoint:
        return config.endpoint
    if config.region == QwenRegion.SINGAPORE.value or (
        config.credential.reference == CREDENTIAL_REF_QWEN_SINGAPORE
    ):
        return "wss://dashscope-intl.aliyuncs.com/api-ws/v1/realtime"
    return "wss://dashscope.aliyuncs.com/api-ws/v1/realtime"


def create_stt_backend_from_resolved_config(
    config: ResolvedSTTConfig,
    *,
    secrets: SecretStore,
    diagnostics_enabled: Callable[[], bool] | None = None,
    gpu_runtime: SharedGpuASRRuntime | None = None,
    gpu_model_path: Path | None = None,
    gpu_device_id: str = "auto",
) -> STTBackend:
    stream_label = config.channel
    keyterms = _resolved_stt_keyterms(config)

    local_cpu_model_by_provider = {
        STT_PROVIDER_LOCAL_PARAKEET_V3: PARAKEET_V3_MODEL_ID,
        STT_PROVIDER_LOCAL_PARAKEET_JAPANESE: PARAKEET_JAPANESE_MODEL_ID,
        STT_PROVIDER_LOCAL_QWEN: LOCAL_STT_MODEL_ID,
    }
    if config.provider == STT_PROVIDER_LOCAL_CPU_AUTO:
        from puripuly_heart.providers.stt.local_cpu import LocalCPUAutoSTTBackend

        return LocalCPUAutoSTTBackend(
            source_language=config.source_language,
            sample_rate_hz=config.sample_rate_hz,
            stream_label=stream_label,
            hotwords=keyterms,
            diagnostics_enabled=diagnostics_enabled,
        )
    if config.provider in local_cpu_model_by_provider:
        from puripuly_heart.providers.stt.local_cpu import create_local_cpu_backend

        return create_local_cpu_backend(
            local_cpu_model_by_provider[config.provider],
            source_language=config.source_language,
            sample_rate_hz=config.sample_rate_hz,
            stream_label=stream_label,
            hotwords=() if config.provider == STT_PROVIDER_LOCAL_QWEN else keyterms,
            diagnostics_enabled=diagnostics_enabled,
        )
    if config.provider == STT_PROVIDER_LOCAL_QWEN_GPU:
        if gpu_runtime is None:
            raise RuntimeError("Local Vulkan ASR worker is not available")
        if config.channel not in {"self", "peer"}:
            raise ValueError(f"Unsupported GPU ASR channel: {config.channel}")
        from puripuly_heart.core.local_gpu_assets import local_gpu_model_path
        from puripuly_heart.providers.stt.local_gpu import LocalGpuSTTBackend

        return LocalGpuSTTBackend(
            runtime=gpu_runtime,
            channel=config.channel,
            model_path=gpu_model_path or local_gpu_model_path(),
            model_id=LOCAL_QWEN_GPU_MODEL_ID,
            device_id=gpu_device_id,
            sample_rate_hz=config.sample_rate_hz,
            source_mode=config.source_mode,
            language_hint=(
                None
                if config.source_mode == "auto"
                else get_local_qwen_language_hint(config.source_language)
            ),
        )

    if config.provider == STT_PROVIDER_DEEPGRAM:
        from puripuly_heart.core.language import get_deepgram_language
        from puripuly_heart.providers.stt.deepgram import DeepgramRealtimeSTTBackend

        api_key = _deepgram_api_key_for_resolved_credential(config.credential, secrets=secrets)
        return DeepgramRealtimeSTTBackend(
            api_key=api_key,
            model=config.model or "nova-3",
            language=get_deepgram_language(config.source_language),
            sample_rate_hz=config.sample_rate_hz,
            keyterms=keyterms,
            stream_label=stream_label,
        )

    if config.provider == STT_PROVIDER_QWEN_ASR:
        from puripuly_heart.core.language import get_qwen_asr_language
        from puripuly_heart.providers.stt.qwen_asr import QwenASRRealtimeSTTBackend

        api_key = _qwen_api_key_for_resolved_credential(config.credential, secrets=secrets)
        return QwenASRRealtimeSTTBackend(
            api_key=api_key,
            model=config.model or "qwen3-asr-flash-realtime",
            endpoint=_qwen_asr_endpoint_for_resolved_config(config),
            language=get_qwen_asr_language(config.source_language),
            sample_rate_hz=config.sample_rate_hz,
        )

    if config.provider == STT_PROVIDER_SONIOX:
        from puripuly_heart.providers.stt.soniox import SonioxRealtimeSTTBackend

        api_key = _soniox_api_key_for_resolved_credential(config.credential, secrets=secrets)
        return SonioxRealtimeSTTBackend(
            api_key=api_key,
            model=config.model or SONIOX_STT_MODEL_RT_V5,
            endpoint=config.endpoint or "wss://stt-rt.soniox.com/transcribe-websocket",
            language_hints=_resolved_soniox_language_hints(config),
            sample_rate_hz=config.sample_rate_hz,
            keepalive_interval_s=_resolved_float_option(
                config.provider_options,
                "keepalive_interval_s",
                default=SONIOX_STT_DEFAULT_KEEPALIVE_INTERVAL_S,
            ),
            trailing_silence_ms=_resolved_int_option(
                config.provider_options,
                "trailing_silence_ms",
                default=SONIOX_STT_DEFAULT_TRAILING_SILENCE_MS,
            ),
            enable_language_identification=_resolved_bool_option(
                config.provider_options,
                "enable_language_identification",
                default=False,
            ),
            language_hints_strict=_resolved_bool_option(
                config.provider_options,
                "language_hints_strict",
                default=False,
            ),
            context_terms=keyterms,
        )

    if config.provider in STT_CUSTOM_PROVIDERS:
        from puripuly_heart.providers.stt.custom import CustomSTTBackend

        api_key = ""
        if config.credential.reference == CREDENTIAL_REF_CUSTOM_STT:
            api_key = (secrets.get("custom_stt_api_key") or "").strip()
        mode = str(config.provider_options.get("mode") or "")
        compatibility = str(config.provider_options.get("compatibility") or "")
        extra = config.provider_options.get("extra")
        extra_mapping = extra if isinstance(extra, Mapping) else {}
        return CustomSTTBackend(
            mode=mode,
            compatibility=compatibility,
            endpoint=config.endpoint or "",
            model=config.model or "",
            api_key=api_key,
            source_language=config.source_language,
            sample_rate_hz=config.sample_rate_hz,
            extra=dict(extra_mapping),
        )

    raise ValueError(f"Unsupported STT provider: {config.provider}")


def resolve_peer_stt_config(settings: AppSettings) -> ResolvedPeerSTTConfig:
    peer_source_language = settings.languages.effective_peer_source
    keyterms: tuple[str, ...] = ()
    provider = _stt_provider_name_or_raise(settings.provider.peer_stt, peer=True)

    if provider == STTProviderName.DEEPGRAM:
        return ResolvedPeerSTTConfig(
            provider=provider,
            source_language=peer_source_language,
            sample_rate_hz=STT_INTERNAL_SAMPLE_RATE_HZ,
            keyterms=keyterms,
            deepgram_model=settings.deepgram_stt.model,
        )

    if provider == STTProviderName.QWEN_ASR:
        return ResolvedPeerSTTConfig(
            provider=provider,
            source_language=peer_source_language,
            sample_rate_hz=STT_INTERNAL_SAMPLE_RATE_HZ,
            keyterms=keyterms,
            qwen_model=settings.qwen_asr_stt.model,
            qwen_region=settings.qwen.region,
        )

    if provider == STTProviderName.SONIOX:
        automatic_soniox = settings.languages.peer_source_mode == "auto"
        from puripuly_heart.core.language import get_soniox_language_hints

        if automatic_soniox:
            mapped_language_hints = tuple(
                dict.fromkeys(
                    hint
                    for language in settings.languages.peer_expected_languages
                    for hint in get_soniox_language_hints(language)
                )
            )
            language_hints = mapped_language_hints or None
        else:
            language_hints = tuple(get_soniox_language_hints(peer_source_language))
        return ResolvedPeerSTTConfig(
            provider=provider,
            source_language=peer_source_language,
            sample_rate_hz=STT_INTERNAL_SAMPLE_RATE_HZ,
            keyterms=keyterms,
            soniox_model=settings.soniox_stt.model,
            soniox_endpoint=settings.soniox_stt.endpoint,
            soniox_keepalive_interval_s=settings.soniox_stt.keepalive_interval_s,
            soniox_trailing_silence_ms=settings.soniox_stt.trailing_silence_ms,
            soniox_enable_language_identification=automatic_soniox,
            soniox_language_hints=language_hints,
        )

    if is_custom_stt_provider(provider):
        return ResolvedPeerSTTConfig(
            provider=provider,
            source_language=peer_source_language,
            sample_rate_hz=STT_INTERNAL_SAMPLE_RATE_HZ,
            keyterms=(),
        )

    if provider in {
        STTProviderName.LOCAL_CPU_AUTO,
        STTProviderName.LOCAL_PARAKEET_V3,
        STTProviderName.LOCAL_PARAKEET_JAPANESE,
        STTProviderName.LOCAL_QWEN,
        STTProviderName.LOCAL_QWEN_GPU,
    }:
        return ResolvedPeerSTTConfig(
            provider=provider,
            source_language=peer_source_language,
            sample_rate_hz=STT_INTERNAL_SAMPLE_RATE_HZ,
            keyterms=(),
        )

    raise ValueError(f"Unsupported peer STT provider: {provider}")


def _resolved_peer_stt_config_from_compatibility_settings(
    settings: AppSettings,
) -> ResolvedSTTConfig:
    return resolve_stt_runtime_config(
        _peer_stt_runtime_intent_from_compatibility_settings(settings)
    )


def resolve_peer_stt_runtime_config(settings: AppSettings) -> ResolvedSTTConfig:
    return _resolved_peer_stt_config_from_compatibility_settings(settings)


def resolve_peer_stt_runtime_config_from_vnext(settings: AppSettingsVNext) -> ResolvedSTTConfig:
    return resolve_stt_runtime_config(peer_stt_runtime_intent_from_vnext(settings))


def build_peer_stt_provider_signature(settings: AppSettings) -> tuple[object, ...]:
    from puripuly_heart.config.settings_vnext.migration import from_legacy_app_settings

    return build_peer_stt_provider_signature_from_vnext(from_legacy_app_settings(settings))


def build_peer_stt_provider_signature_from_vnext(settings: AppSettingsVNext) -> tuple[object, ...]:
    resolved = resolve_peer_stt_runtime_config_from_vnext(settings)
    return (
        resolved.provider,
        resolved.source_language,
        resolved.sample_rate_hz,
        resolved.model,
        resolved.endpoint,
        resolved.region,
        resolved.provider_options.get("keepalive_interval_s"),
        resolved.provider_options.get("trailing_silence_ms"),
        resolved.provider_options.get("enable_language_identification", False),
        resolved.provider_options.get("language_hints"),
        (
            settings.intent.stt.gpu_device_id
            if resolved.provider == STT_PROVIDER_LOCAL_QWEN_GPU
            else None
        ),
        resolved.provider_options.get("language_hints_strict", False),
        resolved.provider_options.get("mode"),
        resolved.provider_options.get("compatibility"),
        resolved.source_mode,
        (custom_stt_secret_generation() if resolved.provider in STT_CUSTOM_PROVIDERS else None),
    )


def build_peer_capture_session_config(
    settings: AppSettings,
    *,
    canonical_settings: AppSettingsVNext | None = None,
) -> PeerCaptureSessionConfig:
    if canonical_settings is None:
        from puripuly_heart.config.settings_vnext.migration import from_legacy_app_settings

        canonical_settings = from_legacy_app_settings(settings)
    return build_peer_capture_session_config_from_vnext(canonical_settings)


def build_peer_capture_session_config_from_vnext(
    settings: AppSettingsVNext,
) -> PeerCaptureSessionConfig:
    backend = resolve_peer_stt_runtime_config_from_vnext(settings)
    provider_signature = build_peer_stt_provider_signature_from_vnext(settings)
    desktop_audio = settings.intent.desktop_audio
    capture_target = resolve_desktop_audio_capture_target(desktop_audio.capture_target)
    target = PeerCaptureTargetIntent(
        kind=capture_target.kind,
        device_name=capture_target.device_name,
        process_kind=capture_target.process_kind,
        executable_identity=capture_target.executable_identity,
        discord_channel=capture_target.discord_channel,
        executable_basename=capture_target.executable_basename,
    )
    model_id = None
    if backend.provider == STTProviderName.LOCAL_QWEN_GPU.value:
        model_id = LOCAL_QWEN_GPU_MODEL_ID
    elif backend.provider in LOCAL_CPU_PROVIDERS:
        model_id = resolve_local_asr_selection(
            backend.provider,
            backend.source_language,
        ).model_id
    local_provider = backend.provider in {
        *LOCAL_CPU_PROVIDERS,
        STTProviderName.LOCAL_QWEN_GPU.value,
    }
    session_options = (
        build_local_asr_session_options(
            source_language=backend.source_language,
            source_mode=backend.source_mode,
        )
        if local_provider
        else None
    )
    capture_signature = (
        desktop_audio.output_device,
        capture_target,
        desktop_audio.vad_speech_threshold,
        desktop_audio.vad_hangover_ms,
        desktop_audio.vad_pre_roll_ms,
        backend.sample_rate_hz,
    )
    return PeerCaptureSessionConfig(
        provider_id=backend.provider,
        output_device=desktop_audio.output_device,
        vad_speech_threshold=desktop_audio.vad_speech_threshold,
        vad_hangover_ms=desktop_audio.vad_hangover_ms,
        vad_pre_roll_ms=desktop_audio.vad_pre_roll_ms,
        provider_signature=provider_signature,
        runtime_signature=(
            backend.source_language,
            desktop_audio.output_device,
            target,
            desktop_audio.vad_speech_threshold,
            desktop_audio.vad_hangover_ms,
            desktop_audio.vad_pre_roll_ms,
            provider_signature,
        ),
        capture_signature=capture_signature,
        capture_target=target,
        language=PeerCaptureLanguageFacts(
            source_mode=backend.source_mode,
            source_language=backend.source_language,
            expected_languages=tuple(settings.intent.languages.peer_expected_languages),
        ),
        target_sample_rate_hz=backend.sample_rate_hz,
        model_id=model_id,
        session_options=session_options,
        provider_context=backend,
        local_provider=local_provider,
        release_backend_after=(
            LOCAL_QWEN_IDLE_RELEASE_SECONDS
            if backend.provider == STTProviderName.LOCAL_QWEN.value
            else None
        ),
        warmup=backend.provider != STTProviderName.LOCAL_QWEN.value,
    )


def build_peer_stt_runtime_signature(
    settings: AppSettings,
    *,
    canonical_settings: AppSettingsVNext | None = None,
) -> tuple[object, ...]:
    return build_peer_capture_session_config(
        settings,
        canonical_settings=canonical_settings,
    ).runtime_signature


def build_peer_stt_provider_request(
    config: PeerCaptureSessionConfig,
    *,
    gpu_device_id: str,
    warmup: bool = False,
) -> ProviderRuntimeBuildRequest:
    backend = config.provider_context
    if not isinstance(backend, ResolvedSTTConfig):
        raise TypeError("Peer capture config requires a resolved STT provider context")
    return ProviderRuntimeBuildRequest(
        config=backend,
        gpu_device_id=gpu_device_id,
        warmup=warmup,
        model_id=config.model_id or backend.model,
        session_options=config.session_options,
    )


def create_peer_stt_backend(
    settings: AppSettings,
    *,
    secrets: SecretStore,
    diagnostics_enabled: Callable[[], bool] | None = None,
    gpu_runtime: SharedGpuASRRuntime | None = None,
    gpu_model_path: Path | None = None,
) -> STTBackend:
    resolved = resolve_peer_stt_runtime_config(settings)
    return create_peer_stt_backend_from_resolved_config(
        resolved,
        secrets=secrets,
        diagnostics_enabled=diagnostics_enabled,
        gpu_runtime=gpu_runtime,
        gpu_model_path=gpu_model_path,
        gpu_device_id=settings.stt.gpu_device_id,
    )


def create_peer_stt_backend_from_resolved_config(
    config: ResolvedSTTConfig,
    *,
    secrets: SecretStore,
    diagnostics_enabled: Callable[[], bool] | None = None,
    gpu_runtime: SharedGpuASRRuntime | None = None,
    gpu_model_path: Path | None = None,
    gpu_device_id: str = "auto",
) -> STTBackend:
    return create_stt_backend_from_resolved_config(
        config,
        secrets=secrets,
        diagnostics_enabled=diagnostics_enabled,
        gpu_runtime=gpu_runtime,
        gpu_model_path=gpu_model_path,
        gpu_device_id=gpu_device_id,
    )
