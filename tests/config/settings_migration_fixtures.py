from __future__ import annotations

import copy
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any

from puripuly_heart.config.audio_host_api import WINDOWS_DIRECTSOUND_HOST_API
from puripuly_heart.config.settings import (
    DEFAULT_DESKTOP_AUDIO_VAD_HANGOVER_MS,
    DEFAULT_OPENROUTER_BROKER_BASE_URL,
    OVERLAY_TARGET_DESKTOP,
    AppSettings,
    CerebrasLLMModel,
    DeepSeekLLMModel,
    GeminiLLMModel,
    OpenRouterCredentialSource,
    OpenRouterFallbackSelectionAlias,
    OpenRouterLLMModel,
    OpenRouterProviderRouting,
    OpenRouterRoutingMode,
    OpenRouterSelectionAlias,
    QwenLLMModel,
    QwenRegion,
    SecretsBackend,
    STTProviderName,
    TranslationConnection,
    TranslationFallbackSettings,
    TranslationModel,
    TranslationSettings,
    from_dict,
    to_dict,
)
from puripuly_heart.ui.overlay_calibration import OverlayCalibration

MAXIMAL_V24_FIXTURE_NAME = "maximal_v24_settings"
LEGACY_COMPATIBILITY_FIXTURE_NAME = "legacy_compatibility_settings"
MISSING_DEFAULTS_FIXTURE_NAME = "missing_field_defaults"

DYNAMIC_MAPPING_PATHS = frozenset(
    {
        "local_llm.extra_body",
        "stt.custom_terms",
        "translation.connection_history",
        "custom_stt.extra",
    }
)

ADR_RESOLVED_CURRENT_DESTINATIONS = {
    "ui.peer_translation_eula_accepted": "state.peer_translation.eula_accepted",
    "ui.integrated_context_enabled": "intent.integrated_context.enabled",
    "ui.integrated_context_bootstrapped": "state.integrated_context.bootstrapped",
}
DECISION_PENDING_CURRENT_DESTINATIONS: dict[str, str] = {}
VNEXT_NATIVE_PERSISTED_LEAF_PATHS = frozenset(
    {
        "intent.desktop_audio.capture_target.kind",
        "intent.desktop_audio.capture_target.device_name",
        "intent.desktop_audio.capture_target.process",
        "intent.translation.http_extension_id",
        "intent.translation.previous_llm_model",
    }
) | frozenset(
    f"state.provider_verification.{provider}.{field}"
    for provider in (
        "alibaba_beijing",
        "alibaba_singapore",
        "cerebras",
        "deepgram",
        "deepseek",
        "google",
        "openrouter",
        "soniox",
    )
    for field in (
        "provider",
        "secret_key",
        "secret_revision",
        "secret_fingerprint",
        "verifier_context",
        "verifier_evidence",
    )
)

CURRENT_USER_INTENT_DESTINATIONS = {
    "audio.input_device": "intent.audio.input_device",
    "audio.input_host_api": "intent.audio.input_host_api",
    "audio.ring_buffer_ms": "intent.audio.ring_buffer_ms",
    "cerebras.llm_model": "intent.translation.cerebras.llm_model",
    "custom_stt.compatibility": "intent.stt.custom.compatibility",
    "custom_stt.endpoint": "intent.stt.custom.endpoint",
    "custom_stt.extra": "intent.stt.custom.extra",
    "custom_stt.mode": "intent.stt.custom.mode",
    "custom_stt.model": "intent.stt.custom.model",
    "deepseek.llm_model": "intent.translation.deepseek.llm_model",
    "deepgram_stt.model": "intent.stt.deepgram.model",
    "desktop_audio.output_device": "intent.desktop_audio.output_device",
    "desktop_audio.vad_hangover_ms": "intent.desktop_audio.vad_hangover_ms",
    "desktop_audio.vad_pre_roll_ms": "intent.desktop_audio.vad_pre_roll_ms",
    "desktop_audio.vad_speech_threshold": "intent.desktop_audio.vad_speech_threshold",
    "languages.peer_source_language": "intent.languages.peer_source_language",
    "languages.peer_target_language": "intent.languages.peer_target_language",
    "languages.peer_source_mode": "intent.languages.peer_source_mode",
    "languages.peer_expected_languages": "intent.languages.peer_expected_languages",
    "languages.recent_source_languages": "intent.languages.recent_source_languages",
    "languages.recent_target_languages": "intent.languages.recent_target_languages",
    "languages.source_language": "intent.languages.source_language",
    "languages.target_language": "intent.languages.target_language",
    "llm.concurrency_limit": "intent.translation.concurrency_limit",
    "gemini.llm_model": "intent.translation.gemini.llm_model",
    "local_llm.base_url": "intent.local_llm.base_url",
    "local_llm.extra_body": "intent.local_llm.extra_body",
    "local_llm.model": "intent.local_llm.model",
    "openrouter.broker_base_url": "intent.translation.openrouter_broker_base_url",
    "openrouter.fallback_selection_alias": "intent.translation.fallback.selection_alias",
    "openrouter.llm_model": "intent.translation.openrouter_model",
    "openrouter.provider_routing": "intent.translation.openrouter_provider_routing",
    "openrouter.routing_mode": "intent.translation.openrouter_routing_mode",
    "openrouter.selected_source": "intent.translation.openrouter_selected_source",
    "openrouter.selection_alias": "intent.translation.openrouter_selection_alias",
    "telemetry.consent": "intent.telemetry.consent",
    "translation.fallback.connection": "intent.translation.fallback.connection",
    "translation.fallback.enabled": "intent.translation.fallback.enabled",
    "translation.fallback.model": "intent.translation.fallback.model",
    "osc.chatbox_address": "intent.osc.chatbox_address",
    "osc.chatbox_clear": "intent.osc.chatbox_clear",
    "osc.chatbox_include_source": "intent.osc.chatbox_include_source",
    "osc.chatbox_max_chars": "intent.osc.chatbox_max_chars",
    "osc.chatbox_send": "intent.osc.chatbox_send",
    "osc.connection_mode": "intent.osc.connection_mode",
    "osc.host": "intent.osc.host",
    "osc.port": "intent.osc.port",
    "osc.receive_port": "intent.osc.receive_port",
    "osc.send_port": "intent.osc.send_port",
    "osc.vrc_mic_intercept": "intent.osc.vrc_mic_intercept",
    "overlay.calibration.background_alpha": "intent.overlay.calibration.background_alpha",
    "overlay.calibration.distance": "intent.overlay.calibration.distance",
    "overlay.calibration.offset_x": "intent.overlay.calibration.offset_x",
    "overlay.calibration.offset_y": "intent.overlay.calibration.offset_y",
    "overlay.calibration.text_scale": "intent.overlay.calibration.text_scale",
    "overlay.desktop_flet.position.x": "intent.overlay.desktop_flet.position.x",
    "overlay.desktop_flet.position.y": "intent.overlay.desktop_flet.position.y",
    "overlay.desktop_flet.size_preset": "intent.overlay.desktop_flet.size_preset",
    "overlay.desktop_flet.swap_caption_languages": (
        "intent.overlay.desktop_flet.swap_caption_languages"
    ),
    "overlay.desktop_flet.visual.background_alpha": (
        "intent.overlay.desktop_flet.visual.background_alpha"
    ),
    "overlay.show_peer_original": "intent.overlay.show_peer_original",
    "overlay.show_translation": "intent.overlay.show_translation",
    "overlay.target": "intent.overlay.target",
    "provider.peer_stt": "intent.peer_stt.provider",
    "provider.stt": "intent.stt.provider",
    "qwen.llm_model": "intent.translation.qwen.llm_model",
    "qwen.region": "intent.translation.qwen.region",
    "qwen_asr_stt.model": "intent.stt.qwen_asr.model",
    "secrets.backend": "intent.secrets.backend",
    "secrets.encrypted_file_path": "intent.secrets.encrypted_file_path",
    "soniox_stt.endpoint": "intent.stt.soniox.endpoint",
    "soniox_stt.keepalive_interval_s": "intent.stt.soniox.keepalive_interval_s",
    "soniox_stt.model": "intent.stt.soniox.model",
    "soniox_stt.trailing_silence_ms": "intent.stt.soniox.trailing_silence_ms",
    "stt.custom_terms": "intent.stt.custom_terms",
    "stt.custom_vocabulary_enabled": "intent.stt.custom_vocabulary_enabled",
    "stt.drain_timeout_s": "intent.stt.drain_timeout_s",
    "stt.gpu_device_id": "intent.stt.gpu_device_id",
    "stt.low_latency_merge_gap_ms": "intent.stt.low_latency_merge_gap_ms",
    "stt.low_latency_mode": "intent.stt.low_latency_mode",
    "stt.low_latency_spec_retry_max": "intent.stt.low_latency_spec_retry_max",
    "stt.low_latency_vad_hangover_ms": "intent.stt.low_latency_vad_hangover_ms",
    "stt.vad_speech_threshold": "intent.stt.vad_speech_threshold",
    "system_prompt": "intent.prompts.system_prompt",
    "translation.connection": "intent.translation.connection",
    "translation.connection_history": "intent.translation.connection_history",
    "translation.gpu_device_id": "intent.translation.gpu_device_id",
    "translation.model": "intent.translation.model",
    "ui.clipboard_auto_translate_enabled": "intent.clipboard.auto_translate_enabled",
    "ui.locale": "intent.ui.locale",
}

CURRENT_COMPATIBILITY_INPUT_DESTINATIONS = {
    "provider.llm": "compatibility_input.provider.llm",
    "qwen_asr_stt.endpoint": "compatibility_input.qwen_asr_stt.endpoint",
}

CURRENT_OPERATIONAL_STATE_DESTINATIONS = {
    "api_key_verified.alibaba_beijing": "state.provider_verification.alibaba_beijing.status",
    "api_key_verified.alibaba_singapore": "state.provider_verification.alibaba_singapore.status",
    "api_key_verified.cerebras": "state.provider_verification.cerebras.status",
    "api_key_verified.deepgram": "state.provider_verification.deepgram.status",
    "api_key_verified.deepseek": "state.provider_verification.deepseek.status",
    "api_key_verified.google": "state.provider_verification.google.status",
    "api_key_verified.openrouter": "state.provider_verification.openrouter.status",
    "api_key_verified.soniox": "state.provider_verification.soniox.status",
    "managed_identity.active_managed_credential_ref": (
        "state.managed_connection.active_managed_credential_ref"
    ),
    "managed_identity.active_managed_expires_at": (
        "state.managed_connection.active_managed_expires_at"
    ),
    "managed_identity.founder_letter_seen_credential_ref": (
        "state.managed_connection.founder_letter_seen_credential_ref"
    ),
    "managed_identity.installation_id": "state.managed_connection.installation_id",
    "managed_identity.local_managed_claim_sources": (
        "state.managed_connection.local_managed_claim_sources"
    ),
    "managed_identity.pending_delivery_ack_delivery_id": (
        "state.managed_connection.pending_delivery_ack_delivery_id"
    ),
    "managed_identity.pending_delivery_ack_expires_at": (
        "state.managed_connection.pending_delivery_ack_expires_at"
    ),
    "managed_identity.pending_delivery_ack_managed_credential_ref": (
        "state.managed_connection.pending_delivery_ack_managed_credential_ref"
    ),
    "managed_identity.pending_delivery_ack_source": (
        "state.managed_connection.pending_delivery_ack_source"
    ),
    "managed_identity.referral_id": "state.managed_connection.referral_id",
    "managed_identity.release_token": "state.managed_connection.release_token",
    "managed_identity.release_token_expires_at": (
        "state.managed_connection.release_token_expires_at"
    ),
    "managed_identity.verified_hardware_hash": ("state.managed_connection.verified_hardware_hash"),
    "managed_identity.verified_hardware_hash_salt_version": (
        "state.managed_connection.verified_hardware_hash_salt_version"
    ),
    "telemetry_state.anonymous_id": "state.telemetry.anonymous_id",
    "telemetry_state.sent_translation_success_dates_utc": (
        "state.telemetry.sent_translation_success_dates_utc"
    ),
    "ui.github_star_prompt_clicked": "state.github_star_prompt.clicked",
    "ui.github_star_prompt_eligible_launch_count": (
        "state.github_star_prompt.eligible_launch_count"
    ),
    "ui.github_star_prompt_last_shown_at": "state.github_star_prompt.last_shown_at",
    "ui.github_star_prompt_show_count": "state.github_star_prompt.show_count",
    "ui.github_star_prompt_translation_success_observed": (
        "state.github_star_prompt.translation_success_observed"
    ),
}

REPAIR_TO_VNEXT_DEFAULT_DESTINATIONS = {
    "local_llm.backend": "intent.local_llm.backend",
}
REPAIR_TO_RUNTIME_CONSTANT_DESTINATIONS = {
    "audio.internal_channels": "runtime_resolution.audio.channels_constant",
    "audio.internal_sample_rate_hz": "runtime_resolution.audio.sample_rate_hz_constant",
}


@dataclass(frozen=True, slots=True)
class FieldClassification:
    category: str
    destination: str
    status: str
    fixture: str
    notes: str = ""
    missing_default_fixture: str = MISSING_DEFAULTS_FIXTURE_NAME


def serialized_field_paths(data: dict[str, Any], prefix: str = "") -> Iterator[str]:
    for key, value in data.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict) and path in DYNAMIC_MAPPING_PATHS:
            yield path
        elif isinstance(value, dict) and value:
            yield from serialized_field_paths(value, path)
        elif isinstance(value, dict):
            yield path
        else:
            yield path


def path_get(data: dict[str, Any], path: str) -> Any:
    current: Any = data
    for part in path.split("."):
        current = current[part]
    return current


def path_remove(data: dict[str, Any], path: str) -> None:
    parts = path.split(".")
    current: Any = data
    for part in parts[:-1]:
        current = current[part]
    current.pop(parts[-1], None)


EXPLICIT_MISSING_FIELD_DEFAULT_EXPECTATIONS: dict[str, Any] = {
    # Partial legacy settings default missing peer STT to Deepgram, while new AppSettings defaults
    # peer STT to local Qwen.
    "provider.peer_stt": STTProviderName.DEEPGRAM.value,
    # Broker URL is a public compatibility surface; missing values restore the production /v1 broker.
    "openrouter.broker_base_url": DEFAULT_OPENROUTER_BROKER_BASE_URL,
    # Current peer desktop-audio VAD default is intentionally lower than older schema defaults.
    "desktop_audio.vad_hangover_ms": DEFAULT_DESKTOP_AUDIO_VAD_HANGOVER_MS,
    # Missing vocabulary restores empty terms and keeps vocabulary enabled.
    "stt.custom_terms": {},
    # Integrated context remains enabled for missing legacy UI settings.
    "ui.integrated_context_enabled": True,
    # Clipboard watcher and GitHub prompt counters are opt-in/operational state defaults.
    "ui.clipboard_auto_translate_enabled": False,
    "ui.github_star_prompt_show_count": 0,
    # Missing referral identity state remains absent rather than inventing a value.
    "managed_identity.referral_id": None,
    "languages.peer_source_mode": "manual",
    "languages.peer_expected_languages": [],
}


def missing_field_default_expectations() -> dict[str, Any]:
    baseline = to_dict(AppSettings())
    expectations: dict[str, Any] = {}
    for path in serialized_field_paths(baseline):
        raw = copy.deepcopy(baseline)
        path_remove(raw, path)
        expectations[path] = path_get(to_dict(from_dict(raw)), path)
    return expectations


def maximal_v24_settings_fixture() -> dict[str, Any]:
    settings = AppSettings()
    settings.provider.stt = STTProviderName.DEEPGRAM
    settings.provider.peer_stt = STTProviderName.SONIOX
    settings.translation = TranslationSettings(
        model=TranslationModel.LOCAL_LLM,
        connection=TranslationConnection.OLLAMA,
        connection_history={
            TranslationModel.GEMMA4.value: TranslationConnection.OPENROUTER,
            TranslationModel.DEEPSEEK_V4_FLASH.value: TranslationConnection.MANAGED_CHINA,
            TranslationModel.GEMINI_37_FLASH.value: TranslationConnection.OFFICIAL_BYOK,
            TranslationModel.GEMINI_31_FLASH_LITE.value: TranslationConnection.OFFICIAL_BYOK,
            TranslationModel.QWEN_35_PLUS.value: TranslationConnection.OFFICIAL_BYOK,
            TranslationModel.LOCAL_LLM.value: TranslationConnection.OLLAMA,
        },
        fallback=TranslationFallbackSettings(
            enabled=True,
            model=TranslationModel.GEMINI_37_FLASH,
            connection=TranslationConnection.OPENROUTER,
        ),
        gpu_device_id="Vulkan1",
    )
    settings.languages.source_language = "ja"
    settings.languages.target_language = "zh-CN"
    settings.languages.peer_source_language = "fr"
    settings.languages.peer_target_language = "es"
    settings.languages.peer_source_mode = "auto"
    settings.languages.peer_expected_languages = ["fr", "ja"]
    settings.languages.recent_source_languages = ["fr", "de", "it"]
    settings.languages.recent_target_languages = ["es", "th", "vi"]
    settings.audio.ring_buffer_ms = 750
    settings.audio.input_host_api = WINDOWS_DIRECTSOUND_HOST_API
    settings.audio.input_device = "Fixture Microphone"
    settings.desktop_audio.output_device = "Fixture Speakers"
    settings.desktop_audio.vad_speech_threshold = 0.4
    settings.desktop_audio.vad_hangover_ms = 650
    settings.desktop_audio.vad_pre_roll_ms = 250
    settings.overlay.target = OVERLAY_TARGET_DESKTOP
    settings.overlay.show_translation = False
    settings.overlay.show_peer_original = False
    settings.overlay.calibration = OverlayCalibration(
        anchor="head_locked",
        offset_x=0.25,
        offset_y=-0.2,
        distance=1.8,
        text_scale=1.3,
        background_alpha=0.5,
    )
    settings.overlay.desktop_flet.size_preset = "large"
    settings.overlay.desktop_flet.position.x = 321
    settings.overlay.desktop_flet.position.y = 654
    settings.overlay.desktop_flet.swap_caption_languages = True
    settings.overlay.desktop_flet.visual.background_alpha = 0.42
    settings.stt.drain_timeout_s = 3.5
    settings.stt.vad_speech_threshold = 0.3
    settings.stt.low_latency_mode = False
    settings.stt.low_latency_vad_hangover_ms = 700
    settings.stt.low_latency_merge_gap_ms = 550
    settings.stt.low_latency_spec_retry_max = 5
    settings.stt.custom_vocabulary_enabled = False
    settings.stt.custom_terms = {"en": ["fixture-term"], "ja": ["フィクスチャ"]}
    settings.stt.gpu_device_id = "vulkan-device-fixture"
    settings.deepgram_stt.model = "nova-2"
    settings.qwen_asr_stt.model = "qwen-asr-fixture"
    settings.soniox_stt.model = "stt-rt-fixture"
    settings.soniox_stt.endpoint = "wss://soniox.fixture.test/transcribe"
    settings.soniox_stt.keepalive_interval_s = 12.5
    settings.soniox_stt.trailing_silence_ms = 250
    settings.custom_stt.mode = "realtime"
    settings.custom_stt.compatibility = "openai_realtime"
    settings.custom_stt.endpoint = "https://custom-stt.fixture.test"
    settings.custom_stt.model = "fixture-transcribe"
    settings.custom_stt.extra = {"prompt": "fixture-prompt", "max_tokens": 16}
    settings.gemini.llm_model = GeminiLLMModel.GEMINI_37_FLASH
    settings.openrouter.llm_model = OpenRouterLLMModel.QWEN_35_FLASH_02_23
    settings.openrouter.routing_mode = OpenRouterRoutingMode.LATENCY
    settings.openrouter.provider_routing = OpenRouterProviderRouting.DEEPSEEK_ONLY
    settings.openrouter.selected_source = OpenRouterCredentialSource.BYOK
    settings.openrouter.selection_alias = OpenRouterSelectionAlias.QWEN35_FLASH_BYOK
    settings.openrouter.fallback_selection_alias = OpenRouterFallbackSelectionAlias.NONE
    settings.openrouter.broker_base_url = "https://broker.fixture.test"
    settings.qwen.region = QwenRegion.SINGAPORE
    settings.qwen.llm_model = QwenLLMModel.QWEN_35_FLASH
    settings.deepseek.llm_model = DeepSeekLLMModel.DEEPSEEK_V4_FLASH
    settings.cerebras.llm_model = CerebrasLLMModel.GEMMA_4_31B
    settings.local_llm.base_url = "http://127.0.0.1:12345/v1"
    settings.local_llm.model = "fixture-local-model"
    settings.local_llm.extra_body = {"temperature": 0.25, "reasoning_effort": "low"}
    settings.llm.concurrency_limit = 7
    settings.osc.host = "192.0.2.25"
    settings.osc.port = 9012
    settings.osc.connection_mode = "manual"
    settings.osc.receive_port = 9013
    settings.osc.chatbox_address = "/fixture/chatbox"
    settings.osc.chatbox_send = False
    settings.osc.chatbox_clear = True
    settings.osc.chatbox_max_chars = 96
    settings.osc.vrc_mic_intercept = True
    settings.osc.chatbox_include_source = True
    settings.secrets.backend = SecretsBackend.ENCRYPTED_FILE
    settings.secrets.encrypted_file_path = "fixture-secrets.json"
    settings.ui.locale = "ja"
    settings.ui.peer_translation_eula_accepted = True
    settings.ui.integrated_context_enabled = False
    settings.ui.integrated_context_bootstrapped = True
    settings.ui.clipboard_auto_translate_enabled = True
    settings.ui.github_star_prompt_clicked = True
    settings.ui.github_star_prompt_last_shown_at = "2026-06-08T00:00:00Z"
    settings.ui.github_star_prompt_show_count = 2
    settings.ui.github_star_prompt_translation_success_observed = True
    settings.ui.github_star_prompt_eligible_launch_count = 3
    settings.api_key_verified.deepgram = True
    settings.api_key_verified.soniox = True
    settings.api_key_verified.google = True
    settings.api_key_verified.openrouter = True
    settings.api_key_verified.deepseek = True
    settings.api_key_verified.alibaba_beijing = True
    settings.api_key_verified.alibaba_singapore = True
    settings.api_key_verified.cerebras = True
    settings.managed_identity.installation_id = "fixture-installation-id"
    settings.managed_identity.release_token = "fixture-release-token"
    settings.managed_identity.release_token_expires_at = "2026-07-08T00:00:00Z"
    settings.managed_identity.verified_hardware_hash = "fixture-hardware-hash"
    settings.managed_identity.verified_hardware_hash_salt_version = 7
    settings.managed_identity.active_managed_credential_ref = "fixture-credential-ref"
    settings.managed_identity.active_managed_expires_at = "2026-07-09T00:00:00Z"
    settings.managed_identity.founder_letter_seen_credential_ref = "fixture-founder-ref"
    settings.managed_identity.referral_id = "7KQ9M2"
    settings.managed_identity.local_managed_claim_sources = ("discord",)
    settings.managed_identity.pending_delivery_ack_source = "discord"
    settings.managed_identity.pending_delivery_ack_delivery_id = "fixture-delivery-id"
    settings.managed_identity.pending_delivery_ack_managed_credential_ref = (
        "fixture-pending-credential-ref"
    )
    settings.managed_identity.pending_delivery_ack_expires_at = "2026-07-10T00:00:00Z"
    settings.telemetry.consent = "allow"
    settings.telemetry_state.anonymous_id = "fixture-telemetry-anonymous-id"
    settings.telemetry_state.sent_translation_success_dates_utc = ["2026-07-01", "2026-07-02"]
    settings.system_prompt = "Fixture system prompt text."
    settings.validate()

    data = to_dict(settings)
    data["audio"]["internal_sample_rate_hz"] = 8000
    data["audio"]["internal_channels"] = "1"
    data["local_llm"]["backend"] = "fixture_backend"
    data["openrouter"]["provider_routing"] = OpenRouterProviderRouting.DEEPSEEK_ONLY.value
    data["openrouter"][
        "fallback_selection_alias"
    ] = OpenRouterFallbackSelectionAlias.QWEN35_FLASH.value
    return data


def legacy_compatibility_settings_fixture() -> dict[str, Any]:
    data = copy.deepcopy(maximal_v24_settings_fixture())
    data["settings_version"] = 17
    data["openrouter"]["credential_source"] = OpenRouterCredentialSource.BYOK.value
    data["openrouter"]["selected_credential_source"] = OpenRouterCredentialSource.MANAGED.value
    data["overlay_calibration"] = {
        "offset_x": 0.42,
        "offset_y": -0.12,
        "distance": 1.6,
        "text_scale": 1.2,
        "background_alpha": 0.33,
    }
    data["overlay"].pop("calibration", None)
    data["overlay"].pop("show_translation", None)
    data["overlay"].pop("show_peer_original", None)
    data["overlay"]["desktop_flet"]["locked"] = True
    data["ui"]["show_overlay_translation"] = False
    data["ui"]["show_overlay_peer_original"] = False
    data["ui"]["overlay_enabled"] = True
    data["ui"]["peer_translation_enabled"] = True
    data["osc"]["cooldown_s"] = 1.5
    data["osc"]["ttl_s"] = 7.0
    data["peer_deepgram_stt"] = {"model": "legacy-peer-deepgram-model"}
    data["peer_qwen_asr_stt"] = {
        "model": "legacy-peer-qwen-model",
        "region": QwenRegion.SINGAPORE.value,
    }
    data["peer_soniox_stt"] = {
        "model": "legacy-peer-soniox-model",
        "endpoint": "wss://legacy-soniox.fixture.test/transcribe",
        "keepalive_interval_s": 13.0,
        "trailing_silence_ms": 275,
    }
    data["system_prompts"] = {"legacy": "legacy fixture prompt"}
    return data


def _put_classification(
    table: dict[str, FieldClassification],
    destinations_by_path: Mapping[str, str],
    *,
    category: str,
    status: str,
    fixture: str,
    notes: str = "",
) -> None:
    for path, destination in destinations_by_path.items():
        if path in table:
            raise ValueError(f"duplicate migration classification path: {path}")
        table[path] = FieldClassification(
            category=category,
            destination=destination,
            status=status,
            fixture=fixture,
            notes=notes,
        )


def _current_migration_classification() -> dict[str, FieldClassification]:
    table: dict[str, FieldClassification] = {}
    _put_classification(
        table,
        CURRENT_USER_INTENT_DESTINATIONS,
        category="persisted_user_intent",
        status="retained",
        fixture=MAXIMAL_V24_FIXTURE_NAME,
    )
    _put_classification(
        table,
        CURRENT_COMPATIBILITY_INPUT_DESTINATIONS,
        category="compatibility_input",
        status="accepted_read_no_vnext_write_projection",
        fixture=MAXIMAL_V24_FIXTURE_NAME,
    )
    _put_classification(
        table,
        CURRENT_OPERATIONAL_STATE_DESTINATIONS,
        category="persisted_operational_state",
        status="retained_or_reclassified",
        fixture=MAXIMAL_V24_FIXTURE_NAME,
    )
    _put_classification(
        table,
        REPAIR_TO_VNEXT_DEFAULT_DESTINATIONS,
        category="compatibility_repair",
        status="raw_fixture_non_default_repairs_to_canonical_default",
        fixture=MAXIMAL_V24_FIXTURE_NAME,
        notes="Maximal raw v24 fixture uses non-default input; current loader repairs it.",
    )
    _put_classification(
        table,
        REPAIR_TO_RUNTIME_CONSTANT_DESTINATIONS,
        category="compatibility_repair",
        status="raw_fixture_non_default_repairs_to_canonical_default_no_vnext_write_projection",
        fixture=MAXIMAL_V24_FIXTURE_NAME,
        notes="Maximal raw v24 fixture uses non-default input; current loader repairs it.",
    )
    table["settings_version"] = FieldClassification(
        category="schema_metadata",
        destination="settings_version",
        status="current_schema_version",
        fixture=MAXIMAL_V24_FIXTURE_NAME,
        notes="Schema metadata, not a user/state setting field; v24 fixture must remain version 24.",
    )
    table["overlay.calibration.anchor"] = FieldClassification(
        category="supported_enum_value",
        destination="intent.overlay.calibration.anchor",
        status="default_supported_value",
        fixture=MAXIMAL_V24_FIXTURE_NAME,
        notes="Current overlay calibration supports 'head_locked' and 'spatial_locked'; other values fail validation.",
    )
    for path, destination in ADR_RESOLVED_CURRENT_DESTINATIONS.items():
        if path in table:
            raise ValueError(f"duplicate migration classification path: {path}")
        category = (
            "persisted_user_intent"
            if path == "ui.integrated_context_enabled"
            else "persisted_operational_state"
        )
        table[path] = FieldClassification(
            category=category,
            destination=destination,
            status="retained_adr_resolved",
            fixture=MAXIMAL_V24_FIXTURE_NAME,
            notes="Accepted ADR 2026-06-08 fixed the vNext intent/state destination.",
        )
    return table


V24_MIGRATION_CLASSIFICATION = _current_migration_classification()

LEGACY_MIGRATION_CLASSIFICATION: dict[str, FieldClassification] = {
    "openrouter.credential_source": FieldClassification(
        category="legacy_input",
        destination="intent.translation.openrouter.selected_source",
        status="accepted_read_drives_migration_when_current_source_absent",
        fixture=LEGACY_COMPATIBILITY_FIXTURE_NAME,
    ),
    "openrouter.selected_credential_source": FieldClassification(
        category="legacy_input",
        destination="intent.translation.openrouter.selected_source",
        status="accepted_read_drives_migration_when_current_source_absent",
        fixture=LEGACY_COMPATIBILITY_FIXTURE_NAME,
    ),
    "osc.cooldown_s": FieldClassification(
        category="retired_input",
        destination="retired",
        status="removed_by_v18_migration",
        fixture=LEGACY_COMPATIBILITY_FIXTURE_NAME,
    ),
    "osc.ttl_s": FieldClassification(
        category="retired_input",
        destination="retired",
        status="removed_by_v18_migration",
        fixture=LEGACY_COMPATIBILITY_FIXTURE_NAME,
    ),
    "overlay.desktop_flet.locked": FieldClassification(
        category="retired_input",
        destination="retired",
        status="accepted_read_not_serialized",
        fixture=LEGACY_COMPATIBILITY_FIXTURE_NAME,
    ),
    "overlay_calibration.background_alpha": FieldClassification(
        category="legacy_input",
        destination="intent.overlay.calibration",
        status="merged_into_overlay_calibration_removed_on_write",
        fixture=LEGACY_COMPATIBILITY_FIXTURE_NAME,
    ),
    "overlay_calibration.distance": FieldClassification(
        category="legacy_input",
        destination="intent.overlay.calibration",
        status="merged_into_overlay_calibration_removed_on_write",
        fixture=LEGACY_COMPATIBILITY_FIXTURE_NAME,
    ),
    "overlay_calibration.offset_x": FieldClassification(
        category="legacy_input",
        destination="intent.overlay.calibration",
        status="merged_into_overlay_calibration_removed_on_write",
        fixture=LEGACY_COMPATIBILITY_FIXTURE_NAME,
    ),
    "overlay_calibration.offset_y": FieldClassification(
        category="legacy_input",
        destination="intent.overlay.calibration",
        status="merged_into_overlay_calibration_removed_on_write",
        fixture=LEGACY_COMPATIBILITY_FIXTURE_NAME,
    ),
    "overlay_calibration.text_scale": FieldClassification(
        category="legacy_input",
        destination="intent.overlay.calibration",
        status="merged_into_overlay_calibration_removed_on_write",
        fixture=LEGACY_COMPATIBILITY_FIXTURE_NAME,
    ),
    "peer_deepgram_stt.model": FieldClassification(
        category="retired_input",
        destination="retired",
        status="removed_by_migration",
        fixture=LEGACY_COMPATIBILITY_FIXTURE_NAME,
    ),
    "peer_qwen_asr_stt.model": FieldClassification(
        category="legacy_input",
        destination="intent.peer_stt.qwen_asr.compatibility",
        status="accepted_read_no_vnext_write_projection",
        fixture=LEGACY_COMPATIBILITY_FIXTURE_NAME,
    ),
    "peer_qwen_asr_stt.region": FieldClassification(
        category="legacy_input",
        destination="intent.peer_stt.qwen_asr.compatibility",
        status="accepted_read_no_vnext_write_projection",
        fixture=LEGACY_COMPATIBILITY_FIXTURE_NAME,
    ),
    "peer_soniox_stt.endpoint": FieldClassification(
        category="legacy_input",
        destination="intent.peer_stt.soniox.compatibility",
        status="accepted_read_no_vnext_write_projection",
        fixture=LEGACY_COMPATIBILITY_FIXTURE_NAME,
    ),
    "peer_soniox_stt.keepalive_interval_s": FieldClassification(
        category="legacy_input",
        destination="intent.peer_stt.soniox.compatibility",
        status="accepted_read_no_vnext_write_projection",
        fixture=LEGACY_COMPATIBILITY_FIXTURE_NAME,
    ),
    "peer_soniox_stt.model": FieldClassification(
        category="legacy_input",
        destination="intent.peer_stt.soniox.compatibility",
        status="accepted_read_no_vnext_write_projection",
        fixture=LEGACY_COMPATIBILITY_FIXTURE_NAME,
    ),
    "peer_soniox_stt.trailing_silence_ms": FieldClassification(
        category="legacy_input",
        destination="intent.peer_stt.soniox.compatibility",
        status="accepted_read_no_vnext_write_projection",
        fixture=LEGACY_COMPATIBILITY_FIXTURE_NAME,
    ),
    "system_prompts.legacy": FieldClassification(
        category="legacy_input",
        destination="intent.prompts.compatibility",
        status="accepted_read_removed_on_write",
        fixture=LEGACY_COMPATIBILITY_FIXTURE_NAME,
    ),
    "ui.overlay_enabled": FieldClassification(
        category="runtime_only_reclassification",
        destination="runtime_controller_state",
        status="dropped_from_persisted_settings",
        fixture=LEGACY_COMPATIBILITY_FIXTURE_NAME,
    ),
    "ui.peer_translation_enabled": FieldClassification(
        category="runtime_only_reclassification",
        destination="runtime_controller_state",
        status="dropped_from_persisted_settings",
        fixture=LEGACY_COMPATIBILITY_FIXTURE_NAME,
    ),
    "ui.show_overlay_peer_original": FieldClassification(
        category="legacy_input",
        destination="intent.overlay.show_peer_original",
        status="merged_into_overlay_removed_on_write",
        fixture=LEGACY_COMPATIBILITY_FIXTURE_NAME,
    ),
    "ui.show_overlay_translation": FieldClassification(
        category="legacy_input",
        destination="intent.overlay.show_translation",
        status="merged_into_overlay_removed_on_write",
        fixture=LEGACY_COMPATIBILITY_FIXTURE_NAME,
    ),
}


def missing_classification_paths(
    paths: set[str], classification: dict[str, FieldClassification]
) -> list[str]:
    return sorted(paths.difference(classification))


def migrated_serialization(data: dict[str, Any]) -> dict[str, Any]:
    settings = from_dict(data)
    return to_dict(settings)
