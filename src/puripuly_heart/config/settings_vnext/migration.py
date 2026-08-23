from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from puripuly_heart.config.overlay_calibration import OverlayCalibration
from puripuly_heart.config.settings_vnext import serialization
from puripuly_heart.config.settings_vnext.schema import (
    DEFAULT_TRANSLATION_FALLBACK_SELECTION_ALIAS,
    VNEXT_SETTINGS_SCHEMA_VERSION,
    AppSettingsVNext,
    AudioIntent,
    CaptureTargetIntent,
    CerebrasTranslationIntent,
    ClipboardIntent,
    CustomSTTIntent,
    DeepgramSTTIntent,
    DeepSeekTranslationIntent,
    DesktopAudioIntent,
    DesktopFletOverlayIntent,
    DesktopFletOverlayPositionIntent,
    DesktopFletOverlayVisualIntent,
    GeminiTranslationIntent,
    GithubStarPromptState,
    IntegratedContextIntent,
    IntegratedContextState,
    LanguageIntent,
    LocalLLMIntent,
    ManagedConnectionState,
    OscIntent,
    OverlayIntent,
    PeerSTTIntent,
    PeerTranslationState,
    PersistedOperationalState,
    PromptIntent,
    ProviderVerificationEntry,
    ProviderVerificationState,
    QwenASRSTTIntent,
    QwenTranslationIntent,
    SecretsIntent,
    SonioxSTTIntent,
    STTIntent,
    TelemetryConsentIntent,
    TelemetryOperationalState,
    TranslationFallbackIntent,
    TranslationIntent,
    UiIntent,
    UserIntentSettings,
    is_safe_compatibility_extension_key,
    normalize_managed_claim_sources,
    with_telemetry_consent,
    with_translation_runtime_policy,
)


def is_vnext_shape_dict(data: Mapping[str, Any]) -> bool:
    return isinstance(data, Mapping) and ("intent" in data or "state" in data)


def is_legacy_shape_dict(data: Mapping[str, Any]) -> bool:
    return isinstance(data, Mapping) and not is_vnext_shape_dict(data)


def is_vnext_settings_dict(data: Mapping[str, Any]) -> bool:
    return is_vnext_shape_dict(data)


_PROVIDER_VERIFICATION_FIELDS = (
    "deepgram",
    "soniox",
    "google",
    "openrouter",
    "deepseek",
    "cerebras",
    "alibaba_beijing",
    "alibaba_singapore",
)
_PROVIDER_VERIFICATION_SECRET_KEYS = {
    "deepgram": "deepgram_api_key",
    "soniox": "soniox_api_key",
    "google": "google_api_key",
    "openrouter": "openrouter_api_key",
    "deepseek": "deepseek_api_key",
    "cerebras": "cerebras_api_key",
    "alibaba_beijing": "alibaba_api_key_beijing",
    "alibaba_singapore": "alibaba_api_key_singapore",
}
_LEGACY_VERIFICATION_REVISION = "legacy-dev-settings"
_LEGACY_VERIFICATION_CONTEXT = {"flow": "legacy_settings_migration"}
_LOCAL_QWEN_PROVIDER = "local_qwen"
_LOCAL_CPU_AUTO_PROVIDER = "local_cpu_auto"
_LOCAL_QWEN_CPU_AUTO_MIGRATION_VERSION = 30
_PEER_SOURCE_AUTO_MIGRATION_VERSION = 31
_MULTI_MODEL_GEMMA_MIGRATION_VERSION = 32
_CEREBRAS_CONNECTION_MIGRATION_VERSION = 35
_DEEPSEEK_V4_PRO_RETIREMENT_MIGRATION_VERSION = 36
_EXPLICIT_LEGACY_GEMMA_FALLBACK_ALIASES = frozenset({"openrouter_gemma4_26b_a4b"})

_TEMPORARY_GENERIC_FALLBACK_ALIASES: dict[str, TranslationFallbackIntent] = {
    "none": TranslationFallbackIntent(enabled=False),
    "deepseek_v4_flash_official": TranslationFallbackIntent(
        enabled=True,
        model="deepseek_v4_flash",
        connection="official_byok",
        selection_alias="deepseek_v4_flash_official",
    ),
    "openrouter_deepseek_v4_flash": TranslationFallbackIntent(
        enabled=True,
        model="deepseek_v4_flash",
        connection="openrouter",
        selection_alias="openrouter_deepseek_v4_flash",
    ),
    "openrouter_gemma4_26b_a4b": TranslationFallbackIntent(
        enabled=True,
        model="gemma4",
        connection="openrouter",
        selection_alias="openrouter_gemma4_26b_a4b",
    ),
    DEFAULT_TRANSLATION_FALLBACK_SELECTION_ALIAS: TranslationFallbackIntent(
        enabled=True,
        model="gemma4_26b_31b",
        connection="openrouter",
        selection_alias="openrouter_gemma4_26b_31b",
    ),
    "openrouter_gemma4_31b": TranslationFallbackIntent(
        enabled=True,
        model="gemma4_31b",
        connection="openrouter",
        selection_alias="openrouter_gemma4_31b",
    ),
    "managed_gemma4_26b_31b": TranslationFallbackIntent(
        enabled=True,
        model="gemma4_26b_31b",
        connection="managed",
        selection_alias="managed_gemma4_26b_31b",
    ),
    "managed_gemma4_31b": TranslationFallbackIntent(
        enabled=True,
        model="gemma4_31b",
        connection="managed",
        selection_alias="managed_gemma4_31b",
    ),
    "cerebras_gemma4_31b": TranslationFallbackIntent(
        enabled=True,
        model="gemma4_31b",
        connection="cerebras",
        selection_alias="cerebras_gemma4_31b",
    ),
}
_FALLBACK_FIELDS_ALIAS: dict[tuple[bool, str, str], str] = {
    (False, "deepseek_v4_flash", "official_byok"): "none",
    (True, "deepseek_v4_flash", "official_byok"): "deepseek_v4_flash_official",
    (True, "deepseek_v4_flash", "openrouter"): "openrouter_deepseek_v4_flash",
    (True, "gemma4", "openrouter"): "openrouter_gemma4_26b_a4b",
    (True, "gemma4_26b_31b", "openrouter"): "openrouter_gemma4_26b_31b",
    (True, "gemma4_31b", "openrouter"): "openrouter_gemma4_31b",
    (True, "gemma4_26b_31b", "managed"): "managed_gemma4_26b_31b",
    (True, "gemma4_31b", "managed"): "managed_gemma4_31b",
    (True, "gemma4_31b", "cerebras"): "cerebras_gemma4_31b",
    (True, "gemma4_31b_cerebras", "official_byok"): "cerebras_gemma4_31b",
    (True, "deepseek_v4_flash", "managed_china"): "deepseek_v4_flash_china",
}
_LEGACY_OPEN_MAPPING_PATHS = frozenset(
    {
        ("translation", "connection_history"),
        ("stt", "custom_terms"),
        ("local_llm", "extra_body"),
        ("custom_stt", "extra"),
        ("system_prompts",),
    }
)
_LEGACY_RETIRED_COMPATIBILITY_PATHS = frozenset(
    {
        ("peer_qwen_asr_stt",),
        ("peer_soniox_stt",),
        ("provider", "peer_soniox_stt"),
    }
)


def _prepare_vnext_migration_dict(data: Mapping[str, Any]) -> dict[str, Any]:
    migrate_local_qwen = _requires_local_qwen_cpu_auto_migration(data.get("settings_version"))
    migrate_peer_source_auto = _requires_peer_source_auto_migration(data.get("settings_version"))
    migrate_multi_model_gemma = _requires_multi_model_gemma_migration(data.get("settings_version"))
    migrate_cerebras_connection = _requires_cerebras_connection_migration(
        data.get("settings_version")
    )
    migrate_deepseek_v4_pro_retirement = _requires_deepseek_v4_pro_retirement_migration(
        data.get("settings_version")
    )
    prepared = dict(copy.deepcopy(data))
    prepared["settings_version"] = VNEXT_SETTINGS_SCHEMA_VERSION
    intent = prepared.get("intent") if isinstance(prepared.get("intent"), dict) else {}
    translation = intent.get("translation") if isinstance(intent.get("translation"), dict) else {}
    if isinstance(intent, dict) and isinstance(translation, dict):
        if migrate_multi_model_gemma:
            _migrate_multi_model_gemma_translation(translation)
        if migrate_cerebras_connection:
            _migrate_cerebras_connection_translation(translation)
        if migrate_deepseek_v4_pro_retirement:
            _migrate_deepseek_v4_pro_translation(translation)
        _migrate_gemini_3_flash_translation(translation)
        fallback = translation.get("fallback")
        if not isinstance(fallback, Mapping):
            translation["fallback"] = _fallback_intent_to_dict(
                _fallback_intent_from_legacy_translation_data(
                    translation,
                    openrouter_data=None,
                )
            )
        translation.pop("fallback_selection_alias", None)
        translation.pop("openrouter_fallback_selection_alias", None)
        intent["translation"] = translation
        prepared["intent"] = intent
    if isinstance(intent, dict):
        osc = intent.get("osc") if isinstance(intent.get("osc"), Mapping) else None
        if isinstance(osc, dict) and "connection_mode" not in osc:
            osc["connection_mode"] = "automatic"
            osc.setdefault("send_port", osc.get("port", 9000))
            osc.setdefault("receive_port", 9001)
        if migrate_peer_source_auto:
            _migrate_peer_source_auto_mode(intent)
        if migrate_local_qwen:
            _migrate_canonical_local_qwen_provider(intent, "stt")
            _migrate_canonical_local_qwen_provider(intent, "peer_stt")
        desktop_audio = (
            dict(intent.get("desktop_audio", {}))
            if isinstance(intent.get("desktop_audio"), Mapping)
            else {}
        )
        if "capture_target" not in desktop_audio:
            desktop_audio["capture_target"] = _capture_target_to_dict(
                _capture_target_from_legacy_output_device(desktop_audio.get("output_device"))
            )
        if "output_device" not in desktop_audio:
            capture_target = desktop_audio.get("capture_target")
            desktop_audio["output_device"] = (
                capture_target.get("device_name", "")
                if isinstance(capture_target, Mapping)
                and capture_target.get("kind") == "named_output_device"
                else ""
            )
        intent["desktop_audio"] = desktop_audio
        prompts = intent.get("prompts") if isinstance(intent.get("prompts"), Mapping) else {}
        if isinstance(prompts, dict):
            _migrate_legacy_timestamp_prompt(prompts)
            intent["prompts"] = prompts
        prepared["intent"] = intent
    return prepared


def _migrate_legacy_timestamp_prompt(prompts: dict[str, Any]) -> None:
    from puripuly_heart.config.settings import (
        _prompt_matches_legacy_timestamp_default,
        _shared_default_prompt,
    )

    raw_system_prompt = prompts.get("system_prompt")
    if isinstance(raw_system_prompt, str) and _prompt_matches_legacy_timestamp_default(
        raw_system_prompt
    ):
        prompts["system_prompt"] = _shared_default_prompt()


def _requires_local_qwen_cpu_auto_migration(settings_version: object) -> bool:
    if isinstance(settings_version, bool):
        return True
    if isinstance(settings_version, int):
        return settings_version < _LOCAL_QWEN_CPU_AUTO_MIGRATION_VERSION
    if isinstance(settings_version, str) and settings_version.strip().isdigit():
        return int(settings_version.strip()) < _LOCAL_QWEN_CPU_AUTO_MIGRATION_VERSION
    return True


def _requires_peer_source_auto_migration(settings_version: object) -> bool:
    if isinstance(settings_version, bool):
        return True
    if isinstance(settings_version, int):
        return settings_version < _PEER_SOURCE_AUTO_MIGRATION_VERSION
    if isinstance(settings_version, str) and settings_version.strip().isdigit():
        return int(settings_version.strip()) < _PEER_SOURCE_AUTO_MIGRATION_VERSION
    return True


def _requires_multi_model_gemma_migration(settings_version: object) -> bool:
    if isinstance(settings_version, bool):
        return True
    if isinstance(settings_version, int):
        return settings_version < _MULTI_MODEL_GEMMA_MIGRATION_VERSION
    if isinstance(settings_version, str) and settings_version.strip().isdigit():
        return int(settings_version.strip()) < _MULTI_MODEL_GEMMA_MIGRATION_VERSION
    return True


def _requires_cerebras_connection_migration(settings_version: object) -> bool:
    if isinstance(settings_version, bool):
        return True
    if isinstance(settings_version, int):
        return settings_version < _CEREBRAS_CONNECTION_MIGRATION_VERSION
    if isinstance(settings_version, str) and settings_version.strip().isdigit():
        return int(settings_version.strip()) < _CEREBRAS_CONNECTION_MIGRATION_VERSION
    return True


def _requires_deepseek_v4_pro_retirement_migration(settings_version: object) -> bool:
    if isinstance(settings_version, bool):
        return True
    if isinstance(settings_version, int):
        return settings_version < _DEEPSEEK_V4_PRO_RETIREMENT_MIGRATION_VERSION
    if isinstance(settings_version, str) and settings_version.strip().isdigit():
        return int(settings_version.strip()) < _DEEPSEEK_V4_PRO_RETIREMENT_MIGRATION_VERSION
    return True


def _migrate_multi_model_gemma_translation(translation: dict[str, Any]) -> None:
    connection = translation.get("connection")
    if connection not in {"managed", "openrouter"}:
        connection = "managed"
    migrated_primary_gemma = translation.get("model") == "gemma4"
    if migrated_primary_gemma:
        translation["model"] = "gemma4_26b_31b"
        translation["openrouter_selection_alias"] = (
            "gemma4_26b_31b_managed" if connection == "managed" else "gemma4_26b_31b_byok"
        )
        translation["openrouter_provider_routing"] = "gemma4_26b_31b_latency"
    history = translation.get("connection_history")
    if isinstance(history, dict) and "gemma4" in history:
        history.setdefault("gemma4_26b_31b", history["gemma4"])

    fallback = translation.get("fallback")
    if not isinstance(fallback, dict):
        return
    if fallback.get("model") != "gemma4":
        return
    if fallback.get("selection_alias") in _EXPLICIT_LEGACY_GEMMA_FALLBACK_ALIASES:
        return
    fallback_connection = fallback.get("connection")
    if fallback_connection not in {"managed", "openrouter"}:
        fallback_connection = connection
    fallback["model"] = "gemma4_26b_31b"
    fallback["connection"] = fallback_connection
    fallback["selection_alias"] = (
        "managed_gemma4_26b_31b"
        if fallback_connection == "managed"
        else "openrouter_gemma4_26b_31b"
    )


def _migrate_cerebras_connection_translation(translation: dict[str, Any]) -> None:
    active_legacy_cerebras = translation.get("model") == "gemma4_31b_cerebras"
    previous_legacy_cerebras = translation.get("previous_llm_model") == "gemma4_31b_cerebras"
    if active_legacy_cerebras:
        translation["model"] = "gemma4_31b"
        translation["connection"] = "cerebras"
    if previous_legacy_cerebras:
        translation["previous_llm_model"] = "gemma4_31b"

    history = translation.get("connection_history")
    if isinstance(history, dict):
        legacy_history_present = "gemma4_31b_cerebras" in history
        history.pop("gemma4_31b_cerebras", None)
        if active_legacy_cerebras or previous_legacy_cerebras:
            history["gemma4_31b"] = "cerebras"
        elif "gemma4_31b" not in history and legacy_history_present:
            history["gemma4_31b"] = "cerebras"

    fallback = translation.get("fallback")
    if not isinstance(fallback, dict):
        return
    fallback_alias = fallback.get("selection_alias")
    if fallback_alias == "cerebras_gemma4_31b" or (
        fallback_alias is None and fallback.get("model") == "gemma4_31b_cerebras"
    ):
        fallback["enabled"] = True
        fallback["model"] = "gemma4_31b"
        fallback["connection"] = "cerebras"
        fallback["selection_alias"] = "cerebras_gemma4_31b"


def _migrate_gemini_3_flash_translation(translation: dict[str, Any]) -> None:
    if translation.get("model") == "gemini3_flash":
        translation["model"] = "gemini37_flash"
    if translation.get("previous_llm_model") == "gemini3_flash":
        translation["previous_llm_model"] = "gemini37_flash"
    gemini = translation.get("gemini")
    if isinstance(gemini, dict) and gemini.get("llm_model") in {
        "gemini-3-flash",
        "gemini-3-flash-preview",
    }:
        gemini["llm_model"] = "gemini-3.7-flash"
    if translation.get("openrouter_model") == "google/gemini-3-flash-preview":
        translation["openrouter_model"] = "google/gemini-3.7-flash"
    if translation.get("openrouter_selection_alias") == "gemini3_flash_byok":
        translation["openrouter_selection_alias"] = "gemini37_flash_byok"
    history = translation.get("connection_history")
    if isinstance(history, dict) and "gemini3_flash" in history:
        history["gemini37_flash"] = history["gemini3_flash"]
        history.pop("gemini3_flash", None)
    fallback = translation.get("fallback")
    if isinstance(fallback, dict) and fallback.get("model") == "gemini3_flash":
        if bool(fallback.get("enabled", False)):
            fallback["enabled"] = True
            fallback["model"] = "gemma4_26b_31b"
            fallback["connection"] = "openrouter"
            fallback["selection_alias"] = DEFAULT_TRANSLATION_FALLBACK_SELECTION_ALIAS
        else:
            fallback["model"] = "deepseek_v4_flash"
            fallback["connection"] = "official_byok"
            fallback["selection_alias"] = "none"


def _migrate_deepseek_v4_pro_translation(translation: dict[str, Any]) -> None:
    if translation.get("model") == "deepseek_v4_pro":
        translation["model"] = "deepseek_v4_flash"
        translation["connection"] = "official_byok"
    if translation.get("previous_llm_model") == "deepseek_v4_pro":
        translation["previous_llm_model"] = "deepseek_v4_flash"
    history = translation.get("connection_history")
    if isinstance(history, dict) and "deepseek_v4_pro" in history:
        history["deepseek_v4_flash"] = "official_byok"
        history.pop("deepseek_v4_pro", None)
    fallback = translation.get("fallback")
    if isinstance(fallback, dict) and fallback.get("model") == "deepseek_v4_pro":
        fallback["model"] = "deepseek_v4_flash"
        fallback["connection"] = "official_byok"


def _migrate_peer_source_auto_mode(intent: dict[str, Any]) -> None:
    raw = intent.get("languages")
    languages = dict(raw) if isinstance(raw, Mapping) else {}
    if languages.get("peer_source_mode") == "soniox_auto":
        languages["peer_source_mode"] = "auto"
        intent["languages"] = languages


def _migrate_canonical_local_qwen_provider(intent: dict[str, Any], key: str) -> None:
    raw = intent.get(key)
    block = dict(raw) if isinstance(raw, Mapping) else {}
    if block.get("provider") == _LOCAL_QWEN_PROVIDER:
        block["provider"] = _LOCAL_CPU_AUTO_PROVIDER
        intent[key] = block


def _migrate_legacy_local_qwen_providers(data: dict[str, Any]) -> None:
    provider = data.get("provider")
    if not isinstance(provider, dict):
        return
    for key in ("stt", "peer_stt"):
        if provider.get(key) == _LOCAL_QWEN_PROVIDER:
            provider[key] = _LOCAL_CPU_AUTO_PROVIDER


def _capture_target_from_legacy_output_device(value: object) -> CaptureTargetIntent:
    if isinstance(value, str) and value.strip():
        return CaptureTargetIntent.named_output_device(value)
    return CaptureTargetIntent.default_output_device()


def _capture_target_to_dict(target: CaptureTargetIntent) -> dict[str, object]:
    process = target.process
    return {
        "kind": target.kind,
        "device_name": target.device_name,
        "process": (
            None
            if process is None
            else {
                "kind": process.kind,
                "executable_identity": process.executable_identity,
                "discord_channel": process.discord_channel,
                "executable_basename": process.executable_basename,
            }
        ),
    }


def _fallback_intent_to_dict(intent: TranslationFallbackIntent) -> dict[str, object]:
    return {
        "enabled": intent.enabled,
        "model": intent.model,
        "connection": intent.connection,
        "selection_alias": intent.selection_alias,
    }


def _fallback_intent_from_temporary_alias(value: object) -> TranslationFallbackIntent | None:
    if not isinstance(value, str):
        return None
    alias = value.strip()
    if not alias:
        return TranslationFallbackIntent(
            selection_alias=DEFAULT_TRANSLATION_FALLBACK_SELECTION_ALIAS
        )
    return _TEMPORARY_GENERIC_FALLBACK_ALIASES.get(alias, TranslationFallbackIntent())


def _fallback_intent_from_legacy_openrouter_alias(
    value: object,
    *,
    selected_source: object,
) -> TranslationFallbackIntent:
    if value is None:
        return TranslationFallbackIntent(
            selection_alias=DEFAULT_TRANSLATION_FALLBACK_SELECTION_ALIAS
        )
    if not isinstance(value, str):
        return TranslationFallbackIntent()
    alias = value.strip()
    if not alias:
        return TranslationFallbackIntent(
            selection_alias=DEFAULT_TRANSLATION_FALLBACK_SELECTION_ALIAS
        )
    if alias in ("none", "qwen35_flash"):
        return TranslationFallbackIntent(enabled=False)
    if alias == "deepseek_v4_flash_china":
        return TranslationFallbackIntent(
            enabled=True,
            model="deepseek_v4_flash",
            connection="managed_china",
            selection_alias="deepseek_v4_flash_china",
        )
    if alias == "deepseek_v4_flash":
        if selected_source in {"managed", "byok"}:
            connection = "openrouter"
        else:
            return TranslationFallbackIntent(enabled=False)
        return TranslationFallbackIntent(
            enabled=True,
            model="deepseek_v4_flash",
            connection=connection,
            selection_alias="openrouter_deepseek_v4_flash",
        )
    return TranslationFallbackIntent(enabled=False)


def _fallback_intent_from_legacy_translation_data(
    translation_data: object,
    *,
    openrouter_data: object,
) -> TranslationFallbackIntent:
    translation = translation_data if isinstance(translation_data, Mapping) else {}
    fallback = translation.get("fallback")
    if isinstance(fallback, Mapping):
        if not fallback:
            return TranslationFallbackIntent(
                selection_alias=DEFAULT_TRANSLATION_FALLBACK_SELECTION_ALIAS
            )
        model = str(fallback.get("model", "deepseek_v4_flash"))
        connection = str(fallback.get("connection", "official_byok"))
        if (
            "selection_alias" not in fallback
            and not bool(fallback.get("enabled", False))
            and model == "deepseek_v4_flash"
            and connection == "official_byok"
        ):
            return TranslationFallbackIntent(
                selection_alias=DEFAULT_TRANSLATION_FALLBACK_SELECTION_ALIAS
            )
        selection_alias = str(
            fallback.get(
                "selection_alias",
                _FALLBACK_FIELDS_ALIAS.get(
                    (bool(fallback.get("enabled", False)), model, connection),
                    "none",
                ),
            )
        )
        return TranslationFallbackIntent(
            enabled=bool(fallback.get("enabled", False)),
            model=model,
            connection=connection,
            selection_alias=selection_alias,
        )
    temporary = _fallback_intent_from_temporary_alias(translation.get("fallback_selection_alias"))
    if temporary is not None:
        return temporary
    openrouter = openrouter_data if isinstance(openrouter_data, Mapping) else None
    return _fallback_intent_from_legacy_openrouter_alias(
        (
            openrouter.get("fallback_selection_alias")
            if isinstance(openrouter, Mapping)
            else translation.get("openrouter_fallback_selection_alias")
        ),
        selected_source=(
            openrouter.get("selected_source")
            if isinstance(openrouter, Mapping)
            else translation.get("openrouter_selected_source")
        ),
    )


def _fallback_intent_from_legacy_raw_dict(data: Mapping[str, Any]) -> TranslationFallbackIntent:
    translation_data = data.get("translation")
    openrouter_data = data.get("openrouter")
    return _fallback_intent_from_legacy_translation_data(
        translation_data,
        openrouter_data=openrouter_data,
    )


def _telemetry_consent_from_legacy_raw_dict(data: Mapping[str, Any]) -> str:
    telemetry = data.get("telemetry") if isinstance(data.get("telemetry"), Mapping) else {}
    return TelemetryConsentIntent(telemetry.get("consent", "unknown")).consent


def _telemetry_state_from_legacy_raw_dict(data: Mapping[str, Any]) -> TelemetryOperationalState:
    telemetry_state = (
        data.get("telemetry_state") if isinstance(data.get("telemetry_state"), Mapping) else {}
    )
    if not telemetry_state:
        telemetry = data.get("telemetry") if isinstance(data.get("telemetry"), Mapping) else {}
        telemetry_state = {
            "anonymous_id": telemetry.get("identifier"),
            "sent_translation_success_dates_utc": telemetry.get("sent_utc_dates", ()),
        }
    return TelemetryOperationalState(
        anonymous_id=telemetry_state.get("anonymous_id"),
        sent_translation_success_dates_utc=telemetry_state.get(
            "sent_translation_success_dates_utc", ()
        ),
    )


def from_dict(data: Mapping[str, Any]) -> AppSettingsVNext:
    """Read either canonical vNext settings or an accepted legacy settings dict."""

    if not isinstance(data, Mapping):
        raise ValueError("settings must be a JSON object")
    if is_vnext_settings_dict(data):
        _validate_vnext_top_level_shape(data)
        return with_translation_runtime_policy(
            serialization.from_dict(_prepare_vnext_migration_dict(data))
        )

    # Legacy compatibility belongs here: use the public legacy migration chain first, then
    # project the normalized AppSettings values into canonical vNext intent/state values.
    from puripuly_heart.config import settings as legacy_settings

    fallback_intent = _fallback_intent_from_legacy_raw_dict(data)
    telemetry_consent = _telemetry_consent_from_legacy_raw_dict(data)
    telemetry_state = _telemetry_state_from_legacy_raw_dict(data)
    prepared_legacy = dict(copy.deepcopy(data))
    _migrate_legacy_local_qwen_providers(prepared_legacy)
    managed_identity = prepared_legacy.get("managed_identity")
    if isinstance(managed_identity, dict):
        pending_delivery_ack_id = managed_identity.get("pending_delivery_ack_id")
        if (
            pending_delivery_ack_id is not None
            and "pending_delivery_ack_delivery_id" not in managed_identity
        ):
            managed_identity["pending_delivery_ack_delivery_id"] = pending_delivery_ack_id
    migrated, _changed = legacy_settings._migrate_settings_dict(prepared_legacy)
    settings = from_legacy_app_settings(
        legacy_settings.from_dict(migrated),
        fallback_intent=fallback_intent,
        preserve_provider_verification=True,
    )
    settings = replace(settings, state=replace(settings.state, telemetry=telemetry_state))
    legacy_template = legacy_settings.to_dict(legacy_settings.AppSettings())
    legacy_extensions = _extract_unknown_legacy_values(migrated, legacy_template, path=())
    if legacy_extensions:
        settings = replace(
            settings,
            compatibility_extensions={"legacy_compatibility": legacy_extensions},
        )
    from puripuly_heart.config.settings_vnext.schema import ensure_telemetry_default_allow

    prepared = serialization.to_dict(settings)
    prepared["settings_version"] = _MULTI_MODEL_GEMMA_MIGRATION_VERSION - 1
    migrated_settings = serialization.from_dict(_prepare_vnext_migration_dict(prepared))
    return ensure_telemetry_default_allow(
        with_telemetry_consent(migrated_settings, telemetry_consent)
    )


def _extract_unknown_legacy_values(
    raw: Mapping[str, Any],
    template: Mapping[str, Any],
    *,
    path: tuple[str, ...],
) -> dict[str, Any]:
    extensions: dict[str, Any] = {}
    for key, value in raw.items():
        child_path = (*path, key)
        if key == "settings_version":
            continue
        if child_path in _LEGACY_RETIRED_COMPATIBILITY_PATHS:
            continue
        if not is_safe_compatibility_extension_key(key):
            continue
        if key not in template:
            extensions[key] = copy.deepcopy(value)
            continue
        template_value = template[key]
        if isinstance(value, Mapping) and isinstance(template_value, Mapping):
            if child_path in _LEGACY_OPEN_MAPPING_PATHS:
                continue
            nested = _extract_unknown_legacy_values(value, template_value, path=child_path)
            if nested:
                extensions[key] = nested
    return extensions


def from_legacy_app_settings(
    settings: object,
    *,
    fallback_intent: TranslationFallbackIntent | None = None,
    preserve_provider_verification: bool = False,
) -> AppSettingsVNext:
    from puripuly_heart.config import settings as legacy_settings

    if not isinstance(settings, legacy_settings.AppSettings):
        raise TypeError("legacy settings migration requires AppSettings")

    data = legacy_settings.to_dict(settings)
    fallback = fallback_intent or _fallback_intent_from_legacy_translation_data(
        data.get("translation"),
        openrouter_data=data.get("openrouter"),
    )
    return with_translation_runtime_policy(
        AppSettingsVNext(
            settings_version=VNEXT_SETTINGS_SCHEMA_VERSION,
            intent=UserIntentSettings(
                translation=TranslationIntent(
                    model=data["translation"]["model"],
                    connection=data["translation"]["connection"],
                    http_extension_id=data["translation"].get("http_extension_id"),
                    previous_llm_model=data["translation"].get("previous_llm_model"),
                    connection_history=dict(data["translation"]["connection_history"]),
                    concurrency_limit=int(data["llm"]["concurrency_limit"]),
                    fallback=fallback,
                    openrouter_broker_base_url=data["openrouter"]["broker_base_url"],
                    openrouter_routing_mode=data["openrouter"]["routing_mode"],
                    openrouter_model=data["openrouter"]["llm_model"],
                    openrouter_selected_source=data["openrouter"]["selected_source"],
                    openrouter_selection_alias=data["openrouter"]["selection_alias"],
                    openrouter_provider_routing=data["openrouter"]["provider_routing"],
                    gemini=GeminiTranslationIntent(
                        llm_model=data["gemini"]["llm_model"],
                    ),
                    deepseek=DeepSeekTranslationIntent(
                        llm_model=data["deepseek"]["llm_model"],
                    ),
                    qwen=QwenTranslationIntent(
                        region=data["qwen"]["region"],
                        llm_model=data["qwen"]["llm_model"],
                    ),
                    cerebras=CerebrasTranslationIntent(
                        llm_model=data["cerebras"]["llm_model"],
                    ),
                    gpu_device_id=str(data["translation"].get("gpu_device_id", "auto")).strip()
                    or "auto",
                ),
                local_llm=LocalLLMIntent(
                    backend=data["local_llm"]["backend"],
                    base_url=data["local_llm"]["base_url"],
                    model=data["local_llm"]["model"],
                    extra_body=dict(data["local_llm"]["extra_body"]),
                ),
                stt=STTIntent(
                    provider=data["provider"]["stt"],
                    drain_timeout_s=float(data["stt"]["drain_timeout_s"]),
                    vad_speech_threshold=float(data["stt"]["vad_speech_threshold"]),
                    low_latency_mode=bool(data["stt"]["low_latency_mode"]),
                    low_latency_vad_hangover_ms=int(data["stt"]["low_latency_vad_hangover_ms"]),
                    low_latency_merge_gap_ms=int(data["stt"]["low_latency_merge_gap_ms"]),
                    low_latency_spec_retry_max=int(data["stt"]["low_latency_spec_retry_max"]),
                    custom_vocabulary_enabled=bool(data["stt"]["custom_vocabulary_enabled"]),
                    custom_terms=copy.deepcopy(data["stt"]["custom_terms"]),
                    gpu_device_id=data["stt"]["gpu_device_id"],
                    deepgram=DeepgramSTTIntent(model=data["deepgram_stt"]["model"]),
                    qwen_asr=QwenASRSTTIntent(model=data["qwen_asr_stt"]["model"]),
                    soniox=SonioxSTTIntent(
                        model=data["soniox_stt"]["model"],
                        endpoint=data["soniox_stt"]["endpoint"],
                        keepalive_interval_s=float(data["soniox_stt"]["keepalive_interval_s"]),
                        trailing_silence_ms=int(data["soniox_stt"]["trailing_silence_ms"]),
                    ),
                    custom=CustomSTTIntent(
                        mode=str(data.get("custom_stt", {}).get("mode", "offline")),
                        compatibility=str(
                            data.get("custom_stt", {}).get(
                                "compatibility",
                                "openai_transcription",
                            )
                        ),
                        endpoint=str(data.get("custom_stt", {}).get("endpoint", "")),
                        model=str(data.get("custom_stt", {}).get("model", "")),
                        extra=copy.deepcopy(data.get("custom_stt", {}).get("extra") or {}),
                    ),
                ),
                peer_stt=PeerSTTIntent(provider=data["provider"]["peer_stt"]),
                languages=LanguageIntent(
                    source_language=data["languages"]["source_language"],
                    target_language=data["languages"]["target_language"],
                    peer_source_language=data["languages"]["peer_source_language"],
                    peer_target_language=data["languages"]["peer_target_language"],
                    peer_source_mode=(
                        "auto"
                        if data["languages"].get("peer_source_mode") == "soniox_auto"
                        else data["languages"].get("peer_source_mode", "manual")
                    ),
                    peer_expected_languages=list(
                        data["languages"].get("peer_expected_languages") or []
                    ),
                    recent_source_languages=list(data["languages"]["recent_source_languages"]),
                    recent_target_languages=list(data["languages"]["recent_target_languages"]),
                ),
                audio=AudioIntent(
                    ring_buffer_ms=int(data["audio"]["ring_buffer_ms"]),
                    input_host_api=data["audio"]["input_host_api"],
                    input_device=data["audio"]["input_device"],
                ),
                desktop_audio=DesktopAudioIntent(
                    output_device=data["desktop_audio"]["output_device"],
                    capture_target=_capture_target_from_legacy_output_device(
                        data["desktop_audio"]["output_device"]
                    ),
                    vad_speech_threshold=float(data["desktop_audio"]["vad_speech_threshold"]),
                    vad_hangover_ms=int(data["desktop_audio"]["vad_hangover_ms"]),
                    vad_pre_roll_ms=int(data["desktop_audio"]["vad_pre_roll_ms"]),
                ),
                overlay=OverlayIntent(
                    target=data["overlay"]["target"],
                    show_translation=bool(data["overlay"]["show_translation"]),
                    show_peer_original=bool(data["overlay"]["show_peer_original"]),
                    calibration=OverlayCalibration(**data["overlay"]["calibration"]),
                    desktop_flet=DesktopFletOverlayIntent(
                        size_preset=data["overlay"]["desktop_flet"]["size_preset"],
                        position=DesktopFletOverlayPositionIntent(
                            x=data["overlay"]["desktop_flet"]["position"]["x"],
                            y=data["overlay"]["desktop_flet"]["position"]["y"],
                        ),
                        swap_caption_languages=(
                            data["overlay"]["desktop_flet"].get("swap_caption_languages") is True
                        ),
                        visual=DesktopFletOverlayVisualIntent(
                            background_alpha=float(
                                data["overlay"]["desktop_flet"]["visual"]["background_alpha"]
                            ),
                        ),
                    ),
                ),
                osc=OscIntent(
                    connection_mode=str(data["osc"].get("connection_mode", "automatic")),
                    host=data["osc"]["host"],
                    port=int(data["osc"]["port"]),
                    send_port=int(data["osc"].get("send_port", data["osc"]["port"])),
                    receive_port=int(data["osc"].get("receive_port", 9001)),
                    chatbox_address=data["osc"]["chatbox_address"],
                    chatbox_send=bool(data["osc"]["chatbox_send"]),
                    chatbox_clear=bool(data["osc"]["chatbox_clear"]),
                    chatbox_max_chars=int(data["osc"]["chatbox_max_chars"]),
                    vrc_mic_intercept=bool(data["osc"]["vrc_mic_intercept"]),
                    chatbox_include_source=bool(data["osc"]["chatbox_include_source"]),
                ),
                secrets=SecretsIntent(
                    backend=data["secrets"]["backend"],
                    encrypted_file_path=data["secrets"]["encrypted_file_path"],
                ),
                ui=UiIntent(locale=data["ui"]["locale"]),
                clipboard=ClipboardIntent(
                    auto_translate_enabled=bool(data["ui"]["clipboard_auto_translate_enabled"]),
                ),
                integrated_context=IntegratedContextIntent(
                    enabled=bool(data["ui"]["integrated_context_enabled"]),
                ),
                telemetry=TelemetryConsentIntent(
                    data.get("telemetry", {}).get("consent", "unknown")
                ),
                prompts=PromptIntent(system_prompt=data["system_prompt"]),
            ),
            state=PersistedOperationalState(
                provider_verification=_provider_verification_state(
                    data["api_key_verified"],
                    preserve_provider_verification=preserve_provider_verification,
                ),
                managed_connection=ManagedConnectionState(
                    installation_id=data["managed_identity"]["installation_id"],
                    release_token=data["managed_identity"]["release_token"],
                    release_token_expires_at=data["managed_identity"]["release_token_expires_at"],
                    verified_hardware_hash=data["managed_identity"]["verified_hardware_hash"],
                    verified_hardware_hash_salt_version=data["managed_identity"][
                        "verified_hardware_hash_salt_version"
                    ],
                    active_managed_credential_ref=data["managed_identity"][
                        "active_managed_credential_ref"
                    ],
                    active_managed_expires_at=data["managed_identity"]["active_managed_expires_at"],
                    founder_letter_seen_credential_ref=data["managed_identity"][
                        "founder_letter_seen_credential_ref"
                    ],
                    referral_id=data["managed_identity"]["referral_id"],
                    local_managed_claim_sources=normalize_managed_claim_sources(
                        data["managed_identity"].get("local_managed_claim_sources")
                    ),
                    pending_delivery_ack_source=data["managed_identity"].get(
                        "pending_delivery_ack_source"
                    ),
                    pending_delivery_ack_delivery_id=data["managed_identity"].get(
                        "pending_delivery_ack_delivery_id"
                    ),
                    pending_delivery_ack_managed_credential_ref=data["managed_identity"].get(
                        "pending_delivery_ack_managed_credential_ref"
                    ),
                    pending_delivery_ack_expires_at=data["managed_identity"].get(
                        "pending_delivery_ack_expires_at"
                    ),
                ),
                github_star_prompt=GithubStarPromptState(
                    clicked=bool(data["ui"]["github_star_prompt_clicked"]),
                    last_shown_at=data["ui"]["github_star_prompt_last_shown_at"],
                    show_count=int(data["ui"]["github_star_prompt_show_count"]),
                    translation_success_observed=bool(
                        data["ui"]["github_star_prompt_translation_success_observed"]
                    ),
                    eligible_launch_count=int(
                        data["ui"]["github_star_prompt_eligible_launch_count"]
                    ),
                ),
                peer_translation=PeerTranslationState(
                    eula_accepted=bool(data["ui"]["peer_translation_eula_accepted"]),
                ),
                integrated_context=IntegratedContextState(
                    bootstrapped=bool(data["ui"]["integrated_context_bootstrapped"]),
                ),
                telemetry=TelemetryOperationalState(
                    anonymous_id=data.get("telemetry_state", {}).get("anonymous_id"),
                    sent_translation_success_dates_utc=data.get("telemetry_state", {}).get(
                        "sent_translation_success_dates_utc"
                    ),
                ),
            ),
        )
    )


def _apply_changed_mapping_values(
    target: dict[str, Any],
    baseline: Mapping[str, object],
    next_values: Mapping[str, object],
) -> None:
    if "kind" in baseline and "kind" in next_values and baseline["kind"] != next_values["kind"]:
        target.clear()
        target.update(copy.deepcopy(dict(next_values)))
        return
    for key in baseline:
        if key not in next_values:
            target.pop(key, None)
    for key, next_value in next_values.items():
        previous_value = baseline.get(key)
        if isinstance(previous_value, Mapping) and isinstance(next_value, Mapping):
            target_value = target.get(key)
            if not isinstance(target_value, dict):
                target_value = {}
                target[key] = target_value
            _apply_changed_mapping_values(target_value, previous_value, next_value)
        elif previous_value != next_value:
            target[key] = copy.deepcopy(next_value)


def apply_legacy_app_settings_delta(
    canonical: AppSettingsVNext,
    base_settings: object,
    next_settings: object,
) -> AppSettingsVNext:
    canonical_fallback = canonical.intent.translation.fallback
    canonical_fallback_fields = (
        canonical_fallback.enabled,
        canonical_fallback.model,
        canonical_fallback.connection,
    )
    base_fallback_fields = _legacy_translation_fallback_fields(base_settings)
    base_fallback_intent = (
        canonical_fallback if base_fallback_fields == canonical_fallback_fields else None
    )
    converted_base = from_legacy_app_settings(
        base_settings,
        fallback_intent=base_fallback_intent,
    )
    converted_next = from_legacy_app_settings(
        next_settings,
        fallback_intent=(
            base_fallback_intent
            if _legacy_translation_fallback_fields(next_settings) == base_fallback_fields
            else None
        ),
    )
    canonical_data = serialization.to_dict(canonical)
    _apply_changed_mapping_values(
        canonical_data,
        serialization.to_dict(converted_base),
        serialization.to_dict(converted_next),
    )
    verification_entries = canonical_data["state"]["provider_verification"]
    base_verification = getattr(base_settings, "api_key_verified", None)
    next_verification = getattr(next_settings, "api_key_verified", None)
    for provider in verification_entries:
        was_verified = bool(getattr(base_verification, provider, False))
        remains_verified = bool(getattr(next_verification, provider, False))
        if was_verified and not remains_verified:
            verification_entries[provider] = {"status": "unknown"}
    return serialization.from_dict(canonical_data)


def _legacy_translation_fallback_fields(settings: object) -> tuple[bool, str, str] | None:
    translation = getattr(settings, "translation", None)
    fallback = getattr(translation, "fallback", None)
    if fallback is None:
        return None
    model = getattr(fallback, "model", None)
    connection = getattr(fallback, "connection", None)
    return (
        bool(getattr(fallback, "enabled", False)),
        str(getattr(model, "value", model)),
        str(getattr(connection, "value", connection)),
    )


def _validate_vnext_top_level_shape(data: Mapping[str, Any]) -> None:
    for section in ("intent", "state"):
        if section not in data:
            raise ValueError(f"vNext settings missing required top-level {section!r} object")
        if not isinstance(data[section], Mapping):
            raise ValueError(f"vNext settings top-level {section!r} must be a JSON object")


def _provider_verification_state(
    raw_verification: Mapping[str, Any],
    *,
    preserve_provider_verification: bool,
) -> ProviderVerificationState:
    entries: dict[str, ProviderVerificationEntry] = {}
    for provider in _PROVIDER_VERIFICATION_FIELDS:
        if preserve_provider_verification and raw_verification.get(provider) is True:
            entries[provider] = ProviderVerificationEntry(
                status="verified",
                provider=provider,
                secret_key=_PROVIDER_VERIFICATION_SECRET_KEYS[provider],
                secret_revision=_LEGACY_VERIFICATION_REVISION,
                verifier_context=_LEGACY_VERIFICATION_CONTEXT,
                verifier_evidence={"source": "legacy_boolean"},
            )
        else:
            entries[provider] = ProviderVerificationEntry(status="unknown")
    return ProviderVerificationState(**entries)


def to_legacy_dict(settings: AppSettingsVNext) -> dict[str, Any]:
    """Project canonical vNext values back to the temporary legacy facade shape.

    Runtime callers still consume AppSettings during this gate. This adapter keeps that public
    facade available while persistence writes the vNext intent/state schema.
    """

    from puripuly_heart.config import settings as legacy_settings

    intent = settings.intent
    state = settings.state
    data = legacy_settings.to_dict(legacy_settings.AppSettings())
    data["settings_version"] = legacy_settings.SETTINGS_SCHEMA_VERSION
    previous_model = intent.translation.previous_llm_model
    previous_connection = (
        intent.translation.connection_history.get(previous_model)
        if previous_model is not None
        else None
    )
    data["provider"]["llm"] = _legacy_provider_llm_for_translation(
        previous_model or intent.translation.model,
        previous_connection or intent.translation.connection,
    )
    data["provider"]["stt"] = intent.stt.provider
    data["provider"]["peer_stt"] = intent.peer_stt.provider
    data["translation"] = {
        "model": intent.translation.model,
        "connection": intent.translation.connection,
        "connection_history": dict(intent.translation.connection_history),
        "fallback": {
            "enabled": intent.translation.fallback.enabled,
            "model": intent.translation.fallback.model,
            "connection": intent.translation.fallback.connection,
        },
        "gpu_device_id": intent.translation.gpu_device_id,
    }
    if (
        intent.translation.model == "custom_http"
        or intent.translation.http_extension_id is not None
    ):
        data["translation"]["http_extension_id"] = intent.translation.http_extension_id
    if intent.translation.previous_llm_model is not None:
        data["translation"]["previous_llm_model"] = intent.translation.previous_llm_model
    data["languages"] = {
        "source_language": intent.languages.source_language,
        "target_language": intent.languages.target_language,
        "peer_source_language": intent.languages.peer_source_language,
        "peer_target_language": intent.languages.peer_target_language,
        "peer_source_mode": intent.languages.peer_source_mode,
        "peer_expected_languages": list(intent.languages.peer_expected_languages),
        "recent_source_languages": list(intent.languages.recent_source_languages),
        "recent_target_languages": list(intent.languages.recent_target_languages),
    }
    data["audio"].update(
        {
            "ring_buffer_ms": intent.audio.ring_buffer_ms,
            "input_host_api": intent.audio.input_host_api,
            "input_device": intent.audio.input_device,
        }
    )
    data["desktop_audio"] = {
        "output_device": intent.desktop_audio.output_device,
        "vad_speech_threshold": intent.desktop_audio.vad_speech_threshold,
        "vad_hangover_ms": intent.desktop_audio.vad_hangover_ms,
        "vad_pre_roll_ms": intent.desktop_audio.vad_pre_roll_ms,
    }
    data["overlay"] = {
        "target": intent.overlay.target,
        "show_translation": intent.overlay.show_translation,
        "show_peer_original": intent.overlay.show_peer_original,
        "calibration": intent.overlay.calibration.to_dict(),
        "desktop_flet": {
            "size_preset": intent.overlay.desktop_flet.size_preset,
            "position": {
                "x": intent.overlay.desktop_flet.position.x,
                "y": intent.overlay.desktop_flet.position.y,
            },
            "swap_caption_languages": intent.overlay.desktop_flet.swap_caption_languages,
            "visual": {
                "background_alpha": intent.overlay.desktop_flet.visual.background_alpha,
            },
        },
    }
    data["stt"] = {
        "drain_timeout_s": intent.stt.drain_timeout_s,
        "vad_speech_threshold": intent.stt.vad_speech_threshold,
        "low_latency_mode": intent.stt.low_latency_mode,
        "low_latency_vad_hangover_ms": intent.stt.low_latency_vad_hangover_ms,
        "low_latency_merge_gap_ms": intent.stt.low_latency_merge_gap_ms,
        "low_latency_spec_retry_max": intent.stt.low_latency_spec_retry_max,
        "custom_vocabulary_enabled": intent.stt.custom_vocabulary_enabled,
        "custom_terms": copy.deepcopy(intent.stt.custom_terms),
        "gpu_device_id": intent.stt.gpu_device_id,
    }
    data["deepgram_stt"] = {"model": intent.stt.deepgram.model}
    data["qwen_asr_stt"]["model"] = intent.stt.qwen_asr.model
    data["soniox_stt"] = {
        "model": intent.stt.soniox.model,
        "endpoint": intent.stt.soniox.endpoint,
        "keepalive_interval_s": intent.stt.soniox.keepalive_interval_s,
        "trailing_silence_ms": intent.stt.soniox.trailing_silence_ms,
    }
    data["custom_stt"] = {
        "mode": intent.stt.custom.mode,
        "compatibility": intent.stt.custom.compatibility,
        "endpoint": intent.stt.custom.endpoint,
        "model": intent.stt.custom.model,
        "extra": copy.deepcopy(intent.stt.custom.extra),
    }
    data["openrouter"].update(
        {
            "routing_mode": intent.translation.openrouter_routing_mode,
            "llm_model": intent.translation.openrouter_model,
            "selected_source": intent.translation.openrouter_selected_source,
            "selection_alias": intent.translation.openrouter_selection_alias,
            "provider_routing": intent.translation.openrouter_provider_routing,
            "fallback_selection_alias": "none",
            "broker_base_url": intent.translation.openrouter_broker_base_url,
        }
    )
    data["gemini"] = {
        "llm_model": intent.translation.gemini.llm_model,
    }
    data["deepseek"] = {
        "llm_model": intent.translation.deepseek.llm_model,
    }
    data["qwen"] = {
        "region": intent.translation.qwen.region,
        "llm_model": intent.translation.qwen.llm_model,
    }
    data["local_llm"] = {
        "backend": intent.local_llm.backend,
        "base_url": intent.local_llm.base_url,
        "model": intent.local_llm.model,
        "extra_body": copy.deepcopy(intent.local_llm.extra_body),
    }
    data["cerebras"] = {
        "llm_model": intent.translation.cerebras.llm_model,
    }
    data["llm"] = {"concurrency_limit": intent.translation.concurrency_limit}
    data["osc"] = {
        "host": intent.osc.host,
        "port": intent.osc.port,
        "chatbox_address": intent.osc.chatbox_address,
        "chatbox_send": intent.osc.chatbox_send,
        "chatbox_clear": intent.osc.chatbox_clear,
        "chatbox_max_chars": intent.osc.chatbox_max_chars,
        "vrc_mic_intercept": intent.osc.vrc_mic_intercept,
        "chatbox_include_source": intent.osc.chatbox_include_source,
    }
    data["osc"].update(
        {
            "connection_mode": intent.osc.connection_mode,
            "send_port": intent.osc.send_port,
            "receive_port": intent.osc.receive_port,
        }
    )
    data["secrets"] = {
        "backend": intent.secrets.backend,
        "encrypted_file_path": intent.secrets.encrypted_file_path,
    }
    data["ui"] = {
        "locale": intent.ui.locale,
        "peer_translation_eula_accepted": state.peer_translation.eula_accepted,
        "integrated_context_enabled": intent.integrated_context.enabled,
        "integrated_context_bootstrapped": state.integrated_context.bootstrapped,
        "clipboard_auto_translate_enabled": intent.clipboard.auto_translate_enabled,
        "github_star_prompt_clicked": state.github_star_prompt.clicked,
        "github_star_prompt_last_shown_at": state.github_star_prompt.last_shown_at,
        "github_star_prompt_show_count": state.github_star_prompt.show_count,
        "github_star_prompt_translation_success_observed": (
            state.github_star_prompt.translation_success_observed
        ),
        "github_star_prompt_eligible_launch_count": (
            state.github_star_prompt.eligible_launch_count
        ),
    }
    data["telemetry"] = {"consent": intent.telemetry.consent}
    data["telemetry_state"] = {
        "anonymous_id": state.telemetry.anonymous_id,
        "sent_translation_success_dates_utc": list(
            state.telemetry.sent_translation_success_dates_utc
        ),
    }
    data["api_key_verified"] = {
        "deepgram": _is_evidence_bound_verified_entry(
            state.provider_verification.deepgram,
            provider="deepgram",
        ),
        "soniox": _is_evidence_bound_verified_entry(
            state.provider_verification.soniox,
            provider="soniox",
        ),
        "google": _is_evidence_bound_verified_entry(
            state.provider_verification.google,
            provider="google",
        ),
        "openrouter": _is_evidence_bound_verified_entry(
            state.provider_verification.openrouter,
            provider="openrouter",
        ),
        "deepseek": _is_evidence_bound_verified_entry(
            state.provider_verification.deepseek,
            provider="deepseek",
        ),
        "cerebras": _is_evidence_bound_verified_entry(
            state.provider_verification.cerebras,
            provider="cerebras",
        ),
        "alibaba_beijing": _is_evidence_bound_verified_entry(
            state.provider_verification.alibaba_beijing,
            provider="alibaba_beijing",
        ),
        "alibaba_singapore": _is_evidence_bound_verified_entry(
            state.provider_verification.alibaba_singapore,
            provider="alibaba_singapore",
        ),
    }
    data["managed_identity"] = {
        "installation_id": state.managed_connection.installation_id,
        "release_token": state.managed_connection.release_token,
        "release_token_expires_at": state.managed_connection.release_token_expires_at,
        "verified_hardware_hash": state.managed_connection.verified_hardware_hash,
        "verified_hardware_hash_salt_version": (
            state.managed_connection.verified_hardware_hash_salt_version
        ),
        "active_managed_credential_ref": state.managed_connection.active_managed_credential_ref,
        "active_managed_expires_at": state.managed_connection.active_managed_expires_at,
        "founder_letter_seen_credential_ref": (
            state.managed_connection.founder_letter_seen_credential_ref
        ),
        "referral_id": state.managed_connection.referral_id,
        "local_managed_claim_sources": list(
            normalize_managed_claim_sources(state.managed_connection.local_managed_claim_sources)
        ),
        "pending_delivery_ack_source": state.managed_connection.pending_delivery_ack_source,
        "pending_delivery_ack_delivery_id": (
            state.managed_connection.pending_delivery_ack_delivery_id
        ),
        "pending_delivery_ack_managed_credential_ref": (
            state.managed_connection.pending_delivery_ack_managed_credential_ref
        ),
        "pending_delivery_ack_expires_at": (
            state.managed_connection.pending_delivery_ack_expires_at
        ),
    }
    data["system_prompt"] = intent.prompts.system_prompt
    return data


def _legacy_provider_llm_for_translation(model: str, connection: str) -> str:
    if model in {"managed_gemma", "managed_gemma_12b"}:
        return "managed_gemma"
    if model == "local_llm":
        return "local_llm"
    if model == "gemma4_31b_cerebras" or (model == "gemma4_31b" and connection == "cerebras"):
        return "cerebras"
    if model in {"gemini37_flash", "gemini31_flash_lite"}:
        if connection == "openrouter":
            return "openrouter"
        return "gemini"
    if model in {"deepseek_v4_flash", "deepseek_v4_pro"} and connection == "official_byok":
        return "deepseek"
    if model == "qwen35_plus":
        return "qwen"
    return "openrouter"


def _is_evidence_bound_verified_entry(
    entry: ProviderVerificationEntry,
    *,
    provider: str,
) -> bool:
    return (
        entry.status == "verified"
        and entry.provider == provider
        and bool(entry.secret_key)
        and bool(entry.verifier_context)
        and bool(entry.secret_revision or entry.secret_fingerprint)
    )


__all__ = [
    "from_dict",
    "from_legacy_app_settings",
    "is_legacy_shape_dict",
    "is_vnext_shape_dict",
    "is_vnext_settings_dict",
    "to_legacy_dict",
]
