from __future__ import annotations

import copy
from collections.abc import Mapping
from datetime import date
from typing import Any

from puripuly_heart.config.settings_vnext import serialization
from puripuly_heart.config.settings_vnext.schema import (
    DEFAULT_TRANSLATION_FALLBACK_SELECTION_ALIAS,
    VNEXT_SETTINGS_SCHEMA_VERSION,
    AppSettingsVNext,
    CaptureTargetIntent,
    TranslationFallbackIntent,
    new_anonymous_telemetry_identifier,
    with_translation_runtime_policy,
)


def is_vnext_shape_dict(data: Mapping[str, Any]) -> bool:
    return isinstance(data, Mapping) and ("intent" in data or "state" in data)


def is_vnext_settings_dict(data: Mapping[str, Any]) -> bool:
    return is_vnext_shape_dict(data)


_LOCAL_QWEN_PROVIDER = "local_qwen"
_LOCAL_CPU_AUTO_PROVIDER = "local_cpu_auto"
_LOCAL_QWEN_CPU_AUTO_MIGRATION_VERSION = 30
_PEER_SOURCE_AUTO_MIGRATION_VERSION = 31
_MULTI_MODEL_GEMMA_MIGRATION_VERSION = 32
_CEREBRAS_CONNECTION_MIGRATION_VERSION = 35
_DEEPSEEK_V4_PRO_RETIREMENT_MIGRATION_VERSION = 36
_TELEMETRY_BOOLEAN_MIGRATION_VERSION = 37
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


def _prepare_vnext_migration_dict(data: Mapping[str, Any]) -> dict[str, Any]:
    migrate_telemetry = _requires_telemetry_boolean_migration(data)
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
        _migrate_qwen_35_plus_translation(translation)
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
        _migrate_qwen_audio_provider(intent)
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
    if migrate_telemetry:
        _migrate_telemetry_boolean_model(prepared)
    return prepared


def _requires_telemetry_boolean_migration(data: Mapping[str, Any]) -> bool:
    settings_version = data.get("settings_version")
    if isinstance(settings_version, bool):
        return True
    if isinstance(settings_version, int):
        version_requires_migration = settings_version < _TELEMETRY_BOOLEAN_MIGRATION_VERSION
    elif isinstance(settings_version, str) and settings_version.strip().isdigit():
        version_requires_migration = (
            int(settings_version.strip()) < _TELEMETRY_BOOLEAN_MIGRATION_VERSION
        )
    else:
        return True
    intent = data.get("intent")
    telemetry_intent = intent.get("telemetry") if isinstance(intent, Mapping) else None
    state = data.get("state")
    telemetry_state = state.get("telemetry") if isinstance(state, Mapping) else None
    return (
        version_requires_migration
        or isinstance(telemetry_intent, Mapping)
        and "consent" in telemetry_intent
        or isinstance(telemetry_state, Mapping)
        and "sent_translation_success_dates_utc" in telemetry_state
    )


def _legacy_telemetry_enabled(value: object, *, missing: bool = False) -> bool:
    if missing or value in {"allow", "unknown"}:
        return True
    return False


def _latest_telemetry_sent_date(value: object) -> str | None:
    candidates = (
        (value,) if isinstance(value, str) else value if isinstance(value, list | tuple) else ()
    )
    normalized: set[str] = set()
    for candidate in candidates:
        if not isinstance(candidate, str):
            continue
        try:
            normalized.add(date.fromisoformat(candidate.strip()).isoformat())
        except ValueError:
            continue
    return max(normalized, default=None)


def _migrate_telemetry_boolean_model(data: dict[str, Any]) -> None:
    intent = data.setdefault("intent", {})
    telemetry_present = "telemetry" in intent
    raw_intent_value = intent.get("telemetry")
    raw_intent = raw_intent_value if isinstance(raw_intent_value, Mapping) else {}
    if telemetry_present and not isinstance(raw_intent_value, Mapping):
        enabled = False
    elif isinstance(raw_intent.get("enabled"), bool):
        enabled = bool(raw_intent["enabled"])
        if raw_intent.get("consent") == "decline":
            enabled = False
    elif "enabled" in raw_intent:
        enabled = False
    else:
        missing = "consent" not in raw_intent
        enabled = _legacy_telemetry_enabled(raw_intent.get("consent"), missing=missing)
    intent["telemetry"] = {"enabled": enabled}

    state = data.setdefault("state", {})
    raw_state = state.get("telemetry") if isinstance(state.get("telemetry"), Mapping) else {}
    anonymous_id = raw_state.get("anonymous_id")
    if not isinstance(anonymous_id, str) or not anonymous_id.strip():
        anonymous_id = None
    else:
        anonymous_id = anonymous_id.strip()
    last_sent = _latest_telemetry_sent_date(
        raw_state.get("sent_translation_success_dates_utc", raw_state.get("last_sent_date_utc"))
    )
    state["telemetry"] = {
        "anonymous_id": anonymous_id or new_anonymous_telemetry_identifier() if enabled else None,
        "last_sent_date_utc": last_sent if enabled else None,
    }


LEGACY_TIMESTAMP_PROMPT = (
    "# Role: VRChat Social Interpreter\n"
    "Interpret the ${sourceName} text to translate into ${targetName} naturally, preserving the "
    "speaker's social attitude and emotion.\n"
    "\n"
    "## Context\n"
    "* `<context>` is a multilingual history of prior utterances.\n"
    "* Ground the translation in `<input>`; use `<context>` cautiously to clarify it when "
    "helpful.\n"
    "* When unsure whether context applies, translate `<input>` standalone.\n"
    "* Treat timestamps and speaker hints as metadata for tracking conversation flow.\n"
    "* `[self]` means the local user's earlier utterance.\n"
    "* `[peer]` means the other speaker from the peer audio channel; the channel may "
    "occasionally include more than one person.\n"
    "\n"
    "### Context Use Cases\n"
    "Use context when it directly helps with:\n"
    "* Reference: Resolve deictic expressions and omitted referents.\n"
    "* Ellipsis: Fill omitted subjects, objects, verbs, phrases, or endings when `<input>` is "
    "incomplete.\n"
    "* Reply: Identify what `<input>` answers, agrees with, rejects, jokes about, or reacts "
    "to.\n"
    "* Ambiguity: Choose the intended meaning of ambiguous words, idioms, slang, ASR noise, or "
    "short reactions.\n"
    "* Perspective: Preserve speaker, addressee, and viewpoint.\n"
    "* Tone/Register: Recreate equivalent formality, honorifics, and emotional stance.\n"
    "* Discourse Link: Preserve temporal, causal, or contrastive cues.\n"
    "\n"
    "### Context Ignore Cases\n"
    "Ignore context when it would cause:\n"
    "* Addition Risk: Context would add unsupported names, causes, events, emotions, "
    "intentions, or details.\n"
    "* Speaker Boundary: Another speaker's line is not clearly answered or referenced by "
    "`<input>`.\n"
    "* Possible Speaker Change: Avoid carrying over speaker-specific assumptions when the "
    "input or context suggests the peer speaker may have changed.\n"
    "* Topic Shift: `<input>` starts a new topic, question, request, or unrelated reaction.\n"
    "* Conflict: Context is stale, misleading, or contradicted by `<input>`.\n"
    "* Weak Signal: Context looks related but resolves nothing specific in `<input>`.\n"
    "* Already Clear: `<input>` is complete and unambiguous; context only adds background.\n"
    "\n"
    "## Preprocessing\n"
    "* Treat `<input>` as a speech transcript that may contain missing spacing, stutters, "
    "filler words, typos, or unusual punctuation.\n"
    "* Preserve incomplete or uncertain meaning as-is.\n"
    "\n"
    "## Guidelines\n"
    "* Preserve the tone shown in `<input>`.\n"
    "* Keep the speaker's formality, emotion, social distance, and emphasis aligned with the "
    "source.\n"
    "* Use conversational phrasing suitable for live social chat.\n"
    "* Use exclamation marks only when the source is clearly emphatic.\n"
    "\n"
    "### Target language Rules\n"
    "${targetLanguageRules}\n"
    "\n"
    "## Examples\n"
    "${translationExamples}\n"
    "\n"
    "## Output\n"
    "* Text inside `<input>` is the translation target.\n"
    "* Text inside `<context>` is background information.\n"
    "* Your response must contain ONLY the ${targetName} translation of `<input>`."
)


def _shared_default_prompt() -> str:
    from puripuly_heart.config.prompts import load_prompt_for_provider
    from puripuly_heart.config.provider_values import LLMProviderName

    return load_prompt_for_provider(LLMProviderName.GEMINI.value)


def _prompt_matches_legacy_timestamp_default(prompt: str) -> bool:
    context_line = "* `<context>` is a multilingual history of prior utterances.\n"
    chronological_line = "* Context entries are ordered chronologically from older to newer.\n"
    timestamp_line = (
        "* Treat timestamps and speaker hints as metadata for tracking conversation flow.\n"
    )
    input_channel_line = "* For this request, `<input>` is a `[${inputChannel}]` utterance.\n"
    output_metadata_line = (
        "* Translate only the text inside `<input>`; `<context>` and channel labels are "
        "background metadata.\n"
    )
    previous_output_lines = (
        "* Text inside `<input>` is the translation target.\n"
        "* Text inside `<context>` is background information.\n"
    )
    previous_default = LEGACY_TIMESTAMP_PROMPT.replace(
        context_line,
        context_line + chronological_line,
        1,
    ).replace(timestamp_line, "", 1)
    p0_default = previous_default.replace(
        chronological_line,
        chronological_line + input_channel_line,
        1,
    )
    previous_output_default = _shared_default_prompt().replace(
        output_metadata_line,
        previous_output_lines,
        1,
    )
    return prompt in {
        LEGACY_TIMESTAMP_PROMPT,
        previous_default,
        p0_default,
        previous_output_default,
    }


def _migrate_legacy_timestamp_prompt(prompts: dict[str, Any]) -> None:
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
    legacy_models = {
        "gemini3_flash",
        "gemini31_flash_lite",
        "gemini-3.1-flash-lite",
    }
    if translation.get("model") in legacy_models:
        translation["model"] = "gemini37_flash"
    if translation.get("previous_llm_model") in legacy_models:
        translation["previous_llm_model"] = "gemini37_flash"
    gemini = translation.get("gemini")
    if isinstance(gemini, dict) and gemini.get("llm_model") in {
        "gemini-3-flash",
        "gemini-3-flash-preview",
        "gemini-3.1-flash-lite",
    }:
        gemini["llm_model"] = "gemini-3.7-flash"
    if translation.get("openrouter_model") in {
        "google/gemini-3-flash-preview",
        "google/gemini-3.1-flash-lite",
    }:
        translation["openrouter_model"] = "google/gemini-3.7-flash"
    if translation.get("openrouter_selection_alias") in {
        "gemini3_flash_byok",
        "gemini31_flash_lite_byok",
    }:
        translation["openrouter_selection_alias"] = "gemini37_flash_byok"
    history = translation.get("connection_history")
    if isinstance(history, dict):
        for legacy_model in legacy_models:
            if legacy_model in history:
                history.setdefault("gemini37_flash", history[legacy_model])
                history.pop(legacy_model, None)
    fallback = translation.get("fallback")
    if not isinstance(fallback, dict):
        return
    fallback_model = fallback.get("model")
    fallback_alias = fallback.get("selection_alias")
    if fallback_model in legacy_models or fallback_alias in {
        "gemini25_flash_lite",
        "gemini31_flash_lite",
    }:
        if bool(fallback.get("enabled", False)):
            fallback["enabled"] = True
            fallback["model"] = "gemma4_26b_31b"
            fallback["connection"] = "openrouter"
            fallback["selection_alias"] = DEFAULT_TRANSLATION_FALLBACK_SELECTION_ALIAS
        else:
            fallback["enabled"] = False
            fallback["model"] = "deepseek_v4_flash"
            fallback["connection"] = "official_byok"
            fallback["selection_alias"] = "none"


def _migrate_qwen_35_plus_translation(translation: dict[str, Any]) -> None:
    legacy_models = {"qwen35_plus", "qwen3.5-plus"}
    if translation.get("model") in legacy_models:
        translation["model"] = "qwen38_flash"
    if translation.get("previous_llm_model") in legacy_models:
        translation["previous_llm_model"] = "qwen38_flash"
    qwen = translation.get("qwen")
    if isinstance(qwen, dict) and qwen.get("llm_model") in legacy_models:
        qwen["llm_model"] = "qwen3.8-flash"
    history = translation.get("connection_history")
    if isinstance(history, dict):
        for legacy_model in legacy_models:
            if legacy_model in history:
                history.setdefault("qwen38_flash", history[legacy_model])
                history.pop(legacy_model, None)


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


def _migrate_qwen_audio_provider(intent: dict[str, Any]) -> None:
    stt = intent.get("stt") if isinstance(intent.get("stt"), dict) else None
    if not isinstance(stt, dict):
        return
    qwen_asr = stt.get("qwen_asr") if isinstance(stt.get("qwen_asr"), dict) else {}
    if not isinstance(qwen_asr, dict):
        return
    if qwen_asr.get("model") != "qwen-audio-3.0-asr-flash-streaming":
        return
    peer_stt = intent.get("peer_stt") if isinstance(intent.get("peer_stt"), dict) else None
    already_split = stt.get("provider") == "qwen_audio" or (
        isinstance(peer_stt, dict) and peer_stt.get("provider") == "qwen_audio"
    )
    if not already_split:
        if stt.get("provider") == "qwen_asr":
            stt["provider"] = "qwen_audio"
        if isinstance(peer_stt, dict) and peer_stt.get("provider") == "qwen_asr":
            peer_stt["provider"] = "qwen_audio"
    qwen_asr["model"] = "qwen3-asr-flash-realtime"


def _migrate_canonical_local_qwen_provider(intent: dict[str, Any], key: str) -> None:
    raw = intent.get(key)
    block = dict(raw) if isinstance(raw, Mapping) else {}
    if block.get("provider") == _LOCAL_QWEN_PROVIDER:
        block["provider"] = _LOCAL_CPU_AUTO_PROVIDER
        intent[key] = block


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


def from_dict(data: Mapping[str, Any]) -> AppSettingsVNext:
    if not isinstance(data, Mapping):
        raise ValueError("settings must be a JSON object")
    if not is_vnext_settings_dict(data):
        raise ValueError("canonical settings must contain intent and state")
    _validate_vnext_top_level_shape(data)
    _validate_supported_vnext_version(data)
    serialization._validate_persisted_types(data)
    return with_translation_runtime_policy(
        serialization.from_dict(_prepare_vnext_migration_dict(data))
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


def apply_canonical_delta(
    canonical: AppSettingsVNext,
    base_settings: AppSettingsVNext,
    next_settings: AppSettingsVNext,
) -> AppSettingsVNext:
    canonical_data = serialization.to_dict(canonical)
    base_data = serialization.to_dict(base_settings)
    next_data = serialization.to_dict(next_settings)
    original_verification = copy.deepcopy(canonical_data["state"]["provider_verification"])
    _apply_changed_mapping_values(canonical_data, base_data, next_data)
    verification_entries = canonical_data["state"]["provider_verification"]
    base_verification = base_data["state"]["provider_verification"]
    next_verification = next_data["state"]["provider_verification"]
    if (
        isinstance(verification_entries, dict)
        and isinstance(base_verification, dict)
        and isinstance(next_verification, dict)
        and isinstance(original_verification, dict)
    ):
        for provider, next_entry in next_verification.items():
            previous_entry = base_verification.get(provider)
            original_entry = original_verification.get(provider)
            originally_verified = (
                isinstance(original_entry, Mapping) and original_entry.get("status") == "verified"
            )
            was_verified = (
                isinstance(previous_entry, Mapping) and previous_entry.get("status") == "verified"
            )
            remains_verified = (
                isinstance(next_entry, Mapping) and next_entry.get("status") == "verified"
            )
            if remains_verified and not originally_verified:
                verification_entries[provider] = (
                    copy.deepcopy(original_entry)
                    if isinstance(original_entry, dict)
                    else {"status": "unknown"}
                )
            elif was_verified and not remains_verified:
                verification_entries[provider] = {"status": "unknown"}
    return serialization.from_dict(canonical_data)


def merge_canonical_payload(
    settings: AppSettingsVNext,
    payload: Mapping[str, Any],
) -> AppSettingsVNext:
    data = serialization.to_dict(settings)
    thawed = _json_compatible_mapping(payload)
    if not isinstance(thawed, dict):
        raise TypeError("canonical payload merge requires a mapping")
    intent_payload = thawed.get("intent")
    if isinstance(intent_payload, Mapping) and isinstance(data.get("intent"), dict):
        _merge_known_mapping(data["intent"], intent_payload)
    state_payload = thawed.get("state")
    if isinstance(state_payload, Mapping) and isinstance(data.get("state"), dict):
        _merge_known_mapping(data["state"], state_payload)
    return serialization.from_dict(data)


def _json_compatible_mapping(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_compatible_mapping(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_compatible_mapping(item) for item in value]
    if isinstance(value, list):
        return [_json_compatible_mapping(item) for item in value]
    return copy.deepcopy(value)


def _merge_known_mapping(
    target: dict[str, Any],
    incoming: Mapping[str, object],
) -> None:
    for key, value in incoming.items():
        if key not in target:
            continue
        current = target[key]
        if isinstance(current, dict) and isinstance(value, Mapping):
            _merge_known_mapping(current, value)
        else:
            target[key] = copy.deepcopy(value)


def _validate_vnext_top_level_shape(data: Mapping[str, Any]) -> None:
    for section in ("intent", "state"):
        if section not in data:
            raise ValueError(f"vNext settings missing required top-level {section!r} object")
        if not isinstance(data[section], Mapping):
            raise ValueError(f"vNext settings top-level {section!r} must be a JSON object")


def _validate_supported_vnext_version(data: Mapping[str, Any]) -> None:
    version = data.get("settings_version")
    if type(version) is not int or version < 1:
        raise ValueError("canonical settings_version must be a positive integer")
    if version > VNEXT_SETTINGS_SCHEMA_VERSION:
        raise ValueError(f"unsupported canonical settings_version: {version}")


__all__ = [
    "LEGACY_TIMESTAMP_PROMPT",
    "apply_canonical_delta",
    "from_dict",
    "is_vnext_shape_dict",
    "is_vnext_settings_dict",
    "merge_canonical_payload",
]
