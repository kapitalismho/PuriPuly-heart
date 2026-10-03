from __future__ import annotations

import copy
import hashlib
from collections.abc import Mapping
from datetime import date
from typing import Any

from puripuly_heart.config.llm_profiles import normalize_legacy_openrouter_model
from puripuly_heart.config.settings_vnext import serialization
from puripuly_heart.config.settings_vnext.schema import (
    VNEXT_SETTINGS_SCHEMA_VERSION,
    AppSettingsVNext,
    CaptureTargetIntent,
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
_CEREBRAS_RETIREMENT_MIGRATION_VERSION = 42
_TELEMETRY_BOOLEAN_MIGRATION_VERSION = 37
_DEEPGRAM_ROLLING_MIGRATION_VERSION = 39
_PROMPT_OVERRIDE_MIGRATION_VERSION = 45
_DEEPSEEK_41_SAVED_CONNECTION_MIGRATION_VERSION = 41
_MANAGED_GEMMA_12B_RETIREMENT_MIGRATION_VERSION = 43
_TRANSLATION_FALLBACK_RETIREMENT_MIGRATION_VERSION = 44
_RELEASED_DEFAULT_PROMPT_SHA256 = frozenset(
    {
        "9badb2a6aa2dca63f57eef67bdd46d8ecbe171b0eff951d1fced92377df8ba6e",
        "a58bc860304d36629e701af1e278d8ac03091f96b748435f87daed1cc4fc8807",
        "e2666581bd2d453c5ffc503a82604594655b926c7fe5c9cb48108715398ae994",
        "5cd516c94c8ad7024d142fb869e39e67caec6ba7dd417de69111077df45bfc54",
    }
)


def _requires_cerebras_retirement_migration(settings_version: object) -> bool:
    if isinstance(settings_version, bool):
        return True
    if isinstance(settings_version, int):
        return settings_version < _CEREBRAS_RETIREMENT_MIGRATION_VERSION
    if isinstance(settings_version, str) and settings_version.strip().isdigit():
        return int(settings_version.strip()) < _CEREBRAS_RETIREMENT_MIGRATION_VERSION
    return True


def _requires_managed_gemma_12b_retirement_migration(settings_version: object) -> bool:
    if isinstance(settings_version, bool):
        return True
    if isinstance(settings_version, int):
        return settings_version < _MANAGED_GEMMA_12B_RETIREMENT_MIGRATION_VERSION
    if isinstance(settings_version, str) and settings_version.strip().isdigit():
        return int(settings_version.strip()) < _MANAGED_GEMMA_12B_RETIREMENT_MIGRATION_VERSION
    return True


def _requires_translation_fallback_retirement_migration(settings_version: object) -> bool:
    if isinstance(settings_version, bool):
        return True
    if isinstance(settings_version, int):
        return settings_version < _TRANSLATION_FALLBACK_RETIREMENT_MIGRATION_VERSION
    if isinstance(settings_version, str) and settings_version.strip().isdigit():
        return int(settings_version.strip()) < _TRANSLATION_FALLBACK_RETIREMENT_MIGRATION_VERSION
    return True


def _requires_translation_concurrency_migration(settings_version: object) -> bool:
    if isinstance(settings_version, bool):
        return True
    if isinstance(settings_version, int):
        return settings_version < 46
    if isinstance(settings_version, str) and settings_version.strip().isdigit():
        return int(settings_version.strip()) < 46
    return True


def _prepare_vnext_migration_dict(data: Mapping[str, Any]) -> dict[str, Any]:
    migrate_telemetry = _requires_telemetry_boolean_migration(data)
    migrate_local_qwen = _requires_local_qwen_cpu_auto_migration(data.get("settings_version"))
    migrate_peer_source_auto = _requires_peer_source_auto_migration(data.get("settings_version"))
    migrate_multi_model_gemma = _requires_multi_model_gemma_migration(data.get("settings_version"))
    migrate_cerebras_retirement = _requires_cerebras_retirement_migration(
        data.get("settings_version")
    )
    migrate_prompt_reset = _requires_prompt_override_migration(data.get("settings_version"))
    migrate_deepgram_rolling = _requires_deepgram_rolling_migration(data.get("settings_version"))
    migrate_deepseek_saved_connections = _requires_deepseek_41_saved_connection_migration(
        data.get("settings_version")
    )
    migrate_managed_gemma_12b_retirement = _requires_managed_gemma_12b_retirement_migration(
        data.get("settings_version")
    )
    migrate_translation_fallback_retirement = _requires_translation_fallback_retirement_migration(
        data.get("settings_version")
    )
    migrate_translation_concurrency = _requires_translation_concurrency_migration(
        data.get("settings_version")
    )
    prepared = dict(copy.deepcopy(data))
    prepared["settings_version"] = VNEXT_SETTINGS_SCHEMA_VERSION
    if migrate_prompt_reset:
        prepared.pop("system_prompt", None)
    intent = prepared.get("intent") if isinstance(prepared.get("intent"), dict) else {}
    translation = intent.get("translation") if isinstance(intent.get("translation"), dict) else {}
    if isinstance(intent, dict) and isinstance(translation, dict):
        retained_combined_history = (
            isinstance(translation.get("connection_history"), dict)
            and "gemma4_26b_31b" in translation["connection_history"]
        )
        if migrate_translation_concurrency and translation.get("concurrency_limit") == 5:
            translation["concurrency_limit"] = 10
        if migrate_multi_model_gemma:
            _migrate_multi_model_gemma_translation(translation)
        if migrate_cerebras_retirement:
            _migrate_retired_cerebras_translation(translation)
        if migrate_managed_gemma_12b_retirement:
            _migrate_retired_managed_gemma_12b_translation(translation)
        _migrate_gemini_3_flash_translation(translation)
        _migrate_qwen_35_plus_translation(translation)
        _migrate_legacy_openrouter_model_translation(translation)
        if migrate_translation_fallback_retirement:
            translation.pop("fallback", None)
            translation.pop("fallback_selection_alias", None)
            translation.pop("openrouter_fallback_selection_alias", None)
        _migrate_deepseek_translation(
            translation,
            migrate_saved_connections=migrate_deepseek_saved_connections,
        )
        _migrate_retired_deepseek_v4_managed_path(translation)
        _consolidate_cloud_gemma_translation(
            translation, retained_combined_history=retained_combined_history
        )
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
            if migrate_prompt_reset:
                prompts.pop("system_prompt", None)
                prompts["system_prompt_override"] = None
            intent["prompts"] = prompts
        prepared["intent"] = intent
    if migrate_telemetry:
        _migrate_telemetry_boolean_model(prepared)
    if migrate_deepgram_rolling:
        _migrate_deepgram_stt_to_rolling(prepared)
    _migrate_cloud_free_tier_providers(prepared)
    return prepared


def _migrate_cloud_free_tier_providers(data: dict[str, Any]) -> None:
    intent = data.get("intent")
    if not isinstance(intent, dict):
        return
    stt = intent.get("stt")
    if not isinstance(stt, dict):
        stt = {}
        intent["stt"] = stt
    if "cloud_free_tier_providers" in stt:
        return
    stt["cloud_free_tier_providers"] = ["gemini_transcribe"]


def _migrate_deepgram_stt_to_rolling(data: dict[str, Any]) -> None:
    intent = data.get("intent")
    if not isinstance(intent, dict):
        return
    stt = intent.get("stt") if isinstance(intent.get("stt"), dict) else {}
    peer_stt = intent.get("peer_stt") if isinstance(intent.get("peer_stt"), dict) else {}
    self_is_deepgram = stt.get("provider") == "deepgram"
    peer_is_deepgram = peer_stt.get("provider") == "deepgram"
    if not self_is_deepgram and not peer_is_deepgram:
        return
    if self_is_deepgram:
        stt["provider"] = "rolling_free"
        intent["stt"] = stt
    if peer_is_deepgram:
        peer_stt["provider"] = "rolling_free"
        intent["peer_stt"] = peer_stt
    if not isinstance(intent.get("stt"), dict):
        intent["stt"] = stt
    intent["stt"]["cloud_free_tier_providers"] = ["gemini_transcribe", "deepgram"]


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


def _stored_system_prompt(data: Mapping[str, Any]) -> str:
    intent = data.get("intent")
    prompts = intent.get("prompts") if isinstance(intent, Mapping) else None
    value = prompts.get("system_prompt") if isinstance(prompts, Mapping) else None
    if not isinstance(value, str):
        value = data.get("system_prompt")
    return value if isinstance(value, str) else ""


def _is_released_default_prompt(value: str) -> bool:
    if value == _shared_default_prompt():
        return True
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return digest in _RELEASED_DEFAULT_PROMPT_SHA256


def system_prompt_backup_text(data: Mapping[str, Any]) -> str | None:
    if is_vnext_shape_dict(data) and not _requires_prompt_override_migration(
        data.get("settings_version")
    ):
        return None
    previous = _stored_system_prompt(data)
    if not previous or _is_released_default_prompt(previous):
        return None
    return previous


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


def _requires_prompt_override_migration(settings_version: object) -> bool:
    if isinstance(settings_version, bool):
        return True
    if isinstance(settings_version, int):
        return settings_version < _PROMPT_OVERRIDE_MIGRATION_VERSION
    if isinstance(settings_version, str) and settings_version.strip().isdigit():
        return int(settings_version.strip()) < _PROMPT_OVERRIDE_MIGRATION_VERSION
    return True


def _requires_deepgram_rolling_migration(settings_version: object) -> bool:
    if isinstance(settings_version, bool):
        return True
    if isinstance(settings_version, int):
        return settings_version < _DEEPGRAM_ROLLING_MIGRATION_VERSION
    if isinstance(settings_version, str) and settings_version.strip().isdigit():
        return int(settings_version.strip()) < _DEEPGRAM_ROLLING_MIGRATION_VERSION
    return True


def _requires_deepseek_41_saved_connection_migration(settings_version: object) -> bool:
    if isinstance(settings_version, bool):
        return True
    if isinstance(settings_version, int):
        return settings_version < _DEEPSEEK_41_SAVED_CONNECTION_MIGRATION_VERSION
    if isinstance(settings_version, str) and settings_version.strip().isdigit():
        return int(settings_version.strip()) < _DEEPSEEK_41_SAVED_CONNECTION_MIGRATION_VERSION
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


def _consolidate_cloud_gemma_translation(
    translation: dict[str, Any], *, retained_combined_history: bool
) -> None:
    combined = "gemma4_26b_31b"
    retired = ("gemma4", "gemma4_31b")
    model = translation.get("model")
    previous = translation.get("previous_llm_model")
    history = translation.get("connection_history")
    if not isinstance(history, dict):
        history = {}
    valid_connections = {"managed", "openrouter"}
    current = translation.get("connection")
    if (
        model in (combined, *retired)
        or previous in retired
        or any(key in history for key in (combined, *retired))
    ):
        if model in (combined, *retired) and current in valid_connections:
            selected = current
        else:
            selected = history.get(combined) if retained_combined_history else None
            if selected not in valid_connections:
                selected = history.get(previous) if previous in retired else None
            if selected not in valid_connections:
                selected = history.get("gemma4")
            if selected not in valid_connections:
                selected = history.get("gemma4_31b")
            if selected not in valid_connections:
                selected = "managed"
        history[combined] = selected
        for old_model in retired:
            history.pop(old_model, None)
        translation["connection_history"] = history
    if previous in retired:
        translation["previous_llm_model"] = combined
    if model in retired:
        translation["model"] = combined

    alias = translation.get("openrouter_selection_alias")
    if alias in {"gemma4_managed", "gemma4_31b_managed"}:
        translation["openrouter_selection_alias"] = "gemma4_26b_31b_managed"
    elif alias in {"gemma4_byok", "gemma4_31b_byok"}:
        translation["openrouter_selection_alias"] = "gemma4_26b_31b_byok"
    if translation.get("model") == combined and current in valid_connections:
        source = "managed" if current == "managed" else "byok"
        translation["openrouter_model"] = "google/gemma-4-26b-a4b-it"
        translation["openrouter_provider_routing"] = "gemma4_26b_31b_latency"
        translation["openrouter_selected_source"] = source
        translation["openrouter_selection_alias"] = f"gemma4_26b_31b_{source}"


def _migrate_retired_cerebras_translation(translation: dict[str, Any]) -> None:
    retired_primary = translation.get("model") == "gemma4_31b_cerebras" or (
        translation.get("model") == "gemma4_31b" and translation.get("connection") == "cerebras"
    )
    if retired_primary:
        translation["model"] = "gemma4_31b"
        translation["connection"] = "openrouter"
        translation["openrouter_model"] = "google/gemma-4-31b-it"
        translation["openrouter_selected_source"] = "byok"
        translation["openrouter_selection_alias"] = "gemma4_31b_byok"
        translation["openrouter_provider_routing"] = "gemma4_31b_latency"

    if translation.get("previous_llm_model") == "gemma4_31b_cerebras":
        translation["previous_llm_model"] = "gemma4_31b"

    history = translation.get("connection_history")
    if isinstance(history, dict):
        retired_history = history.pop("gemma4_31b_cerebras", None)
        if history.get("gemma4_31b") == "cerebras" or retired_history is not None:
            history["gemma4_31b"] = "openrouter"

    translation.pop("cerebras", None)


def _migrate_retired_managed_gemma_12b_translation(translation: dict[str, Any]) -> None:
    retired_primary = translation.get("model") == "managed_gemma_12b"
    if retired_primary:
        translation["model"] = "managed_gemma"
        translation["connection"] = "gpu"

    if translation.get("previous_llm_model") == "managed_gemma_12b":
        translation["previous_llm_model"] = "managed_gemma"

    history = translation.get("connection_history")
    if isinstance(history, dict):
        history.pop("managed_gemma_12b", None)
        if retired_primary:
            history["managed_gemma"] = "gpu"


def _migrate_gemini_3_flash_translation(translation: dict[str, Any]) -> None:
    legacy_models = {
        "gemini3_flash",
        "gemini31_flash_lite",
        "gemini-3.1-flash-lite",
        "gemini37_flash",
    }
    if translation.get("model") in legacy_models:
        translation["model"] = "gemini_flash"
    if translation.get("previous_llm_model") in legacy_models:
        translation["previous_llm_model"] = "gemini_flash"
    gemini = translation.get("gemini")
    if isinstance(gemini, dict) and gemini.get("llm_model") in {
        "gemini-3-flash",
        "gemini-3-flash-preview",
        "gemini-3.1-flash-lite",
        "gemini-3.7-flash",
    }:
        gemini["llm_model"] = "gemini-3.8-flash"
    if translation.get("openrouter_model") in {
        "google/gemini-3-flash-preview",
        "google/gemini-3.1-flash-lite",
        "google/gemini-3.7-flash",
    }:
        translation["openrouter_model"] = "google/gemini-3.8-flash"
    if translation.get("openrouter_selection_alias") in {
        "gemini3_flash_byok",
        "gemini31_flash_lite_byok",
        "gemini37_flash_byok",
    }:
        translation["openrouter_selection_alias"] = "gemini_flash_byok"
    history = translation.get("connection_history")
    if isinstance(history, dict):
        for legacy_model in legacy_models:
            if legacy_model in history:
                history.setdefault("gemini_flash", history[legacy_model])
                history.pop(legacy_model, None)


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


def _migrate_deepseek_translation(
    translation: dict[str, Any],
    *,
    migrate_saved_connections: bool,
) -> None:
    if translation.get("model") == "deepseek_v4_pro":
        translation["model"] = "deepseek_v4_flash_41"
        translation["connection"] = "official_byok"
    if translation.get("previous_llm_model") == "deepseek_v4_pro":
        translation["previous_llm_model"] = "deepseek_v4_flash_41"

    history = translation.get("connection_history")
    if isinstance(history, dict) and "deepseek_v4_pro" in history:
        history["deepseek_v4_flash_41"] = "official_byok"
        history.pop("deepseek_v4_pro", None)

    primary_connection = translation.get("connection")
    hidden_managed_primary = (
        migrate_saved_connections
        and translation.get("model") == "deepseek_v4_flash"
        and primary_connection == "openrouter"
        and translation.get("openrouter_selected_source") == "managed"
        and translation.get("openrouter_selection_alias") == "deepseek_v4_flash_managed"
    )
    if hidden_managed_primary:
        primary_connection = (
            "managed_china"
            if translation.get("openrouter_provider_routing") == "deepseek_only"
            else "managed"
        )
        translation["connection"] = primary_connection
    should_upgrade_primary = primary_connection == "official_byok" or (
        migrate_saved_connections and primary_connection in {"managed", "managed_china"}
    )
    if translation.get("model") == "deepseek_v4_flash" and should_upgrade_primary:
        translation["model"] = "deepseek_v4_flash_41"
        if primary_connection in {"managed", "managed_china"}:
            translation["openrouter_model"] = "deepseek/deepseek-v4.1-flash"
            translation["openrouter_selection_alias"] = "deepseek_v4_flash_41_managed"
            translation["openrouter_provider_routing"] = "deepseek_v4_flash_41_strict"
    if translation.get("model") == "deepseek_v4_flash" and primary_connection == "openrouter":
        translation["openrouter_model"] = "deepseek/deepseek-v4-flash-0731"
        translation["openrouter_selection_alias"] = "deepseek_v4_flash_byok"
        translation["openrouter_provider_routing"] = "deepseek_v4_flash_latency"
        translation["openrouter_selected_source"] = "byok"
    if translation.get("model") == "deepseek_v4_flash_41" and primary_connection == "official_byok":
        translation["openrouter_model"] = "deepseek/deepseek-v4.1-flash"
        translation["openrouter_provider_routing"] = "default"
        deepseek = translation.get("deepseek")
        if isinstance(deepseek, dict):
            deepseek["llm_model"] = "deepseek-flash"

    if isinstance(history, dict):
        old_connection = history.get("deepseek_v4_flash")
        should_upgrade_history = old_connection == "official_byok" or (
            migrate_saved_connections and old_connection in {"managed", "managed_china"}
        )
        if should_upgrade_history:
            history["deepseek_v4_flash_41"] = old_connection
            history.pop("deepseek_v4_flash", None)
    if (
        translation.get("previous_llm_model") == "deepseek_v4_flash"
        and isinstance(history, dict)
        and "deepseek_v4_flash_41" in history
    ):
        translation["previous_llm_model"] = "deepseek_v4_flash_41"

    deepseek = translation.get("deepseek")
    if isinstance(deepseek, dict) and deepseek.get("llm_model") == "deepseek-v4-flash":
        deepseek["llm_model"] = "deepseek-flash"


_RETIRED_DEEPSEEK_V4_MANAGED_CONNECTIONS = frozenset({"managed", "managed_china"})


def _migrate_retired_deepseek_v4_managed_path(translation: dict[str, Any]) -> None:
    history = translation.get("connection_history")
    history_map = history if isinstance(history, dict) else None
    primary_connection = translation.get("connection")
    retired_primary = (
        translation.get("model") == "deepseek_v4_flash"
        and primary_connection in _RETIRED_DEEPSEEK_V4_MANAGED_CONNECTIONS
    )
    retired_history = False
    if retired_primary:
        translation["model"] = "deepseek_v4_flash_41"
        translation["openrouter_model"] = "deepseek/deepseek-v4.1-flash"
        translation["openrouter_selected_source"] = "managed"
        translation["openrouter_selection_alias"] = "deepseek_v4_flash_41_managed"
        translation["openrouter_provider_routing"] = "deepseek_v4_flash_41_strict"
        if history_map is not None:
            history_map["deepseek_v4_flash_41"] = primary_connection
            if history_map.get("deepseek_v4_flash") in _RETIRED_DEEPSEEK_V4_MANAGED_CONNECTIONS:
                history_map.pop("deepseek_v4_flash", None)
    elif history_map is not None:
        saved_connection = history_map.get("deepseek_v4_flash")
        if saved_connection in _RETIRED_DEEPSEEK_V4_MANAGED_CONNECTIONS:
            retired_history = True
            history_map.pop("deepseek_v4_flash", None)
            history_map.setdefault("deepseek_v4_flash_41", saved_connection)
    if translation.get("previous_llm_model") == "deepseek_v4_flash" and (
        retired_primary or retired_history
    ):
        translation["previous_llm_model"] = "deepseek_v4_flash_41"


def _migrate_legacy_openrouter_model_translation(translation: dict[str, Any]) -> None:
    raw_model = translation.get("openrouter_model")
    normalized = normalize_legacy_openrouter_model(raw_model)
    if (
        isinstance(raw_model, str)
        and isinstance(normalized, str)
        and normalized != raw_model.strip()
    ):
        translation["openrouter_model"] = normalized


def _migrate_peer_source_auto_mode(intent: dict[str, Any]) -> None:
    raw = intent.get("languages")
    languages = dict(raw) if isinstance(raw, Mapping) else {}
    if languages.get("peer_source_mode") == "soniox_auto":
        languages["peer_source_mode"] = "auto"
        intent["languages"] = languages


def _migrate_qwen_audio_provider(intent: dict[str, Any]) -> None:
    stt = intent.get("stt") if isinstance(intent.get("stt"), dict) else None
    languages = intent.get("languages") if isinstance(intent.get("languages"), Mapping) else {}
    source_language = languages.get("source_language", "")
    peer_source_language = languages.get("peer_source_language") or source_language
    if isinstance(stt, dict):
        if stt.get("provider") == "qwen_asr":
            stt["provider"] = _retired_qwen_provider_for_language(source_language)
        stt.pop("qwen_asr", None)
    peer_stt = intent.get("peer_stt") if isinstance(intent.get("peer_stt"), dict) else None
    if isinstance(peer_stt, dict) and peer_stt.get("provider") == "qwen_asr":
        peer_stt["provider"] = _retired_qwen_provider_for_language(peer_source_language)


def _retired_qwen_provider_for_language(language: object) -> str:
    normalized = language.strip().lower().split("-", 1)[0] if isinstance(language, str) else ""
    return "rolling_free" if normalized in {"tr", "uk"} else "qwen_audio"


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
