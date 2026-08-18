from __future__ import annotations

import math
from dataclasses import asdict, fields, is_dataclass
from importlib import import_module
from types import ModuleType

import pytest

from puripuly_heart.config.audio_host_api import WINDOWS_WASAPI_COMPATIBILITY_HOST_API
from puripuly_heart.config.settings import (
    DEFAULT_DESKTOP_AUDIO_VAD_HANGOVER_MS,
    DEFAULT_OPENROUTER_BROKER_BASE_URL,
)


def _load_schema_module() -> ModuleType:
    try:
        return import_module("puripuly_heart.config.settings_vnext.schema")
    except ModuleNotFoundError as exc:
        pytest.fail(f"vNext settings schema module should import: {exc}")


def _dataclass_leaf_paths(value: object, prefix: str = "") -> set[str]:
    if not is_dataclass(value) or isinstance(value, type):
        return {prefix} if prefix else set()

    paths: set[str] = set()
    for field in fields(value):
        child = getattr(value, field.name)
        child_path = f"{prefix}.{field.name}" if prefix else field.name
        if is_dataclass(child) and not isinstance(child, type):
            paths.update(_dataclass_leaf_paths(child, child_path))
        else:
            paths.add(child_path)
    return paths


def _dict_leaf_paths(value: object, prefix: str = "") -> set[str]:
    if isinstance(value, dict):
        paths: set[str] = set()
        for key, child in value.items():
            child_path = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(child, dict) and child:
                paths.update(_dict_leaf_paths(child, child_path))
            else:
                paths.add(child_path)
        return paths
    return {prefix} if prefix else set()


def test_vnext_schema_imports_required_roots_and_default_adr_destinations() -> None:
    schema = _load_schema_module()

    settings = schema.AppSettingsVNext()

    assert settings.settings_version == schema.VNEXT_SETTINGS_SCHEMA_VERSION
    assert isinstance(settings.intent, schema.UserIntentSettings)
    assert isinstance(settings.state, schema.PersistedOperationalState)
    assert settings.intent.integrated_context.enabled is True
    assert settings.state.integrated_context.bootstrapped is False
    assert settings.state.peer_translation.eula_accepted is False


def test_vnext_schema_defaults_match_current_persisted_settings_defaults() -> None:
    schema = _load_schema_module()

    settings = schema.AppSettingsVNext()

    assert settings.intent.audio.input_host_api == WINDOWS_WASAPI_COMPATIBILITY_HOST_API
    assert settings.intent.desktop_audio.vad_hangover_ms == DEFAULT_DESKTOP_AUDIO_VAD_HANGOVER_MS
    assert (
        settings.intent.translation.openrouter_broker_base_url == DEFAULT_OPENROUTER_BROKER_BASE_URL
    )


def test_vnext_schema_represents_current_intent_and_state_leaves() -> None:
    schema = _load_schema_module()

    leaf_paths = _dataclass_leaf_paths(schema.AppSettingsVNext())

    assert {
        "intent.audio.input_device",
        "intent.audio.input_host_api",
        "intent.audio.ring_buffer_ms",
        "intent.clipboard.auto_translate_enabled",
        "intent.desktop_audio.capture_target.device_name",
        "intent.desktop_audio.capture_target.kind",
        "intent.desktop_audio.vad_hangover_ms",
        "intent.desktop_audio.vad_pre_roll_ms",
        "intent.desktop_audio.vad_speech_threshold",
        "intent.integrated_context.enabled",
        "intent.telemetry.consent",
        "intent.languages.peer_source_language",
        "intent.languages.peer_target_language",
        "intent.languages.peer_source_mode",
        "intent.languages.peer_expected_languages",
        "intent.languages.recent_source_languages",
        "intent.languages.recent_target_languages",
        "intent.languages.source_language",
        "intent.languages.target_language",
        "intent.local_llm.backend",
        "intent.local_llm.base_url",
        "intent.local_llm.extra_body",
        "intent.local_llm.model",
        "intent.osc.chatbox_address",
        "intent.osc.chatbox_clear",
        "intent.osc.chatbox_include_source",
        "intent.osc.chatbox_max_chars",
        "intent.osc.chatbox_send",
        "intent.osc.host",
        "intent.osc.port",
        "intent.osc.vrc_mic_intercept",
        "intent.overlay.calibration.anchor",
        "intent.overlay.calibration.background_alpha",
        "intent.overlay.calibration.distance",
        "intent.overlay.calibration.offset_x",
        "intent.overlay.calibration.offset_y",
        "intent.overlay.calibration.text_scale",
        "intent.overlay.desktop_flet.position.x",
        "intent.overlay.desktop_flet.position.y",
        "intent.overlay.desktop_flet.size_preset",
        "intent.overlay.desktop_flet.visual.background_alpha",
        "intent.overlay.show_peer_original",
        "intent.overlay.show_translation",
        "intent.overlay.target",
        "intent.peer_stt.provider",
        "intent.prompts.system_prompt",
        "intent.secrets.backend",
        "intent.secrets.encrypted_file_path",
        "intent.stt.custom_terms",
        "intent.stt.custom_vocabulary_enabled",
        "intent.stt.custom.compatibility",
        "intent.stt.custom.endpoint",
        "intent.stt.custom.mode",
        "intent.stt.custom.model",
        "intent.stt.deepgram.model",
        "intent.stt.drain_timeout_s",
        "intent.stt.low_latency_merge_gap_ms",
        "intent.stt.low_latency_mode",
        "intent.stt.low_latency_spec_retry_max",
        "intent.stt.low_latency_vad_hangover_ms",
        "intent.stt.provider",
        "intent.stt.qwen_asr.model",
        "intent.stt.soniox.endpoint",
        "intent.stt.soniox.keepalive_interval_s",
        "intent.stt.soniox.model",
        "intent.stt.soniox.trailing_silence_ms",
        "intent.stt.vad_speech_threshold",
        "intent.translation.concurrency_limit",
        "intent.translation.connection",
        "intent.translation.connection_history",
        "intent.translation.cerebras.llm_model",
        "intent.translation.http_extension_id",
        "intent.translation.fallback.connection",
        "intent.translation.fallback.enabled",
        "intent.translation.fallback.model",
        "intent.translation.fallback.selection_alias",
        "intent.translation.model",
        "intent.translation.openrouter_broker_base_url",
        "intent.translation.openrouter_model",
        "intent.translation.openrouter_provider_routing",
        "intent.translation.openrouter_routing_mode",
        "intent.translation.openrouter_selected_source",
        "intent.translation.openrouter_selection_alias",
        "intent.translation.previous_llm_model",
        "intent.translation.qwen.llm_model",
        "intent.translation.qwen.region",
        "intent.ui.locale",
        "state.github_star_prompt.clicked",
        "state.github_star_prompt.eligible_launch_count",
        "state.github_star_prompt.last_shown_at",
        "state.github_star_prompt.show_count",
        "state.github_star_prompt.translation_success_observed",
        "state.integrated_context.bootstrapped",
        "state.telemetry.anonymous_id",
        "state.telemetry.sent_translation_success_dates_utc",
        "state.managed_connection.active_managed_credential_ref",
        "state.managed_connection.active_managed_expires_at",
        "state.managed_connection.founder_letter_seen_credential_ref",
        "state.managed_connection.installation_id",
        "state.managed_connection.referral_id",
        "state.managed_connection.release_token",
        "state.managed_connection.release_token_expires_at",
        "state.managed_connection.verified_hardware_hash",
        "state.managed_connection.verified_hardware_hash_salt_version",
        "state.peer_translation.eula_accepted",
        "state.provider_verification.alibaba_beijing.status",
        "state.provider_verification.alibaba_singapore.status",
        "state.provider_verification.cerebras.status",
        "state.provider_verification.deepgram.status",
        "state.provider_verification.deepseek.status",
        "state.provider_verification.google.status",
        "state.provider_verification.openrouter.status",
        "state.provider_verification.soniox.status",
    } <= leaf_paths


def test_vnext_peer_auto_detection_defaults_to_manual_without_expected_languages() -> None:
    schema = _load_schema_module()

    languages = schema.AppSettingsVNext().intent.languages

    assert languages.peer_source_mode == "manual"
    assert languages.peer_expected_languages == []


def test_vnext_schema_excludes_runtime_only_legacy_ui_state() -> None:
    schema = _load_schema_module()

    leaf_paths = _dataclass_leaf_paths(schema.AppSettingsVNext())

    assert schema.RUNTIME_ONLY_LEGACY_SETTINGS_PATHS == frozenset(
        {"ui.overlay_enabled", "ui.peer_translation_enabled"}
    )
    assert "intent.ui.overlay_enabled" not in leaf_paths
    assert "state.ui.overlay_enabled" not in leaf_paths
    assert "intent.ui.peer_translation_enabled" not in leaf_paths
    assert "state.ui.peer_translation_enabled" not in leaf_paths


def test_telemetry_schema_defaults_keep_intent_and_operational_state_separate() -> None:
    schema = _load_schema_module()

    settings = schema.AppSettingsVNext()

    assert settings.intent.telemetry.consent == "unknown"
    assert settings.state.telemetry.anonymous_id is None
    assert settings.state.telemetry.sent_translation_success_dates_utc == ()
    assert schema.TELEMETRY_CONSENT_VALUES == frozenset({"unknown", "allow", "decline"})


def test_vnext_schema_default_tree_excludes_raw_provider_api_key_fields() -> None:
    schema = _load_schema_module()

    serialized = asdict(schema.AppSettingsVNext())
    serialized_paths = _dict_leaf_paths(serialized)
    forbidden_field_names = {
        "alibaba_api_key",
        "alibaba_api_key_beijing",
        "alibaba_api_key_singapore",
        "deepgram_api_key",
        "deepseek_api_key",
        "cerebras_api_key",
        "google_api_key",
        "local_llm_api_key",
        "openrouter_api_key",
        "openrouter_managed_api_key",
        "openrouter_managed_qq_api_key",
        "soniox_api_key",
    }

    assert not {
        path
        for path in serialized_paths
        if path.rsplit(".", maxsplit=1)[-1] in forbidden_field_names
    }
    intent = schema.LocalLLMIntent(extra_body={"api_key": "not-a-real-secret"})
    assert intent.extra_body == {"reasoning_effort": "none"}


def test_provider_verification_entry_defaults_to_unknown_without_bound_evidence() -> None:
    schema = _load_schema_module()

    entry = schema.ProviderVerificationEntry()

    assert entry.status == "unknown"
    assert entry.provider is None
    assert entry.secret_key is None
    assert entry.secret_revision is None
    assert entry.secret_fingerprint is None
    assert entry.verifier_context == {}
    assert entry.verifier_evidence == {}


def _bound_provider_verification_entry_kwargs() -> dict[str, object]:
    return {
        "status": "verified",
        "provider": "openrouter",
        "secret_key": "openrouter_api_key",
        "secret_revision": "secret-r1",
        "secret_fingerprint": None,
        "verifier_context": {"flow": "settings.verify_api_key"},
        "verifier_evidence": {"verifier": "openrouter"},
    }


@pytest.mark.parametrize("status", ["verified", "failed", "skipped"])
@pytest.mark.parametrize(
    ("overrides", "case_id"),
    [
        ({"provider": None}, "missing-provider"),
        ({"provider": "   "}, "blank-provider"),
        ({"secret_key": None}, "missing-secret-key"),
        ({"secret_key": "   "}, "blank-secret-key"),
        (
            {"secret_revision": None, "secret_fingerprint": None},
            "missing-secret-binding",
        ),
        (
            {"secret_revision": "", "secret_fingerprint": "   "},
            "blank-secret-binding",
        ),
        ({"verifier_context": {}}, "missing-verifier-context"),
    ],
)
def test_provider_verification_entry_rejects_unbound_non_unknown_entries(
    status: str,
    overrides: dict[str, object],
    case_id: str,
) -> None:
    _ = case_id
    schema = _load_schema_module()
    kwargs = _bound_provider_verification_entry_kwargs()
    kwargs.update(overrides)
    kwargs["status"] = status

    with pytest.raises(ValueError, match="provider verification evidence"):
        schema.ProviderVerificationEntry(**kwargs)


@pytest.mark.parametrize("evidence_key", ["api_key", "raw_response", "response_body"])
def test_provider_verification_entry_rejects_secret_bearing_evidence_keys(
    evidence_key: str,
) -> None:
    schema = _load_schema_module()

    with pytest.raises(ValueError, match="secret-bearing provider verification"):
        schema.ProviderVerificationEntry(
            status="verified",
            provider="openrouter",
            secret_key="openrouter_api_key",
            secret_revision="secret-r1",
            verifier_context={"flow": "settings.verify_api_key"},
            verifier_evidence={evidence_key: "redacted-test-sentinel"},
        )


def test_vnext_schema_excludes_deferred_vnext_only_operational_leaves() -> None:
    schema = _load_schema_module()

    leaf_paths = _dataclass_leaf_paths(schema.AppSettingsVNext())

    assert (
        not {
            "state.provider_verification.alibaba_beijing.credential_hash",
            "state.provider_verification.alibaba_beijing.credential_ref",
            "state.provider_verification.alibaba_beijing.verified_at",
            "state.provider_verification.alibaba_singapore.credential_hash",
            "state.provider_verification.cerebras.credential_hash",
            "state.provider_verification.cerebras.credential_ref",
            "state.provider_verification.cerebras.verified_at",
            "state.provider_verification.alibaba_singapore.credential_ref",
            "state.provider_verification.alibaba_singapore.verified_at",
            "state.provider_verification.deepgram.credential_hash",
            "state.provider_verification.deepgram.credential_ref",
            "state.provider_verification.deepgram.verified_at",
            "state.provider_verification.deepseek.credential_hash",
            "state.provider_verification.deepseek.credential_ref",
            "state.provider_verification.deepseek.verified_at",
            "state.provider_verification.google.credential_hash",
            "state.provider_verification.google.credential_ref",
            "state.provider_verification.google.verified_at",
            "state.provider_verification.openrouter.credential_hash",
            "state.provider_verification.openrouter.credential_ref",
            "state.provider_verification.openrouter.verified_at",
            "state.provider_verification.soniox.credential_hash",
            "state.provider_verification.soniox.credential_ref",
            "state.provider_verification.soniox.verified_at",
            "state.prompts.seen_prompt_ids",
        }
        & leaf_paths
    )


@pytest.mark.parametrize(
    ("extra_body", "expected"),
    [
        ({"model": "reserved"}, {"reasoning_effort": "none"}),
        ({"api-key": "secret alias"}, {"reasoning_effort": "none"}),
        ({"nested": {"authorization": "Bearer token"}}, {"nested": {"reasoning_effort": "none"}}),
        ({"temperature": math.nan}, {"reasoning_effort": "none"}),
        ({"temperature": math.inf}, {"reasoning_effort": "none"}),
        ({"temperature": -math.inf}, {"reasoning_effort": "none"}),
    ],
)
def test_local_llm_extra_body_falls_back_for_non_persistable_values(
    extra_body: dict[object, object],
    expected: dict[object, object],
) -> None:
    schema = _load_schema_module()

    intent = schema.LocalLLMIntent(extra_body=extra_body)
    assert intent.extra_body == expected


@pytest.mark.parametrize(
    "extra_body",
    [
        {"x-api-key": "secret alias"},
        {"xApiKey": "secret alias"},
        {"openai_api_key": "secret alias"},
        {"openaiApiKey": "secret alias"},
        {"OpenAIApiKey": "secret alias"},
        {"clientSecret": "secret alias"},
        {"refreshToken": "secret alias"},
        {"proxy_authorization": "Bearer token"},
        {"nested": {"azure-openai-api-key": "nested secret alias"}},
        {"nested": {"openaiApiKey": "nested secret alias"}},
        {"nested": {"clientSecret": "nested secret alias"}},
        {"nested": {"refreshToken": "nested secret alias"}},
    ],
)
def test_local_llm_extra_body_accepts_legacy_compatible_keys(
    extra_body: dict[object, object],
) -> None:
    schema = _load_schema_module()

    intent = schema.LocalLLMIntent(extra_body=extra_body)
    assert intent.extra_body == extra_body


def test_local_llm_extra_body_skips_non_string_nested_keys() -> None:
    schema = _load_schema_module()

    intent = schema.LocalLLMIntent(extra_body={"nested": {1: "non-string key"}})
    assert intent.extra_body == {"nested": {}}


def test_local_llm_extra_body_accepts_json_safe_nested_mappings() -> None:
    schema = _load_schema_module()

    settings = schema.LocalLLMIntent(
        extra_body={
            "api_key_required": False,
            "metadata": {"temperature_label": "low", "retry_count": 2},
            "tokenizer_model": "cl100k_base",
        }
    )

    assert settings.extra_body == {
        "api_key_required": False,
        "metadata": {"temperature_label": "low", "retry_count": 2},
        "tokenizer_model": "cl100k_base",
    }


def test_vnext_schema_uses_overlay_calibration_value_object() -> None:
    schema = _load_schema_module()

    settings = schema.AppSettingsVNext()

    assert settings.intent.overlay.calibration.__class__.__module__ == (
        "puripuly_heart.config.overlay_calibration"
    )
