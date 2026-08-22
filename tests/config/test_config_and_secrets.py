from __future__ import annotations

import copy
import json
import re
from pathlib import Path

import pytest

from puripuly_heart.config.audio_host_api import (
    WINDOWS_DIRECTSOUND_HOST_API,
    WINDOWS_MME_HOST_API,
    WINDOWS_WASAPI_COMPATIBILITY_HOST_API,
    WINDOWS_WASAPI_HOST_API,
)
from puripuly_heart.config.llm_profiles import (
    OPENROUTER_FALLBACK_SELECTION_ALIASES,
    get_openrouter_llm_profile,
    openrouter_alias_for_fields,
    resolve_openrouter_fallback_model,
)
from puripuly_heart.config.prompts import load_prompt_for_provider
from puripuly_heart.config.settings import (
    LEGACY_QWEN_DEFAULT_PROMPT,
    LEGACY_TIMESTAMP_PROMPT,
    LOCAL_LLM_RESERVED_EXTRA_BODY_KEYS,
    LOCAL_LLM_SENSITIVE_EXTRA_BODY_KEYS,
    SETTINGS_SCHEMA_VERSION,
    AppSettings,
    AudioSettings,
    CerebrasLLMModel,
    DeepSeekLLMModel,
    DeepSeekSettings,
    GeminiLLMModel,
    LLMProviderName,
    LocalLLMBackend,
    LocalLLMSettings,
    OpenRouterCredentialSource,
    OpenRouterFallbackSelectionAlias,
    OpenRouterLLMModel,
    OpenRouterProviderRouting,
    OpenRouterRoutingMode,
    OpenRouterSelectionAlias,
    OpenRouterSettings,
    OSCSettings,
    ProviderSettings,
    QwenLLMModel,
    QwenRegion,
    STTProviderName,
    TranslationConnection,
    TranslationFallbackSettings,
    TranslationModel,
    TranslationSettings,
    _migrate_settings_dict,
    default_translation_connection,
    from_dict,
    load_settings,
    materialize_translation_settings,
    save_settings,
    supported_translation_connections,
    to_dict,
)
from puripuly_heart.core.storage.secrets import (
    EncryptedFileSecretStore,
    KeyringSecretStore,
    mask_secret,
)
from tests.config.settings_vnext_test_helpers import (
    legacy_projected_settings_file,
    load_raw_json_file,
)

DEFAULT_TRANSLATION_FALLBACK_DICT = {
    "enabled": False,
    "model": TranslationModel.DEEPSEEK_V4_FLASH.value,
    "connection": TranslationConnection.OFFICIAL_BYOK.value,
}
MIGRATED_DEFAULT_TRANSLATION_FALLBACK_DICT = {
    "enabled": True,
    "model": TranslationModel.GEMMA4_26B_31B.value,
    "connection": TranslationConnection.OPENROUTER.value,
}


def test_settings_roundtrip(tmp_path):
    path = tmp_path / "settings.json"
    settings = AppSettings()
    save_settings(path, settings)

    loaded = load_settings(path)
    expected = AppSettings()
    expected.languages.recent_source_languages = ["en", "zh-CN", "ja", "ko", "es", "fr"]
    expected.languages.recent_target_languages = ["en", "zh-CN", "ja", "ko", "es", "fr"]
    shared_prompt = load_prompt_for_provider("gemini")
    expected.system_prompt = shared_prompt
    expected.system_prompts = {}
    expected.telemetry.consent = "allow"
    expected.telemetry_state.anonymous_id = loaded.telemetry_state.anonymous_id
    expected.translation.fallback = TranslationFallbackSettings(
        enabled=True,
        model=TranslationModel.GEMMA4_26B_31B,
        connection=TranslationConnection.OPENROUTER,
    )

    assert loaded.telemetry_state.anonymous_id
    assert loaded == expected


def test_peer_and_self_vad_hangover_defaults_are_500_ms() -> None:
    settings = AppSettings()

    assert settings.desktop_audio.vad_hangover_ms == 500
    assert settings.stt.low_latency_vad_hangover_ms == 500
    assert (
        from_dict(
            {"settings_version": SETTINGS_SCHEMA_VERSION, "desktop_audio": {}}
        ).desktop_audio.vad_hangover_ms
        == 500
    )


@pytest.mark.parametrize("legacy_hangover_ms", [450, 500, 600, 700, 900])
def test_schema21_migration_forces_existing_peer_vad_hangover_to_500_ms(
    tmp_path, legacy_hangover_ms: int
) -> None:
    path = tmp_path / "settings.json"
    legacy = to_dict(AppSettings())
    legacy["settings_version"] = 20
    legacy["desktop_audio"]["vad_hangover_ms"] = legacy_hangover_ms
    path.write_text(json.dumps(legacy, ensure_ascii=False, indent=2), encoding="utf-8")

    loaded = load_settings(path)
    persisted = legacy_projected_settings_file(path)

    assert loaded.settings_version == SETTINGS_SCHEMA_VERSION
    assert loaded.desktop_audio.vad_hangover_ms == 500
    assert loaded.stt.low_latency_vad_hangover_ms == 500
    assert persisted["settings_version"] == SETTINGS_SCHEMA_VERSION
    assert persisted["desktop_audio"]["vad_hangover_ms"] == 500


def test_new_user_defaults_peer_voice_to_english_to_korean_cpu_auto() -> None:
    settings = AppSettings()

    assert settings.languages.source_language == "ko"
    assert settings.languages.target_language == "en"
    assert settings.languages.peer_source_language == "en"
    assert settings.languages.peer_target_language == "ko"
    assert settings.languages.effective_peer_source == "en"
    assert settings.languages.effective_peer_target == "ko"
    assert settings.provider.peer_stt == STTProviderName.LOCAL_CPU_AUTO
    assert settings.ui.integrated_context_enabled is True
    assert settings.osc.chatbox_include_source is False


def test_partial_settings_deserialization_preserves_legacy_peer_fallbacks() -> None:
    settings = from_dict({})

    assert settings.languages.source_language == "ko"
    assert settings.languages.target_language == "en"
    assert settings.languages.peer_source_language == ""
    assert settings.languages.peer_target_language == ""
    assert settings.languages.effective_peer_source == "ko"
    assert settings.languages.effective_peer_target == "en"
    assert settings.provider.peer_stt == STTProviderName.DEEPGRAM
    assert settings.ui.integrated_context_enabled is True
    assert settings.osc.chatbox_include_source is False


def test_from_dict_preserves_explicit_integrated_context_disabled() -> None:
    settings = from_dict({"ui": {"integrated_context_enabled": False}})

    assert settings.ui.integrated_context_enabled is False


def test_clipboard_auto_translate_defaults_off_for_new_and_partial_settings() -> None:
    assert AppSettings().ui.clipboard_auto_translate_enabled is False
    assert from_dict({}).ui.clipboard_auto_translate_enabled is False


@pytest.mark.parametrize("enabled", [True, False])
def test_clipboard_auto_translate_roundtrips_through_ui_settings(enabled: bool) -> None:
    settings = AppSettings()
    settings.ui.clipboard_auto_translate_enabled = enabled

    data = to_dict(settings)

    assert data["ui"]["clipboard_auto_translate_enabled"] is enabled
    assert from_dict(data).ui.clipboard_auto_translate_enabled is enabled


def test_from_dict_defaults_missing_clipboard_auto_translate_to_false() -> None:
    raw = to_dict(AppSettings())
    raw["ui"].pop("clipboard_auto_translate_enabled", None)

    loaded = from_dict(raw)

    assert loaded.ui.clipboard_auto_translate_enabled is False


def test_save_settings_writes_via_temp_replace(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "settings.json"
    settings = AppSettings()
    replace_calls: list[tuple[str, str]] = []
    path_type = type(path)
    original_replace = path_type.replace

    def recording_replace(self: Path, target: Path) -> Path:
        replace_calls.append((self.name, Path(target).name))
        return original_replace(self, target)

    monkeypatch.setattr(path_type, "replace", recording_replace)

    save_settings(path, settings)

    assert replace_calls == [("settings.json.tmp", "settings.json")]


def test_save_settings_preserves_existing_file_when_replace_fails(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "settings.json"
    original_payload = {"keep": True}
    path.write_text(json.dumps(original_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    path_type = type(path)

    def failing_replace(self: Path, target: Path) -> Path:
        raise RuntimeError("replace failed")

    monkeypatch.setattr(path_type, "replace", failing_replace)

    with pytest.raises(RuntimeError, match="replace failed"):
        save_settings(path, AppSettings())

    assert load_raw_json_file(path) == original_payload
    assert not path.with_suffix(path.suffix + ".tmp").exists()


def test_settings_validation_rejects_invalid_audio():
    settings = AppSettings(audio=AudioSettings(internal_sample_rate_hz=123))
    with pytest.raises(ValueError):
        settings.validate()


def test_settings_validation_rejects_legacy_8khz_audio() -> None:
    settings = AppSettings(audio=AudioSettings(internal_sample_rate_hz=8000))

    with pytest.raises(ValueError, match="internal_sample_rate_hz"):
        settings.validate()


def test_default_audio_host_api_is_wasapi_compatibility_mode() -> None:
    settings = AppSettings()

    assert settings.audio.input_host_api == WINDOWS_WASAPI_COMPATIBILITY_HOST_API
    assert to_dict(settings)["audio"]["input_host_api"] == WINDOWS_WASAPI_COMPATIBILITY_HOST_API


def test_from_dict_defaults_missing_audio_host_api_to_wasapi_compatibility_mode() -> None:
    raw = to_dict(AppSettings())
    raw["audio"].pop("input_host_api")

    loaded = from_dict(raw)

    assert loaded.audio.input_host_api == WINDOWS_WASAPI_COMPATIBILITY_HOST_API
    assert to_dict(loaded)["audio"]["input_host_api"] == WINDOWS_WASAPI_COMPATIBILITY_HOST_API


def test_from_dict_defaults_missing_audio_block_host_api_to_wasapi_compatibility_mode() -> None:
    loaded = from_dict({})

    assert loaded.audio.input_host_api == WINDOWS_WASAPI_COMPATIBILITY_HOST_API


def test_from_dict_preserves_explicit_blank_audio_host_api() -> None:
    raw = to_dict(AppSettings())
    raw["audio"]["input_host_api"] = ""

    loaded = from_dict(raw)

    assert loaded.audio.input_host_api == ""
    assert to_dict(loaded)["audio"]["input_host_api"] == ""


@pytest.mark.parametrize(
    "host_api",
    [
        WINDOWS_WASAPI_HOST_API,
        WINDOWS_WASAPI_COMPATIBILITY_HOST_API,
        WINDOWS_DIRECTSOUND_HOST_API,
        WINDOWS_MME_HOST_API,
        "Unknown Host API",
    ],
)
def test_from_dict_preserves_explicit_audio_host_api_values(host_api: str) -> None:
    raw = to_dict(AppSettings())
    raw["audio"]["input_host_api"] = host_api

    loaded = from_dict(raw)

    assert loaded.audio.input_host_api == host_api
    assert to_dict(loaded)["audio"]["input_host_api"] == host_api


def test_migrate_v17_preserves_saved_directsound_host_api_and_device() -> None:
    raw = to_dict(AppSettings())
    raw["settings_version"] = 16
    raw["audio"]["input_host_api"] = WINDOWS_DIRECTSOUND_HOST_API
    raw["audio"]["input_device"] = "User Selected Mic"

    migrated, changed = _migrate_settings_dict(raw)

    assert changed is True
    assert migrated["settings_version"] == SETTINGS_SCHEMA_VERSION
    assert migrated["audio"]["input_host_api"] == WINDOWS_DIRECTSOUND_HOST_API
    assert migrated["audio"]["input_device"] == "User Selected Mic"


def test_migrate_v17_normalizes_directsound_host_api_and_preserves_device() -> None:
    raw = to_dict(AppSettings())
    raw["settings_version"] = 16
    raw["audio"]["input_host_api"] = f" {WINDOWS_DIRECTSOUND_HOST_API} "
    raw["audio"]["input_device"] = "Whitespace DirectSound Mic"

    migrated, changed = _migrate_settings_dict(raw)

    assert changed is True
    assert migrated["settings_version"] == SETTINGS_SCHEMA_VERSION
    assert migrated["audio"]["input_host_api"] == WINDOWS_DIRECTSOUND_HOST_API
    assert migrated["audio"]["input_device"] == "Whitespace DirectSound Mic"


def test_migrate_v18_preserves_directsound_when_removing_legacy_osc_rate_limits() -> None:
    assert SETTINGS_SCHEMA_VERSION == 25

    raw = to_dict(AppSettings())
    raw["settings_version"] = 17
    raw["audio"]["input_host_api"] = WINDOWS_DIRECTSOUND_HOST_API
    raw["audio"]["input_device"] = "Reselected DirectSound Mic"
    raw["osc"]["host"] = "192.0.2.10"
    raw["osc"]["port"] = 9010
    raw["osc"]["chatbox_max_chars"] = 72
    raw["osc"]["vrc_mic_intercept"] = True
    raw["osc"]["cooldown_s"] = 1.5
    raw["osc"]["ttl_s"] = 7.0
    expected_osc = dict(raw["osc"])
    expected_osc.pop("cooldown_s")
    expected_osc.pop("ttl_s")

    migrated, changed = _migrate_settings_dict(raw)

    assert changed is True
    assert migrated["settings_version"] == SETTINGS_SCHEMA_VERSION
    assert migrated["audio"]["input_host_api"] == WINDOWS_DIRECTSOUND_HOST_API
    assert migrated["audio"]["input_device"] == "Reselected DirectSound Mic"
    assert migrated["osc"] == expected_osc


def test_load_settings_persists_v17_directsound_preservation(tmp_path) -> None:
    path = tmp_path / "settings.json"
    raw = to_dict(AppSettings())
    raw["settings_version"] = 16
    raw["audio"]["input_host_api"] = WINDOWS_DIRECTSOUND_HOST_API
    raw["audio"]["input_device"] = "Manual DirectSound Mic"
    path.write_text(json.dumps(raw), encoding="utf-8")

    loaded = load_settings(path)
    stored = legacy_projected_settings_file(path)

    assert loaded.audio.input_host_api == WINDOWS_DIRECTSOUND_HOST_API
    assert loaded.audio.input_device == "Manual DirectSound Mic"
    assert stored["settings_version"] == SETTINGS_SCHEMA_VERSION
    assert stored["audio"]["input_host_api"] == WINDOWS_DIRECTSOUND_HOST_API
    assert stored["audio"]["input_device"] == "Manual DirectSound Mic"


def test_load_settings_persists_v18_osc_rate_limit_key_removal(tmp_path) -> None:
    assert SETTINGS_SCHEMA_VERSION == 25

    path = tmp_path / "settings.json"
    raw = to_dict(AppSettings())
    raw["settings_version"] = 17
    raw["osc"]["host"] = "192.0.2.20"
    raw["osc"].pop("connection_mode", None)
    raw["osc"].pop("send_port", None)
    raw["osc"].pop("receive_port", None)
    raw["osc"]["port"] = 9011
    raw["osc"]["chatbox_max_chars"] = 96
    raw["osc"]["vrc_mic_intercept"] = True
    raw["osc"]["cooldown_s"] = 1.5
    raw["osc"]["ttl_s"] = 7.0
    expected_osc = dict(raw["osc"])
    expected_osc.pop("cooldown_s")
    expected_osc.pop("ttl_s")
    expected_osc.update(
        {
            "connection_mode": "automatic",
            "send_port": 9011,
            "receive_port": 9001,
        }
    )
    path.write_text(json.dumps(raw), encoding="utf-8")

    loaded = load_settings(path)
    stored = legacy_projected_settings_file(path)

    assert loaded.settings_version == SETTINGS_SCHEMA_VERSION
    assert loaded.osc.host == "192.0.2.20"
    assert loaded.osc.port == 9011
    assert loaded.osc.chatbox_max_chars == 96
    assert loaded.osc.vrc_mic_intercept is True
    assert stored["settings_version"] == SETTINGS_SCHEMA_VERSION
    assert stored["osc"] == expected_osc


def test_from_dict_ignores_legacy_osc_rate_limit_keys() -> None:
    raw = to_dict(AppSettings())
    raw["osc"]["cooldown_s"] = "bad"
    raw["osc"]["ttl_s"] = "bad"

    loaded = from_dict(raw)
    persisted = to_dict(loaded)

    assert not hasattr(loaded.osc, "cooldown_s")
    assert not hasattr(loaded.osc, "ttl_s")
    assert "cooldown_s" not in persisted["osc"]
    assert "ttl_s" not in persisted["osc"]


def test_settings_validation_rejects_invalid_osc():
    settings = AppSettings(osc=OSCSettings(chatbox_max_chars=0))
    with pytest.raises(ValueError):
        settings.validate()


def test_default_stt_provider_is_local_cpu_auto() -> None:
    settings = AppSettings()

    assert settings.provider.stt == STTProviderName.LOCAL_CPU_AUTO
    assert to_dict(settings)["provider"]["stt"] == STTProviderName.LOCAL_CPU_AUTO.value


def test_self_vad_and_soniox_defaults_match_current_compatibility_defaults() -> None:
    settings = from_dict({})

    assert settings.stt.low_latency_vad_hangover_ms == 500
    assert settings.soniox_stt.model == "stt-rt-v5"
    assert to_dict(settings)["stt"]["low_latency_vad_hangover_ms"] == 500
    assert to_dict(settings)["soniox_stt"]["model"] == "stt-rt-v5"


@pytest.mark.parametrize("legacy_model", ["stt-rt-v3", "stt-rt-v4"])
def test_migrate_settings_dict_upgrades_only_legacy_soniox_defaults(legacy_model: str) -> None:
    raw = to_dict(AppSettings())
    raw["soniox_stt"]["model"] = legacy_model

    migrated, changed = _migrate_settings_dict(raw)

    assert changed is True
    assert migrated["soniox_stt"]["model"] == "stt-rt-v5"


def test_migrate_settings_dict_preserves_custom_soniox_model() -> None:
    raw = to_dict(AppSettings())
    raw["soniox_stt"]["model"] = "custom-soniox-model"

    migrated, _changed = _migrate_settings_dict(raw)

    assert migrated["soniox_stt"]["model"] == "custom-soniox-model"


def test_translation_model_public_member_names_and_values_match_plan() -> None:
    assert tuple((member.name, member.value) for member in TranslationModel) == (
        ("GEMMA4_26B_31B", "gemma4_26b_31b"),
        ("GEMMA4_31B", "gemma4_31b"),
        ("GEMMA4", "gemma4"),
        ("DEEPSEEK_V4_FLASH", "deepseek_v4_flash"),
        ("GEMINI_37_FLASH", "gemini37_flash"),
        ("GEMINI_31_FLASH_LITE", "gemini31_flash_lite"),
        ("QWEN_35_PLUS", "qwen35_plus"),
        ("MANAGED_GEMMA", "managed_gemma"),
        ("MANAGED_GEMMA_12B", "managed_gemma_12b"),
        ("LOCAL_LLM", "local_llm"),
        ("CUSTOM_HTTP", "custom_http"),
    )


def test_local_llm_enum_values_are_stable() -> None:
    assert LLMProviderName.LOCAL_LLM.value == "local_llm"
    assert TranslationModel.LOCAL_LLM.value == "local_llm"
    assert TranslationConnection.OLLAMA.value == "ollama"
    assert LocalLLMBackend.OLLAMA.value == "ollama"


def test_local_llm_settings_default_and_roundtrip() -> None:
    settings = from_dict({})

    assert settings.local_llm.backend == LocalLLMBackend.OLLAMA
    assert settings.local_llm.base_url == "http://127.0.0.1:11434/v1"
    assert settings.local_llm.model == "llama3.1:8b"
    assert settings.local_llm.extra_body == {"reasoning_effort": "none"}

    settings.translation = TranslationSettings(
        model=TranslationModel.LOCAL_LLM,
        connection=TranslationConnection.OLLAMA,
    )
    settings.local_llm.base_url = "http://192.168.0.25:11434/v1/"
    settings.local_llm.model = "qwen2.5:7b"
    settings.local_llm.extra_body = {"enable_thinking": False}
    materialize_translation_settings(settings)

    persisted = to_dict(settings)
    assert persisted["provider"]["llm"] == LLMProviderName.LOCAL_LLM.value
    assert persisted["local_llm"] == {
        "backend": LocalLLMBackend.OLLAMA.value,
        "base_url": "http://192.168.0.25:11434/v1",
        "model": "qwen2.5:7b",
        "extra_body": {"enable_thinking": False},
    }

    loaded = from_dict(persisted)
    assert loaded.provider.llm == LLMProviderName.LOCAL_LLM
    assert loaded.translation.model == TranslationModel.LOCAL_LLM
    assert loaded.translation.connection == TranslationConnection.OLLAMA
    assert loaded.local_llm.base_url == "http://192.168.0.25:11434/v1"
    assert loaded.local_llm.model == "qwen2.5:7b"
    assert loaded.local_llm.extra_body == {"enable_thinking": False}


def test_local_llm_api_key_is_not_serialized_in_settings() -> None:
    settings = AppSettings()
    persisted = to_dict(settings)

    assert "local_llm_api_key" not in persisted
    assert "api_key" not in persisted["local_llm"]
    assert "local_llm" not in persisted["api_key_verified"]


def test_local_llm_stray_api_key_settings_are_ignored_on_roundtrip() -> None:
    raw = to_dict(AppSettings())
    raw["local_llm"]["api_key"] = "do-not-persist"
    raw["api_key_verified"]["local_llm"] = True

    loaded = from_dict(raw)
    persisted = to_dict(loaded)

    assert "api_key" not in persisted["local_llm"]
    assert "local_llm" not in persisted["api_key_verified"]


def test_translation_settings_defaults_to_gemma_managed_with_only_gemma_history() -> None:
    settings = TranslationSettings()

    assert settings.model == TranslationModel.GEMMA4_26B_31B
    assert settings.connection == TranslationConnection.MANAGED
    assert settings.connection_history == {
        TranslationModel.GEMMA4_26B_31B.value: TranslationConnection.MANAGED
    }
    assert to_dict(AppSettings())["translation"] == {
        "model": TranslationModel.GEMMA4_26B_31B.value,
        "connection": TranslationConnection.MANAGED.value,
        "connection_history": {
            TranslationModel.GEMMA4_26B_31B.value: TranslationConnection.MANAGED.value,
        },
        "fallback": DEFAULT_TRANSLATION_FALLBACK_DICT,
        "gpu_device_id": "auto",
    }


def test_public_translation_connection_helpers_match_model_matrix() -> None:
    assert supported_translation_connections(TranslationModel.GEMMA4) == (
        TranslationConnection.MANAGED,
        TranslationConnection.OPENROUTER,
    )
    assert supported_translation_connections(TranslationModel.DEEPSEEK_V4_FLASH) == (
        TranslationConnection.MANAGED,
        TranslationConnection.MANAGED_CHINA,
        TranslationConnection.OPENROUTER,
        TranslationConnection.OFFICIAL_BYOK,
    )
    assert supported_translation_connections(TranslationModel.GEMINI_37_FLASH) == (
        TranslationConnection.OFFICIAL_BYOK,
        TranslationConnection.OPENROUTER,
    )
    assert supported_translation_connections(TranslationModel.GEMINI_31_FLASH_LITE) == (
        TranslationConnection.OFFICIAL_BYOK,
        TranslationConnection.OPENROUTER,
    )
    assert supported_translation_connections(TranslationModel.QWEN_35_PLUS) == (
        TranslationConnection.OFFICIAL_BYOK,
    )
    assert supported_translation_connections(TranslationModel.LOCAL_LLM) == (
        TranslationConnection.OLLAMA,
    )
    assert supported_translation_connections(TranslationModel.GEMMA4_31B) == (
        TranslationConnection.MANAGED,
        TranslationConnection.OPENROUTER,
        TranslationConnection.CEREBRAS,
    )
    assert default_translation_connection(TranslationModel.GEMMA4) == TranslationConnection.MANAGED
    assert (
        default_translation_connection(TranslationModel.GEMINI_37_FLASH)
        == TranslationConnection.OFFICIAL_BYOK
    )
    assert (
        default_translation_connection(TranslationModel.LOCAL_LLM) == TranslationConnection.OLLAMA
    )
    assert (
        default_translation_connection(TranslationModel.GEMMA4_31B) == TranslationConnection.MANAGED
    )


def test_supported_translation_connections_include_ollama_for_local_model() -> None:
    assert supported_translation_connections(TranslationModel.LOCAL_LLM) == (
        TranslationConnection.OLLAMA,
    )
    assert (
        default_translation_connection(TranslationModel.LOCAL_LLM) == TranslationConnection.OLLAMA
    )


def test_from_dict_local_llm_provider_infers_translation_settings() -> None:
    data = to_dict(AppSettings())
    data["provider"]["llm"] = LLMProviderName.LOCAL_LLM.value
    data["local_llm"] = {
        "backend": "ollama",
        "base_url": "http://127.0.0.1:11434/v1",
        "model": "llama3.1:8b",
        "extra_body": {"think": False},
    }
    data.pop("translation", None)

    loaded = from_dict(data)

    assert loaded.translation.model == TranslationModel.LOCAL_LLM
    assert loaded.translation.connection == TranslationConnection.OLLAMA
    assert loaded.provider.llm == LLMProviderName.LOCAL_LLM


def test_from_dict_roundtrips_raw_local_llm_strings() -> None:
    loaded = from_dict(
        {
            "provider": {"llm": "local_llm"},
            "translation": {"model": "local_llm", "connection": "ollama"},
            "local_llm": {
                "backend": "ollama",
                "base_url": "http://mac-studio.local:11434/v1/",
                "model": "gemma3:4b",
                "extra_body": {"thinking": {"type": "disabled"}},
            },
        }
    )

    assert loaded.provider.llm == LLMProviderName.LOCAL_LLM
    assert loaded.translation.model == TranslationModel.LOCAL_LLM
    assert loaded.translation.connection == TranslationConnection.OLLAMA
    assert loaded.local_llm.backend == LocalLLMBackend.OLLAMA
    assert loaded.local_llm.base_url == "http://mac-studio.local:11434/v1"
    assert loaded.local_llm.model == "gemma3:4b"
    assert loaded.local_llm.extra_body == {"thinking": {"type": "disabled"}}


@pytest.mark.parametrize("key", sorted(LOCAL_LLM_RESERVED_EXTRA_BODY_KEYS))
def test_local_llm_rejects_reserved_extra_body_keys(key: str) -> None:
    with pytest.raises(ValueError, match=key):
        LocalLLMSettings(extra_body={key: True}).validate()


@pytest.mark.parametrize("key", sorted(LOCAL_LLM_SENSITIVE_EXTRA_BODY_KEYS))
def test_local_llm_rejects_sensitive_extra_body_keys(key: str) -> None:
    with pytest.raises(ValueError, match=key):
        LocalLLMSettings(extra_body={key: "do-not-persist"}).validate()


def test_local_llm_rejects_non_json_serializable_extra_body() -> None:
    with pytest.raises(ValueError, match="JSON serializable"):
        LocalLLMSettings(extra_body={"callback": object()}).validate()


@pytest.mark.parametrize(
    "value",
    [float("nan"), float("inf"), float("-inf")],
    ids=["nan", "infinity", "negative_infinity"],
)
def test_local_llm_validate_rejects_non_standard_json_constants_extra_body(value: float) -> None:
    with pytest.raises(ValueError, match="JSON serializable"):
        LocalLLMSettings(extra_body={"temperature": value}).validate()


@pytest.mark.parametrize(
    "base_url",
    [
        "ftp://127.0.0.1:11434/v1",
        "http://127.0.0.1:11434/v1?debug=true",
        "http://127.0.0.1:11434/v1#fragment",
        "http://user@127.0.0.1:11434/v1",
        "http://user:pass@127.0.0.1:11434/v1",
        "http://@127.0.0.1:11434/v1",
        "http://127.0.0.1:99999/v1",
    ],
)
def test_local_llm_validate_rejects_invalid_base_url(base_url: str) -> None:
    with pytest.raises(ValueError, match="base url"):
        LocalLLMSettings(base_url=base_url).validate()


def test_local_llm_validate_rejects_empty_model() -> None:
    with pytest.raises(ValueError, match="model"):
        LocalLLMSettings(model="   ").validate()


def test_local_llm_extra_body_key_sets_match_provider_constants() -> None:
    from puripuly_heart.providers.llm.local_openai import (
        LOCAL_OPENAI_RESERVED_EXTRA_BODY_KEYS,
        LOCAL_OPENAI_SENSITIVE_EXTRA_BODY_KEYS,
    )

    assert LOCAL_LLM_RESERVED_EXTRA_BODY_KEYS == LOCAL_OPENAI_RESERVED_EXTRA_BODY_KEYS
    assert LOCAL_LLM_SENSITIVE_EXTRA_BODY_KEYS == LOCAL_OPENAI_SENSITIVE_EXTRA_BODY_KEYS


def test_from_dict_defaults_malformed_local_llm_settings() -> None:
    loaded = from_dict(
        {
            "provider": {"llm": "local_llm"},
            "local_llm": {
                "backend": "unknown",
                "base_url": "",
                "model": "",
                "extra_body": ["not", "a", "dict"],
            },
        }
    )

    assert loaded.local_llm.backend == LocalLLMBackend.OLLAMA
    assert loaded.local_llm.base_url == "http://127.0.0.1:11434/v1"
    assert loaded.local_llm.model == "llama3.1:8b"
    assert loaded.local_llm.extra_body == {"reasoning_effort": "none"}


@pytest.mark.parametrize(
    "value",
    [float("nan"), float("inf"), float("-inf")],
    ids=["nan", "infinity", "negative_infinity"],
)
def test_from_dict_defaults_local_llm_non_standard_json_constants_extra_body(value: float) -> None:
    loaded = from_dict(
        {
            "provider": {"llm": "local_llm"},
            "local_llm": {
                "backend": "ollama",
                "base_url": "http://127.0.0.1:11434/v1",
                "model": "llama3.1:8b",
                "extra_body": {"temperature": value},
            },
        }
    )

    assert loaded.local_llm.extra_body == {"reasoning_effort": "none"}


def test_schema21_migration_adds_local_llm_defaults(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    data = to_dict(AppSettings())
    data["settings_version"] = 21
    data.pop("local_llm", None)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    loaded = load_settings(path)
    persisted = legacy_projected_settings_file(path)

    assert loaded.settings_version == SETTINGS_SCHEMA_VERSION
    assert loaded.local_llm.extra_body == {"reasoning_effort": "none"}
    assert persisted["settings_version"] == SETTINGS_SCHEMA_VERSION
    assert persisted["local_llm"]["base_url"] == "http://127.0.0.1:11434/v1"


def test_schema22_repair_persists_malformed_local_llm(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    data = to_dict(AppSettings())
    data["settings_version"] = 22
    data["provider"]["llm"] = "local_llm"
    data["local_llm"] = {
        "backend": "bad",
        "base_url": "ftp://bad",
        "model": "",
        "extra_body": {"stream": True},
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    loaded = load_settings(path)
    persisted = legacy_projected_settings_file(path)

    assert loaded.local_llm.base_url == "http://127.0.0.1:11434/v1"
    assert loaded.local_llm.model == "llama3.1:8b"
    assert loaded.local_llm.extra_body == {"reasoning_effort": "none"}
    assert persisted["local_llm"]["extra_body"] == {"reasoning_effort": "none"}


@pytest.mark.parametrize(
    "value",
    [float("nan"), float("inf"), float("-inf")],
    ids=["nan", "infinity", "negative_infinity"],
)
def test_schema22_repair_persists_default_for_local_llm_non_standard_json_constants_extra_body(
    tmp_path: Path,
    value: float,
) -> None:
    path = tmp_path / "settings.json"
    data = to_dict(AppSettings())
    data["settings_version"] = 22
    data["provider"]["llm"] = "local_llm"
    data["local_llm"]["extra_body"] = {"temperature": value}
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    loaded = load_settings(path)
    persisted = legacy_projected_settings_file(path)

    assert loaded.local_llm.extra_body == {"reasoning_effort": "none"}
    assert persisted["local_llm"]["extra_body"] == {"reasoning_effort": "none"}


def test_schema22_repair_persists_missing_local_llm(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    data = to_dict(AppSettings())
    data["settings_version"] = 22
    data.pop("local_llm", None)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    loaded = load_settings(path)
    persisted = legacy_projected_settings_file(path)

    assert loaded.local_llm.backend == LocalLLMBackend.OLLAMA
    assert loaded.local_llm.base_url == "http://127.0.0.1:11434/v1"
    assert loaded.local_llm.model == "llama3.1:8b"
    assert loaded.local_llm.extra_body == {"reasoning_effort": "none"}
    assert persisted["local_llm"] == {
        "backend": "ollama",
        "base_url": "http://127.0.0.1:11434/v1",
        "model": "llama3.1:8b",
        "extra_body": {"reasoning_effort": "none"},
    }


def test_materialize_translation_settings_returns_mutated_settings() -> None:
    settings = AppSettings()
    settings.translation = TranslationSettings(
        model=TranslationModel.DEEPSEEK_V4_FLASH,
        connection=TranslationConnection.OFFICIAL_BYOK,
    )

    returned = materialize_translation_settings(settings)

    assert returned is settings
    assert settings.provider.llm == LLMProviderName.DEEPSEEK
    assert settings.deepseek.llm_model == DeepSeekLLMModel.DEEPSEEK_V4_FLASH


def test_legacy_deepseek_v4_pro_translation_migrates_to_v4_flash() -> None:
    assert supported_translation_connections(TranslationModel.DEEPSEEK_V4_FLASH) == (
        TranslationConnection.MANAGED,
        TranslationConnection.MANAGED_CHINA,
        TranslationConnection.OPENROUTER,
        TranslationConnection.OFFICIAL_BYOK,
    )

    data = to_dict(AppSettings())
    data["translation"] = {
        "model": "deepseek_v4_pro",
        "connection": TranslationConnection.OFFICIAL_BYOK.value,
        "connection_history": {
            "deepseek_v4_pro": TranslationConnection.OFFICIAL_BYOK.value,
        },
    }
    data["provider"]["llm"] = LLMProviderName.DEEPSEEK.value
    data["deepseek"] = {"llm_model": "deepseek-v4-pro"}

    loaded = from_dict(data)

    assert loaded.translation.model == TranslationModel.DEEPSEEK_V4_FLASH
    assert loaded.translation.connection == TranslationConnection.OFFICIAL_BYOK
    assert loaded.provider.llm == LLMProviderName.DEEPSEEK
    assert loaded.deepseek.llm_model == DeepSeekLLMModel.DEEPSEEK_V4_FLASH


def test_openrouter_alias_for_fields_does_not_expose_deepseek_v4_pro() -> None:
    assert getattr(OpenRouterLLMModel, "DEEPSEEK_V4_PRO", None) is None
    assert getattr(OpenRouterSelectionAlias, "DEEPSEEK_V4_PRO_BYOK", None) is None
    assert (
        openrouter_alias_for_fields(
            model="deepseek/deepseek-v4-pro",
            source=OpenRouterCredentialSource.BYOK.value,
        )
        is None
    )
    assert (
        openrouter_alias_for_fields(
            model="deepseek/deepseek-v4-pro",
            source=OpenRouterCredentialSource.MANAGED.value,
        )
        is None
    )
    assert get_openrouter_llm_profile("deepseek_v4_pro_byok") is None


def test_materialize_translation_settings_maps_deepseek_managed_china_to_deepseek_only_openrouter() -> (
    None
):
    settings = AppSettings()
    settings.translation = TranslationSettings(
        model=TranslationModel.DEEPSEEK_V4_FLASH,
        connection=TranslationConnection.MANAGED_CHINA,
        connection_history={
            TranslationModel.DEEPSEEK_V4_FLASH.value: TranslationConnection.MANAGED_CHINA,
        },
    )
    settings.openrouter.provider_routing = OpenRouterProviderRouting.DEFAULT

    returned = materialize_translation_settings(settings)

    assert returned is settings
    assert settings.provider.llm == LLMProviderName.OPENROUTER
    assert settings.openrouter.llm_model == OpenRouterLLMModel.DEEPSEEK_V4_FLASH
    assert settings.openrouter.selected_source == OpenRouterCredentialSource.MANAGED
    assert settings.openrouter.selection_alias == OpenRouterSelectionAlias.DEEPSEEK_V4_FLASH_MANAGED
    assert settings.openrouter.provider_routing == OpenRouterProviderRouting.DEEPSEEK_ONLY


def test_to_dict_roundtrips_deepseek_managed_china_provider_routing(tmp_path) -> None:
    path = tmp_path / "settings.json"
    settings = AppSettings()
    settings.translation = TranslationSettings(
        model=TranslationModel.DEEPSEEK_V4_FLASH,
        connection=TranslationConnection.MANAGED_CHINA,
        connection_history={
            TranslationModel.DEEPSEEK_V4_FLASH.value: TranslationConnection.MANAGED_CHINA,
        },
    )

    save_settings(path, settings)
    loaded = load_settings(path)
    persisted = legacy_projected_settings_file(path)

    assert loaded.translation.model == TranslationModel.DEEPSEEK_V4_FLASH
    assert loaded.translation.connection == TranslationConnection.MANAGED_CHINA
    assert loaded.openrouter.provider_routing == OpenRouterProviderRouting.DEEPSEEK_ONLY
    assert persisted["translation"]["connection"] == TranslationConnection.MANAGED_CHINA.value
    assert (
        persisted["openrouter"]["provider_routing"] == OpenRouterProviderRouting.DEEPSEEK_ONLY.value
    )


def test_app_settings_defaults_to_managed_openrouter_gemma_without_fallback() -> None:
    settings = AppSettings()

    assert settings.translation.model == TranslationModel.GEMMA4_26B_31B
    assert settings.translation.connection == TranslationConnection.MANAGED
    assert settings.provider.llm == LLMProviderName.OPENROUTER
    assert settings.openrouter.llm_model == OpenRouterLLMModel.GEMMA_4_26B_A4B_IT
    assert settings.openrouter.selected_source == OpenRouterCredentialSource.MANAGED
    assert settings.openrouter.selection_alias == OpenRouterSelectionAlias.GEMMA4_26B_31B_MANAGED
    assert settings.openrouter.fallback_selection_alias == OpenRouterFallbackSelectionAlias.NONE
    assert settings.translation.fallback == TranslationFallbackSettings()


def test_app_settings_accepts_deepseek_llm_provider_defaults() -> None:
    settings = AppSettings(
        provider=ProviderSettings(llm=LLMProviderName.DEEPSEEK),
        translation=TranslationSettings(
            model=TranslationModel.DEEPSEEK_V4_FLASH,
            connection=TranslationConnection.OFFICIAL_BYOK,
        ),
        deepseek=DeepSeekSettings(),
    )

    settings.validate()

    assert settings.deepseek.llm_model == DeepSeekLLMModel.DEEPSEEK_V4_FLASH
    assert to_dict(settings)["provider"]["llm"] == LLMProviderName.DEEPSEEK.value
    assert to_dict(settings)["deepseek"] == {"llm_model": DeepSeekLLMModel.DEEPSEEK_V4_FLASH.value}


def test_from_dict_preserves_deepseek_llm_provider_model_and_verification() -> None:
    data = to_dict(AppSettings())
    data.pop("translation", None)
    data["provider"]["llm"] = LLMProviderName.DEEPSEEK.value
    data["deepseek"] = {"llm_model": DeepSeekLLMModel.DEEPSEEK_V4_FLASH.value}
    data["api_key_verified"]["deepseek"] = True

    loaded = from_dict(data)

    assert loaded.provider.llm == LLMProviderName.DEEPSEEK
    assert loaded.deepseek.llm_model == DeepSeekLLMModel.DEEPSEEK_V4_FLASH
    assert loaded.api_key_verified.deepseek is True
    persisted = to_dict(loaded)
    assert persisted["provider"]["llm"] == LLMProviderName.DEEPSEEK.value
    assert persisted["deepseek"]["llm_model"] == DeepSeekLLMModel.DEEPSEEK_V4_FLASH.value
    assert persisted["api_key_verified"]["deepseek"] is True


def test_from_dict_backfills_missing_deepseek_settings_and_verification() -> None:
    data = to_dict(AppSettings())
    data.pop("deepseek", None)
    data["api_key_verified"].pop("deepseek", None)

    loaded = from_dict(data)

    assert loaded.deepseek.llm_model == DeepSeekLLMModel.DEEPSEEK_V4_FLASH
    assert loaded.api_key_verified.deepseek is False
    persisted = to_dict(loaded)
    assert persisted["deepseek"] == {"llm_model": DeepSeekLLMModel.DEEPSEEK_V4_FLASH.value}
    assert persisted["api_key_verified"]["deepseek"] is False


def test_gemini_openrouter_alias_and_routing_roundtrip() -> None:
    settings = AppSettings()
    settings.translation = TranslationSettings(
        model=TranslationModel.GEMINI_37_FLASH,
        connection=TranslationConnection.OPENROUTER,
        connection_history={
            TranslationModel.GEMINI_37_FLASH.value: TranslationConnection.OPENROUTER,
        },
    )

    persisted = to_dict(settings)
    loaded = from_dict(persisted)

    assert loaded.translation.model == TranslationModel.GEMINI_37_FLASH
    assert loaded.translation.connection == TranslationConnection.OPENROUTER
    assert loaded.provider.llm == LLMProviderName.OPENROUTER
    assert loaded.openrouter.llm_model == OpenRouterLLMModel.GEMINI_37_FLASH
    assert loaded.openrouter.selected_source == OpenRouterCredentialSource.BYOK
    assert loaded.openrouter.selection_alias == OpenRouterSelectionAlias.GEMINI37_FLASH_BYOK
    assert loaded.openrouter.provider_routing == OpenRouterProviderRouting.GOOGLE_GEMINI_LATENCY


def test_cerebras_settings_default_roundtrip_and_verification_state() -> None:
    raw = to_dict(AppSettings())
    raw.pop("cerebras", None)
    raw["api_key_verified"].pop("cerebras", None)

    loaded = from_dict(raw)
    persisted = to_dict(loaded)

    assert loaded.cerebras.llm_model == CerebrasLLMModel.GEMMA_4_31B
    assert loaded.api_key_verified.cerebras is False
    assert persisted["cerebras"] == {"llm_model": CerebrasLLMModel.GEMMA_4_31B.value}
    assert persisted["api_key_verified"]["cerebras"] is False

    persisted["provider"]["llm"] = LLMProviderName.CEREBRAS.value
    persisted.pop("translation", None)
    persisted["api_key_verified"]["cerebras"] = True
    loaded_verified = from_dict(persisted)
    assert loaded_verified.provider.llm == LLMProviderName.CEREBRAS
    assert loaded_verified.translation.model == TranslationModel.GEMMA4_31B
    assert loaded_verified.translation.connection == TranslationConnection.CEREBRAS
    assert loaded_verified.api_key_verified.cerebras is True


def test_legacy_cerebras_translation_selection_migrates_to_gemma31_connection() -> None:
    raw = to_dict(AppSettings())
    raw["provider"]["llm"] = LLMProviderName.CEREBRAS.value
    raw["translation"] = {
        "model": "gemma4_31b_cerebras",
        "connection": TranslationConnection.OFFICIAL_BYOK.value,
        "connection_history": {
            TranslationModel.GEMMA4_31B.value: TranslationConnection.OPENROUTER.value,
            "gemma4_31b_cerebras": TranslationConnection.OFFICIAL_BYOK.value,
        },
        "fallback": {
            "enabled": True,
            "model": "gemma4_31b_cerebras",
            "connection": TranslationConnection.OFFICIAL_BYOK.value,
        },
    }

    loaded = from_dict(raw)
    persisted = to_dict(loaded)

    assert loaded.translation.model == TranslationModel.GEMMA4_31B
    assert loaded.translation.connection == TranslationConnection.CEREBRAS
    assert loaded.translation.connection_history[TranslationModel.GEMMA4_31B.value] == (
        TranslationConnection.CEREBRAS
    )
    assert loaded.translation.fallback == TranslationFallbackSettings(
        enabled=True,
        model=TranslationModel.GEMMA4_31B,
        connection=TranslationConnection.CEREBRAS,
    )
    assert loaded.provider.llm == LLMProviderName.CEREBRAS
    assert "gemma4_31b_cerebras" not in json.dumps(persisted)


def test_cerebras_api_key_is_not_serialized_in_settings() -> None:
    raw = to_dict(AppSettings())
    raw["cerebras"]["api_key"] = "do-not-persist"
    raw["cerebras_api_key"] = "do-not-persist"

    persisted = to_dict(from_dict(raw))

    assert "cerebras_api_key" not in persisted
    assert "api_key" not in persisted["cerebras"]


def test_openrouter_fallback_aliases_include_curated_openrouter_models() -> None:
    deepseek_fallback = getattr(OpenRouterFallbackSelectionAlias, "DEEPSEEK_V4_FLASH", None)
    deepseek_china_fallback = getattr(
        OpenRouterFallbackSelectionAlias, "DEEPSEEK_V4_FLASH_CHINA", None
    )
    assert deepseek_fallback is not None
    assert deepseek_china_fallback is not None

    assert OPENROUTER_FALLBACK_SELECTION_ALIASES == (
        OpenRouterFallbackSelectionAlias.NONE.value,
        deepseek_fallback.value,
        deepseek_china_fallback.value,
        OpenRouterFallbackSelectionAlias.GEMMA4_26B_31B.value,
        OpenRouterFallbackSelectionAlias.GEMMA4_31B.value,
    )
    assert OpenRouterFallbackSelectionAlias.QWEN35_FLASH.value not in (
        OPENROUTER_FALLBACK_SELECTION_ALIASES
    )


def test_from_dict_preserves_cloud_qwen_asr_provider_value() -> None:
    data = to_dict(AppSettings())
    data["provider"]["stt"] = STTProviderName.QWEN_ASR.value

    loaded = from_dict(data)

    assert loaded.provider.stt == STTProviderName.QWEN_ASR


@pytest.mark.parametrize("legacy_rate", [8000, "8000"])
def test_from_dict_normalizes_legacy_8khz_audio_to_16khz(legacy_rate: int | str) -> None:
    data = to_dict(AppSettings())
    data["audio"]["internal_sample_rate_hz"] = legacy_rate

    loaded = from_dict(data)

    assert loaded.audio.internal_sample_rate_hz == 16000
    assert to_dict(loaded)["audio"]["internal_sample_rate_hz"] == 16000


def test_qwen_asr_endpoint_is_normalized_from_region_on_load_and_save() -> None:
    data = to_dict(AppSettings())
    data["qwen"]["region"] = QwenRegion.BEIJING.value
    data["qwen_asr_stt"]["endpoint"] = "wss://legacy.example.invalid/realtime"

    loaded = from_dict(data)
    persisted = to_dict(loaded)

    assert loaded.qwen_asr_stt.endpoint == loaded.qwen.get_asr_endpoint()
    assert persisted["qwen_asr_stt"]["endpoint"] == loaded.qwen.get_asr_endpoint()


def test_load_settings_infers_missing_qwen_region_from_legacy_asr_endpoint(tmp_path) -> None:
    path = tmp_path / "settings.json"
    legacy = to_dict(AppSettings())
    legacy["qwen"] = {
        "llm_model": QwenLLMModel.QWEN_35_PLUS.value,
    }
    legacy["qwen_asr_stt"]["endpoint"] = "wss://dashscope-intl.aliyuncs.com/api-ws/v1/realtime"
    path.write_text(json.dumps(legacy, ensure_ascii=False, indent=2), encoding="utf-8")

    loaded = load_settings(path)
    persisted = legacy_projected_settings_file(path)

    assert loaded.qwen.region == QwenRegion.SINGAPORE
    assert loaded.qwen_asr_stt.endpoint == loaded.qwen.get_asr_endpoint()
    assert persisted["qwen"]["region"] == QwenRegion.SINGAPORE.value
    assert persisted["qwen_asr_stt"]["endpoint"] == loaded.qwen.get_asr_endpoint()


def test_from_dict_defaults_missing_stt_provider_to_local_cpu_auto() -> None:
    data = to_dict(AppSettings())
    data["provider"].pop("stt", None)

    loaded = from_dict(data)

    assert loaded.provider.stt == STTProviderName.LOCAL_CPU_AUTO


def test_from_dict_maps_legacy_alibaba_provider_to_qwen_asr() -> None:
    data = to_dict(AppSettings())
    data["provider"]["stt"] = "alibaba"

    loaded = from_dict(data)

    assert loaded.provider.stt == STTProviderName.QWEN_ASR


def test_from_dict_falls_back_to_deepgram_for_invalid_persisted_stt_provider() -> None:
    data = to_dict(AppSettings())
    data["provider"]["stt"] = "broken-provider"

    loaded = from_dict(data)

    assert loaded.provider.stt == STTProviderName.DEEPGRAM


def test_peer_stt_provider_roundtrips_through_settings_dict() -> None:
    settings = AppSettings()
    settings.provider.peer_stt = STTProviderName.SONIOX
    settings.peer_qwen_asr_stt.region = QwenRegion.SINGAPORE

    persisted = to_dict(settings)
    reloaded = from_dict(persisted)

    assert reloaded.provider.peer_stt == STTProviderName.SONIOX
    assert reloaded.peer_qwen_asr_stt.region is None
    assert "peer_deepgram_stt" not in persisted
    assert "peer_qwen_asr_stt" not in persisted
    assert "peer_soniox_stt" not in persisted
    assert "cooldown_s" not in persisted["osc"]
    assert "ttl_s" not in persisted["osc"]


def test_to_dict_persists_peer_local_qwen_without_rewriting_runtime_settings() -> None:
    settings = AppSettings()
    settings.provider.peer_stt = STTProviderName.LOCAL_QWEN

    persisted = to_dict(settings)

    assert persisted["provider"]["peer_stt"] == STTProviderName.LOCAL_QWEN.value
    assert settings.provider.peer_stt == STTProviderName.LOCAL_QWEN


def test_from_dict_defaults_missing_peer_stt_provider_to_deepgram() -> None:
    data = to_dict(AppSettings())
    data["provider"].pop("peer_stt", None)

    loaded = from_dict(data)

    assert loaded.provider.peer_stt == STTProviderName.DEEPGRAM


def test_from_dict_falls_back_to_deepgram_for_invalid_peer_stt_provider() -> None:
    data = to_dict(AppSettings())
    data["provider"]["peer_stt"] = "broken-peer-provider"

    loaded = from_dict(data)

    assert loaded.provider.peer_stt == STTProviderName.DEEPGRAM


def test_from_dict_restores_local_qwen_peer_stt_provider() -> None:
    data = to_dict(AppSettings())
    data["provider"]["peer_stt"] = STTProviderName.LOCAL_QWEN.value

    loaded = from_dict(data)

    assert loaded.provider.peer_stt == STTProviderName.LOCAL_QWEN


def test_from_dict_preserves_legacy_malformed_provider_fallback_behavior() -> None:
    loaded = from_dict({"provider": "legacy-string"})

    assert loaded.provider.stt == STTProviderName.DEEPGRAM
    assert loaded.provider.peer_stt == STTProviderName.DEEPGRAM


def test_from_dict_defaults_missing_llm_provider_to_legacy_gemini_and_inactive_openrouter() -> None:
    loaded = from_dict({"provider": {"stt": STTProviderName.LOCAL_QWEN.value}})

    assert loaded.provider.llm == LLMProviderName.GEMINI
    assert loaded.openrouter.selected_source == OpenRouterCredentialSource.NONE
    assert loaded.openrouter.selection_alias is None


def test_load_settings_backfills_peer_provider_defaults_without_copying_self_values(
    tmp_path,
) -> None:
    path = tmp_path / "settings.json"
    legacy = to_dict(AppSettings())
    legacy["provider"].pop("peer_stt", None)
    legacy.pop("peer_deepgram_stt", None)
    legacy.pop("peer_qwen_asr_stt", None)
    legacy.pop("peer_soniox_stt", None)
    legacy["deepgram_stt"]["model"] = "nova-3-medical"
    path.write_text(json.dumps(legacy, ensure_ascii=False, indent=2), encoding="utf-8")

    loaded = load_settings(path)
    persisted = legacy_projected_settings_file(path)

    assert loaded.provider.peer_stt == STTProviderName.DEEPGRAM
    assert loaded.deepgram_stt.model == "nova-3-medical"
    assert persisted["provider"]["peer_stt"] == STTProviderName.DEEPGRAM.value
    assert "peer_deepgram_stt" not in persisted


def test_load_settings_migrates_peer_local_qwen_to_cpu_auto(tmp_path) -> None:
    path = tmp_path / "settings.json"
    legacy = to_dict(AppSettings())
    legacy["provider"]["peer_stt"] = STTProviderName.LOCAL_QWEN.value
    path.write_text(json.dumps(legacy, ensure_ascii=False, indent=2), encoding="utf-8")

    loaded = load_settings(path)
    persisted = legacy_projected_settings_file(path)

    assert loaded.provider.peer_stt == STTProviderName.LOCAL_CPU_AUTO
    assert persisted["provider"]["peer_stt"] == STTProviderName.LOCAL_CPU_AUTO.value


def test_from_dict_recovers_malformed_peer_soniox_override_values() -> None:
    data = to_dict(AppSettings())
    data["peer_soniox_stt"] = {
        "model": "",
        "endpoint": "",
        "keepalive_interval_s": "broken",
        "trailing_silence_ms": "broken",
    }

    loaded = from_dict(data)

    assert loaded.peer_soniox_stt.model is None
    assert loaded.peer_soniox_stt.endpoint is None
    assert loaded.peer_soniox_stt.keepalive_interval_s is None
    assert loaded.peer_soniox_stt.trailing_silence_ms is None


def test_load_settings_backfills_v4_peer_blocks_from_schema3_fixture(tmp_path) -> None:
    path = tmp_path / "settings.json"
    legacy = {
        "settings_version": 3,
        "provider": {
            "stt": STTProviderName.LOCAL_QWEN.value,
            "llm": LLMProviderName.GEMINI.value,
            "peer_soniox_stt": "broken",
        },
        "languages": {
            "source_language": "ko",
            "target_language": "en",
            "peer_source_language": "",
            "peer_target_language": "",
            "recent_source_languages": ["en", "zh-CN", "ja"],
            "recent_target_languages": ["en", "zh-CN", "ja"],
        },
        "audio": {
            "internal_sample_rate_hz": 16000,
            "internal_channels": 1,
            "ring_buffer_ms": 500,
            "input_host_api": "Windows DirectSound",
            "input_device": "",
        },
        "desktop_audio": {
            "output_device": "",
            "vad_speech_threshold": 0.5,
            "vad_hangover_ms": 900,
            "vad_pre_roll_ms": 500,
        },
        "overlay_calibration": AppSettings().overlay_calibration.to_dict(),
        "stt": {
            "drain_timeout_s": 2.0,
            "vad_speech_threshold": 0.35,
            "low_latency_mode": True,
            "low_latency_vad_hangover_ms": 600,
            "low_latency_merge_gap_ms": 600,
            "low_latency_spec_retry_max": 10,
            "custom_vocabulary_enabled": True,
            "custom_terms": {
                "ko": ["아이리", "시나노"],
                "en": ["airi", "shinano"],
                "zh-CN": ["airi", "shinano"],
                "ja": ["airi", "shinano"],
            },
        },
        "deepgram_stt": {"model": "nova-3"},
        "qwen_asr_stt": {
            "model": "qwen3-asr-flash-realtime",
            "endpoint": "wss://dashscope-intl.aliyuncs.com/api-ws/v1/realtime",
        },
        "soniox_stt": {
            "model": "stt-rt-v3",
            "endpoint": "wss://stt-rt.soniox.com/transcribe-websocket",
            "keepalive_interval_s": 10.0,
            "trailing_silence_ms": 100,
        },
        "peer_deepgram_stt": "broken",
        "peer_qwen_asr_stt": ["broken"],
        "peer_soniox_stt": {"model": "", "endpoint": ""},
        "gemini": {"llm_model": GeminiLLMModel.GEMINI_31_FLASH_LITE.value},
        "qwen": {
            "region": QwenRegion.BEIJING.value,
            "llm_model": QwenLLMModel.QWEN_35_PLUS.value,
        },
        "llm": {"concurrency_limit": 5},
        "osc": {
            "host": "127.0.0.1",
            "port": 9000,
            "chatbox_address": "/chatbox/input",
            "chatbox_send": True,
            "chatbox_clear": False,
            "chatbox_max_chars": 144,
            "cooldown_s": 1.5,
            "ttl_s": 7.0,
            "vrc_mic_intercept": False,
            "chatbox_include_source": True,
        },
        "secrets": {
            "backend": "keyring",
            "encrypted_file_path": "secrets.json",
        },
        "ui": {
            "locale": "en",
            "show_overlay_translation": True,
            "show_overlay_peer_original": True,
            "peer_translation_enabled": False,
            "integrated_context_enabled": False,
            "integrated_context_bootstrapped": False,
        },
        "api_key_verified": {
            "deepgram": False,
            "soniox": False,
            "google": False,
            "alibaba_beijing": False,
            "alibaba_singapore": False,
        },
        "system_prompt": "",
        "system_prompts": {},
    }
    path.write_text(json.dumps(legacy, ensure_ascii=False, indent=2), encoding="utf-8")

    loaded = load_settings(path)
    persisted = legacy_projected_settings_file(path)

    assert loaded.settings_version == SETTINGS_SCHEMA_VERSION
    assert loaded.provider.peer_stt == STTProviderName.DEEPGRAM
    assert loaded.peer_qwen_asr_stt.model is None
    assert loaded.peer_qwen_asr_stt.region is None
    assert loaded.peer_soniox_stt.model is None
    assert loaded.peer_soniox_stt.endpoint is None
    assert loaded.peer_soniox_stt.keepalive_interval_s is None
    assert loaded.peer_soniox_stt.trailing_silence_ms is None
    assert persisted["settings_version"] == SETTINGS_SCHEMA_VERSION
    assert persisted["provider"]["peer_stt"] == STTProviderName.DEEPGRAM.value
    assert "peer_deepgram_stt" not in persisted
    assert "peer_qwen_asr_stt" not in persisted
    assert "peer_soniox_stt" not in persisted


def test_from_dict_ignores_legacy_peer_deepgram_override_block() -> None:
    data = to_dict(AppSettings())
    data["peer_deepgram_stt"] = {"model": "nova-3-general"}

    loaded = from_dict(data)
    persisted = to_dict(loaded)

    assert loaded.deepgram_stt.model == AppSettings().deepgram_stt.model
    assert "peer_deepgram_stt" not in persisted


def test_app_settings_validate_checks_peer_provider_blocks() -> None:
    settings = AppSettings()
    settings.peer_soniox_stt.keepalive_interval_s = -1.0

    with pytest.raises(ValueError, match="peer soniox keepalive override must be > 0"):
        settings.validate()


def test_from_dict_recovers_non_dict_provider_payload_to_deepgram() -> None:
    data = to_dict(AppSettings())
    data.pop("translation", None)
    data["provider"] = "broken"

    loaded = from_dict(data)

    assert loaded.provider.stt == STTProviderName.DEEPGRAM
    assert loaded.provider.llm == LLMProviderName.GEMINI


def test_load_settings_persists_invalid_stt_provider_as_deepgram(tmp_path) -> None:
    path = tmp_path / "settings.json"
    legacy = to_dict(AppSettings())
    legacy["provider"]["stt"] = "broken-provider"
    path.write_text(json.dumps(legacy, ensure_ascii=False, indent=2), encoding="utf-8")

    loaded = load_settings(path)

    assert loaded.provider.stt == STTProviderName.DEEPGRAM
    persisted = legacy_projected_settings_file(path)
    assert persisted["provider"]["stt"] == STTProviderName.DEEPGRAM.value


def test_load_settings_persists_non_dict_provider_payload_as_deepgram(tmp_path) -> None:
    path = tmp_path / "settings.json"
    legacy = to_dict(AppSettings())
    legacy.pop("translation", None)
    legacy["provider"] = []
    path.write_text(json.dumps(legacy, ensure_ascii=False, indent=2), encoding="utf-8")

    loaded = load_settings(path)

    assert loaded.provider.stt == STTProviderName.DEEPGRAM
    persisted = legacy_projected_settings_file(path)
    assert persisted["provider"]["stt"] == STTProviderName.DEEPGRAM.value
    assert persisted["provider"]["llm"] == LLMProviderName.GEMINI.value


def test_load_settings_migrates_legacy_concurrency_limit_and_persists(tmp_path):
    path = tmp_path / "settings.json"
    legacy = to_dict(AppSettings())
    legacy.pop("settings_version", None)
    legacy["llm"]["concurrency_limit"] = 1
    path.write_text(json.dumps(legacy, ensure_ascii=False, indent=2), encoding="utf-8")

    loaded = load_settings(path)
    assert loaded.settings_version == SETTINGS_SCHEMA_VERSION
    assert loaded.llm.concurrency_limit == 5

    persisted = legacy_projected_settings_file(path)
    assert persisted["settings_version"] == SETTINGS_SCHEMA_VERSION
    assert persisted["llm"]["concurrency_limit"] == 5


def test_load_settings_migrates_previous_default_concurrency_limit_and_persists(tmp_path):
    path = tmp_path / "settings.json"
    legacy = to_dict(AppSettings())
    legacy["settings_version"] = 5
    legacy["llm"]["concurrency_limit"] = 2
    path.write_text(json.dumps(legacy, ensure_ascii=False, indent=2), encoding="utf-8")

    loaded = load_settings(path)
    assert loaded.settings_version == SETTINGS_SCHEMA_VERSION
    assert loaded.llm.concurrency_limit == 5

    persisted = legacy_projected_settings_file(path)
    assert persisted["settings_version"] == SETTINGS_SCHEMA_VERSION
    assert persisted["llm"]["concurrency_limit"] == 5


@pytest.mark.parametrize("legacy_rate", [8000, "8000"])
def test_migrate_settings_dict_forces_legacy_8khz_audio_to_16khz(
    legacy_rate: int | str,
) -> None:
    legacy = to_dict(AppSettings())
    legacy["settings_version"] = SETTINGS_SCHEMA_VERSION - 1
    legacy["audio"]["internal_sample_rate_hz"] = legacy_rate

    migrated, changed = _migrate_settings_dict(legacy)

    assert changed is True
    assert migrated["settings_version"] == SETTINGS_SCHEMA_VERSION
    assert migrated["audio"]["internal_sample_rate_hz"] == 16000


@pytest.mark.parametrize("legacy_rate", [8000, "8000"])
def test_load_settings_rewrites_migrated_8khz_audio_via_normal_save_path(
    tmp_path, monkeypatch: pytest.MonkeyPatch, legacy_rate: int | str
) -> None:
    path = tmp_path / "settings.json"
    legacy = to_dict(AppSettings())
    legacy["settings_version"] = SETTINGS_SCHEMA_VERSION - 1
    legacy["audio"]["internal_sample_rate_hz"] = legacy_rate
    path.write_text(json.dumps(legacy, ensure_ascii=False, indent=2), encoding="utf-8")

    replace_calls: list[tuple[str, str]] = []
    path_type = type(path)
    original_replace = path_type.replace

    def recording_replace(self: Path, target: Path) -> Path:
        replace_calls.append((self.name, Path(target).name))
        return original_replace(self, target)

    monkeypatch.setattr(path_type, "replace", recording_replace)

    loaded = load_settings(path)

    assert loaded.audio.internal_sample_rate_hz == 16000
    assert replace_calls == [("settings.json.tmp", "settings.json")]

    persisted = legacy_projected_settings_file(path)
    assert persisted["settings_version"] == SETTINGS_SCHEMA_VERSION
    assert persisted["audio"]["internal_sample_rate_hz"] == 16000


def test_load_settings_migration_preserves_custom_concurrency_limit(tmp_path):
    path = tmp_path / "settings.json"
    legacy = to_dict(AppSettings())
    legacy.pop("settings_version", None)
    legacy["llm"]["concurrency_limit"] = 3
    path.write_text(json.dumps(legacy, ensure_ascii=False, indent=2), encoding="utf-8")

    loaded = load_settings(path)
    assert loaded.settings_version == SETTINGS_SCHEMA_VERSION
    assert loaded.llm.concurrency_limit == 3

    persisted = legacy_projected_settings_file(path)
    assert persisted["settings_version"] == SETTINGS_SCHEMA_VERSION
    assert persisted["llm"]["concurrency_limit"] == 3


def test_qwen_llm_model_roundtrip(tmp_path):
    path = tmp_path / "settings.json"
    settings = AppSettings()
    settings.qwen.llm_model = QwenLLMModel.QWEN_35_PLUS
    save_settings(path, settings)

    loaded = load_settings(path)
    assert loaded.qwen.llm_model == QwenLLMModel.QWEN_35_PLUS

    persisted = legacy_projected_settings_file(path)
    assert persisted["qwen"]["llm_model"] == "qwen3.5-plus"


def test_gemini_llm_model_roundtrip(tmp_path):
    path = tmp_path / "settings.json"
    settings = AppSettings()
    settings.gemini.llm_model = GeminiLLMModel.GEMINI_31_FLASH_LITE
    save_settings(path, settings)

    loaded = load_settings(path)
    assert loaded.gemini.llm_model == GeminiLLMModel.GEMINI_31_FLASH_LITE

    persisted = legacy_projected_settings_file(path)
    assert persisted["gemini"]["llm_model"] == "gemini-3.1-flash-lite"


def test_load_settings_migrates_legacy_qwen_mt_flash_model(tmp_path):
    path = tmp_path / "settings.json"
    legacy = to_dict(AppSettings())
    legacy["qwen"]["llm_model"] = "qwen-mt-flash"
    path.write_text(json.dumps(legacy, ensure_ascii=False, indent=2), encoding="utf-8")

    loaded = load_settings(path)
    assert loaded.qwen.llm_model == QwenLLMModel.QWEN_35_PLUS

    persisted = legacy_projected_settings_file(path)
    assert persisted["qwen"]["llm_model"] == "qwen3.5-plus"


def test_load_settings_migrates_legacy_invalid_gemini_model(tmp_path):
    path = tmp_path / "settings.json"
    legacy = to_dict(AppSettings())
    legacy["gemini"]["llm_model"] = "gemini-legacy-foo"
    path.write_text(json.dumps(legacy, ensure_ascii=False, indent=2), encoding="utf-8")

    loaded = load_settings(path)
    assert loaded.gemini.llm_model == GeminiLLMModel.GEMINI_31_FLASH_LITE

    persisted = legacy_projected_settings_file(path)
    assert persisted["gemini"]["llm_model"] == "gemini-3.1-flash-lite"


def test_load_settings_migrates_preview_gemini_flash_lite_to_ga(tmp_path):
    path = tmp_path / "settings.json"
    legacy = to_dict(AppSettings())
    legacy["gemini"]["llm_model"] = "gemini-3.1-flash-lite-preview"
    path.write_text(json.dumps(legacy, ensure_ascii=False, indent=2), encoding="utf-8")

    loaded = load_settings(path)
    assert loaded.gemini.llm_model == GeminiLLMModel.GEMINI_31_FLASH_LITE

    persisted = legacy_projected_settings_file(path)
    assert persisted["gemini"]["llm_model"] == "gemini-3.1-flash-lite"


def test_load_settings_migrates_gemini_3_flash_preview_to_3_7_flash(tmp_path):
    path = tmp_path / "settings.json"
    legacy = to_dict(AppSettings())
    legacy["gemini"]["llm_model"] = "gemini-3-flash-preview"
    legacy["translation"]["model"] = "gemini3_flash"
    legacy["translation"]["connection_history"] = {
        "gemini3_flash": "official_byok",
    }
    path.write_text(json.dumps(legacy, ensure_ascii=False, indent=2), encoding="utf-8")

    loaded = load_settings(path)
    assert loaded.gemini.llm_model == GeminiLLMModel.GEMINI_37_FLASH
    assert loaded.translation.model == TranslationModel.GEMINI_37_FLASH
    assert loaded.translation.connection_history.get("gemini37_flash") == (
        TranslationConnection.OFFICIAL_BYOK
    )

    persisted = legacy_projected_settings_file(path)
    assert persisted["gemini"]["llm_model"] == "gemini-3.7-flash"
    assert persisted["translation"]["model"] == "gemini37_flash"
    assert "gemini3_flash" not in persisted["translation"]["connection_history"]


def test_from_dict_defaults_missing_gemini_model_to_flash_lite():
    data = to_dict(AppSettings())
    data["gemini"] = {}

    loaded = from_dict(data)
    assert loaded.gemini.llm_model == GeminiLLMModel.GEMINI_31_FLASH_LITE


def test_app_settings_defaults_vrc_mic_sync_to_off():
    settings = AppSettings()

    assert settings.osc.vrc_mic_intercept is False


def test_from_dict_defaults_missing_vrc_mic_sync_to_off():
    data = to_dict(AppSettings())
    data.setdefault("osc", {}).pop("vrc_mic_intercept", None)

    loaded = from_dict(data)

    assert loaded.osc.vrc_mic_intercept is False


def test_overlay_display_preferences_roundtrip(tmp_path):
    path = tmp_path / "settings.json"
    settings = AppSettings()
    settings.overlay.show_translation = False
    settings.overlay.show_peer_original = False
    save_settings(path, settings)

    loaded = load_settings(path)

    assert loaded.overlay.show_translation is False
    assert loaded.overlay.show_peer_original is False

    persisted = legacy_projected_settings_file(path)
    assert persisted["overlay"]["show_translation"] is False
    assert persisted["overlay"]["show_peer_original"] is False
    assert "show_overlay_translation" not in persisted["ui"]
    assert "show_overlay_peer_original" not in persisted["ui"]


def test_from_dict_defaults_missing_overlay_display_preferences_to_true():
    data = to_dict(AppSettings())
    data.pop("overlay", None)
    data.setdefault("ui", {}).pop("show_overlay_translation", None)
    data["ui"].pop("show_overlay_peer_original", None)

    loaded = from_dict(data)

    assert loaded.overlay.show_translation is True
    assert loaded.overlay.show_peer_original is True


def test_load_settings_backfills_missing_overlay_display_preferences(tmp_path):
    path = tmp_path / "settings.json"
    legacy = {
        "ui": {"overlay_enabled": True},
        "overlay": {
            "calibration": AppSettings().overlay_calibration.to_dict(),
        },
    }
    path.write_text(json.dumps(legacy, ensure_ascii=False, indent=2), encoding="utf-8")

    loaded = load_settings(path)

    assert loaded.overlay.show_translation is True
    assert loaded.overlay.show_peer_original is True

    persisted = legacy_projected_settings_file(path)
    assert persisted["overlay"]["show_translation"] is True
    assert persisted["overlay"]["show_peer_original"] is True


def test_load_settings_backfills_overlay_display_preferences_when_overlay_section_missing(tmp_path):
    path = tmp_path / "settings.json"
    legacy = to_dict(AppSettings())
    legacy.pop("overlay", None)
    legacy.setdefault("ui", {})["overlay_enabled"] = True
    legacy["ui"].pop("show_overlay_translation", None)
    legacy["ui"].pop("show_overlay_peer_original", None)
    path.write_text(json.dumps(legacy, ensure_ascii=False, indent=2), encoding="utf-8")

    loaded = load_settings(path)

    assert loaded.overlay.show_translation is True
    assert loaded.overlay.show_peer_original is True

    persisted = legacy_projected_settings_file(path)
    assert persisted["overlay"]["show_translation"] is True
    assert persisted["overlay"]["show_peer_original"] is True
    assert "overlay_enabled" not in persisted["ui"]


def test_stt_custom_vocabulary_roundtrip(tmp_path):
    path = tmp_path / "settings.json"
    settings = AppSettings()
    settings.stt.custom_vocabulary_enabled = True
    settings.stt.custom_terms = {
        "ko": [" Puripuly ", "VRChat", "Puripuly", ""],
        "en": ["OSC", " Soniox "],
    }

    save_settings(path, settings)

    loaded = load_settings(path)

    assert loaded.stt.custom_vocabulary_enabled is True
    assert loaded.stt.custom_terms == {
        "ko": ["Puripuly", "VRChat"],
        "en": ["OSC", "Soniox"],
    }

    persisted = legacy_projected_settings_file(path)
    assert persisted["stt"]["custom_vocabulary_enabled"] is True
    assert persisted["stt"]["custom_terms"] == {
        "ko": ["Puripuly", "VRChat"],
        "en": ["OSC", "Soniox"],
    }


def test_stt_custom_vocabulary_missing_keys_default():
    data = to_dict(AppSettings())
    data.setdefault("stt", {}).pop("custom_vocabulary_enabled", None)
    data["stt"].pop("custom_terms", None)

    loaded = from_dict(data)

    assert loaded.stt.custom_vocabulary_enabled is False
    assert loaded.stt.custom_terms == {}


def test_load_settings_backfills_empty_custom_vocabulary_defaults(tmp_path):
    path = tmp_path / "settings.json"
    legacy = to_dict(AppSettings())
    legacy.setdefault("stt", {}).pop("custom_vocabulary_enabled", None)
    legacy["stt"].pop("custom_terms", None)
    path.write_text(json.dumps(legacy, ensure_ascii=False, indent=2), encoding="utf-8")

    loaded = load_settings(path)

    assert loaded.stt.custom_vocabulary_enabled is False
    assert loaded.stt.custom_terms == {}

    persisted = legacy_projected_settings_file(path)
    assert persisted["stt"]["custom_vocabulary_enabled"] is False
    assert persisted["stt"]["custom_terms"] == {}


@pytest.mark.parametrize(
    ("custom_terms", "message"),
    [
        (["Puripuly"], "custom_terms must be a dict[str, list[str]]"),
        ({1: ["Puripuly"]}, "custom_terms keys must be strings"),
        ({"ko": "Puripuly"}, "custom_terms values must be lists of strings"),
        ({"ko": ["Puripuly", 1]}, "custom_terms values must be lists of strings"),
    ],
)
def test_stt_custom_vocabulary_rejects_malformed_shapes(custom_terms, message):
    data = to_dict(AppSettings())
    data.setdefault("stt", {})["custom_terms"] = custom_terms

    with pytest.raises(ValueError, match=re.escape(message)):
        from_dict(data)


def test_stt_custom_vocabulary_preserves_unrelated_language_buckets(tmp_path):
    path = tmp_path / "settings.json"
    settings = AppSettings()
    settings.stt.custom_terms = {
        "ko": ["Puripuly"],
        "zh-CN": ["Qwen"],
        "en": ["OSC"],
    }

    save_settings(path, settings)

    loaded = load_settings(path)
    loaded.stt.custom_terms["ko"] = ["Puripuly", "VRChat"]
    save_settings(path, loaded)

    reloaded = load_settings(path)

    assert reloaded.stt.custom_terms == {
        "ko": ["Puripuly", "VRChat"],
        "zh-CN": ["Qwen"],
        "en": ["OSC"],
    }


def test_stt_custom_vocabulary_roundtrip_caps_terms_to_100(tmp_path):
    path = tmp_path / "settings.json"
    settings = AppSettings()
    settings.stt.custom_vocabulary_enabled = True
    settings.stt.custom_terms = {"ko": [f"term-{i:03d}" for i in range(120)]}

    save_settings(path, settings)

    loaded = load_settings(path)

    assert len(loaded.stt.custom_terms["ko"]) == 100
    assert loaded.stt.custom_terms["ko"][0] == "term-000"
    assert loaded.stt.custom_terms["ko"][-1] == "term-099"


def test_system_prompts_are_not_persisted(tmp_path):
    path = tmp_path / "settings.json"
    settings = AppSettings()
    settings.system_prompts = {
        "gemini": "gemini prompt",
        "openrouter": "openrouter prompt",
        "qwen": "qwen prompt",
    }
    settings.provider.llm = LLMProviderName.QWEN
    settings.translation = TranslationSettings(
        model=TranslationModel.QWEN_35_PLUS,
        connection=TranslationConnection.OFFICIAL_BYOK,
        connection_history={
            TranslationModel.QWEN_35_PLUS.value: TranslationConnection.OFFICIAL_BYOK,
        },
    )
    settings.system_prompt = "qwen prompt"
    save_settings(path, settings)

    persisted = legacy_projected_settings_file(path)
    loaded = load_settings(path)

    assert "system_prompts" not in persisted
    assert loaded.system_prompts == {}
    assert loaded.system_prompt == "qwen prompt"


def test_openrouter_settings_roundtrip(tmp_path):
    path = tmp_path / "settings.json"
    settings = AppSettings(
        provider=ProviderSettings(llm=LLMProviderName.OPENROUTER),
        translation=TranslationSettings(
            model=TranslationModel.GEMMA4,
            connection=TranslationConnection.OPENROUTER,
            connection_history={
                TranslationModel.GEMMA4.value: TranslationConnection.OPENROUTER,
            },
        ),
        openrouter=OpenRouterSettings(
            llm_model=OpenRouterLLMModel.GEMMA_4_26B_A4B_IT,
            routing_mode=OpenRouterRoutingMode.LATENCY,
            selected_source=OpenRouterCredentialSource.BYOK,
            selection_alias=OpenRouterSelectionAlias.GEMMA4_BYOK,
        ),
    )
    save_settings(path, settings)

    loaded = load_settings(path)

    assert loaded.provider.llm == LLMProviderName.OPENROUTER
    assert loaded.translation.model == TranslationModel.GEMMA4
    assert loaded.translation.connection == TranslationConnection.OPENROUTER
    assert loaded.openrouter.llm_model == OpenRouterLLMModel.GEMMA_4_26B_A4B_IT
    assert loaded.openrouter.routing_mode == OpenRouterRoutingMode.LATENCY
    assert loaded.openrouter.selected_source == OpenRouterCredentialSource.BYOK
    assert loaded.openrouter.selection_alias == OpenRouterSelectionAlias.GEMMA4_BYOK


def test_translation_settings_roundtrip_materializes_deepseek_openrouter_byok(tmp_path):
    path = tmp_path / "settings.json"
    settings = AppSettings()
    settings.translation = TranslationSettings(
        model=TranslationModel.DEEPSEEK_V4_FLASH,
        connection=TranslationConnection.OPENROUTER,
        connection_history={
            TranslationModel.DEEPSEEK_V4_FLASH.value: TranslationConnection.OPENROUTER,
        },
    )

    save_settings(path, settings)
    loaded = load_settings(path)
    persisted = legacy_projected_settings_file(path)

    assert persisted["translation"] == {
        "model": "deepseek_v4_flash",
        "connection": "openrouter",
        "connection_history": {"deepseek_v4_flash": "openrouter"},
        "fallback": MIGRATED_DEFAULT_TRANSLATION_FALLBACK_DICT,
        "gpu_device_id": "auto",
    }
    assert loaded.translation.model == TranslationModel.DEEPSEEK_V4_FLASH
    assert loaded.translation.connection == TranslationConnection.OPENROUTER
    assert loaded.provider.llm == LLMProviderName.OPENROUTER
    assert loaded.openrouter.llm_model == OpenRouterLLMModel.DEEPSEEK_V4_FLASH
    assert loaded.openrouter.selected_source == OpenRouterCredentialSource.BYOK
    assert loaded.openrouter.selection_alias == OpenRouterSelectionAlias.DEEPSEEK_V4_FLASH_BYOK


def test_translation_settings_roundtrip_materializes_deepseek_official_byok(tmp_path):
    path = tmp_path / "settings.json"
    settings = AppSettings()
    settings.translation = TranslationSettings(
        model=TranslationModel.DEEPSEEK_V4_FLASH,
        connection=TranslationConnection.OFFICIAL_BYOK,
        connection_history={
            TranslationModel.DEEPSEEK_V4_FLASH.value: TranslationConnection.OFFICIAL_BYOK,
        },
    )

    save_settings(path, settings)
    loaded = load_settings(path)

    assert loaded.translation.model == TranslationModel.DEEPSEEK_V4_FLASH
    assert loaded.translation.connection == TranslationConnection.OFFICIAL_BYOK
    assert loaded.provider.llm == LLMProviderName.DEEPSEEK
    assert loaded.deepseek.llm_model == DeepSeekLLMModel.DEEPSEEK_V4_FLASH


def test_to_dict_infers_default_translation_from_provider_only_qwen_plus() -> None:
    settings = AppSettings()
    settings.provider.llm = LLMProviderName.QWEN
    settings.qwen.llm_model = QwenLLMModel.QWEN_35_PLUS

    serialized = to_dict(settings)

    assert serialized["translation"]["model"] == TranslationModel.QWEN_35_PLUS.value
    assert serialized["translation"]["connection"] == TranslationConnection.OFFICIAL_BYOK.value
    assert serialized["provider"]["llm"] == LLMProviderName.QWEN.value
    assert serialized["qwen"]["llm_model"] == QwenLLMModel.QWEN_35_PLUS.value


def test_to_dict_explicit_translation_wins_over_conflicting_runtime_fields() -> None:
    settings = AppSettings()
    settings.translation = TranslationSettings(
        model=TranslationModel.DEEPSEEK_V4_FLASH,
        connection=TranslationConnection.OFFICIAL_BYOK,
        connection_history={
            TranslationModel.DEEPSEEK_V4_FLASH.value: TranslationConnection.OFFICIAL_BYOK,
        },
    )
    settings.provider.llm = LLMProviderName.QWEN
    settings.qwen.llm_model = QwenLLMModel.QWEN_35_PLUS

    serialized = to_dict(settings)

    assert serialized["translation"]["model"] == TranslationModel.DEEPSEEK_V4_FLASH.value
    assert serialized["translation"]["connection"] == TranslationConnection.OFFICIAL_BYOK.value
    assert serialized["provider"]["llm"] == LLMProviderName.DEEPSEEK.value
    assert serialized["deepseek"]["llm_model"] == DeepSeekLLMModel.DEEPSEEK_V4_FLASH.value


def test_load_settings_translation_block_wins_over_stale_openrouter_deepseek_fields(tmp_path):
    path = tmp_path / "settings.json"
    data = to_dict(AppSettings())
    data["translation"] = {
        "model": TranslationModel.GEMMA4.value,
        "connection": TranslationConnection.OPENROUTER.value,
        "connection_history": {
            TranslationModel.GEMMA4.value: TranslationConnection.OPENROUTER.value,
            TranslationModel.DEEPSEEK_V4_FLASH.value: TranslationConnection.MANAGED.value,
        },
    }
    data["provider"]["llm"] = LLMProviderName.OPENROUTER.value
    data["openrouter"]["llm_model"] = OpenRouterLLMModel.DEEPSEEK_V4_FLASH.value
    data["openrouter"]["selected_source"] = OpenRouterCredentialSource.MANAGED.value
    data["openrouter"]["selection_alias"] = OpenRouterSelectionAlias.DEEPSEEK_V4_FLASH_MANAGED.value
    path.write_text(json.dumps(data), encoding="utf-8")

    loaded = load_settings(path)
    persisted = legacy_projected_settings_file(path)

    assert loaded.translation.model == TranslationModel.GEMMA4_26B_31B
    assert loaded.translation.connection == TranslationConnection.OPENROUTER
    assert loaded.provider.llm == LLMProviderName.OPENROUTER
    assert loaded.openrouter.llm_model == OpenRouterLLMModel.GEMMA_4_26B_A4B_IT
    assert loaded.openrouter.selected_source == OpenRouterCredentialSource.BYOK
    assert loaded.openrouter.selection_alias == OpenRouterSelectionAlias.GEMMA4_26B_31B_BYOK
    assert persisted["translation"]["model"] == TranslationModel.GEMMA4_26B_31B.value
    assert persisted["translation"]["connection"] == TranslationConnection.OPENROUTER.value
    assert persisted["openrouter"]["llm_model"] == OpenRouterLLMModel.GEMMA_4_26B_A4B_IT.value
    assert persisted["openrouter"]["selected_source"] == OpenRouterCredentialSource.BYOK.value
    assert (
        persisted["openrouter"]["selection_alias"]
        == OpenRouterSelectionAlias.GEMMA4_26B_31B_BYOK.value
    )


def test_load_settings_translation_block_wins_for_gemini_over_stale_openrouter_fields(tmp_path):
    path = tmp_path / "settings.json"
    data = to_dict(AppSettings())
    data["translation"] = {
        "model": TranslationModel.GEMINI_31_FLASH_LITE.value,
        "connection": TranslationConnection.OFFICIAL_BYOK.value,
        "connection_history": {
            TranslationModel.GEMINI_31_FLASH_LITE.value: TranslationConnection.OFFICIAL_BYOK.value,
            TranslationModel.DEEPSEEK_V4_FLASH.value: TranslationConnection.MANAGED.value,
        },
    }
    data["provider"]["llm"] = LLMProviderName.OPENROUTER.value
    data["openrouter"]["llm_model"] = OpenRouterLLMModel.DEEPSEEK_V4_FLASH.value
    data["openrouter"]["selected_source"] = OpenRouterCredentialSource.MANAGED.value
    data["openrouter"]["selection_alias"] = OpenRouterSelectionAlias.DEEPSEEK_V4_FLASH_MANAGED.value
    path.write_text(json.dumps(data), encoding="utf-8")

    loaded = load_settings(path)
    persisted = legacy_projected_settings_file(path)

    assert loaded.translation.model == TranslationModel.GEMINI_31_FLASH_LITE
    assert loaded.translation.connection == TranslationConnection.OFFICIAL_BYOK
    assert loaded.provider.llm == LLMProviderName.GEMINI
    assert loaded.gemini.llm_model == GeminiLLMModel.GEMINI_31_FLASH_LITE
    assert persisted["translation"]["model"] == TranslationModel.GEMINI_31_FLASH_LITE.value
    assert persisted["translation"]["connection"] == TranslationConnection.OFFICIAL_BYOK.value
    assert persisted["provider"]["llm"] == LLMProviderName.GEMINI.value
    assert persisted["gemini"]["llm_model"] == GeminiLLMModel.GEMINI_31_FLASH_LITE.value


def test_from_dict_translation_block_wins_over_stale_openrouter_deepseek_fields() -> None:
    data = to_dict(AppSettings())
    data["translation"] = {
        "model": TranslationModel.GEMMA4.value,
        "connection": TranslationConnection.OPENROUTER.value,
        "connection_history": {
            TranslationModel.GEMMA4.value: TranslationConnection.OPENROUTER.value,
            TranslationModel.DEEPSEEK_V4_FLASH.value: TranslationConnection.MANAGED.value,
        },
    }
    data["provider"]["llm"] = LLMProviderName.OPENROUTER.value
    data["openrouter"]["llm_model"] = OpenRouterLLMModel.DEEPSEEK_V4_FLASH.value
    data["openrouter"]["selected_source"] = OpenRouterCredentialSource.MANAGED.value
    data["openrouter"]["selection_alias"] = OpenRouterSelectionAlias.DEEPSEEK_V4_FLASH_MANAGED.value

    loaded = from_dict(data)

    assert loaded.translation.model == TranslationModel.GEMMA4
    assert loaded.translation.connection == TranslationConnection.OPENROUTER
    assert loaded.provider.llm == LLMProviderName.OPENROUTER
    assert loaded.openrouter.llm_model == OpenRouterLLMModel.GEMMA_4_26B_A4B_IT
    assert loaded.openrouter.selected_source == OpenRouterCredentialSource.BYOK
    assert loaded.openrouter.selection_alias == OpenRouterSelectionAlias.GEMMA4_BYOK


def test_from_dict_infers_legacy_gemma_managed_translation_selection() -> None:
    data = to_dict(AppSettings())
    data.pop("translation", None)
    data["provider"]["llm"] = LLMProviderName.OPENROUTER.value
    data["openrouter"]["llm_model"] = OpenRouterLLMModel.GEMMA_4_26B_A4B_IT.value
    data["openrouter"]["selected_source"] = OpenRouterCredentialSource.MANAGED.value
    data["openrouter"]["selection_alias"] = OpenRouterSelectionAlias.GEMMA4_MANAGED.value
    data["openrouter"]["routing_mode"] = "novita_first"

    loaded = from_dict(data)

    assert loaded.translation.model == TranslationModel.GEMMA4_26B_31B
    assert loaded.translation.connection == TranslationConnection.MANAGED
    assert loaded.openrouter.routing_mode == OpenRouterRoutingMode.LATENCY


def test_from_dict_migrates_direct_qwen_flash_main_to_deepseek_managed() -> None:
    data = to_dict(AppSettings())
    data.pop("translation", None)
    data["provider"]["llm"] = LLMProviderName.QWEN.value
    data["qwen"]["llm_model"] = QwenLLMModel.QWEN_35_FLASH.value

    loaded = from_dict(data)

    assert loaded.translation.model == TranslationModel.DEEPSEEK_V4_FLASH
    assert loaded.translation.connection == TranslationConnection.MANAGED
    assert loaded.provider.llm == LLMProviderName.OPENROUTER
    assert loaded.openrouter.llm_model == OpenRouterLLMModel.DEEPSEEK_V4_FLASH
    assert loaded.openrouter.selection_alias == OpenRouterSelectionAlias.DEEPSEEK_V4_FLASH_MANAGED


def test_from_dict_migrates_openrouter_qwen_flash_main_to_deepseek_managed() -> None:
    data = to_dict(AppSettings())
    data.pop("translation", None)
    data["provider"]["llm"] = LLMProviderName.OPENROUTER.value
    data["openrouter"]["llm_model"] = OpenRouterLLMModel.QWEN_35_FLASH_02_23.value
    data["openrouter"]["selected_source"] = OpenRouterCredentialSource.BYOK.value
    data["openrouter"]["selection_alias"] = OpenRouterSelectionAlias.QWEN35_FLASH_BYOK.value
    data["openrouter"][
        "fallback_selection_alias"
    ] = OpenRouterFallbackSelectionAlias.QWEN35_FLASH.value

    loaded = from_dict(data)

    assert loaded.translation.model == TranslationModel.DEEPSEEK_V4_FLASH
    assert loaded.translation.connection == TranslationConnection.MANAGED
    assert loaded.openrouter.fallback_selection_alias == OpenRouterFallbackSelectionAlias.NONE
    assert loaded.openrouter.selection_alias == OpenRouterSelectionAlias.DEEPSEEK_V4_FLASH_MANAGED


def test_from_dict_migrates_openrouter_qwen_flash_main_preserving_routing_and_fallback() -> None:
    data = to_dict(AppSettings())
    data.pop("translation", None)
    data["provider"]["llm"] = LLMProviderName.OPENROUTER.value
    data["openrouter"]["llm_model"] = OpenRouterLLMModel.QWEN_35_FLASH_02_23.value
    data["openrouter"]["selected_source"] = OpenRouterCredentialSource.BYOK.value
    data["openrouter"]["selection_alias"] = OpenRouterSelectionAlias.QWEN35_FLASH_BYOK.value
    data["openrouter"]["routing_mode"] = "novita_first"
    data["openrouter"][
        "fallback_selection_alias"
    ] = OpenRouterFallbackSelectionAlias.QWEN35_FLASH.value

    loaded = from_dict(data)
    persisted = to_dict(loaded)

    assert loaded.translation.model == TranslationModel.DEEPSEEK_V4_FLASH
    assert loaded.translation.connection == TranslationConnection.MANAGED
    assert loaded.openrouter.routing_mode == OpenRouterRoutingMode.LATENCY
    assert loaded.openrouter.fallback_selection_alias == OpenRouterFallbackSelectionAlias.NONE
    assert loaded.openrouter.llm_model == OpenRouterLLMModel.DEEPSEEK_V4_FLASH
    assert loaded.openrouter.selected_source == OpenRouterCredentialSource.MANAGED
    assert loaded.openrouter.selection_alias == OpenRouterSelectionAlias.DEEPSEEK_V4_FLASH_MANAGED
    assert persisted["openrouter"]["routing_mode"] == OpenRouterRoutingMode.LATENCY.value
    assert persisted["openrouter"]["fallback_selection_alias"] == (
        OpenRouterFallbackSelectionAlias.NONE.value
    )


@pytest.mark.parametrize(
    (
        "model",
        "connection",
        "expected_provider",
        "expected_openrouter_model",
        "expected_openrouter_source",
        "expected_openrouter_alias",
        "expected_gemini_model",
        "expected_qwen_model",
        "expected_deepseek_model",
    ),
    [
        (
            TranslationModel.GEMMA4,
            TranslationConnection.MANAGED,
            LLMProviderName.OPENROUTER,
            OpenRouterLLMModel.GEMMA_4_26B_A4B_IT,
            OpenRouterCredentialSource.MANAGED,
            OpenRouterSelectionAlias.GEMMA4_MANAGED,
            None,
            None,
            None,
        ),
        (
            TranslationModel.GEMMA4,
            TranslationConnection.OPENROUTER,
            LLMProviderName.OPENROUTER,
            OpenRouterLLMModel.GEMMA_4_26B_A4B_IT,
            OpenRouterCredentialSource.BYOK,
            OpenRouterSelectionAlias.GEMMA4_BYOK,
            None,
            None,
            None,
        ),
        (
            TranslationModel.DEEPSEEK_V4_FLASH,
            TranslationConnection.MANAGED,
            LLMProviderName.OPENROUTER,
            OpenRouterLLMModel.DEEPSEEK_V4_FLASH,
            OpenRouterCredentialSource.MANAGED,
            OpenRouterSelectionAlias.DEEPSEEK_V4_FLASH_MANAGED,
            None,
            None,
            None,
        ),
        (
            TranslationModel.DEEPSEEK_V4_FLASH,
            TranslationConnection.MANAGED_CHINA,
            LLMProviderName.OPENROUTER,
            OpenRouterLLMModel.DEEPSEEK_V4_FLASH,
            OpenRouterCredentialSource.MANAGED,
            OpenRouterSelectionAlias.DEEPSEEK_V4_FLASH_MANAGED,
            None,
            None,
            None,
        ),
        (
            TranslationModel.DEEPSEEK_V4_FLASH,
            TranslationConnection.OPENROUTER,
            LLMProviderName.OPENROUTER,
            OpenRouterLLMModel.DEEPSEEK_V4_FLASH,
            OpenRouterCredentialSource.BYOK,
            OpenRouterSelectionAlias.DEEPSEEK_V4_FLASH_BYOK,
            None,
            None,
            None,
        ),
        (
            TranslationModel.DEEPSEEK_V4_FLASH,
            TranslationConnection.OFFICIAL_BYOK,
            LLMProviderName.DEEPSEEK,
            None,
            None,
            None,
            None,
            None,
            DeepSeekLLMModel.DEEPSEEK_V4_FLASH,
        ),
        (
            TranslationModel.GEMINI_37_FLASH,
            TranslationConnection.OFFICIAL_BYOK,
            LLMProviderName.GEMINI,
            None,
            None,
            None,
            GeminiLLMModel.GEMINI_37_FLASH,
            None,
            None,
        ),
        (
            TranslationModel.GEMINI_31_FLASH_LITE,
            TranslationConnection.OFFICIAL_BYOK,
            LLMProviderName.GEMINI,
            None,
            None,
            None,
            GeminiLLMModel.GEMINI_31_FLASH_LITE,
            None,
            None,
        ),
        (
            TranslationModel.QWEN_35_PLUS,
            TranslationConnection.OFFICIAL_BYOK,
            LLMProviderName.QWEN,
            None,
            None,
            None,
            None,
            QwenLLMModel.QWEN_35_PLUS,
            None,
        ),
    ],
)
def test_translation_settings_materializes_runtime_matrix(
    tmp_path,
    model,
    connection,
    expected_provider,
    expected_openrouter_model,
    expected_openrouter_source,
    expected_openrouter_alias,
    expected_gemini_model,
    expected_qwen_model,
    expected_deepseek_model,
) -> None:
    path = tmp_path / "settings.json"
    settings = AppSettings()
    settings.translation = TranslationSettings(
        model=model,
        connection=connection,
        connection_history={model.value: connection},
    )

    save_settings(path, settings)
    loaded = load_settings(path)

    assert loaded.translation.model == model
    assert loaded.translation.connection == connection
    assert loaded.provider.llm == expected_provider
    if expected_openrouter_model is not None:
        assert loaded.openrouter.llm_model == expected_openrouter_model
        assert loaded.openrouter.selected_source == expected_openrouter_source
        assert loaded.openrouter.selection_alias == expected_openrouter_alias
    if expected_gemini_model is not None:
        assert loaded.gemini.llm_model == expected_gemini_model
    if expected_qwen_model is not None:
        assert loaded.qwen.llm_model == expected_qwen_model
    if expected_deepseek_model is not None:
        assert loaded.deepseek.llm_model == expected_deepseek_model


def test_load_settings_persists_translation_section_for_legacy_file(tmp_path) -> None:
    path = tmp_path / "settings.json"
    legacy = to_dict(AppSettings())
    legacy["settings_version"] = 19
    legacy.pop("translation", None)
    path.write_text(json.dumps(legacy, ensure_ascii=False, indent=2), encoding="utf-8")

    loaded = load_settings(path)
    persisted = legacy_projected_settings_file(path)

    assert loaded.settings_version == SETTINGS_SCHEMA_VERSION
    assert loaded.translation.model == TranslationModel.GEMMA4_26B_31B
    assert loaded.translation.connection == TranslationConnection.MANAGED
    assert persisted["settings_version"] == SETTINGS_SCHEMA_VERSION
    assert persisted["translation"]["model"] == TranslationModel.GEMMA4_26B_31B.value
    assert persisted["translation"]["connection"] == TranslationConnection.MANAGED.value
    assert (
        persisted["translation"]["connection_history"][TranslationModel.GEMMA4_26B_31B.value]
        == TranslationConnection.MANAGED.value
    )


def test_load_settings_persists_default_translation_for_malformed_non_dict_section(
    tmp_path,
) -> None:
    path = tmp_path / "settings.json"
    raw = to_dict(AppSettings())
    raw["translation"] = ["broken"]
    path.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")

    loaded = load_settings(path)
    persisted = legacy_projected_settings_file(path)

    assert loaded.translation.model == TranslationModel.GEMMA4_26B_31B
    assert loaded.translation.connection == TranslationConnection.MANAGED
    assert persisted["translation"] == {
        "model": TranslationModel.GEMMA4_26B_31B.value,
        "connection": TranslationConnection.MANAGED.value,
        "connection_history": {
            TranslationModel.GEMMA4_26B_31B.value: TranslationConnection.MANAGED.value,
        },
        "fallback": DEFAULT_TRANSLATION_FALLBACK_DICT,
        "gpu_device_id": "auto",
    }


def test_migrate_v20_marks_valid_translation_schema_version_changed() -> None:
    raw = to_dict(AppSettings())
    raw["settings_version"] = 19

    migrated, changed = _migrate_settings_dict(raw)

    assert changed is True
    assert migrated["settings_version"] == SETTINGS_SCHEMA_VERSION
    assert migrated["translation"] == raw["translation"]


def test_invalid_translation_connection_falls_back_to_model_default() -> None:
    data = to_dict(AppSettings())
    data["translation"] = {
        "model": TranslationModel.GEMINI_37_FLASH.value,
        "connection": TranslationConnection.MANAGED.value,
        "connection_history": {
            TranslationModel.GEMINI_37_FLASH.value: TranslationConnection.OPENROUTER.value,
        },
    }

    loaded = from_dict(data)
    persisted = to_dict(loaded)

    assert loaded.translation.model == TranslationModel.GEMINI_37_FLASH
    assert loaded.translation.connection == TranslationConnection.OFFICIAL_BYOK
    assert loaded.provider.llm == LLMProviderName.GEMINI
    assert loaded.gemini.llm_model == GeminiLLMModel.GEMINI_37_FLASH
    assert persisted["translation"] == {
        "model": TranslationModel.GEMINI_37_FLASH.value,
        "connection": TranslationConnection.OFFICIAL_BYOK.value,
        "connection_history": {
            TranslationModel.GEMINI_37_FLASH.value: TranslationConnection.OFFICIAL_BYOK.value,
        },
        "fallback": DEFAULT_TRANSLATION_FALLBACK_DICT,
        "gpu_device_id": "auto",
    }


def test_load_settings_persists_normalized_translation_section(tmp_path) -> None:
    path = tmp_path / "settings.json"
    raw = to_dict(AppSettings())
    raw["translation"] = {
        "model": TranslationModel.GEMINI_31_FLASH_LITE.value,
        "connection": TranslationConnection.OPENROUTER.value,
        "connection_history": {
            TranslationModel.GEMINI_31_FLASH_LITE.value: TranslationConnection.MANAGED.value,
        },
    }
    path.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")

    loaded = load_settings(path)
    persisted = legacy_projected_settings_file(path)

    assert loaded.translation.model == TranslationModel.GEMINI_31_FLASH_LITE
    assert loaded.translation.connection == TranslationConnection.OPENROUTER
    assert persisted["translation"] == {
        "model": TranslationModel.GEMINI_31_FLASH_LITE.value,
        "connection": TranslationConnection.OPENROUTER.value,
        "connection_history": {
            TranslationModel.GEMINI_31_FLASH_LITE.value: TranslationConnection.OPENROUTER.value,
        },
        "fallback": DEFAULT_TRANSLATION_FALLBACK_DICT,
        "gpu_device_id": "auto",
    }


def test_load_settings_persists_materialized_runtime_fields_for_current_translation_schema(
    tmp_path,
) -> None:
    path = tmp_path / "settings.json"
    raw = to_dict(AppSettings())
    raw["settings_version"] = SETTINGS_SCHEMA_VERSION
    raw["translation"] = {
        "model": TranslationModel.DEEPSEEK_V4_FLASH.value,
        "connection": TranslationConnection.OFFICIAL_BYOK.value,
        "connection_history": {
            TranslationModel.DEEPSEEK_V4_FLASH.value: TranslationConnection.OFFICIAL_BYOK.value,
        },
    }
    raw["provider"]["llm"] = LLMProviderName.OPENROUTER.value
    raw["openrouter"]["llm_model"] = OpenRouterLLMModel.GEMMA_4_26B_A4B_IT.value
    raw["openrouter"]["selected_source"] = OpenRouterCredentialSource.MANAGED.value
    raw["openrouter"]["selection_alias"] = OpenRouterSelectionAlias.GEMMA4_MANAGED.value
    raw["openrouter"]["routing_mode"] = OpenRouterRoutingMode.LATENCY.value
    raw["openrouter"][
        "fallback_selection_alias"
    ] = OpenRouterFallbackSelectionAlias.QWEN35_FLASH.value
    path.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")

    loaded = load_settings(path)
    persisted = legacy_projected_settings_file(path)

    assert loaded.translation.model == TranslationModel.DEEPSEEK_V4_FLASH
    assert loaded.translation.connection == TranslationConnection.OFFICIAL_BYOK
    assert loaded.provider.llm == LLMProviderName.DEEPSEEK
    assert persisted["provider"]["llm"] == LLMProviderName.DEEPSEEK.value
    assert persisted["deepseek"]["llm_model"] == DeepSeekLLMModel.DEEPSEEK_V4_FLASH.value
    assert persisted["openrouter"]["routing_mode"] == OpenRouterRoutingMode.LATENCY.value
    assert persisted["openrouter"]["fallback_selection_alias"] == (
        OpenRouterFallbackSelectionAlias.NONE.value
    )


def test_qwen_flash_main_migration_uses_deepseek_connection_history() -> None:
    data = to_dict(AppSettings())
    data["translation"] = {
        "model": "qwen35_flash",
        "connection": TranslationConnection.MANAGED.value,
        "connection_history": {
            TranslationModel.DEEPSEEK_V4_FLASH.value: TranslationConnection.OFFICIAL_BYOK.value,
        },
    }
    data["provider"]["llm"] = LLMProviderName.QWEN.value
    data["qwen"]["llm_model"] = QwenLLMModel.QWEN_35_FLASH.value

    loaded = from_dict(data)

    assert loaded.translation.model == TranslationModel.DEEPSEEK_V4_FLASH
    assert loaded.translation.connection == TranslationConnection.OFFICIAL_BYOK
    assert loaded.provider.llm == LLMProviderName.DEEPSEEK
    assert loaded.deepseek.llm_model == DeepSeekLLMModel.DEEPSEEK_V4_FLASH


def test_openrouter_settings_roundtrip_constructor_without_alias_preserves_model_and_source() -> (
    None
):
    settings = OpenRouterSettings(
        llm_model=OpenRouterLLMModel.QWEN_35_FLASH_02_23,
        selected_source=OpenRouterCredentialSource.BYOK,
    )

    assert settings.llm_model == OpenRouterLLMModel.QWEN_35_FLASH_02_23
    assert settings.selected_source == OpenRouterCredentialSource.BYOK
    assert settings.selection_alias == OpenRouterSelectionAlias.QWEN35_FLASH_BYOK


def test_openrouter_explicit_inactive_state_keeps_selection_alias_none(tmp_path) -> None:
    path = tmp_path / "settings.json"
    settings = AppSettings(
        provider=ProviderSettings(llm=LLMProviderName.GEMINI),
        translation=TranslationSettings(
            model=TranslationModel.GEMINI_31_FLASH_LITE,
            connection=TranslationConnection.OFFICIAL_BYOK,
            connection_history={
                TranslationModel.GEMINI_31_FLASH_LITE.value: TranslationConnection.OFFICIAL_BYOK,
            },
        ),
        openrouter=OpenRouterSettings(
            selected_source=OpenRouterCredentialSource.NONE,
            selection_alias=None,
        ),
    )

    serialized = to_dict(settings)
    save_settings(path, settings)
    loaded = load_settings(path)
    persisted = legacy_projected_settings_file(path)

    assert settings.openrouter.llm_model == OpenRouterLLMModel.GEMMA_4_26B_A4B_IT
    assert settings.openrouter.selected_source == OpenRouterCredentialSource.NONE
    assert settings.openrouter.selection_alias is None
    assert serialized["openrouter"]["selection_alias"] is None
    assert loaded.openrouter.selection_alias is None
    assert loaded.openrouter.selected_source == OpenRouterCredentialSource.NONE
    assert persisted["openrouter"]["selection_alias"] is None


@pytest.mark.parametrize(
    "legacy_alias",
    [
        "gemma4",
        "gemini31_flash_lite",
        OpenRouterSelectionAlias.GEMMA4_MANAGED.value,
        OpenRouterSelectionAlias.GEMMA4_BYOK.value,
    ],
)
def test_from_dict_disables_legacy_openrouter_fallback_aliases(
    legacy_alias: str,
) -> None:
    data = to_dict(AppSettings())
    data["openrouter"]["fallback_selection_alias"] = legacy_alias

    loaded = from_dict(data)

    assert loaded.openrouter.fallback_selection_alias == OpenRouterFallbackSelectionAlias.NONE
    assert to_dict(loaded)["openrouter"]["fallback_selection_alias"] == (
        OpenRouterFallbackSelectionAlias.NONE.value
    )


def test_from_dict_defaults_invalid_openrouter_fallback_to_none() -> None:
    data = to_dict(AppSettings())
    data["openrouter"]["fallback_selection_alias"] = "broken-fallback"

    loaded = from_dict(data)

    assert loaded.openrouter.fallback_selection_alias == OpenRouterFallbackSelectionAlias.NONE


def test_openrouter_legacy_gemini25_flash_lite_fallback_is_not_live_alias() -> None:
    with pytest.raises(KeyError):
        resolve_openrouter_fallback_model("gemini25_flash_lite")


def test_from_dict_disables_legacy_gemini25_flash_lite_fallback() -> None:
    data = to_dict(AppSettings())
    data["openrouter"]["fallback_selection_alias"] = "gemini25_flash_lite"

    loaded = from_dict(data)

    assert loaded.openrouter.fallback_selection_alias == OpenRouterFallbackSelectionAlias.NONE


def test_openrouter_deepseek_v4_flash_aliases_use_stable_slug() -> None:
    expected = "deepseek/deepseek-v4-flash-0731"
    deepseek_model = getattr(OpenRouterLLMModel, "DEEPSEEK_V4_FLASH", None)
    deepseek_managed = getattr(OpenRouterSelectionAlias, "DEEPSEEK_V4_FLASH_MANAGED", None)
    deepseek_byok = getattr(OpenRouterSelectionAlias, "DEEPSEEK_V4_FLASH_BYOK", None)
    deepseek_fallback = getattr(OpenRouterFallbackSelectionAlias, "DEEPSEEK_V4_FLASH", None)
    deepseek_china_fallback = getattr(
        OpenRouterFallbackSelectionAlias, "DEEPSEEK_V4_FLASH_CHINA", None
    )

    assert deepseek_model is not None
    assert deepseek_managed is not None
    assert deepseek_byok is not None
    assert deepseek_fallback is not None
    assert deepseek_china_fallback is not None

    assert deepseek_model.value == expected
    assert (
        openrouter_alias_for_fields(
            model=expected,
            source=OpenRouterCredentialSource.MANAGED.value,
        )
        == deepseek_managed.value
    )
    assert (
        openrouter_alias_for_fields(
            model=expected,
            source=OpenRouterCredentialSource.BYOK.value,
        )
        == deepseek_byok.value
    )
    assert resolve_openrouter_fallback_model(deepseek_fallback.value) == expected
    assert resolve_openrouter_fallback_model(deepseek_china_fallback.value) == expected


def test_from_dict_normalizes_legacy_openrouter_deepseek_model() -> None:
    data = to_dict(AppSettings())
    data["provider"]["llm"] = LLMProviderName.OPENROUTER.value
    data["translation"]["model"] = TranslationModel.DEEPSEEK_V4_FLASH.value
    data["translation"]["connection"] = TranslationConnection.OPENROUTER.value
    data["openrouter"]["llm_model"] = "deepseek/deepseek-v4-flash"
    data["openrouter"]["selected_source"] = OpenRouterCredentialSource.BYOK.value
    data["openrouter"]["selection_alias"] = OpenRouterSelectionAlias.DEEPSEEK_V4_FLASH_BYOK.value

    loaded = from_dict(data)

    assert loaded.openrouter.llm_model == OpenRouterLLMModel.DEEPSEEK_V4_FLASH
    assert loaded.openrouter.llm_model.value == "deepseek/deepseek-v4-flash-0731"


def test_openrouter_settings_roundtrip_persists_deepseek_china_translation_fallback(
    tmp_path,
) -> None:
    path = tmp_path / "settings.json"

    settings = AppSettings(
        provider=ProviderSettings(llm=LLMProviderName.OPENROUTER),
        translation=TranslationSettings(
            fallback=TranslationFallbackSettings(
                enabled=True,
                model=TranslationModel.DEEPSEEK_V4_FLASH,
                connection=TranslationConnection.MANAGED_CHINA,
            )
        ),
        openrouter=OpenRouterSettings(
            selected_source=OpenRouterCredentialSource.BYOK,
            selection_alias=OpenRouterSelectionAlias.GEMMA4_BYOK,
            fallback_selection_alias=OpenRouterFallbackSelectionAlias.NONE,
        ),
    )

    save_settings(path, settings)

    loaded = load_settings(path)
    persisted = legacy_projected_settings_file(path)

    assert loaded.openrouter.fallback_selection_alias == OpenRouterFallbackSelectionAlias.NONE
    assert loaded.translation.fallback == TranslationFallbackSettings(
        enabled=True,
        model=TranslationModel.DEEPSEEK_V4_FLASH,
        connection=TranslationConnection.MANAGED_CHINA,
    )
    assert persisted["openrouter"]["fallback_selection_alias"] == (
        OpenRouterFallbackSelectionAlias.NONE.value
    )
    assert persisted["translation"]["fallback"] == {
        "enabled": True,
        "model": TranslationModel.DEEPSEEK_V4_FLASH.value,
        "connection": TranslationConnection.MANAGED_CHINA.value,
    }


def test_openrouter_settings_roundtrip_persists_deepseek_selection_and_translation_fallback(
    tmp_path,
) -> None:
    path = tmp_path / "settings.json"
    deepseek_model = getattr(OpenRouterLLMModel, "DEEPSEEK_V4_FLASH", None)
    deepseek_managed = getattr(OpenRouterSelectionAlias, "DEEPSEEK_V4_FLASH_MANAGED", None)

    assert deepseek_model is not None
    assert deepseek_managed is not None

    settings = AppSettings(
        provider=ProviderSettings(llm=LLMProviderName.OPENROUTER),
        translation=TranslationSettings(
            model=TranslationModel.DEEPSEEK_V4_FLASH,
            connection=TranslationConnection.MANAGED,
            connection_history={
                TranslationModel.DEEPSEEK_V4_FLASH.value: TranslationConnection.MANAGED,
            },
            fallback=TranslationFallbackSettings(
                enabled=True,
                model=TranslationModel.DEEPSEEK_V4_FLASH,
                connection=TranslationConnection.OPENROUTER,
            ),
        ),
        openrouter=OpenRouterSettings(
            llm_model=deepseek_model,
            routing_mode=OpenRouterRoutingMode.LATENCY,
            selected_source=OpenRouterCredentialSource.MANAGED,
            selection_alias=deepseek_managed,
            fallback_selection_alias=OpenRouterFallbackSelectionAlias.NONE,
        ),
    )

    serialized = to_dict(settings)

    assert serialized["openrouter"]["selection_alias"] == deepseek_managed.value
    assert serialized["openrouter"]["llm_model"] == deepseek_model.value
    assert serialized["openrouter"]["selected_source"] == OpenRouterCredentialSource.MANAGED.value
    assert serialized["openrouter"]["fallback_selection_alias"] == (
        OpenRouterFallbackSelectionAlias.NONE.value
    )
    assert serialized["translation"]["fallback"] == {
        "enabled": True,
        "model": TranslationModel.DEEPSEEK_V4_FLASH.value,
        "connection": TranslationConnection.OPENROUTER.value,
    }

    save_settings(path, settings)

    loaded = load_settings(path)
    persisted = legacy_projected_settings_file(path)

    assert loaded.openrouter.selection_alias == deepseek_managed
    assert loaded.openrouter.llm_model == deepseek_model
    assert loaded.openrouter.selected_source == OpenRouterCredentialSource.MANAGED
    assert loaded.openrouter.fallback_selection_alias == OpenRouterFallbackSelectionAlias.NONE
    assert loaded.translation.fallback == TranslationFallbackSettings(
        enabled=True,
        model=TranslationModel.DEEPSEEK_V4_FLASH,
        connection=TranslationConnection.OPENROUTER,
    )
    assert persisted["openrouter"]["selection_alias"] == deepseek_managed.value
    assert persisted["openrouter"]["llm_model"] == deepseek_model.value
    assert persisted["openrouter"]["selected_source"] == OpenRouterCredentialSource.MANAGED.value
    assert persisted["openrouter"]["fallback_selection_alias"] == (
        OpenRouterFallbackSelectionAlias.NONE.value
    )


def test_openrouter_settings_derives_deepseek_byok_alias_without_explicit_alias() -> None:
    deepseek_model = getattr(OpenRouterLLMModel, "DEEPSEEK_V4_FLASH", None)
    deepseek_byok = getattr(OpenRouterSelectionAlias, "DEEPSEEK_V4_FLASH_BYOK", None)

    assert deepseek_model is not None
    assert deepseek_byok is not None

    settings = OpenRouterSettings(
        llm_model=deepseek_model,
        selected_source=OpenRouterCredentialSource.BYOK,
    )

    assert settings.llm_model == deepseek_model
    assert settings.selected_source == OpenRouterCredentialSource.BYOK
    assert settings.selection_alias == deepseek_byok


def test_openrouter_qwen_flash_main_roundtrip_migrates_to_deepseek_and_disables_legacy_fallback(
    tmp_path,
) -> None:
    path = tmp_path / "settings.json"
    legacy = to_dict(AppSettings())
    legacy.pop("translation", None)
    legacy["provider"]["llm"] = LLMProviderName.OPENROUTER.value
    legacy["openrouter"]["llm_model"] = OpenRouterLLMModel.QWEN_35_FLASH_02_23.value
    legacy["openrouter"]["routing_mode"] = OpenRouterRoutingMode.LATENCY.value
    legacy["openrouter"]["selected_source"] = OpenRouterCredentialSource.MANAGED.value
    legacy["openrouter"]["selection_alias"] = OpenRouterSelectionAlias.QWEN35_FLASH_MANAGED.value
    legacy["openrouter"]["fallback_selection_alias"] = "gemini25_flash_lite"
    path.write_text(json.dumps(legacy, ensure_ascii=False, indent=2), encoding="utf-8")

    loaded = load_settings(path)
    persisted = legacy_projected_settings_file(path)

    assert isinstance(loaded.openrouter.selection_alias, OpenRouterSelectionAlias)
    assert isinstance(loaded.openrouter.fallback_selection_alias, OpenRouterFallbackSelectionAlias)
    assert loaded.translation.model == TranslationModel.DEEPSEEK_V4_FLASH
    assert loaded.translation.connection == TranslationConnection.MANAGED
    assert loaded.openrouter.selection_alias == OpenRouterSelectionAlias.DEEPSEEK_V4_FLASH_MANAGED
    assert loaded.openrouter.fallback_selection_alias == OpenRouterFallbackSelectionAlias.NONE
    assert loaded.translation.fallback == TranslationFallbackSettings()
    assert loaded.openrouter.llm_model == OpenRouterLLMModel.DEEPSEEK_V4_FLASH
    assert loaded.openrouter.selected_source == OpenRouterCredentialSource.MANAGED
    assert (
        persisted["openrouter"]["selection_alias"]
        == OpenRouterSelectionAlias.DEEPSEEK_V4_FLASH_MANAGED.value
    )
    assert persisted["openrouter"]["llm_model"] == OpenRouterLLMModel.DEEPSEEK_V4_FLASH.value
    assert persisted["openrouter"]["selected_source"] == OpenRouterCredentialSource.MANAGED.value
    assert persisted["openrouter"]["fallback_selection_alias"] == (
        OpenRouterFallbackSelectionAlias.NONE.value
    )
    assert persisted["translation"]["fallback"] == DEFAULT_TRANSLATION_FALLBACK_DICT


def test_load_settings_backfills_openrouter_blocks_and_persists(tmp_path):
    path = tmp_path / "settings.json"
    legacy = to_dict(AppSettings())
    legacy["settings_version"] = 4
    legacy.pop("translation", None)
    legacy["provider"]["llm"] = LLMProviderName.GEMINI.value
    legacy.pop("openrouter", None)
    legacy.setdefault("api_key_verified", {}).pop("openrouter", None)
    path.write_text(json.dumps(legacy, ensure_ascii=False, indent=2), encoding="utf-8")

    loaded = load_settings(path)
    persisted = legacy_projected_settings_file(path)

    assert loaded.settings_version == SETTINGS_SCHEMA_VERSION
    assert loaded.openrouter.llm_model == OpenRouterLLMModel.GEMMA_4_26B_A4B_IT
    assert loaded.openrouter.routing_mode == OpenRouterRoutingMode.LATENCY
    assert loaded.openrouter.selected_source == OpenRouterCredentialSource.NONE
    assert loaded.openrouter.selection_alias is None
    assert loaded.api_key_verified.openrouter is False
    assert persisted["settings_version"] == SETTINGS_SCHEMA_VERSION
    assert persisted["openrouter"]["llm_model"] == OpenRouterLLMModel.GEMMA_4_26B_A4B_IT.value
    assert persisted["openrouter"]["routing_mode"] == OpenRouterRoutingMode.LATENCY.value
    assert persisted["openrouter"]["selected_source"] == OpenRouterCredentialSource.NONE.value
    assert persisted["openrouter"]["selection_alias"] is None
    assert persisted["api_key_verified"]["openrouter"] is False


def test_load_settings_backfills_openrouter_aliases_from_legacy_fields(tmp_path) -> None:
    path = tmp_path / "settings.json"
    legacy = to_dict(AppSettings())
    legacy["settings_version"] = SETTINGS_SCHEMA_VERSION - 1
    legacy.pop("translation", None)
    legacy["provider"]["llm"] = LLMProviderName.OPENROUTER.value
    legacy["openrouter"]["llm_model"] = OpenRouterLLMModel.GEMMA_4_26B_A4B_IT.value
    legacy["openrouter"]["selected_source"] = OpenRouterCredentialSource.MANAGED.value
    legacy["openrouter"].pop("selection_alias", None)
    legacy["openrouter"].pop("fallback_selection_alias", None)
    path.write_text(json.dumps(legacy, ensure_ascii=False, indent=2), encoding="utf-8")

    loaded = load_settings(path)
    persisted = legacy_projected_settings_file(path)

    assert loaded.settings_version == SETTINGS_SCHEMA_VERSION
    assert loaded.openrouter.llm_model == OpenRouterLLMModel.GEMMA_4_26B_A4B_IT
    assert loaded.openrouter.selected_source == OpenRouterCredentialSource.MANAGED
    assert loaded.openrouter.selection_alias == OpenRouterSelectionAlias.GEMMA4_26B_31B_MANAGED
    assert loaded.openrouter.fallback_selection_alias == OpenRouterFallbackSelectionAlias.NONE
    assert persisted["settings_version"] == SETTINGS_SCHEMA_VERSION
    assert (
        persisted["openrouter"]["selection_alias"]
        == OpenRouterSelectionAlias.GEMMA4_26B_31B_MANAGED.value
    )
    assert persisted["openrouter"]["fallback_selection_alias"] == (
        OpenRouterFallbackSelectionAlias.NONE.value
    )


def test_load_settings_backfills_openrouter_selected_source_to_byok_for_legacy_openrouter_provider(
    tmp_path,
):
    path = tmp_path / "settings.json"
    legacy = to_dict(AppSettings())
    legacy["settings_version"] = 9
    legacy.pop("translation", None)
    legacy["provider"]["llm"] = LLMProviderName.OPENROUTER.value
    legacy["openrouter"]["llm_model"] = OpenRouterLLMModel.GEMMA_4_26B_A4B_IT.value
    legacy["openrouter"].pop("selected_source", None)
    legacy["openrouter"].pop("selection_alias", None)
    path.write_text(json.dumps(legacy, ensure_ascii=False, indent=2), encoding="utf-8")

    loaded = load_settings(path)
    persisted = legacy_projected_settings_file(path)

    assert loaded.openrouter.selected_source == OpenRouterCredentialSource.BYOK
    assert loaded.openrouter.selection_alias == OpenRouterSelectionAlias.GEMMA4_26B_31B_BYOK
    assert persisted["openrouter"]["selected_source"] == OpenRouterCredentialSource.BYOK.value
    assert (
        persisted["openrouter"]["selection_alias"]
        == OpenRouterSelectionAlias.GEMMA4_26B_31B_BYOK.value
    )


def test_load_settings_normalizes_legacy_active_openrouter_none_selected_source_to_byok(
    tmp_path,
) -> None:
    path = tmp_path / "settings.json"
    legacy = to_dict(AppSettings())
    legacy["settings_version"] = 10
    legacy.pop("translation", None)
    legacy["provider"]["llm"] = LLMProviderName.OPENROUTER.value
    legacy["openrouter"]["llm_model"] = OpenRouterLLMModel.GEMMA_4_26B_A4B_IT.value
    legacy["openrouter"]["selected_source"] = OpenRouterCredentialSource.NONE.value
    legacy["openrouter"].pop("selection_alias", None)
    path.write_text(json.dumps(legacy, ensure_ascii=False, indent=2), encoding="utf-8")

    loaded = load_settings(path)
    persisted = legacy_projected_settings_file(path)

    assert loaded.openrouter.selected_source == OpenRouterCredentialSource.BYOK
    assert loaded.openrouter.selection_alias == OpenRouterSelectionAlias.GEMMA4_26B_31B_BYOK
    assert persisted["settings_version"] == SETTINGS_SCHEMA_VERSION
    assert persisted["openrouter"]["selected_source"] == OpenRouterCredentialSource.BYOK.value
    assert (
        persisted["openrouter"]["selection_alias"]
        == OpenRouterSelectionAlias.GEMMA4_26B_31B_BYOK.value
    )


def test_from_dict_defaults_invalid_openrouter_routing_mode_to_latency() -> None:
    data = to_dict(AppSettings())
    data["openrouter"]["routing_mode"] = "broken"

    loaded = from_dict(data)

    assert loaded.openrouter.routing_mode == OpenRouterRoutingMode.LATENCY


def test_from_dict_ignores_legacy_system_prompts_for_selected_provider():
    data = to_dict(AppSettings())
    data.pop("translation", None)
    data["provider"]["llm"] = "qwen"
    data["system_prompts"] = {
        "gemini": "gemini custom",
        "qwen": "qwen custom",
    }
    data["system_prompt"] = "legacy"

    loaded = from_dict(data)
    assert loaded.system_prompt == "legacy"
    assert loaded.system_prompts == {}


def test_from_dict_ignores_legacy_openrouter_system_prompt_map():
    data = to_dict(AppSettings())
    data.pop("translation", None)
    data["provider"]["llm"] = "openrouter"
    data["system_prompts"] = {
        "gemini": "gemini custom",
        "openrouter": "openrouter custom",
        "qwen": "qwen custom",
    }
    data["system_prompt"] = "legacy"

    loaded = from_dict(data)

    assert loaded.system_prompt == "legacy"
    assert loaded.system_prompts == {}


def test_from_dict_backfills_legacy_system_prompt_to_selected_provider():
    data = to_dict(AppSettings())
    data.pop("translation", None)
    data["provider"]["llm"] = "gemini"
    data["system_prompt"] = "legacy prompt"
    data.pop("system_prompts", None)

    loaded = from_dict(data)
    assert loaded.system_prompts == {}
    assert loaded.system_prompt == "legacy prompt"


def test_load_settings_schema_migration_resets_all_prompt_values(tmp_path) -> None:
    pre_unified_prompt_schema_version = 18
    path = tmp_path / "settings.json"
    legacy = to_dict(AppSettings())
    legacy["settings_version"] = pre_unified_prompt_schema_version
    legacy["system_prompt"] = "old custom prompt"
    legacy["system_prompts"] = {
        "gemini": "old gemini prompt",
        "openrouter": "old openrouter prompt",
        "qwen": "old qwen prompt",
        "deepseek": "old deepseek prompt",
    }
    path.write_text(json.dumps(legacy, ensure_ascii=False, indent=2), encoding="utf-8")

    loaded = load_settings(path)
    persisted = legacy_projected_settings_file(path)
    shared_prompt = load_prompt_for_provider("gemini")

    assert loaded.settings_version == SETTINGS_SCHEMA_VERSION
    assert persisted["settings_version"] == SETTINGS_SCHEMA_VERSION
    assert loaded.system_prompt == shared_prompt
    assert persisted["system_prompt"] == shared_prompt
    assert loaded.system_prompts == {}
    assert "system_prompts" not in persisted


def test_load_settings_migrates_legacy_timestamp_prompt_to_new_default(tmp_path) -> None:
    path = tmp_path / "settings.json"
    legacy = to_dict(AppSettings())
    legacy["settings_version"] = SETTINGS_SCHEMA_VERSION - 1
    legacy["system_prompt"] = LEGACY_TIMESTAMP_PROMPT
    path.write_text(json.dumps(legacy, ensure_ascii=False, indent=2), encoding="utf-8")

    loaded = load_settings(path)
    persisted = legacy_projected_settings_file(path)
    shared_prompt = load_prompt_for_provider("gemini")

    assert loaded.settings_version == SETTINGS_SCHEMA_VERSION
    assert loaded.system_prompt == shared_prompt
    assert persisted["settings_version"] == SETTINGS_SCHEMA_VERSION
    assert persisted["system_prompt"] == shared_prompt


def test_load_settings_preserves_custom_prompt_when_upgrading_schema(tmp_path) -> None:
    path = tmp_path / "settings.json"
    legacy = to_dict(AppSettings())
    legacy["settings_version"] = SETTINGS_SCHEMA_VERSION - 1
    legacy["system_prompt"] = "my customized prompt"
    path.write_text(json.dumps(legacy, ensure_ascii=False, indent=2), encoding="utf-8")

    loaded = load_settings(path)
    persisted = legacy_projected_settings_file(path)

    assert loaded.system_prompt == "my customized prompt"
    assert persisted["system_prompt"] == "my customized prompt"


def test_migrate_settings_dict_prompt_upgrade_is_idempotent() -> None:
    raw = to_dict(AppSettings())
    raw["settings_version"] = SETTINGS_SCHEMA_VERSION - 1
    raw["system_prompt"] = LEGACY_TIMESTAMP_PROMPT

    once, first_changed = _migrate_settings_dict(copy.deepcopy(raw))
    twice, second_changed = _migrate_settings_dict(copy.deepcopy(once))

    assert first_changed is True
    assert second_changed is False
    assert once["system_prompt"] == load_prompt_for_provider("gemini")
    assert twice["system_prompt"] == once["system_prompt"]


def test_migrate_settings_dict_preserves_prompt_with_boundary_whitespace() -> None:
    raw = to_dict(AppSettings())
    raw["settings_version"] = SETTINGS_SCHEMA_VERSION - 1
    raw["system_prompt"] = f"\n{LEGACY_TIMESTAMP_PROMPT}\n"

    migrated, changed = _migrate_settings_dict(copy.deepcopy(raw))

    assert changed is True
    assert migrated["system_prompt"] == f"\n{LEGACY_TIMESTAMP_PROMPT}\n"


def test_load_settings_preserves_prompt_with_boundary_whitespace(tmp_path) -> None:
    path = tmp_path / "settings.json"
    legacy = to_dict(AppSettings())
    legacy["settings_version"] = SETTINGS_SCHEMA_VERSION
    legacy["system_prompt"] = f"  {LEGACY_TIMESTAMP_PROMPT}  "
    path.write_text(json.dumps(legacy, ensure_ascii=False, indent=2), encoding="utf-8")

    loaded = load_settings(path)
    persisted = legacy_projected_settings_file(path)

    assert loaded.system_prompt == f"  {LEGACY_TIMESTAMP_PROMPT}  "
    assert persisted["system_prompt"] == f"  {LEGACY_TIMESTAMP_PROMPT}  "


def test_from_dict_initializes_empty_prompt_fields_to_shared_default() -> None:
    data = to_dict(AppSettings())
    data["system_prompt"] = "  "
    data["system_prompts"] = {}

    loaded = from_dict(data)
    shared_prompt = load_prompt_for_provider("gemini")

    assert loaded.system_prompt == shared_prompt
    assert loaded.system_prompts == {}


def test_prompt_customized_after_migration_survives_save_load(tmp_path) -> None:
    path = tmp_path / "settings.json"
    custom_qwen_prompt = LEGACY_QWEN_DEFAULT_PROMPT
    settings = AppSettings()
    settings.provider.llm = LLMProviderName.QWEN
    settings.translation = TranslationSettings(
        model=TranslationModel.QWEN_35_PLUS,
        connection=TranslationConnection.OFFICIAL_BYOK,
        connection_history={
            TranslationModel.QWEN_35_PLUS.value: TranslationConnection.OFFICIAL_BYOK,
        },
    )
    settings.system_prompt = custom_qwen_prompt

    save_settings(path, settings)

    loaded = load_settings(path)
    persisted = legacy_projected_settings_file(path)

    assert loaded.settings_version == SETTINGS_SCHEMA_VERSION
    assert persisted["settings_version"] == SETTINGS_SCHEMA_VERSION
    assert loaded.system_prompt == custom_qwen_prompt
    assert loaded.system_prompts == {}
    assert persisted["system_prompt"] == custom_qwen_prompt
    assert "system_prompts" not in persisted


def test_load_settings_migrates_legacy_soniox_model_and_persists(tmp_path):
    path = tmp_path / "settings.json"
    legacy = to_dict(AppSettings())
    legacy["settings_version"] = 2
    legacy["soniox_stt"]["model"] = "stt-rt-v3"
    path.write_text(json.dumps(legacy, ensure_ascii=False, indent=2), encoding="utf-8")

    loaded = load_settings(path)
    assert loaded.settings_version == SETTINGS_SCHEMA_VERSION
    assert loaded.soniox_stt.model == "stt-rt-v5"

    persisted = legacy_projected_settings_file(path)
    assert persisted["settings_version"] == SETTINGS_SCHEMA_VERSION
    assert persisted["soniox_stt"]["model"] == "stt-rt-v5"


def test_load_settings_migration_preserves_custom_soniox_model(tmp_path):
    path = tmp_path / "settings.json"
    legacy = to_dict(AppSettings())
    legacy["settings_version"] = 2
    legacy["soniox_stt"]["model"] = "stt-rt-experimental"
    path.write_text(json.dumps(legacy, ensure_ascii=False, indent=2), encoding="utf-8")

    loaded = load_settings(path)
    assert loaded.settings_version == SETTINGS_SCHEMA_VERSION
    assert loaded.soniox_stt.model == "stt-rt-experimental"

    persisted = legacy_projected_settings_file(path)
    assert persisted["settings_version"] == SETTINGS_SCHEMA_VERSION
    assert persisted["soniox_stt"]["model"] == "stt-rt-experimental"


def test_mask_secret():
    assert mask_secret("sk-123456") == "sk-****"
    assert mask_secret("abc", unmasked_prefix=3) == "***"


def test_encrypted_file_secret_store_roundtrip(tmp_path):
    path = tmp_path / "secrets.json"
    store = EncryptedFileSecretStore(path, passphrase="pw")
    store.set("google_api_key", "sk-SECRET")

    assert store.get("google_api_key") == "sk-SECRET"
    store.delete("google_api_key")
    assert store.get("google_api_key") is None


def test_keyring_secret_store_delete_propagates_unexpected_backend_exceptions() -> None:
    class PasswordDeleteError(Exception):
        pass

    class BrokenKeyring:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str]] = []
            self.errors = type("errors", (), {"PasswordDeleteError": PasswordDeleteError})

        def delete_password(self, service_name: str, key: str) -> None:
            self.calls.append((service_name, key))
            raise RuntimeError("keyring delete failed")

    fake_keyring = BrokenKeyring()

    class FakeKeyringSecretStore(KeyringSecretStore):
        def _keyring(self):
            return fake_keyring

    store = FakeKeyringSecretStore(service_name="test-service")

    with pytest.raises(RuntimeError, match="keyring delete failed"):
        store.delete("local_llm_api_key")

    assert fake_keyring.calls == [("test-service", "local_llm_api_key")]


def test_keyring_secret_store_delete_ignores_password_delete_error() -> None:
    class PasswordDeleteError(Exception):
        pass

    class MissingPasswordKeyring:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str, str]] = []
            self.errors = type("errors", (), {"PasswordDeleteError": PasswordDeleteError})

        def delete_password(self, service_name: str, key: str) -> None:
            self.calls.append(("delete", service_name, key))
            raise PasswordDeleteError("password not found")

        def get_password(self, service_name: str, key: str) -> str | None:
            self.calls.append(("get", service_name, key))
            return None

    fake_keyring = MissingPasswordKeyring()

    class FakeKeyringSecretStore(KeyringSecretStore):
        def _keyring(self):
            return fake_keyring

    store = FakeKeyringSecretStore(service_name="test-service")

    store.delete("local_llm_api_key")

    assert fake_keyring.calls == [
        ("delete", "test-service", "local_llm_api_key"),
        ("get", "test-service", "local_llm_api_key"),
    ]


def test_keyring_secret_store_delete_reraises_password_delete_error_when_secret_remains() -> None:
    class PasswordDeleteError(Exception):
        pass

    class StillPresentKeyring:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str, str]] = []
            self.errors = type("errors", (), {"PasswordDeleteError": PasswordDeleteError})

        def delete_password(self, service_name: str, key: str) -> None:
            self.calls.append(("delete", service_name, key))
            raise PasswordDeleteError("delete failed")

        def get_password(self, service_name: str, key: str) -> str | None:
            self.calls.append(("get", service_name, key))
            return "still-present"

    fake_keyring = StillPresentKeyring()

    class FakeKeyringSecretStore(KeyringSecretStore):
        def _keyring(self):
            return fake_keyring

    store = FakeKeyringSecretStore(service_name="test-service")

    with pytest.raises(PasswordDeleteError, match="delete failed"):
        store.delete("local_llm_api_key")

    assert fake_keyring.calls == [
        ("delete", "test-service", "local_llm_api_key"),
        ("get", "test-service", "local_llm_api_key"),
    ]


def test_encrypted_file_secret_store_does_not_store_plaintext(tmp_path):
    path = tmp_path / "secrets.json"
    store = EncryptedFileSecretStore(path, passphrase="pw")
    store.set("k", "sk-SECRET")

    raw = json.loads(path.read_text(encoding="utf-8"))
    assert "sk-SECRET" not in json.dumps(raw)


def test_encrypted_file_secret_store_rejects_wrong_passphrase(tmp_path):
    path = tmp_path / "secrets.json"
    store = EncryptedFileSecretStore(path, passphrase="pw")
    store.set("k", "sk-SECRET")

    wrong = EncryptedFileSecretStore(path, passphrase="wrong")
    with pytest.raises(ValueError):
        wrong.get("k")
