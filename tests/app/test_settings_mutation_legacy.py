from __future__ import annotations

import pytest
from puripuly_heart.app.services.settings_mutation_legacy import (
    ORDER21_TRANSLATION_PROVIDER_SETTINGS_PATHS,
    ORDER22_STT_LANGUAGE_AUDIO_SETTINGS_PATHS,
    ORDER23_OVERLAY_OSC_OUTPUT_SETTINGS_PATHS,
    ORDER24_UI_PROMPT_CLIPBOARD_STATE_SETTINGS_PATHS,
    SettingsPathMutationValidator,
    SettingsPathPatch,
    build_translation_provider_settings_path_patch,
)

from puripuly_heart.app.services import settings_mutation
from puripuly_heart.config.settings import (
    AppSettings,
    TranslationConnection,
    TranslationModel,
    TranslationSettings,
)
from puripuly_heart.core import messages


def test_order21_translation_provider_patch_records_initial_covered_surface_list() -> None:
    assert set(ORDER21_TRANSLATION_PROVIDER_SETTINGS_PATHS) == {
        "translation.model",
        "translation.connection",
        "translation.connection_history",
        "translation.fallback",
        "translation.http_extension_id",
        "translation.previous_llm_model",
        "translation.gpu_device_id",
        "provider.llm",
        "gemini.llm_model",
        "openrouter.llm_model",
        "openrouter.routing_mode",
        "openrouter.provider_routing",
        "openrouter.selected_source",
        "openrouter.selection_alias",
        "openrouter.broker_base_url",
        "qwen.llm_model",
        "qwen.region",
        "deepseek.llm_model",
        "local_llm.backend",
        "local_llm.base_url",
        "local_llm.model",
        "local_llm.extra_body",
        "llm.concurrency_limit",
    }


def test_order21_patch_carries_custom_http_identity_fields() -> None:
    previous = AppSettings()
    next_settings = AppSettings()
    next_settings.translation = TranslationSettings(
        model=TranslationModel.CUSTOM_HTTP,
        connection=TranslationConnection.CUSTOM_HTTP,
        http_extension_id="demo",
        previous_llm_model=TranslationModel.GEMMA4_26B_31B,
    )

    patch = build_translation_provider_settings_path_patch(previous, next_settings)

    assert patch["translation.http_extension_id"] == "demo"
    assert patch["translation.previous_llm_model"] == TranslationModel.GEMMA4_26B_31B


def test_order22_stt_language_audio_patch_records_initial_covered_surface_list() -> None:
    assert set(ORDER22_STT_LANGUAGE_AUDIO_SETTINGS_PATHS) == {
        "provider.stt",
        "provider.peer_stt",
        "languages.source_language",
        "languages.target_language",
        "languages.peer_source_language",
        "languages.peer_target_language",
        "languages.peer_source_mode",
        "languages.peer_expected_languages",
        "languages.recent_source_languages",
        "languages.recent_target_languages",
        "audio.internal_sample_rate_hz",
        "audio.internal_channels",
        "audio.ring_buffer_ms",
        "audio.input_host_api",
        "audio.input_device",
        "desktop_audio.output_device",
        "desktop_audio.vad_speech_threshold",
        "desktop_audio.vad_hangover_ms",
        "desktop_audio.vad_pre_roll_ms",
        "stt.drain_timeout_s",
        "stt.vad_speech_threshold",
        "stt.low_latency_vad_hangover_ms",
        "stt.low_latency_merge_gap_ms",
        "stt.low_latency_spec_retry_max",
        "stt.custom_vocabulary_enabled",
        "stt.custom_terms",
        "stt.gpu_device_id",
        "deepgram_stt.model",
        "qwen_asr_stt.model",
        "soniox_stt.model",
        "soniox_stt.endpoint",
        "soniox_stt.keepalive_interval_s",
        "soniox_stt.trailing_silence_ms",
    }


def test_order23_overlay_osc_output_patch_records_initial_covered_surface_list() -> None:
    assert set(ORDER23_OVERLAY_OSC_OUTPUT_SETTINGS_PATHS) == {
        "overlay.target",
        "overlay.show_translation",
        "overlay.show_peer_original",
        "overlay.calibration.anchor",
        "overlay.calibration.offset_x",
        "overlay.calibration.offset_y",
        "overlay.calibration.distance",
        "overlay.calibration.text_scale",
        "overlay.calibration.background_alpha",
        "overlay.desktop_flet.size_preset",
        "overlay.desktop_flet.position.x",
        "overlay.desktop_flet.position.y",
        "overlay.desktop_flet.swap_caption_languages",
        "overlay.desktop_flet.visual.background_alpha",
        "osc.host",
        "osc.port",
        "osc.connection_mode",
        "osc.send_port",
        "osc.receive_port",
        "osc.chatbox_address",
        "osc.chatbox_send",
        "osc.chatbox_clear",
        "osc.chatbox_max_chars",
        "osc.vrc_mic_intercept",
        "osc.chatbox_include_source",
    }


def test_order24_ui_prompt_clipboard_state_patch_records_initial_covered_surface_list() -> None:
    assert set(ORDER24_UI_PROMPT_CLIPBOARD_STATE_SETTINGS_PATHS) == {
        "secrets.backend",
        "secrets.encrypted_file_path",
        "ui.locale",
        "ui.peer_translation_eula_accepted",
        "ui.integrated_context_bootstrapped",
        "ui.clipboard_auto_translate_enabled",
        "ui.github_star_prompt_clicked",
        "ui.github_star_prompt_last_shown_at",
        "ui.github_star_prompt_show_count",
        "ui.github_star_prompt_translation_success_observed",
        "ui.github_star_prompt_eligible_launch_count",
        "system_prompt",
    }


def test_nondurable_order22_compatibility_fields_are_not_covered() -> None:
    assert {
        "qwen_asr_stt.endpoint",
        "peer_qwen_asr_stt.model",
        "peer_qwen_asr_stt.region",
        "peer_soniox_stt.model",
        "peer_soniox_stt.endpoint",
        "peer_soniox_stt.keepalive_interval_s",
        "peer_soniox_stt.trailing_silence_ms",
    }.isdisjoint(ORDER22_STT_LANGUAGE_AUDIO_SETTINGS_PATHS)


def test_runtime_only_and_nondurable_order23_fields_are_not_covered() -> None:
    assert {
        "ui.overlay_enabled",
        "ui.peer_translation_enabled",
        "active_chatbox_channel",
        "overlay.desktop_flet.locked",
        "overlay.desktop_flet.bounds",
        "overlay.desktop_flet.visual.text_scale",
        "overlay.desktop_flet.visual.outline_width",
        "desktop_audio.output_device",
    }.isdisjoint(ORDER23_OVERLAY_OSC_OUTPUT_SETTINGS_PATHS)


def test_runtime_only_secret_and_legacy_order24_fields_are_not_covered() -> None:
    assert {
        "ui.overlay_enabled",
        "ui.peer_translation_enabled",
        "system_prompts",
        "api_key_verified.openrouter",
        "managed_identity.installation_id",
        "secrets.openrouter_api_key",
        "secrets.deepgram_api_key",
    }.isdisjoint(ORDER24_UI_PROMPT_CLIPBOARD_STATE_SETTINGS_PATHS)


def test_settings_path_patch_builds_typed_mutation_request_for_order21_surface() -> None:
    patch = SettingsPathPatch(
        values_by_path={
            "translation.model": "gemma4",
            "openrouter.selection_alias": "gemma4_byok",
        },
        surface=settings_mutation.SETTINGS_MUTATION_SURFACE_TRANSLATION_PROVIDER,
    )

    request = patch.to_mutation_request(
        expected_revision="settings-r1",
        correlation_id="corr-order21",
    )

    assert request == settings_mutation.SettingsMutationRequest(
        values={
            "translation.model": "gemma4",
            "openrouter.selection_alias": "gemma4_byok",
        },
        expected_revision="settings-r1",
        reason=settings_mutation.SETTINGS_MUTATION_SURFACE_TRANSLATION_PROVIDER,
        correlation_id="corr-order21",
    )


def test_settings_path_patch_builds_typed_mutation_request_for_order22_surface() -> None:
    patch = SettingsPathPatch(
        values_by_path={
            "languages.source_language": "ja",
            "audio.input_device": "Headset Mic",
        },
        surface=settings_mutation.SETTINGS_MUTATION_SURFACE_STT_LANGUAGE_AUDIO,
    )

    request = patch.to_mutation_request(
        expected_revision="settings-r2",
        correlation_id="corr-order22",
    )

    assert request == settings_mutation.SettingsMutationRequest(
        values={
            "languages.source_language": "ja",
            "audio.input_device": "Headset Mic",
        },
        expected_revision="settings-r2",
        reason=settings_mutation.SETTINGS_MUTATION_SURFACE_STT_LANGUAGE_AUDIO,
        correlation_id="corr-order22",
    )


def test_settings_path_patch_builds_typed_mutation_request_for_order23_surface() -> None:
    patch = SettingsPathPatch(
        values_by_path={
            "overlay.show_translation": False,
            "overlay.desktop_flet.size_preset": "large",
            "osc.chatbox_max_chars": 120,
        },
        surface=settings_mutation.SETTINGS_MUTATION_SURFACE_OVERLAY_OSC_OUTPUT,
    )

    request = patch.to_mutation_request(
        expected_revision="settings-r3",
        correlation_id="corr-order23",
    )

    assert request == settings_mutation.SettingsMutationRequest(
        values={
            "overlay.show_translation": False,
            "overlay.desktop_flet.size_preset": "large",
            "osc.chatbox_max_chars": 120,
        },
        expected_revision="settings-r3",
        reason=settings_mutation.SETTINGS_MUTATION_SURFACE_OVERLAY_OSC_OUTPUT,
        correlation_id="corr-order23",
    )


def test_settings_path_patch_builds_typed_mutation_request_for_order24_surface() -> None:
    patch = SettingsPathPatch(
        values_by_path={
            "ui.locale": "ja",
            "ui.clipboard_auto_translate_enabled": True,
            "system_prompt": "custom translation style",
        },
        surface=settings_mutation.SETTINGS_MUTATION_SURFACE_UI_PROMPT_CLIPBOARD_STATE,
    )

    request = patch.to_mutation_request(
        expected_revision="settings-r4",
        correlation_id="corr-order24",
    )

    assert request == settings_mutation.SettingsMutationRequest(
        values={
            "ui.locale": "ja",
            "ui.clipboard_auto_translate_enabled": True,
            "system_prompt": "custom translation style",
        },
        expected_revision="settings-r4",
        reason=settings_mutation.SETTINGS_MUTATION_SURFACE_UI_PROMPT_CLIPBOARD_STATE,
        correlation_id="corr-order24",
    )


@pytest.mark.asyncio
async def test_order21_path_validator_accepts_only_translation_provider_paths() -> None:
    validator = SettingsPathMutationValidator(
        allowed_paths=ORDER21_TRANSLATION_PROVIDER_SETTINGS_PATHS,
        component="settings_mutation",
        operation="validate_translation_provider_patch",
    )
    request = settings_mutation.SettingsMutationRequest(
        values={
            "translation.connection": "openrouter",
            "translation.fallback": {
                "enabled": True,
                "model": "deepseek_v4_flash",
                "connection": "openrouter",
            },
            "local_llm.base_url": "http://127.0.0.1:11434/v1",
            "llm.concurrency_limit": 3,
        },
        expected_revision=None,
        reason=settings_mutation.SETTINGS_MUTATION_SURFACE_TRANSLATION_PROVIDER,
        correlation_id="corr-valid-paths",
    )

    result = await validator.validate(request)

    assert result == settings_mutation.SettingsMutationValidationResult(
        succeeded=True,
        message=None,
        diagnostics=None,
    )


@pytest.mark.asyncio
async def test_order21_path_validator_rejects_out_of_scope_paths_without_secret_values() -> None:
    validator = SettingsPathMutationValidator(
        allowed_paths=ORDER21_TRANSLATION_PROVIDER_SETTINGS_PATHS,
        component="settings_mutation",
        operation="validate_translation_provider_patch",
    )
    request = settings_mutation.SettingsMutationRequest(
        values={
            "stt.low_latency_mode": False,
            "audio.input_device": "default microphone",
            "overlay.target": "desktop",
            "secrets.openrouter_api_key": "secret-value-must-not-leak",
        },
        expected_revision=None,
        reason=settings_mutation.SETTINGS_MUTATION_SURFACE_TRANSLATION_PROVIDER,
        correlation_id="corr-invalid-paths",
    )

    result = await validator.validate(request)

    assert result.succeeded is False
    assert result.message is None
    assert result.diagnostics == messages.ErrorDiagnostics(
        component="settings_mutation",
        operation="validate_translation_provider_patch",
        code="settings_path_not_covered",
        category=messages.DIAGNOSTIC_CATEGORY_TRANSACTION,
        visibility=messages.DIAGNOSTIC_VISIBILITY_BASIC,
        content_policy=messages.CONTENT_POLICY_METADATA_ONLY,
        status_code=None,
        retry_after_ms=None,
        fields={"path": "audio.input_device"},
    )
    assert "secret-value-must-not-leak" not in repr(result)


@pytest.mark.asyncio
async def test_order22_path_validator_accepts_only_stt_language_audio_paths() -> None:
    validator = SettingsPathMutationValidator(
        allowed_paths=ORDER22_STT_LANGUAGE_AUDIO_SETTINGS_PATHS,
        component="settings_mutation",
        operation="validate_stt_language_audio_patch",
    )
    request = settings_mutation.SettingsMutationRequest(
        values={
            "provider.stt": "soniox",
            "provider.peer_stt": "local_qwen",
            "languages.source_language": "ja",
            "audio.input_device": "Headset Mic",
            "desktop_audio.vad_hangover_ms": 900,
            "soniox_stt.trailing_silence_ms": 150,
        },
        expected_revision=None,
        reason=settings_mutation.SETTINGS_MUTATION_SURFACE_STT_LANGUAGE_AUDIO,
        correlation_id="corr-valid-order22-paths",
    )

    result = await validator.validate(request)

    assert result == settings_mutation.SettingsMutationValidationResult(
        succeeded=True,
        message=None,
        diagnostics=None,
    )


@pytest.mark.asyncio
async def test_order22_path_validator_rejects_order21_overlay_and_secret_paths_without_values() -> (
    None
):
    validator = SettingsPathMutationValidator(
        allowed_paths=ORDER22_STT_LANGUAGE_AUDIO_SETTINGS_PATHS,
        component="settings_mutation",
        operation="validate_stt_language_audio_patch",
    )
    request = settings_mutation.SettingsMutationRequest(
        values={
            "translation.model": "gemma4-secret-ish",
            "openrouter.selection_alias": "managed-secret-ish",
            "overlay.target": "desktop-secret-ish",
            "secrets.deepgram_api_key": "secret-value-must-not-leak",
        },
        expected_revision=None,
        reason=settings_mutation.SETTINGS_MUTATION_SURFACE_STT_LANGUAGE_AUDIO,
        correlation_id="corr-invalid-order22-paths",
    )

    result = await validator.validate(request)

    assert result.succeeded is False
    assert result.message is None
    assert result.diagnostics == messages.ErrorDiagnostics(
        component="settings_mutation",
        operation="validate_stt_language_audio_patch",
        code="settings_path_not_covered",
        category=messages.DIAGNOSTIC_CATEGORY_TRANSACTION,
        visibility=messages.DIAGNOSTIC_VISIBILITY_BASIC,
        content_policy=messages.CONTENT_POLICY_METADATA_ONLY,
        status_code=None,
        retry_after_ms=None,
        fields={"path": "openrouter.selection_alias"},
    )
    assert "secret-value-must-not-leak" not in repr(result)
    assert "gemma4-secret-ish" not in repr(result)


@pytest.mark.asyncio
async def test_order23_path_validator_accepts_only_overlay_osc_output_paths() -> None:
    validator = SettingsPathMutationValidator(
        allowed_paths=ORDER23_OVERLAY_OSC_OUTPUT_SETTINGS_PATHS,
        component="settings_mutation",
        operation="validate_overlay_osc_output_patch",
    )
    request = settings_mutation.SettingsMutationRequest(
        values={
            "overlay.target": "desktop",
            "overlay.calibration.distance": 1.4,
            "overlay.desktop_flet.position.x": 24,
            "overlay.desktop_flet.visual.background_alpha": 0.45,
            "osc.host": "127.0.0.1",
            "osc.port": 9001,
            "osc.chatbox_max_chars": 120,
            "osc.chatbox_include_source": True,
        },
        expected_revision=None,
        reason=settings_mutation.SETTINGS_MUTATION_SURFACE_OVERLAY_OSC_OUTPUT,
        correlation_id="corr-valid-order23-paths",
    )

    result = await validator.validate(request)

    assert result == settings_mutation.SettingsMutationValidationResult(
        succeeded=True,
        message=None,
        diagnostics=None,
    )


@pytest.mark.asyncio
async def test_order23_path_validator_rejects_runtime_only_peer_and_secret_paths_without_values() -> (
    None
):
    validator = SettingsPathMutationValidator(
        allowed_paths=ORDER23_OVERLAY_OSC_OUTPUT_SETTINGS_PATHS,
        component="settings_mutation",
        operation="validate_overlay_osc_output_patch",
    )
    request = settings_mutation.SettingsMutationRequest(
        values={
            "active_chatbox_channel": "peer-secret-ish",
            "ui.overlay_enabled": True,
            "ui.peer_translation_enabled": True,
            "secrets.openrouter_api_key": "secret-value-must-not-leak",
        },
        expected_revision=None,
        reason=settings_mutation.SETTINGS_MUTATION_SURFACE_OVERLAY_OSC_OUTPUT,
        correlation_id="corr-invalid-order23-paths",
    )

    result = await validator.validate(request)

    assert result.succeeded is False
    assert result.message is None
    assert result.diagnostics == messages.ErrorDiagnostics(
        component="settings_mutation",
        operation="validate_overlay_osc_output_patch",
        code="settings_path_not_covered",
        category=messages.DIAGNOSTIC_CATEGORY_TRANSACTION,
        visibility=messages.DIAGNOSTIC_VISIBILITY_BASIC,
        content_policy=messages.CONTENT_POLICY_METADATA_ONLY,
        status_code=None,
        retry_after_ms=None,
        fields={"path": "active_chatbox_channel"},
    )
    assert "secret-value-must-not-leak" not in repr(result)
    assert "peer-secret-ish" not in repr(result)


@pytest.mark.asyncio
async def test_order24_path_validator_accepts_only_ui_prompt_clipboard_state_paths() -> None:
    validator = SettingsPathMutationValidator(
        allowed_paths=ORDER24_UI_PROMPT_CLIPBOARD_STATE_SETTINGS_PATHS,
        component="settings_mutation",
        operation="validate_ui_prompt_clipboard_state_patch",
    )
    request = settings_mutation.SettingsMutationRequest(
        values={
            "secrets.backend": "encrypted_file",
            "secrets.encrypted_file_path": "secure-secrets.json",
            "ui.locale": "ja",
            "ui.peer_translation_eula_accepted": True,
            "ui.integrated_context_bootstrapped": True,
            "ui.clipboard_auto_translate_enabled": True,
            "ui.github_star_prompt_clicked": False,
            "ui.github_star_prompt_last_shown_at": "2026-06-08T00:00:00Z",
            "ui.github_star_prompt_show_count": 2,
            "ui.github_star_prompt_translation_success_observed": True,
            "ui.github_star_prompt_eligible_launch_count": 3,
            "system_prompt": "custom translation style",
        },
        expected_revision=None,
        reason=settings_mutation.SETTINGS_MUTATION_SURFACE_UI_PROMPT_CLIPBOARD_STATE,
        correlation_id="corr-valid-order24-paths",
    )

    result = await validator.validate(request)

    assert result == settings_mutation.SettingsMutationValidationResult(
        succeeded=True,
        message=None,
        diagnostics=None,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("allowed_paths", "operation", "path"),
    [
        (
            ORDER22_STT_LANGUAGE_AUDIO_SETTINGS_PATHS,
            "validate_stt_language_audio_patch",
            "stt.low_latency_mode",
        ),
        (
            ORDER24_UI_PROMPT_CLIPBOARD_STATE_SETTINGS_PATHS,
            "validate_ui_prompt_clipboard_state_patch",
            "ui.integrated_context_enabled",
        ),
    ],
)
async def test_retired_policy_paths_are_rejected(
    allowed_paths: tuple[str, ...],
    operation: str,
    path: str,
) -> None:
    validator = SettingsPathMutationValidator(
        allowed_paths=allowed_paths,
        component="settings_mutation",
        operation=operation,
    )

    result = await validator.validate(
        settings_mutation.SettingsMutationRequest(
            values={path: False},
            expected_revision=None,
            reason="retired_policy",
            correlation_id="corr-retired-policy",
        )
    )

    assert result.succeeded is False
    assert result.diagnostics is not None
    assert result.diagnostics.code == "settings_path_not_covered"
    assert result.diagnostics.fields == {"path": path}


@pytest.mark.asyncio
async def test_order24_path_validator_rejects_runtime_secret_and_legacy_paths_without_values() -> (
    None
):
    validator = SettingsPathMutationValidator(
        allowed_paths=ORDER24_UI_PROMPT_CLIPBOARD_STATE_SETTINGS_PATHS,
        component="settings_mutation",
        operation="validate_ui_prompt_clipboard_state_patch",
    )
    request = settings_mutation.SettingsMutationRequest(
        values={
            "api_key_verified.openrouter": True,
            "managed_identity.installation_id": "device-secret-ish",
            "system_prompts": {"openrouter": "prompt-secret-ish"},
            "ui.overlay_enabled": True,
            "ui.peer_translation_enabled": True,
            "secrets.openrouter_api_key": "secret-value-must-not-leak",
        },
        expected_revision=None,
        reason=settings_mutation.SETTINGS_MUTATION_SURFACE_UI_PROMPT_CLIPBOARD_STATE,
        correlation_id="corr-invalid-order24-paths",
    )

    result = await validator.validate(request)

    assert result.succeeded is False
    assert result.message is None
    assert result.diagnostics == messages.ErrorDiagnostics(
        component="settings_mutation",
        operation="validate_ui_prompt_clipboard_state_patch",
        code="settings_path_not_covered",
        category=messages.DIAGNOSTIC_CATEGORY_TRANSACTION,
        visibility=messages.DIAGNOSTIC_VISIBILITY_BASIC,
        content_policy=messages.CONTENT_POLICY_METADATA_ONLY,
        status_code=None,
        retry_after_ms=None,
        fields={"path": "api_key_verified.openrouter"},
    )
    assert "secret-value-must-not-leak" not in repr(result)
    assert "device-secret-ish" not in repr(result)
    assert "prompt-secret-ish" not in repr(result)
