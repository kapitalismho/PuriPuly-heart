from __future__ import annotations

from dataclasses import replace

import pytest
from puripuly_heart.app.services.settings_mutation_legacy import (
    ORDER21_TRANSLATION_PROVIDER_SETTINGS_PATHS,
    ORDER22_STT_LANGUAGE_AUDIO_SETTINGS_PATHS,
    ORDER23_OVERLAY_OSC_OUTPUT_SETTINGS_PATHS,
    ORDER24_UI_PROMPT_CLIPBOARD_STATE_SETTINGS_PATHS,
    SettingsPathMutationValidator,
    SettingsPathPatch,
    build_stt_language_audio_settings_path_patch,
    build_translation_provider_settings_path_patch,
)

from puripuly_heart.app.services import settings_mutation
from puripuly_heart.config.settings_vnext.schema import AppSettingsVNext
from puripuly_heart.core import messages


def test_order21_translation_provider_patch_records_initial_covered_surface_list() -> None:
    assert set(ORDER21_TRANSLATION_PROVIDER_SETTINGS_PATHS) == {
        "intent.translation.model",
        "intent.translation.connection",
        "intent.translation.connection_history",
        "intent.translation.fallback",
        "intent.translation.http_extension_id",
        "intent.translation.previous_llm_model",
        "intent.translation.gpu_device_id",
        "intent.translation.gemini.llm_model",
        "intent.translation.openrouter_model",
        "intent.translation.openrouter_routing_mode",
        "intent.translation.openrouter_provider_routing",
        "intent.translation.openrouter_selected_source",
        "intent.translation.openrouter_selection_alias",
        "intent.translation.openrouter_broker_base_url",
        "intent.translation.qwen.llm_model",
        "intent.translation.qwen.region",
        "intent.translation.deepseek.llm_model",
        "intent.local_llm.backend",
        "intent.local_llm.base_url",
        "intent.local_llm.model",
        "intent.local_llm.extra_body",
        "intent.translation.concurrency_limit",
    }


def test_order21_patch_carries_custom_http_identity_fields() -> None:
    previous = AppSettingsVNext()
    next_settings = replace(
        previous,
        intent=replace(
            previous.intent,
            translation=replace(
                previous.intent.translation,
                model="custom_http",
                connection="custom_http",
                http_extension_id="demo",
                previous_llm_model="gemma4_26b_31b",
            ),
        ),
    )

    patch = build_translation_provider_settings_path_patch(previous, next_settings)

    assert patch["intent.translation.http_extension_id"] == "demo"
    assert patch["intent.translation.previous_llm_model"] == "gemma4_26b_31b"


def test_order22_patch_carries_a_secondary_only_target_change() -> None:
    previous = AppSettingsVNext()
    next_settings = replace(
        previous,
        intent=replace(
            previous.intent,
            languages=replace(previous.intent.languages, secondary_target_language="ja"),
        ),
    )

    patch = build_stt_language_audio_settings_path_patch(previous, next_settings)

    assert patch == {"intent.languages.secondary_target_language": "ja"}


def test_order22_stt_language_audio_patch_records_initial_covered_surface_list() -> None:
    assert set(ORDER22_STT_LANGUAGE_AUDIO_SETTINGS_PATHS) == {
        "intent.stt.provider",
        "intent.peer_stt.provider",
        "intent.languages.source_language",
        "intent.languages.target_language",
        "intent.languages.secondary_target_language",
        "intent.languages.peer_source_language",
        "intent.languages.peer_target_language",
        "intent.languages.peer_source_mode",
        "intent.languages.peer_expected_languages",
        "intent.languages.recent_source_languages",
        "intent.languages.recent_target_languages",
        "intent.audio.ring_buffer_ms",
        "intent.audio.input_host_api",
        "intent.audio.input_device",
        "intent.desktop_audio.output_device",
        "intent.desktop_audio.vad_speech_threshold",
        "intent.desktop_audio.vad_hangover_ms",
        "intent.desktop_audio.vad_pre_roll_ms",
        "intent.stt.drain_timeout_s",
        "intent.stt.vad_speech_threshold",
        "intent.stt.low_latency_vad_hangover_ms",
        "intent.stt.low_latency_merge_gap_ms",
        "intent.stt.low_latency_spec_retry_max",
        "intent.stt.custom_vocabulary_enabled",
        "intent.stt.custom_terms",
        "intent.stt.gpu_device_id",
        "intent.stt.deepgram.model",
        "intent.stt.rolling_enabled",
        "intent.stt.gemini_transcribe.model",
        "intent.stt.elevenlabs_scribe.model",
        "intent.peer_stt.rolling_enabled",
        "intent.stt.qwen_asr.model",
        "intent.stt.soniox.model",
        "intent.stt.soniox.endpoint",
        "intent.stt.soniox.keepalive_interval_s",
        "intent.stt.soniox.trailing_silence_ms",
        "intent.stt.custom.mode",
        "intent.stt.custom.compatibility",
        "intent.stt.custom.endpoint",
        "intent.stt.custom.model",
        "intent.stt.custom.extra",
    }


def test_order23_overlay_osc_output_patch_records_initial_covered_surface_list() -> None:
    assert set(ORDER23_OVERLAY_OSC_OUTPUT_SETTINGS_PATHS) == {
        "intent.overlay.target",
        "intent.overlay.show_translation",
        "intent.overlay.show_peer_original",
        "intent.overlay.calibration.anchor",
        "intent.overlay.calibration.offset_x",
        "intent.overlay.calibration.offset_y",
        "intent.overlay.calibration.distance",
        "intent.overlay.calibration.text_scale",
        "intent.overlay.calibration.background_alpha",
        "intent.overlay.desktop_flet.size_preset",
        "intent.overlay.desktop_flet.position.x",
        "intent.overlay.desktop_flet.position.y",
        "intent.overlay.desktop_flet.swap_caption_languages",
        "intent.overlay.desktop_flet.visual.background_alpha",
        "intent.osc.host",
        "intent.osc.port",
        "intent.osc.connection_mode",
        "intent.osc.send_port",
        "intent.osc.receive_port",
        "intent.osc.chatbox_address",
        "intent.osc.chatbox_send",
        "intent.osc.chatbox_clear",
        "intent.osc.chatbox_max_chars",
        "intent.osc.vrc_mic_intercept",
        "intent.osc.chatbox_include_source",
    }


def test_order24_ui_prompt_clipboard_state_patch_records_initial_covered_surface_list() -> None:
    assert set(ORDER24_UI_PROMPT_CLIPBOARD_STATE_SETTINGS_PATHS) == {
        "intent.secrets.backend",
        "intent.secrets.encrypted_file_path",
        "intent.ui.locale",
        "state.peer_translation.eula_accepted",
        "state.integrated_context.bootstrapped",
        "intent.clipboard.auto_translate_enabled",
        "state.github_star_prompt.clicked",
        "state.github_star_prompt.last_shown_at",
        "state.github_star_prompt.show_count",
        "state.github_star_prompt.translation_success_observed",
        "state.github_star_prompt.eligible_launch_count",
        "intent.prompts.system_prompt",
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
            "intent.translation.model": "gemma4",
            "intent.translation.openrouter_selection_alias": "gemma4_byok",
        },
        surface=settings_mutation.SETTINGS_MUTATION_SURFACE_TRANSLATION_PROVIDER,
    )

    request = patch.to_mutation_request(
        expected_revision="settings-r1",
        correlation_id="corr-order21",
    )

    assert request == settings_mutation.SettingsMutationRequest(
        values={
            "intent.translation.model": "gemma4",
            "intent.translation.openrouter_selection_alias": "gemma4_byok",
        },
        expected_revision="settings-r1",
        reason=settings_mutation.SETTINGS_MUTATION_SURFACE_TRANSLATION_PROVIDER,
        correlation_id="corr-order21",
    )


def test_settings_path_patch_builds_typed_mutation_request_for_order22_surface() -> None:
    patch = SettingsPathPatch(
        values_by_path={
            "intent.languages.source_language": "ja",
            "intent.audio.input_device": "Headset Mic",
        },
        surface=settings_mutation.SETTINGS_MUTATION_SURFACE_STT_LANGUAGE_AUDIO,
    )

    request = patch.to_mutation_request(
        expected_revision="settings-r2",
        correlation_id="corr-order22",
    )

    assert request == settings_mutation.SettingsMutationRequest(
        values={
            "intent.languages.source_language": "ja",
            "intent.audio.input_device": "Headset Mic",
        },
        expected_revision="settings-r2",
        reason=settings_mutation.SETTINGS_MUTATION_SURFACE_STT_LANGUAGE_AUDIO,
        correlation_id="corr-order22",
    )


def test_settings_path_patch_builds_typed_mutation_request_for_order23_surface() -> None:
    patch = SettingsPathPatch(
        values_by_path={
            "intent.overlay.show_translation": False,
            "intent.overlay.desktop_flet.size_preset": "large",
            "intent.osc.chatbox_max_chars": 120,
        },
        surface=settings_mutation.SETTINGS_MUTATION_SURFACE_OVERLAY_OSC_OUTPUT,
    )

    request = patch.to_mutation_request(
        expected_revision="settings-r3",
        correlation_id="corr-order23",
    )

    assert request == settings_mutation.SettingsMutationRequest(
        values={
            "intent.overlay.show_translation": False,
            "intent.overlay.desktop_flet.size_preset": "large",
            "intent.osc.chatbox_max_chars": 120,
        },
        expected_revision="settings-r3",
        reason=settings_mutation.SETTINGS_MUTATION_SURFACE_OVERLAY_OSC_OUTPUT,
        correlation_id="corr-order23",
    )


def test_settings_path_patch_builds_typed_mutation_request_for_order24_surface() -> None:
    patch = SettingsPathPatch(
        values_by_path={
            "intent.ui.locale": "ja",
            "intent.clipboard.auto_translate_enabled": True,
            "intent.prompts.system_prompt": "custom translation style",
        },
        surface=settings_mutation.SETTINGS_MUTATION_SURFACE_UI_PROMPT_CLIPBOARD_STATE,
    )

    request = patch.to_mutation_request(
        expected_revision="settings-r4",
        correlation_id="corr-order24",
    )

    assert request == settings_mutation.SettingsMutationRequest(
        values={
            "intent.ui.locale": "ja",
            "intent.clipboard.auto_translate_enabled": True,
            "intent.prompts.system_prompt": "custom translation style",
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
            "intent.translation.connection": "openrouter",
            "intent.translation.fallback": {
                "enabled": True,
                "model": "deepseek_v4_flash",
                "connection": "openrouter",
            },
            "intent.local_llm.base_url": "http://127.0.0.1:11434/v1",
            "intent.translation.concurrency_limit": 3,
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
            "intent.stt.provider": "soniox",
            "intent.peer_stt.provider": "local_qwen",
            "intent.languages.source_language": "ja",
            "intent.languages.secondary_target_language": "fr",
            "intent.audio.input_device": "Headset Mic",
            "intent.desktop_audio.vad_hangover_ms": 900,
            "intent.stt.soniox.trailing_silence_ms": 150,
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
            "intent.overlay.target": "desktop",
            "intent.overlay.calibration.distance": 1.4,
            "intent.overlay.desktop_flet.position.x": 24,
            "intent.overlay.desktop_flet.visual.background_alpha": 0.45,
            "intent.osc.host": "127.0.0.1",
            "intent.osc.port": 9001,
            "intent.osc.chatbox_max_chars": 120,
            "intent.osc.chatbox_include_source": True,
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
            "intent.secrets.backend": "encrypted_file",
            "intent.secrets.encrypted_file_path": "secure-secrets.json",
            "intent.ui.locale": "ja",
            "state.peer_translation.eula_accepted": True,
            "state.integrated_context.bootstrapped": True,
            "intent.clipboard.auto_translate_enabled": True,
            "state.github_star_prompt.clicked": False,
            "state.github_star_prompt.last_shown_at": "2026-06-08T00:00:00Z",
            "state.github_star_prompt.show_count": 2,
            "state.github_star_prompt.translation_success_observed": True,
            "state.github_star_prompt.eligible_launch_count": 3,
            "intent.prompts.system_prompt": "custom translation style",
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
