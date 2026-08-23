from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

from puripuly_heart.app.ports._settings_values import freeze_settings_values
from puripuly_heart.core.messages import (
    CONTENT_POLICY_METADATA_ONLY,
    DIAGNOSTIC_CATEGORY_TRANSACTION,
    DIAGNOSTIC_VISIBILITY_BASIC,
    ErrorDiagnostics,
)

from .settings_mutation import (
    SETTINGS_MUTATION_SURFACE_OVERLAY_OSC_OUTPUT,
    SETTINGS_MUTATION_SURFACE_STT_LANGUAGE_AUDIO,
    SETTINGS_MUTATION_SURFACE_TRANSLATION_PROVIDER,
    SETTINGS_MUTATION_SURFACE_UI_PROMPT_CLIPBOARD_STATE,
    SettingsMutationCommand,
    SettingsMutationRequest,
    SettingsMutationValidationResult,
)

ORDER21_TRANSLATION_PROVIDER_SETTINGS_PATHS: Final[tuple[str, ...]] = (
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
)

ORDER22_STT_LANGUAGE_AUDIO_SETTINGS_PATHS: Final[tuple[str, ...]] = (
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
)

ORDER23_OVERLAY_OSC_OUTPUT_SETTINGS_PATHS: Final[tuple[str, ...]] = (
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
)

ORDER24_UI_PROMPT_CLIPBOARD_STATE_SETTINGS_PATHS: Final[tuple[str, ...]] = (
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
)

_SURFACE_ALLOWED_PATHS: Final[dict[str, tuple[str, ...]]] = {
    SETTINGS_MUTATION_SURFACE_TRANSLATION_PROVIDER: ORDER21_TRANSLATION_PROVIDER_SETTINGS_PATHS,
    SETTINGS_MUTATION_SURFACE_STT_LANGUAGE_AUDIO: ORDER22_STT_LANGUAGE_AUDIO_SETTINGS_PATHS,
    SETTINGS_MUTATION_SURFACE_OVERLAY_OSC_OUTPUT: ORDER23_OVERLAY_OSC_OUTPUT_SETTINGS_PATHS,
    SETTINGS_MUTATION_SURFACE_UI_PROMPT_CLIPBOARD_STATE: ORDER24_UI_PROMPT_CLIPBOARD_STATE_SETTINGS_PATHS,
}

_SURFACE_VALIDATOR_OPERATION: Final[dict[str, str]] = {
    SETTINGS_MUTATION_SURFACE_TRANSLATION_PROVIDER: "validate_translation_provider_patch",
    SETTINGS_MUTATION_SURFACE_STT_LANGUAGE_AUDIO: "validate_stt_language_audio_patch",
    SETTINGS_MUTATION_SURFACE_OVERLAY_OSC_OUTPUT: "validate_overlay_osc_output_patch",
    SETTINGS_MUTATION_SURFACE_UI_PROMPT_CLIPBOARD_STATE: "validate_ui_prompt_clipboard_state_patch",
}


@dataclass(frozen=True, slots=True)
class SettingsPathPatch:
    values_by_path: Mapping[str, object]
    surface: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "values_by_path",
            freeze_settings_values(self.values_by_path),
        )

    def to_mutation_request(
        self,
        *,
        expected_revision: str | None,
        correlation_id: str | None,
    ) -> SettingsMutationRequest:
        return SettingsMutationRequest(
            values=self.values_by_path,
            expected_revision=expected_revision,
            reason=self.surface,
            correlation_id=correlation_id,
        )


@dataclass(frozen=True, slots=True)
class SettingsPathMutationValidator:
    allowed_paths: tuple[str, ...]
    component: str
    operation: str | None

    def __init__(
        self,
        *,
        allowed_paths: tuple[str, ...],
        component: str,
        operation: str | None,
    ) -> None:
        object.__setattr__(self, "allowed_paths", tuple(allowed_paths))
        object.__setattr__(self, "component", component)
        object.__setattr__(self, "operation", operation)

    async def validate(
        self,
        request: SettingsMutationRequest,
    ) -> SettingsMutationValidationResult:
        allowed = frozenset(self.allowed_paths)
        disallowed_paths = sorted(
            str(path) for path in request.values if not isinstance(path, str) or path not in allowed
        )
        if disallowed_paths:
            return SettingsMutationValidationResult(
                succeeded=False,
                message=None,
                diagnostics=ErrorDiagnostics(
                    component=self.component,
                    operation=self.operation,
                    code="settings_path_not_covered",
                    category=DIAGNOSTIC_CATEGORY_TRANSACTION,
                    visibility=DIAGNOSTIC_VISIBILITY_BASIC,
                    content_policy=CONTENT_POLICY_METADATA_ONLY,
                    status_code=None,
                    retry_after_ms=None,
                    fields={"path": disallowed_paths[0]},
                ),
            )
        return SettingsMutationValidationResult(
            succeeded=True,
            message=None,
            diagnostics=None,
        )


def settings_path_patch_from_command(
    command: SettingsMutationCommand,
) -> SettingsPathPatch:
    return SettingsPathPatch(
        values_by_path=command.values,
        surface=command.surface,
    )


def settings_path_mutation_validator_for_command(
    command: SettingsMutationCommand,
) -> SettingsPathMutationValidator:
    surface = command.surface
    allowed_paths = _SURFACE_ALLOWED_PATHS[surface]
    operation = _SURFACE_VALIDATOR_OPERATION[surface]
    return SettingsPathMutationValidator(
        allowed_paths=allowed_paths,
        component="settings_mutation",
        operation=operation,
    )


@dataclass(frozen=True, slots=True)
class _SettingsPathSnapshot:
    values_by_path: tuple[tuple[str, object], ...]

    @classmethod
    def from_settings(
        cls,
        settings: object,
        *,
        paths: tuple[str, ...],
    ) -> _SettingsPathSnapshot:
        return cls(tuple((path, _get_settings_path_value(settings, path)) for path in paths))

    def patch_to(self, settings: object) -> dict[str, object]:
        patch: dict[str, object] = {}
        for path, previous_value in self.values_by_path:
            next_value = _get_settings_path_value(settings, path)
            if previous_value != next_value:
                patch[path] = next_value
        return patch

    def materialize_base_from(self, settings: object) -> object:
        base_settings = copy.deepcopy(settings)
        for path, previous_value in self.values_by_path:
            _set_settings_path_value(base_settings, path, previous_value)
        return base_settings


def settings_path_snapshot_for_stt_language_audio(settings: object) -> _SettingsPathSnapshot:
    return _SettingsPathSnapshot.from_settings(
        settings, paths=ORDER22_STT_LANGUAGE_AUDIO_SETTINGS_PATHS
    )


def settings_path_snapshot_for_overlay_osc_output(settings: object) -> _SettingsPathSnapshot:
    return _SettingsPathSnapshot.from_settings(
        settings, paths=ORDER23_OVERLAY_OSC_OUTPUT_SETTINGS_PATHS
    )


def settings_path_snapshot_for_ui_prompt_clipboard_state(
    settings: object,
) -> _SettingsPathSnapshot:
    return _SettingsPathSnapshot.from_settings(
        settings, paths=ORDER24_UI_PROMPT_CLIPBOARD_STATE_SETTINGS_PATHS
    )


def build_translation_provider_settings_path_patch(
    previous: object,
    next_settings: object,
) -> dict[str, object]:
    return _build_settings_path_patch(
        previous,
        next_settings,
        paths=ORDER21_TRANSLATION_PROVIDER_SETTINGS_PATHS,
    )


def build_stt_language_audio_settings_path_patch(
    previous: object,
    next_settings: object,
) -> dict[str, object]:
    return _build_settings_path_patch(
        previous,
        next_settings,
        paths=ORDER22_STT_LANGUAGE_AUDIO_SETTINGS_PATHS,
    )


def build_overlay_osc_output_settings_path_patch(
    previous: object,
    next_settings: object,
) -> dict[str, object]:
    return _build_settings_path_patch(
        previous,
        next_settings,
        paths=ORDER23_OVERLAY_OSC_OUTPUT_SETTINGS_PATHS,
    )


def build_ui_prompt_clipboard_state_settings_path_patch(
    previous: object,
    next_settings: object,
) -> dict[str, object]:
    return _build_settings_path_patch(
        previous,
        next_settings,
        paths=ORDER24_UI_PROMPT_CLIPBOARD_STATE_SETTINGS_PATHS,
    )


def _get_settings_path_value(settings: object, path: str) -> object:
    current: object = settings
    for segment in path.split("."):
        current = getattr(current, segment)
    return copy.deepcopy(current)


def _set_settings_path_value(settings: object, path: str, value: object) -> None:
    current: object = settings
    segments = path.split(".")
    for segment in segments[:-1]:
        current = getattr(current, segment)
    setattr(current, segments[-1], _mutable_settings_value(value))


def _build_settings_path_patch(
    previous: object,
    next_settings: object,
    *,
    paths: tuple[str, ...],
) -> dict[str, object]:
    patch: dict[str, object] = {}
    for path in paths:
        previous_value = _get_settings_path_value(previous, path)
        next_value = _get_settings_path_value(next_settings, path)
        if previous_value != next_value:
            patch[path] = next_value
    return patch


def _apply_settings_path_patch(settings: object, patch: Mapping[str, object]) -> None:
    for path, value in patch.items():
        _set_settings_path_value(settings, path, value)


def _mutable_settings_value(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _mutable_settings_value(nested) for key, nested in value.items()}
    if isinstance(value, tuple):
        return [_mutable_settings_value(item) for item in value]
    if isinstance(value, list):
        return [_mutable_settings_value(item) for item in value]
    return copy.deepcopy(value)


__all__ = [
    "ORDER21_TRANSLATION_PROVIDER_SETTINGS_PATHS",
    "ORDER22_STT_LANGUAGE_AUDIO_SETTINGS_PATHS",
    "ORDER23_OVERLAY_OSC_OUTPUT_SETTINGS_PATHS",
    "ORDER24_UI_PROMPT_CLIPBOARD_STATE_SETTINGS_PATHS",
    "SettingsPathMutationValidator",
    "SettingsPathPatch",
    "_SettingsPathSnapshot",
    "_apply_settings_path_patch",
    "_build_settings_path_patch",
    "_get_settings_path_value",
    "_mutable_settings_value",
    "_set_settings_path_value",
    "build_overlay_osc_output_settings_path_patch",
    "build_stt_language_audio_settings_path_patch",
    "build_translation_provider_settings_path_patch",
    "build_ui_prompt_clipboard_state_settings_path_patch",
    "settings_path_mutation_validator_for_command",
    "settings_path_patch_from_command",
    "settings_path_snapshot_for_overlay_osc_output",
    "settings_path_snapshot_for_stt_language_audio",
    "settings_path_snapshot_for_ui_prompt_clipboard_state",
]
