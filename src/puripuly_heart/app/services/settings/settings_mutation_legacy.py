from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

from puripuly_heart.app.ports._settings_values import freeze_settings_values
from puripuly_heart.config.settings_vnext.schema import AppSettingsVNext
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
)

ORDER22_STT_LANGUAGE_AUDIO_SETTINGS_PATHS: Final[tuple[str, ...]] = (
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
    "intent.stt.cloud_free_tier_providers",
    "intent.stt.deepgram.model",
    "intent.stt.gemini_transcribe.model",
    "intent.stt.elevenlabs_scribe.model",
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
)

ORDER23_OVERLAY_OSC_OUTPUT_SETTINGS_PATHS: Final[tuple[str, ...]] = (
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
)

ORDER24_UI_PROMPT_CLIPBOARD_STATE_SETTINGS_PATHS: Final[tuple[str, ...]] = (
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
        settings: AppSettingsVNext,
        *,
        paths: tuple[str, ...],
    ) -> _SettingsPathSnapshot:
        return cls(tuple((path, _get_settings_path_value(settings, path)) for path in paths))

    def patch_to(self, settings: AppSettingsVNext) -> dict[str, object]:
        patch: dict[str, object] = {}
        for path, previous_value in self.values_by_path:
            next_value = _get_settings_path_value(settings, path)
            if previous_value != next_value:
                patch[path] = next_value
        return patch

    def materialize_base_from(self, settings: AppSettingsVNext) -> AppSettingsVNext:
        patch = {path: previous_value for path, previous_value in self.values_by_path}
        return apply_settings_path_patch(settings, patch)


def settings_path_snapshot_for_stt_language_audio(
    settings: AppSettingsVNext,
) -> _SettingsPathSnapshot:
    return _SettingsPathSnapshot.from_settings(
        settings, paths=ORDER22_STT_LANGUAGE_AUDIO_SETTINGS_PATHS
    )


def settings_path_snapshot_for_overlay_osc_output(
    settings: AppSettingsVNext,
) -> _SettingsPathSnapshot:
    return _SettingsPathSnapshot.from_settings(
        settings, paths=ORDER23_OVERLAY_OSC_OUTPUT_SETTINGS_PATHS
    )


def settings_path_snapshot_for_ui_prompt_clipboard_state(
    settings: AppSettingsVNext,
) -> _SettingsPathSnapshot:
    return _SettingsPathSnapshot.from_settings(
        settings, paths=ORDER24_UI_PROMPT_CLIPBOARD_STATE_SETTINGS_PATHS
    )


def build_translation_provider_settings_path_patch(
    previous: AppSettingsVNext,
    next_settings: AppSettingsVNext,
) -> dict[str, object]:
    return _build_settings_path_patch(
        previous,
        next_settings,
        paths=ORDER21_TRANSLATION_PROVIDER_SETTINGS_PATHS,
    )


def build_stt_language_audio_settings_path_patch(
    previous: AppSettingsVNext,
    next_settings: AppSettingsVNext,
) -> dict[str, object]:
    return _build_settings_path_patch(
        previous,
        next_settings,
        paths=ORDER22_STT_LANGUAGE_AUDIO_SETTINGS_PATHS,
    )


def build_overlay_osc_output_settings_path_patch(
    previous: AppSettingsVNext,
    next_settings: AppSettingsVNext,
) -> dict[str, object]:
    return _build_settings_path_patch(
        previous,
        next_settings,
        paths=ORDER23_OVERLAY_OSC_OUTPUT_SETTINGS_PATHS,
    )


def build_ui_prompt_clipboard_state_settings_path_patch(
    previous: AppSettingsVNext,
    next_settings: AppSettingsVNext,
) -> dict[str, object]:
    return _build_settings_path_patch(
        previous,
        next_settings,
        paths=ORDER24_UI_PROMPT_CLIPBOARD_STATE_SETTINGS_PATHS,
    )


def _canonical_settings_dict(settings: AppSettingsVNext) -> dict[str, object]:
    from puripuly_heart.config.settings_vnext import serialization
    from puripuly_heart.config.settings_vnext.schema import AppSettingsVNext

    if not isinstance(settings, AppSettingsVNext):
        raise TypeError("canonical settings path mutation requires AppSettingsVNext")
    return serialization.to_dict(settings)


def _settings_from_canonical_dict(data: dict[str, object]) -> AppSettingsVNext:
    from puripuly_heart.config.settings_vnext import serialization

    return serialization.from_dict(data)


def _get_settings_path_value(settings: AppSettingsVNext, path: str) -> object:
    current: object = _canonical_settings_dict(settings)
    for segment in path.split("."):
        if not isinstance(current, dict):
            raise KeyError(path)
        current = current[segment]
    return copy.deepcopy(current)


def _set_dict_path(data: dict[str, object], path: str, value: object) -> None:
    current: dict[str, object] = data
    segments = path.split(".")
    for segment in segments[:-1]:
        nested = current.get(segment)
        if not isinstance(nested, dict):
            nested = {}
            current[segment] = nested
        current = nested
    current[segments[-1]] = _mutable_settings_value(value)


def apply_settings_path_patch(
    settings: AppSettingsVNext, patch: Mapping[str, object]
) -> AppSettingsVNext:
    data = _canonical_settings_dict(settings)
    for path, value in patch.items():
        _set_dict_path(data, path, value)
    return _settings_from_canonical_dict(data)


def _build_settings_path_patch(
    previous: AppSettingsVNext,
    next_settings: AppSettingsVNext,
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


def _apply_settings_path_patch(
    settings: AppSettingsVNext, patch: Mapping[str, object]
) -> AppSettingsVNext:
    return apply_settings_path_patch(settings, patch)


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
    "apply_settings_path_patch",
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
