from __future__ import annotations

import copy
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from puripuly_heart.app.ports._settings_values import freeze_settings_values
from puripuly_heart.app.ports.settings_view import (
    GeneralSettingsSnapshot,
    OverlaySettingsSnapshot,
    PromptSettingsSnapshot,
    ProviderSettingsSnapshot,
)
from puripuly_heart.app.ports.ui_presentation import UiPresentationPort
from puripuly_heart.config.settings_vnext.schema import AppSettingsVNext

from .settings_mutation_legacy import (
    _SettingsPathSnapshot,
    apply_settings_path_patch,
    build_overlay_osc_output_settings_path_patch,
    build_stt_language_audio_settings_path_patch,
    build_ui_prompt_clipboard_state_settings_path_patch,
    settings_path_snapshot_for_overlay_osc_output,
    settings_path_snapshot_for_stt_language_audio,
    settings_path_snapshot_for_ui_prompt_clipboard_state,
)


@dataclass(frozen=True, slots=True)
class SettingsViewSettingsChange:
    values_by_path: Mapping[str, object]
    pending_settings: AppSettingsVNext
    can_rebase: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "values_by_path", freeze_settings_values(self.values_by_path))
        object.__setattr__(self, "pending_settings", copy.deepcopy(self.pending_settings))


@dataclass(slots=True)
class SettingsProjectionOwner:
    presentation: UiPresentationPort
    config_path: Path
    current_settings: Callable[[], AppSettingsVNext | None]
    _order22_baseline: _SettingsPathSnapshot | None = field(
        init=False,
        default=None,
        repr=False,
    )
    _order23_baseline: _SettingsPathSnapshot | None = field(
        init=False,
        default=None,
        repr=False,
    )
    _order24_baseline: _SettingsPathSnapshot | None = field(
        init=False,
        default=None,
        repr=False,
    )

    @property
    def order22_baseline(self) -> object | None:
        return self._order22_baseline

    def remember_order22(self,         settings: AppSettingsVNext | None) -> None:
        self._order22_baseline = (
            settings_path_snapshot_for_stt_language_audio(settings)
            if settings is not None
            else None
        )

    def remember_order23(self,         settings: AppSettingsVNext | None) -> None:
        self._order23_baseline = (
            settings_path_snapshot_for_overlay_osc_output(settings)
            if settings is not None
            else None
        )

    def remember_order24(self,         settings: AppSettingsVNext | None) -> None:
        self._order24_baseline = (
            settings_path_snapshot_for_ui_prompt_clipboard_state(settings)
            if settings is not None
            else None
        )

    def remember_all(self,         settings: AppSettingsVNext | None) -> None:
        self.remember_order22(settings)
        self.remember_order23(settings)
        self.remember_order24(settings)

    def capture(self, pending_settings: AppSettingsVNext) -> SettingsViewSettingsChange:
        baselines = (
            self._order22_baseline,
            self._order23_baseline,
            self._order24_baseline,
        )
        if any(baseline is None for baseline in baselines):
            return SettingsViewSettingsChange(
                values_by_path={},
                pending_settings=pending_settings,
                can_rebase=False,
            )

        values_by_path: dict[str, object] = {}
        for baseline in baselines:
            if baseline is not None:
                values_by_path.update(baseline.patch_to(pending_settings))
        return SettingsViewSettingsChange(
            values_by_path=values_by_path,
            pending_settings=pending_settings,
            can_rebase=True,
        )

    def merge_with_current(self, change: SettingsViewSettingsChange) -> AppSettingsVNext:
        settings = self.current_settings()
        if not change.can_rebase or settings is None:
            return copy.deepcopy(change.pending_settings)
        return apply_settings_path_patch(settings, change.values_by_path)

    def render_surfaces(
        self,
        *,
        provider: ProviderSettingsSnapshot,
        general: GeneralSettingsSnapshot,
        prompt: PromptSettingsSnapshot,
        overlay: OverlaySettingsSnapshot,
        compatibility_settings: AppSettingsVNext,
        preserve_custom_vocab_draft: bool = False,
    ) -> bool | None:
        try:
            loaded = self.presentation.render_settings(
                provider=provider,
                general=general,
                prompt=prompt,
                overlay=overlay,
                config_path=self.config_path,
                preserve_custom_vocab_draft=preserve_custom_vocab_draft,
            )
        except Exception:
            return None
        self.remember_all(compatibility_settings)
        return bool(loaded)

    def render(
        self,
        settings: AppSettingsVNext,
        *,
        preserve_custom_vocab_draft: bool = False,
    ) -> bool | None:
        from .settings_application import settings_view_surface_snapshots

        provider, general, prompt, overlay = settings_view_surface_snapshots(settings)
        return self.render_surfaces(
            provider=provider,
            general=general,
            prompt=prompt,
            overlay=overlay,
            compatibility_settings=settings,
            preserve_custom_vocab_draft=preserve_custom_vocab_draft,
        )

    def refresh_after_openrouter_pkce_success(
        self,
        *,
        provider: ProviderSettingsSnapshot,
        prompt: PromptSettingsSnapshot,
        compatibility_settings: AppSettingsVNext,
    ) -> bool | None:
        try:
            loaded = self.presentation.refresh_settings_after_openrouter_pkce_success(
                provider=provider,
                prompt=prompt,
                config_path=self.config_path,
            )
        except Exception:
            return None
        self.remember_all(compatibility_settings)
        return bool(loaded)

    def order22_patch_base_and_values(
        self,
        next_settings: AppSettingsVNext,
    ) -> tuple[AppSettingsVNext, dict[str, object]] | None:
        return self._patch_base_and_values(
            next_settings,
            baseline=self._order22_baseline,
            build_patch=build_stt_language_audio_settings_path_patch,
        )

    def order23_patch_base_and_values(
        self,
        next_settings: AppSettingsVNext,
    ) -> tuple[AppSettingsVNext, dict[str, object]] | None:
        return self._patch_base_and_values(
            next_settings,
            baseline=self._order23_baseline,
            build_patch=build_overlay_osc_output_settings_path_patch,
        )

    def order24_patch_base_and_values(
        self,
        next_settings: AppSettingsVNext,
    ) -> tuple[AppSettingsVNext, dict[str, object]] | None:
        return self._patch_base_and_values(
            next_settings,
            baseline=self._order24_baseline,
            build_patch=build_ui_prompt_clipboard_state_settings_path_patch,
        )

    def _patch_base_and_values(
        self,
        next_settings: AppSettingsVNext,
        *,
        baseline: _SettingsPathSnapshot | None,
        build_patch: Callable[[AppSettingsVNext, AppSettingsVNext], dict[str, object]],
    ) -> tuple[AppSettingsVNext, dict[str, object]] | None:
        settings = self.current_settings()
        if settings is None:
            return None
        patch_values = build_patch(settings, next_settings)
        if patch_values or next_settings is settings:
            return settings, patch_values
        if baseline is not None:
            baseline_patch_values = baseline.patch_to(next_settings)
            if baseline_patch_values:
                return baseline.materialize_base_from(settings), baseline_patch_values
        return settings, patch_values
