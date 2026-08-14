from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

from puripuly_heart.app.services.settings_projection import SettingsProjectionOwner

from puripuly_heart.config.settings import AppSettings


def test_projection_rebases_view_change_onto_current_settings() -> None:
    displayed = AppSettings()
    displayed.languages.source_language = "en"
    current = copy.deepcopy(displayed)
    rendered: list[AppSettings] = []
    presentation = SimpleNamespace(
        render_settings=lambda settings, **_kwargs: rendered.append(settings) or True,
    )
    owner = SettingsProjectionOwner(
        presentation=presentation,
        config_path=Path("settings.json"),
        current_settings=lambda: current,
    )

    assert owner.render(displayed) is True
    pending = copy.deepcopy(displayed)
    pending.overlay.show_translation = False
    change = owner.capture(pending)
    current.languages.source_language = "ja"

    merged = owner.merge_with_current(change)

    assert rendered == [displayed]
    assert merged.languages.source_language == "ja"
    assert merged.overlay.show_translation is False


def test_failed_render_preserves_previous_projection_baseline() -> None:
    displayed = AppSettings()
    displayed.languages.source_language = "en"
    current = copy.deepcopy(displayed)

    def fail_render(*_args, **_kwargs) -> bool:
        raise RuntimeError("render failed")

    owner = SettingsProjectionOwner(
        presentation=SimpleNamespace(render_settings=fail_render),
        config_path=Path("settings.json"),
        current_settings=lambda: current,
    )
    owner.remember_all(displayed)
    current.languages.source_language = "ja"

    assert owner.render(current) is None
    pending = copy.deepcopy(displayed)
    pending.overlay.show_translation = False
    merged = owner.merge_with_current(owner.capture(pending))

    assert merged.languages.source_language == "ja"
    assert merged.overlay.show_translation is False


def test_pkce_refresh_uses_specialized_projection_path() -> None:
    settings = AppSettings()
    refreshes: list[tuple[AppSettings, Path]] = []

    def refresh(updated: AppSettings, *, config_path: Path) -> bool:
        refreshes.append((updated, config_path))
        return True

    owner = SettingsProjectionOwner(
        presentation=SimpleNamespace(
            refresh_settings_after_openrouter_pkce_success=refresh,
        ),
        config_path=Path("settings.json"),
        current_settings=lambda: settings,
    )

    assert owner.refresh_after_openrouter_pkce_success(settings) is True
    captured = owner.capture(copy.deepcopy(settings))

    assert refreshes == [(settings, Path("settings.json"))]
    assert captured.can_rebase is True
    assert captured.values_by_path == {}
