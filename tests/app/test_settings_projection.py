from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from puripuly_heart.app.services.settings_application import settings_view_surface_snapshots
from puripuly_heart.app.services.settings_projection import SettingsProjectionOwner

from puripuly_heart.config.settings_vnext.schema import AppSettingsVNext


def _with_source_language(settings: AppSettingsVNext, language: str) -> AppSettingsVNext:
    return replace(
        settings,
        intent=replace(
            settings.intent,
            languages=replace(settings.intent.languages, source_language=language),
        ),
    )


def _with_show_translation(settings: AppSettingsVNext, enabled: bool) -> AppSettingsVNext:
    return replace(
        settings,
        intent=replace(
            settings.intent,
            overlay=replace(settings.intent.overlay, show_translation=enabled),
        ),
    )


def test_projection_rebases_view_change_onto_current_settings() -> None:
    displayed = _with_source_language(AppSettingsVNext(), "en")
    current = copy.deepcopy(displayed)
    rendered: list[dict[str, object]] = []
    presentation = SimpleNamespace(
        render_settings=lambda **kwargs: rendered.append(kwargs) or True,
    )
    owner = SettingsProjectionOwner(
        presentation=presentation,
        config_path=Path("settings.json"),
        current_settings=lambda: current,
    )

    assert owner.render(displayed) is True
    pending = _with_show_translation(displayed, False)
    change = owner.capture(pending)
    current = _with_source_language(current, "ja")

    merged = owner.merge_with_current(change)

    assert rendered[0]["general"].effective_peer_source_language == "en"
    assert merged.intent.languages.source_language == "ja"
    assert merged.intent.overlay.show_translation is False


def test_failed_render_preserves_previous_projection_baseline() -> None:
    displayed = _with_source_language(AppSettingsVNext(), "en")
    current = copy.deepcopy(displayed)

    def fail_render(*_args, **_kwargs) -> bool:
        raise RuntimeError("render failed")

    owner = SettingsProjectionOwner(
        presentation=SimpleNamespace(render_settings=fail_render),
        config_path=Path("settings.json"),
        current_settings=lambda: current,
    )
    owner.remember_all(displayed)
    current = _with_source_language(current, "ja")

    assert owner.render(current) is None
    pending = _with_show_translation(displayed, False)
    merged = owner.merge_with_current(owner.capture(pending))

    assert merged.intent.languages.source_language == "ja"
    assert merged.intent.overlay.show_translation is False


def test_pkce_refresh_uses_specialized_projection_path() -> None:
    settings = AppSettingsVNext()
    refreshes: list[tuple[object, object, Path]] = []

    def refresh(*, provider, prompt, config_path: Path) -> bool:
        refreshes.append((provider, prompt, config_path))
        return True

    owner = SettingsProjectionOwner(
        presentation=SimpleNamespace(
            refresh_settings_after_openrouter_pkce_success=refresh,
        ),
        config_path=Path("settings.json"),
        current_settings=lambda: settings,
    )

    provider, _general, prompt, _overlay = settings_view_surface_snapshots(settings)
    assert (
        owner.refresh_after_openrouter_pkce_success(
            provider=provider,
            prompt=prompt,
            compatibility_settings=settings,
        )
        is True
    )
    captured = owner.capture(copy.deepcopy(settings))

    assert refreshes == [(provider, prompt, Path("settings.json"))]
    assert captured.can_rebase is True
    assert captured.values_by_path == {}
