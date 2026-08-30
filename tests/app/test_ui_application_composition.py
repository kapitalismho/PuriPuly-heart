import json
from pathlib import Path
from types import SimpleNamespace

import puripuly_heart.composition.ui_application as composition_module
from puripuly_heart.app.services.canonical_settings_persistence import compose_settings_owner
from puripuly_heart.app.services.settings_secrets import SettingsSecretsOwner
from puripuly_heart.app.services.ui_application import UiApplicationBoundary
from puripuly_heart.composition.application_settings import load_application_settings
from puripuly_heart.config.settings_vnext import serialization
from puripuly_heart.config.settings_vnext.facade import save_vnext_settings
from puripuly_heart.config.settings_vnext.schema import (
    VNEXT_SETTINGS_SCHEMA_VERSION,
    AppSettingsVNext,
)
from puripuly_heart.ui.presentation_adapter import FletUiPresentationAdapter


def test_composition_forwards_explicit_options_without_a_flet_page(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}
    expected = object()
    presentation = object()
    logging_sinks = object()
    presence = object()

    def compose(**kwargs):
        captured.update(kwargs)
        return expected

    monkeypatch.setattr(composition_module, "compose_application_runtime", compose)

    application = composition_module.compose_ui_application(
        presentation=presentation,
        config_path=Path("settings.json"),
        runtime_logging_sinks=logging_sinks,
        vrchat_osc_presence=presence,
    )

    assert application is expected
    assert captured == {
        "presentation": presentation,
        "config_path": Path("settings.json"),
        "runtime_logging_sinks": logging_sinks,
        "vrchat_osc_presence": presence,
    }
    assert "page" not in captured


def test_real_composition_returns_the_application_boundary(tmp_path: Path) -> None:
    presentation = FletUiPresentationAdapter(
        SimpleNamespace(debug_ui_preview=False),
    )

    application = composition_module.compose_ui_application(
        presentation=presentation,
        config_path=tmp_path / "settings.json",
    )

    assert isinstance(application, UiApplicationBoundary)
    assert isinstance(application.settings_secrets(), SettingsSecretsOwner)


def test_official_settings_loader_first_run_current_and_older_canonical(tmp_path: Path) -> None:
    missing = load_application_settings(settings=compose_settings_owner(tmp_path / "missing.json"))
    assert isinstance(missing, AppSettingsVNext)
    assert missing.settings_version == VNEXT_SETTINGS_SCHEMA_VERSION

    current_path = tmp_path / "current.json"
    save_vnext_settings(current_path, AppSettingsVNext())
    current = load_application_settings(settings=compose_settings_owner(current_path))
    assert isinstance(current, AppSettingsVNext)
    assert current.settings_version == VNEXT_SETTINGS_SCHEMA_VERSION

    older_path = tmp_path / "older.json"
    raw = serialization.to_dict(AppSettingsVNext())
    raw["settings_version"] = VNEXT_SETTINGS_SCHEMA_VERSION - 1
    older_path.write_text(json.dumps(raw), encoding="utf-8")
    older = load_application_settings(settings=compose_settings_owner(older_path))
    assert isinstance(older, AppSettingsVNext)
    assert older.settings_version == VNEXT_SETTINGS_SCHEMA_VERSION
