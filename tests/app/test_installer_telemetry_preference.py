from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import puripuly_heart.app.adapters.settings_vnext_canonical_persistence as persistence_adapter
import puripuly_heart.app.services.installer_telemetry_preference as preference_module
import puripuly_heart.main as main_module
from puripuly_heart.app.ports.canonical_settings_persistence import (
    CanonicalSettingsPersistenceError,
)
from puripuly_heart.config.settings_vnext import serialization
from puripuly_heart.config.settings_vnext.facade import load_vnext_settings
from puripuly_heart.config.settings_vnext.schema import AppSettingsVNext, with_telemetry_enabled


def _write_settings(path: Path, settings: AppSettingsVNext) -> None:
    path.write_text(serialization.to_json_text(settings), encoding="utf-8")


def test_fresh_installer_preference_is_canonical_and_off_clears_identity(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"

    preference_module.persist_installer_telemetry_preference(path, False)

    persisted = json.loads(path.read_text(encoding="utf-8"))
    assert persisted["intent"]["telemetry"]["enabled"] is False
    assert persisted["state"]["telemetry"] == {
        "anonymous_id": None,
        "last_sent_date_utc": None,
    }


def test_upgrade_preserves_unrelated_settings_and_reenabling_creates_identity(
    tmp_path: Path,
) -> None:
    path = tmp_path / "settings.json"
    existing = replace(
        with_telemetry_enabled(AppSettingsVNext(), False),
        intent=replace(
            with_telemetry_enabled(AppSettingsVNext(), False).intent,
            ui=replace(AppSettingsVNext().intent.ui, locale="ja"),
        ),
    )
    _write_settings(path, existing)

    preference_module.persist_installer_telemetry_preference(path, False)
    loaded = load_vnext_settings(path).settings
    assert loaded is not None
    assert loaded.intent.ui.locale == "ja"
    assert loaded.state.telemetry.anonymous_id is None

    preference_module.persist_installer_telemetry_preference(path, True)
    enabled = load_vnext_settings(path).settings
    assert enabled is not None
    assert enabled.intent.ui.locale == "ja"
    assert enabled.intent.telemetry.enabled is True
    assert enabled.state.telemetry.anonymous_id


def test_installer_preference_uses_supported_legacy_telemetry_migration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from puripuly_heart.core.local_translation import assets

    monkeypatch.setattr(assets, "default_models_dir", lambda: tmp_path / "models")
    path = tmp_path / "settings.json"
    raw = serialization.to_dict(AppSettingsVNext())
    raw["settings_version"] = 24
    raw["intent"]["ui"]["locale"] = "ko"
    raw["intent"]["telemetry"] = {"consent": "decline"}
    path.write_text(json.dumps(raw), encoding="utf-8")

    preference_module.persist_installer_telemetry_preference(path, False)

    loaded = load_vnext_settings(path).settings
    assert loaded is not None
    assert loaded.intent.ui.locale == "ko"
    assert loaded.intent.telemetry.enabled is False
    assert loaded.state.telemetry.anonymous_id is None


@pytest.mark.parametrize(
    "content",
    [
        "{not-json",
        json.dumps(
            {
                **serialization.to_dict(AppSettingsVNext()),
                "settings_version": 999,
            }
        ),
    ],
)
def test_invalid_existing_settings_stop_preference_persistence_without_replacement(
    tmp_path: Path, content: str
) -> None:
    path = tmp_path / "settings.json"
    path.write_text(content, encoding="utf-8")

    with pytest.raises((ValueError, RuntimeError)):
        preference_module.persist_installer_telemetry_preference(path, True)

    assert path.read_text(encoding="utf-8") == content


def test_save_failure_is_reported_instead_of_succeeding(monkeypatch, tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    monkeypatch.setattr(
        persistence_adapter,
        "save_vnext_settings",
        lambda *_args: SimpleNamespace(
            ok=False,
            status="save_failed",
            error=SimpleNamespace(message="blocked"),
        ),
    )

    with pytest.raises(CanonicalSettingsPersistenceError) as raised:
        preference_module.persist_installer_telemetry_preference(path, False)
    assert raised.value.status == "save_failed"
    assert not path.exists()


@pytest.mark.parametrize(
    ("failure", "expected_category"),
    [
        (CanonicalSettingsPersistenceError("save_failed", "private path"), "save_failed"),
        (OSError("private path"), "unexpected_error"),
    ],
)
def test_installer_cli_reports_stable_failure_category_without_exception_content(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    failure: Exception,
    expected_category: str,
) -> None:
    def fail(*_args: object) -> None:
        raise failure

    monkeypatch.setattr(
        preference_module,
        "persist_installer_telemetry_preference",
        fail,
    )

    result = main_module.main(
        [
            "--config",
            str(tmp_path / "settings.json"),
            "installer-telemetry-preference",
            "disable",
        ]
    )

    captured = capsys.readouterr()
    fields = {
        key: value
        for token in captured.err.split()
        if "=" in token
        for key, value in (token.split("=", maxsplit=1),)
    }
    assert result == 23
    assert captured.out == ""
    assert fields["operation"] == "disable"
    assert fields["status"] == "failure"
    assert fields["failure_category"] == expected_category
    assert "private path" not in captured.err


def test_installer_cli_keeps_primary_settings_failure_when_diagnostic_stream_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = tmp_path / "settings.json"
    content = "{not-json"
    path.write_text(content, encoding="utf-8")

    class UnwritableStream:
        def write(self, _s: str) -> int:
            raise OSError(22, "invalid stream")

        def flush(self) -> None:
            raise OSError(22, "invalid stream")

    monkeypatch.setattr(main_module.sys, "stderr", UnwritableStream())

    result = main_module.main(
        [
            "--config",
            str(path),
            "installer-telemetry-preference",
            "disable",
        ]
    )

    assert result == 23
    assert path.read_text(encoding="utf-8") == content


def test_installer_cli_persists_before_runtime_logging_or_gui_startup(
    monkeypatch, tmp_path: Path
) -> None:
    path = tmp_path / "settings.json"
    monkeypatch.setattr(
        main_module,
        "configure_main_logging",
        lambda **_kwargs: pytest.fail("installer preference command started app logging"),
    )

    assert (
        main_module.main(
            [
                "--config",
                str(path),
                "installer-telemetry-preference",
                "disable",
            ]
        )
        == 0
    )
    persisted = load_vnext_settings(path).settings
    assert persisted is not None
    assert persisted.intent.telemetry.enabled is False
    assert persisted.state.telemetry.anonymous_id is None
