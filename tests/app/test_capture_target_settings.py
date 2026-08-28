from __future__ import annotations

from pathlib import Path

import pytest
from puripuly_heart.app.services.capture_target_settings import (
    CaptureTargetSettingsError,
    persist_desktop_audio_capture_target,
)

from puripuly_heart.app.adapters import (
    settings_vnext_canonical_persistence as persistence_adapter,
)
from puripuly_heart.app.adapters.settings_vnext_canonical_persistence import (
    SettingsVNextCanonicalPersistenceAdapter,
)
from puripuly_heart.config.settings import AppSettings
from puripuly_heart.config.settings_vnext import compat
from puripuly_heart.config.settings_vnext.facade import load_vnext_settings, save_vnext_settings
from puripuly_heart.config.settings_vnext.migration import from_legacy_app_settings
from puripuly_heart.config.settings_vnext.schema import (
    CaptureTargetIntent,
    ProcessCaptureTargetIntent,
)


def _process_target() -> CaptureTargetIntent:
    return CaptureTargetIntent.process_target(
        ProcessCaptureTargetIntent.vrchat(r"C:\VRChat\VRChat.exe")
    )


def test_capture_target_persistence_creates_an_absent_settings_file(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"

    saved = persist_desktop_audio_capture_target(path, _process_target())

    assert path.is_file()
    assert saved.intent.desktop_audio.capture_target.kind == "process"
    loaded = load_vnext_settings(path)
    assert loaded.ok
    assert loaded.settings is not None
    assert loaded.settings.intent.desktop_audio.capture_target.kind == "process"


def test_capture_target_persistence_updates_valid_canonical_settings(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    save_vnext_settings(path, from_legacy_app_settings(AppSettings()))

    saved = persist_desktop_audio_capture_target(path, _process_target())

    assert saved.intent.desktop_audio.capture_target.kind == "process"
    loaded = load_vnext_settings(path)
    assert loaded.ok
    assert loaded.settings is not None
    assert loaded.settings.intent.desktop_audio.capture_target.kind == "process"


def test_capture_target_persistence_rejects_malformed_existing_settings_without_overwrite(
    tmp_path: Path,
) -> None:
    path = tmp_path / "settings.json"
    original_bytes = b'{"intent":'
    path.write_bytes(original_bytes)

    with pytest.raises(CaptureTargetSettingsError) as raised:
        persist_desktop_audio_capture_target(path, _process_target())

    assert raised.value.status == "parse_failed"
    assert "JSON" not in str(raised.value)
    assert path.read_bytes() == original_bytes


def test_capture_target_persistence_rejects_unreadable_existing_settings_without_overwrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "settings.json"
    original_bytes = b'{"legacy": "must remain untouched"}'
    path.write_bytes(original_bytes)

    def fail_load(_path: Path):
        raise PermissionError("raw unreadable settings detail")

    monkeypatch.setattr(persistence_adapter, "load_vnext_settings", fail_load)

    with pytest.raises(CaptureTargetSettingsError) as raised:
        persist_desktop_audio_capture_target(path, _process_target())

    assert raised.value.status == "load_failed"
    assert "raw unreadable settings detail" not in str(raised.value)
    assert path.read_bytes() == original_bytes


@pytest.mark.parametrize(
    "status",
    [
        compat.SettingsPersistenceStatus.PARSE_FAILED,
        compat.SettingsPersistenceStatus.MIGRATION_FAILED,
        compat.SettingsPersistenceStatus.BACKUP_FAILED,
    ],
)
def test_capture_target_persistence_never_overwrites_a_failed_existing_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    status: compat.SettingsPersistenceStatus,
) -> None:
    path = tmp_path / "settings.json"
    original_bytes = b'{"legacy": "must remain untouched"}'
    path.write_bytes(original_bytes)
    failure = compat.VNextSettingsLoadResult(
        status=status,
        error=compat.SettingsPersistenceError(status, "raw secret failure detail"),
    )
    monkeypatch.setattr(persistence_adapter, "load_vnext_settings", lambda _path: failure)

    with pytest.raises(CaptureTargetSettingsError) as raised:
        persist_desktop_audio_capture_target(path, _process_target())

    assert raised.value.status == status.value
    assert "raw secret failure detail" not in str(raised.value)
    assert path.read_bytes() == original_bytes


def test_capture_target_persistence_rejects_absent_file_migration_failure_without_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "settings.json"

    def fail_projection(*_args, **_kwargs):
        raise RuntimeError("raw migration failure")

    monkeypatch.setattr(
        "puripuly_heart.app.services.capture.capture_target_settings.new_settings_for_first_run",
        fail_projection,
    )

    with pytest.raises(CaptureTargetSettingsError) as raised:
        persist_desktop_audio_capture_target(path, _process_target())

    assert raised.value.status == "migration_failed"
    assert "raw migration failure" not in str(raised.value)
    assert not path.exists()


def test_capture_target_persistence_loads_existing_file_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "settings.json"
    save_vnext_settings(path, from_legacy_app_settings(AppSettings()))
    original_load = persistence_adapter.load_vnext_settings
    loads: list[Path] = []

    def counted_load(incoming: Path):
        loads.append(incoming)
        return original_load(incoming)

    monkeypatch.setattr(persistence_adapter, "load_vnext_settings", counted_load)

    persist_desktop_audio_capture_target(path, _process_target())

    assert loads == [path]


def test_capture_target_projection_failure_does_not_change_persisted_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "settings.json"
    save_vnext_settings(path, from_legacy_app_settings(AppSettings()))
    original_bytes = path.read_bytes()
    original_projection = SettingsVNextCanonicalPersistenceAdapter.compatibility_projection
    projections = 0

    def fail_second_projection(self, settings):
        nonlocal projections
        projections += 1
        if projections == 2:
            raise RuntimeError("raw projection failure")
        return original_projection(self, settings)

    monkeypatch.setattr(
        SettingsVNextCanonicalPersistenceAdapter,
        "compatibility_projection",
        fail_second_projection,
    )

    with pytest.raises(CaptureTargetSettingsError) as raised:
        persist_desktop_audio_capture_target(path, _process_target())

    assert raised.value.status == "save_failed"
    assert "raw projection failure" not in str(raised.value)
    assert path.read_bytes() == original_bytes
