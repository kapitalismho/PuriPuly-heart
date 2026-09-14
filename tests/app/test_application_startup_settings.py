from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from puripuly_heart.app.services.canonical_settings_persistence import (
    SettingsOwnerStartResult,
)
from puripuly_heart.app.services.manual_local_asr_fallback import ManualLocalASRFallbackOwner
from puripuly_heart.composition.application_startup import ApplicationStartupAdapter
from puripuly_heart.config.settings_vnext.schema import AppSettingsVNext
from puripuly_heart.core.local_translation import assets


class _FailingPersistOwner:
    def __init__(self, settings: AppSettingsVNext) -> None:
        self.canonical = settings
        self.authoritative = False

    def remember_projection(self, _settings: AppSettingsVNext) -> None:
        return None

    def save_current(self, failure_sink=None) -> bool:
        if failure_sink is not None:
            failure_sink(RuntimeError("persist failed"))
        return False

    def set_overlay_enabled(self, _enabled: bool) -> None:
        return None

    def set_peer_translation_enabled(self, _enabled: bool) -> None:
        return None


@pytest.mark.asyncio
async def test_startup_restores_pre_fallback_canonical_when_persist_fails() -> None:
    original = replace(
        AppSettingsVNext(),
        intent=replace(
            AppSettingsVNext().intent,
            stt=replace(AppSettingsVNext().intent.stt, provider="local_parakeet_v3"),
            peer_stt=replace(AppSettingsVNext().intent.peer_stt, provider="local_parakeet_ja"),
        ),
    )
    adapter = _startup_adapter(
        settings=_FailingPersistOwner(original),
        settings_loader=lambda: _start_result(original, migrated=False),
    )

    state = await adapter.prepare_startup_settings()

    assert state.settings is original
    assert state.settings.intent.stt.provider == "local_parakeet_v3"
    assert state.fallback_channels == ()


def _start_result(settings: AppSettingsVNext, *, migrated: bool) -> SettingsOwnerStartResult:
    return SettingsOwnerStartResult(settings=settings, migrated=migrated, backup_path=None)


def _startup_adapter(**overrides: object) -> ApplicationStartupAdapter:
    provisioning = Mock()
    provisioning.snapshot = SimpleNamespace(cpu_auto_available=True)
    provisioning.inspect_cpu = AsyncMock()
    provisioning.inspect_gpu = AsyncMock()
    fields: dict[str, object] = {
        "settings": Mock(),
        "settings_loader": lambda: _start_result(AppSettingsVNext(), migrated=False),
        "provisioning": provisioning,
        "gpu_state": lambda: SimpleNamespace(selected_provider_requires_model=False),
        "manual_fallback": ManualLocalASRFallbackOwner(),
        "save_failure_sink": lambda _exc: None,
        "model_asset_failure_sink": lambda _message: None,
        "calibration": Mock(),
        "presentation": Mock(),
        "sync_presentation": lambda: None,
        "notify_fallback": lambda *_args: None,
        "runtime_logging": Mock(),
        "sync_runtime_signatures": lambda _settings: None,
        "pipeline_launcher": Mock(),
        "pipeline": Mock(),
        "sync_local_asr_notice": lambda: None,
        "stt_requires_secret": lambda _provider: False,
        "llm_requires_secret": lambda _provider: False,
        "alibaba_verified_key": lambda: "",
        "managed_translation_available": lambda: False,
        "receiver_active": lambda: False,
        "create_event_bridge": lambda _logging: Mock(),
        "start_event_bridge": lambda _bridge: None,
        "wait_for_event_bridge": AsyncMock(),
        "sync_clipboard": AsyncMock(),
    }
    fields.update(overrides)
    return ApplicationStartupAdapter(**fields)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_settings_migration_removes_retired_gemma_install_and_reports_failures(
    tmp_path,
    monkeypatch,
) -> None:
    models_dir = tmp_path / "models"
    retired_id, retired_filename = assets.RETIRED_MANAGED_GEMMA_INSTALLS[0]
    install_dir = models_dir / retired_id
    install_dir.mkdir(parents=True)
    (install_dir / retired_filename).write_bytes(b"retired")
    staging = models_dir / f"{retired_id}.staging-deadbeef"
    staging.mkdir()
    failures: list[str] = []
    real_rmtree = assets.shutil.rmtree

    def flaky_rmtree(path, *args, **kwargs):
        if str(path).endswith(staging.name):
            raise OSError("locked")
        return real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(assets, "default_models_dir", lambda: models_dir)
    monkeypatch.setattr(assets.shutil, "rmtree", flaky_rmtree)

    adapter = _startup_adapter(
        settings_loader=lambda: _start_result(AppSettingsVNext(), migrated=True),
        model_asset_failure_sink=failures.append,
    )

    await adapter.prepare_startup_settings()

    assert not install_dir.exists()
    assert len(failures) == 1
    assert str(staging) in failures[0]
    assert "locked" in failures[0]

    leftover = models_dir / f"{retired_id}.backup-cafe"
    leftover.mkdir()
    steady_state = _startup_adapter(
        settings_loader=lambda: _start_result(AppSettingsVNext(), migrated=False),
        model_asset_failure_sink=failures.append,
    )

    await steady_state.prepare_startup_settings()

    assert leftover.is_dir()
    assert len(failures) == 1
