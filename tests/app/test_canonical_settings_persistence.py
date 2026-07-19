from __future__ import annotations

import asyncio
import copy
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from puripuly_heart.app.adapters import (
    settings_vnext_canonical_persistence as adapter_module,
)
from puripuly_heart.app.adapters.settings_vnext_canonical_persistence import (
    SettingsVNextCanonicalPersistenceAdapter,
)
from puripuly_heart.app.ports.canonical_settings_persistence import (
    CanonicalSettingsPersistencePort,
    ProviderVerificationBinding,
)
from puripuly_heart.config.settings import AppSettings
from puripuly_heart.config.settings_vnext.facade import load_vnext_settings
from puripuly_heart.config.settings_vnext.schema import AppSettingsVNext
from puripuly_heart.ui import controller as controller_module
from puripuly_heart.ui.controller import GuiController


class MemorySecretStore:
    def __init__(self, values: dict[str, str] | None = None) -> None:
        self.values = dict(values or {})
        self.block_set = False
        self.set_started = threading.Event()
        self.release_set = threading.Event()

    def get(self, key: str) -> str | None:
        return self.values.get(key)

    def set(self, key: str, value: str) -> None:
        if self.block_set:
            self.set_started.set()
            self.release_set.wait(timeout=5)
        self.values[key] = value

    def delete(self, key: str) -> None:
        self.values.pop(key, None)


class BlockingByKeySecretStore(MemorySecretStore):
    def __init__(self, values: dict[str, str], keys: tuple[str, ...]) -> None:
        super().__init__(values)
        self.started = {key: threading.Event() for key in keys}
        self.release = {key: threading.Event() for key in keys}

    def set(self, key: str, value: str) -> None:
        self.started[key].set()
        self.release[key].wait(timeout=5)
        self.values[key] = value


def _controller_with_verified_openrouter(
    path: Path,
    store: MemorySecretStore,
    monkeypatch: pytest.MonkeyPatch,
) -> GuiController:
    controller = GuiController(page=SimpleNamespace(), app=SimpleNamespace(), config_path=path)
    controller.settings = AppSettings()
    controller.settings.api_key_verified.openrouter = True
    owner = controller._get_settings_owner()
    owner.canonical = AppSettingsVNext()
    owner.bind_provider_verification(
        ProviderVerificationBinding(
            provider="openrouter",
            secret_key="openrouter_api_key",
            secret_revision=None,
            secret_fingerprint="sha256:old-secret",
            verifier_context={"flow": "settings_api_key_verification"},
            verifier_evidence={"source": "provider_verifier"},
        )
    )
    controller._vnext_settings_authoritative = True
    owner.persist()
    controller._remember_canonical_legacy_projection(controller.settings)
    monkeypatch.setattr(
        controller_module,
        "create_secret_store",
        lambda *_args, **_kwargs: store,
    )
    return controller


def test_canonical_settings_persistence_port_covers_load_project_delta_save_and_rollback(
    monkeypatch,
) -> None:
    adapter = SettingsVNextCanonicalPersistenceAdapter()
    settings = AppSettings()
    canonical = AppSettingsVNext()
    path = Path("settings.json")
    saved: list[AppSettingsVNext] = []

    assert isinstance(adapter, CanonicalSettingsPersistencePort)

    monkeypatch.setattr(
        adapter_module,
        "load_vnext_settings",
        lambda _path: SimpleNamespace(
            settings=canonical,
            migrated=False,
            backup_path=None,
        ),
    )
    loaded = adapter.load_active(path)
    assert loaded.canonical_settings is canonical
    assert loaded.compatibility_settings.stt.low_latency_mode is True

    projected = adapter.project(settings, canonical=None, authoritative=False)
    assert projected.intent.languages.peer_source_mode == "manual"
    assert projected.intent.languages.peer_expected_languages == []

    assert adapter.project(settings, canonical=canonical, authoritative=True) is canonical
    assert adapter.project(settings, canonical=canonical, authoritative=False) == projected

    updated = copy.deepcopy(settings)
    updated.ui.locale = "ja"
    updated_canonical = adapter.apply_legacy_delta(
        canonical=projected,
        base_settings=settings,
        next_settings=updated,
    )
    assert updated_canonical.intent.ui.locale == "ja"

    monkeypatch.setattr(
        adapter_module,
        "save_vnext_settings",
        lambda _path, value: saved.append(value) or SimpleNamespace(ok=True),
    )
    adapter.persist(path, updated_canonical)
    assert saved == [updated_canonical]

    snapshot = adapter.snapshot(updated_canonical)
    assert snapshot == updated_canonical
    assert snapshot is not updated_canonical
    restored = adapter.rollback(snapshot)
    assert restored == updated_canonical
    assert restored is not snapshot


def test_canonical_delta_requires_bound_evidence_and_preserves_invalidation() -> None:
    adapter = SettingsVNextCanonicalPersistenceAdapter()
    baseline = AppSettings()
    verified_settings = copy.deepcopy(baseline)
    verified_settings.api_key_verified.openrouter = True

    unbound = adapter.apply_legacy_delta(
        canonical=AppSettingsVNext(),
        base_settings=baseline,
        next_settings=verified_settings,
    )

    assert unbound.state.provider_verification.openrouter.status == "unknown"

    verified = adapter.bind_provider_verification(
        unbound,
        ProviderVerificationBinding(
            provider="openrouter",
            secret_key="openrouter_api_key",
            secret_revision=None,
            secret_fingerprint="sha256:credential",
            verifier_context={"flow": "settings_api_key_verification"},
            verifier_evidence={"source": "provider_verifier"},
        ),
    )
    assert verified.state.provider_verification.openrouter.status == "verified"
    assert verified.state.provider_verification.openrouter.secret_key == "openrouter_api_key"

    invalidated_settings = copy.deepcopy(verified_settings)
    invalidated_settings.api_key_verified.openrouter = False
    invalidated = adapter.apply_legacy_delta(
        canonical=verified,
        base_settings=verified_settings,
        next_settings=invalidated_settings,
    )

    assert invalidated.state.provider_verification.openrouter.status == "unknown"


def test_controller_persist_settings_roundtrips_verification_transitions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "settings.json"
    controller = GuiController(page=SimpleNamespace(), app=SimpleNamespace(), config_path=path)
    controller.settings = AppSettings()
    controller.vnext_settings = AppSettingsVNext()
    controller._vnext_settings_authoritative = True
    controller._remember_canonical_legacy_projection(controller.settings)

    raw_secret = "raw-openrouter-secret-value"
    monkeypatch.setattr(
        controller_module,
        "create_secret_store",
        lambda *_args, **_kwargs: SimpleNamespace(get=lambda _key: raw_secret),
    )
    controller.persist_api_key_verification("openrouter", raw_secret, True)

    verified = load_vnext_settings(path)
    assert verified.settings is not None
    entry = verified.settings.state.provider_verification.openrouter
    assert entry.status == "verified"
    assert entry.secret_key == "openrouter_api_key"
    assert entry.secret_revision is None
    assert entry.secret_fingerprint is not None
    assert entry.secret_fingerprint.startswith("sha256:")
    assert entry.verifier_context == {"flow": "settings_api_key_verification"}
    assert entry.verifier_evidence == {"source": "provider_verifier"}
    assert raw_secret not in path.read_text(encoding="utf-8")

    controller.persist_api_key_verification("openrouter", raw_secret, False)

    invalidated = load_vnext_settings(path)
    assert invalidated.settings is not None
    assert invalidated.settings.state.provider_verification.openrouter.status == "unknown"


def test_controller_rejects_verification_for_nonmatching_secret_store_value(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "settings.json"
    controller = GuiController(page=SimpleNamespace(), app=SimpleNamespace(), config_path=path)
    controller.settings = AppSettings()
    controller.vnext_settings = AppSettingsVNext()
    controller._vnext_settings_authoritative = True
    controller._remember_canonical_legacy_projection(controller.settings)
    monkeypatch.setattr(
        controller_module,
        "create_secret_store",
        lambda *_args, **_kwargs: SimpleNamespace(get=lambda _key: "different-secret"),
    )

    with pytest.raises(
        RuntimeError,
        match="verified credential does not match the active SecretStore value",
    ):
        controller.persist_api_key_verification(
            "openrouter",
            "verified-but-not-stored",
            True,
        )

    assert controller.settings.api_key_verified.openrouter is False
    assert controller.vnext_settings.state.provider_verification.openrouter.status == "unknown"
    assert controller._canonical_mutation_depth == 0
    assert not path.exists()


@pytest.mark.asyncio
async def test_provider_secret_change_invalidates_before_reverification_and_relaunch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "settings.json"
    store = MemorySecretStore({"openrouter_api_key": "old-secret"})
    controller = _controller_with_verified_openrouter(path, store, monkeypatch)

    assert await controller.persist_provider_secret_change(
        "openrouter_api_key",
        "new-secret",
    )

    assert store.get("openrouter_api_key") == "new-secret"
    assert controller.settings.api_key_verified.openrouter is False
    assert controller.vnext_settings.state.provider_verification.openrouter.status == "unknown"
    reloaded = load_vnext_settings(path)
    assert reloaded.settings is not None
    assert reloaded.settings.state.provider_verification.openrouter.status == "unknown"


@pytest.mark.asyncio
async def test_provider_secret_change_restores_secret_and_verification_on_commit_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "settings.json"
    store = MemorySecretStore({"openrouter_api_key": "old-secret"})
    controller = _controller_with_verified_openrouter(path, store, monkeypatch)
    persisted_before = path.read_bytes()

    def fail_persist(_path: Path, _settings: AppSettingsVNext) -> None:
        raise OSError("injected persistence failure")

    monkeypatch.setattr(controller._get_settings_owner().persistence, "persist", fail_persist)

    assert not await controller.persist_provider_secret_change(
        "openrouter_api_key",
        "new-secret",
    )

    assert store.get("openrouter_api_key") == "old-secret"
    assert controller.settings.api_key_verified.openrouter is True
    assert controller.vnext_settings.state.provider_verification.openrouter.status == "verified"
    assert path.read_bytes() == persisted_before


@pytest.mark.asyncio
async def test_provider_secret_change_finishes_invalidation_when_caller_is_cancelled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "settings.json"
    store = MemorySecretStore({"openrouter_api_key": "old-secret"})
    store.block_set = True
    controller = _controller_with_verified_openrouter(path, store, monkeypatch)
    task = asyncio.create_task(
        controller.persist_provider_secret_change(
            "openrouter_api_key",
            "new-secret",
        )
    )
    assert await asyncio.to_thread(store.set_started.wait, 2)

    task.cancel()
    store.release_set.set()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert store.get("openrouter_api_key") == "new-secret"
    assert controller.settings.api_key_verified.openrouter is False
    reloaded = load_vnext_settings(path)
    assert reloaded.settings is not None
    assert reloaded.settings.state.provider_verification.openrouter.status == "unknown"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("first_key", "second_key"),
    [
        ("openrouter_api_key", "deepseek_api_key"),
        ("deepseek_api_key", "openrouter_api_key"),
    ],
)
async def test_overlapping_provider_secret_changes_preserve_both_invalidations(
    first_key: str,
    second_key: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "settings.json"
    keys = ("openrouter_api_key", "deepseek_api_key")
    store = BlockingByKeySecretStore(
        {
            "openrouter_api_key": "old-openrouter-secret",
            "deepseek_api_key": "old-deepseek-secret",
        },
        keys,
    )
    controller = _controller_with_verified_openrouter(path, store, monkeypatch)
    controller.settings.api_key_verified.deepseek = True
    controller._get_settings_owner().bind_provider_verification(
        ProviderVerificationBinding(
            provider="deepseek",
            secret_key="deepseek_api_key",
            secret_revision=None,
            secret_fingerprint="sha256:old-deepseek-secret",
            verifier_context={"flow": "settings_api_key_verification"},
            verifier_evidence={"source": "provider_verifier"},
        )
    )
    controller._get_settings_owner().persist()
    controller._remember_canonical_legacy_projection(controller.settings)

    first_task = asyncio.create_task(
        controller.persist_provider_secret_change(first_key, f"new-{first_key}")
    )
    assert await asyncio.to_thread(store.started[first_key].wait, 2)
    second_task = asyncio.create_task(
        controller.persist_provider_secret_change(second_key, f"new-{second_key}")
    )
    await asyncio.sleep(0.05)
    assert not store.started[second_key].is_set()

    store.release[first_key].set()
    assert await asyncio.to_thread(store.started[second_key].wait, 2)
    store.release[second_key].set()

    assert await first_task
    assert await second_task
    assert store.get("openrouter_api_key") == "new-openrouter_api_key"
    assert store.get("deepseek_api_key") == "new-deepseek_api_key"
    assert controller.settings.api_key_verified.openrouter is False
    assert controller.settings.api_key_verified.deepseek is False
    assert controller.vnext_settings.state.provider_verification.openrouter.status == "unknown"
    assert controller.vnext_settings.state.provider_verification.deepseek.status == "unknown"
    reloaded = load_vnext_settings(path)
    assert reloaded.settings is not None
    assert reloaded.settings.state.provider_verification.openrouter.status == "unknown"
    assert reloaded.settings.state.provider_verification.deepseek.status == "unknown"
