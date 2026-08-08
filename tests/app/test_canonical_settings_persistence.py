from __future__ import annotations

import asyncio
import copy
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
from puripuly_heart.app.services.provider_settings import (
    ProviderSettingsOwner,
    provider_verification_context,
)
from puripuly_heart.app.services.provider_verification_binding import (
    ProviderVerificationBindingOwner,
)

from puripuly_heart.app.adapters import (
    settings_vnext_canonical_persistence as adapter_module,
)
from puripuly_heart.app.adapters.settings_vnext_canonical_persistence import (
    SettingsVNextCanonicalPersistenceAdapter,
)
from puripuly_heart.app.adapters.sync_secret_store import SyncSecretStoreAdapter
from puripuly_heart.app.ports.canonical_settings_persistence import (
    CanonicalSettingsPersistencePort,
    ProviderVerificationBinding,
)
from puripuly_heart.app.services.canonical_settings_persistence import SettingsOwner
from puripuly_heart.config.settings import AppSettings
from puripuly_heart.config.settings_vnext.facade import load_vnext_settings
from puripuly_heart.config.settings_vnext.schema import AppSettingsVNext
from puripuly_heart.core.translation_policy import FIXED_TRANSLATION_POLICY


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


def _provider_settings_owner(
    path: Path,
    store: MemorySecretStore,
) -> ProviderSettingsOwner:
    settings = AppSettings()
    owner = SettingsOwner(
        path=path,
        persistence=SettingsVNextCanonicalPersistenceAdapter(),
        canonical=AppSettingsVNext(),
        current=settings,
        authoritative=True,
        projection_snapshot=copy.deepcopy(settings),
    )
    return ProviderSettingsOwner(
        settings=owner,
        binding=ProviderVerificationBindingOwner(
            context_provider=lambda provider: provider_verification_context(
                owner.current,
                provider,
                low_latency=FIXED_TRANSLATION_POLICY.fast_translation_enabled,
            ),
        ),
        secret_store_factory=lambda _settings: SyncSecretStoreAdapter(store),
        active_secret_provider=lambda _settings, key: store.get(key),
    )


def _owner_with_verified_openrouter(
    path: Path,
    store: MemorySecretStore,
) -> ProviderSettingsOwner:
    provider_settings = _provider_settings_owner(path, store)
    owner = provider_settings.settings
    assert owner.current is not None
    owner.current.api_key_verified.openrouter = True
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
    owner.persist()
    owner.remember_projection(owner.current)
    return provider_settings


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

    owner = SettingsOwner(
        path=path,
        persistence=adapter,
        canonical=projected,
        current=copy.deepcopy(settings),
        authoritative=True,
        projection_snapshot=copy.deepcopy(settings),
    )
    owner.current.ui.locale = "ja"
    assert owner.save_current()
    assert owner.current.ui.locale == "ja"
    assert owner.projection_snapshot is not None
    assert owner.projection_snapshot.ui.locale == "ja"
    assert owner.mutation_depth == 0

    failures: list[BaseException] = []

    def fail_save(_path: Path, _value: AppSettingsVNext) -> None:
        raise OSError("injected save failure")

    monkeypatch.setattr(adapter_module, "save_vnext_settings", fail_save)
    owner.current.ui.locale = "ko"
    assert not owner.save_current(failure_sink=failures.append)
    assert len(failures) == 1
    assert isinstance(failures[0], OSError)
    assert owner.current.ui.locale == "ja"
    assert owner.mutation_depth == 0

    with pytest.raises(OSError, match="injected save failure"):
        owner.persist_current()

    monkeypatch.setattr(
        adapter_module,
        "save_vnext_settings",
        lambda _path, value: saved.append(value) or SimpleNamespace(ok=True),
    )
    stale_settings = copy.deepcopy(owner.current)
    persist_managed_identity = owner.managed_identity_persistence_callback(stale_settings)
    active_settings = copy.deepcopy(stale_settings)
    active_settings.ui.locale = "ru"
    owner.current = active_settings
    owner.remember_projection(active_settings)
    owner.persist_current()

    stale_settings.managed_identity.referral_id = "234567"
    persist_managed_identity(stale_settings)

    assert owner.current.ui.locale == "ru"
    assert owner.current.managed_identity.referral_id == "234567"
    assert owner.canonical.state.managed_connection.referral_id == "234567"

    active_before_failure = copy.deepcopy(owner.current)
    stale_before_failure = copy.deepcopy(stale_settings)
    monkeypatch.setattr(adapter_module, "save_vnext_settings", fail_save)
    stale_settings.managed_identity.referral_id = "345678"
    with pytest.raises(OSError, match="injected save failure"):
        persist_managed_identity(stale_settings)
    assert owner.current == active_before_failure
    assert stale_settings == stale_before_failure


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


def test_settings_owner_roundtrips_verification_transitions(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    raw_secret = "raw-openrouter-secret-value"
    provider_settings = _provider_settings_owner(
        path,
        MemorySecretStore({"openrouter_api_key": raw_secret}),
    )
    provider_settings.persist_verification("openrouter", raw_secret, True)

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

    provider_settings.persist_verification("openrouter", raw_secret, False)

    invalidated = load_vnext_settings(path)
    assert invalidated.settings is not None
    assert invalidated.settings.state.provider_verification.openrouter.status == "unknown"


def test_settings_owner_rejects_verification_for_nonmatching_secret_store_value(
    tmp_path: Path,
) -> None:
    path = tmp_path / "settings.json"
    provider_settings = _provider_settings_owner(
        path,
        MemorySecretStore({"openrouter_api_key": "different-secret"}),
    )

    with pytest.raises(
        RuntimeError,
        match="verified credential does not match the active SecretStore value",
    ):
        provider_settings.persist_verification(
            "openrouter",
            "verified-but-not-stored",
            True,
        )

    owner = provider_settings.settings
    assert owner.current is not None
    assert owner.current.api_key_verified.openrouter is False
    assert owner.canonical is not None
    assert owner.canonical.state.provider_verification.openrouter.status == "unknown"
    assert owner.mutation_depth == 0
    assert not path.exists()


@pytest.mark.asyncio
async def test_provider_secret_change_invalidates_before_reverification_and_relaunch(
    tmp_path: Path,
) -> None:
    path = tmp_path / "settings.json"
    store = MemorySecretStore({"openrouter_api_key": "old-secret"})
    provider_settings = _owner_with_verified_openrouter(path, store)

    assert await provider_settings.change_secret(
        "openrouter_api_key",
        "new-secret",
    )

    assert store.get("openrouter_api_key") == "new-secret"
    owner = provider_settings.settings
    assert owner.current is not None
    assert owner.current.api_key_verified.openrouter is False
    assert owner.canonical is not None
    assert owner.canonical.state.provider_verification.openrouter.status == "unknown"
    reloaded = load_vnext_settings(path)
    assert reloaded.settings is not None
    assert reloaded.settings.state.provider_verification.openrouter.status == "unknown"


@pytest.mark.asyncio
async def test_http_extension_secret_change_uses_transaction_without_settings_write(
    tmp_path: Path,
) -> None:
    path = tmp_path / "settings.json"
    store = MemorySecretStore()
    provider_settings = _provider_settings_owner(path, store)
    assert provider_settings.settings.current is not None
    before = copy.deepcopy(provider_settings.settings.current)

    assert await provider_settings.change_secret(
        "http_extension.demo.api_key",
        "extension-secret",
    )

    assert store.get("http_extension.demo.api_key") == "extension-secret"
    assert provider_settings.settings.current == before
    assert not path.exists()


@pytest.mark.asyncio
async def test_provider_secret_change_restores_secret_and_verification_on_commit_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "settings.json"
    store = MemorySecretStore({"openrouter_api_key": "old-secret"})
    provider_settings = _owner_with_verified_openrouter(path, store)
    persisted_before = path.read_bytes()

    def fail_persist(_path: Path, _settings: AppSettingsVNext) -> None:
        raise OSError("injected persistence failure")

    monkeypatch.setattr(provider_settings.settings.persistence, "persist", fail_persist)

    assert not await provider_settings.change_secret(
        "openrouter_api_key",
        "new-secret",
    )

    assert store.get("openrouter_api_key") == "old-secret"
    owner = provider_settings.settings
    assert owner.current is not None
    assert owner.current.api_key_verified.openrouter is True
    assert owner.canonical is not None
    assert owner.canonical.state.provider_verification.openrouter.status == "verified"
    assert path.read_bytes() == persisted_before


@pytest.mark.asyncio
async def test_provider_secret_change_finishes_invalidation_when_caller_is_cancelled(
    tmp_path: Path,
) -> None:
    path = tmp_path / "settings.json"
    store = MemorySecretStore({"openrouter_api_key": "old-secret"})
    store.block_set = True
    provider_settings = _owner_with_verified_openrouter(path, store)
    task = asyncio.create_task(
        provider_settings.change_secret(
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
    assert provider_settings.settings.current is not None
    assert provider_settings.settings.current.api_key_verified.openrouter is False
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
    provider_settings = _owner_with_verified_openrouter(path, store)
    owner = provider_settings.settings
    assert owner.current is not None
    owner.current.api_key_verified.deepseek = True
    owner.bind_provider_verification(
        ProviderVerificationBinding(
            provider="deepseek",
            secret_key="deepseek_api_key",
            secret_revision=None,
            secret_fingerprint="sha256:old-deepseek-secret",
            verifier_context={"flow": "settings_api_key_verification"},
            verifier_evidence={"source": "provider_verifier"},
        )
    )
    owner.persist()
    owner.remember_projection(owner.current)

    first_task = asyncio.create_task(provider_settings.change_secret(first_key, f"new-{first_key}"))
    assert await asyncio.to_thread(store.started[first_key].wait, 2)
    second_task = asyncio.create_task(
        provider_settings.change_secret(second_key, f"new-{second_key}")
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
    assert owner.current is not None
    assert owner.current.api_key_verified.openrouter is False
    assert owner.current.api_key_verified.deepseek is False
    assert owner.canonical is not None
    assert owner.canonical.state.provider_verification.openrouter.status == "unknown"
    assert owner.canonical.state.provider_verification.deepseek.status == "unknown"
    reloaded = load_vnext_settings(path)
    assert reloaded.settings is not None
    assert reloaded.settings.state.provider_verification.openrouter.status == "unknown"
    assert reloaded.settings.state.provider_verification.deepseek.status == "unknown"
