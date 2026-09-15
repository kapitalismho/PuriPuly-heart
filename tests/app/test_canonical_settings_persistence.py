from __future__ import annotations

import asyncio
import copy
import json
import threading
from dataclasses import replace
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
from puripuly_heart.app.services.canonical_settings_persistence import (
    SettingsOwner,
    compose_settings_owner,
    materialize_canonical_translation_settings,
)
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


def _with_locale(settings: AppSettingsVNext, locale: str) -> AppSettingsVNext:
    return replace(
        settings,
        intent=replace(settings.intent, ui=replace(settings.intent.ui, locale=locale)),
    )


def _with_referral(settings: AppSettingsVNext, referral_id: str) -> AppSettingsVNext:
    return replace(
        settings,
        state=replace(
            settings.state,
            managed_connection=replace(
                settings.state.managed_connection,
                referral_id=referral_id,
            ),
        ),
    )


def _provider_settings_owner(
    path: Path,
    store: MemorySecretStore,
) -> ProviderSettingsOwner:
    canonical = AppSettingsVNext()
    owner = SettingsOwner(
        path=path,
        persistence=SettingsVNextCanonicalPersistenceAdapter(),
        canonical=canonical,
        authoritative=True,
        projection_snapshot=copy.deepcopy(canonical),
    )
    return ProviderSettingsOwner(
        settings=owner,
        binding=ProviderVerificationBindingOwner(
            context_provider=lambda provider: provider_verification_context(
                owner.canonical,
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
    assert owner.canonical is not None
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
    owner.remember_projection(owner.require_canonical())
    return provider_settings


def test_canonical_settings_persistence_port_covers_load_save_and_rollback(
    monkeypatch,
) -> None:
    from puripuly_heart.config.settings_vnext.migration import apply_canonical_delta

    adapter = SettingsVNextCanonicalPersistenceAdapter()
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

    updated_canonical = apply_canonical_delta(
        canonical,
        canonical,
        _with_locale(canonical, "ja"),
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
        canonical=updated_canonical,
        authoritative=True,
        projection_snapshot=copy.deepcopy(updated_canonical),
    )
    assert owner.save_current()
    assert owner.require_canonical().intent.ui.locale == "ja"
    assert owner.projection_snapshot is not None
    assert owner.projection_snapshot.intent.ui.locale == "ja"
    assert owner.mutation_depth == 0

    failures: list[BaseException] = []

    def fail_save(_path: Path, _value: AppSettingsVNext) -> None:
        raise OSError("injected save failure")

    monkeypatch.setattr(adapter_module, "save_vnext_settings", fail_save)
    owner.canonical = _with_locale(owner.require_canonical(), "ko")
    assert not owner.save_current(failure_sink=failures.append)
    assert len(failures) == 1
    assert isinstance(failures[0], OSError)
    assert owner.require_canonical().intent.ui.locale == "ja"
    assert owner.mutation_depth == 0

    with pytest.raises(OSError, match="injected save failure"):
        owner.persist_current()

    monkeypatch.setattr(
        adapter_module,
        "save_vnext_settings",
        lambda _path, value: saved.append(value) or SimpleNamespace(ok=True),
    )
    stale_settings = copy.deepcopy(owner.require_canonical())
    persist_managed_identity = owner.managed_identity_persistence_callback(stale_settings)
    active_settings = _with_locale(stale_settings, "ru")
    owner.canonical = active_settings
    owner.remember_projection(active_settings)
    owner.persist_current()

    persist_managed_identity(_with_referral(stale_settings, "234567"))

    assert owner.require_canonical().intent.ui.locale == "ru"
    assert owner.require_canonical().state.managed_connection.referral_id == "234567"

    active_before_failure = copy.deepcopy(owner.require_canonical())
    monkeypatch.setattr(adapter_module, "save_vnext_settings", fail_save)
    with pytest.raises(OSError, match="injected save failure"):
        persist_managed_identity(_with_referral(stale_settings, "345678"))
    assert owner.canonical == active_before_failure


def test_canonical_delta_requires_bound_evidence_and_preserves_invalidation() -> None:
    from puripuly_heart.config.settings_vnext.migration import apply_canonical_delta
    from puripuly_heart.config.settings_vnext.schema import ProviderVerificationEntry

    adapter = SettingsVNextCanonicalPersistenceAdapter()
    baseline = AppSettingsVNext()
    claimed_verified = replace(
        baseline,
        state=replace(
            baseline.state,
            provider_verification=replace(
                baseline.state.provider_verification,
                openrouter=ProviderVerificationEntry(
                    status="verified",
                    provider="openrouter",
                    secret_key="openrouter_api_key",
                    secret_fingerprint="sha256:forged",
                    verifier_context={"flow": "settings_api_key_verification"},
                    verifier_evidence={"source": "forged"},
                ),
            ),
        ),
    )

    unbound = apply_canonical_delta(AppSettingsVNext(), baseline, claimed_verified)

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

    invalidated = apply_canonical_delta(
        verified,
        claimed_verified,
        replace(
            claimed_verified,
            state=replace(
                claimed_verified.state,
                provider_verification=replace(
                    claimed_verified.state.provider_verification,
                    openrouter=ProviderVerificationEntry(status="unknown"),
                ),
            ),
        ),
    )

    assert invalidated.state.provider_verification.openrouter.status == "unknown"


def _seed_retired_managed_gemma_install(models_dir: Path) -> Path:
    from puripuly_heart.core.local_translation import assets

    retired_id, retired_filename = assets.RETIRED_MANAGED_GEMMA_INSTALLS[0]
    install_dir = models_dir / retired_id
    install_dir.mkdir(parents=True, exist_ok=True)
    (install_dir / retired_filename).write_bytes(b"retired-weights")
    return install_dir


def _write_v42_managed_gemma_12b_settings(path: Path) -> None:
    from puripuly_heart.config.settings_vnext import serialization

    raw = serialization.to_dict(AppSettingsVNext())
    raw["settings_version"] = 42
    raw["intent"]["translation"].update({"model": "managed_gemma_12b", "connection": "gpu"})
    path.write_text(json.dumps(raw), encoding="utf-8")


def test_settings_migration_removes_retired_managed_gemma_install_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from puripuly_heart.core.local_translation import assets

    models_dir = tmp_path / "models"
    monkeypatch.setattr(assets, "default_models_dir", lambda: models_dir)
    path = tmp_path / "settings.json"
    _write_v42_managed_gemma_12b_settings(path)
    install_dir = _seed_retired_managed_gemma_install(models_dir)

    started = compose_settings_owner(path).start()

    assert started.migrated is True
    assert started.settings.intent.translation.model == "managed_gemma"
    assert not install_dir.exists()

    leftover = _seed_retired_managed_gemma_install(models_dir)

    steady_state = compose_settings_owner(path).start()

    assert steady_state.migrated is False
    assert leftover.is_dir()


def test_retired_managed_gemma_sweep_emits_safe_bounded_runtime_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from puripuly_heart.core.local_translation import assets

    class RuntimeLogging:
        def __init__(self) -> None:
            self.messages: list[str] = []

        def emit_basic(self, message: str, **_kwargs: object) -> None:
            self.messages.append(message)

        def emit_diagnostic(self, _message: str, **_kwargs: object) -> bool:
            return False

    models_dir = tmp_path / "private-models"
    monkeypatch.setattr(assets, "default_models_dir", lambda: models_dir)
    path = tmp_path / "settings.json"
    _write_v42_managed_gemma_12b_settings(path)
    retired_id, _retired_filename = assets.RETIRED_MANAGED_GEMMA_INSTALLS[0]
    install_dir = _seed_retired_managed_gemma_install(models_dir)
    locked = models_dir / f"{retired_id}.staging-private-token"
    locked.mkdir()
    real_rmtree = assets.shutil.rmtree

    def flaky_rmtree(target, *args, **kwargs):
        if str(target) == str(locked):
            raise OSError("private cleanup failure")
        return real_rmtree(target, *args, **kwargs)

    monkeypatch.setattr(assets.shutil, "rmtree", flaky_rmtree)
    runtime_logging = RuntimeLogging()

    started = compose_settings_owner(
        path,
        retired_asset_cleanup_logging=runtime_logging,
    ).start()

    assert started.migrated is True
    assert not install_dir.exists()
    assert locked.is_dir()
    assert len(runtime_logging.messages) == 1
    receipt = runtime_logging.messages[0]
    assert "outcome=partial" in receipt
    assert "removed=1" in receipt
    assert "failed=1" in receipt
    assert "failure_types=OSError" in receipt
    assert str(models_dir) not in receipt
    assert "private cleanup failure" not in receipt


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
    assert provider_settings.settings.canonical is not None
    before = copy.deepcopy(provider_settings.settings.canonical)

    assert await provider_settings.change_secret(
        "http_extension.demo.api_key",
        "extension-secret",
    )

    assert store.get("http_extension.demo.api_key") == "extension-secret"
    assert provider_settings.settings.canonical == before
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
    assert provider_settings.settings.canonical is not None
    assert (
        provider_settings.settings.canonical.state.provider_verification.openrouter.status
        == "unknown"
    )
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
    owner.remember_projection(owner.require_canonical())

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
    assert owner.canonical is not None
    assert owner.canonical.state.provider_verification.openrouter.status == "unknown"
    assert owner.canonical.state.provider_verification.deepseek.status == "unknown"
    reloaded = load_vnext_settings(path)
    assert reloaded.settings is not None
    assert reloaded.settings.state.provider_verification.openrouter.status == "unknown"
    assert reloaded.settings.state.provider_verification.deepseek.status == "unknown"


@pytest.mark.parametrize("connection", ["managed", "managed_china", "openrouter"])
def test_deepseek_40_materialization_restores_shipped_identity(connection: str) -> None:
    canonical = AppSettingsVNext()
    translation = replace(
        canonical.intent.translation,
        model="deepseek_v4_flash",
        connection=connection,
        openrouter_model="google/gemma-4-31b-it",
        openrouter_provider_routing="gemma4_31b_latency",
    )

    result = materialize_canonical_translation_settings(
        replace(canonical, intent=replace(canonical.intent, translation=translation))
    ).intent.translation

    assert result.model == "deepseek_v4_flash"
    assert result.openrouter_model == "deepseek/deepseek-v4-flash-0731"
    expected_route = (
        "deepseek_v4_flash_china" if connection == "managed_china" else "deepseek_v4_flash_latency"
    )
    assert result.openrouter_provider_routing == expected_route
    expected_alias = (
        "deepseek_v4_flash_byok" if connection == "openrouter" else "deepseek_v4_flash_managed"
    )
    assert result.openrouter_selection_alias == expected_alias


@pytest.mark.parametrize("connection", ["managed", "managed_china", "openrouter"])
def test_deepseek_41_materialization_persists_distinct_identity(connection: str) -> None:
    canonical = AppSettingsVNext()
    translation = replace(
        canonical.intent.translation,
        model="deepseek_v4_flash_41",
        connection=connection,
        openrouter_model="deepseek/deepseek-v4-flash-0731",
        openrouter_provider_routing="deepseek_v4_flash_latency",
    )

    result = materialize_canonical_translation_settings(
        replace(canonical, intent=replace(canonical.intent, translation=translation))
    ).intent.translation

    assert result.model == "deepseek_v4_flash_41"
    assert result.openrouter_model == "deepseek/deepseek-v4.1-flash"
    assert result.openrouter_provider_routing == "deepseek_v4_flash_41_strict"
    expected_alias = (
        "deepseek_v4_flash_41_byok"
        if connection == "openrouter"
        else "deepseek_v4_flash_41_managed"
    )
    assert result.openrouter_selection_alias == expected_alias


def test_old_official_deepseek_primary_materializes_as_41_direct() -> None:
    canonical = AppSettingsVNext()
    translation = replace(
        canonical.intent.translation,
        model="deepseek_v4_flash",
        connection="official_byok",
    )

    result = materialize_canonical_translation_settings(
        replace(canonical, intent=replace(canonical.intent, translation=translation))
    ).intent.translation

    assert result.model == "deepseek_v4_flash_41"
    assert result.connection == "official_byok"
    assert result.deepseek.llm_model == "deepseek-flash"
