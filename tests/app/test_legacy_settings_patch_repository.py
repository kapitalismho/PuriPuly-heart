from __future__ import annotations

import copy
import threading
from pathlib import Path

import pytest

from puripuly_heart.app.adapters.settings_vnext_canonical_persistence import (
    SettingsVNextCanonicalPersistenceAdapter,
)
from puripuly_heart.app.ports.canonical_settings_persistence import (
    ProviderVerificationBinding,
)
from puripuly_heart.app.ports.settings_repository import SettingsCommitRequest
from puripuly_heart.app.services.canonical_settings_persistence import (
    LegacySettingsPatchRepository,
    SettingsOwner,
    legacy_settings_snapshot_values,
)
from puripuly_heart.config.settings import AppSettings, LLMProviderName
from puripuly_heart.config.settings_vnext.schema import AppSettingsVNext


class RecordingPersistence(SettingsVNextCanonicalPersistenceAdapter):
    def __init__(self) -> None:
        self.events: list[str] = []
        self.persist_thread_ids: list[int] = []
        self.fail_persist = False

    def apply_legacy_delta(self, **kwargs):
        self.events.append("update")
        return super().apply_legacy_delta(**kwargs)

    def bind_provider_verification(self, canonical, binding):
        self.events.append("bind")
        return super().bind_provider_verification(canonical, binding)

    def persist(self, path: Path, settings: AppSettingsVNext) -> None:
        _ = path, settings
        self.events.append("persist")
        self.persist_thread_ids.append(threading.get_ident())
        if self.fail_persist:
            raise OSError("injected persistence failure")


def _repository(
    *,
    committed: AppSettings,
    base: AppSettings | None = None,
    surface: str = "translation_provider",
    binding: ProviderVerificationBinding | None = None,
) -> tuple[LegacySettingsPatchRepository, SettingsOwner, RecordingPersistence]:
    persistence = RecordingPersistence()
    owner = SettingsOwner(
        path=Path("settings.json"),
        persistence=persistence,
        canonical=AppSettingsVNext(),
        current=copy.deepcopy(base or committed),
        authoritative=True,
        projection_snapshot=copy.deepcopy(base or committed),
    )
    repository = LegacySettingsPatchRepository(
        owner=owner,
        committed_settings=committed,
        base_settings=base,
        surface=surface,
        provider_verification_binding=binding,
        save_failure_sink=lambda message: persistence.events.append(message),
    )
    return repository, owner, persistence


@pytest.mark.asyncio
async def test_repository_applies_path_patch_and_persists_off_event_loop() -> None:
    base = AppSettings()
    committed = copy.deepcopy(base)
    repository, owner, persistence = _repository(committed=committed, base=base)
    event_loop_thread_id = threading.get_ident()

    result = await repository.save(
        SettingsCommitRequest(
            values={"ui.locale": "ja"},
            expected_revision=None,
            reason="settings.ui_prompt_clipboard_state",
        )
    )

    assert result.succeeded is True
    assert repository.committed_settings.ui.locale == "ja"
    assert persistence.persist_thread_ids != [event_loop_thread_id]
    assert persistence.events == ["update", "persist"]
    assert owner.projection_snapshot is not None
    assert owner.projection_snapshot.ui.locale == "ja"


@pytest.mark.asyncio
async def test_repository_applies_managed_delivery_ack_state_patch() -> None:
    committed = AppSettings()
    repository, _owner, _persistence = _repository(
        committed=committed,
        surface="managed_connection_auth",
    )

    result = await repository.save(
        SettingsCommitRequest(
            values={
                "state": {
                    "managed_connection": {
                        "pending_delivery_ack_source": "discord",
                        "pending_delivery_ack_delivery_id": "delivery-1",
                        "pending_delivery_ack_managed_credential_ref": "managed-ref-1",
                        "verified_hardware_hash_salt_version": 3,
                    }
                }
            },
            expected_revision=None,
            reason="managed_connection_auth",
        )
    )

    managed = repository.committed_settings.managed_identity
    assert result.succeeded is True
    assert managed.pending_delivery_ack_source == "discord"
    assert managed.pending_delivery_ack_delivery_id == "delivery-1"
    assert managed.pending_delivery_ack_managed_credential_ref == "managed-ref-1"
    assert managed.verified_hardware_hash_salt_version == 3


@pytest.mark.asyncio
async def test_repository_binds_provider_verification_before_persistence() -> None:
    binding = ProviderVerificationBinding(
        provider="openrouter",
        secret_key="openrouter_api_key",
        secret_revision=None,
        secret_fingerprint="sha256:fingerprint",
        verifier_context={"flow": "pkce"},
        verifier_evidence={"source": "provider_verifier"},
    )
    committed = AppSettings()
    committed.provider.llm = LLMProviderName.OPENROUTER
    repository, owner, persistence = _repository(
        committed=committed,
        binding=binding,
    )

    result = await repository.save(
        SettingsCommitRequest(
            values=legacy_settings_snapshot_values(committed),
            expected_revision=None,
            reason="openrouter_pkce",
        )
    )

    assert result.succeeded is True
    assert persistence.events.index("bind") < persistence.events.index("persist")
    assert owner.canonical is not None
    assert owner.canonical.state.provider_verification.openrouter.status == "verified"


@pytest.mark.asyncio
async def test_repository_rolls_back_and_returns_safe_diagnostics_on_save_failure() -> None:
    committed = AppSettings()
    repository, owner, persistence = _repository(
        committed=committed,
        surface="provider_secret_change",
    )
    persistence.fail_persist = True

    result = await repository.save(
        SettingsCommitRequest(
            values={"ui.locale": "ko"},
            expected_revision=None,
            reason="provider_secret_change",
        )
    )

    assert result.succeeded is False
    assert result.snapshot is None
    assert result.diagnostics is not None
    assert result.diagnostics.code == "settings_save_failed"
    assert result.diagnostics.fields["surface"] == "provider_secret_change"
    assert repository.committed_settings.ui.locale != "ko"
    assert owner.mutation_depth == 0
    assert persistence.events == [
        "update",
        "persist",
        "Failed to save settings mutation",
    ]


def test_legacy_settings_snapshot_values_normalizes_enums_and_tuples() -> None:
    settings = AppSettings()
    settings.provider.llm = LLMProviderName.OPENROUTER
    settings.managed_identity.local_managed_claim_sources = ("discord",)

    values = legacy_settings_snapshot_values(settings)

    assert values["provider"]["llm"] == "openrouter"
    assert values["managed_identity"]["local_managed_claim_sources"] == ["discord"]
