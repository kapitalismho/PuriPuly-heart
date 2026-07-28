from __future__ import annotations

import copy
import threading
from dataclasses import dataclass, field

import pytest

from puripuly_heart.app.ports.canonical_settings_persistence import (
    ProviderVerificationBinding,
)
from puripuly_heart.app.ports.settings_repository import SettingsCommitRequest
from puripuly_heart.app.services.canonical_settings_persistence import (
    LegacySettingsPatchCallbacks,
    LegacySettingsPatchRepository,
    legacy_settings_snapshot_values,
)
from puripuly_heart.config.settings import AppSettings, LLMProviderName


@dataclass
class CallbackHarness:
    current: AppSettings | None = None
    canonical_snapshot: AppSettings | None = None
    events: list[str] = field(default_factory=list)
    persisted: list[AppSettings] = field(default_factory=list)
    persist_thread_ids: list[int] = field(default_factory=list)
    bindings: list[ProviderVerificationBinding] = field(default_factory=list)
    updates: list[tuple[AppSettings, AppSettings]] = field(default_factory=list)
    fail_persist: bool = False

    def begin(self) -> None:
        self.events.append("begin")

    def update(self, baseline: AppSettings, current: AppSettings) -> None:
        self.events.append("update")
        self.updates.append((copy.deepcopy(baseline), copy.deepcopy(current)))

    def bind(self, binding: ProviderVerificationBinding) -> None:
        self.events.append("bind")
        self.bindings.append(binding)

    def persist(self, settings: AppSettings) -> None:
        self.events.append("persist")
        self.persist_thread_ids.append(threading.get_ident())
        if self.fail_persist:
            raise OSError("injected persistence failure")
        self.persisted.append(copy.deepcopy(settings))

    def rollback(self) -> None:
        self.events.append("rollback")

    def fail(self, message: str) -> None:
        self.events.append(message)

    def remember(self, settings: AppSettings) -> None:
        self.events.append("remember")
        self.current = copy.deepcopy(settings)


def _repository(
    harness: CallbackHarness,
    *,
    committed: AppSettings,
    base: AppSettings | None = None,
    surface: str = "translation_provider",
    binding: ProviderVerificationBinding | None = None,
) -> LegacySettingsPatchRepository:
    return LegacySettingsPatchRepository(
        callbacks=LegacySettingsPatchCallbacks(
            current_settings=lambda: harness.current,
            canonical_projection_snapshot=lambda: harness.canonical_snapshot,
            begin_canonical_mutation=harness.begin,
            update_canonical_from_legacy_delta=harness.update,
            bind_provider_verification=harness.bind,
            persist_settings=harness.persist,
            rollback_canonical_mutation=harness.rollback,
            save_failure_sink=harness.fail,
            remember_canonical_projection=harness.remember,
        ),
        committed_settings=committed,
        base_settings=base,
        surface=surface,
        provider_verification_binding=binding,
    )


@pytest.mark.asyncio
async def test_repository_applies_path_patch_and_persists_off_event_loop() -> None:
    base = AppSettings()
    committed = copy.deepcopy(base)
    harness = CallbackHarness(canonical_snapshot=copy.deepcopy(base))
    repository = _repository(harness, committed=committed, base=base)
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
    assert harness.persisted[0].ui.locale == "ja"
    assert harness.persist_thread_ids != [event_loop_thread_id]
    assert harness.events == ["begin", "update", "persist", "remember"]


@pytest.mark.asyncio
async def test_repository_applies_managed_delivery_ack_state_patch() -> None:
    committed = AppSettings()
    harness = CallbackHarness()
    repository = _repository(
        harness,
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
    harness = CallbackHarness()
    repository = _repository(harness, committed=committed, binding=binding)

    result = await repository.save(
        SettingsCommitRequest(
            values=legacy_settings_snapshot_values(committed),
            expected_revision=None,
            reason="openrouter_pkce",
        )
    )

    assert result.succeeded is True
    assert harness.bindings == [binding]
    assert harness.events.index("bind") < harness.events.index("persist")


@pytest.mark.asyncio
async def test_repository_rolls_back_and_returns_safe_diagnostics_on_save_failure() -> None:
    committed = AppSettings()
    harness = CallbackHarness(fail_persist=True)
    repository = _repository(
        harness,
        committed=committed,
        surface="provider_secret_change",
    )

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
    assert harness.events == [
        "begin",
        "update",
        "persist",
        "rollback",
        "Failed to save settings mutation",
    ]


def test_legacy_settings_snapshot_values_normalizes_enums_and_tuples() -> None:
    settings = AppSettings()
    settings.provider.llm = LLMProviderName.OPENROUTER
    settings.managed_identity.local_managed_claim_sources = ("discord",)

    values = legacy_settings_snapshot_values(settings)

    assert values["provider"]["llm"] == "openrouter"
    assert values["managed_identity"]["local_managed_claim_sources"] == ["discord"]
