from __future__ import annotations

import importlib
from collections.abc import Mapping
from dataclasses import FrozenInstanceError, dataclass, fields, is_dataclass
from typing import Any

import pytest

from puripuly_heart.app.ports import (
    broker_client,
    managed_identity_state,
    secret_store,
    settings_repository,
)
from puripuly_heart.app.services.managed_auth_claims import (
    MANAGED_AUTH_CLAIM_SOURCE_DISCORD,
    MANAGED_AUTH_CLAIM_SOURCE_QQ,
    OPENROUTER_MANAGED_QQ_API_KEY_SECRET,
    OPENROUTER_MANAGED_USER_ID_SECRET,
    OPENROUTER_MANAGED_USER_INSTALLATION_ID_SECRET,
    ManagedAuthClaimGuard,
)
from puripuly_heart.core import messages

SERVICE_MODULE = "puripuly_heart.app.services.managed_connection_auth"
DISCORD_AUTH_PORT_MODULE = "puripuly_heart.app.ports.discord_auth"
LOCAL_IDENTITY_PORT_MODULE = "puripuly_heart.app.ports.managed_identity"

LOCAL_SECRET_KEY = "openrouter_managed_api_key"
RAW_MANAGED_CREDENTIAL = "mconn-order30-managed-credential-must-not-leak"
RAW_EXCEPTION_TEXT = "mconn-order30-raw-exception-text-must-not-leak"
RAW_BROKER_PAYLOAD = "mconn-order30-broker-payload-must-not-leak"

FORBIDDEN_RAW_VALUES = (
    RAW_MANAGED_CREDENTIAL,
    RAW_EXCEPTION_TEXT,
    RAW_BROKER_PAYLOAD,
)


class RecordingLocalIdentity:
    def __init__(
        self,
        result: Any,
        *,
        events: list[tuple[str, str]] | None = None,
        raise_on_preflight: bool = False,
    ) -> None:
        self.result = result
        self.events = events if events is not None else []
        self.raise_on_preflight = raise_on_preflight
        self.requests: list[Any] = []

    async def preflight_managed_identity(self, request: Any) -> Any:
        self.events.append(("identity_preflight", request.local_secret_key))
        self.requests.append(request)
        if self.raise_on_preflight:
            raise RuntimeError(RAW_EXCEPTION_TEXT)
        return self.result


class RecordingDiscordAuth:
    def __init__(
        self,
        result: Any,
        *,
        events: list[tuple[str, str]] | None = None,
        raise_on_start: bool = False,
    ) -> None:
        self.result = result
        self.events = events if events is not None else []
        self.raise_on_start = raise_on_start
        self.requests: list[Any] = []

    async def start_discord_auth(self, request: Any) -> Any:
        self.events.append(("discord_auth", request.correlation_id or ""))
        self.requests.append(request)
        if self.raise_on_start:
            raise RuntimeError(RAW_EXCEPTION_TEXT)
        return self.result


class RecordingBrokerClient:
    def __init__(
        self,
        result: broker_client.BrokerIssueResult | None,
        *,
        events: list[tuple[str, str]] | None = None,
        raise_on_issue: bool = False,
        ack_result: broker_client.ManagedKeyDeliveryAckResult | None = None,
    ) -> None:
        self.result = result
        self.events = events if events is not None else []
        self.raise_on_issue = raise_on_issue
        self.ack_result = ack_result or broker_client.ManagedKeyDeliveryAckResult(
            succeeded=True,
            status="acknowledged",
        )
        self.requests: list[broker_client.BrokerIssueRequest] = []
        self.ack_requests: list[broker_client.ManagedKeyDeliveryAckRequest] = []

    async def issue_managed_connection(
        self,
        request: broker_client.BrokerIssueRequest,
    ) -> broker_client.BrokerIssueResult:
        self.events.append(("broker_issue", request.discord_user_id))
        self.requests.append(request)
        if self.raise_on_issue:
            raise RuntimeError(RAW_EXCEPTION_TEXT)
        if self.result is None:
            pytest.fail("RecordingBrokerClient requires a configured result")
        return self.result

    async def acknowledge_managed_key_delivery(
        self,
        request: broker_client.ManagedKeyDeliveryAckRequest,
    ) -> broker_client.ManagedKeyDeliveryAckResult:
        self.events.append(("delivery_ack", request.delivery_id))
        self.ack_requests.append(request)
        return self.ack_result


class RecordingSecretStore:
    def __init__(
        self,
        *,
        events: list[tuple[str, str]] | None = None,
        write_succeeds: bool = True,
        raise_on_set: bool = False,
        clear_succeeds: bool = True,
        fail_set_keys: set[str] | None = None,
    ) -> None:
        self.events = events if events is not None else []
        self.write_succeeds = write_succeeds
        self.raise_on_set = raise_on_set
        self.clear_succeeds = clear_succeeds
        self.fail_set_keys = fail_set_keys or set()
        self.values: dict[str, str] = {}
        self.set_calls: list[tuple[str, str]] = []

    async def get_secret(self, key: str) -> secret_store.SecretReadResult:
        return secret_store.SecretReadResult(
            key=key,
            value=self.values.get(key),
            revision="secret-current" if key in self.values else None,
            message=None,
            diagnostics=None,
        )

    async def set_secret(self, key: str, value: str) -> secret_store.SecretWriteResult:
        self.events.append(("set_secret", key))
        self.set_calls.append((key, value))
        if self.raise_on_set:
            raise RuntimeError(RAW_EXCEPTION_TEXT)
        succeeded = self.write_succeeds and key not in self.fail_set_keys
        if succeeded:
            self.values[key] = value
        return secret_store.SecretWriteResult(
            succeeded=succeeded,
            key=key,
            revision="secret-r2" if succeeded else None,
            message=None,
            diagnostics=(None if succeeded else _unsafe_diagnostics("secret_store")),
        )

    async def clear_secret(self, key: str) -> secret_store.SecretWriteResult:
        self.events.append(("clear_secret", key))
        if self.clear_succeeds:
            self.values.pop(key, None)
        return secret_store.SecretWriteResult(
            succeeded=self.clear_succeeds,
            key=key,
            revision="secret-cleared" if self.clear_succeeds else None,
            message=None,
            diagnostics=(None if self.clear_succeeds else _safe_diagnostics("secret_clear_failed")),
        )

    async def snapshot_secret(self, key: str) -> secret_store.SecretSnapshot:
        value = self.values.get(key)
        return secret_store.SecretSnapshot(
            key=key,
            value=value,
            revision="secret-current" if value is not None else None,
            existed=value is not None,
        )

    async def restore_secret(
        self,
        snapshot: secret_store.SecretSnapshot,
    ) -> secret_store.SecretWriteResult:
        self.events.append(("restore_secret", snapshot.key))
        if snapshot.existed:
            assert snapshot.value is not None
            self.values[snapshot.key] = snapshot.value
        else:
            self.values.pop(snapshot.key, None)
        return secret_store.SecretWriteResult(
            succeeded=True,
            key=snapshot.key,
            revision="secret-restored",
            message=None,
            diagnostics=None,
        )


class RecordingSettingsRepository:
    def __init__(
        self,
        result: settings_repository.SettingsCommitResult | None,
        *,
        events: list[tuple[str, str]] | None = None,
        raise_on_save: bool = False,
        results: list[settings_repository.SettingsCommitResult] | None = None,
    ) -> None:
        self.result = result
        self.results = list(results or [])
        self.events = events if events is not None else []
        self.raise_on_save = raise_on_save
        self.saved_requests: list[settings_repository.SettingsCommitRequest] = []

    async def load(self) -> settings_repository.SettingsSnapshot:
        raise AssertionError("ManagedConnectionAuthService should not load settings")

    async def save(
        self,
        request: settings_repository.SettingsCommitRequest,
    ) -> settings_repository.SettingsCommitResult:
        self.events.append(("save_settings", request.reason or ""))
        self.saved_requests.append(request)
        if self.raise_on_save:
            raise RuntimeError(RAW_EXCEPTION_TEXT)
        if self.results:
            return self.results.pop(0)
        if self.result is None:
            pytest.fail("RecordingSettingsRepository requires a configured result")
        return self.result


@dataclass
class RecordingManagedState:
    installation_id: str = "install-123"
    release_token: str | None = None
    release_token_expires_at: str | None = None
    verified_hardware_hash: str | None = None
    verified_hardware_hash_salt_version: int | None = None
    active_managed_credential_ref: str | None = None
    active_managed_expires_at: str | None = None
    founder_letter_seen_credential_ref: str | None = None
    referral_id: str | None = None
    local_managed_claim_sources: tuple[str, ...] = ()
    persist_calls: int = 0

    def persist(self) -> None:
        self.persist_calls += 1

    def snapshot(self) -> managed_identity_state.ManagedIdentitySnapshot:
        return managed_identity_state.ManagedIdentitySnapshot(
            installation_id=self.installation_id,
            release_token=self.release_token,
            release_token_expires_at=self.release_token_expires_at,
            verified_hardware_hash=self.verified_hardware_hash,
            verified_hardware_hash_salt_version=self.verified_hardware_hash_salt_version,
            active_managed_credential_ref=self.active_managed_credential_ref,
            active_managed_expires_at=self.active_managed_expires_at,
            founder_letter_seen_credential_ref=self.founder_letter_seen_credential_ref,
            referral_id=self.referral_id,
            local_managed_claim_sources=self.local_managed_claim_sources,
        )

    def restore(self, snapshot: managed_identity_state.ManagedIdentitySnapshot) -> None:
        self.installation_id = snapshot.installation_id
        self.release_token = snapshot.release_token
        self.release_token_expires_at = snapshot.release_token_expires_at
        self.verified_hardware_hash = snapshot.verified_hardware_hash
        self.verified_hardware_hash_salt_version = snapshot.verified_hardware_hash_salt_version
        self.active_managed_credential_ref = snapshot.active_managed_credential_ref
        self.active_managed_expires_at = snapshot.active_managed_expires_at
        self.founder_letter_seen_credential_ref = snapshot.founder_letter_seen_credential_ref
        self.referral_id = snapshot.referral_id
        self.local_managed_claim_sources = snapshot.local_managed_claim_sources


def _service_module() -> Any:
    return importlib.import_module(SERVICE_MODULE)


def _discord_auth_module() -> Any:
    return importlib.import_module(DISCORD_AUTH_PORT_MODULE)


def _managed_identity_module() -> Any:
    return importlib.import_module(LOCAL_IDENTITY_PORT_MODULE)


def _service(
    *,
    identity: RecordingLocalIdentity,
    discord: RecordingDiscordAuth,
    broker: RecordingBrokerClient,
    store: RecordingSecretStore,
    repository: RecordingSettingsRepository,
    claim_guard: ManagedAuthClaimGuard | None = None,
) -> Any:
    auth = _service_module()
    return auth.ManagedConnectionAuthService(
        local_identity=identity,
        discord_auth=discord,
        broker_client=broker,
        secret_store=store,
        settings_repository=repository,
        claim_guard=claim_guard,
    )


def _request(settings_values: Mapping[str, object] | None = None) -> Any:
    auth = _service_module()
    request = auth.ManagedConnectionAuthRequest(
        local_secret_key=LOCAL_SECRET_KEY,
        settings_values=settings_values
        or {
            "intent": {
                "translation": {
                    "connection": "managed",
                    "model": "gemma4",
                }
            },
            "state": {
                "managed_connection": {
                    "active_managed_credential_ref": "hash_123",
                }
            },
        },
        expected_settings_revision="settings-r1",
        reason="managed_connection_auth",
        correlation_id="corr-managed-auth",
        broker_metadata={"surface": "settings_dialog", "flow": "managed_connection_auth"},
    )
    _assert_no_raw_values(request, label="ManagedConnectionAuthRequest")
    return request


def _identity_success() -> Any:
    managed_identity = _managed_identity_module()
    return managed_identity.ManagedIdentityPreflightResult(
        succeeded=True,
        local_public_key="local-public-key-1",
        local_identity_revision="identity-r1",
        message=None,
        diagnostics=None,
    )


def _identity_failure() -> Any:
    managed_identity = _managed_identity_module()
    return managed_identity.ManagedIdentityPreflightResult(
        succeeded=False,
        local_public_key=None,
        local_identity_revision=None,
        message=None,
        diagnostics=_safe_diagnostics("local_identity_failed"),
    )


def _discord_success() -> Any:
    discord_auth = _discord_auth_module()
    return discord_auth.DiscordAuthResult(
        succeeded=True,
        discord_user_id="discord-user-1",
        message=None,
        diagnostics=None,
    )


def _discord_oauth_material_success() -> Any:
    discord_auth = _discord_auth_module()
    return discord_auth.DiscordAuthResult(
        succeeded=True,
        discord_user_id=None,
        message=None,
        diagnostics=None,
        authorization_code="discord-code-1",
        oauth_state="discord-state-1",
        redirect_uri="http://127.0.0.1:62187/discord/callback",
        issue_nonce="issue-nonce-1",
        hardware_hash="hardware-hash-1",
        hardware_hash_salt_version=7,
    )


def _discord_failure() -> Any:
    discord_auth = _discord_auth_module()
    return discord_auth.DiscordAuthResult(
        succeeded=False,
        discord_user_id=None,
        message=None,
        diagnostics=_safe_diagnostics("discord_auth_failed"),
    )


def _broker_success(
    *,
    openrouter_user_id: str | None = None,
    delivery_ack: broker_client.ManagedKeyDeliveryAckMetadata | None = None,
) -> broker_client.BrokerIssueResult:
    return broker_client.BrokerIssueResult(
        succeeded=True,
        broker_connection_id="broker-conn-1",
        managed_secret_key=RAW_MANAGED_CREDENTIAL,
        remote_key_revision="remote-r1",
        message=None,
        diagnostics=None,
        openrouter_user_id=openrouter_user_id,
        delivery_ack=delivery_ack,
    )


def _delivery_ack_metadata() -> broker_client.ManagedKeyDeliveryAckMetadata:
    return broker_client.ManagedKeyDeliveryAckMetadata(
        source="discord",
        delivery_id="delivery-discord-1",
        managed_credential_ref="managed-ref-discord-1",
        expires_at="2026-07-07T00:15:00.000Z",
        delivery_ack_token="delivery-token-discord-1",
    )


def _broker_failure() -> broker_client.BrokerIssueResult:
    return broker_client.BrokerIssueResult(
        succeeded=False,
        broker_connection_id="broker-conn-failed",
        managed_secret_key=RAW_MANAGED_CREDENTIAL,
        remote_key_revision="remote-failed",
        message=None,
        diagnostics=_unsafe_diagnostics("broker_issue_failed"),
    )


def _commit_success() -> settings_repository.SettingsCommitResult:
    return settings_repository.SettingsCommitResult(
        succeeded=True,
        snapshot=settings_repository.SettingsSnapshot(
            values={"intent": {"translation": {"connection": "managed"}}},
            revision="settings-r2",
        ),
        message=None,
        diagnostics=None,
    )


def _commit_failure() -> settings_repository.SettingsCommitResult:
    return settings_repository.SettingsCommitResult(
        succeeded=False,
        snapshot=None,
        message=None,
        diagnostics=_unsafe_diagnostics("settings_commit_failed"),
    )


def _safe_diagnostics(code: str) -> messages.ErrorDiagnostics:
    return messages.ErrorDiagnostics(
        component="test_double",
        operation="test_operation",
        code=code,
        category=messages.DIAGNOSTIC_CATEGORY_TRANSACTION,
        visibility=messages.DIAGNOSTIC_VISIBILITY_BASIC,
        content_policy=messages.CONTENT_POLICY_METADATA_ONLY,
        status_code=None,
        retry_after_ms=None,
        fields={"phase": "test"},
    )


def _unsafe_diagnostics(code: str) -> messages.ErrorDiagnostics:
    return messages.ErrorDiagnostics(
        component="test_double",
        operation="test_operation",
        code=code,
        category=messages.DIAGNOSTIC_CATEGORY_TRANSACTION,
        visibility=messages.DIAGNOSTIC_VISIBILITY_BASIC,
        content_policy=messages.CONTENT_POLICY_METADATA_ONLY,
        status_code=None,
        retry_after_ms=None,
        fields={"payload": RAW_BROKER_PAYLOAD, "phase": "test"},
    )


def _assert_no_raw_values(value: object, *, label: str = "value") -> None:
    rendered = repr(value)
    for index, raw in enumerate(FORBIDDEN_RAW_VALUES, start=1):
        if raw in rendered:
            pytest.fail(f"{label} repr exposed forbidden raw sentinel #{index}")


def _only_item(items: list[Any], *, label: str) -> Any:
    if len(items) != 1:
        pytest.fail(f"{label} count mismatch: actual={len(items)}, expected=1")
    return items[0]


def _assert_no_items(items: list[Any], *, label: str) -> None:
    if items:
        pytest.fail(f"{label} count mismatch: actual={len(items)}, expected=0")


def _assert_pre_broker_unsafe_settings_rejection(
    *,
    result: messages.TransactionResult,
    events: list[tuple[str, str]],
    identity: RecordingLocalIdentity,
    discord: RecordingDiscordAuth,
    broker: RecordingBrokerClient,
    store: RecordingSecretStore,
    repository: RecordingSettingsRepository,
) -> None:
    _assert_no_raw_values(result, label="pre-broker unsafe settings result")
    assert result.status == messages.TRANSACTION_STATUS_PROVIDER_VERIFICATION_FAILED
    assert events == []
    _assert_no_items(identity.requests, label="identity preflight requests")
    _assert_no_items(discord.requests, label="discord auth requests")
    _assert_no_items(broker.requests, label="broker issue requests")
    _assert_no_items(store.set_calls, label="secret writes")
    _assert_no_items(repository.saved_requests, label="settings saves")
    assert result.diagnostics is not None
    assert result.diagnostics.component == "managed_connection_auth"
    assert result.diagnostics.operation == "validate_settings_values"
    assert result.diagnostics.code == "unsafe_settings_values"
    assert result.diagnostics.content_policy == messages.CONTENT_POLICY_METADATA_ONLY
    assert result.diagnostics.fields["phase"] == "validate_settings_values"
    assert result.diagnostics.fields["remote_active"] is False
    assert result.diagnostics.fields["settings_values_accepted"] is False


def _assert_remote_active_unsafe_settings_rejection(
    *,
    result: messages.TransactionResult,
    events: list[tuple[str, str]],
    store: RecordingSecretStore,
    repository: RecordingSettingsRepository,
) -> None:
    _assert_no_raw_values(result, label="remote-active unsafe settings result")
    assert result.status == messages.TRANSACTION_STATUS_REMOTE_ACTIVE_LOCAL_MISSING
    assert events == [
        ("identity_preflight", LOCAL_SECRET_KEY),
        ("discord_auth", "corr-managed-auth"),
        ("broker_issue", "discord-user-1"),
    ]
    _assert_no_items(store.set_calls, label="secret writes")
    _assert_no_items(repository.saved_requests, label="settings saves")
    assert LOCAL_SECRET_KEY not in store.values
    assert result.diagnostics is not None
    assert result.diagnostics.component == "managed_connection_auth"
    assert result.diagnostics.operation == "validate_settings_values"
    assert result.diagnostics.code == "remote_active_unsafe_settings_values"
    assert result.diagnostics.content_policy == messages.CONTENT_POLICY_METADATA_ONLY
    assert result.diagnostics.fields["phase"] == "validate_settings_values_after_broker"
    assert result.diagnostics.fields["remote_active"] is True
    assert result.diagnostics.fields["settings_values_accepted"] is False
    assert result.diagnostics.fields["secret_write_succeeded"] is False
    assert result.diagnostics.fields["settings_commit_succeeded"] is False
    assert result.diagnostics.fields["broker_connection_id"] == "broker-conn-1"
    assert result.diagnostics.fields["remote_key_revision"] == "remote-r1"


def test_request_and_port_dtos_are_frozen_slotted_and_repr_safe() -> None:
    auth = _service_module()
    discord_auth = _discord_auth_module()
    managed_identity = _managed_identity_module()
    settings_values = {"nested": {"aliases": ["managed"], "enabled": True}}
    metadata = {"surface": "settings_dialog"}

    request = auth.ManagedConnectionAuthRequest(
        local_secret_key=LOCAL_SECRET_KEY,
        settings_values=settings_values,
        expected_settings_revision="settings-r1",
        reason="managed_connection_auth",
        correlation_id="corr-managed-auth",
        broker_metadata=metadata,
    )
    identity_request = managed_identity.ManagedIdentityPreflightRequest(
        local_secret_key=LOCAL_SECRET_KEY,
        correlation_id="corr-managed-auth",
        metadata=metadata,
    )
    discord_request = discord_auth.DiscordAuthRequest(
        correlation_id="corr-managed-auth",
        metadata=metadata,
    )
    broker_result = _broker_success()

    for dto in (request, identity_request, discord_request, broker_result):
        assert is_dataclass(dto)
        assert not hasattr(dto, "__dict__")
        _assert_no_raw_values(dto, label=type(dto).__name__)

    with pytest.raises(FrozenInstanceError):
        request.reason = "other"  # type: ignore[misc]
    with pytest.raises(TypeError):
        request.settings_values["nested"] = {}  # type: ignore[index]
    nested = request.settings_values["nested"]
    assert isinstance(nested, Mapping)
    with pytest.raises(TypeError):
        nested["enabled"] = False  # type: ignore[index]

    settings_values["nested"]["aliases"].append("mutated")
    assert request.settings_values["nested"]["aliases"] == ("managed",)  # type: ignore[index]
    with pytest.raises(TypeError):
        identity_request.metadata["surface"] = "other"  # type: ignore[index]
    with pytest.raises(TypeError):
        discord_request.metadata["surface"] = "other"  # type: ignore[index]
    assert "managed_secret_key" in {field.name for field in fields(broker_result)}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "unsafe_path",
    (
        "credentials.api_key",
        "credentials.credential_value",
        "credentials.privateKey",
    ),
)
async def test_secret_bearing_settings_path_is_rejected_before_side_effects(
    unsafe_path: str,
) -> None:
    events: list[tuple[str, str]] = []
    identity = RecordingLocalIdentity(_identity_success(), events=events)
    discord = RecordingDiscordAuth(_discord_success(), events=events)
    broker = RecordingBrokerClient(_broker_success(), events=events)
    store = RecordingSecretStore(events=events)
    repository = RecordingSettingsRepository(_commit_success(), events=events)

    result = await _service(
        identity=identity,
        discord=discord,
        broker=broker,
        store=store,
        repository=repository,
    ).authorize(
        _request(
            {
                "connection": "managed",
                unsafe_path: "configured-by-caller-bug",
            }
        )
    )

    _assert_pre_broker_unsafe_settings_rejection(
        result=result,
        events=events,
        identity=identity,
        discord=discord,
        broker=broker,
        store=store,
        repository=repository,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "settings_values",
    (
        {
            "connection": "managed",
            f"credentials.{RAW_MANAGED_CREDENTIAL}.debug": "safe-value",
        },
        {
            "connection": "managed",
            "credentials.debug": {"nested": ["safe", (f"prefix:{RAW_MANAGED_CREDENTIAL}:suffix",)]},
        },
    ),
    ids=("raw-secret-in-key", "raw-secret-in-nested-value"),
)
async def test_broker_secret_in_settings_is_rejected_before_local_commit(
    settings_values: Mapping[str, object],
) -> None:
    events: list[tuple[str, str]] = []
    identity = RecordingLocalIdentity(_identity_success(), events=events)
    discord = RecordingDiscordAuth(_discord_success(), events=events)
    broker = RecordingBrokerClient(_broker_success(), events=events)
    store = RecordingSecretStore(events=events)
    repository = RecordingSettingsRepository(_commit_success(), events=events)

    result = await _service(
        identity=identity,
        discord=discord,
        broker=broker,
        store=store,
        repository=repository,
    ).authorize(_request(settings_values))

    _assert_remote_active_unsafe_settings_rejection(
        result=result,
        events=events,
        store=store,
        repository=repository,
    )


@pytest.mark.asyncio
async def test_success_preflights_identity_then_discord_auth_then_broker_issue_and_local_commit() -> (
    None
):
    events: list[tuple[str, str]] = []
    identity = RecordingLocalIdentity(_identity_success(), events=events)
    discord = RecordingDiscordAuth(_discord_success(), events=events)
    broker = RecordingBrokerClient(_broker_success(), events=events)
    store = RecordingSecretStore(events=events)
    repository = RecordingSettingsRepository(_commit_success(), events=events)

    result = await _service(
        identity=identity,
        discord=discord,
        broker=broker,
        store=store,
        repository=repository,
    ).authorize(_request())

    _assert_no_raw_values(result, label="managed auth success result")
    assert result.status == messages.TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_APPLIED
    assert events == [
        ("identity_preflight", LOCAL_SECRET_KEY),
        ("discord_auth", "corr-managed-auth"),
        ("broker_issue", "discord-user-1"),
        ("set_secret", LOCAL_SECRET_KEY),
        ("save_settings", "managed_connection_auth"),
    ]

    identity_request = _only_item(identity.requests, label="identity preflight requests")
    assert identity_request.local_secret_key == LOCAL_SECRET_KEY
    assert identity_request.correlation_id == "corr-managed-auth"
    assert identity_request.metadata["surface"] == "settings_dialog"

    discord_request = _only_item(discord.requests, label="discord auth requests")
    assert discord_request.correlation_id == "corr-managed-auth"
    assert discord_request.metadata["flow"] == "managed_connection_auth"

    broker_request = _only_item(broker.requests, label="broker issue requests")
    assert broker_request.discord_user_id == "discord-user-1"
    assert broker_request.local_public_key == "local-public-key-1"
    assert broker_request.local_identity_revision == "identity-r1"
    assert broker_request.metadata["flow"] == "managed_connection_auth"
    with pytest.raises(TypeError):
        broker_request.metadata["flow"] = "other"  # type: ignore[index]

    assert store.values[LOCAL_SECRET_KEY] == RAW_MANAGED_CREDENTIAL
    saved_request = _only_item(repository.saved_requests, label="settings saves")
    _assert_no_raw_values(saved_request, label="settings commit request")
    assert saved_request.expected_revision == "settings-r1"
    assert saved_request.reason == "managed_connection_auth"
    assert saved_request.values["intent"]["translation"]["connection"] == "managed"  # type: ignore[index]


@pytest.mark.asyncio
async def test_delivery_ack_pending_metadata_persists_before_ack_and_clears_after_success() -> None:
    events: list[tuple[str, str]] = []
    metadata = _delivery_ack_metadata()
    identity = RecordingLocalIdentity(_identity_success(), events=events)
    discord = RecordingDiscordAuth(_discord_success(), events=events)
    broker = RecordingBrokerClient(_broker_success(delivery_ack=metadata), events=events)
    store = RecordingSecretStore(events=events)
    repository = RecordingSettingsRepository(_commit_success(), events=events)

    result = await _service(
        identity=identity,
        discord=discord,
        broker=broker,
        store=store,
        repository=repository,
    ).authorize(_request())

    assert result.status == messages.TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_APPLIED
    assert [event[0] for event in events] == [
        "identity_preflight",
        "discord_auth",
        "broker_issue",
        "set_secret",
        "save_settings",
        "set_secret",
        "delivery_ack",
        "clear_secret",
        "save_settings",
    ]
    first_save, second_save = repository.saved_requests
    pending = first_save.values["state"]["managed_connection"]  # type: ignore[index]
    assert pending["pending_delivery_ack_source"] == "discord"
    assert pending["pending_delivery_ack_delivery_id"] == metadata.delivery_id
    assert pending["pending_delivery_ack_managed_credential_ref"] == metadata.managed_credential_ref
    assert "delivery_ack_token" not in repr(first_save.values)
    cleared = second_save.values["state"]["managed_connection"]  # type: ignore[index]
    assert cleared["pending_delivery_ack_source"] is None
    assert broker.ack_requests[0].delivery_id == metadata.delivery_id
    assert "delivery-token-discord-1" not in repr(broker.ack_requests[0])
    assert "delivery-token-discord-1" not in store.values.values()


@pytest.mark.asyncio
async def test_delivery_ack_token_store_failure_does_not_persist_pending_metadata() -> None:
    events: list[tuple[str, str]] = []
    identity = RecordingLocalIdentity(_identity_success(), events=events)
    discord = RecordingDiscordAuth(_discord_success(), events=events)
    broker = RecordingBrokerClient(
        _broker_success(delivery_ack=_delivery_ack_metadata()),
        events=events,
    )
    store = RecordingSecretStore(
        events=events,
        fail_set_keys={"openrouter_managed_delivery_ack_token"},
    )
    repository = RecordingSettingsRepository(_commit_success(), events=events)

    result = await _service(
        identity=identity,
        discord=discord,
        broker=broker,
        store=store,
        repository=repository,
    ).authorize(_request())

    assert result.status == messages.TRANSACTION_STATUS_REMOTE_ACTIVE_LOCAL_MISSING
    assert repository.saved_requests == []
    assert broker.ack_requests == []
    assert result.diagnostics is not None
    assert result.diagnostics.code == "delivery_ack_token_store_failed_before_local_key_write"
    assert LOCAL_SECRET_KEY not in store.values


@pytest.mark.asyncio
async def test_delivery_ack_pending_settings_commit_failure_happens_before_local_key_write() -> (
    None
):
    identity = RecordingLocalIdentity(_identity_success())
    discord = RecordingDiscordAuth(_discord_success())
    broker = RecordingBrokerClient(_broker_success(delivery_ack=_delivery_ack_metadata()))
    store = RecordingSecretStore()
    repository = RecordingSettingsRepository(_commit_failure())

    result = await _service(
        identity=identity,
        discord=discord,
        broker=broker,
        store=store,
        repository=repository,
    ).authorize(_request())

    assert result.status == messages.TRANSACTION_STATUS_REMOTE_ACTIVE_LOCAL_MISSING
    assert LOCAL_SECRET_KEY not in store.values
    assert broker.ack_requests == []
    assert len(repository.saved_requests) == 1


@pytest.mark.asyncio
async def test_delivery_ack_token_clear_failure_keeps_pending_metadata_for_retry() -> None:
    metadata = _delivery_ack_metadata()
    identity = RecordingLocalIdentity(_identity_success())
    discord = RecordingDiscordAuth(_discord_success())
    broker = RecordingBrokerClient(_broker_success(delivery_ack=metadata))
    store = RecordingSecretStore(clear_succeeds=False)
    repository = RecordingSettingsRepository(_commit_success())

    result = await _service(
        identity=identity,
        discord=discord,
        broker=broker,
        store=store,
        repository=repository,
    ).authorize(_request())

    assert result.status == messages.TRANSACTION_STATUS_REMOTE_DELIVERY_ACK_PENDING
    assert len(repository.saved_requests) == 1
    pending = repository.saved_requests[-1].values["state"]["managed_connection"]  # type: ignore[index]
    assert pending["pending_delivery_ack_delivery_id"] == metadata.delivery_id
    assert store.values["openrouter_managed_delivery_ack_token"] == metadata.delivery_ack_token


@pytest.mark.asyncio
async def test_delivery_ack_clear_metadata_commit_failure_keeps_token_for_retry() -> None:
    metadata = _delivery_ack_metadata()
    identity = RecordingLocalIdentity(_identity_success())
    discord = RecordingDiscordAuth(_discord_success())
    broker = RecordingBrokerClient(_broker_success(delivery_ack=metadata))
    store = RecordingSecretStore()
    repository = RecordingSettingsRepository(
        None,
        results=[_commit_success(), _commit_failure()],
    )

    result = await _service(
        identity=identity,
        discord=discord,
        broker=broker,
        store=store,
        repository=repository,
    ).authorize(_request())

    assert result.status == messages.TRANSACTION_STATUS_REMOTE_DELIVERY_ACK_PENDING
    assert len(repository.saved_requests) == 2
    pending = repository.saved_requests[0].values["state"]["managed_connection"]  # type: ignore[index]
    cleared = repository.saved_requests[1].values["state"]["managed_connection"]  # type: ignore[index]
    assert pending["pending_delivery_ack_delivery_id"] == metadata.delivery_id
    assert cleared["pending_delivery_ack_delivery_id"] is None
    assert store.values["openrouter_managed_delivery_ack_token"] == metadata.delivery_ack_token


@pytest.mark.asyncio
async def test_delivery_ack_token_clear_and_metadata_restore_failure_is_not_ignored() -> None:
    metadata = _delivery_ack_metadata()
    identity = RecordingLocalIdentity(_identity_success())
    discord = RecordingDiscordAuth(_discord_success())
    broker = RecordingBrokerClient(_broker_success(delivery_ack=metadata))
    store = RecordingSecretStore(clear_succeeds=False)
    repository = RecordingSettingsRepository(
        None,
        results=[_commit_success(), _commit_success(), _commit_failure()],
    )

    result = await _service(
        identity=identity,
        discord=discord,
        broker=broker,
        store=store,
        repository=repository,
    ).authorize(_request())

    assert result.status == messages.TRANSACTION_STATUS_REMOTE_DELIVERY_ACK_PENDING
    assert len(repository.saved_requests) == 1


@pytest.mark.asyncio
async def test_success_records_and_persists_discord_claim_after_settings_commit() -> None:
    events: list[tuple[str, str]] = []
    identity = RecordingLocalIdentity(_identity_success(), events=events)
    discord = RecordingDiscordAuth(_discord_success(), events=events)
    broker = RecordingBrokerClient(_broker_success(), events=events)
    store = RecordingSecretStore(events=events)
    repository = RecordingSettingsRepository(_commit_success(), events=events)
    managed_state = RecordingManagedState()

    result = await _service(
        identity=identity,
        discord=discord,
        broker=broker,
        store=store,
        repository=repository,
        claim_guard=ManagedAuthClaimGuard(managed_state, store),
    ).authorize(_request())

    assert result.status == messages.TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_APPLIED
    assert managed_state.local_managed_claim_sources == (MANAGED_AUTH_CLAIM_SOURCE_DISCORD,)
    assert managed_state.persist_calls == 1
    assert events[-1] == ("save_settings", "managed_connection_auth")


@pytest.mark.asyncio
async def test_success_stores_managed_openrouter_user_identifier_cache() -> None:
    events: list[tuple[str, str]] = []
    identity = RecordingLocalIdentity(_identity_success(), events=events)
    discord = RecordingDiscordAuth(_discord_success(), events=events)
    broker = RecordingBrokerClient(
        _broker_success(openrouter_user_id=" openrouter-user-1 "),
        events=events,
    )
    store = RecordingSecretStore(events=events)
    repository = RecordingSettingsRepository(_commit_success(), events=events)

    result = await _service(
        identity=identity,
        discord=discord,
        broker=broker,
        store=store,
        repository=repository,
    ).authorize(_request())

    assert result.status == messages.TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_APPLIED
    assert store.values[OPENROUTER_MANAGED_USER_ID_SECRET] == "openrouter-user-1"
    assert store.values[OPENROUTER_MANAGED_USER_INSTALLATION_ID_SECRET] == "identity-r1"
    assert (
        OPENROUTER_MANAGED_USER_ID_SECRET,
        "openrouter-user-1",
    ) in store.set_calls
    assert (
        OPENROUTER_MANAGED_USER_INSTALLATION_ID_SECRET,
        "identity-r1",
    ) in store.set_calls


@pytest.mark.asyncio
async def test_success_accepts_discord_oauth_issue_material_without_user_id() -> None:
    events: list[tuple[str, str]] = []
    identity = RecordingLocalIdentity(_identity_success(), events=events)
    discord = RecordingDiscordAuth(_discord_oauth_material_success(), events=events)
    broker = RecordingBrokerClient(_broker_success(), events=events)
    store = RecordingSecretStore(events=events)
    repository = RecordingSettingsRepository(_commit_success(), events=events)

    result = await _service(
        identity=identity,
        discord=discord,
        broker=broker,
        store=store,
        repository=repository,
    ).authorize(_request())

    assert result.status == messages.TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_APPLIED
    broker_request = _only_item(broker.requests, label="broker issue requests")
    assert broker_request.discord_user_id is None
    assert broker_request.authorization_code == "discord-code-1"
    assert broker_request.oauth_state == "discord-state-1"
    assert broker_request.redirect_uri == "http://127.0.0.1:62187/discord/callback"
    assert broker_request.issue_nonce == "issue-nonce-1"
    assert broker_request.hardware_hash == "hardware-hash-1"
    assert broker_request.hardware_hash_salt_version == 7


@pytest.mark.asyncio
async def test_discord_managed_auth_claim_preflight_blocks_existing_qq_claim() -> None:
    events: list[tuple[str, str]] = []
    identity = RecordingLocalIdentity(_identity_success(), events=events)
    discord = RecordingDiscordAuth(_discord_success(), events=events)
    broker = RecordingBrokerClient(_broker_success(), events=events)
    store = RecordingSecretStore(events=events)
    store.values[OPENROUTER_MANAGED_QQ_API_KEY_SECRET] = "existing-qq-key"
    repository = RecordingSettingsRepository(_commit_success(), events=events)
    managed_state = RecordingManagedState()

    result = await _service(
        identity=identity,
        discord=discord,
        broker=broker,
        store=store,
        repository=repository,
        claim_guard=ManagedAuthClaimGuard(managed_state, store),
    ).authorize(_request())

    assert result.status == messages.TRANSACTION_STATUS_PROVIDER_VERIFICATION_FAILED
    assert result.message is not None
    assert result.message.key == "discord_auth.error.already_claimed_qq"
    assert managed_state.local_managed_claim_sources == (MANAGED_AUTH_CLAIM_SOURCE_QQ,)
    assert managed_state.persist_calls == 1
    assert events == []
    _assert_no_items(identity.requests, label="identity preflight requests")
    _assert_no_items(discord.requests, label="discord auth requests")
    _assert_no_items(broker.requests, label="broker issue requests")
    _assert_no_items(store.set_calls, label="secret writes")
    _assert_no_items(repository.saved_requests, label="settings saves")


@pytest.mark.asyncio
async def test_local_identity_preflight_failure_short_circuits_remote_and_local_commit() -> None:
    events: list[tuple[str, str]] = []
    identity = RecordingLocalIdentity(_identity_failure(), events=events)
    discord = RecordingDiscordAuth(_discord_success(), events=events)
    broker = RecordingBrokerClient(_broker_success(), events=events)
    store = RecordingSecretStore(events=events)
    repository = RecordingSettingsRepository(_commit_success(), events=events)

    result = await _service(
        identity=identity,
        discord=discord,
        broker=broker,
        store=store,
        repository=repository,
    ).authorize(_request())

    _assert_no_raw_values(result, label="preflight failure result")
    assert result.status == messages.TRANSACTION_STATUS_PROVIDER_VERIFICATION_FAILED
    assert events == [("identity_preflight", LOCAL_SECRET_KEY)]
    _assert_no_items(discord.requests, label="discord auth requests")
    _assert_no_items(broker.requests, label="broker issue requests")
    _assert_no_items(store.set_calls, label="secret writes")
    _assert_no_items(repository.saved_requests, label="settings saves")
    assert result.diagnostics is not None
    assert result.diagnostics.component == "managed_connection_auth"
    assert result.diagnostics.operation == "preflight_managed_identity"
    assert result.diagnostics.code == "local_identity_preflight_failed"
    assert result.diagnostics.content_policy == messages.CONTENT_POLICY_METADATA_ONLY


@pytest.mark.asyncio
async def test_discord_auth_failure_short_circuits_broker_and_local_commit() -> None:
    events: list[tuple[str, str]] = []
    identity = RecordingLocalIdentity(_identity_success(), events=events)
    discord = RecordingDiscordAuth(_discord_failure(), events=events)
    broker = RecordingBrokerClient(_broker_success(), events=events)
    store = RecordingSecretStore(events=events)
    repository = RecordingSettingsRepository(_commit_success(), events=events)

    result = await _service(
        identity=identity,
        discord=discord,
        broker=broker,
        store=store,
        repository=repository,
    ).authorize(_request())

    _assert_no_raw_values(result, label="discord auth failure result")
    assert result.status == messages.TRANSACTION_STATUS_PROVIDER_VERIFICATION_FAILED
    assert events == [
        ("identity_preflight", LOCAL_SECRET_KEY),
        ("discord_auth", "corr-managed-auth"),
    ]
    _assert_no_items(broker.requests, label="broker issue requests")
    _assert_no_items(store.set_calls, label="secret writes")
    _assert_no_items(repository.saved_requests, label="settings saves")
    assert result.diagnostics is not None
    assert result.diagnostics.operation == "start_discord_auth"
    assert result.diagnostics.code == "discord_auth_failed"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("broker_result", "raise_on_issue", "expected_code"),
    (
        (_broker_failure(), False, "broker_issue_failed"),
        (None, True, "broker_issue_exception"),
    ),
)
async def test_broker_issue_failure_short_circuits_secret_and_settings_commit(
    broker_result: broker_client.BrokerIssueResult | None,
    raise_on_issue: bool,
    expected_code: str,
) -> None:
    events: list[tuple[str, str]] = []
    identity = RecordingLocalIdentity(_identity_success(), events=events)
    discord = RecordingDiscordAuth(_discord_success(), events=events)
    broker = RecordingBrokerClient(
        broker_result,
        events=events,
        raise_on_issue=raise_on_issue,
    )
    store = RecordingSecretStore(events=events)
    repository = RecordingSettingsRepository(_commit_success(), events=events)

    result = await _service(
        identity=identity,
        discord=discord,
        broker=broker,
        store=store,
        repository=repository,
    ).authorize(_request())

    _assert_no_raw_values(result, label="broker failure result")
    assert result.status == messages.TRANSACTION_STATUS_PROVIDER_VERIFICATION_FAILED
    assert events == [
        ("identity_preflight", LOCAL_SECRET_KEY),
        ("discord_auth", "corr-managed-auth"),
        ("broker_issue", "discord-user-1"),
    ]
    _assert_no_items(store.set_calls, label="secret writes")
    _assert_no_items(repository.saved_requests, label="settings saves")
    assert result.diagnostics is not None
    assert result.diagnostics.operation == "issue_managed_connection"
    assert result.diagnostics.code == expected_code
    assert result.diagnostics.content_policy == messages.CONTENT_POLICY_METADATA_ONLY


@pytest.mark.asyncio
@pytest.mark.parametrize("raise_on_set", (False, True))
async def test_broker_success_then_secret_write_failure_reports_remote_active_local_missing(
    raise_on_set: bool,
) -> None:
    events: list[tuple[str, str]] = []
    identity = RecordingLocalIdentity(_identity_success(), events=events)
    discord = RecordingDiscordAuth(_discord_success(), events=events)
    broker = RecordingBrokerClient(_broker_success(), events=events)
    store = RecordingSecretStore(
        events=events,
        write_succeeds=False,
        raise_on_set=raise_on_set,
    )
    repository = RecordingSettingsRepository(_commit_success(), events=events)

    result = await _service(
        identity=identity,
        discord=discord,
        broker=broker,
        store=store,
        repository=repository,
    ).authorize(_request())

    _assert_no_raw_values(result, label="remote active secret failure result")
    assert result.status == messages.TRANSACTION_STATUS_REMOTE_ACTIVE_LOCAL_MISSING
    assert events == [
        ("identity_preflight", LOCAL_SECRET_KEY),
        ("discord_auth", "corr-managed-auth"),
        ("broker_issue", "discord-user-1"),
        ("set_secret", LOCAL_SECRET_KEY),
    ]
    _assert_no_items(repository.saved_requests, label="settings saves")
    assert LOCAL_SECRET_KEY not in store.values
    assert result.diagnostics is not None
    assert result.diagnostics.operation == "set_managed_secret"
    assert result.diagnostics.code == "remote_active_local_secret_write_failed"
    assert result.diagnostics.content_policy == messages.CONTENT_POLICY_METADATA_ONLY
    assert result.diagnostics.fields["phase"] == "local_secret_write"
    assert result.diagnostics.fields["remote_active"] is True
    assert result.diagnostics.fields["secret_write_succeeded"] is False
    assert result.diagnostics.fields["broker_connection_id"] == "broker-conn-1"
    assert result.diagnostics.fields["remote_key_revision"] == "remote-r1"


@pytest.mark.asyncio
@pytest.mark.parametrize("raise_on_save", (False, True))
async def test_broker_success_then_settings_commit_failure_reports_remote_active_local_missing(
    raise_on_save: bool,
) -> None:
    events: list[tuple[str, str]] = []
    identity = RecordingLocalIdentity(_identity_success(), events=events)
    discord = RecordingDiscordAuth(_discord_success(), events=events)
    broker = RecordingBrokerClient(_broker_success(), events=events)
    store = RecordingSecretStore(events=events)
    repository = RecordingSettingsRepository(
        None if raise_on_save else _commit_failure(),
        events=events,
        raise_on_save=raise_on_save,
    )

    result = await _service(
        identity=identity,
        discord=discord,
        broker=broker,
        store=store,
        repository=repository,
    ).authorize(_request())

    _assert_no_raw_values(result, label="remote active settings failure result")
    assert result.status == messages.TRANSACTION_STATUS_REMOTE_ACTIVE_LOCAL_MISSING
    assert events == [
        ("identity_preflight", LOCAL_SECRET_KEY),
        ("discord_auth", "corr-managed-auth"),
        ("broker_issue", "discord-user-1"),
        ("set_secret", LOCAL_SECRET_KEY),
        ("save_settings", "managed_connection_auth"),
    ]
    assert store.values[LOCAL_SECRET_KEY] == RAW_MANAGED_CREDENTIAL
    assert result.diagnostics is not None
    assert result.diagnostics.operation == "commit_settings"
    assert result.diagnostics.code == "remote_active_local_settings_commit_failed"
    assert result.diagnostics.content_policy == messages.CONTENT_POLICY_METADATA_ONLY
    assert result.diagnostics.fields["phase"] == "local_settings_commit"
    assert result.diagnostics.fields["remote_active"] is True
    assert result.diagnostics.fields["secret_write_succeeded"] is True
    assert result.diagnostics.fields["settings_commit_succeeded"] is False
    assert result.diagnostics.fields["broker_connection_id"] == "broker-conn-1"
    assert result.diagnostics.fields["remote_key_revision"] == "remote-r1"
