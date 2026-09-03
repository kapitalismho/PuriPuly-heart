from __future__ import annotations

from dataclasses import dataclass

import pytest
from puripuly_heart.app.services.managed_auth_claims import (
    MANAGED_AUTH_CLAIM_SOURCE_DISCORD,
    OPENROUTER_MANAGED_API_KEY_SECRET,
    ManagedAuthClaimGuard,
)
from puripuly_heart.app.services.managed_key_delivery_ack import ManagedKeyDeliveryAckService

from puripuly_heart.app.ports import broker_client, managed_identity_state, secret_store
from puripuly_heart.app.services.qq_managed_auth import (
    OPENROUTER_MANAGED_QQ_API_KEY_SECRET,
    OPENROUTER_MANAGED_QQ_STATUS_AUTH_SECRET,
    QqManagedAuthRequest,
    QqManagedAuthService,
)
from puripuly_heart.core import messages

RAW_QQ_IDENTITY = "raw-qq-identity-123"
RAW_QQ_CREDENTIAL = "a" * 64
RAW_MANAGED_KEY = "sk-or-v1-raw-qq-managed-key"


class RecordingBrokerClient:
    def __init__(
        self,
        result: broker_client.QqManagedAssertionResult | Exception,
        *,
        ack_result: broker_client.ManagedKeyDeliveryAckResult | None = None,
    ) -> None:
        self.result = result
        self.ack_result = ack_result or broker_client.ManagedKeyDeliveryAckResult(
            succeeded=True,
            status="acknowledged",
        )
        self.requests: list[broker_client.QqManagedAssertionRequest] = []
        self.ack_requests: list[broker_client.ManagedKeyDeliveryAckRequest] = []

    async def issue_managed_connection(
        self,
        request: broker_client.BrokerIssueRequest,
    ) -> broker_client.BrokerIssueResult:
        raise AssertionError("standard managed issue path must not be used for QQ auth")

    async def assert_qq_managed_identity(
        self,
        request: broker_client.QqManagedAssertionRequest,
    ) -> broker_client.QqManagedAssertionResult:
        self.requests.append(request)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result

    async def acknowledge_managed_key_delivery(
        self,
        request: broker_client.ManagedKeyDeliveryAckRequest,
    ) -> broker_client.ManagedKeyDeliveryAckResult:
        self.ack_requests.append(request)
        return self.ack_result


class RecordingSecretStore:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}
        self.snapshot_fail = False
        self.set_fail = False
        self.set_raise = False
        self.restore_fail = False
        self.clear_fail = False
        self.fail_repeated_set_keys: set[str] = set()
        self.fail_set_keys: set[str] = set()
        self.set_counts: dict[str, int] = {}
        self.snapshots: list[str] = []
        self.set_calls: list[tuple[str, str]] = []
        self.restore_calls: list[secret_store.SecretSnapshot] = []

    async def get_secret(self, key: str) -> secret_store.SecretReadResult:
        value = self.values.get(key)
        return secret_store.SecretReadResult(
            key=key,
            value=value,
            revision="secret-r1" if value is not None else None,
            message=None,
            diagnostics=None,
        )

    async def set_secret(self, key: str, value: str) -> secret_store.SecretWriteResult:
        self.set_calls.append((key, value))
        self.set_counts[key] = self.set_counts.get(key, 0) + 1
        if self.set_raise:
            raise RuntimeError("raw provider failure must not leak")
        if (
            self.set_fail
            or key in self.fail_set_keys
            or (key in self.fail_repeated_set_keys and self.set_counts[key] > 1)
        ):
            return secret_store.SecretWriteResult(
                succeeded=False,
                key=key,
                revision=None,
                message=None,
                diagnostics=None,
            )
        self.values[key] = value
        return secret_store.SecretWriteResult(
            succeeded=True,
            key=key,
            revision="secret-r2",
            message=None,
            diagnostics=None,
        )

    async def clear_secret(self, key: str) -> secret_store.SecretWriteResult:
        if self.clear_fail:
            return secret_store.SecretWriteResult(
                succeeded=False,
                key=key,
                revision=None,
                message=None,
                diagnostics=None,
            )
        self.values.pop(key, None)
        return secret_store.SecretWriteResult(
            succeeded=True,
            key=key,
            revision="secret-r-clear",
            message=None,
            diagnostics=None,
        )

    async def snapshot_secret(self, key: str) -> secret_store.SecretSnapshot:
        self.snapshots.append(key)
        if self.snapshot_fail:
            raise RuntimeError("raw snapshot failure must not leak")
        existed = key in self.values
        return secret_store.SecretSnapshot(
            key=key,
            value=self.values.get(key),
            revision="secret-r1" if existed else None,
            existed=existed,
        )

    async def restore_secret(
        self,
        snapshot: secret_store.SecretSnapshot,
    ) -> secret_store.SecretWriteResult:
        self.restore_calls.append(snapshot)
        if self.restore_fail:
            return secret_store.SecretWriteResult(
                succeeded=False,
                key=snapshot.key,
                revision=None,
                message=None,
                diagnostics=None,
            )
        if snapshot.existed and snapshot.value is not None:
            self.values[snapshot.key] = snapshot.value
        else:
            self.values.pop(snapshot.key, None)
        return secret_store.SecretWriteResult(
            succeeded=True,
            key=snapshot.key,
            revision="secret-r-restored",
            message=None,
            diagnostics=None,
        )


@dataclass
class RecordingManagedState:
    installation_id: str = "install-123"
    release_token: str | None = None
    release_token_expires_at: str | None = None
    verified_hardware_hash: str | None = None
    verified_hardware_hash_salt_version: int | None = None
    active_managed_credential_ref: str | None = "previous-ref"
    active_managed_expires_at: str | None = "2026-07-01T00:00:00.000Z"
    founder_letter_seen_credential_ref: str | None = "previous-ref"
    referral_id: str | None = None
    referral_source: str | None = None
    local_managed_claim_sources: tuple[str, ...] = ()
    pending_delivery_ack_source: str | None = None
    pending_delivery_ack_delivery_id: str | None = None
    pending_delivery_ack_managed_credential_ref: str | None = None
    pending_delivery_ack_expires_at: str | None = None
    pending_managed_operation_id: str | None = None
    pending_managed_operation_source: str | None = None
    pending_managed_operation_installation_id: str | None = None
    pending_managed_operation_state: str | None = None
    fail_persist_calls: int = 0
    persist_calls: int = 0
    restore_calls: int = 0

    def persist(self) -> None:
        self.persist_calls += 1
        if self.fail_persist_calls > 0:
            self.fail_persist_calls -= 1
            raise RuntimeError("raw settings persist failure must not leak")

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
            referral_source=self.referral_source,
            local_managed_claim_sources=self.local_managed_claim_sources,
            pending_delivery_ack_source=self.pending_delivery_ack_source,
            pending_delivery_ack_delivery_id=self.pending_delivery_ack_delivery_id,
            pending_delivery_ack_managed_credential_ref=(
                self.pending_delivery_ack_managed_credential_ref
            ),
            pending_delivery_ack_expires_at=self.pending_delivery_ack_expires_at,
            pending_managed_operation_id=self.pending_managed_operation_id,
            pending_managed_operation_source=self.pending_managed_operation_source,
            pending_managed_operation_installation_id=(
                self.pending_managed_operation_installation_id
            ),
            pending_managed_operation_state=self.pending_managed_operation_state,
        )

    def restore(self, snapshot: managed_identity_state.ManagedIdentitySnapshot) -> None:
        self.restore_calls += 1
        self.installation_id = snapshot.installation_id
        self.release_token = snapshot.release_token
        self.release_token_expires_at = snapshot.release_token_expires_at
        self.verified_hardware_hash = snapshot.verified_hardware_hash
        self.verified_hardware_hash_salt_version = snapshot.verified_hardware_hash_salt_version
        self.active_managed_credential_ref = snapshot.active_managed_credential_ref
        self.active_managed_expires_at = snapshot.active_managed_expires_at
        self.founder_letter_seen_credential_ref = snapshot.founder_letter_seen_credential_ref
        self.referral_id = snapshot.referral_id
        self.referral_source = snapshot.referral_source
        self.local_managed_claim_sources = snapshot.local_managed_claim_sources
        self.pending_delivery_ack_source = snapshot.pending_delivery_ack_source
        self.pending_delivery_ack_delivery_id = snapshot.pending_delivery_ack_delivery_id
        self.pending_delivery_ack_managed_credential_ref = (
            snapshot.pending_delivery_ack_managed_credential_ref
        )
        self.pending_delivery_ack_expires_at = snapshot.pending_delivery_ack_expires_at
        self.pending_managed_operation_id = snapshot.pending_managed_operation_id
        self.pending_managed_operation_source = snapshot.pending_managed_operation_source
        self.pending_managed_operation_installation_id = (
            snapshot.pending_managed_operation_installation_id
        )
        self.pending_managed_operation_state = snapshot.pending_managed_operation_state


def _success_result(
    *,
    managed_credential_ref: str | None = "managed-ref-qq",
    expires_at: str | None = "2026-08-03T06:00:00.000Z",
    delivery_ack: broker_client.ManagedKeyDeliveryAckMetadata | None = None,
) -> broker_client.QqManagedAssertionResult:
    return broker_client.QqManagedAssertionResult(
        succeeded=True,
        managed_secret_key=RAW_MANAGED_KEY,
        entitlement=broker_client.QqManagedEntitlementSnapshot(
            qq_subject_ref="ph-qq-subject-v1_subject",
            managed_credential_ref=managed_credential_ref,
            expires_at=expires_at,
            openrouter_user_id="openrouter-user-qq",
        ),
        failure_subcode=None,
        retry_after_ms=None,
        message=None,
        diagnostics=None,
        delivery_ack=delivery_ack,
    )


def _delivery_ack_metadata() -> broker_client.ManagedKeyDeliveryAckMetadata:
    return broker_client.ManagedKeyDeliveryAckMetadata(
        source="qq",
        delivery_id="delivery-qq-1",
        managed_credential_ref="managed-ref-qq",
        expires_at="2026-07-07T00:15:00.000Z",
        delivery_ack_token="delivery-token-qq-1",
    )


def _failure_result(
    subcode: broker_client.QqManagedAssertionFailureSubcode,
    *,
    retry_after_ms: int | None = None,
) -> broker_client.QqManagedAssertionResult:
    return broker_client.QqManagedAssertionResult(
        succeeded=False,
        managed_secret_key=None,
        entitlement=None,
        failure_subcode=subcode,
        retry_after_ms=retry_after_ms,
        message=None,
        diagnostics=messages.ErrorDiagnostics(
            component="managed_openrouter_broker_client",
            operation="qq_assert",
            code="safe-code",
            category=messages.DIAGNOSTIC_CATEGORY_AUTH,
            visibility=messages.DIAGNOSTIC_VISIBILITY_BASIC,
            content_policy=messages.CONTENT_POLICY_METADATA_ONLY,
            status_code=None,
            retry_after_ms=retry_after_ms,
            fields={"qq_failure_subcode": subcode},
        ),
    )


def _service(
    broker_result: broker_client.QqManagedAssertionResult | Exception,
    *,
    store: RecordingSecretStore | None = None,
    state: RecordingManagedState | None = None,
    ack_result: broker_client.ManagedKeyDeliveryAckResult | None = None,
) -> tuple[
    QqManagedAuthService, RecordingBrokerClient, RecordingSecretStore, RecordingManagedState
]:
    broker = RecordingBrokerClient(broker_result, ack_result=ack_result)
    secret = store or RecordingSecretStore()
    managed_state = state or RecordingManagedState()
    return QqManagedAuthService(broker, secret, managed_state), broker, secret, managed_state


def _request() -> QqManagedAuthRequest:
    return QqManagedAuthRequest(
        qq_identity=RAW_QQ_IDENTITY,
        credential=RAW_QQ_CREDENTIAL,
        asserted_at="2026-07-03T06:00:00.000Z",
        correlation_id="corr-qq",
        metadata={"flow": "qq_managed"},
    )


@pytest.mark.asyncio
async def test_qq_managed_auth_success_stores_qq_key_and_persists_entitlement_state() -> None:
    service, broker, store, state = _service(_success_result())

    result = await service.authenticate(_request())

    assert result.status == messages.TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_APPLIED
    assert store.snapshots == [OPENROUTER_MANAGED_QQ_API_KEY_SECRET]
    status_auth = (
        '{"qq_identity":"raw-qq-identity-123",'
        '"credential":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",'
        '"managed_credential_ref":"managed-ref-qq"}'
    )
    assert store.set_calls == [
        (OPENROUTER_MANAGED_QQ_API_KEY_SECRET, RAW_MANAGED_KEY),
        (OPENROUTER_MANAGED_QQ_STATUS_AUTH_SECRET, status_auth),
    ]
    assert store.values == {
        OPENROUTER_MANAGED_QQ_API_KEY_SECRET: RAW_MANAGED_KEY,
        OPENROUTER_MANAGED_QQ_STATUS_AUTH_SECRET: status_auth,
    }
    assert state.active_managed_credential_ref == "managed-ref-qq"
    assert state.active_managed_expires_at == "2026-08-03T06:00:00.000Z"
    assert state.founder_letter_seen_credential_ref is None
    assert state.referral_source == "qq"
    assert state.persist_calls == 1
    assert broker.requests[0].qq_identity == RAW_QQ_IDENTITY
    assert broker.requests[0].credential == RAW_QQ_CREDENTIAL
    assert broker.requests[0].installation_id == "install-123"
    _assert_no_raw_values(result)


@pytest.mark.asyncio
async def test_qq_managed_auth_clears_previous_account_pass_when_new_account_omits_id() -> None:
    state = RecordingManagedState(
        active_managed_credential_ref="previous-ref",
        referral_id="8H3J4N",
        referral_source="qq",
    )
    service, _broker, _store, state = _service(_success_result(), state=state)

    result = await service.authenticate(_request())

    assert result.status == messages.TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_APPLIED
    assert state.active_managed_credential_ref == "managed-ref-qq"
    assert state.referral_id is None
    assert state.referral_source == "qq"


@pytest.mark.asyncio
async def test_qq_managed_auth_preserves_same_account_pass_when_response_omits_id() -> None:
    state = RecordingManagedState(
        active_managed_credential_ref="managed-ref-qq",
        referral_id="8H3J4N",
        referral_source="qq",
    )
    service, _broker, _store, state = _service(_success_result(), state=state)

    result = await service.authenticate(_request())

    assert result.status == messages.TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_APPLIED
    assert state.referral_id == "8H3J4N"
    assert state.referral_source == "qq"


@pytest.mark.asyncio
async def test_qq_managed_auth_clears_stale_status_auth_when_replacement_fails() -> None:
    store = RecordingSecretStore()
    store.values[OPENROUTER_MANAGED_QQ_STATUS_AUTH_SECRET] = (
        '{"qq_identity":"previous-account",'
        '"credential":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",'
        '"managed_credential_ref":"previous-managed-ref"}'
    )
    store.fail_set_keys.add(OPENROUTER_MANAGED_QQ_STATUS_AUTH_SECRET)
    service, _broker, store, state = _service(_success_result(), store=store)

    result = await service.authenticate(_request())

    assert result.status == messages.TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_APPLIED
    assert state.active_managed_credential_ref == "managed-ref-qq"
    assert OPENROUTER_MANAGED_QQ_STATUS_AUTH_SECRET not in store.values
    assert store.values[OPENROUTER_MANAGED_QQ_API_KEY_SECRET] == RAW_MANAGED_KEY
    _assert_no_raw_values(result)


@pytest.mark.asyncio
async def test_qq_managed_auth_ack_success_stores_pending_then_clears_token_and_metadata() -> None:
    metadata = _delivery_ack_metadata()
    service, broker, store, state = _service(_success_result(delivery_ack=metadata))

    result = await service.authenticate(_request())

    assert result.status == messages.TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_APPLIED
    assert broker.ack_requests[0].delivery_id == metadata.delivery_id
    assert "delivery-token-qq-1" not in repr(broker.ack_requests[0])
    assert (
        "openrouter_managed_qq_delivery_ack_token",
        metadata.delivery_ack_token,
    ) in store.set_calls
    assert "openrouter_managed_qq_delivery_ack_token" not in store.values
    assert state.pending_delivery_ack_source is None
    assert state.pending_delivery_ack_delivery_id is None
    assert state.persist_calls == 2
    _assert_no_raw_values(result)


@pytest.mark.asyncio
async def test_qq_managed_auth_ack_failure_leaves_pending_metadata_and_token() -> None:
    metadata = _delivery_ack_metadata()
    service, broker, store, state = _service(
        _success_result(delivery_ack=metadata),
        ack_result=broker_client.ManagedKeyDeliveryAckResult(
            succeeded=False,
            status="retryable",
        ),
    )

    result = await service.authenticate(_request())

    assert result.status == messages.TRANSACTION_STATUS_REMOTE_DELIVERY_ACK_PENDING
    assert broker.ack_requests[0].delivery_id == metadata.delivery_id
    assert state.pending_delivery_ack_source == "qq"
    assert state.pending_delivery_ack_delivery_id == metadata.delivery_id
    assert state.pending_delivery_ack_managed_credential_ref == metadata.managed_credential_ref
    assert store.values["openrouter_managed_qq_delivery_ack_token"] == metadata.delivery_ack_token
    assert state.persist_calls == 1
    _assert_no_raw_values(result)


@pytest.mark.asyncio
async def test_qq_managed_auth_recovers_pending_ack_before_new_assertion() -> None:
    metadata = _delivery_ack_metadata()
    state = RecordingManagedState(
        pending_delivery_ack_source="qq",
        pending_delivery_ack_delivery_id=metadata.delivery_id,
        pending_delivery_ack_managed_credential_ref=metadata.managed_credential_ref,
        pending_delivery_ack_expires_at=metadata.expires_at,
    )
    store = RecordingSecretStore()
    store.values[OPENROUTER_MANAGED_QQ_API_KEY_SECRET] = RAW_MANAGED_KEY
    store.values["openrouter_managed_qq_delivery_ack_token"] = metadata.delivery_ack_token
    service, broker, store, state = _service(
        _failure_result("lifetime_used"),
        store=store,
        state=state,
        ack_result=broker_client.ManagedKeyDeliveryAckResult(
            succeeded=True,
            status="already_acknowledged",
        ),
    )

    result = await service.authenticate(_request())

    assert result.status == messages.TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_APPLIED
    assert result.diagnostics is not None
    assert result.diagnostics.code == "qq_delivery_ack_recovered"
    assert broker.requests == []
    assert broker.ack_requests[0].delivery_id == metadata.delivery_id
    assert state.pending_delivery_ack_source is None
    assert state.pending_delivery_ack_delivery_id is None
    assert "openrouter_managed_qq_delivery_ack_token" not in store.values
    assert store.values[OPENROUTER_MANAGED_QQ_API_KEY_SECRET] == RAW_MANAGED_KEY
    _assert_no_raw_values(result)


@pytest.mark.asyncio
async def test_qq_managed_auth_does_not_ack_or_reassert_without_local_managed_key() -> None:
    metadata = _delivery_ack_metadata()
    state = RecordingManagedState(
        pending_delivery_ack_source="qq",
        pending_delivery_ack_delivery_id=metadata.delivery_id,
        pending_delivery_ack_managed_credential_ref=metadata.managed_credential_ref,
        pending_delivery_ack_expires_at=metadata.expires_at,
    )
    store = RecordingSecretStore()
    store.values["openrouter_managed_qq_delivery_ack_token"] = metadata.delivery_ack_token
    service, broker, _store, _state = _service(
        _success_result(),
        store=store,
        state=state,
    )

    result = await service.authenticate(_request())

    assert result.status == messages.TRANSACTION_STATUS_REMOTE_DELIVERY_ACK_PENDING
    assert result.diagnostics is not None
    assert result.diagnostics.fields["delivery_ack_status"] == "managed_secret_missing"
    assert broker.requests == []
    assert broker.ack_requests == []
    _assert_no_raw_values(result)


@pytest.mark.asyncio
async def test_qq_managed_auth_ack_token_store_failure_happens_before_local_key_write() -> None:
    metadata = _delivery_ack_metadata()
    store = RecordingSecretStore()
    store.fail_set_keys.add("openrouter_managed_qq_delivery_ack_token")
    service, broker, store, state = _service(
        _success_result(delivery_ack=metadata),
        store=store,
    )

    result = await service.authenticate(_request())

    assert result.status == messages.TRANSACTION_STATUS_REMOTE_ACTIVE_LOCAL_MISSING
    assert result.diagnostics is not None
    assert result.diagnostics.code == "qq_delivery_ack_token_store_failed_before_local_key_write"
    assert OPENROUTER_MANAGED_QQ_API_KEY_SECRET not in store.values
    assert broker.ack_requests == []
    assert state.active_managed_credential_ref == "previous-ref"
    _assert_no_raw_values(result)


@pytest.mark.asyncio
async def test_qq_managed_auth_pending_ack_persist_failure_happens_before_local_key_write() -> None:
    metadata = _delivery_ack_metadata()
    state = RecordingManagedState(
        active_managed_credential_ref="previous-ref",
        referral_id="8H3J4N",
        referral_source="qq",
        fail_persist_calls=1,
    )
    service, broker, store, state = _service(
        _success_result(delivery_ack=metadata),
        state=state,
    )

    result = await service.authenticate(_request())

    assert result.status == messages.TRANSACTION_STATUS_REMOTE_ACTIVE_LOCAL_MISSING
    assert result.diagnostics is not None
    assert (
        result.diagnostics.code == "qq_pending_delivery_ack_persist_failed_before_local_key_write"
    )
    assert OPENROUTER_MANAGED_QQ_API_KEY_SECRET not in store.values
    assert "openrouter_managed_qq_delivery_ack_token" not in store.values
    assert broker.ack_requests == []
    assert state.active_managed_credential_ref == "previous-ref"
    assert state.referral_id == "8H3J4N"
    assert state.referral_source == "qq"
    assert state.pending_delivery_ack_source is None
    assert state.restore_calls == 1


@pytest.mark.asyncio
async def test_shared_ack_service_checks_token_restore_result_after_clear_persist_failure() -> None:
    metadata = _delivery_ack_metadata()
    state = RecordingManagedState(
        pending_delivery_ack_source="qq",
        pending_delivery_ack_delivery_id=metadata.delivery_id,
        pending_delivery_ack_managed_credential_ref=metadata.managed_credential_ref,
        pending_delivery_ack_expires_at=metadata.expires_at,
        fail_persist_calls=1,
    )
    store = RecordingSecretStore()
    store.values["openrouter_managed_qq_delivery_ack_token"] = metadata.delivery_ack_token
    store.set_counts["openrouter_managed_qq_delivery_ack_token"] = 1
    store.fail_repeated_set_keys.add("openrouter_managed_qq_delivery_ack_token")
    broker = RecordingBrokerClient(_success_result())
    service = ManagedKeyDeliveryAckService(broker, store, state)

    result = await service.retry_pending()

    assert result.succeeded is False
    assert result.status == "token_restore_failed"
    assert state.pending_delivery_ack_source == "qq"
    assert "openrouter_managed_qq_delivery_ack_token" not in store.values
    assert result.diagnostics is not None
    assert result.diagnostics.code == "delivery_ack_token_restore_failed"


@pytest.mark.asyncio
async def test_qq_managed_auth_claim_preflight_blocks_existing_discord_claim() -> None:
    _service_instance, broker, store, state = _service(_success_result())
    store.values[OPENROUTER_MANAGED_API_KEY_SECRET] = "existing-discord-key"
    service = QqManagedAuthService(
        broker,
        store,
        state,
        ManagedAuthClaimGuard(state, store),
    )

    result = await service.authenticate(_request())

    assert result.status == messages.TRANSACTION_STATUS_PROVIDER_VERIFICATION_FAILED
    assert result.message is not None
    assert result.message.key == "qq_managed_auth.already_claimed_discord"
    assert state.local_managed_claim_sources == (MANAGED_AUTH_CLAIM_SOURCE_DISCORD,)
    assert state.persist_calls == 1
    assert broker.requests == []
    assert store.set_calls == []


@pytest.mark.asyncio
async def test_qq_managed_auth_success_falls_back_to_subject_ref_snapshot_when_credential_ref_missing() -> (
    None
):
    service, _broker, _store, state = _service(_success_result(managed_credential_ref=None))

    result = await service.authenticate(_request())

    assert result.status == messages.TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_APPLIED
    assert state.active_managed_credential_ref == "ph-qq-subject-v1_subject"
    assert state.active_managed_expires_at == "2026-08-03T06:00:00.000Z"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("subcode", "expected_key", "expected_category", "retry_after_ms"),
    [
        ("invalid_credential", "qq_managed_auth.invalid_credential", "auth", None),
        ("mismatch", "qq_managed_auth.mismatch", "auth", None),
        ("lifetime_used", "qq_managed_auth.lifetime_used", "quota", None),
        ("rate_limited", "qq_managed_auth.rate_limited", "rate_limit", 3000),
        ("key_unavailable", "qq_managed_auth.key_unavailable", "service_unavailable", None),
        ("broker_unavailable", "qq_managed_auth.broker_unavailable", "service_unavailable", None),
    ],
)
async def test_qq_managed_auth_maps_safe_broker_failure_subcodes_without_mutation(
    subcode: broker_client.QqManagedAssertionFailureSubcode,
    expected_key: str,
    expected_category: str,
    retry_after_ms: int | None,
) -> None:
    service, _broker, store, state = _service(
        _failure_result(subcode, retry_after_ms=retry_after_ms)
    )

    result = await service.authenticate(_request())

    assert result.status == messages.TRANSACTION_STATUS_PROVIDER_VERIFICATION_FAILED
    assert result.message is not None
    assert result.message.key == expected_key
    assert result.diagnostics is not None
    assert result.diagnostics.category == expected_category
    assert result.diagnostics.retry_after_ms == retry_after_ms
    assert result.diagnostics.fields["qq_failure_subcode"] == subcode
    assert store.set_calls == []
    assert state.persist_calls == 0
    _assert_no_raw_values(result)


@pytest.mark.asyncio
async def test_qq_managed_auth_broker_exception_returns_safe_failure() -> None:
    service, _broker, store, state = _service(RuntimeError("raw broker payload failure"))

    result = await service.authenticate(_request())

    assert result.status == messages.TRANSACTION_STATUS_PROVIDER_VERIFICATION_FAILED
    assert result.message is not None
    assert result.message.key == "qq_managed_auth.broker_unavailable"
    assert result.diagnostics is not None
    assert result.diagnostics.fields["qq_failure_subcode"] == "broker_unavailable"
    assert store.set_calls == []
    assert state.persist_calls == 0
    _assert_no_raw_values(result)


@pytest.mark.asyncio
async def test_qq_managed_auth_missing_managed_key_does_not_mutate_local_state() -> None:
    result_without_key = broker_client.QqManagedAssertionResult(
        succeeded=True,
        managed_secret_key=None,
        entitlement=broker_client.QqManagedEntitlementSnapshot(
            qq_subject_ref="ph-qq-subject-v1_subject",
            managed_credential_ref="managed-ref-qq",
            expires_at="2026-08-03T06:00:00.000Z",
        ),
        failure_subcode=None,
        retry_after_ms=None,
        message=None,
        diagnostics=None,
    )
    service, _broker, store, state = _service(result_without_key)

    result = await service.authenticate(_request())

    assert result.status == messages.TRANSACTION_STATUS_REMOTE_ACTIVE_LOCAL_MISSING
    assert result.diagnostics is not None
    assert result.diagnostics.fields["qq_failure_subcode"] == "key_unavailable"
    assert store.set_calls == []
    assert state.persist_calls == 0
    _assert_no_raw_values(result)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["snapshot_exception", "set_exception", "set_failed"])
async def test_qq_managed_auth_secret_failures_do_not_persist_settings_or_raw_values(
    mode: str,
) -> None:
    store = RecordingSecretStore()
    if mode == "snapshot_exception":
        store.snapshot_fail = True
    elif mode == "set_exception":
        store.set_raise = True
    else:
        store.set_fail = True
    service, _broker, _store, state = _service(_success_result(), store=store)

    result = await service.authenticate(_request())

    assert result.status == messages.TRANSACTION_STATUS_SECRET_WRITE_FAILED
    assert state.persist_calls == 0
    assert state.active_managed_credential_ref == "previous-ref"
    assert store.values == {}
    _assert_no_raw_values(result)


@pytest.mark.asyncio
async def test_qq_managed_auth_settings_persist_failure_rolls_back_secret_and_state() -> None:
    store = RecordingSecretStore()
    store.values[OPENROUTER_MANAGED_QQ_API_KEY_SECRET] = "previous-qq-key"
    state = RecordingManagedState(fail_persist_calls=1)
    service, _broker, _store, _state = _service(_success_result(), store=store, state=state)

    result = await service.authenticate(_request())

    assert result.status == messages.TRANSACTION_STATUS_SETTINGS_COMMIT_FAILED_SECRET_RESTORED
    assert store.values[OPENROUTER_MANAGED_QQ_API_KEY_SECRET] == "previous-qq-key"
    assert len(store.restore_calls) == 1
    assert state.active_managed_credential_ref == "previous-ref"
    assert state.active_managed_expires_at == "2026-07-01T00:00:00.000Z"
    assert state.founder_letter_seen_credential_ref == "previous-ref"
    assert state.persist_calls == 2
    assert result.diagnostics is not None
    assert result.diagnostics.fields["state_rollback_succeeded"] is True
    assert result.diagnostics.fields["secret_rollback_succeeded"] is True
    assert result.diagnostics.fields["compensation_succeeded"] is True
    _assert_no_raw_values(result)


@pytest.mark.asyncio
async def test_qq_managed_auth_reports_failed_compensation_when_rollback_fails() -> None:
    store = RecordingSecretStore()
    store.values[OPENROUTER_MANAGED_QQ_API_KEY_SECRET] = "previous-qq-key"
    store.restore_fail = True
    state = RecordingManagedState(fail_persist_calls=2)
    service, _broker, _store, _state = _service(_success_result(), store=store, state=state)

    result = await service.authenticate(_request())

    assert result.status == messages.TRANSACTION_STATUS_SETTINGS_COMMIT_FAILED_SECRET_RESTORE_FAILED
    assert store.values[OPENROUTER_MANAGED_QQ_API_KEY_SECRET] == RAW_MANAGED_KEY
    assert state.active_managed_credential_ref == "previous-ref"
    assert result.diagnostics is not None
    assert result.diagnostics.fields["state_rollback_succeeded"] is False
    assert result.diagnostics.fields["secret_rollback_succeeded"] is False
    assert result.diagnostics.fields["compensation_succeeded"] is False
    _assert_no_raw_values(result)


def test_qq_managed_auth_request_repr_hides_raw_credential_values() -> None:
    request = _request()

    assert RAW_QQ_IDENTITY not in repr(request)
    assert RAW_QQ_CREDENTIAL not in repr(request)


def _assert_no_raw_values(value: object) -> None:
    rendered = _render(value)
    forbidden = (
        RAW_QQ_IDENTITY,
        RAW_QQ_CREDENTIAL,
        RAW_MANAGED_KEY,
        "raw broker payload failure",
        "raw settings persist failure",
        "raw provider failure",
        "raw snapshot failure",
    )
    for raw_value in forbidden:
        assert raw_value not in rendered


def _render(value: object) -> str:
    if isinstance(value, dict):
        return repr({key: _render(nested) for key, nested in value.items()})
    if isinstance(value, (list, tuple, set, frozenset)):
        return repr([_render(nested) for nested in value])
    if hasattr(value, "__dataclass_fields__"):
        return repr(
            {
                field_name: _render(getattr(value, field_name))
                for field_name in value.__dataclass_fields__
            }
        )
    return repr(value)
