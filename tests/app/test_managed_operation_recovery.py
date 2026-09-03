from __future__ import annotations

import asyncio
from dataclasses import dataclass
import inspect
from typing import Any

import pytest
from puripuly_heart.app.ports.broker_client import (
    BrokerIssueRequest,
    BrokerIssueResult,
    ManagedKeyDeliveryAckMetadata,
    ManagedKeyDeliveryAckRequest,
    ManagedKeyDeliveryAckResult,
    ManagedOperationStatusRequest,
    ManagedOperationStatusResult,
)
from puripuly_heart.app.ports.discord_auth import DiscordAuthResult
from puripuly_heart.app.ports.managed_identity import ManagedIdentityPreflightResult
from puripuly_heart.app.ports.secret_store import SecretReadResult, SecretWriteResult
from puripuly_heart.app.ports.settings_repository import (
    SettingsCommitRequest,
    SettingsCommitResult,
    SettingsSnapshot,
)
from puripuly_heart.app.services.managed import managed_operation as mop
from puripuly_heart.app.services.managed.managed_auth import (
    ManagedAuthExecutionResult,
    ManagedAuthOwner,
    ManagedAuthState,
)
from puripuly_heart.app.services.managed.managed_connection_auth import (
    ManagedConnectionAuthRequest,
    ManagedConnectionAuthService,
)
from puripuly_heart.app.services.managed.managed_key_delivery_ack import (
    ACK_SOURCE_DISCORD,
    ManagedKeyDeliveryAckService,
)
from puripuly_heart.core import messages

LOCAL_SECRET_KEY = "openrouter_managed_api_key"
RESUME_TOKEN_SECRET = "openrouter_managed_operation_resume_token"
RAW_CREDENTIAL = "sk-or-recovery-test-credential-must-not-leak"
RAW_TOKEN = "recovery-test-token-must-not-leak"
RAW_ACK_TOKEN = "delivery-ack-token-must-not-leak"


def _status(
    operation_status: str,
    client_action: str,
    *,
    credential: str | None = None,
    delivery_ack: ManagedKeyDeliveryAckMetadata | None = None,
    failed_reason: str | None = None,
    referral_id: str | None = None,
    referral_status: str | None = None,
    referral_settlement: str | None = None,
) -> ManagedOperationStatusResult:
    from puripuly_heart.app.ports.broker_client import ManagedOperationReferralSnapshot

    return ManagedOperationStatusResult(
        succeeded=True,
        operation_status=operation_status,
        client_action=client_action,
        failed_reason=failed_reason,
        managed_secret_key=credential,
        managed_credential_ref="managed-ref-1" if credential else None,
        expires_at="2026-12-01T00:00:00.000Z" if credential else None,
        referral_id=referral_id,
        delivery_ack=delivery_ack,
        referral=ManagedOperationReferralSnapshot(
            status=referral_status, settlement=referral_settlement
        )
        if referral_status is not None or referral_settlement is not None
        else None,
    )


def _transport_failure() -> ManagedOperationStatusResult:
    return ManagedOperationStatusResult(
        succeeded=False,
        operation_status="CREATE_UNKNOWN",
        client_action="wait",
    )


def _unknown_operation() -> ManagedOperationStatusResult:
    return ManagedOperationStatusResult(
        succeeded=False,
        operation_status="UNKNOWN_OPERATION",
        client_action="wait",
    )

def _delivery_ack_metadata() -> ManagedKeyDeliveryAckMetadata:
    return ManagedKeyDeliveryAckMetadata(
        source=ACK_SOURCE_DISCORD,
        delivery_id="delivery-op-1",
        managed_credential_ref="managed-ref-1",
        expires_at="2026-12-01T00:15:00.000Z",
        delivery_ack_token=RAW_ACK_TOKEN,
    )


class FakeBroker:
    def __init__(
        self,
        *,
        issue: Any = None,
        issue_script: list[Any] | None = None,
        statuses: list[Any] | None = None,
        resumes: list[Any] | None = None,
        ack: ManagedKeyDeliveryAckResult | None = None,
    ) -> None:
        self.issue = issue
        self.issue_script = list(issue_script) if issue_script is not None else None
        self.statuses = list(statuses or [])
        self.resumes = list(resumes or [])
        self.ack = ack
        self.issue_calls: list[BrokerIssueRequest] = []
        self.status_calls: list[ManagedOperationStatusRequest] = []
        self.resume_calls: list[Any] = []
        self.ack_calls: list[ManagedKeyDeliveryAckRequest] = []

    async def issue_managed_connection(self, request: BrokerIssueRequest) -> BrokerIssueResult:
        self.issue_calls.append(request)
        if self.issue_script is not None:
            outcome = self.issue_script.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome
        if isinstance(self.issue, Exception):
            raise self.issue
        return self.issue

    async def get_managed_operation_status(
        self, request: ManagedOperationStatusRequest
    ) -> ManagedOperationStatusResult:
        self.status_calls.append(request)
        outcome = self.statuses.pop(0)
        if inspect.isawaitable(outcome):
            outcome = await outcome
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    async def resume_managed_operation(self, request: Any) -> ManagedOperationStatusResult:
        self.resume_calls.append(request)
        outcome = self.resumes.pop(0)
        if inspect.isawaitable(outcome):
            outcome = await outcome
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    async def acknowledge_managed_key_delivery(
        self, request: ManagedKeyDeliveryAckRequest
    ) -> ManagedKeyDeliveryAckResult:
        self.ack_calls.append(request)
        assert self.ack is not None
        return self.ack


class FakeSecrets:
    def __init__(self, values: dict[str, str] | None = None) -> None:
        self.values = dict(values or {})

    async def get_secret(self, key: str) -> SecretReadResult:
        return SecretReadResult(
            key=key, value=self.values.get(key), revision=None, message=None, diagnostics=None
        )

    async def set_secret(self, key: str, value: str) -> SecretWriteResult:
        self.values[key] = value
        return SecretWriteResult(
            succeeded=True, key=key, revision="r1", message=None, diagnostics=None
        )

    async def clear_secret(self, key: str) -> SecretWriteResult:
        self.values.pop(key, None)
        return SecretWriteResult(
            succeeded=True, key=key, revision=None, message=None, diagnostics=None
        )

    async def snapshot_secret(self, key: str) -> Any:
        raise AssertionError("unused")

    async def restore_secret(self, snapshot: Any) -> Any:
        raise AssertionError("unused")


class FakeSettings:
    def __init__(self) -> None:
        self.saved: list[SettingsCommitRequest] = []

    async def save(self, request: SettingsCommitRequest) -> SettingsCommitResult:
        self.saved.append(request)
        return SettingsCommitResult(
            succeeded=True,
            snapshot=SettingsSnapshot(revision="r2", values=request.values),
            message=None,
            diagnostics=None,
        )


@dataclass
class FakeState:
    installation_id: str = "install-1"
    release_token: str | None = None
    release_token_expires_at: str | None = None
    verified_hardware_hash: str | None = None
    verified_hardware_hash_salt_version: int | None = None
    active_managed_credential_ref: str | None = None
    active_managed_expires_at: str | None = None
    founder_letter_seen_credential_ref: str | None = None
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
    persist_calls: int = 0

    def persist(self) -> None:
        self.persist_calls += 1

    def snapshot(self) -> Any:
        raise AssertionError("unused")

    def restore(self, snapshot: Any) -> None:
        raise AssertionError("unused")


class FakeIdentity:
    def __init__(self, revision: str = "install-1") -> None:
        self.revision = revision
        self.calls = 0

    async def preflight_managed_identity(self, request: Any) -> ManagedIdentityPreflightResult:
        self.calls += 1
        return ManagedIdentityPreflightResult(
            succeeded=True,
            local_public_key="local-public-key-1",
            local_identity_revision=self.revision,
            message=None,
            diagnostics=None,
        )


class FakeDiscord:
    def __init__(self) -> None:
        self.calls = 0

    async def start_discord_auth(self, request: Any) -> DiscordAuthResult:
        self.calls += 1
        return DiscordAuthResult(
            succeeded=True,
            discord_user_id="discord-user-1",
            message=None,
            diagnostics=None,
            authorization_code="code-1",
            oauth_state="state-1",
            redirect_uri="http://localhost/cb",
            issue_nonce="nonce-1",
            hardware_hash="hw-hash-1",
            hardware_hash_salt_version=7,
        )


def _base_values() -> dict[str, object]:
    return {
        "intent": {"translation": {"connection": "managed"}},
        "state": {"managed_connection": {"installation_id": "install-1"}},
    }


def _service(
    broker: FakeBroker,
    secrets: FakeSecrets,
    settings: FakeSettings,
    state: FakeState,
    identity: FakeIdentity | None = None,
    discord: FakeDiscord | None = None,
    ack_sink: list[Any] | None = None,
) -> ManagedConnectionAuthService:
    return ManagedConnectionAuthService(
        local_identity=identity or FakeIdentity(),
        discord_auth=discord or FakeDiscord(),
        broker_client=broker,  # type: ignore[arg-type]
        secret_store=secrets,  # type: ignore[arg-type]
        settings_repository=settings,  # type: ignore[arg-type]
        claim_guard=None,
        delivery_ack_service=ManagedKeyDeliveryAckService(
            broker_client=broker,  # type: ignore[arg-type]
            secret_store=secrets,  # type: ignore[arg-type]
            managed_state=state,  # type: ignore[arg-type]
        ),
        managed_state=state,  # type: ignore[arg-type]
    )


def _request(
    *,
    ack_sink: list[Any] | None = None,
    progress: list[str] | None = None,
) -> ManagedConnectionAuthRequest:
    return ManagedConnectionAuthRequest(
        local_secret_key=LOCAL_SECRET_KEY,
        settings_values=_base_values(),
        expected_settings_revision=None,
        reason="test",
        correlation_id=None,
        broker_metadata={},
        progress_sink=(progress.append if progress is not None else None),
        max_status_polls=8,
        status_poll_interval_ms=0,
        ack_result_sink=(ack_sink.append if ack_sink is not None else None),
    )


def _issue_success(
    credential: str = RAW_CREDENTIAL,
) -> BrokerIssueResult:
    return BrokerIssueResult(
        succeeded=True,
        broker_connection_id="managed-ref-1",
        managed_secret_key=credential,
        remote_key_revision="managed-ref-1",
        message=None,
        diagnostics=None,
        managed_credential_ref="managed-ref-1",
        expires_at="2026-12-01T00:00:00.000Z",
        delivery_ack=_delivery_ack_metadata(),
    )


def _seed_operation(state: FakeState, secrets: FakeSecrets, installation_id: str = "install-1",
                    operation_id: str | None = None, token: str | None = None) -> tuple[str, str]:
    operation_id = operation_id or mop.new_managed_operation_id()
    token = token or mop.new_managed_operation_resume_token()
    state.pending_managed_operation_id = operation_id
    state.pending_managed_operation_source = "discord"
    state.pending_managed_operation_installation_id = installation_id
    state.pending_managed_operation_state = "CREATING"
    secrets.values[RESUME_TOKEN_SECRET] = token
    return operation_id, token


def _assert_no_raw_boundary(value: object, *forbidden: str) -> None:
    rendered = repr(value)
    for sentinel in forbidden:
        assert sentinel not in rendered


def test_operation_identifiers_have_canonical_length_and_alphabet() -> None:
    for _ in range(100):
        operation_id = mop.new_managed_operation_id()
        token = mop.new_managed_operation_resume_token()
        assert operation_id.startswith("ph-mop-v1_")
        assert len(operation_id) == len("ph-mop-v1_") + 32
        assert len(token) == 43
        assert "=" not in operation_id and "=" not in token
        assert mop.is_valid_operation_id(operation_id)
        assert mop.is_valid_resume_token(token)
    assert not mop.is_valid_operation_id("ph-mop-v1_short")
    assert not mop.is_valid_operation_id(None)
    assert not mop.is_valid_resume_token("abc")
    assert not mop.is_valid_resume_token("a" * 42 + "=")


@pytest.mark.asyncio
async def test_operation_and_token_persisted_before_first_issue_post() -> None:
    broker = FakeBroker(issue=_issue_success(), ack=_ack_success())
    secrets, settings, state = FakeSecrets(), FakeSettings(), FakeState()
    ack_sink: list[Any] = []
    progress: list[str] = []
    seen: dict[str, Any] = {}
    original_issue = broker.issue_managed_connection

    async def _spy_issue(request: BrokerIssueRequest) -> BrokerIssueResult:
        seen["operation_id"] = state.pending_managed_operation_id
        seen["operation_state"] = state.pending_managed_operation_state
        seen["token"] = secrets.values.get(RESUME_TOKEN_SECRET)
        return await original_issue(request)

    broker.issue_managed_connection = _spy_issue  # type: ignore[method-assign]

    result = await _service(broker, secrets, settings, state).authorize(
        _request(ack_sink=ack_sink, progress=progress)
    )

    assert result.status == messages.TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_APPLIED
    assert len(broker.issue_calls) == 1
    issue_request = broker.issue_calls[0]
    assert issue_request.operation_id is not None
    assert mop.is_valid_operation_id(issue_request.operation_id)
    assert issue_request.resume_token is not None
    assert mop.is_valid_resume_token(issue_request.resume_token)
    assert seen["operation_id"] == issue_request.operation_id
    assert seen["token"] == issue_request.resume_token
    assert state.pending_managed_operation_id is None
    assert RESUME_TOKEN_SECRET not in secrets.values
    assert broker.status_calls == []
    assert progress[0] == "preparing"
    for saved in settings.saved:
        _assert_no_raw_boundary(saved.values, RAW_CREDENTIAL, issue_request.resume_token)
    managed_saves = [
        saved.values["state"]["managed_connection"] for saved in settings.saved
    ]
    assert managed_saves[0]["pending_managed_operation_id"] == issue_request.operation_id
    assert managed_saves[-1].get("pending_managed_operation_id") is None
    _assert_no_raw_boundary(result, RAW_CREDENTIAL)


def _ack_success(referral_id: str | None = None) -> ManagedKeyDeliveryAckResult:
    return ManagedKeyDeliveryAckResult(
        succeeded=True,
        status="acknowledged",
        referral_id=referral_id,
    )


@pytest.mark.asyncio
async def test_issue_timeout_routes_to_status_and_resume_without_fresh_issue() -> None:
    resume_ack = _delivery_ack_metadata()
    broker = FakeBroker(
        issue=RuntimeError("timeout"),
        statuses=[
            _status("CREATING", "wait"),
            _status("RETRY_READY", "retry_authorized"),
        ],
        resumes=[
            _status(
                "DELIVERY_PENDING",
                "acknowledge_delivery",
                credential=RAW_CREDENTIAL,
                delivery_ack=resume_ack,
            )
        ],
        ack=_ack_success(),
    )
    secrets, settings, state = FakeSecrets(), FakeSettings(), FakeState()
    progress: list[str] = []

    result = await _service(broker, secrets, settings, state).authorize(
        _request(progress=progress)
    )

    assert result.status == messages.TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_APPLIED
    assert len(broker.issue_calls) == 1
    assert len(broker.status_calls) == 2
    assert len(broker.resume_calls) == 1
    assert "recovering" in progress
    assert secrets.values[LOCAL_SECRET_KEY] == RAW_CREDENTIAL
    assert state.pending_managed_operation_id is None
    status_request = broker.status_calls[0]
    assert mop.is_valid_operation_id(status_request.operation_id)
    assert broker.issue_calls[0].operation_id == status_request.operation_id
    assert status_request.installation_id == "install-1"


@pytest.mark.asyncio
async def test_retry_authorized_without_credential_reissues_same_operation() -> None:
    broker = FakeBroker(
        issue_script=[RuntimeError("timeout"), _issue_success()],
        statuses=[_status("RETRY_READY", "retry_authorized")],
        resumes=[_status("RETRY_READY", "retry_authorized")],
        ack=_ack_success(),
    )
    secrets, settings, state = FakeSecrets(), FakeSettings(), FakeState()

    result = await _service(broker, secrets, settings, state).authorize(_request())

    assert result.status == messages.TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_APPLIED
    assert len(broker.issue_calls) == 2
    assert len(broker.status_calls) == 1
    assert len(broker.resume_calls) == 1
    first_operation = broker.issue_calls[0].operation_id
    assert first_operation is not None
    assert broker.issue_calls[1].operation_id == first_operation
    assert broker.issue_calls[1].resume_token == broker.issue_calls[0].resume_token
    assert broker.status_calls[0].operation_id == first_operation
    assert secrets.values[LOCAL_SECRET_KEY] == RAW_CREDENTIAL
    assert state.pending_managed_operation_id is None


@pytest.mark.asyncio
async def test_restart_resumes_same_operation_without_new_oauth_or_issue() -> None:
    broker = FakeBroker(
        issue=AssertionError("must not issue"),
        statuses=[_status("ACTIVE", "wait")],
        ack=_ack_success(),
    )
    secrets, settings, state = FakeSecrets(), FakeSettings(), FakeState()
    operation_id, token = _seed_operation(state, secrets)
    secrets.values[LOCAL_SECRET_KEY] = RAW_CREDENTIAL
    discord = FakeDiscord()
    identity = FakeIdentity()

    result = await _service(broker, secrets, settings, state, identity, discord).authorize(
        _request()
    )

    assert result.status == messages.TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_APPLIED
    assert discord.calls == 0
    assert broker.issue_calls == []
    assert len(broker.status_calls) == 1
    assert broker.status_calls[0].operation_id == operation_id
    assert broker.status_calls[0].resume_token == token
    assert state.pending_managed_operation_id is None


@pytest.mark.asyncio
async def test_restart_retry_ready_recovers_without_discord_or_issue() -> None:
    resume_ack = _delivery_ack_metadata()
    broker = FakeBroker(
        issue=AssertionError("issue must not be called during restart recovery"),
        statuses=[_status("RETRY_READY", "retry_authorized")],
        resumes=[
            _status(
                "DELIVERY_PENDING",
                "acknowledge_delivery",
                credential=RAW_CREDENTIAL,
                delivery_ack=resume_ack,
            )
        ],
        ack=_ack_success(),
    )
    secrets, settings, state = FakeSecrets(), FakeSettings(), FakeState()
    operation_id, token = _seed_operation(state, secrets)

    class _HardFailDiscord(FakeDiscord):
        async def start_discord_auth(self, request: Any) -> Any:
            raise AssertionError("Discord OAuth must not run during restart recovery")

    discord = _HardFailDiscord()
    identity = FakeIdentity()
    ack_sink: list[Any] = []
    progress: list[str] = []

    result = await _service(broker, secrets, settings, state, identity, discord).authorize(
        _request(ack_sink=ack_sink, progress=progress)
    )

    assert result.status == messages.TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_APPLIED
    assert discord.calls == 0
    assert broker.issue_calls == []
    assert len(broker.status_calls) == 1
    assert broker.status_calls[0].operation_id == operation_id
    assert broker.status_calls[0].resume_token == token
    assert len(broker.resume_calls) == 1
    assert broker.resume_calls[0].operation_id == operation_id
    assert "recovering" in progress
    assert secrets.values[LOCAL_SECRET_KEY] == RAW_CREDENTIAL
    assert len(broker.ack_calls) == 1
    assert broker.ack_calls[0].delivery_id == resume_ack.delivery_id
    assert state.pending_managed_operation_id is None
    assert RESUME_TOKEN_SECRET not in secrets.values
    assert len(ack_sink) == 1


@pytest.mark.asyncio
async def test_restart_transitional_resume_polls_without_discord_or_issue() -> None:
    broker = FakeBroker(
        issue=AssertionError("issue must not be called during restart recovery"),
        statuses=[_status("RECONCILING", "wait")] * 8,
        resumes=[],
        ack=_ack_success(),
    )
    secrets, settings, state = FakeSecrets(), FakeSettings(), FakeState()
    operation_id, token = _seed_operation(state, secrets)

    class _HardFailDiscord(FakeDiscord):
        async def start_discord_auth(self, request: Any) -> Any:
            raise AssertionError("Discord OAuth must not run during restart recovery")

    result = await _service(
        broker, secrets, settings, state, FakeIdentity(), _HardFailDiscord()
    ).authorize(_request())

    assert result.status == messages.TRANSACTION_STATUS_REMOTE_DELIVERY_ACK_PENDING
    assert result.message is not None
    assert result.message.key == "discord_auth.error.recovery_pending"
    assert broker.issue_calls == []
    assert len(broker.status_calls) == 8
    assert broker.resume_calls == []
    assert state.pending_managed_operation_id == operation_id
    assert secrets.values[RESUME_TOKEN_SECRET] == token


@pytest.mark.asyncio
async def test_failed_operation_clears_and_reports_action_required() -> None:
    broker = FakeBroker(
        issue=RuntimeError("timeout"),
        statuses=[_status("FAILED", "action_required", failed_reason="terminal_provider_failure")],
        ack=_ack_success(),
    )
    secrets, settings, state = FakeSecrets(), FakeSettings(), FakeState()
    _seed_operation(state, secrets)
    discord = FakeDiscord()

    result = await _service(broker, secrets, settings, state, FakeIdentity(), discord).authorize(
        _request()
    )

    assert result.status == messages.TRANSACTION_STATUS_REMOTE_ACTIVE_LOCAL_MISSING
    assert result.message is not None and result.message.key == "discord_auth.error.action_required"
    assert state.pending_managed_operation_id is None
    assert RESUME_TOKEN_SECRET not in secrets.values
    assert discord.calls == 0
    assert broker.issue_calls == []


@pytest.mark.asyncio
async def test_authorization_expired_clears_then_allows_fresh_oauth() -> None:
    expired = _status("FAILED", "action_required", failed_reason="authorization_expired")
    broker = FakeBroker(
        issue=_issue_success(),
        statuses=[expired],
        ack=_ack_success(),
    )
    secrets, settings, state = FakeSecrets(), FakeSettings(), FakeState()
    stale_id, _ = _seed_operation(state, secrets)

    first = await _service(broker, secrets, settings, state).authorize(_request())

    assert first.message is not None
    assert first.message.key == "discord_auth.error.authorization_expired"
    assert state.pending_managed_operation_id is None

    second = await _service(broker, secrets, settings, state).authorize(_request())

    assert second.status == messages.TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_APPLIED
    assert len(broker.issue_calls) == 1
    assert broker.issue_calls[0].operation_id != stale_id


@pytest.mark.asyncio
async def test_pending_ack_takes_precedence_over_pending_operation() -> None:
    broker = FakeBroker(
        issue=AssertionError("must not issue"),
        statuses=[],
        ack=_ack_success(referral_id="9XK2M4"),
    )
    secrets, settings, state = FakeSecrets(), FakeSettings(), FakeState()
    state.pending_delivery_ack_source = ACK_SOURCE_DISCORD
    state.pending_delivery_ack_delivery_id = "delivery-op-1"
    state.pending_delivery_ack_managed_credential_ref = "managed-ref-1"
    state.pending_delivery_ack_expires_at = "2026-12-01T00:15:00.000Z"
    secrets.values[LOCAL_SECRET_KEY] = RAW_CREDENTIAL
    secrets.values["openrouter_managed_delivery_ack_token"] = RAW_ACK_TOKEN
    _seed_operation(state, secrets)
    ack_sink: list[Any] = []

    result = await _service(broker, secrets, settings, state).authorize(
        _request(ack_sink=ack_sink)
    )

    assert result.status == messages.TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_APPLIED
    assert broker.issue_calls == []
    assert broker.status_calls == []
    assert state.referral_id == "9XK2M4"
    assert state.referral_source == "discord"
    assert len(ack_sink) == 1


@pytest.mark.asyncio
async def test_active_enables_translation_while_settlement_pending() -> None:
    broker = FakeBroker(
        issue=AssertionError("must not issue"),
        statuses=[
            _status(
                "ACTIVE",
                "wait",
                referral_status="reserved",
                referral_settlement="invitee_pending",
            )
        ],
        ack=_ack_success(),
    )
    secrets, settings, state = FakeSecrets(), FakeSettings(), FakeState()
    _seed_operation(state, secrets)
    secrets.values[LOCAL_SECRET_KEY] = RAW_CREDENTIAL

    result = await _service(broker, secrets, settings, state).authorize(_request())

    assert result.status == messages.TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_APPLIED
    assert broker.issue_calls == []


@pytest.mark.asyncio
async def test_unknown_operation_proceeds_to_fresh_issuance() -> None:
    broker = FakeBroker(
        issue=_issue_success(),
        statuses=[_unknown_operation()],
        ack=_ack_success(),
    )
    secrets, settings, state = FakeSecrets(), FakeSettings(), FakeState()
    stale_id, _ = _seed_operation(state, secrets)
    discord = FakeDiscord()

    result = await _service(broker, secrets, settings, state, FakeIdentity(), discord).authorize(
        _request()
    )

    assert result.status == messages.TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_APPLIED
    assert discord.calls == 1
    assert len(broker.issue_calls) == 1
    assert broker.issue_calls[0].operation_id != stale_id


@pytest.mark.asyncio
async def test_cancellation_preserves_pending_metadata() -> None:
    gate = asyncio.Event()
    broker = FakeBroker(
        issue=RuntimeError("timeout"),
        statuses=[gate.wait()],
        ack=_ack_success(),
    )
    secrets, settings, state = FakeSecrets(), FakeSettings(), FakeState()
    operation_id, token = _seed_operation(state, secrets)

    async def _run() -> None:
        await _service(broker, secrets, settings, state).authorize(
            ManagedConnectionAuthRequest(
                local_secret_key=LOCAL_SECRET_KEY,
                settings_values=_base_values(),
                expected_settings_revision=None,
                reason="test",
                correlation_id=None,
                broker_metadata={},
                max_status_polls=60,
                status_poll_interval_ms=10,
            )
        )

    task = asyncio.ensure_future(_run())
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert state.pending_managed_operation_id == operation_id
    assert secrets.values[RESUME_TOKEN_SECRET] == token


def test_ui_phase_mapping_covers_preparing_recovering_action_required() -> None:
    assert mop.managed_operation_ui_phase(None, None) == "preparing"
    assert mop.managed_operation_ui_phase("CREATING", "wait") == "recovering"
    assert mop.managed_operation_ui_phase("CLEAN", "wait") == "recovering"
    assert mop.managed_operation_ui_phase("RETRY_READY", "retry_authorized") == "recovering"
    assert mop.managed_operation_ui_phase("DELIVERY_PENDING", "acknowledge_delivery") == "ready"
    assert mop.managed_operation_ui_phase("ACTIVE", "wait") == "ready"
    assert mop.managed_operation_ui_phase("FAILED", "wait") == "action_required"
    assert mop.managed_operation_ui_phase("CREATING", "action_required") == "action_required"


def test_status_poll_backoff_grows_then_caps() -> None:
    delays = [mop.status_poll_delay_ms(index, 1000, 5000) for index in range(8)]
    assert delays == [1000, 2000, 4000, 5000, 5000, 5000, 5000, 5000]


def _owner(message_key: str) -> tuple[ManagedAuthOwner, list[tuple[str, object]]]:
    messages_sink: list[tuple[str, object]] = []

    async def _executor(
        _referral_id: str | None,
        _callback: object,
        _recovery: object = None,
    ) -> ManagedAuthExecutionResult:
        return ManagedAuthExecutionResult(succeeded=False, message_key=message_key)

    owner = ManagedAuthOwner(
        state_provider=lambda: ManagedAuthState(
            settings_available=True,
            managed_selected=True,
            managed_china=False,
            local_key_available=False,
            release_service_available=True,
            runtime_available=True,
            ingress_frozen=False,
        ),
        pending_sink=lambda _pending: None,
        qq_executor=None,  # type: ignore[arg-type]
        discord_executor=_executor,
        runtime_ensurer=None,  # type: ignore[arg-type]
        usage_view_sink=lambda _referral_id, _pass_status: None,
        usage_refresh_sink=lambda: None,
        message_sink=lambda key, values: messages_sink.append((key, dict(values))),
        result_sink=lambda _result: None,
        log_sink=lambda _message: None,
    )
    return owner, messages_sink


@pytest.mark.asyncio
async def test_owner_records_action_required_failure_kind() -> None:
    owner, _ = _owner("discord_auth.error.action_required")
    assert await owner.start_discord() is False
    assert owner.last_failure_kind == "action_required"
    assert owner.last_message_key == "discord_auth.error.action_required"


@pytest.mark.asyncio
async def test_owner_records_recovering_failure_kind() -> None:
    owner, _ = _owner("discord_auth.error.recovery_pending")
    assert await owner.start_discord() is False
    assert owner.last_failure_kind == "recovering"


@pytest.mark.asyncio
async def test_owner_records_generic_failure_kind() -> None:
    owner, _ = _owner("discord_auth.error.retry")
    assert await owner.start_discord() is False
    assert owner.last_failure_kind == "failed"


@pytest.mark.asyncio
async def test_secret_boundaries_hold_across_recovery_finalize() -> None:
    resume_ack = _delivery_ack_metadata()
    broker = FakeBroker(
        issue=RuntimeError("timeout"),
        statuses=[_status("RETRY_READY", "retry_authorized")],
        resumes=[
            _status(
                "DELIVERY_PENDING",
                "acknowledge_delivery",
                credential=RAW_CREDENTIAL,
                delivery_ack=resume_ack,
            )
        ],
        ack=_ack_success(),
    )
    secrets, settings, state = FakeSecrets(), FakeSettings(), FakeState()
    token_before = secrets.values.get(RESUME_TOKEN_SECRET)

    result = await _service(broker, secrets, settings, state).authorize(_request())

    assert result.status == messages.TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_APPLIED
    resume_token = broker.status_calls[0].resume_token
    assert resume_token is not None and resume_token != token_before
    for saved in settings.saved:
        _assert_no_raw_boundary(saved.values, RAW_CREDENTIAL, resume_token, RAW_ACK_TOKEN)
    _assert_no_raw_boundary(result, RAW_CREDENTIAL, RAW_ACK_TOKEN)
    if resume_token is not None:
        assert resume_token not in repr(result)


@pytest.mark.asyncio
async def test_owner_service_smoke_timeout_to_ready_reports_recovery() -> None:
    broker = FakeBroker(
        issue_script=[RuntimeError("timeout"), _issue_success()],
        statuses=[_status("RETRY_READY", "retry_authorized")],
        resumes=[_status("RETRY_READY", "retry_authorized")],
        ack=_ack_success(),
    )
    secrets, settings, state = FakeSecrets(), FakeSettings(), FakeState()
    identity, discord = FakeIdentity(), FakeDiscord()
    recovery_marks: list[str] = []
    usage_views: list[tuple[Any, Any]] = []
    ack_sink: list[Any] = []

    async def _executor(
        _referral_id: str | None,
        _callback: object,
        _recovery: object = None,
    ) -> ManagedAuthExecutionResult:
        service = _service(broker, secrets, settings, state, identity, discord)
        result = await service.authorize(_request(ack_sink=ack_sink, progress=recovery_marks))
        if result.status != messages.TRANSACTION_STATUS_SETTINGS_COMMIT_SUCCESS_RUNTIME_APPLIED:
            message = result.message
            return ManagedAuthExecutionResult(
                succeeded=False,
                message_key=(message.key if message is not None else "discord_auth.error.retry"),
                message_kwargs=dict(message.params) if message is not None else {},
            )
        return ManagedAuthExecutionResult(
            succeeded=True,
            referral_bonus_applied=False,
            referral_id=state.referral_id,
            pass_status=None,
            runtime_rebuild="always",
        )

    owner = ManagedAuthOwner(
        state_provider=lambda: ManagedAuthState(
            settings_available=True,
            managed_selected=True,
            managed_china=False,
            local_key_available=False,
            release_service_available=True,
            runtime_available=True,
            ingress_frozen=False,
        ),
        pending_sink=lambda _pending: None,
        qq_executor=None,  # type: ignore[arg-type]
        discord_executor=_executor,
        runtime_ensurer=lambda _mode: _ensure_true(),
        usage_view_sink=lambda referral_id, pass_status: usage_views.append(
            (referral_id, pass_status)
        ),
        usage_refresh_sink=lambda: None,
        message_sink=lambda _key, _values: None,
        result_sink=lambda _result: None,
        log_sink=lambda _message: None,
    )

    assert await owner.start_discord(on_recovery_started=recovery_marks.append) is True
    assert "recovering" in recovery_marks
    assert secrets.values[LOCAL_SECRET_KEY] == RAW_CREDENTIAL
    assert len(broker.issue_calls) == 2
    assert usage_views != []


async def _ensure_true() -> bool:
    return True
