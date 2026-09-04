from __future__ import annotations

from typing import Any

import pytest
from puripuly_heart.core.managed_openrouter_release import (
    ManagedOpenRouterReleaseBehavior,
    ManagedOpenRouterReleaseError,
)

from puripuly_heart.app.ports.broker_client import (
    ManagedKeyDeliveryAckResult,
    ManagedOperationStatusResult,
)
from puripuly_heart.app.services.managed.managed_operation import (
    MANAGED_OPERATION_RESUME_TOKEN_SECRET,
    new_managed_operation_id,
    new_managed_operation_resume_token,
)
from puripuly_heart.core.managed_identity import ensure_managed_identity_bundle
from puripuly_heart.core.storage.secrets import InMemorySecretStore
from tests.core.test_managed_openrouter_release import (
    FakeDiscordOAuthHarness,
    _make_discord_service,
    _make_discord_start_success,
)


class OperationFakeClient:
    def __init__(
        self,
        *,
        statuses: list[Any] | None = None,
        resumes: list[Any] | None = None,
        ack: Any | None = None,
        issue_error: Exception | None = None,
        issue_result: Any | None = None,
        start_result: Any | None = None,
    ) -> None:
        self.statuses = list(statuses or [])
        self.resumes = list(resumes or [])
        self.ack = ack
        self.issue_error = issue_error
        self.issue_result = issue_result
        self.start_result = start_result
        self.calls: list[tuple[str, Any]] = []

    async def get_managed_operation_status(self, request: Any) -> Any:
        self.calls.append(("operation_status", request))
        outcome = self.statuses.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    async def resume_managed_operation(self, request: Any) -> Any:
        self.calls.append(("operation_resume", request))
        outcome = self.resumes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    async def issue_discord_managed_key(self, request: dict[str, object]) -> Any:
        self.calls.append(("discord_issue", dict(request)))
        if self.issue_error is not None:
            raise self.issue_error
        if self.issue_result is not None:
            return self.issue_result
        raise AssertionError("unexpected discord issue call")

    async def start_discord_oauth(self, **kwargs: Any) -> Any:
        self.calls.append(("discord_start", dict(kwargs)))
        if self.start_result is not None:
            return self.start_result
        raise AssertionError("unexpected discord start call")

    async def acknowledge_managed_key_delivery(self, request: Any) -> Any:
        self.calls.append(("delivery_ack", request))
        assert self.ack is not None
        return self.ack


def _status(
    operation_status: str,
    client_action: str,
    *,
    credential: str | None = None,
    failed_reason: str | None = None,
) -> ManagedOperationStatusResult:
    delivery_ack = None
    if credential is not None:
        from puripuly_heart.app.ports.broker_client import ManagedKeyDeliveryAckMetadata

        delivery_ack = ManagedKeyDeliveryAckMetadata(
            source="discord",
            delivery_id="delivery-legacy-1",
            managed_credential_ref="managed-ref-legacy-1",
            expires_at="2026-12-01T00:15:00.000Z",
            delivery_ack_token="legacy-ack-token-must-not-leak",
        )
    return ManagedOperationStatusResult(
        succeeded=True,
        operation_status=operation_status,
        client_action=client_action,
        failed_reason=failed_reason,
        managed_secret_key=credential,
        managed_credential_ref="managed-ref-legacy-1" if credential else None,
        expires_at="2026-12-01T00:00:00.000Z" if credential else None,
        delivery_ack=delivery_ack,
    )


def _seed_operation(service: Any, secrets: InMemorySecretStore) -> tuple[str, str]:
    bundle = ensure_managed_identity_bundle(service.managed_state, secrets)
    operation_id = new_managed_operation_id()
    resume_token = new_managed_operation_resume_token()
    service.managed_state.pending_managed_operation_id = operation_id
    service.managed_state.pending_managed_operation_source = "discord"
    service.managed_state.pending_managed_operation_installation_id = bundle.installation_id
    service.managed_state.pending_managed_operation_state = "CREATING"
    service.managed_state.persist()
    secrets.set(MANAGED_OPERATION_RESUME_TOKEN_SECRET, resume_token)
    return operation_id, resume_token


@pytest.mark.asyncio
async def test_prepare_recovers_expired_operation_without_new_oauth() -> None:
    client: Any = OperationFakeClient(
        statuses=[_status("FAILED", "action_required", failed_reason="authorization_expired")],
    )
    service, settings, secrets, _client, _harness = _make_discord_service(client=client)
    operation_id, _token = _seed_operation(service, secrets)

    result = await service.prepare_for_translation()

    assert result.behavior == ManagedOpenRouterReleaseBehavior.STOP
    assert result.message_key == "discord_auth.error.authorization_expired"
    assert settings.managed_identity.pending_managed_operation_id is None
    assert [name for name, _payload in client.calls] == ["operation_status"]
    assert client.calls[0][1].operation_id == operation_id


@pytest.mark.asyncio
async def test_prepare_waits_on_recoverable_operation_and_preserves_metadata() -> None:
    client: Any = OperationFakeClient(
        statuses=[_status("RECONCILING", "wait")],
    )
    service, settings, secrets, _client, _harness = _make_discord_service(client=client)
    operation_id, token = _seed_operation(service, secrets)

    result = await service.prepare_for_translation()

    assert result.behavior == ManagedOpenRouterReleaseBehavior.RETRY
    assert result.message_key == "discord_auth.error.recovery_pending"
    assert settings.managed_identity.pending_managed_operation_id == operation_id
    assert secrets.get(MANAGED_OPERATION_RESUME_TOKEN_SECRET) == token


@pytest.mark.asyncio
async def test_prepare_resumes_retry_authorized_operation_to_ready() -> None:
    ack = ManagedKeyDeliveryAckResult(succeeded=True, status="acknowledged")
    client: Any = OperationFakeClient(
        statuses=[_status("RETRY_READY", "retry_authorized")],
        resumes=[_status("DELIVERY_PENDING", "acknowledge_delivery", credential="managed-key")],
        ack=ack,
    )
    service, settings, secrets, _client, _harness = _make_discord_service(client=client)
    operation_id, _token = _seed_operation(service, secrets)

    result = await service.prepare_for_translation()

    assert result.behavior == ManagedOpenRouterReleaseBehavior.READY
    assert result.local_key_available is True
    assert settings.managed_identity.pending_managed_operation_id is None
    assert [name for name, _payload in client.calls] == [
        "operation_status",
        "operation_resume",
        "delivery_ack",
    ]
    status_requests = [payload for name, payload in client.calls if name == "operation_status"]
    assert len(status_requests) == 1
    assert status_requests[0].operation_id == operation_id
    assert "discord_issue" not in [name for name, _payload in client.calls]


@pytest.mark.asyncio
async def test_prepare_clears_unknown_operation_and_starts_fresh_oauth() -> None:
    from puripuly_heart.app.ports.broker_client import ManagedOperationStatusResult as StatusResult

    terminal_error = ManagedOpenRouterReleaseError(
        code="discord_lifetime_used",
        error_class="terminal",
        message="lifetime used",
        operation="discord_issue",
        subcode="discord_lifetime_used",
    )
    harness = FakeDiscordOAuthHarness()
    client: Any = OperationFakeClient(
        statuses=[
            StatusResult(
                succeeded=False,
                operation_status="UNKNOWN_OPERATION",
                client_action="wait",
            )
        ],
        start_result=_make_discord_start_success(redirect_uri=harness.redirect_uri),
        issue_error=terminal_error,
    )
    service, settings, secrets, _client, _harness = _make_discord_service(
        client=client, harness=harness
    )
    stale_id, _token = _seed_operation(service, secrets)

    result = await service.prepare_for_translation()

    assert result.behavior == ManagedOpenRouterReleaseBehavior.STOP
    names = [name for name, _payload in client.calls]
    assert names[0] == "operation_status"
    assert "discord_start" in names
    assert "discord_issue" in names
    issue_calls = [payload for name, payload in client.calls if name == "discord_issue"]
    assert len(issue_calls) == 1
    assert issue_calls[0]["operation_id"] != stale_id
    assert settings.managed_identity.pending_managed_operation_id is None


@pytest.mark.asyncio
async def test_discord_issue_transport_error_preserves_operation_for_resume() -> None:
    transport_error = ManagedOpenRouterReleaseError(
        code="trial_unavailable",
        error_class="retryable",
        message="broker transport failure",
        operation="discord_issue",
    )
    harness = FakeDiscordOAuthHarness()
    client: Any = OperationFakeClient(
        issue_error=transport_error,
        start_result=_make_discord_start_success(redirect_uri=harness.redirect_uri),
    )
    service, settings, secrets, _client, _harness = _make_discord_service(
        client=client, harness=harness
    )

    result = await service.prepare_for_translation()

    assert result.behavior == ManagedOpenRouterReleaseBehavior.RETRY
    assert result.message_key == "discord_auth.error.recovery_pending"
    operation_id = settings.managed_identity.pending_managed_operation_id
    assert operation_id is not None
    assert secrets.get(MANAGED_OPERATION_RESUME_TOKEN_SECRET) is not None
    issue_calls = [payload for name, payload in client.calls if name == "discord_issue"]
    assert len(issue_calls) == 1
    assert issue_calls[0]["operation_id"] == operation_id
    assert "resume_token" in issue_calls[0]
