from __future__ import annotations

import json
import socket
import ssl
from collections.abc import Callable

import httpx
import pytest

from puripuly_heart.app.ports.broker_client import (
    ManagedKeyDeliveryAckRequest,
    QqManagedAssertionRequest,
)
from puripuly_heart.core.diagnostics import sanitize_diagnostics
from puripuly_heart.core.openrouter.managed_openrouter_release import (
    ManagedOpenRouterChallengeSuccess,
    ManagedOpenRouterDiscordStartSuccess,
    ManagedOpenRouterFingerprintSalt,
    ManagedOpenRouterIssueSuccess,
    ManagedOpenRouterReleaseError,
    ManagedOpenRouterVerifySuccess,
)


class TrackingTransport(httpx.AsyncBaseTransport):
    def __init__(self, handler: Callable[[httpx.Request], httpx.Response]) -> None:
        self._handler = handler
        self.closed = False

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        return self._handler(request)

    async def aclose(self) -> None:
        self.closed = True


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["timeout", "proxy", "tls", "dns"])
async def test_qq_transport_failure_keeps_safe_network_classification(kind):
    cause = {
        "timeout": httpx.ReadTimeout("secret-response"),
        "proxy": httpx.ProxyError("http://secret:password@proxy"),
        "tls": ssl.SSLCertVerificationError("secret-certificate"),
        "dns": socket.gaierror(-2, "secret-host"),
    }[kind]

    def fail(request):
        error = httpx.ConnectError("raw-credential")
        error.__cause__ = cause
        error.connection_diagnostics = {
            "proxy_source": "system",
            "tls_source": "windows",
            "tls_backend": "native",
        }
        raise error

    from puripuly_heart.app.ports.broker_client import QqManagedAssertionRequest

    client, transport = _build_client(fail)
    result = await client.assert_qq_managed_identity(
        QqManagedAssertionRequest(
            qq_identity="secret-qq-identity",
            credential="a" * 64,
            asserted_at="2026-10-12T00:00:00.000Z",
            metadata={},
        )
    )
    assert result.failure_subcode == "broker_unavailable"
    assert result.diagnostics.fields["network_error_kind"] == kind
    assert result.diagnostics.fields["proxy_source"] == "system"
    for secret in ("secret-response", "password", "raw-credential", "secret-qq-identity"):
        assert secret not in repr(result)
    await client.close()


def _build_client(
    handler: Callable[[httpx.Request], httpx.Response],
    *,
    base_url: str = "https://broker.example.test",
) -> tuple[object, TrackingTransport]:
    from puripuly_heart.core.openrouter.managed_openrouter_broker_client import (
        HttpManagedOpenRouterBrokerClient,
    )

    transport = TrackingTransport(handler)
    return (
        HttpManagedOpenRouterBrokerClient(base_url=base_url, transport=transport, timeout=1.0),
        transport,
    )


@pytest.mark.asyncio
async def test_challenge_parses_fingerprint_salt() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/v1/trial/challenge"
        assert json.loads(request.content) == {
            "installation_id": "install-123",
            "device_public_key": "device-public-key-123",
            "app_version": "2.0.0",
        }
        return httpx.Response(
            200,
            json={
                "challenge": "challenge-123",
                "challenge_expires_at": "2026-04-10T06:05:00.000Z",
                "fingerprint_salt": {
                    "version": 7,
                    "salt": "fingerprint-salt-123",
                },
                "managed_state": {
                    "lifecycle": "none",
                    "managed_availability": True,
                },
                "current_entitlement": None,
            },
        )

    client, _transport = _build_client(handler)

    result = await client.challenge(
        installation_id="install-123",
        device_public_key="device-public-key-123",
        app_version="2.0.0",
    )

    assert result == ManagedOpenRouterChallengeSuccess(
        challenge="challenge-123",
        challenge_expires_at="2026-04-10T06:05:00.000Z",
        fingerprint_salt=ManagedOpenRouterFingerprintSalt(
            version=7,
            salt="fingerprint-salt-123",
        ),
    )
    await client.close()


@pytest.mark.asyncio
async def test_start_discord_oauth_posts_redirect_and_parses_success_payload() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/v1/auth/discord/start"
        assert json.loads(request.content) == {
            "installation_id": "install-discord-123",
            "device_public_key": "device-public-key-123",
            "redirect_uri": "http://127.0.0.1:62187/discord/callback",
            "app_version": "2.0.0",
        }
        return httpx.Response(
            200,
            json={
                "authorization_url": "https://discord.com/oauth2/authorize?state=state-123",
                "redirect_uri": "http://127.0.0.1:62187/discord/callback",
                "oauth_session_expires_at": "2026-04-30T06:05:00.000Z",
                "issue_nonce": "issue-nonce-123",
                "fingerprint_salt": {
                    "version": 7,
                    "salt": "fingerprint-salt-123",
                },
                "fingerprint_salt_version": 7,
            },
        )

    client, _transport = _build_client(handler)

    result = await client.start_discord_oauth(
        installation_id="install-discord-123",
        device_public_key="device-public-key-123",
        redirect_uri="http://127.0.0.1:62187/discord/callback",
        app_version="2.0.0",
    )

    assert result == ManagedOpenRouterDiscordStartSuccess(
        authorization_url="https://discord.com/oauth2/authorize?state=state-123",
        redirect_uri="http://127.0.0.1:62187/discord/callback",
        oauth_session_expires_at="2026-04-30T06:05:00.000Z",
        issue_nonce="issue-nonce-123",
        fingerprint_salt=ManagedOpenRouterFingerprintSalt(
            version=7,
            salt="fingerprint-salt-123",
        ),
        fingerprint_salt_version=7,
    )
    await client.close()


@pytest.mark.asyncio
async def test_verify_parses_success_payload() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/v1/trial/challenge/verify"
        assert json.loads(request.content) == {
            "installation_id": "install-123",
            "device_public_key": "device-public-key-123",
            "challenge": "challenge-123",
            "challenge_expires_at": "2026-04-10T06:05:00.000Z",
            "hardware_hash": "hardware-hash-123",
            "app_version": "2.0.0",
            "signed_at": "2026-04-10T06:00:45.000Z",
            "signature": "signature-123",
        }
        return httpx.Response(
            200,
            json={
                "release_token": "release-token-123",
                "release_token_expires_at": "2026-04-10T06:15:00.000Z",
                "managed_state": {
                    "lifecycle": "pending_release",
                    "managed_availability": True,
                },
            },
        )

    client, _transport = _build_client(handler)

    result = await client.verify(
        {
            "installation_id": "install-123",
            "device_public_key": "device-public-key-123",
            "challenge": "challenge-123",
            "challenge_expires_at": "2026-04-10T06:05:00.000Z",
            "hardware_hash": "hardware-hash-123",
            "app_version": "2.0.0",
            "signed_at": "2026-04-10T06:00:45.000Z",
            "signature": "signature-123",
        }
    )

    assert result == ManagedOpenRouterVerifySuccess(
        release_token="release-token-123",
        release_token_expires_at="2026-04-10T06:15:00.000Z",
    )
    await client.close()


@pytest.mark.asyncio
async def test_issue_parses_success_payload() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/v1/providers/openrouter/issue"
        assert json.loads(request.content) == {
            "installation_id": "install-123",
            "device_public_key": "device-public-key-123",
            "release_token": "release-token-123",
            "hardware_hash": "hardware-hash-123",
            "reason": "llm_start",
            "budget_usd": 0.07,
            "model": "google/gemma-4-26b-a4b-it",
            "signed_at": "2026-04-10T06:00:45.000Z",
            "signature": "signature-123",
        }
        return httpx.Response(
            200,
            json={
                "openrouter_api_key": "managed-openrouter-api-key",
                "managed_credential_ref": "managed-credential-ref-123",
                "expires_at": "2026-10-10T06:00:00.000Z",
                "managed_state": {
                    "lifecycle": "active",
                    "managed_availability": True,
                },
                "budget_usd": 0.07,
                "model": "google/gemma-4-26b-a4b-it",
            },
        )

    client, _transport = _build_client(handler)

    result = await client.issue(
        {
            "installation_id": "install-123",
            "device_public_key": "device-public-key-123",
            "release_token": "release-token-123",
            "hardware_hash": "hardware-hash-123",
            "reason": "llm_start",
            "budget_usd": 0.07,
            "model": "google/gemma-4-26b-a4b-it",
            "signed_at": "2026-04-10T06:00:45.000Z",
            "signature": "signature-123",
        }
    )

    assert result == ManagedOpenRouterIssueSuccess(
        openrouter_api_key="managed-openrouter-api-key",
        managed_credential_ref="managed-credential-ref-123",
        expires_at="2026-10-10T06:00:00.000Z",
    )
    await client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "metadata",
    [
        {},
        {
            "delivery_id": "delivery-legacy",
            "delivery_ack_token": "synthetic-sensitive-ack-token",
            "delivery_ack_expires_at": "2099-01-01T00:00:00Z",
        },
        {"delivery_id": 7, "delivery_ack_token": None},
    ],
)
async def test_legacy_issue_rejects_ack_required_wire_contract(metadata) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/providers/openrouter/issue"
        return httpx.Response(
            200,
            json={
                "openrouter_api_key": "synthetic-sensitive-legacy-key",
                "managed_credential_ref": "legacy-ref",
                "delivery_ack_required": True,
                **metadata,
            },
        )

    client, transport = _build_client(handler)
    try:
        with pytest.raises(ManagedOpenRouterReleaseError) as exc_info:
            await client.issue({})
        error = exc_info.value
        assert error.code == "managed_delivery_protocol_mismatch"
        assert error.error_class == "terminal"
        assert error.operation == "issue"
        assert error.http_status == 200
        assert error.failure_diagnostics.status_code == 200
        assert error.failure_diagnostics.category == "invalid_response"
        assert error.failure_diagnostics.fields["stage"] == "protocol_parse"
        assert error.failure_diagnostics.fields["dispatch_certainty"] == "response_received"
        assert "synthetic-sensitive" not in repr(error.failure_diagnostics)
    finally:
        await client.close()
    assert transport.closed


@pytest.mark.asyncio
async def test_issue_discord_managed_key_posts_signed_payload_and_parses_success() -> None:
    request_body = {
        "code": "discord-oauth-code-123",
        "state": "discord-state-123",
        "installation_id": "install-discord-123",
        "device_public_key": "device-public-key-123",
        "redirect_uri": "http://127.0.0.1:62187/discord/callback",
        "hardware_hash": "hardware-hash-123",
        "hardware_hash_salt_version": 7,
        "app_version": "2.0.0",
        "reason": "llm_start",
        "budget_usd": 0.07,
        "model": "google/gemma-4-26b-a4b-it",
        "issue_nonce": "issue-nonce-123",
        "signed_at": "2026-04-30T06:00:30.000Z",
        "signature_alg": "ed25519",
        "signature": "signature-123",
    }

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/v1/providers/openrouter/discord/issue"
        assert json.loads(request.content) == {
            **request_body,
            "delivery_ack_supported": True,
        }
        return httpx.Response(
            200,
            json={
                "openrouter_api_key": "managed-openrouter-api-key",
                "managed_credential_ref": "managed-credential-ref-123",
                "expires_at": "2026-07-30T06:00:00.000Z",
                "openrouter_user_id": " user-123 ",
                "managed_state": {
                    "lifecycle": "active",
                    "managed_availability": True,
                },
                "budget_usd": 0.07,
                "model": "google/gemma-4-26b-a4b-it",
            },
        )

    client, _transport = _build_client(handler)

    result = await client.issue_discord_managed_key(request_body)

    assert result == ManagedOpenRouterIssueSuccess(
        openrouter_api_key="managed-openrouter-api-key",
        managed_credential_ref="managed-credential-ref-123",
        expires_at="2026-07-30T06:00:00.000Z",
        openrouter_user_id="user-123",
    )
    await client.close()


@pytest.mark.asyncio
async def test_issue_discord_managed_key_parses_delivery_ack_metadata() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/v1/providers/openrouter/discord/issue"
        assert json.loads(request.content)["delivery_ack_supported"] is True
        return httpx.Response(
            200,
            json={
                "openrouter_api_key": "managed-openrouter-api-key",
                "managed_credential_ref": "managed-credential-ref-123",
                "expires_at": "2026-07-30T06:00:00.000Z",
                "delivery_ack_required": True,
                "delivery_id": "delivery-discord-123",
                "delivery_ack_token": "delivery-ack-token-123",
                "delivery_ack_expires_at": "2026-07-30T06:05:00.000Z",
            },
        )

    client, _transport = _build_client(handler)

    result = await client.issue_discord_managed_key({"code": "discord-oauth-code-123"})

    assert result.delivery_ack_required is True
    assert result.delivery_id == "delivery-discord-123"
    assert result.delivery_ack_token == "delivery-ack-token-123"
    assert result.delivery_ack_expires_at == "2026-07-30T06:05:00.000Z"
    assert "delivery-ack-token-123" not in repr(result)
    await client.close()


@pytest.mark.asyncio
async def test_assert_qq_managed_identity_posts_credentials_and_maps_issued_snapshot() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/v1/auth/qq/assert"
        assert json.loads(request.content) == {
            "qq_identity": "qq-user-123",
            "credential": "a" * 64,
            "asserted_at": "2026-07-03T06:00:00.000Z",
            "delivery_ack_supported": True,
        }
        return httpx.Response(
            200,
            json={
                "ok": True,
                "status": "issued",
                "qq_subject_ref": "ph-qq-subject-v1_subject",
                "openrouter_api_key": "qq-managed-openrouter-api-key",
                "managed_credential_ref": "managed-ref-qq",
                "expires_at": "2026-08-03T06:00:00.000Z",
                "openrouter_user_id": " user-qq ",
            },
        )

    client, _transport = _build_client(handler)

    result = await client.assert_qq_managed_identity(
        QqManagedAssertionRequest(
            qq_identity="qq-user-123",
            credential="a" * 64,
            asserted_at="2026-07-03T06:00:00.000Z",
            metadata={"flow": "qq_managed"},
        )
    )

    assert result.succeeded is True
    assert result.managed_secret_key == "qq-managed-openrouter-api-key"
    assert result.entitlement is not None
    assert result.entitlement.qq_subject_ref == "ph-qq-subject-v1_subject"
    assert result.entitlement.managed_credential_ref == "managed-ref-qq"
    assert result.entitlement.expires_at == "2026-08-03T06:00:00.000Z"
    assert result.entitlement.openrouter_user_id == "user-qq"
    assert "qq-managed-openrouter-api-key" not in repr(result)
    await client.close()


@pytest.mark.asyncio
async def test_assert_qq_managed_identity_parses_delivery_ack_metadata() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/v1/auth/qq/assert"
        assert json.loads(request.content)["delivery_ack_supported"] is True
        return httpx.Response(
            200,
            json={
                "ok": True,
                "status": "delivery_pending",
                "qq_subject_ref": "ph-qq-subject-v1_subject",
                "openrouter_api_key": "qq-managed-openrouter-api-key",
                "managed_credential_ref": "managed-ref-qq",
                "expires_at": "2026-08-03T06:00:00.000Z",
                "delivery_ack_required": True,
                "delivery_id": "delivery-qq-123",
                "delivery_ack_token": "delivery-ack-token-qq",
                "delivery_ack_expires_at": "2026-08-03T06:05:00.000Z",
            },
        )

    client, _transport = _build_client(handler)

    result = await client.assert_qq_managed_identity(
        QqManagedAssertionRequest(
            qq_identity="qq-user-123",
            credential="a" * 64,
            asserted_at="2026-07-03T06:00:00.000Z",
            metadata={"flow": "qq_managed"},
        )
    )

    assert result.succeeded is True
    assert result.delivery_ack is not None
    assert result.delivery_ack.source == "qq"
    assert result.delivery_ack.delivery_id == "delivery-qq-123"
    assert result.delivery_ack.managed_credential_ref == "managed-ref-qq"
    assert result.delivery_ack.delivery_ack_token == "delivery-ack-token-qq"
    assert result.delivery_ack.expires_at == "2026-08-03T06:05:00.000Z"
    assert "delivery-ack-token-qq" not in repr(result.delivery_ack)
    await client.close()


@pytest.mark.asyncio
async def test_acknowledge_managed_key_delivery_posts_token_and_maps_success() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/v1/providers/openrouter/managed-key-delivery/ack"
        assert json.loads(request.content) == {
            "delivery_id": "delivery-123",
            "managed_credential_ref": "managed-credential-ref-123",
            "delivery_ack_token": "delivery-ack-token-123",
        }
        return httpx.Response(
            200,
            json={
                "ok": True,
                "status": "acknowledged",
            },
        )

    client, _transport = _build_client(handler)

    result = await client.acknowledge_managed_key_delivery(
        ManagedKeyDeliveryAckRequest(
            delivery_id="delivery-123",
            managed_credential_ref="managed-credential-ref-123",
            delivery_ack_token="delivery-ack-token-123",
        )
    )

    assert result.succeeded is True
    assert result.status == "acknowledged"
    assert result.diagnostics is None
    await client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("rejection", "error_class", "http_status", "expected_status"),
    [
        ("invalid", "terminal", 404, "invalid"),
        ("expired", "terminal", 410, "expired"),
        ("mismatched", "terminal", 409, "mismatched"),
        ("failed", "retryable", 409, "retryable"),
        ("rate_limited", "retryable", 429, "retryable"),
    ],
)
async def test_acknowledge_managed_key_delivery_maps_broker_error(
    rejection, error_class, http_status, expected_status
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/v1/providers/openrouter/managed-key-delivery/ack"
        return httpx.Response(
            http_status,
            json={
                "error": {
                    "code": "rate_limited" if rejection == "rate_limited" else "invalid_request",
                    "class": error_class,
                    "subcode": f"delivery_ack_{rejection}",
                    "message": "synthetic-sensitive delivery ACK rejection",
                    "retry_after_ms": 19,
                }
            },
        )

    client, _transport = _build_client(handler)

    result = await client.acknowledge_managed_key_delivery(
        ManagedKeyDeliveryAckRequest(
            delivery_id="delivery-123",
            managed_credential_ref="managed-credential-ref-123",
            delivery_ack_token="delivery-ack-token-123",
        )
    )

    assert result.succeeded is False
    assert result.status == expected_status
    assert result.diagnostics is not None
    assert result.diagnostics.code == "managed_key_delivery_ack_error"
    assert result.diagnostics.fields["broker_subcode"] == f"delivery_ack_{rejection}"
    assert result.diagnostics.status_code == http_status
    assert result.diagnostics.retry_after_ms == 19
    assert result.diagnostics.fields["ack_status"] == expected_status
    assert "synthetic-sensitive" not in repr(result.diagnostics)
    normalized = sanitize_diagnostics(result.diagnostics)
    assert normalized.fields["broker_subcode"] == f"delivery_ack_{rejection}"
    assert normalized.fields["broker_code"] == (
        "rate_limited" if rejection == "rate_limited" else "invalid_request"
    )
    assert normalized.fields["ack_status"] == expected_status
    assert normalized.status_code == http_status
    await client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error_payload", "expected_subcode"),
    [
        (
            {
                "code": "invalid_request",
                "class": "security_fail",
                "subcode": "qq_credential_invalid",
                "message": "invalid credential",
            },
            "invalid_credential",
        ),
        (
            {
                "code": "invalid_request",
                "class": "security_fail",
                "subcode": "qq_identity_mismatch",
                "message": "mismatch",
            },
            "mismatch",
        ),
        (
            {
                "code": "trial_not_eligible",
                "class": "terminal",
                "subcode": "qq_lifetime_used",
                "message": "used",
            },
            "lifetime_used",
        ),
        (
            {
                "code": "rate_limited",
                "class": "retryable",
                "subcode": "qq_auth_rate_limited",
                "retry_after_ms": 3000,
                "message": "rate limited",
            },
            "rate_limited",
        ),
        (
            {
                "code": "internal_error",
                "class": "retryable",
                "subcode": None,
                "message": "temporarily unavailable",
            },
            "key_unavailable",
        ),
    ],
)
async def test_assert_qq_managed_identity_maps_recoverable_error_subcodes(
    error_payload: dict[str, object],
    expected_subcode: str,
) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": error_payload})

    client, _transport = _build_client(handler)

    result = await client.assert_qq_managed_identity(
        QqManagedAssertionRequest(
            qq_identity="qq-user-123",
            credential="b" * 64,
            asserted_at="2026-07-03T06:00:00.000Z",
            metadata={"flow": "qq_managed"},
        )
    )

    assert result.succeeded is False
    assert result.managed_secret_key is None
    assert result.failure_subcode == expected_subcode
    assert result.entitlement is None
    assert result.diagnostics is not None
    assert result.diagnostics.content_policy == "metadata_only"
    assert result.diagnostics.fields["qq_failure_subcode"] == expected_subcode
    rendered = repr(result)
    assert "qq-user-123" not in rendered
    assert "bbbb" not in rendered
    await client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["verified", "already_verified"])
async def test_assert_qq_managed_identity_maps_legacy_verified_without_key_to_key_unavailable(
    status: str,
) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "ok": True,
                "status": status,
                "qq_subject_ref": "ph-qq-subject-v1_subject",
            },
        )

    client, _transport = _build_client(handler)

    result = await client.assert_qq_managed_identity(
        QqManagedAssertionRequest(
            qq_identity="qq-user-123",
            credential="c" * 64,
            asserted_at="2026-07-03T06:00:00.000Z",
            metadata={"flow": "qq_managed"},
        )
    )

    assert result.succeeded is False
    assert result.failure_subcode == "key_unavailable"
    assert result.diagnostics is not None
    assert result.diagnostics.fields["status"] == status
    await client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model",
    [
        "google/gemma-4-26b-a4b-it",
        "qwen/qwen3.5-flash-02-23",
        "google/gemini-2.5-flash-lite",
    ],
)
async def test_issue_accepts_curated_managed_model_pool(model: str) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["model"] == model
        return httpx.Response(
            200,
            json={
                "openrouter_api_key": "managed-openrouter-api-key",
                "managed_credential_ref": "managed-credential-ref-123",
                "expires_at": "2026-10-10T06:00:00.000Z",
                "managed_state": {
                    "lifecycle": "active",
                    "managed_availability": True,
                },
                "budget_usd": 0.07,
                "model": model,
            },
        )

    client, _transport = _build_client(handler)

    result = await client.issue(
        {
            "installation_id": "install-123",
            "device_public_key": "device-public-key-123",
            "release_token": "release-token-123",
            "hardware_hash": "hardware-hash-123",
            "reason": "llm_start",
            "budget_usd": 0.07,
            "model": model,
            "signed_at": "2026-04-10T06:00:45.000Z",
            "signature": "signature-123",
        }
    )

    assert result.openrouter_api_key == "managed-openrouter-api-key"
    await client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [
        {
            "openrouter_api_key": "managed-openrouter-api-key",
            "expires_at": "2026-10-10T06:00:00.000Z",
        },
        {
            "openrouter_api_key": "managed-openrouter-api-key",
            "managed_credential_ref": "managed-credential-ref-123",
        },
        {
            "openrouter_api_key": "managed-openrouter-api-key",
            "managed_credential_ref": None,
            "expires_at": "2026-10-10T06:00:00.000Z",
        },
        {
            "openrouter_api_key": "managed-openrouter-api-key",
            "managed_credential_ref": "managed-credential-ref-123",
            "expires_at": None,
        },
    ],
)
async def test_issue_accepts_missing_or_null_optional_success_fields(
    payload: dict[str, object],
) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                **payload,
                "managed_state": {
                    "lifecycle": "active",
                    "managed_availability": True,
                },
                "budget_usd": 0.07,
                "model": "google/gemma-4-26b-a4b-it",
            },
        )

    client, _transport = _build_client(handler)

    result = await client.issue(
        {
            "installation_id": "install-123",
            "device_public_key": "device-public-key-123",
            "release_token": "release-token-123",
            "hardware_hash": "hardware-hash-123",
            "reason": "llm_start",
            "budget_usd": 0.07,
            "model": "google/gemma-4-26b-a4b-it",
            "signed_at": "2026-04-10T06:00:45.000Z",
            "signature": "signature-123",
        }
    )

    assert result == ManagedOpenRouterIssueSuccess(
        openrouter_api_key="managed-openrouter-api-key",
        managed_credential_ref=(
            payload.get("managed_credential_ref") if "managed_credential_ref" in payload else None
        ),
        expires_at=payload.get("expires_at") if "expires_at" in payload else None,
    )
    await client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("payload", "expected_openrouter_user_id"),
    [
        ({}, None),
        ({"openrouter_user_id": None}, None),
        ({"openrouter_user_id": "   "}, None),
        ({"openrouter_user_id": 123}, None),
        ({"openrouter_user_id": {"id": "user-123"}}, None),
        ({"openrouter_user_id": " user-123 "}, "user-123"),
    ],
)
async def test_issue_tolerates_optional_opaque_openrouter_user_id(
    payload: dict[str, object],
    expected_openrouter_user_id: str | None,
) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "openrouter_api_key": "managed-openrouter-api-key",
                "managed_credential_ref": "managed-credential-ref-123",
                "expires_at": "2026-10-10T06:00:00.000Z",
                **payload,
                "managed_state": {
                    "lifecycle": "active",
                    "managed_availability": True,
                },
                "budget_usd": 0.07,
                "model": "google/gemma-4-26b-a4b-it",
            },
        )

    client, _transport = _build_client(handler)

    result = await client.issue(
        {
            "installation_id": "install-123",
            "device_public_key": "device-public-key-123",
            "release_token": "release-token-123",
            "hardware_hash": "hardware-hash-123",
            "reason": "llm_start",
            "budget_usd": 0.07,
            "model": "google/gemma-4-26b-a4b-it",
            "signed_at": "2026-04-10T06:00:45.000Z",
            "signature": "signature-123",
        }
    )

    assert result == ManagedOpenRouterIssueSuccess(
        openrouter_api_key="managed-openrouter-api-key",
        managed_credential_ref="managed-credential-ref-123",
        expires_at="2026-10-10T06:00:00.000Z",
        openrouter_user_id=expected_openrouter_user_id,
    )
    await client.close()


@pytest.mark.asyncio
async def test_nested_broker_error_envelope_becomes_release_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429,
            json={
                "error": {
                    "code": "trial_unavailable",
                    "class": "retryable",
                    "subcode": "broker_backoff",
                    "retry_after_ms": 9000,
                    "message": "broker is temporarily unavailable",
                },
                "managed_state": {
                    "lifecycle": "none",
                    "managed_availability": True,
                },
                "current_entitlement": None,
            },
        )

    client, _transport = _build_client(handler)

    with pytest.raises(ManagedOpenRouterReleaseError) as exc_info:
        await client.issue(
            {
                "installation_id": "install-123",
                "device_public_key": "device-public-key-123",
                "release_token": "release-token-123",
                "hardware_hash": "hardware-hash-123",
                "reason": "llm_start",
                "budget_usd": 0.07,
                "model": "google/gemma-4-26b-a4b-it",
                "signed_at": "2026-04-10T06:00:45.000Z",
                "signature": "signature-123",
            }
        )

    assert exc_info.value == ManagedOpenRouterReleaseError(
        operation="issue",
        code="trial_unavailable",
        error_class="retryable",
        subcode="broker_backoff",
        retry_after_ms=9000,
        message="broker is temporarily unavailable",
        managed_lifecycle="none",
    )
    await client.close()


@pytest.mark.asyncio
async def test_issue_preserves_managed_key_unrecoverable_subcode() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            409,
            json={
                "error": {
                    "code": "trial_not_eligible",
                    "class": "terminal",
                    "subcode": "managed_key_unrecoverable",
                    "retry_after_ms": None,
                    "message": "managed key was already issued and cannot be recovered",
                }
            },
        )

    client, _transport = _build_client(handler)

    with pytest.raises(ManagedOpenRouterReleaseError) as exc_info:
        await client.issue(
            {
                "installation_id": "install-123",
                "device_public_key": "device-public-key-123",
                "release_token": "release-token-123",
                "hardware_hash": "hardware-hash-123",
                "reason": "llm_start",
                "budget_usd": 0.07,
                "model": "google/gemma-4-26b-a4b-it",
                "signed_at": "2026-04-10T06:00:45.000Z",
                "signature": "signature-123",
            }
        )

    assert exc_info.value == ManagedOpenRouterReleaseError(
        operation="issue",
        code="trial_not_eligible",
        error_class="terminal",
        subcode="managed_key_unrecoverable",
        retry_after_ms=None,
        message="managed key was already issued and cannot be recovered",
    )
    await client.close()


@pytest.mark.asyncio
async def test_issue_parses_revoked_lifecycle_from_error_envelope() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            409,
            json={
                "error": {
                    "code": "trial_not_eligible",
                    "class": "terminal",
                    "subcode": None,
                    "retry_after_ms": None,
                    "message": "manual review required",
                },
                "managed_state": {
                    "lifecycle": "revoked",
                    "managed_availability": False,
                },
                "current_entitlement": None,
            },
        )

    client, _transport = _build_client(handler)

    with pytest.raises(ManagedOpenRouterReleaseError) as exc_info:
        await client.issue(
            {
                "installation_id": "install-123",
                "device_public_key": "device-public-key-123",
                "release_token": "release-token-123",
                "hardware_hash": "hardware-hash-123",
                "reason": "llm_start",
                "budget_usd": 0.07,
                "model": "google/gemma-4-26b-a4b-it",
                "signed_at": "2026-04-10T06:00:45.000Z",
                "signature": "signature-123",
            }
        )

    assert exc_info.value.managed_lifecycle == "revoked"
    await client.close()


@pytest.mark.asyncio
async def test_issue_preserves_challenge_expired_release_token_subcode() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            410,
            json={
                "error": {
                    "code": "challenge_expired",
                    "class": "retryable",
                    "subcode": "release_token_expired",
                    "retry_after_ms": 0,
                    "message": "release_token has expired and must be reissued",
                }
            },
        )

    client, _transport = _build_client(handler)

    with pytest.raises(ManagedOpenRouterReleaseError) as exc_info:
        await client.issue(
            {
                "installation_id": "install-123",
                "device_public_key": "device-public-key-123",
                "release_token": "release-token-123",
                "hardware_hash": "hardware-hash-123",
                "reason": "llm_start",
                "budget_usd": 0.07,
                "model": "google/gemma-4-26b-a4b-it",
                "signed_at": "2026-04-10T06:00:45.000Z",
                "signature": "signature-123",
            }
        )

    assert exc_info.value == ManagedOpenRouterReleaseError(
        operation="issue",
        code="challenge_expired",
        error_class="retryable",
        subcode="release_token_expired",
        retry_after_ms=0,
        message="release_token has expired and must be reissued",
    )
    await client.close()


@pytest.mark.asyncio
async def test_issue_legacy_top_level_release_token_expired_becomes_retryable_malformed_error() -> (
    None
):
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            410,
            json={
                "error": {
                    "code": "release_token_expired",
                    "class": "retryable",
                    "subcode": None,
                    "retry_after_ms": None,
                    "message": "legacy top-level release token expiration code",
                }
            },
        )

    client, _transport = _build_client(handler)

    with pytest.raises(ManagedOpenRouterReleaseError) as exc_info:
        await client.issue(
            {
                "installation_id": "install-123",
                "device_public_key": "device-public-key-123",
                "release_token": "release-token-123",
                "hardware_hash": "hardware-hash-123",
                "reason": "llm_start",
                "budget_usd": 0.07,
                "model": "google/gemma-4-26b-a4b-it",
                "signed_at": "2026-04-10T06:00:45.000Z",
                "signature": "signature-123",
            }
        )

    assert exc_info.value.code == "trial_unavailable"
    assert exc_info.value.error_class == "retryable"
    assert "malformed error payload" in exc_info.value.message
    await client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error_payload",
    [
        {
            "code": "surprise_error",
            "class": "retryable",
            "subcode": None,
            "retry_after_ms": None,
            "message": "unexpected code",
        },
        {
            "code": "trial_unavailable",
            "class": "surprise_class",
            "subcode": None,
            "retry_after_ms": None,
            "message": "unexpected class",
        },
    ],
)
async def test_unknown_broker_error_vocabulary_becomes_retryable_malformed_error(
    error_payload: dict[str, object],
) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": error_payload})

    client, _transport = _build_client(handler)

    with pytest.raises(ManagedOpenRouterReleaseError) as exc_info:
        await client.verify(
            {
                "installation_id": "install-123",
                "device_public_key": "device-public-key-123",
                "challenge": "challenge-123",
                "challenge_expires_at": "2026-04-10T06:05:00.000Z",
                "hardware_hash": "hardware-hash-123",
                "app_version": "2.0.0",
                "signed_at": "2026-04-10T06:00:45.000Z",
                "signature": "signature-123",
            }
        )

    assert exc_info.value.code == "trial_unavailable"
    assert exc_info.value.error_class == "retryable"
    assert "malformed error payload" in exc_info.value.message
    await client.close()


@pytest.mark.asyncio
async def test_malformed_discord_start_response_becomes_retryable_release_error() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "redirect_uri": "http://127.0.0.1:62187/discord/callback",
                "oauth_session_expires_at": "2026-04-30T06:05:00.000Z",
                "issue_nonce": "issue-nonce-123",
                "fingerprint_salt": {
                    "version": 7,
                    "salt": "fingerprint-salt-123",
                },
                "fingerprint_salt_version": 7,
            },
        )

    client, _transport = _build_client(handler)

    with pytest.raises(ManagedOpenRouterReleaseError) as exc_info:
        await client.start_discord_oauth(
            installation_id="install-discord-123",
            device_public_key="device-public-key-123",
            redirect_uri="http://127.0.0.1:62187/discord/callback",
            app_version="2.0.0",
        )

    assert exc_info.value.operation == "discord_start"
    assert exc_info.value.code == "trial_unavailable"
    assert exc_info.value.error_class == "retryable"
    assert "malformed payload" in exc_info.value.message
    await client.close()


def test_rejects_broker_base_url_with_path_prefix() -> None:
    with pytest.raises(ValueError, match="path prefix"):
        _build_client(
            lambda _request: httpx.Response(200, json={}),
            base_url="https://broker.example.test/prefix",
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("label", "handler"),
    [
        (
            "timeout",
            lambda request: (_ for _ in ()).throw(
                httpx.ReadTimeout("request timed out", request=request)
            ),
        ),
        (
            "network",
            lambda request: (_ for _ in ()).throw(
                httpx.ConnectError("network unavailable", request=request)
            ),
        ),
        (
            "malformed_json",
            lambda _request: httpx.Response(
                200,
                headers={"content-type": "application/json"},
                content=b"{",
            ),
        ),
    ],
)
async def test_transport_failures_and_malformed_json_become_retryable_release_errors(
    label: str,
    handler: Callable[[httpx.Request], httpx.Response],
) -> None:
    client, _transport = _build_client(handler)

    with pytest.raises(ManagedOpenRouterReleaseError) as exc_info:
        await client.verify(
            {
                "installation_id": "install-123",
                "device_public_key": "device-public-key-123",
                "challenge": "challenge-123",
                "challenge_expires_at": "2026-04-10T06:05:00.000Z",
                "hardware_hash": "hardware-hash-123",
                "app_version": "2.0.0",
                "signed_at": "2026-04-10T06:00:45.000Z",
                "signature": "signature-123",
            }
        )

    assert exc_info.value.code == "trial_unavailable"
    assert exc_info.value.error_class == "retryable"
    assert exc_info.value.retry_after_ms is None
    assert isinstance(exc_info.value.message, str)
    assert exc_info.value.message
    await client.close()


@pytest.mark.asyncio
async def test_close_closes_underlying_client() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "release_token": "release-token-123",
                "release_token_expires_at": "2026-04-10T06:15:00.000Z",
            },
        )

    client, transport = _build_client(handler)

    await client.verify(
        {
            "installation_id": "install-123",
            "device_public_key": "device-public-key-123",
            "challenge": "challenge-123",
            "challenge_expires_at": "2026-04-10T06:05:00.000Z",
            "hardware_hash": "hardware-hash-123",
            "app_version": "2.0.0",
            "signed_at": "2026-04-10T06:00:45.000Z",
            "signature": "signature-123",
        }
    )

    internal_client = client._client
    assert internal_client is not None
    assert internal_client.is_closed is False
    assert transport.closed is False

    await client.close()

    assert internal_client.is_closed is True
    assert transport.closed is True
    assert client._client is None


@pytest.mark.asyncio
async def test_get_managed_operation_status_posts_identity_and_parses_contract() -> None:
    from puripuly_heart.app.ports.broker_client import ManagedOperationStatusRequest

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/v1/providers/openrouter/managed-operation/status"
        assert json.loads(request.content) == {
            "operation_id": "ph-mop-v1_AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
            "installation_id": "install-1",
            "resume_token": "token-1",
        }
        return httpx.Response(
            200,
            json={
                "ok": True,
                "operation_id": "ph-mop-v1_AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
                "issue_source": "discord",
                "state": "RECONCILING",
                "client_action": "wait",
                "failure_reason": None,
                "attempt_count": 1,
                "current_attempt_index": 1,
                "auth_expires_at": "2026-12-01T01:00:00.000Z",
                "attempts": [
                    {
                        "attempt_index": 1,
                        "provider_key_name": "puripuly-heart:mop:discord:AAAAAAAAAAAAAAAA:a1",
                        "managed_credential_ref": "managed-ref-1",
                        "outcome": "unknown",
                    }
                ],
                "delivery": {
                    "delivery_id": "delivery-1",
                    "status": "cleanup_required",
                    "expires_at": "2026-12-01T00:15:00.000Z",
                },
            },
        )

    client, _transport = _build_client(handler)

    result = await client.get_managed_operation_status(
        ManagedOperationStatusRequest(
            operation_id="ph-mop-v1_AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
            installation_id="install-1",
            resume_token="token-1",
        )
    )

    assert result.succeeded is True
    assert result.operation_status == "RECONCILING"
    assert result.client_action == "wait"
    assert result.attempt is not None
    assert result.attempt.attempt_index == 1
    assert result.attempt.outcome == "unknown"
    assert result.attempt.managed_credential_ref == "managed-ref-1"
    assert result.delivery is not None
    assert result.delivery.status == "cleanup_required"
    assert result.failed_reason is None
    assert result.managed_secret_key is None
    await client.close()


@pytest.mark.asyncio
async def test_resume_managed_operation_parses_failed_body_with_failure_reason() -> None:
    from puripuly_heart.app.ports.broker_client import ManagedOperationResumeRequest

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/v1/providers/openrouter/managed-operation/resume"
        assert json.loads(request.content) == {
            "operation_id": "ph-mop-v1_BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB",
            "installation_id": "install-1",
            "resume_token": "token-2",
        }
        return httpx.Response(
            200,
            json={
                "ok": True,
                "operation_id": "ph-mop-v1_BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB",
                "issue_source": "discord",
                "state": "FAILED",
                "client_action": "action_required",
                "failure_reason": "terminal_provider_failure",
                "attempt_count": 2,
                "current_attempt_index": 2,
                "auth_expires_at": "2026-12-01T01:00:00.000Z",
                "attempts": [],
                "delivery": None,
            },
        )

    client, _transport = _build_client(handler)

    result = await client.resume_managed_operation(
        ManagedOperationResumeRequest(
            operation_id="ph-mop-v1_BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB",
            installation_id="install-1",
            resume_token="token-2",
        )
    )

    assert result.succeeded is True
    assert result.operation_status == "FAILED"
    assert result.client_action == "action_required"
    assert result.failed_reason == "terminal_provider_failure"
    assert result.attempt is None
    assert result.delivery is None
    await client.close()


@pytest.mark.asyncio
async def test_managed_operation_status_malformed_payload_is_retryable_wait() -> None:
    from puripuly_heart.app.ports.broker_client import ManagedOperationStatusRequest

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"state": "BOGUS", "client_action": "wait"})

    client, _transport = _build_client(handler)

    result = await client.get_managed_operation_status(
        ManagedOperationStatusRequest(operation_id="ph-mop-v1_AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
    )

    assert result.succeeded is False
    assert result.operation_status == "CREATE_UNKNOWN"
    assert result.client_action == "wait"
    await client.close()


@pytest.mark.asyncio
async def test_managed_operation_status_not_found_maps_unknown_operation() -> None:
    from puripuly_heart.app.ports.broker_client import ManagedOperationStatusRequest

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            404,
            json={"ok": False, "code": "invalid_request", "message": "unknown operation"},
        )

    client, _transport = _build_client(handler)

    result = await client.get_managed_operation_status(
        ManagedOperationStatusRequest(operation_id="ph-mop-v1_AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
    )

    assert result.succeeded is False
    assert result.operation_status == "UNKNOWN_OPERATION"
    await client.close()


@pytest.mark.asyncio
async def test_managed_operation_gone_with_failed_body_reports_terminal() -> None:
    from puripuly_heart.app.ports.broker_client import ManagedOperationStatusRequest

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            410,
            json={
                "ok": True,
                "operation_id": "ph-mop-v1_AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
                "issue_source": "discord",
                "state": "FAILED",
                "client_action": "action_required",
                "failure_reason": "authorization_expired",
                "attempt_count": 0,
                "current_attempt_index": 0,
                "auth_expires_at": "2026-12-01T01:00:00.000Z",
                "attempts": [],
                "delivery": None,
            },
        )

    client, _transport = _build_client(handler)

    result = await client.get_managed_operation_status(
        ManagedOperationStatusRequest(operation_id="ph-mop-v1_AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
    )

    assert result.succeeded is True
    assert result.operation_status == "FAILED"
    assert result.failed_reason == "authorization_expired"
    await client.close()


@pytest.mark.asyncio
async def test_managed_operation_gone_with_error_envelope_reports_expired() -> None:
    from puripuly_heart.app.ports.broker_client import ManagedOperationStatusRequest

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            410,
            json={
                "ok": False,
                "code": "authorization_expired",
                "message": "operation authorization expired",
            },
        )

    client, _transport = _build_client(handler)

    result = await client.get_managed_operation_status(
        ManagedOperationStatusRequest(operation_id="ph-mop-v1_AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
    )

    assert result.succeeded is True
    assert result.operation_status == "FAILED"
    assert result.failed_reason == "authorization_expired"
    await client.close()


@pytest.mark.asyncio
async def test_managed_operation_status_transport_error_is_retryable_wait() -> None:
    from puripuly_heart.app.ports.broker_client import ManagedOperationStatusRequest

    def handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("broker unreachable")

    client, _transport = _build_client(handler)

    result = await client.get_managed_operation_status(
        ManagedOperationStatusRequest(operation_id="ph-mop-v1_AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
    )

    assert result.succeeded is False
    assert result.operation_status == "CREATE_UNKNOWN"
    assert result.client_action == "wait"
    await client.close()


def _delivery_pending_status_body(
    *,
    issue_source: object = "qq",
    operation_id: str = "ph-mop-v1_AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
) -> dict[str, object]:
    body: dict[str, object] = {
        "ok": True,
        "operation_id": operation_id,
        "state": "DELIVERY_PENDING",
        "client_action": "acknowledge_delivery",
        "failure_reason": None,
        "attempt_count": 1,
        "current_attempt_index": 1,
        "auth_expires_at": "2026-12-01T01:00:00.000Z",
        "attempts": [],
        "delivery": {
            "delivery_id": "delivery-1",
            "status": "pending",
            "expires_at": "2026-12-01T00:15:00.000Z",
        },
        "openrouter_api_key": "managed-openrouter-api-key",
        "managed_credential_ref": "managed-ref-1",
        "expires_at": "2026-12-01T00:00:00.000Z",
        "delivery_ack_required": True,
        "delivery_id": "delivery-1",
        "delivery_ack_token": "delivery-ack-token-1",
        "delivery_ack_expires_at": "2026-12-01T00:15:00.000Z",
    }
    if issue_source is not None:
        body["issue_source"] = issue_source
    return body


@pytest.mark.asyncio
async def test_status_managed_operation_uses_issue_source_for_delivery_ack() -> None:
    from puripuly_heart.app.ports.broker_client import ManagedOperationStatusRequest

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/v1/providers/openrouter/managed-operation/status"
        assert json.loads(request.content) == {
            "operation_id": "ph-mop-v1_AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
            "installation_id": "install-1",
            "resume_token": "token-1",
        }
        return httpx.Response(200, json=_delivery_pending_status_body(issue_source="qq"))

    client, _transport = _build_client(handler)

    result = await client.get_managed_operation_status(
        ManagedOperationStatusRequest(
            operation_id="ph-mop-v1_AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
            installation_id="install-1",
            resume_token="token-1",
            source="qq",
        )
    )

    assert result.succeeded is True
    assert result.operation_status == "DELIVERY_PENDING"
    assert result.delivery_ack is not None
    assert result.delivery_ack.source == "qq"
    assert result.delivery_ack.delivery_id == "delivery-1"
    assert result.delivery_ack.delivery_ack_token == "delivery-ack-token-1"
    assert "delivery-ack-token-1" not in repr(result.delivery_ack)
    await client.close()


@pytest.mark.asyncio
async def test_status_managed_operation_rejects_mismatched_issue_source() -> None:
    from puripuly_heart.app.ports.broker_client import ManagedOperationStatusRequest

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_delivery_pending_status_body(issue_source="discord"))

    client, _transport = _build_client(handler)

    result = await client.get_managed_operation_status(
        ManagedOperationStatusRequest(
            operation_id="ph-mop-v1_AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
            installation_id="install-1",
            resume_token="token-1",
            source="qq",
        )
    )

    assert result.succeeded is False
    assert result.operation_status == "CREATE_UNKNOWN"
    assert result.client_action == "wait"
    assert result.delivery_ack is None
    await client.close()


@pytest.mark.asyncio
async def test_status_managed_operation_rejects_unknown_issue_source() -> None:
    from puripuly_heart.app.ports.broker_client import ManagedOperationStatusRequest

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_delivery_pending_status_body(issue_source="wechat"))

    client, _transport = _build_client(handler)

    result = await client.get_managed_operation_status(
        ManagedOperationStatusRequest(
            operation_id="ph-mop-v1_AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
            installation_id="install-1",
            resume_token="token-1",
            source="qq",
        )
    )

    assert result.succeeded is False
    assert result.operation_status == "CREATE_UNKNOWN"
    assert result.client_action == "wait"
    assert result.delivery_ack is None
    await client.close()


@pytest.mark.asyncio
async def test_resume_managed_operation_falls_back_to_request_source() -> None:
    from puripuly_heart.app.ports.broker_client import ManagedOperationResumeRequest

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/v1/providers/openrouter/managed-operation/resume"
        assert json.loads(request.content) == {
            "operation_id": "ph-mop-v1_BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB",
            "installation_id": "install-1",
            "resume_token": "token-2",
        }
        return httpx.Response(
            200,
            json=_delivery_pending_status_body(
                issue_source=None,
                operation_id="ph-mop-v1_BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB",
            ),
        )

    client, _transport = _build_client(handler)

    result = await client.resume_managed_operation(
        ManagedOperationResumeRequest(
            operation_id="ph-mop-v1_BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB",
            installation_id="install-1",
            resume_token="token-2",
            source="qq",
        )
    )

    assert result.succeeded is True
    assert result.delivery_ack is not None
    assert result.delivery_ack.source == "qq"
    await client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "payload", "expected_code", "expected_class", "shape"),
    [
        (403, b"<html>private-denial</html>", "invalid_request", "terminal", "non_json"),
        (429, b"<html>private-limit</html>", "rate_limited", "retryable", "non_json"),
        (
            429,
            {"ok": False, "code": "rate_limited", "message": "limited"},
            "rate_limited",
            "retryable",
            "flat_error",
        ),
        (
            429,
            {"error": {"code": "rate_limited", "class": "retryable", "message": "limited"}},
            "rate_limited",
            "retryable",
            "public_error_envelope",
        ),
        (200, b"{private-malformed", "trial_unavailable", "retryable", "non_json"),
        (200, [], "trial_unavailable", "retryable", "non_object"),
        (200, {}, "trial_unavailable", "retryable", "malformed_success"),
    ],
)
async def test_discord_and_qq_response_failures_keep_safe_outcome_metadata(
    status, payload, expected_code, expected_class, shape
):
    def response(request):
        kwargs = {"content": payload} if isinstance(payload, bytes) else {"json": payload}
        return httpx.Response(
            status,
            headers={"Retry-After": "7"},
            **kwargs,
            extensions={
                "connection_diagnostics": {
                    "transport": "https",
                    "tls_source": "explicit_file",
                    "tls_backend": "openssl",
                    "proxy_source": "system",
                    "proxy_url": "private-proxy-url",
                }
            },
        )

    client, _ = _build_client(response)
    try:
        with pytest.raises(ManagedOpenRouterReleaseError) as caught:
            await client.issue_discord_managed_key({"code": "private-oauth-code"})
        error = caught.value
        assert (error.code, error.error_class) == (expected_code, expected_class)
        diagnostics = error.failure_diagnostics
        assert diagnostics.status_code == status
        assert diagnostics.retry_after_ms == 7000
        assert diagnostics.fields["dispatch_certainty"] == "response_received"
        assert diagnostics.fields["response_shape"] == shape
        assert diagnostics.fields["stage"] == "protocol_parse"
        assert diagnostics.fields["proxy_source"] == "system"
        assert diagnostics.fields["tls_source"] == "explicit_file"
        qq = await client.assert_qq_managed_identity(
            QqManagedAssertionRequest("private-identity", "a" * 64, "2026-01-01", {})
        )
        assert not qq.succeeded
        assert qq.diagnostics.status_code == status
        assert qq.diagnostics.fields["response_shape"] == shape
        assert qq.diagnostics.fields["dispatch_certainty"] == "response_received"
        assert qq.diagnostics.fields["proxy_source"] == "system"
        assert qq.diagnostics.fields["tls_source"] == "explicit_file"
        for secret in ("private-denial", "private-limit", "private-malformed", "private-proxy-url"):
            assert secret not in repr(diagnostics)
            assert secret not in repr(qq)
    finally:
        await client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("retry", [-1, True, 10**20, 0, 7000])
async def test_flat_rate_limit_retry_is_bounded_and_status_recovery_can_succeed(retry):
    from puripuly_heart.app.ports.broker_client import ManagedOperationStatusRequest

    calls = []

    def response(request):
        calls.append(json.loads(request.content))
        if len(calls) == 1:
            return httpx.Response(
                429,
                headers={"Retry-After": "3"},
                json={
                    "ok": False,
                    "code": "rate_limited",
                    "message": "limited",
                    "retry_after_ms": retry,
                },
            )
        return httpx.Response(200, json={"state": "ACTIVE", "client_action": "wait"})

    client, _ = _build_client(response)
    request = ManagedOperationStatusRequest(
        "ph-mop-v1_AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA", "installation", "private-resume", "qq"
    )
    try:
        failure = await client.get_managed_operation_status(request)
        assert not failure.succeeded
        assert failure.operation_status == "CREATE_UNKNOWN"
        assert failure.client_action == "wait"
        assert failure.diagnostics.status_code == 429
        expected = min(retry, 86400000) if type(retry) is int and retry >= 0 else 3000
        assert failure.diagnostics.retry_after_ms == expected
        assert failure.diagnostics.fields["broker_code"] == "rate_limited"
        success = await client.get_managed_operation_status(request)
        assert success.succeeded
        assert success.operation_status == "ACTIVE"
        assert calls[0] == calls[1]
    finally:
        await client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("status,state", [(409, "RETRY_READY"), (200, "FAILED")])
async def test_legacy_resume_guidance_stops_impossible_resume_without_losing_state(status, state):
    from puripuly_heart.app.ports.broker_client import ManagedOperationResumeRequest

    client, _ = _build_client(
        lambda _: httpx.Response(
            status,
            json={
                "ok": False,
                "state": state,
                "client_action": "retry_authorized",
                "failure_reason": "terminal_provider_failure" if state == "FAILED" else None,
                "code": "resume_issuance_unavailable",
                "message": "restart authorization",
            },
        )
    )
    try:
        result = await client.resume_managed_operation(
            ManagedOperationResumeRequest("ph-mop-v1_AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
        )
        assert result.succeeded
        assert result.operation_status == state
        assert result.client_action == "action_required"
        assert result.diagnostics.status_code == status
        assert result.diagnostics.fields["broker_code"] == "resume_issuance_unavailable"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_retry_after_http_date_is_retained_and_bounded():
    from datetime import UTC, datetime, timedelta
    from email.utils import format_datetime

    client, _ = _build_client(
        lambda _: httpx.Response(
            429,
            text="limited",
            headers={"Retry-After": format_datetime(datetime.now(UTC) + timedelta(seconds=30))},
        )
    )
    try:
        with pytest.raises(ManagedOpenRouterReleaseError) as caught:
            await client.issue_discord_managed_key({})
        assert 25000 <= caught.value.retry_after_ms <= 30000
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_local_ca_configuration_is_actionable_before_broker_dispatch(monkeypatch, tmp_path):
    from puripuly_heart.core.openrouter.managed_openrouter_broker_client import (
        HttpManagedOpenRouterBrokerClient,
    )

    monkeypatch.setenv("SSL_CERT_FILE", str(tmp_path / "private-ca-path"))
    client = HttpManagedOpenRouterBrokerClient("http://127.0.0.1:9")
    try:
        with pytest.raises(ManagedOpenRouterReleaseError) as caught:
            await client.issue_discord_managed_key({})
        error = caught.value
        assert error.error_class == "terminal"
        assert error.subcode == "invalid_tls_configuration"
        assert error.http_status is None
        assert error.failure_diagnostics.fields["dispatch_certainty"] == "not_dispatched"
        assert (
            error.failure_diagnostics.fields["configuration_error"] == "invalid_tls_configuration"
        )
        assert "private-ca-path" not in repr(error.failure_diagnostics)
        qq = await client.assert_qq_managed_identity(
            QqManagedAssertionRequest("private-identity", "a" * 64, "2026-01-01", {})
        )
        assert qq.diagnostics.fields["dispatch_certainty"] == "not_dispatched"
        assert qq.diagnostics.fields["configuration_error"] == "invalid_tls_configuration"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_removed_referral_input_is_rejected_without_dispatch():
    calls = []
    client, _ = _build_client(lambda request: calls.append(request))
    try:
        with pytest.raises(TypeError):
            await client.start_discord_oauth(
                installation_id="installation",
                device_public_key="device",
                redirect_uri="http://127.0.0.1:62187/discord/callback",
                app_version="test",
                referral_id="removed",
            )
        with pytest.raises(TypeError):
            QqManagedAssertionRequest("identity", "a" * 64, "2026-01-01", {}, referral_id="removed")
        assert calls == []
        assert not hasattr(client, "get_qq_managed_status")
        assert not hasattr(client, "get_trial_status")
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_denied_status_observation_does_not_invent_terminal_remote_state():
    from puripuly_heart.app.ports.broker_client import ManagedOperationStatusRequest

    client, _ = _build_client(lambda _: httpx.Response(403, text="<html>denied</html>"))
    try:
        result = await client.get_managed_operation_status(
            ManagedOperationStatusRequest("ph-mop-v1_AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
        )
        assert not result.succeeded
        assert result.operation_status == "CREATE_UNKNOWN"
        assert result.client_action == "action_required"
        assert result.diagnostics.status_code == 403
        assert result.diagnostics.fields["dispatch_certainty"] == "response_received"
    finally:
        await client.close()


def test_safe_transport_metadata_and_identity_survive_diagnostic_projection():
    from uuid import uuid4

    from puripuly_heart.core.diagnostics import normalize_diagnostic
    from puripuly_heart.core.openrouter.managed_openrouter_broker_client import (
        _parse_error_response,
        _qq_assertion_failure_from_error,
    )

    error = _parse_error_response(
        httpx.Response(
            429,
            headers={"Retry-After": "7"},
            text="<html>private-response</html>",
            extensions={
                "connection_diagnostics": {
                    "transport": "https",
                    "tls_source": "windows",
                    "tls_backend": "native",
                    "proxy_source": "system",
                    "private_proxy_url": "private-route",
                }
            },
        ),
        operation="qq_assert",
    )
    result = _qq_assertion_failure_from_error(error)
    assert result.diagnostics.diagnostic_id == error.failure_diagnostics.diagnostic_id
    record = normalize_diagnostic(
        result.diagnostics, instance_id=str(uuid4()), severity="error", disposition="failed"
    )
    assert record["diagnostic_id"] == result.diagnostics.diagnostic_id
    assert record["status_code"] == 429
    assert record["retry_after_ms"] == 7000
    assert record["stage"] == "protocol_parse"
    assert record["fields"]["proxy_source"] == "system"
    assert record["fields"]["tls_source"] == "windows"
    assert record["fields"]["dispatch_certainty"] == "response_received"
    assert record["fields"]["response_shape"] == "non_json"
    assert "private-" not in repr(record)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mismatch",
    [
        "operation",
        "installation",
        "source",
        "delivery",
        "credential",
        "malformed_source",
        "malformed_attempt",
        "malformed_delivery",
    ],
)
async def test_operation_response_binding_rejects_wrong_or_malformed_delivery(mismatch):
    from puripuly_heart.app.ports.broker_client import ManagedOperationStatusRequest

    payload = _delivery_pending_status_body()
    if mismatch == "operation":
        payload["operation_id"] = "ph-mop-v1_BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB"
    elif mismatch == "installation":
        payload["installation_id"] = "another-installation"
    elif mismatch == "source":
        payload["issue_source"] = "discord"
    elif mismatch == "delivery":
        payload["delivery"]["delivery_id"] = "another-delivery"
    elif mismatch == "credential":
        payload["attempts"] = [{"managed_credential_ref": "another-reference"}]
    elif mismatch == "malformed_source":
        payload["issue_source"] = ["private-shape"]
    elif mismatch == "malformed_attempt":
        payload["attempts"] = [{"outcome": ["private-shape"]}]
    else:
        payload["delivery"]["status"] = ["private-shape"]
    responses = iter([payload, _delivery_pending_status_body()])
    client, _ = _build_client(lambda _: httpx.Response(200, json=next(responses)))
    request = ManagedOperationStatusRequest(
        _delivery_pending_status_body()["operation_id"], "installation", "private-resume", "qq"
    )
    try:
        failed = await client.get_managed_operation_status(request)
        assert not failed.succeeded
        assert failed.operation_status == "CREATE_UNKNOWN"
        assert failed.managed_secret_key is None
        assert failed.delivery_ack is None
        assert failed.diagnostics.status_code == 200
        assert failed.diagnostics.fields["response_shape"] == "malformed_success"
        assert failed.diagnostics.fields["dispatch_certainty"] == "response_received"
        assert "private-shape" not in repr(failed.diagnostics)
        corrected = await client.get_managed_operation_status(request)
        assert corrected.succeeded
        assert corrected.delivery_ack.source == "qq"
        assert corrected.delivery_ack.managed_credential_ref == corrected.managed_credential_ref
        assert corrected.delivery_ack.delivery_id == corrected.delivery.delivery_id
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_malformed_qq_and_ack_enum_shapes_are_observable_failures():
    client, _ = _build_client(
        lambda _: httpx.Response(200, json={"ok": True, "status": ["private-shape"]})
    )
    try:
        qq = await client.assert_qq_managed_identity(
            QqManagedAssertionRequest("private-identity", "a" * 64, "2026-01-01", {})
        )
        ack = await client.acknowledge_managed_key_delivery(
            ManagedKeyDeliveryAckRequest("delivery", "ref", "private-ack")
        )
        assert not qq.succeeded
        assert not ack.succeeded
        assert qq.diagnostics.status_code == ack.diagnostics.status_code == 200
        assert qq.diagnostics.fields["response_shape"] == "malformed_success"
        assert ack.diagnostics.fields["response_shape"] == "malformed_success"
        assert "private-shape" not in repr(qq.diagnostics)
        assert "private-shape" not in repr(ack.diagnostics)
    finally:
        await client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("resume", [False, True])
@pytest.mark.parametrize("source", ["discord", "qq"])
async def test_minimal_authorization_expired_fallback_is_authoritative_without_message(
    resume, source
):
    from puripuly_heart.app.ports.broker_client import (
        ManagedOperationResumeRequest,
        ManagedOperationStatusRequest,
    )

    client, _ = _build_client(lambda _: httpx.Response(410, json={"code": "authorization_expired"}))
    request_type = ManagedOperationResumeRequest if resume else ManagedOperationStatusRequest
    method = client.resume_managed_operation if resume else client.get_managed_operation_status
    try:
        result = await method(
            request_type(
                "ph-mop-v1_AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
                "installation",
                "private-resume",
                source,
            )
        )
        assert result.succeeded
        assert result.operation_status == "FAILED"
        assert result.client_action == "action_required"
        assert result.failed_reason == "authorization_expired"
        assert result.diagnostics.status_code == 410
        assert result.diagnostics.fields["broker_code"] == "authorization_expired"
    finally:
        await client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["discord", "qq"])
@pytest.mark.parametrize("field", ["operation_id", "installation_id", "issue_source"])
async def test_initial_issue_rejects_wrong_echo_but_accepts_correct_binding(source, field):
    operation_id = "ph-mop-v1_AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    payload = {
        "status": "issued",
        "operation_id": operation_id,
        "installation_id": "installation",
        "issue_source": source,
        "qq_subject_ref": "subject",
        "openrouter_api_key": "synthetic-key",
        "managed_credential_ref": "managed-ref",
        "delivery_ack_required": True,
        "delivery_id": "delivery",
        "delivery_ack_token": "synthetic-ack",
    }
    wrong = {**payload, field: "wrong-binding"}
    responses = iter([wrong, payload])
    client, _ = _build_client(lambda _: httpx.Response(200, json=next(responses)))
    try:
        if source == "discord":
            request = {"installation_id": "installation", "operation_id": operation_id}
            with pytest.raises(ManagedOpenRouterReleaseError) as caught:
                await client.issue_discord_managed_key(request)
            assert caught.value.failure_diagnostics.fields["response_shape"] == "malformed_success"
            correct = await client.issue_discord_managed_key(request)
            assert correct.openrouter_api_key == "synthetic-key"
            assert correct.managed_credential_ref == "managed-ref"
        else:
            request = QqManagedAssertionRequest(
                "identity",
                "a" * 64,
                "2026-01-01",
                {},
                installation_id="installation",
                operation_id=operation_id,
            )
            rejected = await client.assert_qq_managed_identity(request)
            assert not rejected.succeeded
            assert rejected.managed_secret_key is None
            assert rejected.diagnostics.fields["response_shape"] == "malformed_success"
            correct = await client.assert_qq_managed_identity(request)
            assert correct.succeeded
            assert correct.managed_secret_key == "synthetic-key"
            assert correct.delivery_ack.source == "qq"
            assert correct.entitlement.managed_credential_ref == "managed-ref"
    finally:
        await client.close()
