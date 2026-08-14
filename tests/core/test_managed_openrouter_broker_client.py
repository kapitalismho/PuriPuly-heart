from __future__ import annotations

import inspect
import json
from collections.abc import Callable

import httpx
import pytest
from puripuly_heart.core.managed_openrouter_release import (
    ManagedOpenRouterChallengeSuccess,
    ManagedOpenRouterDiscordStartSuccess,
    ManagedOpenRouterFingerprintSalt,
    ManagedOpenRouterIssueSuccess,
    ManagedOpenRouterReleaseError,
    ManagedOpenRouterVerifySuccess,
    TalkTogetherPassStatus,
)

from puripuly_heart.app.ports.broker_client import (
    ManagedKeyDeliveryAckRequest,
    QqManagedAssertionRequest,
)


class TrackingTransport(httpx.AsyncBaseTransport):
    def __init__(self, handler: Callable[[httpx.Request], httpx.Response]) -> None:
        self._handler = handler
        self.closed = False

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        return self._handler(request)

    async def aclose(self) -> None:
        self.closed = True


def _build_client(
    handler: Callable[[httpx.Request], httpx.Response],
    *,
    base_url: str = "https://broker.example.test",
) -> tuple[object, TrackingTransport]:
    from puripuly_heart.core.managed_openrouter_broker_client import (
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
@pytest.mark.parametrize(
    ("raw_referral_id", "expected_referral_id"),
    [
        ("  7kq9m2  ", "7KQ9M2"),
        ("  invalid friend id  ", "INVALID FRIEND ID"),
    ],
)
async def test_start_discord_oauth_forwards_trimmed_uppercase_non_empty_referral_id(
    raw_referral_id: str,
    expected_referral_id: str,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body == {
            "installation_id": "install-discord-123",
            "device_public_key": "device-public-key-123",
            "redirect_uri": "http://127.0.0.1:62187/discord/callback",
            "app_version": "2.0.0",
            "referral_id": expected_referral_id,
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

    signature = inspect.signature(client.start_discord_oauth)
    assert "referral_id" in signature.parameters
    await client.start_discord_oauth(
        installation_id="install-discord-123",
        device_public_key="device-public-key-123",
        redirect_uri="http://127.0.0.1:62187/discord/callback",
        app_version="2.0.0",
        referral_id=raw_referral_id,
    )

    await client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("empty_referral_id", [None, "", "   "])
async def test_start_discord_oauth_omits_empty_referral_id(
    empty_referral_id: str | None,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body == {
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

    signature = inspect.signature(client.start_discord_oauth)
    assert "referral_id" in signature.parameters
    await client.start_discord_oauth(
        installation_id="install-discord-123",
        device_public_key="device-public-key-123",
        redirect_uri="http://127.0.0.1:62187/discord/callback",
        app_version="2.0.0",
        referral_id=empty_referral_id,
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
async def test_issue_discord_managed_key_parses_referral_success_hints() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "openrouter_api_key": "managed-openrouter-api-key",
                "managed_credential_ref": "managed-credential-ref-123",
                "expires_at": "2026-07-30T06:00:00.000Z",
                "openrouter_user_id": " user-123 ",
                "referral_bonus_applied": True,
                "referral_id": " 7kq9m2 ",
            },
        )

    client, _transport = _build_client(handler)

    result = await client.issue_discord_managed_key({"code": "discord-oauth-code-123"})

    assert result == ManagedOpenRouterIssueSuccess(
        openrouter_api_key="managed-openrouter-api-key",
        managed_credential_ref="managed-credential-ref-123",
        expires_at="2026-07-30T06:00:00.000Z",
        openrouter_user_id="user-123",
        referral_bonus_applied=True,
        referral_id="7KQ9M2",
    )
    await client.close()


@pytest.mark.asyncio
async def test_issue_discord_managed_key_parses_talk_together_pass_status() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "openrouter_api_key": "managed-openrouter-api-key",
                "referral_id": " 7kq9m2 ",
                "talk_together_pass": {
                    "pass_id": "7KQ9M2",
                    "invite_count": 1,
                    "invite_limit": 5,
                    "bonus_translations_per_friend": 200,
                },
            },
        )

    client, _transport = _build_client(handler)

    result = await client.issue_discord_managed_key({"code": "discord-oauth-code-123"})

    assert result.referral_id == "7KQ9M2"
    assert result.pass_status == TalkTogetherPassStatus(
        pass_id="7KQ9M2",
        invite_count=1,
        invite_limit=5,
        bonus_translations_per_friend=200,
    )
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
                "referral_bonus_applied": True,
                "referral_id": " 7kq9m2 ",
                "talk_together_pass": {
                    "pass_id": "7KQ9M2",
                    "invite_count": 2,
                    "invite_limit": 5,
                    "bonus_translations_per_friend": 200,
                },
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
    assert result.referral_bonus_applied is True
    assert result.referral_id == "7KQ9M2"
    assert result.pass_status == TalkTogetherPassStatus(
        pass_id="7KQ9M2",
        invite_count=2,
        invite_limit=5,
        bonus_translations_per_friend=200,
    )
    await client.close()


@pytest.mark.asyncio
async def test_acknowledge_managed_key_delivery_maps_broker_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/v1/providers/openrouter/managed-key-delivery/ack"
        return httpx.Response(
            409,
            json={
                "error": {
                    "code": "invalid_request",
                    "class": "terminal",
                    "subcode": "delivery_ack_invalid",
                    "message": "delivery ACK failed",
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
    assert result.status == "delivery_ack_invalid"
    assert result.diagnostics is not None
    assert result.diagnostics.code == "managed_key_delivery_ack_error"
    assert result.diagnostics.fields["broker_subcode"] == "delivery_ack_invalid"
    await client.close()


@pytest.mark.asyncio
async def test_record_translation_success_day_posts_telemetry_payload() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/v1/telemetry/translation-success-day"
        assert json.loads(request.content) == {
            "signal": "translation_success_day",
            "telemetry_identifier": "anon-telemetry-id-123456",
            "active_date_utc": "2026-07-03",
        }
        return httpx.Response(200, json={"ok": True})

    client, _transport = _build_client(handler)

    result = await client.record_translation_success_day(
        "anon-telemetry-id-123456",
        "2026-07-03",
    )

    assert result is True
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
    ("extra_payload", "expected_referral_bonus_applied", "expected_referral_id"),
    [
        ({}, False, None),
        ({"referral_bonus_applied": None, "referral_id": None}, False, None),
        ({"referral_bonus_applied": False, "referral_id": " 7kq9m2 "}, False, "7KQ9M2"),
        ({"referral_bonus_applied": "true", "referral_id": "ABC120"}, False, None),
        ({"referral_bonus_applied": 1, "referral_id": 12345}, False, None),
    ],
)
async def test_issue_discord_managed_key_tolerates_referral_hint_compatibility_fields(
    extra_payload: dict[str, object],
    expected_referral_bonus_applied: bool,
    expected_referral_id: str | None,
) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "openrouter_api_key": "managed-openrouter-api-key",
                "managed_credential_ref": "managed-credential-ref-123",
                **extra_payload,
            },
        )

    client, _transport = _build_client(handler)

    result = await client.issue_discord_managed_key({"code": "discord-oauth-code-123"})

    assert result == ManagedOpenRouterIssueSuccess(
        openrouter_api_key="managed-openrouter-api-key",
        managed_credential_ref="managed-credential-ref-123",
        referral_bonus_applied=expected_referral_bonus_applied,
        referral_id=expected_referral_id,
    )
    await client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("extra_payload", "expected_referral_id"),
    [
        ({"referral_id": " 7kq9m2 "}, "7KQ9M2"),
        ({}, None),
        ({"referral_id": None}, None),
        ({"referral_id": "ABC120"}, None),
        ({"referral_id": 12345}, None),
    ],
)
async def test_get_trial_status_tolerates_optional_referral_id_compatibility_fields(
    extra_payload: dict[str, object],
    expected_referral_id: str | None,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/v1/trial/status"
        assert request.url.params.get("installation_id") == "install-discord-123"
        assert request.headers["X-Puripuly-Timestamp"] == "2026-04-30T06:00:30.000Z"
        assert request.headers["X-Puripuly-Signature"] == "signature-123"
        return httpx.Response(
            200,
            json={
                "managed_state": {
                    "lifecycle": "active",
                    "managed_availability": True,
                },
                "current_entitlement": None,
                "onboarding_eligibility": {
                    "eligible": False,
                    "reason": "active",
                    "requires_discord_oauth": False,
                },
                **extra_payload,
            },
        )

    client, _transport = _build_client(handler)

    get_trial_status = getattr(client, "get_trial_status", None)
    assert callable(get_trial_status)
    result = await get_trial_status(
        installation_id="install-discord-123",
        timestamp="2026-04-30T06:00:30.000Z",
        signature="signature-123",
    )

    assert getattr(result, "referral_id", None) == expected_referral_id
    await client.close()


@pytest.mark.asyncio
async def test_get_trial_status_parses_talk_together_pass_status() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "managed_state": {"lifecycle": "active", "managed_availability": True},
                "current_entitlement": None,
                "onboarding_eligibility": {
                    "eligible": False,
                    "reason": "active",
                    "requires_discord_oauth": False,
                },
                "referral_id": "7KQ9M2",
                "talk_together_pass": {
                    "pass_id": "7KQ9M2",
                    "invite_count": 2,
                    "invite_limit": 5,
                    "bonus_translations_per_friend": 200,
                },
            },
        )

    client, _transport = _build_client(handler)

    result = await client.get_trial_status(
        installation_id="install-discord-123",
        timestamp="2026-04-30T06:00:30.000Z",
        signature="signature-123",
    )

    assert result.referral_id == "7KQ9M2"
    assert result.pass_status == TalkTogetherPassStatus(
        pass_id="7KQ9M2",
        invite_count=2,
        invite_limit=5,
        bonus_translations_per_friend=200,
    )
    await client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "talk_together_pass",
    [
        None,
        {},
        {"pass_id": "ABC120", "invite_count": 1, "invite_limit": 5},
        {"pass_id": "8H3J4N", "invite_count": 1, "invite_limit": 5},
        {"pass_id": "7KQ9M2", "invite_count": True, "invite_limit": 5},
        {"pass_id": "7KQ9M2", "invite_count": 1.5, "invite_limit": 5},
        {"pass_id": "7KQ9M2", "invite_count": "1", "invite_limit": 5},
        {"pass_id": "7KQ9M2", "invite_count": None, "invite_limit": 5},
        {"pass_id": "7KQ9M2", "invite_count": -1, "invite_limit": 5},
        {"pass_id": "7KQ9M2", "invite_count": 1, "invite_limit": 0},
        {"pass_id": "7KQ9M2", "invite_count": 2**63, "invite_limit": 5},
    ],
)
async def test_talk_together_pass_malformed_required_fields_are_absent(
    talk_together_pass: object,
) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        payload: dict[str, object] = {
            "openrouter_api_key": "managed-openrouter-api-key",
            "referral_id": "7KQ9M2",
        }
        if talk_together_pass is not None:
            payload["talk_together_pass"] = talk_together_pass
        return httpx.Response(200, json=payload)

    client, _transport = _build_client(handler)

    result = await client.issue_discord_managed_key({"code": "discord-oauth-code-123"})

    assert result.referral_id == "7KQ9M2"
    assert result.pass_status is None
    await client.close()


@pytest.mark.asyncio
async def test_talk_together_pass_defaults_malformed_bonus_only() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "openrouter_api_key": "managed-openrouter-api-key",
                "referral_id": "7KQ9M2",
                "talk_together_pass": {
                    "pass_id": "7KQ9M2",
                    "invite_count": 1,
                    "invite_limit": 5,
                    "bonus_translations_per_friend": "200",
                },
            },
        )

    client, _transport = _build_client(handler)

    result = await client.issue_discord_managed_key({"code": "discord-oauth-code-123"})

    assert result.pass_status == TalkTogetherPassStatus(
        pass_id="7KQ9M2",
        invite_count=1,
        invite_limit=5,
        bonus_translations_per_friend=200,
    )
    await client.close()


@pytest.mark.asyncio
async def test_talk_together_pass_defaults_missing_bonus_only() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "openrouter_api_key": "managed-openrouter-api-key",
                "referral_id": "7KQ9M2",
                "talk_together_pass": {
                    "pass_id": "7KQ9M2",
                    "invite_count": 1,
                    "invite_limit": 5,
                },
            },
        )

    client, _transport = _build_client(handler)

    result = await client.issue_discord_managed_key({"code": "discord-oauth-code-123"})

    assert result.pass_status == TalkTogetherPassStatus(
        pass_id="7KQ9M2",
        invite_count=1,
        invite_limit=5,
        bonus_translations_per_friend=200,
    )
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
