from __future__ import annotations

import base64
import hashlib
import json
import time
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa

from puripuly_heart.core.chatgpt.oauth import (
    CHATGPT_AGENT_NAME,
    CHATGPT_API_RESOURCE,
    CHATGPT_DYNAMIC_CLIENT_ID,
    CHATGPT_ISSUER,
    CHATGPT_JWKS_URL,
    CHATGPT_PLAN_SCOPE,
    ChatGptAuthError,
    ChatGptIdTokenVerifier,
    ChatGptReauthRequired,
    build_authorization_request,
    refresh_access_token,
    token_set_from_response,
)


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


class _Signer:
    def __init__(self, kid: str = "kid-1") -> None:
        self.kid = kid
        self.key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

    def jwk(self) -> dict[str, str]:
        numbers = self.key.public_key().public_numbers()
        return {
            "kty": "RSA",
            "alg": "RS256",
            "kid": self.kid,
            "n": _b64url(numbers.n.to_bytes((numbers.n.bit_length() + 7) // 8, "big")),
            "e": _b64url(numbers.e.to_bytes((numbers.e.bit_length() + 7) // 8, "big")),
        }

    def token(self, claims: dict[str, object], *, alg: str = "RS256") -> str:
        header = _b64url(json.dumps({"alg": alg, "kid": self.kid}).encode())
        payload = _b64url(json.dumps(claims).encode())
        signed = f"{header}.{payload}".encode("ascii")
        signature = self.key.sign(signed, padding.PKCS1v15(), hashes.SHA256())
        return f"{header}.{payload}.{_b64url(signature)}"


def _jwks_client(signer: _Signer) -> httpx.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == CHATGPT_JWKS_URL
        return httpx.Response(200, json={"keys": [signer.jwk()]})

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _claims(**overrides: object) -> dict[str, object]:
    claims: dict[str, object] = {
        "iss": CHATGPT_ISSUER,
        "aud": "oaiapp_client",
        "sub": "user-sub",
        "email": "user@example.com",
        "nonce": "nonce-1",
        "exp": int(time.time()) + 3600,
    }
    claims.update(overrides)
    return claims


def test_first_authorization_registers_dynamic_client_with_host_and_plan_scope() -> None:
    request = build_authorization_request(
        redirect_uri="http://127.0.0.1:1455/auth/callback",
        host_id="urn:uuid:host",
        client_id=None,
    )
    params = {k: v[0] for k, v in parse_qs(urlsplit(request.url).query).items()}

    assert params["client_id"] == CHATGPT_DYNAMIC_CLIENT_ID
    assert params["agent_name_hint"] == CHATGPT_AGENT_NAME
    assert params["ext_agent_host_id"] == "urn:uuid:host"
    assert params["resource"] == CHATGPT_API_RESOURCE
    assert CHATGPT_PLAN_SCOPE in params["scope"].split()
    assert "offline_access" in params["scope"].split()
    assert params["state"] == request.state
    assert params["nonce"] == request.nonce
    assert params["code_challenge_method"] == "S256"
    expected_challenge = _b64url(hashlib.sha256(request.code_verifier.encode()).digest())
    assert params["code_challenge"] == expected_challenge


def test_reauthorization_reuses_issued_client_without_agent_name_hint() -> None:
    request = build_authorization_request(
        redirect_uri="http://127.0.0.1:1456/auth/callback",
        host_id="urn:uuid:host",
        client_id="oaiapp_saved",
        login_hint="user@example.com",
    )
    params = {k: v[0] for k, v in parse_qs(urlsplit(request.url).query).items()}

    assert params["client_id"] == "oaiapp_saved"
    assert "agent_name_hint" not in params
    assert params["login_hint"] == "user@example.com"
    assert params["ext_agent_host_id"] == "urn:uuid:host"


def test_token_set_keeps_previous_refresh_token_and_reports_plan_scope() -> None:
    tokens = token_set_from_response(
        {"access_token": "a", "expires_in": 3600, "scope": f"openid {CHATGPT_PLAN_SCOPE}"},
        previous_refresh_token="r-old",
        now=100.0,
    )
    assert tokens.refresh_token == "r-old"
    assert tokens.expires_at == 3700.0
    assert tokens.plan_usage_granted is True
    assert (
        token_set_from_response(
            {"access_token": "a", "refresh_token": "r", "scope": "openid"}
        ).plan_usage_granted
        is False
    )


async def test_id_token_verifier_accepts_valid_signature_and_claims() -> None:
    signer = _Signer()
    async with _jwks_client(signer) as http:
        identity = await ChatGptIdTokenVerifier(http).verify(
            signer.token(_claims()), client_id="oaiapp_client", nonce="nonce-1"
        )
    assert identity.subject == "user-sub"
    assert identity.email == "user@example.com"


@pytest.mark.parametrize(
    "overrides",
    [
        {"nonce": "other"},
        {"aud": "oaiapp_other"},
        {"iss": "https://example.com"},
        {"exp": int(time.time()) - 3600},
        {"sub": ""},
    ],
)
async def test_id_token_verifier_rejects_mismatched_claims(overrides: dict[str, object]) -> None:
    signer = _Signer()
    async with _jwks_client(signer) as http:
        with pytest.raises(ChatGptAuthError) as exc_info:
            await ChatGptIdTokenVerifier(http).verify(
                signer.token(_claims(**overrides)), client_id="oaiapp_client", nonce="nonce-1"
            )
    assert exc_info.value.code == "invalid_id_token"


async def test_id_token_verifier_rejects_signature_from_unknown_key() -> None:
    published = _Signer("kid-1")
    forger = _Signer("kid-1")
    async with _jwks_client(published) as http:
        with pytest.raises(ChatGptAuthError):
            await ChatGptIdTokenVerifier(http).verify(
                forger.token(_claims()), client_id="oaiapp_client", nonce="nonce-1"
            )


@pytest.mark.parametrize(
    ("status", "error", "expected"),
    [
        (400, "invalid_grant", ChatGptReauthRequired),
        (400, "refresh_token_reused", ChatGptReauthRequired),
        (401, "invalid_client", ChatGptReauthRequired),
        (503, "temporarily_unavailable", ChatGptAuthError),
    ],
)
async def test_refresh_classifies_terminal_and_transient_failures(
    status: int, error: str, expected: type[Exception]
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json={"error": error})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        with pytest.raises(expected) as exc_info:
            await refresh_access_token(http, client_id="oaiapp_client", refresh_token="r")
    assert type(exc_info.value) is expected
    assert exc_info.value.code == error


async def test_exchange_malformed_success_object_retains_received_response():
    from puripuly_heart.core.chatgpt.oauth import exchange_authorization_code

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _request: httpx.Response(200, json={}))
    ) as client:
        with pytest.raises(ChatGptAuthError) as raised:
            await exchange_authorization_code(
                client,
                client_id="synthetic-client",
                code="synthetic-code",
                code_verifier="synthetic-verifier",
                redirect_uri="http://127.0.0.1/synthetic",
            )
    assert raised.value.diagnostics.category == "invalid_response"
    assert raised.value.diagnostics.status_code == 200
    assert raised.value.diagnostics.fields["response_shape"] == "malformed_success"
    assert "synthetic" not in str(raised.value.diagnostics)
