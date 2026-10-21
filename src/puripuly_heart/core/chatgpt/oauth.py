from __future__ import annotations

import base64
import hashlib
import json
import secrets
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Final
from urllib.parse import urlencode

import httpx
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa

from puripuly_heart.core.messages import ErrorDiagnostics

CHATGPT_ISSUER: Final = "https://auth.openai.com"
CHATGPT_AUTHORIZE_URL: Final = f"{CHATGPT_ISSUER}/api/accounts/authorize"
CHATGPT_TOKEN_URL: Final = f"{CHATGPT_ISSUER}/api/accounts/oauth/token"
CHATGPT_REVOKE_URL: Final = f"{CHATGPT_ISSUER}/api/accounts/oauth/revoke"
CHATGPT_JWKS_URL: Final = f"{CHATGPT_ISSUER}/.well-known/jwks.json"
CHATGPT_API_RESOURCE: Final = "https://api.openai.com/v1"
CHATGPT_DYNAMIC_CLIENT_ID: Final = "dynamic_agent_client"
CHATGPT_PLAN_SCOPE: Final = "chatgpt.tokens.use.direct"
CHATGPT_SCOPES: Final[tuple[str, ...]] = (
    "openid",
    "profile",
    "email",
    "offline_access",
    "resource.invoke",
    CHATGPT_PLAN_SCOPE,
)
CHATGPT_AGENT_NAME: Final = "PuriPuly Heart"

_ID_TOKEN_LEEWAY_S: Final = 120
_TERMINAL_REFRESH_ERRORS: Final = frozenset(
    {
        "invalid_grant",
        "invalid_refresh_token",
        "token_expired",
        "refresh_token_expired",
        "refresh_token_invalidated",
        "refresh_token_reused",
        "invalid_client",
    }
)


class ChatGptAuthError(RuntimeError):
    diagnostic_provider = "chatgpt"

    def __init__(
        self,
        code: str,
        *,
        status_code: int | None = None,
        diagnostics: ErrorDiagnostics | None = None,
        secondary_diagnostics: tuple[ErrorDiagnostics, ...] = (),
        secondary_omitted_count: int = 0,
    ) -> None:
        super().__init__(f"ChatGPT authorization failed ({code})")
        self.code = code
        self.status_code = status_code
        self.diagnostics = diagnostics
        self.secondary_diagnostics = secondary_diagnostics[:4]
        self.secondary_omitted_count = secondary_omitted_count + max(
            0, len(secondary_diagnostics) - 4
        )


class ChatGptReauthRequired(ChatGptAuthError):
    chatgpt_reauth_required = True

    def __init__(self, code: str = "reauth_required", *, status_code: int | None = None) -> None:
        super().__init__(code, status_code=status_code)


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64url_decode(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def new_host_id() -> str:
    return f"urn:uuid:{uuid.uuid4()}"


@dataclass(frozen=True, slots=True)
class ChatGptAuthorizationRequest:
    url: str = field(repr=False)
    state: str = field(repr=False)
    nonce: str = field(repr=False)
    code_verifier: str = field(repr=False)
    redirect_uri: str
    client_id: str | None


def build_authorization_request(
    *,
    redirect_uri: str,
    host_id: str,
    client_id: str | None,
    login_hint: str | None = None,
    force_consent: bool = False,
) -> ChatGptAuthorizationRequest:
    state = secrets.token_urlsafe(32)
    nonce = secrets.token_urlsafe(32)
    verifier = secrets.token_urlsafe(64)
    params: dict[str, str] = {
        "client_id": client_id or CHATGPT_DYNAMIC_CLIENT_ID,
        "ext_agent_host_id": host_id,
        "response_type": "code",
        "redirect_uri": redirect_uri,
        "scope": " ".join(CHATGPT_SCOPES),
        "resource": CHATGPT_API_RESOURCE,
        "state": state,
        "nonce": nonce,
        "code_challenge_method": "S256",
        "code_challenge": _b64url(hashlib.sha256(verifier.encode("ascii")).digest()),
    }
    if client_id is None:
        params["agent_name_hint"] = CHATGPT_AGENT_NAME
    if login_hint:
        params["login_hint"] = login_hint
    if force_consent:
        params["prompt"] = "consent"
    return ChatGptAuthorizationRequest(
        url=f"{CHATGPT_AUTHORIZE_URL}?{urlencode(params)}",
        state=state,
        nonce=nonce,
        code_verifier=verifier,
        redirect_uri=redirect_uri,
        client_id=client_id,
    )


@dataclass(frozen=True, slots=True)
class ChatGptTokenSet:
    access_token: str = field(repr=False)
    refresh_token: str = field(repr=False)
    expires_at: float
    scopes: tuple[str, ...]
    id_token: str | None = field(default=None, repr=False)

    @property
    def plan_usage_granted(self) -> bool:
        return CHATGPT_PLAN_SCOPE in self.scopes


@dataclass(frozen=True, slots=True)
class ChatGptIdentity:
    subject: str
    email: str | None


def token_set_from_response(
    payload: Mapping[str, object],
    *,
    previous_refresh_token: str | None = None,
    now: float | None = None,
) -> ChatGptTokenSet:
    access_token = payload.get("access_token")
    if not isinstance(access_token, str) or not access_token:
        raise ChatGptAuthError("token_response_invalid")
    refresh_token = payload.get("refresh_token")
    if not isinstance(refresh_token, str) or not refresh_token:
        refresh_token = previous_refresh_token
    if not refresh_token:
        raise ChatGptAuthError("token_response_invalid")
    expires_in = payload.get("expires_in")
    ttl = float(expires_in) if isinstance(expires_in, int | float) and expires_in > 0 else 3600.0
    scope = payload.get("scope")
    scopes = tuple(scope.split()) if isinstance(scope, str) else ()
    id_token = payload.get("id_token")
    return ChatGptTokenSet(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_at=(time.time() if now is None else now) + ttl,
        scopes=scopes,
        id_token=id_token if isinstance(id_token, str) and id_token else None,
    )


def _oauth_error_code(response: httpx.Response) -> str | None:
    try:
        payload = response.json()
    except ValueError:
        return None
    if not isinstance(payload, dict):
        return None
    error = payload.get("error")
    if isinstance(error, dict):
        error = error.get("code") or error.get("type")
    allowed = _TERMINAL_REFRESH_ERRORS | {
        "access_denied",
        "invalid_request",
        "invalid_scope",
        "unauthorized_client",
        "server_error",
        "temporarily_unavailable",
    }
    return error if isinstance(error, str) and error in allowed else None


async def exchange_authorization_code(
    http: httpx.AsyncClient,
    *,
    client_id: str,
    code: str,
    code_verifier: str,
    redirect_uri: str,
) -> Mapping[str, object]:
    response = await http.post(
        CHATGPT_TOKEN_URL,
        data={
            "grant_type": "authorization_code",
            "client_id": client_id,
            "code": code,
            "code_verifier": code_verifier,
            "redirect_uri": redirect_uri,
            "resource": CHATGPT_API_RESOURCE,
        },
    )
    if response.status_code != 200:
        raise ChatGptAuthError(
            _oauth_error_code(response) or "token_exchange_failed",
            status_code=response.status_code,
        )
    payload = response.json()
    if not isinstance(payload, dict):
        raise ChatGptAuthError("token_response_invalid")
    return payload


async def refresh_access_token(
    http: httpx.AsyncClient,
    *,
    client_id: str,
    refresh_token: str,
) -> Mapping[str, object]:
    response = await http.post(
        CHATGPT_TOKEN_URL,
        data={
            "grant_type": "refresh_token",
            "client_id": client_id,
            "refresh_token": refresh_token,
            "resource": CHATGPT_API_RESOURCE,
        },
    )
    if response.status_code == 200:
        payload = response.json()
        if not isinstance(payload, dict):
            raise ChatGptAuthError("token_response_invalid")
        return payload
    code = _oauth_error_code(response) or "refresh_failed"
    if code in _TERMINAL_REFRESH_ERRORS:
        raise ChatGptReauthRequired(code, status_code=response.status_code)
    raise ChatGptAuthError(code, status_code=response.status_code)


@dataclass(frozen=True, slots=True)
class ChatGptRevocationResult:
    revoked: bool
    status_code: int


async def revoke_refresh_token(
    http: httpx.AsyncClient,
    *,
    client_id: str,
    refresh_token: str,
) -> ChatGptRevocationResult:
    response = await http.post(
        CHATGPT_REVOKE_URL,
        data={
            "token": refresh_token,
            "token_type_hint": "refresh_token",
            "client_id": client_id,
        },
    )
    return ChatGptRevocationResult(response.status_code == 200, response.status_code)


def _decode_jwt(token: str) -> tuple[dict[str, object], dict[str, object], bytes, bytes]:
    parts = token.split(".")
    if len(parts) != 3:
        raise ChatGptAuthError("invalid_id_token")
    try:
        header = json.loads(_b64url_decode(parts[0]))
        claims = json.loads(_b64url_decode(parts[1]))
        signature = _b64url_decode(parts[2])
    except (ValueError, json.JSONDecodeError) as exc:
        raise ChatGptAuthError("invalid_id_token") from exc
    if not isinstance(header, dict) or not isinstance(claims, dict):
        raise ChatGptAuthError("invalid_id_token")
    return header, claims, f"{parts[0]}.{parts[1]}".encode("ascii"), signature


def _rsa_public_key(jwk: Mapping[str, object]) -> rsa.RSAPublicKey:
    n = jwk.get("n")
    e = jwk.get("e")
    if not isinstance(n, str) or not isinstance(e, str):
        raise ChatGptAuthError("invalid_id_token")
    return rsa.RSAPublicNumbers(
        int.from_bytes(_b64url_decode(e), "big"),
        int.from_bytes(_b64url_decode(n), "big"),
    ).public_key()


@dataclass(slots=True)
class ChatGptIdTokenVerifier:
    http: httpx.AsyncClient
    jwks_url: str = CHATGPT_JWKS_URL
    issuer: str = CHATGPT_ISSUER
    clock: Callable[[], float] = time.time
    _keys: dict[str, Mapping[str, object]] = field(default_factory=dict, repr=False)

    async def _load_keys(self) -> None:
        response = await self.http.get(self.jwks_url)
        if response.status_code != 200:
            raise ChatGptAuthError("jwks_unavailable", status_code=response.status_code)
        payload = response.json()
        keys = payload.get("keys") if isinstance(payload, dict) else None
        if not isinstance(keys, list):
            raise ChatGptAuthError("jwks_unavailable")
        self._keys = {
            str(key["kid"]): key for key in keys if isinstance(key, dict) and "kid" in key
        }

    async def verify(self, token: str, *, client_id: str, nonce: str) -> ChatGptIdentity:
        header, claims, signed, signature = _decode_jwt(token)
        if header.get("alg") != "RS256":
            raise ChatGptAuthError("invalid_id_token")
        kid = str(header.get("kid") or "")
        if kid not in self._keys:
            await self._load_keys()
        jwk = self._keys.get(kid)
        if jwk is None:
            raise ChatGptAuthError("invalid_id_token")
        try:
            _rsa_public_key(jwk).verify(signature, signed, padding.PKCS1v15(), hashes.SHA256())
        except Exception as exc:
            raise ChatGptAuthError("invalid_id_token") from exc
        audience = claims.get("aud")
        audiences = audience if isinstance(audience, list) else [audience]
        now = self.clock()
        exp = claims.get("exp")
        if (
            claims.get("iss") != self.issuer
            or client_id not in audiences
            or claims.get("nonce") != nonce
            or not isinstance(exp, int | float)
            or exp + _ID_TOKEN_LEEWAY_S < now
        ):
            raise ChatGptAuthError("invalid_id_token")
        subject = claims.get("sub")
        if not isinstance(subject, str) or not subject:
            raise ChatGptAuthError("invalid_id_token")
        email = claims.get("email")
        return ChatGptIdentity(subject=subject, email=email if isinstance(email, str) else None)


__all__ = [
    "CHATGPT_AGENT_NAME",
    "CHATGPT_API_RESOURCE",
    "CHATGPT_AUTHORIZE_URL",
    "CHATGPT_DYNAMIC_CLIENT_ID",
    "CHATGPT_ISSUER",
    "CHATGPT_JWKS_URL",
    "CHATGPT_PLAN_SCOPE",
    "CHATGPT_REVOKE_URL",
    "CHATGPT_SCOPES",
    "CHATGPT_TOKEN_URL",
    "ChatGptAuthError",
    "ChatGptAuthorizationRequest",
    "ChatGptIdTokenVerifier",
    "ChatGptIdentity",
    "ChatGptReauthRequired",
    "ChatGptTokenSet",
    "build_authorization_request",
    "exchange_authorization_code",
    "new_host_id",
    "refresh_access_token",
    "revoke_refresh_token",
    "token_set_from_response",
]
