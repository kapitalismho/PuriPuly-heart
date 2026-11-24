from __future__ import annotations

import asyncio
import os
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from time import monotonic
from urllib.parse import urlsplit

import httpx

from puripuly_heart.app.adapters._agent_relay_http import (
    AgentConnectionError,
    agent_client,
    close_agent_client,
    error_envelope,
    post_stream,
    read_json,
    response_details,
    response_failure,
)
from puripuly_heart.app.ports.agent_chat import (
    AgentFailureEvidence,
    AgentServiceError,
    AgentTransportError,
)
from puripuly_heart.core.lifecycle import LifecycleScope
from puripuly_heart.core.managed_identity import ManagedIdentityBundle, encode_base64url


class AgentRelayTokenOwner:
    def __init__(
        self,
        broker_base_url: Callable[[], str],
        identity: Callable[[], Awaitable[ManagedIdentityBundle]],
        *,
        app_version: str,
        allow_loopback_http: bool = False,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        self._broker_base_url = broker_base_url
        self._allow_loopback_http = allow_loopback_http
        self._identity = identity
        self._app_version = app_version
        self._clock = clock
        self._scope = LifecycleScope("agent-relay-auth")
        self._closed = False
        self._token: str | None = None
        self._refresh_at = 0.0
        self._retry_at = 0.0
        self._rate_limit: AgentFailureEvidence | None = None
        self._refresh_task: asyncio.Task[str | AgentTransportError] | None = None
        self._client: httpx.AsyncClient | None = None

    def _broker_endpoint(self) -> str:
        base_url = self._broker_base_url()
        parsed = urlsplit(base_url)
        if (
            not parsed.hostname
            or not (
                parsed.scheme == "https"
                or (
                    self._allow_loopback_http
                    and parsed.scheme == "http"
                    and parsed.hostname in {"127.0.0.1", "::1"}
                )
            )
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path not in ("", "/")
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("Agent token broker requires HTTPS or explicit source loopback HTTP")
        return base_url.rstrip("/") + "/v1/puri/token"

    async def installation_id(self) -> str:
        return (await self._identity()).installation_id

    async def get_token(self) -> str:
        if self._closed:
            raise AgentServiceError("service_authentication_failed")
        if self._token is not None and self._clock() < self._refresh_at:
            return self._token
        if self._clock() < self._retry_at:
            raise AgentServiceError("service_rate_limited", evidence=self._rate_limit)
        if self._refresh_task is None:
            self._refresh_task = self._scope.create_task(self._refresh(), name="token-refresh")
        task = self._refresh_task
        try:
            result = await asyncio.shield(task)
            if self._closed:
                raise AgentServiceError("service_authentication_failed")
            if isinstance(result, AgentTransportError):
                raise result
            return result
        finally:
            if task.done() and self._refresh_task is task:
                self._refresh_task = None

    def invalidate(self, token: str) -> None:
        if self._token == token:
            self._token = None
            self._refresh_at = 0.0

    async def _refresh(self) -> str | AgentTransportError:
        self._token = None
        try:
            endpoint = self._broker_endpoint()
            bundle = await self._identity()
            if self._client is None:
                self._client = agent_client("broker_token", timeout=20)
            for attempt in range(2):
                request = bundle.sign_puri_token_request(
                    app_version=self._app_version,
                    signed_at=datetime.now(UTC)
                    .isoformat(timespec="milliseconds")
                    .replace("+00:00", "Z"),
                    nonce=encode_base64url(os.urandom(32)),
                )
                requested_at = self._clock()
                try:
                    async with post_stream(
                        self._client, endpoint, "broker_token", json=request
                    ) as response:
                        payload = await read_json(response, "broker_token")
                except AgentConnectionError as error:
                    if attempt or not error.retryable:
                        return error
                    await asyncio.sleep(1)
                    continue
                error_body = error_envelope(payload)
                fields = {"response_shape": "public_error_envelope"} if error_body else {}
                status = response.status_code
                if status == 429:
                    delay = self._retry_delay(response, error_body)
                    self._retry_at = self._clock() + delay
                    self._rate_limit = response_details(
                        response,
                        "broker_token",
                        "rate_limit",
                        fields=fields,
                        retry_after_ms=int(delay * 1000),
                    )
                    return AgentServiceError("service_rate_limited", evidence=self._rate_limit)
                if 500 <= status < 600:
                    if attempt:
                        return AgentServiceError(
                            "service_authentication_unavailable",
                            evidence=response_details(
                                response, "broker_token", "service_unavailable", fields=fields
                            ),
                        )
                    await asyncio.sleep(1)
                    continue
                break
            if status != 200:
                if (
                    error_body.get("code") == "invalid_request"
                    and error_body.get("class") == "security_fail"
                    and (status, error_body.get("subcode")) in (
                        (401, "puri_proof_invalid"),
                        (409, "installation_binding_mismatch"),
                    )
                ):
                    return AgentServiceError(
                        "service_authentication_failed",
                        evidence=response_details(response, "broker_token", "auth", fields=fields),
                    )
                return response_failure(
                    response, "broker_token", "unexpected_http_status", fields=fields
                )
            parsed = self._parse_token(payload)
            if parsed is None:
                return response_failure(
                    response,
                    "broker_token",
                    "invalid_response",
                    fields={"response_shape": "malformed_success"},
                )
            if self._closed:
                return AgentServiceError("service_authentication_failed")
            token, remaining = parsed
            self._token = token
            self._rate_limit = None
            self._refresh_at = min(requested_at + 3570, self._clock() + remaining - 30)
            return token
        except AgentTransportError as error:
            return error
        except Exception:
            return AgentServiceError("service_authentication_failed")

    @staticmethod
    def _parse_token(payload: dict | None) -> tuple[str, float] | None:
        if payload is None:
            return None
        token = payload.get("access_token")
        expires_at = payload.get("expires_at")
        if (
            payload.get("token_type") != "Bearer"
            or type(payload.get("expires_in")) is not int
            or payload["expires_in"] != 3600
            or not isinstance(token, str)
            or not token
            or any(char.isspace() for char in token)
            or not isinstance(expires_at, str)
        ):
            return None
        try:
            expiry = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
            if expiry.utcoffset() is None:
                return None
            remaining = min(3600.0, (expiry - datetime.now(UTC)).total_seconds())
        except (ValueError, OverflowError):
            return None
        return (token, remaining) if remaining > 30 else None

    @staticmethod
    def _retry_delay(response: httpx.Response, error: dict) -> float:
        milliseconds = (
            error.get("retry_after_ms")
            if error.get("code") == "rate_limited" and error.get("class") == "retryable"
            else None
        )
        if type(milliseconds) in (int, float) and 0 < milliseconds <= 86_400_000:
            return max(1.0, milliseconds / 1000)
        header = response.headers.get("retry-after", "")
        if len(header) <= 5 and header.isascii() and header.isdecimal() and 0 < int(header) <= 86_400:
            return float(header)
        return 30.0

    async def close(self) -> None:
        self._closed = True
        self._token = None
        try:
            await self._scope.close()
        finally:
            self._refresh_task = None
            if self._client is not None:
                await close_agent_client(self._client, "broker_token")
