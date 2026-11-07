from __future__ import annotations

import asyncio
import os
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from time import monotonic
from urllib.parse import urlsplit

import httpx

from puripuly_heart.app.ports.agent_chat import AgentServiceError, AgentTransportError
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

    async def get_token(self) -> str:
        if self._closed:
            raise AgentServiceError("service_authentication_failed")
        if self._token is not None and self._clock() < self._refresh_at:
            return self._token
        if self._clock() < self._retry_at:
            raise AgentServiceError("service_rate_limited")
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
                self._client = httpx.AsyncClient(
                    trust_env=False, follow_redirects=False, timeout=20
                )
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
                    response = await self._client.post(endpoint, json=request)
                except httpx.TimeoutException, httpx.NetworkError:
                    if attempt:
                        return AgentTransportError()
                    await asyncio.sleep(1)
                    continue
                if response.status_code == 429:
                    delay = self._retry_delay(response)
                    self._retry_at = self._clock() + delay
                    return AgentServiceError("service_rate_limited")
                if 500 <= response.status_code < 600:
                    if attempt:
                        return AgentServiceError("service_authentication_unavailable")
                    await asyncio.sleep(1)
                    continue
                break
            if response.status_code != 200:
                return AgentServiceError("service_authentication_failed")
            payload = response.json()
            if not isinstance(payload, dict):
                return AgentServiceError("service_authentication_failed")
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
                return AgentServiceError("service_authentication_failed")
            expiry = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
            if expiry.utcoffset() is None:
                return AgentServiceError("service_authentication_failed")
            remaining = min(3600.0, (expiry - datetime.now(UTC)).total_seconds())
            if remaining <= 30:
                return AgentServiceError("service_authentication_failed")
            if self._closed:
                return AgentServiceError("service_authentication_failed")
            self._token = token
            self._refresh_at = min(requested_at + 3570, self._clock() + remaining - 30)
            return token
        except httpx.HTTPError:
            return AgentTransportError()
        except Exception:
            return AgentServiceError("service_authentication_failed")

    @staticmethod
    def _retry_delay(response: httpx.Response) -> float:
        try:
            payload = response.json()
        except ValueError:
            payload = None
        error = payload.get("error") if isinstance(payload, dict) else None
        milliseconds = error.get("retry_after_ms") if isinstance(error, dict) else None
        if type(milliseconds) in (int, float) and 0 < milliseconds <= 86_400_000:
            return max(1.0, milliseconds / 1000)
        header = response.headers.get("retry-after", "")
        if header.isascii() and header.isdecimal() and 0 < int(header) <= 86_400:
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
                await self._client.aclose()
