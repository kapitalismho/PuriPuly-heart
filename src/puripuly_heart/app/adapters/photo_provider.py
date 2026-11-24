from __future__ import annotations

import hashlib
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from urllib.parse import urlsplit

import httpx

from puripuly_heart.app.adapters.photo_access import PhotoManagedCredentialOwner, photo_token_error
from puripuly_heart.app.ports.agent_chat import AgentTokenPort, AgentTransportError
from puripuly_heart.app.ports.settings_secrets import SettingsSecretStorePort
from puripuly_heart.config.service_endpoints import DEFAULT_PHOTO_RELAY_ENDPOINT
from puripuly_heart.core import network_clients
from puripuly_heart.core.chatgpt.session import ChatGptSession
from puripuly_heart.core.external_network import NetworkConfigurationError
from puripuly_heart.core.first_party_http import response_evidence
from puripuly_heart.core.owned_thread import run_owned_thread_call
from puripuly_heart.core.photo.lifecycle import PhotoRoute
from puripuly_heart.core.photo.protocol import PhotoProviderError
from puripuly_heart.providers.photo import (
    OpenAIPhotoConnection,
    OpenRouterPhotoConnection,
    RelayPhotoConnection,
    first_party_photo_http_error,
    make_photo_connection,
    report_first_party_photo_transport_failure,
)


def photo_route_kind(route: PhotoRoute | None) -> str | None:
    if route is None:
        return None
    if route.identity == "luna":
        return "chatgpt"
    if route.identity.startswith("google-"):
        return "gemini"
    if route.identity.startswith("openai-"):
        return "openai"
    return "managed"


@dataclass(slots=True)
class PhotoProviderAdapter:
    session: ChatGptSession
    secret_store: Callable[[], SettingsSecretStorePort]
    token_owner: AgentTokenPort | None = None
    relay_endpoint: str = DEFAULT_PHOTO_RELAY_ENDPOINT
    allow_loopback_http: bool = False
    benchmark_route: str | None = None
    broker_base_url: Callable[[], str] = lambda: ""
    installation_id: Callable[[], Awaitable[str]] | None = None
    selected_route: Callable[[], str] = lambda: "managed"
    _comparison_access: PhotoManagedCredentialOwner | None = field(default=None, init=False)
    _http: httpx.AsyncClient | None = field(default=None, init=False)
    _allowance_seen: tuple[int, int, date] | None = field(default=None, init=False)

    async def resolve_route(self) -> PhotoRoute | None:
        route = self.benchmark_route or self.selected_route()
        if route == "chatgpt":
            return await run_owned_thread_call(self._chatgpt_route)
        if route == "gemini":
            return await run_owned_thread_call(self._gemini_route)
        if route == "managed":
            if self.token_owner is None:
                return None
            return PhotoRoute("managed-luna", self._managed)
        if route == "managed-openrouter":
            if self.token_owner is None or self.installation_id is None:
                return None
            if self._comparison_access is None:
                self._comparison_access = PhotoManagedCredentialOwner(
                    self.token_owner,
                    self.installation_id,
                    self.broker_base_url,
                    self.secret_store,
                    self._client,
                    allow_loopback_http=self.allow_loopback_http,
                )
            return PhotoRoute("benchmark-managed-openrouter-luna", self._managed_openrouter)
        if route in ("openai", "openai-api"):
            return await run_owned_thread_call(self._openai_route)
        if route != "personal":
            raise ValueError("unsupported photo access")
        return await run_owned_thread_call(self._observe_route)

    async def allowance(self) -> tuple[int, int] | None:
        endpoint = urlsplit(self.relay_endpoint)
        if (
            self.token_owner is None
            or not endpoint.hostname
            or endpoint.path != "/v1/photo"
            or endpoint.query
            or endpoint.fragment
            or not (
                endpoint.scheme == "https"
                or self.allow_loopback_http
                and endpoint.scheme == "http"
                and endpoint.hostname in {"127.0.0.1", "::1"}
            )
        ):
            return None
        try:
            token = await self.token_owner.get_token()
            async with network_clients.external_async_client(
                timeout=httpx.Timeout(20), follow_redirects=False
            ) as client:
                response = await client.post(
                    self.relay_endpoint + "/allowance",
                    json={},
                    headers={
                        "Authorization": f"Bearer {token}",
                        "User-Agent": network_clients.first_party_user_agent(),
                    },
                )
            if response.status_code != 200 or "response_shape" in response_evidence(
                response, body=response.content
            ):
                error = first_party_photo_http_error(
                    response, operation="photo_relay_allowance", body=response.content
                )
                if error.code == "photo_authentication_failed":
                    self.token_owner.invalidate(token)
                return None
            body = response.json()
        except AgentTransportError as error:
            photo_token_error(error)
            return None
        except (httpx.RequestError, OSError) as error:
            report_first_party_photo_transport_failure(error, operation="photo_relay_allowance")
            return None
        except Exception:
            return None
        if not isinstance(body, dict):
            return None
        remaining = body.get("remaining")
        allowance = body.get("allowance")
        if (
            type(remaining) is not int
            or type(allowance) is not int
            or allowance < 1
            or not 0 <= remaining <= allowance
        ):
            return None
        self._allowance_seen = (remaining, allowance, datetime.now(timezone.utc).date())
        return remaining, allowance

    def cached_allowance(self) -> tuple[int, int] | None:
        seen = self._allowance_seen
        if seen is None or seen[2] != datetime.now(timezone.utc).date():
            return None
        return seen[0], seen[1]

    def _client(self) -> httpx.AsyncClient:
        if self._http is None:
            self._http = network_clients.external_async_client(
                timeout=httpx.Timeout(240, connect=20),
                follow_redirects=False,
                limits=httpx.Limits(
                    max_connections=1, max_keepalive_connections=1, keepalive_expiry=30
                ),
            )
        return self._http

    async def _token(self) -> str:
        if self.token_owner is None:
            raise PhotoProviderError("photo_authentication_failed", retryable=False)
        try:
            return await self.token_owner.get_token()
        except AgentTransportError as error:
            raise photo_token_error(error) from None

    def _managed(self, model: str | None):
        if model not in (None, "gpt-6-luna"):
            raise PhotoProviderError("photo_request_rejected", retryable=False)
        try:
            client = self._client()
        except NetworkConfigurationError as error:
            report_first_party_photo_transport_failure(error, operation="photo_relay")
            raise PhotoProviderError("photo_transport_failed", retryable=False) from None
        return RelayPhotoConnection(
            self.relay_endpoint,
            self._token,
            client=client,
            allow_loopback_http=self.allow_loopback_http,
            invalidate=self.token_owner.invalidate if self.token_owner is not None else None,
        )

    def _managed_openrouter(self, model: str | None):
        if model not in (None, "openai/gpt-6-luna") or self._comparison_access is None:
            raise PhotoProviderError("photo_request_rejected", retryable=False)
        return OpenRouterPhotoConnection(
            self._comparison_access.get_credential, client=self._client()
        )

    async def close(self) -> None:
        if self._http is not None:
            await self._http.aclose()
            self._http = None

    def _observe_route(self) -> PhotoRoute | None:
        return self._chatgpt_route() or self._gemini_route()

    def _chatgpt_route(self) -> PhotoRoute | None:
        if self.session.status().signed_in is not True:
            return None
        return PhotoRoute(
            "luna",
            lambda model: make_photo_connection(
                route="luna", chatgpt_session=self.session, model=model
            ),
        )

    def _openai_route(self) -> PhotoRoute | None:
        try:
            key = self.secret_store().get("openai_api_key")
        except Exception:
            return None
        if not key:
            return None

        def connect(model: str | None):
            if model not in (None, "gpt-6-luna"):
                raise PhotoProviderError("photo_request_rejected", retryable=False)
            return OpenAIPhotoConnection(key, client=self._client())

        return PhotoRoute("openai-" + hashlib.sha256(key.encode()).hexdigest(), connect)

    def _gemini_route(self) -> PhotoRoute | None:
        try:
            key = self.secret_store().get("gemini_api_key")
        except Exception:
            return None
        if not key:
            return None
        return PhotoRoute(
            "google-" + hashlib.sha256(key.encode()).hexdigest(),
            lambda model: make_photo_connection(route="google", gemini_api_key=key, model=model),
        )
