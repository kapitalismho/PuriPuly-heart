from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from urllib.parse import urlsplit

import httpx

from puripuly_heart.app.ports.agent_chat import AgentTokenPort, AgentTransportError
from puripuly_heart.app.ports.settings_secrets import SettingsSecretStorePort
from puripuly_heart.config.service_endpoints import (
    DEFAULT_OPENROUTER_BROKER_BASE_URL,
    LEGACY_OPENROUTER_BROKER_BASE_URL,
)
from puripuly_heart.core import network_clients
from puripuly_heart.core.first_party_http import response_evidence
from puripuly_heart.core.owned_thread import run_owned_thread_call
from puripuly_heart.core.photo.protocol import PhotoProviderError
from puripuly_heart.providers.photo import (
    emit_photo_service_failure,
    first_party_photo_http_error,
    report_first_party_photo_transport_failure,
)

_PHOTO_ACCESS_PATH = "/v1/photo/access"


def photo_token_error(error: AgentTransportError) -> PhotoProviderError:
    code = {
        "service_authentication_failed": "photo_authentication_failed",
        "service_authentication_unavailable": "photo_provider_unavailable",
        "service_rate_limited": "photo_quota_exhausted",
        "edge_challenge": "photo_stream_invalid",
        "unexpected_html": "photo_stream_invalid",
        "invalid_response": "photo_stream_invalid",
        "unexpected_http_status": "photo_request_rejected",
    }.get(error.code, "photo_transport_failed")
    evidence = error.evidence
    if evidence is not None:
        emit_photo_service_failure(
            code,
            operation="photo_broker_token",
            category=evidence.category,
            fields=evidence.fields,
            status_code=evidence.status_code,
            retry_after_ms=evidence.retry_after_ms,
        )
    return PhotoProviderError(code, retryable=False)


def _credential_scope(endpoint: str, identity: str) -> str:
    if endpoint == DEFAULT_OPENROUTER_BROKER_BASE_URL + _PHOTO_ACCESS_PATH:
        endpoint = LEGACY_OPENROUTER_BROKER_BASE_URL + _PHOTO_ACCESS_PATH
    return "photo_openrouter_" + hashlib.sha256(f"{endpoint}\n{identity}".encode()).hexdigest()


class PhotoManagedCredentialOwner:
    def __init__(
        self,
        token_owner: AgentTokenPort,
        identity: Callable[[], Awaitable[str]],
        broker_base_url: Callable[[], str],
        secret_store: Callable[[], SettingsSecretStorePort],
        client: Callable[[], httpx.AsyncClient],
        *,
        allow_loopback_http: bool = False,
    ) -> None:
        self._token_owner = token_owner
        self._identity = identity
        self._broker_base_url = broker_base_url
        self._secret_store = secret_store
        self._client = client
        self._allow_loopback_http = allow_loopback_http
        self._lock = asyncio.Lock()
        self._cached: tuple[str, dict] | None = None

    def _endpoint(self) -> str:
        base = self._broker_base_url()
        url = urlsplit(base)
        if (
            not url.hostname
            or url.username is not None
            or url.password is not None
            or url.path not in ("", "/")
            or url.query
            or url.fragment
            or not (
                url.scheme == "https"
                or self._allow_loopback_http
                and url.scheme == "http"
                and url.hostname in {"127.0.0.1", "::1"}
            )
        ):
            raise PhotoProviderError("photo_authentication_failed", retryable=False)
        return base.rstrip("/") + _PHOTO_ACCESS_PATH

    async def _request(self, endpoint: str, body: dict) -> dict:
        try:
            token = await self._token_owner.get_token()
            response = await self._client().post(
                endpoint,
                json=body,
                headers={
                    "Authorization": f"Bearer {token}",
                    "User-Agent": network_clients.first_party_user_agent(),
                },
                timeout=60,
                follow_redirects=False,
            )
            if (
                response.status_code != 200
                and not (response.status_code == 202 and body.get("action") == "revoke")
            ) or "response_shape" in response_evidence(response, body=response.content):
                error = first_party_photo_http_error(
                    response,
                    operation="photo_broker_access",
                    body=response.content,
                    auth_code="invalid_token",
                )
                if error.code == "photo_authentication_failed":
                    self._token_owner.invalidate(token)
                raise error
            try:
                result = response.json()
            except (ValueError, RecursionError):
                raise first_party_photo_http_error(
                    response,
                    operation="photo_broker_access",
                    body=response.content,
                    auth_code="invalid_token",
                ) from None
            if not isinstance(result, dict):
                raise first_party_photo_http_error(
                    response,
                    operation="photo_broker_access",
                    body=response.content,
                    auth_code="invalid_token",
                )
            return result
        except AgentTransportError as error:
            raise photo_token_error(error) from None
        except PhotoProviderError:
            raise
        except Exception as error:
            if isinstance(error, (httpx.RequestError, OSError)):
                report_first_party_photo_transport_failure(error, operation="photo_broker_access")
            raise PhotoProviderError("photo_transport_failed", retryable=False) from None

    @staticmethod
    def _valid(record: object) -> bool:
        if not isinstance(record, dict) or record.get("model") != "openai/gpt-6-luna":
            return False
        fields = (
            "key",
            "operation_id",
            "managed_credential_ref",
            "delivery_id",
            "delivery_ack_token",
            "expires_at",
        )
        if not all(isinstance(record.get(name), str) and record[name] for name in fields):
            return False
        if type(record.get("acknowledged")) is not bool:
            return False
        try:
            return datetime.fromisoformat(record["expires_at"]) > datetime.now(UTC)
        except TypeError, ValueError:
            return False

    async def get_credential(self) -> str:
        async with self._lock:
            endpoint = self._endpoint()
            identity = await self._identity()
            scope = _credential_scope(endpoint, identity)
            record = (
                self._cached[1] if self._cached is not None and self._cached[0] == scope else None
            )
            if record is None:
                raw = await run_owned_thread_call(lambda: self._secret_store().get(scope))
                try:
                    record = json.loads(raw) if raw else None
                except TypeError, ValueError:
                    raise PhotoProviderError(
                        "photo_authentication_failed", retryable=False
                    ) from None
            if isinstance(record, dict) and record.get("revocation_pending"):
                raise PhotoProviderError("photo_authentication_failed", retryable=False)
            if not self._valid(record):
                record = await self._request(endpoint, {"action": "get"})
                if not self._valid(record) or record["acknowledged"] is not False:
                    raise PhotoProviderError("photo_authentication_failed", retryable=False)
                encoded = json.dumps(record, separators=(",", ":"))
                await run_owned_thread_call(lambda: self._secret_store().set(scope, encoded))
            if not record["acknowledged"]:
                ack = await self._request(
                    endpoint,
                    {
                        "action": "acknowledge",
                        **{
                            key: record[key]
                            for key in (
                                "operation_id",
                                "managed_credential_ref",
                                "delivery_id",
                                "delivery_ack_token",
                            )
                        },
                    },
                )
                if (
                    ack.get("acknowledged") is not True
                    or ack.get("operation_id") != record["operation_id"]
                ):
                    raise PhotoProviderError("photo_authentication_failed", retryable=False)
                record = {**record, "acknowledged": True}
                encoded = json.dumps(record, separators=(",", ":"))
                await run_owned_thread_call(lambda: self._secret_store().set(scope, encoded))
            self._cached = scope, record
            return record["key"]

    async def revoke(self) -> bool:
        async with self._lock:
            endpoint = self._endpoint()
            identity = await self._identity()
            scope = _credential_scope(endpoint, identity)
            raw = await run_owned_thread_call(lambda: self._secret_store().get(scope))
            if not raw:
                return False
            record = json.loads(raw)
            self._cached = None
            encoded = json.dumps({**record, "revocation_pending": True}, separators=(",", ":"))
            await run_owned_thread_call(lambda: self._secret_store().set(scope, encoded))
            result = await self._request(
                endpoint, {"action": "revoke", "operation_id": record["operation_id"]}
            )
            if result.get("revoked") is not True:
                return False
            await run_owned_thread_call(lambda: self._secret_store().delete(scope))
            return True
