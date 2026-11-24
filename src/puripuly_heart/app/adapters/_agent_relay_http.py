from __future__ import annotations

import json
from collections.abc import AsyncIterator, Awaitable
from contextlib import asynccontextmanager
from typing import Literal
from urllib.parse import SplitResult, urlsplit

import httpx

from puripuly_heart.app.ports.agent_chat import AgentFailureEvidence, AgentTransportError
from puripuly_heart.core.external_network import ProxyPolicy, ProxyRoute
from puripuly_heart.core.first_party_http import response_evidence, response_media_type
from puripuly_heart.core.http_client_logging import suppress_http_client_logs
from puripuly_heart.core.messages import DiagnosticCategory, DiagnosticFieldValue
from puripuly_heart.core.network_clients import external_async_client, first_party_user_agent
from puripuly_heart.core.network_diagnostics import (
    TRANSPORT_ENUM_FIELDS,
    classify_transport_error,
    safe_transport_fields,
)

AgentEndpoint = Literal["broker_token", "relay_classify", "relay_response", "relay_support"]
_HTTP_FAILURES = (httpx.HTTPError, httpx.InvalidURL, OSError)


class AgentConnectionError(AgentTransportError):
    def __init__(
        self,
        exc: Exception,
        endpoint: AgentEndpoint,
        *,
        response: httpx.Response | None = None,
        streaming: bool = False,
    ) -> None:
        kind = classify_transport_error(exc)
        self.retryable = kind in {"network", "timeout"} and isinstance(
            exc, (httpx.TimeoutException, httpx.NetworkError)
        )
        fields = {
            key: value for key, value in safe_transport_fields(exc).items() if value != "unknown"
        }
        if kind is not None:
            fields["network_error_kind"] = kind
        code = {"proxy": "proxy_failed", "tls": "tls_failed"}.get(kind, "relay_unavailable")
        configuration = fields.get("configuration_error")
        if configuration == "invalid_tls_configuration":
            code = "tls_failed"
        elif configuration in {"invalid_proxy_configuration", "unsupported_proxy_configuration"}:
            code = "proxy_failed"
        if streaming:
            code = "stream_interrupted"
            fields["agent_failure"] = code
        category: DiagnosticCategory = "timeout" if kind == "timeout" else "network"
        if response is None:
            fields.setdefault("stage", "request_send")
            fields.setdefault("dispatch_certainty", "possible")
            evidence = AgentFailureEvidence(category, {**fields, "agent_endpoint": endpoint})
        else:
            evidence = response_details(
                response, endpoint, category, stage="response_body", fields=fields
            )
        super().__init__(code, evidence=evidence)


def is_loopback_http(url: SplitResult) -> bool:
    return url.scheme == "http" and url.hostname in {"127.0.0.1", "::1"}


class _AgentLoopbackProxyPolicy(ProxyPolicy):
    def route(self, url: str) -> ProxyRoute:
        if is_loopback_http(urlsplit(url)):
            return ProxyRoute(None, "direct")
        return super().route(url)


def agent_client(
    endpoint: AgentEndpoint, *, timeout: float, allow_loopback_http: bool
) -> httpx.AsyncClient:
    try:
        return external_async_client(
            follow_redirects=False,
            timeout=timeout,
            headers={"User-Agent": first_party_user_agent()},
            policy=_AgentLoopbackProxyPolicy() if allow_loopback_http else None,
        )
    except OSError as exc:
        error = AgentConnectionError(exc, endpoint)
    raise error


async def _http_call[T](
    operation: Awaitable[T],
    endpoint: AgentEndpoint,
    *,
    response: httpx.Response | None = None,
    streaming: bool = False,
) -> T:
    try:
        with suppress_http_client_logs():
            return await operation
    except _HTTP_FAILURES as exc:
        error = AgentConnectionError(exc, endpoint, response=response, streaming=streaming)
    raise error


async def close_agent_client(client: httpx.AsyncClient, endpoint: AgentEndpoint) -> None:
    await _http_call(client.aclose(), endpoint)


@asynccontextmanager
async def post_stream(
    client: httpx.AsyncClient, url: str, endpoint: AgentEndpoint, **options
) -> AsyncIterator[httpx.Response]:
    try:
        request = client.build_request("POST", url, **options)
    except (httpx.InvalidURL, UnicodeError, ValueError) as exc:
        error = AgentConnectionError(exc, endpoint)
    else:
        response = await _http_call(client.send(request, stream=True), endpoint)
        try:
            yield response
        finally:
            await _http_call(response.aclose(), endpoint, response=response)
        return
    raise error


def response_details(
    response: httpx.Response,
    endpoint: AgentEndpoint,
    category: DiagnosticCategory,
    *,
    stage: str = "protocol_parse",
    fields: dict[str, DiagnosticFieldValue] | None = None,
    retry_after_ms: int | None = None,
) -> AgentFailureEvidence:
    connection = response.extensions.get("connection_diagnostics", {})
    metadata: dict[str, DiagnosticFieldValue] = {
        key: value
        for key in ("transport", "tls_source", "tls_backend", "proxy_source")
        if type(value := connection.get(key)) is str and value in TRANSPORT_ENUM_FIELDS[key]
    }
    metadata.update(response_evidence(response))
    metadata.update(fields or {})
    metadata.update(agent_endpoint=endpoint, stage=stage, dispatch_certainty="response_received")
    status = response.status_code
    return AgentFailureEvidence(
        category, metadata, status if 100 <= status <= 599 else None, retry_after_ms
    )


def response_failure(
    response: httpx.Response,
    endpoint: AgentEndpoint,
    code: str,
    *,
    category: DiagnosticCategory = "invalid_response",
    fields: dict[str, DiagnosticFieldValue] | None = None,
) -> AgentTransportError:
    return AgentTransportError(
        code,
        evidence=response_details(
            response, endpoint, category, fields={**(fields or {}), "agent_failure": code}
        ),
    )


def check_response_shape(
    response: httpx.Response, endpoint: AgentEndpoint, *, body: bytes = b""
) -> None:
    fields = response_evidence(response, body=body)
    if fields.get("response_shape") == "edge_challenge":
        raise response_failure(response, endpoint, "edge_challenge", fields=fields)
    if fields.get("response_shape") == "html":
        raise response_failure(response, endpoint, "unexpected_html", fields=fields)


async def read_json(response: httpx.Response, endpoint: AgentEndpoint) -> dict | None:
    check_response_shape(response, endpoint)
    media_type = response_media_type(response)
    if (
        response.status_code == 200
        and endpoint != "relay_response"
        and media_type == "application/json"
    ):
        body = await _http_call(response.aread(), endpoint, response=response)
    else:
        body = bytearray()
        chunks = response.aiter_bytes()
        try:
            while True:
                try:
                    chunk = await _http_call(anext(chunks), endpoint, response=response)
                except StopAsyncIteration:
                    break
                remaining = 64 * 1024 - len(body)
                if len(chunk) > remaining:
                    body.extend(chunk[: max(0, 512 - len(body))])
                    check_response_shape(response, endpoint, body=bytes(body[:512]))
                    return None
                check_prefix = len(body) < 512
                body.extend(chunk)
                if check_prefix:
                    check_response_shape(response, endpoint, body=bytes(body[:512]))
        finally:
            await chunks.aclose()
    check_response_shape(response, endpoint, body=bytes(body[:512]))
    if media_type == "application/json":
        try:
            payload = json.loads(body)
        except (ValueError, RecursionError):
            return None
        if isinstance(payload, dict):
            return payload
    return None


async def response_lines(response: httpx.Response, endpoint: AgentEndpoint) -> AsyncIterator[str]:
    lines = response.aiter_lines()
    try:
        while True:
            try:
                line = await _http_call(anext(lines), endpoint, response=response, streaming=True)
            except StopAsyncIteration:
                return
            yield line
    finally:
        await lines.aclose()


def error_envelope(payload: dict | None) -> dict:
    error = None if payload is None else payload.get("error")
    return error if isinstance(error, dict) and isinstance(error.get("code"), str) else {}
