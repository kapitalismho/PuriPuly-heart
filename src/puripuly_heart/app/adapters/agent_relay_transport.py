from __future__ import annotations

import json
import re
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import aclosing, asynccontextmanager
from dataclasses import dataclass
from urllib.parse import urlsplit

import httpx

from puripuly_heart.app.adapters._agent_relay_http import (
    AgentEndpoint,
    agent_client,
    check_response_shape,
    close_agent_client,
    error_envelope,
    is_loopback_http,
    post_stream,
    read_json,
    response_details,
    response_failure,
    response_lines,
)
from puripuly_heart.app.ports.agent_chat import (
    AGENT_CHAT_PROTOCOL,
    AgentEffort,
    AgentResponseEvent,
    AgentRoute,
    AgentServiceError,
    AgentTokenPort,
    AgentTransportError,
)
from puripuly_heart.core.first_party_http import response_media_type


@dataclass(slots=True)
class _TurnRequests:
    turn_id: str
    count: int = 0


class AgentRelayTransport:
    def __init__(
        self, endpoint: str, token_owner: AgentTokenPort, *, allow_loopback_http: bool = False
    ) -> None:
        parsed = urlsplit(endpoint)
        if (
            not parsed.hostname
            or not (
                parsed.scheme == "https"
                or (allow_loopback_http and is_loopback_http(parsed))
            )
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path != "/v1/respond"
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(
                "Agent Chat requires HTTPS /v1/respond or explicit source loopback HTTP"
            )
        self._allow_loopback_http = allow_loopback_http
        self._endpoint = endpoint
        self._classification_endpoint = endpoint.removesuffix("/respond") + "/classify"
        self._support_endpoint = endpoint.removesuffix("/respond") + "/support"
        self._token_owner = token_owner
        self._turn_requests: _TurnRequests | None = None
        self._client: httpx.AsyncClient | None = None
        self._closed = False

    async def prepare(self) -> None:
        await self._token_owner.get_token()

    @staticmethod
    def _json_body(body: dict, limit: int) -> bytes:
        try:
            content = json.dumps(
                body, ensure_ascii=False, allow_nan=False, separators=(",", ":")
            ).encode("utf-8")
        except (TypeError, ValueError):
            pass
        else:
            if len(content) > limit:
                raise AgentServiceError("request_too_large")
            return content
        raise AgentTransportError("invalid_request")

    @asynccontextmanager
    async def _request(
        self,
        endpoint: str,
        operation: AgentEndpoint,
        *,
        content: bytes,
        headers: dict[str, str],
        turn_id: str | None = None,
    ) -> AsyncIterator[httpx.Response]:
        budget = None
        if turn_id is not None:
            if self._turn_requests is None or self._turn_requests.turn_id != turn_id:
                self._turn_requests = _TurnRequests(turn_id)
            budget = self._turn_requests
        for attempt in range(2):
            if budget is not None and budget.count >= 15:
                raise AgentServiceError("service_turn_exhausted")
            token = await self._token_owner.get_token()
            if self._closed:
                raise AgentTransportError()
            if self._client is None:
                self._client = agent_client(
                    operation, timeout=180, allow_loopback_http=self._allow_loopback_http
                )
            if budget is not None:
                if budget.count >= 15:
                    raise AgentServiceError("service_turn_exhausted")
                budget.count += 1
            async with post_stream(
                self._client,
                endpoint,
                operation,
                content=content,
                headers={**headers, "Authorization": f"Bearer {token}"},
            ) as response:
                if response.status_code == 200:
                    check_response_shape(response, operation)
                    yield response
                    return
                error = error_envelope(await read_json(response, operation))
                if (
                    attempt == 0
                    and response.status_code == 401
                    and error.get("code") == "service_authentication_failed"
                ):
                    self._token_owner.invalidate(token)
                else:
                    raise self._response_error(response, operation, error)

    async def classify(self, *, input: list[dict], turn_id: str) -> dict:
        return await self._post(
            self._classification_endpoint,
            {"agent_protocol": AGENT_CHAT_PROTOCOL, "input": input, "turn_id": turn_id},
        )

    async def respond(
        self,
        *,
        input: list[dict],
        turn_id: str,
        effort: AgentEffort,
        route: AgentRoute,
        allow_tools: bool,
    ) -> AsyncGenerator[AgentResponseEvent, None]:
        if effort not in ("low", "medium", "high"):
            raise AgentTransportError("invalid_effort")
        content = self._json_body(
            {
                "agent_protocol": AGENT_CHAT_PROTOCOL,
                "input": input,
                "turn_id": turn_id,
                "effort": effort,
                "route": route,
                "allow_tools": allow_tools,
            },
            1024 * 1024,
        )
        operation = "relay_response"
        async with self._request(
            self._endpoint,
            operation,
            content=content,
            turn_id=turn_id,
            headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        ) as response:
            if response_media_type(response) != "text/event-stream":
                await read_json(response, operation)
                raise response_failure(response, operation, "stream_invalid")
            final = None
            data = []
            async with aclosing(response_lines(response, operation)) as lines:
                async for line in lines:
                    if line.startswith("data:"):
                        data.append(line[5:].removeprefix(" "))
                    elif not line and data:
                        event = self._stream_event(response, data)
                        data.clear()
                        if final is not None:
                            raise response_failure(response, operation, "stream_invalid")
                        if event["type"] == "completed":
                            final = event
                        else:
                            yield event
                    elif line.lstrip().startswith("<"):
                        check_response_shape(response, operation, body=line[:512].encode())
            if data or final is None:
                raise response_failure(
                    response, operation, "stream_interrupted", category="network"
                )
            yield final

    @staticmethod
    def _stream_event(response: httpx.Response, data: list[str]) -> AgentResponseEvent:
        operation = "relay_response"
        try:
            event = json.loads("\n".join(data))
        except (ValueError, RecursionError):
            event = None
        if isinstance(event, dict):
            kind = event.get("type")
            if kind == "error":
                code = error_envelope(event).get("code")
                if code == "service_shutdown":
                    raise AgentServiceError(
                        "service_shutdown",
                        evidence=response_details(response, operation, "service_unavailable"),
                    )
                if code == "upstream_unavailable":
                    raise response_failure(
                        response, operation, "stream_interrupted", category="service_unavailable"
                    )
            elif (
                kind == "text_delta"
                and set(event) == {"type", "text"}
                and isinstance(event["text"], str)
            ):
                return event
            elif (
                kind == "completed"
                and set(event) == {"type", "response"}
                and isinstance(event["response"], dict)
                and isinstance(event["response"].get("output"), list)
            ):
                return event
        raise response_failure(response, operation, "stream_invalid")

    async def upload_support(
        self,
        *,
        bundle: bytes,
        ticket: str,
        app_version: str,
        diagnostic_codes: tuple[str, ...],
    ) -> dict:
        if (
            not bundle
            or len(bundle) > 25_000_000
            or re.fullmatch(r"PP-[0-9A-F]{16}", ticket) is None
            or re.fullmatch(r"[A-Za-z0-9.+_-]{1,64}", app_version) is None
            or len(diagnostic_codes) > 10
            or any(
                re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,95}", code) is None
                for code in diagnostic_codes
            )
        ):
            raise AgentTransportError("invalid_support_report")
        operation = "relay_support"
        async with self._request(
            self._support_endpoint,
            operation,
            content=bundle,
            headers={
                "Content-Type": "application/gzip",
                "X-Agent-Protocol": "3",
                "X-Support-Ticket": ticket,
                "X-App-Version": app_version,
                "X-Diagnostic-Codes": json.dumps(diagnostic_codes, separators=(",", ":")),
            },
        ) as response:
            result = await read_json(response, operation)
            if result is None or result.get("ticket") != ticket:
                raise response_failure(response, operation, "invalid_support_receipt")
        return result

    async def _post(self, endpoint: str, body: dict) -> dict:
        content = self._json_body(body, 64 * 1024)
        operation = "relay_classify"
        async with self._request(
            endpoint,
            operation,
            content=content,
            turn_id=body["turn_id"],
            headers={"Content-Type": "application/json"},
        ) as response:
            result = await read_json(response, operation)
            if result is None:
                raise response_failure(response, operation, "invalid_response")
        return result

    @staticmethod
    def _response_error(
        response: httpx.Response, operation: AgentEndpoint, error: dict
    ) -> AgentTransportError:
        fields = {"response_shape": "public_error_envelope"} if error else {}
        status = response.status_code
        code = error.get("code")
        match status, code:
            case (401 | 503, "service_authentication_failed"):
                service_code, category = "service_authentication_failed", "auth"
            case (400, "unsupported_agent_protocol"):
                service_code, category = "unsupported_protocol", "invalid_response"
            case (413, "request_too_large"):
                service_code, category = "request_too_large", "invalid_response"
            case (429, "service_busy" | "service_turn_exhausted"):
                service_code, category = code, "rate_limit"
            case (503, "service_shutdown"):
                service_code, category = "service_shutdown", "service_unavailable"
            case (503 | 429, "service_allowance_exhausted"):
                service_code, category = "service_allowance_exhausted", "quota"
            case (429, _):
                service_code, category = "service_rate_limited", "rate_limit"
            case _:
                return response_failure(response, operation, "unexpected_http_status", fields=fields)
        return AgentServiceError(
            service_code,
            evidence=response_details(response, operation, category, fields=fields),
        )

    async def close(self) -> None:
        self._closed = True
        try:
            if self._client is not None:
                await close_agent_client(self._client, "relay_response")
        finally:
            await self._token_owner.close()
