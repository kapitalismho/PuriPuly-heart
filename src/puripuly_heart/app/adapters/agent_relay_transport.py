from __future__ import annotations

import json
import re
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from urllib.parse import urlsplit

import httpx

from puripuly_heart.app.ports.agent_chat import (
    AGENT_CHAT_PROTOCOL,
    AgentEffort,
    AgentResponseEvent,
    AgentRoute,
    AgentServiceError,
    AgentTokenPort,
    AgentTransportError,
)


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
                or (
                    allow_loopback_http
                    and parsed.scheme == "http"
                    and parsed.hostname in {"127.0.0.1", "::1"}
                )
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
        self._endpoint = endpoint
        self._classification_endpoint = endpoint.removesuffix("/respond") + "/classify"
        self._support_endpoint = endpoint.removesuffix("/respond") + "/support"
        self._token_owner = token_owner
        self._turn_requests: _TurnRequests | None = None
        self._client = httpx.AsyncClient(trust_env=False, follow_redirects=False, timeout=180)

    async def prepare(self) -> None:
        await self._token_owner.get_token()

    @staticmethod
    def _json_body(body: dict, limit: int) -> bytes:
        try:
            content = json.dumps(
                body, ensure_ascii=False, allow_nan=False, separators=(",", ":")
            ).encode("utf-8")
        except (TypeError, ValueError) as error:
            raise AgentTransportError("invalid_request") from error
        if len(content) > limit:
            raise AgentServiceError("request_too_large")
        return content

    @asynccontextmanager
    async def _request(
        self,
        endpoint: str,
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
            if budget is not None:
                if budget.count >= 15:
                    raise AgentServiceError("service_turn_exhausted")
                budget.count += 1
            async with self._client.stream(
                "POST",
                endpoint,
                content=content,
                headers={**headers, "Authorization": f"Bearer {token}"},
            ) as response:
                if response.status_code == 401 and attempt == 0:
                    self._token_owner.invalidate(token)
                else:
                    yield response
                    return

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
        if effort not in ("low", "high", "xhigh"):
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
        completed = False
        try:
            async with self._request(
                self._endpoint,
                content=content,
                turn_id=turn_id,
                headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
            ) as response:
                if response.status_code != 200:
                    await response.aread()
                    self._raise_response_error(response)
                if response.headers.get("content-type", "").split(";")[0] != "text/event-stream":
                    raise AgentTransportError("invalid_response")
                data = []
                async for line in response.aiter_lines():
                    if line.startswith("data:"):
                        data.append(line[5:].removeprefix(" "))
                    elif not line and data:
                        event = json.loads("\n".join(data))
                        data.clear()
                        if not isinstance(event, dict) or completed:
                            raise AgentTransportError("invalid_response")
                        kind = event.get("type")
                        if kind == "error":
                            error = event.get("error")
                            code = error.get("code") if isinstance(error, dict) else None
                            self._raise_code(code)
                        if (
                            kind == "text_delta"
                            and set(event) == {"type", "text"}
                            and isinstance(event["text"], str)
                        ):
                            yield event
                        elif (
                            kind == "completed"
                            and set(event) == {"type", "response"}
                            and isinstance(event["response"], dict)
                        ):
                            completed = True
                            yield event
                        else:
                            raise AgentTransportError("invalid_response")
                if data or not completed:
                    raise AgentTransportError("invalid_response")
        except (httpx.HTTPError, ValueError) as from_error:
            raise AgentTransportError() from from_error

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
        try:
            async with self._request(
                self._support_endpoint,
                content=bundle,
                headers={
                    "Content-Type": "application/gzip",
                    "X-Agent-Protocol": "3",
                    "X-Support-Ticket": ticket,
                    "X-App-Version": app_version,
                    "X-Diagnostic-Codes": json.dumps(diagnostic_codes, separators=(",", ":")),
                },
            ) as response:
                await response.aread()
                if response.status_code != 200:
                    raise AgentTransportError("support_upload_failed")
                result = response.json()
        except (httpx.HTTPError, ValueError) as from_error:
            raise AgentTransportError("support_upload_failed") from from_error
        if not isinstance(result, dict) or result.get("ticket") != ticket:
            raise AgentTransportError("invalid_support_receipt")
        return result

    async def _post(self, endpoint: str, body: dict) -> dict:
        content = self._json_body(body, 64 * 1024)
        try:
            async with self._request(
                endpoint,
                content=content,
                turn_id=body["turn_id"],
                headers={"Content-Type": "application/json"},
            ) as response:
                await response.aread()
                if response.status_code != 200:
                    self._raise_response_error(response)
                result = response.json()
        except (httpx.HTTPError, ValueError) as from_error:
            raise AgentTransportError() from from_error
        if not isinstance(result, dict):
            raise AgentTransportError("invalid_response")
        return result

    @staticmethod
    def _raise_code(code: object) -> None:
        if code == "unsupported_agent_protocol":
            code = "unsupported_protocol"
        if code in (
            "service_authentication_failed",
            "service_authentication_unavailable",
            "service_rate_limited",
            "service_turn_exhausted",
            "service_busy",
            "request_too_large",
            "service_allowance_exhausted",
            "service_shutdown",
            "unsupported_protocol",
        ):
            raise AgentServiceError(code)
        raise AgentTransportError("relay_request_failed")

    def _raise_response_error(self, response: httpx.Response) -> None:
        if response.status_code in (401, 403):
            raise AgentServiceError("service_authentication_failed")
        try:
            payload = response.json()
        except ValueError:
            payload = None
        error = payload.get("error") if isinstance(payload, dict) else None
        self._raise_code(error.get("code") if isinstance(error, dict) else None)

    async def close(self) -> None:
        try:
            await self._client.aclose()
        finally:
            await self._token_owner.close()
