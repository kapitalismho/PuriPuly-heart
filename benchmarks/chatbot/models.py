from __future__ import annotations

import asyncio
import copy
import json
import os
import re
from time import monotonic
from typing import Any

import httpx
from websockets.exceptions import ConnectionClosed, InvalidStatus

from puripuly_heart.app.services.chatgpt_account import ChatGptAccountOwner
from puripuly_heart.core import network_clients
from puripuly_heart.core.chatgpt.oauth import ChatGptAuthError, ChatGptReauthRequired
from puripuly_heart.core.chatgpt.session import ChatGptSession
from puripuly_heart.core.storage.secrets import KeyringSecretStore

JUDGE_MODEL = "gpt-6.1-sol"
JUDGE_EFFORT = "xhigh"
_MODELS = {"user": ("gpt-6.1-terra", "medium"), "judge": (JUDGE_MODEL, JUDGE_EFFORT)}
_ROLE_TIMEOUTS = {"user": 180, "judge": 300}
_HTTP_URL = "https://api.openai.com/v1/responses"
_WS_URL = "wss://api.openai.com/v1/responses"
_BENCHMARK_KEYRING_SERVICE = "puripuly-heart-agent-simulator"
_AUTH_FAILURE_CODES = {
    "invalid_grant",
    "invalid_refresh_token",
    "token_expired",
    "refresh_token_expired",
    "refresh_token_invalidated",
    "refresh_token_reused",
    "invalid_client",
    "signed_out",
    "plan_scope_missing",
    "token_response_invalid",
    "refresh_network_error",
    "refresh_failed",
    "access_denied",
    "invalid_request",
    "invalid_scope",
    "unauthorized_client",
    "server_error",
    "temporarily_unavailable",
}
_ACCESS_CODES = {
    "subscription_sharing_usage_limit_exceeded",
    "subscription_sharing_user_not_eligible",
    "subscription_sharing_invalid_user",
    "subscription_sharing_usage_unavailable",
    "subscription_sharing_user_unavailable",
    "subscription_sharing_unsupported_capability",
    "subscription_sharing_route_not_supported",
    "invalid_api_key",
    "authentication_error",
    "permission_denied",
    "model_not_found",
    "insufficient_quota",
    "rate_limit_exceeded",
}
_SIGN_IN_CODES = {
    "cancelled",
    "timeout",
    "access_denied",
    "authorization_failed",
    "plan_scope_missing",
    "loopback_unavailable",
    "network_error",
    "in_progress",
    "invalid_id_token",
    "registration_incomplete",
    "state_mismatch",
    "client_mismatch",
}
_SUBSCRIPTION_DENIAL_CODES = _ACCESS_CODES | {
    "subscription_unavailable",
    "authentication_failed",
    "reauthentication_required",
    "access_refused",
}
_FALLBACK_CODES = _SUBSCRIPTION_DENIAL_CODES | {
    "network_error",
    "timeout",
    "http_unavailable",
}


def _cost(model: str, path: str, usage: object) -> tuple[float | None, dict]:
    if path == "subscription_ws":
        return None, {"kind": "subscription_billing_separate"}
    rates = (1.0, 0.05, 1.25, 5.0) if model == "gpt-6.1-terra" else (2.0, 0.1, 2.5, 10.0)
    basis = {
        "kind": "standard_estimate_not_invoice",
        "currency": "USD",
        "as_of": "2026-10-30",
        "source": f"https://developers.openai.com/api/docs/models/{model}",
        "usd_per_million_tokens": dict(
            zip(("input", "cached", "cache_write_30min", "output"), rates)
        ),
        "usage_status": "missing_or_invalid",
    }
    if not isinstance(usage, dict):
        return None, basis
    details = usage.get("input_tokens_details", {})
    if not isinstance(details, dict):
        return None, basis
    input_tokens, output = usage.get("input_tokens"), usage.get("output_tokens")
    cached, written = details.get("cached_tokens", 0), details.get("cache_write_tokens", 0)
    counts = (input_tokens, output, cached, written)
    if any(type(count) is not int or count < 0 or count > 2**53 - 1 for count in counts):
        return None, basis
    if cached + written > input_tokens:
        return None, basis
    total = usage.get("total_tokens")
    if total is not None and (type(total) is not int or total != input_tokens + output):
        return None, basis
    output_details = usage.get("output_tokens_details", {})
    if not isinstance(output_details, dict):
        return None, basis
    reasoning = output_details.get("reasoning_tokens", 0)
    if type(reasoning) is not int or not 0 <= reasoning <= output:
        return None, basis
    long_context = input_tokens > 272000
    input_multiplier, output_multiplier = (2, 1.5) if long_context else (1, 1)
    cost = (
        ((input_tokens - cached - written) * rates[0] + cached * rates[1] + written * rates[2])
        * input_multiplier
        + output * rates[3] * output_multiplier
    ) / 1_000_000
    basis.update(
        {
            "usage_status": "valid",
            "long_context": long_context,
            "input_multiplier": input_multiplier,
            "output_multiplier": output_multiplier,
        }
    )
    return cost, basis


class SimulatorModelError(RuntimeError):
    def __init__(self, code: str, evidence: dict[str, Any]) -> None:
        self.code = code
        self.evidence = copy.deepcopy(evidence)
        super().__init__(code)


def _usage(value: object) -> dict | None:
    if not isinstance(value, dict):
        return None
    result = {}
    for key, item in value.items():
        if not isinstance(key, str) or not re.fullmatch(r"[a-z_]{1,64}", key):
            continue
        if isinstance(item, dict):
            result[key] = _usage(item)
        elif item is None or type(item) in {int, float, bool}:
            result[key] = item
        else:
            result[key] = "[redacted]"
    return result


def _returned_model(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str) and re.fullmatch(r"gpt-[a-z0-9.-]{1,80}", value):
        return value
    return "[redacted]"


def _reject_constant(value: str) -> None:
    raise ValueError("non-finite JSON")


def _json_type(value: object) -> str:
    return {
        type(None): "null",
        bool: "boolean",
        int: "number",
        float: "number",
        str: "string",
        list: "array",
        dict: "object",
    }.get(type(value), "other")


def _shape_label(value: dict, key: str, allowed: tuple[str, ...]) -> str:
    if key not in value:
        return "missing"
    label = value[key]
    return label if isinstance(label, str) and label in allowed else "invalid"


def _handshake_diagnostics(response: Any) -> dict:
    diagnostics: dict[str, Any] = {
        "response_content_kind": "unknown",
        "error_code": "unknown",
        "error_type": "unknown",
    }
    body = response.body
    if not body:
        diagnostics["response_content_kind"] = "empty"
    elif isinstance(body, (bytes, bytearray)) and len(body) <= 65536:
        try:
            document = json.loads(body, parse_constant=_reject_constant)
        except ValueError, UnicodeError, RecursionError:
            try:
                text = body.decode("utf-8")
            except UnicodeError:
                diagnostics["response_content_kind"] = "binary"
            else:
                diagnostics["response_content_kind"] = (
                    "html"
                    if text.lstrip().lower().startswith(("<html", "<!doctype html"))
                    else "text"
                )
        else:
            diagnostics["response_content_kind"] = "json"
            error = document.get("error") if isinstance(document, dict) else None
            if isinstance(error, dict):
                code, kind = error.get("code"), error.get("type")
                if (
                    isinstance(code, str)
                    and code in _SUBSCRIPTION_DENIAL_CODES | _AUTH_FAILURE_CODES
                ):
                    diagnostics["error_code"] = code
                if isinstance(kind, str) and kind in {
                    "invalid_request_error",
                    "authentication_error",
                    "permission_error",
                    "not_found_error",
                    "rate_limit_error",
                    "server_error",
                    "insufficient_quota",
                }:
                    diagnostics["error_type"] = kind
    for header, field, pattern in (
        ("x-request-id", "request_id", r"[A-Za-z0-9_-]{1,128}"),
        ("retry-after", "retry_after_s", r"[0-9]{1,6}(?:\.[0-9]{1,3})?"),
    ):
        try:
            value = response.headers.get(header)
        except Exception:
            continue
        if not isinstance(value, str) or not re.fullmatch(pattern, value):
            continue
        if field == "retry_after_s":
            number = float(value)
            if number <= 86400:
                diagnostics[field] = number
        else:
            diagnostics[field] = value
    return diagnostics


class SubscriptionConnectGate:
    def __init__(self, min_interval_s: float = 1.25) -> None:
        self._min_interval_s = min_interval_s
        self._lock = asyncio.Lock()
        self._last_start: float | None = None

    async def wait(self) -> None:
        async with self._lock:
            while self._last_start is not None:
                remaining = self._min_interval_s - (monotonic() - self._last_start)
                if remaining <= 0:
                    break
                await asyncio.sleep(remaining)
            self._last_start = monotonic()


class SimulatorModels:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        sign_in: bool = False,
        reauthenticate: bool = False,
        subscription_gate: SubscriptionConnectGate | None = None,
    ) -> None:
        self._api_key = (
            api_key if api_key is not None else os.environ.get("OPENAI_API_KEY")
        ) or None
        self._sign_in = sign_in or reauthenticate
        self._reauthenticate = reauthenticate
        self._preparation_error: str | None = None
        self._prepared = False
        self._closed = False
        self._store: KeyringSecretStore | None = None
        self._session: ChatGptSession | None = None
        self._account: ChatGptAccountOwner | None = None
        self._http: httpx.AsyncClient | None = None
        self._subscription_gate = (
            subscription_gate if subscription_gate is not None else SubscriptionConnectGate()
        )
        self._subscription_locks = {role: asyncio.Lock() for role in _MODELS}
        self._subscription_sockets: dict[str, tuple[int, Any]] = {}
        self._connection_sequence = 0
        self._subscription_disabled: set[str] = set()
        self._attempts: list[dict] = []

    @property
    def attempts(self) -> list[dict]:
        return copy.deepcopy(self._attempts)

    @property
    def access_evidence(self) -> list[dict]:
        return self.attempts

    def _error(self, code: str, evidence: dict) -> SimulatorModelError:
        return SimulatorModelError(code, {**evidence, "attempts": self.attempts})

    async def prepare(self) -> None:
        if self._closed:
            raise self._error("closed", {})
        if self._preparation_error:
            raise self._error(self._preparation_error, {})
        if not self._prepared:
            self._prepared = True
            if self._sign_in:
                self._store = KeyringSecretStore(service_name=_BENCHMARK_KEYRING_SERVICE)
                self._session = ChatGptSession(secret_store=lambda: self._store)
                self._account = ChatGptAccountOwner(session=self._session)
                status = self._session.status()
                if status.signed_in is None:
                    self._preparation_error = "keyring_unavailable"
                    self._attempts.append(
                        {
                            "access_path": "subscription_ws",
                            "phase": "sign_in",
                            "status": "not_available",
                            "code": "keyring_unavailable",
                        }
                    )
                    raise self._error("keyring_unavailable", {})
                auth_failure_code = None
                result = None
                try:
                    if self._reauthenticate:
                        result = await self._account.connect()
                        succeeded, code = result.succeeded, result.failure_code
                    elif status.signed_in:
                        succeeded, code = True, "reused_login"
                    else:
                        try:
                            saved_client_id = self._session.client_id()
                        except Exception:
                            self._preparation_error = "keyring_unavailable"
                            raise
                        if saved_client_id:
                            succeeded, code = False, "reauthentication_required"
                        else:
                            result = await self._account.connect()
                            succeeded, code = result.succeeded, result.failure_code
                except ChatGptAuthError as exc:
                    succeeded = False
                    code = (
                        "reauthentication_required"
                        if isinstance(exc, ChatGptReauthRequired)
                        else "authentication_failed"
                    )
                    auth_failure_code = (
                        exc.code if exc.code in _AUTH_FAILURE_CODES else "authentication_failed"
                    )
                    if exc.diagnostics and exc.diagnostics.operation in {
                        "authorization_request",
                        "credential_store",
                        "secret_read",
                        "refresh_store",
                    }:
                        self._preparation_error = "keyring_unavailable"
                    if exc.code not in _AUTH_FAILURE_CODES | _SIGN_IN_CODES:
                        self._preparation_error = self._preparation_error or "sign_in_failed"
                except Exception:
                    succeeded, code = False, "sign_in_failed"
                    self._preparation_error = self._preparation_error or "sign_in_failed"
                diagnostics = getattr(result, "diagnostics", None)
                if code == "credential_store_failed" or (
                    diagnostics
                    and diagnostics.operation
                    in {
                        "secret_read",
                        "authorization_request",
                        "credential_store",
                    }
                ):
                    self._preparation_error = "keyring_unavailable"
                elif not succeeded and code not in _SIGN_IN_CODES | _AUTH_FAILURE_CODES | {
                    "reauthentication_required",
                    "authentication_failed",
                }:
                    self._preparation_error = self._preparation_error or "sign_in_failed"
                self._attempts.append(
                    {
                        "access_path": "subscription_ws",
                        "phase": "sign_in",
                        "status": "available" if succeeded else "not_available",
                        "code": (
                            ("reused_login" if code == "reused_login" else "signed_in")
                            if succeeded
                            else (
                                code
                                if code
                                in _SIGN_IN_CODES
                                | _AUTH_FAILURE_CODES
                                | {
                                    "reauthentication_required",
                                    "authentication_failed",
                                    "credential_store_failed",
                                }
                                else "sign_in_failed"
                            )
                        ),
                        **({"auth_failure_code": auth_failure_code} if auth_failure_code else {}),
                    }
                )
                if self._preparation_error:
                    raise self._error(self._preparation_error, {})
                if not succeeded:
                    self._subscription_disabled.update(_MODELS)
            else:
                self._subscription_disabled.update(_MODELS)
                self._attempts.append(
                    {
                        "access_path": "subscription_ws",
                        "phase": "sign_in",
                        "status": "not_available",
                        "code": "subscription_not_attempted",
                    }
                )
        if self._subscription_disabled == set(_MODELS) and not self._api_key:
            code = self._attempts[-1]["code"] if self._sign_in else "access_unavailable"
            raise self._error(code, {})

    async def probe(self) -> dict:
        schema = {
            "type": "object",
            "properties": {"ok": {"type": "boolean"}},
            "required": ["ok"],
            "additionalProperties": False,
        }
        results = {}
        for role in _MODELS:
            try:
                result = await self.complete(
                    role=role,
                    instructions='Return the JSON object {"ok": true}.',
                    payload={"purpose": "synthetic model access probe"},
                    schema=schema,
                )
                if result["value"] != {"ok": True}:
                    raise self._error("probe_result_invalid", {"model": result["model"]})
                results[role] = {"status": "completed", **result}
            except SimulatorModelError as exc:
                results[role] = {"status": "failed", "code": exc.code, "evidence": exc.evidence}
        return {"roles": results, "attempts": self.attempts}

    async def complete(self, *, role: str, instructions: str, payload: dict, schema: dict) -> dict:
        if role not in _MODELS:
            raise self._error("invalid_role", {})
        if not self._sign_in or role in self._subscription_disabled:
            return await self._complete(
                role=role, instructions=instructions, payload=payload, schema=schema
            )
        async with self._subscription_locks[role]:
            try:
                return await self._complete(
                    role=role, instructions=instructions, payload=payload, schema=schema
                )
            except BaseException:
                await self._close_subscription(role)
                raise

    async def _complete(self, *, role: str, instructions: str, payload: dict, schema: dict) -> dict:
        await self.prepare()
        model, effort = _MODELS[role]
        try:
            body = {
                "model": model,
                "reasoning": {"effort": effort},
                "store": False,
                "max_output_tokens": 25000,
                "instructions": instructions,
                "input": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": json.dumps(payload, ensure_ascii=False, allow_nan=False),
                            }
                        ],
                    }
                ],
                "text": {
                    "format": {
                        "type": "json_schema",
                        "name": "simulator_result",
                        "strict": True,
                        "schema": schema,
                    }
                },
            }
            json.dumps(body, allow_nan=False)
        except TypeError, ValueError:
            raise self._error("invalid_request", {}) from None
        paths = ["subscription_ws"] if role not in self._subscription_disabled else []
        if self._api_key:
            paths.append("api_key_http")
        for path in paths:
            evidence = {
                "role": role,
                "access_path": path,
                "model": model,
                "requested_model": model,
                "returned_model": None,
                "effort": effort,
                "timeout_s": _ROLE_TIMEOUTS[role],
                "usage": None,
                "cost_usd": None,
                "cost_basis": _cost(model, path, None)[1],
            }
            if path == "subscription_ws":
                evidence["subscription_wire"] = {
                    "connection_action": "not_started",
                    "connection_sequence": None,
                    "handshake_status": "not_started",
                    "connection_admission_wait_s": 0.0,
                    "provider_elapsed_s": None,
                    "event_counts": {},
                    "finalized_items": [],
                    "output_source": "terminal",
                }
            try:
                started = monotonic()
                try:
                    async with asyncio.timeout(_ROLE_TIMEOUTS[role]):
                        response = (
                            await self._subscription(
                                body, role=role, wire=evidence["subscription_wire"]
                            )
                            if path == "subscription_ws"
                            else await self._api(body, timeout_s=_ROLE_TIMEOUTS[role])
                        )
                finally:
                    evidence["transport_elapsed_s"] = monotonic() - started
                if path == "subscription_ws":
                    evidence["subscription_wire"] = response.pop("_subscription_wire")
                evidence.update(
                    {
                        "returned_model": _returned_model(response.get("model")),
                        "usage": _usage(response.get("usage")),
                        "terminal_status": (
                            response.get("status")
                            if response.get("status")
                            in {
                                "completed",
                                "failed",
                                "incomplete",
                                "cancelled",
                                "queued",
                                "in_progress",
                            }
                            else "unknown"
                        ),
                    }
                )
                evidence["cost_usd"], evidence["cost_basis"] = _cost(
                    model, path, response.get("usage")
                )
                if response.get("model") is not None and response.get("model") != model:
                    raise SimulatorModelError("model_mismatch", evidence)
                status = response.get("status")
                if status != "completed":
                    code = self._response_code(response)
                    raise SimulatorModelError(
                        (
                            code
                            if code in _ACCESS_CODES
                            else (
                                "response_incomplete"
                                if status == "incomplete"
                                else "response_not_completed"
                            )
                        ),
                        evidence,
                    )
                if response.get("model") is None:
                    raise SimulatorModelError("model_mismatch", evidence)
                value = self._value(response)
            except SimulatorModelError as exc:
                evidence.update(exc.evidence)
                code = exc.code
            except InvalidStatus as exc:
                status = exc.response.status_code
                evidence["http_status"] = status
                evidence["subscription_wire"]["handshake_status"] = "rejected"
                evidence["subscription_wire"]["handshake_diagnostics"] = _handshake_diagnostics(
                    exc.response
                )
                code = self._http_code(status)
            except TimeoutError, httpx.TimeoutException:
                code = "timeout"
            except ChatGptAuthError as exc:
                evidence["auth_failure_code"] = (
                    exc.code if exc.code in _AUTH_FAILURE_CODES else "authentication_failed"
                )
                if exc.diagnostics and exc.diagnostics.operation in {
                    "secret_read",
                    "refresh_store",
                }:
                    self._preparation_error = code = "keyring_unavailable"
                elif exc.code == "refresh_network_error":
                    code = "network_error"
                elif not isinstance(exc, ChatGptReauthRequired) and (
                    (exc.status_code is not None and exc.status_code >= 500)
                    or exc.code in {"server_error", "temporarily_unavailable"}
                ):
                    code = "http_unavailable"
                    if exc.status_code is not None:
                        evidence["http_status"] = exc.status_code
                else:
                    code = (
                        "reauthentication_required"
                        if isinstance(exc, ChatGptReauthRequired)
                        else "authentication_failed"
                    )
            except httpx.HTTPError, ConnectionClosed, OSError:
                code = "network_error"
            except Exception:
                code = "client_error"
            else:
                self._attempts.append({**evidence, "status": "completed", "code": "completed"})
                return {"value": value, **evidence}
            if path == "subscription_ws":
                await self._close_subscription(role)
            self._attempts.append({**evidence, "status": "failed", "code": code})
            if path == "subscription_ws" and code in _FALLBACK_CODES:
                if code in _SUBSCRIPTION_DENIAL_CODES:
                    self._subscription_disabled.add(role)
                if self._api_key:
                    continue
            raise self._error(code, evidence) from None
        raise self._error("access_unavailable", {})

    @staticmethod
    def _response_code(response: dict) -> str | None:
        error = response.get("error")
        code = error.get("code") if isinstance(error, dict) else None
        return code if isinstance(code, str) and code in _ACCESS_CODES else None

    @staticmethod
    def _http_code(status: int) -> str:
        if status == 401:
            return "authentication_failed"
        if status in {402, 403, 404, 429}:
            return "access_refused"
        return "http_unavailable" if status >= 500 else "http_error"

    async def _api(self, body: dict, *, timeout_s: int) -> dict:
        if self._http is None:
            self._http = network_clients.external_async_client(timeout=None, follow_redirects=False)
        response = await self._http.post(
            _HTTP_URL,
            headers={"Authorization": f"Bearer {self._api_key}"},
            json={**body, "stream": False},
            timeout=timeout_s,
        )
        try:
            result = json.loads(response.content, parse_constant=_reject_constant)
        except ValueError, UnicodeError:
            raise SimulatorModelError(
                "malformed_response", {"http_status": response.status_code}
            ) from None
        if not isinstance(result, dict):
            raise SimulatorModelError("malformed_response", {"http_status": response.status_code})
        if not response.is_success:
            cost, basis = _cost(body["model"], "api_key_http", result.get("usage"))
            raise SimulatorModelError(
                self._response_code(result) or self._http_code(response.status_code),
                {
                    "http_status": response.status_code,
                    "usage": _usage(result.get("usage")),
                    "returned_model": _returned_model(result.get("model")),
                    "cost_usd": cost,
                    "cost_basis": basis,
                },
            )
        return result

    async def _subscription(self, body: dict, *, role: str, wire: dict) -> dict:
        if self._session is None:
            raise SimulatorModelError("subscription_unavailable", {})
        provider_started = None
        try:
            connection = self._subscription_sockets.get(role)
            if connection is None:
                token = await self._session.access_token()
                connector = network_clients.external_websocket_connect(
                    _WS_URL,
                    additional_headers={"Authorization": f"Bearer {token}"},
                    open_timeout=15,
                    max_size=4 * 1024 * 1024,
                    ping_interval=20,
                    ping_timeout=20,
                )
                admission_started = monotonic()
                try:
                    await self._subscription_gate.wait()
                finally:
                    wire["connection_admission_wait_s"] = monotonic() - admission_started
                if self._closed:
                    raise SimulatorModelError("closed", {})
                self._connection_sequence += 1
                sequence = self._connection_sequence
                wire.update(
                    {
                        "connection_action": "open",
                        "connection_sequence": sequence,
                        "handshake_status": "started",
                    }
                )
                provider_started = monotonic()
                socket = await connector
                self._subscription_sockets[role] = (sequence, socket)
                wire["handshake_status"] = "opened"
                if self._closed:
                    raise SimulatorModelError("closed", {})
            else:
                sequence, socket = connection
                wire.update(
                    {
                        "connection_action": "reuse",
                        "connection_sequence": sequence,
                        "handshake_status": "not_attempted",
                    }
                )
                provider_started = monotonic()
            await socket.send(
                json.dumps(
                    {"type": "response.create", **body, "previous_response_id": None},
                    allow_nan=False,
                )
            )
            finalized_items: dict[int, dict] = {}
            invalid_finalized_items = False
            while True:
                try:
                    event = json.loads(await socket.recv(), parse_constant=_reject_constant)
                except ValueError, UnicodeError:
                    raise SimulatorModelError(
                        "malformed_response", {"subscription_wire": wire}
                    ) from None
                if not isinstance(event, dict):
                    raise SimulatorModelError("malformed_response", {"subscription_wire": wire})
                kind = event.get("type")
                label = (
                    kind
                    if kind
                    in (
                        "response.created",
                        "response.in_progress",
                        "response.output_item.added",
                        "response.output_item.done",
                        "response.content_part.added",
                        "response.content_part.done",
                        "response.output_text.delta",
                        "response.output_text.done",
                        "response.completed",
                        "response.failed",
                        "response.incomplete",
                        "error",
                    )
                    else "other"
                )
                wire["event_counts"][label] = wire["event_counts"].get(label, 0) + 1
                if kind == "response.output_item.done":
                    index, item = event.get("output_index"), event.get("item")
                    valid_index = type(index) is int and index >= 0
                    shape = {
                        "output_index": index if valid_index else "invalid",
                        "item_type": _json_type(item),
                    }
                    if isinstance(item, dict):
                        shape.update(
                            {
                                "type": _shape_label(item, "type", ("message", "reasoning")),
                                "role": _shape_label(item, "role", ("assistant",)),
                                "status": _shape_label(
                                    item, "status", ("completed", "in_progress", "incomplete")
                                ),
                                "content_type": _json_type(item.get("content")),
                            }
                        )
                    wire["finalized_items"].append(shape)
                    if not valid_index or index in finalized_items or not isinstance(item, dict):
                        invalid_finalized_items = True
                    else:
                        finalized_items[index] = item
                if kind in {"response.completed", "response.failed", "response.incomplete"}:
                    response = event.get("response")
                    if not isinstance(response, dict):
                        raise SimulatorModelError("malformed_response", {"subscription_wire": wire})
                    if response.get("status") != kind.removeprefix("response."):
                        raise SimulatorModelError(
                            "malformed_response",
                            {
                                "usage": _usage(response.get("usage")),
                                "returned_model": _returned_model(response.get("model")),
                                "subscription_wire": wire,
                            },
                        )
                    output = response.get("output")
                    wire["terminal_output_present"] = "output" in response
                    wire["terminal_output_count"] = (
                        len(output) if isinstance(output, list) else None
                    )
                    # Only finalized items can fill an omitted terminal output;
                    # deltas never establish completion or replace terminal content.
                    if (
                        kind == "response.completed"
                        and ("output" not in response or output == [])
                        and (finalized_items or invalid_finalized_items)
                    ):
                        if invalid_finalized_items or sorted(finalized_items) != list(
                            range(len(finalized_items))
                        ):
                            raise SimulatorModelError(
                                "malformed_response",
                                {
                                    "usage": _usage(response.get("usage")),
                                    "returned_model": _returned_model(response.get("model")),
                                    "terminal_status": response["status"],
                                    "subscription_wire": wire,
                                },
                            )
                        response["output"] = [
                            finalized_items[index] for index in range(len(finalized_items))
                        ]
                        wire["output_source"] = "finalized_items"
                    response["_subscription_wire"] = wire
                    return response
                if kind == "error":
                    status = event.get("status")
                    raise SimulatorModelError(
                        self._response_code(event)
                        or (
                            self._http_code(status) if type(status) is int else "subscription_error"
                        ),
                        {
                            "http_status": status if type(status) is int else None,
                            "usage": _usage(event.get("usage")),
                            "returned_model": _returned_model(event.get("model")),
                            "subscription_wire": wire,
                        },
                    )
        except BaseException:
            if provider_started is not None:
                wire["provider_elapsed_s"] = monotonic() - provider_started
            if wire["handshake_status"] == "started":
                wire["handshake_status"] = "failed"
            await self._close_subscription(role)
            raise
        finally:
            if provider_started is not None and wire["provider_elapsed_s"] is None:
                wire["provider_elapsed_s"] = monotonic() - provider_started

    async def _close_subscription(self, role: str) -> bool:
        connection = self._subscription_sockets.pop(role, None)
        if connection is None:
            return True
        try:
            await connection[1].close()
        except Exception:
            return False
        return True

    @staticmethod
    def _value(response: dict) -> dict:
        output = response.get("output")
        diagnostics: dict[str, Any] = {
            "output_present": "output" in response,
            "output_type": _json_type(output),
            "output_count": len(output) if isinstance(output, list) else None,
        }

        def failure(reason: str, code: str = "malformed_result") -> SimulatorModelError:
            return SimulatorModelError(
                code,
                {
                    "result_diagnostics": {**diagnostics, "reason": reason},
                },
            )

        if not isinstance(output, list):
            raise failure("output_not_list")
        parts = []
        for item_index, item in enumerate(output):
            if isinstance(item, dict) and item.get("type") == "reasoning":
                continue
            diagnostics.pop("content_index", None)
            diagnostics.pop("content", None)
            if not isinstance(item, dict):
                diagnostics.update(
                    {"item_index": item_index, "item": {"json_type": _json_type(item)}}
                )
                raise failure("item_not_object")
            diagnostics.update(
                {
                    "item_index": item_index,
                    "item": {
                        "json_type": "object",
                        "type": _shape_label(item, "type", ("message",)),
                        "role": _shape_label(
                            item, "role", ("assistant", "user", "system", "developer", "tool")
                        ),
                        "status": _shape_label(
                            item, "status", ("completed", "in_progress", "incomplete")
                        ),
                        "content_type": _json_type(item.get("content")),
                        "content_count": (
                            len(item["content"]) if isinstance(item.get("content"), list) else None
                        ),
                    },
                }
            )
            if item.get("type") != "message":
                raise failure("unexpected_item_type")
            if item.get("role") != "assistant":
                raise failure("unexpected_item_role")
            if item.get("status") != "completed":
                raise failure("message_not_completed")
            if not isinstance(item.get("content"), list):
                raise failure("content_not_list")
            for content_index, content in enumerate(item["content"]):
                diagnostics["content_index"] = content_index
                diagnostics["content"] = {"json_type": _json_type(content)}
                if not isinstance(content, dict):
                    raise failure("content_not_object")
                diagnostics["content"].update(
                    {
                        "type": _shape_label(content, "type", ("output_text", "refusal")),
                        "text_type": _json_type(content.get("text")),
                    }
                )
                if content.get("type") == "refusal":
                    raise failure("content_refusal", "response_refusal")
                if content.get("type") != "output_text":
                    raise failure("unexpected_content_type")
                if not isinstance(content.get("text"), str):
                    raise failure("text_not_string")
                parts.append(content["text"])
        text = "".join(parts)
        try:
            value = json.loads(text, parse_constant=_reject_constant)
        except json.JSONDecodeError:
            diagnostics["json_category"] = "invalid_syntax" if text.strip() else "empty_output"
            raise failure("invalid_json") from None
        except ValueError:
            diagnostics["json_category"] = "non_finite_number"
            raise failure("invalid_json") from None
        if not isinstance(value, dict):
            diagnostics.update(
                {"json_category": "non_object", "json_value_type": _json_type(value)}
            )
            raise failure("json_not_object")
        return value

    async def close(self) -> None:
        self._closed = True
        failures = False
        for role in _MODELS:
            if not await self._close_subscription(role):
                failures = True
        try:
            if self._http is not None:
                await self._http.aclose()
        except Exception:
            failures = True
        try:
            if self._account is not None:
                await self._account.close()
        except Exception:
            failures = True
        finally:
            self._http = None
            self._account = None
            self._session = None
            self._api_key = None
            self._store = None
        if failures:
            raise self._error("close_failed", {})
