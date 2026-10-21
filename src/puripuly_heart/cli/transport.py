from __future__ import annotations

import asyncio
import hmac
import json
import secrets
from collections.abc import AsyncIterator
from contextlib import aclosing
from typing import Any

PROTOCOL_VERSION = 1
MAX_REQUEST_BYTES = 64 * 1024
MAX_RESPONSE_BYTES = 1024 * 1024
READ_TIMEOUT = 30.0
MAX_CONNECTIONS = 64
WRITE_TIMEOUT = 5.0


class ControlTransportError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _encode(value: dict[str, Any], *, limit: int = MAX_RESPONSE_BYTES) -> bytes:
    data = json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )
    if len(data) + 1 > limit:
        code = "request_too_large" if limit == MAX_REQUEST_BYTES else "response_too_large"
        raise ControlTransportError(code, "Control message exceeds the protocol limit")
    return data + b"\n"


async def _read(reader: asyncio.StreamReader) -> dict[str, Any]:
    try:
        data = await asyncio.wait_for(reader.readline(), READ_TIMEOUT)
    except (asyncio.LimitOverrunError, ValueError) as exc:
        raise ControlTransportError(
            "request_too_large", "Control request exceeds the protocol limit"
        ) from exc
    if not data:
        raise ControlTransportError("connection_lost", "Control connection closed")
    if len(data) > MAX_REQUEST_BYTES or not data.endswith(b"\n"):
        raise ControlTransportError(
            "request_too_large", "Control request exceeds the protocol limit"
        )
    try:
        value = json.loads(data)
    except (UnicodeError, ValueError) as exc:
        raise ControlTransportError("malformed_request", "Invalid JSON control request") from exc
    if not isinstance(value, dict):
        raise ControlTransportError("malformed_request", "Control request must be a JSON object")
    return value


class ControlServer:
    def __init__(self, control: Any, *, instance_id: str, token: str):
        from puripuly_heart.app.services.application_control import (
            InvalidArgumentsError,
            UnknownOperationError,
        )
        from puripuly_heart.core.diagnostics import DiagnosticQueryArgumentsError

        self._unknown_operation_error = UnknownOperationError
        self._argument_errors = (InvalidArgumentsError, DiagnosticQueryArgumentsError)
        self.control = control
        self.instance_id = instance_id
        self._token = token
        self._server: asyncio.AbstractServer | None = None
        self._closing = False
        self._clients: set[asyncio.Task[Any]] = set()
        self._inflight = 0
        self._drained = asyncio.Event()
        self._drained.set()

    async def start(self) -> int:
        if self._server is not None:
            raise RuntimeError("Control server already started")
        self._server = await asyncio.start_server(
            self._serve, host="127.0.0.1", port=0, limit=MAX_REQUEST_BYTES + 1
        )
        return self._server.sockets[0].getsockname()[1]

    def freeze(self) -> None:
        self._closing = True
        if self._server is not None:
            self._server.close()

    async def wait_for_idle(self, timeout: float = 5.0) -> None:
        try:
            await asyncio.wait_for(self._drained.wait(), timeout)
        except TimeoutError:
            pass

    def _finish_dispatch(self) -> None:
        self._inflight -= 1
        if not self._inflight:
            self._drained.set()

    async def close(self) -> None:
        self.freeze()
        tasks = tuple(task for task in self._clients if task is not asyncio.current_task())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        if self._server is not None:
            await self._server.wait_closed()

    async def _serve(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        task = asyncio.current_task()
        if len(self._clients) >= MAX_CONNECTIONS:
            await self._send_error(writer, None, "busy", "Too many local control connections")
            writer.close()
            return
        if task is not None:
            self._clients.add(task)
        request_id: str | None = None
        dispatching = False
        try:
            request = await _read(reader)
            request_id = (
                request.get("request_id") if isinstance(request.get("request_id"), str) else None
            )
            if request.get("protocol") != PROTOCOL_VERSION:
                raise ControlTransportError(
                    "unsupported_protocol", "Unsupported control protocol version"
                )
            if request.get("instance_id") != self.instance_id:
                raise ControlTransportError(
                    "stale_instance", "Control instance changed; rediscover it"
                )
            token = request.get("token")
            if not isinstance(token, str) or not hmac.compare_digest(token, self._token):
                raise ControlTransportError("unauthorized", "Invalid local control credential")
            if self._closing:
                raise ControlTransportError("shutting_down", "Application is shutting down")
            method = request.get("method")
            arguments = request.get("arguments", {})
            if not isinstance(arguments, dict):
                raise ControlTransportError("malformed_request", "Arguments must be an object")
            self._inflight += 1
            self._drained.clear()
            dispatching = True
            if method == "capabilities":
                result = self.control.capabilities()
            elif method == "query":
                name = _required_string(arguments, "name")
                result = await self.control.query(name, _object(arguments, "values"))
            elif method == "submit":
                name = _required_string(arguments, "name")
                result = await self.control.submit(
                    name,
                    _object(arguments, "values"),
                    request_id=_required_string(arguments, "request_id"),
                    expected_revision=arguments.get("expected_revision"),
                )
            elif method in ("operation", "cancel"):
                operation_id = _required_string(arguments, "operation_id")
                result = await getattr(self.control, method)(operation_id)
            elif method == "wait":
                operation_id = _required_string(arguments, "operation_id")
                timeout = arguments.get("timeout")
                if timeout is not None and (
                    isinstance(timeout, bool)
                    or not isinstance(timeout, (int, float))
                    or not 0 <= timeout <= 3600
                ):
                    raise ControlTransportError(
                        "malformed_request", "Timeout must be between 0 and 3600 seconds"
                    )
                result = await self.control.wait(operation_id, timeout=timeout)
            elif method == "subscribe":
                topics = arguments.get("topics")
                if (
                    not isinstance(topics, list)
                    or not topics
                    or len(topics) > 16
                    or any(not isinstance(topic, str) for topic in topics)
                ):
                    raise ControlTransportError(
                        "malformed_request", "Specify up to 16 subscription topics"
                    )
                channel = arguments.get("channel")
                if channel is not None and (
                    not isinstance(channel, str) or channel not in ("self", "peer")
                ):
                    raise ControlTransportError("malformed_request", "Channel must be self or peer")
                for flag in ("include_transcripts", "include_translations"):
                    if type(arguments.get(flag, False)) is not bool:
                        raise ControlTransportError("malformed_request", f"{flag} must be boolean")
                after = arguments.get("after")
                if after is not None and (type(after) is not int or after < 0):
                    raise ControlTransportError(
                        "malformed_request",
                        "Subscription cursor must be a nonnegative sequence number",
                    )
                cursor_instance_id = arguments.get("cursor_instance_id")
                if cursor_instance_id is not None and (
                    not isinstance(cursor_instance_id, str) or len(cursor_instance_id) > 64
                ):
                    raise ControlTransportError(
                        "malformed_request", "Invalid cursor instance identity"
                    )
                subscription: AsyncIterator[dict[str, Any]] = self.control.subscribe(
                    topics=topics,
                    channel=channel,
                    include_transcripts=arguments.get("include_transcripts", False),
                    include_translations=arguments.get("include_translations", False),
                    after=arguments.get("after"),
                    **(
                        {"cursor_instance_id": cursor_instance_id}
                        if cursor_instance_id is not None
                        else {}
                    ),
                )
                await self._send(
                    writer,
                    {
                        "protocol": PROTOCOL_VERSION,
                        "instance_id": self.instance_id,
                        "request_id": request_id,
                        "result": {"subscribed": topics},
                    },
                )
                self._finish_dispatch()
                dispatching = False
                disconnected = asyncio.create_task(
                    reader.read(1), name="control:subscriber-disconnect"
                )
                pending_event = None
                async with aclosing(subscription):
                    try:
                        while not self._closing:
                            pending_event = asyncio.create_task(
                                anext(subscription), name="control:subscriber-event"
                            )
                            completed, _ = await asyncio.wait(
                                (pending_event, disconnected),
                                return_when=asyncio.FIRST_COMPLETED,
                            )
                            if disconnected in completed:
                                pending_event.cancel()
                                await asyncio.gather(pending_event, return_exceptions=True)
                                break
                            try:
                                event = pending_event.result()
                            except StopAsyncIteration:
                                break
                            await self._send(
                                writer,
                                {
                                    "protocol": PROTOCOL_VERSION,
                                    "instance_id": self.instance_id,
                                    "request_id": request_id,
                                    "event": event,
                                },
                            )
                    finally:
                        disconnected.cancel()
                        if pending_event is not None:
                            pending_event.cancel()
                        await asyncio.gather(
                            disconnected,
                            *([pending_event] if pending_event is not None else []),
                            return_exceptions=True,
                        )
                return
            else:
                raise ControlTransportError("unknown_method", "Unknown control method")
            await self._send(
                writer,
                {
                    "protocol": PROTOCOL_VERSION,
                    "instance_id": self.instance_id,
                    "request_id": request_id,
                    "result": result,
                },
            )
            if method == "submit" and name == "app.stop":
                try:
                    await asyncio.wait_for(reader.read(1), 3.0)
                except TimeoutError, ConnectionError:
                    pass
            self._finish_dispatch()
            dispatching = False
        except asyncio.CancelledError, ConnectionError, BrokenPipeError:
            pass
        except ControlTransportError as exc:
            await self._send_error(writer, request_id, exc.code, str(exc))
        except self._unknown_operation_error:
            await self._send_error(
                writer,
                request_id,
                "unknown_or_expired_operation",
                "Operation identity is unknown or expired in this application instance",
            )
        except self._argument_errors:
            await self._send_error(
                writer, request_id, "invalid_arguments", "Control arguments are invalid"
            )
        except Exception:
            await self._send_error(
                writer, request_id, "operation_error", "Control request could not complete"
            )
        finally:
            if dispatching:
                self._finish_dispatch()
            if task is not None:
                self._clients.discard(task)
            writer.close()
            try:
                await writer.wait_closed()
            except OSError:
                pass

    async def _send(self, writer: asyncio.StreamWriter, response: dict[str, Any]) -> None:
        writer.write(_encode(response))
        await asyncio.wait_for(writer.drain(), WRITE_TIMEOUT)

    async def _send_error(
        self, writer: asyncio.StreamWriter, request_id: str | None, code: str, message: str
    ) -> None:
        try:
            await self._send(
                writer,
                {
                    "protocol": PROTOCOL_VERSION,
                    "instance_id": self.instance_id,
                    "request_id": request_id,
                    "error": {"code": code, "message": message},
                },
            )
        except OSError, TimeoutError:
            pass


def _required_string(values: dict[str, Any], name: str) -> str:
    value = values.get(name)
    if not isinstance(value, str) or not value or len(value) > 256:
        raise ControlTransportError("malformed_request", f"{name} must be a nonempty string")
    return value


def _object(values: dict[str, Any], name: str) -> dict[str, Any]:
    value = values.get(name, {})
    if not isinstance(value, dict):
        raise ControlTransportError("malformed_request", f"{name} must be an object")
    return value


async def request(
    record: Any, method: str, arguments: dict[str, Any] | None = None, *, timeout: float = 30.0
) -> dict[str, Any]:
    try:
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection("127.0.0.1", record.port, limit=MAX_RESPONSE_BYTES + 1), timeout
        )
    except (OSError, TimeoutError) as exc:
        raise ControlTransportError(
            "instance_unavailable", "Control host unavailable; rediscover the instance"
        ) from exc
    try:
        correlation_id = secrets.token_hex(12)
        writer.write(
            _encode(
                {
                    "protocol": PROTOCOL_VERSION,
                    "instance_id": record.instance_id,
                    "token": record.token,
                    "request_id": correlation_id,
                    "method": method,
                    "arguments": arguments or {},
                },
                limit=MAX_REQUEST_BYTES,
            )
        )
        await asyncio.wait_for(writer.drain(), timeout)
        response = await asyncio.wait_for(_read_response(reader), timeout)
        if response.get("request_id") != correlation_id:
            raise ControlTransportError("malformed_response", "Control response identity mismatch")
        if response.get("instance_id") != record.instance_id:
            raise ControlTransportError("stale_instance", "Control instance changed; rediscover it")
        if "error" in response:
            error = response["error"]
            if not isinstance(error, dict):
                raise ControlTransportError("malformed_response", "Invalid control error")
            raise ControlTransportError(
                str(error.get("code", "operation_error")),
                str(error.get("message", "Control request failed")),
            )
        result = response.get("result")
        if not isinstance(result, dict):
            raise ControlTransportError("malformed_response", "Control result must be an object")
        return result
    except TimeoutError as exc:
        raise ControlTransportError(
            "unknown_execution",
            "Control request timed out; execution state is unknown, do not replay mutations",
        ) from exc
    except OSError as exc:
        raise ControlTransportError(
            "connection_lost", "Control host disconnected; execution state is unknown"
        ) from exc
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except OSError:
            pass


async def follow(record: Any, arguments: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
    try:
        reader, writer = await asyncio.open_connection(
            "127.0.0.1", record.port, limit=MAX_RESPONSE_BYTES + 1
        )
    except OSError as exc:
        raise ControlTransportError("instance_unavailable", "Control host unavailable") from exc
    try:
        correlation_id = secrets.token_hex(12)
        writer.write(
            _encode(
                {
                    "protocol": PROTOCOL_VERSION,
                    "instance_id": record.instance_id,
                    "token": record.token,
                    "request_id": correlation_id,
                    "method": "subscribe",
                    "arguments": arguments,
                },
                limit=MAX_REQUEST_BYTES,
            )
        )
        await writer.drain()
        while True:
            response = await _read_response(reader)
            if response.get("request_id") != correlation_id:
                raise ControlTransportError("malformed_response", "Control event identity mismatch")
            if response.get("instance_id") != record.instance_id:
                raise ControlTransportError("stale_instance", "Control instance changed")
            if "error" in response:
                error = response["error"]
                if not isinstance(error, dict):
                    raise ControlTransportError("malformed_response", "Invalid control error")
                raise ControlTransportError(str(error.get("code")), str(error.get("message")))
            if "event" in response:
                yield response["event"]
    except OSError as exc:
        raise ControlTransportError("connection_lost", "Control event stream disconnected") from exc
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except OSError:
            pass


async def _read_response(reader: asyncio.StreamReader) -> dict[str, Any]:
    try:
        data = await reader.readline()
    except ValueError as exc:
        raise ControlTransportError(
            "response_too_large", "Control response exceeds the protocol limit"
        ) from exc
    if not data:
        raise ControlTransportError(
            "connection_lost", "Control host disconnected; execution state may be unknown"
        )
    if len(data) > MAX_RESPONSE_BYTES or not data.endswith(b"\n"):
        raise ControlTransportError(
            "response_too_large", "Control response exceeds the protocol limit"
        )
    try:
        value = json.loads(data)
    except (UnicodeError, ValueError) as exc:
        raise ControlTransportError("malformed_response", "Invalid control response") from exc
    if not isinstance(value, dict) or value.get("protocol") != PROTOCOL_VERSION:
        raise ControlTransportError("malformed_response", "Invalid control response version")
    return value
