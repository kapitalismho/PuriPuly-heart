from __future__ import annotations

import asyncio
import json
import logging
from collections import deque
from collections.abc import Awaitable, Callable, Coroutine, Mapping
from dataclasses import dataclass, field
from typing import Any, Final
from uuid import UUID

from websockets.exceptions import InvalidStatus

from puripuly_heart.core import network_clients
from puripuly_heart.config.runtime_resolution import OPENAI_MODEL_GPT_6_LUNA
from puripuly_heart.core.chatgpt.oauth import ChatGptAuthError, ChatGptReauthRequired
from puripuly_heart.core.chatgpt.session import ChatGptAccessTokenPort
from puripuly_heart.core.error_messages import format_error_report_for_log, provider_failure_report
from puripuly_heart.core.llm.latency import current_attempt
from puripuly_heart.core.llm.provider import LLMProvider, LLMRequestExecution
from puripuly_heart.core.observability import ProviderObservationPort
from puripuly_heart.domain.models import Translation
from puripuly_heart.providers.llm.messages import build_translation_user_message

logger = logging.getLogger(__name__)

CHATGPT_RESPONSES_WS_URL: Final = "wss://api.openai.com/v1/responses"
_SUBSCRIPTION_STATUS_BY_CODE: Final[Mapping[str, int]] = {
    "subscription_sharing_usage_limit_exceeded": 429,
    "subscription_sharing_user_not_eligible": 403,
    "subscription_sharing_invalid_user": 401,
    "subscription_sharing_usage_unavailable": 503,
    "subscription_sharing_user_unavailable": 503,
    "subscription_sharing_unsupported_capability": 400,
    "subscription_sharing_route_not_supported": 403,
}


class ChatGptPlanResponseError(RuntimeError):
    diagnostic_provider = "chatgpt"

    def __init__(self, status_code: int | None, code: str | None = None) -> None:
        self.status_code = status_code
        self.subscription_code = code
        detail = f"status={status_code}" + (f" code={code}" if code else "")
        super().__init__(f"ChatGPT plan request failed ({detail})")


WebSocketConnector = Callable[[str, Mapping[str, str]], Awaitable[Any]]


async def _connect_websocket(url: str, headers: Mapping[str, str]) -> Any:
    return await network_clients.external_websocket_connect(url,
    additional_headers=dict(headers),
    max_size=None,
    ping_interval=20,
    ping_timeout=20,
    open_timeout=15,)


def _build_system_prompt(*, system_prompt: str, source_language: str, target_language: str) -> str:
    if "{source_language}" not in system_prompt:
        return system_prompt
    return system_prompt.format(source_language=source_language, target_language=target_language)


def _error_from_event(event: Mapping[str, object]) -> ChatGptPlanResponseError:
    kind = event.get("type")
    error: object = None
    status: object = None
    if kind == "error":
        error = event.get("error")
        status = event.get("status")
    else:
        response = event.get("response")
        if isinstance(response, Mapping):
            error = response.get("error") or response.get("incomplete_details")
    code = error.get("code") if isinstance(error, Mapping) else None
    code = code if isinstance(code, str) and code else None
    status_code = status if isinstance(status, int) else None
    if code in _SUBSCRIPTION_STATUS_BY_CODE:
        status_code = _SUBSCRIPTION_STATUS_BY_CODE[code]
    if status_code is None and kind == "response.incomplete":
        code = code or "incomplete"
    return ChatGptPlanResponseError(status_code, code)


@dataclass(slots=True, eq=False)
class _PooledConnection:
    socket: Any
    token: str = field(repr=False)
    token_generation: int
    pool_epoch: int = 0
    used: bool = False


@dataclass(slots=True, eq=False)
class _AdmissionTicket:
    utterance_id: UUID
    message: str
    max_attempts: int
    reauth_available: bool
    queued_at: float
    future: asyncio.Future[_RequestExecution]


class _RequestExecution:
    def __init__(
        self,
        provider: ChatGptPlanLLMProvider,
        ticket: _AdmissionTicket,
        connections: list[_PooledConnection],
    ) -> None:
        self._provider = provider
        self._utterance_id = ticket.utterance_id
        self._message = ticket.message
        self._reauth_available = ticket.reauth_available
        self._wait_ms = max(0, round((asyncio.get_running_loop().time() - ticket.queued_at) * 1000))
        self._leases: list[_PooledConnection | None] = list(connections)
        self._attempt_count = len(connections)
        self._closed = False
        self._close_task: asyncio.Task[None] | None = None

    @property
    def attempt_count(self) -> int:
        return self._attempt_count

    async def translate_attempt(self, attempt_index: int) -> Translation:
        if self._closed:
            raise asyncio.CancelledError
        if not 0 <= attempt_index < self._attempt_count:
            raise IndexError("ChatGPT plan attempt index is out of range")
        connection = self._leases[attempt_index]
        if connection is None:
            raise RuntimeError("ChatGPT plan attempt was already consumed")
        self._leases[attempt_index] = None
        try:
            translated = await self._provider._run_reserved(
                connection, self._message, self._wait_ms, self
            )
        except ChatGptPlanResponseError as exc:
            if (
                self._closed
                or self._provider._closed
                or connection.pool_epoch != self._provider._pool_epoch
            ):
                raise asyncio.CancelledError from exc
            if (
                not self._reauth_available
                or exc.status_code != 401
                or exc.subscription_code is not None
            ):
                self._provider._log_failure("translate", exc)
                raise
            replacement = await self._provider._admit_message(
                self._utterance_id, self._message, max_attempts=1, reauth_available=False
            )
            try:
                return await replacement.translate_attempt(0)
            finally:
                await replacement.close()
        except Exception as exc:
            if (
                self._closed
                or self._provider._closed
                or connection.pool_epoch != self._provider._pool_epoch
            ):
                raise asyncio.CancelledError from exc
            self._provider._log_failure("translate", exc)
            raise
        return Translation(utterance_id=self._utterance_id, text=translated)

    async def close(self) -> None:
        if self._close_task is None:
            self._closed = True
            connections = [connection for connection in self._leases if connection is not None]
            self._leases = [None] * self._attempt_count
            self._close_task = self._provider._maintain(self._provider._return_unused(connections))
        try:
            await asyncio.shield(self._close_task)
        except asyncio.CancelledError:
            await self._close_task
            raise


@dataclass(slots=True)
class ChatGptPlanLLMProvider(LLMProvider):
    session: ChatGptAccessTokenPort
    model: str = OPENAI_MODEL_GPT_6_LUNA
    url: str = CHATGPT_RESPONSES_WS_URL
    max_connections: int = 6
    prepared_connections: int = 6
    request_timeout_s: float = 30.0
    max_output_chars: int = 2000
    runtime_logging: ProviderObservationPort | None = None
    connector: WebSocketConnector = _connect_websocket
    _idle: list[_PooledConnection] = field(init=False, default_factory=list, repr=False)
    _reserved: set[_PooledConnection] = field(init=False, default_factory=set, repr=False)
    _waiters: deque[_AdmissionTicket] = field(init=False, default_factory=deque, repr=False)
    _open_count: int = field(init=False, default=0, repr=False)
    _condition: asyncio.Condition | None = field(init=False, default=None, repr=False)
    _closed: bool = field(init=False, default=False, repr=False)
    _pool_epoch: int = field(init=False, default=0, repr=False)
    _requests: set[asyncio.Task[str]] = field(init=False, default_factory=set, repr=False)
    _auth_tasks: set[asyncio.Task[str]] = field(init=False, default_factory=set, repr=False)
    _openings: dict[asyncio.Task[_PooledConnection], asyncio.Future[None]] = field(
        init=False, default_factory=dict, repr=False
    )
    _maintenance: set[asyncio.Task[None]] = field(init=False, default_factory=set, repr=False)
    _dispatch_handle: asyncio.Handle | None = field(init=False, default=None, repr=False)
    _close_task: asyncio.Task[None] | None = field(init=False, default=None, repr=False)

    def _cond(self) -> asyncio.Condition:
        if self._condition is None:
            self._condition = asyncio.Condition()
        return self._condition

    def _maintain(self, operation: Coroutine[Any, Any, None]) -> asyncio.Task[None]:
        task = asyncio.create_task(operation)
        self._maintenance.add(task)
        task.add_done_callback(self._maintenance_finished)
        return task

    def _maintenance_finished(self, task: asyncio.Task[None]) -> None:
        self._maintenance.discard(task)
        if not task.cancelled():
            task.exception()

    def _schedule_dispatch(self) -> None:
        if not self._closed and self._dispatch_handle is None:
            self._dispatch_handle = asyncio.get_running_loop().call_soon(self._start_dispatch)

    def _start_dispatch(self) -> None:
        self._dispatch_handle = None
        if not self._closed:
            self._maintain(self._dispatch())

    async def _dispatch(self) -> None:
        async with self._cond():
            if self._closed:
                return
            self._waiters = deque(ticket for ticket in self._waiters if not ticket.future.done())
            stale = [connection for connection in self._idle if not self._valid(connection)]
            if stale:
                self._idle = [connection for connection in self._idle if connection not in stale]
                self._retire(stale)
            grants: list[tuple[_AdmissionTicket, list[_PooledConnection]]] = []
            while self._waiters and self._idle:
                ticket = self._waiters.popleft()
                grants.append((ticket, [self._idle.pop()]))
            if not self._waiters:
                for ticket, connections in grants:
                    if ticket.max_attempts == 2 and self._idle:
                        connections.append(self._idle.pop())
            for ticket, connections in grants:
                self._reserved.update(connections)
                ticket.future.set_result(_RequestExecution(self, ticket, connections))
            missing = min(
                max(0, len(self._waiters) - len(self._openings)),
                max(0, self.max_connections - self._open_count),
            )
            for _ in range(missing):
                self._start_opening()

    def _valid(self, connection: _PooledConnection) -> bool:
        return (
            not self._closed
            and connection.pool_epoch == self._pool_epoch
            and connection.token_generation == self.session.token_generation
            and getattr(connection.socket, "close_code", None) is None
        )

    def _start_opening(self) -> asyncio.Task[_PooledConnection]:
        self._open_count += 1
        pool_epoch = self._pool_epoch
        task = asyncio.create_task(self._open_slot(pool_epoch))
        finished: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self._openings[task] = finished
        task.add_done_callback(
            lambda completed: self._maintain(self._finish_opening(completed, pool_epoch, finished))
        )
        return task

    async def _open_slot(self, pool_epoch: int) -> _PooledConnection:
        async with asyncio.timeout(15):
            return await self._open(pool_epoch)

    async def _finish_opening(
        self,
        task: asyncio.Task[_PooledConnection],
        pool_epoch: int,
        finished: asyncio.Future[None],
    ) -> None:
        connection: _PooledConnection | None = None
        failure: BaseException | None = None
        try:
            connection = task.result()
        except BaseException as exc:
            failure = exc
        try:
            async with self._cond():
                self._openings.pop(task)
                if connection is None:
                    self._open_count -= 1
                    retirement = None
                    if failure is not None and not isinstance(failure, asyncio.CancelledError):
                        while self._waiters and self._waiters[0].future.done():
                            self._waiters.popleft()
                        if self._waiters and pool_epoch == self._pool_epoch:
                            self._waiters.popleft().future.set_exception(failure)
                        else:
                            self._log_failure("prepare_connections", failure)
                elif self._valid(connection):
                    self._idle.append(connection)
                    retirement = None
                else:
                    retirement = self._retire([connection])
                self._schedule_dispatch()
            if retirement is not None:
                await asyncio.shield(retirement)
        finally:
            finished.set_result(None)

    async def prepare_connections(self) -> None:
        async with self._cond():
            if self._closed:
                return
            missing = max(
                0, min(self.prepared_connections, self.max_connections) - self._open_count
            )
            tasks = [self._start_opening() for _ in range(missing)]
            completion = asyncio.gather(*(self._openings[task] for task in tasks))
        try:
            await asyncio.gather(*tasks, return_exceptions=True)
            await asyncio.shield(completion)
        except asyncio.CancelledError:
            for task in tasks:
                task.cancel()
            results = await asyncio.gather(*tasks, return_exceptions=True)
            await completion
            async with self._cond():
                created = [
                    item
                    for item in results
                    if isinstance(item, _PooledConnection) and item in self._idle
                ]
                for connection in created:
                    self._idle.remove(connection)
                retirement = self._retire(created)
            if retirement is not None:
                await asyncio.shield(retirement)
            raise

    async def admit_request(
        self,
        *,
        utterance_id: UUID,
        text: str,
        system_prompt: str,
        source_language: str,
        target_language: str,
        context: str = "",
        scene_participant_count: int | None = None,
        max_output_tokens: int | None = None,
        max_attempts: int = 2,
    ) -> LLMRequestExecution:
        _ = max_output_tokens
        body = {
            "type": "response.create",
            "model": self.model,
            "store": False,
            "reasoning": {"effort": "none"},
            "instructions": _build_system_prompt(
                system_prompt=system_prompt,
                source_language=source_language,
                target_language=target_language,
            ),
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": build_translation_user_message(
                                text=text,
                                context=context,
                                scene_participant_count=scene_participant_count,
                            ),
                        }
                    ],
                }
            ],
        }
        return await self._admit_message(utterance_id, json.dumps(body), max_attempts=max_attempts)

    async def _admit_message(
        self,
        utterance_id: UUID,
        message: str,
        *,
        max_attempts: int,
        reauth_available: bool = True,
    ) -> _RequestExecution:
        if max_attempts not in (1, 2):
            raise ValueError("ChatGPT plan requests support one or two attempts")
        pool_epoch = self._pool_epoch
        queued_at = asyncio.get_running_loop().time()
        while True:
            if self._closed:
                raise RuntimeError("ChatGPT plan provider is closed")
            ticket: _AdmissionTicket | None = None
            try:
                await self._current_token_generation()
                async with self._cond():
                    if self._closed or pool_epoch != self._pool_epoch:
                        raise asyncio.CancelledError
                    ticket = _AdmissionTicket(
                        utterance_id,
                        message,
                        max_attempts,
                        reauth_available,
                        queued_at,
                        asyncio.get_running_loop().create_future(),
                    )
                    self._waiters.append(ticket)
                    self._schedule_dispatch()
                return await ticket.future
            except asyncio.CancelledError:
                execution = None
                async with self._cond():
                    if ticket is not None:
                        if ticket in self._waiters:
                            self._waiters.remove(ticket)
                        if (
                            ticket.future.done()
                            and not ticket.future.cancelled()
                            and ticket.future.exception() is None
                        ):
                            execution = ticket.future.result()
                    self._schedule_dispatch()
                if execution is not None:
                    await execution.close()
                raise
            except ChatGptPlanResponseError as exc:
                if self._closed or pool_epoch != self._pool_epoch:
                    raise asyncio.CancelledError from exc
                if (
                    not reauth_available
                    or exc.status_code != 401
                    or exc.subscription_code is not None
                ):
                    raise
                reauth_available = False
            except Exception as exc:
                if self._closed or pool_epoch != self._pool_epoch:
                    raise asyncio.CancelledError from exc
                raise

    async def translate(
        self,
        *,
        utterance_id: UUID,
        text: str,
        system_prompt: str,
        source_language: str,
        target_language: str,
        context: str = "",
        scene_participant_count: int | None = None,
        max_output_tokens: int | None = None,
    ) -> Translation:
        try:
            execution = await self.admit_request(
                utterance_id=utterance_id,
                text=text,
                system_prompt=system_prompt,
                source_language=source_language,
                target_language=target_language,
                context=context,
                scene_participant_count=scene_participant_count,
                max_output_tokens=max_output_tokens,
                max_attempts=1,
            )
        except Exception as exc:
            self._log_failure("translate", exc)
            raise
        try:
            return await execution.translate_attempt(0)
        finally:
            await execution.close()

    async def release_connections(self) -> None:
        task = self._maintain(self._release_pool())
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            await task
            raise

    async def _release_pool(self) -> None:
        async with self._cond():
            self._pool_epoch += 1
            for ticket in self._waiters:
                ticket.future.cancel()
            self._waiters.clear()
            available = list(self._idle) + list(self._reserved)
            self._idle.clear()
            self._reserved.clear()
            openings = tuple(self._openings.items())
            for task, _ in openings:
                task.cancel()
            auth_tasks = tuple(self._auth_tasks)
            for task in auth_tasks:
                task.cancel()
            retirement = self._retire(available)
        await asyncio.gather(
            *(task for task, _ in openings),
            *(finished for _, finished in openings),
            *auth_tasks,
            *([retirement] if retirement is not None else []),
            return_exceptions=True,
        )

    async def close(self) -> None:
        if self._close_task is None:
            self._closed = True
            if self._dispatch_handle is not None:
                self._dispatch_handle.cancel()
                self._dispatch_handle = None
            self._close_task = asyncio.create_task(self._close_pool())
        try:
            await asyncio.shield(self._close_task)
        except asyncio.CancelledError:
            await self._close_task
            raise

    async def _close_pool(self) -> None:
        requests = tuple(self._requests)
        for task in requests:
            task.cancel()
        await asyncio.gather(self._release_pool(), *requests, return_exceptions=True)
        while self._maintenance:
            await asyncio.gather(*tuple(self._maintenance), return_exceptions=True)

    async def _run_reserved(
        self,
        connection: _PooledConnection,
        message: str,
        wait_ms: int,
        execution: _RequestExecution,
    ) -> str:
        observation = current_attempt()
        if observation is not None:
            observation.transport = "websocket"
            observation.connection_reused = connection.used
            observation.connection_wait_ms = (observation.connection_wait_ms or 0) + wait_ms
        task: asyncio.Task[str] | None = None
        started = asyncio.Event()
        entered = asyncio.Event()
        abandoned = asyncio.Event()
        try:
            if self._closed or execution._closed or connection.pool_epoch != self._pool_epoch:
                raise asyncio.CancelledError
            token_generation = await self._current_token_generation()
            async with self._cond():
                if self._closed or execution._closed or connection.pool_epoch != self._pool_epoch:
                    raise asyncio.CancelledError
                if (
                    connection not in self._reserved
                    or connection.token_generation != token_generation
                    or not self._valid(connection)
                ):
                    raise RuntimeError("ChatGPT plan connection reservation is no longer valid")
                task = asyncio.create_task(
                    self._run_connection(connection, message, started, abandoned, entered)
                )
                self._requests.add(task)
                task.add_done_callback(self._request_finished)
            try:
                return await asyncio.shield(task)
            except asyncio.CancelledError:
                if not started.is_set():
                    abandoned.set()
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
                raise
        finally:
            if not entered.is_set():
                cleanup = self._maintain(self._return_unused([connection]))
                try:
                    await asyncio.shield(cleanup)
                except asyncio.CancelledError:
                    await cleanup
                    raise

    def _request_finished(self, task: asyncio.Task[str]) -> None:
        self._requests.discard(task)
        if not task.cancelled():
            task.exception()

    async def _run_connection(
        self,
        connection: _PooledConnection,
        message: str,
        started: asyncio.Event,
        abandoned: asyncio.Event,
        entered: asyncio.Event,
    ) -> str:
        entered.set()
        reusable = False
        try:
            text = await asyncio.wait_for(
                self._exchange(connection, message, started, abandoned),
                timeout=self.request_timeout_s,
            )
            reusable = True
            return text
        except ChatGptPlanResponseError as exc:
            if exc.status_code == 401:
                self.session.invalidate_access_token(connection.token)
            raise
        finally:
            if started.is_set():
                await self._release(connection, reusable=reusable)
            else:
                await self._return_unused([connection])

    async def _exchange(
        self,
        connection: _PooledConnection,
        message: str,
        started: asyncio.Event,
        abandoned: asyncio.Event,
    ) -> str:
        observation = current_attempt()
        async with self._cond():
            if abandoned.is_set() or self._closed or connection.pool_epoch != self._pool_epoch:
                raise asyncio.CancelledError
            if connection not in self._reserved or not self._valid(connection):
                raise RuntimeError("ChatGPT plan connection reservation is no longer valid")
            self._reserved.remove(connection)
            connection.used = True
            started.set()
            if observation is not None:
                observation.mark_sent()
        await connection.socket.send(message)
        parts: list[str] = []
        length = 0
        while True:
            event = json.loads(await connection.socket.recv())
            if not isinstance(event, dict):
                continue
            kind = event.get("type")
            if kind == "response.output_text.delta":
                delta = event.get("delta")
                if isinstance(delta, str):
                    if delta and observation is not None and observation.first_text_at is None:
                        observation.mark_first_text()
                    parts.append(delta)
                    length += len(delta)
                    if length > self.max_output_chars:
                        raise RuntimeError("ChatGPT plan response was truncated by length limit")
            elif kind == "response.completed":
                if observation is not None:
                    observation.record_openai_response(event.get("response"))
                result = "".join(parts).strip()
                if not result:
                    raise RuntimeError("ChatGPT plan response contained empty message content")
                return result
            elif kind in ("response.failed", "response.incomplete", "error"):
                raise _error_from_event(event)

    async def _return_unused(self, connections: list[_PooledConnection]) -> None:
        stale: list[_PooledConnection] = []
        async with self._cond():
            for connection in connections:
                if connection not in self._reserved:
                    continue
                self._reserved.remove(connection)
                if self._valid(connection):
                    self._idle.append(connection)
                else:
                    stale.append(connection)
            retirement = self._retire(stale)
            self._schedule_dispatch()
        if retirement is not None:
            await asyncio.shield(retirement)

    async def _release(self, connection: _PooledConnection, *, reusable: bool) -> None:
        async with self._cond():
            if reusable and self._valid(connection):
                self._idle.append(connection)
                retirement = None
            else:
                retirement = self._retire([connection])
            self._schedule_dispatch()
        if retirement is not None:
            try:
                await asyncio.shield(retirement)
            except asyncio.CancelledError:
                await retirement
                raise

    def _retire(self, connections: list[_PooledConnection]) -> asyncio.Task[None] | None:
        if not connections:
            return None
        return self._maintain(self._retire_connections(connections))

    async def _retire_connections(self, connections: list[_PooledConnection]) -> None:
        try:
            await asyncio.gather(
                *(self._discard(connection) for connection in connections),
                return_exceptions=True,
            )
        finally:
            async with self._cond():
                self._open_count -= len(connections)
                self._schedule_dispatch()

    async def _current_token_generation(self) -> int:
        observation = current_attempt()
        started_at = observation.request.clock() if observation is not None else None
        task = asyncio.create_task(self.session.access_token())
        self._auth_tasks.add(task)
        try:
            await task
        except ChatGptReauthRequired:
            raise
        except ChatGptAuthError as exc:
            raise ChatGptPlanResponseError(exc.status_code or 503, exc.code) from exc
        finally:
            self._auth_tasks.discard(task)
            if observation is not None and started_at is not None:
                observation.auth_ms = (observation.auth_ms or 0) + max(
                    0, round((observation.request.clock() - started_at) * 1000)
                )
        return self.session.token_generation

    async def _open(self, pool_epoch: int) -> _PooledConnection:
        observation = current_attempt()
        auth_started_at = observation.request.clock() if observation is not None else None
        try:
            token = await self.session.access_token()
        finally:
            if observation is not None and auth_started_at is not None:
                observation.auth_ms = (observation.auth_ms or 0) + max(
                    0, round((observation.request.clock() - auth_started_at) * 1000)
                )
        if self._closed or pool_epoch != self._pool_epoch:
            raise asyncio.CancelledError
        generation = self.session.token_generation
        connect_started_at = observation.request.clock() if observation is not None else None
        try:
            socket = await self.connector(self.url, {"Authorization": f"Bearer {token}"})
        except InvalidStatus as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status == 401:
                self.session.invalidate_access_token(token)
            raise ChatGptPlanResponseError(status) from exc
        finally:
            if observation is not None and connect_started_at is not None:
                observation.connect_ms = max(
                    0, round((observation.request.clock() - connect_started_at) * 1000)
                )
        return _PooledConnection(
            socket=socket, token=token, token_generation=generation, pool_epoch=pool_epoch
        )

    async def _discard(self, connection: _PooledConnection) -> None:
        close = getattr(connection.socket, "close", None)
        if callable(close):
            try:
                await close()
            except Exception:
                return

    def _log_failure(self, operation: str, exc: BaseException) -> None:
        report = provider_failure_report(exc, provider="chatgpt", operation=operation)
        rendered = "[Basic][LLM] ChatGPT plan request failed [%s]: %s" % (
            operation,
            format_error_report_for_log(report),
        )
        if self.runtime_logging is not None:
            self.runtime_logging.emit_basic(rendered, level=logging.ERROR)
            return
        logger.error(rendered)


__all__ = [
    "CHATGPT_RESPONSES_WS_URL",
    "ChatGptPlanLLMProvider",
    "ChatGptPlanResponseError",
]
