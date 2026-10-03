from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Mapping
from uuid import uuid4

import pytest
from websockets.exceptions import InvalidStatus

from puripuly_heart.core.error_messages import provider_failure_report
from puripuly_heart.core.llm.provider import LLMRequestExecution
from puripuly_heart.providers.llm.chatgpt_plan import (
    ChatGptPlanLLMProvider,
    ChatGptPlanResponseError,
)

Script = Callable[[dict[str, object]], list[dict[str, object]]]


def _completed(text: str) -> Script:
    def script(_request: dict[str, object]) -> list[dict[str, object]]:
        return [
            {"type": "response.created", "response": {"id": "resp"}},
            {"type": "response.output_text.delta", "delta": text[: len(text) // 2]},
            {"type": "response.output_text.delta", "delta": text[len(text) // 2 :]},
            {"type": "response.completed", "response": {"id": "resp", "output": []}},
        ]

    return script


class _FakeSocket:
    def __init__(self, server: _FakeServer, token: str) -> None:
        self.server = server
        self.token = token
        self.sent: list[dict[str, object]] = []
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self.close_code: int | None = None

    async def send(self, raw: str) -> None:
        request = json.loads(raw)
        self.sent.append(request)
        self.server.requests.append(request)
        if self.server.gate is not None:
            await self.server.gate.wait()
        for event in self.server.script(request):
            await self._queue.put(json.dumps(event))

    async def recv(self) -> str:
        return await self._queue.get()

    async def close(self) -> None:
        self.close_code = 1000
        self.server.closed += 1


class _FakeServer:
    def __init__(self, script: Script) -> None:
        self.script = script
        self.sockets: list[_FakeSocket] = []
        self.requests: list[dict[str, object]] = []
        self.closed = 0
        self.reject_tokens: set[str] = set()
        self.gate: asyncio.Event | None = None

    async def connect(self, url: str, headers: Mapping[str, str]) -> _FakeSocket:
        assert url == "wss://api.openai.com/v1/responses"
        token = headers["Authorization"].removeprefix("Bearer ")
        if token in self.reject_tokens:
            raise InvalidStatus(type("Response", (), {"status_code": 401})())
        socket = _FakeSocket(self, token)
        self.sockets.append(socket)
        return socket


class _FakeSession:
    def __init__(self) -> None:
        self.token = "token-1"
        self._generation = 1
        self.invalidated: list[str] = []

    @property
    def token_generation(self) -> int:
        return self._generation

    async def access_token(self) -> str:
        return self.token

    def invalidate_access_token(self, token: str) -> None:
        self.invalidated.append(token)
        if token == self.token:
            self.token = f"token-{self._generation + 1}"
            self._generation += 1

    def rotate(self) -> None:
        self._generation += 1
        self.token = f"token-{self._generation}"


def _provider(
    server: _FakeServer, session: _FakeSession, **kwargs: object
) -> ChatGptPlanLLMProvider:
    return ChatGptPlanLLMProvider(session=session, connector=server.connect, **kwargs)


async def _translate(provider: ChatGptPlanLLMProvider, text: str = "안녕") -> str:
    result = await provider.translate(
        utterance_id=uuid4(),
        text=text,
        system_prompt="Translate {source_language} to {target_language}.",
        source_language="Korean",
        target_language="English",
        max_output_tokens=64,
    )
    return result.text


async def _admit(
    provider: ChatGptPlanLLMProvider, text: str = "안녕", *, max_attempts: int = 2
) -> LLMRequestExecution:
    return await provider.admit_request(
        utterance_id=uuid4(),
        text=text,
        system_prompt="Translate {source_language} to {target_language}.",
        source_language="Korean",
        target_language="English",
        max_attempts=max_attempts,
    )


async def _wait_for_requests(server: _FakeServer, count: int) -> None:
    async with asyncio.timeout(1):
        while len(server.requests) < count:
            await asyncio.sleep(0)


async def test_translate_sends_plan_compatible_request_and_reuses_connection() -> None:
    server = _FakeServer(_completed("Hello there"))
    provider = _provider(server, _FakeSession())

    assert await _translate(provider) == "Hello there"
    assert await _translate(provider, "고마워") == "Hello there"

    assert len(server.sockets) == 1
    request = server.requests[0]
    assert request["type"] == "response.create"
    assert request["model"] == "gpt-6-luna"
    assert request["store"] is False
    assert request["reasoning"] == {"effort": "none"}
    assert request["instructions"] == "Translate Korean to English."
    assert "prompt_cache_options" not in request
    assert "stream" not in request
    assert "temperature" not in request
    assert "max_output_tokens" not in request
    assert "stream_id" not in request
    message = request["input"][0]
    assert message == {
        "type": "message",
        "role": "user",
        "content": [{"type": "input_text", "text": "<input>\n안녕\n</input>"}],
    }
    assert server.requests[1]["instructions"] == request["instructions"]
    assert server.requests[1]["input"][0] != message
    await provider.close()
    assert server.closed == 1


async def test_plan_omits_unsupported_explicit_cache_fields_and_preserves_custom_prompt() -> None:
    server = _FakeServer(_completed("ok"))
    provider = _provider(server, _FakeSession())
    custom_prompt = "Custom instructions\nKeep {literal} exactly as written."
    try:
        for text, context, participants in (
            ("first input", "first context", 2),
            ("second input", "second context", 3),
        ):
            await provider.translate(
                utterance_id=uuid4(),
                text=text,
                system_prompt=custom_prompt,
                source_language="Korean",
                target_language="English",
                context=context,
                scene_participant_count=participants,
                max_output_tokens=37,
            )

        for request in server.requests:
            assert request["store"] is False
            assert request["reasoning"] == {"effort": "none"}
            assert "max_output_tokens" not in request
            user_message = request["input"][-1]
            assert user_message["role"] == "user"
            assert len(user_message["content"]) == 1
            assert user_message["content"][0]["type"] == "input_text"
            assert "prompt_cache_breakpoint" not in user_message["content"][0]
            assert request["instructions"] == custom_prompt
            assert "prompt_cache_options" not in request
            assert len(request["input"]) == 1
        assert server.requests[0]["input"][-1] != server.requests[1]["input"][-1]
    finally:
        await provider.close()


async def test_concurrent_requests_use_separate_connections_within_pool_limit() -> None:
    server = _FakeServer(_completed("ok"))
    server.gate = asyncio.Event()
    provider = _provider(server, _FakeSession(), max_connections=2)

    tasks = [asyncio.create_task(_translate(provider)) for _ in range(3)]
    await asyncio.sleep(0.01)
    assert len(server.sockets) == 2
    server.gate.set()
    assert await asyncio.gather(*tasks) == ["ok", "ok", "ok"]
    assert len(server.sockets) == 2
    await provider.close()


async def test_usage_limit_error_discards_connection_and_maps_to_user_message() -> None:
    def script(_request: dict[str, object]) -> list[dict[str, object]]:
        return [
            {
                "type": "error",
                "status": 429,
                "error": {"code": "subscription_sharing_usage_limit_exceeded"},
            }
        ]

    server = _FakeServer(script)
    provider = _provider(server, _FakeSession())

    with pytest.raises(ChatGptPlanResponseError) as exc_info:
        await _translate(provider)

    assert exc_info.value.status_code == 429
    assert server.closed == 1
    report = provider_failure_report(exc_info.value, provider="chatgpt", operation="translate")
    assert report.message.key == "provider.chatgpt.usage_limit"


async def test_not_eligible_failure_maps_to_user_message() -> None:
    def script(_request: dict[str, object]) -> list[dict[str, object]]:
        return [
            {
                "type": "response.failed",
                "response": {"error": {"code": "subscription_sharing_user_not_eligible"}},
            }
        ]

    provider = _provider(_FakeServer(script), _FakeSession())
    with pytest.raises(ChatGptPlanResponseError) as exc_info:
        await _translate(provider)
    report = provider_failure_report(exc_info.value, provider="chatgpt", operation="translate")
    assert report.message.key == "provider.chatgpt.not_eligible"


async def test_handshake_unauthorized_invalidates_token_and_retries_once() -> None:
    server = _FakeServer(_completed("ok"))
    session = _FakeSession()
    server.reject_tokens.add("token-1")
    provider = _provider(server, session)

    assert await _translate(provider) == "ok"
    assert session.invalidated == ["token-1"]
    assert [socket.token for socket in server.sockets] == ["token-2"]
    await provider.close()


async def test_refreshed_token_replaces_idle_connection() -> None:
    server = _FakeServer(_completed("ok"))
    session = _FakeSession()
    provider = _provider(server, session)

    await _translate(provider)
    session.rotate()
    await _translate(provider)

    assert [socket.token for socket in server.sockets] == ["token-1", "token-2"]
    assert server.sockets[0].close_code is not None
    await provider.close()


async def test_runaway_output_is_rejected_and_connection_discarded() -> None:
    server = _FakeServer(_completed("x" * 50))
    provider = _provider(server, _FakeSession(), max_output_chars=10)

    with pytest.raises(RuntimeError, match="length limit"):
        await _translate(provider)
    assert server.closed == 1


async def test_prepare_connections_opens_idle_connections_for_first_utterances() -> None:
    server = _FakeServer(_completed("ok"))
    provider = _provider(server, _FakeSession(), prepared_connections=2)

    await provider.prepare_connections()
    assert len(server.sockets) == 2
    assert server.requests == []

    await asyncio.gather(_translate(provider), _translate(provider))
    assert len(server.sockets) == 2
    await provider.close()
    assert server.closed == 2


async def test_release_closes_idle_connections_and_drains_in_flight_ones() -> None:
    server = _FakeServer(_completed("ok"))
    provider = _provider(server, _FakeSession(), prepared_connections=2)
    await provider.prepare_connections()

    server.gate = asyncio.Event()
    in_flight = asyncio.create_task(_translate(provider))
    await asyncio.sleep(0.01)
    await provider.release_connections()
    assert server.closed == 1

    server.gate.set()
    assert await in_flight == "ok"
    assert server.closed == 2

    assert await _translate(provider) == "ok"
    assert await _translate(provider) == "ok"
    assert len(server.sockets) == 3
    assert server.closed == 2
    await provider.close()


async def test_cancelled_preparation_closes_partial_connections_and_unblocks_next_request() -> None:
    server = _FakeServer(_completed("ready"))
    pending = asyncio.Event()
    gate = asyncio.Event()
    calls = 0

    async def connect(url: str, headers: Mapping[str, str]) -> _FakeSocket:
        nonlocal calls
        calls += 1
        if calls == 3:
            pending.set()
            await gate.wait()
        return await server.connect(url, headers)

    provider = ChatGptPlanLLMProvider(
        session=_FakeSession(), connector=connect, prepared_connections=3, max_connections=3
    )
    try:
        preparation = asyncio.create_task(provider.prepare_connections())
        await pending.wait()
        preparation.cancel()
        with pytest.raises(asyncio.CancelledError):
            await preparation
        assert server.closed == 2
        assert await asyncio.wait_for(_translate(provider), timeout=1) == "ready"
    finally:
        await provider.close()
    assert server.closed == 3


async def test_cancelled_release_finishes_closing_every_detached_connection() -> None:
    server = _FakeServer(_completed("ready"))
    provider = _provider(server, _FakeSession(), prepared_connections=3)
    await provider.prepare_connections()
    started = asyncio.Event()
    gate = asyncio.Event()
    for socket in server.sockets:
        original = socket.close

        async def slow_close(original: Callable = original) -> None:
            started.set()
            await gate.wait()
            await original()

        socket.close = slow_close
    try:
        release = asyncio.create_task(provider.release_connections())
        await started.wait()
        release.cancel()
        gate.set()
        with pytest.raises(asyncio.CancelledError):
            await release
        assert [socket.close_code for socket in server.sockets] == [1000, 1000, 1000]
    finally:
        gate.set()
        await provider.close()


async def test_cancelled_response_drains_before_connection_reuse_without_mixing_text() -> None:
    server = _FakeServer(lambda _request: [])
    provider = _provider(server, _FakeSession(), max_connections=1)
    first = asyncio.create_task(_translate(provider, "first"))
    try:
        await _wait_for_requests(server, 1)
        socket = server.sockets[0]
        socket._queue.put_nowait(json.dumps({"type": "response.output_text.delta", "delta": "old"}))
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        assert server.closed == 0

        server.script = _completed("new")
        second = asyncio.create_task(_translate(provider, "second"))
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(asyncio.shield(second), timeout=0.02)
        assert len(server.requests) == 1
        socket._queue.put_nowait(
            json.dumps({"type": "response.output_text.delta", "delta": " tail"})
        )
        socket._queue.put_nowait(json.dumps({"type": "response.completed"}))

        assert await asyncio.wait_for(second, timeout=1) == "new"
        assert len(server.sockets) == 1
        assert server.closed == 0
    finally:
        await provider.close()
    assert server.closed == 1


@pytest.mark.parametrize("failure", ["timeout", "error"])
async def test_cancelled_response_failure_frees_pool_slot(failure: str) -> None:
    server = _FakeServer(lambda _request: [])
    provider = _provider(server, _FakeSession(), max_connections=1, request_timeout_s=0.1)
    first = asyncio.create_task(_translate(provider))
    try:
        await _wait_for_requests(server, 1)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        server.script = _completed("next")
        second = asyncio.create_task(_translate(provider))
        if failure == "error":
            server.sockets[0]._queue.put_nowait(
                json.dumps({"type": "response.failed", "response": {"error": {"code": "failed"}}})
            )
        assert await asyncio.wait_for(second, timeout=1) == "next"
        assert server.sockets[0].close_code == 1000
        assert len(server.sockets) == 2
    finally:
        await provider.close()


@pytest.mark.parametrize("shutdown", ["release", "close"])
async def test_shutdown_retires_draining_connection(shutdown: str) -> None:
    server = _FakeServer(lambda _request: [])
    provider = _provider(server, _FakeSession(), max_connections=1)
    first = asyncio.create_task(_translate(provider))
    try:
        await _wait_for_requests(server, 1)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        if shutdown == "close":
            await asyncio.wait_for(provider.close(), timeout=1)
            assert server.closed == 1
            with pytest.raises(RuntimeError, match="closed"):
                await _translate(provider)
        else:
            await provider.release_connections()
            server.sockets[0]._queue.put_nowait(
                json.dumps({"type": "response.output_text.delta", "delta": "old"})
            )
            server.sockets[0]._queue.put_nowait(json.dumps({"type": "response.completed"}))
            server.script = _completed("next")
            assert await asyncio.wait_for(_translate(provider), timeout=1) == "next"
            assert server.sockets[0].close_code == 1000
            assert len(server.sockets) == 2
    finally:
        await provider.close()


async def test_cancelled_pool_waiter_never_sends_an_abandoned_request() -> None:
    server = _FakeServer(lambda _request: [])
    provider = _provider(server, _FakeSession(), max_connections=1)
    first = asyncio.create_task(_translate(provider, "first"))
    try:
        await _wait_for_requests(server, 1)
        abandoned = asyncio.create_task(_translate(provider, "abandoned"))
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(asyncio.shield(abandoned), timeout=0.02)
        abandoned.cancel()
        with pytest.raises(asyncio.CancelledError):
            await abandoned
        server.sockets[0]._queue.put_nowait(
            json.dumps({"type": "response.output_text.delta", "delta": "first"})
        )
        server.sockets[0]._queue.put_nowait(json.dumps({"type": "response.completed"}))
        assert await asyncio.wait_for(first, timeout=1) == "first"
        server.script = _completed("last")
        assert await asyncio.wait_for(_translate(provider, "last"), timeout=1) == "last"
        assert len(server.requests) == 2
        assert len(server.sockets) == 1
    finally:
        await provider.close()


async def test_six_ready_connections_admit_three_immediate_pairs() -> None:
    server = _FakeServer(_completed("paired"))
    server.gate = asyncio.Event()
    provider = _provider(server, _FakeSession())
    executions: list[LLMRequestExecution] = []
    try:
        await provider.prepare_connections()
        executions = await asyncio.gather(*(_admit(provider, str(index)) for index in range(3)))
        assert [execution.attempt_count for execution in executions] == [2, 2, 2]
        attempts = [
            asyncio.create_task(execution.translate_attempt(index))
            for execution in executions
            for index in range(execution.attempt_count)
        ]
        await _wait_for_requests(server, 6)
        assert [len(socket.sent) for socket in server.sockets] == [1] * 6
        server.gate.set()
        assert [result.text for result in await asyncio.gather(*attempts)] == ["paired"] * 6
    finally:
        server.gate.set()
        await asyncio.gather(*(execution.close() for execution in executions))
        await provider.close()
    assert server.closed == 6


async def test_ready_capacity_goes_to_every_first_attempt_before_duplicates() -> None:
    server = _FakeServer(_completed("ok"))
    provider = _provider(server, _FakeSession())
    executions: list[LLMRequestExecution] = []
    try:
        await provider.prepare_connections()
        executions = await asyncio.gather(*(_admit(provider, str(index)) for index in range(4)))
        assert [execution.attempt_count for execution in executions] == [2, 2, 1, 1]
        results = await asyncio.gather(
            *(execution.translate_attempt(0) for execution in executions)
        )
        assert [result.text for result in results] == ["ok"] * 4
        assert len(server.requests) == 4
    finally:
        await asyncio.gather(*(execution.close() for execution in executions))
        await provider.close()
    assert server.closed == 6


async def test_released_slots_cover_queued_primaries_before_optional_duplicates() -> None:
    server = _FakeServer(_completed("ok"))
    provider = _provider(server, _FakeSession(), max_connections=3, prepared_connections=3)
    executions: list[LLMRequestExecution] = []
    try:
        await provider.prepare_connections()
        pair = await _admit(provider)
        single = await _admit(provider, max_attempts=1)
        executions.extend([pair, single])
        first = asyncio.create_task(_admit(provider, "first"))
        second = asyncio.create_task(_admit(provider, "second"))
        await asyncio.sleep(0)
        await pair.close()
        granted = await asyncio.wait_for(asyncio.gather(first, second), timeout=1)
        executions.extend(granted)
        assert [execution.attempt_count for execution in granted] == [1, 1]
        results = await asyncio.gather(*(execution.translate_attempt(0) for execution in granted))
        assert [result.text for result in results] == ["ok", "ok"]
        assert len(server.sockets) == 3
    finally:
        await asyncio.gather(*(execution.close() for execution in executions))
        await provider.close()


async def test_single_ready_connection_starts_without_waiting_and_never_upgrades() -> None:
    server = _FakeServer(_completed("single"))
    server.gate = asyncio.Event()
    provider = _provider(server, _FakeSession(), max_connections=2, prepared_connections=2)
    executions: list[LLMRequestExecution] = []
    try:
        await provider.prepare_connections()
        held = await _admit(provider, max_attempts=1)
        execution = await asyncio.wait_for(_admit(provider), timeout=1)
        executions.extend([held, execution])
        assert execution.attempt_count == 1
        attempt = asyncio.create_task(execution.translate_attempt(0))
        await _wait_for_requests(server, 1)
        await held.close()
        assert execution.attempt_count == 1
        assert len(server.requests) == 1
        with pytest.raises(IndexError):
            await execution.translate_attempt(1)
        server.gate.set()
        assert (await attempt).text == "single"
        assert len(server.requests) == 1
    finally:
        server.gate.set()
        await asyncio.gather(*(execution.close() for execution in executions))
        await provider.close()


async def test_queued_requests_are_fifo_and_do_not_reuse_an_occupied_socket() -> None:
    server = _FakeServer(_completed("done"))
    server.gate = asyncio.Event()
    provider = _provider(server, _FakeSession(), max_connections=1)
    executions: list[LLMRequestExecution] = []
    try:
        await provider.prepare_connections()
        held = await _admit(provider)
        executions.append(held)
        oldest = asyncio.create_task(_admit(provider, "oldest"))
        newest = asyncio.create_task(_admit(provider, "newest"))
        await asyncio.sleep(0)
        await held.close()
        first = await asyncio.wait_for(oldest, timeout=1)
        executions.append(first)
        assert not newest.done()
        attempt = asyncio.create_task(first.translate_attempt(0))
        await _wait_for_requests(server, 1)
        assert "oldest" in server.requests[0]["input"][-1]["content"][0]["text"]
        await first.close()
        assert not newest.done()
        server.gate.set()
        assert (await attempt).text == "done"
        second = await asyncio.wait_for(newest, timeout=1)
        executions.append(second)
        assert (await second.translate_attempt(0)).text == "done"
        assert len(server.sockets) == 1
        assert len(server.requests) == 2
    finally:
        server.gate.set()
        await asyncio.gather(*(execution.close() for execution in executions))
        await provider.close()


@pytest.mark.parametrize("slots", [1, 2])
async def test_cancellation_after_grant_returns_unclaimed_reservations(slots: int) -> None:
    admission: asyncio.Task[LLMRequestExecution] | None = None

    class CancelAfterGrantProvider(ChatGptPlanLLMProvider):
        async def _dispatch(self) -> None:
            waiting = bool(self._waiters)
            await super()._dispatch()
            if waiting and admission is not None:
                admission.cancel()

    server = _FakeServer(_completed("next"))
    provider = CancelAfterGrantProvider(
        session=_FakeSession(), connector=server.connect, max_connections=slots
    )
    execution = None
    try:
        await provider.prepare_connections()
        admission = asyncio.create_task(_admit(provider))
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(admission, timeout=1)
        admission = None
        assert server.requests == []
        execution = await asyncio.wait_for(_admit(provider, "next"), timeout=1)
        assert execution.attempt_count == slots
        results = await asyncio.gather(
            *(execution.translate_attempt(index) for index in range(slots))
        )
        assert [result.text for result in results] == ["next"] * slots
        assert len(server.sockets) == slots
    finally:
        if execution is not None:
            await execution.close()
        await provider.close()
    assert server.closed == slots


async def test_cancellation_before_send_returns_an_admitted_lease() -> None:
    entered = asyncio.Event()
    gate = asyncio.Event()

    class GatedSession(_FakeSession):
        blocked = False

        async def access_token(self) -> str:
            if self.blocked:
                entered.set()
                await gate.wait()
            return await super().access_token()

    session = GatedSession()
    server = _FakeServer(_completed("next"))
    provider = _provider(server, session, max_connections=1)
    execution = await _admit(provider)
    try:
        session.blocked = True
        attempt = asyncio.create_task(execution.translate_attempt(0))
        await entered.wait()
        attempt.cancel()
        with pytest.raises(asyncio.CancelledError):
            await attempt
        await execution.close()
        assert server.requests == []
        session.blocked = False
        assert await asyncio.wait_for(_translate(provider), timeout=1) == "next"
        assert len(server.sockets) == 1
    finally:
        gate.set()
        await execution.close()
        await provider.close()


async def test_closed_execution_returns_only_unused_slots_while_sent_attempt_drains() -> None:
    server = _FakeServer(lambda _request: [])
    provider = _provider(server, _FakeSession(), max_connections=2, prepared_connections=2)
    await provider.prepare_connections()
    execution = await _admit(provider)
    try:
        loser = asyncio.create_task(execution.translate_attempt(0))
        await _wait_for_requests(server, 1)
        occupied = next(socket for socket in server.sockets if socket.sent)
        loser.cancel()
        with pytest.raises(asyncio.CancelledError):
            await loser
        await asyncio.wait_for(execution.close(), timeout=1)
        await execution.close()
        with pytest.raises(asyncio.CancelledError):
            await execution.translate_attempt(1)
        server.script = _completed("next")
        assert await asyncio.wait_for(_translate(provider), timeout=1) == "next"
        assert len(occupied.sent) == 1
        occupied._queue.put_nowait(
            json.dumps({"type": "response.output_text.delta", "delta": "old tail"})
        )
        occupied._queue.put_nowait(json.dumps({"type": "response.completed"}))
    finally:
        await execution.close()
        await provider.close()
    assert server.closed == 2


@pytest.mark.parametrize("invalidate", ["token", "release", "close"])
async def test_late_reservation_invalidation_never_sends_stale_work(invalidate: str) -> None:
    server = _FakeServer(_completed("fresh"))
    session = _FakeSession()
    provider = _provider(server, session, max_connections=2, prepared_connections=2)
    await provider.prepare_connections()
    execution = await _admit(provider)
    try:
        if invalidate == "token":
            session.rotate()
            with pytest.raises(RuntimeError, match="reservation"):
                await execution.translate_attempt(0)
        else:
            if invalidate == "release":
                await provider.release_connections()
            else:
                await provider.close()
            with pytest.raises(asyncio.CancelledError):
                await execution.translate_attempt(0)
        await execution.close()
        assert server.requests == []
        assert server.closed == 2
        if invalidate != "close":
            assert await asyncio.wait_for(_translate(provider), timeout=1) == "fresh"
            assert server.sockets[-1].token == session.token
    finally:
        await execution.close()
        await provider.close()


async def test_opening_failure_fails_waiter_and_restores_physical_capacity() -> None:
    server = _FakeServer(_completed("recovered"))
    entered = asyncio.Event()
    gate = asyncio.Event()
    failed = False

    async def connect(url: str, headers: Mapping[str, str]) -> _FakeSocket:
        nonlocal failed
        if not failed:
            failed = True
            entered.set()
            await gate.wait()
            raise OSError("opening failed")
        return await server.connect(url, headers)

    provider = ChatGptPlanLLMProvider(session=_FakeSession(), connector=connect, max_connections=1)
    try:
        admission = asyncio.create_task(_admit(provider))
        await entered.wait()
        gate.set()
        with pytest.raises(OSError, match="opening failed"):
            await asyncio.wait_for(admission, timeout=1)
        assert server.requests == []
        assert await asyncio.wait_for(_translate(provider), timeout=1) == "recovered"
        assert len(server.sockets) == 1
    finally:
        gate.set()
        await provider.close()


@pytest.mark.parametrize("shutdown", ["release", "close"])
async def test_shutdown_retires_late_opening_and_cancels_admission(shutdown: str) -> None:
    server = _FakeServer(_completed("fresh"))
    entered = asyncio.Event()
    gate = asyncio.Event()
    blocked = True

    async def connect(url: str, headers: Mapping[str, str]) -> _FakeSocket:
        if blocked:
            entered.set()
            try:
                await gate.wait()
            except asyncio.CancelledError:
                pass
        return await server.connect(url, headers)

    provider = ChatGptPlanLLMProvider(session=_FakeSession(), connector=connect, max_connections=1)
    try:
        admission = asyncio.create_task(_admit(provider))
        await entered.wait()
        if shutdown == "release":
            await asyncio.wait_for(provider.release_connections(), timeout=1)
        else:
            await asyncio.wait_for(provider.close(), timeout=1)
        with pytest.raises(asyncio.CancelledError):
            await admission
        assert server.requests == []
        assert server.closed == 1
        assert server.sockets[0].close_code == 1000
        if shutdown == "release":
            blocked = False
            assert await asyncio.wait_for(_translate(provider), timeout=1) == "fresh"
            assert len(server.sockets) == 2
    finally:
        gate.set()
        await provider.close()


async def test_close_owns_reservations_opening_and_draining_exchange() -> None:
    server = _FakeServer(lambda _request: [])
    entered = asyncio.Event()
    gate = asyncio.Event()

    async def connect(url: str, headers: Mapping[str, str]) -> _FakeSocket:
        if len(server.sockets) == 3:
            entered.set()
            try:
                await gate.wait()
            except asyncio.CancelledError:
                pass
        return await server.connect(url, headers)

    provider = ChatGptPlanLLMProvider(
        session=_FakeSession(), connector=connect, max_connections=4, prepared_connections=3
    )
    await provider.prepare_connections()
    execution = await _admit(provider)
    held = await _admit(provider, max_attempts=1)
    try:
        draining = asyncio.create_task(execution.translate_attempt(0))
        await _wait_for_requests(server, 1)
        draining.cancel()
        with pytest.raises(asyncio.CancelledError):
            await draining
        queued = asyncio.create_task(_admit(provider))
        await entered.wait()
        await asyncio.wait_for(provider.close(), timeout=1)
        with pytest.raises(asyncio.CancelledError):
            await queued
        await asyncio.gather(execution.close(), held.close())
        assert len(server.requests) == 1
        assert len(server.sockets) == 4
        assert server.closed == 4
        assert all(socket.close_code == 1000 for socket in server.sockets)
    finally:
        gate.set()
        await asyncio.gather(execution.close(), held.close())
        await provider.close()


async def test_response_unauthorized_retries_once_with_the_same_logical_payload() -> None:
    calls = 0

    def script(request: dict[str, object]) -> list[dict[str, object]]:
        nonlocal calls
        calls += 1
        if calls == 1:
            return [{"type": "error", "status": 401}]
        text = request["input"][-1]["content"][0]["text"]
        return _completed(text)(request)

    server = _FakeServer(script)
    session = _FakeSession()
    provider = _provider(server, session, max_connections=1)
    try:
        translated = await _translate(provider, "retry text")
        assert "retry text" in translated
        assert session.invalidated == ["token-1"]
        assert [socket.token for socket in server.sockets] == ["token-1", "token-2"]
        assert server.closed == 1
        assert len(server.requests) == 2
    finally:
        await provider.close()


async def test_repeated_unauthorized_response_stops_after_one_retry() -> None:
    server = _FakeServer(lambda _request: [{"type": "error", "status": 401}])
    session = _FakeSession()
    provider = _provider(server, session, max_connections=1)
    try:
        with pytest.raises(ChatGptPlanResponseError) as exc_info:
            await asyncio.wait_for(_translate(provider), timeout=1)
        assert exc_info.value.status_code == 401
        assert len(server.requests) == 2
        assert session.invalidated == ["token-1", "token-2"]
        assert server.closed == 2
    finally:
        await provider.close()


@pytest.mark.parametrize("status", [401, 503])
async def test_released_in_flight_failure_does_not_reopen_pool(status: int) -> None:
    server = _FakeServer(lambda _request: [])
    provider = _provider(server, _FakeSession(), max_connections=1)
    try:
        translation = asyncio.create_task(_translate(provider))
        await _wait_for_requests(server, 1)
        await provider.release_connections()
        server.sockets[0]._queue.put_nowait(
            json.dumps({"type": "error", "status": status, "error": {"code": "failed"}})
        )
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(translation, timeout=1)
        assert len(server.requests) == 1
        assert len(server.sockets) == 1
        assert server.closed == 1
    finally:
        await provider.close()


async def test_cancellation_between_exchange_scheduling_and_send_never_sends() -> None:
    attempt: asyncio.Task | None = None

    class CancelBeforeSendProvider(ChatGptPlanLLMProvider):
        async def _exchange(self, connection, message, started, abandoned) -> str:
            if attempt is not None:
                attempt.cancel()
                await asyncio.sleep(0)
            return await super()._exchange(connection, message, started, abandoned)

    server = _FakeServer(_completed("next"))
    provider = CancelBeforeSendProvider(
        session=_FakeSession(), connector=server.connect, max_connections=1
    )
    execution = await _admit(provider)
    try:
        attempt = asyncio.create_task(execution.translate_attempt(0))
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(attempt, timeout=1)
        attempt = None
        await execution.close()
        assert server.requests == []
        assert await asyncio.wait_for(_translate(provider), timeout=1) == "next"
        assert len(server.sockets) == 1
    finally:
        await execution.close()
        await provider.close()


async def test_abandoned_admission_during_opening_never_sends_or_leaks_capacity() -> None:
    server = _FakeServer(_completed("next"))
    entered = asyncio.Event()
    gate = asyncio.Event()

    async def connect(url: str, headers: Mapping[str, str]) -> _FakeSocket:
        entered.set()
        await gate.wait()
        return await server.connect(url, headers)

    provider = ChatGptPlanLLMProvider(session=_FakeSession(), connector=connect, max_connections=1)
    try:
        admission = asyncio.create_task(_admit(provider, "abandoned"))
        await entered.wait()
        admission.cancel()
        with pytest.raises(asyncio.CancelledError):
            await admission
        next_request = asyncio.create_task(_translate(provider, "next"))
        gate.set()
        assert await asyncio.wait_for(next_request, timeout=1) == "next"
        assert len(server.sockets) == 1
        assert len(server.requests) == 1
        assert "next" in server.requests[0]["input"][-1]["content"][0]["text"]
    finally:
        gate.set()
        await provider.close()
    assert server.closed == 1


async def test_token_rotated_during_opening_retires_socket_before_admission() -> None:
    server = _FakeServer(_completed("fresh"))
    session = _FakeSession()
    entered = asyncio.Event()
    gate = asyncio.Event()

    async def connect(url: str, headers: Mapping[str, str]) -> _FakeSocket:
        entered.set()
        await gate.wait()
        return await server.connect(url, headers)

    provider = ChatGptPlanLLMProvider(session=session, connector=connect, max_connections=1)
    try:
        translation = asyncio.create_task(_translate(provider))
        await entered.wait()
        session.rotate()
        gate.set()
        assert await asyncio.wait_for(translation, timeout=1) == "fresh"
        assert [socket.token for socket in server.sockets] == ["token-1", "token-2"]
        assert server.sockets[0].sent == []
        assert server.sockets[0].close_code == 1000
        assert session.invalidated == []
        assert len(server.requests) == 1
    finally:
        gate.set()
        await provider.close()


async def test_cancelled_opening_before_start_does_not_keep_a_physical_slot() -> None:
    class CancelFirstOpeningProvider(ChatGptPlanLLMProvider):
        cancel_first = True

        def _start_opening(self):
            task = super()._start_opening()
            if self.cancel_first:
                self.cancel_first = False
                task.cancel()
            return task

    server = _FakeServer(_completed("next"))
    provider = CancelFirstOpeningProvider(
        session=_FakeSession(), connector=server.connect, max_connections=1
    )
    try:
        await asyncio.wait_for(provider.prepare_connections(), timeout=1)
        assert server.sockets == []
        assert await asyncio.wait_for(_translate(provider), timeout=1) == "next"
        assert len(server.sockets) == 1
    finally:
        await provider.close()
    assert server.closed == 1


@pytest.mark.parametrize("phase", ["admission", "lease"])
@pytest.mark.parametrize("shutdown", ["release", "close"])
async def test_shutdown_cancels_authentication_before_sending(phase: str, shutdown: str) -> None:
    entered = asyncio.Event()
    gate = asyncio.Event()

    class GatedSession(_FakeSession):
        blocked = False

        async def access_token(self) -> str:
            if self.blocked:
                entered.set()
                await gate.wait()
            return await super().access_token()

    session = GatedSession()
    server = _FakeServer(_completed("fresh"))
    provider = _provider(server, session, max_connections=1)
    execution = None
    try:
        if phase == "lease":
            execution = await _admit(provider)
        session.blocked = True
        operation = asyncio.create_task(
            _admit(provider) if execution is None else execution.translate_attempt(0)
        )
        await entered.wait()
        if shutdown == "release":
            await asyncio.wait_for(provider.release_connections(), timeout=1)
        else:
            await asyncio.wait_for(provider.close(), timeout=1)
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(operation, timeout=1)
        if execution is not None:
            await execution.close()
        assert server.requests == []
        assert all(socket.close_code == 1000 for socket in server.sockets)
        if shutdown == "release":
            session.blocked = False
            assert await asyncio.wait_for(_translate(provider), timeout=1) == "fresh"
    finally:
        gate.set()
        if execution is not None:
            await execution.close()
        await provider.close()


async def test_repeated_handshake_unauthorized_stops_after_one_retry() -> None:
    server = _FakeServer(_completed("unexpected"))
    server.reject_tokens.update({"token-1", "token-2", "token-3"})
    session = _FakeSession()
    provider = _provider(server, session, max_connections=1)
    try:
        with pytest.raises(ChatGptPlanResponseError) as exc_info:
            await asyncio.wait_for(_translate(provider), timeout=1)
        assert exc_info.value.status_code == 401
        assert session.invalidated == ["token-1", "token-2"]
        assert server.requests == []
        assert server.sockets == []
    finally:
        await provider.close()


async def test_executions_with_the_same_utterance_id_keep_distinct_payloads_and_slots() -> None:
    def script(request: dict[str, object]) -> list[dict[str, object]]:
        text = request["input"][-1]["content"][0]["text"]
        return _completed("first" if "first" in text else "second")(request)

    server = _FakeServer(script)
    provider = _provider(server, _FakeSession(), max_connections=2, prepared_connections=2)
    executions: list[LLMRequestExecution] = []
    utterance_id = uuid4()
    try:
        await provider.prepare_connections()
        executions = await asyncio.gather(
            *(
                provider.admit_request(
                    utterance_id=utterance_id,
                    text=text,
                    system_prompt="Translate.",
                    source_language="Korean",
                    target_language="English",
                    max_attempts=1,
                )
                for text in ("first", "second")
            )
        )
        results = await asyncio.gather(
            *(execution.translate_attempt(0) for execution in executions)
        )
        assert [(result.utterance_id, result.text) for result in results] == [
            (utterance_id, "first"),
            (utterance_id, "second"),
        ]
        assert [len(socket.sent) for socket in server.sockets] == [1, 1]
        with pytest.raises(RuntimeError, match="already consumed"):
            await executions[0].translate_attempt(0)
        assert len(server.requests) == 2
    finally:
        await asyncio.gather(*(execution.close() for execution in executions))
        await provider.close()


@pytest.mark.parametrize("shutdown", ["release", "close"])
async def test_shutdown_during_opening_authentication_never_starts_a_socket(
    shutdown: str,
) -> None:
    entered = asyncio.Event()
    gate = asyncio.Event()

    class GatedSession(_FakeSession):
        calls = 0

        async def access_token(self) -> str:
            self.calls += 1
            if self.calls == 2:
                entered.set()
                try:
                    await gate.wait()
                except asyncio.CancelledError:
                    pass
            return await super().access_token()

    server = _FakeServer(_completed("fresh"))
    provider = _provider(server, GatedSession(), max_connections=1)
    try:
        admission = asyncio.create_task(_admit(provider))
        await entered.wait()
        if shutdown == "release":
            await asyncio.wait_for(provider.release_connections(), timeout=1)
        else:
            await asyncio.wait_for(provider.close(), timeout=1)
        with pytest.raises(asyncio.CancelledError):
            await admission
        assert server.requests == []
        assert server.sockets == []
        if shutdown == "release":
            assert await asyncio.wait_for(_translate(provider), timeout=1) == "fresh"
    finally:
        gate.set()
        await provider.close()
