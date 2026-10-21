from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from types import SimpleNamespace
from uuid import uuid4

from puripuly_heart.app.services.llm_connection_readiness import (
    LlmConnectionReadinessOwner,
    apply_llm_connection_mode,
)
from puripuly_heart.providers.llm.chatgpt_plan import ChatGptPlanLLMProvider


class _Session:
    token_generation = 1

    async def access_token(self) -> str:
        return "test-token"

    def invalidate_access_token(self, _token: str) -> None:
        raise AssertionError("Unexpected authentication retry")


class _Socket:
    def __init__(self, server: _Server) -> None:
        self.server = server
        self.close_code: int | None = None
        self.events: asyncio.Queue[str] = asyncio.Queue()
        self.close_started = asyncio.Event()
        self.closed = asyncio.Event()
        self.close_gate: asyncio.Event | None = None

    async def send(self, request: str) -> None:
        self.server.requests.append((self, request))
        self.events.put_nowait(json.dumps({"type": "response.output_text.delta", "delta": "Hello"}))
        self.events.put_nowait(
            json.dumps({"type": "response.completed", "response": {"id": "resp", "output": []}})
        )

    async def recv(self) -> str:
        return await self.events.get()

    async def close(self) -> None:
        self.close_started.set()
        if self.close_gate is not None:
            await self.close_gate.wait()
        self.close_code = 1000
        self.closed.set()


class _Server:
    def __init__(self) -> None:
        self.sockets: list[_Socket] = []
        self.requests: list[tuple[_Socket, str]] = []
        self.attempts: list[asyncio.Event] = []
        self.connecting: asyncio.Queue[asyncio.Event] = asyncio.Queue()
        self.connected: asyncio.Queue[_Socket] = asyncio.Queue()
        self.reject_new_connections = False
        self.gate: asyncio.Event | None = None

    async def connect(self, _url: str, _headers: Mapping[str, str]) -> _Socket:
        if self.reject_new_connections:
            raise AssertionError("Manual translation opened an unprepared connection")
        finished = asyncio.Event()
        self.attempts.append(finished)
        self.connecting.put_nowait(finished)
        try:
            if self.gate is not None:
                await self.gate.wait()
            socket = _Socket(self)
            self.sockets.append(socket)
            self.connected.put_nowait(socket)
            return socket
        finally:
            finished.set()

    async def wait_connected(self, count: int = 1) -> list[_Socket]:
        async with asyncio.timeout(1):
            return [await self.connected.get() for _ in range(count)]

    async def wait_connecting(self, count: int = 1) -> list[asyncio.Event]:
        async with asyncio.timeout(1):
            return [await self.connecting.get() for _ in range(count)]


def _owner(provider: ChatGptPlanLLMProvider, state: dict[str, bool]) -> LlmConnectionReadinessOwner:
    return LlmConnectionReadinessOwner(
        llm_provider=lambda: SimpleNamespace(
            provider=SimpleNamespace(inner=SimpleNamespace(primary=provider))
        ),
        translation_enabled=lambda: state["translation"],
    )


async def _applied(owner: LlmConnectionReadinessOwner) -> None:
    task = owner._task
    assert task is not None
    await asyncio.wait_for(asyncio.shield(task), 1)


async def _translate(provider: ChatGptPlanLLMProvider):
    return await provider.translate(
        utterance_id=uuid4(),
        text="안녕",
        system_prompt="Translate Korean to English.",
        source_language="Korean",
        target_language="English",
    )


async def test_manual_translation_reuses_standby_connection_without_capture() -> None:
    server = _Server()
    provider = ChatGptPlanLLMProvider(
        session=_Session(), connector=server.connect, prepared_connections=2, max_connections=2
    )
    state = {"translation": False}
    owner = _owner(provider, state)
    try:
        owner.sync()
        owner.sync()
        await _applied(owner)
        (standby,) = await server.wait_connected()
        assert server.requests == []
        assert len(server.attempts) == 1
        server.reject_new_connections = True

        assert (await asyncio.wait_for(_translate(provider), 1)).text == "Hello"
        assert [socket for socket, _ in server.requests] == [standby]
        owner.sync()
        await _applied(owner)
        assert len(server.attempts) == 1
        assert standby.close_code is None
    finally:
        await owner.close()
        await provider.close()
    assert standby.close_code == 1000


async def test_fast_off_on_reuses_standby_socket_during_blocked_expansion() -> None:
    server = _Server()
    provider = ChatGptPlanLLMProvider(
        session=_Session(), connector=server.connect, prepared_connections=2, max_connections=2
    )
    state = {"translation": False}
    owner = _owner(provider, state)
    gate = asyncio.Event()
    try:
        owner.sync()
        (standby,) = await server.wait_connected()
        await server.wait_connecting()
        server.gate = gate
        state["translation"] = True
        owner.sync()
        owner.sync()
        await _applied(owner)
        await server.wait_connecting()
        assert len(server.attempts) == 2

        state["translation"] = False
        owner.sync()
        await _applied(owner)
        assert standby.close_code is None
        state["translation"] = True
        owner.sync()
        await _applied(owner)
        assert standby.close_code is None
        gate.set()
        (expanded,) = await server.wait_connected()
        server.reject_new_connections = True

        results = await asyncio.wait_for(
            asyncio.gather(_translate(provider), _translate(provider)), 1
        )
        assert [result.text for result in results] == ["Hello", "Hello"]
        assert {socket for socket, _ in server.requests} == {standby, expanded}
        assert all(socket.close_code is None for socket in server.sockets)
        owner.sync()
        await _applied(owner)
        assert len(server.sockets) == 2
    finally:
        gate.set()
        await owner.close()
        await provider.close()


async def test_off_keeps_one_socket_while_surplus_close_is_blocked() -> None:
    server = _Server()
    provider = ChatGptPlanLLMProvider(
        session=_Session(), connector=server.connect, prepared_connections=2, max_connections=2
    )
    state = {"translation": True}
    owner = _owner(provider, state)
    gate = asyncio.Event()
    try:
        owner.sync()
        sockets = await server.wait_connected(2)
        for socket in sockets:
            socket.close_gate = gate
        state["translation"] = False
        owner.sync()
        await _applied(owner)
        closing = [asyncio.create_task(socket.close_started.wait()) for socket in sockets]
        try:
            await asyncio.wait_for(asyncio.wait(closing, return_when=asyncio.FIRST_COMPLETED), 1)
        finally:
            for task in closing:
                task.cancel()
            await asyncio.gather(*closing, return_exceptions=True)
        assert sum(socket.close_started.is_set() for socket in sockets) == 1
        retained = next(socket for socket in sockets if not socket.close_started.is_set())
        server.reject_new_connections = True

        assert (await asyncio.wait_for(_translate(provider), 1)).text == "Hello"
        assert [socket for socket, _ in server.requests] == [retained]
        assert retained.close_code is None
        gate.set()
        for socket in sockets:
            if socket is not retained:
                await asyncio.wait_for(socket.closed.wait(), 1)
        assert retained.close_code is None
    finally:
        gate.set()
        await owner.close()
        await provider.close()


async def test_replacement_starts_before_previous_blocked_preparation_finishes() -> None:
    old_server = _Server()
    old_server.gate = asyncio.Event()
    new_server = _Server()
    old_provider = ChatGptPlanLLMProvider(
        session=_Session(), connector=old_server.connect, prepared_connections=2, max_connections=2
    )
    new_provider = ChatGptPlanLLMProvider(
        session=_Session(), connector=new_server.connect, prepared_connections=2, max_connections=2
    )
    current = [old_provider]
    owner = LlmConnectionReadinessOwner(
        llm_provider=lambda: current[0], translation_enabled=lambda: True
    )
    try:
        owner.sync()
        pending = await old_server.wait_connecting(2)
        current[0] = new_provider
        owner.sync()
        await _applied(owner)
        await new_server.wait_connected(2)
        assert not old_server.gate.is_set()
        assert old_server.requests == []
        assert new_server.requests == []
        new_server.reject_new_connections = True
        assert (await asyncio.wait_for(_translate(new_provider), 1)).text == "Hello"

        old_server.gate.set()
        for finished in pending:
            await asyncio.wait_for(finished.wait(), 1)
        for socket in old_server.sockets:
            await asyncio.wait_for(socket.closed.wait(), 1)
        assert len(old_server.attempts) == 2
    finally:
        old_server.gate.set()
        await owner.close()
        await old_provider.close()
        await new_provider.close()


async def test_suspend_prevents_queued_startup_and_resume_uses_latest_mode() -> None:
    server = _Server()
    provider = ChatGptPlanLLMProvider(
        session=_Session(), connector=server.connect, prepared_connections=2, max_connections=2
    )
    state = {"translation": True}
    owner = _owner(provider, state)
    try:
        owner.sync()
        await asyncio.wait_for(owner.suspend(), 1)
        state["translation"] = False
        owner.sync()
        await _applied(owner)
        assert server.attempts == []
        owner.resume()
        (standby,) = await server.wait_connected()
        assert len(server.attempts) == 1
        assert standby.close_code is None
        assert server.requests == []
    finally:
        await owner.close()
        await provider.close()


async def test_suspended_replacement_cannot_rewarm_during_concurrent_sync() -> None:
    old_server = _Server()
    new_server = _Server()
    old_provider = ChatGptPlanLLMProvider(
        session=_Session(), connector=old_server.connect, prepared_connections=2, max_connections=2
    )
    new_provider = ChatGptPlanLLMProvider(
        session=_Session(), connector=new_server.connect, prepared_connections=2, max_connections=2
    )
    current = [old_provider]
    state = {"translation": False}
    owner = LlmConnectionReadinessOwner(
        llm_provider=lambda: current[0], translation_enabled=lambda: state["translation"]
    )
    gate = asyncio.Event()
    try:
        owner.sync()
        (standby,) = await old_server.wait_connected()
        standby.close_gate = gate
        await asyncio.wait_for(owner.suspend(), 1)
        await asyncio.wait_for(standby.close_started.wait(), 1)
        assert not gate.is_set()
        current[0] = new_provider

        async def sync(enabled: bool) -> None:
            state["translation"] = enabled
            owner.sync()

        await asyncio.gather(sync(True), sync(False), sync(True))
        await _applied(owner)
        assert new_server.attempts == []
        assert len(old_server.attempts) == 1
        gate.set()
        await asyncio.wait_for(standby.closed.wait(), 1)
        owner.sync()
        await _applied(owner)
        assert new_server.attempts == []
        owner.resume()
        await new_server.wait_connected(2)
        assert new_server.requests == []
        assert len(old_server.attempts) == 1
    finally:
        gate.set()
        await owner.close()
        await old_provider.close()
        await new_provider.close()


async def test_close_prevents_later_startup_without_closing_provider_runtime() -> None:
    server = _Server()
    server.gate = asyncio.Event()
    replacement_server = _Server()
    provider = ChatGptPlanLLMProvider(
        session=_Session(), connector=server.connect, prepared_connections=2, max_connections=2
    )
    replacement = ChatGptPlanLLMProvider(session=_Session(), connector=replacement_server.connect)
    current = [provider]
    owner = LlmConnectionReadinessOwner(
        llm_provider=lambda: current[0], translation_enabled=lambda: True
    )
    try:
        owner.sync()
        pending = await server.wait_connecting(2)
        await asyncio.wait_for(owner.close(), 1)
        current[0] = replacement
        owner.sync()
        owner.resume()
        await owner.suspend()
        await owner.close()
        server.gate.set()
        for finished in pending:
            await asyncio.wait_for(finished.wait(), 1)
        for socket in server.sockets:
            await asyncio.wait_for(socket.closed.wait(), 1)
        assert replacement_server.attempts == []
        assert len(server.attempts) == 2

        assert (await asyncio.wait_for(_translate(provider), 1)).text == "Hello"
        assert len(server.requests) == 1
        assert replacement_server.attempts == []
    finally:
        server.gate.set()
        await owner.close()
        await provider.close()
        await replacement.close()
    assert all(socket.close_code == 1000 for socket in server.sockets)


async def test_unsupported_routes_are_noop_without_owning_provider_shutdown() -> None:
    class UnsupportedProvider:
        async def close(self) -> None:
            raise AssertionError("Readiness does not own provider shutdown")

    provider = SimpleNamespace(provider=SimpleNamespace(inner=UnsupportedProvider()))
    for mode in ("inactive", "standby", "active"):
        assert not await apply_llm_connection_mode(provider, mode)
    assert not await apply_llm_connection_mode(None, "standby")
    owner = LlmConnectionReadinessOwner(
        llm_provider=lambda: provider, translation_enabled=lambda: False
    )
    owner.sync()
    await owner.suspend()
    owner.resume()
    await owner.close()
    owner.sync()
    assert owner._task is None


async def test_close_releases_replaced_provider_before_queued_cleanup_runs() -> None:
    server = _Server()
    provider = ChatGptPlanLLMProvider(session=_Session(), connector=server.connect)
    current: list[object | None] = [provider]
    owner = LlmConnectionReadinessOwner(
        llm_provider=lambda: current[0], translation_enabled=lambda: False
    )
    try:
        owner.sync()
        (standby,) = await server.wait_connected()
        current[0] = object()
        owner.sync()
        await asyncio.wait_for(owner.close(), 1)
        await asyncio.wait_for(standby.closed.wait(), 1)
        assert len(server.attempts) == 1
        assert server.requests == []
        owner.sync()
        owner.resume()
        assert standby.close_code == 1000
    finally:
        await owner.close()
        await provider.close()
