from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from types import SimpleNamespace
from uuid import uuid4

from puripuly_heart.app.services.llm_connection_readiness import LlmConnectionReadinessOwner
from puripuly_heart.providers.llm.chatgpt_plan import ChatGptPlanLLMProvider


class _Session:
    token_generation = 1

    async def access_token(self) -> str:
        return "test-token"

    def invalidate_access_token(self, _token: str) -> None:
        raise AssertionError("Unexpected authentication retry")


class _Socket:
    def __init__(self) -> None:
        self.close_code: int | None = None
        self.events: asyncio.Queue[str] = asyncio.Queue()
        self.close_started = asyncio.Event()
        self.close_gate: asyncio.Event | None = None

    async def send(self, _request: str) -> None:
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


class _Server:
    def __init__(self) -> None:
        self.sockets: list[_Socket] = []
        self.reject_new_connections = False
        self.gate: asyncio.Event | None = None
        self.started = asyncio.Event()
        self.attempts = 0

    async def connect(self, _url: str, _headers: Mapping[str, str]) -> _Socket:
        if self.reject_new_connections:
            raise AssertionError("Manual translation opened an unprepared connection")
        self.attempts += 1
        if self.attempts == 2:
            self.started.set()
        if self.gate is not None:
            await self.gate.wait()
        socket = _Socket()
        self.sockets.append(socket)
        return socket


def _owner(provider: ChatGptPlanLLMProvider, state: dict[str, bool]) -> LlmConnectionReadinessOwner:
    wrapped = SimpleNamespace(provider=SimpleNamespace(inner=SimpleNamespace(primary=provider)))
    return LlmConnectionReadinessOwner(
        llm_provider=lambda: wrapped,
        translation_enabled=lambda: state["translation"],
    )


async def test_manual_translation_reuses_prepared_connection_without_capture() -> None:
    server = _Server()
    provider = ChatGptPlanLLMProvider(
        session=_Session(), connector=server.connect, prepared_connections=2, max_connections=2
    )
    state = {"translation": True}
    owner = _owner(provider, state)
    try:
        owner.sync()
        await owner._task
        server.reject_new_connections = True
        result = await provider.translate(
            utterance_id=uuid4(),
            text="안녕",
            system_prompt="Translate Korean to English.",
            source_language="Korean",
            target_language="English",
        )
        assert result.text == "Hello"

        state["translation"] = False
        owner.sync()
        await owner._task
        assert [socket.close_code for socket in server.sockets] == [1000, 1000]
    finally:
        await owner.close()
        await provider.close()


async def test_disabling_translation_during_preparation_closes_eventual_connections() -> None:
    server = _Server()
    server.gate = asyncio.Event()
    provider = ChatGptPlanLLMProvider(
        session=_Session(), connector=server.connect, prepared_connections=2, max_connections=2
    )
    state = {"translation": True}
    owner = _owner(provider, state)
    try:
        owner.sync()
        await server.started.wait()
        state["translation"] = False
        owner.sync()
        server.gate.set()
        await owner._task
        assert [socket.close_code for socket in server.sockets] == [1000, 1000]
    finally:
        await owner.close()
        await provider.close()


async def test_owner_close_during_release_does_not_start_queued_preparation() -> None:
    server = _Server()
    provider = ChatGptPlanLLMProvider(
        session=_Session(), connector=server.connect, prepared_connections=2, max_connections=2
    )
    state = {"translation": True}
    owner = _owner(provider, state)
    gate = asyncio.Event()
    try:
        owner.sync()
        await owner._task
        for socket in server.sockets:
            socket.close_gate = gate
        state["translation"] = False
        owner.sync()
        await server.sockets[0].close_started.wait()
        state["translation"] = True
        owner.sync()
        closing = asyncio.create_task(owner.close())
        await asyncio.sleep(0)
        gate.set()
        await closing
        assert [socket.close_code for socket in server.sockets] == [1000, 1000]
    finally:
        gate.set()
        await owner.close()
        await provider.close()


async def test_replacement_is_prewarmed_during_previous_preparation() -> None:
    old_server = _Server()
    old_server.gate = asyncio.Event()
    new_server = _Server()
    old_provider = ChatGptPlanLLMProvider(
        session=_Session(), connector=old_server.connect, prepared_connections=2
    )
    new_provider = ChatGptPlanLLMProvider(
        session=_Session(), connector=new_server.connect, prepared_connections=2
    )
    current = [old_provider]
    owner = LlmConnectionReadinessOwner(
        llm_provider=lambda: current[0], translation_enabled=lambda: True
    )
    try:
        owner.sync()
        await old_server.started.wait()
        current[0] = new_provider
        owner.sync()
        old_server.gate.set()
        await owner._task
        new_server.reject_new_connections = True
        result = await new_provider.translate(
            utterance_id=uuid4(),
            text="안녕",
            system_prompt="Translate Korean to English.",
            source_language="Korean",
            target_language="English",
        )
        assert result.text == "Hello"
    finally:
        old_server.gate.set()
        await owner.close()
        await old_provider.close()
        await new_provider.close()
