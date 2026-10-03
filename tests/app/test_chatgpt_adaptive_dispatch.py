from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from uuid import uuid4

from puripuly_heart.app.services.llm_connection_readiness import prepare_llm_connections
from puripuly_heart.app.wiring.wiring_llm_factory import create_llm_provider_from_resolved_config
from puripuly_heart.config.runtime_resolution import (
    RuntimeResolutionInput,
    TranslationRuntimeIntent,
    resolve_llm_config,
)
from puripuly_heart.core.storage.secrets import InMemorySecretStore


class _Session:
    token_generation = 1

    async def access_token(self) -> str:
        return "fixture-token"

    def invalidate_access_token(self, token: str) -> None:
        raise AssertionError("Unexpected authentication failure")


class _Socket:
    def __init__(self, server: _Server) -> None:
        self.server = server
        self.events: asyncio.Queue[str] = asyncio.Queue()
        self.close_code: int | None = None

    async def send(self, raw: str) -> None:
        request = json.loads(raw)
        self.server.requests.append((self, request))
        self.server.received.put_nowait((self, request))

    async def recv(self) -> str:
        return await self.events.get()

    async def close(self) -> None:
        self.close_code = 1000

    def complete(self, text: str) -> None:
        self.events.put_nowait(json.dumps({"type": "response.output_text.delta", "delta": text}))
        self.events.put_nowait(json.dumps({"type": "response.completed"}))


class _Server:
    def __init__(self) -> None:
        self.requests: list[tuple[_Socket, dict[str, object]]] = []
        self.received: asyncio.Queue[tuple[_Socket, dict[str, object]]] = asyncio.Queue()
        self.sockets: list[_Socket] = []

    async def connect(self, url: str, headers: Mapping[str, str]) -> _Socket:
        socket = _Socket(self)
        self.sockets.append(socket)
        return socket


async def _composed_provider(server: _Server):
    secrets = InMemorySecretStore()
    secrets.set("chatgpt_refresh_token", "fixture-refresh-token")
    config = resolve_llm_config(
        RuntimeResolutionInput(
            translation=TranslationRuntimeIntent(model="gpt_6_luna", connection="chatgpt")
        )
    )
    provider = create_llm_provider_from_resolved_config(
        config, secrets=secrets, chatgpt_session=_Session()
    )
    provider.inner.primary.connector = server.connect
    await prepare_llm_connections(provider)
    return provider


async def _translate(provider, text: str):
    return await provider.translate(
        utterance_id=uuid4(),
        text=text,
        system_prompt="Translate Korean to English.",
        source_language="Korean",
        target_language="English",
    )


async def test_composed_chatgpt_races_spare_connections_without_waiting_for_primary() -> None:
    server = _Server()
    provider = await _composed_provider(server)
    operation = asyncio.create_task(_translate(provider, "한 문장"))
    try:
        async with asyncio.timeout(1):
            first_socket, _ = await server.received.get()
            second_socket, _ = await server.received.get()
        second_socket.complete("The faster translation")
        result = await asyncio.wait_for(operation, 1)
        assert result.text == "The faster translation"
        assert first_socket.close_code is None
        first_socket.complete("The slower translation")
    finally:
        await provider.close()
        await asyncio.gather(operation, return_exceptions=True)
    assert len(server.requests) == 2
    assert all(socket.close_code == 1000 for socket in server.sockets)


async def test_composed_chatgpt_covers_queued_first_requests_before_duplicates() -> None:
    server = _Server()
    provider = await _composed_provider(server)
    texts = [f"문장 {index}" for index in range(6)]
    operations = [asyncio.create_task(_translate(provider, text)) for text in texts]
    try:
        async with asyncio.timeout(1):
            received = [await server.received.get() for _ in texts]
        submitted = [request["input"][0]["content"][0]["text"] for _, request in received]
        assert [[text for text in texts if text in message] for message in submitted] == [
            [text] for text in texts
        ]
        for index, (socket, _) in enumerate(received):
            socket.complete(f"Translation {index}")
        results = await asyncio.wait_for(asyncio.gather(*operations), 1)
        assert [result.text for result in results] == [f"Translation {index}" for index in range(6)]
    finally:
        await provider.close()
        await asyncio.gather(*operations, return_exceptions=True)
    assert len(server.requests) == len(texts)
    assert all(socket.close_code == 1000 for socket in server.sockets)


async def test_composed_single_failure_retries_behind_an_already_waiting_request() -> None:
    server = _Server()
    provider = await _composed_provider(server)
    held = []
    operations = []
    try:
        for _ in range(5):
            held.append(
                await provider.inner.primary.admit_request(
                    utterance_id=uuid4(),
                    text="Reserved without sending",
                    system_prompt="Translate Korean to English.",
                    source_language="Korean",
                    target_language="English",
                    max_attempts=1,
                )
            )
        first = asyncio.create_task(_translate(provider, "실패할 요청"))
        operations.append(first)
        async with asyncio.timeout(1):
            failed_socket, _ = await server.received.get()
        waiting = asyncio.create_task(_translate(provider, "먼저 기다린 요청"))
        operations.append(waiting)
        await asyncio.sleep(0)
        failed_socket.events.put_nowait(
            json.dumps(
                {
                    "type": "response.failed",
                    "response": {"error": {"code": "server_error", "status_code": 500}},
                }
            )
        )
        async with asyncio.timeout(1):
            waiting_socket, waiting_request = await server.received.get()
        assert "먼저 기다린 요청" in waiting_request["input"][0]["content"][0]["text"]
        waiting_socket.complete("Already waiting")
        assert (await asyncio.wait_for(waiting, 1)).text == "Already waiting"
        async with asyncio.timeout(1):
            recovery_socket, recovery_request = await server.received.get()
        assert "실패할 요청" in recovery_request["input"][0]["content"][0]["text"]
        recovery_socket.complete("Recovered translation")
        assert (await asyncio.wait_for(first, 1)).text == "Recovered translation"
    finally:
        await provider.close()
        await asyncio.gather(*operations, return_exceptions=True)
        for execution in held:
            await execution.close()
    assert len(server.requests) == 3
    assert all(socket.close_code == 1000 for socket in server.sockets)
