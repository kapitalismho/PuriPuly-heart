from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from uuid import uuid4

import pytest

from puripuly_heart.core.local_translation.assets import e4b_gemma_spec, gemma_26b_spec
from puripuly_heart.core.local_translation.runtime import ManagedGemmaRuntimeOwner
from puripuly_heart.core.local_translation.runtime_profile import GemmaRuntimePaths
from puripuly_heart.providers.llm.managed_gemma import (
    HttpxManagedGemmaTransport,
    ManagedGemmaLLMProvider,
)


class ProcessBoundary:
    def __init__(self):
        self.returncode = None
        self.exited = asyncio.Event()

    def terminate(self):
        self.returncode = 0
        self.exited.set()

    def kill(self):
        self.returncode = -9
        self.exited.set()

    async def wait(self):
        await self.exited.wait()
        return self.returncode


@dataclass
class CompletionServer:
    started: asyncio.Queue = field(default_factory=asyncio.Queue)
    gates: dict[str, asyncio.Event] = field(default_factory=dict)
    active: dict[int, str] = field(default_factory=dict)
    commands: list[tuple[str, ...]] = field(default_factory=list)
    peak_active: int = 0
    prefix_calls: int = 0
    failures: set[str] = field(default_factory=set)
    settlement_gates: dict[str, asyncio.Event] = field(default_factory=dict)
    settlement_observed: asyncio.Event = field(default_factory=asyncio.Event)

    async def handle(self, reader, writer):
        try:
            headers = (await reader.readuntil(b"\r\n\r\n")).decode()
            method, path, _protocol = headers.splitlines()[0].split()
            length = next(
                (
                    int(line.split(":", 1)[1])
                    for line in headers.splitlines()[1:]
                    if line.lower().startswith("content-length:")
                ),
                0,
            )
            body = json.loads(await reader.readexactly(length)) if length else {}
            status = 200
            if path == "/health":
                payload = {"status": "ok"}
            elif path == "/apply-template":
                payload = {"prompt": body["messages"][1]["content"]}
            elif path == "/slots":
                self.settlement_observed.set()
                for slot, text in tuple(self.active.items()):
                    gate = self.settlement_gates.get(text)
                    if gate is not None and gate.is_set():
                        del self.active[slot]
                payload = [{"id": slot, "is_processing": slot in self.active} for slot in range(3)]
            elif path == "/completion" and body.get("n_predict") == 1:
                self.prefix_calls += 1
                assert body["id_slot"] not in self.active
                payload = {"content": "prefix", "timings": {}}
            elif path == "/completion":
                text = body["prompt"].split("<input>\n", 1)[1].split("\n</input>", 1)[0]
                slot = body["id_slot"]
                assert slot not in self.active
                self.active[slot] = text
                self.peak_active = max(self.peak_active, len(self.active))
                self.gates[text] = asyncio.Event()
                await self.started.put((text, slot))
                await self.gates[text].wait()
                if text in self.failures:
                    status = 503
                    payload = {"error": "controlled request failure"}
                else:
                    del self.active[slot]
                    payload = {
                        "content": f"translated:{text}",
                        "timings": {
                            "prompt_n": 12,
                            "cache_n": 8,
                            "predicted_n": 4,
                            "prompt_ms": 1.0,
                            "predicted_ms": 2.0,
                        },
                    }
            else:
                raise AssertionError((method, path))
            encoded = json.dumps(payload).encode()
            writer.write(
                f"HTTP/1.1 {status} Result\r\nContent-Type: application/json\r\n"
                f"Content-Length: {len(encoded)}\r\nConnection: close\r\n\r\n".encode() + encoded
            )
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()


@pytest.fixture
async def loopback_runtime(tmp_path):
    boundary = CompletionServer()
    server = await asyncio.start_server(boundary.handle, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    executable = tmp_path / "llama-server.exe"
    executable.write_bytes(b"controlled process boundary")
    process = ProcessBoundary()

    async def provision(**kwargs):
        return None

    async def version(_executable):
        return None

    async def launch(command, cwd):
        boundary.commands.append(command)
        return process

    owner = ManagedGemmaRuntimeOwner(
        install_dir=tmp_path,
        runtime_paths=GemmaRuntimePaths(executable, executable),
        provisioner=provision,
        process_factory=launch,
        version_validator=version,
        transport_factory=lambda url: HttpxManagedGemmaTransport(
            url, timeout=2, poll_interval_s=0.01
        ),
        port_allocator=lambda: port,
        shutdown_timeout_s=0.1,
    )
    try:
        yield boundary, owner, process
    finally:
        for gate in (*boundary.gates.values(), *boundary.settlement_gates.values()):
            gate.set()
        await owner.close()
        server.close()
        await server.wait_closed()


def translate(provider, text, *, reverse=False):
    utterance_id = uuid4()
    task = asyncio.create_task(
        provider.translate(
            utterance_id=utterance_id,
            text=text,
            system_prompt="translate",
            source_language="en" if reverse else "ko",
            target_language="ko" if reverse else "en",
        )
    )
    return utterance_id, task


@pytest.mark.asyncio
@pytest.mark.parametrize("spec", [e4b_gemma_spec(), gemma_26b_spec()], ids=["e4b", "26b"])
async def test_real_http_cancellation_keeps_native_work_owned_and_other_provider_requests_progressing(
    loopback_runtime,
    spec,
):
    server, owner, process = loopback_runtime
    provider = ManagedGemmaLLMProvider(runtime=owner, backend="gpu", spec=spec)
    cancelled_id, cancelled = translate(provider, "cancelled")
    _, cancelled_slot = await asyncio.wait_for(server.started.get(), 2)
    valid_id, valid = translate(provider, "valid", reverse=True)
    _, valid_slot = await asyncio.wait_for(server.started.get(), 2)
    assert valid_slot != cancelled_slot
    cancelled.cancel()
    manual_id, manual = translate(provider, "manual")
    await asyncio.sleep(0)
    assert not cancelled.done()
    assert server.peak_active == 2
    server.gates["valid"].set()
    result = await asyncio.wait_for(valid, 2)
    assert (result.utterance_id, result.text) == (valid_id, "translated:valid")
    _, manual_slot = await asyncio.wait_for(server.started.get(), 2)
    assert manual_slot != cancelled_slot
    server.gates["manual"].set()
    result = await asyncio.wait_for(manual, 2)
    assert (result.utterance_id, result.text) == (manual_id, "translated:manual")
    assert not cancelled.done()
    server.gates["cancelled"].set()
    with pytest.raises(asyncio.CancelledError):
        await cancelled
    assert server.active == {}
    assert server.peak_active == 2
    assert len(server.commands) == 1
    assert process.returncode is None


@pytest.mark.asyncio
async def test_real_http_failure_retains_slot_until_idle_without_stopping_valid_work(
    loopback_runtime,
):
    import httpx

    server, owner, process = loopback_runtime
    provider = ManagedGemmaLLMProvider(runtime=owner, backend="gpu")
    server.failures.add("failed")
    server.settlement_gates["failed"] = asyncio.Event()
    _, failed = translate(provider, "failed")
    _, failed_slot = await asyncio.wait_for(server.started.get(), 2)
    valid_id, valid = translate(provider, "valid", reverse=True)
    await asyncio.wait_for(server.started.get(), 2)
    server.gates["failed"].set()
    await asyncio.wait_for(server.settlement_observed.wait(), 2)
    assert not failed.done()
    server.gates["valid"].set()
    result = await asyncio.wait_for(valid, 2)
    assert (result.utterance_id, result.text) == (valid_id, "translated:valid")
    assert server.active == {failed_slot: "failed"}
    server.settlement_gates["failed"].set()
    with pytest.raises(httpx.HTTPStatusError):
        await asyncio.wait_for(failed, 2)
    _, reuse = translate(provider, "reuse")
    await asyncio.wait_for(server.started.get(), 2)
    server.gates["reuse"].set()
    assert (await asyncio.wait_for(reuse, 2)).text == "translated:reuse"
    assert process.returncode is None
    assert len(server.commands) == 1
