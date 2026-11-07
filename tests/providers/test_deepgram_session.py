from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

import pytest

from puripuly_heart.providers.stt.deepgram import _DeepgramSDKSession


def _make_session(
    *,
    model: str = "nova-3",
    keyterms: list[str] | None = None,
    stream_label: str | None = None,
) -> _DeepgramSDKSession:
    return _DeepgramSDKSession(
        api_key="k",
        model=model,
        language="en",
        sample_rate_hz=16000,
        connect_timeout_s=5.0,
        keyterms=keyterms or [],
        stream_label=stream_label,
    )


@pytest.mark.asyncio
async def test_deepgram_session_report_error_is_emitted_once() -> None:
    session = _make_session()

    err = RuntimeError("boom")
    session._report_error(err)
    session._report_error(RuntimeError("second"))
    await asyncio.sleep(0)

    assert session._error_reported is True
    assert await session._event_projection._legacy_events.get() is err
    assert session._event_projection._legacy_events.empty()


@pytest.fixture
def controlled_connection(monkeypatch: pytest.MonkeyPatch):
    from deepgram.core.events import EventType

    from puripuly_heart.core import network_clients

    class Connection:
        def __init__(self):
            self.callbacks = {}
            self.write_started = asyncio.Event()
            self.write_gate = asyncio.Event()
            self.write_gate.set()
            self.listen_started = asyncio.Event()
            self.closed = asyncio.Event()
            self.written = []
            self.write_error = None

        def on(self, event, callback):
            self.callbacks[event] = callback

        async def start_listening(self):
            self.callbacks[EventType.OPEN](None)
            self.listen_started.set()
            try:
                await self.closed.wait()
            finally:
                self.callbacks[EventType.CLOSE](None)

        async def send_media(self, data):
            self.write_started.set()
            await self.write_gate.wait()
            if self.write_error is not None:
                raise self.write_error
            self.written.append(data)

        async def send_finalize(self):
            self.written.append("Finalize")

        async def send_close_stream(self):
            self.written.append("CloseStream")

        async def send_keep_alive(self):
            self.written.append("KeepAlive")

    connection = Connection()
    clients = []
    original_factory = network_clients.external_async_client

    def client_factory(**kwargs):
        client = original_factory(**kwargs)
        clients.append(client)
        return client

    @asynccontextmanager
    async def connect(*_args, **_kwargs):
        try:
            yield connection
        finally:
            connection.closed.set()

    monkeypatch.setattr(network_clients, "external_async_client", client_factory)
    monkeypatch.setattr("puripuly_heart.providers.stt.sdk_network.deepgram_listen_connect", connect)
    return connection, clients


@pytest.mark.asyncio
async def test_deepgram_writer_ack_waits_for_write_and_close_drains_fifo(controlled_connection):
    connection, clients = controlled_connection
    session = _make_session()
    await session.start()
    connection.write_gate.clear()
    first = asyncio.create_task(session._write_payload(b"first"))
    await asyncio.wait_for(connection.write_started.wait(), 1)
    second = asyncio.create_task(session._write_payload(b"second"))
    await asyncio.sleep(0)
    assert not first.done()
    assert not second.done()
    await session.on_speech_end()
    closing = asyncio.create_task(session.close())
    await asyncio.sleep(0)
    assert not closing.done()
    connection.write_gate.set()
    await asyncio.wait_for(asyncio.gather(first, second, closing), 1)
    assert connection.written == [b"first", b"second", "Finalize"]
    assert connection.closed.is_set()
    assert all(client.is_closed for client in clients)
    assert all(
        task.done() for task in (session._send_task, session._recv_task, session._keepalive_task)
    )


@pytest.mark.asyncio
async def test_deepgram_writer_failure_fails_inflight_and_pending_writes(controlled_connection):
    connection, clients = controlled_connection
    session = _make_session()
    await session.start()
    connection.write_gate.clear()
    error = OSError("controlled writer failure")
    connection.write_error = error
    first = asyncio.create_task(session._write_payload(b"first"))
    await asyncio.wait_for(connection.write_started.wait(), 1)
    second = asyncio.create_task(session._write_payload(b"second"))
    await asyncio.sleep(0)
    connection.write_gate.set()
    results = await asyncio.wait_for(asyncio.gather(first, second, return_exceptions=True), 1)
    assert results[0] is error
    assert isinstance(results[1], RuntimeError)
    with pytest.raises(OSError, match="controlled writer failure"):
        await anext(session.events())
    await asyncio.wait_for(session.close(), 1)
    assert connection.closed.is_set()
    assert all(client.is_closed for client in clients)
    assert all(
        task.done() for task in (session._send_task, session._recv_task, session._keepalive_task)
    )


@pytest.mark.asyncio
async def test_deepgram_legacy_audio_overflow_reserves_control_capacity():
    session = _make_session()
    for _ in range(256):
        await session.send_audio(b"pcm")
    with pytest.raises(RuntimeError, match="audio queue overflow"):
        await session.send_audio(b"overflow")
    await session.on_speech_end()
    await session.on_speech_end()
    await asyncio.wait_for(session.close(), 1)


@pytest.mark.asyncio
async def test_deepgram_cancelled_close_fails_blocked_writes_and_joins_owned_tasks(
    controlled_connection,
):
    connection, clients = controlled_connection
    session = _make_session()
    await session.start()
    connection.write_gate.clear()
    first = asyncio.create_task(session._write_payload(b"first"))
    await asyncio.wait_for(connection.write_started.wait(), 1)
    second = asyncio.create_task(session._write_payload(b"second"))
    await asyncio.sleep(0)
    closing = asyncio.create_task(session.close())
    await asyncio.sleep(0)
    closing.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(closing, 1)
    results = await asyncio.wait_for(asyncio.gather(first, second, return_exceptions=True), 1)
    assert all(isinstance(result, RuntimeError) for result in results)
    assert connection.closed.is_set()
    assert all(client.is_closed for client in clients)
    assert all(
        task.done() for task in (session._send_task, session._recv_task, session._keepalive_task)
    )
