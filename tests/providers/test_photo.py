from __future__ import annotations

import asyncio
import contextlib
import json
import time
from pathlib import Path

import httpx
import pytest
from websockets.asyncio.server import serve

from puripuly_heart.core.photo.image import load_prepared_photo
from puripuly_heart.core.photo.parsing import PhotoBlockParser
from puripuly_heart.core.photo.protocol import (
    PhotoProviderError,
    PhotoRequestSent,
    PhotoStreamCompleted,
    PhotoTextDelta,
)
from puripuly_heart.providers.photo import (
    GooglePhotoConnection,
    LunaPhotoConnection,
    OpenAIPhotoConnection,
    RelayPhotoConnection,
)


class SyntheticSession:
    token_generation = 1

    def __init__(self) -> None:
        self.calls = 0
        self.invalidated = 0

    async def access_token(self) -> str:
        self.calls += 1
        return "synthetic-session-token"

    def invalidate_access_token(self, token: str) -> None:
        assert token == "synthetic-session-token"
        self.invalidated += 1


async def synthetic_photo(tmp_path: Path):
    fixtures = Path(__file__).parents[1] / "core" / "photo" / "fixtures"
    source = tmp_path / "source.png"
    jpeg = tmp_path / "send.jpg"
    source.write_bytes((fixtures / "synthetic.png").read_bytes())
    jpeg.write_bytes((fixtures / "synthetic.jpg").read_bytes())
    return await load_prepared_photo(
        {
            "source_path": str(source),
            "jpeg_path": str(jpeg),
            "width": 1001,
            "height": 777,
            "scale": 720 / 777,
            "offset_x": (1280 - round(1001 * 720 / 777)) // 2,
            "offset_y": 0,
            "preparation_ms": 1.0,
        },
        directory=tmp_path,
    )


class ControlledGoogle:
    def __init__(self, chunks: list[dict | str], *, status: int = 200, metadata_status: int = 200):
        self.chunks = chunks
        self.status = status
        self.metadata_status = metadata_status
        self.metadata_body = None
        self.requests = []
        self.connections = 0
        self.gate = asyncio.Event()
        self.headers_gate = asyncio.Event()
        self.headers_gate.set()
        self.pause_after = None
        self.post_received = asyncio.Event()
        self.disconnected = asyncio.Event()
        self.writers = set()
        self.handlers = set()

    async def __aenter__(self):
        self.server = await asyncio.start_server(self.handle, "127.0.0.1", 0)
        port = self.server.sockets[0].getsockname()[1]
        self.base_url = f"http://127.0.0.1:{port}/v1beta/models"
        return self

    async def __aexit__(self, *exc):
        self.gate.set()
        self.headers_gate.set()
        self.server.close()
        await self.server.wait_closed()
        for writer in list(self.writers):
            writer.close()
        if self.handlers:
            await asyncio.gather(*list(self.handlers), return_exceptions=True)

    async def handle(self, reader, writer):
        task = asyncio.current_task()
        self.handlers.add(task)
        self.writers.add(writer)
        self.connections += 1
        try:
            while True:
                first = await reader.readline()
                if not first:
                    return
                method, target, _ = first.decode("ascii").strip().split(" ")
                headers = {}
                while (line := await reader.readline()) != b"\r\n":
                    if not line:
                        return
                    key, value = line.decode("ascii").split(":", 1)
                    headers[key.lower()] = value.strip()
                body = await reader.readexactly(int(headers.get("content-length", 0)))
                self.requests.append(
                    (method, target, json.loads(body) if body else None, headers, time.monotonic())
                )
                if method == "GET":
                    model = target.rsplit("/", 1)[1]
                    data = json.dumps(self.metadata_body or {"name": f"models/{model}"}).encode()
                    writer.write(
                        f"HTTP/1.1 {self.metadata_status} Metadata\r\nContent-Type: application/json\r\nContent-Length: {len(data)}\r\n\r\n".encode()
                        + data
                    )
                    await writer.drain()
                    continue
                self.post_received.set()
                await self.headers_gate.wait()
                writer.write(
                    f"HTTP/1.1 {self.status} Stream\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n".encode()
                )
                await writer.drain()
                for index, chunk in enumerate(self.chunks):
                    if index == self.pause_after:
                        await self.gate.wait()
                    data = (
                        "data: " + (chunk if isinstance(chunk, str) else json.dumps(chunk)) + "\n\n"
                    ).encode()
                    writer.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
                    await writer.drain()
                writer.write(b"0\r\n\r\n")
                await writer.drain()
        except ConnectionError, asyncio.IncompleteReadError:
            pass
        finally:
            self.writers.discard(writer)
            writer.close()
            with contextlib.suppress(ConnectionError):
                await writer.wait_closed()
            self.handlers.discard(task)
            self.disconnected.set()


def _google_chunk(text: str | None = None, *, finish: str | None = None, thought: bool = False):
    candidate = {"content": {"parts": []}}
    if text is not None:
        candidate["content"]["parts"].append({"text": text, "thought": thought})
    if finish is not None:
        candidate["finishReason"] = finish
    return {"candidates": [candidate], "modelVersion": "gemini-3.1-flash-lite"}


async def test_luna_dedicated_preopen_send_timing_partial_blocks_and_completion(tmp_path: Path):
    photo = await synthetic_photo(tmp_path)
    requests = []
    connections = []
    release = asyncio.Event()
    disconnected = asyncio.Event()
    block = '{"bbox":[200,100,600,300],"source":"Hello","translation":"안녕"}'

    async def handle(socket):
        connections.append(socket.request.headers.get("Authorization"))
        try:
            requests.append((json.loads(await socket.recv()), time.monotonic()))
            await socket.send(
                json.dumps({"type": "response.output_text.delta", "delta": '{"blocks":[' + block})
            )
            await release.wait()
            await socket.send(json.dumps({"type": "response.output_text.delta", "delta": "]}"}))
            await socket.send(
                json.dumps(
                    {
                        "type": "response.completed",
                        "response": {
                            "status": "completed",
                            "model": "gpt-6-luna",
                            "usage": {"input_tokens": 10, "output_tokens": 20},
                        },
                    }
                )
            )
            await socket.wait_closed()
        finally:
            disconnected.set()

    session = SyntheticSession()
    async with serve(handle, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        connection = LunaPhotoConnection(session, endpoint=f"ws://127.0.0.1:{port}")
        await connection.preopen()
        await asyncio.sleep(0.03)
        assert connections == ["Bearer synthetic-session-token"]
        assert requests == []
        assert session.calls == 1
        assert connection.connect_ms > 0
        pull_time = time.monotonic()
        stream = connection.stream(photo, "ko")
        sent = await anext(stream)
        assert isinstance(sent, PhotoRequestSent)
        delta = await anext(stream)
        assert isinstance(delta, PhotoTextDelta)
        assert pull_time <= sent.monotonic_time <= requests[0][1]
        events = PhotoBlockParser().feed(delta.text)
        assert any(event.kind == "block_complete" for event in events)
        request = requests[0][0]
        assert request["model"] == "gpt-6-luna"
        assert request["reasoning"] == {"effort": "none"}
        assert request["store"] is False
        assert "temperature" not in request
        assert request["text"]["format"]["strict"] is True
        assert (
            request["input"][0]["content"][0]["image_url"]
            == f"data:image/jpeg;base64,{photo.jpeg_base64}"
        )
        assert str(photo.source_path) not in json.dumps(request)
        assert not release.is_set()
        release.set()
        remaining = [event async for event in stream]
        assert isinstance(remaining[-1], PhotoStreamCompleted)
        assert remaining[-1].usage == {"input_tokens": 10, "output_tokens": 20}
        await asyncio.wait_for(disconnected.wait(), 1)


async def test_google_metadata_only_preopen_and_selected_body_thought_filtering_partial_stream(
    tmp_path: Path,
):
    photo = await synthetic_photo(tmp_path)
    block = '{"box_2d":[100,200,400,600],"source":"Hello","translation":"안녕"}'
    chunks = [
        _google_chunk("private thought", thought=True),
        _google_chunk('{"blocks":[' + block),
        _google_chunk("]}", finish="STOP"),
        {"usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 20}},
    ]
    async with ControlledGoogle(chunks) as server:
        server.pause_after = 2
        connection = GooglePhotoConnection(
            "synthetic-key", model="gemini-3.1-flash-lite", base_url=server.base_url
        )
        await connection.preopen()
        assert [(method, body) for method, _, body, _, _ in server.requests] == [("GET", None)]
        assert connection.connect_ms > 0
        pull_time = time.monotonic()
        stream = connection.stream(photo, "ko")
        sent = await anext(stream)
        assert isinstance(sent, PhotoRequestSent)
        delta = await anext(stream)
        assert delta == PhotoTextDelta('{"blocks":[' + block)
        events = PhotoBlockParser(google=True).feed(delta.text)
        assert any(event.kind == "block_complete" for event in events)
        assert server.connections == 1
        assert [request[0] for request in server.requests] == ["GET", "POST"]
        method, target, body, headers, received = server.requests[-1]
        assert pull_time <= sent.monotonic_time <= received
        assert target.endswith("/gemini-3.1-flash-lite:streamGenerateContent?alt=sse")
        assert headers["x-goog-api-key"] == "synthetic-key"
        assert body["contents"][0]["parts"][0] == {
            "inline_data": {"mime_type": "image/jpeg", "data": photo.jpeg_base64}
        }
        config = body["generationConfig"]
        assert config["maxOutputTokens"] == 8192
        assert config["thinkingConfig"] == {"thinkingLevel": "MINIMAL"}
        assert config["mediaResolution"] == "MEDIA_RESOLUTION_HIGH"
        assert not {"temperature", "topP", "topK"}.intersection(config)
        schema = config["responseJsonSchema"]["properties"]["blocks"]["items"]
        assert schema["required"] == ["box_2d", "source", "translation"]
        assert schema["properties"]["box_2d"]["items"] == {"type": "integer"}
        assert "normalized to 0-1000" in body["systemInstruction"]["parts"][0]["text"]
        assert not server.gate.is_set()
        server.gate.set()
        remaining = [event async for event in stream]
        assert remaining[-1] == PhotoStreamCompleted(
            {"promptTokenCount": 10, "candidatesTokenCount": 20}, "gemini-3.1-flash-lite"
        )
        await asyncio.wait_for(server.disconnected.wait(), 1)


async def test_google_send_start_is_observable_before_response_headers(tmp_path: Path):
    photo = await synthetic_photo(tmp_path)
    async with ControlledGoogle([_google_chunk('{"blocks":[]}', finish="STOP")]) as server:
        server.headers_gate.clear()
        connection = GooglePhotoConnection("synthetic-key", base_url=server.base_url)
        await connection.preopen()
        stream = connection.stream(photo, "ko")
        sent = await asyncio.wait_for(anext(stream), 1)
        assert isinstance(sent, PhotoRequestSent)
        await asyncio.wait_for(server.post_received.wait(), 1)
        assert not server.headers_gate.is_set()
        server.headers_gate.set()
        result = [event async for event in stream]
        assert isinstance(result[-1], PhotoStreamCompleted)


@pytest.mark.parametrize(
    ("chunks", "status", "code", "retryable", "quota"),
    [
        ([_google_chunk('{"blocks":[]}')], 200, "photo_stream_incomplete", True, False),
        (
            [_google_chunk("partial", finish="MAX_TOKENS")],
            200,
            "photo_stream_incomplete",
            True,
            False,
        ),
        ([_google_chunk("unsafe", finish="SAFETY")], 200, "photo_content_rejected", False, False),
        (
            [{"promptFeedback": {"blockReason": "SAFETY"}}],
            200,
            "photo_content_rejected",
            False,
            False,
        ),
        (
            [{"error": {"code": 429, "message": "secret-image-text-private-path"}}],
            200,
            "photo_quota_exhausted",
            True,
            True,
        ),
        ([], 429, "photo_quota_exhausted", True, True),
        ([], 401, "photo_authentication_failed", False, False),
        ([], 400, "photo_request_rejected", False, False),
        ([], 503, "photo_provider_unavailable", True, False),
    ],
)
async def test_google_failures_never_create_completion_and_are_sanitized(
    tmp_path, caplog, chunks, status, code, retryable, quota
):
    photo = await synthetic_photo(tmp_path)
    async with ControlledGoogle(chunks, status=status) as server:
        connection = GooglePhotoConnection("synthetic-key", base_url=server.base_url)
        events = []
        with pytest.raises(PhotoProviderError) as failure:
            async for event in connection.stream(photo, "ko"):
                events.append(event)
        assert isinstance(events[0], PhotoRequestSent)
        assert not any(isinstance(event, PhotoStreamCompleted) for event in events)
        assert (failure.value.code, failure.value.retryable, failure.value.quota) == (
            code,
            retryable,
            quota,
        )
        assert str(failure.value) == code
        assert "synthetic-key" not in caplog.text
        assert "secret-image-text-private-path" not in caplog.text
        assert str(photo.source_path) not in caplog.text


async def test_preopen_rejection_consumes_no_send_or_photo_payload(tmp_path):
    photo = await synthetic_photo(tmp_path)
    async with ControlledGoogle([], metadata_status=403) as server:
        connection = GooglePhotoConnection("synthetic-key", base_url=server.base_url)
        events = []
        with pytest.raises(PhotoProviderError, match="photo_authentication_failed"):
            async for event in connection.stream(photo, "ko"):
                events.append(event)
        assert events == []
        assert [(request[0], request[2]) for request in server.requests] == [("GET", None)]


@pytest.mark.parametrize(
    ("terminal", "code", "retryable"),
    [
        (
            {
                "type": "error",
                "error": {
                    "code": "subscription_sharing_usage_limit_exceeded",
                    "message": "secret-image-text-private-path",
                },
            },
            "photo_usage_limit",
            False,
        ),
        ({"type": "response.incomplete", "response": {}}, "photo_stream_incomplete", True),
        (
            {"type": "response.failed", "response": {"error": {"code": "invalid_access_token"}}},
            "photo_authentication_failed",
            False,
        ),
        ({"type": "response.refusal.done"}, "photo_content_rejected", False),
    ],
)
async def test_luna_terminal_errors_are_safe_and_not_success(
    tmp_path, caplog, terminal, code, retryable
):
    photo = await synthetic_photo(tmp_path)
    session = SyntheticSession()

    async def handle(socket):
        await socket.recv()
        await socket.send(
            json.dumps({"type": "response.output_text.delta", "delta": '{"blocks":['})
        )
        await socket.send(json.dumps(terminal))
        await socket.wait_closed()

    async with serve(handle, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        connection = LunaPhotoConnection(session, endpoint=f"ws://127.0.0.1:{port}")
        events = []
        with pytest.raises(PhotoProviderError) as failure:
            async for event in connection.stream(photo, "ko"):
                events.append(event)
        assert (failure.value.code, failure.value.retryable) == (code, retryable)
        assert [type(event) for event in events] == [PhotoRequestSent, PhotoTextDelta]
        assert session.invalidated == (1 if code == "photo_authentication_failed" else 0)
        assert "secret-image-text-private-path" not in caplog.text
        assert "synthetic-session-token" not in caplog.text


async def test_luna_cancel_drains_transport_and_does_not_leave_consumer_hanging(tmp_path):
    photo = await synthetic_photo(tmp_path)
    disconnected = asyncio.Event()

    async def handle(socket):
        await socket.recv()
        await socket.wait_closed()
        disconnected.set()

    async with serve(handle, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        connection = LunaPhotoConnection(SyntheticSession(), endpoint=f"ws://127.0.0.1:{port}")
        stream = connection.stream(photo, "ko")
        assert isinstance(await anext(stream), PhotoRequestSent)
        waiting = asyncio.create_task(anext(stream))
        await asyncio.sleep(0)
        waiting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiting
        await asyncio.wait_for(disconnected.wait(), 1)
        assert not any(
            task.get_name() in {"photo-provider-stream", "photo-provider-close"}
            for task in asyncio.all_tasks()
        )


async def test_google_close_cancels_response_wait_and_drains_socket(tmp_path):
    photo = await synthetic_photo(tmp_path)
    async with ControlledGoogle([]) as server:
        server.headers_gate.clear()
        connection = GooglePhotoConnection("synthetic-key", base_url=server.base_url)
        stream = connection.stream(photo, "ko")
        assert isinstance(await anext(stream), PhotoRequestSent)
        await asyncio.wait_for(server.post_received.wait(), 1)
        await connection.close()
        with pytest.raises(StopAsyncIteration):
            await asyncio.wait_for(anext(stream), 1)
        assert not any(
            task.get_name() in {"photo-provider-stream", "photo-provider-close"}
            for task in asyncio.all_tasks()
        )


async def test_early_consumer_close_drains_repeatedly_cancelled_cleanup(tmp_path):
    photo = await synthetic_photo(tmp_path)
    cleanup_started = asyncio.Event()
    release_cleanup = asyncio.Event()
    disconnected = asyncio.Event()

    async def handle(socket):
        await socket.recv()
        await socket.wait_closed()
        disconnected.set()

    class SlowCloseLuna(LunaPhotoConnection):
        async def _close_transport(self):
            cleanup_started.set()
            await release_cleanup.wait()
            await super()._close_transport()

    async with serve(handle, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        connection = SlowCloseLuna(SyntheticSession(), endpoint=f"ws://127.0.0.1:{port}")
        stream = connection.stream(photo, "ko")
        assert isinstance(await anext(stream), PhotoRequestSent)
        closing = asyncio.create_task(stream.aclose())
        await asyncio.wait_for(cleanup_started.wait(), 1)
        closing.cancel()
        await asyncio.sleep(0)
        closing.cancel()
        await asyncio.sleep(0)
        assert not closing.done()
        release_cleanup.set()
        await asyncio.wait_for(closing, 1)
        await asyncio.wait_for(disconnected.wait(), 1)
        assert not any(
            task.get_name() in {"photo-provider-stream", "photo-provider-close"}
            for task in asyncio.all_tasks()
        )


async def test_google_invalid_key_reason_is_terminal_authentication_not_generic_bad_request(
    tmp_path, caplog
):
    photo = await synthetic_photo(tmp_path)
    async with ControlledGoogle([], metadata_status=400) as server:
        server.metadata_body = {
            "error": {
                "code": 400,
                "message": "private-secret-value",
                "details": [{"reason": "API_KEY_INVALID"}],
            }
        }
        connection = GooglePhotoConnection("synthetic-key", base_url=server.base_url)
        events = []
        with pytest.raises(PhotoProviderError) as failure:
            async for event in connection.stream(photo, "ko"):
                events.append(event)
        assert events == []
        assert failure.value.code == "photo_authentication_failed"
        assert failure.value.retryable is False
        assert "private-secret-value" not in caplog.text


async def test_luna_primary_failure_survives_sanitized_cleanup_failure(tmp_path, caplog):
    photo = await synthetic_photo(tmp_path)

    async def handle(socket):
        await socket.recv()
        await socket.send(
            json.dumps(
                {
                    "type": "error",
                    "error": {"code": "subscription_sharing_usage_limit_exceeded"},
                }
            )
        )
        await socket.wait_closed()

    class FailingCloseLuna(LunaPhotoConnection):
        async def _close_transport(self):
            await super()._close_transport()
            raise OSError("private-close-secret")

    async with serve(handle, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        connection = FailingCloseLuna(SyntheticSession(), endpoint=f"ws://127.0.0.1:{port}")
        with pytest.raises(PhotoProviderError) as failure:
            async for _ in connection.stream(photo, "ko"):
                pass
        assert failure.value.code == "photo_usage_limit"
        assert "photo_transport_failed" in caplog.text
        assert "private-close-secret" not in caplog.text
        assert not any(
            task.get_name() in {"photo-provider-stream", "photo-provider-close"}
            for task in asyncio.all_tasks()
        )


async def test_api_baseline_streams_partial_blocks_without_setup_requests_or_paid_replay(tmp_path):
    photo = await synthetic_photo(tmp_path)
    events = [
        {
            "type": "response.output_text.delta",
            "delta": '{"blocks":[{"bbox":[0,0,100,100],"source":"one","translation":"하나"}',
        },
        {"type": "response.output_text.delta", "delta": "]}"},
        {
            "type": "response.completed",
            "response": {
                "status": "completed",
                "model": "gpt-6-luna",
                "usage": {"output_tokens": 30},
            },
        },
    ]
    async with ControlledGoogle(events) as server:
        server.pause_after = 1
        connection = OpenAIPhotoConnection("synthetic-api-key")
        connection.endpoint = server.base_url.removesuffix("/v1beta/models") + "/v1/responses"
        await connection.preopen()
        assert server.requests == []
        stream = connection.stream(photo, "ko")
        assert isinstance(await anext(stream), PhotoRequestSent)
        first = await anext(stream)
        assert isinstance(first, PhotoTextDelta)
        parsed = PhotoBlockParser().feed(first.text)
        assert any(event.kind == "block_complete" for event in parsed)
        assert not server.gate.is_set()
        server.gate.set()
        remaining = [event async for event in stream]
        assert remaining[-1] == PhotoStreamCompleted({"output_tokens": 30}, "gpt-6-luna")
        assert len(server.requests) == 1
        assert server.requests[0][2]["model"] == "gpt-6-luna"
        assert server.requests[0][3]["authorization"] == "Bearer synthetic-api-key"


@pytest.mark.parametrize(
    "terminal",
    [
        {"type": "response.incomplete", "response": {"status": "incomplete"}},
        {"type": "response.refusal.delta", "delta": "private refusal"},
        {"type": "response.output_text.delta", "delta": '{"blocks":['},
    ],
)
async def test_api_uncertain_or_unusable_output_never_authorizes_automatic_retry(
    tmp_path, terminal
):
    photo = await synthetic_photo(tmp_path)
    async with ControlledGoogle([terminal]) as server:
        connection = OpenAIPhotoConnection("synthetic-api-key")
        connection.endpoint = server.base_url.removesuffix("/v1beta/models") + "/v1/responses"
        events = []
        with pytest.raises(PhotoProviderError) as failure:
            async for event in connection.stream(photo, "ko"):
                events.append(event)
        assert failure.value.retryable is False
        assert not any(isinstance(event, PhotoStreamCompleted) for event in events)
        assert len(server.requests) == 1


async def test_relay_uses_dedicated_photo_contract_without_per_photo_coordination(tmp_path):
    photo = await synthetic_photo(tmp_path)
    tokens = []

    async def token():
        tokens.append(True)
        return "synthetic-puri-token"

    events = [
        {"type": "text_delta", "text": '{"blocks":[]}'},
        {
            "type": "completed",
            "response": {"status": "completed", "model": "gpt-6-luna", "usage": {}},
        },
    ]
    async with ControlledGoogle(events) as server:
        endpoint = server.base_url.removesuffix("/v1beta/models") + "/v1/photo"
        connection = RelayPhotoConnection(endpoint, token, allow_loopback_http=True)
        await connection.preopen()
        assert server.requests == []
        result = [event async for event in connection.stream(photo, "ko")]
        assert isinstance(result[-1], PhotoStreamCompleted)
        assert len(server.requests) == 1
        sent = server.requests[0][2]
        assert set(sent) == {
            "photo_protocol",
            "request_id",
            "image_base64",
            "image_size",
            "target_language",
        }
        assert sent["photo_protocol"] == 2
        assert sent["image_size"] == [1280, 720]
        assert sent["image_base64"] == photo.jpeg_base64
        assert sent["target_language"] == "ko"
        assert server.requests[0][3]["authorization"] == "Bearer synthetic-puri-token"


async def test_api_completion_drains_sse_terminator_and_reuses_the_connection(tmp_path):
    photo = await synthetic_photo(tmp_path)
    events = [
        {"type": "response.output_text.delta", "delta": '{"blocks":[]}'},
        {"type": "response.completed", "response": {"status": "completed", "model": "gpt-6-luna"}},
        "[DONE]",
    ]
    async with (
        ControlledGoogle(events) as server,
        httpx.AsyncClient(
            trust_env=False,
            limits=httpx.Limits(max_connections=1),
            timeout=2,
        ) as client,
    ):
        server.pause_after = 2
        endpoint = server.base_url.removesuffix("/v1beta/models") + "/v1/responses"
        connection = OpenAIPhotoConnection("synthetic-api-key", client=client)
        connection.endpoint = endpoint
        async with contextlib.aclosing(connection.stream(photo, "ko")) as stream:
            assert isinstance(await anext(stream), PhotoRequestSent)
            assert isinstance(await anext(stream), PhotoTextDelta)
            assert isinstance(await anext(stream), PhotoStreamCompleted)
            assert not server.gate.is_set()
            server.gate.set()
            assert [event async for event in stream] == []
        following = OpenAIPhotoConnection("synthetic-api-key", client=client)
        following.endpoint = endpoint
        result = [event async for event in following.stream(photo, "ko")]
        assert isinstance(result[-1], PhotoStreamCompleted)
        assert len(server.requests) == 2
        assert server.connections == 1


async def test_api_consumer_failure_releases_single_connection_pool_for_next_photo(tmp_path):
    photo = await synthetic_photo(tmp_path)
    events = [
        {"type": "response.output_text.delta", "delta": "invalid photo JSON"} for _ in range(80)
    ] + [
        {"type": "response.completed", "response": {"status": "completed", "model": "gpt-6-luna"}},
    ]
    async with (
        ControlledGoogle(events) as server,
        httpx.AsyncClient(
            trust_env=False,
            limits=httpx.Limits(max_connections=1),
            timeout=2,
        ) as client,
    ):
        endpoint = server.base_url.removesuffix("/v1beta/models") + "/v1/responses"
        connection = OpenAIPhotoConnection("synthetic-api-key", client=client)
        connection.endpoint = endpoint
        async with contextlib.aclosing(connection.stream(photo, "ko")) as stream:
            assert isinstance(await anext(stream), PhotoRequestSent)
            assert isinstance(await anext(stream), PhotoTextDelta)
        server.chunks = [
            {"type": "response.output_text.delta", "delta": '{"blocks":[]}'},
            {
                "type": "response.completed",
                "response": {"status": "completed", "model": "gpt-6-luna"},
            },
        ]
        following = OpenAIPhotoConnection("synthetic-api-key", client=client)
        following.endpoint = endpoint
        result = [event async for event in following.stream(photo, "ko")]
        assert isinstance(result[-1], PhotoStreamCompleted)
        assert len(server.requests) == 2


async def test_portrait_preparation_maps_boxes_and_prompts_in_its_upright_canvas(tmp_path):
    from puripuly_heart.providers.photo import _luna_payload

    source = tmp_path / "source.png"
    jpeg = tmp_path / "send.jpg"
    fixtures = Path(__file__).parents[1] / "core" / "photo" / "fixtures"
    source.write_bytes((fixtures / "synthetic.png").read_bytes())
    jpeg.write_bytes((fixtures / "synthetic.jpg").read_bytes())
    photo = await load_prepared_photo(
        {
            "source_path": str(source),
            "jpeg_path": str(jpeg),
            "width": 2160,
            "height": 3840,
            "scale": 1 / 3,
            "offset_x": 0,
            "offset_y": 0,
            "preparation_ms": 1.0,
        },
        directory=tmp_path,
    )
    assert photo.send_size == (720, 1280)
    assert photo.map_box((0, 640, 720, 1280), google=False).bbox == (0.0, 1920.0, 2160.0, 3840.0)
    assert photo.map_box((500, 0, 1000, 1000), google=True).bbox == (0.0, 1920.0, 2160.0, 3840.0)
    text = _luna_payload(photo, "ko")["input"][0]["content"][1]["text"]
    assert text.endswith("The image is 720x1280 pixels.")
