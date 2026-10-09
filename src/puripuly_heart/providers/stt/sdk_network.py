from __future__ import annotations

import asyncio
from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import Any

import httpx
from elevenlabs.realtime import CommitStrategy, ScribeRealtime
from elevenlabs.realtime.connection import RealtimeConnection
from websockets.asyncio.client import ClientConnection
from websockets.exceptions import InvalidStatus

from puripuly_heart.core import network_clients


class ExternalScribeRealtime(ScribeRealtime):
    async def _connect_audio(self, options: Any) -> RealtimeConnection:
        audio_format = options.get("audio_format")
        sample_rate = options.get("sample_rate")
        if not audio_format or not sample_rate:
            raise ValueError("audio_format and sample_rate are required for manual audio mode")
        url = self._build_websocket_url(
            model_id=options["model_id"],
            audio_format=audio_format.value,
            commit_strategy=options.get("commit_strategy", CommitStrategy.MANUAL).value,
            **self._shared_url_kwargs(options),
        )
        websocket = await network_clients.external_websocket_connect(
            url, additional_headers=self._connection_headers(options)
        )
        connection = RealtimeConnection(
            websocket=websocket, current_sample_rate=sample_rate, ffmpeg_process=None
        )
        connection._message_task = asyncio.create_task(connection._start_message_handler())
        connection._emit("open")
        return connection


@asynccontextmanager
async def deepgram_listen_connect(
    client: Any, *, _close_deadline: Callable[[], float] | None = None, **options: Any
):
    from deepgram.core.api_error import ApiError
    from deepgram.listen.v1.socket_client import AsyncV1SocketClient

    wrapper = client.listen.v1._client_wrapper
    url = wrapper.get_environment().production + "/v1/listen"
    query = httpx.QueryParams()
    for name, value in options.items():
        if value is None:
            continue
        if name == "keyterm" and isinstance(value, (list, tuple)):
            for term in value:
                query = query.add(name, term)
        else:
            query = query.add(name, value)
    headers = wrapper.get_headers()

    class DeepgramConnection(ClientConnection):
        async def handshake(self, *args: Any, **kwargs: Any) -> None:
            try:
                await super().handshake(*args, **kwargs)
            except InvalidStatus as exc:
                status = exc.response.status_code
                raise ApiError(
                    status_code=status,
                    headers=dict(headers),
                    body=(
                        "Websocket initialized with invalid credentials."
                        if status == 401
                        else "Unexpected error when initializing websocket connection."
                    ),
                ) from exc

        async def close(self, *args: Any, **kwargs: Any) -> None:
            try:
                if not asyncio.current_task().cancelling():
                    if _close_deadline is None:
                        await super().close(*args, **kwargs)
                    else:
                        async with asyncio.timeout_at(_close_deadline()):
                            await super().close(*args, **kwargs)
            except TimeoutError:
                pass
            finally:
                if not self.connection_lost_waiter.done():
                    self.transport.abort()
                await self.wait_closed()
                task = self.keepalive_task
                if task is not None and task is not asyncio.current_task():
                    await asyncio.gather(task, return_exceptions=True)

    async with network_clients.external_websocket_connect(
        url + f"?{query}", additional_headers=headers, create_connection=DeepgramConnection
    ) as protocol:
        yield AsyncV1SocketClient(websocket=protocol)
