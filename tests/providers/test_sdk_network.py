from __future__ import annotations

import asyncio
import json
from urllib.parse import parse_qs, urlsplit

import pytest
from websockets.asyncio.server import serve

from puripuly_heart.core import network_clients
from puripuly_heart.providers.stt.genai_network import configure_live_network
from puripuly_heart.providers.stt.sdk_network import ExternalScribeRealtime, deepgram_listen_connect
from tests.core.test_external_network import certificates as certificates
from tests.core.test_external_network import isolated_network_environment as isolated_network_environment
from tests.core.test_external_network import recording_proxy


@pytest.mark.parametrize("https_proxy", [False, True])
async def test_scribe_sdk_verified_proxy_audio_event_flow(certificates, monkeypatch, https_proxy):
    from elevenlabs.realtime import AudioFormat, CommitStrategy, RealtimeEvents

    root, _, contexts = certificates
    monkeypatch.setenv("SSL_CERT_FILE", str(root))
    requests = []
    async def handler(ws):
        requests.append((ws.request.path, ws.request.headers, json.loads(await ws.recv())))
        await ws.send(json.dumps({"message_type": "partial_transcript", "text": "recognized"}))
        await ws.wait_closed()
    async with recording_proxy(contexts["valid"] if https_proxy else None) as (proxy, recorded):
        monkeypatch.setenv("ALL_PROXY", proxy)
        async with serve(handler, "127.0.0.1", 0, ssl=contexts["valid"]) as server:
            scribe = ExternalScribeRealtime(api_key="test-key", base_url=f"wss://localhost:{server.sockets[0].getsockname()[1]}")
            connection = await scribe.connect({
                "model_id": "scribe_v2_realtime", "audio_format": AudioFormat.PCM_16000,
                "sample_rate": 16000, "commit_strategy": CommitStrategy.MANUAL,
                "language_code": "en", "keyterms": ["PuriPuly"],
            })
            result = asyncio.get_running_loop().create_future()
            connection.on(RealtimeEvents.PARTIAL_TRANSCRIPT, lambda event: result.set_result(event))
            try:
                await connection.send({"audio_base_64": "AAA="})
                assert (await asyncio.wait_for(result, 5))["text"] == "recognized"
            finally:
                await connection.close()
    path, headers, audio = requests[0]
    query = parse_qs(urlsplit(path).query)
    assert query["model_id"] == ["scribe_v2_realtime"]
    assert query["language_code"] == ["en"]
    assert query["keyterms"] == ["PuriPuly"]
    assert headers["xi-api-key"] == "test-key"
    assert audio["message_type"] == "input_audio_chunk"
    assert audio["audio_base_64"] == "AAA="
    assert len(recorded) == 1


@pytest.mark.parametrize("https_proxy", [False, True])
async def test_deepgram_sdk_verified_proxy_message_flow(certificates, monkeypatch, https_proxy):
    from deepgram import DeepgramClient
    from deepgram.environment import DeepgramClientEnvironment

    root, _, contexts = certificates
    monkeypatch.setenv("SSL_CERT_FILE", str(root))
    requests = []
    async def handler(ws):
        requests.append((ws.request.path, ws.request.headers, await ws.recv()))
        await ws.send(json.dumps({
            "type": "Results", "channel_index": [0, 1], "duration": 1.0, "start": 0.0,
            "is_final": True, "speech_final": True,
            "channel": {"alternatives": [{"transcript": "recognized", "confidence": 1.0, "words": []}]},
            "metadata": {"request_id": "00000000-0000-0000-0000-000000000001", "model_info": {}, "model_uuid": "00000000-0000-0000-0000-000000000001"},
        }))
        await ws.wait_closed()
    async with recording_proxy(contexts["valid"] if https_proxy else None) as (proxy, recorded):
        monkeypatch.setenv("ALL_PROXY", proxy)
        async with serve(handler, "127.0.0.1", 0, ssl=contexts["valid"]) as server:
            url = f"wss://localhost:{server.sockets[0].getsockname()[1]}"
            def operation():
                with network_clients.external_client() as http_client:
                    client = DeepgramClient(
                        api_key="test-key", httpx_client=http_client,
                        environment=DeepgramClientEnvironment(base=url, production=url, agent=url),
                    )
                    with deepgram_listen_connect(
                        client, model="nova-3", language="en", encoding="linear16", sample_rate=16000,
                        channels=1, interim_results=False, punctuate=True, vad_events=False,
                        endpointing=False, keyterm=["first", "second"],
                    ) as connection:
                        connection.send_media(b"\0\0")
                        return connection.recv().channel["alternatives"][0]["transcript"]
            assert await asyncio.wait_for(asyncio.to_thread(operation), 10) == "recognized"
    path, headers, audio = requests[0]
    query = parse_qs(urlsplit(path).query)
    assert query["keyterm"] == ["first", "second"]
    assert query["interim_results"] == ["false"]
    assert query["endpointing"] == ["false"]
    assert query["sample_rate"] == ["16000"]
    assert headers["Authorization"] == "Token test-key"
    assert headers["x-deepgram-session-id"]
    assert audio == b"\0\0"
    assert len(recorded) == 1


@pytest.mark.parametrize("https_proxy", [False, True])
async def test_genai_live_verified_proxy_setup_audio_flow(certificates, monkeypatch, https_proxy):
    from google import genai
    from google.genai import types

    root, _, contexts = certificates
    monkeypatch.setenv("SSL_CERT_FILE", str(root))
    messages = []
    async def handler(ws):
        messages.append(json.loads(await ws.recv()))
        await ws.send(json.dumps({"setupComplete": {}}))
        messages.append(json.loads(await ws.recv()))
        await ws.send(json.dumps({"serverContent": {"inputTranscription": {"text": "recognized"}, "turnComplete": True}}))
        await ws.wait_closed()
    async with recording_proxy(contexts["valid"] if https_proxy else None) as (proxy, recorded):
        monkeypatch.setenv("ALL_PROXY", proxy)
        async with serve(handler, "127.0.0.1", 0, ssl=contexts["valid"]) as server:
            url = f"https://localhost:{server.sockets[0].getsockname()[1]}"
            options = types.HttpOptions(**network_clients.genai_http_options(), base_url=url)
            client = genai.Client(api_key="test-key", http_options=options)
            configure_live_network(client, options)
            try:
                async with client.aio.live.connect(model="gemini-test", config={"response_modalities": ["TEXT"]}) as session:
                    await session.send_realtime_input(audio=types.Blob(data=b"\0\0", mime_type="audio/pcm;rate=16000"))
                    async for event in session.receive():
                        if event.server_content and event.server_content.input_transcription:
                            assert event.server_content.input_transcription.text == "recognized"
                            break
            finally:
                await client.aio.aclose()
                client.close()
    assert "setup" in messages[0]
    assert messages[1]["realtime_input"]["audio"]["data"] == "AAA="
    assert len(recorded) == 1
