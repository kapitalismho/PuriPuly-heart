import asyncio
import json
from dataclasses import replace
from uuid import uuid4

import pytest

from puripuly_heart.app.wiring.wiring_llm_factory import _LazyFactoryLLMProvider
from puripuly_heart.config.overlay_calibration import OverlayCalibration
from puripuly_heart.core import external_network, network_clients
from puripuly_heart.core.clock import FakeClock
from puripuly_heart.core.llm.fallback_racing import FallbackRacingLLMProvider, LLMProviderAttempt
from puripuly_heart.core.llm.provider import SemaphoreLLMProvider
from puripuly_heart.core.orchestrator.configuration import (
    TranslationRuntimeConfig,
    TranslationRuntimeConfigurationOwner,
)
from puripuly_heart.core.overlay.presenter import OverlayPresenter
from puripuly_heart.domain.events import STTFinalEvent, UIEventType
from puripuly_heart.domain.models import Transcript
from puripuly_heart.providers.llm.deepseek import DeepSeekLLMProvider
from puripuly_heart.providers.llm.openai import OpenAILLMProvider
from tests.helpers.fakes import RecordingOscQueue
from tests.helpers.translation_owners import compose_translation_test_harness


@pytest.mark.parametrize("provider_type", [DeepSeekLLMProvider, OpenAILLMProvider])
async def test_translation_and_overlay_resume_after_proxy_change(monkeypatch, provider_type):
    hits = [0, 0]

    async def respond(reader, writer, index):
        try:
            header = await reader.readuntil(b"\r\n\r\n")
            length = next(
                (
                    int(line.split(b":", 1)[1])
                    for line in header.split(b"\r\n")
                    if line.lower().startswith(b"content-length:")
                ),
                0,
            )
            await reader.readexactly(length)
            hits[index] += 1
            body = json.dumps(
                {
                    "choices": [
                        {
                            "message": {"role": "assistant", "content": "译文"},
                            "finish_reason": "stop",
                        }
                    ]
                }
            ).encode()
            writer.write(
                b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nConnection: close\r\nContent-Length: "
                + str(len(body)).encode()
                + b"\r\n\r\n"
                + body
            )
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()

    server_a = await asyncio.start_server(lambda r, w: respond(r, w, 0), "127.0.0.1", 0)
    server_b = await asyncio.start_server(lambda r, w: respond(r, w, 1), "127.0.0.1", 0)
    proxies = [f"http://127.0.0.1:{s.sockets[0].getsockname()[1]}" for s in (server_a, server_b)]
    system = {"http": proxies[0]}
    monkeypatch.setattr(external_network.urllib.request, "getproxies_environment", lambda: {})
    monkeypatch.setattr(external_network, "_windows_proxy_settings", lambda: (dict(system), ""))
    base = provider_type(
        api_key="synthetic",
        base_url="http://translation.invalid/v1",
        model="synthetic",
        timeout=0.5,
    )
    fallback_base = provider_type(
        api_key="synthetic",
        base_url="http://translation.invalid/v1",
        model="synthetic",
        timeout=0.5,
    )
    fallback = _LazyFactoryLLMProvider(factory=lambda: fallback_base)
    racing = FallbackRacingLLMProvider(
        attempts=(
            LLMProviderAttempt(base),
            LLMProviderAttempt(fallback, start_after_ms=10000, start_on_primary_error=True),
        ),
        loser_grace_ms=0,
    )
    llm = SemaphoreLLMProvider(racing, asyncio.Semaphore(2))
    config = TranslationRuntimeConfigurationOwner(
        TranslationRuntimeConfig(
            translation_enabled=True,
            peer_translation_enabled=True,
            source_language="en",
            target_language="zh-CN",
            peer_source_language="en",
            peer_target_language="zh-CN",
        )
    )
    clock = FakeClock(_now=10.0)
    harness = compose_translation_test_harness(
        stt=None,
        llm=llm,
        osc=RecordingOscQueue(),
        translation_runtime_configuration=config,
        clock=clock,
    )
    presenter = OverlayPresenter(calibration=OverlayCalibration(), clock=clock)
    sequence = 0

    async def say(generation):
        nonlocal sequence
        sequence += 1
        clock.advance(10)
        t = Transcript(
            uuid4(),
            f"Fresh source {sequence}.",
            True,
            channel="peer",
            created_at=clock.now(),
            publication_generation=generation,
            source_order=sequence,
        )
        await harness.dispatch_stt_event(STTFinalEvent(utterance_id=t.utterance_id, transcript=t))
        await asyncio.wait_for(harness.translation_turns.wait_for_idle(), 2)
        await asyncio.wait_for(harness.output_runtime.wait_for_peer_output_idle(), 2)
        events = []
        while not harness.ui_events.empty():
            events.append(harness.ui_events.get_nowait())
        return events, t

    try:
        await harness.start()
        await harness.output_runtime.replace_overlay_sink(presenter)
        events, _ = await say(1)
        assert any(e.type is UIEventType.TRANSLATION_DONE for e in events)
        await fallback.translate(
            utterance_id=uuid4(),
            text="Earlier fallback request",
            system_prompt="Translate",
            source_language="en",
            target_language="zh-CN",
        )
        config.transform(
            lambda c: replace(c, translation_enabled=False, peer_translation_enabled=False)
        )
        await presenter.update_translation_enabled(False)
        harness.output_runtime.retire_peer_generation(1)
        await harness.peer_owner.reset_provider_channel("peer")
        system["http"] = proxies[1]
        server_a.close()
        await server_a.wait_closed()
        async with network_clients.external_async_client(timeout=0.5) as fresh:
            assert (
                await fresh.post("http://translation.invalid/v1/chat/completions", json={})
            ).status_code == 200
        config.transform(
            lambda c: replace(c, translation_enabled=True, peer_translation_enabled=True)
        )
        harness.output_runtime.activate_peer_generation(2)
        await presenter.update_translation_enabled(True)
        recovered = []
        for _ in range(3):
            events, transcript = await say(2)
            recovered.append(any(e.type is UIEventType.TRANSLATION_DONE for e in events))
            assert not any(e.type is UIEventType.ERROR for e in events)
            assert any(
                b.primary_text == "译文" and b.secondary_text == transcript.text
                for b in presenter.snapshot().blocks
            )
        assert recovered == [True] * 3
        fallback_result = await fallback.translate(
            utterance_id=uuid4(),
            text="Fallback request after resume",
            system_prompt="Translate",
            source_language="en",
            target_language="zh-CN",
        )
        assert fallback_result.text == "译文"
        assert hits == [2, 5]
        assert config.snapshot().value.translation_enabled and presenter.translation_enabled
    finally:
        await harness.stop()
        await presenter.close()
        for server in (server_a, server_b):
            server.close()
            await server.wait_closed()
