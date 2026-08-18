from __future__ import annotations

import json
from uuid import uuid4

import httpx
import pytest

from puripuly_heart.core.local_translation.runtime import (
    ManagedGemmaMetrics,
    ManagedGemmaResponse,
)
from puripuly_heart.providers.llm.managed_gemma import (
    HttpxManagedGemmaTransport,
    ManagedGemmaLLMProvider,
)


class FakeRuntime:
    def __init__(self) -> None:
        self.calls = []
        self.released = False

    async def translate(self, **kwargs):
        self.calls.append(kwargs)
        return ManagedGemmaResponse(
            text="hello",
            metrics=ManagedGemmaMetrics(1, 1, 1, 1.0, 1.0, 1.0),
        )

    async def release(self) -> None:
        self.released = True


@pytest.mark.asyncio
async def test_provider_keeps_llama_details_behind_llm_boundary() -> None:
    runtime = FakeRuntime()
    provider = ManagedGemmaLLMProvider(runtime=runtime, backend="gpu", vulkan_device="Vulkan3")
    utterance_id = uuid4()

    result = await provider.translate(
        utterance_id=utterance_id,
        text="안녕",
        system_prompt="translate",
        source_language="ko",
        target_language="en",
        context="prior",
    )

    assert result.utterance_id == utterance_id
    assert result.text == "hello"
    assert runtime.calls == [
        {
            "backend": "gpu",
            "source_language": "ko",
            "target_language": "en",
            "system_prompt": "translate",
            "user_message": "<context>\nprior\n</context>\n\n<input>\n안녕\n</input>",
            "vulkan_device": "Vulkan3",
        }
    ]
    await provider.close()
    assert runtime.released
    with pytest.raises(RuntimeError, match="provider is closed"):
        await provider.translate(
            utterance_id=uuid4(),
            text="late",
            system_prompt="translate",
            source_language="ko",
            target_language="en",
        )
    assert len(runtime.calls) == 1


@pytest.mark.asyncio
async def test_http_transport_prefills_cache_and_extracts_llama_metrics() -> None:
    bodies = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        bodies.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": " hello "}}],
                "usage": {
                    "prompt_tokens": 30,
                    "prompt_tokens_details": {"cached_tokens": 25},
                    "completion_tokens": 5,
                },
                "timings": {
                    "prompt_ms": 7.5,
                    "predicted_ms": 125.0,
                    "predicted_per_second": 40.0,
                    "draft_n": 8,
                    "draft_n_accepted": 5,
                },
            },
        )

    transport = HttpxManagedGemmaTransport("http://127.0.0.1:38191")
    transport._client = httpx.AsyncClient(
        base_url=transport.base_url,
        transport=httpx.MockTransport(handler),
    )
    await transport.wait_until_ready(timeout_s=0.1)
    await transport.prepare_prefix(system_prompt="system")
    response = await transport.translate(system_prompt="system", user_message="input")
    await transport.close()

    assert bodies[0]["cache_prompt"] is True
    assert bodies[0]["messages"] == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": " "},
    ]
    assert bodies[0]["max_tokens"] == 1
    assert bodies[1]["messages"][0] == bodies[0]["messages"][0]
    assert "max_tokens" not in bodies[1]
    assert response.text == "hello"
    assert response.metrics == ManagedGemmaMetrics(
        prompt_tokens=30,
        cached_prompt_tokens=25,
        completion_tokens=5,
        prompt_ms=7.5,
        generation_ms=125.0,
        generation_tps=40.0,
        drafted_tokens=8,
        accepted_tokens=5,
    )


@pytest.mark.asyncio
async def test_http_transport_retains_client_when_close_fails() -> None:
    class FailOnceClient:
        def __init__(self) -> None:
            self.close_calls = 0

        async def aclose(self) -> None:
            self.close_calls += 1
            if self.close_calls == 1:
                raise RuntimeError("close failed")

    transport = HttpxManagedGemmaTransport("http://127.0.0.1:38191")
    client = FailOnceClient()
    transport._client = client

    with pytest.raises(RuntimeError, match="close failed"):
        await transport.close()
    assert transport._client is client

    await transport.close()
    assert transport._client is None
