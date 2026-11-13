from __future__ import annotations

import json
import logging
from uuid import uuid4

import httpx
import pytest

from puripuly_heart.core.diagnostics import (
    deserialize_diagnostics,
    exception_diagnostics,
    serialize_diagnostics,
)
from puripuly_heart.core.local_translation.runtime import (
    ManagedGemmaMetrics,
    ManagedGemmaResponse,
)
from puripuly_heart.providers.llm.managed_gemma import (
    HttpxManagedGemmaTransport,
    ManagedGemmaLLMProvider,
)


class FakeRuntime:
    def __init__(self, *, error: Exception | None = None) -> None:
        self.calls = []
        self.released = False
        self.error = error

    async def translate(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return ManagedGemmaResponse(
            text="hello",
            metrics=ManagedGemmaMetrics(1, 1, 1, 1.0, 1.0, 1.0),
        )

    async def release(self) -> None:
        self.released = True


class SpyRuntimeLogging:
    def __init__(self) -> None:
        self.basic_messages: list[tuple[str, int]] = []

    def emit_basic(self, message: str, *, level: int = logging.INFO) -> None:
        self.basic_messages.append((message, level))


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
async def test_provider_logs_basic_translate_failure() -> None:
    runtime = FakeRuntime(error=RuntimeError("llama down"))
    runtime_logging = SpyRuntimeLogging()
    provider = ManagedGemmaLLMProvider(
        runtime=runtime,
        backend="gpu",
        runtime_logging=runtime_logging,
    )

    with pytest.raises(RuntimeError) as exc_info:
        await provider.translate(
            utterance_id=uuid4(),
            text="안녕",
            system_prompt="translate",
            source_language="ko",
            target_language="en",
        )

    assert exc_info.value is runtime.error
    assert runtime_logging.basic_messages == []


@pytest.mark.asyncio
async def test_http_transport_prefills_cache_and_extracts_llama_metrics() -> None:
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        body = json.loads(request.content)
        requests.append((request.url.path, body))
        if request.url.path == "/apply-template":
            return httpx.Response(
                200,
                json={"prompt": f"rendered:{body['messages'][1]['content']}"},
            )
        return httpx.Response(
            200,
            json={
                "content": " hello ",
                "timings": {
                    "prompt_n": 30,
                    "cache_n": 25,
                    "predicted_n": 5,
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
    await transport.prepare_prefix(system_prompt="system", slot_id=0)
    response = await transport.translate(system_prompt="system", user_message="input", slot_id=0)
    await transport.close()

    assert [path for path, _body in requests] == [
        "/apply-template",
        "/completion",
        "/apply-template",
        "/completion",
    ]
    assert requests[0][1]["messages"] == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": " "},
    ]
    assert requests[0][1]["add_generation_prompt"] is True
    assert requests[1][1] == {
        "prompt": "rendered: ",
        "stream": False,
        "temperature": 0.6,
        "cache_prompt": True,
        "id_slot": 0,
        "n_predict": 1,
    }
    assert requests[2][1]["messages"][0] == requests[0][1]["messages"][0]
    assert requests[3][1] == {
        "prompt": "rendered:input",
        "stream": False,
        "temperature": 0.6,
        "cache_prompt": True,
        "id_slot": 0,
    }
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


@pytest.mark.asyncio
async def test_http_transport_restores_and_saves_slot_prefix() -> None:
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append((request.method, str(request.url), request.content))
        if request.url.path == "/slots/0" and "action=restore" in str(request.url):
            return httpx.Response(200, json={"n_restored": 12})
        if request.url.path == "/slots/1" and "action=save" in str(request.url):
            return httpx.Response(200, json={"n_saved": 12})
        return httpx.Response(500)

    transport = HttpxManagedGemmaTransport("http://127.0.0.1:38191")
    transport._client = httpx.AsyncClient(
        base_url=transport.base_url,
        transport=httpx.MockTransport(handler),
    )

    restored = await transport.restore_prefix(filename="abc.cpu.bin", slot_id=0)
    await transport.save_prefix(filename="abc.cpu.bin", slot_id=1)
    await transport.close()

    assert restored is True
    assert [item[1] for item in requests] == [
        "http://127.0.0.1:38191/slots/0?action=restore",
        "http://127.0.0.1:38191/slots/1?action=save",
    ]


@pytest.mark.asyncio
async def test_http_transport_restore_miss_returns_false() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    transport = HttpxManagedGemmaTransport("http://127.0.0.1:38191")
    transport._client = httpx.AsyncClient(
        base_url=transport.base_url,
        transport=httpx.MockTransport(handler),
    )

    restored = await transport.restore_prefix(filename="missing.bin", slot_id=0)
    await transport.close()

    assert restored is False


@pytest.mark.asyncio
async def test_readiness_timeout_retains_last_validated_http_failure():
    transport = HttpxManagedGemmaTransport("http://127.0.0.1:38191", poll_interval_s=0.001)
    transport._client = httpx.AsyncClient(
        base_url=transport.base_url,
        transport=httpx.MockTransport(lambda _request: httpx.Response(503)),
    )
    try:
        with pytest.raises(TimeoutError) as failure:
            await transport.wait_until_ready(timeout_s=0.005)
        assert failure.value.__cause__.failure_diagnostics.status_code == 503
    finally:
        await transport.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["restore_prefix", "save_prefix"])
async def test_prefix_request_failure_preserves_secondary_settlement_failure(operation):
    transport = HttpxManagedGemmaTransport("http://127.0.0.1:38191")
    transport._client = httpx.AsyncClient(
        base_url=transport.base_url,
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                500 if request.url.path == "/slots" else 503,
                json={"message": "PRIVATE_SECRET"},
            )
        ),
    )
    try:
        with pytest.raises(
            RuntimeError if operation == "restore_prefix" else httpx.HTTPStatusError
        ) as failure:
            await getattr(transport, operation)(filename="cache.bin", slot_id=0)
        diagnostics = failure.value.failure_diagnostics
        assert diagnostics.status_code == 503
        assert diagnostics.causes[0].status_code == 500
        assert diagnostics.causes[0].fields["cleanup_confirmed"] is False
        assert "PRIVATE_SECRET" not in repr(diagnostics)
    finally:
        await transport.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("truncated", [False, True])
async def test_provider_preserves_runtime_recovery_and_secondary_values_without_history(truncated):
    original = exception_diagnostics(
        PermissionError(13, "PRIVATE_SECRET", "C:/Users/private/file"),
        component="model_assets",
        operation="runtime",
        code="model_assets.runtime_failed",
    )
    cleanup = exception_diagnostics(
        OSError(13, "PRIVATE_CLEANUP_SECRET"),
        component="model_assets",
        operation="slot_cleanup",
        code="model_assets.cleanup_failed",
    )

    class RecoveredRuntime(FakeRuntime):
        async def translate(self, **kwargs):
            return ManagedGemmaResponse(
                "hello",
                ManagedGemmaMetrics(1, 1, 1, 1.0, 1.0, 1.0),
                recovery_diagnostics=(original,),
                secondary_diagnostics=(cleanup,),
                recovery_diagnostics_truncated=truncated,
                secondary_diagnostics_truncated=truncated,
            )

    provider = ManagedGemmaLLMProvider(runtime=RecoveredRuntime(), backend="cpu")
    result = await provider.translate(
        utterance_id=uuid4(),
        text="input",
        system_prompt="translate",
        source_language="ko",
        target_language="en",
    )
    await provider.close()
    assert result.text == "hello"
    assert len(result.recovered_failures) == 1
    recovered = serialize_diagnostics(deserialize_diagnostics(result.recovered_failures[0]))
    assert recovered == serialize_diagnostics(original)
    assert "PRIVATE_SECRET" not in repr(recovered)
    assert len(result.secondary_diagnostics) == 1
    secondary = serialize_diagnostics(deserialize_diagnostics(result.secondary_diagnostics[0]))
    assert secondary == serialize_diagnostics(cleanup)
    assert original.diagnostic_id != secondary["diagnostic_id"]
    assert "PRIVATE_CLEANUP_SECRET" not in repr(secondary)
    with pytest.raises(TypeError):
        result.secondary_diagnostics[0]["fields"]["errno"] = 42
    assert result.recovered_failures_truncated is truncated
    assert result.secondary_diagnostics_truncated is truncated
