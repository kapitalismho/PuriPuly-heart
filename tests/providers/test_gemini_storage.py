from __future__ import annotations

import json
from uuid import uuid4

import httpx
import pytest
from google.genai import errors, types

from puripuly_heart.core import network_clients
from puripuly_heart.providers.llm.error_details import LLMResponseError
from puripuly_heart.providers.llm.gemini import GeminiLLMProvider


@pytest.fixture
async def controlled_genai(monkeypatch):
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "false")
    requests = []
    response = {
        "status": 200,
        "body": {
            "candidates": [
                {
                    "content": {"role": "model", "parts": [{"text": " translated "}]},
                    "finishReason": "STOP",
                }
            ]
        },
    }

    def respond(request):
        requests.append(request)
        if len(requests) > 1:
            return httpx.Response(
                400,
                json={
                    "error": {
                        "code": 400,
                        "message": "unexpected follow-up request",
                        "status": "INVALID_ARGUMENT",
                    }
                },
            )
        return httpx.Response(response["status"], json=response["body"])

    transport = httpx.MockTransport(respond)
    sync_client = httpx.Client(transport=transport, trust_env=False)
    async_client = httpx.AsyncClient(transport=transport, trust_env=False)
    options = {
        "base_url": "https://synthetic.invalid",
        "httpx_client": sync_client,
        "httpx_async_client": async_client,
    }
    monkeypatch.setattr(network_clients, "genai_http_options", lambda: options)
    provider = GeminiLLMProvider(api_key="synthetic-not-a-credential", model="gemini-3.8-flash")
    try:
        yield provider, requests, response, sync_client, async_client
    finally:
        await provider.close()
        await async_client.aclose()
        sync_client.close()


async def _invoke(provider, operation):
    if operation == "warmup":
        return await provider.warmup()
    return await provider.translate(
        utterance_id=uuid4(),
        text="hello",
        system_prompt="Translate {source_language} to {target_language}.",
        source_language="en",
        target_language="ko",
        context="prior -> context",
        scene_participant_count=2,
        max_output_tokens=64,
    )


@pytest.mark.parametrize("operation", ["translation", "warmup"])
@pytest.mark.parametrize("finish_reason", ["STOP", "CONTINUATION"])
async def test_generate_content_serializes_explicit_storage_opt_out(
    controlled_genai, operation, finish_reason
):
    provider, requests, response, sync_client, async_client = controlled_genai
    response["body"]["candidates"][0]["finishReason"] = finish_reason
    if finish_reason == "CONTINUATION":
        response["body"]["continuationToken"] = "c3ludGhldGljLWNvbnRpbnVhdGlvbg=="
    result = await _invoke(provider, operation)
    if operation == "translation":
        assert result.text == "translated"
    else:
        assert result is None
    assert len(requests) == 1
    request = requests[0]
    assert request.method == "POST"
    assert str(request.url) == (
        "https://synthetic.invalid/v1beta/models/gemini-3.8-flash:generateContent"
    )
    body = json.loads(request.content)
    assert body["store"] is False
    assert body["generationConfig"] == {
        "temperature": 0.6,
        "thinkingConfig": {"thinking_level": "LOW"},
        **({"maxOutputTokens": 64} if operation == "translation" else {}),
    }
    assert body["systemInstruction"]["parts"] == [
        {"text": "Translate en to ko." if operation == "translation" else "Reply with OK only."}
    ]
    assert body["contents"] == [
        {
            "role": "user",
            "parts": [
                {
                    "text": (
                        "<context>\nprior -> context\n</context>\n\n"
                        "<scene>\nPeople: 2\n</scene>\n\n<input>\nhello\n</input>"
                        if operation == "translation"
                        else "<input>\nwarmup\n</input>"
                    )
                }
            ],
        }
    ]
    await provider.close()
    assert sync_client.is_closed
    assert async_client.is_closed


@pytest.mark.parametrize("operation", ["translation", "warmup"])
@pytest.mark.parametrize("failure", ["provider_error", "missing_text"])
async def test_generate_content_preserves_errors_and_owned_cleanup(
    controlled_genai, operation, failure
):
    provider, requests, response, sync_client, async_client = controlled_genai
    if failure == "provider_error":
        response.update(
            status=400,
            body={
                "error": {
                    "code": 400,
                    "message": "synthetic rejection",
                    "status": "INVALID_ARGUMENT",
                }
            },
        )
        expected = errors.ClientError
    else:
        response["body"] = {"candidates": []}
        expected = LLMResponseError
    with pytest.raises(expected) as caught:
        await _invoke(provider, operation)
    if failure == "provider_error":
        assert caught.value.code == 400
        assert "synthetic rejection" in str(caught.value)
    else:
        assert caught.value.failure_diagnostics.category == "invalid_response"
    assert len(requests) == 1
    assert json.loads(requests[0].content)["store"] is False
    await provider.close()
    assert sync_client.is_closed
    assert async_client.is_closed


@pytest.mark.parametrize("operation", ["translation", "warmup"])
async def test_storage_config_failure_propagates_without_opt_out_fallback(
    controlled_genai, monkeypatch, operation
):
    provider, requests, _, sync_client, async_client = controlled_genai
    original_config = types.GenerateContentConfig
    failure = ValueError("synthetic storage configuration rejection")
    configs = []

    def reject_storage(**kwargs):
        configs.append(kwargs)
        if "store" in kwargs:
            raise failure
        return original_config(**kwargs)

    monkeypatch.setattr(types, "GenerateContentConfig", reject_storage)
    with pytest.raises(ValueError) as caught:
        await _invoke(provider, operation)
    assert caught.value is failure
    assert len(configs) == 1
    assert configs[0]["store"] is False
    assert requests == []
    await provider.close()
    assert sync_client.is_closed
    assert async_client.is_closed
