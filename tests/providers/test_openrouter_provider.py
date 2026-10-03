from __future__ import annotations

import logging
from dataclasses import dataclass
from uuid import uuid4

import httpx
import pytest

from puripuly_heart.config.llm_profiles import OPENROUTER_MODEL_GPT_6_LUNA
from puripuly_heart.config.runtime_resolution import (
    OpenRouterRuntimeIntent,
    RuntimeResolutionInput,
    TranslationRuntimeIntent,
    resolve_llm_config,
)
from puripuly_heart.core.diagnostic_validation import (
    DIAGNOSTIC_SINK_DASHBOARD,
    DIAGNOSTIC_VALIDATION_STATUS_ACCEPTED,
    validate_diagnostics_for_sink,
)
from puripuly_heart.core.error_messages import provider_failure_report
from puripuly_heart.core.openrouter_routing import (
    OpenRouterProviderRouting,
    OpenRouterRoutingMode,
)
from puripuly_heart.providers.llm.openrouter import (
    HttpxOpenRouterClient,
    OpenRouterClient,
    OpenRouterKeyMetadata,
    OpenRouterLLMProvider,
    OpenRouterResponseError,
)


@dataclass
class FakeOpenRouterClient(OpenRouterClient):
    last_call: dict[str, object] | None = None
    closed: bool = False

    async def translate(
        self,
        *,
        text: str,
        system_prompt: str,
        source_language: str,
        target_language: str,
        context: str = "",
        scene_participant_count: int | None = None,
    ) -> str:
        self.last_call = {
            "text": text,
            "system_prompt": system_prompt,
            "source_language": source_language,
            "target_language": target_language,
            "context": context,
            "scene_participant_count": scene_participant_count,
        }
        return "TRANSLATED"

    async def close(self) -> None:
        self.closed = True


class SpyRuntimeLogging:
    def __init__(self, *, detailed_return: bool = False) -> None:
        self.detailed_return = detailed_return
        self.detailed_messages: list[tuple[str, int]] = []
        self.basic_messages: list[tuple[str, int]] = []

    def emit_diagnostic(self, message: str, *, level: int = logging.INFO) -> bool:
        self.detailed_messages.append((message, level))
        return self.detailed_return

    def emit_basic(self, message: str, *, level: int = logging.INFO) -> None:
        self.basic_messages.append((message, level))


class FakeResponse:
    status_code = 200
    headers: dict[str, str] = {}

    def __init__(self, data: dict | None = None):
        self._data = data or {"choices": [{"message": {"content": "OK"}}]}

    def json(self):
        return self._data

    def raise_for_status(self):
        pass


class FakeAsyncClient:
    def __init__(
        self,
        *,
        response_data: dict | None = None,
    ):
        self.last_request: dict = {}
        self.requests: list[dict] = []
        self.closed = False
        self._response_data = response_data

    async def aclose(self):
        self.closed = True

    async def post(self, url, **kwargs):
        request = {"url": url, **kwargs}
        self.last_request = request
        self.requests.append(request)
        return FakeResponse(self._response_data)


@pytest.mark.asyncio
async def test_openrouter_provider_uses_injected_client() -> None:
    fake = FakeOpenRouterClient()
    provider = OpenRouterLLMProvider(api_key="k", client=fake)

    utterance_id = uuid4()
    out = await provider.translate(
        utterance_id=utterance_id,
        text="hello",
        system_prompt="PROMPT",
        source_language="ko-KR",
        target_language="en",
    )

    assert out.utterance_id == utterance_id
    assert out.text == "TRANSLATED"
    assert fake.last_call == {
        "text": "hello",
        "system_prompt": "PROMPT",
        "source_language": "ko-KR",
        "target_language": "en",
        "context": "",
        "scene_participant_count": None,
    }


@pytest.mark.asyncio
async def test_openrouter_provider_close_closes_owned_http_client_and_not_injected_client(
    monkeypatch,
) -> None:
    fake_client = FakeAsyncClient()
    monkeypatch.setattr("httpx.AsyncClient", lambda **_kwargs: fake_client)
    provider = OpenRouterLLMProvider(api_key="k")

    await provider.translate(
        utterance_id=uuid4(),
        text="hello",
        system_prompt="PROMPT",
        source_language="ko-KR",
        target_language="en",
    )
    await provider.close()

    assert fake_client.closed is True

    injected = FakeOpenRouterClient()
    owner = OpenRouterLLMProvider(api_key="k", client=injected)
    await owner.close()

    assert injected.closed is False


@pytest.mark.asyncio
async def test_openrouter_provider_propagates_max_tokens_to_request(monkeypatch) -> None:
    fake_client = FakeAsyncClient()
    monkeypatch.setattr("httpx.AsyncClient", lambda **_kwargs: fake_client)
    provider = OpenRouterLLMProvider(api_key="k", max_tokens=17)

    await provider.translate(
        utterance_id=uuid4(),
        text="hello",
        system_prompt="PROMPT",
        source_language="ko-KR",
        target_language="en",
    )

    assert fake_client.last_request["json"]["max_tokens"] == 17


@pytest.mark.asyncio
async def test_openrouter_provider_propagates_user_identifier_to_request(monkeypatch) -> None:
    fake_client = FakeAsyncClient()
    monkeypatch.setattr("httpx.AsyncClient", lambda **_kwargs: fake_client)
    provider = OpenRouterLLMProvider(api_key="k", user_identifier="user-123")

    await provider.translate(
        utterance_id=uuid4(),
        text="hello",
        system_prompt="PROMPT",
        source_language="ko-KR",
        target_language="en",
    )

    assert fake_client.last_request["json"]["user"] == "user-123"


@pytest.mark.asyncio
async def test_openrouter_verify_api_key_uses_key_endpoint(monkeypatch) -> None:
    seen: dict[str, object] = {}

    class FakeResponse:
        status_code = 200

    class FakeAsyncClient:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def get(self, url, **kwargs):
            seen["url"] = url
            seen["headers"] = kwargs["headers"]
            return FakeResponse()

    monkeypatch.setattr("httpx.AsyncClient", FakeAsyncClient)

    ok = await OpenRouterLLMProvider.verify_api_key("secret")

    assert ok is True
    assert seen["url"] == "https://openrouter.ai/api/v1/key"
    assert seen["headers"]["Authorization"] == "Bearer secret"


@pytest.mark.asyncio
async def test_openrouter_fetch_key_metadata_uses_key_endpoint(monkeypatch) -> None:
    seen: dict[str, object] = {}

    class FakeResponse:
        status_code = 200

        def json(self):
            return {
                "data": {
                    "limit": 0.07,
                    "limit_remaining": 0.05,
                    "usage": 0.02,
                }
            }

        def raise_for_status(self):
            return None

    class FakeAsyncClient:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def get(self, url, **kwargs):
            seen["url"] = url
            seen["headers"] = kwargs["headers"]
            return FakeResponse()

    monkeypatch.setattr("httpx.AsyncClient", FakeAsyncClient)

    metadata = await OpenRouterLLMProvider.fetch_key_metadata("secret")

    assert metadata == OpenRouterKeyMetadata(limit_usd=0.07, remaining_usd=0.05, usage_usd=0.02)
    assert seen["url"] == "https://openrouter.ai/api/v1/key"
    assert seen["headers"]["Authorization"] == "Bearer secret"


@pytest.mark.asyncio
async def test_httpx_openrouter_client_builds_reasoning_disabled_request_with_latency_sort(
    monkeypatch,
) -> None:
    fake_client = FakeAsyncClient()
    monkeypatch.setattr("httpx.AsyncClient", lambda **_kwargs: fake_client)

    client = HttpxOpenRouterClient(
        api_key="test-key",
        model="google/gemma-4-26b-a4b-it",
        base_url="https://example",
        user_identifier="  managed-user-123  ",
    )
    result = await client.translate(
        text="hello",
        system_prompt="SYSTEM",
        source_language="ko-KR",
        target_language="en",
        context='- "previous"',
    )

    assert result == "OK"
    assert fake_client.last_request["url"] == "https://example/chat/completions"
    headers = fake_client.last_request["headers"]
    assert headers["Authorization"] == "Bearer test-key"
    assert headers["Content-Type"] == "application/json"

    body = fake_client.last_request["json"]
    assert body["model"] == "google/gemma-4-26b-a4b-it"
    assert body["max_tokens"] == 100
    assert body["reasoning"] == {"effort": "none"}
    assert body["temperature"] == 0.6
    assert body["user"] == "managed-user-123"
    assert body["provider"] == {
        "sort": {"by": "latency"},
        "only": ["cloudflare", "dekallm/bf16", "nextbit/bf16", "makora"],
        "allow_fallbacks": True,
    }
    assert body["messages"][0] == {"role": "system", "content": "SYSTEM"}
    assert "prompt_cache_options" not in body
    assert body["messages"][1]["role"] == "user"
    assert "<context>" in body["messages"][1]["content"]
    assert "</context>" in body["messages"][1]["content"]
    assert "<input>\nhello\n</input>" in body["messages"][1]["content"]
    assert "Input: hello" not in body["messages"][1]["content"]


@pytest.mark.asyncio
async def test_httpx_openrouter_luna_disables_reasoning_and_omits_temperature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_client = FakeAsyncClient()
    monkeypatch.setattr(httpx, "AsyncClient", lambda **_kwargs: fake_client)
    client = HttpxOpenRouterClient(
        api_key="test-key",
        model=OPENROUTER_MODEL_GPT_6_LUNA,
        base_url="https://example",
    )

    result = await client.translate(
        text="조금만 더 기다려 주세요.",
        system_prompt="Translate {source_language} to {target_language}.",
        source_language="Korean",
        target_language="English",
        context="At a station, someone asks a companion to wait.",
        max_output_tokens=37,
    )

    assert result == "OK"
    assert fake_client.last_request["json"]["model"] == OPENROUTER_MODEL_GPT_6_LUNA
    assert fake_client.last_request["json"]["reasoning"] == {"effort": "none"}
    assert fake_client.last_request["json"]["max_tokens"] == 37
    assert "temperature" not in fake_client.last_request["json"]
    assert fake_client.last_request["json"]["prompt_cache_options"] == {
        "mode": "explicit",
        "ttl": "30m",
    }
    assert fake_client.last_request["json"]["messages"] == [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "Translate Korean to English.",
                    "prompt_cache_breakpoint": {"mode": "explicit"},
                }
            ],
        },
        {
            "role": "user",
            "content": (
                "<context>\nAt a station, someone asks a companion to wait.\n</context>\n\n"
                "<input>\n조금만 더 기다려 주세요.\n</input>"
            ),
        },
    ]
    await client.close()


@pytest.mark.parametrize(
    ("models", "explicit_cache"),
    [
        ((OPENROUTER_MODEL_GPT_6_LUNA,), True),
        (("google/gemma-4-26b-a4b-it",), False),
        ((f"{OPENROUTER_MODEL_GPT_6_LUNA}-preview",), False),
        ((OPENROUTER_MODEL_GPT_6_LUNA, "google/gemma-4-26b-a4b-it"), False),
        (("google/gemma-4-26b-a4b-it", OPENROUTER_MODEL_GPT_6_LUNA), False),
    ],
)
def test_openrouter_cache_schema_requires_luna_only_routing_and_preserves_custom_prompt(
    models: tuple[str, ...], explicit_cache: bool
) -> None:
    client = HttpxOpenRouterClient(
        api_key="test-key",
        model=models[0],
        models=models,
        user_identifier="managed-user-123",
    )
    custom_prompt = "Custom instructions\nKeep {literal} exactly as written."
    requests = [
        client._build_request_body(
            text=text,
            system_prompt=custom_prompt,
            source_language="Korean",
            target_language="English",
            context=context,
            scene_participant_count=participants,
            max_output_tokens=37,
        )
        for text, context, participants in (
            ("first input", "first context", 2),
            ("second input", "second context", 3),
        )
    ]

    expected_content: object = custom_prompt
    if explicit_cache:
        expected_content = [
            {
                "type": "text",
                "text": custom_prompt,
                "prompt_cache_breakpoint": {"mode": "explicit"},
            }
        ]
    for body in requests:
        assert body["messages"][0] == {"role": "system", "content": expected_content}
        assert isinstance(body["messages"][1]["content"], str)
        assert body["max_tokens"] == 37
        assert body["user"] == "managed-user-123"
        if len(models) == 1:
            assert body["model"] == models[0]
            assert "models" not in body
        else:
            assert body["models"] == list(models)
            assert "model" not in body
        if explicit_cache:
            assert body["prompt_cache_options"] == {"mode": "explicit", "ttl": "30m"}
        else:
            assert "prompt_cache_options" not in body
    assert requests[0]["messages"][0] == requests[1]["messages"][0]
    assert requests[0]["messages"][1] != requests[1]["messages"][1]


@pytest.mark.asyncio
async def test_httpx_openrouter_client_gemma_uses_cloudflare_dekallm_nextbit_makora_routing(
    monkeypatch,
) -> None:
    fake_client = FakeAsyncClient()
    monkeypatch.setattr("httpx.AsyncClient", lambda **_kwargs: fake_client)

    client = HttpxOpenRouterClient(
        api_key="test-key",
        model="google/gemma-4-26b-a4b-it",
        base_url="https://example",
    )
    await client.translate(
        text="hello",
        system_prompt="SYSTEM",
        source_language="ko-KR",
        target_language="en",
    )

    body = fake_client.last_request["json"]
    assert body["provider"] == {
        "sort": {"by": "latency"},
        "only": ["cloudflare", "dekallm/bf16", "nextbit/bf16", "makora"],
        "allow_fallbacks": True,
    }


@pytest.mark.asyncio
async def test_httpx_openrouter_client_google_gemini_latency_denies_data_collection(
    monkeypatch,
) -> None:
    fake_client = FakeAsyncClient()
    monkeypatch.setattr("httpx.AsyncClient", lambda **_kwargs: fake_client)

    client = HttpxOpenRouterClient(
        api_key="test-key",
        model="google/gemini-3.8-flash",
        base_url="https://example",
        provider_routing=OpenRouterProviderRouting.GOOGLE_GEMINI_LATENCY,
    )
    await client.translate(
        text="hello",
        system_prompt="SYSTEM",
        source_language="ko-KR",
        target_language="en",
    )

    body = fake_client.last_request["json"]
    assert body["provider"] == {
        "sort": "latency",
        "only": ["google-vertex", "google-ai-studio"],
        "allow_fallbacks": True,
        "data_collection": "deny",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "provider_routing",
    [
        OpenRouterProviderRouting.DEEPSEEK_ONLY,
        OpenRouterProviderRouting.DEEPSEEK_V4_FLASH_LATENCY,
    ],
)
async def test_httpx_openrouter_client_deepseek_41_routing_is_strict(
    monkeypatch,
    provider_routing: OpenRouterProviderRouting,
) -> None:
    fake_client = FakeAsyncClient()
    monkeypatch.setattr("httpx.AsyncClient", lambda **_kwargs: fake_client)

    client = HttpxOpenRouterClient(
        api_key="test-key",
        model="deepseek/deepseek-v4.1-flash",
        base_url="https://example",
        routing_mode=OpenRouterRoutingMode.LATENCY,
        provider_routing=provider_routing,
    )
    await client.translate(
        text="hello",
        system_prompt="SYSTEM",
        source_language="ko-KR",
        target_language="zh-CN",
    )

    body = fake_client.last_request["json"]
    assert body["provider"] == {
        "only": ["deepseek", "wafer"],
        "allow_fallbacks": False,
    }


@pytest.mark.asyncio
async def test_httpx_openrouter_client_deepseek_41_model_overrides_stale_route(
    monkeypatch,
) -> None:
    fake_client = FakeAsyncClient()
    monkeypatch.setattr("httpx.AsyncClient", lambda **_kwargs: fake_client)

    client = HttpxOpenRouterClient(
        api_key="test-key",
        model="deepseek/deepseek-v4.1-flash",
        base_url="https://example",
        provider_routing=OpenRouterProviderRouting.GEMMA4_26B_31B_LATENCY,
    )
    await client.translate(
        text="hello",
        system_prompt="SYSTEM",
        source_language="ko-KR",
        target_language="zh-CN",
    )

    body = fake_client.last_request["json"]
    assert body["provider"] == {
        "only": ["deepseek", "wafer"],
        "allow_fallbacks": False,
    }


@pytest.mark.asyncio
async def test_httpx_openrouter_client_deepseek_40_uses_requested_provider_pool(
    monkeypatch,
) -> None:
    fake_client = FakeAsyncClient()
    monkeypatch.setattr("httpx.AsyncClient", lambda **_kwargs: fake_client)
    client = HttpxOpenRouterClient(
        api_key="test-key",
        model="deepseek/deepseek-v4-flash-0731",
        base_url="https://example",
        provider_routing=OpenRouterProviderRouting.GEMMA4_26B_31B_LATENCY,
    )
    await client.translate(
        text="hello",
        system_prompt="SYSTEM",
        source_language="ko-KR",
        target_language="zh-CN",
    )

    assert fake_client.last_request["json"]["provider"] == {
        "only": [
            "makora",
            "together",
            "wafer/fast",
            "baidu/fp8",
        ],
        "sort": {"by": "latency", "partition": "none"},
        "allow_fallbacks": True,
    }


@pytest.mark.asyncio
async def test_httpx_openrouter_client_deepseek_40_china_pins_baidu(
    monkeypatch,
) -> None:
    fake_client = FakeAsyncClient()
    monkeypatch.setattr("httpx.AsyncClient", lambda **_kwargs: fake_client)
    client = HttpxOpenRouterClient(
        api_key="test-key",
        model="deepseek/deepseek-v4-flash-0731",
        base_url="https://example",
        provider_routing=OpenRouterProviderRouting.DEEPSEEK_V4_FLASH_CHINA,
    )
    await client.translate(
        text="hello",
        system_prompt="SYSTEM",
        source_language="ko-KR",
        target_language="zh-CN",
    )

    assert fake_client.last_request["json"]["provider"] == {
        "only": ["baidu/fp8"],
        "allow_fallbacks": False,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("model", "primary_connection", "expected_route", "expected_preferences"),
    [
        (
            "deepseek_v4_flash",
            "openrouter",
            "deepseek_v4_flash_latency",
            {
                "only": [
                    "makora",
                    "together",
                    "wafer/fast",
                    "baidu/fp8",
                ],
                "sort": {"by": "latency", "partition": "none"},
                "allow_fallbacks": True,
            },
        ),
        (
            "deepseek_v4_flash_41",
            "managed_china",
            "deepseek_v4_flash_41_strict",
            {
                "only": ["deepseek", "wafer"],
                "allow_fallbacks": False,
            },
        ),
    ],
)
async def test_resolved_deepseek_fallback_preserves_primary_provider_pool(
    monkeypatch,
    model: str,
    primary_connection: str,
    expected_route: str,
    expected_preferences: dict[str, object],
) -> None:
    fake_client = FakeAsyncClient()
    monkeypatch.setattr("httpx.AsyncClient", lambda **_kwargs: fake_client)
    resolved = resolve_llm_config(
        RuntimeResolutionInput(
            translation=TranslationRuntimeIntent(
                model=model,
                connection=primary_connection,
            ),
            openrouter=OpenRouterRuntimeIntent(selected_source="byok"),
        )
    )

    assert resolved.fallback is not None
    fallback = resolved.fallback.target
    assert fallback.provider_routing == expected_route
    client = HttpxOpenRouterClient(
        api_key="test-key",
        model=fallback.model,
        base_url="https://example",
        provider_routing=OpenRouterProviderRouting(fallback.provider_routing),
    )
    await client.translate(
        text="hello",
        system_prompt="SYSTEM",
        source_language="ko-KR",
        target_language="zh-CN",
    )
    assert fake_client.last_request["json"]["provider"] == expected_preferences


@pytest.mark.asyncio
async def test_httpx_openrouter_client_gemma_pool_ignores_explicit_latency_routing_mode(
    monkeypatch,
) -> None:
    fake_client = FakeAsyncClient()
    monkeypatch.setattr("httpx.AsyncClient", lambda **_kwargs: fake_client)

    client = HttpxOpenRouterClient(
        api_key="test-key",
        model="google/gemma-4-26b-a4b-it",
        base_url="https://example",
        routing_mode=OpenRouterRoutingMode.LATENCY,
    )
    result = await client.translate(
        text="hello",
        system_prompt="SYSTEM",
        source_language="ko-KR",
        target_language="en",
    )

    assert result == "OK"
    body = fake_client.last_request["json"]
    assert body["provider"] == {
        "sort": {"by": "latency"},
        "only": ["cloudflare", "dekallm/bf16", "nextbit/bf16", "makora"],
        "allow_fallbacks": True,
    }


@pytest.mark.asyncio
async def test_httpx_openrouter_client_translate_raises_on_length_finish_reason(
    monkeypatch,
) -> None:
    fake_client = FakeAsyncClient(
        response_data={
            "choices": [
                {
                    "message": {"content": "partial"},
                    "finish_reason": "length",
                }
            ]
        }
    )
    monkeypatch.setattr("httpx.AsyncClient", lambda **_kwargs: fake_client)

    client = HttpxOpenRouterClient(api_key="k", model="m", base_url="https://example")

    with pytest.raises(RuntimeError, match="truncated"):
        await client.translate(
            text="hello",
            system_prompt="SYSTEM",
            source_language="ko",
            target_language="en",
        )


@pytest.mark.asyncio
async def test_httpx_openrouter_client_logs_basic_translate_failure_without_runtime_logging(
    monkeypatch, caplog: pytest.LogCaptureFixture
) -> None:
    class ErrorResponse(FakeResponse):
        status_code = 429

        def __init__(self):
            super().__init__({"error": {"message": "quota exceeded"}})

        def raise_for_status(self):
            raise RuntimeError("quota exceeded")

    class ErrorAsyncClient(FakeAsyncClient):
        async def post(self, url, **kwargs):
            request = {"url": url, **kwargs}
            self.last_request = request
            self.requests.append(request)
            return ErrorResponse()

    fake_client = ErrorAsyncClient()
    monkeypatch.setattr("httpx.AsyncClient", lambda **_kwargs: fake_client)

    client = HttpxOpenRouterClient(api_key="k", model="m", base_url="https://example")

    with caplog.at_level(logging.INFO, logger="puripuly_heart.providers.llm.openrouter"):
        with pytest.raises(OpenRouterResponseError):
            await client.translate(
                text="hello",
                system_prompt="SYSTEM",
                source_language="ko",
                target_language="en",
            )

    assert len(caplog.records) == 1
    assert caplog.records[0].levelno == logging.ERROR
    assert "quota exceeded" not in caplog.messages[0]


@pytest.mark.asyncio
async def test_httpx_openrouter_client_runtime_logging_logs_basic_translate_failure(
    monkeypatch, caplog: pytest.LogCaptureFixture
) -> None:
    class ErrorResponse(FakeResponse):
        status_code = 429

        def __init__(self):
            super().__init__({"error": {"message": "quota exceeded"}})

    class ErrorAsyncClient(FakeAsyncClient):
        async def post(self, url, **kwargs):
            request = {"url": url, **kwargs}
            self.last_request = request
            self.requests.append(request)
            return ErrorResponse()

    fake_client = ErrorAsyncClient()
    monkeypatch.setattr("httpx.AsyncClient", lambda **_kwargs: fake_client)
    runtime_logging = SpyRuntimeLogging(detailed_return=False)

    client = HttpxOpenRouterClient(
        api_key="k",
        model="m",
        base_url="https://example",
        runtime_logging=runtime_logging,
    )

    with caplog.at_level(logging.INFO, logger="puripuly_heart.providers.llm.openrouter"):
        with pytest.raises(OpenRouterResponseError):
            await client.translate(
                text="hello",
                system_prompt="SYSTEM",
                source_language="ko",
                target_language="en",
            )

    assert runtime_logging.detailed_messages == []
    assert len(runtime_logging.basic_messages) == 1
    failure, level = runtime_logging.basic_messages[0]
    assert level == logging.ERROR
    assert "category=rate_limit code=provider.rate_limit" in failure
    assert "operation=translate status=429 provider=openrouter" in failure
    assert "exception_type=OpenRouterResponseError" in failure
    assert "quota exceeded" not in failure
    assert caplog.messages == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("metadata", "expected_key", "expected_fields"),
    [
        (
            {"limit_source": "openrouter_credits"},
            "provider.openrouter.insufficient_credits",
            {"limit_source": "openrouter_credits"},
        ),
        (
            {"limit_source": "openrouter_key_limit"},
            "provider.openrouter.key_limit",
            {"limit_source": "openrouter_key_limit"},
        ),
        (
            {"limit_source": "openrouter_in_flight_budget", "reason": "in_flight_budget_exhausted"},
            "provider.openrouter.temporary_limit",
            {
                "limit_source": "openrouter_in_flight_budget",
                "limit_reason": "in_flight_budget_exhausted",
            },
        ),
        (
            {"limit_source": "openrouter_credits", "reason": "weight_exceeds_budget"},
            "provider.openrouter.payment_required",
            {"limit_source": "openrouter_credits", "limit_reason": "weight_exceeds_budget"},
        ),
        (
            {"limit_source": "unknown", "reason": "in_flight_budget_exhausted"},
            "provider.openrouter.payment_required",
            {},
        ),
        ({"limit_source": ["openrouter_credits"]}, "provider.openrouter.payment_required", {}),
        (None, "provider.openrouter.payment_required", {}),
    ],
)
async def test_openrouter_402_subcause_reaches_safe_report(
    monkeypatch: pytest.MonkeyPatch,
    metadata: object,
    expected_key: str,
    expected_fields: dict[str, str],
) -> None:
    secret = "sk-provider-secret-123456789"
    payload = {"error": {"message": f"Authorization: Bearer {secret}", "metadata": metadata}}
    response = httpx.Response(
        402,
        json=payload,
        headers={"Retry-After": "12"},
        request=httpx.Request("POST", "https://example/chat/completions"),
    )

    class ErrorClient(FakeAsyncClient):
        async def post(self, url, **kwargs):
            return response

    monkeypatch.setattr("httpx.AsyncClient", lambda **_kwargs: ErrorClient())
    client = HttpxOpenRouterClient(api_key=secret, model="m", base_url="https://example")
    with pytest.raises(OpenRouterResponseError) as failure:
        await client.translate(
            text="hi", system_prompt="prompt", source_language="ko", target_language="en"
        )
    report = provider_failure_report(failure.value, provider="llm", operation="translate")

    assert report.message.key == expected_key
    assert report.message.params["provider"] == "openrouter"
    assert report.diagnostics.status_code == 402
    assert report.diagnostics.category == "quota"
    assert report.diagnostics.retry_after_ms == 12_000
    for key, value in expected_fields.items():
        assert report.diagnostics.fields[key] == value
    assert ("limit_source" in report.diagnostics.fields) == ("limit_source" in expected_fields)
    assert ("limit_reason" in report.diagnostics.fields) == ("limit_reason" in expected_fields)
    assert validate_diagnostics_for_sink(report.diagnostics, DIAGNOSTIC_SINK_DASHBOARD).status == (
        DIAGNOSTIC_VALIDATION_STATUS_ACCEPTED
    )
    assert secret not in repr(failure.value)
    assert secret not in repr(report)
    await client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("body", [b"not json sk-provider-secret-123456789", b"[]"])
async def test_openrouter_402_unparseable_body_is_neutral_and_not_logged(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, body: bytes
) -> None:
    response = httpx.Response(
        402,
        content=body,
        request=httpx.Request("POST", "https://example/chat/completions"),
    )

    class ErrorClient(FakeAsyncClient):
        async def post(self, url, **kwargs):
            return response

    monkeypatch.setattr("httpx.AsyncClient", lambda **_kwargs: ErrorClient())
    client = HttpxOpenRouterClient(api_key="sk-provider-secret-123456789", model="m")
    with caplog.at_level(logging.INFO, logger="puripuly_heart.providers.llm.openrouter"):
        with pytest.raises(OpenRouterResponseError) as failure:
            await client.translate(
                text="hi", system_prompt="prompt", source_language="ko", target_language="en"
            )
    report = provider_failure_report(failure.value, provider="llm", operation="translate")
    assert report.message.key == "provider.openrouter.payment_required"
    assert report.diagnostics.status_code == 402
    assert "sk-provider-secret-123456789" not in repr(failure.value)
    assert "sk-provider-secret-123456789" not in repr(report)
    assert "sk-provider-secret-123456789" not in repr(caplog.messages)
    await client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("retry_after", "expected"),
    [("0", 0), ("86400", 86_400_000), ("86401", None), ("-1", None), ("not-a-date", None)],
)
async def test_openrouter_retry_after_only_accepts_bounded_valid_values(
    monkeypatch: pytest.MonkeyPatch, retry_after: str, expected: int | None
) -> None:
    response = httpx.Response(
        402,
        json={"error": {"metadata": {"limit_source": "openrouter_in_flight_budget"}}},
        headers={"Retry-After": retry_after},
        request=httpx.Request("POST", "https://example/chat/completions"),
    )

    class ErrorClient(FakeAsyncClient):
        async def post(self, url, **kwargs):
            return response

    monkeypatch.setattr("httpx.AsyncClient", lambda **_kwargs: ErrorClient())
    client = HttpxOpenRouterClient(api_key="test", model="m")
    try:
        with pytest.raises(OpenRouterResponseError) as failure:
            await client.translate(
                text="hi", system_prompt="prompt", source_language="ko", target_language="en"
            )
        report = provider_failure_report(failure.value, provider="llm", operation="translate")
        assert report.message.key == "provider.openrouter.temporary_limit"
        assert report.diagnostics.retry_after_ms == expected
    finally:
        await client.close()
