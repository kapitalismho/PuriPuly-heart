from __future__ import annotations

from puripuly_heart.config.settings import OpenRouterProviderRouting
from puripuly_heart.providers.llm.openrouter import (
    HttpxOpenRouterClient,
    _build_provider_preferences,
)


def _body(client: HttpxOpenRouterClient) -> dict[str, object]:
    return client._build_request_body(
        text="hello",
        system_prompt="translate",
        source_language="en",
        target_language="ko",
        context="",
    )


def test_openrouter_single_model_request_uses_model_only() -> None:
    body = _body(
        HttpxOpenRouterClient(
            api_key="key",
            model="google/gemma-4-31b-it",
            models=("google/gemma-4-31b-it",),
        )
    )

    assert body["model"] == "google/gemma-4-31b-it"
    assert "models" not in body


def test_openrouter_unified_request_uses_models_only() -> None:
    body = _body(
        HttpxOpenRouterClient(
            api_key="key",
            model="google/gemma-4-26b-a4b-it",
            models=(
                "google/gemma-4-26b-a4b-it",
                "google/gemma-4-31b-it",
            ),
            provider_routing=OpenRouterProviderRouting.GEMMA4_26B_31B_LATENCY,
        )
    )

    assert body["models"] == [
        "google/gemma-4-26b-a4b-it",
        "google/gemma-4-31b-it",
    ]
    assert "model" not in body


def test_gemma_semantic_routes_produce_issue_provider_policies() -> None:
    assert _build_provider_preferences(OpenRouterProviderRouting.GEMMA4_26B_31B_LATENCY) == {
        "only": ["cloudflare", "coreweave/bf16", "friendli"],
        "sort": {"by": "latency", "partition": "none"},
        "allow_fallbacks": True,
    }
    assert _build_provider_preferences(OpenRouterProviderRouting.GEMMA4_31B_LATENCY) == {
        "only": ["coreweave/bf16", "friendli"],
        "sort": {"by": "latency"},
        "allow_fallbacks": True,
    }
    assert _build_provider_preferences(OpenRouterProviderRouting.GEMMA4_26B_LATENCY) == {
        "only": ["cloudflare", "parasail/bf16"],
        "sort": {"by": "latency"},
        "allow_fallbacks": True,
    }
    assert _build_provider_preferences(OpenRouterProviderRouting.GEMMA4_31B_CEREBRAS_ONLY) == {
        "only": ["cerebras/fp16"],
        "allow_fallbacks": False,
    }
