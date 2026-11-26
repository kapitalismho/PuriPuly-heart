from __future__ import annotations

from puripuly_heart.core.openrouter_routing import OpenRouterProviderRouting
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


def test_openrouter_internal_emergency_uses_single_model_and_modelrun_only() -> None:
    body = _body(
        HttpxOpenRouterClient(
            api_key="key",
            model="google/gemma-4-31b-it",
            provider_routing=OpenRouterProviderRouting.GEMMA4_31B_MODELRUN_ONLY,
        )
    )

    assert body["model"] == "google/gemma-4-31b-it"
    assert "models" not in body
    assert body["provider"] == {"only": ["modelrun"], "allow_fallbacks": False}


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


def test_gemma_combined_and_internal_emergency_provider_policies() -> None:
    assert _build_provider_preferences(OpenRouterProviderRouting.GEMMA4_26B_31B_LATENCY) == {
        "only": [
            "makora",
            "deepinfra/fp8",
            "kt",
        ],
        "max_price": {"prompt": 0.19, "completion": 0.39},
        "sort": {"by": "latency", "partition": "none"},
        "allow_fallbacks": True,
    }
    assert _build_provider_preferences(OpenRouterProviderRouting.GEMMA4_31B_KOREA) == {
        "only": ["kt"],
        "allow_fallbacks": False,
    }
    assert _build_provider_preferences(OpenRouterProviderRouting.GEMMA4_31B_MODELRUN_ONLY) == {
        "only": ["modelrun"],
        "allow_fallbacks": False,
    }


def test_openrouter_korea_uses_single_31b_model_and_kt_only() -> None:
    body = _body(
        HttpxOpenRouterClient(
            api_key="key",
            model="google/gemma-4-31b-it",
            provider_routing=OpenRouterProviderRouting.GEMMA4_31B_KOREA,
        )
    )

    assert body["model"] == "google/gemma-4-31b-it"
    assert "models" not in body
    assert body["provider"] == {"only": ["kt"], "allow_fallbacks": False}
