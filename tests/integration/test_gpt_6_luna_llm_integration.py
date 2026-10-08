from __future__ import annotations

import os
from uuid import uuid4

import pytest

from puripuly_heart.app.wiring.wiring_secrets_factory import create_secret_store
from puripuly_heart.config.llm_profiles import OPENROUTER_MODEL_GPT_6_LUNA
from puripuly_heart.config.paths import default_settings_path
from puripuly_heart.config.prompts import render_translation_prompt_template, resolve_system_prompt
from puripuly_heart.config.runtime_resolution import OPENAI_MODEL_GPT_6_LUNA
from puripuly_heart.config.settings_vnext.facade import load_vnext_settings
from puripuly_heart.providers.llm.openai import HttpxOpenAIClient, OpenAILLMProvider
from puripuly_heart.providers.llm.openrouter import HttpxOpenRouterClient, OpenRouterLLMProvider
from tests.integration.helpers import integration_mark

pytestmark = integration_mark()


async def _run_contextual_korean_cases(
    *,
    route: str,
    model: str,
    temperature_setting: str,
    provider: OpenAILLMProvider | OpenRouterLLMProvider,
    client: HttpxOpenAIClient | HttpxOpenRouterClient,
) -> None:
    cases = (
        (
            "English",
            "잠깐만 더 기다려 주세요.",
            "At the station, the speaker asks their companion to wait a little longer.",
            "wait",
        ),
        (
            "Japanese",
            "그럼 여기서 잠시 쉬었다 가요.",
            "Two hikers are tired on a mountain trail and agree to rest here before continuing.",
            "休",
        ),
    )
    prompt_template = resolve_system_prompt(None, model=model)
    for target_language, text, context, expected_fragment in cases:
        prompt = render_translation_prompt_template(
            prompt_template,
            source_name="Korean",
            target_name=target_language,
        )
        translation = await provider.translate(
            utterance_id=uuid4(),
            text=text,
            system_prompt=prompt,
            source_language="Korean",
            target_language=target_language,
            context=context,
            scene_participant_count=2,
            max_output_tokens=64,
        )
        translated = translation.text.strip()
        assert translated
        assert translated != text
        if target_language == "English":
            assert expected_fragment in translated.casefold()
        else:
            assert expected_fragment in translated
        reasoning_tokens = client.last_reasoning_tokens
        reasoning_evidence = "not_exposed" if reasoning_tokens is None else str(reasoning_tokens)
        output_limit = "max_completion_tokens=64" if route == "openai" else "max_tokens=64"
        print(
            f"route={route} model={model} target={target_language} "
            f"reasoning_effort=none temperature={temperature_setting} {output_limit} "
            f"reasoning_tokens={reasoning_evidence} translation=valid"
        )


def _configured_api_key(*, env_var: str, secret_key: str) -> str | None:
    value = os.getenv(env_var)
    if isinstance(value, str) and value.strip():
        return value.strip()
    settings_path = default_settings_path()
    try:
        loaded = load_vnext_settings(settings_path)
        settings = loaded.settings
        if settings is None:
            return None
        value = create_secret_store(
            settings.intent.secrets,
            config_path=settings_path,
        ).get(secret_key)
    except Exception:
        return None
    if not isinstance(value, str):
        return None
    return value.strip() or None


@pytest.mark.asyncio
async def test_openai_gpt_6_luna_contextual_translation_smoke() -> None:
    api_key = _configured_api_key(env_var="OPENAI_API_KEY", secret_key="openai_api_key")
    if api_key is None:
        pytest.skip("OPENAI_API_KEY or its configured secret-store entry is unavailable")
    client = HttpxOpenAIClient(api_key=api_key, model=OPENAI_MODEL_GPT_6_LUNA)
    provider = OpenAILLMProvider(api_key=api_key, model=OPENAI_MODEL_GPT_6_LUNA, client=client)
    try:
        await _run_contextual_korean_cases(
            route="openai",
            model=OPENAI_MODEL_GPT_6_LUNA,
            temperature_setting="0.6",
            provider=provider,
            client=client,
        )
    finally:
        await provider.close()
        await client.close()


@pytest.mark.asyncio
async def test_openrouter_gpt_6_luna_contextual_translation_smoke() -> None:
    api_key = _configured_api_key(
        env_var="OPENROUTER_API_KEY",
        secret_key="openrouter_api_key",
    )
    if api_key is None:
        pytest.skip("OPENROUTER_API_KEY or its configured secret-store entry is unavailable")
    client = HttpxOpenRouterClient(api_key=api_key, model=OPENROUTER_MODEL_GPT_6_LUNA)
    provider = OpenRouterLLMProvider(
        api_key=api_key,
        model=OPENROUTER_MODEL_GPT_6_LUNA,
        client=client,
    )
    try:
        await _run_contextual_korean_cases(
            route="openrouter",
            model=OPENROUTER_MODEL_GPT_6_LUNA,
            temperature_setting="0.6",
            provider=provider,
            client=client,
        )
    finally:
        await provider.close()
        await client.close()
