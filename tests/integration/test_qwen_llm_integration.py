from __future__ import annotations

import os

import pytest

from puripuly_heart.providers.llm.qwen_async import AsyncQwenLLMProvider
from tests.integration.helpers import (
    get_async_qwen_base_url,
    integration_mark,
    require_env,
    run_llm_smoke,
    suppressed_runtime_logger,
)

pytestmark = integration_mark()


@pytest.mark.asyncio
async def test_async_qwen_llm_translation_smoke() -> None:
    api_key = require_env("ALIBABA_API_KEY")

    provider = AsyncQwenLLMProvider(
        api_key=api_key,
        base_url=get_async_qwen_base_url(),
        model=os.getenv("QWEN_LLM_MODEL", "qwen3.8-flash"),
        runtime_logging=suppressed_runtime_logger(),
    )

    await run_llm_smoke(provider)
