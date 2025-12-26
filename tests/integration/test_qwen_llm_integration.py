from __future__ import annotations

import os
from uuid import uuid4

import pytest

from ajebal_daera_translator.providers.llm.qwen import QwenLLMProvider

pytestmark = pytest.mark.skipif(
    os.getenv("INTEGRATION") != "1", reason="set INTEGRATION=1 to run integration tests"
)


@pytest.mark.asyncio
async def test_qwen_llm_translation_smoke() -> None:
    api_key = os.getenv("ALIBABA_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        pytest.skip("missing env var ALIBABA_API_KEY (or DASHSCOPE_API_KEY)")

    try:
        import dashscope  # noqa: F401
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError(
            "dashscope is required for this integration test; install project dependencies."
        ) from exc

    provider = QwenLLMProvider(
        api_key=api_key,
        model=os.getenv("QWEN_LLM_MODEL", "qwen-mt-flash"),
    )

    translation = await provider.translate(
        utterance_id=uuid4(),
        text="안녕하세요",
        system_prompt="Translate from ${sourceName} to ${targetName}.",
        source_language="ko",
        target_language="en",
        context="",
    )

    assert translation.text
