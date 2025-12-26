import asyncio
import os
import sys
import time

sys.path.insert(0, "src")

from uuid import uuid4

from ajebal_daera_translator.providers.llm.gemini import GeminiLLMProvider


async def run():
    api_key = os.environ.get("GOOGLE_API_KEY", "")
    if not api_key:
        print("Missing GOOGLE_API_KEY")
        return

    llm = GeminiLLMProvider(api_key=api_key)

    tests = [
        "안녕하세요",
        "오늘 날씨가 좋네요",
        "이 게임 정말 재미있어요",
    ]

    print("=" * 60, flush=True)
    print("LLM TRANSLATION LATENCY TEST", flush=True)
    print("=" * 60, flush=True)

    total = 0

    for text in tests:
        start = time.perf_counter()
        try:
            result = await llm.translate(
                utterance_id=uuid4(),
                text=text,
                system_prompt="Translate naturally.",
                source_language="ko-KR",
                target_language="en-US",
            )
            latency = (time.perf_counter() - start) * 1000
            total += latency

            print(f"  Input: {text}", flush=True)
            print(f"  Output: {result.text}", flush=True)
            print(f"  Latency: {latency:.0f} ms", flush=True)
            print("-" * 30, flush=True)
        except Exception as e:
            print(f"  Error: {e}", flush=True)

    if len(tests) > 0:
        print("=" * 60, flush=True)
        print(f"Average LLM Latency: {total / len(tests):.0f} ms", flush=True)
        print("=" * 60, flush=True)


if __name__ == "__main__":
    asyncio.run(run())
