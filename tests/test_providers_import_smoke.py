def test_providers_import_smoke():
    from puripuly_heart.providers.llm import (
        gemini,  # noqa: F401
        qwen,  # noqa: F401
    )
    from puripuly_heart.providers.stt import (
        deepgram,  # noqa: F401
        qwen_asr,  # noqa: F401
    )
