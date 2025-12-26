def test_providers_import_smoke():
    from ajebal_daera_translator.providers.llm import (
        gemini,  # noqa: F401
        qwen,  # noqa: F401
    )
    from ajebal_daera_translator.providers.stt import (
        deepgram,  # noqa: F401
        qwen_asr,  # noqa: F401
    )
