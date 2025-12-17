def test_providers_import_smoke():
    from ajebal_daera_translator.providers.llm import gemini  # noqa: F401
    from ajebal_daera_translator.providers.stt import google_speech_v2  # noqa: F401

