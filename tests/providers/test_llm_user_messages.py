from puripuly_heart.providers.llm import deepseek, openrouter, qwen, qwen_async
from puripuly_heart.providers.llm.gemini import GoogleGenaiGeminiClient
from puripuly_heart.providers.llm.messages import build_translation_user_message


def test_build_translation_user_message_with_context() -> None:
    assert build_translation_user_message(text="hello", context='- [self] "hi"') == (
        '<context>\n- [self] "hi"\n</context>\n\n' "<input>\nhello\n</input>"
    )


def test_build_translation_user_message_without_context() -> None:
    assert build_translation_user_message(text="hello", context="") == "<input>\nhello\n</input>"


def test_openai_compatible_provider_builders_use_tagged_input() -> None:
    expected = '<context>\n- [self] "hi"\n</context>\n\n<input>\nhello\n</input>'

    assert qwen._build_user_message(text="hello", context='- [self] "hi"') == expected
    assert qwen_async._build_user_message(text="hello", context='- [self] "hi"') == expected
    assert deepseek._build_user_message(text="hello", context='- [self] "hi"') == expected
    assert openrouter._build_user_message(text="hello", context='- [self] "hi"') == expected


def test_gemini_build_request_uses_tagged_input() -> None:
    client = GoogleGenaiGeminiClient(api_key="key", model="model")

    _system_prompt, user_message = client._build_request(
        operation="translate",
        text="hello",
        system_prompt="PROMPT",
        source_language="en",
        target_language="ko",
        context='- [self] "hi"',
    )

    assert user_message == ('<context>\n- [self] "hi"\n</context>\n\n<input>\nhello\n</input>')
