from puripuly_heart.providers.llm.gemini import GoogleGenaiGeminiClient
from puripuly_heart.providers.llm.messages import build_translation_user_message


def test_build_translation_user_message_with_context() -> None:
    assert build_translation_user_message(text="hello", context='- [self] "hi"') == (
        '<context>\n- [self] "hi"\n</context>\n\n<input>\nhello\n</input>'
    )


def test_build_translation_user_message_without_context() -> None:
    assert build_translation_user_message(text="hello", context="") == "<input>\nhello\n</input>"


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


def test_build_translation_user_message_with_scene_and_context() -> None:
    assert build_translation_user_message(
        text="hello", context='- [self] "hi"', scene_participant_count=2
    ) == (
        '<context>\n- [self] "hi"\n</context>\n\n'
        "<scene>\nPeople: 2\n</scene>\n\n"
        "<input>\nhello\n</input>"
    )


def test_build_translation_user_message_with_scene_without_context() -> None:
    assert (
        build_translation_user_message(text="hello", context="", scene_participant_count=2)
        == "<scene>\nPeople: 2\n</scene>\n\n<input>\nhello\n</input>"
    )


def test_build_translation_user_message_none_is_byte_identical() -> None:
    with_context = build_translation_user_message(
        text="hello", context='- [self] "hi"', scene_participant_count=None
    )
    assert with_context == build_translation_user_message(text="hello", context='- [self] "hi"')
    without_context = build_translation_user_message(
        text="hello", context="", scene_participant_count=None
    )
    assert without_context == build_translation_user_message(text="hello", context="")
    assert without_context == "<input>\nhello\n</input>"


def test_build_translation_user_message_rejects_invalid_scene() -> None:
    baseline = build_translation_user_message(text="hello", context='- [self] "hi"')
    assert (
        build_translation_user_message(
            text="hello", context='- [self] "hi"', scene_participant_count=True
        )
        == baseline
    )
    assert (
        build_translation_user_message(
            text="hello", context='- [self] "hi"', scene_participant_count=False
        )
        == baseline
    )
    assert (
        build_translation_user_message(
            text="hello", context='- [self] "hi"', scene_participant_count=0
        )
        == baseline
    )
    assert (
        build_translation_user_message(
            text="hello", context='- [self] "hi"', scene_participant_count=-1
        )
        == baseline
    )
    assert (
        build_translation_user_message(
            text="hello",
            context='- [self] "hi"',
            scene_participant_count="2",  # type: ignore[arg-type]
        )
        == baseline
    )
    assert (
        build_translation_user_message(
            text="hello",
            context='- [self] "hi"',
            scene_participant_count=2.0,  # type: ignore[arg-type]
        )
        == baseline
    )
    assert "<scene>" not in baseline
