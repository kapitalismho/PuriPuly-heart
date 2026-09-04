from pathlib import Path

PROMPT_PATH = Path("prompts/translation_prompt.md")

REQUIRED_PLACEHOLDERS = (
    "${sourceTextRef}",
    "${targetName}",
    "${inputChannel}",
    "${targetLanguageRulesSection}",
    "${translationExamplesSection}",
)


def _prompt_text() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8")


def _prompt_lines() -> list[str]:
    return [line.casefold() for line in _prompt_text().splitlines()]


def test_translation_prompt_declares_all_render_placeholders() -> None:
    text = _prompt_text()

    for placeholder in REQUIRED_PLACEHOLDERS:
        assert placeholder in text


def test_translation_prompt_states_context_ordering_relation() -> None:
    lines = _prompt_lines()
    ordering_lines = [
        line for line in lines if "chronologically" in line and "older" in line and "newer" in line
    ]

    assert ordering_lines, "prompt must state chronological context ordering"
    ordering = ordering_lines[0]
    assert ordering.index("older") < ordering.index(
        "newer"
    ), "context ordering must be older before newer"


def test_translation_prompt_states_self_and_peer_speaker_semantics() -> None:
    lines = _prompt_lines()

    channel_lines = [
        line
        for line in lines
        if "[self]" in line and "[peer]" in line and "channel labels are fixed" in line
    ]
    assert channel_lines, "prompt must define fixed [self] and [peer] channel labels"
    channels = channel_lines[0]
    assert "local-user" in channels
    assert "peer-audio" in channels
    assert "different people" in channels

    input_lines = [
        line
        for line in lines
        if "<input>" in line and "${inputchannel}" in line and "same labels" in line
    ]
    assert input_lines, "prompt must map current input to the fixed channel labels"


def test_translation_prompt_excludes_timestamp_and_competing_legend_semantics() -> None:
    lines = _prompt_lines()

    for line in lines:
        assert "timestamp" not in line
        assert "relative age" not in line
        assert "relative-age" not in line
        assert "ago" not in line
        assert "[others]" not in line
