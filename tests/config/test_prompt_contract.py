from pathlib import Path

PROMPT_PATH = Path("prompts/translation_prompt.md")

REQUIRED_PLACEHOLDERS = (
    "${sourceName}",
    "${targetName}",
    "${targetLanguageRules}",
    "${translationExamples}",
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

    self_lines = [line for line in lines if "[self]" in line]
    assert self_lines, "prompt must define the [self] legend"
    assert "local user" in self_lines[0]
    assert "earlier" in self_lines[0]

    peer_lines = [line for line in lines if "[peer]" in line]
    assert peer_lines, "prompt must define the [peer] legend"
    assert "peer audio channel" in peer_lines[0]
    assert "more than one person" in peer_lines[0]


def test_translation_prompt_excludes_timestamp_and_competing_legend_semantics() -> None:
    lines = _prompt_lines()

    for line in lines:
        assert "timestamp" not in line
        assert "relative age" not in line
        assert "relative-age" not in line
        assert "ago" not in line
        assert "[others]" not in line
