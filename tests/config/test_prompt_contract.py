from pathlib import Path


def test_translation_prompt_contains_dynamic_policy_contract() -> None:
    text = Path("prompts/translation_prompt.md").read_text(encoding="utf-8")

    assert "${sourceName}" in text
    assert "${targetName}" in text
    assert "${targetLanguageRules}" in text
    assert "${translationExamples}" in text
    assert "Context entries are ordered chronologically from older to newer." in text
    assert "* `[self]` means the local user's earlier utterance." in text
    assert (
        "* `[peer]` means the other speaker from the peer audio channel; "
        "the channel may occasionally include more than one person."
    ) in text
    assert "[others]" not in text


def test_translation_prompt_does_not_reference_timestamps() -> None:
    text = Path("prompts/translation_prompt.md").read_text(encoding="utf-8")

    assert "timestamps" not in text
    assert "timestamp" not in text
    assert "ago" not in text
    assert "relative age" not in text
    assert "Speaker labels, brackets, timestamps" not in text
    assert "Do not copy speaker labels" not in text
    assert "relative-age markers" not in text
    assert "unless they were literally spoken" not in text
    assert "Do not invent facts from metadata" not in text
    assert "Plain-text legend" not in text
    assert "* `[self]` means the local user's earlier utterance." in text
    assert (
        "* `[peer]` means the other speaker from the peer audio channel; "
        "the channel may occasionally include more than one person."
    ) in text
