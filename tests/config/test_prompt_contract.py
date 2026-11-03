from pathlib import Path

import pytest

import puripuly_heart.config.prompts as prompts_module
from puripuly_heart.config.prompts import (
    get_prompts_dir,
    get_translation_prompt_template,
    render_translation_prompt_template,
    resolve_system_prompt,
    warm_prompt_cache,
)


@pytest.mark.parametrize("model", [None, "gpt-6-luna", "openai/gpt-6-luna"])
def test_common_prefix_is_independent_of_translation_settings(model: str | None) -> None:
    template = get_translation_prompt_template(model=model)
    requests = [
        render_translation_prompt_template(
            template,
            source_name=source,
            target_name=target,
            input_channel=channel,
            source_specified=specified,
        )
        for source, target, channel, specified in (
            ("Korean", "English", "self", True),
            ("English", "Japanese", "peer", True),
            ("English", "French", "peer", False),
        )
    ]
    prefixes = [request.split("## Translation Settings\n", 1)[0] for request in requests]
    assert prefixes[0] == prefixes[1] == prefixes[2]
    assert requests[0] != requests[1] != requests[2]
    assert all("${" not in request for request in requests)


@pytest.mark.parametrize("model", ["gpt-6-luna", "openai/gpt-6-luna"])
def test_luna_selection_preserves_custom_prompt_and_other_models(model: str) -> None:
    generic = resolve_system_prompt(None)
    luna = resolve_system_prompt(None, model=model)
    assert luna != generic
    assert resolve_system_prompt("  ", model=model) == luna
    custom = "Translate ${sourceName} as a poem.\nKeep this custom layout."
    assert resolve_system_prompt(custom, model=model) == custom
    assert resolve_system_prompt(None, model="other-model") == generic


def test_luna_connections_share_the_same_default_profile() -> None:
    assert resolve_system_prompt(None, model="gpt-6-luna") == resolve_system_prompt(
        None, model="openai/gpt-6-luna"
    )


@pytest.mark.parametrize("model", [None, "gpt-6-luna", "openai/gpt-6-luna"])
@pytest.mark.parametrize(
    ("source", "target", "specified"),
    [("Korean", "English", True), ("English", "French", True), ("English", "Japanese", False)],
)
@pytest.mark.parametrize("kind", ["normal", "preview", "continuation"])
def test_peer_request_kind_selects_rendered_instruction(
    model: str | None, source: str, target: str, specified: bool, kind: str
) -> None:
    template = get_translation_prompt_template(model=model)
    options = dict(
        source_name=source,
        target_name=target,
        input_channel="peer",
        source_specified=specified,
    )
    normal = render_translation_prompt_template(template, **options)
    rendered = render_translation_prompt_template(template, **options, peer_request_kind=kind)
    provisional = (
        (get_prompts_dir() / "peer_provisional_instruction.md").read_text(encoding="utf-8").strip()
    )
    continuation = (
        (get_prompts_dir() / "peer_continuation_instruction.md").read_text(encoding="utf-8").strip()
    )
    expected = normal
    if kind == "preview":
        expected = "".join(
            line for line in normal.splitlines(keepends=True) if not line.startswith("* Ellipsis:")
        )
        expected += f"\n\n{provisional}"
    elif kind == "continuation":
        expected += f"\n\n{continuation}"
    while "\n\n\n" in expected:
        expected = expected.replace("\n\n\n", "\n\n")
    assert rendered == expected
    assert (provisional in rendered) == (kind == "preview")
    assert (continuation in rendered) == (kind == "continuation")
    assert "${" not in rendered
    assert "\n\n\n" not in rendered


@pytest.mark.parametrize("model", [None, "gpt-6-luna", "openai/gpt-6-luna"])
@pytest.mark.parametrize("kind", ["normal", "preview", "continuation"])
def test_peer_request_kind_does_not_change_self_prompt(model: str | None, kind: str) -> None:
    template = get_translation_prompt_template(model=model)
    normal = render_translation_prompt_template(
        template, source_name="English", target_name="Japanese"
    )
    assert (
        render_translation_prompt_template(
            template, source_name="English", target_name="Japanese", peer_request_kind=kind
        )
        == normal
    )


@pytest.mark.parametrize("kind", ["normal", "preview", "continuation"])
def test_custom_templates_opt_in_to_request_specific_sections(kind: str) -> None:
    custom = "Translate ${sourceName} to ${targetName}.\n${unknown}\nKeep my spacing.\n"
    assert (
        render_translation_prompt_template(
            custom,
            source_name="Korean",
            target_name="English",
            input_channel="peer",
            peer_request_kind=kind,
        )
        == "Translate Korean to English.\n${unknown}\nKeep my spacing.\n"
    )


def test_custom_template_conditionals_render_without_depending_on_prose() -> None:
    custom = (
        "Start\n${completeSourceOnly}Custom complete source policy\n"
        "${endCompleteSourceOnly}End${peerProvisionalSection}${peerContinuationSection}"
    )
    normal = render_translation_prompt_template(
        custom, source_name="English", target_name="French", input_channel="peer"
    )
    preview = render_translation_prompt_template(
        custom,
        source_name="English",
        target_name="French",
        input_channel="peer",
        peer_request_kind="preview",
    )
    assert normal == "Start\nCustom complete source policy\nEnd"
    assert preview.startswith("Start\nEnd\n\n")
    assert "Custom complete source policy" not in preview


def test_editable_peer_instructions_follow_prompt_cache_lifecycle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "translation_prompt.md").write_text(
        "Base${peerProvisionalSection}${peerContinuationSection}", encoding="utf-8"
    )
    provisional = tmp_path / "peer_provisional_instruction.md"
    continuation = tmp_path / "peer_continuation_instruction.md"
    provisional.write_text("  Editable preview policy\n", encoding="utf-8")
    continuation.write_text("Editable continuation policy", encoding="utf-8")
    monkeypatch.setenv("PURIPULY_HEART_PROMPTS_DIR", str(tmp_path))
    warm_prompt_cache()
    provisional.write_text("New preview policy", encoding="utf-8")

    def fail_read(_path: Path) -> str:
        raise AssertionError("rendering must use the warmed prompt cache")

    with monkeypatch.context() as warmed:
        warmed.setattr(prompts_module, "_read_prompt_text", fail_read)
        for kind, expected in (
            ("normal", "Base"),
            ("preview", "Base\n\nEditable preview policy"),
            ("continuation", "Base\n\nEditable continuation policy"),
        ):
            assert (
                render_translation_prompt_template(
                    get_translation_prompt_template(),
                    source_name="English",
                    target_name="French",
                    input_channel="peer",
                    peer_request_kind=kind,
                )
                == expected
            )
    prompts_module._reset_prompt_cache_for_tests()
    assert (
        render_translation_prompt_template(
            get_translation_prompt_template(),
            source_name="English",
            target_name="French",
            input_channel="peer",
            peer_request_kind="preview",
        )
        == "Base\n\nNew preview policy"
    )


@pytest.mark.parametrize(
    ("kind", "section", "filename"),
    [
        ("preview", "peerProvisionalSection", "peer_provisional_instruction.md"),
        ("continuation", "peerContinuationSection", "peer_continuation_instruction.md"),
    ],
)
@pytest.mark.parametrize("content", [None, "", " \n\t"])
def test_opted_in_peer_request_requires_nonempty_instruction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
    section: str,
    filename: str,
    content: str | None,
) -> None:
    template = "Base${peerProvisionalSection}${peerContinuationSection}"
    (tmp_path / "translation_prompt.md").write_text(template, encoding="utf-8")
    if content is not None:
        (tmp_path / filename).write_text(content, encoding="utf-8")
    monkeypatch.setenv("PURIPULY_HEART_PROMPTS_DIR", str(tmp_path))
    warm_prompt_cache()
    assert get_translation_prompt_template() == template

    def fail_read(_path: Path) -> str:
        raise AssertionError("instruction validation must use the warmed prompt cache")

    monkeypatch.setattr(prompts_module, "_read_prompt_text", fail_read)
    options = dict(source_name="English", target_name="French")
    assert render_translation_prompt_template(template, **options, input_channel="peer") == "Base"
    assert (
        render_translation_prompt_template(
            template, **options, input_channel="self", peer_request_kind=kind
        )
        == "Base"
    )
    assert (
        render_translation_prompt_template(
            "Legacy ${targetName}",
            **options,
            input_channel="peer",
            peer_request_kind=kind,
        )
        == "Legacy French"
    )
    unrelated_section = (
        "peerContinuationSection"
        if section == "peerProvisionalSection"
        else "peerProvisionalSection"
    )
    assert (
        render_translation_prompt_template(
            f"Base${{{unrelated_section}}}",
            **options,
            input_channel="peer",
            peer_request_kind=kind,
        )
        == "Base"
    )
    error_type = FileNotFoundError if content is None else ValueError
    with pytest.raises(error_type, match=filename.replace(".", r"\.")):
        render_translation_prompt_template(
            template, **options, input_channel="peer", peer_request_kind=kind
        )
