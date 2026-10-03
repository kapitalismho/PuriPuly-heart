import pytest

from puripuly_heart.config.prompts import (
    get_translation_prompt_template,
    render_translation_prompt_template,
    resolve_system_prompt,
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
