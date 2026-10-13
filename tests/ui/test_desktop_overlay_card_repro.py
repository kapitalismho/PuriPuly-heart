from __future__ import annotations

from puripuly_heart.ui.desktop_overlay_card_repro import caption_card_repro_stages
from puripuly_heart.ui.desktop_overlay_surface.renderer import build_desktop_caption_plan


def test_synthetic_scenario_preserves_distinct_repeated_sources_and_current_card_replay() -> None:
    stages = {stage.name: stage for stage in caption_card_repro_stages()}
    pair = stages["a_b"].snapshot
    appended = stages["a_b_a_accumulation"]
    replay = stages["replacement_replay"].snapshot
    before = {block.id: block for block in pair.blocks}
    after = {block.id: block for block in appended.snapshot.blocks}
    assert appended.source_units == ("source:a1", "source:b1", "source:a2")
    assert after["card:a"].primary_text == "Again. Again."
    assert after["card:a"].secondary_text == "Encore. Encore."
    assert after["card:a"].card_revision > before["card:a"].card_revision
    assert after["card:a"].occupant_key == before["card:a"].occupant_key
    assert after["card:b"] == before["card:b"]
    assert {block.id: block for block in replay.blocks} == after
    assert [slot.block_id for slot in build_desktop_caption_plan(replay).slots] == [
        "card:b",
        "card:a",
    ]


def test_synthetic_scenario_exercises_renderer_style_divider_handoff_and_removal() -> None:
    stages = {stage.name: stage.snapshot for stage in caption_card_repro_stages()}
    before = stages["replacement_replay"].blocks[1]
    after = stages["same_text_style_redraw"].blocks[1]
    assert before.primary_text == after.primary_text
    assert before.secondary_text == after.secondary_text
    assert before.speaker_style != after.speaker_style
    handoff = build_desktop_caption_plan(stages["epoch_handoff_gray"])
    assert {line.color for line in handoff.slots[0].lines} == {"#FFD700"}
    assert {line.color for line in handoff.slots[1].lines} == {"#B4B4B4"}
    assert build_desktop_caption_plan(stages["established_overflow_divider"]).speaker_divider
    assert not build_desktop_caption_plan(stages["unknown_gray_no_divider"]).speaker_divider
    assert {line.color for line in build_desktop_caption_plan(stages["self_white"]).lines} == {
        "#FFFFFF"
    }
    assert not stages["python_owned_removal"].blocks
