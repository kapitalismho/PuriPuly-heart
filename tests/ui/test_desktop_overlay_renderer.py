from __future__ import annotations

import asyncio
import builtins
import contextlib
import inspect
import json
import logging
import os
import subprocess
import sys
import traceback
from dataclasses import replace
from pathlib import Path
from typing import Any

import flet as ft
import pytest
import websockets

from puripuly_heart.core.overlay.bridge import OverlayBridge
from puripuly_heart.core.overlay.manifest import OVERLAY_CONTRACT_VERSION, OverlayLaunchManifest
from puripuly_heart.core.overlay.protocol import (
    OverlayPresentationBlock,
    OverlayPresentationCalibration,
    OverlayPresentationSnapshot,
)
from puripuly_heart.ui import desktop_overlay, desktop_window_zorder, flet_desktop_runtime
from puripuly_heart.ui.desktop_overlay_surface.contract import (
    _DESKTOP_CAPTION_GOLD,
    _DESKTOP_CAPTION_LINE_HEIGHT,
    _DESKTOP_CAPTION_MAX_VISIBLE_LINES,
    _DESKTOP_CAPTION_MAX_VISIBLE_SLOTS,
    _DESKTOP_CAPTION_MIN_DYNAMIC_CARD_WIDTH,
    _DESKTOP_CAPTION_PRIMARY_REGION_ALIGNMENT_Y,
    _DESKTOP_CAPTION_SIZE_PRESETS,
    _DESKTOP_CAPTION_TEXT_STACK_ALIGNMENT_Y,
    _DESKTOP_CAPTION_WHITE,
    _DESKTOP_PREVIEW_BACKGROUND_ALPHA_PRESETS,
)
from puripuly_heart.ui.fonts import (
    FONT_FAMILY_NOTO_SANS,
    FONT_FAMILY_NOTO_SANS_CJK_JP,
    assets_dir,
)


def _manifest(**overrides: object) -> OverlayLaunchManifest:
    values: dict[str, object] = {
        "contract_version": OVERLAY_CONTRACT_VERSION,
        "app_version": "test",
        "overlay_instance_id": "desktop-overlay-test",
        "bridge_url": "ws://127.0.0.1:8765",
        "session_token": "test-session-token",
        "parent_pid": 1234,
        "startup_deadline_ms": 1000,
        "log_dir": "logs",
        "log_level": "INFO",
        "locale": "en",
        "logging_mode": "basic",
    }
    values.update(overrides)
    return OverlayLaunchManifest(**values)  # type: ignore[arg-type]


def _write_manifest(tmp_path: Path, manifest: OverlayLaunchManifest) -> Path:
    path = tmp_path / "overlay-manifest.json"
    path.write_text(json.dumps(manifest.to_dict()), encoding="utf-8")
    return path


def _block(
    block_id: str,
    *,
    channel: str,
    block_variant: str,
    appearance_seq: int,
    primary_text: str,
    secondary_text: str = "",
    secondary_enabled: bool = False,
    primary_language: str | None = None,
    secondary_language: str | None = None,
) -> OverlayPresentationBlock:
    return OverlayPresentationBlock(
        id=block_id,
        occupant_key=f"{channel}:{block_id}",
        appearance_seq=appearance_seq,
        channel=channel,  # type: ignore[arg-type]
        block_variant=block_variant,  # type: ignore[arg-type]
        primary_text=primary_text,
        secondary_text=secondary_text,
        secondary_enabled=secondary_enabled,
        primary_language=primary_language,
        secondary_language=secondary_language,
    )


def test_desktop_overlay_snapshot_mapping_table_documents_current_block_contract() -> None:
    rows = {
        (row.snapshot_field, row.block_type, row.slot): row
        for row in desktop_overlay.DESKTOP_CAPTION_MAPPING_TABLE
    }

    assert rows[("blocks[]", "active_self/self", "primary")].role == "active_self_source"
    assert rows[("blocks[]", "active_self/self", "primary")].color == _DESKTOP_CAPTION_WHITE
    assert rows[("blocks[]", "active_self/self", "secondary")].role == "active_self_translation"
    assert rows[("blocks[]", "active_self/self", "secondary")].color == _DESKTOP_CAPTION_WHITE
    assert rows[("blocks[]", "active_self/self", "secondary")].truncation.startswith("max 1 line")
    active_peer_row = rows[("blocks[]", "active_peer/peer", "primary")]
    assert active_peer_row.role == "active_peer_source"
    assert active_peer_row.promoted is True
    assert active_peer_row.color == _DESKTOP_CAPTION_GOLD
    assert rows[("blocks[]", "finalized/peer translated", "primary")].role == ("peer_translation")
    assert rows[("blocks[]", "finalized/peer translated", "primary")].color == _DESKTOP_CAPTION_GOLD
    assert (
        rows[("blocks[]", "finalized/peer translated", "secondary")].color == _DESKTOP_CAPTION_GOLD
    )
    assert rows[("blocks[]", "finalized/peer translated", "secondary")].truncation.startswith(
        "max 1 line"
    )
    peer_source_only_row = rows[("blocks[]", "finalized/peer source-only", "primary")]
    assert peer_source_only_row.promoted is True
    assert peer_source_only_row.color == _DESKTOP_CAPTION_GOLD
    assert peer_source_only_row.truncation == (
        "max 2 lines; drops before active and translated primary lines"
    )
    assert rows[("blocks[]", "finalized/self", "secondary")].role == "self_translation"
    assert rows[("blocks[]", "finalized/self", "secondary")].color == _DESKTOP_CAPTION_WHITE
    assert rows[("blocks[]", "finalized/self", "secondary")].truncation.startswith("max 1 line")
    self_secondary_only_row = rows[("blocks[]", "finalized/self secondary-only", "primary")]
    assert self_secondary_only_row.role == "self_translation"
    assert self_secondary_only_row.promoted is True
    assert self_secondary_only_row.color == _DESKTOP_CAPTION_WHITE
    assert rows[("calibration", "all", "none")].role == "desktop_visual_ignored"
    assert rows[("blocks[]", "none/edit", "none")].role == "edit_no_caption_empty_card"
    assert rows[("blocks[]", "none/edit", "none")].truncation == (
        "renders empty caption card with centered lock text action"
    )
    assert rows[("blocks[]", "none/pass_through", "none")].truncation == (
        "renders no text and no background"
    )


def test_desktop_overlay_snapshot_mapping_table_matches_emitted_caption_lines() -> None:
    row_by_block_and_role = {
        (row.block_type, row.role): row for row in desktop_overlay.DESKTOP_CAPTION_MAPPING_TABLE
    }
    cases = [
        (
            "active_self/self",
            _block(
                "self-active",
                channel="self",
                block_variant="active_self",
                appearance_seq=10,
                primary_text="active self source",
                secondary_text="active self translation",
                secondary_enabled=True,
            ),
            ("active self source", "active self translation"),
        ),
        (
            "active_peer/peer",
            _block(
                "peer-active",
                channel="peer",
                block_variant="active_peer",
                appearance_seq=20,
                primary_text="",
                secondary_text="active peer source",
                secondary_enabled=True,
            ),
            ("active peer source",),
        ),
        (
            "finalized/peer translated",
            _block(
                "peer-translated",
                channel="peer",
                block_variant="finalized",
                appearance_seq=30,
                primary_text="peer translation",
                secondary_text="peer original",
                secondary_enabled=True,
            ),
            ("peer translation", "peer original"),
        ),
        (
            "finalized/peer source-only",
            _block(
                "peer-source-only",
                channel="peer",
                block_variant="finalized",
                appearance_seq=40,
                primary_text="",
                secondary_text="peer source only",
                secondary_enabled=True,
            ),
            ("peer source only",),
        ),
        (
            "finalized/self",
            _block(
                "self-finalized",
                channel="self",
                block_variant="finalized",
                appearance_seq=50,
                primary_text="self source",
                secondary_text="self translation",
                secondary_enabled=True,
            ),
            ("self source", "self translation"),
        ),
        (
            "finalized/self secondary-only",
            _block(
                "self-secondary-only",
                channel="self",
                block_variant="finalized",
                appearance_seq=60,
                primary_text="",
                secondary_text="self translation only",
                secondary_enabled=True,
            ),
            ("self translation only",),
        ),
    ]

    for block_type, block, expected_texts in cases:
        plan = desktop_overlay.build_desktop_caption_plan(
            OverlayPresentationSnapshot(blocks=[block])
        )
        assert tuple(line.text for line in plan.lines) == expected_texts
        for line in plan.lines:
            row = row_by_block_and_role[(block_type, line.role)]
            assert row.slot == line.slot, (block_type, line.role)
            assert row.promoted is line.promoted, (block_type, line.role)
            assert row.color == line.color, (block_type, line.role)
            assert int(row.priority.split(maxsplit=1)[0]) == line.priority, (
                block_type,
                line.role,
            )


def test_desktop_overlay_snapshot_mapping_roles_secondary_promotion_and_channel_colors() -> None:
    active_self_plan = desktop_overlay.build_desktop_caption_plan(
        OverlayPresentationSnapshot(
            revision=12,
            blocks=[
                _block(
                    "self-active",
                    channel="self",
                    block_variant="active_self",
                    appearance_seq=10,
                    primary_text="I can hear you",
                    secondary_text="들려요",
                    secondary_enabled=True,
                )
            ],
        )
    )
    peer_translated_plan = desktop_overlay.build_desktop_caption_plan(
        OverlayPresentationSnapshot(
            revision=13,
            blocks=[
                _block(
                    "peer-translated",
                    channel="peer",
                    block_variant="finalized",
                    appearance_seq=11,
                    primary_text="좋아요",
                    secondary_text="Sounds good",
                    secondary_enabled=True,
                )
            ],
        )
    )
    active_peer_plan = desktop_overlay.build_desktop_caption_plan(
        OverlayPresentationSnapshot(
            revision=14,
            blocks=[
                _block(
                    "peer-active",
                    channel="peer",
                    block_variant="active_peer",
                    appearance_seq=12,
                    primary_text="",
                    secondary_text="typing live source",
                    secondary_enabled=True,
                )
            ],
        )
    )

    line_by_text = {
        line.text: line
        for plan in (active_self_plan, peer_translated_plan, active_peer_plan)
        for line in plan.lines
    }
    assert line_by_text["I can hear you"].role == "active_self_source"
    assert line_by_text["I can hear you"].slot == "primary"
    assert line_by_text["I can hear you"].color == _DESKTOP_CAPTION_WHITE
    assert line_by_text["들려요"].role == "active_self_translation"
    assert line_by_text["들려요"].slot == "secondary"
    assert line_by_text["들려요"].color == _DESKTOP_CAPTION_WHITE
    assert line_by_text["typing live source"].role == "active_peer_source"
    assert line_by_text["typing live source"].slot == "primary"
    assert line_by_text["typing live source"].promoted is True
    assert line_by_text["typing live source"].color == _DESKTOP_CAPTION_GOLD
    assert line_by_text["좋아요"].role == "peer_translation"
    assert line_by_text["좋아요"].color == _DESKTOP_CAPTION_GOLD
    assert line_by_text["Sounds good"].role == "peer_source_original"
    assert line_by_text["Sounds good"].slot == "secondary"
    assert line_by_text["Sounds good"].color == _DESKTOP_CAPTION_GOLD
    for plan in (active_self_plan, peer_translated_plan, active_peer_plan):
        assert sum(line.max_lines for line in plan.lines) <= 3


@pytest.mark.parametrize(
    "block",
    [
        _block(
            "active-peer-disabled-secondary",
            channel="peer",
            block_variant="active_peer",
            appearance_seq=1,
            primary_text="",
            secondary_text="disabled active peer source",
            secondary_enabled=False,
        ),
        _block(
            "peer-source-only-disabled-secondary",
            channel="peer",
            block_variant="finalized",
            appearance_seq=2,
            primary_text="",
            secondary_text="disabled peer source only",
            secondary_enabled=False,
        ),
        _block(
            "self-secondary-only-disabled-secondary",
            channel="self",
            block_variant="finalized",
            appearance_seq=3,
            primary_text="",
            secondary_text="disabled self translation only",
            secondary_enabled=False,
        ),
    ],
)
def test_desktop_overlay_caption_rendering_disabled_secondary_only_blocks_do_not_promote(
    block: OverlayPresentationBlock,
) -> None:
    plan = desktop_overlay.build_desktop_caption_plan(OverlayPresentationSnapshot(blocks=[block]))

    assert plan.lines == ()
    assert plan.surface_visible is False


def test_desktop_overlay_caption_rendering_no_caption_states_use_empty_moving_card_and_transparent_locked() -> (
    None
):
    empty_snapshot = OverlayPresentationSnapshot(revision=2, blocks=[])

    edit_plan = desktop_overlay.build_desktop_caption_plan(
        empty_snapshot,
        interaction_mode="edit",
        locale="ja",
    )
    locked_plan = desktop_overlay.build_desktop_caption_plan(
        empty_snapshot,
        interaction_mode="pass_through",
        locale="ja",
    )

    assert edit_plan.lines == ()
    assert edit_plan.surface_visible is True
    assert edit_plan.background_alpha == pytest.approx(0.6)
    assert edit_plan.background_color == "#99000000"
    assert locked_plan.lines == ()
    assert locked_plan.surface_visible is False
    assert locked_plan.background_alpha == 0
    assert locked_plan.background_color == "transparent"


def test_desktop_overlay_visual_config_uses_preset_tokens_and_no_outline_text() -> None:
    snapshot = OverlayPresentationSnapshot(
        blocks=[
            _block(
                "peer-translated",
                channel="peer",
                block_variant="finalized",
                appearance_seq=1,
                primary_text="안녕하세요 👋",
                secondary_text="Hello there 👋",
                secondary_enabled=True,
            )
        ]
    )
    plan = desktop_overlay.build_desktop_caption_plan(
        snapshot,
        window_width=1344,
        window_height=336,
        visual_state=desktop_overlay.DesktopCaptionVisualState(
            text_scale=1.0,
            background_alpha=0.38,
            outline_width=None,
        ),
        locale="ko",
    )

    assert plan.primary_font_size == _DESKTOP_CAPTION_SIZE_PRESETS["medium"].primary_font_size
    assert plan.secondary_font_size == _DESKTOP_CAPTION_SIZE_PRESETS["medium"].secondary_font_size
    assert plan.outline_width == 0
    assert plan.background_color == "#61000000"
    assert plan.padding_horizontal == _DESKTOP_CAPTION_SIZE_PRESETS["medium"].padding_horizontal
    assert plan.padding_vertical == _DESKTOP_CAPTION_SIZE_PRESETS["medium"].padding_vertical
    assert plan.text_width == 1300
    assert plan.border_radius == _DESKTOP_CAPTION_SIZE_PRESETS["medium"].border_radius

    surface = desktop_overlay.build_desktop_caption_surface(plan)
    assert surface.bgcolor == ft.Colors.TRANSPARENT
    assert surface.border_radius == _DESKTOP_CAPTION_SIZE_PRESETS["medium"].border_radius
    assert isinstance(surface.content, ft.Stack)
    assert len(surface.content.controls) == 1
    slot_column = surface.content.controls[0]
    outer_slot = slot_column.controls[0]
    inner_card = outer_slot.content
    assert outer_slot.bgcolor == ft.Colors.TRANSPARENT
    assert inner_card.bgcolor == "#61000000"
    assert inner_card.border_radius == _DESKTOP_CAPTION_SIZE_PRESETS["medium"].border_radius
    assert inner_card.padding.left == 22
    assert inner_card.padding.top == 10
    text_layer = inner_card.content
    assert text_layer.bgcolor == ft.Colors.TRANSPARENT
    assert text_layer.width == pytest.approx(plan.slots[0].card_text_width)
    column = text_layer.content
    assert column.scroll is None
    assert all(isinstance(control.content, ft.Text) for control in column.controls)
    assert not any(
        getattr(control, "bgcolor", None) in {"#C3CEDA", "#D2A24F"}
        for control in _walk_control_tree(inner_card)
    )
    first_text = column.controls[0].content
    assert first_text.color == _DESKTOP_CAPTION_GOLD
    assert first_text.text_align == ft.TextAlign.CENTER
    assert first_text.overflow == ft.TextOverflow.ELLIPSIS
    assert first_text.style.height == pytest.approx(_DESKTOP_CAPTION_LINE_HEIGHT)
    assert first_text.style.foreground is None


def test_desktop_overlay_locked_slot_uses_dynamic_inner_card_width_for_short_text() -> None:
    snapshot = OverlayPresentationSnapshot(
        blocks=[
            _block(
                "short-peer",
                channel="peer",
                block_variant="finalized",
                appearance_seq=1,
                primary_text="응",
            )
        ]
    )

    plan = desktop_overlay.build_desktop_caption_plan(
        snapshot,
        window_width=1344,
        window_height=336,
        visual_state=desktop_overlay.DesktopCaptionVisualState(background_alpha=0.5),
        interaction_mode="pass_through",
    )

    slot = plan.slots[0]
    assert slot.card_width < plan.window_width
    assert slot.card_width == pytest.approx(_DESKTOP_CAPTION_MIN_DYNAMIC_CARD_WIDTH)
    assert slot.card_text_width == pytest.approx(slot.card_width - (plan.padding_horizontal * 2))

    surface = desktop_overlay.build_desktop_caption_surface(plan)
    slot_column = surface.content.controls[0]
    outer_slot = slot_column.controls[0]
    assert outer_slot.width == 1344
    assert outer_slot.height == pytest.approx(plan.slot_height)
    assert outer_slot.bgcolor == ft.Colors.TRANSPARENT

    inner_card = outer_slot.content
    assert inner_card.width == pytest.approx(slot.card_width)
    assert inner_card.bgcolor == "#80000000"
    assert inner_card.border_radius == _DESKTOP_CAPTION_SIZE_PRESETS["medium"].border_radius
    assert inner_card.padding.left == 22
    assert inner_card.padding.top == 10

    text_layer = inner_card.content
    assert text_layer.width == pytest.approx(slot.card_text_width)
    first_line_region = text_layer.content.controls[0]
    assert first_line_region.width == pytest.approx(slot.card_text_width)
    assert first_line_region.content.width == pytest.approx(slot.card_text_width)


def test_desktop_overlay_modest_latin_width_estimate_keeps_medium_caption_tight() -> None:
    snapshot = OverlayPresentationSnapshot(
        blocks=[
            _block(
                "latin-peer",
                channel="peer",
                block_variant="finalized",
                appearance_seq=1,
                primary_text="talk like real friends",
            )
        ]
    )

    plan = desktop_overlay.build_desktop_caption_plan(
        snapshot,
        window_width=1344,
        window_height=336,
        interaction_mode="pass_through",
    )

    slot = plan.slots[0]
    assert 520.0 <= slot.card_width <= 580.0
    assert slot.card_text_width == pytest.approx(slot.card_width - (plan.padding_horizontal * 2))


def test_desktop_overlay_edit_mode_caption_text_keeps_full_preset_width() -> None:
    snapshot = OverlayPresentationSnapshot(
        blocks=[
            _block(
                "edit-peer",
                channel="peer",
                block_variant="finalized",
                appearance_seq=1,
                primary_text="응",
            )
        ]
    )
    edit_plan = desktop_overlay.build_desktop_caption_plan(
        snapshot,
        window_width=1344,
        window_height=336,
        interaction_mode="edit",
    )

    edit_surface = desktop_overlay.build_desktop_caption_surface(edit_plan)
    full_background = edit_surface.content.controls[0]
    slot_column = edit_surface.content.controls[-1]
    outer_slot = slot_column.controls[0]
    inner_card = outer_slot.content
    text_layer = inner_card.content

    assert full_background.bgcolor == edit_plan.background_color
    assert outer_slot.width == edit_plan.window_width
    assert outer_slot.bgcolor == ft.Colors.TRANSPARENT
    assert inner_card.width == edit_plan.window_width
    assert inner_card.bgcolor == ft.Colors.TRANSPARENT
    assert text_layer.width == edit_plan.text_width


def test_desktop_overlay_dynamic_inner_card_width_clamps_to_narrow_window() -> None:
    snapshot = OverlayPresentationSnapshot(
        blocks=[
            _block(
                "narrow-peer",
                channel="peer",
                block_variant="finalized",
                appearance_seq=1,
                primary_text="narrow window caption",
            )
        ]
    )

    plan = desktop_overlay.build_desktop_caption_plan(
        snapshot,
        window_width=240,
        window_height=160,
        interaction_mode="pass_through",
    )

    slot = plan.slots[0]
    assert slot.card_width == pytest.approx(plan.window_width)
    assert slot.card_text_width == pytest.approx(
        max(1.0, plan.window_width - (plan.padding_horizontal * 2))
    )

    surface = desktop_overlay.build_desktop_caption_surface(plan)
    outer_slot = surface.content.controls[0].controls[0]
    inner_card = outer_slot.content
    text_layer = inner_card.content
    first_line_region = text_layer.content.controls[0]
    assert outer_slot.width == plan.window_width
    assert outer_slot.bgcolor == ft.Colors.TRANSPARENT
    assert inner_card.width == pytest.approx(plan.window_width)
    assert text_layer.width == pytest.approx(slot.card_text_width)
    assert first_line_region.width == pytest.approx(slot.card_text_width)


def test_desktop_overlay_caption_text_has_no_shadow_or_stroke() -> None:
    snapshot = OverlayPresentationSnapshot(
        blocks=[
            _block(
                "peer-translated",
                channel="peer",
                block_variant="finalized",
                appearance_seq=1,
                primary_text="오늘은 천천히 말해줘서 고마워요",
                secondary_text="Thanks for speaking slowly today.",
                secondary_enabled=True,
            )
        ]
    )
    plan = desktop_overlay.build_desktop_caption_plan(snapshot)

    surface = desktop_overlay.build_desktop_caption_surface(plan)
    slot_column = surface.content.controls[0]
    outer_slot = slot_column.controls[0]
    inner_card = outer_slot.content
    text_layer = inner_card.content
    text_column = text_layer.content

    primary_text = text_column.controls[0].content
    secondary_text = text_column.controls[1].content
    for text in (primary_text, secondary_text):
        assert text.style.foreground is None
        assert text.style.shadow is None


def test_desktop_overlay_uses_fixed_two_turn_slots_with_secondary_one_line() -> None:
    snapshot = OverlayPresentationSnapshot(
        blocks=[
            _block(
                "self-finalized",
                channel="self",
                block_variant="finalized",
                appearance_seq=1,
                primary_text="self source line",
                secondary_text="self translation line",
                secondary_enabled=True,
            ),
            _block(
                "peer-translated",
                channel="peer",
                block_variant="finalized",
                appearance_seq=2,
                primary_text="peer translated line",
                secondary_text="peer original line",
                secondary_enabled=True,
            ),
            _block(
                "overflow-extra",
                channel="self",
                block_variant="finalized",
                appearance_seq=3,
                primary_text="must not create a third desktop slot",
            ),
        ]
    )

    plan = desktop_overlay.build_desktop_caption_plan(
        snapshot, window_width=1344, window_height=336
    )

    assert plan.max_visible_slots == _DESKTOP_CAPTION_MAX_VISIBLE_SLOTS
    assert len(plan.slots) == 2
    assert [slot.block_id for slot in plan.slots] == ["self-finalized", "peer-translated"]
    assert [slot.channel for slot in plan.slots] == ["self", "peer"]
    assert [(line.text, line.max_lines) for line in plan.lines] == [
        ("self source line", 2),
        ("self translation line", 1),
        ("peer translated line", 2),
        ("peer original line", 1),
    ]
    assert plan.overflow_strategy == (
        "two-turn-slots:presenter-selected-blocks,primary-two-lines,secondary-one-line"
    )


def test_desktop_overlay_slots_reserve_stable_line_regions() -> None:
    snapshot = OverlayPresentationSnapshot(
        blocks=[
            _block(
                "self-active",
                channel="self",
                block_variant="active_self",
                appearance_seq=1,
                primary_text="말하는 중인 원문",
                secondary_text="",
                secondary_enabled=False,
            ),
            _block(
                "peer-finalized",
                channel="peer",
                block_variant="finalized",
                appearance_seq=2,
                primary_text="번역된 상대 발화",
                secondary_text="original peer utterance",
                secondary_enabled=True,
            ),
        ]
    )

    plan = desktop_overlay.build_desktop_caption_plan(
        snapshot,
        window_width=1344,
        window_height=336,
        visual_state=desktop_overlay.DesktopCaptionVisualState(background_alpha=0.38),
    )

    expected_slot_height = (336 - 10) / 2
    expected_primary_height = 41 * 1.24 * 2
    expected_secondary_height = 25 * 1.24
    assert plan.slot_height == pytest.approx(expected_slot_height)
    assert plan.primary_region_height == pytest.approx(expected_primary_height)
    assert plan.secondary_region_height == pytest.approx(expected_secondary_height)
    assert (
        plan.slot_height
        - (plan.padding_vertical * 2)
        - plan.primary_region_height
        - plan.secondary_region_height
    ) >= 5

    surface = desktop_overlay.build_desktop_caption_surface(plan)
    slot_column = surface.content.controls[0]
    assert slot_column.spacing == 10
    assert slot_column.height == pytest.approx((expected_slot_height * 2) + 10)

    first_slot, second_slot = slot_column.controls
    assert first_slot.height == pytest.approx(expected_slot_height)
    assert second_slot.height == pytest.approx(expected_slot_height)
    first_inner_card = first_slot.content
    second_inner_card = second_slot.content
    first_text_layer = first_inner_card.content
    second_text_layer = second_inner_card.content
    assert first_inner_card.alignment == ft.Alignment.CENTER
    assert first_text_layer.alignment == ft.Alignment.CENTER
    assert second_text_layer.alignment.y == pytest.approx(_DESKTOP_CAPTION_TEXT_STACK_ALIGNMENT_Y)

    first_column = first_text_layer.content
    (first_primary_region,) = first_column.controls
    assert first_primary_region.height == pytest.approx(expected_primary_height)
    assert first_primary_region.alignment == ft.Alignment.CENTER

    second_column = second_text_layer.content
    second_primary_region, second_secondary_region = second_column.controls
    assert second_primary_region.height == pytest.approx(expected_primary_height)
    assert second_primary_region.alignment.y == pytest.approx(
        _DESKTOP_CAPTION_PRIMARY_REGION_ALIGNMENT_Y
    )
    assert second_secondary_region.height == pytest.approx(expected_secondary_height)
    assert second_secondary_region.alignment == ft.Alignment.CENTER
    assert second_secondary_region.content.value == "original peer utterance"


def test_desktop_overlay_secondary_disabled_slots_do_not_reserve_secondary_region() -> None:
    snapshot = OverlayPresentationSnapshot(
        blocks=[
            _block(
                "disabled-secondary",
                channel="self",
                block_variant="active_self",
                appearance_seq=1,
                primary_text="말하는 중인 원문",
                secondary_text="disabled translation",
                secondary_enabled=False,
            )
        ]
    )

    plan = desktop_overlay.build_desktop_caption_plan(
        snapshot,
        window_width=1344,
        window_height=336,
        visual_state=desktop_overlay.DesktopCaptionVisualState(background_alpha=0.38),
    )

    assert [(line.text, line.slot) for line in plan.lines] == [("말하는 중인 원문", "primary")]

    surface = desktop_overlay.build_desktop_caption_surface(plan)
    slot_column = surface.content.controls[0]
    outer_slot = slot_column.controls[0]
    inner_card = outer_slot.content
    text_layer = inner_card.content
    assert inner_card.alignment == ft.Alignment.CENTER
    assert text_layer.alignment == ft.Alignment.CENTER
    text_column = text_layer.content
    (primary_region,) = text_column.controls
    assert primary_region.alignment == ft.Alignment.CENTER
    assert primary_region.content.value == "말하는 중인 원문"


def test_desktop_overlay_secondary_enabled_empty_slots_reserve_secondary_region() -> None:
    snapshot = OverlayPresentationSnapshot(
        blocks=[
            _block(
                "pending-secondary",
                channel="self",
                block_variant="active_self",
                appearance_seq=1,
                primary_text="말하는 중인 원문",
                secondary_text="",
                secondary_enabled=True,
            )
        ]
    )

    plan = desktop_overlay.build_desktop_caption_plan(
        snapshot,
        window_width=1344,
        window_height=336,
        visual_state=desktop_overlay.DesktopCaptionVisualState(background_alpha=0.38),
    )

    assert [(line.text, line.slot) for line in plan.lines] == [("말하는 중인 원문", "primary")]

    surface = desktop_overlay.build_desktop_caption_surface(plan)
    slot_column = surface.content.controls[0]
    outer_slot = slot_column.controls[0]
    inner_card = outer_slot.content
    text_layer = inner_card.content
    assert inner_card.alignment == ft.Alignment.CENTER
    assert text_layer.alignment.y == pytest.approx(_DESKTOP_CAPTION_TEXT_STACK_ALIGNMENT_Y)
    text_column = text_layer.content
    primary_region, reserved_secondary_region = text_column.controls
    assert primary_region.alignment.y == pytest.approx(_DESKTOP_CAPTION_PRIMARY_REGION_ALIGNMENT_Y)
    assert primary_region.content.value == "말하는 중인 원문"
    assert reserved_secondary_region.height == pytest.approx(plan.secondary_region_height)
    assert reserved_secondary_region.alignment == ft.Alignment.CENTER
    assert reserved_secondary_region.content.value == ""
    assert reserved_secondary_region.content.max_lines == 1


def test_desktop_overlay_secondary_only_promoted_lines_do_not_reserve_secondary_region() -> None:
    snapshot = OverlayPresentationSnapshot(
        blocks=[
            _block(
                "peer-secondary-only",
                channel="peer",
                block_variant="finalized",
                appearance_seq=1,
                primary_text="",
                secondary_text="peer source only",
                secondary_enabled=True,
            ),
            _block(
                "self-secondary-only",
                channel="self",
                block_variant="finalized",
                appearance_seq=2,
                primary_text="",
                secondary_text="self translation only",
                secondary_enabled=True,
            ),
        ]
    )

    plan = desktop_overlay.build_desktop_caption_plan(
        snapshot,
        window_width=1344,
        window_height=336,
        visual_state=desktop_overlay.DesktopCaptionVisualState(background_alpha=0.38),
    )

    assert [(line.text, line.slot, line.promoted) for line in plan.lines] == [
        ("peer source only", "primary", True),
        ("self translation only", "primary", True),
    ]

    surface = desktop_overlay.build_desktop_caption_surface(plan)
    slot_column = surface.content.controls[0]
    assert len(slot_column.controls) == 2
    for outer_slot, expected_text in zip(
        slot_column.controls,
        ("peer source only", "self translation only"),
        strict=True,
    ):
        inner_card = outer_slot.content
        text_layer = inner_card.content
        assert inner_card.alignment == ft.Alignment.CENTER
        assert text_layer.alignment == ft.Alignment.CENTER
        text_column = text_layer.content
        (primary_region,) = text_column.controls
        assert primary_region.alignment == ft.Alignment.CENTER
        assert primary_region.content.value == expected_text


def test_desktop_overlay_edit_mode_background_covers_full_window() -> None:
    plan = desktop_overlay.build_desktop_caption_plan(
        OverlayPresentationSnapshot(blocks=[]),
        window_width=1344,
        window_height=336,
        interaction_mode="edit",
        visual_state=desktop_overlay.DesktopCaptionVisualState(background_alpha=0.5),
    )

    surface = desktop_overlay.build_desktop_caption_surface(plan)

    assert surface.width == 1344
    assert surface.height == 336
    assert isinstance(surface.content, ft.Stack)
    full_background = surface.content.controls[0]
    assert isinstance(full_background, ft.Container)
    assert full_background.bgcolor == "#80000000"
    assert full_background.border_radius == _DESKTOP_CAPTION_SIZE_PRESETS["medium"].border_radius
    assert (
        full_background.left,
        full_background.top,
        full_background.right,
        full_background.bottom,
    ) == (0, 0, 0, 0)


def test_desktop_overlay_active_captions_do_not_receive_alpha_bonus() -> None:
    snapshot = OverlayPresentationSnapshot(
        blocks=[
            _block(
                "self-active",
                channel="self",
                block_variant="active_self",
                appearance_seq=1,
                primary_text="active source",
            )
        ]
    )

    plan = desktop_overlay.build_desktop_caption_plan(
        snapshot,
        window_width=1344,
        window_height=336,
        visual_state=desktop_overlay.DesktopCaptionVisualState(background_alpha=0.5),
        interaction_mode="edit",
    )

    assert any(line.active for line in plan.lines)
    assert plan.background_alpha == pytest.approx(0.5)
    assert plan.background_color == "#80000000"


def test_desktop_overlay_caps_to_presenter_selected_two_turn_slots_without_reprioritizing() -> None:
    snapshot = OverlayPresentationSnapshot(
        blocks=[
            _block(
                "old-self",
                channel="self",
                block_variant="finalized",
                appearance_seq=1,
                primary_text="older finalized self source",
                secondary_text="older finalized self translation",
                secondary_enabled=True,
            ),
            _block(
                "peer-translated",
                channel="peer",
                block_variant="finalized",
                appearance_seq=2,
                primary_text="newer translated peer result",
                secondary_text="newer peer original",
                secondary_enabled=True,
            ),
            _block(
                "peer-active",
                channel="peer",
                block_variant="active_peer",
                appearance_seq=3,
                primary_text="",
                secondary_text="newest active peer source",
                secondary_enabled=True,
            ),
        ],
    )

    plan = desktop_overlay.build_desktop_caption_plan(snapshot)

    visible_text = [line.text for line in plan.lines]
    assert visible_text == [
        "older finalized self source",
        "older finalized self translation",
        "newer translated peer result",
        "newer peer original",
    ]
    assert [slot.block_id for slot in plan.slots] == ["old-self", "peer-translated"]
    assert sum(line.max_lines for line in plan.lines) == _DESKTOP_CAPTION_MAX_VISIBLE_LINES
    assert "newest active peer source" not in visible_text
    assert plan.overflow_strategy == (
        "two-turn-slots:presenter-selected-blocks,primary-two-lines,secondary-one-line"
    )


def test_desktop_overlay_caption_rendering_preserves_cjk_emoji_and_minimum_secondary_size() -> None:
    snapshot = OverlayPresentationSnapshot(
        blocks=[
            _block(
                "mixed-script",
                channel="peer",
                block_variant="finalized",
                appearance_seq=1,
                primary_text="今日は PuriPuly Heart 좋아요 😊",
                secondary_text="今日は mixed 원문 😊",
                secondary_enabled=True,
            )
        ]
    )

    plan = desktop_overlay.build_desktop_caption_plan(
        snapshot,
        window_width=1152,
        window_height=288,
        locale="zh-CN",
    )

    assert [line.text for line in plan.lines] == [
        "今日は PuriPuly Heart 좋아요 😊",
        "今日は mixed 원문 😊",
    ]
    assert plan.primary_font_size == _DESKTOP_CAPTION_SIZE_PRESETS["small"].primary_font_size
    assert plan.secondary_font_size == _DESKTOP_CAPTION_SIZE_PRESETS["small"].secondary_font_size
    assert {line.font_family for line in plan.lines} == {"Noto Sans CJK SC"}


@pytest.mark.parametrize(
    (
        "block",
        "line_text",
        "expected_family",
        "expected_weight",
        "expected_flet_weight",
    ),
    [
        pytest.param(
            _block(
                "explicit-ja-latin-primary",
                channel="self",
                block_variant="finalized",
                appearance_seq=1,
                primary_text="Arigato for the live captions",
                primary_language="ja-JP",
            ),
            "Arigato for the live captions",
            "Noto Sans CJK JP",
            "medium",
            ft.FontWeight.W_500,
            id="explicit-primary-ja-latin-text",
        ),
        pytest.param(
            _block(
                "heuristic-cjk-primary",
                channel="peer",
                block_variant="finalized",
                appearance_seq=1,
                primary_text="오늘도 captions are readable",
            ),
            "오늘도 captions are readable",
            "Noto Sans CJK JP",
            "medium",
            ft.FontWeight.W_500,
            id="heuristic-cjk-text-without-language",
        ),
        pytest.param(
            _block(
                "general-english-primary",
                channel="self",
                block_variant="finalized",
                appearance_seq=1,
                primary_text="Live captions stay readable tonight",
                primary_language="en-US",
            ),
            "Live captions stay readable tonight",
            "Noto Sans",
            "semibold",
            ft.FontWeight.W_600,
            id="general-english-text",
        ),
        pytest.param(
            _block(
                "explicit-ja-secondary",
                channel="peer",
                block_variant="finalized",
                appearance_seq=1,
                primary_text="Translated result",
                secondary_text="Original text in romaji",
                secondary_enabled=True,
                primary_language="en-US",
                secondary_language="jpn",
            ),
            "Original text in romaji",
            "Noto Sans CJK JP",
            "medium",
            ft.FontWeight.W_500,
            id="explicit-secondary-ja-latin-text",
        ),
    ],
)
def test_desktop_overlay_caption_font_policy_uses_language_metadata_for_jp_unified_cjk(
    block: OverlayPresentationBlock,
    line_text: str,
    expected_family: str,
    expected_weight: str,
    expected_flet_weight: object,
) -> None:
    plan = desktop_overlay.build_desktop_caption_plan(
        OverlayPresentationSnapshot(blocks=[block]),
        locale="en",
    )

    line_by_text = {line.text: line for line in plan.lines}
    line = line_by_text[line_text]
    assert line.font_family == expected_family
    assert line.weight == expected_weight

    surface = desktop_overlay.build_desktop_caption_surface(plan)
    text_control = _caption_text_control(surface, line_text)
    assert text_control.font_family == expected_family
    assert text_control.style.font_family == expected_family
    assert text_control.weight == expected_flet_weight
    assert text_control.style.weight == expected_flet_weight


def test_desktop_overlay_caption_font_policy_uses_latin_and_jp_unified_cjk_faces_without_packaged_fonts() -> (
    None
):
    snapshot = OverlayPresentationSnapshot(
        blocks=[
            _block(
                "latin-only",
                channel="self",
                block_variant="finalized",
                appearance_seq=1,
                primary_text="Live captions stay readable tonight",
            ),
            _block(
                "mixed-cjk",
                channel="peer",
                block_variant="finalized",
                appearance_seq=2,
                primary_text="오늘도 captions are readable",
            ),
        ]
    )

    plan = desktop_overlay.build_desktop_caption_plan(snapshot, locale="en")

    assert [(line.text, line.font_family) for line in plan.lines] == [
        ("Live captions stay readable tonight", "Noto Sans"),
        ("오늘도 captions are readable", "Noto Sans CJK JP"),
    ]


def test_desktop_overlay_caption_cjk_font_follows_ui_locale() -> None:
    snapshot = OverlayPresentationSnapshot(
        blocks=[
            _block(
                "cjk-line",
                channel="self",
                block_variant="finalized",
                appearance_seq=1,
                primary_text="오늘도 captions are readable",
            )
        ]
    )

    ko_plan = desktop_overlay.build_desktop_caption_plan(snapshot, locale="ko")
    ja_plan = desktop_overlay.build_desktop_caption_plan(snapshot, locale="ja")
    zh_plan = desktop_overlay.build_desktop_caption_plan(snapshot, locale="zh-CN")

    assert ko_plan.lines[0].font_family == "Noto Sans CJK KR"
    assert ja_plan.lines[0].font_family == "Noto Sans CJK JP"
    assert zh_plan.lines[0].font_family == "Noto Sans CJK SC"


def test_desktop_overlay_caption_weight_uses_semibold_for_general_text() -> None:
    plan = desktop_overlay.build_desktop_caption_plan(
        OverlayPresentationSnapshot(
            blocks=[
                _block(
                    "caption-weight",
                    channel="peer",
                    block_variant="finalized",
                    appearance_seq=1,
                    primary_text="Readable semibold caption",
                )
            ]
        )
    )

    assert {line.weight for line in plan.lines} == {"semibold"}

    surface = desktop_overlay.build_desktop_caption_surface(plan)
    slot_column = surface.content.controls[0]
    outer_slot = slot_column.controls[0]
    inner_card = outer_slot.content
    text_layer = inner_card.content
    text_column = text_layer.content
    text_control = text_column.controls[0].content
    assert text_control.weight == ft.FontWeight.W_600
    assert text_control.style.weight == ft.FontWeight.W_600


class RecordingLifecycleSink:
    def __init__(self) -> None:
        self.events: list[dict[str, object]] = []

    async def emit(self, event: dict[str, object]) -> None:
        self.events.append(dict(event))


class RecordingWindowZOrderPort:
    def __init__(
        self,
        *,
        on_reassert: Any | None = None,
        result: desktop_window_zorder.WindowZOrderResult | None = None,
        error: Exception | None = None,
        reveal_result: desktop_window_zorder.WindowVisibilityConfirmation | None = None,
        reveal_error: Exception | None = None,
        placement_result: desktop_window_zorder.WindowBoundsConfirmation | None = None,
        placement_error: Exception | None = None,
        on_placement: Any | None = None,
    ) -> None:
        self.on_reassert = on_reassert
        self.result = result or desktop_window_zorder.WindowZOrderResult(
            applied=True,
            reason="applied",
            click_through_confirmed=True,
            topmost_style_present=True,
        )
        self.error = error
        self.reveal_result = reveal_result or desktop_window_zorder.WindowVisibilityConfirmation(
            confirmed=True,
            reason="confirmed",
            hwnd=4242,
            title_confirmed=True,
            visible_confirmed=True,
            bounds_confirmed=True,
        )
        self.reveal_error = reveal_error
        self.placement_result = placement_result or desktop_window_zorder.WindowBoundsConfirmation(
            confirmed=True,
            reason="confirmed",
            hwnd=4242,
            title_confirmed=True,
            bounds_confirmed=True,
        )
        self.placement_error = placement_error
        self.on_placement = on_placement
        self.reveal_titles: list[str] = []
        self.placement_calls: list[tuple[str, int, int, int, int]] = []
        self.bound_pids: list[int] = []
        self.reassert_calls = 0
        self.close_calls = 0

    def bind_process(self, pid: int) -> None:
        self.bound_pids.append(pid)

    async def confirm_window_bounds(
        self,
        expected_title: str,
        *,
        x: int,
        y: int,
        width: int,
        height: int,
    ) -> desktop_window_zorder.WindowBoundsConfirmation:
        self.placement_calls.append((expected_title, x, y, width, height))
        if self.on_placement is not None:
            self.on_placement()
        if self.placement_error is not None:
            raise self.placement_error
        return self.placement_result

    async def confirm_window_visible(
        self,
        expected_title: str,
        *,
        x: int,
        y: int,
        width: int,
        height: int,
    ) -> desktop_window_zorder.WindowVisibilityConfirmation:
        _ = (x, y, width, height)
        self.reveal_titles.append(expected_title)
        if self.reveal_error is not None:
            raise self.reveal_error
        return self.reveal_result

    async def reassert_topmost_after_click_through(
        self,
    ) -> desktop_window_zorder.WindowZOrderResult:
        self.reassert_calls += 1
        if self.on_reassert is not None:
            self.on_reassert()
        if self.error is not None:
            raise self.error
        return self.result

    def close(self) -> None:
        self.close_calls += 1


class RecordingWebSocket:
    def __init__(self) -> None:
        self.sent_messages: list[dict[str, object]] = []

    async def send(self, payload: str) -> None:
        decoded = json.loads(payload)
        assert isinstance(decoded, dict)
        self.sent_messages.append(decoded)


class FakeFletWindow:
    _PROTECTED_PROPERTIES = {
        "visible",
        "always_on_top",
        "frameless",
        "shadow",
        "skip_task_bar",
        "resizable",
        "maximizable",
        "title_bar_hidden",
        "title_bar_buttons_hidden",
        "bgcolor",
    }

    def __init__(self, app: FakeFletApp) -> None:
        self._record_writes = False
        self._app = app
        self.visible: bool = True
        self.frameless: bool | None = None
        self.always_on_top: bool | None = None
        self.shadow: bool | None = None
        self.skip_task_bar: bool | None = None
        self.resizable: bool | None = None
        self.title_bar_hidden: bool | None = None
        self.title_bar_buttons_hidden: bool | None = None
        self.maximizable: bool | None = None
        self.bgcolor: object | None = None
        self.icon: str | None = None
        self.ignore_mouse_events: bool | None = None
        self.left: int | float = 0
        self.top: int | float = 0
        self.width: int | float = 0
        self.height: int | float = 0
        self.on_event: Any | None = None
        self.close_calls = 0
        self.destroy_calls = 0
        self.start_resizing_calls = 0
        self.ready_wait_calls = 0
        self.protected_writes: list[str] = []
        self._record_writes = True

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_record_writes", False) and name in self._PROTECTED_PROPERTIES:
            self.protected_writes.append(name)
        super().__setattr__(name, value)

    async def close(self) -> None:
        self.close_calls += 1
        self._app.closed.set()

    async def destroy(self) -> None:
        self.destroy_calls += 1
        self._app.closed.set()

    async def wait_until_ready_to_show(self) -> None:
        self.ready_wait_calls += 1
        return None

    def start_resizing(self, *_args: object, **_kwargs: object) -> None:
        self.start_resizing_calls += 1
        raise AssertionError("Flet 0.28.3 start_resizing must not be used")


class FakeFletPage:
    def __init__(self, app: FakeFletApp) -> None:
        self.window = FakeFletWindow(app)
        self.title: str | None = None
        self.controls: list[object] = []
        self.bgcolor: object | None = None
        self.padding: object | None = None
        self.spacing: object | None = None
        self.horizontal_alignment: object | None = None
        self.vertical_alignment: object | None = None
        self.update_calls = 0
        self.add_calls = 0
        self.clean_calls = 0
        self.render_snapshots: list[dict[str, object]] = []
        self.visibility_updates: list[bool] = []
        self.run_task_calls = 0
        self.tasks: list[asyncio.Task[object]] = []

    def add(self, *controls: object) -> None:
        self.add_calls += 1
        self.controls.extend(controls)

    def clean(self) -> None:
        self.clean_calls += 1
        self.controls.clear()

    def update(self) -> None:
        self.update_calls += 1
        for control in self.controls:
            for item in _walk_control_tree(control):
                if isinstance(item, ft.WindowDragArea):
                    item.before_update()
        self.visibility_updates.append(self.window.visible)
        self.render_snapshots.append(
            {
                "ignore_mouse_events": self.window.ignore_mouse_events,
                "texts": _page_text_values(self),
                "has_drag_area": _page_contains_control_type(self, ft.WindowDragArea),
                "card_count": len(_caption_card_controls(self)),
            }
        )

    def run_task(self, func: Any, *args: object, **kwargs: object) -> asyncio.Task[object]:
        assert asyncio.iscoroutinefunction(func)
        self.run_task_calls += 1
        result = func(*args, **kwargs)
        if inspect.isawaitable(result):
            task = asyncio.create_task(result)  # type: ignore[arg-type]
        else:

            async def _completed() -> object:
                return result

            task = asyncio.create_task(_completed())
        self.tasks.append(task)
        return task


class JumpingRevealFletPage(FakeFletPage):
    def __init__(self, app: FakeFletApp) -> None:
        super().__init__(app)
        self.reveal_jump_done = False

    def update(self) -> None:
        super().update()
        if (
            not self.reveal_jump_done
            and self.window.visible is True
            and self.window.ignore_mouse_events is True
        ):
            self.window.left = 608
            self.window.top = 1117
            self.reveal_jump_done = True


class FakeFletApp:
    def __init__(self) -> None:
        self.closed = asyncio.Event()
        self.page = FakeFletPage(self)
        self.targets: list[Any] = []

    async def run(self, target: Any) -> None:
        self.targets.append(target)
        result = target(self.page)
        if inspect.isawaitable(result):
            await result
        await self.closed.wait()


class JumpingRevealFletApp(FakeFletApp):
    def __init__(self) -> None:
        self.closed = asyncio.Event()
        self.page = JumpingRevealFletPage(self)
        self.targets: list[Any] = []


class FakeWindowEvent:
    def __init__(self, event_type: object) -> None:
        self.type = event_type
        self.data = event_type


def _walk_control_tree(control: object) -> list[object]:
    seen: list[object] = [control]
    for attr in ("content", "leading", "trailing"):
        child = getattr(control, attr, None)
        if child is not None:
            seen.extend(_walk_control_tree(child))
    children = getattr(control, "controls", None)
    if isinstance(children, list | tuple):
        for child in children:
            seen.extend(_walk_control_tree(child))
    return seen


def _caption_text_control(control: object, text: str) -> ft.Text:
    for item in _walk_control_tree(control):
        if isinstance(item, ft.Text) and item.value == text:
            return item
    raise AssertionError(f"caption text control not found: {text}")


def _page_text_values(page: FakeFletPage) -> set[str]:
    values: set[str] = set()
    for control in page.controls:
        for item in _walk_control_tree(control):
            if getattr(item, "visible", True) is False:
                continue
            value = getattr(item, "value", None)
            if isinstance(value, str) and value:
                values.add(value)
            content = getattr(item, "content", None)
            if isinstance(content, str) and content:
                values.add(content)
    return values


def _find_control_with_text(page: FakeFletPage, text: str) -> object:
    for control in page.controls:
        for item in _walk_control_tree(control):
            if getattr(item, "content", None) == text or getattr(item, "value", None) == text:
                return item
    raise AssertionError(f"control text not found: {text}")


def _find_text_button(page: FakeFletPage, text: str) -> ft.TextButton:
    for control in page.controls:
        for item in _walk_control_tree(control):
            if isinstance(item, ft.TextButton) and getattr(item, "content", None) == text:
                return item
    raise AssertionError(f"text button not found: {text}")


def _text_buttons(page: FakeFletPage) -> list[ft.TextButton]:
    return [
        item
        for control in page.controls
        for item in _walk_control_tree(control)
        if isinstance(item, ft.TextButton)
    ]


def _page_contains_control_type(page: FakeFletPage, control_type: type[object]) -> bool:
    return any(
        isinstance(item, control_type)
        for control in page.controls
        for item in _walk_control_tree(control)
    )


OLD_OVERLAY_LOCAL_RENDERER_TEXT = {
    "Move captions",
    "Lock captions",
    "Reset to bottom center",
    "Drag edges to resize",
    "You can move this again from the main window.",
    "Captions will appear here",
    "Outline width",
    "Text scale",
}


def _caption_card_controls(page: FakeFletPage) -> list[ft.Container]:
    cards: list[ft.Container] = []
    for control in page.controls:
        for item in _walk_control_tree(control):
            if (
                isinstance(item, ft.Container)
                and getattr(item, "bgcolor", None) in {"#80000000", "#99000000"}
                and getattr(item, "border_radius", None) in {14, 16, 18, 20}
            ):
                cards.append(item)
    return cards


def _assert_no_overlay_local_renderer_text(page: FakeFletPage) -> None:
    assert _page_text_values(page).isdisjoint(OLD_OVERLAY_LOCAL_RENDERER_TEXT)


def test_desktop_overlay_caption_card_width_grows_only_within_same_block() -> None:
    window = desktop_overlay.FletDesktopRendererWindow()

    short_plan = desktop_overlay.build_desktop_caption_plan(
        OverlayPresentationSnapshot(
            blocks=[
                _block(
                    "same-peer",
                    channel="peer",
                    block_variant="finalized",
                    appearance_seq=1,
                    primary_text="응",
                )
            ]
        ),
        window_width=1344,
        window_height=336,
    )
    widened_plan = desktop_overlay.build_desktop_caption_plan(
        OverlayPresentationSnapshot(
            blocks=[
                _block(
                    "same-peer",
                    channel="peer",
                    block_variant="finalized",
                    appearance_seq=1,
                    primary_text="오늘 만나서 정말 반가웠어요 다음에도 같이 이야기해요",
                )
            ]
        ),
        window_width=1344,
        window_height=336,
    )
    shortened_plan = desktop_overlay.build_desktop_caption_plan(
        OverlayPresentationSnapshot(
            blocks=[
                _block(
                    "same-peer",
                    channel="peer",
                    block_variant="finalized",
                    appearance_seq=1,
                    primary_text="응",
                )
            ]
        ),
        window_width=1344,
        window_height=336,
    )

    short_applied = window._plan_with_grow_only_caption_card_widths(short_plan)
    widened_applied = window._plan_with_grow_only_caption_card_widths(widened_plan)
    shortened_applied = window._plan_with_grow_only_caption_card_widths(shortened_plan)

    assert short_applied.slots[0].card_width < widened_applied.slots[0].card_width
    assert shortened_plan.slots[0].card_width == pytest.approx(short_plan.slots[0].card_width)
    assert shortened_applied.slots[0].card_width == pytest.approx(
        widened_applied.slots[0].card_width
    )
    assert shortened_applied.slots[0].card_text_width == pytest.approx(
        shortened_applied.slots[0].card_width - (shortened_applied.padding_horizontal * 2)
    )
    assert shortened_applied.lines == tuple(
        line for slot in shortened_applied.slots for line in slot.lines
    )

    shortened_surface = desktop_overlay.build_desktop_caption_surface(shortened_applied)
    outer_slot = shortened_surface.content.controls[0].controls[0]
    inner_card = outer_slot.content
    text_layer = inner_card.content
    first_line_region = text_layer.content.controls[0]
    assert inner_card.width == pytest.approx(shortened_applied.slots[0].card_width)
    assert text_layer.width == pytest.approx(shortened_applied.slots[0].card_text_width)
    assert first_line_region.width == pytest.approx(shortened_applied.slots[0].card_text_width)


def test_desktop_overlay_caption_card_width_resets_for_new_block_even_same_occupant() -> None:
    window = desktop_overlay.FletDesktopRendererWindow()

    old_block = replace(
        _block(
            "old-peer",
            channel="peer",
            block_variant="finalized",
            appearance_seq=1,
            primary_text="오늘 만나서 정말 반가웠어요 다음에도 같이 이야기해요",
        ),
        occupant_key="peer:same-speaker",
    )
    new_block = replace(
        _block(
            "new-peer",
            channel="peer",
            block_variant="finalized",
            appearance_seq=2,
            primary_text="응",
        ),
        occupant_key="peer:same-speaker",
    )

    long_plan = desktop_overlay.build_desktop_caption_plan(
        OverlayPresentationSnapshot(blocks=[old_block]),
        window_width=1344,
        window_height=336,
    )
    new_short_plan = desktop_overlay.build_desktop_caption_plan(
        OverlayPresentationSnapshot(blocks=[new_block]),
        window_width=1344,
        window_height=336,
    )

    long_applied = window._plan_with_grow_only_caption_card_widths(long_plan)
    new_short_applied = window._plan_with_grow_only_caption_card_widths(new_short_plan)

    assert long_applied.slots[0].card_width > new_short_applied.slots[0].card_width
    assert new_short_applied.slots[0].card_width == pytest.approx(
        new_short_plan.slots[0].card_width
    )


def test_desktop_overlay_preview_fixtures_cover_required_local_qa_cases() -> None:
    catalog = desktop_overlay.build_desktop_overlay_preview_catalog(locale="en")

    assert tuple(preset.id for preset in catalog.size_presets) == (
        "xlarge",
        "large",
        "medium",
        "small",
        "xsmall",
        "tiny",
    )
    assert tuple(catalog.background_alpha_presets) == _DESKTOP_PREVIEW_BACKGROUND_ALPHA_PRESETS
    assert tuple(surface.id for surface in catalog.background_surfaces) == (
        "bright",
        "dark",
        "busy",
    )

    required_tags = {
        "ko",
        "ja",
        "zh-CN",
        "en",
        "mixed_script",
        "emoji",
        "self",
        "peer",
        "primary",
        "secondary",
        "active",
        "finalized",
        "long_wrap",
        "no_caption",
    }
    coverage_tags = {tag for fixture in catalog.fixtures for tag in fixture.coverage_tags}
    assert required_tags <= coverage_tags
    assert len({fixture.id for fixture in catalog.fixtures}) == len(catalog.fixtures)

    long_wrap_fixture = next(
        fixture for fixture in catalog.fixtures if "long_wrap" in fixture.coverage_tags
    )
    long_wrap_texts = [
        text
        for block in long_wrap_fixture.snapshot.blocks
        for text in (block.primary_text, block.secondary_text)
    ]
    assert any(len(text) >= 90 for text in long_wrap_texts)

    for fixture in catalog.fixtures:
        if "no_caption" in fixture.coverage_tags:
            assert fixture.snapshot.blocks == []
            continue
        assert fixture.snapshot.blocks, fixture.id
        plan = desktop_overlay.build_desktop_caption_plan(
            fixture.snapshot,
            window_width=1344,
            window_height=320,
            visual_state=desktop_overlay.DesktopCaptionVisualState(
                text_scale=1.0,
                background_alpha=0.5,
                outline_width=None,
            ),
        )
        assert plan.lines, fixture.id


def test_desktop_overlay_preview_no_caption_fixture_supports_manual_qa_states() -> None:
    catalog = desktop_overlay.build_desktop_overlay_preview_catalog(locale="en")

    fixture = next(fixture for fixture in catalog.fixtures if "no_caption" in fixture.coverage_tags)

    assert fixture.label == desktop_overlay.t_for_locale(
        "en", "settings.overlay.desktop.preview.fixture.no_captions"
    )
    edit_plan = desktop_overlay.build_desktop_caption_plan(
        fixture.snapshot,
        interaction_mode="edit",
        locale="en",
    )
    locked_plan = desktop_overlay.build_desktop_caption_plan(
        fixture.snapshot,
        interaction_mode="pass_through",
        locale="en",
    )
    assert edit_plan.lines == ()
    assert edit_plan.surface_visible is True
    assert edit_plan.background_alpha == pytest.approx(0.6)
    assert locked_plan.lines == ()
    assert locked_plan.surface_visible is False


def test_desktop_overlay_preview_uses_edit_mode_for_no_caption_empty_state() -> None:
    app = FakeFletApp()
    catalog = desktop_overlay.build_desktop_overlay_preview_catalog(locale="en")
    window = desktop_overlay.FletDesktopRendererWindow(
        app_runner=app.run,
        event_sink=RecordingLifecycleSink().emit,
        locale="en",
        preview_catalog=catalog,
        bounds_debounce_s=0.01,
    )
    no_caption_fixture = next(
        fixture for fixture in catalog.fixtures if "no_caption" in fixture.coverage_tags
    )

    window._preview_fixture_id = no_caption_fixture.id  # noqa: SLF001 - preview state test
    window._page = app.page  # noqa: SLF001 - render preview without starting Flet app loop
    window._interaction_mode = "edit"  # noqa: SLF001 - preview must expose edit empty state

    window._render_page()  # noqa: SLF001 - verify preview rendering contract

    assert desktop_overlay.t_for_locale(
        "en", "settings.overlay.desktop.empty_state.action.lock"
    ) in _page_text_values(app.page)
    assert len(_text_buttons(app.page)) == 1


def test_desktop_overlay_preview_fixture_data_secret_guard_rejects_bearer_tokens() -> None:
    catalog = desktop_overlay.build_desktop_overlay_preview_catalog(locale="en")

    assert desktop_overlay.preview_fixture_secret_findings(catalog) == ()

    malicious_fixture = replace(
        catalog.fixtures[0],
        snapshot=OverlayPresentationSnapshot(
            revision=999,
            blocks=[
                _block(
                    "preview-malicious-token",
                    channel="self",
                    block_variant="active_self",
                    appearance_seq=1,
                    primary_text="Authorization: Bearer real-preview-token-material",
                    secondary_text="sk-live-not-a-fixture-token-material",
                    secondary_enabled=True,
                )
            ],
        ),
    )
    unsafe_catalog = replace(catalog, fixtures=(malicious_fixture,))

    findings = desktop_overlay.preview_fixture_secret_findings(unsafe_catalog)

    assert len(findings) == 2
    assert all("korean_long_wrap" in finding for finding in findings)
    assert all("real-preview-token-material" not in finding for finding in findings)
    assert all("sk-live-not-a-fixture-token-material" not in finding for finding in findings)

    malicious_id_fixture = replace(
        catalog.fixtures[0],
        id="sk-live-fixture-id-token-material",
    )

    id_findings = desktop_overlay.preview_fixture_secret_findings(
        replace(catalog, fixtures=(malicious_id_fixture,))
    )

    assert id_findings
    assert all("sk-live-fixture-id-token-material" not in finding for finding in id_findings)


def test_desktop_overlay_preview_fixture_data_secret_guard_recurses_snapshot_metadata() -> None:
    catalog = desktop_overlay.build_desktop_overlay_preview_catalog(locale="en")
    fixture = replace(
        catalog.fixtures[0],
        snapshot=OverlayPresentationSnapshot(
            revision=1000,
            calibration=OverlayPresentationCalibration(
                anchor="Bearer calibration-anchor-token-material",
            ),
            blocks=[
                OverlayPresentationBlock(
                    id="preview-safe-metadata",
                    occupant_key="self:preview-safe-metadata",
                    appearance_seq=1,
                    channel="self",
                    block_variant="active_self",
                    primary_text="Safe preview text",
                    secondary_text="Safe secondary text",
                    secondary_enabled=True,
                    update_id="sk-live-update-id-token-material",
                    session_scope="Bearer session-scope-token-material",
                    source_text_hash="sk-live-source-hash-token-material",
                    logical_turn_key="Bearer logical-turn-token-material",
                )
            ],
        ),
    )

    findings = desktop_overlay.preview_fixture_secret_findings(
        replace(catalog, fixtures=(fixture,))
    )

    assert len(findings) == 5
    assert any("snapshot.calibration.anchor" in finding for finding in findings)
    assert any("snapshot.blocks[0].update_id" in finding for finding in findings)
    assert any("snapshot.blocks[0].session_scope" in finding for finding in findings)
    assert any("snapshot.blocks[0].source_text_hash" in finding for finding in findings)
    assert any("snapshot.blocks[0].logical_turn_key" in finding for finding in findings)
    assert all("token-material" not in finding for finding in findings)


def test_desktop_overlay_preview_fixture_data_secret_guard_scans_catalog_controls() -> None:
    catalog = desktop_overlay.build_desktop_overlay_preview_catalog(locale="en")
    unsafe_catalog = replace(
        catalog,
        labels=replace(
            catalog.labels,
            fixture="Bearer catalog-label-token-material",
        ),
        size_presets=(
            replace(
                catalog.size_presets[0],
                label="sk-live-size-preset-token-material",
            ),
            *catalog.size_presets[1:],
        ),
        background_surfaces=(
            replace(
                catalog.background_surfaces[0],
                bgcolor="Bearer surface-bg-token-material",
            ),
            *catalog.background_surfaces[1:],
        ),
    )

    findings = desktop_overlay.preview_fixture_secret_findings(unsafe_catalog)

    assert len(findings) == 3
    assert any("labels.fixture" in finding for finding in findings)
    assert any("size_presets[0].label" in finding for finding in findings)
    assert any("background_surfaces[0].bgcolor" in finding for finding in findings)
    assert all("token-material" not in finding for finding in findings)


def test_desktop_overlay_lifecycle_redaction_covers_common_secret_key_variants() -> None:
    redacted = desktop_overlay._redact_event(
        {
            "type": "runtime_error",
            "payload": {
                "api_key": "plain-api-key-value",
                "access_token": "plain-access-token-value",
                "sessionToken": "camel-session-token-value",
                "authorization_header": "plain-auth-header-value",
                "safe": "visible",
            },
        }
    )

    assert redacted == {
        "type": "runtime_error",
        "payload": {
            "api_key": "<redacted>",
            "access_token": "<redacted>",
            "sessionToken": "<redacted>",
            "authorization_header": "<redacted>",
            "safe": "visible",
        },
    }


def test_desktop_overlay_preview_fixture_data_packaging_readiness_is_embedded() -> None:
    sources = desktop_overlay.desktop_overlay_preview_fixture_data_sources()

    assert sources == (
        desktop_overlay.DesktopOverlayPreviewFixtureDataSource(
            source_kind="embedded_python_module",
            module="puripuly_heart.ui.desktop_overlay_surface.renderer",
            package_data_globs=(),
            hiddenimports=(),
        ),
    )
    assert all(not source.package_data_globs for source in sources)
    assert all(not source.hiddenimports for source in sources)


def test_desktop_overlay_preview_guard_stops_before_rendering_unsafe_fixture_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog = desktop_overlay.build_desktop_overlay_preview_catalog(locale="en")
    malicious_fixture = replace(
        catalog.fixtures[0],
        snapshot=OverlayPresentationSnapshot(
            revision=999,
            blocks=[
                _block(
                    "preview-malicious-token",
                    channel="self",
                    block_variant="active_self",
                    appearance_seq=1,
                    primary_text="Authorization: Bearer real-preview-token-material",
                )
            ],
        ),
    )
    unsafe_catalog = replace(catalog, fixtures=(malicious_fixture,))
    monkeypatch.setattr(
        desktop_overlay,
        "build_desktop_overlay_preview_catalog",
        lambda *, locale=None: unsafe_catalog,
    )

    def fail_app_runner(_target: Any) -> object:
        raise AssertionError("unsafe preview fixture data must not be rendered")

    assert desktop_overlay.run_preview(app_runner=fail_app_runner, locale="en") == 1


def test_desktop_overlay_preview_guard_avoids_provider_broker_stt_translation_secretstore_and_settings_save_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from puripuly_heart.config import settings as settings_module

    forbidden_prefixes = (
        "puripuly_heart.app.wiring",
        "puripuly_heart.core.managed_openrouter_broker_client",
        "puripuly_heart.core.storage.secrets",
        "puripuly_heart.core.stt",
        "puripuly_heart.providers",
    )
    original_import = builtins.__import__

    def guarded_import(
        name: str,
        globals_: dict[str, object] | None = None,
        locals_: dict[str, object] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> object:
        if level == 0 and any(
            name == prefix or name.startswith(f"{prefix}.") for prefix in forbidden_prefixes
        ):
            raise AssertionError(f"preview must not import {name}")
        return original_import(name, globals_, locals_, fromlist, level)

    def fail_settings_save(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("preview must not call settings-save paths")

    def fail_write_text(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("preview must not persist fixture or settings data")

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(settings_module, "save_settings", fail_settings_save)
    monkeypatch.setattr(Path, "write_text", fail_write_text)

    app = FakeFletApp()

    def run_preview_target(target: Any) -> None:
        target(app.page)

    assert desktop_overlay.run_preview(app_runner=run_preview_target, locale="en") == 0
    assert app.page.controls


def test_desktop_overlay_preview_fixtures_use_real_overlay_window_surface_and_edit_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_renderer_path(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("preview must not load a renderer manifest or start bridge runtime")

    monkeypatch.setattr(desktop_overlay, "load_renderer_manifest", fail_renderer_path)
    monkeypatch.setattr(desktop_overlay, "DesktopOverlayRenderer", fail_renderer_path)

    app = FakeFletApp()

    def run_preview_target(target: Any) -> None:
        app.targets.append(target)
        result = target(app.page)
        assert not inspect.isawaitable(result)

    assert desktop_overlay.run_preview(app_runner=run_preview_target, locale="en") == 0

    assert app.page.window.frameless is True
    assert app.page.title == "PuriPuly Overlay"
    assert app.page.window.icon == "icons/icon.ico"
    assert app.page.window.always_on_top is True
    assert app.page.window.shadow is False
    assert app.page.window.skip_task_bar is False
    assert app.page.window.resizable is False
    assert app.page.window.bgcolor == ft.Colors.TRANSPARENT
    assert app.page.bgcolor == ft.Colors.TRANSPARENT
    assert app.page.window.ignore_mouse_events is False
    assert app.page.window.width >= desktop_overlay._DESKTOP_PREVIEW_STAGE_WIDTH
    assert app.page.window.height >= desktop_overlay._DESKTOP_PREVIEW_STAGE_HEIGHT
    assert not _page_contains_control_type(app.page, ft.WindowDragArea)

    visible_text = _page_text_values(app.page)
    assert (
        desktop_overlay.t_for_locale("en", "settings.overlay.desktop.preview.fixture")
        in visible_text
    )
    assert desktop_overlay.t_for_locale("en", "settings.overlay.desktop.size.title") in visible_text
    assert (
        desktop_overlay.t_for_locale("en", "settings.overlay.desktop.preview.background_surface")
        in visible_text
    )
    assert "Outline width" not in visible_text
    assert "Text scale" not in visible_text


@pytest.mark.asyncio
async def test_desktop_overlay_preview_controls_apply_size_preset_without_outline_controls() -> (
    None
):
    app = FakeFletApp()
    sink = RecordingLifecycleSink()
    catalog = desktop_overlay.build_desktop_overlay_preview_catalog(locale="en")
    window = desktop_overlay.FletDesktopRendererWindow(
        app_runner=app.run,
        event_sink=sink.emit,
        locale="en",
        bounds_debounce_s=0.01,
        preview_catalog=catalog,
    )

    try:
        await window.start(catalog.fixtures[0].snapshot)
        assert desktop_overlay.t_for_locale(
            "en", "settings.overlay.desktop.preview.background_surface"
        ) in _page_text_values(app.page)
        assert "Outline width" not in _page_text_values(app.page)

        large_control = _find_control_with_text(app.page, "Large")
        large_handler = getattr(large_control, "on_click", None)
        assert callable(large_handler)
        large_handler(None)
        if app.page.tasks:
            await asyncio.gather(*app.page.tasks)

        assert app.page.window.ignore_mouse_events is False
        assert app.page.window.width == _DESKTOP_CAPTION_SIZE_PRESETS["large"].window_width
        assert app.page.window.height == _DESKTOP_CAPTION_SIZE_PRESETS["large"].window_height
        visible_text = _page_text_values(app.page)
        assert (
            desktop_overlay.t_for_locale(
                "en", "settings.overlay.desktop.preview.background_surface"
            )
            in visible_text
        )
        assert sink.events == []
    finally:
        await window.close()


@pytest.mark.asyncio
async def test_desktop_overlay_preview_post_start_updates_retain_caption_surface_and_window_state() -> (
    None
):
    app = FakeFletApp()
    catalog = desktop_overlay.build_desktop_overlay_preview_catalog(locale="en")
    window = desktop_overlay.FletDesktopRendererWindow(
        app_runner=app.run,
        event_sink=RecordingLifecycleSink().emit,
        locale="en",
        preview_catalog=catalog,
    )

    try:
        await window.start(catalog.fixtures[0].snapshot)
        model = window._retained_caption_surface
        assert model is not None
        identities = (
            app.page.controls[0],
            model.caption_surface,
            *model.slot_containers,
            *model.primary_texts,
            *model.secondary_texts,
        )
        protected_writes = len(app.page.window.protected_writes)
        initial_bounds = (
            app.page.window.left,
            app.page.window.top,
            app.page.window.width,
            app.page.window.height,
        )

        for label in (
            catalog.fixtures[1].label,
            "50%",
            catalog.background_surfaces[1].label,
        ):
            handler = getattr(_find_control_with_text(app.page, label), "on_click", None)
            assert callable(handler)
            handler(None)
            assert identities == (
                app.page.controls[0],
                model.caption_surface,
                *model.slot_containers,
                *model.primary_texts,
                *model.secondary_texts,
            )
            assert (
                app.page.window.left,
                app.page.window.top,
                app.page.window.width,
                app.page.window.height,
            ) == initial_bounds

        size_handler = getattr(_find_control_with_text(app.page, "Large"), "on_click", None)
        assert callable(size_handler)
        size_handler(None)
        assert (app.page.window.left, app.page.window.top) == initial_bounds[:2]
        assert (app.page.window.width, app.page.window.height) == (1600, 400)

        await window.dispatch_runtime_control(
            {"command": "set_interaction_mode", "mode": "pass_through"}
        )
        window._on_preview_keyboard_event(type("PreviewKeyEvent", (), {"key": "e"})())
        await asyncio.gather(*app.page.tasks)

        assert app.page.window.ignore_mouse_events is False
        assert identities == (
            app.page.controls[0],
            model.caption_surface,
            *model.slot_containers,
            *model.primary_texts,
            *model.secondary_texts,
        )
        assert app.page.add_calls == 1
        assert app.page.clean_calls == 0
        assert len(app.page.window.protected_writes) == protected_writes
    finally:
        await window.close()


def test_desktop_overlay_preview_i18n_labels_resolve_for_all_controls() -> None:
    catalog = desktop_overlay.build_desktop_overlay_preview_catalog(locale="ja")

    assert desktop_overlay.t_for_locale("en", "desktop_overlay.window.title") == "PuriPuly Overlay"
    assert desktop_overlay.t_for_locale("ko", "desktop_overlay.window.title") == "PuriPuly Overlay"
    assert desktop_overlay.t_for_locale("ja", "desktop_overlay.window.title") == "PuriPuly Overlay"
    assert (
        desktop_overlay.t_for_locale("zh-CN", "desktop_overlay.window.title") == "PuriPuly Overlay"
    )
    assert catalog.labels.fixture == "サンプル字幕"
    assert catalog.labels.size_preset == "オーバーレイサイズ"
    assert catalog.labels.background_alpha == "背景の透明度"
    assert catalog.labels.background_surface == "プレビュー背景"
    assert [preset.label for preset in catalog.size_presets] == [
        "さらに大きく",
        "大きめ",
        "標準",
        "小さめ",
        "さらに小さく",
        "最小",
    ]
    assert [surface.label for surface in catalog.background_surfaces] == [
        "明るい背景",
        "暗い背景",
        "にぎやかなデスクトップ",
    ]
    for fixture in catalog.fixtures:
        assert fixture.label.strip()
        assert not fixture.label.startswith("settings.")


def test_desktop_overlay_preview_fixtures_run_local_app_without_renderer_or_persistence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_renderer_path(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("preview must not load a renderer manifest or start bridge runtime")

    def fail_write_text(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("preview must not persist settings or write local fixture data")

    monkeypatch.setattr(desktop_overlay, "load_renderer_manifest", fail_renderer_path)
    monkeypatch.setattr(desktop_overlay, "DesktopOverlayRenderer", fail_renderer_path)
    monkeypatch.setattr(Path, "write_text", fail_write_text)

    app = FakeFletApp()

    def run_preview_target(target: Any) -> None:
        app.targets.append(target)
        result = target(app.page)
        assert not inspect.isawaitable(result)

    assert desktop_overlay.run_preview(app_runner=run_preview_target, locale="en") == 0
    assert len(app.targets) == 1

    visible_text = _page_text_values(app.page)
    assert (
        desktop_overlay.t_for_locale("en", "settings.overlay.desktop.preview.fixture")
        in visible_text
    )
    assert desktop_overlay.t_for_locale("en", "settings.overlay.desktop.size.title") in visible_text
    assert (
        desktop_overlay.t_for_locale("en", "settings.overlay.desktop.preview.background_alpha")
        in visible_text
    )
    assert (
        desktop_overlay.t_for_locale("en", "settings.overlay.desktop.preview.background_surface")
        in visible_text
    )
    assert "Outline width" not in visible_text
    assert "Text scale" not in visible_text
    assert {"65%", "50%", "40%", "20%"} <= visible_text
    assert {
        desktop_overlay.t_for_locale("en", "settings.overlay.desktop.size.option.small"),
        desktop_overlay.t_for_locale("en", "settings.overlay.desktop.size.option.medium"),
        desktop_overlay.t_for_locale("en", "settings.overlay.desktop.size.option.large"),
        desktop_overlay.t_for_locale("en", "settings.overlay.desktop.size.option.xlarge"),
    } <= visible_text
    assert {
        desktop_overlay.t_for_locale(
            "en", "settings.overlay.desktop.preview.background_surface.bright"
        ),
        desktop_overlay.t_for_locale(
            "en", "settings.overlay.desktop.preview.background_surface.dark"
        ),
        desktop_overlay.t_for_locale(
            "en", "settings.overlay.desktop.preview.background_surface.busy"
        ),
    } <= visible_text
    assert (
        desktop_overlay.t_for_locale(
            "en", "settings.overlay.desktop.preview.fixture.korean_long_wrap"
        )
        in visible_text
    )
    assert any("긴 문장" in text for text in visible_text)


@pytest.mark.asyncio
async def test_desktop_overlay_registers_bundled_noto_cjk_on_the_flet_page() -> None:
    app = FakeFletApp()
    window = desktop_overlay.FletDesktopRendererWindow(
        app_runner=app.run,
        event_sink=RecordingLifecycleSink().emit,
        locale="en",
        window_z_order_port=RecordingWindowZOrderPort(),
        window_process_info_provider=lambda: (4321, None),
    )

    try:
        await window.start(OverlayPresentationSnapshot(revision=1, blocks=[]))
    finally:
        await window.close()

    assert app.page.fonts[FONT_FAMILY_NOTO_SANS] == "/fonts/NotoSansCJK-Medium.ttc"
    assert app.page.fonts[FONT_FAMILY_NOTO_SANS_CJK_JP] == "/fonts/NotoSansCJK-Medium.ttc"
    assert app.page.fonts["Noto Sans CJK KR"] == "/fonts/NotoSansCJK-Medium.ttc"
    assert app.page.fonts["Noto Sans CJK SC"] == "/fonts/NotoSansCJK-Medium.ttc"


@pytest.mark.asyncio
async def test_default_flet_app_runner_starts_hidden_to_prevent_startup_flash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    async def fake_run_async(**kwargs: object) -> None:
        calls.append(dict(kwargs))

    def target(_page: object) -> None:
        return None

    monkeypatch.setattr(ft, "run_async", fake_run_async)

    await desktop_overlay._default_flet_app_runner(target)  # noqa: SLF001 - verify runner policy

    assert calls == [
        {
            "main": target,
            "view": ft.AppView.FLET_APP_HIDDEN,
            "assets_dir": str(assets_dir()),
        }
    ]


@pytest.mark.asyncio
async def test_flet_launcher_patch_reports_started_process_pid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import flet_desktop

    started_processes: list[tuple[int, str | None]] = []
    fake_process = type("FakeProcess", (), {"pid": 4321})()

    async def fake_hidden_launcher(
        page_url: str,
        assets_dir: str | None,
        hidden: bool,
    ) -> tuple[object, str]:
        assert (page_url, assets_dir, hidden) == ("flet://desktop-overlay", "assets", True)
        return fake_process, "pid-file"

    monkeypatch.setattr(
        flet_desktop_runtime,
        "open_hidden_view",
        fake_hidden_launcher,
    )

    with flet_desktop_runtime.patch_hidden_view_launcher(
        on_process_started=lambda pid, pid_file: started_processes.append((pid, pid_file))
    ):
        process, pid_file = await flet_desktop.open_flet_view_async(
            "flet://desktop-overlay",
            "assets",
            True,
        )

    assert process is fake_process
    assert pid_file == "pid-file"
    assert started_processes == [(4321, "pid-file")]


def test_desktop_overlay_prefers_flet_pid_file_for_window_binding(tmp_path: Path) -> None:
    pid_file = tmp_path / "flet.pid"
    pid_file.write_text("5678", encoding="utf-8")
    port = RecordingWindowZOrderPort()
    window = desktop_overlay.FletDesktopRendererWindow(
        app_runner=FakeFletApp().run,
        window_z_order_port=port,
        window_process_info_provider=lambda: None,
    )

    window._record_flet_process(4321, str(pid_file))
    window._bind_window_z_order_process()

    assert port.bound_pids == [5678]


def test_desktop_overlay_requires_process_provider_for_custom_runner_and_zorder() -> None:
    with pytest.raises(ValueError, match="window_process_info_provider"):
        desktop_overlay.FletDesktopRendererWindow(
            app_runner=FakeFletApp().run,
            window_z_order_port=RecordingWindowZOrderPort(),
        )


@pytest.mark.asyncio
async def test_desktop_overlay_confirms_hidden_bounds_before_flet_reveal() -> None:
    import flet as real_flet

    assert real_flet.Window.__dataclass_fields__["visible"].default is True

    app = FakeFletApp()
    assert app.page.window.visible is True
    placement_visibility: list[list[bool]] = []
    port = RecordingWindowZOrderPort(
        on_placement=lambda: placement_visibility.append(list(app.page.visibility_updates))
    )
    window = desktop_overlay.FletDesktopRendererWindow(
        app_runner=app.run,
        event_sink=RecordingLifecycleSink().emit,
        locale="en",
        bounds_debounce_s=0.01,
        window_z_order_port=port,
        window_process_info_provider=lambda: (4321, None),
    )
    window.prime_startup_runtime_controls(
        (
            {
                "command": "apply_window_bounds",
                "x": 320,
                "y": 720,
                "width": 1344,
                "height": 320,
            },
        )
    )
    assert inspect.iscoroutinefunction(window._handle_page)

    try:
        await window.start(OverlayPresentationSnapshot(revision=1, blocks=[]))
        await asyncio.sleep(0)
        for task in list(app.page.tasks):
            await task

        assert port.bound_pids == [4321]
        assert app.page.window.ready_wait_calls == 1
        assert app.page.visibility_updates[:2] == [False, True]
        assert placement_visibility == [[False]]
        assert port.placement_calls == [
            (
                desktop_overlay.t_for_locale(
                    "en", "desktop_overlay.window.title", default="PuriPuly Overlay"
                ),
                320,
                720,
                1344,
                320,
            )
        ]
        assert port.reveal_titles == [
            desktop_overlay.t_for_locale(
                "en", "desktop_overlay.window.title", default="PuriPuly Overlay"
            )
        ], (
            "the renderer must ask the platform window port to confirm visibility after "
            "Flet applies the visible patch"
        )
        assert app.page.title == port.reveal_titles[0]
    finally:
        await window.close()


@pytest.mark.asyncio
async def test_desktop_overlay_preview_reveal_does_not_use_platform_window_port() -> None:
    app = FakeFletApp()
    port = RecordingWindowZOrderPort()
    catalog = desktop_overlay.build_desktop_overlay_preview_catalog(locale="en")
    window = desktop_overlay.FletDesktopRendererWindow(
        app_runner=app.run,
        event_sink=RecordingLifecycleSink().emit,
        locale="en",
        preview_catalog=catalog,
        window_z_order_port=port,
        window_process_info_provider=lambda: None,
    )

    try:
        await window.start(catalog.fixtures[0].snapshot)
        await asyncio.sleep(0)
        for task in list(app.page.tasks):
            await task

        assert port.reveal_titles == []
        assert port.placement_calls == []
    finally:
        await window.close()


def test_desktop_overlay_preview_app_runner_is_awaitable_inside_running_loop() -> None:
    runner = desktop_overlay._REAL_DEFAULT_PREVIEW_APP_RUNNER
    source = inspect.getsource(runner)
    assert "ft.run(" not in source, "synchronous ft.run wraps asyncio.run and cannot nest"

    calls: list[dict[str, object]] = []

    class FakeFletModule:
        def run_async(self, *, main: Any) -> Any:
            async def _runner() -> str:
                calls.append({"main": main})
                return "ran"

            return _runner()

    def target(_page: object) -> None:
        return None

    async def exercise() -> object:
        fake = FakeFletModule()
        with monkeypatched_flet_module(fake):
            result = runner(target)
            assert inspect.isawaitable(result)
            return await result

    assert asyncio.run(exercise()) == "ran"
    assert calls == [{"main": target}]


@contextlib.contextmanager
def monkeypatched_flet_module(fake: object) -> Any:
    real = sys.modules.get("flet")
    sys.modules["flet"] = fake  # type: ignore[assignment]
    try:
        yield
    finally:
        if real is None:
            del sys.modules["flet"]
        else:
            sys.modules["flet"] = real


@pytest.mark.asyncio
async def test_hidden_flet_view_launcher_uses_client_hidden_startup_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import flet_desktop

    created: list[dict[str, object]] = []
    requested_hidden: list[bool] = []
    fake_process = object()

    def fake_locate_and_unpack(page_url: str, assets_dir: str, hidden: bool):
        assert page_url == "flet://desktop-overlay"
        assert assets_dir == "assets"
        requested_hidden.append(hidden)
        return (
            ["C:/fake/flet.exe", page_url, "pid-file", assets_dir],
            {"FLET_HIDE_WINDOW_ON_START": "true"} if hidden else {},
            "pid-file",
        )

    async def fake_create_subprocess_exec(*args: object, **kwargs: object) -> object:
        created.append({"args": args, "kwargs": kwargs})
        return fake_process

    monkeypatch.setattr(
        flet_desktop,
        "__locate_and_unpack_flet_view",
        fake_locate_and_unpack,
    )
    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    process, pid_file = await flet_desktop_runtime.open_hidden_view(
        "flet://desktop-overlay",
        "assets",
        True,
    )

    assert process is fake_process
    assert pid_file == "pid-file"
    assert created[0]["args"] == (
        "C:/fake/flet.exe",
        "flet://desktop-overlay",
        "pid-file",
        "assets",
    )
    kwargs = created[0]["kwargs"]
    assert requested_hidden == [True]
    assert kwargs["env"] == {"FLET_HIDE_WINDOW_ON_START": "true"}
    if os.name == "nt":
        assert kwargs["creationflags"] & subprocess.CREATE_NO_WINDOW
        assert kwargs["creationflags"] & subprocess.BELOW_NORMAL_PRIORITY_CLASS
        assert "startupinfo" not in kwargs
    else:
        assert "creationflags" not in kwargs
        assert "startupinfo" not in kwargs


@pytest.mark.asyncio
async def test_desktop_overlay_reveals_first_window_update_after_chrome_bounds_and_content() -> (
    None
):
    app = FakeFletApp()
    port = RecordingWindowZOrderPort()
    window = desktop_overlay.FletDesktopRendererWindow(
        app_runner=app.run,
        event_sink=RecordingLifecycleSink().emit,
        locale="en",
        bounds_debounce_s=0.01,
        window_z_order_port=port,
        window_process_info_provider=lambda: (4321, None),
    )
    window.prime_startup_runtime_controls(
        (
            {
                "command": "apply_window_bounds",
                "x": 320,
                "y": 720,
                "width": 1344,
                "height": 320,
            },
            {"command": "set_interaction_mode", "mode": "pass_through"},
        )
    )

    try:
        await window.start(OverlayPresentationSnapshot(revision=1, blocks=[]))

        assert app.page.window.visible is True
        assert app.page.visibility_updates == [False, True]
        assert app.page.window.frameless is True
        assert app.page.window.shadow is False
        assert app.page.window.resizable is False
        assert app.page.window.always_on_top is True
        assert (app.page.window.left, app.page.window.top) == (320, 720)
        assert (app.page.window.width, app.page.window.height) == (1344, 320)
        assert port.placement_calls == [
            (
                desktop_overlay.t_for_locale(
                    "en", "desktop_overlay.window.title", default="PuriPuly Overlay"
                ),
                320,
                720,
                1344,
                320,
            )
        ]
        assert app.page.render_snapshots[0] == {
            "ignore_mouse_events": False,
            "texts": {
                desktop_overlay.t_for_locale(
                    "en", "settings.overlay.desktop.empty_state.action.lock"
                )
            },
            "has_drag_area": True,
            "card_count": 1,
        }
    finally:
        await window.close()


@pytest.mark.asyncio
async def test_desktop_overlay_start_waits_for_visible_confirmation_before_ready() -> None:
    class BlockingVisibilityPort(RecordingWindowZOrderPort):
        def __init__(self) -> None:
            super().__init__()
            self.confirmation_started = asyncio.Event()
            self.release_confirmation = asyncio.Event()

        async def confirm_window_visible(
            self,
            expected_title: str,
            *,
            x: int,
            y: int,
            width: int,
            height: int,
        ) -> desktop_window_zorder.WindowVisibilityConfirmation:
            _ = (x, y, width, height)
            self.reveal_titles.append(expected_title)
            self.confirmation_started.set()
            await self.release_confirmation.wait()
            return self.reveal_result

    app = FakeFletApp()
    port = BlockingVisibilityPort()
    sink = RecordingLifecycleSink()
    window = desktop_overlay.FletDesktopRendererWindow(
        app_runner=app.run,
        event_sink=sink.emit,
        locale="en",
        window_z_order_port=port,
        window_process_info_provider=lambda: (4321, None),
    )
    window.prime_startup_runtime_controls(
        (
            {
                "command": "apply_window_bounds",
                "x": 320,
                "y": 720,
                "width": 1344,
                "height": 320,
            },
        )
    )

    start_task = asyncio.create_task(
        window.start(OverlayPresentationSnapshot(revision=1, blocks=[]))
    )
    await port.confirmation_started.wait()

    assert start_task.done() is False
    assert window.startup_generation == 0
    assert app.page.window.visible is True
    assert callable(app.page.window.on_event)
    app.page.window.on_event(FakeWindowEvent(ft.WindowEventType.MOVED))
    await asyncio.sleep(0)
    assert sink.events == []

    port.release_confirmation.set()
    await start_task

    assert window.startup_generation == 1
    await window.close()


@pytest.mark.asyncio
async def test_desktop_overlay_unconfirmed_bounds_never_becomes_visible_or_ready() -> None:
    app = FakeFletApp()
    port = RecordingWindowZOrderPort(
        placement_result=desktop_window_zorder.WindowBoundsConfirmation(
            confirmed=False,
            reason="bounds_not_retained",
        )
    )
    window = desktop_overlay.FletDesktopRendererWindow(
        app_runner=app.run,
        event_sink=RecordingLifecycleSink().emit,
        locale="en",
        window_z_order_port=port,
        window_process_info_provider=lambda: (4321, None),
    )
    window.prime_startup_runtime_controls(
        (
            {
                "command": "apply_window_bounds",
                "x": 320,
                "y": 720,
                "width": 1344,
                "height": 320,
            },
        )
    )

    with pytest.raises(RuntimeError, match="Flet page configuration failed") as excinfo:
        await window.start(OverlayPresentationSnapshot(revision=1, blocks=[]))

    assert "canonical bounds were not confirmed" in str(excinfo.value.__cause__)
    assert window.startup_generation == 0
    assert app.page.window.visible is False
    assert port.reveal_titles == []
    await window.close()


@pytest.mark.asyncio
async def test_desktop_overlay_unconfirmed_visibility_never_becomes_ready() -> None:
    app = FakeFletApp()
    port = RecordingWindowZOrderPort(
        reveal_result=desktop_window_zorder.WindowVisibilityConfirmation(
            confirmed=False,
            reason="visibility_not_retained",
        )
    )
    window = desktop_overlay.FletDesktopRendererWindow(
        app_runner=app.run,
        event_sink=RecordingLifecycleSink().emit,
        locale="en",
        window_z_order_port=port,
        window_process_info_provider=lambda: (4321, None),
    )
    window.prime_startup_runtime_controls(
        (
            {
                "command": "apply_window_bounds",
                "x": 320,
                "y": 720,
                "width": 1344,
                "height": 320,
            },
        )
    )

    with pytest.raises(RuntimeError, match="Flet page configuration failed") as excinfo:
        await window.start(OverlayPresentationSnapshot(revision=1, blocks=[]))

    assert "visibility was not confirmed" in str(excinfo.value.__cause__)
    assert window.startup_phase == "bounds_confirmed"
    assert window.startup_generation == 0
    await window.close()


@pytest.mark.asyncio
async def test_desktop_overlay_native_ready_failure_reports_page_configured_phase() -> None:
    app = FakeFletApp()

    async def fail_native_ready() -> None:
        raise RuntimeError("native readiness failed")

    app.page.window.wait_until_ready_to_show = fail_native_ready
    window = desktop_overlay.FletDesktopRendererWindow(
        app_runner=app.run,
        event_sink=RecordingLifecycleSink().emit,
        locale="en",
    )

    with pytest.raises(RuntimeError, match="Flet page configuration failed") as excinfo:
        await window.start(OverlayPresentationSnapshot(revision=1, blocks=[]))

    assert "native readiness failed" in str(excinfo.value.__cause__)
    assert window.startup_phase == "page_configured"
    assert app.page.window.visible is False
    await window.close()


def test_desktop_overlay_startup_timeouts_cover_hidden_window_handshake() -> None:
    from puripuly_heart.app.services.overlay.overlay_application import (
        OVERLAY_STARTUP_TIMEOUT_MS,
    )

    nested_handshake_s = (
        desktop_overlay.DESKTOP_OVERLAY_WAIT_UNTIL_READY_TIMEOUT_S
        + 0.5
        + (desktop_window_zorder.WINDOWS_WINDOW_VISIBILITY_TIMEOUT_S * 2)
        + 2.0
    )
    assert desktop_overlay.DESKTOP_OVERLAY_RENDERER_STARTUP_TIMEOUT_S >= nested_handshake_s
    assert OVERLAY_STARTUP_TIMEOUT_MS / 1000.0 > (
        desktop_overlay.DESKTOP_OVERLAY_RENDERER_STARTUP_TIMEOUT_S + 1.0
    )


@pytest.mark.asyncio
async def test_desktop_overlay_wait_until_ready_timeout_reports_page_configured_phase() -> None:
    app = FakeFletApp()

    async def hang_ready() -> None:
        await asyncio.Event().wait()

    app.page.window.wait_until_ready_to_show = hang_ready
    window = desktop_overlay.FletDesktopRendererWindow(
        app_runner=app.run,
        event_sink=RecordingLifecycleSink().emit,
        locale="en",
        wait_until_ready_timeout_s=0.05,
    )

    with pytest.raises(RuntimeError, match="Flet page configuration failed") as excinfo:
        await window.start(OverlayPresentationSnapshot(revision=1, blocks=[]))

    assert "native window was not ready to show" in str(excinfo.value.__cause__)
    assert window.startup_phase == "page_configured"
    assert app.page.window.visible is False
    await window.close()


@pytest.mark.asyncio
async def test_desktop_overlay_startup_timeout_after_page_exists_reports_phase() -> None:
    app = FakeFletApp()

    async def hang_ready() -> None:
        await asyncio.Event().wait()

    app.page.window.wait_until_ready_to_show = hang_ready
    window = desktop_overlay.FletDesktopRendererWindow(
        app_runner=app.run,
        event_sink=RecordingLifecycleSink().emit,
        locale="en",
        startup_timeout_s=0.05,
        wait_until_ready_timeout_s=5.0,
    )

    with pytest.raises(RuntimeError, match="startup timed out before ready"):
        await window.start(OverlayPresentationSnapshot(revision=1, blocks=[]))

    assert window.startup_phase == "page_configured"
    await window.close()


@pytest.mark.asyncio
async def test_desktop_overlay_reasserts_topmost_after_locked_render_update() -> None:
    app = FakeFletApp()
    observations: list[tuple[bool | None, int, dict[str, object]]] = []
    port = RecordingWindowZOrderPort(
        on_reassert=lambda: observations.append(
            (
                app.page.window.ignore_mouse_events,
                app.page.update_calls,
                dict(app.page.render_snapshots[-1]),
            )
        )
    )
    window = desktop_overlay.FletDesktopRendererWindow(
        app_runner=app.run,
        event_sink=RecordingLifecycleSink().emit,
        locale="en",
        window_z_order_port=port,
        window_process_info_provider=lambda: (4321, None),
    )

    try:
        await window.start(OverlayPresentationSnapshot(revision=1, blocks=[]))
        updates_before_lock = app.page.update_calls

        await window.dispatch_runtime_control(
            {"command": "set_interaction_mode", "mode": "pass_through"}
        )
        z_order_task = window._window_z_order_task
        if z_order_task is not None:
            await z_order_task
        await window.dispatch_runtime_control(
            {"command": "set_interaction_mode", "mode": "pass_through"}
        )
        await window.dispatch_runtime_control({"command": "set_interaction_mode", "mode": "edit"})

        assert observations == [
            (
                True,
                updates_before_lock + 1,
                {
                    "ignore_mouse_events": True,
                    "texts": set(),
                    "has_drag_area": True,
                    "card_count": 0,
                },
            )
        ]
        assert port.reassert_calls == 1
    finally:
        await window.close()

    assert port.close_calls == 1


@pytest.mark.asyncio
async def test_desktop_overlay_keeps_locked_state_when_zorder_port_fails(
    caplog: pytest.LogCaptureFixture,
) -> None:
    app = FakeFletApp()
    sink = RecordingLifecycleSink()
    port = RecordingWindowZOrderPort(error=RuntimeError("z-order failed"))
    window = desktop_overlay.FletDesktopRendererWindow(
        app_runner=app.run,
        event_sink=sink.emit,
        locale="en",
        window_z_order_port=port,
        window_process_info_provider=lambda: (4321, None),
    )

    try:
        await window.start(OverlayPresentationSnapshot(revision=1, blocks=[]))

        with caplog.at_level(logging.WARNING):
            await window.dispatch_runtime_control(
                {"command": "set_interaction_mode", "mode": "pass_through"}
            )
            z_order_task = window._window_z_order_task
            if z_order_task is not None:
                await z_order_task

        assert app.page.window.ignore_mouse_events is True
        assert sink.events[-1] == {
            "type": "overlay_event",
            "payload": {"event": "interaction_mode_changed", "mode": "pass_through"},
        }
        assert "reason=port_error exception_type=RuntimeError" in caplog.text
    finally:
        await window.close()


@pytest.mark.asyncio
async def test_desktop_overlay_cancels_stale_zorder_before_edit_event() -> None:
    class BlockingWindowZOrderPort(RecordingWindowZOrderPort):
        def __init__(self) -> None:
            super().__init__()
            self.started = asyncio.Event()
            self.cancelled = asyncio.Event()

        async def reassert_topmost_after_click_through(
            self,
        ) -> desktop_window_zorder.WindowZOrderResult:
            self.reassert_calls += 1
            self.started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.cancelled.set()
                raise

    app = FakeFletApp()
    sink = RecordingLifecycleSink()
    port = BlockingWindowZOrderPort()
    window = desktop_overlay.FletDesktopRendererWindow(
        app_runner=app.run,
        event_sink=sink.emit,
        locale="en",
        window_z_order_port=port,
        window_process_info_provider=lambda: (4321, None),
    )

    try:
        await window.start(OverlayPresentationSnapshot(revision=1, blocks=[]))
        await window.dispatch_runtime_control(
            {"command": "set_interaction_mode", "mode": "pass_through"}
        )
        await port.started.wait()
        await window.dispatch_runtime_control({"command": "set_interaction_mode", "mode": "edit"})

        assert port.cancelled.is_set()
        assert window._window_z_order_task is None
        assert app.page.window.ignore_mouse_events is False
        assert [event["payload"]["mode"] for event in sink.events] == [
            "pass_through",
            "edit",
        ]
    finally:
        await window.close()


@pytest.mark.asyncio
async def test_desktop_overlay_rejects_late_page_after_close() -> None:
    app = FakeFletApp()
    port = RecordingWindowZOrderPort()
    window = desktop_overlay.FletDesktopRendererWindow(
        app_runner=app.run,
        window_z_order_port=port,
        window_process_info_provider=lambda: (4321, None),
    )

    await window.close()
    await window._handle_page(app.page)

    assert window._page is None
    assert app.page.window.close_calls == 1
    assert port.bound_pids == []
    assert port.close_calls == 1


@pytest.mark.asyncio
async def test_desktop_overlay_drops_mode_transition_queued_before_close() -> None:
    app = FakeFletApp()
    sink = RecordingLifecycleSink()
    port = RecordingWindowZOrderPort()
    window = desktop_overlay.FletDesktopRendererWindow(
        app_runner=app.run,
        event_sink=sink.emit,
        window_z_order_port=port,
        window_process_info_provider=lambda: (4321, None),
    )
    await window.start(OverlayPresentationSnapshot(revision=1, blocks=[]))
    updates_before_close = app.page.update_calls

    await window._interaction_mode_lock.acquire()
    transition_task = asyncio.create_task(
        window.dispatch_runtime_control({"command": "set_interaction_mode", "mode": "pass_through"})
    )
    await asyncio.sleep(0)
    close_task = asyncio.create_task(window.close())
    await asyncio.sleep(0)
    window._interaction_mode_lock.release()
    await asyncio.gather(transition_task, close_task)

    assert app.page.update_calls == updates_before_close
    assert sink.events == []
    assert window._window_z_order_task is None
    assert port.reassert_calls == 0
    assert port.close_calls == 1


@pytest.mark.asyncio
async def test_desktop_overlay_detail_logs_startup_render_and_snapshot_updates(
    capsys: pytest.CaptureFixture[str],
) -> None:
    app = FakeFletApp()
    window = desktop_overlay.FletDesktopRendererWindow(
        app_runner=app.run,
        event_sink=RecordingLifecycleSink().emit,
        locale="en",
        bounds_debounce_s=0.01,
    )
    residual = window.prime_startup_runtime_controls(
        (
            {"logging_mode": "detailed"},
            {"command": "set_interaction_mode", "mode": "pass_through"},
            {
                "command": "apply_window_bounds",
                "x": 320,
                "y": 720,
                "width": 1344,
                "height": 320,
            },
        )
    )

    try:
        assert residual == ()
        await window.start(OverlayPresentationSnapshot(revision=1, blocks=[]))

        startup_output = capsys.readouterr().out
        assert "[DesktopOverlay][Detail] render" in startup_output
        assert "revision=1" in startup_output
        assert "interaction_mode=edit" in startup_output
        assert "surface_visible=True" in startup_output
        assert "line_count=0" in startup_output
        assert "content_kind=drag_area_with_empty_lock_action" in startup_output
        assert "window=1344x320" in startup_output
        assert "bounds_epoch" not in startup_output

        await window.dispatch_runtime_control(
            {"command": "set_interaction_mode", "mode": "pass_through"}
        )

        await window.dispatch_snapshot(
            OverlayPresentationSnapshot(
                revision=2,
                blocks=[
                    _block(
                        "peer-translated",
                        channel="peer",
                        block_variant="finalized",
                        appearance_seq=1,
                        primary_text="좋아요",
                        secondary_text="Sounds good",
                        secondary_enabled=True,
                    )
                ],
            )
        )

        update_output = capsys.readouterr().out
        assert "[DesktopOverlay][Detail] snapshot_update revision=2 blocks=1" in update_output
        assert "[DesktopOverlay][Detail] render" in update_output
        assert "revision=2" in update_output
        assert "surface_visible=True" in update_output
        assert "line_count=2" in update_output
        assert "content_kind=caption_surface" in update_output
        assert "bounds_epoch" not in update_output

        await window.dispatch_runtime_control({"logging_mode": "basic"})
        capsys.readouterr()
        await window.dispatch_snapshot(OverlayPresentationSnapshot(revision=3, blocks=[]))

        assert capsys.readouterr().out == ""
    finally:
        await window.close()


@pytest.mark.asyncio
async def test_desktop_overlay_detail_logs_layout_diagnostics_only_when_detailed(
    capsys: pytest.CaptureFixture[str],
) -> None:
    app = FakeFletApp()
    window = desktop_overlay.FletDesktopRendererWindow(
        app_runner=app.run,
        event_sink=RecordingLifecycleSink().emit,
        locale="en",
        bounds_debounce_s=0.01,
    )
    window.prime_startup_runtime_controls(
        (
            {"logging_mode": "detailed"},
            {
                "command": "apply_window_bounds",
                "x": 320,
                "y": 720,
                "width": 1344,
                "height": 336,
            },
        )
    )
    short_peer = _block(
        "peer-translated",
        channel="peer",
        block_variant="finalized",
        appearance_seq=1,
        primary_text="응",
        secondary_text="Sounds good",
        secondary_enabled=True,
    )
    long_peer = _block(
        "peer-translated",
        channel="peer",
        block_variant="finalized",
        appearance_seq=1,
        primary_text="This visible caption is intentionally long enough to widen the card.",
        secondary_text="Sounds good",
        secondary_enabled=True,
    )

    try:
        await window.start(OverlayPresentationSnapshot(revision=1, blocks=[]))
        await window.dispatch_runtime_control(
            {"command": "set_interaction_mode", "mode": "pass_through"}
        )
        capsys.readouterr()

        await window.dispatch_snapshot(OverlayPresentationSnapshot(revision=2, blocks=[short_peer]))

        first_output = capsys.readouterr().out
        assert "snapshot_update revision=2 blocks=1" in first_output
        assert "peer-translated" not in first_output
        assert "peer:peer-translated" not in first_output
        assert "Sounds good" not in first_output
        assert "render_transition revision=2" in first_output
        assert "content_kind transparent_host->caption_surface" in first_output
        assert "surface_visible False->True" in first_output
        assert "slot_count 0->1" in first_output
        assert "line_count 0->2" in first_output
        assert "render_width revision=2 slot=0" in first_output
        assert "key=" not in first_output
        assert "previous_floor=0.0" in first_output

        await window.dispatch_snapshot(OverlayPresentationSnapshot(revision=3, blocks=[long_peer]))
        capsys.readouterr()

        await window.dispatch_snapshot(OverlayPresentationSnapshot(revision=4, blocks=[short_peer]))

        floor_output = capsys.readouterr().out
        assert "render_width revision=4 slot=0" in floor_output
        assert "floor_hit=True" in floor_output

        await window.dispatch_runtime_control({"logging_mode": "basic"})
        capsys.readouterr()
        await window.dispatch_snapshot(OverlayPresentationSnapshot(revision=5, blocks=[short_peer]))

        assert capsys.readouterr().out == ""
    finally:
        await window.close()


@pytest.mark.asyncio
async def test_desktop_overlay_flet_window_starts_frameless_transparent_moving_empty_card() -> None:
    app = FakeFletApp()
    sink = RecordingLifecycleSink()
    window = desktop_overlay.FletDesktopRendererWindow(
        app_runner=app.run,
        event_sink=sink.emit,
        locale="en",
        bounds_debounce_s=0.01,
    )

    try:
        await window.start(OverlayPresentationSnapshot(revision=1, blocks=[]))

        page = app.page
        assert page.window.frameless is True
        assert page.title == "PuriPuly Overlay"
        assert page.window.icon == "icons/icon.ico"
        assert page.window.always_on_top is True
        assert page.window.shadow is False
        assert page.window.skip_task_bar is False
        assert page.window.resizable is False
        assert page.window.maximizable is False
        assert page.window.bgcolor == ft.Colors.TRANSPARENT
        assert page.bgcolor == ft.Colors.TRANSPARENT
        assert page.window.ignore_mouse_events is False
        assert page.window.title_bar_hidden is None
        assert page.window.title_bar_buttons_hidden is None
        assert page.window.on_event is not None
        assert page.window.start_resizing_calls == 0
        assert page.window.width == _DESKTOP_CAPTION_SIZE_PRESETS["medium"].window_width
        assert page.window.height == _DESKTOP_CAPTION_SIZE_PRESETS["medium"].window_height

        lock_label = desktop_overlay.t_for_locale(
            "en", "settings.overlay.desktop.empty_state.action.lock"
        )
        assert _page_text_values(page) == {lock_label}
        _assert_no_overlay_local_renderer_text(page)
        assert [button.content for button in _text_buttons(page)] == [lock_label]
        assert _page_contains_control_type(page, ft.WindowDragArea)
        cards = _caption_card_controls(page)
        assert len(cards) == 1
        assert cards[0].bgcolor == "#99000000"
    finally:
        await window.close()


@pytest.mark.asyncio
async def test_desktop_overlay_empty_pass_through_keeps_window_drag_area_content_visible() -> None:
    app = FakeFletApp()
    window = desktop_overlay.FletDesktopRendererWindow(
        app_runner=app.run,
        event_sink=RecordingLifecycleSink().emit,
        locale="en",
        bounds_debounce_s=0.01,
    )

    try:
        await window.start(OverlayPresentationSnapshot(revision=1, blocks=[]))
        await window.dispatch_runtime_control(
            {"command": "set_interaction_mode", "mode": "pass_through"}
        )
        if app.page.tasks:
            await asyncio.gather(*app.page.tasks)

        model = window._retained_caption_surface
        assert model is not None
        assert model.drag_content_host is not None
        assert model.drag_content_host.visible is True
        assert model.caption_surface.visible is False
        drag_area = next(
            item
            for control in app.page.controls
            for item in _walk_control_tree(control)
            if isinstance(item, ft.WindowDragArea)
        )
        assert drag_area.content is model.drag_content_host
        drag_area.before_update()
        assert app.page.window.ignore_mouse_events is True
        assert app.page.add_calls == 1
        assert app.page.clean_calls == 0
    finally:
        await window.close()


@pytest.mark.asyncio
async def test_desktop_overlay_empty_moving_state_renders_text_only_lock_action() -> None:
    app = FakeFletApp()
    sink = RecordingLifecycleSink()
    window = desktop_overlay.FletDesktopRendererWindow(
        app_runner=app.run,
        event_sink=sink.emit,
        locale="ko",
        bounds_debounce_s=0.01,
    )

    try:
        await window.start(OverlayPresentationSnapshot(revision=1, blocks=[]))

        lock_label = desktop_overlay.t_for_locale(
            "ko", "settings.overlay.desktop.empty_state.action.lock"
        )
        assert lock_label in _page_text_values(app.page)
        action = _find_text_button(app.page, lock_label)
        assert action.width >= 44
        assert action.height >= 44
        assert action.tooltip == lock_label
        assert action.style.bgcolor == ft.Colors.TRANSPARENT
        assert action.style.overlay_color == ft.Colors.TRANSPARENT
        assert action.style.elevation == 0
        assert action.style.animation_duration == 0
        assert action.style.color[ft.ControlState.DEFAULT] == "#FFF8F4"
        assert action.style.color[ft.ControlState.FOCUSED] == "#FF6B6B"
        assert action.style.color[ft.ControlState.HOVERED] == "#FF6B6B"
        action_text_style = action.style.text_style
        assert action_text_style.shadow is None
        action_padding = action.style.padding
        required_label_width = desktop_overlay._estimated_caption_line_width(
            lock_label,
            int(action_text_style.size),
        )
        assert action.width >= required_label_width + action_padding.left + action_padding.right
        assert len(_text_buttons(app.page)) == 1
        assert _page_contains_control_type(app.page, ft.WindowDragArea)
        assert len(_caption_card_controls(app.page)) == 1
        assert action.width < app.page.window.width
        assert action.height < app.page.window.height
        root = app.page.controls[0]
        stack = root.content
        assert isinstance(stack, ft.Stack)
        assert isinstance(stack.controls[0], ft.WindowDragArea)
        assert stack.controls[1] is action
    finally:
        await window.close()


@pytest.mark.asyncio
async def test_desktop_overlay_empty_lock_action_hides_when_captions_arrive() -> None:
    app = FakeFletApp()
    sink = RecordingLifecycleSink()
    window = desktop_overlay.FletDesktopRendererWindow(
        app_runner=app.run,
        event_sink=sink.emit,
        locale="ko",
        bounds_debounce_s=0.01,
    )
    captions = OverlayPresentationSnapshot(
        revision=2,
        blocks=[
            _block(
                "peer-translated",
                channel="peer",
                block_variant="finalized",
                appearance_seq=1,
                primary_text="좋아요",
                secondary_text="Sounds good",
                secondary_enabled=True,
            )
        ],
    )

    try:
        await window.start(OverlayPresentationSnapshot(revision=1, blocks=[]))
        lock_label = desktop_overlay.t_for_locale(
            "ko", "settings.overlay.desktop.empty_state.action.lock"
        )
        assert lock_label in _page_text_values(app.page)

        await window.dispatch_snapshot(captions)

        assert lock_label not in _page_text_values(app.page)
        assert {"좋아요", "Sounds good"} <= _page_text_values(app.page)
        assert len(_text_buttons(app.page)) == 1
        assert _text_buttons(app.page)[0].visible is False
        assert _page_contains_control_type(app.page, ft.WindowDragArea)
    finally:
        await window.close()


@pytest.mark.asyncio
async def test_desktop_overlay_display_matrix_moving_and_locked_with_captions() -> None:
    app = FakeFletApp()
    sink = RecordingLifecycleSink()
    window = desktop_overlay.FletDesktopRendererWindow(
        app_runner=app.run,
        event_sink=sink.emit,
        locale="en",
        bounds_debounce_s=0.01,
    )
    captions = OverlayPresentationSnapshot(
        revision=2,
        blocks=[
            _block(
                "peer-translated",
                channel="peer",
                block_variant="finalized",
                appearance_seq=1,
                primary_text="좋아요",
                secondary_text="Sounds good",
                secondary_enabled=True,
            )
        ],
    )

    try:
        await window.start(captions)

        assert app.page.window.ignore_mouse_events is False
        assert _page_contains_control_type(app.page, ft.WindowDragArea)
        assert {"좋아요", "Sounds good"} <= _page_text_values(app.page)
        assert len(_caption_card_controls(app.page)) == 1

        chrome_before_lock = (
            app.page.window.frameless,
            app.page.window.shadow,
            app.page.window.resizable,
            app.page.window.always_on_top,
            app.page.window.title_bar_hidden,
            app.page.window.title_bar_buttons_hidden,
        )

        await window.dispatch_runtime_control(
            {"command": "set_interaction_mode", "mode": "pass_through"}
        )
        if app.page.tasks:
            await asyncio.gather(*app.page.tasks)

        assert app.page.window.ignore_mouse_events is True
        assert (
            app.page.window.frameless,
            app.page.window.shadow,
            app.page.window.resizable,
            app.page.window.always_on_top,
            app.page.window.title_bar_hidden,
            app.page.window.title_bar_buttons_hidden,
        ) == chrome_before_lock
        assert {"좋아요", "Sounds good"} <= _page_text_values(app.page)
        assert _page_contains_control_type(app.page, ft.WindowDragArea)
        assert len(_caption_card_controls(app.page)) == 1
        _assert_no_overlay_local_renderer_text(app.page)
        assert sink.events[-1] == {
            "type": "overlay_event",
            "payload": {"event": "interaction_mode_changed", "mode": "pass_through"},
        }
    finally:
        await window.close()


@pytest.mark.asyncio
async def test_desktop_overlay_render_page_applies_grow_only_card_width_to_visible_surface() -> (
    None
):
    app = FakeFletApp()
    window = desktop_overlay.FletDesktopRendererWindow(
        app_runner=app.run,
        event_sink=RecordingLifecycleSink().emit,
        locale="en",
        bounds_debounce_s=0.01,
    )
    window.prime_startup_runtime_controls(
        (
            {
                "command": "apply_window_bounds",
                "x": 320,
                "y": 720,
                "width": 1344,
                "height": 336,
            },
        )
    )

    same_short_block = _block(
        "same-peer",
        channel="peer",
        block_variant="finalized",
        appearance_seq=1,
        primary_text="응",
    )
    same_long_block = _block(
        "same-peer",
        channel="peer",
        block_variant="finalized",
        appearance_seq=1,
        primary_text=(
            "This visible caption is intentionally long enough to widen "
            "the dynamic card beyond its minimum width."
        ),
    )

    try:
        await window.start(OverlayPresentationSnapshot(revision=1, blocks=[]))
        await window.dispatch_runtime_control(
            {"command": "set_interaction_mode", "mode": "pass_through"}
        )

        await window.dispatch_snapshot(
            OverlayPresentationSnapshot(revision=2, blocks=[same_short_block])
        )
        short_width = _caption_card_controls(app.page)[0].width

        await window.dispatch_snapshot(
            OverlayPresentationSnapshot(revision=3, blocks=[same_long_block])
        )
        long_width = _caption_card_controls(app.page)[0].width

        await window.dispatch_snapshot(
            OverlayPresentationSnapshot(revision=4, blocks=[same_short_block])
        )
        shortened_width = _caption_card_controls(app.page)[0].width

        assert short_width < long_width
        assert shortened_width == pytest.approx(long_width)
    finally:
        await window.close()


@pytest.mark.asyncio
async def test_desktop_overlay_locked_size_change_clears_stale_card_width_floor() -> None:
    app = FakeFletApp()
    window = desktop_overlay.FletDesktopRendererWindow(
        app_runner=app.run,
        event_sink=RecordingLifecycleSink().emit,
        locale="en",
        bounds_debounce_s=0.01,
    )
    window.prime_startup_runtime_controls(
        (
            {
                "command": "apply_window_bounds",
                "x": 320,
                "y": 720,
                "width": 1344,
                "height": 336,
            },
        )
    )
    same_long_block = _block(
        "same-peer",
        channel="peer",
        block_variant="finalized",
        appearance_seq=1,
        primary_text=(
            "This visible caption is intentionally long enough to widen "
            "the dynamic card to the current overlay width."
        ),
    )
    same_short_block = _block(
        "same-peer",
        channel="peer",
        block_variant="finalized",
        appearance_seq=1,
        primary_text="응",
    )

    try:
        await window.start(OverlayPresentationSnapshot(revision=1, blocks=[]))
        await window.dispatch_runtime_control(
            {"command": "set_interaction_mode", "mode": "pass_through"}
        )
        await window.dispatch_snapshot(
            OverlayPresentationSnapshot(revision=2, blocks=[same_long_block])
        )
        await window.dispatch_snapshot(
            OverlayPresentationSnapshot(revision=3, blocks=[same_short_block])
        )
        assert _caption_card_controls(app.page)[0].width == pytest.approx(1344)

        await window.dispatch_runtime_control(
            {
                "command": "apply_window_bounds",
                "x": 420,
                "y": 780,
                "width": 1152,
                "height": 288,
            }
        )

        assert app.page.window.width == 1152
        assert app.page.window.height == 288
        assert _caption_card_controls(app.page)[0].width == pytest.approx(320.0)
    finally:
        await window.close()


def test_desktop_overlay_window_size_change_detection_keeps_position_only_cache() -> None:
    app = FakeFletApp()
    app.page.window.width = 1152
    app.page.window.height = 288

    assert not desktop_overlay._page_window_size_differs_from_bounds(  # noqa: SLF001 - verify invalidation boundary
        app.page,
        {"x": 400, "y": 500, "width": 1152, "height": 288},
    )
    assert desktop_overlay._page_window_size_differs_from_bounds(  # noqa: SLF001 - verify invalidation boundary
        app.page,
        {"x": 400, "y": 500, "width": 1153, "height": 288},
    )
    assert desktop_overlay._page_window_size_differs_from_bounds(  # noqa: SLF001 - verify invalidation boundary
        app.page,
        {"x": 400, "y": 500, "width": 1152, "height": 289},
    )


@pytest.mark.asyncio
async def test_desktop_overlay_empty_lock_action_switches_to_pass_through() -> None:
    app = FakeFletApp()
    sink = RecordingLifecycleSink()
    window = desktop_overlay.FletDesktopRendererWindow(
        app_runner=app.run,
        event_sink=sink.emit,
        locale="en",
        bounds_debounce_s=0.01,
    )

    try:
        await window.start(OverlayPresentationSnapshot(revision=1, blocks=[]))
        action = _find_text_button(
            app.page,
            desktop_overlay.t_for_locale("en", "settings.overlay.desktop.empty_state.action.lock"),
        )

        action.on_click(None)
        if app.page.tasks:
            await asyncio.gather(*app.page.tasks)

        assert app.page.window.ignore_mouse_events is True
        assert _page_text_values(app.page) == set()
        assert _caption_card_controls(app.page) == []
        assert _page_contains_control_type(app.page, ft.WindowDragArea)
        assert sink.events[-1] == {
            "type": "overlay_event",
            "payload": {"event": "interaction_mode_changed", "mode": "pass_through"},
        }
    finally:
        await window.close()


@pytest.mark.asyncio
async def test_desktop_overlay_display_matrix_locked_no_captions_is_fully_transparent() -> None:
    app = FakeFletApp()
    sink = RecordingLifecycleSink()
    window = desktop_overlay.FletDesktopRendererWindow(
        app_runner=app.run,
        event_sink=sink.emit,
        locale="en",
        bounds_debounce_s=0.01,
    )

    try:
        await window.start(OverlayPresentationSnapshot(revision=1, blocks=[]))

        lock_label = desktop_overlay.t_for_locale(
            "en", "settings.overlay.desktop.empty_state.action.lock"
        )
        assert app.page.window.frameless is True
        assert app.page.window.shadow is False
        assert app.page.window.resizable is False
        assert app.page.window.always_on_top is True
        assert app.page.window.ignore_mouse_events is False
        assert lock_label in _page_text_values(app.page)
        assert len(_text_buttons(app.page)) == 1
        assert _page_contains_control_type(app.page, ft.WindowDragArea)
        assert len(_caption_card_controls(app.page)) == 1

        await window.dispatch_runtime_control(
            {"command": "set_interaction_mode", "mode": "pass_through"}
        )

        assert app.page.window.title_bar_hidden is None
        assert app.page.window.title_bar_buttons_hidden is None
        assert app.page.window.resizable is False
        assert app.page.window.frameless is True
        assert app.page.window.shadow is False
        assert app.page.window.always_on_top is True
        assert app.page.window.ignore_mouse_events is True
        assert _page_text_values(app.page) == set()
        assert _caption_card_controls(app.page) == []
        assert _page_contains_control_type(app.page, ft.WindowDragArea)

        await window.dispatch_runtime_control({"command": "set_interaction_mode", "mode": "edit"})

        assert app.page.window.title_bar_hidden is None
        assert app.page.window.title_bar_buttons_hidden is None
        assert app.page.window.resizable is False
        assert app.page.window.ignore_mouse_events is False
        assert lock_label in _page_text_values(app.page)
        assert len(_text_buttons(app.page)) == 1
        assert _page_contains_control_type(app.page, ft.WindowDragArea)
        assert len(_caption_card_controls(app.page)) == 1
    finally:
        await window.close()


@pytest.mark.asyncio
async def test_desktop_overlay_preset_visual_tokens_match_product_table() -> None:
    expected = {
        "tiny": (640, 160, 20, 12, 10, 2, 10, 4, 620),
        "xsmall": (960, 240, 29, 18, 14, 6, 12, 6, 932),
        "small": (1152, 288, 35, 21, 18, 8, 14, 8, 1116),
        "medium": (1344, 336, 41, 25, 22, 10, 16, 10, 1300),
        "large": (1600, 400, 50, 30, 26, 12, 18, 12, 1548),
        "xlarge": (1792, 448, 56, 34, 30, 14, 20, 14, 1732),
    }

    for preset_id, (
        width,
        height,
        primary,
        secondary,
        padding_h,
        padding_v,
        radius,
        slot_gap,
        text_width,
    ) in expected.items():
        plan = desktop_overlay.build_desktop_caption_plan(
            OverlayPresentationSnapshot(
                blocks=[
                    _block(
                        f"{preset_id}-caption",
                        channel="peer",
                        block_variant="finalized",
                        appearance_seq=1,
                        primary_text="caption",
                    )
                ]
            ),
            window_width=width,
            window_height=height,
            interaction_mode="edit",
        )

        assert plan.size_preset == preset_id
        assert plan.window_width == width
        assert plan.window_height == height
        assert plan.primary_font_size == primary
        assert plan.secondary_font_size == secondary
        assert plan.padding_horizontal == padding_h
        assert plan.padding_vertical == padding_v
        assert plan.text_width == text_width
        assert plan.border_radius == radius
        assert plan.slot_gap == slot_gap
        assert plan.secondary_line_max_lines == 1
        expected_slot_height = (height - slot_gap) / 2
        assert plan.slot_height == pytest.approx(expected_slot_height)
        assert plan.primary_region_height == pytest.approx(primary * 1.24 * 2)
        assert plan.secondary_region_height == pytest.approx(secondary * 1.24)
        assert (
            plan.slot_height
            - (plan.padding_vertical * 2)
            - plan.primary_region_height
            - plan.secondary_region_height
        ) >= 5

        surface = desktop_overlay.build_desktop_caption_surface(plan)
        assert isinstance(surface.content, ft.Stack)
        slot_column = surface.content.controls[-1]
        outer_slot = slot_column.controls[0]
        inner_card = outer_slot.content
        assert outer_slot.height == pytest.approx(expected_slot_height)
        assert inner_card.padding.left == padding_h
        assert inner_card.padding.top == padding_v
        assert surface.border_radius == radius


@pytest.mark.asyncio
async def test_desktop_overlay_shipping_surface_has_no_overlay_local_controls() -> None:
    app = FakeFletApp()
    sink = RecordingLifecycleSink()
    window = desktop_overlay.FletDesktopRendererWindow(
        app_runner=app.run,
        event_sink=sink.emit,
        locale="en",
        bounds_debounce_s=0.01,
    )

    try:
        await window.start(OverlayPresentationSnapshot(revision=1, blocks=[]))
        assert app.page.run_task_calls == 0
        for task in list(app.page.tasks):
            await task
        assert sink.events == []
        _assert_no_overlay_local_renderer_text(app.page)
        buttons = _text_buttons(app.page)
        assert [button.content for button in buttons] == [
            desktop_overlay.t_for_locale("en", "settings.overlay.desktop.empty_state.action.lock")
        ]
        assert not any(
            isinstance(item, ft.ElevatedButton)
            for control in app.page.controls
            for item in _walk_control_tree(control)
        )
    finally:
        await window.close()


@pytest.mark.asyncio
async def test_desktop_overlay_interaction_mode_bounds_and_visual_runtime_controls_validate_atomically() -> (
    None
):
    app = FakeFletApp()
    sink = RecordingLifecycleSink()
    window = desktop_overlay.FletDesktopRendererWindow(
        app_runner=app.run,
        event_sink=sink.emit,
        locale="en",
        bounds_debounce_s=0.01,
    )

    try:
        await window.start(OverlayPresentationSnapshot(revision=1, blocks=[]))
        await window.dispatch_runtime_control(
            {
                "command": "apply_window_bounds",
                "x": 320,
                "y": 720,
                "width": 1280,
                "height": 330,
            }
        )
        await window.dispatch_runtime_control(
            {
                "command": "apply_visual_config",
                "text_scale": 1.25,
                "background_alpha": 0.65,
                "outline_width": 2.5,
            }
        )
        await window.dispatch_runtime_control(
            {"command": "set_interaction_mode", "mode": "pass_through"}
        )
        state_after_valid_controls = (
            app.page.window.left,
            app.page.window.top,
            app.page.window.width,
            app.page.window.height,
            app.page.window.ignore_mouse_events,
            _page_text_values(app.page),
        )

        await window.dispatch_runtime_control({"command": "set_interaction_mode", "mode": "bogus"})
        await window.dispatch_runtime_control(
            {"command": "apply_window_bounds", "x": 1, "y": 2, "width": 0, "height": 0}
        )
        await window.dispatch_runtime_control(
            {
                "command": "apply_visual_config",
                "text_scale": True,
                "background_alpha": 2.0,
                "outline_width": -1.0,
            }
        )
        await window.dispatch_runtime_control({"command": "unknown_desktop_command"})

        assert (
            app.page.window.left,
            app.page.window.top,
            app.page.window.width,
            app.page.window.height,
            app.page.window.ignore_mouse_events,
            _page_text_values(app.page),
        ) == state_after_valid_controls
        assert sink.events[-1] == {
            "type": "overlay_event",
            "payload": {"event": "interaction_mode_changed", "mode": "pass_through"},
        }
    finally:
        await window.close()


@pytest.mark.asyncio
async def test_desktop_overlay_post_start_paths_update_retained_controls_in_place() -> None:
    app = FakeFletApp()
    window = desktop_overlay.FletDesktopRendererWindow(
        app_runner=app.run,
        event_sink=RecordingLifecycleSink().emit,
        locale="en",
    )
    initial_snapshot = OverlayPresentationSnapshot(
        revision=1,
        blocks=[
            _block(
                "self-active",
                channel="self",
                block_variant="active_self",
                appearance_seq=1,
                primary_text="self transcript",
            )
        ],
    )
    peer_translation = OverlayPresentationSnapshot(
        revision=2,
        blocks=[
            _block(
                "self-finalized",
                channel="self",
                block_variant="finalized",
                appearance_seq=1,
                primary_text="self transcript",
                secondary_text="self translation",
                secondary_enabled=True,
            ),
            _block(
                "peer-finalized",
                channel="peer",
                block_variant="finalized",
                appearance_seq=2,
                primary_text="peer translation",
                secondary_text="peer original",
                secondary_enabled=True,
            ),
        ],
    )

    try:
        await window.start(initial_snapshot)
        model = window._retained_caption_surface
        assert model is not None
        identities = (
            app.page.controls[0],
            model.caption_surface,
            *model.slot_containers,
            *model.primary_texts,
            *model.secondary_texts,
        )
        protected_writes = len(app.page.window.protected_writes)

        await window.dispatch_snapshot(peer_translation)
        await window.dispatch_runtime_control(
            {
                "command": "apply_visual_config",
                "text_scale": 1.25,
                "background_alpha": 0.5,
                "outline_width": None,
            }
        )
        await window.dispatch_runtime_control(
            {
                "command": "apply_window_bounds",
                "x": 320,
                "y": 720,
                "width": 1152,
                "height": 288,
            }
        )
        assert (app.page.window.left, app.page.window.top) == (320, 720)
        assert (app.page.window.width, app.page.window.height) == (1152, 288)

        await window.dispatch_snapshot(OverlayPresentationSnapshot(revision=3, blocks=[]))
        action = _find_text_button(
            app.page,
            desktop_overlay.t_for_locale("en", "settings.overlay.desktop.empty_state.action.lock"),
        )
        action.on_click(None)
        await asyncio.gather(*app.page.tasks)

        assert app.page.window.ignore_mouse_events is True
        assert model.caption_surface.visible is False
        await window.dispatch_runtime_control({"command": "set_interaction_mode", "mode": "edit"})
        assert app.page.window.ignore_mouse_events is False
        assert identities == (
            app.page.controls[0],
            model.caption_surface,
            *model.slot_containers,
            *model.primary_texts,
            *model.secondary_texts,
        )
        assert app.page.add_calls == 1
        assert app.page.clean_calls == 0
        assert len(app.page.window.protected_writes) == protected_writes
    finally:
        await window.close()


@pytest.mark.asyncio
async def test_desktop_overlay_window_bounds_events_debounce_zero_samples_and_programmatic_echoes() -> (
    None
):
    app = FakeFletApp()
    sink = RecordingLifecycleSink()
    window = desktop_overlay.FletDesktopRendererWindow(
        app_runner=app.run,
        event_sink=sink.emit,
        locale="en",
        bounds_debounce_s=0.01,
    )

    try:
        await window.start(OverlayPresentationSnapshot(revision=1, blocks=[]))
        assert callable(app.page.window.on_event)

        app.page.window.left = 0
        app.page.window.top = 0
        app.page.window.width = 0
        app.page.window.height = 0
        app.page.window.on_event(FakeWindowEvent(ft.WindowEventType.MOVE))
        await asyncio.sleep(0.03)
        assert sink.events == []

        app.page.window.left = 100
        app.page.window.top = 200
        app.page.window.width = 900
        app.page.window.height = 240
        app.page.window.on_event(FakeWindowEvent(ft.WindowEventType.MOVE))
        app.page.window.on_event(FakeWindowEvent(ft.WindowEventType.MOVED))
        await asyncio.sleep(0.03)
        assert sink.events == [
            {
                "type": "overlay_event",
                "payload": {
                    "event": "window_bounds_changed",
                    "source": "user",
                    "persist": True,
                    "generation": 1,
                    "x": 100,
                    "y": 200,
                    "width": 900,
                    "height": 240,
                },
            }
        ]

        await window.dispatch_runtime_control(
            {
                "command": "apply_window_bounds",
                "x": 320,
                "y": 720,
                "width": 1280,
                "height": 330,
            }
        )
        app.page.window.on_event(FakeWindowEvent(ft.WindowEventType.RESIZED))
        await asyncio.sleep(0.03)
        assert len(sink.events) == 1

        app.page.window.left = 300
        app.page.window.top = 400
        app.page.window.width = 1280
        app.page.window.height = 330
        app.page.window.on_event(FakeWindowEvent(ft.WindowEventType.MOVED))
        await window.dispatch_runtime_control(
            {
                "command": "apply_window_bounds",
                "x": 500,
                "y": 600,
                "width": 1280,
                "height": 330,
            }
        )
        await asyncio.sleep(0.03)
        assert len(sink.events) == 1

        app.page.window.left = 360
        app.page.window.top = 700
        app.page.window.width = 1280
        app.page.window.height = 330
        app.page.window.on_event(FakeWindowEvent(ft.WindowEventType.RESIZE))
        await asyncio.sleep(0.03)
        assert sink.events[-1]["payload"] == {
            "event": "window_bounds_changed",
            "source": "user",
            "persist": True,
            "generation": 1,
            "x": 360,
            "y": 700,
            "width": 1280,
            "height": 330,
        }
    finally:
        await window.close()


@pytest.mark.asyncio
async def test_desktop_overlay_shutdown_cancels_queued_bounds_callback_without_event() -> None:
    app = FakeFletApp()
    sink = RecordingLifecycleSink()
    window = desktop_overlay.FletDesktopRendererWindow(
        app_runner=app.run,
        event_sink=sink.emit,
        locale="en",
        bounds_debounce_s=0.01,
    )

    await window.start(OverlayPresentationSnapshot(revision=1, blocks=[]))
    assert callable(app.page.window.on_event)
    app.page.window.left = 120
    app.page.window.top = 240
    app.page.window.width = 900
    app.page.window.height = 240

    app.page.window.on_event(FakeWindowEvent(ft.WindowEventType.MOVED))
    await window.close()
    if app.page.tasks:
        await asyncio.gather(*app.page.tasks, return_exceptions=True)
    await asyncio.sleep(0.03)

    assert sink.events == []
    assert app.page.window.on_event is None
    bounds_task = window._bounds_sample_task  # noqa: SLF001 - assert shutdown cleanup
    assert bounds_task is None or bounds_task.done()


@pytest.mark.asyncio
async def test_desktop_overlay_bounds_programmatic_echo_gate_is_generation_based() -> None:
    app = FakeFletApp()
    sink = RecordingLifecycleSink()
    window = desktop_overlay.FletDesktopRendererWindow(
        app_runner=app.run,
        event_sink=sink.emit,
        locale="en",
        bounds_debounce_s=0.01,
    )

    try:
        await window.start(OverlayPresentationSnapshot(revision=1, blocks=[]))
        assert callable(app.page.window.on_event)
        await window.dispatch_runtime_control(
            {
                "command": "apply_window_bounds",
                "x": 320,
                "y": 720,
                "width": 1280,
                "height": 330,
            }
        )

        app.page.window.on_event(FakeWindowEvent(ft.WindowEventType.RESIZED))
        await asyncio.sleep(0.03)
        assert sink.events == []

        await window.dispatch_runtime_control(
            {
                "command": "apply_window_bounds",
                "x": 500,
                "y": 600,
                "width": 1280,
                "height": 330,
            }
        )
        await asyncio.sleep(0.30)
        app.page.window.on_event(FakeWindowEvent(ft.WindowEventType.RESIZED))
        await asyncio.sleep(0.03)
        assert sink.events == []

        app.page.window.left = 360
        app.page.window.top = 700
        app.page.window.width = 1280
        app.page.window.height = 330
        app.page.window.on_event(FakeWindowEvent(ft.WindowEventType.MOVED))
        await asyncio.sleep(0.03)

        assert sink.events == [
            {
                "type": "overlay_event",
                "payload": {
                    "event": "window_bounds_changed",
                    "source": "user",
                    "persist": True,
                    "generation": 1,
                    "x": 360,
                    "y": 700,
                    "width": 1280,
                    "height": 330,
                },
            }
        ]
    finally:
        await window.close()


@pytest.mark.asyncio
async def test_desktop_overlay_retired_generation_cannot_emit_bounds() -> None:
    app = FakeFletApp()
    sink = RecordingLifecycleSink()
    window = desktop_overlay.FletDesktopRendererWindow(
        app_runner=app.run,
        event_sink=sink.emit,
        locale="en",
        bounds_debounce_s=0.0,
    )

    await window.start(OverlayPresentationSnapshot(revision=1, blocks=[]))
    assert window.startup_generation == 1
    await window.close()

    app.closed.clear()
    await window.start(OverlayPresentationSnapshot(revision=2, blocks=[]))
    assert window.startup_generation == 2
    app.page.window.left = 100
    app.page.window.top = 200
    app.page.window.width = 900
    app.page.window.height = 240

    await window._schedule_bounds_sample(1)
    await asyncio.sleep(0)

    assert sink.events == []
    await window.close()


@pytest.mark.asyncio
async def test_desktop_overlay_drops_bounds_event_while_runtime_locked_after_unlock() -> None:
    app = FakeFletApp()
    sink = RecordingLifecycleSink()
    window = desktop_overlay.FletDesktopRendererWindow(
        app_runner=app.run,
        event_sink=sink.emit,
        locale="en",
        bounds_debounce_s=0.01,
    )

    try:
        await window.start(OverlayPresentationSnapshot(revision=1, blocks=[]))
        await window.dispatch_runtime_control(
            {"command": "set_interaction_mode", "mode": "pass_through"}
        )
        assert callable(app.page.window.on_event)

        app.page.window.left = 608
        app.page.window.top = 1117
        app.page.window.width = 1344
        app.page.window.height = 320
        app.page.window.on_event(FakeWindowEvent(ft.WindowEventType.MOVED))
        await window.dispatch_runtime_control({"command": "set_interaction_mode", "mode": "edit"})
        await asyncio.sleep(0.03)

        assert all(
            event.get("payload", {}).get("event") != "window_bounds_changed"
            for event in sink.events
        )
    finally:
        await window.close()


class FakeRendererWindow:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.closed = asyncio.Event()
        self.close_calls = 0
        self.snapshots: list[OverlayPresentationSnapshot] = []
        self.runtime_controls: list[dict[str, object]] = []
        self.primed_runtime_controls: list[dict[str, object]] = []

    def prime_startup_runtime_controls(
        self,
        payloads: tuple[dict[str, object], ...],
    ) -> tuple[dict[str, object], ...]:
        self.primed_runtime_controls.extend(dict(payload) for payload in payloads)
        return ()

    async def start(self, initial_snapshot: OverlayPresentationSnapshot) -> None:
        self.snapshots.append(initial_snapshot)
        self.started.set()

    async def run_until_closed(self) -> None:
        await self.closed.wait()

    async def close(self) -> None:
        self.close_calls += 1
        self.closed.set()

    async def dispatch_snapshot(self, snapshot: OverlayPresentationSnapshot) -> None:
        self.snapshots.append(snapshot)

    async def dispatch_runtime_control(self, payload: dict[str, object]) -> None:
        self.runtime_controls.append(dict(payload))

    async def advance_snapshot_history(self, snapshot: OverlayPresentationSnapshot) -> None:
        _ = snapshot

    def renderer_visual_state(self) -> dict[str, object]:
        return {
            "slot_count": 0,
            "line_count": 0,
            "surface_visible": False,
            "interaction_mode": "edit",
            "window_width": 1344,
            "window_height": 336,
        }

    def renderer_visual_state_for_snapshot(
        self,
        snapshot: OverlayPresentationSnapshot,
    ) -> dict[str, object]:
        _ = snapshot
        return self.renderer_visual_state()


class BatchingRendererWindow(FakeRendererWindow):
    def __init__(self) -> None:
        super().__init__()
        self.history_snapshots: list[OverlayPresentationSnapshot] = []
        self.rendered_snapshot = asyncio.Event()
        self.execution: list[str] = []

    async def dispatch_snapshot(self, snapshot: OverlayPresentationSnapshot) -> None:
        await super().dispatch_snapshot(snapshot)
        self.execution.append(f"snapshot:{snapshot.revision}")
        self.rendered_snapshot.set()

    async def dispatch_runtime_control(self, payload: dict[str, object]) -> None:
        await super().dispatch_runtime_control(payload)
        self.execution.append(f"control:{payload['command']}")

    async def advance_snapshot_history(self, snapshot: OverlayPresentationSnapshot) -> None:
        self.history_snapshots.append(snapshot)

    def renderer_visual_state(self) -> dict[str, object]:
        return {
            "slot_count": 1,
            "line_count": 1,
            "surface_visible": True,
            "interaction_mode": "locked",
            "window_width": 1344,
            "window_height": 336,
        }


class FailingBatchingRendererWindow(BatchingRendererWindow):
    async def dispatch_snapshot(self, snapshot: OverlayPresentationSnapshot) -> None:
        raise RuntimeError("must not enter persisted diagnostics")


class RecordingRendererDiagnosticPort:
    def __init__(self) -> None:
        self.records: list[dict[str, object]] = []
        self.closed = False
        self.requires_commit_acknowledgement = False

    async def emit(self, envelope: desktop_overlay.RendererDiagnosticEnvelope) -> None:
        self.records.append(dict(envelope.record))

    async def wait_for_commit_ack(
        self,
        acknowledgement: desktop_overlay.RendererCommitAcknowledgement,
    ) -> None:
        _ = acknowledgement

    async def close(self) -> None:
        self.closed = True


def _scheduled_snapshot(revision: int, text: str) -> OverlayPresentationSnapshot:
    return OverlayPresentationSnapshot(
        revision=revision,
        blocks=[
            _block(
                "scheduled-peer",
                channel="peer",
                block_variant="finalized",
                appearance_seq=1,
                primary_text=text,
            )
        ],
    )


@pytest.mark.asyncio
async def test_desktop_overlay_snapshot_batching_is_fifo_latest_only_and_reports_acknowledged_commit() -> (
    None
):
    token = "scheduled-batching-token"
    bridge = OverlayBridge(
        session_token=token,
        initial_snapshot=OverlayPresentationSnapshot(revision=1),
        heartbeat_interval_ms=20,
        desktop_runtime_controls_enabled=True,
    )
    await bridge.start()
    window = BatchingRendererWindow()
    port = RecordingRendererDiagnosticPort()
    renderer = desktop_overlay.DesktopOverlayRenderer(
        _manifest(bridge_url=bridge.url, session_token=token),
        window=window,
        lifecycle_sink=RecordingLifecycleSink(),
        parent_monitor=FakeParentMonitor(),
        diagnostic_port=port,
    )
    revision_ten = _scheduled_snapshot(10, "first valid")
    revision_nine = _scheduled_snapshot(9, "stale")
    revision_eleven = _scheduled_snapshot(11, "latest valid")
    try:
        await renderer.enqueue_snapshot(revision_ten)
        await renderer.enqueue_snapshot(revision_nine)
        await renderer.enqueue_snapshot(revision_eleven)
        await renderer.enqueue_snapshot(_scheduled_snapshot(11, "equal"))
        run_task = asyncio.create_task(renderer.run())
        await _next_bridge_event(bridge, expected_type="overlay_ready")
        await asyncio.wait_for(window.rendered_snapshot.wait(), timeout=1.0)

        assert [snapshot.revision for snapshot in window.history_snapshots] == [10]
        assert [snapshot.revision for snapshot in window.snapshots] == [1, 11]
        assert [
            (record["event_type"], record["renderer_revision"], record["actual_disposition"])
            for record in port.records
        ] == [
            ("receipt", 10, "superseded"),
            ("supersession", 10, "superseded"),
            ("receipt", 9, "stale"),
            ("stale", 9, "stale"),
            ("receipt", 11, "committed"),
            ("receipt", 11, "stale"),
            ("stale", 11, "stale"),
            ("render_start", 11, "committed"),
            ("render_commit", 11, "committed"),
        ]
        assert port.records[-1]["render_commit_acknowledged"] is False
        assert all(set(record) == set(port.records[0]) for record in port.records)

        await bridge.broadcast_shutdown()
        assert await asyncio.wait_for(run_task, timeout=1.0) == 0
        assert port.closed is True
    finally:
        await renderer.shutdown()
        await bridge.stop()


@pytest.mark.asyncio
async def test_desktop_overlay_snapshot_batching_matches_sequential_width_reference() -> None:
    short = _scheduled_snapshot(2, "응")
    long = _scheduled_snapshot(3, "This caption is long enough to grow the retained width floor.")
    final_short = _scheduled_snapshot(4, "응")
    sequential_app = FakeFletApp()
    sequential_window = desktop_overlay.FletDesktopRendererWindow(
        app_runner=sequential_app.run,
        event_sink=RecordingLifecycleSink().emit,
        locale="en",
    )
    await sequential_window.start(OverlayPresentationSnapshot(revision=1, blocks=[]))
    await sequential_window.dispatch_runtime_control(
        {"command": "set_interaction_mode", "mode": "pass_through"}
    )
    await sequential_window.dispatch_snapshot(short)
    await sequential_window.dispatch_snapshot(long)
    await sequential_window.dispatch_snapshot(final_short)
    sequential_width = _caption_card_controls(sequential_app.page)[0].width
    await sequential_window.close()

    token = "scheduled-width-token"
    bridge = OverlayBridge(
        session_token=token,
        initial_snapshot=OverlayPresentationSnapshot(revision=1, blocks=[]),
        heartbeat_interval_ms=20,
        desktop_runtime_controls_enabled=True,
    )
    bridge.set_initial_desktop_runtime_controls(
        [
            {
                "command": "apply_window_bounds",
                "x": 320,
                "y": 720,
                "width": 1344,
                "height": 320,
            }
        ]
    )
    await bridge.start()
    app = FakeFletApp()
    window = desktop_overlay.FletDesktopRendererWindow(
        app_runner=app.run,
        event_sink=RecordingLifecycleSink().emit,
        locale="en",
    )
    renderer = desktop_overlay.DesktopOverlayRenderer(
        _manifest(bridge_url=bridge.url, session_token=token),
        window=window,
        lifecycle_sink=RecordingLifecycleSink(),
        parent_monitor=FakeParentMonitor(),
    )
    try:
        await renderer.enqueue_runtime_control(
            {"command": "set_interaction_mode", "mode": "pass_through"}
        )
        await renderer.enqueue_snapshot(short)
        await renderer.enqueue_snapshot(long)
        await renderer.enqueue_snapshot(final_short)
        run_task = asyncio.create_task(renderer.run())
        await _next_bridge_event(bridge, expected_type="overlay_ready")
        for _ in range(20):
            if _page_text_values(app.page) == {"응"}:
                break
            await asyncio.sleep(0.01)
        assert _page_text_values(app.page) == {"응"}
        assert _caption_card_controls(app.page)[0].width == pytest.approx(sequential_width)
        await bridge.broadcast_shutdown()
        assert await asyncio.wait_for(run_task, timeout=1.0) == 0
    finally:
        await renderer.shutdown()
        await bridge.stop()


@pytest.mark.asyncio
async def test_desktop_overlay_runtime_control_barrier_continues_through_real_queue_loop() -> None:
    token = "runtime-control-barrier-token"
    bridge = OverlayBridge(
        session_token=token,
        initial_snapshot=OverlayPresentationSnapshot(revision=1),
        heartbeat_interval_ms=20,
        desktop_runtime_controls_enabled=True,
    )
    await bridge.start()
    window = BatchingRendererWindow()
    renderer = desktop_overlay.DesktopOverlayRenderer(
        _manifest(bridge_url=bridge.url, session_token=token),
        window=window,
        lifecycle_sink=RecordingLifecycleSink(),
        parent_monitor=FakeParentMonitor(),
    )
    try:
        run_task = asyncio.create_task(renderer.run())
        await _next_bridge_event(bridge, expected_type="overlay_ready")
        await renderer.enqueue_snapshot(_scheduled_snapshot(2, "before control"))
        await renderer.enqueue_runtime_control({"command": "set_interaction_mode", "mode": "edit"})
        await renderer.enqueue_snapshot(_scheduled_snapshot(3, "after control"))
        await asyncio.wait_for(window.rendered_snapshot.wait(), timeout=1.0)
        for _ in range(20):
            if window.execution == ["snapshot:2", "control:set_interaction_mode", "snapshot:3"]:
                break
            await asyncio.sleep(0.01)
        assert window.execution == ["snapshot:2", "control:set_interaction_mode", "snapshot:3"]
        await bridge.broadcast_shutdown()
        assert await asyncio.wait_for(run_task, timeout=1.0) == 0
    finally:
        await renderer.shutdown()
        await bridge.stop()


@pytest.mark.asyncio
async def test_desktop_overlay_renderer_records_safe_failed_disposition_through_owned_port() -> (
    None
):
    token = "scheduled-failure-token"
    bridge = OverlayBridge(
        session_token=token,
        initial_snapshot=OverlayPresentationSnapshot(revision=1),
        heartbeat_interval_ms=20,
        desktop_runtime_controls_enabled=True,
    )
    await bridge.start()
    port = RecordingRendererDiagnosticPort()
    renderer = desktop_overlay.DesktopOverlayRenderer(
        _manifest(bridge_url=bridge.url, session_token=token),
        window=FailingBatchingRendererWindow(),
        lifecycle_sink=RecordingLifecycleSink(),
        parent_monitor=FakeParentMonitor(),
        diagnostic_port=port,
    )
    try:
        run_task = asyncio.create_task(renderer.run())
        await _next_bridge_event(bridge, expected_type="overlay_ready")
        await renderer.enqueue_snapshot(_scheduled_snapshot(2, "visible text"))
        assert await asyncio.wait_for(run_task, timeout=1.0) == 1
        assert port.records[-1]["event_type"] == "failed"
        assert port.records[-1]["actual_disposition"] == "failed"
        assert port.closed is True
    finally:
        await renderer.shutdown()
        await bridge.stop()


@pytest.mark.asyncio
async def test_desktop_overlay_renderer_diagnostic_local_port_waits_for_delayed_acknowledgement() -> (
    None
):
    token = "delayed-acknowledgement-token"
    bridge = OverlayBridge(
        session_token=token,
        initial_snapshot=OverlayPresentationSnapshot(revision=1),
        heartbeat_interval_ms=20,
        desktop_runtime_controls_enabled=True,
    )
    await bridge.start()
    port = desktop_overlay.DiagnosticLocalRendererPort(acknowledgement_timeout_s=1.0)
    renderer = desktop_overlay.DesktopOverlayRenderer(
        _manifest(bridge_url=bridge.url, session_token=token),
        window=BatchingRendererWindow(),
        lifecycle_sink=RecordingLifecycleSink(),
        parent_monitor=FakeParentMonitor(),
        diagnostic_port=port,
    )
    try:
        run_task = asyncio.create_task(renderer.run())
        await _next_bridge_event(bridge, expected_type="overlay_ready")
        await renderer.enqueue_snapshot(_scheduled_snapshot(2, "delayed acknowledgement"))
        envelope = None
        for _ in range(8):
            candidate = await asyncio.wait_for(port.next_event(), timeout=1.0)
            if candidate.acknowledgement is not None:
                envelope = candidate
                break
        assert envelope is not None
        assert envelope.record["event_type"] == "render_commit"
        assert run_task.done() is False
        await asyncio.sleep(0.02)
        assert run_task.done() is False
        assert port.acknowledge_render_commit(2) is True
        acknowledgement = await asyncio.wait_for(port.next_event(), timeout=1.0)
        assert acknowledgement.record["event_type"] == "render_commit_acknowledgement"
        assert acknowledgement.record["render_commit_acknowledged"] is True
        await bridge.broadcast_shutdown()
        assert await asyncio.wait_for(run_task, timeout=1.0) == 0
    finally:
        await renderer.shutdown()
        await bridge.stop()


@pytest.mark.asyncio
async def test_desktop_overlay_renderer_diagnostic_local_acknowledgement_timeout_fails_runtime() -> (
    None
):
    token = "acknowledgement-timeout-token"
    bridge = OverlayBridge(
        session_token=token,
        initial_snapshot=OverlayPresentationSnapshot(revision=1),
        heartbeat_interval_ms=20,
        desktop_runtime_controls_enabled=True,
    )
    await bridge.start()
    port = desktop_overlay.DiagnosticLocalRendererPort(acknowledgement_timeout_s=0.01)
    renderer = desktop_overlay.DesktopOverlayRenderer(
        _manifest(bridge_url=bridge.url, session_token=token),
        window=BatchingRendererWindow(),
        lifecycle_sink=RecordingLifecycleSink(),
        parent_monitor=FakeParentMonitor(),
        diagnostic_port=port,
    )
    try:
        run_task = asyncio.create_task(renderer.run())
        await _next_bridge_event(bridge, expected_type="overlay_ready")
        await renderer.enqueue_snapshot(_scheduled_snapshot(2, "timeout"))
        assert await asyncio.wait_for(run_task, timeout=1.0) == 1
        with pytest.raises(desktop_overlay.RendererDiagnosticPortClosed):
            await port.next_event()
    finally:
        await renderer.shutdown()
        await bridge.stop()


@pytest.mark.asyncio
async def test_desktop_overlay_renderer_diagnostic_local_acknowledgement_failure_fails_runtime() -> (
    None
):
    token = "acknowledgement-failure-token"
    bridge = OverlayBridge(
        session_token=token,
        initial_snapshot=OverlayPresentationSnapshot(revision=1),
        heartbeat_interval_ms=20,
        desktop_runtime_controls_enabled=True,
    )
    await bridge.start()
    port = desktop_overlay.DiagnosticLocalRendererPort(acknowledgement_timeout_s=1.0)
    renderer = desktop_overlay.DesktopOverlayRenderer(
        _manifest(bridge_url=bridge.url, session_token=token),
        window=BatchingRendererWindow(),
        lifecycle_sink=RecordingLifecycleSink(),
        parent_monitor=FakeParentMonitor(),
        diagnostic_port=port,
    )
    try:
        run_task = asyncio.create_task(renderer.run())
        await _next_bridge_event(bridge, expected_type="overlay_ready")
        await renderer.enqueue_snapshot(_scheduled_snapshot(2, "failure"))
        for _ in range(8):
            envelope = await asyncio.wait_for(port.next_event(), timeout=1.0)
            if envelope.acknowledgement is not None:
                break
        assert port.fail_render_commit(2, "local_route_failed") is True
        assert await asyncio.wait_for(run_task, timeout=1.0) == 1
    finally:
        await renderer.shutdown()
        await bridge.stop()


@pytest.mark.asyncio
async def test_desktop_overlay_renderer_diagnostic_local_port_close_and_cancellation_end_ack_wait() -> (
    None
):
    token = "acknowledgement-close-token"
    bridge = OverlayBridge(
        session_token=token,
        initial_snapshot=OverlayPresentationSnapshot(revision=1),
        heartbeat_interval_ms=20,
        desktop_runtime_controls_enabled=True,
    )
    await bridge.start()
    port = desktop_overlay.DiagnosticLocalRendererPort(acknowledgement_timeout_s=1.0)
    renderer = desktop_overlay.DesktopOverlayRenderer(
        _manifest(bridge_url=bridge.url, session_token=token),
        window=BatchingRendererWindow(),
        lifecycle_sink=RecordingLifecycleSink(),
        parent_monitor=FakeParentMonitor(),
        diagnostic_port=port,
    )
    try:
        run_task = asyncio.create_task(renderer.run())
        await _next_bridge_event(bridge, expected_type="overlay_ready")
        await renderer.enqueue_snapshot(_scheduled_snapshot(2, "close"))
        for _ in range(8):
            envelope = await asyncio.wait_for(port.next_event(), timeout=1.0)
            if envelope.acknowledgement is not None:
                break
        await port.close()
        assert await asyncio.wait_for(run_task, timeout=1.0) == 1
    finally:
        await renderer.shutdown()
        await bridge.stop()


@pytest.mark.asyncio
async def test_desktop_overlay_renderer_cancellation_ends_diagnostic_acknowledgement_wait() -> None:
    token = "acknowledgement-cancellation-token"
    bridge = OverlayBridge(
        session_token=token,
        initial_snapshot=OverlayPresentationSnapshot(revision=1),
        heartbeat_interval_ms=20,
        desktop_runtime_controls_enabled=True,
    )
    await bridge.start()
    port = desktop_overlay.DiagnosticLocalRendererPort(acknowledgement_timeout_s=1.0)
    renderer = desktop_overlay.DesktopOverlayRenderer(
        _manifest(bridge_url=bridge.url, session_token=token),
        window=BatchingRendererWindow(),
        lifecycle_sink=RecordingLifecycleSink(),
        parent_monitor=FakeParentMonitor(),
        diagnostic_port=port,
    )
    try:
        run_task = asyncio.create_task(renderer.run())
        await _next_bridge_event(bridge, expected_type="overlay_ready")
        await renderer.enqueue_snapshot(_scheduled_snapshot(2, "cancel"))
        for _ in range(8):
            envelope = await asyncio.wait_for(port.next_event(), timeout=1.0)
            if envelope.acknowledgement is not None:
                break
        run_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await run_task
        with pytest.raises(desktop_overlay.RendererDiagnosticPortClosed):
            await port.next_event()
    finally:
        await renderer.shutdown()
        await bridge.stop()


@pytest.mark.asyncio
async def test_desktop_overlay_renderer_default_diagnostic_port_outputs_only_safe_records(
    capsys: pytest.CaptureFixture[str],
) -> None:
    token = "scheduled-default-port-token"
    bridge = OverlayBridge(
        session_token=token,
        initial_snapshot=OverlayPresentationSnapshot(revision=1),
        heartbeat_interval_ms=20,
        desktop_runtime_controls_enabled=True,
    )
    await bridge.start()
    window = BatchingRendererWindow()
    renderer = desktop_overlay.DesktopOverlayRenderer(
        _manifest(bridge_url=bridge.url, session_token=token, logging_mode="detailed"),
        window=window,
        lifecycle_sink=RecordingLifecycleSink(),
        parent_monitor=FakeParentMonitor(),
    )
    try:
        run_task = asyncio.create_task(renderer.run())
        await _next_bridge_event(bridge, expected_type="overlay_ready")
        await renderer.enqueue_snapshot(_scheduled_snapshot(2, "caption must not appear"))
        await asyncio.wait_for(window.rendered_snapshot.wait(), timeout=1.0)
        output = capsys.readouterr().out
        assert '"record_type": "renderer_event"' in output
        assert '"renderer_revision": 2' in output
        assert "caption must not appear" not in output
        assert "scheduled-peer" not in output
        await bridge.broadcast_shutdown()
        assert await asyncio.wait_for(run_task, timeout=1.0) == 0
    finally:
        await renderer.shutdown()
        await bridge.stop()


def test_desktop_overlay_renderer_diagnostic_policy_requires_exact_safe_flat_schema() -> None:
    committed = {
        "schema_version": 1,
        "record_type": "renderer_event",
        "event_type": "render_commit",
        "renderer_revision": 11,
        "actual_disposition": "committed",
        "render_commit_acknowledged": False,
        "slot_count": 1,
        "line_count": 2,
        "surface_visible": True,
        "interaction_mode": "locked",
        "window_width": 1344,
        "window_height": 336,
    }

    assert desktop_overlay.validate_desktop_renderer_event(committed) == committed
    for unsafe_record in (
        {**committed, "record_type": "revision_outcome"},
        {**committed, "schema_version": True},
        {**committed, "schema_version": 1.0},
        {**committed, "caption_text": "must never persist"},
        {**committed, "occupant_key": "user-derived"},
        {**committed, "session_token": "credential"},
        {**committed, "file_path": "C:/secret.txt"},
        {**committed, "raw_exception": "Traceback (most recent call last):"},
        {**committed, "provider_payload": "payload"},
        {**committed, "window_width": {"nested": 1344}},
        {**committed, "line_count": [2]},
        {**committed, "renderer_revision": True},
        {**committed, "slot_count": True},
        {**committed, "line_count": True},
        {**committed, "window_width": True},
    ):
        assert desktop_overlay.validate_desktop_renderer_event(unsafe_record) is None
    assert (
        desktop_overlay.validate_desktop_renderer_event(
            {**committed, "render_commit_acknowledged": True}
        )
        is None
    )


class FailingStartWindow(FakeRendererWindow):
    async def start(self, initial_snapshot: OverlayPresentationSnapshot) -> None:
        await super().start(initial_snapshot)
        raise RuntimeError("window bootstrap failed")


class SensitiveFailingStartWindow(FakeRendererWindow):
    def __init__(self, secret: str) -> None:
        super().__init__()
        self.secret = secret

    async def start(self, initial_snapshot: OverlayPresentationSnapshot) -> None:
        await super().start(initial_snapshot)
        raise RuntimeError(f"window bootstrap failed with token {self.secret}")


class FakeParentMonitor:
    def __init__(self) -> None:
        self.exited = asyncio.Event()
        self.started = asyncio.Event()

    async def wait_for_parent_exit(self, stop_event: asyncio.Event) -> None:
        self.started.set()
        exit_task = asyncio.create_task(self.exited.wait())
        stop_task = asyncio.create_task(stop_event.wait())
        try:
            done, _pending = await asyncio.wait(
                {exit_task, stop_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if stop_task in done:
                return
        finally:
            for task in (exit_task, stop_task):
                if not task.done():
                    task.cancel()
            await asyncio.gather(exit_task, stop_task, return_exceptions=True)


class ClosableFakeParentMonitor(FakeParentMonitor):
    def __init__(self) -> None:
        super().__init__()
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1


async def _next_bridge_event(
    bridge: OverlayBridge,
    *,
    expected_type: str,
) -> dict[str, Any]:
    while True:
        event = await asyncio.wait_for(bridge.messages.get(), timeout=1.0)
        if event.get("type") == expected_type:
            return event


def test_desktop_overlay_parent_monitor_factory_prefers_windows_handle() -> None:
    monitor = desktop_overlay.create_parent_monitor(
        4321,
        is_windows=True,
        open_windows_handle=lambda pid: f"handle-{pid}",
    )

    assert isinstance(monitor, desktop_overlay.WindowsParentHandleMonitor)
    assert monitor.handle == "handle-4321"


def test_desktop_overlay_parent_monitor_factory_falls_back_when_handle_unavailable() -> None:
    monitor = desktop_overlay.create_parent_monitor(
        4321,
        is_windows=True,
        open_windows_handle=lambda _pid: None,
    )

    assert isinstance(monitor, desktop_overlay.BridgeDisconnectParentMonitor)
    assert monitor.parent_pid == 4321


@pytest.mark.asyncio
async def test_desktop_overlay_windows_handle_fallback_does_not_probe_with_os_kill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_os_kill(_pid: int, _signal: int) -> None:
        raise AssertionError("Windows parent monitor fallback must not call os.kill(pid, 0)")

    monkeypatch.setattr(desktop_overlay.os, "kill", fail_os_kill)
    monitor = desktop_overlay.create_parent_monitor(
        4321,
        is_windows=True,
        open_windows_handle=lambda _pid: None,
    )
    stop_event = asyncio.Event()

    wait_task = asyncio.create_task(monitor.wait_for_parent_exit(stop_event))
    await asyncio.sleep(0)
    stop_event.set()

    await asyncio.wait_for(wait_task, timeout=1.0)


def test_desktop_overlay_manifest_rejects_non_loopback_bridge_and_redacts_token(
    tmp_path: Path,
) -> None:
    token = "super-secret-session-token"
    manifest_path = _write_manifest(
        tmp_path,
        _manifest(bridge_url="wss://example.com:8765/overlay", session_token=token),
    )

    with pytest.raises(desktop_overlay.DesktopOverlayStartupError) as exc_info:
        desktop_overlay.load_renderer_manifest(manifest_path)

    error = exc_info.value
    assert error.failure_reason == "manifest_invalid"
    assert token not in str(error)
    assert token not in repr(error)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("session_token", None),
        ("session_token", ""),
        ("parent_pid", True),
        ("startup_deadline_ms", False),
    ],
)
def test_desktop_overlay_manifest_rejects_missing_or_bool_required_fields(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    payload = _manifest().to_dict()
    payload[field] = value
    manifest_path = tmp_path / "overlay-manifest.json"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(desktop_overlay.DesktopOverlayStartupError) as exc_info:
        desktop_overlay.load_renderer_manifest(manifest_path)

    assert exc_info.value.failure_reason == "manifest_invalid"


@pytest.mark.parametrize("url", ["ws://127.0.0.1:8765", "ws://[::1]:8765"])
def test_desktop_overlay_manifest_accepts_documented_loopback_bridge_urls(
    tmp_path: Path,
    url: str,
) -> None:
    manifest_path = _write_manifest(tmp_path, _manifest(bridge_url=url))

    manifest = desktop_overlay.load_renderer_manifest(manifest_path)

    assert manifest.bridge_url == url


@pytest.mark.asyncio
async def test_desktop_overlay_bridge_lifecycle_ready_after_auth_snapshot_and_window_start() -> (
    None
):
    token = "ready-session-token"
    bridge = OverlayBridge(
        session_token=token,
        initial_snapshot=OverlayPresentationSnapshot(revision=7),
        heartbeat_interval_ms=20,
        desktop_runtime_controls_enabled=True,
    )
    await bridge.start()
    sink = RecordingLifecycleSink()
    window = FakeRendererWindow()
    parent_monitor = FakeParentMonitor()
    renderer = desktop_overlay.DesktopOverlayRenderer(
        _manifest(bridge_url=bridge.url, session_token=token),
        window=window,
        lifecycle_sink=sink,
        parent_monitor=parent_monitor,
    )

    try:
        run_task = asyncio.create_task(renderer.run())
        ready_event = await _next_bridge_event(bridge, expected_type="overlay_ready")

        assert ready_event == {
            "type": "overlay_ready",
            "overlay_instance_id": "desktop-overlay-test",
        }
        assert window.started.is_set()
        assert window.snapshots[0].revision == 7
        assert sink.events[-1] == {
            "type": "overlay_ready",
            "overlay_instance_id": "desktop-overlay-test",
        }
        assert token not in json.dumps(sink.events)

        await bridge.broadcast_shutdown()
        assert await asyncio.wait_for(run_task, timeout=1.0) == 0
    finally:
        await renderer.shutdown()
        await bridge.stop()


@pytest.mark.asyncio
async def test_desktop_overlay_malformed_initial_snapshot_is_startup_error_with_fallback() -> None:
    token = "malformed-initial-secret"
    received: asyncio.Queue[dict[str, object]] = asyncio.Queue()

    async def handler(connection: Any) -> None:
        auth = json.loads(await connection.recv())
        assert auth == {"type": "auth", "session_token": token}
        await connection.send(
            json.dumps(
                {
                    "type": "snapshot",
                    "payload": {"revision": 1, "calibration": {}, "blocks": "not-a-list"},
                }
            )
        )
        message = json.loads(await asyncio.wait_for(connection.recv(), timeout=1.0))
        await received.put(message)

    server = await websockets.serve(handler, "127.0.0.1", 0, ping_interval=None)
    host, port = server.sockets[0].getsockname()[:2]
    sink = RecordingLifecycleSink()
    renderer = desktop_overlay.DesktopOverlayRenderer(
        _manifest(bridge_url=f"ws://{host}:{port}", session_token=token),
        window=FakeRendererWindow(),
        lifecycle_sink=sink,
        parent_monitor=FakeParentMonitor(),
    )

    try:
        assert await renderer.run() == 1
        bridge_event = await asyncio.wait_for(received.get(), timeout=1.0)
    finally:
        await renderer.shutdown()
        server.close()
        await server.wait_closed()

    expected = {"type": "startup_error", "failure_reason": "renderer_init_failed"}
    assert bridge_event == expected
    assert expected in sink.events
    assert sink.events[-1] == {
        "type": "shutdown_complete",
        "overlay_instance_id": "desktop-overlay-test",
    }
    assert token not in json.dumps(sink.events)
    assert token not in json.dumps(bridge_event)


@pytest.mark.asyncio
async def test_desktop_overlay_rejects_unframed_initial_runtime_controls() -> None:
    token = "unframed-initial-token"
    received: asyncio.Queue[dict[str, object]] = asyncio.Queue()

    async def handler(connection: Any) -> None:
        auth = json.loads(await connection.recv())
        assert auth == {"type": "auth", "session_token": token}
        await connection.send(
            json.dumps(
                {
                    "type": "snapshot",
                    "payload": OverlayPresentationSnapshot(revision=1).to_dict(),
                }
            )
        )
        await received.put(json.loads(await asyncio.wait_for(connection.recv(), timeout=1.0)))

    server = await websockets.serve(handler, "127.0.0.1", 0, ping_interval=None)
    host, port = server.sockets[0].getsockname()[:2]
    window = FakeRendererWindow()
    renderer = desktop_overlay.DesktopOverlayRenderer(
        _manifest(bridge_url=f"ws://{host}:{port}", session_token=token),
        window=window,
        lifecycle_sink=RecordingLifecycleSink(),
        parent_monitor=FakeParentMonitor(),
    )

    try:
        assert await renderer.run() == 1
        bridge_event = await asyncio.wait_for(received.get(), timeout=1.0)
    finally:
        await renderer.shutdown()
        server.close()
        await server.wait_closed()

    assert bridge_event == {
        "type": "startup_error",
        "failure_reason": "runtime_control_invalid",
    }
    assert window.snapshots == []


@pytest.mark.asyncio
async def test_desktop_overlay_rejects_flet_startup_without_canonical_bounds() -> None:
    token = "missing-bounds-token"
    bridge = OverlayBridge(
        session_token=token,
        initial_snapshot=OverlayPresentationSnapshot(revision=1),
        desktop_runtime_controls_enabled=True,
    )
    await bridge.start()
    app = FakeFletApp()
    renderer = desktop_overlay.DesktopOverlayRenderer(
        _manifest(bridge_url=bridge.url, session_token=token),
        window=desktop_overlay.FletDesktopRendererWindow(
            app_runner=app.run,
            event_sink=RecordingLifecycleSink().emit,
            locale="en",
        ),
        lifecycle_sink=RecordingLifecycleSink(),
        parent_monitor=FakeParentMonitor(),
    )

    try:
        assert await renderer.run() == 1
        bridge_event = await _next_bridge_event(bridge, expected_type="startup_error")
    finally:
        await renderer.shutdown()
        await bridge.stop()

    assert bridge_event == {
        "type": "startup_error",
        "failure_reason": "window_configuration_failed",
    }
    assert app.page.visibility_updates == []


@pytest.mark.asyncio
async def test_desktop_overlay_window_start_failure_reports_window_configuration_error(
    caplog: pytest.LogCaptureFixture,
) -> None:
    token = "window-start-secret-token"
    bridge = OverlayBridge(
        session_token=token,
        initial_snapshot=OverlayPresentationSnapshot(revision=3),
        heartbeat_interval_ms=20,
        desktop_runtime_controls_enabled=True,
    )
    await bridge.start()
    sink = RecordingLifecycleSink()
    renderer = desktop_overlay.DesktopOverlayRenderer(
        _manifest(bridge_url=bridge.url, session_token=token),
        window=SensitiveFailingStartWindow(token),
        lifecycle_sink=sink,
        parent_monitor=FakeParentMonitor(),
    )

    try:
        caplog.set_level(logging.WARNING, logger="puripuly_heart.ui.desktop_overlay")
        assert await renderer.run() == 1
        bridge_event = await _next_bridge_event(bridge, expected_type="startup_error")
    finally:
        await renderer.shutdown()
        await bridge.stop()

    expected = {"type": "startup_error", "failure_reason": "window_configuration_failed"}
    assert bridge_event == expected
    assert expected in sink.events
    assert sink.events[-1] == {
        "type": "shutdown_complete",
        "overlay_instance_id": "desktop-overlay-test",
    }
    assert token not in json.dumps(sink.events)
    assert token not in json.dumps(bridge_event)
    assert any(
        "Renderer startup failed" in record.message
        and "exception_type=RuntimeError" in record.message
        and "exception_message=window bootstrap failed with token <redacted>" in record.message
        and "exception_traceback=" in record.message
        and record.exc_info is None
        for record in caplog.records
    )
    formatted_tracebacks = "\n".join(
        "".join(traceback.format_exception(*record.exc_info))
        for record in caplog.records
        if record.exc_info is not None
    )
    assert token not in json.dumps(caplog.messages)
    assert token not in formatted_tracebacks


@pytest.mark.asyncio
async def test_desktop_overlay_later_malformed_snapshot_is_ignored_and_controls_dispatch() -> None:
    token = "later-snapshot-token"
    bridge = OverlayBridge(
        session_token=token,
        initial_snapshot=OverlayPresentationSnapshot(revision=1),
        heartbeat_interval_ms=20,
        desktop_runtime_controls_enabled=True,
    )
    await bridge.start()
    window = FakeRendererWindow()
    renderer = desktop_overlay.DesktopOverlayRenderer(
        _manifest(bridge_url=bridge.url, session_token=token),
        window=window,
        lifecycle_sink=RecordingLifecycleSink(),
        parent_monitor=FakeParentMonitor(),
    )

    try:
        run_task = asyncio.create_task(renderer.run())
        await _next_bridge_event(bridge, expected_type="overlay_ready")

        await bridge._broadcast_json(  # noqa: SLF001 - inject malformed renderer input
            {"type": "snapshot", "payload": {"revision": 2, "calibration": {}, "blocks": "bad"}}
        )
        await bridge.broadcast_desktop_runtime_control(
            {"command": "set_interaction_mode", "mode": "edit"}
        )
        await asyncio.sleep(0.05)

        assert [snapshot.revision for snapshot in window.snapshots] == [1]
        assert window.runtime_controls == [{"command": "set_interaction_mode", "mode": "edit"}]

        await bridge.broadcast_shutdown()
        assert await asyncio.wait_for(run_task, timeout=1.0) == 0
    finally:
        await renderer.shutdown()
        await bridge.stop()


@pytest.mark.asyncio
async def test_desktop_overlay_initial_bounds_control_applies_before_ready_event_and_lock_is_ignored() -> (
    None
):
    token = "initial-control-token"
    initial_controls = [
        {
            "command": "apply_window_bounds",
            "x": 320,
            "y": 720,
            "width": 1600,
            "height": 384,
        },
        {"command": "set_interaction_mode", "mode": "pass_through"},
    ]
    bridge = OverlayBridge(
        session_token=token,
        initial_snapshot=OverlayPresentationSnapshot(revision=1),
        heartbeat_interval_ms=20,
        desktop_runtime_controls_enabled=True,
    )
    bridge.set_initial_desktop_runtime_controls(initial_controls)
    await bridge.start()
    window = FakeRendererWindow()
    renderer = desktop_overlay.DesktopOverlayRenderer(
        _manifest(bridge_url=bridge.url, session_token=token),
        window=window,
        lifecycle_sink=RecordingLifecycleSink(),
        parent_monitor=FakeParentMonitor(),
    )

    try:
        run_task = asyncio.create_task(renderer.run())
        await _next_bridge_event(bridge, expected_type="overlay_ready")

        assert window.runtime_controls == []

        await bridge.broadcast_shutdown()
        assert await asyncio.wait_for(run_task, timeout=1.0) == 0
    finally:
        await renderer.shutdown()
        await bridge.stop()


@pytest.mark.asyncio
async def test_desktop_overlay_initial_pass_through_control_does_not_lock_startup() -> None:
    token = "initial-locked-real-window-token"
    bridge = OverlayBridge(
        session_token=token,
        initial_snapshot=OverlayPresentationSnapshot(revision=1, blocks=[]),
        heartbeat_interval_ms=20,
        desktop_runtime_controls_enabled=True,
    )
    bridge.set_initial_desktop_runtime_controls(
        [
            {
                "command": "apply_window_bounds",
                "x": 320,
                "y": 720,
                "width": 1344,
                "height": 320,
            },
            {
                "command": "apply_visual_config",
                "text_scale": 1.0,
                "background_alpha": 0.5,
                "outline_width": None,
            },
            {"command": "set_interaction_mode", "mode": "pass_through"},
        ]
    )
    await bridge.start()
    app = FakeFletApp()
    window = desktop_overlay.FletDesktopRendererWindow(
        app_runner=app.run,
        event_sink=RecordingLifecycleSink().emit,
        locale="en",
        bounds_debounce_s=0.01,
    )
    renderer = desktop_overlay.DesktopOverlayRenderer(
        _manifest(bridge_url=bridge.url, session_token=token),
        window=window,
        lifecycle_sink=RecordingLifecycleSink(),
        parent_monitor=FakeParentMonitor(),
    )

    try:
        run_task = asyncio.create_task(renderer.run())
        await _next_bridge_event(bridge, expected_type="overlay_ready")

        assert app.page.render_snapshots
        first_render = app.page.render_snapshots[0]
        assert first_render == {
            "ignore_mouse_events": False,
            "texts": {
                desktop_overlay.t_for_locale(
                    "en", "settings.overlay.desktop.empty_state.action.lock"
                )
            },
            "has_drag_area": True,
            "card_count": 1,
        }
        assert app.page.window.frameless is True
        assert app.page.window.shadow is False
        assert app.page.window.resizable is False
        assert app.page.window.always_on_top is True
        assert app.page.window.title_bar_hidden is None
        assert app.page.window.title_bar_buttons_hidden is None
        assert app.page.window.ignore_mouse_events is False
        assert all(snapshot == first_render for snapshot in app.page.render_snapshots)
        assert app.page.visibility_updates == [False, True]

        await bridge.broadcast_shutdown()
        assert await asyncio.wait_for(run_task, timeout=1.0) == 0
    finally:
        await renderer.shutdown()
        await bridge.stop()


@pytest.mark.asyncio
async def test_desktop_overlay_primed_initial_controls_are_not_replayed_after_start() -> None:
    token = "initial-control-replay-token"
    initial_controls = [
        {
            "command": "apply_window_bounds",
            "x": 320,
            "y": 720,
            "width": 1344,
            "height": 320,
        },
        {
            "command": "apply_visual_config",
            "text_scale": 1.0,
            "background_alpha": 0.5,
            "outline_width": None,
        },
        {"command": "set_interaction_mode", "mode": "edit"},
    ]
    bridge = OverlayBridge(
        session_token=token,
        initial_snapshot=OverlayPresentationSnapshot(revision=1, blocks=[]),
        heartbeat_interval_ms=20,
        desktop_runtime_controls_enabled=True,
    )
    bridge.set_initial_desktop_runtime_controls(initial_controls)
    await bridge.start()
    app = FakeFletApp()
    window = desktop_overlay.FletDesktopRendererWindow(
        app_runner=app.run,
        event_sink=RecordingLifecycleSink().emit,
        locale="en",
        bounds_debounce_s=0.01,
    )
    renderer = desktop_overlay.DesktopOverlayRenderer(
        _manifest(bridge_url=bridge.url, session_token=token),
        window=window,
        lifecycle_sink=RecordingLifecycleSink(),
        parent_monitor=FakeParentMonitor(),
    )

    try:
        run_task = asyncio.create_task(renderer.run())
        await _next_bridge_event(bridge, expected_type="overlay_ready")

        assert app.page.render_snapshots
        assert app.page.visibility_updates == [False, True]
        assert app.page.render_snapshots[0] == {
            "ignore_mouse_events": False,
            "texts": {
                desktop_overlay.t_for_locale(
                    "en", "settings.overlay.desktop.empty_state.action.lock"
                )
            },
            "has_drag_area": True,
            "card_count": 1,
        }
        assert all(
            snapshot == app.page.render_snapshots[0] for snapshot in app.page.render_snapshots
        )
        assert (app.page.window.left, app.page.window.top) == (320, 720)
        assert (app.page.window.width, app.page.window.height) == (1344, 320)

        await bridge.broadcast_shutdown()
        assert await asyncio.wait_for(run_task, timeout=1.0) == 0
    finally:
        await renderer.shutdown()
        await bridge.stop()


@pytest.mark.asyncio
async def test_desktop_overlay_events_are_bridge_only_and_do_not_use_stdout_fallback(
    capsys: pytest.CaptureFixture[str],
) -> None:
    websocket = RecordingWebSocket()
    renderer = desktop_overlay.DesktopOverlayRenderer(
        _manifest(),
        window=FakeRendererWindow(),
        lifecycle_sink=desktop_overlay.StdoutLifecycleSink(),
        parent_monitor=FakeParentMonitor(),
    )
    renderer._websocket = websocket  # noqa: SLF001 - verify renderer channel routing
    overlay_event = {
        "type": "overlay_event",
        "payload": {"event": "interaction_mode_changed", "mode": "pass_through"},
    }

    try:
        await renderer._emit_lifecycle(overlay_event)  # noqa: SLF001 - verify routing
        captured = capsys.readouterr()

        assert websocket.sent_messages == [overlay_event]
        assert captured.out == ""
        assert captured.err == ""
    finally:
        await renderer.shutdown()


@pytest.mark.asyncio
async def test_desktop_overlay_lifecycle_errors_keep_stderr_fallback(
    capsys: pytest.CaptureFixture[str],
) -> None:
    websocket = RecordingWebSocket()
    renderer = desktop_overlay.DesktopOverlayRenderer(
        _manifest(),
        window=FakeRendererWindow(),
        lifecycle_sink=desktop_overlay.StdoutLifecycleSink(),
        parent_monitor=FakeParentMonitor(),
    )
    renderer._websocket = websocket  # noqa: SLF001 - verify renderer channel routing
    runtime_error = {"type": "runtime_error", "failure_reason": "runtime_disconnected"}

    try:
        await renderer._emit_lifecycle(runtime_error)  # noqa: SLF001 - verify routing
        captured = capsys.readouterr()

        assert websocket.sent_messages == [runtime_error]
        assert captured.out == ""
        assert json.loads(captured.err) == runtime_error
    finally:
        await renderer.shutdown()


@pytest.mark.asyncio
async def test_desktop_overlay_lifecycle_sink_ignores_closed_parent_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ClosedStream:
        def write(self, _value: str) -> int:
            raise OSError(22, "Invalid argument")

        def flush(self) -> None:
            raise OSError(22, "Invalid argument")

    monkeypatch.setattr(desktop_overlay.sys, "stderr", ClosedStream())

    await desktop_overlay.StdoutLifecycleSink().emit(
        {"type": "runtime_error", "failure_reason": "runtime_disconnected"}
    )


@pytest.mark.asyncio
async def test_desktop_overlay_invalid_runtime_control_reports_error_without_dispatch() -> None:
    token = "runtime-control-token"
    bridge = OverlayBridge(
        session_token=token,
        initial_snapshot=OverlayPresentationSnapshot(revision=1),
        heartbeat_interval_ms=20,
        desktop_runtime_controls_enabled=True,
    )
    await bridge.start()
    sink = RecordingLifecycleSink()
    window = FakeRendererWindow()
    renderer = desktop_overlay.DesktopOverlayRenderer(
        _manifest(bridge_url=bridge.url, session_token=token),
        window=window,
        lifecycle_sink=sink,
        parent_monitor=FakeParentMonitor(),
    )

    try:
        run_task = asyncio.create_task(renderer.run())
        await _next_bridge_event(bridge, expected_type="overlay_ready")

        await bridge._broadcast_json(
            {"type": "runtime_control", "payload": ["bad"]}
        )  # noqa: SLF001
        runtime_error = await _next_bridge_event(bridge, expected_type="runtime_error")

        assert runtime_error == {
            "type": "runtime_error",
            "failure_reason": "runtime_control_invalid",
        }
        assert window.runtime_controls == []
        assert await asyncio.wait_for(run_task, timeout=1.0) == 1
    finally:
        await renderer.shutdown()
        await bridge.stop()


@pytest.mark.asyncio
async def test_desktop_overlay_parent_monitor_loss_reports_error_and_shutdown_is_idempotent() -> (
    None
):
    token = "parent-loss-token"
    bridge = OverlayBridge(
        session_token=token,
        initial_snapshot=OverlayPresentationSnapshot(revision=1),
        heartbeat_interval_ms=20,
        desktop_runtime_controls_enabled=True,
    )
    await bridge.start()
    window = FakeRendererWindow()
    parent_monitor = FakeParentMonitor()
    renderer = desktop_overlay.DesktopOverlayRenderer(
        _manifest(bridge_url=bridge.url, session_token=token),
        window=window,
        lifecycle_sink=RecordingLifecycleSink(),
        parent_monitor=parent_monitor,
    )

    try:
        run_task = asyncio.create_task(renderer.run())
        await _next_bridge_event(bridge, expected_type="overlay_ready")
        await asyncio.wait_for(parent_monitor.started.wait(), timeout=1.0)

        parent_monitor.exited.set()
        runtime_error = await _next_bridge_event(bridge, expected_type="runtime_error")

        assert runtime_error == {"type": "runtime_error", "failure_reason": "runtime_disconnected"}
        assert await asyncio.wait_for(run_task, timeout=1.0) == 1

        await renderer.shutdown()
        await renderer.shutdown()
        assert window.close_calls == 1
        assert renderer.is_shutdown is True
    finally:
        await renderer.shutdown()
        await bridge.stop()


@pytest.mark.asyncio
async def test_desktop_overlay_startup_failure_closes_parent_monitor_once() -> None:
    parent_monitor = ClosableFakeParentMonitor()
    renderer = desktop_overlay.DesktopOverlayRenderer(
        _manifest(bridge_url="ws://192.0.2.10:8765"),
        window=FakeRendererWindow(),
        lifecycle_sink=RecordingLifecycleSink(),
        parent_monitor=parent_monitor,
    )

    assert await renderer.run() == 1
    await renderer.shutdown()

    assert parent_monitor.close_calls == 1
