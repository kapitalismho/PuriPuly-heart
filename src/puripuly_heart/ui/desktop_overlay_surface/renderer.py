from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import replace
from typing import Any

from puripuly_heart.config.settings import (
    DESKTOP_FLET_DEFAULT_HEIGHT,
    DESKTOP_FLET_DEFAULT_SIZE_PRESET,
    DESKTOP_FLET_DEFAULT_WIDTH,
    DESKTOP_FLET_SIZE_PRESET_DISPLAY_ORDER,
    DESKTOP_FLET_SIZE_PRESET_ORDER,
    DESKTOP_FLET_SIZE_PRESETS,
    DesktopFletOverlayVisualSettings,
)
from puripuly_heart.core.overlay.protocol import (
    OverlayPresentationBlock,
    OverlayPresentationSnapshot,
)
from puripuly_heart.ui.desktop_overlay_surface.contract import (
    _DESKTOP_CAPTION_BACKGROUND_RGB,
    _DESKTOP_CAPTION_CJK_FONT_FAMILY,
    _DESKTOP_CAPTION_CJK_LANGUAGE_PRIMARY_SUBTAGS,
    _DESKTOP_CAPTION_CJK_WIDTH_EM,
    _DESKTOP_CAPTION_DYNAMIC_WIDTH_SAFETY,
    _DESKTOP_CAPTION_EMOJI_WIDTH_EM,
    _DESKTOP_CAPTION_LATIN_FONT_FAMILY,
    _DESKTOP_CAPTION_LATIN_NARROW_WIDTH_EM,
    _DESKTOP_CAPTION_LATIN_WIDE_WIDTH_EM,
    _DESKTOP_CAPTION_LINE_HEIGHT,
    _DESKTOP_CAPTION_MAX_VISIBLE_LINES,
    _DESKTOP_CAPTION_MAX_VISIBLE_SLOTS,
    _DESKTOP_CAPTION_MIN_DYNAMIC_CARD_WIDTH,
    _DESKTOP_CAPTION_PRIMARY_MAX_LINES,
    _DESKTOP_CAPTION_PRIMARY_REGION_ALIGNMENT_Y,
    _DESKTOP_CAPTION_PUNCT_WIDTH_EM,
    _DESKTOP_CAPTION_SECONDARY_MAX_LINES,
    _DESKTOP_CAPTION_SIZE_PRESETS,
    _DESKTOP_CAPTION_SPACE_WIDTH_EM,
    _DESKTOP_CAPTION_TEXT_STACK_ALIGNMENT_Y,
    _DESKTOP_CAPTION_TRANSPARENT,
    _DESKTOP_CAPTION_WHITE,
    _DESKTOP_EMPTY_LOCK_ACTION_DEFAULT_COLOR,
    _DESKTOP_EMPTY_LOCK_ACTION_DEFAULT_LABEL,
    _DESKTOP_EMPTY_LOCK_ACTION_FOCUS_COLOR,
    _DESKTOP_EMPTY_LOCK_ACTION_HORIZONTAL_PADDING,
    _DESKTOP_EMPTY_LOCK_ACTION_I18N_KEY,
    _DESKTOP_EMPTY_LOCK_ACTION_MIN_HIT_TARGET,
    _DESKTOP_EMPTY_LOCK_ACTION_TEXT_WIDTH_SAFETY,
    _DESKTOP_EMPTY_LOCK_ACTION_VERTICAL_PADDING,
    _DESKTOP_INTERACTION_MODE_EDIT,
    _DESKTOP_PREVIEW_BACKGROUND_ALPHA_PRESETS,
    _DESKTOP_PREVIEW_BACKGROUND_SURFACE_DATA,
    DesktopCaptionLine,
    DesktopCaptionPlan,
    DesktopCaptionSizePreset,
    DesktopCaptionSlot,
    DesktopCaptionVisualState,
    DesktopOverlayPreviewBackgroundSurface,
    DesktopOverlayPreviewCatalog,
    DesktopOverlayPreviewFixture,
    DesktopOverlayPreviewFixtureDataSource,
    DesktopOverlayPreviewLabels,
    DesktopOverlayPreviewSizePreset,
    _clamp,
    _desktop_caption_color_for_channel,
    _positive_int_or_default,
    _RetainedDesktopCaptionSurface,
)
from puripuly_heart.ui.fonts import noto_cjk_family_for_ui_locale
from puripuly_heart.ui.i18n import t_for_locale

_DESKTOP_PREVIEW_SECRET_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "bearer_token",
        re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{8,}\b", re.IGNORECASE),
    ),
    (
        "api_key",
        re.compile(r"\b(?:sk|rk|pk)-(?:live|prod|test)?-?[A-Za-z0-9_-]{12,}\b", re.IGNORECASE),
    ),
)


def build_desktop_caption_plan(
    snapshot: OverlayPresentationSnapshot,
    *,
    window_width: int | float = DESKTOP_FLET_DEFAULT_WIDTH,
    window_height: int | float = DESKTOP_FLET_DEFAULT_HEIGHT,
    visual_state: DesktopCaptionVisualState | None = None,
    interaction_mode: str = "pass_through",
    locale: str | None = None,
) -> DesktopCaptionPlan:
    """Map the current overlay snapshot contract into a deterministic caption plan."""

    width = _positive_int_or_default(window_width, DESKTOP_FLET_DEFAULT_WIDTH)
    height = _positive_int_or_default(window_height, DESKTOP_FLET_DEFAULT_HEIGHT)
    visual = _validated_visual_state(visual_state)
    preset = _desktop_caption_size_preset_for_dimensions(width, height)
    primary_font_size = preset.primary_font_size
    secondary_font_size = preset.secondary_font_size
    outline_width = 0.0
    cjk_font_family = noto_cjk_family_for_ui_locale(locale)

    candidate_slots = _caption_slots_with_ui_cjk_font(
        _caption_slots_for_snapshot(
            snapshot,
            primary_font_size=primary_font_size,
            secondary_font_size=secondary_font_size,
        ),
        cjk_font_family=cjk_font_family,
    )
    slots = tuple(
        _caption_slot_with_dynamic_width(
            slot,
            padding_horizontal=preset.padding_horizontal,
            max_card_width=width,
        )
        for slot in candidate_slots[:_DESKTOP_CAPTION_MAX_VISIBLE_SLOTS]
    )
    lines = tuple(line for slot in slots for line in slot.lines)

    full_window_background_visible = interaction_mode == _DESKTOP_INTERACTION_MODE_EDIT
    surface_visible = bool(slots) or full_window_background_visible
    background_alpha = 0.0
    if surface_visible:
        background_alpha = visual.background_alpha
    slot_height = max(
        1.0,
        (float(height) - preset.slot_gap) / _DESKTOP_CAPTION_MAX_VISIBLE_SLOTS,
    )
    primary_region_height = (
        primary_font_size * _DESKTOP_CAPTION_LINE_HEIGHT * _DESKTOP_CAPTION_PRIMARY_MAX_LINES
    )
    secondary_region_height = (
        secondary_font_size * _DESKTOP_CAPTION_LINE_HEIGHT * _DESKTOP_CAPTION_SECONDARY_MAX_LINES
    )
    return DesktopCaptionPlan(
        slots=slots,
        lines=lines,
        size_preset=preset.id,
        window_width=width,
        window_height=height,
        text_width=max(1, width - (preset.padding_horizontal * 2)),
        primary_font_size=primary_font_size,
        secondary_font_size=secondary_font_size,
        outline_width=outline_width,
        padding_horizontal=preset.padding_horizontal,
        padding_vertical=preset.padding_vertical,
        slot_gap=preset.slot_gap,
        slot_height=slot_height,
        primary_region_height=primary_region_height,
        secondary_region_height=secondary_region_height,
        border_radius=preset.border_radius,
        background_alpha=background_alpha,
        background_color=_caption_background_color(background_alpha),
        surface_visible=surface_visible,
        full_window_background_visible=full_window_background_visible,
        cjk_font_family=cjk_font_family,
    )


def desktop_empty_lock_action_label(locale: str | None) -> str:
    return t_for_locale(
        locale,
        _DESKTOP_EMPTY_LOCK_ACTION_I18N_KEY,
        default=_DESKTOP_EMPTY_LOCK_ACTION_DEFAULT_LABEL,
    )


def _desktop_empty_lock_action_font_size(plan: DesktopCaptionPlan) -> int:
    return max(_DESKTOP_EMPTY_LOCK_ACTION_MIN_HIT_TARGET, plan.primary_font_size)


def _desktop_empty_lock_action_width(label: str, font_size: int) -> float:
    return max(
        _DESKTOP_EMPTY_LOCK_ACTION_MIN_HIT_TARGET,
        _estimated_caption_line_width(label, font_size)
        + (_DESKTOP_EMPTY_LOCK_ACTION_HORIZONTAL_PADDING * 2)
        + _DESKTOP_EMPTY_LOCK_ACTION_TEXT_WIDTH_SAFETY,
    )


def build_desktop_empty_lock_action(
    plan: DesktopCaptionPlan,
    *,
    label: str,
    on_click: Callable[[object], object] | None,
) -> Any:
    """Build the bounded text-only lock action shown in empty moving mode."""

    import flet as ft

    font_size = _desktop_empty_lock_action_font_size(plan)
    text_style = ft.TextStyle(
        size=font_size,
        height=1.0,
        weight=ft.FontWeight.BOLD,
        font_family=_desktop_caption_font_family_for_text(
            label, ui_locale_cjk_family=plan.cjk_font_family
        ),
        decoration=None,
    )
    return ft.TextButton(
        content=label,
        tooltip=label,
        on_click=on_click,
        width=_desktop_empty_lock_action_width(label, font_size),
        height=max(
            _DESKTOP_EMPTY_LOCK_ACTION_MIN_HIT_TARGET,
            font_size + (_DESKTOP_EMPTY_LOCK_ACTION_VERTICAL_PADDING * 2),
        ),
        style=ft.ButtonStyle(
            color={
                ft.ControlState.DEFAULT: _DESKTOP_EMPTY_LOCK_ACTION_DEFAULT_COLOR,
                ft.ControlState.HOVERED: _DESKTOP_EMPTY_LOCK_ACTION_FOCUS_COLOR,
                ft.ControlState.FOCUSED: _DESKTOP_EMPTY_LOCK_ACTION_FOCUS_COLOR,
            },
            bgcolor=ft.Colors.TRANSPARENT,
            overlay_color=ft.Colors.TRANSPARENT,
            elevation=0,
            padding=ft.Padding.symmetric(
                horizontal=_DESKTOP_EMPTY_LOCK_ACTION_HORIZONTAL_PADDING,
                vertical=_DESKTOP_EMPTY_LOCK_ACTION_VERTICAL_PADDING,
            ),
            text_style=text_style,
            mouse_cursor=ft.MouseCursor.CLICK,
            animation_duration=0,
        ),
    )


def build_desktop_caption_surface(plan: DesktopCaptionPlan) -> Any:
    """Build no-outline fixed-slot Flet caption controls from a caption plan."""

    import flet as ft

    stack_controls: list[Any] = []
    if plan.full_window_background_visible:
        stack_controls.append(
            ft.Container(
                bgcolor=plan.background_color,
                border_radius=plan.border_radius,
                alignment=ft.Alignment.CENTER,
                left=0,
                top=0,
                right=0,
                bottom=0,
            )
        )
    slot_controls = [_build_flet_caption_slot(ft, plan, slot) for slot in plan.slots]
    if slot_controls:
        slot_stack_height = (plan.slot_height * len(slot_controls)) + (
            plan.slot_gap * max(0, len(slot_controls) - 1)
        )
        stack_controls.append(
            ft.Column(
                controls=slot_controls,
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                alignment=ft.MainAxisAlignment.CENTER,
                spacing=plan.slot_gap,
                tight=True,
                width=plan.window_width,
                height=slot_stack_height,
            )
        )
    return ft.Container(
        content=ft.Stack(
            controls=stack_controls,
            width=plan.window_width,
            height=plan.window_height,
        ),
        width=plan.window_width,
        height=plan.window_height,
        bgcolor=ft.Colors.TRANSPARENT,
        border_radius=plan.border_radius,
        alignment=ft.Alignment.CENTER,
        visible=plan.surface_visible,
    )


def build_desktop_transparent_sizing_host(plan: DesktopCaptionPlan) -> Any:
    """Build a transparent, layout-stable host for locked empty runtime state."""

    import flet as ft

    return ft.Container(
        width=plan.window_width,
        height=plan.window_height,
        bgcolor=ft.Colors.TRANSPARENT,
        alignment=ft.Alignment.CENTER,
    )


def build_desktop_overlay_preview_catalog(
    *,
    locale: str | None = None,
) -> DesktopOverlayPreviewCatalog:
    """Return local-only desktop overlay preview fixtures and visual presets."""

    def text(key: str) -> str:
        return t_for_locale(locale, key)

    fixtures = tuple(
        DesktopOverlayPreviewFixture(
            id=fixture_id,
            i18n_key=i18n_key,
            label=text(i18n_key),
            snapshot=snapshot,
            coverage_tags=frozenset(coverage_tags),
        )
        for fixture_id, i18n_key, snapshot, coverage_tags in _desktop_preview_fixture_data()
    )
    size_presets = tuple(
        _preview_size_preset(preset_id, locale=locale)
        for preset_id in DESKTOP_FLET_SIZE_PRESET_DISPLAY_ORDER
    )
    background_surfaces = tuple(
        DesktopOverlayPreviewBackgroundSurface(
            id=surface_id,
            i18n_key=i18n_key,
            label=text(i18n_key),
            bgcolor=bgcolor,
        )
        for surface_id, i18n_key, bgcolor in _DESKTOP_PREVIEW_BACKGROUND_SURFACE_DATA
    )
    labels = DesktopOverlayPreviewLabels(
        fixture=text("settings.overlay.desktop.preview.fixture"),
        size_preset=text("settings.overlay.desktop.size.title"),
        background_alpha=text("settings.overlay.desktop.preview.background_alpha"),
        background_surface=text("settings.overlay.desktop.preview.background_surface"),
    )
    return DesktopOverlayPreviewCatalog(
        fixtures=fixtures,
        background_surfaces=background_surfaces,
        size_presets=size_presets,
        background_alpha_presets=_DESKTOP_PREVIEW_BACKGROUND_ALPHA_PRESETS,
        labels=labels,
    )


def _preview_size_preset(
    preset_id: str,
    *,
    locale: str | None,
) -> DesktopOverlayPreviewSizePreset:
    preset = _DESKTOP_CAPTION_SIZE_PRESETS[preset_id]
    i18n_key = f"settings.overlay.desktop.size.option.{preset_id}"
    return DesktopOverlayPreviewSizePreset(
        id=preset.id,
        label=t_for_locale(locale, i18n_key),
        i18n_key=i18n_key,
        window_width=preset.window_width,
        window_height=preset.window_height,
        primary_font_size=preset.primary_font_size,
        secondary_font_size=preset.secondary_font_size,
        padding_horizontal=preset.padding_horizontal,
        padding_vertical=preset.padding_vertical,
        border_radius=preset.border_radius,
    )


def preview_fixture_secret_findings(
    catalog: DesktopOverlayPreviewCatalog | None = None,
) -> tuple[str, ...]:
    """Return redacted diagnostics for credential-like preview fixture content."""

    catalog = catalog or build_desktop_overlay_preview_catalog(locale="en")
    findings: list[str] = []
    for fixture in catalog.fixtures:
        fixture_identifier = _safe_preview_fixture_identifier(fixture.id)
        for field_path, value in _iter_preview_guard_strings(
            _preview_fixture_guard_payload(fixture)
        ):
            for pattern_name, pattern in _DESKTOP_PREVIEW_SECRET_PATTERNS:
                if pattern.search(value):
                    findings.append(
                        f"fixture {fixture_identifier} field {field_path} matched {pattern_name}"
                    )
    for field_path, value in _iter_preview_guard_strings(
        _preview_catalog_control_guard_payload(catalog)
    ):
        for pattern_name, pattern in _DESKTOP_PREVIEW_SECRET_PATTERNS:
            if pattern.search(value):
                findings.append(f"preview catalog field {field_path} matched {pattern_name}")
    return tuple(findings)


def desktop_overlay_preview_fixture_data_sources() -> tuple[
    DesktopOverlayPreviewFixtureDataSource,
    ...,
]:
    """Describe preview fixture data sources for packaging readiness checks."""

    return (
        DesktopOverlayPreviewFixtureDataSource(
            source_kind="embedded_python_module",
            module=__name__,
        ),
    )


def _preview_fixture_guard_payload(fixture: DesktopOverlayPreviewFixture) -> dict[str, object]:
    return {
        "id": fixture.id,
        "label": fixture.label,
        "i18n_key": fixture.i18n_key,
        "coverage_tags": tuple(sorted(fixture.coverage_tags)),
        "snapshot": fixture.snapshot.to_dict(),
    }


def _preview_catalog_control_guard_payload(
    catalog: DesktopOverlayPreviewCatalog,
) -> dict[str, object]:
    return {
        "background_surfaces": tuple(
            {
                "id": surface.id,
                "label": surface.label,
                "i18n_key": surface.i18n_key,
                "bgcolor": surface.bgcolor,
            }
            for surface in catalog.background_surfaces
        ),
        "size_presets": tuple(
            {
                "id": preset.id,
                "label": preset.label,
                "i18n_key": preset.i18n_key,
                "window_width": preset.window_width,
                "window_height": preset.window_height,
                "primary_font_size": preset.primary_font_size,
                "secondary_font_size": preset.secondary_font_size,
                "padding_horizontal": preset.padding_horizontal,
                "padding_vertical": preset.padding_vertical,
                "border_radius": preset.border_radius,
            }
            for preset in catalog.size_presets
        ),
        "background_alpha_presets": tuple(catalog.background_alpha_presets),
        "labels": {
            "fixture": catalog.labels.fixture,
            "size_preset": catalog.labels.size_preset,
            "background_alpha": catalog.labels.background_alpha,
            "background_surface": catalog.labels.background_surface,
        },
    }


def _iter_preview_guard_strings(value: object, path: str = "") -> tuple[tuple[str, str], ...]:
    strings: list[tuple[str, str]] = []
    if isinstance(value, str):
        strings.append((path, value))
    elif isinstance(value, dict):
        for key, item in value.items():
            key_path = str(key) if not path else f"{path}.{key}"
            strings.extend(_iter_preview_guard_strings(item, key_path))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            strings.extend(_iter_preview_guard_strings(item, f"{path}[{index}]"))
    return tuple(strings)


def _safe_preview_fixture_identifier(fixture_id: str) -> str:
    for _, pattern in _DESKTOP_PREVIEW_SECRET_PATTERNS:
        if pattern.search(fixture_id):
            return "<redacted-fixture-id>"
    return fixture_id


def _desktop_preview_fixture_data() -> tuple[
    tuple[str, str, OverlayPresentationSnapshot, frozenset[str]],
    ...,
]:
    return (
        (
            "korean_long_wrap",
            "settings.overlay.desktop.preview.fixture.korean_long_wrap",
            OverlayPresentationSnapshot(
                revision=1,
                blocks=[
                    _preview_block(
                        "preview-ko-long-active-self",
                        channel="self",
                        block_variant="active_self",
                        appearance_seq=10,
                        primary_text=(
                            "긴 문장 미리보기입니다. 한국어 자막이 화면 너비에 맞춰 "
                            "자연스럽게 줄바꿈되는지 확인하기 위해 일부러 길게 작성했습니다. "
                            "밝은 배경에서도 반투명 자막 카드가 읽기 쉬운지 살펴보세요."
                        ),
                        secondary_text=(
                            "This long Korean sample checks wrapping, source color, "
                            "and the secondary translation line."
                        ),
                        secondary_enabled=True,
                    )
                ],
            ),
            frozenset({"ko", "en", "self", "primary", "secondary", "active", "long_wrap"}),
        ),
        (
            "japanese_peer_finalized",
            "settings.overlay.desktop.preview.fixture.japanese_peer_finalized",
            OverlayPresentationSnapshot(
                revision=2,
                blocks=[
                    _preview_block(
                        "preview-ja-peer-finalized",
                        channel="peer",
                        block_variant="finalized",
                        appearance_seq=20,
                        primary_text="今日はゆっくり話してくれてありがとう。字幕カードも見やすいです。",
                        secondary_text="Thanks for speaking slowly today. The caption card is easy to read.",
                        secondary_enabled=True,
                    )
                ],
            ),
            frozenset({"ja", "en", "peer", "primary", "secondary", "finalized"}),
        ),
        (
            "chinese_self_finalized",
            "settings.overlay.desktop.preview.fixture.chinese_self_finalized",
            OverlayPresentationSnapshot(
                revision=3,
                blocks=[
                    _preview_block(
                        "preview-zh-self-finalized",
                        channel="self",
                        block_variant="finalized",
                        appearance_seq=30,
                        primary_text="我这边的桌面字幕会保持居中，并且在深色背景上也要清晰。",
                        secondary_text="My desktop captions stay centered and readable on dark backgrounds.",
                        secondary_enabled=True,
                    )
                ],
            ),
            frozenset({"zh-CN", "en", "self", "primary", "secondary", "finalized"}),
        ),
        (
            "english_active_peer",
            "settings.overlay.desktop.preview.fixture.english_active_peer",
            OverlayPresentationSnapshot(
                revision=4,
                blocks=[
                    _preview_block(
                        "preview-en-active-peer",
                        channel="peer",
                        block_variant="active_peer",
                        appearance_seq=40,
                        primary_text="",
                        secondary_text="Live peer captions are arriving right now...",
                        secondary_enabled=True,
                    )
                ],
            ),
            frozenset({"en", "peer", "primary", "active"}),
        ),
        (
            "mixed_script_emoji",
            "settings.overlay.desktop.preview.fixture.mixed_script_emoji",
            OverlayPresentationSnapshot(
                revision=5,
                blocks=[
                    _preview_block(
                        "preview-mixed-emoji-peer",
                        channel="peer",
                        block_variant="finalized",
                        appearance_seq=50,
                        primary_text="今日は PuriPuly Heart 좋아요 你好 😊✨",
                        secondary_text="Mixed source: hello, 안녕, こんにちは, 你好 🎮",
                        secondary_enabled=True,
                    )
                ],
            ),
            frozenset(
                {
                    "mixed_script",
                    "emoji",
                    "en",
                    "ko",
                    "ja",
                    "zh-CN",
                    "peer",
                    "primary",
                    "secondary",
                    "finalized",
                }
            ),
        ),
        (
            "no_captions",
            "settings.overlay.desktop.preview.fixture.no_captions",
            OverlayPresentationSnapshot(revision=6, blocks=[]),
            frozenset({"no_caption", "edit_placeholder", "pass_through_transparent"}),
        ),
    )


def _preview_block(
    block_id: str,
    *,
    channel: str,
    block_variant: str,
    appearance_seq: int,
    primary_text: str,
    secondary_text: str,
    secondary_enabled: bool,
) -> OverlayPresentationBlock:
    return OverlayPresentationBlock(
        id=block_id,
        occupant_key=f"preview:{channel}:{block_id}",
        appearance_seq=appearance_seq,
        channel=channel,  # type: ignore[arg-type]
        block_variant=block_variant,  # type: ignore[arg-type]
        primary_text=primary_text,
        secondary_text=secondary_text,
        secondary_enabled=secondary_enabled,
    )


def _caption_slots_for_snapshot(
    snapshot: OverlayPresentationSnapshot,
    *,
    primary_font_size: int,
    secondary_font_size: int,
) -> tuple[DesktopCaptionSlot, ...]:
    slots: list[DesktopCaptionSlot] = []
    for block in sorted(snapshot.blocks, key=lambda item: (item.appearance_seq, item.occupant_key)):
        lines = _caption_lines_for_block(
            block,
            primary_font_size=primary_font_size,
            secondary_font_size=secondary_font_size,
        )
        if not lines:
            continue
        slots.append(
            DesktopCaptionSlot(
                block_id=block.id,
                occupant_key=block.occupant_key,
                channel=block.channel,
                block_variant=block.block_variant,
                appearance_seq=block.appearance_seq,
                lines=lines,
                secondary_enabled=block.secondary_enabled,
                active=block.block_variant in {"active_self", "active_peer"},
            )
        )
    return tuple(slots)


def _caption_lines_for_snapshot(
    snapshot: OverlayPresentationSnapshot,
    *,
    primary_font_size: int,
    secondary_font_size: int,
) -> tuple[DesktopCaptionLine, ...]:
    return tuple(
        line
        for slot in _caption_slots_for_snapshot(
            snapshot,
            primary_font_size=primary_font_size,
            secondary_font_size=secondary_font_size,
        )
        for line in slot.lines
    )


def _caption_lines_for_block(
    block: OverlayPresentationBlock,
    *,
    primary_font_size: int,
    secondary_font_size: int,
) -> tuple[DesktopCaptionLine, ...]:
    primary_text = block.primary_text.strip()
    secondary_text = block.secondary_text.strip()
    if not primary_text and not secondary_text:
        return ()

    if block.block_variant == "active_self":
        return _self_active_lines(
            block,
            primary_text=primary_text,
            secondary_text=secondary_text,
            primary_font_size=primary_font_size,
            secondary_font_size=secondary_font_size,
        )
    if block.block_variant == "active_peer":
        return _peer_active_lines(
            block,
            primary_text=primary_text,
            secondary_text=secondary_text,
            primary_font_size=primary_font_size,
            secondary_font_size=secondary_font_size,
        )
    if block.channel == "peer":
        return _peer_finalized_lines(
            block,
            primary_text=primary_text,
            secondary_text=secondary_text,
            primary_font_size=primary_font_size,
            secondary_font_size=secondary_font_size,
        )
    return _self_finalized_lines(
        block,
        primary_text=primary_text,
        secondary_text=secondary_text,
        primary_font_size=primary_font_size,
        secondary_font_size=secondary_font_size,
    )


def _self_active_lines(
    block: OverlayPresentationBlock,
    *,
    primary_text: str,
    secondary_text: str,
    primary_font_size: int,
    secondary_font_size: int,
) -> tuple[DesktopCaptionLine, ...]:
    lines: list[DesktopCaptionLine] = []
    if primary_text:
        lines.append(
            _caption_line(
                block,
                text=primary_text,
                role="active_self_source",
                slot="primary",
                priority=100,
                max_lines=_DESKTOP_CAPTION_PRIMARY_MAX_LINES,
                font_size=primary_font_size,
                language=block.primary_language,
                active=True,
            )
        )
    if secondary_text and block.secondary_enabled:
        lines.append(
            _caption_line(
                block,
                text=secondary_text,
                role="active_self_translation",
                slot="secondary",
                priority=85,
                max_lines=_DESKTOP_CAPTION_SECONDARY_MAX_LINES,
                font_size=secondary_font_size,
                language=block.secondary_language,
                active=True,
            )
        )
    return tuple(lines)


def _peer_active_lines(
    block: OverlayPresentationBlock,
    *,
    primary_text: str,
    secondary_text: str,
    primary_font_size: int,
    secondary_font_size: int,
) -> tuple[DesktopCaptionLine, ...]:
    readable_text = primary_text or (secondary_text if block.secondary_enabled else "")
    if not readable_text:
        return ()
    promoted = not primary_text and bool(secondary_text) and block.secondary_enabled
    return (
        _caption_line(
            block,
            text=readable_text,
            role="active_peer_source",
            slot="primary" if not promoted else "primary",
            priority=95,
            max_lines=_DESKTOP_CAPTION_PRIMARY_MAX_LINES,
            font_size=primary_font_size if promoted else secondary_font_size,
            language=block.secondary_language if promoted else block.primary_language,
            promoted=promoted,
            active=True,
        ),
    )


def _peer_finalized_lines(
    block: OverlayPresentationBlock,
    *,
    primary_text: str,
    secondary_text: str,
    primary_font_size: int,
    secondary_font_size: int,
) -> tuple[DesktopCaptionLine, ...]:
    lines: list[DesktopCaptionLine] = []
    if primary_text:
        lines.append(
            _caption_line(
                block,
                text=primary_text,
                role="peer_translation",
                slot="primary",
                priority=90,
                max_lines=_DESKTOP_CAPTION_PRIMARY_MAX_LINES,
                font_size=primary_font_size,
                language=block.primary_language,
            )
        )
        if secondary_text and block.secondary_enabled:
            lines.append(
                _caption_line(
                    block,
                    text=secondary_text,
                    role="peer_source_original",
                    slot="secondary",
                    priority=70,
                    max_lines=_DESKTOP_CAPTION_SECONDARY_MAX_LINES,
                    font_size=secondary_font_size,
                    language=block.secondary_language,
                )
            )
        return tuple(lines)
    if secondary_text and block.secondary_enabled:
        return (
            _caption_line(
                block,
                text=secondary_text,
                role="peer_source_original",
                slot="primary",
                priority=60,
                max_lines=_DESKTOP_CAPTION_PRIMARY_MAX_LINES,
                font_size=primary_font_size,
                language=block.secondary_language,
                promoted=True,
            ),
        )
    return ()


def _self_finalized_lines(
    block: OverlayPresentationBlock,
    *,
    primary_text: str,
    secondary_text: str,
    primary_font_size: int,
    secondary_font_size: int,
) -> tuple[DesktopCaptionLine, ...]:
    lines: list[DesktopCaptionLine] = []
    if primary_text:
        lines.append(
            _caption_line(
                block,
                text=primary_text,
                role="self_source",
                slot="primary",
                priority=65,
                max_lines=_DESKTOP_CAPTION_PRIMARY_MAX_LINES,
                font_size=primary_font_size,
                language=block.primary_language,
            )
        )
        if secondary_text and block.secondary_enabled:
            lines.append(
                _caption_line(
                    block,
                    text=secondary_text,
                    role="self_translation",
                    slot="secondary",
                    priority=50,
                    max_lines=_DESKTOP_CAPTION_SECONDARY_MAX_LINES,
                    font_size=secondary_font_size,
                    language=block.secondary_language,
                )
            )
        return tuple(lines)
    if secondary_text and block.secondary_enabled:
        return (
            _caption_line(
                block,
                text=secondary_text,
                role="self_translation",
                slot="primary",
                priority=55,
                max_lines=_DESKTOP_CAPTION_PRIMARY_MAX_LINES,
                font_size=primary_font_size,
                language=block.secondary_language,
                promoted=True,
            ),
        )
    return ()


def _caption_line(
    block: OverlayPresentationBlock,
    *,
    text: str,
    role: str,
    slot: str,
    priority: int,
    max_lines: int,
    font_size: int,
    language: str | None = None,
    promoted: bool = False,
    active: bool = False,
) -> DesktopCaptionLine:
    uses_cjk_font_policy = _desktop_caption_uses_cjk_font_policy(text, language)
    return DesktopCaptionLine(
        text=text,
        role=role,
        slot=slot,
        color=_desktop_caption_color_for_channel(block.channel),
        priority=priority,
        block_id=block.id,
        channel=block.channel,
        block_variant=block.block_variant,
        appearance_seq=block.appearance_seq,
        max_lines=max_lines,
        font_size=font_size,
        font_family=(
            _DESKTOP_CAPTION_CJK_FONT_FAMILY
            if uses_cjk_font_policy
            else _DESKTOP_CAPTION_LATIN_FONT_FAMILY
        ),
        weight="medium" if uses_cjk_font_policy else "semibold",
        promoted=promoted,
        active=active,
    )


def _caption_slot_with_dynamic_width(
    slot: DesktopCaptionSlot,
    *,
    padding_horizontal: int,
    max_card_width: int,
) -> DesktopCaptionSlot:
    max_width = max(1.0, float(max_card_width))
    minimum_width = min(_DESKTOP_CAPTION_MIN_DYNAMIC_CARD_WIDTH, max_width)
    estimated_text_width = max(
        (_estimated_caption_line_width(line.text, line.font_size) for line in slot.lines),
        default=0.0,
    )
    estimated_card_width = (
        estimated_text_width
        + (float(padding_horizontal) * 2)
        + _DESKTOP_CAPTION_DYNAMIC_WIDTH_SAFETY
    )
    card_width = min(max_width, max(minimum_width, estimated_card_width))
    card_text_width = max(1.0, card_width - (float(padding_horizontal) * 2))
    return replace(slot, card_width=card_width, card_text_width=card_text_width)


def _caption_card_width_memory_key(slot: DesktopCaptionSlot) -> tuple[str, str, int]:
    return (slot.block_id, slot.occupant_key, slot.appearance_seq)


def _estimated_caption_line_width(text: str, font_size: int) -> float:
    return sum(_estimated_caption_char_width(char, font_size) for char in text)


def _estimated_caption_char_width(char: str, font_size: int) -> float:
    codepoint = ord(char)
    if char.isspace():
        return font_size * _DESKTOP_CAPTION_SPACE_WIDTH_EM
    if _is_caption_emoji_or_symbol(codepoint):
        return font_size * _DESKTOP_CAPTION_EMOJI_WIDTH_EM
    if _is_caption_cjk_or_hangul(codepoint):
        return font_size * _DESKTOP_CAPTION_CJK_WIDTH_EM
    if char in ".,;:!?'\"-–—()[]{}·…":
        return font_size * _DESKTOP_CAPTION_PUNCT_WIDTH_EM
    if char in "ilI|":
        return font_size * _DESKTOP_CAPTION_LATIN_NARROW_WIDTH_EM
    if char.isascii():
        return font_size * _DESKTOP_CAPTION_LATIN_WIDE_WIDTH_EM
    return font_size * _DESKTOP_CAPTION_CJK_WIDTH_EM


def _is_caption_cjk_or_hangul(codepoint: int) -> bool:
    return (
        0x1100 <= codepoint <= 0x11FF
        or 0x3040 <= codepoint <= 0x30FF
        or 0x3400 <= codepoint <= 0x4DBF
        or 0x4E00 <= codepoint <= 0x9FFF
        or 0xAC00 <= codepoint <= 0xD7AF
        or 0xF900 <= codepoint <= 0xFAFF
    )


def _is_caption_emoji_or_symbol(codepoint: int) -> bool:
    return 0x1F000 <= codepoint <= 0x1FAFF


def _desktop_caption_char_is_cjk(char: str) -> bool:
    return _is_caption_cjk_or_hangul(ord(char))


def _caption_slots_with_ui_cjk_font(
    slots: tuple[DesktopCaptionSlot, ...],
    *,
    cjk_font_family: str,
) -> tuple[DesktopCaptionSlot, ...]:
    if cjk_font_family == _DESKTOP_CAPTION_CJK_FONT_FAMILY:
        return slots
    return tuple(
        replace(
            slot,
            lines=tuple(
                replace(line, font_family=cjk_font_family)
                if line.font_family == _DESKTOP_CAPTION_CJK_FONT_FAMILY
                else line
                for line in slot.lines
            ),
        )
        for slot in slots
    )


def _desktop_caption_font_family_for_text(
    text: str,
    language: str | None = None,
    *,
    ui_locale_cjk_family: str | None = None,
) -> str:
    if _desktop_caption_uses_cjk_font_policy(text, language):
        return ui_locale_cjk_family or _DESKTOP_CAPTION_CJK_FONT_FAMILY
    return _DESKTOP_CAPTION_LATIN_FONT_FAMILY


def _desktop_caption_uses_cjk_font_policy(text: str, language: str | None = None) -> bool:
    return _desktop_caption_language_is_cjk(language) or _desktop_caption_text_contains_cjk(text)


def _desktop_caption_language_is_cjk(language: str | None) -> bool:
    primary_subtag = _desktop_caption_language_primary_subtag(language)
    return primary_subtag in _DESKTOP_CAPTION_CJK_LANGUAGE_PRIMARY_SUBTAGS


def _desktop_caption_language_primary_subtag(language: str | None) -> str | None:
    if language is None:
        return None
    normalized = language.strip().replace("_", "-").lower()
    if not normalized:
        return None
    return next((part for part in normalized.split("-") if part), None)


def _desktop_caption_text_contains_cjk(text: str) -> bool:
    return any(_desktop_caption_char_is_cjk(char) for char in text)


def _select_visible_caption_lines(
    candidates: tuple[DesktopCaptionLine, ...],
) -> tuple[DesktopCaptionLine, ...]:
    selected: list[DesktopCaptionLine] = []
    used_lines = 0
    for line in sorted(
        candidates,
        key=lambda item: (item.priority, item.appearance_seq, -_slot_order(item.slot), item.text),
        reverse=True,
    ):
        if used_lines + line.max_lines > _DESKTOP_CAPTION_MAX_VISIBLE_LINES:
            continue
        selected.append(line)
        used_lines += line.max_lines
        if used_lines >= _DESKTOP_CAPTION_MAX_VISIBLE_LINES:
            break
    return tuple(sorted(selected, key=lambda item: (item.appearance_seq, _slot_order(item.slot))))


def _slot_order(slot: str) -> int:
    if slot in {"primary", "primary_promoted", "primary_placeholder"}:
        return 0
    return 1


def _validated_visual_state(
    visual_state: DesktopCaptionVisualState | None,
) -> DesktopCaptionVisualState:
    source = visual_state or DesktopCaptionVisualState()
    settings = DesktopFletOverlayVisualSettings(
        text_scale=source.text_scale,
        background_alpha=source.background_alpha,
        outline_width=source.outline_width,
    )
    settings.validate()
    return DesktopCaptionVisualState(
        text_scale=settings.text_scale,
        background_alpha=settings.background_alpha,
        outline_width=settings.outline_width,
    )


def _desktop_caption_size_preset_for_dimensions(
    width: int,
    height: int,
) -> DesktopCaptionSizePreset:
    for preset_id in DESKTOP_FLET_SIZE_PRESET_ORDER:
        preset = _DESKTOP_CAPTION_SIZE_PRESETS[preset_id]
        settings_dimensions = DESKTOP_FLET_SIZE_PRESETS[preset_id]
        if (preset.window_width, preset.window_height) != settings_dimensions:
            raise RuntimeError("desktop caption preset dimensions diverged from settings")
        if width == preset.window_width and height == preset.window_height:
            return preset
    return _DESKTOP_CAPTION_SIZE_PRESETS[DESKTOP_FLET_DEFAULT_SIZE_PRESET]


def _caption_background_color(background_alpha: float) -> str:
    if background_alpha <= 0:
        return _DESKTOP_CAPTION_TRANSPARENT
    alpha = int(round(_clamp(background_alpha, 0.0, 1.0) * 255))
    return f"#{alpha:02X}{_DESKTOP_CAPTION_BACKGROUND_RGB}"


def _background_transparency_label_for_alpha(background_alpha: float) -> str:
    transparency = 1.0 - _clamp(background_alpha, 0.0, 1.0)
    return f"{int(round(transparency * 100))}%"


def _build_flet_caption_slot(ft: Any, plan: DesktopCaptionPlan, slot: DesktopCaptionSlot) -> Any:
    if plan.full_window_background_visible:
        card_text_width = plan.text_width
        card_width = plan.window_width
    else:
        card_text_width = slot.card_text_width or plan.text_width
        card_width = slot.card_width or plan.window_width
    slot_lines = _slot_lines_with_reserved_regions(
        slot,
        secondary_font_size=plan.secondary_font_size,
        font_family=slot.lines[0].font_family if slot.lines else None,
    )
    has_secondary_region = any(line.slot == "secondary" for line in slot_lines)
    line_controls = [
        _build_flet_caption_line(
            ft,
            plan,
            line,
            text_width=card_text_width,
            center_primary_region=not has_secondary_region,
        )
        for line in slot_lines
    ]
    column = ft.Column(
        controls=line_controls,
        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
        spacing=0,
        tight=True,
        scroll=None,
    )
    text_layer = ft.Container(
        content=column,
        width=card_text_width,
        bgcolor=ft.Colors.TRANSPARENT,
        alignment=(
            ft.Alignment(0, _DESKTOP_CAPTION_TEXT_STACK_ALIGNMENT_Y)
            if has_secondary_region
            else ft.Alignment.CENTER
        ),
    )
    inner_card = ft.Container(
        content=text_layer,
        width=card_width,
        height=plan.slot_height,
        bgcolor=(
            ft.Colors.TRANSPARENT if plan.full_window_background_visible else plan.background_color
        ),
        border_radius=plan.border_radius,
        padding=ft.Padding.symmetric(
            horizontal=plan.padding_horizontal,
            vertical=plan.padding_vertical,
        ),
        alignment=ft.Alignment.CENTER,
    )
    return ft.Container(
        content=inner_card,
        width=plan.window_width,
        height=plan.slot_height,
        bgcolor=ft.Colors.TRANSPARENT,
        alignment=ft.Alignment.CENTER,
    )


def _build_flet_caption_line(
    ft: Any,
    plan: DesktopCaptionPlan,
    line: DesktopCaptionLine,
    *,
    text_width: float,
    center_primary_region: bool = False,
) -> Any:
    height = plan.primary_region_height if line.slot == "primary" else plan.secondary_region_height
    return ft.Container(
        content=_build_flet_text(ft, line, text_width),
        width=text_width,
        height=height,
        bgcolor=ft.Colors.TRANSPARENT,
        alignment=_caption_line_region_alignment(
            ft,
            line,
            center_primary_region=center_primary_region,
        ),
    )


def _caption_line_region_alignment(
    ft: Any,
    line: DesktopCaptionLine,
    *,
    center_primary_region: bool = False,
) -> Any:
    if line.slot == "primary":
        if center_primary_region:
            return ft.Alignment.CENTER
        return ft.Alignment(0, _DESKTOP_CAPTION_PRIMARY_REGION_ALIGNMENT_Y)
    return ft.Alignment.CENTER


def _slot_lines_with_reserved_regions(
    slot: DesktopCaptionSlot,
    *,
    secondary_font_size: int,
    font_family: str | None,
) -> tuple[DesktopCaptionLine, ...]:
    primary_lines = tuple(line for line in slot.lines if line.slot == "primary")
    secondary_lines = tuple(line for line in slot.lines if line.slot == "secondary")
    if secondary_lines:
        return (*primary_lines, secondary_lines[0])
    if not _slot_should_reserve_empty_secondary_region(slot, primary_lines):
        return primary_lines
    return (
        *primary_lines,
        DesktopCaptionLine(
            text="",
            role="reserved_secondary",
            slot="secondary",
            color=_DESKTOP_CAPTION_WHITE,
            priority=0,
            block_id=slot.block_id,
            channel=slot.channel,
            block_variant=slot.block_variant,
            appearance_seq=slot.appearance_seq,
            max_lines=_DESKTOP_CAPTION_SECONDARY_MAX_LINES,
            font_size=secondary_font_size,
            font_family=font_family,
        ),
    )


def _slot_should_reserve_empty_secondary_region(
    slot: DesktopCaptionSlot,
    primary_lines: tuple[DesktopCaptionLine, ...],
) -> bool:
    if not slot.secondary_enabled:
        return False
    return any(not line.promoted for line in primary_lines)


def _build_flet_text(
    ft: Any,
    line: DesktopCaptionLine,
    text_width: float,
) -> Any:
    return ft.Text(
        value=line.text,
        width=text_width,
        text_align=ft.TextAlign.CENTER,
        font_family=line.font_family,
        size=line.font_size,
        weight=_flet_font_weight(ft, line.weight),
        max_lines=line.max_lines,
        overflow=ft.TextOverflow.ELLIPSIS,
        no_wrap=False,
        color=line.color,
        style=ft.TextStyle(
            size=line.font_size,
            height=line.line_height,
            weight=_flet_font_weight(ft, line.weight),
            font_family=line.font_family,
            foreground=None,
        ),
    )


def _flet_font_weight(ft: Any, weight: str) -> Any:
    if weight == "semibold":
        return ft.FontWeight.W_600
    if weight == "medium":
        return ft.FontWeight.W_500
    if weight == "bold":
        return ft.FontWeight.BOLD
    return None


def _retained_placeholder_line(slot: str) -> DesktopCaptionLine:
    return DesktopCaptionLine(
        text="",
        role="retained_placeholder",
        slot=slot,
        color=_DESKTOP_CAPTION_WHITE,
        priority=0,
        block_id="",
        channel="self",
        block_variant="finalized",
        appearance_seq=0,
        max_lines=(
            _DESKTOP_CAPTION_PRIMARY_MAX_LINES
            if slot == "primary"
            else _DESKTOP_CAPTION_SECONDARY_MAX_LINES
        ),
        font_size=1,
        font_family=_DESKTOP_CAPTION_LATIN_FONT_FAMILY,
    )


def _build_retained_desktop_caption_surface(
    ft: Any,
    plan: DesktopCaptionPlan,
    *,
    empty_lock_label: str,
    on_empty_lock: Callable[[object], object] | None,
    include_drag_area: bool,
) -> _RetainedDesktopCaptionSurface:
    full_background = ft.Container(
        left=0,
        top=0,
        right=0,
        bottom=0,
        alignment=ft.Alignment.CENTER,
    )
    slot_containers: list[Any] = []
    cards: list[Any] = []
    text_layers: list[Any] = []
    primary_regions: list[Any] = []
    secondary_regions: list[Any] = []
    primary_texts: list[Any] = []
    secondary_texts: list[Any] = []
    for _index in range(_DESKTOP_CAPTION_MAX_VISIBLE_SLOTS):
        primary_text = _build_flet_text(ft, _retained_placeholder_line("primary"), 1)
        secondary_text = _build_flet_text(ft, _retained_placeholder_line("secondary"), 1)
        primary_region = ft.Container(content=primary_text, bgcolor=ft.Colors.TRANSPARENT)
        secondary_region = ft.Container(content=secondary_text, bgcolor=ft.Colors.TRANSPARENT)
        column = ft.Column(
            controls=[primary_region, secondary_region],
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            spacing=0,
            tight=True,
            scroll=None,
        )
        text_layer = ft.Container(content=column, bgcolor=ft.Colors.TRANSPARENT)
        card = ft.Container(content=text_layer, alignment=ft.Alignment.CENTER)
        slot_container = ft.Container(
            content=card,
            bgcolor=ft.Colors.TRANSPARENT,
            alignment=ft.Alignment.CENTER,
        )
        slot_containers.append(slot_container)
        cards.append(card)
        text_layers.append(text_layer)
        primary_regions.append(primary_region)
        secondary_regions.append(secondary_region)
        primary_texts.append(primary_text)
        secondary_texts.append(secondary_text)
    slot_column = ft.Column(
        controls=slot_containers,
        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
        alignment=ft.MainAxisAlignment.CENTER,
        tight=True,
    )
    caption_stack = ft.Stack(
        controls=[full_background, slot_column],
        alignment=ft.Alignment.CENTER,
    )
    caption_surface = ft.Container(
        content=caption_stack,
        bgcolor=ft.Colors.TRANSPARENT,
        alignment=ft.Alignment.CENTER,
    )
    empty_lock_action = build_desktop_empty_lock_action(
        plan,
        label=empty_lock_label,
        on_click=on_empty_lock,
    )
    drag_content_host: Any | None = None
    caption_content: Any = caption_surface
    if include_drag_area:
        drag_content_host = ft.Container(
            content=caption_surface,
            bgcolor=ft.Colors.TRANSPARENT,
            alignment=ft.Alignment.CENTER,
            visible=True,
        )
        caption_content = ft.WindowDragArea(content=drag_content_host, maximizable=False)
    surface_host = ft.Stack(
        controls=[caption_content, empty_lock_action],
        alignment=ft.Alignment.CENTER,
    )
    root = ft.Container(
        content=surface_host,
        padding=0,
        bgcolor=ft.Colors.TRANSPARENT,
        alignment=ft.Alignment.CENTER,
    )
    model = _RetainedDesktopCaptionSurface(
        root=root,
        surface_host=surface_host,
        drag_content_host=drag_content_host,
        caption_surface=caption_surface,
        caption_stack=caption_stack,
        full_background=full_background,
        slot_column=slot_column,
        slot_containers=tuple(slot_containers),
        cards=tuple(cards),
        text_layers=tuple(text_layers),
        primary_regions=tuple(primary_regions),
        secondary_regions=tuple(secondary_regions),
        primary_texts=tuple(primary_texts),
        secondary_texts=tuple(secondary_texts),
        empty_lock_action=empty_lock_action,
    )
    _apply_retained_desktop_caption_plan(ft, model, plan, empty_lock_label=empty_lock_label)
    return model


def _apply_retained_desktop_caption_plan(
    ft: Any,
    model: _RetainedDesktopCaptionSurface,
    plan: DesktopCaptionPlan,
    *,
    empty_lock_label: str,
) -> None:
    model.root.width = plan.window_width
    model.root.height = plan.window_height
    model.surface_host.width = plan.window_width
    model.surface_host.height = plan.window_height
    if model.drag_content_host is not None:
        model.drag_content_host.width = plan.window_width
        model.drag_content_host.height = plan.window_height
        model.drag_content_host.visible = True
    model.caption_surface.width = plan.window_width
    model.caption_surface.height = plan.window_height
    model.caption_surface.border_radius = plan.border_radius
    model.caption_surface.visible = plan.surface_visible
    model.caption_stack.width = plan.window_width
    model.caption_stack.height = plan.window_height
    model.full_background.visible = plan.full_window_background_visible
    model.full_background.bgcolor = (
        plan.background_color if plan.full_window_background_visible else ft.Colors.TRANSPARENT
    )
    model.full_background.border_radius = plan.border_radius
    visible_slot_count = len(plan.slots)
    model.slot_column.visible = bool(visible_slot_count)
    model.slot_column.width = plan.window_width
    model.slot_column.height = (plan.slot_height * visible_slot_count) + (
        plan.slot_gap * max(0, visible_slot_count - 1)
    )
    model.slot_column.spacing = plan.slot_gap
    for index, slot_container in enumerate(model.slot_containers):
        slot = plan.slots[index] if index < visible_slot_count else None
        slot_container.visible = slot is not None
        slot_container.width = plan.window_width
        slot_container.height = plan.slot_height
        if slot is None:
            continue
        if plan.full_window_background_visible:
            card_width = plan.window_width
            card_text_width = plan.text_width
        else:
            card_width = slot.card_width or plan.window_width
            card_text_width = slot.card_text_width or plan.text_width
        card = model.cards[index]
        text_layer = model.text_layers[index]
        card.width = card_width
        card.height = plan.slot_height
        card.bgcolor = (
            ft.Colors.TRANSPARENT if plan.full_window_background_visible else plan.background_color
        )
        card.border_radius = plan.border_radius
        card.padding = ft.Padding.symmetric(
            horizontal=plan.padding_horizontal,
            vertical=plan.padding_vertical,
        )
        text_layer.width = card_text_width
        primary_line = next((line for line in slot.lines if line.slot == "primary"), None)
        secondary_line = next((line for line in slot.lines if line.slot == "secondary"), None)
        reserve_secondary = (
            secondary_line is not None
            or _slot_should_reserve_empty_secondary_region(
                slot,
                (primary_line,) if primary_line is not None else (),
            )
        )
        text_layer.alignment = (
            ft.Alignment(0, _DESKTOP_CAPTION_TEXT_STACK_ALIGNMENT_Y)
            if reserve_secondary
            else ft.Alignment.CENTER
        )
        _apply_retained_caption_line(
            ft,
            model.primary_regions[index],
            model.primary_texts[index],
            primary_line,
            width=card_text_width,
            height=plan.primary_region_height,
            center=not reserve_secondary,
        )
        _apply_retained_caption_line(
            ft,
            model.secondary_regions[index],
            model.secondary_texts[index],
            secondary_line,
            width=card_text_width,
            height=plan.secondary_region_height,
            center=True,
            visible=reserve_secondary,
            fallback_font_family=(
                primary_line.font_family
                if primary_line is not None
                else _DESKTOP_CAPTION_LATIN_FONT_FAMILY
            ),
            fallback_font_size=plan.secondary_font_size,
        )
    show_empty_lock = plan.full_window_background_visible and not plan.slots
    model.empty_lock_action.visible = show_empty_lock
    model.empty_lock_action.text = empty_lock_label if show_empty_lock else ""
    model.empty_lock_action.tooltip = empty_lock_label if show_empty_lock else None
    model.empty_lock_action.width = _desktop_empty_lock_action_width(
        empty_lock_label,
        _desktop_empty_lock_action_font_size(plan),
    )
    model.empty_lock_action.height = max(
        _DESKTOP_EMPTY_LOCK_ACTION_MIN_HIT_TARGET,
        _desktop_empty_lock_action_font_size(plan)
        + (_DESKTOP_EMPTY_LOCK_ACTION_VERTICAL_PADDING * 2),
    )


def _apply_retained_caption_line(
    ft: Any,
    region: Any,
    text: Any,
    line: DesktopCaptionLine | None,
    *,
    width: float,
    height: float,
    center: bool,
    visible: bool = True,
    fallback_font_family: str | None = None,
    fallback_font_size: int | None = None,
) -> None:
    displayed_line = line or DesktopCaptionLine(
        text="",
        role="retained_placeholder",
        slot="secondary" if fallback_font_size is not None else "primary",
        color=_DESKTOP_CAPTION_WHITE,
        priority=0,
        block_id="",
        channel="self",
        block_variant="finalized",
        appearance_seq=0,
        max_lines=(
            _DESKTOP_CAPTION_SECONDARY_MAX_LINES
            if fallback_font_size is not None
            else _DESKTOP_CAPTION_PRIMARY_MAX_LINES
        ),
        font_size=fallback_font_size or 1,
        font_family=fallback_font_family or _DESKTOP_CAPTION_LATIN_FONT_FAMILY,
    )
    region.visible = visible
    region.width = width
    region.height = height
    region.alignment = _caption_line_region_alignment(
        ft,
        displayed_line,
        center_primary_region=center,
    )
    text.value = displayed_line.text
    text.width = width
    text.font_family = displayed_line.font_family
    text.size = displayed_line.font_size
    text.weight = _flet_font_weight(ft, displayed_line.weight)
    text.max_lines = displayed_line.max_lines
    text.color = displayed_line.color
    text.style = ft.TextStyle(
        size=displayed_line.font_size,
        height=displayed_line.line_height,
        weight=_flet_font_weight(ft, displayed_line.weight),
        font_family=displayed_line.font_family,
        foreground=None,
    )
