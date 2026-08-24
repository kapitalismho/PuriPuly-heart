from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from puripuly_heart.config.settings import (
    DESKTOP_FLET_DEFAULT_BACKGROUND_ALPHA,
    DESKTOP_FLET_DEFAULT_TEXT_SCALE,
)
from puripuly_heart.core.overlay.protocol import OverlayPresentationSnapshot
from puripuly_heart.ui.fonts import FONT_FAMILY_NOTO_SANS, FONT_FAMILY_NOTO_SANS_CJK_JP

_DESKTOP_CAPTION_WHITE = "#FFFFFF"

_DESKTOP_CAPTION_GOLD = "#FFD700"

_DESKTOP_CAPTION_LATIN_FONT_FAMILY = FONT_FAMILY_NOTO_SANS

_DESKTOP_CAPTION_CJK_FONT_FAMILY = FONT_FAMILY_NOTO_SANS_CJK_JP

_DESKTOP_CAPTION_CJK_LANGUAGE_PRIMARY_SUBTAGS = frozenset(
    {"ko", "kor", "ja", "jpn", "zh", "zho", "chi", "cmn", "yue"}
)

_DESKTOP_CAPTION_BACKGROUND_RGB = "000000"

_DESKTOP_CAPTION_TRANSPARENT = "transparent"

_DESKTOP_CAPTION_MAX_VISIBLE_SLOTS = 2

_DESKTOP_CAPTION_MAX_VISIBLE_LINES = 6

_DESKTOP_CAPTION_PRIMARY_MAX_LINES = 2

_DESKTOP_CAPTION_SECONDARY_MAX_LINES = 1

_DESKTOP_CAPTION_LINE_HEIGHT = 1.24

_DESKTOP_CAPTION_PRIMARY_REGION_ALIGNMENT_Y = -0.5

_DESKTOP_CAPTION_TEXT_STACK_ALIGNMENT_Y = -0.08

_DESKTOP_CAPTION_MIN_DYNAMIC_CARD_WIDTH = 320.0

_DESKTOP_CAPTION_DYNAMIC_WIDTH_SAFETY = 24.0

_DESKTOP_CAPTION_CJK_WIDTH_EM = 1.0

_DESKTOP_CAPTION_LATIN_WIDE_WIDTH_EM = 0.62

_DESKTOP_CAPTION_LATIN_NARROW_WIDTH_EM = 0.42

_DESKTOP_CAPTION_SPACE_WIDTH_EM = 0.32

_DESKTOP_CAPTION_PUNCT_WIDTH_EM = 0.38

_DESKTOP_CAPTION_EMOJI_WIDTH_EM = 1.15

_DESKTOP_CAPTION_OVERFLOW_STRATEGY = (
    "two-turn-slots:presenter-selected-blocks,primary-two-lines,secondary-one-line"
)

_DESKTOP_INTERACTION_MODE_EDIT = "edit"

_DESKTOP_INTERACTION_MODE_PASS_THROUGH = "pass_through"

_DESKTOP_INTERACTION_MODES = {
    _DESKTOP_INTERACTION_MODE_EDIT,
    _DESKTOP_INTERACTION_MODE_PASS_THROUGH,
}

_DESKTOP_PREVIEW_BACKGROUND_ALPHA_PRESETS = (0.35, 0.5, 0.6, 0.8)

_DESKTOP_PREVIEW_DEFAULT_BACKGROUND_ALPHA = DESKTOP_FLET_DEFAULT_BACKGROUND_ALPHA

_DESKTOP_PREVIEW_DEFAULT_BACKGROUND_SURFACE_ID = "bright"

_DESKTOP_PREVIEW_STAGE_WIDTH = 1180

_DESKTOP_PREVIEW_STAGE_HEIGHT = 420

_DESKTOP_PREVIEW_BACKGROUND_SURFACE_DATA = (
    ("bright", "settings.overlay.desktop.preview.background_surface.bright", "#FFFFFF"),
    ("dark", "settings.overlay.desktop.preview.background_surface.dark", "#111827"),
    ("busy", "settings.overlay.desktop.preview.background_surface.busy", "#1F2937"),
)

_DESKTOP_EMPTY_LOCK_ACTION_I18N_KEY = "settings.overlay.desktop.empty_state.action.lock"

_DESKTOP_EMPTY_LOCK_ACTION_DEFAULT_LABEL = "Lock"

_DESKTOP_EMPTY_LOCK_ACTION_DEFAULT_COLOR = "#FFF8F4"

_DESKTOP_EMPTY_LOCK_ACTION_FOCUS_COLOR = "#FF6B6B"

_DESKTOP_EMPTY_LOCK_ACTION_MIN_HIT_TARGET = 44

_DESKTOP_EMPTY_LOCK_ACTION_HORIZONTAL_PADDING = 28

_DESKTOP_EMPTY_LOCK_ACTION_VERTICAL_PADDING = 12

_DESKTOP_EMPTY_LOCK_ACTION_TEXT_WIDTH_SAFETY = 24


def _desktop_caption_color_for_channel(channel: str) -> str:
    if channel == "peer":
        return _DESKTOP_CAPTION_GOLD
    return _DESKTOP_CAPTION_WHITE


@dataclass(frozen=True, slots=True)
class DesktopCaptionMappingRule:
    snapshot_field: str
    block_type: str
    role: str
    slot: str
    promoted: bool
    color: str
    priority: str
    truncation: str


DESKTOP_CAPTION_MAPPING_TABLE: tuple[DesktopCaptionMappingRule, ...] = (
    DesktopCaptionMappingRule(
        snapshot_field="blocks[]",
        block_type="active_self/self",
        role="active_self_source",
        slot="primary",
        promoted=False,
        color=_DESKTOP_CAPTION_WHITE,
        priority="100 newest active/interim source",
        truncation="max 2 lines; retained before secondary and finalized lines",
    ),
    DesktopCaptionMappingRule(
        snapshot_field="blocks[]",
        block_type="active_self/self",
        role="active_self_translation",
        slot="secondary",
        promoted=False,
        color=_desktop_caption_color_for_channel("self"),
        priority="85 active/interim secondary",
        truncation="max 1 line; drops before active primary",
    ),
    DesktopCaptionMappingRule(
        snapshot_field="blocks[]",
        block_type="active_peer/peer",
        role="active_peer_source",
        slot="primary",
        promoted=True,
        color=_desktop_caption_color_for_channel("peer"),
        priority="95 newest active/interim peer source",
        truncation="max 2 lines; retained before finalized secondary lines",
    ),
    DesktopCaptionMappingRule(
        snapshot_field="blocks[]",
        block_type="finalized/peer translated",
        role="peer_translation",
        slot="primary",
        promoted=False,
        color=_DESKTOP_CAPTION_GOLD,
        priority="90 peer translated primary; newer appearance wins ties",
        truncation="max 2 lines; outranks older finalized source/self lines",
    ),
    DesktopCaptionMappingRule(
        snapshot_field="blocks[]",
        block_type="finalized/peer translated",
        role="peer_source_original",
        slot="secondary",
        promoted=False,
        color=_desktop_caption_color_for_channel("peer"),
        priority="70 peer original/source secondary",
        truncation="max 1 line; drops before peer translated primary",
    ),
    DesktopCaptionMappingRule(
        snapshot_field="blocks[]",
        block_type="finalized/peer source-only",
        role="peer_source_original",
        slot="primary",
        promoted=True,
        color=_desktop_caption_color_for_channel("peer"),
        priority="60 peer source-only finalized",
        truncation="max 2 lines; drops before active and translated primary lines",
    ),
    DesktopCaptionMappingRule(
        snapshot_field="blocks[]",
        block_type="finalized/self",
        role="self_source",
        slot="primary",
        promoted=False,
        color=_DESKTOP_CAPTION_WHITE,
        priority="65 self/source finalized; newer appearance wins ties",
        truncation="max 2 lines; older finalized drops first",
    ),
    DesktopCaptionMappingRule(
        snapshot_field="blocks[]",
        block_type="finalized/self",
        role="self_translation",
        slot="secondary",
        promoted=False,
        color=_desktop_caption_color_for_channel("self"),
        priority="50 self translation secondary",
        truncation="max 1 line; drops before finalized primary lines",
    ),
    DesktopCaptionMappingRule(
        snapshot_field="blocks[]",
        block_type="finalized/self secondary-only",
        role="self_translation",
        slot="primary",
        promoted=True,
        color=_desktop_caption_color_for_channel("self"),
        priority="55 self translation secondary-only promoted primary",
        truncation="max 2 lines; drops before active and peer translated primary lines",
    ),
    DesktopCaptionMappingRule(
        snapshot_field="calibration",
        block_type="all",
        role="desktop_visual_ignored",
        slot="none",
        promoted=False,
        color="none",
        priority="not rendered",
        truncation="desktop caption visual state comes from repaired desktop visual config",
    ),
    DesktopCaptionMappingRule(
        snapshot_field="blocks[]",
        block_type="none/edit",
        role="edit_no_caption_empty_card",
        slot="none",
        promoted=False,
        color="none",
        priority="0 edit-mode empty caption surface",
        truncation="renders empty caption card with centered lock text action",
    ),
    DesktopCaptionMappingRule(
        snapshot_field="blocks[]",
        block_type="none/pass_through",
        role="pass_through_no_caption",
        slot="none",
        promoted=False,
        color="none",
        priority="not rendered",
        truncation="renders no text and no background",
    ),
)


@dataclass(frozen=True, slots=True)
class DesktopCaptionSizePreset:
    id: str
    window_width: int
    window_height: int
    primary_font_size: int
    secondary_font_size: int
    padding_horizontal: int
    padding_vertical: int
    border_radius: int
    slot_gap: int


_DESKTOP_CAPTION_SIZE_PRESETS: dict[str, DesktopCaptionSizePreset] = {
    "tiny": DesktopCaptionSizePreset("tiny", 640, 160, 20, 12, 10, 2, 10, 4),
    "xsmall": DesktopCaptionSizePreset("xsmall", 960, 240, 29, 18, 14, 6, 12, 6),
    "small": DesktopCaptionSizePreset("small", 1152, 288, 35, 21, 18, 8, 14, 8),
    "medium": DesktopCaptionSizePreset("medium", 1344, 336, 41, 25, 22, 10, 16, 10),
    "large": DesktopCaptionSizePreset("large", 1600, 400, 50, 30, 26, 12, 18, 12),
    "xlarge": DesktopCaptionSizePreset("xlarge", 1792, 448, 56, 34, 30, 14, 20, 14),
}


@dataclass(frozen=True, slots=True)
class DesktopCaptionVisualState:
    text_scale: float = DESKTOP_FLET_DEFAULT_TEXT_SCALE
    background_alpha: float = DESKTOP_FLET_DEFAULT_BACKGROUND_ALPHA
    outline_width: float | None = None
    swap_caption_languages: bool = False


@dataclass(frozen=True, slots=True)
class DesktopCaptionLine:
    text: str
    role: str
    slot: str
    color: str
    priority: int
    block_id: str
    channel: str
    block_variant: str
    appearance_seq: int
    max_lines: int
    font_size: int
    font_family: str | None
    line_height: float = _DESKTOP_CAPTION_LINE_HEIGHT
    weight: str = "semibold"
    promoted: bool = False
    active: bool = False


@dataclass(frozen=True, slots=True)
class DesktopCaptionSlot:
    block_id: str
    occupant_key: str
    channel: str
    block_variant: str
    appearance_seq: int
    lines: tuple[DesktopCaptionLine, ...]
    secondary_enabled: bool
    card_width: float = 0.0
    card_text_width: float = 0.0
    active: bool = False


@dataclass(frozen=True, slots=True)
class DesktopCaptionPlan:
    slots: tuple[DesktopCaptionSlot, ...]
    lines: tuple[DesktopCaptionLine, ...]
    size_preset: str
    window_width: int
    window_height: int
    text_width: int
    primary_font_size: int
    secondary_font_size: int
    outline_width: float
    padding_horizontal: int
    padding_vertical: int
    slot_gap: int
    slot_height: float
    primary_region_height: float
    secondary_region_height: float
    border_radius: int
    background_alpha: float
    background_color: str
    surface_visible: bool
    full_window_background_visible: bool
    cjk_font_family: str = _DESKTOP_CAPTION_CJK_FONT_FAMILY
    no_scrollbars: bool = True
    max_visible_lines: int = _DESKTOP_CAPTION_MAX_VISIBLE_LINES
    max_visible_slots: int = _DESKTOP_CAPTION_MAX_VISIBLE_SLOTS
    secondary_line_max_lines: int = _DESKTOP_CAPTION_SECONDARY_MAX_LINES
    overflow_strategy: str = _DESKTOP_CAPTION_OVERFLOW_STRATEGY


@dataclass(frozen=True, slots=True)
class DesktopOverlayPreviewFixture:
    id: str
    label: str
    i18n_key: str
    snapshot: OverlayPresentationSnapshot
    coverage_tags: frozenset[str]


@dataclass(frozen=True, slots=True)
class DesktopOverlayPreviewSizePreset:
    id: str
    label: str
    i18n_key: str
    window_width: int
    window_height: int
    primary_font_size: int
    secondary_font_size: int
    padding_horizontal: int
    padding_vertical: int
    border_radius: int


@dataclass(frozen=True, slots=True)
class DesktopOverlayPreviewBackgroundSurface:
    id: str
    label: str
    i18n_key: str
    bgcolor: str


@dataclass(frozen=True, slots=True)
class DesktopOverlayPreviewLabels:
    fixture: str
    size_preset: str
    background_alpha: str
    background_surface: str


@dataclass(frozen=True, slots=True)
class DesktopOverlayPreviewCatalog:
    fixtures: tuple[DesktopOverlayPreviewFixture, ...]
    background_surfaces: tuple[DesktopOverlayPreviewBackgroundSurface, ...]
    size_presets: tuple[DesktopOverlayPreviewSizePreset, ...]
    background_alpha_presets: tuple[float, ...]
    labels: DesktopOverlayPreviewLabels


@dataclass(frozen=True, slots=True)
class DesktopOverlayPreviewFixtureDataSource:
    source_kind: str
    module: str
    package_data_globs: tuple[str, ...] = ()
    hiddenimports: tuple[str, ...] = ()


def _positive_int_or_default(value: int | float, default: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return default
    if value <= 0:
        return default
    return int(round(value))


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return min(maximum, max(minimum, value))


@dataclass(slots=True)
class _RetainedDesktopCaptionSurface:
    root: Any
    surface_host: Any
    drag_content_host: Any | None
    caption_surface: Any
    caption_stack: Any
    full_background: Any
    slot_column: Any
    slot_containers: tuple[Any, ...]
    cards: tuple[Any, ...]
    text_layers: tuple[Any, ...]
    primary_regions: tuple[Any, ...]
    secondary_regions: tuple[Any, ...]
    primary_texts: tuple[Any, ...]
    secondary_texts: tuple[Any, ...]
    empty_lock_action: Any
