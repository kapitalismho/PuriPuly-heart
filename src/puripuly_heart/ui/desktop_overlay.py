from __future__ import annotations

import argparse
import asyncio
import contextlib
import inspect
import json
import logging
import math
import os
import re
import sys
import traceback
from collections.abc import Awaitable, Callable, Mapping
from concurrent.futures import Future as ConcurrentFuture
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlsplit

import websockets
from websockets.exceptions import ConnectionClosed

from puripuly_heart.config.settings import (
    DESKTOP_FLET_DEFAULT_BACKGROUND_ALPHA as DESKTOP_FLET_DEFAULT_BACKGROUND_ALPHA,
)
from puripuly_heart.config.settings import (
    DESKTOP_FLET_DEFAULT_HEIGHT,
    DESKTOP_FLET_DEFAULT_SIZE_PRESET,
    DESKTOP_FLET_DEFAULT_WIDTH,
    DESKTOP_FLET_MAX_BACKGROUND_ALPHA,
    DESKTOP_FLET_MAX_OUTLINE_WIDTH,
    DESKTOP_FLET_MAX_TEXT_SCALE,
    DESKTOP_FLET_MIN_BACKGROUND_ALPHA,
    DESKTOP_FLET_MIN_HEIGHT,
    DESKTOP_FLET_MIN_OUTLINE_WIDTH,
    DESKTOP_FLET_MIN_TEXT_SCALE,
    DESKTOP_FLET_MIN_WIDTH,
)
from puripuly_heart.config.settings import (
    DESKTOP_FLET_DEFAULT_TEXT_SCALE as DESKTOP_FLET_DEFAULT_TEXT_SCALE,
)
from puripuly_heart.config.settings import (
    DESKTOP_FLET_SIZE_PRESET_DISPLAY_ORDER as DESKTOP_FLET_SIZE_PRESET_DISPLAY_ORDER,
)
from puripuly_heart.config.settings import (
    DESKTOP_FLET_SIZE_PRESET_ORDER as DESKTOP_FLET_SIZE_PRESET_ORDER,
)
from puripuly_heart.config.settings import (
    DESKTOP_FLET_SIZE_PRESETS as DESKTOP_FLET_SIZE_PRESETS,
)
from puripuly_heart.config.settings import (
    DesktopFletOverlayVisualSettings as DesktopFletOverlayVisualSettings,
)
from puripuly_heart.core.diagnostic_validation import (
    DESKTOP_RENDERER_EVENT_SCHEMA_VERSION,
    validate_desktop_renderer_event,
)
from puripuly_heart.core.overlay.manifest import (
    OVERLAY_CONTRACT_VERSION,
    OverlayLaunchManifest,
    normalize_overlay_logging_mode,
)
from puripuly_heart.core.overlay.protocol import (
    OverlayPresentationBlock as OverlayPresentationBlock,
)
from puripuly_heart.core.overlay.protocol import (
    OverlayPresentationSnapshot,
)
from puripuly_heart.ui.desktop_overlay_startup import (
    DesktopOverlayStartupCoordinator,
    DesktopOverlayStartupPhase,
)
from puripuly_heart.ui.desktop_overlay_surface.contract import (
    _DESKTOP_CAPTION_BACKGROUND_RGB as _DESKTOP_CAPTION_BACKGROUND_RGB,
)
from puripuly_heart.ui.desktop_overlay_surface.contract import (
    _DESKTOP_CAPTION_CJK_FONT_FAMILY as _DESKTOP_CAPTION_CJK_FONT_FAMILY,
)
from puripuly_heart.ui.desktop_overlay_surface.contract import (
    _DESKTOP_CAPTION_CJK_LANGUAGE_PRIMARY_SUBTAGS as _DESKTOP_CAPTION_CJK_LANGUAGE_PRIMARY_SUBTAGS,
)
from puripuly_heart.ui.desktop_overlay_surface.contract import (
    _DESKTOP_CAPTION_CJK_WIDTH_EM as _DESKTOP_CAPTION_CJK_WIDTH_EM,
)
from puripuly_heart.ui.desktop_overlay_surface.contract import (
    _DESKTOP_CAPTION_DYNAMIC_WIDTH_SAFETY as _DESKTOP_CAPTION_DYNAMIC_WIDTH_SAFETY,
)
from puripuly_heart.ui.desktop_overlay_surface.contract import (
    _DESKTOP_CAPTION_EMOJI_WIDTH_EM as _DESKTOP_CAPTION_EMOJI_WIDTH_EM,
)
from puripuly_heart.ui.desktop_overlay_surface.contract import (
    _DESKTOP_CAPTION_GOLD as _DESKTOP_CAPTION_GOLD,
)
from puripuly_heart.ui.desktop_overlay_surface.contract import (
    _DESKTOP_CAPTION_LATIN_FONT_FAMILY as _DESKTOP_CAPTION_LATIN_FONT_FAMILY,
)
from puripuly_heart.ui.desktop_overlay_surface.contract import (
    _DESKTOP_CAPTION_LATIN_NARROW_WIDTH_EM as _DESKTOP_CAPTION_LATIN_NARROW_WIDTH_EM,
)
from puripuly_heart.ui.desktop_overlay_surface.contract import (
    _DESKTOP_CAPTION_LATIN_WIDE_WIDTH_EM as _DESKTOP_CAPTION_LATIN_WIDE_WIDTH_EM,
)
from puripuly_heart.ui.desktop_overlay_surface.contract import (
    _DESKTOP_CAPTION_LINE_HEIGHT as _DESKTOP_CAPTION_LINE_HEIGHT,
)
from puripuly_heart.ui.desktop_overlay_surface.contract import (
    _DESKTOP_CAPTION_MAX_VISIBLE_LINES as _DESKTOP_CAPTION_MAX_VISIBLE_LINES,
)
from puripuly_heart.ui.desktop_overlay_surface.contract import (
    _DESKTOP_CAPTION_MAX_VISIBLE_SLOTS as _DESKTOP_CAPTION_MAX_VISIBLE_SLOTS,
)
from puripuly_heart.ui.desktop_overlay_surface.contract import (
    _DESKTOP_CAPTION_MIN_DYNAMIC_CARD_WIDTH as _DESKTOP_CAPTION_MIN_DYNAMIC_CARD_WIDTH,
)
from puripuly_heart.ui.desktop_overlay_surface.contract import (
    _DESKTOP_CAPTION_OVERFLOW_STRATEGY as _DESKTOP_CAPTION_OVERFLOW_STRATEGY,
)
from puripuly_heart.ui.desktop_overlay_surface.contract import (
    _DESKTOP_CAPTION_PRIMARY_MAX_LINES as _DESKTOP_CAPTION_PRIMARY_MAX_LINES,
)
from puripuly_heart.ui.desktop_overlay_surface.contract import (
    _DESKTOP_CAPTION_PRIMARY_REGION_ALIGNMENT_Y as _DESKTOP_CAPTION_PRIMARY_REGION_ALIGNMENT_Y,
)
from puripuly_heart.ui.desktop_overlay_surface.contract import (
    _DESKTOP_CAPTION_PUNCT_WIDTH_EM as _DESKTOP_CAPTION_PUNCT_WIDTH_EM,
)
from puripuly_heart.ui.desktop_overlay_surface.contract import (
    _DESKTOP_CAPTION_SECONDARY_MAX_LINES as _DESKTOP_CAPTION_SECONDARY_MAX_LINES,
)
from puripuly_heart.ui.desktop_overlay_surface.contract import (
    _DESKTOP_CAPTION_SIZE_PRESETS as _DESKTOP_CAPTION_SIZE_PRESETS,
)
from puripuly_heart.ui.desktop_overlay_surface.contract import (
    _DESKTOP_CAPTION_SPACE_WIDTH_EM as _DESKTOP_CAPTION_SPACE_WIDTH_EM,
)
from puripuly_heart.ui.desktop_overlay_surface.contract import (
    _DESKTOP_CAPTION_TEXT_STACK_ALIGNMENT_Y as _DESKTOP_CAPTION_TEXT_STACK_ALIGNMENT_Y,
)
from puripuly_heart.ui.desktop_overlay_surface.contract import (
    _DESKTOP_CAPTION_TRANSPARENT as _DESKTOP_CAPTION_TRANSPARENT,
)
from puripuly_heart.ui.desktop_overlay_surface.contract import (
    _DESKTOP_CAPTION_WHITE as _DESKTOP_CAPTION_WHITE,
)
from puripuly_heart.ui.desktop_overlay_surface.contract import (
    _DESKTOP_EMPTY_LOCK_ACTION_DEFAULT_COLOR as _DESKTOP_EMPTY_LOCK_ACTION_DEFAULT_COLOR,
)
from puripuly_heart.ui.desktop_overlay_surface.contract import (
    _DESKTOP_EMPTY_LOCK_ACTION_DEFAULT_LABEL as _DESKTOP_EMPTY_LOCK_ACTION_DEFAULT_LABEL,
)
from puripuly_heart.ui.desktop_overlay_surface.contract import (
    _DESKTOP_EMPTY_LOCK_ACTION_FOCUS_COLOR as _DESKTOP_EMPTY_LOCK_ACTION_FOCUS_COLOR,
)
from puripuly_heart.ui.desktop_overlay_surface.contract import (
    _DESKTOP_EMPTY_LOCK_ACTION_HORIZONTAL_PADDING as _DESKTOP_EMPTY_LOCK_ACTION_HORIZONTAL_PADDING,
)
from puripuly_heart.ui.desktop_overlay_surface.contract import (
    _DESKTOP_EMPTY_LOCK_ACTION_I18N_KEY as _DESKTOP_EMPTY_LOCK_ACTION_I18N_KEY,
)
from puripuly_heart.ui.desktop_overlay_surface.contract import (
    _DESKTOP_EMPTY_LOCK_ACTION_MIN_HIT_TARGET as _DESKTOP_EMPTY_LOCK_ACTION_MIN_HIT_TARGET,
)
from puripuly_heart.ui.desktop_overlay_surface.contract import (
    _DESKTOP_EMPTY_LOCK_ACTION_TEXT_WIDTH_SAFETY as _DESKTOP_EMPTY_LOCK_ACTION_TEXT_WIDTH_SAFETY,
)
from puripuly_heart.ui.desktop_overlay_surface.contract import (
    _DESKTOP_EMPTY_LOCK_ACTION_VERTICAL_PADDING as _DESKTOP_EMPTY_LOCK_ACTION_VERTICAL_PADDING,
)
from puripuly_heart.ui.desktop_overlay_surface.contract import (
    _DESKTOP_INTERACTION_MODE_EDIT as _DESKTOP_INTERACTION_MODE_EDIT,
)
from puripuly_heart.ui.desktop_overlay_surface.contract import (
    _DESKTOP_INTERACTION_MODE_PASS_THROUGH as _DESKTOP_INTERACTION_MODE_PASS_THROUGH,
)
from puripuly_heart.ui.desktop_overlay_surface.contract import (
    _DESKTOP_INTERACTION_MODES as _DESKTOP_INTERACTION_MODES,
)
from puripuly_heart.ui.desktop_overlay_surface.contract import (
    _DESKTOP_PREVIEW_BACKGROUND_ALPHA_PRESETS as _DESKTOP_PREVIEW_BACKGROUND_ALPHA_PRESETS,
)
from puripuly_heart.ui.desktop_overlay_surface.contract import (
    _DESKTOP_PREVIEW_BACKGROUND_SURFACE_DATA as _DESKTOP_PREVIEW_BACKGROUND_SURFACE_DATA,
)
from puripuly_heart.ui.desktop_overlay_surface.contract import (
    _DESKTOP_PREVIEW_DEFAULT_BACKGROUND_ALPHA as _DESKTOP_PREVIEW_DEFAULT_BACKGROUND_ALPHA,
)
from puripuly_heart.ui.desktop_overlay_surface.contract import (
    _DESKTOP_PREVIEW_DEFAULT_BACKGROUND_SURFACE_ID as _DESKTOP_PREVIEW_DEFAULT_BACKGROUND_SURFACE_ID,
)
from puripuly_heart.ui.desktop_overlay_surface.contract import (
    _DESKTOP_PREVIEW_STAGE_HEIGHT as _DESKTOP_PREVIEW_STAGE_HEIGHT,
)
from puripuly_heart.ui.desktop_overlay_surface.contract import (
    _DESKTOP_PREVIEW_STAGE_WIDTH as _DESKTOP_PREVIEW_STAGE_WIDTH,
)
from puripuly_heart.ui.desktop_overlay_surface.contract import (
    DESKTOP_CAPTION_MAPPING_TABLE as DESKTOP_CAPTION_MAPPING_TABLE,
)
from puripuly_heart.ui.desktop_overlay_surface.contract import (
    DesktopCaptionLine as DesktopCaptionLine,
)
from puripuly_heart.ui.desktop_overlay_surface.contract import (
    DesktopCaptionMappingRule as DesktopCaptionMappingRule,
)
from puripuly_heart.ui.desktop_overlay_surface.contract import (
    DesktopCaptionPlan as DesktopCaptionPlan,
)
from puripuly_heart.ui.desktop_overlay_surface.contract import (
    DesktopCaptionSizePreset as DesktopCaptionSizePreset,
)
from puripuly_heart.ui.desktop_overlay_surface.contract import (
    DesktopCaptionSlot as DesktopCaptionSlot,
)
from puripuly_heart.ui.desktop_overlay_surface.contract import (
    DesktopCaptionVisualState as DesktopCaptionVisualState,
)
from puripuly_heart.ui.desktop_overlay_surface.contract import (
    DesktopOverlayPreviewBackgroundSurface as DesktopOverlayPreviewBackgroundSurface,
)
from puripuly_heart.ui.desktop_overlay_surface.contract import (
    DesktopOverlayPreviewCatalog as DesktopOverlayPreviewCatalog,
)
from puripuly_heart.ui.desktop_overlay_surface.contract import (
    DesktopOverlayPreviewFixture as DesktopOverlayPreviewFixture,
)
from puripuly_heart.ui.desktop_overlay_surface.contract import (
    DesktopOverlayPreviewFixtureDataSource as DesktopOverlayPreviewFixtureDataSource,
)
from puripuly_heart.ui.desktop_overlay_surface.contract import (
    DesktopOverlayPreviewLabels as DesktopOverlayPreviewLabels,
)
from puripuly_heart.ui.desktop_overlay_surface.contract import (
    DesktopOverlayPreviewSizePreset as DesktopOverlayPreviewSizePreset,
)
from puripuly_heart.ui.desktop_overlay_surface.contract import (
    _clamp as _clamp,
)
from puripuly_heart.ui.desktop_overlay_surface.contract import (
    _desktop_caption_color_for_channel as _desktop_caption_color_for_channel,
)
from puripuly_heart.ui.desktop_overlay_surface.contract import (
    _positive_int_or_default as _positive_int_or_default,
)
from puripuly_heart.ui.desktop_overlay_surface.contract import (
    _RetainedDesktopCaptionSurface as _RetainedDesktopCaptionSurface,
)
from puripuly_heart.ui.desktop_overlay_surface.renderer import (
    _DESKTOP_PREVIEW_SECRET_PATTERNS as _DESKTOP_PREVIEW_SECRET_PATTERNS,
)
from puripuly_heart.ui.desktop_overlay_surface.renderer import (
    _apply_retained_caption_line as _apply_retained_caption_line,
)
from puripuly_heart.ui.desktop_overlay_surface.renderer import (
    _apply_retained_desktop_caption_plan as _apply_retained_desktop_caption_plan,
)
from puripuly_heart.ui.desktop_overlay_surface.renderer import (
    _background_transparency_label_for_alpha as _background_transparency_label_for_alpha,
)
from puripuly_heart.ui.desktop_overlay_surface.renderer import (
    _build_flet_caption_line as _build_flet_caption_line,
)
from puripuly_heart.ui.desktop_overlay_surface.renderer import (
    _build_flet_caption_slot as _build_flet_caption_slot,
)
from puripuly_heart.ui.desktop_overlay_surface.renderer import (
    _build_flet_text as _build_flet_text,
)
from puripuly_heart.ui.desktop_overlay_surface.renderer import (
    _build_retained_desktop_caption_surface as _build_retained_desktop_caption_surface,
)
from puripuly_heart.ui.desktop_overlay_surface.renderer import (
    _caption_background_color as _caption_background_color,
)
from puripuly_heart.ui.desktop_overlay_surface.renderer import (
    _caption_card_width_memory_key as _caption_card_width_memory_key,
)
from puripuly_heart.ui.desktop_overlay_surface.renderer import (
    _caption_line as _caption_line,
)
from puripuly_heart.ui.desktop_overlay_surface.renderer import (
    _caption_line_region_alignment as _caption_line_region_alignment,
)
from puripuly_heart.ui.desktop_overlay_surface.renderer import (
    _caption_lines_for_block as _caption_lines_for_block,
)
from puripuly_heart.ui.desktop_overlay_surface.renderer import (
    _caption_lines_for_snapshot as _caption_lines_for_snapshot,
)
from puripuly_heart.ui.desktop_overlay_surface.renderer import (
    _caption_slot_with_dynamic_width as _caption_slot_with_dynamic_width,
)
from puripuly_heart.ui.desktop_overlay_surface.renderer import (
    _caption_slots_for_snapshot as _caption_slots_for_snapshot,
)
from puripuly_heart.ui.desktop_overlay_surface.renderer import (
    _desktop_caption_char_is_cjk as _desktop_caption_char_is_cjk,
)
from puripuly_heart.ui.desktop_overlay_surface.renderer import (
    _desktop_caption_font_family_for_text as _desktop_caption_font_family_for_text,
)
from puripuly_heart.ui.desktop_overlay_surface.renderer import (
    _desktop_caption_language_is_cjk as _desktop_caption_language_is_cjk,
)
from puripuly_heart.ui.desktop_overlay_surface.renderer import (
    _desktop_caption_language_primary_subtag as _desktop_caption_language_primary_subtag,
)
from puripuly_heart.ui.desktop_overlay_surface.renderer import (
    _desktop_caption_size_preset_for_dimensions as _desktop_caption_size_preset_for_dimensions,
)
from puripuly_heart.ui.desktop_overlay_surface.renderer import (
    _desktop_caption_text_contains_cjk as _desktop_caption_text_contains_cjk,
)
from puripuly_heart.ui.desktop_overlay_surface.renderer import (
    _desktop_caption_uses_cjk_font_policy as _desktop_caption_uses_cjk_font_policy,
)
from puripuly_heart.ui.desktop_overlay_surface.renderer import (
    _desktop_empty_lock_action_font_size as _desktop_empty_lock_action_font_size,
)
from puripuly_heart.ui.desktop_overlay_surface.renderer import (
    _desktop_empty_lock_action_width as _desktop_empty_lock_action_width,
)
from puripuly_heart.ui.desktop_overlay_surface.renderer import (
    _desktop_preview_fixture_data as _desktop_preview_fixture_data,
)
from puripuly_heart.ui.desktop_overlay_surface.renderer import (
    _estimated_caption_char_width as _estimated_caption_char_width,
)
from puripuly_heart.ui.desktop_overlay_surface.renderer import (
    _estimated_caption_line_width as _estimated_caption_line_width,
)
from puripuly_heart.ui.desktop_overlay_surface.renderer import (
    _flet_font_weight as _flet_font_weight,
)
from puripuly_heart.ui.desktop_overlay_surface.renderer import (
    _is_caption_cjk_or_hangul as _is_caption_cjk_or_hangul,
)
from puripuly_heart.ui.desktop_overlay_surface.renderer import (
    _is_caption_emoji_or_symbol as _is_caption_emoji_or_symbol,
)
from puripuly_heart.ui.desktop_overlay_surface.renderer import (
    _iter_preview_guard_strings as _iter_preview_guard_strings,
)
from puripuly_heart.ui.desktop_overlay_surface.renderer import (
    _peer_active_lines as _peer_active_lines,
)
from puripuly_heart.ui.desktop_overlay_surface.renderer import (
    _peer_finalized_lines as _peer_finalized_lines,
)
from puripuly_heart.ui.desktop_overlay_surface.renderer import (
    _preview_block as _preview_block,
)
from puripuly_heart.ui.desktop_overlay_surface.renderer import (
    _preview_catalog_control_guard_payload as _preview_catalog_control_guard_payload,
)
from puripuly_heart.ui.desktop_overlay_surface.renderer import (
    _preview_fixture_guard_payload as _preview_fixture_guard_payload,
)
from puripuly_heart.ui.desktop_overlay_surface.renderer import (
    _preview_size_preset as _preview_size_preset,
)
from puripuly_heart.ui.desktop_overlay_surface.renderer import (
    _retained_placeholder_line as _retained_placeholder_line,
)
from puripuly_heart.ui.desktop_overlay_surface.renderer import (
    _safe_preview_fixture_identifier as _safe_preview_fixture_identifier,
)
from puripuly_heart.ui.desktop_overlay_surface.renderer import (
    _select_visible_caption_lines as _select_visible_caption_lines,
)
from puripuly_heart.ui.desktop_overlay_surface.renderer import (
    _self_active_lines as _self_active_lines,
)
from puripuly_heart.ui.desktop_overlay_surface.renderer import (
    _self_finalized_lines as _self_finalized_lines,
)
from puripuly_heart.ui.desktop_overlay_surface.renderer import (
    _slot_lines_with_reserved_regions as _slot_lines_with_reserved_regions,
)
from puripuly_heart.ui.desktop_overlay_surface.renderer import (
    _slot_order as _slot_order,
)
from puripuly_heart.ui.desktop_overlay_surface.renderer import (
    _slot_should_reserve_empty_secondary_region as _slot_should_reserve_empty_secondary_region,
)
from puripuly_heart.ui.desktop_overlay_surface.renderer import (
    _validated_visual_state as _validated_visual_state,
)
from puripuly_heart.ui.desktop_overlay_surface.renderer import (
    build_desktop_caption_plan as build_desktop_caption_plan,
)
from puripuly_heart.ui.desktop_overlay_surface.renderer import (
    build_desktop_caption_surface as build_desktop_caption_surface,
)
from puripuly_heart.ui.desktop_overlay_surface.renderer import (
    build_desktop_empty_lock_action as build_desktop_empty_lock_action,
)
from puripuly_heart.ui.desktop_overlay_surface.renderer import (
    build_desktop_overlay_preview_catalog as build_desktop_overlay_preview_catalog,
)
from puripuly_heart.ui.desktop_overlay_surface.renderer import (
    build_desktop_transparent_sizing_host as build_desktop_transparent_sizing_host,
)
from puripuly_heart.ui.desktop_overlay_surface.renderer import (
    desktop_empty_lock_action_label as desktop_empty_lock_action_label,
)
from puripuly_heart.ui.desktop_overlay_surface.renderer import (
    desktop_overlay_preview_fixture_data_sources as desktop_overlay_preview_fixture_data_sources,
)
from puripuly_heart.ui.desktop_overlay_surface.renderer import (
    preview_fixture_secret_findings as preview_fixture_secret_findings,
)
from puripuly_heart.ui.desktop_window_zorder import (
    NoopWindowZOrderPort,
    WindowBoundsConfirmation,
    WindowVisibilityConfirmation,
    WindowZOrderPort,
    create_window_z_order_port,
)
from puripuly_heart.ui.flet_desktop_runtime import (
    FletDesktopViewProcessOwner,
    patch_hidden_view_launcher,
)
from puripuly_heart.ui.flet_runtime import invoke_control_method
from puripuly_heart.ui.fonts import assets_dir, register_fonts
from puripuly_heart.ui.i18n import t_for_locale

logger = logging.getLogger(__name__)

DESKTOP_OVERLAY_RENDERER_STARTUP_TIMEOUT_S = 12.0
DESKTOP_OVERLAY_WAIT_UNTIL_READY_TIMEOUT_S = 3.0
_LOOPBACK_BRIDGE_HOSTS = {"127.0.0.1", "::1"}
_SENSITIVE_EVENT_KEYS = {
    "accesstoken",
    "apikey",
    "authorization",
    "authorizationheader",
    "bearer",
    "secret",
    "sessiontoken",
    "token",
}
_STARTUP_FAILURE_EXIT_CODE = 1
_RUNTIME_FAILURE_EXIT_CODE = 1
_SUCCESS_EXIT_CODE = 0
_REQUIRED_MANIFEST_STRING_FIELDS = {
    "app_version",
    "bridge_url",
    "locale",
    "log_dir",
    "log_level",
    "logging_mode",
    "overlay_instance_id",
    "session_token",
}
_REQUIRED_MANIFEST_INT_FIELDS = {"contract_version", "parent_pid", "startup_deadline_ms"}
_DESKTOP_WINDOW_BOUNDS_EVENT_NAMES = {"MOVE", "MOVED", "RESIZE", "RESIZED"}
_PROGRAMMATIC_BOUNDS_ECHO_TOLERANCE_PX = 2.0


def _emit_desktop_lifecycle_trace(
    component: str,
    event: str,
    fields: Mapping[str, object],
) -> None:
    print(
        json.dumps(
            {
                "type": "overlay_trace",
                "component": component,
                "event": event,
                **fields,
            },
            sort_keys=True,
        ),
        flush=True,
    )


# Reviewable snapshot mapping table required before renderer coding.
# Current contract inspected in core.overlay.protocol/state:
# OverlayPresentationSnapshot(revision, calibration, blocks[]), where blocks[]
# contains OverlayPresentationBlock(channel self|peer, block_variant
# active_self|active_peer|finalized, primary_text, secondary_text,
# secondary_enabled, appearance_seq). Desktop visual sizing is owned by repaired
# desktop visual settings/runtime controls, so snapshot.calibration is not mapped
# to desktop caption visual state.


class DesktopOverlayStartupError(Exception):
    def __init__(self, failure_reason: str, message: str) -> None:
        super().__init__(message)
        self.failure_reason = failure_reason


class LifecycleSink(Protocol):
    async def emit(self, event: dict[str, object]) -> None: ...


@dataclass(slots=True)
class RendererCommitAcknowledgement:
    renderer_revision: int
    _completed: asyncio.Event = field(default_factory=asyncio.Event)
    _failure: str | None = None

    def acknowledge(self) -> None:
        self._completed.set()

    def fail(self, reason: str) -> None:
        self._failure = reason
        self._completed.set()

    @property
    def acknowledged(self) -> bool:
        return self._completed.is_set() and self._failure is None

    async def wait(self, timeout_s: float) -> None:
        try:
            await asyncio.wait_for(self._completed.wait(), timeout=timeout_s)
        except TimeoutError as exc:
            raise RendererDiagnosticAcknowledgementTimeout from exc
        if self._failure is not None:
            raise RendererDiagnosticAcknowledgementFailure(self._failure)


class RendererDiagnosticAcknowledgementTimeout(Exception):
    pass


class RendererDiagnosticAcknowledgementFailure(Exception):
    pass


class RendererDiagnosticPortClosed(Exception):
    pass


_DIAGNOSTIC_PORT_CLOSED = object()


@dataclass(frozen=True, slots=True)
class RendererDiagnosticEnvelope:
    record: Mapping[str, object]
    acknowledgement: RendererCommitAcknowledgement | None = None


class RendererDiagnosticPort(Protocol):
    requires_commit_acknowledgement: bool

    async def emit(self, envelope: RendererDiagnosticEnvelope) -> None: ...
    async def wait_for_commit_ack(self, acknowledgement: RendererCommitAcknowledgement) -> None: ...
    async def close(self) -> None: ...


@dataclass(slots=True)
class DetailedRendererDiagnosticPort:
    logging_mode: str
    closed: bool = False
    requires_commit_acknowledgement: bool = False

    async def emit(self, envelope: RendererDiagnosticEnvelope) -> None:
        if self.closed or self.logging_mode != "detailed":
            return
        print(
            f"[DesktopOverlay][Detail] {json.dumps(dict(envelope.record), sort_keys=True)}",
            flush=True,
        )

    async def close(self) -> None:
        self.closed = True

    async def wait_for_commit_ack(self, acknowledgement: RendererCommitAcknowledgement) -> None:
        _ = acknowledgement


@dataclass(slots=True)
class DiagnosticLocalRendererPort:
    acknowledgement_timeout_s: float = 1.0
    requires_commit_acknowledgement: bool = True
    _events: asyncio.Queue[RendererDiagnosticEnvelope | object] = field(
        default_factory=asyncio.Queue
    )
    _closed: asyncio.Event = field(default_factory=asyncio.Event)
    _acknowledgements: dict[int, RendererCommitAcknowledgement] = field(default_factory=dict)

    async def emit(self, envelope: RendererDiagnosticEnvelope) -> None:
        if self._closed.is_set():
            raise RendererDiagnosticPortClosed
        acknowledgement = envelope.acknowledgement
        if acknowledgement is not None:
            self._acknowledgements[acknowledgement.renderer_revision] = acknowledgement
        await self._events.put(envelope)

    async def next_event(self) -> RendererDiagnosticEnvelope:
        event = await self._events.get()
        if event is _DIAGNOSTIC_PORT_CLOSED:
            await self._events.put(_DIAGNOSTIC_PORT_CLOSED)
            raise RendererDiagnosticPortClosed
        if not isinstance(event, RendererDiagnosticEnvelope):
            raise RendererDiagnosticPortClosed
        return event

    def acknowledge_render_commit(self, renderer_revision: int) -> bool:
        acknowledgement = self._acknowledgements.pop(renderer_revision, None)
        if acknowledgement is None:
            return False
        acknowledgement.acknowledge()
        return True

    def fail_render_commit(self, renderer_revision: int, reason: str) -> bool:
        acknowledgement = self._acknowledgements.pop(renderer_revision, None)
        if acknowledgement is None:
            return False
        acknowledgement.fail(reason)
        return True

    async def wait_for_commit_ack(self, acknowledgement: RendererCommitAcknowledgement) -> None:
        await acknowledgement.wait(self.acknowledgement_timeout_s)

    async def close(self) -> None:
        if self._closed.is_set():
            return
        self._closed.set()
        while True:
            try:
                self._events.get_nowait()
            except asyncio.QueueEmpty:
                break
        await self._events.put(_DIAGNOSTIC_PORT_CLOSED)
        for acknowledgement in self._acknowledgements.values():
            acknowledgement.fail("port_closed")
        self._acknowledgements.clear()


@dataclass(slots=True)
class DiagnosticIngressGate:
    _expected_revisions: tuple[int, ...] = ()
    _queued_revisions: set[int] = field(default_factory=set)
    _queued: asyncio.Event = field(default_factory=asyncio.Event)
    _released: asyncio.Event = field(default_factory=asyncio.Event)

    async def hold(self, revisions: tuple[int, ...]) -> None:
        if not revisions or self._expected_revisions:
            raise RuntimeError("diagnostic ingress gate is already active")
        self._expected_revisions = revisions
        self._queued_revisions.clear()
        self._queued.clear()
        self._released.clear()

    async def snapshot_enqueued(self, revision: int) -> None:
        if revision not in self._expected_revisions:
            return
        self._queued_revisions.add(revision)
        if self._queued_revisions == set(self._expected_revisions):
            self._queued.set()

    async def wait_until_queued(self, timeout_s: float) -> None:
        try:
            await asyncio.wait_for(self._queued.wait(), timeout=timeout_s)
        except TimeoutError as exc:
            raise RendererDiagnosticAcknowledgementTimeout from exc

    def release(self) -> None:
        if not self._queued.is_set():
            raise RuntimeError("diagnostic ingress batch was not fully queued")
        self._released.set()

    async def wait_for_release(self, revision: int) -> None:
        if not self._expected_revisions or revision != self._expected_revisions[0]:
            return
        await self._released.wait()
        self._expected_revisions = ()
        self._queued_revisions.clear()
        self._queued.clear()
        self._released.clear()


class RendererWindow(Protocol):
    async def start(self, initial_snapshot: OverlayPresentationSnapshot) -> None: ...
    async def run_until_closed(self) -> None: ...
    async def close(self) -> None: ...
    async def dispatch_snapshot(self, snapshot: OverlayPresentationSnapshot) -> None: ...
    async def dispatch_runtime_control(self, payload: dict[str, object]) -> None: ...
    async def advance_snapshot_history(self, snapshot: OverlayPresentationSnapshot) -> None: ...
    def renderer_visual_state(self) -> dict[str, object]: ...
    def renderer_visual_state_for_snapshot(
        self,
        snapshot: OverlayPresentationSnapshot,
    ) -> dict[str, object]: ...


class ParentMonitor(Protocol):
    async def wait_for_parent_exit(self, stop_event: asyncio.Event) -> None: ...


@dataclass(frozen=True, slots=True)
class _RuntimeOutcome:
    exit_code: int


@dataclass(frozen=True, slots=True)
class _DesktopRenderTrace:
    content_kind: str
    surface_visible: bool
    slot_count: int
    line_count: int
    window_width: int
    window_height: int
    background_alpha: float


class StdoutLifecycleSink:
    async def emit(self, event: dict[str, object]) -> None:
        safe_event = _redact_event(event)
        if safe_event.get("type") == "overlay_event":
            return
        stream = (
            sys.stderr
            if safe_event.get("type") in {"startup_error", "runtime_error"}
            else sys.stdout
        )
        with contextlib.suppress(OSError):
            print(json.dumps(safe_event, sort_keys=True), file=stream, flush=True)


type FletAppRunner = Callable[[Callable[[Any], object]], Awaitable[None]]
type OverlayEventSink = Callable[[dict[str, object]], Awaitable[None]]
type PreviewAppRunner = Callable[[Callable[[Any], object]], object]
type FletProcessInfoProvider = Callable[[], tuple[int, str | None] | None]
type ScheduledCallbackTask = asyncio.Future[Any] | ConcurrentFuture[Any]


async def _default_flet_app_runner(
    target: Callable[[Any], object],
    *,
    on_process_started: Callable[[int, str | None], None] | None = None,
    process_owner: FletDesktopViewProcessOwner | None = None,
) -> None:
    import flet as ft

    with patch_hidden_view_launcher(
        on_process_started=on_process_started,
        process_owner=process_owner,
    ):
        await ft.run_async(
            main=target,
            view=ft.AppView.FLET_APP_HIDDEN,
            assets_dir=str(assets_dir()),
        )


def _default_preview_app_runner(target: Callable[[Any], object]) -> object:
    import flet as ft

    return ft.run_async(main=target)


_REAL_DEFAULT_PREVIEW_APP_RUNNER = _default_preview_app_runner


class FletDesktopRendererWindow:
    """Flet 0.86.1 transparent desktop overlay window boundary.

    The renderer remains persistence-free: this class only applies runtime
    controls to the Flet page/window and emits renderer-originated overlay
    events for the parent/controller to decide whether and how to persist.
    """

    def __init__(
        self,
        *,
        app_runner: FletAppRunner | None = None,
        event_sink: OverlayEventSink | None = None,
        locale: str | None = None,
        logging_mode: str = "basic",
        bounds_debounce_s: float = 0.15,
        startup_timeout_s: float = DESKTOP_OVERLAY_RENDERER_STARTUP_TIMEOUT_S,
        wait_until_ready_timeout_s: float = DESKTOP_OVERLAY_WAIT_UNTIL_READY_TIMEOUT_S,
        preview_catalog: DesktopOverlayPreviewCatalog | None = None,
        window_z_order_port: WindowZOrderPort | None = None,
        window_process_info_provider: FletProcessInfoProvider | None = None,
        view_process_owner: FletDesktopViewProcessOwner | None = None,
    ) -> None:
        if (
            app_runner is not None
            and window_z_order_port is not None
            and window_process_info_provider is None
        ):
            raise ValueError(
                "window_process_info_provider is required with a custom app_runner "
                "and window_z_order_port"
            )
        if window_z_order_port is not None:
            self._window_z_order_port = window_z_order_port
        elif app_runner is None and preview_catalog is None:
            self._window_z_order_port = create_window_z_order_port()
        else:
            self._window_z_order_port = NoopWindowZOrderPort()
        self._window_z_order_required = window_z_order_port is not None or (
            app_runner is None and preview_catalog is None and os.name == "nt"
        )
        self._structured_lifecycle_trace_enabled = app_runner is None and preview_catalog is None
        if app_runner is None:
            self._view_process_owner = view_process_owner or FletDesktopViewProcessOwner(
                trace_sink=self._record_process_lifecycle,
            )

            async def run_default_app(target: Callable[[Any], object]) -> None:
                await _default_flet_app_runner(
                    target,
                    on_process_started=self._record_flet_process,
                    process_owner=self._view_process_owner,
                )

            self._app_runner = run_default_app
        else:
            self._app_runner = app_runner
            self._view_process_owner = view_process_owner
        self._window_process_info_provider = window_process_info_provider
        self._event_sink = event_sink
        self._locale = locale
        self._logging_mode = normalize_overlay_logging_mode(logging_mode)
        self._bounds_debounce_s = max(0.0, float(bounds_debounce_s))
        self._startup_timeout_s = max(0.1, float(startup_timeout_s))
        self._wait_until_ready_timeout_s = max(0.1, float(wait_until_ready_timeout_s))
        self._preview_catalog = preview_catalog
        self._preview_fixture_id = preview_catalog.fixtures[0].id if preview_catalog else None
        self._preview_background_surface_id = _DESKTOP_PREVIEW_DEFAULT_BACKGROUND_SURFACE_ID
        self._preview_background_alpha = _DESKTOP_PREVIEW_DEFAULT_BACKGROUND_ALPHA
        self._preview_size_preset_id = DESKTOP_FLET_DEFAULT_SIZE_PRESET
        self._snapshot = OverlayPresentationSnapshot()
        self._last_snapshot_revision = -1
        self._visual_state = DesktopCaptionVisualState()
        self._interaction_mode = _DESKTOP_INTERACTION_MODE_EDIT
        self._startup_visual_state: DesktopCaptionVisualState | None = None
        self._startup_window_bounds: dict[str, int | float] | None = None
        self._page: Any | None = None
        self._page_ready = asyncio.Event()
        self._closed = asyncio.Event()
        self._app_task: asyncio.Task[None] | None = None
        self._page_start_error: BaseException | None = None
        self._flet_process_pid: int | None = None
        self._flet_pid_file: str | None = None
        self._startup_generation = 0
        self._startup_coordinator: DesktopOverlayStartupCoordinator | None = None
        self._interaction_mode_lock = asyncio.Lock()
        self._interaction_generation = 0
        self._window_z_order_task: ScheduledCallbackTask | None = None
        self._bounds_sample_task: asyncio.Task[None] | None = None
        self._scheduled_callback_tasks: set[ScheduledCallbackTask] = set()
        self._programmatic_bounds_signatures: dict[
            int,
            set[tuple[float, float, float, float]],
        ] = {}
        self._last_reported_bounds: tuple[float, float, float, float] | None = None
        self._caption_card_width_floor_by_block: dict[tuple[str, str, int], float] = {}
        self._last_render_trace: _DesktopRenderTrace | None = None
        self._retained_caption_surface: _RetainedDesktopCaptionSurface | None = None
        self._preview_stage: Any | None = None
        self._preview_backdrop: Any | None = None
        self._preview_busy_background: Any | None = None
        self._preview_option_buttons: dict[tuple[str, str], Any] = {}

    @property
    def startup_generation(self) -> int:
        coordinator = self._startup_coordinator
        return coordinator.generation if coordinator is not None and coordinator.ready else 0

    @property
    def startup_phase(self) -> str | None:
        coordinator = self._startup_coordinator
        return coordinator.phase.value if coordinator is not None else None

    def prime_startup_runtime_controls(
        self,
        payloads: tuple[dict[str, object], ...],
    ) -> tuple[dict[str, object], ...]:
        """Apply startup controls that must affect the first Flet page render.

        Returns controls that were not consumed during priming and still need
        normal runtime dispatch after the Flet page exists.
        """

        self._startup_visual_state = None
        self._startup_window_bounds = None
        residual: list[dict[str, object]] = []
        for payload in payloads:
            command = payload.get("command")
            if command is None and "logging_mode" in payload:
                if self._set_logging_mode(payload.get("logging_mode")):
                    continue
                residual.append(payload)
                continue
            if command == "set_interaction_mode":
                continue
            if command == "apply_visual_config":
                visual_state = _parse_runtime_visual_state(payload)
                if visual_state is not None:
                    self._startup_visual_state = visual_state
                else:
                    residual.append(payload)
                continue
            if command == "apply_window_bounds":
                bounds = _parse_runtime_window_bounds(payload)
                if bounds is not None:
                    self._startup_window_bounds = bounds
                else:
                    residual.append(payload)
                continue
            residual.append(payload)
        return tuple(residual)

    async def start(self, initial_snapshot: OverlayPresentationSnapshot) -> None:
        if self._preview_catalog is not None:
            self._snapshot = self._preview_selected_fixture().snapshot
            self._visual_state = self._preview_visual_state()
        else:
            self._snapshot = initial_snapshot
            self._visual_state = self._startup_visual_state or DesktopCaptionVisualState()
            previous_coordinator = self._startup_coordinator
            if previous_coordinator is not None:
                previous_coordinator.retire()
            self._startup_generation += 1
            self._startup_coordinator = DesktopOverlayStartupCoordinator(
                self._startup_generation,
                trace_sink=self._record_startup_lifecycle,
            )
        self._programmatic_bounds_signatures.clear()
        self._last_snapshot_revision = self._snapshot.revision
        self._page_ready.clear()
        self._closed.clear()
        self._page_start_error = None
        self._retained_caption_surface = None
        self._preview_stage = None
        self._preview_backdrop = None
        self._preview_busy_background = None
        self._preview_option_buttons.clear()
        self._interaction_mode = _DESKTOP_INTERACTION_MODE_EDIT
        if self._app_task is None or self._app_task.done():
            page_handler = (
                self._handle_preview_page
                if self._preview_catalog is not None
                else self._handle_page
            )
            self._app_task = asyncio.create_task(self._app_runner(page_handler))

        ready_task = asyncio.create_task(self._page_ready.wait())
        try:
            done, _pending = await asyncio.wait(
                {ready_task, self._app_task},
                timeout=self._startup_timeout_s,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if ready_task not in done:
                if self._app_task in done:
                    await self._app_task
                if self._page is not None:
                    phase = self.startup_phase or "launched"
                    raise RuntimeError(
                        f"desktop overlay startup timed out before ready (phase={phase})"
                    )
                raise RuntimeError("desktop overlay Flet page was not created")
            if self._page_start_error is not None:
                if self._app_task.done():
                    await asyncio.gather(self._app_task, return_exceptions=True)
                raise RuntimeError(
                    "desktop overlay Flet page configuration failed"
                ) from self._page_start_error
        finally:
            if not ready_task.done():
                ready_task.cancel()
            await asyncio.gather(ready_task, return_exceptions=True)

    async def run_until_closed(self) -> None:
        task = self._app_task
        if task is None:
            await self._closed.wait()
            return
        try:
            await task
        finally:
            self._closed.set()

    async def close(self) -> None:
        self._closed.set()
        coordinator = self._startup_coordinator
        if coordinator is not None:
            coordinator.retire()
        self._programmatic_bounds_signatures.clear()
        async with self._interaction_mode_lock:
            self._interaction_generation += 1
            await self._cancel_window_z_order_task()
        page = self._page
        if page is not None:
            page.window.on_event = None
        await self._cancel_scheduled_callback_tasks()
        await self._cancel_bounds_sample()

        async def close_page_window() -> None:
            if page is None:
                return
            window = page.window
            try:
                await invoke_control_method(window, "close")
            except Exception:
                destroy = getattr(window, "destroy", None)
                if callable(destroy):
                    with contextlib.suppress(Exception):
                        await invoke_control_method(window, "destroy")

        process_owner = self._view_process_owner
        if process_owner is None:
            await close_page_window()
        else:
            await process_owner.close()

        task = self._app_task
        if task is not None and not task.done():
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=1.0)
            except TimeoutError:
                if page is not None:
                    destroy = getattr(page.window, "destroy", None)
                    if callable(destroy):
                        with contextlib.suppress(Exception):
                            await invoke_control_method(page.window, "destroy")
                    try:
                        await asyncio.wait_for(asyncio.shield(task), timeout=0.5)
                    except TimeoutError:
                        task.cancel()
                        await asyncio.gather(task, return_exceptions=True)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(
                    "[DesktopOverlay] Window cleanup failed: exception_type=%s",
                    type(exc).__name__,
                )
        self._window_z_order_port.close()

    async def dispatch_snapshot(self, snapshot: OverlayPresentationSnapshot) -> None:
        if snapshot.revision <= self._last_snapshot_revision:
            return
        self._emit_detailed_log(
            f"snapshot_update revision={snapshot.revision} blocks={len(snapshot.blocks)}"
        )
        self._snapshot = snapshot
        self._last_snapshot_revision = snapshot.revision
        self._render_page()

    async def advance_snapshot_history(self, snapshot: OverlayPresentationSnapshot) -> None:
        page = self._page
        if page is None or self._preview_catalog is not None:
            return
        plan = build_desktop_caption_plan(
            snapshot,
            window_width=_page_window_number(page, "width", DESKTOP_FLET_DEFAULT_WIDTH),
            window_height=_page_window_number(page, "height", DESKTOP_FLET_DEFAULT_HEIGHT),
            visual_state=self._visual_state,
            interaction_mode=self._interaction_mode,
            locale=self._locale,
        )
        self._plan_with_grow_only_caption_card_widths(plan)

    def renderer_visual_state(self) -> dict[str, object]:
        trace = self._last_render_trace
        if trace is None:
            return {
                "slot_count": 0,
                "line_count": 0,
                "surface_visible": False,
                "interaction_mode": "locked",
                "window_width": DESKTOP_FLET_DEFAULT_WIDTH,
                "window_height": DESKTOP_FLET_DEFAULT_HEIGHT,
            }
        return {
            "slot_count": trace.slot_count,
            "line_count": trace.line_count,
            "surface_visible": trace.surface_visible,
            "interaction_mode": (
                "edit" if self._interaction_mode == _DESKTOP_INTERACTION_MODE_EDIT else "locked"
            ),
            "window_width": trace.window_width,
            "window_height": trace.window_height,
        }

    def renderer_visual_state_for_snapshot(
        self,
        snapshot: OverlayPresentationSnapshot,
    ) -> dict[str, object]:
        page = self._page
        if page is None:
            return self.renderer_visual_state()
        plan = build_desktop_caption_plan(
            snapshot,
            window_width=_page_window_number(page, "width", DESKTOP_FLET_DEFAULT_WIDTH),
            window_height=_page_window_number(page, "height", DESKTOP_FLET_DEFAULT_HEIGHT),
            visual_state=self._visual_state,
            interaction_mode=self._interaction_mode,
            locale=self._locale,
        )
        return {
            "slot_count": len(plan.slots),
            "line_count": len(plan.lines),
            "surface_visible": plan.surface_visible,
            "interaction_mode": (
                "edit" if self._interaction_mode == _DESKTOP_INTERACTION_MODE_EDIT else "locked"
            ),
            "window_width": plan.window_width,
            "window_height": plan.window_height,
        }

    async def dispatch_runtime_control(self, payload: dict[str, object]) -> None:
        if "logging_mode" in payload and payload.get("command") is None:
            self._set_logging_mode(payload.get("logging_mode"))
            return
        command = payload.get("command")
        if command == "set_interaction_mode":
            mode = payload.get("mode")
            if not isinstance(mode, str) or mode not in _DESKTOP_INTERACTION_MODES:
                logger.warning("[DesktopOverlay] Ignoring invalid interaction mode control")
                return
            await self._set_interaction_mode(mode, emit_event=True)
            return
        if command == "apply_window_bounds":
            bounds = _parse_runtime_window_bounds(payload)
            if bounds is None:
                logger.warning("[DesktopOverlay] Ignoring invalid window bounds control")
                return
            self._emit_detailed_log(
                "runtime_control command=apply_window_bounds "
                f"x={bounds['x']} y={bounds['y']} width={bounds['width']} "
                f"height={bounds['height']}"
            )
            await self._cancel_bounds_sample()
            self._apply_window_bounds(bounds)
            return
        if command == "apply_visual_config":
            visual_state = _parse_runtime_visual_state(payload)
            if visual_state is None:
                logger.warning("[DesktopOverlay] Ignoring invalid visual config control")
                return
            self._visual_state = visual_state
            self._emit_detailed_log(
                "runtime_control command=apply_visual_config "
                f"text_scale={visual_state.text_scale} "
                f"background_alpha={visual_state.background_alpha} "
                f"outline_width={visual_state.outline_width} "
                f"swap_caption_languages={visual_state.swap_caption_languages}"
            )
            self._render_page()
            return
        logger.warning("[DesktopOverlay] Ignoring unsupported desktop runtime control: %r", command)

    async def _handle_page(self, page: Any) -> None:
        if self._closed.is_set():
            self._page_ready.set()
            await self._close_late_page(page)
            return
        self._page = page
        coordinator = self._startup_coordinator
        if coordinator is None:
            self._page_start_error = RuntimeError("desktop overlay startup coordinator is missing")
            self._page_ready.set()
            raise self._page_start_error
        try:
            self._bind_window_z_order_process()
            self._configure_base_window(page)
            self._render_page()
            coordinator.advance(DesktopOverlayStartupPhase.PAGE_CONFIGURED)
        except Exception as exc:
            self._page_start_error = exc
            self._page_ready.set()
            raise
        await self._finish_hidden_window_startup(coordinator)

    def _handle_preview_page(self, page: Any) -> None:
        if self._closed.is_set():
            self._page_ready.set()
            return
        self._page = page
        try:
            self._configure_base_window(page)
            self._render_page()
            self._page_ready.set()
        except Exception as exc:
            self._page_start_error = exc
            self._page_ready.set()
            raise

    async def _finish_hidden_window_startup(
        self,
        coordinator: DesktopOverlayStartupCoordinator,
    ) -> None:
        try:
            page = self._page
            if page is None:
                raise RuntimeError("desktop overlay Flet page is missing")
            try:
                await asyncio.wait_for(
                    invoke_control_method(page.window, "wait_until_ready_to_show"),
                    timeout=self._wait_until_ready_timeout_s,
                )
            except TimeoutError as exc:
                raise RuntimeError("desktop overlay native window was not ready to show") from exc
            coordinator.advance(DesktopOverlayStartupPhase.NATIVE_READY)
            bounds_confirmation = await self._confirm_window_bounds()
            coordinator.advance(
                DesktopOverlayStartupPhase.BOUNDS_CONFIRMED,
                canonical_bounds=dict(self._startup_window_bounds or {}),
                observed_bounds=(
                    bounds_confirmation.observed_bounds if bounds_confirmation is not None else None
                ),
            )
            if self._closed.is_set() or not coordinator.accepts(self._startup_generation):
                raise RuntimeError("desktop overlay startup generation was retired")
            visibility_confirmation = await self._show_configured_window()
            coordinator.advance(
                DesktopOverlayStartupPhase.VISIBLE_CONFIRMED,
                canonical_bounds=dict(self._startup_window_bounds or {}),
                observed_bounds=(
                    visibility_confirmation.observed_bounds
                    if visibility_confirmation is not None
                    else None
                ),
            )
            coordinator.advance(DesktopOverlayStartupPhase.READY)
        except Exception as exc:
            self._page_start_error = exc
            raise
        finally:
            self._page_ready.set()

    async def _close_late_page(self, page: Any) -> None:
        window = page.window
        try:
            await invoke_control_method(window, "close")
        except Exception:
            destroy = getattr(window, "destroy", None)
            if callable(destroy):
                with contextlib.suppress(Exception):
                    await invoke_control_method(window, "destroy")

    def _record_flet_process(self, pid: int, pid_file: str | None) -> None:
        if self._closed.is_set():
            return
        self._flet_process_pid = int(pid) if int(pid) > 0 else None
        self._flet_pid_file = pid_file

    def _record_process_lifecycle(self, event: str, fields: dict[str, object]) -> None:
        if self._structured_lifecycle_trace_enabled:
            _emit_desktop_lifecycle_trace("flet_view_process", event, fields)
        details = " ".join(f"{key}={value}" for key, value in sorted(fields.items()))
        self._emit_detailed_log(f"flet_process event={event} {details}".rstrip())

    def _record_startup_lifecycle(self, event: str, fields: dict[str, object]) -> None:
        if self._structured_lifecycle_trace_enabled:
            _emit_desktop_lifecycle_trace(
                "desktop_window_startup",
                event,
                {"geometry_authority": "flet", **fields},
            )
        details = " ".join(f"{key}={value}" for key, value in sorted(fields.items()))
        self._emit_detailed_log(f"startup event={event} {details}".rstrip())

    def _bind_window_z_order_process(self) -> None:
        provider = self._window_process_info_provider
        if provider is not None:
            process_info = provider()
            if process_info is not None:
                self._record_flet_process(*process_info)
        pid = self._flet_process_pid
        source = "launcher"
        pid_file = self._flet_pid_file
        if pid_file:
            with contextlib.suppress(Exception):
                recorded_pid = int(Path(pid_file).read_text(encoding="utf-8").strip())
                if recorded_pid > 0:
                    pid = recorded_pid
                    source = "pid_file"
        if pid is None:
            return
        self._window_z_order_port.bind_process(pid)
        self._emit_detailed_log(f"window_process_bound source={source} pid={pid}")

    def _configure_base_window(self, page: Any) -> None:
        import flet as ft

        window = page.window
        register_fonts(page)
        page.title = self._window_title()
        window.icon = "icons/icon.ico"
        window.frameless = True
        window.always_on_top = True
        window.shadow = False
        window.skip_task_bar = False
        window.resizable = False
        window.maximizable = False
        window.bgcolor = ft.Colors.TRANSPARENT
        if self._preview_catalog is None:
            window.visible = False
        window.ignore_mouse_events = (
            self._interaction_mode == _DESKTOP_INTERACTION_MODE_PASS_THROUGH
        )
        if self._preview_catalog is not None:
            size_preset = self._preview_selected_size_preset()
            window.width = max(
                size_preset.window_width,
                _DESKTOP_PREVIEW_STAGE_WIDTH,
            )
            window.height = max(size_preset.window_height, _DESKTOP_PREVIEW_STAGE_HEIGHT)
        elif self._startup_window_bounds is not None:
            bounds = self._startup_window_bounds
            window.left = bounds["x"]
            window.top = bounds["y"]
            window.width = bounds["width"]
            window.height = bounds["height"]
            self._track_programmatic_bounds(bounds)
            coordinator = self._startup_coordinator
            if coordinator is not None:
                coordinator.record("bounds_applied", canonical_bounds=dict(bounds))
        elif _finite_non_bool_number(getattr(window, "width", None)) in {None, 0}:
            window.width = DESKTOP_FLET_DEFAULT_WIDTH
        if self._preview_catalog is None and _finite_non_bool_number(
            getattr(window, "height", None)
        ) in {None, 0}:
            window.height = DESKTOP_FLET_DEFAULT_HEIGHT
        window.on_event = self._on_window_event
        if hasattr(window, "min_width"):
            window.min_width = DESKTOP_FLET_MIN_WIDTH
        if hasattr(window, "min_height"):
            window.min_height = DESKTOP_FLET_MIN_HEIGHT
        if self._preview_catalog is not None:
            page.on_keyboard_event = self._on_preview_keyboard_event
        page.bgcolor = ft.Colors.TRANSPARENT
        if hasattr(page, "padding"):
            page.padding = 0
        if hasattr(page, "spacing"):
            page.spacing = 0

    def _on_empty_lock_action_click(self, _event: object | None = None) -> None:
        self._run_page_task(self._lock_from_empty_action)

    async def _lock_from_empty_action(self) -> None:
        await self._set_interaction_mode(
            _DESKTOP_INTERACTION_MODE_PASS_THROUGH,
            emit_event=True,
        )

    def _render_page(self) -> None:
        page = self._page
        if page is None:
            return
        import flet as ft

        if self._preview_catalog is not None:
            plan = self._current_preview_caption_plan()
            if self._retained_caption_surface is None:
                root = self._build_preview_root(ft, plan)
                page.add(root)
                self._apply_interaction_window_chrome()
            else:
                self._apply_preview_surface(ft, plan)
            page.update()
            return

        raw_plan = build_desktop_caption_plan(
            self._snapshot,
            window_width=_page_window_number(page, "width", DESKTOP_FLET_DEFAULT_WIDTH),
            window_height=_page_window_number(page, "height", DESKTOP_FLET_DEFAULT_HEIGHT),
            visual_state=self._visual_state,
            interaction_mode=self._interaction_mode,
            locale=self._locale,
        )
        previous_width_floors = dict(self._caption_card_width_floor_by_block)
        plan = self._plan_with_grow_only_caption_card_widths(raw_plan)
        self._emit_caption_width_diagnostics(raw_plan, plan, previous_width_floors)
        if self._interaction_mode == _DESKTOP_INTERACTION_MODE_EDIT:
            content_kind = (
                "drag_area_with_empty_lock_action"
                if plan.full_window_background_visible and not plan.slots
                else "drag_area"
            )
        else:
            content_kind = "caption_surface" if plan.surface_visible else "transparent_host"
        self._emit_detailed_log(
            "render "
            f"revision={self._snapshot.revision} "
            f"blocks={len(self._snapshot.blocks)} "
            f"interaction_mode={self._interaction_mode} "
            f"surface_visible={plan.surface_visible} "
            f"line_count={len(plan.lines)} "
            f"content_kind={content_kind} "
            f"window={plan.window_width}x{plan.window_height} "
            f"background_alpha={plan.background_alpha}"
        )
        self._emit_render_transition(
            _DesktopRenderTrace(
                content_kind=content_kind,
                surface_visible=plan.surface_visible,
                slot_count=len(plan.slots),
                line_count=len(plan.lines),
                window_width=plan.window_width,
                window_height=plan.window_height,
                background_alpha=plan.background_alpha,
            )
        )
        if self._retained_caption_surface is None:
            self._retained_caption_surface = _build_retained_desktop_caption_surface(
                ft,
                plan,
                empty_lock_label=desktop_empty_lock_action_label(self._locale),
                on_empty_lock=self._on_empty_lock_action_click,
                include_drag_area=True,
            )
            page.add(self._retained_caption_surface.root)
            self._apply_interaction_window_chrome()
        else:
            _apply_retained_desktop_caption_plan(
                ft,
                self._retained_caption_surface,
                plan,
                empty_lock_label=desktop_empty_lock_action_label(self._locale),
            )
        page.update()

    def _apply_interaction_window_chrome(self) -> None:
        page = self._page
        if page is None:
            return
        locked = self._interaction_mode == _DESKTOP_INTERACTION_MODE_PASS_THROUGH
        window = page.window
        window.ignore_mouse_events = locked

    async def _show_configured_window(
        self,
    ) -> WindowVisibilityConfirmation | None:
        page = self._page
        if page is None:
            return None
        window = page.window
        coordinator = self._startup_coordinator
        if coordinator is not None:
            coordinator.record("show_requested")
        window.visible = True
        page.update()
        return await self._confirm_window_visible()

    async def _confirm_window_bounds(
        self,
    ) -> WindowBoundsConfirmation | None:
        bounds = self._startup_window_bounds
        if bounds is None:
            return None
        title = self._window_title()
        try:
            result = await self._window_z_order_port.confirm_window_bounds(
                title,
                x=int(round(float(bounds["x"]))),
                y=int(round(float(bounds["y"]))),
                width=int(round(float(bounds["width"]))),
                height=int(round(float(bounds["height"]))),
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._emit_detailed_log(
                "window_bounds_confirmation "
                f"reason=port_error exception_type={type(exc).__name__}"
            )
            if self._window_z_order_required:
                logger.warning(
                    "[DesktopOverlay] Desktop overlay window bounds confirmation failed: "
                    "reason=port_error exception_type=%s",
                    type(exc).__name__,
                )
            return None
        self._emit_detailed_log(
            "window_bounds_confirmation "
            f"reason={result.reason} confirmed={result.confirmed} "
            f"title_confirmed={result.title_confirmed} "
            f"bounds_confirmed={result.bounds_confirmed} "
            f"win32_error={result.win32_error}"
        )
        if self._window_z_order_required and not result.confirmed:
            raise RuntimeError(
                "desktop overlay canonical bounds were not confirmed: "
                f"reason={result.reason} win32_error={result.win32_error}"
            )
        return result

    async def _confirm_window_visible(
        self,
    ) -> WindowVisibilityConfirmation | None:
        title = self._window_title()
        bounds = self._startup_window_bounds
        if bounds is None:
            return None
        try:
            result = await self._window_z_order_port.confirm_window_visible(
                title,
                x=int(round(float(bounds["x"]))),
                y=int(round(float(bounds["y"]))),
                width=int(round(float(bounds["width"]))),
                height=int(round(float(bounds["height"]))),
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._emit_detailed_log(
                "window_visibility_confirmation "
                f"reason=port_error exception_type={type(exc).__name__}"
            )
            if self._window_z_order_required:
                logger.warning(
                    "[DesktopOverlay] Desktop overlay window visibility confirmation failed: "
                    "reason=port_error exception_type=%s",
                    type(exc).__name__,
                )
            return None
        self._emit_detailed_log(
            "window_visibility_confirmation "
            f"reason={result.reason} confirmed={result.confirmed} "
            f"title_confirmed={result.title_confirmed} "
            f"visible_confirmed={result.visible_confirmed} "
            f"bounds_confirmed={result.bounds_confirmed} "
            f"win32_error={result.win32_error}"
        )
        if self._window_z_order_required and not result.confirmed:
            raise RuntimeError(
                "desktop overlay visibility was not confirmed: "
                f"reason={result.reason} win32_error={result.win32_error}"
            )
        return result

    def _window_title(self) -> str:
        return t_for_locale(
            self._locale,
            "desktop_overlay.window.title",
            default="PuriPuly Overlay",
        )

    def _build_preview_root(self, ft: Any, plan: DesktopCaptionPlan) -> Any:
        self._retained_caption_surface = _build_retained_desktop_caption_surface(
            ft,
            plan,
            empty_lock_label=desktop_empty_lock_action_label(self._locale),
            on_empty_lock=self._on_empty_lock_action_click,
            include_drag_area=False,
        )
        self._preview_busy_background = self._build_preview_busy_background(
            ft,
            self._preview_selected_size_preset(),
        )
        self._preview_stage = ft.Stack(
            controls=[self._preview_busy_background, self._retained_caption_surface.surface_host],
            alignment=ft.Alignment.CENTER,
        )
        self._preview_backdrop = ft.Container(
            content=self._preview_stage,
            padding=24,
            border_radius=20,
            alignment=ft.Alignment.CENTER,
        )
        root = ft.Container(
            content=ft.Column(
                controls=[
                    self._build_preview_controls(ft),
                    self._preview_backdrop,
                ],
                spacing=12,
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                scroll=ft.ScrollMode.AUTO,
            ),
            padding=16,
            bgcolor="#101827",
            alignment=ft.Alignment.CENTER,
        )
        self._apply_preview_surface(ft, plan)
        return root

    def _apply_preview_surface(self, ft: Any, plan: DesktopCaptionPlan) -> None:
        model = self._retained_caption_surface
        stage = self._preview_stage
        backdrop = self._preview_backdrop
        busy_background = self._preview_busy_background
        if model is None or stage is None or backdrop is None or busy_background is None:
            return
        _apply_retained_desktop_caption_plan(
            ft,
            model,
            plan,
            empty_lock_label=desktop_empty_lock_action_label(self._locale),
        )
        size_preset = self._preview_selected_size_preset()
        background_surface = self._preview_selected_background_surface()
        stage.width = size_preset.window_width
        stage.height = size_preset.window_height
        backdrop.width = size_preset.window_width
        backdrop.height = size_preset.window_height
        backdrop.bgcolor = background_surface.bgcolor
        busy_background.visible = background_surface.id == "busy"
        busy_background.width = size_preset.window_width
        busy_background.height = size_preset.window_height
        self._refresh_preview_option_buttons()

    def _refresh_preview_option_buttons(self) -> None:
        selected_by_group = {
            "fixture": self._preview_fixture_id,
            "size_preset": self._preview_size_preset_id,
            "background_alpha": str(self._preview_background_alpha),
            "background_surface": self._preview_background_surface_id,
        }
        for (group, value), button in self._preview_option_buttons.items():
            button.disabled = selected_by_group.get(group) == value

    def _build_preview_controls(self, ft: Any) -> Any:
        catalog = self._preview_catalog
        if catalog is None:
            return ft.Container()
        labels = catalog.labels
        return ft.Column(
            controls=[
                self._build_preview_button_group(
                    ft,
                    labels.fixture,
                    "fixture",
                    [
                        (fixture.id, fixture.label, fixture.id == self._preview_fixture_id)
                        for fixture in catalog.fixtures
                    ],
                    self._set_preview_fixture,
                ),
                self._build_preview_button_group(
                    ft,
                    labels.size_preset,
                    "size_preset",
                    [
                        (preset.id, preset.label, preset.id == self._preview_size_preset_id)
                        for preset in catalog.size_presets
                    ],
                    self._set_preview_size_preset,
                ),
                self._build_preview_button_group(
                    ft,
                    labels.background_alpha,
                    "background_alpha",
                    [
                        (
                            str(value),
                            _background_transparency_label_for_alpha(value),
                            value == self._preview_background_alpha,
                        )
                        for value in catalog.background_alpha_presets
                    ],
                    lambda value: self._set_preview_background_alpha(float(value)),
                ),
                self._build_preview_button_group(
                    ft,
                    labels.background_surface,
                    "background_surface",
                    [
                        (
                            surface.id,
                            surface.label,
                            surface.id == self._preview_background_surface_id,
                        )
                        for surface in catalog.background_surfaces
                    ],
                    self._set_preview_background_surface,
                ),
            ],
            spacing=6,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            tight=True,
        )

    def _build_preview_button_group(
        self,
        ft: Any,
        label: str,
        group: str,
        items: list[tuple[str, str, bool]],
        on_select: Callable[[str], None],
    ) -> Any:
        buttons = []
        for value, text, selected in items:
            button = ft.ElevatedButton(
                content=text,
                on_click=lambda _event, selected_value=value: self._select_preview(
                    selected_value,
                    on_select,
                ),
                disabled=selected,
                style=ft.ButtonStyle(elevation=0),
            )
            self._preview_option_buttons[(group, value)] = button
            buttons.append(button)
        return ft.Column(
            controls=[
                ft.Text(label, size=12, weight=ft.FontWeight.BOLD, color="#FFE7D6"),
                ft.Row(
                    controls=buttons,
                    spacing=6,
                    alignment=ft.MainAxisAlignment.CENTER,
                    wrap=True,
                ),
            ],
            spacing=4,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            tight=True,
        )

    def _build_preview_surface_backdrop(self, ft: Any, caption_surface: Any) -> Any:
        surface = self._preview_selected_background_surface()
        size_preset = self._preview_selected_size_preset()
        controls: list[Any] = []
        if surface.id == "busy":
            controls.append(self._build_preview_busy_background(ft, size_preset))
        controls.append(caption_surface)
        content: Any = caption_surface
        if len(controls) > 1:
            content = ft.Stack(
                controls=controls,
                width=size_preset.window_width,
                height=size_preset.window_height,
                alignment=ft.Alignment.CENTER,
            )
        return ft.Container(
            content=content,
            width=size_preset.window_width,
            height=size_preset.window_height,
            bgcolor=surface.bgcolor,
            padding=24,
            border_radius=20,
            alignment=ft.Alignment.CENTER,
        )

    def _build_preview_busy_background(
        self,
        ft: Any,
        size_preset: DesktopOverlayPreviewSizePreset,
    ) -> Any:
        colors = (
            "#475569",
            "#7C3AED",
            "#0EA5E9",
            "#F97316",
            "#22C55E",
            "#334155",
        )
        rows = []
        for row_index in range(5):
            rows.append(
                ft.Row(
                    controls=[
                        ft.Container(
                            width=140 + (column_index % 3) * 46,
                            height=54 + ((row_index + column_index) % 2) * 22,
                            bgcolor=colors[(row_index + column_index) % len(colors)],
                            border_radius=14,
                            opacity=0.72,
                        )
                        for column_index in range(5)
                    ],
                    spacing=12,
                    alignment=ft.MainAxisAlignment.CENTER,
                )
            )
        return ft.Container(
            content=ft.Column(
                controls=rows,
                spacing=12,
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            ),
            width=size_preset.window_width,
            height=size_preset.window_height,
            alignment=ft.Alignment.CENTER,
        )

    def _select_preview(self, value: str, on_select: Callable[[str], None]) -> None:
        on_select(value)
        self._render_page()

    def _set_preview_fixture(self, fixture_id: str) -> None:
        catalog = self._preview_catalog
        if catalog is None or not any(fixture.id == fixture_id for fixture in catalog.fixtures):
            return
        self._preview_fixture_id = fixture_id
        self._snapshot = self._preview_selected_fixture().snapshot

    def _set_preview_background_alpha(self, value: float) -> None:
        catalog = self._preview_catalog
        if catalog is None or value not in catalog.background_alpha_presets:
            return
        self._preview_background_alpha = value
        self._visual_state = self._preview_visual_state()

    def _set_preview_size_preset(self, preset_id: str) -> None:
        catalog = self._preview_catalog
        if catalog is None or not any(preset.id == preset_id for preset in catalog.size_presets):
            return
        self._preview_size_preset_id = preset_id
        self._apply_preview_window_size()
        self._visual_state = self._preview_visual_state()

    def _set_preview_background_surface(self, surface_id: str) -> None:
        catalog = self._preview_catalog
        if catalog is None or not any(
            surface.id == surface_id for surface in catalog.background_surfaces
        ):
            return
        self._preview_background_surface_id = surface_id

    def _preview_selected_fixture(self) -> DesktopOverlayPreviewFixture:
        catalog = self._preview_catalog
        assert catalog is not None
        for fixture in catalog.fixtures:
            if fixture.id == self._preview_fixture_id:
                return fixture
        return catalog.fixtures[0]

    def _preview_selected_background_surface(self) -> DesktopOverlayPreviewBackgroundSurface:
        catalog = self._preview_catalog
        assert catalog is not None
        for surface in catalog.background_surfaces:
            if surface.id == self._preview_background_surface_id:
                return surface
        return catalog.background_surfaces[0]

    def _preview_selected_size_preset(self) -> DesktopOverlayPreviewSizePreset:
        catalog = self._preview_catalog
        assert catalog is not None
        for preset in catalog.size_presets:
            if preset.id == self._preview_size_preset_id:
                return preset
        return catalog.size_presets[1]

    def _apply_preview_window_size(self) -> None:
        page = self._page
        if page is None or self._preview_catalog is None:
            return
        preset = self._preview_selected_size_preset()
        page.window.width = preset.window_width
        page.window.height = preset.window_height

    def _current_preview_caption_plan(self) -> DesktopCaptionPlan:
        preset = self._preview_selected_size_preset()
        plan = build_desktop_caption_plan(
            self._preview_selected_fixture().snapshot,
            window_width=preset.window_width,
            window_height=preset.window_height,
            visual_state=self._preview_visual_state(),
            interaction_mode=self._interaction_mode,
            locale=self._locale,
        )
        return self._plan_with_grow_only_caption_card_widths(plan)

    def _plan_with_grow_only_caption_card_widths(
        self,
        plan: DesktopCaptionPlan,
    ) -> DesktopCaptionPlan:
        if not plan.slots:
            self._caption_card_width_floor_by_block.clear()
            return plan
        if plan.full_window_background_visible:
            return plan

        active_keys = {_caption_card_width_memory_key(slot) for slot in plan.slots}
        for key in tuple(self._caption_card_width_floor_by_block):
            if key not in active_keys:
                del self._caption_card_width_floor_by_block[key]

        grown_slots: list[DesktopCaptionSlot] = []
        for slot in plan.slots:
            key = _caption_card_width_memory_key(slot)
            previous_width = self._caption_card_width_floor_by_block.get(key, 0.0)
            card_width = _clamp(max(slot.card_width, previous_width), 1.0, float(plan.window_width))
            self._caption_card_width_floor_by_block[key] = card_width
            grown_slots.append(
                replace(
                    slot,
                    card_width=card_width,
                    card_text_width=max(1.0, card_width - (plan.padding_horizontal * 2)),
                )
            )
        return replace(
            plan,
            slots=tuple(grown_slots),
            lines=tuple(line for slot in grown_slots for line in slot.lines),
        )

    def _emit_caption_width_diagnostics(
        self,
        raw_plan: DesktopCaptionPlan,
        applied_plan: DesktopCaptionPlan,
        previous_width_floors: dict[tuple[str, str, int], float],
    ) -> None:
        if self._logging_mode != "detailed":
            return
        raw_slots_by_key = {_caption_card_width_memory_key(slot): slot for slot in raw_plan.slots}
        for slot_index, slot in enumerate(applied_plan.slots):
            key = _caption_card_width_memory_key(slot)
            raw_slot = raw_slots_by_key.get(key)
            if raw_slot is None:
                continue
            previous_floor = previous_width_floors.get(key, 0.0)
            floor_hit = slot.card_width > raw_slot.card_width + 0.01
            self._emit_detailed_log(
                "render_width "
                f"revision={self._snapshot.revision} "
                f"slot={slot_index} "
                f"raw_card_width={raw_slot.card_width:.1f} "
                f"applied_card_width={slot.card_width:.1f} "
                f"raw_text_width={raw_slot.card_text_width:.1f} "
                f"applied_text_width={slot.card_text_width:.1f} "
                f"previous_floor={previous_floor:.1f} "
                f"floor_hit={floor_hit} "
                f"line_count={len(slot.lines)}"
            )

    def _emit_render_transition(self, trace: _DesktopRenderTrace) -> None:
        previous = self._last_render_trace
        self._last_render_trace = trace
        if previous is None:
            return
        self._emit_detailed_log(
            "render_transition "
            f"revision={self._snapshot.revision} "
            f"content_kind {previous.content_kind}->{trace.content_kind} "
            f"surface_visible {previous.surface_visible}->{trace.surface_visible} "
            f"slot_count {previous.slot_count}->{trace.slot_count} "
            f"line_count {previous.line_count}->{trace.line_count} "
            f"window {previous.window_width}x{previous.window_height}->"
            f"{trace.window_width}x{trace.window_height} "
            f"background_alpha {previous.background_alpha:.3f}->{trace.background_alpha:.3f}"
        )

    def _preview_visual_state(self) -> DesktopCaptionVisualState:
        return DesktopCaptionVisualState(
            background_alpha=self._preview_background_alpha,
        )

    def _on_preview_keyboard_event(self, event: object) -> None:
        key = str(getattr(event, "key", "")).lower()
        if key not in {"e", "escape"}:
            return
        self._run_page_task(self._return_preview_to_edit_mode)

    async def _return_preview_to_edit_mode(self) -> None:
        await self._set_interaction_mode(_DESKTOP_INTERACTION_MODE_EDIT, emit_event=True)

    async def _set_interaction_mode(self, mode: str, *, emit_event: bool) -> None:
        async with self._interaction_mode_lock:
            if self._closed.is_set():
                return
            if mode not in _DESKTOP_INTERACTION_MODES:
                return
            if mode == self._interaction_mode:
                return
            previous_mode = self._interaction_mode
            self._interaction_mode = mode
            self._interaction_generation += 1
            generation = self._interaction_generation
            await self._cancel_window_z_order_task()
            if self._closed.is_set():
                return
            self._emit_detailed_log(f"interaction_mode {previous_mode}->{mode}")
            self._apply_interaction_window_chrome()
            self._render_page()
            if mode == _DESKTOP_INTERACTION_MODE_PASS_THROUGH:

                async def reassert_window_z_order() -> None:
                    await self._reassert_window_z_order(generation)

                task = self._run_page_task(reassert_window_z_order)
                self._window_z_order_task = task
                if task is not None:
                    task.add_done_callback(self._clear_window_z_order_task)
            if emit_event:
                await self._emit_overlay_event({"event": "interaction_mode_changed", "mode": mode})

    async def _reassert_window_z_order(self, generation: int) -> None:
        try:
            result = await self._window_z_order_port.reassert_topmost_after_click_through()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if not self._window_z_order_result_is_current(generation):
                return
            self._emit_detailed_log(
                f"topmost_reassert reason=port_error exception_type={type(exc).__name__}"
            )
            if self._window_z_order_required:
                logger.warning(
                    "[DesktopOverlay] Topmost z-order re-assertion failed: "
                    "reason=port_error exception_type=%s",
                    type(exc).__name__,
                )
        else:
            if not self._window_z_order_result_is_current(generation):
                return
            self._emit_detailed_log(
                "topmost_reassert "
                f"reason={result.reason} applied={result.applied} "
                f"click_through_confirmed={result.click_through_confirmed} "
                f"topmost_style_present={result.topmost_style_present} "
                f"win32_error={result.win32_error}"
            )
            if self._window_z_order_required and not result.applied:
                logger.warning(
                    "[DesktopOverlay] Topmost z-order re-assertion failed: "
                    "reason=%s win32_error=%s",
                    result.reason,
                    result.win32_error,
                )

    def _window_z_order_result_is_current(self, generation: int) -> bool:
        return (
            not self._closed.is_set()
            and self._interaction_mode == _DESKTOP_INTERACTION_MODE_PASS_THROUGH
            and self._interaction_generation == generation
        )

    def _clear_window_z_order_task(self, task: object) -> None:
        if self._window_z_order_task is task:
            self._window_z_order_task = None

    async def _cancel_window_z_order_task(self) -> None:
        task = self._window_z_order_task
        self._window_z_order_task = None
        if task is None or task.done():
            return
        task.cancel()
        awaitable = task if isinstance(task, asyncio.Future) else asyncio.wrap_future(task)
        await asyncio.gather(awaitable, return_exceptions=True)

    def _apply_window_bounds(self, bounds: dict[str, int | float]) -> None:
        page = self._page
        if page is None:
            return
        if _page_window_size_differs_from_bounds(page, bounds):
            self._caption_card_width_floor_by_block.clear()
        self._emit_detailed_log(
            "apply_window_bounds "
            f"x={bounds['x']} y={bounds['y']} width={bounds['width']} "
            f"height={bounds['height']}"
        )
        self._apply_window_bounds_without_rerender(bounds)
        self._track_programmatic_bounds(bounds)
        self._render_page()

    def _apply_window_bounds_without_rerender(self, bounds: dict[str, int | float]) -> None:
        page = self._page
        if page is None:
            return
        window = page.window
        window.left = bounds["x"]
        window.top = bounds["y"]
        window.width = bounds["width"]
        window.height = bounds["height"]

    def _on_window_event(self, event: object) -> None:
        if self._closed.is_set():
            return
        if not _is_window_bounds_event(event):
            return
        coordinator = self._startup_coordinator
        if coordinator is None or not coordinator.ready:
            self._emit_detailed_log("bounds_sample dropped reason=startup_not_ready")
            return
        generation = coordinator.generation
        self._emit_detailed_log(
            f"window_event type={getattr(event, 'type', getattr(event, 'data', None))} "
            f"interaction_mode={self._interaction_mode}"
        )
        if self._interaction_mode != _DESKTOP_INTERACTION_MODE_EDIT:
            self._emit_detailed_log(
                "bounds_sample dropped reason=event_interaction_mode "
                f"interaction_mode={self._interaction_mode}"
            )
            return

        async def schedule_bounds_sample() -> None:
            await self._schedule_bounds_sample(generation)

        self._run_page_task(schedule_bounds_sample)

    async def _schedule_bounds_sample(self, generation: int) -> None:
        if not self._startup_generation_is_ready(generation):
            return
        await self._cancel_bounds_sample()
        if not self._startup_generation_is_ready(generation):
            return
        self._emit_detailed_log(
            f"bounds_sample scheduled interaction_mode={self._interaction_mode}"
        )
        self._bounds_sample_task = asyncio.create_task(
            self._emit_debounced_bounds_sample(generation)
        )

    async def _cancel_bounds_sample(self) -> None:
        task = self._bounds_sample_task
        self._bounds_sample_task = None
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def _emit_debounced_bounds_sample(self, generation: int) -> None:
        if not self._startup_generation_is_ready(generation):
            return
        if self._bounds_debounce_s > 0:
            await asyncio.sleep(self._bounds_debounce_s)
        if not self._startup_generation_is_ready(generation):
            return
        bounds = _sample_page_window_bounds(self._page)
        if bounds is None:
            self._emit_detailed_log("bounds_sample dropped reason=no_bounds")
            return
        signature = _bounds_signature(bounds)
        if self._is_programmatic_bounds_echo(signature, generation):
            self._emit_detailed_log(
                "bounds_sample dropped reason=programmatic_echo "
                f"x={bounds['x']} y={bounds['y']} width={bounds['width']} "
                f"height={bounds['height']}"
            )
            return
        if self._interaction_mode != _DESKTOP_INTERACTION_MODE_EDIT:
            self._emit_detailed_log(
                "bounds_sample dropped reason=interaction_mode "
                f"interaction_mode={self._interaction_mode} "
                f"x={bounds['x']} y={bounds['y']} width={bounds['width']} "
                f"height={bounds['height']}"
            )
            return
        if signature == self._last_reported_bounds:
            self._emit_detailed_log(
                "bounds_sample dropped reason=unchanged "
                f"x={bounds['x']} y={bounds['y']} width={bounds['width']} "
                f"height={bounds['height']}"
            )
            return
        self._last_reported_bounds = signature
        self._emit_detailed_log(
            "bounds_sample emitted source=user persist=True "
            f"x={bounds['x']} y={bounds['y']} width={bounds['width']} "
            f"height={bounds['height']}"
        )
        await self._emit_overlay_event(
            {
                "event": "window_bounds_changed",
                "source": "user",
                "persist": True,
                "generation": generation,
                **bounds,
            }
        )

    def _run_page_task(self, func: Callable[[], Awaitable[None]]) -> ScheduledCallbackTask | None:
        if self._closed.is_set():
            return None
        page = self._page
        if page is not None:
            run_task = getattr(page, "run_task", None)
            if callable(run_task):
                task = run_task(func)
                self._track_scheduled_callback_task(task)
                return task if isinstance(task, (asyncio.Future, ConcurrentFuture)) else None
        task = asyncio.create_task(func())
        self._track_scheduled_callback_task(task)
        return task

    async def _emit_overlay_event(self, payload: dict[str, object]) -> None:
        if self._closed.is_set():
            return
        if self._event_sink is None:
            return
        await self._event_sink({"type": "overlay_event", "payload": payload})

    def _set_logging_mode(self, mode: object) -> bool:
        try:
            normalized_mode = normalize_overlay_logging_mode(mode)
        except Exception:
            return False
        self._logging_mode = normalized_mode
        self._emit_detailed_log(f"logging_mode mode={normalized_mode}")
        return True

    def _emit_detailed_log(self, message: str) -> None:
        if self._logging_mode != "detailed":
            return
        print(f"[DesktopOverlay][Detail] {message}", flush=True)

    def _track_programmatic_bounds(self, bounds: Mapping[str, int | float]) -> None:
        coordinator = self._startup_coordinator
        if coordinator is None or coordinator.retired:
            return
        self._programmatic_bounds_signatures.setdefault(coordinator.generation, set()).add(
            _bounds_signature(bounds)
        )

    def _startup_generation_is_ready(self, generation: int) -> bool:
        coordinator = self._startup_coordinator
        if self._closed.is_set() or coordinator is None:
            return False
        if not coordinator.accepts(generation):
            coordinator.reject("bounds_callback", generation)
            return False
        return coordinator.ready

    def _is_programmatic_bounds_echo(
        self,
        signature: tuple[float, float, float, float],
        generation: int,
    ) -> bool:
        signatures = self._programmatic_bounds_signatures.get(generation)
        if not signatures:
            return False
        matched = next(
            (
                candidate
                for candidate in signatures
                if _bounds_signatures_close(signature, candidate)
            ),
            None,
        )
        if matched is None:
            return False
        signatures.discard(matched)
        if not signatures:
            self._programmatic_bounds_signatures.pop(generation, None)
        return True

    def _track_scheduled_callback_task(self, task: object) -> None:
        if not isinstance(task, (asyncio.Future, ConcurrentFuture)):
            return
        self._scheduled_callback_tasks.add(task)
        task.add_done_callback(self._scheduled_callback_tasks.discard)

    async def _cancel_scheduled_callback_tasks(self) -> None:
        tasks = tuple(self._scheduled_callback_tasks)
        self._scheduled_callback_tasks.clear()
        if not tasks:
            return
        current_task = asyncio.current_task()
        awaitables: list[asyncio.Future[Any]] = []
        for task in tasks:
            if task is current_task:
                continue
            task.cancel()
            if isinstance(task, asyncio.Future):
                awaitables.append(task)
            else:
                awaitables.append(asyncio.wrap_future(task))
        if awaitables:
            await asyncio.gather(*awaitables, return_exceptions=True)


def _page_window_number(page: Any, field_name: str, default: int) -> int | float:
    return getattr(page.window, field_name, default) or default


def _page_window_size_differs_from_bounds(
    page: Any,
    bounds: dict[str, int | float],
) -> bool:
    window = page.window
    current_width = _finite_non_bool_number(getattr(window, "width", None))
    current_height = _finite_non_bool_number(getattr(window, "height", None))
    if current_width is None or current_height is None:
        return True
    width_changed = float(current_width) != float(bounds["width"])
    height_changed = float(current_height) != float(bounds["height"])
    return width_changed or height_changed


def _parse_runtime_window_bounds(
    payload: dict[str, object],
) -> dict[str, int | float] | None:
    x = _finite_non_bool_number(payload.get("x"))
    y = _finite_non_bool_number(payload.get("y"))
    width = _finite_non_bool_number(payload.get("width"))
    height = _finite_non_bool_number(payload.get("height"))
    if x is None or y is None or width is None or height is None:
        return None
    if width < DESKTOP_FLET_MIN_WIDTH or height < DESKTOP_FLET_MIN_HEIGHT:
        return None
    return {"x": x, "y": y, "width": width, "height": height}


def _parse_runtime_visual_state(payload: dict[str, object]) -> DesktopCaptionVisualState | None:
    text_scale = _finite_non_bool_number(payload.get("text_scale"))
    background_alpha = _finite_non_bool_number(payload.get("background_alpha"))
    outline_width_raw = payload.get("outline_width")
    if text_scale is None or background_alpha is None:
        return None
    if not DESKTOP_FLET_MIN_TEXT_SCALE <= text_scale <= DESKTOP_FLET_MAX_TEXT_SCALE:
        return None
    if (
        not DESKTOP_FLET_MIN_BACKGROUND_ALPHA
        <= background_alpha
        <= DESKTOP_FLET_MAX_BACKGROUND_ALPHA
    ):
        return None
    outline_width: float | None = None
    if outline_width_raw is not None:
        outline_number = _finite_non_bool_number(outline_width_raw)
        if outline_number is None:
            return None
        if not DESKTOP_FLET_MIN_OUTLINE_WIDTH <= outline_number <= DESKTOP_FLET_MAX_OUTLINE_WIDTH:
            return None
        outline_width = float(outline_number)
    swap_caption_languages = payload.get("swap_caption_languages", False)
    if not isinstance(swap_caption_languages, bool):
        return None
    return DesktopCaptionVisualState(
        text_scale=float(text_scale),
        background_alpha=float(background_alpha),
        outline_width=outline_width,
        swap_caption_languages=swap_caption_languages,
    )


def _sample_page_window_bounds(page: Any | None) -> dict[str, int | float] | None:
    if page is None:
        return None
    window = page.window
    bounds = {
        "x": _finite_non_bool_number(getattr(window, "left", None)),
        "y": _finite_non_bool_number(getattr(window, "top", None)),
        "width": _finite_non_bool_number(getattr(window, "width", None)),
        "height": _finite_non_bool_number(getattr(window, "height", None)),
    }
    if any(value is None for value in bounds.values()):
        return None
    typed_bounds = {key: value for key, value in bounds.items() if value is not None}
    if (
        typed_bounds["x"] == 0
        and typed_bounds["y"] == 0
        and typed_bounds["width"] == 0
        and typed_bounds["height"] == 0
    ):
        return None
    if typed_bounds["width"] <= 0 or typed_bounds["height"] <= 0:
        return None
    return typed_bounds


def _is_window_bounds_event(event: object) -> bool:
    event_type = getattr(event, "type", None)
    if event_type is None:
        event_type = getattr(event, "data", None)
    event_name = getattr(event_type, "name", None)
    if event_name is None:
        event_name = getattr(event_type, "value", None)
    if event_name is None:
        event_name = str(event_type)
    return str(event_name).split(".")[-1].upper() in _DESKTOP_WINDOW_BOUNDS_EVENT_NAMES


def _bounds_signature(bounds: dict[str, int | float]) -> tuple[float, float, float, float]:
    return (
        float(bounds["x"]),
        float(bounds["y"]),
        float(bounds["width"]),
        float(bounds["height"]),
    )


def _bounds_signatures_close(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> bool:
    return all(
        abs(left - right) <= _PROGRAMMATIC_BOUNDS_ECHO_TOLERANCE_PX
        for left, right in zip(first, second, strict=True)
    )


def _finite_non_bool_number(value: object) -> int | float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(value):
        return None
    return value


@dataclass(slots=True)
class PollingParentMonitor:
    parent_pid: int
    poll_interval_s: float = 1.0

    async def wait_for_parent_exit(self, stop_event: asyncio.Event) -> None:
        while not stop_event.is_set():
            if not self._pid_exists(self.parent_pid):
                return
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=self.poll_interval_s)
            except TimeoutError:
                continue

    @staticmethod
    def _pid_exists(parent_pid: int) -> bool:
        if parent_pid <= 0:
            return False
        try:
            os.kill(parent_pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except OSError:
            return True
        return True


@dataclass(slots=True)
class BridgeDisconnectParentMonitor:
    """Windows-safe fallback when no parent handle can be opened.

    The bridge connection is owned by the parent process; if the parent exits, the
    bridge reader reports the disconnect. This monitor intentionally performs no
    PID probing so Windows fallback cannot signal or terminate the parent.
    """

    parent_pid: int

    async def wait_for_parent_exit(self, stop_event: asyncio.Event) -> None:
        _ = self.parent_pid
        await stop_event.wait()


@dataclass(slots=True)
class WindowsParentHandleMonitor:
    handle: object
    poll_interval_s: float = 0.25
    wait_handle_signaled: Callable[[object], bool] | None = None
    close_handle: Callable[[object], None] | None = None
    _closed: bool = field(init=False, default=False)

    async def wait_for_parent_exit(self, stop_event: asyncio.Event) -> None:
        wait_handle_signaled = self.wait_handle_signaled or _default_windows_handle_signaled
        try:
            while not stop_event.is_set():
                if await asyncio.to_thread(wait_handle_signaled, self.handle):
                    return
                try:
                    await asyncio.wait_for(stop_event.wait(), timeout=self.poll_interval_s)
                except TimeoutError:
                    continue
        finally:
            await asyncio.to_thread(self.close)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        close_handle = self.close_handle or _default_close_windows_handle
        close_handle(self.handle)


def _default_open_windows_parent_handle(parent_pid: int) -> object | None:
    if os.name != "nt" or parent_pid <= 0:
        return None
    try:
        import ctypes

        synchronize = 0x00100000
        handle = ctypes.windll.kernel32.OpenProcess(synchronize, False, int(parent_pid))
    except Exception:
        return None
    if not handle:
        return None
    return int(handle)


def _default_windows_handle_signaled(handle: object) -> bool:
    if os.name != "nt":
        return False
    try:
        import ctypes

        wait_object_0 = 0x00000000
        result = ctypes.windll.kernel32.WaitForSingleObject(int(handle), 0)
    except Exception:
        return False
    return result == wait_object_0


def _default_close_windows_handle(handle: object) -> None:
    if os.name != "nt":
        return
    with contextlib.suppress(Exception):
        import ctypes

        ctypes.windll.kernel32.CloseHandle(int(handle))


def create_parent_monitor(
    parent_pid: int,
    *,
    is_windows: bool | None = None,
    open_windows_handle: Callable[[int], object | None] | None = None,
) -> ParentMonitor:
    windows = os.name == "nt" if is_windows is None else is_windows
    if windows:
        opener = open_windows_handle or _default_open_windows_parent_handle
        handle = opener(parent_pid)
        if handle is not None:
            return WindowsParentHandleMonitor(handle=handle)
        logger.warning(
            "[DesktopOverlay] Unable to open parent process handle; "
            "relying on bridge disconnect for parent-loss detection"
        )
        return BridgeDisconnectParentMonitor(parent_pid=parent_pid)
    return PollingParentMonitor(parent_pid=parent_pid)


def validate_desktop_bridge_url(bridge_url: str) -> str:
    try:
        parsed = urlsplit(bridge_url)
        port = parsed.port
    except ValueError as exc:
        raise ValueError("desktop overlay bridge_url is invalid") from exc

    if parsed.scheme != "ws":
        raise ValueError("desktop overlay bridge_url must use ws")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("desktop overlay bridge_url must not include credentials")
    if parsed.hostname not in _LOOPBACK_BRIDGE_HOSTS:
        raise ValueError("desktop overlay bridge_url must be loopback-only")
    if port is None or port <= 0:
        raise ValueError("desktop overlay bridge_url must include a positive port")
    return bridge_url


def load_renderer_manifest(config_path: Path) -> OverlayLaunchManifest:
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise DesktopOverlayStartupError(
            "manifest_invalid",
            "desktop overlay launch manifest is invalid",
        ) from exc
    if not isinstance(payload, dict):
        raise DesktopOverlayStartupError(
            "manifest_invalid",
            "desktop overlay launch manifest is invalid",
        )

    try:
        _validate_manifest_payload_shape(payload)
        manifest = OverlayLaunchManifest.from_dict(payload)
        _validate_runtime_manifest(manifest)
    except DesktopOverlayStartupError:
        raise
    except Exception as exc:
        raise DesktopOverlayStartupError(
            "manifest_invalid",
            "desktop overlay launch manifest is invalid",
        ) from exc
    return manifest


class DesktopOverlayRenderer:
    def __init__(
        self,
        manifest: OverlayLaunchManifest,
        *,
        window: RendererWindow | None = None,
        lifecycle_sink: LifecycleSink | None = None,
        parent_monitor: ParentMonitor | None = None,
        diagnostic_port: RendererDiagnosticPort | None = None,
        diagnostic_ingress_gate: DiagnosticIngressGate | None = None,
    ) -> None:
        self.manifest = manifest
        self.lifecycle_sink = lifecycle_sink or StdoutLifecycleSink()
        self.window = window or FletDesktopRendererWindow(
            event_sink=self._emit_lifecycle,
            locale=manifest.locale,
            logging_mode=manifest.logging_mode,
        )
        self.parent_monitor = parent_monitor or create_parent_monitor(manifest.parent_pid)
        self.diagnostic_port = diagnostic_port or DetailedRendererDiagnosticPort(
            logging_mode=manifest.logging_mode
        )
        self._diagnostic_ingress_gate = diagnostic_ingress_gate
        self._shutdown_event = asyncio.Event()
        self._shutdown_lock = asyncio.Lock()
        self._shutdown_complete = False
        self._websocket: Any | None = None
        self._tasks: set[asyncio.Task[_RuntimeOutcome | None]] = set()
        self._ui_queue: asyncio.Queue[tuple[str, object]] = asyncio.Queue()
        self._last_accepted_snapshot_revision = -1

    @property
    def is_shutdown(self) -> bool:
        return self._shutdown_complete

    async def run(self) -> int:
        unexpected_startup_failure_reason = "renderer_init_failed"
        try:
            _validate_runtime_manifest(self.manifest)
            unexpected_startup_failure_reason = "bridge_auth_failed"
            websocket = await self._connect_bridge()
            self._websocket = websocket
            await websocket.send(
                json.dumps({"type": "auth", "session_token": self.manifest.session_token})
            )
            unexpected_startup_failure_reason = "renderer_init_failed"
            initial_snapshot, initial_runtime_controls = (
                await self._receive_initial_snapshot_and_runtime_controls(websocket)
            )
            unexpected_startup_failure_reason = "window_configuration_failed"
            prime_startup_runtime_controls = getattr(
                self.window,
                "prime_startup_runtime_controls",
                None,
            )
            startup_runtime_controls_to_dispatch = initial_runtime_controls
            if callable(prime_startup_runtime_controls):
                startup_runtime_controls_to_dispatch = prime_startup_runtime_controls(
                    initial_runtime_controls
                )
            canonical_bounds_present = any(
                payload.get("command") == "apply_window_bounds"
                and _parse_runtime_window_bounds(payload) is not None
                for payload in initial_runtime_controls
            )
            if isinstance(self.window, FletDesktopRendererWindow) and not canonical_bounds_present:
                raise DesktopOverlayStartupError(
                    "window_configuration_failed",
                    "desktop overlay startup requires canonical window bounds",
                )
            await self.window.start(initial_snapshot)
            self._last_accepted_snapshot_revision = initial_snapshot.revision
            for payload in startup_runtime_controls_to_dispatch:
                await self.window.dispatch_runtime_control(payload)
            unexpected_startup_failure_reason = "renderer_init_failed"
            self._start_runtime_tasks(websocket)
            ready_event: dict[str, object] = {"type": "overlay_ready"}
            ready_event["overlay_instance_id"] = self.manifest.overlay_instance_id
            startup_generation = getattr(self.window, "startup_generation", 0)
            if isinstance(startup_generation, int) and startup_generation > 0:
                ready_event["generation"] = startup_generation
            if isinstance(startup_generation, int) and startup_generation > 0:
                _emit_desktop_lifecycle_trace(
                    "desktop_renderer",
                    "overlay_ready_emitted",
                    {
                        "generation": startup_generation,
                        "overlay_instance_id": self.manifest.overlay_instance_id,
                    },
                )
            await self._emit_lifecycle(ready_event)
            outcome = await self._wait_for_runtime_outcome()
            return outcome.exit_code
        except DesktopOverlayStartupError as exc:
            await self._emit_lifecycle(
                {"type": "startup_error", "failure_reason": exc.failure_reason}
            )
            return _STARTUP_FAILURE_EXIT_CODE
        except Exception as exc:
            safe_exception_message = _redact_renderer_startup_exception_text(
                str(exc),
                self.manifest,
            )
            safe_exception_traceback = _redact_renderer_startup_exception_text(
                "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
                self.manifest,
            )
            logger.warning(
                "[DesktopOverlay] Renderer startup failed: "
                "exception_type=%s exception_message=%s exception_traceback=%s",
                type(exc).__name__,
                safe_exception_message,
                safe_exception_traceback,
            )
            failure_event: dict[str, object] = {
                "type": "startup_error",
                "failure_reason": unexpected_startup_failure_reason,
            }
            startup_phase = getattr(self.window, "startup_phase", None)
            if isinstance(startup_phase, str):
                failure_event["startup_phase"] = startup_phase
            await self._emit_lifecycle(failure_event)
            return _STARTUP_FAILURE_EXIT_CODE
        finally:
            await self.shutdown()

    async def shutdown(self) -> None:
        async with self._shutdown_lock:
            if self._shutdown_complete:
                return
            self._shutdown_event.set()

            window_closed = False
            try:
                await self.window.close()
            except Exception:
                pass
            else:
                window_closed = True
            if window_closed:
                await self._emit_lifecycle(
                    {
                        "type": "shutdown_complete",
                        "overlay_instance_id": self.manifest.overlay_instance_id,
                    }
                )

            websocket = self._websocket
            self._websocket = None
            if websocket is not None:
                with contextlib.suppress(Exception):
                    await websocket.close()

            current_task = asyncio.current_task()
            pending_tasks = [
                task for task in self._tasks if task is not current_task and not task.done()
            ]
            for task in pending_tasks:
                task.cancel()
            if pending_tasks:
                await asyncio.gather(*pending_tasks, return_exceptions=True)
            if self._tasks:
                await asyncio.gather(*self._tasks, return_exceptions=True)
            self._tasks.clear()
            with contextlib.suppress(Exception):
                await self.diagnostic_port.close()
            await _close_parent_monitor(self.parent_monitor)
            self._shutdown_complete = True

    async def _connect_bridge(self) -> Any:
        timeout_s = max(0.1, self.manifest.startup_deadline_ms / 1000.0)
        try:
            return await asyncio.wait_for(
                websockets.connect(self.manifest.bridge_url, ping_interval=None),
                timeout=timeout_s,
            )
        except Exception as exc:
            raise DesktopOverlayStartupError(
                "bridge_auth_failed",
                "desktop overlay bridge authentication failed",
            ) from exc

    async def _receive_initial_snapshot_and_runtime_controls(
        self,
        websocket: Any,
    ) -> tuple[OverlayPresentationSnapshot, tuple[dict[str, object], ...]]:
        timeout_s = max(0.1, self.manifest.startup_deadline_ms / 1000.0)
        try:
            raw_message = await asyncio.wait_for(websocket.recv(), timeout=timeout_s)
            message = _load_bridge_message(raw_message)
        except DesktopOverlayStartupError:
            raise
        except Exception as exc:
            raise DesktopOverlayStartupError(
                "renderer_init_failed",
                "desktop overlay initial snapshot is invalid",
            ) from exc

        message_type = message.get("type")
        if message_type == "auth_error":
            raise DesktopOverlayStartupError(
                "bridge_auth_failed",
                "desktop overlay bridge authentication failed",
            )
        if message_type != "snapshot":
            raise DesktopOverlayStartupError(
                "renderer_init_failed",
                "desktop overlay initial snapshot is invalid",
            )
        try:
            snapshot = _parse_snapshot_message(message)
        except Exception as exc:
            raise DesktopOverlayStartupError(
                "renderer_init_failed",
                "desktop overlay initial snapshot is invalid",
            ) from exc
        raw_controls = message.get("startup_runtime_controls")
        if not isinstance(raw_controls, list):
            raise DesktopOverlayStartupError(
                "runtime_control_invalid",
                "desktop overlay initial runtime controls are not framed with the snapshot",
            )
        controls: list[dict[str, object]] = []
        for raw_control in raw_controls:
            payload = _parse_runtime_control_payload(
                {"type": "runtime_control", "payload": raw_control}
            )
            if payload is None:
                raise DesktopOverlayStartupError(
                    "runtime_control_invalid",
                    "desktop overlay initial runtime control is invalid",
                )
            controls.append(payload)
        return snapshot, tuple(controls)

    def _start_runtime_tasks(self, websocket: Any) -> None:
        self._tasks = {
            asyncio.create_task(self._bridge_reader_loop(websocket)),
            asyncio.create_task(self._parent_monitor_loop()),
            asyncio.create_task(self._window_loop()),
            asyncio.create_task(self._ui_update_loop()),
            asyncio.create_task(self._heartbeat_loop()),
        }

    async def _wait_for_runtime_outcome(self) -> _RuntimeOutcome:
        while self._tasks:
            done, _pending = await asyncio.wait(
                self._tasks,
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in done:
                self._tasks.discard(task)
                try:
                    result = task.result()
                except asyncio.CancelledError:
                    continue
                except Exception as exc:
                    logger.warning(
                        "[DesktopOverlay] Runtime task failed: exception_type=%s",
                        type(exc).__name__,
                    )
                    await self._emit_runtime_error("runtime_crashed")
                    return _RuntimeOutcome(_RUNTIME_FAILURE_EXIT_CODE)
                if isinstance(result, _RuntimeOutcome):
                    return result
            if self._shutdown_event.is_set():
                return _RuntimeOutcome(_SUCCESS_EXIT_CODE)
        return _RuntimeOutcome(_SUCCESS_EXIT_CODE)

    async def _bridge_reader_loop(self, websocket: Any) -> _RuntimeOutcome:
        try:
            async for raw_message in websocket:
                outcome = await self._handle_bridge_message(raw_message)
                if outcome is not None:
                    return outcome
        except ConnectionClosed:
            if self._shutdown_event.is_set():
                return _RuntimeOutcome(_SUCCESS_EXIT_CODE)
            await self._emit_runtime_error("runtime_disconnected")
            return _RuntimeOutcome(_RUNTIME_FAILURE_EXIT_CODE)
        if self._shutdown_event.is_set():
            return _RuntimeOutcome(_SUCCESS_EXIT_CODE)
        await self._emit_runtime_error("runtime_disconnected")
        return _RuntimeOutcome(_RUNTIME_FAILURE_EXIT_CODE)

    async def _handle_bridge_message(self, raw_message: object) -> _RuntimeOutcome | None:
        try:
            message = _load_bridge_message(raw_message)
        except ValueError:
            logger.warning("[DesktopOverlay] Ignoring malformed bridge message")
            return None

        message_type = message.get("type")
        if message_type == "heartbeat":
            return None
        if message_type == "shutdown":
            return _RuntimeOutcome(_SUCCESS_EXIT_CODE)
        if message_type == "snapshot":
            try:
                snapshot = _parse_snapshot_message(message)
            except Exception:
                logger.warning("[DesktopOverlay] Ignoring malformed snapshot update")
                return None
            await self.enqueue_snapshot(snapshot)
            return None
        if message_type == "runtime_control":
            payload = _parse_runtime_control_payload(message)
            if payload is None:
                await self._emit_runtime_error("runtime_control_invalid")
                return _RuntimeOutcome(_RUNTIME_FAILURE_EXIT_CODE)
            await self.enqueue_runtime_control(payload)
            return None
        logger.warning(
            "[DesktopOverlay] Ignoring unsupported bridge message type: %r", message_type
        )
        return None

    async def enqueue_snapshot(self, snapshot: OverlayPresentationSnapshot) -> None:
        await self._ui_queue.put(("snapshot", snapshot))
        if self._diagnostic_ingress_gate is not None:
            await self._diagnostic_ingress_gate.snapshot_enqueued(snapshot.revision)

    async def enqueue_runtime_control(self, payload: dict[str, object]) -> None:
        await self._ui_queue.put(("runtime_control", dict(payload)))

    async def _ui_update_loop(self) -> _RuntimeOutcome | None:
        while not self._shutdown_event.is_set():
            queue_task = asyncio.create_task(self._ui_queue.get())
            stop_task = asyncio.create_task(self._shutdown_event.wait())
            try:
                done, _pending = await asyncio.wait(
                    {queue_task, stop_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if stop_task in done:
                    return None
                kind, payload = queue_task.result()
            finally:
                for task in (queue_task, stop_task):
                    if not task.done():
                        task.cancel()
                await asyncio.gather(queue_task, stop_task, return_exceptions=True)

            try:
                if kind == "snapshot" and isinstance(payload, OverlayPresentationSnapshot):
                    if self._diagnostic_ingress_gate is not None:
                        await self._diagnostic_ingress_gate.wait_for_release(payload.revision)
                    barrier = await self._dispatch_pending_snapshot_batch(payload)
                    if barrier is not None:
                        barrier_kind, barrier_payload = barrier
                        if barrier_kind == "runtime_control" and isinstance(barrier_payload, dict):
                            await self.window.dispatch_runtime_control(barrier_payload)
                elif kind == "runtime_control" and isinstance(payload, dict):
                    await self.window.dispatch_runtime_control(payload)
            except Exception:
                await self._emit_runtime_error("window_configuration_failed")
                return _RuntimeOutcome(_RUNTIME_FAILURE_EXIT_CODE)
        return None

    async def _dispatch_pending_snapshot_batch(
        self,
        first_snapshot: OverlayPresentationSnapshot,
    ) -> tuple[str, object] | None:
        snapshots = [first_snapshot]
        barrier: tuple[str, object] | None = None
        while True:
            try:
                kind, payload = self._ui_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            if kind == "snapshot" and isinstance(payload, OverlayPresentationSnapshot):
                snapshots.append(payload)
                continue
            barrier = (kind, payload)
            break

        accepted: list[OverlayPresentationSnapshot] = []
        dispositions: list[tuple[OverlayPresentationSnapshot, str]] = []
        for snapshot in snapshots:
            if snapshot.revision <= self._last_accepted_snapshot_revision:
                dispositions.append((snapshot, "stale"))
                continue
            self._last_accepted_snapshot_revision = snapshot.revision
            accepted.append(snapshot)
            dispositions.append((snapshot, "accepted"))

        committed_snapshot = accepted[-1] if accepted else None
        for snapshot, disposition in dispositions:
            outcome = (
                "committed"
                if snapshot is committed_snapshot
                else "superseded" if disposition == "accepted" else disposition
            )
            await self._emit_renderer_diagnostic(
                event_type="receipt",
                snapshot=snapshot,
                disposition=outcome,
            )
            if disposition == "stale":
                await self._emit_renderer_diagnostic(
                    event_type="stale",
                    snapshot=snapshot,
                    disposition="stale",
                )
                continue
            if snapshot is not committed_snapshot:
                await self._advance_snapshot_history(snapshot)
                await self._emit_renderer_diagnostic(
                    event_type="supersession",
                    snapshot=snapshot,
                    disposition="superseded",
                )

        if committed_snapshot is not None:
            await self._emit_renderer_diagnostic(
                event_type="render_start",
                snapshot=committed_snapshot,
                disposition="committed",
                visual_state=self._renderer_visual_state_for_snapshot(committed_snapshot),
            )
            try:
                await self.window.dispatch_snapshot(committed_snapshot)
            except Exception:
                await self._emit_renderer_diagnostic(
                    event_type="failed",
                    snapshot=committed_snapshot,
                    disposition="failed",
                )
                raise
            acknowledgement = RendererCommitAcknowledgement(committed_snapshot.revision)
            await self._emit_renderer_diagnostic(
                event_type="render_commit",
                snapshot=committed_snapshot,
                disposition="committed",
                visual_state=self._renderer_visual_state(),
                acknowledgement=acknowledgement,
            )
            if self.diagnostic_port.requires_commit_acknowledgement:
                try:
                    await self.diagnostic_port.wait_for_commit_ack(acknowledgement)
                except Exception:
                    await self._emit_renderer_diagnostic(
                        event_type="failed",
                        snapshot=committed_snapshot,
                        disposition="failed",
                    )
                    raise
                await self._emit_renderer_diagnostic(
                    event_type="render_commit_acknowledgement",
                    snapshot=committed_snapshot,
                    disposition="committed",
                    visual_state=self._renderer_visual_state(),
                    render_commit_acknowledged=True,
                )
        return barrier

    async def _advance_snapshot_history(self, snapshot: OverlayPresentationSnapshot) -> None:
        await self.window.advance_snapshot_history(snapshot)

    def _renderer_visual_state(self) -> dict[str, object]:
        return dict(self.window.renderer_visual_state())

    def _renderer_visual_state_for_snapshot(
        self,
        snapshot: OverlayPresentationSnapshot,
    ) -> dict[str, object]:
        return dict(self.window.renderer_visual_state_for_snapshot(snapshot))

    async def _emit_renderer_diagnostic(
        self,
        *,
        event_type: str,
        snapshot: OverlayPresentationSnapshot,
        disposition: str,
        visual_state: Mapping[str, object] | None = None,
        acknowledgement: RendererCommitAcknowledgement | None = None,
        render_commit_acknowledged: bool = False,
    ) -> None:
        safe_visual_state = dict(visual_state or self._renderer_visual_state_for_snapshot(snapshot))
        record = validate_desktop_renderer_event(
            {
                "schema_version": DESKTOP_RENDERER_EVENT_SCHEMA_VERSION,
                "record_type": "renderer_event",
                "event_type": event_type,
                "renderer_revision": snapshot.revision,
                "actual_disposition": disposition,
                "render_commit_acknowledged": render_commit_acknowledged,
                **safe_visual_state,
            }
        )
        if record is None:
            return
        await self.diagnostic_port.emit(
            RendererDiagnosticEnvelope(record=record, acknowledgement=acknowledgement)
        )

    async def _parent_monitor_loop(self) -> _RuntimeOutcome | None:
        try:
            await self.parent_monitor.wait_for_parent_exit(self._shutdown_event)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "[DesktopOverlay] Parent monitor failed: exception_type=%s",
                type(exc).__name__,
            )
            return None
        if self._shutdown_event.is_set():
            return None
        await self._emit_runtime_error("runtime_disconnected")
        return _RuntimeOutcome(_RUNTIME_FAILURE_EXIT_CODE)

    async def _window_loop(self) -> _RuntimeOutcome | None:
        try:
            await self.window.run_until_closed()
        except asyncio.CancelledError:
            raise
        except Exception:
            await self._emit_runtime_error("window_configuration_failed")
            return _RuntimeOutcome(_RUNTIME_FAILURE_EXIT_CODE)
        if self._shutdown_event.is_set():
            return None
        return _RuntimeOutcome(_SUCCESS_EXIT_CODE)

    async def _heartbeat_loop(self) -> None:
        while not self._shutdown_event.is_set():
            try:
                await asyncio.wait_for(self._shutdown_event.wait(), timeout=1.0)
            except TimeoutError:
                continue

    async def _emit_runtime_error(self, failure_reason: str) -> None:
        await self._emit_lifecycle({"type": "runtime_error", "failure_reason": failure_reason})

    async def _emit_lifecycle(self, event: dict[str, object]) -> None:
        safe_event = _redact_event(event)
        websocket = self._websocket
        if websocket is not None:
            with contextlib.suppress(Exception):
                await websocket.send(json.dumps(safe_event))
        await self.lifecycle_sink.emit(safe_event)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="puripuly-heart desktop-overlay")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--config",
        type=Path,
        help="Path to overlay launch manifest JSON",
    )
    mode.add_argument(
        "--preview",
        action="store_true",
        help="Run a local desktop overlay preview",
    )
    return parser


def run_renderer(config_path: Path) -> int:
    return asyncio.run(_run_renderer_async(config_path))


async def _run_renderer_async(config_path: Path) -> int:
    sink = StdoutLifecycleSink()
    try:
        manifest = load_renderer_manifest(config_path)
    except DesktopOverlayStartupError as exc:
        await sink.emit({"type": "startup_error", "failure_reason": exc.failure_reason})
        return _STARTUP_FAILURE_EXIT_CODE
    renderer = DesktopOverlayRenderer(manifest, lifecycle_sink=sink)
    return await renderer.run()


def run_preview(
    *,
    app_runner: PreviewAppRunner | None = None,
    locale: str | None = None,
) -> int:
    catalog = build_desktop_overlay_preview_catalog(locale=locale)
    secret_findings = preview_fixture_secret_findings(catalog)
    if secret_findings:
        for finding in secret_findings:
            logger.error("Unsafe desktop overlay preview fixture data: %s", finding)
        return _STARTUP_FAILURE_EXIT_CODE
    runner = app_runner or _default_preview_app_runner
    return asyncio.run(
        _run_preview_async(
            catalog=catalog,
            app_runner=runner,
            locale=locale,
            allow_no_page=app_runner is not None or runner is not _REAL_DEFAULT_PREVIEW_APP_RUNNER,
        )
    )


async def _run_preview_async(
    *,
    catalog: DesktopOverlayPreviewCatalog,
    app_runner: PreviewAppRunner,
    locale: str | None,
    allow_no_page: bool,
) -> int:
    async def preview_app_runner(target: Callable[[Any], object]) -> None:
        result = app_runner(target)
        if inspect.isawaitable(result):
            await result

    async def preview_event_sink(event: dict[str, object]) -> None:
        logger.debug("Desktop overlay preview event: %r", _redact_event(event))

    window = FletDesktopRendererWindow(
        app_runner=preview_app_runner,
        event_sink=preview_event_sink,
        locale=locale,
        preview_catalog=catalog,
    )
    try:
        try:
            await window.start(catalog.fixtures[0].snapshot)
        except RuntimeError:
            if allow_no_page and window._page is None:
                return _SUCCESS_EXIT_CODE
            raise
        await window.run_until_closed()
    finally:
        await window.close()
    return _SUCCESS_EXIT_CODE


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.preview:
        return run_preview()
    return run_renderer(args.config)


def _validate_runtime_manifest(manifest: OverlayLaunchManifest) -> None:
    if manifest.contract_version != OVERLAY_CONTRACT_VERSION:
        raise DesktopOverlayStartupError(
            "contract_mismatch",
            "desktop overlay contract version is not supported",
        )
    try:
        validate_desktop_bridge_url(manifest.bridge_url)
    except ValueError as exc:
        raise DesktopOverlayStartupError(
            "manifest_invalid",
            "desktop overlay launch manifest is invalid",
        ) from exc
    if not manifest.session_token:
        raise DesktopOverlayStartupError(
            "manifest_invalid",
            "desktop overlay launch manifest is invalid",
        )
    if manifest.parent_pid <= 0 or manifest.startup_deadline_ms <= 0:
        raise DesktopOverlayStartupError(
            "manifest_invalid",
            "desktop overlay launch manifest is invalid",
        )
    if not manifest.log_dir or not manifest.log_level or not manifest.locale:
        raise DesktopOverlayStartupError(
            "manifest_invalid",
            "desktop overlay launch manifest is invalid",
        )


def _validate_manifest_payload_shape(payload: dict[object, object]) -> None:
    for field_name in _REQUIRED_MANIFEST_STRING_FIELDS:
        value = payload.get(field_name)
        if not isinstance(value, str) or not value:
            raise DesktopOverlayStartupError(
                "manifest_invalid",
                "desktop overlay launch manifest is invalid",
            )
    for field_name in _REQUIRED_MANIFEST_INT_FIELDS:
        value = payload.get(field_name)
        if not isinstance(value, int) or isinstance(value, bool):
            raise DesktopOverlayStartupError(
                "manifest_invalid",
                "desktop overlay launch manifest is invalid",
            )


def _load_bridge_message(raw_message: object) -> dict[str, object]:
    if not isinstance(raw_message, str):
        raise ValueError("desktop overlay bridge message must be text JSON")
    payload = json.loads(raw_message)
    if not isinstance(payload, dict):
        raise ValueError("desktop overlay bridge message must decode to an object")
    return payload


def _parse_snapshot_message(message: dict[str, object]) -> OverlayPresentationSnapshot:
    payload = message.get("payload")
    if not isinstance(payload, dict):
        raise ValueError("desktop overlay snapshot payload must be an object")
    return OverlayPresentationSnapshot.from_dict(payload)


def _parse_runtime_control_payload(message: dict[str, object]) -> dict[str, object] | None:
    payload = message.get("payload")
    if not isinstance(payload, dict):
        return None
    if "logging_mode" in payload:
        if set(payload) != {"logging_mode"} or not isinstance(payload.get("logging_mode"), str):
            return None
        return dict(payload)
    command = payload.get("command")
    if not isinstance(command, str) or not command:
        return None
    return dict(payload)


def _redact_renderer_startup_exception_text(
    text: str,
    manifest: OverlayLaunchManifest,
) -> str:
    redacted = text
    if manifest.session_token:
        redacted = redacted.replace(manifest.session_token, "<redacted>")
    for _, pattern in _DESKTOP_PREVIEW_SECRET_PATTERNS:
        redacted = pattern.sub("<redacted>", redacted)
    return redacted


def _redact_event(event: dict[str, object]) -> dict[str, object]:
    redacted = _redact_value(event)
    if isinstance(redacted, dict):
        return redacted
    return {"type": "runtime_error", "failure_reason": "unknown"}


def _redact_value(value: object) -> object:
    if isinstance(value, dict):
        result: dict[str, object] = {}
        for key, item in value.items():
            key_text = str(key)
            if _is_sensitive_event_key(key_text):
                result[key_text] = "<redacted>"
            else:
                result[key_text] = _redact_value(item)
        return result
    if isinstance(value, list):
        return [_redact_value(item) for item in value]
    return value


def _is_sensitive_event_key(key: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]", "", key.lower())
    return normalized in _SENSITIVE_EVENT_KEYS


async def _close_parent_monitor(parent_monitor: ParentMonitor) -> None:
    close = getattr(parent_monitor, "close", None)
    if not callable(close):
        return
    result = close()
    if asyncio.iscoroutine(result):
        await result


if __name__ == "__main__":
    raise SystemExit(main())
