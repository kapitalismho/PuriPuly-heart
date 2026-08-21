from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import flet as ft

from puripuly_heart.app.language_selection import LanguageSelectionChange
from puripuly_heart.app.ports.ui_models import ManagedGemmaNoticeAction
from puripuly_heart.ui.gpu_notice import GpuNoticeAction


@dataclass(frozen=True, slots=True)
class DashboardTranslationIntents:
    submit_message: Callable[[str, str], None]
    toggle_translation: Callable[[bool], None]
    change_language: Callable[[LanguageSelectionChange], None]
    report_input_activity: Callable[[bool], None]


@dataclass(frozen=True, slots=True)
class DashboardCaptureIntents:
    toggle_self_capture: Callable[[bool], None]
    toggle_peer_capture: Callable[[bool], None]
    toggle_overlay: Callable[[bool], None]
    retry_peer_process_capture: Callable[[], object]
    run_gpu_notice_action: Callable[[GpuNoticeAction], object]
    run_managed_gemma_notice_action: Callable[[ManagedGemmaNoticeAction], object] | None = None


class DashboardIntentConsumer(Protocol):
    def bind_dashboard_intents(
        self,
        *,
        translation: DashboardTranslationIntents,
        capture: DashboardCaptureIntents,
    ) -> None: ...


class DashboardCaptureSlotProvider(Protocol):
    def self_capture_control(self) -> ft.Control: ...

    def peer_capture_control(self) -> ft.Control: ...

    def overlay_control(self) -> ft.Control: ...


@dataclass(frozen=True, slots=True)
class DashboardSurfaceSlots:
    self_capture: ft.Control
    peer_capture: ft.Control
    translation: ft.Control
    overlay: ft.Control
    display: ft.Control
    language: ft.Control

    @classmethod
    def from_capture_provider(
        cls,
        provider: DashboardCaptureSlotProvider,
        *,
        translation: ft.Control,
        display: ft.Control,
        language: ft.Control,
    ) -> DashboardSurfaceSlots:
        return cls(
            self_capture=provider.self_capture_control(),
            peer_capture=provider.peer_capture_control(),
            translation=translation,
            overlay=provider.overlay_control(),
            display=display,
            language=language,
        )


@dataclass(frozen=True, slots=True)
class DashboardSurfaceRegions:
    root: ft.Control
    shell_content: ft.Column
    main_surface: ft.Row
    control_region: ft.Container
    info_region: ft.Container
    control_grid: ft.Column
    top_controls: ft.Row
    bottom_controls: ft.Row
    info_stack: ft.Column
    display_card_slot: ft.Container
    language_card_slot: ft.Container


__all__ = [
    "DashboardCaptureIntents",
    "DashboardCaptureSlotProvider",
    "DashboardIntentConsumer",
    "DashboardSurfaceRegions",
    "DashboardSurfaceSlots",
    "DashboardTranslationIntents",
]
