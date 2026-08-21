from __future__ import annotations

import copy

from puripuly_heart.config.resolved import ResolvedOverlayConfig
from puripuly_heart.config.runtime_resolution import OverlayRuntimeIntent
from puripuly_heart.config.runtime_resolution import (
    resolve_overlay_config as resolve_overlay_runtime_config,
)
from puripuly_heart.config.settings import AppSettings


def _desktop_overlay_options_from_settings(settings: AppSettings) -> dict[str, object]:
    desktop_settings = copy.deepcopy(settings.overlay.desktop_flet)
    desktop_settings.validate()
    visual = desktop_settings.visual
    return {
        "size_preset": desktop_settings.size_preset,
        "position": {
            "x": desktop_settings.position.x,
            "y": desktop_settings.position.y,
        },
        "locked": desktop_settings.locked,
        "swap_caption_languages": desktop_settings.swap_caption_languages,
        "visual": {
            "text_scale": visual.text_scale,
            "background_alpha": visual.background_alpha,
            "outline_width": visual.outline_width,
        },
    }


def resolve_overlay_config(settings: AppSettings) -> ResolvedOverlayConfig:
    return resolve_overlay_runtime_config(
        OverlayRuntimeIntent(
            enabled=settings.ui.overlay_enabled,
            target=settings.overlay.target,
            show_translation=settings.overlay.show_translation,
            show_peer_original=settings.overlay.show_peer_original,
            calibration=settings.overlay.calibration.to_dict(),
            desktop_overlay_options=_desktop_overlay_options_from_settings(settings),
        )
    )
