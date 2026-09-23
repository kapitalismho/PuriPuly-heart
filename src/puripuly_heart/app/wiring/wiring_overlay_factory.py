from __future__ import annotations

from puripuly_heart.config.desktop_overlay_values import DESKTOP_FLET_DEFAULT_TEXT_SCALE
from puripuly_heart.config.resolved import ResolvedOverlayConfig
from puripuly_heart.config.runtime_resolution import OverlayRuntimeIntent
from puripuly_heart.config.runtime_resolution import (
    resolve_overlay_config as resolve_overlay_runtime_config,
)
from puripuly_heart.config.settings_vnext.schema import AppSettingsVNext, DesktopFletOverlayIntent


def _desktop_overlay_options_from_vnext(
    desktop: DesktopFletOverlayIntent,
    *,
    locked: bool,
) -> dict[str, object]:
    visual = desktop.visual
    return {
        "size_preset": desktop.size_preset,
        "position": {
            "x": desktop.position.x,
            "y": desktop.position.y,
        },
        "locked": locked,
        "swap_caption_languages": desktop.swap_caption_languages,
        "visual": {
            "text_scale": DESKTOP_FLET_DEFAULT_TEXT_SCALE,
            "background_alpha": visual.background_alpha,
            "outline_width": None,
        },
    }


def overlay_runtime_intent_from_vnext(
    settings: AppSettingsVNext,
    *,
    enabled: bool,
    locked: bool = False,
) -> OverlayRuntimeIntent:
    overlay = settings.intent.overlay
    return OverlayRuntimeIntent(
        enabled=enabled,
        target=overlay.target,
        show_translation=overlay.show_translation,
        show_peer_original=overlay.show_peer_original,
        calibration=overlay.calibration.to_dict(),
        desktop_overlay_options=_desktop_overlay_options_from_vnext(
            overlay.desktop_flet,
            locked=locked,
        ),
    )


def resolve_overlay_config(intent: OverlayRuntimeIntent) -> ResolvedOverlayConfig:
    return resolve_overlay_runtime_config(intent)


def resolve_overlay_config_from_vnext(
    settings: AppSettingsVNext,
    *,
    enabled: bool,
    locked: bool = False,
) -> ResolvedOverlayConfig:
    return resolve_overlay_config(
        overlay_runtime_intent_from_vnext(
            settings,
            enabled=enabled,
            locked=locked,
        )
    )
