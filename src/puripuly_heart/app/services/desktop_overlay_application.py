from __future__ import annotations

import asyncio
import copy
import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from puripuly_heart.app.ports.desktop_overlay import (
    DesktopBounds,
    DesktopOverlayPolicy,
    DesktopRuntimeControl,
    DesktopWorkAreaPort,
)
from puripuly_heart.app.services.canonical_settings_persistence import SettingsOwner
from puripuly_heart.app.services.overlay_application import OverlayApplicationOwner
from puripuly_heart.app.services.settings_application import SettingsApplicationOwner
from puripuly_heart.config.resolved import OVERLAY_TARGET_DESKTOP, ResolvedOverlayConfig
from puripuly_heart.core.runtime.desktop_overlay_bounds import (
    DesktopOverlayBoundsOwner,
    is_finite_non_bool_number,
)

DESKTOP_BOUNDS_PERSIST_DEBOUNCE_S = 0.05
DESKTOP_INTERACTION_MODE_EDIT = "edit"
DESKTOP_INTERACTION_MODE_PASS_THROUGH = "pass_through"
DESKTOP_INTERACTION_MODES = frozenset(
    {
        DESKTOP_INTERACTION_MODE_EDIT,
        DESKTOP_INTERACTION_MODE_PASS_THROUGH,
    }
)

SettingsApplicationProvider = Callable[[], SettingsApplicationOwner]
OverlayApplicationProvider = Callable[[], OverlayApplicationOwner]
DesktopPresentationSink = Callable[[str, bool], None]
DesktopDetailedLogSink = Callable[[str, int, Exception | None], object]


@dataclass(slots=True)
class DesktopOverlayApplicationOwner:
    settings: SettingsOwner
    settings_application_provider: SettingsApplicationProvider
    overlay_provider: OverlayApplicationProvider
    work_area: DesktopWorkAreaPort
    policy: DesktopOverlayPolicy
    presentation_sink: DesktopPresentationSink
    log_detailed: DesktopDetailedLogSink
    _interaction_mode: str = field(
        init=False,
        default=DESKTOP_INTERACTION_MODE_EDIT,
        repr=False,
    )
    _bounds: DesktopOverlayBoundsOwner = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._bounds = DesktopOverlayBoundsOwner(
            persist_bounds=self.persist_bounds,
            debounce_seconds=lambda: DESKTOP_BOUNDS_PERSIST_DEBOUNCE_S,
            minimum_width=self.policy.minimum_width,
            minimum_height=self.policy.minimum_height,
            diagnostics_sink=lambda event, metadata: self._log(
                f"[DesktopOverlay][Bounds] event={event} metadata={dict(metadata)}",
                logging.WARNING,
            ),
        )

    @property
    def interaction_mode(self) -> str:
        return self._interaction_mode

    @property
    def captions_locked(self) -> bool:
        return self._interaction_mode == DESKTOP_INTERACTION_MODE_PASS_THROUGH

    @property
    def bounds_owner(self) -> DesktopOverlayBoundsOwner:
        return self._bounds

    def set_interaction_mode(self, mode: object) -> bool:
        if not isinstance(mode, str) or mode not in DESKTOP_INTERACTION_MODES:
            return False
        previous = self._interaction_mode
        self._interaction_mode = mode
        if previous != mode:
            self.presentation_sink(mode, self.captions_locked)
        return True

    def initial_controls(
        self,
        config: ResolvedOverlayConfig,
    ) -> list[DesktopRuntimeControl]:
        desktop_options = config.desktop_overlay_options
        position = desktop_options.get("position")
        if not isinstance(position, Mapping):
            position = {}
        visual = desktop_options.get("visual")
        if not isinstance(visual, Mapping):
            visual = {}
        width, height = self.dimensions(desktop_options.get("size_preset"))
        x = position.get("x")
        y = position.get("y")
        bounds = (
            {"x": x, "y": y, "width": width, "height": height}
            if is_finite_non_bool_number(x) and is_finite_non_bool_number(y)
            else self.centered_bounds(width=width, height=height)
        )
        text_scale = visual.get("text_scale", self.policy.default_text_scale)
        background_alpha = visual.get(
            "background_alpha",
            self.policy.default_background_alpha,
        )
        outline_width = visual.get("outline_width")
        swap_caption_languages = bool(desktop_options.get("swap_caption_languages", False))
        self._log(
            "[DesktopOverlay][Launch] "
            f"target=desktop locked={bool(desktop_options.get('locked', False))} "
            f"interaction_mode={DESKTOP_INTERACTION_MODE_EDIT} "
            f"size_preset={desktop_options.get('size_preset')} "
            f"x={bounds['x']} y={bounds['y']} width={bounds['width']} "
            f"height={bounds['height']} text_scale={text_scale} "
            f"background_alpha={background_alpha} outline_width={outline_width} "
            f"swap_caption_languages={swap_caption_languages}"
        )
        return [
            {"command": "apply_window_bounds", **bounds},
            {
                "command": "apply_visual_config",
                "text_scale": text_scale,
                "background_alpha": background_alpha,
                "outline_width": outline_width,
                "swap_caption_languages": swap_caption_languages,
            },
            {
                "command": "set_interaction_mode",
                "mode": DESKTOP_INTERACTION_MODE_EDIT,
            },
        ]

    def dimensions(self, size_preset: object) -> tuple[int, int]:
        if isinstance(size_preset, str) and size_preset in self.policy.size_presets:
            return self.policy.size_presets[size_preset]
        return self.policy.size_presets[self.policy.default_size_preset]

    def launch_bounds(self, desktop_settings: object) -> DesktopBounds:
        position = getattr(desktop_settings, "position", None)
        x = getattr(position, "x", None)
        y = getattr(position, "y", None)
        width, height = self.dimensions(getattr(desktop_settings, "size_preset", None))
        if is_finite_non_bool_number(x) and is_finite_non_bool_number(y):
            return {"x": x, "y": y, "width": width, "height": height}
        return self.centered_bounds(width=width, height=height)

    def centered_bounds(
        self,
        *,
        width: int | float,
        height: int | float,
    ) -> DesktopBounds:
        work_area = self.work_area.primary_work_area()
        if work_area is None:
            return {"x": 0, "y": 0, "width": width, "height": height}
        left, top, work_width, work_height = work_area
        if not (
            is_finite_non_bool_number(left)
            and is_finite_non_bool_number(top)
            and is_finite_non_bool_number(work_width)
            and is_finite_non_bool_number(work_height)
            and work_width > 0
            and work_height > 0
        ):
            return {"x": 0, "y": 0, "width": width, "height": height}
        return {
            "x": left + ((work_width - width) / 2),
            "y": top + ((work_height - height) / 2),
            "width": width,
            "height": height,
        }

    def drain_pending_user_bounds_events(self) -> None:
        runtime = self.overlay_provider().runtime
        queue = runtime.renderer_events_or_none() if runtime is not None else None
        if queue is None:
            return
        retained: list[dict[str, object]] = []
        dropped = 0
        while True:
            try:
                event = queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            if self._is_user_bounds_event(event):
                dropped += 1
            else:
                retained.append(event)
        for event in retained:
            queue.put_nowait(event)
        if dropped:
            self._log(f"[DesktopOverlay][Bounds] drained_pending_user_bounds count={dropped}")

    async def set_captions_locked(self, locked: bool) -> None:
        overlay = self.overlay_provider()
        if self.settings.current is None or overlay.state != "connected":
            return
        if overlay.active_target != OVERLAY_TARGET_DESKTOP or overlay.current_bridge() is None:
            return
        mode = DESKTOP_INTERACTION_MODE_PASS_THROUGH if locked else DESKTOP_INTERACTION_MODE_EDIT
        if await self.broadcast({"command": "set_interaction_mode", "mode": mode}):
            self.set_interaction_mode(mode)

    async def set_size_preset(self, size_preset: str) -> None:
        current = self.settings.current
        if current is None:
            return
        normalized = (
            size_preset
            if size_preset in self.policy.size_presets
            else self.policy.default_size_preset
        )
        if current.overlay.desktop_flet.size_preset == normalized:
            return
        updated = copy.deepcopy(current)
        updated.overlay.desktop_flet.size_preset = normalized
        await self.settings_application_provider().apply(updated)

    async def reset_position(self) -> None:
        await self._reset()

    async def broadcast(self, payload: DesktopRuntimeControl) -> bool:
        overlay = self.overlay_provider()
        if overlay.active_target != OVERLAY_TARGET_DESKTOP:
            return False
        bridge = overlay.current_bridge()
        if bridge is None:
            return False
        broadcast = getattr(bridge, "broadcast_desktop_runtime_control", None)
        if not callable(broadcast):
            return False
        try:
            await broadcast(payload)
        except Exception as exc:
            self._log(
                "[Overlay] Failed to send desktop runtime control",
                logging.WARNING,
                exc,
            )
            return False
        return True

    async def broadcast_bounds(self, bounds: DesktopBounds) -> None:
        payload: DesktopRuntimeControl = {"command": "apply_window_bounds", **bounds}
        if await self.broadcast(payload):
            self._bounds.track_apply_control(payload)

    async def consume_renderer_events(
        self,
        queue: asyncio.Queue[dict[str, object]],
        overlay_instance_id: str,
    ) -> None:
        while True:
            try:
                event = await queue.get()
                await self.handle_renderer_event(
                    event,
                    overlay_instance_id=overlay_instance_id,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._log(
                    "[Overlay] Ignoring desktop renderer event after application error",
                    logging.WARNING,
                    exc,
                )

    async def handle_renderer_event(
        self,
        event: object,
        *,
        overlay_instance_id: str | None = None,
    ) -> None:
        overlay = self.overlay_provider()
        if overlay_instance_id is not None:
            runtime = overlay.runtime
            if runtime is None or not runtime.is_current_instance_id(overlay_instance_id):
                return
        if overlay.active_target != OVERLAY_TARGET_DESKTOP or not isinstance(event, dict):
            return
        payload = event.get("payload")
        if not isinstance(payload, dict):
            return
        event_type = payload.get("event")
        if event_type == "window_bounds_changed":
            await self.handle_bounds_changed(payload)
        elif event_type == "reset_to_bottom_center_requested":
            await self._reset()
        elif event_type == "interaction_mode_changed":
            self.set_interaction_mode(payload.get("mode"))

    async def handle_bounds_changed(self, payload: Mapping[object, object]) -> None:
        if not self._bounds.is_valid_event_payload(payload):
            self._log(
                "[DesktopOverlay][Bounds] ignored reason=invalid_payload "
                f"keys={sorted(str(key) for key in payload)} "
                f"source={payload.get('source')} persist={payload.get('persist')}"
            )
            return
        bounds = self._bounds.bounds_from_payload(payload)
        if bounds is None:
            self._log(
                "[DesktopOverlay][Bounds] ignored reason=invalid_bounds "
                f"source={payload.get('source')} persist={payload.get('persist')}"
            )
            return
        source = payload.get("source")
        self._log(
            "[DesktopOverlay][Bounds] received "
            f"source={source} persist={payload.get('persist')} "
            f"interaction_mode={self._interaction_mode} "
            f"x={bounds['x']} y={bounds['y']} width={bounds['width']} "
            f"height={bounds['height']}"
        )
        if source in {"programmatic", "launch_repair"}:
            self._log(
                "[DesktopOverlay][Bounds] ignored reason=programmatic_source "
                f"source={source} x={bounds['x']} y={bounds['y']} "
                f"width={bounds['width']} height={bounds['height']}"
            )
            self._bounds.discard_suppressed(bounds)
            return
        if source == "reset":
            self._log(
                "[DesktopOverlay][Bounds] reset_requested "
                f"x={bounds['x']} y={bounds['y']} width={bounds['width']} "
                f"height={bounds['height']}"
            )
            await self._reset()
            return
        if source == "user" and self._interaction_mode != DESKTOP_INTERACTION_MODE_EDIT:
            self._log(
                "[DesktopOverlay][Bounds] ignored reason=locked_interaction_mode "
                f"interaction_mode={self._interaction_mode} "
                f"x={bounds['x']} y={bounds['y']} width={bounds['width']} "
                f"height={bounds['height']}"
            )
            return
        if self._bounds.consume_suppressed(bounds):
            self._log(
                "[DesktopOverlay][Bounds] ignored reason=suppressed_signature "
                f"x={bounds['x']} y={bounds['y']} width={bounds['width']} "
                f"height={bounds['height']}"
            )
            return
        self._bounds.schedule_persistence(bounds)
        self._log(
            "[DesktopOverlay][Bounds] scheduled_persist "
            f"x={bounds['x']} y={bounds['y']} width={bounds['width']} "
            f"height={bounds['height']}"
        )

    async def persist_bounds(self, bounds: DesktopBounds) -> None:
        current = self.settings.current
        overlay = self.overlay_provider()
        if current is None or overlay.active_target != OVERLAY_TARGET_DESKTOP:
            return
        if self._bounds.bounds_from_payload(bounds) is None:
            return
        updated = copy.deepcopy(current)
        desktop = updated.overlay.desktop_flet
        desktop.position.x = bounds["x"]
        desktop.position.y = bounds["y"]
        desktop.position.validate()
        application = self.settings_application_provider()
        if not await application.apply_overlay_osc_output(updated):
            return
        if not application.results.committed():
            return
        self._log(
            "[DesktopOverlay][Bounds] persisted "
            f"x={bounds['x']} y={bounds['y']} width={bounds['width']} "
            f"height={bounds['height']} size_preset={desktop.size_preset}"
        )

    async def _reset(self) -> None:
        current = self.settings.current
        overlay = self.overlay_provider()
        if current is None:
            return
        configured_desktop = (
            OverlayApplicationOwner.normalized_target(current.overlay.target)
            == OVERLAY_TARGET_DESKTOP
        )
        renderer_active = bool(
            overlay.active_target == OVERLAY_TARGET_DESKTOP and overlay.current_bridge() is not None
        )
        if not configured_desktop and not renderer_active:
            return
        await self._bounds.cancel()
        self.drain_pending_user_bounds_events()
        updated = copy.deepcopy(current)
        desktop = updated.overlay.desktop_flet
        desktop.position.x = None
        desktop.position.y = None
        desktop.validate()
        application = self.settings_application_provider()
        routed = await application.apply_overlay_osc_output(updated)
        if routed and not application.results.committed():
            return
        if self.settings.current is not None:
            self.settings.current.overlay.desktop_flet.locked = False
        self.set_interaction_mode(DESKTOP_INTERACTION_MODE_EDIT)
        if not renderer_active:
            return
        await self.broadcast(
            {
                "command": "set_interaction_mode",
                "mode": DESKTOP_INTERACTION_MODE_EDIT,
            }
        )
        await self.broadcast_bounds(self.center_bounds_for_current_preset())

    def center_bounds_for_current_preset(self) -> DesktopBounds:
        current = self.settings.current
        if current is None:
            width, height = self.dimensions("medium")
        else:
            width, height = self.dimensions(current.overlay.desktop_flet.size_preset)
        return self.centered_bounds(width=width, height=height)

    def runtime_is_running_for_settings(self, settings: Any) -> bool:
        overlay = self.overlay_provider()
        return bool(
            settings.ui.overlay_enabled
            and overlay.active_target == OVERLAY_TARGET_DESKTOP
            and overlay.current_bridge() is not None
        )

    def center_preserving_bounds(
        self,
        *,
        previous_desktop_settings: object,
        next_size_preset: object,
    ) -> DesktopBounds:
        previous = self.launch_bounds(previous_desktop_settings)
        width, height = self.dimensions(next_size_preset)
        center_x = previous["x"] + (previous["width"] / 2)
        center_y = previous["y"] + (previous["height"] / 2)
        return {
            "x": center_x - (width / 2),
            "y": center_y - (height / 2),
            "width": width,
            "height": height,
        }

    def prepare_settings_update(
        self,
        previous_settings: Any | None,
        next_settings: Any,
    ) -> tuple[DesktopRuntimeControl, ...]:
        if previous_settings is None:
            return ()
        previous = copy.deepcopy(previous_settings.overlay.desktop_flet)
        previous.validate()
        next_desktop = next_settings.overlay.desktop_flet
        next_desktop.validate()
        if not self.runtime_is_running_for_settings(next_settings):
            return ()
        controls: list[DesktopRuntimeControl] = []
        if previous.size_preset != next_desktop.size_preset:
            self._bounds.discard()
            self.drain_pending_user_bounds_events()
            bounds = self.center_preserving_bounds(
                previous_desktop_settings=previous,
                next_size_preset=next_desktop.size_preset,
            )
            if previous.position.x is not None and previous.position.y is not None:
                next_desktop.position.x = bounds["x"]
                next_desktop.position.y = bounds["y"]
                next_desktop.position.validate()
            controls.append({"command": "apply_window_bounds", **bounds})
        if (
            previous.visual.text_scale != next_desktop.visual.text_scale
            or previous.visual.background_alpha != next_desktop.visual.background_alpha
            or previous.visual.outline_width != next_desktop.visual.outline_width
            or previous.swap_caption_languages != next_desktop.swap_caption_languages
        ):
            controls.append(
                {
                    "command": "apply_visual_config",
                    "text_scale": next_desktop.visual.text_scale,
                    "background_alpha": next_desktop.visual.background_alpha,
                    "outline_width": next_desktop.visual.outline_width,
                    "swap_caption_languages": next_desktop.swap_caption_languages,
                }
            )
        return tuple(controls)

    async def prepare_persistence(
        self,
        previous_settings: Any,
        next_settings: Any,
    ) -> None:
        previous = copy.deepcopy(previous_settings.overlay.desktop_flet)
        previous.validate()
        next_desktop = next_settings.overlay.desktop_flet
        next_desktop.validate()
        if (
            previous.size_preset == next_desktop.size_preset
            or not self.runtime_is_running_for_settings(next_settings)
        ):
            return
        await self._bounds.cancel()
        self.drain_pending_user_bounds_events()
        if previous.position.x is None or previous.position.y is None:
            return
        bounds = self.center_preserving_bounds(
            previous_desktop_settings=previous,
            next_size_preset=next_desktop.size_preset,
        )
        next_desktop.position.x = bounds["x"]
        next_desktop.position.y = bounds["y"]
        next_desktop.position.validate()

    def sync_from_settings(self, settings: Any) -> None:
        overlay = self.overlay_provider()
        if (
            OverlayApplicationOwner.normalized_target(settings.overlay.target)
            != OVERLAY_TARGET_DESKTOP
        ):
            return
        if overlay.active_target == OVERLAY_TARGET_DESKTOP and overlay.current_bridge() is not None:
            return
        self.set_interaction_mode(DESKTOP_INTERACTION_MODE_EDIT)

    async def apply_controls(
        self,
        controls: tuple[DesktopRuntimeControl, ...],
    ) -> None:
        for payload in controls:
            if payload.get("command") == "apply_window_bounds":
                bounds = self._bounds.bounds_from_payload(payload)
                if bounds is not None:
                    await self.broadcast_bounds(bounds)
            else:
                await self.broadcast(payload)

    async def close(self) -> None:
        await self._bounds.close()

    @staticmethod
    def _is_user_bounds_event(event: object) -> bool:
        if not isinstance(event, dict):
            return False
        payload = event.get("payload")
        return bool(
            isinstance(payload, dict)
            and payload.get("event") == "window_bounds_changed"
            and payload.get("source") == "user"
            and payload.get("persist") is True
        )

    def _log(
        self,
        message: str,
        level: int = logging.INFO,
        exception: Exception | None = None,
    ) -> None:
        self.log_detailed(message, level, exception)


__all__ = [
    "DESKTOP_BOUNDS_PERSIST_DEBOUNCE_S",
    "DESKTOP_INTERACTION_MODE_EDIT",
    "DESKTOP_INTERACTION_MODE_PASS_THROUGH",
    "DESKTOP_INTERACTION_MODES",
    "DesktopOverlayApplicationOwner",
]
