from __future__ import annotations

import asyncio
import math
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import TypeAlias

DesktopBounds: TypeAlias = dict[str, int | float]
DesktopBoundsDiagnosticsSink = Callable[[str, Mapping[str, object]], None]


def is_finite_non_bool_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


@dataclass(slots=True)
class DesktopOverlayBoundsOwner:
    persist_bounds: Callable[[DesktopBounds], Awaitable[None]]
    debounce_seconds: Callable[[], float]
    minimum_width: int
    minimum_height: int
    diagnostics_sink: DesktopBoundsDiagnosticsSink | None = None
    _persist_task: asyncio.Task[None] | None = field(
        init=False,
        default=None,
        repr=False,
    )
    _pending_bounds: DesktopBounds | None = field(
        init=False,
        default=None,
        repr=False,
    )
    _suppressed_signatures: set[tuple[float, float, float, float]] = field(
        init=False,
        default_factory=set,
        repr=False,
    )

    @property
    def owner_name(self) -> str:
        return "DesktopOverlayBoundsOwner"

    @property
    def persist_task(self) -> asyncio.Task[None] | None:
        return self._persist_task

    @property
    def pending_bounds(self) -> DesktopBounds | None:
        return self._pending_bounds

    def replace_pending_bounds(self, bounds: DesktopBounds | None) -> None:
        self._pending_bounds = dict(bounds) if bounds is not None else None

    def lifecycle_owner_snapshot(self) -> dict[str, object]:
        return {
            "owner": self.owner_name,
            "resource_fields": ("_persist_task", "_pending_bounds"),
            "stop_ingress": "discard pending desktop bounds",
            "shutdown_policy": "cancel and gather bounds persistence task",
            "late_callback_rule": "only the current task clears owner state",
        }

    def bounds_from_payload(
        self,
        payload: Mapping[object, object],
    ) -> DesktopBounds | None:
        x = payload.get("x")
        y = payload.get("y")
        width = payload.get("width")
        height = payload.get("height")
        if not (
            is_finite_non_bool_number(x)
            and is_finite_non_bool_number(y)
            and is_finite_non_bool_number(width)
            and is_finite_non_bool_number(height)
        ):
            return None
        if width < self.minimum_width or height < self.minimum_height:
            return None
        return {
            "x": x,
            "y": y,
            "width": width,
            "height": height,
        }

    def is_valid_event_payload(self, payload: Mapping[object, object]) -> bool:
        source = payload.get("source")
        persist = payload.get("persist")
        if source not in {"user", "reset", "programmatic", "launch_repair"}:
            return False
        expected_persist = source in {"user", "reset"}
        return bool(
            payload.get("event") == "window_bounds_changed"
            and isinstance(persist, bool)
            and persist is expected_persist
            and self.bounds_from_payload(payload) is not None
        )

    def track_apply_control(self, payload: Mapping[str, object]) -> None:
        if payload.get("command") != "apply_window_bounds":
            return
        bounds = self.bounds_from_payload(payload)
        if bounds is None:
            return
        self._suppressed_signatures.add(self._signature(bounds))

    def consume_suppressed(self, bounds: DesktopBounds) -> bool:
        signature = self._signature(bounds)
        if signature not in self._suppressed_signatures:
            return False
        self._suppressed_signatures.discard(signature)
        return True

    def discard_suppressed(self, bounds: DesktopBounds) -> None:
        self._suppressed_signatures.discard(self._signature(bounds))

    def clear_suppressed(self) -> None:
        self._suppressed_signatures.clear()

    def schedule_persistence(self, bounds: DesktopBounds) -> None:
        self._pending_bounds = dict(bounds)
        task = self._persist_task
        if task is not None and not task.done():
            task.cancel()
        self._persist_task = asyncio.create_task(
            self.persist_after_debounce(),
            name=f"{self.owner_name}:persist",
        )

    async def persist_after_debounce(self) -> None:
        current_task = asyncio.current_task()
        try:
            await asyncio.sleep(max(0.0, float(self.debounce_seconds())))
            bounds = self._pending_bounds
            self._pending_bounds = None
            if bounds is None:
                return
            await self.persist_bounds(dict(bounds))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._emit(
                "desktop_overlay_bounds_persistence_failed",
                {"error_type": type(exc).__name__},
            )
        finally:
            if self._persist_task is current_task:
                self._persist_task = None

    async def cancel(self) -> None:
        current_task = asyncio.current_task()
        task = self._persist_task
        self._persist_task = None
        self._pending_bounds = None
        if task is not None and task is not current_task and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    def discard(self) -> None:
        try:
            current_task = asyncio.current_task()
        except RuntimeError:
            current_task = None
        task = self._persist_task
        self._persist_task = None
        self._pending_bounds = None
        if task is not None and task is not current_task and not task.done():
            task.cancel()

    async def close(self) -> None:
        await self.cancel()

    @staticmethod
    def _signature(bounds: DesktopBounds) -> tuple[float, float, float, float]:
        return (
            float(bounds["x"]),
            float(bounds["y"]),
            float(bounds["width"]),
            float(bounds["height"]),
        )

    def _emit(self, event: str, metadata: Mapping[str, object]) -> None:
        if self.diagnostics_sink is None:
            return
        try:
            self.diagnostics_sink(event, metadata)
        except Exception:
            return


__all__ = [
    "DesktopBounds",
    "DesktopOverlayBoundsOwner",
    "is_finite_non_bool_number",
]
