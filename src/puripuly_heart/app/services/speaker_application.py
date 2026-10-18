from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path

from puripuly_heart.app.ports.speaker_assets import SpeakerAssets, SpeakerAssetSnapshot
from puripuly_heart.config.settings_vnext.schema import SpeakerIntent
from puripuly_heart.core.audio.ownership import CaptureStreamInput
from puripuly_heart.core.speaker import SpeakerEvidence, SpeakerRuntime


class SpeakerApplicationOwner:
    def __init__(
        self,
        *,
        assets: SpeakerAssets,
        runtime_factory: Callable[[SpeakerIntent, SpeakerAssetSnapshot], SpeakerRuntime],
    ) -> None:
        self.assets = assets
        self._runtime_factory = runtime_factory
        self._intent = SpeakerIntent()
        self._runtime: SpeakerRuntime | None = None
        self._start_task: asyncio.Task[None] | None = None
        self._cleanup_runtime: SpeakerRuntime | None = None
        self._generation = 0
        self._closed = False
        self._state = "disabled"
        self._reason: str | None = None
        self._asset_snapshot: SpeakerAssetSnapshot | None = None

    def snapshot(self) -> dict[str, object]:
        runtime_owner = self._runtime or self._cleanup_runtime
        runtime = runtime_owner.snapshot() if runtime_owner is not None else None
        cleanup_pending = self._cleanup_runtime is not None
        return {
            "intent": self._intent,
            "state": (
                "cleanup_pending"
                if cleanup_pending
                else runtime.state if runtime is not None and self._reason is None else self._state
            ),
            "reason": (
                runtime.error
                if cleanup_pending and runtime is not None
                else self._reason or (runtime.error if runtime is not None else None)
            ),
            "generation": self._generation,
            "assets": self._asset_snapshot,
            "install_phase": getattr(self.assets, "phase", "idle"),
            "cleanup_pending": cleanup_pending,
            "runtime": runtime,
        }

    async def prepare(self) -> dict[str, object]:
        self._asset_snapshot = await self.assets.inspect()
        return self.snapshot()

    async def apply(self, intent: SpeakerIntent) -> dict[str, object]:
        if self._closed:
            raise RuntimeError("speaker owner is closed")
        if (
            intent.enabled
            and self._intent.enabled
            and intent.profile == self._intent.profile
            and intent.vulkan_device_id == self._intent.vulkan_device_id
            and self._runtime is not None
            and self._runtime.snapshot().ready
        ):
            self._intent = intent
            return self.snapshot()
        self._generation += 1
        generation = self._generation
        self._intent = intent
        await self._stop_runtime()
        if generation != self._generation:
            return self.snapshot()
        self._reason = None
        if not intent.enabled:
            self._state = "disabled"
            return self.snapshot()
        self._state = "starting"
        self._asset_snapshot = await self.assets.inspect()
        if generation != self._generation:
            return self.snapshot()
        if not self._asset_snapshot.ready:
            self._state = "unavailable"
            self._reason = self._asset_snapshot.reason
            return self.snapshot()
        try:
            runtime = self._runtime_factory(intent, self._asset_snapshot)
        except Exception as exc:
            self._state = "unavailable"
            self._reason = type(exc).__name__
            return self.snapshot()
        self._runtime = runtime
        task = asyncio.create_task(runtime.start(), name="speaker:start")
        self._start_task = task
        try:
            await asyncio.shield(task)
            if generation == self._generation:
                snapshot = runtime.snapshot()
                self._state = snapshot.state
                self._reason = snapshot.error
        except asyncio.CancelledError:
            if generation == self._generation:
                await self._stop_runtime()
                self._state = "unavailable"
                self._reason = "startup_cancelled"
            raise
        except Exception as exc:
            if generation == self._generation:
                await self._stop_runtime()
                self._state = "unavailable"
                self._reason = type(exc).__name__
        return self.snapshot()

    async def install(self, *, source_path: Path | None = None) -> dict[str, object]:
        if self._intent.enabled or self._cleanup_runtime is not None:
            raise ValueError("disable speaker analysis before replacing its model")
        self._asset_snapshot = await self.assets.install(
            download_consent=self._intent.download_consent, source_path=source_path
        )
        return self.snapshot()

    async def cancel_install(self) -> None:
        await self.assets.cancel()

    def observe_input(self, capture_session_id: str, event: CaptureStreamInput) -> None:
        runtime = self._runtime
        if runtime is not None:
            try:
                runtime.observe_input(capture_session_id, event)
            except Exception as exc:
                runtime.end_capture(capture_session_id, "speaker_input_failed")
                self._state = "unavailable"
                self._reason = type(exc).__name__

    def end_capture(self, capture_session_id: str, reason: str) -> None:
        if self._runtime is not None:
            self._runtime.end_capture(capture_session_id, reason)

    def reset(self) -> None:
        if self._runtime is not None:
            self._runtime.reset(reason="manual_reset")

    def drain_evidence(self, max_items: int) -> tuple[SpeakerEvidence, ...]:
        if self._runtime is None:
            return ()
        return self._runtime.drain_evidence(max_items)

    async def _stop_runtime(self) -> None:
        runtime, self._runtime = self._runtime, None
        task, self._start_task = self._start_task, None
        if task is not None and not task.done():
            task.cancel()
        if runtime is not None:
            self._cleanup_runtime = runtime
        try:
            if self._cleanup_runtime is not None:
                await self._cleanup_runtime.stop()
                self._cleanup_runtime = None
        finally:
            if task is not None:
                await asyncio.gather(task, return_exceptions=True)

    async def close(self) -> None:
        self._closed = True
        self._generation += 1
        try:
            await self._stop_runtime()
        finally:
            await self.assets.close()
        self._state = "closed"
