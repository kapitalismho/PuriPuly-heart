from __future__ import annotations

import asyncio
import math
from collections.abc import Awaitable, Callable
from pathlib import Path

from puripuly_heart.core.lifecycle import LifecycleScope
from puripuly_heart.core.photo.bridge import PhotoBridge
from puripuly_heart.core.photo.capture import PhotoCaptureWatcher
from puripuly_heart.core.photo.image import PreparedPhoto, load_prepared_photo
from puripuly_heart.core.photo.lifecycle import PhotoLifecycle, PhotoRoute
from puripuly_heart.core.photo.process import PhotoProcessRunner, PhotoProcessSupervisor


class PhotoApplicationOwner:
    def __init__(
        self,
        *,
        resolve_route: Callable[[], Awaitable[PhotoRoute | None]],
        target_language: Callable[[], str],
        diagnostic: Callable[[str, dict], None],
        watcher: PhotoCaptureWatcher | None = None,
        runner: PhotoProcessRunner | None = None,
        backend: str = "openvr",
        temporary_root: Path | None = None,
        poll_interval: float = 0.25,
    ) -> None:
        self._resolve_route = resolve_route
        self._diagnostic = diagnostic
        self._watcher = watcher or PhotoCaptureWatcher()
        self._poll_interval = poll_interval
        self._scope = LifecycleScope("PhotoApplicationOwner")
        self._started = False
        self._frozen = False
        self.bridge = PhotoBridge(control=self._control)
        self.lifecycle = PhotoLifecycle(
            resolve_route=resolve_route,
            target_language=target_language,
            publish=self.bridge.publish,
            diagnostic=diagnostic,
            temporary_root=temporary_root,
            prepare=self._prepare,
        )
        self.supervisor = PhotoProcessSupervisor(
            bridge=self.bridge,
            renderer_changed=self.lifecycle.renderer_changed,
            diagnostic=diagnostic,
            runner=runner,
            backend=backend,
            temporary_root=temporary_root,
        )
        self.bridge.abort_preparation = self.supervisor.abort_preparation

    async def _prepare(
        self,
        photo_id: str,
        attempt_id: int,
        source_path: Path,
        directory: Path,
        *,
        expected_file_id: tuple[int, int],
    ) -> PreparedPhoto:
        result = await self.bridge.prepare(
            photo_id,
            attempt_id,
            source_path,
            directory,
            expected_file_id=expected_file_id,
        )
        for stage in ("decode", "resize", "jpeg"):
            duration = result.get(f"{stage}_ms")
            if type(duration) in (int, float) and math.isfinite(duration) and duration >= 0:
                self._diagnostic(
                    stage,
                    {"photo_id": photo_id, "attempt_id": attempt_id, "duration_ms": duration},
                )
        return await load_prepared_photo(result, directory=directory)

    async def start(self) -> None:
        if self._started or self._frozen:
            return
        self._started = True
        try:
            await self._watcher.start()
        except Exception:
            self._diagnostic("capture_unavailable", {"failure_code": "photo_capture_unavailable"})
        if not self._frozen:
            self._scope.create_task(self._watch(), name="capture-policy")

    async def _watch(self) -> None:
        while not self._frozen:
            try:
                route = await self._resolve_route()
                await self.lifecycle.refresh(route)
                capture = await self._watcher.poll()
                if capture is not None and not self._frozen:
                    if await self.lifecycle.captured(capture):
                        await self.supervisor.start()
            except Exception:
                self._diagnostic(
                    "capture_unavailable", {"failure_code": "photo_capture_unavailable"}
                )
            await asyncio.sleep(self._poll_interval)

    def _control(self, value: dict, generation: int) -> dict | None:
        if self._frozen:
            return None
        kind = value["type"]
        if kind == "photo_pull":
            self.lifecycle.pull(value["photo_id"], value["event_id"])
        elif kind == "photo_hidden":
            self.lifecycle.hidden(value["photo_id"])
        elif kind == "photo_first_block_drawn":
            return self.lifecycle.first_block_drawn(
                value["photo_id"], value["attempt_id"], value["block_id"], generation
            )
        elif kind == "photo_image_timing":
            self._diagnostic(
                f"image_{value['stage']}",
                {key: value[key] for key in ("photo_id", "attempt_id", "block_id", "duration_ms")},
            )
        return None

    def stop_ingress(self) -> None:
        self._frozen = True
        self.lifecycle.stop_ingress()

    async def close(self) -> None:
        self.stop_ingress()
        await self._scope.close()
        try:
            await self.lifecycle.close()
        finally:
            await self.supervisor.close()
