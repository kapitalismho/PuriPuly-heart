from __future__ import annotations

import asyncio
import json
import math
import threading
from collections.abc import Awaitable, Callable
from pathlib import Path

from puripuly_heart.core.lifecycle import LifecycleScope
from puripuly_heart.core.photo.bridge import PhotoBridge, valid_wrist_calibration
from puripuly_heart.core.photo.capture import PhotoCapture, PhotoCaptureWatcher
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
        enabled: Callable[[], bool] = lambda: True,
        wrist_calibration_path: Path | None = None,
        discard_translated: Callable[[PhotoCapture], Awaitable[None]] | None = None,
    ) -> None:
        self._resolve_route = resolve_route
        self._wrist_calibration_path = wrist_calibration_path
        self._diagnostic = diagnostic
        self._watcher = watcher or PhotoCaptureWatcher()
        self._enabled = enabled
        self._policy_lock = threading.Lock()
        self._policy_changed = asyncio.Event()
        self._policy_settled = asyncio.Event()
        self._applied_revision = -1
        self._policy_error: Exception | None = None
        self._terminal_error: Exception | None = None
        self._policy_revision = 0
        self._capture_generation = 0
        self._disable_requested = False
        self._loop: asyncio.AbstractEventLoop | None = None
        self._ready: asyncio.Future | None = None
        self._watching = False
        self._capture_task: asyncio.Task | None = None
        self._route: PhotoRoute | None = None
        self._scope = LifecycleScope("PhotoApplicationOwner")
        self._started = False
        self._frozen = False
        self.bridge = PhotoBridge(control=self._control)
        self.lifecycle = PhotoLifecycle(
            resolve_route=self._resolve_current_route,
            target_language=target_language,
            publish=self.bridge.publish,
            diagnostic=diagnostic,
            temporary_root=temporary_root,
            prepare=self._prepare,
            discard_translated=discard_translated,
        )
        self.supervisor = PhotoProcessSupervisor(
            bridge=self.bridge,
            renderer_changed=self.lifecycle.renderer_changed,
            diagnostic=diagnostic,
            runner=runner,
            backend=backend,
            temporary_root=temporary_root,
            interaction=self._interaction,
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

    def toggle_debug_limited(self) -> bool:
        self.lifecycle.debug_limited = not self.lifecycle.debug_limited
        return self.lifecycle.debug_limited

    def _interaction(self) -> dict | None:
        if self._wrist_calibration_path is None:
            return None
        try:
            calibration = json.loads(self._wrist_calibration_path.read_text(encoding="utf-8"))
        except OSError, ValueError:
            return None
        return {"wrist_calibration": calibration} if valid_wrist_calibration(calibration) else None

    def notify_policy_changed(self) -> None:
        disabled = not self._enabled()
        with self._policy_lock:
            self._policy_revision += 1
            if disabled:
                self._capture_generation += 1
                self._disable_requested = True
            loop = self._loop
        if loop is not None and not self._frozen:
            loop.call_soon_threadsafe(self._policy_changed.set)

    def _capture_current(self, generation: int) -> bool:
        with self._policy_lock:
            current = generation == self._capture_generation and not self._disable_requested
        return current and not self._frozen and self._enabled()

    async def _resolve_current_route(self) -> PhotoRoute | None:
        while not self._frozen and self._enabled():
            with self._policy_lock:
                revision = self._policy_revision
            route = await self._resolve_route()
            with self._policy_lock:
                current = revision == self._policy_revision
            if current:
                self._route = route if not self._frozen and self._enabled() else None
                return self._route
        return None

    async def start(self) -> None:
        if self._frozen:
            return
        if not self._started:
            self._started = True
            with self._policy_lock:
                self._loop = asyncio.get_running_loop()
            self._ready = self._loop.create_future()
            self._policy_changed.set()
            self._scope.create_task(self._watch(), name="capture-policy")
        if self._ready is not None:
            await asyncio.shield(self._ready)

    async def wait_policy_applied(self) -> None:
        if not self._started or self._frozen:
            return
        await self.start()
        while not self._frozen:
            self._policy_settled.clear()
            with self._policy_lock:
                if self._terminal_error is not None:
                    raise self._terminal_error
                if self._applied_revision >= self._policy_revision:
                    error = self._policy_error
                    if error is not None:
                        raise error
                    return
            await self._policy_settled.wait()

    async def _cancel_capture(self) -> None:
        task, self._capture_task = self._capture_task, None
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def _next_capture(self) -> PhotoCapture | Exception:
        try:
            return await self._watcher.wait_capture()
        except Exception as exc:
            return exc

    async def _close_watcher(self) -> None:
        await self._cancel_capture()
        self._watching = False
        await self._watcher.close()

    def _capture_failed(self) -> None:
        self._diagnostic("capture_unavailable", {"failure_code": "photo_capture_unavailable"})

    async def _apply_policy(self) -> None:
        with self._policy_lock:
            disable_requested = self._disable_requested
            self._disable_requested = False
        if disable_requested or not self._enabled():
            await self._close_watcher()
            self._route = None
            await self.lifecycle.refresh(None)
        if self._frozen or not self._enabled():
            return
        if not self._watching:
            try:
                await self._watcher.start()
                self._watching = True
            except Exception:
                await self._close_watcher()
                raise
        if self._frozen or not self._enabled():
            return
        self._route = await self._resolve_current_route()
        await self.lifecycle.refresh(self._route)

    async def _watch(self) -> None:
        capture_generation = 0
        try:
            while not self._frozen:
                if self._policy_changed.is_set():
                    self._policy_changed.clear()
                    with self._policy_lock:
                        revision = self._policy_revision
                    error = None
                    try:
                        await self._apply_policy()
                    except Exception as exc:
                        error = exc
                        self._capture_failed()
                        self._route = None
                        await self.lifecycle.refresh(None)
                    with self._policy_lock:
                        current = revision == self._policy_revision
                        if current:
                            self._applied_revision = revision
                            self._policy_error = error
                    if not current:
                        self._policy_changed.set()
                        continue
                    self._policy_settled.set()
                    if self._ready is not None and not self._ready.done():
                        self._ready.set_result(None)
                if self._frozen:
                    break
                await self.lifecycle.refresh(self._route)
                if self._watching and self._capture_task is None:
                    with self._policy_lock:
                        capture_generation = self._capture_generation
                    self._capture_task = self._scope.create_task(
                        self._next_capture(), name="capture"
                    )
                policy = self._scope.create_task(self._policy_changed.wait(), name="policy-wake")
                changed = self._scope.create_task(
                    self.lifecycle.wait_changed(), name="lifecycle-wake"
                )
                waiters = {policy, changed}
                if self._capture_task is not None:
                    waiters.add(self._capture_task)
                try:
                    done, _ = await asyncio.wait(
                        waiters,
                        timeout=self.lifecycle.refresh_delay,
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                finally:
                    policy.cancel()
                    changed.cancel()
                    await asyncio.gather(policy, changed, return_exceptions=True)
                if self._policy_changed.is_set():
                    continue
                if self._capture_task is not None and self._capture_task in done:
                    task, self._capture_task = self._capture_task, None
                    capture = task.result()
                    if isinstance(capture, Exception):
                        self._capture_failed()
                        await self._close_watcher()
                        continue
                    if not self._capture_current(capture_generation):
                        continue
                    try:
                        accepted = await self.lifecycle.captured(
                            capture,
                            is_current=lambda generation=capture_generation: self._capture_current(
                                generation
                            ),
                        )
                        if accepted and self._capture_current(capture_generation):
                            await self.supervisor.start()
                    except Exception:
                        self._capture_failed()
        except Exception as exc:
            self._terminal_error = exc
            self._capture_failed()
        finally:
            try:
                await self._close_watcher()
            except Exception as exc:
                if self._terminal_error is None:
                    self._terminal_error = exc
                    self._capture_failed()
            finally:
                self._policy_settled.set()
                if self._ready is not None and not self._ready.done():
                    self._ready.set_result(None)

    def _control(self, value: dict, generation: int) -> dict | None:
        if self._frozen or not self._enabled():
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
        elif kind == "photo_interaction":
            self._diagnostic(
                "interaction", {key: item for key, item in value.items() if key != "type"}
            )
        elif kind == "photo_image_timing":
            self._diagnostic(
                f"image_{value['stage']}",
                {key: value[key] for key in ("photo_id", "attempt_id", "block_id", "duration_ms")},
            )
        return None

    def stop_ingress(self) -> None:
        self._frozen = True
        with self._policy_lock:
            self._capture_generation += 1
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._policy_changed.set)
        self.lifecycle.stop_ingress()
        self._policy_settled.set()

    async def close(self) -> None:
        self.stop_ingress()
        try:
            await self._scope.close()
        finally:
            try:
                await self._watcher.close()
            finally:
                try:
                    await self.lifecycle.close()
                finally:
                    try:
                        await self.supervisor.close()
                    finally:
                        if self._ready is not None and not self._ready.done():
                            self._ready.set_result(None)
        if self._terminal_error is not None:
            raise self._terminal_error
