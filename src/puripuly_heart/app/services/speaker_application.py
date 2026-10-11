from __future__ import annotations

import asyncio
import shutil
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from puripuly_heart.config.settings_vnext.schema import SpeakerIntent
from puripuly_heart.core.audio.ownership import CaptureStreamInput
from puripuly_heart.core.local_asr.local_stt_runtime_installer import (
    LocalSTTProvisioningLease,
    _promote_staging_install,
    ensure_local_stt_installed,
)
from puripuly_heart.core.speaker import SpeakerEvidence, SpeakerRuntime
from puripuly_heart.runtime_layout import current_runtime_layout


@dataclass(frozen=True, slots=True)
class SpeakerAssetSnapshot:
    ready: bool
    artifact_path: Path
    runtime_dir: Path
    reason: str | None = None
    metadata: dict[str, object] | None = None


class SpeakerAssets(Protocol):
    async def inspect(self) -> SpeakerAssetSnapshot: ...
    async def install(
        self, *, download_consent: bool, source_path: Path | None = None
    ) -> SpeakerAssetSnapshot: ...
    async def cancel(self) -> None: ...
    async def close(self) -> None: ...


class NativeSpeakerAssets:
    def __init__(
        self, *, artifact_path: Path | None = None, runtime_dir: Path | None = None
    ) -> None:
        layout = current_runtime_layout()
        self.artifact_path = (
            artifact_path
            or layout.model_cache_root
            / "speaker"
            / "nemotron-3-diarization-f16-vulkan"
            / "model.gguf"
        )
        self.runtime_dir = runtime_dir or layout.native("nemotron")
        self._install_task: asyncio.Task | None = None
        self._cancel_event: threading.Event | None = None
        self._closed = False
        self.phase = "idle"

    async def inspect(self) -> SpeakerAssetSnapshot:
        def inspect() -> SpeakerAssetSnapshot:
            from puripuly_heart.app.adapters.nemotron import (
                inspect_artifact,
                validate_artifact_provenance,
                validate_runtime,
            )

            metadata = {}
            try:
                metadata["artifact"] = inspect_artifact(self.artifact_path)
                metadata["provenance"] = validate_artifact_provenance(
                    self.artifact_path, metadata=metadata["artifact"]
                )
                metadata["runtime"] = validate_runtime(self.runtime_dir)
                return SpeakerAssetSnapshot(
                    True, self.artifact_path, self.runtime_dir, metadata=metadata
                )
            except Exception as exc:
                return SpeakerAssetSnapshot(
                    False,
                    self.artifact_path,
                    self.runtime_dir,
                    reason=type(exc).__name__,
                    metadata=metadata,
                )

        try:
            return await asyncio.to_thread(inspect)
        except ImportError:
            return SpeakerAssetSnapshot(
                False, self.artifact_path, self.runtime_dir, reason="native_runtime_unavailable"
            )

    async def install(
        self, *, download_consent: bool, source_path: Path | None = None
    ) -> SpeakerAssetSnapshot:
        if self._closed:
            raise RuntimeError("speaker assets are closed")
        if self._install_task is not None and not self._install_task.done():
            raise RuntimeError("speaker installation already running")
        if source_path is None and not download_consent:
            raise ValueError("explicit independent speaker model download consent is required")

        cancel_event = threading.Event()
        self._cancel_event = cancel_event
        self.phase = "installing"
        self._install_task = asyncio.create_task(
            (
                self._install_local(source_path, download_consent, cancel_event)
                if source_path is not None
                else self._install_download(cancel_event)
            ),
            name="speaker:install",
        )
        try:
            await asyncio.shield(self._install_task)
            return await self.inspect()
        except asyncio.CancelledError:
            await self.cancel()
            raise
        finally:
            if self._install_task.done():
                self.phase = "idle"

    async def _install_local(
        self, source_path: Path, download_consent: bool, cancel_event: threading.Event
    ) -> None:
        from puripuly_heart.app.adapters.nemotron import (
            acquire_artifact,
            validate_artifact_provenance,
        )

        install_dir = self.artifact_path.parent
        staging_dir = install_dir.with_name(f"{install_dir.name}.staging-{uuid4().hex}")
        lease = await asyncio.to_thread(
            LocalSTTProvisioningLease.acquire,
            model_root=install_dir.parent,
            wait=True,
            cancel_event=cancel_event,
        )
        if lease is None:
            raise RuntimeError("speaker provisioning lease unavailable")
        try:
            await acquire_artifact(
                staging_dir / self.artifact_path.name,
                download_consent=download_consent,
                source_path=source_path,
                cancel_event=cancel_event,
            )
            await asyncio.to_thread(
                validate_artifact_provenance, staging_dir / self.artifact_path.name
            )
            await asyncio.to_thread(
                _promote_staging_install,
                staging_dir=staging_dir,
                install_dir=install_dir,
                cancel_event=cancel_event,
            )
        finally:
            await asyncio.to_thread(shutil.rmtree, staging_dir, ignore_errors=True)
            await asyncio.to_thread(lease.close)

    async def _install_download(self, cancel_event: threading.Event) -> None:
        from puripuly_heart.app.adapters.nemotron import (
            adopt_preconverted_artifact,
            verified_preconverted_manifest,
        )

        manifest = verified_preconverted_manifest()
        install_dir = self.artifact_path.parent
        download_root = install_dir.parent / f"{install_dir.name}.download-{uuid4().hex}"
        lease = await asyncio.to_thread(
            LocalSTTProvisioningLease.acquire,
            model_root=install_dir.parent,
            wait=True,
            cancel_event=cancel_event,
        )
        if lease is None:
            raise RuntimeError("speaker provisioning lease unavailable")
        try:
            if self.artifact_path.is_file():
                try:
                    await asyncio.to_thread(adopt_preconverted_artifact, self.artifact_path)
                    return
                except OSError, ValueError:
                    pass
            self.phase = "downloading"
            await ensure_local_stt_installed(
                model_root=download_root,
                manifest=manifest,
                preferred_source="huggingface",
                cancel_event=cancel_event,
            )
            staging_dir = download_root / manifest.install_dirname
            await asyncio.to_thread(
                adopt_preconverted_artifact, staging_dir / self.artifact_path.name
            )
            await asyncio.to_thread(
                _promote_staging_install,
                staging_dir=staging_dir,
                install_dir=install_dir,
                cancel_event=cancel_event,
            )
        finally:
            await asyncio.to_thread(shutil.rmtree, download_root, ignore_errors=True)
            await asyncio.to_thread(lease.close)

    async def cancel(self) -> None:
        if self._cancel_event is not None:
            self._cancel_event.set()
        if self._install_task is not None:
            await asyncio.shield(asyncio.gather(self._install_task, return_exceptions=True))
            self.phase = "idle"

    async def close(self) -> None:
        self._closed = True
        await self.cancel()


def create_speaker_runtime(intent: SpeakerIntent, assets: SpeakerAssetSnapshot) -> SpeakerRuntime:
    from puripuly_heart.app.adapters.nemotron import NemotronBackend

    return SpeakerRuntime(
        NemotronBackend(
            artifact_path=assets.artifact_path,
            runtime_dir=assets.runtime_dir,
            device=int(intent.vulkan_device_id),
            profile=intent.profile,
        )
    )


class SpeakerApplicationOwner:
    def __init__(
        self,
        *,
        assets: SpeakerAssets,
        runtime_factory: Callable[
            [SpeakerIntent, SpeakerAssetSnapshot], SpeakerRuntime
        ] = create_speaker_runtime,
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
