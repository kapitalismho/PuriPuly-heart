from __future__ import annotations

import asyncio
import inspect
import threading
import time
from collections import deque
from collections.abc import Awaitable, Callable
from pathlib import Path

from puripuly_heart.core.local_asr_provisioning import (
    LocalASRInstallRequest,
    LocalASRInstallResult,
    LocalASRModelProvisioningState,
    LocalASRProvisioningActivity,
    LocalASRProvisioningBackend,
    LocalASRProvisioningDiagnostic,
    LocalASRProvisioningSnapshot,
)
from puripuly_heart.core.local_gpu_assets import (
    LocalGPUInstallSnapshot,
    inspect_local_gpu_install,
    load_local_gpu_asset_manifest,
)
from puripuly_heart.core.local_stt_assets import (
    LOCAL_QWEN_GPU_MODEL_ID,
    REQUIRED_CPU_LOCAL_STT_MODEL_IDS,
    InstalledLocalSTTManifest,
    LocalSTTAssetManifest,
    LocalSTTInstallState,
    LocalSTTManifestInvalidError,
    load_local_stt_asset_manifest,
)
from puripuly_heart.core.local_stt_catalog import (
    LocalCPUInstallSnapshot,
    LocalCPUModelInstall,
    inspect_local_cpu_model_installs,
)
from puripuly_heart.core.local_stt_download_port import HuggingFaceDownloadPort
from puripuly_heart.core.local_stt_runtime_installer import (
    LocalSTTRuntimeInstallCancelled,
    LocalSTTRuntimeInstallError,
    RuntimeLocalSTTStatusUpdate,
    cleanup_local_stt_install_residue,
    ensure_local_stt_installed,
)
from puripuly_heart.core.runtime.local_stt_download import LocalSTTDownloadRuntime

ProvisioningStateChanged = Callable[
    [LocalASRProvisioningSnapshot],
    Awaitable[None] | None,
]
ProvisioningDiagnosticSink = Callable[
    [LocalASRProvisioningDiagnostic],
    Awaitable[None] | None,
]
CPUModelInspector = Callable[..., LocalCPUInstallSnapshot]
GPUModelInspector = Callable[..., LocalGPUInstallSnapshot]
ManifestLoader = Callable[[str], LocalSTTAssetManifest]
GPUManifestLoader = Callable[[], LocalSTTAssetManifest]
ProvisioningInstaller = Callable[..., Awaitable[InstalledLocalSTTManifest]]
ProvisioningResidueCleaner = Callable[..., tuple[Path, ...] | None]
DownloadRuntimeFactory = Callable[[], LocalSTTDownloadRuntime]


class LocalASRProvisioningOwner:
    resource_fields = (
        "_cpu_install_runtime",
        "_gpu_install_runtime",
        "installer cancel events",
        "Xet helper processes",
        "staging and backup directories",
        "model-root cross-process provisioning lease",
    )
    stop_ingress = "reject new inspect and install commands"
    shutdown_policy = "cancel CPU and GPU install tasks, signal installers, and await close"
    late_callback_rule = "ignore status callbacks whose owner generation is no longer current"

    def __init__(
        self,
        *,
        model_root: Path | None = None,
        state_changed: ProvisioningStateChanged | None = None,
        diagnostic_sink: ProvisioningDiagnosticSink | None = None,
        diagnostics_capacity: int = 256,
        cpu_model_inspector: CPUModelInspector = inspect_local_cpu_model_installs,
        gpu_model_inspector: GPUModelInspector = inspect_local_gpu_install,
        manifest_loader: ManifestLoader = load_local_stt_asset_manifest,
        gpu_manifest_loader: GPUManifestLoader = load_local_gpu_asset_manifest,
        installer: ProvisioningInstaller = ensure_local_stt_installed,
        residue_cleaner: ProvisioningResidueCleaner = cleanup_local_stt_install_residue,
        huggingface_downloader: HuggingFaceDownloadPort | None = None,
        download_runtime_factory: DownloadRuntimeFactory = LocalSTTDownloadRuntime,
    ) -> None:
        if diagnostics_capacity < 1:
            raise ValueError("diagnostics_capacity must be positive")
        self._model_root = model_root
        self._state_changed = state_changed
        self._diagnostic_sink = diagnostic_sink
        self._cpu_model_inspector = cpu_model_inspector
        self._gpu_model_inspector = gpu_model_inspector
        self._manifest_loader = manifest_loader
        self._gpu_manifest_loader = gpu_manifest_loader
        self._installer = installer
        self._residue_cleaner = residue_cleaner
        self._huggingface_downloader = huggingface_downloader
        self._cpu_install_runtime = download_runtime_factory()
        self._gpu_install_runtime = download_runtime_factory()
        self._models = {
            model_id: LocalASRModelProvisioningState(
                model_id=model_id,
                backend="cpu",
                integrity="missing",
            )
            for model_id in REQUIRED_CPU_LOCAL_STT_MODEL_IDS
        }
        self._models[LOCAL_QWEN_GPU_MODEL_ID] = LocalASRModelProvisioningState(
            model_id=LOCAL_QWEN_GPU_MODEL_ID,
            backend="gpu",
            integrity="not_requested",
        )
        self._activities: dict[LocalASRProvisioningBackend, LocalASRProvisioningActivity] = {}
        self._diagnostics: deque[LocalASRProvisioningDiagnostic] = deque(
            maxlen=diagnostics_capacity
        )
        self._revision = 0
        self._closing = False
        self._closed = False
        self._close_lock = asyncio.Lock()
        self._cleanup_lock = asyncio.Lock()
        self._startup_cleanup_complete = False

    @property
    def owner_name(self) -> str:
        return "LocalASRProvisioningOwner"

    @property
    def snapshot(self) -> LocalASRProvisioningSnapshot:
        return LocalASRProvisioningSnapshot(
            models=tuple(self._models.values()),
            required_cpu_model_ids=REQUIRED_CPU_LOCAL_STT_MODEL_IDS,
            gpu_model_id=LOCAL_QWEN_GPU_MODEL_ID,
            activities=tuple(self._activities.values()),
            revision=self._revision,
            closed=self._closed,
        )

    @property
    def diagnostics(self) -> tuple[LocalASRProvisioningDiagnostic, ...]:
        return tuple(self._diagnostics)

    @property
    def is_closed(self) -> bool:
        return self._closed

    def lifecycle_owner_snapshot(self) -> dict[str, object]:
        return {
            "owner": self.owner_name,
            "resource_fields": self.resource_fields,
            "stop_ingress": self.stop_ingress,
            "shutdown_policy": self.shutdown_policy,
            "late_callback_rule": self.late_callback_rule,
        }

    async def inspect_cpu(
        self,
        model_ids: tuple[str, ...] | None = None,
        *,
        verify_checksums: bool = False,
    ) -> LocalASRProvisioningSnapshot:
        self._require_open("inspect CPU assets")
        await self._ensure_startup_cleanup(wait_for_lease=False)
        requested = model_ids or REQUIRED_CPU_LOCAL_STT_MODEL_IDS
        self._validate_model_ids("cpu", requested)
        installs = await asyncio.to_thread(
            self._inspect_cpu_models,
            requested,
            verify_checksums,
        )
        for install in installs.models:
            current = self._models[install.model_id]
            operation = current.operation if current.operation == "downloading" else "idle"
            self._models[install.model_id] = self._state_from_install(
                model_id=install.model_id,
                backend="cpu",
                state=install.state,
                operation=operation,
            )
        await self._publish_state()
        return self.snapshot

    async def inspect_gpu(
        self,
        *,
        explicit_intent: bool,
        verify_checksums: bool = False,
    ) -> LocalASRProvisioningSnapshot:
        self._require_open("inspect GPU assets")
        await self._ensure_startup_cleanup(wait_for_lease=False)
        if not explicit_intent:
            self._models[LOCAL_QWEN_GPU_MODEL_ID] = LocalASRModelProvisioningState(
                model_id=LOCAL_QWEN_GPU_MODEL_ID,
                backend="gpu",
                integrity="not_requested",
            )
            await self._publish_state()
            return self.snapshot
        gpu_snapshot = await asyncio.to_thread(
            self._inspect_gpu_model,
            verify_checksums,
        )
        state = gpu_snapshot.state or LocalSTTInstallState(status="missing")
        current = self._models[LOCAL_QWEN_GPU_MODEL_ID]
        operation = current.operation if current.operation == "downloading" else "idle"
        self._models[LOCAL_QWEN_GPU_MODEL_ID] = self._state_from_install(
            model_id=LOCAL_QWEN_GPU_MODEL_ID,
            backend="gpu",
            state=state,
            operation=operation,
        )
        await self._publish_state()
        return self.snapshot

    def start_install(
        self,
        request: LocalASRInstallRequest,
    ) -> asyncio.Task[LocalASRInstallResult]:
        self._require_open("start install")
        self._validate_request(request)
        runtime = self._runtime_for(request.backend)
        return runtime.start(
            origin=request.origin,
            run_download=lambda cancel_event, generation: self._run_install(
                request,
                cancel_event=cancel_event,
                generation=generation,
            ),
        )

    async def report_model_validation_failure(
        self,
        model_id: str,
        *,
        failure_type: str,
    ) -> LocalASRProvisioningSnapshot:
        self._require_open("report model validation failure")
        if model_id not in self._models:
            raise ValueError("unknown Local ASR model identity")
        current = self._models[model_id]
        self._models[model_id] = LocalASRModelProvisioningState(
            model_id=model_id,
            backend=current.backend,
            integrity="invalid",
        )
        await self._emit_diagnostic(
            LocalASRProvisioningDiagnostic(
                event="validation",
                backend=current.backend,
                model_id=model_id,
                outcome="failed",
                failure_type=failure_type,
            )
        )
        await self._publish_state()
        return self.snapshot

    async def cancel_install(self, backend: LocalASRProvisioningBackend) -> None:
        await self._runtime_for(backend).cancel()

    async def close(self) -> None:
        if self._closed and not self._activities:
            return
        async with self._close_lock:
            if self._closed and not self._activities:
                return
            self._closing = True
            self._closed = True
            await self._publish_state()
            results = await asyncio.gather(
                self._cpu_install_runtime.close(),
                self._gpu_install_runtime.close(),
                return_exceptions=True,
            )
            self._activities.clear()
            self._closing = False
            await self._publish_state()
            failures = [result for result in results if isinstance(result, BaseException)]
            if len(failures) == 1:
                raise failures[0]
            if failures:
                raise ExceptionGroup("Local ASR provisioning close failed", failures)

    async def _run_install(
        self,
        request: LocalASRInstallRequest,
        *,
        cancel_event: threading.Event,
        generation: int,
    ) -> LocalASRInstallResult:
        await self._ensure_startup_cleanup(
            wait_for_lease=True,
            cancel_event=cancel_event,
        )
        runtime = self._runtime_for(request.backend)
        installed: list[str] = []
        failed: list[str] = []
        cancelled = False
        active_model_id: str | None = None
        try:
            for model_id in request.model_ids:
                active_model_id = model_id
                started_at = time.monotonic()
                self._set_operation(
                    model_id,
                    operation="downloading",
                    activity=LocalASRProvisioningActivity(
                        backend=request.backend,
                        model_id=model_id,
                        origin=request.origin,
                        progress_percent=0,
                        generation=generation,
                    ),
                )
                await self._emit_diagnostic(
                    LocalASRProvisioningDiagnostic(
                        event="install",
                        backend=request.backend,
                        model_id=model_id,
                        origin=request.origin,
                        outcome="started",
                    )
                )
                await self._publish_state()

                async def on_status(update: RuntimeLocalSTTStatusUpdate) -> None:
                    await runtime.dispatch_status_update(
                        update,
                        generation=generation,
                        on_status=lambda current: self._handle_install_status(
                            request,
                            model_id=model_id,
                            generation=generation,
                            update=current,
                        ),
                    )

                try:
                    manifest = self._manifest_for(request.backend, model_id)
                    installed_manifest = await self._installer(
                        model_id=model_id,
                        manifest=manifest,
                        locale=request.locale,
                        model_root=self._model_root,
                        on_status=on_status,
                        cancel_event=cancel_event,
                        huggingface_downloader=self._huggingface_downloader,
                    )
                    if not runtime.is_current_generation(generation):
                        cancelled = True
                        break
                    verified = await self._verify_promoted_model(request.backend, model_id)
                    if not verified:
                        raise LocalSTTRuntimeInstallError(
                            "promoted local ASR asset failed integrity validation"
                        )
                    self._models[model_id] = LocalASRModelProvisioningState(
                        model_id=model_id,
                        backend=request.backend,
                        integrity="ready",
                    )
                    if installed_manifest.model_id != model_id:
                        raise LocalSTTRuntimeInstallError(
                            "installed manifest does not match requested model"
                        )
                    installed.append(model_id)
                    await self._emit_diagnostic(
                        LocalASRProvisioningDiagnostic(
                            event="install",
                            backend=request.backend,
                            model_id=model_id,
                            origin=request.origin,
                            outcome="ready",
                            elapsed_seconds=time.monotonic() - started_at,
                        )
                    )
                except LocalSTTRuntimeInstallCancelled:
                    cancelled = True
                    await self._restore_integrity_after_interruption(request.backend, model_id)
                    self._set_operation(model_id, operation="cancelled")
                    await self._emit_diagnostic(
                        LocalASRProvisioningDiagnostic(
                            event="install",
                            backend=request.backend,
                            model_id=model_id,
                            origin=request.origin,
                            outcome="cancelled",
                            elapsed_seconds=time.monotonic() - started_at,
                        )
                    )
                    break
                except Exception as exc:
                    failed.append(model_id)
                    await self._restore_integrity_after_interruption(request.backend, model_id)
                    self._set_operation(model_id, operation="download_failed")
                    await self._emit_diagnostic(
                        LocalASRProvisioningDiagnostic(
                            event="install",
                            backend=request.backend,
                            model_id=model_id,
                            origin=request.origin,
                            outcome="failed",
                            elapsed_seconds=time.monotonic() - started_at,
                            failure_type=type(exc).__name__,
                        )
                    )
                finally:
                    await self._publish_state()
        except asyncio.CancelledError:
            cancelled = True
            if active_model_id is not None:
                await self._restore_integrity_after_interruption(
                    request.backend,
                    active_model_id,
                )
                self._set_operation(active_model_id, operation="cancelled")
                await self._emit_diagnostic(
                    LocalASRProvisioningDiagnostic(
                        event="install",
                        backend=request.backend,
                        model_id=active_model_id,
                        origin=request.origin,
                        outcome="cancelled",
                    )
                )
        finally:
            activity = self._activities.get(request.backend)
            if activity is not None and activity.generation == generation:
                self._activities.pop(request.backend, None)
            await self._publish_state()
        return LocalASRInstallResult(
            request=request,
            installed_model_ids=tuple(installed),
            failed_model_ids=tuple(failed),
            cancelled=cancelled,
            snapshot=self.snapshot,
        )

    async def _ensure_startup_cleanup(
        self,
        *,
        wait_for_lease: bool,
        cancel_event: threading.Event | None = None,
    ) -> None:
        if self._startup_cleanup_complete:
            return
        async with self._cleanup_lock:
            if self._startup_cleanup_complete:
                return
            started_at = time.monotonic()
            cleanup_finished = False
            try:
                manifests = tuple(
                    self._manifest_loader(model_id) for model_id in REQUIRED_CPU_LOCAL_STT_MODEL_IDS
                ) + (self._gpu_manifest_loader(),)
                removed = await asyncio.to_thread(
                    self._residue_cleaner,
                    model_root=self._model_root,
                    install_dirnames=tuple(manifest.install_dirname for manifest in manifests),
                    wait_for_lease=wait_for_lease,
                    cancel_event=cancel_event,
                )
                if removed is None:
                    return
                cleanup_finished = True
                await self._emit_diagnostic(
                    LocalASRProvisioningDiagnostic(
                        event="cleanup",
                        origin="startup",
                        outcome="ready",
                        elapsed_seconds=time.monotonic() - started_at,
                    )
                )
                if removed:
                    await self._publish_state()
            except Exception as exc:
                cleanup_finished = True
                await self._emit_diagnostic(
                    LocalASRProvisioningDiagnostic(
                        event="cleanup",
                        origin="startup",
                        outcome="failed",
                        elapsed_seconds=time.monotonic() - started_at,
                        failure_type=type(exc).__name__,
                    )
                )
            finally:
                if cleanup_finished:
                    self._startup_cleanup_complete = True

    async def _handle_install_status(
        self,
        request: LocalASRInstallRequest,
        *,
        model_id: str,
        generation: int,
        update: RuntimeLocalSTTStatusUpdate,
    ) -> None:
        activity = self._activities.get(request.backend)
        if activity is None or activity.generation != generation or activity.model_id != model_id:
            return
        self._activities[request.backend] = LocalASRProvisioningActivity(
            backend=request.backend,
            model_id=model_id,
            origin=request.origin,
            progress_percent=update.percent,
            generation=generation,
        )
        if update.status == "downloading":
            self._set_operation(model_id, operation="downloading")
        await self._publish_state()

    async def _verify_promoted_model(
        self,
        backend: LocalASRProvisioningBackend,
        model_id: str,
    ) -> bool:
        if backend == "cpu":
            installs = await asyncio.to_thread(
                self._inspect_cpu_models,
                (model_id,),
                True,
            )
            state = installs.state_for(model_id)
            self._models[model_id] = self._state_from_install(
                model_id=model_id,
                backend="cpu",
                state=state,
            )
            return state.status == "ready" and state.installed_manifest is not None
        gpu_snapshot = await asyncio.to_thread(self._inspect_gpu_model, True)
        state = gpu_snapshot.state or LocalSTTInstallState(status="missing")
        self._models[model_id] = self._state_from_install(
            model_id=model_id,
            backend="gpu",
            state=state,
        )
        return gpu_snapshot.activation_allowed

    async def _restore_integrity_after_interruption(
        self,
        backend: LocalASRProvisioningBackend,
        model_id: str,
    ) -> None:
        if backend == "cpu":
            installs = await asyncio.to_thread(
                self._inspect_cpu_models,
                (model_id,),
                False,
            )
            self._models[model_id] = self._state_from_install(
                model_id=model_id,
                backend="cpu",
                state=installs.state_for(model_id),
            )
            return
        gpu_snapshot = await asyncio.to_thread(self._inspect_gpu_model, False)
        state = gpu_snapshot.state or LocalSTTInstallState(status="missing")
        self._models[model_id] = self._state_from_install(
            model_id=model_id,
            backend="gpu",
            state=state,
        )

    def _inspect_cpu_models(
        self,
        model_ids: tuple[str, ...],
        verify_checksums: bool,
    ) -> LocalCPUInstallSnapshot:
        installs: list[LocalCPUModelInstall] = []
        for model_id in model_ids:
            try:
                manifest = self._manifest_loader(model_id)
                if manifest.model_id != model_id:
                    raise LocalSTTManifestInvalidError(
                        "local ASR manifest model_id does not match catalog identity"
                    )
                snapshot = self._cpu_model_inspector(
                    (model_id,),
                    self._model_root,
                    manifests={model_id: manifest},
                    verify_checksums=verify_checksums,
                )
                installs.append(snapshot.models[0])
            except Exception:
                installs.append(
                    LocalCPUModelInstall(
                        model_id=model_id,
                        state=LocalSTTInstallState(status="invalid"),
                    )
                )
        return LocalCPUInstallSnapshot(models=tuple(installs))

    def _inspect_gpu_model(self, verify_checksums: bool) -> LocalGPUInstallSnapshot:
        try:
            manifest = self._gpu_manifest_loader()
            if manifest.model_id != LOCAL_QWEN_GPU_MODEL_ID:
                raise LocalSTTManifestInvalidError(
                    "local GPU manifest model_id does not match catalog identity"
                )
            return self._gpu_model_inspector(
                explicit_opt_in=True,
                model_root=self._model_root,
                verify_checksums=verify_checksums,
                manifest=manifest,
            )
        except Exception:
            return LocalGPUInstallSnapshot(
                explicit_opt_in=True,
                status="invalid",
                state=LocalSTTInstallState(status="invalid"),
            )

    def _manifest_for(
        self,
        backend: LocalASRProvisioningBackend,
        model_id: str,
    ) -> LocalSTTAssetManifest:
        manifest = (
            self._gpu_manifest_loader() if backend == "gpu" else self._manifest_loader(model_id)
        )
        if manifest.model_id != model_id:
            raise LocalSTTRuntimeInstallError(
                "local ASR manifest model_id does not match requested model"
            )
        return manifest

    def _validate_request(self, request: LocalASRInstallRequest) -> None:
        if not request.origin.strip():
            raise ValueError("install origin must not be empty")
        self._validate_model_ids(request.backend, request.model_ids)
        if request.backend == "gpu" and not request.explicit_gpu_intent:
            raise ValueError("GPU installation requires explicit application intent")
        if request.backend == "gpu" and request.model_ids != (LOCAL_QWEN_GPU_MODEL_ID,):
            raise ValueError("GPU installation requires the selected GPU model identity")

    @staticmethod
    def _validate_model_ids(
        backend: LocalASRProvisioningBackend,
        model_ids: tuple[str, ...],
    ) -> None:
        if not model_ids:
            raise ValueError("model_ids must not be empty")
        if len(model_ids) != len(set(model_ids)):
            raise ValueError("model_ids must be unique")
        allowed = (
            frozenset(REQUIRED_CPU_LOCAL_STT_MODEL_IDS)
            if backend == "cpu"
            else frozenset({LOCAL_QWEN_GPU_MODEL_ID})
        )
        if not set(model_ids) <= allowed:
            raise ValueError(f"unknown {backend} Local ASR model identity")

    def _runtime_for(self, backend: LocalASRProvisioningBackend) -> LocalSTTDownloadRuntime:
        return self._cpu_install_runtime if backend == "cpu" else self._gpu_install_runtime

    def _require_open(self, operation: str) -> None:
        if self._closing or self._closed:
            state = "closing" if self._closing else "closed"
            raise RuntimeError(f"{self.owner_name} is {state}; cannot {operation}")

    def _set_operation(
        self,
        model_id: str,
        *,
        operation: str,
        activity: LocalASRProvisioningActivity | None = None,
    ) -> None:
        current = self._models[model_id]
        self._models[model_id] = LocalASRModelProvisioningState(
            model_id=current.model_id,
            backend=current.backend,
            integrity=current.integrity,
            operation=operation,
        )
        if activity is not None:
            self._activities[current.backend] = activity

    @staticmethod
    def _state_from_install(
        *,
        model_id: str,
        backend: LocalASRProvisioningBackend,
        state: LocalSTTInstallState,
        operation: str = "idle",
    ) -> LocalASRModelProvisioningState:
        return LocalASRModelProvisioningState(
            model_id=model_id,
            backend=backend,
            integrity=state.status,
            operation=operation,
        )

    async def _publish_state(self) -> None:
        self._revision += 1
        if self._state_changed is None:
            return
        result = self._state_changed(self.snapshot)
        if inspect.isawaitable(result):
            await result

    async def _emit_diagnostic(self, diagnostic: LocalASRProvisioningDiagnostic) -> None:
        self._diagnostics.append(diagnostic)
        if self._diagnostic_sink is None:
            return
        result = self._diagnostic_sink(diagnostic)
        if inspect.isawaitable(result):
            await result


__all__ = [
    "CPUModelInspector",
    "DownloadRuntimeFactory",
    "GPUManifestLoader",
    "GPUModelInspector",
    "LocalASRProvisioningOwner",
    "ManifestLoader",
    "ProvisioningDiagnosticSink",
    "ProvisioningInstaller",
    "ProvisioningStateChanged",
]
