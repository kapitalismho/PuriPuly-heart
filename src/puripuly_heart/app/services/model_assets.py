from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from puripuly_heart.app.services.model_assets_catalog import ModelAssetChoice, model_asset_choices
from puripuly_heart.core.diagnostics import serialize_diagnostics
from puripuly_heart.core.feature_diagnostics import log_feature
from puripuly_heart.core.local_asr.local_asr_provisioning import LocalASRInstallRequest
from puripuly_heart.core.local_asr.local_stt_assets import (
    LOCAL_QWEN_GPU_MODEL_ID,
    default_local_stt_model_root,
    inspect_local_stt_install_state,
    load_local_stt_asset_manifest,
)
from puripuly_heart.core.model_download_sources import MODEL_DOWNLOAD_SOURCES
from puripuly_heart.core.messages import ErrorDiagnostics
from puripuly_heart.core.model_asset_diagnostics import (
    model_asset_failure_code,
    model_asset_failure_diagnostics,
)

MODEL_ASSETS_ORIGIN = "model_assets"
_INTEGRITY_STATUS = {"ready": "installed", "missing": "missing", "invalid": "invalid"}


class ModelAssetsError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        failure_code: str,
        failure_diagnostics: ErrorDiagnostics | None = None,
    ) -> None:
        super().__init__(message)
        self.failure_code = failure_code
        self.failure_diagnostics = (
            failure_diagnostics
            if failure_diagnostics is not None
            else model_asset_failure_diagnostics(self)
        )
        self.secondary_diagnostics: tuple[ErrorDiagnostics, ...] = ()


@dataclass(frozen=True, slots=True)
class ModelAssetEntry:
    model_id: str
    name: str
    description_key: str
    purpose: str
    size_bytes: int
    status: str
    progress_percent: int | None = None
    failure_code: str | None = None
    importable: bool = False
    required_files: tuple[str, ...] = ()
    failure_diagnostics: ErrorDiagnostics | None = None
    recovery_diagnostics: tuple[ErrorDiagnostics, ...] = ()
    secondary_diagnostics: tuple[ErrorDiagnostics, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "name": self.name,
            "purpose": self.purpose,
            "size_bytes": self.size_bytes,
            "status": self.status,
            "progress_percent": self.progress_percent,
            "failure_code": self.failure_code,
            "importable": self.importable,
            "required_files": list(self.required_files),
            "failure_diagnostics": (
                serialize_diagnostics(self.failure_diagnostics)
                if self.failure_diagnostics is not None
                else None
            ),
            "recovery_diagnostics": [
                serialize_diagnostics(value) for value in self.recovery_diagnostics
            ],
            "secondary_diagnostics": [
                serialize_diagnostics(value) for value in self.secondary_diagnostics
            ],
        }


@dataclass(slots=True)
class _Operation:
    model_id: str
    kind: str
    progress_percent: int | None = None


def _failure_code(error: BaseException) -> str:
    return model_asset_failure_code(error)


@dataclass(slots=True)
class ModelAssetsApplication:
    provisioning: Callable[[], object | None]
    gemma: Callable[[], object | None]
    locale: Callable[[], str | None]
    active_model_ids: Callable[[], frozenset[str]] = field(default=lambda: frozenset())
    model_root: Path | None = None
    choices: tuple[ModelAssetChoice, ...] = field(default_factory=model_asset_choices)
    _integrity: dict[str, str] = field(init=False, default_factory=dict)
    _failures: dict[str, str] = field(init=False, default_factory=dict)
    _failure_diagnostics: dict[str, ErrorDiagnostics] = field(init=False, default_factory=dict)
    _recoveries: dict[str, tuple[ErrorDiagnostics, ...]] = field(init=False, default_factory=dict)
    _secondaries: dict[str, tuple[ErrorDiagnostics, ...]] = field(init=False, default_factory=dict)
    _inspection_diagnostics: dict[str, ErrorDiagnostics] = field(init=False, default_factory=dict)
    _operation: _Operation | None = field(init=False, default=None)
    _cancel_requested: bool = field(init=False, default=False)

    def entries(self) -> tuple[ModelAssetEntry, ...]:
        entries: list[ModelAssetEntry] = []
        for choice in self.choices:
            status = self._integrity.get(choice.model_id, "missing")
            progress = None
            operation = self._operation
            if operation is not None and operation.model_id == choice.model_id:
                status = {"delete": "deleting", "import": "importing"}.get(
                    operation.kind, "downloading"
                )
                progress = operation.progress_percent
            entries.append(
                ModelAssetEntry(
                    model_id=choice.model_id,
                    name=choice.name,
                    description_key=choice.description_key,
                    purpose=choice.purpose,
                    size_bytes=choice.size_bytes,
                    status=status,
                    progress_percent=progress,
                    failure_code=self._failures.get(choice.model_id),
                    importable=choice.importable,
                    required_files=choice.required_files,
                    failure_diagnostics=self._failure_diagnostics.get(choice.model_id),
                    recovery_diagnostics=self._recoveries.get(choice.model_id, ()),
                    secondary_diagnostics=self._secondaries.get(choice.model_id, ()) + (
                        (self._inspection_diagnostics[choice.model_id],)
                        if choice.model_id in self._inspection_diagnostics
                        else ()
                    ),
                )
            )
        return tuple(entries)

    def entry(self, model_id: str) -> ModelAssetEntry:
        return next(item for item in self.entries() if item.model_id == model_id)

    @property
    def busy(self) -> bool:
        return self._operation is not None

    async def refresh(self) -> tuple[ModelAssetEntry, ...]:
        inspections = await asyncio.to_thread(self._inspect_all)
        for model_id, (status, diagnostics) in inspections.items():
            self._integrity[model_id] = status
            if diagnostics is not None:
                self._inspection_diagnostics[model_id] = diagnostics
            else:
                self._inspection_diagnostics.pop(model_id, None)
        return self.entries()

    async def download(self, model_id: str, *, source: str | None = None) -> ModelAssetEntry:
        choice = self._choice(model_id)
        if source is not None and source not in MODEL_DOWNLOAD_SOURCES:
            raise ModelAssetsError("Unsupported model source", failure_code="unsupported_source")
        return await self._run(choice, "download", source=source)

    async def import_directory(self, model_id: str, directory: Path) -> ModelAssetEntry:
        choice = self._choice(model_id)
        if not choice.importable:
            raise ModelAssetsError(
                "Model does not support import", failure_code="unsupported_source"
            )
        files = tuple(directory / name for name in choice.required_files)
        missing = [path.name for path in files if not path.is_file()]
        if missing:
            self._failures[model_id] = "missing_file"
            log_feature("ModelDownload", "preflight", "missing_file")
            error = ModelAssetsError(
                "Missing model files: " + ", ".join(missing), failure_code="missing_file"
            )
            self._failure_diagnostics[model_id] = error.failure_diagnostics
            raise error
        return await self._run(choice, "import", source="local", files=files)

    async def cancel(self, model_id: str) -> bool:
        operation = self._operation
        if operation is None or operation.model_id != model_id or operation.kind == "delete":
            return False
        self._cancel_requested = True
        choice = self._choice(model_id)
        if choice.purpose == "translation":
            owner = self.gemma()
            if owner is not None:
                owner.cancel_assets(model_id)
        else:
            owner = self.provisioning()
            if owner is not None:
                await owner.cancel_install(self._backend(model_id), origin=MODEL_ASSETS_ORIGIN)
        return True

    async def delete(self, model_id: str) -> ModelAssetEntry:
        choice = self._choice(model_id)
        if model_id in self.active_model_ids():
            self._failures[model_id] = "model_in_use"
            error = ModelAssetsError("Model is in use", failure_code="model_in_use")
            self._failure_diagnostics[model_id] = error.failure_diagnostics
            raise error
        return await self._run(choice, "delete")

    async def _run(
        self,
        choice: ModelAssetChoice,
        kind: str,
        *,
        source: str | None = None,
        files: tuple[Path, ...] = (),
    ) -> ModelAssetEntry:
        if self._operation is not None:
            raise ModelAssetsError("Another model task is active", failure_code="task_busy")
        self._operation = _Operation(choice.model_id, kind, 0 if kind != "delete" else None)
        self._cancel_requested = False
        self._failures.pop(choice.model_id, None)
        self._failure_diagnostics.pop(choice.model_id, None)
        self._recoveries.pop(choice.model_id, None)
        self._secondaries.pop(choice.model_id, None)
        self._inspection_diagnostics.pop(choice.model_id, None)
        failure: ModelAssetsError | None = None
        try:
            if kind == "delete":
                await self._delete(choice)
            elif choice.purpose == "translation":
                await self._install_gemma(choice, source=source, files=files)
            else:
                await self._install_asr(choice, source=source)
        except ModelAssetsError as error:
            failure = error
            error.secondary_diagnostics = self._secondaries.get(choice.model_id, ())
            if error.failure_code != "cancelled":
                self._failures[choice.model_id] = error.failure_code
                self._failure_diagnostics[choice.model_id] = error.failure_diagnostics
            raise
        except Exception as error:
            code = "cancelled" if self._cancel_requested else _failure_code(error)
            if code != "cancelled":
                self._failures[choice.model_id] = code
            failure = ModelAssetsError(
                "Model task cancelled" if code == "cancelled" else "Model asset operation failed",
                failure_code=code,
                failure_diagnostics=model_asset_failure_diagnostics(
                    error,
                    operation="remove" if kind == "delete" else "gemma_install",
                    code=(
                        "model_assets.remove_failed"
                        if kind == "delete"
                        else "model_assets.install_failed"
                    ),
                    fields={"model_id": choice.model_id},
                ),
            )
            failure.secondary_diagnostics = getattr(error, "secondary_diagnostics", ())
            self._secondaries[choice.model_id] = failure.secondary_diagnostics
            if code != "cancelled":
                self._failure_diagnostics[choice.model_id] = failure.failure_diagnostics
                log_feature("ModelDownload", kind, code, error=error)
            raise failure from error
        finally:
            self._operation = None
            self._cancel_requested = False
            try:
                status, diagnostics = await asyncio.to_thread(self._inspect_one, choice)
                self._integrity[choice.model_id] = status
                if diagnostics is not None:
                    self._inspection_diagnostics[choice.model_id] = diagnostics
                    if failure is not None:
                        failure.secondary_diagnostics += (diagnostics,)
            except Exception as error:
                diagnostics = model_asset_failure_diagnostics(
                    error,
                    operation="inspection",
                    code="model_assets.inspect_failed",
                    fields={"model_id": choice.model_id},
                )
                self._inspection_diagnostics[choice.model_id] = diagnostics
                if failure is not None:
                    failure.secondary_diagnostics += (diagnostics,)
                log_feature(
                    "ModelDownload",
                    "inspection",
                    "inspect_failed",
                    error=error,
                    level=logging.WARNING,
                )
        return self.entry(choice.model_id)

    async def _install_asr(self, choice: ModelAssetChoice, *, source: str | None) -> None:
        owner = self.provisioning()
        if owner is None:
            raise ModelAssetsError(
                "Model owner unavailable", failure_code="model_owner_unavailable"
            )
        backend = self._backend(choice.model_id)
        if owner.snapshot.activity_for(backend) is not None:
            raise ModelAssetsError("Another model task is active", failure_code="task_busy")

        def on_status(update) -> None:
            percent = getattr(update, "percent", None)
            if self._operation is not None and isinstance(percent, int):
                self._operation.progress_percent = percent

        result = await owner.start_install(
            LocalASRInstallRequest(
                backend=backend,
                model_ids=(choice.model_id,),
                locale=self.locale(),
                origin=MODEL_ASSETS_ORIGIN,
                explicit_gpu_intent=backend == "gpu",
                preferred_source=source,
            ),
            on_status=on_status,
        )
        self._secondaries[choice.model_id] = result.secondary_diagnostics
        if result.cancelled:
            raise ModelAssetsError(
                "Model task cancelled",
                failure_code="cancelled",
                failure_diagnostics=result.failure_diagnostics,
            )
        if result.failed_model_ids:
            code = result.failure_codes[0] if result.failure_codes else "unknown"
            raise ModelAssetsError(
                "Model installation failed",
                failure_code=code,
                failure_diagnostics=result.failure_diagnostics,
            )
        self._recoveries[choice.model_id] = result.recovery_diagnostics

    async def _install_gemma(
        self,
        choice: ModelAssetChoice,
        *,
        source: str | None,
        files: tuple[Path, ...],
    ) -> None:
        owner = self.gemma()
        if owner is None:
            raise ModelAssetsError(
                "Model owner unavailable", failure_code="model_owner_unavailable"
            )

        def on_status(update) -> None:
            if self._operation is None:
                return
            total = getattr(update, "total_bytes", 0) or 0
            done = getattr(update, "downloaded_bytes", 0) or 0
            if total:
                self._operation.progress_percent = min(100, int(done * 100 / total))

        result = await owner.install_assets(
            model_id=choice.model_id,
            source=source,
            files=files,
            on_status=on_status,
            locale=self.locale(),
        )
        self._recoveries[choice.model_id] = result.recovery_diagnostics
        self._secondaries[choice.model_id] = result.secondary_diagnostics

    async def _delete(self, choice: ModelAssetChoice) -> None:
        if choice.purpose == "translation":
            owner = self.gemma()
            if owner is None:
                raise ModelAssetsError(
                    "Model owner unavailable", failure_code="model_owner_unavailable"
                )
            await owner.delete_assets(choice.model_id)
            return
        owner = self.provisioning()
        if owner is None:
            raise ModelAssetsError(
                "Model owner unavailable", failure_code="model_owner_unavailable"
            )
        await owner.delete_model(choice.model_id)

    def _inspect_all(self) -> dict[str, tuple[str, ErrorDiagnostics | None]]:
        return {choice.model_id: self._inspect_one(choice) for choice in self.choices}

    def _inspect_one(self, choice: ModelAssetChoice) -> tuple[str, ErrorDiagnostics | None]:
        try:
            if choice.purpose == "translation":
                owner = self.gemma()
                if owner is None:
                    return "missing", None
                state = owner.install_state(choice.model_id)
            else:
                manifest = load_local_stt_asset_manifest(choice.model_id)
                root = self.model_root or default_local_stt_model_root()
                state = inspect_local_stt_install_state(
                    root / manifest.install_dirname, manifest=manifest
                )
            return _INTEGRITY_STATUS.get(state.status, "invalid"), state.failure_diagnostics
        except Exception as error:
            return "invalid", model_asset_failure_diagnostics(
                error,
                operation="inspection",
                code="model_assets.inspect_failed",
                fields={"model_id": choice.model_id},
            )

    def _choice(self, model_id: str) -> ModelAssetChoice:
        for choice in self.choices:
            if choice.model_id == model_id:
                return choice
        raise ModelAssetsError("Unknown model", failure_code="unknown_model")

    @staticmethod
    def _backend(model_id: str) -> str:
        return "gpu" if model_id == LOCAL_QWEN_GPU_MODEL_ID else "cpu"


__all__ = [
    "MODEL_ASSETS_ORIGIN",
    "ModelAssetEntry",
    "ModelAssetsApplication",
    "ModelAssetsError",
]
