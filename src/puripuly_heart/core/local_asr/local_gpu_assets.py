from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .local_stt_assets import (
    LOCAL_QWEN_GPU_MODEL_ID,
    LocalSTTAssetManifest,
    LocalSTTInstallState,
    default_local_stt_model_root,
    inspect_local_stt_install_state,
    load_local_stt_asset_manifest,
)

LOCAL_QWEN_GPU_ENGINE = "transcribe.cpp-vulkan"
LOCAL_QWEN_GPU_MODEL_FILENAME = "Qwen3-ASR-1.7B-Q6_K.gguf"
LocalGPUOptInStatus = Literal["not_requested", "missing", "invalid", "ready"]


@dataclass(frozen=True, slots=True)
class LocalGPUInstallSnapshot:
    explicit_opt_in: bool
    status: LocalGPUOptInStatus
    model_id: str = LOCAL_QWEN_GPU_MODEL_ID
    state: LocalSTTInstallState | None = None

    @property
    def activation_allowed(self) -> bool:
        return (
            self.explicit_opt_in
            and self.status == "ready"
            and self.state is not None
            and self.state.installed_manifest is not None
            and self.state.installed_manifest.model_id == self.model_id
        )


def load_local_gpu_asset_manifest() -> LocalSTTAssetManifest:
    manifest = load_local_stt_asset_manifest(LOCAL_QWEN_GPU_MODEL_ID)
    if manifest.engine != LOCAL_QWEN_GPU_ENGINE:
        raise ValueError("local GPU model manifest engine is not strict Vulkan transcribe.cpp")
    return manifest


def inspect_local_gpu_install(
    *,
    explicit_opt_in: bool,
    model_root: Path | None = None,
    verify_checksums: bool = True,
    manifest: LocalSTTAssetManifest | None = None,
) -> LocalGPUInstallSnapshot:
    if not explicit_opt_in:
        return LocalGPUInstallSnapshot(explicit_opt_in=False, status="not_requested")
    resolved_manifest = manifest or load_local_gpu_asset_manifest()
    root = model_root or default_local_stt_model_root()
    state = inspect_local_stt_install_state(
        root / resolved_manifest.install_dirname,
        manifest=resolved_manifest,
        verify_checksums=verify_checksums,
    )
    return LocalGPUInstallSnapshot(
        explicit_opt_in=True,
        status=state.status,
        state=state,
    )


def local_gpu_model_path(
    model_root: Path | None = None,
    *,
    manifest: LocalSTTAssetManifest | None = None,
) -> Path:
    resolved_manifest = manifest or load_local_gpu_asset_manifest()
    root = model_root or default_local_stt_model_root()
    matching = [
        item
        for item in resolved_manifest.files
        if item.relative_path == LOCAL_QWEN_GPU_MODEL_FILENAME
    ]
    if len(matching) != 1:
        raise ValueError("local GPU model manifest must contain exactly one Q6_K model file")
    return root / resolved_manifest.install_dirname / matching[0].relative_path
