from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from puripuly_heart.core.local_translation.assets import (
    GEMMA_DRAFT_FILENAME,
    GEMMA_MODEL_FILENAME,
)

LLAMA_CPP_BUILD = "b10423"
LLAMA_CPP_COMMIT = "a94d563ed801d1da1b8c2432946de07d0231bb3d"
LLAMA_CPP_CPU_ARCHIVE = "llama-b10423-bin-win-cpu-x64.zip"
LLAMA_CPP_CPU_ARCHIVE_SIZE = 18_456_396
LLAMA_CPP_CPU_ARCHIVE_SHA256 = "b5a396f113a344578c0766331704bd541fd743c4c8e92858bea18440ee0ab19a"
LLAMA_CPP_VULKAN_ARCHIVE = "llama-b10423-bin-win-vulkan-x64.zip"
LLAMA_CPP_VULKAN_ARCHIVE_SIZE = 34_563_676
LLAMA_CPP_VULKAN_ARCHIVE_SHA256 = "510447fb021c80a264b2181c885b5f2ce9cc5b66c65d447cd1f9ce7ba81dc222"
LLAMA_CPP_RUNTIME_DIRNAME = "llama.cpp-b10423"
MANAGED_GEMMA_MODEL_ALIAS = "puripuly-gemma-4-e4b-q4"

GemmaBackend = Literal["cpu", "gpu"]
EffectiveGemmaBackend = Literal["cpu", "vulkan"]


@dataclass(frozen=True, slots=True)
class GemmaRuntimePaths:
    cpu_server: Path
    vulkan_server: Path


def default_llama_runtime_root() -> Path:
    configured = os.getenv("PURIPULY_HEART_LLAMA_CPP_ROOT")
    if configured:
        return Path(configured).resolve()
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS) / "_runtime" / LLAMA_CPP_RUNTIME_DIRNAME
    return Path.cwd() / "build" / "llama.cpp" / LLAMA_CPP_RUNTIME_DIRNAME


def default_gemma_runtime_paths(root: Path | None = None) -> GemmaRuntimePaths:
    resolved = (root or default_llama_runtime_root()).resolve()
    return GemmaRuntimePaths(
        cpu_server=resolved / "cpu" / "llama-server.exe",
        vulkan_server=resolved / "vulkan" / "llama-server.exe",
    )


def build_gemma_server_command(
    *,
    executable: Path,
    install_dir: Path,
    backend: GemmaBackend,
    port: int,
    vulkan_device: str = "Vulkan0",
) -> tuple[str, ...]:
    common = (
        str(executable),
        "--model",
        str(install_dir / GEMMA_MODEL_FILENAME),
        "--alias",
        MANAGED_GEMMA_MODEL_ALIAS,
        "--load-mode",
        "mmap",
        "--threads",
        "4",
        "--threads-batch",
        "4",
        "--ctx-size",
        "4096",
        "--parallel",
        "1",
        "--batch-size",
        "512",
        "--ubatch-size",
        "512",
        "--cache-type-k",
        "f16",
        "--cache-type-v",
        "f16",
        "--cache-prompt",
        "--reasoning",
        "off",
        "--reasoning-budget",
        "0",
        "--warmup",
        "--perf",
        "--metrics",
        "--no-webui",
        "--threads-http",
        "1",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    )
    if backend == "gpu":
        return common + (
            "--device",
            vulkan_device,
            "--n-gpu-layers",
            "99",
            "--flash-attn",
            "on",
        )
    return common + (
        "--device",
        "none",
        "--n-gpu-layers",
        "0",
        "--spec-draft-model",
        str(install_dir / GEMMA_DRAFT_FILENAME),
        "--spec-type",
        "draft-mtp",
        "--spec-draft-n-max",
        "4",
        "--spec-draft-n-min",
        "1",
        "--spec-draft-device",
        "none",
        "--spec-draft-ngl",
        "0",
        "--spec-draft-threads",
        "4",
        "--spec-draft-threads-batch",
        "4",
        "--spec-draft-type-k",
        "f16",
        "--spec-draft-type-v",
        "f16",
        "--flash-attn",
        "auto",
    )


__all__ = [
    "EffectiveGemmaBackend",
    "GemmaBackend",
    "GemmaRuntimePaths",
    "LLAMA_CPP_BUILD",
    "LLAMA_CPP_COMMIT",
    "LLAMA_CPP_CPU_ARCHIVE",
    "LLAMA_CPP_CPU_ARCHIVE_SHA256",
    "LLAMA_CPP_CPU_ARCHIVE_SIZE",
    "LLAMA_CPP_RUNTIME_DIRNAME",
    "LLAMA_CPP_VULKAN_ARCHIVE",
    "LLAMA_CPP_VULKAN_ARCHIVE_SHA256",
    "LLAMA_CPP_VULKAN_ARCHIVE_SIZE",
    "MANAGED_GEMMA_MODEL_ALIAS",
    "build_gemma_server_command",
    "default_gemma_runtime_paths",
    "default_llama_runtime_root",
]
