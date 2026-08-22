from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from puripuly_heart.config.paths import user_config_dir
from puripuly_heart.core.local_translation.assets import (
    GEMMA_12B_MODEL_ID,
    GemmaModelSpec,
    e4b_gemma_spec,
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
MANAGED_GEMMA_12B_MODEL_ALIAS = "puripuly-gemma-4-12b-q4"
THREADS_PROFILE_FILENAME = "llama_cpp_threads.json"

# Generalization rule from benchmark (Stage 10):
#   threads       = min(ceil(physical_cores * 0.5), 6)
#   threads_batch = min(ceil(logical_cores * 0.75), 12)
#   draft_threads = 1 (fixed)
THREADS_PHYSICAL_FRACTION = 0.5
THREADS_MAX = 6
THREADS_BATCH_LOGICAL_FRACTION = 0.75
THREADS_BATCH_MAX = 12
DRAFT_THREADS = 1

GemmaBackend = Literal["cpu", "gpu"]
EffectiveGemmaBackend = Literal["cpu", "vulkan"]


@dataclass(frozen=True, slots=True)
class GemmaRuntimePaths:
    cpu_server: Path
    vulkan_server: Path


@dataclass(frozen=True, slots=True)
class LlamaCppThreadProfile:
    threads: int
    threads_batch: int
    draft_threads: int


def _physical_cores() -> int:
    try:
        import psutil
    except ImportError:
        return max(1, os.cpu_count() or 1)
    count = psutil.cpu_count(logical=False)
    if not count:
        return max(1, os.cpu_count() or 1)
    return count


def _logical_cores() -> int:
    return max(1, os.cpu_count() or 1)


def derive_thread_profile(physical_cores: int, logical_cores: int) -> LlamaCppThreadProfile:
    threads = min(math.ceil(physical_cores * THREADS_PHYSICAL_FRACTION), THREADS_MAX)
    threads_batch = min(
        math.ceil(logical_cores * THREADS_BATCH_LOGICAL_FRACTION), THREADS_BATCH_MAX
    )
    return LlamaCppThreadProfile(
        threads=max(1, threads),
        threads_batch=max(1, threads_batch),
        draft_threads=DRAFT_THREADS,
    )


def default_threads_profile_path() -> Path:
    return user_config_dir() / THREADS_PROFILE_FILENAME


def load_or_detect_thread_profile(
    path: Path | None = None,
    *,
    physical_cores: int | None = None,
    logical_cores: int | None = None,
) -> LlamaCppThreadProfile:
    resolved = path or default_threads_profile_path()
    if resolved.is_file():
        try:
            raw = json.loads(resolved.read_text(encoding="utf-8"))
            return LlamaCppThreadProfile(
                threads=int(raw["threads"]),
                threads_batch=int(raw["threads_batch"]),
                draft_threads=int(raw["draft_threads"]),
            )
        except (OSError, KeyError, TypeError, ValueError):
            pass
    detected = derive_thread_profile(
        physical_cores if physical_cores is not None else _physical_cores(),
        logical_cores if logical_cores is not None else _logical_cores(),
    )
    save_thread_profile(detected, resolved)
    return detected


def save_thread_profile(
    profile: LlamaCppThreadProfile,
    path: Path | None = None,
) -> None:
    resolved = path or default_threads_profile_path()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "threads": profile.threads,
        "threads_batch": profile.threads_batch,
        "draft_threads": profile.draft_threads,
    }
    resolved.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


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


def gemma_model_alias(spec: GemmaModelSpec) -> str:
    if spec.model_id == GEMMA_12B_MODEL_ID:
        return MANAGED_GEMMA_12B_MODEL_ALIAS
    return MANAGED_GEMMA_MODEL_ALIAS


def build_gemma_server_command(
    *,
    executable: Path,
    install_dir: Path,
    backend: GemmaBackend,
    port: int,
    vulkan_device: str = "Vulkan0",
    threads_profile: LlamaCppThreadProfile | None = None,
    slot_save_path: Path | None = None,
    spec: GemmaModelSpec | None = None,
) -> tuple[str, ...]:
    if threads_profile is None:
        threads_profile = load_or_detect_thread_profile()
    resolved_spec = spec or e4b_gemma_spec()
    cache_type = "q8_0" if resolved_spec.model_id == GEMMA_12B_MODEL_ID else "f16"
    common = (
        str(executable),
        "--model",
        str(install_dir / resolved_spec.model_filename),
        "--alias",
        gemma_model_alias(resolved_spec),
        "--load-mode",
        "mmap",
        "--ctx-size",
        "3072",
        "--parallel",
        "2",
        "--batch-size",
        "512",
        "--ubatch-size",
        "512",
        "--cache-type-k",
        cache_type,
        "--cache-type-v",
        cache_type,
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
        "--prio",
        "-1",
    )
    if slot_save_path is not None:
        common = common + ("--slot-save-path", str(slot_save_path))
    if backend == "gpu":
        command = common + (
            "--threads",
            "4",
            "--threads-batch",
            "4",
            "--device",
            vulkan_device,
            "--n-gpu-layers",
            "99",
            "--flash-attn",
            "on",
        )
    else:
        command = common + (
            "--threads",
            str(threads_profile.threads),
            "--threads-batch",
            str(threads_profile.threads_batch),
            "--device",
            "none",
            "--n-gpu-layers",
            "0",
            "--flash-attn",
            "off",
        )
        if resolved_spec.draft_filename is not None:
            command = command + (
                "--spec-draft-model",
                str(install_dir / resolved_spec.draft_filename),
                "--spec-type",
                "draft-mtp",
                "--spec-draft-n-max",
                "3",
                "--spec-draft-n-min",
                "2",
                "--spec-draft-p-min",
                "0.0",
                "--spec-draft-device",
                "none",
                "--spec-draft-ngl",
                "0",
                "--spec-draft-threads",
                str(threads_profile.draft_threads),
                "--spec-draft-threads-batch",
                str(threads_profile.draft_threads),
                "--spec-draft-type-k",
                "f16",
                "--spec-draft-type-v",
                "f16",
            )
    if resolved_spec.model_id == GEMMA_12B_MODEL_ID:
        return command + ("--swa-full",)
    return command


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
    "MANAGED_GEMMA_12B_MODEL_ALIAS",
    "MANAGED_GEMMA_MODEL_ALIAS",
    "THREADS_PROFILE_FILENAME",
    "LlamaCppThreadProfile",
    "build_gemma_server_command",
    "gemma_model_alias",
    "default_gemma_runtime_paths",
    "default_llama_runtime_root",
    "default_threads_profile_path",
    "derive_thread_profile",
    "load_or_detect_thread_profile",
    "save_thread_profile",
]
