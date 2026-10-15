from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from puripuly_heart.config.paths import user_config_dir
from puripuly_heart.core.local_translation.assets import (
    GEMMA_26B_MODEL_ID,
    GemmaModelSpec,
    e4b_gemma_spec,
)
from puripuly_heart.runtime_layout import current_runtime_layout

LLAMA_CPP_BUILD = "b11537"
LLAMA_CPP_COMMIT = "8e2d31e0eb658d0382d3106a61369dff81f0c080"
LLAMA_CPP_CPU_ARCHIVE = "llama-b11537-bin-win-cpu-x64.zip"
LLAMA_CPP_CPU_ARCHIVE_SIZE = 19_513_247
LLAMA_CPP_CPU_ARCHIVE_SHA256 = "a05c2d35ebe9787489277d6e039b391fd006384a6c49d069395f3a7b47186fd2"
LLAMA_CPP_VULKAN_ARCHIVE = "llama-b11537-bin-win-vulkan-x64.zip"
LLAMA_CPP_VULKAN_ARCHIVE_SIZE = 33_458_882
LLAMA_CPP_VULKAN_ARCHIVE_SHA256 = "ae34d6a850c5d362cdfc0ef8f88f31fc499dab6df8853cec7887a32891aa5bbe"
LLAMA_CPP_RUNTIME_DIRNAME = "llama.cpp-b11537"
LLAMA_CPP_SOURCE_RELATIVE_DIR = Path("llama.cpp") / LLAMA_CPP_RUNTIME_DIRNAME
MANAGED_GEMMA_MODEL_ALIAS = "puripuly-gemma-4-e4b-q4"
MANAGED_GEMMA_26B_MODEL_ALIAS = "puripuly-gemma-4-26b-a4b-q4"
GEMMA_RUNTIME_PROFILE_VERSION = "managed-gemma-v2"
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
        except OSError, KeyError, TypeError, ValueError:
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
    layout = current_runtime_layout()
    if layout.host_kind == "source":
        return layout.native(*LLAMA_CPP_SOURCE_RELATIVE_DIR.parts)
    return layout.native("_runtime", LLAMA_CPP_RUNTIME_DIRNAME)


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
    threads_profile: LlamaCppThreadProfile | None = None,
    slot_save_path: Path | None = None,
    spec: GemmaModelSpec | None = None,
) -> tuple[str, ...]:
    resolved_spec = spec or e4b_gemma_spec()
    is_26b = resolved_spec.model_id == GEMMA_26B_MODEL_ID
    if backend not in {"cpu", "gpu"}:
        raise ValueError("managed Gemma backend must be cpu or gpu")
    if is_26b and backend != "gpu":
        raise ValueError("Gemma 4 26B requires a working Vulkan GPU; select E4B for CPU inference")
    if backend == "cpu" and threads_profile is None:
        threads_profile = load_or_detect_thread_profile()
    common = (
        str(executable),
        "--model",
        str(install_dir / resolved_spec.model_filename),
        "--alias",
        MANAGED_GEMMA_26B_MODEL_ALIAS if is_26b else MANAGED_GEMMA_MODEL_ALIAS,
        "--load-mode",
        "none" if is_26b else "mmap",
        "--ctx-size",
        "4608",
        "--parallel",
        "3",
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
        if is_26b:
            command += (
                "--n-cpu-moe",
                "24",
                "--moe-cache-mib",
                "2048",
                "--swa-full",
                "--cache-ram",
                "0",
                "--fit",
                "off",
            )
    else:
        assert threads_profile is not None
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
    return command


__all__ = [
    "GEMMA_RUNTIME_PROFILE_VERSION",
    "MANAGED_GEMMA_26B_MODEL_ALIAS",
    "EffectiveGemmaBackend",
    "GemmaBackend",
    "GemmaRuntimePaths",
    "LLAMA_CPP_BUILD",
    "LLAMA_CPP_COMMIT",
    "LLAMA_CPP_CPU_ARCHIVE",
    "LLAMA_CPP_CPU_ARCHIVE_SHA256",
    "LLAMA_CPP_CPU_ARCHIVE_SIZE",
    "LLAMA_CPP_RUNTIME_DIRNAME",
    "LLAMA_CPP_SOURCE_RELATIVE_DIR",
    "LLAMA_CPP_VULKAN_ARCHIVE",
    "LLAMA_CPP_VULKAN_ARCHIVE_SHA256",
    "LLAMA_CPP_VULKAN_ARCHIVE_SIZE",
    "MANAGED_GEMMA_MODEL_ALIAS",
    "THREADS_PROFILE_FILENAME",
    "LlamaCppThreadProfile",
    "build_gemma_server_command",
    "default_gemma_runtime_paths",
    "default_llama_runtime_root",
    "default_threads_profile_path",
    "derive_thread_profile",
    "load_or_detect_thread_profile",
    "save_thread_profile",
]
