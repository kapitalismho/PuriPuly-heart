from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import psutil


def _git(repository_root: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=repository_root, text=True, encoding="utf-8"
    ).strip()


def run_provenance(
    repository_root: Path,
    requested_argv: tuple[str, ...],
    *,
    deterministic_seed: int,
    deterministic_kernels: bool,
) -> dict[str, Any]:
    status = _git(repository_root, "status", "--porcelain=v1", "--untracked-files=all")
    return {
        "git_commit": _git(repository_root, "rev-parse", "HEAD"),
        "git_branch": _git(repository_root, "branch", "--show-current"),
        "git_dirty": bool(status),
        "git_status_porcelain": status.splitlines() if status else [],
        "cwd": str(Path.cwd().resolve()),
        "requested_argv": list(requested_argv),
        "worker_argv": [str(value) for value in sys.argv],
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": sys.version.split()[0],
        "host": {
            "node": platform.node(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "processor_identifier": os.environ.get("PROCESSOR_IDENTIFIER"),
            "logical_cpu_count": psutil.cpu_count(logical=True),
            "physical_cpu_count": psutil.cpu_count(logical=False),
            "total_ram_bytes": int(psutil.virtual_memory().total),
        },
        "runtime_controls": {
            "deterministic_seed": deterministic_seed,
            "deterministic_kernels": deterministic_kernels,
            "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
            "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
            "numexpr_num_threads": os.environ.get("NUMEXPR_NUM_THREADS"),
        },
    }
