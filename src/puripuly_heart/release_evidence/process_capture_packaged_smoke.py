from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import sys
import time
from pathlib import Path
from typing import Sequence

PACKAGED_SMOKE_SCHEMA = "puripuly-heart/process-capture-packaged-smoke/v1"
HELPER_EXE_NAME = "PuriPulyHeartProcessCaptureSmoke.exe"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _loaded_artifact(module: object, artifact_root: Path) -> dict[str, str]:
    origin = Path(str(getattr(module, "__file__", ""))).resolve()
    artifact = origin if origin.is_file() else Path(sys.executable).resolve()
    if not _inside(origin, artifact_root) or not _inside(artifact, artifact_root):
        raise RuntimeError("runtime module originated outside artifact root")
    return {
        "origin": str(origin),
        "artifact": str(artifact),
        "sha256": _sha256(artifact),
    }


def run_smoke(*, artifact_root: Path, report_path: Path) -> int:
    artifact_root = artifact_root.resolve()
    helper_path = Path(sys.executable).resolve()
    if not _inside(helper_path, artifact_root):
        return 1
    platform_module = importlib.import_module("puripuly_heart.config.process_capture_platform")
    process_source = importlib.import_module("puripuly_heart.core.audio.process_source")
    availability = platform_module.get_process_capture_platform_availability()
    if not availability.available:
        return 2
    artifact_native_modules = tuple((artifact_root / "proctap").glob("_native*.pyd"))
    if len(artifact_native_modules) != 1:
        return 1
    capture = None
    try:
        proctap = importlib.import_module("proctap")
        runtime_native = importlib.import_module("proctap._native")
        runtime_native_path = Path(runtime_native.__file__).resolve()
        artifact_native_path = artifact_native_modules[0].resolve()
        loaded_modules = {
            "platform": _loaded_artifact(platform_module, artifact_root),
            "process_source": _loaded_artifact(process_source, artifact_root),
            "proctap": _loaded_artifact(proctap, artifact_root),
            "proctap_native": _loaded_artifact(runtime_native, artifact_root),
        }
        if runtime_native_path != artifact_native_path:
            raise RuntimeError("runtime native does not equal packaged native")
        runtime_native_hash = _sha256(runtime_native_path)
        artifact_native_hash = _sha256(artifact_native_path)
        if runtime_native_hash != artifact_native_hash:
            raise RuntimeError("runtime native hash does not equal packaged native")
        capture = process_source.ProcTapProcessAudioCaptureFactory().create(
            pid=os.getpid(),
            on_data=lambda _data, _frames: None,
        )
        native_process_specific = process_source.verify_proctap_process_specific(capture)
        capture.start()
        time.sleep(0.1)
        report = {
            "schema": PACKAGED_SMOKE_SCHEMA,
            "status": "passed",
            "proctap_version": importlib.metadata.version("proc-tap"),
            "proctap_module": str(Path(proctap.__file__).resolve()),
            "runtime_native_module": str(runtime_native_path),
            "runtime_native_sha256": runtime_native_hash,
            "artifact_native_module": str(artifact_native_path),
            "artifact_native_sha256": artifact_native_hash,
            "runtime_loaded_modules": loaded_modules,
            "helper_executable": str(helper_path),
            "helper_executable_sha256": _sha256(helper_path),
            "native_process_specific": native_process_specific,
            "capture_started": True,
            "device_fallback_used": False,
            "credentials_used": False,
            "network_used": False,
            "release_only_helper": True,
        }
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return 0
    except Exception as exc:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(
                {
                    "schema": PACKAGED_SMOKE_SCHEMA,
                    "status": "failed",
                    "failure_type": type(exc).__name__,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return 1
    finally:
        if capture is not None:
            capture.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="PuriPulyHeartProcessCaptureSmoke")
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run_smoke(
        artifact_root=args.artifact_root.resolve(),
        report_path=args.report.resolve(),
    )


if __name__ == "__main__":
    raise SystemExit(main())
