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

from puripuly_heart.runtime_layout import current_runtime_layout

PACKAGED_SMOKE_SCHEMA = "puripuly-heart/process-capture-packaged-smoke/v2"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inside(path: Path, root: Path) -> bool:
    return path.is_relative_to(root)


def _loaded_artifact(module: object, artifact_root: Path) -> dict[str, str]:
    origin = Path(str(getattr(module, "__file__", ""))).resolve()
    if not _inside(origin, artifact_root):
        raise RuntimeError("runtime module originated outside artifact root")
    artifact = origin if origin.is_file() else artifact_root / "app" / "python.zip"
    if not artifact.is_file():
        raise RuntimeError("runtime module artifact is unavailable")
    return {"origin": str(origin), "artifact": str(artifact), "sha256": _sha256(artifact)}


def run_smoke(*, artifact_root: Path, report_path: Path) -> int:
    artifact_root = artifact_root.resolve()
    runtime_executable = Path(sys.executable).resolve()
    capture = None
    try:
        layout = current_runtime_layout()
        if (
            layout.host_kind != "native"
            or layout.native_runtime_root != artifact_root
            or runtime_executable != artifact_root / "python.exe"
        ):
            raise RuntimeError("capture smoke requires the intended native Python payload")
        platform_module = importlib.import_module("puripuly_heart.config.process_capture_platform")
        process_source = importlib.import_module("puripuly_heart.core.audio.process_source")
        availability = platform_module.get_process_capture_platform_availability()
        if not availability.available:
            return 2
        artifact_native_modules = tuple(
            (artifact_root / "site-packages" / "proctap").glob("_native*.pyd")
        )
        if len(artifact_native_modules) != 1:
            raise RuntimeError("native payload must contain exactly one ProcTap extension")
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
        capture = process_source.ProcTapProcessAudioCaptureFactory().create(
            pid=os.getpid(), on_data=lambda _data, _frames: None
        )
        native_process_specific = process_source.verify_proctap_process_specific(capture)
        if native_process_specific is not True:
            raise RuntimeError("capture is not process-specific")
        capture.start()
        time.sleep(0.1)
        capture.close()
        capture = None
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
            "runtime_executable": str(runtime_executable),
            "runtime_executable_sha256": _sha256(runtime_executable),
            "native_process_specific": True,
            "capture_started": True,
            "capture_closed": True,
            "device_fallback_used": False,
            "credentials_used": False,
            "network_used": False,
        }
        result = 0
    except Exception as exc:
        report = {
            "schema": PACKAGED_SMOKE_SCHEMA,
            "status": "failed",
            "failure_type": type(exc).__name__,
        }
        result = 1
    finally:
        if capture is not None:
            capture.close()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="native-process-capture-smoke")
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run_smoke(artifact_root=args.artifact_root.resolve(), report_path=args.report.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
