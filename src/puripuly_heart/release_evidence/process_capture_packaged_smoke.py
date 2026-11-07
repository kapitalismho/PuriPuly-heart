from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Sequence

from puripuly_heart.runtime_layout import current_runtime_layout

PACKAGED_SMOKE_SCHEMA = "puripuly-heart/process-capture-packaged-smoke/v3"


def _loaded_module_path(module: object, artifact_root: Path) -> str:
    origin = Path(str(getattr(module, "__file__", ""))).resolve()
    if not origin.is_relative_to(artifact_root):
        raise RuntimeError(f"runtime module originated outside artifact root: {origin}")
    return str(origin)


def _failure(exc: Exception) -> dict[str, str]:
    return {
        "type": type(exc).__name__,
        "message": str(exc),
        "traceback": "".join(traceback.format_exception(exc)),
    }


def run_smoke(*, artifact_root: Path, report_path: Path) -> int:
    report: dict[str, object] = {
        "schema": PACKAGED_SMOKE_SCHEMA,
        "status": "failed",
        "stage": "runtime_layout",
        "artifact_root": str(artifact_root),
        "runtime_executable": sys.executable,
        "runtime_native_module": None,
        "runtime_loaded_modules": {},
        "proctap_version": None,
        "native_process_specific": False,
        "capture_started": False,
        "capture_closed": False,
        "unavailable_reason": None,
        "failure": None,
        "cleanup_failure": None,
    }
    capture = None
    result = 1
    try:
        artifact_root = artifact_root.resolve()
        runtime_executable = Path(sys.executable).resolve()
        report["artifact_root"] = str(artifact_root)
        report["runtime_executable"] = str(runtime_executable)
        layout = current_runtime_layout()
        if (
            layout.host_kind != "native"
            or layout.native_runtime_root != artifact_root
            or layout.app_resource_root != artifact_root / "app"
            or runtime_executable != artifact_root / "python.exe"
        ):
            raise RuntimeError("capture smoke requires the intended native Python payload")
        report["stage"] = "platform_availability"
        platform_module = importlib.import_module("puripuly_heart.config.process_capture_platform")
        availability = platform_module.get_process_capture_platform_availability()
        if not availability.available:
            report["status"] = "unavailable"
            report["unavailable_reason"] = availability.reason
            result = 2
        else:
            report["stage"] = "packaged_modules"
            artifact_native_modules = tuple(
                (artifact_root / "site-packages" / "proctap").glob("_native*.pyd")
            )
            if len(artifact_native_modules) != 1:
                raise RuntimeError("native payload must contain exactly one ProcTap extension")
            proctap = importlib.import_module("proctap")
            runtime_native = importlib.import_module("proctap._native")
            process_source = importlib.import_module("puripuly_heart.core.audio.process_source")
            report["runtime_loaded_modules"] = {
                name: _loaded_module_path(module, artifact_root)
                for name, module in (
                    ("platform", platform_module),
                    ("process_source", process_source),
                    ("proctap", proctap),
                    ("proctap_native", runtime_native),
                )
            }
            runtime_native_path = Path(runtime_native.__file__).resolve()
            report["runtime_native_module"] = str(runtime_native_path)
            if runtime_native_path != artifact_native_modules[0].resolve():
                raise RuntimeError("runtime native does not equal packaged native")
            report["proctap_version"] = importlib.metadata.version("proc-tap")
            report["stage"] = "capture_create"
            capture = process_source.ProcTapProcessAudioCaptureFactory().create(
                pid=os.getpid(), on_data=lambda _data, _frames: None
            )
            report["stage"] = "capture_verify"
            if process_source.verify_proctap_process_specific(capture) is not True:
                raise RuntimeError("capture is not process-specific")
            report["native_process_specific"] = True
            report["stage"] = "capture_start"
            capture.start()
            report["capture_started"] = True
            report["stage"] = "capture_run"
            time.sleep(0.1)
            result = 0
    except Exception as exc:
        report["failure"] = _failure(exc)
    finally:
        if capture is not None:
            if result == 0:
                report["stage"] = "capture_close"
            try:
                capture.close()
                report["capture_closed"] = True
            except Exception as exc:
                if report["failure"] is None:
                    report["failure"] = _failure(exc)
                else:
                    report["cleanup_failure"] = _failure(exc)
                result = 1
    if result == 0:
        report["status"] = "passed"
        report["stage"] = "complete"
    serialized = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if result != 0:
        print(serialized, file=sys.stderr, end="")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(serialized, encoding="utf-8")
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
