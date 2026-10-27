from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from puripuly_heart.config.process_capture_platform import ProcessCapturePlatformAvailability
from puripuly_heart.release_evidence import process_capture_packaged_smoke as smoke
from puripuly_heart.release_evidence.windows_process_distribution import validate_runtime_report


class _Capture:
    def __init__(self) -> None:
        self.events: list[str] = []
        self.start_failure: Exception | None = None
        self.close_failure: Exception | None = None

    def start(self) -> None:
        self.events.append("start")
        if self.start_failure is not None:
            raise self.start_failure

    def close(self) -> None:
        self.events.append("close")
        if self.close_failure is not None:
            raise self.close_failure


@pytest.fixture
def packaged_smoke(tmp_path: Path, monkeypatch) -> SimpleNamespace:
    artifact_root = tmp_path / "설치 with spaces"
    artifact_native = artifact_root / "site-packages" / "proctap" / "_native.cp314-win_amd64.pyd"
    archive = artifact_root / "app" / "python.zip"
    runtime = artifact_root / "python.exe"
    artifact_native.parent.mkdir(parents=True)
    archive.parent.mkdir()
    artifact_native.write_bytes(b"packaged-native")
    archive.write_bytes(b"collected-modules")
    runtime.write_bytes(b"python")
    capture = _Capture()
    platform_module = SimpleNamespace(
        __file__=archive / "puripuly_heart" / "config" / "process_capture_platform.pyc",
        get_process_capture_platform_availability=lambda: ProcessCapturePlatformAvailability(
            available=True
        ),
    )
    source_module = SimpleNamespace(
        __file__=archive / "puripuly_heart" / "core" / "audio" / "process_source.pyc",
        ProcTapProcessAudioCaptureFactory=lambda: SimpleNamespace(create=lambda **_kwargs: capture),
        verify_proctap_process_specific=lambda _capture: True,
    )
    modules = {
        "puripuly_heart.config.process_capture_platform": platform_module,
        "puripuly_heart.core.audio.process_source": source_module,
        "proctap": SimpleNamespace(__file__=artifact_native.parent / "__init__.py"),
        "proctap._native": SimpleNamespace(__file__=artifact_native),
    }
    monkeypatch.setattr(smoke.importlib, "import_module", modules.__getitem__)
    monkeypatch.setattr(smoke.importlib.metadata, "version", lambda _name: "1.1.1")
    monkeypatch.setattr(smoke.sys, "executable", str(runtime))
    monkeypatch.setenv("PURIPULY_HEART_NATIVE_RESOURCE_ROOT", str(archive.parent))
    monkeypatch.setenv("PURIPULY_HEART_NATIVE_RUNTIME_ROOT", str(artifact_root))
    monkeypatch.setattr(smoke.time, "sleep", lambda _seconds: None)
    return SimpleNamespace(
        artifact_root=artifact_root,
        report_path=tmp_path / "report.json",
        capture=capture,
        modules=modules,
    )


def test_native_smoke_starts_closes_and_validates_packaged_origin(packaged_smoke) -> None:
    assert (
        smoke.run_smoke(
            artifact_root=packaged_smoke.artifact_root, report_path=packaged_smoke.report_path
        )
        == 0
    )
    report = json.loads(packaged_smoke.report_path.read_text(encoding="utf-8"))
    validate_runtime_report(report, expected_root=packaged_smoke.artifact_root)
    assert report["runtime_native_sha256"] == report["artifact_native_sha256"]
    assert report["capture_started"] is True
    assert report["capture_closed"] is True
    assert report["native_process_specific"] is True
    assert report["device_fallback_used"] is False
    assert packaged_smoke.capture.events == ["start", "close"]
    for name, value in (("device_fallback_used", True), ("capture_closed", False)):
        with pytest.raises(ValueError, match="strict validation"):
            validate_runtime_report(
                {**report, name: value}, expected_root=packaged_smoke.artifact_root
            )
    native = Path(report["artifact_native_module"])
    native.write_bytes(b"modified-native")
    with pytest.raises(ValueError, match="hash does not match"):
        validate_runtime_report(report, expected_root=packaged_smoke.artifact_root)


def test_native_smoke_start_failure_reports_failure_and_attempts_cleanup(packaged_smoke) -> None:
    packaged_smoke.capture.start_failure = RuntimeError("synthetic capture start failed")
    assert (
        smoke.run_smoke(
            artifact_root=packaged_smoke.artifact_root, report_path=packaged_smoke.report_path
        )
        == 1
    )
    assert json.loads(packaged_smoke.report_path.read_text(encoding="utf-8")) == {
        "schema": smoke.PACKAGED_SMOKE_SCHEMA,
        "status": "failed",
        "failure_type": "RuntimeError",
    }
    assert packaged_smoke.capture.events == ["start", "close"]


def test_native_smoke_cleanup_failure_cannot_publish_success(packaged_smoke) -> None:
    failure = RuntimeError("synthetic capture close failed")
    packaged_smoke.capture.close_failure = failure
    with pytest.raises(RuntimeError) as raised:
        smoke.run_smoke(
            artifact_root=packaged_smoke.artifact_root, report_path=packaged_smoke.report_path
        )
    assert raised.value is failure
    assert not packaged_smoke.report_path.exists()


@pytest.mark.parametrize(
    "boundary", ["interpreter", "source_host", "native_origin", "code_origin", "strict_mode"]
)
def test_native_smoke_rejects_wrong_runtime_without_capture(
    packaged_smoke, monkeypatch, tmp_path, boundary
) -> None:
    if boundary == "interpreter":
        monkeypatch.setattr(smoke.sys, "executable", str(tmp_path / "python.exe"))
    elif boundary == "source_host":
        monkeypatch.delenv("PURIPULY_HEART_NATIVE_RESOURCE_ROOT")
    elif boundary == "native_origin":
        swapped = tmp_path / "_native.pyd"
        swapped.write_bytes(b"external")
        packaged_smoke.modules["proctap._native"].__file__ = swapped
    elif boundary == "code_origin":
        packaged_smoke.modules["puripuly_heart.core.audio.process_source"].__file__ = (
            tmp_path / "external.py"
        )
    else:
        packaged_smoke.modules[
            "puripuly_heart.core.audio.process_source"
        ].verify_proctap_process_specific = lambda _capture: False
    assert (
        smoke.run_smoke(
            artifact_root=packaged_smoke.artifact_root, report_path=packaged_smoke.report_path
        )
        == 1
    )
    assert json.loads(packaged_smoke.report_path.read_text(encoding="utf-8"))["status"] == "failed"
    assert "start" not in packaged_smoke.capture.events
