from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from puripuly_heart.config.process_capture_platform import ProcessCapturePlatformAvailability
from puripuly_heart.release_evidence import process_capture_packaged_smoke as smoke


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
    artifact_root = tmp_path / "artifact"
    artifact_native = artifact_root / "proctap" / "_native.cp314-win_amd64.pyd"
    runtime_native = artifact_native
    package_file = artifact_root / "modules.pkg"
    helper_file = artifact_root / "PuriPulyHeartProcessCaptureSmoke.exe"
    artifact_native.parent.mkdir(parents=True)
    artifact_native.write_bytes(b"packaged-native")
    package_file.write_text("collected-modules", encoding="utf-8")
    helper_file.write_bytes(b"helper")
    capture = _Capture()
    platform_module = SimpleNamespace(
        __file__=package_file,
        get_process_capture_platform_availability=lambda: ProcessCapturePlatformAvailability(
            available=True
        ),
    )
    source_module = SimpleNamespace(
        __file__=package_file,
        ProcTapProcessAudioCaptureFactory=lambda: SimpleNamespace(create=lambda **_kwargs: capture),
        verify_proctap_process_specific=lambda _capture: True,
    )
    monkeypatch.setattr(
        smoke.importlib,
        "import_module",
        lambda name: {
            "puripuly_heart.config.process_capture_platform": platform_module,
            "puripuly_heart.core.audio.process_source": source_module,
            "proctap": SimpleNamespace(__file__=package_file),
            "proctap._native": SimpleNamespace(__file__=runtime_native),
        }[name],
    )
    monkeypatch.setattr(smoke.importlib.metadata, "version", lambda _name: "1.1.1")
    monkeypatch.setattr(smoke.sys, "executable", str(helper_file))
    monkeypatch.setattr(smoke.time, "sleep", lambda _seconds: None)

    return SimpleNamespace(
        artifact_root=artifact_root,
        report_path=tmp_path / "report.json",
        capture=capture,
    )


def test_release_only_smoke_starts_strict_factory_and_hashes_packaged_native(
    packaged_smoke: SimpleNamespace,
) -> None:
    assert (
        smoke.run_smoke(
            artifact_root=packaged_smoke.artifact_root,
            report_path=packaged_smoke.report_path,
        )
        == 0
    )
    report = json.loads(packaged_smoke.report_path.read_text(encoding="utf-8"))
    assert report["schema"] == smoke.PACKAGED_SMOKE_SCHEMA
    assert report["status"] == "passed"
    assert report["runtime_native_sha256"] == report["artifact_native_sha256"]
    assert report["release_only_helper"] is True
    assert report["native_process_specific"] is True
    assert report["capture_started"] is True
    assert report["device_fallback_used"] is False
    assert report["credentials_used"] is False
    assert report["network_used"] is False
    assert packaged_smoke.capture.events == ["start", "close"]


def test_release_only_smoke_start_failure_reports_failure_and_attempts_cleanup(
    packaged_smoke: SimpleNamespace,
) -> None:
    packaged_smoke.capture.start_failure = RuntimeError("synthetic capture start failed")

    exit_code = smoke.run_smoke(
        artifact_root=packaged_smoke.artifact_root,
        report_path=packaged_smoke.report_path,
    )

    report = json.loads(packaged_smoke.report_path.read_text(encoding="utf-8"))
    assert exit_code == 1
    assert report == {
        "schema": smoke.PACKAGED_SMOKE_SCHEMA,
        "status": "failed",
        "failure_type": "RuntimeError",
    }
    assert packaged_smoke.capture.events == ["start", "close"]


def test_release_only_smoke_cleanup_exception_propagates_instead_of_returning_success(
    packaged_smoke: SimpleNamespace,
) -> None:
    failure = RuntimeError("synthetic capture close failed")
    packaged_smoke.capture.close_failure = failure

    with pytest.raises(RuntimeError) as raised:
        smoke.run_smoke(
            artifact_root=packaged_smoke.artifact_root,
            report_path=packaged_smoke.report_path,
        )

    assert raised.value is failure
    assert packaged_smoke.capture.events == ["start", "close"]


def test_release_only_smoke_rejects_helper_outside_artifact_root(
    tmp_path: Path, monkeypatch
) -> None:
    helper = tmp_path / "helper" / "PuriPulyHeartProcessCaptureSmoke.exe"
    helper.parent.mkdir()
    helper.write_bytes(b"helper")
    monkeypatch.setattr(smoke.sys, "executable", str(helper))

    assert (
        smoke.run_smoke(artifact_root=tmp_path / "artifact", report_path=tmp_path / "report.json")
        == 1
    )


def test_release_only_smoke_rejects_swapped_runtime_native(tmp_path: Path, monkeypatch) -> None:
    artifact = tmp_path / "artifact"
    native = artifact / "proctap" / "_native.cp314-win_amd64.pyd"
    swapped = artifact / "duplicate" / "_native.cp314-win_amd64.pyd"
    helper = artifact / "PuriPulyHeartProcessCaptureSmoke.exe"
    module_artifact = artifact / "modules.pkg"
    native.parent.mkdir(parents=True)
    swapped.parent.mkdir()
    native.write_bytes(b"production")
    swapped.write_bytes(b"swapped")
    helper.write_bytes(b"helper")
    module_artifact.write_bytes(b"modules")
    platform_module = SimpleNamespace(
        __file__=module_artifact,
        get_process_capture_platform_availability=lambda: ProcessCapturePlatformAvailability(
            available=True
        ),
    )
    monkeypatch.setattr(smoke.sys, "executable", str(helper))
    monkeypatch.setattr(
        smoke.importlib,
        "import_module",
        lambda name: {
            "puripuly_heart.config.process_capture_platform": platform_module,
            "puripuly_heart.core.audio.process_source": SimpleNamespace(__file__=module_artifact),
            "proctap": SimpleNamespace(__file__=module_artifact),
            "proctap._native": SimpleNamespace(__file__=swapped),
        }[name],
    )

    assert smoke.run_smoke(artifact_root=artifact, report_path=tmp_path / "report.json") == 1
