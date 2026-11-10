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
        report_path=tmp_path / "reports" / "capture.json",
        capture=capture,
        modules=modules,
    )


def _run(packaged_smoke) -> tuple[int, dict[str, object]]:
    result = smoke.run_smoke(
        artifact_root=packaged_smoke.artifact_root, report_path=packaged_smoke.report_path
    )
    return result, json.loads(packaged_smoke.report_path.read_text(encoding="utf-8"))


def test_native_smoke_starts_closes_and_validates_packaged_origin(packaged_smoke) -> None:
    result, report = _run(packaged_smoke)

    assert result == 0
    validate_runtime_report(report, expected_root=packaged_smoke.artifact_root)
    assert report["capture_started"] is True
    assert report["capture_closed"] is True
    assert report["native_process_specific"] is True
    assert report["failure"] is None
    assert report["cleanup_failure"] is None
    assert packaged_smoke.capture.events == ["start", "close"]
    assert report["runtime_loaded_modules"] == {
        name: str(Path(packaged_smoke.modules[module].__file__).resolve())
        for name, module in (
            ("platform", "puripuly_heart.config.process_capture_platform"),
            ("process_source", "puripuly_heart.core.audio.process_source"),
            ("proctap", "proctap"),
            ("proctap_native", "proctap._native"),
        )
    }
    assert not any("sha256" in key for key in report)


def test_native_smoke_start_failure_reports_cause_and_attempts_cleanup(
    packaged_smoke, capsys
) -> None:
    failure = RuntimeError("capture start failed")
    failure.__cause__ = OSError("WASAPI HRESULT=0x88890004")
    packaged_smoke.capture.start_failure = failure

    result, report = _run(packaged_smoke)

    assert result == 1
    assert report["status"] == "failed"
    assert report["stage"] == "capture_start"
    assert report["failure"]["type"] == "RuntimeError"
    assert report["failure"]["message"] == "capture start failed"
    assert "WASAPI HRESULT=0x88890004" in report["failure"]["traceback"]
    assert report["capture_started"] is False
    assert report["capture_closed"] is True
    assert packaged_smoke.capture.events == ["start", "close"]
    assert json.loads(capsys.readouterr().err) == report


def test_native_smoke_cleanup_failure_reports_failure_without_retry(packaged_smoke, capsys) -> None:
    packaged_smoke.capture.close_failure = RuntimeError("capture close failed")

    result, report = _run(packaged_smoke)

    assert result == 1
    assert report["status"] == "failed"
    assert report["stage"] == "capture_close"
    assert report["failure"]["message"] == "capture close failed"
    assert report["capture_started"] is True
    assert report["capture_closed"] is False
    assert packaged_smoke.capture.events == ["start", "close"]
    assert json.loads(capsys.readouterr().err) == report
    with pytest.raises(ValueError, match="strict validation"):
        validate_runtime_report(report, expected_root=packaged_smoke.artifact_root)


def test_native_smoke_preserves_start_and_cleanup_failures(packaged_smoke) -> None:
    packaged_smoke.capture.start_failure = RuntimeError("capture start failed")
    packaged_smoke.capture.close_failure = OSError("capture cleanup failed")

    result, report = _run(packaged_smoke)

    assert result == 1
    assert report["stage"] == "capture_start"
    assert report["failure"]["message"] == "capture start failed"
    assert report["cleanup_failure"]["type"] == "OSError"
    assert report["cleanup_failure"]["message"] == "capture cleanup failed"
    assert report["capture_closed"] is False
    assert packaged_smoke.capture.events == ["start", "close"]


def test_native_smoke_unavailable_platform_writes_report(packaged_smoke, capsys) -> None:
    packaged_smoke.modules[
        "puripuly_heart.config.process_capture_platform"
    ].get_process_capture_platform_availability = lambda: ProcessCapturePlatformAvailability(
        available=False, reason="unsupported_windows_build"
    )
    del packaged_smoke.modules["puripuly_heart.core.audio.process_source"]
    del packaged_smoke.modules["proctap"]

    result, report = _run(packaged_smoke)

    assert result == 2
    assert report["status"] == "unavailable"
    assert report["stage"] == "platform_availability"
    assert report["unavailable_reason"] == "unsupported_windows_build"
    assert report["capture_started"] is False
    assert report["failure"] is None
    assert packaged_smoke.capture.events == []
    assert json.loads(capsys.readouterr().err) == report


@pytest.mark.parametrize(
    "module",
    [
        "puripuly_heart.config.process_capture_platform",
        "proctap",
        "proctap._native",
        "puripuly_heart.core.audio.process_source",
    ],
)
def test_native_smoke_reports_missing_packaged_dependency(
    packaged_smoke, monkeypatch, module
) -> None:
    def import_module(name):
        if name == module:
            raise ImportError(f"DLL or dependency unavailable: {module}")
        return packaged_smoke.modules[name]

    monkeypatch.setattr(smoke.importlib, "import_module", import_module)

    result, report = _run(packaged_smoke)

    assert result == 1
    assert report["failure"]["type"] == "ImportError"
    assert report["failure"]["message"] == f"DLL or dependency unavailable: {module}"
    assert report["stage"] == (
        "platform_availability"
        if module == "puripuly_heart.config.process_capture_platform"
        else "packaged_modules"
    )
    assert packaged_smoke.capture.events == []


def test_native_smoke_reports_factory_failure_without_claiming_capture(packaged_smoke) -> None:
    def create(**_kwargs):
        raise RuntimeError("Failed to initialize process loopback (HRESULT=0x80070490)")

    packaged_smoke.modules[
        "puripuly_heart.core.audio.process_source"
    ].ProcTapProcessAudioCaptureFactory = lambda: SimpleNamespace(create=create)

    result, report = _run(packaged_smoke)

    assert result == 1
    assert report["stage"] == "capture_create"
    assert "HRESULT=0x80070490" in report["failure"]["message"]
    assert report["capture_started"] is False
    assert report["capture_closed"] is False
    assert packaged_smoke.capture.events == []


@pytest.mark.parametrize(
    ("boundary", "stage"),
    [
        ("interpreter", "runtime_layout"),
        ("source_host", "runtime_layout"),
        ("resource_root", "runtime_layout"),
        ("native_origin", "packaged_modules"),
        ("code_origin", "packaged_modules"),
        ("strict_mode", "capture_verify"),
    ],
)
def test_native_smoke_rejects_wrong_runtime_without_capture(
    packaged_smoke, monkeypatch, tmp_path, boundary, stage
) -> None:
    if boundary == "interpreter":
        monkeypatch.setattr(smoke.sys, "executable", str(tmp_path / "python.exe"))
    elif boundary == "source_host":
        monkeypatch.delenv("PURIPULY_HEART_NATIVE_RESOURCE_ROOT")
    elif boundary == "resource_root":
        monkeypatch.setenv("PURIPULY_HEART_NATIVE_RESOURCE_ROOT", str(tmp_path))
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

    result, report = _run(packaged_smoke)

    assert result == 1
    assert report["status"] == "failed"
    assert report["stage"] == stage
    assert "start" not in packaged_smoke.capture.events
    assert packaged_smoke.capture.events == (["close"] if boundary == "strict_mode" else [])


@pytest.mark.parametrize("count", [0, 2])
def test_native_smoke_rejects_missing_or_ambiguous_native_payload(packaged_smoke, count) -> None:
    native = Path(packaged_smoke.modules["proctap._native"].__file__)
    if count == 0:
        native.unlink()
    else:
        (native.parent / "_native.extra.pyd").write_bytes(b"extra")

    result, report = _run(packaged_smoke)

    assert result == 1
    assert report["stage"] == "packaged_modules"
    assert (
        report["failure"]["message"] == "native payload must contain exactly one ProcTap extension"
    )
    assert packaged_smoke.capture.events == []


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("capture_closed", False),
        ("capture_started", False),
        ("native_process_specific", False),
        ("failure", {"message": "failed"}),
        ("cleanup_failure", {"message": "failed"}),
        ("unavailable_reason", "unsupported_system"),
    ],
)
def test_runtime_report_rejects_unsuccessful_capture(packaged_smoke, name, value) -> None:
    _, report = _run(packaged_smoke)

    with pytest.raises(ValueError, match="strict validation"):
        validate_runtime_report({**report, name: value}, expected_root=packaged_smoke.artifact_root)


def test_runtime_report_rejects_external_module_paths(packaged_smoke, tmp_path) -> None:
    _, report = _run(packaged_smoke)
    report["runtime_loaded_modules"]["process_source"] = str(tmp_path / "source.py")

    with pytest.raises(ValueError, match="outside the expected artifact root"):
        validate_runtime_report(report, expected_root=packaged_smoke.artifact_root)


def test_runtime_report_rejects_mismatched_native_path(packaged_smoke) -> None:
    _, report = _run(packaged_smoke)
    report["runtime_native_module"] = str(
        packaged_smoke.artifact_root / "site-packages" / "proctap" / "_native.other.pyd"
    )

    with pytest.raises(ValueError, match="does not match its loaded path"):
        validate_runtime_report(report, expected_root=packaged_smoke.artifact_root)
