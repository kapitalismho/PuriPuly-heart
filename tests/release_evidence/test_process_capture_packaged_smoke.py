from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from puripuly_heart.config.process_capture_platform import ProcessCapturePlatformAvailability
from puripuly_heart.release_evidence import process_capture_packaged_smoke as smoke


def test_release_only_smoke_starts_strict_factory_and_hashes_packaged_native(
    tmp_path: Path, monkeypatch
) -> None:
    artifact_native = tmp_path / "artifact" / "proctap" / "_native.cp312-win_amd64.pyd"
    runtime_native = artifact_native
    package_file = tmp_path / "artifact" / "modules.pkg"
    helper_file = tmp_path / "artifact" / "PuriPulyHeartProcessCaptureSmoke.exe"
    artifact_native.parent.mkdir(parents=True)
    artifact_native.write_bytes(b"packaged-native")
    package_file.write_text("collected-modules", encoding="utf-8")
    helper_file.write_bytes(b"helper")
    report_path = tmp_path / "report.json"
    capture = SimpleNamespace(
        _backend=SimpleNamespace(_native=SimpleNamespace(is_process_specific=lambda: True)),
        started=False,
        closed=False,
    )
    capture.start = lambda: setattr(capture, "started", True)
    capture.close = lambda: setattr(capture, "closed", True)
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

    assert smoke.run_smoke(artifact_root=tmp_path / "artifact", report_path=report_path) == 0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["release_only_helper"] is True
    assert report["native_process_specific"] is True
    assert report["capture_started"] is True
    assert report["device_fallback_used"] is False
    assert report["credentials_used"] is False
    assert report["network_used"] is False
    assert capture.closed is True


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
    native = artifact / "proctap" / "_native.cp312-win_amd64.pyd"
    swapped = artifact / "duplicate" / "_native.cp312-win_amd64.pyd"
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
