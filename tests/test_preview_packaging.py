from scripts.preview.build_portable import copy_payload_file, extended_path, package_copy_ignore


def test_beta_rejects_missing_gpu_runtime(tmp_path):
    import pytest

    from scripts.preview.build_portable import validate_beta_inputs

    with pytest.raises(FileNotFoundError):
        validate_beta_inputs(tmp_path)


def test_beta_requires_overlay_after_gpu_runtime_validation(tmp_path, monkeypatch):
    import pytest

    from puripuly_heart.core.overlay import openvr_vendor
    from puripuly_heart.release_evidence import gpu_worker_distribution
    from scripts.preview.build_portable import validate_beta_inputs

    monkeypatch.setattr(
        gpu_worker_distribution, "validate_gpu_worker_runtime", lambda root: {"validated": True}
    )
    monkeypatch.setattr(openvr_vendor, "validate_vendored_openvr_bundle", lambda root: None)
    with pytest.raises(SystemExit, match="compiled overlay"):
        validate_beta_inputs(tmp_path)
    overlay = tmp_path / "build" / "overlay"
    overlay.mkdir(parents=True)
    (overlay / "PuriPulyHeartOverlay.exe").write_bytes(b"test-only")
    assert validate_beta_inputs(tmp_path) == {"validated": True}


def test_beta_stages_pinned_openvr_beside_overlay_and_passes_packaged_guard(tmp_path):
    from puripuly_heart.core.overlay.openvr_vendor import OPENVR_VENDOR_DLL_SHA256
    from puripuly_heart.core.overlay.process_runners import DefaultOverlayProcessRunner
    from scripts.preview.build_portable import stage_beta_openvr_runtime
    from tests.helpers.paths import REPO_ROOT

    directory = tmp_path / "app" / "build" / "overlay"
    directory.mkdir(parents=True)
    executable = directory / "PuriPulyHeartOverlay.exe"
    executable.write_bytes(b"mock-executable-only")
    assert stage_beta_openvr_runtime(REPO_ROOT, tmp_path) == OPENVR_VENDOR_DLL_SHA256
    assert (
        DefaultOverlayProcessRunner.ensure_bundled_openvr_runtime_dll(executable)
        == directory / "openvr_api.dll"
    )


def test_runtime_files_with_windows_long_names_are_copied(tmp_path):
    source = tmp_path / ("source-" + "s" * 80)
    destination = tmp_path / "destination"
    source.mkdir()
    destination.mkdir()
    filename = "dependency-" + "d" * 170 + ".py"
    original = source / filename
    copied = destination / filename
    with open(extended_path(original), "wb") as handle:
        handle.write(b"verified runtime payload")
    copy_payload_file(original, copied)
    with open(extended_path(copied), "rb") as handle:
        assert handle.read() == b"verified runtime payload"


def test_build_exclusions_do_not_remove_runtime_utilities(tmp_path):
    ignore = package_copy_ignore(tmp_path)
    assert "pip" in ignore(tmp_path, ["pip", "flet", "__pycache__"])
    assert "pip.py" not in ignore(tmp_path / "flet" / "utils", ["pip.py", "__pycache__"])
    assert "__pycache__" in ignore(tmp_path / "flet" / "utils", ["pip.py", "__pycache__"])


def test_beta_rejects_missing_vc_runtime(tmp_path):
    import pytest

    from scripts.preview.build_portable import stage_beta_vc_runtime

    with pytest.raises(SystemExit, match="Visual C"):
        stage_beta_vc_runtime(tmp_path, None)


def test_beta_stages_crt_beside_each_independent_process(tmp_path):
    from scripts.preview.build_portable import stage_beta_vc_runtime

    source = tmp_path / "crt"
    source.mkdir()
    for name in ("msvcp140.dll", "vcruntime140.dll", "vcruntime140_1.dll"):
        (source / name).write_bytes(b"fake-test-crt")
    for relative in (
        "runtime",
        "flet/flet",
        "app/build/gpu_worker",
        "app/build/overlay",
        "app/build/llama.cpp/test/cpu",
        "app/build/llama.cpp/test/vulkan",
    ):
        (tmp_path / "payload" / relative).mkdir(parents=True)
    hashes = stage_beta_vc_runtime(tmp_path / "payload", source)
    assert len(hashes) == 3
    assert (tmp_path / "payload/app/build/gpu_worker/msvcp140.dll").read_bytes() == b"fake-test-crt"
    assert (tmp_path / "payload/flet/flet/vcruntime140.dll").is_file()
