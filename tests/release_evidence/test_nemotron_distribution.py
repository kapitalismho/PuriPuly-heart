from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from puripuly_heart.release_evidence.nemotron_distribution import (
    MANIFEST,
    SOURCE_REVISION,
    main,
    stage_runtime,
    validate_runtime,
)


def _artifact(root: Path) -> dict:
    files = {}
    names = [
        f"bin/{name}.dll"
        for name in (
            "nemo_speech_asr_c",
            "nemo_speech_asr",
            "ggml",
            "ggml-base",
            "ggml-cpu",
            "ggml-vulkan",
        )
    ]
    names += [
        f"share/licenses/nemo-speech/{name}"
        for name in (
            "LICENSE",
            "NOTICE",
            "THIRD_PARTY_NOTICES.md",
            "third_party/ggml/LICENSE",
        )
    ]
    names += [
        f"share/licenses/nemo-speech/third_party/{name}/copyright"
        for name in ("abseil", "protobuf", "sentencepiece", "utf8-range")
    ]
    for name in names:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode())
        files[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {
        "schema_version": 1,
        "abi": "puripuly-nemotron-1",
        "source": {"repository": "NVIDIA/NeMo-Speech.cpp", "revision": SOURCE_REVISION},
        "dependencies": {"llama.cpp": "a" * 40, "vcpkg": "b" * 40},
        "patches": {"native/nemotron/vulkan.patch": "c" * 64},
        "policy": {
            "backend": "Vulkan",
            "neural_cpu_fallback": False,
            "precision": "F16",
            "schedule_guard": True,
            "physical_gpu_only": True,
        },
        "files": files,
    }
    (root / MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


def test_optional_and_required_absence(tmp_path: Path) -> None:
    assert validate_runtime(tmp_path / "absent") == {"bundled": False}
    with pytest.raises(FileNotFoundError, match="required Nemotron"):
        validate_runtime(tmp_path / "absent", required=True)


def test_stage_preserves_inventory_licenses_and_asr_isolation(tmp_path: Path) -> None:
    source = tmp_path / "source"
    manifest = _artifact(source)
    target = tmp_path / "distribution"
    target.mkdir()
    asr = target / "ggml.dll"
    asr.write_bytes(b"independent ASR")
    result = stage_runtime(source, target / "nemotron")
    assert result["bundled"] is True
    assert result["files"] == manifest["files"]
    assert asr.read_bytes() == b"independent ASR"
    assert not (target / "python.exe").exists()  # Runtime never supplies another interpreter.
    assert validate_runtime(target / "nemotron") == result
    with pytest.raises(FileExistsError, match="refusing to merge"):
        stage_runtime(source, target / "nemotron")


@pytest.mark.parametrize(
    "change", ["missing", "changed", "stale", "missing-license", "missing-dll-in-manifest"]
)
def test_inventory_rejects_incomplete_or_stale_artifact(tmp_path: Path, change: str) -> None:
    manifest = _artifact(tmp_path)
    if change == "missing":
        (tmp_path / "bin/ggml-vulkan.dll").unlink()
    elif change == "changed":
        (tmp_path / "bin/ggml.dll").write_bytes(b"old build")
    elif change == "stale":
        (tmp_path / "bin/stale.dll").write_bytes(b"old build")
    else:
        name = (
            "bin/ggml.dll"
            if change == "missing-dll-in-manifest"
            else "share/licenses/nemo-speech/LICENSE"
        )
        del manifest["files"][name]
        (tmp_path / name).unlink()
        (tmp_path / MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError):
        validate_runtime(tmp_path)


@pytest.mark.parametrize(
    "field,value",
    [
        ("abi", "wrong"),
        ("source", {"repository": "NVIDIA/NeMo-Speech.cpp", "revision": "a" * 40}),
        ("dependencies", {}),
        ("patches", {}),
        (
            "policy",
            {
                "backend": "CPU",
                "neural_cpu_fallback": True,
                "precision": "F16",
                "schedule_guard": True,
            },
        ),
    ],
)
def test_incompatible_runtime_is_rejected(tmp_path: Path, field: str, value: object) -> None:
    manifest = _artifact(tmp_path)
    manifest[field] = value
    (tmp_path / MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError):
        validate_runtime(tmp_path)


@pytest.mark.parametrize(
    "name",
    [
        "../outside.dll",
        "C:/outside.dll",
        "bin\\outside.dll",
        "model.gguf",
        "ggml.dll",
        "bin/CON.dll",
        "bin/trailing.dll.",
        "bin/name?.dll",
    ],
)
def test_unsafe_paths_weights_and_unisolated_binaries_rejected(tmp_path: Path, name: str) -> None:
    manifest = _artifact(tmp_path)
    manifest["files"][name] = "a" * 64
    (tmp_path / MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError):
        validate_runtime(tmp_path)


def test_cli_stages_distribution(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    source = tmp_path / "source"
    _artifact(source)
    assert main([str(source), "--required", "--stage-to", str(tmp_path / "nemotron")]) == 0
    assert json.loads(capsys.readouterr().out)["bundled"] is True


def test_native_build_and_release_share_explicit_policy() -> None:
    repo = Path(__file__).resolve().parents[2]
    build = (repo / "scripts/ci/build-native-experimental.ps1").read_text(encoding="utf-8")
    release = (repo / "scripts/ci/build-native-release-artifacts.ps1").read_text(encoding="utf-8")
    for script in (build, release):
        assert '$NemotronRuntimeDir = ""' in script
        assert "$RequireNemotronRuntime" in script
        assert "puripuly_heart.release_evidence.nemotron_distribution" in script
    assert '(Join-Path $artifactRoot "nemotron")' in build
    assert '(Join-Path $OutputDir "nemotron")' in release
    assert release.index("$nemotronArguments") < release.index('"validate-target"')


def test_reused_distribution_must_match_explicit_build(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _artifact(source)
    target = tmp_path / "nemotron"
    stage_runtime(source, target)
    assert main([str(target), "--expected-runtime", str(source)]) == 0
    manifest = json.loads((target / MANIFEST).read_text(encoding="utf-8"))
    manifest["patches"]["native/nemotron/vulkan.patch"] = "d" * 64
    (target / MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="differs from explicitly selected build"):
        main([str(target), "--expected-runtime", str(source)])


@pytest.mark.parametrize("guard", [None, False, 1])
def test_pre_physical_gpu_guard_artifacts_are_incompatible(tmp_path: Path, guard: object) -> None:
    manifest = _artifact(tmp_path)
    if guard is None:
        del manifest["policy"]["physical_gpu_only"]
    else:
        manifest["policy"]["physical_gpu_only"] = guard
    (tmp_path / MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="physical_gpu_only"):
        validate_runtime(tmp_path)


def test_staging_rejects_a_different_complete_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import shutil

    source = tmp_path / "source"
    _artifact(source)
    copytree = shutil.copytree

    def copy_different_build(source_root: Path, destination: Path, *args, **kwargs):
        result = copytree(source_root, destination, *args, **kwargs)
        if Path(source_root) == source:
            manifest_path = Path(destination) / MANIFEST
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["patches"]["native/nemotron/vulkan.patch"] = "d" * 64
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        return result

    monkeypatch.setattr(
        "puripuly_heart.release_evidence.nemotron_distribution.shutil.copytree",
        copy_different_build,
    )
    with pytest.raises(ValueError, match="changed while staging"):
        stage_runtime(source, tmp_path / "destination")


@pytest.mark.parametrize("dependency", ["abseil", "protobuf", "sentencepiece", "utf8-range"])
def test_rebuilt_inventory_cannot_omit_linked_dependency_notices(
    tmp_path: Path, dependency: str
) -> None:
    manifest = _artifact(tmp_path)
    name = f"share/licenses/nemo-speech/third_party/{dependency}/copyright"
    del manifest["files"][name]
    (tmp_path / name).unlink()
    (tmp_path / MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="lacks required files"):
        validate_runtime(tmp_path)
