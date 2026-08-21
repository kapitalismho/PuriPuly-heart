from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path

import pytest

from puripuly_heart.release_evidence import managed_gemma_distribution as distribution

ROOT = Path(__file__).resolve().parents[2]


def _identity(path: Path) -> dict[str, int | str]:
    return {
        "size": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _write_package(package_root: Path) -> None:
    runtime_root = package_root / distribution.PACKAGED_RUNTIME_RELATIVE_DIR
    archives: dict[str, object] = {}
    for contract in distribution.ARCHIVES:
        backend_root = runtime_root / contract.backend
        backend_root.mkdir(parents=True)
        records = []
        for name in sorted(contract.required_files):
            path = backend_root / name
            path.write_bytes(f"{contract.backend}:{name}".encode())
            records.append({"path": name, **_identity(path)})
        archives[contract.backend] = {
            "filename": contract.filename,
            "size": contract.size,
            "sha256": contract.sha256,
            "url": contract.url,
            "files": records,
        }
    provenance_root = package_root / distribution.PROVENANCE_RELATIVE_DIR
    provenance_root.mkdir(parents=True)
    provenance = {}
    for name in ("LICENSE", "README.md"):
        source = ROOT / distribution.PROVENANCE_RELATIVE_DIR / name
        target = provenance_root / name
        target.write_bytes(source.read_bytes())
        provenance[name] = _identity(target)
    manifest = {
        "schema": distribution.MANIFEST_SCHEMA,
        "release": {
            "build": distribution.LLAMA_CPP_BUILD,
            "commit": distribution.LLAMA_CPP_COMMIT,
            "url": distribution.RELEASE_URL,
            "runtime_dirname": distribution.LLAMA_CPP_RUNTIME_DIRNAME,
            "packaged_relative_dir": distribution.PACKAGED_RUNTIME_RELATIVE_DIR.as_posix(),
        },
        "archives": archives,
        "provenance": provenance,
    }
    (runtime_root / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_runtime_archive_stages_only_server_dependency_closure(tmp_path: Path) -> None:
    contract = distribution.ArchiveContract(
        backend="cpu",
        filename="runtime.zip",
        size=0,
        sha256="unused",
        required_files=frozenset(distribution.COMMON_REQUIRED_FILES),
    )
    archive_path = tmp_path / contract.filename
    with zipfile.ZipFile(archive_path, "w") as archive:
        for name in sorted(distribution.COMMON_REQUIRED_FILES):
            archive.writestr(name, name)
        archive.writestr("llama-cli.exe", "excluded")
        archive.writestr("llama-quantize-impl.dll", "excluded")

    records = distribution._extract_runtime_archive(
        archive_path,
        tmp_path / "runtime",
        contract,
    )

    assert {record["path"] for record in records} == distribution.COMMON_REQUIRED_FILES
    assert {path.name for path in (tmp_path / "runtime").iterdir()} == (
        distribution.COMMON_REQUIRED_FILES
    )


@pytest.mark.parametrize(
    "forbidden_name",
    [
        "gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf",
        "mtp-gemma-4-E4B-it.gguf",
        "gemma-4-12B-it-qat-UD-Q4_K_XL.gguf",
        "mmproj-model-f16.gguf",
    ],
)
def test_runtime_archive_rejects_model_artifacts(
    tmp_path: Path,
    forbidden_name: str,
) -> None:
    contract = distribution.ArchiveContract(
        backend="cpu",
        filename="runtime.zip",
        size=0,
        sha256="unused",
        required_files=frozenset(distribution.COMMON_REQUIRED_FILES),
    )
    archive_path = tmp_path / contract.filename
    with zipfile.ZipFile(archive_path, "w") as archive:
        for name in sorted(distribution.COMMON_REQUIRED_FILES):
            archive.writestr(name, name)
        archive.writestr(forbidden_name, "model")

    with pytest.raises(RuntimeError, match="model artifact"):
        distribution._extract_runtime_archive(
            archive_path,
            tmp_path / "runtime",
            contract,
        )


def test_package_verifier_proves_both_backends_and_no_model_weights(tmp_path: Path) -> None:
    _write_package(tmp_path)

    result = distribution.verify_package(tmp_path)

    assert result["backends"] == ["cpu", "vulkan"]
    assert result["launched_backends"] == []
    assert result["model_artifacts"] == []


def test_runtime_root_validator_binds_release_and_file_identities(tmp_path: Path) -> None:
    _write_package(tmp_path)
    runtime_root = tmp_path / distribution.PACKAGED_RUNTIME_RELATIVE_DIR

    manifest = distribution.validate_runtime_root(runtime_root)

    assert manifest["release"]["build"] == distribution.LLAMA_CPP_BUILD
    assert manifest["release"]["commit"] == distribution.LLAMA_CPP_COMMIT

    server = runtime_root / "vulkan" / "llama-server.exe"
    server.write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="(?:size|identity) mismatch"):
        distribution.validate_runtime_root(runtime_root)


def test_package_verifier_rejects_model_anywhere_in_installed_tree(tmp_path: Path) -> None:
    _write_package(tmp_path)
    model = tmp_path / "puripuly_heart" / "data" / "models" / "unexpected.gguf"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"model")

    with pytest.raises(RuntimeError, match="model artifacts found in package"):
        distribution.verify_package(tmp_path)


@pytest.mark.parametrize(
    "provenance",
    [
        {},
        {
            "../../LICENSE": {
                "size": distribution.LICENSE_SIZE,
                "sha256": distribution.LICENSE_SHA256,
            }
        },
    ],
)
def test_package_verifier_requires_fixed_provenance(
    tmp_path: Path, provenance: dict[str, object]
) -> None:
    _write_package(tmp_path)
    manifest_path = tmp_path / distribution.PACKAGED_RUNTIME_RELATIVE_DIR / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["provenance"] = provenance
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RuntimeError, match="provenance identity mismatch"):
        distribution.verify_package(tmp_path)


def test_package_verifier_rejects_nested_runtime_content(tmp_path: Path) -> None:
    _write_package(tmp_path)
    nested = tmp_path / distribution.PACKAGED_RUNTIME_RELATIVE_DIR / "cpu" / "nested"
    nested.mkdir()
    (nested / "llama-cli.exe").write_bytes(b"unexpected")

    with pytest.raises(RuntimeError, match="contains nested content"):
        distribution.verify_package(tmp_path)


def test_pyinstaller_normalization_keeps_owned_layout_and_removes_root_duplicates(
    tmp_path: Path,
) -> None:
    prepared = tmp_path / "build" / "llama.cpp" / distribution.LLAMA_CPP_RUNTIME_DIRNAME / "cpu"
    prepared.mkdir(parents=True)
    source = prepared / "llama.dll"
    source.write_bytes(b"runtime")
    external = tmp_path / "external.dll"
    external.write_bytes(b"external")
    binaries = [
        (
            f"{distribution.PACKAGED_RUNTIME_RELATIVE_DIR.as_posix()}/cpu/llama.dll",
            str(source),
            "BINARY",
        ),
        ("llama.dll", str(source), "BINARY"),
        ("external.dll", str(external), "BINARY"),
    ]

    distribution.normalize_pyinstaller_binaries(binaries, tmp_path)

    assert [binary[0] for binary in binaries] == [
        f"{distribution.PACKAGED_RUNTIME_RELATIVE_DIR.as_posix()}/cpu/llama.dll",
        "external.dll",
    ]


def test_package_verifier_rejects_runtime_copy_outside_owned_directory(tmp_path: Path) -> None:
    _write_package(tmp_path)
    nested = tmp_path / distribution.PACKAGED_RUNTIME_RELATIVE_DIR / "cpu" / "llama-server-impl.dll"
    (tmp_path / nested.name).write_bytes(nested.read_bytes())

    with pytest.raises(RuntimeError, match="escaped their owned package directory"):
        distribution.verify_package(tmp_path)


def test_package_verifier_rejects_nested_runtime_copy_outside_owned_directory(
    tmp_path: Path,
) -> None:
    _write_package(tmp_path)
    source = tmp_path / distribution.PACKAGED_RUNTIME_RELATIVE_DIR / "cpu" / "llama-server-impl.dll"
    duplicate = tmp_path / "alternate" / "nested-runtime.dll"
    duplicate.parent.mkdir(parents=True)
    duplicate.write_bytes(source.read_bytes())

    with pytest.raises(RuntimeError, match="escaped their owned package directory"):
        distribution.verify_package(tmp_path)


def test_prepare_runtime_revalidates_archive_bytes_after_download(
    monkeypatch, tmp_path: Path
) -> None:
    valid_archive = io.BytesIO()
    with zipfile.ZipFile(valid_archive, "w") as archive:
        for name in sorted(distribution.COMMON_REQUIRED_FILES):
            archive.writestr(name, name)
    valid_payload = valid_archive.getvalue()
    contract = distribution.ArchiveContract(
        backend="cpu",
        filename="runtime.zip",
        size=len(valid_payload),
        sha256=hashlib.sha256(valid_payload).hexdigest(),
        required_files=frozenset(distribution.COMMON_REQUIRED_FILES),
    )
    monkeypatch.setattr(distribution, "ARCHIVES", (contract,))

    def replace_after_validated_download(_contract, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"substituted")

    monkeypatch.setattr(distribution, "_download", replace_after_validated_download)

    with pytest.raises(RuntimeError, match="size mismatch"):
        distribution.prepare_runtime(
            ROOT,
            tmp_path / "cache",
            tmp_path / "prepared",
        )


def test_prepare_runtime_reuses_valid_output_without_replacement(
    monkeypatch, tmp_path: Path
) -> None:
    package_root = tmp_path / "package"
    _write_package(package_root)
    output_root = package_root / distribution.PACKAGED_RUNTIME_RELATIVE_DIR

    monkeypatch.setattr(
        distribution,
        "_download",
        lambda *_args, **_kwargs: pytest.fail("valid output should be reused"),
    )

    result = distribution.prepare_runtime(ROOT, tmp_path / "cache", output_root)

    assert result["release"]["build"] == distribution.LLAMA_CPP_BUILD


def test_installer_verifier_rejects_install_time_managed_gemma_download(tmp_path: Path) -> None:
    installer_path = tmp_path / "installer.iss"
    installer_path.write_text(
        '[Files]\nSource: "{#MyPackagedAppDir}\\*"\n'
        "https://huggingface.co/unsloth/gemma-4-E4B-it-qat-GGUF/resolve/revision/"
        "gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="install.*download identities"):
        distribution.verify_installer(installer_path)


@pytest.mark.parametrize("identity", ["another-model.gguf", "mmproj-model-f16.bin"])
def test_installer_verifier_rejects_any_model_artifact_identity(
    tmp_path: Path, identity: str
) -> None:
    installer_path = tmp_path / "installer.iss"
    installer_path.write_text(
        '[Files]\nSource: "{#MyPackagedAppDir}\\*"\n' + identity,
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="model artifact identities"):
        distribution.verify_installer(installer_path)


def test_installer_verifier_rejects_application_postinstall_launch(tmp_path: Path) -> None:
    installer_path = tmp_path / "installer.iss"
    installer_path.write_text(
        '[Files]\nSource: "{#MyPackagedAppDir}\\*"\n'
        '[Run]\nFilename: "{app}\\{#MyAppExeName}"; Flags: nowait postinstall\n',
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="launches the application postinstall"):
        distribution.verify_installer(installer_path)


def test_release_paths_prepare_verify_and_install_pinned_runtime_without_gemma_download() -> None:
    workflow = (ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
    release_script = (ROOT / "scripts" / "ci" / "build-release-artifacts.ps1").read_text(
        encoding="utf-8"
    )
    build_spec = (ROOT / "build.spec").read_text(encoding="utf-8")
    installer = (ROOT / "installer.iss").read_text(encoding="utf-8")
    module = "puripuly_heart.release_evidence.managed_gemma_distribution"

    workflow_prepare = workflow.index(f"-m {module} prepare")
    workflow_build = workflow.index("-m PyInstaller")
    workflow_verify = workflow.index(f"-m {module} verify-package")
    assert workflow_prepare < workflow_build < workflow_verify
    assert release_script.index('"prepare"') < release_script.index('"PyInstaller"')
    assert release_script.count('"verify-package"') == 3
    assert release_script.count('"verify-installer"') == 1
    assert (
        "managed_gemma_runtime_datas = [] if release_smoke else pyinstaller_data_entries(Path.cwd())"
        in build_spec
    )
    assert (
        distribution.verify_installer(ROOT / "installer.iss")["managed_gemma_install_download"]
        is False
    )
    assert "[Run]" not in installer
    for forbidden in (
        "unsloth/gemma-4-E4B-it-qat-GGUF",
        "gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf",
        "mtp-gemma-4-E4B-it.gguf",
        "unsloth/gemma-4-12B-it-qat-GGUF",
        "gemma-4-12B-it-qat-UD-Q4_K_XL.gguf",
    ):
        assert forbidden not in installer


def test_llama_cpp_license_and_notice_are_shipped_from_pinned_provenance() -> None:
    license_path = ROOT / distribution.PROVENANCE_RELATIVE_DIR / "LICENSE"
    notice = (ROOT / "src" / "puripuly_heart" / "data" / "THIRD_PARTY_NOTICES.txt").read_text(
        encoding="utf-8"
    )

    license_bytes = license_path.read_bytes()
    assert b"\r\n" not in license_bytes
    assert hashlib.sha256(license_bytes).hexdigest() == distribution.LICENSE_SHA256
    assert "llama.cpp CPU and Vulkan runtime" in notice
    assert distribution.LLAMA_CPP_COMMIT in notice
