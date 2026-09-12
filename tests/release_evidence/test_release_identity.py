from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from puripuly_heart.release_evidence import release_identity as identity
from tests.helpers.paths import REPO_ROOT as ROOT

VERSION = "2.6.1"
TAG = "v2.6.1"
INSTALLER_EXE = "PuriPulyHeart-Setup-2.6.1.exe"
SOURCE_SHA = "a" * 40


def _artifact_entries(paths: list) -> list[dict[str, object]]:
    entries = []
    for index, path in enumerate(paths):
        file_identity = identity.file_identity(path)
        entries.append(
            {
                "role": f"candidate-{index}",
                "filename": path.name,
                "size": file_identity["size"],
                "sha256": file_identity["sha256"],
            }
        )
    return entries


def test_rendered_body_references_produced_installer() -> None:
    body = identity.render_release_body("Download `{{INSTALLER_EXE}}` below.", INSTALLER_EXE)

    assert INSTALLER_EXE in body
    assert "{{INSTALLER_EXE}}" not in body


def test_release_surface_accepts_matching_tag_title_body_and_assets() -> None:
    body = f"Download `{INSTALLER_EXE}` from the Assets section below, then run it."

    assert (
        identity.verify_release_surface(
            version=VERSION,
            tag=TAG,
            title=TAG,
            installer_exe=INSTALLER_EXE,
            body_text=body,
            asset_names=[
                INSTALLER_EXE,
                "PuriPulyHeart-soxr-third-party-source-bundle.zip",
            ],
        )
        == INSTALLER_EXE
    )


def test_release_surface_rejects_disagreeing_title_body_and_asset_selection() -> None:
    body = f"Download `{INSTALLER_EXE}` from the Assets section below."

    with pytest.raises(RuntimeError, match="title mismatch"):
        identity.verify_release_surface(
            version=VERSION,
            tag=TAG,
            title="v9.9.9",
            installer_exe=INSTALLER_EXE,
            body_text=body,
            asset_names=[INSTALLER_EXE],
        )
    with pytest.raises(RuntimeError, match="unrendered"):
        identity.verify_release_surface(
            version=VERSION,
            tag=TAG,
            title=TAG,
            installer_exe=INSTALLER_EXE,
            body_text="Download `{{INSTALLER_EXE}}` below.",
            asset_names=[INSTALLER_EXE],
        )
    with pytest.raises(RuntimeError, match="does not reference"):
        identity.verify_release_surface(
            version=VERSION,
            tag=TAG,
            title=TAG,
            installer_exe=INSTALLER_EXE,
            body_text="Download the installer below.",
            asset_names=[INSTALLER_EXE],
        )
    with pytest.raises(RuntimeError, match="do not include"):
        identity.verify_release_surface(
            version=VERSION,
            tag=TAG,
            title=TAG,
            installer_exe=INSTALLER_EXE,
            body_text=body,
            asset_names=["PuriPulyHeart-soxr-third-party-source-bundle.zip"],
        )


def test_release_surface_rejects_tag_version_disagreement() -> None:
    with pytest.raises(RuntimeError, match="tag mismatch"):
        identity.verify_release_surface(
            version="2.6.2",
            tag=TAG,
            title=TAG,
            installer_exe="PuriPulyHeart-Setup-2.6.2.exe",
            body_text="Download `PuriPulyHeart-Setup-2.6.2.exe` below.",
            asset_names=["PuriPulyHeart-Setup-2.6.2.exe"],
        )


def test_local_provenance_omits_hosted_run_claim(tmp_path) -> None:
    first = tmp_path / "first.bin"
    first.write_bytes(b"first-candidate")
    provenance = identity.build_provenance(
        version=VERSION,
        tag=TAG,
        source_sha=SOURCE_SHA,
        build_origin="local",
        artifacts=_artifact_entries([first]),
    )

    assert provenance["build_origin"] == "local"
    assert "workflow" not in provenance

    with pytest.raises(RuntimeError, match="must not carry"):
        identity.build_provenance(
            version=VERSION,
            tag=TAG,
            source_sha=SOURCE_SHA,
            build_origin="local",
            run_id="123",
            run_attempt="1",
            artifacts=_artifact_entries([first]),
        )


def test_hosted_provenance_requires_run_metadata(tmp_path) -> None:
    first = tmp_path / "first.bin"
    first.write_bytes(b"first-candidate")

    with pytest.raises(RuntimeError, match="requires repository"):
        identity.build_provenance(
            version=VERSION,
            tag=TAG,
            source_sha=SOURCE_SHA,
            build_origin="github-hosted",
            repository="owner/repo",
            server_url="https://github.com",
            artifacts=_artifact_entries([first]),
        )

    provenance = identity.build_provenance(
        version=VERSION,
        tag=TAG,
        source_sha=SOURCE_SHA,
        build_origin="github-hosted",
        repository="owner/repo",
        server_url="https://github.com",
        run_id="123",
        run_attempt="1",
        artifacts=_artifact_entries([first]),
    )

    assert provenance["workflow"] == {
        "server_url": "https://github.com",
        "run_id": "123",
        "run_attempt": "1",
    }


def test_provenance_hashes_are_reverified_against_bytes(tmp_path) -> None:
    first = tmp_path / "first.bin"
    second = tmp_path / "second.bin"
    first.write_bytes(b"first-candidate")
    second.write_bytes(b"second-candidate")
    provenance = identity.build_provenance(
        version=VERSION,
        tag=TAG,
        source_sha=SOURCE_SHA,
        build_origin="local",
        artifacts=_artifact_entries([first, second]),
    )
    provenance_path = tmp_path / "provenance.json"
    identity.write_provenance_file(provenance_path, provenance)

    reloaded = json.loads(provenance_path.read_text(encoding="utf-8"))
    identity.verify_assets_against_provenance(reloaded, [first, second])

    second.write_bytes(b"second-candidate-mutated")
    with pytest.raises(RuntimeError, match="disagrees"):
        identity.verify_assets_against_provenance(reloaded, [first, second])

    third = tmp_path / "third.bin"
    third.write_bytes(b"unrecorded-candidate")
    with pytest.raises(RuntimeError, match="no provenance record"):
        identity.verify_assets_against_provenance(reloaded, [first, third])


def test_source_built_executables_carry_release_product_identity() -> None:
    pytest.importorskip("pefile")
    candidates = [
        ROOT / "dist" / "PuriPulyHeart" / "PuriPulyHeart.exe",
        ROOT / "dist" / "PuriPulyHeart" / "PuriPulyHeartGpuWorker.exe",
        ROOT / "build" / "overlay" / "PuriPulyHeartOverlay.exe",
        ROOT / "installer_output" / INSTALLER_EXE,
    ]
    missing = [path for path in candidates if not path.is_file()]
    if missing:
        pytest.skip(f"source-built Windows artifacts are absent: {missing[0]}")

    for candidate in candidates:
        metadata = identity.verify_pe_product_metadata(candidate, expected_version=VERSION)
        assert metadata["ProductName"].strip() == "PuriPuly <3"
        assert metadata["ProductVersion"].strip() == VERSION


def test_source_built_executable_rejects_wrong_release_version() -> None:
    pytest.importorskip("pefile")
    candidate = ROOT / "dist" / "PuriPulyHeart" / "PuriPulyHeart.exe"
    if not candidate.is_file():
        pytest.skip("source-built Windows package is absent")

    with pytest.raises(RuntimeError, match="ProductVersion"):
        identity.verify_pe_product_metadata(candidate, expected_version="9.9.9")


def _write_soxr_fixture(repo: Path, package: Path) -> None:
    inputs = repo / "build" / "soxr-release-inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    source_bytes = b"fake-source-archive"
    bundle_path = inputs / "fake-source-bundle.zip"
    with zipfile.ZipFile(bundle_path, "w") as archive:
        archive.writestr(
            "manifest.json",
            json.dumps(
                {
                    "sources": [
                        {
                            "filename": "fake-source.tar.gz",
                            "sha256": hashlib.sha256(source_bytes).hexdigest(),
                            "modifications": ["fix.patch"],
                        }
                    ]
                }
            ),
        )
        archive.writestr("fake-source.tar.gz", source_bytes)
        archive.writestr("fix.patch", b"patch")
    (inputs / "manifest.json").write_text(
        json.dumps(
            {"third_party_source_bundle_path": "build/soxr-release-inputs/fake-source-bundle.zip"}
        ),
        encoding="utf-8",
    )
    license_dir = repo / "src" / "puripuly_heart" / "data" / "licenses"
    license_dir.mkdir(parents=True, exist_ok=True)
    (license_dir / "COPYING.LGPL-2.1.txt").write_bytes(b"fake-lgpl")
    compliance = package / "third_party" / "soxr"
    compliance.mkdir(parents=True, exist_ok=True)
    (compliance / "COPYING.LGPL-2.1.txt").write_bytes(b"fake-lgpl")
    (compliance / bundle_path.name).write_bytes(bundle_path.read_bytes())


def test_verify_packaged_licenses_accepts_matching_tree_and_bundle(tmp_path, monkeypatch) -> None:
    repo = tmp_path / "repo"
    package = tmp_path / "package"
    first = package / "nested" / "FIRST.txt"
    first.parent.mkdir(parents=True)
    first.write_bytes(b"first-license")
    second = package / "SECOND.txt"
    second.write_bytes(b"second-license")
    monkeypatch.setattr(
        identity,
        "PACKAGED_LICENSE_PAYLOADS",
        (
            ("nested\\FIRST.txt", hashlib.sha256(b"first-license").hexdigest()),
            ("SECOND.txt", hashlib.sha256(b"second-license").hexdigest()),
        ),
    )
    _write_soxr_fixture(repo, package)

    payloads = identity.verify_packaged_license_payloads(package)
    summary = identity.verify_soxr_packaging(package, repo)

    assert len(payloads) == 2
    assert summary["bundle"]["filename"] == "fake-source-bundle.zip"
    assert [source["filename"] for source in summary["bundle"]["sources"]] == ["fake-source.tar.gz"]
    assert summary["compliance"] == {
        "license": "COPYING.LGPL-2.1.txt",
        "bundle": "fake-source-bundle.zip",
    }


def test_verify_packaged_licenses_rejects_tampered_or_missing_payload(
    tmp_path, monkeypatch
) -> None:
    package = tmp_path / "package"
    package.mkdir()
    (package / "KEPT.txt").write_bytes(b"kept-license")
    monkeypatch.setattr(
        identity,
        "PACKAGED_LICENSE_PAYLOADS",
        (
            ("KEPT.txt", hashlib.sha256(b"kept-license").hexdigest()),
            ("MISSING.txt", hashlib.sha256(b"absent-license").hexdigest()),
        ),
    )

    with pytest.raises(RuntimeError, match="not found"):
        identity.verify_packaged_license_payloads(package)

    (package / "MISSING.txt").write_bytes(b"tampered-license")
    with pytest.raises(RuntimeError, match="sha256 mismatch"):
        identity.verify_packaged_license_payloads(package)


def test_verify_soxr_packaging_rejects_tampered_source_or_modification(tmp_path) -> None:
    repo = tmp_path / "repo"
    package = tmp_path / "package"
    package.mkdir()
    _write_soxr_fixture(repo, package)
    bundle_path = repo / "build" / "soxr-release-inputs" / "fake-source-bundle.zip"

    with zipfile.ZipFile(bundle_path) as archive:
        stored = {name: archive.read(name) for name in archive.namelist()}
    stored["fake-source.tar.gz"] = b"tampered-source-archive"
    with zipfile.ZipFile(bundle_path, "w") as archive:
        for name, payload in stored.items():
            archive.writestr(name, payload)
    with pytest.raises(RuntimeError, match="hash mismatch for"):
        identity.verify_soxr_packaging(package, repo)

    _write_soxr_fixture(repo, package)
    with zipfile.ZipFile(bundle_path, "a") as archive:
        archive.writestr("unlisted-extra.txt", b"extra")
    (package / "third_party" / "soxr" / bundle_path.name).write_bytes(bundle_path.read_bytes())
    identity.verify_soxr_packaging(package, repo)


def test_source_built_package_matches_recorded_license_and_source_bundle_provenance() -> None:
    package = ROOT / "dist" / "PuriPulyHeart"
    manifest = ROOT / "build" / "soxr-release-inputs" / "manifest.json"
    if not package.is_dir() or not manifest.is_file():
        pytest.skip("source-built Windows package is absent")

    payloads = identity.verify_packaged_license_payloads(package)
    summary = identity.verify_soxr_packaging(package, ROOT)

    assert len(payloads) == 26
    assert summary["bundle"]["filename"] == "PuriPulyHeart-soxr-third-party-source-bundle.zip"
