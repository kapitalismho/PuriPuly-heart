from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from puripuly_heart.release_evidence import release_identity as identity
from tests.helpers.native_pe import write_pe


@pytest.mark.parametrize(
    "version,product,error",
    [
        ("2.6.1.0", "Unrelated Product", "ProductName mismatch"),
        ("9.9.9.0", identity.PRODUCT_NAME, "binary ProductVersion mismatch"),
    ],
)
def test_rejects_wrong_pe_product_metadata(
    tmp_path: Path, version: str, product: str, error: str
) -> None:
    executable = write_pe(
        tmp_path / "PuriPulyHeart.exe",
        version=version,
        metadata={"ProductName": product, "ProductVersion": "2.6.1"},
    )
    with pytest.raises(RuntimeError, match=error):
        identity.verify_pe_product_metadata(executable, expected_version="2.6.1")


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
        "PACKAGED_LICENSE_PATHS",
        ("nested\\FIRST.txt", "SECOND.txt"),
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


def test_verify_packaged_licenses_rejects_missing_or_empty_payload(tmp_path, monkeypatch) -> None:
    package = tmp_path / "package"
    package.mkdir()
    (package / "KEPT.txt").write_bytes(b"kept-license")
    monkeypatch.setattr(
        identity,
        "PACKAGED_LICENSE_PATHS",
        ("KEPT.txt", "MISSING.txt"),
    )

    with pytest.raises(RuntimeError, match="not found"):
        identity.verify_packaged_license_payloads(package)

    (package / "MISSING.txt").write_bytes(b" \r\n\t")
    with pytest.raises(RuntimeError, match="is empty"):
        identity.verify_packaged_license_payloads(package)


def test_verify_packaged_licenses_rejects_path_resolving_outside_package(
    tmp_path, monkeypatch
) -> None:
    package = tmp_path / "package"
    package.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "LICENSE.txt").write_bytes(b"outside-license")
    linked = package / "linked"
    try:
        linked.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlinks are unavailable")
    monkeypatch.setattr(identity, "PACKAGED_LICENSE_PATHS", ("linked/LICENSE.txt",))

    with pytest.raises(RuntimeError, match="not found"):
        identity.verify_packaged_license_payloads(package)


@pytest.mark.parametrize("missing", ["fake-source.tar.gz", "fix.patch"])
def test_verify_soxr_packaging_rejects_missing_sources_or_modifications(
    tmp_path: Path, missing: str
) -> None:
    repo = tmp_path / "repo"
    package = tmp_path / "package"
    _write_soxr_fixture(repo, package)
    bundle_path = package / "third_party" / "soxr" / "fake-source-bundle.zip"
    with zipfile.ZipFile(bundle_path) as archive:
        stored = {name: archive.read(name) for name in archive.namelist() if name != missing}
    with zipfile.ZipFile(bundle_path, "w") as archive:
        for name, payload in stored.items():
            archive.writestr(name, payload)
    with pytest.raises(RuntimeError, match="missing"):
        identity.verify_soxr_packaging(package, repo)


def test_verify_soxr_packaging_rejects_missing_packaged_license(tmp_path) -> None:
    repo = tmp_path / "repo"
    package = tmp_path / "package"
    _write_soxr_fixture(repo, package)
    (package / "third_party" / "soxr" / "COPYING.LGPL-2.1.txt").unlink()

    with pytest.raises(RuntimeError):
        identity.verify_soxr_packaging(package, repo)
