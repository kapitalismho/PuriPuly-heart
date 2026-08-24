from __future__ import annotations

import hashlib
import json

import pytest

from puripuly_heart.core.local_translation import assets


def _asset(filename: str, content: bytes) -> assets.GemmaAsset:
    return assets.GemmaAsset(
        filename=filename,
        size_bytes=len(content),
        sha256=hashlib.sha256(content).hexdigest(),
    )


def _write_install(tmp_path, monkeypatch, *, corrupt: bool = False):
    target = b"target"
    draft = b"draft"
    pinned = (_asset("target.gguf", target), _asset("draft.gguf", draft))
    monkeypatch.setattr(assets, "GEMMA_ASSETS", pinned)
    tmp_path.mkdir(exist_ok=True)
    (tmp_path / "target.gguf").write_bytes(b"broken" if corrupt else target)
    (tmp_path / "draft.gguf").write_bytes(draft)
    (tmp_path / assets.GEMMA_INSTALLED_MANIFEST_FILENAME).write_text(
        json.dumps(assets.InstalledGemmaManifest.expected().to_dict()),
        encoding="utf-8",
    )
    return pinned


def test_pinned_gemma_contract_has_exact_target_and_drafter() -> None:
    assert assets.GEMMA_REPO_ID == "unsloth/gemma-4-E4B-it-qat-GGUF"
    assert assets.GEMMA_REVISION == "8c5a9e4fd5482e2be20fe0bf013b4c262a8f4265"
    assert assets.GEMMA_UPSTREAM_REPO_ID == "google/gemma-4-E4B-it"
    assert assets.GEMMA_LICENSE == "Apache-2.0"
    assert assets.GEMMA_LICENSE_URL == "https://www.apache.org/licenses/LICENSE-2.0"
    assert assets.InstalledGemmaManifest.expected().upstream_repo_id == "google/gemma-4-E4B-it"
    assert assets.InstalledGemmaManifest.expected().license == "Apache-2.0"
    assert assets.InstalledGemmaManifest.expected().license_url == (
        "https://www.apache.org/licenses/LICENSE-2.0"
    )
    assert [(item.filename, item.size_bytes, item.sha256) for item in assets.GEMMA_ASSETS] == [
        (
            "gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf",
            4_215_695_776,
            "df0fd4ee07072c607c29a0a1cb4f98918426cca12f45a2776bdd6ee6d09a4de3",
        ),
        (
            "mtp-gemma-4-E4B-it.gguf",
            59_678_016,
            "423074e537504b4f9ec5eafed5c639fac82c96631626efccacdd3c4039b20605",
        ),
    ]


def test_pinned_gemma_12b_contract_has_target_without_drafter() -> None:
    assert assets.GEMMA_12B_REPO_ID == "unsloth/gemma-4-12B-it-qat-GGUF"
    assert assets.GEMMA_12B_REVISION == "980b060c40a8539ac159e0501a3e0f66a6365af3"
    assert assets.GEMMA_12B_UPSTREAM_REPO_ID == "google/gemma-4-12B-it"
    assert assets.GEMMA_12B_SPEC.draft_filename is None
    assert assets.InstalledGemmaManifest.expected(assets.GEMMA_12B_SPEC).files == (
        "gemma-4-12B-it-qat-UD-Q4_K_XL.gguf",
    )
    assert [(item.filename, item.size_bytes, item.sha256) for item in assets.GEMMA_12B_ASSETS] == [
        (
            "gemma-4-12B-it-qat-UD-Q4_K_XL.gguf",
            6_716_356_800,
            "90fd44e29e0d7cffeb0fd00dc73cfdab9ed0b0e95306ecf7821ea634c940c370",
        ),
    ]


def test_full_validation_rejects_checksum_mismatch(tmp_path, monkeypatch) -> None:
    _write_install(tmp_path, monkeypatch, corrupt=True)

    with pytest.raises(assets.GemmaInstallInvalidError, match="checksum mismatch"):
        assets.validate_gemma_install(tmp_path)


def test_inspection_reports_ready_without_hashing_valid_sized_assets(tmp_path, monkeypatch) -> None:
    _write_install(tmp_path, monkeypatch)

    state = assets.inspect_gemma_install(tmp_path)

    assert state.status == "ready"
    assert state.manifest == assets.InstalledGemmaManifest.expected()
