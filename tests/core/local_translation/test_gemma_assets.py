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


def test_full_validation_rejects_checksum_mismatch(tmp_path, monkeypatch) -> None:
    _write_install(tmp_path, monkeypatch, corrupt=True)

    with pytest.raises(assets.GemmaInstallInvalidError, match="checksum mismatch"):
        assets.validate_gemma_install(tmp_path)


def test_inspection_reports_ready_without_hashing_valid_sized_assets(tmp_path, monkeypatch) -> None:
    _write_install(tmp_path, monkeypatch)

    state = assets.inspect_gemma_install(tmp_path)

    assert state.status == "ready"
    assert state.manifest == assets.InstalledGemmaManifest.expected()


def test_retired_managed_gemma_install_and_leftovers_are_removed(tmp_path) -> None:
    models_dir = tmp_path / "models"
    retired_id, retired_filename = assets.RETIRED_MANAGED_GEMMA_INSTALLS[0]
    retired_install = models_dir / retired_id
    retired_install.mkdir(parents=True)
    (retired_install / retired_filename).write_bytes(b"retired")
    staging = models_dir / f"{retired_id}.staging-deadbeef"
    staging.mkdir()
    (staging / retired_filename).write_bytes(b"partial")
    backup = models_dir / f"{retired_id}.backup-cafe"
    backup.mkdir()
    current_install = models_dir / assets.GEMMA_INSTALL_DIRNAME
    current_install.mkdir()
    (current_install / assets.GEMMA_MODEL_FILENAME).write_bytes(b"current")
    unrelated = models_dir / "qwen3-asr-0.6b-int8-sherpa"
    unrelated.mkdir()

    removed = assets.remove_retired_managed_gemma_installs(models_dir)

    assert set(removed) == {retired_install, staging, backup}
    assert not retired_install.exists()
    assert not staging.exists()
    assert not backup.exists()
    assert (current_install / assets.GEMMA_MODEL_FILENAME).read_bytes() == b"current"
    assert unrelated.is_dir()
    assert assets.remove_retired_managed_gemma_installs(models_dir) == ()


def test_retired_install_directory_without_retired_payload_is_kept(tmp_path) -> None:
    models_dir = tmp_path / "models"
    retired_id, _retired_filename = assets.RETIRED_MANAGED_GEMMA_INSTALLS[0]
    unrelated_content = models_dir / retired_id
    unrelated_content.mkdir(parents=True)
    (unrelated_content / "notes.txt").write_text("kept", encoding="utf-8")

    assert assets.remove_retired_managed_gemma_installs(models_dir) == ()
    assert (unrelated_content / "notes.txt").is_file()

    (unrelated_content / "notes.txt").unlink()
    (unrelated_content / assets.GEMMA_INSTALLED_MANIFEST_FILENAME).write_text(
        json.dumps({"model_id": retired_id}),
        encoding="utf-8",
    )

    assert assets.remove_retired_managed_gemma_installs(models_dir) == (unrelated_content,)
    assert not unrelated_content.exists()


def test_retired_install_removal_reports_failures_and_keeps_sweeping(tmp_path, monkeypatch) -> None:
    models_dir = tmp_path / "models"
    retired_id, retired_filename = assets.RETIRED_MANAGED_GEMMA_INSTALLS[0]
    install_dir = models_dir / retired_id
    install_dir.mkdir(parents=True)
    (install_dir / retired_filename).write_bytes(b"retired")
    staging = models_dir / f"{retired_id}.staging-deadbeef"
    staging.mkdir()
    failures: list[tuple[object, OSError]] = []
    real_rmtree = assets.shutil.rmtree

    def flaky_rmtree(path, *args, **kwargs):
        if str(path).endswith(staging.name):
            raise OSError("locked")
        return real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(assets.shutil, "rmtree", flaky_rmtree)

    removed = assets.remove_retired_managed_gemma_installs(
        models_dir,
        on_failure=lambda path, exc: failures.append((path, exc)),
    )

    assert removed == (install_dir,)
    assert not install_dir.exists()
    assert staging.is_dir()
    assert len(failures) == 1
    assert failures[0][0] == staging
    assert isinstance(failures[0][1], OSError)
