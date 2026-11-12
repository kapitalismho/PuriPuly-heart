from __future__ import annotations

import uuid
from pathlib import Path

import pytest

from puripuly_heart.app.services.managed_gemma_translation import (
    ManagedGemmaTranslationOwner,
)
from puripuly_heart.core.local_asr.local_stt_runtime_installer import (
    LocalSTTProvisioningLease,
)
from puripuly_heart.core.local_translation import assets as gemma_assets

GEMMA_E4B = gemma_assets.GEMMA_MODEL_ID
GEMMA_26B = gemma_assets.GEMMA_26B_MODEL_ID


class _StubRuntime:
    async def release(self) -> None:
        return None


def _owner() -> ManagedGemmaTranslationOwner:
    return ManagedGemmaTranslationOwner(runtime=_StubRuntime())  # type: ignore[arg-type]


def _residue(root: Path, dirname: str, suffix: str, *, payload: bytes = b"x" * 16) -> Path:
    residue = root / f"{dirname}.{suffix}-{uuid.uuid4().hex}"
    (residue / "nested").mkdir(parents=True)
    (residue / "nested" / "weights.bin").write_bytes(payload)
    return residue


def _install(root: Path, dirname: str) -> Path:
    install = root / dirname
    (install / "nested").mkdir(parents=True)
    (install / "nested" / "weights.bin").write_bytes(b"live")
    return install


@pytest.fixture()
def models_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "models"
    root.mkdir()
    monkeypatch.setattr(gemma_assets, "default_models_dir", lambda: root)
    return root


def test_interrupted_delete_residue_is_reconciled_on_next_inspect(
    models_root: Path,
) -> None:
    trash = _residue(models_root, GEMMA_E4B, "deleting")

    owner = _owner()
    status = owner.install_status(GEMMA_E4B)

    assert status == "missing"
    assert not trash.exists()


def test_interrupted_provisioning_and_promote_residue_is_reconciled(
    models_root: Path,
) -> None:
    staging = _residue(models_root, GEMMA_26B, "staging")
    backup = _residue(models_root, GEMMA_26B, "backup")

    owner = _owner()
    _ = owner.install_status(GEMMA_26B)

    assert not staging.exists()
    assert not backup.exists()


def test_residue_reconcile_keeps_current_install_and_unrelated_content(
    models_root: Path,
) -> None:
    own_trash = _residue(models_root, GEMMA_E4B, "deleting")
    own_staging = _residue(models_root, GEMMA_E4B, "staging")
    other_trash = _residue(models_root, "other-model", "deleting")
    install = _install(models_root, GEMMA_E4B)
    plain_file_residue = models_root / f"{GEMMA_E4B}.deleting-file"
    plain_file_residue.write_bytes(b"residue")
    near_miss = models_root / f"{GEMMA_E4B}.stagingx-keep"
    near_miss.mkdir()
    keeper = models_root / "keep-me.txt"
    keeper.write_text("notes")

    owner = _owner()
    _ = owner.install_status(GEMMA_E4B)

    assert not own_trash.exists()
    assert not own_staging.exists()
    assert not plain_file_residue.exists()
    assert other_trash.exists()
    assert near_miss.is_dir()
    assert keeper.read_text() == "notes"
    assert (install / "nested" / "weights.bin").read_bytes() == b"live"


def test_residue_reconcile_retries_after_provisioning_lease_is_released(
    models_root: Path,
) -> None:
    trash = _residue(models_root, GEMMA_E4B, "deleting")
    owner = _owner()

    lease = LocalSTTProvisioningLease.acquire(model_root=models_root, wait=False)
    assert lease is not None
    try:
        _ = owner.install_status(GEMMA_E4B)
        assert trash.exists()
    finally:
        lease.close()

    _ = owner.install_status(GEMMA_E4B)
    assert not trash.exists()


def test_delete_assets_reconciles_preexisting_staging_residue(models_root: Path) -> None:
    install = _install(models_root, GEMMA_E4B)
    staging = _residue(models_root, GEMMA_E4B, "staging")
    owner = _owner()

    import asyncio

    asyncio.run(owner.delete_assets(GEMMA_E4B))

    assert not install.exists()
    assert not staging.exists()


def test_cleanup_function_removes_all_three_patterns_and_reports_paths(
    models_root: Path,
) -> None:
    staging = _residue(models_root, GEMMA_E4B, "staging")
    backup = _residue(models_root, GEMMA_E4B, "backup")
    trash = _residue(models_root, GEMMA_E4B, "deleting")

    removed = gemma_assets.cleanup_gemma_install_residue(
        install_dirnames=(GEMMA_E4B,),
    )

    assert removed is not None
    assert set(removed) == {staging, backup, trash}
    assert not staging.exists()
    assert not backup.exists()
    assert not trash.exists()


def test_cleanup_function_is_noop_when_models_root_is_missing(tmp_path: Path) -> None:
    missing = tmp_path / "absent-models"

    removed = gemma_assets.cleanup_gemma_install_residue(models_dir=missing)

    assert removed == ()
    assert not missing.exists()


def test_cleanup_function_rejects_unsafe_dirnames(models_root: Path) -> None:
    for bad in (("",), (".hidden",), ("../escape",), ("a/b",)):
        with pytest.raises(ValueError):
            gemma_assets.cleanup_gemma_install_residue(install_dirnames=bad)
