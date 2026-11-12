from __future__ import annotations

import uuid
from pathlib import Path

from puripuly_heart.core.local_asr.local_stt_runtime_installer import (
    cleanup_local_stt_install_residue,
)
from puripuly_heart.core.model_install_removal import remove_model_install_dir


def _deleting_residue(root: Path, dirname: str, payload: bytes = b"x" * 16) -> Path:
    trash = root / f"{dirname}.deleting-{uuid.uuid4().hex}"
    (trash / "nested").mkdir(parents=True)
    (trash / "nested" / "weights.bin").write_bytes(payload)
    return trash


def test_startup_cleanup_removes_interrupted_deletion_residue(tmp_path: Path) -> None:
    trash = _deleting_residue(tmp_path, "model")

    reconciled = cleanup_local_stt_install_residue(
        model_root=tmp_path,
        install_dirnames=("model",),
    )

    assert trash in reconciled
    assert not trash.exists()


def test_startup_cleanup_removes_silently_failed_deletion_residue(
    tmp_path: Path, monkeypatch
) -> None:
    install = tmp_path / "model"
    (install / "nested").mkdir(parents=True)
    (install / "nested" / "weights.bin").write_bytes(b"x")

    with monkeypatch.context() as patched:
        patched.setattr("shutil.rmtree", lambda *args, **kwargs: None)
        assert remove_model_install_dir(install) is True

    residue = [path for path in tmp_path.iterdir() if path.name.startswith("model.deleting-")]
    assert len(residue) == 1
    assert not install.exists()

    reconciled = cleanup_local_stt_install_residue(
        model_root=tmp_path,
        install_dirnames=("model",),
    )

    assert residue[0] in reconciled
    assert not residue[0].exists()


def test_startup_cleanup_keeps_unrelated_models_and_current_install(
    tmp_path: Path,
) -> None:
    own_trash = _deleting_residue(tmp_path, "model")
    other_trash = _deleting_residue(tmp_path, "other-model")
    install = tmp_path / "model"
    install.mkdir()
    (install / "weights.bin").write_bytes(b"live")

    cleanup_local_stt_install_residue(model_root=tmp_path, install_dirnames=("model",))

    assert not own_trash.exists()
    assert other_trash.exists()
    assert (install / "weights.bin").read_bytes() == b"live"


def test_startup_cleanup_still_restores_backup_and_removes_staging(
    tmp_path: Path,
) -> None:
    backup = tmp_path / "model.backup"
    staging = tmp_path / "model.staging-abcd"
    backup.mkdir()
    staging.mkdir()
    (backup / "weights.bin").write_bytes(b"ready")

    reconciled = cleanup_local_stt_install_residue(
        model_root=tmp_path,
        install_dirnames=("model",),
    )

    assert {path.name for path in reconciled} == {"model.backup", "model.staging-abcd"}
    assert (tmp_path / "model" / "weights.bin").read_bytes() == b"ready"
    assert not backup.exists()
    assert not staging.exists()


def test_removal_entry_point_still_reclaims_previous_residue(tmp_path: Path) -> None:
    install = tmp_path / "model"
    (install / "nested").mkdir(parents=True)
    (install / "nested" / "weights.bin").write_bytes(b"x")
    old_residue = tmp_path / "model.deleting-old"
    old_residue.mkdir()

    assert remove_model_install_dir(install) is True

    assert not install.exists()
    assert not old_residue.exists()
    assert list(tmp_path.iterdir()) == []
    assert remove_model_install_dir(install) is False
