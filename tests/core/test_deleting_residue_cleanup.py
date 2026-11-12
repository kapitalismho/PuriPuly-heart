from __future__ import annotations

import uuid
from pathlib import Path

import pytest

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


def test_startup_cleanup_removes_regular_file_deletion_residue(tmp_path: Path) -> None:
    trash = tmp_path / "model.deleting-file"
    trash.write_bytes(b"residue")

    reconciled = cleanup_local_stt_install_residue(
        model_root=tmp_path,
        install_dirnames=("model",),
    )

    assert trash in reconciled
    assert not trash.exists()


@pytest.mark.parametrize("target_exists", [True, False], ids=["directory", "dangling"])
def test_startup_cleanup_unlinks_deletion_symlinks_without_touching_targets(
    tmp_path: Path, target_exists: bool
) -> None:
    target = tmp_path / "external"
    target.mkdir()
    sentinel = target / "weights.bin"
    sentinel.write_bytes(b"preserve")
    link_target = target if target_exists else target / "missing"
    root = tmp_path / "models"
    root.mkdir()
    trash = root / "model.deleting-link"
    try:
        trash.symlink_to(link_target, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"symlink permission unavailable: {type(error).__name__}")

    reconciled = cleanup_local_stt_install_residue(
        model_root=root,
        install_dirnames=("model",),
    )

    assert trash in reconciled
    assert not trash.is_symlink()
    assert not trash.exists()
    assert sentinel.read_bytes() == b"preserve"
    if not target_exists:
        assert not link_target.exists()
