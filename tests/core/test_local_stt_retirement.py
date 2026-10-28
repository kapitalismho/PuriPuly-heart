from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from uuid import uuid4

import pytest

from puripuly_heart.core.local_asr import retired_cpu_assets as retirement_module
from puripuly_heart.core.local_asr.local_stt_assets import (
    FUN_ASR_MLT_NANO_MODEL_ID,
    FUN_ASR_NANO_MODEL_ID,
    LOCAL_QWEN_GPU_MODEL_ID,
    ORUKEET_MODEL_ID,
    PARAKEET_JAPANESE_MODEL_ID,
)
from puripuly_heart.core.local_asr.local_stt_runtime_installer import LocalSTTProvisioningLease
from puripuly_heart.core.local_asr.retired_cpu_assets import cleanup_retired_cpu_asr_assets

OLD_QWEN = "qwen3-asr-0.6b-int8-sherpa"
OLD_PARAKEET = "parakeet-tdt-0.6b-v3-int8-sherpa"


def _seed(root: Path, model_id: str, suffix: str = "") -> Path:
    directory = root / f"{model_id}{suffix}"
    directory.mkdir(parents=True)
    (directory / "installed-manifest.json").write_text(
        json.dumps(
            {
                "manifest_version": 1,
                "model_id": model_id,
                "engine": "sherpa-onnx",
                "install_dirname": model_id,
                "selected_source": "huggingface",
                "selected_revision": "prior-install",
            }
        ),
        encoding="utf-8",
    )
    (directory / "decoder.int8.onnx").write_bytes(b"synthetic-retired-payload")
    return directory


def test_removes_recognized_main_backup_staging_deleting_without_restore(tmp_path) -> None:
    directories = [
        _seed(tmp_path, model_id, suffix)
        for model_id in (OLD_QWEN, OLD_PARAKEET)
        for suffix in ("", ".backup", f".staging-{uuid4().hex}", f".deleting-{uuid4().hex}")
    ]
    protected = {}
    for name in (
        PARAKEET_JAPANESE_MODEL_ID,
        LOCAL_QWEN_GPU_MODEL_ID,
        ORUKEET_MODEL_ID,
        FUN_ASR_NANO_MODEL_ID,
        FUN_ASR_MLT_NANO_MODEL_ID,
        "unrelated",
    ):
        directory = tmp_path / name
        directory.mkdir()
        sentinel = directory / "sentinel"
        sentinel.write_bytes(name.encode())
        protected[sentinel] = sentinel.read_bytes()
    backup_only_root = tmp_path / "backup-only"
    _seed(backup_only_root, OLD_QWEN, ".backup")
    result = cleanup_retired_cpu_asr_assets(model_root=tmp_path)
    assert result.removed_count == 8
    assert result.pending_count == 0
    assert result.error_codes == ()
    assert all(not directory.exists() for directory in directories)
    assert all(path.read_bytes() == payload for path, payload in protected.items())
    second = cleanup_retired_cpu_asr_assets(model_root=backup_only_root)
    assert second.removed_count == 1
    assert not (backup_only_root / OLD_QWEN).exists()


def test_unknown_files_and_unrecognized_directories_are_protected(tmp_path) -> None:
    unknown_neighbor = _seed(tmp_path, OLD_QWEN)
    (unknown_neighbor / "user-recording.wav").write_bytes(b"private")
    unrecognized = tmp_path / OLD_PARAKEET
    unrecognized.mkdir()
    (unrecognized / "decoder.int8.onnx").write_bytes(b"not-a-known-model")
    unrelated_residue = _seed(tmp_path, OLD_QWEN, ".staging-user-notes")
    result = cleanup_retired_cpu_asr_assets(model_root=tmp_path)
    assert result.removed_count == 0
    assert result.pending_count == 2
    assert result.error_codes == ("unrecognized_contents",)
    assert (unknown_neighbor / "decoder.int8.onnx").exists()
    assert (unknown_neighbor / "user-recording.wav").read_bytes() == b"private"
    assert unrecognized.exists() and unrelated_residue.exists()


def test_partial_delete_failure_stays_recognized_and_retryable(tmp_path, monkeypatch) -> None:
    directory = _seed(tmp_path, OLD_QWEN)
    original = Path.unlink
    failed_payload = directory / "decoder.int8.onnx"

    def fail_payload(path, *args, **kwargs):
        if path == failed_payload:
            raise PermissionError("private-path-must-not-appear")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_payload)
    failed = cleanup_retired_cpu_asr_assets(model_root=tmp_path)
    assert failed.removed_count == 0
    assert failed.pending_count == 1
    assert failed.error_codes == ("delete_failed",)
    assert (directory / ".retirement-pending.json").exists()
    assert "private" not in repr(failed)
    monkeypatch.setattr(Path, "unlink", original)
    retried = cleanup_retired_cpu_asr_assets(model_root=tmp_path)
    assert retried.removed_count == 1
    assert retried.pending_count == 0
    assert not directory.exists()
    assert cleanup_retired_cpu_asr_assets(model_root=tmp_path).removed_count == 0


def test_final_directory_removal_failure_retains_retry_identity(tmp_path, monkeypatch) -> None:
    directory = _seed(tmp_path, OLD_PARAKEET)
    original = Path.rmdir

    def fail_final(path):
        if path == directory:
            raise PermissionError("in-use")
        return original(path)

    monkeypatch.setattr(Path, "rmdir", fail_final)
    failed = cleanup_retired_cpu_asr_assets(model_root=tmp_path)
    assert failed.pending_count == 1
    assert (directory / ".retirement-pending.json").exists()
    monkeypatch.setattr(Path, "rmdir", original)
    assert cleanup_retired_cpu_asr_assets(model_root=tmp_path).removed_count == 1


def test_cleanup_respects_provisioning_lease(tmp_path) -> None:
    directory = _seed(tmp_path, OLD_QWEN)
    lease = LocalSTTProvisioningLease.acquire(model_root=tmp_path, wait=False)
    assert lease is not None
    try:
        result = cleanup_retired_cpu_asr_assets(model_root=tmp_path)
        assert result.removed_count == 0
        assert result.pending_count == 1
        assert result.error_codes == ("lease_busy",)
        assert directory.exists()
    finally:
        lease.close()
    assert cleanup_retired_cpu_asr_assets(model_root=tmp_path).removed_count == 1


@pytest.mark.parametrize("position", ["model", "payload", "root"])
def test_symbolic_links_never_follow_or_remove_targets(tmp_path, position) -> None:
    target = tmp_path / "external"
    target.mkdir()
    sentinel = target / "decoder.int8.onnx"
    sentinel.write_bytes(b"preserve")
    root = tmp_path / "models"
    root.mkdir()
    try:
        if position == "model":
            (root / OLD_QWEN).symlink_to(target, target_is_directory=True)
        elif position == "payload":
            directory = _seed(root, OLD_QWEN)
            (directory / "decoder.int8.onnx").unlink()
            (directory / "decoder.int8.onnx").symlink_to(sentinel)
        else:
            root.rmdir()
            root.symlink_to(target, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"symlink permission unavailable: {type(error).__name__}")
    result = cleanup_retired_cpu_asr_assets(model_root=root)
    assert result.removed_count == 0
    assert result.pending_count == 1
    assert result.error_codes == ("protected_reparse_point",)
    assert sentinel.read_bytes() == b"preserve"


@pytest.mark.skipif(os.name != "nt", reason="Windows junction semantics")
def test_windows_junction_is_protected_without_following_target(tmp_path) -> None:
    root = tmp_path / "models"
    root.mkdir()
    target = _seed(tmp_path / "external", OLD_QWEN)
    link = root / OLD_QWEN
    subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(link), str(target)], check=True, capture_output=True
    )
    try:
        result = cleanup_retired_cpu_asr_assets(model_root=root)
        assert result.removed_count == 0
        assert result.pending_count == 1
        assert result.error_codes == ("protected_reparse_point",)
        assert (target / "decoder.int8.onnx").read_bytes() == b"synthetic-retired-payload"
    finally:
        link.rmdir()


def test_payload_only_staging_requires_actual_matching_bytes(tmp_path, monkeypatch) -> None:
    payload = b"known-export-payload"
    monkeypatch.setitem(
        retirement_module._RETIRED_PAYLOADS,
        OLD_QWEN,
        {"decoder.int8.onnx": (len(payload), hashlib.sha256(payload).hexdigest())},
    )
    matching = tmp_path / f"{OLD_QWEN}.staging-{uuid4().hex}"
    matching.mkdir()
    (matching / "decoder.int8.onnx").write_bytes(payload)
    unknown = tmp_path / f"{OLD_QWEN}.staging-{uuid4().hex}"
    unknown.mkdir()
    (unknown / "decoder.int8.onnx").write_bytes(b"x" * len(payload))
    result = cleanup_retired_cpu_asr_assets(model_root=tmp_path)
    assert result.removed_count == 1
    assert result.pending_count == 1
    assert not matching.exists()
    assert (unknown / "decoder.int8.onnx").read_bytes() == b"x" * len(payload)


@pytest.mark.parametrize(
    ("model_id", "source"),
    [
        (OLD_QWEN, "huggingface"),
        (OLD_QWEN, "modelscope"),
        (OLD_PARAKEET, "huggingface"),
        (OLD_PARAKEET, "hf_mirror"),
    ],
)
def test_exact_old_setup_source_staging_is_cleaned_but_unknown_contents_survive(
    tmp_path, model_id, source
):
    recognized = _seed(tmp_path / "recognized", model_id, f".staging-{source}")
    protected = _seed(tmp_path / "protected", model_id, f".staging-{source}")
    (protected / "user-file").write_bytes(b"preserve")
    cleaned = cleanup_retired_cpu_asr_assets(model_root=recognized.parent)
    unknown = cleanup_retired_cpu_asr_assets(model_root=protected.parent)
    assert cleaned.removed_count == 1 and cleaned.pending_count == 0
    assert not recognized.exists()
    assert unknown.removed_count == 0 and unknown.pending_count == 1
    assert unknown.error_codes == ("unrecognized_contents",)
    assert (protected / "user-file").read_bytes() == b"preserve"


@pytest.mark.parametrize(
    ("model_id", "source"),
    [(OLD_QWEN, "hf_mirror"), (OLD_PARAKEET, "modelscope"), (OLD_QWEN, "user-notes")],
)
def test_never_produced_old_setup_source_names_remain_unrecognized(tmp_path, model_id, source):
    candidate = _seed(tmp_path, model_id, f".staging-{source}")
    result = cleanup_retired_cpu_asr_assets(model_root=tmp_path)
    assert result.removed_count == 0
    assert candidate.exists()


def test_old_setup_source_staging_partial_failure_remains_retryable(tmp_path, monkeypatch):
    directory = _seed(tmp_path, OLD_QWEN, ".staging-modelscope")
    original = Path.unlink

    def fail_payload(path, *args, **kwargs):
        if path == directory / "decoder.int8.onnx":
            raise PermissionError("controlled source-staging lock")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_payload)
    failed = cleanup_retired_cpu_asr_assets(model_root=tmp_path)
    assert failed.removed_count == 0 and failed.pending_count == 1
    assert failed.error_codes == ("delete_failed",)
    assert (directory / ".retirement-pending.json").exists()
    monkeypatch.setattr(Path, "unlink", original)
    retried = cleanup_retired_cpu_asr_assets(model_root=tmp_path)
    assert retried.removed_count == 1 and retried.pending_count == 0
    assert not directory.exists()
