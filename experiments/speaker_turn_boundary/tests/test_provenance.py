from __future__ import annotations

from pathlib import Path

from experiments.speaker_turn_boundary.provenance import (
    ERES_STANDARD_MODEL_ID,
    ERES_STANDARD_SHA256,
    ERES_W24_MODEL_ID,
    ERES_W24_SHA256,
    FS_EEND_REPO,
    LS_EEND_ONNX_REPO,
    all_artifacts,
    eres_artifacts,
    fs_eend_checkpoint_artifacts,
    ls_eend_onnx_artifacts,
    verify_artifact_file,
)


def test_ls_eend_variants_covered():
    artifacts = ls_eend_onnx_artifacts()
    kinds = [artifact.artifact_id.split(":")[0] for artifact in artifacts]
    for variant in ("L-AMI", "L-CALLHOME", "L-DIHARD-II", "L-DIHARD-III"):
        assert kinds.count(variant) == 2


def test_fs_eend_checkpoints_covered():
    artifacts = fs_eend_checkpoint_artifacts()
    names = [artifact.file_name for artifact in artifacts]
    assert names == ["ami.ckpt", "ch.ckpt", "dih2.ckpt", "dih3.ckpt"]


def test_eres_artifacts_official():
    artifacts = {artifact.artifact_id.split(":")[0]: artifact for artifact in eres_artifacts()}
    assert artifacts["E-standard"].source_url.endswith(ERES_STANDARD_MODEL_ID)
    assert artifacts["E-standard"].sha256 == ERES_STANDARD_SHA256
    assert artifacts["E-w24s4ep4"].source_url.endswith(ERES_W24_MODEL_ID)
    assert artifacts["E-w24s4ep4"].sha256 == ERES_W24_SHA256
    assert artifacts["E-standard"].sidecar["embedding_size"] == 192


def test_all_artifacts_consistent():
    artifacts = all_artifacts()
    identifiers = [artifact.artifact_id for artifact in artifacts]
    assert len(identifiers) == len(set(identifiers))
    for artifact in artifacts:
        assert len(artifact.sha256) == 64
        assert artifact.revision
        assert artifact.license
        assert artifact.size_bytes > 0


def test_provenance_gate_rejects_tampered(tmp_dir: Path):
    artifact = ls_eend_onnx_artifacts()[0]
    path = tmp_dir / "model.onnx"
    path.write_bytes(b"tampered")
    ok, reason = verify_artifact_file(artifact, path)
    assert ok is False
    assert "size mismatch" in reason


def test_provenance_gate_rejects_missing(tmp_dir: Path):
    artifact = ls_eend_onnx_artifacts()[0]
    ok, reason = verify_artifact_file(artifact, tmp_dir / "missing.onnx")
    assert ok is False
    assert reason == "missing"


def test_artifact_dict_roundtrip():
    artifact = ls_eend_onnx_artifacts()[0]
    data = artifact.to_dict()
    assert data["license"] == "mit"
    assert data["sha256"] == artifact.sha256
    assert "sidecar" not in data


def test_upstream_identities():
    assert FS_EEND_REPO == "https://github.com/Audio-WestlakeU/FS-EEND"
    assert LS_EEND_ONNX_REPO == "https://huggingface.co/GradientDescent2718/LS-EEND-ONNX"
