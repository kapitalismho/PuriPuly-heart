import copy
import hashlib

import pytest

from experiments.psem_sortformer_adaptation_depth import preflight
from experiments.psem_sortformer_adaptation_depth.preflight import (
    build_preflight,
    resolve_paths,
)


def test_material_execution_guard_fails_closed(monkeypatch) -> None:
    with pytest.raises(preflight.PreflightError, match="blocked_pending_lean_runner_alignment"):
        preflight.require_material_execution_ready()

    monkeypatch.setattr(preflight, "load_json", lambda _path: {})
    with pytest.raises(preflight.PreflightError, match="unavailable"):
        preflight.require_material_execution_ready()

    monkeypatch.setattr(
        preflight,
        "load_json",
        lambda _path: {
            "material_execution": {
                "status": "ready",
                "required_status_for_material_execution": "ready",
            }
        },
    )
    preflight.require_material_execution_ready()


def test_static_preflight_passes_without_external_assets(monkeypatch) -> None:
    monkeypatch.setattr(preflight, "_git_state", lambda: {"head": "a" * 40, "dirty": []})
    receipt = build_preflight(resolve_paths(), static_only=True)
    assert receipt["static_contract_valid"] is True
    assert receipt["ready_for_runtime_audit"] is False


def test_runtime_preflight_fails_closed_without_external_assets(monkeypatch) -> None:
    monkeypatch.setattr(preflight, "_git_state", lambda: {"head": "a" * 40, "dirty": []})
    for name in (
        "PSEM_SORTFORMER_NEMO_PATH",
        "PSEM_CORPUS_ROOT",
        "PSEM_REFERENCE_ROOT",
        "PSEM_ADAPTATION_OUTPUT_ROOT",
        "PSEM_ALLOW_EVAL",
    ):
        monkeypatch.delenv(name, raising=False)
    receipt = build_preflight(resolve_paths())
    assert receipt["ready_for_runtime_audit"] is False
    failed = {row["id"] for row in receipt["checks"] if not row["passed"]}
    assert "runtime.material_execution_authorized" in failed
    assert "runtime.checkpoint_path" in failed
    assert "runtime.corpus_root" in failed
    assert "runtime.reference_root" in failed
    assert "runtime.output_root" in failed


def test_whole_config_binding_rejects_mutated_control(monkeypatch) -> None:
    original = preflight.load_json

    def mutated(path):
        value = original(path)
        if path == preflight.CONFIG_PATH:
            value = copy.deepcopy(value)
            value["evaluation"]["eval_threshold_selection_allowed"] = True
        return value

    monkeypatch.setattr(preflight, "load_json", mutated)
    checks = {row["id"]: row for row in preflight.static_checks()}
    assert checks["config.controls_exact"]["passed"] is False


def test_unknown_split_role_is_rejected(monkeypatch) -> None:
    original = preflight.load_json
    split_path = preflight.REPOSITORY_ROOT / preflight.EXPECTED_ARTIFACTS["split_manifest"][0]

    def mutated(path):
        value = original(path)
        if path == split_path:
            value = copy.deepcopy(value)
            value["assignments"]["sources"][0]["role"] = "UNKNOWN"
        return value

    monkeypatch.setattr(preflight, "load_json", mutated)
    checks = {row["id"]: row for row in preflight.static_checks()}
    assert checks["dataset.split_roles_exact"]["passed"] is False


def test_dirty_worktree_is_rejected(monkeypatch) -> None:
    monkeypatch.setattr(
        preflight,
        "_git_state",
        lambda: {"head": "a" * 40, "dirty": [" M experiments/example.py"]},
    )
    receipt = build_preflight(resolve_paths(), static_only=True)
    checks = {row["id"]: row for row in receipt["checks"]}
    assert checks["git.worktree_clean"]["passed"] is False
    assert receipt["static_contract_valid"] is False


def test_output_root_must_be_existing_and_external(tmp_path) -> None:
    external = tmp_path / "output"
    external.mkdir()
    file_path = tmp_path / "file"
    file_path.write_text("not a directory", encoding="utf-8")
    assert preflight._safe_external_output_root(external)
    assert not preflight._safe_external_output_root(file_path)
    assert not preflight._safe_external_output_root(preflight.REPOSITORY_ROOT)
    assert not preflight._safe_external_output_root(preflight.PACKAGE_ROOT)


def test_waveform_bytes_are_verified(monkeypatch, tmp_path) -> None:
    audio = tmp_path / "corpus" / "audio.wav"
    audio.parent.mkdir()
    audio.write_bytes(b"wrong")
    monkeypatch.setattr(preflight, "EXPECTED_SOURCE_COUNT", 1)
    monkeypatch.setattr(
        preflight,
        "_source_rows",
        lambda: [
            {
                "source_id": "source-1",
                "audio_ref": "audio.wav",
                "waveform_size_bytes": 5,
                "waveform_sha256": hashlib.sha256(b"right").hexdigest(),
            }
        ],
    )
    result = preflight._bound_waveform_check(audio.parent)
    assert result["passed"] is False
    assert result["observed"]["failures"] == ["source-1"]


def test_reference_checkout_provenance_is_exact(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        preflight,
        "validate_reference_checkout",
        lambda _: {**preflight.EXPECTED_REFERENCE, "commit": "0" * 40},
    )
    result = preflight._reference_check(tmp_path)
    assert result["passed"] is False
