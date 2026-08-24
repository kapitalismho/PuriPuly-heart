from __future__ import annotations

import json
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path

import pytest

from experiments.psem_training_strategy_gate import preflight as preflight_module
from experiments.psem_training_strategy_gate import run as run_module
from experiments.psem_training_strategy_gate.preflight import (
    AUTHORITY,
    BINDING_KEYS,
    CONFIG_PATH,
    CONTRACT_PATH,
    CONTRACT_VERSION,
    EXPECTED_CHECK_IDS,
    EXPERIMENT_ID,
    ExperimentPreflightError,
    PreflightPaths,
    _model_checks,
    _source_checks,
    _static_contract_checks,
    build_preflight,
    canonical_sha256,
    load_json,
    require_passing_preflight,
    resolve_paths,
)


def _receipt(output_root: Path, *, ready: bool = True) -> dict:
    roots = {
        name: str((output_root.parent / name).resolve())
        for name in ("cache_root", "corpus_root", "reference_root")
    }
    roots["output_root"] = str(output_root.resolve())
    binding = {key: "1" * 64 for key in BINDING_KEYS}
    binding["experiment_id"] = EXPERIMENT_ID
    binding["git_commit"] = "a" * 40
    checks = [
        {"id": check_id, "passed": True, "expected": True, "observed": True}
        for check_id in EXPECTED_CHECK_IDS
    ]
    if not ready:
        checks[0]["passed"] = False
    payload = {
        "schema_version": 1,
        "artifact_role": "psem_experiment_preflight",
        "experiment_id": EXPERIMENT_ID,
        "contract_version": CONTRACT_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "authority": AUTHORITY,
        "binding": binding,
        "git": {"commit": "a" * 40, "dirty": False, "dirty_paths": []},
        "paths": roots,
        "checks": checks,
        "failed_checks": [] if ready else [checks[0]["id"]],
        "ready_for_material_run": ready,
    }
    return {**payload, "payload_sha256": canonical_sha256(payload)}


def _write_receipt(output_root: Path, receipt: dict) -> Path:
    path = output_root / "preflight" / "experiment_receipt.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(receipt), encoding="utf-8")
    return path


def test_scientific_contract_controls_are_exact_and_full_document_pinned() -> None:
    contract = load_json(CONTRACT_PATH)
    config = load_json(CONFIG_PATH)
    check = _static_contract_checks(contract, config)[0]
    assert check["passed"] is True
    assert check["expected"]["contract_canonical_sha256"] == (
        "4b4f6a9dfbdf3c9c0c7ce85b210cc1a405d309b8d88b829d9655e0005642e1d0"
    )
    assert check["expected"]["config_canonical_sha256"] == (
        "3faf132c4df56e77651583fe3de292d52e14bab2fa2e5b2a2e177235c5fb28d2"
    )


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("common_head", "dropout", 0.2),
        ("sampling", "handoff_positive_fraction", 0.3),
        ("optimization", "optimizer", "SGD"),
        ("evaluation", "thresholds", "fixed"),
        ("pretrained_checkpoint", "model_id", "mhubert-147"),
    ],
)
def test_any_required_control_drift_fails_static_gate(
    section: str,
    key: str,
    value,
) -> None:
    config = deepcopy(load_json(CONFIG_PATH))
    config[section][key] = value
    assert _static_contract_checks(load_json(CONTRACT_PATH), config)[0]["passed"] is False


def test_arm_whitelist_and_augmentation_drift_fail_static_gate() -> None:
    config = deepcopy(load_json(CONFIG_PATH))
    config["arms"]["FINETUNE-WAVLM"]["trainable_transformer_layers"] = [11]
    assert _static_contract_checks(load_json(CONTRACT_PATH), config)[0]["passed"] is False
    config = deepcopy(load_json(CONFIG_PATH))
    config["augmentation"].append("speed_perturbation")
    assert _static_contract_checks(load_json(CONTRACT_PATH), config)[0]["passed"] is False


def test_preflight_fails_closed_without_runtime_receipts() -> None:
    receipt = build_preflight(
        PreflightPaths(None, None, None, None),
        verify_source_bytes=False,
    )
    assert receipt["ready_for_material_run"] is False
    assert tuple(row["id"] for row in receipt["checks"]) == EXPECTED_CHECK_IDS
    assert "paths.roots_safe" in receipt["failed_checks"]
    assert "model.wavlm_checkpoint_files_exact" in receipt["failed_checks"]
    assert "sources.bound_waveforms_resolve" in receipt["failed_checks"]
    assert "sources.byte_identity_verification_enabled" in receipt["failed_checks"]
    assert "runtime_receipt.gradient_canary" in receipt["failed_checks"]
    assert "runtime_receipt.weight_update_canary" in receipt["failed_checks"]


def test_malformed_source_and_reference_are_machine_rejections(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = tmp_path / "source_manifest.jsonl"
    manifest.write_text("{not-json}\n", encoding="utf-8")
    monkeypatch.setattr(preflight_module, "SOURCE_MANIFEST_PATH", manifest)
    checks = _source_checks(tmp_path, tmp_path, verify_source_bytes=True)
    assert [row["id"] for row in checks] == [
        "sources.manifest_identity",
        "sources.bound_waveforms_resolve",
        "sources.byte_identity_verification_enabled",
        "sources.forced_alignment_reference_exact",
    ]
    assert checks[1]["passed"] is False
    assert checks[3]["passed"] is False
    assert "error_type" in checks[3]["observed"]


def test_malformed_or_traversing_model_registry_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps(
            {
                "models": [
                    {
                        "model_id": "wavlm-base-plus",
                        "loader_class": "transformers.WavLMModel",
                        "repository": "https://huggingface.co/microsoft/wavlm-base-plus",
                        "revision": "4c66d4806a428f2e922ccfa1a962776e232d487b",
                        "required_files": [{"path": "../../outside.bin"}],
                    },
                    "malformed-row",
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(preflight_module, "SOURCE_REGISTRY_PATH", registry)
    checks = _model_checks(tmp_path / "cache")
    assert checks[0]["passed"] is False
    assert checks[2]["passed"] is False
    assert any("unsafe model path" in value for value in checks[2]["observed"]["failures"])


def test_resolve_paths_redirects_unsafe_output_to_external_fallback() -> None:
    paths = resolve_paths(output_root=CONTRACT_PATH.parent)
    assert paths.output_root == preflight_module.DEFAULT_OUTPUT_ROOT
    assert paths.errors
    assert paths.output_root is not None
    assert not paths.output_root.is_relative_to(preflight_module.REPOSITORY_ROOT)


def test_cli_writes_rejection_receipt_without_configured_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fallback = tmp_path / "fallback"
    rejection = {"ready_for_material_run": False}
    monkeypatch.setattr(preflight_module, "DEFAULT_OUTPUT_ROOT", fallback)
    monkeypatch.setattr(run_module, "DEFAULT_OUTPUT_ROOT", fallback)
    monkeypatch.setattr(run_module, "build_preflight", lambda *args, **kwargs: rejection)
    monkeypatch.delenv("SRSCD_CACHE_ROOT", raising=False)
    assert run_module.main(["preflight", "--skip-source-byte-hashes"]) == 2
    path = fallback / "preflight" / "experiment_receipt.json"
    assert json.loads(path.read_text(encoding="utf-8")) == rejection


def test_material_guard_rejects_forged_minimal_receipt(tmp_path: Path) -> None:
    payload = {
        "ready_for_material_run": True,
        "failed_checks": [],
        "binding": {},
        "git": {"commit": "a" * 40, "dirty": False, "dirty_paths": []},
    }
    receipt = {**payload, "payload_sha256": canonical_sha256(payload)}
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ExperimentPreflightError, match="schema is incomplete"):
        require_passing_preflight(path)


def test_material_guard_rejects_incomplete_check_inventory(tmp_path: Path) -> None:
    output = tmp_path / "output"
    receipt = _receipt(output)
    receipt["checks"].pop()
    payload = dict(receipt)
    payload.pop("payload_sha256")
    receipt["payload_sha256"] = canonical_sha256(payload)
    path = _write_receipt(output, receipt)
    with pytest.raises(ExperimentPreflightError, match="check inventory"):
        require_passing_preflight(path)


def test_material_guard_rejects_git_commit_not_equal_to_binding(tmp_path: Path) -> None:
    output = tmp_path / "output"
    receipt = _receipt(output)
    receipt["git"]["commit"] = "b" * 40
    payload = dict(receipt)
    payload.pop("payload_sha256")
    receipt["payload_sha256"] = canonical_sha256(payload)
    path = _write_receipt(output, receipt)
    with pytest.raises(ExperimentPreflightError, match="binding identity"):
        require_passing_preflight(path)


def test_material_guard_revalidates_current_external_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "output"
    receipt = _receipt(output)
    path = _write_receipt(output, receipt)
    current = _receipt(output, ready=False)
    monkeypatch.setattr(preflight_module, "build_preflight", lambda *args, **kwargs: current)
    with pytest.raises(ExperimentPreflightError, match="current preflight revalidation failed"):
        require_passing_preflight(path)


def test_material_guard_rejects_changed_passing_external_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "output"
    receipt = _receipt(output)
    path = _write_receipt(output, receipt)
    current = _receipt(output)
    current["checks"][0]["observed"] = "changed"
    payload = dict(current)
    payload.pop("payload_sha256")
    current["payload_sha256"] = canonical_sha256(payload)
    monkeypatch.setattr(preflight_module, "build_preflight", lambda *args, **kwargs: current)
    with pytest.raises(ExperimentPreflightError, match="receipt is stale"):
        require_passing_preflight(path)
