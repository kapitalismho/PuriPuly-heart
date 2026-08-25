from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.psem_relative_occupancy_gate import (
    authorize_eval,
    derive_relative_occupancy,
    eval_access,
    preflight,
)
from experiments.psem_relative_occupancy_gate.derive_relative_occupancy import (
    DerivationError,
)
from experiments.psem_relative_occupancy_gate.eval_access import (
    EvalAccessError,
    claim_eval_authorization,
    consumption_receipt_path,
    validate_opened_eval_manifest,
    validate_unused_eval_authorization,
)
from experiments.psem_relative_occupancy_gate.io_utils import (
    CONFIG_PATH,
    ExperimentError,
    canonical_sha256,
    data_dir,
    safe_child,
    safe_output_path,
    sha256_file,
    write_json,
    write_jsonl,
)
from experiments.psem_relative_occupancy_gate.provenance import load_frozen_dataset


def test_output_path_cannot_enter_immutable_v2() -> None:
    with pytest.raises(ExperimentError, match="immutable V2"):
        safe_output_path(data_dir() / "forged.json")


def test_output_path_cannot_overwrite_experiment_source() -> None:
    with pytest.raises(ExperimentError, match="results root"):
        safe_output_path(CONFIG_PATH)


def test_external_relative_path_cannot_escape_root(tmp_path: Path) -> None:
    with pytest.raises(ExperimentError, match="relative path"):
        safe_child(tmp_path, "../escape.wav", "waveform")


def test_eval_derivation_requires_selection_and_one_use_authorization(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(derive_relative_occupancy, "corpus_root", lambda value: tmp_path)
    monkeypatch.setattr(derive_relative_occupancy, "reference_root", lambda value: tmp_path)
    with pytest.raises(DerivationError, match="one-use authorization"):
        derive_relative_occupancy.derive_rows(
            corpus=tmp_path,
            reference=tmp_path,
            roles=["PSEM-STRATEGY-EVAL"],
            frozen_selection=tmp_path / "forged.json",
        )


def test_frozen_dataset_has_exact_source_and_role_bindings() -> None:
    dataset = load_frozen_dataset()
    assert len(dataset.sources) == 93
    assert len(dataset.source_ids("PSEM-STRATEGY-TRAIN")) == 64
    assert len(dataset.source_ids("PSEM-STRATEGY-DEV")) == 10
    assert len(dataset.source_ids("PSEM-STRATEGY-EVAL")) == 19


def test_eval_authorization_is_bound_to_one_manifest_target(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    selection_path = tmp_path / "selection.json"
    authorization_path = tmp_path / "authorization.json"
    manifest_path = tmp_path / "eval_manifest.jsonl"
    selection = {
        "schema_version": "psem.relative_occupancy.dev_selection.v1",
        "role": "PSEM-STRATEGY-DEV",
        "eval_status": "sealed",
        "eval_open_authorized": False,
        "eval_open_count": 0,
    }
    selection["selection_sha256"] = canonical_sha256(selection)
    write_json(selection_path, selection)
    verification_path = tmp_path / "model_gate_verification.json"
    write_json(verification_path, {"passed": True})
    authorization = {
        "schema_version": "psem.relative_occupancy.eval_authorization.v1",
        "role": "PSEM-STRATEGY-EVAL",
        "authorization_state": "authorized_for_one_manifest_derivation",
        "use_limit": 1,
        "accepted_c2_head": "a" * 40,
        "selection_sha256": selection["selection_sha256"],
        "selection_file_sha256": sha256_file(selection_path),
        "model_gate_verification_path": str(verification_path.resolve()),
        "model_gate_verification_sha256": sha256_file(verification_path),
        "manifest_output_path": str(manifest_path.resolve()),
    }
    authorization["authorization_sha256"] = canonical_sha256(authorization)
    write_json(authorization_path, authorization)
    monkeypatch.setattr(eval_access, "_current_head", lambda: "a" * 40)
    monkeypatch.setattr(
        eval_access, "validate_frozen_selection_bindings", lambda *_: None
    )
    observed_selection, observed_authorization = validate_unused_eval_authorization(
        selection_path=selection_path,
        authorization_path=authorization_path,
        manifest_output=manifest_path,
    )
    assert observed_selection == selection
    assert observed_authorization == authorization
    _, _, claim = claim_eval_authorization(
        selection_path=selection_path,
        authorization_path=authorization_path,
        manifest_output=manifest_path,
    )
    assert claim["accepted_c2_head"] == "a" * 40
    assert consumption_receipt_path(authorization_path).is_file()
    with pytest.raises(EvalAccessError, match="already been consumed"):
        claim_eval_authorization(
            selection_path=selection_path,
            authorization_path=authorization_path,
            manifest_output=manifest_path,
        )


def test_opened_eval_manifest_binds_selection_and_access_receipt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    selection_path = tmp_path / "selection.json"
    manifest_path = tmp_path / "eval_manifest.jsonl"
    access_path = tmp_path / "eval_manifest_access_receipt.json"
    authorization_path = tmp_path / "authorization.json"
    verification_path = tmp_path / "model_gate_verification.json"
    selection = {
        "schema_version": "psem.relative_occupancy.dev_selection.v1",
        "role": "PSEM-STRATEGY-DEV",
        "eval_status": "sealed",
        "eval_open_authorized": False,
        "eval_open_count": 0,
    }
    selection["selection_sha256"] = canonical_sha256(selection)
    write_json(selection_path, selection)
    write_json(verification_path, {"passed": True})
    authorization = {
        "schema_version": "psem.relative_occupancy.eval_authorization.v1",
        "role": "PSEM-STRATEGY-EVAL",
        "authorization_state": "authorized_for_one_manifest_derivation",
        "use_limit": 1,
        "accepted_c2_head": "a" * 40,
        "selection_sha256": selection["selection_sha256"],
        "selection_file_sha256": sha256_file(selection_path),
        "model_gate_verification_path": str(verification_path.resolve()),
        "model_gate_verification_sha256": sha256_file(verification_path),
        "manifest_output_path": str(manifest_path.resolve()),
    }
    authorization["authorization_sha256"] = canonical_sha256(authorization)
    write_json(authorization_path, authorization)
    claim_path = consumption_receipt_path(authorization_path)
    claim = {
        "schema_version": "psem.relative_occupancy.eval_consumption.v1",
        "role": "PSEM-STRATEGY-EVAL",
        "authorization_sha256": authorization["authorization_sha256"],
        "authorization_file_sha256": sha256_file(authorization_path),
        "accepted_c2_head": "a" * 40,
        "manifest_output_path": str(manifest_path.resolve()),
    }
    claim["claim_sha256"] = canonical_sha256(claim)
    write_json(claim_path, claim)
    row = {
        "source_id": "eval-source",
        "role": "PSEM-STRATEGY-EVAL",
        "eval_status": "opened_once",
        "eval_selection_sha256": selection["selection_sha256"],
        "eval_authorization_sha256": authorization["authorization_sha256"],
        "config_sha256": sha256_file(CONFIG_PATH),
    }
    row["row_sha256"] = canonical_sha256(row)
    write_jsonl(manifest_path, [row])
    access = {
        "schema_version": "psem.relative_occupancy.eval_access_receipt.v1",
        "role": "PSEM-STRATEGY-EVAL",
        "open_count": 1,
        "selection_sha256": selection["selection_sha256"],
        "accepted_c2_head": "a" * 40,
        "authorization_path": str(authorization_path.resolve()),
        "authorization_file_sha256": sha256_file(authorization_path),
        "authorization_sha256": authorization["authorization_sha256"],
        "consumption_receipt_path": str(claim_path.resolve()),
        "consumption_receipt_sha256": sha256_file(claim_path),
        "model_gate_verification_sha256": sha256_file(verification_path),
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "source_count": 1,
    }
    access["access_sha256"] = canonical_sha256(access)
    write_json(access_path, access)
    monkeypatch.setattr(
        "experiments.psem_relative_occupancy_gate.eval_access.load_frozen_dataset",
        lambda: SimpleNamespace(source_ids=lambda role: ("eval-source",)),
    )
    monkeypatch.setattr(eval_access, "_current_head", lambda: "a" * 40)
    monkeypatch.setattr(
        eval_access, "validate_frozen_selection_bindings", lambda *_: None
    )
    rows, observed_selection = validate_opened_eval_manifest(
        manifest_path=manifest_path,
        access_path=access_path,
        selection_path=selection_path,
        authorization_path=authorization_path,
    )
    assert rows == [row]
    assert observed_selection == selection
    copied_access_path = tmp_path / "copied_access.json"
    write_json(copied_access_path, access)
    with pytest.raises(EvalAccessError, match="path is not canonical"):
        validate_opened_eval_manifest(
            manifest_path=manifest_path,
            access_path=copied_access_path,
            selection_path=selection_path,
            authorization_path=authorization_path,
        )


def test_eval_recovery_accepts_only_bound_same_manifest_commit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repo_root = tmp_path / "repo"
    package = repo_root / "experiments" / "psem_relative_occupancy_gate"
    package.mkdir(parents=True)
    contract_path = package / "contract.py"
    contract_path.write_text("after\n", encoding="utf-8")
    config_path = package / "config.json"
    config_path.write_text("{}\n", encoding="utf-8")
    selection_path = tmp_path / "selection.json"
    manifest_path = tmp_path / "eval_manifest.jsonl"
    access_path = eval_access.access_receipt_path(manifest_path)
    authorization_path = tmp_path / "eval_authorization.json"
    claim_path = eval_access.consumption_receipt_path(authorization_path)
    verification_path = tmp_path / "model_gate_verification.json"
    write_json(verification_path, {"passed": True})
    write_jsonl(manifest_path, [{"source_id": "eval"}])
    write_json(access_path, {"access": True})
    write_json(claim_path, {"claim": True})
    before_hash = "1" * 64
    after_hash = sha256_file(contract_path)
    selection = {
        "selection_sha256": "selection",
        "artifact_bindings": {
            "contract_files": {"contract.py": before_hash},
        },
    }
    write_json(selection_path, selection)
    authorization = {
        "schema_version": "psem.relative_occupancy.eval_authorization.v1",
        "role": "PSEM-STRATEGY-EVAL",
        "authorization_state": "authorized_for_one_manifest_derivation",
        "use_limit": 1,
        "accepted_c2_head": "a" * 40,
        "selection_sha256": "selection",
        "selection_file_sha256": sha256_file(selection_path),
        "model_gate_verification_path": str(verification_path.resolve()),
        "model_gate_verification_sha256": sha256_file(verification_path),
        "manifest_output_path": str(manifest_path.resolve()),
    }
    authorization["authorization_sha256"] = canonical_sha256(authorization)
    write_json(authorization_path, authorization)
    relative_path = "experiments/psem_relative_occupancy_gate/contract.py"
    recovery = {
        "schema_version": eval_access.EVAL_RECOVERY_SCHEMA_VERSION,
        "role": "PSEM-STRATEGY-EVAL",
        "recovery_state": "authorized_for_same_opened_manifest_resume",
        "recovery_reason": eval_access.EVAL_RECOVERY_REASON,
        "accepted_c2_head": "a" * 40,
        "recovery_head": "b" * 40,
        "authorization_sha256": authorization["authorization_sha256"],
        "authorization_file_sha256": sha256_file(authorization_path),
        "selection_sha256": "selection",
        "selection_file_sha256": sha256_file(selection_path),
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "access_receipt_path": str(access_path.resolve()),
        "access_receipt_sha256": sha256_file(access_path),
        "consumption_receipt_path": str(claim_path.resolve()),
        "consumption_receipt_sha256": sha256_file(claim_path),
        "manifest_open_count": 1,
        "additional_manifest_derivations": 0,
        "prior_eval_aggregate_count": 0,
        "changed_files": {
            relative_path: {
                "before_sha256": before_hash,
                "after_sha256": after_hash,
            }
        },
        "contract_overrides": {
            "contract.py": {
                "before_sha256": before_hash,
                "after_sha256": after_hash,
            }
        },
    }
    recovery["recovery_sha256"] = canonical_sha256(recovery)
    write_json(eval_access.recovery_receipt_path(authorization_path), recovery)
    observed_overrides: list[dict[str, dict[str, str]] | None] = []
    monkeypatch.setattr(eval_access, "PACKAGE_ROOT", package)
    monkeypatch.setattr(eval_access, "CONFIG_PATH", config_path)
    monkeypatch.setattr(eval_access, "EVAL_RECOVERY_ALLOWED_PATHS", {relative_path})
    monkeypatch.setattr(eval_access, "EVAL_RECOVERY_REQUIRED_PATHS", {relative_path})
    monkeypatch.setattr(eval_access, "_current_head", lambda: "b" * 40)
    monkeypatch.setattr(eval_access, "_tracked_worktree_is_clean", lambda _root: True)
    monkeypatch.setattr(eval_access, "_git_is_ancestor", lambda _base, _head: True)
    monkeypatch.setattr(
        eval_access, "_git_changed_paths", lambda _base, _head: {relative_path}
    )
    monkeypatch.setattr(
        eval_access,
        "_git_file_sha256",
        lambda _head, _path: before_hash,
    )
    monkeypatch.setattr(
        eval_access,
        "validate_frozen_selection_bindings",
        lambda _path, _selection, contract_overrides=None: observed_overrides.append(
            contract_overrides
        ),
    )
    assert (
        eval_access.load_eval_authorization(
            authorization_path,
            selection_path=selection_path,
            selection=selection,
            manifest_output=manifest_path,
        )
        == authorization
    )
    assert observed_overrides == [recovery["contract_overrides"]]
    monkeypatch.setattr(eval_access, "_current_head", lambda: "a" * 40)
    with pytest.raises(EvalAccessError, match="recovery binding mismatch"):
        eval_access.load_eval_authorization(
            authorization_path,
            selection_path=selection_path,
            selection=selection,
            manifest_output=manifest_path,
        )
    monkeypatch.setattr(eval_access, "_current_head", lambda: "b" * 40)
    monkeypatch.setattr(eval_access, "_git_is_ancestor", lambda _base, _head: False)
    with pytest.raises(EvalAccessError, match="recovery binding mismatch"):
        eval_access.load_eval_authorization(
            authorization_path,
            selection_path=selection_path,
            selection=selection,
            manifest_output=manifest_path,
        )


def test_eval_recovery_cleanliness_whitelists_only_eval_outputs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repo_root = tmp_path / "repo"
    package = repo_root / "experiments" / "psem_relative_occupancy_gate"
    eval_root = package / "results" / "eval"
    eval_root.mkdir(parents=True)
    monkeypatch.setattr(eval_access, "PACKAGE_ROOT", package)

    def status(stdout: bytes) -> None:
        monkeypatch.setattr(
            eval_access.subprocess,
            "run",
            lambda *_args, **_kwargs: SimpleNamespace(stdout=stdout),
        )

    status(b"?? experiments/psem_relative_occupancy_gate/results/eval/receipt.json\0")
    assert eval_access._tracked_worktree_is_clean(eval_root)
    status(b"?? arbitrary.txt\0")
    assert not eval_access._tracked_worktree_is_clean(eval_root)
    status(b" M experiments/psem_relative_occupancy_gate/eval_access.py\0")
    assert not eval_access._tracked_worktree_is_clean(eval_root)


def test_eval_authorization_revalidates_accepted_dev_evidence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    package = tmp_path / "package"
    results = tmp_path / "results"
    package.mkdir()
    results.mkdir()
    contract_path = package / "contract.py"
    config_path = package / "config.json"
    contract_path.write_text("contract\n", encoding="utf-8")
    config_path.write_text("{}\n", encoding="utf-8")
    names = {
        "gate0": "gate0_oracle_metrics.json",
        "gate0_verification": "gate0_verification.json",
        "gate1": "gate1_metrics.json",
        "gate1_product": "gate1_product_frontier.json",
        "gate1_topology": "gate1_topology_slices.json",
        "gate1_latency": "gate1_latency_breakdown.json",
        "gate1_events": "gate1_event_ledger.jsonl",
        "gate2": "gate2_metrics.json",
        "gate2_events": "gate2_event_ledger.jsonl",
        "product": "product_frontiers.json",
        "topology": "topology_slices.json",
        "latency": "latency_breakdown.json",
        "sortformer": "sortformer_model_receipt.json",
        "lseend": "lseend_model_receipt.json",
    }
    paths = {key: results / name for key, name in names.items()}
    for key, path in paths.items():
        payload: dict[str, object] = {"artifact": key}
        if key == "gate0":
            payload["contract_artifacts"] = {
                "contract.py": sha256_file(contract_path)
            }
        write_json(path, payload)
    manifest_path = results / "relative_occupancy_manifest.jsonl"
    write_jsonl(manifest_path, [{"source_id": "dev"}])
    selection_path = results / "dev_selection_receipt.json"
    selection = {
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "config_sha256": sha256_file(config_path),
        "gate0_sha256": sha256_file(paths["gate0"]),
        "gate0_verification_sha256": sha256_file(paths["gate0_verification"]),
        "artifact_bindings": {
            "contract_files": {"contract.py": sha256_file(contract_path)},
            "gate0_metrics_sha256": sha256_file(paths["gate0"]),
            "gate0_verification_sha256": sha256_file(paths["gate0_verification"]),
            "gate1_metrics_sha256": sha256_file(paths["gate1"]),
            "gate1_product_frontier_sha256": sha256_file(paths["gate1_product"]),
            "gate1_topology_slices_sha256": sha256_file(paths["gate1_topology"]),
            "gate1_latency_breakdown_sha256": sha256_file(paths["gate1_latency"]),
            "gate1_event_ledger_sha256": sha256_file(paths["gate1_events"]),
            "gate2_metrics_sha256": sha256_file(paths["gate2"]),
            "gate2_event_ledger_sha256": sha256_file(paths["gate2_events"]),
            "product_frontiers_sha256": sha256_file(paths["product"]),
            "topology_slices_sha256": sha256_file(paths["topology"]),
            "latency_breakdown_sha256": sha256_file(paths["latency"]),
            "model_receipts": {
                "sortformer": sha256_file(paths["sortformer"]),
                "lseend": sha256_file(paths["lseend"]),
            },
        },
    }
    write_json(selection_path, selection)
    verification = {
        "artifact_sha256": {
            key: sha256_file(paths[key])
            for key in (
                "gate0_verification",
                "gate1",
                "gate2",
                "product",
                "topology",
                "latency",
            )
        }
        | {"selection": sha256_file(selection_path)}
    }
    monkeypatch.setattr(authorize_eval, "PACKAGE_ROOT", package)
    monkeypatch.setattr(authorize_eval, "CONFIG_PATH", config_path)
    monkeypatch.setattr(eval_access, "PACKAGE_ROOT", package)
    monkeypatch.setattr(eval_access, "CONFIG_PATH", config_path)
    authorize_eval._validate_dev_evidence(
        selection_path=selection_path,
        selection=selection,
        verification=verification,
    )
    contract_path.write_text("changed\n", encoding="utf-8")
    with pytest.raises(
        authorize_eval.EvalAuthorizationError, match="contract changed"
    ):
        authorize_eval._validate_dev_evidence(
            selection_path=selection_path,
            selection=selection,
            verification=verification,
        )


def test_preflight_receipt_rejects_forged_stable_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt = {
        "schema_version": "psem.relative_occupancy.preflight.v1",
        "authority": {"ref": "authority", "sha256": "pin"},
        "config_path": "config",
        "config_sha256": "config-sha",
        "dataset": {"source_count": 93},
        "paths": {
            "corpus_root": "corpus",
            "reference_root": "reference",
            "research_root": "research",
            "lseend_root": "lseend",
        },
        "reference_receipt": {"commit": "reference"},
        "model_source_checkouts": {},
        "environment": {"python": "python"},
        "eval_status": "sealed",
        "checks": [{"id": "binding", "passed": True, "detail": "exact"}],
        "passed": True,
    }
    monkeypatch.setattr(preflight, "run_preflight", lambda **kwargs: receipt)
    forged = copy.deepcopy(receipt)
    forged["checks"][0]["detail"] = "forged"
    with pytest.raises(preflight.PreflightError, match="does not match"):
        preflight.validate_preflight_receipt(forged)
