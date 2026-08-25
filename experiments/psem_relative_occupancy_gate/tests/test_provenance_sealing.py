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
