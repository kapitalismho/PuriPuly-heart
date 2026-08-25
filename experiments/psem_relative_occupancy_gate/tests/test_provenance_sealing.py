from __future__ import annotations

import copy
from pathlib import Path

import pytest

from experiments.psem_relative_occupancy_gate import derive_relative_occupancy, preflight
from experiments.psem_relative_occupancy_gate.derive_relative_occupancy import (
    DerivationError,
)
from experiments.psem_relative_occupancy_gate.io_utils import (
    ExperimentError,
    data_dir,
    safe_child,
    safe_output_path,
)
from experiments.psem_relative_occupancy_gate.provenance import load_frozen_dataset


def test_output_path_cannot_enter_immutable_v2() -> None:
    with pytest.raises(ExperimentError, match="immutable V2"):
        safe_output_path(data_dir() / "forged.json")


def test_external_relative_path_cannot_escape_root(tmp_path: Path) -> None:
    with pytest.raises(ExperimentError, match="relative path"):
        safe_child(tmp_path, "../escape.wav", "waveform")


def test_eval_derivation_is_unconditionally_sealed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(derive_relative_occupancy, "corpus_root", lambda value: tmp_path)
    monkeypatch.setattr(derive_relative_occupancy, "reference_root", lambda value: tmp_path)
    with pytest.raises(DerivationError, match="EVAL is sealed"):
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
