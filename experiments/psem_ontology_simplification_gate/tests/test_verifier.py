from __future__ import annotations

from pathlib import Path

import pytest

from experiments.psem_ontology_simplification_gate.verify_results import (
    ResultVerificationError,
    _verify_expected_artifact_hashes,
)
from experiments.psem_relative_occupancy_gate.io_utils import sha256_file

MATERIAL_RESULT_NAMES = (
    "ontology_sufficiency.json",
    "PATH_DECISION.md",
    "anchor_dropout_slices.json",
    "global_overlap_diagnostic.json",
    "product_frontiers.json",
    "paired_session_deltas.json",
    "bootstrap_intervals.json",
    "sortformer_simple_anchor_metrics.json",
    "lseend_simple_anchor_metrics.json",
    "sortformer_anchor_overlap_metrics.json",
    "lseend_anchor_overlap_metrics.json",
    "production_vad_speech_gate.jsonl",
    "production_vad_replay_receipt.json",
    "production_vad_sensitivity.json",
)


def _sealed_artifacts(root: Path) -> tuple[list[Path], dict[str, str]]:
    paths = []
    for name in MATERIAL_RESULT_NAMES:
        path = root / name
        path.write_text(f'{{"artifact":"{name}"}}\n', encoding="utf-8")
        paths.append(path)
    return paths, {path.name: sha256_file(path) for path in paths}


@pytest.mark.parametrize("mutated_name", MATERIAL_RESULT_NAMES)
def test_expected_result_seal_rejects_every_material_artifact_drift(
    tmp_path: Path, mutated_name: str
) -> None:
    paths, expected = _sealed_artifacts(tmp_path)
    _verify_expected_artifact_hashes(
        result_dir=tmp_path,
        paths=paths,
        expected_hashes=expected,
        require_complete=True,
    )
    (tmp_path / mutated_name).write_text('{"mutated":true}\n', encoding="utf-8")
    with pytest.raises(ResultVerificationError, match="digest mismatch"):
        _verify_expected_artifact_hashes(
            result_dir=tmp_path,
            paths=paths,
            expected_hashes=expected,
            require_complete=True,
        )


def test_expected_result_seal_rejects_incomplete_coverage(tmp_path: Path) -> None:
    paths, expected = _sealed_artifacts(tmp_path)
    expected.pop("bootstrap_intervals.json")
    with pytest.raises(ResultVerificationError, match="coverage mismatch"):
        _verify_expected_artifact_hashes(
            result_dir=tmp_path,
            paths=paths,
            expected_hashes=expected,
            require_complete=True,
        )
