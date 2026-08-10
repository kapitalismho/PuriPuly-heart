from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from experiments.speaker_representation_scd.execution_guard import (
    ExecutionGuardError,
    action_receipt_is_authoritative,
    validate_worker_execution,
)
from experiments.speaker_representation_scd.provenance import (
    sha256_file,
    with_self_sha256,
)
from experiments.speaker_representation_scd.r1_gate import REPOSITORY_ROOT
from experiments.speaker_representation_scd.r2l_gate import AUTHORITY
from experiments.speaker_representation_scd.r2l_materialize import (
    R2LValidationError,
    derive_candidates,
    load_legacy_documents,
)
from experiments.speaker_representation_scd.r3_gate import validate_r3_gate
from experiments.speaker_representation_scd.run_provenance import run_provenance

ANCHOR_INDEX_RELATIVE_PATH = Path("data/r3/legacy_common_gt/anchor_index.jsonl")
ANCHOR_INDEX_MANIFEST_RELATIVE_PATH = Path("data/r3/legacy_common_gt/anchor_index.manifest.json")
ANCHOR_RECEIPT_RELATIVE_PATH = Path("manifests/r3/legacy_common_gt/anchor_index_receipt.json")
EXPECTED_POSITIVE_COUNT = 450
EXPECTED_NEGATIVE_COUNT = 360
EXPECTED_PAIR_COUNT = 313


class R3PrepareError(RuntimeError):
    pass


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _digest(rows: list[dict[str, Any]]) -> str:
    value = hashlib.sha256()
    for row in rows:
        value.update(_canonical_bytes(row) + b"\n")
    return value.hexdigest()


def build_anchor_index(documents: dict[str, Any], derived: dict[str, Any]) -> list[dict[str, Any]]:
    ledger = documents["design_ledger"]
    positives = derived["positives"]
    negatives = derived["negatives"]
    pairs = derived["pairs"]
    if _digest(positives) != ledger["candidate_ledger"]["positive_rows_sha256"]:
        raise R3PrepareError("positive anchor rows differ from the frozen design ledger")
    if _digest(negatives) != ledger["candidate_ledger"]["negative_rows_sha256"]:
        raise R3PrepareError("negative anchor rows differ from the frozen design ledger")
    if _digest(pairs) != ledger["matching"]["pair_rows_sha256"]:
        raise R3PrepareError("matched pairs differ from the frozen design ledger")
    pair_by_positive: dict[str, dict[str, Any]] = {}
    pair_by_negative: dict[str, dict[str, Any]] = {}
    for pair in pairs:
        pair_by_positive[str(pair["positive_id"])] = pair
        pair_by_negative[str(pair["negative_id"])] = pair

    def row_for(candidate: dict[str, Any], pair: dict[str, Any] | None) -> dict[str, Any]:
        return {
            "candidate_id": candidate["candidate_id"],
            "class": candidate["class"],
            "kind": candidate["kind"],
            "episode_id": candidate["episode_id"],
            "session_id": candidate["session_id"],
            "waveform_id": candidate["wav_sha256"],
            "coordinate": int(candidate["coordinate"]),
            "corpus": candidate["corpus"],
            "language": candidate["language"],
            "block_id": candidate["block_id"],
            "synthetic_manifest": candidate.get("synthetic_manifest"),
            "duration_ms": int(candidate["duration_ms"]),
            "gap_ms": int(candidate["gap_ms"]),
            "stress": candidate["stress"],
            "pair_id": pair["pair_id"] if pair else None,
        }

    rows = [
        row_for(candidate, pair_by_positive.get(candidate["candidate_id"]))
        for candidate in sorted(positives, key=lambda row: row["candidate_id"])
    ]
    rows.extend(
        row_for(candidate, pair_by_negative.get(candidate["candidate_id"]))
        for candidate in sorted(negatives, key=lambda row: row["candidate_id"])
    )
    if len(rows) != EXPECTED_POSITIVE_COUNT + EXPECTED_NEGATIVE_COUNT:
        raise R3PrepareError("anchor index count differs from the frozen ledger")
    if sum(1 for row in rows if row.get("pair_id")) != 2 * EXPECTED_PAIR_COUNT:
        raise R3PrepareError("anchor index pair binding differs from the frozen ledger")
    return rows


def run_prepare(cache_root: Path, requested_argv: tuple[str, ...]) -> Path:
    result_path = cache_root / ANCHOR_RECEIPT_RELATIVE_PATH
    validate_worker_execution(cache_root, expected_receipt=result_path)
    if result_path.exists():
        if action_receipt_is_authoritative(cache_root, result_path, "r3-prepare"):
            raise R3PrepareError(
                f"refusing to rerun anchor preparation with completed evidence: {result_path}"
            )
    gate = validate_r3_gate(cache_root=cache_root.resolve())
    if not gate.valid:
        raise R3PrepareError("; ".join(gate.errors))
    if gate.allowed_actions.get("r3_prepare") is not True:
        raise R3PrepareError("R3 anchor preparation is not authorized")
    documents = load_legacy_documents(REPOSITORY_ROOT)
    try:
        derived = derive_candidates(documents)
    except R2LValidationError as error:
        raise R3PrepareError(str(error)) from error
    rows = build_anchor_index(documents, derived)
    index_path = cache_root / ANCHOR_INDEX_RELATIVE_PATH
    manifest_path = cache_root / ANCHOR_INDEX_MANIFEST_RELATIVE_PATH
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
        ),
        encoding="utf-8",
    )
    manifest = with_self_sha256(
        {
            "schema_version": 1,
            "artifact_role": "r3_anchor_index_manifest",
            "experiment_id": "speaker_representation_scd_v1",
            "authority": AUTHORITY,
            "scope": "legacy-common-gt-v1",
            "sha256": sha256_file(index_path),
            "row_count": len(rows),
            "positive_count": EXPECTED_POSITIVE_COUNT,
            "negative_count": EXPECTED_NEGATIVE_COUNT,
            "pair_count": EXPECTED_PAIR_COUNT,
            "design_ledger_positive_rows_sha256": documents["design_ledger"]["candidate_ledger"][
                "positive_rows_sha256"
            ],
            "design_ledger_negative_rows_sha256": documents["design_ledger"]["candidate_ledger"][
                "negative_rows_sha256"
            ],
            "design_ledger_pair_rows_sha256": documents["design_ledger"]["matching"][
                "pair_rows_sha256"
            ],
        }
    )
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    document = with_self_sha256(
        {
            "schema_version": 1,
            "artifact_role": "r3_anchor_index_receipt",
            "experiment_id": "speaker_representation_scd_v1",
            "authority": AUTHORITY,
            "scope": "legacy-common-gt-v1",
            "anchor_index": {
                "relative_to_cache_root": ANCHOR_INDEX_RELATIVE_PATH.as_posix(),
                "sha256": sha256_file(index_path),
            },
            "anchor_index_manifest": {
                "relative_to_cache_root": ANCHOR_INDEX_MANIFEST_RELATIVE_PATH.as_posix(),
                "sha256": sha256_file(manifest_path),
                "self_sha256": manifest["self_sha256"],
            },
            "supervision_binding": {
                "execution_id": os.environ.get("SRSCD_EXECUTION_LEASE_TOKEN"),
                "expected_receipt_relative_path": ANCHOR_RECEIPT_RELATIVE_PATH.as_posix(),
                "authority": "requires_completed_usage_attestation",
            },
            "provenance": {
                "authority": AUTHORITY,
                "execution_identity": {
                    "run_id": uuid4().hex,
                    "process_id": os.getpid(),
                    "started_at_utc": datetime.now(UTC).isoformat(),
                },
                "run_provenance": run_provenance(
                    REPOSITORY_ROOT,
                    requested_argv,
                    deterministic_seed=0,
                    deterministic_kernels=False,
                ),
                "code_sha256": sha256_file(Path(__file__).resolve()),
            },
        }
    )
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", choices=("prepare",), required=True)
    parser.parse_args(argv)
    cache_root = Path(os.environ["SRSCD_CACHE_ROOT"]).resolve()
    requested = tuple(
        [sys.executable, "-m", __package__ + ".r3_prepare", *(argv or sys.argv[1:])]
    )
    print(run_prepare(cache_root, requested))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ExecutionGuardError as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1) from error
