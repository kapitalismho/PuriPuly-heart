from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from experiments.psem_relative_occupancy_gate.contracts import ActivityInterval
from experiments.psem_relative_occupancy_gate.io_utils import (
    CONFIG_PATH,
    canonical_sha256,
    config,
    corpus_root,
    data_dir,
    reference_root,
    safe_child,
    safe_output_path,
    sha256_file,
    write_jsonl,
)
from experiments.psem_relative_occupancy_gate.provenance import load_frozen_dataset
from experiments.psem_training_strategy_gate.data.forced_alignment_reference import (
    validate_reference_checkout,
)
from experiments.psem_training_strategy_gate.data.reference_normalization import (
    normalize_reference_session,
    open_reference_checkout,
)


class DerivationError(RuntimeError):
    pass


def derive_rows(
    *,
    corpus: Path,
    reference: Path,
    roles: Iterable[str],
    frozen_selection: Path | None,
) -> list[dict[str, Any]]:
    corpus = corpus_root(corpus)
    reference = reference_root(reference)
    requested_roles = tuple(sorted(set(roles)))
    if not requested_roles:
        raise DerivationError("at least one role is required")
    if "PSEM-STRATEGY-EVAL" in requested_roles:
        raise DerivationError(
            "EVAL is sealed until the frozen DEV selection implementation is accepted"
        )
    if frozen_selection is not None:
        raise DerivationError("a frozen selection receipt is invalid before EVAL authorization")
    allowed_roles = {"PSEM-STRATEGY-TRAIN", "PSEM-STRATEGY-DEV"}
    if not set(requested_roles) <= allowed_roles:
        raise DerivationError("requested role is outside the frozen V2 contract")
    dataset = load_frozen_dataset()
    split_path = data_dir() / "split_manifest.json"
    source_manifest_path = data_dir() / "source_manifest.jsonl"
    normalization_manifest_path = data_dir() / "normalization_manifest.jsonl"
    role_map = {source_id: str(row["role"]) for source_id, row in dataset.assignments.items()}
    checkout = open_reference_checkout(reference)
    opening_provenance = dict(checkout.provenance)
    cfg = config()
    result: list[dict[str, Any]] = []
    for source_id in sorted(source for source, role in role_map.items() if role in requested_roles):
        source = dataset.sources[source_id]
        session = normalize_reference_session(source, corpus, checkout)
        regenerated = session.manifest_row()
        accepted = dataset.normalizations[source_id]
        if canonical_sha256(regenerated) != canonical_sha256(accepted):
            raise DerivationError(f"V2 reconstruction mismatch for {source_id}")
        waveform = safe_child(corpus, str(source["audio_ref"]), f"waveform {source_id}")
        if (
            waveform.is_symlink()
            or not waveform.is_file()
            or waveform.stat().st_size != int(source["waveform_size_bytes"])
            or sha256_file(waveform) != str(source["waveform_sha256"])
        ):
            raise DerivationError(f"waveform identity mismatch: {source_id}")
        label_rows = session.labels.to_dict()
        intervals = tuple(ActivityInterval.from_dict(row) for row in label_rows["intervals"])
        interval_rows = [row.to_dict() for row in intervals]
        row = {
            "schema_version": "psem.relative_occupancy.manifest_row.v1",
            "ontology": cfg["experiment_id"],
            "source_id": source_id,
            "corpus": source["corpus"],
            "session_id": source["session_id"],
            "role": role_map[source_id],
            "component_id": dataset.assignments[source_id]["component_id"],
            "sample_rate_hz": 16000,
            "scored_start_sample": session.scored_start_sample,
            "scored_end_sample": session.scored_end_sample,
            "audio_ref": source["audio_ref"],
            "audio_path": str(waveform),
            "waveform_sha256": source["waveform_sha256"],
            "waveform_size_bytes": source["waveform_size_bytes"],
            "source_duration_samples": source["duration_samples"],
            "source_speaker_ids": source["speaker_ids"],
            "source_annotation_ref": source["annotation_ref"],
            "source_annotation_sha256": source["annotation_sha256"],
            "source_manifest_row_sha256": canonical_sha256(source),
            "normalization_manifest_row_sha256": canonical_sha256(accepted),
            "reference_ref": regenerated["reference_ref"],
            "reference_sha256": regenerated["reference_sha256"],
            "reference_repository": regenerated["reference_repository"],
            "reference_commit": regenerated["reference_commit"],
            "reference_git_tree": regenerated["reference_git_tree"],
            "reference_metadata_files": regenerated["reference_metadata_files"],
            "reference_metadata_sha256": regenerated["reference_metadata_sha256"],
            "speaker_mapping_sha256": regenerated["speaker_mapping_sha256"],
            "intervals": interval_rows,
            "intervals_sha256": canonical_sha256(interval_rows),
            "transitions": label_rows["transitions"],
            "topology_episodes": label_rows["topology_episodes"],
            "v2_exposure": label_rows["exposure"],
            "v2_canonical_intervals_sha256": regenerated["canonical_intervals_sha256"],
            "v2_label_result_sha256": regenerated["label_result_sha256"],
            "v2_nonlexical_mask_sha256": regenerated["nonlexical_mask_sha256"],
            "v2_source_record_sha256": regenerated["source_record_sha256"],
            "dataset_freeze_file_sha256": cfg["dataset"]["freeze_file_sha256"],
            "dataset_freeze_payload_sha256": cfg["dataset"]["freeze_payload_sha256"],
            "source_manifest_sha256": sha256_file(source_manifest_path),
            "normalization_manifest_sha256": sha256_file(normalization_manifest_path),
            "split_manifest_sha256": sha256_file(split_path),
            "reference_checkout_sha256": canonical_sha256(opening_provenance),
            "config_sha256": sha256_file(CONFIG_PATH),
            "eval_selection_sha256": None,
            "eval_status": "sealed",
        }
        row["row_sha256"] = canonical_sha256(row)
        result.append(row)
    if not result:
        raise DerivationError("requested roles selected no sources")
    closing_provenance = validate_reference_checkout(reference)
    if closing_provenance != opening_provenance:
        raise DerivationError("reference checkout changed during derivation")
    if load_frozen_dataset().summary() != dataset.summary():
        raise DerivationError("frozen V2 identity changed during derivation")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-root", type=Path)
    parser.add_argument("--reference-root", type=Path)
    parser.add_argument("--roles", nargs="+", required=True)
    parser.add_argument("--frozen-selection", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = derive_rows(
        corpus=corpus_root(args.corpus_root),
        reference=reference_root(args.reference_root),
        roles=args.roles,
        frozen_selection=args.frozen_selection.resolve() if args.frozen_selection else None,
    )
    output = safe_output_path(args.output)
    write_jsonl(output, rows)
    print(json.dumps({"output": str(output), "source_count": len(rows)}, sort_keys=True))


if __name__ == "__main__":
    main()
