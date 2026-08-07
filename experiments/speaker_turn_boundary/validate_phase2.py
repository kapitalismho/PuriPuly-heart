from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.speaker_turn_boundary.config import EXPERIMENT_DATA_DIR
from experiments.speaker_turn_boundary.corpus import external
from experiments.speaker_turn_boundary.corpus.phase2_schemas import Phase2Manifest
from experiments.speaker_turn_boundary.corpus.validation import (
    collect_validation_report,
    manifest_identity_evidence,
)

D1_IDS = ["ls_dev", "ls_held_out_clean", "ls_held_out_other"]
D2_COUNTERPARTS = {"ami_dev_pilot": ["ami_held_out_pilot"], "ami_held_out_pilot": ["ami_dev_pilot"]}
D3_HELD_OUT_IDS = [
    "ls_held_out_clean",
    "ls_held_out_other",
    "ami_held_out_pilot",
    "alimeeting_eval_pilot",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the committed Phase 2 manifests from the repo data dir against the "
            "external wav roots and write data/results/phase2_d{1,2,3}_validation.json"
        )
    )
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument(
        "--wav-root",
        type=Path,
        action="append",
        default=None,
        help=(
            "extra wav root(s); defaults resolve to the repo data dir, the external "
            "phase 2 build root (<corpus root>/phase2_build), and the corpus root"
        ),
    )
    args = parser.parse_args()

    data_dir = args.data_dir or EXPERIMENT_DATA_DIR
    manifest_dir = data_dir / "manifests"
    if args.wav_root:
        wav_roots = list(args.wav_root)
    else:
        wav_roots = [data_dir, external.phase2_build_root(), external.corpus_root()]

    def load(manifest_id: str) -> Phase2Manifest:
        return Phase2Manifest.load(manifest_dir / f"{manifest_id}.json")

    def report_for(manifest: Phase2Manifest, counterpart_ids: list[str]) -> dict:
        entry = collect_validation_report(
            manifest,
            wav_roots,
            counterpart_manifests=[load(cid) for cid in counterpart_ids],
        )
        entry.update(manifest_identity_evidence(manifest_dir / f"{manifest.manifest_id}.json"))
        return entry

    d1_report = {
        manifest_id: report_for(load(manifest_id), [m for m in D1_IDS if m != manifest_id])
        for manifest_id in D1_IDS
    }
    d2_report = {
        manifest_id: report_for(load(manifest_id), D2_COUNTERPARTS.get(manifest_id, []))
        for manifest_id in ["ami_dev_pilot", "ami_held_out_pilot", "alimeeting_eval_pilot"]
    }
    d3_report = report_for(load("mixed_dev_pool"), D3_HELD_OUT_IDS)

    results_dir = data_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "phase2_d1_validation.json": d1_report,
        "phase2_d2_validation.json": d2_report,
        "phase2_d3_validation.json": d3_report,
    }
    ok = True
    for name, payload in outputs.items():
        path = results_dir / name
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        print(f"validation report: {path}")
    for name, entry in d1_report.items():
        passed = entry["valid"] and entry["manifest_canonical_bytes_ok"]
        ok = ok and passed
        print(f"{name}: {'PASS' if passed else 'FAIL'}")
    for name, entry in d2_report.items():
        passed = entry["valid"] and entry["manifest_canonical_bytes_ok"]
        ok = ok and passed
        print(f"{name}: {'PASS' if passed else 'FAIL'}")
    d3_passed = d3_report["valid"] and d3_report["manifest_canonical_bytes_ok"]
    ok = ok and d3_passed
    print(f"mixed_dev_pool: {'PASS' if d3_passed else 'FAIL'}")
    for evidence in d3_report["global_actor_disjointness"]:
        print(
            "AMI global actors: "
            f"{len(evidence['global_actors_a'])} dev vs "
            f"{len(evidence['global_actors_b'])} held-out, "
            f"overlap={evidence['overlap'] if evidence['overlap'] else 'none'}"
        )
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
