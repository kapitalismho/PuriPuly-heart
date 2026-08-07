from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.speaker_turn_boundary.corpus import external
from experiments.speaker_turn_boundary.corpus.mixing import (
    build_mixed_dev_manifest,
    mixed_dev_validation_report,
)
from experiments.speaker_turn_boundary.corpus.phase2_schemas import Phase2Manifest
from experiments.speaker_turn_boundary.corpus.validation import manifest_identity_evidence


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Phase 2 D3 mixed development pool with disjointness checks"
    )
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            "output dir holding manifests/ and the mixed pool manifest "
            "(default: external phase 2 build root <corpus root>/phase2_build)"
        ),
    )
    args = parser.parse_args()

    root = args.root or external.corpus_root()
    out_dir = args.out or external.phase2_build_root()
    manifest_dir = out_dir / "manifests"

    def load(manifest_id: str) -> Phase2Manifest:
        return Phase2Manifest.load(manifest_dir / f"{manifest_id}.json")

    source_ids = ["ls_dev"]
    for real_dev in ["ami_dev_pilot"]:
        path = manifest_dir / f"{real_dev}.json"
        if path.is_file():
            source_ids.append(real_dev)
    sources = [load(manifest_id) for manifest_id in source_ids]
    held_out_ids: list[str] = []
    for held_out_id in [
        "ls_held_out_clean",
        "ls_held_out_other",
        "ami_held_out_pilot",
        "alimeeting_eval_pilot",
    ]:
        path = manifest_dir / f"{held_out_id}.json"
        if path.is_file():
            held_out_ids.append(held_out_id)
    held_out = [load(manifest_id) for manifest_id in held_out_ids]
    wav_roots = [out_dir, root]
    manifest, problems = build_mixed_dev_manifest(
        manifest_id="mixed_dev_pool",
        out_dir=out_dir,
        source_manifests=sources,
        held_out_manifests=held_out,
        wav_roots=wav_roots,
    )
    print(f"built {manifest.manifest_id}: {len(manifest.cases)} cases, hash={manifest.hash}")
    print(f"components: {[s.manifest_id for s in sources]}")
    print(f"held-out counterparts: {held_out_ids}")
    print("disjointness problems:", problems if problems else "none")
    report = mixed_dev_validation_report(manifest, wav_roots, held_out)
    report.update(manifest_identity_evidence(manifest_dir / f"{manifest.manifest_id}.json"))
    report_path = out_dir / "results" / "phase2_d3_validation.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"validation report: {report_path}")
    if problems:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
