from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.speaker_turn_boundary.corpus import external
from experiments.speaker_turn_boundary.corpus.librispeech import (
    acquire_librispeech,
    build_librispeech_manifest,
    build_split_index,
)
from experiments.speaker_turn_boundary.corpus.validation import (
    collect_validation_report,
    manifest_identity_evidence,
)

MANIFEST_MAP = {
    "dev-clean": "ls_dev",
    "test-clean": "ls_held_out_clean",
    "test-other": "ls_held_out_other",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Phase 2 D1 LibriSpeech-based deterministic synthetic manifests"
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["dev-clean", "test-clean", "test-other"],
        help="LibriSpeech splits to build (default: all three)",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="external corpus root (default: $TEMP/opencode/stb_phase2_corpora)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            "output dir for manifests/generated wavs/validation reports "
            "(default: external phase 2 build root <corpus root>/phase2_build)"
        ),
    )
    parser.add_argument("--skip-download", action="store_true", help="do not download archives")
    parser.add_argument("--validate", action="store_true", help="run full validation after build")
    args = parser.parse_args()

    root = args.root or external.corpus_root()
    corpus_dir = root / "LibriSpeech"
    if not args.skip_download:
        corpus_dir = acquire_librispeech(root)
    out_dir = args.out or external.phase2_build_root()
    manifests: list[tuple[str, object]] = []
    for split in args.splits:
        index = build_split_index(corpus_dir, split)
        manifest_id = MANIFEST_MAP[split]
        manifest = build_librispeech_manifest(
            split=split,
            manifest_id=manifest_id,
            out_dir=out_dir,
            index=index,
            seed=args.seed,
        )
        manifest_path = out_dir / "manifests" / f"{manifest_id}.json"
        print(f"built {manifest_id}: {len(manifest.cases)} cases, hash={manifest.hash}")
        print(f"  manifest file: {manifest_path}")
        manifests.append((manifest_id, manifest))
    if args.validate:
        report = {}
        for manifest_id, manifest in manifests:
            counterparts = [m for other_id, m in manifests if other_id != manifest_id]
            report[manifest_id] = collect_validation_report(
                manifest, out_dir, counterpart_manifests=counterparts
            )
            report[manifest_id].update(
                manifest_identity_evidence(out_dir / "manifests" / f"{manifest_id}.json")
            )
        report_path = out_dir / "results" / "phase2_d1_validation.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        print(f"validation report: {report_path}")
        ok = all(
            entry["valid"] and entry["manifest_canonical_bytes_ok"] for entry in report.values()
        )
        print("D1 validation:", "PASS" if ok else "FAIL")
        if not ok:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
