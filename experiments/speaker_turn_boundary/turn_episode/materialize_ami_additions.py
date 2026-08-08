"""Materialize the Phase 1 addendum AMI audio additions (approved plan).

Downloads the 8 frozen-selected AMI Mix-Headset wavs from the AMI corpus mirror, verifies
each as 16 kHz mono PCM16 with annotation-consistent duration, records per-file SHA-256 in
a materialization manifest, and refuses to proceed when the selected meeting set changes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.speaker_turn_boundary.corpus import external
from experiments.speaker_turn_boundary.corpus.ami import (
    _load_meetings_xml,
    ami_mirror_url,
)
from experiments.speaker_turn_boundary.vad_baseline import load_canonical_wav

PLAN_BLOB = "24340f488f1bb46c666a5fc15eef2fc87ef1f826"

# Frozen selection (Phase 1 addendum): series-unique, hash-stratified, 8 meetings.
AMI_ADDITIONS = (
    "ES2010d",
    "EN2002c",
    "ES2006c",
    "IN1014",
    "TS3006a",
    "IS1007d",
    "EN2001d",
    "IS1006b",
)

DURATION_TOLERANCE_S = 2.0


class MaterializationError(RuntimeError):
    pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize approved AMI audio additions")
    parser.add_argument(
        "--corpus-root",
        type=Path,
        default=None,
        help="corpus root (default: STB_PHASE2_CORPORA_ROOT or TEMP/opencode/stb_phase2_corpora)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="manifest output directory (default: results/turn_episode_v1)",
    )
    args = parser.parse_args()

    corpus_root = args.corpus_root or external.corpus_root()
    if not corpus_root.is_dir():
        raise MaterializationError(f"corpus root not found: {corpus_root}")
    if args.out is None:
        args.out = Path(__file__).resolve().parent.parent / "results" / "turn_episode_v1"
    args.out.mkdir(parents=True, exist_ok=True)

    annotations_dir = corpus_root / "ami" / "annotations"
    meetings_meta = _load_meetings_xml(annotations_dir)

    entries: dict[str, dict[str, object]] = {}
    for meeting_id in AMI_ADDITIONS:
        meta = meetings_meta.get(meeting_id) or {}
        duration_s = float(meta.get("duration_s") or 0.0)
        if duration_s <= 0:
            raise MaterializationError(f"no annotation duration for {meeting_id}")
        destination = corpus_root / "ami" / "audio" / meeting_id / f"{meeting_id}.Mix-Headset.wav"
        url = ami_mirror_url(meeting_id)
        external.download_file(url, destination, timeout_seconds=120)
        try:
            samples = load_canonical_wav(destination)
        except Exception as exc:  # CanonicalAudioError
            raise MaterializationError(f"{meeting_id}: invalid canonical wav: {exc}") from exc
        decoded_duration_s = samples.size / 16000.0
        if abs(decoded_duration_s - duration_s) > DURATION_TOLERANCE_S:
            raise MaterializationError(
                f"{meeting_id}: decoded {decoded_duration_s:.1f}s vs annotation "
                f"{duration_s:.1f}s (tolerance {DURATION_TOLERANCE_S}s)"
            )
        entries[meeting_id] = {
            "url": url,
            "destination": str(destination),
            "decoded_duration_s": round(decoded_duration_s, 3),
            "annotation_duration_s": duration_s,
            "sha256": external.sha256_file(destination),
            "size_bytes": destination.stat().st_size,
        }
        print(f"materialized {meeting_id} {entries[meeting_id]['sha256'][:12]}")

    manifest = {
        "schema_version": "turn_episode_v1.ami_materialization",
        "plan_blob": PLAN_BLOB,
        "selection_rule": (
            "eligible=local words.xml annotations, series not in "
            "{ES2003,ES2004,IS1008,IS1009}; order by sha256(meeting_id); "
            "series-unique; stop at 8"
        ),
        "meetings": entries,
    }
    out_path = args.out / "ami_materialization_manifest.json"
    out_path.write_text(
        json.dumps(manifest, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
