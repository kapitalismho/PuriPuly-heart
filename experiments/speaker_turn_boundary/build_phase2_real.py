from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.speaker_turn_boundary.corpus import external
from experiments.speaker_turn_boundary.corpus.alimeeting import (
    acquire_alimeeting_eval,
    build_alimeeting_manifest,
)
from experiments.speaker_turn_boundary.corpus.ami import (
    AMI_DEV_PILOT_SESSIONS,
    AMI_HELD_OUT_PILOT_SESSIONS,
    acquire_ami_annotations,
    acquire_ami_meetings,
    build_ami_manifest,
    load_ami_meeting,
)
from experiments.speaker_turn_boundary.corpus.puripuly_like import (
    check_authorized_inputs,
    make_provisional_puripuly_manifest,
    write_puripuly_import_template,
)
from experiments.speaker_turn_boundary.corpus.validation import (
    collect_validation_report,
    manifest_identity_evidence,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Phase 2 D2/D4 artifacts (AMI, AliMeeting, PuriPuly-like)"
    )
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
            "output dir for manifests/validation reports "
            "(default: external phase 2 build root <corpus root>/phase2_build)"
        ),
    )
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--alimeeting-session-ids", nargs="+", default=None)
    parser.add_argument(
        "--puripuly-authorized-roots",
        nargs="+",
        default=[],
        help="explicitly authorized local dirs to scan read-only for PuriPuly-like audio",
    )
    args = parser.parse_args()

    root = args.root or external.corpus_root()
    out_dir = args.out or external.phase2_build_root()
    if not args.skip_download:
        acquire_ami_annotations(root)
        acquire_ami_meetings(
            AMI_DEV_PILOT_SESSIONS + AMI_HELD_OUT_PILOT_SESSIONS,
            root,
        )
    ami_meta: dict[str, dict[str, str]] = {}
    for meeting_id in AMI_DEV_PILOT_SESSIONS + AMI_HELD_OUT_PILOT_SESSIONS:
        wav_path = root / "ami" / "audio" / meeting_id / f"{meeting_id}.Mix-Headset.wav"
        if wav_path.is_file():
            meeting = load_ami_meeting(
                meeting_id,
                "pilot",
                wav_path,
                root / "ami" / "annotations",
            )
            ami_meta[meeting_id] = {
                "participants": ",".join(meeting.participants),
                "duration_samples": str(meeting.duration_samples),
                "words": str(len(meeting.words)),
            }
    ami_dev = build_ami_manifest(
        meetings=AMI_DEV_PILOT_SESSIONS,
        split_role="pilot_dev",
        manifest_id="ami_dev_pilot",
        root=root,
        out_dir=out_dir,
    )
    ami_held = build_ami_manifest(
        meetings=AMI_HELD_OUT_PILOT_SESSIONS,
        split_role="pilot_held_out",
        manifest_id="ami_held_out_pilot",
        root=root,
        out_dir=out_dir,
    )
    print(f"built {ami_dev.manifest_id}: {len(ami_dev.cases)} cases, hash={ami_dev.hash}")
    print(f"built {ami_held.manifest_id}: {len(ami_held.cases)} cases, hash={ami_held.hash}")
    alimeeting_manifest = None
    try:
        eval_dir = root / "alimeeting"
        if not args.skip_download and not (eval_dir / "Eval_Ali").is_dir():
            acquire_alimeeting_eval(root)
        if eval_dir.is_dir():
            alimeeting_manifest = build_alimeeting_manifest(
                manifest_id="alimeeting_eval_pilot",
                split_role="pilot_held_out",
                root=root,
                out_dir=out_dir,
                session_ids=args.alimeeting_session_ids,
            )
            print(
                f"built {alimeeting_manifest.manifest_id}: "
                f"{len(alimeeting_manifest.cases)} cases, hash={alimeeting_manifest.hash}"
            )
    except Exception as exc:  # noqa: BLE001
        print(f"AliMeeting pilot BLOCKED: {exc!r}")
    availability = check_authorized_inputs([Path(p) for p in args.puripuly_authorized_roots])
    print("PuriPuly-like availability:", json.dumps(availability, indent=2))
    write_puripuly_import_template(out_dir / "puripuly_import_template.json")
    provisional = make_provisional_puripuly_manifest(
        manifest_id="puripuly_like_provisional",
        out_dir=out_dir,
        availability=availability,
        annotation_note=(
            "No authorized PuriPuly-like audio was available during Phase 2; "
            "detector/domain conclusions must remain provisional until the import "
            "schema is filled. Fill data/puripuly_import_template.json and rebuild."
        ),
    )
    print(
        f"built {provisional.manifest_id}: {len(provisional.cases)} cases, hash={provisional.hash}"
    )
    report: dict[str, object] = {}
    for manifest, counterparts in [
        (ami_dev, [ami_held]),
        (ami_held, [ami_dev]),
    ]:
        report[manifest.manifest_id] = collect_validation_report(
            manifest, root, counterpart_manifests=counterparts
        )
        report[manifest.manifest_id].update(
            manifest_identity_evidence(out_dir / "manifests" / f"{manifest.manifest_id}.json")
        )
    if alimeeting_manifest is not None:
        report[alimeeting_manifest.manifest_id] = collect_validation_report(
            alimeeting_manifest, root
        )
        report[alimeeting_manifest.manifest_id].update(
            manifest_identity_evidence(
                out_dir / "manifests" / f"{alimeeting_manifest.manifest_id}.json"
            )
        )
    report_path = out_dir / "results" / "phase2_d2_validation.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"validation report: {report_path}")


if __name__ == "__main__":
    main()
