from __future__ import annotations

from pathlib import Path

from experiments.speaker_turn_boundary.corpus.phase2_schemas import (
    Phase2Case,
    Phase2Manifest,
    make_phase2_manifest,
)
from experiments.speaker_turn_boundary.corpus.validation import (
    check_disjoint_global_actors,
    check_disjoint_sessions,
    check_disjoint_speakers,
    collect_validation_report,
)

MIXED_DEV_ROLE = "dev_pool"


def build_mixed_dev_manifest(
    *,
    manifest_id: str,
    out_dir: Path,
    source_manifests: list[Phase2Manifest],
    held_out_manifests: list[Phase2Manifest],
    wav_roots: list[Path],
) -> tuple[Phase2Manifest, list[str]]:
    cases: list[Phase2Case] = []
    groups: list[str] = []
    for source in source_manifests:
        cases.extend(source.cases)
        groups.extend(source.disjointness_groups)
    groups = sorted(set(groups))
    manifest = make_phase2_manifest(
        manifest_id=manifest_id,
        split_role=MIXED_DEV_ROLE,
        corpus={
            "name": "mixed_development_pool",
            "components": [source.manifest_id for source in source_manifests],
            "local_wav_root": str(out_dir),
            "wav_roots": [str(root) for root in wav_roots],
        },
        build={
            "script": "corpus.mixing.build_mixed_dev_manifest",
            "includes_synthetic_dev": any(
                "librispeech" in str(source.corpus.get("name")) for source in source_manifests
            ),
            "includes_real_meeting_dev": any(
                source.corpus.get("name") in {"ami", "alimeeting"} for source in source_manifests
            ),
            "includes_puripuly_like": any(
                source.corpus.get("name") == "puripuly_like" for source in source_manifests
            ),
        },
        disjointness_groups=groups,
        generator={"script": "build_phase2_mixed.py"},
        cases=cases,
    )
    manifest_path = out_dir / "manifests" / f"{manifest_id}.json"
    manifest.write(manifest_path)
    problems: list[str] = []
    for held_out in held_out_manifests:
        problems.extend(check_disjoint_speakers(manifest, held_out))
        problems.extend(check_disjoint_sessions(manifest, held_out))
        problems.extend(check_disjoint_global_actors(manifest, held_out))
    return manifest, problems


def mixed_dev_validation_report(
    manifest: Phase2Manifest,
    wav_roots: list[Path],
    held_out_manifests: list[Phase2Manifest],
) -> dict[str, object]:
    return collect_validation_report(
        manifest,
        wav_roots,
        counterpart_manifests=held_out_manifests,
    )
