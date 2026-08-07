from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from experiments.speaker_turn_boundary.config import CANONICAL_SAMPLE_RATE_HZ
from experiments.speaker_turn_boundary.corpus.phase2_schemas import (
    PURIPULY_IMPORT_SCHEMA,
    Phase2Manifest,
    make_phase2_manifest,
)
from experiments.speaker_turn_boundary.ground_truth import SpeakerRegion

D4_REQUIRED_MINUTES = 20.0


@dataclass(slots=True)
class PuripulyImportCase:
    case_id: str
    wav_path: str
    duration_samples: int
    wav_sha256: str
    language: str
    condition: dict[str, Any]
    regions: list[SpeakerRegion] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "wav_path": self.wav_path,
            "duration_samples": self.duration_samples,
            "wav_sha256": self.wav_sha256,
            "language": self.language,
            "condition": self.condition,
            "regions": [region.to_dict() for region in self.regions],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PuripulyImportCase":
        return cls(
            case_id=str(data["case_id"]),
            wav_path=str(data["wav_path"]),
            duration_samples=int(data["duration_samples"]),
            wav_sha256=str(data["wav_sha256"]),
            language=str(data.get("language", "")),
            condition=dict(data.get("condition") or {}),
            regions=[SpeakerRegion.from_dict(r) for r in data.get("regions") or []],
        )


def write_puripuly_import_template(path: Path) -> dict[str, Any]:
    template = {
        "schema_version": PURIPULY_IMPORT_SCHEMA,
        "canonical_sample_rate_hz": CANONICAL_SAMPLE_RATE_HZ,
        "language_note": "Korean / Japanese / English / mixed",
        "condition_note": "game/voice-chat/Opus, different mics/gains, short reactions, no-gap handoff, interruption/overlap",
        "privacy": "raw audio stays outside Git; only hashes, sample-exact regions, and metadata enter the repo",
        "cases": [],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(template, indent=2, sort_keys=True), encoding="utf-8")
    return template


def check_authorized_inputs(authorized_roots: list[Path]) -> dict[str, Any]:
    found: list[dict[str, Any]] = []
    total_seconds = 0.0
    for root in authorized_roots:
        if not root.is_dir():
            continue
        for wav_path in sorted(root.rglob("*.wav")):
            import wave as wave_module

            with wave_module.open(str(wav_path), "rb") as handle:
                seconds = handle.getnframes() / max(handle.getframerate(), 1)
                sample_rate_hz = handle.getframerate()
                channels = handle.getnchannels()
            found.append(
                {
                    "path": str(wav_path),
                    "seconds": round(seconds, 3),
                    "sample_rate_hz": sample_rate_hz,
                    "channels": channels,
                }
            )
            total_seconds += seconds
    return {
        "authorized_inputs_found": len(found),
        "total_seconds": round(total_seconds, 3),
        "meets_20_30_minutes": total_seconds >= D4_REQUIRED_MINUTES * 60.0,
        "inputs": found,
    }


def make_provisional_puripuly_manifest(
    *,
    manifest_id: str,
    out_dir: Path,
    availability: dict[str, Any],
    annotation_note: str,
) -> Phase2Manifest:
    manifest = make_phase2_manifest(
        manifest_id=manifest_id,
        split_role="acceptance_provisional",
        corpus={
            "name": "puripuly_like",
            "availability": availability,
            "annotation_note": annotation_note,
            "import_schema": PURIPULY_IMPORT_SCHEMA,
            "import_template": "corpus/puripuly_like.py:write_puripuly_import_template",
        },
        build={
            "script": "corpus.puripuly_like.make_provisional_puripuly_manifest",
            "status": "provisional_no_audio",
        },
        disjointness_groups=[],
        generator={"script": "build_phase2_real.py"},
        cases=[],
    )
    manifest_path = out_dir / "manifests" / f"{manifest_id}.json"
    manifest.write(manifest_path)
    return manifest
