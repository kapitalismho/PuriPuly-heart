from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from experiments.speaker_turn_boundary.config import BASELINE_SHA, CANONICAL_SAMPLE_RATE_HZ
from experiments.speaker_turn_boundary.ground_truth import SpeakerRegion
from experiments.speaker_turn_boundary.schemas import canonical_json, sha256_hex
from experiments.speaker_turn_boundary.vad_baseline import (
    CanonicalAudioError,
    load_canonical_wav,
)

PHASE2_MANIFEST_SCHEMA = "experiments.speaker_turn_boundary.manifest.phase2.v1"
PURIPULY_IMPORT_SCHEMA = "experiments.speaker_turn_boundary.puripuly_import.v1"


class Phase2SchemaError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class SourceRef:
    role: str
    speaker: str
    session: str
    utterance: str
    file_sha256: str
    original_start_sample: int
    original_end_sample: int
    trimmed_start_sample: int
    trimmed_end_sample: int
    cut_start_sample: int
    cut_end_sample: int
    gain: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "speaker": self.speaker,
            "session": self.session,
            "utterance": self.utterance,
            "file_sha256": self.file_sha256,
            "original_start_sample": self.original_start_sample,
            "original_end_sample": self.original_end_sample,
            "trimmed_start_sample": self.trimmed_start_sample,
            "trimmed_end_sample": self.trimmed_end_sample,
            "cut_start_sample": self.cut_start_sample,
            "cut_end_sample": self.cut_end_sample,
            "gain": self.gain,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SourceRef":
        return cls(
            role=str(data["role"]),
            speaker=str(data["speaker"]),
            session=str(data["session"]),
            utterance=str(data["utterance"]),
            file_sha256=str(data["file_sha256"]),
            original_start_sample=int(data["original_start_sample"]),
            original_end_sample=int(data["original_end_sample"]),
            trimmed_start_sample=int(data["trimmed_start_sample"]),
            trimmed_end_sample=int(data["trimmed_end_sample"]),
            cut_start_sample=int(data["cut_start_sample"]),
            cut_end_sample=int(data["cut_end_sample"]),
            gain=float(data["gain"]),
        )


@dataclass(frozen=True, slots=True)
class SpliceSpec:
    a_end_sample: int
    b_onset_sample: int
    gap_samples: int | None
    overlap_samples: int | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "a_end_sample": self.a_end_sample,
            "b_onset_sample": self.b_onset_sample,
            "gap_samples": self.gap_samples,
            "overlap_samples": self.overlap_samples,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SpliceSpec":
        return cls(
            a_end_sample=int(data["a_end_sample"]),
            b_onset_sample=int(data["b_onset_sample"]),
            gap_samples=None if data.get("gap_samples") is None else int(data["gap_samples"]),
            overlap_samples=(
                None if data.get("overlap_samples") is None else int(data["overlap_samples"])
            ),
        )


@dataclass(frozen=True, slots=True)
class TransformSpec:
    name: str
    params: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "params": self.params}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TransformSpec":
        return cls(name=str(data["name"]), params=dict(data.get("params") or {}))


@dataclass(frozen=True, slots=True)
class ZeroGapEvidence:
    b_onset_is_a_end: bool
    pre_junction_rms: float
    post_junction_rms: float
    junction_peak_abs: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "b_onset_is_a_end": self.b_onset_is_a_end,
            "pre_junction_rms": self.pre_junction_rms,
            "post_junction_rms": self.post_junction_rms,
            "junction_peak_abs": self.junction_peak_abs,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ZeroGapEvidence":
        return cls(
            b_onset_is_a_end=bool(data["b_onset_is_a_end"]),
            pre_junction_rms=float(data["pre_junction_rms"]),
            post_junction_rms=float(data["post_junction_rms"]),
            junction_peak_abs=float(data["junction_peak_abs"]),
        )


@dataclass(frozen=True, slots=True)
class Phase2Case:
    case_id: str
    wav_relative_path: str
    duration_samples: int
    wav_sha256: str
    seed: int
    regions: list[SpeakerRegion] = field(default_factory=list)
    kind: str = ""
    condition: dict[str, Any] = field(default_factory=dict)
    sources: list[SourceRef] = field(default_factory=list)
    splice: SpliceSpec | None = None
    transforms: list[TransformSpec] = field(default_factory=list)
    zero_gap_evidence: ZeroGapEvidence | None = None
    active_speech_samples: int = 0

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "case_id": self.case_id,
            "wav_relative_path": self.wav_relative_path,
            "duration_samples": self.duration_samples,
            "wav_sha256": self.wav_sha256,
            "seed": self.seed,
            "regions": [region.to_dict() for region in self.regions],
            "kind": self.kind,
            "condition": self.condition,
            "sources": [source.to_dict() for source in self.sources],
            "transforms": [transform.to_dict() for transform in self.transforms],
            "active_speech_samples": self.active_speech_samples,
        }
        if self.splice is not None:
            data["splice"] = self.splice.to_dict()
        if self.zero_gap_evidence is not None:
            data["zero_gap_evidence"] = self.zero_gap_evidence.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Phase2Case":
        return cls(
            case_id=str(data["case_id"]),
            wav_relative_path=str(data["wav_relative_path"]),
            duration_samples=int(data["duration_samples"]),
            wav_sha256=str(data["wav_sha256"]),
            seed=int(data["seed"]),
            regions=[SpeakerRegion.from_dict(region) for region in data.get("regions") or []],
            kind=str(data.get("kind", "")),
            condition=dict(data.get("condition") or {}),
            sources=[SourceRef.from_dict(source) for source in data.get("sources") or []],
            splice=None if data.get("splice") is None else SpliceSpec.from_dict(data["splice"]),
            transforms=[TransformSpec.from_dict(t) for t in data.get("transforms") or []],
            zero_gap_evidence=(
                None
                if data.get("zero_gap_evidence") is None
                else ZeroGapEvidence.from_dict(data["zero_gap_evidence"])
            ),
            active_speech_samples=int(data.get("active_speech_samples", 0)),
        )

    def to_v1_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "wav_relative_path": self.wav_relative_path,
            "duration_samples": self.duration_samples,
            "wav_sha256": self.wav_sha256,
            "seed": self.seed,
            "regions": [region.to_dict() for region in self.regions],
        }


@dataclass(frozen=True, slots=True)
class Phase2Manifest:
    manifest_id: str
    schema_version: str
    split_role: str
    baseline_sha: str
    canonical_sample_rate_hz: int
    corpus: dict[str, Any]
    build: dict[str, Any]
    disjointness_groups: list[str] = field(default_factory=list)
    generator: dict[str, Any] = field(default_factory=dict)
    cases: list[Phase2Case] = field(default_factory=list)

    @property
    def hash(self) -> str:
        return sha256_hex(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_id": self.manifest_id,
            "schema_version": self.schema_version,
            "split_role": self.split_role,
            "baseline_sha": self.baseline_sha,
            "canonical_sample_rate_hz": self.canonical_sample_rate_hz,
            "corpus": self.corpus,
            "build": self.build,
            "disjointness_groups": self.disjointness_groups,
            "generator": self.generator,
            "cases": [case.to_dict() for case in self.cases],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Phase2Manifest":
        return cls(
            manifest_id=str(data["manifest_id"]),
            schema_version=str(data["schema_version"]),
            split_role=str(data["split_role"]),
            baseline_sha=str(data["baseline_sha"]),
            canonical_sample_rate_hz=int(data["canonical_sample_rate_hz"]),
            corpus=dict(data["corpus"]),
            build=dict(data["build"]),
            disjointness_groups=[str(g) for g in data.get("disjointness_groups") or []],
            generator=dict(data.get("generator") or {}),
            cases=[Phase2Case.from_dict(case) for case in data.get("cases") or []],
        )

    @classmethod
    def load(cls, path: Path) -> "Phase2Manifest":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def write(self, path: Path) -> str:
        content = canonical_json(self.to_dict())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return self.hash


def write_phase2_manifest(path: Path, manifest: Phase2Manifest) -> str:
    return manifest.write(path)


def expected_change_kind(condition: dict[str, Any], kind: str) -> str | None:
    if kind == "different_speaker_overlap":
        return "interruption_onset"
    if kind == "different_speaker_gap":
        if condition.get("gap_ms") == 0:
            return "clean_handoff"
        return "gap_speaker_change"
    if kind in {"same_speaker", "gain_variation", "silence", "noise_only"}:
        return None
    return None


def validate_phase2_manifest(manifest: Phase2Manifest, wav_roots: Path | list[Path]) -> list[str]:
    if isinstance(wav_roots, Path):
        roots = [wav_roots]
    else:
        roots = list(wav_roots)
    errors: list[str] = []
    if manifest.schema_version != PHASE2_MANIFEST_SCHEMA:
        errors.append(f"unsupported schema {manifest.schema_version}")
    if manifest.canonical_sample_rate_hz != CANONICAL_SAMPLE_RATE_HZ:
        errors.append("manifest canonical sample rate must be 16000")
    seen_ids: set[str] = set()
    for case in manifest.cases:
        if case.case_id in seen_ids:
            errors.append(f"duplicate case_id {case.case_id}")
        seen_ids.add(case.case_id)
        if not case.wav_relative_path:
            continue
        wav_path: Path | None = None
        for root in roots:
            candidate = (root / case.wav_relative_path).resolve()
            if candidate.is_file():
                wav_path = candidate
                break
        if wav_path is None:
            errors.append(f"case {case.case_id}: wav missing at {case.wav_relative_path}")
            continue
        try:
            samples = load_canonical_wav(wav_path)
        except CanonicalAudioError as exc:
            errors.append(f"case {case.case_id}: invalid canonical wav: {exc}")
            continue
        if samples.size != case.duration_samples:
            errors.append(
                f"case {case.case_id}: duration mismatch ({samples.size} != {case.duration_samples})"
            )
        actual_hash = hashlib.sha256(wav_path.read_bytes()).hexdigest()
        if actual_hash != case.wav_sha256:
            errors.append(f"case {case.case_id}: wav sha256 mismatch")
        if case.regions:
            total = sum(r.end_sample - r.start_sample for r in case.regions)
            if total != case.duration_samples:
                errors.append(f"case {case.case_id}: regions do not tile the wav")
            for region in case.regions:
                if region.start_sample < 0 or region.end_sample > case.duration_samples:
                    errors.append(f"case {case.case_id}: region out of bounds")
    return errors


def make_phase2_manifest(
    *,
    manifest_id: str,
    split_role: str,
    corpus: dict[str, Any],
    build: dict[str, Any],
    disjointness_groups: list[str],
    generator: dict[str, Any],
    cases: list[Phase2Case],
) -> Phase2Manifest:
    return Phase2Manifest(
        manifest_id=manifest_id,
        schema_version=PHASE2_MANIFEST_SCHEMA,
        split_role=split_role,
        baseline_sha=BASELINE_SHA,
        canonical_sample_rate_hz=CANONICAL_SAMPLE_RATE_HZ,
        corpus=corpus,
        build=build,
        disjointness_groups=disjointness_groups,
        generator=generator,
        cases=cases,
    )
