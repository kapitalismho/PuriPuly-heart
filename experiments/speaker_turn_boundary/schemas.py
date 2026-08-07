from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from experiments.speaker_turn_boundary.config import (
    CANONICAL_SAMPLE_RATE_HZ,
    MANIFEST_SCHEMA_VERSION,
)
from experiments.speaker_turn_boundary.ground_truth import SpeakerRegion
from experiments.speaker_turn_boundary.vad_baseline import (
    CanonicalAudioError,
    load_canonical_wav,
)


class SchemaValidationError(ValueError):
    pass


def canonical_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, indent=2, ensure_ascii=False)


def sha256_hex(data: Any) -> str:
    return hashlib.sha256(canonical_json(data).encode("utf-8")).hexdigest()


def write_canonical_json(path: Path, data: Any) -> str:
    content = canonical_json(data)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return sha256_hex(data)


@dataclass(frozen=True, slots=True)
class ManifestCase:
    case_id: str
    wav_relative_path: str
    duration_samples: int
    wav_sha256: str
    seed: int
    regions: list[SpeakerRegion] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "wav_relative_path": self.wav_relative_path,
            "duration_samples": self.duration_samples,
            "wav_sha256": self.wav_sha256,
            "seed": self.seed,
            "regions": [region.to_dict() for region in self.regions],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ManifestCase":
        return cls(
            case_id=str(data["case_id"]),
            wav_relative_path=str(data["wav_relative_path"]),
            duration_samples=int(data["duration_samples"]),
            wav_sha256=str(data["wav_sha256"]),
            seed=int(data["seed"]),
            regions=[SpeakerRegion.from_dict(region) for region in data.get("regions") or []],
        )


@dataclass(frozen=True, slots=True)
class DatasetManifest:
    manifest_id: str
    schema_version: str
    baseline_sha: str
    canonical_sample_rate_hz: int
    generator: dict[str, Any]
    cases: list[ManifestCase] = field(default_factory=list)

    @property
    def hash(self) -> str:
        return sha256_hex(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_id": self.manifest_id,
            "schema_version": self.schema_version,
            "baseline_sha": self.baseline_sha,
            "canonical_sample_rate_hz": self.canonical_sample_rate_hz,
            "generator": self.generator,
            "cases": [case.to_dict() for case in self.cases],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DatasetManifest":
        return cls(
            manifest_id=str(data["manifest_id"]),
            schema_version=str(data["schema_version"]),
            baseline_sha=str(data["baseline_sha"]),
            canonical_sample_rate_hz=int(data["canonical_sample_rate_hz"]),
            generator=dict(data["generator"]),
            cases=[ManifestCase.from_dict(case) for case in data.get("cases") or []],
        )

    @classmethod
    def load(cls, path: Path) -> "DatasetManifest":
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_dict(data)


def validate_manifest(manifest: DatasetManifest, wav_root: Path) -> None:
    if manifest.schema_version != MANIFEST_SCHEMA_VERSION:
        raise SchemaValidationError(f"unsupported manifest schema {manifest.schema_version}")
    if manifest.canonical_sample_rate_hz != CANONICAL_SAMPLE_RATE_HZ:
        raise SchemaValidationError(
            "manifest canonical_sample_rate_hz must be "
            f"{CANONICAL_SAMPLE_RATE_HZ}, got {manifest.canonical_sample_rate_hz}"
        )
    for case in manifest.cases:
        wav_path = (wav_root / case.wav_relative_path).resolve()
        if not wav_path.is_file():
            raise SchemaValidationError(
                f"case {case.case_id}: wav missing at {case.wav_relative_path}"
            )
        try:
            samples = load_canonical_wav(wav_path)
        except CanonicalAudioError as exc:
            raise SchemaValidationError(
                f"case {case.case_id}: invalid canonical wav: {exc}"
            ) from exc
        if samples.size != case.duration_samples:
            raise SchemaValidationError(
                f"case {case.case_id}: duration mismatch "
                f"({samples.size} != {case.duration_samples})"
            )
        actual_hash = hashlib.sha256(wav_path.read_bytes()).hexdigest()
        if actual_hash != case.wav_sha256:
            raise SchemaValidationError(
                f"case {case.case_id}: wav sha256 mismatch " f"({actual_hash} != {case.wav_sha256})"
            )


@dataclass(frozen=True, slots=True)
class RunResult:
    result_id: str
    schema_version: str
    baseline_sha: str
    profile_id: str
    manifest_id: str
    manifest_sha256: str
    seed: int
    runtime_metadata: dict[str, Any]
    started_at_utc: str
    finished_at_utc: str
    epochs: list[dict[str, Any]]
    coalescing: dict[str, Any]
    result_sha256: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "result_id": self.result_id,
            "schema_version": self.schema_version,
            "baseline_sha": self.baseline_sha,
            "profile_id": self.profile_id,
            "manifest_id": self.manifest_id,
            "manifest_sha256": self.manifest_sha256,
            "seed": self.seed,
            "runtime_metadata": self.runtime_metadata,
            "started_at_utc": self.started_at_utc,
            "finished_at_utc": self.finished_at_utc,
            "epochs": self.epochs,
            "coalescing": self.coalescing,
        }

    def with_self_hash(self) -> "RunResult":
        return RunResult(
            result_id=self.result_id,
            schema_version=self.schema_version,
            baseline_sha=self.baseline_sha,
            profile_id=self.profile_id,
            manifest_id=self.manifest_id,
            manifest_sha256=self.manifest_sha256,
            seed=self.seed,
            runtime_metadata=self.runtime_metadata,
            started_at_utc=self.started_at_utc,
            finished_at_utc=self.finished_at_utc,
            epochs=self.epochs,
            coalescing=self.coalescing,
            result_sha256=sha256_hex(self.to_dict()),
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunResult":
        return cls(
            result_id=str(data["result_id"]),
            schema_version=str(data["schema_version"]),
            baseline_sha=str(data["baseline_sha"]),
            profile_id=str(data["profile_id"]),
            manifest_id=str(data["manifest_id"]),
            manifest_sha256=str(data["manifest_sha256"]),
            seed=int(data["seed"]),
            runtime_metadata=dict(data["runtime_metadata"]),
            started_at_utc=str(data["started_at_utc"]),
            finished_at_utc=str(data["finished_at_utc"]),
            epochs=list(data["epochs"]),
            coalescing=dict(data["coalescing"]),
            result_sha256=str(data.get("result_sha256", "")),
        )

    def verify_self_hash(self) -> bool:
        return self.result_sha256 == sha256_hex(self.to_dict())

    def write(self, path: Path) -> str:
        hashed = self.with_self_hash()
        data = dict(hashed.to_dict())
        data["result_sha256"] = hashed.result_sha256
        write_canonical_json(path, data)
        return hashed.result_sha256
