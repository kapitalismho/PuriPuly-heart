from __future__ import annotations

import argparse
import hashlib
import wave
from pathlib import Path

import numpy as np

from experiments.speaker_turn_boundary.config import (
    BASELINE_SHA,
    CANONICAL_SAMPLE_RATE_HZ,
    DEFAULT_GENERATOR_SEED,
    EXPERIMENT_DATA_DIR,
    MANIFEST_SCHEMA_VERSION,
)
from experiments.speaker_turn_boundary.ground_truth import SpeakerRegion
from experiments.speaker_turn_boundary.schemas import (
    DatasetManifest,
    ManifestCase,
    write_canonical_json,
)
from experiments.speaker_turn_boundary.synthetic import (
    CaseSpec,
    build_default_cases,
    case_regions,
    pcm16_bytes,
)


def _write_wav(path: Path, samples: np.ndarray, *, sample_rate_hz: int) -> None:
    pcm = pcm16_bytes(samples)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate_hz)
        wav_file.writeframes(pcm)


def _manifest_case(case: CaseSpec, wav_path: Path, wav_relative_path: str) -> ManifestCase:
    samples = np.asarray(case.audio, dtype=np.float32)
    regions = [
        SpeakerRegion(
            audio_epoch=0,
            start_sample=int(region["start_sample"]),
            end_sample=int(region["end_sample"]),
            speakers=frozenset(region["speakers"]),
        )
        for region in case_regions(case)
    ]
    return ManifestCase(
        case_id=case.case_id,
        wav_relative_path=wav_relative_path,
        duration_samples=int(samples.size),
        wav_sha256=hashlib.sha256(wav_path.read_bytes()).hexdigest(),
        seed=case.seed,
        regions=regions,
    )


def build(out_dir: Path, *, seed: int, manifest_id: str) -> DatasetManifest:
    cases = build_default_cases(seed=seed)
    manifest_cases: list[ManifestCase] = []
    for case in cases:
        wav_relative_path = f"generated/{case.case_id}.wav"
        wav_path = out_dir / wav_relative_path
        _write_wav(
            wav_path,
            np.asarray(case.audio, dtype=np.float32),
            sample_rate_hz=CANONICAL_SAMPLE_RATE_HZ,
        )
        manifest_cases.append(_manifest_case(case, wav_path, wav_relative_path))
    manifest = DatasetManifest(
        manifest_id=manifest_id,
        schema_version=MANIFEST_SCHEMA_VERSION,
        baseline_sha=BASELINE_SHA,
        canonical_sample_rate_hz=CANONICAL_SAMPLE_RATE_HZ,
        generator={"script": "build_synthetic_cases.py", "seed": seed},
        cases=manifest_cases,
    )
    manifest_path = out_dir / "manifests" / f"{manifest_id}.json"
    write_canonical_json(manifest_path, manifest.to_dict())
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build deterministic Phase 0 synthetic cases and dataset manifest"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=EXPERIMENT_DATA_DIR,
        help="experiment data directory (default: experiments/speaker_turn_boundary/data)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_GENERATOR_SEED,
        help="deterministic generator seed (default: %(default)s)",
    )
    parser.add_argument(
        "--manifest-id",
        default="b0_phase0",
        help="manifest id (default: %(default)s)",
    )
    args = parser.parse_args()
    manifest = build(args.out, seed=args.seed, manifest_id=args.manifest_id)
    print(f"wrote manifest {args.out / 'manifests' / f'{args.manifest_id}.json'}")
    print(f"manifest_id={manifest.manifest_id} hash={manifest.hash}")
    for case in manifest.cases:
        print(
            f"case {case.case_id}: duration_samples={case.duration_samples} "
            f"wav_sha256={case.wav_sha256}"
        )


if __name__ == "__main__":
    main()
