from __future__ import annotations

import argparse
import hashlib
import wave
from pathlib import Path

import numpy as np

from experiments.speaker_turn_boundary.config import (
    BASELINE_SHA,
    CANONICAL_SAMPLE_RATE_HZ,
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
    FORMANT_VOWEL_A,
    FORMANT_VOWEL_I,
    FORMANT_VOWEL_O,
    CaseSpec,
    envelope_edges,
    formant_vowel,
    silence,
)

PHASE1_MANIFEST_ID = "phase1_dev"
PHASE1_SEED = 2026


def build_default_cases(seed: int = PHASE1_SEED) -> list[CaseSpec]:
    speech_a = formant_vowel(
        2.0,
        formants=FORMANT_VOWEL_A,
        seed=31,
        amplitude=1.2,
        sample_rate_hz=CANONICAL_SAMPLE_RATE_HZ,
    )
    speech_b = formant_vowel(
        1.6,
        formants=FORMANT_VOWEL_I,
        seed=32,
        amplitude=1.2,
        sample_rate_hz=CANONICAL_SAMPLE_RATE_HZ,
    )
    speech_o = formant_vowel(
        1.2,
        formants=FORMANT_VOWEL_O,
        seed=33,
        amplitude=1.2,
        sample_rate_hz=CANONICAL_SAMPLE_RATE_HZ,
    )
    lead = silence(0.5)
    tail = silence(0.5)
    mixed_ab = envelope_edges(
        (
            formant_vowel(
                0.5,
                formants=FORMANT_VOWEL_A,
                seed=34,
                amplitude=0.9,
                sample_rate_hz=CANONICAL_SAMPLE_RATE_HZ,
            )
            + formant_vowel(
                0.5,
                formants=FORMANT_VOWEL_I,
                seed=35,
                amplitude=0.9,
                sample_rate_hz=CANONICAL_SAMPLE_RATE_HZ,
            )
        )
        / 2.0
    )
    return [
        CaseSpec(
            case_id="zero_gap_handoff_ab",
            seed=seed,
            segments=[
                ("silence", lead),
                ("speech_a", speech_a),
                ("speech_b", speech_b),
                ("silence", tail),
            ],
        ),
        CaseSpec(
            case_id="overlap_300ms_ab",
            seed=seed,
            segments=[
                ("silence", lead),
                ("speech_a", speech_a),
                ("speech_ab", mixed_ab),
                ("speech_b", speech_o),
                ("silence", tail),
            ],
        ),
    ]


def case_regions(case: CaseSpec) -> list[dict[str, object]]:
    regions: list[dict[str, object]] = []
    start_sample = 0
    for kind, samples in case.segments:
        end_sample = start_sample + int(samples.size)
        if kind == "speech_a":
            speakers: list[str] = ["A"]
        elif kind == "speech_b":
            speakers = ["B"]
        elif kind == "speech_ab":
            speakers = ["A", "B"]
        else:
            speakers = []
        regions.append(
            {
                "start_sample": start_sample,
                "end_sample": end_sample,
                "speakers": speakers,
            }
        )
        start_sample = end_sample
    return regions


def pcm16_bytes(samples: np.ndarray) -> bytes:
    scaled = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
    pcm = np.round(scaled * 32767.0).astype(np.int16)
    return pcm.tobytes()


def _write_wav(path: Path, samples: np.ndarray, *, sample_rate_hz: int) -> None:
    pcm = pcm16_bytes(samples)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate_hz)
        wav_file.writeframes(pcm)


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
        regions = [
            SpeakerRegion(
                audio_epoch=0,
                start_sample=int(region["start_sample"]),
                end_sample=int(region["end_sample"]),
                speakers=frozenset(region["speakers"]),
            )
            for region in case_regions(case)
        ]
        manifest_cases.append(
            ManifestCase(
                case_id=case.case_id,
                wav_relative_path=wav_relative_path,
                duration_samples=int(np.asarray(case.audio).size),
                wav_sha256=hashlib.sha256(wav_path.read_bytes()).hexdigest(),
                seed=case.seed,
                regions=regions,
            )
        )
    manifest = DatasetManifest(
        manifest_id=manifest_id,
        schema_version=MANIFEST_SCHEMA_VERSION,
        baseline_sha=BASELINE_SHA,
        canonical_sample_rate_hz=CANONICAL_SAMPLE_RATE_HZ,
        generator={"script": "build_phase1_cases.py", "seed": seed},
        cases=manifest_cases,
    )
    manifest_path = out_dir / "manifests" / f"{manifest_id}.json"
    write_canonical_json(manifest_path, manifest.to_dict())
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Phase 1 deterministic dev cases (zero-gap handoff, overlap)"
    )
    parser.add_argument("--out", type=Path, default=EXPERIMENT_DATA_DIR)
    parser.add_argument("--seed", type=int, default=PHASE1_SEED)
    parser.add_argument("--manifest-id", default=PHASE1_MANIFEST_ID)
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
