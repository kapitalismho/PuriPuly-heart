from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from experiments.speaker_turn_boundary.config import CANONICAL_SAMPLE_RATE_HZ
from experiments.speaker_turn_boundary.corpus import external
from experiments.speaker_turn_boundary.corpus.external import (
    CorpusError,
    decode_flac_to_pcm16,
    ffprobe_duration_seconds,
)
from experiments.speaker_turn_boundary.corpus.phase2_schemas import (
    Phase2Case,
    Phase2Manifest,
    SourceRef,
    SpliceSpec,
    TransformSpec,
    ZeroGapEvidence,
    make_phase2_manifest,
)
from experiments.speaker_turn_boundary.ground_truth import SpeakerRegion, active_speech_sample_count

LIBRISPEECH_BASE_URL = "https://www.openslr.org/resources/12"
LIBRISPEECH_LICENSE = "CC BY 4.0"
LIBRISPEECH_SOURCE = "https://www.openslr.org/12/"

ARCHIVES: dict[str, dict[str, Any]] = {
    "dev-clean": {
        "url": f"{LIBRISPEECH_BASE_URL}/dev-clean.tar.gz",
        "md5": "42e2234ba48799c1f50f24a7926300a1",
    },
    "test-clean": {
        "url": f"{LIBRISPEECH_BASE_URL}/test-clean.tar.gz",
        "md5": "32fa31d27d2e1cad72775fee3f4849a9",
    },
    "test-other": {
        "url": f"{LIBRISPEECH_BASE_URL}/test-other.tar.gz",
        "md5": "fb5a50374b501bb3bac4815ee91d3135",
    },
}

DURATION_TARGETS = [2.0, 1.5, 1.0, 0.75, 0.5, "stress"]
STRESS_DURATION_RANGE = (0.30, 0.50)
GAPS_MS = [800, 300, 100, 0]
OVERLAPS_MS = [100, 300, 500]
LEAD_SILENCE_S = 0.1
TAIL_SILENCE_S = 0.1
TRIM_FRAME_MS = 10.0
TRIM_RELATIVE_RMS_THRESHOLD = 0.01
TRIM_ABS_RMS_FLOOR = 1e-3
TARGET_EXTRA_MARGIN_S = 0.4
OPUS_BITRATE_KBPS = 32
NOISE_SNR_DB = 15.0
BANDLIMIT_HZ = 6000.0
JUNCTION_GUARD_RMS = 2.5e-3
FINAL_JUNCTION_MIN_RMS = 1e-3
JUNCTION_WINDOW_SAMPLES = int(CANONICAL_SAMPLE_RATE_HZ * 40.0 / 1000.0)
MAX_JUNCTION_GUARD_ATTEMPTS = 20

CASE_COUNTS = {
    "positive_per_combo": 4,
    "same_speaker_per_combo": 2,
    "gain_per_combo": 1,
    "stress_per_combo": 1,
    "silence": 4,
    "noise_only": 4,
    "bandlimit": 2,
}

DURATION_LABELS: dict[str, str] = {
    "2.0": "d200",
    "1.5": "d150",
    "1.0": "d100",
    "0.75": "d075",
    "0.5": "d050",
    "stress": "d030",
}


@dataclass(frozen=True, slots=True)
class UtteranceInfo:
    split: str
    speaker: str
    chapter: str
    utterance_id: str
    path: Path
    transcript: str
    duration_seconds: float

    @property
    def session(self) -> str:
        return f"{self.speaker}-{self.chapter}"


@dataclass(slots=True)
class SplitIndex:
    split: str
    utterances: list[UtteranceInfo] = field(default_factory=list)
    by_id: dict[str, UtteranceInfo] = field(default_factory=dict)
    by_speaker: dict[str, list[UtteranceInfo]] = field(default_factory=dict)
    speakers: list[str] = field(default_factory=list)


def acquire_librispeech(root: Path | None = None) -> Path:
    import shutil
    import tempfile

    root = root or external.corpus_root()
    archive_dir = root / "archives"
    corpus_dir = root / "LibriSpeech"
    for split, spec in ARCHIVES.items():
        archive = archive_dir / f"{split}.tar.gz"
        external.download_file(spec["url"], archive, expected_md5=spec["md5"])
        target_split = corpus_dir / split
        if target_split.is_dir():
            continue
        with tempfile.TemporaryDirectory(
            prefix=f"librispeech_extract_{split}_", dir=str(root)
        ) as tmp:
            extracted = external.extract_tar_gz(archive, Path(tmp))
            extracted_root = extracted / "LibriSpeech" / split
            if not extracted_root.is_dir():
                raise CorpusError(f"unexpected LibriSpeech layout in {spec['url']}")
            target_split.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(extracted_root), str(target_split))
    return corpus_dir


def list_split_dirs(corpus_dir: Path, split: str) -> Path:
    candidate = corpus_dir / split
    if candidate.is_dir():
        return candidate
    return corpus_dir / "LibriSpeech" / split


def build_split_index(corpus_dir: Path, split: str) -> SplitIndex:
    split_dir = list_split_dirs(corpus_dir, split)
    if not split_dir.is_dir():
        raise CorpusError(f"LibriSpeech split {split} not found under {corpus_dir}")
    index = SplitIndex(split=split)
    for speaker_dir in sorted(split_dir.iterdir()):
        if not speaker_dir.is_dir():
            continue
        for chapter_dir in sorted(speaker_dir.iterdir()):
            transcript_path = chapter_dir / f"{speaker_dir.name}-{chapter_dir.name}.trans.txt"
            transcripts: dict[str, str] = {}
            if transcript_path.is_file():
                for line in transcript_path.read_text(encoding="utf-8").splitlines():
                    fields = line.split(" ", 1)
                    if len(fields) == 2:
                        transcripts[fields[0]] = fields[1]
            for flac_path in sorted(chapter_dir.glob("*.flac")):
                utterance_id = flac_path.stem
                duration = ffprobe_duration_seconds(flac_path)
                index.utterances.append(
                    UtteranceInfo(
                        split=split,
                        speaker=speaker_dir.name,
                        chapter=chapter_dir.name,
                        utterance_id=utterance_id,
                        path=flac_path,
                        transcript=transcripts.get(utterance_id, ""),
                        duration_seconds=duration,
                    )
                )
    index.utterances.sort(key=lambda u: u.utterance_id)
    for utterance in index.utterances:
        index.by_id[utterance.utterance_id] = utterance
        index.by_speaker.setdefault(utterance.speaker, []).append(utterance)
    index.speakers = sorted(index.by_speaker)
    return index


TRIM_WINDOW_SAMPLES = int(CANONICAL_SAMPLE_RATE_HZ * 40.0 / 1000.0)


def trim_energy(samples: np.ndarray) -> tuple[int, int] | None:
    n = int(samples.size)
    if n < TRIM_WINDOW_SAMPLES:
        return None
    squared = np.square(samples.astype(np.float64))
    cumulative = np.concatenate([[0.0], np.cumsum(squared)])
    window_sum = cumulative[TRIM_WINDOW_SAMPLES:] - cumulative[:-TRIM_WINDOW_SAMPLES]
    window_rms = np.sqrt(window_sum / TRIM_WINDOW_SAMPLES)
    peak_rms = float(window_rms.max())
    if peak_rms <= 0.0:
        return None
    threshold = max(peak_rms * TRIM_RELATIVE_RMS_THRESHOLD, TRIM_ABS_RMS_FLOOR)
    above = np.flatnonzero(window_rms >= threshold)
    if above.size == 0:
        return None
    start = int(above[0])
    end = min(n, int(above[-1]) + TRIM_WINDOW_SAMPLES)
    return start, end


def _utterance_sha256(utterance: UtteranceInfo) -> str:
    return hashlib.sha256(utterance.path.read_bytes()).hexdigest()


def _pick_speakers(
    rng: random.Random,
    speakers: list[str],
    *,
    same: bool,
) -> tuple[str, str]:
    if len(speakers) < 2:
        raise CorpusError(f"need at least 2 speakers, got {len(speakers)}")
    a = rng.choice(speakers)
    if same:
        return a, a
    others = [s for s in speakers if s != a]
    return a, rng.choice(others)


def _pick_utterance(
    rng: random.Random,
    index: SplitIndex,
    speaker: str,
    target_len: int,
    exclude_id: str | None = None,
) -> UtteranceInfo:
    candidates = [
        u
        for u in index.by_speaker.get(speaker, [])
        if u.utterance_id != exclude_id
        and u.duration_seconds >= (target_len / CANONICAL_SAMPLE_RATE_HZ) + TARGET_EXTRA_MARGIN_S
    ]
    if not candidates:
        raise CorpusError(f"speaker {speaker} has no utterance long enough for target {target_len}")
    candidates.sort(key=lambda u: u.utterance_id)
    start = rng.randrange(len(candidates))
    for offset in range(len(candidates)):
        candidate = candidates[(start + offset) % len(candidates)]
        samples = decode_flac_to_pcm16(candidate.path)
        trimmed = trim_energy(samples)
        if trimmed is not None and (trimmed[1] - trimmed[0]) >= target_len:
            return candidate
    raise CorpusError(f"speaker {speaker}: no utterance with >= {target_len} trimmed samples")


def _load_role_source(
    index: SplitIndex,
    role: str,
    speaker: str,
    target_len: int,
    rng: random.Random,
    exclude_utterance_id: str | None = None,
) -> tuple[UtteranceInfo, np.ndarray, int, int, int, int, int]:
    utterance = _pick_utterance(rng, index, speaker, target_len, exclude_id=exclude_utterance_id)
    samples = decode_flac_to_pcm16(utterance.path)
    trimmed = trim_energy(samples)
    if trimmed is None:
        raise CorpusError(f"utterance {utterance.utterance_id} has no speech frames")
    trim_start, trim_end = trimmed
    if trim_end - trim_start < target_len:
        raise CorpusError(f"utterance {utterance.utterance_id}: trimmed region shorter than target")
    if role == "B":
        cut = samples[trim_start : trim_start + target_len]
        cut_start = trim_start
        cut_end = trim_start + target_len
    else:
        cut = samples[trim_end - target_len : trim_end]
        cut_start = trim_end - target_len
        cut_end = trim_end
    return utterance, cut, trim_start, trim_end, cut_start, cut_end, int(samples.size)


def _duration_target(rng: random.Random, label: str) -> float:
    if label != "stress":
        return float(label)
    low, high = STRESS_DURATION_RANGE
    return round(rng.uniform(low, high) * 100.0) / 100.0


def _valid_overlaps(duration_label: str) -> list[int]:
    if duration_label == "stress":
        min_duration_ms = int(STRESS_DURATION_RANGE[0] * 1000)
    else:
        min_duration_ms = int(float(duration_label) * 1000)
    return [overlap_ms for overlap_ms in OVERLAPS_MS if overlap_ms < min_duration_ms]


def _region(regions: list[SpeakerRegion], start: int, end: int, speakers: set[str]) -> None:
    if end <= start:
        return
    regions.append(
        SpeakerRegion(
            audio_epoch=0,
            start_sample=start,
            end_sample=end,
            speakers=frozenset(speakers),
        )
    )


def _noise_signal(case_seed: int, duration_s: float, kind: str) -> np.ndarray:
    rng = np.random.default_rng(case_seed)
    n = int(duration_s * CANONICAL_SAMPLE_RATE_HZ)
    white = rng.standard_normal(n).astype(np.float32)
    if kind == "white_soft":
        return white * 0.02
    if kind == "white_moderate":
        return white * 0.05
    if kind == "lowpass":
        low = _bandlimit_numpy(white, BANDLIMIT_HZ)
        return low * 0.05
    raise ValueError(f"unknown noise kind {kind}")


def _bandlimit_numpy(samples: np.ndarray, cutoff_hz: float) -> np.ndarray:
    alpha = 1.0 - np.exp(-2.0 * np.pi * cutoff_hz / CANONICAL_SAMPLE_RATE_HZ)
    out = np.empty_like(samples)
    state = 0.0
    for i in range(int(samples.size)):
        state += alpha * (float(samples[i]) - state)
        out[i] = state
    return out


def _apply_gain(samples: np.ndarray, factor: float) -> np.ndarray:
    return np.clip(samples.astype(np.float32) * factor, -1.0, 1.0)


def _apply_opus(samples: np.ndarray) -> np.ndarray:
    decoded = external.encode_opus_to_pcm16(
        samples,
        bitrate_kbps=OPUS_BITRATE_KBPS,
        sample_rate_hz=CANONICAL_SAMPLE_RATE_HZ,
    )
    n = int(samples.size)
    if decoded.size < n:
        decoded = np.pad(decoded, (0, n - decoded.size))
    return decoded[:n]


def _apply_noise(samples: np.ndarray, case_seed: int, snr_db: float) -> np.ndarray:
    rng = np.random.default_rng(case_seed)
    signal_rms = float(np.sqrt(np.mean(np.square(samples.astype(np.float64)))))
    noise_rms = signal_rms / (10.0 ** (snr_db / 20.0))
    noise = rng.standard_normal(int(samples.size)).astype(np.float32) * noise_rms
    return np.clip(samples.astype(np.float32) + noise, -1.0, 1.0)


def _apply_transforms(
    samples: np.ndarray, transforms: list[TransformSpec], case_seed: int
) -> np.ndarray:
    for transform in transforms:
        if transform.name == "opus":
            samples = _apply_opus(samples)
        elif transform.name == "gain":
            samples = _apply_gain(samples, float(transform.params["factor"]))
        elif transform.name == "noise":
            samples = _apply_noise(
                samples, case_seed, float(transform.params.get("snr_db", NOISE_SNR_DB))
            )
        elif transform.name == "bandlimit":
            samples = _bandlimit_numpy(samples, float(transform.params.get("hz", BANDLIMIT_HZ)))
        else:
            raise ValueError(f"unknown transform {transform.name}")
    return samples


def _zero_gap_evidence(samples: np.ndarray, junction: int) -> ZeroGapEvidence:
    pre = samples[max(0, junction - 640) : junction]
    post = samples[junction : junction + 640]
    pre_rms = float(np.sqrt(np.mean(np.square(pre.astype(np.float64))))) if pre.size else 0.0
    post_rms = float(np.sqrt(np.mean(np.square(post.astype(np.float64))))) if post.size else 0.0
    peak = float(np.max(np.abs(samples[max(0, junction - 5) : min(samples.size, junction + 5)])))
    return ZeroGapEvidence(
        b_onset_is_a_end=True,
        pre_junction_rms=pre_rms,
        post_junction_rms=post_rms,
        junction_peak_abs=peak,
    )


def _source_ref(
    index: SplitIndex,
    role: str,
    speaker: str,
    utterance: UtteranceInfo,
    total_samples: int,
    trimmed_start: int,
    trimmed_end: int,
    cut_start: int,
    cut_end: int,
    gain: float,
) -> SourceRef:
    return SourceRef(
        role=role,
        speaker=speaker,
        session=utterance.session,
        utterance=utterance.utterance_id,
        file_sha256=_utterance_sha256(utterance),
        original_start_sample=0,
        original_end_sample=total_samples,
        trimmed_start_sample=trimmed_start,
        trimmed_end_sample=trimmed_end,
        cut_start_sample=cut_start,
        cut_end_sample=cut_end,
        gain=gain,
    )


def build_splice(
    a: np.ndarray,
    b: np.ndarray,
    *,
    gap_ms: int | None,
    overlap_ms: int | None,
    lead_samples: int,
    tail_samples: int,
    speaker_a: str = "A",
    speaker_b: str = "B",
) -> tuple[np.ndarray, list[SpeakerRegion], SpliceSpec]:
    a_start = lead_samples
    a_end = a_start + int(a.size)
    if gap_ms is not None:
        gap_samples = int(gap_ms * CANONICAL_SAMPLE_RATE_HZ / 1000.0)
        b_onset = a_end + gap_samples
        segments = [np.zeros(lead_samples, dtype=np.float32), a]
        if gap_samples > 0:
            segments.append(np.zeros(gap_samples, dtype=np.float32))
        segments.append(b)
        segments.append(np.zeros(tail_samples, dtype=np.float32))
        audio = np.concatenate(segments)
        b_end = b_onset + int(b.size)
        regions: list[SpeakerRegion] = []
        _region(regions, 0, a_start, set())
        _region(regions, a_start, a_end, {speaker_a})
        _region(regions, a_end, b_onset, set())
        _region(regions, b_onset, b_end, {speaker_b})
        _region(regions, b_end, int(audio.size), set())
        return (
            audio,
            regions,
            SpliceSpec(
                a_end_sample=a_end,
                b_onset_sample=b_onset,
                gap_samples=gap_samples,
                overlap_samples=None,
            ),
        )
    if overlap_ms is not None:
        overlap_samples = int(overlap_ms * CANONICAL_SAMPLE_RATE_HZ / 1000.0)
        if overlap_samples >= int(a.size) or overlap_samples >= int(b.size):
            raise CorpusError(f"overlap {overlap_ms} ms must be smaller than both active durations")
        b_onset = a_end - overlap_samples
        b_end = b_onset + int(b.size)
        mixed = np.zeros(a_end + int(b.size) - overlap_samples, dtype=np.float32)
        mixed[a_start:a_end] = a
        mixed[b_onset:b_end] = b
        mix_start = max(a_start, b_onset)
        mix_end = min(a_end, b_end)
        if mix_end > mix_start:
            mixed[mix_start:mix_end] = (
                a[mix_start - a_start : mix_end - a_start]
                + b[mix_start - b_onset : mix_end - b_onset]
            ) / 2.0
        audio = np.concatenate([mixed, np.zeros(tail_samples, dtype=np.float32)])
        regions = []
        _region(regions, 0, a_start, set())
        _region(regions, a_start, b_onset, {speaker_a})
        _region(regions, b_onset, a_end, {speaker_a, speaker_b})
        _region(regions, a_end, b_end, {speaker_b})
        _region(regions, b_end, int(audio.size), set())
        return (
            audio,
            regions,
            SpliceSpec(
                a_end_sample=a_end,
                b_onset_sample=b_onset,
                gap_samples=None,
                overlap_samples=overlap_samples,
            ),
        )
    raise ValueError("one of gap_ms or overlap_ms must be set")


def _junction_window_rms(samples: np.ndarray, junction: int) -> tuple[float, float]:
    pre = samples[max(0, junction - JUNCTION_WINDOW_SAMPLES) : junction]
    post = samples[junction : junction + JUNCTION_WINDOW_SAMPLES]
    pre_rms = float(np.sqrt(np.mean(np.square(pre.astype(np.float64))))) if pre.size else 0.0
    post_rms = float(np.sqrt(np.mean(np.square(post.astype(np.float64))))) if post.size else 0.0
    return pre_rms, post_rms


def _window_rms(samples: np.ndarray, start: int, end: int) -> float:
    window = samples[start:end]
    if window.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(window.astype(np.float64)))))


def _case_plan(
    split: str,
    index: SplitIndex,
    *,
    rng: random.Random,
    case_seed: int,
    case_id: str,
    kind: str,
    duration_target: float,
    gap_ms: int | None,
    overlap_ms: int | None,
    b_gain: float,
    same_speaker: bool,
    transforms: list[TransformSpec],
) -> dict[str, Any]:
    target_len = int(round(duration_target * CANONICAL_SAMPLE_RATE_HZ))
    speaker_a, speaker_b = _pick_speakers(rng, index.speakers, same=same_speaker)
    (
        utterance_a,
        cut_a,
        trimmed_a_start,
        trimmed_a_end,
        cut_a_start,
        cut_a_end,
        total_a,
    ) = _load_role_source(index, "A", speaker_a, target_len, rng)
    for attempt in range(MAX_JUNCTION_GUARD_ATTEMPTS):
        if (
            attempt == 0
            or gap_ms != 0
            or _window_rms(cut_a, cut_a.size - JUNCTION_WINDOW_SAMPLES, cut_a.size)
            >= JUNCTION_GUARD_RMS
        ):
            break
        (
            utterance_a,
            cut_a,
            trimmed_a_start,
            trimmed_a_end,
            cut_a_start,
            cut_a_end,
            total_a,
        ) = _load_role_source(index, "A", speaker_a, target_len, rng)
    else:
        raise CorpusError(
            f"case {case_id}: no A source with trailing junction window >= {JUNCTION_GUARD_RMS}"
        )
    (
        utterance_b,
        cut_b,
        trimmed_b_start,
        trimmed_b_end,
        cut_b_start,
        cut_b_end,
        total_b,
    ) = _load_role_source(
        index,
        "B",
        speaker_b,
        target_len,
        rng,
        exclude_utterance_id=utterance_a.utterance_id if same_speaker else None,
    )
    for attempt in range(MAX_JUNCTION_GUARD_ATTEMPTS):
        if (
            attempt == 0
            or gap_ms != 0
            or _window_rms(cut_b, 0, JUNCTION_WINDOW_SAMPLES) >= JUNCTION_GUARD_RMS
        ):
            break
        (
            utterance_b,
            cut_b,
            trimmed_b_start,
            trimmed_b_end,
            cut_b_start,
            cut_b_end,
            total_b,
        ) = _load_role_source(
            index,
            "B",
            speaker_b,
            target_len,
            rng,
            exclude_utterance_id=utterance_a.utterance_id if same_speaker else None,
        )
    else:
        raise CorpusError(
            f"case {case_id}: no B source with leading junction window >= {JUNCTION_GUARD_RMS}"
        )
    if b_gain != 1.0:
        cut_b = _apply_gain(cut_b, b_gain)
    audio, regions, splice = build_splice(
        cut_a,
        cut_b,
        gap_ms=gap_ms,
        overlap_ms=overlap_ms,
        lead_samples=int(LEAD_SILENCE_S * CANONICAL_SAMPLE_RATE_HZ),
        tail_samples=int(TAIL_SILENCE_S * CANONICAL_SAMPLE_RATE_HZ),
        speaker_a=speaker_a,
        speaker_b=speaker_b,
    )
    audio = _apply_transforms(audio, transforms, case_seed)
    if gap_ms is not None and gap_ms == 0:
        final_pre, final_post = _junction_window_rms(audio, splice.a_end_sample)
        if final_pre < FINAL_JUNCTION_MIN_RMS or final_post < FINAL_JUNCTION_MIN_RMS:
            raise CorpusError(
                f"case {case_id}: gap=0 final junction degraded by transforms "
                f"(pre_rms={final_pre:.6f} post_rms={final_post:.6f})"
            )
    sources = [
        _source_ref(
            index,
            "A",
            speaker_a,
            utterance_a,
            total_a,
            trimmed_a_start,
            trimmed_a_end,
            cut_a_start,
            cut_a_end,
            1.0,
        ),
        _source_ref(
            index,
            "B",
            speaker_b,
            utterance_b,
            total_b,
            trimmed_b_start,
            trimmed_b_end,
            cut_b_start,
            cut_b_end,
            b_gain,
        ),
    ]
    return {
        "audio": audio,
        "regions": regions,
        "splice": splice,
        "sources": sources,
    }


def _transforms_for(kind: str, rng: random.Random) -> list[TransformSpec]:
    if kind == "stress_opus":
        return [TransformSpec("opus", {"bitrate_kbps": OPUS_BITRATE_KBPS})]
    if kind == "stress_gain":
        return [TransformSpec("gain", {"factor": round(rng.uniform(0.5, 1.5), 3)})]
    if kind == "stress_noise":
        return [TransformSpec("noise", {"snr_db": NOISE_SNR_DB})]
    if kind == "stress_opus_noise":
        return [
            TransformSpec("opus", {"bitrate_kbps": OPUS_BITRATE_KBPS}),
            TransformSpec("noise", {"snr_db": NOISE_SNR_DB}),
        ]
    if kind == "stress_bandlimit":
        return [TransformSpec("bandlimit", {"hz": BANDLIMIT_HZ})]
    return []


def build_librispeech_manifest(
    *,
    split: str,
    manifest_id: str,
    out_dir: Path,
    index: SplitIndex,
    seed: int = 2026,
) -> Phase2Manifest:
    rng = random.Random(seed)
    cases: list[Phase2Case] = []
    case_number = 0

    def add_case(
        case_id: str,
        kind: str,
        condition: dict[str, Any],
        gap_ms: int | None,
        overlap_ms: int | None,
        duration_target: float,
        *,
        b_gain: float = 1.0,
        same_speaker: bool = False,
        transforms: list[TransformSpec] | None = None,
    ) -> None:
        nonlocal case_number
        case_number += 1
        plan = _case_plan(
            split,
            index,
            rng=rng,
            case_seed=seed + case_number,
            case_id=case_id,
            kind=kind,
            duration_target=duration_target,
            gap_ms=gap_ms,
            overlap_ms=overlap_ms,
            b_gain=b_gain,
            same_speaker=same_speaker,
            transforms=transforms or [],
        )
        wav_relative = f"generated/{case_id}.wav"
        wav_path = out_dir / wav_relative
        external.write_pcm16_wav(wav_path, plan["audio"], sample_rate_hz=CANONICAL_SAMPLE_RATE_HZ)
        regions = plan["regions"]
        splice = plan["splice"]
        zero_gap_evidence = None
        if gap_ms is not None and gap_ms == 0 and splice is not None:
            from experiments.speaker_turn_boundary.vad_baseline import load_canonical_wav

            wav_samples = load_canonical_wav(wav_path)
            zero_gap_evidence = _zero_gap_evidence(wav_samples, splice.a_end_sample)
        cases.append(
            Phase2Case(
                case_id=case_id,
                wav_relative_path=wav_relative,
                duration_samples=int(plan["audio"].size),
                wav_sha256=hashlib.sha256(wav_path.read_bytes()).hexdigest(),
                seed=seed + case_number,
                regions=regions,
                kind=kind,
                condition=condition,
                sources=plan["sources"],
                splice=splice,
                transforms=transforms or [],
                zero_gap_evidence=zero_gap_evidence,
                active_speech_samples=active_speech_sample_count(regions),
            )
        )

    positive_kinds = [
        ("different_speaker_gap", "gap", GAPS_MS),
        ("different_speaker_overlap", "overlap", None),
    ]
    for duration_label in DURATION_TARGETS:
        target = _duration_target(rng, duration_label)
        dur_label = DURATION_LABELS[str(duration_label)]
        for kind, transition, values in positive_kinds:
            if transition == "overlap":
                values = _valid_overlaps(duration_label)
            for value in values:
                for idx in range(CASE_COUNTS["positive_per_combo"]):
                    if transition == "gap":
                        condition = {
                            "duration_target_s": target,
                            "transition": "gap",
                            "gap_ms": value,
                            "overlap_ms": None,
                        }
                        add_case(
                            f"{split}_{kind}_{dur_label}_g{value:04d}_{idx:02d}",
                            kind,
                            condition,
                            gap_ms=value,
                            overlap_ms=None,
                            duration_target=target,
                        )
                    else:
                        condition = {
                            "duration_target_s": target,
                            "transition": "overlap",
                            "gap_ms": None,
                            "overlap_ms": value,
                        }
                        add_case(
                            f"{split}_{kind}_{dur_label}_o{value:04d}_{idx:02d}",
                            kind,
                            condition,
                            gap_ms=None,
                            overlap_ms=value,
                            duration_target=target,
                        )
    stress_combos = [
        (2.0, "gap", 0, "stress_opus"),
        (2.0, "gap", 100, "stress_gain"),
        (2.0, "overlap", 300, "stress_noise"),
        (0.5, "gap", 0, "stress_opus_noise"),
        (0.5, "overlap", 300, "stress_opus"),
        (0.5, "gap", 100, "stress_noise"),
        (2.0, "overlap", 100, "stress_gain"),
        (0.5, "gap", 0, "stress_bandlimit"),
    ]
    for duration_target, transition, value, stress_kind in stress_combos:
        dur_label = DURATION_LABELS.get(str(duration_target), f"d{int(duration_target*100):03d}")
        kind = "different_speaker_gap" if transition == "gap" else "different_speaker_overlap"
        transforms = _transforms_for(stress_kind, rng)
        for idx in range(CASE_COUNTS["stress_per_combo"]):
            condition = {
                "duration_target_s": duration_target,
                "transition": transition,
                "gap_ms": value if transition == "gap" else None,
                "overlap_ms": value if transition == "overlap" else None,
                "stress": stress_kind,
            }
            add_case(
                f"{split}_{kind}_{dur_label}_{stress_kind}_{idx:02d}",
                kind,
                condition,
                gap_ms=value if transition == "gap" else None,
                overlap_ms=value if transition == "overlap" else None,
                duration_target=duration_target,
                transforms=transforms,
            )
    for duration_label in [2.0, 1.0, 0.5]:
        dur_label = DURATION_LABELS[str(duration_label)]
        for gap_ms in GAPS_MS:
            for idx in range(CASE_COUNTS["same_speaker_per_combo"]):
                condition = {
                    "duration_target_s": duration_label,
                    "transition": "gap",
                    "gap_ms": gap_ms,
                    "overlap_ms": None,
                }
                add_case(
                    f"{split}_same_speaker_{dur_label}_g{gap_ms:04d}_{idx:02d}",
                    "same_speaker",
                    condition,
                    gap_ms=gap_ms,
                    overlap_ms=None,
                    duration_target=float(duration_label),
                    same_speaker=True,
                )
    for duration_label in [2.0, 1.0, 0.5]:
        dur_label = DURATION_LABELS[str(duration_label)]
        for gap_ms in [300, 0]:
            for idx in range(CASE_COUNTS["gain_per_combo"]):
                gain = round(rng.uniform(0.5, 1.5), 3)
                condition = {
                    "duration_target_s": float(duration_label),
                    "transition": "gap",
                    "gap_ms": gap_ms,
                    "overlap_ms": None,
                    "b_gain": gain,
                }
                add_case(
                    f"{split}_gain_variation_{dur_label}_g{gap_ms:04d}_{idx:02d}",
                    "gain_variation",
                    condition,
                    gap_ms=gap_ms,
                    overlap_ms=None,
                    duration_target=float(duration_label),
                    b_gain=gain,
                    same_speaker=True,
                )
    silence_durations = [0.5, 2.0, 5.0, 10.0]
    for idx, duration_s in enumerate(silence_durations):
        case_seed = seed + case_number + idx
        rng_np = np.random.default_rng(case_seed)
        n = int(duration_s * CANONICAL_SAMPLE_RATE_HZ)
        audio = rng_np.uniform(-1e-4, 1e-4, n).astype(np.float32)
        regions = [SpeakerRegion(0, 0, n, frozenset())]
        wav_relative = f"generated/{split}_silence_{idx:02d}.wav"
        wav_path = out_dir / wav_relative
        external.write_pcm16_wav(wav_path, audio, sample_rate_hz=CANONICAL_SAMPLE_RATE_HZ)
        cases.append(
            Phase2Case(
                case_id=f"{split}_silence_{idx:02d}",
                wav_relative_path=wav_relative,
                duration_samples=n,
                wav_sha256=hashlib.sha256(wav_path.read_bytes()).hexdigest(),
                seed=case_seed,
                regions=regions,
                kind="silence",
                condition={"duration_s": duration_s},
                active_speech_samples=0,
            )
        )
    noise_kinds = ["white_soft", "white_moderate", "lowpass", "white_moderate"]
    noise_durations = [2.0, 2.0, 2.0, 4.0]
    for idx, (noise_kind, duration_s) in enumerate(zip(noise_kinds, noise_durations)):
        case_seed = seed + 1000 + idx
        audio = _noise_signal(case_seed, duration_s, noise_kind)
        n = int(audio.size)
        regions = [SpeakerRegion(0, 0, n, frozenset())]
        wav_relative = f"generated/{split}_noise_only_{idx:02d}.wav"
        wav_path = out_dir / wav_relative
        external.write_pcm16_wav(wav_path, audio, sample_rate_hz=CANONICAL_SAMPLE_RATE_HZ)
        cases.append(
            Phase2Case(
                case_id=f"{split}_noise_only_{idx:02d}",
                wav_relative_path=wav_relative,
                duration_samples=n,
                wav_sha256=hashlib.sha256(wav_path.read_bytes()).hexdigest(),
                seed=case_seed,
                regions=regions,
                kind="noise_only",
                condition={"noise_kind": noise_kind, "duration_s": duration_s},
                active_speech_samples=0,
            )
        )
    manifest = make_phase2_manifest(
        manifest_id=manifest_id,
        split_role="dev" if split == "dev-clean" else "held_out",
        corpus={
            "name": "librispeech",
            "version": "1",
            "license": LIBRISPEECH_LICENSE,
            "source": LIBRISPEECH_SOURCE,
            "archives": {
                key: {"url": spec["url"], "md5": spec["md5"], "size_bytes": _archive_size(key)}
                for key, spec in ARCHIVES.items()
                if key == split
            },
            "split": split,
        },
        build={
            "script": "corpus.librispeech.build_librispeech_manifest",
            "seed": seed,
            "trim_method": "energy_rms_frame10ms_rel0.01_floor1e-3",
            "cut_method": "A_trailing_window_B_leading_window_after_trim",
            "lead_tail_silence_s": [LEAD_SILENCE_S, TAIL_SILENCE_S],
            "opus_bitrate_kbps": OPUS_BITRATE_KBPS,
            "noise_snr_db": NOISE_SNR_DB,
            "config_hash": hashlib.sha256(
                canonical_config(split, seed).encode("utf-8")
            ).hexdigest(),
        },
        disjointness_groups=[f"librispeech_{split}"],
        generator={"script": "build_phase2_cases.py", "seed": seed},
        cases=cases,
    )
    write_manifest(manifest, out_dir / "manifests" / f"{manifest_id}.json")
    return manifest


def _archive_size(split: str) -> int | None:
    archive = external.archive_root() / f"{split}.tar.gz"
    if archive.is_file():
        return archive.stat().st_size
    return None


def canonical_config(split: str, seed: int) -> str:
    from experiments.speaker_turn_boundary.corpus.phase2_schemas import canonical_json

    return canonical_json(
        {
            "split": split,
            "seed": seed,
            "trim": [TRIM_FRAME_MS, TRIM_RELATIVE_RMS_THRESHOLD, TRIM_ABS_RMS_FLOOR],
            "durations": DURATION_TARGETS,
            "stress_range": list(STRESS_DURATION_RANGE),
            "gaps_ms": GAPS_MS,
            "overlaps_ms": OVERLAPS_MS,
            "counts": CASE_COUNTS,
            "opus_bitrate_kbps": OPUS_BITRATE_KBPS,
            "noise_snr_db": NOISE_SNR_DB,
            "bandlimit_hz": BANDLIMIT_HZ,
        }
    )


def write_manifest(manifest: Phase2Manifest, path: Path) -> str:
    return manifest.write(path)
