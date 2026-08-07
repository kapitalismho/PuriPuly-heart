from __future__ import annotations

import re
import wave as wave_module
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from experiments.speaker_turn_boundary.config import CANONICAL_SAMPLE_RATE_HZ
from experiments.speaker_turn_boundary.corpus import external
from experiments.speaker_turn_boundary.corpus.external import (
    CorpusError,
    extract_tar_gz,
)
from experiments.speaker_turn_boundary.corpus.phase2_schemas import (
    Phase2Case,
    Phase2Manifest,
    SourceRef,
    make_phase2_manifest,
)
from experiments.speaker_turn_boundary.ground_truth import SpeakerRegion, active_speech_sample_count

ALIMEETING_EVAL_URL = (
    "https://speech-lab-share-data.oss-cn-shanghai.aliyuncs.com/AliMeeting/openlr/Eval_Ali.tar.gz"
)
ALIMEETING_LICENSE = "CC BY-SA 4.0"
ALIMEETING_SOURCE = "https://www.openslr.org/119/"
ALIMEETING_EVAL_SIZE_BYTES = 3673718355
ALIMEETING_RECIPE = "far_field_array_channel_0_16k_mono"

_MEETING_KEY_PATTERN = re.compile(r"^(?P<key>R\d+_M\d+)")
_SPEAKER_TIER_PATTERN = re.compile(r"^N_(?P<speaker>SPK\d+)$")


@dataclass(frozen=True, slots=True)
class TextGridInterval:
    speaker: str
    start_time_s: float
    end_time_s: float
    text: str


@dataclass(slots=True)
class AlimeetingSession:
    session_id: str
    far_wav_path: Path
    textgrid_path: Path
    intervals: list[TextGridInterval]
    speakers: list[str]


def acquire_alimeeting_eval(root: Path | None = None) -> Path:
    root = root or external.corpus_root()
    archive = external.download_file(
        ALIMEETING_EVAL_URL,
        root / "archives" / "alimeeting_eval.tar.gz",
        timeout_seconds=120,
    )
    target = root / "alimeeting"
    if not (target / "Eval_Ali").is_dir():
        extract_tar_gz(archive, target)
    return target


def parse_textgrid(path: Path) -> list[tuple[str, list[tuple[float, float, str]]]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    tiers: list[tuple[str, list[tuple[float, float, str]]]] = []
    current_name: str | None = None
    current_class: str | None = None
    current_intervals: list[tuple[float, float, str]] = []
    interval_start: float | None = None
    interval_end: float | None = None

    def flush_tier() -> None:
        nonlocal current_name, current_class, current_intervals
        if current_name is not None and current_class == "IntervalTier":
            tiers.append((current_name, list(current_intervals)))
        current_name = None
        current_class = None
        current_intervals = []

    for raw_line in lines:
        line = raw_line.strip()
        if line.startswith("item ["):
            flush_tier()
            continue
        if line.startswith("intervals"):
            continue
        if line.startswith("class"):
            flush_tier()
            current_class = _unquote(line.split("=", 1)[1].strip())
            continue
        if line.startswith("name"):
            current_name = _unquote(line.split("=", 1)[1].strip())
            continue
        if line.startswith("xmin"):
            interval_start = float(line.split("=", 1)[1].strip())
            interval_end = None
            continue
        if line.startswith("xmax"):
            interval_end = float(line.split("=", 1)[1].strip())
            continue
        if line.startswith("text") and current_name is not None:
            text = _unquote(line.split("=", 1)[1].strip())
            if interval_start is not None and interval_end is not None:
                current_intervals.append((interval_start, interval_end, text))
            interval_start = None
            interval_end = None
            continue
    flush_tier()
    return tiers


def _unquote(value: str) -> str:
    if len(value) >= 2 and value.startswith('"') and value.endswith('"'):
        return value[1:-1]
    return value


def load_wav_channel_0(path: Path) -> np.ndarray:
    with wave_module.open(str(path), "rb") as handle:
        if handle.getframerate() != CANONICAL_SAMPLE_RATE_HZ:
            raise CorpusError(f"{path.name}: expected 16 kHz, got {handle.getframerate()} Hz")
        channels = handle.getnchannels()
        frames = handle.readframes(handle.getnframes())
    pcm = np.frombuffer(frames, dtype=np.int16)
    if channels > 1:
        pcm = pcm.reshape(-1, channels)[:, 0]
    return pcm.astype(np.float32) / 32768.0


def intervals_to_regions(
    intervals: list[TextGridInterval],
    duration_samples: int,
    sample_rate_hz: int = CANONICAL_SAMPLE_RATE_HZ,
) -> tuple[list[SpeakerRegion], dict[str, int]]:
    boundaries: list[int] = []
    spans: list[tuple[int, int, str]] = []
    skipped = 0
    for interval in intervals:
        start_sample = int(round(interval.start_time_s * sample_rate_hz))
        end_sample = int(round(interval.end_time_s * sample_rate_hz))
        if end_sample <= start_sample:
            skipped += 1
            continue
        if start_sample < 0 or end_sample > duration_samples:
            end_sample = min(end_sample, duration_samples)
        spans.append((start_sample, end_sample, interval.speaker))
        boundaries.append(start_sample)
        boundaries.append(end_sample)
    boundaries = sorted(set(boundaries))
    if not boundaries:
        return [SpeakerRegion(0, 0, duration_samples, frozenset())], {
            "skipped_zero_length_intervals": skipped
        }
    intervals_out: list[tuple[int, int, frozenset[str]]] = []
    for index in range(len(boundaries) - 1):
        start = boundaries[index]
        end = boundaries[index + 1]
        if end <= start:
            continue
        speakers: set[str] = set()
        for span_start, span_end, speaker in spans:
            if span_end <= start or span_start >= end:
                continue
            speakers.add(speaker)
        intervals_out.append((start, end, frozenset(speakers)))
    regions: list[SpeakerRegion] = []
    for start, end, speakers in intervals_out:
        if regions and regions[-1].speakers == speakers and not regions[-1].ambiguous:
            regions[-1] = SpeakerRegion(
                audio_epoch=0,
                start_sample=regions[-1].start_sample,
                end_sample=end,
                speakers=speakers,
            )
            continue
        regions.append(
            SpeakerRegion(
                audio_epoch=0,
                start_sample=start,
                end_sample=end,
                speakers=speakers,
            )
        )
    if boundaries[0] > 0:
        regions.insert(0, SpeakerRegion(0, 0, boundaries[0], frozenset()))
    if boundaries[-1] < duration_samples:
        regions.append(SpeakerRegion(0, boundaries[-1], duration_samples, frozenset()))
    return regions, {"skipped_zero_length_intervals": skipped}


def index_alimeeting_eval(root: Path) -> list[AlimeetingSession]:
    far_dir = root / "alimeeting" / "Eval_Ali" / "Eval_Ali_far"
    audio_dir = far_dir / "audio_dir"
    textgrid_dir = far_dir / "textgrid_dir"
    if not audio_dir.is_dir() or not textgrid_dir.is_dir():
        raise CorpusError(f"AliMeeting far layout not found under {far_dir}")
    sessions: list[AlimeetingSession] = []
    for wav_path in sorted(audio_dir.glob("R*_M*_MS*.wav")):
        match = _MEETING_KEY_PATTERN.match(wav_path.name)
        if not match:
            raise CorpusError(f"unexpected AliMeeting far wav name {wav_path.name}")
        session_id = match.group("key")
        textgrid_path = textgrid_dir / f"{session_id}.TextGrid"
        if not textgrid_path.is_file():
            raise CorpusError(f"missing TextGrid for {session_id}")
        intervals: list[TextGridInterval] = []
        speakers: set[str] = set()
        for tier_name, tier_intervals in parse_textgrid(textgrid_path):
            tier_match = _SPEAKER_TIER_PATTERN.match(tier_name)
            speaker = tier_match.group("speaker") if tier_match else tier_name
            speakers.add(speaker)
            for start, end, text in tier_intervals:
                if not text.strip():
                    continue
                intervals.append(
                    TextGridInterval(
                        speaker=speaker,
                        start_time_s=start,
                        end_time_s=end,
                        text=text,
                    )
                )
        intervals.sort(key=lambda i: (i.start_time_s, i.end_time_s))
        sessions.append(
            AlimeetingSession(
                session_id=session_id,
                far_wav_path=wav_path,
                textgrid_path=textgrid_path,
                intervals=intervals,
                speakers=sorted(speakers),
            )
        )
    sessions.sort(key=lambda s: s.session_id)
    return sessions


def materialize_channel_0_wav(session: AlimeetingSession, root: Path) -> tuple[Path, int, str]:
    samples = load_wav_channel_0(session.far_wav_path)
    destination = root / "alimeeting" / "far_ch0" / f"{session.session_id}.wav"
    external.write_pcm16_wav(destination, samples, sample_rate_hz=CANONICAL_SAMPLE_RATE_HZ)
    import hashlib

    return destination, int(samples.size), hashlib.sha256(destination.read_bytes()).hexdigest()


def build_alimeeting_manifest(
    *,
    manifest_id: str,
    split_role: str,
    root: Path,
    out_dir: Path,
    session_ids: list[str] | None = None,
) -> Phase2Manifest:
    sessions = index_alimeeting_eval(root)
    selected = [s for s in sessions if session_ids is None or s.session_id in session_ids]
    if not selected:
        raise CorpusError("no AliMeeting sessions found; download and extract first")
    cases: list[Phase2Case] = []
    for session in selected:
        ch0_wav_path, duration_samples, wav_sha256 = materialize_channel_0_wav(session, root)
        regions, stats = intervals_to_regions(session.intervals, duration_samples)
        relative_path = str(ch0_wav_path.relative_to(root)).replace("\\", "/")
        condition: dict[str, object] = {
            "corpus": "alimeeting",
            "session_id": session.session_id,
            "split_role": split_role,
            "recording_condition": ALIMEETING_RECIPE,
            "far_wav": session.far_wav_path.name,
            "textgrid": session.textgrid_path.name,
            "materialized_ch0_wav": ch0_wav_path.name,
            "interval_tiers": session.speakers,
            "intervals": len(session.intervals),
            "parse_stats": stats,
        }
        sources = [
            SourceRef(
                role=f"speaker_{index}",
                speaker=speaker,
                session=session.session_id,
                utterance=session.session_id,
                file_sha256=wav_sha256,
                original_start_sample=0,
                original_end_sample=duration_samples,
                trimmed_start_sample=0,
                trimmed_end_sample=duration_samples,
                cut_start_sample=0,
                cut_end_sample=duration_samples,
                gain=1.0,
            )
            for index, speaker in enumerate(session.speakers)
        ]
        cases.append(
            Phase2Case(
                case_id=f"alimeeting_{session.session_id}",
                wav_relative_path=relative_path,
                duration_samples=duration_samples,
                wav_sha256=wav_sha256,
                seed=0,
                regions=regions,
                kind="real_meeting",
                condition=condition,
                sources=sources,
                active_speech_samples=active_speech_sample_count(regions),
            )
        )
    manifest = make_phase2_manifest(
        manifest_id=manifest_id,
        split_role=split_role,
        corpus={
            "name": "alimeeting",
            "version": "M2MeT eval",
            "license": ALIMEETING_LICENSE,
            "source": ALIMEETING_SOURCE,
            "eval_url": ALIMEETING_EVAL_URL,
            "eval_size_bytes": ALIMEETING_EVAL_SIZE_BYTES,
            "local_wav_root": str(root),
            "single_channel_recipe": ALIMEETING_RECIPE,
        },
        build={
            "script": "corpus.alimeeting.build_alimeeting_manifest",
            "annotation": "TextGrid IntervalTiers per participant (N_SPKxxxx)",
            "overlap_rule": "simultaneous speech intervals across tiers produce multi-speaker regions",
            "session_key_rule": "session = Rxxxx_Mxxxx prefix of far wav and TextGrid names",
            "materialization": "channel 0 of the 8-channel far wav written as canonical 16 kHz mono PCM16",
        },
        disjointness_groups=[f"alimeeting_{split_role}"],
        generator={"script": "build_phase2_real.py"},
        cases=cases,
    )
    manifest_path = out_dir / "manifests" / f"{manifest_id}.json"
    manifest.write(manifest_path)
    return manifest
