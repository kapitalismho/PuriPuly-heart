from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from experiments.speaker_turn_boundary.config import CANONICAL_SAMPLE_RATE_HZ
from experiments.speaker_turn_boundary.corpus import external
from experiments.speaker_turn_boundary.corpus.external import (
    CorpusError,
    extract_zip,
    sha256_file,
)
from experiments.speaker_turn_boundary.corpus.phase2_schemas import (
    Phase2Case,
    Phase2Manifest,
    SourceRef,
    make_phase2_manifest,
)
from experiments.speaker_turn_boundary.ground_truth import SpeakerRegion, active_speech_sample_count

AMI_BASE = "https://groups.inf.ed.ac.uk/ami"
AMI_ESTIMATE_URL = "https://homepages.inf.ed.ac.uk/cgi/simonk/AMI/estimate.cgi"
AMI_ANNOTATIONS_URL = (
    "https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip"
)
AMI_ANNOTATIONS_VERSION = "1.6.2"
AMI_LICENSE = "CC BY 4.0"
AMI_MIX_STREAM = "Mix-Headset"

AMI_SB_DEV_SESSIONS = ["ES2003", "ES2011", "IS1008", "TS3004", "TS3006"]
AMI_SC_HELD_OUT_SESSIONS = ["ES2004", "ES2014", "IS1009", "TS3003", "TS3007"]

AMI_DEV_PILOT_SESSIONS = ["ES2003a", "IS1008a"]
AMI_HELD_OUT_PILOT_SESSIONS = ["ES2004a", "IS1009a"]

_WORDS_FILE_PATTERN = re.compile(
    r"^(?P<meeting>[A-Z]{2}\d{4}[a-d])\.(?P<speaker>[A-Z])\.words\.xml$"
)


@dataclass(frozen=True, slots=True)
class AmiWord:
    speaker: str
    start_time_s: float
    end_time_s: float
    text: str
    ambiguous: bool


@dataclass(frozen=True, slots=True)
class AmiMeeting:
    meeting_id: str
    split: str
    wav_path: Path
    wav_sha256: str
    duration_samples: int
    words: list[AmiWord]
    participants: list[str]
    partition_meta: dict[str, str]


def ami_mirror_url(meeting_id: str, stream: str = AMI_MIX_STREAM) -> str:
    return f"{AMI_BASE}/AMICorpusMirror//amicorpus/{meeting_id}/audio/{meeting_id}.{stream}.wav"


def resolve_ami_download_urls(
    meeting_ids: list[str], stream: str = AMI_MIX_STREAM
) -> dict[str, str]:
    return {meeting_id: ami_mirror_url(meeting_id, stream) for meeting_id in meeting_ids}


def acquire_ami_annotations(root: Path | None = None) -> Path:
    root = root or external.corpus_root()
    archive = external.download_file(
        AMI_ANNOTATIONS_URL,
        root / "archives" / "ami_public_manual_1.6.2.zip",
        timeout_seconds=120,
    )
    target = root / "ami" / "annotations"
    if not (target / "words").is_dir():
        extract_zip(archive, target)
    return target


def acquire_ami_meetings(
    meetings: list[str],
    root: Path | None = None,
    *,
    stream: str = AMI_MIX_STREAM,
) -> dict[str, Path]:
    root = root or external.corpus_root()
    audio_dir = root / "ami" / "audio"
    urls = resolve_ami_download_urls(meetings, stream)
    paths: dict[str, Path] = {}
    for meeting_id, url in urls.items():
        destination = audio_dir / meeting_id / f"{meeting_id}.{stream}.wav"
        external.download_file(url, destination, timeout_seconds=120)
        paths[meeting_id] = destination
    return paths


def _load_words_xml(words_path: Path) -> list[AmiWord]:
    tree = ET.parse(str(words_path))
    root_elem = tree.getroot()
    words: list[AmiWord] = []
    for element in root_elem.iter():
        if element.tag.lower() != "w":
            continue
        start = element.get("starttime")
        end = element.get("endtime")
        if start is None or end is None:
            continue
        text = "".join(element.itertext()).strip()
        words.append(
            AmiWord(
                speaker="",
                start_time_s=float(start),
                end_time_s=float(end),
                text=text,
                ambiguous="%" in text,
            )
        )
    words.sort(key=lambda w: (w.start_time_s, w.end_time_s))
    return words


def _speaker_from_filename(words_path: Path, meeting_id: str) -> str:
    match = _WORDS_FILE_PATTERN.match(words_path.name)
    if not match or match.group("meeting") != meeting_id:
        raise CorpusError(f"unexpected AMI words filename {words_path.name}")
    return f"{meeting_id}.Participant{match.group('speaker')}"


def _load_meetings_xml(annotations_dir: Path) -> dict[str, dict[str, str]]:
    meetings_path = annotations_dir / "corpusResources" / "meetings.xml"
    if not meetings_path.is_file():
        candidates = sorted(annotations_dir.rglob("meetings.xml"))
        if not candidates:
            return {}
        meetings_path = candidates[0]
    tree = ET.parse(str(meetings_path))
    meetings: dict[str, dict[str, str]] = {}
    for element in tree.getroot().iter():
        if element.tag != "meeting":
            continue
        observation = element.get("observation")
        if not observation:
            continue
        agents: dict[str, str] = {}
        for speaker_element in element.iter():
            if speaker_element.tag != "speaker":
                continue
            agent = speaker_element.get("nxt_agent")
            if agent:
                agents[agent] = speaker_element.get("global_name", "")
        meetings[observation] = {
            "visibility": element.get("visibility", ""),
            "seen_type": element.get("seen_type", ""),
            "k10": element.get("k10", ""),
            "k5": element.get("k5", ""),
            "duration_s": element.get("duration", ""),
            "agents": agents,
        }
    return meetings


def words_to_regions(
    words: list[AmiWord],
    duration_samples: int,
    sample_rate_hz: int = CANONICAL_SAMPLE_RATE_HZ,
) -> tuple[list[SpeakerRegion], dict[str, int]]:
    boundaries: list[int] = []
    word_spans: list[tuple[int, int, str, bool]] = []
    skipped = 0
    for word in words:
        start_sample = int(round(word.start_time_s * sample_rate_hz))
        end_sample = int(round(word.end_time_s * sample_rate_hz))
        if end_sample <= start_sample:
            skipped += 1
            continue
        if start_sample < 0 or end_sample > duration_samples:
            end_sample = min(end_sample, duration_samples)
        word_spans.append((start_sample, end_sample, word.speaker, word.ambiguous))
        boundaries.append(start_sample)
        boundaries.append(end_sample)
    boundaries = sorted(set(boundaries))
    if not boundaries:
        return [SpeakerRegion(0, 0, duration_samples, frozenset())], {
            "skipped_zero_length_words": skipped
        }
    intervals: list[tuple[int, int, frozenset[str], bool]] = []
    for index in range(len(boundaries) - 1):
        start = boundaries[index]
        end = boundaries[index + 1]
        if end <= start:
            continue
        speakers: set[str] = set()
        ambiguous = False
        for span_start, span_end, speaker, span_ambiguous in word_spans:
            if span_end <= start or span_start >= end:
                continue
            speakers.add(speaker)
            if span_ambiguous:
                ambiguous = True
        intervals.append((start, end, frozenset(speakers), ambiguous))
    regions: list[SpeakerRegion] = []
    for start, end, speakers, ambiguous in intervals:
        if regions and regions[-1].speakers == speakers and regions[-1].ambiguous == ambiguous:
            regions[-1] = SpeakerRegion(
                audio_epoch=0,
                start_sample=regions[-1].start_sample,
                end_sample=end,
                speakers=speakers,
                ambiguous=ambiguous,
            )
            continue
        regions.append(
            SpeakerRegion(
                audio_epoch=0,
                start_sample=start,
                end_sample=end,
                speakers=speakers,
                ambiguous=ambiguous,
            )
        )
    if boundaries[0] > 0:
        regions.insert(0, SpeakerRegion(0, 0, boundaries[0], frozenset()))
    if boundaries[-1] < duration_samples:
        regions.append(SpeakerRegion(0, boundaries[-1], duration_samples, frozenset()))
    return regions, {"skipped_zero_length_words": skipped}


def load_ami_meeting(
    meeting_id: str,
    split: str,
    wav_path: Path,
    annotations_dir: Path,
) -> AmiMeeting:
    if not wav_path.is_file():
        raise CorpusError(f"AMI wav missing for {meeting_id}: {wav_path}")
    samples = _load_wav_verify_16k_mono(wav_path)
    words_dir = annotations_dir / "words"
    if not words_dir.is_dir():
        raise CorpusError(f"AMI words annotations missing for {meeting_id}")
    words: list[AmiWord] = []
    for words_path in sorted(words_dir.glob(f"{meeting_id}.*.words.xml")):
        speaker = _speaker_from_filename(words_path, meeting_id)
        for word in _load_words_xml(words_path):
            words.append(
                AmiWord(
                    speaker=speaker,
                    start_time_s=word.start_time_s,
                    end_time_s=word.end_time_s,
                    text=word.text,
                    ambiguous=word.ambiguous,
                )
            )
    if not words:
        raise CorpusError(f"no words found for {meeting_id}")
    words.sort(key=lambda w: (w.start_time_s, w.end_time_s))
    participants = sorted({word.speaker for word in words})
    meetings_meta = _load_meetings_xml(annotations_dir)
    return AmiMeeting(
        meeting_id=meeting_id,
        split=split,
        wav_path=wav_path,
        wav_sha256=sha256_file(wav_path),
        duration_samples=int(samples.size),
        words=words,
        participants=participants,
        partition_meta=meetings_meta.get(meeting_id, {}),
    )


def _load_wav_verify_16k_mono(path: Path) -> np.ndarray:
    import wave as wave_module

    with wave_module.open(str(path), "rb") as handle:
        if handle.getnchannels() != 1 or handle.getframerate() != CANONICAL_SAMPLE_RATE_HZ:
            raise CorpusError(
                f"{path.name}: expected 16 kHz mono, got {handle.getframerate()} Hz "
                f"{handle.getnchannels()} channels"
            )
        frames = handle.readframes(handle.getnframes())
    pcm = np.frombuffer(frames, dtype=np.int16)
    return pcm.astype(np.float32) / 32768.0


def build_ami_manifest(
    *,
    meetings: list[str],
    split_role: str,
    manifest_id: str,
    root: Path,
    out_dir: Path,
) -> Phase2Manifest:
    annotations_dir = root / "ami" / "annotations"
    cases: list[Phase2Case] = []
    for meeting_id in meetings:
        wav_path = root / "ami" / "audio" / meeting_id / f"{meeting_id}.{AMI_MIX_STREAM}.wav"
        meeting = load_ami_meeting(meeting_id, split_role, wav_path, annotations_dir)
        regions, stats = words_to_regions(meeting.words, meeting.duration_samples)
        relative_path = str(wav_path.relative_to(root)).replace("\\", "/")
        sources = [
            SourceRef(
                role=f"speaker_{index}",
                speaker=speaker,
                session=meeting_id,
                utterance=meeting_id,
                file_sha256=meeting.wav_sha256,
                original_start_sample=0,
                original_end_sample=meeting.duration_samples,
                trimmed_start_sample=0,
                trimmed_end_sample=meeting.duration_samples,
                cut_start_sample=0,
                cut_end_sample=meeting.duration_samples,
                gain=1.0,
            )
            for index, speaker in enumerate(meeting.participants)
        ]
        condition: dict[str, object] = {
            "corpus": "ami",
            "meeting_id": meeting_id,
            "split_role": split_role,
            "recording_condition": "headset_mix_16k_mono",
            "stream": AMI_MIX_STREAM,
            "words": len(meeting.words),
            "parse_stats": stats,
            "partition_meta": meeting.partition_meta,
        }
        cases.append(
            Phase2Case(
                case_id=f"ami_{meeting_id}",
                wav_relative_path=relative_path,
                duration_samples=meeting.duration_samples,
                wav_sha256=meeting.wav_sha256,
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
            "name": "ami",
            "version": "1.6.2",
            "license": AMI_LICENSE,
            "source": AMI_BASE,
            "annotations_version": AMI_ANNOTATIONS_VERSION,
            "annotations_url": AMI_ANNOTATIONS_URL,
            "local_wav_root": str(root),
            "partition": "scenario_only",
        },
        build={
            "script": "corpus.ami.build_ami_manifest",
            "single_channel_recipe": "Mix-Headset 16 kHz mono",
            "annotation": "words.xml manual word-level v1.6.2, per-participant files",
            "speaker_rule": "speaker derived from words filename {meeting}.{letter}.words.xml as {meeting}.Participant{letter}",
            "ambiguous_rule": "word text containing % marks covering region ambiguous",
        },
        disjointness_groups=[f"ami_{split_role}"],
        generator={"script": "build_phase2_real.py"},
        cases=cases,
    )
    manifest_path = out_dir / "manifests" / f"{manifest_id}.json"
    manifest.write(manifest_path)
    return manifest


def default_ami_root() -> Path:
    return external.corpus_root() / "ami"
