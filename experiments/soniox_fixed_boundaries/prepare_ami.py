from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import wave
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator

import numpy as np

from puripuly_heart.core.audio.format import AudioCaptureSpan, AudioFrameF32
from puripuly_heart.core.audio.ownership import (
    AudioSegmentSettingsSnapshot,
    OwnedVadEvent,
    PeerAudioSegmentLedger,
)
from puripuly_heart.core.runtime.audio_vad_loop import run_audio_vad_loop
from puripuly_heart.core.vad.gating import SpeechEnd, create_peer_vad_gating
from puripuly_heart.core.vad.silero import SileroVadOnnx

ROOT = Path(__file__).resolve().parent
RATE = 16_000
FRAME = 512
WINDOWS = {
    "ES2002a": (2_640_000, 7_440_000),
    "IS1004a": (4_800_000, 9_600_000),
}
SPEAKERS = {
    "ES2002a": {"A": "MEE006", "B": "FEE005", "C": "MEE007", "D": "MEE008"},
    "IS1004a": {"A": "MIO019", "B": "MIE090", "C": "MIO022", "D": "MIO047"},
}
LICENSE_BASIS = (
    "AMI Meeting Corpus, CC BY 4.0 (https://groups.inf.ed.ac.uk/ami/corpus/); "
    "in-archive LICENCE.txt of ami_public_manual_1.6.2.zip verified CC BY 4.0"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def cut_window(source: Path, output: Path, start: int, end: int) -> None:
    with wave.open(str(source), "rb") as src:
        if (
            src.getframerate() != RATE
            or src.getnchannels() != 1
            or src.getsampwidth() != 2
            or src.getcomptype() != "NONE"
        ):
            raise ValueError(f"{source} is not mono 16 kHz PCM16")
        if end > src.getnframes():
            raise ValueError(f"{source} is shorter than the selected window")
        src.setpos(start)
        frames = src.readframes(end - start)
    if len(frames) != (end - start) * 2:
        raise RuntimeError(f"short read from {source}")
    with wave.open(str(output), "wb") as dst:
        dst.setnchannels(1)
        dst.setsampwidth(2)
        dst.setframerate(RATE)
        dst.writeframes(frames)


def build_reference(archive: Path, meeting: str, window_start: int, output: Path) -> dict[str, object]:
    items: list[dict[str, object]] = []
    turns: list[dict[str, object]] = []
    with zipfile.ZipFile(archive) as bundle:
        license_text = " ".join(
            bundle.read("LICENCE.txt").decode("utf-8", errors="replace").split()
        )
        if "Creative Commons Attribution 4.0 International" not in license_text:
            raise ValueError("AMI archive license is not the verified CC BY 4.0 text")
        for agent, speaker in SPEAKERS[meeting].items():
            member = f"words/{meeting}.{agent}.words.xml"
            root = ET.fromstring(bundle.read(member))
            for node in root.iter():
                kind = local_name(node.tag)
                if kind not in {"w", "vocalsound"}:
                    continue
                try:
                    start_s = float(node.attrib["starttime"])
                    end_s = float(node.attrib["endtime"])
                except (KeyError, ValueError):
                    continue
                start = round(start_s * RATE) - window_start
                end = round(end_s * RATE) - window_start
                if end <= 0 or start >= 300 * RATE:
                    continue
                start = max(0, start)
                end = min(300 * RATE, max(start, end))
                if kind == "w":
                    text = "".join(node.itertext()).strip()
                    if not text or node.attrib.get("punc") == "true":
                        continue
                    items.append(
                        {
                            "kind": "word",
                            "speaker_id": speaker,
                            "agent": agent,
                            "source_start_sample": start,
                            "source_end_sample": end,
                            "text": text,
                        }
                    )
                else:
                    items.append(
                        {
                            "kind": "vocal_sound",
                            "speaker_id": speaker,
                            "agent": agent,
                            "source_start_sample": start,
                            "source_end_sample": end,
                            "sound_type": node.attrib.get("type", "unknown"),
                        }
                    )
        for agent, speaker in SPEAKERS[meeting].items():
            member = f"segments/{meeting}.{agent}.segments.xml"
            root = ET.fromstring(bundle.read(member))
            for node in root.iter():
                if local_name(node.tag) != "segment":
                    continue
                try:
                    start = round(float(node.attrib["transcriber_start"]) * RATE) - window_start
                    end = round(float(node.attrib["transcriber_end"]) * RATE) - window_start
                except (KeyError, ValueError):
                    continue
                if end <= 0 or start >= 300 * RATE:
                    continue
                turns.append(
                    {
                        "speaker_id": speaker,
                        "agent": agent,
                        "source_start_sample": max(0, start),
                        "source_end_sample": min(300 * RATE, end),
                    }
                )
    turns.sort(key=lambda turn: (int(turn["source_start_sample"]), str(turn["speaker_id"])))
    items.sort(key=lambda item: (int(item["source_start_sample"]), str(item["speaker_id"])))
    reference = {
        "schema_version": 1,
        "meeting_id": meeting,
        "window_source_samples": [0, 300 * RATE],
        "meeting_window_start_sample": window_start,
        "speaker_map": SPEAKERS[meeting],
        "provenance": {
            "corpus": "AMI Meeting Corpus manual annotations 1.6.2",
            "transcription": "two/three-pass human transcript; one channel per participant",
            "timing": "forced alignment of the human transcript; word boundaries are estimates",
            "overlap": "derived from cross-speaker word-time intersections; no manual overlap layer",
            "turns": (
                "human NXT segments/<meeting>.<agent>.segments.xml spans, ordered by "
                "transcriber_start"
            ),
            "annotation_archive_sha256": sha256_file(archive),
            "source": "https://groups.inf.ed.ac.uk/ami/corpus/transcription.shtml",
        },
        "items": items,
        "turns": turns,
    }
    output.write_text(json.dumps(reference, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return reference


@dataclass
class WavWindowSource:
    path: Path
    capture_epoch: int
    terminal_reason: str = "closed"

    async def frames(self) -> AsyncIterator[AudioFrameF32]:
        sequence = 0
        cursor = 0
        with wave.open(str(self.path), "rb") as wav:
            while cursor < wav.getnframes():
                count = min(4096, wav.getnframes() - cursor)
                raw = wav.readframes(count)
                samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
                end = cursor + count
                yield AudioFrameF32(
                    samples=samples,
                    sample_rate_hz=RATE,
                    channels=1,
                    capture=AudioCaptureSpan(
                        capture_epoch=self.capture_epoch,
                        callback_sequence=sequence,
                        source_sample_rate_hz=RATE,
                        source_start_sample=cursor,
                        source_end_sample=end,
                        source_start_monotonic_s=cursor / RATE,
                        source_end_monotonic_s=end / RATE,
                    ),
                )
                cursor = end
                sequence += 1


async def freeze_schedule(
    path: Path, capture_epoch: int
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    model_path = (
        ROOT.parents[1] / "src" / "puripuly_heart" / "data" / "vad" / "silero_vad.onnx"
    )
    vad = create_peer_vad_gating(
        SileroVadOnnx(model_path),
        sample_rate_hz=RATE,
        ring_buffer_ms=500,
        hangover_ms=500,
    )
    ledger = PeerAudioSegmentLedger(
        activation_generation=1,
        settings=AudioSegmentSettingsSnapshot(
            provider_id="soniox",
            provider_signature=("soniox", "stt-rt-v5"),
            runtime_signature=("issue-157",),
            source_mode="ami_mix_headset",
            source_language="en",
            expected_languages=("en",),
            target_sample_rate_hz=RATE,
            vad_speech_threshold=0.5,
            vad_hangover_ms=500,
            vad_pre_roll_ms=500,
            delivery_profile_requested="off",
            delivery_profile_effective="off",
        ),
    )
    ends: dict[str, SpeechEnd] = {}

    class Sink:
        async def handle_owned_vad_event(self, owned: OwnedVadEvent) -> None:
            if isinstance(owned.event, SpeechEnd):
                ends[str(owned.segment.identity.segment_id)] = owned.event

        async def handle_vad_event(self, event: object) -> None:
            raise AssertionError(f"unowned event: {event!r}")

    await run_audio_vad_loop(
        source=WavWindowSource(path, capture_epoch),
        vad=vad,
        sink=Sink(),
        target_sample_rate_hz=RATE,
        segment_ledger=ledger,
        monotonic_clock=lambda: 0.0,
    )
    emitted: list[dict[str, object]] = []
    unsupported: list[dict[str, object]] = []
    for snapshot in ledger.snapshots:
        if not snapshot.content_ranges:
            continue
        start = snapshot.content_ranges[0].normalized_start_sample
        end = snapshot.content_ranges[-1].normalized_end_sample
        if start is None or end is None:
            raise RuntimeError("controller segment lacks normalized source coordinates")
        event = ends.get(str(snapshot.identity.segment_id))
        trailing = 0 if event is None else event.trailing_silence_ms * RATE // 1000
        duration = end - start
        if snapshot.seal_reason == "delivery_deadline":
            boundary_type = "hard_6s"
        elif snapshot.seal_reason == "delivery_pause" and duration >= 4 * RATE:
            boundary_type = "pause_224ms"
        elif snapshot.seal_reason == "delivery_pause":
            boundary_type = "natural_hangover"
        elif snapshot.seal_reason == "source_eof":
            boundary_type = "source_eof"
        else:
            unsupported.append(
                {
                    "segment_order": snapshot.identity.segment_order,
                    "seal_reason": snapshot.seal_reason,
                    "source_span": [start, end],
                }
            )
            continue
        emitted.append(
            {
                "id": f"segment-{snapshot.identity.segment_order:04d}",
                "source_start_sample": start,
                "source_end_sample": end,
                "prefix_spans": [
                    [span.normalized_start_sample, span.normalized_end_sample]
                    for span in snapshot.context_ranges
                    if span.normalized_start_sample is not None
                    and span.normalized_end_sample is not None
                    and span.normalized_end_sample <= start
                ],
                "boundary_type": boundary_type,
                "transmitted_trailing_silence_samples": trailing,
                "speech_end_source_sample": end - trailing if trailing else None,
                "vad_classification_note": (
                    "Silero peer VAD probability threshold 0.5 at 512-sample frames; "
                    "classification is model-derived and forced word timing is an independent "
                    "estimate"
                ),
                "controller_seal_reason": snapshot.seal_reason,
            }
        )
    return emitted, unsupported

def coverage(reference: dict[str, object], segments: list[dict[str, object]]) -> list[str]:
    items = list(reference["items"])
    turns = list(reference["turns"])
    turn_speakers = [str(turn["speaker_id"]) for turn in turns]
    tags = {"silence"}
    if any(item["kind"] == "vocal_sound" and item.get("sound_type") == "laugh" for item in items):
        tags.add("laughter")
    if any(item["boundary_type"] == "pause_224ms" for item in segments):
        tags.add("pause_boundary")
    if any(item["boundary_type"] == "hard_6s" for item in segments):
        tags.add("hard_boundary")
    if any(int(turn["source_end_sample"]) - int(turn["source_start_sample"]) >= 6 * RATE for turn in turns):
        tags.add("continuation_over_6s")
    if any(int(turn["source_end_sample"]) - int(turn["source_start_sample"]) < RATE for turn in turns):
        tags.add("brief_response")
    if any(
        turn_speakers[index] == turn_speakers[index + 2] != turn_speakers[index + 1]
        for index in range(len(turn_speakers) - 2)
    ):
        tags.add("aba")
    if any(len(set(turn_speakers[index : index + 3])) == 3 for index in range(len(turn_speakers) - 2)):
        tags.add("abc")
    overlaps = [
        (left, right)
        for index, left in enumerate(turns)
        for right in turns[index + 1 :]
        if left["speaker_id"] != right["speaker_id"]
        and int(left["source_start_sample"]) < int(right["source_end_sample"])
        and int(right["source_start_sample"]) < int(left["source_end_sample"])
    ]
    if overlaps:
        tags.update({"overlap", "interruption"})
    if any(
        turn_speakers[index] == turn_speakers[index + 1]
        for index in range(len(turn_speakers) - 1)
    ):
        tags.add("same_speaker_continuation")
    return sorted(tags)


async def prepare(args: argparse.Namespace) -> None:
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    recordings = []
    audit = []
    for capture_epoch, (meeting, (start, end)) in enumerate(WINDOWS.items(), start=1):
        source = args.audio_dir / f"{meeting}.Mix-Headset.wav"
        window = args.audio_dir / f"{meeting}.Mix-Headset.300s.wav"
        reference_path = args.reference_dir / f"{meeting}.reference.json"
        cut_window(source, window, start, end)
        reference = build_reference(args.annotation_zip, meeting, start, reference_path)
        segments, excluded = await freeze_schedule(window, capture_epoch)
        recordings.append(
            {
                "id": meeting,
                "path": str(window.relative_to(args.manifest.parent)),
                "sha256": sha256_file(window),
                "source_url": (
                    "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/HeadsetAudio/"
                    f"{meeting}.Mix-Headset.wav"
                ),
                "source_full_file_sha256": sha256_file(source),
                "source_meeting_window_samples": [start, end],
                "license_or_consent_basis": LICENSE_BASIS,
                "capture_epoch": capture_epoch,
                "boundary_schedule_scope": "intact_profile_off_baseline_4s_224ms_6s_including_natural_hangover_and_eof",
                "human_reference": {
                    "path": str(reference_path.relative_to(args.manifest.parent)),
                    "sha256": sha256_file(reference_path),
                    "checked_by": "AMI manual annotation provenance (two/three-pass human transcript)",
                    "checked_utc_date": "2026-09-12",
                },
                "session_source_start_sample": 0,
                "session_source_end_sample": 300 * RATE,
                "coverage_tags": coverage(reference, segments),
                "segments": segments,
            }
        )
        audit.append({"meeting_id": meeting, "qualifying_segments": len(segments), "excluded_segments": excluded})
    manifest["manifest_id"] = "issue-157-ami-es2002a-is1004a"
    manifest["recordings"] = recordings
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "prepared", "recordings": [item["id"] for item in recordings], "audit": str(args.audit)}, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare the two authorized AMI Issue #157 windows")
    parser.add_argument("--audio-dir", type=Path, default=ROOT / "selected_audio")
    parser.add_argument("--reference-dir", type=Path, default=ROOT / "human_references")
    parser.add_argument("--annotation-zip", type=Path, default=ROOT / "human_references" / "ami_public_manual_1.6.2.zip")
    parser.add_argument("--manifest", type=Path, default=ROOT / "manifest.json")
    parser.add_argument("--audit", type=Path, default=ROOT / "run_artifacts" / "schedule_audit.json")
    args = parser.parse_args()
    asyncio.run(prepare(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
