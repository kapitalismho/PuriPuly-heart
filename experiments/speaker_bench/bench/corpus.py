"""Build benchmark cases and case-local references from the frozen corpus spec."""

from __future__ import annotations

import argparse
import hashlib
import json
import wave
from pathlib import Path

import numpy as np

from bench import reference, sources

BENCH_ROOT = Path(__file__).resolve().parents[1]
SPEC = BENCH_ROOT / "corpus" / "spec.json"
REFERENCES = BENCH_ROOT / "corpus" / "references"
MANIFEST = BENCH_ROOT / "corpus" / "manifest.json"
AUDIO = sources.CACHE / "cases"
RATE = sources.SAMPLE_RATE_HZ
FRAME = 320
PAD_S = 0.05
TARGET_DBFS = -24.0


def _active_bounds(x: np.ndarray) -> tuple[int, int]:
    n = x.size // FRAME
    if n == 0:
        return 0, x.size
    rms = np.sqrt((x[: n * FRAME].reshape(n, FRAME) ** 2).mean(1) + 1e-12)
    db = 20 * np.log10(rms)
    threshold = max(-45.0, float(db.max()) - 35.0)
    active = np.flatnonzero(db > threshold)
    if active.size == 0:
        return 0, x.size
    return int(active[0] * FRAME), int((active[-1] + 1) * FRAME)


def _prepare_clip(x: np.ndarray, gain: float) -> tuple[np.ndarray, int, int]:
    lo, hi = _active_bounds(x)
    pad = int(PAD_S * RATE)
    start, end = max(0, lo - pad), min(x.size, hi + pad)
    clip = x[start:end].astype(np.float64)
    active = x[lo:hi].astype(np.float64)
    rms = float(np.sqrt((active**2).mean() + 1e-12))
    clip *= (10 ** (TARGET_DBFS / 20) / rms) * gain
    return clip, lo - start, hi - start


def _load(event: dict) -> tuple[np.ndarray, dict, str]:
    kind = event["source"]
    if kind == "librispeech":
        audio, info = sources.librispeech(event["id"])
        text = sources.librispeech_metadata()[event["id"]]["text"]
        return audio, {**info, "id": event["id"]}, text
    if kind == "common_voice":
        audio, info = sources.common_voice(event["path"])
        return audio, info, event["text"]
    if kind == "tts":
        audio, info = sources.tts(event["text"], event["voice"])
        return audio, info, event["text"]
    raise ValueError(f"unknown event source {kind}")


def build_assembled(case: dict) -> tuple[np.ndarray, list[dict], list[dict]]:
    placed = []
    timeline_end = case.get("lead_silence_s", 0.5)
    previous = None
    for event in case["events"]:
        audio, info, text = _load(event)
        clip, speech_lo, speech_hi = _prepare_clip(audio, float(event.get("gain", 1.0)))
        if "within_previous_s" in event:
            speech_start = previous["speech_start"] + float(event["within_previous_s"])
        else:
            speech_start = timeline_end + float(event.get("gap_s", 0.0))
        offset = int(round((speech_start * RATE))) - speech_lo
        if offset < 0:
            raise ValueError(f"{case['case_id']}: event placed before audio start")
        record = {
            "speaker": event["speaker"],
            "offset": offset,
            "clip": clip,
            "speech_start": (offset + speech_lo) / RATE,
            "speech_end": (offset + speech_hi) / RATE,
            "text": text,
            "info": info,
        }
        placed.append(record)
        timeline_end = max(timeline_end, record["speech_end"])
        if "within_previous_s" not in event:
            previous = record
    total = int(round((timeline_end + case.get("tail_silence_s", 1.0)) * RATE))
    total = max(total, max(p["offset"] + p["clip"].size for p in placed))
    mix = np.zeros(total, dtype=np.float64)
    for p in placed:
        mix[p["offset"] : p["offset"] + p["clip"].size] += p["clip"]
    peak = float(np.abs(mix).max())
    if peak > 0.95:
        mix *= 0.95 / peak
    turns = [
        {
            "speaker": p["speaker"],
            "start": round(p["speech_start"], 4),
            "end": round(p["speech_end"], 4),
            "text": p["text"],
            "words": None,
        }
        for p in placed
    ]
    provenance = [
        {"speaker": p["speaker"], "offset_samples": p["offset"], **p["info"]} for p in placed
    ]
    return mix.astype(np.float32), turns, provenance


def build_ami(case: dict) -> tuple[np.ndarray, list[dict], list[dict]]:
    meeting, start, end = case["meeting"], float(case["start_s"]), float(case["end_s"])
    audio, info = sources.ami_window(meeting, start, end)
    words, word_info = sources.ami_words(meeting)
    turns: list[dict] = []
    for word in words:
        if word["start"] < start or word["end"] > end:
            continue
        local = {
            "text": word["text"],
            "start": round(word["start"] - start, 4),
            "end": round(word["end"] - start, 4),
        }
        if (
            turns
            and turns[-1]["speaker"] == word["speaker"]
            and local["start"] - turns[-1]["end"] < 0.3
        ):
            turns[-1]["words"].append(local)
            turns[-1]["end"] = max(turns[-1]["end"], local["end"])
            turns[-1]["text"] += " " + local["text"]
            continue
        turns.append(
            {
                "speaker": word["speaker"],
                "start": local["start"],
                "end": local["end"],
                "text": local["text"],
                "words": [local],
            }
        )
    return audio, turns, [{**info, **word_info}]


def select_ami(meeting: str, *, length_s: float, gap_s: float, step_s: float = 5.0) -> dict:
    words, _ = sources.ami_words(meeting)
    last = max(w["end"] for w in words)
    best = None
    t = 0.0
    while t + length_s <= last:
        window = [w for w in words if w["start"] >= t and w["end"] <= t + length_s]
        if len({w["speaker"] for w in window}) >= 2:
            raw = {
                "case_id": "probe",
                "language": "en",
                "audio": {"samples": int(length_s * RATE), "sample_rate_hz": RATE},
                "turns": [
                    {
                        "speaker": w["speaker"],
                        "start": w["start"],
                        "end": w["end"],
                        "text": w["text"],
                        "words": [w],
                    }
                    for w in window
                ],
            }
            ref = reference.build(raw, gap_s=gap_s)
            kinds = [tr.kind for tr in ref.transitions]
            score = (kinds.count("continuous"), kinds.count("paused"), -kinds.count("overlap"))
            if best is None or score > best[0]:
                best = (score, t)
        t += step_s
    return {
        "meeting": meeting,
        "start_s": best[1],
        "end_s": best[1] + length_s,
        "selection_score": best[0],
    }


def _write_wav(path: Path, audio: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.clip(np.round(audio * 32767), -32768, 32767).astype("<i2").tobytes()
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(RATE)
        out.writeframes(pcm)
    return hashlib.sha256(pcm).hexdigest()


def build(case_ids: list[str] | None = None) -> dict:
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    manifest = {
        "schema": "speaker-bench/manifest/v1",
        "spec_sha256": hashlib.sha256(SPEC.read_bytes()).hexdigest(),
        "cases": [],
    }
    REFERENCES.mkdir(parents=True, exist_ok=True)
    for case in spec["cases"]:
        if case_ids and case["case_id"] not in case_ids:
            continue
        if case.get("kind") == "ami":
            audio, turns, provenance = build_ami(case)
        else:
            audio, turns, provenance = build_assembled(case)
        wav = AUDIO / f"{case['case_id']}.wav"
        pcm_sha = _write_wav(wav, audio)
        ref = {
            "schema": "speaker-bench/reference/v1",
            "case_id": case["case_id"],
            "category": case["category"],
            "language": case["language"],
            "synthetic": case.get("kind") != "ami",
            "synthetic_voice": any(p.get("synthetic_voice") for p in provenance),
            "description": case.get("description", ""),
            "audio": {
                "file": wav.name,
                "pcm_sha256": pcm_sha,
                "samples": int(audio.size),
                "sample_rate_hz": RATE,
            },
            "turns": turns,
            "provenance": provenance,
        }
        (REFERENCES / f"{case['case_id']}.json").write_text(
            json.dumps(ref, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        manifest["cases"].append(
            {
                "case_id": case["case_id"],
                "category": case["category"],
                "language": case["language"],
                "kind": case.get("kind", "assembled"),
                "duration_s": round(audio.size / RATE, 3),
                "speakers": len({t["speaker"] for t in turns}),
                "pcm_sha256": pcm_sha,
            }
        )
    if not case_ids:
        MANIFEST.write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("--case", action="append")
    s = sub.add_parser("select-ami")
    s.add_argument("meetings", nargs="+")
    s.add_argument("--length-s", type=float, default=75.0)
    s.add_argument("--gap-s", type=float, default=0.5)
    args = parser.parse_args()
    if args.command == "build":
        result = build(args.case)
        total = sum(c["duration_s"] for c in result["cases"])
        print(json.dumps({"cases": len(result["cases"]), "duration_s": round(total, 1)}))
    else:
        for meeting in args.meetings:
            print(json.dumps(select_ami(meeting, length_s=args.length_s, gap_s=args.gap_s)))


if __name__ == "__main__":
    main()
