from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import io
import json
import math
import platform
import re
import sys
import unicodedata
import wave
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import requests
from scipy.io import wavfile
from scipy.ndimage import binary_closing, binary_dilation

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = Path(__file__).resolve().parent
INPUTS = BENCHMARK / "inputs"
EVIDENCE = BENCHMARK / "data-preparation.json"
RATE = 16000
SEED = 30820261027
NS = "{http://nite.sourceforge.net/}"


def digest(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def key(value: str) -> str:
    return hashlib.sha256(f"{SEED}:{value}".encode()).hexdigest()


def fetch(record: dict, data_root: Path, offline: bool) -> dict:
    path = data_root / record["path"]
    if path.exists():
        if digest(path) != record["sha256"]:
            raise ValueError(f"Cached source SHA256 mismatch: {record['path']}")
        return dict(record, bytes=path.stat().st_size, acquisition="verified cached source")
    if offline:
        raise FileNotFoundError(f"Offline source missing: {record['path']}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("wb") as output:
        if record.get("range_request"):
            start, total = 0, None
            while total is None or start < total:
                end = start + 1024 * 1024 - 1
                if total is not None:
                    end = min(end, total - 1)
                with requests.get(
                    record["url"], headers={"Range": f"bytes={start}-{end}"}, timeout=120
                ) as response:
                    response.raise_for_status()
                    if response.status_code == 200 and start == 0:
                        output.write(response.content)
                        break
                    match = re.fullmatch(
                        r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", "")
                    )
                    if (
                        response.status_code != 206
                        or match is None
                        or int(match[1]) != start
                        or len(response.content) != int(match[2]) - start + 1
                    ):
                        raise ValueError(f"Invalid source byte-range response: {record['url']}")
                    total = int(match[3])
                    output.write(response.content)
                    start = int(match[2]) + 1
        else:
            with requests.get(record["url"], stream=True, timeout=120) as response:
                response.raise_for_status()
                for chunk in response.iter_content(1024 * 1024):
                    output.write(chunk)
    if digest(temporary) != record["sha256"]:
        raise ValueError(f"Downloaded source SHA256 mismatch: {record['url']}")
    temporary.replace(path)
    return dict(record, bytes=path.stat().st_size, acquisition="downloaded and SHA256 verified")


def read_audio(path: Path, start_s: float = 0, end_s: float | None = None) -> np.ndarray:
    with wave.open(str(path), "rb") as source:
        if (source.getframerate(), source.getnchannels(), source.getsampwidth()) != (RATE, 1, 2):
            raise ValueError(f"Pinned source must be mono 16kHz PCM16: {path}")
        start = round(start_s * RATE)
        end = source.getnframes() if end_s is None else round(end_s * RATE)
        if not 0 <= start < end <= source.getnframes():
            raise ValueError(f"Invalid source interval: {path}: {start_s}-{end_s}")
        source.setpos(start)
        raw = source.readframes(end - start)
    if len(raw) != 2 * (end - start):
        raise ValueError(f"Truncated PCM source: {path}")
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768


def source_audio(row: dict, data_root: Path) -> np.ndarray:
    if row.get("source_parquet"):
        if digest(data_root / row["input_path"]) != row["input_sha256"]:
            raise ValueError(f"Frozen parquet source audio SHA256 differs: {row['id']}")
        rate, audio = wavfile.read(data_root / row["input_path"])
        if rate != RATE or audio.ndim != 1 or audio.dtype != np.float32:
            raise ValueError(f"Pinned IHM input must be16kHz mono FLOAT: {row['id']}")
        if len(audio) != round(row["duration_s"] * RATE):
            raise ValueError(f"IHM input duration differs: {row['id']}")
        return audio
    return read_audio(data_root / row["input_path"], row["input_start_s"], row["input_end_s"])


def activity(audio: np.ndarray, annotated_fallback: bool = False) -> tuple[np.ndarray, dict]:
    padded = np.pad(audio, (0, (-len(audio)) % 320))
    rms = np.sqrt(np.mean(padded.reshape(-1, 320).astype(np.float64) ** 2, axis=1))
    threshold = max(10 ** (-50 / 20), float(np.max(rms)) * 10 ** (-30 / 20))
    frames = binary_closing(rms > threshold, structure=np.ones(3), border_value=0)
    frames = binary_dilation(frames, structure=np.ones(3))
    mask = np.repeat(frames, 320)[: len(audio)]
    if not np.any(mask):
        if not annotated_fallback or not np.any(audio):
            raise ValueError("No active speech under shared original energy estimate")
        return np.ones(len(audio), dtype=bool), dict(
            active_fraction=1.0, fallback="human_EAF_interval"
        )
    return mask, dict(threshold_dbfs=20 * math.log10(threshold), active_fraction=float(mask.mean()))


def pink(count: int, seed: int) -> np.ndarray:
    white = np.random.default_rng(seed).normal(size=count)
    spectrum = np.fft.rfft(white)
    frequencies = np.fft.rfftfreq(count, 1 / RATE)
    scale = np.zeros_like(frequencies)
    scale[1:] = 1 / np.sqrt(np.maximum(frequencies[1:], 20))
    noise = np.fft.irfft(spectrum * scale, n=count)
    noise -= np.mean(noise)
    return noise.astype(np.float32)


def noisy(
    audio: np.ndarray, mask: np.ndarray, noise: np.ndarray, snr: int
) -> tuple[np.ndarray, dict]:
    speech_power = float(np.mean(audio[mask].astype(np.float64) ** 2))
    noise_power = float(np.mean(noise[mask].astype(np.float64) ** 2))
    factor = math.sqrt(speech_power / (noise_power * 10 ** (snr / 10)))
    additive = noise * factor
    mixed = audio + additive
    gain = min(1.0, 0.98 / max(float(np.max(np.abs(mixed))), 1e-12))
    measured = 10 * math.log10(
        speech_power / float(np.mean(additive[mask].astype(np.float64) ** 2))
    )
    return mixed * gain, dict(
        snr_target_db=snr,
        snr_measured_db=measured,
        joint_gain=gain,
        peak=float(np.max(np.abs(mixed * gain))),
    )


def pcm16(audio: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(audio.astype(np.float64) * 32768), -32768, 32767).astype("<i2")


def write_audio(path: Path, audio: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as output:
        output.setparams((1, 2, RATE, 0, "NONE", "not compressed"))
        output.writeframes(pcm16(audio).tobytes())
    return digest(path)


def exported_snr(clean: np.ndarray, augmented: np.ndarray, mask: np.ndarray, gain: float) -> float:
    signal = clean.astype(np.float64) * gain
    residual = augmented.astype(np.float64) - signal
    return 10 * math.log10(float(np.mean(signal[mask] ** 2)) / float(np.mean(residual[mask] ** 2)))


def reconstruct_english(data_root: Path, sources: dict) -> None:
    import pyarrow.parquet as parquet

    target_meetings = {"ES2004d", "IS1009c", "TS3003c", "TS3003d"}
    candidates, rejected = [], Counter()
    global_offset = 0
    shards = [source for source in sources["files"] if source["kind"] == "parquet"]
    for shard in shards:
        index = 0
        for batch in parquet.ParquetFile(data_root / shard["path"]).iter_batches(batch_size=32):
            for row in batch.to_pylist():
                row_index = index
                index += 1
                if row["meeting_id"] not in target_meetings:
                    continue
                rate, audio = wavfile.read(io.BytesIO(row["audio"]["bytes"]))
                if rate != RATE or audio.ndim != 1 or audio.dtype != np.float32:
                    raise ValueError(f"Unexpected pinned IHM audio format: {row['audio_id']}")
                duration = len(audio) / RATE
                length = (
                    "brief"
                    if 0.5 <= duration <= 2
                    else "short" if 2 < duration <= 5 else "medium" if 5 < duration <= 15 else None
                )
                if length is None or not row["text"].strip():
                    rejected["outside_duration_or_empty_reference"] += 1
                    continue
                if abs(duration - (row["end_time"] - row["begin_time"])) > 0.001:
                    rejected["audio_metadata_duration_mismatch"] += 1
                    continue
                try:
                    mask, energy = activity(audio)
                    speech_end = (np.flatnonzero(mask)[-1] // 320 + 1) * 320 / RATE
                    speech_end = min(duration, speech_end)
                    end_method = "Shared clean-source20ms energy estimate; max(-50dBFS,peak-30dB), three-frame closing/dilation; not candidate-derived or lexical truth."
                except ValueError:
                    if not np.any(audio):
                        rejected["zero_audio_with_speech_reference"] += 1
                        continue
                    energy = dict(
                        mask_empty=True,
                        active_fraction=0.0,
                        fallback="upstream_human_utterance_interval",
                    )
                    speech_end = duration
                    end_method = "Upstream human/Kaldi utterance interval end because unchanged energy mask is empty; timing-only, not an observed acoustic end."
                identifier = f"ihm-test-{row['audio_id']}"
                candidates.append(
                    dict(
                        id=identifier,
                        language="en",
                        reference=row["text"],
                        duration_s=duration,
                        source="AMI Edinburgh IHM test; upstream Kaldi chunking and normalized human reference",
                        source_revision=sources["english_reconstruction"]["revision"],
                        split="test",
                        speaker_id=row["speaker_id"],
                        conversation_id=row["meeting_id"],
                        source_recording_id=row["meeting_id"],
                        source_channel=int(row["microphone_id"][1:]),
                        source_microphone_id=row["microphone_id"],
                        source_annotation_id=row["audio_id"],
                        source_start_s=row["begin_time"],
                        source_end_s=row["end_time"],
                        source_parquet=shard["path"],
                        source_parquet_sha256=shard["sha256"],
                        source_row_index=row_index,
                        source_dataset_row_index=global_offset + row_index,
                        source_audio_path=row["audio"]["path"],
                        input_path=f"sources/ami-ihm/clips/{identifier}.wav",
                        input_sha256=hashlib.sha256(row["audio"]["bytes"]).hexdigest(),
                        input_start_s=0,
                        input_end_s=duration,
                        input_format="upstream FLOAT WAV",
                        group="medium" if length == "medium" else "short",
                        length_bucket=length,
                        activity=energy,
                        speech_end_s=speech_end,
                        speech_end_method=end_method,
                        license="CC-BY-4.0",
                        variant="clean",
                        noise_pair_id=None,
                        reconstruction="Approved test-split reconstruction; not exact #308 followup selection",
                    )
                )
        global_offset += index
    selected = []
    for length in ("brief", "short", "medium"):
        pools = {
            meeting: sorted(
                [
                    row
                    for row in candidates
                    if row["conversation_id"] == meeting and row["length_bucket"] == length
                ],
                key=lambda row: key("reconstruct:" + row["id"]),
            )
            for meeting in sorted(target_meetings)
        }
        for _ in range(10):
            for meeting, pool in pools.items():
                if not pool:
                    raise ValueError(f"Insufficient IHM test candidates: {meeting}/{length}")
                selected.append(pool.pop(0))
    eligible = [row for row in selected if not row["activity"].get("mask_empty")]
    parents = {
        row["id"] for row in sorted(eligible, key=lambda row: key("noise:" + row["id"]))[:30]
    }
    if len(parents) != 30 or len({row["speaker_id"] for row in selected}) < 4:
        raise ValueError("Reconstructed input needs30noise parents and speaker spread")
    for row in selected:
        row.update(noise_pair_selected=row["id"] in parents, runtime_subset=row["id"] in parents)
    path = INPUTS / "selection-en.jsonl"
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected), encoding="utf-8"
    )
    sources["english_reconstruction"].update(
        selection_rule="Ten per target meeting per bucket, deterministic SHA256(seed30820261027:reconstruct:id) order; actual waveform duration; no candidate outputs",
        rejected_candidate_counts=dict(rejected),
        candidate_count=len(candidates),
        row_count=120,
        clean_length_buckets=dict(Counter(row["length_bucket"] for row in selected)),
        speaker_labels=sorted({row["speaker_id"] for row in selected}),
    )
    save_json(INPUTS / "sources.json", sources)


def materialize_english(data_root: Path) -> None:
    import pyarrow.parquet as parquet

    rows = load_jsonl(INPUTS / "selection-en.jsonl")
    needed = {}
    for row in rows:
        path = data_root / row["input_path"]
        if path.exists() and digest(path) == row["input_sha256"]:
            continue
        needed.setdefault(row["source_parquet"], {})[row["source_row_index"]] = row
    for shard, wanted in needed.items():
        index = 0
        for batch in parquet.ParquetFile(data_root / shard).iter_batches(
            batch_size=32, columns=["audio"]
        ):
            for record in batch.to_pylist():
                row = wanted.get(index)
                index += 1
                if row is None:
                    continue
                encoded = record["audio"]["bytes"]
                if hashlib.sha256(encoded).hexdigest() != row["input_sha256"]:
                    raise ValueError(f"Frozen parquet row audio differs: {row['id']}")
                path = data_root / row["input_path"]
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(encoded)
                del wanted[row["source_row_index"]]
            if not wanted:
                break
        if wanted:
            raise ValueError(f"Frozen parquet rows not found: {shard}")


def render(language: str, data_root: Path) -> None:
    selection = load_jsonl(INPUTS / f"selection-{language}.jsonl")
    if len(selection) != 120 or sum(row["noise_pair_selected"] for row in selection) != 30:
        raise ValueError("Frozen selection must contain 120 originals and 30 noise parents")
    rows = []
    historical_ja = json.loads((INPUTS / "sources.json").read_text(encoding="utf-8")).get(
        "historical_ja_float_hashes", {}
    )
    manifest = INPUTS / f"{language}.jsonl"
    for selected in selection:
        audio = source_audio(selected, data_root)
        if language == "en" and selected["activity"].get("mask_empty"):
            if selected["noise_pair_selected"]:
                raise ValueError("Recovered empty-energy clean originals cannot be noise parents")
            mask, energy = None, selected["activity"]
        else:
            mask, energy = activity(audio, annotated_fallback=language == "ja")
        base = dict(
            selected,
            duration_s=len(audio) / RATE,
            variant="clean",
            noise_pair_id=None,
            activity=energy,
            path=f"{language}/{selected['id']}.wav",
        )
        base["sha256"] = write_audio(data_root / base["path"], audio)
        rows.append(base)
        if not selected["noise_pair_selected"]:
            continue
        seed = int(key(selected["id"])[:16], 16)
        noise = pink(len(audio), seed)
        for snr in (15, 5):
            augmented, evidence = noisy(audio, mask, noise, snr)
            evidence.update(
                seed=seed,
                recipe="original #308 seeded FFT pink/fan-like; same realization at 15 and 5 dB",
            )
            identifier = f"{selected['id']}-noise{snr}"
            row = dict(
                base,
                id=identifier,
                variant=f"noise{snr}",
                noise_pair_id=selected["id"],
                noise_kind="pink",
                noise_realization=1,
                noise=evidence,
                runtime_subset=snr == 15,
                path=f"{language}/{identifier}.wav",
            )
            row.pop("historical_float_sha256", None)
            if historical_ja.get(identifier):
                row["historical_float_sha256"] = historical_ja[identifier]
            row["sha256"] = write_audio(data_root / row["path"], augmented)
            rows.append(row)
    for index, seconds in enumerate((1.0, 2.0, 3.0, 2.5, 4.0)):
        audio = (
            np.zeros(round(seconds * RATE), dtype=np.float32)
            if index < 2
            else pink(round(seconds * RATE), SEED + index) * 0.01
        )
        identifier = f"{language}-control-{index}"
        path = f"{language}/{identifier}.wav"
        rows.append(
            dict(
                id=identifier,
                language=language,
                reference="",
                duration_s=seconds,
                variant="control",
                noise_pair_id=None,
                path=path,
                sha256=write_audio(data_root / path, audio),
                group="control",
                source="synthetic silence/pink-noise control, no speech",
                source_revision=f"seed:{SEED + index}" if index >= 2 else "exact zero PCM",
                split="control",
                speaker_id=None,
                conversation_id=None,
                source_recording_id=None,
                source_channel=0,
                source_start_s=0,
                source_end_s=seconds,
                speech_end_s=None,
                speech_end_method="not applicable: no speech",
                runtime_subset=False,
            )
        )
    manifest.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    replay_path = INPUTS / f"replay-{language}.json"
    replay = json.loads(replay_path.read_text(encoding="utf-8"))
    source = replay["source"]
    clean = None
    for path in source["source_paths"]:
        audio = read_audio(data_root / path, source["window_start_s"], source["window_end_s"])
        if clean is None:
            clean = np.zeros_like(audio)
        clean += audio / len(source["source_paths"])
    if clean is None or len(clean) != 120 * RATE:
        raise ValueError("Frozen natural replay must contain exactly 120 seconds")
    if language == "en" and source.get("reconstructed_channel_policy"):
        for turn in replay["turns"]:
            left = round(turn["offset_s"] * RATE)
            right = left + round(turn["duration_s"] * RATE)
            try:
                turn_mask, turn_energy = activity(clean[left:right])
                end = min(
                    (right - left) / RATE, (np.flatnonzero(turn_mask)[-1] // 320 + 1) * 320 / RATE
                )
                method = "Recomputed shared clean Mix-Headset20ms energy estimate inside recovered original human interval; overlap/background may bias it; not candidate-derived."
            except ValueError:
                end = (right - left) / RATE
                turn_energy = dict(mask_empty=True, fallback="original_human_interval")
                method = "Recovered original human interval end; new mixed-source energy mask empty. Timing-only, not an observed acoustic end."
            turn.update(
                speech_end_s=turn["offset_s"] + end, speech_end_method=method, activity=turn_energy
            )
    mask, energy = activity(clean)
    augmented, evidence = noisy(clean, mask, pink(len(clean), replay["noise_seed"]), 15)
    evidence.update(
        seed=replay["noise_seed"],
        recipe="original #308 seeded FFT pink/fan-like at 15dB; active RMS over complete clean window",
        activity=energy,
    )
    replay["noise"] = evidence
    replay["sha256"] = {
        variant: write_audio(data_root / replay["paths"][variant], values)
        for variant, values in (("clean", clean), ("noise15", augmented))
    }
    save_json(replay_path, replay)


def eaf_rows(path: Path) -> dict:
    document = ET.parse(path).getroot()
    if document.find("HEADER").attrib["TIME_UNITS"] != "milliseconds":
        raise ValueError("EAF time units must be milliseconds")
    times = {
        slot.attrib["TIME_SLOT_ID"]: int(slot.attrib["TIME_VALUE"]) / 1000
        for slot in document.findall("./TIME_ORDER/TIME_SLOT")
    }
    return {
        (tier.attrib["TIER_ID"], node.attrib["ANNOTATION_ID"]): (
            node.findtext("ANNOTATION_VALUE", ""),
            times[node.attrib["TIME_SLOT_REF1"]],
            times[node.attrib["TIME_SLOT_REF2"]],
        )
        for tier in document.findall("TIER")
        for node in tier.findall("./ANNOTATION/ALIGNABLE_ANNOTATION")
    }


def check_reference(row: dict, language: str, data_root: Path, replay: bool = False) -> dict:
    reference = row["source_reference"] if replay else row["reference"]
    if language == "ja":
        recording = row["speaker_id"].split(":")[0]
        tier = row.get("source_tier", row["speaker_id"].split(":")[-1])
        identifier = row["source_annotation_id"].split("-")[-1]
        actual = eaf_rows(data_root / f"sources/ingcrowd/{recording}.eaf")[(tier, identifier)]
        if actual != (reference, row["source_start_s"], row["source_end_s"]):
            raise ValueError(f"EAF reference/boundaries differ: {row['id']}")
    elif row.get("source_parquet"):
        recording = row["conversation_id"]
        with zipfile.ZipFile(data_root / "sources/ami/ami_public_manual_1.6.2.zip") as archive:
            meetings = ET.fromstring(archive.read("corpusResources/meetings.xml"))
            meeting = next(node for node in meetings if node.attrib.get("observation") == recording)
            speaker = next(
                node for node in meeting if int(node.attrib["channel"]) == row["source_channel"]
            )
            agent = speaker.attrib["nxt_agent"]
            words = ET.fromstring(archive.read(f"words/{recording}.{agent}.words.xml"))
            nodes = [
                node
                for node in words
                if node.tag == "w"
                and float(node.attrib.get("starttime", "-1")) >= row["source_start_s"] - 0.05
                and float(node.attrib.get("endtime", "inf")) <= row["source_end_s"] + 0.05
            ]
        manual = " ".join(node.text or "" for node in nodes)

        def normalized(text: str) -> str:
            return " ".join(
                "".join(
                    character if unicodedata.category(character)[0] in "LNM" else " "
                    for character in unicodedata.normalize("NFKC", text).casefold()
                ).split()
            )

        id_bounds = row["source_annotation_id"].rsplit("_", 2)[-2:]
        identifier_bounds_match = all(
            abs(int(value) / 100 - row[field]) < 0.001
            for value, field in zip(id_bounds, ("source_start_s", "source_end_s"))
        )
        if not identifier_bounds_match:
            raise ValueError(f"IHM row ID/bounds differ: {row['id']}")
        return dict(
            id=row["id"],
            reference=reference,
            source_start_s=row["source_start_s"],
            source_end_s=row["source_end_s"],
            upstream_id_bounds_match=True,
            upstream_waveform_duration_metadata_error_s=row["duration_s"]
            - (row["source_end_s"] - row["source_start_s"]),
            original_manual_word_text=manual,
            manual_word_ids=[node.attrib[NS + "id"] for node in nodes],
            human_reference_normalized_match=normalized(manual) == normalized(reference),
            reference_policy="Upstream human/Kaldi-normalized text preserved; compare original manual word XML within exposed interval±50ms. Differences remain visible, not rewritten from candidate outputs.",
        )
    else:
        identifier = row["source_annotation_id"]
        recording = identifier.split(".")[0]
        word_ids = row.get("reference_word_ids") or row.get("annotation_word_ids")
        agent = word_ids[0].split(".")[1]
        with zipfile.ZipFile(data_root / "sources/ami/ami_public_manual_1.6.2.zip") as archive:
            words = ET.fromstring(archive.read(f"words/{recording}.{agent}.words.xml"))
            wanted = set(word_ids)
            nodes = [node for node in words if node.attrib.get(NS + "id") in wanted]
            if [node.attrib[NS + "id"] for node in nodes] != word_ids or " ".join(
                node.text or "" for node in nodes
            ) != reference:
                raise ValueError(f"AMI human word reference differs: {row['id']}")
            segments = ET.fromstring(archive.read(f"segments/{recording}.{agent}.segments.xml"))
            segment = next(node for node in segments if node.attrib[NS + "id"] == identifier)
            bounds = (
                float(segment.attrib["transcriber_start"]),
                float(segment.attrib["transcriber_end"]),
            )
            if bounds != (row["source_start_s"], row["source_end_s"]):
                raise ValueError(f"AMI original segment boundary differs: {row['id']}")
    return dict(
        id=row["id"],
        reference=reference,
        source_start_s=row["source_start_s"],
        source_end_s=row["source_end_s"],
        human_annotation_reference_and_bounds_match=True,
    )


def check(language: str, data_root: Path) -> dict:
    rows = load_jsonl(INPUTS / f"{language}.jsonl")
    counts = Counter(row["variant"] for row in rows)
    if (
        counts != {"clean": 120, "noise15": 30, "noise5": 30, "control": 5}
        or len({row["id"] for row in rows}) != 185
    ):
        raise ValueError(f"Incorrect frozen manifest counts/IDs: {counts}")
    originals = {row["id"]: row for row in rows if row["variant"] == "clean"}
    selection = {row["id"]: row for row in load_jsonl(INPUTS / f"selection-{language}.jsonl")}
    expected_parents = {row["id"] for row in selection.values() if row["noise_pair_selected"]}
    if set(originals) != set(selection) or len(expected_parents) != 30:
        raise ValueError("Clean IDs or noise parents differ from frozen selection")
    for variant in ("noise15", "noise5"):
        if {row["noise_pair_id"] for row in rows if row["variant"] == variant} != expected_parents:
            raise ValueError(f"Noise parent set differs from frozen selection: {variant}")
    checks, snrs = [], []
    for row in rows:
        path = data_root / row["path"]
        if digest(path) != row["sha256"]:
            raise ValueError(f"Prepared SHA256 mismatch: {row['id']}")
        audio = read_audio(path)
        if len(audio) != round(row["duration_s"] * RATE):
            raise ValueError(f"Prepared duration mismatch: {row['id']}")
        if row["variant"] in ("clean", "control") and row["noise_pair_id"] is not None:
            raise ValueError("Only augmentations may carry a noise parent")
        if row["variant"] == "clean":
            source = source_audio(selection[row["id"]], data_root)
            if row["reference"] != selection[row["id"]]["reference"] or not np.array_equal(
                pcm16(source), pcm16(audio)
            ):
                raise ValueError(f"Original crop/reference differs: {row['id']}")
        elif row["variant"].startswith("noise"):
            parent = originals[row["noise_pair_id"]]
            if row["reference"] != parent["reference"]:
                raise ValueError(f"Noisy reference differs: {row['id']}")
            clean = read_audio(data_root / parent["path"])
            source_clean = source_audio(selection[parent["id"]], data_root)
            mask, _ = activity(source_clean, annotated_fallback=language == "ja")
            seed = int(key(parent["id"])[:16], 16)
            expected, _ = noisy(
                source_clean, mask, pink(len(clean), seed), row["noise"]["snr_target_db"]
            )
            if row["noise"]["seed"] != seed or not np.array_equal(pcm16(expected), pcm16(audio)):
                raise ValueError(f"Original seeded noise recipe differs: {row['id']}")
            measured = exported_snr(clean, audio, mask, row["noise"]["joint_gain"])
            snrs.append(
                dict(
                    id=row["id"],
                    target_db=row["noise"]["snr_target_db"],
                    pcm16_snr_db=measured,
                    error_db=measured - row["noise"]["snr_target_db"],
                    peak=float(np.max(np.abs(audio))),
                )
            )
        elif row["variant"] == "control":
            index = int(row["id"].rsplit("-", 1)[1])
            expected = (
                np.zeros(len(audio), dtype=np.float32)
                if index < 2
                else pink(len(audio), SEED + index) * 0.01
            )
            if not np.array_equal(pcm16(expected), pcm16(audio)):
                raise ValueError(f"Original control recipe differs: {row['id']}")
    clean_rows = list(originals.values())
    indices = (0, 1, 40, 80) if language == "en" else (0, 1, 108, 109)
    for index in indices:
        row = clean_rows[index]
        spot = check_reference(row, language, data_root)
        audio = read_audio(data_root / row["path"])
        spot.update(
            crop_pcm16_samples_match=True,
            speech_end_s=row["speech_end_s"],
            first_20ms_rms_dbfs=20
            * math.log10(max(float(np.sqrt(np.mean(audio[:320].astype(np.float64) ** 2))), 1e-12)),
            last_20ms_rms_dbfs=20
            * math.log10(max(float(np.sqrt(np.mean(audio[-320:].astype(np.float64) ** 2))), 1e-12)),
        )
        checks.append(spot)
    replay = json.loads((INPUTS / f"replay-{language}.json").read_text(encoding="utf-8"))
    replay_audio = {}
    for variant, name in replay["paths"].items():
        path = data_root / name
        if digest(path) != replay["sha256"][variant]:
            raise ValueError(f"Replay SHA256 differs: {language}/{variant}")
        replay_audio[variant] = read_audio(path)
        if len(replay_audio[variant]) != 120 * RATE:
            raise ValueError("Replay duration must be 120 seconds")
    source = replay["source"]
    native = np.zeros(120 * RATE, dtype=np.float32)
    for name in source["source_paths"]:
        native += read_audio(
            data_root / name, source["window_start_s"], source["window_end_s"]
        ) / len(source["source_paths"])
    if not np.array_equal(pcm16(native), pcm16(replay_audio["clean"])):
        raise ValueError("Natural replay source samples differ")
    for turn in replay["turns"]:
        if (
            abs(
                turn["offset_s"]
                - (max(turn["source_start_s"], source["window_start_s"]) - source["window_start_s"])
            )
            > 1 / RATE
            or not turn["offset_s"]
            <= turn["speech_end_s"]
            <= turn["offset_s"] + turn["duration_s"] + 1 / RATE
        ):
            raise ValueError(f"Replay source timing differs: {turn['id']}")
    replay_checks = [
        check_reference(turn, language, data_root, replay=True) for turn in replay["turns"][:2]
    ]
    mask, _ = activity(native)
    return dict(
        variant_counts=dict(counts),
        all_185_sha256_headers_durations_checked=True,
        all_120_clean_crop_samples_and_references_match_frozen_selection=True,
        clean_duration_s=sum(row["duration_s"] for row in clean_rows),
        clean_length_buckets=dict(
            Counter(row.get("length_bucket", row["group"]) for row in clean_rows)
        ),
        conversations=sorted({row["conversation_id"] for row in clean_rows}),
        speaker_labels=sorted({row["speaker_id"] for row in clean_rows}),
        noise_parent_ids=[row["id"] for row in clean_rows if row["noise_pair_selected"]],
        noise_export_checks=snrs,
        maximum_pcm16_snr_error_db=max(abs(row["error_db"]) for row in snrs),
        spot_checks=checks,
        replay=dict(
            duration_s=120,
            continuous_source_pcm16_samples_match=True,
            turn_count=len(replay["turns"]),
            reference_available_count=sum(turn["reference_available"] for turn in replay["turns"]),
            original_turn_timing_bounds_checked=True,
            annotation_spot_checks=replay_checks,
            pcm16_snr_db=exported_snr(
                replay_audio["clean"], replay_audio["noise15"], mask, replay["noise"]["joint_gain"]
            ),
        ),
        auditory_review_performed=False,
        review_limit="Human annotation reference/interval identity and source crop PCM sample integrity checked, not listening review. Shared recovered energy speech ends are estimates, not acoustic ground truth; overlap/room noise may bias them.",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare reconstructed AMI IHM test EN and exact recovered IngCrowd JA inputs for #353; no ASR or application startup."
    )
    parser.add_argument("--data-root", type=Path, default=BENCHMARK / "data")
    parser.add_argument("--language", choices=("en", "ja", "all"), default="all")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Require SHA256-matching sources already under DATA_ROOT.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Check frozen outputs and source references without rendering or downloading.",
    )
    parser.add_argument(
        "--acquire-only", action="store_true", help="Acquire and hash sources without rendering."
    )
    args = parser.parse_args()
    languages = ("en", "ja") if args.language == "all" else (args.language,)
    sources = json.loads((INPUTS / "sources.json").read_text(encoding="utf-8"))
    record = json.loads(EVIDENCE.read_text(encoding="utf-8")) if EVIDENCE.exists() else {}
    record.update(
        issue="https://github.com/kapitalismho/PuriPuly-heart/issues/353",
        preparation_code="benchmarks/asr/prepare.py",
        command=" ".join(["python", "benchmarks/asr/prepare.py", *sys.argv[1:]]),
        recorded_at_utc=datetime.now(timezone.utc).isoformat(),
        python=platform.python_version(),
        dependencies={
            name: importlib.metadata.version(name) for name in ("numpy", "scipy", "requests")
        },
        output_format=dict(
            sample_rate_hz=RATE,
            channels=1,
            encoding="PCM16 little-endian WAV",
            conversion=(
                "Historical prepared WAVs were FLOAT. Recreate original float32 crop/mix/noise, "
                "then round-to-nearest ties-to-even(sample*32768), saturate to signed16 and "
                "export via stdlib wave. PCM16 hashes replace historical FLOAT hashes; source "
                "PCM16 crop samples are unchanged. Pinned original sources already16kHz mono; "
                "no resampling required."
            ),
            quantization_limit=(
                "Post-PCM16 effective SNR includes quantization; actual per-file SNR retained. "
                "Historical active-speech FLOAT recipe still targets exactly15/5dB."
            ),
        ),
        seed=SEED,
        noise_recipe=dict(
            parent_seed="int(SHA256(f'{30820261027}:{parent_id}')[:16],16)",
            en_parent_selection=sources["en_noise_parent_eligibility"]["rule"],
            ja_parent_selection="Recovered original30:20short+10medium",
            spectrum=(
                "default_rng(seed).normal; rFFT; non-DC scale1/sqrt(max(freq,20Hz)); "
                "DC0; irFFT; mean removed; float32"
            ),
            active_speech=(
                "320-sample20ms RMS frames; max(-50dBFS,peak-30dB), three-frame closing then "
                "dilation; JA human EAF full-interval fallback if mask empty"
            ),
            mixing=(
                "sqrt(clean_active_power/(noise_active_power*10**(SNR/10))); "
                "joint gain min(1,0.98/mixed_peak)"
            ),
            controls="durations1,2,3,2.5,4s; first2 exact silence; last3pink(seed=30820261027+index)*0.01",
        ),
        provenance="inputs/sources.json",
        limitations=sources["limitations"],
        recovered_english_noise_parent_eligibility=sources.get(
            "recovered_english_noise_parent_eligibility"
        ),
        english_noise_parent_eligibility=sources.get("en_noise_parent_eligibility"),
        source_license_checks=sources.get("license_source_checks", []),
        inference_performed=False,
    )
    acquired, failures = [], []
    selected_sources = [source for source in sources["files"] if source["language"] in languages]
    with ThreadPoolExecutor(max_workers=4) as pool:
        pending = {
            pool.submit(fetch, source, args.data_root, args.offline or args.check_only): source
            for source in selected_sources
        }
        for future in as_completed(pending):
            source = pending[future]
            try:
                acquired.append(future.result())
                print(json.dumps(dict(acquired=source["path"])), flush=True)
            except Exception as error:
                failure = dict(
                    path=source["path"], url=source["url"], error=f"{type(error).__name__}: {error}"
                )
                failures.append(failure)
                print(json.dumps(dict(failed=failure)), flush=True)
    prior = {source["path"]: source for source in record.get("sources", [])}
    prior.update({source["path"]: source for source in acquired})
    record["sources"] = sorted(prior.values(), key=lambda source: source["path"])
    record["acquisition_failures"] = failures
    record.setdefault("attempts", []).append(
        dict(
            command=record["command"],
            recorded_at_utc=datetime.now(timezone.utc).isoformat(),
            acquired_source_count=len(acquired),
            acquisition_failures=failures,
            mode=(
                "check-only"
                if args.check_only
                else "acquire-only" if args.acquire_only else "render-and-check"
            ),
        )
    )
    save_json(EVIDENCE, record)
    if failures:
        raise SystemExit(
            f"{len(failures)} source failures; exact URLs/errors in {EVIDENCE.relative_to(ROOT)}"
        )
    if not args.acquire_only:
        for language in languages:
            try:
                if not args.check_only:
                    if language == "en" and sources.get("english_reconstruction"):
                        if not load_jsonl(INPUTS / "selection-en.jsonl")[0].get("source_parquet"):
                            reconstruct_english(args.data_root, sources)
                        materialize_english(args.data_root)
                    render(language, args.data_root)
                record.setdefault("languages", {})[language] = check(language, args.data_root)
            except Exception as error:
                record["attempts"][-1]["preparation_failure"] = dict(
                    language=language, error=f"{type(error).__name__}: {error}"
                )
                save_json(EVIDENCE, record)
                raise
            print(
                json.dumps(
                    dict(prepared=language, counts=record["languages"][language]["variant_counts"])
                ),
                flush=True,
            )
        if "en" in languages and sources.get("english_reconstruction"):
            record["dependencies"]["pyarrow"] = importlib.metadata.version("pyarrow")
            record["english_reconstruction"] = json.loads(
                (INPUTS / "sources.json").read_text(encoding="utf-8")
            )["english_reconstruction"]
        record["frozen_input_hashes"] = {
            path.name: digest(path)
            for path in sorted(INPUTS.iterdir())
            if path.suffix in (".json", ".jsonl")
        }
        record["preparation_code_sha256"] = digest(Path(__file__))
        record["preparation_complete"] = all(
            language in record.get("languages", {}) for language in ("en", "ja")
        )
        save_json(EVIDENCE, record)


if __name__ == "__main__":
    main()
