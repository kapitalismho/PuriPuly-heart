from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import re
import shutil
import subprocess
import sys
import tarfile
import wave
import xml.etree.ElementTree as ET
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = Path(__file__).resolve().parent
DATA = ROOT / ".tmp/issue441/input"
ORIGINAL = ROOT / "../../../benchmarks/asr/data"
MANIFEST = BENCHMARK / "issue441-inputs.json"
RATE = 16000
NS = "{http://nite.sourceforge.net/}"
ALI_REV = "cef2837c4a1a0762074522c1bbbbb90eb4d1acf5"
ALI_BASE = f"https://huggingface.co/datasets/ggfox00000/dia-alimeeting-test/resolve/{ALI_REV}"
KO_OFFSET = round(66.58811791383219 * RATE) / RATE


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def relative(path: Path) -> str:
    import os

    return Path(os.path.relpath(path.resolve(), ROOT)).as_posix()


def fetch(url: str, path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    with requests.get(url, stream=True, timeout=(30, 180)) as response:
        response.raise_for_status()
        if response.status_code != 200:
            raise ValueError(f"Full acquisition returned {response.status_code}: {url}")
        with temporary.open("wb") as output:
            for chunk in response.iter_content(1024 * 1024):
                output.write(chunk)
    temporary.replace(path)


def write_wav(path: Path, pcm: bytes) -> None:
    with wave.open(str(path), "wb") as output:
        output.setparams((1, 2, RATE, 0, "NONE", "not compressed"))
        output.writeframes(pcm)


def read_wav(path: Path, offset: float = 0, duration: float = 120) -> bytes:
    with wave.open(str(path), "rb") as stream:
        if (stream.getframerate(), stream.getnchannels(), stream.getsampwidth()) != (RATE, 1, 2):
            raise ValueError(f"Not mono 16k PCM16: {path}")
        stream.setpos(round(offset * RATE))
        pcm = stream.readframes(round(duration * RATE))
    if len(pcm) != round(duration * RATE) * 2:
        raise ValueError(f"Incomplete source window: {path}")
    return pcm


def acquire() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    for name in ("label", "sound"):
        host = "openslr.elda.org" if name == "sound" else "openslr.trmal.net"
        fetch(f"https://{host}/resources/113/{name}.tgz", DATA / f"seoul-{name}.tgz")
    flac = DATA / "s13m26m1.flac"
    if not flac.exists():
        nested = DATA / "seoul-sound.zip"
        with tarfile.open(DATA / "seoul-sound.tgz") as archive:
            member = next(m for m in archive.getmembers() if m.name.endswith(".zip"))
            with archive.extractfile(member) as source, nested.open("wb") as target:
                shutil.copyfileobj(source, target, 1024 * 1024)
        with zipfile.ZipFile(nested) as archive:
            member = next(
                n
                for n in archive.namelist()
                if not n.startswith("__MACOSX/") and Path(n).name == flac.name
            )
            with archive.open(member) as source, flac.open("wb") as target:
                shutil.copyfileobj(source, target)
        nested.unlink()
    ko = DATA / "ko-120.wav"
    if not ko.exists():
        subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-v",
                "error",
                "-i",
                str(flac),
                "-ss",
                str(KO_OFFSET),
                "-t",
                "120",
                "-ar",
                str(RATE),
                "-ac",
                "1",
                "-c:a",
                "pcm_s16le",
                str(ko),
            ],
            check=True,
        )
    zh = DATA / "zh-120.wav"
    source_range = DATA / "ali-source-range.bin"
    if not source_range.exists():
        url = ALI_BASE + "/audio/far/R8002_M8002_MS802.wav"
        response = requests.get(url, headers={"Range": "bytes=0-30724095"}, timeout=180)
        response.raise_for_status()
        if (
            response.status_code != 206
            or response.headers.get("Content-Range") != "bytes 0-30724095/529029200"
        ):
            raise ValueError("AliMeeting original eight-channel range was not honored")
        source_range.write_bytes(response.content)
        (DATA / "ali-acquisition.json").write_text(
            json.dumps(
                dict(
                    url=url,
                    revision=ALI_REV,
                    content_range=response.headers["Content-Range"],
                    input_channel=0,
                    downloaded_bytes=len(response.content),
                    data_offset=80,
                    source_range_sha256=digest(source_range),
                    source_range_path=relative(source_range),
                ),
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    if not zh.exists():
        with wave.open(io.BytesIO(source_range.read_bytes())) as source:
            if (source.getframerate(), source.getnchannels(), source.getsampwidth()) != (
                RATE,
                8,
                2,
            ):
                raise ValueError("AliMeeting channel/header contract changed")
            pcm = source.readframes(RATE * 120)
        mono = np.frombuffer(pcm, dtype="<i2").reshape(-1, 8)[:, 0].copy()
        write_wav(zh, mono.tobytes())
    fetch(ALI_BASE + "/rttm/far/R8002_M8002_MS802.rttm", DATA / "ali-R8002.rttm")


def ami_rows() -> list[dict]:
    rows = []
    with zipfile.ZipFile(ORIGINAL / "sources/ami/ami_public_manual_1.6.2.zip") as archive:
        meeting = next(
            m
            for m in ET.fromstring(archive.read("corpusResources/meetings.xml"))
            if m.attrib.get("observation") == "ES2004d"
        )
        for speaker in meeting:
            agent = speaker.attrib["nxt_agent"]
            words = list(ET.fromstring(archive.read(f"words/ES2004d.{agent}.words.xml")))
            indices = {w.attrib[NS + "id"]: i for i, w in enumerate(words)}
            for segment in ET.fromstring(archive.read(f"segments/ES2004d.{agent}.segments.xml")):
                nodes = []
                for child in segment:
                    ids = re.findall(r"id\(([^)]+)\)", child.attrib["href"])
                    nodes.extend(words[indices[ids[0]] : indices[ids[-1]] + 1])
                lexical = [w for w in nodes if w.tag == "w"]
                text = " ".join(w.text or "" for w in lexical)
                available = bool(re.search(r"[A-Za-z0-9]", text)) and all(
                    w.tag == "w" and w.attrib.get("trunc") != "true" for w in nodes
                )
                timed_words = [
                    dict(
                        start=float(w.attrib["starttime"]),
                        end=float(w.attrib["endtime"]),
                        text=w.text or "",
                        id=w.attrib[NS + "id"],
                    )
                    for w in lexical
                    if "starttime" in w.attrib
                    and "endtime" in w.attrib
                    and float(w.attrib["endtime"]) > float(w.attrib["starttime"])
                ]
                rows.append(
                    dict(
                        id=segment.attrib[NS + "id"],
                        start=float(segment.attrib["transcriber_start"]),
                        end=float(segment.attrib["transcriber_end"]),
                        speaker=speaker.attrib["global_name"],
                        text=text,
                        reference_available=available,
                        words=timed_words,
                    )
                )
    return sorted(rows, key=lambda row: (row["start"], row["end"], row["id"]))


def ja_rows() -> list[dict]:
    document = ET.parse(ORIGINAL / "sources/ingcrowd/japanese-themed-dialogue01.eaf").getroot()
    times = {
        s.attrib["TIME_SLOT_ID"]: int(s.attrib["TIME_VALUE"]) / 1000
        for s in document.findall("./TIME_ORDER/TIME_SLOT")
    }
    rows = []
    for tier in document.findall("TIER"):
        for node in tier.findall("./ANNOTATION/ALIGNABLE_ANNOTATION"):
            text = node.findtext("ANNOTATION_VALUE", "")
            if text.strip():
                rows.append(
                    dict(
                        id=node.attrib["ANNOTATION_ID"],
                        start=times[node.attrib["TIME_SLOT_REF1"]],
                        end=times[node.attrib["TIME_SLOT_REF2"]],
                        speaker=tier.attrib["TIER_ID"],
                        text=text,
                        reference_available="<" not in text and ">" not in text,
                    )
                )
    return sorted(rows, key=lambda row: (row["start"], row["end"], row["id"]))


def ko_rows() -> list[dict]:
    with tarfile.open(DATA / "seoul-label.tgz") as archive:
        member = next(m for m in archive.getmembers() if m.name.endswith(".zip"))
        with zipfile.ZipFile(io.BytesIO(archive.extractfile(member).read())) as labels:
            name = next(
                n
                for n in labels.namelist()
                if not n.startswith("__MACOSX/") and Path(n).name == "s13m26m1.TextGrid"
            )
            data = labels.read(name)
    (DATA / "s13m26m1.TextGrid").write_bytes(data)
    text = data.decode("utf-16" if data[:2] in (b"\xfe\xff", b"\xff\xfe") else "utf-8-sig")
    rows = []
    for block in re.finditer(r"item \[(\d+)\]:(.*?)(?=\n\s*item \[\d+\]:|\Z)", text, re.S):
        if re.search(r'name = "([^"]*)"', block[2])[1] != "utt.ortho.":
            continue
        for interval in re.finditer(
            r'intervals \[(\d+)\]:\s*xmin = ([\d.eE+-]+)\s*xmax = ([\d.eE+-]+)\s*text = "((?:[^"]|"")*)"',
            block[2],
        ):
            value = interval[4].replace('""', '"')
            if not value.strip() or not (
                "<IVER>" in value
                or re.search(r"[\uac00-\ud7a3]", value)
                or value in ("<UNKNOWN>", "<PRIVATE.INFO>")
            ):
                continue
            rows.append(
                dict(
                    id="s13m26m1-" + interval[1],
                    start=float(interval[2]),
                    end=float(interval[3]),
                    speaker="interviewer-male" if "<IVER>" in value else "s13m26",
                    text=value,
                    reference_available="<" not in value,
                    uncertainty="Single utterance tier: unannotated overlap/crosstalk remains possible; interviewer text unavailable.",
                )
            )
    return rows


def zh_rows() -> list[dict]:
    rows = []
    for i, line in enumerate((DATA / "ali-R8002.rttm").read_text(encoding="utf-8").splitlines()):
        fields = line.split()
        if fields and fields[0] == "SPEAKER":
            start, duration = float(fields[3]), float(fields[4])
            rows.append(
                dict(
                    id=f"R8002-rttm-{i}",
                    start=start,
                    end=start + duration,
                    speaker=fields[7],
                    text="",
                    reference_available=False,
                    uncertainty="Original speaker RTTM timing only; no lexical reference acquired.",
                )
            )
    return rows


def vad_probabilities(pcm: bytes) -> np.ndarray:
    sys.path.insert(0, str(ROOT / "src"))
    from puripuly_heart.core.vad.silero import SileroVadOnnx

    detector = SileroVadOnnx(ROOT / "src/puripuly_heart/data/vad/silero_vad.onnx")
    samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768
    return np.array(
        [
            detector.speech_probability(samples[index : index + 512], sample_rate_hz=RATE)
            for index in range(0, len(samples) - 511, 512)
        ]
    )


def source_for(language: str, offset: float) -> dict:
    if language in ("en", "ja"):
        prior = json.loads(
            (BENCHMARK / f"inputs/replay-{language}.json").read_text(encoding="utf-8")
        )["source"]
        if language == "en":
            url = "https://huggingface.co/datasets/FluidInference/ami-corpus-mirror"
            audio = ORIGINAL / "sources/ami/ES2004d.Mix-Headset.wav"
        else:
            url = "https://huggingface.co/datasets/IngCrowd/japanese-themed-dialogue"
            audio = ORIGINAL / "sources/ingcrowd/japanese-themed-dialogue01_audio.wav"
        annotation = ORIGINAL / prior["annotation_path"]
        return dict(
            url=url,
            revision=prior["revision"],
            license="CC-BY-4.0" if language == "en" else "CC-BY-NC-4.0",
            offset_s=offset,
            notes=prior["channel_policy"] + " Noncommercial evaluation; no audio redistribution.",
            attribution=prior["attribution"],
            original_audio_sha256=digest(audio),
            annotation_sha256=digest(annotation),
            annotation_revision=prior.get("annotation_revision", prior["revision"]),
            original_audio_path=relative(audio),
            annotation_path=relative(annotation),
        )
    if language == "ko":
        return dict(
            url="https://www.openslr.org/113",
            revision="OpenSLR113 archives; content SHA256 pins below",
            license="CC-BY-NC-2.0",
            offset_s=offset,
            notes="Seoul s13m26m1 spontaneous two-person interview. Original mono timeline; ffmpeg PCM16 conversion/resampling only. No silence modification. Single-tier annotation cannot independently establish no overlap. Noncommercial evaluation only; audio not redistributed.",
            attribution="Yun et al. (2015), The Korean Corpus of Spontaneous Speech; NRF-2012S1A5A2A03034027",
            archive_url="https://openslr.elda.org/resources/113/sound.tgz",
            original_audio_path=relative(DATA / "s13m26m1.flac"),
            original_audio_sha256=digest(DATA / "s13m26m1.flac"),
            annotation_path=relative(DATA / "s13m26m1.TextGrid"),
            annotation_sha256=digest(DATA / "s13m26m1.TextGrid"),
            label_archive_sha256=digest(DATA / "seoul-label.tgz"),
            requested_historical_offset_s=66.58811791383219,
            offset_rounding_s=offset - 66.58811791383219,
        )
    acquisition = json.loads((DATA / "ali-acquisition.json").read_text(encoding="utf-8"))
    return dict(
        url=acquisition["url"],
        revision=ALI_REV,
        license="CC-BY-SA-4.0",
        offset_s=offset,
        notes="AliMeeting Test R8002_M8002_MS802 original eight-channel far-field WAV, fixed zero-based channel 0 throughout [0,120). Official OpenSLR119 CC-BY-SA-4.0 governs corpus, not mirror card CC-BY-NC claim. No unknown mono mirror conversion used. No silence edits; audio not redistributed.",
        attribution="Yu et al., AliMeeting / Alibaba Speech Lab",
        authoritative_source="https://www.openslr.org/119",
        acquisition=acquisition,
        annotation_path=relative(DATA / "ali-R8002.rttm"),
        annotation_sha256=digest(DATA / "ali-R8002.rttm"),
        source_hash_scope="Range-checked source bytes and exact fixed-channel extraction; full 529029200-byte original not downloaded or hashed.",
    )


def make_record(
    identifier: str, language: str, rows: list[dict], path: Path, offset: float, diagnostic: bool
) -> dict:
    pcm = read_wav(path)
    probabilities = vad_probabilities(pcm)
    audio = np.frombuffer(pcm, dtype="<i2").astype(np.float64) / 32768
    spans = []
    for row in rows:
        if row["start"] >= offset + 120 or row["end"] <= offset:
            continue
        truncated = row["start"] < offset or row["end"] > offset + 120
        spans.append(
            dict(
                id=identifier + "-" + row["id"],
                start_s=round(max(0, row["start"] - offset), 6),
                end_s=round(min(120, row["end"] - offset), 6),
                speaker=row["speaker"],
                text=row["text"],
                reference_available=row["reference_available"] and not truncated,
                uncertainty=row.get(
                    "uncertainty",
                    "Human corpus annotation; not listened or independently acoustically adjudicated.",
                )
                + (" Window-truncated lexical reference excluded." if truncated else ""),
                source_annotation_id=row["id"],
                source_start_s=row["start"],
                source_end_s=row["end"],
                boundary_truncated=truncated,
            )
        )
    overlaps = []
    for i, left in enumerate(spans):
        for right in spans[i + 1 :]:
            a, b = max(left["start_s"], right["start_s"]), min(left["end_s"], right["end_s"])
            if left["speaker"] != right["speaker"] and b > a:
                overlaps.append(
                    dict(
                        id=f"{identifier}-overlap-{len(overlaps):02d}",
                        start_s=a,
                        end_s=b,
                        speakers=sorted([left["speaker"], right["speaker"]]),
                        span_ids=[left["id"], right["id"]],
                        reference_method="Human annotation intersection; not acoustically listened.",
                    )
                )
    candidates = []
    for span in spans:
        row = next(r for r in rows if r["id"] == span["source_annotation_id"])
        points = []
        if language == "en":
            for word in row["words"]:
                if word["end"] - word["start"] >= 0.08:
                    points.append(((word["start"] + word["end"]) / 2 - offset, word["text"]))
        else:
            points = [
                (float(t), None)
                for t in np.arange(span["start_s"] + 0.12, span["end_s"] - 0.12, 0.064)
            ]
        for t, word in points:
            if not span["start_s"] + 0.06 < t < span["end_s"] - 0.06:
                continue
            if any(o["start_s"] - 0.08 <= t <= o["end_s"] + 0.08 for o in overlaps):
                continue
            frame = min(len(probabilities) - 1, int(t * RATE // 512))
            probability = float(probabilities[frame])
            samples = audio[max(0, round((t - 0.032) * RATE)) : round((t + 0.032) * RATE)]
            rms = math.sqrt(float(np.mean(samples**2)))
            if probability < 0.6 or rms < 10 ** (-55 / 20):
                continue
            candidates.append(
                dict(
                    time_s=round(t, 6),
                    span_id=span["id"],
                    speaker=span["speaker"],
                    span_end_s=span["end_s"],
                    vad_probability=round(probability, 6),
                    rms_dbfs=round(20 * math.log10(rms), 3),
                    reference_word=word,
                )
            )
    selected = []
    by_span = {s["id"]: [p for p in candidates if p["span_id"] == s["id"]] for s in spans}
    for _ in range(20):
        if len(selected) == 20:
            break
        for span in spans:
            if len(selected) == 20:
                break
            choices = [
                p
                for p in by_span[span["id"]]
                if all(abs(p["time_s"] - q["time_s"]) >= 1.2 for q in selected)
            ]
            if choices:
                selected.append(
                    max(choices, key=lambda p: (p["vad_probability"], p["rms_dbfs"], -p["time_s"]))
                )
    if len(selected) != 20:
        raise ValueError(
            f"{identifier}: only {len(selected)} conservative voiced anchors available"
        )
    selected.sort(key=lambda p: p["time_s"])
    anchors = [dict(id=f"{identifier}-anchor-{i:02d}", **point) for i, point in enumerate(selected)]
    changes = []
    for previous, current in zip(spans, spans[1:]):
        if previous["speaker"] != current["speaker"]:
            changes.append(
                dict(
                    id=f"{identifier}-change-{len(changes):02d}",
                    time_s=current["start_s"],
                    from_speaker=previous["speaker"],
                    to_speaker=current["speaker"],
                    kind=(
                        "overlap_onset"
                        if previous["end_s"] > current["start_s"]
                        else "speaker_change"
                    ),
                    notes="Annotation-derived onset; preceding span in onset order, not a silence guarantee.",
                    source_span_id=previous["id"],
                    target_span_id=current["id"],
                    gap_s=round(current["start_s"] - previous["end_s"], 6),
                )
            )
    if len(changes) > 4:
        indices = np.linspace(0, len(changes) - 1, 4).round().astype(int)
        changes = [changes[int(i)] for i in indices]
    return dict(
        id=identifier,
        language=language,
        diagnostic=diagnostic,
        audio_path=relative(path),
        sha256=digest(path),
        sample_rate=RATE,
        duration_s=120.0,
        source=source_for(language, offset),
        spans=spans,
        anchors=anchors,
        overlaps=overlaps,
        changes=changes,
        turn_opportunities=[],
        signal_check=dict(
            method="Existing checkout SileroVadOnnx, independent of candidate outputs; 512-sample frames, speech_probability >=0.6 and 64ms RMS >=-55dBFS; anchor spacing >=1.2s and >=80ms outside annotated overlap.",
            vad_model_sha256=digest(ROOT / "src/puripuly_heart/data/vad/silero_vad.onnx"),
            audible_review=False,
            limitations="VAD/energy supports speech presence, not lexical truth, speaker identity or absence of unannotated crosstalk. AMI anchors additionally lie inside original corpus word intervals.",
        ),
    )


def diagnostic_opportunities(records: list[dict]) -> None:
    specs = {
        "ja-diagnostic": [
            (
                "a103",
                "a104",
                "unsafe",
                "その is an incomplete determiner; the following お茶 continues the same noun phrase.",
            ),
            (
                "a104",
                "a105",
                "unsafe",
                "お茶 is followed by particle が; noun phrase is continuing.",
            ),
            (
                "a105",
                "a106",
                "unsafe",
                "が requires the following predicate, despite the 850ms annotated pause.",
            ),
            (
                "a107",
                "a108",
                "unsafe",
                "緑茶なんだけど何だろう introduces the following explanation of drinking drops.",
            ),
            (
                "a109",
                "a110",
                "unsafe",
                "なんかそれは is an unfinished demonstrative/topic and not an ending.",
            ),
            (
                "a110",
                "a111",
                "intermediate",
                "私結婚してるんだけど supplies a complete background clause before the husband/shared-interest continuation; not conversational completion.",
            ),
            (
                "a112",
                None,
                "ending",
                "The complete thought ends with 定期的には行ってるかもしれない; the partner then starts a new response.",
            ),
            (
                "a205",
                "a206",
                "unsafe",
                "色んな味とか飲み方で is incomplete: で leads into 楽しみ方が変わる.",
            ),
            (
                "a206",
                None,
                "ending",
                "ハマる人はハマりそうだよね。そういうの concludes the partner response; the other speaker agrees.",
            ),
            (
                "a119",
                "a120",
                "unsafe",
                "何だろう is hesitation within the explanation of repeatedly visiting, followed by its reason.",
            ),
            (
                "a123",
                "a124",
                "unsafe",
                "そこ、も is an incomplete subject/topic; the predicate follows after 930ms.",
            ),
            (
                "a124",
                None,
                "ending",
                "最近のお気に入り。楽しーみな場所かもしれない。お出掛け先としては concludes the visit explanation before the partner response.",
            ),
        ],
        "en-diagnostic": [
            (
                "ES2004d.sync.702",
                None,
                "ending",
                "The quoted response example is complete: it will shout back I am here or something.",
            ),
            (
                "ES2004d.sync.360",
                None,
                "ending",
                "The complete joking answer I am under the sofa can stand alone; overlap is scored separately.",
            ),
            (
                "ES2004d.sync.362",
                None,
                "ending",
                "Or that would be too complicated is a complete objection, not a continuing fragment.",
            ),
            (
                "ES2004d.sync.366",
                None,
                "ending",
                "Like here is a complete deictic answer in the placement discussion.",
            ),
            (
                "ES2004d.sync.708",
                None,
                "ending",
                "Yeah yeah something like that completes an agreement response.",
            ),
            (
                "ES2004d.sync.372",
                None,
                "ending",
                "Yeah that would make sense is a complete short agreement, with concurrent speech retained.",
            ),
        ],
    }
    for record in records:
        for left_id, right_id, kind, notes in specs.get(record["id"], []):
            left = next(s for s in record["spans"] if s["source_annotation_id"] == left_id)
            right = next(
                (s for s in record["spans"] if s["source_annotation_id"] == right_id), None
            )
            end = right["start_s"] if right else left["end_s"]
            semantic_reference = (
                "Independent human EAF transcript"
                if record["language"] == "ja"
                else "AMI human manual1.6.2 transcript"
            )
            record["turn_opportunities"].append(
                dict(
                    id=f"{record['id']}-turn-{len(record['turn_opportunities']):02d}",
                    time_s=left["end_s"],
                    kind=kind,
                    start_s=left["end_s"],
                    end_s=max(left["end_s"], end),
                    notes=notes
                    + " Text-annotation hypothesis only; not listened. Acoustic/prosodic uncertainty excluded from semantic claims.",
                    reference_text=left["text"] + (" | " + right["text"] if right else ""),
                    span_ids=[left["id"]] + ([right["id"]] if right else []),
                    semantic_reference=semantic_reference + "; text-only adjudication",
                    semantic_uncertainty="Prosody and actual acoustic end not independently verified.",
                )
            )
        if record["id"] == "en-diagnostic":
            clauses = [
                (
                    284.76,
                    284.76,
                    ["ES2004d.C.words450", "ES2004d.C.words452"],
                    "intermediate",
                    "You can have a remote that tells you exactly where it is .",
                    "Complete proposal sentence inside the 55s reference span; a useful intermediate translation boundary, not an end of the speaker floor.",
                ),
                (
                    291.74,
                    291.74,
                    ["ES2004d.C.words474", "ES2004d.C.words476"],
                    "intermediate",
                    "Um the standard , there's be a standard transmission with the T_V_ using using all the standard chips that we've talked about .",
                    "Complete transmission-design clause within continued long speech; abbreviations are not code switching.",
                ),
                (
                    304.62,
                    304.62,
                    ["ES2004d.C.words512", "ES2004d.C.words515"],
                    "intermediate",
                    "it would need two separate P_C_B_s , so it would be joined at the hinge through some sort of cabling .",
                    "Complete hardware explanation within continued speech; preserves the number two.",
                ),
                (
                    316.57,
                    316.57,
                    ["ES2004d.C.words542", "ES2004d.C.words544"],
                    "intermediate",
                    "we'd just be using using the same thing to transmit the data .",
                    "Complete explanation before the next unfinished noun phrase, inside a greater-than-10-second span.",
                ),
                (
                    318.36,
                    319.88,
                    ["ES2004d.C.words546", "ES2004d.C.words547"],
                    "unsafe",
                    "And the infrared | um sender would be on the on one of the bot the bottom layer",
                    "Infrared is the modifier of sender across the original word-timestamp gap; cutting here leaves an incomplete subject.",
                ),
                (
                    358.14,
                    359.30,
                    ["ES2004d.C.words654", "ES2004d.C.words655"],
                    "unsafe",
                    "touch screen which yeah would | uh have all of all of the the available functions",
                    "Relative clause which would still requires have and its complement after the original word-timestamp gap.",
                ),
            ]
            for source_start, source_end, word_ids, kind, text, notes in clauses:
                span = next(
                    s
                    for s in record["spans"]
                    if s["source_start_s"] <= source_start <= s["source_end_s"]
                    and s["speaker"] == "MEE014"
                )
                span_ids = [span["id"]]
                if kind == "unsafe":
                    following = min(
                        (
                            s
                            for s in record["spans"]
                            if s["source_start_s"] > span["source_end_s"]
                            and s["speaker"] == span["speaker"]
                        ),
                        key=lambda s: s["source_start_s"],
                    )
                    span_ids.append(following["id"])
                local_start = round(source_start - record["source"]["offset_s"], 6)
                local_end = round(source_end - record["source"]["offset_s"], 6)
                record["turn_opportunities"].append(
                    dict(
                        id=f"en-diagnostic-turn-{len(record['turn_opportunities']):02d}",
                        time_s=local_start,
                        kind=kind,
                        start_s=local_start,
                        end_s=local_end,
                        notes=notes
                        + " Original manual word endpoints used exactly; no interpolation or invented tolerance. Zero-width clauses support nearest-boundary offsets only, not binary success.",
                        reference_text=text,
                        span_ids=span_ids,
                        reference_word_ids=word_ids,
                        boundary_interval_method="Exact adjacent original AMI word end/start timestamps",
                        binary_boundary_eligible=local_end > local_start,
                        semantic_reference="AMI human manual1.6.2 transcript; text-only clause hypothesis",
                        semantic_uncertainty="Word times may be forced-aligned; actual acoustic pause and prosody not independently verified.",
                    )
                )


def coverage_summary(records: list[dict]) -> dict:
    returns, short, negations, numbers, near_gapless, long_speech, overlap = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for record in records:
        spans = record["spans"]
        for first, middle, last in zip(spans, spans[1:], spans[2:]):
            if first["speaker"] == last["speaker"] != middle["speaker"]:
                returns.append(
                    dict(
                        recording=record["id"],
                        span_ids=[s["id"] for s in (first, middle, last)],
                        method="Annotation onset-order A-B-A; backchannel overlap retained, not a verified floor switch.",
                    )
                )
        for previous, current in zip(spans, spans[1:]):
            gap = round(current["start_s"] - previous["end_s"], 6)
            if previous["speaker"] != current["speaker"] and 0 <= gap <= 0.1:
                near_gapless.append(
                    dict(
                        recording=record["id"], span_ids=[previous["id"], current["id"]], gap_s=gap
                    )
                )
        for span in spans:
            if span["end_s"] - span["start_s"] > 10:
                long_speech.append(
                    dict(
                        recording=record["id"],
                        span_id=span["id"],
                        duration_s=round(span["end_s"] - span["start_s"], 6),
                        limitation="Continuous annotated turn, not a guarantee of uninterrupted acoustic voicing.",
                    )
                )
            if span["reference_available"]:
                example = dict(recording=record["id"], span_id=span["id"], text=span["text"])
                if span["end_s"] - span["start_s"] <= 1.2:
                    short.append(example)
                if re.search(
                    r"\b(?:no|not|never|haven't|won't|don't)\b|ない|なく|아니|않",
                    span["text"],
                    re.I,
                ):
                    negations.append(example)
                if re.search(
                    r"\b(?:one|two|three|four|five|ten)\b|\d|二人|一|二|三|두|하나",
                    span["text"],
                    re.I,
                ):
                    numbers.append(example)
        overlap.extend(dict(recording=record["id"], overlap_id=o["id"]) for o in record["overlaps"])
    pauses = [
        dict(recording=r["id"], opportunity_id=o["id"], gap_s=round(o["end_s"] - o["start_s"], 6))
        for r in records
        for o in r["turn_opportunities"]
        if o["kind"] == "unsafe" and 0.3 <= o["end_s"] - o["start_s"] <= 1.2
    ]
    intermediate = [
        dict(recording=r["id"], opportunity_id=o["id"])
        for r in records
        for o in r["turn_opportunities"]
        if o["kind"] == "intermediate"
    ]

    def supported(examples: list[dict], limit: str) -> dict:
        return dict(
            status="annotation_supported" if examples else "unavailable",
            count=len(examples),
            examples=examples[:5],
            uncertainty=limit,
        )

    return dict(
        a_b_a_returns=supported(
            returns,
            "Speaker labels/onset order from human references; overlapping backchannels may not be conversational floor changes.",
        ),
        short_replies=supported(
            short,
            "Duration and lexical references support short utterances, not independently listened reply intent.",
        ),
        negations=supported(
            negations, "Human transcript lexical match; no candidate-derived words."
        ),
        numbers=supported(
            numbers,
            "Human transcript number/quantifier mentions; not an acoustic numeric-accuracy oracle.",
        ),
        continuing_pauses_0_3_to_1_2_s=supported(
            pauses,
            "Text establishes syntactic continuation; interval gap is annotation-derived, not independently measured silence or prosody.",
        ),
        near_gapless_changes=supported(
            near_gapless, "0-100ms annotation gaps; acoustic boundary timing unverified."
        ),
        overlap_backchannels=supported(
            overlap,
            "Cross-speaker human annotation intersection; no listened acoustic verification; KO overlap unknown. Short acknowledgement references are retained.",
        ),
        code_switching=dict(
            status="unavailable_unverified",
            count=0,
            notes="No genuine phrase/clause language switch established from references. JA loanwords and EN TV/PCB/LCD acronyms are not counted.",
        ),
        speech_over_10_s=supported(
            long_speech, "Long original annotated spans retain all within-span silence and overlap."
        ),
        useful_intermediate_boundaries=dict(
            **supported(
                intermediate,
                "Complete text-derived clauses/background statements; usefulness is a pre-run hypothesis, not heard prosodic ground truth.",
            ),
            binary_eligible_intervals=sum(
                o["kind"] == "intermediate" and o["end_s"] > o["start_s"]
                for r in records
                for o in r["turn_opportunities"]
            ),
            offset_only_points=sum(
                o["kind"] == "intermediate" and o["end_s"] == o["start_s"]
                for r in records
                for o in r["turn_opportunities"]
            ),
        ),
        semantic_exclusions=[
            "Tagged/unknown lexical spans are not used to invent semantic endings.",
            "KO interviewer and ZH timing-only references have no turn-policy semantic labels.",
            "No claim of verified code switching or measured within-word acoustic pauses.",
        ],
    )


def freeze() -> None:
    if MANIFEST.exists():
        raise FileExistsError(f"Refusing to replace frozen manifest: {MANIFEST}")
    english, japanese = ami_rows(), ja_rows()
    for language, offset in [("en", 259.0), ("ja", 312.0)]:
        original = ORIGINAL / (
            "sources/ami/ES2004d.Mix-Headset.wav"
            if language == "en"
            else "sources/ingcrowd/japanese-themed-dialogue01_audio.wav"
        )
        write_wav(DATA / f"{language}-diagnostic-120.wav", read_wav(original, offset))
    records = [
        make_record("en", "en", english, ORIGINAL / "en/replay-clean.wav", 32.368, False),
        make_record("ja", "ja", japanese, ORIGINAL / "ja/replay-clean.wav", 86.04, False),
        make_record("ko", "ko", ko_rows(), DATA / "ko-120.wav", KO_OFFSET, False),
        make_record("zh", "zh", zh_rows(), DATA / "zh-120.wav", 0, False),
        make_record("en-diagnostic", "en", english, DATA / "en-diagnostic-120.wav", 259, True),
        make_record("ja-diagnostic", "ja", japanese, DATA / "ja-diagnostic-120.wav", 312, True),
    ]
    diagnostic_opportunities(records)
    for record in records:
        record["speaker_reference_coverage"] = [
            dict(
                speaker=speaker,
                attribution_anchor_count=sum(a["speaker"] == speaker for a in record["anchors"]),
                lexical_reference_span_count=sum(
                    s["speaker"] == speaker and s["reference_available"] for s in record["spans"]
                ),
            )
            for speaker in sorted({s["speaker"] for s in record["spans"]})
        ]
        if record["language"] == "ko":
            interviewer_anchors = sum(a["speaker"] == "interviewer-male" for a in record["anchors"])
            record["ko_reference_limit"] = dict(
                interviewer_attribution_anchors=interviewer_anchors,
                lexical_reference_for_interviewer=False,
                attribution_reference="Original IVER speaker/time annotations plus independent voice checks; lexical unavailability does not invalidate the speaker/time reference.",
                coverage_bias=(
                    "Interviewer attribution anchors present; overlap remains unverified."
                    if interviewer_anchors
                    else "No interviewer anchor met the automatic eligibility checks; KO attribution is biased toward the interviewee."
                ),
                overlap_reference="Unavailable: one annotation tier cannot represent concurrent speakers.",
            )
    manifest = dict(
        schema_version=1,
        reference_method="Automatically selected human corpus annotations plus independent existing Silero VAD/energy checks. No human/audio listening; user explicitly authorized automatic selection instead. No candidate model/provider output inspected or used. Semantics are transcript-derived hypotheses only; acoustic/prosodic ambiguity is not ground truth.",
        frozen_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        recordings=records,
        total_duration_s=720,
        distribution="Audio remains local/ignored; publish source pins, licenses and annotation references only. Corpus text retains its source license, not the repository code license.",
        known_limits=[
            "No listening was performed.",
            "KO single-tier references cannot establish absence of unannotated overlap; interviewer has no lexical reference.",
            "ZH has independent speaker RTTM but no lexical transcript in this freeze.",
            "No verified code-switching case; acronyms/loanwords alone are not counted as code switching.",
            "Turn opportunities are annotation-derived semantic hypotheses, not acoustically adjudicated endings.",
        ],
    )
    manifest["coverage_summary"] = coverage_summary(records)
    manifest["diagnostic_coverage_summary"] = coverage_summary(
        [r for r in records if r["diagnostic"]]
    )
    manifest["human_annotation_provenance"] = dict(
        en="AMI manual1.6.2 original human segments/words; word times may be forced-aligned. No new listening.",
        ja="IngCrowd original manually transcribed/aligned EAF; no upstream ASR per publisher. No new listening.",
        ko="Seoul original human utt.ortho. TextGrid. Interviewer IVER has timing/speaker only; single tier cannot encode simultaneous speakers.",
        zh="AliMeeting original speaker RTTM from pinned mirror. Lexical references unavailable in this freeze.",
    )
    validate(manifest)
    manifest["machine_checks"] = dict(
        wav_headers="passed: six mono 16000Hz PCM16 WAVs, exactly 1920000 frames each",
        hashes="passed: prepared WAV SHA256 matches manifest; original source/annotation pins recorded",
        mapping="passed: EN/JA sample-exact contiguous original crops; ZH exact fixed original channel 0; KO exact declared ffmpeg decode/resample/crop",
        manifest_invariants="passed: unique IDs, 2-4 reference speakers, bounded spans/opportunities, each anchor inside exactly one annotated span outside every annotated overlap",
        speech_presence="passed during anchor construction: existing Silero >=0.6 and 64ms RMS >=-55dBFS at all 120 anchors",
        not_established=[
            "Heard lexical or speaker accuracy",
            "Absence of unannotated overlap/crosstalk",
            "Prosodic turn endings",
            "Genuine code switching",
        ],
    )
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            dict(
                manifest=relative(MANIFEST),
                sha256=digest(MANIFEST),
                recordings=len(records),
                duration_s=720,
                anchors=sum(len(r["anchors"]) for r in records),
                changes=sum(len(r["changes"]) for r in records),
                turn_opportunities=sum(len(r["turn_opportunities"]) for r in records),
                validation="WAV/hash/source mapping and manifest invariants passed",
            ),
            ensure_ascii=False,
        )
    )


def validate(manifest: dict) -> None:
    ids = set()
    assert manifest["schema_version"] == 1 and len(manifest["recordings"]) == 6
    for record in manifest["recordings"]:
        pcm = read_wav(ROOT / record["audio_path"])
        with wave.open(str(ROOT / record["audio_path"]), "rb") as stream:
            assert stream.getnframes() == RATE * 120 and stream.getcomptype() == "NONE"
        assert (
            len(pcm) == RATE * 120 * 2 and digest(ROOT / record["audio_path"]) == record["sha256"]
        )
        assert 2 <= len({s["speaker"] for s in record["spans"]}) <= 4
        spans = {s["id"]: s for s in record["spans"]}
        for span in spans.values():
            assert 0 <= span["start_s"] < span["end_s"] <= 120
        for group in ("spans", "anchors", "overlaps", "changes", "turn_opportunities"):
            for item in record[group]:
                assert item["id"] not in ids
                ids.add(item["id"])
        for anchor in record["anchors"]:
            span = spans[anchor["span_id"]]
            assert (
                span["speaker"] == anchor["speaker"]
                and span["start_s"] < anchor["time_s"] < span["end_s"]
            )
            assert (
                anchor["span_end_s"] == span["end_s"]
                and anchor["vad_probability"] >= 0.6
                and anchor["rms_dbfs"] >= -55
            )
            assert not any(
                o["start_s"] <= anchor["time_s"] <= o["end_s"] for o in record["overlaps"]
            )
            assert sum(s["start_s"] <= anchor["time_s"] <= s["end_s"] for s in record["spans"]) == 1
        for overlap in record["overlaps"]:
            left, right = (spans[identifier] for identifier in overlap["span_ids"])
            assert overlap["start_s"] == max(left["start_s"], right["start_s"])
            assert overlap["end_s"] == min(left["end_s"], right["end_s"])
            assert overlap["start_s"] < overlap["end_s"] and len(set(overlap["speakers"])) == 2
        for change in record["changes"]:
            source, target = spans[change["source_span_id"]], spans[change["target_span_id"]]
            assert (
                source["speaker"]
                == change["from_speaker"]
                != change["to_speaker"]
                == target["speaker"]
            )
            assert change["time_s"] == target["start_s"]
        for opportunity in record["turn_opportunities"]:
            assert opportunity["kind"] in ("ending", "intermediate", "unsafe")
            assert (
                0 <= opportunity["start_s"] <= opportunity["time_s"] <= opportunity["end_s"] <= 120
            )
            assert all(identifier in spans for identifier in opportunity["span_ids"])
        if record["language"] in ("en", "ja"):
            source = ROOT / record["source"]["original_audio_path"]
            assert pcm == read_wav(source, record["source"]["offset_s"])
        elif record["language"] == "zh":
            with wave.open(io.BytesIO((DATA / "ali-source-range.bin").read_bytes())) as stream:
                assert (stream.getframerate(), stream.getnchannels(), stream.getsampwidth()) == (
                    RATE,
                    8,
                    2,
                )
                source_pcm = np.frombuffer(stream.readframes(RATE * 120), dtype="<i2").reshape(
                    -1, 8
                )
            assert pcm == source_pcm[:, 0].copy().tobytes()
        else:
            source_pcm = subprocess.run(
                [
                    "ffmpeg",
                    "-nostdin",
                    "-v",
                    "error",
                    "-i",
                    str(ROOT / record["source"]["original_audio_path"]),
                    "-ss",
                    str(record["source"]["offset_s"]),
                    "-t",
                    "120",
                    "-ar",
                    str(RATE),
                    "-ac",
                    "1",
                    "-c:a",
                    "pcm_s16le",
                    "-f",
                    "s16le",
                    "pipe:1",
                ],
                capture_output=True,
                check=True,
            ).stdout
            assert pcm == source_pcm
    assert sum(len(r["anchors"]) for r in manifest["recordings"]) == 120
    assert sum(len(r["turn_opportunities"]) for r in manifest["recordings"]) == 24


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Acquire/freeze #441 annotation-derived inputs, without listening or ASR outputs."
    )
    parser.add_argument("action", choices=("acquire", "freeze", "validate", "inspect"))
    args = parser.parse_args()
    if args.action == "acquire":
        acquire()
    elif args.action == "freeze":
        freeze()
    elif args.action == "validate":
        validate(json.loads(MANIFEST.read_text(encoding="utf-8")))
        print(json.dumps(dict(manifest=relative(MANIFEST), sha256=digest(MANIFEST), valid=True)))
    else:
        for row in ami_rows():
            if 259 <= row["start"] < 379:
                print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
