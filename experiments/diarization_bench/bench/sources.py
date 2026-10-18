from __future__ import annotations

import base64
import hashlib
import os
import subprocess
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

SAMPLE_RATE_HZ = 16_000
CACHE = Path(
    os.environ.get(
        "DIARIZATION_BENCH_CACHE", Path.home() / ".cache" / "puripuly-heart" / "diarization-bench"
    )
)

LIBRISPEECH_REPO = "JinchuanTian/librispeech-test-clean-plain-tts"
LIBRISPEECH_REVISION = "4ce3c2c3808c492a9b0761f7cd704b74e7c494f7"
AMI_REPO = "ggfox00000/dia-AMICorpus-all"
AMI_REVISION = "a249db53de253a3a0a864c93a51d613159dfec7c"
PUBLIC_CORPUS = Path(
    os.environ.get(
        "DIARIZATION_BENCH_PUBLIC_CORPUS",
        Path.home() / ".cache" / "puripuly-heart" / "sim-v02-public-corpus-600-v1",
    )
)
AMI_LOCAL_AUDIO = Path(
    os.environ.get(
        "DIARIZATION_BENCH_AMI_AUDIO",
        Path(__file__).resolve().parents[4]
        / "soniox_diar_test"
        / "experiments"
        / "ami_benchmark"
        / "source_audio",
    )
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, target: Path) -> Path:
    if target.is_file():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as response, partial.open("wb") as out:
        while chunk := response.read(1 << 20):
            out.write(chunk)
    partial.replace(target)
    return target


def decode(path: Path) -> np.ndarray:
    result = subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-loglevel",
            "error",
            "-i",
            str(path),
            "-ac",
            "1",
            "-ar",
            str(SAMPLE_RATE_HZ),
            "-f",
            "s16le",
            "-",
        ],
        check=True,
        capture_output=True,
    )
    return np.frombuffer(result.stdout, dtype="<i2").astype(np.float32) / 32768.0


def librispeech(utt_id: str) -> tuple[np.ndarray, dict]:
    speaker, chapter, _ = utt_id.split("-")
    relative = f"audio/{speaker}/{chapter}/{utt_id}.flac"
    url = f"https://huggingface.co/datasets/{LIBRISPEECH_REPO}/resolve/{LIBRISPEECH_REVISION}/{relative}"
    path = _download(url, CACHE / "librispeech" / relative)
    return decode(path), {
        "dataset": "LibriSpeech test-clean",
        "license": "CC BY 4.0",
        "url": url,
        "sha256": sha256_file(path),
    }


def librispeech_metadata() -> dict[str, dict]:
    import json

    url = f"https://huggingface.co/datasets/{LIBRISPEECH_REPO}/resolve/{LIBRISPEECH_REVISION}/dialogues.jsonl"
    path = _download(url, CACHE / "librispeech" / f"dialogues-{LIBRISPEECH_REVISION[:12]}.jsonl")
    rows = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            meta = json.loads(line)["metadata"]
            rows[meta["utt_id"]] = meta
    return rows


def common_voice(relative_path: str) -> tuple[np.ndarray, dict]:
    path = PUBLIC_CORPUS / relative_path
    if not path.is_file():
        raise FileNotFoundError(f"public corpus clip missing: {path}")
    return decode(path), {
        "dataset": "Common Voice 17.0",
        "license": "CC0-1.0",
        "relative_path": relative_path,
        "sha256": sha256_file(path),
    }


def ami_audio(meeting: str) -> tuple[Path, dict]:
    name = f"{meeting}.Mix-Headset.wav"
    local = AMI_LOCAL_AUDIO / name
    if local.is_file():
        path = local
    else:
        url = f"https://huggingface.co/datasets/{AMI_REPO}/resolve/{AMI_REVISION}/amicorpus/{meeting}/audio/{name}"
        path = _download(url, CACHE / "ami" / name)
    return path, {
        "dataset": "AMI Meeting Corpus (Mix-Headset)",
        "license": "CC BY 4.0",
        "mirror": f"{AMI_REPO}@{AMI_REVISION}",
        "sha256": sha256_file(path),
    }


def ami_window(meeting: str, start_s: float, end_s: float) -> tuple[np.ndarray, dict]:
    path, info = ami_audio(meeting)
    audio = decode(path)
    lo, hi = int(round(start_s * SAMPLE_RATE_HZ)), int(round(end_s * SAMPLE_RATE_HZ))
    if hi > audio.size:
        raise ValueError(f"AMI window exceeds {meeting}")
    return audio[lo:hi].copy(), {**info, "window_s": [start_s, end_s]}


def ami_words(meeting: str) -> tuple[list[dict], dict]:
    words: list[dict] = []
    files = {}
    for participant in "ABCDE":
        name = f"{meeting}.{participant}.words.xml"
        url = (
            f"https://huggingface.co/datasets/{AMI_REPO}/resolve/{AMI_REVISION}"
            f"/ami_public_manual_1.6.2/words/{name}"
        )
        try:
            path = _download(url, CACHE / "ami" / "words" / name)
        except Exception:
            continue
        files[name] = sha256_file(path)
        root = ET.parse(path).getroot()
        for element in root:
            tag = element.tag.split("}")[-1]
            if tag != "w" or element.get("punc") == "true":
                continue
            start, end = element.get("starttime"), element.get("endtime")
            if start is None or end is None or not (element.text or "").strip():
                continue
            words.append(
                {
                    "speaker": participant,
                    "start": float(start),
                    "end": float(end),
                    "text": element.text.strip(),
                }
            )
    words.sort(key=lambda w: (w["start"], w["end"], w["speaker"]))
    return words, {
        "annotation": "AMI manual annotations v1.6.2 words",
        "license": "CC BY 4.0",
        "mirror": f"{AMI_REPO}@{AMI_REVISION}",
        "files": files,
    }


def tts(text: str, voice: str) -> tuple[np.ndarray, dict]:
    key = hashlib.sha256(f"{voice}|{text}".encode()).hexdigest()[:16]
    path = CACHE / "tts" / f"{key}.wav"
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        literal = str(path).replace("'", "''")
        script = (
            "Add-Type -AssemblyName System.Speech; "
            "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            f"try {{ $s.SelectVoice('{voice}'); $s.SetOutputToWaveFile('{literal}'); "
            f"$s.Speak('{text.replace(chr(39), chr(39) * 2)}') }} finally {{ $s.Dispose() }}"
        )
        encoded = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
        subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded],
            check=True,
            capture_output=True,
            timeout=60,
        )
    return decode(path), {
        "dataset": "Windows SAPI synthetic speech",
        "voice": voice,
        "text": text,
        "sha256": sha256_file(path),
        "synthetic_voice": True,
    }
