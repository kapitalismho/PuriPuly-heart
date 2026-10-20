from __future__ import annotations

import html
import io
import json
import re
import struct
import tarfile
from pathlib import Path

import fleurs
import httpx
import numpy as np

DATA_VERSION = 1
SAMPLE_RATE = fleurs.SAMPLE_RATE
BASE_URL = "https://huggingface.co/datasets/espnet/yodas2/resolve/main/data/{subset}"
SOURCES = {
    "ko_conv": {"subset": "ko000", "shard": 12, "audio_id": "3BPeG47L7Hc", "start_s": 0.0},
    "ja_conv": {"subset": "ja000", "shard": 3, "audio_id": "19cQStzeIgE", "start_s": 0.0},
    "en_conv": {"subset": "en000", "shard": 4, "audio_id": "06inwHyT7l0", "start_s": 22.0},
}
DURATION_S = 125.0
LABELS = [
    re.compile(r"(?<![A-Za-z])[wｗ]{2,}(?![A-Za-z])"),
    re.compile(r"\[[^\]]*\]"),
    re.compile(r"[（(][^）)]{0,12}[）)]"),
    re.compile(r"(?<![^\s。」！？!?、])[^\s）)。、「」！？!?]{1,10}[）)]"),
    re.compile(r"(^|\s)[A-Z][a-z]+ [A-Z][a-z]+:\s"),
    re.compile(r"(^|\s)(-|–|>>)\s*"),
]


def clean_caption(text: str) -> str:
    text = html.unescape(text)
    for pattern in LABELS:
        text = pattern.sub(" ", text)
    return " ".join(text.split())


def _captions(client: httpx.Client, source: dict) -> list[tuple[float, float, str]]:
    url = f"{BASE_URL.format(subset=source['subset'])}/text/{source['shard']:08d}.json"
    response = client.get(url)
    response.raise_for_status()
    item = next(i for i in response.json() if i["audio_id"] == source["audio_id"])
    captions = []
    for key, text in item["text"].items():
        start, end = (int(part) / 100 for part in key.split("-")[-2:])
        captions.append((start, end, text))
    return sorted(captions)


def _read_exact(stream, size: int) -> bytes:
    chunks, remaining = [], size
    while remaining > 0:
        chunk = stream.read(min(remaining, 1 << 20))
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _read_window(stream, start_s: float, duration_s: float) -> tuple[np.ndarray, int]:
    header = _read_exact(stream, 12)
    if header[:4] != b"RIFF" or header[8:12] != b"WAVE":
        raise ValueError("YODAS2 audio is not a RIFF/WAVE file")
    fmt = None
    while True:
        chunk_id, size = struct.unpack("<4sI", _read_exact(stream, 8))
        if chunk_id == b"data":
            break
        body = _read_exact(stream, size + (size & 1))
        if chunk_id == b"fmt ":
            fmt = struct.unpack_from("<HHIIHH", body)
    if fmt is None or fmt[0] != 1 or fmt[5] != 16:
        raise ValueError(f"unsupported YODAS2 WAV format {fmt}")
    channels, rate = fmt[1], fmt[2]
    frame = 2 * channels
    _read_exact(stream, int(start_s * rate) * frame)
    data = _read_exact(stream, int(duration_s * rate) * frame)
    pcm = np.frombuffer(data[: len(data) // frame * frame], dtype="<i2").astype(np.float32)
    pcm = pcm.reshape(-1, channels).mean(axis=1) / 32768.0
    return pcm, rate


def _resample(pcm: np.ndarray, rate: int) -> np.ndarray:
    if rate == SAMPLE_RATE:
        return pcm
    up, down = SAMPLE_RATE, rate
    divisor = np.gcd(up, down)
    up, down = up // divisor, down // divisor
    half = 32 * max(up, down)
    n = np.arange(-half, half + 1)
    cutoff = 0.45 / max(up, down)
    taps = 2 * cutoff * np.sinc(2 * cutoff * n) * np.kaiser(len(n), 8.0) * up
    stretched = np.zeros(len(pcm) * up, dtype=np.float32)
    stretched[::up] = pcm
    return np.convolve(stretched, taps, mode="same")[::down].astype(np.float32)


def build(name: str, data_dir: Path) -> None:
    source = SOURCES[name]
    start_s = source["start_s"]
    with httpx.Client(timeout=120, follow_redirects=True) as client:
        captions = _captions(client, source)
        url = f"{BASE_URL.format(subset=source['subset'])}/audio/{source['shard']:08d}.tar.gz"
        with client.stream("GET", url) as response:
            response.raise_for_status()
            stream = io.BufferedReader(
                fleurs._IterStream(response.iter_bytes()), buffer_size=1 << 16
            )
            with tarfile.open(fileobj=stream, mode="r|gz") as archive:
                member = next(
                    m for m in archive if m.name.rsplit("/", 1)[-1] == f"{source['audio_id']}.wav"
                )
                pcm, rate = _read_window(archive.extractfile(member), start_s, DURATION_S)
    audio = fleurs._level(_resample(pcm, rate))
    duration = len(audio) / SAMPLE_RATE
    sentences = []
    for start, end, text in captions:
        text = clean_caption(text)
        if not text or start < start_s or start - start_s >= duration:
            continue
        sentences.append(
            {
                "start_s": round(start - start_s, 3),
                "end_s": round(min(end - start_s, duration), 3),
                "text": text,
            }
        )
    data_dir.mkdir(parents=True, exist_ok=True)
    fleurs._write_wav(data_dir / f"{name}.wav", audio)
    reference = {
        "version": DATA_VERSION,
        "config": name,
        "source": f"{url}#{source['audio_id']}",
        "license": "CC-BY-3.0 (espnet/yodas2)",
        "audio_id": source["audio_id"],
        "window_s": [start_s, round(start_s + duration, 3)],
        "unit": "caption",
        "sample_rate": SAMPLE_RATE,
        "duration_s": round(duration, 3),
        "sentences": sentences,
    }
    (data_dir / f"{name}.json").write_text(
        json.dumps(reference, ensure_ascii=False, indent=1), encoding="utf-8"
    )


def ensure(names: list[str], data_dir: Path) -> None:
    for name in names:
        meta = data_dir / f"{name}.json"
        if (data_dir / f"{name}.wav").exists() and meta.exists():
            if json.loads(meta.read_text(encoding="utf-8")).get("version") == DATA_VERSION:
                continue
        print(f"[data] building {name} from YODAS2 {SOURCES[name]['subset']}", flush=True)
        build(name, data_dir)
