from __future__ import annotations

import io
import json
import struct
import tarfile
import wave
from collections.abc import Iterator
from pathlib import Path

import httpx
import numpy as np

SAMPLE_RATE = 16000
TARGET_S = 120.0
GAP_S = 0.15
EDGE_S = 0.08
TARGET_RMS = 0.1
BASE_URL = "https://huggingface.co/datasets/google/fleurs/resolve/main/data/{config}"


class _IterStream(io.RawIOBase):
    def __init__(self, chunks: Iterator[bytes]) -> None:
        self._chunks = chunks
        self._buffer = b""

    def readable(self) -> bool:
        return True

    def readinto(self, buffer) -> int:
        while not self._buffer:
            try:
                self._buffer = next(self._chunks)
            except StopIteration:
                return 0
        size = min(len(buffer), len(self._buffer))
        buffer[:size] = self._buffer[:size]
        self._buffer = self._buffer[size:]
        return size


def _decode_wav(data: bytes) -> np.ndarray:
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError("FLEURS audio is not a RIFF/WAVE file")
    fmt = None
    payload = None
    offset = 12
    while offset + 8 <= len(data):
        chunk_id = data[offset : offset + 4]
        size = struct.unpack_from("<I", data, offset + 4)[0]
        body = data[offset + 8 : offset + 8 + size]
        if chunk_id == b"fmt ":
            fmt = struct.unpack_from("<HHIIHH", body)
        elif chunk_id == b"data":
            payload = body
        offset += 8 + size + (size & 1)
    if fmt is None or payload is None:
        raise ValueError("FLEURS audio is missing fmt or data chunks")
    tag, channels, rate, _, _, bits = fmt
    if rate != SAMPLE_RATE:
        raise ValueError(f"unsupported FLEURS sample rate {rate}")
    if tag == 1 and bits == 16:
        pcm = np.frombuffer(payload, dtype="<i2").astype(np.float32) / 32768.0
    elif tag == 3 and bits == 32:
        pcm = np.frombuffer(payload, dtype="<f4").astype(np.float32)
    else:
        raise ValueError(f"unsupported FLEURS WAV format tag={tag} bits={bits}")
    if channels > 1:
        pcm = pcm[: len(pcm) // channels * channels].reshape(-1, channels).mean(axis=1)
    return pcm


def _frame_rms(pcm: np.ndarray) -> np.ndarray:
    frame = int(0.02 * SAMPLE_RATE)
    count = len(pcm) // frame
    return np.sqrt(np.mean(pcm[: count * frame].reshape(count, frame) ** 2, axis=1))


def _level(pcm: np.ndarray) -> np.ndarray:
    energy = _frame_rms(pcm)
    if energy.size == 0 or float(np.percentile(energy, 95)) <= 0:
        return pcm
    pcm = pcm * (TARGET_RMS / float(np.percentile(energy, 95)))
    peak = float(np.abs(pcm).max())
    return pcm * (0.95 / peak) if peak > 0.95 else pcm


def _trim(pcm: np.ndarray) -> np.ndarray:
    frame = int(0.02 * SAMPLE_RATE)
    energy = _frame_rms(pcm)
    if energy.size == 0:
        return pcm
    threshold = 0.08 * float(np.percentile(energy, 95))
    voiced = np.nonzero(energy > threshold)[0]
    if voiced.size == 0:
        return pcm
    edge = int(EDGE_S * SAMPLE_RATE)
    start = max(0, int(voiced[0]) * frame - edge)
    end = min(len(pcm), (int(voiced[-1]) + 1) * frame + edge)
    return pcm[start:end]


def _write_wav(path: Path, pcm: np.ndarray) -> None:
    samples = np.clip(np.round(pcm * 32767.0), -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(SAMPLE_RATE)
        writer.writeframes(samples.tobytes())


def _transcripts(client: httpx.Client, config: str) -> dict[str, tuple[int, str]]:
    response = client.get(f"{BASE_URL.format(config=config)}/test.tsv")
    response.raise_for_status()
    rows = {}
    for line in response.text.splitlines():
        columns = line.split("\t")
        if len(columns) >= 3 and columns[0].isdigit():
            rows[columns[1]] = (int(columns[0]), columns[2])
    return rows


def build(config: str, data_dir: Path) -> None:
    seen: set[int] = set()
    pieces: list[np.ndarray] = []
    sentences = []
    cursor = 0.0
    with httpx.Client(timeout=120, follow_redirects=True) as client:
        transcripts = _transcripts(client, config)
        url = f"{BASE_URL.format(config=config)}/audio/test.tar.gz"
        with client.stream("GET", url) as response:
            response.raise_for_status()
            stream = io.BufferedReader(_IterStream(response.iter_bytes()), buffer_size=1 << 16)
            with tarfile.open(fileobj=stream, mode="r|gz") as archive:
                for member in archive:
                    name = member.name.rsplit("/", 1)[-1]
                    if not member.isfile() or name not in transcripts:
                        continue
                    sentence_id, text = transcripts[name]
                    if sentence_id in seen:
                        continue
                    seen.add(sentence_id)
                    pcm = _trim(_level(_decode_wav(archive.extractfile(member).read())))
                    start = cursor
                    end = start + len(pcm) / SAMPLE_RATE
                    pieces.append(pcm)
                    pieces.append(np.zeros(int(GAP_S * SAMPLE_RATE), dtype=np.float32))
                    sentences.append(
                        {
                            "id": sentence_id,
                            "file": name,
                            "start_s": round(start, 3),
                            "end_s": round(end, 3),
                            "text": text,
                        }
                    )
                    cursor = end + GAP_S
                    if cursor >= TARGET_S:
                        break
    audio = np.concatenate(pieces)
    data_dir.mkdir(parents=True, exist_ok=True)
    _write_wav(data_dir / f"{config}.wav", audio)
    reference = {
        "config": config,
        "source": f"{BASE_URL.format(config=config)}/audio/test.tar.gz",
        "license": "CC-BY-4.0 (google/fleurs)",
        "sample_rate": SAMPLE_RATE,
        "gap_s": GAP_S,
        "duration_s": round(len(audio) / SAMPLE_RATE, 3),
        "sentences": sentences,
    }
    (data_dir / f"{config}.json").write_text(
        json.dumps(reference, ensure_ascii=False, indent=1), encoding="utf-8"
    )


def ensure(configs: list[str], data_dir: Path) -> None:
    for config in configs:
        if (data_dir / f"{config}.wav").exists() and (data_dir / f"{config}.json").exists():
            continue
        print(f"[data] building {config} from FLEURS test split", flush=True)
        build(config, data_dir)


def load(config: str, data_dir: Path) -> tuple[dict, bytes]:
    reference = json.loads((data_dir / f"{config}.json").read_text(encoding="utf-8"))
    with wave.open(str(data_dir / f"{config}.wav")) as reader:
        pcm = reader.readframes(reader.getnframes())
    return reference, pcm
