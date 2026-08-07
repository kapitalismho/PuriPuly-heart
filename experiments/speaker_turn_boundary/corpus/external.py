from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import tarfile
import urllib.error
import urllib.request
import wave
import zipfile
from pathlib import Path

import numpy as np

from experiments.speaker_turn_boundary.config import CANONICAL_SAMPLE_RATE_HZ

DEFAULT_CORPUS_ROOT = (
    Path(os.environ.get("TEMP", str(Path.home() / "tmp"))) / "opencode" / "stb_phase2_corpora"
)
CORPUS_ROOT_ENV = "STB_PHASE2_CORPORA_ROOT"


class CorpusError(RuntimeError):
    pass


class ToolUnavailableError(CorpusError):
    pass


def corpus_root() -> Path:
    configured = os.environ.get(CORPUS_ROOT_ENV)
    if configured:
        return Path(configured).expanduser().resolve()
    return DEFAULT_CORPUS_ROOT


def archive_root() -> Path:
    return corpus_root() / "archives"


def derived_root() -> Path:
    return corpus_root() / "derived"


def phase2_build_root() -> Path:
    return corpus_root() / "phase2_build"


def sha256_file(path: Path, chunk_bytes: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def md5_file(path: Path, chunk_bytes: int = 1 << 20) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def download_file(
    url: str,
    destination: Path,
    *,
    expected_md5: str | None = None,
    expected_sha256: str | None = None,
    timeout_seconds: int = 60,
) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        if expected_md5 is not None and md5_file(destination) == expected_md5:
            return destination
        if expected_sha256 is not None and sha256_file(destination) == expected_sha256:
            return destination
        if expected_md5 is None and expected_sha256 is None:
            return destination
        destination.unlink()
    existing_size = 0
    mode = "wb"
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "stb-phase2-benchmark/1.0"},
    )
    if destination.is_file():
        existing_size = destination.stat().st_size
        mode = "ab"
        request.add_header("Range", f"bytes={existing_size}-")
    try:
        response = urllib.request.urlopen(request, timeout=timeout_seconds)
    except urllib.error.HTTPError as exc:
        if exc.code == 416 and existing_size > 0:
            response = urllib.request.urlopen(
                urllib.request.Request(
                    url,
                    headers={"User-Agent": "stb-phase2-benchmark/1.0"},
                ),
                timeout=timeout_seconds,
            )
            existing_size = 0
            mode = "wb"
        else:
            raise
    with response, destination.open(mode) as handle:
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            handle.write(chunk)
    if expected_md5 is not None and md5_file(destination) != expected_md5:
        raise CorpusError(f"md5 mismatch for {destination.name}")
    if expected_sha256 is not None and sha256_file(destination) != expected_sha256:
        raise CorpusError(f"sha256 mismatch for {destination.name}")
    return destination


def extract_tar_gz(archive: Path, target_dir: Path) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as handle:
        handle.extractall(target_dir, filter="data")
    return target_dir


def extract_zip(archive: Path, target_dir: Path) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as handle:
        handle.extractall(target_dir)
    return target_dir


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def run_ffmpeg(args: list[str], *, check: bool = True) -> subprocess.CompletedProcess:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise ToolUnavailableError("ffmpeg not found on PATH")
    result = subprocess.run(
        [ffmpeg, "-hide_banner", "-loglevel", "error", *args], capture_output=True
    )
    if check and result.returncode != 0:
        raise CorpusError(
            f"ffmpeg failed ({result.returncode}): {result.stderr.decode('utf-8', 'replace')[-500:]}"
        )
    return result


def ffprobe_duration_seconds(path: Path) -> float:
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        raise ToolUnavailableError("ffprobe not found on PATH")
    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True,
    )
    if result.returncode != 0:
        raise CorpusError(
            f"ffprobe failed on {path.name}: {result.stderr.decode('utf-8', 'replace')[-300:]}"
        )
    return float(result.stdout.decode("utf-8", "replace").strip())


def decode_flac_to_pcm16(path: Path, sample_rate_hz: int = CANONICAL_SAMPLE_RATE_HZ) -> np.ndarray:
    result = run_ffmpeg(
        [
            "-i",
            str(path),
            "-f",
            "s16le",
            "-acodec",
            "pcm_s16le",
            "-ar",
            str(sample_rate_hz),
            "-ac",
            "1",
            "pipe:1",
        ]
    )
    pcm = np.frombuffer(result.stdout, dtype=np.int16)
    return pcm.astype(np.float32) / 32768.0


def encode_opus_to_pcm16(
    samples: np.ndarray,
    *,
    bitrate_kbps: int,
    sample_rate_hz: int = CANONICAL_SAMPLE_RATE_HZ,
) -> np.ndarray:
    pcm16 = np.clip(samples, -1.0, 1.0)
    pcm16 = np.round(pcm16 * 32767.0).astype(np.int16)
    encode = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "s16le",
            "-ar",
            str(sample_rate_hz),
            "-ac",
            "1",
            "-i",
            "pipe:0",
            "-c:a",
            "libopus",
            "-b:a",
            f"{bitrate_kbps}k",
            "-ar",
            str(sample_rate_hz),
            "-ac",
            "1",
            "-f",
            "ogg",
            "pipe:1",
        ],
        input=pcm16.tobytes(),
        capture_output=True,
    )
    if encode.returncode != 0:
        raise CorpusError(f"opus encode failed: {encode.stderr.decode('utf-8', 'replace')[-500:]}")
    decode = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            "pipe:0",
            "-f",
            "s16le",
            "-acodec",
            "pcm_s16le",
            "-ar",
            str(sample_rate_hz),
            "-ac",
            "1",
            "pipe:1",
        ],
        input=encode.stdout,
        capture_output=True,
    )
    if decode.returncode != 0:
        raise CorpusError(f"opus decode failed: {decode.stderr.decode('utf-8', 'replace')[-500:]}")
    pcm = np.frombuffer(decode.stdout, dtype=np.int16)
    return pcm.astype(np.float32) / 32768.0


def write_pcm16_wav(path: Path, samples: np.ndarray, *, sample_rate_hz: int) -> None:
    scaled = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
    pcm = np.round(scaled * 32767.0).astype(np.int16)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate_hz)
        wav_file.writeframes(pcm.tobytes())
