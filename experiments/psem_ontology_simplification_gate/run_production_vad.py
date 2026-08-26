from __future__ import annotations

import argparse
import wave
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort

from experiments.psem_relative_occupancy_gate.io_utils import (
    canonical_sha256,
    load_json,
    load_jsonl,
    sha256_file,
    write_json,
    write_jsonl,
)
from puripuly_heart.core.vad.bundled import (
    SILERO_VAD_RESOURCE_SHA256,
    SILERO_VAD_VERSION,
    bundled_silero_vad_onnx_path,
)
from puripuly_heart.core.vad.gating import (
    PEER_MAX_SEGMENT_MS,
    PEER_VAD_START_COMMIT_CHUNKS,
    PEER_VAD_START_DEBOUNCE_CHUNKS,
    SpeechChunk,
    SpeechEnd,
    SpeechStart,
    create_peer_vad_gating,
)
from puripuly_heart.core.vad.silero import SileroVadOnnx

PACKAGE_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = PACKAGE_ROOT.parents[1]
CONFIG_PATH = PACKAGE_ROOT / "config.json"
PREDECESSOR_ROOT = REPOSITORY_ROOT / "experiments" / "psem_relative_occupancy_gate"
MODEL_PATH = REPOSITORY_ROOT / "src" / "puripuly_heart" / "data" / "vad" / "silero_vad.onnx"
SAMPLE_RATE_HZ = 16000


class ProductionVadReplayError(RuntimeError):
    pass


def _validate_profile(vad_cfg: dict[str, Any]) -> None:
    expected_profile = {
        "profile": "peer",
        "backend": "CPUExecutionProvider",
        "start_debounce_chunks": PEER_VAD_START_DEBOUNCE_CHUNKS,
        "start_commit_chunks": PEER_VAD_START_COMMIT_CHUNKS,
        "max_segment_ms": PEER_MAX_SEGMENT_MS,
        "chunk_samples": 512,
        "source_support": "pre_roll_plus_committed_chunks_through_speech_end_excluding_trailing_hangover",
    }
    for field, expected in expected_profile.items():
        if vad_cfg.get(field) != expected:
            raise ProductionVadReplayError(f"production VAD profile mismatch: {field}")


def _merge_spans(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    for start, end in sorted(spans):
        if end <= start:
            continue
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(result[-1][1], end))
        else:
            result.append((start, end))
    return result


def _replay_source(row: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    vad_cfg = cfg["speech_gate"]["production_vad"]
    audio_path = Path(str(row["audio_path"]))
    if not audio_path.is_file():
        raise ProductionVadReplayError(f"audio is unavailable: {audio_path}")
    actual_sha256 = sha256_file(audio_path)
    if actual_sha256 != row["waveform_sha256"]:
        raise ProductionVadReplayError(f"audio digest mismatch: {audio_path}")
    audio_size_bytes = audio_path.stat().st_size
    if audio_size_bytes != int(row["waveform_size_bytes"]):
        raise ProductionVadReplayError(f"audio size mismatch: {audio_path}")
    engine = SileroVadOnnx(MODEL_PATH)
    gating = create_peer_vad_gating(
        engine,
        sample_rate_hz=SAMPLE_RATE_HZ,
        ring_buffer_ms=int(vad_cfg["pre_roll_ms"]),
        speech_threshold=float(vad_cfg["speech_threshold"]),
        hangover_ms=int(vad_cfg["hangover_ms"]),
    )
    chunk_samples = int(vad_cfg["chunk_samples"])
    spans: list[tuple[int, int]] = []
    active_start: int | None = None
    processed_samples = 0
    with wave.open(str(audio_path), "rb") as reader:
        if (
            reader.getframerate() != SAMPLE_RATE_HZ
            or reader.getnchannels() != 1
            or reader.getsampwidth() != 2
        ):
            raise ProductionVadReplayError(f"audio contract mismatch: {audio_path}")
        total_samples = reader.getnframes()
        if total_samples != int(row["source_duration_samples"]):
            raise ProductionVadReplayError(f"audio duration mismatch: {audio_path}")
        while True:
            payload = reader.readframes(chunk_samples)
            if not payload:
                break
            original_samples = len(payload) // 2
            chunk = np.frombuffer(payload, dtype="<i2").astype(np.float32)
            if original_samples < chunk_samples:
                chunk = np.pad(chunk, (0, chunk_samples - original_samples))
            chunk /= 32768.0
            events = gating.process_chunk(chunk)
            chunk_start = processed_samples
            chunk_end = min(total_samples, chunk_start + original_samples)
            for event in events:
                if isinstance(event, SpeechStart):
                    buffered_count = 1 + sum(isinstance(value, SpeechChunk) for value in events)
                    active_start = max(
                        0,
                        chunk_start
                        - (buffered_count - 1) * chunk_samples
                        - int(np.asarray(event.pre_roll).size),
                    )
                elif isinstance(event, SpeechEnd) and active_start is not None:
                    if event.reason == "silence":
                        trailing_samples = int(
                            round(event.trailing_silence_ms * SAMPLE_RATE_HZ / 1000.0)
                        )
                        trailing_samples = max(
                            0, trailing_samples - (chunk_samples - original_samples)
                        )
                        speech_end = max(active_start, chunk_end - trailing_samples)
                    else:
                        speech_end = chunk_end
                    spans.append((min(active_start, total_samples), min(speech_end, total_samples)))
                    active_start = None
            processed_samples = chunk_end
    if active_start is not None:
        spans.append((active_start, processed_samples))
    spans = _merge_spans(spans)
    return {
        "schema_version": "psem.ontology_simplification.production_vad_source.v1",
        "source_id": str(row["source_id"]),
        "audio_path": str(audio_path),
        "audio_sha256": actual_sha256,
        "audio_size_bytes": audio_size_bytes,
        "scored_start_sample": int(row["scored_start_sample"]),
        "scored_end_sample": int(row["scored_end_sample"]),
        "audio_length_samples": int(total_samples),
        "processed_samples": processed_samples,
        "ignored_tail_samples": int(total_samples) - processed_samples,
        "speech_spans": [{"start_sample": start, "end_sample": end} for start, end in spans],
        "speech_span_count": len(spans),
        "speech_seconds": sum(end - start for start, end in spans) / SAMPLE_RATE_HZ,
    }


def run(role: str) -> None:
    cfg = load_json(CONFIG_PATH)
    vad_cfg = cfg["speech_gate"]["production_vad"]
    if SILERO_VAD_VERSION != vad_cfg["model_version"]:
        raise ProductionVadReplayError("Silero version binding mismatch")
    if SILERO_VAD_RESOURCE_SHA256 != vad_cfg["model_sha256"]:
        raise ProductionVadReplayError("Silero declared digest binding mismatch")
    if sha256_file(MODEL_PATH) != vad_cfg["model_sha256"]:
        raise ProductionVadReplayError("Silero model digest mismatch")
    _validate_profile(vad_cfg)
    bundled_path = Path(str(bundled_silero_vad_onnx_path()))
    if bundled_path.resolve() != MODEL_PATH.resolve():
        raise ProductionVadReplayError("bundled Silero path differs from the pinned model")
    manifest_path = PREDECESSOR_ROOT / "results" / role / "relative_occupancy_manifest.jsonl"
    manifest = load_jsonl(manifest_path)
    source_ids = [str(value["source_id"]) for value in manifest]
    if len(source_ids) != len(set(source_ids)):
        raise ProductionVadReplayError("duplicate source in production VAD manifest")
    rows = [_replay_source(row, cfg) for row in manifest]
    output_dir = PACKAGE_ROOT / "results" / role
    gate_path = output_dir / "production_vad_speech_gate.jsonl"
    write_jsonl(gate_path, rows)
    receipt = {
        "schema_version": "psem.ontology_simplification.production_vad_receipt.v1",
        "role": "PSEM-STRATEGY-DEV" if role == "dev" else "PSEM-STRATEGY-EVAL",
        "config_sha256": sha256_file(CONFIG_PATH),
        "manifest_path": str(manifest_path.relative_to(REPOSITORY_ROOT)),
        "manifest_sha256": sha256_file(manifest_path),
        "source_count": len(rows),
        "source_ids_sha256": canonical_sha256([value["source_id"] for value in rows]),
        "speech_gate_path": str(gate_path.relative_to(REPOSITORY_ROOT)),
        "speech_gate_sha256": sha256_file(gate_path),
        "model": {
            "version": SILERO_VAD_VERSION,
            "path": str(MODEL_PATH.relative_to(REPOSITORY_ROOT)),
            "sha256": sha256_file(MODEL_PATH),
            "backend": "CPUExecutionProvider",
            "onnxruntime_version": ort.__version__,
        },
        "runner_source_sha256": sha256_file(Path(__file__)),
        "profile": vad_cfg,
        "total_audio_seconds": sum(int(value["audio_length_samples"]) for value in rows)
        / SAMPLE_RATE_HZ,
        "total_vad_speech_seconds": sum(float(value["speech_seconds"]) for value in rows),
        "row_payload_sha256": canonical_sha256(rows),
    }
    write_json(output_dir / "production_vad_replay_receipt.json", receipt)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=("dev", "eval"), required=True)
    args = parser.parse_args()
    run(args.role)


if __name__ == "__main__":
    main()
