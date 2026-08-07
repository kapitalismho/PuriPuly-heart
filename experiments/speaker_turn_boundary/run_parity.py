from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from experiments.speaker_turn_boundary.adapters.eres2netv2 import kaldi_fbank_numpy
from experiments.speaker_turn_boundary.config import EXPERIMENT_RESULTS_DIR
from experiments.speaker_turn_boundary.frontend import (
    Resampler16k8k,
    StreamingLSEENDFrontend,
    frontend_profile,
    model_frame_count_offline,
    output_frame_available_16k_count,
    output_frame_center_16k,
    output_frame_lookback_16k,
    stream_whole_file,
)
from experiments.speaker_turn_boundary.metadata import collect_runtime_metadata
from experiments.speaker_turn_boundary.provenance import (
    LS_EEND_VARIANTS,
    all_artifacts,
    verify_artifact_file,
)
from experiments.speaker_turn_boundary.schemas import canonical_json, sha256_hex
from experiments.speaker_turn_boundary.vad_baseline import load_canonical_wav

GOLDEN_CASE_IDS = ("golden_two_utterance_gap400", "golden_single_utterance", "golden_silence")

MODELSCOPE_EXAMPLE_WAVS = {
    "speaker1_a_cn_16k.wav": "5f20ce0ddc378ca3239d3ce864b1142726a46a1221ae553912e4e142045df58b",
    "speaker1_b_cn_16k.wav": "20745dc08a4281894d146140b99b9ef7417ac681119b7f7202f553cdf1a85f65",
    "speaker2_a_cn_16k.wav": "8a6cffa452df32ef10503f7992f22ffcdd7f16c4e0273d13311bc5cdcb13abf4",
}


def load_parity_audio(data_dir: Path) -> dict[str, np.ndarray]:
    audio: dict[str, np.ndarray] = {}
    for case_id in GOLDEN_CASE_IDS:
        wav_path = data_dir / "generated" / f"{case_id}.wav"
        if wav_path.is_file():
            audio[case_id] = load_canonical_wav(wav_path)
    return audio


def load_modelscope_examples(cache_dir: Path) -> dict[str, np.ndarray]:
    audio: dict[str, np.ndarray] = {}
    for file_name, expected_hash in MODELSCOPE_EXAMPLE_WAVS.items():
        wav_path = cache_dir / file_name
        if wav_path.is_file() and _sha256_of(wav_path) == expected_hash:
            audio[file_name] = load_canonical_wav(wav_path)
    return audio


def _sha256_of(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare_chunked_vs_whole(audio: dict[str, np.ndarray], chunk_samples: int) -> dict[str, object]:
    results: dict[str, object] = {}
    for name, samples in audio.items():
        whole = stream_whole_file(samples)
        chunked = _run_chunked(samples, chunk_samples)
        aligned = min(whole.shape[0], chunked.shape[0])
        if aligned == 0:
            results[name] = {
                "whole_frames": int(whole.shape[0]),
                "chunked_frames": int(chunked.shape[0]),
                "max_abs_error": None,
                "frames_equal": None,
            }
            continue
        max_error = float(np.abs(whole[:aligned] - chunked[:aligned]).max())
        results[name] = {
            "whole_frames": int(whole.shape[0]),
            "chunked_frames": int(chunked.shape[0]),
            "max_abs_error": max_error,
            "frames_equal": max_error == 0.0,
        }
    return results


def _run_chunked(samples: np.ndarray, chunk_samples: int) -> np.ndarray:
    resampler = Resampler16k8k()
    frontend = StreamingLSEENDFrontend()
    frames: list[np.ndarray] = []
    offset = 0
    while offset < samples.size:
        resampled = resampler.push(samples[offset : offset + chunk_samples])
        emitted = frontend.push_audio(resampled)
        if emitted.size:
            frames.append(emitted)
        offset += chunk_samples
    tail = frontend.finalize()
    if tail.size:
        frames.append(tail)
    if not frames:
        return np.zeros((0, frontend_padded_dim()), dtype=np.float32)
    return np.concatenate(frames, axis=0)


def frontend_padded_dim() -> int:
    from experiments.speaker_turn_boundary.frontend import LS_EEND_MODEL_INPUT_DIM

    return LS_EEND_MODEL_INPUT_DIM


def run_frontend_checks(audio: dict[str, np.ndarray], chunk_samples: int) -> dict[str, object]:
    frames_from_offline: dict[str, int] = {}
    for name, samples in audio.items():
        resampler = Resampler16k8k()
        resampled = resampler.push(samples)
        frames_from_offline[name] = model_frame_count_offline(resampled.size)
    return {
        "offline_model_frame_counts": frames_from_offline,
        "mapping": {
            "output_frame_center_16k": {
                "0": output_frame_center_16k(0),
                "9": output_frame_center_16k(9),
            },
            "output_frame_available_16k_count": {
                "0": output_frame_available_16k_count(0),
                "9": output_frame_available_16k_count(9),
            },
            "lookback_16k": output_frame_lookback_16k(),
            "initial_latency_ms": round(output_frame_available_16k_count(0) / 16.0, 3),
        },
    }


def verify_local_artifacts(
    hf_root: Path, ckpt_root: Path, eres_std_root: Path, eres_w24_root: Path
) -> dict[str, object]:
    report: dict[str, object] = {}
    for artifact in all_artifacts():
        if artifact.kind == "onnx_sidecar":
            continue
        if artifact.kind == "onnx_step_model":
            variant_dir = artifact.artifact_id.split(":")[0]
            info = LS_EEND_VARIANTS[variant_dir]
            path = hf_root / info["dir"] / artifact.file_name
        elif artifact.artifact_id.startswith("FS-EEND:"):
            path = ckpt_root / artifact.file_name
        elif artifact.artifact_id.startswith("E-standard:"):
            path = eres_std_root / artifact.file_name
        else:
            path = eres_w24_root / artifact.file_name
        ok, reason = verify_artifact_file(artifact, path)
        report[artifact.artifact_id] = {"verified": ok, "reason": reason, "path": str(path)}
    return report


def checkpoints_local(model_root: Path) -> dict[str, dict[str, object]]:
    found: dict[str, dict[str, object]] = {}
    for variant, info in LS_EEND_VARIANTS.items():
        onnx_path = model_root / info["dir"] / info["onnx"]
        sidecar_path = model_root / info["dir"] / info["sidecar"]
        found[variant] = {
            "onnx_path": str(onnx_path),
            "onnx_present": onnx_path.is_file(),
            "sidecar_present": sidecar_path.is_file(),
        }
    return found


def run_parity(
    *,
    data_dir: Path,
    hf_root: Path,
    ckpt_root: Path,
    eres_std_root: Path,
    eres_w24_root: Path,
    cache_dir: Path,
    out_dir: Path,
    chunk_samples: int,
) -> dict[str, object]:
    audio = load_parity_audio(data_dir)
    audio.update(load_modelscope_examples(cache_dir))
    started_at = datetime.now(timezone.utc).isoformat()
    records: dict[str, object] = {}
    records["resampler_determinism"] = _resampler_determinism_check()
    chunked_vs_whole = compare_chunked_vs_whole(audio, chunk_samples)
    records["chunked_vs_whole_file_frontend"] = chunked_vs_whole
    records["frontend_checks"] = run_frontend_checks(audio, chunk_samples)
    records["frontend_profile"] = frontend_profile().to_dict()
    records["artifacts"] = verify_local_artifacts(hf_root, ckpt_root, eres_std_root, eres_w24_root)
    records["checkpoints"] = checkpoints_local(hf_root)
    records["fbank_smoke"] = _fbank_smoke(audio)
    finished_at = datetime.now(timezone.utc).isoformat()
    payload = {
        "parity_id": "phase1_frontend_parity",
        "started_at_utc": started_at,
        "finished_at_utc": finished_at,
        "runtime_metadata": collect_runtime_metadata(),
        "records": records,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "parity_frontend.json"
    path.write_text(canonical_json(payload), encoding="utf-8")
    print(f"wrote {path}")
    print(f"parity_hash={sha256_hex(payload)}")
    return payload


def _resampler_determinism_check() -> dict[str, object]:
    rng = np.random.default_rng(11)
    samples = rng.normal(0, 0.1, 20000).astype(np.float32)
    first = Resampler16k8k().push(samples)
    second = Resampler16k8k().push(samples)
    return {
        "outputs_equal": bool(np.array_equal(first, second)),
        "output_count": int(first.size),
    }


def _fbank_smoke(audio: dict[str, np.ndarray]) -> dict[str, object]:
    first = next(iter(audio.values()))
    fbank = kaldi_fbank_numpy(first)
    return {
        "frames": int(fbank.shape[0]),
        "dim": int(fbank.shape[1]),
        "mean": float(fbank.mean()) if fbank.size else None,
        "std": float(fbank.std()) if fbank.size else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 1 frontend/provenance parity checks (main environment)"
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--hf-root", type=Path, required=True)
    parser.add_argument("--ckpt-root", type=Path, required=True)
    parser.add_argument("--eres-std-root", type=Path, required=True)
    parser.add_argument("--eres-w24-root", type=Path, required=True)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("C:/Users/salee/AppData/Local/Temp/opencode/parity_cache"),
    )
    parser.add_argument("--out", type=Path, default=EXPERIMENT_RESULTS_DIR)
    parser.add_argument("--chunk-samples", type=int, default=512)
    args = parser.parse_args()
    run_parity(
        data_dir=args.data_dir,
        hf_root=args.hf_root,
        ckpt_root=args.ckpt_root,
        eres_std_root=args.eres_std_root,
        eres_w24_root=args.eres_w24_root,
        cache_dir=args.cache_dir,
        out_dir=args.out,
        chunk_samples=args.chunk_samples,
    )


if __name__ == "__main__":
    main()
