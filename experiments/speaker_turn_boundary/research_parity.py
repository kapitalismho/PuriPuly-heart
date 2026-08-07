from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from experiments.speaker_turn_boundary.adapters.eres2netv2 import (
    ERES_SAMPLE_RATE_HZ,
    EresEmbeddingRuntime,
    kaldi_fbank_numpy,
)
from experiments.speaker_turn_boundary.frontend import (
    StreamingLSEENDFrontend,
    extract_logmel23_cummn_offline,
)
from experiments.speaker_turn_boundary.provenance import (
    ERES_STANDARD_REVISION,
    ERES_W24_REVISION,
    FS_EEND_REVISION,
    LS_EEND_ONNX_REVISION,
)
from experiments.speaker_turn_boundary.schemas import canonical_json, sha256_hex
from experiments.speaker_turn_boundary.vad_baseline import load_canonical_wav

GOLDEN_CASE_IDS = ("golden_two_utterance_gap400", "golden_single_utterance", "golden_silence")

MODELSCOPE_EXAMPLE_WAVS = {
    "speaker1_a_cn_16k.wav": "5f20ce0ddc378ca3239d3ce864b1142726a46a1221ae553912e4e142045df58b",
    "speaker1_b_cn_16k.wav": "20745dc08a4281894d146140b99b9ef7417ac681119b7f7202f553cdf1a85f65",
    "speaker2_a_cn_16k.wav": "8a6cffa452df32ef10503f7992f22ffcdd7f16c4e0273d13311bc5cdcb13abf4",
}


def _load_parity_audio(data_dir: Path, cache_dir: Path) -> dict[str, np.ndarray]:
    import hashlib

    audio: dict[str, np.ndarray] = {}
    for case_id in GOLDEN_CASE_IDS:
        wav_path = data_dir / "generated" / f"{case_id}.wav"
        if wav_path.is_file():
            audio[case_id] = load_canonical_wav(wav_path)
    for file_name, expected_hash in MODELSCOPE_EXAMPLE_WAVS.items():
        wav_path = cache_dir / file_name
        if wav_path.is_file():
            actual = hashlib.sha256(wav_path.read_bytes()).hexdigest()
            if actual == expected_hash:
                audio[file_name] = load_canonical_wav(wav_path)
    return audio


def _resample8k(samples_16k: np.ndarray) -> np.ndarray:
    from experiments.speaker_turn_boundary.frontend import Resampler16k8k

    return Resampler16k8k().push(samples_16k)


def ls_eend_frontend_parity(audio: dict[str, np.ndarray], fs_eend_root: Path) -> dict[str, object]:
    sys.path.insert(0, str(fs_eend_root / "LS-EEND"))
    from datasets.feature import extract_fbank

    results: dict[str, object] = {}
    for name, samples_16k in audio.items():
        samples_8k = _resample8k(samples_16k)
        wav_path = _write_temp_wav(samples_8k, 8000, name)
        reference_offline = extract_fbank(
            wav_path,
            context_size=7,
            input_transform="logmel23_cummn",
            frame_size=200,
            frame_shift=80,
            subsampling=10,
        ).numpy()
        mine_offline = extract_logmel23_cummn_offline(samples_8k)
        mine_stream = _my_streaming(samples_8k)
        aligned_offline = min(reference_offline.shape[0], mine_offline.shape[0])
        aligned_stream = min(reference_offline.shape[0], mine_stream.shape[0])
        results[name] = {
            "reference_offline_frames": int(reference_offline.shape[0]),
            "mine_offline_frames": int(mine_offline.shape[0]),
            "mine_stream_frames": int(mine_stream.shape[0]),
            "offline_max_abs_error": (
                float(
                    np.abs(
                        reference_offline[:aligned_offline] - mine_offline[:aligned_offline]
                    ).max()
                )
                if aligned_offline
                else None
            ),
            "stream_max_abs_error": (
                float(
                    np.abs(reference_offline[:aligned_stream] - mine_stream[:aligned_stream]).max()
                )
                if aligned_stream
                else None
            ),
        }
    return results


def ls_eend_thirdparty_streaming_deviation(
    audio: dict[str, np.ndarray], hf_example_dir: Path
) -> dict[str, object]:
    sys.path.insert(0, str(hf_example_dir))
    sys.path.insert(0, str(hf_example_dir / "datasets"))

    from ls_eend_common import config_from_metadata
    from ls_eend_streaming_common import StreamingFeatureExtractor

    metadata = json.loads(
        (hf_example_dir.parent / "AMI" / "ls_eend_ami_step.json").read_text(encoding="utf-8")
    )
    config = config_from_metadata(metadata)
    results: dict[str, object] = {}
    for name, samples_16k in audio.items():
        samples_8k = _resample8k(samples_16k)
        extractor = StreamingFeatureExtractor(config)
        collected: list[np.ndarray] = []
        offset = 0
        while offset < samples_8k.size:
            emitted = extractor.push_audio(samples_8k[offset : offset + 4000])
            if emitted.size:
                collected.append(emitted)
            offset += 4000
        tail = extractor.finalize()
        if tail.size:
            collected.append(tail)
        third_party = (
            np.concatenate(collected, axis=0) if collected else np.zeros((0, 345), np.float32)
        )
        official = extract_logmel23_cummn_offline(samples_8k)
        aligned = min(third_party.shape[0], official.shape[0])
        results[name] = {
            "third_party_stream_frames": int(third_party.shape[0]),
            "official_frames": int(official.shape[0]),
            "max_abs_error": (
                float(np.abs(third_party[:aligned] - official[:aligned]).max()) if aligned else None
            ),
        }
    return results


def _my_streaming(samples_8k: np.ndarray) -> np.ndarray:
    frontend = StreamingLSEENDFrontend()
    collected: list[np.ndarray] = []
    offset = 0
    while offset < samples_8k.size:
        emitted = frontend.push_audio(samples_8k[offset : offset + 4000])
        if emitted.size:
            collected.append(emitted)
        offset += 4000
    tail = frontend.finalize()
    if tail.size:
        collected.append(tail)
    if not collected:
        return np.zeros((0, 345), dtype=np.float32)
    return np.concatenate(collected, axis=0)


def ls_eend_neural_parity(
    audio: dict[str, np.ndarray],
    fs_eend_root: Path,
    hf_root: Path,
    ckpt_root: Path,
) -> dict[str, object]:
    sys.path.insert(0, str(fs_eend_root / "LS-EEND"))
    import torch
    from datasets.feature import extract_fbank
    from nnet.model.onl_conformer_retention_enc_1dcnn_tfm_retention_enc_linear_non_autoreg_pos_enc_l2norm_emb_loss_mask import (
        OnlineConformerRetentionDADiarization,
        StreamingConv1d,
    )

    common_params = {
        "n_units": 256,
        "n_heads": 4,
        "enc_n_layers": 4,
        "feed_forward_expansion_factor": 4,
        "conv_expansion_factor": 2,
        "dropout": 0.1,
        "conv_kernel_size": 16,
        "half_step_residual": True,
        "recurrent_chunk_size": 500,
        "dec_n_layers": 2,
        "conv_delay": 9,
    }
    in_size = (2 * 7 + 1) * 23
    variants = {
        "L-AMI": ("ami.ckpt", "AMI/ls_eend_ami_step.onnx", 4, 100000),
        "L-CALLHOME": ("ch.ckpt", "CALLHOME/ls_eend_callhome_step.onnx", 7, 10000),
        "L-DIHARD-II": ("dih2.ckpt", "DIHARD II/ls_eend_dih2_step.onnx", 10, 100000),
        "L-DIHARD-III": ("dih3.ckpt", "DIHARD III/ls_eend_dih3_step.onnx", 10, 100000),
    }
    results: dict[str, object] = {}
    for variant, (ckpt_name, onnx_rel, max_speakers, max_seqlen) in variants.items():
        params = dict(common_params)
        params["max_seqlen"] = max_seqlen
        model = OnlineConformerRetentionDADiarization(n_speakers=None, in_size=in_size, **params)
        state_dict = torch.load(str(ckpt_root / ckpt_name), map_location="cpu", weights_only=False)
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        cleaned = {
            (k[len("model.") :] if k.startswith("model.") else k): v for k, v in state_dict.items()
        }
        cleaned = {
            k.replace("dec.attractor_decoder.layers.", "dec.layers."): v for k, v in cleaned.items()
        }
        missing, unexpected = model.load_state_dict(cleaned, strict=False)
        model.eval()
        kernel_size = 2 * model.delay + 1
        streaming_cnn = StreamingConv1d(model.n_units, model.n_units, kernel_size=kernel_size)
        streaming_cnn.conv.load_state_dict(model.cnn.state_dict())
        streaming_cnn.eval()

        variant_results: dict[str, object] = {}
        for name, samples_16k in audio.items():
            samples_8k = _resample8k(samples_16k)
            wav_path = _write_temp_wav(samples_8k, 8000, name)
            feat = extract_fbank(
                wav_path,
                context_size=7,
                input_transform="logmel23_cummn",
                frame_size=200,
                frame_shift=80,
                subsampling=10,
            )
            enc_states, dec_states = _init_states(model, torch.device("cpu"))
            with torch.no_grad():
                torch_logits = _official_streaming_predict(
                    model, streaming_cnn, feat, max_speakers + 2, enc_states, dec_states
                )
            onnx_logits = _onnx_predict(hf_root / onnx_rel, feat.numpy())
            aligned = min(torch_logits.shape[0], onnx_logits.shape[0])
            if aligned == 0:
                variant_results[name] = {"frames": 0}
                continue
            torch_probs = 1.0 / (1.0 + np.exp(-torch_logits[:aligned]))
            onnx_probs = 1.0 / (1.0 + np.exp(-onnx_logits[:aligned]))
            variant_results[name] = {
                "torch_frames": int(torch_logits.shape[0]),
                "onnx_frames": int(onnx_logits.shape[0]),
                "prob_max_abs_error": float(np.abs(torch_probs - onnx_probs).max()),
                "prob_mean_abs_error": float(np.abs(torch_probs - onnx_probs).mean()),
                "missing_keys": len(missing),
                "unexpected_keys": len(unexpected),
            }
        results[variant] = variant_results
    return results


def _init_states(model, device):
    import torch

    n_enc_layers = len(model.enc.encoder.layers)
    n_dec_layers = len(model.dec.layers)
    enc_conv_ksize = model.enc.encoder._conv_kernel_size
    enc_states = {
        "ret_states": [dict() for _ in range(n_enc_layers)],
        "conv_caches": [
            torch.zeros(1, model.n_units, enc_conv_ksize - 1, device=device)
            for _ in range(n_enc_layers)
        ],
    }
    dec_states = [dict() for _ in range(n_dec_layers)]
    return enc_states, dec_states


def _official_streaming_predict(model, streaming_cnn, feat, max_nspks, enc_states, dec_states):
    import torch

    streaming_cnn.buffer.clear()
    streaming_cnn.t = 0
    preds = []
    dec_t = 0

    def step(emb_t, dec_t_local):
        emb_t_conv = streaming_cnn(emb_t.transpose(1, 2))
        if emb_t_conv is None:
            return None, dec_t_local
        emb_t_conv = emb_t_conv.transpose(1, 2)
        emb_t_conv = emb_t_conv / torch.norm(emb_t_conv, dim=-1, keepdim=True)
        attractor_t = model.dec.forward_one_step(emb_t_conv, dec_t_local, max_nspks, dec_states)
        attractor_t = attractor_t / torch.norm(attractor_t, dim=-1, keepdim=True)
        y_t = torch.matmul(emb_t_conv.unsqueeze(dim=-2), attractor_t.transpose(-1, -2)).squeeze(
            dim=-2
        )
        return y_t, dec_t_local + 1

    for t in range(feat.shape[0]):
        x_t = feat[t : t + 1].unsqueeze(0)
        emb_t = model.enc.forward_one_step(
            x_t, t, enc_states["ret_states"], enc_states["conv_caches"]
        )
        y_t, dec_t = step(emb_t, dec_t)
        if y_t is not None:
            preds.append(y_t)
    for _ in range(model.delay):
        emb_zero = torch.zeros(1, 1, model.n_units)
        y_t, dec_t = step(emb_zero, dec_t)
        if y_t is not None:
            preds.append(y_t)
    return torch.cat(preds, dim=1).squeeze(0).numpy()


def _onnx_predict(onnx_path: Path, features: np.ndarray) -> np.ndarray:

    import onnxruntime as ort

    metadata_path = onnx_path.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    state = {
        key: np.zeros(tuple(shape), dtype=np.float32)
        for key, shape in metadata["state_shapes"].items()
    }
    output_names = [output.name for output in session.get_outputs()]
    logits: list[np.ndarray] = []
    for frame_index, frame in enumerate(features):
        should_decode = 1.0 if frame_index >= 9 else 0.0
        outputs = session.run(
            output_names,
            {
                "frame": frame.reshape(1, 1, -1).astype(np.float32),
                "enc_ret_kv": state["enc_ret_kv"],
                "enc_ret_scale": state["enc_ret_scale"],
                "enc_conv_cache": state["enc_conv_cache"],
                "dec_ret_kv": state["dec_ret_kv"],
                "dec_ret_scale": state["dec_ret_scale"],
                "top_buffer": state["top_buffer"],
                "ingest": np.array([1.0], dtype=np.float32),
                "decode": np.array([should_decode], dtype=np.float32),
            },
        )
        named = dict(zip(output_names, outputs))
        state = {
            "enc_ret_kv": named["enc_ret_kv_out"],
            "enc_ret_scale": named["enc_ret_scale_out"],
            "enc_conv_cache": named["enc_conv_cache_out"],
            "dec_ret_kv": named["dec_ret_kv_out"],
            "dec_ret_scale": named["dec_ret_scale_out"],
            "top_buffer": named["top_buffer_out"],
        }
        if should_decode == 1.0:
            logits.append(named["full_logits"].reshape(1, -1))
    pending = len(features) - len(logits)
    for _ in range(pending):
        outputs = session.run(
            output_names,
            {
                "frame": np.zeros((1, 1, features.shape[1]), dtype=np.float32),
                "enc_ret_kv": state["enc_ret_kv"],
                "enc_ret_scale": state["enc_ret_scale"],
                "enc_conv_cache": state["enc_conv_cache"],
                "dec_ret_kv": state["dec_ret_kv"],
                "dec_ret_scale": state["dec_ret_scale"],
                "top_buffer": state["top_buffer"],
                "ingest": np.array([0.0], dtype=np.float32),
                "decode": np.array([1.0], dtype=np.float32),
            },
        )
        named = dict(zip(output_names, outputs))
        state = {
            "enc_ret_kv": named["enc_ret_kv_out"],
            "enc_ret_scale": named["enc_ret_scale_out"],
            "enc_conv_cache": named["enc_conv_cache_out"],
            "dec_ret_kv": named["dec_ret_kv_out"],
            "dec_ret_scale": named["dec_ret_scale_out"],
            "top_buffer": named["top_buffer_out"],
        }
        logits.append(named["full_logits"].reshape(1, -1))
    return np.concatenate(logits, axis=0)


frame_index = 0


def _write_temp_wav(samples: np.ndarray, sample_rate: int, name: str) -> Path:
    import tempfile

    import soundfile as sf

    temp_dir = Path(tempfile.mkdtemp(prefix="stb_parity_wav_"))
    path = temp_dir / f"{name.replace('.', '_')}.wav"
    sf.write(str(path), np.asarray(samples, dtype=np.float32), sample_rate, subtype="FLOAT")
    return path


def eres_parity(
    audio: dict[str, np.ndarray],
    eres_std_root: Path,
    eres_w24_root: Path,
    eres_onnx_root: Path,
    speaker_root: Path,
) -> dict[str, object]:
    import torch
    import torchaudio

    sys.path.insert(0, str(speaker_root))
    from speakerlab.models.eres2net.ERes2NetV2 import ERes2NetV2

    configs = {
        "E-standard": (
            eres_std_root / "pretrained_eres2netv2.ckpt",
            26,
            2,
            2,
            eres_onnx_root / "eres2netv2.onnx",
        ),
        "E-w24s4ep4": (
            eres_w24_root / "pretrained_eres2netv2w24s4ep4.ckpt",
            24,
            4,
            4,
            eres_onnx_root / "eres2netv2_w24s4ep4.onnx",
        ),
    }
    results: dict[str, object] = {}
    for tag, (ckpt_path, base_width, scale, expansion, onnx_path) in configs.items():
        state = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model = ERes2NetV2(
            feat_dim=80, embedding_size=192, baseWidth=base_width, scale=scale, expansion=expansion
        )
        model.load_state_dict(state)
        model.eval()
        runtime = EresEmbeddingRuntime(str(onnx_path))
        tag_results: dict[str, object] = {}
        for name, samples in audio.items():
            samples = samples[: int(3.0 * ERES_SAMPLE_RATE_HZ)]
            fbank_ref = torchaudio.compliance.kaldi.fbank(
                torch.from_numpy(samples).unsqueeze(0),
                num_mel_bins=80,
                sample_frequency=16000,
                dither=0.0,
            ).numpy()
            fbank_ref = fbank_ref - fbank_ref.mean(axis=0, keepdims=True)
            fbank_mine = kaldi_fbank_numpy(samples)
            fbank_mine = fbank_mine - fbank_mine.mean(axis=0, keepdims=True)
            with torch.no_grad():
                torch_emb = model(torch.from_numpy(fbank_ref[None, :, :])).numpy()[0]
            onnx_emb = runtime.embed(samples)
            tag_results[name] = {
                "fbank_max_abs_error": float(np.abs(fbank_ref - fbank_mine).max()),
                "embedding_cosine": float(
                    np.dot(torch_emb, onnx_emb)
                    / (np.linalg.norm(torch_emb) * np.linalg.norm(onnx_emb) + 1e-12)
                ),
                "embedding_mae": float(np.abs(torch_emb - onnx_emb).mean()),
            }
        results[tag] = tag_results
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 1 reference-environment parity checks (torch/librosa)"
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--hf-root", type=Path, required=True)
    parser.add_argument("--ckpt-root", type=Path, required=True)
    parser.add_argument("--eres-std-root", type=Path, required=True)
    parser.add_argument("--eres-w24-root", type=Path, required=True)
    parser.add_argument("--eres-onnx-root", type=Path, required=True)
    parser.add_argument("--fs-eend-root", type=Path, required=True)
    parser.add_argument("--speaker-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    audio = _load_parity_audio(args.data_dir, args.cache_dir)
    started_at = datetime.now(timezone.utc).isoformat()
    records: dict[str, object] = {}
    records["ls_eend_frontend"] = ls_eend_frontend_parity(audio, args.fs_eend_root)
    records["ls_eend_thirdparty_streaming_deviation"] = ls_eend_thirdparty_streaming_deviation(
        audio, args.hf_root / "example"
    )
    records["ls_eend_neural"] = ls_eend_neural_parity(
        audio, args.fs_eend_root, args.hf_root, args.ckpt_root
    )
    records["eres"] = eres_parity(
        audio, args.eres_std_root, args.eres_w24_root, args.eres_onnx_root, args.speaker_root
    )
    finished_at = datetime.now(timezone.utc).isoformat()
    payload = {
        "parity_id": "phase1_research_parity",
        "revisions": {
            "fs_eend": FS_EEND_REVISION,
            "ls_eend_onnx": LS_EEND_ONNX_REVISION,
            "eres_standard": ERES_STANDARD_REVISION,
            "eres_w24": ERES_W24_REVISION,
        },
        "started_at_utc": started_at,
        "finished_at_utc": finished_at,
        "records": records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(canonical_json(payload), encoding="utf-8")
    print(f"wrote {args.out}")
    print(f"parity_hash={sha256_hex(payload)}")


if __name__ == "__main__":
    main()
