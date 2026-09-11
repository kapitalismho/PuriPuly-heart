"""Direct upstream F0 restore plus bounded compute probes.

Runs under the isolated temp interpreter with PYTHONPATH pointing at the
frozen NeMo checkout. Bypasses the production validator gates, therefore
every output here is DIRECT_UPSTREAM_RESTORE_NON_PARITY and never frozen
parity. Real-audio crops run as isolated crops with no recurrent prefix
state. Usage: iso_restore.py <checkpoint> <out_json> <crops_json>
"""
from __future__ import annotations

import datetime
import hashlib
import json
import sys
import time
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utcnow() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def timed_forward(model, wave, lengths, warmup: int, iters: int) -> dict:
    import torch

    with torch.no_grad():
        for _ in range(warmup):
            model(wave, lengths)
    laps = []
    with torch.no_grad():
        for _ in range(iters):
            mark = time.perf_counter()
            preds = model(wave, lengths)
            laps.append(time.perf_counter() - mark)
    return {
        "warmup_iters": warmup,
        "timed_iters": iters,
        "laps_seconds": laps,
        "mean_seconds": sum(laps) / len(laps),
        "output_type": type(preds).__name__,
        "output_shape": list(preds.shape),
        "output_mean": float(preds.mean()),
    }


def main() -> int:
    checkpoint = Path(sys.argv[1])
    out_path = Path(sys.argv[2])
    crops = json.loads(sys.argv[3])
    record: dict = {
        "label": "DIRECT_UPSTREAM_RESTORE_NON_PARITY",
        "validator_gates_bypassed": [
            "dependency_lock",
            "container_image_identity",
            "accelerator_identity",
            "symbol_origin",
        ],
        "checkpoint": str(checkpoint),
        "started_at": utcnow(),
    }
    try:
        tick = time.perf_counter()
        digest = sha256_file(checkpoint)
        record["sha_seconds"] = time.perf_counter() - tick
        record["sha256"] = digest
        record["size_bytes"] = checkpoint.stat().st_size
        import torch

        torch.set_num_threads(1)
        record["torch_version"] = torch.__version__
        record["cuda_available"] = torch.cuda.is_available()
        tick = time.perf_counter()
        from nemo.collections.asr.models.sortformer_diar_models import (
            SortformerEncLabelModel,
        )

        record["import_seconds"] = time.perf_counter() - tick
        record["symbol_file"] = str(Path(__import__("inspect").getfile(SortformerEncLabelModel)).resolve())
        tick = time.perf_counter()
        model = SortformerEncLabelModel.restore_from(
            restore_path=str(checkpoint), map_location="cpu"
        )
        record["restore_seconds"] = time.perf_counter() - tick
        model.eval()
        params = sum(int(p.numel()) for p in model.parameters())
        record["model"] = {
            "class": type(model).__name__,
            "device": str(next(model.parameters()).device),
            "params": params,
            "streaming_mode": bool(getattr(model, "streaming_mode", False)),
        }
        gen = torch.Generator().manual_seed(7301)
        synth = torch.randn(1, 112000, generator=gen, dtype=torch.float32) * 0.05
        synth_lengths = torch.tensor([112000], dtype=torch.long)
        synthetic = timed_forward(model, synth, synth_lengths, 1, 2)
        synthetic["seed"] = 7301
        synthetic["input_kind"] = "synthetic_gaussian"
        record["synthetic_supplemental"] = synthetic
        import soundfile as sf

        real_forwards = []
        for crop in crops:
            wav_path = Path(crop["wav"])
            info = sf.info(str(wav_path))
            data, rate = sf.read(
                str(wav_path),
                start=int(crop["start"]),
                stop=int(crop["end"]),
                dtype="int16",
                always_2d=False,
            )
            if int(rate) != 16000:
                raise RuntimeError(f"unexpected sample rate: {rate}")
            if int(data.shape[0]) != 112000:
                raise RuntimeError(f"unexpected crop length: {data.shape}")
            payload_sha = hashlib.sha256(data.tobytes()).hexdigest()
            wave = torch.from_numpy(data.astype("float32") / 32768.0).unsqueeze(0)
            lengths = torch.tensor([112000], dtype=torch.long)
            result = timed_forward(model, wave, lengths, 1, 2)
            result["case_id"] = crop["case_id"]
            result["meeting"] = crop["meeting"]
            result["wav"] = str(wav_path)
            result["wav_frames"] = int(info.frames)
            result["wav_rate"] = int(info.samplerate)
            result["crop_start"] = int(crop["start"])
            result["crop_end"] = int(crop["end"])
            result["crop_samples"] = int(data.shape[0])
            result["payload_dtype"] = str(data.dtype)
            result["payload_sha256"] = payload_sha
            result["input_kind"] = "real_meeting_crop"
            real_forwards.append(result)
        record["real_forwards"] = real_forwards
        record["status"] = "ok"
        record["error"] = None
    except Exception as exc:
        record["status"] = "failed"
        record["error"] = {"type": type(exc).__name__, "message": str(exc)[:800]}
    record["finished_at"] = utcnow()
    out_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
