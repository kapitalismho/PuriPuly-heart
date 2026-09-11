"""Bounded F0 prefix-support probe (authorized, no paid calls).

Full 7s payload clip vs causal prefix (first 4s + 1040ms lookahead) through the
EXACT #121 streaming preparation (process_signal + streaming_feat_loader +
_streaming_step via TrainableSortformerPSEM.sortformer_evidence, eval mode,
no_grad) - never plain model.forward. Compares native evidence
(probabilities/activity_logits/final_temporal_hidden) over the first 50 frames
(4s), whose charged 1040ms lookahead is fully inside the prefix. No H head, no
training, no captures, no latency credit. Scoped evidence only, never global.

Run (isolated env): C:/tmp/psem-timing-iso/Scripts/python.exe
  C:/tmp/psem-prefix/prefix_probe.py
Writes: C:/tmp/psem-prefix/prefix_results.json (copied into provenance by hand).
"""
import copy
import hashlib
import json
import math
import sys
import time
import tracemalloc
import wave
from pathlib import Path

REPO = Path(
    r"C:\Users\salee\Documents\dev\puripuly_heart\.worktrees\puripuly_heart"
    r"\experiment-v2-speaker-change-turn-boundaries-ls"
)
NEMO_TREE = Path(r"C:\tmp\psem-nemo\NeMo")
NEMO_REV = "1a3c291b3ef0f0e11b72f789b185e1f1bda39bd6"
CHECKPOINT = (REPO / ".cache" / "issue-107-assets" / "checkpoints"
              / "diar_streaming_sortformer_4spk-v2.1.nemo")
CHECKPOINT_SHA = "8abd32832159c6ac1148c926b7276f35ba34582c444e559dce1f1253fea42ef8"
FRAME_SAMPLES = 1280
LOOKAHEAD = 16640
FULL_SAMPLES = 112000
PREFIX_SAMPLES = 64000 + LOOKAHEAD  # 80640
COMPARE_FRAMES = 50
ATOL = 1e-6

CLIPS = {
    "NP1": (r"C:\Users\salee\.psem-corpus\ami\audio\ES2009c\ES2009c.Mix-Headset.wav",
            19741040),
    "NP2": (r"C:\Users\salee\.psem-corpus\ami\audio\ES2009d\ES2009d.Mix-Headset.wav",
            33405760),
    "NP3": (r"C:\Users\salee\.psem-corpus\ami\audio\ES2002b\ES2002b.Mix-Headset.wav",
            2553520),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(1 << 20)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def read_clip(wav_path: str, start: int, count: int):
    with wave.open(wav_path, "rb") as reader:
        assert reader.getframerate() == 16000 and reader.getnchannels() == 1
        reader.setpos(start)
        raw = reader.readframes(count)
    import numpy as np
    pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return pcm


def main() -> None:
    sys.path.insert(0, str(NEMO_TREE))  # extracted repo root exposes top-level `nemo`
    sys.path.insert(0, str(REPO))
    # NOTE: extracted tree root must expose top-level `nemo`
    import nemo  # noqa: F401
    assert Path(nemo.__file__).resolve().is_relative_to(NEMO_TREE.resolve()), \
        nemo.__file__
    import torch

    assert sha256_file(CHECKPOINT) == CHECKPOINT_SHA, "checkpoint identity differs"
    from nemo.collections.asr.models.sortformer_diar_models import (
        SortformerEncLabelModel,
    )
    from nemo.collections.asr.parts.utils.asr_multispeaker_utils import (
        get_ats_targets,
    )
    from experiments.psem_sortformer_adaptation_depth.nemo_adapter import (
        TrainableSortformerPSEM,
    )
    from experiments.psem_sortformer_adaptation_depth.runtime_audit import (
        LOW_LATENCY_STREAMING,
    )

    tracemalloc.start()
    t0 = time.perf_counter()
    sortformer = SortformerEncLabelModel.restore_from(
        restore_path=str(CHECKPOINT), map_location=torch.device("cpu"))
    sortformer.streaming_mode = True
    sortformer.async_streaming = False
    for field, value in LOW_LATENCY_STREAMING.items():
        setattr(sortformer.sortformer_modules, field, value)
        if hasattr(sortformer._cfg.sortformer_modules, field):
            setattr(sortformer._cfg.sortformer_modules, field, value)
    sortformer.sortformer_modules._check_streaming_parameters()
    model = TrainableSortformerPSEM(sortformer, get_ats_targets,
                                    NEMO_TREE).to("cpu")
    model.eval()
    load_seconds = time.perf_counter() - t0
    _, peak_py = tracemalloc.get_traced_memory()

    observed_preset = {k: int(getattr(sortformer.sortformer_modules, k))
                       for k in LOW_LATENCY_STREAMING}
    assert observed_preset == LOW_LATENCY_STREAMING, observed_preset

    results = {
        "checkpoint_sha256": CHECKPOINT_SHA,
        "nemo_rev": NEMO_REV,
        "nemo_tree": str(NEMO_TREE),
        "streaming_preset": observed_preset,
        "preparation": "TrainableSortformerPSEM.sortformer_evidence "
                       "(process_signal + streaming_feat_loader + _streaming_step), "
                       "eval, no_grad; never plain model.forward",
        "load_seconds_cpu": load_seconds,
        "load_peak_python_bytes": peak_py,
        "clips": {},
    }

    with torch.no_grad():
        for case, (wav_path, start) in CLIPS.items():
            pcm = read_clip(wav_path, start, FULL_SAMPLES)
            full = pcm[:87 * FRAME_SAMPLES]
            dropped = FULL_SAMPLES - 87 * FRAME_SAMPLES
            assert dropped == 640, dropped
            prefix = pcm[:PREFIX_SAMPLES]
            assert len(prefix) == 63 * FRAME_SAMPLES
            entry = {"payload_start": start, "full_used": len(full),
                     "prefix_used": len(prefix), "full_dropped_trailing": dropped}

            def run(samples):
                import numpy as np  # noqa: F401
                waveform = torch.from_numpy(
                    np.ascontiguousarray(samples)).unsqueeze(0)
                lengths = torch.tensor([waveform.shape[1]], dtype=torch.long)
                reset = torch.zeros((1, waveform.shape[1] // FRAME_SAMPLES, 1),
                                    dtype=torch.bool)
                reset[:, 0, 0] = True
                trace0 = tracemalloc.get_traced_memory()[0]
                start_t = time.perf_counter()
                evidence = model.sortformer_evidence(
                    waveform, lengths, state_reset=reset)
                wall = time.perf_counter() - start_t
                trace1 = tracemalloc.get_traced_memory()[0]
                return evidence, wall, trace1 - trace0

            ev_full, wall_full, mem_full = run(full)
            ev_prefix, wall_prefix, mem_prefix = run(prefix)
            assert ev_full.probabilities.shape[1] == 87
            assert ev_prefix.probabilities.shape[1] == 63

            deltas = {}
            for name in ("probabilities", "activity_logits",
                         "final_temporal_hidden"):
                a = getattr(ev_full, name)[0, :COMPARE_FRAMES]
                b = getattr(ev_prefix, name)[0, :COMPARE_FRAMES]
                d = (a - b).abs().max().item()
                deltas[name] = d
            worst = max(deltas.values())
            # beyond-lookahead zone must diverge or match; recorded, never gated
            tail = {}
            for name in ("probabilities", "activity_logits"):
                a = getattr(ev_full, name)[0, 50:63]
                b = getattr(ev_prefix, name)[0, 50:63]
                tail[name] = (a - b).abs().max().item()
            entry.update({
                "wall_full_s": wall_full, "wall_prefix_s": wall_prefix,
                "mem_full_bytes": mem_full, "mem_prefix_bytes": mem_prefix,
                "charged_prefix_max_abs_delta": deltas,
                "charged_prefix_worst": worst,
                "charged_prefix_stable_at_1e-6": bool(worst <= ATOL),
                "beyond_lookahead_tail_delta": tail,
            })
            results["clips"][case] = entry
            print(f"{case}: worst={worst:.3e} stable={worst <= ATOL} "
                  f"wall={wall_full:.1f}/{wall_prefix:.1f}s", flush=True)

    dest = Path(r"C:\tmp\psem-prefix\prefix_results.json")
    dest.write_text(json.dumps(results, indent=2, sort_keys=True))
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
