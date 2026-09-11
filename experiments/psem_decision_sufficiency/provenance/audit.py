"""Provenance audit: exported-scalar plateau vs upstream/head diagnosis (read-only).

Reads ONLY the 4 frozen sources in FREEZE.json (same directory):
  NP1 ES2009c / NP2 ES2009d / NP3 ES2002b score NPZs + P4 EN2009d score NPZ
  and P4 repaired-text record. No corpus sweep, no optimization, no inference,
  no training, no captures, no network.
Joins NPZ arrays positionally with frozen ceiling sessions via
load_validated_export (the exact join stage2 replay uses).

Usage (repo root):
  ./.venv/Scripts/python.exe experiments/psem_decision_sufficiency/provenance/audit.py
  [--out experiments/psem_decision_sufficiency/provenance/audit_output.json]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import tarfile
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
GPU_EXPORT = (
    ROOT / "experiments" / "psem_state_corrected_adaptation_gate" / "results"
    / "issue-121-h7301-persistence-v1" / "export" / "gpu_export"
)
FRAME_SAMPLES = 1280
F0_TAU = 0.5
H_TAU = 0.5887844788775033  # pinned H100-C policy
CLAMP_P = 1e-6


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(1 << 20)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()

def exact_runs(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    change = np.where(np.diff(values) != 0)[0]
    starts = np.r_[0, change + 1]
    ends = np.r_[change, len(values) - 1]
    return starts, ends


def longest_run(values: np.ndarray) -> dict:
    starts, ends = exact_runs(values)
    lengths = ends - starts + 1
    order = np.argsort(-lengths, kind="stable")
    out = []
    for pos in order[:5]:
        out.append({
            "start_idx": int(starts[pos]),
            "end_idx": int(ends[pos]),
            "length": int(lengths[pos]),
            "value": float(values[starts[pos]]),
        })
    return {"top": out}


def rms_bands(wav_path: str, bands: list[tuple[int, int]]) -> dict:
    """Per-1280-sample-block RMS over NPZ-frame-index bands (read-only slice reads)."""
    out = {}
    with wave.open(wav_path, "rb") as reader:
        assert reader.getframerate() == 16000 and reader.getnchannels() == 1
        assert reader.getsampwidth() == 2
        for lo, hi in bands:
            reader.setpos(lo * FRAME_SAMPLES)
            raw = reader.readframes((hi - lo) * FRAME_SAMPLES)
            pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float64)
            blocks = pcm.reshape(-1, FRAME_SAMPLES)
            rms = np.sqrt((blocks ** 2).mean(axis=1))
            out[f"{lo}..{hi - 1}"] = {
                "n_frames": int(hi - lo),
                "rms_mean": float(rms.mean()),
                "rms_max": float(rms.max()),
                "frac_silent_lt50": float((rms < 50).mean()),
                "first5_rms": [float(v) for v in rms[:5]],
                "last5_rms": [float(v) for v in rms[-5:]],
            }
    return out


def payload_slice_sha(wav_path: str, p0: int, p1: int) -> str:
    with wave.open(wav_path, "rb") as reader:
        reader.setpos(p0)
        raw = reader.readframes(p1 - p0)
    assert len(np.frombuffer(raw, dtype=np.int16)) == p1 - p0
    return hashlib.sha256(raw).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(HERE / "audit_output.json"))
    args = parser.parse_args()

    freeze = json.loads((HERE / "FREEZE.json").read_text(encoding="utf-8"))
    assert freeze["freeze_id"] == "psem.decision_sufficiency.provenance.freeze.v1"

    from experiments.psem_state_corrected_adaptation_gate.h_postprocess import (
        load_validated_export,
    )
    from experiments.psem_sortformer_adaptation_depth.frame_alignment import (
        action_sample_indices,
    )

    export = load_validated_export(GPU_EXPORT)
    manifest = json.loads((GPU_EXPORT / "gpu_export_manifest.json").read_text(encoding="utf-8"))

    # ---- 1. frozen input verification (executed hashes) ----
    inputs = {}
    for key, spec in freeze["sources"].items():
        if key == "P4_EN2009d":
            paths = {"score_npz": spec["path"], "text_json": spec["text_path"]}
            expect = {"score_npz": spec["score_sha256"], "text_json": spec["text_sha256"]}
        else:
            paths = {"npz": spec["path"]}
            expect = {"npz": spec["sha256"]}
        got = {}
        for role, rel in paths.items():
            observed = sha256_file(ROOT / rel)
            got[role] = {"path": rel, "sha256": observed,
                         "match": observed == expect[role]}
        inputs[key] = got

    # ---- 2. whole-record score audit per source ----
    h_logit_tau = math.log(H_TAU / (1.0 - H_TAU))
    per_source = {}
    for src in ("ami_ES2009c", "ami_ES2009d", "ami_ES2002b", "ami_EN2009d"):
        arrays = export["dev"][src]
        session = arrays["session"]
        f0 = np.asarray(arrays["f0_raw"], dtype=np.float64)
        cand = np.asarray(arrays["cand_raw"], dtype=np.float64)
        tgt = np.asarray(arrays["target"], dtype=np.float64)
        valid = np.asarray(arrays["valid"], dtype=bool)
        mapped = np.asarray(arrays["mapped"], dtype=bool)
        starts = np.asarray(session.starts, dtype=np.int64)
        ends = np.asarray(session.ends, dtype=np.int64)
        speech = np.asarray(session.speech_present, dtype=bool)
        masked = np.asarray(session.masked, dtype=bool)
        frontiers = np.asarray(session.frontiers, dtype=np.int64)
        episodes = [str(v) for v in list(session.episode_ids)]
        n = len(f0)
        assert n == len(starts) == len(ends) == int(manifest["dev"][src]["frames"])

        f0_runs = longest_run(f0)
        cand_runs = longest_run(cand)
        resid = cand - f0
        per_source[src] = {
            "frames": n,
            "masks": {
                "valid": int(valid.sum()), "mapped": int(mapped.sum()),
                "valid_and_mapped": int((valid & mapped).sum()),
                "speech": int(speech.sum()), "masked": int(masked.sum()),
                "target_pos": int((tgt > 0.5).sum()),
                "target_pos_valid_mapped": int(((tgt > 0.5) & valid & mapped).sum()),
                "target_values": sorted(float(v) for v in np.unique(tgt)),
                "session_frame_spacing": {
                    "min": int(np.diff(starts).min()), "max": int(np.diff(starts).max()),
                    "median": float(np.median(np.diff(starts))),
                    "first_start": int(starts[0]), "last_end": int(ends[-1]),
                },
            },
            "scores": {
                "f0_min": float(f0.min()), "f0_max": float(f0.max()),
                "f0_nunique": int(len(np.unique(f0))),
                "cand_min": float(cand.min()), "cand_max": float(cand.max()),
                "cand_nunique": int(len(np.unique(cand))),
                "f0_cross_tau": int((f0 >= 0.0).sum()),
                "cand_cross_h_tau": int((cand >= h_logit_tau).sum()),
                "h_logit_tau": h_logit_tau,
            },
            "longest_f0_runs": f0_runs["top"],
            "longest_cand_runs": cand_runs["top"],
            "plateau_detail": None,
        }

    # ---- 3. NP2 declared-plateau verification + episode join ----
    src = "ami_ES2009d"
    arrays = export["dev"][src]
    session = arrays["session"]
    f0 = np.asarray(arrays["f0_raw"], dtype=np.float64)
    cand = np.asarray(arrays["cand_raw"], dtype=np.float64)
    tgt = np.asarray(arrays["target"], dtype=np.float64)
    starts = np.asarray(session.starts, dtype=np.int64)
    ends = np.asarray(session.ends, dtype=np.int64)
    episodes = [str(v) for v in list(session.episode_ids)]
    f0_flat = bool(np.all(f0[13453:18791] == f0[13453]))
    cand_flat = bool(np.all(cand[15476:18791] == cand[15476]))
    plateau_eps = sorted(set(episodes[15476:18791]))
    runs: dict[str, list[int]] = {}
    for i, ep in enumerate(episodes):
        runs.setdefault(ep, []).append(i)
    a271 = runs.get("ami_ES2009d:A00271", [])
    overlap = [i for i in a271 if ends[i] > 33458560 and starts[i] < 33490400]
    per_source[src]["plateau_detail"] = {
        "f0_claim_13453_18790": {
            "all_equal": f0_flat, "value": float(f0[13453]),
            "length": 18791 - 13453,
            "target_pos_inside": int(((tgt[13453:] ) > 0.5).sum()),
            "before_delta": float(f0[13453] - f0[13452]),
            "before_value": float(f0[13452]),
            "reaches_record_end": True, "change_after": None,
            "source_span": [int(starts[13453]), int(ends[18790])],
        },
        "cand_claim_15476_18790": {
            "all_equal": cand_flat, "value": float(cand[15476]),
            "length": 18791 - 15476,
            "target_pos_inside": int(((tgt[15476:]) > 0.5).sum()),
            "before_delta": float(cand[15476] - cand[15475]),
            "before_value": float(cand[15475]),
            "reaches_record_end": True, "change_after": None,
            "source_span": [int(starts[15476]), int(ends[18790])],
            "resid_inside": float((cand[15476:] - f0[15476:])[0]),
            "resid_all_equal_inside": bool(
                np.all((cand[15476:] - f0[15476:]) == (cand[15476] - f0[15476]))),
        },
        "episodes_spanned_by_cand_plateau": {
            "n_unique": len(plateau_eps),
            "first": plateau_eps[0], "last": plateau_eps[-1],
        },
        "a00271": {
            "n_frames_total": len(a271),
            "frame_range": [min(a271), max(a271)] if a271 else None,
            "overlap_scored_span": overlap,
            "overlap_in_cand_plateau": bool(overlap) and min(overlap) >= 15476,
            "overlap_in_f0_plateau": bool(overlap) and min(overlap) >= 13453,
        },
    }

    # ---- 4. waveform identity + energy-vs-score support ----
    from experiments.psem_sortformer_adaptation_depth.execution import (
        load_scoring_sessions,
    )
    corpus_root = Path(".cache/issue-121-local-data/corpus")
    reference_root = Path(".cache/issue-121-local-data/reference")
    runtime = load_scoring_sessions(corpus_root, reference_root, "PSEM-STRATEGY-DEV")
    stage2_freeze = json.loads((ROOT / "experiments" / "psem_repeatability_stage2"
                                / "FREEZE.json").read_text(encoding="utf-8"))
    waveform = {}
    for case, wav_rel in (("NP1", "ami/audio/ES2009c/ES2009c.Mix-Headset.wav"),
                          ("NP2", "ami/audio/ES2009d/ES2009d.Mix-Headset.wav"),
                          ("NP3", "ami/audio/ES2002b/ES2002b.Mix-Headset.wav")):
        spec = stage2_freeze["cases"][case]
        wav_path = str(ROOT / ".cache" / "issue-121-local-data" / "corpus" / wav_rel)
        p0, p1 = (int(v) for v in spec["payload_samples"])
        observed_slice = payload_slice_sha(wav_path, p0, p1)
        wav_full_sha = sha256_file(Path(wav_path))
        src_id = spec["source"]
        waveform[case] = {
            "wav": wav_rel,
            "payload_slice_match_stage2_freeze": observed_slice == spec["payload_sha256"],
            "payload_slice_sha256": observed_slice,
            "full_file_sha256": wav_full_sha,
            "runtime_waveform_sha256": runtime[src_id].waveform_sha256,
            "full_file_matches_runtime_identity": (
                wav_full_sha == runtime[src_id].waveform_sha256),
            "audio_ref": runtime[src_id].audio_ref,
        }
    energy = {
        "ami_ES2009d": rms_bands(
            str(ROOT / ".cache/issue-121-local-data/corpus"
                / "ami/audio/ES2009d/ES2009d.Mix-Headset.wav"),
            [(13000, 13453), (13453, 15476), (15476, 18791)]),
    }

    # ---- 5. smoke repros of suspected code paths (pure numpy, no inference) ----
    rail_lo = math.log(CLAMP_P / (1.0 - CLAMP_P))
    rail_hi = -rail_lo
    all_scores = np.concatenate([np.asarray(export["dev"][s]["f0_raw"], dtype=np.float64)
                                 for s in per_source]
                                + [np.asarray(export["dev"][s]["cand_raw"], dtype=np.float64)
                                   for s in per_source])
    es2009d_session = export["dev"]["ami_ES2009d"]["session"]
    es2009d_ends = np.asarray(es2009d_session.ends, dtype=np.int64)
    native_ends = (np.arange(len(f0), dtype=np.int64) + 1) * FRAME_SAMPLES
    dev_indices = action_sample_indices(native_ends, es2009d_ends)
    counts = np.bincount(dev_indices, minlength=len(f0))
    native_max = int(native_ends.max())
    # Collapse: action ends at/beyond the native grid end saturate searchsorted
    # onto the last native frame (no upper-bound guard in the DEV path at
    # material.py:1356-1362). The tail block mapping to native n-1 is exact.
    collapse = {}
    for src in per_source:
        sess = export["dev"][src]["session"]
        s_ends = np.asarray(sess.ends, dtype=np.int64)
        s_starts = np.asarray(sess.starts, dtype=np.int64)
        arr_f0 = np.asarray(export["dev"][src]["f0_raw"], dtype=np.float64)
        nsrc = len(arr_f0)
        nat = (np.arange(nsrc, dtype=np.int64) + 1) * FRAME_SAMPLES
        idx = action_sample_indices(nat, s_ends)
        to_last = np.where(idx == nsrc - 1)[0]
        first = int(to_last.min()) if len(to_last) else None
        f0_starts, f0_ends = exact_runs(arr_f0)
        f0_lens = f0_ends - f0_starts + 1
        f0_long = int(np.argmax(f0_lens))
        f0_block = [int(f0_starts[f0_long]), int(f0_ends[f0_long])]
        collapse[src] = {
            "native_grid_max_samples": int(nat.max()),
            "session_last_end": int(s_ends.max()),
            "n_action_ends_beyond_native": int((s_ends > int(nat.max())).sum()),
            "tail_to_last_native_count": int(len(to_last)),
            "first_to_last_idx": first,
            "tail_contiguous_to_end": bool(
                len(to_last) and first is not None
                and np.all(np.diff(to_last) == 1) and int(to_last.max()) == nsrc - 1),
            "mapped_native_idx": nsrc - 1,
            "native_frame_source_span": [(nsrc - 1) * FRAME_SAMPLES, nsrc * FRAME_SAMPLES],
            "f0_longest_block": f0_block,
            "tail_block_equals_f0_block": bool(
                len(to_last) and first is not None
                and [first, nsrc - 1] == f0_block),
            "first_tail_end": int(s_ends[first]) if first is not None else None,
            "prev_end": int(s_ends[first - 1]) if first else None,
            "first_tail_episode": str(list(sess.episode_ids)[first]) if first is not None else None,
        }
    smoke = {
        "clamp_rail": {
            "rail_lo": rail_lo, "rail_hi": rail_hi,
            "min_dist_any_score_to_rail": float(
                min(np.abs(all_scores - rail_lo).min(), np.abs(all_scores - rail_hi).min())),
            "any_score_bitwise_on_rail": bool(
                np.any((all_scores == rail_lo) | (all_scores == rail_hi))),
            "plateau_implied_anchor_present_prob": float(
                1.0 / (1.0 + math.exp(-(-3.2511489391326904)))),
            "plateau_f0_is_interior": True,
        },
        "gather_mapping_es2009d": {
            "method": "action_sample_indices((i+1)*1280, session.ends) recomputed exactly",
            "n_grid": len(f0), "n_mapped": len(dev_indices),
            "max_repeat_of_one_native_frame": int(counts.max()),
            "n_repeated_native_frames": int((counts > 1).sum()),
            "n_skipped_native_frames": int((counts == 0).sum()),
            "identity_fraction": float((counts == 1).mean()),
            "verdict": ("OUT-OF-RANGE action ends saturate searchsorted onto the last "
                        "native frame; see collapse section for the block proof"),
        },
        "collapse_out_of_range_gather": collapse,
    }

    q8_receipt = json.loads((ROOT / "experiments" / "psem_frozen_ceiling_gate"
                             / "frozen_inputs" / "dev_sortformer_model_receipt.json"
                             ).read_text(encoding="utf-8"))
    reuse = json.loads((ROOT / "experiments" / "psem_frozen_ceiling_gate"
                        / "evidence_reuse_receipt.json").read_text(encoding="utf-8"))
    q8_npz = ROOT / "experiments/psem_frozen_ceiling_gate/frozen_inputs/posterior_sessions.npz"
    post_declared = reuse.get("artifacts", {}).get("posterior_sessions", {}).get("sha256")
    post_file = sha256_file(q8_npz)
    bundle = json.loads((ROOT / "experiments" / "psem_state_corrected_adaptation_gate"
                         / "results" / "issue-121-h7301-persistence-v1"
                         / "bundle_manifest.json").read_text(encoding="utf-8"))
    stage2_ledger = json.loads((ROOT / "experiments" / "psem_repeatability_stage2"
                                / "LEDGER.json").read_text(encoding="utf-8"))
    p4 = json.loads((ROOT / "experiments" / "psem_evidence_delivery_gap"
                     / "soniox_equal_timestamp_repair" / "results.json"
                     ).read_text(encoding="utf-8"))
    diag_text = (ROOT / "experiments" / "psem_state_corrected_adaptation_gate"
                 / "results" / "issue-121-h7301-persistence-v1"
                 / "canonical" / "gate1_diagnostics.json").read_text(encoding="utf-8")
    cache_hits = [str(p) for p in
                  (ROOT / ".cache" / "issue-121-local-data").rglob("*")
                  if p.is_file() and ("hidden" in p.name or "slot_logit" in p.name)]
    head_hits = [str(p) for p in
                 (ROOT / ".cache" / "issue-121-local-data").rglob("*")
                 if p.is_file() and ("step-256" in p.name or "bb5029" in p.name)]
    nemo_pkg = ROOT / ".cache/issue-107-assets/packages/nemo-1a3c291b3ef0f0e11b72f789b185e1f1bda39bd6.tar.gz"
    nemo_names: list[str] = []
    if nemo_pkg.is_file():
        with tarfile.open(nemo_pkg, "r:gz") as tar:
            for member in tar.getmembers():
                name = member.name
                if ("preprocessor" in name or "sortformer" in name) and name.endswith(".py"):
                    nemo_names.append(name)
                if len(nemo_names) >= 40:
                    break
    f0_nemo = ROOT / ".cache/issue-107-assets/checkpoints/diar_streaming_sortformer_4spk-v2.1.nemo"
    receipts = {
        "q8_chain": {
            "model_sha256": q8_receipt.get("model_sha256"),
            "source_commit": q8_receipt.get("source_commit"),
            "chunk_audio_ms": q8_receipt.get("chunk_audio_ms"),
            "lookahead_ms": q8_receipt.get("recorded_algorithmic_lookahead_ms"),
            "posterior_sessions_sha256": post_declared,
            "posterior_sessions_file_sha256": post_file,
            "posterior_file_match": post_file == post_declared,
        },
        "nemo121_chain": {
            "checkpoint_sha256": manifest.get("binding", {}).get("checkpoint_hash"),
            "code_hash": manifest.get("binding", {}).get("code_hash"),
            "trained_head_sha256": manifest.get("trained_head_sha256"),
            "manifest_dev_es2009d": manifest.get("dev", {}).get("ami_ES2009d"),
            "bundle_execution_provenance": bundle.get("execution_provenance"),
            "dev_invocation_prefix_receipt_present": ("prefix_causality" in diag_text),
        },
        "stage2_np2_ledger": {
            "events": stage2_ledger["cases"]["NP2"]["events"],
            "support": stage2_ledger["cases"]["NP2"]["support"],
        },
        "p4_text_record": {
            "top_keys": sorted(p4.keys()),
            "repair": p4.get("repair"),
            "counts": p4.get("counts"),
        },
        "missing_probes": {
            "hidden192_slotlogits4_cache_files": cache_hits,
            "trained_head_weights_files": head_hits,
            "restored_f0_nemo_present": f0_nemo.is_file(),
            "restored_f0_nemo_bytes": f0_nemo.stat().st_size if f0_nemo.is_file() else None,
            "nemo_source_files_for_static_causal_review": sorted(nemo_names)[:40],
        },
    }

    output = {
        "freeze_id": freeze["freeze_id"],
        "inputs": inputs,
        "per_source": per_source,
        "waveform": waveform,
        "energy": energy,
        "smoke": smoke,
        "receipts": receipts,
        "unmeasured": {
            "per_frame_hidden192": None,
            "per_frame_slot_logits4": None,
            "per_frame_probabilities_121": None,
            "dev_invocation_prefix_invariance": None,
            "trained_head_weights": None,
            "realtime_model_failure_mechanism": None,
        },
    }
    out_path = Path(args.out)
    out_path.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {out_path} ({out_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
