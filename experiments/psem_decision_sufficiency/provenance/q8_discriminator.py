"""Q8 independent-observation discriminator (alignment-only mock, read-only).

Compares EXACT original broken projection vs corrected projection on the SAME
stored Q8 trajectory (posterior_sessions.npz s008 = ami_ES2009d, 18791
action-aligned rows from the transcribe.cpp Q8 streaming run). No models run,
no inference, no training. LABEL: INDEPENDENT_OBSERVATION - never the same
upstream feature as the #121 NeMo scores. Does NOT repair H (head/cache missing).

Broken projector: replicates the DEV join defect bit-exactly on the shared
rows - rows[13453..18790] replaced by rows[18790] (the audited ES2009d
tail_to_last block). Corrected projector: rows as stored (the Q8 run covered
full audio; its alignment is source-clock true).

Usage (repo root): PYTHONPATH=. ./.venv/Scripts/python.exe
  experiments/psem_decision_sufficiency/provenance/q8_discriminator.py
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
BOUNDARY = 33469760  # NP2 GT boundary b, frozen (root FREEZE NP2.boundary_samples)
EP = "ami_ES2009d:A00271"
COLLAPSE_START = 13453  # audited ES2009d tail_to_last block start
LAST = 18790


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(1 << 20)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    from experiments.psem_frozen_ceiling_gate.build_ceiling_examples import (
        load_sessions,
    )
    from experiments.psem_sortformer_adaptation_depth.frame_alignment import (
        mapping_from_action_probabilities,
    )

    q8_path = (ROOT / "experiments" / "psem_frozen_ceiling_gate"
               / "frozen_inputs" / "posterior_sessions.npz")
    store = np.load(q8_path, allow_pickle=True)
    probs = np.asarray(store["s008_probabilities"], dtype=np.float64)
    alive = np.asarray(store["s008_alive"], dtype=bool)
    assert probs.shape == (18791, 4)

    sessions = {s.source_id: s for s in load_sessions()}
    session = sessions["ami_ES2009d"]
    starts = np.asarray(session.starts, dtype=np.int64)
    ends = np.asarray(session.ends, dtype=np.int64)
    assert int(ends[-1]) == 33838400 and len(starts) == 18791

    slots, mapping_rows = mapping_from_action_probabilities(session, probs, alive)
    anchor_slot = int(slots[EP])
    mapped = sum(1 for r in mapping_rows if r["status"] == "mapped")

    corrected = probs.copy()
    broken = probs.copy()
    broken[COLLAPSE_START:] = probs[LAST]

    lo, hi = 18600, 18641  # neighborhood covering all 12 A00271 frames
    ep_frames = [i for i in range(lo, hi)
                 if str(list(session.episode_ids)[i]) == EP]
    assert ep_frames == list(range(18615, 18627)), ep_frames
    left = [i for i in ep_frames if int(ends[i]) <= BOUNDARY]
    right = [i for i in ep_frames if int(starts[i]) >= BOUNDARY]
    straddle = [i for i in ep_frames if i not in left and i not in right]
    assert len(left) + len(straddle) + len(right) == len(ep_frames), (left, straddle, right)
    def track(rows):
        return [float(rows[i, anchor_slot]) for i in range(lo, hi)]

    corr, brok = track(corrected), track(broken)
    corr_ep = [float(corrected[i, anchor_slot]) for i in ep_frames]
    brok_ep = [float(broken[i, anchor_slot]) for i in ep_frames]
    corr_left = float(np.mean([corrected[i, anchor_slot] for i in left]))
    corr_right = float(np.mean([corrected[i, anchor_slot] for i in right]))

    out = {
        "label": "INDEPENDENT_OBSERVATION (Q8 streaming posteriors; "
                 "never the same upstream feature as #121 NeMo scores)",
        "inputs": {
            "posterior_sessions.npz": sha256_file(q8_path),
            "mechanism": "transcribe.cpp Q8_0 streaming, 480ms chunks, "
                         "1040ms lookahead (frozen_inputs receipt)",
        },
        "observation": "s008 ami_ES2009d, 18791 action-aligned rows",
        "mapping": {"episode": EP, "anchor_slot": anchor_slot,
                    "episodes_mapped": mapped, "episodes_total": len(mapping_rows)},
        "projectors": {
            "broken_exact_original": f"rows[{COLLAPSE_START}..{LAST}] = rows[{LAST}] "
                                     "(audited DEV saturation block)",
            "corrected": "stored rows unmodified (full-audio Q8 alignment)",
        },
        "window": {"frame_range": [lo, hi - 1], "episode_frames": ep_frames,
                   "boundary": BOUNDARY, "left_frames": left,
                   "straddle_frames_intact_uncertain": straddle, "right_frames": right},
        "corrected_selected_posterior": corr,
        "broken_selected_posterior": brok,
        "episode_summary": {
            "corrected_range": float(max(corr_ep) - min(corr_ep)),
            "corrected_left_mean": corr_left,
            "corrected_right_mean": corr_right,
            "corrected_drop_left_to_right": corr_left - corr_right,
            "broken_range": float(max(brok_ep) - min(brok_ep)),
            "transition_restored": bool(
                (max(brok_ep) - min(brok_ep)) == 0.0
                and (max(corr_ep) - min(corr_ep)) > 0.05
                and corr_left > corr_right),
        },
        "non_claims": [
            "not NeMo scores, not repaired H predictions (head/cache missing)",
            "not a causal real-time proof (arrival timing unmeasured)",
            "slot mapping computed on corrected rows with the frozen rule",
        ],
    }
    dest = HERE / "q8_discriminator.json"
    dest.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {dest} ({dest.stat().st_size} bytes)")
    print("transition_restored:",
          out["episode_summary"]["transition_restored"],
          "drop:", round(out["episode_summary"]["corrected_drop_left_to_right"], 4))


if __name__ == "__main__":
    main()
