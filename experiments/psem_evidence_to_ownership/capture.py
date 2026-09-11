"""P2 E2O-1 capture helper: at most ONE single-continuous-session attempt per case.

Usage:
    python experiments/psem_evidence_to_ownership/capture.py --case P3-G04
    python experiments/psem_evidence_to_ownership/capture.py --case P2-G03

Each attempt uses the existing provider path (stt-rt-v5, chunk 512,
realtime 32ms pacing, trailing 100ms, one finalize, no hold/preroll/split)
and preserves raw messages incl arrivals/control/finals plus the exact
accepted text/word groups. Failures are recorded, never retried beyond the
frozen budget of two attempts. SONIOX_API_KEY is read from .env.local and
never written to outputs.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import wave
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

EXP = ROOT / "experiments" / "psem_evidence_to_ownership"
FREEZE = json.loads((EXP / "FREEZE.json").read_text(encoding="utf-8"))
PROBE_RUN = ROOT / "experiments" / "psem_evidence_delivery_gap" / "text_partition_probe" / "run.py"

CASE_AUDIO = {
    "P3-G04": ("ES2009a", "ES2009a.Mix-Headset.wav", 9041600, 9148800),
    "P2-G03": ("ES2002b", "ES2002b.Mix-Headset.wav", 6542400, 6660800),
}
CORPUS = Path("C:/Users/salee/.psem-corpus/ami/audio")


def load_probe_module():
    spec = importlib.util.spec_from_file_location("e2o1_probe_run", PROBE_RUN)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_soniox_key() -> str:
    for line in (ROOT / ".env.local").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("export "):
            line = line[len("export "):]
        if line.startswith("SONIOX_API_KEY="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise RuntimeError("SONIOX_API_KEY missing")


async def capture_case(case: str) -> Path:
    from puripuly_heart.providers.stt.soniox import SonioxRealtimeSTTBackend

    if case not in CASE_AUDIO:
        raise SystemExit(f"unknown case {case}; want one of {sorted(CASE_AUDIO)}")
    meeting, fname, p0, p1 = CASE_AUDIO[case]
    path = CORPUS / meeting / fname
    with wave.open(str(path), "rb") as r:
        assert r.getframerate() == 16000 and r.getnchannels() == 1
        total = r.getnframes()
        assert 0 <= p0 < p1 <= total, f"payload out of range ({p0},{p1} vs {total})"
        r.setpos(p0)
        raw = r.readframes(p1 - p0)
    pcm = np.frombuffer(raw, dtype=np.int16).copy()
    assert len(pcm) == p1 - p0
    payload_sha = hashlib.sha256(raw).hexdigest()

    probe = load_probe_module()
    backend = SonioxRealtimeSTTBackend(api_key=load_soniox_key(), language_hints=["en"])
    if int(getattr(backend, "trailing_silence_ms", 100)) != 100:
        backend.trailing_silence_ms = 100
    sess = await probe.run_session(backend, pcm, p0)
    usable = bool(sess["final_text"]) and any(
        s["emitted"] and s["post"] for s in sess["emit_snapshots"]
    )
    if usable:
        tokens, trace = probe.accepted_stream(sess["emit_snapshots"])
        assert "".join(t["text"] for t in tokens) == sess["final_text"]
        groups = probe.build_groups(sess["final_text"], tokens, p0)
        assert "".join(g["text"] for g in groups) == sess["final_text"]
    else:
        tokens, trace, groups = [], {"note": "no-usable-final"}, []
    out = {
        "case": case,
        "freeze_id": FREEZE["freeze_id"],
        "captured_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "audio": {"path": str(path), "payload_samples": [p0, p1], "payload_sha256": payload_sha},
        "session": {
            "model": "stt-rt-v5", "chunk_samples": 512, "paced": "realtime-32ms",
            "hold_s": 0.0, "provider_sessions": 1,
            "session_open_s": sess["session_open_s"],
            "trailing_silence_ms": sess["trailing_silence_ms"],
            "trailing_silence_samples": sess["trailing_silence_samples"],
            "finalize_wall": sess["finalize_wall"], "final_wall": sess["final_wall"],
            "final_latency_s": sess["final_latency_s"], "final_status": sess["final_status"],
            "nfinal_post": sess["nfinal_post"],
            "collection_elapsed_s": sess["collection_elapsed_s"], "wall_s": sess["wall_s"],
            "translations": 0, "openrouter_calls": 0,
        },
        "raw_evidence": {
            "n_raw_tokens": len(sess["raw_tokens"]),
            "raw_msg_with_tokens": sess["raw_msg_with_tokens"],
            "raw_msg_without_tokens": sess["raw_msg_without_tokens"],
            "control_fin_end": sess["control_fin_end"],
            "dropped_field_names": sess["dropped_field_names"],
            "empty_ack_count": sess["empty_ack_count"],
            "raw_tokens": sess["raw_tokens"],
            "emit_snapshots": sess["emit_snapshots"],
        },
        "accepted": {"final_text": sess["final_text"], "n_tokens": len(tokens),
                     "tokens": tokens, "normalization_trace": trace},
        "groups": groups,
        "received_finals": sess["received_finals"],
        "pre_finalize_finals": sess["pre_finalize_finals"],
        "partials_tail": sess["partials_tail"],
    }
    dest = EXP / "captures" / f"{case}.json"
    dest.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"case={case} final_chars={len(sess['final_text'])} tokens={len(tokens)} "
          f"groups={len(groups)} status={sess['final_status']} -> {dest}")
    return dest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True, choices=sorted(CASE_AUDIO))
    args = ap.parse_args()
    import asyncio

    try:
        asyncio.run(capture_case(args.case))
    except Exception as exc:
        dest = EXP / "captures" / f"{args.case}.failure.json"
        dest.write_text(json.dumps({
            "case": args.case, "freeze_id": FREEZE["freeze_id"],
            "captured_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "status": "capture-failed", "error": type(exc).__name__,
            "detail": str(exc)[:500],
        }, indent=1) + "\n", encoding="utf-8")
        print(f"case={args.case} FAILED {type(exc).__name__}: {exc} -> {dest}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
