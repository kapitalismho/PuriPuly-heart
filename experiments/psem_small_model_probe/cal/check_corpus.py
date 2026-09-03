"""Verify local corpus audio for frozen probe sessions (owned by psem-corpus).

Usage (from repo root):
  set PSEM_CORPUS_ROOT=C:\\Users\\salee\\.psem-corpus
  python experiments/psem_small_model_probe/cal/check_corpus.py [--sha-spans N]
"""
import hashlib
import json
import sys
import wave
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "experiments" / "psem_small_model_probe" / "cal"))
from audio_resolve import (  # noqa: E402
    SAMPLE_RATE_HZ,
    audio_ref_for_row,
    load_span,
    resolve_audio,
    span_for_regime,
)

MANIFEST = REPO / "experiments/psem_small_model_probe/manifest/manifest.jsonl"


def session_rows():
    first = {}
    order = []
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = (row["corpus"], row["session_id"])
        if key not in first:
            first[key] = row
            order.append(key)
    return [first[k] for k in order]


def check_session(row, sha_spans=1):
    ref = audio_ref_for_row(row)
    try:
        path = resolve_audio(row)
    except FileNotFoundError as exc:
        return {"ok": False, "ref": ref, "error": str(exc)}
    try:
        with wave.open(str(path), "rb") as r:
            fmt = (r.getframerate(), r.getnchannels(), r.getsampwidth() * 8)
            frames = r.getnframes()
        assert fmt == (SAMPLE_RATE_HZ, 1, 16), f"bad format {fmt}"
        # Native (O) + causal (C) spans for every episode row of this session,
        # plus the max evaluation_end_ms across episodes (eval window bound).
        eval_end = 0
        for line in MANIFEST.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            ep = json.loads(line)
            if (ep["corpus"], ep["session_id"]) != (row["corpus"], row["session_id"]):
                continue
            for regime in ("O", "C"):
                s, e = span_for_regime(ep, regime)
                load_span(path, s, e)
            eval_end = max(eval_end, int(ep["evaluation_end_ms"]))
        total_ms = frames * 1000 // SAMPLE_RATE_HZ
        assert eval_end * 16 <= frames, f"eval overrun {eval_end}ms > {total_ms}ms"
        shas = []
        if sha_spans:
            s, e = span_for_regime(row, "O")
            data = load_span(path, s, e)
            shas.append(hashlib.sha256(data).hexdigest()[:16])
        return {"ok": True, "ref": ref, "path": str(path),
                "frames": frames, "eval_end_ms": eval_end, "sha16": shas}
    except Exception as exc:  # fail-closed: report, don't raise
        return {"ok": False, "ref": ref, "path": str(path), "error": str(exc)}


def main():
    rows = session_rows()
    print(f"sessions={len(rows)}")
    n_ok = 0
    for row in rows:
        res = check_session(row)
        n_ok += res["ok"]
        status = "PASS" if res["ok"] else "FAIL"
        extra = res.get("path", res.get("error"))
        print(f"[{status}] {row['corpus']}/{row['session_id']} ref={res['ref']} {extra}")
    print(f"resolved={n_ok}/{len(rows)}")
    return 0 if n_ok == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
