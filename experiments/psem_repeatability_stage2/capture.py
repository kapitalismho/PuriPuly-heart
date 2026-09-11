"""Stage2 capture helper: exactly ONE single-continuous-session attempt per new case.

Usage:
    python experiments/psem_repeatability_stage2/capture.py --case NP1
    python experiments/psem_repeatability_stage2/capture.py --case NP2
    python experiments/psem_repeatability_stage2/capture.py --case NP3

Each attempt uses the existing provider path (stt-rt-v5, chunk 512,
realtime 32ms pacing, trailing 100ms, one finalize, no hold/pre-roll/split)
and preserves raw messages incl arrivals/control/finals plus the exact
accepted text/word groups. Additionally it instruments ACTUAL websocket
audio-send completion experiment-locally (queue-put is not flush): every
chunk carries its source sample range, scheduled target wall, enqueue wall,
and measured flush start/end walls on the arm monotonic clock.

Budget (frozen): exactly THREE provider sessions across NP1/NP2/NP3. A
crash-safe attempt journal (captures/attempt_journal.jsonl) is appended
BEFORE a session opens. Pre-connect failures with zero provider session do
not consume budget; post-connect failures do and are never retried.
SONIOX_API_KEY is read from .env.local and never written to outputs.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.util
import json
import sys
import time
import wave
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

EXP = ROOT / "experiments" / "psem_repeatability_stage2"
CAP = EXP / "captures"
FREEZE = json.loads((EXP / "FREEZE.json").read_text(encoding="utf-8"))
PROBE_RUN = ROOT / "experiments" / "psem_evidence_delivery_gap" / "text_partition_probe" / "run.py"

CHUNK = 512
DRAIN_TIMEOUT_S = 60.0
QUIESCE_S = 3.0
ARM_TIMEOUT_S = 300.0
CLOSE_TIMEOUT_S = 10.0
CASES = ("NP1", "NP2", "NP3")


def journal(event: dict) -> None:
    CAP.mkdir(parents=True, exist_ok=True)
    line = json.dumps(event, ensure_ascii=False)
    with open(CAP / "attempt_journal.jsonl", "a", encoding="utf-8") as fh:
        fh.write(line + "\n")
        fh.flush()


def journal_entries() -> list:
    p = CAP / "attempt_journal.jsonl"
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def budget_allows(case: str) -> tuple[bool, str]:
    es = [e for e in journal_entries() if e.get("freeze_id") == FREEZE["freeze_id"]]
    mine = [e for e in es if e.get("case") == case]
    if any(e.get("event") == "completed" for e in mine):
        return False, "case already has a completed capture; no replacements"
    if any(e.get("event") == "post-connect-failure" for e in mine):
        return False, "case already consumed its budget with a post-connect failure; no retries"
    opened = sum(1 for e in es if e.get("event") == "session-opened")
    opened_mine = sum(1 for e in mine if e.get("event") == "session-opened")
    if opened_mine == 0 and opened >= 3:
        return False, "budget exhausted: 3 provider sessions already opened"
    return True, "ok"


def load_probe_module():
    spec = importlib.util.spec_from_file_location("s2_probe_run", PROBE_RUN)
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

    ok, reason = budget_allows(case)
    if not ok:
        raise SystemExit(f"budget refused for {case}: {reason}")
    cspec = FREEZE["cases"][case]
    p0, p1 = (int(v) for v in cspec["payload_samples"])
    path = Path(cspec["audio"]["path"])
    with wave.open(str(path), "rb") as r:
        assert r.getframerate() == 16000 and r.getnchannels() == 1
        total = r.getnframes()
        assert 0 <= p0 < p1 <= total, f"payload out of range ({p0},{p1} vs {total})"
        r.setpos(p0)
        raw = r.readframes(p1 - p0)
    pcm = np.frombuffer(raw, dtype=np.int16).copy()
    assert len(pcm) == p1 - p0
    payload_sha = hashlib.sha256(raw).hexdigest()
    assert payload_sha == cspec["payload_sha256"], "payload bytes differ from freeze"

    journal({"freeze_id": FREEZE["freeze_id"], "case": case,
             "event": "attempt-journaled",
             "utc": datetime.now(timezone.utc).isoformat(timespec="seconds")})
    probe = load_probe_module()
    t_open_start = time.monotonic()
    backend = SonioxRealtimeSTTBackend(api_key=load_soniox_key(), language_hints=["en"])
    if int(getattr(backend, "trailing_silence_ms", 100)) != 100:
        backend.trailing_silence_ms = 100
    try:
        session = await backend.open_session()
    except Exception as exc:
        journal({"freeze_id": FREEZE["freeze_id"], "case": case,
                 "event": "pre-connect-failure",
                 "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                 "error": type(exc).__name__, "detail": str(exc)[:300]})
        raise
    arm_t0 = time.monotonic()
    open_offset_s = round(arm_t0 - t_open_start, 3)
    journal({"freeze_id": FREEZE["freeze_id"], "case": case,
             "event": "session-opened",
             "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "open_offset_s": open_offset_s})

    cls = type(session)
    orig_handle = cls._handle_message
    orig_emit = cls._emit_final_text
    orig_put = cls._put_event
    raw_tokens: list = []
    raw_msg_with_tokens = 0
    raw_msg_without_tokens = 0
    control_fin_end = 0
    dropped_fields: set = set()
    emit_snaps: list = []
    empty_acks = 0
    chunk_ledger: list = []
    control_sends: list = []
    pending_audio: list = []

    def tee_handle(self, message) -> None:
        nonlocal raw_msg_with_tokens, raw_msg_without_tokens, control_fin_end
        try:
            text = message.decode("utf-8", errors="ignore") if isinstance(message, bytes) else message
            data = json.loads(text)
        except ValueError:
            data = {}
        tokens = data.get("tokens") if isinstance(data, dict) else None
        if isinstance(tokens, list) and tokens:
            raw_msg_with_tokens += 1
            for token in tokens:
                if not isinstance(token, dict):
                    continue
                if str(token.get("text", "") or "") in ("<fin>", "<end>"):
                    control_fin_end += 1
                rec = probe.sanitize_token(token, dropped_fields)
                rec["arrival_wall"] = round(time.monotonic() - arm_t0, 3)
                raw_tokens.append(rec)
        else:
            raw_msg_without_tokens += 1
        return orig_handle(self, message)

    def tee_emit(self) -> bool:
        pre = [{"text": tok.text, "end_ms": tok.end_ms} for tok in self._final_tokens]
        out = orig_emit(self)
        post = [{"text": tok.text, "end_ms": tok.end_ms} for tok in self._final_tokens]
        emit_snaps.append({"pre": pre, "post": post, "emitted": bool(out),
                           "wall": round(time.monotonic() - arm_t0, 3)})
        return out

    def tee_put(self, event) -> None:
        nonlocal empty_acks
        try:
            is_final = bool(getattr(event, "is_final", False))
            text = str(getattr(event, "text", "") or "")
        except Exception:
            return orig_put(self, event)
        if is_final and text == "":
            empty_acks += 1
        return orig_put(self, event)

    ws = session._ws
    orig_send = ws.send

    async def tee_send(data) -> None:
        start = round(time.monotonic() - arm_t0, 3)
        try:
            return await orig_send(data)
        finally:
            end = round(time.monotonic() - arm_t0, 3)
            if isinstance(data, bytes):
                if pending_audio and len(data) != 3200:
                    rec = pending_audio.pop(0)
                    rec.update({"flush_start_wall": start, "flush_end_wall": end,
                                "nbytes": len(data), "kind": "audio"})
                    chunk_ledger.append(rec)
                else:
                    control_sends.append({"kind": "trailing-silence" if len(data) == 3200 else "bytes",
                                          "nbytes": len(data),
                                          "flush_start_wall": start, "flush_end_wall": end})
            else:
                try:
                    kind = json.loads(data).get("type", "json") if isinstance(data, str) else "json"
                except ValueError:
                    kind = "text"
                control_sends.append({"kind": str(kind),
                                      "flush_start_wall": start, "flush_end_wall": end})

    cls._handle_message = tee_handle
    cls._emit_final_text = tee_emit
    cls._put_event = tee_put
    ws.send = tee_send  # experiment-local instance wrap; production untouched
    log: list = []

    async def pump() -> None:
        try:
            async for ev in session.events():
                log.append({"wall": time.monotonic(), "text": ev.text,
                            "is_final": bool(ev.is_final)})
        except Exception as exc:
            log.append({"wall": time.monotonic(), "error": type(exc).__name__,
                        "detail": str(exc)[:200]})

    pump_task = asyncio.create_task(pump())
    post_connect_failed = False
    try:
        async with asyncio.timeout(ARM_TIMEOUT_S):
            t0 = time.monotonic()
            n_chunks = (len(pcm) + CHUNK - 1) // CHUNK
            for idx in range(n_chunks):
                off = idx * CHUNK
                frag = pcm[off:off + CHUNK]
                target = t0 + (off / CHUNK) * (CHUNK / 16000.0)
                delay = target - time.monotonic()
                if delay > 0:
                    await asyncio.sleep(delay)
                sched_wall = round(target - arm_t0, 3)
                enq_wall = round(time.monotonic() - arm_t0, 3)
                pending_audio.append({"idx": idx,
                                      "src_range": [p0 + off, p0 + off + len(frag)],
                                      "sched_wall": sched_wall,
                                      "enqueue_wall": enq_wall})
                await session.send_audio(frag.tobytes())
            finalize_wall = round(time.monotonic() - arm_t0, 3)
            await session.on_speech_end()
            fw = arm_t0 + finalize_wall
            deadline = time.monotonic() + DRAIN_TIMEOUT_S
            while time.monotonic() < deadline:
                posts = [e for e in log if e.get("is_final") and "error" not in e and e["wall"] >= fw]
                if posts and time.monotonic() - posts[-1]["wall"] >= QUIESCE_S:
                    break
                await asyncio.sleep(0.05)
            post_finals = [e for e in log if e.get("is_final") and "error" not in e and e["wall"] >= fw]
            pre_finals = [e["text"] for e in log if e.get("is_final") and "error" not in e and e["wall"] < fw]
            nonempty = [e for e in post_finals if e["text"]]
            sel = nonempty[-1] if nonempty else (post_finals[-1] if post_finals else None)
            collection_elapsed_s = round(time.monotonic() - (arm_t0 + finalize_wall), 3)
    except Exception as exc:
        post_connect_failed = True
        journal({"freeze_id": FREEZE["freeze_id"], "case": case,
                 "event": "post-connect-failure",
                 "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                 "error": type(exc).__name__, "detail": str(exc)[:300]})
        dest = CAP / f"{case}.failure.json"
        dest.write_text(json.dumps({
            "case": case, "freeze_id": FREEZE["freeze_id"],
            "captured_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "status": "capture-failed", "error": type(exc).__name__,
            "detail": str(exc)[:500],
        }, indent=1) + "\n", encoding="utf-8")
        raise SystemExit(f"case={case} FAILED post-connect {type(exc).__name__} -> {dest}")
    finally:
        cls._handle_message = orig_handle
        cls._emit_final_text = orig_emit
        cls._put_event = orig_put
        try:
            await asyncio.wait_for(session.close(), timeout=CLOSE_TIMEOUT_S)
        except Exception:
            pass
        pump_task.cancel()
        await asyncio.gather(pump_task, return_exceptions=True)

    if post_connect_failed:
        raise SystemExit(1)
    usable = bool(sel and sel["text"]) and any(s["emitted"] and s["post"] for s in emit_snaps)
    if usable:
        tokens, trace = probe.accepted_stream(emit_snaps)
        assert "".join(t["text"] for t in tokens) == sel["text"]
        groups = probe.build_groups(sel["text"], tokens, p0)
        assert "".join(g["text"] for g in groups) == sel["text"]
    else:
        tokens, trace, groups = [], {"note": "no-usable-final"}, []
    received = [{"text": e["text"], "rel_wall": round(e["wall"] - arm_t0, 3),
                 "after_finalize": e["wall"] >= fw} for e in log if e.get("is_final") and "error" not in e]
    out = {
        "case": case,
        "freeze_id": FREEZE["freeze_id"],
        "captured_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "audio": {"path": str(path), "payload_samples": [p0, p1], "payload_sha256": payload_sha},
        "session": {
            "model": "stt-rt-v5", "chunk_samples": CHUNK, "paced": "realtime-32ms",
            "hold_s": 0.0, "provider_sessions": 1,
            "open_offset_s": open_offset_s,
            "trailing_silence_ms": 100,
            "trailing_silence_samples": 1600,
            "finalize_wall": finalize_wall,
            "seal_wall": round(sel["wall"] - arm_t0, 3) if sel else None,
            "final_latency_s": round(sel["wall"] - (arm_t0 + finalize_wall), 3) if sel else None,
            "final_status": ("ok" if sel and sel["text"]
                             else ("empty-ack" if sel else "timeout")),
            "nfinal_post": len(post_finals),
            "collection_elapsed_s": collection_elapsed_s,
            "wall_s": round(time.monotonic() - arm_t0, 3),
            "translations": 0, "openrouter_calls": 0,
        },
        "chunk_ledger": chunk_ledger,
        "control_sends": control_sends,
        "flush_note": ("queue-put is not flush: sched_wall is the paced target, enqueue_wall "
                       "is queue-put, flush_end_wall is measured websocket send completion. "
                       "Evidence frontier of a sample = flush_end_wall of its chunk; sample "
                       "receipt (enqueue side) is recorded separately and never equated."),
        "raw_evidence": {
            "n_raw_tokens": len(raw_tokens),
            "raw_msg_with_tokens": raw_msg_with_tokens,
            "raw_msg_without_tokens": raw_msg_without_tokens,
            "control_fin_end": control_fin_end,
            "dropped_field_names": sorted(dropped_fields),
            "raw_tokens": raw_tokens,
            "emit_snapshots": emit_snaps,
        },
        "accepted": {"final_text": sel["text"] if sel else "", "n_tokens": len(tokens),
                     "tokens": tokens, "normalization_trace": trace},
        "groups": groups,
        "received_finals": received,
        "pre_finalize_finals": pre_finals,
        "partials_tail": [e["text"] for e in log if not e.get("is_final")][-3:],
    }
    dest = CAP / f"{case}.json"
    dest.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    journal({"freeze_id": FREEZE["freeze_id"], "case": case, "event": "completed",
             "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "final_chars": len(sel["text"]) if sel else 0, "tokens": len(tokens),
             "groups": len(groups), "status": out["session"]["final_status"]})
    print(f"case={case} final_chars={len(sel['text']) if sel else 0} tokens={len(tokens)} "
          f"groups={len(groups)} chunks_flushed={len(chunk_ledger)} "
          f"status={out['session']['final_status']} -> {dest}")
    return dest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True, choices=list(CASES))
    args = ap.parse_args()
    asyncio.run(capture_case(args.case))


if __name__ == "__main__":
    main()
