"""Soniox realtime reference benchmark capture v2: boring direct websocket, own file.

Repairs v1 protocol bug (EOS never physically sent; quiesce treated as
finish). v2 sends one physical empty binary EOS then waits for server
finished:true inside a 15s bound; quiesce is never success.
Production provider untouched.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import time
import wave
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments" / "soniox_realtime_reference_benchmark"
CAP = EXP / "captures"
FREEZE = json.loads((EXP / "FREEZE.json").read_text(encoding="utf-8"))

CHUNK = 512
CONNECT_TIMEOUT_S = 10.0
CLOSE_TIMEOUT_S = 10.0
POST_EOS_BOUND_S = 15.0
KEEPALIVE_S = 10.0
TRAIL_SAMPLES = 48000
MANUAL_PRE_ZEROS = 3200
KEEP_FIELDS = ("text", "start_ms", "end_ms", "is_final", "language", "confidence", "speaker")


def load_soniox_key() -> str:
    for line in (ROOT / ".env.local").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("export "):
            line = line[len("export "):]
        if line.startswith("SONIOX_API_KEY="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise RuntimeError("SONIOX_API_KEY missing")


def load_source(case: str) -> tuple[np.ndarray, int, str, Path]:
    spec = FREEZE["cases"][case]
    path = Path(spec["audio_path"])
    p0, p1 = 0, 0
    if "payload_samples" in spec:
        p0, p1 = (int(v) for v in spec["payload_samples"])
        expect = FREEZE["input_hashes_sha256"][f"{case}_payload_sha256"]
    else:
        p0, p1 = (int(v) for v in spec["source_samples"])
    with wave.open(str(path), "rb") as r:
        assert r.getframerate() == 16000 and r.getnchannels() == 1 and r.getsampwidth() == 2
        total = r.getnframes()
        assert 0 <= p0 < p1 <= total, f"payload out of range ({p0},{p1} vs {total})"
        r.setpos(p0)
        raw = r.readframes(p1 - p0)
    if "payload_samples" not in spec:
        expect = FREEZE["input_hashes_sha256"]["EN2009d_source_slice_sha256"]
    digest = hashlib.sha256(raw).hexdigest()
    assert digest == expect, f"source bytes differ from freeze {digest} != {expect}"
    pcm = np.frombuffer(raw, dtype=np.int16).copy()
    assert len(pcm) == p1 - p0
    return pcm, p0, digest, path


def journal(event: dict) -> None:
    CAP.mkdir(parents=True, exist_ok=True)
    with (CAP / "attempt_journal.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(event, ensure_ascii=False) + "\n")
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


def budget_allows(session_id: str) -> tuple[bool, str]:
    es = [e for e in journal_entries()
          if e.get("freeze_id") == FREEZE["freeze_id"] and e.get("event") == "attempt-journaled"]
    if len(es) >= 8:
        return False, f"v2 budget spent {len(es)}/8"
    if any(e.get("session_id") == session_id for e in es):
        return False, f"{session_id} already attempted no retries"
    return True, "ok"


def sanitize_token(token: dict, dropped: set) -> dict:
    rec: dict = {}
    for key in KEEP_FIELDS:
        if key in token:
            rec[key] = token[key]
    for key in token:
        if key not in KEEP_FIELDS:
            dropped.add(str(key))
    return rec


async def capture_session(case: str, profile: str) -> Path:
    import websockets

    session_id = f"{case}-{profile}"
    ok, reason = budget_allows(session_id)
    if not ok:
        raise SystemExit(f"budget refused for {session_id}: {reason}")
    payload, p0, digest, path = load_source(case)
    trail = np.zeros(TRAIL_SAMPLES, dtype=np.int16)

    journal({"freeze_id": FREEZE["freeze_id"], "session_id": session_id,
             "event": "attempt-journaled",
             "utc": datetime.now(timezone.utc).isoformat(timespec="seconds")})
    key = load_soniox_key()
    assert len(key) == FREEZE["secrets"]["key_length"], "key length mismatch"
    arm_start = time.monotonic()
    arm_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")

    config = {"api_key": key,
              "model": FREEZE["config_sent"]["model"],
              "audio_format": FREEZE["config_sent"]["audio_format"],
              "sample_rate": FREEZE["config_sent"]["sample_rate"],
              "num_channels": FREEZE["config_sent"]["num_channels"],
              "enable_endpoint_detection": FREEZE["config_sent"]["enable_endpoint_detection"],
              "language_hints": list(FREEZE["config_sent"]["language_hints"]),
              "enable_speaker_diarization": True}
    del key

    chunks: list = []
    sends: list = []
    dropped: set = set()
    n_fin_end = 0
    server_revision = None
    failure: dict | None = None
    first_send_wall: float | None = None

    try:
        ws = await asyncio.wait_for(
            websockets.connect(FREEZE["endpoint"], ping_interval=None,
                               open_timeout=CONNECT_TIMEOUT_S),
            timeout=CONNECT_TIMEOUT_S + 5)
    except Exception as exc:
        journal({"freeze_id": FREEZE["freeze_id"], "session_id": session_id,
                 "event": "pre-connect-failure",
                 "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                 "error": type(exc).__name__, "detail": str(exc)[:300]})
        dest = CAP / f"{session_id}.failure.json"
        dest.write_text(json.dumps({"session_id": session_id,
                                    "freeze_id": FREEZE["freeze_id"],
                                    "captured_at_utc": arm_utc, "status": "pre-connect-failure",
                                    "error": type(exc).__name__,
                                    "detail": str(exc)[:500]}, indent=1) + "\n",
                        encoding="utf-8")
        raise SystemExit(f"{session_id} pre-connect FAILED {type(exc).__name__} -> {dest}")

    open_wall = round(time.monotonic() - arm_start, 3)
    journal({"freeze_id": FREEZE["freeze_id"], "session_id": session_id,
             "event": "session-opened",
             "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "open_offset_s": open_wall})
    recv_q: asyncio.Queue = asyncio.Queue()

    async def recv_loop() -> None:
        try:
            async for msg in ws:
                recv_q.put_nowait((time.monotonic(), msg))
        except Exception as exc:
            recv_q.put_nowait((time.monotonic(), exc))
        finally:
            recv_q.put_nowait((time.monotonic(), None))

    async def drain_into_log(log: list) -> None:
        nonlocal n_fin_end, server_revision
        while not recv_q.empty():
            wall, msg = recv_q.get_nowait()
            if msg is None or isinstance(msg, Exception):
                log.append({"wall": wall, "error": type(msg).__name__ if msg else "closed"})
                continue
            t0 = time.monotonic()
            try:
                text = msg.decode("utf-8", errors="ignore") if isinstance(msg, bytes) else msg
                data = json.loads(text)
            except ValueError:
                data = {}
            proc_ms = round((time.monotonic() - t0) * 1000, 3)
            if not isinstance(data, dict):
                data = {}
            if server_revision is None:
                for k in ("server_revision", "model_revision", "revision"):
                    if k in data:
                        server_revision = str(data[k])
            toks = data.get("tokens")
            rec: dict = {"arrival_from_open": round(wall - arm_start, 3),
                         "arrival_wall": None, "proc_ms": proc_ms,
                         "has_tokens": bool(isinstance(toks, list) and toks),
                         "n_tokens": len(toks) if isinstance(toks, list) else 0,
                         "other_keys": sorted(k for k in data if k != "tokens")}
            if "finished" in data:
                rec["finished"] = data["finished"]
            if isinstance(toks, list) and toks:
                sanitized = []
                for token in toks:
                    if not isinstance(token, dict):
                        continue
                    if str(token.get("text", "") or "") in ("<fin>", "<end>"):
                        n_fin_end += 1
                    sanitized.append(sanitize_token(token, dropped))
                rec["tokens"] = sanitized
            log.append(rec)

    async def send_chunk(frag: bytes, idx: int, src0: int) -> None:
        t0 = time.monotonic()
        await ws.send(frag)
        t1 = time.monotonic()
        chunks.append({"idx": idx, "src_range": [src0, src0 + len(frag) // 2],
                       "sched_wall": None, "send_start": round(t0 - arm_start, 3),
                       "send_end": round(t1 - arm_start, 3)})

    finalize_wall = None
    finalize_delivered_wait_s = 0.0
    eos_send_start = None
    eos_send_end = None
    finished_wall = None
    finished_value = None
    status = "unknown"
    msg_log: list = []
    try:
        async with asyncio.timeout(900):
            await ws.send(json.dumps(config))
            recv_task = asyncio.create_task(recv_loop())
            t0 = time.monotonic()
            first_send_wall = t0
            idx = 0

            async def paced(frag: bytes, off: int, src0: int) -> None:
                target = t0 + off / 16000.0
                delay = target - time.monotonic()
                if delay > 0:
                    await asyncio.sleep(delay)
                await send_chunk(frag, idx, src0)
                chunks[-1]["sched_wall"] = round(target - t0, 3)
                await drain_into_log(msg_log)

            async def wait_delivered(end_off: int) -> float:
                due = t0 + end_off / 16000.0
                now = time.monotonic()
                if due > now:
                    await asyncio.sleep(due - now)
                    return round(due - now, 3)
                return 0.0

            n_src = (len(payload) + CHUNK - 1) // CHUNK
            for i in range(n_src):
                off = i * CHUNK
                frag = payload[off:off + CHUNK]
                idx = i
                await paced(frag.tobytes(), off, p0 + off)
            delivered = len(payload)

            if profile == "MANUAL":
                pre = trail[:MANUAL_PRE_ZEROS]
                for i in range(0, len(pre), CHUNK):
                    frag = pre[i:i + CHUNK]
                    idx += 1
                    await paced(frag.tobytes(), delivered + i, p0 + delivered + i)
                delivered += len(pre)
                finalize_delivered_wait_s = await wait_delivered(delivered)
                await drain_into_log(msg_log)
                await ws.send(json.dumps({"type": "finalize"}))
                finalize_wall = round(time.monotonic() - t0, 3)
                sends.append({"kind": "finalize", "wall": finalize_wall,
                              "delivered_off": delivered,
                              "delivered_wait_s": finalize_delivered_wait_s})
                await drain_into_log(msg_log)
                rest = trail[MANUAL_PRE_ZEROS:]
                base = delivered
                for i in range(0, len(rest), CHUNK):
                    frag = rest[i:i + CHUNK]
                    idx += 1
                    await paced(frag.tobytes(), base + i, p0 + base + i)
                delivered = base + len(rest)
            else:
                for i in range(0, len(trail), CHUNK):
                    frag = trail[i:i + CHUNK]
                    idx += 1
                    await paced(frag.tobytes(), len(payload) + i, p0 + len(payload) + i)
                delivered = len(payload) + len(trail)

            s0 = time.monotonic()
            await ws.send(b"")
            s1 = time.monotonic()
            eos_send_start = round(s0 - t0, 3)
            eos_send_end = round(s1 - t0, 3)
            sends.append({"kind": "eos-empty-binary", "send_start": eos_send_start,
                          "send_end": eos_send_end, "delivered_off": delivered})
            deadline = time.monotonic() + POST_EOS_BOUND_S
            last_send_at = time.monotonic()
            while time.monotonic() < deadline:
                await drain_into_log(msg_log)
                hit = [m for m in msg_log if m.get("finished") is True]
                if hit:
                    finished_wall = hit[0]["arrival_from_open"] - (first_send_wall - arm_start)
                    finished_wall = round(finished_wall, 3)
                    finished_value = True
                    status = "complete"
                    break
                await asyncio.sleep(0.05)
                now = time.monotonic()
                if now - last_send_at >= KEEPALIVE_S:
                    try:
                        await ws.send(json.dumps({"type": "keepalive"}))
                        last_send_at = time.monotonic()
                        sends.append({"kind": "keepalive",
                                      "wall": round(last_send_at - t0, 3)})
                    except Exception:
                        break
            else:
                status = "incomplete"
            await drain_into_log(msg_log)
            if status == "unknown":
                status = "incomplete"
            if status == "incomplete":
                journal({"freeze_id": FREEZE["freeze_id"], "session_id": session_id,
                         "event": "post-eos-incomplete",
                         "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                         "detail": "finished:true not observed within 15s bound"})
            recv_task.cancel()
            await asyncio.gather(recv_task, return_exceptions=True)
    except Exception as exc:
        failure = {"error": type(exc).__name__, "detail": str(exc)[:500]}
        status = "failed"
        journal({"freeze_id": FREEZE["freeze_id"], "session_id": session_id,
                 "event": "post-connect-failure",
                 "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                 **failure})

    for m in msg_log:
        if m.get("arrival_from_open") is not None:
            m["arrival_wall"] = (round(m["arrival_from_open"] - (first_send_wall - arm_start), 3)
                                 if first_send_wall is not None else None)
    try:
        await asyncio.wait_for(ws.close(), timeout=CLOSE_TIMEOUT_S)
        close_note = "ok"
    except Exception as exc:
        close_note = type(exc).__name__
    close_wall = round(time.monotonic() - arm_start, 3)

    out = {
        "session_id": session_id, "case": case, "profile": profile,
        "freeze_id": FREEZE["freeze_id"], "captured_at_utc": arm_utc,
        "status": status,
        "audio": {"path": str(path), "src0": p0, "src1": p0 + len(payload),
                  "src_sha256": digest, "trailing_zeros": TRAIL_SAMPLES,
                  "chunk_samples": CHUNK, "pacing": "realtime-32ms-start-quantized-32ms"},
        "session": {"endpoint": FREEZE["endpoint"], "model_requested": "stt-rt-v5",
                    "model_server_revision": server_revision or "unknown",
                    "config_sent_redacted": {k: (v if k != "api_key" else "<redacted>")
                                             for k, v in config.items()},
                    "open_offset_s": open_wall,
                    "first_send_offset_s": round(first_send_wall - arm_start, 3),
                    "finalize_wall": finalize_wall,
                    "finalize_delivered_wait_s": finalize_delivered_wait_s,
                    "eos_send_start": eos_send_start, "eos_send_end": eos_send_end,
                    "finished_wall": finished_wall, "finished_value": finished_value,
                    "close_wall": close_wall, "close_note": close_note,
                    "n_messages": len(msg_log),
                    "n_with_tokens": sum(1 for m in msg_log if m.get("has_tokens")),
                    "n_finished_true": sum(1 for m in msg_log if m.get("finished") is True),
                    "n_fin_end": n_fin_end,
                    "failure": failure},
        "chunk_ledger": chunks, "control_sends": sends,
        "dropped_field_names": sorted(dropped),
        "messages": msg_log,
    }
    dest = CAP / f"{session_id}.json"
    dest.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    journal({"freeze_id": FREEZE["freeze_id"], "session_id": session_id,
             "event": "completed",
             "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "status": status,
             "n_messages": len(msg_log),
             "n_with_tokens": out["session"]["n_with_tokens"],
             "failure": bool(failure)})
    print(f"{session_id} status={status} msgs={len(msg_log)} "
          f"with_tokens={out['session']['n_with_tokens']} fin_end={n_fin_end} "
          f"finished={finished_value} failure={failure} -> {dest}")
    return dest


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--profile", required=True, choices=("NATURAL", "MANUAL"))
    args = ap.parse_args()
    asyncio.run(capture_session(args.case, args.profile))


if __name__ == "__main__":
    main()
