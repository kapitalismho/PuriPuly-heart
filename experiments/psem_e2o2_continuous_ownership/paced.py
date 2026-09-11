from __future__ import annotations

import ctypes
import json
import os
import socket
import subprocess
import time
from pathlib import Path

PACED_EXE = Path("C:/tmp/psem-e2o2-paced/bin/transcribe-cli.exe")
PACED_DUMP_ROOT = Path("C:/tmp/psem-e2o2-paced-dumps")
EXP = Path(__file__).resolve().parent
ROOT = EXP.parents[1]

class LARGE_INTEGER(ctypes.Structure):
    _fields_ = [("QuadPart", ctypes.c_longlong)]


def _run():
    from experiments.psem_e2o2_continuous_ownership import run as R
    return R

def qpc():
    v = LARGE_INTEGER()
    ctypes.windll.kernel32.QueryPerformanceCounter(ctypes.byref(v))
    return int(v.QuadPart)


def qpf():
    v = LARGE_INTEGER()
    ctypes.windll.kernel32.QueryPerformanceFrequency(ctypes.byref(v))
    return int(v.QuadPart)


def qpc_s(counts, freq, origin):
    return (counts - origin) / float(freq)


class LiveDecoder:
    def __init__(self):
        self.last = None
        self.anchor_slot = None
        self.pending = None
        self.pend_start = None
        self.pend_n = 0
        self.prev_end = None
        self.seg_n = 0
        self.events = []
        self.n_frames_seen = 0
        self.gaps = []

    def ingest_chunk(self, emit_start_frame, rows):
        R = _run()
        new = []
        for j, row in enumerate(rows):
            i = int(emit_start_frame) + j
            self.n_frames_seen += 1
            lab = R.classify_masked(row)
            if lab in ("OVERLAP", "NONE"):
                if self.pending is not None:
                    self.gaps.append({"kind": "overlap-none-reset", "frame": i, "lab": lab})
                self.pending, self.pend_n, self.prev_end = None, 0, None
                continue
            if self.last is None:
                self.last = lab
                self.pending, self.pend_n, self.prev_end = None, 0, None
                if self.anchor_slot is None:
                    self.anchor_slot = lab
                continue
            if lab == self.last:
                self.pending, self.pend_n, self.prev_end = None, 0, None
                continue
            s = i * R.FRAME
            e = (i + 1) * R.FRAME
            if self.prev_end is not None and s != self.prev_end:
                self.pending, self.pend_n = None, 0
                self.prev_end = None
            if self.pending is None or self.pending != lab:
                self.pending = lab
                self.pend_start = s
                self.pend_n = 0
                self.prev_end = s
            dur = e - s
            need = R.CONFIRMATION - self.pend_n
            if dur >= need:
                self.seg_n += 1
                relation = "CURRENT" if (self.anchor_slot is not None and lab == self.anchor_slot) else "OTHER"
                semantic = "CONTINUE_CURRENT" if relation == "CURRENT" else "SEPARATE_OTHER"
                ev = {
                    "event_id": f"e.{self.seg_n}",
                    "boundary": int(self.pend_start),
                    "confirm_frame": int(i),
                    "frontier": int(e),
                    "candidate_slot": int(lab),
                    "relation": relation,
                    "segment_id": f"{relation}-{self.seg_n}",
                    "semantic": semantic,
                    "uncertainty_samples": R.FRAME,
                    "clock": "qpc_receipt",
                    "probs_row": [float(x) for x in row],
                }
                self.events.append(ev)
                new.append(ev)
                self.last = lab
                self.pending, self.pend_n, self.prev_end = None, 0, None
                continue
            self.pend_n += dur
            self.prev_end = e
        return new


def lifetime_exercises():
    R = _run()
    out = {}
    dec = LiveDecoder()
    cap = {"chunk_ledger": [{"src_range": [0, 4000], "flush_end_wall": 1.0}]}
    pay = [0, 4000]
    groups = [
        {"idx": 0, "text": "a", "start_src": 0, "end_src": 1000, "token_refs": [{"o": 0}]},
        {"idx": 1, "text": "b", "start_src": 1000, "end_src": 2000, "token_refs": [{"o": 1}]},
    ]
    invalid = R.r1_project(groups, cap, pay, 7.0, [{"event_id": "x", "boundary": 99999, "semantic": "SEPARATE_OTHER", "avail": 1.0}])
    out["invalid_capture"] = {
        "pass": any(h.get("outcome") == "invalid_scope" for h in invalid["history"]),
        "history": invalid["history"],
    }
    rows = [[1, 0, 0, 0]] * 10 + [[0, 1, 0, 0]] * 1 + [[0, 0, 0, 0]] * 3 + [[0, 1, 0, 0]] * 8
    dec.ingest_chunk(0, rows)
    out["discontinuity"] = {
        "pass": any(g.get("kind") == "overlap-none-reset" for g in dec.gaps) and len(dec.events) >= 1,
        "n_events": len(dec.events),
        "n_gaps": len(dec.gaps),
    }
    r1_a = R.r1_project(groups, cap, pay, 7.0, dec.events)
    cap2 = {"chunk_ledger": [{"src_range": [0, 4000], "flush_end_wall": 0.5}], "session": {"model": "rolled"}}
    r1_b = R.r1_project(groups, cap2, pay, 7.0, dec.events)
    out["rollover"] = {
        "pass": r1_a["n_seals"] == r1_b["n_seals"],
        "note": "physical ASR session replacement does not reset PSEM decoder last-slot or prior events",
        "n_seals": r1_a["n_seals"],
        "decoder_last": dec.last,
    }
    n_pass = sum(1 for v in out.values() if v.get("pass"))
    return {"checks": out, "n_pass": n_pass, "n_total": len(out)}



def _listen():
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    srv.settimeout(900)
    return srv, srv.getsockname()[1]


def run_paced_one(case_id, force=False):
    R = _run()
    spec = R.FREEZE["cases"][case_id]
    source_id = spec["source"]
    pay = spec["payload_samples"]
    owned = EXP / "paced" / case_id
    owned.mkdir(parents=True, exist_ok=True)
    arrivals_p = owned / "arrivals.json"
    if arrivals_p.exists() and not force:
        prev = R.load_json(arrivals_p)
        if prev.get("ok"):
            return prev
    pin = R.FREEZE["prefixes"][source_id]
    wav = Path(pin["wav"])
    dump = PACED_DUMP_ROOT / case_id
    dump.mkdir(parents=True, exist_ok=True)
    srv, port = _listen()
    env = os.environ.copy()
    for k, v in R.FREEZE["profile"]["env"].items():
        env[k] = v
    env["TRANSCRIBE_DUMP_DIR"] = str(dump)
    env["TRANSCRIBE_PSEM_PACE_16KHZ"] = "1"
    env["TRANSCRIBE_PSEM_PACE_FROM_SAMPLE"] = str(int(pay[0]))
    env["TRANSCRIBE_PSEM_PACE_UNTIL_SAMPLE"] = str(int(pay[1] + 20 * R.FRAME))
    env["TRANSCRIBE_PSEM_EVENTS_TCP"] = f"127.0.0.1:{port}"
    env["TRANSCRIBE_PSEM_CAUSAL_FRONTEND"] = "1"
    cmd = [str(PACED_EXE), "-m", str(R.FREEZE["profile"]["model"]), "--backend", "vulkan", str(wav)]
    t0 = time.perf_counter()
    utc0 = R.utc_now()
    so = open(dump / "stdout.txt", "w", encoding="utf-8", errors="replace")
    se = open(dump / "stderr.txt", "w", encoding="utf-8", errors="replace")
    proc = subprocess.Popen(cmd, env=env, stdout=so, stderr=se, cwd=str(dump))
    lines = []
    decoder = LiveDecoder()
    applied = []
    cap_path = Path(spec["capture"])
    if not cap_path.is_absolute():
        cap_path = ROOT / spec["capture"]
    cap = R.load_json(cap_path)
    from experiments.psem_product_translation.capture import corrected_groups_for_capture
    cap_n = R.normalize_capture(cap)
    groups = corrected_groups_for_capture(cap_n)
    terminal, terminal_src = R.terminal_of(case_id, cap_n)
    sync = None
    ready = None
    done = None
    pace_qpc0 = None
    freq = qpf()
    excluded = False
    conn = None
    try:
        conn, _ = srv.accept()
        conn.settimeout(900)
        buf = b""
        live_events = []
        while True:
            if proc.poll() is not None and not buf:
                extra = conn.recv(65536)
                if not extra:
                    break
                buf += extra
            else:
                try:
                    chunk = conn.recv(65536)
                except socket.timeout:
                    if proc.poll() is not None:
                        break
                    continue
                if not chunk:
                    break
                buf += chunk
            while b"\n" in buf:
                raw, buf = buf.split(b"\n", 1)
                if not raw.strip():
                    continue
                receipt = qpc()
                msg = json.loads(raw.decode("utf-8"))
                msg["_receipt_qpc"] = receipt
                lines.append(msg)
                typ = msg.get("type")
                if typ == "sync":
                    sync = msg
                    freq = int(msg.get("qpf") or freq)
                elif typ == "ready":
                    ready = msg
                    pace_qpc0 = int(msg.get("qpc"))
                    excluded = bool(msg.get("excluded_startup_scope"))
                elif typ == "chunk":
                    if pace_qpc0 is None:
                        pace_qpc0 = int(msg.get("pace_qpc0") or msg.get("qpc"))
                    rows = msg.get("probs") or []
                    new_ev = decoder.ingest_chunk(msg.get("emit_start_frame") or 0, rows)
                    emit_qpc = int(msg.get("qpc"))
                    for ev in new_ev:
                        rec_s = qpc_s(receipt, freq, pace_qpc0)
                        em_s = qpc_s(emit_qpc, freq, pace_qpc0)
                        ev["avail"] = rec_s
                        ev["receipt_clip_s"] = rec_s
                        ev["emit_clip_s"] = em_s
                        ev["source_support_s"] = (int(msg.get("raw_support_end_sample") or 0) - pay[0]) / float(R.HZ)
                        ev["qpc_receipt"] = receipt
                        ev["qpc_emit"] = emit_qpc
                        ev["qpf"] = freq
                        ev["pace_qpc0"] = pace_qpc0
                        ev["causal_gpu_availability"] = True
                        live_events.append(ev)
                        snap = R.r1_project(groups, cap_n, pay, terminal, live_events)
                        applied.append({
                            "event_id": ev["event_id"],
                            "receipt_clip_s": rec_s,
                            "emit_clip_s": em_s,
                            "terminal": terminal,
                            "outcome": (snap["history"][-1]["outcome"] if snap["history"] else None),
                            "n_seals": snap["n_seals"],
                            "semantic": ev["semantic"],
                            "boundary": ev["boundary"],
                            "probs_row": ev.get("probs_row"),
                        })
                elif typ == "done":
                    done = msg
        if conn is not None:
            conn.close()
    finally:
        srv.close()
        try:
            proc.wait(timeout=30)
        except Exception:
            proc.kill()
        so.close()
        se.close()
    rc = proc.returncode
    r1_final = R.r1_project(groups, cap_n, pay, terminal, decoder.events)
    r2_final = R.r2_partition(groups, pay, terminal, decoder.events)
    rec = {
        "ok": rc == 0 and sync is not None and not excluded,
        "case": case_id,
        "source_id": source_id,
        "utc0": utc0,
        "wall_s": time.perf_counter() - t0,
        "rc": rc,
        "cmd": cmd,
        "exe": str(PACED_EXE),
        "exe_sha256": R.sha256_file(PACED_EXE),
        "wav_sha256": R.sha256_file(wav),
        "dump": str(dump),
        "port": port,
        "payload": pay,
        "terminal": terminal,
        "terminal_source": terminal_src,
        "sync": sync,
        "ready": ready,
        "done": done,
        "excluded_startup_scope": excluded,
        "n_tcp_lines": len(lines),
        "n_paced_chunks": sum(1 for m in lines if m.get("type") == "chunk"),
        "n_live_events": len(decoder.events),
        "n_applied_ops": len(applied),
        "applied_at_arrival": applied,
        "events": decoder.events,
        "r1": {"n_seals": r1_final["n_seals"], "history": r1_final["history"], "labels": {str(k): v for k, v in r1_final["labels"].items()}},
        "r2": {"labels": {str(k): v for k, v in r2_final["labels"].items()}, "research_only": True},
        "epoch": "QueryPerformanceCounter shared across C++ producer and Python receiver",
        "skip_full_mel": (sync or {}).get("skip_full_mel"),
        "causal_gpu_availability": bool(sync) and not excluded,
        "note": "avail is receiver receipt on QPC since pace_qpc0; emit_clip_s is producer finish; neither is file-mode poll",
    }
    R.write_json(arrivals_p, rec)
    R.write_json(owned / "tcp_lines.json", lines)
    return rec


def run_paced_all(force=False):
    R = _run()
    out = {}
    order = ["R2", "T1", "COMBINED", "SINGLE_ES2009d", "SINGLE_ES2009c", "R1", "NP3", "BC1", "NP1", "NP2"]
    for cid in order:
        out[cid] = run_paced_one(cid, force=force)
        slim = {k: out[cid].get(k) for k in ("ok", "case", "source_id", "wall_s", "n_live_events", "n_paced_chunks", "excluded_startup_scope", "rc")}
        print(json.dumps(slim, indent=1))
    R.write_json(EXP / "paced_summary.json", {
        k: {kk: vv.get(kk) for kk in ("ok", "wall_s", "n_live_events", "n_paced_chunks", "excluded_startup_scope", "exe_sha256")}
        for k, vv in out.items()
    })
    return out
