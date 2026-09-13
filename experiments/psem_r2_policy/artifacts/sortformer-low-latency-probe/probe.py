from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import socket
import statistics
import subprocess
import sys
import time
import wave
from collections import Counter
from decimal import Decimal
from pathlib import Path
from typing import Any

import ijson

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
EXP = ROOT / "experiments/psem_r2_policy"
CAPSULE = EXP / ".capsule/fa94e3ec3f5e"
RAW = EXP / "artifacts/dev/ES2009d/20260912T130134673943Z.json"
PACING = EXP / "artifacts/dev-supervised/cases/ES2009d/attempt-2/pacing.json"
AUDIO = Path(r"C:/Users/salee/AppData/Local/Temp/opencode/stb_phase2_corpora/ami/audio/ES2009d/ES2009d.Mix-Headset.wav")
EXE = Path(r"C:/tmp/psem-e2o2-paced/bin/transcribe-cli.exe")
MODEL = Path(r"C:/tmp/psem-vulkan-fp16-model/diar_streaming_sortformer_4spk-v2.1-F16.gguf")
SOURCE_HZ = 16000
FRAME_SAMPLES = 1280
PREFIXES = (180, 300, 600)
TAIL_SECONDS = 16
STREAM_KEYS = (
    "TRANSCRIBE_SORTFORMER_STREAM_PRESET",
    "TRANSCRIBE_SORTFORMER_STREAM_CHUNK_LEN",
    "TRANSCRIBE_SORTFORMER_STREAM_FIFO_LEN",
    "TRANSCRIBE_SORTFORMER_STREAM_SPKCACHE_LEN",
    "TRANSCRIBE_SORTFORMER_STREAM_UPDATE_PERIOD",
    "TRANSCRIBE_SORTFORMER_STREAM_RC",
    "TRANSCRIBE_SORTFORMER_STREAM_LC",
)
PROFILES = {
    "recorded-default": {
        "chunk_len": 188,
        "chunk_left_context": 1,
        "chunk_right_context": 1,
        "fifo_len": 0,
        "spkcache_len": 188,
        "spkcache_update_period": 188,
    },
    "official-low-latency": {
        "chunk_len": 6,
        "chunk_left_context": 1,
        "chunk_right_context": 7,
        "fifo_len": 188,
        "spkcache_len": 188,
        "spkcache_update_period": 144,
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(jsonable(payload), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def activate_capsule() -> None:
    for path in (CAPSULE / "src", CAPSULE):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)


def raw_parents() -> list[dict[str, Any]]:
    keep = {
        "index",
        "parent_id",
        "outcome",
        "terminal_outcome",
        "seal_reason",
        "status",
        "text_authority",
        "failure_reason",
        "text",
        "span",
        "assignment",
        "conserved",
        "marks",
        "tokens",
    }
    result = []
    with RAW.open("rb") as handle:
        for parent in ijson.items(handle, "parents.item"):
            projected = {key: parent.get(key) for key in keep}
            r0 = parent.get("r0") or {}
            projected["r0"] = {
                "contamination": r0.get("contamination"),
                "guard": r0.get("guard"),
            }
            result.append(jsonable(projected))
    return result


def old_clock_fit() -> dict[str, Any]:
    payload = json.loads(PACING.read_text(encoding="utf-8"))
    points = [(float(row[1]), float(row[0])) for row in payload["window_rates"][:0]]
    summary_path = Path(payload["summary_path"])
    if not summary_path.is_file():
        summary_path = EXP / "artifacts/dev-supervised/cases/ES2009d/attempt-2/summary.json"
    with summary_path.open("rb") as handle:
        samples = next(ijson.items(handle, "capture_timing.feed_progress.samples"))
    points = [(float(source_s), float(monotonic_s)) for monotonic_s, source_s in samples]
    n = len(points)
    mean_x = sum(x for x, _ in points) / n
    mean_y = sum(y for _, y in points) / n
    denom = sum((x - mean_x) ** 2 for x, _ in points)
    slope = sum((x - mean_x) * (y - mean_y) for x, y in points) / denom
    intercept = mean_y - slope * mean_x
    residuals = [y - (intercept + slope * x) for x, y in points]
    anchors = [y - x for x, y in points]
    return {
        "source": "capture_timing.feed_progress.samples",
        "summary_path": str(summary_path.relative_to(ROOT)),
        "point_count": n,
        "source_span_s": [points[0][0], points[-1][0]],
        "monotonic_span_s": [points[0][1], points[-1][1]],
        "feed_grid": [[monotonic_s, source_s] for source_s, monotonic_s in points],
        "exact_source_zero_available": False,
        "missing_anchor": "The original DEV artifacts discarded native pace_qpc0/qpf and persist no explicit ASR feed-start. The one-second feed-progress grid brackets source position but cannot establish an exact source-zero/monotonic mapping.",
        "diagnostic_linear_fit_not_an_exact_anchor": {
            "monotonic_intercept_at_source_zero_s": intercept,
            "monotonic_per_source_s": slope,
            "unit_slope_error": slope - 1.0,
            "residual_max_abs_s": max(abs(value) for value in residuals),
            "residual_median_abs_s": statistics.median(abs(value) for value in residuals),
            "direct_zero_first_s": anchors[0],
            "direct_zero_median_s": statistics.median(anchors),
            "direct_zero_last_s": anchors[-1],
            "direct_zero_spread_s": max(anchors) - min(anchors),
        },
    }


def gt_boundaries(words: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    previous = None
    for word in words:
        if previous is not None:
            if word["role"] != previous["role"] and word["start_src"] >= previous["end_src"]:
                result.append({
                    "at_src": int(word["start_src"]),
                    "prev_end_src": int(previous["end_src"]),
                    "from_role": str(previous["role"]),
                    "to_role": str(word["role"]),
                })
        if previous is None or word["end_src"] >= previous["end_src"]:
            previous = word
    return result


def parent_contained(parent: dict[str, Any], limit: int) -> bool:
    span = parent.get("span") or []
    return len(span) == 2 and span[0] is not None and span[1] is not None and 0 <= int(span[0]) <= int(span[1]) <= limit


def prepare() -> None:
    activate_capsule()
    from experiments.psem_r2_policy.metrics import load_ami_words

    parents = raw_parents()
    words = load_ami_words("ES2009d")
    boundaries = gt_boundaries(words)
    candidates = []
    chosen = None
    for seconds in PREFIXES:
        limit = seconds * SOURCE_HZ
        contained = [parent for parent in parents if parent_contained(parent, limit)]
        nonempty = [parent for parent in contained if str(parent.get("text") or "")]
        eligible = [
            parent
            for parent in nonempty
            if bool(((parent.get("r0") or {}).get("contamination") or {}).get("eligible"))
        ]
        boundary_hits = {
            int(boundary["at_src"])
            for boundary in boundaries
            if 0 < int(boundary["at_src"]) <= limit
        }
        r0_contaminated = sum(
            int(((parent.get("r0") or {}).get("contamination") or {}).get("contaminated_chars") or 0)
            for parent in nonempty
        )
        row = {
            "prefix_seconds": seconds,
            "fully_contained_parents": len(contained),
            "fully_contained_nonempty_parents": len(nonempty),
            "paired_scorable_sequential_parents": len(eligible),
            "distinct_nonoverlap_sequential_gt_boundaries_in_prefix": len(boundary_hits),
            "boundary_samples": sorted(boundary_hits),
            "r0_contaminated_chars": r0_contaminated,
            "qualifies": len(nonempty) >= 20 and len(boundary_hits) >= 5 and r0_contaminated > 0,
        }
        candidates.append(row)
        if chosen is None and row["qualifies"]:
            chosen = row
    if chosen is None:
        write_json(HERE / "selection.json", {"status": "insufficient", "candidates": candidates})
        raise SystemExit("No preregistered prefix qualifies")
    selected_samples = int(chosen["prefix_seconds"]) * SOURCE_HZ
    selected_parents = [parent for parent in parents if parent_contained(parent, selected_samples)]
    projection_samples = selected_samples + TAIL_SECONDS * SOURCE_HZ
    projection = HERE / "input-prefix-plus-tail.wav"
    with wave.open(str(AUDIO), "rb") as source:
        params = source.getparams()
        if source.getframerate() != SOURCE_HZ:
            raise RuntimeError(f"unexpected source rate: {source.getframerate()}")
        if projection_samples > source.getnframes():
            projection_samples = source.getnframes()
        frames = source.readframes(projection_samples)
    with wave.open(str(projection), "wb") as target:
        target.setparams(params)
        target.writeframes(frames)
    clock = old_clock_fit()
    payload = {
        "schema": "SORTFORMER-LOW-LATENCY-PROBE-SELECTION-1",
        "status": "selected",
        "selection_rule_interpretation": "The three conjuncts are evaluated independently on frozen old input: fully contained nonempty original parents, distinct non-overlap role changes in the AMI GT word sequence within the prefix, and frozen R0 contaminated characters. GT is not supplied to inference or replay.",
        "candidates": candidates,
        "selected": chosen,
        "evaluation_end_sample": selected_samples,
        "context_tail_seconds": TAIL_SECONDS,
        "projection_real_source_samples": projection_samples,
        "projection": {"path": projection.name, "sha256": sha256(projection), "bytes": projection.stat().st_size},
        "selected_parents": selected_parents,
        "accounting": {
            "all_original_parents": len(parents),
            "selected_fully_contained": len(selected_parents),
            "selected_nonempty": sum(bool(str(parent.get("text") or "")) for parent in selected_parents),
            "excluded_after_prefix_or_crossing_boundary": sum(not parent_contained(parent, selected_samples) for parent in parents),
            "excluded_missing_span": sum(len(parent.get("span") or []) != 2 for parent in parents),
        },
        "recorded_clock": clock,
        "identities": {
            "raw": {"path": str(RAW.relative_to(ROOT)), "sha256": sha256(RAW), "bytes": RAW.stat().st_size},
            "source_audio": {"path": str(AUDIO), "sha256": sha256(AUDIO), "bytes": AUDIO.stat().st_size},
            "exe": {"path": str(EXE), "sha256": sha256(EXE), "bytes": EXE.stat().st_size},
            "model": {"path": str(MODEL), "sha256": sha256(MODEL), "bytes": MODEL.stat().st_size},
            "capsule_manifest": {"path": str((CAPSULE / "capsule_manifest.json").relative_to(ROOT)), "sha256": sha256(CAPSULE / "capsule_manifest.json")},
        },
    }
    write_json(HERE / "selection.json", payload)
    print(json.dumps({"status": "SELECTION_FROZEN", "prefix_seconds": chosen["prefix_seconds"], "parents": len(selected_parents), "projection_sha256": payload["projection"]["sha256"]}))


def profile_env(profile: str, port: int, projection_samples: int, dump_dir: Path) -> tuple[dict[str, str], dict[str, Any]]:
    env = os.environ.copy()
    inherited = {key: env.get(key) for key in STREAM_KEYS if key in env}
    for key in STREAM_KEYS:
        env.pop(key, None)
    config = PROFILES[profile]
    env.update(
        {
            "TRANSCRIBE_PSEM_PACE_16KHZ": "1",
            "TRANSCRIBE_PSEM_EVENTS_TCP": f"127.0.0.1:{port}",
            "TRANSCRIBE_PSEM_CAUSAL_FRONTEND": "1",
            "TRANSCRIBE_PSEM_PACE_FROM_SAMPLE": "0",
            "TRANSCRIBE_PSEM_PACE_UNTIL_SAMPLE": str(projection_samples),
            "TRANSCRIBE_DUMP_DIR": str(dump_dir),
            "TRANSCRIBE_SORTFORMER_STREAM_CHUNK_LEN": str(config["chunk_len"]),
            "TRANSCRIBE_SORTFORMER_STREAM_FIFO_LEN": str(config["fifo_len"]),
            "TRANSCRIBE_SORTFORMER_STREAM_SPKCACHE_LEN": str(config["spkcache_len"]),
            "TRANSCRIBE_SORTFORMER_STREAM_UPDATE_PERIOD": str(config["spkcache_update_period"]),
            "TRANSCRIBE_SORTFORMER_STREAM_RC": str(config["chunk_right_context"]),
            "TRANSCRIBE_SORTFORMER_STREAM_LC": str(config["chunk_left_context"]),
        }
    )
    return env, {"isolated_inherited_stream_overrides": inherited, "requested": config, "applied_env": {key: env[key] for key in STREAM_KEYS if key in env}}


def run_profile(profile: str) -> None:
    selection = json.loads((HERE / "selection.json").read_text(encoding="utf-8"))
    if selection.get("status") != "selected":
        raise RuntimeError("selection is not frozen")
    projection = HERE / selection["projection"]["path"]
    if sha256(projection) != selection["projection"]["sha256"]:
        raise RuntimeError("projection identity changed")
    profile_dir = HERE / profile
    if profile_dir.exists():
        raise RuntimeError(f"profile evidence already exists: {profile_dir}")
    profile_dir.mkdir()
    dump_dir = profile_dir / "dump"
    dump_dir.mkdir()
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    server.settimeout(60)
    port = int(server.getsockname()[1])
    env, environment = profile_env(profile, port, int(selection["projection_real_source_samples"]), dump_dir)
    stdout_path = profile_dir / "native.stdout.log"
    stderr_path = profile_dir / "native.stderr.log"
    started_monotonic = time.monotonic()
    started_epoch = time.time()
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        process = subprocess.Popen([str(EXE), "-m", str(MODEL), "--backend", "vulkan", str(projection)], env=env, stdout=stdout, stderr=stderr)
        try:
            conn, peer = server.accept()
            conn.settimeout(1.0)
            buffer = b""
            messages = []
            ready = None
            done = None
            while True:
                try:
                    chunk = conn.recv(1 << 20)
                except socket.timeout:
                    if process.poll() is not None:
                        chunk = b""
                    else:
                        continue
                if chunk:
                    buffer += chunk
                    while b"\n" in buffer:
                        raw, buffer = buffer.split(b"\n", 1)
                        if not raw.strip():
                            continue
                        message = json.loads(raw.decode("utf-8"))
                        message["_python_receipt_monotonic_s"] = time.monotonic()
                        messages.append(message)
                        if message.get("type") == "ready" and ready is None:
                            ready = message
                            print(f"READY {profile} native_qpc={message['qpc']} qpf={message['qpf']}", flush=True)
                        if message.get("type") == "done":
                            done = message
                else:
                    if process.poll() is not None:
                        break
            exit_code = process.wait(timeout=30)
        finally:
            conn.close() if "conn" in locals() else None
            server.close()
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=10)
    finished_monotonic = time.monotonic()
    if exit_code != 0 or ready is None or done is None:
        raise RuntimeError(f"native run incomplete: exit={exit_code} ready={ready is not None} done={done is not None}")
    native_path = profile_dir / "native-events.jsonl"
    native_path.write_text("".join(json.dumps(item, separators=(",", ":")) + "\n" for item in messages), encoding="utf-8")
    trace_path = dump_dir / "diar.trace.json"
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    for child in dump_dir.iterdir():
        if child != trace_path:
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    chunks = [item for item in messages if item.get("type") == "chunk"]
    effective = {key: trace[key] for key in ("chunk_len", "chunk_left_context", "chunk_right_context", "fifo_len", "spkcache_len", "spkcache_update_period")}
    if effective != PROFILES[profile]:
        raise RuntimeError(f"effective profile mismatch: {effective!r}")
    qpf = float(ready["qpf"])
    qpc0 = int(ready["qpc"])
    receipt_source = [(int(item["qpc"]) - qpc0) / qpf for item in chunks]
    frame_ages = []
    for item, receipt_s in zip(chunks, receipt_source):
        for offset in range(int(item["emit_count"])):
            frame_end = (int(item["emit_start_frame"]) + offset + 1) * FRAME_SAMPLES / SOURCE_HZ
            frame_ages.append(receipt_s - frame_end)
    intervals = [right - left for left, right in zip(receipt_source, receipt_source[1:])]
    support_lag = [receipt_s - float(item["raw_support_end_sample"]) / SOURCE_HZ for item, receipt_s in zip(chunks, receipt_source)]
    run = {
        "schema": "SORTFORMER-LOW-LATENCY-PROFILE-RUN-1",
        "profile": profile,
        "command": [str(EXE), "-m", str(MODEL), "--backend", "vulkan", str(projection)],
        "environment": environment,
        "process": {"started_epoch_s": started_epoch, "started_monotonic_s": started_monotonic, "finished_monotonic_s": finished_monotonic, "wall_s": finished_monotonic - started_monotonic, "exit_code": exit_code, "ready_received": True, "done_received": True, "peer": list(peer)},
        "new_clock": {"source_zero_contract": "native ready.qpc equals pace_qpc0 for pace_from_sample=0", "qpc0": qpc0, "qpf": int(qpf), "ready_python_receipt_monotonic_s": ready["_python_receipt_monotonic_s"], "done_qpc": done["qpc"], "done_source_s": (int(done["qpc"]) - qpc0) / qpf},
        "effective_config": effective,
        "source": {"projection_sha256": sha256(projection), "declared_samples": selection["projection_real_source_samples"], "ready_n_samples": next(item["n_samples"] for item in messages if item.get("type") == "sync"), "maximum_n_visible": max(int(item["n_visible"]) for item in chunks), "maximum_raw_support_end_sample": max(int(item["raw_support_end_sample"]) for item in chunks), "done_stream_total_frames": done["stream_total_n"], "output_frames": sum(int(item["emit_count"]) for item in chunks)},
        "cadence": {"chunk_messages": len(chunks), "emit_count_distribution": dict(Counter(int(item["emit_count"]) for item in chunks)), "first_receipt_source_s": receipt_source[0], "last_receipt_source_s": receipt_source[-1], "interarrival_s": None if not intervals else {"min": min(intervals), "median": statistics.median(intervals), "max": max(intervals)}, "support_lag_s": {"first": support_lag[0], "last": support_lag[-1], "min": min(support_lag), "median": statistics.median(support_lag), "max": max(support_lag), "last_minus_first": support_lag[-1] - support_lag[0]}, "frame_evidence_age_s": {"min": min(frame_ages), "median": statistics.median(frame_ages), "max": max(frame_ages)}},
        "artifacts": {"native_events": {"path": str(native_path.relative_to(HERE)), "sha256": sha256(native_path), "bytes": native_path.stat().st_size}, "trace": {"path": str(trace_path.relative_to(HERE)), "sha256": sha256(trace_path), "bytes": trace_path.stat().st_size}, "stdout": {"path": str(stdout_path.relative_to(HERE)), "sha256": sha256(stdout_path), "bytes": stdout_path.stat().st_size}, "stderr": {"path": str(stderr_path.relative_to(HERE)), "sha256": sha256(stderr_path), "bytes": stderr_path.stat().st_size}},
    }
    write_json(profile_dir / "run.json", run)
    print(json.dumps({"status": "COMPLETED", "profile": profile, "exit_code": exit_code, "chunks": len(chunks), "effective": effective}), flush=True)


def load_native_messages(profile: str) -> list[dict[str, Any]]:
    path = HERE / profile / "native-events.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def unit_rows(units: Any) -> list[dict[str, Any]]:
    return [{"group_id": item.group_id, "relation": item.relation, "text": item.text, "token_indexes": list(item.token_indexes), "start_source_sample": item.start_source_sample, "end_source_sample": item.end_source_sample} for item in units]


def admission_source_sensitivity(admission_monotonic_s: float, feed_grid: list[list[float]]) -> dict[str, float]:
    before = None
    after = None
    for monotonic_s, source_s in feed_grid:
        if float(monotonic_s) <= admission_monotonic_s:
            before = (float(monotonic_s), float(source_s))
        if float(monotonic_s) >= admission_monotonic_s:
            after = (float(monotonic_s), float(source_s))
            break
    if before is None or after is None:
        raise RuntimeError(f"admission outside recorded feed grid: {admission_monotonic_s}")
    if after[0] == before[0]:
        estimate = before[1]
    else:
        weight = (admission_monotonic_s - before[0]) / (after[0] - before[0])
        estimate = before[1] + weight * (after[1] - before[1])
    return {
        "lower_source_s": before[1],
        "piecewise_linear_estimate_source_s": estimate,
        "upper_source_s": after[1],
        "bracket_width_s": after[1] - before[1],
    }

def earliest_full_coverage_s(evidence: tuple[Any, ...], start_sample: int, end_sample: int) -> float | None:
    for cutoff in sorted({float(item.available_at_monotonic_s) for item in evidence}):
        intervals = sorted(
            (max(start_sample, int(item.start_sample)), min(end_sample, int(item.end_sample)))
            for item in evidence
            if float(item.available_at_monotonic_s) <= cutoff
            and int(item.end_sample) > start_sample
            and int(item.start_sample) < end_sample
        )
        frontier = start_sample
        for left, right in intervals:
            if right <= left:
                continue
            if left > frontier:
                break
            frontier = max(frontier, right)
            if frontier >= end_sample:
                return cutoff
    return None



def compare() -> None:
    activate_capsule()
    from experiments.psem_r2_policy.metrics import live_parent_ledger, load_ami_words, pair_parent_guard, score_live_ledger
    from experiments.psem_r2_policy.sortformer_live import LiveTransitionDecoder, hypothesis_from_live_event
    from puripuly_heart.core.audio.pretranslation_ownership import PretranslationEvidence, assign_ownership_units
    from puripuly_heart.core.stt.backend import STTTimedToken

    selection = json.loads((HERE / "selection.json").read_text(encoding="utf-8"))
    feed_grid = selection["recorded_clock"]["feed_grid"]
    words = load_ami_words("ES2009d")
    generation = "bounded-local-continuous-state"
    profile_decoders = {}
    profile_runs = {}
    for profile in PROFILES:
        run = json.loads((HERE / profile / "run.json").read_text(encoding="utf-8"))
        profile_runs[profile] = run
        messages = load_native_messages(profile)
        ready = next(item for item in messages if item.get("type") == "ready")
        qpc0 = int(ready["qpc"])
        qpf = float(ready["qpf"])
        decoder = LiveTransitionDecoder()
        for item in messages:
            if item.get("type") != "chunk":
                continue
            receipt_source_s = (int(item["qpc"]) - qpc0) / qpf
            decoder.ingest_chunk(int(item["emit_start_frame"]), item["probs"], available_at_monotonic_s=receipt_source_s, receipt_kind="native_arrival")
        profile_decoders[profile] = decoder
    totals = {profile: Counter() for profile in ("r0", *PROFILES)}
    for profile, decoder in profile_decoders.items():
        totals[profile]["native_frames"] = decoder.n_frames_seen
        totals[profile]["native_transition_events"] = len(decoder.events)
        totals[profile]["native_evidence_intervals"] = len(decoder.evidence)
    parent_results = []
    severe = {profile: [] for profile in PROFILES}
    for parent in selection["selected_parents"]:
        text = str(parent.get("text") or "")
        tokens_payload = list(parent.get("tokens") or ())
        admission = (parent.get("marks") or {}).get("translation_admission")
        admission_sensitivity = None if admission is None else admission_source_sensitivity(float(admission), feed_grid)
        row = {"parent_id": parent.get("parent_id"), "span": parent.get("span"), "text_chars": len(text), "empty": not bool(text), "admission_monotonic_s": admission, "admission_source_sensitivity": admission_sensitivity, "profiles": {}}
        if not text or not tokens_payload or admission is None:
            row["excluded_reason"] = "empty_or_missing_tokens_or_admission"
            parent_results.append(row)
            continue
        tokens = tuple(STTTimedToken(text=str(item["text"]), language="en", start_ms=item.get("start_ms"), end_ms=item.get("end_ms"), timing=item.get("timing"), source_start_sample=item.get("source_start_sample"), source_end_sample=item.get("source_end_sample")) for item in tokens_payload)
        r0_units = [{"group_id": "CURRENT-0", "relation": "CURRENT", "text": text, "token_indexes": list(range(len(tokens))), "start_source_sample": min((item.get("source_start_sample") for item in tokens_payload if item.get("source_start_sample") is not None), default=None), "end_source_sample": max((item.get("source_end_sample") for item in tokens_payload if item.get("source_end_sample") is not None), default=None)}]
        r0_scored = score_live_ledger(live_parent_ledger(parent_text=text, tokens=tokens_payload, units=r0_units, marks=parent.get("marks") or {}, meeting="ES2009d", seal_reasons=(parent.get("seal_reason"),)), words=words)
        recorded_r0 = (parent.get("r0") or {}).get("contamination") or {}
        fidelity_fields = ("eligible", "sequential_target", "reason", "attributable_chars", "contaminated_chars", "mixed_chars", "unaligned_chars", "unknown_chars", "eligible_units", "coverage")
        row["r0"] = {
            "recorded_contamination": recorded_r0,
            "recomputed_whole_parent_contamination": r0_scored.get("contamination"),
            "recorded_core_score_match": all((r0_scored.get("contamination") or {}).get(key) == recorded_r0.get(key) for key in fidelity_fields),
            "guard": r0_scored.get("guard"),
            "units": r0_units,
        }
        r0_cont = recorded_r0
        totals["r0"]["unknown_chars"] += int(r0_cont.get("unknown_chars") or 0)
        totals["r0"]["mixed_chars"] += int(r0_cont.get("mixed_chars") or 0)
        totals["r0"]["unaligned_chars"] += int(r0_cont.get("unaligned_chars") or 0)
        if r0_cont.get("eligible"):
            totals["r0"]["primary_eligible"] += 1
            totals["r0"]["attributable_chars"] += int(r0_cont.get("attributable_chars") or 0)
            totals["r0"]["contaminated_chars"] += int(r0_cont.get("contaminated_chars") or 0)
        totals["r0"]["units"] += 1
        for profile, decoder in profile_decoders.items():
            events = tuple(hypothesis_from_live_event(item, capture_epoch=1, producer_generation=generation, reference_generation=generation) for item in decoder.events)
            evidence = tuple(PretranslationEvidence(capture_epoch=1, start_sample=int(item.start_sample), end_sample=int(item.end_sample), available_at_monotonic_s=float(item.available_at_monotonic_s), relation=str(item.relation), producer_generation=generation, reference_generation=generation, reference_valid=True) for item in decoder.evidence)
            admission_estimate = float(admission_sensitivity["piecewise_linear_estimate_source_s"])
            admission_lower = float(admission_sensitivity["lower_source_s"])
            admission_upper = float(admission_sensitivity["upper_source_s"])
            units, late, reasons = assign_ownership_units(tokens, events, admitted_at_monotonic_s=admission_estimate, capture_epoch=1, evidence=evidence[-4096:])
            lower_units, _lower_late, lower_reasons = assign_ownership_units(tokens, events, admitted_at_monotonic_s=admission_lower, capture_epoch=1, evidence=evidence[-4096:])
            upper_units, _upper_late, upper_reasons = assign_ownership_units(tokens, events, admitted_at_monotonic_s=admission_upper, capture_epoch=1, evidence=evidence[-4096:])
            units_payload = unit_rows(units)
            scored = score_live_ledger(live_parent_ledger(parent_text=text, tokens=tokens_payload, units=units_payload, marks=parent.get("marks") or {}, meeting="ES2009d", seal_reasons=(parent.get("seal_reason"),)), words=words)
            guard = pair_parent_guard(r0_scored.get("guard"), scored.get("guard"))
            span = parent.get("span") or [None, None]
            timely_inside = []
            if len(span) == 2 and span[0] is not None and span[1] is not None:
                timely_inside = [item for item in events if int(span[0]) < int(item.estimated_transition_sample) < int(span[1]) and float(item.available_at_monotonic_s) <= admission_estimate]
            reconstruction = "".join(item["text"] for item in units_payload)
            coverage_blocked = "insufficient_evidence_coverage" in reasons
            mapped_starts = [int(item["source_start_sample"]) for item in tokens_payload if item.get("source_start_sample") is not None]
            mapped_ends = [int(item["source_end_sample"]) for item in tokens_payload if item.get("source_end_sample") is not None]
            full_coverage_s = None if not mapped_starts or not mapped_ends else earliest_full_coverage_s(evidence, min(mapped_starts), max(mapped_ends))
            blocker = None
            uncertain_tokens = [
                {"token_index": index, "text": str(item.get("text") or ""), "timing": item.get("timing")}
                for index, item in enumerate(tokens_payload)
                if item.get("timing") in {"unmapped", "end_only", "invalid"}
                or item.get("source_start_sample") is None
                or item.get("source_end_sample") is None
            ]
            if coverage_blocked and uncertain_tokens:
                blocker = {"kind": "unmapped_or_uncertain_asr_token", "tokens": uncertain_tokens, "detail": "At least one accepted ASR token lacks a complete valid source interval, so U13 cannot establish full-parent coverage."}
            elif coverage_blocked:
                blocker = {"kind": "insufficient_native_coverage_frontier", "earliest_full_coverage_source_s": full_coverage_s, "after_estimated_admission_s": None if full_coverage_s is None else full_coverage_s - admission_estimate}
            profile_row = {
                "ownership_evidence_status": "UNVERIFIED_INTERPOLATED_ADMISSION_SENSITIVITY",
                "ownership_evidence_basis": "Native probabilities and receipt QPC are actual for this local profile; admission source time is interpolated from the old one-second feed grid and is not an exact historical-admission comparison.",
                "units": units_payload,
                "reconstructed_exact": reconstruction == text,
                "unknown_reasons": list(reasons),
                "late_ignored": list(late),
                "confirmed_transitions_inside_parent_at_estimated_admission": len(timely_inside),
                "timely_event_but_full_parent_coverage_blocked": bool(timely_inside and coverage_blocked),
                "model_detection_failure_at_estimated_admission": "no_confirmed_transition" in reasons,
                "blocker_descriptor": blocker,
                "feed_grid_bracket_sensitivity": {
                    "lower_source_s": admission_lower,
                    "lower_units": len(lower_units),
                    "lower_partitions": max(len(lower_units) - 1, 0),
                    "lower_reasons": list(lower_reasons),
                    "upper_source_s": admission_upper,
                    "upper_units": len(upper_units),
                    "upper_partitions": max(len(upper_units) - 1, 0),
                    "upper_reasons": list(upper_reasons),
                },
                "contamination": scored.get("contamination"),
                "guard": guard,
            }
            row["profiles"][profile] = profile_row
            totals[profile]["parents_replayed"] += 1
            totals[profile]["units"] += len(units_payload)
            totals[profile]["extra_units"] += max(len(units_payload) - 1, 0)
            totals[profile]["unknown_units"] += sum(item["relation"] == "UNKNOWN" for item in units_payload)
            totals[profile]["reconstruction_failures"] += reconstruction != text
            totals[profile]["timely_transition_parents"] += bool(timely_inside)
            totals[profile]["timely_event_coverage_blocked_parents"] += bool(timely_inside and coverage_blocked)
            totals[profile]["coverage_abstentions"] += coverage_blocked
            totals[profile]["detection_failure_abstentions"] += "no_confirmed_transition" in reasons
            contamination = scored.get("contamination") or {}
            totals[profile]["unknown_chars"] += int(contamination.get("unknown_chars") or 0)
            totals[profile]["mixed_chars"] += int(contamination.get("mixed_chars") or 0)
            totals[profile]["unaligned_chars"] += int(contamination.get("unaligned_chars") or 0)
            if contamination.get("eligible"):
                totals[profile]["primary_eligible"] += 1
                totals[profile]["attributable_chars"] += int(contamination.get("attributable_chars") or 0)
                totals[profile]["contaminated_chars"] += int(contamination.get("contaminated_chars") or 0)
            if guard.get("severe"):
                severe[profile].append({"parent_id": parent.get("parent_id"), "failures": guard.get("failures"), "guard": guard})
        parent_results.append(row)
    for profile in PROFILES:
        totals[profile]["new_severe_guards"] = len(severe[profile])
    default = totals["recorded-default"]
    low = totals["official-low-latency"]
    sensitivity_summary = {
        "status": "UNVERIFIED_INTERPOLATED_ADMISSION_SENSITIVITY",
        "estimated_timely_parents": {
            "recorded_default": default["timely_transition_parents"],
            "official_low_latency": low["timely_transition_parents"],
        },
        "partitions": {
            "recorded_default": default["extra_units"],
            "official_low_latency": low["extra_units"],
        },
        "contamination": {
            "recorded_default": {"contaminated_chars": default["contaminated_chars"], "attributable_chars": default["attributable_chars"]},
            "official_low_latency": {"contaminated_chars": low["contaminated_chars"], "attributable_chars": low["attributable_chars"]},
        },
        "interpretation": "The estimate changes timely-parent availability from 0 to 2, but partitions remain 0 and contamination remains 22/1034 for both profiles. This does not mean zero useful evidence; it means neither estimated-timely case passes every unchanged partition requirement at the interpolated cutoff.",
    }
    blockers_by_parent = {
        item["parent_id"]: item["profiles"]["official-low-latency"]
        for item in parent_results
        if item["parent_id"] in {
            "8465ae6c-5f27-457f-bcc1-42944ff0294e",
            "684dd105-8c15-498b-9848-4911f3a5f825",
        }
    }
    reviewed_blockers = [
        {
            "parent_id": "8465ae6c-5f27-457f-bcc1-42944ff0294e",
            "profile": "official-low-latency",
            "cause": blockers_by_parent["8465ae6c-5f27-457f-bcc1-42944ff0294e"]["blocker_descriptor"],
            "feed_grid_bracket_sensitivity": blockers_by_parent["8465ae6c-5f27-457f-bcc1-42944ff0294e"]["feed_grid_bracket_sensitivity"],
        },
        {
            "parent_id": "684dd105-8c15-498b-9848-4911f3a5f825",
            "profile": "official-low-latency",
            "cause": blockers_by_parent["684dd105-8c15-498b-9848-4911f3a5f825"]["blocker_descriptor"],
            "feed_grid_bracket_sensitivity": blockers_by_parent["684dd105-8c15-498b-9848-4911f3a5f825"]["feed_grid_bracket_sensitivity"],
            "interpretation": "Complete native coverage arrives after the interpolated admission estimate, while the upper feed-grid bracket changes this parent from coverage abstention to a partition.",
        },
    ]
    result = {
        "schema": "SORTFORMER-LOW-LATENCY-PROBE-RESULT-1",
        "scope": "bounded engineering probe; no population efficacy claim",
        "instrumentation": {
            "native_capture_adapter": "Dedicated local TCP capture adapter implemented in probe.py, not the pinned NativeSortformerProducer wrapper.",
            "reason": "The existing wrapper cannot isolate per-profile stream overrides while retaining native pace QPC, raw support, visibility, service timing, effective trace configuration, and dump metadata required by this probe.",
            "reused_interfaces": ["LiveTransitionDecoder", "hypothesis_from_live_event", "assign_ownership_units", "live_parent_ledger", "score_live_ledger", "pair_parent_guard"],
            "not_tested": ["NativeSortformerProducer wrapper", "full live ASR/translation pipeline", "product runtime"],
        },
        "policy": {"revision": "U13-COVERAGE-2", "guard": "U10-GUARD-3", "threshold": 0.5, "confirmation_samples": 1600, "added_wait_samples": 0, "evidence_capacity": 4096},
        "clock_contract": {"verified": False, "precise_missing_anchor": selection["recorded_clock"]["missing_anchor"], "recorded_metadata": selection["recorded_clock"], "new_mapping_verified": "Each new native ready qpc is actual pace source zero; each new chunk qpc is mapped to source seconds by that run's qpf.", "ownership_comparison_status": "UNVERIFIED sensitivity only: frozen admissions are converted with piecewise-linear interpolation of the original one-second feed-progress grid. This is not an exact common source-time mapping and is not used as an acceptance claim.", "no_first_arrival_shift": True},
        "selection": {key: selection[key] for key in ("selected", "evaluation_end_sample", "context_tail_seconds", "projection_real_source_samples", "accounting")},
        "profile_runs": {profile: {"evidence_status": "ACTUAL_LOCAL_NATIVE_CAPTURE", "effective_config": run["effective_config"], "source": run["source"], "cadence": run["cadence"], "process": run["process"]} for profile, run in profile_runs.items()},
        "splits": {key: dict(value) for key, value in totals.items()},
        "ownership_sensitivity": sensitivity_summary,
        "reviewed_blockers": reviewed_blockers,
        "severe_guards": severe,
        "parents": parent_results,
        "decision": {
            "usage_fix_operational": profile_runs["official-low-latency"]["effective_config"] == PROFILES["official-low-latency"] and int(profile_runs["official-low-latency"]["source"]["output_frames"]) > 0,
            "large_paid_rerun_of_unchanged_r2": "NOT_ESTABLISHED_BY_THIS_PROBE",
            "new_instrumented_experiment": "NOT_ADJUDICATED_BY_THIS_PROBE",
            "ownership_effect": "INDETERMINATE",
            "interpretation": "The probe establishes the low-latency native usage/cadence change. It cannot establish benefit or absence of benefit for unchanged R2: the exact old admission clock is unavailable, the interpolated estimate yields no partitions, and the upper feed-grid bracket flips one coverage decision.",
            "limitations": ["Single preregistered ES2009d prefix only.", "Frozen ASR text/admissions and offline ownership replay, not a live ASR or translation rerun.", "Frozen-admission ownership results are explicitly unverified because the original exact source-zero anchor is unavailable.", "The null ownership sensitivity is clock-fragile: the upper feed-grid bracket permits a partition for 684dd105.", "Dedicated probe TCP capture tested the native executable and reused decoder/partition/scorer interfaces, but did not test NativeSortformerProducer or the full live pipeline.", "Paced wall duration is not reported as compute RTF."],
        },
    }
    write_json(HERE / "result.json", result)
    conclusion = {
        "actual_native_result": {
            "status": "VERIFIED_LOCAL_CAPTURE",
            "usage_fix_operational": result["decision"]["usage_fix_operational"],
            "recorded_default": {"native_frames": default["native_frames"], "native_transition_events": default["native_transition_events"], "native_evidence_intervals": default["native_evidence_intervals"]},
            "official_low_latency": {"native_frames": low["native_frames"], "native_transition_events": low["native_transition_events"], "native_evidence_intervals": low["native_evidence_intervals"]},
        },
        "ownership_sensitivity": sensitivity_summary,
        "reviewed_blockers": reviewed_blockers,
        "decision": result["decision"],
    }
    write_json(HERE / "conclusion.json", conclusion)
    print(json.dumps(conclusion, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    run = sub.add_parser("run-profile")
    run.add_argument("profile", choices=tuple(PROFILES))
    sub.add_parser("compare")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    elif args.command == "run-profile":
        run_profile(args.profile)
    else:
        compare()


if __name__ == "__main__":
    main()
