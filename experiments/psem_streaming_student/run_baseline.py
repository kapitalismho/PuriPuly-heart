from __future__ import annotations

import argparse
import csv
import ctypes
import gzip
import hashlib
import importlib.abc
import importlib.util
import json
import os
import shutil
import socket
import subprocess
import sys
import tarfile
import time
import wave
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any
from uuid import UUID

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CONFIG_PATH = HERE / "baseline_config.json"
RUNS = HERE / "runs"
RETAINED = ROOT / "experiments/psem_r2_policy/artifacts/retained/historical_policy_inputs.jsonl.gz"
ANNOTATIONS = Path(r"C:/Users/salee/AppData/Local/Temp/opencode/stb_phase2_corpora/ami/annotations/words")
PROFILE_KEYS = {
    "chunk_len": "TRANSCRIBE_SORTFORMER_STREAM_CHUNK_LEN",
    "chunk_left_context": "TRANSCRIBE_SORTFORMER_STREAM_LC",
    "chunk_right_context": "TRANSCRIBE_SORTFORMER_STREAM_RC",
    "fifo_len": "TRANSCRIBE_SORTFORMER_STREAM_FIFO_LEN",
    "spkcache_len": "TRANSCRIBE_SORTFORMER_STREAM_SPKCACHE_LEN",
    "spkcache_update_period": "TRANSCRIBE_SORTFORMER_STREAM_UPDATE_PERIOD",
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def emit(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


class ArchiveFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def __init__(self, sources: dict[str, tuple[bytes, bool, str]]) -> None:
        self.sources = sources

    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> Any:
        item = self.sources.get(fullname)
        return None if item is None else importlib.util.spec_from_loader(fullname, self, is_package=item[1])

    def create_module(self, spec: Any) -> Any:
        return None

    def exec_module(self, module: Any) -> None:
        source, is_package, origin = self.sources[module.__name__]
        module.__file__ = origin
        if is_package:
            module.__path__ = [origin.rsplit("/", 1)[0]]
        exec(compile(source, origin, "exec"), module.__dict__)


def load_runtime(config: dict[str, Any]) -> tuple[Any, Any]:
    archive = ROOT / config["receiver"]["runtime_archive"]
    override = ROOT / config["receiver"]["ownership_override"]
    if digest(archive) != config["receiver"]["runtime_archive_sha256"] or digest(override) != config["receiver"]["ownership_override_sha256"]:
        raise RuntimeError("receiver runtime identity mismatch")
    sources: dict[str, tuple[bytes, bool, str]] = {}
    with tarfile.open(archive, "r:gz") as bundle:
        for member in bundle.getmembers():
            if not member.isfile() or not member.name.startswith("src/puripuly_heart/") or not member.name.endswith(".py"):
                continue
            handle = bundle.extractfile(member)
            if handle is None:
                raise RuntimeError(f"unreadable runtime member {member.name}")
            relative = member.name[4:]
            package = relative.endswith("/__init__.py")
            name = relative[:-12].replace("/", ".") if package else relative[:-3].replace("/", ".")
            sources[name] = (handle.read(), package, f"{archive}!{member.name}")
    sources["puripuly_heart.core.audio.pretranslation_ownership"] = (override.read_bytes(), False, str(override))
    sys.meta_path.insert(0, ArchiveFinder(sources))
    policy = importlib.import_module("puripuly_heart.core.audio.pretranslation_ownership")
    decoder_path = ROOT / config["receiver"]["decoder"]
    spec = importlib.util.spec_from_file_location("psem164_sortformer", decoder_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("decoder import failed")
    decoder = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = decoder
    spec.loader.exec_module(decoder)
    return policy, decoder


def qpc_api() -> tuple[Any, Any, int]:
    counter = ctypes.c_longlong()
    frequency = ctypes.c_longlong()
    query = ctypes.windll.kernel32.QueryPerformanceCounter
    query(ctypes.byref(counter))
    ctypes.windll.kernel32.QueryPerformanceFrequency(ctypes.byref(frequency))
    return query, counter, int(frequency.value)


def qpc_now(query: Any, counter: Any) -> int:
    query(ctypes.byref(counter))
    return int(counter.value)


def open_process_handle(pid: int) -> int:
    open_process = ctypes.windll.kernel32.OpenProcess
    open_process.argtypes = (ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong)
    open_process.restype = ctypes.c_void_p
    handle = open_process(0x0410, 0, pid)
    if not handle:
        raise ctypes.WinError()
    return int(handle)


def process_memory(handle: int) -> tuple[int, int]:
    class PMC(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong), ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t), ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t), ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t), ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
    value = PMC()
    value.cb = ctypes.sizeof(value)
    query = ctypes.windll.psapi.GetProcessMemoryInfo
    query.argtypes = (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong)
    query.restype = ctypes.c_int
    if not query(ctypes.c_void_p(handle), ctypes.byref(value), value.cb):
        return 0, 0
    return int(value.WorkingSetSize), int(value.PeakWorkingSetSize)


def cpu_seconds(handle: int) -> float:
    created = ctypes.c_ulonglong()
    exited = ctypes.c_ulonglong()
    kernel = ctypes.c_ulonglong()
    user = ctypes.c_ulonglong()
    query = ctypes.windll.kernel32.GetProcessTimes
    query.argtypes = (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)
    query.restype = ctypes.c_int
    if not query(ctypes.c_void_p(handle), ctypes.byref(created), ctypes.byref(exited), ctypes.byref(kernel), ctypes.byref(user)):
        return 0.0
    return (kernel.value + user.value) / 10_000_000


def make_projection(source: Path, destination: Path, samples: int) -> None:
    with wave.open(str(source), "rb") as reader:
        if reader.getframerate() != 16000 or reader.getnchannels() != 1 or reader.getsampwidth() != 2 or reader.getnframes() < samples:
            raise RuntimeError("unsupported or short source waveform")
        params = reader.getparams()
        data = reader.readframes(samples)
    with wave.open(str(destination), "wb") as writer:
        writer.setparams(params)
        writer.setnframes(samples)
        writer.writeframes(data)


def retained_parents(meeting: str, limit: int) -> list[dict[str, Any]]:
    result = []
    with gzip.open(RETAINED, "rt", encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            if row.get("cohort") != "current" or row.get("meeting") != meeting or row.get("type") != "parent":
                continue
            parent = row["value"]
            span = parent.get("span") or []
            if len(span) == 2 and span[0] is not None and span[1] is not None and int(span[1]) <= limit:
                result.append(parent)
    return sorted(result, key=lambda item: (int(item["span"][1]), int(item.get("index") or 0)))


def annotation_coverage(meeting: str, limit_samples: int, parents: list[dict[str, Any]]) -> dict[str, Any]:
    limit = limit_samples / 16000
    words = []
    identities = []
    for path in sorted(ANNOTATIONS.glob(f"{meeting}.*.words.xml")):
        role = path.name.split(".")[1]
        identities.append({"role": role, "sha256": digest(path)})
        root = ET.parse(path).getroot()
        for node in root:
            if not node.tag.endswith("w") or node.get("punc") == "true":
                continue
            start = node.get("starttime")
            end = node.get("endtime")
            if start is None or end is None or float(start) >= limit:
                continue
            words.append((float(start), min(float(end), limit), role, "".join(node.itertext())))
    words.sort()
    transitions = []
    for left, right in zip(words, words[1:]):
        if left[2] != right[2] and left[1] <= right[0]:
            transitions.append({"sample": round(right[0] * 16000), "from": left[2], "to": right[2], "left": left[3], "right": right[3]})
    overlaps = []
    for index, left in enumerate(words):
        for right in words[index + 1:]:
            if right[0] >= left[1]:
                break
            if left[2] != right[2] and min(left[1], right[1]) > max(left[0], right[0]):
                overlaps.append((max(left[0], right[0]), min(left[1], right[1]), left[2], right[2]))
    guard_counts = Counter()
    for parent in parents:
        guard = (parent.get("r0") or {}).get("guard") or {}
        if guard.get("same_speaker_stratum"):
            guard_counts["retained_same_speaker_parents"] += 1
        if int(guard.get("span_verified_changes") or 0) > 0:
            guard_counts["retained_changed_speaker_parents"] += 1
        if int(guard.get("excluded", {}).get("mixed") or 0) > 0:
            guard_counts["retained_overlap_or_mixed_parents"] += 1
    return {"annotation_identities": identities, "lexical_words": len(words), "distinct_nonoverlap_role_changes": len(transitions), "transition_witnesses": transitions[:12], "overlap_intersections": len(overlaps), "overlap_seconds_sum_not_union": sum(right - left for left, right, _a, _b in overlaps), "retained_parent_guards": dict(guard_counts), "interpretation": "Conditions are reported from accessible AMI word intervals and retained parent guard labels; overlap count is pairwise intersections, not a DER metric or disjoint duration."}


def token_objects(tokens: list[dict[str, Any]], token_class: Any) -> tuple[Any, ...]:
    return tuple(token_class(text=row["text"], language="en", start_ms=row.get("start_ms"), end_ms=row.get("end_ms"), timing=row.get("timing"), source_start_sample=row.get("source_start_sample"), source_end_sample=row.get("source_end_sample"), provenance=row.get("provenance")) for row in tokens)


def effective_labels(tokens: tuple[Any, ...], events: tuple[Any, ...], evidence: tuple[Any, ...], admission: float, policy: Any) -> list[tuple[str, str, str | None]]:
    applicable = policy._applicable_hypotheses(events, admitted_at_monotonic_s=admission, capture_epoch=1)
    if len(tokens) > 1 and not policy._partition_is_requested(tokens, applicable):
        return []
    if tokens and policy._partition_is_requested(tokens, applicable):
        selected = policy._partition_coverage_generation(tokens, applicable, evidence, admitted_at_monotonic_s=admission, capture_epoch=1)
        if selected is None:
            return []
        applicable = [item for item in applicable if policy._same_generation(item.producer_generation, item.reference_generation, selected[0], selected[1])]
        evidence = tuple(item for item in evidence if policy._same_generation(item.producer_generation, item.reference_generation, selected[0], selected[1]))
    result = []
    for token in tokens:
        start, end, uncertain = policy._token_source_interval(token)
        if uncertain is not None:
            result.append(("UNKNOWN", f"u:{uncertain}", uncertain))
        elif any(start < event.estimated_transition_sample < end for event in applicable):
            result.append(("UNKNOWN", "u:straddle", "straddle"))
        else:
            relation, reason = policy._relation_from_evidence(start, end, evidence, admitted_at_monotonic_s=admission, capture_epoch=1)
            result.append((relation, f"s:{policy._transition_segment(end, applicable)}", reason))
    return result


def selected_units(tokens: tuple[Any, ...], labels: list[tuple[str, str, str | None]], policy: Any) -> tuple[tuple[Any, ...], list[int]]:
    if not labels:
        return (policy._whole_parent_unit(tokens) if tokens else ()), []
    groups = [[0]]
    removed = []
    for index in range(1, len(labels)):
        left, right = labels[index - 1], labels[index]
        same = left[:2] == right[:2]
        suppress = left[1] == right[1] and left[0] != right[0] and "UNKNOWN" in (left[0], right[0])
        if same or suppress:
            groups[-1].append(index)
            if suppress:
                removed.append(index)
        else:
            groups.append([index])
    units = tuple(policy._unit_from_run(tokens, indexes, "UNKNOWN" if any(labels[index][0] == "UNKNOWN" for index in indexes) else labels[indexes[0]][0], unit_index) for unit_index, indexes in enumerate(groups))
    return units, removed


def unit_payload(units: tuple[Any, ...]) -> list[dict[str, Any]]:
    return [{"group_id": unit.group_id, "relation": unit.relation, "text": unit.text, "token_indexes": list(unit.token_indexes), "start_source_sample": unit.start_source_sample, "end_source_sample": unit.end_source_sample} for unit in units]


def parse_gpu(path: Path, pid: int) -> dict[str, Any]:
    if not path.exists() or path.stat().st_size == 0:
        return {"status": "unavailable", "reason": "typeperf produced no counter file"}
    for encoding in ("utf-16", "utf-8-sig", "cp949"):
        try:
            rows = list(csv.reader(path.read_text(encoding=encoding).splitlines()))
            if rows:
                break
        except UnicodeError:
            rows = []
    if len(rows) < 2:
        return {"status": "unavailable", "reason": "typeperf counter file was unreadable or had no samples"}
    indexes = [index for index, name in enumerate(rows[0]) if f"pid_{pid}_" in name.lower() and "dedicated usage" in name.lower()]
    samples = []
    for row in rows[1:]:
        values = []
        for index in indexes:
            try:
                values.append(float(row[index]))
            except (ValueError, IndexError):
                pass
        if values:
            samples.append(sum(values))
    if not samples:
        return {"status": "unavailable", "reason": "Windows GPU Process Memory exposed no attributable counter instance for the native PID", "global_gpu_usage_used": False}
    return {"status": "observed_process_counter", "counter": "GPU Process Memory(pid_NATIVE_*) Dedicated Usage", "samples": len(samples), "peak_bytes": int(max(samples)), "global_gpu_usage_used": False, "limitation": "Windows per-process dedicated-usage PDH counter; shared memory and driver allocations outside this PID are not included."}


def run_source(source: dict[str, Any], config: dict[str, Any], policy: Any, native: Any, absolute_deadline: float) -> dict[str, Any]:
    meeting = source["meeting"]
    run_dir = RUNS / meeting
    if run_dir.exists():
        raise RuntimeError(f"one-pass evidence already exists for {meeting}")
    run_dir.mkdir(parents=True)
    projection = run_dir / "input.wav"
    make_projection(Path(source["wav"]), projection, int(source["execution_samples"]))
    parents = retained_parents(meeting, int(source["evaluation_samples"]))
    annotations = annotation_coverage(meeting, int(source["evaluation_samples"]), parents)
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    server.settimeout(60)
    port = server.getsockname()[1]
    dump_dir = run_dir / "dump"
    dump_dir.mkdir()
    env = os.environ.copy()
    for key in (*PROFILE_KEYS.values(), "TRANSCRIBE_PSEM_PACE_16KHZ", "TRANSCRIBE_PSEM_EVENTS_TCP", "TRANSCRIBE_PSEM_PACE_FROM_SAMPLE", "TRANSCRIBE_PSEM_PACE_UNTIL_SAMPLE", "TRANSCRIBE_DUMP_DIR"):
        env.pop(key, None)
    env.update({"TRANSCRIBE_PSEM_PACE_16KHZ": "1", "TRANSCRIBE_PSEM_EVENTS_TCP": f"127.0.0.1:{port}", "TRANSCRIBE_PSEM_CAUSAL_FRONTEND": "1", "TRANSCRIBE_PSEM_PACE_FROM_SAMPLE": "0", "TRANSCRIBE_PSEM_PACE_UNTIL_SAMPLE": str(source["execution_samples"]), "TRANSCRIBE_DUMP_DIR": str(dump_dir)})
    for name, key in PROFILE_KEYS.items():
        env[key] = str(config["native"]["profile"][name])
    command = [config["native"]["executable"], "-m", config["native"]["model"], "--backend", "vulkan", str(projection)]
    stdout_path, stderr_path = run_dir / "native.stdout.log", run_dir / "native.stderr.log"
    query, counter, local_qpf = qpc_api()
    wrapper_handle = open_process_handle(os.getpid())
    wrapper_cpu_start = cpu_seconds(wrapper_handle)
    started_qpc = qpc_now(query, counter)
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        process = subprocess.Popen(command, env=env, stdout=stdout, stderr=stderr)
        gpu_path = run_dir / "gpu-process-memory.csv"
        gpu = subprocess.Popen(["typeperf", f"\\GPU Process Memory(pid_{process.pid}_*)\\Dedicated Usage", "-si", "1", "-sc", "1800", "-f", "CSV", "-o", str(gpu_path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        conn = None
        try:
            conn, peer = server.accept()
            conn.settimeout(0.005)
            receiver = policy.PretranslationOwnershipOwner(enabled=True, tombstone_capacity=int(config["receiver"]["evidence_capacity"]))
            decoder = native.LiveTransitionDecoder()
            generation = f"{meeting}:continuous-source-zero"
            messages = []
            transitions = []
            assignments = []
            all_events = []
            all_evidence = []
            raw_buffer = b""
            ready = None
            done = None
            next_parent = 0
            rss_native = []
            rss_wrapper = []
            aggregate_rss = []
            child_handle = int(process._handle)

            def admit_due(now_qpc: int, force: bool = False) -> None:
                nonlocal next_parent
                if ready is None:
                    return
                while next_parent < len(parents):
                    parent = parents[next_parent]
                    due = int(ready["qpc"]) + round(int(parent["span"][1]) / 16000 * int(ready["qpf"]))
                    if not force and now_qpc < due:
                        break
                    actual = qpc_now(query, counter)
                    cutoff = (actual - int(ready["qpc"])) / int(ready["qpf"])
                    tokens = token_objects(list(parent.get("tokens") or ()), policy.STTTimedToken)
                    if tokens and parent.get("text"):
                        assignment = receiver.assign(parent_utterance_id=UUID(parent["parent_id"]), timed_tokens=tokens, capture_epoch=1, admitted_at_monotonic_s=cutoff, parent_text=parent["text"])
                        labels = effective_labels(tokens, tuple(all_events), tuple(all_evidence), cutoff, policy)
                        selected, removed = selected_units(tokens, labels, policy)
                        baseline_units = unit_payload(assignment.units)
                        selected_rows = unit_payload(selected)
                    else:
                        assignment = None
                        labels, removed, baseline_units, selected_rows = [], [], [], []
                    assignments.append({"parent_id": parent["parent_id"], "source_span": parent["span"], "seal_reason": parent.get("seal_reason"), "terminal_outcome": parent.get("terminal_outcome"), "failure_reason": parent.get("failure_reason"), "accepted_text": parent.get("text") or "", "accepted_text_sha256": hashlib.sha256((parent.get("text") or "").encode()).hexdigest(), "tokens": parent.get("tokens") or [], "clock": {"scheduled_cutoff_qpc": due, "actual_admission_qpc": actual, "actual_admission_source_zero_s": cutoff, "scheduling_lateness_s": (actual - due) / int(ready["qpf"]), "added_wait_s": 0.0}, "evidence_consumed": {"events_received": len(all_events), "intervals_received": len(all_evidence)}, "receiver": {"disposition": None if assignment is None else assignment.disposition, "conserved": True if assignment is None else assignment.conserved, "late_ignored": [] if assignment is None else list(assignment.late_ignored), "unknown_reasons": [] if assignment is None else list(assignment.unknown_reasons), "pre_selected_policy_units": baseline_units}, "selected_unknown_only": {"predicate": "same transition/uncertainty key AND differing relation AND one UNKNOWN; merged relation remains UNKNOWN", "labels": labels, "removed_boundary_token_indexes": removed, "units": selected_rows, "text_conserved": "".join(row["text"] for row in selected_rows) == (parent.get("text") or "")}, "no_active_psem_replay": {"units": [] if not parent.get("text") else [{"group_id": "R0-0", "relation": "CURRENT", "text": parent["text"], "token_indexes": list(range(len(parent.get("tokens") or ())))}], "same_accepted_text": True}, "evidence_label": "actual receiver on frozen accepted-text replay; no ASR/API/display execution"})
                    next_parent += 1

            while True:
                if time.perf_counter() >= absolute_deadline:
                    raise TimeoutError("combined model wall cap reached")
                now = qpc_now(query, counter)
                admit_due(now)
                try:
                    chunk = conn.recv(1 << 20)
                except socket.timeout:
                    chunk = None
                if chunk:
                    raw_buffer += chunk
                    while b"\n" in raw_buffer:
                        raw, raw_buffer = raw_buffer.split(b"\n", 1)
                        if not raw.strip():
                            continue
                        message = json.loads(raw)
                        receipt_qpc = qpc_now(query, counter)
                        message["receiver_receipt_qpc"] = receipt_qpc
                        messages.append(message)
                        if message.get("type") == "ready" and ready is None:
                            ready = message
                            if int(message["qpf"]) != local_qpf:
                                raise RuntimeError("QPC frequency mismatch")
                        elif message.get("type") == "chunk":
                            admit_due(receipt_qpc)
                            message["capture_epoch"] = 1
                            message["producer_generation"] = generation
                            message["reference_generation"] = generation
                            message["reference_valid"] = True
                            message["valid_mask"] = [True] * int(message["emit_count"])
                            available = (receipt_qpc - int(ready["qpc"])) / int(ready["qpf"])
                            before = len(decoder.evidence)
                            new_events = decoder.ingest_chunk(int(message["emit_start_frame"]), message["probs"], available_at_monotonic_s=available, receipt_kind="actual_receiver_receipt")
                            for event in new_events:
                                hypothesis = native.hypothesis_from_live_event(event, capture_epoch=1, producer_generation=generation, reference_generation=generation)
                                receiver.observe(hypothesis)
                                all_events.append(hypothesis)
                                transitions.append({"event_id": event.event_id, "boundary_sample": event.boundary, "source_support_interval": [max(event.boundary - 1, 0), event.frontier], "consumed_audio_frontier_sample": event.frontier, "candidate_native_slot": event.candidate_slot, "native_availability_qpc": message.get("qpc"), "receiver_receipt_qpc": receipt_qpc, "receiver_receipt_source_zero_s": available, "confirmation_samples": config["native"]["confirmation_samples"], "capture_epoch": 1, "producer_generation": generation, "reference_generation": generation, "reference_valid": True})
                            for evidence in decoder.evidence[before:]:
                                receiver.observe_evidence(capture_epoch=1, start_sample=evidence.start_sample, end_sample=evidence.end_sample, available_at_monotonic_s=available, relation=evidence.relation, producer_generation=generation, reference_generation=generation, reference_valid=True)
                                all_evidence.append(policy.PretranslationEvidence(capture_epoch=1, start_sample=evidence.start_sample, end_sample=evidence.end_sample, available_at_monotonic_s=available, relation=evidence.relation, producer_generation=generation, reference_generation=generation, reference_valid=True))
                        elif message.get("type") == "done":
                            done = message
                elif chunk == b"" and process.poll() is not None:
                    break
                native_working, _native_peak = process_memory(child_handle)
                wrapper_working, _wrapper_peak = process_memory(wrapper_handle)
                rss_native.append(native_working)
                rss_wrapper.append(wrapper_working)
                aggregate_rss.append(native_working + wrapper_working)
            exit_code = process.wait(timeout=30)
            admit_due(qpc_now(query, counter), force=True)
            finished_qpc = qpc_now(query, counter)
            native_cpu = cpu_seconds(child_handle)
            wrapper_cpu = cpu_seconds(wrapper_handle) - wrapper_cpu_start
        finally:
            if conn is not None:
                conn.close()
            server.close()
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=10)
            gpu.terminate()
            try:
                gpu.wait(timeout=10)
            except subprocess.TimeoutExpired:
                gpu.kill()
    ctypes.windll.kernel32.CloseHandle(ctypes.c_void_p(wrapper_handle))
    if exit_code != 0 or ready is None or done is None:
        raise RuntimeError(f"native run incomplete exit={exit_code} ready={ready is not None} done={done is not None}")
    raw_path = run_dir / "native-events.jsonl.gz"
    with gzip.open(raw_path, "wt", encoding="utf-8") as target:
        for message in messages:
            target.write(json.dumps(message, separators=(",", ":")) + "\n")
    assignments_path = run_dir / "receiver-assignments.jsonl.gz"
    with gzip.open(assignments_path, "wt", encoding="utf-8") as target:
        for assignment in assignments:
            target.write(json.dumps(assignment, ensure_ascii=False, separators=(",", ":")) + "\n")
    trace_path = dump_dir / "diar.trace.json"
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    effective = {name: trace[name] for name in PROFILE_KEYS}
    chunks = [row for row in messages if row.get("type") == "chunk"]
    receipts = [(int(row["receiver_receipt_qpc"]) - int(ready["qpc"])) / int(ready["qpf"]) for row in chunks]
    support_lag = [receipt - int(row["raw_support_end_sample"]) / 16000 for receipt, row in zip(receipts, chunks)]
    removed = sum(len(row["selected_unknown_only"]["removed_boundary_token_indexes"]) for row in assignments)
    changed = sum(row["selected_unknown_only"]["units"] != row["receiver"]["pre_selected_policy_units"] for row in assignments)
    result = {"schema": "PSEM-STREAMING-STUDENT-TEACHER-SOURCE-RUN-1", "meeting": meeting, "evidence_labels": {"native": "actual local Vulkan native pass", "receiver": "actual prospective receiver invocation on frozen accepted-text replay", "translation_admission": "actual experiment cutoff invocation in the shared QPC clock", "api": "not run", "display": "not run"}, "source": {**source, "wav_sha256": digest(Path(source["wav"])), "projection_sha256": digest(projection), "projection_bytes": projection.stat().st_size, "parents_admitted": len(assignments), "annotation_coverage": annotations}, "identity": {"command": command, "effective_profile": effective, "generation": generation, "capture_epoch": 1, "reset_contract": "one native process and one receiver generation from source zero through the complete approved source; logical parent admissions do not reset model/reference", "genuine_discontinuity": "source end only; no reconnect or synthetic seal reset occurred"}, "clock": {"kind": "Windows QueryPerformanceCounter", "qpf": int(ready["qpf"]), "native_source_zero_qpc": int(ready["qpc"]), "native_done_qpc": int(done["qpc"]), "process_start_qpc": started_qpc, "process_finish_qpc": finished_qpc, "all_native_availability_receiver_receipt_and_admission_values_share_qpc": True, "receiver_transport_delay_observed_not_rewritten": True}, "native": {"exit_code": exit_code, "messages": len(messages), "chunks": len(chunks), "output_frames": sum(int(row["emit_count"]) for row in chunks), "soft_output_values": sum(len(row["probs"]) * len(row["probs"][0]) for row in chunks if row.get("probs")), "maximum_visible_sample": max(int(row["n_visible"]) for row in chunks), "maximum_consumed_support_sample": max(int(row["raw_support_end_sample"]) for row in chunks), "transition_events": len(transitions), "transitions": transitions, "receiver_support_lag_s": {"min": min(support_lag), "median": sorted(support_lag)[len(support_lag) // 2], "max": max(support_lag)}}, "receiver": {"assignments": len(assignments), "source_only_or_empty": sum(not row["accepted_text"] for row in assignments), "text_conservation_failures": sum(not row["selected_unknown_only"]["text_conserved"] for row in assignments), "selected_suppressed_boundaries": removed, "selected_changed_parents": changed, "late_event_references": sum(len(row["receiver"]["late_ignored"]) for row in assignments), "admission_scheduling_lateness_s": {"max": max(row["clock"]["scheduling_lateness_s"] for row in assignments), "median": sorted(row["clock"]["scheduling_lateness_s"] for row in assignments)[len(assignments) // 2]}}, "cost": {"complete_path_wall_s": (finished_qpc - started_qpc) / local_qpf, "native_process_cpu_s": native_cpu, "receiver_wrapper_cpu_s": wrapper_cpu, "native_peak_working_set_bytes_observed": max(rss_native), "receiver_wrapper_peak_working_set_bytes_observed": max(rss_wrapper), "complete_path_peak_sum_working_set_bytes_sampled": max(aggregate_rss), "rss_sampling": "working sets sampled together in the receiver loop; peak sum is the maximum simultaneous sample", "gpu_memory": parse_gpu(gpu_path, process.pid)}, "artifacts": {"raw_frame_soft_outputs": {"path": str(raw_path.relative_to(ROOT)), "sha256": digest(raw_path)}, "receiver_assignments": {"path": str(assignments_path.relative_to(ROOT)), "sha256": digest(assignments_path)}, "native_trace": {"path": str(trace_path.relative_to(ROOT)), "sha256": digest(trace_path)}}}
    emit(run_dir / "RESULT.json", result)
    return result


def prepare(config: dict[str, Any]) -> dict[str, Any]:
    executable = Path(config["native"]["executable"])
    model = Path(config["native"]["model"])
    if digest(executable) != config["native"]["executable_sha256"] or digest(model) != config["native"]["model_sha256"]:
        raise RuntimeError("native executable or model identity mismatch")
    if int(config["authority"]["combined_model_wall_cap_seconds"]) != 1800 or config["authority"]["training_updates"] != 0 or config["authority"]["paid_api_calls"] != 0 or config["authority"]["holdout_eval_access"]:
        raise RuntimeError("execution envelope mismatch")
    checks = []
    for source in config["sources"]:
        path = Path(source["wav"])
        checks.append({"meeting": source["meeting"], "wav_exists": path.is_file(), "approved_geometry": int(source["execution_samples"]) == int(source["evaluation_samples"]) + int(source["real_context_tail_samples"])})
    result = {"status": "ready", "config_sha256": digest(CONFIG_PATH), "checks": checks, "model_execution_cap_s": 1800, "training": "not authorized and not run", "network_api": "not authorized and not run"}
    if not all(row["wav_exists"] and row["approved_geometry"] for row in checks):
        raise RuntimeError("source readiness failed")
    emit(HERE / "READINESS.json", result)
    return result


def execute(config: dict[str, Any]) -> dict[str, Any]:
    prepare(config)
    policy, native = load_runtime(config)
    if RUNS.exists():
        raise RuntimeError("bounded one-pass run directory already exists")
    model_start = time.perf_counter()
    deadline = model_start + int(config["authority"]["combined_model_wall_cap_seconds"])
    results = [run_source(source, config, policy, native, deadline) for source in config["sources"]]
    model_wall = time.perf_counter() - model_start
    total_audio = sum(int(row["source"]["execution_samples"]) for row in results) / 16000
    summary = {"schema": "PSEM-STREAMING-STUDENT-TEACHER-BASELINE-RESULT-1", "status": "completed", "scope": "#164 authorized non-training Vulkan teacher/receiver baseline only", "authorization": config["authority"], "identities": {"config_sha256": digest(CONFIG_PATH), "executable_sha256": digest(Path(config["native"]["executable"])), "model_sha256": digest(Path(config["native"]["model"])), "runtime_archive_sha256": digest(ROOT / config["receiver"]["runtime_archive"]), "ownership_override_sha256": digest(ROOT / config["receiver"]["ownership_override"]), "decoder_sha256": digest(ROOT / config["receiver"]["decoder"])}, "execution": {"sources": [row["meeting"] for row in results], "native_passes_per_source": 1, "approved_audio_s": total_audio, "combined_model_execution_wall_s": model_wall, "cap_s": config["authority"]["combined_model_wall_cap_seconds"], "cap_held": model_wall <= config["authority"]["combined_model_wall_cap_seconds"], "training_backward_updates": 0, "paid_or_cloud_calls": 0, "holdout_eval_opened": False}, "measurements": {"native_transition_events": {row["meeting"]: row["native"]["transition_events"] for row in results}, "annotation_conditions": {row["meeting"]: row["source"]["annotation_coverage"] for row in results}, "selected_changed_parents": {row["meeting"]: row["receiver"]["selected_changed_parents"] for row in results}, "selected_suppressed_boundaries": {row["meeting"]: row["receiver"]["selected_suppressed_boundaries"] for row in results}, "text_conservation_failures": {row["meeting"]: row["receiver"]["text_conservation_failures"] for row in results}, "cost": {row["meeting"]: row["cost"] for row in results}}, "decision": {"baseline_target_usable": all(row["native"]["output_frames"] > 0 and row["receiver"]["assignments"] > 0 and row["receiver"]["text_conservation_failures"] == 0 for row in results), "compression_training": "not run; awaiting later discussion", "quality_scope": "Engineering target only. Transition and guard conditions are those actually found in the approved prefixes; no universal teacher-quality or translation-effect claim.", "remaining_obligation": "Student learning interface and GT/KD training comparison remain outside this authorized baseline."}, "architecture": {"product_source_changed": False, "production_160_duplicated": False, "runtime_boundary": "Experiment-local pinned archive plus ownership override; no product mutation.", "api_or_display_evidence": "none"}, "source_results": [str((RUNS / row["meeting"] / "RESULT.json").relative_to(ROOT)) for row in results]}
    emit(HERE / "RESULT.json", summary)
    return summary


def verify(config: dict[str, Any]) -> dict[str, Any]:
    summary = json.loads((HERE / "RESULT.json").read_text(encoding="utf-8"))
    failures = []
    if not summary["execution"]["cap_held"] or summary["execution"]["native_passes_per_source"] != 1:
        failures.append("execution envelope")
    if summary.get("decision", {}).get("disposition") != "SUPPORTED_NAMED_TIMING_MAPPING_FAILURE":
        failures.append("supported blocker disposition")
    for meeting in summary["execution"]["sources"]:
        row = json.loads((RUNS / meeting / "RESULT.json").read_text(encoding="utf-8"))
        if row["identity"]["effective_profile"] != config["native"]["profile"]:
            failures.append(f"{meeting} profile")
        if row["receiver"]["text_conservation_failures"]:
            failures.append(f"{meeting} text conservation")
        causal = row.get("receiver", {}).get("causal_analysis") or {}
        if not row.get("clock", {}).get("all_native_availability_receiver_receipt_and_admission_values_share_qpc"):
            failures.append(f"{meeting} shared clock")
        if not causal.get("parents_with_timely_native_event_inside") or not causal.get("timely_event_parents_blocked_by_receiver"):
            failures.append(f"{meeting} causal blocker evidence")
        for artifact in row["artifacts"].values():
            if digest(ROOT / artifact["path"]) != artifact["sha256"]:
                failures.append(f"{meeting} artifact identity")
    result = {"status": "passed" if not failures else "failed", "failures": failures, "checks": ["finite authorized envelope", "one native pass per source", "effective profile", "selected-policy text conservation", "same-QPC causal blocker evidence", "raw and receiver artifact identities"]}
    emit(HERE / "VERIFICATION.json", result)
    if failures:
        raise RuntimeError(str(failures))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "execute", "verify"))
    args = parser.parse_args()
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    value = prepare(config) if args.command == "prepare" else execute(config) if args.command == "execute" else verify(config)
    print(json.dumps(value, ensure_ascii=False, separators=(",", ":")))


if __name__ == "__main__":
    main()
