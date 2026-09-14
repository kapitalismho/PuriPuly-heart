from __future__ import annotations

import argparse
import csv
import ctypes
import gzip
import hashlib
import importlib.abc
import importlib.util
import json
import math
import re
import os
import shutil
import socket
import subprocess
import sys
import tarfile
import struct
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
OUTPUT_ROOT = HERE
RUNS = OUTPUT_ROOT / "runs"
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
    path.write_bytes((json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))

def configure_paths(config_path: Path, config: dict[str, Any]) -> None:
    global CONFIG_PATH, OUTPUT_ROOT, RUNS
    CONFIG_PATH = config_path.resolve()
    configured = config.get("output_root")
    OUTPUT_ROOT = (ROOT / configured).resolve() if configured else HERE
    if OUTPUT_ROOT != HERE and HERE not in OUTPUT_ROOT.parents:
        raise RuntimeError("output_root must remain inside the experiment directory")
    RUNS = OUTPUT_ROOT / "runs"


def current_source_revision() -> str:
    completed = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True)
    return completed.stdout.strip()


def execution_identities(config: dict[str, Any], source_revision: str | None) -> dict[str, Any]:
    return {
        "source_revision": source_revision or "pending Director commit; execution prohibited",
        "committed_runner_config_analysis_verified": source_revision is not None,
        "config_sha256": digest(CONFIG_PATH),
        "runner_sha256": digest(Path(__file__)),
        "analysis_sha256": digest(HERE / "analyze_baseline.py"),
        "executable_sha256": digest(Path(config["native"]["executable"])),
        "model_sha256": digest(Path(config["native"]["model"])),
        "runtime_archive_sha256": digest(ROOT / config["receiver"]["runtime_archive"]),
        "ownership_override_sha256": digest(ROOT / config["receiver"]["ownership_override"]),
        "decoder_sha256": digest(ROOT / config["receiver"]["decoder"]),
    }

def matches_committed_blob(revision: str, path: Path) -> bool:
    relative = path.resolve().relative_to(ROOT).as_posix()
    committed = subprocess.run(["git", "rev-parse", f"{revision}:{relative}"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()
    current = subprocess.run(["git", "hash-object", relative], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()
    return current == committed
def committed_blob_digest(revision: str, path: Path) -> str:
    relative = path.resolve().relative_to(ROOT).as_posix()
    completed = subprocess.run(["git", "show", f"{revision}:{relative}"], cwd=ROOT, check=True, capture_output=True)
    return hashlib.sha256(completed.stdout).hexdigest()


def verify_execution_identities(config: dict[str, Any], recorded: dict[str, Any], postprocessing: dict[str, Any]) -> list[str]:
    revision = recorded.get("source_revision")
    failures = []
    committed_paths = {
        "runner_sha256": Path(__file__),
        "analysis_sha256": HERE / "analyze_baseline.py",
        "config_sha256": CONFIG_PATH,
    }
    if recorded.get("committed_runner_config_analysis_verified") is not True or not isinstance(revision, str) or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        failures.append("execution identities")
    else:
        try:
            if any(recorded.get(name) != committed_blob_digest(revision, path) for name, path in committed_paths.items()):
                failures.append("execution identities")
        except (subprocess.CalledProcessError, ValueError):
            failures.append("execution identities")
    retained_paths = {
        "config_sha256": CONFIG_PATH,
        "executable_sha256": Path(config["native"]["executable"]),
        "model_sha256": Path(config["native"]["model"]),
        "runtime_archive_sha256": ROOT / config["receiver"]["runtime_archive"],
        "ownership_override_sha256": ROOT / config["receiver"]["ownership_override"],
        "decoder_sha256": ROOT / config["receiver"]["decoder"],
    }
    if any(recorded.get(name) != digest(path) for name, path in retained_paths.items()):
        failures.append("execution identities")
    current_analysis = digest(HERE / "analyze_baseline.py")
    if recorded.get("posthoc_analysis_sha256") != current_analysis or postprocessing.get("posthoc_analysis_sha256") != current_analysis:
        failures.append("posthoc reporting identity")
    return list(dict.fromkeys(failures))


def validate_source_revision(source_revision: str | None) -> str:
    if source_revision is None or re.fullmatch(r"[0-9a-f]{40}", source_revision) is None:
        raise RuntimeError("execute requires --source-revision with the exact committed 40-character revision")
    current = current_source_revision()
    if current != source_revision:
        raise RuntimeError(f"source revision mismatch: requested {source_revision}, current {current}")
    for path in (Path(__file__), HERE / "analyze_baseline.py", CONFIG_PATH):
        if not matches_committed_blob(current, path):
            raise RuntimeError(f"execution input is not the committed revision: {path}")
    return current


def timing_summary(values: list[int]) -> dict[str, Any]:
    ordered = sorted(values)
    return {"samples": len(ordered), "sum_us": sum(ordered), "min_us": ordered[0], "median_us": ordered[len(ordered) // 2], "max_us": ordered[-1]}


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


def run_source(source: dict[str, Any], config: dict[str, Any], policy: Any, native: Any, absolute_deadline: float, source_revision: str) -> dict[str, Any]:
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
    server.settimeout(0.01)
    port = server.getsockname()[1]
    dump_dir = run_dir / "dump"
    dump_dir.mkdir()
    env = os.environ.copy()
    for key in (*PROFILE_KEYS.values(), "TRANSCRIBE_PSEM_PACE_16KHZ", "TRANSCRIBE_PSEM_EVENTS_TCP", "TRANSCRIBE_PSEM_PACE_FROM_SAMPLE", "TRANSCRIBE_PSEM_PACE_UNTIL_SAMPLE", "TRANSCRIBE_DUMP_DIR"):
        env.pop(key, None)
    env.update({"TRANSCRIBE_PSEM_PACE_16KHZ": "1", "TRANSCRIBE_PSEM_EVENTS_TCP": f"127.0.0.1:{port}", "TRANSCRIBE_PSEM_CAUSAL_FRONTEND": "1", "TRANSCRIBE_PSEM_PACE_FROM_SAMPLE": "0", "TRANSCRIBE_PSEM_PACE_UNTIL_SAMPLE": str(source["execution_samples"]), "TRANSCRIBE_DUMP_DIR": str(dump_dir)})
    for name, key in PROFILE_KEYS.items():
        env[key] = str(config["native"]["profile"][name])
    command = [config["native"]["executable"], "-m", config["native"]["model"], "--backend", config["native"]["backend"], str(projection)]
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
        child_handle = int(process._handle)
        memory_samples = []

        def sample_memory() -> None:
            native_working, _native_peak = process_memory(child_handle)
            wrapper_working, _wrapper_peak = process_memory(wrapper_handle)
            memory_samples.append({"qpc": qpc_now(query, counter), "native_working_set_bytes": native_working, "wrapper_working_set_bytes": wrapper_working, "aggregate_working_set_bytes": native_working + wrapper_working})

        try:
            accept_deadline = time.perf_counter() + 60
            while conn is None:
                if process.poll() is not None:
                    raise RuntimeError("native process exited before receiver connection")
                if time.perf_counter() >= accept_deadline:
                    raise TimeoutError("native receiver connection timeout")
                try:
                    conn, peer = server.accept()
                except socket.timeout:
                    sample_memory()
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
            sample_memory()

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
                sample_memory()
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
    memory_path = run_dir / "process-memory-samples.csv"
    with memory_path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=("qpc", "native_working_set_bytes", "wrapper_working_set_bytes", "aggregate_working_set_bytes"))
        writer.writeheader()
        writer.writerows(memory_samples)
    trace_path = dump_dir / "diar.trace.json"
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    effective = {name: trace[name] for name in PROFILE_KEYS}
    chunks = [row for row in messages if row.get("type") == "chunk"]
    receipts = [(int(row["receiver_receipt_qpc"]) - int(ready["qpc"])) / int(ready["qpf"]) for row in chunks]
    support_lag = [receipt - int(row["raw_support_end_sample"]) / 16000 for receipt, row in zip(receipts, chunks)]
    removed = sum(len(row["selected_unknown_only"]["removed_boundary_token_indexes"]) for row in assignments)
    changed = sum(row["selected_unknown_only"]["units"] != row["receiver"]["pre_selected_policy_units"] for row in assignments)
    trace_chunks = trace["chunks"]
    trace_timing = {name: timing_summary([int(row[name]) for row in trace_chunks]) for name in ("service_us", "frontend_us", "graph_a_us", "graph_b_us", "host_us")}
    result = {
        "schema": "PSEM-STREAMING-STUDENT-TEACHER-SOURCE-RUN-1",
        "meeting": meeting,
        "evidence_labels": {"native": "actual local Vulkan native pass", "receiver": "actual prospective receiver invocation on frozen accepted-text replay", "translation_admission": "actual experiment cutoff invocation in the shared QPC clock", "api": "not run", "display": "not run"},
        "source": {**source, "wav_sha256": digest(Path(source["wav"])), "projection_sha256": digest(projection), "projection_bytes": projection.stat().st_size, "parents_admitted": len(assignments), "annotation_coverage": annotations},
        "identity": {**execution_identities(config, source_revision), "command": command, "effective_profile": effective, "generation": generation, "capture_epoch": 1, "reset_contract": "one native process and one receiver generation from source zero through the complete approved source; logical parent admissions do not reset model/reference", "genuine_discontinuity": "source end only; no reconnect or synthetic seal reset occurred"},
        "clock": {"kind": "Windows QueryPerformanceCounter", "qpf": int(ready["qpf"]), "native_source_zero_qpc": int(ready["qpc"]), "native_done_qpc": int(done["qpc"]), "process_start_qpc": started_qpc, "process_finish_qpc": finished_qpc, "all_native_availability_receiver_receipt_and_admission_values_share_qpc": True, "receiver_transport_delay_observed_not_rewritten": True},
        "native": {"exit_code": exit_code, "messages": len(messages), "chunks": len(chunks), "output_frames": sum(int(row["emit_count"]) for row in chunks), "soft_output_values": sum(len(row["probs"]) * len(row["probs"][0]) for row in chunks if row.get("probs")), "maximum_visible_sample": max(int(row["n_visible"]) for row in chunks), "maximum_consumed_support_sample": max(int(row["raw_support_end_sample"]) for row in chunks), "transition_events": len(transitions), "transitions": transitions, "receiver_support_lag_s": {"min": min(support_lag), "median": sorted(support_lag)[len(support_lag) // 2], "max": max(support_lag)}},
        "receiver": {"assignments": len(assignments), "source_only_or_empty": sum(not row["accepted_text"] for row in assignments), "text_conservation_failures": sum(not row["selected_unknown_only"]["text_conserved"] for row in assignments), "selected_suppressed_boundaries": removed, "selected_changed_parents": changed, "late_event_references": sum(len(row["receiver"]["late_ignored"]) for row in assignments), "admission_scheduling_lateness_s": {"max": max(row["clock"]["scheduling_lateness_s"] for row in assignments), "median": sorted(row["clock"]["scheduling_lateness_s"] for row in assignments)[len(assignments) // 2]}},
        "cost": {
            "complete_path_wall_s": (finished_qpc - started_qpc) / local_qpf,
            "wall_interpretation": "Paced end-to-end process/receiver wall time; not compute time or compute RTF.",
            "native_process_cpu_s": native_cpu,
            "receiver_wrapper_cpu_s": wrapper_cpu,
            "native_peak_working_set_bytes_observed": max(row["native_working_set_bytes"] for row in memory_samples),
            "receiver_wrapper_peak_working_set_bytes_observed": max(row["wrapper_working_set_bytes"] for row in memory_samples),
            "complete_path_peak_sum_working_set_bytes_sampled": max(row["aggregate_working_set_bytes"] for row in memory_samples),
            "rss_sampling": "Native and wrapper working sets are paired in each QPC-stamped sample; the aggregate peak is the maximum paired sample, not a sum of independent peaks.",
            "memory_samples": len(memory_samples),
            "native_trace_timing_us": {**trace_timing, "initialization_us": int(trace["initialization_us"]), "load_us": int(trace["load_us"]), "interpretation": "Native-reported trace timers. service_us includes paced service elapsed; component timers are retained by name and are not labeled GPU-kernel compute or compute RTF."},
            "gpu_memory": parse_gpu(gpu_path, process.pid),
        },
        "artifacts": {
            "raw_frame_soft_outputs": {"path": str(raw_path.relative_to(ROOT)), "sha256": digest(raw_path)},
            "receiver_assignments": {"path": str(assignments_path.relative_to(ROOT)), "sha256": digest(assignments_path)},
            "native_trace": {"path": str(trace_path.relative_to(ROOT)), "sha256": digest(trace_path)},
            "soft_target_tensor": {"path": str((dump_dir / "diar.probs.f32").relative_to(ROOT)), "sha256": digest(dump_dir / "diar.probs.f32")},
            "soft_target_metadata": {"path": str((dump_dir / "diar.probs.json").relative_to(ROOT)), "sha256": digest(dump_dir / "diar.probs.json")},
            "process_memory_samples": {"path": str(memory_path.relative_to(ROOT)), "sha256": digest(memory_path)},
        },
    }
    emit(run_dir / "RESULT.json", result)
    return result


def prepare(config: dict[str, Any], source_revision: str | None = None) -> dict[str, Any]:
    executable = Path(config["native"]["executable"])
    model = Path(config["native"]["model"])
    if digest(executable) != config["native"]["executable_sha256"] or digest(model) != config["native"]["model_sha256"]:
        raise RuntimeError("native executable or model identity mismatch")
    authority = config["authority"]
    if int(authority["combined_model_wall_cap_seconds"]) != 1800 or authority["training_updates"] != 0 or authority["paid_api_calls"] != 0 or authority["holdout_eval_access"]:
        raise RuntimeError("execution envelope mismatch")
    if authority.get("run_id") == "#164-BASELINE-RERUN-1" and (authority.get("additional_native_passes_per_source") != 1 or OUTPUT_ROOT == HERE):
        raise RuntimeError("rerun output/envelope mismatch")
    checks = []
    for source in config["sources"]:
        path = Path(source["wav"])
        checks.append({"meeting": source["meeting"], "wav_exists": path.is_file(), "approved_geometry": int(source["execution_samples"]) == int(source["evaluation_samples"]) + int(source["real_context_tail_samples"])})
    if not all(row["wav_exists"] and row["approved_geometry"] for row in checks):
        raise RuntimeError("source readiness failed")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    result = {
        "status": "ready_to_execute" if source_revision else "ready_for_director_commit",
        "execution_permitted": source_revision is not None,
        "identities": execution_identities(config, source_revision),
        "checks": checks,
        "output_root": str(OUTPUT_ROOT.relative_to(ROOT)),
        "runs_path": str(RUNS.relative_to(ROOT)),
        "model_execution_cap_s": 1800,
        "additional_native_passes_per_source": authority.get("additional_native_passes_per_source", 0),
        "training": "not authorized and not run",
        "network_api": "not authorized and not run",
    }
    emit(OUTPUT_ROOT / "READINESS.json", result)
    return result


def smoke(config: dict[str, Any]) -> dict[str, Any]:
    wrapper_handle = open_process_handle(os.getpid())
    child = subprocess.Popen([sys.executable, "-B", "-c", "sum(i*i for i in range(2000000)); import time; time.sleep(0.1)"])
    try:
        child_handle = int(child._handle)
        native_working, _ = process_memory(child_handle)
        wrapper_working, _ = process_memory(wrapper_handle)
        sample = {"native_working_set_bytes": native_working, "wrapper_working_set_bytes": wrapper_working, "aggregate_working_set_bytes": native_working + wrapper_working}
        child.wait(timeout=5)
        child_cpu = cpu_seconds(child_handle)
        wrapper_cpu = cpu_seconds(wrapper_handle)
    finally:
        if child.poll() is None:
            child.terminate()
            child.wait(timeout=5)
        ctypes.windll.kernel32.CloseHandle(ctypes.c_void_p(wrapper_handle))
    policy, native = load_runtime(config)
    decoder = native.LiveTransitionDecoder()
    decoder.ingest_chunk(0, [[0.9, 0.1, 0.1, 0.1], [0.1, 0.9, 0.1, 0.1], [0.1, 0.9, 0.1, 0.1]], available_at_monotonic_s=0.0, receipt_kind="no-model-smoke")
    receiver = policy.PretranslationOwnershipOwner(enabled=True, tombstone_capacity=int(config["receiver"]["evidence_capacity"]))
    if min(sample.values()) <= 0 or child_cpu <= 0 or wrapper_cpu < 0 or len(decoder.events) != 1 or receiver is None:
        raise RuntimeError("no-model instrumentation/orchestrator smoke failed")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    result = {"status": "passed", "model_execution": False, "training_or_backward": False, "process_measurement": sample, "child_cpu_s": child_cpu, "wrapper_cpu_total_s": wrapper_cpu, "synthetic_decoder_events": len(decoder.events), "runtime_identities": execution_identities(config, None)}
    emit(OUTPUT_ROOT / "PREEXECUTION_SMOKE.json", result)
    return result


def execute(config: dict[str, Any], source_revision: str | None) -> dict[str, Any]:
    revision = validate_source_revision(source_revision)
    prepare(config, revision)
    if RUNS.exists():
        raise RuntimeError("bounded one-pass rerun directory already exists")
    policy, native = load_runtime(config)
    model_start = time.perf_counter()
    deadline = model_start + int(config["authority"]["combined_model_wall_cap_seconds"])
    results = [run_source(source, config, policy, native, deadline, revision) for source in config["sources"]]
    model_wall = time.perf_counter() - model_start
    total_audio = sum(int(row["source"]["execution_samples"]) for row in results) / 16000
    summary = {"schema": "PSEM-STREAMING-STUDENT-TEACHER-BASELINE-RESULT-1", "status": "completed_pending_analysis", "scope": config["authority"].get("run_id", "#164 baseline"), "authorization": config["authority"], "identities": execution_identities(config, revision), "execution": {"sources": [row["meeting"] for row in results], "native_passes_per_source": 1, "approved_audio_s": total_audio, "combined_paced_orchestration_wall_s": model_wall, "cap_s": config["authority"]["combined_model_wall_cap_seconds"], "cap_held": model_wall <= config["authority"]["combined_model_wall_cap_seconds"], "training_backward_updates": 0, "paid_or_cloud_calls": 0, "holdout_eval_opened": False}, "measurements": {"native_transition_events": {row["meeting"]: row["native"]["transition_events"] for row in results}, "annotation_conditions": {row["meeting"]: row["source"]["annotation_coverage"] for row in results}, "selected_changed_parents": {row["meeting"]: row["receiver"]["selected_changed_parents"] for row in results}, "selected_suppressed_boundaries": {row["meeting"]: row["receiver"]["selected_suppressed_boundaries"] for row in results}, "text_conservation_failures": {row["meeting"]: row["receiver"]["text_conservation_failures"] for row in results}, "cost": {row["meeting"]: row["cost"] for row in results}}, "decision": {"baseline_target_usable": all(row["native"]["output_frames"] > 0 and row["receiver"]["assignments"] > 0 and row["receiver"]["text_conservation_failures"] == 0 for row in results), "compression_training": "not run; GPU training method discussion selected first", "quality_scope": "Additional cost measurement only; no universal teacher-quality or translation-effect claim."}, "architecture": {"product_source_changed": False, "production_160_duplicated": False, "runtime_boundary": "Experiment-local pinned archive plus ownership override; no product mutation.", "api_or_display_evidence": "none"}, "source_results": [str((RUNS / row["meeting"] / "RESULT.json").relative_to(ROOT)) for row in results]}
    emit(OUTPUT_ROOT / "RESULT.json", summary)
    return summary


def read_jsonl_gz(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as source:
        return [json.loads(line) for line in source]


def decode_recorded_transitions(messages: list[dict[str, Any]], threshold: float, confirmation_samples: int, frame_samples: int) -> list[dict[str, int]]:
    last = None
    pending = None
    pending_start = None
    pending_samples = 0
    previous_end = None
    events = []
    for message in messages:
        if message.get("type") != "chunk":
            continue
        for offset, probabilities in enumerate(message["probs"]):
            labels = [index for index, value in enumerate(probabilities) if float(value) >= threshold]
            label = labels[0] if len(labels) == 1 else None
            frame = int(message["emit_start_frame"]) + offset
            start = frame * frame_samples
            end = start + frame_samples
            if label is None:
                pending, pending_samples, previous_end = None, 0, None
            elif last is None:
                last = label
                pending, pending_samples, previous_end = None, 0, None
            elif label == last:
                pending, pending_samples, previous_end = None, 0, None
            else:
                if previous_end is not None and start != previous_end:
                    pending, pending_samples, previous_end = None, 0, None
                if pending is None or pending != label:
                    pending, pending_start, pending_samples, previous_end = label, start, 0, start
                need = confirmation_samples - pending_samples
                if frame_samples >= need:
                    events.append({"boundary_sample": int(pending_start if pending_start is not None else start), "frontier_sample": end, "candidate_slot": label, "native_qpc": int(message["qpc"]), "receipt_qpc": int(message["receiver_receipt_qpc"])})
                    last = label
                    pending, pending_samples, previous_end = None, 0, None
                else:
                    pending_samples += frame_samples
                    previous_end = end
    return events


def verify_recorded_source(run_dir: Path, row: dict[str, Any], config: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
    failures = []
    messages = read_jsonl_gz(run_dir / "native-events.jsonl.gz")
    assignments = read_jsonl_gz(run_dir / "receiver-assignments.jsonl.gz")
    chunks = [message for message in messages if message.get("type") == "chunk"]
    ready = [message for message in messages if message.get("type") == "ready"]
    qpf_values = {int(message["qpf"]) for message in messages if "qpf" in message}
    if len(ready) != 1 or qpf_values != {int(row["clock"]["qpf"])}:
        failures.append("raw QPC provenance")
        return failures, {}
    decoded = decode_recorded_transitions(messages, float(config["native"]["threshold"]), int(config["native"]["confirmation_samples"]), int(config["native"]["frame_samples"]))
    recorded = row["native"]["transitions"]
    decoded_facts = [(event["boundary_sample"], event["frontier_sample"], event["candidate_slot"], event["native_qpc"], event["receipt_qpc"]) for event in decoded]
    recorded_facts = [(int(event["boundary_sample"]), int(event["consumed_audio_frontier_sample"]), int(event["candidate_native_slot"]), int(event["native_availability_qpc"]), int(event["receiver_receipt_qpc"])) for event in recorded]
    if decoded_facts != recorded_facts:
        failures.append("raw transition reconstruction")
    timely_parents = set()
    cutoff_lags = []
    qpf = next(iter(qpf_values))
    for assignment in assignments:
        expected_cutoff = int(ready[0]["qpc"]) + round(int(assignment["source_span"][1]) * qpf / int(config["native"]["sample_rate"]))
        if int(assignment["clock"]["scheduled_cutoff_qpc"]) != expected_cutoff or float(assignment["clock"]["added_wait_s"]) != 0:
            failures.append("raw no-wait cutoff mapping")
        end_frame = int(assignment["source_span"][1]) // int(config["native"]["frame_samples"])
        containing = [message for message in chunks if int(message["emit_start_frame"]) <= end_frame < int(message["emit_start_frame"]) + int(message["emit_count"])]
        if len(containing) != 1:
            failures.append("raw end-frame coverage")
            continue
        lag = (int(containing[0]["receiver_receipt_qpc"]) - int(assignment["clock"]["scheduled_cutoff_qpc"])) / qpf
        cutoff_lags.append(lag)
        if lag <= 0:
            failures.append("raw cutoff coverage direction")
        for event in decoded:
            if int(assignment["source_span"][0]) < event["boundary_sample"] < int(assignment["source_span"][1]) and event["receipt_qpc"] <= int(assignment["clock"]["actual_admission_qpc"]):
                timely_parents.add(assignment["parent_id"])
    causal = row["receiver"]["causal_analysis"]
    if len(timely_parents) != int(causal["parents_with_timely_native_event_inside"]):
        failures.append("raw timely transition count")
    if len(cutoff_lags) != int(causal["cutoff_coverage"]["parents_whose_end_covering_native_frame_arrived_after_cutoff"]):
        failures.append("raw cutoff parent count")
    metadata = json.loads((run_dir / "dump" / "diar.probs.json").read_text(encoding="utf-8"))
    tensor = (run_dir / "dump" / "diar.probs.f32").read_bytes()
    raw_probabilities = [float(value) for message in chunks for probability_row in message["probs"] for value in probability_row]
    if metadata.get("dtype") != "f32" or metadata.get("layout") != "row-major" or metadata.get("shape") != [len(raw_probabilities) // 4, 4] or len(tensor) != 4 * len(raw_probabilities):
        failures.append("soft tensor geometry")
    else:
        values = struct.unpack(f"<{len(raw_probabilities)}f", tensor)
        if any(not math.isfinite(value) or abs(value - raw) > 1e-7 for value, raw in zip(values, raw_probabilities)):
            failures.append("soft tensor/raw correspondence")
    facts = {"parents": len(assignments), "timely_transition_parents": len(timely_parents), "parents_missing_end_frame_at_cutoff": sum(lag > 0 for lag in cutoff_lags), "scheduled_cutoff_end_frame_lag_s": {"min": min(cutoff_lags), "max": max(cutoff_lags)}, "soft_values": len(raw_probabilities), "soft_slots": 4}
    return failures, facts


def verify(config: dict[str, Any]) -> dict[str, Any]:
    summary = json.loads((OUTPUT_ROOT / "RESULT.json").read_text(encoding="utf-8"))
    failures = []
    raw_facts = {}
    paired_memory_checked = False
    if not summary["execution"]["cap_held"] or summary["execution"]["native_passes_per_source"] != 1:
        failures.append("execution envelope")
    if summary.get("decision", {}).get("disposition") != "CUTOFF_CONDITIONAL_PARTIAL_BASELINE" or not summary.get("decision", {}).get("baseline_target_usable"):
        failures.append("partial baseline disposition")
    recorded_identities = summary.get("identities", {})
    executed_identity_names: tuple[str, ...] = ()
    if "runner_sha256" in recorded_identities:
        executed_identity_names = ("source_revision", "committed_runner_config_analysis_verified", "config_sha256", "runner_sha256", "analysis_sha256", "executable_sha256", "model_sha256", "runtime_archive_sha256", "ownership_override_sha256", "decoder_sha256")
        failures.extend(verify_execution_identities(config, recorded_identities, summary.get("postprocessing", {})))
    for meeting in summary["execution"]["sources"]:
        run_dir = RUNS / meeting
        row = json.loads((run_dir / "RESULT.json").read_text(encoding="utf-8"))
        if row["identity"]["effective_profile"] != config["native"]["profile"]:
            failures.append(f"{meeting} profile")
        if executed_identity_names and any(row["identity"].get(name) != recorded_identities.get(name) for name in executed_identity_names):
            failures.append(f"{meeting} execution identities")
        if row["receiver"]["text_conservation_failures"]:
            failures.append(f"{meeting} text conservation")
        source_failures, raw_facts[meeting] = verify_recorded_source(run_dir, row, config)
        failures.extend(f"{meeting} {failure}" for failure in source_failures)
        for artifact in row["artifacts"].values():
            if digest(ROOT / artifact["path"]) != artifact["sha256"]:
                failures.append(f"{meeting} artifact identity")
        memory_artifact = row["artifacts"].get("process_memory_samples")
        if memory_artifact:
            paired_memory_checked = True
            with (ROOT / memory_artifact["path"]).open("r", encoding="utf-8", newline="") as source:
                memory_rows = [{key: int(value) for key, value in item.items()} for item in csv.DictReader(source)]
            if not memory_rows or any(item["aggregate_working_set_bytes"] != item["native_working_set_bytes"] + item["wrapper_working_set_bytes"] for item in memory_rows):
                failures.append(f"{meeting} paired memory samples")
            elif max(item["aggregate_working_set_bytes"] for item in memory_rows) != row["cost"].get("complete_path_peak_sum_working_set_bytes_sampled", row["cost"].get("complete_path_peak_sum_working_set_bytes_observed")):
                failures.append(f"{meeting} aggregate memory peak")
    if sum(value.get("parents", 0) for value in raw_facts.values()) != 38 or sum(value.get("parents_missing_end_frame_at_cutoff", 0) for value in raw_facts.values()) != 38:
        failures.append("aggregate raw causal facts")
    checks = ["finite authorized envelope", "one native pass per source"]
    if executed_identity_names:
        checks.extend(["executed runner/config/analyzer identities against immutable source-revision blobs", "current posthoc reporting analyzer identity"])
    checks.extend(["effective profile", "selected-policy text conservation", "raw QPC cutoff and transition reconstruction", "four-slot soft tensor/raw correspondence"])
    if paired_memory_checked:
        checks.append("back-to-back near-simultaneous memory aggregate")
    checks.append("artifact manifest identities")
    identity_verification = {
        "executed_identity_provenance": "validated against immutable committed blobs" if executed_identity_names and "execution identities" not in failures else "unavailable for original run" if not executed_identity_names else "failed",
        "executed_source_revision": recorded_identities.get("source_revision") if executed_identity_names else None,
        "executed_analysis_sha256": recorded_identities.get("analysis_sha256") if executed_identity_names else None,
        "current_posthoc_analysis_sha256": digest(HERE / "analyze_baseline.py") if executed_identity_names else None,
    }
    result = {
        "status": "passed" if not failures else "failed",
        "failures": failures,
        "checks": checks,
        "identity_verification": identity_verification,
        "raw_recomputed": raw_facts,
    }
    emit(OUTPUT_ROOT / "VERIFICATION.json", result)
    if failures:
        raise RuntimeError(str(failures))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "smoke", "execute", "verify"))
    parser.add_argument("--config", type=Path, default=HERE / "baseline_config.json")
    parser.add_argument("--source-revision")
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    configure_paths(config_path, config)
    if args.command == "prepare":
        value = prepare(config)
    elif args.command == "smoke":
        value = smoke(config)
    elif args.command == "execute":
        value = execute(config, args.source_revision)
    else:
        value = verify(config)
    print(json.dumps(value, ensure_ascii=False, separators=(",", ":")))


if __name__ == "__main__":
    main()
