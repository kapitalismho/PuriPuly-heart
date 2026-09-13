from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.util
import json
import secrets
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
CASES_FILE = HERE / "cases.jsonl"
PLAN_FILE = HERE / "plan.json"
DEFAULT_EXECUTION_DIR = HERE / "execution"
OFFICIAL_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
MAX_ATTEMPTS = 224
EXPECTED_ATTEMPTS = 221
MAX_TOTAL_OUTPUT_TOKENS = 22400
MAX_TOKENS = 100
RAW_RESPONSE_LIMIT = 131072
MODEL = "google/gemma-4-26b-a4b-it"
PROVIDER = {"order": ["wafer", "cloudflare", "deepinfra"], "only": ["wafer", "cloudflare", "deepinfra"], "allow_fallbacks": True}
SEVERE_KINDS = {
    "number loss or invention",
    "question or statement reversal",
    "negation or agreement reversal",
    "wrong claim attribution",
    "material omission or invention",
}

_HELPER_PATH = HERE.parent / "meaning-first-probe/run_e1.py"
_SPEC = importlib.util.spec_from_file_location("unknown_partition_durable_helpers", _HELPER_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("cannot load durable runner helpers")
_HELPERS = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_HELPERS)
durable_write = _HELPERS.durable_write
durable_json = _HELPERS.durable_json
append_event = _HELPERS.append_event
execution_lock = _HELPERS.execution_lock
credential = _HELPERS.credential
validate_endpoint = _HELPERS.validate_endpoint
historical_body = _HELPERS._PREPARE.historical_body


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_cases() -> list[dict[str, Any]]:
    return [json.loads(line) for line in CASES_FILE.read_text(encoding="utf-8").splitlines() if line]


def execution_mode(directory: Path, endpoint: str) -> tuple[Path, str]:
    resolved = directory.expanduser().resolve()
    canonical_directory = DEFAULT_EXECUTION_DIR.resolve()
    if endpoint == OFFICIAL_ENDPOINT:
        if resolved != canonical_directory:
            raise RuntimeError("official endpoint requires the canonical execution directory")
        return resolved, "official_provider"
    if resolved == canonical_directory:
        raise RuntimeError("canonical execution directory accepts only the official endpoint")
    return resolved, "loopback_rehearsal"


def instance_id(parent_id: str, replicate: int, body_sha256: str, occurrence_rank: int) -> str:
    return digest_bytes(canonical([parent_id, replicate, body_sha256, occurrence_rank]).encode("utf-8"))


def validate_frozen() -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    plan = load_json(PLAN_FILE)
    cases = load_cases()
    if plan.get("schema") != "UNKNOWN-PARTITION-PROBE-PLAN-1" or plan.get("baseline") != "f0a650d17cf7463097b24b01d9d03e4223c83b78":
        raise RuntimeError("plan identity mismatch")
    if digest_file(CASES_FILE) != plan.get("cases_sha256"):
        raise RuntimeError("cases catalog hash mismatch")
    authority = plan.get("authority")
    expected_authority = {
        "maximum_http_attempts": MAX_ATTEMPTS,
        "expected_catalog_attempts": EXPECTED_ATTEMPTS,
        "maximum_output_tokens": MAX_TOTAL_OUTPUT_TOKENS,
        "max_tokens_per_attempt": MAX_TOKENS,
        "attempts_per_instance": 1,
        "automatic_retries": 0,
        "account_or_pricing_queries": False,
        "old_budget_ledger_mutation": False,
        "native_or_asr_calls": False,
        "holdout_access": False,
        "production_or_remote_mutation": False,
    }
    if not isinstance(authority, dict) or any(authority.get(key) != value for key, value in expected_authority.items()):
        raise RuntimeError("finite execution authority mismatch")
    if len(cases) != 20 or sum(case.get("removed_boundary_count", 0) > 0 for case in cases) != 18:
        raise RuntimeError("case coverage mismatch")
    catalog: dict[str, dict[str, Any]] = {}
    logical_bindings = 0
    for case in cases:
        if case.get("schema") != "UNKNOWN-PARTITION-PROBE-CASE-1" or case.get("replicates") not in {1, 3}:
            raise RuntimeError("case schema or repeat mismatch")
        accepted = case.get("accepted_text")
        if not isinstance(accepted, str) or digest_bytes(accepted.encode("utf-8")) != case.get("accepted_text_sha256"):
            raise RuntimeError("accepted source identity mismatch")
        for arm in ("baseline_units", "intervention_units"):
            units = case.get(arm)
            if not isinstance(units, list) or "".join(unit.get("text", "") for unit in units) != accepted:
                raise RuntimeError("unit text conservation mismatch")
            if [unit.get("request_occurrence") for unit in units] != list(range(len(units))):
                raise RuntimeError("request occurrence order mismatch")
            body_counts: dict[str, int] = {}
            for unit in units:
                prompt = plan.get("historical_request", {}).get("system_prompt")
                if not isinstance(prompt, str) or len(prompt) != 3354 or digest_bytes(prompt.encode("utf-8")) != "21415c9498366a564b4fba7a47bef685c47dfebf7ec2faa9ea59f7d0bcaa9d66":
                    raise RuntimeError("historical prompt mismatch")
                body = historical_body({"system_prompt": prompt, "text": unit["text"]})
                expected_body = {
                    "messages": [
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": f"<input>\n{unit['text']}\n</input>"},
                    ],
                    "model": MODEL,
                    "reasoning": {"effort": "none"},
                    "temperature": 0.6,
                    "provider": PROVIDER,
                    "max_tokens": MAX_TOKENS,
                }
                if body != expected_body:
                    raise RuntimeError("historical provider body mismatch")
                body_sha = digest_bytes(canonical(body).encode("utf-8"))
                if body_sha != unit.get("canonical_body_sha256"):
                    raise RuntimeError("provider body fingerprint mismatch")
                rank = body_counts.get(body_sha, 0)
                if unit.get("body_occurrence_rank") != rank:
                    raise RuntimeError("repeated child occurrence rank mismatch")
                body_counts[body_sha] = rank + 1
                logical_bindings += case["replicates"]
                for replicate in range(case["replicates"]):
                    identity = instance_id(case["parent_id"], replicate, body_sha, rank)
                    item = catalog.setdefault(identity, {
                        "instance_id": identity,
                        "parent_id": case["parent_id"],
                        "replicate": replicate,
                        "body_sha256": body_sha,
                        "body_occurrence_rank": rank,
                        "body": body,
                        "bindings": [],
                    })
                    if item["body"] != body:
                        raise RuntimeError("instance identity collision")
                    item["bindings"].append({"parent_id": case["parent_id"], "arm": arm, "unit_occurrence": unit["request_occurrence"]})
    items = list(catalog.values())
    if len(items) != EXPECTED_ATTEMPTS or len(items) > MAX_ATTEMPTS or len(items) * MAX_TOKENS > MAX_TOTAL_OUTPUT_TOKENS:
        raise RuntimeError("frozen request catalog exceeds finite caps or expected census")
    if logical_bindings != 291:
        raise RuntimeError(f"logical request binding census mismatch: {logical_bindings}")
    return plan, cases, items


def create_or_load_private_key(directory: Path, cases: list[dict[str, Any]], *, create: bool = True) -> dict[str, Any]:
    path = directory / "private_key.json"
    expected_entries = [(case["parent_id"], replicate) for case in cases for replicate in range(case["replicates"])]
    if not path.exists():
        if not create:
            raise RuntimeError("finalized execution requires its original private mapping")
        entries = []
        used = set()
        for parent_id, replicate in expected_entries:
            opaque = secrets.token_hex(16)
            while opaque in used:
                opaque = secrets.token_hex(16)
            used.add(opaque)
            mapping = {"X": "baseline_units", "Y": "intervention_units"} if secrets.randbits(1) else {"X": "intervention_units", "Y": "baseline_units"}
            entries.append({"opaque_case_id": opaque, "parent_id": parent_id, "replicate": replicate, "mapping": mapping})
        value = {"schema": "UNKNOWN-PARTITION-PROBE-PRIVATE-KEY-1", "private": True, "created_at": now(), "cases_sha256": digest_file(CASES_FILE), "entries": entries}
        durable_json(path, value, exclusive=True)
    value = load_json(path)
    if value.get("schema") != "UNKNOWN-PARTITION-PROBE-PRIVATE-KEY-1" or value.get("private") is not True or value.get("cases_sha256") != digest_file(CASES_FILE):
        raise RuntimeError("private mapping identity mismatch")
    entries = value.get("entries")
    if not isinstance(entries, list) or [(entry.get("parent_id"), entry.get("replicate")) for entry in entries] != expected_entries:
        raise RuntimeError("private mapping coverage mismatch")
    if len({entry.get("opaque_case_id") for entry in entries}) != len(entries):
        raise RuntimeError("private mapping opaque identities repeat")
    for entry in entries:
        if set(entry.get("mapping", {})) != {"X", "Y"} or set(entry["mapping"].values()) != {"baseline_units", "intervention_units"}:
            raise RuntimeError("private arm mapping invalid")
    return value


def create_or_load_catalog_order(directory: Path, catalog: list[dict[str, Any]]) -> list[str]:
    path = directory / "catalog_order.json"
    expected = {item["instance_id"] for item in catalog}
    if not path.exists():
        order = list(expected)
        secrets.SystemRandom().shuffle(order)
        value = {"schema": "UNKNOWN-PARTITION-PROBE-CATALOG-ORDER-1", "created_at": now(), "cases_sha256": digest_file(CASES_FILE), "instance_ids": order}
        durable_json(path, value, exclusive=True)
    value = load_json(path)
    order = value.get("instance_ids")
    if value.get("schema") != "UNKNOWN-PARTITION-PROBE-CATALOG-ORDER-1" or value.get("cases_sha256") != digest_file(CASES_FILE):
        raise RuntimeError("catalog order identity mismatch")
    if not isinstance(order, list) or len(order) != EXPECTED_ATTEMPTS or len(set(order)) != EXPECTED_ATTEMPTS or set(order) != expected:
        raise RuntimeError("catalog order is not an exact permutation")
    return order


def load_journal(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    events = []
    with path.open("r", encoding="utf-8") as source:
        for number, line in enumerate(source, 1):
            if not line.endswith("\n"):
                raise RuntimeError(f"journal line {number} is not durably terminated")
            event = json.loads(line)
            if not isinstance(event, dict):
                raise RuntimeError("journal event is not an object")
            events.append(event)
    return events


def validate_journal(events: list[dict[str, Any]], catalog: list[dict[str, Any]], endpoint: str | None = None) -> tuple[set[str], dict[str, dict[str, Any]]]:
    expected = {item["instance_id"]: item for item in catalog}
    started: set[str] = set()
    starts: dict[str, dict[str, Any]] = {}
    terminal: dict[str, dict[str, Any]] = {}
    endpoints = set()
    for event in events:
        if event.get("schema") != "UNKNOWN-PARTITION-PROBE-ATTEMPT-EVENT-1":
            raise RuntimeError("journal schema mismatch")
        identity = event.get("instance_id")
        if identity not in expected:
            raise RuntimeError("journal contains non-frozen instance")
        item = expected[identity]
        kind = event.get("event")
        if kind == "attempt_started":
            if identity in started or not isinstance(event.get("attempt_id"), str):
                raise RuntimeError("journal repeats an attempted instance")
            if event.get("request_body_sha256") != item["body_sha256"] or event.get("bindings") != item["bindings"]:
                raise RuntimeError("journal start binding mismatch")
            recorded_endpoint = event.get("endpoint")
            if not isinstance(recorded_endpoint, str):
                raise RuntimeError("journal endpoint missing")
            validate_endpoint(recorded_endpoint)
            endpoints.add(recorded_endpoint)
            started.add(identity); starts[identity] = event
        elif kind == "attempt_terminal":
            if identity not in started or identity in terminal or event.get("attempt_id") != starts[identity]["attempt_id"]:
                raise RuntimeError("journal terminal identity mismatch")
            terminal[identity] = event
        else:
            raise RuntimeError("journal event kind invalid")
    if len(started) > MAX_ATTEMPTS or len(endpoints) > 1:
        raise RuntimeError("journal violates global attempt or endpoint cap")
    if endpoint is not None and endpoints and endpoints != {endpoint}:
        raise RuntimeError("journal endpoint differs from requested endpoint")
    return started, terminal


def response_outcome(response: httpx.Response, raw: bytes, truncated: bool) -> dict[str, Any]:
    base = {
        "http_status": response.status_code,
        "raw_response_body_base64": base64.b64encode(raw).decode("ascii"),
        "raw_response_body_sha256": digest_bytes(raw),
        "raw_response_body_bytes": len(raw),
        "response_body_truncated": truncated,
    }
    if truncated:
        return {**base, "outcome": "response_body_truncated", "error": "response exceeded retained body limit", "finish_reason": None, "content": None, "actual_model": None, "actual_provider": None, "usage": None, "usage_cost": None}
    try:
        envelope = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return {**base, "outcome": "invalid_response_json", "error": f"{type(exc).__name__}: {exc}", "finish_reason": None, "content": None, "actual_model": None, "actual_provider": None, "usage": None, "usage_cost": None}
    if not isinstance(envelope, dict):
        return {**base, "outcome": "invalid_response_envelope", "error": "response JSON is not an object", "finish_reason": None, "content": None, "actual_model": None, "actual_provider": None, "usage": None, "usage_cost": None}
    usage = envelope.get("usage") if isinstance(envelope.get("usage"), dict) else None
    metadata = {
        "actual_model": envelope.get("model") if isinstance(envelope.get("model"), str) else None,
        "actual_provider": envelope.get("provider") if isinstance(envelope.get("provider"), str) else None,
        "usage": usage,
        "usage_cost": usage.get("cost") if usage is not None else None,
    }
    if response.status_code != 200:
        return {**base, **metadata, "outcome": "http_error", "error": f"HTTP {response.status_code}", "finish_reason": None, "content": None}
    choices = envelope.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return {**base, **metadata, "outcome": "missing_choices", "error": "missing first choice", "finish_reason": None, "content": None}
    choice = choices[0]
    finish_reason = choice.get("finish_reason") if isinstance(choice.get("finish_reason"), str) else None
    message = choice.get("message")
    content = message.get("content") if isinstance(message, dict) else None
    completion_tokens = usage.get("completion_tokens") if usage is not None else None
    if finish_reason == "length":
        return {**base, **metadata, "outcome": "finish_reason_length", "error": "provider reported length truncation", "finish_reason": finish_reason, "content": content if isinstance(content, str) else None}
    if isinstance(completion_tokens, int) and completion_tokens > MAX_TOKENS:
        return {**base, **metadata, "outcome": "usage_output_cap_violation", "error": "provider usage exceeds frozen per-attempt output cap", "finish_reason": finish_reason, "content": content if isinstance(content, str) else None}
    if not isinstance(content, str) or not content.strip():
        return {**base, **metadata, "outcome": "empty_content", "error": "first choice has no nonempty plain text", "finish_reason": finish_reason, "content": content if isinstance(content, str) else None}
    return {**base, **metadata, "outcome": "success", "error": None, "finish_reason": finish_reason, "content": content}


def dispatch(client: httpx.Client, endpoint: str, api_key: str, item: dict[str, Any]) -> dict[str, Any]:
    raw = bytearray()
    truncated = False
    with client.stream("POST", endpoint, headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}, content=canonical(item["body"]).encode("utf-8")) as response:
        for chunk in response.iter_bytes():
            remaining = RAW_RESPONSE_LIMIT - len(raw)
            if remaining > 0:
                raw.extend(chunk[:remaining])
            if len(chunk) > remaining:
                truncated = True
                break
        return response_outcome(response, bytes(raw), truncated)


def build_blind_packet(cases: list[dict[str, Any]], private: dict[str, Any], terminal: dict[str, dict[str, Any]]) -> dict[str, Any]:
    case_by_parent = {case["parent_id"]: case for case in cases}
    packet_cases = []
    for key in private["entries"]:
        case = case_by_parent[key["parent_id"]]
        candidates = {}
        for label in ("X", "Y"):
            arm = key["mapping"][label]
            outputs = []
            unavailable = []
            for unit in case[arm]:
                identity = instance_id(case["parent_id"], key["replicate"], unit["canonical_body_sha256"], unit["body_occurrence_rank"])
                event = terminal.get(identity)
                if event is None:
                    unavailable.append("attempted_indeterminate_or_not_started")
                elif event.get("outcome") != "success":
                    unavailable.append(str(event.get("outcome")))
                else:
                    outputs.append(event["content"])
            if unavailable:
                candidates[label] = {"status": "unavailable", "reasons": unavailable, "ordered_korean_text": None}
            else:
                candidates[label] = {"status": "available", "ordered_korean_text": outputs}
        packet_cases.append({
            "opaque_case_id": key["opaque_case_id"],
            "accepted_source_parent": case["accepted_text"],
            "frozen_source_facts": case["source_facts_frozen_before_generation"],
            "candidates": candidates,
        })
    return {
        "schema": "UNKNOWN-PARTITION-PROBE-BLIND-PACKET-1",
        "partial_blinding": "Candidate split shape remains visible. Arm names, hypothesis, grouping policy, group, parent identity, and provenance are withheld.",
        "rating_contract": {
            "meaning": ["X", "Y", "equal", "unjudgeable"],
            "judge": "Observable meaning only: numbers, questions, negation, agreement, and claim attribution; split count, purity, and readability alone are not meaning.",
            "readability": "optional and separate",
            "new_severe_error_kinds": sorted(SEVERE_KINDS),
            "new_severe_errors_relative_to_other": "Each entry identifies an error introduced in that candidate relative to the other candidate, not every absolute error category present.",
            "evidence": "Every rating supplies a nonempty case-specific explanation.",
            "source_uncertainty": "Every rating supplies a nonempty rater judgment about semantic ambiguity in the English source; this is not native UNKNOWN metadata.",
            "unavailable": "If either candidate is unavailable, both preferences are unjudgeable, clear_win is false, and both new-severe lists are empty.",
            "identical": "Byte-identical ordered candidates require both preferences equal, clear_win false, and both new-severe lists empty.",
        },
        "ratings_template": {
            "schema": "UNKNOWN-PARTITION-PROBE-RATINGS-1",
            "rater": "nonempty",
            "ratings": [{
                "opaque_case_id": "from packet",
                "meaning_preference": "X|Y|equal|unjudgeable",
                "clear_win": False,
                "readability_preference": "X|Y|equal|unjudgeable",
                "evidence": "nonempty case-specific semantic evidence",
                "source_uncertainty": "nonempty semantic-English ambiguity judgment or explicit none observed",
                "new_severe_errors_relative_to_other": {"X": [], "Y": []},
            }],
        },
        "cases": packet_cases,
    }


def write_packet(directory: Path, packet: dict[str, Any]) -> Path:
    path = directory / "blind_packet.json"
    if path.exists() and load_json(path) != packet:
        raise RuntimeError("blind packet exists with different content")
    if not path.exists():
        durable_json(path, packet, exclusive=True)
    return path

def validate_finalized_evidence(directory: Path) -> dict[str, Any] | None:
    path = directory / "execution_provenance.json"
    if not path.exists():
        return None
    provenance = load_json(path)
    if provenance.get("schema") != "UNKNOWN-PARTITION-PROBE-EXECUTION-PROVENANCE-1":
        raise RuntimeError("finalized execution provenance schema mismatch")
    journal = directory / "attempts.jsonl"
    if not journal.exists() or digest_file(journal) != provenance.get("journal_sha256"):
        raise RuntimeError("finalized journal differs from its committed evidence")
    packet = directory / "blind_packet.json"
    if not packet.exists() or digest_file(packet) != provenance.get("blind_packet_sha256"):
        raise RuntimeError("finalized blind packet differs from its committed evidence")
    return provenance


def execute(directory: Path, endpoint: str, timeout: float) -> dict[str, Any]:
    endpoint = validate_endpoint(endpoint)
    directory, mode = execution_mode(directory, endpoint)
    if timeout <= 0 or timeout > 120:
        raise ValueError("timeout must be greater than zero and at most 120 seconds")
    plan, cases, catalog = validate_frozen()
    by_id = {item["instance_id"]: item for item in catalog}
    with execution_lock(directory):
        validate_finalized_evidence(directory)
        private = create_or_load_private_key(directory, cases)
        order = create_or_load_catalog_order(directory, catalog)
        journal_path = directory / "attempts.jsonl"
        events = load_journal(journal_path)
        started, terminal = validate_journal(events, catalog, endpoint)
        api_key = "rehearsal-dummy-key" if mode == "loopback_rehearsal" else credential()
        if api_key is None:
            raise RuntimeError("OpenRouter credential unavailable")
        client_timeout = httpx.Timeout(timeout, connect=min(timeout, 15.0))
        with httpx.Client(timeout=client_timeout, follow_redirects=False) as client:
            for sequence, identity in enumerate(order, 1):
                if identity in started:
                    continue
                if len(started) >= MAX_ATTEMPTS:
                    raise RuntimeError("global HTTP attempt cap reached")
                item = by_id[identity]
                start = {
                    "schema": "UNKNOWN-PARTITION-PROBE-ATTEMPT-EVENT-1", "event": "attempt_started",
                    "attempt_id": secrets.token_hex(16), "instance_id": identity, "sequence": sequence,
                    "request_body_sha256": item["body_sha256"], "bindings": item["bindings"],
                    "endpoint": endpoint, "started_at": now(),
                }
                append_event(journal_path, start)
                started.add(identity)
                try:
                    outcome = dispatch(client, endpoint, api_key, item)
                except KeyboardInterrupt:
                    raise
                except Exception as exc:
                    outcome = {
                        "outcome": "transport_error", "error": f"{type(exc).__name__}: {exc}", "http_status": None,
                        "raw_response_body_base64": None, "raw_response_body_sha256": None, "raw_response_body_bytes": 0,
                        "response_body_truncated": False, "finish_reason": None, "content": None,
                        "actual_model": None, "actual_provider": None, "usage": None, "usage_cost": None,
                    }
                terminal_event = {
                    "schema": "UNKNOWN-PARTITION-PROBE-ATTEMPT-EVENT-1", "event": "attempt_terminal",
                    "attempt_id": start["attempt_id"], "instance_id": identity, "finished_at": now(), **outcome,
                }
                append_event(journal_path, terminal_event)
                terminal[identity] = terminal_event
        events = load_journal(journal_path)
        started, terminal = validate_journal(events, catalog, endpoint)
        packet_path = write_packet(directory, build_blind_packet(cases, private, terminal))
        provenance = {
            "schema": "UNKNOWN-PARTITION-PROBE-EXECUTION-PROVENANCE-1", "mode": mode,
            "endpoint": endpoint, "cases_sha256": plan["cases_sha256"],
            "journal_sha256": digest_file(journal_path), "blind_packet_sha256": digest_file(packet_path),
            "attempt_started": len(started), "attempt_terminal": len(terminal), "rerun_duplicate_attempts": 0,
            "indeterminate": len(started - set(terminal)),
            "successful": sum(event.get("outcome") == "success" for event in terminal.values()),
            "failed": sum(event.get("outcome") != "success" for event in terminal.values()),
            "reported_completion_tokens": sum(int(event.get("usage", {}).get("completion_tokens", 0)) for event in terminal.values() if isinstance(event.get("usage"), dict) and isinstance(event["usage"].get("completion_tokens", 0), int)),
            "reported_cost": sum(float(event.get("usage_cost") or 0) for event in terminal.values()),
        }
        durable_json(directory / "execution_provenance.json", provenance)
        return {**provenance, "blind_packet": str(packet_path), "private_key": str(directory / "private_key.json")}


def validate_ratings(ratings: Any, packet: dict[str, Any], *, locked: bool) -> list[dict[str, Any]]:
    top_keys = {"schema", "rater", "ratings", *(("blind_packet_sha256",) if locked else ())}
    if not isinstance(ratings, dict) or set(ratings) != top_keys:
        raise RuntimeError("ratings schema invalid")
    rows = ratings.get("ratings")
    if ratings.get("schema") != "UNKNOWN-PARTITION-PROBE-RATINGS-1" or not isinstance(ratings.get("rater"), str) or not ratings["rater"].strip() or not isinstance(rows, list):
        raise RuntimeError("ratings schema invalid")
    if locked and ratings.get("blind_packet_sha256") != digest_bytes((json.dumps(packet, ensure_ascii=False, indent=2) + "\n").encode("utf-8")):
        raise RuntimeError("locked ratings blind packet binding mismatch")
    packet_cases = packet.get("cases")
    if not isinstance(packet_cases, list) or [row.get("opaque_case_id") for row in rows if isinstance(row, dict)] != [case.get("opaque_case_id") for case in packet_cases if isinstance(case, dict)]:
        raise RuntimeError("ratings coverage or order mismatch")
    expected_row_keys = {
        "opaque_case_id", "meaning_preference", "clear_win", "readability_preference",
        "evidence", "source_uncertainty", "new_severe_errors_relative_to_other",
    }
    for row, case in zip(rows, packet_cases):
        if not isinstance(row, dict) or set(row) != expected_row_keys:
            raise RuntimeError("rating row schema invalid")
        meaning = row.get("meaning_preference")
        readability = row.get("readability_preference")
        clear = row.get("clear_win")
        if meaning not in {"X", "Y", "equal", "unjudgeable"} or readability not in {"X", "Y", "equal", "unjudgeable"} or not isinstance(clear, bool):
            raise RuntimeError("rating value invalid")
        if not isinstance(row.get("evidence"), str) or not row["evidence"].strip() or not isinstance(row.get("source_uncertainty"), str) or not row["source_uncertainty"].strip():
            raise RuntimeError("rating evidence and source uncertainty must be nonempty strings")
        severe = row.get("new_severe_errors_relative_to_other")
        if not isinstance(severe, dict) or set(severe) != {"X", "Y"}:
            raise RuntimeError("new severe error schema invalid")
        for label in ("X", "Y"):
            entries = severe[label]
            if not isinstance(entries, list):
                raise RuntimeError("new severe error list invalid")
            for entry in entries:
                if not isinstance(entry, dict) or set(entry) != {"kind", "detail"} or entry.get("kind") not in SEVERE_KINDS or not isinstance(entry.get("detail"), str) or not entry["detail"].strip():
                    raise RuntimeError("new severe error entry invalid")
        candidates = case.get("candidates")
        if not isinstance(candidates, dict) or set(candidates) != {"X", "Y"}:
            raise RuntimeError("blind packet candidate schema invalid")
        available = all(isinstance(candidates[label], dict) and candidates[label].get("status") == "available" for label in ("X", "Y"))
        identical = available and canonical(candidates["X"].get("ordered_korean_text")).encode("utf-8") == canonical(candidates["Y"].get("ordered_korean_text")).encode("utf-8")
        if not available:
            if meaning != "unjudgeable" or readability != "unjudgeable" or clear or severe["X"] or severe["Y"]:
                raise RuntimeError("unavailable candidate rating must be unjudgeable with no clear win or new severe errors")
        elif identical:
            if meaning != "equal" or readability != "equal" or clear or severe["X"] or severe["Y"]:
                raise RuntimeError("byte-identical ordered candidates must be equal with no clear win or new severe errors")
        if clear:
            if meaning not in {"X", "Y"}:
                raise RuntimeError("clear win requires an X or Y meaning preference")
            if severe[meaning]:
                raise RuntimeError("clear winner cannot carry its own new severe error")
    return rows


def lock_ratings(directory: Path, ratings_path: Path) -> dict[str, Any]:
    directory = directory.resolve()
    provenance = validate_finalized_evidence(directory)
    if provenance is None:
        raise RuntimeError("ratings require finalized execution evidence")
    packet_path = directory / "blind_packet.json"
    packet = load_json(packet_path)
    ratings = load_json(ratings_path)
    rows = validate_ratings(ratings, packet, locked=False)
    locked_value = {**ratings, "blind_packet_sha256": digest_file(packet_path)}
    encoded = (json.dumps(locked_value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    locked_path = directory / "ratings.locked.json"
    durable_write(locked_path, encoded, exclusive=True)
    return {"locked": str(locked_path), "sha256": digest_bytes(encoded), "blind_packet_sha256": provenance["blind_packet_sha256"], "ratings": len(rows)}


def decode(directory: Path) -> dict[str, Any]:
    directory = directory.resolve()
    provenance = validate_finalized_evidence(directory)
    if provenance is None:
        raise RuntimeError("decode requires finalized execution evidence")
    _plan, cases, _catalog = validate_frozen()
    private = create_or_load_private_key(directory, cases, create=False)
    packet = load_json(directory / "blind_packet.json")
    ratings_path = directory / "ratings.locked.json"
    ratings = load_json(ratings_path)
    rows = validate_ratings(ratings, packet, locked=True)
    mapping = {entry["opaque_case_id"]: entry for entry in private["entries"]}
    decoded = []
    for row in rows:
        key = mapping[row["opaque_case_id"]]
        arm_by_label = {label: arm for label, arm in key["mapping"].items()}
        meaning = row["meaning_preference"]
        readability = row["readability_preference"]
        decoded.append({
            "parent_id": key["parent_id"],
            "replicate": key["replicate"],
            "meaning_preference": arm_by_label[meaning].replace("_units", "") if meaning in {"X", "Y"} else meaning,
            "clear_win": row["clear_win"],
            "readability_preference": arm_by_label[readability].replace("_units", "") if readability in {"X", "Y"} else readability,
            "evidence": row["evidence"],
            "source_uncertainty": row["source_uncertainty"],
            "new_severe_errors_relative_to_other": {arm_by_label[label].replace("_units", ""): row["new_severe_errors_relative_to_other"][label] for label in ("X", "Y")},
        })
    result = {
        "schema": "UNKNOWN-PARTITION-PROBE-DECODED-RATINGS-1",
        "ratings_locked_sha256": digest_file(ratings_path),
        "blind_packet_sha256": provenance["blind_packet_sha256"],
        "denominator_note": "Primary denominator is 18 distinct changed DEV parents in three meeting-family clusters out of 2367 nonempty DEV parents; repeats and two no-op guards are separate.",
        "decoded": decoded,
    }
    path = directory / "decoded_ratings.json"
    if path.exists() and load_json(path) != result:
        raise RuntimeError("decoded ratings already differ")
    if not path.exists():
        durable_json(path, result, exclusive=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("check")
    execute_parser = subparsers.add_parser("execute")
    execute_parser.add_argument("--execution-dir", type=Path, default=DEFAULT_EXECUTION_DIR)
    execute_parser.add_argument("--endpoint", default=OFFICIAL_ENDPOINT)
    execute_parser.add_argument("--timeout", type=float, default=45.0)
    lock_parser = subparsers.add_parser("lock-ratings")
    lock_parser.add_argument("ratings", type=Path)
    lock_parser.add_argument("--execution-dir", type=Path, default=DEFAULT_EXECUTION_DIR)
    decode_parser = subparsers.add_parser("decode")
    decode_parser.add_argument("--execution-dir", type=Path, default=DEFAULT_EXECUTION_DIR)
    args = parser.parse_args()
    if args.command == "check":
        plan, cases, catalog = validate_frozen()
        result = {"ok": True, "cases": len(cases), "changed_parents": sum(case["removed_boundary_count"] > 0 for case in cases), "catalog_attempts": len(catalog), "maximum_output_tokens": len(catalog) * MAX_TOKENS, "cases_sha256": plan["cases_sha256"], "credential_available": credential() is not None}
    elif args.command == "execute":
        result = execute(args.execution_dir, args.endpoint, args.timeout)
    elif args.command == "lock-ratings":
        result = lock_ratings(args.execution_dir, args.ratings)
    else:
        result = decode(args.execution_dir)
    print(canonical(result))


if __name__ == "__main__":
    main()
