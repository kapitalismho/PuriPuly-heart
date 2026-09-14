from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.util
import json
import re
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

HERE = Path(__file__).resolve().parent
CASES_FILE = HERE / "cases.jsonl"
PLAN_FILE = HERE / "plan.json"
DEFAULT_EXECUTION_DIR = HERE / "execution"
OFFICIAL_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
MAX_ATTEMPTS = 54
EXPECTED_ATTEMPTS = 54
MAX_TOKENS = 512
MAX_TOTAL_OUTPUT_TOKENS = 27648
RAW_RESPONSE_LIMIT = 131072
SEVERE_KINDS = {
    "source_omission", "unsupported_addition", "meaning_inversion", "number_error",
    "question_or_agreement_error", "claim_attribution_error", "cross_segment_mapping", "content_duplication",
}

PREPARE_PATH = HERE / "prepare.py"
HELPER_PATH = HERE.parent / "meaning-first-probe" / "run_e1.py"


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PREPARE = load_module("batch_unknown_prepare", PREPARE_PATH)
HELPERS = load_module("batch_unknown_durable_helpers", HELPER_PATH)
durable_write = HELPERS.durable_write
durable_json = HELPERS.durable_json
append_event = HELPERS.append_event
execution_lock = HELPERS.execution_lock
credential = HELPERS.credential
validate_endpoint = HELPERS.validate_endpoint


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicate_pairs)


def reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_cases() -> list[dict[str, Any]]:
    return [json.loads(line, object_pairs_hook=reject_duplicate_pairs) for line in CASES_FILE.read_text(encoding="utf-8").splitlines() if line]


def body_for(case: dict[str, Any], arm: str) -> dict[str, Any]:
    return PREPARE.request_body(case, arm)


def instance_id(parent_id: str, replicate: int, body_sha256: str) -> str:
    return digest_bytes(canonical([parent_id, replicate, body_sha256]).encode("utf-8"))


def catalog_for(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    catalog: dict[str, dict[str, Any]] = {}
    for case in cases:
        for replicate in range(case["replicates"]):
            for arm in ("A", "B"):
                body = body_for(case, arm)
                body_sha = digest_bytes(canonical(body).encode("utf-8"))
                identity = instance_id(case["parent_id"], replicate, body_sha)
                binding = {"parent_id": case["parent_id"], "replicate": replicate, "arm": arm}
                if identity in catalog:
                    if catalog[identity]["body"] != body or catalog[identity]["parent_id"] != case["parent_id"] or catalog[identity]["replicate"] != replicate:
                        raise RuntimeError("request identity collision")
                    catalog[identity]["bindings"].append(binding)
                else:
                    catalog[identity] = {"instance_id": identity, "parent_id": case["parent_id"], "replicate": replicate, "body_sha256": body_sha, "body": body, "expected_ids": [f"s{i}" for i in range(len(case["source_partitions"][arm]))], "bindings": [binding]}
    return list(catalog.values())


def validate_frozen() -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    plan = load_json(PLAN_FILE)
    cases = load_cases()
    PREPARE.validate(cases, plan)
    if plan["inputs"]["meaning-first-probe/run_e1.py"] != digest_file(HELPER_PATH):
        raise RuntimeError("durable helper identity mismatch")
    if plan["code_sha256"] != {"prepare.py": digest_file(PREPARE_PATH), "run.py": digest_file(Path(__file__))}:
        raise RuntimeError("runner identity mismatch")
    catalog = catalog_for(cases)
    if len(catalog) != EXPECTED_ATTEMPTS or sum(len(item["bindings"]) for item in catalog) != 56:
        raise RuntimeError("frozen catalog cardinality mismatch")
    if sum(MAX_TOKENS for _ in catalog) != MAX_TOTAL_OUTPUT_TOKENS or any(item["body"].get("max_tokens") != MAX_TOKENS for item in catalog):
        raise RuntimeError("output cap mismatch")
    return plan, cases, catalog


def execution_mode(directory: Path, endpoint: str) -> tuple[Path, str]:
    resolved = directory.expanduser().resolve()
    canonical_dir = DEFAULT_EXECUTION_DIR.resolve()
    if endpoint == OFFICIAL_ENDPOINT:
        if resolved != canonical_dir:
            raise RuntimeError("official endpoint requires the canonical execution directory")
        return resolved, "official_provider"
    if resolved == canonical_dir:
        raise RuntimeError("loopback rehearsal cannot use the canonical execution directory")
    return resolved, "loopback_rehearsal"


def create_or_load_private_key(directory: Path, cases: list[dict[str, Any]], *, create: bool = True) -> dict[str, Any]:
    path = directory / "private_key.json"
    expected = [(case["parent_id"], replicate) for case in cases for replicate in range(case["replicates"])]
    if not path.exists():
        if not create:
            raise RuntimeError("finalized execution requires its original private mapping")
        entries = []
        used: set[str] = set()
        for parent_id, replicate in expected:
            opaque = secrets.token_hex(16)
            while opaque in used:
                opaque = secrets.token_hex(16)
            used.add(opaque)
            mapping = {"X": "A", "Y": "B"} if secrets.randbits(1) else {"X": "B", "Y": "A"}
            entries.append({"opaque_case_id": opaque, "parent_id": parent_id, "replicate": replicate, "mapping": mapping})
        durable_json(path, {"schema": "BATCH-UNKNOWN-PROBE-PRIVATE-1", "private": True, "created_at": now(), "cases_sha256": digest_file(CASES_FILE), "entries": entries}, exclusive=True)
    value = load_json(path)
    if set(value) != {"schema", "private", "created_at", "cases_sha256", "entries"} or value["schema"] != "BATCH-UNKNOWN-PROBE-PRIVATE-1" or value["private"] is not True or value["cases_sha256"] != digest_file(CASES_FILE):
        raise RuntimeError("private mapping identity mismatch")
    entries = value["entries"]
    if [(entry.get("parent_id"), entry.get("replicate")) for entry in entries if isinstance(entry, dict)] != expected or len({entry.get("opaque_case_id") for entry in entries}) != 28:
        raise RuntimeError("private mapping coverage mismatch")
    for entry in entries:
        if set(entry) != {"opaque_case_id", "parent_id", "replicate", "mapping"} or set(entry["mapping"]) != {"X", "Y"} or set(entry["mapping"].values()) != {"A", "B"}:
            raise RuntimeError("private arm mapping invalid")
    return value


def create_or_load_catalog_order(directory: Path, catalog: list[dict[str, Any]]) -> list[str]:
    path = directory / "catalog_order.json"
    expected = {item["instance_id"] for item in catalog}
    if not path.exists():
        order = list(expected)
        secrets.SystemRandom().shuffle(order)
        durable_json(path, {"schema": "BATCH-UNKNOWN-PROBE-CATALOG-ORDER-1", "created_at": now(), "cases_sha256": digest_file(CASES_FILE), "instance_ids": order}, exclusive=True)
    value = load_json(path)
    order = value.get("instance_ids")
    if value.get("schema") != "BATCH-UNKNOWN-PROBE-CATALOG-ORDER-1" or value.get("cases_sha256") != digest_file(CASES_FILE) or not isinstance(order, list) or len(order) != EXPECTED_ATTEMPTS or len(set(order)) != EXPECTED_ATTEMPTS or set(order) != expected:
        raise RuntimeError("catalog order is not an exact frozen permutation")
    return order


def execution_identity(plan: dict[str, Any], catalog: list[dict[str, Any]]) -> dict[str, Any]:
    return {"schema": "BATCH-UNKNOWN-PROBE-EXECUTION-IDENTITY-1", "plan_sha256": digest_file(PLAN_FILE), "cases_sha256": plan["cases_sha256"], "code_sha256": plan["code_sha256"], "source_pins": plan["inputs"], "prompt_sha256": plan["request"]["system_prompt_sha256"], "catalog_sha256": digest_bytes(canonical([{key: item[key] for key in ("instance_id", "body_sha256", "bindings")} for item in catalog]).encode("utf-8")), "attempt_cap": MAX_ATTEMPTS, "output_token_cap": MAX_TOTAL_OUTPUT_TOKENS}


def write_or_validate_identity(directory: Path, identity: dict[str, Any]) -> None:
    path = directory / "execution_identity.json"
    if path.exists():
        if load_json(path) != identity:
            raise RuntimeError("execution identity differs from frozen catalog/code/source")
    else:
        durable_json(path, identity, exclusive=True)


def load_journal(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    events = []
    with path.open("r", encoding="utf-8") as source:
        for number, line in enumerate(source, 1):
            if not line.endswith("\n"):
                raise RuntimeError(f"journal line {number} is not durably terminated")
            events.append(json.loads(line, object_pairs_hook=reject_duplicate_pairs))
    return events


def validate_journal(events: list[dict[str, Any]], catalog: list[dict[str, Any]], endpoint: str | None = None) -> tuple[set[str], dict[str, dict[str, Any]]]:
    expected = {item["instance_id"]: item for item in catalog}
    started: set[str] = set(); starts: dict[str, dict[str, Any]] = {}; terminal: dict[str, dict[str, Any]] = {}; endpoints: set[str] = set()
    for event in events:
        if event.get("schema") != "BATCH-UNKNOWN-PROBE-ATTEMPT-EVENT-1" or event.get("instance_id") not in expected:
            raise RuntimeError("journal schema or instance mismatch")
        identity = event["instance_id"]; item = expected[identity]
        if event.get("event") == "attempt_started":
            if identity in started or not isinstance(event.get("attempt_id"), str) or event.get("request_body_sha256") != item["body_sha256"] or event.get("bindings") != item["bindings"]:
                raise RuntimeError("journal repeats or misbinds an attempt")
            validate_endpoint(event.get("endpoint")); endpoints.add(event["endpoint"]); started.add(identity); starts[identity] = event
        elif event.get("event") == "attempt_terminal":
            if identity not in started or identity in terminal or event.get("attempt_id") != starts[identity]["attempt_id"]:
                raise RuntimeError("journal terminal identity mismatch")
            terminal[identity] = event
        else:
            raise RuntimeError("journal event kind invalid")
    if len(started) > MAX_ATTEMPTS or len(endpoints) > 1 or (endpoint is not None and endpoints and endpoints != {endpoint}):
        raise RuntimeError("journal violates endpoint or attempt cap")
    return started, terminal


def parse_translation_content(content: str, expected_ids: list[str]) -> dict[str, Any]:
    framing = "plain_json"
    operations: list[str] = []
    candidate = content
    match = re.fullmatch(r"```(?:json)?\r?\n([\s\S]*?)\r?\n```", content)
    if match:
        candidate = match.group(1)
        framing = "entire_outer_fence"
        operations = ["unwrapped_exactly_one_entire_outer_json_or_plain_fence"]
    elif "```" in content:
        raise ValueError("content contains a non-entire or multiple Markdown fence")
    parsed = json.loads(candidate, object_pairs_hook=reject_duplicate_pairs)
    if not isinstance(parsed, dict) or set(parsed) != {"translations"} or not isinstance(parsed["translations"], list):
        raise ValueError("output root must contain exactly translations array")
    raw_order: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in parsed["translations"]:
        if not isinstance(item, dict) or set(item) != {"id", "text"} or not isinstance(item["id"], str) or not isinstance(item["text"], str) or not item["text"].strip():
            raise ValueError("translation item schema or text invalid")
        if item["id"] in seen:
            raise ValueError("duplicate translation ID")
        seen.add(item["id"]); raw_order.append({"id": item["id"], "text": item["text"]})
    if seen != set(expected_ids) or len(raw_order) != len(expected_ids):
        raise ValueError("missing or extra translation IDs")
    by_id = {item["id"]: item for item in raw_order}
    normalized = [by_id[identity] for identity in expected_ids]
    if [item["id"] for item in raw_order] != expected_ids:
        operations.append("reordered_complete_unique_id_array_to_source_order")
    return {"framing_status": framing, "normalization_operations": operations, "raw_translation_order": raw_order, "ordered_translations": normalized}


def response_outcome(response: httpx.Response, raw: bytes, truncated: bool, expected_ids: list[str]) -> dict[str, Any]:
    base = {"http_status": response.status_code, "raw_response_body_base64": base64.b64encode(raw).decode("ascii"), "raw_response_body_sha256": digest_bytes(raw), "raw_response_body_bytes": len(raw), "response_body_truncated": truncated}
    failure = {"finish_reason": None, "content": None, "actual_model": None, "actual_provider": None, "usage": None, "usage_cost": None, "framing_status": "not_examined", "normalization_operations": [], "raw_translation_order": None, "ordered_translations": None}
    if truncated:
        return {**base, **failure, "outcome": "response_body_truncated", "error": "response exceeded retained body limit"}
    try:
        envelope = json.loads(raw, object_pairs_hook=reject_duplicate_pairs)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        return {**base, **failure, "outcome": "invalid_response_json", "error": f"{type(exc).__name__}: {exc}"}
    if not isinstance(envelope, dict):
        return {**base, **failure, "outcome": "invalid_response_envelope", "error": "response JSON is not an object"}
    usage = envelope.get("usage") if isinstance(envelope.get("usage"), dict) else None
    metadata = {"actual_model": envelope.get("model") if isinstance(envelope.get("model"), str) else None, "actual_provider": envelope.get("provider") if isinstance(envelope.get("provider"), str) else None, "usage": usage, "usage_cost": usage.get("cost") if usage else None}
    if response.status_code != 200:
        return {**base, **failure, **metadata, "outcome": "http_error", "error": f"HTTP {response.status_code}"}
    choices = envelope.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return {**base, **failure, **metadata, "outcome": "missing_choices", "error": "missing first choice"}
    choice = choices[0]; finish_reason = choice.get("finish_reason") if isinstance(choice.get("finish_reason"), str) else None
    message = choice.get("message"); content = message.get("content") if isinstance(message, dict) else None
    completion_tokens = usage.get("completion_tokens") if usage else None
    common = {**base, **failure, **metadata, "finish_reason": finish_reason, "content": content if isinstance(content, str) else None}
    if finish_reason == "length":
        return {**common, "outcome": "finish_reason_length", "error": "provider reported length truncation"}
    if finish_reason != "stop":
        return {**common, "outcome": "unexpected_finish_reason", "error": "provider did not report finish_reason stop"}
    if isinstance(completion_tokens, int) and completion_tokens > MAX_TOKENS:
        return {**common, "outcome": "usage_output_cap_violation", "error": "provider usage exceeds frozen per-attempt cap"}
    if not isinstance(content, str) or not content.strip():
        return {**common, "outcome": "empty_content", "error": "first choice has no nonempty content"}
    try:
        validated = parse_translation_content(content, expected_ids)
    except (json.JSONDecodeError, ValueError) as exc:
        return {**common, "outcome": "invalid_translation_output", "error": f"{type(exc).__name__}: {exc}"}
    return {**common, **validated, "outcome": "success", "error": None}


def dispatch(client: httpx.Client, endpoint: str, api_key: str, item: dict[str, Any]) -> dict[str, Any]:
    raw = bytearray(); truncated = False
    with client.stream("POST", endpoint, headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}, content=canonical(item["body"]).encode("utf-8")) as response:
        for chunk in response.iter_bytes():
            remaining = RAW_RESPONSE_LIMIT - len(raw)
            if remaining > 0:
                raw.extend(chunk[:remaining])
            if len(chunk) > remaining:
                truncated = True; break
        return response_outcome(response, bytes(raw), truncated, item["expected_ids"])


def terminal_for(case: dict[str, Any], replicate: int, arm: str, terminal: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    body_sha = digest_bytes(canonical(body_for(case, arm)).encode("utf-8"))
    return terminal.get(instance_id(case["parent_id"], replicate, body_sha))


def build_blind_packet(cases: list[dict[str, Any]], private: dict[str, Any], terminal: dict[str, dict[str, Any]]) -> dict[str, Any]:
    by_parent = {case["parent_id"]: case for case in cases}; packet_cases = []
    for key in private["entries"]:
        case = by_parent[key["parent_id"]]; candidates = {}
        for label in ("X", "Y"):
            arm = key["mapping"][label]; event = terminal_for(case, key["replicate"], arm, terminal)
            source_segments = [{"id": f"s{i}", "text": unit["text"]} for i, unit in enumerate(case["source_partitions"][arm])]
            if event is None:
                candidates[label] = {"status": "unavailable", "reason": "attempted_indeterminate_or_not_started", "source_segments": source_segments, "ordered_translations": None}
            elif event.get("outcome") != "success":
                candidates[label] = {"status": "unavailable", "reason": event.get("outcome"), "source_segments": source_segments, "ordered_translations": None}
            else:
                candidates[label] = {"status": "available", "reason": None, "source_segments": source_segments, "ordered_translations": event["ordered_translations"]}
        packet_cases.append({"opaque_case_id": key["opaque_case_id"], "source_parent": case["accepted_text"], "frozen_source_facts": case["source_facts_frozen_before_generation"], "candidates": candidates})
    return {"schema": "BATCH-TRANSLATION-BLIND-PACKET-1", "partial_blinding": "Candidate shape and source correspondence are visible. Parent IDs, groups, policy labels, segmentation metadata, hypothesis, and arm mapping are withheld.", "rating_contract": {"meaning": ["X", "Y", "equal", "unjudgeable"], "judge": "Judge meaning independently of readability; preserve numbers, questions, negation, uncertainty, and attribution. Note cross-segment mapping or duplicated content when observable.", "new_severe_error_kinds": sorted(SEVERE_KINDS), "relative_errors": "Each structured item is a specific error newly present in that candidate relative to the other, not a coarse list of all errors.", "source_uncertainty": "Required and about ambiguity in the English source, not hidden segmentation metadata.", "unavailable": "If either candidate is unavailable, meaning and readability are unjudgeable, clear_win is false, and both relative severe lists are empty.", "identical": "Identical available candidate source/text pairs require equal/equal, no clear win, and empty relative severe lists.", "clear_winner": "A clear winner cannot carry its own new severe error."}, "ratings_template": {"schema": "BATCH-TRANSLATION-RATINGS-1", "rater": "nonempty", "ratings": [{"opaque_case_id": "packet order", "meaning_preference": "X|Y|equal|unjudgeable", "clear_win": False, "readability_preference": "X|Y|equal|unjudgeable", "evidence": "case-specific", "source_uncertainty": "nonempty", "new_severe_errors_relative_to_other": {"X": [], "Y": []}}]}, "cases": packet_cases}


def write_packet(directory: Path, packet: dict[str, Any]) -> Path:
    path = directory / "blind_packet.json"
    if path.exists() and load_json(path) != packet:
        raise RuntimeError("blind packet exists with different content")
    if not path.exists(): durable_json(path, packet, exclusive=True)
    return path


def validate_finalized(directory: Path) -> dict[str, Any] | None:
    path = directory / "execution_provenance.json"
    if not path.exists(): return None
    value = load_json(path)
    if value.get("schema") != "BATCH-UNKNOWN-PROBE-EXECUTION-PROVENANCE-1": raise RuntimeError("finalized provenance schema mismatch")
    for name, key in (("attempts.jsonl", "journal_sha256"), ("blind_packet.json", "blind_packet_sha256"), ("execution_identity.json", "execution_identity_sha256"), ("private_key.json", "private_key_sha256"), ("catalog_order.json", "catalog_order_sha256")):
        target = directory / name
        if not target.exists() or digest_file(target) != value.get(key): raise RuntimeError(f"finalized {name} differs from committed hash")
    plan, _cases, catalog = validate_frozen()
    if load_json(directory / "execution_identity.json") != execution_identity(plan, catalog):
        raise RuntimeError("saved execution identity does not match the current frozen plan")
    return value


def execute(directory: Path, endpoint: str, timeout: float) -> dict[str, Any]:
    endpoint = validate_endpoint(endpoint); directory, mode = execution_mode(directory, endpoint)
    if timeout <= 0 or timeout > 120: raise ValueError("timeout must be greater than zero and at most 120 seconds")
    plan, cases, catalog = validate_frozen(); by_id = {item["instance_id"]: item for item in catalog}
    with execution_lock(directory):
        finalized = validate_finalized(directory)
        private = create_or_load_private_key(directory, cases)
        order = create_or_load_catalog_order(directory, catalog)
        identity = execution_identity(plan, catalog); write_or_validate_identity(directory, identity)
        journal_path = directory / "attempts.jsonl"; events = load_journal(journal_path); started, terminal = validate_journal(events, catalog, endpoint)
        if finalized is not None:
            return finalized
        api_key = "rehearsal-dummy-key" if mode == "loopback_rehearsal" else credential()
        if api_key is None: raise RuntimeError("OpenRouter credential unavailable")
        with httpx.Client(timeout=httpx.Timeout(timeout, connect=min(timeout, 15.0)), follow_redirects=False) as client:
            for sequence, identity_value in enumerate(order, 1):
                if identity_value in started: continue
                if len(started) >= MAX_ATTEMPTS: raise RuntimeError("global HTTP attempt cap reached")
                item = by_id[identity_value]
                start = {"schema": "BATCH-UNKNOWN-PROBE-ATTEMPT-EVENT-1", "event": "attempt_started", "attempt_id": secrets.token_hex(16), "instance_id": identity_value, "sequence": sequence, "request_body_sha256": item["body_sha256"], "bindings": item["bindings"], "endpoint": endpoint, "started_at": now()}
                append_event(journal_path, start); started.add(identity_value)
                try: outcome = dispatch(client, endpoint, api_key, item)
                except KeyboardInterrupt: raise
                except Exception as exc:
                    outcome = {"outcome": "transport_error", "error": f"{type(exc).__name__}: {exc}", "http_status": None, "raw_response_body_base64": None, "raw_response_body_sha256": None, "raw_response_body_bytes": 0, "response_body_truncated": False, "finish_reason": None, "content": None, "actual_model": None, "actual_provider": None, "usage": None, "usage_cost": None, "framing_status": "not_examined", "normalization_operations": [], "raw_translation_order": None, "ordered_translations": None}
                append_event(journal_path, {"schema": "BATCH-UNKNOWN-PROBE-ATTEMPT-EVENT-1", "event": "attempt_terminal", "attempt_id": start["attempt_id"], "instance_id": identity_value, "finished_at": now(), **outcome}); terminal[identity_value] = outcome
        events = load_journal(journal_path); started, terminal = validate_journal(events, catalog, endpoint)
        packet_path = write_packet(directory, build_blind_packet(cases, private, terminal))
        provenance = {"schema": "BATCH-UNKNOWN-PROBE-EXECUTION-PROVENANCE-1", "mode": mode, "endpoint": endpoint, "cases_sha256": plan["cases_sha256"], "journal_sha256": digest_file(journal_path), "blind_packet_sha256": digest_file(packet_path), "execution_identity_sha256": digest_file(directory / "execution_identity.json"), "private_key_sha256": digest_file(directory / "private_key.json"), "catalog_order_sha256": digest_file(directory / "catalog_order.json"), "attempt_started": len(started), "attempt_terminal": len(terminal), "rerun_duplicate_attempts": 0, "indeterminate": len(started - set(terminal)), "successful": sum(event.get("outcome") == "success" for event in terminal.values()), "failed": sum(event.get("outcome") != "success" for event in terminal.values()), "reported_completion_tokens": sum(event.get("usage", {}).get("completion_tokens", 0) for event in terminal.values() if isinstance(event.get("usage"), dict) and isinstance(event["usage"].get("completion_tokens"), int)), "reported_cost": sum(float(event.get("usage_cost") or 0) for event in terminal.values())}
        durable_json(directory / "execution_provenance.json", provenance, exclusive=True)
        return provenance


def candidate_identity(candidate: dict[str, Any]) -> Any:
    return (candidate.get("status"), candidate.get("source_segments"), candidate.get("ordered_translations"))


def validate_ratings(value: Any, packet: dict[str, Any], *, locked: bool) -> list[dict[str, Any]]:
    expected_top = {"schema", "rater", "ratings", *( ["blind_packet_sha256"] if locked else [] )}
    if not isinstance(value, dict) or set(value) != expected_top or value.get("schema") != "BATCH-TRANSLATION-RATINGS-1" or not isinstance(value.get("rater"), str) or not value["rater"].strip() or not isinstance(value.get("ratings"), list): raise RuntimeError("ratings schema invalid")
    if locked and value.get("blind_packet_sha256") != digest_bytes((json.dumps(packet, ensure_ascii=False, indent=2) + "\n").encode("utf-8")): raise RuntimeError("locked ratings packet binding mismatch")
    rows = value["ratings"]; cases = packet.get("cases")
    if [row.get("opaque_case_id") for row in rows if isinstance(row, dict)] != [case.get("opaque_case_id") for case in cases]: raise RuntimeError("ratings coverage/order mismatch")
    fields = {"opaque_case_id", "meaning_preference", "clear_win", "readability_preference", "evidence", "source_uncertainty", "new_severe_errors_relative_to_other"}
    for row, case in zip(rows, cases):
        if not isinstance(row, dict) or set(row) != fields or row["meaning_preference"] not in {"X", "Y", "equal", "unjudgeable"} or row["readability_preference"] not in {"X", "Y", "equal", "unjudgeable"} or not isinstance(row["clear_win"], bool): raise RuntimeError("rating row invalid")
        if not isinstance(row["evidence"], str) or not row["evidence"].strip() or not isinstance(row["source_uncertainty"], str) or not row["source_uncertainty"].strip(): raise RuntimeError("rating evidence invalid")
        severe = row["new_severe_errors_relative_to_other"]
        if not isinstance(severe, dict) or set(severe) != {"X", "Y"}: raise RuntimeError("relative severe schema invalid")
        for label in ("X", "Y"):
            if not isinstance(severe[label], list): raise RuntimeError("relative severe list invalid")
            for item in severe[label]:
                if not isinstance(item, dict) or set(item) != {"kind", "detail"} or item.get("kind") not in SEVERE_KINDS or not isinstance(item.get("detail"), str) or not item["detail"].strip(): raise RuntimeError("relative severe item invalid")
        candidates = case["candidates"]; available = all(candidates[label]["status"] == "available" for label in ("X", "Y")); identical = available and candidate_identity(candidates["X"]) == candidate_identity(candidates["Y"])
        if not available and (row["meaning_preference"] != "unjudgeable" or row["readability_preference"] != "unjudgeable" or row["clear_win"] or severe["X"] or severe["Y"]): raise RuntimeError("unavailable pair must be unjudgeable")
        if identical and (row["meaning_preference"] != "equal" or row["readability_preference"] != "equal" or row["clear_win"] or severe["X"] or severe["Y"]): raise RuntimeError("identical pair must be equal")
        if row["clear_win"] and (row["meaning_preference"] not in {"X", "Y"} or severe[row["meaning_preference"]]): raise RuntimeError("clear winner invalid")
    return rows


def lock_ratings(directory: Path, ratings_path: Path) -> dict[str, Any]:
    directory = directory.resolve(); provenance = validate_finalized(directory)
    if provenance is None: raise RuntimeError("ratings require finalized execution")
    packet_path = directory / "blind_packet.json"; packet = load_json(packet_path); ratings = load_json(ratings_path); rows = validate_ratings(ratings, packet, locked=False)
    locked = {**ratings, "blind_packet_sha256": digest_file(packet_path)}; encoded = (json.dumps(locked, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    durable_write(directory / "ratings.locked.json", encoded, exclusive=True)
    return {"ratings": len(rows), "ratings_locked_sha256": digest_bytes(encoded), "blind_packet_sha256": provenance["blind_packet_sha256"]}


def decode(directory: Path) -> dict[str, Any]:
    directory = directory.resolve(); provenance = validate_finalized(directory)
    if provenance is None: raise RuntimeError("decode requires finalized execution")
    plan, cases, _ = validate_frozen(); private = create_or_load_private_key(directory, cases, create=False); packet = load_json(directory / "blind_packet.json")
    ratings_path = directory / "ratings.locked.json"
    if not ratings_path.exists(): raise RuntimeError("locked ratings missing")
    ratings = load_json(ratings_path); rows = validate_ratings(ratings, packet, locked=True); mapping = {entry["opaque_case_id"]: entry for entry in private["entries"]}; by_parent = {case["parent_id"]: case for case in cases}
    decoded = []
    for row in rows:
        key = mapping[row["opaque_case_id"]]; arm_by_label = key["mapping"]; pref = row["meaning_preference"]; read = row["readability_preference"]
        decoded.append({"parent_id": key["parent_id"], "replicate": key["replicate"], "cluster_id": by_parent[key["parent_id"]]["cluster_id"], "no_op_guard": by_parent[key["parent_id"]]["no_op_guard"], "meaning_B_relative_to_A": ("improved" if arm_by_label[pref] == "B" else "worse") if pref in {"X", "Y"} else pref, "clear_win": row["clear_win"], "readability_B_relative_to_A": ("improved" if arm_by_label[read] == "B" else "worse") if read in {"X", "Y"} else read, "evidence": row["evidence"], "source_uncertainty": row["source_uncertainty"], "new_severe_errors_relative_to_other": {arm_by_label[label]: row["new_severe_errors_relative_to_other"][label] for label in ("X", "Y")}})
    primary_changed = [row for row in decoded if row["replicate"] == 0 and not row["no_op_guard"]]; counts = {key: sum(row["meaning_B_relative_to_A"] == key for row in primary_changed) for key in ("improved", "equal", "worse", "unjudgeable")}
    clear = [row for row in primary_changed if row["clear_win"] and row["meaning_B_relative_to_A"] == "improved"]
    risks = [entry for row in decoded for entry in row["new_severe_errors_relative_to_other"]["B"]]
    repeat_stability = {parent_id: [{"replicate": row["replicate"], "meaning_B_relative_to_A": row["meaning_B_relative_to_A"], "clear_win": row["clear_win"]} for row in decoded if row["parent_id"] == parent_id] for parent_id in plan["design"]["repeat_parent_ids"]}
    result = {"schema": "BATCH-UNKNOWN-PROBE-DECODED-1", "ratings_locked_sha256": digest_file(ratings_path), "blind_packet_sha256": provenance["blind_packet_sha256"], "primary_changed_counts": counts, "primary_cluster_distribution": plan["design"]["primary_cluster_distribution"], "no_op_guards": [row for row in decoded if row["replicate"] == 0 and row["no_op_guard"]], "repeat_stability_not_independent": repeat_stability, "idealized_r7_two_free_slots": plan["design"]["idealized_r7"], "exploratory_signal": {"clear_B_wins": len(clear), "clear_win_groups": sorted({row["cluster_id"] for row in clear}), "clear_win_signal_met": len(clear) >= 3 and len({row["cluster_id"] for row in clear}) >= 2, "new_severe_B_relative_errors": risks, "interpretation": "The clear-win signal and separate risk list are descriptive exploratory evidence only; neither is an automatic adoption veto, and neither resets the existing adoption."}, "decoded_pairs": decoded}
    path = directory / "decoded_ratings.json"
    if path.exists() and load_json(path) != result: raise RuntimeError("decoded result already differs")
    if not path.exists(): durable_json(path, result, exclusive=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(); sub = parser.add_subparsers(dest="command", required=True); sub.add_parser("check")
    execute_parser = sub.add_parser("execute"); execute_parser.add_argument("--execution-dir", type=Path, default=DEFAULT_EXECUTION_DIR); execute_parser.add_argument("--endpoint", default=OFFICIAL_ENDPOINT); execute_parser.add_argument("--timeout", type=float, default=45.0)
    lock_parser = sub.add_parser("lock-ratings"); lock_parser.add_argument("ratings", type=Path); lock_parser.add_argument("--execution-dir", type=Path, default=DEFAULT_EXECUTION_DIR)
    decode_parser = sub.add_parser("decode"); decode_parser.add_argument("--execution-dir", type=Path, default=DEFAULT_EXECUTION_DIR)
    args = parser.parse_args()
    if args.command == "check":
        plan, cases, catalog = validate_frozen(); result = {"ok": True, "parents": len(cases), "pairs": sum(case["replicates"] for case in cases), "catalog_attempts": len(catalog), "logical_arm_bindings": sum(len(item["bindings"]) for item in catalog), "maximum_output_tokens": len(catalog) * MAX_TOKENS, "cases_sha256": plan["cases_sha256"], "credential_available": credential() is not None}
    elif args.command == "execute": result = execute(args.execution_dir, args.endpoint, args.timeout)
    elif args.command == "lock-ratings": result = lock_ratings(args.execution_dir, args.ratings)
    else: result = decode(args.execution_dir)
    print(canonical(result))


if __name__ == "__main__":
    main()
