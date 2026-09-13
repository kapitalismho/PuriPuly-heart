from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.util
import json
import os
import secrets
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import urlparse

import httpx

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_PREPARE_SPEC = importlib.util.spec_from_file_location("meaning_first_probe_prepare", HERE / "prepare.py")
if _PREPARE_SPEC is None or _PREPARE_SPEC.loader is None:
    raise RuntimeError("cannot load frozen prepare.py")
_PREPARE = importlib.util.module_from_spec(_PREPARE_SPEC)
_PREPARE_SPEC.loader.exec_module(_PREPARE)
provider_body = _PREPARE.provider_body
validate_response = _PREPARE.validate_response

AUTHORITY_FILE = HERE / "execution_authority.json"
CASES_FILE = HERE / "cases.jsonl"
DEFAULT_EXECUTION_DIR = HERE / "execution"
OFFICIAL_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
EXPECTED_HASHES = {
    "prepare.py": "1802817a406da298832710db680ae7d8e91838e3cd98f242365c0917b002a905",
    "plan.json": "23a639e8c0918f9a6f1e298534841e9710a369b0e7abb9a70a4f9cebcdf69c9e",
    "cases.jsonl": "4724612e95c88852054be39ce55ba558365f3511a52a02889a95be59d5942dbb",
    "decision.json": "16844888d4bef3693dcaa3c5e9fb49c09c567987a272641d9c7fe4498889f155",
}
EXPECTED_CASE_IDS = ["Q1", "Q2", "Q3", "Q4", "R1", "R2", "C1", "C2", "S1", "S2", "U1", "O1"]
EXPECTED_MODEL = "google/gemma-4-26b-a4b-it"
EXPECTED_PROVIDER = {
    "order": ["wafer", "cloudflare", "deepinfra"],
    "only": ["wafer", "cloudflare", "deepinfra"],
    "allow_fallbacks": True,
}
RAW_RESPONSE_LIMIT = 131_072
SEVERE_KINDS = {
    "actual negation/agreement reversal",
    "wrong claim ownership",
    "omission",
    "invention",
}

def _is_guard_category(value: Any) -> bool:
    return isinstance(value, str) and value.startswith("guard_")


def _normalize_segment_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    return " ".join(value.split())


def _candidate_identity(candidate: Any) -> Any:
    if not isinstance(candidate, dict):
        return (None, None)
    if candidate.get("status") != "available":
        return (candidate.get("status"), candidate.get("reason"))
    segments = candidate.get("segments")
    if not isinstance(segments, list) or not segments:
        return ("available", None)
    normalized: list[Any] = []
    for item in segments:
        if not isinstance(item, dict):
            return ("available", None)
        if not isinstance(item.get("id"), str):
            return ("available", None)
        normalized.append((item.get("id"), _normalize_segment_text(item.get("text"))))
    return ("available", tuple(normalized))


def _candidates_share_identical_guard_output(candidates: Any) -> bool:
    if not isinstance(candidates, dict):
        return False
    first = candidates.get("X")
    second = candidates.get("Y")
    if not isinstance(first, dict) or not isinstance(second, dict):
        return False
    if first.get("status") != "available" or second.get("status") != "available":
        return False
    return _candidate_identity(first) == _candidate_identity(second)


def _resolve_execution_dir(value: Path) -> Path:
    return Path(value).expanduser().resolve()


def _require_execution_mode(directory: Path, endpoint: str) -> tuple[Path, str]:
    resolved = _resolve_execution_dir(directory)
    canonical = _resolve_execution_dir(DEFAULT_EXECUTION_DIR)
    if endpoint == OFFICIAL_ENDPOINT:
        if resolved != canonical:
            raise RuntimeError("official endpoint requires the canonical execution directory")
        return resolved, "official_provider"
    if resolved == canonical:
        raise RuntimeError("the canonical execution directory accepts only the official endpoint")
    return resolved, "loopback_simulation"


def _journal_provenance(journal_path: Path, events: list[dict[str, Any]], started: set[str], terminal: dict[str, dict[str, Any]]) -> dict[str, Any]:
    endpoints = sorted({event.get("endpoint") for event in events if event.get("event") == "attempt_started" and isinstance(event.get("endpoint"), str)})
    outcome_counts: dict[str, int] = {}
    for event in events:
        if event.get("event") == "attempt_terminal":
            outcome = event.get("outcome")
            if isinstance(outcome, str):
                outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1
    successful = sum(1 for event in terminal.values() if event.get("outcome") == "success")
    failed = sum(1 for event in terminal.values() if event.get("outcome") != "success")
    if len(endpoints) == 1 and endpoints[0] == OFFICIAL_ENDPOINT:
        mode = "official_provider"
    elif len(endpoints) == 1:
        try:
            checked = validate_endpoint(endpoints[0])
        except ValueError:
            checked = ""
        mode = "loopback_simulation" if checked and checked != OFFICIAL_ENDPOINT else "mixed_invalid"
    elif not endpoints:
        mode = "unknown_empty"
    else:
        mode = "mixed_invalid"
    return {
        "schema": "meaning-first-e1-journal-provenance-1",
        "journal_sha256": digest_file(journal_path) if journal_path.exists() else None,
        "endpoint_set": endpoints,
        "execution_mode": mode,
        "attempt_started": len([event for event in events if event.get("event") == "attempt_started"]),
        "attempt_terminal": len([event for event in events if event.get("event") == "attempt_terminal"]),
        "unique_attempts": len(started),
        "successful": successful,
        "failed": failed,
        "indeterminate": len(started) - len(terminal),
        "outcome_counts": outcome_counts,
    }


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def digest_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(65_536), b""):
            h.update(chunk)
    return h.hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def durable_write(path: Path, data: bytes, *, exclusive: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(fd, "wb") as target:
                target.write(data)
                target.flush()
                os.fsync(target.fileno())
        except BaseException:
            try:
                path.unlink()
            except OSError:
                pass
            raise
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp")
    try:
        with temporary.open("xb") as target:
            target.write(data)
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def durable_json(path: Path, value: Any, *, exclusive: bool = False) -> None:
    durable_write(path, (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8"), exclusive=exclusive)


def append_event(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab", buffering=0) as target:
        target.write((canonical(value) + "\n").encode("utf-8"))
        os.fsync(target.fileno())


@contextmanager
def execution_lock(directory: Path) -> Iterator[None]:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "execution.lock"
    target = path.open("a+b")
    target.seek(0, os.SEEK_END)
    if target.tell() == 0:
        target.write(b"0")
        target.flush()
        os.fsync(target.fileno())
    target.seek(0)
    try:
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(target.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(target.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        target.close()
        raise RuntimeError("another E1 execution process holds the journal lock") from exc
    try:
        yield
    finally:
        target.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(target.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(target.fileno(), fcntl.LOCK_UN)
        target.close()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_journal(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    result = []
    with path.open("r", encoding="utf-8") as source:
        for number, line in enumerate(source, 1):
            if not line.endswith("\n"):
                raise RuntimeError(f"journal line {number} is not durably terminated")
            event = json.loads(line)
            if not isinstance(event, dict):
                raise RuntimeError(f"journal line {number} is not an object")
            result.append(event)
    return result


def load_cases() -> list[dict[str, Any]]:
    records = [json.loads(line) for line in CASES_FILE.read_text(encoding="utf-8").splitlines() if line]
    return [row for row in records if row.get("record_type") == "E1_frozen_case"]


def check_frozen() -> dict[str, Any]:
    authority = load_json(AUTHORITY_FILE)
    if authority.get("schema") != "MEANING-FIRST-E1-EXECUTION-AUTHORITY-1" or authority.get("authority_id") != "E1-USER-EXECUTE-1":
        raise RuntimeError("execution authority identity mismatch")
    if authority.get("frozen_input_sha256") != EXPECTED_HASHES:
        raise RuntimeError("execution authority hash pins differ from runner pins")
    actual_hashes = {name: digest_file(HERE / name) for name in EXPECTED_HASHES}
    if actual_hashes != EXPECTED_HASHES:
        raise RuntimeError("frozen input hash mismatch")
    scope = authority.get("scope")
    expected_scope = {
        "dev_parents": 12,
        "logical_arm_bindings": 24,
        "maximum_unique_http_translation_attempts": 20,
        "maximum_attempts_per_request_fingerprint": 1,
        "max_tokens_per_request": 512,
        "model": EXPECTED_MODEL,
        "temperature": 0.6,
        "request_bodies": "Exactly the canonical provider request bodies frozen in cases.jsonl; share identical bodies across arm bindings.",
        "automatic_retries": 0,
        "decision_sensitive_repeats": 0,
        "account_pricing_billing_queries": False,
        "old_budget_ledger_mutation": False,
        "native_asr_model_runs": False,
        "holdout_access": False,
        "production_policy_changes": False,
        "E2_capture": False,
        "git_remote_mutations": False,
    }
    if scope != expected_scope:
        raise RuntimeError("execution authority scope mismatch")
    cases = load_cases()
    if [case.get("case_id") for case in cases] != EXPECTED_CASE_IDS or len({case.get("parent_id") for case in cases}) != 12:
        raise RuntimeError("frozen case coverage or order mismatch")
    fingerprints: dict[str, dict[str, Any]] = {}
    bindings = 0
    guard_count = 0
    for case in cases:
        accepted = case.get("accepted_text")
        if not isinstance(accepted, str) or digest_bytes(accepted.encode("utf-8")) != case.get("accepted_text_sha256"):
            raise RuntimeError(f"accepted input identity mismatch for {case.get('case_id')}")
        if not isinstance(case.get("source_gt_facts_frozen_before_generation"), dict):
            raise RuntimeError(f"missing frozen fact table for {case.get('case_id')}")
        for arm_name in ("B", "G"):
            binding = case["arms"][arm_name]
            payload = binding["logical_payload"]
            segments = payload.get("segments")
            if payload.get("schema") != "meaning-first-e1-input-1" or payload.get("parent_text") != accepted:
                raise RuntimeError(f"logical payload mismatch for {case['case_id']} {arm_name}")
            if not isinstance(segments, list) or "".join(item.get("text", "") for item in segments if isinstance(item, dict)) != accepted:
                raise RuntimeError(f"segment concatenation mismatch for {case['case_id']} {arm_name}")
            expected_ids = [item.get("id") for item in segments]
            if expected_ids != binding.get("expected_output_ids") or any(not isinstance(item, str) for item in expected_ids):
                raise RuntimeError(f"expected output IDs mismatch for {case['case_id']} {arm_name}")
            body = binding["provider_request_body"]
            if body != provider_body(payload):
                raise RuntimeError(f"provider body reconstruction mismatch for {case['case_id']} {arm_name}")
            if body.get("model") != EXPECTED_MODEL or body.get("temperature") != 0.6 or body.get("max_tokens") != 512:
                raise RuntimeError(f"model configuration mismatch for {case['case_id']} {arm_name}")
            if body.get("reasoning") != {"effort": "none"} or body.get("provider") != EXPECTED_PROVIDER:
                raise RuntimeError(f"provider configuration mismatch for {case['case_id']} {arm_name}")
            fingerprint = digest_bytes(canonical(body).encode("utf-8"))
            if fingerprint != binding.get("canonical_request_sha256"):
                raise RuntimeError(f"request fingerprint mismatch for {case['case_id']} {arm_name}")
            prior = fingerprints.setdefault(fingerprint, body)
            if prior != body:
                raise RuntimeError("request fingerprint collision")
            bindings += 1
        if case["category"].startswith("guard_"):
            guard_count += 1
            if case["arms"]["B"]["canonical_request_sha256"] != case["arms"]["G"]["canonical_request_sha256"]:
                raise RuntimeError(f"guard does not share one request for {case['case_id']}")
    if bindings != 24 or len(fingerprints) != 20 or guard_count != 4:
        raise RuntimeError("authorized request cardinality mismatch")
    return {
        "ok": True,
        "authority_id": authority["authority_id"],
        "cases": len(cases),
        "logical_bindings": bindings,
        "unique_requests": len(fingerprints),
        "max_tokens": 512,
        "model": EXPECTED_MODEL,
        "frozen_input_sha256": actual_hashes,
    }


def credential() -> str | None:
    try:
        from experiments.psem_r2_policy.credentials import load_runtime_secrets

        value = load_runtime_secrets().get("OPENROUTER_API_KEY")
        if isinstance(value, str) and value.strip():
            return value.strip()
    except Exception:
        pass
    try:
        from puripuly_heart.core.storage.secrets import KeyringSecretStore

        for service in ("puripuly-heart-vnext", "puripuly-heart"):
            value = KeyringSecretStore(service_name=service).get("openrouter_api_key")
            if isinstance(value, str) and value.strip():
                return value.strip()
    except Exception:
        return None
    return None


def validate_endpoint(value: str) -> str:
    parsed = urlparse(value)
    if value == OFFICIAL_ENDPOINT:
        return value
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("endpoint must be the official OpenRouter endpoint or local loopback HTTP")
    if parsed.path != "/api/v1/chat/completions" or parsed.username is not None or parsed.password is not None:
        raise ValueError("loopback endpoint must use /api/v1/chat/completions without user information")
    return value


def create_or_load_private_key(directory: Path, cases: list[dict[str, Any]]) -> dict[str, Any]:
    path = directory / "private_key.json"
    if not path.exists():
        entries = []
        used = set()
        for case in cases:
            opaque = secrets.token_hex(12)
            while opaque in used:
                opaque = secrets.token_hex(12)
            used.add(opaque)
            if secrets.randbits(1):
                mapping = {"X": "B", "Y": "G"}
            else:
                mapping = {"X": "G", "Y": "B"}
            entries.append({
                "opaque_case_id": opaque,
                "case_id": case["case_id"],
                "parent_id": case["parent_id"],
                "group": case["cluster_id"],
                "category": case["category"],
                "mapping": mapping,
            })
        value = {
            "schema": "meaning-first-e1-private-key-1",
            "created_at": now(),
            "cases_sha256": digest_file(CASES_FILE),
            "cases": entries,
        }
        durable_json(path, value, exclusive=True)
    return validate_private_key(load_json(path), cases)


def validate_private_key(value: Any, cases: list[dict[str, Any]]) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {"schema", "created_at", "cases_sha256", "cases"}:
        raise RuntimeError("private key schema invalid")
    if value.get("schema") != "meaning-first-e1-private-key-1" or value.get("cases_sha256") != EXPECTED_HASHES["cases.jsonl"]:
        raise RuntimeError("private key identity mismatch")
    entries = value.get("cases")
    if not isinstance(entries, list) or [entry.get("case_id") for entry in entries if isinstance(entry, dict)] != EXPECTED_CASE_IDS:
        raise RuntimeError("private key case coverage mismatch")
    opaque = [entry.get("opaque_case_id") for entry in entries]
    if any(not isinstance(item, str) or not item for item in opaque) or len(set(opaque)) != 12:
        raise RuntimeError("private key opaque IDs invalid")
    for entry, case in zip(entries, cases):
        expected = {
            "case_id": case["case_id"],
            "parent_id": case["parent_id"],
            "group": case["cluster_id"],
            "category": case["category"],
        }
        if any(entry.get(name) != expected_value for name, expected_value in expected.items()):
            raise RuntimeError("private key frozen case binding mismatch")
        if set(entry) != {"opaque_case_id", "case_id", "parent_id", "group", "category", "mapping"}:
            raise RuntimeError("private key case schema invalid")
        if set(entry.get("mapping", {})) != {"X", "Y"} or set(entry["mapping"].values()) != {"B", "G"}:
            raise RuntimeError("private key arm mapping invalid")
    return value


def request_catalog(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    catalog: dict[str, dict[str, Any]] = {}
    for case in cases:
        for arm in ("B", "G"):
            binding = case["arms"][arm]
            fingerprint = binding["canonical_request_sha256"]
            item = catalog.setdefault(fingerprint, {
                "fingerprint": fingerprint,
                "body": binding["provider_request_body"],
                "expected_ids": binding["expected_output_ids"],
                "bindings": [],
            })
            if item["body"] != binding["provider_request_body"] or item["expected_ids"] != binding["expected_output_ids"]:
                raise RuntimeError("shared request binding mismatch")
            item["bindings"].append({"case_id": case["case_id"], "arm": arm})
    return list(catalog.values())


def existing_attempts(events: list[dict[str, Any]]) -> tuple[set[str], dict[str, dict[str, Any]]]:
    started: set[str] = set()
    starts: dict[str, dict[str, Any]] = {}
    terminal: dict[str, dict[str, Any]] = {}
    for event in events:
        if event.get("schema") != "meaning-first-e1-attempt-event-1":
            raise RuntimeError("journal event schema invalid")
        fingerprint = event.get("request_fingerprint")
        if not isinstance(fingerprint, str):
            raise RuntimeError("journal event missing request fingerprint")
        if event.get("event") == "attempt_started":
            if fingerprint in started or not isinstance(event.get("attempt_id"), str):
                raise RuntimeError("journal contains repeated or invalid request attempt")
            started.add(fingerprint)
            starts[fingerprint] = event
        elif event.get("event") == "attempt_terminal":
            if fingerprint not in started or fingerprint in terminal:
                raise RuntimeError("journal terminal event has no unique start")
            if event.get("attempt_id") != starts[fingerprint]["attempt_id"]:
                raise RuntimeError("journal terminal attempt identity mismatch")
            terminal[fingerprint] = event
        else:
            raise RuntimeError("journal event type invalid")
    if len(started) > 20:
        raise RuntimeError("journal exceeds global request limit")
    return started, terminal


def validate_journal_catalog(events: list[dict[str, Any]], catalog: list[dict[str, Any]], endpoint: str | None = None) -> tuple[set[str], dict[str, dict[str, Any]]]:
    started, terminal = existing_attempts(events)
    expected = {item["fingerprint"]: item for item in catalog}
    if not started <= set(expected):
        raise RuntimeError("journal contains a non-frozen request fingerprint")
    endpoints: set[str] = set()
    for event in events:
        if event["event"] != "attempt_started":
            continue
        item = expected[event["request_fingerprint"]]
        if event.get("request_body_sha256") != digest_bytes(canonical(item["body"]).encode("utf-8")) or event.get("bindings") != item["bindings"]:
            raise RuntimeError("journal start does not match its frozen request binding")
        recorded_endpoint = event.get("endpoint")
        if not isinstance(recorded_endpoint, str):
            raise RuntimeError("journal start endpoint missing")
        try:
            validate_endpoint(recorded_endpoint)
        except ValueError as exc:
            raise RuntimeError("journal contains a non-approved endpoint") from exc
        endpoints.add(recorded_endpoint)
    if len(endpoints) > 1:
        raise RuntimeError("journal contains mixed endpoints")
    if endpoint is not None and endpoints and endpoints != {endpoint}:
        raise RuntimeError("journal endpoint differs from the requested endpoint")
    return started, terminal


def response_outcome(response: httpx.Response, raw: bytes, truncated: bool, expected_ids: list[str]) -> dict[str, Any]:
    base = {
        "http_status": response.status_code,
        "response_body_truncated": truncated,
        "raw_response_body_base64": base64.b64encode(raw).decode("ascii"),
        "raw_response_body_sha256": digest_bytes(raw),
        "raw_response_body_bytes": len(raw),
    }
    if truncated:
        return {**base, "outcome": "response_body_truncated", "error": "response exceeded retained body limit"}
    if response.status_code != 200:
        return {**base, "outcome": "http_error", "error": f"HTTP {response.status_code}", "actual_model": None, "actual_provider": None, "usage": None, "usage_cost": None}
    try:
        envelope = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return {**base, "outcome": "invalid_response_json", "error": f"{type(exc).__name__}: {exc}"}
    if not isinstance(envelope, dict):
        return {**base, "outcome": "invalid_response_envelope", "error": "response JSON is not an object"}
    usage = envelope.get("usage") if isinstance(envelope.get("usage"), dict) else None
    metadata = {
        "actual_model": envelope.get("model") if isinstance(envelope.get("model"), str) else None,
        "actual_provider": envelope.get("provider") if isinstance(envelope.get("provider"), str) else None,
        "usage": usage,
        "usage_cost": usage.get("cost") if usage is not None else None,
    }
    choices = envelope.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return {**base, **metadata, "outcome": "missing_choices", "error": "missing first choice"}
    choice = choices[0]
    if choice.get("finish_reason") == "length":
        return {**base, **metadata, "outcome": "finish_reason_length", "error": "provider reported length truncation"}
    message = choice.get("message")
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str) or not content.strip():
        return {**base, **metadata, "outcome": "empty_content", "error": "first choice has no nonempty text content"}
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        return {**base, **metadata, "outcome": "invalid_content_json", "error": f"JSONDecodeError: {exc}", "parsed_text": content}
    valid, validation_error = validate_response(parsed, expected_ids)
    if not valid:
        return {**base, **metadata, "outcome": validation_error, "error": validation_error, "parsed_text": content}
    return {**base, **metadata, "outcome": "success", "error": None, "parsed_text": content, "validated_output": parsed}


def dispatch(client: httpx.Client, endpoint: str, api_key: str, item: dict[str, Any]) -> dict[str, Any]:
    raw = bytearray()
    truncated = False
    with client.stream(
        "POST",
        endpoint,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        content=canonical(item["body"]).encode("utf-8"),
    ) as response:
        for chunk in response.iter_bytes():
            remaining = RAW_RESPONSE_LIMIT - len(raw)
            if remaining > 0:
                raw.extend(chunk[:remaining])
            if len(chunk) > remaining:
                truncated = True
                break
        return response_outcome(response, bytes(raw), truncated, item["expected_ids"])


def source_provenance(cases: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema": "meaning-first-e1-source-provenance-1",
        "cases_sha256": EXPECTED_HASHES["cases.jsonl"],
        "cases": [{
            "case_id": case["case_id"],
            "parent_id": case["parent_id"],
            "meeting": case["meeting"],
            "group": case["cluster_id"],
            "category": case["category"],
            "accepted_text_sha256": case["accepted_text_sha256"],
            "annotation_identities_sha256": digest_bytes(canonical(case["annotation_identities"]).encode("utf-8")),
            "token_provenance_sha256": digest_bytes(canonical(case["token_provenance"]).encode("utf-8")),
        } for case in cases],
    }


def write_provenance(directory: Path, cases: list[dict[str, Any]]) -> None:
    value = source_provenance(cases)
    path = directory / "source_provenance.json"
    if path.exists() and load_json(path) != value:
        raise RuntimeError("source provenance artifact differs")
    if not path.exists():
        durable_json(path, value, exclusive=True)


def write_execution_provenance(directory: Path, value: dict[str, Any]) -> None:
    path = directory / "execution_provenance.json"
    if path.exists() and load_json(path) != value:
        raise RuntimeError("execution provenance artifact differs from the journal")
    if not path.exists():
        durable_json(path, value, exclusive=True)

def build_blind_packet(directory: Path, cases: list[dict[str, Any]], private: dict[str, Any], terminal: dict[str, dict[str, Any]]) -> dict[str, Any]:
    case_by_id = {case["case_id"]: case for case in cases}
    packet_cases = []
    for private_case in private["cases"]:
        case = case_by_id[private_case["case_id"]]
        candidates = {}
        for label in ("X", "Y"):
            arm = private_case["mapping"][label]
            binding = case["arms"][arm]
            event = terminal.get(binding["canonical_request_sha256"])
            if event is None:
                candidate = {"status": "unavailable", "reason": "attempted_indeterminate"}
            elif event.get("outcome") != "success":
                candidate = {"status": "unavailable", "reason": event.get("outcome")}
            else:
                output = event["validated_output"]
                candidate = {"status": "available", "segments": output["translations"]}
            candidates[label] = candidate
        packet_cases.append({
            "opaque_case_id": private_case["opaque_case_id"],
            "accepted_source": case["accepted_text"],
            "frozen_source_facts": case["source_gt_facts_frozen_before_generation"],
            "context": "",
            "candidates": candidates,
        })
    return {
        "schema": "meaning-first-e1-blind-packet-1",
        "partial_blinding": "Candidate segment counts remain visible. Arm identity, case category, group, policy, hypothesis, and expected winner are withheld.",
        "rating_contract": {
            "meaning_preference": ["X", "Y", "equal", "unjudgeable"],
            "readability_preference": ["X", "Y", "equal", "unjudgeable"],
            "supplementary_each_candidate": {"adequacy": "integer 0-3", "faithfulness": "integer 0-3", "korean_fluency": "integer 0-3"},
            "severe_error_values": sorted(SEVERE_KINDS),
            "new_severe_errors_relative_to_other": "for X and Y list {kind, evidence}; judge whether this candidate introduces a severe error absent from the other candidate even when both share the same absolute category; identical candidates introduce nothing new",
            "clear_win": "true only for a clear main-meaning advantage between actually different available candidates; identical candidates, unavailable candidates, and guard no-cut controls are never clear wins",
            "source_facts_valid": "required boolean for X and Y in every case; judge against the unchanged frozen fact table",
            "unavailable": "retain every case in the denominator and rate main meaning unjudgeable when either candidate is unavailable",
        },
        "ratings_schema": {
            "schema": "meaning-first-e1-ratings-1",
            "rater": {"identity": "nonempty", "independence": "fresh_independent_read_only_agent", "blinding": "partial_blind"},
            "ratings": [{
                "opaque_case_id": "from packet",
                "meaning_preference": "X|Y|equal|unjudgeable",
                "clear_win": False,
                "readability_preference": "X|Y|equal|unjudgeable",
                "supplementary": {"X": {"adequacy": 0, "faithfulness": 0, "korean_fluency": 0}, "Y": {"adequacy": 0, "faithfulness": 0, "korean_fluency": 0}},
                "severe_errors": {"X": [], "Y": []},
                "new_severe_errors_relative_to_other": {"X": [], "Y": []},
                "source_facts_valid": {"X": False, "Y": False},
            }],
        },
        "cases": packet_cases,
    }


def execute(directory: Path, endpoint: str, timeout: float) -> dict[str, Any]:
    endpoint = validate_endpoint(endpoint)
    directory, requested_mode = _require_execution_mode(directory, endpoint)
    if timeout <= 0 or timeout > 120:
        raise ValueError("timeout must be greater than 0 and at most 120 seconds")
    check = check_frozen()
    cases = load_cases()
    catalog = request_catalog(cases)
    with execution_lock(directory):
        journal_path = directory / "attempts.jsonl"
        events = load_journal(journal_path)
        started, terminal = validate_journal_catalog(events, catalog, endpoint)
        api_key = credential()
        if api_key is None:
            raise RuntimeError("OpenRouter credential unavailable")
        private = create_or_load_private_key(directory, cases)
        write_provenance(directory, cases)
        client_timeout = httpx.Timeout(timeout, connect=min(timeout, 15.0))
        with httpx.Client(timeout=client_timeout, follow_redirects=False) as client:
            for sequence, item in enumerate(catalog, 1):
                fingerprint = item["fingerprint"]
                if fingerprint in started:
                    continue
                if len(started) >= 20:
                    raise RuntimeError("global request attempt limit reached")
                start_event = {
                    "schema": "meaning-first-e1-attempt-event-1",
                    "event": "attempt_started",
                    "attempt_id": secrets.token_hex(16),
                    "sequence": sequence,
                    "request_fingerprint": fingerprint,
                    "request_body_sha256": digest_bytes(canonical(item["body"]).encode("utf-8")),
                    "bindings": item["bindings"],
                    "endpoint": endpoint,
                    "started_at": now(),
                }
                append_event(journal_path, start_event)
                started.add(fingerprint)
                try:
                    outcome = dispatch(client, endpoint, api_key, item)
                except KeyboardInterrupt:
                    outcome = {"outcome": "interrupted", "error": "KeyboardInterrupt", "http_status": None, "response_body_truncated": False}
                    append_event(journal_path, {"schema": "meaning-first-e1-attempt-event-1", "event": "attempt_terminal", "attempt_id": start_event["attempt_id"], "request_fingerprint": fingerprint, "finished_at": now(), **outcome})
                    raise
                except Exception as exc:
                    outcome = {"outcome": "transport_error", "error": f"{type(exc).__name__}: {exc}", "http_status": None, "response_body_truncated": False}
                terminal_event = {"schema": "meaning-first-e1-attempt-event-1", "event": "attempt_terminal", "attempt_id": start_event["attempt_id"], "request_fingerprint": fingerprint, "finished_at": now(), **outcome}
                append_event(journal_path, terminal_event)
                terminal[fingerprint] = terminal_event
        events = load_journal(journal_path)
        started, terminal = validate_journal_catalog(events, catalog, endpoint)
        provenance = _journal_provenance(journal_path, events, started, terminal)
        if provenance["execution_mode"] != requested_mode:
            raise RuntimeError("journal execution mode differs from the requested mode")
        write_execution_provenance(directory, provenance)
        packet = build_blind_packet(directory, cases, private, terminal)
        packet_path = directory / "blind_packet.json"
        if packet_path.exists() and load_json(packet_path) != packet:
            raise RuntimeError("blind packet exists with different content")
        if not packet_path.exists():
            durable_json(packet_path, packet, exclusive=True)
        return {
            **check,
            "credential_available": True,
            "execution_provenance": provenance,
            "attempted": len(started),
            "terminal": len(terminal),
            "indeterminate": len(started - set(terminal)),
            "successful": sum(event.get("outcome") == "success" for event in terminal.values()),
            "failed": sum(event.get("outcome") != "success" for event in terminal.values()),
            "blind_packet": str(packet_path.resolve()),
        }


def _validate_new_severe_items(items: Any, absolute: list[str]) -> list[dict[str, Any]]:
    if not isinstance(items, list):
        raise RuntimeError("new severe error rating invalid")
    seen: set[str] = set()
    result: list[dict[str, Any]] = []
    absolute_set = set(absolute)
    for entry in items:
        if not isinstance(entry, dict) or set(entry) != {"kind", "evidence"}:
            raise RuntimeError("new severe error rating invalid")
        kind = entry.get("kind")
        evidence = entry.get("evidence")
        if kind not in SEVERE_KINDS or kind in seen or kind not in absolute_set:
            raise RuntimeError("new severe error rating invalid")
        if not isinstance(evidence, str) or not evidence.strip():
            raise RuntimeError("new severe error rating invalid")
        seen.add(kind)
        result.append({"kind": kind, "evidence": evidence})
    return result


def validate_ratings(ratings: Any, packet: dict[str, Any], private: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(ratings, dict) or ratings.get("schema") != "meaning-first-e1-ratings-1" or set(ratings) != {"schema", "rater", "ratings"}:
        raise RuntimeError("ratings top-level schema invalid")
    rater = ratings.get("rater")
    if not isinstance(rater, dict) or set(rater) != {"identity", "independence", "blinding"} or not isinstance(rater.get("identity"), str) or not rater["identity"].strip() or rater.get("independence") != "fresh_independent_read_only_agent" or rater.get("blinding") != "partial_blind":
        raise RuntimeError("rater provenance invalid")
    rows = ratings.get("ratings")
    if not isinstance(rows, list):
        raise RuntimeError("ratings must be an array")
    expected_opaque = [entry["opaque_case_id"] for entry in private["cases"]]
    if [row.get("opaque_case_id") for row in rows if isinstance(row, dict)] != expected_opaque:
        raise RuntimeError("ratings must contain all opaque cases exactly once in packet order")
    packet_status = {case["opaque_case_id"]: case["candidates"] for case in packet["cases"]}
    private_status = {case["opaque_case_id"]: case for case in private["cases"]}
    expected_fields = {
        "opaque_case_id", "meaning_preference", "clear_win", "readability_preference",
        "supplementary", "severe_errors", "new_severe_errors_relative_to_other",
        "source_facts_valid",
    }
    for row in rows:
        if set(row) != expected_fields:
            raise RuntimeError("rating item fields invalid")
        if row["meaning_preference"] not in {"X", "Y", "equal", "unjudgeable"} or row["readability_preference"] not in {"X", "Y", "equal", "unjudgeable"} or not isinstance(row["clear_win"], bool):
            raise RuntimeError("rating preference invalid")
        if row["clear_win"] and row["meaning_preference"] not in {"X", "Y"}:
            raise RuntimeError("clear_win requires a candidate preference")
        candidates = packet_status[row["opaque_case_id"]]
        if any(candidate.get("status") != "available" for candidate in candidates.values()) and row["meaning_preference"] != "unjudgeable":
            raise RuntimeError("unavailable candidate requires unjudgeable meaning")
        if _is_guard_category(private_status[row["opaque_case_id"]]["category"]) and row["clear_win"]:
            raise RuntimeError("guard cases cannot be clear wins")
        identical = _candidates_share_identical_guard_output(candidates)
        if identical and (row["meaning_preference"] != "equal" or row["readability_preference"] != "equal" or row["clear_win"]):
            raise RuntimeError("identical available candidates require equal preferences and cannot be a clear win")
        if set(row["supplementary"]) != {"X", "Y"} or set(row["severe_errors"]) != {"X", "Y"} or set(row["new_severe_errors_relative_to_other"]) != {"X", "Y"} or set(row["source_facts_valid"]) != {"X", "Y"}:
            raise RuntimeError("candidate rating coverage invalid")
        for label in ("X", "Y"):
            scores = row["supplementary"][label]
            if set(scores) != {"adequacy", "faithfulness", "korean_fluency"} or any(type(score) is not int or not 0 <= score <= 3 for score in scores.values()):
                raise RuntimeError("supplementary score invalid")
            errors = row["severe_errors"][label]
            if not isinstance(errors, list) or len(errors) != len(set(errors)) or not set(errors) <= SEVERE_KINDS:
                raise RuntimeError("severe error rating invalid")
            _validate_new_severe_items(row["new_severe_errors_relative_to_other"][label], errors)
            if identical and row["new_severe_errors_relative_to_other"][label]:
                raise RuntimeError("identical candidates cannot introduce a relative severe error")
            if type(row["source_facts_valid"][label]) is not bool:
                raise RuntimeError("source fact validity must be boolean")
    return rows


def assemble(directory: Path, ratings_path: Path) -> dict[str, Any]:
    check_frozen()
    directory = _resolve_execution_dir(directory)
    cases = load_cases()
    private = validate_private_key(load_json(directory / "private_key.json"), cases)
    if load_json(directory / "source_provenance.json") != source_provenance(cases):
        raise RuntimeError("source provenance artifact differs from frozen inputs")
    journal_path = directory / "attempts.jsonl"
    events = load_journal(journal_path)
    catalog = request_catalog(cases)
    started, terminal = validate_journal_catalog(events, catalog)
    provenance = _journal_provenance(journal_path, events, started, terminal)
    if provenance["unique_attempts"] != len(catalog) or len(provenance["endpoint_set"]) != 1:
        raise RuntimeError("journal does not contain the complete frozen request set from one endpoint")
    _, expected_mode = _require_execution_mode(directory, provenance["endpoint_set"][0])
    if provenance["execution_mode"] != expected_mode or load_json(directory / "execution_provenance.json") != provenance:
        raise RuntimeError("execution provenance does not match the journal and execution directory")
    packet = load_json(directory / "blind_packet.json")
    if packet != build_blind_packet(directory, cases, private, terminal):
        raise RuntimeError("blind packet does not match frozen bindings and terminal journal data")
    ratings_bytes = ratings_path.read_bytes()
    ratings = json.loads(ratings_bytes)
    rows = validate_ratings(ratings, packet, private)
    locked_path = directory / "ratings.locked.json"
    normalized = (json.dumps(ratings, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    if locked_path.exists():
        if locked_path.read_bytes() != normalized:
            raise RuntimeError("different ratings are already locked")
    else:
        durable_write(locked_path, normalized, exclusive=True)
    locked_sha = digest_file(locked_path)
    private_by_opaque = {entry["opaque_case_id"]: entry for entry in private["cases"]}
    packet_by_opaque = {case["opaque_case_id"]: case for case in packet["cases"]}
    counts = {key: 0 for key in ("improve", "equal", "worse", "unjudgeable")}
    readability = {key: 0 for key in ("improve", "equal", "worse", "unjudgeable")}
    clear_wins = []
    severe_introduced = []
    score_values = {arm: {metric: [] for metric in ("adequacy", "faithfulness", "korean_fluency")} for arm in ("B", "G")}
    decoded = []
    for row in rows:
        key = private_by_opaque[row["opaque_case_id"]]
        inverse = {arm: label for label, arm in key["mapping"].items()}
        candidates = packet_by_opaque[row["opaque_case_id"]]["candidates"]
        preference = row["meaning_preference"]
        if preference in {"equal", "unjudgeable"}:
            decoded_main = preference
        else:
            decoded_main = "improve" if key["mapping"][preference] == "G" else "worse"
        counts[decoded_main] += 1
        read_preference = row["readability_preference"]
        if read_preference in {"equal", "unjudgeable"}:
            decoded_readability = read_preference
        else:
            decoded_readability = "improve" if key["mapping"][read_preference] == "G" else "worse"
        readability[decoded_readability] += 1
        if row["clear_win"] and decoded_main == "improve" and not _is_guard_category(key["category"]):
            clear_wins.append({"case_id": key["case_id"], "group": key["group"]})
        b_errors = set(row["severe_errors"][inverse["B"]])
        g_errors = set(row["severe_errors"][inverse["G"]])
        for error in row["new_severe_errors_relative_to_other"][inverse["G"]]:
            severe_introduced.append({"case_id": key["case_id"], "error": error["kind"], "evidence": error["evidence"]})
        for arm in ("B", "G"):
            if candidates[inverse[arm]].get("status") != "available":
                continue
            scores = row["supplementary"][inverse[arm]]
            for metric, score in scores.items():
                score_values[arm][metric].append(score)
        decoded.append({
            "case_id": key["case_id"],
            "group": key["group"],
            "category": key["category"],
            "meaning": decoded_main,
            "clear_win": row["clear_win"],
            "readability": decoded_readability,
            "severe_errors": {"B": sorted(b_errors), "G": sorted(g_errors)},
            "new_severe_errors_relative_to_other": {arm: row["new_severe_errors_relative_to_other"][inverse[arm]] for arm in ("B", "G")},
            "source_facts_valid": {arm: row["source_facts_valid"][inverse[arm]] for arm in ("B", "G")},
            "supplementary": {arm: row["supplementary"][inverse[arm]] if candidates[inverse[arm]].get("status") == "available" else None for arm in ("B", "G")},
        })
    guards = []
    for row in rows:
        key = private_by_opaque[row["opaque_case_id"]]
        if not key["category"].startswith("guard_"):
            continue
        inverse = {arm: label for label, arm in key["mapping"].items()}
        candidates = packet_by_opaque[row["opaque_case_id"]]["candidates"]
        valid = all(candidates[label].get("status") == "available" for label in ("X", "Y"))
        facts = all(row["source_facts_valid"][inverse[arm]] for arm in ("B", "G"))
        no_severe = all(not row["severe_errors"][inverse[arm]] for arm in ("B", "G"))
        guards.append({"case_id": key["case_id"], "pass": valid and facts and no_severe, "structurally_available": valid, "all_frozen_facts_preserved": facts, "no_severe_error": no_severe})
    supplementary = {
        arm: {metric: {"mean": sum(values) / len(values) if values else None, "rated": len(values), "denominator": 12} for metric, values in metrics.items()}
        for arm, metrics in score_values.items()
    }
    clear_groups = sorted({item["group"] for item in clear_wins})
    threshold_criteria = len(clear_wins) >= 3 and len(clear_groups) >= 2 and not severe_introduced and len(guards) == 4 and all(item["pass"] for item in guards)
    real_evidence = provenance["execution_mode"] == "official_provider"
    result = {
        "schema": "meaning-first-e1-final-result-1",
        "assembled_at": now(),
        "ratings_locked_sha256": locked_sha,
        "execution_provenance": provenance,
        "rater": ratings["rater"],
        "partial_blind": True,
        "denominator": 12,
        "meaning_counts_G_relative_to_B": counts,
        "readability_counts_G_relative_to_B": readability,
        "clear_G_wins": {"count": len(clear_wins), "groups": clear_groups, "cases": clear_wins},
        "new_severe_errors_in_G_relative_to_B": severe_introduced,
        "absolute_guards": guards,
        "supplementary_0_to_3": supplementary,
        "progress_threshold": {
            "met": threshold_criteria and real_evidence,
            "criteria_met_from_ratings": threshold_criteria,
            "eligible_as_real_E1_evidence": real_evidence,
            "requires_separate_E2_authorization": True,
        },
        "decoded_cases": decoded,
    }
    result_path = directory / "final_result.json"
    if result_path.exists():
        existing = load_json(result_path)
        comparable_existing = {key: value for key, value in existing.items() if key != "assembled_at"}
        comparable_result = {key: value for key, value in result.items() if key != "assembled_at"}
        if comparable_existing != comparable_result:
            raise RuntimeError("final result exists with different decoded content")
        return existing
    durable_json(result_path, result, exclusive=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("check")
    execute_parser = subparsers.add_parser("execute")
    execute_parser.add_argument("--execution-dir", type=Path, default=DEFAULT_EXECUTION_DIR)
    execute_parser.add_argument("--endpoint", default=OFFICIAL_ENDPOINT)
    execute_parser.add_argument("--timeout", type=float, default=45.0)
    assemble_parser = subparsers.add_parser("assemble")
    assemble_parser.add_argument("ratings", type=Path)
    assemble_parser.add_argument("--execution-dir", type=Path, default=DEFAULT_EXECUTION_DIR)
    args = parser.parse_args()
    if args.command == "check":
        result = {**check_frozen(), "credential_available": credential() is not None}
    elif args.command == "execute":
        result = execute(args.execution_dir, args.endpoint, args.timeout)
    else:
        result = assemble(args.execution_dir, args.ratings)
    print(canonical(result))


if __name__ == "__main__":
    main()
