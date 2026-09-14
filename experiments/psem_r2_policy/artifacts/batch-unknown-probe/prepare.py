from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
SOURCE_DIR = HERE.parent / "unknown-partition-probe"
SOURCE_CASES = SOURCE_DIR / "cases.jsonl"
SOURCE_PLAN = SOURCE_DIR / "plan.json"
CASES_FILE = HERE / "cases.jsonl"
PLAN_FILE = HERE / "plan.json"
RUNNER_FILE = HERE / "run.py"
SOURCE_CASES_SHA256 = "30446bacef1f1a1a28eee111c5cff79d21a1a8b184d87874a7830458f7a704da"
SOURCE_RUNNER_SHA256 = "e044bb3606aaa9ed97284ed2cec20239c85c739b0148445f408bebebeed3d15a"
HELPER_SHA256 = "a8850e7880a45c7ca7675a510efc35b07ac873b41cd085b01b9ff12a283d499c"
REPEAT_IDS = [
    "9a75496a-26b7-42f8-a41b-f6ed81d41dcc",
    "8766ce13-7ef6-4f8b-a09a-86cd0883a603",
    "5c333233-1622-4850-a60e-ca8811613108",
    "f3c5a2b5-fee1-47a8-953b-e1f80e8094a4",
]
NO_OP_IDS = [
    "147e5b68-0232-4e76-a587-8ddfbb2b817a",
    "d928e75b-4caf-4368-a614-b230a84ab69b",
]
SEGMENT_FIELDS = ("group_id", "relation", "text", "token_indexes", "start_source_sample", "end_source_sample")
PROMPT = """# Role: Meaning-preserving Korean batch translator
Translate the complete English parent into natural Korean while preserving every positional segment.

The input is JSON. `parent_text` is the complete parent and `segments` partitions that exact text in order. Read the whole parent before translating each segment. Segment IDs are positional correspondence labels only: they are not person identities, speaker names, or evidence that a segment is pure single-speaker speech.

Return plain JSON only, with no Markdown or code fence. The entire response must have exactly this root shape: {\"translations\":[{\"id\":\"s0\",\"text\":\"...\"}]}. Return exactly one translation object for every input segment ID. IDs may be returned in any order, but each input ID must occur exactly once. Do not merge, omit, duplicate, or copy the whole parent into each item.

Preserve conversational meaning, question/answer relations, agreement, negation, correction, uncertainty, numbers, and claim attribution. Do not invent names, roles, relationships, facts, or speaker identities. Do not invert negation or agreement. Do not rewrite multiple turns as one monologue. Keep incomplete or noisy ASR meaning incomplete when it cannot be recovered from the shared parent. Translate only the supplied source text."""
MODEL = "google/gemma-4-26b-a4b-it"
PROVIDER = {"order": ["wafer", "cloudflare", "deepinfra"], "only": ["wafer", "cloudflare", "deepinfra"], "allow_fallbacks": True}


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


def load_source() -> list[dict[str, Any]]:
    if digest_file(SOURCE_CASES) != SOURCE_CASES_SHA256 or digest_file(SOURCE_DIR / "run.py") != SOURCE_RUNNER_SHA256:
        raise RuntimeError("fixed UNKNOWN probe source identity mismatch")
    rows = [json.loads(line) for line in SOURCE_CASES.read_text(encoding="utf-8").splitlines() if line]
    if len(rows) != 20:
        raise RuntimeError("fixed source must contain exactly 20 parents")
    return rows


def compact_segment(unit: dict[str, Any]) -> dict[str, Any]:
    if not all(field in unit for field in SEGMENT_FIELDS):
        raise RuntimeError("source partition field missing")
    return {field: unit[field] for field in SEGMENT_FIELDS}


def build_cases() -> list[dict[str, Any]]:
    result = []
    for source in load_source():
        result.append({
            "schema": "BATCH-UNKNOWN-PROBE-CASE-1",
            "parent_id": source["parent_id"],
            "parent_index": source["parent_index"],
            "accepted_text": source["accepted_text"],
            "accepted_text_sha256": source["accepted_text_sha256"],
            "meeting": source["meeting"],
            "cluster_id": source["cluster_id"],
            "no_op_guard": source["no_op_guard"],
            "replicates": source["replicates"],
            "repeat_selection": source["repeat_selection"],
            "source_facts_frozen_before_generation": source["source_facts_frozen_before_generation"],
            "source_partitions": {
                "A": [compact_segment(unit) for unit in source["baseline_units"]],
                "B": [compact_segment(unit) for unit in source["intervention_units"]],
            },
        })
    return result


def request_body(parent: dict[str, Any], arm: str) -> dict[str, Any]:
    segments = [{"id": f"s{index}", "text": unit["text"]} for index, unit in enumerate(parent["source_partitions"][arm])]
    payload = {"parent_text": parent["accepted_text"], "segments": segments}
    return {
        "messages": [
            {"role": "system", "content": PROMPT},
            {"role": "user", "content": f"<input>\n{canonical(payload)}\n</input>"},
        ],
        "model": MODEL,
        "temperature": 0.6,
        "reasoning": {"effort": "none"},
        "provider": PROVIDER,
        "max_tokens": 512,
    }


def build_plan(cases_bytes: bytes) -> dict[str, Any]:
    cases = [json.loads(line) for line in cases_bytes.decode("utf-8").splitlines() if line]
    tails = []
    for case in cases:
        tails.append({
            "parent_id": case["parent_id"],
            "segment_counts": {arm: len(case["source_partitions"][arm]) for arm in ("A", "B")},
            "idealized_tail_replacements_two_free_slots": {arm: max(0, len(case["source_partitions"][arm]) - 2) for arm in ("A", "B")},
        })
    return {
        "schema": "BATCH-UNKNOWN-PROBE-PLAN-1",
        "baseline": "49bab295f9801455d2f1925ff73a9ca6aaaba376",
        "status": "prepared_no_paid_calls",
        "question": "Does UNKNOWN cut removal still add meaning value after complete parent context is shared?",
        "interpretation": {"A": "existing PSEM partitions with one #160-shaped parent batch translation", "B": "the same parent batch translation after adopted UNKNOWN-only cut suppression"},
        "limits": {
            "probe_type": "#160-adapted PSEM text probe, not Soniox R4 segmentation or production #160 implementation",
            "selection_disclosure": "The exact cases and repeats were already used in an independent fragment experiment and those results are known. The new matched batch outputs were not inspected for selection. This is neither confirmation nor generalization evidence.",
            "claims_not_made": ["actual UI pacing or caption visibility", "native or ASR behavior", "speaker purity or identity", "production adoption", "HOLDOUT generalization"],
        },
        "authority": {"maximum_http_attempts": 54, "expected_catalog_attempts": 54, "logical_arm_bindings": 56, "max_tokens_per_attempt": 512, "maximum_output_tokens": 27648, "automatic_retries": 0, "paid_calls_during_preparation": 0, "account_or_pricing_queries": False},
        "request": {"model": MODEL, "temperature": 0.6, "reasoning": {"effort": "none"}, "provider": PROVIDER, "max_tokens": 512, "system_prompt": PROMPT, "system_prompt_sha256": digest_bytes(PROMPT.encode("utf-8")), "context": "complete parent_text visible in both arms; only segment partitions differ", "response_root": {"translations": [{"id": "s0", "text": "..."}]}},
        "inputs": {"unknown-partition-probe/cases.jsonl": SOURCE_CASES_SHA256, "unknown-partition-probe/run.py": SOURCE_RUNNER_SHA256, "meaning-first-probe/run_e1.py": HELPER_SHA256},
        "cases_sha256": digest_bytes(cases_bytes),
        "code_sha256": {"prepare.py": digest_file(Path(__file__)), "run.py": digest_file(RUNNER_FILE)},
        "design": {"parents": 20, "changed_parents": 18, "no_op_guards": NO_OP_IDS, "repeat_parent_ids": REPEAT_IDS, "replicate_pairs": 28, "logical_parent_calls_per_arm_binding": 1, "primary_source_units": {"A": 112, "B": 75}, "primary_cluster_distribution": {"EN2009": 11, "ES2009": 6, "ES2002": 1}, "randomization": "fresh opaque X/Y mapping per parent replicate and random unique-request catalog order persisted before dispatch", "sharing": "only an identical canonical request body within the same parent replicate shares one HTTP result", "idealized_r7": {"assumptions": "all results ready, two free caption slots, no external occupants, expiry, protection, or pressure", "formula_per_parent_arm": "max(0, segment_count - 2) one-second replacement admissions", "per_parent": tails, "totals": {arm: sum(row["idealized_tail_replacements_two_free_slots"][arm] for row in tails) for arm in ("A", "B")}}},
        "evaluation": {"pairs": 28, "primary_changed_distinct": 18, "guards": 2, "repeat_pairs": 8, "main_judgment": "meaning improvement/equal/worse/unjudgeable independent of readability; assess incorrect cross-segment mapping or duplication when observable", "exploratory_signal": "descriptive first; at least 3 clear B wins across at least 2 groups versus new severe regressions and repeat worsening is a transparent exploratory signal, not a production veto", "prior_adoption": "existing UNKNOWN adoption is not reset by a zero-regression gate"},
        "execution": {"command": "python -B experiments/psem_r2_policy/artifacts/batch-unknown-probe/run.py execute", "canonical_execution_directory": "experiments/psem_r2_policy/artifacts/batch-unknown-probe/execution", "endpoint": "https://openrouter.ai/api/v1/chat/completions", "failure": "a durable start without terminal is indeterminate and never resubmitted; no retry or format repair call"},
    }


def encoded_cases(cases: list[dict[str, Any]]) -> bytes:
    return ("".join(canonical(case) + "\n" for case in cases)).encode("utf-8")


def validate(cases: list[dict[str, Any]], plan: dict[str, Any]) -> dict[str, Any]:
    source = load_source()
    if len(cases) != 20 or [case["parent_id"] for case in cases] != [row["parent_id"] for row in source]:
        raise RuntimeError("case coverage/order mismatch")
    if [case["replicates"] for case in cases].count(3) != 4 or [case["parent_id"] for case in cases if case["replicates"] == 3] != REPEAT_IDS:
        raise RuntimeError("repeat coverage mismatch")
    if [case["parent_id"] for case in cases if case["no_op_guard"]] != NO_OP_IDS:
        raise RuntimeError("no-op guard coverage mismatch")
    logical = unique = 0
    for case, original in zip(cases, source):
        if digest_bytes(case["accepted_text"].encode("utf-8")) != case["accepted_text_sha256"] or case["accepted_text"] != original["accepted_text"]:
            raise RuntimeError("accepted text identity mismatch")
        for arm, source_name in (("A", "baseline_units"), ("B", "intervention_units")):
            expected = [compact_segment(unit) for unit in original[source_name]]
            if case["source_partitions"][arm] != expected or "".join(unit["text"] for unit in expected) != case["accepted_text"]:
                raise RuntimeError("source partition text conservation mismatch")
        for replicate in range(case["replicates"]):
            bodies = [canonical(request_body(case, arm)) for arm in ("A", "B")]
            logical += 2
            unique += len(set(bodies))
    encoded = encoded_cases(cases)
    if logical != 56 or unique != 54:
        raise RuntimeError("catalog cardinality mismatch")
    if plan.get("cases_sha256") != digest_bytes(encoded) or plan.get("code_sha256") != {"prepare.py": digest_file(Path(__file__)), "run.py": digest_file(RUNNER_FILE)}:
        raise RuntimeError("frozen code/case identity mismatch")
    if plan.get("request", {}).get("system_prompt_sha256") != digest_bytes(PROMPT.encode("utf-8")):
        raise RuntimeError("prompt identity mismatch")
    return {"ok": True, "parents": 20, "changed_parents": 18, "no_op_guards": 2, "replicate_pairs": 28, "logical_arm_bindings": logical, "catalog_attempts": unique, "maximum_output_tokens": unique * 512, "cases_sha256": digest_bytes(encoded), "code_sha256": plan["code_sha256"], "source_cases_sha256": SOURCE_CASES_SHA256}


def prepare() -> dict[str, Any]:
    cases = build_cases()
    data = encoded_cases(cases)
    CASES_FILE.write_bytes(data)
    plan = build_plan(data)
    PLAN_FILE.write_bytes((json.dumps(plan, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    return validate(cases, plan)


def check() -> dict[str, Any]:
    cases = [json.loads(line) for line in CASES_FILE.read_text(encoding="utf-8").splitlines() if line]
    return validate(cases, json.loads(PLAN_FILE.read_text(encoding="utf-8")))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "check"))
    args = parser.parse_args()
    print(canonical(prepare() if args.command == "prepare" else check()))


if __name__ == "__main__":
    main()
