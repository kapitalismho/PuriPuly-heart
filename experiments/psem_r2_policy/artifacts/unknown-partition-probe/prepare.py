from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.abc
import importlib.util
import json
import sys
import tarfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
POLICY = ROOT / "experiments/psem_r2_policy"
RETAINED = POLICY / "artifacts/retained"
POLICY_BUNDLE = RETAINED / "historical_policy_inputs.jsonl.gz"
RUNTIME_ARCHIVE = POLICY / "audio-runtime-af26d1d3.tar.gz"
OWNERSHIP_OVERRIDE = POLICY / "runtime_overrides/pretranslation_ownership.py"
SORTFORMER = POLICY / "sortformer_live.py"
METRICS = POLICY / "metrics.py"
MEANING_PREPARE = POLICY / "artifacts/meaning-first-probe/prepare.py"
CASES_FILE = HERE / "cases.jsonl"
PLAN_FILE = HERE / "plan.json"
VERIFICATION_FILE = HERE / "verification.json"
BASELINE = "f0a650d17cf7463097b24b01d9d03e4223c83b78"
MODEL = "google/gemma-4-26b-a4b-it"
PROMPT_SHA256 = "21415c9498366a564b4fba7a47bef685c47dfebf7ec2faa9ea59f7d0bcaa9d66"
EXPECTED_INPUTS = {
    "historical_policy_inputs.jsonl.gz": "74d17270917aa96db28be517608faf468af051021f9fde50fd1fb44188eef006",
    "audio-runtime-af26d1d3.tar.gz": "819465e74b847e0a46c0e0968b52d76fccc1a95fa146716411d945690e164416",
    "pretranslation_ownership.py": "650d26b7d1d5edafff80b76f7925d9e7ef547aea5b06402d18087e158ccea772",
    "sortformer_live.py": "eef709441a660261d09f925eee55245cb4a5543e59d6e5a17fe7699f70d74dbc",
    "metrics.py": "5315389d62f8a4f3a58fc3038e743f7f50e29a2c715391ef59cb9edfcad5b031",
    "historical_body_source.py": "1802817a406da298832710db680ae7d8e91838e3cd98f242365c0917b002a905",
}
SIX_FIELDS = ("group_id", "relation", "text", "token_indexes", "start_source_sample", "end_source_sample")
NO_OP_GUARDS = {
    "147e5b68-0232-4e76-a587-8ddfbb2b817a",
    "d928e75b-4caf-4368-a614-b230a84ab69b",
}
WITNESS = "9a75496a-26b7-42f8-a41b-f6ed81d41dcc"
EXPECTED_REPEAT_IDS = [
    WITNESS,
    "8766ce13-7ef6-4f8b-a09a-86cd0883a603",
    "5c333233-1622-4850-a60e-ca8811613108",
    "f3c5a2b5-fee1-47a8-953b-e1f80e8094a4",
]


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_path_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class ArchiveFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def __init__(self, sources: dict[str, tuple[bytes, bool, str]]) -> None:
        self.sources = sources

    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> Any:
        entry = self.sources.get(fullname)
        if entry is None:
            return None
        return importlib.util.spec_from_loader(fullname, self, is_package=entry[1])

    def create_module(self, spec: Any) -> Any:
        return None

    def exec_module(self, module: Any) -> None:
        source, is_package, origin = self.sources[module.__name__]
        module.__file__ = origin
        if is_package:
            module.__path__ = [origin.rsplit("/", 1)[0]]
        exec(compile(source, origin, "exec"), module.__dict__)


def install_archive_runtime() -> ArchiveFinder:
    if digest_file(RUNTIME_ARCHIVE) != EXPECTED_INPUTS["audio-runtime-af26d1d3.tar.gz"]:
        raise RuntimeError("runtime archive hash mismatch")
    sources: dict[str, tuple[bytes, bool, str]] = {}
    with tarfile.open(RUNTIME_ARCHIVE, "r:gz") as archive:
        for member in archive.getmembers():
            if not member.isfile() or not member.name.startswith("src/puripuly_heart/") or not member.name.endswith(".py"):
                continue
            handle = archive.extractfile(member)
            if handle is None:
                raise RuntimeError(f"cannot read archive member {member.name}")
            relative = member.name[len("src/") :]
            is_package = relative.endswith("/__init__.py")
            module_name = relative[:-12].replace("/", ".") if is_package else relative[:-3].replace("/", ".")
            sources[module_name] = (handle.read(), is_package, f"{RUNTIME_ARCHIVE}!{member.name}")
    override = OWNERSHIP_OVERRIDE.read_bytes()
    if digest_bytes(override) != EXPECTED_INPUTS["pretranslation_ownership.py"]:
        raise RuntimeError("ownership override hash mismatch")
    module_name = "puripuly_heart.core.audio.pretranslation_ownership"
    sources[module_name] = (override, False, str(OWNERSHIP_OVERRIDE))
    finder = ArchiveFinder(sources)
    sys.meta_path.insert(0, finder)
    return finder


def iter_bundle():
    with gzip.open(POLICY_BUNDLE, "rt", encoding="utf-8") as source:
        for line in source:
            yield json.loads(line)


def native_row(label: int | str) -> list[float]:
    if isinstance(label, int):
        row = [0.0] * 4
        row[label] = 1.0
        return row
    if label == "OVERLAP":
        return [1.0, 1.0, 0.0, 0.0]
    if label == "NONE":
        return [0.0] * 4
    raise RuntimeError(f"unknown native label {label!r}")


def replay_chunks(rows: list[dict[str, Any]], slive: Any) -> Any:
    decoder = slive.LiveTransitionDecoder()
    index = 0
    while index < len(rows):
        first = rows[index]
        key = (int(first["emit_start_frame"]), float(first["available_at_monotonic_s"]), str(first.get("receipt_kind") or "native_arrival"))
        batch = [native_row(first["label"])]
        index += 1
        while index < len(rows):
            item = rows[index]
            candidate = (int(item["emit_start_frame"]), float(item["available_at_monotonic_s"]), str(item.get("receipt_kind") or "native_arrival"))
            if candidate != key:
                break
            batch.append(native_row(item["label"]))
            index += 1
        decoder.ingest_chunk(key[0], batch, available_at_monotonic_s=key[1], receipt_kind=key[2])
    return decoder


def comparable(units: Any) -> list[dict[str, Any]]:
    rows = [{field: getattr(unit, field) for field in SIX_FIELDS} for unit in units]
    for row in rows:
        row["token_indexes"] = list(row["token_indexes"])
    return rows


def recorded_comparable(units: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{field: row.get(field) for field in SIX_FIELDS} for row in units]


def token_objects(rows: list[dict[str, Any]], token_class: Any) -> tuple[Any, ...]:
    return tuple(token_class(
        text=row["text"], language="en", start_ms=row.get("start_ms"), end_ms=row.get("end_ms"),
        timing=row.get("timing"), source_start_sample=row.get("source_start_sample"),
        source_end_sample=row.get("source_end_sample"), provenance=row.get("provenance"),
    ) for row in rows)


def replay_inputs(decoder: Any, slive: Any, po: Any, generation: object) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    events = tuple(slive.hypothesis_from_live_event(
        event, capture_epoch=1, producer_generation=generation, reference_generation=generation,
    ) for event in decoder.events)
    evidence = tuple(po.PretranslationEvidence(
        capture_epoch=1, start_sample=item.start_sample, end_sample=item.end_sample,
        available_at_monotonic_s=item.available_at_monotonic_s, relation=item.relation,
        producer_generation=generation, reference_generation=generation, reference_valid=True,
    ) for item in decoder.evidence)
    return events, evidence


def effective_labels(tokens: tuple[Any, ...], events: tuple[Any, ...], evidence: tuple[Any, ...], admission: float, po: Any) -> tuple[list[tuple[str, str, str | None]], list[Any], tuple[Any, ...]]:
    applicable = po._applicable_hypotheses(events, admitted_at_monotonic_s=admission, capture_epoch=1)
    if len(tokens) > 1 and not po._partition_is_requested(tokens, applicable):
        return [], applicable, evidence
    if tokens and po._partition_is_requested(tokens, applicable):
        selected = po._partition_coverage_generation(tokens, applicable, evidence, admitted_at_monotonic_s=admission, capture_epoch=1)
        if selected is None:
            return [], applicable, evidence
        applicable = [item for item in applicable if po._same_generation(item.producer_generation, item.reference_generation, selected[0], selected[1])]
        evidence = tuple(item for item in evidence if po._same_generation(item.producer_generation, item.reference_generation, selected[0], selected[1]))
    labels = []
    for token in tokens:
        start, end, uncertain = po._token_source_interval(token)
        if uncertain is not None:
            labels.append(("UNKNOWN", f"u:{uncertain}", uncertain))
            continue
        assert start is not None and end is not None
        if any(start < event.estimated_transition_sample < end for event in applicable):
            labels.append(("UNKNOWN", "u:straddle", "straddle"))
            continue
        relation, reason = po._relation_from_evidence(start, end, evidence, admitted_at_monotonic_s=admission, capture_epoch=1)
        labels.append((relation, f"s:{po._transition_segment(end, applicable)}", reason))
    return labels, applicable, evidence


def intervention_units(tokens: tuple[Any, ...], labels: list[tuple[str, str, str | None]], po: Any) -> tuple[tuple[Any, ...], list[dict[str, Any]], list[int]]:
    if not labels:
        units = po._whole_parent_unit(tokens) if tokens else ()
        audit = [[{"token_index": index, "claim": "UNKNOWN", "key": "whole-parent", "uncertainty": "whole_parent_gate"} for index in range(len(tokens))]] if tokens else []
        return units, [{"unit_index": index, "constituents": rows} for index, rows in enumerate(audit)], []
    groups = [[0]]
    removed: list[int] = []
    for index in range(1, len(labels)):
        left = labels[index - 1]
        right = labels[index]
        same_baseline = left[:2] == right[:2]
        suppress_unknown_only = left[1] == right[1] and left[0] != right[0] and "UNKNOWN" in (left[0], right[0])
        if same_baseline or suppress_unknown_only:
            groups[-1].append(index)
            if suppress_unknown_only:
                removed.append(index)
        else:
            groups.append([index])
    units = []
    audits = []
    for unit_index, indexes in enumerate(groups):
        relation = "UNKNOWN" if any(labels[index][0] == "UNKNOWN" for index in indexes) else labels[indexes[0]][0]
        units.append(po._unit_from_run(tokens, indexes, relation, unit_index))
        audits.append({
            "unit_index": unit_index,
            "constituents": [{"token_index": index, "claim": labels[index][0], "key": labels[index][1], "uncertainty": labels[index][2]} for index in indexes],
        })
    return tuple(units), audits, removed


def unit_record(unit: Any, audit: dict[str, Any] | None = None) -> dict[str, Any]:
    value = {field: getattr(unit, field) for field in SIX_FIELDS}
    value["token_indexes"] = list(value["token_indexes"])
    if audit is not None:
        value["experimental_constituent_evidence"] = audit["constituents"]
    return value


def source_facts(parent: dict[str, Any]) -> dict[str, Any]:
    text = parent["text"]
    lowered = text.lower()
    markers = [marker for marker in ("not", "no", "n't", "yes", "yeah", "right", "agree", "disagree") if marker in lowered]
    numeric = [token["text"] for token in parent["tokens"] if any(char.isdigit() for char in token["text"]) or any(word in token["text"].lower().split() for word in ("one", "two", "three", "four", "five", "ten", "nineteen", "thirty", "fifty"))]
    return {
        "accepted_text": text,
        "question_mark_count": text.count("?"),
        "negation_agreement_markers_present": markers,
        "number_bearing_token_texts": numeric,
        "claim_attribution_annotations": parent["r0"]["attribution"],
    }


def historical_template_and_builder(parents: list[tuple[str, dict[str, Any]]]) -> tuple[dict[str, Any], Any]:
    if digest_file(MEANING_PREPARE) != EXPECTED_INPUTS["historical_body_source.py"]:
        raise RuntimeError("historical body helper hash mismatch")
    helper = load_path_module("unknown_partition_historical_body", MEANING_PREPARE)
    prompts = {request["system_prompt"] for _, parent in parents for request in parent.get("requests", [])}
    if len(prompts) != 1:
        raise RuntimeError("historical system prompt is not unique")
    prompt = next(iter(prompts))
    if len(prompt) != 3354 or digest_bytes(prompt.encode("utf-8")) != PROMPT_SHA256:
        raise RuntimeError("historical system prompt identity mismatch")
    return {"system_prompt": prompt}, helper.historical_body


def prepare_data() -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    actual_inputs = {
        "historical_policy_inputs.jsonl.gz": digest_file(POLICY_BUNDLE),
        "audio-runtime-af26d1d3.tar.gz": digest_file(RUNTIME_ARCHIVE),
        "pretranslation_ownership.py": digest_file(OWNERSHIP_OVERRIDE),
        "sortformer_live.py": digest_file(SORTFORMER),
        "metrics.py": digest_file(METRICS),
        "historical_body_source.py": digest_file(MEANING_PREPARE),
    }
    if actual_inputs != EXPECTED_INPUTS:
        raise RuntimeError(f"frozen input hash mismatch: {actual_inputs}")
    parents: list[tuple[str, dict[str, Any]]] = []
    chunks: dict[str, list[dict[str, Any]]] = defaultdict(list)
    header = None
    for row in iter_bundle():
        if row.get("type") == "header":
            header = row
        elif row.get("cohort") == "current" and row.get("type") == "parent":
            parents.append((row["meeting"], row["value"]))
        elif row.get("cohort") == "current" and row.get("type") == "native_chunk":
            chunks[row["meeting"]].append(row["value"])
    if len(parents) != 2482 or header is None:
        raise RuntimeError("retained current cohort census mismatch")
    template, historical_body = historical_template_and_builder(parents)
    finder = install_archive_runtime()
    try:
        slive = load_path_module("unknown_partition_sortformer_live", SORTFORMER)
        import puripuly_heart.core.audio.pretranslation_ownership as po
        from puripuly_heart.core.stt.backend import STTTimedToken
        generation = object()
        meeting_inputs = {meeting: replay_inputs(replay_chunks(rows, slive), slive, po, generation) for meeting, rows in chunks.items()}
        records = []
        replay_parents = replay_units = intervention_total = 0
        split_rows = []
        removed_total = 0
        for meeting, parent in parents:
            if not parent.get("text"):
                continue
            tokens = token_objects(parent["tokens"], STTTimedToken)
            events, all_evidence = meeting_inputs[meeting]
            admission = float(parent["marks"]["translation_admission"])
            evidence = tuple(item for item in all_evidence if item.available_at_monotonic_s <= admission)[-4096:]
            units, late, reasons = po.assign_ownership_units(tokens, events, admitted_at_monotonic_s=admission, capture_epoch=1, evidence=evidence)
            if comparable(units) != recorded_comparable(parent["units"]):
                raise RuntimeError(f"exact replay mismatch: {parent['parent_id']} actual={comparable(units)!r} recorded={recorded_comparable(parent['units'])!r}")
            if "".join(unit.text for unit in units) != parent["text"] or "".join(token.text for token in tokens) != parent["text"]:
                raise RuntimeError(f"source conservation failure: {parent['parent_id']}")
            replay_parents += 1
            replay_units += len(units)
            labels, applicable, effective_evidence = effective_labels(tokens, events, evidence, admission, po)
            candidate, audits, removed = intervention_units(tokens, labels, po)
            if "".join(unit.text for unit in candidate) != parent["text"] or [index for unit in candidate for index in unit.token_indexes] != list(range(len(tokens))):
                raise RuntimeError(f"intervention conservation failure: {parent['parent_id']}")
            intervention_total += len(candidate)
            removed_total += len(removed)
            if len(units) <= 1:
                if comparable(candidate) != comparable(units):
                    raise RuntimeError(f"non-split parent changed: {parent['parent_id']}")
                continue
            boundary_labels = []
            for index in range(1, len(labels)):
                boundary_labels.append({
                    "right_token_index": index,
                    "left_claim": labels[index - 1][0], "right_claim": labels[index][0],
                    "left_key": labels[index - 1][1], "right_key": labels[index][1],
                    "baseline_boundary": labels[index - 1][:2] != labels[index][:2],
                    "removed_by_intervention": index in removed,
                })
            case = {
                "schema": "UNKNOWN-PARTITION-PROBE-CASE-1",
                "meeting": meeting,
                "cluster_id": parent["cluster_id"],
                "parent_index": parent["index"],
                "parent_id": parent["parent_id"],
                "source_span_samples": parent["span"],
                "accepted_text": parent["text"],
                "accepted_text_sha256": digest_bytes(parent["text"].encode("utf-8")),
                "tokens": parent["tokens"],
                "source_facts_frozen_before_generation": source_facts(parent),
                "transition_keys_frozen_before_generation": [{"hypothesis_id": item.hypothesis_id, "estimated_transition_sample": item.estimated_transition_sample, "available_at_monotonic_s": item.available_at_monotonic_s} for item in applicable],
                "boundary_audit": boundary_labels,
                "baseline_units": [unit_record(unit) for unit in units],
                "intervention_units": [unit_record(unit, audit) for unit, audit in zip(candidate, audits)],
                "removed_boundary_count": len(removed),
                "no_op_guard": parent["parent_id"] in NO_OP_GUARDS,
                "reconstructed_generation_identity_limit": "One synthetic producer/reference generation identity is used for this replay because literal historical object identity was not retained.",
                "late_ignored": list(late),
                "unknown_reasons": list(reasons),
            }
            split_rows.append(case)
        if replay_parents != 2367 or replay_units != 2459 or intervention_total != 2422 or removed_total != 37:
            raise RuntimeError(f"DEV census mismatch: {replay_parents}, {replay_units}, {intervention_total}, {removed_total}")
        if len(split_rows) != 20 or sum(row["removed_boundary_count"] > 0 for row in split_rows) != 18:
            raise RuntimeError("split cohort mismatch")
        if {row["parent_id"] for row in split_rows if row["removed_boundary_count"] == 0} != NO_OP_GUARDS:
            raise RuntimeError("no-op guard identity mismatch")
        cluster_maxima = {}
        for cluster in ("ES2009", "ES2002", "EN2009"):
            eligible = [row for row in split_rows if row["cluster_id"] == cluster]
            maximum = max(row["removed_boundary_count"] for row in eligible)
            cluster_maxima[cluster] = min(row["parent_id"] for row in eligible if row["removed_boundary_count"] == maximum)
        repeat_ids = [WITNESS, cluster_maxima["ES2009"], cluster_maxima["ES2002"], cluster_maxima["EN2009"]]
        if repeat_ids != EXPECTED_REPEAT_IDS:
            raise RuntimeError(f"repeat selection mismatch: {repeat_ids}")
        for case in split_rows:
            case["replicates"] = 3 if case["parent_id"] in repeat_ids else 1
            case["repeat_selection"] = next((reason for reason, pid in [("previously_seen_35_witness", WITNESS), ("maximum_removed_ES2009", cluster_maxima["ES2009"]), ("maximum_removed_ES2002", cluster_maxima["ES2002"]), ("maximum_removed_EN2009", cluster_maxima["EN2009"])] if pid == case["parent_id"]), None)
            for arm_name in ("baseline_units", "intervention_units"):
                body_occurrences: Counter[str] = Counter()
                for occurrence, unit in enumerate(case[arm_name]):
                    request = {**template, "text": unit["text"]}
                    body = historical_body(request)
                    fingerprint = digest_bytes(canonical(body).encode("utf-8"))
                    unit["request_occurrence"] = occurrence
                    unit["body_occurrence_rank"] = body_occurrences[fingerprint]
                    body_occurrences[fingerprint] += 1
                    unit["canonical_body_sha256"] = fingerprint
            records.append(case)
    finally:
        sys.meta_path.remove(finder)
    primary_bindings = sum(len(case[arm]) for case in records for arm in ("baseline_units", "intervention_units"))
    catalog_keys = set()
    for case in records:
        for replicate in range(case["replicates"]):
            for arm in ("baseline_units", "intervention_units"):
                for unit in case[arm]:
                    catalog_keys.add((case["parent_id"], replicate, unit["canonical_body_sha256"], unit["body_occurrence_rank"]))
    if primary_bindings != 187 or len(catalog_keys) != 221:
        raise RuntimeError(f"request census mismatch: bindings={primary_bindings}, catalog={len(catalog_keys)}")
    cases_payload = "".join(canonical(record) + "\n" for record in records)
    cases_sha = digest_bytes(cases_payload.encode("utf-8"))
    plan = {
        "schema": "UNKNOWN-PARTITION-PROBE-PLAN-1",
        "baseline": BASELINE,
        "status": "prepared_no_paid_calls",
        "cases_sha256": cases_sha,
        "authority": {
            "source": "current user instruction for this finite experiment; no old ledger headroom is used",
            "maximum_http_attempts": 224,
            "expected_catalog_attempts": 221,
            "maximum_output_tokens": 22400,
            "max_tokens_per_attempt": 100,
            "attempts_per_instance": 1,
            "automatic_retries": 0,
            "account_or_pricing_queries": False,
            "old_budget_ledger_mutation": False,
            "native_or_asr_calls": False,
            "holdout_access": False,
            "production_or_remote_mutation": False,
        },
        "inputs": actual_inputs,
        "historical_request": {
            "system_prompt_characters": 3354,
            "system_prompt_sha256": PROMPT_SHA256,
            "system_prompt": template["system_prompt"],
            "body_builder": "unchanged meaning-first-probe/prepare.py historical_body(request)",
            "model": MODEL,
            "temperature": 0.6,
            "reasoning": {"effort": "none"},
            "provider": {"order": ["wafer", "cloudflare", "deepinfra"], "only": ["wafer", "cloudflare", "deepinfra"], "allow_fallbacks": True},
            "max_tokens": 100,
            "context": "",
            "user_payload": "child text only wrapped in <input> tags",
            "output_contract": "A successful response is nonempty plain text. JSON is neither requested nor required. finish_reason=length is failure.",
        },
        "replay": {
            "nonempty_parents": 2367, "baseline_units": 2459, "intervention_units": 2422,
            "removed_boundaries": 37, "new_boundaries": 0, "split_parents": 20,
            "changed_parents": 18, "no_op_guards": sorted(NO_OP_GUARDS),
            "generation_identity_limit": "Literal historical producer/reference generation objects were not retained; one synthetic shared identity is reconstructed per preparation run. Exact six-field equality is nevertheless required for every emitted baseline unit.",
        },
        "selection": {
            "all_actual_split_parents": 20,
            "primary_independent_parents": 18,
            "guards_reported_separately": 2,
            "repeat_parent_ids": repeat_ids,
            "repeat_rule": "Three total replicates for the known 35 witness and the lexicographically first maximum-removed parent in each independent group; repeats are not independent parents.",
            "witness_disclosure": "9a75496a was previously seen; selection is decision-sensitive and source-based.",
            "selection_blindness": "Sources, accepted text, tokens, GT attribution, and selection were frozen without inspecting any new output or ranking historical translation quality.",
        },
        "execution": {
            "request_counts": {
                "primary_baseline_bindings": 112,
                "primary_intervention_bindings": 75,
                "primary_shared_body_instances": 50,
                "primary_intervention_only_body_instances": 25,
                "primary_unique_http_attempts": 137,
                "additional_repeat_http_attempts": 84,
                "total_http_attempts": 221,
                "total_logical_unit_bindings_including_repeats": 291,
            },
            "command": "python experiments/psem_r2_policy/artifacts/unknown-partition-probe/run.py execute",
            "default_endpoint": "https://openrouter.ai/api/v1/chat/completions",
            "canonical_execution_directory": "experiments/psem_r2_policy/artifacts/unknown-partition-probe/execution",
            "randomization": "Persist one cryptographically randomized finite catalog order before dispatch.",
            "sharing": "Within the same parent and replicate, matching body fingerprints and repeated-body occurrence ranks share one result across arms. Repeated occurrences remain separate; different replicates never share.",
            "failure": "Started without terminal is indeterminate and never resubmitted; failures and truncation are never promoted to translation text.",
        },
        "evaluation": {
            "primary_denominator": "18 independent changed DEV parents out of 2367 nonempty DEV parents",
            "repeat_reporting": "Four repeat parents, three replicates each, reported as stability evidence rather than extra independent samples.",
            "guards": "The two unchanged split guards are reported separately and cannot count as improvements because both arms share requests.",
            "main_judgment": "Observable meaning: numbers, questions, negation, agreement, and claim attribution. Split count, purity, and readability alone are not semantic improvement.",
            "ratings": ["improved", "equal", "worse", "unjudgeable"],
            "readability": "Optional and separate.",
            "failure_rule": "Unavailable output makes the pair unjudgeable; empty failure is never scored as zero quality.",
            "exploratory_support": "At least three clear improved parents across at least two groups, no new severe regression, repeat cases show no worsening, and any claimed stable win improves in at least two of three replicates. Weaker isolated wins are limited-case evidence.",
            "limits": "Underpowered exploratory DEV evidence; not production acceptance.",
        },
    }
    verification = {
        "schema": "UNKNOWN-PARTITION-PROBE-VERIFICATION-1",
        "commands": [
            "python -B experiments/psem_r2_policy/artifacts/unknown-partition-probe/prepare.py all",
            "python -B experiments/psem_r2_policy/artifacts/unknown-partition-probe/run.py check",
        ],
        "result": {
            "ok": True, "nonempty_parents": 2367, "baseline_units": 2459,
            "intervention_units": 2422, "removed_boundaries": 37, "split_parents": 20,
            "changed_parents": 18, "no_op_guards": 2, "catalog_attempts": 221,
            "maximum_output_tokens": 22100,
        },
        "input_sha256": actual_inputs,
        "cases_sha256": cases_sha,
        "focused_invariants": [
            "exact six-field baseline equality for all 2459 units",
            "full text and token occurrence conservation",
            "UNKNOWN-only same-key boundary suppression",
            "direct CURRENT/OTHER same-key boundary retained",
            "straddle and uncertainty keys preserved without imputation",
            "known witness merges and both no-op guards remain unchanged",
        ],
        "runner_rehearsal": {
            "command": "OMP Python Eval named Reverify final loopback runner: isolated ThreadingHTTPServer calling run.execute after an injected KeyboardInterrupt, then calling run.execute again; temporary execution directory removed.",
            "result": {
                "catalog_instances": 221,
                "crash_before_terminal_observed": True,
                "indeterminate_not_resubmitted": 1,
                "network_requests": 220,
                "terminal_events": 220,
                "outcomes": {"success": 217, "http_error": 1, "finish_reason_length": 1, "response_body_truncated": 1},
                "rerun_network_requests": 0,
                "rerun_duplicate_attempts": 0,
                "lock_contention_blocked": True,
                "cases_hash_tamper_blocked": True,
                "catalog_order_tamper_blocked": True,
                "blind_packet_case_replicates": 28,
            },
            "canonical_execution_written": False,
        },
    }
    return plan, records, verification


def write_outputs(plan: dict[str, Any], records: list[dict[str, Any]], verification: dict[str, Any]) -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    cases_payload = "".join(canonical(record) + "\n" for record in records)
    CASES_FILE.write_bytes(cases_payload.encode("utf-8"))
    PLAN_FILE.write_bytes((json.dumps(plan, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    VERIFICATION_FILE.write_bytes((json.dumps(verification, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))


def verify_outputs(expected: tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]) -> dict[str, Any]:
    plan, records, verification = expected
    actual_plan = json.loads(PLAN_FILE.read_text(encoding="utf-8"))
    actual_records = [json.loads(line) for line in CASES_FILE.read_text(encoding="utf-8").splitlines() if line]
    actual_verification = json.loads(VERIFICATION_FILE.read_text(encoding="utf-8"))
    if (actual_plan, actual_records, actual_verification) != (plan, records, verification):
        raise RuntimeError("prepared artifacts differ from deterministic replay")
    witness = next(row for row in records if row["parent_id"] == WITNESS)
    if witness["removed_boundary_count"] != 1 or len(witness["baseline_units"]) != 4 or len(witness["intervention_units"]) != 3:
        raise RuntimeError("35 witness intervention mismatch")
    for guard_id in NO_OP_GUARDS:
        guard = next(row for row in records if row["parent_id"] == guard_id)
        if [{field: unit[field] for field in SIX_FIELDS} for unit in guard["baseline_units"]] != [{field: unit[field] for field in SIX_FIELDS} for unit in guard["intervention_units"]]:
            raise RuntimeError(f"no-op guard changed: {guard_id}")
    hypothetical = [("CURRENT", "s:0", None), ("OTHER", "s:0", None)]
    class Token:
        def __init__(self, text: str) -> None:
            self.text = text
            self.language = "en"
            self.source_start_sample = None
            self.source_end_sample = None
    class Unit:
        pass
    class FakePolicy:
        @staticmethod
        def _unit_from_run(tokens: Any, indexes: list[int], relation: str, unit_index: int) -> Any:
            unit = Unit()
            unit.group_id = f"{relation}-{unit_index}"; unit.relation = relation
            unit.text = "".join(tokens[index].text for index in indexes); unit.token_indexes = tuple(indexes)
            unit.start_source_sample = None; unit.end_source_sample = None
            return unit
    kept, _, removed = intervention_units((Token("A"), Token("B")), hypothetical, FakePolicy)
    if len(kept) != 2 or removed:
        raise RuntimeError("hypothetical same-key CURRENT/OTHER boundary was removed")
    for case in records:
        for unit in case["intervention_units"]:
            claims = [item["claim"] for item in unit["experimental_constituent_evidence"]]
            if "UNKNOWN" in claims and unit["relation"] != "UNKNOWN":
                raise RuntimeError("UNKNOWN constituent promoted to concrete relation")
    return {**verification["result"], "cases_sha256": plan["cases_sha256"], "input_sha256": plan["inputs"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "verify", "all"), nargs="?", default="all")
    args = parser.parse_args()
    expected = prepare_data()
    if args.command in {"prepare", "all"}:
        write_outputs(*expected)
    if args.command in {"verify", "all"}:
        print(canonical(verify_outputs(expected)))


if __name__ == "__main__":
    main()
