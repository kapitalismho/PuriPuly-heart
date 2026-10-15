from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from prepare import EXTERNAL, HERE, sanitize, save


def project_app_evidence(data):
    fields = {"translation.model", "translation.connection", "translation.gpu_device_id", "languages", "stt.provider", "peer_stt.provider", "telemetry.enabled", "clipboard.enabled", "chatbox.activation_notice.enabled"}
    for command in data.get("commands", []):
        arguments = command["arguments"]
        output = command["output"]
        if arguments == ["settings", "choices"] and "choices" in output:
            projected = {key: value for key, value in output.items() if key in {"output_version", "instance_id", "revision", "extensions"}}
            projected["choices"] = {key: value for key, value in output["choices"].items() if key in fields}
            for key in ("field_types", "field_schemas"):
                projected[key] = {name: value for name, value in output.get(key, {}).items() if name in fields}
            command["output"] = projected
            command["projection"] = "Focused control choices; unrelated real audio/device/process inventory omitted."
        elif arguments == ["settings", "current"] and "settings" in output:
            intent = output["settings"].get("intent", {})
            projected = {key: value for key, value in output.items() if key in {"output_version", "instance_id", "revision"}}
            projected["observed_intent"] = {key: intent.get(key) for key in ("languages", "telemetry", "clipboard", "osc", "integrated_context")}
            projected["observed_intent"]["translation"] = {key: intent.get("translation", {}).get(key) for key in ("model", "connection", "gpu_device_id", "concurrency_limit")}
            projected["observed_intent"]["stt_provider"] = intent.get("stt", {}).get("provider")
            projected["observed_intent"]["peer_stt_provider"] = intent.get("peer_stt", {}).get("provider")
            command["output"] = projected
            command["projection"] = "Focused observed intent only; full synthetic canonical configuration and redacted secret/prompt fields not published."
    return data


def app_summary(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    result = {"raw": path.name, "label": data["label"], "failure": data.get("failure"), "isolation": data.get("isolation"), "cycles": [], "command_statuses": dict(Counter(str(c["output"].get("status")) for c in data.get("commands", []))), "command_errors": [c for c in data.get("commands", []) if c["exit_code"] != 0], "foreign_cache_control": data.get("foreign_cache_control"), "udp_datagram_count": len(data.get("udp_datagrams", []))}
    for cycle in data.get("cycles", []):
        trace = cycle.get("trace", [])
        translations = [row for row in trace if row["kind"] == "translate_completed"]
        correlated = []
        for item in cycle.get("translations", []):
            receipt = item["receipt"]
            ids = set(receipt.get("child_turn_ids", [])) | {receipt.get("turn_id")}
            events = [event for event in item["events"] if event.get("topic") == "translation" and event.get("utterance_id") in ids and event.get("instance_id") == receipt.get("instance_id")]
            correlated.append({"admission_terminal_applied": receipt.get("terminal") is True and receipt.get("status") == "applied" and receipt.get("admission") == "admitted", "matched_completed_events": len(events), "source": item["source"], "texts": [event.get("text") for event in events]})
        pairs = []
        for row in trace:
            if row["kind"] == "owner_ready":
                readiness = row["readiness"]
                pair = (readiness["source_language"], readiness["target_language"], row["active_slot"])
                if not pairs or pair != pairs[-1]:
                    pairs.append(pair)
        result["cycles"].append({"cycle": cycle["cycle"], "instance_id": cycle["ready"].get("instance_id"), "host_exit_code": cycle.get("host_exit_code"), "host_identities": [r for r in trace if r["kind"] == "host_identity"], "effective_backends": sorted({r["readiness"]["effective_backend"] for r in trace if r["kind"] == "owner_ready"}), "pair_slot_sequence": pairs, "first_translation": translations[0] if translations else None, "completed_translations": len(translations), "admission_event_correlations": correlated, "prefix_prepares": [r for r in trace if r["kind"] == "prepare_prefix"], "restores": [r for r in trace if r["kind"] == "restore_prefix"], "slot_http": [r for r in trace if r["kind"] == "slot_http"], "cache_lookups": [r for r in trace if r["kind"] == "cache_lookup"], "native_exits": [r for r in trace if r["kind"] == "native_exit"], "cleanup": [r for r in trace if r["kind"] in {"transport_closed", "owner_close_observed", "owner_close_failed"}], "network_denied": [r for r in trace if r["kind"] == "network_denied"]})
        result["cycles"][-1]["gpu_counters_after_native_exit"] = cycle.get("gpu_counters_after_native_exit")
        result["cycles"][-1]["resource_snapshots"] = [r for r in trace if r["kind"] in {"native_resources", "native_resource_observation_failed"}]
        result["cycles"][-1]["observed_history_entry_counts"] = sorted({r["context_entries"] for r in translations if "context_entries" in r})
    return result


def main():
    results = HERE / "results"
    for path in results.iterdir():
        if path.suffix in {".txt", ".json", ".jsonl"}:
            text = path.read_text(encoding="utf-8")
            sanitized = sanitize(text)
            if sanitized != text:
                path.write_text(sanitized, encoding="utf-8")
    failed = results / "app-old-cpu-app.json"
    if failed.exists():
        data = json.loads(failed.read_text(encoding="utf-8"))
        root = EXTERNAL / "apps/app-old-cpu"
        if (root / "trace.jsonl").exists():
            data["failure_trace"] = [json.loads(line) for line in sanitize((root / "trace.jsonl").read_text(encoding="utf-8")).splitlines()]
        save(failed.name, data)
        for path in root.glob("native-*.txt"):
            (results / f"app-old-cpu-failed-{path.name}").write_text(sanitize(path.read_text(encoding="utf-8")), encoding="utf-8")
    for path in sorted(results.glob("*-app.json")):
        save(path.name, project_app_evidence(json.loads(path.read_text(encoding="utf-8"))))
    apps = {path.stem: app_summary(path) for path in sorted(results.glob("*-app.json"))}
    save("app-summary.json", apps)
    sampling = {}
    canonical_templates = {}
    for path in sorted(results.glob("*-props.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        template = data.get("chat_template")
        template_hash = hashlib.sha256(template.encode()).hexdigest() if isinstance(template, str) else data.get("chat_template_sha256")
        sampling[path.stem] = {"build_info": data.get("build_info"), "chat_template_sha256": template_hash, "default_generation_settings": data.get("default_generation_settings"), "total_slots": data.get("total_slots"), "modalities": data.get("modalities")}
        if template_hash in canonical_templates:
            data.pop("chat_template", None)
            data["chat_template_sha256"] = template_hash
            data["chat_template_evidence"] = canonical_templates[template_hash]
            save(path.name, data)
        else:
            canonical_templates[template_hash] = path.name
    save("sampling-template-summary.json", sampling)
    compact_path = results / "producer-compaction.json"
    compact_records = json.loads(compact_path.read_text(encoding="utf-8")) if compact_path.exists() else {}
    pattern = re.compile(r"\b(?:error|failed|assert|fatal|NaN|warning)\b|\s[EW]\s|cache state|cache.*(?:evict|limit)|checkpoint|state_seq|n_read|n_written|lazy read enabled|enabling prefetch|offloaded\s+\d+/\d+|buffer size|draft-mtp|system_info|n_ctx\s|n_ctx_seq|n_batch\s|n_ubatch\s", re.IGNORECASE)
    for path in sorted(results.glob("*-producer.txt")) + sorted(results.glob("app-*-native-*.txt")):
        text = path.read_text(encoding="utf-8")
        if text.startswith("COMPACT PRODUCER EVIDENCE"):
            continue
        lines = [line for line in text.splitlines() if line.strip()]
        selected = set(range(min(400, len(lines)))) | set(range(max(0, len(lines) - 50), len(lines)))
        for index, line in enumerate(lines):
            if pattern.search(line):
                selected.add(index)
        statistics = [i for i, line in enumerate(lines) if "statistics" in line and "draft-mtp:" in line]
        if statistics:
            selected.add(statistics[-1])
        compact_records[path.name] = {"normalized_sanitized_original_sha256": hashlib.sha256(text.encode()).hexdigest(), "normalized_original_bytes": len(text.encode()), "nonempty_lines": len(lines), "retained_lines": len(selected), "policy": "first400 + last50 nonempty lines; all failure/warning/cache-state/checkpoint-state/PLE/offload/buffer/MTP evidence. Complete per-request outcomes remain in requests.jsonl or actual-app JSON. Full producer original is retained only in owned external run/app storage; its original line endings/private paths are not the normalized sanitized hash here."}
        header = "COMPACT PRODUCER EVIDENCE\n" + json.dumps(compact_records[path.name], ensure_ascii=False) + "\n\n"
        path.write_text(header + "\n".join(lines[i] for i in sorted(selected)) + "\n", encoding="utf-8")
    save("producer-compaction.json", compact_records)
    index = []
    for path in sorted(results.iterdir()):
        if path.is_file() and path.name != "evidence-index.json":
            raw = path.read_bytes()
            index.append({"path": path.name, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()})
    save("evidence-index.json", {"entries": index, "total_bytes": sum(row["bytes"] for row in index)})
    print(json.dumps({"evidence_files": len(index), "total_bytes": sum(row["bytes"] for row in index), "app_reports": len(apps), "compacted_producers": len(compact_records)}))


if __name__ == "__main__":
    main()
