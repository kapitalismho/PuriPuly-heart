from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

from summarize import distribution

PHASES = ("first_prefix", "hot_same_pair", "pair_alternation", "three_slots")
FIELDS = ("prefill_ms", "prefill_tps", "decode_ms", "decode_tps", "prompt_evaluated_tokens", "prompt_cached_tokens", "prompt_total_tokens", "predicted_output_tokens", "decode_eval_tokens")


def native_records(path):
    if not path.exists():
        return []
    result, pending = [], None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        prompt = re.search(r"prompt eval time =\s*([\d.]+) ms /\s*(\d+) tokens.*?([\d.]+) tokens per second", line)
        generation = re.search(r"(?<!prompt )eval time =\s*([\d.]+) ms /\s*(\d+) tokens.*?([\d.]+) tokens per second", line)
        if prompt:
            pending = {"prefill_ms": float(prompt[1]), "prompt_evaluated_tokens": int(prompt[2]), "prefill_tps": float(prompt[3])}
        elif generation and pending is not None:
            pending.update(decode_ms=float(generation[1]), decode_eval_tokens=int(generation[2]), decode_tps=float(generation[3]))
            result.append(pending)
            pending = None
    return result


def stats(records):
    return {field: distribution([row[field] for row in records if row.get(field) is not None]) for field in FIELDS}


def cell(group, field, tail=False):
    value = group.get(field, {})
    if not value.get("n"):
        return "N/A"
    return f"{value['median']:.2f} / {value['p95']:.2f}" if tail else f"{value['median']:.2f}"


def table(rows):
    lines = ["| Run | n | prefill_ms med / p95 | prefill_tps med | decode_ms med / p95 | decode_tps med | P eval / C hit / O output / D printed med |", "| --- | ---: | ---: | ---: | ---: | ---: | --- |"]
    for label, group in rows:
        lines.append(f"| {label} | {group['prefill_ms']['n']} | {cell(group,'prefill_ms',True)} | {cell(group,'prefill_tps')} | {cell(group,'decode_ms',True)} | {cell(group,'decode_tps')} | {' / '.join(cell(group,k) for k in ('prompt_evaluated_tokens','prompt_cached_tokens','predicted_output_tokens','decode_eval_tokens'))} |")
    return lines


def recover_managed(log_path, record):
    attempts, requests = {}, {}
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines() if log_path.exists() else []
    for line in lines:
        pid = re.search(r"\bpid=(\d+)", line)
        if pid is None or "[Diagnostic][LlmLatency]" not in line:
            continue
        fields = dict(re.findall(r"(\w+)=([^\s]+)", line.split("[Diagnostic][LlmLatency]", 1)[1]))
        key = (pid[1], fields.get("request_id"))
        if " attempt_end " in line and fields.get("provider") == "managed_gemma" and fields.get("status") == "success":
            attempts[key] = fields
        elif " request_end " in line and fields.get("status") == "success":
            requests[key] = fields
    by_turn = {fields["utterance_id"]: (key, attempts[key]) for key, fields in requests.items() if key in attempts and "utterance_id" in fields}
    recovered = []
    for translation in record.get("translations", []):
        turn = translation["event"].get("utterance_id")
        if turn not in by_turn:
            continue
        key, fields = by_turn[turn]
        count = int(fields["input_tokens"])
        cached = int(fields["cached_input_tokens"]) if fields.get("cached_input_tokens") not in (None, "none") else None
        milliseconds = float(fields["server_prompt_ms"])
        recovered.append({"host_pid":int(key[0]),"request_id":key[1],"utterance_id":turn,"prefill_ms":milliseconds,"prefill_tps":count*1000/milliseconds if milliseconds > 0 else None,"decode_ms":float(fields["server_generation_ms"]),"decode_tps":float(fields["server_generation_tps"]),"prompt_evaluated_tokens":count,"prompt_cached_tokens":cached,"prompt_total_tokens":count+cached if cached is not None else None,"predicted_output_tokens":int(fields["output_tokens"]),"decode_eval_tokens":None})
    preparations = []
    for window in record.get("preparation_windows", []):
        before, after = window.get("metrics_before"), window.get("metrics_after")
        row = {"kind":window["kind"],"wall_seconds":window["wall_seconds"],"prefill_ms":None,"prefill_tps":None,"decode_ms":None,"decode_tps":None,"prompt_evaluated_tokens":None,"prompt_cached_tokens":None,"predicted_output_tokens":None,"decode_eval_tokens":None}
        if before is not None and after is not None:
            keys = ("llamacpp:prompt_tokens_total","llamacpp:prompt_tokens_cached_total","llamacpp:prompt_seconds_total","llamacpp:tokens_predicted_total","llamacpp:tokens_predicted_seconds_total")
            delta = {key:after[key]-before[key] for key in keys if key in before and key in after}
            if len(delta) == len(keys) and all(v >= 0 for v in delta.values()):
                row.update(prompt_evaluated_tokens=delta[keys[0]],prompt_cached_tokens=delta[keys[1]],prefill_ms=delta[keys[2]]*1000,predicted_output_tokens=delta[keys[3]],decode_ms=delta[keys[4]]*1000)
                row["prefill_tps"] = row["prompt_evaluated_tokens"]*1000/row["prefill_ms"] if row["prefill_ms"] > 0 else None
                row["decode_tps"] = after.get("llamacpp:predicted_tokens_seconds") if after.get("llamacpp:predicted_tokens_seconds",0) > 0 and row["decode_ms"] > 0 else None
        if row["prefill_ms"] is None:
            row["unavailable_reason"] = "No before/after native-counter bracket, or process reset; startup/preparation native duration not estimated from cumulative snapshot."
        preparations.append(row)
    return recovered, preparations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()
    timing_dir = args.results / "timings"
    timing_dir.mkdir(exist_ok=True)
    matrices, apps = [], []
    for path in sorted(args.results.glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(record, dict) or "profile" not in record:
            continue
        if "requests" in record:
            requests = record["requests"]
            logs = native_records(args.raw_root / "runs" / record["profile"] / "server.log")
            aligned = len(logs) == len(requests) and bool(requests)
            phases, observations = {}, []
            offset = 0
            for phase in PHASES:
                phase_requests = [r for r in requests if r["phase"] == phase]
                phase_rows = []
                phase_logs = logs[offset:offset+len(phase_requests)] if aligned else []
                for request in phase_requests:
                    timing = request.get("timings") or {}
                    row = {"phase":phase,"index":request["index"],"prefill_ms":timing.get("prompt_ms"),"prefill_tps":timing.get("prompt_per_second"),"decode_ms":timing.get("predicted_ms"),"decode_tps":timing.get("predicted_per_second"),"prompt_evaluated_tokens":timing.get("prompt_n"),"prompt_cached_tokens":timing.get("cache_n"),"prompt_total_tokens":request.get("tokens_evaluated"),"predicted_output_tokens":timing.get("predicted_n"),"final_KV_tokens_cached_not_prompt_hit_count":request.get("tokens_cached")}
                    phase_rows.append(row)
                    observations.append(row)
                group = stats(phase_rows)
                group["decode_eval_tokens"] = distribution([r["decode_eval_tokens"] for r in phase_logs])
                group["native_ordered_phase_records"] = phase_logs
                group["native_reported_count_matches_api_output_multiset"] = sorted(r["decode_eval_tokens"] for r in phase_logs) == sorted(r["predicted_output_tokens"] for r in phase_rows) if phase_logs and all(r.get("predicted_output_tokens") is not None for r in phase_rows) else None
                phases[phase] = group
                offset += len(phase_requests)
            item = {"profile":record["profile"],"status":record["status"],"source_file":path.name,"error_or_unavailable_reason":(record.get("safety_cutoff") or {}).get("classification") or record.get("error") or (None if aligned else "Native log/request counts differ or log unavailable; native decode-eval token count unavailable."),"phases":phases,"observations":observations,"association_limit":"API timing fields belong to their original requests. Native log records are assigned only to phase barriers when total counts match; three-slot completion order is NOT joined to request submission order. Native decode-eval counts are phase distributions, not fabricated per-request joins."}
            matrices.append(item)
        elif path.name.startswith("app-") and "attempt" in record:
            translations = record.get("translations", [])
            managed = record["profile"] == "managed-e4b-stock"
            managed_recovery = None
            if managed:
                raw = args.raw_root / "app" / f"{record['profile']}-attempt{record['attempt']}"
                rows, preparations = recover_managed(raw / "Local/puripuly-heart/puripuly_heart.log", record)
                if not rows:
                    rows = record.get("managed_native_diagnostics", [])
                if not preparations:
                    preparations = record.get("managed_preparation_windows", [])
                source = "Structured [Diagnostic][LlmLatency] server timings from owned isolated log, joined by host PID/request ID to request_end utterance UUID and completed event. Legacy ManagedGemma Performance lines were privacy-redacted. Prefill TPS derives actual uncached evaluations/native ms. Native decode-eval token count unavailable (stock stderr DEVNULL); not replaced by output count."
                servers = record.get("managed_servers", [])
                gpu_proof = bool(servers) and all("--device" in s["command"] and s["command"][s["command"].index("--device")+1] == "Vulkan0" and s.get("exited_after_app_stop") for s in servers) and record.get("resource_peaks",{}).get("server",{}).get("process_gpu_dedicated_bytes",0) > 0
                managed_recovery = {"profile":record["profile"],"source_file":path.name,"original_capture_status":record["status"],"status":"completed_with_recovered_native_diagnostics" if len(rows) == len(translations) == 30 and gpu_proof else "inconclusive","reason":"Capture expected legacy backend Performance lines, but current log policy redacts them. Preserve original classification; structured native diagnostic counters plus owned explicit Vulkan children and nonzero RX allocations recover the evidence without inference rerun or product/log-policy change.","effective_vulkan_process_evidence":gpu_proof,"native_observations":rows,"preparation_windows":preparations,"prefix_groups":{name:stats(selected) for name,selected in (("evaluating_prefix_windows",[r for r in preparations if r.get("prompt_evaluated_tokens",0) and r.get("prefill_ms") is not None]),("zero_eval_apply_or_restore_windows",[r for r in preparations if r.get("prefill_ms") == 0]))},"limits":"models.status backend=gpu is requested state, not device proof. Native stderr/legacy effective-backend line unavailable. CPU executable absent prevents fallback. Prefix-window native cumulative counters differ in scope/rounding from response timings; no unbracketed startup estimate. Prefix n_predict1 may expose a 1us decode timer floor with no generation loop; no output-count-based inflated TPS."}
                (args.results / "managed-e4b-recovery.json").write_text(json.dumps(managed_recovery,ensure_ascii=False,indent=2),encoding="utf-8")
            else:
                rows = native_records(args.raw_root / "app" / f"{record['profile']}-attempt{record['attempt']}" / "server.log")
                source = "Original ordered native server timing lines; actual prompt-eval and generation-eval token counts. Generic OAI app did not persist API prompt-total/cache-hit/output counters, which remain unavailable."
            aligned = len(rows) == len(translations) and bool(translations)
            groups = {"all_text_submissions":stats(rows)}
            if aligned:
                for index, row in enumerate(rows):
                    row.update(translation_ordinal=index, cycle=translations[index]["cycle"], pair=translations[index]["pair"])
                first = [row for index,row in enumerate(rows) if index == 0 or translations[index]["cycle"] != translations[index-1]["cycle"]]
                subsequent = [row for index,row in enumerate(rows) if index != 0 and translations[index]["cycle"] == translations[index-1]["cycle"]]
                groups.update(first_text_per_fresh_app=stats(first), subsequent_text_submissions=stats(subsequent))
                hot, alternation = [], []
                for index, row in enumerate(rows):
                    if index == 0 or translations[index]["cycle"] != translations[index-1]["cycle"]:
                        continue
                    (hot if translations[index]["pair"] == translations[index-1]["pair"] else alternation).append(row)
                groups.update(hot_same_pair=stats(hot),pair_alternation=stats(alternation),three_slots=stats([]))
            item = {"profile":record["profile"],"attempt":record["attempt"],"status":record["status"],"source_file":path.name,"source":source,"native_count_matches_translation_count":aligned,"error_or_unavailable_reason":record.get("error",record.get("evidence_limit")) if not rows else None,"groups":groups,"observations":rows,"association_limit":"Ordered association for sequential synthetic text submissions only; no native task-ID to app UUID proof. App completion UUID matching is independently retained in original records. First text of a new app is NOT necessarily cold prefix: generic server survives app restart; managed prefix preparation/restore precedes text.","wall_metric_boundary":"CLI text-submit launch to matched completed translation event; not native inference alone or audio/STT/GUI latency."}
            item["unavailable_phase_reasons"] = {"three_slots":"Actual app probe submitted sequentially; simultaneous occupancy is measured in standalone matrix, not fabricated for app.","first_prefix":"First text per fresh app is not guaranteed cold: explicit preparation/restore or surviving external server precedes it."}
            if managed_recovery is not None:
                item.update(original_capture_status=record["status"],status=managed_recovery["status"],managed_recovery_file="managed-e4b-recovery.json",preparation_windows=managed_recovery["preparation_windows"])
                item["association_limit"] = source + " First text follows explicit managed preparation/restore, not a guaranteed cold prefix."
            apps.append(item)
        else:
            continue
        (timing_dir / path.name).write_text(json.dumps(item, ensure_ascii=False, indent=2), encoding="utf-8")
    diagnostics = []
    for path in sorted(args.results.glob("memory-attribution-attempt*.json")):
        diagnostic = json.loads(path.read_text(encoding="utf-8"))
        for profile in diagnostic.get("profiles", []):
            rows = []
            for request in profile.get("requests", []):
                timing = request.get("timings") or {}
                rows.append({"prefill_ms":timing.get("prompt_ms"),"prefill_tps":timing.get("prompt_per_second"),"decode_ms":timing.get("predicted_ms"),"decode_tps":timing.get("predicted_per_second"),"prompt_evaluated_tokens":timing.get("prompt_n"),"prompt_cached_tokens":timing.get("cache_n"),"prompt_total_tokens":request.get("tokens_evaluated"),"predicted_output_tokens":timing.get("predicted_n")})
            diagnostic_row = {"profile":profile["profile"],"source_file":path.name,"status":profile["status"],"scope":"Three attribution-diagnostic texts, not a latency-selection benchmark; same already-tested flags.","observations":rows,"group":stats(rows)}
            native = native_records(args.raw_root / "memory" / path.stem.removeprefix("memory-attribution-") / profile["profile"] / "server.log")
            diagnostic_row["native_ordered_records"] = native
            diagnostic_row["group"]["decode_eval_tokens"] = distribution([r["decode_eval_tokens"] for r in native])
            diagnostics.append(diagnostic_row)
            (timing_dir / f"{path.stem}-{profile['profile']}.json").write_text(json.dumps(diagnostic_row,indent=2),encoding="utf-8")
    result = {
        "definitions": {
            "prefill_ms": "Native duration evaluating only prompt_n uncached tokens, never the full cached prompt.",
            "prefill_tps": "Native prompt_per_second/logged prompt TPS; managed uses actual uncached evaluated_tokens*1000/prompt_ms per observation, not ratio of medians.",
            "decode_ms": "Native predicted_ms / generation_ms / printed eval-time duration.",
            "decode_tps": "Native reported rate, retained verbatim; do not recompute from the printed/output token count.",
            "prompt_evaluated_tokens": "Actual uncached prompt evaluations.",
            "prompt_cached_tokens": "timings.cache_n / cached_prompt_tokens, preexisting prompt KV reuse, not final cache occupancy.",
            "prompt_total_tokens": "API tokens_evaluated total prompt (evaluated+cached); unavailable for generic app logs.",
            "predicted_output_tokens": "API predicted_n / managed completion_tokens, including first sampled output.",
            "decode_eval_tokens": "Native perf line's PRINTED eval-token count, not an independently measured number of pure timed forward passes. Observed printed count matches API output count; the native reported TPS convention can account one fewer. Managed stderr absent: this printed counter is unavailable, not estimated.",
            "rate_example": "3072 first-prefix: API/printed count21,423.682ms,reported47.205215tokens/s; rate*duration implies20, not21. Preserve native rate rather than inflate it by dividing21/duration.",
            "tails": "Nearest-rank p95; every field retains n/median/p95/min/max. Durations ms; TPS tokens/s.",
            "scope": "All standalone/app identities, failed and pressure-qualified rows, plus separately labeled attribution texts."
        },
        "matrix": matrices,
        "app": apps,
        "memory_diagnostic_work": diagnostics,
    }
    (args.results / "timing-summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown = ["# Split native prefill / decode timings — all tested options", "", "Native timings are **not audio end-to-end latency**. App wall time is separately CLI text-submit launch → matched translation-completion event, excluding audio collection, endpointing, STT and GUI rendering. Failed/unavailable fields are N/A, never invented zero.", "", "**Tokens:** P = actual uncached prompt evaluations; C = preexisting prompt KV hits; O = API predicted output; D = native perf line's **printed** eval-token count. API tokens_evaluated is total prompt; tokens_cached is final KV occupancy, not C. Do not divide cached prompt by prefill time. Printed D can equal O while native TPS accounts O−1: observed21 output/printed tokens,423.682ms,47.205215tokens/s (rate numerator20). We retain reported TPS; D is not a traced pure forward-pass counter. Generic logs lack C/O/total; stock managed stderr lacks D. JSON retains actual source conventions and spread.", "", "Throughput medians are per-observation rates, not ratios of medians. Pressure-qualified/repeat identities remain; consult RESULTS.md/repair-matrix.json before ranking. Standalone concurrent native logs are phase-only, not false per-request joins. Generic app timings are ordered; managed structured diagnostics additionally join host/request ID→utterance UUID.", ""]
    markdown += ["**Reference labels:** `e4b-product` in standalone tables is the historical standalone E4B control; in app tables it is a **generic user-managed endpoint reference, not current managed product latency**. `managed-e4b-stock` is the true managed owner/prefix-slot reference.", "", "App hot_same_pair and pair_alternation group actual adjacent same-pair versus changed-pair submissions within each fresh app process (n4/n24 for successful30-text runs). App three_slots is N/A: submissions were sequential; three-slot occupancy is separately measured in the standalone matrix.", ""]
    for phase in PHASES:
        markdown += [f"## Standalone: {phase}", ""] + table([(r["profile"],r["phases"][phase]) for r in matrices]) + [""]
    for group in ("all_text_submissions", "first_text_per_fresh_app", "subsequent_text_submissions", "hot_same_pair", "pair_alternation", "three_slots"):
        markdown += [f"## Actual app native inference: {group}", "", "First text per app instance is not a guaranteed cold-prefix observation; explicit managed preparation/restore is outside the text-submit wall timer.", ""] + table([(f"{r['profile']} / attempt{r['attempt']}",r["groups"].get(group,stats([]))) for r in apps]) + [""]
    recovery_path = args.results / "managed-e4b-recovery.json"
    if recovery_path.exists():
        recovery = json.loads(recovery_path.read_text(encoding="utf-8"))
        markdown += ["## True managed E4B: separate prefix evaluation / zero-eval restore windows", "", "Native counter deltas bracket completed language-apply operations only; CLI preparation wall includes provider/settings/restore overhead. Unbracketed initial enable is explicitly N/A. Zero-evaluation windows can be resident or restored prefixes; TPS unavailable rather than inflated. The prefix one-output-token request has no normal decode loop; native timer floors are not throughput benchmarks.", ""] + table(list(recovery["prefix_groups"].items())) + [""]
    markdown += ["## Bounded memory-attribution diagnostic work (not ranking runs)", ""] + table([(r["profile"],r["group"]) for r in diagnostics]) + [""]
    markdown += ["## Missing / failed records", ""]
    markdown += [f"- {r['profile']}: {r['error_or_unavailable_reason']}" for r in matrices+apps if r.get("error_or_unavailable_reason")]
    markdown += ["", "Per-run sidecars are under results/timings/; originals are not rewritten. Complete distributions/raw native observations are in results/timing-summary.json. True managed prefix-preparation/restore wall windows and any available native metric snapshots remain in its original app record, separately from translation durations.", ""]
    args.markdown.write_text("\n".join(markdown), encoding="utf-8")
    print(json.dumps({"matrix_run_identities":len(matrices),"app_run_identities":len(apps),"standalone_timed_requests":sum(len(r["observations"]) for r in matrices),"app_native_timing_records":sum(len(r["observations"]) for r in apps)}))


if __name__ == "__main__":
    main()
