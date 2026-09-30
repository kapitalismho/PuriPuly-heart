from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import json
import sys
from hashlib import sha256
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
GAP = ROOT.parents[1]
ARM = "D_no_S200"


def harness():
    sys.path.insert(0, str(GAP))
    import factorial
    return factorial


BASE = harness()


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def signature(events):
    return [{k:v for k,v in e.items() if k != "data"} for e in events]


def baseline_usage():
    entries = load(BASE.OUT / "quota_ledger.json")["entries"]
    if len(entries) != 16 or any(e["status"] != "complete" for e in entries):
        raise ValueError("Require the completed immutable16-call baseline")
    return {"requests": len(entries), "audio_seconds": sum(e["actual_audio_seconds"] for e in entries), "websocket_seconds": sum(e["active_websocket_seconds"] for e in entries)}


def prepare(case, cfg):
    events, timeline, info = BASE.stream(case, "D", cfg)
    removed = 0
    retained = []
    control_remap = []
    pcm_remap = []
    original_source = []
    for event in events:
        if event["type"] == "pcm" and event["kind"] == "S200":
            removed += len(event["data"]) // 2
            continue
        new = dict(event)
        if event["type"] == "pcm":
            original_source.append(event)
            new["provider_start_sample"] -= removed
            new["provider_end_sample"] -= removed
            pcm_remap.append({"old_provider_start_sample": event["provider_start_sample"], "old_provider_end_sample": event["provider_end_sample"], "new_provider_start_sample": new["provider_start_sample"], "new_provider_end_sample": new["provider_end_sample"], "source_start_sample": event["source_start_sample"], "source_end_sample": event["source_end_sample"], "available_source_sample": event["available_source_sample"], "sha256": event["sha256"]})
        else:
            new["provider_cut_sample"] -= removed
            if event["type"] == "finalize":
                control_remap.append({"old_provider_cut_sample": event["provider_cut_sample"], "new_provider_cut_sample": new["provider_cut_sample"], "available_source_sample": event["available_source_sample"], "terminal": event["terminal"]})
        retained.append(new)
    trace = load(BASE.OUT / "traces" / (case["id"] + ".json"))
    new_timeline = []
    provider = 0
    pieces = []
    for p in trace["pieces"]:
        if p["kind"] == "S200":
            continue
        length = p["provider_end_sample"] - p["provider_start_sample"]
        new_timeline.append({"provider_start_s": provider / BASE.RATE, "provider_end_s": (provider + length) / BASE.RATE, "source_start_s": p["source_start_sample"] / BASE.RATE, "source_end_s": p["source_end_sample"] / BASE.RATE, "boundary": bool(provider)})
        pieces.append({**p, "original_D_provider_start_sample": p["provider_start_sample"], "original_D_provider_end_sample": p["provider_end_sample"], "provider_start_sample": provider, "provider_end_sample": provider + length})
        provider += length
    source_events = [e for e in retained if e["type"] == "pcm"]
    if [(e["data"], e["source_start_sample"], e["source_end_sample"], e["available_source_sample"]) for e in source_events] != [(e["data"], e["source_start_sample"], e["source_end_sample"], e["available_source_sample"]) for e in original_source]:
        raise ValueError("S200 removal changed source PCM/replay/availability")
    if sum(len(e["data"]) for e in source_events) != provider * 2:
        raise ValueError("Source map/sample count mismatch")
    actual_D = load(BASE.OUT / "live" / case["id"] / "D" / "result.json")
    actual_log = [json.loads(line) for line in (BASE.OUT / "live" / case["id"] / "D" / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    actual_pcm = [e for e in actual_log if e["event"] == "pcm_write"]
    if [(e["sha256"],e["provider_start_sample"],e["provider_end_sample"],e["available_source_sample"]) for e in actual_pcm] != [(e["sha256"],e["provider_start_sample"],e["provider_end_sample"],e["available_source_sample"]) for e in events if e["type"] == "pcm"]:
        raise ValueError("Base sender does not reproduce immutable original-D packets")
    if not actual_D["completed"] or actual_D["pcm_sha256"] != info["pcm_sha256"]:
        raise ValueError("Original D receipt mismatch")
    control_fields = ("available_source_sample", "terminal")
    if [tuple(e[k] for k in control_fields) for e in retained if e["type"] == "finalize"] != [tuple(e[k] for k in control_fields) for e in events if e["type"] == "finalize"]:
        raise ValueError("Manual controls changed")
    cuts = []
    for cut in info["cut_positions"]:
        removed_before = sum(p["provider_end_sample"]-p["provider_start_sample"] for p in trace["pieces"] if p["kind"] == "S200" and p["provider_end_sample"] <= cut["provider_cut_sample"])
        cuts.append({**cut, "original_D_provider_cut_sample": cut["provider_cut_sample"], "provider_cut_sample": cut["provider_cut_sample"]-removed_before})
    ablation = {"intervention": "S200 only off", "removed_samples": removed, "removed_seconds": removed / BASE.RATE, "original_D_pcm_sha256": info["pcm_sha256"], "original_D_result_sha256": digest(BASE.OUT / "live" / case["id"] / "D" / "result.json"), "original_D_events_sha256": digest(BASE.OUT / "live" / case["id"] / "D" / "events.jsonl"), "control_remap": control_remap, "source_packet_remap": pcm_remap, "pieces": pieces, "same_source_packets_replay_availability_controls": True, "pacing_limit": "Paced S200 removal can advance manual controls and successors; no claim of equal wall time"}
    info = {**info, "arm": ARM, "sent_seconds": provider / BASE.RATE, "pcm_sha256": sha256(b"".join(e["data"] for e in source_events)).hexdigest(), "cut_positions": cuts, "ablation": ablation}
    return retained, new_timeline, info


def plan():
    cfg = load(ROOT / "config.json")
    base_cfg = BASE.config()
    rows = []
    for row in cfg["order"]:
        events, timeline, info = prepare(BASE.cases()[row["case_id"]], base_cfg)
        rows.append({**info, "timeout_seconds": row["timeout_seconds"], "event_signature_sha256": sha256(json.dumps(signature(events),sort_keys=True).encode()).hexdigest(), "timeline": timeline})
    used = baseline_usage()
    total_audio = sum(r["sent_seconds"] for r in rows)
    total_ws = sum(r["timeout_seconds"] for r in rows)
    auth = cfg["authorization"]
    if len(rows) != 2 or total_audio > auth["max_new_audio_seconds"] + 1e-9 or total_ws > auth["max_new_websocket_seconds"]:
        raise ValueError("Ablation bounds exceeded")
    if used["requests"]+2 > min(auth["max_cumulative_requests"],auth["global_request_cap"]) or used["audio_seconds"]+total_audio > min(auth["max_cumulative_audio_seconds"],auth["global_audio_seconds_cap"]) + 1e-9 or used["websocket_seconds"]+total_ws > min(auth["max_cumulative_websocket_seconds"],auth["global_websocket_seconds_cap"]) + 1e-9:
        raise ValueError("Cumulative provider quota exceeded")
    result = {"authorization": auth, "rows": rows, "baseline_usage": used, "baseline_quota_sha256": digest(BASE.OUT / "quota_ledger.json"), "base_config_sha256": digest(GAP / "experiment_config.json"), "ablation_config_sha256": digest(ROOT / "config.json"), "base_harness_sha256": digest(GAP / "factorial.py"), "ablation_runner_sha256": digest(Path(__file__)), "environment": BASE.environment(), "new_requests": 2, "new_audio_seconds": total_audio, "new_websocket_bound_seconds": total_ws}
    path = ROOT / "run_plan.json"
    if path.exists() and load(path) != result:
        raise ValueError("Frozen ablation plan changed")
    if not path.exists():
        BASE.save(path,result)
    return result


async def execute(live=False):
    frozen = plan()
    cfg = BASE.config()
    ledger_path = ROOT / "quota_ledger.json"
    ledger = load(ledger_path) if ledger_path.exists() else {"baseline_usage": frozen["baseline_usage"], "entries": []}
    if live and (not frozen["authorization"]["granted"] or not BASE.credential()):
        raise RuntimeError("Authorization or configured credential unavailable")
    if live and any(e["status"] != "complete" for e in ledger["entries"]):
        raise RuntimeError("Prior uncertain/failed ablation; no retry")
    calls_started = 0
    for row in frozen["rows"]:
        identity = row["case_id"] + "/" + ARM
        if live and any(e["identity"] == identity for e in ledger["entries"]):
            continue
        if live:
            auth = frozen["authorization"]
            entries = ledger["entries"]
            if len(entries)+1 > 2 or sum(e["reserved_audio_seconds"] for e in entries)+row["sent_seconds"] > auth["max_new_audio_seconds"]+1e-9 or sum(e["reserved_websocket_seconds"] for e in entries)+row["timeout_seconds"] > auth["max_new_websocket_seconds"]:
                raise ValueError("Ablation cumulative reservation exceeded")
            entry = {"identity": identity, "status": "reserved_before_connect", "reserved_audio_seconds": row["sent_seconds"], "reserved_websocket_seconds": row["timeout_seconds"]}
            entries.append(entry)
            BASE.save(ledger_path,ledger)
            calls_started += 1
        prepared = prepare(BASE.cases()[row["case_id"]],cfg)
        result = await BASE.replay(BASE.cases()[row["case_id"]],ARM,cfg,live=live,timeout=row["timeout_seconds"],prepared_stream=prepared,output_root=ROOT)
        if live:
            entry.update(status="complete" if result["completed"] else "failed_no_retry",actual_audio_seconds=result["actual_audio_seconds"],active_websocket_seconds=result["active_websocket_seconds"])
            BASE.save(ledger_path,ledger)
        if not result["completed"]:
            raise RuntimeError("Ablation incomplete; stop without retry")
    print(json.dumps({"ablation_calls_this_invocation":calls_started,"planned_new_audio_seconds":frozen["new_audio_seconds"],"websocket_bound_seconds":frozen["new_websocket_bound_seconds"],"stop_after_declared_cases":True}),flush=True)


def project():
    sys.path.insert(0,str(GAP.parent))
    from run import conversion
    for row in load(ROOT / "config.json")["order"]:
        case_id = row["case_id"]
        result = load(ROOT / "live" / case_id / ARM / "result.json")
        if not result["completed"]:
            raise ValueError("Incomplete ablation cannot be scored")
        rttm, artifact = conversion(result,BASE.cases()[case_id]["duration_s"])
        BASE.save(ROOT / "sdk_output" / ARM / (case_id+".json"),artifact)
        path = ROOT / "predictions" / ARM / (case_id+".rttm")
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(rttm.encode("utf-8"))
    print("projected_ablation=2 sdk=@soniox/node@2.3.0")


def score():
    sys.path.insert(0,str(BASE.BENCH))
    from score import score_case
    if importlib.metadata.version("pyannote.metrics") != "4.0.0":
        raise ValueError("Pinned scorer required")
    rows = [score_case(BASE.cases()[row["case_id"]],ROOT / "predictions" / ARM) for row in load(ROOT / "config.json")["order"]]
    BASE.save(ROOT / "scores.json",{"pyannote.metrics":"4.0.0","collar":0,"overlap_included":True,"mapping":"one entire-case optimal speaker mapping","per_arm":{ARM:rows}})
    print(json.dumps(rows,indent=2))


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--live",action="store_true")
    parser.add_argument("--project",action="store_true")
    parser.add_argument("--score",action="store_true")
    args = parser.parse_args()
    if args.project:
        project()
    elif args.score:
        score()
    else:
        await execute(live=args.live)


if __name__ == "__main__":
    asyncio.run(main())
