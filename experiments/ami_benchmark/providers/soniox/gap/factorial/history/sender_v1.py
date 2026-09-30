from __future__ import annotations

import argparse
import asyncio
from collections import deque
from hashlib import sha256
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time
import wave

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
OUT = ROOT / "factorial"
BENCH = ROOT.parents[2]
REPO = BENCH.parents[1]
RATE = 16000
sys.path.insert(0, str(REPO / "src"))


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def config():
    return json.loads((ROOT / "experiment_config.json").read_text(encoding="utf-8"))


def cases():
    manifest = json.loads((BENCH / "manifest.json").read_text(encoding="utf-8"))
    return {c["id"]: c for c in manifest["cases"]}


def audio(case):
    path = BENCH / case["audio"]
    with wave.open(str(path), "rb") as wav:
        if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) != (RATE, 1, 2):
            raise ValueError("Expected mono 16kHz PCM16")
        pcm = wav.readframes(wav.getnframes())
    if sha256(path.read_bytes()).hexdigest() != case["clip_wav_sha256"] or sha256(pcm).hexdigest() != case["clip_pcm_sha256"]:
        raise ValueError("Frozen audio digest mismatch")
    return pcm


def app_pcm(pcm):
    from puripuly_heart.core.audio.format import float32_to_pcm16le_bytes, pcm16le_bytes_to_float32
    return float32_to_pcm16le_bytes(pcm16le_bytes_to_float32(pcm))


def credential():
    import keyring
    return os.environ.get("SONIOX_API_KEY") or keyring.get_password("puripuly-heart", "soniox_api_key")


async def freeze(case, cfg):
    import numpy as np
    from puripuly_heart.config.resolved import vad_exit_threshold
    from puripuly_heart.core.audio.format import AudioCaptureSpan
    from puripuly_heart.core.audio.listen_delivery import ListenDeliveryController
    from puripuly_heart.core.audio.ownership import AudioSegmentSettingsSnapshot, PeerAudioSegmentLedger
    from puripuly_heart.core.audio.smart_turn import SmartTurnInferenceOwner, SMART_TURN_COMPLETE_THRESHOLD
    from puripuly_heart.core.vad.bundled import bundled_silero_vad_onnx_path
    from puripuly_heart.core.vad.gating import SpeechChunk, SpeechEnd, SpeechStart, create_peer_vad_gating
    from puripuly_heart.core.vad.silero import SileroVadOnnx
    path = OUT / "traces" / (case["id"] + ".json")
    if path.exists():
        raise FileExistsError("Trace already frozen: " + str(path))
    pcm = audio(case)
    encoded = app_pcm(pcm)
    opts = cfg["segmentation"]
    vad = create_peer_vad_gating(SileroVadOnnx(bundled_silero_vad_onnx_path()), sample_rate_hz=RATE, ring_buffer_ms=opts["pre_roll_ms"], speech_threshold=opts["speech_threshold"], hangover_ms=opts["hangover_ms"])
    settings = AudioSegmentSettingsSnapshot(provider_id="soniox", provider_signature=(cfg["provider"]["model"],), runtime_signature=("factorial-frozen-peer",), source_mode="manual", source_language="en", expected_languages=("en",), target_sample_rate_hz=RATE, vad_speech_threshold=opts["speech_threshold"], vad_exit_threshold=vad_exit_threshold(opts["speech_threshold"]), vad_hangover_ms=opts["hangover_ms"], vad_pre_roll_ms=opts["pre_roll_ms"], delivery_profile_effective="on", delivery_availability="ready", delivery_threshold=SMART_TURN_COMPLETE_THRESHOLD)
    clock = [0.0]
    smart = SmartTurnInferenceOwner(clock=lambda: clock[0])
    ledger = PeerAudioSegmentLedger(activation_generation=1, settings=settings)
    pieces, cuts = [], []
    provider_sample = 0
    turn = -1

    async def emit(owned):
        nonlocal provider_sample, turn
        event = owned.event
        if isinstance(event, SpeechStart):
            turn += 1
            spans = (*event.pre_roll_capture, *event.chunk_capture)
        elif isinstance(event, SpeechChunk):
            spans = event.chunk_capture
        elif isinstance(event, SpeechEnd):
            content = owned.segment.content_sample_count * 1000 / RATE
            padding = 3200 if event.reason == "delivery_deadline" or (event.reason == "delivery_pause" and 4000 <= content < 7000) else 0
            if padding:
                pieces.append({"kind": "S200", "turn": turn, "source_start_sample": None, "source_end_sample": None, "provider_start_sample": provider_sample, "provider_end_sample": provider_sample + padding, "available_source_sample": round(clock[0] * RATE), "pcm_sha256": sha256(bytes(padding * 2)).hexdigest()})
                provider_sample += padding
            cuts.append({"turn": turn, "reason": event.reason, "source_cut_sample": round(clock[0] * RATE), "app_provider_cut_sample": provider_sample, "content_ms": content, "observed_trailing_silence_ms": event.trailing_silence_ms})
            return
        else:
            return
        for span in spans:
            a, b = span.source_start_sample, span.source_end_sample
            pieces.append({"kind": "source", "turn": turn, "source_start_sample": a, "source_end_sample": b, "provider_start_sample": provider_sample, "provider_end_sample": provider_sample + b - a, "available_source_sample": round(clock[0] * RATE), "source_pcm_sha256": sha256(pcm[a * 2:b * 2]).hexdigest(), "pcm_sha256": sha256(encoded[a * 2:b * 2]).hexdigest()})
            provider_sample += b - a

    controller = ListenDeliveryController(vad=vad, ledger=ledger, emit=emit, monotonic_clock=lambda: clock[0], smart_turn_owner=smart)
    samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768
    size = vad.chunk_samples
    started = time.monotonic()
    try:
        for index, a in enumerate(range(0, len(samples) - size + 1, size)):
            b = a + size
            await asyncio.sleep(max(0, started + b / RATE - time.monotonic()))
            clock[0] = b / RATE
            span = AudioCaptureSpan(capture_epoch=0, callback_sequence=index, source_sample_rate_hz=RATE, source_start_sample=a, source_end_sample=b, source_start_monotonic_s=a / RATE, source_end_monotonic_s=b / RATE, normalized_sample_rate_hz=RATE, normalized_start_sample=a, normalized_end_sample=b)
            for event in vad.process_owned_chunk(samples[a:b], (span,)):
                await controller.handle_vad_event(event)
            await controller.observe_acoustic_chunk(speech_observed=vad.last_observation_was_speech, capture=(span,))
        clock[0] = len(samples) / RATE
        end = vad.seal_active(reason="source_eof")
        if end is not None:
            await controller.handle_vad_event(end)
        snapshot = smart.snapshot
        trace = {"case_id": case["id"], "baseline_revision": cfg["baseline_revision"], "segmentation": opts, "original_samples": len(samples), "original_pcm_sha256": sha256(pcm).hexdigest(), "pieces": pieces, "cuts": cuts, "app_samples": provider_sample, "dropped_eof_samples": len(samples) % size, "smart_turn": {"availability": snapshot.availability, "inference_count": snapshot.inference_count, "busy_skip_count": snapshot.busy_skip_count}, "environment": environment()}
        save(path, trace)
        print(json.dumps({"frozen": case["id"], "cuts": len(cuts), "app_seconds": provider_sample / RATE, "smart_turn": trace["smart_turn"]}), flush=True)
    finally:
        await controller.close()
        await smart.close()


def environment():
    model_paths = ("src/puripuly_heart/data/vad/silero_vad.onnx", "src/puripuly_heart/data/models/smart-turn-v3.2-cpu.onnx")
    return {"python": platform.python_version(), "platform": platform.platform(), "versions": {x: importlib.metadata.version(x) for x in ("numpy", "onnxruntime", "websockets", "keyring")}, "model_sha256": {p: sha256((REPO / p).read_bytes()).hexdigest() for p in model_paths}}


def stream(case, arm, cfg):
    trace_path = OUT / "traces" / (case["id"] + ".json")
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    pcm = audio(case)
    if trace["original_pcm_sha256"] != sha256(pcm).hexdigest() or trace["segmentation"] != cfg["segmentation"]:
        raise ValueError("Trace input/config changed")
    if arm in ("A", "B"):
        pieces = [{"kind": "source", "source_start_sample": 0, "source_end_sample": len(pcm) // 2, "provider_start_sample": 0, "provider_end_sample": len(pcm) // 2, "available_source_sample": 0}]
        cuts = sorted(set(c["source_cut_sample"] for c in trace["cuts"])) if arm == "B" else []
    else:
        pcm = app_pcm(pcm)
        pieces = trace["pieces"]
        cuts = sorted(set(c["app_provider_cut_sample"] for c in trace["cuts"])) if arm == "D" else []
    chunk_cuts = sorted(set(c["source_cut_sample"] for c in trace["cuts"])) if arm in ("A", "B") else sorted(set(c["app_provider_cut_sample"] for c in trace["cuts"]))
    events, timeline = [], []
    digest = sha256()
    provider_end = 0
    for piece in pieces:
        a, b = piece["provider_start_sample"], piece["provider_end_sample"]
        boundaries = sorted(set([a, b] + list(range(a, b, cfg["wire"]["chunk_samples"])) + [x for x in chunk_cuts if a < x < b]))
        sa = piece["source_start_sample"]
        timeline.append({"provider_start_s": a / RATE, "provider_end_s": b / RATE, "source_start_s": sa / RATE if sa is not None else None, "source_end_s": piece["source_end_sample"] / RATE if sa is not None else None, "boundary": bool(a)})
        for x, y in zip(boundaries, boundaries[1:]):
            data = pcm[(sa + x - a) * 2:(sa + y - a) * 2] if sa is not None else bytes((y - x) * 2)
            available = max(piece["available_source_sample"], sa + y - a if sa is not None else 0)
            events.append({"type": "pcm", "provider_start_sample": x, "provider_end_sample": y, "available_source_sample": available, "kind": piece["kind"], "sha256": sha256(data).hexdigest(), "data": data})
            digest.update(data)
            if y in cuts:
                events.append({"type": "finalize", "provider_cut_sample": y, "available_source_sample": max(available, next((c["source_cut_sample"] for c in trace["cuts"] if c["app_provider_cut_sample"] == y), available)) if arm == "D" else available, "terminal": False})
        provider_end = b
    events.append({"type": "finalize", "provider_cut_sample": provider_end, "available_source_sample": len(pcm) // 2, "terminal": True})
    events.append({"type": "end_stream", "provider_cut_sample": provider_end, "available_source_sample": len(pcm) // 2})
    return events, timeline, {"case_id": case["id"], "arm": arm, "source_seconds": len(pcm) / RATE / 2, "sent_seconds": provider_end / RATE, "pcm_sha256": digest.hexdigest(), "trace_sha256": sha256(trace_path.read_bytes()).hexdigest(), "finalizations": sum(e["type"] == "finalize" for e in events), "cut_positions": [{"source_cut_sample": c["source_cut_sample"], "provider_cut_sample": c["source_cut_sample"] if arm in ("A", "B") else c["app_provider_cut_sample"], "reason": c["reason"]} for c in trace["cuts"]]}


def plan(cfg):
    rows = []
    all_cases = cases()
    for case_id, order in zip(cfg["subset"], cfg["arm_order_by_case"]):
        for arm in order:
            _, _, info = stream(all_cases[case_id], arm, cfg)
            rows.append({**info, "timeout_seconds": all_cases[case_id]["duration_s"] * 1.5 + 45})
    for case_id in cfg["subset"]:
        group = {r["arm"]: r for r in rows if r["case_id"] == case_id}
        if group["A"]["pcm_sha256"] != group["B"]["pcm_sha256"] or group["C"]["pcm_sha256"] != group["D"]["pcm_sha256"]:
            raise ValueError("Factorial PCM invariant failed")
    proposal = {"rows": rows, "requests": len(rows), "audio_seconds": sum(r["sent_seconds"] for r in rows), "websocket_timeout_seconds": sum(r["timeout_seconds"] for r in rows), "reserved_timestamp": cfg["timestamp_diagnostic"], "config_sha256": sha256((ROOT / "experiment_config.json").read_bytes()).hexdigest(), "harness_sha256": sha256(Path(__file__).read_bytes()).hexdigest(), "manifest_sha256": sha256((BENCH / "manifest.json").read_bytes()).hexdigest(), "environment": environment()}
    auth = cfg["authorization"]
    if proposal["requests"] + 4 > auth["max_requests"] or proposal["audio_seconds"] + 80 > auth["max_audio_seconds"] or proposal["websocket_timeout_seconds"] + 300 > auth["max_active_websocket_seconds"]:
        raise ValueError("Frozen plan exceeds authorization")
    path = OUT / "run_plan.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != proposal:
        raise ValueError("Refusing to alter frozen plan")
    if not path.exists():
        save(path, proposal)
    return proposal


async def replay(case, arm, cfg, live=False, timeout=0, fault=None):
    events, timeline, info = stream(case, arm, cfg)
    folder = OUT / ("live" if live else ("dry_" + fault if fault else "dry")) / case["id"] / arm
    if live and folder.exists():
        raise FileExistsError("No implicit retry or overwrite")
    folder.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    log_file = (folder / "events.jsonl").open("w", encoding="utf-8")
    pending = deque()
    tokens = []
    raw_file = (folder / "responses.jsonl").open("w", encoding="utf-8")
    virtual = 0.0
    received_bytes = 0
    fin_count = 0
    completed = False
    sent_samples = 0
    last_audio_end = 0.0
    ws = None
    receiver = None
    failure = None

    def now():
        return time.monotonic() - started if live else virtual

    def log(event_type, **fields):
        log_file.write(json.dumps({"event": event_type, "elapsed_s": now(), "pending_completions": len(pending), **fields}) + "\n")
        log_file.flush()

    async def receive():
        nonlocal fin_count, completed, received_bytes
        async for raw in ws:
            received_bytes += len(raw.encode("utf-8") if isinstance(raw, str) else raw)
            if received_bytes > cfg["wire"]["max_receive_bytes"]:
                raise RuntimeError("Receive byte bound exceeded")
            message = json.loads(raw)
            raw_file.write(json.dumps({"received_elapsed_s": now(), "response": message}, ensure_ascii=False) + "\n")
            raw_file.flush()
            log("receive", response_bytes=len(raw), processed_counters={k:v for k,v in message.items() if k in ("final_audio_proc_ms", "total_audio_proc_ms") or "processed" in k}, finished=message.get("finished"))
            if any(k in message for k in ("error", "error_code", "error_type")):
                raise RuntimeError("Provider error; retained in raw response")
            for token in message.get("tokens") or []:
                if not token.get("is_final"):
                    continue
                if token.get("text") == "<fin>":
                    fin_count += 1
                    matched = pending.popleft() if pending else None
                    log("fin", control=matched, association="FIFO diagnostic only, not documented turn identity")
                elif token.get("text") != "<end>":
                    tokens.append({**token, "request_id": message.get("request_id"), "received_elapsed_s": now()})
                    log("stable_token", token_index=len(tokens)-1, has_speaker=token.get("speaker") is not None, start_ms=token.get("start_ms"), end_ms=token.get("end_ms"))
            if message.get("finished"):
                completed = True
                return
        raise RuntimeError("Connection ended without finished receipt")

    async def run():
        nonlocal ws, receiver, virtual, sent_samples, last_audio_end, fin_count, completed
        if live:
            import websockets
            ws = await websockets.connect(cfg["provider"]["endpoint"], ping_interval=None, open_timeout=cfg["wire"]["open_timeout_seconds"], close_timeout=2, max_size=4 * 1024 * 1024, max_queue=16)
            log("connected")
            wire_config = {k:v for k,v in cfg["provider"].items() if k != "endpoint"}
            await ws.send(json.dumps({**wire_config, "api_key": credential()}))
            log("config_write", config=wire_config)
            receiver = asyncio.create_task(receive())
        last_write = now()
        for event in events:
            if receiver is not None and receiver.done():
                await receiver
                raise RuntimeError("Provider finished before writer completed")
            target = max(event["available_source_sample"] / RATE, last_audio_end)
            while live and now() < target:
                delay = min(target - now(), cfg["wire"]["keepalive_seconds"])
                await asyncio.sleep(delay)
                if now() - last_write >= cfg["wire"]["keepalive_seconds"] and now() < target:
                    await ws.send(json.dumps({"type": "keepalive"}))
                    log("control_write", control="keepalive")
                    last_write = now()
            if not live:
                virtual = target
                if fault == "stall" and event["type"] == "pcm":
                    virtual += 0.003
            log("source_available", available_source_sample=event["available_source_sample"], writer_target_s=target, pacing_lateness_s=max(0, now()-target))
            before = now()
            if event["type"] == "pcm":
                if live:
                    await ws.send(event["data"])
                sent_samples += len(event["data"]) // 2
                log("pcm_write", **{k:v for k,v in event.items() if k not in ("data", "type")}, bytes=len(event["data"]), write_stall_s=now()-before)
                last_audio_end = now() + len(event["data"]) / RATE / 2
            elif event["type"] == "finalize":
                if len(pending) >= cfg["wire"]["max_pending_finalizations"]:
                    raise RuntimeError("Pending completion bound exceeded")
                control = {k:v for k,v in event.items() if k != "type"}
                pending.append({**control, "written_elapsed_s": now()})
                if live:
                    await ws.send(json.dumps({"type": "finalize"}))
                log("control_write", control="finalize", **control, write_stall_s=now()-before)
                if not live and fault != "missing_fin":
                    fin_count += 1
                    log("simulated_fin", control=pending.popleft(), accuracy_evidence=False)
            else:
                if live:
                    deadline = now() + cfg["wire"]["terminal_timeout_seconds"]
                    while pending:
                        if now() >= deadline:
                            raise TimeoutError("Terminal finalize ACK deadline exceeded")
                        if receiver.done():
                            await receiver
                            raise RuntimeError("Stream finished with pending completion")
                        await asyncio.sleep(0.01)
                    await ws.send("")
                log("control_write", control="end_stream")
            last_write = now()
        if live:
            await asyncio.wait_for(receiver, cfg["wire"]["terminal_timeout_seconds"])
        else:
            completed = not pending
        if pending or fin_count != info["finalizations"] or not completed:
            raise RuntimeError("Missing terminal/completion barrier")
        if sent_samples != round(info["sent_seconds"] * RATE):
            raise RuntimeError("PCM accounting mismatch")

    try:
        await asyncio.wait_for(run(), timeout - 5) if live else await run()
    except Exception as exc:
        failure = type(exc).__name__
        log("failure", exception_type=failure)
    finally:
        if receiver is not None:
            receiver.cancel()
            await asyncio.gather(receiver, return_exceptions=True)
        if ws is not None:
            await ws.close()
        elapsed = now()
        log("closed", completed=completed, actual_audio_seconds=sent_samples / RATE, active_websocket_seconds=time.monotonic()-started if live else 0)
        raw_file.close()
        log_file.close()
        result = {**info, "mode": "live" if live else "dry", "completed": completed and not pending and failure is None, "failure": failure, "fin_count": fin_count, "pending_completions": list(pending), "actual_audio_seconds": sent_samples / RATE, "active_websocket_seconds": time.monotonic()-started if live else 0, "replay_elapsed_seconds": elapsed, "timeline": timeline, "tokens": tokens, "raw_responses_sha256": sha256((folder / "responses.jsonl").read_bytes()).hexdigest(), "config": {k:v for k,v in cfg["provider"].items() if k != "endpoint"}, "environment": environment()}
        result["native_artifact_sha256"] = sha256(json.dumps(tokens, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        save(folder / "result.json", result)
        print(json.dumps({k:result[k] for k in ("case_id", "arm", "mode", "completed", "failure", "actual_audio_seconds", "active_websocket_seconds", "fin_count")}), flush=True)
    return result


async def execute(cfg):
    frozen = plan(cfg)
    if not cfg["authorization"]["granted"] or not credential():
        raise RuntimeError("Authorization or configured credential missing")
    path = OUT / "quota_ledger.json"
    ledger = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"entries": []}
    if any(e["status"] != "complete" for e in ledger["entries"]):
        raise RuntimeError("Previous failed/uncertain request needs Director decision; no automatic retry or continuation")
    all_cases = cases()
    for row in frozen["rows"]:
        identity = row["case_id"] + "/" + row["arm"]
        if any(e["identity"] == identity for e in ledger["entries"]):
            continue
        auth = cfg["authorization"]
        if len(ledger["entries"]) + 1 > auth["max_requests"] or sum(e["reserved_audio_seconds"] for e in ledger["entries"]) + row["sent_seconds"] > auth["max_audio_seconds"] or sum(e["reserved_websocket_seconds"] for e in ledger["entries"]) + row["timeout_seconds"] > auth["max_active_websocket_seconds"]:
            raise RuntimeError("Cumulative quota reservation exceeded")
        entry = {"identity": identity, "reserved_audio_seconds": row["sent_seconds"], "reserved_websocket_seconds": row["timeout_seconds"], "status": "reserved_before_connect"}
        ledger["entries"].append(entry)
        save(path, ledger)
        result = await replay(all_cases[row["case_id"]], row["arm"], cfg, live=True, timeout=row["timeout_seconds"])
        entry.update(status="complete" if result["completed"] else "failed_no_retry", actual_audio_seconds=result["actual_audio_seconds"], active_websocket_seconds=result["active_websocket_seconds"])
        save(path, ledger)
        if not result["completed"]:
            raise RuntimeError("Planned run failed; stop for decision, no retry")


def project(cfg):
    sys.path.insert(0, str(ROOT.parent))
    from run import conversion
    for case_id in cfg["subset"]:
        for arm in "ABCD":
            result = json.loads((OUT / "live" / case_id / arm / "result.json").read_text(encoding="utf-8"))
            if not result["completed"]:
                raise ValueError("Cannot project incomplete provider run")
            rttm, artifact = conversion(result, cases()[case_id]["duration_s"])
            save(OUT / "sdk_output" / arm / (case_id + ".json"), artifact)
            pred = OUT / "predictions" / arm / (case_id + ".rttm")
            pred.parent.mkdir(parents=True, exist_ok=True)
            pred.write_bytes(rttm.encode("utf-8"))
    print("projected=16 sdk=@soniox/node@2.3.0 group_by=speaker", flush=True)


def score_subset(cfg):
    sys.path.insert(0, str(BENCH))
    from score import score_case
    if importlib.metadata.version("pyannote.metrics") != "4.0.0":
        raise ValueError("Pinned scorer required")
    rows = {arm:[score_case(cases()[c], OUT / "predictions" / arm) for c in cfg["subset"]] for arm in "ABCD"}
    save(OUT / "scores.json", {"pyannote.metrics": "4.0.0", "collar": 0, "overlap_included": True, "mapping": "one optimal entire-case speaker mapping", "per_arm": rows})
    print(json.dumps(rows, indent=2))


def word_diagnostics(cfg):
    from diagnose import reference_words, recognized_words, identity_diagnostic
    inventory = json.loads((ROOT / "ami_word_reference_inventory.json").read_text(encoding="utf-8"))
    rows = {}
    for case_id in cfg["subset"]:
        reference, gaps = reference_words(cases()[case_id], inventory)
        rows[case_id] = {"reference_gaps": gaps, "arms": {}}
        for arm in "ABCD":
            result = json.loads((OUT / "live" / case_id / arm / "result.json").read_text(encoding="utf-8"))
            if not result["completed"]:
                raise ValueError("Cannot diagnose an incomplete run")
            rows[case_id]["arms"][arm] = identity_diagnostic(reference, recognized_words(result["tokens"]))
    save(OUT / "word_diagnostics.json", {"method": "existing conservative exact lexical anchor diagnostic; no native times/widths; one global per-case mapping; conditional identity is not transcription accuracy", "reference_inventory_sha256": sha256((ROOT / "ami_word_reference_inventory.json").read_bytes()).hexdigest(), "cases": rows})
    print("word_diagnostics=16 independent_lexical_alignment=True")


def summarize(cfg):
    scores = json.loads((OUT / "scores.json").read_text(encoding="utf-8"))
    per_arm = {arm: {row["id"]: row for row in scores["per_arm"][arm]} for arm in "ABCD"}
    contrasts = {}
    reception = {}
    for case_id in cfg["subset"]:
        contrasts[case_id] = {}
        for pair in ("AB", "CD", "AC", "BD"):
            left, right = (per_arm[a][case_id] for a in pair)
            contrasts[case_id][pair] = {"der_difference_percentage_points_right_minus_left": 100 * (right["der"] - left["der"]), **{k + "_seconds_difference": right[k]-left[k] for k in ("missed detection", "false alarm", "confusion")}}
        reception[case_id] = {}
        for arm in "ABCD":
            folder = OUT / "live" / case_id / arm
            result = json.loads((folder / "result.json").read_text(encoding="utf-8"))
            events = [json.loads(line) for line in (folder / "events.jsonl").read_text(encoding="utf-8").splitlines()]
            pcm = [e for e in events if e["event"] == "pcm_write"]
            fin = [e for e in events if e["event"] == "fin"]
            tokens = result["tokens"]
            attributed = [t for t in tokens if t.get("speaker") is not None]
            controls = [e for e in events if e["event"] == "control_write" and e["control"] == "finalize"]
            transitions = []
            for control in controls:
                successor = next((e for e in pcm if e["elapsed_s"] > control["elapsed_s"]), None)
                receipt = next((e for e in fin if e["control"] and e["control"]["written_elapsed_s"] == control["elapsed_s"]), None)
                transitions.append({"provider_cut_sample": control["provider_cut_sample"], "terminal": control["terminal"], "finalize_written_s": control["elapsed_s"], "fin_received_s": receipt["elapsed_s"] if receipt else None, "successor_pcm_s": successor["elapsed_s"] if successor else None})
            reception[case_id][arm] = {"first_stable_token_s": tokens[0]["received_elapsed_s"] if tokens else None, "first_stable_attributed_token_s": attributed[0]["received_elapsed_s"] if attributed else None, "stable_tokens": len(tokens), "missing_speaker_tokens": len(tokens)-len(attributed), "maximum_pending_completions": max(e["pending_completions"] for e in events), "max_send_stall_seconds": max(e["write_stall_s"] for e in pcm), "max_source_availability_to_pcm_write_seconds": max(e["elapsed_s"] - e["available_source_sample"]/RATE for e in pcm), "actual_audio_seconds": result["actual_audio_seconds"], "active_websocket_seconds": result["active_websocket_seconds"], "finalizations": result["finalizations"], "transitions": transitions, "readiness_limit": "Provider evidence arrival only. No caption publication, physical source-speech end, translation, VRChat or HMD readiness measurement."}
    pooled = {}
    for arm in "ABCD":
        rows = list(per_arm[arm].values())
        sums = {k: sum(r[k] for r in rows) for k in ("total", "missed detection", "false alarm", "confusion")}
        pooled[arm] = {**sums, "der": sum(sums[k] for k in ("missed detection", "false alarm", "confusion")) / sums["total"]}
    save(OUT / "summary.json", {"subset_only": True, "single_run_no_significance_claim": True, "pooled": pooled, "contrasts": contrasts, "reception": reception, "timestamp_status": "Unresolved; independently verified acoustic anchors missing; no correction", "application_readiness": "Not established by these live recorded-file streams"})
    print(json.dumps({"pooled": pooled, "contrasts": contrasts}, indent=2))


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--project", action="store_true")
    parser.add_argument("--score", action="store_true")
    parser.add_argument("--words", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--case")
    parser.add_argument("--fault", choices=("stall", "missing_fin"))
    args = parser.parse_args()
    cfg = config()
    if args.freeze:
        for case_id in [args.case] if args.case else cfg["subset"]:
            await freeze(cases()[case_id], cfg)
    elif args.project:
        project(cfg)
    elif args.score:
        score_subset(cfg)
    elif args.words:
        word_diagnostics(cfg)
    elif args.summarize:
        summarize(cfg)
    elif args.live:
        if args.case or args.fault:
            raise ValueError("Live execution only follows frozen complete plan")
        await execute(cfg)
    else:
        proposal = plan(cfg)
        for row in proposal["rows"]:
            if not args.case or args.case == row["case_id"]:
                await replay(cases()[row["case_id"]], row["arm"], cfg, fault=args.fault)
        print(json.dumps({"planned_requests": proposal["requests"], "planned_audio_seconds": proposal["audio_seconds"], "planned_websocket_bound_seconds": proposal["websocket_timeout_seconds"]}), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
