from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import queue
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.request

import psutil
from run_matrix import REPO, sanitize
from windows_metrics import Sampler
from summarize_timings import recover_managed, stats


def capture_diagnostics(probe, result):
    log_path = probe / "Local/puripuly-heart/puripuly_heart.log"
    native, preparations = recover_managed(log_path, result)
    result["capture_status_before_extraction"] = result["status"]
    result["managed_native_diagnostics"] = native
    result["managed_preparation_windows"] = preparations
    result["native_timing_source"] = "Structured Diagnostic/LlmLatency, joined host PID/request ID to request_end utterance UUID; legacy Performance text may be privacy-redacted."
    unique_native = len({(r["host_pid"], r["request_id"]) for r in native})
    unique_turns = len({r["utterance_id"] for r in native})
    servers = result.get("managed_servers", [])
    gpu_proof = bool(servers) and all("--device" in s["command"] and s["command"][s["command"].index("--device")+1] == "Vulkan0" and s.get("exited_after_app_stop") for s in servers) and result.get("resource_peaks", {}).get("server", {}).get("process_gpu_dedicated_bytes", 0) > 0
    result["structured_native_joins"] = {"records":len(native), "unique_terminal_request_records":unique_native, "unique_completed_utterances":unique_turns, "effective_vulkan_process_evidence":gpu_proof}
    if result["status"] in {"completed", "inconclusive_effective_backend_or_metrics", "completed_with_structured_native_diagnostics"}:
        valid = len(native) == len(result.get("translations", [])) == unique_native == unique_turns == 30 and gpu_proof and len(result.get("host_release", [])) == 2 and all(r.get("exited") for r in result["host_release"])
        result["status"] = "completed_with_structured_native_diagnostics" if valid else "inconclusive_structured_native_evidence"
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--e4b", type=Path)
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--extract-only", action="store_true")
    parser.add_argument("--extraction-output", type=Path)
    args = parser.parse_args()
    profile = "managed-e4b-stock"
    probe = args.root / "app" / f"{profile}-attempt{args.attempt}"
    output = REPO / "experiments/gemma4_26b_qat/results" / f"app-{profile}-attempt{args.attempt}.json"
    if args.extract_only:
        target = args.extraction_output
        if target is None or not target.resolve().is_relative_to(args.root.resolve()):
            parser.error("Extraction requires an explicit external owned-root output")
        if target.exists():
            raise FileExistsError("Preserve prior extraction output")
        result = capture_diagnostics(probe, json.loads(output.read_text(encoding="utf-8")))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"status":result["status"],"joins":result["structured_native_joins"],"native_phase_values":stats(result["managed_native_diagnostics"]),"bracketed_prefix_evaluations":sum(bool(r.get("prompt_evaluated_tokens")) for r in result["managed_preparation_windows"]),"zero_eval_apply_restore_windows":sum(r.get("prefill_ms") == 0 for r in result["managed_preparation_windows"])}))
        if result["status"] != "completed_with_structured_native_diagnostics":
            raise RuntimeError("Retained-log structured extraction did not establish the required joined records")
        return
    if args.e4b is None:
        parser.error("A live managed probe requires the existing verified E4B model path")
    if probe.exists() or output.exists():
        raise FileExistsError("Preserve existing managed reference; choose a new attempt")
    probe.mkdir(parents=True)
    environment = {k: v for k, v in os.environ.items() if not any(term in k.upper() for term in ("API_KEY", "API_TOKEN", "ACCESS_TOKEN", "HF_TOKEN", "PASSPHRASE")) and not k.startswith(("PURIPULY_HEART_", "LLAMA_"))}
    for key, folder in (("LOCALAPPDATA", "Local"), ("APPDATA", "Roaming"), ("USERPROFILE", "Profile"), ("HOME", "Profile"), ("TEMP", "Temp"), ("TMP", "Temp")):
        directory = probe / folder
        directory.mkdir(exist_ok=True)
        environment[key] = str(directory)
    environment.update(PYTHONPATH=str(REPO / "src"), PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1", PURIPULY_HEART_PROMPTS_DIR=str(REPO / "prompts"), PURIPULY_HEART_SECRETS_PASSPHRASE="issue245-managed-reference-synthetic-only", PURIPULY_HEART_LLAMA_CPP_ROOT=str(probe / "runtime"))
    from puripuly_heart.core.local_translation.assets import InstalledGemmaManifest, e4b_gemma_spec, validate_gemma_install
    from puripuly_heart.config.settings_vnext.serialization import from_dict, to_json_text
    install = probe / "Local/puripuly-heart/models" / e4b_gemma_spec().install_dirname
    install.mkdir(parents=True)
    for asset in e4b_gemma_spec().assets:
        source = args.e4b.parent / asset.filename
        if not source.is_file():
            raise FileNotFoundError(f"Required existing E4B asset unavailable: {asset.filename}")
        shutil.copy2(source, install / asset.filename)
    (install / "installed-manifest.json").write_text(json.dumps(InstalledGemmaManifest.expected(e4b_gemma_spec()).to_dict(), indent=2), encoding="utf-8")
    validate_gemma_install(install, verify_checksums=True)
    runtime = probe / "runtime/vulkan"
    shutil.copytree(args.root / "runtimes/b10423", runtime)
    result = {"profile": profile, "attempt": args.attempt, "scope": "Unmodified actual managed E4B Vulkan owner/provider/prefix slots with stock b10423 command; isolated headless manual text, not audio E2E.", "controlled_boundary": "CPU executable deliberately not staged: any Vulkan fallback fails closed. Both pinned E4B model and installation-required MTP artifact validated before startup; drafter is not loaded by GPU command. Keyless inactive Soniox, owned UDP sink, redirected roots and probe-only encrypted secrets.", "commands": [], "translations": [], "events": [], "instances": [], "managed_servers": [], "preparation_windows": [], "verified_assets": [{"file": a.filename, "size_bytes": a.size_bytes, "sha256": a.sha256} for a in e4b_gemma_spec().assets], "native_stderr": "Unavailable: stock managed process factory directs stdout/stderr to DEVNULL. No wrapper or product replacement used.", "wall_timing_boundary": "Text-submit CLI launch through matched completed translation event; excludes prior explicit prefix preparation, audio collection/endpointing/STT and GUI rendering."}
    config = probe / "settings.json"
    sink = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sink.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
    sink.bind(("127.0.0.1", 0))
    sink.settimeout(0.2)
    port = sink.getsockname()[1]
    settings = from_dict({"intent": {"translation": {"model": "managed_gemma", "connection": "gpu", "gpu_device_id": "Vulkan0"}, "stt": {"provider": "soniox"}, "peer_stt": {"provider": "soniox"}, "secrets": {"backend": "encrypted_file", "encrypted_file_path": "probe-secrets.json"}, "telemetry": {"enabled": False}, "clipboard": {"auto_translate_enabled": False}, "languages": {"source_language": "ko", "target_language": "en", "secondary_target_language": ""}, "osc": {"connection_mode": "off", "host": "127.0.0.1", "port": port, "send_port": port, "activation_notice_enabled": False}}})
    config.write_text(to_json_text(settings), encoding="utf-8")
    cli = [sys.executable, "-m", "puripuly_heart.main", "cli", "--config", str(config), "--timeout", "180"]
    identity = subprocess.run([sys.executable, "-c", "import json,sys,puripuly_heart;print(json.dumps({'package':puripuly_heart.__file__,'python_version':sys.version}))"], env=environment, capture_output=True, text=True, encoding="utf-8", timeout=30)
    result["source_import_identity"] = json.loads(identity.stdout)
    result["source_import_identity"].update(interpreter="<APP_PYTHON>/python.exe", launcher_sha256=hashlib.sha256(Path(sys.executable).read_bytes()).hexdigest())
    stop = threading.Event()
    datagrams, samples, samplers = [], [], []
    active = {"pid": None, "sampler": None, "server": None, "phase": "app_start"}
    events = queue.Queue()
    observer = None
    host_active = False

    def public(value):
        if isinstance(value, dict):
            return {k: public(v) for k, v in value.items()}
        if isinstance(value, list):
            return [public(v) for v in value]
        return sanitize(value, args.root) if isinstance(value, str) else value

    def drain():
        while not stop.is_set():
            try:
                datagrams.append(sink.recv(65536))
            except socket.timeout:
                pass

    def collect():
        while not stop.is_set():
            pid = active["pid"]
            if pid is not None and active["server"] is None:
                try:
                    for child in psutil.Process(pid).children(recursive=True):
                        if child.name().lower() != "llama-server.exe":
                            continue
                        command = child.cmdline()
                        if Path(command[0]).resolve() != (runtime / "llama-server.exe").resolve():
                            continue
                        sampler = Sampler(child.pid)
                        samplers.append(sampler)
                        record = {"pid": child.pid, "parent_host_pid": pid, "create_time": child.create_time(), "command": command, "counter_errors": sampler.errors, "base": f"http://127.0.0.1:{command[command.index('--port')+1]}"}
                        result["managed_servers"].append(record)
                        active["server"] = (record, sampler)
                        break
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            item = {"time": time.perf_counter(), "phase": active["phase"], "app": active["sampler"].sample() if active["sampler"] else None, "server": active["server"][1].sample() if active["server"] else None}
            samples.append(item)
            stop.wait(0.25)

    def metrics():
        owned = active["server"]
        if owned is None:
            return None
        try:
            with urllib.request.urlopen(owned[0]["base"] + "/metrics", timeout=3) as response:
                text = response.read().decode()
            return {line.split()[0]: float(line.split()[1]) for line in text.splitlines() if line and not line.startswith("#") and len(line.split()) >= 2}
        except (OSError, ValueError):
            return None

    def invoke(*parts):
        started = time.perf_counter()
        completed = subprocess.run([*cli, *parts], env=environment, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=210)
        try:
            value = json.loads(completed.stdout)
        except ValueError:
            value = {"stdout": completed.stdout, "stderr": completed.stderr}
        if parts[:2] == ("settings", "choices"):
            value = {"choices": {k: v for k, v in value.get("choices", {}).items() if k == "languages" or k.startswith("translation.")}}
        if parts[:2] == ("models", "status"):
            value = {"gemma": value.get("gemma")}
        result["commands"].append({"arguments": list(parts), "seconds": time.perf_counter()-started, "returncode": completed.returncode, "result": value})
        if completed.returncode:
            diagnostic = value.get("transaction", {}).get("diagnostics", {})
            if parts[:2] != ("settings", "apply") or value.get("status") != "degraded" or not value.get("terminal") or diagnostic.get("component") != "provider.stt" or diagnostic.get("code") != "stt.provider_not_ready":
                raise RuntimeError(f"Supported CLI operation failed: {parts[:2]} status{completed.returncode}")
            result.setdefault("controlled_degraded_operations", []).append(value)
        return value

    def prepare(kind, *parts):
        active["phase"] = kind
        before = metrics()
        started = time.perf_counter()
        receipt = invoke(*parts)
        result["preparation_windows"].append({"kind": kind, "wall_seconds": time.perf_counter()-started, "metrics_before": before, "metrics_after": metrics(), "operation_id": receipt.get("operation_id")})
        return receipt

    def receipt_ids(value):
        if isinstance(value, dict):
            return ({value["turn_id"]} if isinstance(value.get("turn_id"), str) else set()) | set(value.get("child_turn_ids", [])) | set().union(*(receipt_ids(v) for v in value.values()))
        if isinstance(value, list):
            return set().union(*(receipt_ids(v) for v in value))
        return set()

    def observe(stream):
        for line in stream:
            try:
                event = json.loads(line)
                result["events"].append(event)
                events.put(event)
            except ValueError:
                pass

    drain_thread = threading.Thread(target=drain)
    collector = threading.Thread(target=collect)
    drain_thread.start()
    collector.start()
    try:
        for cycle in range(2):
            active["phase"] = "app_start"
            host_active = True
            invoke("app", "start", "--background")
            status = invoke("app", "status")
            result["instances"].append(status)
            if any(c.get("desired_active") or c.get("effective_active") or c.get("capture", {}).get("has_source") for c in status["channels"].values()) or status.get("microphone_test", {}).get("effective_active"):
                raise RuntimeError("Isolation capture guard failed")
            active["pid"] = status["host_pid"]
            active["sampler"] = Sampler(status["host_pid"])
            samplers.append(active["sampler"])
            for query in (("capabilities",), ("settings", "current"), ("settings", "choices"), ("capture", "status"), ("osc", "status"), ("overlay", "status")):
                invoke(*query)
            prepare("translation_enable_and_initial_prefix", "translation", "set", "on")
            invoke("models", "status")
            observer = subprocess.Popen([*cli, "events", "follow", "--topic", "translation", "--topic", "error", "--channel", "self", "--include-translations", "--after", "0"], env=environment, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8")
            observer_thread = threading.Thread(target=observe, args=(observer.stdout,))
            observer_thread.start()
            cases = [("ko", "en", "유나는 3시가 아니라 4시에 도착해요."), ("en", "ko", "Mina did not buy 12 tickets; she bought 2."), ("ja", "en", "田中さんにはまだパスワードを送らないでください。"), ("zh-CN", "en", "小林没有取消预约，只是把时间改到了星期五。"), ("ko", "en", "그 열쇠는 제 것이 아니라 민수 것이에요.")]
            for rep in range(3):
                for index, (source, target, text) in enumerate(cases):
                    current = invoke("settings", "current")
                    prepare("language_apply_prefix_prepare_or_restore", "settings", "apply", "--arguments", json.dumps({"changes": {"languages": {"source": source, "target": target, "secondary_target": ""}}}), "--expected-revision", str(current["revision"]))
                    persisted = invoke("settings", "current")
                    pair = persisted["settings"]["intent"]["languages"]
                    effective = invoke("app", "status")
                    if (pair["source_language"], pair["target_language"]) != (source, target) or not effective["translation_enabled"] or not effective["translation_runtime_ready"]:
                        raise RuntimeError("Persisted/effective managed translation state mismatch")
                    owned = active["server"]
                    if owned is None or "--device" not in owned[0]["command"] or owned[0]["command"][owned[0]["command"].index("--device")+1] == "none":
                        raise RuntimeError("No observed owned Vulkan model process")
                    with urllib.request.urlopen(owned[0]["base"] + "/slots", timeout=5) as response:
                        slots = json.load(response)
                    message = probe / f"input-{cycle}-{rep}-{index}.txt"
                    message.write_text(text, encoding="utf-8")
                    active["phase"] = "text_submit_to_completion"
                    before_metrics = metrics()
                    started = time.perf_counter()
                    receipt = invoke("text", "submit", "--file", str(message))
                    admitted = receipt_ids(receipt)
                    if not admitted:
                        raise RuntimeError("No admitted translation turn identity")
                    completed = None
                    deadline = time.monotonic() + 120
                    while time.monotonic() < deadline:
                        try:
                            event = events.get(timeout=0.25)
                        except queue.Empty:
                            continue
                        if event.get("topic") == "translation" and event.get("text") and event.get("utterance_id") in admitted:
                            completed = event
                            break
                        if event.get("topic") == "error":
                            diagnostic = event.get("error", {}).get("diagnostics", {})
                            if diagnostic.get("component") == "provider.stt" and diagnostic.get("code") == "stt.provider_not_ready":
                                continue
                            raise RuntimeError("Non-STT application error event")
                    if completed is None:
                        raise TimeoutError("Managed translation completion missing")
                    result["translations"].append({"cycle": cycle, "rep": rep, "pair": source+"->"+target, "input": text, "receipt": receipt, "event": completed, "completion_wall_seconds": time.perf_counter()-started, "metrics_before": before_metrics, "metrics_after": metrics(), "slots_before_submit": slots})
            if cycle == 0:
                current = invoke("settings", "current")
                prepare("persist_pair_before_restart", "settings", "apply", "--arguments", json.dumps({"changes": {"languages": {"source": "en", "target": "ko", "secondary_target": ""}}}), "--expected-revision", str(current["revision"]))
            active["phase"] = "app_stop_managed_release"
            owned = active["server"]
            invoke("app", "stop")
            host_active = False
            result.setdefault("host_release", []).append({"pid": active["pid"], "exited": not psutil.pid_exists(active["pid"])})
            if owned:
                owned[0]["exited_after_app_stop"] = not psutil.pid_exists(owned[0]["pid"])
            active.update(pid=None, sampler=None, server=None)
            observer.terminate()
            observer.wait(timeout=10)
            observer_thread.join(timeout=5)
            observer = None
        result["status"] = "completed"
    except Exception as error:
        result["status"] = "failed"
        result["error"] = f"{type(error).__name__}: {error}"
    finally:
        if host_active:
            try:
                invoke("app", "stop")
            except Exception:
                result["host_cleanup_limit"] = "Supported stop failed; inspect only owned instance"
        if observer is not None:
            observer.terminate()
            observer.wait(timeout=10)
        stop.set()
        collector.join()
        drain_thread.join()
        sink.close()
        for sampler in samplers:
            sampler.close()
    result["udp_sink"] = {"datagrams": len(datagrams), "bytes": sum(map(len, datagrams)), "destination": "Owned exclusive loopback sink held through managed teardown"}
    result["resource_sample_count"] = len(samples)
    result["resource_peaks"] = {owner: {key: max(row[owner][key] for row in samples if row.get(owner) and row[owner].get(key) is not None) for key in {k for row in samples if row.get(owner) for k in row[owner]} if key != "time" and any(row.get(owner) and row[owner].get(key) is not None for row in samples)} for owner in ("server", "app")}
    result = capture_diagnostics(probe, result)
    (probe / "samples.json").write_text(json.dumps(samples), encoding="utf-8")
    output.write_text(json.dumps(public(result), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"profile": profile, "status": result["status"], "error": result.get("error"), "completed_translations": len(result["translations"]), "structured_native_rows": len(result["managed_native_diagnostics"]), "host_release": result.get("host_release"), "servers": [{"pid": s["pid"], "exited": s.get("exited_after_app_stop")} for s in result["managed_servers"]]}))


if __name__ == "__main__":
    main()
