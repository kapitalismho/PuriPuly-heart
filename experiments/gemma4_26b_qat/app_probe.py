from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import queue
import socket
import subprocess
import sys
import threading
import time
import urllib.request

import psutil
from run_matrix import REPO, sanitize
from windows_metrics import Sampler

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--e4b", required=True, type=Path)
    parser.add_argument("--attempt", type=int, default=1)
    args = parser.parse_args()
    probe = args.root / "app" / f"{args.profile}-attempt{args.attempt}"
    probe.mkdir(parents=True, exist_ok=True)
    result = {"profile": args.profile, "scope": "actual unmodified source app using user-managed local_llm/ollama endpoint; not managed runtime integration", "commands": [], "translations": [], "events": [], "instances": []}
    result["attempt"] = args.attempt
    result["controlled_boundary"] = "Inactive, intentionally keyless Soniox STT may degrade language runtime application; retain exact receipts. Manual translation only, not ASR/full live-pipeline readiness."
    stop = threading.Event()
    sink = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sink.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
    sink.bind(("127.0.0.1", 0))
    sink.settimeout(0.2)
    sink_port = sink.getsockname()[1]
    datagrams = []
    def drain():
        while not stop.is_set():
            try:
                datagrams.append(sink.recv(65536))
            except socket.timeout:
                pass
    sink_thread = threading.Thread(target=drain, daemon=True)
    sink_thread.start()
    environment = {k:v for k,v in os.environ.items() if not any(term in k.upper() for term in ("API_KEY", "API_TOKEN", "ACCESS_TOKEN", "HF_TOKEN", "PASSPHRASE")) and not k.startswith("PURIPULY_HEART_") and not k.startswith("LLAMA_")}
    for key, folder in (("LOCALAPPDATA", "Local"), ("APPDATA", "Roaming"), ("USERPROFILE", "Profile"), ("HOME", "Profile"), ("TEMP", "Temp"), ("TMP", "Temp")):
        path = probe / folder
        path.mkdir(exist_ok=True)
        environment[key] = str(path)
    environment.update(PYTHONPATH=str(REPO / "src"), PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8", PURIPULY_HEART_PROMPTS_DIR=str(REPO / "prompts"), PURIPULY_HEART_SECRETS_PASSPHRASE="issue245-isolated-synthetic-probe-only")
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    base = f"http://127.0.0.1:{port}"
    profile = json.loads((REPO / "experiments" / "gemma4_26b_qat" / "results" / f"{args.profile}.json").read_text(encoding="utf-8"))
    command = [part.replace("<ARTIFACT_ROOT>", str(args.root)).replace("<MODEL>/", str(args.root / "models") + os.sep) for part in profile["command"]]
    command[command.index("--port")+1] = str(port)
    if args.profile == "e4b-product":
        command[command.index("--model")+1] = str(args.e4b)
    command.extend(["--n-predict", "96", "--seed", "42"])
    result["server_command"] = [sanitize(x, args.root) if x != str(args.e4b) else "<E4B_MODEL>" for x in command]
    config = probe / "settings.json"
    from puripuly_heart.config.settings_vnext.serialization import from_dict, to_json_text
    settings = from_dict({"intent": {"translation": {"model": "local_llm", "connection": "ollama"}, "local_llm": {"backend": "ollama", "base_url": base + "/v1", "model": "issue245", "extra_body": {"reasoning_effort": "none", "temperature": 0.6}}, "stt": {"provider": "soniox"}, "peer_stt": {"provider": "soniox"}, "secrets": {"backend": "encrypted_file", "encrypted_file_path": "probe-secrets.json"}, "telemetry": {"enabled": False}, "clipboard": {"auto_translate_enabled": False}, "languages": {"source_language": "ko", "target_language": "en", "secondary_target_language": ""}, "osc": {"connection_mode": "off", "host": "127.0.0.1", "port": sink_port, "send_port": sink_port, "activation_notice_enabled": False}}})
    if config.exists():
        raise RuntimeError("Use a new app probe directory rather than overwriting existing state")
    config.write_text(to_json_text(settings), encoding="utf-8")
    cli = [sys.executable, "-m", "puripuly_heart.main", "cli", "--config", str(config), "--timeout", "180"]
    import_probe = subprocess.run([sys.executable, "-c", "import json,sys,puripuly_heart,puripuly_heart.runtime_layout as r; print(json.dumps({'interpreter':sys.executable,'package':puripuly_heart.__file__,'layout_module':r.__file__}))"], env=environment, capture_output=True, text=True, encoding="utf-8", timeout=30)
    result["source_import_identity"] = json.loads(import_probe.stdout)
    result["source_import_identity"]["interpreter"] = "<APP_PYTHON>/python.exe"
    result["source_import_identity"]["python_version"] = sys.version
    result["source_import_identity"]["launcher_sha256"] = hashlib.sha256(Path(sys.executable).read_bytes()).hexdigest()
    def public_value(value):
        if isinstance(value, dict):
            return {key: public_value(child) for key, child in value.items()}
        if isinstance(value, list):
            return [public_value(child) for child in value]
        if isinstance(value, str):
            return sanitize(value, args.root)
        return value
    def invoke(*parts):
        started = time.perf_counter()
        completed = subprocess.run([*cli, *parts], env=environment, capture_output=True, encoding="utf-8", errors="replace", timeout=210)
        try:
            value = json.loads(completed.stdout)
        except ValueError:
            value = {"stdout": completed.stdout, "stderr": completed.stderr}
        if parts[:2] == ("settings", "choices"):
            value = {key: value[key] for key in ("instance_id", "revision", "fields", "field_schemas") if key in value} | {"choices": {key: child for key, child in value.get("choices", {}).items() if key == "languages" or key.startswith("translation.")}}
        if parts[:2] == ("settings", "current"):
            intent = value.get("settings", {}).get("intent", {})
            value = {key: value[key] for key in ("instance_id", "revision") if key in value} | {"settings": {"intent": {key: intent[key] for key in ("translation", "local_llm", "languages", "telemetry", "clipboard", "osc") if key in intent}}}
        result["commands"].append({"arguments": [sanitize(x, args.root) for x in parts], "returncode": completed.returncode, "seconds": time.perf_counter()-started, "result": value})
        if completed.returncode:
            diagnostic = value.get("transaction", {}).get("diagnostics", {})
            controlled_stt = parts[:2] == ("settings", "apply") and value.get("terminal") is True and value.get("status") == "degraded" and diagnostic.get("component") == "provider.stt" and diagnostic.get("code") == "stt.provider_not_ready"
            if not controlled_stt:
                raise RuntimeError(f"CLI {parts[:2]} failed: {completed.returncode}: {sanitize(completed.stderr, args.root)}")
            result.setdefault("controlled_degraded_operations", []).append(value)
        return value
    def receipt_ids(value):
        found = set()
        if isinstance(value, dict):
            if isinstance(value.get("turn_id"), str):
                found.add(value["turn_id"])
            found.update(value.get("child_turn_ids", []))
            for child in value.values():
                found.update(receipt_ids(child))
        elif isinstance(value, list):
            for child in value:
                found.update(receipt_ids(child))
        return found
    server = observer = None
    host_active = False
    sampling_stop = threading.Event()
    app_samplers = []
    collector = None
    server_sampler = app_sampler = None
    samples = []
    events = queue.Queue()
    startup = subprocess.STARTUPINFO()
    startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startup.wShowWindow = 0
    with (probe / "server.log").open("wb") as server_log:
        try:
            import ctypes
            ctypes.WinDLL("kernel32", use_last_error=True).SetConsoleCtrlHandler(None, False)
            server = subprocess.Popen(command, stdout=server_log, stderr=server_log, env=environment, creationflags=subprocess.CREATE_NEW_CONSOLE, startupinfo=startup)
            server_sampler = Sampler(server.pid)
            def collect():
                while not sampling_stop.is_set():
                    active_app = app_sampler
                    samples.append({"time": time.perf_counter(), "server": server_sampler.sample(), "app": active_app.sample() if active_app is not None else None})
                    sampling_stop.wait(0.25)
            collector = threading.Thread(target=collect)
            collector.start()
            deadline = time.monotonic() + 360
            while time.monotonic() < deadline:
                if server.poll() is not None:
                    raise RuntimeError("Standalone finalist server exited before readiness")
                try:
                    with urllib.request.urlopen(base + "/health", timeout=1) as response:
                        if response.status == 200:
                            break
                except OSError:
                    pass
                time.sleep(0.2)
            else:
                raise TimeoutError("Standalone finalist server readiness timeout")
            for cycle in range(2):
                host_active = True
                invoke("app", "start", "--background")
                caps = invoke("capabilities")
                (probe / f"capabilities-{cycle}.json").write_text(json.dumps(caps, ensure_ascii=False, indent=2), encoding="utf-8")
                status = invoke("app", "status")
                result["instances"].append(status)
                if any(channel.get("desired_active") or channel.get("effective_active") for channel in status["channels"].values()):
                    raise RuntimeError("Isolation guard: capture was unexpectedly active")
                if status.get("microphone_test", {}).get("effective_active"):
                    raise RuntimeError("Isolation guard: microphone test was unexpectedly active")
                host_pid = status["host_pid"]
                app_sampler = Sampler(host_pid)
                app_samplers.append(app_sampler)
                for query in (("settings", "current"), ("settings", "choices"), ("capture", "status"), ("osc", "status"), ("overlay", "status"), ("query", "providers.status")):
                    invoke(*query)
                invoke("translation", "set", "on")
                observer = subprocess.Popen([*cli, "events", "follow", "--topic", "translation", "--topic", "error", "--channel", "self", "--include-translations", "--after", "0"], env=environment, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8")
                def observe(stream):
                    for line in stream:
                        try:
                            event = json.loads(line)
                            result["events"].append(event)
                            events.put(event)
                        except ValueError:
                            pass
                observer_thread = threading.Thread(target=observe, args=(observer.stdout,))
                observer_thread.start()
                cases = [("ko", "en", "유나는 3시가 아니라 4시에 도착해요."), ("en", "ko", "Mina did not buy 12 tickets; she bought 2."), ("ja", "en", "田中さんにはまだパスワードを送らないでください。"), ("zh-CN", "en", "小林没有取消预约，只是把时间改到了星期五。"), ("ko", "en", "그 열쇠는 제 것이 아니라 민수 것이에요.")]
                for rep in range(3):
                    for index, (source, target, text) in enumerate(cases):
                        current = invoke("settings", "current")
                        invoke("settings", "apply", "--arguments", json.dumps({"changes": {"languages": {"source": source, "target": target, "secondary_target": ""}}}), "--expected-revision", str(current["revision"]))
                        persisted = invoke("settings", "current")
                        actual_pair = persisted["settings"]["intent"]["languages"]
                        if (actual_pair["source_language"], actual_pair["target_language"]) != (source, target):
                            raise RuntimeError("Persisted language pair does not match requested pair")
                        effective = invoke("app", "status")
                        if not effective["translation_enabled"] or not effective["translation_runtime_ready"]:
                            raise RuntimeError("Effective translation state is not enabled/ready")
                        message = probe / f"input-{cycle}-{rep}-{index}.txt"
                        message.write_text(text, encoding="utf-8")
                        before = time.perf_counter()
                        receipt = invoke("text", "submit", "--file", str(message))
                        admitted_ids = receipt_ids(receipt)
                        if not admitted_ids:
                            raise RuntimeError("Text receipt did not establish admitted turn identity")
                        deadline = time.monotonic() + 120
                        completed = None
                        while time.monotonic() < deadline:
                            try:
                                event = events.get(timeout=0.25)
                            except queue.Empty:
                                continue
                            if event.get("topic") == "translation" and event.get("text") and event.get("utterance_id") in admitted_ids:
                                completed = event
                                break
                            if event.get("topic") == "error":
                                diagnostic = event.get("error", {}).get("diagnostics", {})
                                if diagnostic.get("component") == "provider.stt" and diagnostic.get("code") == "stt.provider_not_ready":
                                    result.setdefault("controlled_stt_events", []).append(event)
                                    continue
                                raise RuntimeError("Application emitted non-STT error event")
                        if completed is None:
                            raise TimeoutError("No completed translation content event")
                        result["translations"].append({"cycle": cycle, "rep": rep, "pair": source + "->" + target, "input": text, "receipt": receipt, "event": completed, "completion_wall_seconds": time.perf_counter()-before})
                if cycle == 0:
                    current = invoke("settings", "current")
                    invoke("settings", "apply", "--arguments", json.dumps({"changes": {"languages": {"source": "en", "target": "ko", "secondary_target": ""}}}), "--expected-revision", str(current["revision"]))
                    result["persisted_pair_before_restart"] = invoke("settings", "current")
                invoke("app", "stop")
                host_active = False
                result.setdefault("host_release", []).append({"pid": host_pid, "exited": not psutil.pid_exists(host_pid)})
                app_sampler = None
                observer.terminate()
                observer.wait(timeout=10)
                observer_thread.join(timeout=5)
                observer = None
            result["status"] = "completed"
        except Exception as error:
            result["status"] = "failed"
            result["error"] = sanitize(f"{type(error).__name__}: {error}", args.root)
        finally:
            if host_active:
                try:
                    invoke("app", "stop")
                except Exception:
                    result["host_cleanup_limit"] = "Supported stop failed; inspect owned instance before continuing."
            if observer is not None:
                observer.terminate()
                observer.wait(timeout=10)
            if server is not None and server.poll() is None:
                subprocess.run([sys.executable, str(REPO / "experiments" / "gemma4_26b_qat" / "run_matrix.py"), "--signal", str(server.pid)], capture_output=True, timeout=10)
                try:
                    server.wait(timeout=30)
                    result["server_shutdown"] = "graceful_console_ctrl_c"
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait(timeout=10)
                    result["server_shutdown"] = "forced_after_timeout"
            if server is not None:
                result["server_exit_code"] = server.returncode
            sampling_stop.set()
            if collector is not None:
                collector.join()
            if server_sampler is not None:
                server_sampler.close()
            for owned_sampler in app_samplers:
                owned_sampler.close()
            stop.set()
            sink_thread.join()
            sink.close()
    result["udp_sink"] = {"datagrams": len(datagrams), "bytes": sum(map(len, datagrams)), "destination": "owned exclusive loopback sink held through teardown"}
    result["resource_sample_count"] = len(samples)
    result["resource_peaks"] = {owner: {key: max(row[owner][key] for row in samples if row.get(owner) and row[owner].get(key) is not None) for key in {k for row in samples if row.get(owner) for k in row[owner]} if key != "time" and any(row.get(owner) and row[owner].get(key) is not None for row in samples)} for owner in ("server", "app")}
    log = (probe / "server.log").read_text(encoding="utf-8", errors="replace")
    result["placement_cache_log"] = [sanitize(line, args.root) for line in log.splitlines() if any(x in line for x in ("llama_moe_cache", "MoE cache", "Vulkan", "offload", "buffer size"))]
    (probe / "samples.json").write_text(json.dumps(samples), encoding="utf-8")
    output = REPO / "experiments" / "gemma4_26b_qat" / "results" / f"app-{args.profile}-attempt{args.attempt}.json"
    output.write_text(json.dumps(public_value(result), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"profile": args.profile, "status": result["status"], "error": result.get("error"), "completed_translations": len(result["translations"]), "host_release": result.get("host_release"), "server_shutdown": result.get("server_shutdown"), "udp_sink": result["udp_sink"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
