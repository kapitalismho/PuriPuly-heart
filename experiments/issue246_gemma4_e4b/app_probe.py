from __future__ import annotations

import argparse
import json
import os
import secrets
import shutil
import socket
import subprocess
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path

from prepare import EXTERNAL, HERE, MODEL, RELEASES, ROOT, sanitize, save

sys.path.insert(0, str(ROOT / "src"))
from puripuly_heart.app.adapters.settings_vnext_canonical_persistence import SettingsVNextCanonicalPersistenceAdapter
from puripuly_heart.config.settings_vnext.defaults import new_settings_for_first_run
from puripuly_heart.config.settings_vnext.schema import SecretsIntent, with_telemetry_enabled


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", choices=RELEASES, required=True)
    parser.add_argument("--backend", choices=("cpu", "vulkan"), required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--cycles", type=int, default=2)
    parser.add_argument("--lazy", choices=("on", "off", "auto"))
    parser.add_argument("--foreign-cache-label")
    parser.add_argument("--cache-ram", type=int)
    parser.add_argument("--warm-turns", type=int, default=0)
    args = parser.parse_args()
    root = EXTERNAL / "apps" / args.label
    root.mkdir(parents=True, exist_ok=False)
    config = root / "settings.json"
    evidence = {"label": args.label, "release": args.release, "backend": args.backend, "runtime_identity": "process-local exact build/commit injection; maintained constants unchanged", "discovery_boundary": "PsutilCurrentUserProcessSnapshots.snapshots replaced with synthetic empty tuple; no scene/process inventory reads", "gui": "not_run", "commands": [], "cycles": []}
    udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp.bind(("127.0.0.1", 0))
    udp.settimeout(0.2)
    port = udp.getsockname()[1]
    datagrams = []
    stop_udp = threading.Event()
    def drain_udp():
        while not stop_udp.is_set():
            try:
                data, sender = udp.recvfrom(65536)
                datagrams.append({"monotonic": time.perf_counter(), "hex": data.hex(), "sender_loopback": sender[0] == "127.0.0.1"})
            except socket.timeout:
                pass
    udp_thread = threading.Thread(target=drain_udp, daemon=True)
    udp_thread.start()
    environment = dict(os.environ)
    removed = []
    for name in tuple(environment):
        upper = name.upper()
        if any(marker in upper for marker in ("API_KEY", "AUTH_TOKEN", "ACCESS_TOKEN", "CREDENTIAL", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")) or upper.startswith(("PURIPULY_HEART_NATIVE_", "LLAMA_ARG_")) or upper in {"PURIPULY_HEART_PROMPTS_DIR", "GGML_CPU_TILED_MM", "GGML_CPU_TILED_MM_FORCE"}:
            removed.append(name)
            environment.pop(name)
    for name in ("localappdata", "appdata", "profile", "tmp", "hf"):
        (root / name).mkdir()
    environment.update(PYTHONPATH=str(ROOT / "src"), PYTHONNOUSERSITE="1", LOCALAPPDATA=str(root / "localappdata"), APPDATA=str(root / "appdata"), USERPROFILE=str(root / "profile"), TEMP=str(root / "tmp"), TMP=str(root / "tmp"), HF_HOME=str(root / "hf"), HF_HUB_CACHE=str(root / "hf/hub"), HF_XET_CACHE=str(root / "hf/xet"), HF_HUB_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1", PURIPULY_HEART_SECRETS_PASSPHRASE=secrets.token_urlsafe(48), PURIPULY_HEART_LLAMA_CPP_ROOT=str(EXTERNAL / args.release), PP246_ARM=str(root), PP246_CONFIG=str(config), PP246_EXTERNAL=str(EXTERNAL), PP246_IDENTITY_BUILD=args.release, PP246_IDENTITY_COMMIT=RELEASES[args.release]["commit"], PYTHONIOENCODING="utf-8")
    if args.lazy:
        environment["PP246_LAZY"] = args.lazy
    if args.cache_ram is not None:
        environment["PP246_CACHE_RAM"] = str(args.cache_ram)
    model_dir = root / "localappdata/puripuly-heart/models" / MODEL.name
    model_dir.mkdir(parents=True)
    for path in MODEL.iterdir():
        if path.suffix == ".gguf":
            os.link(path, model_dir / path.name)
        elif path.name == "installed-manifest.json":
            shutil.copyfile(path, model_dir / path.name)
    profile = {"threads": 4, "threads_batch": 12, "draft_threads": 1}
    (root / "localappdata/puripuly-heart/llama_cpp_threads.json").write_text(json.dumps(profile), encoding="utf-8")
    settings = new_settings_for_first_run("en-US")
    settings = replace(settings, intent=replace(settings.intent, secrets=SecretsIntent("encrypted_file", "isolated-secrets.json"), translation=replace(settings.intent.translation, model="managed_gemma", connection="cpu" if args.backend == "cpu" else "gpu"), stt=replace(settings.intent.stt, provider="soniox"), peer_stt=replace(settings.intent.peer_stt, provider="soniox"), languages=replace(settings.intent.languages, source_language="en", target_language="ko", secondary_target_language=""), osc=replace(settings.intent.osc, connection_mode="off", host="127.0.0.1", port=port, send_port=port, activation_notice_enabled=False), clipboard=replace(settings.intent.clipboard, auto_translate_enabled=False)))
    SettingsVNextCanonicalPersistenceAdapter().persist(config, with_telemetry_enabled(settings, False))
    from puripuly_heart.app.wiring.wiring_secrets_factory import create_secret_store
    secret_store = create_secret_store(settings.intent.secrets, config_path=config, passphrase=environment["PURIPULY_HEART_SECRETS_PASSPHRASE"])
    secret_store.set("soniox_api_key", "synthetic-nonworking-" + secrets.token_urlsafe(24))
    evidence["isolation"] = {"settings_seeded_before_first_start": True, "secrets": "isolated encrypted_file, throwaway passphrase; synthetic nonworking Soniox construction token only; no keyring, verification or capture", "credential_environment_names_removed": removed, "models": "verified Q4/drafter hardlinks into isolated model installation; no automatic downloads", "hf_offline": True, "telemetry": False, "network": "fail-closed non-loopback Python DNS/connect/connect_ex guard", "osc": "mode off plus owned bound loopback UDP sink; no external output", "capture": "never enabled", "overlay_clipboard": "off", "threads_persisted": profile}
    if args.foreign_cache_label:
        source_cache = EXTERNAL / "apps" / args.foreign_cache_label / "localappdata/puripuly-heart/gemma-prefix-cache"
        destination_cache = root / "localappdata/puripuly-heart/gemma-prefix-cache"
        shutil.copytree(source_cache, destination_cache)
        evidence["foreign_cache_control"] = {"source_label": args.foreign_cache_label, "copied_filenames": sorted(p.name for p in destination_cache.iterdir()), "policy": "exact process-local runtime identity must make foreign files ineligible; never restore cross-revision KV"}
    host = follower = None
    host_output = None
    events = []
    event_condition = threading.Condition()
    def command(*arguments, input_text=None, allow_failure=False):
        argv = [sys.executable, "-m", "puripuly_heart.main", "cli", "--config", str(config), "--timeout", "120", *arguments]
        start = time.perf_counter()
        result = subprocess.run(argv, input=input_text, capture_output=True, encoding="utf-8", errors="replace", env=environment, cwd=ROOT, timeout=150)
        try:
            output = json.loads(result.stdout)
        except ValueError:
            output = {"unparsed_stdout": result.stdout}
        item = {"arguments": list(arguments), "exit_code": result.returncode, "wall_ms": (time.perf_counter() - start) * 1000, "output": output, "stderr": result.stderr}
        evidence["commands"].append(item)
        print(json.dumps({"label": args.label, "command": arguments, "exit": result.returncode, "status": output.get("status")}, ensure_ascii=True), flush=True)
        if result.returncode and not allow_failure:
            raise RuntimeError(f"CLI command failed: {arguments}: {output}: {result.stderr}")
        return output
    def read_events(process):
        for line in process.stdout:
            try:
                event = json.loads(line)
            except ValueError:
                continue
            with event_condition:
                events.append(event)
                event_condition.notify_all()
    def submit(text):
        before = len(events)
        result = command("text", "submit", "--stdin", input_text=text)
        if result.get("terminal") is not True or result.get("status") != "applied":
            raise RuntimeError("synthetic text did not receive terminal applied admission")
        ids = set(result.get("child_turn_ids", [])) | {result.get("turn_id")}
        def selected_events():
            return [event for event in events[before:] if event.get("topic") == "translation" and event.get("utterance_id") in ids and event.get("instance_id") == result.get("instance_id")]
        deadline = time.monotonic() + 45
        with event_condition:
            while not selected_events() and time.monotonic() < deadline:
                event_condition.wait(timeout=max(0, deadline - time.monotonic()))
        matching = selected_events()
        if not matching:
            raise RuntimeError("no completed translation event after admitted synthetic text")
        return {"receipt": result, "events": matching, "source": text}
    try:
        for cycle in range(args.cycles):
            ready = root / "ready.json"
            ready.unlink(missing_ok=True)
            cycle_trace = root / "trace.jsonl"
            if cycle_trace.exists():
                cycle_trace.rename(root / f"trace-cycle-{cycle - 1}.jsonl")
            host_output = (root / f"host-{cycle}.txt").open("w", encoding="utf-8")
            host = subprocess.Popen([sys.executable, str(HERE / "app_host.py")], env=environment, cwd=ROOT, stdout=host_output, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
            deadline = time.monotonic() + 90
            while not ready.exists() and time.monotonic() < deadline:
                if host.poll() is not None:
                    raise RuntimeError(f"host exited during startup: {host.returncode}")
                time.sleep(0.1)
            if not ready.exists():
                raise TimeoutError("isolated app readiness file absent")
            cycle_result = {"cycle": cycle, "ready": json.loads(ready.read_text(encoding="utf-8")), "host_pid": host.pid, "translations": []}
            evidence["cycles"].append(cycle_result)
            command("app", "status")
            if cycle == 0:
                for argv in (("capabilities",), ("settings", "choices"), ("settings", "current"), ("capture", "status"), ("osc", "status"), ("overlay", "status"), ("secrets", "presence")):
                    command(*argv)
            follower = subprocess.Popen([sys.executable, "-m", "puripuly_heart.main", "cli", "--config", str(config), "events", "follow", "--topic", "translation", "--channel", "self", "--include-translations"], env=environment, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8")
            event_thread = threading.Thread(target=read_events, args=(follower,), daemon=True)
            event_thread.start()
            time.sleep(0.5)
            command("translation", "set", "on")
            sequence = [("en", "ko", "Mina did not buy 12 tickets."), ("ko", "en", "지수는 오후 3시 15분에 와요. 늦지 마세요."), ("en", "ko", "The blue key is not mine."), ("ja", "en", "ユナは青い鍵を持っていません。"), ("de", "en", "Anna hat nicht 17, sondern 7 Karten."), ("en", "ko", "Alex will wait for 7 minutes, not 17."), ("ko", "en", "Alex, B-12 방이 아니라 B-21 방이에요."), ("en", "ko", "Mina did not buy 12 tickets.")]
            if cycle:
                sequence = sequence[:1]
            for source, target, text in sequence:
                command("settings", "apply", "--arguments", json.dumps({"changes": {"languages": {"source": source, "target": target, "secondary_target": ""}}}))
                cycle_result["translations"].append(submit(text))
            if cycle == 0:
                for index in range(args.warm_turns):
                    cycle_result["translations"].append({**submit(f"Mina did not buy {12 + index} tickets."), "phase": "warm_real_history"})
            command("query", "diagnostics.history")
            command("query", "diagnostics.health")
            command("models", "status")
            command("translation", "set", "off")
            command("app", "stop")
            host.wait(timeout=30)
            cycle_result["host_exit_code"] = host.returncode
            follower.terminate()
            follower.wait(timeout=10)
            follower = None
            event_thread.join(timeout=5)
            host_output.close()
            host_output = None
            host = None
            trace = [json.loads(line) for line in (root / "trace.jsonl").read_text(encoding="utf-8").splitlines()]
            cycle_result["trace"] = trace
            from measure import gpu_memory
            cycle_result["gpu_counters_after_native_exit"] = [{"pid": item["pid"], "observation": gpu_memory(item["pid"])} for item in trace if item["kind"] == "native_exit"]
            for native in sorted(root.glob("native-*.txt")):
                (HERE / "results" / f"{args.label}-cycle{cycle}-{native.name}").write_text(sanitize(native.read_text(encoding="utf-8")), encoding="utf-8")
            (HERE / "results" / f"{args.label}-cycle{cycle}-host.txt").write_text(sanitize((root / f"host-{cycle}.txt").read_text(encoding="utf-8", errors="replace")), encoding="utf-8")
    except BaseException as exc:
        evidence["failure"] = {"type": type(exc).__name__, "message": sanitize(str(exc))}
        raise
    finally:
        if follower and follower.poll() is None:
            follower.terminate()
            follower.wait(timeout=10)
        if host and host.poll() is None:
            try:
                command("app", "stop", allow_failure=True)
                host.wait(timeout=20)
            except Exception:
                host.terminate()
                host.wait(timeout=10)
                evidence["forced_host_cleanup"] = True
        if host_output:
            host_output.close()
        stop_udp.set()
        udp_thread.join(timeout=2)
        udp.close()
        evidence["udp_datagrams"] = datagrams
        evidence["events"] = events
        evidence["owned_tree_files"] = [str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()]
        if "failure" in evidence and (root / "trace.jsonl").exists():
            evidence["failure_trace"] = [json.loads(line) for line in (root / "trace.jsonl").read_text(encoding="utf-8").splitlines()]
            for native in root.glob("native-*.txt"):
                (HERE / "results" / f"{args.label}-failed-{native.name}").write_text(sanitize(native.read_text(encoding="utf-8")), encoding="utf-8")
        serialized = sanitize(json.dumps(evidence, ensure_ascii=False)).replace(str(root), "<ARM>")
        save(args.label + "-app.json", json.loads(serialized))
        for path in root.glob("host-*.txt"):
            (HERE / "results" / f"{args.label}-{path.name}").write_text(sanitize(path.read_text(encoding="utf-8", errors="replace")).replace(str(root), "<ARM>"), encoding="utf-8")


if __name__ == "__main__":
    main()
