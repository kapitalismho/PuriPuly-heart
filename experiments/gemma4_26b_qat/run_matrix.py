from __future__ import annotations

import argparse
import concurrent.futures
import ctypes
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

from windows_metrics import Sampler
from windows_memory import mapped_gguf_observation

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
os.environ["PURIPULY_HEART_PROMPTS_DIR"] = str(REPO / "prompts")
from puripuly_heart.config.prompts import get_default_prompt, render_translation_prompt_template
from puripuly_heart.providers.llm.messages import build_translation_user_message

CASES = [
    ("English", "Korean", "Mina did not buy 12 tickets; she bought 2.", ["2", "12"]),
    ("Korean", "English", "유나는 3시가 아니라 4시에 도착해요.", ["3", "4"]),
    ("Japanese", "English", "この薬は一日2回です。3回ではありません。", ["2", "3"]),
    ("Chinese (Simplified)", "English", "小林没有取消预约，只是把时间改到了星期五。", []),
    ("English", "Korean", "Please wait here. I will be back in 5 minutes.", ["5"]),
    ("Korean", "English", "그 열쇠는 제 것이 아니라 민수 것이에요.", []),
    ("Japanese", "English", "田中さんにはまだパスワードを送らないでください。", []),
    ("Chinese (Simplified)", "English", "如果下雨，我们就不去公园，在家见面。", []),
    ("English", "Korean", "I thought you were joking, but I was wrong.", []),
    ("Korean", "English", "아까 말한 파란 문 말고 빨간 문으로 와 주세요.", []),
    ("Japanese", "English", "料金は1500円で、飲み物は含まれていません。", ["1500"]),
    ("Chinese (Simplified)", "English", "我不是生你的气，我只是有点累。", []),
]
PROFILES = {
    "new-full": ("b11537", ["--n-gpu-layers", "99"]),
    "new-full-swa": ("b11537", ["--n-gpu-layers", "99", "--swa-full"]),
    "new-full-nocram": ("b11537", ["--n-gpu-layers", "99", "--cache-ram", "0"]),
    "new-cpu-moe-cache2048-swa": ("b11537", ["--n-gpu-layers", "99", "--cpu-moe", "--moe-cache-mib", "2048", "--swa-full"]),
    "new-full-swa-nocram": ("b11537", ["--n-gpu-layers", "99", "--swa-full", "--cache-ram", "0"]),
    "new-partial14-swa": ("b11537", ["--n-gpu-layers", "14", "--swa-full"]),
    "new-partial14-swa-nocram": ("b11537", ["--n-gpu-layers", "14", "--swa-full", "--cache-ram", "0"]),
    "new-cpu-moe-off-swa": ("b11537", ["--n-gpu-layers", "99", "--cpu-moe", "--swa-full"]),
    "new-cpu-moe-cache3072-swa": ("b11537", ["--n-gpu-layers", "99", "--cpu-moe", "--moe-cache-mib", "3072", "--swa-full"]),
    "new-cpu-moe-off-swa-nocram": ("b11537", ["--n-gpu-layers", "99", "--cpu-moe", "--swa-full", "--cache-ram", "0"]),
    "new-cpu-moe-cache3072-swa-nocram": ("b11537", ["--n-gpu-layers", "99", "--cpu-moe", "--moe-cache-mib", "3072", "--swa-full", "--cache-ram", "0"]),
    "new-partial14": ("b11537", ["--n-gpu-layers", "14"]),
    "new-partial20": ("b11537", ["--n-gpu-layers", "20"]),
    "new-cpu-moe-off": ("b11537", ["--n-gpu-layers", "99", "--cpu-moe"]),
    **{f"new-cpu-moe-cache{n}": ("b11537", ["--n-gpu-layers", "99", "--cpu-moe", "--moe-cache-mib", str(n)]) for n in (1024, 2048, 3072, 4096)},
    "new-ncpu24-off": ("b11537", ["--n-gpu-layers", "99", "--n-cpu-moe", "24"]),
    "new-ncpu24-cache2048": ("b11537", ["--n-gpu-layers", "99", "--n-cpu-moe", "24", "--moe-cache-mib", "2048"]),
    "new-ncpu24-cache2048-swa": ("b11537", ["--n-gpu-layers", "99", "--n-cpu-moe", "24", "--moe-cache-mib", "2048", "--swa-full"]),
    "new-ncpu24-cache2048-swa-nocram": ("b11537", ["--n-gpu-layers", "99", "--n-cpu-moe", "24", "--moe-cache-mib", "2048", "--swa-full", "--cache-ram", "0"]),
    "new-ncpu12-off": ("b11537", ["--n-gpu-layers", "99", "--n-cpu-moe", "12"]),
    "new-ncpu12-cache2048": ("b11537", ["--n-gpu-layers", "99", "--n-cpu-moe", "12", "--moe-cache-mib", "2048"]),
    "new-ncpu12-cache2048-swa": ("b11537", ["--n-gpu-layers", "99", "--n-cpu-moe", "12", "--moe-cache-mib", "2048", "--swa-full"]),
    "new-ncpu12-cache2048-swa-nocram": ("b11537", ["--n-gpu-layers", "99", "--n-cpu-moe", "12", "--moe-cache-mib", "2048", "--swa-full", "--cache-ram", "0"]),
    "new-ncpu12-off-swa-nocram": ("b11537", ["--n-gpu-layers", "99", "--n-cpu-moe", "12", "--swa-full", "--cache-ram", "0"]),
    "new-ncpu12-cache1024-swa-nocram": ("b11537", ["--n-gpu-layers", "99", "--n-cpu-moe", "12", "--moe-cache-mib", "1024", "--swa-full", "--cache-ram", "0"]),
    "new-ncpu12-cache3072-swa-nocram": ("b11537", ["--n-gpu-layers", "99", "--n-cpu-moe", "12", "--moe-cache-mib", "3072", "--swa-full", "--cache-ram", "0"]),
    "new-ncpu12-cache2048-swa-nocram-repair-control": ("b11537", ["--n-gpu-layers", "99", "--n-cpu-moe", "12", "--moe-cache-mib", "2048", "--swa-full", "--cache-ram", "0"]),
    "new-ncpu12-off-swa-nocram-repair2": ("b11537", ["--n-gpu-layers", "99", "--n-cpu-moe", "12", "--swa-full", "--cache-ram", "0"]),
    "new-ncpu12-cache1024-swa-nocram-repair2": ("b11537", ["--n-gpu-layers", "99", "--n-cpu-moe", "12", "--moe-cache-mib", "1024", "--swa-full", "--cache-ram", "0"]),
    "new-ncpu12-cache2048-swa-nocram-repair2-control": ("b11537", ["--n-gpu-layers", "99", "--n-cpu-moe", "12", "--moe-cache-mib", "2048", "--swa-full", "--cache-ram", "0"]),
    "old-full": ("b10423", ["--n-gpu-layers", "99"]),
    "e4b-product": ("b10423", ["--n-gpu-layers", "99"]),
}


def post(base: str, path: str, body: dict) -> dict:
    request = urllib.request.Request(base + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=300) as response:
        return json.load(response)


def request_translation(base: str, case: tuple, slot: int, phase: str, index: int, temperature: float) -> dict:
    source, target, text, numbers = case
    system = render_translation_prompt_template(get_default_prompt(), source_name=source, target_name=target)
    user = build_translation_user_message(text=text, context='- [self] "Let us meet at the station."\n- [peer] "Which entrance?"\n- [self] "The east entrance."\n- [peer] "All right."\n- [self] "I will wait nearby."')
    started = time.perf_counter()
    record = {"started_perf": started}
    prompt = post(base, "/apply-template", {"messages": [{"role": "system", "content": system}, {"role": "user", "content": user}], "add_generation_prompt": True})["prompt"]
    body = {"prompt": prompt, "n_predict": 96, "temperature": temperature, "seed": 42, "cache_prompt": True, "id_slot": slot, "stream": True}
    request = urllib.request.Request(base + "/completion", json.dumps(body).encode(), {"Content-Type": "application/json"})
    output, ttft, last = "", None, {}
    with urllib.request.urlopen(request, timeout=300) as response:
        for line in response:
            if line.startswith(b"data: "):
                raw = line[6:].strip()
                if raw == b"[DONE]":
                    continue
                payload = json.loads(raw)
                content = payload.get("content", "")
                if content and ttft is None:
                    ttft = time.perf_counter() - started
                output += content
                if payload.get("stop"):
                    last = payload
    elapsed = time.perf_counter() - started
    quality = {"nonempty": bool(output.strip()), "numbers_present": all(n in output.replace(",", "") for n in numbers), "no_reasoning": not any(t in output.lower() for t in ("<think", "<|channel>", "analysis:", "<thought")), "not_truncated": not last.get("truncated", False) and last.get("stop_type") != "limit"}
    return {**record, "ended_perf": time.perf_counter(), "phase": phase, "index": index, "slot": slot, "pair": f"{source}->{target}", "input": text, "output": output.strip(), "wall_seconds": elapsed, "ttft_seconds": ttft, "timings": last.get("timings"), "tokens_evaluated": last.get("tokens_evaluated"), "tokens_cached": last.get("tokens_cached"), "tokens_predicted": last.get("tokens_predicted"), "stop_type": last.get("stop_type"), "quality_checks": quality, "system_sha256": hashlib.sha256(system.encode()).hexdigest(), "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(), "temperature": temperature, "seed": 42, "max_output_tokens": 96, "status": "completed"}


def signal_console(pid: int):
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.FreeConsole()
    if not kernel.AttachConsole(pid):
        raise ctypes.WinError(ctypes.get_last_error())
    kernel.SetConsoleCtrlHandler(None, True)
    if not kernel.GenerateConsoleCtrlEvent(0, 0):
        raise ctypes.WinError(ctypes.get_last_error())
    time.sleep(0.5)
    kernel.FreeConsole()


def sanitize(text: str, root: Path) -> str:
    return text.replace(str(root), "<ARTIFACT_ROOT>").replace(str(root).replace("\\", "/"), "<ARTIFACT_ROOT>").replace(str(REPO), "<CHECKOUT>")


def classify_owned_safety_stop(result):
    if result.get("safety_cutoff"):
        result["status"] = "stopped_safety_cutoff"
        result["shutdown"] = "owned_memory_safety_cutoff"
    return result


def run(args, profile: str, *, load_mode="mmap", output_dir=None, environment_overrides=None, safety_policy=None):
    build, placement = PROFILES[profile]
    model = args.e4b if profile == "e4b-product" else args.root / "models" / "gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf"
    executable = args.root / "runtimes" / build / "llama-server.exe"
    output_dir = output_dir or args.root / "runs" / profile
    if output_dir.exists() or (args.results / f"{profile}.json").exists():
        raise FileExistsError(f"Preserve existing evidence; use a new profile identity/root: {profile}")
    output_dir.mkdir(parents=True)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    base = f"http://127.0.0.1:{port}"
    command = [str(executable), "--model", str(model), "--alias", "issue245", "--load-mode", load_mode, "--ctx-size", "4608", "--parallel", "3", "--batch-size", "512", "--ubatch-size", "512", "--threads", "4", "--threads-batch", "4", "--cache-type-k", "f16", "--cache-type-v", "f16", "--cache-prompt", "--reasoning", "off", "--reasoning-budget", "0", "--warmup", "--perf", "--metrics", "--no-webui", "--threads-http", "1", "--host", "127.0.0.1", "--port", str(port), "--prio", "-1", "--device", "Vulkan0", "--flash-attn", "on", "--fit", "off", *placement]
    command.extend(["--verbosity", "4"])
    result = {"profile": profile, "build": build, "baseline_commit": "7a8151be6114a950f062e2430bd36f899d57e4de", "command": [sanitize(x, args.root) if x != str(model) else f"<MODEL>/{model.name}" for x in command], "status": "started", "requests": [], "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    started = time.perf_counter()
    startup = subprocess.STARTUPINFO()
    startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startup.wShowWindow = 0
    environment = {k: v for k, v in os.environ.items() if not k.startswith("LLAMA_") and k not in {"HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"}}
    result["load_mode"] = load_mode
    if environment_overrides is not None:
        result["allowlisted_backend_environment_inherited"] = {key: environment.get(key) for key in sorted(environment_overrides)}
        for key, value in environment_overrides.items():
            if value is None:
                environment.pop(key, None)
            else:
                environment[key] = str(value)
        result["allowlisted_backend_environment_effective"] = {key: environment.get(key) for key in sorted(environment_overrides)}
    stop = threading.Event()
    samples = []
    phase = ["load"]
    sampler = None
    process = None
    thread = None
    with (output_dir / "server.log").open("wb") as log:
        try:
            ctypes.WinDLL("kernel32", use_last_error=True).SetConsoleCtrlHandler(None, False)
            process = subprocess.Popen(command, stdout=log, stderr=log, env=environment, creationflags=subprocess.CREATE_NEW_CONSOLE, startupinfo=startup)
            result["server_pid"] = process.pid
            sampler = Sampler(process.pid)
            result["counter_errors"] = sampler.errors
            result["gpu_adapter_luid"] = sampler.adapter
            if safety_policy is not None:
                result["owned_safety_policy"] = safety_policy
            def sampling():
                low_since = None
                while not stop.is_set():
                    sample = sampler.sample()
                    sample["phase"] = phase[0]
                    samples.append(sample)
                    if safety_policy is not None:
                        low = sample.get("system_available_physical_bytes", float("inf")) < safety_policy["minimum_available_physical_bytes"] or sample.get("system_commit_headroom_bytes", float("inf")) < safety_policy["minimum_commit_headroom_bytes"]
                        low_since = (low_since if low_since is not None else sample["time"]) if low else None
                        if low_since is not None and sample["time"] - low_since >= safety_policy["sustained_seconds"] and process.poll() is None:
                            result["safety_cutoff"] = {"elapsed_seconds": sample["time"]-started, "sustained_seconds": sample["time"]-low_since, "available_physical_bytes": sample.get("system_available_physical_bytes"), "commit_headroom_bytes": sample.get("system_commit_headroom_bytes"), "classification": "Owned conservative safety termination, NOT a runtime OOM"}
                            result["shutdown"] = "owned_memory_safety_cutoff"
                            process.kill()
                            process.wait(timeout=10)
                            return
                    stop.wait(0.25)
            thread = threading.Thread(target=sampling)
            thread.start()
            deadline = started + 360
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"Server exited before readiness: {process.returncode}")
                try:
                    with urllib.request.urlopen(base + "/health", timeout=1) as response:
                        if response.status == 200:
                            break
                except (OSError, urllib.error.HTTPError):
                    pass
                if time.perf_counter() > deadline:
                    raise TimeoutError("Readiness timeout after 360 seconds")
                time.sleep(0.2)
            result["cold_readiness_seconds"] = time.perf_counter() - started
            result["mapped_file_at_ready"] = mapped_gguf_observation(process.pid)
            with urllib.request.urlopen(base + "/slots") as response:
                result["slots"] = json.load(response)
            result["prefix_metadata"] = []
            for source, target, _, _ in CASES[:4]:
                system = render_translation_prompt_template(get_default_prompt(), source_name=source, target_name=target)
                wrapped = post(base, "/apply-template", {"messages": [{"role": "system", "content": system}, {"role": "user", "content": " "}], "add_generation_prompt": True})["prompt"]
                result["prefix_metadata"].append({"pair": source + "->" + target, "system_sha256": hashlib.sha256(system.encode()).hexdigest(), "raw_system_tokens": len(post(base, "/tokenize", {"content": system, "add_special": False, "parse_special": True})["tokens"]), "wrapped_system_blank_user_tokens": len(post(base, "/tokenize", {"content": wrapped, "add_special": False, "parse_special": True})["tokens"])})
            phase[0] = "first_prefix"
            for i in range(3):
                result["requests"].append(request_translation(base, CASES[i], i, "first_prefix", i, 0.0))
            phase[0] = "hot_same_pair"
            for rep in range(args.repetitions):
                result["requests"].append(request_translation(base, CASES[rep % 3 * 4], 0, "hot_same_pair", rep, 0.6))
            phase[0] = "pair_alternation"
            for rep in range(args.repetitions):
                for i in range(4):
                    result["requests"].append(request_translation(base, CASES[(rep % 3)*4+i], i % 3, "pair_alternation", rep*4+i, 0.6))
            phase[0] = "three_slots"
            for rep in range(args.repetitions):
                with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                    futures = [executor.submit(request_translation, base, CASES[(rep % 3)*4+i], i, "three_slots", rep*3+i, 0.6) for i in range(3)]
                    for future in futures:
                        result["requests"].append(future.result())
            result["mapped_file_after_repeated"] = mapped_gguf_observation(process.pid)
            result["status"] = "completed"
        except Exception as error:
            result["status"] = "failed"
            result["error"] = sanitize(f"{type(error).__name__}: {error}", args.root)
        finally:
            phase[0] = "shutdown"
            if process is not None and process.poll() is None:
                signaling = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--signal", str(process.pid)], capture_output=True, timeout=10)
                result["graceful_signal_exit"] = signaling.returncode
                try:
                    process.wait(timeout=30)
                    result["shutdown"] = "graceful_console_ctrl_c"
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
                    result["shutdown"] = "forced_after_graceful_timeout"
            if process is not None:
                result["process_exit_code"] = process.returncode
            stop.set()
            if thread is not None:
                thread.join()
            if sampler is not None:
                result["counter_instances"] = sampler.instances
                sampler.close()
    result = classify_owned_safety_stop(result)
    result["elapsed_seconds"] = time.perf_counter()-started
    if samples:
        result["sample_count"] = len(samples)
        result["memory_peaks"] = {key: max(s[key] for s in samples if s.get(key) is not None) for key in samples[0] if key not in {"time", "cpu_seconds", "phase"} and any(s.get(key) is not None for s in samples)}
        result["phase_memory_peaks"] = {p: {key: max(s[key] for s in samples if s["phase"] == p and s.get(key) is not None) for key in samples[0] if key not in {"time", "cpu_seconds", "phase"} and any(s["phase"] == p and s.get(key) is not None for s in samples)} for p in sorted({s["phase"] for s in samples})}
        result["cpu_seconds"] = max(s.get("cpu_seconds", 0) for s in samples)
        result["cpu_one_core_percent"] = result["cpu_seconds"] / result["elapsed_seconds"] * 100
        for s in samples:
            s["time"] -= started
    for request in result["requests"]:
        request["started_seconds"] = request.pop("started_perf") - started
        request["ended_seconds"] = request.pop("ended_perf") - started
    (output_dir / "samples.json").write_text(json.dumps(samples), encoding="utf-8")
    raw_log = (output_dir / "server.log").read_text(encoding="utf-8", errors="replace")
    result["placement_cache_log"] = [sanitize(line, args.root) for line in raw_log.splitlines() if any(term in line for term in ("MoE cache", "llama_moe_cache:", "offload", "buffer size", "n_ctx", "type  ", "type ", "model params", "model ftype", "Using device", "Vulkan", "CPU_REPACK"))]
    result["host_allocation_log"] = [sanitize(line, args.root) for line in raw_log.splitlines() if any(term in line for term in ("pinned", "import", "CPU_Mapped", "CPU_REPACK", "Vulkan_Host", "host buffer", "host memory", "host_ptr"))]
    result["server_log_sha256"] = hashlib.sha256(raw_log.encode()).hexdigest()
    (output_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    args.results.mkdir(parents=True, exist_ok=True)
    (args.results / f"{profile}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: result.get(k) for k in ("profile", "status", "error", "cold_readiness_seconds", "shutdown", "process_exit_code", "memory_peaks", "cpu_seconds", "sample_count")}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path)
    parser.add_argument("--e4b", type=Path)
    parser.add_argument("--results", type=Path, default=REPO / "experiments" / "gemma4_26b_qat" / "results")
    parser.add_argument("--profiles", nargs="+", choices=list(PROFILES), default=list(PROFILES))
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--signal", type=int)
    args = parser.parse_args()
    if args.signal:
        signal_console(args.signal)
        return
    for profile in args.profiles:
        run(args, profile)


if __name__ == "__main__":
    main()
