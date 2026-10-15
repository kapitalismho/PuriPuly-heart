from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
import psutil

from prepare import EXTERNAL, HERE, MODEL, ROOT, RELEASES, sanitize, save

sys.path.insert(0, str(ROOT / "src"))
os.environ["PURIPULY_HEART_PROMPTS_DIR"] = str(ROOT / "prompts")
from puripuly_heart.config.prompts import get_translation_prompt_template, render_translation_prompt_template
from puripuly_heart.core.local_translation.runtime_profile import LlamaCppThreadProfile, build_gemma_server_command
from puripuly_heart.core.orchestrator.channel_runtime import ChannelRuntime
from puripuly_heart.core.orchestrator.context import ContextResolver
from puripuly_heart.providers.llm.messages import build_translation_user_message

CASES = [
    ("en-ko-negation", "English", "Korean", "Mina did not buy 12 tickets.", ["12"], ["Mina", "미나"]),
    ("ko-en-time", "Korean", "English", "지수는 오후 3시 15분에 와요. 늦지 마세요.", ["3:15"], ["Jisu", "Ji-su", "Jisoo", "Ji Soo"]),
    ("ja-en-negation", "Japanese", "English", "ユナは青い鍵を持っていません。", [], ["Yuna"]),
    ("zh-en-number", "Chinese (Simplified)", "English", "林没有取消周五的两张票。", [], ["Lin"]),
    ("en-ja-name", "English", "Japanese", "Ask Kai to wait for 7 minutes, not 17.", ["7", "17"], ["Kai", "カイ"]),
    ("en-ko-decimal", "English", "Korean", "The red box costs 4.50 dollars, not 45.", ["45"], []),
    ("ko-en-tone", "Korean", "English", "괜찮아요. 지금은 도와주지 않아도 돼요.", [], []),
    ("ja-ko-time", "Japanese", "Korean", "明日は午前9時に駅で会いましょう。", ["9"], []),
    ("en-ko-context", "English", "Korean", "I meant the blue one, not the red one.", [], []),
    ("zh-en-quote", "Chinese (Simplified)", "English", "请告诉安娜：“我不会迟到。”", [], ["Anna"]),
    ("en-ja-fragment", "English", "Japanese", "Maybe later, after lunch.", [], []),
    ("ko-en-mixed", "Korean", "English", "Alex, B-12 방이 아니라 B-21 방이에요.", ["B-12", "B-21"], ["Alex"]),
]
TUNING = [
    ("tune-en-ko", "English", "Korean", "Please wait here for a moment.", [], []),
    ("tune-ko-en", "Korean", "English", "오늘은 조금 피곤하지만 괜찮아요.", [], []),
    ("tune-ja-en", "Japanese", "English", "次の部屋で友達を待っています。", [], []),
]
CONTEXT = "[self] We are meeting near the station.\n[peer] The blue box is beside the door.\n[self] Mina will arrive later.\n[peer] I have not bought the tickets yet.\n[self] Let us check the time first."
LEDGER_LOCK = threading.Lock()


def hashtext(text):
    return hashlib.sha256(text.encode()).hexdigest()


def prompt_for(case):
    return render_translation_prompt_template(get_translation_prompt_template(), source_name=case[1], target_name=case[2], input_channel="self", source_specified=True)


def literal_context(case, context):
    resolver = ContextResolver()
    now = resolver.clock.now()
    codes = {"English": "en", "Korean": "ko", "Japanese": "ja"}
    source, target = codes[case[1]], codes[case[2]]
    channels = {"self": ChannelRuntime(channel="self"), "peer": ChannelRuntime(channel="peer")}
    pairs = {"self": (source, target), "peer": (target, source)}
    for channel, runtime in channels.items():
        source_code, target_code = pairs[channel]
        runtime.remember_context("FILTERED_SYNTHETIC_STALE", timestamp=now - 61, source_language=source_code, target_language=target_code)
        runtime.remember_context("FILTERED_SYNTHETIC_WRONG_PAIR", timestamp=now - 1, source_language="wrong-source", target_language="wrong-target")
    for index, line in enumerate(context.splitlines()):
        channel, text = line[1:].split("] ", 1)
        source_code, target_code = pairs[channel]
        channels[channel].remember_context(text, timestamp=now - (5 - index), source_language=source_code, target_language=target_code)
    texts, formatted, mode = resolver.resolve_selected_for_request(runtime=channels["self"], other_runtime=channels["peer"], source_language=source, target_language=target, other_source_language=target, other_target_language=source)
    assert len(texts) == 5 and mode == "integrated" and "FILTERED_SYNTHETIC" not in formatted
    return formatted


def snapshot(proc):
    try:
        info = proc.memory_info()
        cpu = proc.cpu_times()
        return {"working_set": info.rss, "private_commit": info.private, "peak_working_set": info.peak_wset, "peak_private_commit": info.peak_pagefile, "cpu_seconds": cpu.user + cpu.system, "threads": proc.num_threads()}
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return None


def gpu_memory(pid):
    script = f"Get-CimInstance Win32_PerfFormattedData_GPUPerformanceCounters_GPUProcessMemory -Filter \"Name LIKE 'pid_{pid}_%'\" | Select-Object Name,DedicatedUsage,SharedUsage | ConvertTo-Json -Compress"
    result = subprocess.run(["powershell", "-NoProfile", "-Command", script], capture_output=True, text=True, timeout=20)
    if result.returncode == 0 and not result.stdout.strip():
        return {"status": "no_matching_process_counter_instances", "instances": 0, "dedicated_bytes": None, "shared_bytes": None}
    try:
        data = json.loads(result.stdout)
        items = data if isinstance(data, list) else [data]
        return {"dedicated_bytes": sum(int(x["DedicatedUsage"]) for x in items), "shared_bytes": sum(int(x["SharedUsage"]) for x in items), "instances": len(items)}
    except (ValueError, KeyError, TypeError):
        return {"status": "unavailable", "error": result.stderr.strip()}


class Server:
    def __init__(self, release, backend, label, extras=(), mtp=True):
        self.release, self.backend, self.label = release, backend, label
        self.directory = EXTERNAL / "runs" / label
        self.directory.mkdir(parents=True, exist_ok=False)
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        binary = next((EXTERNAL / release / backend).rglob("llama-server.exe"))
        self.command = list(build_gemma_server_command(executable=binary, install_dir=MODEL, backend="cpu" if backend == "cpu" else "gpu", port=port, threads_profile=LlamaCppThreadProfile(4, 12, 1), slot_save_path=self.directory))
        if not mtp:
            filtered = []
            index = 0
            while index < len(self.command):
                if self.command[index].startswith("--spec-"):
                    index += 2
                else:
                    filtered.append(self.command[index])
                    index += 1
            self.command = filtered
        self.command += list(extras) + ["--verbosity", "4"]
        self.client = httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=90, trust_env=False)
        self.log_path = self.directory / "producer.txt"
        self.log_handle = self.log_path.open("w", encoding="utf-8")
        self.started = time.perf_counter()
        self.process = subprocess.Popen(self.command, stdout=self.log_handle, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, creationflags=subprocess.CREATE_NO_WINDOW)
        self.proc = psutil.Process(self.process.pid)
        self.metadata = {"release": release, "commit": RELEASES[release]["commit"], "backend": backend, "label": label, "command": [sanitize(x) for x in self.command], "pid": self.process.pid, "utc": datetime.now(timezone.utc).isoformat(), "host_cpu_before_percent": psutil.cpu_percent(interval=0.2), "os_page_cache": "uncontrolled; model hash and earlier reads warm OS cache; process restart is not cold OS cache"}
        try:
            while time.perf_counter() - self.started < 120:
                if self.process.poll() is not None:
                    raise RuntimeError(f"server exited {self.process.returncode}")
                try:
                    if self.client.get("/health").status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                time.sleep(0.1)
            else:
                raise TimeoutError("server readiness timed out")
            self.metadata["readiness_ms"] = (time.perf_counter() - self.started) * 1000
            self.metadata["M1_model_ready"] = snapshot(self.proc)
            self.metadata["gpu_M1"] = gpu_memory(self.process.pid)
            props = self.client.get("/props").json()
            save(f"{label}-props.json", props)
            self.metadata["mapped_model_regions"] = [{"path": sanitize(m.path), "rss": m.rss} for m in self.proc.memory_maps(grouped=True) if m.path.endswith(".gguf")]
        except BaseException:
            self.close()
            raise

    def template(self, system, user):
        response = self.client.post("/apply-template", json={"messages": [{"role": "system", "content": system}, {"role": "user", "content": user}], "add_generation_prompt": True})
        response.raise_for_status()
        return response.json()["prompt"]

    def prefill(self, case, slot):
        system = prompt_for(case)
        formatted = self.template(system, " ")
        tokenized = self.client.post("/tokenize", json={"content": formatted, "add_special": False}).json()
        start = time.perf_counter()
        response = self.client.post("/completion", json={"prompt": formatted, "stream": False, "temperature": 0.6, "seed": 42, "cache_prompt": True, "id_slot": slot, "n_predict": 1})
        response.raise_for_status()
        data = response.json()
        self.metadata.setdefault("prefixes", []).append({"slot": slot, "pair": list(case[1:3]), "system_sha256": hashtext(system), "formatted_sha256": hashtext(formatted), "actual_prefix_tokens": len(tokenized["tokens"]), "wall_ms": (time.perf_counter() - start) * 1000, "timings": data.get("timings"), "memory": snapshot(self.proc)})
        (HERE / "results" / f"prompt-{case[1]}-{case[2]}.txt").write_text(system, encoding="utf-8")

    def request(self, case, slot, index, phase, temperature=0.6):
        before = snapshot(self.proc)
        start = time.perf_counter()
        row = {"arm": self.label, "request_index": index, "phase": phase, "case": case[0], "pair": list(case[1:3]), "slot": slot, "sampling": {"temperature": temperature, "seed": 42, "n_predict": 128}, "status": "error"}
        peaks = {"working_set": before["working_set"], "private_commit": before["private_commit"]}
        stop = threading.Event()
        def monitor():
            while not stop.wait(0.025):
                sample = snapshot(self.proc)
                if sample:
                    for key in peaks:
                        peaks[key] = max(peaks[key], sample[key])
        monitor_thread = threading.Thread(target=monitor, daemon=True)
        monitor_thread.start()
        try:
            context = CONTEXT
            if phase in {"tuning", "final", "soak", "contention", "literal"}:
                context = context.replace("[self] We are", f"[self] Step {index}: we are", 1)
            if phase == "literal":
                context = literal_context(case, context)
            row["history"] = {"representation": "real_resolver_literal" if phase == "literal" else "controlled_labelled", "entries": len(context.splitlines()), "sha256": hashtext(context)}
            user = build_translation_user_message(text=case[3], context=context, scene_participant_count=2)
            formatted = self.template(prompt_for(case), user)
            applied = time.perf_counter()
            body = {"prompt": formatted, "stream": True, "temperature": temperature, "seed": 42, "cache_prompt": True, "id_slot": slot, "n_predict": 128, "return_tokens": True}
            chunks, first, final = [], None, None
            with self.client.stream("POST", "/completion", json=body) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line.startswith("data: "):
                        continue
                    if line[6:] == "[DONE]":
                        continue
                    event = json.loads(line[6:])
                    content = event.get("content", "")
                    if content:
                        if first is None:
                            first = time.perf_counter()
                        chunks.append(content)
                    if event.get("stop"):
                        final = event
            end = time.perf_counter()
            output = "".join(chunks)
            if final is None:
                raise RuntimeError("missing terminal completion event")
            reasons = []
            if not output.strip():
                reasons.append("empty")
            if final.get("stopped_limit") or final.get("stop_type") == "limit":
                reasons.append("truncated")
            if any(tag in output for tag in ("<think>", "<|channel>", "<context>", "<input>", "�")):
                reasons.append("format_or_reasoning_leak")
            if any(expected not in output for expected in case[4]):
                reasons.append("number_marker_missing")
            if case[5] and not any(name.lower() in output.lower() for name in case[5]):
                reasons.append("name_marker_missing")
            row.update(status="completed" if not reasons else "quality_flag", flags=reasons, output=output, output_sha256=hashtext(output), wall_ms=(end - start) * 1000, template_ms=(applied - start) * 1000, ttft_ms=None if first is None else (first - start) * 1000, timings=final.get("timings"), terminal={k: final.get(k) for k in ("stop", "stop_type", "stopped_eos", "stopped_limit", "tokens_predicted", "tokens_evaluated", "tokens_cached", "generation_settings")})
        except Exception as exc:
            row["error"] = sanitize(f"{type(exc).__name__}: {exc}")
            row["wall_ms"] = (time.perf_counter() - start) * 1000
        finally:
            stop.set()
            monitor_thread.join()
        after = snapshot(self.proc)
        row.update(memory_after=after, sampled_request_peaks=peaks)
        if after:
            row["cpu_seconds"] = after["cpu_seconds"] - before["cpu_seconds"]
            row["cpu_percent_one_core"] = row["cpu_seconds"] / (row["wall_ms"] / 1000) * 100
        compact = row.get("terminal", {}).get("generation_settings")
        if compact:
            self.metadata.setdefault("effective_generation_settings", compact)
            row["terminal"].pop("generation_settings", None)
        with LEDGER_LOCK:
            with (HERE / "results" / "requests.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        return row

    def close(self):
        if not hasattr(self, "process"):
            return
        if self.process.poll() is None:
            self.metadata["M5_before_shutdown"] = snapshot(self.proc)
            self.metadata["gpu_M5"] = gpu_memory(self.process.pid)
            try:
                self.metadata["slots_end"] = self.client.get("/slots").json()
            except Exception:
                pass
            self.process.terminate()
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.metadata["forced_kill"] = True
                self.process.kill()
                self.process.wait(timeout=5)
        self.metadata["exit_code"] = self.process.returncode
        self.metadata["owned_process_exited"] = not self.proc.is_running()
        self.metadata["host_cpu_after_percent"] = psutil.cpu_percent(interval=0.2)
        self.client.close()
        self.log_handle.close()
        text = self.log_path.read_text(encoding="utf-8", errors="replace")
        (HERE / "results" / f"{self.label}-producer.txt").write_text(sanitize(text), encoding="utf-8")
        save(f"{self.label}-run.json", self.metadata)


def run(args):
    server = None
    try:
        server = Server(args.release, args.backend, args.label, args.extra, mtp=getattr(args, "mtp", True))
        cases = TUNING if args.phase == "tuning" else CASES[:3] if args.phase in {"final", "soak", "contention", "literal"} else CASES
        slots = {}
        lru = []
        for i in range(args.count):
            case = cases[i % len(cases)]
            pair = tuple(case[1:3])
            if pair not in slots:
                if len(slots) == 3:
                    expired = lru.pop(0)
                    slot = slots.pop(expired)
                else:
                    slot = len(slots)
                slots[pair] = slot
                server.prefill(case, slot)
            slot = slots[pair]
            if pair in lru:
                lru.remove(pair)
            lru.append(pair)
            row = server.request(case, slot, i, args.phase, args.temperature)
            print(json.dumps({k: row.get(k) for k in ("arm", "request_index", "case", "status", "wall_ms", "ttft_ms", "output")}, ensure_ascii=True), flush=True)
            if row["status"] == "error":
                break
            if args.phase == "soak":
                available = psutil.virtual_memory().available
                private = row.get("memory_after", {}).get("private_commit", 0)
                if available < 1536 * 1048576 or private > 14 * 1073741824:
                    server.metadata["resource_stop"] = {"completed_requests": i + 1, "requested_requests": args.count, "available_bytes": available, "private_commit_bytes": private, "reason": "owned experiment safety bound"}
                    break
    except Exception as exc:
        save(f"{args.label}-failure.json", {"status": "failed", "error": sanitize(f"{type(exc).__name__}: {exc}")})
        raise
    finally:
        if server:
            server.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", choices=RELEASES, required=True)
    parser.add_argument("--backend", choices=("cpu", "vulkan"), required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--phase", choices=("tuning", "quality", "final", "soak", "contention", "transitions", "literal"), default="quality")
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("extra", nargs="*")
    run(parser.parse_args())
