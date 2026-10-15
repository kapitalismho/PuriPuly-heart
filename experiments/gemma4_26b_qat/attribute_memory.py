from __future__ import annotations

import argparse
import ctypes
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time
import urllib.request

from run_matrix import REPO, CASES, request_translation, sanitize
from windows_memory import observe_memory
from windows_metrics import Sampler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--attempt", type=int, default=1)
    args = parser.parse_args()
    destination = REPO / "experiments/gemma4_26b_qat/results" / f"memory-attribution-attempt{args.attempt}.json"
    raw = args.root / "memory" / f"attempt{args.attempt}"
    if destination.exists() or raw.exists():
        raise FileExistsError("Preserve prior diagnostic attempt")
    raw.mkdir(parents=True)
    result = {"purpose":"Bounded owned n12 OFF/2048 ready and post-three-text memory attribution, not a new latency benchmark or broad sweep.","reviewed_candidate":"84f04d09b6235f07da5fb0248ec7571e75befcc7","baseline_commit":"7a8151be6114a950f062e2430bd36f899d57e4de","profiles":[]}
    for identity in ("new-ncpu12-off-swa-nocram-repair2", "new-ncpu12-cache2048-swa-nocram-repair2-control"):
        profile = json.loads((REPO / "experiments/gemma4_26b_qat/results" / f"{identity}.json").read_text(encoding="utf-8"))
        command = [value.replace("<ARTIFACT_ROOT>", str(args.root)).replace("<MODEL>/", str(args.root / "models")+os.sep) for value in profile["command"]]
        with socket.socket() as reserved:
            reserved.bind(("127.0.0.1", 0))
            port = reserved.getsockname()[1]
        command[command.index("--port")+1] = str(port)
        base = f"http://127.0.0.1:{port}"
        row = {"profile":identity,"command":[sanitize(v,args.root) for v in command],"requests":[],"snapshots":{}}
        process = sampler = collector = None
        stop = threading.Event()
        samples = []
        phase = ["load"]
        folder = raw / identity
        folder.mkdir()
        started = time.perf_counter()
        with (folder / "server.log").open("wb") as log:
            try:
                ctypes.WinDLL("kernel32", use_last_error=True).SetConsoleCtrlHandler(None, False)
                startup = subprocess.STARTUPINFO()
                startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                startup.wShowWindow = 0
                environment = {k:v for k,v in os.environ.items() if not k.startswith("LLAMA_") and k not in {"HF_TOKEN","HUGGING_FACE_HUB_TOKEN"}}
                process = subprocess.Popen(command, stdout=log, stderr=log, env=environment, creationflags=subprocess.CREATE_NEW_CONSOLE, startupinfo=startup)
                row["pid"] = process.pid
                sampler = Sampler(process.pid)
                def collect():
                    while not stop.is_set():
                        sample = sampler.sample()
                        sample["phase"] = phase[0]
                        samples.append(sample)
                        stop.wait(0.25)
                collector = threading.Thread(target=collect)
                collector.start()
                deadline = time.monotonic()+360
                while time.monotonic() < deadline:
                    if process.poll() is not None:
                        raise RuntimeError("Owned model exited before readiness")
                    try:
                        with urllib.request.urlopen(base+"/health",timeout=1) as response:
                            if response.status == 200:
                                break
                    except OSError:
                        pass
                    time.sleep(0.2)
                else:
                    raise TimeoutError("Owned model readiness timeout")
                row["cold_readiness_seconds"] = time.perf_counter()-started
                phase[0] = "ready_snapshot"
                row["snapshots"]["ready"] = observe_memory(process.pid)
                phase[0] = "three_frozen_synthetic_texts"
                for index, case in enumerate(CASES[:3]):
                    request = request_translation(base,case,index,"attribution_work",index,0.0)
                    request["started_seconds"] = request.pop("started_perf")-started
                    request["ended_seconds"] = request.pop("ended_perf")-started
                    row["requests"].append(request)
                phase[0] = "post_work_snapshot"
                row["snapshots"]["post_work"] = observe_memory(process.pid)
                row["status"] = "completed"
            except Exception as error:
                row["status"] = "failed"
                row["error"] = sanitize(f"{type(error).__name__}: {error}",args.root)
            finally:
                phase[0] = "shutdown"
                if process is not None and process.poll() is None:
                    subprocess.run([sys.executable, str(REPO / "experiments/gemma4_26b_qat/run_matrix.py"), "--signal", str(process.pid)], capture_output=True, timeout=10)
                    try:
                        process.wait(timeout=30)
                        row["shutdown"] = "graceful_console_ctrl_c"
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=10)
                        row["shutdown"] = "forced_after_timeout"
                if process is not None:
                    row["process_exit_code"] = process.returncode
                stop.set()
                if collector is not None:
                    collector.join()
                if sampler is not None:
                    sampler.close()
        (folder / "samples.json").write_text(json.dumps(samples),encoding="utf-8")
        row["sample_count"] = len(samples)
        row["placement_cache_log"] = [sanitize(line,args.root) for line in (folder/"server.log").read_text(encoding="utf-8",errors="replace").splitlines() if any(term in line for term in ("buffer size", "MoE cache", "llama_moe_cache:", "offloaded"))]
        result["profiles"].append(row)
        print(json.dumps({"profile":identity,"status":row["status"],"error":row.get("error"),"snapshot_groups":{k:v["groups"] for k,v in row["snapshots"].items()},"process_exit_code":row.get("process_exit_code")}),flush=True)
    result["limits"] = "Snapshot queries do not read/touch target pages. VA/working-set scan is sequential and bracketed by process+PDH counters, not atomic. GPU physical memory mappings cannot be labeled duplicate system DRAM; actual driver allocation flags/callsites unavailable. Diagnostic request timings are retained but not used to rank profiles."
    destination.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")


if __name__ == "__main__":
    main()
