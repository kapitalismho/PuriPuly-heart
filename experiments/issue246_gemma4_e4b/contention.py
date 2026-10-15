from __future__ import annotations

import argparse
import subprocess
import sys
import threading
import time

import psutil

from measure import CASES, Server, snapshot
from prepare import save


def worker_trees(workers):
    records = []
    for worker in workers:
        parent = psutil.Process(worker.pid)
        tree = [parent] + parent.children(recursive=True)
        records.append({"launcher_pid": worker.pid, "processes": [{"pid": proc.pid, "creation_time": proc.create_time(), "name": proc.name(), "affinity": proc.cpu_affinity(), **snapshot(proc)} for proc in tree]})
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", required=True)
    parser.add_argument("--backend", choices=("cpu", "vulkan"), required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--lazy", choices=("on", "off", "auto"))
    parser.add_argument("--cache-ram", type=int)
    args = parser.parse_args()
    extras = ["--lazy-mode", args.lazy] if args.lazy else []
    if args.cache_ram is not None:
        extras += ["--cache-ram", str(args.cache_ram)]
    evidence = {"release": args.release, "backend": args.backend, "label": args.label, "controlled_load": {}, "primary_requests": []}
    workers = []
    primary = background = None
    stop = threading.Event()
    background_rows = []
    load_thread = None
    try:
        if args.backend == "cpu":
            code = "import time; end=time.perf_counter()+180; x=1\nwhile time.perf_counter()<end: x=(x*1664525+1013904223)%4294967296"
            for logical in range(6):
                worker = subprocess.Popen([sys.executable, "-c", code], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=subprocess.CREATE_NO_WINDOW)
                workers.append(worker)
                deadline = time.perf_counter() + 5
                while True:
                    parent = psutil.Process(worker.pid)
                    tree = [parent] + parent.children(recursive=True)
                    for proc in tree:
                        proc.cpu_affinity([logical])
                        proc.nice(psutil.NORMAL_PRIORITY_CLASS)
                    if any(sum(proc.cpu_times()[:2]) > 0 for proc in tree):
                        break
                    if time.perf_counter() >= deadline:
                        raise RuntimeError("Owned CPU load did not accumulate CPU time")
                    time.sleep(0.05)
            evidence["controlled_load"] = {"type": "six synthetic integer-loop process trees", "affinity_logical_ids": list(range(6)), "priority": "normal", "physical_core_mapping": "not asserted", "maximum_duration_s": 180, "launcher_boundary": "Windows venv redirector may be idle; all owned descendants are pinned and accounted"}
            evidence["worker_trees_after_start"] = worker_trees(workers)
        else:
            background = Server("b10423", "vulkan", args.label + "-background", ["--cache-ram", "0"])
            background.prefill(CASES[0], 0)
            def gpu_loop():
                for i in range(500):
                    if stop.is_set():
                        break
                    row = background.request(CASES[0], 0, i, "contention")
                    background_rows.append(row)
                    if row["status"] == "error":
                        break
            load_thread = threading.Thread(target=gpu_loop, daemon=True)
            load_thread.start()
            evidence["controlled_load"] = {"type": "second owned same-Q4 full-Vulkan translation server", "backend": "Vulkan0", "background_release": "b10423", "mtp": False, "background_cache_ram_mib": 0, "unrelated_user_gpu_workloads": "untouched", "claim": "reproducible inference contention, not a VRChat/game simulation"}
        evidence["load_start_monotonic"] = time.perf_counter()
        primary = Server(args.release, args.backend, args.label, extras)
        for slot, case in enumerate(CASES[:3]):
            primary.prefill(case, slot)
        evidence["measurement_start_monotonic"] = time.perf_counter()
        if workers:
            evidence["worker_trees_measurement_start"] = worker_trees(workers)
        for i in range(21):
            evidence["primary_requests"].append(primary.request(CASES[i % 3], i % 3, i, "contention"))
        evidence["measurement_end_monotonic"] = time.perf_counter()
        if workers:
            evidence["worker_trees_measurement_end"] = worker_trees(workers)
        evidence["primary_memory_end"] = snapshot(primary.proc)
    finally:
        stop.set()
        if load_thread:
            load_thread.join(timeout=90)
        if primary:
            primary.close()
        if background:
            background.close()
        evidence["background_completed_requests"] = len(background_rows)
        evidence["background_errors"] = [r for r in background_rows if r["status"] != "completed"]
        evidence["worker_trees_before_stop"] = worker_trees(workers) if workers else []
        for worker in workers:
            if worker.poll() is None:
                parent = psutil.Process(worker.pid)
                for proc in reversed(parent.children(recursive=True)):
                    proc.terminate()
                try:
                    worker.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    worker.terminate()
                    worker.wait(timeout=5)
        evidence["owned_load_stopped"] = all(worker.poll() is not None for worker in workers) and (load_thread is None or not load_thread.is_alive())
        evidence["load_end_monotonic"] = time.perf_counter()
        save(args.label + "-contention.json", evidence)


if __name__ == "__main__":
    main()
