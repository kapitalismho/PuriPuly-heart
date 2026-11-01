from __future__ import annotations

import importlib.metadata
import json
import platform
import subprocess
import sys
import threading
from pathlib import Path

import psutil


def sha256(path):
    import hashlib

    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def identity(root, worker):
    import puripuly_heart

    cpu = platform.processor()
    if sys.platform == "win32":
        import winreg

        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
        ) as key:
            cpu = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    memory = psutil.virtual_memory()
    versions = {}
    for name in ("numpy", "sherpa-onnx", "sherpa-onnx-core", "onnxruntime", "psutil"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    native = []
    for path in sorted(worker.parent.glob("*")):
        if path.suffix.lower() in (".exe", ".dll") or path.name == "asr-worker-runtime.json":
            native.append({"file": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)})
    inventory_path = worker.parent / "asr-worker-runtime.json"
    inventory = (
        json.loads(inventory_path.read_text(encoding="utf-8")) if inventory_path.exists() else None
    )
    if inventory is not None:
        inventory.pop("target_dir", None)
    return {
        "checkout": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "import_root": Path(puripuly_heart.__file__).resolve().relative_to(root).as_posix(),
        "python": platform.python_version(),
        "python_executable": (
            Path(sys.executable).resolve().relative_to(root).as_posix()
            if Path(sys.executable).resolve().is_relative_to(root)
            else "external-venv/Scripts/python.exe"
        ),
        "os": platform.platform(),
        "cpu": cpu,
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": psutil.cpu_count(),
        "ram_total_bytes": memory.total,
        "ram_available_bytes": memory.available,
        "versions": versions,
        "worker_build": native,
        "worker_runtime_inventory": inventory,
    }


class ResourceSampler:
    def __init__(self):
        self.process = psutil.Process()
        self.cpu = {}
        self.peaks = {}
        self.peak_tree_rss = 0
        self.errors = set()
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.sample()
        self.baseline_cpu = sum(self.cpu.values())
        self.thread.start()

    def sample(self):
        with self.lock:
            tree_rss = 0
            try:
                processes = [self.process] + self.process.children(recursive=True)
            except psutil.Error as error:
                self.errors.add(type(error).__name__)
                processes = [self.process]
            for process in processes:
                try:
                    cpu = process.cpu_times()
                    self.cpu[process.pid] = cpu.user + cpu.system
                    memory = process.memory_info()
                    tree_rss += memory.rss
                    peak = getattr(memory, "peak_wset", memory.rss)
                    self.peaks[process.pid] = max(self.peaks.get(process.pid, 0), peak)
                except psutil.Error as error:
                    self.errors.add(type(error).__name__)
            self.peak_tree_rss = max(self.peak_tree_rss, tree_rss)
            return sum(self.cpu.values()) - getattr(self, "baseline_cpu", 0)

    def _run(self):
        while not self.stop_event.wait(0.05):
            self.sample()

    def snapshot(self):
        cpu = self.sample()
        return {
            "cpu_seconds": cpu,
            "sampled_peak_tree_rss_bytes": self.peak_tree_rss,
            "sum_process_peak_working_set_bytes": sum(self.peaks.values()),
            "observed_processes": len(self.peaks),
            "sampling_interval_ms": 50,
            "resource_errors": sorted(self.errors),
        }

    def close(self):
        result = self.snapshot()
        self.stop_event.set()
        self.thread.join()
        return result
