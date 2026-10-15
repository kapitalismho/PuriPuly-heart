from __future__ import annotations

import ctypes as C
import time
from ctypes import wintypes as W


class ProcessMemory(C.Structure):
    _fields_ = [("cb", W.DWORD), ("PageFaultCount", W.DWORD)] + [
        (n, C.c_size_t)
        for n in (
            "PeakWorkingSetSize",
            "WorkingSetSize",
            "QuotaPeakPagedPoolUsage",
            "QuotaPagedPoolUsage",
            "QuotaPeakNonPagedPoolUsage",
            "QuotaNonPagedPoolUsage",
            "PagefileUsage",
            "PeakPagefileUsage",
            "PrivateUsage",
        )
    ]


class MemoryStatus(C.Structure):
    _fields_ = [("length", W.DWORD), ("load", W.DWORD)] + [
        (n, C.c_ulonglong)
        for n in (
            "total_physical",
            "available_physical",
            "total_pagefile",
            "available_pagefile",
            "total_virtual",
            "available_virtual",
            "available_extended",
        )
    ]


class CounterUnion(C.Union):
    _fields_ = [("doubleValue", C.c_double), ("largeValue", C.c_longlong)]


class CounterValue(C.Structure):
    _anonymous_ = ("value",)
    _fields_ = [("status", W.DWORD), ("value", CounterUnion)]


class CounterItem(C.Structure):
    _fields_ = [("name", W.LPWSTR), ("value", CounterValue)]


class Sampler:
    def __init__(self, pid: int, adapter: str = "luid_0x00000000_0x0000f60b_phys_0"):
        self.pid = pid
        self.adapter = adapter.lower()
        self.instances = {}
        self.kernel = C.WinDLL("kernel32", use_last_error=True)
        self.kernel.OpenProcess.restype = W.HANDLE
        self.kernel.OpenProcess.argtypes = [W.DWORD, W.BOOL, W.DWORD]
        self.kernel.CloseHandle.argtypes = [W.HANDLE]
        self.handle = self.kernel.OpenProcess(0x410, False, pid)
        self.psapi = C.WinDLL("psapi", use_last_error=True)
        self.psapi.GetProcessMemoryInfo.argtypes = [W.HANDLE, C.POINTER(ProcessMemory), W.DWORD]
        self.kernel.GetProcessTimes.argtypes = [
            W.HANDLE,
            C.POINTER(W.FILETIME),
            C.POINTER(W.FILETIME),
            C.POINTER(W.FILETIME),
            C.POINTER(W.FILETIME),
        ]
        self.pdh = C.WinDLL("pdh")
        self.pdh.PdhOpenQueryW.argtypes = [W.LPCWSTR, C.c_size_t, C.POINTER(W.HANDLE)]
        self.pdh.PdhAddEnglishCounterW.argtypes = [
            W.HANDLE,
            W.LPCWSTR,
            C.c_size_t,
            C.POINTER(W.HANDLE),
        ]
        self.pdh.PdhCollectQueryData.argtypes = [W.HANDLE]
        self.pdh.PdhGetFormattedCounterArrayW.argtypes = [
            W.HANDLE,
            W.DWORD,
            C.POINTER(W.DWORD),
            C.POINTER(W.DWORD),
            C.c_void_p,
        ]
        self.pdh.PdhCloseQuery.argtypes = [W.HANDLE]
        self.query = W.HANDLE()
        self.pdh.PdhOpenQueryW(None, 0, C.byref(self.query))
        self.counters = {}
        self.errors = {}
        for name, path in {
            "process_gpu_dedicated_bytes": r"\GPU Process Memory(*)\Dedicated Usage",
            "process_gpu_shared_bytes": r"\GPU Process Memory(*)\Shared Usage",
            "process_gpu_committed_bytes": r"\GPU Process Memory(*)\Total Committed",
            "device_gpu_dedicated_bytes": r"\GPU Adapter Memory(*)\Dedicated Usage",
            "device_gpu_shared_bytes": r"\GPU Adapter Memory(*)\Shared Usage",
            "device_gpu_committed_bytes": r"\GPU Adapter Memory(*)\Total Committed",
            "gpu_engine_percent": r"\GPU Engine(*)\Utilization Percentage",
            "system_pages_input_per_second": r"\Memory\Pages Input/sec",
            "system_page_reads_per_second": r"\Memory\Page Reads/sec",
        }.items():
            handle = W.HANDLE()
            code = self.pdh.PdhAddEnglishCounterW(self.query, path, 0, C.byref(handle))
            if code:
                self.errors[name] = hex(code & 0xFFFFFFFF)
            else:
                self.counters[name] = handle
        self.pdh.PdhCollectQueryData(self.query)

    def sample(self) -> dict:
        now = time.perf_counter()
        result = {"time": now}
        memory = ProcessMemory()
        memory.cb = C.sizeof(memory)
        if self.psapi.GetProcessMemoryInfo(self.handle, C.byref(memory), memory.cb):
            result.update(
                {
                    name: int(getattr(memory, name))
                    for name in (
                        "WorkingSetSize",
                        "PeakWorkingSetSize",
                        "PrivateUsage",
                        "PeakPagefileUsage",
                        "PageFaultCount",
                    )
                }
            )
        creation, exit_time, kernel, user = (W.FILETIME() for _ in range(4))
        if self.kernel.GetProcessTimes(
            self.handle, C.byref(creation), C.byref(exit_time), C.byref(kernel), C.byref(user)
        ):
            result["cpu_seconds"] = (
                (kernel.dwHighDateTime << 32)
                + kernel.dwLowDateTime
                + (user.dwHighDateTime << 32)
                + user.dwLowDateTime
            ) / 1e7
        status = MemoryStatus()
        status.length = C.sizeof(status)
        self.kernel.GlobalMemoryStatusEx(C.byref(status))
        result.update(
            system_available_physical_bytes=status.available_physical,
            system_committed_bytes=status.total_pagefile - status.available_pagefile,
            system_commit_limit_bytes=status.total_pagefile,
            system_commit_headroom_bytes=status.available_pagefile,
        )
        self.pdh.PdhCollectQueryData(self.query)
        for name, counter in self.counters.items():
            size, count = W.DWORD(), W.DWORD()
            code = self.pdh.PdhGetFormattedCounterArrayW(
                counter, 0x200, C.byref(size), C.byref(count), None
            )
            if not size.value:
                result[name] = None
                continue
            buffer = C.create_string_buffer(size.value)
            code = self.pdh.PdhGetFormattedCounterArrayW(
                counter, 0x200, C.byref(size), C.byref(count), buffer
            )
            if code:
                result[name] = None
                continue
            items = C.cast(buffer, C.POINTER(CounterItem))
            values = []
            labels = []
            for i in range(count.value):
                item = items[i]
                if item.value.status not in (0, 1):
                    continue
                label = item.name or ""
                if "gpu_" in name and self.adapter not in label.lower():
                    continue
                if name.startswith("process_") or name == "gpu_engine_percent":
                    if not label.startswith(f"pid_{self.pid}_"):
                        continue
                values.append(item.value.doubleValue)
                labels.append(label)
            self.instances[name] = sorted(set(self.instances.get(name, [])) | set(labels))
            result[name] = (
                (max(values) if name == "gpu_engine_percent" else sum(values)) if values else None
            )
        return result

    def close(self):
        self.pdh.PdhCloseQuery(self.query)
        self.kernel.CloseHandle(self.handle)
