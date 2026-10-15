from __future__ import annotations

import ctypes as C
from ctypes import wintypes as W
from collections import defaultdict
from pathlib import Path
import time

from windows_metrics import Sampler

PAGE = 4096
MEM_COMMIT = 0x1000
MEM_FREE = 0x10000
TYPES = {0x20000: "MEM_PRIVATE", 0x40000: "MEM_MAPPED", 0x1000000: "MEM_IMAGE"}


class BasicInfo(C.Structure):
    _fields_ = [("BaseAddress", C.c_void_p), ("AllocationBase", C.c_void_p), ("AllocationProtect", W.DWORD), ("PartitionId", W.WORD), ("RegionSize", C.c_size_t), ("State", W.DWORD), ("Protect", W.DWORD), ("Type", W.DWORD)]


class RegionInfo(C.Structure):
    _fields_ = [("AllocationBase", C.c_void_p), ("AllocationProtect", W.DWORD), ("Flags", W.DWORD), ("RegionSize", C.c_size_t), ("CommitSize", C.c_size_t)]


class WorkingSetEntry(C.Structure):
    _fields_ = [("VirtualAddress", C.c_void_p), ("Flags", C.c_size_t)]


def observe_memory(pid: int, *, gguf_only: bool = False) -> dict:
    started = time.perf_counter()
    sampler = Sampler(pid)
    kernel = sampler.kernel
    psapi = sampler.psapi
    kernel.VirtualQueryEx.argtypes = [W.HANDLE, C.c_void_p, C.POINTER(BasicInfo), C.c_size_t]
    kernel.VirtualQueryEx.restype = C.c_size_t
    psapi.GetMappedFileNameW.argtypes = [W.HANDLE, C.c_void_p, W.LPWSTR, W.DWORD]
    psapi.GetMappedFileNameW.restype = W.DWORD
    psapi.QueryWorkingSetEx.argtypes = [W.HANDLE, C.c_void_p, W.DWORD]
    psapi.QueryWorkingSetEx.restype = W.BOOL
    memory = C.WinDLL("kernelbase", use_last_error=True)
    query_region = memory.QueryVirtualMemoryInformation
    query_region.argtypes = [W.HANDLE, C.c_void_p, C.c_int, C.c_void_p, C.c_size_t, C.POINTER(C.c_size_t)]
    query_region.restype = W.BOOL
    result = {"pid": pid, "pointer_bytes": C.sizeof(C.c_void_p), "page_bytes": PAGE, "basic_info_struct_bytes": C.sizeof(BasicInfo), "region_info_struct_bytes": C.sizeof(RegionInfo), "method": "VirtualQueryEx64; QueryVirtualMemoryInformation MemoryRegionInfo once per AllocationBase; GetMappedFileNameW basename only; QueryWorkingSetEx batched Valid/Shared bits, no reads/touches/locks.", "errors": [], "allocation_info_failure_count": 0, "working_set_query_unavailable_bytes": 0}
    result["counters_before"] = sampler.sample()
    allocations, regions = {}, []
    address = 0
    try:
        while address < 0x0000800000000000:
            basic = BasicInfo()
            if not kernel.VirtualQueryEx(sampler.handle, C.c_void_p(address), C.byref(basic), C.sizeof(basic)):
                error = C.get_last_error()
                result["enumeration_terminal_error"] = error
                break
            base, size = int(basic.BaseAddress or 0), int(basic.RegionSize)
            next_address = base + size
            if not size or next_address <= address:
                result["errors"].append("Non-advancing VirtualQueryEx region")
                break
            if basic.State != MEM_FREE:
                allocation_base = int(basic.AllocationBase or base)
                if allocation_base not in allocations:
                    name_buffer = C.create_unicode_buffer(32768)
                    mapped = psapi.GetMappedFileNameW(sampler.handle, C.c_void_p(base), name_buffer, len(name_buffer))
                    filename = Path(name_buffer.value).name if mapped else None
                    category = "GGUF" if filename and filename.lower().endswith(".gguf") else ("image:" + filename if filename and basic.Type == 0x1000000 else ("mapped_file:" + filename if filename else TYPES.get(basic.Type, "other")))
                    info, returned = RegionInfo(), C.c_size_t()
                    ok = query_region(sampler.handle, C.c_void_p(base), 0, C.byref(info), C.sizeof(info), C.byref(returned))
                    item = {"allocation_id":len(allocations), "category":category, "filename":filename, "virtual_span_bytes":0, "committed_region_span_bytes":0, "commit_charge_bytes":int(info.CommitSize) if ok else None, "flags": {name:bool(info.Flags & (1 << bit)) for bit,name in enumerate(("Private","MappedDataFile","MappedImage","MappedPageFile","MappedPhysical","DirectMapped"))} if ok else None, "valid_bytes":0, "valid_shareable_bytes":0, "valid_private_bytes":0, "invalid_bytes":0, "working_set_unavailable_bytes":0, "regions":[]}
                    if not ok:
                        item["allocation_info_error"] = C.get_last_error()
                        result["allocation_info_failure_count"] += 1
                    allocations[allocation_base] = item
                item = allocations[allocation_base]
                item["virtual_span_bytes"] += size
                if basic.State == MEM_COMMIT:
                    item["committed_region_span_bytes"] += size
                    regions.append((base, size, item))
                item["regions"].append({"state":int(basic.State), "type":TYPES.get(basic.Type,hex(basic.Type)), "protect":hex(basic.Protect), "size_bytes":size})
            address = next_address
        buffer = (WorkingSetEntry * 8192)()
        for base, size, item in regions:
            if gguf_only and item["category"] != "GGUF":
                continue
            for offset in range(0, size, PAGE * len(buffer)):
                count = min(len(buffer), (size-offset+PAGE-1)//PAGE)
                for index in range(count):
                    buffer[index].VirtualAddress = base+offset+index*PAGE
                    buffer[index].Flags = 0
                if not psapi.QueryWorkingSetEx(sampler.handle, C.byref(buffer), count*C.sizeof(WorkingSetEntry)):
                    missing = count*PAGE
                    item["working_set_unavailable_bytes"] += missing
                    result["working_set_query_unavailable_bytes"] += missing
                    if len(result["errors"]) < 20:
                        result["errors"].append(f"QueryWorkingSetEx error {C.get_last_error()} allocation{item['allocation_id']}")
                    continue
                for index in range(count):
                    flags = buffer[index].Flags
                    if flags & 1:
                        item["valid_bytes"] += PAGE
                        item["valid_shareable_bytes" if flags & (1 << 15) else "valid_private_bytes"] += PAGE
                    else:
                        item["invalid_bytes"] += PAGE
        result["counters_after"] = sampler.sample()
    finally:
        sampler.close()
    groups = defaultdict(lambda: {"allocations":0,"virtual_span_bytes":0,"committed_region_span_bytes":0,"commit_charge_bytes_known":0,"commit_charge_unavailable_allocations":0,"valid_bytes":0,"valid_private_bytes":0,"valid_shareable_bytes":0,"invalid_bytes":0,"working_set_unavailable_bytes":0})
    for item in allocations.values():
        category = "GGUF" if item["category"] == "GGUF" else ("image" if item["category"].startswith("image:") else "mapped_file" if item["category"].startswith("mapped_file:") else item["category"])
        group = groups[category]
        group["allocations"] += 1
        for key in ("virtual_span_bytes","committed_region_span_bytes","valid_bytes","valid_private_bytes","valid_shareable_bytes","invalid_bytes","working_set_unavailable_bytes"):
            group[key] += item[key]
        if item["commit_charge_bytes"] is None:
            group["commit_charge_unavailable_allocations"] += 1
        else:
            group["commit_charge_bytes_known"] += item["commit_charge_bytes"]
    result.update(groups=dict(groups), allocation_count=len(allocations), region_count=sum(len(a["regions"]) for a in allocations.values()), allocations=sorted(allocations.values(),key=lambda a:a["virtual_span_bytes"],reverse=True), elapsed_seconds=time.perf_counter()-started, gguf_only_working_set=gguf_only)
    result["limits"] = "Non-atomic sequential no-touch observation bracketed by process/PDH samples, not an allocation call-stack trace. Valid/Shared are Windows page-query bits, not proof that device mappings are duplicate system DRAM. MEM_COMMIT RegionSize is not private commit; per-allocation CommitSize counted once. GPU driver charge may not appear in CPU VA CommitSize. Invalid/query-failed pages are not silently treated as measured resident zero."
    return result


def mapped_gguf_observation(pid: int) -> dict:
    snapshot = observe_memory(pid, gguf_only=True)
    return {key: snapshot[key] for key in ("pid", "pointer_bytes", "page_bytes", "elapsed_seconds", "errors", "counters_before", "counters_after", "limits")} | {"gguf": snapshot["groups"].get("GGUF"), "mapped_file_allocations": [a for a in snapshot["allocations"] if a["category"] == "GGUF"], "scope": "Only GGUF pages queried for working-set residency; non-GGUF zero-initialized scan accumulators are intentionally not exposed."}
