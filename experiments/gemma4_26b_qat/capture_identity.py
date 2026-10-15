from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
import struct
import subprocess

from acquire import identity


def gguf_identity(path: Path) -> dict:
    with path.open("rb") as stream:
        def number(fmt):
            size = struct.calcsize(fmt)
            return struct.unpack("<" + fmt, stream.read(size))[0]
        def text():
            return stream.read(number("Q")).decode("utf-8", errors="replace")
        def value(kind):
            formats = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
            if kind in formats:
                return number(formats[kind])
            if kind == 8:
                return text()
            if kind == 9:
                element_type, count = number("I"), number("Q")
                for _ in range(count):
                    value(element_type)
                return {"array_type": element_type, "count": count}
            raise ValueError(f"Unknown GGUF metadata kind {kind}")
        magic = stream.read(4)
        if magic != b"GGUF":
            raise ValueError("Not GGUF")
        version, tensors, pairs = number("I"), number("Q"), number("Q")
        metadata = {}
        for _ in range(pairs):
            key, kind = text(), number("I")
            item = value(kind)
            if isinstance(item, str) and len(item) > 2000:
                item = {"characters": len(item), "sha256": hashlib.sha256(item.encode()).hexdigest()}
            metadata[key] = item
        histogram = collections.Counter()
        dimensions = {}
        for _ in range(tensors):
            name, n_dim = text(), number("I")
            shape = [number("Q") for _ in range(n_dim)]
            kind, offset = number("I"), number("Q")
            histogram[kind] += 1
            if any(term in name for term in ("ffn_gate_exps", "ffn_up_exps", "ffn_down_exps", "token_embd")):
                dimensions[name] = {"shape": shape, "ggml_type_id": kind}
        return {"gguf_version": version, "tensor_count": tensors, "metadata": metadata, "tensor_type_histogram_ggml_ids": dict(histogram), "expert_embedding_dimensions": dimensions}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--e4b", type=Path, required=True)
    args = parser.parse_args()
    ps = '$ErrorActionPreference="Stop"; @{os=(Get-CimInstance Win32_OperatingSystem | Select-Object Caption,Version,BuildNumber,TotalVisibleMemorySize,FreePhysicalMemory); cpu=(Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors,MaxClockSpeed); ram=(Get-CimInstance Win32_PhysicalMemory | Select-Object Capacity,ConfiguredClockSpeed,Speed,SMBIOSMemoryType); gpu=(Get-CimInstance Win32_VideoController | Select-Object Name,DriverVersion); disks=(Get-CimInstance Win32_DiskDrive | Select-Object Model,Size,InterfaceType); volumes=(Get-Volume | Select-Object DriveLetter,Size,SizeRemaining); pagefile=(Get-CimInstance Win32_PageFileUsage | Select-Object AllocatedBaseSize,CurrentUsage,PeakUsage); power=(powercfg /getactivescheme); pci=(Get-PnpDevice -Class Display | Where-Object FriendlyName -eq "AMD Radeon RX 7900 XTX" | Get-PnpDeviceProperty | Where-Object KeyName -match "PciDevice.*(Link|Bus)|LocationInfo" | Select-Object KeyName,Data)} | ConvertTo-Json -Depth 6'
    ps = '[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new(); ' + ps
    command = subprocess.run(["powershell.exe", "-NoProfile", "-Command", ps], capture_output=True, encoding="utf-8", errors="replace")
    result = {"baseline_commit": "7a8151be6114a950f062e2430bd36f899d57e4de", "initial_dirty_state": "clean", "host": json.loads(command.stdout), "vulkan_summary": subprocess.run(["vulkaninfo.exe", "--summary"], capture_output=True, encoding="utf-8", errors="replace").stdout, "runtimes": {}, "models": {}}
    for build in ("b10423", "b11537"):
        executable = args.root / "runtimes" / build / "llama-server.exe"
        result["runtimes"][build] = {"binary": identity(executable), "version": subprocess.run([str(executable), "--version"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding="utf-8", errors="replace").stdout}
        help_output = subprocess.run([str(executable), "--help"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding="utf-8", errors="replace").stdout
        (args.root / f"{build}-help.txt").write_text(help_output, encoding="utf-8")
    for path in (args.root / "models" / "gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf", args.e4b):
        result["models"][path.name] = {**identity(path), **gguf_identity(path)}
    acquisition = json.loads((args.root / "acquisition.json").read_text())
    result["acquisition"] = acquisition
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"host_captured": bool(result["host"]), "models": {k: {"size": v["size"], "sha256": v["sha256"], "tensor_types": v["tensor_type_histogram_ggml_ids"]} for k,v in result["models"].items()}}))


if __name__ == "__main__":
    main()
