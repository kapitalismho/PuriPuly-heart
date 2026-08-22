from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from puripuly_heart.core.local_translation.runtime_profile import default_gemma_runtime_paths

_LLAMA_DEVICE_PATTERN = re.compile(
    r"^\s*(Vulkan\d+)\s*:\s*(.+?)(?:\s*\([^)]*\))?\s*$",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class LlamaVulkanDevice:
    device_id: str
    display_name: str


def resolve_llama_vulkan_device(device_id: object) -> str:
    if isinstance(device_id, str) and device_id.strip() and device_id.strip() != "auto":
        return device_id.strip()
    return "Vulkan0"


def parse_llama_list_devices(output: str) -> tuple[LlamaVulkanDevice, ...]:
    devices: list[LlamaVulkanDevice] = []
    seen: set[str] = set()
    for line in output.splitlines():
        match = _LLAMA_DEVICE_PATTERN.fullmatch(line.strip())
        if match is None:
            continue
        device_id = match.group(1)
        if device_id in seen:
            continue
        seen.add(device_id)
        devices.append(
            LlamaVulkanDevice(
                device_id=device_id,
                display_name=match.group(2).strip() or device_id,
            )
        )
    return tuple(devices)


def list_llama_vulkan_devices(
    executable: Path | None = None,
    *,
    timeout_s: float = 10.0,
) -> tuple[LlamaVulkanDevice, ...]:
    server = executable or default_gemma_runtime_paths().vulkan_server
    if not server.is_file():
        return ()
    try:
        completed = subprocess.run(
            [str(server), "--list-devices"],
            check=False,
            capture_output=True,
            text=True,
            timeout=max(0.1, float(timeout_s)),
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.TimeoutExpired):
        return ()
    return parse_llama_list_devices("\n".join((completed.stdout, completed.stderr)))


__all__ = [
    "LlamaVulkanDevice",
    "list_llama_vulkan_devices",
    "parse_llama_list_devices",
    "resolve_llama_vulkan_device",
]
