from __future__ import annotations

from puripuly_heart.core.local_translation.devices import (
    parse_llama_list_devices,
    resolve_llama_vulkan_device,
)


def test_parse_llama_list_devices_keeps_vulkan_ids_and_strips_vendor() -> None:
    output = """
ggml_vulkan: Found 2 Vulkan devices:
  Vulkan0: NVIDIA GeForce RTX 4070 (NVIDIA)
  Vulkan1: AMD Radeon Graphics (AMD)
  CPU: 16 threads
"""

    devices = parse_llama_list_devices(output)

    assert [(device.device_id, device.display_name) for device in devices] == [
        ("Vulkan0", "NVIDIA GeForce RTX 4070"),
        ("Vulkan1", "AMD Radeon Graphics"),
    ]


def test_resolve_llama_vulkan_device_defaults_auto_to_vulkan0() -> None:
    assert resolve_llama_vulkan_device("auto") == "Vulkan0"
    assert resolve_llama_vulkan_device("  ") == "Vulkan0"
    assert resolve_llama_vulkan_device(None) == "Vulkan0"
    assert resolve_llama_vulkan_device("Vulkan2") == "Vulkan2"
