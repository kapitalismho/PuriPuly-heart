from pathlib import Path

from puripuly_heart.core.local_translation.runtime_profile import (
    LLAMA_CPP_BUILD,
    LLAMA_CPP_COMMIT,
    build_gemma_server_command,
)


def _values(command: tuple[str, ...], flag: str) -> list[str]:
    return [command[index + 1] for index, item in enumerate(command[:-1]) if item == flag]


def test_cpu_profile_preserves_fixed_common_and_mtp_contract(tmp_path: Path) -> None:
    command = build_gemma_server_command(
        executable=Path("llama-server.exe"),
        install_dir=tmp_path,
        backend="cpu",
        port=38191,
    )

    assert LLAMA_CPP_BUILD == "b10423"
    assert LLAMA_CPP_COMMIT == "a94d563ed801d1da1b8c2432946de07d0231bb3d"
    assert _values(command, "--load-mode") == ["mmap"]
    assert _values(command, "--threads") == ["4"]
    assert _values(command, "--threads-batch") == ["4"]
    assert _values(command, "--ctx-size") == ["4096"]
    assert _values(command, "--parallel") == ["1"]
    assert _values(command, "--batch-size") == ["512"]
    assert _values(command, "--ubatch-size") == ["512"]
    assert _values(command, "--cache-type-k") == ["f16"]
    assert _values(command, "--cache-type-v") == ["f16"]
    assert "--cache-prompt" in command
    assert _values(command, "--reasoning") == ["off"]
    assert _values(command, "--reasoning-budget") == ["0"]
    assert "--warmup" in command
    assert "--perf" in command
    assert "--metrics" in command
    assert "--no-webui" in command
    assert _values(command, "--threads-http") == ["1"]
    assert _values(command, "--host") == ["127.0.0.1"]
    assert _values(command, "--device") == ["none"]
    assert _values(command, "--n-gpu-layers") == ["0"]
    assert _values(command, "--spec-type") == ["draft-mtp"]
    assert _values(command, "--spec-draft-n-max") == ["4"]
    assert _values(command, "--spec-draft-n-min") == ["1"]
    assert _values(command, "--spec-draft-device") == ["none"]
    assert _values(command, "--spec-draft-ngl") == ["0"]
    assert _values(command, "--spec-draft-threads") == ["4"]
    assert _values(command, "--spec-draft-threads-batch") == ["4"]
    assert _values(command, "--spec-draft-type-k") == ["f16"]
    assert _values(command, "--spec-draft-type-v") == ["f16"]
    assert _values(command, "--flash-attn") == ["auto"]


def test_gpu_profile_is_vulkan_full_offload_without_mtp(tmp_path: Path) -> None:
    command = build_gemma_server_command(
        executable=Path("llama-server.exe"),
        install_dir=tmp_path,
        backend="gpu",
        port=38192,
        vulkan_device="Vulkan2",
    )

    assert _values(command, "--load-mode") == ["mmap"]
    assert _values(command, "--threads") == ["4"]
    assert _values(command, "--threads-batch") == ["4"]
    assert _values(command, "--device") == ["Vulkan2"]
    assert _values(command, "--n-gpu-layers") == ["99"]
    assert _values(command, "--flash-attn") == ["on"]
    assert not any(item.startswith("--spec-") for item in command)
