from pathlib import Path

from puripuly_heart.core.local_translation.assets import GEMMA_12B_SPEC
from puripuly_heart.core.local_translation.runtime_profile import (
    LLAMA_CPP_BUILD,
    LLAMA_CPP_COMMIT,
    MANAGED_GEMMA_12B_MODEL_ALIAS,
    LlamaCppThreadProfile,
    build_gemma_server_command,
)

CPU_PROFILE = LlamaCppThreadProfile(threads=4, threads_batch=12, draft_threads=1)


def _values(command: tuple[str, ...], flag: str) -> list[str]:
    return [command[index + 1] for index, item in enumerate(command[:-1]) if item == flag]


def test_cpu_profile_preserves_fixed_common_and_mtp_contract(tmp_path: Path) -> None:
    command = build_gemma_server_command(
        executable=Path("llama-server.exe"),
        install_dir=tmp_path,
        backend="cpu",
        port=38191,
        threads_profile=CPU_PROFILE,
    )

    assert LLAMA_CPP_BUILD == "b10423"
    assert LLAMA_CPP_COMMIT == "a94d563ed801d1da1b8c2432946de07d0231bb3d"
    assert _values(command, "--load-mode") == ["mmap"]
    assert _values(command, "--threads") == ["4"]
    assert _values(command, "--threads-batch") == ["12"]
    assert _values(command, "--ctx-size") == ["3072"]
    assert _values(command, "--parallel") == ["2"]
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
    assert _values(command, "--spec-draft-n-max") == ["3"]
    assert _values(command, "--spec-draft-n-min") == ["2"]
    assert _values(command, "--spec-draft-p-min") == ["0.0"]
    assert _values(command, "--spec-draft-device") == ["none"]
    assert _values(command, "--spec-draft-ngl") == ["0"]
    assert _values(command, "--spec-draft-threads") == ["1"]
    assert _values(command, "--spec-draft-threads-batch") == ["1"]
    assert _values(command, "--spec-draft-type-k") == ["f16"]
    assert _values(command, "--spec-draft-type-v") == ["f16"]
    assert _values(command, "--flash-attn") == ["off"]
    assert _values(command, "--prio") == ["-1"]
    assert "--slot-save-path" not in command
    assert "--swa-full" not in command


def test_slot_save_path_is_opt_in(tmp_path: Path) -> None:
    cache_dir = tmp_path / "prefix-cache"
    command = build_gemma_server_command(
        executable=Path("llama-server.exe"),
        install_dir=tmp_path,
        backend="cpu",
        port=38191,
        threads_profile=CPU_PROFILE,
        slot_save_path=cache_dir,
    )

    assert _values(command, "--slot-save-path") == [str(cache_dir)]


def test_gpu_profile_is_vulkan_full_offload_without_mtp(tmp_path: Path) -> None:
    command = build_gemma_server_command(
        executable=Path("llama-server.exe"),
        install_dir=tmp_path,
        backend="gpu",
        port=38192,
        vulkan_device="Vulkan2",
        threads_profile=CPU_PROFILE,
    )

    assert _values(command, "--load-mode") == ["mmap"]
    assert _values(command, "--threads") == ["4"]
    assert _values(command, "--threads-batch") == ["4"]
    assert _values(command, "--device") == ["Vulkan2"]
    assert _values(command, "--n-gpu-layers") == ["99"]
    assert _values(command, "--flash-attn") == ["on"]
    assert _values(command, "--prio") == ["-1"]
    assert _values(command, "--cache-type-k") == ["f16"]
    assert _values(command, "--cache-type-v") == ["f16"]
    assert _values(command, "--ctx-size") == ["3072"]
    assert _values(command, "--parallel") == ["2"]
    assert "--swa-full" not in command
    assert not any(item.startswith("--spec-") for item in command)


def test_gpu_12b_profile_is_vulkan_full_offload_without_mtp(tmp_path: Path) -> None:
    command = build_gemma_server_command(
        executable=Path("llama-server.exe"),
        install_dir=tmp_path,
        backend="gpu",
        port=38193,
        vulkan_device="Vulkan0",
        threads_profile=CPU_PROFILE,
        spec=GEMMA_12B_SPEC,
    )

    assert _values(command, "--model") == [str(tmp_path / GEMMA_12B_SPEC.model_filename)]
    assert _values(command, "--alias") == [MANAGED_GEMMA_12B_MODEL_ALIAS]
    assert _values(command, "--n-gpu-layers") == ["99"]
    assert _values(command, "--flash-attn") == ["on"]
    assert _values(command, "--cache-type-k") == ["q8_0"]
    assert _values(command, "--cache-type-v") == ["q8_0"]
    assert "--swa-full" in command
    assert not any(item.startswith("--spec-") for item in command)


def test_cpu_12b_profile_omits_mtp(tmp_path: Path) -> None:
    command = build_gemma_server_command(
        executable=Path("llama-server.exe"),
        install_dir=tmp_path,
        backend="cpu",
        port=38194,
        threads_profile=CPU_PROFILE,
        spec=GEMMA_12B_SPEC,
    )

    assert _values(command, "--model") == [str(tmp_path / GEMMA_12B_SPEC.model_filename)]
    assert not any(item.startswith("--spec-") for item in command)
    assert _values(command, "--device") == ["none"]
    assert _values(command, "--n-gpu-layers") == ["0"]
    assert _values(command, "--cache-type-k") == ["q8_0"]
    assert _values(command, "--cache-type-v") == ["q8_0"]
    assert "--swa-full" in command


def test_derive_thread_profile_applies_generalization_rule() -> None:
    from puripuly_heart.core.local_translation.runtime_profile import derive_thread_profile

    assert derive_thread_profile(4, 8) == LlamaCppThreadProfile(2, 6, 1)
    assert derive_thread_profile(4, 4) == LlamaCppThreadProfile(2, 3, 1)
    assert derive_thread_profile(6, 12) == LlamaCppThreadProfile(3, 9, 1)
    assert derive_thread_profile(8, 16) == LlamaCppThreadProfile(4, 12, 1)
    assert derive_thread_profile(12, 24) == LlamaCppThreadProfile(6, 12, 1)
    assert derive_thread_profile(16, 32) == LlamaCppThreadProfile(6, 12, 1)


def test_load_or_detect_thread_profile_detects_and_persists(
    tmp_path: Path,
) -> None:
    from puripuly_heart.core.local_translation.runtime_profile import (
        load_or_detect_thread_profile,
    )

    path = tmp_path / "threads.json"
    profile = load_or_detect_thread_profile(path, physical_cores=8, logical_cores=16)
    assert profile == LlamaCppThreadProfile(4, 12, 1)
    assert path.is_file()

    reloaded = load_or_detect_thread_profile(path, physical_cores=4, logical_cores=4)
    assert reloaded == LlamaCppThreadProfile(4, 12, 1)


def test_load_or_detect_thread_profile_corrupt_file_redetects(
    tmp_path: Path,
) -> None:
    from puripuly_heart.core.local_translation.runtime_profile import (
        load_or_detect_thread_profile,
    )

    path = tmp_path / "threads.json"
    path.write_text("not json", encoding="utf-8")
    profile = load_or_detect_thread_profile(path, physical_cores=8, logical_cores=16)
    assert profile == LlamaCppThreadProfile(4, 12, 1)
    assert path.is_file()
