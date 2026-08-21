from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

import puripuly_heart.core.local_translation.runtime as runtime_module
from puripuly_heart.core.local_translation.assets import GEMMA_12B_SPEC
from puripuly_heart.core.local_translation.runtime import (
    ManagedGemmaMetrics,
    ManagedGemmaResponse,
    ManagedGemmaRuntimeClosedError,
    ManagedGemmaRuntimeError,
    ManagedGemmaRuntimeOwner,
)
from puripuly_heart.core.local_translation.runtime_profile import GemmaRuntimePaths


class FakeProcess:
    def __init__(self) -> None:
        self.returncode = None
        self.terminated = False
        self.killed = False

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = 0

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9

    async def wait(self) -> int:
        return self.returncode or 0


class FakeTransport:
    def __init__(self, base_url: str, *, prefix_failure: BaseException | None = None) -> None:
        self.base_url = base_url
        self.ready_calls = []
        self.prefixes = []
        self.prefix_slots: list[int] = []
        self.restores: list[str] = []
        self.restore_slots: list[int] = []
        self.saves: list[str] = []
        self.save_slots: list[int] = []
        self.restore_hits: set[str] = set()
        self.translations = []
        self.translation_slots: list[int] = []
        self.closed = False
        self.prefix_failure = prefix_failure
        self.response = ManagedGemmaResponse(
            text="translated",
            metrics=ManagedGemmaMetrics(
                prompt_tokens=24,
                cached_prompt_tokens=20,
                completion_tokens=4,
                prompt_ms=5.0,
                generation_ms=100.0,
                generation_tps=40.0,
                drafted_tokens=6,
                accepted_tokens=4,
            ),
        )

    async def wait_until_ready(self, *, timeout_s: float) -> None:
        self.ready_calls.append(timeout_s)

    async def prepare_prefix(self, *, system_prompt: str, slot_id: int) -> None:
        if self.prefix_failure is not None:
            raise self.prefix_failure
        self.prefixes.append(system_prompt)
        self.prefix_slots.append(slot_id)

    async def restore_prefix(self, *, filename: str, slot_id: int) -> bool:
        self.restores.append(filename)
        self.restore_slots.append(slot_id)
        return filename in self.restore_hits

    async def save_prefix(self, *, filename: str, slot_id: int) -> None:
        self.saves.append(filename)
        self.save_slots.append(slot_id)

    async def translate(self, *, system_prompt: str, user_message: str, slot_id: int):
        self.translations.append((system_prompt, user_message))
        self.translation_slots.append(slot_id)
        return self.response

    async def close(self) -> None:
        self.closed = True


class HangingTransport(FakeTransport):
    def __init__(self, base_url: str) -> None:
        super().__init__(base_url)
        self.translation_started = asyncio.Event()

    async def translate(self, *, system_prompt: str, user_message: str, slot_id: int):
        self.translation_started.set()
        await asyncio.Event().wait()


class UncooperativeTranslationTransport(FakeTransport):
    def __init__(self, base_url: str) -> None:
        super().__init__(base_url)
        self.translation_started = asyncio.Event()
        self.allow_finish = asyncio.Event()

    async def translate(self, *, system_prompt: str, user_message: str, slot_id: int):
        self.translation_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await self.allow_finish.wait()
        return self.response


class BlockingSecondPrefixTransport(FakeTransport):
    def __init__(self, base_url: str) -> None:
        super().__init__(base_url)
        self.second_prefix_started = asyncio.Event()
        self.allow_second_prefix = asyncio.Event()

    async def prepare_prefix(self, *, system_prompt: str, slot_id: int) -> None:
        if self.prefixes:
            self.second_prefix_started.set()
            await self.allow_second_prefix.wait()
        await super().prepare_prefix(system_prompt=system_prompt, slot_id=slot_id)


class DyingPrefixTransport(FakeTransport):
    def __init__(self, base_url: str, process: FakeProcess) -> None:
        super().__init__(base_url)
        self.process = process

    async def prepare_prefix(self, *, system_prompt: str, slot_id: int) -> None:
        await super().prepare_prefix(system_prompt=system_prompt, slot_id=slot_id)
        self.process.returncode = 1


class FailOnceCloseTransport(FakeTransport):
    def __init__(self, base_url: str) -> None:
        super().__init__(base_url)
        self.close_calls = 0

    async def close(self) -> None:
        self.close_calls += 1
        if self.close_calls == 1:
            raise RuntimeError("close failed")
        await super().close()


class RetryKillProcess(FakeProcess):
    def __init__(self) -> None:
        super().__init__()
        self.kill_calls = 0

    def terminate(self) -> None:
        raise RuntimeError("terminate failed")

    def kill(self) -> None:
        self.kill_calls += 1
        if self.kill_calls == 1:
            raise RuntimeError("kill failed")
        super().kill()


def _runtime(
    tmp_path: Path,
    *,
    fail_gpu: bool = False,
    transport_failures: tuple[BaseException | None, ...] = (),
    transport_builder=None,
    process_builder=None,
    shutdown_timeout_s: float = 5.0,
    prefix_cache=None,
):
    cpu = tmp_path / "cpu" / "llama-server.exe"
    gpu = tmp_path / "gpu" / "llama-server.exe"
    cpu.parent.mkdir()
    gpu.parent.mkdir()
    cpu.write_bytes(b"cpu")
    gpu.write_bytes(b"gpu")
    commands = []
    processes = []
    transports = []
    provision_calls = []
    logs = []

    async def provisioner(**kwargs):
        provision_calls.append(kwargs)

    async def process_factory(command):
        commands.append(command)
        if fail_gpu and command[0] == str(gpu):
            raise RuntimeError("Vulkan unavailable")
        process = process_builder() if process_builder is not None else FakeProcess()
        processes.append(process)
        return process

    def transport_factory(base_url):
        failure = (
            transport_failures[len(transports)]
            if len(transports) < len(transport_failures)
            else None
        )
        transport = (
            transport_builder(base_url)
            if transport_builder is not None
            else FakeTransport(base_url, prefix_failure=failure)
        )
        transports.append(transport)
        return transport

    owner = ManagedGemmaRuntimeOwner(
        install_dir=tmp_path / "models",
        runtime_paths=GemmaRuntimePaths(cpu_server=cpu, vulkan_server=gpu),
        provisioner=provisioner,
        process_factory=process_factory,
        transport_factory=transport_factory,
        port_allocator=lambda: 38191 + len(commands),
        log_sink=lambda message, level: logs.append((message, level)),
        shutdown_timeout_s=shutdown_timeout_s,
        prefix_cache=prefix_cache,
    )
    return owner, commands, processes, transports, provision_calls, logs


@pytest.mark.asyncio
async def test_readiness_prefills_once_and_rebuilds_for_language_pair(tmp_path: Path) -> None:
    owner, commands, processes, transports, provision_calls, _logs = _runtime(tmp_path)

    first = await owner.prepare(
        backend="cpu",
        source_language="ko",
        target_language="en",
        system_prompt="Translate {source_language} to {target_language}",
    )
    same = await owner.prepare(
        backend="cpu",
        source_language="ko",
        target_language="en",
        system_prompt="Translate {source_language} to {target_language}",
    )
    changed = await owner.prepare(
        backend="cpu",
        source_language="ja",
        target_language="en",
        system_prompt="Translate {source_language} to {target_language}",
    )

    assert first.prefix_identity == same.prefix_identity
    assert changed.prefix_identity != first.prefix_identity
    assert len(provision_calls) == 1
    assert len(commands) == 1
    assert len(processes) == 1
    assert transports[0].prefixes == ["Translate ko to en", "Translate ja to en"]
    assert transports[0].prefix_slots == [0, 1]


@pytest.mark.asyncio
async def test_gpu_start_failure_falls_back_internally_to_cpu(tmp_path: Path) -> None:
    owner, commands, _processes, transports, provision_calls, logs = _runtime(
        tmp_path, fail_gpu=True
    )

    readiness = await owner.prepare(
        backend="gpu",
        source_language="ko",
        target_language="en",
        system_prompt="translate",
        vulkan_device="Vulkan1",
    )

    assert readiness.requested_backend == "gpu"
    assert readiness.effective_backend == "cpu"
    assert len(provision_calls) == 1
    assert [command[0] for command in commands] == [
        str(tmp_path / "gpu" / "llama-server.exe"),
        str(tmp_path / "cpu" / "llama-server.exe"),
    ]
    assert "--spec-draft-model" not in commands[0]
    assert "--spec-draft-model" in commands[1]
    assert transports[0].prefixes == ["translate"]
    assert any("backend_fallback requested=gpu effective=cpu" in message for message, _ in logs)


@pytest.mark.asyncio
async def test_translation_is_serialized_uses_current_prefix_and_logs_only_metrics(
    tmp_path: Path,
) -> None:
    owner, _commands, _processes, transports, _provision_calls, logs = _runtime(tmp_path)
    source_text = "private subtitle"

    first, second = await asyncio.gather(
        owner.translate(
            backend="cpu",
            source_language="ko",
            target_language="en",
            system_prompt="Translate {source_language} to {target_language}",
            user_message=source_text,
        ),
        owner.translate(
            backend="cpu",
            source_language="ko",
            target_language="en",
            system_prompt="Translate {source_language} to {target_language}",
            user_message="second",
        ),
    )

    assert first.text == second.text == "translated"
    assert transports[0].prefixes == ["Translate ko to en"]
    assert transports[0].translations == [
        ("Translate ko to en", source_text),
        ("Translate ko to en", "second"),
    ]
    assert transports[0].translation_slots == [0, 0]
    messages = [message for message, _level in logs]
    assert all(source_text not in message for message in messages)
    assert all("translated" not in message for message in messages)
    assert any(
        "backend=cpu language_pair=ko->en prompt_tokens=24 cached_prompt_tokens=20 "
        "completion_tokens=4 prompt_ms=5.000 generation_ms=100.000 generation_tps=40.000 "
        "drafted_tokens=6 accepted_tokens=4" in message
        for message in messages
    )


@pytest.mark.asyncio
async def test_release_stops_process_and_allows_reselection_but_close_is_terminal(
    tmp_path: Path,
) -> None:
    owner, _commands, processes, transports, _provision_calls, _logs = _runtime(tmp_path)
    arguments = {
        "backend": "cpu",
        "source_language": "ko",
        "target_language": "en",
        "system_prompt": "translate",
    }
    await owner.prepare(**arguments)

    await owner.release()

    assert processes[0].terminated
    assert transports[0].closed
    await owner.prepare(**arguments)
    assert len(processes) == 2
    await owner.close()
    assert processes[1].terminated
    with pytest.raises(ManagedGemmaRuntimeClosedError):
        await owner.prepare(**arguments)


@pytest.mark.asyncio
async def test_vulkan_prefill_failure_stops_gpu_and_prepares_cpu_prefix(tmp_path: Path) -> None:
    owner, commands, processes, transports, _provision_calls, logs = _runtime(
        tmp_path,
        transport_failures=(RuntimeError("Vulkan inference failed"), None),
    )

    readiness = await owner.prepare(
        backend="gpu",
        source_language="ko",
        target_language="en",
        system_prompt="translate",
    )

    assert readiness.effective_backend == "cpu"
    assert len(commands) == 2
    assert processes[0].terminated
    assert transports[0].closed
    assert transports[1].prefixes == ["translate"]
    assert any("reason=vulkan_prefill_failed" in message for message, _level in logs)


@pytest.mark.asyncio
async def test_prefix_cancellation_stops_process_and_transport(tmp_path: Path) -> None:
    owner, _commands, processes, transports, _provision_calls, _logs = _runtime(
        tmp_path,
        transport_failures=(asyncio.CancelledError(),),
    )

    with pytest.raises(asyncio.CancelledError):
        await owner.prepare(
            backend="cpu",
            source_language="ko",
            target_language="en",
            system_prompt="translate",
        )

    assert processes[0].terminated
    assert transports[0].closed
    assert owner.readiness is None


@pytest.mark.asyncio
async def test_repeated_gpu_request_reuses_cpu_fallback_and_prefix(tmp_path: Path) -> None:
    owner, commands, _processes, transports, provision_calls, _logs = _runtime(
        tmp_path,
        fail_gpu=True,
    )
    arguments = {
        "backend": "gpu",
        "source_language": "ko",
        "target_language": "en",
        "system_prompt": "translate",
    }

    first = await owner.prepare(**arguments)
    second = await owner.prepare(**arguments)

    assert first == second
    assert first.effective_backend == "cpu"
    assert len(commands) == 2
    assert len(provision_calls) == 1
    assert transports[0].prefixes == ["translate"]


@pytest.mark.asyncio
async def test_repeated_gpu_request_reuses_cpu_after_prefill_fallback(tmp_path: Path) -> None:
    owner, commands, _processes, transports, provision_calls, _logs = _runtime(
        tmp_path,
        transport_failures=(RuntimeError("Vulkan inference failed"), None),
    )
    arguments = {
        "backend": "gpu",
        "source_language": "ko",
        "target_language": "en",
        "system_prompt": "translate",
    }

    first = await owner.prepare(**arguments)
    second = await owner.prepare(**arguments)

    assert first == second
    assert first.effective_backend == "cpu"
    assert len(commands) == 2
    assert len(provision_calls) == 1
    assert transports[1].prefixes == ["translate"]


@pytest.mark.asyncio
async def test_gpu_device_change_restarts_runtime_with_new_device(tmp_path: Path) -> None:
    owner, commands, processes, _transports, provision_calls, _logs = _runtime(tmp_path)
    arguments = {
        "backend": "gpu",
        "source_language": "ko",
        "target_language": "en",
        "system_prompt": "translate",
    }

    await owner.prepare(**arguments, vulkan_device="Vulkan0")
    await owner.prepare(**arguments, vulkan_device="Vulkan1")

    assert len(commands) == 2
    assert len(provision_calls) == 2
    assert processes[0].terminated
    first_device = commands[0].index("--device")
    second_device = commands[1].index("--device")
    assert commands[0][first_device + 1] == "Vulkan0"
    assert commands[1][second_device + 1] == "Vulkan1"


@pytest.mark.asyncio
async def test_model_spec_change_restarts_runtime_without_mtp(tmp_path: Path) -> None:
    owner, commands, processes, _transports, provision_calls, _logs = _runtime(tmp_path)
    arguments = {
        "backend": "gpu",
        "source_language": "ko",
        "target_language": "en",
        "system_prompt": "translate",
    }

    await owner.prepare(**arguments)
    await owner.prepare(**arguments, spec=GEMMA_12B_SPEC)

    assert len(commands) == 2
    assert len(provision_calls) == 2
    assert processes[0].terminated
    assert provision_calls[1]["spec"] == GEMMA_12B_SPEC
    assert str(tmp_path / "models" / GEMMA_12B_SPEC.model_filename) in commands[1]
    assert not any(item.startswith("--spec-") for item in commands[1])


@pytest.mark.asyncio
async def test_runtime_commit_change_rebuilds_prefix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner, _commands, _processes, transports, _provision_calls, _logs = _runtime(tmp_path)
    arguments = {
        "backend": "cpu",
        "source_language": "ko",
        "target_language": "en",
        "system_prompt": "translate",
    }

    first = await owner.prepare(**arguments)
    monkeypatch.setattr(runtime_module, "LLAMA_CPP_COMMIT", "replacement-commit")
    second = await owner.prepare(**arguments)

    assert first.prefix_identity != second.prefix_identity
    assert transports[0].prefixes == ["translate", "translate"]


@pytest.mark.asyncio
async def test_readiness_is_hidden_while_language_prefix_changes(tmp_path: Path) -> None:
    owner, _commands, _processes, transports, _provision_calls, _logs = _runtime(
        tmp_path,
        transport_builder=BlockingSecondPrefixTransport,
    )
    await owner.prepare(
        backend="cpu",
        source_language="ko",
        target_language="en",
        system_prompt="translate",
    )

    change = asyncio.create_task(
        owner.prepare(
            backend="cpu",
            source_language="ja",
            target_language="en",
            system_prompt="translate",
        )
    )
    await transports[0].second_prefix_started.wait()

    assert owner.readiness is None
    transports[0].allow_second_prefix.set()
    readiness = await change
    assert readiness.source_language == "ja"


@pytest.mark.asyncio
async def test_child_exit_during_prefix_never_publishes_readiness(tmp_path: Path) -> None:
    processes = []
    owner, _commands, processes, transports, _provision_calls, _logs = _runtime(
        tmp_path,
        transport_builder=lambda base_url: DyingPrefixTransport(base_url, processes[-1]),
    )

    with pytest.raises(ManagedGemmaRuntimeError, match="prefix preparation failed"):
        await owner.prepare(
            backend="cpu",
            source_language="ko",
            target_language="en",
            system_prompt="translate",
        )

    assert owner.readiness is None
    assert transports[0].closed


@pytest.mark.asyncio
async def test_dead_process_reprovisions_before_restart_and_clears_readiness(
    tmp_path: Path,
) -> None:
    owner, commands, processes, _transports, provision_calls, _logs = _runtime(tmp_path)
    arguments = {
        "backend": "cpu",
        "source_language": "ko",
        "target_language": "en",
        "system_prompt": "translate",
    }
    await owner.prepare(**arguments)
    processes[0].returncode = 1

    assert owner.readiness is None
    await owner.prepare(**arguments)

    assert len(commands) == 2
    assert len(provision_calls) == 2


@pytest.mark.asyncio
async def test_invalid_prompt_fails_before_provisioning_or_process_launch(tmp_path: Path) -> None:
    owner, commands, _processes, _transports, provision_calls, _logs = _runtime(tmp_path)

    with pytest.raises(KeyError):
        await owner.prepare(
            backend="cpu",
            source_language="ko",
            target_language="en",
            system_prompt="{source_language} {unknown}",
        )

    assert commands == []
    assert provision_calls == []
    assert owner.readiness is None


@pytest.mark.asyncio
async def test_transport_factory_failure_cleans_owned_child_process(tmp_path: Path) -> None:
    owner, _commands, processes, _transports, _provision_calls, _logs = _runtime(tmp_path)

    def fail_transport(_base_url):
        raise RuntimeError("transport factory failed")

    owner._transport_factory = fail_transport
    with pytest.raises(ManagedGemmaRuntimeError, match="CPU startup failed"):
        await owner.prepare(
            backend="cpu",
            source_language="ko",
            target_language="en",
            system_prompt="translate",
        )

    assert processes[0].terminated
    assert owner.readiness is None


@pytest.mark.asyncio
async def test_close_cancels_hanging_translation_before_teardown(tmp_path: Path) -> None:
    owner, _commands, processes, transports, _provision_calls, _logs = _runtime(
        tmp_path,
        transport_builder=HangingTransport,
    )
    operation = asyncio.create_task(
        owner.translate(
            backend="cpu",
            source_language="ko",
            target_language="en",
            system_prompt="translate",
            user_message="input",
        )
    )
    while not transports:
        await asyncio.sleep(0)
    await transports[0].translation_started.wait()

    await asyncio.wait_for(owner.close(), timeout=1.0)

    assert operation.cancelled()
    assert processes[0].terminated
    assert transports[0].closed


@pytest.mark.asyncio
async def test_close_bounds_uncooperative_operation_and_hides_readiness(tmp_path: Path) -> None:
    owner, _commands, processes, transports, _provision_calls, _logs = _runtime(
        tmp_path,
        transport_builder=UncooperativeTranslationTransport,
        shutdown_timeout_s=0.1,
    )
    operation = asyncio.create_task(
        owner.translate(
            backend="cpu",
            source_language="ko",
            target_language="en",
            system_prompt="translate",
            user_message="input",
        )
    )
    while not transports:
        await asyncio.sleep(0)
    await transports[0].translation_started.wait()

    with pytest.raises(ManagedGemmaRuntimeError, match="operations did not stop"):
        await asyncio.wait_for(owner.close(), timeout=0.5)

    assert owner.readiness is None
    assert processes[0].returncode is None
    transports[0].allow_finish.set()
    with pytest.raises(ManagedGemmaRuntimeClosedError):
        await operation
    await owner.close()
    assert processes[0].terminated


@pytest.mark.asyncio
async def test_failed_cleanup_retains_resources_for_retry(tmp_path: Path) -> None:
    owner, _commands, processes, transports, _provision_calls, _logs = _runtime(
        tmp_path,
        transport_builder=FailOnceCloseTransport,
        process_builder=RetryKillProcess,
    )
    await owner.prepare(
        backend="cpu",
        source_language="ko",
        target_language="en",
        system_prompt="translate",
    )

    with pytest.raises(BaseExceptionGroup):
        await owner.close()
    assert processes[0].returncode is None
    assert not transports[0].closed

    await owner.close()
    assert processes[0].returncode == -9
    assert transports[0].closed
    await owner.close()


@pytest.mark.asyncio
async def test_prefix_cache_restore_skips_prefill_after_process_restart(tmp_path: Path) -> None:
    from puripuly_heart.core.local_translation.prefix_cache import GemmaPrefixCache

    cache = GemmaPrefixCache(tmp_path / "prefix-cache")
    restore_hits: set[str] = set()

    def transport_builder(base_url: str) -> FakeTransport:
        transport = FakeTransport(base_url)
        transport.restore_hits = restore_hits
        return transport

    owner, commands, _processes, transports, _provision_calls, _logs = _runtime(
        tmp_path,
        transport_builder=transport_builder,
        prefix_cache=cache,
    )
    first = await owner.prepare(
        backend="cpu",
        source_language="ko",
        target_language="en",
        system_prompt="translate",
    )

    assert transports[0].prefixes
    assert transports[0].saves
    filename = transports[0].saves[0]
    (cache.cache_dir / filename).write_bytes(b"kv")
    restore_hits.add(filename)
    await owner.release()

    second = await owner.prepare(
        backend="cpu",
        source_language="ko",
        target_language="en",
        system_prompt="translate",
    )

    assert first.prefix_identity == second.prefix_identity
    assert "--slot-save-path" in commands[0]
    assert transports[-1].restores == [filename]
    assert transports[-1].restore_slots == [0]
    assert transports[-1].prefixes == []


@pytest.mark.asyncio
async def test_two_identities_stay_resident_without_second_prefill(tmp_path: Path) -> None:
    owner, _commands, _processes, transports, _provision_calls, _logs = _runtime(tmp_path)
    template = "Translate {source_language} to {target_language}"

    await owner.prepare(
        backend="cpu",
        source_language="ko",
        target_language="en",
        system_prompt=template,
    )
    await owner.prepare(
        backend="cpu",
        source_language="en",
        target_language="ko",
        system_prompt=template,
    )
    await owner.prepare(
        backend="cpu",
        source_language="ko",
        target_language="en",
        system_prompt=template,
    )
    await owner.translate(
        backend="cpu",
        source_language="en",
        target_language="ko",
        system_prompt=template,
        user_message="hello",
    )

    assert transports[0].prefixes == ["Translate ko to en", "Translate en to ko"]
    assert transports[0].prefix_slots == [0, 1]
    assert transports[0].translation_slots == [1]


@pytest.mark.asyncio
async def test_third_identity_evicts_only_lru_slot(tmp_path: Path) -> None:
    owner, _commands, _processes, transports, _provision_calls, _logs = _runtime(tmp_path)
    template = "Translate {source_language} to {target_language}"

    await owner.prepare(
        backend="cpu",
        source_language="ko",
        target_language="en",
        system_prompt=template,
    )
    await owner.prepare(
        backend="cpu",
        source_language="en",
        target_language="ko",
        system_prompt=template,
    )
    await owner.prepare(
        backend="cpu",
        source_language="ko",
        target_language="en",
        system_prompt=template,
    )
    await owner.prepare(
        backend="cpu",
        source_language="ja",
        target_language="en",
        system_prompt=template,
    )
    await owner.prepare(
        backend="cpu",
        source_language="ko",
        target_language="en",
        system_prompt=template,
    )

    assert transports[0].prefixes == [
        "Translate ko to en",
        "Translate en to ko",
        "Translate ja to en",
    ]
    assert transports[0].prefix_slots == [0, 1, 1]
