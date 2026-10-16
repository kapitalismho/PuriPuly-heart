from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

import puripuly_heart.core.local_translation.runtime as runtime_module
from puripuly_heart.core.local_translation.assets import (
    GEMMA_LICENSE,
    GEMMA_LICENSE_URL,
    GemmaModelSpec,
    e4b_gemma_spec,
    gemma_26b_spec,
)
from puripuly_heart.core.local_translation.runtime import (
    ManagedGemmaMetrics,
    ManagedGemmaResponse,
    ManagedGemmaRuntimeClosedError,
    ManagedGemmaRuntimeError,
    ManagedGemmaRuntimeOwner,
)
from puripuly_heart.core.local_translation.runtime_profile import GemmaRuntimePaths


@pytest.fixture(autouse=True)
def isolated_thread_profile(tmp_path, monkeypatch):
    from puripuly_heart.core.local_translation import runtime_profile

    monkeypatch.setattr(
        runtime_profile, "default_threads_profile_path", lambda: tmp_path / "threads.json"
    )


class FakeProcess:
    def __init__(self) -> None:
        self._returncode: int | None = None
        self._exit = asyncio.Event()
        self.terminated = False
        self.killed = False

    @property
    def returncode(self) -> int | None:
        return self._returncode

    @returncode.setter
    def returncode(self, value: int | None) -> None:
        self._returncode = value
        if value is not None:
            self._exit.set()

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = 0

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9

    async def wait(self) -> int:
        await self._exit.wait()
        assert self.returncode is not None
        return self.returncode


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

    async def wait_slot_idle(self, *, slot_id: int) -> None:
        return None

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


class NeverReadyTransport(FakeTransport):
    async def wait_until_ready(self, *, timeout_s: float) -> None:
        self.ready_calls.append(timeout_s)
        await asyncio.Event().wait()


class ImmediateExitProcess(FakeProcess):
    def __init__(self) -> None:
        super().__init__()
        self.returncode = 23


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

    async def validate_version(_executable: Path) -> None:
        return None

    async def provisioner(**kwargs):
        provision_calls.append(kwargs)

    async def process_factory(command, _cwd):
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
        version_validator=validate_version,
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
    owner, commands, _processes, transports, provision_calls, _logs = _runtime(
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


@pytest.mark.asyncio
async def test_startup_reports_child_exit_and_falls_back_without_waiting_for_timeout(
    tmp_path: Path,
) -> None:
    owner, commands, _processes, _transports, _provision_calls, _logs = _runtime(
        tmp_path,
        transport_builder=NeverReadyTransport,
        process_builder=ImmediateExitProcess,
    )

    with pytest.raises(ManagedGemmaRuntimeError):
        await asyncio.wait_for(
            owner.prepare(
                backend="gpu",
                source_language="ko",
                target_language="en",
                system_prompt="translate",
            ),
            timeout=0.5,
        )

    assert len(commands) == 2
    assert owner.readiness is None


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
async def test_prefix_cancellation_keeps_settled_runtime_available_for_retry(
    tmp_path: Path,
) -> None:
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

    assert not processes[0].terminated
    assert not transports[0].closed
    assert owner.readiness is None
    transports[0].prefix_failure = None
    await owner.prepare(
        backend="cpu",
        source_language="ko",
        target_language="en",
        system_prompt="translate",
    )
    assert len(processes) == 1
    await owner.close()


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
    owner, commands, processes, _transports, _provision_calls, _logs = _runtime(tmp_path)
    drafter_less_spec = GemmaModelSpec(
        model_id="gemma-4-test-qat",
        repo_id="example/drafter-less",
        revision="abc123",
        model_filename="target.gguf",
        draft_filename=None,
        install_dirname="gemma-drafter-less",
        upstream_repo_id="example/upstream",
        license=GEMMA_LICENSE,
        license_url=GEMMA_LICENSE_URL,
        assets=(),
        sources={},
    )
    arguments = {
        "backend": "gpu",
        "source_language": "ko",
        "target_language": "en",
        "system_prompt": "translate",
    }

    await owner.prepare(**arguments)
    await owner.prepare(**arguments, spec=drafter_less_spec)

    assert len(commands) == 2
    assert processes[0].terminated
    assert str(tmp_path / "models" / drafter_less_spec.model_filename) in commands[1]
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

    with pytest.raises(ManagedGemmaRuntimeError):
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
    with pytest.raises(ManagedGemmaRuntimeError):
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

    with pytest.raises(ManagedGemmaRuntimeError):
        await asyncio.wait_for(owner.close(), timeout=0.5)

    assert owner.readiness is None
    assert processes[0].terminated
    transports[0].allow_finish.set()
    with pytest.raises(asyncio.CancelledError):
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
@pytest.mark.parametrize("backend", ["cpu", "gpu"])
async def test_prefix_cache_restore_skips_prefill_after_process_restart(
    tmp_path: Path,
    backend: str,
) -> None:
    from puripuly_heart.core.local_translation.prefix_cache import GemmaPrefixCache

    cache = GemmaPrefixCache(tmp_path / "prefix-cache")
    restore_hits: set[str] = set()

    def transport_builder(base_url: str) -> FakeTransport:
        transport = FakeTransport(base_url)
        transport.restore_hits = restore_hits
        return transport

    owner, _commands, _processes, transports, _provision_calls, _logs = _runtime(
        tmp_path,
        transport_builder=transport_builder,
        prefix_cache=cache,
    )
    first = await owner.prepare(
        backend=backend,
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
        backend=backend,
        source_language="ko",
        target_language="en",
        system_prompt="translate",
    )

    assert first.prefix_identity == second.prefix_identity
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
async def test_fourth_identity_evicts_only_lru_slot(tmp_path: Path) -> None:
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
        source_language="de",
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
        "Translate de to en",
    ]
    assert transports[0].prefix_slots == [0, 1, 2, 1]


class ControlledTransport(FakeTransport):
    def __init__(self, base_url: str) -> None:
        super().__init__(base_url)
        self.started: asyncio.Queue[tuple[str, int]] = asyncio.Queue()
        self.gates: dict[str, asyncio.Event] = {}
        self.active: dict[int, str] = {}
        self.peak_active = 0
        self.failures: set[str] = set()
        self.settlements: dict[int, asyncio.Event] = {}
        self.settling = asyncio.Event()
        self.fail_prompt: str | None = None

    async def prepare_prefix(self, *, system_prompt: str, slot_id: int) -> None:
        assert slot_id not in self.active
        if system_prompt == self.fail_prompt:
            raise RuntimeError("prefix backend failure")
        await super().prepare_prefix(system_prompt=system_prompt, slot_id=slot_id)

    async def restore_prefix(self, *, filename: str, slot_id: int) -> bool:
        assert slot_id not in self.active
        return await super().restore_prefix(filename=filename, slot_id=slot_id)

    async def save_prefix(self, *, filename: str, slot_id: int) -> None:
        assert slot_id not in self.active
        await super().save_prefix(filename=filename, slot_id=slot_id)

    async def translate(self, *, system_prompt: str, user_message: str, slot_id: int):
        assert slot_id not in self.active
        self.active[slot_id] = user_message
        self.peak_active = max(self.peak_active, len(self.active))
        self.gates[user_message] = asyncio.Event()
        await self.started.put((user_message, slot_id))
        await self.gates[user_message].wait()
        if user_message in self.failures:
            raise RuntimeError("request failed")
        del self.active[slot_id]
        return ManagedGemmaResponse(user_message, self.response.metrics)

    async def wait_slot_idle(self, *, slot_id: int) -> None:
        self.settling.set()
        gate = self.settlements.get(slot_id)
        if gate is not None:
            await gate.wait()
        self.active.pop(slot_id, None)


def _request(owner, text: str, *, backend="gpu", spec=None, reverse=False):
    return asyncio.create_task(
        owner.translate(
            backend=backend,
            spec=spec,
            source_language="en" if reverse else "ko",
            target_language="ko" if reverse else "en",
            system_prompt="Translate {source_language} to {target_language}",
            user_message=text,
        )
    )


async def _started(transport: ControlledTransport) -> tuple[str, int]:
    return await asyncio.wait_for(transport.started.get(), timeout=1.0)


@pytest.mark.asyncio
@pytest.mark.parametrize("spec", [e4b_gemma_spec(), gemma_26b_spec()], ids=["e4b", "26b"])
@pytest.mark.parametrize("reverse", [False, True], ids=["same-pair", "opposite-pair"])
async def test_gpu_overlaps_two_requests_and_reuses_only_idle_slots(tmp_path, spec, reverse):
    owner, commands, processes, transports, _calls, _logs = _runtime(
        tmp_path, transport_builder=ControlledTransport
    )
    await owner.prepare(
        backend="gpu",
        spec=spec,
        source_language="ko",
        target_language="en",
        system_prompt="Translate {source_language} to {target_language}",
    )
    transport = transports[0]
    first = _request(owner, "self", spec=spec)
    _, first_slot = await _started(transport)
    second = _request(owner, "peer", spec=spec, reverse=reverse)
    _, second_slot = await _started(transport)
    third = _request(owner, "manual", spec=spec, reverse=True)
    await asyncio.sleep(0)
    assert first_slot != second_slot
    assert transport.peak_active == 2
    assert len(commands) == len(processes) == 1
    assert not third.done()
    transport.gates["self"].set()
    assert (await first).text == "self"
    _, third_slot = await _started(transport)
    assert third_slot != second_slot
    assert transport.peak_active == 2
    transport.gates["manual"].set()
    assert (await third).text == "manual"
    transport.gates["peer"].set()
    assert (await second).text == "peer"
    assert len(transport.prefixes) == 3
    assert not processes[0].terminated
    await owner.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_gpu", [False, True], ids=["cpu", "gpu-fallback"])
async def test_effective_cpu_serializes_pending_requests(tmp_path, fail_gpu):
    owner, _commands, _processes, transports, _calls, _logs = _runtime(
        tmp_path, fail_gpu=fail_gpu, transport_builder=ControlledTransport
    )
    backend = "gpu" if fail_gpu else "cpu"
    await owner.prepare(
        backend=backend, source_language="ko", target_language="en", system_prompt="translate"
    )
    transport = transports[0]
    first = _request(owner, "self", backend=backend)
    _, first_slot = await _started(transport)
    second = _request(owner, "peer", backend=backend, reverse=True)
    await asyncio.sleep(0)
    assert transport.started.empty()
    transport.gates["self"].set()
    assert (await first).text == "self"
    await _started(transport)
    transport.gates["peer"].set()
    assert (await second).text == "peer"
    assert transport.peak_active == 1
    assert owner.readiness.effective_backend == "cpu"
    await owner.close()


@pytest.mark.asyncio
async def test_cancelled_request_drains_without_blocking_other_channel_or_reusing_busy_slot(
    tmp_path,
):
    owner, _commands, processes, transports, _calls, _logs = _runtime(
        tmp_path, transport_builder=ControlledTransport
    )
    await owner.prepare(
        backend="gpu", source_language="ko", target_language="en", system_prompt="translate"
    )
    transport = transports[0]
    first = _request(owner, "cancelled")
    _, cancelled_slot = await _started(transport)
    second = _request(owner, "valid", reverse=True)
    _, valid_slot = await _started(transport)
    first.cancel()
    first.cancel()
    third = _request(owner, "next")
    await asyncio.sleep(0)
    assert not first.done()
    assert not processes[0].terminated
    transport.gates["valid"].set()
    assert (await second).text == "valid"
    _, next_slot = await _started(transport)
    assert next_slot != cancelled_slot
    transport.gates["next"].set()
    assert (await third).text == "next"
    assert not first.done()
    transport.gates["cancelled"].set()
    with pytest.raises(asyncio.CancelledError):
        await first
    fourth = _request(owner, "reuse", reverse=True)
    await _started(transport)
    transport.gates["reuse"].set()
    assert (await fourth).text == "reuse"
    assert not processes[0].terminated
    await owner.close()


@pytest.mark.asyncio
async def test_failure_waits_for_server_settlement_while_other_work_completes(tmp_path):
    owner, _commands, processes, transports, _calls, _logs = _runtime(
        tmp_path, transport_builder=ControlledTransport
    )
    await owner.prepare(
        backend="gpu", source_language="ko", target_language="en", system_prompt="translate"
    )
    transport = transports[0]
    first = _request(owner, "failed")
    _, failed_slot = await _started(transport)
    second = _request(owner, "valid", reverse=True)
    await _started(transport)
    transport.failures.add("failed")
    transport.settlements[failed_slot] = asyncio.Event()
    transport.gates["failed"].set()
    await transport.settling.wait()
    third = _request(owner, "next")
    transport.gates["valid"].set()
    assert (await second).text == "valid"
    _, next_slot = await _started(transport)
    assert next_slot != failed_slot
    transport.gates["next"].set()
    assert (await third).text == "next"
    assert not first.done()
    transport.settlements[failed_slot].set()
    with pytest.raises(RuntimeError, match="request failed"):
        await first
    assert not processes[0].terminated
    await owner.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("new_model", [False, True], ids=["backend", "model"])
async def test_runtime_transition_drains_work_and_rejects_superseded_results(tmp_path, new_model):
    owner, commands, processes, transports, _calls, _logs = _runtime(
        tmp_path, transport_builder=ControlledTransport
    )
    await owner.prepare(
        backend="gpu", source_language="ko", target_language="en", system_prompt="translate"
    )
    transport = transports[0]
    first = _request(owner, "self")
    await _started(transport)
    second = _request(owner, "peer", reverse=True)
    await _started(transport)
    change = asyncio.create_task(
        owner.prepare(
            backend="gpu" if new_model else "cpu",
            spec=gemma_26b_spec() if new_model else e4b_gemma_spec(),
            source_language="ja",
            target_language="en",
            system_prompt="translate",
        )
    )
    await asyncio.sleep(0)
    assert owner.readiness is None
    assert len(commands) == 1
    assert not processes[0].terminated
    transport.gates["self"].set()
    with pytest.raises(ManagedGemmaRuntimeError, match="superseded"):
        await first
    assert not processes[0].terminated
    transport.gates["peer"].set()
    with pytest.raises(ManagedGemmaRuntimeError, match="superseded"):
        await second
    ready = await asyncio.wait_for(change, timeout=1.0)
    assert ready.effective_backend == ("vulkan" if new_model else "cpu")
    assert len(commands) == 2
    assert processes[0].terminated
    await owner.close()


@pytest.mark.asyncio
async def test_prefix_failure_fallback_drains_other_valid_gpu_work_before_stopping(tmp_path):
    owner, commands, processes, transports, _calls, _logs = _runtime(
        tmp_path, transport_builder=ControlledTransport
    )
    await owner.prepare(
        backend="gpu", source_language="ko", target_language="en", system_prompt="translate"
    )
    transport = transports[0]
    first = _request(owner, "valid")
    await _started(transport)
    transport.fail_prompt = "fail prefix"
    change = asyncio.create_task(
        owner.prepare(
            backend="gpu",
            source_language="ja",
            target_language="en",
            system_prompt="fail prefix",
        )
    )
    await asyncio.sleep(0)
    assert not processes[0].terminated
    transport.gates["valid"].set()
    assert (await first).text == "valid"
    ready = await asyncio.wait_for(change, timeout=1.0)
    assert ready.effective_backend == "cpu"
    assert processes[0].terminated
    assert len(commands) == 2
    await owner.close()


@pytest.mark.asyncio
async def test_26b_rejects_cpu_before_assets_and_never_falls_back_on_gpu_failure(tmp_path):
    owner, commands, _processes, _transports, calls, _logs = _runtime(tmp_path, fail_gpu=True)
    arguments = dict(
        spec=gemma_26b_spec(), source_language="ko", target_language="en", system_prompt="translate"
    )
    with pytest.raises(ManagedGemmaRuntimeError, match="requires a working Vulkan GPU"):
        await owner.prepare(backend="cpu", **arguments)
    assert calls == commands == []
    with pytest.raises(ManagedGemmaRuntimeError, match="startup failed"):
        await owner.prepare(backend="gpu", **arguments)
    assert len(commands) == 1
    assert "--device" in commands[0]
    assert owner.readiness is None


@pytest.mark.asyncio
async def test_26b_prefix_failure_never_substitutes_cpu_or_e4b(tmp_path):
    owner, commands, processes, _transports, _calls, _logs = _runtime(
        tmp_path, transport_failures=(RuntimeError("GPU memory"),)
    )
    with pytest.raises(ManagedGemmaRuntimeError, match="prefix preparation failed"):
        await owner.prepare(
            backend="gpu",
            spec=gemma_26b_spec(),
            source_language="ko",
            target_language="en",
            system_prompt="translate",
        )
    assert len(commands) == 1
    assert processes[0].terminated
    assert owner.readiness is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "version,returncode,valid",
    [
        ("version: 0.6.0-dev (build 11537, commit 8e2d31e0e)", 0, True),
        (
            "version: 0.6.0-dev (build 11537, commit 8e2d31e0eb658d0382d3106a61369dff81f0c080)",
            0,
            True,
        ),
        ("version: 0.6.0-dev (build 10423, commit a94d563ed)", 0, False),
        ("version: 0.6.0-dev (build 11537, commit a94d563ed)", 0, False),
        ("version: 0.6.0-dev (build 11537, commit 8e2d31e0e)", 1, False),
        ("unknown", 0, False),
    ],
)
async def test_executable_version_rejects_unpinned_build_or_commit(
    tmp_path,
    monkeypatch,
    version,
    returncode,
    valid,
):
    class VersionProcess:
        async def communicate(self):
            return version.encode(), None

    process = VersionProcess()
    process.returncode = returncode
    calls = []

    async def execute(*args, **kwargs):
        calls.append(args)
        return process

    monkeypatch.setattr(runtime_module.asyncio, "create_subprocess_exec", execute)
    executable = tmp_path / "override" / "llama-server.exe"
    if valid:
        await runtime_module.validate_gemma_runtime_version(executable)
    else:
        with pytest.raises(ManagedGemmaRuntimeError, match="Incompatible llama.cpp executable"):
            await runtime_module.validate_gemma_runtime_version(executable)
    assert calls == [(str(executable), "--version")]


@pytest.mark.asyncio
async def test_incompatible_runtime_override_is_rejected_without_cpu_substitution(tmp_path):
    owner, commands, _processes, _transports, _calls, _logs = _runtime(tmp_path)

    async def reject(_executable):
        raise runtime_module.ManagedGemmaRuntimeVersionError("Use b11537; remove override")

    owner._version_validator = reject
    with pytest.raises(ManagedGemmaRuntimeError, match="remove override"):
        await owner.prepare(
            backend="gpu",
            source_language="ko",
            target_language="en",
            system_prompt="translate",
        )
    assert commands == []
    assert owner.readiness is None


@pytest.mark.asyncio
async def test_missing_runtime_executable_fails_before_launch_with_distinct_error(tmp_path):
    owner, commands, _processes, _transports, _calls, _logs = _runtime(tmp_path)
    (tmp_path / "cpu" / "llama-server.exe").unlink()
    (tmp_path / "gpu" / "llama-server.exe").unlink()

    with pytest.raises(ManagedGemmaRuntimeError) as caught:
        await owner.prepare(
            backend="cpu",
            source_language="ko",
            target_language="en",
            system_prompt="translate",
        )
    assert isinstance(caught.value.__cause__, runtime_module.ManagedGemmaRuntimeMissingError)
    assert commands == []
    assert owner.readiness is None


@pytest.mark.asyncio
async def test_prefix_identity_separates_effective_backend_and_fixed_profile(tmp_path, monkeypatch):
    owner, _commands, _processes, _transports, _calls, _logs = _runtime(tmp_path)
    arguments = dict(source_language="ko", target_language="en", system_prompt="translate")
    cpu = await owner.prepare(backend="cpu", **arguments)
    gpu = await owner.prepare(backend="gpu", **arguments)
    assert cpu.prefix_identity != gpu.prefix_identity
    monkeypatch.setattr(runtime_module, "GEMMA_RUNTIME_PROFILE_VERSION", "replacement-profile")
    upgraded = await owner.prepare(backend="gpu", **arguments)
    assert upgraded.prefix_identity != gpu.prefix_identity
    await owner.close()


@pytest.mark.asyncio
async def test_lru_eviction_keeps_both_active_prefixes_untouched(tmp_path):
    owner, _commands, _processes, transports, _calls, _logs = _runtime(
        tmp_path, transport_builder=ControlledTransport
    )
    template = "Translate {source_language} to {target_language}"
    for source in ("ko", "ja", "de"):
        await owner.prepare(
            backend="gpu",
            source_language=source,
            target_language="en",
            system_prompt=template,
        )
    transport = transports[0]
    first = _request(owner, "self")
    _, first_slot = await _started(transport)
    second = _request(owner, "peer", reverse=True)
    _, second_slot = await _started(transport)
    third = asyncio.create_task(
        owner.prepare(
            backend="gpu",
            source_language="fr",
            target_language="en",
            system_prompt=template,
        )
    )
    await asyncio.sleep(0)
    assert not third.done()
    transport.gates["self"].set()
    assert (await first).text == "self"
    await asyncio.wait_for(third, timeout=1)
    assert transport.prefix_slots[-1] != second_slot
    assert transport.active == {second_slot: "peer"}
    transport.gates["peer"].set()
    assert (await second).text == "peer"
    await owner.prepare(
        backend="gpu",
        source_language="en",
        target_language="ko",
        system_prompt=template,
    )
    assert transport.prefix_slots[-1] != second_slot
    await owner.close()


@pytest.mark.asyncio
async def test_shutdown_drains_both_gpu_requests_and_rejects_waiting_manual_request(tmp_path):
    owner, _commands, processes, transports, _calls, _logs = _runtime(
        tmp_path, transport_builder=ControlledTransport
    )
    await owner.prepare(
        backend="gpu",
        source_language="ko",
        target_language="en",
        system_prompt="translate",
    )
    transport = transports[0]
    first = _request(owner, "self")
    await _started(transport)
    second = _request(owner, "peer", reverse=True)
    await _started(transport)
    third = _request(owner, "manual")
    closing = asyncio.create_task(owner.close())
    await asyncio.sleep(0)
    assert owner.readiness is None
    assert not processes[0].terminated
    transport.gates["self"].set()
    transport.gates["peer"].set()
    outcomes = await asyncio.gather(first, second, third, return_exceptions=True)
    assert all(isinstance(outcome, asyncio.CancelledError) for outcome in outcomes)
    await closing
    assert processes[0].terminated
    assert transport.closed


@pytest.mark.asyncio
async def test_unobservable_failed_slot_is_not_reused_before_other_result_and_restart(tmp_path):
    class UnobservableTransport(ControlledTransport):
        async def wait_slot_idle(self, *, slot_id: int) -> None:
            self.settling.set()
            raise TimeoutError("slot state unavailable")

    owner, commands, processes, transports, _calls, _logs = _runtime(
        tmp_path, transport_builder=UnobservableTransport
    )
    await owner.prepare(
        backend="gpu",
        source_language="ko",
        target_language="en",
        system_prompt="translate",
    )
    transport = transports[0]
    first = _request(owner, "failed")
    await _started(transport)
    second = _request(owner, "valid", reverse=True)
    await _started(transport)
    transport.failures.add("failed")
    transport.gates["failed"].set()
    with pytest.raises(RuntimeError, match="request failed"):
        await first
    replacement = asyncio.create_task(
        owner.prepare(
            backend="gpu",
            source_language="ja",
            target_language="en",
            system_prompt="translate",
        )
    )
    await asyncio.sleep(0)
    assert len(commands) == 1
    assert not processes[0].terminated
    transport.gates["valid"].set()
    assert (await second).text == "valid"
    await asyncio.wait_for(replacement, timeout=1)
    assert processes[0].terminated
    assert len(commands) == 2
    await owner.close()


@pytest.mark.asyncio
async def test_cancelled_prefix_drain_does_not_block_other_completed_result(tmp_path):
    class BlockingPrefixTransport(ControlledTransport):
        def __init__(self, base_url):
            super().__init__(base_url)
            self.prefix_started = asyncio.Event()
            self.finish_prefix = asyncio.Event()

        async def prepare_prefix(self, *, system_prompt, slot_id):
            if system_prompt == "blocked prefix":
                self.prefix_started.set()
                await self.finish_prefix.wait()
            await super().prepare_prefix(system_prompt=system_prompt, slot_id=slot_id)

    owner, _commands, processes, transports, _calls, _logs = _runtime(
        tmp_path, transport_builder=BlockingPrefixTransport
    )
    await owner.prepare(
        backend="gpu",
        source_language="ko",
        target_language="en",
        system_prompt="translate",
    )
    transport = transports[0]
    valid = _request(owner, "valid")
    await _started(transport)
    preparation = asyncio.create_task(
        owner.prepare(
            backend="gpu",
            source_language="ja",
            target_language="en",
            system_prompt="blocked prefix",
        )
    )
    await transport.prefix_started.wait()
    preparation.cancel()
    await asyncio.sleep(0)
    transport.gates["valid"].set()
    assert (await asyncio.wait_for(valid, timeout=1)).text == "valid"
    assert not preparation.done()
    assert not processes[0].terminated
    transport.finish_prefix.set()
    with pytest.raises(asyncio.CancelledError):
        await preparation
    assert not processes[0].terminated
    await owner.prepare(
        backend="gpu",
        source_language="ja",
        target_language="en",
        system_prompt="blocked prefix",
    )
    assert len(processes) == 1
    await owner.close()
