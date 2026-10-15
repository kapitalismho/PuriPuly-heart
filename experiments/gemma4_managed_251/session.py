from __future__ import annotations

import argparse
import asyncio
import dataclasses
import hashlib
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import traceback
from collections.abc import Mapping
from pathlib import Path
from uuid import UUID, uuid4

from windows_metrics import C, MemoryStatus, Sampler

COMPARISON_SEED = 2512026


SYNTHETIC = (
    "Mina has 7 blue tickets, not 9. Please meet her at gate 3 after lunch.",
    "The small robot is called Nori. Its battery is 42 percent; do not turn it off.",
    "Correction: the meeting is on Tuesday at 14:30, not Thursday at 13:40.",
    "Please carry the red box to room 12 and leave the green box beside the door.",
)
KOREAN = (
    "미나는 파란 표 7장을 가지고 있어요. 9장이 아니에요. 점심 후 3번 입구에서 만나요.",
    "작은 로봇의 이름은 노리예요. 배터리가 42퍼센트 남았으니 끄지 마세요.",
    "정정할게요. 회의는 목요일 13시 40분이 아니라 화요일 14시 30분이에요.",
    "빨간 상자는 12번 방으로 옮기고 초록 상자는 문 옆에 놓아 주세요.",
)


def plain(value):
    if dataclasses.is_dataclass(value):
        return {
            field.name: plain(getattr(value, field.name)) for field in dataclasses.fields(value)
        }
    if isinstance(value, Mapping):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [plain(item) for item in value]
    if isinstance(value, (Path, UUID)):
        return str(value)
    if hasattr(value, "value"):
        return plain(value.value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return type(value).__name__


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class Journal:
    def __init__(self, path: Path):
        self.stream = path.open("a", encoding="utf-8", buffering=1)
        self.lock = threading.Lock()

    def add(self, kind: str, **data):
        value = {"t": time.perf_counter(), "kind": kind, **plain(data)}
        with self.lock:
            self.stream.write(json.dumps(value, ensure_ascii=False) + "\n")
        return value

    def close(self):
        self.stream.close()


def identify(source: Path) -> dict:
    import puripuly_heart
    from puripuly_heart.runtime_layout import current_runtime_layout

    files = {
        str(path.relative_to(source)).replace("\\", "/"): sha(path)
        for path in sorted((source / "src" / "puripuly_heart").rglob("*.py"))
    }
    git = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=source, capture_output=True, text=True, check=True
    )
    return {
        "head": git.stdout.strip(),
        "python": sys.executable,
        "python_version": sys.version,
        "import": str(Path(puripuly_heart.__file__).resolve()),
        "layout": plain(current_runtime_layout()),
        "source_sha256": files,
        "source_tree_sha256": hashlib.sha256(
            json.dumps(files, sort_keys=True).encode()
        ).hexdigest(),
        "harness_sha256": {
            path.name: sha(path) for path in sorted(Path(__file__).parent.glob("*.py"))
        },
    }


def prepare(args):
    from puripuly_heart.core.local_translation.assets import (
        InstalledGemmaManifest,
        e4b_gemma_spec,
        gemma_26b_spec,
        validate_gemma_install,
    )

    args.assets.mkdir(parents=True, exist_ok=True)
    evidence = {
        "candidate": identify(args.source),
        "models": {},
        "runtime": {"root": str(args.runtime_root), "files": {}},
    }
    for spec, original in (
        (e4b_gemma_spec(), args.e4b_source),
        (gemma_26b_spec(), args.model26_source.parent),
    ):
        target = args.assets / spec.install_dirname
        target.mkdir(parents=True, exist_ok=True)
        observations = []
        for asset in spec.assets:
            source = original / asset.filename
            observed = {
                "filename": asset.filename,
                "size_bytes": source.stat().st_size,
                "sha256": sha(source),
            }
            if observed["size_bytes"] != asset.size_bytes or observed["sha256"] != asset.sha256:
                raise RuntimeError(f"asset identity mismatch: {asset.filename}")
            destination = target / asset.filename
            if not destination.exists():
                os.link(source, destination)
            if not os.path.samefile(source, destination):
                raise RuntimeError("prepared assets must be verified hardlinks")
            observations.append(observed)
        (target / "installed-manifest.json").write_text(
            json.dumps(InstalledGemmaManifest.expected(spec).to_dict()), encoding="utf-8"
        )
        validate_gemma_install(target, spec=spec, verify_checksums=False)
        evidence["models"][spec.model_id] = {
            "spec": plain(spec),
            "files": observations,
            "originals_modified": False,
        }
    for backend in ("cpu", "vulkan"):
        files = {
            str(path.relative_to(args.runtime_root)).replace("\\", "/"): {
                "size_bytes": path.stat().st_size,
                "sha256": sha(path),
            }
            for path in sorted((args.runtime_root / backend).iterdir())
            if path.is_file()
        }
        evidence["runtime"]["files"].update(files)
    evidence["runtime"]["manifest_sha256"] = sha(args.runtime_root / "manifest.json")
    (args.assets / "assets.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    (args.root / "prepare.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")


def stage_model(args, spec):
    from puripuly_heart.core.local_translation.assets import (
        default_gemma_install_dir,
        validate_gemma_install,
    )

    original = args.assets / spec.install_dirname
    destination = default_gemma_install_dir(spec)
    destination.mkdir(parents=True, exist_ok=True)
    for asset in spec.assets:
        source = original / asset.filename
        if sha(source) != asset.sha256:
            raise RuntimeError(f"model identity changed before lifecycle: {asset.filename}")
        os.link(source, destination / asset.filename)
    shutil.copyfile(original / "installed-manifest.json", destination / "installed-manifest.json")
    validate_gemma_install(destination, verify_checksums=False, spec=spec)


def seed(args, sink_port: int):
    from puripuly_heart.app.adapters.settings_vnext_canonical_persistence import (
        SettingsVNextCanonicalPersistenceAdapter,
    )
    from puripuly_heart.config.settings_vnext.defaults import new_settings_for_first_run
    from puripuly_heart.config.settings_vnext.schema import SecretsIntent, with_telemetry_enabled
    from puripuly_heart.core.local_translation.assets import e4b_gemma_spec, gemma_26b_spec

    manifest = json.loads((args.assets / "assets.json").read_text(encoding="utf-8"))
    runtime = Path(manifest["runtime"]["root"])
    expected = args.source / "build" / "llama.cpp" / "llama.cpp-b11537"
    if runtime.resolve() != expected.resolve():
        os.environ["PURIPULY_HEART_LLAMA_CPP_ROOT"] = str(runtime)
    for relative, entry in manifest["runtime"]["files"].items():
        if sha(runtime / relative) != entry["sha256"]:
            raise RuntimeError(f"runtime staged identity changed: {relative}")
    stage_model(args, gemma_26b_spec() if args.model == "managed_gemma_26b" else e4b_gemma_spec())
    canonical = new_settings_for_first_run("en-US")
    canonical = dataclasses.replace(
        canonical,
        intent=dataclasses.replace(
            canonical.intent,
            secrets=SecretsIntent("encrypted_file", "synthetic-secrets.json"),
            stt=dataclasses.replace(canonical.intent.stt, provider="soniox"),
            peer_stt=dataclasses.replace(canonical.intent.peer_stt, provider="soniox"),
            clipboard=dataclasses.replace(canonical.intent.clipboard, auto_translate_enabled=False),
            osc=dataclasses.replace(
                canonical.intent.osc,
                connection_mode="off",
                host="127.0.0.1",
                port=sink_port,
                send_port=sink_port,
                activation_notice_enabled=False,
                chatbox_include_source=False,
            ),
            translation=dataclasses.replace(
                canonical.intent.translation, model=args.model, connection=args.backend
            ),
            languages=dataclasses.replace(
                canonical.intent.languages,
                source_language="en",
                target_language="ko",
                peer_source_language="en",
                peer_target_language="ko",
                peer_source_mode="manual",
            ),
        ),
    )
    config = args.root / "settings.json"
    SettingsVNextCanonicalPersistenceAdapter().persist(
        config, with_telemetry_enabled(canonical, False)
    )
    return config


class OscSink:
    def __init__(self, journal):
        self.journal = journal
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind(("127.0.0.1", 0))
        self.socket.settimeout(0.1)
        self.port = self.socket.getsockname()[1]
        self.stop_event = threading.Event()
        self.messages = []
        self.thread = threading.Thread(target=self.receive, daemon=True)
        self.thread.start()

    def receive(self):
        from pythonosc.osc_packet import OscPacket

        while not self.stop_event.is_set():
            try:
                data, sender = self.socket.recvfrom(65535)
            except socket.timeout:
                continue
            except OSError:
                return
            try:
                messages = [
                    {"address": timed.message.address, "parameters": timed.message.params}
                    for timed in OscPacket(data).messages
                ]
                self.messages.extend(messages)
                self.journal.add(
                    "osc_received", messages=messages, sender_loopback=sender[0] == "127.0.0.1"
                )
            except Exception as error:
                self.journal.add("osc_decode_failure", error=type(error).__name__, bytes=len(data))

    def close(self):
        self.stop_event.set()
        self.thread.join(timeout=2)
        self.socket.close()


class Capture:
    def __init__(self, process, command, original, path, journal):
        self.process = process
        self.command = command
        self.original = original
        self.path = path
        self.journal = journal
        self.reader = asyncio.create_task(self.drain())

    @property
    def returncode(self):
        return self.process.returncode

    def terminate(self):
        self.journal.add("owned_terminate", pid=self.process.pid)
        self.process.terminate()

    def kill(self):
        self.journal.add("owned_kill", pid=self.process.pid)
        self.process.kill()

    async def wait(self):
        result = await self.process.wait()
        await self.reader
        self.journal.add("owned_exit", pid=self.process.pid, exit_code=result)
        return result

    async def drain(self):
        with self.path.open("wb") as output:
            while line := await self.process.stdout.readline():
                output.write(line)
                self.journal.add(
                    "native",
                    pid=self.process.pid,
                    line=line.decode("utf-8", errors="replace").rstrip(),
                )


class Monitor:
    def __init__(self, pid, journal, phase):
        self.pid, self.journal, self.phase = pid, journal, phase
        self.stop_event = threading.Event()
        self.cutoff = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def run(self):
        sampler = Sampler(self.pid, adapter="")
        low_since = None
        selected = None
        self.journal.add(
            "counter_setup",
            errors=sampler.errors,
            adapter_selection="discover_from_owned_process_instances",
        )
        try:
            while not self.stop_event.is_set():
                sample = sampler.sample()
                process_labels = sampler.instances.get("process_gpu_dedicated_bytes", [])
                adapters = sorted(
                    set(
                        match.group(0).lower()
                        for label in process_labels
                        if (
                            match := re.search(
                                r"luid_0x[0-9a-f]+_0x[0-9a-f]+_phys_\d+", label, re.I
                            )
                        )
                    )
                )
                if selected is None and len(adapters) == 1:
                    selected = adapters[0]
                    sampler.adapter = selected
                    self.journal.add(
                        "adapter_discovered",
                        adapter=selected,
                        process_instance_labels=process_labels,
                    )
                self.journal.add(
                    "resource",
                    phase=self.phase(),
                    pid=self.pid,
                    sample=sample,
                    adapter=selected,
                    counter_instances=sampler.instances,
                )
                low = (
                    sample["system_available_physical_bytes"] < 256 * 1024**2
                    or sample["system_commit_headroom_bytes"] < 512 * 1024**2
                )
                low_since = (low_since or sample["time"]) if low else None
                if low_since is not None and sample["time"] - low_since >= 8:
                    self.cutoff.set()
                    self.journal.add(
                        "watchdog_cutoff",
                        sustained_s=sample["time"] - low_since,
                        action="owned_host_shutdown_requested",
                        sample=sample,
                    )
                    return
                self.stop_event.wait(0.25)
        except BaseException as error:
            self.journal.add("counter_failure", error=type(error).__name__, detail=str(error))
        finally:
            sampler.close()

    def close(self):
        self.stop_event.set()
        self.thread.join(timeout=5)


async def cli(config: Path, journal: Journal, *arguments: str, stdin=None, timeout=180):
    command = [
        sys.executable,
        "-m",
        "puripuly_heart.main",
        "cli",
        "--config",
        str(config),
        "--timeout",
        str(timeout),
        *arguments,
    ]
    started = time.perf_counter()
    process = await asyncio.create_subprocess_exec(
        *command,
        stdin=asyncio.subprocess.PIPE if stdin is not None else asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await asyncio.wait_for(
        process.communicate(None if stdin is None else stdin.encode("utf-8")), timeout + 15
    )
    parsed = [json.loads(line) for line in stdout.decode("utf-8-sig").splitlines() if line.strip()]
    journal.add(
        "cli",
        arguments=list(arguments),
        exit_code=process.returncode,
        output=parsed,
        stderr=stderr.decode("utf-8", errors="replace"),
        started=started,
        completed=time.perf_counter(),
    )
    if process.returncode:
        receipt = parsed[-1] if parsed else {}
        transaction = receipt.get("transaction", {})
        diagnostics = transaction.get("diagnostics", {})
        inactive_stt_only = (
            process.returncode == 4
            and tuple(arguments[:2]) == ("settings", "apply")
            and receipt.get("terminal") is True
            and receipt.get("status") == "degraded"
            and transaction.get("status") == "settings_commit_success_runtime_degraded"
            and diagnostics.get("component") == "provider.stt"
            and diagnostics.get("code") == "stt.provider_not_ready"
        )
        if not inactive_stt_only:
            raise RuntimeError(
                f"CLI command failed: {' '.join(arguments[:2])}: {process.returncode}"
            )
        current = await cli(config, journal, "settings", "current")
        state = await cli(config, journal, "app", "status")
        changes = json.loads(arguments[arguments.index("--arguments") + 1])["changes"]
        intent = current["settings"]["intent"]
        for field, value in changes.items():
            if field == "languages":
                names = {
                    "source": "source_language",
                    "target": "target_language",
                    "peer_source": "peer_source_language",
                    "peer_target": "peer_target_language",
                }
                if any(intent["languages"][names[key]] != item for key, item in value.items()):
                    raise RuntimeError("committed language state does not match synthetic request")
            elif field.startswith("translation."):
                if intent["translation"][field.split(".", 1)[1]] != value:
                    raise RuntimeError(
                        "committed translation state does not match synthetic request"
                    )
        if any(
            item["desired_active"] or item["effective_active"]
            for item in state["channels"].values()
        ):
            raise RuntimeError("inactive STT degradation unexpectedly enabled physical capture")
        journal.add(
            "known_inactive_stt_degradation",
            original_receipt=receipt,
            committed_fields=changes,
            capture_off_observed=True,
        )
    return parsed[-1]


async def poll_until(predicate, timeout=120):
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0.01)


class Fixture:
    def __init__(self, args, journal):
        self.args, self.journal = args, journal
        self.phase = "startup"
        self.captures, self.monitors = [], []
        self.requests, self.results, self.events = [], [], []
        self.units = {"self": 0, "peer": 0}
        self.generation = 1
        self.tasks = set()
        self.host = self.access = self.control = None
        self.sink = OscSink(journal)
        self.config = seed(args, self.sink.port)
        self.observed_slots = []
        self.finished = asyncio.Event()
        self.checks = {}
        self.spawn_fault_used = False
        self.response_fault_channel = None
        self.response_fault_used = False
        self.native_completions = []
        from contextvars import ContextVar

        self.http_context = ContextVar("issue251_native_http_observation", default=None)
        self.observed_http_clients = set()
        self.fixed_history = True

    def task(self, awaitable):
        task = asyncio.create_task(awaitable)
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)
        return task

    def validate_profile(self, command):
        import math

        import psutil

        def value(option):
            return command[command.index(option) + 1]

        cpu = value("--device") == "none"
        model26 = "26B-A4B" in value("--model")
        expected = {
            "--ctx-size": "4608",
            "--parallel": "3",
            "--batch-size": "512",
            "--ubatch-size": "512",
            "--cache-type-k": "f16",
            "--cache-type-v": "f16",
            "--reasoning": "off",
            "--reasoning-budget": "0",
            "--prio": "-1",
            "--threads-http": "1",
            "--host": "127.0.0.1",
        }
        physical, logical = psutil.cpu_count(logical=False), psutil.cpu_count(logical=True)
        if cpu:
            if (
                self.args.case not in {"cpu", "fallback"}
                and self.phase != "active-backend-transition"
            ):
                raise RuntimeError("unexpected CPU fallback invalidates the intended GPU execution")
            expected.update(
                {
                    "--load-mode": "mmap",
                    "--n-gpu-layers": "0",
                    "--flash-attn": "off",
                    "--threads": str(max(1, min(math.ceil(physical * 0.5), 6))),
                    "--threads-batch": str(max(1, min(math.ceil(logical * 0.75), 12))),
                    "--spec-type": "draft-mtp",
                    "--spec-draft-n-max": "3",
                    "--spec-draft-n-min": "2",
                    "--spec-draft-p-min": "0.0",
                    "--spec-draft-device": "none",
                    "--spec-draft-ngl": "0",
                    "--spec-draft-threads": "1",
                    "--spec-draft-threads-batch": "1",
                    "--spec-draft-type-k": "f16",
                    "--spec-draft-type-v": "f16",
                }
            )
            if model26 or Path(value("--spec-draft-model")).name != "mtp-gemma-4-E4B-it.gguf":
                raise RuntimeError("unsupported model/draft reached CPU profile")
        else:
            expected.update(
                {
                    "--n-gpu-layers": "99",
                    "--flash-attn": "on",
                    "--threads": "4",
                    "--threads-batch": "4",
                }
            )
            if "--spec-draft-model" in command:
                raise RuntimeError("GPU profile unexpectedly loads a drafter")
            if model26:
                expected.update(
                    {
                        "--load-mode": "none",
                        "--n-cpu-moe": "24",
                        "--moe-cache-mib": "2048",
                        "--cache-ram": "0",
                        "--fit": "off",
                    }
                )
                if "--swa-full" not in command:
                    raise RuntimeError("26B profile lost full sliding-window cache")
            else:
                expected["--load-mode"] = self.args.mode if self.args.mode != "product" else "mmap"
                if any(
                    option in command
                    for option in ("--n-cpu-moe", "--moe-cache-mib", "--swa-full", "--cache-ram")
                ):
                    raise RuntimeError("26B tuning leaked into E4B profile")
        if any(value(option) != item for option, item in expected.items()):
            raise RuntimeError("effective managed profile differs from issue251 contract")
        if any(
            option not in command
            for option in ("--cache-prompt", "--warmup", "--perf", "--metrics", "--no-webui")
        ):
            raise RuntimeError("managed profile lost required lifecycle/cache arguments")
        self.journal.add(
            "profile_contract",
            effective_backend="cpu" if cpu else "vulkan",
            model="26B" if model26 else "E4B",
            physical_cores=physical,
            logical_cores=logical,
            required_values=expected,
            matched=True,
        )

    async def process_factory(self, command, cwd):
        original = tuple(command)
        effective = list(command)
        if (
            self.args.case in {"fallback", "26b-failure"}
            and not self.spawn_fault_used
            and "vulkan" in Path(command[0]).parts
        ):
            self.spawn_fault_used = True
            self.journal.add(
                "injected_spawn_failure",
                selected_model=self.args.model,
                original_command=original,
                failure_boundary="reject_Vulkan_spawn_once_before_native_process",
            )
            raise OSError("synthetic issue251 Vulkan spawn rejection")
        if self.args.mode != "product":
            if (
                "managed-gemma-e4b" not in " ".join(effective)
                and self.args.model != "managed_gemma"
            ):
                raise RuntimeError("loading-mode experiment is E4B only")
            index = effective.index("--load-mode")
            effective[index + 1] = self.args.mode
        effective += ["--verbosity", "4"]
        self.journal.add(
            "managed_spawn_attempt",
            phase=self.phase,
            original_command=original,
            effective_command=effective,
        )
        self.validate_profile(effective)
        baseline_started = time.perf_counter()
        baseline = MemoryStatus()
        baseline.length = C.sizeof(baseline)
        if not C.WinDLL("kernel32", use_last_error=True).GlobalMemoryStatusEx(C.byref(baseline)):
            raise C.WinError(C.get_last_error())
        self.journal.add(
            "prelaunch_system_baseline",
            phase=self.phase,
            observer_pid=os.getpid(),
            sample_started=baseline_started,
            sample_completed=time.perf_counter(),
            system_available_physical_bytes=baseline.available_physical,
            system_committed_bytes=baseline.total_pagefile - baseline.available_pagefile,
            system_commit_limit_bytes=baseline.total_pagefile,
            system_commit_headroom_bytes=baseline.available_pagefile,
        )
        process = await asyncio.create_subprocess_exec(
            *effective,
            cwd=cwd,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        capture = Capture(
            process,
            tuple(effective),
            original,
            self.args.root / f"native-{len(self.captures)}.log",
            self.journal,
        )
        self.captures.append(capture)
        if self.phase == "startup":
            self.phase = "model-load"
        self.journal.add(
            "managed_spawn",
            pid=process.pid,
            parent_pid=os.getpid(),
            original_command=original,
            effective_command=effective,
            executable_sha256=sha(Path(effective[0])),
            live_owned_model_processes=sum(item.returncode is None for item in self.captures),
        )
        self.monitors.append(Monitor(process.pid, self.journal, lambda: self.phase))
        self.task(self.slots(capture))
        return capture

    async def slots(self, capture):
        import httpx

        port = int(capture.command[capture.command.index("--port") + 1])
        async with httpx.AsyncClient(
            base_url=f"http://127.0.0.1:{port}", timeout=1, trust_env=False
        ) as client:
            while capture.returncode is None and not self.finished.is_set():
                query_started = time.perf_counter()
                query_phase = self.phase
                try:
                    response = await client.get("/slots")
                    if response.status_code == 200:
                        value = response.json()
                        row = self.journal.add(
                            "slots",
                            phase=query_phase,
                            query_started=query_started,
                            pid=capture.process.pid,
                            slots=value,
                        )
                        self.observed_slots.append(row)
                except httpx.HTTPError, ValueError:
                    pass
                await asyncio.sleep(0.025)

    async def start(self):
        from puripuly_heart.cli.host import HostedApplication
        from puripuly_heart.composition import application_runtime as composition
        from puripuly_heart.composition.headless_application import HeadlessApplicationPresentation
        from puripuly_heart.core.control_instance import acquire
        from puripuly_heart.core.windows_process_ownership import retain_current_process_job

        retain_current_process_job()
        original_create = composition.create_managed_gemma_runtime

        def observed_runtime(**kwargs):
            owner = original_create(**kwargs)
            owner._process_factory = self.process_factory
            sink = owner._log_sink

            def observe(message, level):
                self.journal.add("managed_log", message=message, level=level)
                if sink is not None:
                    sink(message, level)

            owner._log_sink = observe
            return owner

        composition.create_managed_gemma_runtime = observed_runtime
        from puripuly_heart.core.llm.latency import current_request
        from puripuly_heart.providers.llm.error_details import LLMResponseError
        from puripuly_heart.providers.llm.managed_gemma import HttpxManagedGemmaTransport

        original_completion = HttpxManagedGemmaTransport._completion
        original_require_client = HttpxManagedGemmaTransport._require_client

        def observed_require_client(transport):
            client = original_require_client(transport)
            if client not in self.observed_http_clients:
                self.observed_http_clients.add(client)
                original_post = client.post

                async def observed_post(url, *args, **kwargs):
                    if str(url) == "/completion":
                        original_body = kwargs["json"]
                        body = dict(original_body)
                        if self.args.case == "abba":
                            body["seed"] = COMPARISON_SEED
                        kwargs["json"] = body
                        prompt_hash = hashlib.sha256(body["prompt"].encode()).hexdigest()
                        context = self.http_context.get()
                        if context is not None:
                            context["native_request_prompt_sha256"] = prompt_hash
                        self.journal.add(
                            "native_http_payload",
                            phase=self.phase,
                            context=context,
                            original_parameters={
                                key: value
                                for key, value in original_body.items()
                                if key != "prompt"
                            },
                            effective_parameters={
                                key: value for key, value in body.items() if key != "prompt"
                            },
                            native_prompt_sha256=prompt_hash,
                            native_prompt_chars=len(body["prompt"]),
                            seed_control=(
                                "fixed_comparison_only"
                                if self.args.case == "abba"
                                else "unmodified_product"
                            ),
                        )
                    return await original_post(url, *args, **kwargs)

                client.post = observed_post
            return client

        HttpxManagedGemmaTransport._require_client = observed_require_client

        async def observed_completion(transport, **kwargs):
            request = current_request()
            messages = kwargs["messages"]
            purpose = (
                "prefix"
                if kwargs.get("max_tokens") == 1 and not messages[-1]["content"].strip()
                else "translation"
            )
            if purpose == "prefix" and request is None and self.phase == "model-load":
                self.phase = "startup:first-prefix"
            rendered = json.dumps(messages, ensure_ascii=False, separators=(",", ":"))
            started = time.perf_counter()
            port = transport.base_url.rsplit(":", 1)[-1].rstrip("/")
            capture = next(
                (
                    item
                    for item in self.captures
                    if item.command[item.command.index("--port") + 1] == port
                ),
                None,
            )
            native_pid = None if capture is None else capture.process.pid
            identity = (
                None
                if request is None
                else {
                    "utterance_id": str(request.utterance_id),
                    "channel": request.channel,
                    "source_language": request.source_language,
                    "target_language": request.target_language,
                    "parent_utterance_id": str(request.parent_utterance_id),
                }
            )
            self.journal.add(
                "native_request",
                identity=identity,
                pid=native_pid,
                slot_id=kwargs["slot_id"],
                phase=self.phase,
                messages=messages,
                rendered_messages_sha256=hashlib.sha256(rendered.encode()).hexdigest(),
                max_tokens=kwargs.get("max_tokens"),
                started=started,
                purpose=purpose,
            )
            http_observation = {
                "purpose": purpose,
                "identity": identity,
                "slot_id": kwargs["slot_id"],
                "pid": native_pid,
            }
            token = self.http_context.set(http_observation)
            try:
                payload = await original_completion(transport, **kwargs)
            finally:
                self.http_context.reset(token)
            echoed_hash = (
                hashlib.sha256(payload["prompt"].encode()).hexdigest()
                if isinstance(payload.get("prompt"), str)
                else None
            )
            echo_matches = echoed_hash == http_observation.get("native_request_prompt_sha256")
            echo_representation = "verbatim" if echo_matches else "unmatched"
            if (
                not echo_matches
                and isinstance(payload.get("prompt"), str)
                and payload["prompt"].startswith("<bos>")
                and hashlib.sha256(payload["prompt"][5:].encode()).hexdigest()
                == http_observation.get("native_request_prompt_sha256")
            ):
                echo_matches = True
                echo_representation = "native_added_single_leading_bos"
            row = self.journal.add(
                "native_completion",
                identity=identity,
                pid=native_pid,
                slot_id=kwargs["slot_id"],
                phase=self.phase,
                started=started,
                completed=time.perf_counter(),
                payload=payload,
                purpose=purpose,
                native_prompt_echo_matches=echo_matches,
                native_prompt_echo_representation=echo_representation,
                rendered_messages_sha256=hashlib.sha256(rendered.encode()).hexdigest(),
            )
            self.native_completions.append(row)
            if self.args.case == "abba" and (
                not echo_matches
                or payload.get("generation_settings", {}).get("seed") != COMPARISON_SEED
            ):
                raise RuntimeError(
                    "comparison native prompt/seed did not match the effective controlled payload"
                )
            if (
                purpose == "translation"
                and request is not None
                and request.channel == self.response_fault_channel
                and not self.response_fault_used
            ):
                self.response_fault_used = True
                self.journal.add(
                    "injected_response_failure",
                    identity=identity,
                    slot_id=kwargs["slot_id"],
                    native_completed=True,
                    failure_boundary="after_real_completion_before_client_acceptance",
                )
                raise LLMResponseError(
                    "synthetic issue251 response rejection after native completion",
                    provider="managed_gemma",
                )
            return payload

        HttpxManagedGemmaTransport._completion = observed_completion

        def retain(access):
            self.access = access

        presentation = HeadlessApplicationPresentation()
        boundary = composition.compose_application_runtime(
            presentation=presentation, config_path=self.config, local_asr_evidence_sink=retain
        )
        presentation.application = boundary
        self.host = HostedApplication(boundary, acquire(self.config))
        await self.host.start()
        self.control = boundary.control()
        self.task(self.consume_events())
        self.journal.add(
            "instance",
            host_pid=os.getpid(),
            instance_id=self.host.instance_id,
            candidate=identify(self.args.source),
            assets_manifest_sha256=sha(self.args.assets / "assets.json"),
        )
        for arguments in (
            ("capabilities",),
            ("settings", "choices"),
            ("settings", "current"),
            ("app", "status"),
            ("capture", "status"),
            ("osc", "status"),
            ("models", "status"),
        ):
            await cli(self.config, self.journal, *arguments)
        state = await self.control.query("app.status", {})
        self.journal.add("pre_translation_state", state=state)
        if any(
            channel["desired_active"] or channel["effective_active"]
            for channel in state["channels"].values()
        ):
            raise RuntimeError("physical capture must remain off")
        if (
            state["microphone_test"]["desired_active"]
            or state["microphone_test"]["effective_active"]
        ):
            raise RuntimeError("microphone test must remain off")
        if self.args.case == "preflight":
            self.checks["preflight_no_managed_process"] = not self.captures
            return
        if self.args.case == "26b-failure":
            try:
                await cli(self.config, self.journal, "translation", "set", "on", timeout=240)
            except RuntimeError:
                self.journal.add("expected_control_rejection", boundary="26B_Vulkan_spawn_failure")
            status = await self.control.query("models.status", {})
            owner = self.control.gemma()
            native_readiness = None if owner is None else owner.runtime.readiness
            self.journal.add(
                "no_fallback_state",
                models=status,
                app=await self.control.query("app.status", {}),
                native_readiness=native_readiness,
            )
            if not self.spawn_fault_used or self.captures:
                raise RuntimeError("26B failure unexpectedly spawned fallback runtime")
            application = await self.control.query("app.status", {})
            gemma = status.get("gemma") or {}
            if (
                gemma.get("state") != "failed"
                or gemma.get("backend") != "gpu"
                or gemma.get("model_id") != "gemma-4-26b-a4b-it-qat-ud-q4-k-xl"
                or gemma.get("error_type") is None
                or application["translation_enabled"]
                or application["translation_runtime_ready"] is not True
                or owner is None
                or native_readiness is not None
            ):
                raise RuntimeError(
                    "26B rejection must disable effective translation, retain provider binding, "
                    "and preserve failed GPU/native-not-ready state without fallback"
                )
            self.journal.add(
                "no_fallback_assertion",
                failed_model=gemma.get("model_id"),
                failure_code=gemma.get("failure_code"),
                failure_phase=gemma.get("failure_phase"),
                error_type=gemma.get("error_type"),
                translation_enabled=False,
                provider_binding_available=True,
                native_readiness=None,
                fallback_processes=0,
            )
            self.checks["26b_no_fallback"] = True
            self.task(self.watchdog())
            await cli(self.config, self.journal, "translation", "set", "on", timeout=240)
            await self.activate_synthetic_peer()
            await self.smoke_workload()
            status = await self.control.query("models.status", {})
            application = await self.control.query("app.status", {})
            selection = self.control.settings.canonical.intent.translation
            capture = self.captures[-1]
            command = capture.command
            completions = [
                row for row in self.native_completions if row["purpose"] == "translation"
            ]
            if (
                selection.model != "managed_gemma_26b"
                or selection.connection != "gpu"
                or not application["translation_enabled"]
                or application["translation_runtime_ready"] is not True
                or status["gemma"]["state"] != "ready"
                or status["gemma"]["backend"] != "gpu"
                or status["gemma"]["model_id"] != gemma["model_id"]
                or owner.runtime.readiness is None
                or capture.returncode is not None
                or command[command.index("--device") + 1] == "none"
                or Path(command[command.index("--model") + 1]).name
                != "gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf"
                or len(completions) != 6
                or any(row["pid"] != capture.process.pid for row in completions)
            ):
                raise RuntimeError("explicit same-selection retry did not retain ready 26B Vulkan")
            self.journal.add(
                "explicit_retry_outcome",
                completed_translations=len(self.results),
                managed_processes=len(self.captures),
                models=status,
                selection_unchanged=True,
                translation_enabled=application["translation_enabled"],
                provider_binding_available=application["translation_runtime_ready"],
                native_ready=True,
                pid=capture.process.pid,
                native_translation_completions=len(completions),
            )
            return
        self.task(self.watchdog())
        await cli(self.config, self.journal, "translation", "set", "on", timeout=240)
        await self.activate_synthetic_peer()

    async def activate_synthetic_peer(self):
        from puripuly_heart.app.wiring.wiring_translation_runtime_configuration import (
            replace_translation_runtime_effective_flags,
        )

        self.control.settings.set_peer_translation_enabled(True)
        replace_translation_runtime_effective_flags(
            self.access.translation_runtime_configuration,
            peer_translation_enabled=True,
            integrated_context_enabled=True,
        )
        components = self.access.components_provider()
        components.output_runtime.activate_peer_generation(self.generation)
        config = self.access.translation_runtime_configuration.snapshot().value
        if not config.translation_enabled or not config.peer_translation_enabled:
            raise RuntimeError("effective translation toggles are not enabled")
        languages = self.control.settings.canonical.intent.languages
        for field in (
            "source_language",
            "target_language",
            "peer_source_language",
            "peer_target_language",
        ):
            if getattr(config, field) != getattr(languages, field):
                raise RuntimeError(
                    "effective language projection differs from committed synthetic selection"
                )
        state = await self.control.query("app.status", {})
        if any(
            item["desired_active"] or item["effective_active"]
            for item in state["channels"].values()
        ):
            raise RuntimeError("controlled translation ingress must not enable physical capture")
        self.journal.add(
            "controlled_capture_projection",
            physical_capture_enabled=False,
            consent_accepted=False,
            synthetic_peer_intent=True,
            effective_translation_config=config,
            owner_types={
                "self": type(self.access.self_vad).__name__,
                "peer": type(self.access.peer_vad).__name__,
            },
        )

    async def watchdog(self):
        while not self.finished.is_set():
            if any(monitor.cutoff.is_set() for monitor in self.monitors):
                self.checks["watchdog"] = "safety_cutoff_failure"
                self.finished.set()
                await self.host.close()
                raise RuntimeError("memory watchdog requested owned-only shutdown")
            await asyncio.sleep(0.1)

    async def consume_events(self):
        async for event in self.control.events.subscribe(
            topics=[], channel=None, include_transcripts=True, include_translations=True, after=0
        ):
            row = self.journal.add("app_event", event=event)
            self.events.append(row)

    async def seed_history(self):
        if not self.fixed_history:
            return
        await self.access.self_vad.translation_turns.wait_for_idle()
        configuration = self.access.translation_runtime_configuration.snapshot().value
        entries = []
        for channel, owner, source, target, text in (
            (
                "self",
                self.access.self_vad,
                configuration.source_language,
                configuration.target_language,
                "Mina is the courier. Nori is a small robot.",
            ),
            (
                "peer",
                self.access.peer_vad,
                configuration.peer_source_language,
                configuration.peer_target_language,
                "The delivery is planned for Tuesday. The room number is 12.",
            ),
        ):
            owner.runtime.clear_context()
            stamp = owner.clock.now()
            owner.runtime.remember_context(
                text, timestamp=stamp, source_language=source, target_language=target
            )
            entries.append(
                {
                    "channel": channel,
                    "text": text,
                    "source_language": source,
                    "target_language": target,
                    "order": len(entries),
                }
            )
        self.journal.add(
            "fixed_history_seed", entries=entries, boundary="settled_group_measurement_control"
        )

    async def request(self, channel, text, label, release=None):
        from puripuly_heart.core.stt.backend import STTRecognitionUnit
        from puripuly_heart.domain.recognition import (
            RecognitionStreamIdentity,
            RecognitionUnitIdentity,
        )

        if release is not None:
            await release.wait()
        started = time.perf_counter()
        self.units[channel] += 1
        unit_id = uuid4()
        identity = RecognitionUnitIdentity(
            RecognitionStreamIdentity(
                channel, self.generation, 1, "synthetic-251", ("fixed-input",)
            ),
            unit_id,
            self.units[channel],
        )
        request = self.journal.add(
            "request",
            label=label,
            phase=self.phase,
            channel=channel,
            unit_id=str(unit_id),
            text=text,
            started=started,
        )
        self.requests.append(request)
        owner = self.access.peer_vad if channel == "peer" else self.access.self_vad
        await owner.handle_recognition_unit(STTRecognitionUnit(identity=identity, text=text))
        new_ids = {
            key
            for key, bundle in owner.runtime.utterances.items()
            if key == unit_id
            or (
                bundle.final is not None
                and any(
                    origin.identity.unit_id == unit_id
                    for origin in bundle.final.recognition_origins
                )
            )
        }
        request["owner_utterance_ids"] = [str(item) for item in new_ids]
        self.journal.add(
            "request_admitted",
            unit_id=str(unit_id),
            channel=channel,
            owner_utterance_ids=[str(item) for item in new_ids],
            admitted=time.perf_counter(),
        )

        async def wait_result():
            while True:
                candidates = [
                    (key, bundle)
                    for key, bundle in owner.runtime.utterances.items()
                    if key == unit_id or key in new_ids
                ]
                for key, bundle in candidates:
                    translation = getattr(bundle, "translation", None)
                    if translation is not None:
                        return key, translation
                if self.finished.is_set():
                    raise RuntimeError("fixture stopping before request completion")
                await asyncio.sleep(0.005)

        async with asyncio.timeout(180):
            key, translation = await wait_result()
            await poll_until(
                lambda: any(
                    row["event"].get("topic") == "translation"
                    and row["event"].get("utterance_id") == str(key)
                    and row["event"].get("channel") == channel
                    for row in self.events
                ),
                timeout=180,
            )
        completed = time.perf_counter()
        result = self.journal.add(
            "request_complete",
            label=label,
            phase=self.phase,
            channel=channel,
            unit_id=str(unit_id),
            result_id=str(key),
            translation=translation,
            started=started,
            completed=completed,
            latency_ms=(completed - started) * 1000,
            quality={
                "nonempty": bool(translation.text.strip()),
                "reasoning_leak": "<think>" in translation.text.lower(),
            },
        )
        self.results.append(result)
        return result

    async def group(self, label, concurrent, index, pair_kind):
        self.phase = f"{pair_kind}:{label}"
        await self.seed_history()
        texts = (
            SYNTHETIC[index % 4],
            (KOREAN if pair_kind == "opposite" else SYNTHETIC)[(index + 1) % 4],
        )
        if concurrent:
            release = asyncio.Event()
            tasks = [
                self.task(self.request(channel, text, label, release))
                for channel, text in zip(("self", "peer"), texts)
            ]
            started = time.perf_counter()
            self.journal.add("pair_release", label=label, phase=self.phase, released=started)
            release.set()
            results = await asyncio.gather(*tasks)
        else:
            started = time.perf_counter()
            results = [
                await self.request(channel, text, label)
                for channel, text in zip(("self", "peer"), texts)
            ]
        self.journal.add(
            "pair_complete",
            label=label,
            phase=self.phase,
            concurrent=concurrent,
            makespan_ms=(time.perf_counter() - started) * 1000,
            results=[result["result_id"] for result in results],
        )
        self.prefix_snapshot()

    def prefix_snapshot(self):
        from puripuly_heart.core.local_translation.prefix_cache import (
            default_gemma_prefix_cache_dir,
        )

        directory = default_gemma_prefix_cache_dir()
        files = [
            {
                "name": str(path.relative_to(directory)).replace("\\", "/"),
                "size_bytes": path.stat().st_size,
            }
            for path in sorted(directory.rglob("*"))
            if path.is_file()
        ]
        self.journal.add("prefix_snapshot", phase=self.phase, files=files)

    async def smoke_workload(self):
        self.phase = "same:first-prefix"
        await self.seed_history()
        await self.request("self", SYNTHETIC[0], "smoke-self")
        await self.seed_history()
        await self.request("peer", SYNTHETIC[1], "smoke-peer")
        await self.group("smoke-overlap", True, 2, "same")
        await cli(
            self.config,
            self.journal,
            "settings",
            "apply",
            "--arguments",
            json.dumps({"changes": {"languages": {"peer_source": "ko", "peer_target": "en"}}}),
        )
        await self.activate_synthetic_peer()
        await self.group("smoke-overlap", True, 0, "opposite")
        await self.access.self_vad.translation_turns.wait_for_idle()
        await asyncio.sleep(3)
        output = self.access.components_provider().output_runtime
        self.journal.add(
            "routing_snapshot",
            decisions=output.routing_decisions,
            delivery_health=output.diagnostic_delivery_health,
        )
        self.checks["smoke_completed"] = len(self.results)
        if len(self.results) != 6 or len(self.captures) != 1:
            raise RuntimeError(
                "functional managed smoke did not complete six translations on one process"
            )

    async def workload(self):
        for pair_kind in ("same", "opposite"):
            await self.seed_history()
            if pair_kind == "opposite":
                await cli(
                    self.config,
                    self.journal,
                    "settings",
                    "apply",
                    "--arguments",
                    json.dumps(
                        {"changes": {"languages": {"peer_source": "ko", "peer_target": "en"}}}
                    ),
                )
                await self.activate_synthetic_peer()
            self.phase = f"{pair_kind}:first-prefix"
            for channel in ("self", "peer"):
                text = KOREAN[0] if channel == "peer" and pair_kind == "opposite" else SYNTHETIC[0]
                await self.seed_history()
                await self.request(channel, text, "first-prefix")
            for index in range(4):
                channel = "self" if index % 2 == 0 else "peer"
                self.phase = f"{pair_kind}:single"
                text = (
                    KOREAN[index]
                    if channel == "peer" and pair_kind == "opposite"
                    else SYNTHETIC[index]
                )
                await self.seed_history()
                await self.request(channel, text, f"single-{index}")
            for index in range(2):
                await self.group(f"serial-{index}", False, index, pair_kind)
            for index in range(6):
                await self.group(f"overlap-{index}", True, index, pair_kind)
        await self.access.self_vad.translation_turns.wait_for_idle()
        self.phase = "route-drain"
        await asyncio.sleep(3)
        output = self.access.components_provider().output_runtime
        routes = output.routing_decisions
        self.journal.add(
            "routing_snapshot", decisions=routes, delivery_health=output.diagnostic_delivery_health
        )
        self.checks.update(
            completed_translations=len(self.results),
            expected_translations=44,
            one_managed_process=len(self.captures) == 1,
            positive_self_chatbox=any(
                message["address"] == "/chatbox/input"
                and message["parameters"]
                and str(message["parameters"][0]).strip()
                for message in self.sink.messages
            ),
            peer_chatbox_denied=any(item.reason == "peer_chatbox_denied" for item in routes),
        )
        if not self.checks["positive_self_chatbox"] or not self.checks["peer_chatbox_denied"]:
            raise RuntimeError("positive Self chatbox or denied Peer route evidence is missing")
        command = self.captures[-1].command
        cpu = command[command.index("--device") + 1] == "none"
        active_max = max(
            sum(bool(slot.get("is_processing")) for slot in row["slots"])
            for row in self.observed_slots
        )
        if active_max > (1 if cpu else 2):
            raise RuntimeError("effective backend exceeded active native slot admission")
        if cpu and not any(
            (row["payload"].get("timings") or {}).get("draft_n", 0) > 0
            for row in self.native_completions
            if row.get("purpose") == "translation"
        ):
            raise RuntimeError("CPU MTP was configured but no native draft tokens were observed")
        if len(self.results) != 44 or len(self.captures) != 1:
            raise RuntimeError("managed lifecycle completion invariant failed")

    def active_slot_count(self):
        if not self.observed_slots or not self.captures:
            return 0
        row = self.observed_slots[-1]
        if (
            row["phase"] != self.phase
            or row["pid"] != self.captures[-1].process.pid
            or time.perf_counter() - row["t"] > 0.2
        ):
            return 0
        return sum(
            bool(slot.get("is_processing")) and slot.get("params", {}).get("n_predict") != 1
            for slot in row["slots"]
        )

    async def safety_workload(self):
        self.fixed_history = False
        self.phase = "natural-history"
        self.access.self_vad.runtime.clear_context()
        self.access.peer_vad.runtime.clear_context()
        await self.request("self", SYNTHETIC[0], "natural-self")
        await self.request("peer", KOREAN[1], "natural-peer")
        safety_pid = self.captures[-1].process.pid
        for ordered_channel in ("peer", "self"):
            self.phase = f"ordered-{ordered_channel}"
            inputs = KOREAN if ordered_channel == "peer" else SYNTHETIC
            label = f"ordered-{ordered_channel}-first"
            first = self.task(self.request(ordered_channel, " ".join(inputs), label))
            await poll_until(
                lambda: any(
                    row.get("owner_utterance_ids") and row["label"] == label
                    for row in self.requests
                )
            )
            other_channel = "self" if ordered_channel == "peer" else "peer"
            other = self.task(
                self.request(
                    other_channel,
                    "Good morning." if other_channel == "self" else "안녕하세요.",
                    f"ordered-{ordered_channel}-other",
                )
            )
            second = self.task(
                self.request(
                    ordered_channel,
                    "고마워요." if ordered_channel == "peer" else "Thank you.",
                    f"ordered-{ordered_channel}-second",
                )
            )
            first_result, other_result, second_result = await asyncio.gather(first, other, second)

            def sequence(result):
                return next(
                    row["event"]["sequence"]
                    for row in self.events
                    if row["event"].get("topic") == "translation"
                    and row["event"].get("utterance_id") == result["result_id"]
                )

            if sequence(first_result) >= sequence(second_result):
                raise RuntimeError("same-channel translation publication order changed")
            if other_result["completed"] >= first_result["completed"]:
                raise RuntimeError("other channel did not progress during ordered long request")
            self.journal.add(
                "ordering_outcome",
                channel=ordered_channel,
                first_result_id=first_result["result_id"],
                second_result_id=second_result["result_id"],
                first_sequence=sequence(first_result),
                second_sequence=sequence(second_result),
                other_completed_before_first=True,
            )
        for cancelled in ("peer", "self"):
            self.phase = f"cancel-{cancelled}"
            release = asyncio.Event()
            pair = {
                channel: self.task(
                    self.request(
                        channel,
                        (SYNTHETIC if channel == "self" else KOREAN)[2],
                        f"cancel-{cancelled}",
                        release,
                    )
                )
                for channel in ("self", "peer")
            }
            release.set()
            await poll_until(lambda: self.active_slot_count() == 2, timeout=30)
            cancelled_request = next(
                row
                for row in reversed(self.requests)
                if row["channel"] == cancelled and row["label"] == f"cancel-{cancelled}"
            )
            cancelled_ids = set(cancelled_request.get("owner_utterance_ids", [])) | {
                cancelled_request["unit_id"]
            }
            self.journal.add(
                "cancellation_requested",
                channel=cancelled,
                request_ids=sorted(cancelled_ids),
                active_slots=self.active_slot_count(),
                pid=safety_pid,
            )
            await self.access.self_vad.translation_turns.cancel_pending(channel=cancelled)
            pair[cancelled].cancel()
            await asyncio.gather(pair[cancelled], return_exceptions=True)
            survivor = await pair["self" if cancelled == "peer" else "peer"]
            await self.access.self_vad.translation_turns.wait_for_idle()
            await asyncio.sleep(0.05)
            if any(
                row["event"].get("topic") == "translation"
                and row["event"].get("utterance_id") in cancelled_ids
                for row in self.events
            ):
                raise RuntimeError("cancelled request published a late translation")
            if len(self.captures) != 1 or self.captures[-1].process.pid != safety_pid:
                raise RuntimeError("cancellation replaced the shared managed process")
            self.journal.add(
                "cancellation_outcome",
                cancelled_channel=cancelled,
                survivor=survivor,
                same_process_pid=self.captures[-1].process.pid,
                active_slots_after=self.active_slot_count(),
                stale_publications=0,
            )
            await self.request(
                cancelled, SYNTHETIC[1] if cancelled == "self" else KOREAN[1], "cancel-slot-reuse"
            )
        self.phase = "response-failure"
        self.response_fault_channel = "self"
        release = asyncio.Event()
        failed = self.task(self.request("self", "Thank you.", "response-fault", release))
        survivor = self.task(
            self.request("peer", " ".join(KOREAN), "response-fault-survivor", release)
        )
        release.set()
        await poll_until(lambda: self.response_fault_used)
        survivor_result = await survivor
        await self.access.self_vad.translation_turns.wait_for_idle()
        failed.cancel()
        await asyncio.gather(failed, return_exceptions=True)
        self.response_fault_channel = None
        rejected = next(row for row in reversed(self.requests) if row["label"] == "response-fault")
        rejected_ids = set(rejected.get("owner_utterance_ids", [])) | {rejected["unit_id"]}
        if any(
            row["event"].get("topic") == "translation"
            and row["event"].get("utterance_id") in rejected_ids
            for row in self.events
        ):
            raise RuntimeError("rejected native response was published")
        if len(self.captures) != 1 or self.captures[-1].process.pid != safety_pid:
            raise RuntimeError("request-local response failure replaced the shared server")
        self.journal.add(
            "response_failure_outcome",
            survivor=survivor_result,
            active_slots_after=self.active_slot_count(),
            stale_publications=0,
        )
        await self.request("self", SYNTHETIC[1], "failure-slot-reuse")
        self.phase = "manual-capacity"
        release = asyncio.Event()
        pair = [
            self.task(
                self.request(
                    channel,
                    (SYNTHETIC if channel == "self" else KOREAN)[3],
                    "manual-capacity",
                    release,
                )
            )
            for channel in ("self", "peer")
        ]
        release.set()
        await poll_until(lambda: self.active_slot_count() == 2, timeout=30)
        manual = self.task(
            cli(self.config, self.journal, "text", "submit", "--stdin", stdin=SYNTHETIC[2])
        )
        await asyncio.gather(*pair, manual)
        await self.access.self_vad.translation_turns.wait_for_idle()
        self.journal.add(
            "manual_capacity_outcome",
            observed_max_active=max(
                sum(bool(slot.get("is_processing")) for slot in row["slots"])
                for row in self.observed_slots
            ),
        )
        if any(
            sum(bool(slot.get("is_processing")) for slot in row["slots"]) > 2
            for row in self.observed_slots
        ):
            raise RuntimeError("manual admission exceeded managed GPU capacity")
        for source, target in (("en", "ja"), ("en", "fr"), ("en", "de"), ("en", "ko")):
            self.phase = f"prefix-lru-{source}-{target}"
            await cli(
                self.config,
                self.journal,
                "settings",
                "apply",
                "--arguments",
                json.dumps({"changes": {"languages": {"source": source, "target": target}}}),
            )
            await self.activate_synthetic_peer()
            await self.request("self", SYNTHETIC[0], "prefix-lru")
        self.phase = "active-backend-transition"
        pending = self.task(self.request("self", SYNTHETIC[3], "retired-backend"))
        await poll_until(lambda: self.active_slot_count() > 0, timeout=30)
        changes = {"translation.connection": "cpu"}
        if self.args.model == "managed_gemma_26b":
            changes["translation.model"] = "managed_gemma"
            from puripuly_heart.core.local_translation.assets import e4b_gemma_spec

            stage_model(self.args, e4b_gemma_spec())
        await cli(
            self.config,
            self.journal,
            "settings",
            "apply",
            "--arguments",
            json.dumps({"changes": changes}),
            timeout=240,
        )
        pending.cancel()
        await asyncio.gather(pending, return_exceptions=True)
        await self.activate_synthetic_peer()
        await self.request("self", "Hello.", "cpu-after-transition")
        await self.request("peer", "안녕하세요.", "peer-cpu-after-transition")
        self.phase = "cpu-to-gpu-transition"
        await cli(
            self.config,
            self.journal,
            "settings",
            "apply",
            "--arguments",
            json.dumps({"changes": {"translation.connection": "gpu"}}),
            timeout=240,
        )
        recovery_required = (
            self.captures[-1].command[self.captures[-1].command.index("--device") + 1] == "none"
        )
        if recovery_required:
            previous_cpu_pid = self.captures[-1].process.pid
            self.journal.add(
                "supported_backend_recovery_requested",
                reason="committed_gpu_intent_but_observed_native_cpu",
                previous_cpu_pid=previous_cpu_pid,
                models=await self.control.query("models.status", {}),
            )
            await cli(self.config, self.journal, "translation", "set", "off")
            await cli(self.config, self.journal, "translation", "set", "on", timeout=240)
            command = self.captures[-1].command
            if (
                self.captures[-1].process.pid == previous_cpu_pid
                or command[command.index("--device") + 1] == "none"
            ):
                raise RuntimeError("supported translation toggle did not activate a new GPU child")
            self.journal.add(
                "supported_backend_recovery_prepared",
                current_gpu_pid=self.captures[-1].process.pid,
                models=await self.control.query("models.status", {}),
            )
        await self.activate_synthetic_peer()
        await self.group("after-return-to-gpu", True, 1, "opposite")
        await self.group("after-return-to-gpu-repeat", True, 2, "opposite")
        capture = self.captures[-1]
        phases = {"opposite:after-return-to-gpu", "opposite:after-return-to-gpu-repeat"}
        completions = [
            row
            for row in self.native_completions
            if row["purpose"] == "translation" and row["phase"] in phases
        ]
        status = await self.control.query("models.status", {})
        application = await self.control.query("app.status", {})
        if (
            capture.returncode is not None
            or capture.command[capture.command.index("--device") + 1] == "none"
            or len(completions) != 4
            or any(row["pid"] != capture.process.pid for row in completions)
            or {row["identity"]["channel"] for row in completions} != {"self", "peer"}
            or status["gemma"]["state"] != "ready"
            or status["gemma"]["backend"] != "gpu"
            or status["gemma"]["model_id"] != "gemma-4-e4b-it-qat-ud-q4-k-xl"
            or not application["translation_enabled"]
            or self.control.gemma().runtime.readiness is None
        ):
            raise RuntimeError(
                "repeated GPU-return provider requests did not retain ready E4B Vulkan"
            )
        self.journal.add(
            "backend_return_outcome",
            pid=capture.process.pid,
            recovery_required=recovery_required,
            models=status,
            native_ready=True,
            translation_enabled=application["translation_enabled"],
            requests=[
                {
                    "identity": row["identity"],
                    "phase": row["phase"],
                    "pid": row["pid"],
                    "slot_id": row["slot_id"],
                }
                for row in completions
            ],
        )
        self.phase = "active-shutdown"
        self.task(self.request("self", " ".join(SYNTHETIC), "shutdown-self"))
        self.task(self.request("peer", " ".join(KOREAN), "shutdown-peer"))
        await poll_until(lambda: self.active_slot_count() == 2, timeout=30)
        admission = await self.access.self_vad.submit_text(
            "The synthetic shutdown queue has one manual turn."
        )
        self.journal.add(
            "shutdown_waiting_manual_admission",
            admission=admission,
            active_native_slots=self.active_slot_count(),
            pid=self.captures[-1].process.pid,
        )

    async def close(self):
        self.phase = "shutdown"
        self.finished.set()
        if self.host is not None:
            if self.access is not None:
                output = self.access.components_provider().output_runtime
                self.journal.add(
                    "routing_snapshot",
                    decisions=output.routing_decisions,
                    delivery_health=output.diagnostic_delivery_health,
                )
            await self.host.close()
        await asyncio.sleep(2)
        for monitor in self.monitors:
            monitor.close()
        for task in tuple(self.tasks):
            task.cancel()
        await asyncio.gather(*tuple(self.tasks), return_exceptions=True)
        self.sink.close()
        port_states = []
        for capture in self.captures:
            port = int(capture.command[capture.command.index("--port") + 1])
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as handle:
                handle.settimeout(1)
                accepting = handle.connect_ex(("127.0.0.1", port)) == 0
            port_states.append(
                {"pid": capture.process.pid, "port": port, "accepting_after_shutdown": accepting}
            )
        self.journal.add("shutdown_ports", states=port_states)
        self.journal.add(
            "cleanup",
            managed_processes=[
                {"pid": item.process.pid, "returncode": item.returncode} for item in self.captures
            ],
            remaining_owned_tasks=len(self.tasks),
            checks=self.checks,
        )


async def lifecycle(args, journal):
    fixture = Fixture(args, journal)
    failure = None
    try:
        await fixture.start()
        if args.case in {"smoke", "smoke26b"}:
            await fixture.smoke_workload()
        elif args.case in {"safety", "26b"}:
            await fixture.workload()
            await fixture.safety_workload()
        elif args.case not in {"26b-failure", "preflight"}:
            await fixture.workload()
    except BaseException as error:
        failure = error
        journal.add(
            "failure",
            stage=fixture.phase,
            error=type(error).__name__,
            detail=str(error),
            traceback=traceback.format_exc(),
        )
    finally:
        try:
            await fixture.close()
        except BaseException as error:
            journal.add(
                "cleanup_failure",
                error=type(error).__name__,
                detail=str(error),
                traceback=traceback.format_exc(),
            )
            failure = failure or error
    return 1 if failure is not None else 0


async def supported_manual(config, journal, text):
    follower = await asyncio.create_subprocess_exec(
        sys.executable,
        "-u",
        "-m",
        "puripuly_heart.main",
        "cli",
        "--config",
        str(config),
        "events",
        "follow",
        "--topic",
        "translation",
        "--include-translations",
        "--after",
        "0",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        await asyncio.sleep(0.5)
        receipt = await cli(config, journal, "text", "submit", "--stdin", stdin=text)
        async with asyncio.timeout(180):
            while True:
                line = await follower.stdout.readline()
                if not line:
                    raise RuntimeError("supported manual event stream ended before completion")
                event = json.loads(line)
                journal.add("supported_manual_event", event=event)
                if event.get("topic") == "translation":
                    if event.get("channel") != "self" or not event.get("text", "").strip():
                        raise RuntimeError("supported manual translation association is invalid")
                    allowed = set(receipt.get("child_turn_ids", [])) | {receipt.get("turn_id")}
                    if event.get("utterance_id") not in allowed:
                        raise RuntimeError("supported manual turn association differs from receipt")
                    return event
    finally:
        follower.terminate()
        await follower.communicate()


class StockObservation:
    def __init__(self, config, journal):
        self.config, self.journal = config, journal
        self.hosts, self.children, self.monitors = {}, {}, {}
        self.finished = asyncio.Event()
        self.task = asyncio.create_task(self.observe())

    async def register_host(self):
        import psutil

        state = await cli(self.config, self.journal, "app", "status")
        process = psutil.Process(state["host_pid"])
        self.hosts[process.pid] = process.create_time()
        self.journal.add(
            "stock_host_identity",
            pid=process.pid,
            create_time=process.create_time(),
            instance_id=state["instance_id"],
            executable=process.exe(),
            command=process.cmdline(),
            cwd=process.cwd(),
            source_import_root=os.environ.get("PYTHONPATH"),
        )

    async def observe(self):
        import psutil

        while not self.finished.is_set():
            for host_pid, created in tuple(self.hosts.items()):
                try:
                    host = psutil.Process(host_pid)
                    if host.create_time() != created:
                        continue
                    for process in host.children(recursive=True):
                        command = process.cmdline()
                        if process.name().lower() != "llama-server.exe" or "--model" not in command:
                            continue
                        key = (process.pid, process.create_time())
                        if key not in self.children:
                            self.children[key] = command
                            self.monitors[key] = Monitor(
                                process.pid, self.journal, lambda: "supported-cli"
                            )
                            self.journal.add(
                                "stock_managed_identity",
                                pid=process.pid,
                                create_time=key[1],
                                host_pid=host_pid,
                                command=command,
                                executable_sha256=sha(Path(process.exe())),
                            )
                except psutil.NoSuchProcess, psutil.AccessDenied:
                    continue
            for key, monitor in tuple(self.monitors.items()):
                if monitor.cutoff.is_set():
                    try:
                        process = psutil.Process(key[0])
                        if process.create_time() == key[1]:
                            process.terminate()
                            self.journal.add(
                                "owned_watchdog_terminate", pid=key[0], create_time=key[1]
                            )
                    except psutil.NoSuchProcess:
                        pass
                    self.finished.set()
                    return
            await asyncio.sleep(0.1)

    async def close(self):
        import psutil

        self.finished.set()
        await self.task
        for monitor in self.monitors.values():
            monitor.close()
        children = []
        for (pid, created), command in self.children.items():
            try:
                present = psutil.Process(pid).create_time() == created
            except psutil.NoSuchProcess:
                present = False
            port = int(command[command.index("--port") + 1])
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as handle:
                handle.settimeout(1)
                accepting = handle.connect_ex(("127.0.0.1", port)) == 0
            children.append(
                {
                    "pid": pid,
                    "create_time": created,
                    "process_present": present,
                    "accepting_after_stop": accepting,
                }
            )
        self.journal.add("stock_cleanup", children=children)
        if any(item["process_present"] or item["accepting_after_stop"] for item in children):
            raise RuntimeError("supported CLI stop left an observed owned native process or port")


async def supported(args, journal):
    journal.add("stock_candidate", candidate=identify(args.source))
    sink = OscSink(journal)
    config = seed(args, sink.port)
    started = False
    observation = StockObservation(config, journal)
    try:
        await cli(config, journal, "app", "start", "--background")
        started = True
        await observation.register_host()
        for command in (
            ("capabilities",),
            ("settings", "choices"),
            ("settings", "current"),
            ("app", "status"),
            ("capture", "status"),
            ("osc", "status"),
            ("models", "status"),
        ):
            await cli(config, journal, *command)
        await cli(config, journal, "translation", "set", "on", timeout=240)
        await supported_manual(config, journal, SYNTHETIC[0])
        from puripuly_heart.core.local_translation.assets import gemma_26b_spec

        stage_model(args, gemma_26b_spec())
        await cli(
            config,
            journal,
            "settings",
            "apply",
            "--arguments",
            json.dumps(
                {
                    "changes": {
                        "translation.model": "managed_gemma_26b",
                        "translation.connection": "gpu",
                    }
                }
            ),
        )
        await cli(config, journal, "app", "stop")
        started = False
        await cli(config, journal, "app", "start", "--background")
        started = True
        await observation.register_host()
        current = await cli(config, journal, "settings", "current")
        if current["settings"]["intent"]["translation"]["model"] != "managed_gemma_26b":
            raise RuntimeError(
                "explicit 26B model selection did not persist across fresh host restart"
            )
        await cli(config, journal, "app", "status")
        await cli(config, journal, "translation", "set", "on", timeout=240)
        await supported_manual(config, journal, SYNTHETIC[1])
        return 0
    finally:
        try:
            if started:
                await cli(config, journal, "app", "stop")
        finally:
            try:
                await observation.close()
            finally:
                sink.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "lifecycle", "supported", "summarize"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path)
    parser.add_argument("--e4b-source", type=Path)
    parser.add_argument("--model26-source", type=Path)
    parser.add_argument("--model", default="managed_gemma")
    parser.add_argument("--backend", default="gpu")
    parser.add_argument("--mode", default="product")
    parser.add_argument("--case", default="abba")
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    if args.action == "prepare":
        prepare(args)
        return 0
    if args.action == "summarize":
        from summarize import summarize

        summarize(args.root)
        return 0
    journal = Journal(args.root / "events.jsonl")
    try:
        if args.action == "lifecycle":
            return asyncio.run(lifecycle(args, journal))
        if args.action == "supported":
            return asyncio.run(supported(args, journal))
    except BaseException as error:
        journal.add(
            "fatal", error=type(error).__name__, detail=str(error), traceback=traceback.format_exc()
        )
        return 1
    finally:
        journal.close()


if __name__ == "__main__":
    raise SystemExit(main())
