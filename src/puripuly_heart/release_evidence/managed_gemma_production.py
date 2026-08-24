from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import re
import subprocess
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from uuid import uuid4

from puripuly_heart.app.adapters.settings_vnext_canonical_persistence import (
    SettingsVNextCanonicalPersistenceAdapter,
)
from puripuly_heart.app.services.managed_gemma_translation import (
    ManagedGemmaTranslationOwner,
)
from puripuly_heart.app.wiring.wiring_managed_gemma import managed_gemma_selection
from puripuly_heart.app.wiring.wiring_translation_backend import (
    create_translation_backend,
)
from puripuly_heart.config.settings_vnext.schema import (
    AppSettingsVNext,
    TranslationFallbackIntent,
)
from puripuly_heart.core.http_extensions import HttpExtensionRegistry
from puripuly_heart.core.lifecycle import LifecycleScope, start_lifecycle_task
from puripuly_heart.core.llm.provider import SemaphoreLLMProvider
from puripuly_heart.core.local_translation.assets import (
    GEMMA_ASSETS,
    GEMMA_REPO_ID,
    GEMMA_REVISION,
    validate_gemma_install,
)
from puripuly_heart.core.local_translation.runtime import ManagedGemmaRuntimeOwner
from puripuly_heart.core.local_translation.runtime_profile import default_gemma_runtime_paths
from puripuly_heart.core.storage.secrets import InMemorySecretStore
from puripuly_heart.core.translation_backend import (
    LlmTranslationBackend,
    TranslationBackendRequest,
)
from puripuly_heart.providers.llm.managed_gemma import (
    HttpxManagedGemmaTransport,
    ManagedGemmaLLMProvider,
)
from puripuly_heart.release_evidence.managed_gemma_distribution import (
    validate_runtime_root,
)

SCHEMA = "puripuly-heart/managed-gemma-production-evidence/v1"
PERFORMANCE_PREFIX = "[ManagedGemma][Performance] "
SHORT_TRANSLATION_INPUTS = ("Hello.", "Thank you.", "See you soon.")
METRIC_KEYS = {
    "prompt_tokens",
    "cached_prompt_tokens",
    "completion_tokens",
    "prompt_ms",
    "generation_ms",
    "generation_tps",
    "drafted_tokens",
    "accepted_tokens",
}


@dataclass(slots=True)
class _CapturedRuntimeProcess:
    process: asyncio.subprocess.Process
    command: tuple[str, ...]
    output_chunks: list[bytes]
    scope: LifecycleScope = field(repr=False)
    reader_task: asyncio.Task[None] = field(repr=False)

    @property
    def returncode(self) -> int | None:
        return self.process.returncode

    @property
    def output(self) -> str:
        return b"".join(self.output_chunks).decode("utf-8", errors="replace")

    def terminate(self) -> None:
        self.process.terminate()

    def kill(self) -> None:
        self.process.kill()

    async def wait(self) -> int:
        returncode = await self.process.wait()
        await self.reader_task
        return returncode


def _captured_process_factory(
    captures: list[_CapturedRuntimeProcess],
):
    async def factory(command: tuple[str, ...]) -> _CapturedRuntimeProcess:
        observed_command = command + ("--verbosity", "4")
        process = await asyncio.create_subprocess_exec(
            observed_command[0],
            *observed_command[1:],
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        stdout = process.stdout
        if stdout is None:
            process.kill()
            await process.wait()
            raise RuntimeError("managed Gemma evidence could not capture runtime output")
        output_chunks: list[bytes] = []

        async def read_output() -> None:
            while chunk := await stdout.read(65536):
                output_chunks.append(chunk)

        scope = LifecycleScope("managed-gemma-production-runtime-output")
        capture = _CapturedRuntimeProcess(
            process=process,
            command=observed_command,
            output_chunks=output_chunks,
            scope=scope,
            reader_task=start_lifecycle_task(scope, read_output(), name="stdout-drain"),
        )
        captures.append(capture)
        return capture

    return factory


def _parse_performance_log(message: str) -> dict[str, object]:
    if not message.startswith(PERFORMANCE_PREFIX):
        raise RuntimeError("managed Gemma performance log prefix is missing")
    fields = dict(re.findall(r"([a-z_]+)=([^\s]+)", message[len(PERFORMANCE_PREFIX) :]))
    required = {
        "backend",
        "language_pair",
        "prompt_tokens",
        "completion_tokens",
        "prompt_ms",
        "generation_ms",
        "generation_tps",
    }
    if not required.issubset(fields):
        raise RuntimeError("managed Gemma performance log is incomplete")
    parsed: dict[str, object] = {
        "backend": fields["backend"],
        "language_pair": fields["language_pair"],
    }
    for key in METRIC_KEYS:
        value = fields.get(key)
        if value is None:
            continue
        parsed[key] = float(value) if key.endswith("_ms") or key.endswith("_tps") else int(value)
    return parsed


def _validate_metrics(metrics: dict[str, object], *, effective_backend: str) -> None:
    if metrics.get("backend") != effective_backend:
        raise RuntimeError("managed Gemma performance backend identity mismatch")
    if int(metrics.get("prompt_tokens", 0)) <= 0:
        raise RuntimeError("managed Gemma prompt token count is missing")
    if int(metrics.get("completion_tokens", 0)) <= 0:
        raise RuntimeError("managed Gemma completion token count is missing")
    cached_prompt_tokens = metrics.get("cached_prompt_tokens")
    if not isinstance(cached_prompt_tokens, int) or cached_prompt_tokens <= 0:
        raise RuntimeError("managed Gemma cached prompt token count is missing")
    generation_ms = float(metrics.get("generation_ms", 0.0))
    if not math.isfinite(generation_ms) or generation_ms <= 0:
        raise RuntimeError("managed Gemma generation timing is missing")
    generation_tps = float(metrics.get("generation_tps", 0.0))
    if not math.isfinite(generation_tps) or generation_tps <= 0:
        raise RuntimeError("managed Gemma generation speed is missing")
    prompt_ms = float(metrics.get("prompt_ms", 0.0))
    if not math.isfinite(prompt_ms) or prompt_ms <= 0:
        raise RuntimeError("managed Gemma prompt timing is missing")


def _command_value(command: tuple[str, ...], option: str) -> str:
    try:
        index = command.index(option)
        return command[index + 1]
    except (ValueError, IndexError) as exc:
        raise RuntimeError(f"managed Gemma evidence command is missing {option}") from exc


def _runtime_observation(
    capture: _CapturedRuntimeProcess,
    *,
    backend: str,
) -> dict[str, object]:
    if capture.returncode is None:
        raise RuntimeError("managed Gemma evidence runtime remained alive after close")
    device = _command_value(capture.command, "--device")
    n_gpu_layers = int(_command_value(capture.command, "--n-gpu-layers"))
    if backend == "cpu":
        if device != "none" or n_gpu_layers != 0 or "--spec-draft-model" not in capture.command:
            raise RuntimeError("managed Gemma CPU evidence command profile mismatch")
        return {
            "device": device,
            "n_gpu_layers": n_gpu_layers,
            "mtp_enabled": True,
            "process_exited": True,
        }
    if device == "none" or n_gpu_layers != 99 or "--spec-draft-model" in capture.command:
        raise RuntimeError("managed Gemma Vulkan evidence command profile mismatch")
    offload = re.search(r"offloaded\s+(\d+)/(\d+)\s+layers to GPU", capture.output)
    selected = re.search(
        rf"using device\s+{re.escape(device)}\s+\(([^)]+)\)",
        capture.output,
    )
    if offload is None or selected is None:
        raise RuntimeError("managed Gemma Vulkan device telemetry is missing")
    offloaded_layers = int(offload.group(1))
    total_layers = int(offload.group(2))
    if offloaded_layers <= 0 or offloaded_layers != total_layers:
        raise RuntimeError("managed Gemma Vulkan model was not fully offloaded")
    return {
        "device": device,
        "device_name": selected.group(1),
        "n_gpu_layers": n_gpu_layers,
        "offloaded_layers": offloaded_layers,
        "total_layers": total_layers,
        "mtp_enabled": False,
        "process_exited": True,
    }


async def _close_backend_and_owner(
    translation_backend: LlmTranslationBackend | None,
    owner: ManagedGemmaTranslationOwner,
    captures: tuple[_CapturedRuntimeProcess, ...] = (),
) -> None:
    failures: list[BaseException] = []
    if translation_backend is not None:
        try:
            await translation_backend.close()
        except BaseException as exc:
            failures.append(exc)
    try:
        await owner.close()
    except BaseException as exc:
        failures.append(exc)
    for capture in captures:
        try:
            await capture.scope.close()
        except BaseException as exc:
            failures.append(exc)
    if len(failures) == 1:
        raise failures[0]
    if failures:
        raise BaseExceptionGroup("managed Gemma evidence cleanup failed", failures)


def _settings(backend: str) -> object:
    canonical = AppSettingsVNext()
    canonical = replace(
        canonical,
        intent=replace(
            canonical.intent,
            translation=replace(
                canonical.intent.translation,
                model="managed_gemma",
                connection=backend,
                fallback=TranslationFallbackIntent(selection_alias="deepseek_v4_flash_official"),
            ),
            languages=replace(
                canonical.intent.languages,
                source_language="en",
                target_language="ko",
            ),
        ),
    )
    return SettingsVNextCanonicalPersistenceAdapter().compatibility_projection(canonical)


def _is_managed_only_backend(backend: object) -> bool:
    if not isinstance(backend, LlmTranslationBackend):
        return False
    provider = backend.provider
    while isinstance(provider, SemaphoreLLMProvider):
        provider = provider.inner
    return isinstance(provider, ManagedGemmaLLMProvider)


async def _run_backend(
    *,
    backend: str,
    model_dir: Path,
    runtime_root: Path,
    iterations: int,
) -> dict[str, object]:
    settings = _settings(backend)
    performance_logs: list[str] = []
    statuses: list[str] = []
    captures: list[_CapturedRuntimeProcess] = []
    runtime = ManagedGemmaRuntimeOwner(
        install_dir=model_dir,
        runtime_paths=default_gemma_runtime_paths(runtime_root),
        process_factory=_captured_process_factory(captures),
        transport_factory=lambda base_url: HttpxManagedGemmaTransport(base_url),
        log_sink=lambda message, _level: (
            performance_logs.append(message) if message.startswith(PERFORMANCE_PREFIX) else None
        ),
    )
    owner = ManagedGemmaTranslationOwner(
        runtime=runtime,
        status_sink=lambda snapshot: statuses.append(snapshot.state),
    )
    translation_backend: LlmTranslationBackend | None = None
    report: dict[str, object] | None = None
    try:
        selection = managed_gemma_selection(settings)
        activation = await owner.prepare(selection)
        expected_effective = "cpu" if backend == "cpu" else "vulkan"
        if activation.readiness.requested_backend != backend:
            raise RuntimeError("managed Gemma requested backend identity mismatch")
        if activation.readiness.effective_backend != expected_effective:
            raise RuntimeError(
                f"managed Gemma {backend} evidence used "
                f"{activation.readiness.effective_backend} instead of {expected_effective}"
            )
        translation_backend = create_translation_backend(
            settings,
            secrets=InMemorySecretStore(),
            http_extensions=HttpExtensionRegistry(model_dir.parent / "unused-http-extensions"),
            managed_gemma_runtime=activation.runtime,
            managed_gemma_release=activation.release,
        )
        if not _is_managed_only_backend(translation_backend):
            raise RuntimeError("managed Gemma provider fallback remained effective")
        translations: list[dict[str, object]] = []
        for index in range(iterations):
            text = SHORT_TRANSLATION_INPUTS[index % len(SHORT_TRANSLATION_INPUTS)]
            before_logs = len(performance_logs)
            started = time.perf_counter()
            result = await translation_backend.translate(
                TranslationBackendRequest(
                    utterance_id=uuid4(),
                    text=text,
                    system_prompt=selection.system_prompt,
                    source_language=selection.source_language,
                    target_language=selection.target_language,
                )
            )
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            if not result.text.strip() or result.source_text != text:
                raise RuntimeError("managed Gemma returned an invalid translation")
            if len(performance_logs) != before_logs + 1:
                raise RuntimeError("managed Gemma emitted an unexpected performance log count")
            metrics = _parse_performance_log(performance_logs[-1])
            _validate_metrics(metrics, effective_backend=expected_effective)
            translations.append(
                {
                    "sequence": index + 1,
                    "output_chars": len(result.text),
                    "elapsed_ms": round(elapsed_ms, 3),
                    "metrics": metrics,
                }
            )
        report = {
            "requested_backend": backend,
            "effective_backend": activation.readiness.effective_backend,
            "language_pair": f"{selection.source_language}->{selection.target_language}",
            "prefix_identity": activation.readiness.prefix_identity,
            "provider_fallback_effective": False,
            "translations": translations,
        }
    finally:
        await _close_backend_and_owner(translation_backend, owner, tuple(captures))
    if report is None:
        raise RuntimeError("managed Gemma evidence report was not produced")
    if statuses[-1:] != ["closed"]:
        raise RuntimeError("managed Gemma evidence owner did not reach closed state")
    if len(captures) != 1:
        raise RuntimeError("managed Gemma evidence started an unexpected runtime process count")
    report["status_sequence"] = list(statuses)
    report["runtime_observation"] = _runtime_observation(captures[0], backend=backend)
    return report


async def run_production_evidence(
    *,
    candidate: str,
    model_dir: Path,
    runtime_root: Path,
    iterations: int,
) -> dict[str, object]:
    if not re.fullmatch(r"[0-9a-f]{40}", candidate):
        raise ValueError("candidate must be a full lowercase Git SHA")
    if iterations < 2:
        raise ValueError("production evidence requires at least two translations per backend")
    validate_gemma_install(model_dir)
    runtime_manifest = validate_runtime_root(runtime_root)
    reports = []
    for backend in ("cpu", "gpu"):
        reports.append(
            await _run_backend(
                backend=backend,
                model_dir=model_dir,
                runtime_root=runtime_root,
                iterations=iterations,
            )
        )
    return {
        "schema": SCHEMA,
        "candidate": candidate,
        "model": {
            "repo_id": GEMMA_REPO_ID,
            "revision": GEMMA_REVISION,
            "files": [
                {
                    "filename": asset.filename,
                    "size": asset.size_bytes,
                    "sha256": asset.sha256,
                }
                for asset in GEMMA_ASSETS
            ],
        },
        "runtime": {
            "build": runtime_manifest["release"]["build"],
            "commit": runtime_manifest["release"]["commit"],
            "manifest_schema": runtime_manifest["schema"],
            "manifest_sha256": hashlib.sha256(
                (runtime_root / "manifest.json").read_bytes()
            ).hexdigest(),
            "tree_verified": True,
        },
        "iterations_per_backend": iterations,
        "backends": reports,
        "privacy": {
            "source_text_recorded": False,
            "translated_text_recorded": False,
            "system_prompt_recorded": False,
        },
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=3)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    report = asyncio.run(
        run_production_evidence(
            candidate=args.candidate,
            model_dir=args.model_dir.resolve(),
            runtime_root=args.runtime_root.resolve(),
            iterations=args.iterations,
        )
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.report.with_suffix(f"{args.report.suffix}.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
