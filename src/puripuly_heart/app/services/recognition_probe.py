from __future__ import annotations

import asyncio
import wave
from collections.abc import Callable
from dataclasses import replace
from typing import Literal

import numpy as np

from puripuly_heart.core.audio.format import float32_to_pcm16le_bytes, resample_f32_linear
from puripuly_heart.core.diagnostics import exception_diagnostics
from puripuly_heart.core.error_messages import stt_failure_report
from puripuly_heart.core.local_asr.local_asr_provider_runtime import (
    LocalASRProviderRuntimePort,
    ProviderRuntimeBuildRequest,
)
from puripuly_heart.core.messages import ErrorDiagnostics
from puripuly_heart.core.owned_thread import run_owned_thread_call
from puripuly_heart.core.stt.recognition_probe import recognize_probe_audio
from puripuly_heart.core.stt.scoped_engine import ScopedRecognitionEngine
from puripuly_heart.runtime_layout import current_runtime_layout

RECOGNITION_PROBE_METADATA = {
    "recognition_result": "not_started",
    "captured_generation": None,
    "current_generation": None,
    "captured_settings_revision": None,
    "current_settings_revision": None,
    "final_event": None,
    "configured_provider": None,
    "exercised_provider": None,
    "recognition_scope": "isolated_configured_backend_session",
    "fixture_duration_seconds": 26336 / 22050,
    "accuracy": "not_exercised",
    "capture": "not_exercised",
    "translation": "not_exercised",
    "output_delivery": "not_exercised",
    "display": "not_exercised",
}


def load_recognition_probe_audio(sample_rate_hz: int) -> bytes:
    path = current_runtime_layout().package_resource("data", "diagnostics", "eung.wav")
    with wave.open(str(path), "rb") as audio:
        if (
            audio.getnchannels(),
            audio.getsampwidth(),
            audio.getframerate(),
            audio.getnframes(),
        ) != (1, 2, 22050, 26336):
            raise ValueError("recognition_probe_asset_invalid")
        pcm = audio.readframes(audio.getnframes())
    if len(pcm) != 26336 * 2:
        raise ValueError("recognition_probe_asset_invalid")
    if sample_rate_hz == 22050:
        return pcm
    samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
    return float32_to_pcm16le_bytes(
        resample_f32_linear(samples, from_rate_hz=22050, to_rate_hz=sample_rate_hz)
    )


async def run_recognition_probe(
    runtime_provider: Callable[[], LocalASRProviderRuntimePort | None],
    channel: Literal["self", "peer"],
    request_provider: Callable[[], ProviderRuntimeBuildRequest] | None = None,
    settings_revision_provider: Callable[[], int] | None = None,
) -> dict:
    result = {
        **RECOGNITION_PROBE_METADATA,
        "channel": channel,
        "status": "action_required",
        "recognition_result": "unavailable",
        "error": {"code": "recognition_backend_unavailable"},
    }
    runtime = runtime_provider()
    if runtime is None:
        return result
    handle = runtime.recognition_probe_handle(channel)
    original, generation = handle.current_provider_generation()
    request = None
    revision = settings_revision_provider() if settings_revision_provider is not None else None
    settings = None
    failure = None
    try:
        if request_provider is not None:
            request = request_provider()
        result.update(
            captured_generation=generation,
            current_generation=generation,
            captured_settings_revision=revision,
            current_settings_revision=revision,
        )
        async with runtime.recognition_probe_provider(channel, request) as (
            engine,
            admitted_generation,
        ):
            if admitted_generation != generation:
                result.update(status="failed", recognition_result="stale")
            elif (
                isinstance(engine, ScopedRecognitionEngine)
                and engine.is_live
                and not engine.probe_sessions
                and engine.probe_session_factory is not None
                and engine.probe_settings is not None
            ):
                settings = engine.probe_settings
                result.update(
                    configured_language=settings.source_language,
                    source_mode=settings.source_mode,
                    configured_provider=settings.provider_id,
                    recognition_scope=(
                        "single_eligible_rolling_member"
                        if settings.provider_id == "rolling_free"
                        else "isolated_configured_backend_session"
                    ),
                )
                expected_scope = (
                    (request.provider_id, request.provider_signature, request.runtime_signature)
                    if request is not None
                    else engine.scoped_settings_scope
                )
                if engine.scoped_settings_scope != expected_scope:
                    result["error"] = {"code": "recognition_configuration_not_applied"}
                else:
                    try:
                        pcm = await run_owned_thread_call(
                            lambda: load_recognition_probe_audio(settings.target_sample_rate_hz)
                        )
                    except (OSError, ValueError, wave.Error) as exc:
                        diagnostics = exception_diagnostics(
                            exc,
                            component="recognition_probe",
                            operation="recognition_probe",
                            code="recognition_probe_asset_unavailable",
                        )
                        result.update(
                            error={"code": "recognition_probe_asset_unavailable"},
                            diagnostics=diagnostics,
                            diagnostic_id=diagnostics.diagnostic_id,
                        )
                    else:
                        outcome = await recognize_probe_audio(
                            engine.probe_session_factory,
                            settings,
                            channel,
                            pcm,
                            engine.watchdog_resolver(settings),
                            engine.probe_sessions,
                        )
                        result.update(
                            status=(
                                "applied"
                                if outcome.result == "succeeded" and outcome.cleanup_complete
                                else "failed"
                            ),
                            recognition_result=(
                                outcome.result if outcome.cleanup_complete else "failed"
                            ),
                            recognition_execution_result=outcome.result,
                            final_event=outcome.final_event,
                            exercised_provider=outcome.exercised_provider,
                            input_samples=len(pcm) // 2,
                            sample_rate_hz=settings.target_sample_rate_hz,
                            cleanup_complete=outcome.cleanup_complete,
                            secondary_diagnostics=outcome.secondary_diagnostics,
                        )
                        if outcome.result == "succeeded" and outcome.cleanup_complete:
                            result.pop("error", None)
                        else:
                            result["error"] = {"code": "recognition_probe_no_final_result"}
                        diagnostics = outcome.diagnostics
                        if diagnostics is None and outcome.result != "succeeded":
                            diagnostics = ErrorDiagnostics(
                                component="recognition_probe",
                                operation="recognition_probe",
                                code="recognition_probe_no_final_result",
                                category="invalid_response",
                                visibility="diagnostic_only",
                                content_policy="metadata_only",
                                status_code=None,
                                retry_after_ms=None,
                                fields={},
                            )
                        if diagnostics is not None:
                            diagnostics = replace(diagnostics, channel=channel)
                            result.update(
                                diagnostics=diagnostics,
                                diagnostic_id=diagnostics.diagnostic_id,
                                failure_category=diagnostics.category,
                            )
    except asyncio.CancelledError as exc:
        result.update(status="cancelled", recognition_result="cancelled")
        result.pop("error", None)
        result["secondary_diagnostics"] = tuple(
            item
            for item in (
                getattr(exc, "failure_diagnostics", None),
                *getattr(exc, "secondary_diagnostics", ()),
            )
            if isinstance(item, ErrorDiagnostics)
        )
    except Exception as exc:
        failure = exc
        result["secondary_diagnostics"] = tuple(
            item
            for item in getattr(exc, "secondary_diagnostics", ())
            if isinstance(item, ErrorDiagnostics)
        )
    current = runtime_provider()
    result["current_generation"] = (
        current.recognition_probe_handle(channel).generation if current is not None else None
    )
    current_revision = (
        settings_revision_provider() if settings_revision_provider is not None else None
    )
    result["current_settings_revision"] = current_revision
    if failure is not None:
        diagnostics = stt_failure_report(
            failure,
            provider=settings.provider_id if settings is not None else "recognition",
            channel=channel,
            operation="recognition_probe",
        ).diagnostics
        result.update(
            status="failed",
            recognition_result="failed",
            error={"code": "recognition_probe_failed"},
            failure_category=diagnostics.category,
            diagnostics=diagnostics,
            diagnostic_id=diagnostics.diagnostic_id,
        )
    task = asyncio.current_task()
    if result["status"] == "cancelled" or (task is not None and task.cancelling()):
        result.update(status="cancelled", recognition_result="cancelled")
        result.pop("error", None)
        return result
    current_provider, current_generation = handle.current_provider_generation()
    if (
        current is not runtime
        or generation != current_generation
        or original is not current_provider
        or revision != current_revision
        or (isinstance(original, ScopedRecognitionEngine) and not original.is_live)
    ):
        return {
            **result,
            "status": "failed",
            "recognition_result": "stale",
            "error": {"code": "stale_provider_completion"},
        }
    return result
