from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field

from puripuly_heart.app.ports.provider_verifier import ProviderVerifierPort
from puripuly_heart.core.lifecycle import LifecycleScope, start_lifecycle_task

ProviderStatusVerificationDiagnosticsSink = Callable[
    [str, Mapping[str, object], BaseException | None],
    None,
]


@dataclass(frozen=True, slots=True)
class ConfiguredProviderStatusVerificationRequest:
    llm_runtime_present: bool
    stt_runtime_present: bool
    llm_provider: str
    stt_provider: str
    llm_requires_secret: bool
    stt_requires_secret: bool
    runtime_translation_enabled: bool
    managed_openrouter_can_attempt: bool
    openrouter_managed_selected: bool
    gemini_model: str
    qwen_selected_model: str
    qwen_fallback_models: tuple[str, ...]
    qwen_base_url: str
    fast_translation_enabled: bool
    google_api_key: str = field(repr=False)
    openrouter_api_key: str = field(repr=False)
    deepseek_api_key: str = field(repr=False)
    qwen_api_key: str = field(repr=False)
    deepgram_api_key: str = field(repr=False)
    soniox_api_key: str = field(repr=False)


@dataclass(frozen=True, slots=True)
class ConfiguredProviderStatusVerificationResult:
    llm_valid: bool
    stt_valid: bool
    translation_needs_key: bool
    stt_needs_key: bool
    translation_enabled_update: bool | None
    stt_enabled_update: bool | None


ConfiguredProviderStatusResultHandler = Callable[
    [ConfiguredProviderStatusVerificationResult],
    Awaitable[None] | None,
]
ConfiguredProviderStatusRequestFactory = Callable[
    [],
    ConfiguredProviderStatusVerificationRequest
    | Awaitable[ConfiguredProviderStatusVerificationRequest | None]
    | None,
]


def _provider_status_verification_scope() -> LifecycleScope:
    return LifecycleScope("ProviderStatusVerificationOwner")


@dataclass(slots=True)
class ProviderStatusVerificationOwner:
    verifier: ProviderVerifierPort
    diagnostics_sink: ProviderStatusVerificationDiagnosticsSink | None = None
    _task_scope: LifecycleScope = field(
        init=False,
        default_factory=_provider_status_verification_scope,
        repr=False,
    )
    _task_sequence: int = field(init=False, default=0, repr=False)
    _ingress_stopped: bool = field(init=False, default=False, repr=False)

    @property
    def owner_name(self) -> str:
        return "ProviderStatusVerificationOwner"

    @property
    def active_task_names(self) -> tuple[str, ...]:
        return self._task_scope.active_task_names

    def lifecycle_owner_snapshot(self) -> dict[str, object]:
        return {
            "owner": self.owner_name,
            "resource_fields": ("_task_scope", "_task_sequence"),
            "stop_ingress": "reject new provider-status verification work",
            "shutdown_policy": "cancel and await every provider-status verification task",
            "late_callback_rule": "drop provider-status publication after ingress stops",
        }

    def schedule(
        self,
        *,
        request_factory: ConfiguredProviderStatusRequestFactory,
        result_handler: ConfiguredProviderStatusResultHandler,
    ) -> bool:
        if self._ingress_stopped or self._task_scope.is_closed:
            return False

        async def run() -> None:
            try:
                request_outcome = request_factory()
                request = (
                    await request_outcome
                    if inspect.isawaitable(request_outcome)
                    else request_outcome
                )
                if request is None:
                    return
                result = await self.verify(request)
                if self._ingress_stopped or self._task_scope.is_closed:
                    return
                outcome = result_handler(result)
                if inspect.isawaitable(outcome):
                    await outcome
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._emit(
                    "provider_status_verification_failed",
                    {"error_type": type(exc).__name__},
                    exc,
                )

        self._task_sequence += 1
        coroutine = run()
        try:
            start_lifecycle_task(
                self._task_scope,
                coroutine,
                name=f"verification-{self._task_sequence}",
            )
        except Exception as exc:
            coroutine.close()
            self._emit(
                "provider_status_verification_schedule_failed",
                {"error_type": type(exc).__name__},
                exc,
            )
            return False
        return True

    async def verify(
        self,
        request: ConfiguredProviderStatusVerificationRequest,
    ) -> ConfiguredProviderStatusVerificationResult:
        qwen_selected_valid: bool | None = None
        qwen_any_valid: bool | None = None

        async def verify_qwen_selected() -> bool:
            nonlocal qwen_selected_valid
            if qwen_selected_valid is not None:
                return qwen_selected_valid
            if not request.qwen_api_key:
                qwen_selected_valid = False
                return False
            qwen_selected_valid = await self.verifier.verify_qwen_llm_api_key(
                request.qwen_api_key,
                base_url=request.qwen_base_url,
                model=request.qwen_selected_model,
                low_latency=request.fast_translation_enabled,
            )
            return qwen_selected_valid

        async def verify_any_qwen_model() -> bool:
            nonlocal qwen_any_valid
            if qwen_any_valid is not None:
                return qwen_any_valid
            if await verify_qwen_selected():
                qwen_any_valid = True
                return True
            if not request.qwen_api_key:
                qwen_any_valid = False
                return False
            for fallback_model in request.qwen_fallback_models:
                if fallback_model == request.qwen_selected_model:
                    continue
                if await self.verifier.verify_qwen_llm_api_key(
                    request.qwen_api_key,
                    base_url=request.qwen_base_url,
                    model=fallback_model,
                    low_latency=request.fast_translation_enabled,
                ):
                    qwen_any_valid = True
                    return True
            qwen_any_valid = False
            return False

        llm_valid = False
        if request.llm_runtime_present:
            try:
                if request.llm_provider == "gemini":
                    llm_valid = await self.verifier.verify_api_key(
                        "google",
                        request.google_api_key,
                        model=request.gemini_model,
                    )
                elif request.llm_provider == "openrouter":
                    if request.openrouter_managed_selected and not request.openrouter_api_key:
                        llm_valid = request.managed_openrouter_can_attempt
                    else:
                        llm_valid = bool(
                            request.openrouter_api_key
                        ) and await self.verifier.verify_api_key(
                            "openrouter",
                            request.openrouter_api_key,
                        )
                elif request.llm_provider == "deepseek":
                    llm_valid = bool(
                        request.deepseek_api_key
                    ) and await self.verifier.verify_api_key(
                        "deepseek",
                        request.deepseek_api_key,
                    )
                elif request.llm_provider == "qwen":
                    llm_valid = await verify_qwen_selected()
                elif request.llm_provider == "local_llm":
                    llm_valid = True
                else:
                    llm_valid = True
            except Exception:
                llm_valid = False

        stt_valid = not request.stt_requires_secret
        if request.stt_runtime_present and request.stt_requires_secret:
            try:
                if request.stt_provider == "deepgram":
                    stt_valid = await self.verifier.verify_api_key(
                        "deepgram",
                        request.deepgram_api_key,
                    )
                elif request.stt_provider == "qwen_asr":
                    stt_valid = await verify_any_qwen_model()
                elif request.stt_provider == "soniox":
                    stt_valid = await self.verifier.verify_api_key(
                        "soniox",
                        request.soniox_api_key,
                    )
                else:
                    stt_valid = True
            except Exception:
                stt_valid = False

        return ConfiguredProviderStatusVerificationResult(
            llm_valid=llm_valid,
            stt_valid=stt_valid,
            translation_needs_key=(request.llm_requires_secret if not llm_valid else False),
            stt_needs_key=(request.stt_requires_secret if not stt_valid else False),
            translation_enabled_update=(
                False
                if not llm_valid
                else (
                    request.runtime_translation_enabled
                    if request.llm_provider == "local_llm"
                    else None
                )
            ),
            stt_enabled_update=False if not stt_valid else None,
        )

    def stop_ingress(self) -> None:
        self._ingress_stopped = True

    async def close(self) -> None:
        self.stop_ingress()
        await self._task_scope.close()

    def _emit(
        self,
        event: str,
        metadata: Mapping[str, object],
        exception: BaseException | None = None,
    ) -> None:
        if self.diagnostics_sink is None:
            return
        try:
            self.diagnostics_sink(event, metadata, exception)
        except Exception:
            return


__all__ = [
    "ConfiguredProviderStatusRequestFactory",
    "ConfiguredProviderStatusResultHandler",
    "ConfiguredProviderStatusVerificationRequest",
    "ConfiguredProviderStatusVerificationResult",
    "ProviderStatusVerificationOwner",
]
