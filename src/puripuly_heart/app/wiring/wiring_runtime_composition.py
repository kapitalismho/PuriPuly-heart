from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from puripuly_heart.app.services.provider_settings import ProviderApplicationOwner
from puripuly_heart.core.runtime.self_capture import SelfCaptureSessionOwner

from .wiring_managed_account import ManagedAccountComponents
from .wiring_provider_runtime import ProviderRuntimeComponents
from .wiring_runtime_pipeline import RuntimePipelineLauncher


@dataclass(frozen=True, slots=True)
class RuntimeCompositionComponents:
    self_capture_owner: Callable[[], SelfCaptureSessionOwner]
    provider_runtime: ProviderRuntimeComponents
    managed_account: ManagedAccountComponents
    provider_application: ProviderApplicationOwner
    pipeline_launcher: RuntimePipelineLauncher
