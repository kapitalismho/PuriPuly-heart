import sys as _sys
from importlib import import_module as _import_module

from puripuly_heart.app import adapters as _parent

for _short in (
    "self_capture_admission",
    "self_capture_audio_loop",
    "self_capture_provider",
    "self_capture_source",
    "self_capture_vad",
    "self_capture_vad_sink",
):
    _module = _import_module(f".{_short}", __name__)
    _sys.modules[f"puripuly_heart.app.adapters.{_short}"] = _module
    setattr(_parent, _short, _module)
del _module, _short

from .self_capture_admission import SelfCaptureAdmissionAdapter
from .self_capture_audio_loop import (
    SelfCaptureAudioGateProvider,
    SelfCaptureAudioLoopAdapter,
    SelfCaptureAudioLoopDetailedEnabled,
    SelfCaptureAudioLoopDetailedLog,
    SelfCaptureAudioLoopRunner,
)
from .self_capture_provider import SelfCaptureProviderAdapter
from .self_capture_source import (
    SelfCaptureAudioSourceFactory,
    SelfCaptureChannelDecision,
    SelfCaptureDetailedLog,
    SelfCaptureDeviceResolver,
    SelfCaptureHostApiNormalizer,
    SelfCaptureSourceAdapter,
    SelfCaptureSourceWrapper,
)
from .self_capture_vad import (
    SelfCaptureVadAdapter,
    SelfCaptureVadDetailedLog,
    SelfCaptureVadDiagnosticsEnabled,
    SelfCaptureVadEngineFactory,
    SelfCaptureVadGatingFactory,
    SelfCaptureVadModelPathResolver,
)
from .self_capture_vad_sink import SelfCaptureVadSinkAdapter

__all__ = [
    "SelfCaptureAdmissionAdapter",
    "SelfCaptureAudioGateProvider",
    "SelfCaptureAudioLoopAdapter",
    "SelfCaptureAudioLoopDetailedEnabled",
    "SelfCaptureAudioLoopDetailedLog",
    "SelfCaptureAudioLoopRunner",
    "SelfCaptureAudioSourceFactory",
    "SelfCaptureChannelDecision",
    "SelfCaptureDetailedLog",
    "SelfCaptureDeviceResolver",
    "SelfCaptureHostApiNormalizer",
    "SelfCaptureProviderAdapter",
    "SelfCaptureSourceAdapter",
    "SelfCaptureSourceWrapper",
    "SelfCaptureVadAdapter",
    "SelfCaptureVadDetailedLog",
    "SelfCaptureVadDiagnosticsEnabled",
    "SelfCaptureVadEngineFactory",
    "SelfCaptureVadGatingFactory",
    "SelfCaptureVadModelPathResolver",
    "SelfCaptureVadSinkAdapter",
    "self_capture_admission",
    "self_capture_audio_loop",
    "self_capture_provider",
    "self_capture_source",
    "self_capture_vad",
    "self_capture_vad_sink",
]
