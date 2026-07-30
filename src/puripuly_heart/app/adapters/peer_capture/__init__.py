import sys as _sys
from importlib import import_module as _import_module

from puripuly_heart.app import adapters as _parent

for _short in (
    "peer_capture_admission",
    "peer_capture_audio_loop",
    "peer_capture_inventory",
    "peer_capture_provider",
    "peer_capture_source",
    "peer_capture_target_resolver",
    "peer_capture_vad",
    "peer_capture_vad_sink",
):
    _module = _import_module(f".{_short}", __name__)
    _sys.modules[f"puripuly_heart.app.adapters.{_short}"] = _module
    setattr(_parent, _short, _module)
del _module, _short

from .peer_capture_admission import (
    PeerCaptureAdmissionAdapter,
    PeerCaptureLocalReadiness,
    PeerCaptureRuntimeAvailable,
)
from .peer_capture_audio_loop import (
    PeerCaptureAudioLoopAdapter,
    PeerCaptureAudioLoopDetailedEnabled,
    PeerCaptureAudioLoopDetailedLog,
    PeerCaptureAudioLoopRunner,
)
from .peer_capture_inventory import (
    PeerCaptureTargetRuntimeEffectsAdapter,
    WindowsLoopbackDeviceInventoryAdapter,
    WindowsProcessCaptureInventoryAdapter,
)
from .peer_capture_provider import PeerCaptureProviderAdapter
from .peer_capture_source import (
    PeerCaptureDetailedEnabled,
    PeerCaptureDetailedLog,
    PeerCaptureLoopbackSourceFactory,
    PeerCapturePipelineFactory,
    PeerCaptureProcessSourceFactory,
    PeerCaptureProcessWatcherFactory,
    PeerCaptureSourceAdapter,
    PeerCaptureSourceWrapper,
)
from .peer_capture_target_resolver import (
    PeerCaptureProcessResolution,
    PeerCaptureProcessResolver,
    PeerCaptureProcessResolverFactory,
    PeerCaptureTargetResolverAdapter,
)
from .peer_capture_vad import (
    PeerCaptureVadAdapter,
    PeerCaptureVadDetailedLog,
    PeerCaptureVadDiagnosticsEnabled,
    PeerCaptureVadEngineFactory,
    PeerCaptureVadGatingFactory,
    PeerCaptureVadModelPathResolver,
)
from .peer_capture_vad_sink import PeerCaptureVadSinkAdapter

__all__ = [
    "PeerCaptureAdmissionAdapter",
    "PeerCaptureAudioLoopAdapter",
    "PeerCaptureAudioLoopDetailedEnabled",
    "PeerCaptureAudioLoopDetailedLog",
    "PeerCaptureAudioLoopRunner",
    "PeerCaptureDetailedEnabled",
    "PeerCaptureDetailedLog",
    "PeerCaptureLocalReadiness",
    "PeerCaptureLoopbackSourceFactory",
    "PeerCapturePipelineFactory",
    "PeerCaptureProcessResolution",
    "PeerCaptureProcessResolver",
    "PeerCaptureProcessResolverFactory",
    "PeerCaptureProcessSourceFactory",
    "PeerCaptureProcessWatcherFactory",
    "PeerCaptureProviderAdapter",
    "PeerCaptureRuntimeAvailable",
    "PeerCaptureSourceAdapter",
    "PeerCaptureSourceWrapper",
    "PeerCaptureTargetResolverAdapter",
    "PeerCaptureTargetRuntimeEffectsAdapter",
    "PeerCaptureVadAdapter",
    "PeerCaptureVadDetailedLog",
    "PeerCaptureVadDiagnosticsEnabled",
    "PeerCaptureVadEngineFactory",
    "PeerCaptureVadGatingFactory",
    "PeerCaptureVadModelPathResolver",
    "PeerCaptureVadSinkAdapter",
    "WindowsLoopbackDeviceInventoryAdapter",
    "WindowsProcessCaptureInventoryAdapter",
    "peer_capture_admission",
    "peer_capture_audio_loop",
    "peer_capture_inventory",
    "peer_capture_provider",
    "peer_capture_source",
    "peer_capture_target_resolver",
    "peer_capture_vad",
    "peer_capture_vad_sink",
]
