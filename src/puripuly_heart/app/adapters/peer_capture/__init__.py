from importlib import import_module as _import_module

_SUBMODULES = frozenset(
    {
        "peer_capture_admission",
        "peer_capture_audio_loop",
        "peer_capture_inventory",
        "peer_capture_provider",
        "peer_capture_source",
        "peer_capture_target_resolver",
        "peer_capture_vad",
        "peer_capture_vad_sink",
    }
)
_EXPORT_SOURCES = {
    "PeerCaptureAdmissionAdapter": "peer_capture_admission",
    "PeerCaptureAudioLoopAdapter": "peer_capture_audio_loop",
    "PeerCaptureAudioLoopDetailedEnabled": "peer_capture_audio_loop",
    "PeerCaptureAudioLoopDetailedLog": "peer_capture_audio_loop",
    "PeerCaptureAudioLoopRunner": "peer_capture_audio_loop",
    "PeerCaptureDetailedEnabled": "peer_capture_source",
    "PeerCaptureDetailedLog": "peer_capture_source",
    "PeerCaptureLocalReadiness": "peer_capture_admission",
    "PeerCaptureLoopbackSourceFactory": "peer_capture_source",
    "PeerCapturePipelineFactory": "peer_capture_source",
    "PeerCaptureProcessResolution": "peer_capture_target_resolver",
    "PeerCaptureProcessResolver": "peer_capture_target_resolver",
    "PeerCaptureProcessResolverFactory": "peer_capture_target_resolver",
    "PeerCaptureProcessSourceFactory": "peer_capture_source",
    "PeerCaptureProcessWatcherFactory": "peer_capture_source",
    "PeerCaptureProviderAdapter": "peer_capture_provider",
    "PeerCaptureRuntimeAvailable": "peer_capture_admission",
    "PeerCaptureSourceAdapter": "peer_capture_source",
    "PeerCaptureSourceWrapper": "peer_capture_source",
    "PeerCaptureTargetResolverAdapter": "peer_capture_target_resolver",
    "PeerCaptureTargetRuntimeEffectsAdapter": "peer_capture_inventory",
    "PeerCaptureVadAdapter": "peer_capture_vad",
    "PeerCaptureVadDetailedLog": "peer_capture_vad",
    "PeerCaptureVadDiagnosticsEnabled": "peer_capture_vad",
    "PeerCaptureVadEngineFactory": "peer_capture_vad",
    "PeerCaptureVadGatingFactory": "peer_capture_vad",
    "PeerCaptureVadModelPathResolver": "peer_capture_vad",
    "PeerCaptureVadSinkAdapter": "peer_capture_vad_sink",
    "WindowsLoopbackDeviceInventoryAdapter": "peer_capture_inventory",
    "WindowsProcessCaptureInventoryAdapter": "peer_capture_inventory",
}

__all__ = [
    "peer_capture_admission",
    "peer_capture_audio_loop",
    "peer_capture_inventory",
    "peer_capture_provider",
    "peer_capture_source",
    "peer_capture_target_resolver",
    "peer_capture_vad",
    "peer_capture_vad_sink",
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
]


def __getattr__(name: str) -> object:
    if name in _SUBMODULES:
        return _import_module(f".{name}", __name__)
    source = _EXPORT_SOURCES.get(name)
    if source is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(_import_module(f".{source}", __name__), name)
