from importlib import import_module as _import_module

_SUBMODULES = frozenset(
    {
        "self_capture_admission",
        "self_capture_audio_loop",
        "self_capture_provider",
        "self_capture_source",
        "self_capture_vad",
        "self_capture_vad_sink",
    }
)
_EXPORT_SOURCES = {
    "SelfCaptureAdmissionAdapter": "self_capture_admission",
    "SelfCaptureAudioGateProvider": "self_capture_audio_loop",
    "SelfCaptureAudioLoopAdapter": "self_capture_audio_loop",
    "SelfCaptureAudioLoopDetailedEnabled": "self_capture_audio_loop",
    "SelfCaptureAudioLoopDetailedLog": "self_capture_audio_loop",
    "SelfCaptureAudioLoopRunner": "self_capture_audio_loop",
    "SelfCaptureAudioSourceFactory": "self_capture_source",
    "SelfCaptureChannelDecision": "self_capture_source",
    "SelfCaptureDetailedLog": "self_capture_source",
    "SelfCaptureDeviceResolver": "self_capture_source",
    "SelfCaptureHostApiNormalizer": "self_capture_source",
    "SelfCaptureProviderAdapter": "self_capture_provider",
    "SelfCaptureSourceAdapter": "self_capture_source",
    "SelfCaptureSourceWrapper": "self_capture_source",
    "SelfCaptureVadAdapter": "self_capture_vad",
    "SelfCaptureVadDetailedLog": "self_capture_vad",
    "SelfCaptureVadDiagnosticsEnabled": "self_capture_vad",
    "SelfCaptureVadEngineFactory": "self_capture_vad",
    "SelfCaptureVadGatingFactory": "self_capture_vad",
    "SelfCaptureVadModelPathResolver": "self_capture_vad",
    "SelfCaptureVadSinkAdapter": "self_capture_vad_sink",
}

__all__ = [
    "self_capture_admission",
    "self_capture_audio_loop",
    "self_capture_provider",
    "self_capture_source",
    "self_capture_vad",
    "self_capture_vad_sink",
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
]


def __getattr__(name: str) -> object:
    if name in _SUBMODULES:
        return _import_module(f".{name}", __name__)
    source = _EXPORT_SOURCES.get(name)
    if source is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(_import_module(f".{source}", __name__), name)
