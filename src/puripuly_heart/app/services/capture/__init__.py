from importlib import import_module as _import_module

_SUBMODULES = frozenset(
    {
        "capture_target_settings",
        "peer_capture_target",
        "peer_capture_target_application",
        "self_capture_application",
    }
)
_EXPORT_SOURCES = {
    "CaptureTargetSettingsError": "capture_target_settings",
    "Localizer": "peer_capture_target_application",
    "PeerCaptureTargetApplicationOwner": "peer_capture_target_application",
    "PeerCaptureTargetResolutionService": "peer_capture_target",
    "persist_desktop_audio_capture_target": "capture_target_settings",
    "SelfCaptureApplicationOwner": "self_capture_application",
    "SelfCaptureApplicationSettings": "self_capture_application",
    "SettingsPresentationSink": "peer_capture_target_application",
    "WarningReset": "peer_capture_target_application",
}

__all__ = [
    "capture_target_settings",
    "CaptureTargetSettingsError",
    "Localizer",
    "peer_capture_target",
    "peer_capture_target_application",
    "PeerCaptureTargetApplicationOwner",
    "PeerCaptureTargetResolutionService",
    "persist_desktop_audio_capture_target",
    "self_capture_application",
    "SelfCaptureApplicationOwner",
    "SelfCaptureApplicationSettings",
    "SettingsPresentationSink",
    "WarningReset",
]


def __getattr__(name: str) -> object:
    if name in _SUBMODULES:
        return _import_module(f".{name}", __name__)
    source = _EXPORT_SOURCES.get(name)
    if source is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(_import_module(f".{source}", __name__), name)
