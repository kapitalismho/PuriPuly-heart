import sys as _sys
from importlib import import_module as _import_module

from puripuly_heart.app import services as _parent

for _short in (
    "capture_target_settings",
    "peer_capture_target",
    "peer_capture_target_application",
    "self_capture_application",
):
    _module = _import_module(f".{_short}", __name__)
    _sys.modules[f"puripuly_heart.app.services.{_short}"] = _module
    setattr(_parent, _short, _module)
del _module, _short

from .capture_target_settings import (
    CaptureTargetSettingsError,
    persist_desktop_audio_capture_target,
)
from .peer_capture_target import PeerCaptureTargetResolutionService
from .peer_capture_target_application import (
    Localizer,
    PeerCaptureTargetApplicationOwner,
    SettingsPresentationSink,
    WarningReset,
)
from .self_capture_application import (
    SelfCaptureApplicationOwner,
    SelfCaptureApplicationSettings,
)

__all__ = [
    "CaptureTargetSettingsError",
    "Localizer",
    "PeerCaptureTargetApplicationOwner",
    "PeerCaptureTargetResolutionService",
    "SelfCaptureApplicationOwner",
    "SelfCaptureApplicationSettings",
    "SettingsPresentationSink",
    "WarningReset",
    "capture_target_settings",
    "peer_capture_target",
    "peer_capture_target_application",
    "persist_desktop_audio_capture_target",
    "self_capture_application",
]
