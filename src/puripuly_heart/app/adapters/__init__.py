from __future__ import annotations

from puripuly_heart._compat import install_moved_module_aliases as _install_moved_module_aliases

__all__ = []

_install_moved_module_aliases(
    __name__,
    {
        "peer_capture_admission": "puripuly_heart.app.adapters.peer_capture.peer_capture_admission",
        "peer_capture_audio_loop": (
            "puripuly_heart.app.adapters.peer_capture.peer_capture_audio_loop"
        ),
        "peer_capture_inventory": "puripuly_heart.app.adapters.peer_capture.peer_capture_inventory",
        "peer_capture_provider": "puripuly_heart.app.adapters.peer_capture.peer_capture_provider",
        "peer_capture_source": "puripuly_heart.app.adapters.peer_capture.peer_capture_source",
        "peer_capture_target_resolver": (
            "puripuly_heart.app.adapters.peer_capture.peer_capture_target_resolver"
        ),
        "peer_capture_vad": "puripuly_heart.app.adapters.peer_capture.peer_capture_vad",
        "peer_capture_vad_sink": "puripuly_heart.app.adapters.peer_capture.peer_capture_vad_sink",
        "self_capture_admission": "puripuly_heart.app.adapters.self_capture.self_capture_admission",
        "self_capture_audio_loop": (
            "puripuly_heart.app.adapters.self_capture.self_capture_audio_loop"
        ),
        "self_capture_provider": "puripuly_heart.app.adapters.self_capture.self_capture_provider",
        "self_capture_source": "puripuly_heart.app.adapters.self_capture.self_capture_source",
        "self_capture_vad": "puripuly_heart.app.adapters.self_capture.self_capture_vad",
        "self_capture_vad_sink": "puripuly_heart.app.adapters.self_capture.self_capture_vad_sink",
    },
)
