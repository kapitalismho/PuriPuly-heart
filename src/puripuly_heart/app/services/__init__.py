"""Application service transaction boundaries."""

from puripuly_heart._compat import install_moved_module_aliases as _install_moved_module_aliases

_install_moved_module_aliases(
    __name__,
    {
        "capture_target_settings": "puripuly_heart.app.services.capture.capture_target_settings",
        "peer_capture_target": "puripuly_heart.app.services.capture.peer_capture_target",
        "peer_capture_target_application": (
            "puripuly_heart.app.services.capture.peer_capture_target_application"
        ),
        "self_capture_application": "puripuly_heart.app.services.capture.self_capture_application",
    },
)
