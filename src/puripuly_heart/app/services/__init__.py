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
        "local_asr_cpu_repair": "puripuly_heart.app.services.local_asr.local_asr_cpu_repair",
        "local_asr_diagnostics": "puripuly_heart.app.services.local_asr.local_asr_diagnostics",
        "local_asr_gpu_provisioning": (
            "puripuly_heart.app.services.local_asr.local_asr_gpu_provisioning"
        ),
        "local_asr_readiness": "puripuly_heart.app.services.local_asr.local_asr_readiness",
        "local_asr_selection": "puripuly_heart.app.services.local_asr.local_asr_selection",
    },
)
