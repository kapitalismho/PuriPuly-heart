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
        "overlay_application": "puripuly_heart.app.services.overlay.overlay_application",
        "overlay_calibration": "puripuly_heart.app.services.overlay.overlay_calibration",
        "overlay_calibration_application": (
            "puripuly_heart.app.services.overlay.overlay_calibration_application"
        ),
        "overlay_generation_start": "puripuly_heart.app.services.overlay.overlay_generation_start",
        "overlay_session_transition": (
            "puripuly_heart.app.services.overlay.overlay_session_transition"
        ),
        "managed_auth": "puripuly_heart.app.services.managed.managed_auth",
        "managed_auth_claims": "puripuly_heart.app.services.managed.managed_auth_claims",
        "managed_connection_auth": "puripuly_heart.app.services.managed.managed_connection_auth",
        "managed_key_delivery_ack": "puripuly_heart.app.services.managed.managed_key_delivery_ack",
        "managed_status_refresh": "puripuly_heart.app.services.managed.managed_status_refresh",
        "managed_usage": "puripuly_heart.app.services.managed.managed_usage",
        "provider_credential_verification": (
            "puripuly_heart.app.services.provider.provider_credential_verification"
        ),
        "provider_runtime_apply": "puripuly_heart.app.services.provider.provider_runtime_apply",
        "provider_secret_change": "puripuly_heart.app.services.provider.provider_secret_change",
        "provider_settings": "puripuly_heart.app.services.provider.provider_settings",
        "provider_verification_binding": (
            "puripuly_heart.app.services.provider.provider_verification_binding"
        ),
        "settings_application": "puripuly_heart.app.services.settings.settings_application",
        "settings_mutation": "puripuly_heart.app.services.settings.settings_mutation",
        "settings_mutation_legacy": "puripuly_heart.app.services.settings.settings_mutation_legacy",
        "settings_projection": "puripuly_heart.app.services.settings.settings_projection",
        "settings_runtime_effects": "puripuly_heart.app.services.settings.settings_runtime_effects",
        "settings_transaction_result": (
            "puripuly_heart.app.services.settings.settings_transaction_result"
        ),
    },
)
