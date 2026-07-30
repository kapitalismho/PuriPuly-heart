__all__ = ["wiring"]

from puripuly_heart._compat import install_moved_module_aliases as _install_moved_module_aliases

_install_moved_module_aliases(
    __name__,
    {
        "wiring_application_runtime_logging": (
            "puripuly_heart.app.wiring.wiring_application_runtime_logging"
        ),
        "wiring_capture_runtime": "puripuly_heart.app.wiring.wiring_capture_runtime",
        "wiring_composition": "puripuly_heart.app.wiring.wiring_composition",
        "wiring_llm_factory": "puripuly_heart.app.wiring.wiring_llm_factory",
        "wiring_local_asr_application": "puripuly_heart.app.wiring.wiring_local_asr_application",
        "wiring_local_asr_provider_runtime": (
            "puripuly_heart.app.wiring.wiring_local_asr_provider_runtime"
        ),
        "wiring_managed_account": "puripuly_heart.app.wiring.wiring_managed_account",
        "wiring_managed_auth_factory": "puripuly_heart.app.wiring.wiring_managed_auth_factory",
        "wiring_microphone_test": "puripuly_heart.app.wiring.wiring_microphone_test",
        "wiring_overlay_factory": "puripuly_heart.app.wiring.wiring_overlay_factory",
        "wiring_peer_application": "puripuly_heart.app.wiring.wiring_peer_application",
        "wiring_provider_runtime": "puripuly_heart.app.wiring.wiring_provider_runtime",
        "wiring_provider_runtime_policy": "puripuly_heart.app.wiring.wiring_provider_runtime_policy",
        "wiring_runtime_composition": "puripuly_heart.app.wiring.wiring_runtime_composition",
        "wiring_runtime_pipeline": "puripuly_heart.app.wiring.wiring_runtime_pipeline",
        "wiring_secrets_factory": "puripuly_heart.app.wiring.wiring_secrets_factory",
        "wiring_stt_factory": "puripuly_heart.app.wiring.wiring_stt_factory",
        "wiring_translation_runtime_configuration": (
            "puripuly_heart.app.wiring.wiring_translation_runtime_configuration"
        ),
        "wiring_vrc_mic_sync": "puripuly_heart.app.wiring.wiring_vrc_mic_sync",
    },
)
