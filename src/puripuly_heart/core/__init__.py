__all__ = ["clock"]

from puripuly_heart._compat import install_moved_module_aliases as _install_moved_module_aliases

_install_moved_module_aliases(
    __name__,
    {
        "discord_managed_oauth": "puripuly_heart.core.discord.discord_managed_oauth",
        "discord_oauth_loopback": "puripuly_heart.core.discord.discord_oauth_loopback",
        "oauth_callback_page": "puripuly_heart.core.discord.oauth_callback_page",
        "managed_openrouter_broker_client": (
            "puripuly_heart.core.openrouter.managed_openrouter_broker_client"
        ),
        "managed_openrouter_release": "puripuly_heart.core.openrouter.managed_openrouter_release",
        "openrouter_credentials": "puripuly_heart.core.openrouter.openrouter_credentials",
        "openrouter_handoff": "puripuly_heart.core.openrouter.openrouter_handoff",
        "openrouter_metadata": "puripuly_heart.core.openrouter.openrouter_metadata",
        "openrouter_pkce": "puripuly_heart.core.openrouter.openrouter_pkce",
        "local_asr_provider_runtime": "puripuly_heart.core.local_asr.local_asr_provider_runtime",
        "local_asr_provisioning": "puripuly_heart.core.local_asr.local_asr_provisioning",
        "local_gpu_assets": "puripuly_heart.core.local_asr.local_gpu_assets",
        "local_qwen_runtime": "puripuly_heart.core.local_asr.local_qwen_runtime",
        "local_stt_assets": "puripuly_heart.core.local_asr.local_stt_assets",
        "local_stt_catalog": "puripuly_heart.core.local_asr.local_stt_catalog",
        "local_stt_download_port": "puripuly_heart.core.local_asr.local_stt_download_port",
        "local_stt_huggingface_xet_adapter": (
            "puripuly_heart.core.local_asr.local_stt_huggingface_xet_adapter"
        ),
        "local_stt_runtime_installer": "puripuly_heart.core.local_asr.local_stt_runtime_installer",
    },
)
