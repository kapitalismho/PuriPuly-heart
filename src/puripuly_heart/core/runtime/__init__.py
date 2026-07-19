from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from puripuly_heart.core.runtime.clipboard import ClipboardRuntime
    from puripuly_heart.core.runtime.github_star_prompt import GithubStarPromptRuntime
    from puripuly_heart.core.runtime.local_asr_provisioning import LocalASRProvisioningOwner
    from puripuly_heart.core.runtime.local_stt_download import LocalSTTDownloadRuntime
    from puripuly_heart.core.runtime.logging import RuntimeLoggingCloseError, RuntimeLoggingService
    from puripuly_heart.core.runtime.mic_test import MicTestRuntime
    from puripuly_heart.core.runtime.oauth import OAuthRuntime
    from puripuly_heart.core.runtime.output import OutputRuntime
    from puripuly_heart.core.runtime.peer_channel import (
        PeerChannelRuntime,
        PeerChannelRuntimeState,
        PeerRuntimeConfig,
        SpeechChannelRuntime,
    )
    from puripuly_heart.core.runtime.receiver import OscReceiverRuntime, VrcMicReceiverRuntime

__all__ = [
    "PeerChannelRuntime",
    "PeerChannelRuntimeState",
    "PeerRuntimeConfig",
    "ClipboardRuntime",
    "GithubStarPromptRuntime",
    "LocalASRProvisioningOwner",
    "LocalSTTDownloadRuntime",
    "MicTestRuntime",
    "OAuthRuntime",
    "OscReceiverRuntime",
    "OutputRuntime",
    "RuntimeLoggingCloseError",
    "RuntimeLoggingService",
    "SpeechChannelRuntime",
    "VrcMicReceiverRuntime",
]


def __getattr__(name: str) -> object:
    if name in __all__:
        if name == "ClipboardRuntime":
            from puripuly_heart.core.runtime import clipboard

            return getattr(clipboard, name)
        if name == "GithubStarPromptRuntime":
            from puripuly_heart.core.runtime import github_star_prompt

            return getattr(github_star_prompt, name)
        if name == "LocalASRProvisioningOwner":
            from puripuly_heart.core.runtime import local_asr_provisioning

            return getattr(local_asr_provisioning, name)
        if name == "LocalSTTDownloadRuntime":
            from puripuly_heart.core.runtime import local_stt_download

            return getattr(local_stt_download, name)
        if name in {"RuntimeLoggingCloseError", "RuntimeLoggingService"}:
            from puripuly_heart.core.runtime import logging as runtime_logging

            return getattr(runtime_logging, name)
        if name == "MicTestRuntime":
            from puripuly_heart.core.runtime import mic_test

            return getattr(mic_test, name)
        if name == "OAuthRuntime":
            from puripuly_heart.core.runtime import oauth

            return getattr(oauth, name)
        if name == "OutputRuntime":
            from puripuly_heart.core.runtime import output

            return getattr(output, name)
        if name in {"OscReceiverRuntime", "VrcMicReceiverRuntime"}:
            from puripuly_heart.core.runtime import receiver

            return getattr(receiver, name)
        from puripuly_heart.core.runtime import peer_channel

        return getattr(peer_channel, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
