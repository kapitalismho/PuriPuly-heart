from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

RuntimePipelineAsyncCallback = Callable[[], Awaitable[None]]
RuntimePipelineCloseCallback = Callable[[], Awaitable[None] | None]
RuntimePipelineOutputStartCallback = Callable[[bool], Awaitable[None]]


@dataclass(frozen=True, slots=True)
class RuntimePipelineStartCallbacks:
    start_output: RuntimePipelineOutputStartCallback
    open_self_ingress: RuntimePipelineAsyncCallback
    open_peer_ingress: RuntimePipelineAsyncCallback
    start_translation_turns: RuntimePipelineAsyncCallback
    start_local_asr: RuntimePipelineAsyncCallback


@dataclass(frozen=True, slots=True)
class RuntimePipelineCloseCallbacks:
    close_self_capture: RuntimePipelineCloseCallback
    close_peer_capture: RuntimePipelineCloseCallback
    close_self_ingress: RuntimePipelineCloseCallback
    close_peer_ingress: RuntimePipelineCloseCallback
    close_translation_turns: RuntimePipelineCloseCallback
    close_output: RuntimePipelineCloseCallback
    close_self_channel: RuntimePipelineCloseCallback
    close_peer_channel: RuntimePipelineCloseCallback
    close_local_asr: RuntimePipelineCloseCallback
    close_llm: RuntimePipelineCloseCallback
    close_sender: RuntimePipelineCloseCallback
