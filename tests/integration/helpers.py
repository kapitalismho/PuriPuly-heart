from __future__ import annotations

import asyncio
import inspect
import ipaddress
import os
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Awaitable, Callable, Mapping
from urllib.parse import urlsplit
from uuid import uuid4

import numpy as np
import pytest

from puripuly_heart.config.provider_values import QwenRegion

INTEGRATION_ENV = "INTEGRATION"
EVENT_POLL_TIMEOUT_S = float(os.getenv("INTEGRATION_EVENT_POLL_TIMEOUT_S", "0.1"))
WARMUP_DELAY_S = float(os.getenv("INTEGRATION_WARMUP_S", "0.5"))
CHUNK_DELAY_S = float(os.getenv("INTEGRATION_CHUNK_DELAY_S", "0.05"))
RESULT_TIMEOUT_S = float(os.getenv("INTEGRATION_RESULT_TIMEOUT_S", "30"))
OSC_TIMEOUT_S = float(os.getenv("INTEGRATION_OSC_TIMEOUT_S", "15"))
ITERATION_DELAY_S = float(os.getenv("INTEGRATION_ITERATION_DELAY_S", "1.0"))
OPEN_SESSION_TIMEOUT_S = float(os.getenv("INTEGRATION_OPEN_TIMEOUT_S", "15"))
LOCAL_QWEN_STT_SAMPLE_RATE_HZ = 16000


@dataclass(frozen=True, slots=True)
class LLMSmokeInput:
    text: str = "안녕하세요"
    source_language: str = "ko"
    target_language: str = "en"
    system_prompt: str = "Translate from ${sourceName} to ${targetName}."
    context: str = ""


@dataclass(slots=True)
class SuppressedRuntimeLogger:
    emitted_count: int = 0

    def emit_basic(self, *_args: object, **_kwargs: object) -> None:
        self.emitted_count += 1

    def emit_detailed(self, *_args: object, **_kwargs: object) -> bool:
        self.emitted_count += 1
        return False

    def emit_detailed_lazy(self, *_args: object, **_kwargs: object) -> bool:
        self.emitted_count += 1
        return False


def integration_mark():
    return pytest.mark.skipif(
        os.getenv(INTEGRATION_ENV) != "1",
        reason="set INTEGRATION=1 to run integration tests",
    )


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        pytest.skip(f"missing env var {name}")
    return value


def require_module(module: str, *, reason: str) -> None:
    try:
        __import__(module)
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError(reason) from exc


def require_optional_module(module: str, *, reason: str):
    try:
        return __import__(module)
    except ImportError:
        pytest.skip(reason)


def suppressed_runtime_logger() -> SuppressedRuntimeLogger:
    return SuppressedRuntimeLogger()


def assert_non_empty_translation(translation: object) -> None:
    text = getattr(translation, "text", "")
    assert isinstance(text, str) and text.strip()


async def close_async_resource(resource: object) -> None:
    close = getattr(resource, "close", None)
    if not callable(close):
        return
    result = close()
    if inspect.isawaitable(result):
        await result


async def run_llm_smoke(
    provider: object,
    *,
    smoke_input: LLMSmokeInput | None = None,
    close: bool = True,
):
    smoke = smoke_input or LLMSmokeInput()
    try:
        translation = await provider.translate(
            utterance_id=uuid4(),
            text=smoke.text,
            system_prompt=smoke.system_prompt,
            source_language=smoke.source_language,
            target_language=smoke.target_language,
            context=smoke.context,
        )
        assert_non_empty_translation(translation)
        return translation
    finally:
        if close:
            await close_async_resource(provider)


def resolve_test_audio_path(
    *, env_var: str = "TEST_AUDIO_PATH", filename: str = "test_speech.wav"
) -> Path:
    audio_env = os.getenv(env_var)
    if audio_env:
        return Path(audio_env)
    return Path(__file__).resolve().parents[2] / ".test_audio" / filename


def load_audio_wav(path: str | Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as f:
        sample_rate = f.getframerate()
        n_frames = f.getnframes()
        audio_data = f.readframes(n_frames)

    samples_int16 = np.frombuffer(audio_data, dtype=np.int16)
    samples_f32 = samples_int16.astype(np.float32) / 32768.0
    return samples_f32, sample_rate


def require_test_audio_path(path: str | Path | None = None) -> Path:
    resolved = Path(path) if path is not None else resolve_test_audio_path()
    if not resolved.exists() or not resolved.is_file():
        pytest.skip("test audio is unavailable")
    return resolved


def require_supported_audio_sample_rate(
    sample_rate_hz: int,
    *,
    expected_sample_rate_hz: int = LOCAL_QWEN_STT_SAMPLE_RATE_HZ,
) -> None:
    if sample_rate_hz != expected_sample_rate_hz:
        pytest.skip(f"unsupported audio sample rate; expected {expected_sample_rate_hz} Hz")


def load_required_audio_wav(
    path: str | Path | None = None,
    *,
    expected_sample_rate_hz: int = LOCAL_QWEN_STT_SAMPLE_RATE_HZ,
) -> tuple[np.ndarray, int]:
    audio_path = require_test_audio_path(path)
    try:
        samples, sample_rate = load_audio_wav(audio_path)
    except (OSError, EOFError, wave.Error):
        pytest.skip("test audio is unavailable")
    require_supported_audio_sample_rate(
        sample_rate,
        expected_sample_rate_hz=expected_sample_rate_hz,
    )
    return samples, sample_rate


def chunk_audio(
    samples: np.ndarray, *, sample_rate_hz: int, chunk_ms: int | None = None
) -> tuple[list[np.ndarray], int]:
    if chunk_ms is None:
        chunk_ms = int(os.getenv("INTEGRATION_CHUNK_MS", "100"))
    chunk_samples = int(sample_rate_hz * (chunk_ms / 1000.0))
    if chunk_samples <= 0:
        raise ValueError("chunk size must be positive")
    chunks = [samples[i : i + chunk_samples] for i in range(0, len(samples), chunk_samples)]
    return chunks, chunk_samples


def qwen_region_from_env() -> QwenRegion:
    region_raw = os.getenv("QWEN_REGION", QwenRegion.BEIJING.value).lower()
    try:
        return QwenRegion(region_raw)
    except ValueError:
        return QwenRegion.BEIJING


def get_qwen_asr_endpoint() -> str:
    if qwen_region_from_env() is QwenRegion.SINGAPORE:
        default = "wss://dashscope-intl.aliyuncs.com/api-ws/v1/realtime"
    else:
        default = "wss://dashscope.aliyuncs.com/api-ws/v1/realtime"
    return os.getenv("QWEN_ASR_ENDPOINT", default)


def get_qwen_base_url() -> str:
    if qwen_region_from_env() is QwenRegion.SINGAPORE:
        default = "https://dashscope-intl.aliyuncs.com/api/v1"
    else:
        default = "https://dashscope.aliyuncs.com/api/v1"
    return os.getenv("QWEN_BASE_URL", default)


def to_async_qwen_base_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    if normalized.endswith("/compatible-mode/v1"):
        return normalized
    if normalized.endswith("/api/v1"):
        return normalized[: -len("/api/v1")] + "/compatible-mode/v1"
    return normalized + "/compatible-mode/v1"


def get_async_qwen_base_url() -> str:
    return to_async_qwen_base_url(get_qwen_base_url())


def get_local_qwen_stt_model_dir() -> Path:
    model_dir = os.getenv("LOCAL_QWEN_STT_MODEL_DIR")
    if model_dir:
        return Path(model_dir)
    from puripuly_heart.core.local_stt_assets import default_local_stt_model_dir

    return default_local_stt_model_dir()


def require_local_qwen_model_assets(model_dir: str | Path | None = None) -> Path:
    resolved = Path(model_dir) if model_dir is not None else get_local_qwen_stt_model_dir()
    from puripuly_heart.core.local_stt_assets import (
        LocalSTTManifestInvalidError,
        LocalSTTModelMissingError,
        validate_local_stt_runtime_ready,
    )

    try:
        validate_local_stt_runtime_ready(resolved)
    except (LocalSTTManifestInvalidError, LocalSTTModelMissingError):
        pytest.skip("local Qwen STT model assets are unavailable")
    return resolved


def skip_if_local_qwen_runtime_unavailable(exc: BaseException) -> None:
    from puripuly_heart.core.local_stt_assets import LocalQwenSherpaLoadError

    if isinstance(exc, LocalQwenSherpaLoadError):
        pytest.skip("local Qwen Sherpa runtime is unavailable")
    raise exc


def _is_loopback_host(host: str | None) -> bool:
    if not host:
        return False
    normalized = host.strip("[]").lower()
    if normalized == "localhost":
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


def require_local_llm_loopback(
    base_url: str,
    *,
    allow_remote_env: str = "LOCAL_LLM_ALLOW_REMOTE",
) -> None:
    if _is_loopback_host(urlsplit(base_url).hostname):
        return
    if os.getenv(allow_remote_env) == "1":
        return
    pytest.skip(f"non-loopback local LLM endpoint; set {allow_remote_env}=1 to opt in")


async def require_local_llm_server(
    *,
    base_url: str,
    model: str,
    api_key: str = "",
    extra_body: Mapping[str, object] | None = None,
    verify_connection: Callable[..., Awaitable[bool] | bool] | None = None,
) -> None:
    require_local_llm_loopback(base_url)
    if verify_connection is None:
        from puripuly_heart.providers.llm.local_openai import LocalOpenAICompatibleLLMProvider

        verify_connection = LocalOpenAICompatibleLLMProvider.verify_connection
    result = verify_connection(
        base_url=base_url,
        model=model,
        api_key=api_key,
        extra_body=extra_body,
    )
    if inspect.isawaitable(result):
        result = await result
    if not result:
        pytest.skip("local LLM server is unavailable")


@dataclass(slots=True)
class MockOscSender:
    messages: list[str] = field(default_factory=list)
    typing_states: list[bool] = field(default_factory=list)

    def send_chatbox(self, text: str) -> None:
        self.messages.append(text)

    def send_typing(self, is_typing: bool) -> None:
        self.typing_states.append(is_typing)


class SimpleClock:
    def now(self) -> float:
        return time.time()


async def next_ui_event(queue: asyncio.Queue, *, timeout_s: float | None = None):
    timeout = EVENT_POLL_TIMEOUT_S if timeout_s is None else timeout_s
    try:
        return await asyncio.wait_for(queue.get(), timeout=timeout)
    except asyncio.TimeoutError:
        return None


async def wait_for_event(event: asyncio.Event, *, timeout_s: float | None = None) -> bool:
    timeout = RESULT_TIMEOUT_S if timeout_s is None else timeout_s
    try:
        await asyncio.wait_for(event.wait(), timeout=timeout)
        return True
    except asyncio.TimeoutError:
        return False


async def send_vad_events(
    owner,
    utterance_id,
    chunks: list[np.ndarray],
    *,
    chunk_delay_s: float | None = None,
) -> None:
    if not chunks:
        return
    from puripuly_heart.core.vad.gating import SpeechChunk, SpeechEnd, SpeechStart

    delay = CHUNK_DELAY_S if chunk_delay_s is None else chunk_delay_s
    pre_roll = np.zeros(len(chunks[0]), dtype=np.float32)
    await owner.handle_vad_event(SpeechStart(utterance_id, pre_roll=pre_roll, chunk=chunks[0]))
    await asyncio.sleep(delay)
    for chunk in chunks[1:]:
        await owner.handle_vad_event(SpeechChunk(utterance_id, chunk=chunk))
        await asyncio.sleep(delay)
    await owner.handle_vad_event(SpeechEnd(utterance_id))


async def stream_silence(
    session, *, frames: int = 10, frame_bytes: int = 1024, delay_s: float = 0.032
) -> None:
    silence = b"\0" * frame_bytes
    for _ in range(frames):
        await session.send_audio(silence)
        await asyncio.sleep(delay_s)


async def drain_and_close(
    session, *, drain_timeout_s: float = 30.0, close_timeout_s: float = 5.0
) -> None:
    async def _drain():
        async for _ in session.events():
            pass

    try:
        await asyncio.wait_for(_drain(), timeout=drain_timeout_s)
    finally:
        await asyncio.wait_for(session.close(), timeout=close_timeout_s)


async def open_session(backend, *, timeout_s: float | None = None):
    timeout = OPEN_SESSION_TIMEOUT_S if timeout_s is None else timeout_s
    return await asyncio.wait_for(backend.open_session(), timeout=timeout)
