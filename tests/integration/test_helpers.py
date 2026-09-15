from __future__ import annotations

import asyncio
import wave
from pathlib import Path
from uuid import UUID

import pytest

from puripuly_heart.domain.models import Translation
from tests.integration import helpers

pytestmark = helpers.integration_mark()


def _write_silent_wav(path: Path, *, sample_rate_hz: int) -> None:
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(sample_rate_hz)
        audio.writeframes(b"\0\0" * (sample_rate_hz // 10))


class _SuccessfulProvider:
    def __init__(self) -> None:
        self.closed = False
        self.translate_kwargs: dict[str, object] | None = None

    async def translate(self, **kwargs: object) -> Translation:
        self.translate_kwargs = dict(kwargs)
        utterance_id = kwargs["utterance_id"]
        assert isinstance(utterance_id, UUID)
        return Translation(utterance_id=utterance_id, text=" hello ")

    async def close(self) -> None:
        self.closed = True


class _EmptyProvider:
    def __init__(self) -> None:
        self.closed = False

    async def translate(self, **kwargs: object) -> Translation:
        utterance_id = kwargs["utterance_id"]
        assert isinstance(utterance_id, UUID)
        return Translation(utterance_id=utterance_id, text="   ")

    async def close(self) -> None:
        self.closed = True


class _TimeoutDrainSession:
    def __init__(self) -> None:
        self.closed = False

    async def events(self):
        await asyncio.sleep(60)
        if False:
            yield None

    async def close(self) -> None:
        self.closed = True


class _FailingDrainSession:
    def __init__(self) -> None:
        self.closed = False

    async def events(self):
        if False:
            yield None
        raise RuntimeError("drain failed")

    async def close(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_run_llm_smoke_uses_standard_input_asserts_output_and_closes() -> None:
    provider = _SuccessfulProvider()

    translation = await helpers.run_llm_smoke(provider)

    assert translation.text == " hello "
    assert provider.closed is True
    assert provider.translate_kwargs is not None
    assert provider.translate_kwargs["text"] == "안녕하세요"
    assert provider.translate_kwargs["source_language"] == "ko"
    assert provider.translate_kwargs["target_language"] == "en"
    assert provider.translate_kwargs["system_prompt"] == (
        "Translate from ${sourceName} to ${targetName}."
    )
    assert provider.translate_kwargs["context"] == ""


@pytest.mark.asyncio
async def test_run_llm_smoke_closes_provider_when_output_is_empty() -> None:
    provider = _EmptyProvider()

    with pytest.raises(AssertionError):
        await helpers.run_llm_smoke(provider)

    assert provider.closed is True


def test_suppressed_runtime_logger_discards_messages_without_rendering_lazy_text() -> None:
    rendered = False
    logger = helpers.suppressed_runtime_logger()

    def _mark_rendered_and_return_secret() -> str:
        nonlocal rendered
        rendered = True
        return "secret lazy text"

    logger.emit_basic("secret request text")
    logger.emit_diagnostic("secret response text")
    logger.emit_diagnostic_lazy(lambda: _mark_rendered_and_return_secret())

    assert rendered is False
    assert logger.emitted_count == 3
    assert "secret" not in repr(logger)


def test_suppressed_runtime_logger_detailed_methods_return_false() -> None:
    rendered = False
    logger = helpers.suppressed_runtime_logger()

    def _mark_rendered_and_return_secret() -> str:
        nonlocal rendered
        rendered = True
        return "secret lazy text"

    assert logger.emit_diagnostic("secret response text") is False
    assert logger.emit_diagnostic_lazy(lambda: _mark_rendered_and_return_secret()) is False

    assert rendered is False


@pytest.mark.asyncio
async def test_drain_and_close_closes_session_when_drain_times_out() -> None:
    session = _TimeoutDrainSession()

    with pytest.raises(asyncio.TimeoutError):
        await helpers.drain_and_close(session, drain_timeout_s=0.001, close_timeout_s=0.1)

    assert session.closed is True


@pytest.mark.asyncio
async def test_drain_and_close_closes_session_when_drain_raises() -> None:
    session = _FailingDrainSession()

    with pytest.raises(RuntimeError, match="drain failed"):
        await helpers.drain_and_close(session, drain_timeout_s=0.1, close_timeout_s=0.1)

    assert session.closed is True


def test_require_optional_module_skips_missing_local_runtime_dependency() -> None:
    with pytest.raises(pytest.skip.Exception) as excinfo:
        helpers.require_optional_module(
            "puripuly_heart_missing_optional_dependency_for_test",
            reason="local runtime dependency is unavailable",
        )

    assert "local runtime dependency is unavailable" in str(excinfo.value)


def test_require_local_qwen_model_assets_skips_missing_model_dir(tmp_path: Path) -> None:
    missing_model_dir = tmp_path / "missing-model"

    with pytest.raises(pytest.skip.Exception) as excinfo:
        helpers.require_local_qwen_model_assets(missing_model_dir)

    reason = str(excinfo.value)
    assert "local Qwen STT model assets are unavailable" in reason
    assert str(missing_model_dir) not in reason


def test_skip_if_local_qwen_runtime_unavailable_converts_load_error_to_skip() -> None:
    from puripuly_heart.core.local_stt_assets import LocalQwenSherpaLoadError

    with pytest.raises(pytest.skip.Exception) as excinfo:
        helpers.skip_if_local_qwen_runtime_unavailable(
            LocalQwenSherpaLoadError("raw backend bootstrap failure")
        )

    reason = str(excinfo.value)
    assert "local Qwen Sherpa runtime is unavailable" in reason
    assert "raw backend bootstrap failure" not in reason


def test_load_required_audio_wav_skips_missing_audio(tmp_path: Path) -> None:
    missing_audio = tmp_path / "missing.wav"

    with pytest.raises(pytest.skip.Exception) as excinfo:
        helpers.load_required_audio_wav(missing_audio)

    assert "test audio is unavailable" in str(excinfo.value)


def test_load_required_audio_wav_skips_unsupported_sample_rate(tmp_path: Path) -> None:
    audio_path = tmp_path / "speech.wav"
    _write_silent_wav(audio_path, sample_rate_hz=8000)

    with pytest.raises(pytest.skip.Exception) as excinfo:
        helpers.load_required_audio_wav(audio_path, expected_sample_rate_hz=16000)

    assert "unsupported audio sample rate" in str(excinfo.value)


@pytest.mark.asyncio
async def test_require_local_llm_server_skips_when_connection_probe_fails() -> None:
    seen_kwargs: dict[str, object] = {}

    async def verify_connection(**kwargs: object) -> bool:
        seen_kwargs.update(kwargs)
        return False

    with pytest.raises(pytest.skip.Exception) as excinfo:
        await helpers.require_local_llm_server(
            base_url="http://127.0.0.1:11434/v1",
            model="llama3.1:8b",
            verify_connection=verify_connection,
        )

    assert "local LLM server is unavailable" in str(excinfo.value)
    assert seen_kwargs["base_url"] == "http://127.0.0.1:11434/v1"
    assert seen_kwargs["model"] == "llama3.1:8b"


def test_require_local_llm_loopback_skips_remote_url_without_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LOCAL_LLM_ALLOW_REMOTE_FOR_TEST", raising=False)

    with pytest.raises(pytest.skip.Exception) as excinfo:
        helpers.require_local_llm_loopback(
            "https://example.invalid/v1",
            allow_remote_env="LOCAL_LLM_ALLOW_REMOTE_FOR_TEST",
        )

    assert "non-loopback local LLM endpoint" in str(excinfo.value)


@pytest.mark.parametrize(
    ("base_url", "expected"),
    [
        (
            "https://dashscope.aliyuncs.com/api/v1",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        ),
        (
            "https://dashscope-intl.aliyuncs.com/api/v1/",
            "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        ),
        (
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        ),
        (
            "https://example.invalid/custom",
            "https://example.invalid/custom/compatible-mode/v1",
        ),
    ],
)
def test_to_async_qwen_base_url_converts_sync_dashscope_urls(
    base_url: str,
    expected: str,
) -> None:
    assert helpers.to_async_qwen_base_url(base_url) == expected
