from __future__ import annotations

import asyncio
import contextlib
import io
import json
import wave
from collections.abc import Callable, Mapping
from typing import Any

import httpx

from puripuly_heart.core.stt.custom import (
    CUSTOM_STT_MODE_OFFLINE,
    CUSTOM_STT_VALIDATION_AUTH_FAILURE,
    CUSTOM_STT_VALIDATION_COMPATIBILITY_MISMATCH,
    CUSTOM_STT_VALIDATION_MODEL_UNAVAILABLE,
    CUSTOM_STT_VALIDATION_READY,
    CUSTOM_STT_VALIDATION_TRANSCRIPTION_UNVERIFIED,
    CUSTOM_STT_VALIDATION_UNREACHABLE,
    CustomSTTConfigurationError,
    CustomSTTConnectionValidation,
    append_custom_stt_query,
    classify_http_failure,
    language_hint_for_source,
    normalize_custom_stt_extra,
    resolve_openai_realtime_url,
    resolve_openai_transcription_url,
    sanitize_custom_stt_text,
    sanitize_endpoint_for_display,
    validate_mode_compatibility,
)

_SILENCE_WAV_DURATION_S = 0.1
_VALIDATION_TIMEOUT = httpx.Timeout(connect=5.0, read=8.0, write=5.0, pool=5.0)
_STREAM_CONNECT_TIMEOUT_S = 5.0
_SESSION_READY_EVENT_TYPES = frozenset({"session.created", "session.updated"})


def pcm16le_to_wav(pcm16le: bytes, *, sample_rate_hz: int) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate_hz)
        handle.writeframes(pcm16le)
    return buffer.getvalue()


def silent_wav_bytes(*, sample_rate_hz: int) -> bytes:
    frame_count = max(1, int(sample_rate_hz * _SILENCE_WAV_DURATION_S))
    return pcm16le_to_wav(b"\x00\x00" * frame_count, sample_rate_hz=sample_rate_hz)


def authorization_headers(api_key: str) -> dict[str, str]:
    secret = api_key.strip()
    if not secret:
        return {}
    return {"Authorization": f"Bearer {secret}"}


def safe_body_excerpt(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return sanitize_custom_stt_text(value[:240])


def extract_transcript_text(payload: object) -> str | None:
    if not isinstance(payload, dict):
        return None
    text = payload.get("text")
    if isinstance(text, str):
        return text
    transcript = payload.get("transcript")
    if isinstance(transcript, str):
        return transcript
    nested = payload.get("transcription")
    if isinstance(nested, dict):
        nested_text = nested.get("text")
        if isinstance(nested_text, str):
            return nested_text
    return None


def parse_realtime_event(raw: object) -> dict[str, Any] | None:
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="ignore")
    if not isinstance(raw, str):
        return None
    try:
        payload = json.loads(raw)
    except ValueError:
        return None
    return payload if isinstance(payload, dict) else None


async def validate_custom_stt_connection(
    *,
    mode: str,
    compatibility: str,
    endpoint: str,
    model: str,
    api_key: str = "",
    source_language: str = "",
    sample_rate_hz: int = 16000,
    extra: Mapping[str, object] | None = None,
    http_client_factory: Callable[..., httpx.AsyncClient] = httpx.AsyncClient,
    websocket_connect: Callable[..., Any] | None = None,
) -> CustomSTTConnectionValidation:
    try:
        validate_mode_compatibility(mode, compatibility)
        normalized_extra = normalize_custom_stt_extra(extra)
        if mode == CUSTOM_STT_MODE_OFFLINE:
            return await _validate_offline_connection(
                endpoint=endpoint,
                model=model,
                api_key=api_key,
                source_language=source_language,
                sample_rate_hz=sample_rate_hz,
                extra=normalized_extra,
                http_client_factory=http_client_factory,
            )
        return await _validate_realtime_connection(
            endpoint=endpoint,
            model=model,
            api_key=api_key,
            source_language=source_language,
            sample_rate_hz=sample_rate_hz,
            extra=normalized_extra,
            websocket_connect=websocket_connect,
        )
    except CustomSTTConfigurationError as exc:
        return CustomSTTConnectionValidation(
            status=CUSTOM_STT_VALIDATION_COMPATIBILITY_MISMATCH,
            message=str(exc),
            endpoint=sanitize_endpoint_for_display(endpoint),
        )


async def _validate_offline_connection(
    *,
    endpoint: str,
    model: str,
    api_key: str,
    source_language: str,
    sample_rate_hz: int,
    extra: Mapping[str, object],
    http_client_factory: Callable[..., httpx.AsyncClient],
) -> CustomSTTConnectionValidation:
    url = append_custom_stt_query(resolve_openai_transcription_url(endpoint), extra)
    display = sanitize_endpoint_for_display(url)
    data: dict[str, str] = {}
    if model and "model" not in extra:
        data["model"] = model
    language = language_hint_for_source(source_language)
    if language and "language" not in extra:
        data["language"] = language
    for key, value in extra.items():
        data[key] = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    try:
        async with http_client_factory(
            timeout=_VALIDATION_TIMEOUT,
            trust_env=False,
            follow_redirects=False,
        ) as client:
            response = await client.post(
                url,
                headers=authorization_headers(api_key),
                data=data,
                files={
                    "file": (
                        "probe.wav",
                        silent_wav_bytes(sample_rate_hz=sample_rate_hz),
                        "audio/wav",
                    )
                },
            )
    except httpx.HTTPError:
        return CustomSTTConnectionValidation(
            status=CUSTOM_STT_VALIDATION_UNREACHABLE,
            message="Custom STT server is unreachable",
            endpoint=display,
        )
    if response.status_code in {401, 403}:
        return CustomSTTConnectionValidation(
            status=CUSTOM_STT_VALIDATION_AUTH_FAILURE,
            message="Custom STT authentication failed",
            endpoint=display,
        )
    excerpt = safe_body_excerpt(response.text)
    if response.status_code >= 400:
        category = classify_http_failure(response.status_code, excerpt)
        message = {
            CUSTOM_STT_VALIDATION_MODEL_UNAVAILABLE: "Custom STT model is unavailable",
            CUSTOM_STT_VALIDATION_COMPATIBILITY_MISMATCH: "Custom STT API compatibility mismatch",
            CUSTOM_STT_VALIDATION_UNREACHABLE: "Custom STT server is unreachable",
        }.get(category, "Server is reachable but transcription could not be verified")
        return CustomSTTConnectionValidation(
            status=category,
            message=message,
            endpoint=display,
        )
    try:
        payload = response.json()
    except ValueError:
        return CustomSTTConnectionValidation(
            status=CUSTOM_STT_VALIDATION_TRANSCRIPTION_UNVERIFIED,
            message="Server is reachable but transcription could not be verified",
            endpoint=display,
        )
    if extract_transcript_text(payload) is None:
        return CustomSTTConnectionValidation(
            status=CUSTOM_STT_VALIDATION_TRANSCRIPTION_UNVERIFIED,
            message="Server is reachable but transcription could not be verified",
            endpoint=display,
        )
    return CustomSTTConnectionValidation(
        status=CUSTOM_STT_VALIDATION_READY,
        message="Custom STT is compatible and ready",
        endpoint=display,
    )


async def _validate_realtime_connection(
    *,
    endpoint: str,
    model: str,
    api_key: str,
    source_language: str,
    sample_rate_hz: int,
    extra: Mapping[str, object],
    websocket_connect: Callable[..., Any] | None,
) -> CustomSTTConnectionValidation:
    _ = sample_rate_hz
    url = append_custom_stt_query(resolve_openai_realtime_url(endpoint), extra)
    if model and "model" not in extra:
        url = append_custom_stt_query(url, {"model": model})
    display = sanitize_endpoint_for_display(url)
    connect = websocket_connect
    if connect is None:
        import websockets

        connect = websockets.connect
    headers = {
        **authorization_headers(api_key),
        "OpenAI-Beta": "realtime=v1",
    }
    try:
        try:
            ws = await asyncio.wait_for(
                connect(
                    url,
                    additional_headers=headers,
                    open_timeout=_STREAM_CONNECT_TIMEOUT_S,
                    ping_interval=None,
                ),
                timeout=_STREAM_CONNECT_TIMEOUT_S,
            )
        except TypeError:
            ws = await asyncio.wait_for(
                connect(
                    url,
                    extra_headers=headers,
                    open_timeout=_STREAM_CONNECT_TIMEOUT_S,
                    ping_interval=None,
                ),
                timeout=_STREAM_CONNECT_TIMEOUT_S,
            )
    except Exception as exc:
        return _validation_from_connect_failure(exc, display=display)
    try:
        language = language_hint_for_source(source_language)
        session: dict[str, object] = {
            "type": "session.update",
            "session": {
                "input_audio_transcription": ({"model": model} if model else {})
                | ({"language": language} if language else {}),
            },
        }
        await ws.send(json.dumps(session))
        raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
    except Exception:
        with contextlib.suppress(Exception):
            await ws.close()
        return CustomSTTConnectionValidation(
            status=CUSTOM_STT_VALIDATION_TRANSCRIPTION_UNVERIFIED,
            message="Server is reachable but transcription could not be verified",
            endpoint=display,
        )
    with contextlib.suppress(Exception):
        await ws.close()
    event = parse_realtime_event(raw)
    if event is None:
        return CustomSTTConnectionValidation(
            status=CUSTOM_STT_VALIDATION_COMPATIBILITY_MISMATCH,
            message="Custom STT API compatibility mismatch",
            endpoint=display,
        )
    event_type = str(event.get("type") or "")
    if event_type == "error" or event.get("error"):
        error = event.get("error")
        message = ""
        if isinstance(error, dict):
            message = str(error.get("message") or error.get("code") or "")
        lowered = message.lower()
        if "auth" in lowered or "unauthorized" in lowered or "api key" in lowered:
            status = CUSTOM_STT_VALIDATION_AUTH_FAILURE
            text = "Custom STT authentication failed"
        elif "model" in lowered:
            status = CUSTOM_STT_VALIDATION_MODEL_UNAVAILABLE
            text = "Custom STT model is unavailable"
        else:
            status = CUSTOM_STT_VALIDATION_COMPATIBILITY_MISMATCH
            text = "Custom STT API compatibility mismatch"
        return CustomSTTConnectionValidation(status=status, message=text, endpoint=display)
    if event_type in _SESSION_READY_EVENT_TYPES:
        return CustomSTTConnectionValidation(
            status=CUSTOM_STT_VALIDATION_TRANSCRIPTION_UNVERIFIED,
            message="Server is reachable but transcription could not be verified",
            endpoint=display,
        )
    return CustomSTTConnectionValidation(
        status=CUSTOM_STT_VALIDATION_TRANSCRIPTION_UNVERIFIED,
        message="Server is reachable but transcription could not be verified",
        endpoint=display,
    )


def _validation_from_connect_failure(
    exc: BaseException,
    *,
    display: str,
) -> CustomSTTConnectionValidation:
    text = sanitize_custom_stt_text(str(exc))
    lowered = text.lower()
    status_code = getattr(exc, "status_code", None)
    if status_code is None:
        status_code = getattr(exc, "status", None)
    if isinstance(status_code, int):
        category = classify_http_failure(status_code, text)
        message = {
            CUSTOM_STT_VALIDATION_AUTH_FAILURE: "Custom STT authentication failed",
            CUSTOM_STT_VALIDATION_MODEL_UNAVAILABLE: "Custom STT model is unavailable",
            CUSTOM_STT_VALIDATION_COMPATIBILITY_MISMATCH: "Custom STT API compatibility mismatch",
        }.get(category)
        if message is not None:
            return CustomSTTConnectionValidation(
                status=category,
                message=message,
                endpoint=display,
            )
    if any(token in lowered for token in ("401", "403", "unauthorized", "forbidden", "api key")):
        return CustomSTTConnectionValidation(
            status=CUSTOM_STT_VALIDATION_AUTH_FAILURE,
            message="Custom STT authentication failed",
            endpoint=display,
        )
    if "404" in lowered or "not found" in lowered:
        return CustomSTTConnectionValidation(
            status=CUSTOM_STT_VALIDATION_COMPATIBILITY_MISMATCH,
            message="Custom STT API compatibility mismatch",
            endpoint=display,
        )
    return CustomSTTConnectionValidation(
        status=CUSTOM_STT_VALIDATION_UNREACHABLE,
        message="Custom STT server is unreachable",
        endpoint=display,
    )
