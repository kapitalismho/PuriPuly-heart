"""Live Soniox-to-overlay production-path smoke for issue #178.

Routes deterministic synthetic speech (Windows SAPI-generated local WAVs, no
microphone, no private content) through the REAL production ingress and
presentation path:

  WAV -> live scoped Soniox session (begin_turn/send_turn_audio/seal_turn,
  turn_events terminal) -> PeerTranslationChannelOwner.handle_provider_turn_terminal
  -> translation turns/output projection -> OverlayPresenter

Construction mirrors tests/integration/test_speaker_transition_pipeline.py;
the only substitution is live scoped terminal events for synthetic
_SonioxSession JSON tokens. The translation backend is a deterministic echo
provider (explicitly a stand-in); all speaker-evidence plumbing is real.

Secrets: the keyring soniox_api_key is read in-memory only, never logged,
never persisted. Persisted artifacts contain only generated test text and
sanitized timing metadata (speaker ids, source ms bounds, confidence,
overlap flags, scope equality) -- no keys, no audio.

Usage:
  uv run python experiments/overlay_speaker_transition/runtime_validation/\
live_soniox_production_path_smoke.py [--audio-dir C:/t/ovr178/audio]
  [--out experiments/overlay_speaker_transition/runtime_validation/\
live_production_path_result.json]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
import tempfile
import wave
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from puripuly_heart.app.wiring.wiring_secrets_factory import create_secret_store
from puripuly_heart.config.paths import default_settings_path
from puripuly_heart.config.settings_vnext.facade import load_vnext_settings
from puripuly_heart.core.audio.format import AudioCaptureSpan, resample_f32_linear
from puripuly_heart.core.audio.ownership import (
    AudioSegmentIdentity,
    AudioSegmentSettingsSnapshot,
    AudioSegmentSnapshot,
    AudioSegmentTerminalReceipt,
)
from puripuly_heart.core.clock import FakeClock
from puripuly_heart.core.overlay.presenter import OverlayPresenter
from puripuly_heart.core.stt.backend import (
    STTProviderTurnIdentity,
    STTProviderTurnRequest,
    STTProviderTurnTerminal,
    STTSessionProjection,
)
from puripuly_heart.domain.models import Translation
from puripuly_heart.providers.stt.soniox import _SonioxSession
from puripuly_heart.ui.overlay_calibration import OverlayCalibration
from tests.helpers.fakes import RecordingOscQueue
from tests.helpers.translation_owners import compose_translation_test_harness

SAPI_SPEAKERS = (
    ("Microsoft David Desktop", "The overlay shows speaker changes clearly.", "voice1_david.wav"),
    ("Microsoft Zira Desktop", "A second voice is now speaking today.", "voice2_zira.wav"),
)

_SAPI_PS = """Add-Type -AssemblyName System.Speech
$outDir = "{outdir}"
New-Item -ItemType Directory -Force -Path $outDir | Out-Null
{commands}
Get-ChildItem $outDir | Format-Table Name, Length -AutoSize
"""

_SAPI_ONE = (
    "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
    '$s.SelectVoice("{voice}"); $s.SetOutputToWaveFile("{f}"); '
    '$s.Speak("{text}"); $s.Dispose()'
)


def ensure_synthetic_audio(audio_dir: Path) -> list[Path]:
    missing = [audio_dir / name for _, _, name in SAPI_SPEAKERS if not (audio_dir / name).is_file()]
    if not missing:
        return [audio_dir / name for _, _, name in SAPI_SPEAKERS]
    audio_dir.mkdir(parents=True, exist_ok=True)
    commands = "\n".join(
        _SAPI_ONE.format(voice=voice, f=str(audio_dir / name), text=text)
        for voice, text, name in SAPI_SPEAKERS
    )
    with tempfile.NamedTemporaryFile("w", suffix=".ps1", delete=False) as tmp:
        tmp.write(_SAPI_PS.format(outdir=str(audio_dir), commands=commands))
        script = tmp.name
    proc = subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", script],
        capture_output=True,
        text=True,
        timeout=120,
    )
    Path(script).unlink(missing_ok=True)
    if proc.returncode != 0:
        raise RuntimeError(f"SAPI generation failed: {proc.stderr[:400]}")
    paths = [audio_dir / name for _, _, name in SAPI_SPEAKERS]
    still_missing = [str(p) for p in paths if not p.is_file()]
    if still_missing:
        raise RuntimeError(f"SAPI files still missing: {still_missing}")
    return paths


def load_wav_16k_mono(path: Path) -> bytes:
    with wave.open(str(path), "rb") as w:
        nch, sw, fr, nf = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
        raw = w.readframes(nf)
    if sw != 2:
        raise ValueError(f"unsupported sample width {sw} in {path}")
    arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    arr = arr.reshape(-1, nch).mean(axis=1).astype(np.float32)
    if fr != 16000:
        arr = resample_f32_linear(arr, from_rate_hz=fr, to_rate_hz=16000)
    return (np.clip(arr, -1.0, 1.0) * 32767.0).round().astype("<i2").tobytes()


def _span(callback_sequence: int, start: int, end: int) -> AudioCaptureSpan:
    return AudioCaptureSpan(
        capture_epoch=1,
        callback_sequence=callback_sequence,
        source_sample_rate_hz=16000,
        source_start_sample=start,
        source_end_sample=end,
        source_start_monotonic_s=start / 16000.0,
        source_end_monotonic_s=end / 16000.0,
        normalized_sample_rate_hz=16000,
        normalized_start_sample=start,
        normalized_end_sample=end,
    )


def _settings() -> AudioSegmentSettingsSnapshot:
    return AudioSegmentSettingsSnapshot(
        provider_id="soniox",
        provider_signature=("soniox",),
        runtime_signature=("soniox",),
        source_mode="desktop",
        source_language="en",
        expected_languages=("en",),
        target_sample_rate_hz=16000,
        vad_speech_threshold=0.4,
        vad_hangover_ms=800,
        vad_pre_roll_ms=500,
    )


def _load_live_key() -> str:
    config_path = default_settings_path()
    result = load_vnext_settings(config_path)
    if result.settings is None:
        raise RuntimeError("settings load failed")
    store = create_secret_store(result.settings.intent.secrets, config_path=config_path)
    key = store.get("soniox_api_key") or ""
    if not key:
        raise RuntimeError("soniox_api_key absent from configured SecretStore")
    return key


async def live_scoped_terminals(api_key: str, wav_paths: list[Path]) -> tuple[list[dict], str]:
    """Run one live scoped session; return sanitized terminal records + scope head."""
    from puripuly_heart.core.stt.backend import STTProviderEpochEnded

    epoch = "live-prod-smoke"
    session = _SonioxSession(
        api_key=api_key,
        model="stt-rt-v5",
        endpoint="wss://stt-rt.soniox.com/transcribe-websocket",
        sample_rate_hz=16000,
        language_hints=["en"],
        context_terms=[],
        keepalive_interval_s=10.0,
        trailing_silence_ms=100,
        connect_timeout_s=10.0,
        enable_language_identification=False,
        enable_speaker_diarization=True,
        projection=STTSessionProjection(mode="scoped", provider_epoch_id=epoch),
    )
    await session.start()
    scope = session.speaker_session_scope
    records: list[dict] = []
    try:
        for order, wav in enumerate(wav_paths, start=1):
            pcm = load_wav_16k_mono(wav)
            segment_id = uuid4()
            segment = AudioSegmentIdentity(1, order, segment_id, 1)
            identity = STTProviderTurnIdentity(
                segment=segment,
                provider_epoch_id=epoch,
                provider_turn_id=f"turn-{order}",
            )
            await session.begin_turn(
                STTProviderTurnRequest(identity=identity, settings=_settings(), channel="peer")
            )
            chunk_samples = 16000 * 320 // 1000
            chunk_bytes = chunk_samples * 2
            sequence = 0
            for offset in range(0, len(pcm), chunk_bytes):
                chunk = pcm[offset : offset + chunk_bytes]
                samples = len(chunk) // 2
                sequence += 1
                await session.send_turn_audio(
                    identity,
                    chunk,
                    payload_sequence=sequence,
                    source_ranges=(
                        _span(offset // chunk_bytes, offset // 2, offset // 2 + samples),
                    ),
                    context_only=False,
                )
            await session.seal_turn(
                identity,
                sealed_content_ranges=(_span(9000 + order, 0, len(pcm) // 2),),
                seal_reason="silence",
                observed_trailing_silence_ms=500,
            )
            terminal: STTProviderTurnTerminal | None = None
            async for event in session.turn_events():
                if isinstance(event, STTProviderTurnTerminal) and event.identity == identity:
                    terminal = event
                    break
                if isinstance(event, STTProviderEpochEnded):
                    raise RuntimeError(f"epoch ended mid-turn: {event.reason}")
            if terminal is None:
                raise RuntimeError(f"no live terminal for turn {order}")
            records.append(
                {
                    "order": order,
                    "segment_id": str(segment_id),
                    "text": terminal.text,
                    "outcome": terminal.outcome,
                    "text_authority": terminal.text_authority,
                    "epoch_disposition": terminal.epoch_disposition,
                    "language_runs": [
                        {"text": run.text, "language": run.language}
                        for run in terminal.final_language_runs
                    ],
                    "speaker_runs": [
                        {
                            "text": run.text,
                            "speaker_id": run.speaker_id,
                            "source_start_ms": run.source_start_ms,
                            "source_end_ms": run.source_end_ms,
                            "speaker_confidence": run.speaker_confidence,
                            "overlaps_previous": run.overlaps_previous,
                        }
                        for run in terminal.final_speaker_runs
                    ],
                    "session_scope_head": scope[:8],
                }
            )
    finally:
        await session.close()
    return records, scope


@dataclass(slots=True)
class _EchoTranslationProvider:
    """Deterministic translation stand-in; speaker plumbing stays real."""

    calls: list[str] = field(default_factory=list)

    async def translate(
        self,
        *,
        utterance_id,
        text: str,
        system_prompt: str,
        source_language: str,
        target_language: str,
        context: str = "",
        scene_participant_count: int | None = None,
    ) -> Translation:
        _ = (system_prompt, source_language, target_language, context, scene_participant_count)
        self.calls.append(text)
        return Translation(
            utterance_id=utterance_id,
            text=f"translated:{text}",
            source_text=text,
            source_language=source_language,
            target_language=target_language,
            channel="peer",
        )

    async def close(self) -> None:
        return None


@dataclass(slots=True)
class _RecordingOverlaySink:
    presenter: OverlayPresenter
    events: list[object] = field(default_factory=list)

    async def emit(self, event: object) -> None:
        self.events.append(event)
        await self.presenter.emit(event)  # type: ignore[arg-type]


def _terminal_from_live(record: dict, *, generation: int) -> tuple[object, STTProviderTurnTerminal]:
    from puripuly_heart.domain.models import FinalLanguageRun, FinalSpeakerRun

    segment_id = uuid4()
    segment_identity = AudioSegmentIdentity(generation, record["order"], segment_id, 1)
    provider_identity = STTProviderTurnIdentity(
        segment=segment_identity,
        provider_epoch_id=f"epoch-{generation}",
        provider_turn_id=f"turn-{generation}-{record['order']}",
    )
    snapshot = AudioSegmentSnapshot(
        identity=segment_identity,
        settings=_settings(),
        content_ranges=(),
        context_ranges=(),
        failed_ranges=(),
        content_sample_count=0,
        context_sample_count=0,
        failed_normalized_sample_count=0,
        failed_source_sample_count=0,
        prefix_context_sample_count=0,
        synthetic_context_sample_count=0,
        genuine_onset=True,
        state="terminal",
        opened_at_monotonic_s=0.0,
        sealed_at_monotonic_s=0.0,
        seal_reason="silence",
    )
    receipt = AudioSegmentTerminalReceipt(
        identity=segment_identity,
        outcome="final",
        segment=snapshot,
        terminal_at_monotonic_s=0.0,
        provider_epoch_id=provider_identity.provider_epoch_id,
        provider_turn_id=provider_identity.provider_turn_id,
        text_authority="authoritative",
    )
    terminal = STTProviderTurnTerminal(
        identity=provider_identity,
        outcome="final",
        text=record["text"],
        final_language_runs=tuple(
            FinalLanguageRun(run["text"], run["language"]) for run in record["language_runs"]
        ),
        final_speaker_runs=tuple(
            FinalSpeakerRun(
                run["text"],
                run["speaker_id"],
                "live-session",
                source_start_ms=run["source_start_ms"],
                source_end_ms=run["source_end_ms"],
                speaker_confidence=run["speaker_confidence"],
                overlaps_previous=run["overlaps_previous"],
            )
            for run in record["speaker_runs"]
        ),
        text_authority="authoritative",
    )
    return receipt, terminal


async def run_policy(records: list[dict]) -> dict:
    clock = FakeClock(_now=100.0)
    presenter = OverlayPresenter(calibration=OverlayCalibration(), clock=clock)
    provider = _EchoTranslationProvider()
    overlay = _RecordingOverlaySink(presenter)
    harness = compose_translation_test_harness(
        stt=None,
        llm=provider,
        osc=RecordingOscQueue(),
        peer_translation_enabled=True,
        overlay_sink=overlay,
        clock=clock,
    )
    harness.output_runtime.activate_peer_generation(1)
    await harness.start()
    try:
        for record in records:
            receipt, terminal = _terminal_from_live(record, generation=1)
            harness.record_peer_speech_end_for_test(receipt.identity.segment_id)
            await harness.peer_owner.handle_provider_turn_terminal(receipt, terminal)
            await harness.peer_owner.translation_turns.wait_for_idle()
            await harness.output_runtime.wait_for_peer_output_idle()
        translations = [
            event for event in overlay.events if getattr(event, "type", None) == "translation_final"
        ]
        return {
            "translation_calls": list(provider.calls),
            "claims": [
                {
                    "source_text": event.source_text,
                    "speaker_transition": event.speaker_transition,
                    "claim_id_tail": (
                        str(event.speaker_transition_claim_id)[-8:]
                        if event.speaker_transition_claim_id
                        else None
                    ),
                }
                for event in translations
            ],
            "blocks": [
                {
                    "id_tail": block.id[-8:],
                    "channel": block.channel,
                    "speaker_style": block.speaker_style,
                    "primary_text": block.primary_text,
                }
                for block in presenter.snapshot().blocks
            ],
            "revision": presenter.snapshot().revision,
        }
    finally:
        await harness.stop()
        await presenter.close()


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-dir", type=Path, default=Path("C:/t/ovr178/audio"))
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT
        / "experiments/overlay_speaker_transition/runtime_validation"
        / "live_production_path_result.json",
    )
    args = parser.parse_args()

    wav_paths = ensure_synthetic_audio(args.audio_dir)
    audio_meta = []
    for wav in wav_paths:
        with wave.open(str(wav), "rb") as w:
            audio_meta.append(
                {
                    "file": wav.name,
                    "channels": w.getnchannels(),
                    "width": w.getsampwidth(),
                    "rate": w.getframerate(),
                    "seconds": round(w.getnframes() / w.getframerate(), 2),
                }
            )

    api_key = _load_live_key()
    key_present = bool(api_key)
    records, scope = await live_scoped_terminals(api_key, wav_paths)
    del api_key

    policy_result = await run_policy(records)

    result = {
        "method": "live scoped Soniox ingress -> PeerTranslationChannelOwner."
        "handle_provider_turn_terminal -> translation turns/output projection "
        "-> OverlayPresenter; deterministic echo translation stand-in, real "
        "speaker plumbing",
        "key": "keyring soniox_api_key in-memory only; present="
        + str(key_present)
        + "; value never logged or persisted",
        "audio": audio_meta,
        "live_terminals": records,
        "single_session_scope": len({record["session_scope_head"] for record in records}) == 1,
        "policy": "temporary_turn_emphasis",
        "presentation": policy_result,
    }
    args.out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps(policy_result["blocks"], indent=1))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
