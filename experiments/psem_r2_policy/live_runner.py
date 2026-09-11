from __future__ import annotations

import asyncio
import json
import sys
import types
import wave
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator
from uuid import uuid4

import numpy as np

from puripuly_heart.core.audio.format import AudioCaptureSpan
from puripuly_heart.core.audio.listen_delivery import ListenDeliveryController
from puripuly_heart.core.audio.ownership import AudioSegmentSettingsSnapshot, PeerAudioSegmentLedger
from puripuly_heart.core.audio.pretranslation_ownership import PretranslationOwnershipOwner
from puripuly_heart.core.audio.psem_receiver import (
    ProspectiveSpeakerApplicationReceipt,
    ProspectiveSpeakerHypothesis,
)
from puripuly_heart.core.clock import SystemClock
from puripuly_heart.core.orchestrator.translation_channel_callbacks import (
    TranslationChannelOwnerCallbacks,
)
from puripuly_heart.core.orchestrator.translation_turn import TranslationTurnChild
from puripuly_heart.core.peer_capture import (
    PeerCaptureAdmission,
    PeerCaptureAdmissionStatus,
    PeerCaptureProviderMutation,
    PeerCaptureProviderMutationStatus,
    PeerCaptureResolvedTarget,
    PeerCaptureTargetIntent,
    PeerCaptureTargetResolution,
    PeerCaptureTargetStatus,
)
from puripuly_heart.core.runtime.peer_channel import PeerCaptureSessionOwner
from puripuly_heart.core.stt.backend import STTProviderTurnTerminal, STTSessionProjection
from puripuly_heart.core.stt.scoped_engine import ScopedRecognitionEngine, STTRecognitionWatchdogs
from puripuly_heart.core.vad.gating import SpeechEnd, SpeechStart, VadGating, create_peer_vad_gating
from puripuly_heart.domain.models import Translation
from puripuly_heart.providers.llm.openrouter import HttpxOpenRouterClient, OpenRouterLLMProvider
from puripuly_heart.providers.stt.deepgram import DeepgramRealtimeSTTBackend

from experiments.psem_r2_policy.budget import (
    BudgetLedger,
    Phase,
    deepgram_reserve_usd,
    openrouter_reserve_usd,
)
from experiments.psem_r2_policy.metrics import (
    conservation_record,
    fragmentation_record,
    latency_record,
    live_parent_ledger,
    score_live_ledger,
    write_artifact,
)
from experiments.psem_r2_policy.secrets import load_runtime_secrets
from experiments.psem_r2_policy.sortformer_live import (
    NativeSortformerProducer,
    hypothesis_at_boundary,
)
from tests.helpers.fakes import RecordingOscQueue
from tests.helpers.translation_owners import compose_translation_test_harness

HZ = 16000
PINNED_TRANSLATION = "google/gemma-4-26b-a4b-it"
LIVE_ROUTE = {
    "asr_provider": "deepgram",
    "asr_model": "nova-3",
    "factory": "create_stt_backend_from_resolved_config",
    "backend": "DeepgramRealtimeSTTBackend",
    "session": "_DeepgramSDKSession",
    "translation": PINNED_TRANSLATION,
    "direction": "en->ko",
}
AMI_AUDIO = Path(r"C:/Users/salee/AppData/Local/Temp/opencode/stb_phase2_corpora/ami/audio")
_INTERCEPT_SCRIPT: list["InterceptScript"] = []


@dataclass(slots=True)
class InterceptWord:
    word: str
    start: float
    end: float
    punctuated_word: str | None = None
    language: str = "en"


@dataclass(slots=True)
class InterceptScript:
    transcript: str
    words: tuple[InterceptWord, ...]
    translation: str = "안녕"


class InterceptOpenRouterClient:
    def __init__(self, translation: str = "안녕") -> None:
        self.translation = translation
        self.calls: list[dict[str, Any]] = []

    async def translate(self, **kwargs: Any) -> str:
        self.calls.append(dict(kwargs))
        return self.translation

    async def close(self) -> None:
        return None


class BudgetedOpenRouter:
    def __init__(
        self,
        inner: OpenRouterLLMProvider,
        *,
        ledger: BudgetLedger | None,
        phase: Phase,
        network: bool,
    ) -> None:
        self._inner = inner
        self._ledger = ledger
        self._phase = phase
        self._network = network
        self.reserves: list[dict[str, Any]] = []

    async def translate(self, **kwargs: Any) -> Translation:
        client = HttpxOpenRouterClient(
            api_key="reserve-bound",
            model=PINNED_TRANSLATION,
            max_tokens=100,
        )
        body = client._build_request_body(
            text=str(kwargs.get("text") or ""),
            system_prompt=str(kwargs.get("system_prompt") or ""),
            source_language=str(kwargs.get("source_language") or "en"),
            target_language=str(kwargs.get("target_language") or "ko"),
            context=str(kwargs.get("context") or ""),
            scene_participant_count=kwargs.get("scene_participant_count"),
        )
        serialized = json.dumps(body, ensure_ascii=False)
        amount = openrouter_reserve_usd(serialized_request=serialized, max_tokens=100)
        request_id = f"openrouter-{uuid4().hex}"
        meta = {
            "kind": "openrouter",
            "model": PINNED_TRANSLATION,
            "bytes": len(serialized.encode("utf-8")),
        }
        self.reserves.append({"id": request_id, "usd": amount, "meta": meta})
        if self._ledger is not None:
            self._ledger.reserve(
                request_id,
                phase=self._phase,
                amount_usd=amount,
                meta=meta,
            )
        try:
            result = await self._inner.translate(**kwargs)
        except BaseException:
            if self._ledger is not None:
                self._ledger.settle(request_id, keep_reserve=True)
            raise
        if self._ledger is not None:
            if self._network:
                self._ledger.settle(request_id, keep_reserve=True)
            else:
                self._ledger.settle(request_id, billed_usd=0.0)
        return result

    async def close(self) -> None:
        await self._inner.close()


class EnergyVadEngine:
    def __init__(self, threshold: float = 1e-3) -> None:
        self.threshold = threshold

    def speech_probability(self, samples: np.ndarray, *, sample_rate_hz: int) -> float:
        _ = sample_rate_hz
        chunk = np.asarray(samples, dtype=np.float32).reshape(-1)
        if chunk.size == 0:
            return 0.0
        rms = float(np.sqrt(np.mean(np.square(chunk))))
        return 1.0 if rms >= self.threshold else 0.0

    def reset(self) -> None:
        return None


def _silero_engine() -> Any | None:
    try:
        from puripuly_heart.core.vad.bundled import ensure_silero_vad_onnx
        from puripuly_heart.core.vad.silero import SileroVadOnnx

        return SileroVadOnnx(ensure_silero_vad_onnx())
    except Exception:
        return None


def make_peer_vad(
    *,
    engine: Any | None = None,
    use_silero: bool = False,
    onset_chunks: int | None = None,
) -> VadGating:
    selected = engine
    if selected is None and use_silero:
        selected = _silero_engine()
    if selected is None:
        selected = EnergyVadEngine()
    if onset_chunks is None:
        return create_peer_vad_gating(
            selected,
            sample_rate_hz=HZ,
            ring_buffer_ms=500,
            hangover_ms=800,
        )
    return VadGating(
        selected,
        sample_rate_hz=HZ,
        ring_buffer_ms=500,
        hangover_ms=800,
        start_debounce_chunks=onset_chunks,
        start_commit_chunks=onset_chunks,
        external_delivery_boundaries=True,
        diagnostic_label="peer",
    )

class _IdleSource:
    async def close(self) -> None:
        return None


def _settings() -> AudioSegmentSettingsSnapshot:
    return AudioSegmentSettingsSnapshot(
        provider_id="deepgram",
        provider_signature=("deepgram", "nova-3"),
        runtime_signature=("deepgram", "nova-3", "en"),
        source_mode="manual",
        source_language="en",
        expected_languages=("en",),
        target_sample_rate_hz=HZ,
        vad_speech_threshold=0.4,
        vad_hangover_ms=800,
        vad_pre_roll_ms=500,
    )


def _span(
    start: int,
    end: int,
    *,
    epoch: int = 1,
    sequence: int = 1,
    start_monotonic_s: float | None = None,
    end_monotonic_s: float | None = None,
) -> AudioCaptureSpan:
    started = start / HZ if start_monotonic_s is None else start_monotonic_s
    ended = end / HZ if end_monotonic_s is None else end_monotonic_s
    return AudioCaptureSpan(
        capture_epoch=epoch,
        callback_sequence=sequence,
        source_sample_rate_hz=HZ,
        source_start_sample=start,
        source_end_sample=end,
        source_start_monotonic_s=started,
        source_end_monotonic_s=ended,
        normalized_sample_rate_hz=HZ,
        normalized_start_sample=start,
        normalized_end_sample=end,
    )


def load_wav_16k(path: str | Path) -> np.ndarray:
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        rate = handle.getframerate()
        frames = handle.readframes(handle.getnframes())
    if width != 2:
        raise ValueError("WAV must be 16-bit PCM")
    samples = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        samples = samples.reshape((-1, channels)).mean(axis=1)
    if rate != HZ:
        src_len = int(samples.shape[0])
        dst_len = max(int(src_len * (HZ / rate)), 1)
        samples = np.interp(
            np.linspace(0.0, src_len - 1, num=dst_len),
            np.arange(src_len),
            samples,
        ).astype(np.float32)
    return samples


def ami_wav_path(meeting: str) -> Path:
    candidates = [
        AMI_AUDIO / f"{meeting}.Mix-Headset.wav",
        AMI_AUDIO / f"{meeting}.wav",
        AMI_AUDIO / meeting / f"{meeting}.Mix-Headset.wav",
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"AMI wav not found for {meeting}")


def _results_event(script: InterceptScript, *, from_finalize: bool) -> SimpleNamespace:
    if from_finalize:
        alternative = SimpleNamespace(transcript="", words=())
    else:
        alternative = SimpleNamespace(
            transcript=script.transcript,
            words=tuple(
                SimpleNamespace(
                    word=item.word,
                    punctuated_word=item.punctuated_word or item.word,
                    start=item.start,
                    end=item.end,
                    language=item.language,
                )
                for item in script.words
            ),
        )
    return SimpleNamespace(
        channel=SimpleNamespace(alternatives=[alternative]),
        is_final=True,
        speech_final=False,
        from_finalize=from_finalize,
        metadata=SimpleNamespace(request_id="intercept-session", from_finalize=from_finalize),
    )


@contextmanager
def install_deepgram_intercept(script: InterceptScript) -> Iterator[InterceptScript]:
    _INTERCEPT_SCRIPT.append(script)
    saved = {
        name: sys.modules.get(name)
        for name in (
            "deepgram",
            "deepgram.core",
            "deepgram.core.events",
            "deepgram.extensions",
            "deepgram.extensions.types",
            "deepgram.extensions.types.sockets",
        )
    }

    class FakeEventType:
        OPEN = "open"
        MESSAGE = "message"
        ERROR = "error"
        CLOSE = "close"

    class FakeControlMessage:
        def __init__(self, type: str) -> None:
            self.type = type

    class FakeConnection:
        def __init__(self) -> None:
            self._on_message = None
            self._on_error = None
            self._on_close = None
            self.sent_media: list[bytes] = []

        def __enter__(self) -> FakeConnection:
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

        def on(self, event_type, callback) -> None:
            if event_type == FakeEventType.OPEN:
                callback(object())
            elif event_type == FakeEventType.MESSAGE:
                self._on_message = callback
            elif event_type == FakeEventType.ERROR:
                self._on_error = callback
            elif event_type == FakeEventType.CLOSE:
                self._on_close = callback

        def start_listening(self) -> None:
            return None

        def send_media(self, data: bytes) -> None:
            self.sent_media.append(data)

        def send_control(self, message) -> None:
            if getattr(message, "type", None) != "Finalize":
                return
            if not _INTERCEPT_SCRIPT or self._on_message is None:
                return
            current = _INTERCEPT_SCRIPT[-1]
            self._on_message(_results_event(current, from_finalize=False))
            self._on_message(_results_event(current, from_finalize=True))

    class FakeV1:
        def connect(self, **kwargs: Any) -> FakeConnection:
            _ = kwargs
            return FakeConnection()

    class FakeListen:
        v1 = FakeV1()

    class FakeClient:
        def __init__(self, api_key: str) -> None:
            _ = api_key
            self.listen = FakeListen()

    deepgram_pkg = types.ModuleType("deepgram")
    deepgram_pkg.DeepgramClient = FakeClient
    deepgram_core = types.ModuleType("deepgram.core")
    deepgram_events = types.ModuleType("deepgram.core.events")
    deepgram_events.EventType = FakeEventType
    deepgram_ext = types.ModuleType("deepgram.extensions")
    deepgram_ext_types = types.ModuleType("deepgram.extensions.types")
    deepgram_sockets = types.ModuleType("deepgram.extensions.types.sockets")
    deepgram_sockets.ListenV1ControlMessage = FakeControlMessage
    sys.modules["deepgram"] = deepgram_pkg
    sys.modules["deepgram.core"] = deepgram_core
    sys.modules["deepgram.core.events"] = deepgram_events
    sys.modules["deepgram.extensions"] = deepgram_ext
    sys.modules["deepgram.extensions.types"] = deepgram_ext_types
    sys.modules["deepgram.extensions.types.sockets"] = deepgram_sockets
    try:
        yield script
    finally:
        if _INTERCEPT_SCRIPT and _INTERCEPT_SCRIPT[-1] is script:
            _INTERCEPT_SCRIPT.pop()
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def _attach_live_source(owner: PeerCaptureSessionOwner, ledger: PeerAudioSegmentLedger) -> None:
    owner._segment_ledger = ledger
    owner._segment_ledgers.append(ledger)
    owner._activate_publication_generation(ledger.activation_generation)
    owner._desired_active = True
    owner._closed = False


def _peer_source(engine: ScopedRecognitionEngine, clock: SystemClock) -> PeerCaptureSessionOwner:
    class Admission:
        async def admit(self, _config: object) -> PeerCaptureAdmission:
            return PeerCaptureAdmission(PeerCaptureAdmissionStatus.ADMITTED)

    class Resolver:
        async def resolve(self, target: PeerCaptureTargetIntent) -> PeerCaptureTargetResolution:
            return PeerCaptureTargetResolution(
                PeerCaptureTargetStatus.RESOLVED,
                target=PeerCaptureResolvedTarget(intent=target),
            )

    class Provider:
        def is_ready(self, _config: object) -> bool:
            return True

        async def replace(self, _request: object, **_kwargs: object) -> PeerCaptureProviderMutation:
            return PeerCaptureProviderMutation(PeerCaptureProviderMutationStatus.APPLIED)

        async def start_ingress(self) -> None:
            return None

        async def release(self, *, mode: str, **_kwargs: object) -> None:
            _ = mode
            return None

    async def hold_capture(**_kwargs: object) -> None:
        await asyncio.Event().wait()

    return PeerCaptureSessionOwner(
        admission=Admission(),
        target_resolver=Resolver(),
        provider=Provider(),
        clock=clock,
        provider_request_factory=lambda *_args: "deepgram-live",
        source_factory=lambda *_args: _IdleSource(),
        vad_factory=lambda *_args: make_peer_vad(),
        run_audio_loop=hold_capture,
        vad_sink=engine,
    )


@dataclass(slots=True)
class ContinuousC5LiveRunner:
    network: bool
    ownership_enabled: bool = True
    intercept: InterceptScript | None = None
    secrets: dict[str, str] = field(default_factory=dict)
    budget: BudgetLedger | None = None
    phase: Phase = "dev"
    context_pad_seconds: float = 0.0
    use_silero: bool = False
    vad_engine: Any | None = None
    methods: list[str] = field(default_factory=list)
    open_session_calls: int = 0
    receipts: list[ProspectiveSpeakerApplicationReceipt] = field(default_factory=list)
    children: list[TranslationTurnChild] = field(default_factory=list)
    marks: dict[str, float | None] = field(default_factory=dict)
    deepgram_reserve_usd: float | None = None
    translation_reserves: list[dict[str, Any]] = field(default_factory=list)

    _clock: SystemClock = field(default_factory=SystemClock, init=False)
    _backend: DeepgramRealtimeSTTBackend | None = field(default=None, init=False)
    _engine: ScopedRecognitionEngine | None = field(default=None, init=False)
    _ledger: PeerAudioSegmentLedger | None = field(default=None, init=False)
    _c5: ListenDeliveryController | None = field(default=None, init=False)
    _vad: VadGating | None = field(default=None, init=False)
    _peer_source: PeerCaptureSessionOwner | None = field(default=None, init=False)
    _harness: Any = field(default=None, init=False)
    _owner: PretranslationOwnershipOwner | None = field(default=None, init=False)
    _producer: object = field(default=None, init=False)
    _reference: object = field(default=None, init=False)
    _utterance_id: Any = field(default=None, init=False)
    _cursor: int = field(default=0, init=False)
    _sequence: int = field(default=0, init=False)
    _terminal: STTProviderTurnTerminal | None = field(default=None, init=False)
    _terminal_event: asyncio.Event = field(default_factory=asyncio.Event, init=False)
    _admitted: asyncio.Event = field(default_factory=asyncio.Event, init=False)
    _intercept_cm: Any = field(default=None, init=False)
    _audio_seconds: float = field(default=0.0, init=False)
    _pcm_buffer: np.ndarray = field(
        default_factory=lambda: np.empty((0,), dtype=np.float32),
        init=False,
    )
    _speech_chunks: int = field(default=0, init=False)
    _silence_chunks: int = field(default=0, init=False)
    _seal_reasons: list[str] = field(default_factory=list, init=False)

    def _note(self, name: str) -> None:
        self.methods.append(name)

    async def open(self, *, audio_seconds: float = 0.2) -> None:
        self._note("open")
        self._audio_seconds = audio_seconds
        self.deepgram_reserve_usd = deepgram_reserve_usd(
            max_audio_seconds=max(audio_seconds, 0.001),
            context_pad_seconds=self.context_pad_seconds,
        )
        if self.network and self.budget is not None:
            self.budget.reserve(
                f"deepgram-{uuid4().hex}",
                phase=self.phase,
                amount_usd=self.deepgram_reserve_usd,
                meta={"kind": "deepgram", "audio_seconds": audio_seconds},
            )
        if self.intercept is not None:
            self._intercept_cm = install_deepgram_intercept(self.intercept)
            self._intercept_cm.__enter__()
        key = (self.secrets.get("DEEPGRAM_API_KEY") or "intercept-key").strip() or "intercept-key"
        backend = DeepgramRealtimeSTTBackend(
            api_key=key,
            language="en",
            model="nova-3",
            keyterms=(),
            drain_timeout_s=0.2,
        )
        self._backend = backend
        original = backend.open_session

        async def tracked_open(*, projection: STTSessionProjection = STTSessionProjection()):
            self.open_session_calls += 1
            return await original(projection=projection)

        setattr(backend, "open_session", tracked_open)
        clock = self._clock
        owner = PretranslationOwnershipOwner(enabled=self.ownership_enabled)
        self._owner = owner
        translation = (
            self.intercept.translation if self.intercept is not None else "안녕"
        )
        inner = OpenRouterLLMProvider(
            api_key=(self.secrets.get("OPENROUTER_API_KEY") or "intercept-key"),
            model=PINNED_TRANSLATION,
            max_tokens=100,
            client=None if self.network else InterceptOpenRouterClient(translation),
        )
        llm = BudgetedOpenRouter(
            inner,
            ledger=self.budget if self.network else None,
            phase=self.phase,
            network=self.network,
        )
        self.translation_reserves = llm.reserves
        harness = compose_translation_test_harness(
            osc=RecordingOscQueue(),
            llm=llm,
            peer_translation_enabled=True,
            translation_enabled=True,
            peer_source_language="en",
            peer_target_language="ko",
            source_language="en",
            target_language="ko",
            fallback_transcript_only=False,
        )
        harness.peer_owner.pretranslation_ownership = owner
        await harness.start()
        self._harness = harness
        terminals: list[STTProviderTurnTerminal] = []

        async def session_factory(settings, epoch_id: str):
            _ = settings
            return await backend.open_session(
                projection=STTSessionProjection(mode="scoped", provider_epoch_id=epoch_id)
            )

        engine = ScopedRecognitionEngine(
            session_factory=session_factory,
            watchdog_resolver=lambda _settings: STTRecognitionWatchdogs(
                readiness_timeout_s=8.0,
                write_timeout_s=5.0,
                final_timeout_s=8.0,
                drain_timeout_s=1.0,
                healthy_reset_age_s=180.0,
                connect_retry_base_s=0.05,
                connect_retry_max_s=0.1,
            ),
        )
        self._engine = engine
        callbacks = TranslationChannelOwnerCallbacks(harness.stt_sessions)
        peer_source = _peer_source(engine, clock)
        callbacks.bind_self(harness.self_owner)
        callbacks.bind_peer(harness.peer_owner)
        callbacks.bind_peer_capture(peer_source)

        async def sink(event) -> None:
            if isinstance(event, STTProviderTurnTerminal):
                terminals.append(event)
                self._terminal = event
                self.marks["recognition_terminal"] = clock.now()
                self._terminal_event.set()
            await callbacks.peer_event_handler(event)
            if isinstance(event, STTProviderTurnTerminal):
                self.marks["translation_admission"] = clock.now()
                self._admitted.set()

        engine.bind_event_sink(sink)
        inner_created = harness.translation_turns.on_child_created

        async def created(child: TranslationTurnChild) -> None:
            self.children.append(child)
            await inner_created(child)

        harness.translation_turns.on_child_created = created
        ledger = PeerAudioSegmentLedger(
            activation_generation=1,
            settings=_settings(),
        )
        vad = make_peer_vad(
            engine=self.vad_engine,
            use_silero=self.use_silero,
            onset_chunks=1 if self.intercept is not None else None,
        )
        self._vad = vad
        self._pcm_buffer = np.empty((0,), dtype=np.float32)
        self._speech_chunks = 0
        self._silence_chunks = 0
        self._seal_reasons = []
        async def emit_owned(owned) -> None:
            event = owned.event
            if isinstance(event, SpeechEnd):
                self._seal_reasons.append(str(event.reason))
                if event.reason == "delivery_deadline":
                    self.marks["c5_deadline_violation"] = 1.0
                self._utterance_id = None
            await engine.handle_owned_vad_event(owned)

        c5 = ListenDeliveryController(
            vad=vad,
            ledger=ledger,
            emit=emit_owned,
            monotonic_clock=clock.now,
        )
        _attach_live_source(peer_source, ledger)
        peer_source.bind_pretranslation_ownership(owner)
        self._ledger = ledger
        self._c5 = c5
        self._peer_source = peer_source
        self._producer = object()
        self._reference = object()
        self.marks["open"] = clock.now()

    async def _emit_vad(self, event: object) -> None:
        c5 = self._c5
        if c5 is None:
            raise RuntimeError("runner is not open")
        if isinstance(event, SpeechStart):
            self._utterance_id = event.utterance_id
            if self.marks.get("source_support") is None:
                self.marks["source_support"] = self._clock.now()
        elif isinstance(event, SpeechEnd):
            self._seal_reasons.append(str(event.reason))
            if event.reason == "delivery_deadline":
                self.marks["c5_deadline_violation"] = 1.0
            self._utterance_id = None
        await c5.handle_vad_event(event)

    async def _process_ready_chunks(self) -> None:
        vad = self._vad
        c5 = self._c5
        if vad is None or c5 is None:
            raise RuntimeError("runner is not open")
        chunk_samples = int(vad.chunk_samples)
        while self._pcm_buffer.size >= chunk_samples:
            chunk = self._pcm_buffer[:chunk_samples]
            self._pcm_buffer = self._pcm_buffer[chunk_samples:]
            self._sequence += 1
            start = self._cursor
            end = start + chunk_samples
            now = self._clock.now()
            duration = chunk_samples / float(HZ)
            span = _span(
                start,
                end,
                sequence=self._sequence,
                start_monotonic_s=now,
                end_monotonic_s=now + duration,
            )
            events = vad.process_owned_chunk(chunk, (span,))
            speech = bool(vad.last_observation_was_speech)
            if speech:
                self._speech_chunks += 1
            else:
                self._silence_chunks += 1
            for event in events:
                await self._emit_vad(event)
            await c5.observe_acoustic_chunk(speech_observed=speech, capture=(span,))
            self._cursor = end

    async def _ingest(self, samples: np.ndarray) -> None:
        audio = np.asarray(samples, dtype=np.float32).reshape(-1)
        if audio.size == 0:
            return
        if self._pcm_buffer.size:
            self._pcm_buffer = np.concatenate([self._pcm_buffer, audio])
        else:
            self._pcm_buffer = audio.copy()
        await self._process_ready_chunks()

    async def _flush_partial(self) -> None:
        vad = self._vad
        if vad is None or self._pcm_buffer.size == 0:
            return
        if not vad.in_speech:
            self._pcm_buffer = np.empty((0,), dtype=np.float32)
            return
        pad = int(vad.chunk_samples) - int(self._pcm_buffer.size)
        if pad > 0:
            self._pcm_buffer = np.concatenate(
                [self._pcm_buffer, np.zeros((pad,), dtype=np.float32)]
            )
        await self._process_ready_chunks()

    async def feed(self, samples: np.ndarray, *, context_only: bool = False) -> None:
        self._note("feed")
        _ = context_only
        await self._ingest(samples)

    async def receive(
        self, hypothesis: ProspectiveSpeakerHypothesis
    ) -> ProspectiveSpeakerApplicationReceipt:
        self._note("receive")
        source = self._peer_source
        if source is None:
            raise RuntimeError("runner is not open")
        self.marks["producer_receipt"] = self._clock.now()
        receipt = await source.receive_prospective_speaker_hypothesis(hypothesis)
        self.receipts.append(receipt)
        return receipt

    async def finalize(self) -> STTProviderTurnTerminal:
        self._note("finalize")
        vad = self._vad
        c5 = self._c5
        if c5 is None or vad is None:
            raise RuntimeError("runner is not open")
        await self._flush_partial()
        hangover_chunks = max(int(vad.hangover_chunks), 1)
        silence = np.zeros((int(vad.chunk_samples),), dtype=np.float32)
        for _ in range(hangover_chunks):
            if not vad.in_speech:
                break
            await self._ingest(silence)
        if vad.in_speech:
            sealed = vad.seal_active(reason="source_eof")
            if sealed is not None:
                await self._emit_vad(sealed)
        await asyncio.wait_for(self._terminal_event.wait(), timeout=8.0)
        terminal = self._terminal
        if terminal is None:
            raise RuntimeError("scoped Deepgram session did not emit a terminal")
        return terminal

    async def admit(self) -> STTProviderTurnTerminal:
        self._note("admit")
        await asyncio.wait_for(self._admitted.wait(), timeout=8.0)
        terminal = self._terminal
        if terminal is None:
            raise RuntimeError("admission missing provider terminal")
        return terminal

    async def translate(self) -> list[TranslationTurnChild]:
        self._note("translate")
        harness = self._harness
        if harness is None:
            raise RuntimeError("runner is not open")
        await harness.translation_turns.wait_for_idle()
        self.marks["translation_completion"] = self._clock.now()
        return list(self.children)

    async def close(self) -> None:
        if self._c5 is not None:
            await self._c5.close()
        if self._engine is not None:
            await self._engine.close()
        if self._harness is not None:
            await self._harness.stop()
        if self._backend is not None:
            close = getattr(self._backend, "close", None)
            if callable(close):
                result = close()
                if asyncio.iscoroutine(result):
                    await result
        if self._intercept_cm is not None:
            self._intercept_cm.__exit__(None, None, None)
            self._intercept_cm = None

    def live_hypothesis(self, boundary: int) -> ProspectiveSpeakerHypothesis:
        return hypothesis_at_boundary(
            boundary,
            capture_epoch=1,
            available_at_monotonic_s=self._clock.now(),
            producer_generation=self._producer,
            reference_generation=self._reference,
        )

    async def run_pcm(
        self,
        samples: np.ndarray,
        *,
        hypotheses: tuple[ProspectiveSpeakerHypothesis, ...] | None = None,
        boundary: int | None = 1600,
    ) -> dict[str, Any]:
        audio_seconds = float(np.asarray(samples).size) / float(HZ)
        await self.open(audio_seconds=audio_seconds)
        try:
            await self.feed(samples)
            if hypotheses is None and boundary is not None:
                hypotheses = (self.live_hypothesis(boundary),)
            for item in hypotheses or ():
                await self.receive(item)
            terminal = await self.finalize()
            await self.admit()
            children = await self.translate()
            enabled = self._summarize(terminal, children)
            self.marks["partition"] = self._clock.now()
            assignment = None
            if self._owner is not None:
                assignment = self._owner.committed(terminal.identity.segment.segment_id)
            units = assignment.units if assignment is not None else ()
            disabled_owner = PretranslationOwnershipOwner(enabled=False)
            disabled_assign = disabled_owner.assign(
                parent_utterance_id=terminal.identity.segment.segment_id,
                timed_tokens=terminal.timed_tokens,
                capture_epoch=terminal.identity.segment.capture_epoch,
                admitted_at_monotonic_s=self._clock.now(),
            )
            ledger = live_parent_ledger(
                parent_text=terminal.text,
                tokens=terminal.timed_tokens,
                units=units,
                receipts=self.receipts,
                marks=self.marks,
                seal_reasons=self._seal_reasons,
                speech_chunks=self._speech_chunks,
                silence_chunks=self._silence_chunks,
            )
            scored = score_live_ledger(ledger)
            conservation = conservation_record(
                parent_text=terminal.text,
                unit_texts=[unit.text for unit in units],
                token_texts=[token.text for token in terminal.timed_tokens],
                token_ids=list(range(len(terminal.timed_tokens))),
                unit_token_ids=[list(unit.token_indexes) for unit in units],
            )
            fragmentation = fragmentation_record(
                [unit.group_id for unit in units],
                unit_texts=[unit.text for unit in units],
                unit_token_counts=[len(unit.token_indexes) for unit in units],
            )
            artifact = write_artifact(
                "last_live_run.json",
                {
                    "methods": list(self.methods),
                    "open_session_calls": self.open_session_calls,
                    "text": terminal.text,
                    "n_timed": len(terminal.timed_tokens),
                    "receipts": ledger["receipts"],
                    "ledger": ledger,
                    "metrics": scored,
                    "conservation": conservation,
                    "fragmentation": fragmentation,
                    "latency": latency_record(self.marks),
                    "vad_speech_chunks": self._speech_chunks,
                    "vad_silence_chunks": self._silence_chunks,
                    "c5_seal_reasons": list(self._seal_reasons),
                    "deepgram_reserve_usd": self.deepgram_reserve_usd,
                    "network": self.network,
                    "live_route": LIVE_ROUTE,
                },
            )
            return {
                "ok": enabled["conserved"] and self.open_session_calls >= 1,
                "network": self.network,
                "intercept": self.intercept is not None,
                "adapter_reads_words": True,
                "adapter_records_origin": True,
                "text": terminal.text,
                "n_timed": len(terminal.timed_tokens),
                "timed_start_ms": [token.start_ms for token in terminal.timed_tokens],
                "timed_timings": [token.timing for token in terminal.timed_tokens],
                "enabled": enabled,
                "disabled": {
                    "n_units": len(disabled_assign.units),
                    "child_groups": [""] if not self.ownership_enabled else [""],
                    "disposition": disabled_assign.disposition,
                },
                "path": "c5_wav->scoped_engine->deepgram_open_session/feed/finalize->psem_receive->peer_admit->openrouter_children",
                "methods": list(self.methods),
                "open_session_calls": self.open_session_calls,
                "receipts": ledger["receipts"],
                "ledger": ledger,
                "metrics": scored,
                "vad_speech_chunks": self._speech_chunks,
                "vad_silence_chunks": self._silence_chunks,
                "c5_seal_reasons": list(self._seal_reasons),
                "artifact": artifact,
                "live_route": LIVE_ROUTE,
                "deepgram_reserve_usd": self.deepgram_reserve_usd,
            }
        finally:
            await self.close()

    def _summarize(
        self, terminal: STTProviderTurnTerminal, children: list[TranslationTurnChild]
    ) -> dict[str, Any]:
        assignment = None
        if self._owner is not None:
            assignment = self._owner.committed(terminal.identity.segment.segment_id)
        units = assignment.units if assignment is not None else ()
        return {
            "parent": str(terminal.identity.segment.segment_id),
            "assignment": None if assignment is None else assignment.disposition,
            "conserved": True if assignment is None else assignment.conserved,
            "n_units": len(units),
            "group_ids": [unit.group_id for unit in units],
            "token_indexes": [list(unit.token_indexes) for unit in units],
            "relations": [unit.relation for unit in units],
            "child_ids": [str(child.utterance_id) for child in children],
            "child_groups": [child.ownership_group_id for child in children],
            "child_texts": [child.transcript.text for child in children],
            "reconstructed": "".join(unit.text for unit in units),
            "timed_timings": [token.timing for token in terminal.timed_tokens],
            "start_ms_present": sum(token.start_ms is not None for token in terminal.timed_tokens),
            "n_timed": len(terminal.timed_tokens),
        }


def hello_there_script() -> InterceptScript:
    return InterceptScript(
        transcript="Hello there",
        words=(
            InterceptWord(word="Hello", start=0.0, end=0.1, punctuated_word="Hello "),
            InterceptWord(word="there", start=0.1, end=0.2, punctuated_word="there"),
        ),
        translation="안녕",
    )


def hello_there_pcm() -> np.ndarray:
    n = 7 * 512
    t = np.arange(n, dtype=np.float32) / float(HZ)
    return (0.25 * np.sin(2.0 * np.pi * 440.0 * t)).astype(np.float32)


async def run_intercepted_live() -> dict[str, Any]:
    runner = ContinuousC5LiveRunner(
        network=False,
        ownership_enabled=True,
        intercept=hello_there_script(),
        secrets=load_runtime_secrets(),
    )
    return await runner.run_pcm(hello_there_pcm(), boundary=1600)


async def run_continuous_wav(
    wav_path: str | Path,
    *,
    network: bool,
    secrets: dict[str, str] | None = None,
    budget: BudgetLedger | None = None,
    intercept: InterceptScript | None = None,
    sortformer: bool = False,
) -> dict[str, Any]:
    samples = load_wav_16k(wav_path)
    runner = ContinuousC5LiveRunner(
        network=network,
        ownership_enabled=True,
        intercept=intercept,
        secrets=secrets or load_runtime_secrets(),
        budget=budget,
        use_silero=intercept is None,
    )
    producer = None
    if sortformer:
        producer = NativeSortformerProducer(wav_path, clock=runner._clock.now)
        if producer.available:
            producer.start()
    try:
        audio_seconds = float(samples.size) / float(HZ)
        await runner.open(audio_seconds=audio_seconds)
        chunk = 512
        offset = 0
        while offset < samples.size:
            end = min(offset + chunk, samples.size)
            await runner.feed(samples[offset:end])
            offset = end
            if producer is not None:
                for event in producer.poll():
                    await runner.receive(
                        hypothesis_at_boundary(
                            event.boundary,
                            capture_epoch=1,
                            available_at_monotonic_s=event.available_at_monotonic_s,
                            producer_generation=runner._producer,
                            reference_generation=runner._reference,
                            hypothesis_id=event.event_id,
                            local_slot=event.candidate_slot,
                        )
                    )
        terminal = await runner.finalize()
        await runner.admit()
        children = await runner.translate()
        return {
            "ok": True,
            "network": network,
            "text": terminal.text,
            "n_timed": len(terminal.timed_tokens),
            "n_children": len(children),
            "methods": list(runner.methods),
            "open_session_calls": runner.open_session_calls,
            "receipts": [
                {
                    "hypothesis_id": item.hypothesis_id,
                    "disposition": item.disposition,
                    "available_at_monotonic_s": item.available_at_monotonic_s,
                    "applied_at_monotonic_s": item.applied_at_monotonic_s,
                }
                for item in runner.receipts
            ],
            "live_route": LIVE_ROUTE,
            "deepgram_reserve_usd": runner.deepgram_reserve_usd,
        }
    finally:
        if producer is not None:
            producer.close()
        await runner.close()
