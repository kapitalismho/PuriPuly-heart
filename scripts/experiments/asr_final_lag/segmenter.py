from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import fleurs
import numpy as np

from puripuly_heart.core.audio.smart_turn import (
    SMART_TURN_COMPLETE_THRESHOLD,
    prepare_smart_turn_audio,
)
from puripuly_heart.core.audio.smart_turn_features import compute_whisper_log_mel_features
from puripuly_heart.core.vad.silero import SileroVadOnnx

SAMPLE_RATE = fleurs.SAMPLE_RATE
FRAME = fleurs.VAD_FRAME
FRAME_S = FRAME / SAMPLE_RATE
MIN_VOICED_FRAMES = fleurs.VAD_MIN_FRAMES
PROBE_S = 0.200
ONSET_RESET_S = 0.5
WINDOW_S = 8.0
PRE_ROLL_S = 0.5
APP_REPLAY_S = 0.3
TURN_END_S = 0.8
SMART_THRESHOLD = 0.5
SMART_TURN_MODEL = fleurs.REPO_ROOT / "src/puripuly_heart/data/models/smart-turn-v3.2-cpu.onnx"
POLICIES = ("app", "pause", "smart", "fixed3")


@dataclass
class Pause:
    start_s: float
    end_s: float
    turn_start_s: float
    score: float | None = None

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


@dataclass
class Analysis:
    duration_s: float
    voiced: np.ndarray
    onsets: list[float]
    pauses: list[Pause]

    def pause_at(self, t: float) -> Pause | None:
        for pause in self.pauses:
            if pause.start_s <= t < pause.end_s:
                return pause
        return None


@dataclass(frozen=True)
class Cut:
    t: float
    kind: str
    resume_s: float | None


@dataclass(frozen=True)
class Segment:
    start_s: float
    open_s: float
    end_s: float
    cut: str


class SmartTurn:
    def __init__(self, path: Path = SMART_TURN_MODEL) -> None:
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.inter_op_num_threads = 1
        options.intra_op_num_threads = 2
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(path), sess_options=options, providers=["CPUExecutionProvider"]
        )

    def score(self, audio: np.ndarray) -> float:
        prepared = prepare_smart_turn_audio(audio, sample_rate_hz=SAMPLE_RATE)
        features = compute_whisper_log_mel_features(prepared)
        outputs = self.session.run(None, {"input_features": np.expand_dims(features, axis=0)})
        return float(np.asarray(outputs[0]).reshape(-1)[0])


def to_float(pcm: bytes | np.ndarray) -> np.ndarray:
    if isinstance(pcm, bytes):
        return np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
    return np.asarray(pcm, dtype=np.float32)


def voiced_frames(audio: np.ndarray) -> np.ndarray:
    vad = SileroVadOnnx(fleurs.VAD_MODEL)
    count = len(audio) // FRAME
    raw = np.array(
        [
            vad.speech_probability(audio[i * FRAME : (i + 1) * FRAME], sample_rate_hz=SAMPLE_RATE)
            >= fleurs.VAD_THRESHOLD
            for i in range(count)
        ]
    )
    voiced = np.zeros(count, dtype=bool)
    i = 0
    while i < count:
        if raw[i]:
            j = i
            while j < count and raw[j]:
                j += 1
            if j - i >= MIN_VOICED_FRAMES:
                voiced[i:j] = True
            i = j
        else:
            i += 1
    return voiced


def analyze_audio(pcm: bytes | np.ndarray, smart: SmartTurn | None = None) -> Analysis:
    audio = to_float(pcm)
    voiced = voiced_frames(audio)
    duration = len(audio) / SAMPLE_RATE
    onsets, pauses = [], []
    turn_start = None
    i, count = 0, len(voiced)
    while i < count:
        if not voiced[i]:
            i += 1
            continue
        start = i * FRAME_S
        if turn_start is None or (pauses and pauses[-1].duration_s >= ONSET_RESET_S):
            turn_start = start
        onsets.append(start)
        j = i
        while j < count and voiced[j]:
            j += 1
        k = j
        while k < count and not voiced[k]:
            k += 1
        pauses.append(Pause(j * FRAME_S, k * FRAME_S if k < count else duration, turn_start))
        i = k
    if smart is not None:
        for pause in pauses:
            if pause.duration_s < PROBE_S:
                continue
            probe = pause.start_s + PROBE_S
            begin = max(pause.turn_start_s, probe - WINDOW_S)
            window = audio[int(begin * SAMPLE_RATE) : int(probe * SAMPLE_RATE)]
            pause.score = smart.score(window)
    return Analysis(duration, voiced, onsets, pauses)


def _is_voiced(analysis: Analysis, t: float) -> bool:
    index = int(t / FRAME_S)
    return 0 <= index < len(analysis.voiced) and bool(analysis.voiced[index])


def cut_schedule(
    analysis: Analysis, policy: str, smart_threshold: float = SMART_THRESHOLD
) -> list[Cut]:
    cuts: list[Cut] = []
    opened: float | None = None
    t = 0.0
    while t < analysis.duration_s:
        t = round(t + FRAME_S, 6)
        voiced = _is_voiced(analysis, t - FRAME_S / 2)
        if opened is None:
            if voiced:
                opened = t - FRAME_S
            continue
        age = t - opened
        pause = None if voiced else analysis.pause_at(t - FRAME_S / 2)
        paused = 0.0 if pause is None else t - pause.start_s
        cut = _decide(policy, age, paused, pause, voiced, t, smart_threshold)
        if cut is None:
            continue
        cuts.append(cut)
        opened = t if cut.resume_s is not None else None
    return cuts


def _decide(policy, age, paused, pause, voiced, t, threshold) -> Cut | None:
    score = None if pause is None else pause.score
    probed = paused >= PROBE_S and score is not None
    if policy == "app":
        if age >= 6.0:
            return Cut(t, "speech" if voiced else "pause", t - APP_REPLAY_S)
        if age >= 5.0 and paused >= 0.128:
            return Cut(t, "pause", None)
        if age >= 4.0 and paused >= 0.192:
            return Cut(t, "pause", None)
        if paused >= 0.512 and probed and score >= SMART_TURN_COMPLETE_THRESHOLD:
            return Cut(t, "end", None)
        if paused >= TURN_END_S:
            return Cut(t, "end", None)
        return None
    if policy == "pause":
        if paused >= TURN_END_S:
            return Cut(t, "end", None)
        if age >= 2.0 and paused >= PROBE_S:
            return Cut(t, "pause", None)
    elif policy == "smart":
        if paused >= TURN_END_S:
            return Cut(t, "end", None)
        if probed and paused < PROBE_S + FRAME_S and score >= threshold:
            return Cut(t, "smart", None)
    elif policy == "fixed3":
        if paused >= TURN_END_S:
            return Cut(t, "end", None)
        if age >= 3.0:
            return Cut(t, "speech" if voiced else "pause", t if voiced else None)
        return None
    if age >= 6.0 and paused >= 0.08:
        return Cut(t, "pause", None)
    if age >= 10.0:
        return Cut(t, "speech" if voiced else "pause", t if voiced else None)
    return None


def segments(analysis: Analysis, cuts: list[Cut]) -> list[Segment]:
    result: list[Segment] = []
    onsets = analysis.onsets
    opening: tuple[float, float] | None = None
    if onsets:
        opening = (max(0.0, onsets[0] - PRE_ROLL_S), onsets[0])
    for cut in cuts:
        if opening is None:
            continue
        start, open_s = opening
        result.append(Segment(round(start, 3), round(open_s, 3), round(cut.t, 3), cut.kind))
        if cut.resume_s is not None:
            opening = (cut.resume_s, cut.t)
            continue
        following = next((o for o in onsets if o >= cut.t), None)
        opening = None if following is None else (max(cut.t, following - PRE_ROLL_S), following)
    if opening is not None and opening[1] < analysis.duration_s:
        start, open_s = opening
        result.append(
            Segment(round(start, 3), round(open_s, 3), round(analysis.duration_s, 3), "eof")
        )
    return result
