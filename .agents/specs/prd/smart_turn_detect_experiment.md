# 1. Product goal and proposed change

PuriPuly is a real-time speech translation application primarily used with VRChat.

It processes two audio channels:

- **Self:** the local user's microphone.
- **Peer:** captured audio from another speaker.

Both channels need to decide where one spoken turn ends so that speech-to-text can finalize the transcript and translation can begin.

Today, PuriPuly primarily uses a conventional voice activity detector, or VAD, for this decision.

A VAD answers a narrow acoustic question:

```text
Is the current audio speech or non-speech?

```

It does not understand whether the speaker has completed a sentence or is merely pausing in the middle of one.

For example:

```text
"I was thinking that..."

400 ms pause

"...we should try it tomorrow."

```

A short VAD hangover may split this into two separate speech segments even though the speaker intended it as one turn.

PuriPuly currently compensates for this with speculative translation.

Conceptually:

```text
short VAD silence
    -> finalize the current STT segment
    -> begin translation immediately
    -> detect that the speaker resumed
    -> merge the transcripts
    -> cancel or replace the earlier translation

```

This strategy intentionally spends additional translation work to achieve two goals at once:

1. Keep the VAD hangover short so completed speech feels responsive.
2. Recover when the short VAD incorrectly cuts a turn at a mid-sentence pause.

The cost is additional state management, cancelled translation requests, repeated translation calls, and different behavior between the self and peer pipelines.

We want to evaluate a different architecture:

```text
VAD detects a silence candidate
    -> Smart Turn evaluates whether the spoken turn sounds complete
    -> end early when the model approves
    -> otherwise continue listening
    -> enforce a fixed hard timeout as a safety boundary
    -> translate once after the turn is committed

```

The proposed initial policy is:

```text
224 ms silence: Smart Turn probe 1
416 ms silence: Smart Turn probe 2
608 ms silence: Smart Turn probe 3
800 ms silence: unconditional hard end

```

The intended behavior is:

- Clear completed turns may finish around 300–500 ms after accounting for inference.
- Ambiguous pauses receive more time before the transcript is cut.
- No turn is held beyond the 800 ms hard boundary.
- The existing speculative translation path remains enabled during initial experiments as a recovery mechanism.
- Speculative translation is removed only if measurements show that the new endpoint policy is sufficiently reliable.

The target is not a perfect turn detector.

The actual product question is:

```text
Can an 800 ms VAD safety boundary plus selective Smart Turn early exits
reduce false segmentation relative to the current effective 512 ms VAD,
while preserving similar perceived latency and reducing speculative
translation retries?

```

---

# 2. What Smart Turn v3.2 is

Smart Turn v3.2 is an open-source, audio-native end-of-turn classification model developed by the Pipecat community.

Unlike a conventional VAD, Smart Turn does not merely determine whether the current frame contains speech. It analyzes the recent spoken turn and estimates whether the speaker appears to have finished their thought.

It can use information that a VAD does not model directly, including:

- Speech rhythm.
- Intonation and prosody.
- Grammatical completion cues present in the audio.
- Hesitation patterns.
- Whether the final words sound like a completed phrase or an unfinished continuation.

Smart Turn v3.2 is based on a Whisper Tiny audio encoder with a classification head. It has approximately eight million parameters.

Available variants include:

- An approximately 8 MB int8 CPU model.
- A larger unquantized model intended primarily for GPU execution.

PuriPuly should evaluate the CPU int8 model because the application already performs audio capture, STT, translation, UI work, and possibly local inference on the same machine.

The model supports multiple languages, including Korean, Japanese, English, Chinese, and other languages relevant to PuriPuly.

## Input expectations

Smart Turn expects:

- 16 kHz audio.
- Mono PCM audio.
- Up to the latest eight seconds of the current user turn.
- Shorter inputs left-padded with zeroes.
- Longer turns truncated from the beginning so the most recent eight seconds remain.

The model should receive the whole available current turn, not only the most recent silence or the final short speech fragment.

Conceptually:

```text
current turn audio:
"I was thinking that..."

plus the current silence

    -> Smart Turn score

```

It should not receive only:

```text
the last 200 ms of silence

```

## Smart Turn is not a continuously streaming VAD replacement

Smart Turn is intended to work together with a lightweight VAD such as Silero.

The VAD remains responsible for detecting that a possible pause has begun. Smart Turn is invoked only after enough silence has accumulated to create a meaningful endpoint candidate.

```text
Silero VAD
    -> detects continuing silence
    -> schedules Smart Turn evaluation

Smart Turn
    -> estimates whether the current spoken turn is complete

```

This distinction is important:

```text
VAD:
"Is there speech right now?"

Smart Turn:
"Given the speech so far, does this sound like the end of the turn?"

```

## Smart Turn returns a score, not a guaranteed decision

The model returns an end-of-turn score.

That score must not be treated as a perfectly calibrated probability. A score of `0.90` does not automatically guarantee a 90% chance that ending the turn is safe.

The product policy must determine:

- At which silence points the model is evaluated.
- Which threshold is required.
- Whether later evaluations use the same threshold.
- How long the system waits before an unconditional hard end.
- How stale inference results are discarded when speech resumes.

Threshold selection must therefore be based on complete policy measurements rather than a hard-coded interpretation of the raw score.

## Repeated probing is an experimental PuriPuly policy

The proposed repeated-probe behavior is not assumed to be the official or optimal Smart Turn integration.

We are specifically testing whether fresh evaluations later in the same silence provide additional useful information:

```text
224 ms: evaluate the current turn
416 ms: evaluate again with a newer audio snapshot
608 ms: evaluate again if still unresolved
800 ms: end regardless of model output

```

A later probe may help because the model receives a longer pause and a newer causal snapshot.

It may also hurt because every additional probe creates another opportunity for a false end-of-turn prediction.

For that reason, the experiment must evaluate the full policy:

```text
first threshold crossing across all probes
plus the hard timeout

```

It is not sufficient to report the accuracy of each individual probe independently.

## Intended role in PuriPuly

Smart Turn is not being introduced as an autonomous logical conversation manager.

Its proposed role is narrower:

```text
A semantic and prosodic gate that allows the VAD boundary
to fire earlier when the end of the spoken turn appears clear.

```

The hard VAD timeout remains the final safety mechanism.

The desired long-term pipeline is:

```text
audio
    -> Silero VAD pause candidate
    -> Smart Turn early-end gate
    -> STT finalization
    -> one final translation

```

If the experiment succeeds, the same basic endpoint state machine may eventually be used for both self and peer audio.

However, the code path may be shared while thresholds or timeout profiles remain channel-specific if real measurements show that peer audio is more difficult because of compression, background noise, overlapping speakers, and VRChat audio processing.

---

# 3. Why this must be introduced experimentally

Smart Turn has shown better latency-versus-false-cutoff tradeoffs than a silence-only VAD in public end-of-turn benchmarks, but it still makes both kinds of errors:

- **False complete:** it ends a turn during a mid-sentence pause.
- **False incomplete:** it fails to end a turn after the speaker has actually finished.

False complete is the more important risk for this project because it permanently splits a transcript unless a recovery mechanism remains available.

Repeated evaluation also changes the risk profile:

```text
More probes
    -> more opportunities to recover missed true endings
    -> more opportunities to falsely approve a mid-turn pause

```

Therefore, this work must proceed through small gates:

1. Evaluate repeated probes on public prediction artifacts.
2. Measure CPU inference cost locally.
3. Run shadow inference without changing behavior.
4. Enable active early endings while retaining speculative recovery.
5. Disable speculative translation only after the earlier stages pass.

Do not skip directly to replacing the production endpoint or deleting speculative translation.