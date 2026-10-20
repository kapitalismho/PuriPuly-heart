# ASR partial/final timing probe

Measures how soon each streaming ASR route turns continuous speech into partial and final text without forced finalization, and what finalize requests change. The result decides whether Peer early release can rely on provider finals alone or needs a partial-text decision plus a forced finalize.

## Run

Requirements: [uv](https://docs.astral.sh/uv/) and API keys in the repository-root `.env.local` (`.env` and process environment variables also work). Providers without a key are skipped.

| Provider | Variable names (first match wins) |
| --- | --- |
| Soniox | `SONIOX_API_KEY` |
| Deepgram | `DEEPGRAM_API_KEY` |
| Qwen Audio (DashScope) | `ALIBABA_API_KEY_BEIJING`, `ALIBABA_API_KEY`, `DASHSCOPE_API_KEY`, `QWEN_API_KEY`, `ALIBABA_CLOUD_API_KEY` |
| Gemini Transcribe | `GOOGLE_API_KEY_2`, `GOOGLE_API_KEY2`, `GEMINI_API_KEY_2`, `GOOGLE2_API_KEY` |

```powershell
uv run scripts/experiments/asr_final_lag/run.py
```

- Other variable names: `--key gemini=MY_VARIABLE` (repeatable).
- Subsets: `--langs ko_kr`, `--providers soniox deepgram`, `--variants natural`.
- Qwen uses `ALIBABA_API_KEY_BEIJING` first. The default endpoint is the Beijing host `wss://dashscope.aliyuncs.com/api-ws/v1/inference`.
- International DashScope keys need `--qwen-endpoint wss://dashscope-intl.aliyuncs.com/api-ws/v1/inference`.
- A full run takes about 8 minutes. Each language streams about 2 minutes in real time with all of its runs in parallel. Billed audio is about 1 hour 10 minutes in total: Soniox 18 min, Deepgram 24 min, Qwen Audio 12 min, Gemini 18 min.

## Share results

Each run writes `results/<UTC timestamp>/`. Commit and push that folder:

```powershell
git add scripts/experiments/asr_final_lag/results/<timestamp>
git commit -m "Record ASR timing probe results"
git push
```

| File | Content |
| --- | --- |
| `summary.md` | Readable tables per language and the final-only versus partial-decision comparison |
| `summary.json` | The same metrics with per-sentence rows |
| `runs/*.json.gz` | Per-run event log: text deltas, word timestamps, finalize controls, a small raw-message sample, errors |
| `reference/*.json` | Sentence texts and speech boundaries of the input audio |
| `meta.json` | OS name and version, Python and package versions, selected variants, key variable names |

Results contain no API keys, host names or user names. The runner refuses to write a result file that contains a key.

## Variants

| Variant | Providers | Behavior |
| --- | --- | --- |
| `natural` | all | Application settings, no forced finalize. Deepgram enables `interim_results` and Gemini records `interim_input_transcription` to observe partials; the application ignores both. |
| `nointerim` | Deepgram | Exact application settings (`interim_results=false`) to check that interim results do not change final timing. |
| `finalize_gap` | Soniox, Deepgram, Gemini | Finalize 0.10 s after each sentence's speech end, inside the 0.2 s silence before the next sentence (best case for a pause-aligned finalize). |
| `finalize_mid` | Soniox, Deepgram, Gemini | Finalize every 2.5 s regardless of speech (worst case for word damage). |
| `semantic` | Qwen Audio | `semantic_punctuation_enabled=true`; the application uses `false`. |

Provider settings mirror the application adapters: Soniox `stt-rt-v5` with endpoint detection off and diarization on, Deepgram `nova-3` with endpointing off, Qwen Audio `qwen-audio-3.1-asr-flash-streaming` with the Peer `max_sentence_silence=800`, and Gemini `gemini-3.5-transcribe-live` with automatic activity detection (500 ms prefix, 400 ms silence). Finalize is the Soniox `finalize` message, the Deepgram `Finalize` message and the Gemini `audio_stream_end` input, which the application sends when a Peer segment ends. For Gemini, `fin ack` is the first final transcript after `audio_stream_end`.

## Input audio

The first files of each language's FLEURS test archive (CC-BY-4.0) are level-normalized, trimmed to their speech and joined into about 2 minutes of continuous read speech with 0.2 s of silence between sentences. Speech bounds come from the repository's bundled Silero VAD model (`src/puripuly_heart/data/vad/silero_vad.onnx`). The first run builds the audio into `data/` (ignored by git) by streaming only the needed part of the archive, and rebuilds it when the data version changes.

Each reference sentence records its clip bounds (`start_s`, `end_s`) and its speech bounds (`speech_start_s`, `speech_end_s`). Sentence-end timings and `finalize_gap` use `speech_end_s`.

## Metrics

`summary.md` defines each column. Sentence-end timings compare when the text covers each reference sentence end with that sentence's speech end (last Silero speech frame; provider word end timestamps are typically 0.1-0.3 s earlier). Word timings use provider word or token end timestamps. `release diff` compares the partial text of each sentence at its `partial stable` time with the final text of that sentence, which shows whether a partial-text release would translate different words than the final.

Re-analyze an existing results folder:

```powershell
uv run --with numpy --with rapidfuzz python scripts/experiments/asr_final_lag/analyze.py scripts/experiments/asr_final_lag/results/<timestamp> 0.25
```

The last argument is the assumed decider round trip in seconds for the estimated partial-decision path.

## Limits

- Read speech with clear sentence ends, not conversation.
- Coverage uses character alignment to the reference. A recognized sentence end that differs from the reference (for example kanji versus kana) can be reported as missing or late.
- Timings include the network path from the machine that runs the probe; `connect s` records connection setup time.
- Soniox token timestamps advance by about 70-80 ms per finalize request, so Soniox word timings in the `finalize_*` variants drift ahead of the audio clock. Sentence-end timings do not depend on provider timestamps.
