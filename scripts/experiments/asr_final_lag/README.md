# ASR partial/final timing probe

Measures how soon each streaming ASR route turns continuous speech into partial and final text without forced finalization, and what finalize requests change. A second script replays the recorded partial transcripts through decision models to test when text could be released for translation. A third script tests decision-model splits of long utterances and judges the chunked translations. The results decide how Peer early release should find release points.

## Next step

`RESEARCH.md` summarizes the goal, all experiments, conclusions and open questions for the next session. The runs listed in `HANDOFF.md` are recorded. Findings are in the `NOTE.md` files of `results/20261020T185455Z` (Qwen Audio cut and Gemini end variants), `results/interim-20261020T190834Z` (Gemini interim release) and `results/split-20261020T194622Z` (long-utterance split probe).

## Run the ASR probe

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
- Data: `--data ko_kr ja_jp en_us` (FLEURS read speech, the default) and `ko_conv ja_conv en_conv` (YODAS2 conversation).
- Subsets: `--providers soniox deepgram`, `--variants natural`.
- `--parallel N` runs at most N variants of one provider at the same time (default: all at once).
- Qwen uses `ALIBABA_API_KEY_BEIJING` first. The default endpoint is the Beijing host `wss://dashscope.aliyuncs.com/api-ws/v1/inference`.
- International DashScope keys need `--qwen-endpoint wss://dashscope-intl.aliyuncs.com/api-ws/v1/inference`.
- A full FLEURS run takes about 8 minutes. Each data set streams about 2 minutes in real time with all of its runs in parallel. Billed audio is about 1 hour 10 minutes in total: Soniox 18 min, Deepgram 24 min, Qwen Audio 12 min, Gemini 18 min.

## Share results

Each ASR run writes `results/<UTC timestamp>/`, each decision replay writes `results/decide-<UTC timestamp>/` and each split probe writes `results/split-<UTC timestamp>/`. Commit and push those folders:

```powershell
git add scripts/experiments/asr_final_lag/results
git commit -m "Record ASR timing probe results"
git push
```

| File | Content |
| --- | --- |
| `summary.md` | Readable tables |
| `summary.json` | The same metrics with per-sentence (ASR) or per-stream (replay) rows |
| `runs/*.json.gz` | ASR event log: text deltas, word timestamps, finalize controls, a small raw-message sample, errors |
| `reference/*.json` | Reference texts and timing of the input audio |
| `calls/*.jsonl.gz` | Replay call log: virtual time, round trip, answer, the state sent to the model |
| `meta.json` | Run settings, package versions, key variable names |

Results contain no API keys, host names or user names. The scripts refuse to write a result that contains a key.

## ASR variants

| Variant | Providers | Behavior |
| --- | --- | --- |
| `natural` | all | Application settings, no forced finalize. Deepgram enables `interim_results` and Gemini records `interim_input_transcription` to observe partials; the application ignores both. |
| `nointerim` | Deepgram | Exact application settings (`interim_results=false`) to check that interim results do not change final timing. |
| `finalize_gap` | Soniox, Deepgram, Gemini | Finalize 0.10 s after each sentence's speech end, inside the 0.2 s silence before the next sentence (best case for a pause-aligned finalize). |
| `finalize_mid` | Soniox, Deepgram, Gemini | Finalize every 2.5 s regardless of speech (worst case for word damage). |
| `semantic` | Qwen Audio | `semantic_punctuation_enabled=true`; the application uses `false`. |
| `cut_app` | Qwen Audio | One task per local segment on two alternating connections, cut where the current Peer rules would cut (see `segmenter.py`): Smart Turn complete (0.75) then 512 ms, otherwise 800 ms; 4 s/192 ms; 5 s/128 ms; 6 s hard cut with 0.3 s replay. |
| `cut_pause` | Qwen Audio | As `cut_app`, but cut at the first 200 ms pause after 2 s. |
| `cut_smart` | Qwen Audio | As `cut_app`, but cut at a 200 ms pause when Smart Turn scores at least 0.5. |
| `cut_fixed3` | Qwen Audio | As `cut_app`, but cut every 3 s even during speech. |
| `cut_smart_ctx` | Qwen Audio | `cut_smart` plus the previous tasks' text (last 400 characters) as ASR context in `payload.input.context`. The context format follows the Fun-ASR client events and is not confirmed for Qwen Audio 3.1. |
| `end_app` | Gemini | Continuous stream with `audio_stream_end` at every `cut_app` cut, as the application does now. |
| `end_smart` | Gemini | Continuous stream with `audio_stream_end` only at Smart Turn cuts and 800 ms pauses, never during speech. |

Provider settings mirror the application adapters: Soniox `stt-rt-v5` with endpoint detection off and diarization on, Deepgram `nova-3` with endpointing off, Qwen Audio `qwen-audio-3.1-asr-flash-streaming` with the Peer `max_sentence_silence=800`, and Gemini `gemini-3.5-transcribe-live` with automatic activity detection (500 ms prefix, 400 ms silence). Finalize is the Soniox `finalize` message, the Deepgram `Finalize` message and the Gemini `audio_stream_end` input, which the application sends when a Peer segment ends. For Gemini, `fin ack` is the first final transcript after `audio_stream_end`.

## Input audio

FLEURS read speech: the first files of each language's FLEURS test archive (CC-BY-4.0) are level-normalized, trimmed to their speech and joined into about 2 minutes of continuous read speech with 0.2 s of silence between sentences. Speech bounds come from the repository's bundled Silero VAD model (`src/puripuly_heart/data/vad/silero_vad.onnx`). Each reference sentence records its clip bounds (`start_s`, `end_s`) and its speech bounds (`speech_start_s`, `speech_end_s`). Sentence-end timings and `finalize_gap` use `speech_end_s`.

YODAS2 conversation (CC-BY-3.0 YouTube audio with uploaded captions): 125 s from one video per language, resampled from 24 kHz to 16 kHz and level-normalized.

| Data | Video id | Window | Content |
| --- | --- | --- | --- |
| `ko_conv` | `3BPeG47L7Hc` (`ko000` shard 12) | 0-125 s | Two hosts discussing a housing development, overlapping turns |
| `ja_conv` | `19cQStzeIgE` (`ja000` shard 3) | 0-125 s | Two creators chatting, short turns and laughter |
| `en_conv` | `06inwHyT7l0` (`en000` shard 4) | 22-147 s | Three podcast hosts talking about a tabletop game |

Caption lines become the reference units after speaker labels, sound tags and laughter runs are removed. Captions are uploader text: they are not sentence-segmented and can differ from the speech, so the ASR summary marks these data sets as caption units.

The first run builds the audio into `data/` (ignored by git) by streaming only the needed part of each archive, and rebuilds it when the data version changes.

## ASR metrics

`summary.md` defines each column. Sentence-end timings compare when the text covers each reference sentence end with that sentence's speech end (last Silero speech frame; provider word end timestamps are typically 0.1-0.3 s earlier). Word timings use provider word or token end timestamps. `release diff` compares the partial text of each sentence at its `partial stable` time with the final text of that sentence, which shows whether a partial-text release would translate different words than the final.

Re-analyze an existing results folder:

```powershell
uv run --with numpy --with rapidfuzz python scripts/experiments/asr_final_lag/analyze.py scripts/experiments/asr_final_lag/results/<timestamp> 0.25
```

The last argument is the assumed decider round trip in seconds for the estimated partial-decision path.

## Decision model replay

`decide.py` takes the newest ASR results that contain each data set's `natural` runs and replays every provider's partial transcript in virtual time. Each model call advances that stream's clock by the measured round trip, so the number of calls and the release times match what one in-flight request per lane would give.

- Text sent to the models is lowercased and has punctuation removed.
- Proposal lane: whenever the transcript changes, a `choice` question over numbered markers placed after each new word (each new character in Japanese, at most 12) asks for the last marker where a sentence or self-contained clause ends.
- Verify lane: a candidate is checked once more with a `noul` question when 3 English, 2 Korean or 3 Japanese characters of right context have arrived (`before` / `after` state).
- Policies computed from the same calls: release at the proposal answer (`immediate`), release when the right-context check agrees (`verify`), and `verify +silence`, which also releases a candidate after 0.8 s without transcript change.
- Baselines: natural finals, a punctuation rule on partials, and whole-transcript debounce at 0.3 s and 0.6 s.
- Labels are sentence-final punctuation and commas in the Soniox final transcript of the same audio (hindsight), mapped onto each provider's transcript. Latency is measured from the Soniox end timestamp of the labeled sentence's last character.

Defaults: models `typesafe/jev-1.14`, `inception/mercury-decide`, `perplexity/pplx-decider-v1.2`, `perplexity/pplx-decider-flash-v1.2`; providers Soniox, Deepgram, Gemini and Qwen Audio; all six data sets. Options: `--models`, `--providers`, `--data`, `--results <folders>`, `--concurrency` (in-flight requests per model, default 6), `--key MY_VARIABLE`.

`--fake-decider` replaces the models with a label oracle and makes no API calls. It checks the replay and scoring: the oracle should reach close to 100% recall with no mid-sentence releases.

## Long-utterance split probe

`split.py` tests whether a decision model can split long utterances that have no punctuation release, before the speaker reaches a sentence end. The results decide whether Peer should replace its 4-6 s acoustic cut rules with text splits for streaming routes.

- Punctuation in the partial text releases text as usual (the `punct rule` of the replay above).
- When no punctuation release happens for 3 s (`--trigger`) after new text starts, a split policy may release an earlier part of the unreleased text. Each later trigger repeats this every 3 s until the next punctuation release.
- Candidates are word ends (characters in Japanese) with at least 4 Korean or Japanese characters or 8 English characters before them and 2-3 characters of right context.
- Each candidate is asked once per model and question, with the earlier text as `previous`, the unreleased text up to the candidate as `before`, and the rest of the partial text as `after`:
  - `Q1` (sentence): whether `before` ends a sentence or self-contained clause.
  - `Q2` (translation): whether a subtitle for `before` in the target language would stay correct once `after` is heard. Targets: Korean to English, Japanese to Korean, English to Korean (`--target ko=Japanese` changes one).
- Decision policies release the latest candidate at or above a threshold. `+forced` releases the highest-scored candidate at the second trigger (6 s) when none passes.
- Baselines: no split, the latest word end (a stand-in for time-based acoustic cuts), the latest provider comma, and the longest pause in Soniox word timing.
- A split is released at its trigger time plus a 0.4 s finalize allowance plus, for decision models, the slowest round trip of that trigger's calls.
- Quality: every split utterance is translated chunk by chunk with the application's translation prompt and earlier source lines as context, and as a whole. A judge model compares them (`same`, `minor`, `changed`) and flags `premature` subtitles that mislead before the next chunk appears.
- `--fake` replaces decision models with a punctuation-label oracle and translation with an echo, and makes no API calls.

Results go to `results/split-<UTC timestamp>/` with the same layout as the replay: `summary.md`, `summary.json`, `meta.json` and `calls/` (decision calls, translations and judgments).

## Cut schedules and Smart Turn

`segmenter.py` runs the bundled Silero VAD (32 ms frames, threshold 0.5, at least two voiced frames) and the bundled Smart Turn v3.2 model on the probe audio before streaming. Smart Turn scores each pause when it reaches 200 ms, with up to 8 s of audio that starts at the speech onset after the last pause of at least 0.5 s. The cut policies above are computed from these frames, so every variant of a data set cuts at the same audio times. Policies other than `app` end a turn at an 800 ms pause, cut at an 80 ms pause after 6 s and cut during speech after 10 s.

`smartturn_eval.py` (no API calls) labels every 200 ms pause with the Soniox final token that ended there: sentence-final punctuation, a comma or neither. It reports how many sentence ends have a 200 ms pause, how well Smart Turn separates sentence ends from other pauses, and a preview of each cut policy.

```powershell
uv run scripts/experiments/asr_final_lag/smartturn_eval.py
```

## Gemini interim release replay

`interim_release.py` (no API calls) replays recorded Gemini runs (`natural`, `end_smart`, `end_app`). It compares releasing each final as it arrives with releasing interim text up to its last sentence-final punctuation once that prefix has been unchanged for 0, 0.6 or 1.0 s. When a final arrives, only the rest is released, either from the final (located by text alignment) or from the last interim. It reports per-character latency from Soniox timestamps and CER, deletions and insertions against the reference. Each results folder needs a Soniox `natural` run of the same data for timing.

```powershell
uv run scripts/experiments/asr_final_lag/interim_release.py
```

## Limits

- FLEURS is read speech with a pause at every sentence boundary. YODAS2 windows are three single videos, not a balanced conversation set.
- Coverage uses character alignment to the reference. A recognized sentence end that differs from the reference (for example kanji versus kana) can be reported as missing or late.
- Replay labels come from Soniox punctuation, so the punctuation-rule baseline is close to the labels by construction, and clause-level releases are only credited at commas.
- Timings include the network path from the machine that runs the probe; `connect s` and the replay round trips record it.
- Soniox token timestamps advance by about 70-80 ms per finalize request, so Soniox word timings in the `finalize_*` variants drift ahead of the audio clock. Sentence-end timings do not depend on provider timestamps.
