# Peer early release research log

This document hands the research over to a new session. It records the goal, how the question changed, every experiment so far with its results, the current conclusions and the open questions. Raw evidence lives in `results/*/` (each folder has `summary.md`, and most have a `NOTE.md` with the interpretation).

- Branch: `claude/tender-wright-4x2d4y`. No pull request exists. No application code was changed; all work is under `scripts/experiments/asr_final_lag/` plus three `.gitignore` exception lines at the repository root.
- Workflow: a cloud session writes scripts and analyzes results. Live API runs happen on the maintainer's machine, often through a local agent that follows `HANDOFF.md`, and the results are pushed to the branch.
- Keys are in the repository-root `.env.local` (gitignored). Never ask for key values, never print them, never commit them. The scripts refuse to write a result file that contains a key.

## 1. Problem and goal

### 1.1 The user-visible problem

When a remote speaker (Peer) keeps talking without a long pause, translated subtitles appear very late. Self (the user's own speech) is out of scope: its VAD hangover is effectively 500 ms (`STT_DEFAULT_VAD_HANGOVER_MS`, `src/puripuly_heart/config/runtime_resolution.py`), and the user does not need to read their own translation quickly.

### 1.2 Current Peer pipeline (application code on `dev`)

- Segmentation is acoustic (`src/puripuly_heart/core/audio/listen_delivery.py`, `ListenDeliveryController`):
  - Smart Turn v3.2 is asked at a 224 ms pause (`SMART_PROBE_MS`). The segment ends at 512 ms if Smart Turn says complete (threshold 0.75), otherwise at 800 ms.
  - After 4 s, a 192 ms pause ends the segment. After 5 s, a 128 ms pause does.
  - At 6 s the segment is cut during speech (`HARD_LIMIT_S`), and the next segment replays 300 ms of audio.
- Translation starts only after a segment is sealed and its final text arrives. Peer ignores partial text for translation.
- Provider behavior at a seal:
  - Soniox and Deepgram: a finalize message.
  - Gemini Transcribe: every seal, including the 4/5 s steps and the 6 s hard cut, sends `audio_stream_end` on a continuous stream. Each Gemini final becomes an independent unit with no link to a local segment.
  - Qwen Audio 3.1: one task per segment (`finish-task` at the seal), two connections per channel, results published in source order.
- So in continuous speech the first words of a segment wait 4-6 s for the cut, then for the final, then for translation.

### 1.3 Goal as last stated by the user

- Release translations stably and quickly for utterances of 3-6 s.
- Utterances longer than 6 s must also produce intermediate phrases.
- For this research, ignore existing contracts and scope decisions (for example #211 R1/R4/R8, "no speculative Peer translation", Gemini final-only, Qwen one task per segment). Adopting a result in the application will need those decisions revisited separately.
- Providers in scope: Soniox, Deepgram, Gemini Transcribe and Qwen Audio 3.1. Qwen3-ASR is out of scope.
- Smart Turn must be scored 200 ms into a pause; that is the recommended setting and how the model was trained.

## 2. How the question evolved

1. **Decision model as a turn detector.** The starting idea: use a text decision model (TypeSafe Jev, Inception Mercury Decide, Perplexity Decider v1.2 and Decider Flash v1.2, all on the OpenRouter Decisions API) to decide whether a turn is complete. Cost was never the problem: a few cents per audio hour even when called on every partial update.
2. **Narrowed to Peer.** The real issue is not the end-of-turn decision but releasing translatable text while the speaker keeps talking. Idea: a "text layer" that decides translation units, with acoustic segmentation kept only for ASR, speaker tracking and safety.
3. **ASR timing probe.** Whether a text layer can work depends on when text arrives. Natural finals turned out to be far too slow, and partial text with ASR punctuation turned out to be fast and mostly stable (section 4.1-4.2).
4. **Decision-model replay.** A replay of recorded partials through the decision models (marker choice plus a right-context check) lost to a plain punctuation rule (section 4.3). The user noted that an earlier simultaneous-interpretation experiment with an open-source sentence-end model had failed the same way.
5. **Punctuation plus decision model.** The user considered the punctuation rule not generally usable, and asked where a decision model could still help. The remaining fit: choosing a split point in long stretches without punctuation ("splitting long sentences"), which led to the Q1/Q2 split probe (section 4.6).
6. **Provider controls.** Soniox and Deepgram can finalize mid-stream. Gemini and Qwen needed their own approaches, informed by their issue history (section 6). This led to the Smart Turn evaluation, Qwen cut variants, Gemini end variants and the interim release replay (sections 4.5, 4.7, 4.8).

## 3. Tools, data and metrics

### 3.1 Scripts (all in this folder)

| File | Purpose | API |
| --- | --- | --- |
| `run.py`, `providers.py` | Stream probe audio in real time to each ASR provider and record every partial, final, finalize and word timestamp. Variants in `README.md`. | ASR providers |
| `analyze.py` | Turn a run folder into `summary.md/json`: sentence-end timings, CER, release diff, punctuation timing. | none |
| `fleurs.py`, `yodas.py` | Build read-speech and conversation probe audio and references. | downloads |
| `segmenter.py` | Silero VAD frames, Smart Turn scores at 200 ms pauses, and the cut schedules (`app`, `pause`, `smart`, `fixed3`) used by the Qwen and Gemini variants. | none |
| `smartturn_eval.py` | Label 200 ms pauses with Soniox punctuation and evaluate Smart Turn. | none |
| `decide.py` | Replay recorded partials through decision models in virtual time (marker choice + right-context verify). | OpenRouter |
| `split.py` | Long-utterance split probe (punctuation release + split policies after 3 s), with chunked translation and an LLM judge. | OpenRouter |
| `interim_release.py` | Replay Gemini runs with interim release at punctuation and final remainder. | none |
| `HANDOFF.md` | Instructions for the local agent (last used for the runs in section 4.6-4.8). | - |

### 3.2 Data

- FLEURS read speech (`ko_kr`, `ja_jp`, `en_us`): about 2 minutes per language, sentences joined with 0.2 s silence. Every boundary has a pause, which favors pause and punctuation rules.
- YODAS2 conversation (`ko_conv`, `ja_conv`, `en_conv`): 125 s from one video per language. Korean is two real-estate hosts with overlapping turns, Japanese is two creators with short turns and laughter, English is a three-host podcast with almost no pauses. References are uploaded caption lines, not sentences; Japanese captions differ from the speech, so Japanese CER (0.29-0.42) is mostly caption mismatch.

### 3.3 Labels and metrics to keep in mind

- Sentence and clause labels come from the Soniox final punctuation of the same audio (hindsight). They are noisy, and they make the punctuation-rule baseline partly circular for Soniox.
- Latency is measured from the speech end (Silero) or from the Soniox end timestamp of a character, depending on the script; each `summary.md` says which.
- Split quality: chunk-by-chunk translation (`google/gemma-4-26b-a4b-it`, the app's default, with the app's Peer prompt) versus whole-utterance translation, judged by `openai/gpt-6.1-sol` as `same`, `minor` or `changed`, plus a `premature` flag for a subtitle that misleads before the next chunk.

## 4. Results

### 4.1 ASR timing: finals are too slow, partials are fast

Read speech (`results/20261020T055613Z`), p50 seconds after the sentence's speech end:

| | Soniox | Deepgram | Gemini | Qwen Audio |
| --- | --- | --- | --- | --- |
| Final without forced finalize | 5.4-5.5 | 2.4-2.8 | 32-68 | 8-11 |
| Partial covers the sentence end | 0.41-0.51 | 0.28-0.78 | 0.33-0.62 | 0.53-1.27 |
| Partial shows terminal punctuation | 0.44-0.57 | 0.28-0.78 | 0.33-0.62 | 1.29-2.10 |
| Final after finalize in the pause | 0.33-0.35 | 0.31-0.40 | 0.47-0.50 | - |
| CER rise from finalize every 2.5 s during speech | +0.2-2.3 pt | +3.0-6.8 pt | +12-18 pt | - |

Conversation (`results/20261020T070225Z`), final p50 / p90 after the caption line end: Soniox 5.5-5.8 / 6.5-7.1 s, Deepgram 2.9-3.1 / 4.2-6.8 s, Gemini 3.7-12.0 / 12.2-44.0 s (7 finals in 125 s for Korean and English), Qwen Audio 7.0-15.1 / 25.4-27.7 s. Partial text covers each caption end 0.4-1.2 s after its speech end for all providers.

Other provider facts measured:

- Soniox finalize ack 0.21-0.27 s. Each finalize shifts Soniox token timestamps by 70-80 ms. Mid-speech finalize adds a spurious terminal mark at almost every finalize point (42-45 per run).
- Deepgram finals come in `is_final` chunks of about 4 s that can contain a sentence end and the start of the next sentence. Deepgram removed partial punctuation at 3 sentence ends in read speech.
- Gemini `interim_input_transcription` is cumulative within an utterance and resets after each final. Without a 400 ms pause, Gemini treated a whole recording as one utterance. It ignored 3 of 43 `audio_stream_end` requests at sentence ends.
- Whole-partial debounce does not work in continuous speech: a short window cuts mid-sentence constantly (Soniox 59-260, Deepgram 109-121, Gemini 182-223 mid cuts per 2 minutes at 0.2-0.3 s), a long window misses most sentence ends.

### 4.2 Partial stability (conversation)

Release at each terminal mark in the partial text, compared with the provider's own final for the same span (`results/20261020T070225Z/NOTE.md`). Cells: latency p50 / p90, chunks with any difference, CER.

| provider | W = 0 | W = 1.0 s (prefix unchanged for W) |
| --- | --- | --- |
| Soniox | 1.06 / 1.62 s, 1%, 0.001 | 2.18 / 2.73 s, 1%, 0.001 |
| Deepgram | 0.64 / 1.61 s, 35%, 0.050 | 1.73 / 2.72 s, 22%, 0.034 |
| Gemini | 0.68 / 1.02 s, 44%, 0.093 | 1.76 / 2.25 s, 34%, 0.083 |
| Qwen Audio | 2.10 / 3.62 s, 36%, 0.044 | 3.10 / 4.73 s, 31%, 0.038 |

- Soniox partial text at punctuation is effectively final.
- Revisions: Soniox 2% of updates, Deepgram 28%, Gemini interim 11%, Qwen Audio 32%. Revised text age p90: Soniox 0.96 s, Deepgram 1.42 s, Gemini 0.96 s, Qwen 2.35 s (max 4.2 s).
- Gemini finals are a separate decode: they differ from the last interim by CER 0.047-0.082 in Korean and English and 0.264 in Japanese.

### 4.3 Decision model as a boundary detector: negative

`results/decide-20261020T072017Z` (four models, four providers, six data sets). Conversation, pooled:

| policy | sentence | mid | recall | latency p50 / p90 |
| --- | --- | --- | --- | --- |
| punct rule | 82% | 11% | 89% | 1.00 / 3.35 s |
| pplx-decider-v1.2, immediate p>=0.5 | 52% | 35% | 48% | 2.82 / 11.17 s |
| pplx-decider-v1.2, verify v>=0.5 | 81% | 8% | 35% | 6.35 / 17.81 s |
| natural final | 46% | 44% | 29% | 5.07 / 17.85 s |

- Mercury, Jev and Flash were worse than pplx-decider-v1.2.
- The marker-choice question was the bottleneck: offered a true sentence end that already had words after it, models picked it in 5-43% of calls and mostly answered `none`.
- A replay design flaw (a boundary first offered at the transcript end was never re-offered) affected 12-25% of conversation labels.
- The right-context `noul` check does discriminate: AUC 0.87-0.88 for pplx-decider-v1.2, 0.81-0.83 Mercury, 0.77-0.79 Jev, 0.76-0.88 Flash. True boundaries get only 0.62-0.65 on average, so 0.5 is strict. Labels are noisy, so these are lower bounds.
- Waiting for 2-3 characters of right context plus a 0.57 s round trip puts a floor of about 2.0-2.3 s (Qwen 2.7-3.2 s) on any right-context policy.
- Calls and cost: 21,116 calls for about $0.20 in total. Per audio hour: Jev $0.158, Mercury $0.038, pplx-decider-v1.2 $0.030, Flash $0.017. RTT p50 0.38-0.57 s from the replay machine. Flash needed 152 retries under 24 parallel streams.

Latency ladder in conversation (p50, Soniox/Deepgram/Gemini, from the recorded logs):

| release point | latency |
| --- | --- |
| Punctuation rule (measured) | 0.5-1.1 s |
| Decision model asked when the partial ends | floor 1.3 s |
| Same, after 0.3 s without change | floor 1.5-1.6 s |
| Decision model after 2-3 characters of right context | floor 2.0-2.2 s |
| Soniox final | 5.5 s |

Why text-only boundary detection loses (the same failure as the user's earlier open-source sentence-end model):

- At the moment speech pauses, text alone cannot tell "집에 갔어" from "집에 갔어 근데…". Right context fixes this but costs about 2 s.
- The text model has no prosody. Pause length alone already separates conversational sentence ends from other pauses with AUC 0.68-0.74, and only 17-29% of conversational sentence ends have a pause of 0.5 s or more.
- ASR punctuation wins because it comes from a model that heard the audio: it is an acoustic boundary model whose output is free.
- Other measurements made for this question:
  - Using a decision model to reject false partial punctuation would cost more than it saves. False terminal marks occur about twice a minute (about 7% of releases) for Deepgram, Gemini and Qwen and almost never for Soniox. At v>=0.5 the model rejects 29% of true ends, and 9% at 0.2.
  - Replacing Qwen's late punctuation with a decision model gives no gain (floor 2.7-3.2 s versus 2.5-2.9 s).
  - BudouX (Japanese phrase splitter) misses 44% of sentence ends in unpunctuated conversation.

### 4.4 Is the punctuation rule used elsewhere?

Yes, usually with fallbacks (from a source review; links in section 9):

- AssemblyAI Universal-3.5/3.6 Pro: at 100 ms silence it checks for `. ? !`; at 1000 ms it ends the turn regardless.
- LiveCaptions-Translator: translates partial captions up to the last `.?!。？！`, with change-count and idle fallbacks.
- Azure "Semantic" segmentation and Speechmatics translation finalize at sentence-ending punctuation.
- Counterexamples: Soniox translation streams chunks without waiting for sentences; Google re-translation, wait-k, Seamless and SimulStreaming translate prefixes. VRChat tools VRCT and TaSTT use silence only.
- Known failure modes: partial punctuation that later disappears (AWS docs example), unbounded waits in run-on speech, long sentences, tiny fragments ("OK."), abbreviations and decimals.

The user has said the punctuation rule "cannot be used in general" but did not say why (see open questions).

### 4.5 Smart Turn at 200 ms pauses

`results/smartturn-20261020T185012Z` (offline).

- Only 35% of conversational sentence ends have a pause of at least 200 ms (Korean 24%, Japanese 42%, English 13%); read speech 27%. Any pause-only cut rule can release at most about a third of conversational sentence ends.
- AUC sentence end versus mid-sentence pause: 0.725 in conversation, 0.911 in read speech.
- Conversation at p >= 0.5: 9.3 cuts per minute, 67% at sentence ends, 9% at commas, 24% mid-sentence; sentence recall 0.26.

### 4.6 Long-utterance split probe (Q1/Q2), two runs

`results/split-20261020T190859Z` (Soniox, Gemini) and `results/split-20261020T194622Z` (Soniox, Deepgram, Gemini); interpretation in the second folder's `NOTE.md`.

Design: punctuation releases as usual. When no punctuation release happens for 3 s after new text starts, a policy may release an earlier part. A split costs 0.4 s of finalize allowance plus the decision round trip.

- Q1: "Does `before` end a sentence or self-contained clause?"
- Q2: "Would the {target} translation of `before` alone stay correct once `after` is heard?" (Korean to English, Japanese to Korean, English to Korean.)

Pooled, first run / second run:

| policy | latency long p90 | same | changed | premature |
| --- | --- | --- | --- | --- |
| punctuation only | 8.6 / 8.1 s | - | - | - |
| time cut (latest word) | 4.0 / 4.1 s | 17% / 20% | 31% / 36% | 8% / 17% |
| provider comma | 7.0 / 6.6 s | 57% / 52% | 17% / 24% | 4% / 3% |
| longest pause | 4.8 / 4.8 s | 44% / 37% | 16% / 27% | 12% / 10% |
| pplx-decider Q2 p>=0.5 | 5.1 / 5.5 s | 33% / 32% | 44% / 24% | 22% / 8% |
| pplx-decider Q2 p>=0.7 | 5.7 / 6.5 s | 50% / 48% | 35% / 23% | 12% / 6% |
| mercury-decide Q2 p>=0.5 | 5.0 / 5.1 s | 38% / 21% | 34% / 46% | 10% / 15% |

- No decision-model policy beat the simple rules in both runs, and Q2 results swung strongly between runs. Q1 alone rarely passes its threshold.
- Splitting barely moves p50 (about 3.4 s to 2.5-3.3 s) because a split cannot happen before 3.4 s; the gain is at p90 (8 s down to 4-5 s).
- Provider comma is the safest (3-4% premature) but rare. Longest pause gives the best balance of speed and safety.
- Japanese splits are judged `changed` much more often, but on 2-14 utterances.
- Each policy has 23-66 judged utterances pooled; per-language and per-provider cells are too small to compare.
- Cost: $0.33 and $0.43 per run, mostly the judge.

### 4.7 Qwen Audio 3.1 cut variants (live)

`results/20261020T185455Z/NOTE.md`. One task per segment on two alternating connections; cut schedules from `segmenter.py`. Cells: CER, then final p50 / p90 after the caption line end.

| variant | ko_conv | ja_conv | en_conv |
| --- | --- | --- | --- |
| natural (server endpointing) | 0.094, 14.8 / 25.1 s | 0.291, 6.5 / 27.5 s | 0.089, 10.7 / 23.9 s |
| cut_pause (200 ms pause after 2 s) | 0.105, 2.9 / 7.9 s | 0.313, 2.5 / 6.3 s | 0.083, 3.6 / 8.7 s |
| cut_smart (Smart Turn >= 0.5 at 200 ms) | 0.109, 3.2 / 7.9 s | 0.328, 2.7 / 13.9 s | 0.080, 4.0 / 8.7 s |
| cut_smart_ctx (+ previous text as context) | 0.107, 3.1 / 7.8 s | 0.315, 3.2 / 6.9 s | 0.064, 3.8 / 8.1 s |
| cut_app (approximation of current rules) | 0.135, 3.1 / 5.7 s | 0.330, 3.1 / 8.2 s | 0.079, 2.6 / 4.7 s |
| cut_fixed3 (every 3 s, also during speech) | 0.216, 2.2 / 3.8 s | 0.372, 2.7 / 6.4 s | 0.106, 1.9 / 3.1 s |

- Pause cuts bring Qwen finals from 6.5-14.8 s to 2.5-4.0 s (p50) with CER within about 2 points of natural.
- Smart Turn cuts are not better than any 200 ms pause after 2 s.
- Cuts during speech lose roughly 1-2 characters each (Korean `deleted` 104 with 3 s cuts versus 44 with pause cuts). The 6 s cut with 300 ms replay also duplicates and loses text in Korean.
- p90 stays at 6-9 s: English has few 200 ms pauses (median `cut_pause` segment 9.0 s; 8 of 17 segments end at the 10 s fallback).
- `finish-task` to `task-finished`: 0.25-0.45 s p50. No task failed. `usage` is not reported.
- Context (`payload.input.context`, last 400 characters of earlier tasks, format borrowed from Fun-ASR docs): accepted without error. The first task (no context) gives identical text in both runs; later tasks differ, so the server uses it. Deleted characters fell in all three languages (44 to 37, 91 to 67, 51 to 28). One run each.
- Releasing Qwen partial text at punctuation inside cut tasks (offline replay): Korean CER rises 5-10 points (unsafe); Japanese gains nothing; English gains about 1 s at p50 and 2-3 s at p90 with no CER change.

### 4.8 Gemini end variants and interim release

`results/20261020T185455Z/NOTE.md` and `results/interim-20261020T190834Z/NOTE.md`. Cells: CER, deleted characters, final p50 / p90.

| variant | ko_conv | ja_conv | en_conv |
| --- | --- | --- | --- |
| natural (no `audio_stream_end`) | 0.111, 17, 8.5 / 43.9 s | 0.349, 95, 3.1 / 12.1 s | 0.093, 39, 8.6 / 40.5 s |
| end_app (every 4-6 s cut, current app) | 0.126, 32, 2.4 / 4.7 s | 0.355, 102, 2.6 / 12.1 s | 0.101, 64, 2.6 / 5.1 s |
| end_smart (Smart Turn cuts and 800 ms pauses) | 0.112, 23, 3.5 / 11.5 s | 0.370, 85, 2.5 / 9.2 s | 0.081, 37, 6.8 / 21.9 s |

- The current behavior loses words (+15 Korean, +25 English deleted characters). Finalize acknowledgements sometimes take seconds (Japanese p90 4.7-9.3 s).
- Interim release (terminal mark in interim, prefix unchanged for 0.6 s, final used only for the remainder), per-character latency p50 / p90 and CER:

| data | finals only | interim, run 070225Z | interim, run 185455Z |
| --- | --- | --- | --- |
| ko_conv | 8.2-10.4 / 43 s, 0.111 | 2.3 / 4.6 s, 0.119 | 2.1 / 4.3 s, 0.130 |
| ja_conv | 2.2-2.4 / 4.6-7.3 s, 0.349 | 1.8 / 4.2 s, 0.420 | 1.7 / 3.3 s, 0.341 |
| en_conv | 10.7-10.8 / 43.6 s, 0.093 | 3.9 / 11.1 s, 0.099 | 3.6 / 10.9 s, 0.091 |

- Gemini finals had identical CER across the two runs; interim text did not. The +7 point Japanese cost of the first run did not repeat. Treat the interim cost as 0-2 points with occasional bad runs.
- English p90 stays near 11 s because run-on speech has no terminal marks in interim text; it needs splitting.
- Interim release combined with `end_app` is bad (CER 0.180 Korean, 0.130 English); with `end_smart` it behaves like natural.

## 5. Answers so far

### 5.1 Does a decision model as a turn detector improve Peer latency?

Not as a boundary or turn detector. In every test the ASR's own punctuation was faster and more accurate, and a right-context check has a structural floor near 2 s. Cost and API latency were never the issue.

Where a decision model could still fit, and what the data says:

| role | status |
| --- | --- |
| Find sentence ends in partial text | Rejected (4.3) |
| Filter false partial punctuation | Rejected: rejects too many true ends (4.3) |
| Replace Qwen's late punctuation | Rejected: no faster (4.3) |
| Choose a split point after 3 s without punctuation | Not better than comma or longest pause in two runs; noisy (4.6) |
| Choose a Qwen cut point in real time | Not possible: Qwen text arrives 1-2 s late, so audio cues must decide (4.7) |
| Shadow questions (filler, ASR hallucination, needs previous context) | Not tested |

### 5.2 Early release for utterances of 3 s or more

Per provider, the best measured path:

| provider | sentence release | long stretches without punctuation | measured p50 (sections 4.2, 4.7, 4.8) |
| --- | --- | --- | --- |
| Soniox | Partial text at terminal marks, no wait (partial is effectively final) | Comma, then longest pause | about 1 s |
| Deepgram | Partial at terminal marks with a short stability wait, or Finalize only in a pause | Same | 0.6-1.7 s depending on the wait |
| Gemini | Interim at terminal marks, 0.6 s stable; finals only for the remainder; `audio_stream_end` only at real turn ends | Same, on interim text | 1.7-2.3 s (English 3.6-3.9 s) |
| Qwen Audio 3.1 | Task finals; cut tasks at pauses (200 ms after 2 s) and pass previous text as context; English may also release partial at terminal marks | No good answer yet (p90 6-9 s) | about 3 s |

The latency definitions differ slightly between sources (speech end of a sentence, caption line end, or Soniox character end), so compare rows only roughly. Translation time is not included.

## 6. Lessons from the Gemini and Qwen issue history

Gemini (redesigned four times between 08-28 and 10-01: #102/#120, #134/#135, #174, #194/#205):

- Most churn came from assuming "one local segment = one Gemini final". Finals have no ordering guarantee, no offsets and no IDs; they can arrive early, late, merged, or after successor audio (#212).
- Finals can drop content that interim had ("A+B in interim, only B as final", #218). Manual activity mode did not fix it (#264).
- Mid-speech cuts lose words (#194). Manual activity also produced ActivityEnd without a transcript on silence (#174).
- Reconcile by text, not by correspondence.

Qwen Audio (three structures in seven weeks: per-segment task #105, continuous task #255 v1, two-slot per-segment task #255 v2 / PR #257):

- A continuous task with server endpointing gave a first final at 9.05 s and still split sentences at internal pauses; it was rejected the same day.
- `finish-task` ends the task; there is no flush-and-continue. Warm task start about 56 ms; final 0.13-0.5 s after `finish-task`; idle connections expire at 60 s (replaced at 55 s).
- No context carries across tasks by default; the 300 ms replay duplicated words ("No, three." became "No, no, three.").
- Estimated cost model: about 61 fixed input tokens per task plus about 13 per audio second (four samples, unconfirmed), so frequent cuts have a token overhead.
- Gains in scripted tests did not transfer to the live service.

Common: do not pair provider finals with local segments; avoid cutting during speech; confirm offline gains on the live service.

## 7. Proposed Peer structure (hypothesis, not validated end to end)

```text
audio ─ acoustic layer (kept): VAD, Smart Turn turn end, speaker change (#211), safety ceiling
      └ ASR stream (not cut on time rules for Soniox/Deepgram/Gemini)
           └ text layer (new): release up to a terminal mark in partial/interim text
                               ├ 3 s without a release → comma, else longest pause
                               ├ speaker change or long silence → release immediately
                               └ final text reconciles the remainder by text alignment
```

- Smart Turn: keep for turn ends only. It adds nothing over a plain 200 ms pause for mid-utterance cuts.
- 4 s/192 ms and 5 s/128 ms steps: drop for Soniox, Deepgram and Gemini; for Qwen replace with "200 ms pause after 2 s".
- 6 s hard cut: remove for Gemini (send `audio_stream_end` only at turn ends). For Qwen, do not cut during speech except a 10 s safety cut, and drop the 300 ms replay.
- Decision model: off the common path. Re-test only if a better split question or more data changes 4.6.

## 8. Open questions and doubts

Measurement validity:

1. Run-to-run noise is unknown for most results. Every live variant ran once; Gemini interim CER moved by up to 8 points between two runs, and Q2 `changed` moved from 44% to 24%. Repeat runs are needed before trusting differences of a few points.
2. Three conversation videos (one per language) are a small and specific sample: overlapping Korean hosts, laughing Japanese creators, a run-on English podcast. VRChat speech (short turns, background noise, loopback audio) may behave differently.
3. References are caption lines, and labels are Soniox punctuation. A human or strong-LLM labeled boundary set ("translatable here") would make the boundary and split comparisons fairer.
4. The `app` cut policy in `segmenter.py` approximates the application (no hangover modeling, simplified pre-roll). The current application's real Peer latency on the same audio was never measured (an offline run of `ListenDeliveryController` would do it without API cost).
5. Latencies exclude translation time and display. End-to-end "word spoken to subtitle shown" was never measured.
6. The split judge is one model with no calibration against human judgments, and there is no baseline for translation noise (how often the same text translated twice is judged `changed`).

Product and design questions:

7. Why does the user consider the punctuation rule not generally usable? Partial punctuation that disappears, providers without punctuation, run-on speech, or something else? The answer decides what a fallback must cover.
8. Which open-source sentence-end model was used in the earlier simultaneous-interpretation experiment, and did it fail by missing ends, being late, or cutting wrongly?
9. How should released partial or interim text be reconciled when the final differs (correct the shown subtitle, ignore, or append)? Gemini finals can drop or reword content; Deepgram partials differ in 22-35% of chunks.
10. Is 3 s the right trigger for splitting, and how should premature subtitles be weighed against latency? A split cannot help p50 below about 3.4 s.
11. Short replies and fillers ("네", "うん", "yeah"): merge, show, or skip? Not measured.
12. Privacy and operations for any decision-model use: Peer text goes to a new third party (opt-in, `PRIVACY.md`); Perplexity direct calls are limited to 10 requests per second per organization.

Provider questions:

13. Qwen long utterances without 200 ms pauses (p90 6-9 s): would age-dependent pause thresholds (for example 120 ms after 4 s, 80 ms after 5 s) or overlapping tasks stitched by Qwen word timestamps help without word loss?
14. Qwen context: is the borrowed `payload.input.context` format officially supported for Qwen Audio 3.1, and is its benefit repeatable? In `cut_smart_ctx`, 51 of 81 tasks (63%) started before the previous task's final arrived, so their context lacked the immediately preceding segment. The measured benefit therefore comes mostly from older text. Waiting for the previous final (about 0.3 s) and then sending the buffered audio as a burst would close that gap at a small latency cost.
15. Qwen cost per task (about 61 fixed tokens) was not visible in `usage` for this model; the real billing overhead of frequent cuts is unconfirmed.
16. Gemini `silence_duration_ms` below 400 ms was never tested. Does it give earlier finals without more fragmentation or dropped content?
17. Deepgram Finalize only when a terminal mark coincides with a pause of 150-200 ms: not yet run.
18. Soniox frequent finalize: Soniox docs warn that very frequent finalize can drop the connection; the proposed design avoids finalize during speech, but the safe frequency was not measured.

## 9. Next experiment candidates

| # | Experiment | Cost | Decision it informs |
| --- | --- | --- | --- |
| A | End-to-end offline replay of the proposed structure per provider (punctuation, comma, longest pause, Qwen pause cuts) on the recorded runs, including a translation-time estimate | none | Whether the combined pipeline meets the 3-6 s goal |
| B | Run the current `ListenDeliveryController` offline on the same audio to get the real application baseline | none | Size of the improvement |
| C | Repeat the key live variants 2-3 times (Qwen `cut_pause`, `cut_smart_ctx`; Gemini natural and `end_smart`) | about 40 min Qwen audio, 20 min Gemini | Separates real effects from noise |
| D | Qwen long-utterance variants: age-dependent pause thresholds, overlap + timestamp stitching, context wait | about 20 min Qwen audio | Qwen p90 |
| E | Deepgram Finalize at "terminal mark + pause"; Gemini `silence_duration_ms` 150-200 ms | about 15 min audio | Provider-specific release paths |
| F | Boundary labels from a strong LLM over full transcripts, then re-score split policies | a few dollars | Fair split comparison |
| G | More conversation data (other videos, ideally VRChat-like audio without private data) | depends | Generality |

## 10. Practical notes

- `README.md` lists every command, variant and output file. `HANDOFF.md` shows the format that worked for the local agent (exact commands, expected time and billed audio, "do not edit scripts", "an error column is a result, not a crash", push only `results`).
- `analyze.py` now starts the first partial match at the beginning of the reference; before that fix, a short misrecognized first word could match far ahead and corrupt partial columns (`-19 s` values in the Qwen cut rows).
- Alignment pitfalls: use semi-global alignment (`_semi_global_end`) for prefix-versus-full text; global Levenshtein opcodes are ambiguous there. Lowercase and strip punctuation before sending text to decision models, or capitalization leaks boundaries.
- Lint: `black` 26.5.1 and `ruff` 0.16.1 pass on the folder.

## 11. Sources

- Decision models: [OpenRouter Decisions API (Jev guide)](https://openrouter.ai/docs/guides/community/jev), [Mercury Decide](https://openrouter.ai/inception/mercury-decide-20260930), [pplx-decider-flash-v1.2](https://huggingface.co/perplexity-ai/pplx-decider-flash-v1.2), [Perplexity Decisions quickstart](https://docs.perplexity.ai/docs/decisions/quickstart).
- Provider controls: [Soniox manual finalization](https://soniox.com/docs/stt/rt/manual-finalization), [Deepgram Finalize](https://developers.deepgram.com/docs/finalize), [Gemini Live API](https://ai.google.dev/api/live), [Gemini live transcription](https://ai.google.dev/gemini-api/docs/live-api/live-transcribe).
- Punctuation in practice: [AssemblyAI Universal-3.5/3.6 Pro streaming](https://assemblyai.com/docs/streaming/migration-guides/universal-to-universal-3-5-pro-streaming), [LiveCaptions-Translator](https://github.com/SakiRinn/LiveCaptions-Translator), [Azure segmentation](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/how-to-recognize-speech), [Speechmatics translation](https://docs.speechmatics.com/speech-to-text/features/translation), [Soniox real-time translation](https://soniox.com/docs/stt/rt/real-time-translation), [AWS partial results](https://docs.aws.amazon.com/transcribe/latest/dg/streaming-partial-results.html).
- Research: [Cho, Niehues, Waibel 2017](https://isl.iar.kit.edu/english/NMT-Based_Segmentation.php), [Iranzo-Sánchez et al. 2020](https://aclanthology.org/2020.emnlp-main.206/), [Oda et al. 2014](https://aclanthology.org/P14-2090.pdf), [Arivazhagan et al. 2020](https://arxiv.org/abs/1912.03393), [STACL / wait-k](https://arxiv.org/abs/1810.08398), [Seamless Streaming](https://arxiv.org/abs/2312.05187).
- Data: [espnet/yodas2](https://huggingface.co/datasets/espnet/yodas2), FLEURS (CC-BY-4.0).
