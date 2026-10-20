# Notes on this run

## Input audio did not have continuous speech

- The probe version used for this run trimmed clips with an energy threshold that kept leading and trailing non-speech.
- Measured silence between sentences: en_us 1.06 s p50 / 2.10 s p90, ja_jp 3.25 / 4.24 s, ko_kr 1.27 / 2.66 s. The intended gap was 0.15 s.
- The run therefore measures sentences followed by long pauses, not continuous Peer speech.
- Natural finals that end on silence are optimistic here: Qwen Audio (`max_sentence_silence=800`) and Gemini (400 ms activity silence) had a pause after almost every sentence.

## Changes made after the run

- `reference/*.json` gained `speech_start_s` and `speech_end_s` from the bundled Silero VAD model, computed on the rebuilt input audio. The rebuilt references matched the committed ones exactly before the fields were added.
- `summary.md` and `summary.json` were regenerated with the updated analyzer. Sentence-end timings now start at `speech_end_s` instead of the clip end.
- `runs/` and `meta.json` are unchanged.

## Columns that do not measure what their names say

- `finalize_gap` sent finalize 0.10 s after the clip end, which was 0.0-3.0 s after the speech end. Its `final` timings and the `finalize at gap` column do not show a pause-aligned finalize.

## Observations that do not depend on the pauses

- Finalize acknowledgement: 0.20-0.29 s p50 for Soniox and Deepgram.
- Partial text covers a sentence end 0.4-0.5 s (Soniox), 0.5-0.8 s (Deepgram) and 0.6-0.9 s (Qwen Audio) after the speech end, and does not retract afterwards.
- `release diff`: the partial text of a sentence at that time matched the final text in all but 0-1 sentences per language for Soniox and Deepgram (at most 0.2% of characters). Qwen Audio differed in 2-4 sentences (1-3%).
- Soniox without endpoint detection finalizes about 5.6 s after the speech end, even with long pauses.
- Deepgram `nointerim` and `natural` have the same final timing.
- Finalizing every 2.5 s raised CER by 0.007-0.015 for Soniox and 0.015-0.055 for Deepgram.
- Soniox token timestamps advance by about 70-80 ms per finalize request: +1.0 s after 13 finalizes and +3.8 s after 49.
- Qwen Audio `semantic` ko_kr CER 0.268 comes from one hallucinated list of language names inside one sentence; the transcript recovered afterwards.
