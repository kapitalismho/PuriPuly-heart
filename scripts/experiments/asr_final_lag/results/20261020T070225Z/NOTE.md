# Notes on this run

## Input

- 125 s YODAS2 windows from one video per language (`ko_conv`, `ja_conv`, `en_conv`), `natural` variant only.
- Reference units are uploaded caption lines, not sentences. Caption ends are not sentence ends, so `punct (missing)` and the partial-decision table do not apply. English captions are long run-on lines, which is why 41-42 of 52 English caption ends show no punctuation.
- Japanese CER (0.30-0.41) is mostly caption text that differs from the speech and kanji/kana differences, not recognition errors.

## What the run shows

- Natural finals in conversation arrive much later than partials. Final p50 / p90 after the caption end:
  - Soniox: 5.5-5.8 / 6.5-7.1 s.
  - Deepgram: 2.9-3.1 / 4.2-6.8 s.
  - Gemini: 3.7-12.0 / 12.2-44.0 s, with 7 finals in 125 s for Korean and English.
  - Qwen Audio: 7.0-15.1 / 25.4-27.7 s, with 6-10 finals in 125 s.
- Partial text covers each caption end 0.4-1.2 s (p50) after its speech end for all providers.
- Sentence boundaries for the decision model replay come from the Soniox final punctuation of this run (see `../decide-20261020T072017Z/NOTE.md`).

## Partial-text stability (added after the decision replay)

Text was released at each sentence-final punctuation mark in the partial text and compared with the same span of the provider's own final text. With an agreement window W, the partial prefix had to stay unchanged for W seconds before release.

- Latency runs from the Soniox end timestamp of the chunk's last character.
- `differ` counts chunks with any character difference. It includes chunk-boundary alignment shifts, so it is an upper bound on word revisions.
- Cells show latency p50 / p90, `differ` and CER.

| provider | W = 0 | W = 0.6 s | W = 1.0 s |
| --- | --- | --- | --- |
| Soniox | 1.06 / 1.62 s, 1%, 0.001 | 1.67 / 2.22 s, 2%, 0.003 | 2.18 / 2.73 s, 1%, 0.001 |
| Deepgram | 0.64 / 1.61 s, 35%, 0.050 | 1.25 / 2.22 s, 33%, 0.048 | 1.73 / 2.72 s, 22%, 0.034 |
| Gemini | 0.68 / 1.02 s, 44%, 0.093 | 1.32 / 1.77 s, 33%, 0.085 | 1.76 / 2.25 s, 34%, 0.083 |
| Qwen Audio | 2.10 / 3.62 s, 36%, 0.044 | 2.70 / 4.22 s, 35%, 0.043 | 3.10 / 4.73 s, 31%, 0.038 |

- Soniox partial text at punctuation is effectively final.
- Sampled Deepgram, Gemini and Qwen Audio differences mix real word revisions with boundary shifts. Real revisions include "修正" → "修練" (Deepgram), "살찐" → "살집" (Gemini) and "やった" → "言った" (Gemini).
- Mid-speech finalize in the FLEURS run (`20261020T055613Z`, `finalize_mid`) raised CER by 0.2-2.3 points for Soniox, 3-7 points for Deepgram and 12-18 points for Gemini. Soniox also added terminal punctuation at almost every finalize point (42-45 extra ends per run).
