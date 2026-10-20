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
