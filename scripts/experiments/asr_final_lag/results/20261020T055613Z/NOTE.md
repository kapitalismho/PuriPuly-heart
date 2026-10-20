# Notes on this run

## Input

- Silero-trimmed FLEURS read speech with 0.20 s of silence at every sentence boundary.
- Every boundary has a pause and every sentence is complete and well formed. This favors pause-based and punctuation-based boundary detection over what conversational Peer speech would give.

## Changes made after the run

- `summary.md` and `summary.json` were regenerated with a later analyzer that adds `punct (missing)` and `extra ends`. `runs/`, `reference/` and `meta.json` are unchanged.

## Provider behavior seen in the runs

- Gemini `interim_input_transcription` is cumulative within an utterance and resets after each final. Without a 400 ms pause, Gemini `natural` treated the whole recording as one utterance and sent 2 finals.
- Gemini ignored 3 of 43 `audio_stream_end` requests sent at sentence ends (en_us 1, ko_kr 2). The final for that sentence arrived only after the next `audio_stream_end`, 5-10 s later.
- Gemini `audio_stream_end` every 2.5 s during speech dropped words: 70-215 deletions per language against 4-20 in `natural`, and CER rose by 0.12-0.18.
- Soniox token timestamps advance by about 80 ms per finalize request, as in the first run.
- Deepgram removed sentence-end punctuation that its partials had shown at 3 sentence ends (ja_jp 2, ko_kr 1). Soniox, Gemini and Qwen Audio kept it.

## Measurement artifacts

- English `punct (missing)` and Deepgram ko_kr `finalize_gap` outliers come from text alignment, not provider timing: "3:2" was transcribed as "three to two", and "802.11g" as "팔 백 이 점 일 일 g".
- Deepgram ko_kr `finalize_gap` sentence 6: the final returned for the finalize lacked the sentence's last syllable, so coverage completed only with a later final.
