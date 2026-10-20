# Handoff: Peer early-release probes (local run)

Branch: `claude/tender-wright-4x2d4y`. Run the three steps below in order from the repository root, then push the results. A cloud session analyzes them.

Already recorded, no need to rerun:

- `results/smartturn-20261020T185012Z`: Smart Turn scored at 200 ms pauses (offline).
- `results/interim-20261020T185034Z`: Gemini interim release replay of the earlier runs (offline).

## Rules

- Keys come from `.env.local`: `ALIBABA_API_KEY_BEIJING` (Qwen Audio), `GOOGLE_API_KEY_2` (Gemini), `SONIOX_API_KEY` and `OPENROUTER_API_KEY`. Never print, paste or commit key values; the scripts refuse to write results that contain a key.
- Do not edit the scripts or provider settings. If a step crashes, stop and report the error message and traceback (without key values).
- A variant that ends with an `error` column in `summary.md` is a result, not a crash. Report it and continue.
- Commit only `scripts/experiments/asr_final_lag/results`.

## Step 1: Qwen Audio cuts and Gemini ends (live, about 15-20 minutes)

```powershell
git pull
uv run scripts/experiments/asr_final_lag/run.py --data ko_conv ja_conv en_conv --providers qwen gemini soniox --variants natural cut_app cut_pause cut_smart cut_fixed3 cut_smart_ctx end_app end_smart --parallel 3
```

- Answers how much a Qwen Audio cut costs in accuracy and how fast its final comes back, and whether Gemini loses words when `audio_stream_end` is sent only at Smart Turn ends instead of the current cuts.
- Qwen Audio runs one task per segment on two connections, like the application. Soniox `natural` is included as the timing reference for step 2.
- Billed audio is about 37 minutes for Qwen Audio, 19 minutes for Gemini and 6 minutes for Soniox.
- `cut_smart_ctx` sends ASR context in a format that is not confirmed for Qwen Audio 3.1. A `task-failed` note on that variant is an expected possible outcome.
- Check the printed `[run]` lines and the "Cut and end variants" table in the new `results/<timestamp>/summary.md`.

## Step 2: Gemini interim replay with the new runs (offline, about 1 minute)

```powershell
uv run scripts/experiments/asr_final_lag/interim_release.py
```

- Answers how early and how accurately Gemini interim text can be released when finals are only used for the remainder, including the `end_smart` and `end_app` runs from step 1.

## Step 3: long-utterance split probe (OpenRouter, about 15 minutes, about 1.5 USD)

```powershell
uv run scripts/experiments/asr_final_lag/split.py --providers soniox deepgram gemini
```

- Answers whether a decision model, with the sentence question (Q1) or the translation question (Q2), can split utterances that run 3 s or more without punctuation, judged by chunked translation quality.
- It checks each model with one small request first and stops on an authorization or not-found error.

## Push

```powershell
git add scripts/experiments/asr_final_lag/results
git commit -m "Record Qwen cut, Gemini end and long-utterance split probes"
git push origin claude/tender-wright-4x2d4y
```

## Report back

- The new results folder names.
- Any crash, `error` column, `task-failed` note or `[check] ... stopped` line, quoted as printed.
- No analysis is needed.
