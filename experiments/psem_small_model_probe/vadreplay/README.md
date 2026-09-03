# Gate 5 — production VAD replay (issue #117)

Dev-only EVAL probe: V2 EVAL sessions reused per program approval; no
unbiased generalization claim; V3 fresh holdout required for selection claims.

## What

Replays the ONE causal winner — firered x regime C at frozen tau=0.05 —
with the speech gate swapped from GT anchor speech to production Silero VAD
spans (peer profile: threshold 0.5, 512-sample chunks, pre-roll 500 ms,
hangover 500 ms, max_segment 7000 ms; gate = pre-roll + committed chunks
through speech end excluding trailing hangover). Same 48 MAIN48 rows, same
causal 1 s bind, same `CommonPersistenceDecoder` (500 ms confirmer / 300 ms
sensitivity), same headline metrics. neovad is NOT replayed (collapsed 8/8
missed under the GT gate; nothing to retain).

## Run

```bat
set PSEM_CORPUS_ROOT=C:\Users\salee\.psem-corpus
set PYTHONPATH=<repo-root>
.venv\Scripts\python.exe experiments/psem_small_model_probe/vadreplay/run_replay.py
```

~55 s on the reference machine (6 sessions, full-source VAD + 48 causal binds).

## Files

- `run_replay.py` — the gate. Imports metrics/span loading/GT helpers/frozen
  taus from `main`/`cal` (never reimplemented); imports the production VAD
  engine + peer gating factory from `src` (read-only).
- `results/summary.md` — GT-gate vs prod-VAD side-by-side, gate agreement,
  retained-improvement fraction, verdict, CPU note.
- `results/replay.jsonl` — per-step gates + episode headers (gitignored).
- `results/replay_summary.json` — machine-readable payload incl. per-episode
  GT/prod outcomes and CUT-hit id sets (gitignored).

## Reading the verdict

- GT good / prod bad → VAD timing/gating problem (do NOT fine-tune the
  observation model).
- Both good → integration clean.
- Both bad → observation model is the bottleneck.

Result: both good — 3/3 GT-detected CUT episodes retained under prod-VAD,
missed 5/8 → 1/8; false cuts 13 → 21 because the prod gate (any-speech) is
wider than the GT anchor gate. See `results/summary.md`.
