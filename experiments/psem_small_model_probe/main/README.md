# MAIN48 Gates 3+4 — native-ceiling (O) + causal (C) scoring (issue #117)

Scores all 48 MAIN48 rows x {firered, neovad} x {O, C} with the EXACT frozen
Gate 2 taus (`../cal/results/thresholds.json`, copied to
`results/thresholds_frozen.json`). No retuning, no MAIN48 viewing for tuning.

## Run

```bat
set PSEM_CORPUS_ROOT=C:\Users\salee\.psem-corpus
PYTHONPATH=. .venv/Scripts/python experiments/psem_small_model_probe/main/run_main.py
```

Options: `--adapters firered neovad`, `--regimes O C` (O always runs first;
if BOTH models collapse on O — missed_rate == 1.0 on the CUT set at the
frozen tau — the runner FAILS FAST and skips C). Refuses on manifest sha
mismatch. Reads only: `../cal/*`, `../manifest/*`, `../adapter/*`.

## Method (Gate 2 pattern, reused)

GT speech gates every frame before the decoder; regime C rows with
`causal_bindable=false` stay UNBOUND (all MAIN48 rows are bindable, so this
is a no-op here). The 500 ms confirmation decoder emits headline CUTs while
a 300 ms sensitivity stream is recorded alongside (reversal detection only,
no frontier). Per-step JSONL carries wall-clock `step_ms` around
`adapter.step()` only.

## Outputs (`results/`; `*.jsonl` gitignored)

- `{model}_{regime}_main.jsonl` — 48 episode headers + 24000 step lines
- `{model}_{regime}_calibration.jsonl` — 19 per-tau replay aggregates
  (frozen tau flagged; replay only, no selection)
- `thresholds_frozen.json`, `cpu.json`, `summary.md` (tracked)

Headlines: contamination s/active-speech-h, false cuts, missed replacement
rate, delay p50/p90 SPLIT source-boundary error vs decision delay; mandatory
A->A+B->A KEEP vs A->A+B->B CUT views (MAIN48 has zero A->A+B->A rows —
reported as n=0; KEEP rests on A/overlap_return/A+A+B). Frame AUPRC/F1,
unbound fraction, role-flip: diagnostics only.

CPU hard gate: p99 step < 10 ms chunk, RTF <= 0.25, no backlog.
