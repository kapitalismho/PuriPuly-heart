# Gate 2 CAL12 threshold runner scaffold (issue #117)

> V2 EVAL sessions reused as dev-only probe per program approval;
> no unbiased generalization claim; V3 fresh holdout required for selection claims.

## Layout

- `audio_resolve.py` — manifest row -> on-disk WAV (no copying into the repo),
  exact 16 kHz/ms sample math, fail-closed spans, waveform re-hash.
- `run_cal.py` — CAL12 runner: adapters x regimes (O native 5 s, C causal 1 s)
  x episodes -> per-step JSONL + per-episode headers + `thresholds.json` +
  `summary.md` under `results/`.
- `metrics.py` — four headline metrics, KEEP/CUT topology views, fixed-priority
  single-scalar threshold rule.

## Env vars

| var | role |
|---|---|
| `PSEM_CORPUS_ROOT` | primary corpus root (`<root>/<audio_ref>`) |
| `PSEM_REFERENCE_ROOT` | fallback root when the primary is unset/missing |
`audio_ref` comes from the row, else the cached V2 GT interval sources
(`psem_relative_occupancy_gate/results/{dev,eval}/...`), else the corpus
convention (`ami/audio/<SID>/<SID>.Mix-Headset.wav`,
`alimeeting/far_ch0/<SID>.wav`). Missing env/paths raise, naming what is
missing. Audio is never copied into the repo.

## Run

```bash
PYTHONPATH=. python experiments/psem_small_model_probe/cal/run_cal.py --dry-run   # no audio needed
PYTHONPATH=. python experiments/psem_small_model_probe/cal/run_cal.py             # real audio via env roots
PYTHONPATH=. python experiments/psem_small_model_probe/cal/run_cal.py --adapters stub
```

The runner verifies `manifest.jsonl` sha256 against `dataset_freeze.json`
and refuses on mismatch. Adapters import lazily; missing weights fall back
to `StubAdapter` flagged `stub_fallback` in every header. Regime C rows with
`causal_bindable=false` stay UNBOUND (HOLD, no inference). `results/*.jsonl`
is gitignored; `thresholds.json` + `summary.md` are always written.

## Rules enforced here

- Single scalar per model x regime; no topology/corpus/episode-specific taus,
  no model-specific persistence; frozen after writing `thresholds.json`.
- Headline metrics only: contamination s/speech-h, false cuts, missed rate,
  replacement-delay p50/p90 split (source-boundary error vs decision delay).
  Frame AUPRC/F1, unbound fraction, role-flip: diagnostics only.
- No training, no MAIN48/EXT24 scoring, no dependency changes.
