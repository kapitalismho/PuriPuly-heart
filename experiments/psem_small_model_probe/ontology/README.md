# ONTOLOGY16 downstream challenge (Gate 6, issue #117)

> **Dev-only EVAL note:** everything under this directory is a frozen-gate
> experiment harness. It never runs in production, never touches the live
> pipeline, and its outputs (`results/*.jsonl`) are gitignored. Do not import
> from `ontology/` outside this gate.

## Pipeline (substitution, explicit)

The live pipeline (local ASR provider runtime under
`src/puripuly_heart/core/local_asr/` + translation turn ownership under
`src/puripuly_heart/core/orchestrator/translation_turn.py`) is GPU-oriented and
no CPU model is readily runnable here: `sherpa_onnx` is installed but no model
weights are cached, the HF hub cache is empty, translation needs an LLM owner,
and the corpus root carries audio only (no word-level GT transcripts). Adding
either would be a heavy new dependency, so this gate substitutes a
deterministic GT-interval-grounded simulation:

- Same audio spans for K and T via `PSEM_CORPUS_ROOT` + `cal/audio_resolve.py`
  (`load_span` over each evaluation window; PCM sha recorded per episode).
- Same deterministic scoring pass applied to both renderings under blind X/Y
  masking (`random.Random(117)` shuffle; `blind_id` stored in each render row;
  direction-agnostic comparator; mapping revealed only in `decision.md`).
- GT-derived frontiers only: T's cut F = first ms in `[T_trans-2000, T_trans]`
  with non-anchor activity (overlap onset); stratum from manifest topology
  (`A->A+B->B` = C4, else C3).

## Renderings / metrics / rule

- K (Simple Anchor): old `[win_start, T_trans)` -> anchor; new -> NEXT single
  speaker at/after `T_trans` (NONE if absent). T (takeover): old
  `[win_start, F)` -> anchor; new `[F, win_end)` -> dominant other speaker B.
- (a) `contam_old_ms`: non-attributed-speaker active ms in old segment.
- (b) `loss_risk`: cut straddles an utterance (any speaker active on both sides
  within +/-200 ms); orphaned/duplicated ms are 0 by partition construction.
- (c) `attr_prec_new`: fraction of speech-active ms in the new segment where
  the attributed speaker is active (translation-coherence proxy: garbage
  attribution in => garbage translation out).
- T-better <=> contam reduction >= 50 ms AND no loss-risk increase AND
  precision gain. Counts: >=4 reopen ownership; <=1 retain Simple Anchor; 2-3
  diagnostic/HOLD-only, no new primitive.

## Run

```cmd
set PSEM_CORPUS_ROOT=C:\Users\salee\.psem-corpus
PYTHONPATH=. python experiments/psem_small_model_probe/ontology/run_ontology.py
```
