# Next-Agent Handoff: Legacy-Only Frozen Representation Screen

## 1. Mission

Continue the experiment from `R2-L`, using only the already available ERes/LS-EEND common-GT
data. Complete the reduced frozen representation screen through candidate selection without
training a model.

```text
R2-L legacy validation and reduced cost forecast
  -> owner approval
  -> reduced R3 representation probe
  -> reduced R4 continuous zero-shot SCD
  -> candidate selection
```

Do not acquire or use Zeroth-Korean, JVS, D5, or another new corpus.

## 2. Read First

1. Repository `AGENTS.md`
2. `.agents/goals/goal-experiment-plan-en/STATE.md`
3. `experiments/speaker_representation_scd/EXPERIMENT_PLAN.en.md`
4. `experiments/speaker_representation_scd/R0_DATASET_DECISION.md`
5. `experiments/speaker_representation_scd/R2_DEVELOPMENT_MATERIALIZATION_GATE.md`
6. `experiments/speaker_representation_scd/configs/protocol/reduced_pretraining_screen.json`

Preserve unrelated dirty changes under `experiments/speaker_turn_boundary/` and `.agents/specs/`.
Do not stage, edit, revert, or commit them.

## 3. What Is Already Complete

- R0 scientific protocol and model identities exist.
- The locked research environment exists.
- All four model artifacts have already been acquired.
- mHuBERT-147, WavLM Base+, UniSpeech-SAT Base+, and ERes2NetV2 pre-pooling smoke/parity work is
  complete enough for the experiment.
- Existing ERes2NetV2 final-embedding and LS-EEND results already exist on the shared GT.

Do not repeat model acquisition, four-encoder smoke, ERes-final inference, or LS-EEND inference.

The current Git base before this documentation amendment is
`c4ed4e70149c9f93e9b452af992d7a32bdf8024b`. Reconcile the actual working tree and Goal STATE on
arrival because this handoff may be transferred before a documentation commit is created.

## 4. Sole Dataset

| Item | Identity |
| --- | --- |
| Manifest | `experiments/speaker_turn_boundary/results/turn_episode_v1/episode_manifest_dev.json` |
| Byte SHA-256 | `a5df5e56c2917ba174fca7892fbaa092f2b57705cad7a77bf9a347ba94cddfee` |
| Canonical content SHA-256 | `deb1713cd581c93cc13c47103643041c7551993985379869cfa0ca9a407dff68` |
| Total episodes | 804 |
| Diagnostic episodes | 695 |
| Source identities | 616 |
| Unique WAV bytes | 600 |
| R3 candidate ceiling | 450 positive + 360 negative = 810 rows |
| Existing matched pairs | 313 |

The panel is `development-known`. Results support paired exploratory candidate selection, not a
fresh confirmatory or broad multilingual claim. Current language claims are limited to the actual
English/Mandarin coverage. Missing Korean/Japanese/code-switch/whisper conditions remain missing.

## 5. Start Here: R2-L

The present `r2_execute.py`, `r2_gate.py`, `r2_materialize.py`, and
`configs/r2/development_materialization_gate.json` were built around Zeroth/JVS acquisition. Do not
run these entrypoints unchanged.

Make the smallest adaptation that creates a legacy-only path with these properties:

1. No archive download, extraction, remote request, or public-corpus registry dependency.
2. Revalidate the exact legacy manifest, WAV, annotation, event, pair, and block identities.
3. Reference existing WAVs read-only where stable; copy into the external cache only if stable
   independent addressing requires it.
4. Freeze shared R3 rows from the existing eligible inventory, capped at 810 rows.
5. Freeze an R4 panel capped at six source hours before seeing any new encoder score.
6. Generate causal trailing-window coordinates for 100/300/500 ms at a 1,600-sample hop.
7. Produce a reduced wall-time, memory, and derived-storage forecast using the accepted R1 smoke
   measurements and the actual completed ledger.
8. Emit self-identifying machine-readable receipts under a clearly named
   `legacy-common-gt-v1` scope.

Implementation/harness work is subordinate to obtaining valid experimental measurements. Add only
the checks needed to prevent wrong data, unfair rows, invalid latency, unsafe resource use, or
irreproducible results. Do not spend time broadly hardening the harness.

## 6. Mandatory Stop Before Neural Measurement

After R2-L, report all of the following to the owner and stop:

- exact source/session/WAV/event counts and exclusions;
- exact R3 rows, contexts, layers/taps, and estimated feature-inference count;
- exact R4 source hours, context, hop, promoted-layer rule, and estimated inference count;
- expected wall time, peak RAM, and derived storage for each model and the total;
- the exact commands that would start R3 and R4;
- confirmation that no ERes-final or LS-EEND rerun is included;
- confirmation that Zeroth, JVS, D5, and training are excluded.

Do not interpret this handoff as R3/R4 approval. Wait for an explicit owner response.

## 7. How Experiment Execution Must Be Delegated

Actual experiment execution is not run directly by the coordinator. Use the Orca CLI to open an
OpenCode terminal and assign the execution to a worker agent. This is a durable owner instruction
for simple experiment runs.

The coordinator should supervise by event-driven messages and otherwise wake at approximately
15-minute intervals. Do not continuously poll the worker. A worker message should be handled as
soon as it arrives.

The worker must receive:

- the exact approved command;
- working directory and external `SRSCD_CACHE_ROOT`;
- resource ceilings and one-model-at-a-time rule;
- prohibited actions;
- expected receipts/results;
- instruction to stop and report on data mismatch, legacy-process contention, resource-ceiling
  risk, or any need to change the approved experiment.

## 8. Reduced R3 After Approval

Run all four encoders:

- mHuBERT-147
- WavLM Base+
- UniSpeech-SAT Base+
- ERes2NetV2 pre-pooling

Use the same rows and detector-independent pooling conditions:

- contexts: 100, 300, 500 ms;
- representative layers/taps already frozen by the plan;
- mean pooling;
- adjacent and prototype cosine diagnostics where defined;
- shared pair/block definitions;
- ROC-AUC, EER, overlap coefficient, boundary trajectory, and condition/source breakdowns.

Promote one layer/tap per encoder by the deterministic rule in the protocol. Prefer 300 ms as the
common context; use the all-encoder 500 ms fallback only if the frozen fallback condition is met.

## 9. Reduced R4 After R3

Use one promoted layer/tap per encoder on the frozen panel of at most six source hours:

- primary context: 300 ms or the protocol-wide 500 ms fallback;
- primary hop: 100 ms;
- identical adjacent/prototype/hysteresis detector family across encoders;
- thresholds and operating points chosen only within this development-known screen;
- Boundary Precision/Recall/F1 at 100/250/500 ms;
- availability latency median/p90/p95;
- false events per hour and missed changes;
- compute latency, RTF, peak RAM, and derived storage;
- shared-timeline plots for clean transition, overlap, and backchannel cases when present.

Only the top two encoders receive the optional 50 ms-hop sensitivity. Do not expand the layer,
context, hop, or threshold grid merely because capacity is available.

## 10. Existing Baseline Relationship

Reuse machine-readable ERes-final and LS-EEND event outputs only where their audio, GT, timing, and
metric identities match the frozen legacy panel.

```text
representation table
  four new frozen representations only

event table
  four new zero-shot detectors
  plus existing ERes-final and existing LS-EEND contextual rows
```

LS-EEND does not enter representation cosine/AUC/EER ranking. Do not convert it into an artificial
embedding score. Do not rerun either legacy model.

## 11. Completion Condition

Stop the current study after a reproducible candidate-selection report identifies:

- representation winner;
- zero-shot event leader;
- efficient-backbone/Pareto candidate;
- encoders, layers/taps, and contexts worth carrying into a separately approved learned-head study;
- weaknesses and missing dataset conditions that limit the conclusion.

No learned probe, weighted layer fusion, SCD head, fine-tuning, public test selection, push, merge,
publication, production change, or cleanup is authorized by this handoff.
