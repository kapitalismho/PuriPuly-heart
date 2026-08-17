# PSEM trainable formulation gate

Experiment execution stopped on 2026-08-17. The consolidated scope, results, and limitations are documented in [`EXPERIMENT_STOP_SUMMARY.ko.md`](EXPERIMENT_STOP_SUMMARY.ko.md). This experiment did not fine-tune pretrained encoder parameters and did not run a scratch arm, so it does not answer fine-tuning versus scratch.

This experiment answers GitHub issue #72 plus the owner's 2026-08-16 scope amendment with three pinned models and the same four arms for each model:

- `eres2netv2-standard-prepool`
- `wavlm-base-plus`
- `mhubert-147`

| Arm | Encoder | Target |
| --- | --- | --- |
| `A-FROZEN-DIRECT` | frozen | direct speaker-change event |
| `B-TRAINABLE-DIRECT` | shared residual output adapter | direct speaker-change event |
| `C-FROZEN-STATE` | frozen | local speech state and speaker relation |
| `D-TRAINABLE-STATE` | shared residual output adapter | local speech state and speaker relation |

All twelve arms reuse the R7-B meeting folds, labels, source-time grid, fixed-lag context, event semantics, one-to-one matching, and false-events-per-source-hour accounting. Each 100 ms cell is represented by the pinned model's final embedding over the preceding 500 ms. ERes cells are extracted directly from the receipt-bound waveform bytes with the pinned checkpoint and verified source implementation. The local head receives cell frontiers from `t - 500 ms` through `t + 1000 ms`.

The primary comparison scans the complete prediction-score range and selects each arm's operating point by maximum mean F1 across the 100, 250, and 500 ms matching collars. Precision, recall, and F1 are reported for all three collars in parallel. False events per hour is retained as context and at the issue's 1/5/10/20 FE/h compatibility reference rows; it is not a product policy, a score-range limit, or the rule used to select the headline threshold.

Structured arms supervise every relation consumed by the decoder: adjacent singleton endpoints and singleton endpoints separated only by silence. Diagnostics report the combined decoder-path relation quality and the silence-gap subset separately.

The end-to-end structured result does not by itself identify a structured-representation failure. State/relation quality and the structured-to-event projection are reported separately so projection or calibration loss is not attributed to representation learning without evidence.

B and D use the same bounded output-adapter recipe: a 64-dimensional residual bottleneck attached after the encoder output. The pretrained model parameters remain fixed; task-loss gradients update only the output adapter and task head. This is not pretrained-encoder fine-tuning. There is no adaptation-method sweep.

Artifacts are written outside the repository under `%SRSCD_CACHE_ROOT%/results/psem_trainable_formulation_gate_v1/`.

```powershell
$env:SRSCD_CACHE_ROOT = 'C:\Users\salee\AppData\Local\puripuly-heart-research\speaker_representation_scd_v1'
$env:UV_CACHE_DIR = "$env:SRSCD_CACHE_ROOT\uv"
uv run --project experiments\speaker_representation_scd\environment --frozen python -m experiments.psem_trainable_formulation_gate.run prepare
uv run --project experiments\speaker_representation_scd\environment --frozen python -m experiments.psem_trainable_formulation_gate.run extract --model-id wavlm-base-plus
uv run --project experiments\speaker_representation_scd\environment --frozen python -m experiments.psem_trainable_formulation_gate.run extract --model-id mhubert-147
uv run --project experiments\speaker_representation_scd\environment --frozen python -m experiments.psem_trainable_formulation_gate.run extract --model-id eres2netv2-standard-prepool
uv run --project experiments\speaker_representation_scd\environment --frozen python -m experiments.psem_trainable_formulation_gate.run develop --model-id wavlm-base-plus --arm A-FROZEN-DIRECT
uv run --project experiments\speaker_representation_scd\environment --frozen python -m experiments.psem_trainable_formulation_gate.run develop --model-id wavlm-base-plus --arm C-FROZEN-STATE
uv run --project experiments\speaker_representation_scd\environment --frozen python -m experiments.psem_trainable_formulation_gate.run develop --model-id wavlm-base-plus --arm B-TRAINABLE-DIRECT
uv run --project experiments\speaker_representation_scd\environment --frozen python -m experiments.psem_trainable_formulation_gate.run develop --model-id wavlm-base-plus --arm D-TRAINABLE-STATE
uv run --project experiments\speaker_representation_scd\environment --frozen python -m experiments.psem_trainable_formulation_gate.run report
```

Run the four `develop` actions in the same order for each of the three model IDs before generating the report.
