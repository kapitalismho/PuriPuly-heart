# PSEM training-strategy data census

The official model experiment uses only `PSEM-STRATEGY-DATA-v2`. Its complete immutable
census is [`data/v2/DATA_CENSUS.md`](data/v2/DATA_CENSUS.md).

| Role | Meetings | Scored hours | Direct | Gap handoff | Same gap | Overlap return | Overlap takeover | Short return |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TRAIN | 64 | 32.458378 | 855 | 4057 | 18506 | 3745 | 2423 | 2162 |
| DEV | 10 | 6.105825 | 129 | 631 | 4282 | 709 | 375 | 377 |
| EVAL | 19 | 10.007499 | 197 | 1472 | 5864 | 918 | 499 | 791 |

The dataset preflight passes all natural exposure, meeting, topology, stable-singleton,
ongoing-overlap, identity-component, waveform, known-speaker, prior-selection, exact WavLM
pretraining-overlap, annotation coverage, mask, and non-substitution gates. DEV and EVAL are
entirely natural. No synthetic supplement is part of the frozen dataset.

The source of truth is the hash-bound `data/v2/dataset_freeze.json`, not this summary.
