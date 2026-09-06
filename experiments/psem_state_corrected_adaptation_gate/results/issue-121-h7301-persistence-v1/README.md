# Issue 121 H7301 persistence bundle

This durable result bundle packages the frozen H7301 CPU postprocess outputs and the posthoc persistence analysis requested for publication review. It is **STOP / inconclusive; retain F0** evidence, not a candidate acceptance or a new arm authorization. No H7302, T2, TA, or EVAL arm is opened by this bundle.

## Reproduce the analysis

Run from the repository root. The command reads only the durable export and canonical gzip/diagnostic files below; it does not retrain, run GPU inference, or recompute the full unique-score frontier.

```text
uv run python -m experiments.psem_state_corrected_adaptation_gate.results.issue-121-h7301-persistence-v1.persistence_analysis \
  --export-dir experiments/psem_state_corrected_adaptation_gate/results/issue-121-h7301-persistence-v1/export/gpu_export \
  --frontier experiments/psem_state_corrected_adaptation_gate/results/issue-121-h7301-persistence-v1/canonical/dev_frontier.json.gz \
  --diagnostics experiments/psem_state_corrected_adaptation_gate/results/issue-121-h7301-persistence-v1/canonical/gate1_diagnostics.json \
  --out-json experiments/psem_state_corrected_adaptation_gate/results/issue-121-h7301-persistence-v1/persistence_analysis.json \
  --out-md experiments/psem_state_corrected_adaptation_gate/results/issue-121-h7301-persistence-v1/PERSISTENCE_ANALYSIS.md
```

The output is deterministic JSON plus the rendered Markdown evidence. The script loads the repository's frozen session snapshot and reference ledgers through the existing decoder path. It uses exact raw `dev_frontier.json` thresholds, valid-and-mapped DEV frames for AP, and the configured 500 ms product-event alignment tolerance for matched/unmatched run traces.

## Canonical evidence

The four canonical postprocess outputs are copied byte-for-byte from the completed local rerun, except that the 1.06 GB frontier is stored as deterministic gzip because GitHub rejects files over 100 MB:

- [`canonical/calibration_metrics.json`](canonical/calibration_metrics.json)
- [`canonical/dev_frontier.json.gz`](canonical/dev_frontier.json.gz)
- [`canonical/gate1_diagnostics.json`](canonical/gate1_diagnostics.json)
- [`canonical/gate1_decision_evidence.md`](canonical/gate1_decision_evidence.md)

The gzip stream uses gzip level 9 with an mtime of zero. `bundle_manifest.json` records both the compressed SHA-256 and the original uncompressed frontier size/SHA-256. To extract the exact original bytes:

```text
uv run python -c "import gzip; from pathlib import Path; src=Path('experiments/psem_state_corrected_adaptation_gate/results/issue-121-h7301-persistence-v1/canonical/dev_frontier.json.gz'); dst=src.with_suffix(''); dst.write_bytes(gzip.decompress(src.read_bytes()))"
```

The immutable export manifest and training metrics are retained at [`export/gpu_export/gpu_export_manifest.json`](export/gpu_export/gpu_export_manifest.json) and [`training_metrics.json`](training_metrics.json). The export directory contains the 11 CALIB and 10 DEV numeric NPZ payloads named by expected dataset ID. It contains no audio, transcripts, PII, `.nemo` files, checkpoints, `.env` files, SSH keys, or process logs.

## Scientific summary

The DEV candidate ranking AP is `0.487681950383668` pooled over valid-and-mapped frames. Candidate source-macro AP is AMI `0.491683249822344`, AliMeeting `0.430875816411725`, and all ten sources `0.473441019799158`. DEV F0 AP is `0.150603849645612` pooled and is reported as a posthoc analysis value; the earlier decision report correctly described it as unavailable in that earlier bundle and was not preregistered with this metric.

At exact selected global C-envelope thresholds, positive-run traces reproduce:

- H100: matched `n=599`, median/p90/max `480/589/960 ms`; unmatched `n=663`, `300/1000/4700 ms`.
- H300: matched `n=655`, `500/668.2/995 ms`; unmatched `n=545`, `400/1400/4700 ms`.
- H500: matched `n=344`, `500/694.5/972 ms`; unmatched `n=196`, `892/2131/4700 ms`.

Holding the exact H500 score threshold while changing only the horizon yields unmatched run median/p90/max `300/1000/4700 ms` at H100, `400/1300/4700 ms` at H300, and `892/2131/4700 ms` at H500. These are descriptive persistence/confirmation traces. Unmatched is a timing/matching residual and is not a categorical ground-truth false-positive label; no causal source-level explanation is proven.

The full result, thresholds, source AP rows, run ledgers, hashes, and execution receipt are in [`persistence_analysis.json`](persistence_analysis.json), [`PERSISTENCE_ANALYSIS.md`](PERSISTENCE_ANALYSIS.md), and [`bundle_manifest.json`](bundle_manifest.json). Parent decision context is [`../../STATE_CORRECTED_ADAPTATION_DECISION.md`](../../STATE_CORRECTED_ADAPTATION_DECISION.md), and the parent experiment README links this durable bundle.

Observed frozen execution provenance: CPU postprocess wall duration `29m18s`, 53 FIT sources, 142 training steps, 11 CALIB sources, and 10 DEV sources. Formal commit review remains outstanding. This bundle is prepared for Director review; it has not been committed, pushed, released, or posted.
