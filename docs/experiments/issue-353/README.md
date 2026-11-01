# English/Japanese CPU ASR comparison

Experiment for [#353](https://github.com/kapitalismho/PuriPuly-heart/issues/353). Production model routing and application behavior are unchanged. The runner uses resident CPU models, two inference threads, batch size one, and serial candidate execution. Two inference threads are not a whole-process thread cap.

Results: [comparison and recommendations](results/report.md), [saved machine-readable summary](results/summary.json), and [execution evidence](execution-evidence.json). All 740 primary calls and eight paced replays completed on the existing runtime. English Granite is a promising replacement on this sample; retain Japanese sherpa CTC. The report qualifies timing under low available RAM and preserves raw no-speech-marker behavior.

## Inputs and access

- English: 120 reconstructed AMI IHM test utterances, 40 each at 0.5–2, >2–5, and >5–15 seconds, from four meetings and 12 source-speaker labels. Human references retain upstream Kaldi normalization. This is **not** the exact #308 follow-up sample: the original individual-headset server returned HTTP 403 during full acquisition.
- Japanese: the exact recovered #308 IngCrowd selection and human references: 120 utterances, 108 at 0.5–5 seconds and 12 at >5–15 seconds, five conversations and ten conversation-local speaker tiers. These tiers do not establish globally distinct people. The corpus is unsplit.
- Each language adds two noisy derivatives of 30 frozen clean parents (15 and 5 dB SNR) and five silence/noise controls: 185 inputs per model/language cell, 740 primary decodes across four cells.
- Replay uses genuine continuous 120-second windows, clean and pink15. English uses ES2004d's original Mix-Headset recording, **not** #308's mean of four individual headsets. Japanese uses the original mixed Zoom recording. Shared estimated speech ends are not acoustic ground truth.

All prepared audio is 16 kHz mono PCM16 WAV. Historical/upstream FLOAT hashes are distinct from current PCM16 hashes. [Frozen inputs](inputs/), [source pins and recipe](inputs/sources.json), and [preparation evidence](data-preparation.json) retain identities, boundaries, references, augmentation parents, hashes, checks, and failed recovery attempts. No model output supplies a reference.

AMI attribution: AMI Consortium; Carletta et al. (2005), CC-BY-4.0. IngCrowd attribution: IngCrowd Inc. / 株式会社イングクラウド, CC-BY-NC-4.0; observe its noncommercial restriction. Corpus text retains its corpus license separately from repository code. Public source availability is not a waiver of upstream terms. IBM lists AMI among its training sources; exact overlap is unknown. Weights and audio remain outside Git.

## Prepare

Run from the repository root on Windows with the repository's locked Python environment. PyArrow is an isolated preparation-only dependency; it is not an application dependency.

```powershell
uv sync --locked --extra dev
uv run --no-sync --with pyarrow==23.0.1 python scripts/asr_benchmark/prepare.py --data-root .tmp/asr353/data --language all
uv run --no-sync --with pyarrow==23.0.1 python scripts/asr_benchmark/prepare.py --data-root .tmp/asr353/data --language all --offline --check-only
```

Preparation downloads hash-pinned public sources and uses the tracked frozen selection; it does not require the unpushed #308 commits. `--offline` requires cached sources; `--check-only` checks existing outputs rather than rendering them. The latter still needs original sources to validate references and boundaries. All audio/cache files belong below the configurable `--data-root`.

The baseline assets are the current app's `orukeet-q8-0-transcribe-cpu` and `parakeet-tdt-ctc-0.6b-ja-int8-sherpa`. For an app instance that needs provisioning, the supported commands are:

```powershell
uv run --locked puripuly models download --model-id orukeet-q8-0-transcribe-cpu
uv run --locked puripuly models download --model-id parakeet-tdt-ctc-0.6b-ja-int8-sherpa
```

These are normal app asset operations, not isolated benchmark commands; use the intended app instance and inspect terminal operation results. The experiment reuses installed assets read-only and verifies the pinned file hashes. See the [CLI guide](../../cli.md) for instance and asset-operation behavior. Model licenses are recorded in [models.json](models.json).

Download the challenger separately:

```powershell
uv run --locked python scripts/asr_benchmark/run.py --mode download --model granite --model-root .tmp/asr353/models
```

This downloads only the pinned multilingual Granite artifact at `62f64d9d5ad35989b06caf199cafe7155944a22b`, verifies its 505,663,136-byte size and SHA-256, and does not install it in the app. **Q8_0 means 8-bit weight quantization**, not all-INT8 execution or arithmetic equivalence with ONNX INT8.

## Run each pair

Use an explicit existing staged app worker built from the recorded transcribe.cpp revision. In this experiment's worktree layout the worker is in the parent checkout; in a standalone checkout use its own `build/asr_worker/PuriPulyHeartASRWorker.exe`. Keep the worker's adjacent DLLs and runtime inventory together. The runner records runtime identities; a different build produces a different experiment.

```powershell
$Worker = (Resolve-Path ../../../build/asr_worker/PuriPulyHeartASRWorker.exe).Path
$AppModels = Join-Path $env:LOCALAPPDATA puripuly-heart/models
$Common = @('--data-root', '.tmp/asr353/data', '--model-root', '.tmp/asr353/models', '--app-model-root', $AppModels, '--worker', $Worker)
$Out = '.tmp/asr353/rerun-results'
uv run --locked python scripts/asr_benchmark/run.py @Common --output $Out --mode smoke --model granite --language en --limit 9
uv run --locked python scripts/asr_benchmark/run.py @Common --output $Out --mode smoke --model granite --language ja --limit 3
uv run --locked python scripts/asr_benchmark/run.py @Common --output $Out --mode primary --model orukeet --language en
uv run --locked python scripts/asr_benchmark/run.py @Common --output $Out --mode primary --model granite --language en
uv run --locked python scripts/asr_benchmark/run.py @Common --output $Out --mode primary --model sherpa-ja --language ja
uv run --locked python scripts/asr_benchmark/run.py @Common --output $Out --mode primary --model granite --language ja
uv run --locked python scripts/asr_benchmark/run.py @Common --output $Out --mode replay --model orukeet --language en
uv run --locked python scripts/asr_benchmark/run.py @Common --output $Out --mode replay --model granite --language en
uv run --locked python scripts/asr_benchmark/run.py @Common --output $Out --mode replay --model sherpa-ja --language ja
uv run --locked python scripts/asr_benchmark/run.py @Common --output $Out --mode replay --model granite --language ja
```

Each replay invocation runs clean and noise15: eight replay runs in total. Keep candidates serial and avoid other CPU-heavy preparation while measuring. Prompts, custom vocabulary, context carryover, external correction, translation, TTS, overlays, and network inference are excluded. Initialization and first decode are recorded separately from warmed calls. The PCM-input-to-text clock includes WAV/file IPC for the worker-backed models and the in-process call for Japanese sherpa CTC; native decode-only timing is a different boundary.

Replay reuses the current scoped Self endpoint/queue policy, with independent 1× ingress and terminal consumer availability as the output boundary. It bypasses the full capture owner's pre-provider retention layer and does not measure GUI presentation. First-partial latency is N/A for these final-only paths.

Without `--limit`, smoke decodes only the first three clean inputs. Explicit limits select the first rows in manifest order; primary mode defaults to all 185 inputs.

## Regenerate saved results without inference

```powershell
uv run --locked python scripts/asr_benchmark/score.py --results docs/experiments/issue-353/results --output .tmp/asr353/saved-summary.json
uv run --locked python scripts/asr_benchmark/score.py --results .tmp/asr353/rerun-results --output .tmp/asr353/rerun-summary.json
```

English uses pooled word Levenshtein error; Japanese uses pooled character Levenshtein error. Both reference and hypothesis receive NFKC and casefold, retaining Unicode letter/number/mark categories. English punctuation becomes separators; Japanese punctuation and spaces are removed. No kana/kanji equivalence rewrite hides `何か` versus `なんか`. Raw references and hypotheses remain available. Noise comparisons use their own 30 clean parents, not the whole clean120 set. Empty speech outputs remain in accuracy denominators; failure and timeout counts remain separate from successful-call timing.

The reusable code is in [scripts/asr_benchmark](../../../scripts/asr_benchmark/). A new compatible model can add a configuration entry; a new execution API needs a thin adapter while retaining the same frozen inputs, scorer, and output records. This is a bounded single-host experiment, not a framework, CI benchmark, or production replacement decision.
