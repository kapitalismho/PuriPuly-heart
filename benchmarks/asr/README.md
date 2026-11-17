# CPU ASR benchmarks

Resident-model comparisons and paced replay on frozen English/Japanese conversation inputs. Running this harness does not change application settings or production routing.

## Workspace

- [models.json](models.json) selects models and execution settings. Registered models reuse the application's catalog, integrity checks, and decoding owners.
- [inputs/](inputs/) holds source pins, references, and audio manifests. Local `data/`, `models/`, `results/`, and generated preparation evidence are ignored. Do not commit weights, audio, or generated results.

## Prepare data

Reuse existing audio or run [prepare.py](prepare.py) from the repository root. PyArrow is preparation-only:

```powershell
uv run --no-sync --with pyarrow==23.0.1 python benchmarks/asr/prepare.py --language all
```

`--offline` requires cached pinned sources. `--check-only` validates existing inputs without downloading or rendering, but still writes `benchmarks/asr/data-preparation.json`; preserve any previous preparation evidence before invoking it.

## Run

Run [run.py](run.py) in the locked project environment with a [staged ASR worker](../../.github/CONTRIBUTING.md#native-asr-worker-builds). Keep its DLLs and inventory together. The default is `build/asr_worker/PuriPulyHeartASRWorker.exe`; `--worker` selects another build.

```powershell
uv run --locked python benchmarks/asr/run.py --mode download --model granite --source hf_mirror
uv run --locked python benchmarks/asr/run.py --mode smoke --model granite --language en
uv run --locked python benchmarks/asr/run.py --mode primary --model granite --language en
uv run --locked python benchmarks/asr/run.py --mode replay --model granite --language en
uv run --locked python benchmarks/asr/score.py
```

Substitute a model/language supported by the catalog, such as `sherpa-ja`/`ja`. Downloads use the local `models/` directory; inference can reuse installed models through `--app-model-root`. Run candidates serially: CPU-only, resident, batch one, with two requested inference threads.

## Interpret and extend

[score.py](score.py) regenerates `results/summary.json` from saved results without inference.

- Compare candidates on the same frozen audio and independent references. English uses pooled WER; Japanese uses pooled CER. Empty and failed speech count toward accuracy.
- Primary timings cover preloaded PCM-to-text. Replay covers the paced Self endpoint/queue path. Neither measures end-to-end latency.
- AMI IHM (AMI Consortium, Carletta et al., 2005) is CC-BY-4.0. IngCrowd Inc. / 株式会社イングクラウド is CC-BY-NC-4.0 (noncommercial).
- IBM lists AMI among training sources; exact overlap is unknown. IngCrowd is unsplit.

Add models through [models.json](models.json) or a thin [adapter](adapters.py). Other languages need frozen manifests and audio via `--inputs` and `--data-root`. Reuse the common scoring contract.

## Retired native-Soniox experiment

The #373 capture/replay runner, speaker-policy scorer, and MT-inspection commands are retired from the maintained tree. They are separate from the CPU preparation, resident inference, paced replay, and scoring commands above; no replacement generic harness was introduced.

- [Frozen #373 report](https://github.com/kapitalismho/PuriPuly-heart/blob/d27e1bc555c4925a2f3d3cbe4a00ec1b7a101055/benchmarks/asr/issue373-results.md), [labels](https://github.com/kapitalismho/PuriPuly-heart/blob/d27e1bc555c4925a2f3d3cbe4a00ec1b7a101055/benchmarks/asr/issue373-references.json), and [historical source tree](https://github.com/kapitalismho/PuriPuly-heart/tree/d27e1bc555c4925a2f3d3cbe4a00ec1b7a101055/benchmarks/asr) retain the original results and commands. Those commands describe the pinned historical checkout, not the current layout.
- [#441](https://github.com/kapitalismho/PuriPuly-heart/issues/441) owns the remaining result processing/write-up. Its local historical source, audio, and saved outputs are retained; retirement does not rerun or close that experiment.
- The [cleanup and preservation record](../../docs/workspace-cleanup.md) identifies source pins, local-only evidence limits, retained workspace categories, and verification.
