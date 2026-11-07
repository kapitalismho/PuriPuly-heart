# CPU ASR benchmarks

Resident-model comparisons and paced replay on frozen English/Japanese conversation inputs. Running this harness does not change application settings or production routing.

## Workspace

- [models.json](models.json) selects models and execution settings. Registered models reuse the application's catalog, integrity checks, and decoding owners.
- [inputs/](inputs/) holds source pins, references, and audio manifests. Local `data/`, `models/`, `results/`, and generated preparation evidence are ignored. Do not commit weights, audio, or generated results.

## Prepare data

English uses AMI IHM test data: AMI Consortium, Carletta et al. (2005), CC-BY-4.0. Japanese uses IngCrowd Inc. / 株式会社イングクラウド data under CC-BY-NC-4.0, including its noncommercial restriction. IBM lists AMI among training sources; exact overlap is unknown. IngCrowd is unsplit.

Reuse existing audio or prepare it explicitly from the repository root. PyArrow is preparation-only:

```powershell
uv run --no-sync --with pyarrow==23.0.1 python benchmarks/asr/prepare.py --language all
```

`--offline` requires cached pinned sources; `--check-only` validates existing inputs without downloading or rendering. Reference text must not come from candidate output.

## Run

Use the locked project environment and a [staged ASR worker](../../.github/CONTRIBUTING.md#native-asr-worker-builds), keeping its DLLs and inventory together. The default worker is `build/asr_worker/PuriPulyHeartASRWorker.exe`; `--worker` selects another build.

```powershell
uv run --locked python benchmarks/asr/run.py --mode download --model granite --source hf_mirror
uv run --locked python benchmarks/asr/run.py --mode smoke --model granite --language en
uv run --locked python benchmarks/asr/run.py --mode primary --model granite --language en
uv run --locked python benchmarks/asr/run.py --mode replay --model granite --language en
uv run --locked python benchmarks/asr/score.py
```

Substitute a model/language supported by the catalog, such as `sherpa-ja`/`ja`. Downloads use the local `models/` directory; inference can reuse installed models through `--app-model-root`. Run candidates serially: CPU-only, resident, batch one, with two requested inference threads.

## Interpret and extend

- [score.py](score.py) regenerates `results/summary.json` from saved results without inference. English uses pooled WER; Japanese uses pooled CER. Empty and failed speech remain in denominators; noise is compared with its clean parents.
- Replay measures the paced Self endpoint/queue path, not capture or GUI latency. Compare recorded model/runtime identities; historical scores are not a current baseline.
- Add models through `models.json` or a thin [adapter](adapters.py). Other languages require frozen manifests and audio via `--inputs` and `--data-root`, not just a different language flag. Reuse the common scoring contract.

[Original experiment and results](https://github.com/kapitalismho/PuriPuly-heart/tree/0246def7b70f737c374525c628b5ea0251b88eed/docs/experiments/issue-353).
