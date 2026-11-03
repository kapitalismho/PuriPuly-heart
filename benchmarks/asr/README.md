# CPU ASR benchmarks

Reusable resident-model runner, paced replay, common scorer, and frozen English/Japanese conversation inputs from [#353](https://github.com/kapitalismho/PuriPuly-heart/issues/353). Production routing is not changed by running this harness. The original report and raw results remain at [the merged experiment commit](https://github.com/kapitalismho/PuriPuly-heart/tree/0246def7b70f737c374525c628b5ea0251b88eed/docs/experiments/issue-353).

## One local workspace

Everything is rooted here, independent of the caller's working directory:

- Tracked: the six Python scripts, `models.json`, this README, and `inputs/` manifests/source pins.
- Ignored: `data/` for prepared audio and original source caches; `models/` for model files and installation metadata; `results/` for transcripts, run metadata, replay summaries and scores; `data-preparation.json` for generated preparation evidence.

No weights, corpus audio, generated transcripts or benchmark results belong in Git. The old `local-data/asr/`, `scripts/asr_benchmark/` and tracked experiment-results locations are not used.

`models.json` selects current application models. Their supported languages, source revisions, sizes and SHA-256 values come from the canonical application catalog/manifests rather than a second set of pins. Both registered-model downloads and native decoding reuse the application's existing owners. Granite's marker-only no-speech normalization therefore applies; its output is not identical to the historical raw experimental adapter.

## Models and a short smoke

Run these commands from the repository root using its locked Python environment:

```powershell
uv run --locked python benchmarks/asr/run.py --mode download --model granite --source hf_mirror
uv run --locked python benchmarks/asr/run.py --mode download --model gigaam --source huggingface
uv run --locked python benchmarks/asr/run.py --mode smoke --model granite --language en
uv run --locked python benchmarks/asr/run.py --mode smoke --model sherpa-ja --language ja
```

Downloads install only into this workspace's `models/`, use the normal staged integrity checks and Hugging Face/HF-mirror failover, and do not change application settings or start an app host. Use the other model keys in `models.json` to provision retained baselines. Each installed model has its canonical model-ID subdirectory and `installed-manifest.json`.

The default worker is `build/asr_worker/PuriPulyHeartASRWorker.exe` in the repository. Keep its adjacent DLLs and runtime inventory together. `--worker` selects another existing staged build; metadata records its identity. A different build is a different experiment. `--app-model-root` can reuse an existing application's installed models read-only instead of this workspace's model root.

Smoke defaults to three distinct clean clips. `--limit` overrides that selection and takes the first manifest rows, which can interleave noisy derivatives. Models remain resident, CPU-only, batch one, with two requested inference threads; this is not a whole-process thread cap. Run candidates serially. No prompts, custom vocabulary, context carryover, translation, TTS, GUI, overlays or network inference are included.

## Frozen data and optional preparation

The shipped manifests cover English and Japanese only. English has 120 reconstructed AMI IHM test utterances, 40 each at 0.5–2, >2–5 and >5–15 seconds, four meetings and 12 source-speaker labels. It is not the exact older #308 headset selection. Japanese retains the recovered IngCrowd selection: 108 short and 12 medium utterances, five conversations and ten conversation-local speaker tiers, not ten proven distinct people.

Each language has 30 frozen clean parents with seeded pink/fan-like 15 dB and 5 dB derivatives, plus five silence/noise controls: 185 primary inputs. Genuine 120-second replay windows have clean and 15 dB variants. Inputs are shared 16 kHz mono PCM16 WAV; source pins, references, boundaries, augmentation parents and hashes are in `inputs/`.

Existing local audio is reused. Missing historical audio is not recovered automatically. Preparation is an explicit optional operation; PyArrow is preparation-only, not an application dependency:

```powershell
uv run --no-sync --with pyarrow==23.0.1 python benchmarks/asr/prepare.py --language all
uv run --no-sync --with pyarrow==23.0.1 python benchmarks/asr/prepare.py --language all --offline --check-only
```

`--offline` requires cached pinned sources. `--check-only` checks existing outputs and source references without rendering or downloading. `--data-root` overrides the local audio/cache root. No candidate output supplies ground truth.

AMI attribution: AMI Consortium; Carletta et al. (2005), CC-BY-4.0. IngCrowd attribution: IngCrowd Inc. / 株式会社イングクラウド, CC-BY-NC-4.0, including its noncommercial restriction. Corpus references retain their corpus licenses separately from repository code. IBM lists AMI among its training sources; exact overlap is unknown. IngCrowd is unsplit.

## Primary, replay and scoring

```powershell
uv run --locked python benchmarks/asr/run.py --mode primary --model granite --language en
uv run --locked python benchmarks/asr/run.py --mode primary --model sherpa-ja --language ja
uv run --locked python benchmarks/asr/run.py --mode replay --model granite --language en
uv run --locked python benchmarks/asr/score.py
```

Primary defaults to all 185 rows. Each replay invocation runs clean and noise15, with independent 1× ingress through the current Self endpoint/scoped queue policy. This bypasses capture-owner retention and ends at the experiment terminal consumer, not GUI presentation. Source-end estimates and input coverage are not acoustic ground truth or lexical correctness. First-partial latency is N/A for these final-only paths.

The scorer reads saved `*-primary.jsonl`, run metadata and `*-replay-summary.json` without importing or loading an ASR model. It writes `results/summary.json`; `--results` and `--output` override these paths. English uses pooled word Levenshtein error, Japanese pooled character error; normalization is NFKC + casefold and Unicode L/N/M retention, with no kana/kanji-equivalence rewrite. Empty and failed speech stay in accuracy denominators; failures/timeouts remain separately visible. Noise is compared against its own clean parents.

## Another model or dataset

A registered application model needs a `models.json` selection entry. An unregistered GGUF challenger can use the existing `gguf` adapter with explicit `languages`, pinned `repository`/`revision`, and `files` entries (`root: "experiment"`, `path`, `bytes`, `sha256`); place those verified files under `models/` yourself. Another execution API needs a thin adapter, not a new framework.

`--language` accepts the selected model's declared language codes, but this does not create data. Russian GigaAM needs a supplied `ru.jsonl`; no Russian comparison corpus or benchmark result is claimed here. Use `--inputs` for another frozen manifest directory and `--data-root` for its audio. Input rows retain `id`, `path`, `sha256`, `language`, `duration_s`, `reference`, `variant` (`clean`, `noise15`, `noise5`, `control`) and `noise_pair_id` where applicable. Replay additionally requires `replay-<language>.json`. The common scorer defines WER for English and CER for other languages; document any different scoring contract explicitly.
