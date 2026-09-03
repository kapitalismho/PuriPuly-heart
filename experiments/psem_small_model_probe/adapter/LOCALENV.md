# Local CPU env (psem-localenv, 2026-09-03)

Machine: Windows 11, AMD Ryzen 7 9800X3D, RX 7900 XTX. CPU-only inference
(`CPUExecutionProvider`, `run_opts device cpu`, `map_location cpu`).
Interpreter: worktree `.venv`, Python 3.12.10. No `pyproject.toml`/`uv.lock`
changes — everything below is local-env only.

## Install commands run

Pre-existing in `.venv` (not installed by me): `torch 2.13.0+cpu`,
`onnxruntime 1.28.0`, `numpy 2.5.1`.

```bat
uv pip install --python "<worktree>\.venv\Scripts\python.exe" speechbrain
uv pip install --python "<worktree>\.venv\Scripts\python.exe" "neovad @ git+https://github.com/NeovisionSAS/neovad.git@3d82cbb5b787e195d437cd2047dda4c73fce7f0b"
```

Pulled in: `speechbrain 1.1.1`, `torchaudio 2.11.0`, `neovad 0.1.0`
(pinned rev `3d82cbb5`, same rev as `vendor.json`), `einops 0.8.2`,
`hyperpyyaml 1.2.3`, `soundfile 0.14.0`, `sentencepiece 0.2.2`.

## ECAPA cache

`$FIRERED_ECAPA_DIR` unset, so blobs went to the adapter default
`experiments/psem_small_model_probe/adapter/vendor/spkrec-ecapa-voxceleb/`
(all 6 files, byte sizes match `vendor.json`; SHAs in
`vendor/spkrec-ecapa-voxceleb.receipt.json`).
`embedding_model.ckpt` sha256 `0575cb64…e3d0126a` (83,316,686 bytes).
Directory must stay untracked — never commit the blobs.

## verify_vendor.py live result (synthetic 440 Hz PCM only)

`PSEM_CORPUS_ROOT` unset (sibling psem-corpus owns corpus audio), so
synthetic PCM only. 20/20 PASS, exit 0:

- FireRed live: `bind=~170ms` (ECAPA enroll, 1 s span), step over 100 ms
  chunks x20 `median=~5.2ms p95=~96ms RTF=0.056`, `anchor0=0.058`,
  `speech=None`, `ecapa=0575cb64845e`. First-step p95 is onnx session
  warmup; steady-state is the median.
- NeoVAD live: step over 10 ms chunks x50 `median=0.80ms p95=1.14ms
  RTF=0.080`, `anchor=0.004 speech=0.094` on silence.
- Stub/error-path contracts still PASS (stub weights provably unloadable,
  bind-before-reset/step-guard guards intact).

RTF numbers are informational only — Gate 3/4 owns the RTF<=0.25 gate on
real episodes. No gate enforced here.

## Code changes made for live verification

- `adapter/neovad.py` `_model_or_raise`: vendored `gru.pt` is a portable
  `{config, state_dict}` checkpoint, not a pickled module — now built via
  upstream `VADModel.load()` (lazy import, legacy `torch.load` fallback
  kept). Helper `_vad_model_cls()` drops the adapter dir from `sys.path`
  around that import because this module is itself named `neovad` and
  shadows the dependency when adapter scripts run directly.
- `adapter/verify_vendor.py`: live branches — FireRed reset→bind→step with
  timing when the ECAPA cache + runtime exist (else old error-path checks);
  NeoVAD real `step()` + timing when torch exists; summary lines updated.
  No decoder/threshold changes.

## smoke.py

6/6 PASS, unchanged (stub timings ~0.001ms; real-model timing lives in
verify_vendor.py per above).
