# Prefix probe environment manifest (re-exec docs, no inference here)

## Isolated env (existing, untouched by this workstream)

- Python: `C:/tmp/psem-timing-iso/Scripts/python.exe`
  (`torch 2.13.0+cpu`, `python 3.12.10`, `pytorch_lightning 2.4.0`,
  `omegaconf 2.3.1`; stdlib `wave/json/tracemalloc`; third-party `numpy`).
  No `nemo` package installed in the env (by design; source tree supplied below).

## NeMo source tree (existing, read-only)

- Archive: `.cache/issue-107-assets/packages/nemo-1a3c291b3ef0f0e11b72f789b185e1f1bda39bd6.tar.gz`
  sha256 `41b8969a55b41e9b8a8d61aa13b2cb2b433d843cd74c73c86bc22933a98fbf8f`
- Extracted (scratch, outside repo): `C:/tmp/psem-nemo/NeMo`
  (`git rev-parse HEAD` = `1a3c291b3ef0f0e11b72f789b185e1f1bda39bd6`, clean).
- Wired via `sys.path` inside the probe (repo root + NeMo tree root); the probe
  asserts `nemo.__file__` resolves under the pinned tree before loading.

## Checkpoint (existing, read-only, verified pre-load inside the probe)

- `.cache/issue-107-assets/checkpoints/diar_streaming_sortformer_4spk-v2.1.nemo`
  sha256 `8abd32832159c6ac1148c926b7276f35ba34582c444e559dce1f1253fea42ef8`
  (471367680 bytes).

## Corpus clips (existing, read-only slice reads; NOT new captures)

- `C:/Users/salee/.psem-corpus/ami/audio/ES2009c/ES2009c.Mix-Headset.wav` @19741040
- `C:/Users/salee/.psem-corpus/ami/audio/ES2009d/ES2009d.Mix-Headset.wav` @33405760
- `C:/Users/salee/.psem-corpus/ami/audio/ES2002b/ES2002b.Mix-Headset.wav` @2553520
  112000 samples each; int16/32768 scaling (torchaudio.load equivalent).

## Re-exec commands

```powershell
# F0 prefix probe (isolated env; ~17 s; writes C:/tmp/psem-prefix/prefix_results.json)
C:/tmp/psem-timing-iso/Scripts/python.exe experiments/psem_decision_sufficiency/provenance/prefix_probe.py

# Q8 discriminator (.venv; ~2 s; writes provenance/q8_discriminator.json)
$env:PYTHONPATH="."
./.venv/Scripts/python.exe experiments/psem_decision_sufficiency/provenance/q8_discriminator.py

# Source audit (.venv; ~5 s; writes provenance/audit_output.json)
$env:PYTHONPATH="."
./.venv/Scripts/python.exe experiments/psem_decision_sufficiency/provenance/audit.py
```

## Scoped regression tests (no broad suites)

```powershell
./.venv/Scripts/python.exe -m pytest experiments/psem_state_corrected_adaptation_gate/tests/test_material_contract.py experiments/psem_sortformer_adaptation_depth/tests/test_frame_alignment.py
./.venv/Scripts/python.exe -m pytest experiments/psem_sortformer_adaptation_depth/tests/test_receipts.py
```
