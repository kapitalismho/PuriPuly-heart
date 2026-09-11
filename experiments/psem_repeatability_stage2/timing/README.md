# Stage-2 timing probe (CPU-only, max 3 episodes)

Executable: `probe.py` plus `iso_restore.py` (isolated direct-restore helper).
Pre-execution bound: `freeze.json`. Sibling consumes `results.json` ONLY
(`schema: psem-repeatability-stage2-timing.v1`).

## Command (repository root)

    .venv/Scripts/python experiments/psem_repeatability_stage2/timing/probe.py

## Actual output (2026-09-09, Ryzen 7 9800X3D, torch 2.13.0+cpu, no CUDA)
    ATTEMPT ok f0_checkpoint_verify 0.553338s
    ATTEMPT ok nemo_checkout_verify 0.053405s
    ATTEMPT FAILED f0_restore_pinned_gate0lock 0.336615s NeMoAdapterError: dependency lock identity is invalid
    ATTEMPT FAILED f0_restore_pinned_107lock 0.331684s NeMoAdapterError: dependency lock identity is invalid
    ATTEMPT ok h7301_head_availability 0.000434s
    ATTEMPT ok export_manifest_validate 0.000289s
    ATTEMPT ok npz_load_NP1 0.005856s
    ATTEMPT ok npz_load_NP2 0.004454s
    ATTEMPT ok npz_load_NP3 0.004668s
    ATTEMPT ok architecture_proxy_head_init 0.002068s
    ATTEMPT ok architecture_proxy_head_forward_full_chunk 0.107753s
    ATTEMPT ok architecture_proxy_head_forward_NP1 0.025205s
    ATTEMPT ok architecture_proxy_head_forward_NP2 0.025464s
    ATTEMPT ok architecture_proxy_head_forward_NP3 0.025632s
    ATTEMPT ok iso_env_freeze 0.507694s
    ATTEMPT ok direct_upstream_restore_iso 13.919782s
    MEASUREMENT NP1 frozen_evidence_npz_load MEASURED wall_seconds=0.005855700001120567
    MEASUREMENT NP2 frozen_evidence_npz_load MEASURED wall_seconds=0.004453799978364259
    MEASUREMENT NP3 frozen_evidence_npz_load MEASURED wall_seconds=0.004668000037781894
    MEASUREMENT NP1 architecture_proxy_head_forward_7s MEASURED wall_seconds=0.0010902150010224433
    MEASUREMENT NP2 architecture_proxy_head_forward_7s MEASURED wall_seconds=0.0010947650007437915
    MEASUREMENT NP3 architecture_proxy_head_forward_7s MEASURED wall_seconds=0.0011146449978696182
    MEASUREMENT NP1 backbone_compute_probe_real_7s MEASURED wall_seconds=0.23921689999406226
    MEASUREMENT NP2 backbone_compute_probe_real_7s MEASURED wall_seconds=0.24786415000562556
    MEASUREMENT NP3 backbone_compute_probe_real_7s MEASURED wall_seconds=0.24667205000878312
    WROTE experiments\psem_repeatability_stage2\timing\results.json

## What the numbers mean

Pinned parity stays blocked at lock identity for both the preserved gate0
lock and the canonical issue-107 receipts lock (neither file modified).
The isolated path restores the real F0 weights (117693960 params, cpu,
sha-verified, import ~7.3 s, restore ~1.2 s) through the exact upstream
`SortformerEncLabelModel.restore_from` call, then runs the exact frozen 7s
payloads as isolated cold forwards with no prefix state:

- NP1 ES2009c [19741040,19853040], payload sha `953164ee20f47052…`, laps [0.241, 0.237]
- NP2 ES2009d [33405760,33517760], payload sha `461e7f5d93fde955…`, laps [0.249, 0.246]
- NP3 ES2002b [2553520,2665520], payload sha `e9d38552ce83f568…`, laps [0.251, 0.243]

All crops are 112000 int16 samples at 16 kHz mono from the meeting files in
the isolated audio cache (full hashes and file frame counts in
`results.json`); all outputs are [1,88,4]. This is
DIRECT_UPSTREAM_RESTORE_NON_PARITY: validator gates bypassed, no prefix
cache, no event replay, frozen export logits untouched, no availability
accepted. The synthetic Gaussian probe (~0.26 s) is kept as supplemental
evidence inside the isolated attempt only, never in case rows.
Head timing (~1.1 ms per 87-frame forward) is ARCHITECTURE_PROXY_UNTRAINED.
No synthetic or CPU timing here implies any H bound or any GPU behavior.

## Exact gate evidence (source proves the requirement)

- Accelerator: `experiments/psem_sortformer_adaptation_depth/nemo_adapter.py:92-117`,
  `_accelerator_identity` raises unless exactly one CUDA device is exposed.
- Container: same file `:83-89`, `_container_image_identity` raises unless
  `PSEM_CONTAINER_IMAGE_IDENTITY` equals the pinned digest.
- Lock: same file `:147-193`, `validate_dependency_lock` requires current
  python/platform/container/accelerator plus the exact installed inventory.
- Upstream call: same file `:348-353`.
- Lock convention: `issue_107_launch.py:479-481`, lock generated at runtime by
  `write_dependency_lock` into run receipts.

## Hash receipt

- probe.py: `245e3ee88953e2951be543d0060db1d46efacade879a21ba9be025b546d016c4`
- iso_restore.py: `d8f2e89c820b90da8a8048089e00ef4bc6eec6e68a77fa5988f387016c490382`
- freeze.json: `b153439d04309d2ec3b2945ac513599d8b64cf5ae0cc7a76e92f96946fc96358`
- results.json: `e5c88982839fb6b33f47dd82570a12681fd3c3e039491355133b7f0862ba2c9a`
- inputs: checkpoint `8abd32832159c6ac1148c926b7276f35ba34582c444e559dce1f1253fea42ef8`
  (471367680 bytes); export manifest `7ff72366ffa6182e5b3ef7824507294f8dc66073c99287744039a6a7701bc131`;
  dev npz NP1 `5e175c6f…dae76c04f`, NP2 `0ab81949…88f0fa96fa`, NP3 `ef0e84b1…2442db1eba`
  (full values in `freeze.json` / `results.json` assets).
- isolated env and audio cache: `C:/tmp/psem-timing-iso` and
  `C:/tmp/psem-timing-audio` (outside repo; repo `.venv` unmodified).
