# R2 Development-Only Acquisition and Materialization Gate

## Status

- Scope: development-known data only
- Execution state: gate candidate; no corpus archive has been downloaded
- Upstream checkpoint: accepted R1 commit `ac35b473e4ff932a3ab358a011ad9b21cbf63ca6`
- Confirmatory state: sealed
- Training state: forbidden

This checkpoint opens only the path required to construct the R2 development waveform inventory
and deterministic continuous-window coordinate ledger. It does not authorize pooled-feature
extraction, cache calibration, confirmatory access, or training. The archive and materialization
commands must not be run until this exact gate candidate receives independent acceptance and is
preserved in a local commit.

## Development population

| Source | Materialized development population | Relationship to the existing experiment |
| --- | --- | --- |
| Legacy common-GT | The 695 `diagnostic_dev` episodes in the exact 804-row manifest, resolving to 616 source identities and 600 unique WAV bytes | Exact audio/GT bridge used by the ERes/LS-EEND Phase-4 comparison; development-known only |
| Zeroth-Korean | The first 20 `train_data_01` speaker IDs sorted by `(SHA-256(ID), ID)`, after proving train/test speaker disjointness | Adds controlled Korean development evidence; official test remains confirmatory |
| JVS | The 20 fixed development speakers in the R0 split contract, across normal, non-parallel, whisper, and falsetto audio | Adds controlled Japanese and same-speaker nuisance evidence; the reserved 20 speakers remain confirmatory |

The legacy panel is not resplit and is not promoted to confirmatory evidence. Its existing
thresholds, detector states, feature caches, and shortlist conclusions are not imported.

## Official public-source identity

Zeroth-Korean is acquired from the official SLR40 external archive URL recorded by OpenSLR. The
gate records the returned archive byte count, SHA-256, final URL, and selected HTTP identity
headers before any member is materialized.

JVS is acquired from the 3.5 GB Google Drive file linked by the official JVS project page. The
fixed Drive file ID is `19oAw8wWn3Y7z6CKChRdAyGOB9yupL_Xt`. JVS audio remains restricted to
the published academic/non-commercial research terms and is not product evidence.

Neither release publishes a checksum used by this repository. The first supervised acquisition
therefore freezes the exact downloaded archive bytes in the archive receipt. A pre-existing final
archive or partial file is rejected rather than silently trusted or resumed.

## Combined-archive isolation

The Zeroth and JVS public archives contain both development-eligible and reserved material. The
archive itself may be downloaded and its member names may be enumerated because deterministic
speaker selection requires release metadata. Payload reads are stricter:

- Zeroth payload reads are permitted only for selected `train_data_01` FLAC members.
- No `test_data_01` member payload may be opened.
- JVS payload reads are permitted only for the fixed 20 development speakers and the four frozen
  condition directories.
- No reserved JVS speaker member payload may be opened.
- Absolute paths, parent traversal, Windows drive paths, backslashes, links, devices, duplicate
  names, and case-colliding names fail closed before materialization.

Before any selected payload is opened, archive metadata must establish the official Zeroth
population of 105 train and 10 test speakers and the complete JVS population of 100 speakers.
Every JVS speaker must expose exactly 100 `parallel100`, 30 `nonpara30`, 10 `whisper10`, and 10
`falsetto10` WAV members. These release inventories and the fixed development/reserved selections
are preserved in the archive and materialization receipts.

The combined archives remain opaque source artifacts. Their mere storage does not make their
reserved members development data and does not authorize later confirmatory use.

## Canonical waveform contract

Every materialized waveform is a complete mono, signed 16-bit PCM WAV at 16 kHz. Zeroth FLAC is
decoded without a sample-rate change. JVS 24 kHz PCM is converted with the frozen
`torchaudio.functional.resample` Kaiser-sinc parameters in the machine-readable gate.

The legacy source copies must match the exact SHA-256 already present in the episode manifest.
Each canonical waveform is represented exactly once in
`data/r2/development/waveform_inventory.jsonl`, with the full PCM range from sample zero through
`num_samples` marked eligible. Speaker, condition, source-member, and legacy-session mappings are
kept separately in `source_metadata.jsonl` so the forecast inventory schema remains stable.

## Coordinate contract

For every canonical waveform, one JSONL shard is generated for the reduced set of trailing
windows at 100, 300, and 500 ms. The first frontier is
`eligible_start + context`; subsequent frontiers advance by exactly 1600 samples (100 ms) through
the eligible end. The existing R1 forecast code independently regenerates every expected row and
rejects omissions, additions, reordering, altered frontiers, and waveform-binding changes.

No encoder is loaded and no neural inference occurs in R2 materialization.

## Execution and resource boundary

The only public entrypoint is:

```text
python -m experiments.speaker_representation_scd.r2_execute archives
python -m experiments.speaker_representation_scd.r2_execute materialize
```

Both actions reuse the accepted experiment-wide Windows Job Object and exclusive execution lease.
They run sequentially with one worker, eight CPU threads, a 24 GiB hard job-memory ceiling, a
24-hour per-action ceiling, a 24-hour cumulative reduced-screen ceiling, and continuous fail-closed detection of
the legacy experiment. Every authoritative action receipt requires one matching completed usage
attestation.

The storage gates remain 25 GiB for source downloads, 20 GiB for derived cache, 50 GiB for the
external root, and at least 55 GiB free before an action. The external cache remains outside the
repository on `C:`. Before materialization, selected member metadata is used to project canonical
PCM bytes and all coordinate rows. Each output is reserved against both ceilings before writing,
and final filesystem usage is checked again before the action can complete.

## Outputs

Archive acquisition produces:

```text
manifests/r2/development/development_archive_receipt.json
sources/r2/development/zeroth-korean-development/zeroth_korean.tar.gz
sources/r2/development/jvs-development/jvs_ver1.zip
```

Development materialization produces:

```text
manifests/r2/development/development_acquisition_receipt.json
manifests/r2/development/development_coordinate_ledger.json
manifests/r2/development/development_materialization_receipt.json
data/r2/development/waveform_inventory.jsonl
data/r2/development/source_metadata.jsonl
data/r2/development/coordinates/<source>/<waveform>.jsonl
sources/r2/development/<source>/waveforms/*.wav
```

The materialization receipt binds the archive receipt, acquisition receipt, coordinate ledger,
gate identity, execution-code manifest, run provenance, and completed supervision identity.

## Explicitly disabled

```text
confirmatory audio or annotation payload access  no
VoxConverse or AISHELL-4 acquisition              no
Zeroth official-test payload extraction           no
reserved JVS speaker payload extraction            no
cache calibration                                  no
pooled feature extraction                          no
full hidden-state extraction                       no
neural inference                                   no
training                                           no
production changes                                 no
```

The next checkpoint after accepted and successful R2 materialization is a separate, minimal cache
calibration gate. Full extraction remains blocked until the resulting measured forecast is
independently reviewed and accepted.
