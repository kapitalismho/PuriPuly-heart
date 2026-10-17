# SELF-STREAM-2A publication fixture style repair

## Result and scope

The bounded publication repair is complete. The three exact requested CI commands passed against the current checkout. No product sources, maintained tests, maintained documentation, dependency metadata, Git state, branches, remotes, or GitHub content were changed by this worker. The only non-fixture change is the authorized Ruff/Black archival-path configuration in `pyproject.toml`.

Authority: [issue #233, SELF-STREAM-2A](https://github.com/kapitalismho/PuriPuly-heart/issues/233). The closure amendment waives mandatory physical VRChat/HMD observation; it does not turn that unperformed observation into a pass. This report is a publication-style receipt, not a new product completion review or publication/merge action. Starting labels supplied by the Director were HEAD `ef9e06ac`, reviewed/exercised product `27e`, and upstream `origin/dev` `f1ab9a4`. No Git command was used to inspect or change those labels.

## Original failure, retained without rerunning it for confirmation

The supplied original exact-command failure, `uv run --locked ruff check .` (retained session artifact 460), contained **16 errors: 15 I001 import-order/layout errors and one F401 unused `uuid.UUID` import**. All were task-fixture errors. Black had not executed because the original commands were chained with `&&`.

Paths below are relative to `tests/fixtures/self_grace_233/`; locations are the original pre-format lines.

| Original file | Diagnostics | Original line(s) / disposition |
| --- | --- | --- |
| `audio/evidence_support.py` | I001 | 1 |
| `audio/historical_75c4485a/run_production_audio.py` | I001 × 2 | 1, 25; frozen archival source, excluded without edits |
| `audio/replay_recorded_schedule.py` | I001 × 2 | 1, 15 |
| `audio/run_isolated_app.py` | I001 | 1 |
| `audio/run_isolated_app_f1ab9a4_probe01.py` | I001 | 1 |
| `audio/run_production_audio.py` | I001 × 2 | 1, 26 |
| `audio/run_production_audio_f1ab9a4_probe01.py` | I001 × 2 | 1, 26 |
| `audio/summarize_evidence.py` | I001 | 1 |
| `experiment.py` | I001; F401 unused uuid.UUID | 1, 14 |
| `gui/run_presentation.py` | I001 | 1 |
| `run.py` | I001 | 54 |

The two historical-runner I001 errors were resolved by the explicitly authorized exact-path exclusion. The other 14 errors were fixed mechanically in active task scripts. No active probe was excluded or given a `noqa`, rule disable, or per-file ignore.

## Preservation before formatting

Before formatting any active Python file, `publication_preformat.zip` was created with the exact original bytes of **all 15 active task Python sources**, including the four already formatted scripts, and `manifest.json`. The archive contains **248,280 original Python bytes**. It contains no `src/` production tree and does not duplicate either frozen archival production-source file.

- Archive SHA-256: `f93c22854120db3a161d8054ea41b4cf62a5673dd688c0e3e1d06f35bbd5fa67`
- Embedded `manifest.json` SHA-256: `8c32d42a337b945f54670afba1270197c2779d31c7f16ad05979973928e59df6`
- Archive CRC check passed and every archived Python member was byte-equal to the captured original before active edits.
- The manifest lists repository-relative names, original byte counts/hashes, and assertion counts. These original runner bytes preserve exact pre-publication fixture provenance for the previous `27e` evidence; formatting does not rewrite historical report hashes.

Original active-source SHA-256 values:

| Original active source | Original SHA-256 |
| --- | --- |
| `audio/evidence_support.py` | `5c1601769389f17cbc457f28f0a5c80156d4e0c4f00e47b38755eab83ca17045` |
| `audio/replay_recorded_schedule.py` | `daa16cf3926e8c15c113881fa062acbf116fb9b912229a28e8f0bd129104f53f` |
| `audio/run_isolated_app.py` | `3975c2369302fa806ec3870a5f432045364c6aefcce5f09b1a1f134917ec1537` |
| `audio/run_isolated_app_f1ab9a4_probe01.py` | `f99e9c20b08b5ea7e1c6c98108716d80291dff6e4099ed1f76b66832e8c7167d` |
| `audio/run_production_audio.py` | `950640ff7b8f43eb165d2041718109ced7d50a824ddca8b6d912c2d3f582501d` |
| `audio/run_production_audio_f1ab9a4_probe01.py` | `80eee83d27dad0022270b9d30a63513566bf9d8648985f96ca73bcc8f030f6ce` |
| `audio/summarize_evidence.py` | `8a967723d1e0a0eb383e3b5a88c3ca1a3c20a8c3ac8ea820b821317a32ca9669` |
| `audio/summarize_runtime_evidence.py` | `4edfbcd969b86e13a2140061a3aaa033a6a4ee6a783b0a39d048e3d68a9d8f1d` |
| `experiment.py` | `c25c4b5b6b873d3d524cfb9eda90b87e3075ce148887e1259032655d3a5ff02a` |
| `gui/owner_presentation.py` | `28d325288651d0a7b7f798823006cbf740b020d5bc35372320b64a553ef4bcd9` |
| `gui/run_native_observation.py` | `82142027e066c4f1bb222cb88b2f5cc87aabc5c60f3130e328c3f7cfcd22660f` |
| `gui/run_owner_presentation.py` | `789fdd7b5a50e42cfe94fc1afc95be345f558d5dd29a7ee965b6c095610747fa` |
| `gui/run_presentation.py` | `37b293e6e52b78d4bd654b2463b598fd0636d61b472707fa055e3868364b17b6` |
| `run.py` | `8298c7b1e3c2a4ba4cb946859e4864d25edfd9a20bf0421499de1e53033a1a70` |
| `stream2/run.py` | `c8bb14cf342049a002c23ea3a73d4bbcf4ac02078483bc47f9d5d5658f53ba93` |

Frozen historical sources remained byte-identical through the repair:

| Frozen archival source | Unchanged SHA-256 |
| --- | --- |
| `audio/historical_75c4485a/run_production_audio.py` | `4394a3dca2770bc9620e426aea13949c294b6b24ff323cd5419c44269a561386` |
| `correctness_results/source/self_translation_channel.py` | `ec77ef5e91ddde932de59ed8a036e101da72373042b000b516ce831159e58e2f` |

## Exact changes and semantic protection

Only the following two repository-relative paths were added to Ruff `extend-exclude` and Black's existing `force-exclude`:

- `tests/fixtures/self_grace_233/audio/historical_75c4485a/run_production_audio.py`
- `tests/fixtures/self_grace_233/correctness_results/source/self_translation_channel.py`

Black's existing `build\.spec` alternative was preserved. The new Black alternatives match the complete named fixture paths, anchored at `.py$`; they do not exclude the active runner or an entire evidence directory. Ruff's rule selection/ignores and all other tool configuration remain unchanged.

Active changes were previewed in memory using Ruff's I/F401 fixes and Black before writing active sources. ASTs were compared after normalizing only ordering within adjacent import blocks, import-alias order, and the specifically identified unused `experiment.py` `from uuid import UUID` alias. The original experiment contained no AST `Name` reference to `UUID`; the `uuid4` import remains. **No nonmechanical delta was found or applied.** Every assertion AST was compared unchanged and in order, preserving **150 assertions across all 15 active sources**. Applied source bytes matched those AST-checked formatter candidates.

The formatter commands applied only to the assigned fixture subtree:

```text
uv run --locked ruff check --select I,F401 --fix tests/fixtures/self_grace_233
Found 14 errors (14 fixed, 0 remaining).

uv run --locked black tests/fixtures/self_grace_233
11 files reformatted, 4 files left unchanged.
```

| Changed active task source | Mechanical delta | Assertions preserved |
| --- | --- | ---: |
| `audio/evidence_support.py` | Import ordering; Black layout | 0 |
| `audio/replay_recorded_schedule.py` | Import ordering; Black layout | 0 |
| `audio/run_isolated_app.py` | Import ordering; Black layout | 0 |
| `audio/run_isolated_app_f1ab9a4_probe01.py` | Import ordering; Black layout | 0 |
| `audio/run_production_audio.py` | Import ordering; Black layout | 0 |
| `audio/run_production_audio_f1ab9a4_probe01.py` | Import ordering; Black layout | 0 |
| `audio/summarize_evidence.py` | Import ordering; Black layout | 0 |
| `audio/summarize_runtime_evidence.py` | Black layout only | 0 |
| `experiment.py` | Import ordering; remove unused `uuid.UUID`; Black layout | 49 |
| `gui/run_presentation.py` | Import ordering; Black layout | 0 |
| `run.py` | Import ordering; Black layout | 0 |

Unchanged active sources: `gui/owner_presentation.py`, `gui/run_native_observation.py`, `gui/run_owner_presentation.py`, and `stream2/run.py`. No runtime branching, probe admission, tolerance, expected result, assertion, selector, retry, fallback, or provider behavior was changed.

## Exact current CI command results

These commands were executed separately, so a failed earlier command could not hide a later result. All exited zero.

```text
$ uv run --locked ruff check .
exit: 0
All checks passed!
```

```text
$ uv run --locked black --check .
exit: 0
All done! ✨ 🍰 ✨
1109 files would be left unchanged.
```

```text
$ uv run --locked python scripts/ci/check_test_selection.py
exit: 0
PR test selection covers 493 offline files
Excluded tests/integration: Mixed live-provider, credential, model-download, and other opt-in integration prerequisites; not authorized for ordinary offline PR execution.
Intentionally excluded files: 9
```

## Focused executable and behavior observations

Rationale: style checks alone could pass after accidentally deleting assertions, hiding an active script, or breaking import-time execution. Pre-apply AST/assertion equivalence, exact exclusions, executable help, and a controlled current-owner smoke close those specific gaps without rerunning broad or live-provider tests.

Current executable identity was observed in a fresh `uv run --locked python -B -c ...` process:

- Python 3.14.7, 64-bit AMD64; `<checkout>/.venv/Scripts/python.exe`.
- Product import: `src/puripuly_heart/__init__.py`.
- Current runner import: `tests/fixtures/self_grace_233/stream2/run.py`.
- Harness-owner import: `tests/helpers/translation_owners.py`.
- Ruff 0.16.1, Black 26.5.1, pytest 9.1.1, pytest-asyncio 1.4.0.

The following help commands each executed and exited zero. They did not start audio decoding, the application, or a GUI.

| Command | Exit |
| --- | ---: |
| `uv run --locked python -B -m tests.fixtures.self_grace_233.stream2.run --help` | 0 |
| `uv run --locked python -B tests/fixtures/self_grace_233/run.py --help` | 0 |
| `uv run --locked python -B tests/fixtures/self_grace_233/audio/run_production_audio.py --help` | 0 |
| `uv run --locked python -B tests/fixtures/self_grace_233/audio/run_production_audio_f1ab9a4_probe01.py --help` | 0 |
| `uv run --locked python -B tests/fixtures/self_grace_233/audio/run_isolated_app.py --help` | 0 |
| `uv run --locked python -B tests/fixtures/self_grace_233/audio/run_isolated_app_f1ab9a4_probe01.py --help` | 0 |
| `uv run --locked python -B tests/fixtures/self_grace_233/audio/summarize_runtime_evidence.py --help` | 0 |
| `uv run --locked python -B tests/fixtures/self_grace_233/gui/run_presentation.py --help` | 0 |

Current controlled fixture smoke:

```text
uv run --locked python -B -m tests.fixtures.self_grace_233.stream2.run --output tests/fixtures/self_grace_233/publication_style_smoke.json
exit: 0
8 scenario results passed; both fast/late fixed-source profiles passed.
```

`publication_style_smoke.json` retains the actual report. It exercised identical-final deduplication, changed final, late predecessor ordering with successor computation, and dual-target progression, each with absent and disabled overlays. Both fixed-source profiles closed `alpha`, then `bravo`, made two provider calls, cancelled none, and produced `en:alpha`, then `en:bravo` at the controlled local sender. These are the runner's existing assertions; none were replaced or weakened. This was a script smoke, not a pytest collection claim.

An additional fresh-process helper smoke exited zero and retained its exact `python -c` argument/source and stdout in `publication_style_checks.json`. It checked:

- Evidence serialization for a synthetic dataclass, UUID, and path, plus the archive digest.
- Both empty and populated summarizer distributions and matching digest helpers.
- The historical experiment still rejects accidental current-tree import with its original error; no isolation check was bypassed.
- Local loopback address lookup remains admitted; non-loopback lookup is rejected before DNS and default VRChat port lookup remains rejected.

The helper made no network connection and wrote no files. The current-owner smoke used synthetic text/VAD ingress, a controlled provider and local sender; it did not open a UDP receiver, use ASR/model hardware, launch GUI/native probes, read private audio, or use paid providers. No broad test suite or live/native probe was run. The full historical replay was not rerun because its normal path invokes `git archive` and this assignment excludes Git execution. `audio/replay_recorded_schedule.py` was AST-equivalent after import/layout fixes; its intentional historical-fixture import guard remains unchanged rather than being suppressed for direct current-tree execution.

## Evidence files and frozen handoff

- `publication_preformat.zip`: original active Python bytes and embedded manifest, SHA-256 `f93c22854120db3a161d8054ea41b4cf62a5673dd688c0e3e1d06f35bbd5fa67`.
- `publication_style_checks.json`: exact command outputs, pre-apply comparison method/results, all current active-source hashes, two frozen-source hashes, Python/tool identity, help outputs, and reproducible helper-smoke argv. SHA-256 `d8e5fafb99fca2b3724287f7fcc413625e22bd5e8f6f6a81857a5fd22c691312`.
- `publication_style_smoke.json`: actual controlled current-runner report, SHA-256 `fd306cdcefb3e4d2841fd63163344d57c630c9166a5a02aefe9fc45ad2fdd117`.
- Current `pyproject.toml` SHA-256: `9e6cb4b4256ed77de62a73fbddf96827febc93a4302213bb2399ddb36246cf01`.

The handoff consists of the eleven mechanically formatted active fixture files, `pyproject.toml`, and these four task-local records (including this report). Product/maintained-test/docs surfaces were intentionally unchanged. The recorded source hashes in `publication_style_checks.json` pin the reviewed style candidate; any subsequent source/config changes invalidate affected checks. The later creation of task-local Markdown/JSON records introduces no Python test-selection or style input.

Previous `27e` product/GUI/local-ASR evidence is retained, not claimed re-executed here. This repair's applicability is source-semantic equivalence plus the explicit current checks and controlled smoke above. It adds no physical destination, population/performance, CI-server, merge, or release claim. Bounded independent review and final-tree publication remain the Director's responsibility. No further repair decision is pending from this worker.
