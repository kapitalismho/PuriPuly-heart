# Workspace cleanup and ASR retirement

Implementation record for [#447](https://github.com/kapitalismho/PuriPuly-heart/issues/447), under [#446](https://github.com/kapitalismho/PuriPuly-heart/issues/446). Inventory and cleanup were performed on 2026-11-17 in the existing primary development workspace, not only in the fresh implementation worktree. The starting `dev`/implementation baseline was `508a13809474a6a14948227c18558cfd090cc35c`; both had clean tracked state at initial inspection.

The owner chose to accumulate #447, #449, and #448 on the same branch and publish one combined PR later. This record does not claim a push, PR, merge, remote tracking update, or issue closure. Publish the current navigation with that combined change; keep #441's result-write-up obligation separate.

## Maintained surface

Removed from the current `benchmarks/asr/` tree:

- `issue373.py`: historical Soniox capture and production-path replay.
- `issue373_score.py`: historical speaker-policy scoring.
- `issue373_mt.py`: historical MT inspection.
- `issue373-results.md` and `issue373-references.json`: original report and labels remain at the immutable references below, without rewritten results.

Tracked caller searches found no production, independent CPU-benchmark, or test dependency on these tools. No dedicated machinery-only tests required removal. Useful native-speaker/caption and component tests remain. The remaining demonstrated consumers are in the preserved historical #441 checkout, including its cloud recorder, MT/display adapter, scorer, lexical evaluation, and ignored Soniox diagnosis probe. They retain their own historical dependencies; they are not maintained current-tree entrypoints.

No component was extracted: the independent CPU benchmark already owns preparation, resident inference, paced replay, resource observation, and scoring without the #373 harness. Renaming Soniox-specific code would create an abstraction without a current consumer. The [benchmark guide](../benchmarks/asr/README.md) retains the supported commands, frozen inputs, and licensing boundaries. Production ASR, caption policy, and application settings are unchanged.

Maintained CI/release scripts remain in `scripts/ci/` and `scripts/release/`. The broad `scripts/` ignore/allowlist machinery is replaced by `scripts/local/` plus exact exclusions for the five existing root-level personal scripts. Previously hidden maintained helpers, including `check_test_selection.py` and the Windows release-evidence entrypoints/thresholds, are discoverable without force-adding. Personal tools were neither executed nor moved. [Contributor guidance](../.github/CONTRIBUTING.md#pr-verification) identifies the maintained and disposable locations.

## Historical #373 references

- [Report](https://github.com/kapitalismho/PuriPuly-heart/blob/d27e1bc555c4925a2f3d3cbe4a00ec1b7a101055/benchmarks/asr/issue373-results.md), [labels](https://github.com/kapitalismho/PuriPuly-heart/blob/d27e1bc555c4925a2f3d3cbe4a00ec1b7a101055/benchmarks/asr/issue373-references.json), and [source directory](https://github.com/kapitalismho/PuriPuly-heart/tree/d27e1bc555c4925a2f3d3cbe4a00ec1b7a101055/benchmarks/asr) are pinned to accepted head `d27e1bc555c4925a2f3d3cbe4a00ec1b7a101055`.
- Actual paired product revisions were B0 [`24859f9e6788f4ace4afbdd971193d673441a9c8`](https://github.com/kapitalismho/PuriPuly-heart/commit/24859f9e6788f4ace4afbdd971193d673441a9c8) and B1 [`e7163508bd728cf42043bd65701d10b53eaf3902`](https://github.com/kapitalismho/PuriPuly-heart/commit/e7163508bd728cf42043bd65701d10b53eaf3902). The archival head is not a substitute producing revision.
- [Completion](https://github.com/kapitalismho/PuriPuly-heart/issues/373#issuecomment-6703319850) records integration as `b9fbef5e20024bed9a35aee9441b3495c26ea257`.
- GitHub resolved the archive and both product commits; the pinned report was readable. All five historical files retained in the #441 checkout matched the archive blobs byte for byte.
- The report's original commands apply to its pinned historical checkouts, not the current maintained tree. Historical source/input/license references retain their meaning.

The expected `.tmp/issue373` and `build/issue-373-display` locations were absent in the inspected primary and #441 workspaces before cleanup. The public report retains results and hashes, but this work does not claim availability of those raw traces or screenshots. This is a bounded location check, not a machine-wide absence claim. No historical results were regenerated.

## #441 preservation boundary

The entire registered #441 worktree, its branch/history, ignored evidence, runtime, and analysis sources remain untouched. Its clean tracked source/report tip is `b28e92edfdb06dca7d74f2815a3e964feb2a3002`. That tip was not available through GitHub's commit API and had no remote branch at inspection; do not present a constructed blob URL as a published report.

Preserved local evidence includes frozen input/acquisition records, all original and selected runs, the 28-record selection index, original and corrected scores, failed attempts, Nemotron execution/runtime snapshots, MT/display source variants and observations, lexical evaluation, and Soniox-input diagnosis. Shared primary ASR audio, annotations, models, preparation records, and the Python environment remain available to that checkout. Restricted audio and raw private traces remain outside public artifacts.

### Producing source references

The product baseline is `9f7474b94b885a7a9a4777b43a3a05026e3e248e`. These are retained local provenance references, not claims that every session ran a pristine, independently attested Git tree:

| Selected records | Recorded checkout |
| --- | --- |
| N, all six; S0 EN/KO; E0 EN/JA/KO | `7d73eef770f20da4867f268bf144bd9ead16758b` |
| S0 authorized JA repeat/ZH/both diagnostics; S1 all six; E0 ZH/both diagnostics | `ff7f0f04319cfd63809732440a3db960aeceb2c3` |
| T1 both diagnostics; T2 EN diagnostic | `006b2a2600558c864cde7308f5f90a38a59ea56b` |
| T2 JA diagnostic | `c2b17d25852091f88aa310f4a60478590868c7fb` |

The initial `7d73eef...` checkout did not contain the issue-local runners. Initial cloud cohort provenance records SHA-256 `b8993dcab0fb92ec6df5e9b499062a9b52ebac843c658ede2d932efa45e9701d`; the executed Nemotron snapshot records `b28eeb9ab8d3bec9fc03e4e3d05debc99c564cc49223f9a6eae168851da5de9a`. Both exact sources were subsequently committed at `ff7f0f04319cfd63809732440a3db960aeceb2c3`. The saved Nemotron snapshot remains retained.

The turn-condition runner SHA-256 is `9d1686c1a122f2ab851827e431d8fb0da3802af5ba61a3b92c5647dc5c0e9926`, identical at the two recorded T checkouts and cadence-repair commit `242062dc4e4f7d82a1d512f62bb29e2f8481328f`.

Later retained checkpoints are formatting `af28b95c5ff25541793fd7338c126e35f0130764`, measurement correction `97941ec220c22a862c00a2d1743660965101075c`, MT/display `57fce7b607746a3deebb65ff8b3bf25d7416dece`, lexical freeze `3796f90552ac7ab1f6d3ae3267218c54995e7d46`, lexical execution `93e03e754ed2f85a2e235c2818811f8a10b07507`, and Soniox diagnosis `63c8be10c174390cb6c450bc8b9cf0292517f5ac`.

The MT/display source-snapshot record preserves the invalid initial variant as well as corrected MT, corrected display, and final adapter bytes. The ignored diagnosis probe is also preserved (SHA-256 `6ec1c2f1616ac15cbe782cd5369fc31366b9bb2664a77a1a68d1014b23a4413c`). A final commit alone does not replace these execution-source snapshots.

Most individual cloud records retain the checkout/import location rather than an execution-time dirty diff or runner hash. The score JSONs do not embed a scorer checkout; their reviewed report checkpoints and retained outputs supply context. These pre-existing provenance limits remain explicit. No rerun or invented execution pin was used to fill them.

The frozen #441 manifest hash is `dc9b9e08ddc55c2f1e501de007ceaf8d7c1b24cd529c9c5e0c2f0dcc81acf294`. Its source boundaries remain AMI CC-BY-4.0, IngCrowd CC-BY-NC-4.0, Seoul/OpenSLR 113 CC-BY-NC-2.0, and AliMeeting CC-BY-SA-4.0. The independent benchmark's AMI/IngCrowd source pins and manifests remain unchanged. Retention does not authorize audio redistribution or upgrade annotation-derived references to newly listened ground truth.

## Actual development-workspace cleanup

All paths in this section are relative to the primary workspace. Inventory considered current consumers, reproducibility, preservation value, process use, and ownership before removal. Logical file lengths below are not measured free-disk-space changes.

| Removed material | Logical bytes | Removal basis |
| --- | ---: | --- |
| `.pytest_cache/`, `.ruff_cache/`, `.tmp/__pycache__/`, `scripts/__pycache__/`, `scripts/ci/__pycache__/`, `native/windows_host/__pycache__/` | 2,205,356 | Generated test/lint/bytecode caches, including source-less obsolete script cache remnants; not experiment evidence. |
| `.tmp/agent_restart_handshake_probe.py` | 539 | One-off probe with a previous fixed process identity; imported removed `cli.gui_restart`; no current caller found. Separate GUI state/results remain. |
| `build/gui-smoke-work/build/PuriPulyHeart.pkg` | 25,357,112 | SHA-256-identical to retained `build/gui-smoke-work-2/build/PuriPulyHeart.pkg` (`86e67b1faa3bf916ec66e3777c4e670ba3db109d76a5d0dfed2827525fea6eb4`). Nonidentical EXEs and diagnostics remain. |
| `build/llama.cpp/llama.cpp-b10423/cpu/` | 41,372,000 | All 22 files matched members of the retained CPU ZIP; current runtime selects b11537; no executing binary found under this path. |
| `build/llama.cpp/llama.cpp-b10423/vulkan/` | 94,957,920 | All 23 files matched members of the retained Vulkan ZIP; same current-use check. |
| **Total** | **163,892,927** | **About 156.3 MiB of logical file content.** |

The b10423 archives remain under `build/download-cache/llama.cpp/` as `llama-b10423-bin-win-cpu-x64.zip` and `llama-b10423-bin-win-vulkan-x64.zip`; the sibling b10423 `manifest.json` remains. Removal commands completed successfully. No blanket clean, recursive worktree deletion, or running-process termination was performed.

### Initial retention decisions

| Category | Disposition and reason |
| --- | --- |
| Primary ASR data/models/preparation records; entire #441 checkout and evidence | Retained for independent benchmark use and the unfinished result write-up, including historical execution dependencies and failed attempts. |
| `.venv/`, `target/`, current ASR/overlay build trees, b11537 runtime and download caches, root/service Node dependencies | Retained: demonstrated current consumers, active environment use, or material rebuild/download cost. |
| Settings, credentials, settings backups, isolated GUI state, personal agent/tool state | Retained: private saved state and unrelated ownership; contents were not needed for deletion decisions. |
| `.venv-wsl/` | Pending owner confirmation: lack of an observed WSL process does not prove the environment's custom state is disposable. |
| Historical installers, custom native variants, evidence ZIPs, checksums, source bundles, and install logs | Pending: no byte-equivalent remote/archive replacement established. |
| Other `.tmp/` probes/results, old build/runtime verification and size-audit trees, test audio, coverage/log evidence | Pending: mixed generated material and potentially unique evidence. Exact release-repair commands remain as producing-method evidence. |
| Five root-level personal scripts | Retained and individually ignored: ownership or replaceability not established. New personal tools belong in `scripts/local/`. |
| Registered worktrees | All retained. Two contain unrelated dirty work; signing and #441 contain unintegrated commits. A clean tracked tree does not establish permission to remove its ignored state. |

Registered ownership checked: primary `dev`, current `issue-447`, `issue-441`, signing integration, `special-thanks`, and the existing dated agent worktree. The latter and `special-thanks` had unrelated tracked changes. No worktree was removed or pruned. Some large environment metadata walks were bounded and skipped links; this record is not a complete disk-size audit.

### Owner-authorized additional cleanup for #446

On 2026-11-17, the owner requested more aggressive cleanup before the combined PR and explicitly released three previously uncertain categories: historical runnable bundle dependencies/intermediates, historical installer binaries, and the old WSL environment. This supersedes the corresponding initial retention decisions above. Losing complete historical runnable bundles, exact installer bytes, and any customized WSL package state was disclosed before approval; these removals are not presented as byte-for-byte reproducible cleanup.

| Additional removed material | Logical bytes |
| --- | ---: |
| Primary `target/`, `build/asr-worker-cargo/`, and the historical size-audit Cargo target | 7,790,365,190 |
| Historical GUI/Flet/size-audit bundle dependency payloads | 1,673,812,612 |
| Historical bundle build intermediates | 234,924,102 |
| 26 historical installer executables | 2,968,458,326 |
| `.venv-wsl/` | 456,526,657 |
| 89 generated Python bytecode directories | 143,061,685 |
| **Additional total** | **13,267,148,572** |

The additional total is about **12.36 GiB of logical file content**, not measured reclaimed disk space. The explicit removal inventory contained 551 ignored, untracked paths and 43,133 regular files. No selected path remains. All 354 inventoried retained files kept their original size and SHA-256.

Retained material includes distinct historical application/worker executables, application resources, specifications, TOC/warning/xref records, evidence ZIPs, checksums, logs, source bundles, current staged ASR/overlay runtimes, current model runtimes/downloads, the shared Python environment, personal scripts, private state, and every registered worktree. The primary ASR inputs and the entire #441 checkout/evidence remain untouched. Deleted Cargo intermediates are not substitutes for source provenance; the staged runtime manifests remain.

A private record outside the checkout preserves the exact removal inventory, hashes of the 26 discarded installers, the 93-entry WSL package inventory, and retained-file identities. It contains no credential values and is not a public artifact. Windows removal initially stopped at the WSL `lib64` symlink; Ubuntu removed that link without following it, after which the remaining classified removals completed. No process was terminated.

The old populated-checkout service dependencies and local metadata require the separate [service migration procedure](../services/README.md#local-checkout-migration) when `dev` receives the combined layout. Historical evidence remains evidence, not a supported executable surface.

## Verification boundaries

Checks used CPython 3.14.10 on Windows with the existing primary environment and explicit current-checkout `src`/root import paths. Bytecode generation and pytest's cache provider were disabled; synthetic inputs and outputs used temporary isolated directories.

| Executed check | Observed result and boundary |
| --- | --- |
| `python -B benchmarks/asr/prepare.py --help`, `run.py --help`, `score.py --help` | All exited 0 with usage and empty stderr. CLI availability, not inference. |
| Imports of `prepare`, `run`, `score`, `resources`, `adapters`, and `replay` | All six resolved to the current checkout without constructing models or adapters. |
| `score.py --results <synthetic-saved-inputs> --output <temporary-summary>` | Exit 0. Each EN/JA fixture had 12 reference units and 7 errors: exact, deletion, failed recognition, and empty-success cases. WER/CER were both 7/12; failed/empty counts and duration aggregation matched. Synthetic replay-summary ingestion also matched. |
| Seven selected native-grouping/VAD test functions in `tests/core/test_native_peer_caption.py` and `tests/core/test_vad_gating.py` | 14 parametrized cases collected and passed, no skips. Preserved source coverage, missing attribution, incompatible/broken boundaries, punctuation, VAD progress, and short-candidate rejection. No assertions changed. |
| `prepare.py --language all --offline --check-only --data-root <existing-primary-data>` in the documented `pyarrow==23.0.1` overlay | Exit 0, `preparation_complete=true`, zero acquisition failures. Each language validated 185 hash/header/duration records, all 120 clean crop/reference selections, and continuous replay source PCM correspondence. Offline input validation, not decoding or paced replay. |

No saved real CPU result files were found in the bounded result-location search. Scoring therefore used controlled saved JSONL, not claimed historical inference results. No resident inference or paced replay was executed; their implementation and entrypoints remain unchanged, and no component was extracted.

The initial direct-interpreter preparation check failed with `PackageNotFoundError` for optional `pyarrow` metadata. Only that affected check was repeated after supplying the guide's pinned optional dependency through `uv run --project <current-checkout> --no-sync --python <primary-python> --with pyarrow==23.0.1`, with `UV_PROJECT_ENVIRONMENT` pointing to the existing primary environment. The overlay reported PyArrow 23.0.1; the project environment was not synchronized. Source/audio downloads were disabled.

Even `--check-only` writes `benchmarks/asr/data-preparation.json`. Both attempts first refused an existing current-checkout evidence file, then removed only their generated record. Primary preparation evidence, frozen manifests, and audio were not modified. Temporary scoring/test resources were removed; the optional dependency cache remains managed by uv.

- Sixteen `git check-ignore --no-index` behavior checks passed: existing/new maintained entrypoints are discoverable; all five existing personal scripts, `scripts/local/`, settings/secrets, and ASR data/results/models remain excluded.
- Read-only consumer inspection covered tracked sources and relevant ignored historical scripts. Public #373 pins were reachable; local #441 producing snapshots and retained source blobs were compared where described above.
- No completed acoustic/provider experiment was rerun. No model download, live provider request, production setting change, UI claim, or #441 scientific conclusion is part of this cleanup.

No affected protected-manual drift is intentionally left: the parent explicitly authorizes the directly affected benchmark/contributor navigation updates. Architecture, CLI, privacy, and release contracts are unchanged.
