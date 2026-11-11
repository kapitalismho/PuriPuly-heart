# Native distribution size

The approved scope and before/after execution evidence are recorded in [issue #400](https://github.com/kapitalismho/PuriPuly-heart/issues/400). Its [plan B decision](https://github.com/kapitalismho/PuriPuly-heart/issues/400#issuecomment-6794564407) preserves application features, CPU/GPU execution, recovery, offline use, and model/runtime download policy. Separately provisioned model weights are not application-payload savings.

## Packaging boundary

The native builder runs `native_distribution prune-runtime` on fresh dependency staging before Python compilation, dependency indexing, archiving, and final inventory generation. It does not modify the developer environment or prune an already compiled package.

- Qualified NumPy 2.5.1 and SciPy 1.18.0 test subtrees are selected explicitly. Runtime helpers such as `numpy.testing` and `scipy._lib._testutils`, licenses/notices, and native files remain. `scipy/special/tests` remains because an upstream precomputation utility imports it. Other packages and unlisted paths are unchanged.
- Exactly three DLL copies under `site-packages/Scripts` are excluded after validating their wheel ownership, bytes, and retained `sherpa_onnx/lib` counterparts. The import libraries and both distinct ONNX Runtime distributions remain.
- Packaged Japanese CTC explicitly preloads ONNX Runtime from `site-packages/onnxruntime/capi`. The builder no longer copies those DLLs to `_runtime/local_qwen`. Directory registration, explicit preload ordering, retained handles, and source-mode discovery remain owned by `local_sherpa_runtime`.
- The affected wheel RECORD files describe the retained payload. Pruning evidence records original wheel metadata identities, exact removed files, retained exceptions, and canonical DLL identities. Dependency upgrades require reassessing the qualified exclusions; unknown versions or mismatching ownership/bytes fail preparation.

The omitted upstream dependency self-tests are not available in the installed application. This does not remove application tests from the repository or introduce a first-use runtime download.

## Installation and upgrades

The experimental build writes `BuildRoot/evidence/native-packaging-pruning.json`. Its `removed_files` contains actual staging removals. Its `obsolete_files` additionally identifies the two former `_runtime/local_qwen` copies from their retained canonical DLL bytes; these copies are not staged in a new build.

Release assembly passes that evidence to `native_installer_cleanup` through `--packaging-pruning`. Keep the matching evidence when using `-SkipNativeBuild`; a finished artifact alone is not sufficient. The generated cleanup include remains pinned to the final native artifact manifest.

Cleanup combines the selected obsolete identities with the existing legacy inventory. A path owned by both sources can retain their independently known size/SHA-256 alternatives. The cleanup helper hashes the protected open file once and deletes it only if a complete approved identity matches. Current packaged paths, protected user state, modified or unknown bytes, locked files, hardlinks, and reparse targets are preserved. Legitimate deletions can proceed despite another file being preserved, and a later setup run can retry.

This is not general directory mirroring or permission to remove files absent from the newest inventory. Uninstaller data and unrelated historical residue are not selected for deletion. The llama CPU/Vulkan trees, ASR worker runtimes, model assets, and provisioning behavior are unchanged.

## Comparable measurements

Record source revision, locked dependency/native inputs, build profile, artifact manifest identity, and installer identity for both builds. Rebuild changed application code and archives. Disclose reused inputs and failed or incomplete attempts.

Report these measurements separately:

- Logical bytes in the final artifact inventory, with the manifest itself identified separately when it is not self-inventoried.
- Compressed installer EXE bytes using the same compression and signing profile.
- Fresh installed logical and allocated bytes, separating installer-generated files.
- Upgrade footprint and selected cleanup results, including preserved modified/unknown files and generated uninstall data.
- Separately provisioned model weights or any deferred downloads; moved bytes are not eliminated bytes.

The compressed sizes of selected ZIP members are candidate contributions, not exact rebuilt ZIP or installer savings. Duplicate logical bytes may already compress well. An existing upgraded directory is not a clean-install baseline, and matching file lengths are not a complete integrity audit.

Size results require affected packaged behavior evidence: startup, scientific audio conversion, VAD/turn inference, Japanese CTC decoding and DLL provenance in both initialization orders, and isolated installation/update/uninstall outcomes. A smaller directory or successful build is insufficient.
