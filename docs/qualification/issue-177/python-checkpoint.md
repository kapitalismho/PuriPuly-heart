# Issue 177 — Outcome C and source/PyInstaller boundary checkpoint

## Result

**Ready for committed independent review, but not blanket issue/release approval.** The final Windows CPython 3.14.7 candidate passes the assigned software, native-runtime, packaged CPU/GPU/Xet, GUI, release-identity, and isolated installer gates. The remaining gaps require external hardware/services or a safe published-upgrade environment: physical microphone/loopback, HMD/OpenVR, VRChat, cloud-provider credentials/endpoints, and upgrade of an actual published installer identity.

## Candidate and identity

- Final source/test/document snapshot: commit `107ed1820005872561c98809c0d4734fe8374cfd` at `C:/c177/source`. `C:/c177/source-snapshot-identity-107ed182.json` compares all 1,320 tracked files with that commit and records zero mismatches.
- The compiled product source is commit `651f0c3fecc1451f5511c0f3b098a117d2fada9c`. The only paths changed between it and `107ed182` are two tests and `shutdown-qualification.md`; product code is byte-identical, so the already-built payload remains the final product candidate.
- Package: `C:/c177/output/dist-final-651f0c3f/PuriPulyHeart` (1,147 files, 586,292,867 bytes), built with ordinary-GIL official CPython 3.14.7 x64, PyInstaller 6.21.0, Flet/Flet Desktop 1.0.0, and Inno Setup 6.6.1.
- `PuriPulyHeart.exe`: SHA-256 `affc6412869f7a24b669932e5cecca0bd2f899b49309e82b8bfe1c91d55c72f1`.
- `PuriPulyHeartOverlay.exe`: SHA-256 `5e606fb85fed84d9e8205f3df520476e9ed71815273ec5dd6027c36d81ee87a1`.
- `PuriPulyHeartGpuWorker.exe`: SHA-256 `4c75c7ab522d10c24f5d6a8b49fddadd5e97458e2d59d9524b821c42b5a7ddef`.
- Final isolated installer: `C:/c177/output/installer-final-107ed182/PuriPulyHeart-Setup-2.7.0.exe`, 188,527,795 bytes, SHA-256 `65dc41025ec33811b3fa1a55bc57bab3cdbefe52f7994bb147c2df1b33cdb9ae`.
- Compliance source bundle: `PuriPulyHeart-soxr-third-party-source-bundle.zip`, 483,046 bytes, SHA-256 `159af704190e6c0f604e3d0c11f0b6a92f3ef082caf103cc8a187156dd42d50e`.
- `C:/c177/release-provenance-107ed182.json` is the output of the actual `verify-build` gate and binds version `2.7.0`, tag `v2.7.0`, final source SHA, installer, main executable, GPU worker, shipped overlay, and compliance bundle. Its SHA-256 is `3cfc85b8942a97d6fad6039a607894fff6a523320383fa4c56908d272d9da8b9`.

## Distribution and native evidence repairs

| Repair | Resolution and evidence |
|---|---|
| Clean-export Python license | **PASS:** the committed blob, clean `git archive` export, and final snapshot copy of `PYTHON-3.14.7-LICENSE.txt` are byte-identical: 35,407 bytes, SHA-256 `935cf13e19f8c31b497d20b05d73623431a226b230c3599bc30fa3348979bc68`. `C:/c177/license-clean-export-proof-107ed182.json`. |
| Packaged licenses and soxr compliance | **PASS:** the actual `verify-packaged-licenses` gate found 26 license payloads and verified both `COPYING.LGPL-2.1.txt` and the source bundle, including pinned hashes for python-soxr 1.1.0, libsoxr 0.1.3, and zeroconf 0.150.0. `C:/c177/final-packaged-license-gate.json`, SHA-256 `e2296a770b5bcea93961d1b429b12f79871298266558c43005b21e109701d019`. |
| Real packaged Xet cancellation/restart | **PASS:** the exact packaged EXE downloaded the pinned public 1,692,554,208-byte Qwen GPU asset through the Xet worker. Cancellation after observed 1% progress completed in 0.068 s with no installed model, staging residue, or worker survivor; restart completed in 59.24 s, installed the checksum-verified file (`c75a961b…e4e1`), and owner close left no survivor. `C:/c177/xet-final/packaged-xet-cancel-restart.json`, SHA-256 `ac211cc3be454341f69816699dc6293662365e602dc8d8a3de37f589c7700576`. |
| Packaged CPU failure contract | **PASS:** the exact packaged command was invoked with missing model/audio roots, exited 1, and wrote a valid `status=failed` report with `RuntimeError` and traceback instead of silently succeeding or omitting evidence. `C:/c177/cpu-packaged-failure-107ed182.json`, SHA-256 `bc84e62fb27c05e67f28c88e7f3283cdc40a07a132710a8ffcca0cf52db2f744`. |
| Provenance binds the shipped overlay | **PASS:** the staged and shipped `dist` overlay are byte-identical at SHA-256 `5e606fb8…87a1`; `verify-build --overlay-exe` was run against the shipped `dist` file, not PyInstaller staging. `C:/c177/final-evidence-index-107ed182.json` records the equality, and `C:/c177/verify-build-gate-107ed182.json` records the successful actual gate. |

## Definitive evidence

| Check | Result and evidence |
|---|---|
| Full suite | **PASS:** 6,083 collected, 6,046 passed and 37 skipped, with zero failures/errors in 210.331 s on CPython 3.14.7. JUnit `C:/c177/full-suite-107ed182.xml`, SHA-256 `8e2ed4ecf2f88092104d92df6b0095add622d9dd022fa47115ddf69f5d2b1fc7`. |
| Real CPython native workload | **PASS:** ordinary GIL, AMD64; ORT 1.28.0 CPU/Azure, NumPy 2.5.1; five real Silero VAD inferences, three SmartTurn ONNX inferences, and native soxr 48 kHz→16 kHz resampling. `C:/c177/native-probe-314.json`. |
| Packaged CPU ASR | **PASS:** three pinned installed models/public WAVs produced nonempty final texts of 79/50/105 characters, with CPU RTF 0.039/0.030/0.130. The report binds frozen CPython 3.14.7 and the exact final EXE hash. `C:/c177/cpu-packaged-107ed182.json`, SHA-256 `6efd177a2f8b130e6592643a65be8afe10149bfb994ef79cd79e387eb1ca9990`. |
| Packaged GPU production composition | **PASS:** Radeon RX 7900 XTX (`vulkan-index-0`); real Self/Peer inference, in-flight handoff, worker-failure controller recovery, fresh worker reactivation, authoritative Self/Peer finals, and clean shutdown all completed through the production composition. `C:/c177/gpu-production-composition-107ed182.json`, SHA-256 `574439c150ae3814b0bf59870bf24f2c35cdf7ef53a131e0ab4befe6670b17fd`. |
| Shutdown and actual host loss | **PASS:** three isolated Unicode-CWD/profile/config/cache GUI runs were bound to exact host/viewer PID ancestry; double `WM_CLOSE` produced exit 0 and no survivors. In the separate actual-host-loss run, only host PID 29940 was killed; its exact cache viewer child PID 24036 disappeared with exit 15 and no survivor/manual cleanup. The package EXE hash is bound in `docs/qualification/issue-177/shutdown-qualification.md`. |
| Release headless gates | **PASS:** from an unrelated Unicode/space CWD, `--version`, `gui-startup-check`, local-Qwen, soxr, hf-xet, Unicode/space config telemetry-disable, and the strict packaged process-capture helper all exited 0. The helper reported native process-specific capture and no fallback. `C:/c177/headless-gates-651f0c3f.json`, SHA-256 `ac4e34bae9a9eaa563caded80297afa811d0bf10a0f0ae67ad02d7ffbb33839c`. |
| GUI | **PASS:** final package and final installed main GUI plus the packaged desktop-overlay preview were bound through exact PID ancestry to their visible Flet HWND, captured directly, closed through `WM_CLOSE`, exited 0, and left no owned survivors. `C:/c177/package-gui-final.{json,png}`, `package-overlay-final.{json,png}`, and `installed-gui-final.{json,png}` visibly show the localized dashboard/overlay without missing glyphs. |
| Native/package inventory | **PASS:** CPython 3.14 and cp314/abi3 native extensions, ONNX Runtime, sherpa, ProcTap, hf-xet, soxr plus sibling DLL, standard PortAudio, OpenVR, CPU/Vulkan llama.cpp, overlay, and GPU worker are present. Unsupported ASIO is absent. Recursive filesystem and ZIP/WHL inspection found none of the excluded development packages. `C:/c177/artifact-inventory-107ed182.json`, SHA-256 `b9b5cb29c2cd688b51f37f6cf1f9565bb3b221317f5e7a210b2353bf0573ac1f`. |
| Isolated installer | **PASS, with the isolation exception documented below:** fresh alternate AppId `{7E641A13-223B-4F0C-A779-8F9096F92177}`, isolated app-data/dir/group, current-user silent install to a Unicode path, installed headless and visible GUI checks, exact installed payload/compliance hashes, deliberate `soxr.dll` corruption repaired by reinstall, corrected explicit-config telemetry-disable preserved through reinstall, and silent uninstall removed the isolated install directory, app data, Start Menu group, and uninstall registry key. The production installer/registry identity was not used. `C:/c177/installer-final-{initial,maintenance,uninstall}-verification.json`. |

### Installer isolation exception

The first installed-app verification mistakenly ran `installer-telemetry-preference disable` once **without** `--config`. The command therefore resolved the stable production path `%LOCALAPPDATA%\puripuly-heart\settings.json`; because that file existed, the application loaded its canonical settings, persisted them with telemetry disabled, and reloaded them for verification. The command returned 0. There is no pre-command snapshot, so this record cannot establish the prior telemetry value or claim the canonical rewrite preserved every original byte. No automatic restoration was attempted.

The subsequent diagnostic search of that production settings file used the exact case-sensitive pattern `"telemetry"|"enabled"` to explain why the intended isolated file still showed telemetry enabled. It exposed only the matched telemetry/integrated-context snippets (including post-write telemetry `enabled: false` and telemetry `anonymous_id: null`); it was not evidence of isolation and is not used as qualification evidence. The retained installer qualification was rerun with explicit `--config C:\Users\salee\AppData\Local\puripuly-heart-c177-final-7e641a13\settings.json`, which is the command recorded in `installer-final-initial-verification.json`.

Separately, packaged headless commands run without `--config` initialize the default file logger under `%LOCALAPPDATA%\puripuly-heart`; they may therefore have created or appended the production `puripuly_heart.log`, although those commands did not load production settings. GUI, GPU, Xet, telemetry-persistence, installer, reinstall, and uninstall evidence used explicit or environment-isolated roots as recorded in their reports. Installer logs show only the alternate AppId, Unicode install root, and `puripuly-heart-c177-final-7e641a13` app-data root; uninstall removed that isolated root. Accordingly, the defensible claim is that the production **installer/registry identity** was untouched, not that the production user-data directory was untouched.

The Director disclosed the error to the user. The user explicitly selected **enable telemetry**. The Director then ran the final candidate with `--config C:/Users/salee/AppData/Local/puripuly-heart/settings.json installer-telemetry-preference enable`; it returned 0 after the canonical settings owner's built-in persist-and-reload verification. This is an authorized change to enabled, not restoration of an unknown previous preference or anonymous identifier. No further production configuration inspection or log cleanup was performed.

## Release workflow mapping

`.github/workflows/release.yml` (SHA-256 `b2ca728af578a2ef7584694e61d0d921928f074e85c61b58e055f2228ade093e`) builds with the pinned interpreter, runs the packaged headless/license/process-capture gates, stages the compliance payload, and binds the shipped overlay in `verify-build`. The final local run exercised those actual gate implementations and command paths against the exact candidate.

**No GitHub Actions workflow run occurred.** This is a local exact-workflow gate replica and must not be represented as hosted CI evidence.

## Subsequent build-only SDK isolation repair

A rebuild of the later shared-shutdown source `459e0030` logged one public `GET https://huggingface.co/api/agent-harnesses` returning `200`. The retained log and unknown authentication-header status are documented in `shutdown-qualification.md`. Source inspection attributes the call to `collect_submodules("huggingface_hub")`: PyInstaller imports the SDK's CLI package in an isolated child; its output-mode initialization asks the agent-harness registry for uncached metadata. This is not an intentionally executed cloud workload.

`build.spec` now sets `HF_HUB_OFFLINE=1` in the build process before application imports and package collection. PyInstaller's isolated children inherit that supported SDK setting. It does not remove collected modules, suppress build exceptions, modify installed application code, or force offline behavior on production model downloads.

A cold-cache reproduction used a recording/rejecting HTTP boundary inside the actual PyInstaller isolated child, with socket/DNS rejection as a backstop and fresh profile/cache roots. The baseline attempted the registry GET once, but no request reached the transport. SDK offline mode produced zero HTTP/socket/DNS attempts. Both collections returned the same 182 module names, including `huggingface_hub.cli`; the canonical module-set digest is `9eccf4f2ad757c975910c81ec1dd17cde65ffef145d6aa905d7c8e1be325ccd3`. Identities: CPython 3.14.7, PyInstaller 6.21.0, huggingface_hub 1.26.0, HTTPX 0.28.1.

The post-edit check started without the offline flag, executed the compiled assignment from the actual spec, and then ran the real SDK collection in the guarded isolated child. It observed the inherited flag and SDK constant enabled, no network attempt, and the unchanged module set. This exercises the changed policy and collection path, not the complete spec, `Analysis`, `EXE`, `COLLECT`, or a new package rebuild.

Current evidence record: `.tmp/issue-177-build-isolation/final-evidence-v2/record.json`, byte SHA-256 `53bc4b8f95db85ea51874b50ebd3735648d4c0b8980a6f5ff24de431dbe6f602`. Its external `SHA256SUMS.txt` has byte SHA-256 `007bf5397f603fe7887dce256a9e04bb35d517a7c4efdf6a99de9d0787909c23`. These bind the exact witness scripts, original reports, previous evidence projection and edited `build.spec` bytes (`0d07711cfc2a10c4b4d36afd39d2e2daaffe21f770a65309d07ee9cf1c251cf0`).

The original baseline and post-edit reports remain unchanged, with actual byte hashes `dc1386a1452f42b575cf6c5d6423e4ac9afac701f108bbc66697db50e18cd67c` and `32f16327337a952820faf0de22eb8cf902d6a1ff82dfa800a996c4b9eb3b0a51`. Their internal self-digest fields were calculated before those fields were inserted and are not final-file identities; the separate final evidence record corrects that metadata ambiguity without rewriting observations or claiming another execution. No full-build network-free claim follows from this focused repair.

Independent review identified `zero_real_egress` as a literal constraint in the legacy witness dictionaries rather than a measured result. The current projection excludes it from its proof set and pass claims. It retains the derived attempt counts, inherited flag/SDK state and module-set equality, and separately records the process-local HTTP/socket/DNS guard mechanism. No OS-level egress monitor ran. Historical reports and the previous projection remain unchanged; this correction changes evidence metadata, not the execution history.

## External gaps retained

- **Published upgrade:** deliberately not attempted. The production AppId is occupied and this workstation has no disposable clean VM/snapshot; forcing it would risk the real installation. Fresh isolated install/reinstall/uninstall evidence is not published-upgrade evidence.
- **Physical surfaces:** no controlled physical microphone/loopback source, HMD/OpenVR runtime, or VRChat session was supplied.
- **Cloud surfaces:** no qualification credentials/endpoints were supplied.

Outcome C's independently reproducible software/native checkpoint is ready for a fresh committed review. These external gaps remain explicit and prevent treating this checkpoint as blanket completion of issue #177 or final release approval.
