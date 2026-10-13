# Desktop preview acceptance — 2026-10-13

## Identity / protected baseline

- Fork: RICHARDwuxiaofei/PuriPuly-heart; branch `feat/desktop-cn-download-audio-test`.
- Package code commit: `18c3db591011a49edddbec8e6b177fbb1acdaa85`. Any subsequent documentation-only commit does not change this package's application code.
- Upstream dev baseline: `eac4fb465665f15edcaa7d794f467411185ca947`; QQ baseline: `677a9ee837b7e19abfa758998e86f8bce6a28b3a`.
- QQ PR #226 remained OPEN against dev, with head 677a9ee. No new upstream PR, merge, production deployment or Release was made.
- Module commits: A `88fee86f`, B `c145bbb9`, C `6eae25f0`; later acceptance/security/packaging corrections are separate commits on this branch.
- Original tool retained unchanged: `D:/CODE/PuriPuly-heart-2.6.1/PuriPuly-heart-2.6.1/scripts/qq_issue_api_key.py`, SHA256 `d439bec49a6e4a99e010b31bd26dcd7d6cd3d666484cb47a1dcb578ac5d38ffe`. It and its configuration are not included in these commits or the preview.

## Test matrix

| Scope | Result | Evidence / boundary |
| --- | --- | --- |
| Final code relevant regression | PASS | 2,246 passed, 2 integration tests skipped by default; 34 existing deprecation warnings. Architecture, configuration, UI/overlays, QQ, models, audio, quick translation, logs, packaging. `desktop-package-code-regression.xml` outside Git. |
| Windows process integration | PASS | 4 real-process tests passed separately with INTEGRATION=1: kill-on-close assignment, ten owned process cycles, normal parent exit and parent loss. These are real hidden helper processes, not ten GUI/game cycles. |
| Final targeted acceptance | PASS | 25 tests: child loopback/credential filtering, recorder/playback, privacy-safe persistent logs, profile isolation, packaging. Tests overlap the regression suite and are not added to its count. |
| Ruff / Black / whitespace | PASS | No Ruff errors; changed Python files checked/formatted; git diff whitespace check passed. |
| Model import and download behavior | PASS (fixtures/Mock) | Exact pinned contract, hash/size errors, missing/wrong files, installed model protection, valid import/manifest/no redundant download, Range validation, bounded timeout retries, cancellation and disk-full. Small assets replace multi-GB data only in tests. |
| Mirror specified files reachable | PASS (current network) | Both real fixed-revision files returned responses to Range bytes=0-15. Proxy/TUN state was not changed. This is not a mainland direct-download test. |
| Mirror-contained actual download route | FAIL / rejected safely | hf-mirror.com redirected through huggingface.co to us.aws.cdn.hf.co. The mirror downloader rejects the foreign redirect before contacting it. It must not be advertised as a working domestic large-file source for this revision. |
| Full pinned Gemma download/hash/import without proxy/TUN | NOT_TESTED | No repeated 4.27 GB download, no matching existing model in the normal model directory, no alternate quantization or ModelScope substitution. Use official source or verified offline import. |
| Model GUI failure / close / normal logs | PASS (source Windows GUI) | Local import with no files displayed the Chinese missing-file explanation. After closing the dialog, the normal logs page and rotating file retained ModelDownload missing_file. |
| Quick window input/error/lifecycle | PASS (source/final Windows GUI + behavior tests) | Opaque independent window and pin switch; Microsoft Pinyin composition Enter did not submit, next Enter submitted, also in final 18c3db59 with the main window minimized. Shift+Enter inserted newline in source GUI. Backend-not-ready retained input and emitted normal QuickTranslate diagnostics. Hiding retained its normal log; main exit closed the child. UUID reordering, repeat-text, retry and bounds are behavior tests. |
| Main minimized / independent window | PASS (final Windows GUI) | After minimizing the main test window, the opaque quick window remained visible with its own input/buttons. Its actual TCP listeners remained loopback only. |
| Quick hide/reopen | PASS (final Windows GUI) | Hiding removed the window from visible surfaces; reopening reused the same child PID 33988 and retained input/result state. The error remained on the normal logs page while hidden. |
| Final GUI exit | PASS | Closing the main window closed the hidden quick window. Process inspection found neither owned PID 30512/33988 nor final-package Flet processes; shutdown reported failure_count=0 and status=ok. |
| Japanese IME, game fullscreen, DPI/monitor changes | NOT_TESTED (real GUI/game) | Do not claim unit/behavior tests prove these. Exclusive fullscreen is not guaranteed. No injection, hooks or anti-cheat bypass. |
| Actual microphone capture/playback | PASS (Windows preview 00826ae6) | Realtek microphone array, WASAPI, 48,000 Hz, stereo, 10 seconds. Both raw and pre-VAD/STT PCM output streams completed. No cloud calls, export or gain change. Later classification and child-network changes were separately tested. Listening quality is not assessed by these tools. |
| Audio failure stage / closed-surface persistence | PASS (final Windows GUI + Mock/behavior) | Output OSError produces playback_failed, not capture-start failure. After closing the recording window, the normal logs page and rotating file retained playback_failed / OSError / os_code=5; the secret-marker exception text was absent. The existing CLI confirmed capture off and no recording buffer. Behavior tests additionally remount log views. |
| Final controlled audio-failure GUI simulation | PASS (Windows GUI with simulated output failure) | After the user manually handled the Windows prompt, final 18c3db59 recorded real 10-second input, then a process-local simulated OutputStream OSError produced the correct Chinese playback-failed message. Persistent logs contain playback_failed / OSError / os_code=5 and omit the secret-marker exception text. No real hardware-disconnection claim; no automated permission response or security-setting change. The package does not contain the external fault-probe script. |
| Final EXE/runtime startup | PASS to initialized main UI | Bundled CPython/Flet, required prompts, fonts/locales, llama.cpp provenance and ASR/Xet SDK imports. Startup completion was recorded. System permission prompts still require the user; this is not official Setup validation. |
| Listener scope | PASS (actual process inspection) | Final test process TCP listeners were only 127.0.0.1 / ::1. New presentation child overrides inherited public bind/web flags and does not inherit API-key/token/PSK variables. System proxy/VPN/firewall settings unchanged. |
| Existing QQ / manual text / desktop and VR overlay behavior | PASS (automated regression), NOT_TESTED (production/live VR/ASR) | No real user credentials, production Key issuance, paid cloud recognition/translation or repeated free quota claims. Real model/GPU-ASR/native VR worker/game paths still need manual validation. |

## Safe acceptance log excerpts

Source Windows GUI, visible after corresponding surfaces closed:

```text
2026-10-12 23:41:12.671 [ERROR] [ModelDownload] failure stage=installation code=missing_file exception_type=GemmaProvisioningError
2026-10-12 23:57:31.257 [ERROR] [QuickTranslate] failure stage=submission code=backend_not_ready exception_type=RuntimeError
```

Actual preview playback (raw, then pre-VAD/STT PCM; INFO, not a recognition/upload claim):

```text
2026-10-13 00:26:51.710 [INFO] [AudioTest] lifecycle stage=playback code=playback_started
2026-10-13 00:27:02.308 [INFO] [AudioTest] lifecycle stage=playback code=playback_completed
2026-10-13 00:28:51.593 [INFO] [AudioTest] lifecycle stage=playback code=playback_started
2026-10-13 00:29:02.223 [INFO] [AudioTest] lifecycle stage=playback code=playback_completed
```

Final Windows GUI controlled output failure (not a real hardware failure):

```text
2026-10-13 01:04:02.819 [ERROR] [AudioTest] failure stage=playback code=playback_failed exception_type=OSError os_code=5
2026-10-13 01:08:48.819 [ERROR] [QuickTranslate] failure stage=submission code=backend_not_ready exception_type=RuntimeError
```

No QQ number, credential, Key, token, proxy password, signed URL, raw response or audio was added to technical diagnostics. Log rotation remains the project's normal bounded asynchronous mechanism.

## Package / limitations / next actions

The package is at `D:/CODE/PuriPuly-heart-2.6.1/Preview-final-package`, with `PuriPulyHeartPreview.exe`, `BUILD.json`, README and this report. Its shareable archive is `PuriPulyHeart-preview-18c3db59.zip` beside this directory. Prefer extraction to a short path; deeply nested Windows paths can still exceed SDK filename limits. This is a portable bundled CPython 3.14.3 + locked Flet desktop build, not the pinned native Setup/Release toolchain.

Data/model/log paths are under the package's `preview-data`; the Keyring namespace is `puripuly-heart.test.desktop-preview`. Production defaults are untouched; encrypted-file paths outside the isolated directory are rejected. No environment fallback to production keys, no telemetry opt-in, no copied personal config or recordings. Multiple preview copies share the test Keyring namespace, but not the production namespace. The archive excludes preview-data and the external fault-probe script.

Manual follow-up: handle any Windows security prompt yourself; configure a separate test backend without re-claiming existing QQ quota; test offline import with the exact two pinned files, Japanese IME, actual translation, main minimization and borderless game behavior. Optional ASR-on-recording and WAV export were not added; recording/playback stay local and in memory. Native VR/GPU ASR binaries and models are not certified by this preview.

The four failed build directories were deleted only after explicit user approval. Source worktrees, the old key tool, the earlier working Preview-ready and Preview-delivery copies, and the final package were retained. They can be distinguished by their BUILD.json commits.
