# Desktop features: separate upstream PR drafts

Base: upstream `dev` eac4fb465665f15edcaa7d794f467411185ca947 plus QQ diagnostics branch 677a9ee837b7e19abfa758998e86f8bce6a28b3a (PR #226). Feature branch: `feat/desktop-cn-download-audio-test`. These drafts are not submitted PRs.

Acceptance details and PASS / FAIL / NOT_TESTED boundaries: [desktop-feature-test-results.md](desktop-feature-test-results.md). Module PRs need the QQ baseline and shared diagnostics/settings groundwork; separate feature commits are not assumed to be dependency-free cherry-picks.

## A — feat(models): add scoped Gemma download sources and verified offline import

### Problem / root cause
Pinned Gemma downloads are inaccessible for some users; selecting a mirror cannot merely change presentation or silently redirect public files back overseas.

### Changes
Source settings flow into the owned downloader process before SDK initialization. Anonymous, TLS-verified mirror requests reject foreign redirects and validate Range responses. Bounded timeouts/retries, progress bytes/current file, cancellation, and verified offline import retain fixed revision/size/hash and atomic installation. Existing models are protected. Failures use normal persistent ModelDownload diagnostics.

### Tests / limitations
Small fixtures exercise import, manifests, hash/size errors, cancellation, Range, retry and disk failure. Real probes of both pinned mirror files returned redirects through huggingface.co to us.aws.cdn.hf.co. Mirror reachability PASS; mirror-contained download FAIL (explicitly rejected); full multi-GB download without proxy/TUN NOT_TESTED. No alternate quantization or ModelScope substitution.

## B — feat(desktop): add independent pinned quick translation window

### Problem / root cause
Output-only subtitles do not offer a convenient input surface to single-monitor players.

### Changes
An opaque top-level Flet child uses private inherited pipes, existing manual translation ownership and UUID correlation. Pin/bounds persist through settings; hide/reopen reuses one child; owned shutdown avoids orphan processes. Blank/pending requests are suppressed, same text can be sent later, failures preserve input, no extra secrets/providers/forced OSC. Failures use persistent QuickTranslate diagnostics.

### Tests / limitations
Owner/UI fixtures exercise duplicate IDs, reordered results, same-text submissions, retry and bounds. Windows GUI Chinese candidate Enter did not submit; subsequent Enter submitted; Shift+Enter inserted a newline. Backend-not-ready was shown with input retained, and remained in logs after hiding. Japanese IME, actual translation backend/model switching and game fullscreen need manual verification.

## C — feat(audio): add bounded local microphone recording and playback

### Problem / root cause
An input meter cannot reveal what PCM was captured or distinguish device/format errors from recognition issues.

### Changes
The existing microphone-test owner captures bounded in-memory raw and pre-VAD/STT PCM using actual device metadata and shared format conversion. Default 10/max 30 seconds, no automatic gain, no cloud calls or default export. Playback uses a local output stream, pauses SELF/LISTEN, and only restores still-current user intent. Closing releases devices/buffers. Failures use persistent AudioTest diagnostics.

### Tests / limitations
Fixtures cover amplitude preservation, exact stereo conversion, silence, format changes, max duration, playback failure, cancellation/thread cleanup and no forced restoration. Windows capture opened Realtek microphone array through WASAPI at 48 kHz stereo and stopped at 10 seconds. Listening quality, hardware disconnection and real normal-ASR/VR runtime require manual verification.

## Compatibility / security

QQ PR #226 is unchanged. No Broker, free quota, credential protocol or production configuration changes. No production issue-key requests. The 2.6.x issue-key tool remains outside these commits. Static safe diagnostic fields reuse normal UI logs and bounded rotating files. Locale keys are supplied for en, zh-CN, ja, ko, ru. Preview configuration and secret storage are both isolated; this is not a formal Release.
