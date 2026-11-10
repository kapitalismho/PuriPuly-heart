# Issue 386: managed photo access and comparison

Execution date: 2026-11-10. Related: [#386](https://github.com/kapitalismho/PuriPuly-heart/issues/386), [#342](https://github.com/kapitalismho/PuriPuly-heart/issues/342), [#385](https://github.com/kapitalismho/PuriPuly-heart/issues/385), [#389](https://github.com/kapitalismho/PuriPuly-heart/issues/389).

## Status and scope

The primary Relay implementation works through the existing photo application owner, provider boundary, native image preparation, and native offscreen rendering. Relay remains the intended hosted route. Personal Gemini and ChatGPT routes remain independent; failures never change funding automatically.

**Issue 386 is not ready to close.** Actual headset/first-display timing is unavailable; the personal ChatGPT account is quota-limited; a management credential for live restricted OpenRouter enforcement is unavailable. Valid schema completion also does not establish usable rendering or complete text coverage. No merge or production deployment occurred.

There is no new Python product OCR UI, funding setting, CLI setting, or localization change. Source-only comparison selection is available through `PURIPULY_PHOTO_BENCHMARK_ROUTE`: `personal` (default), `managed` (Relay), `openai-api`, or `managed-openrouter`. Packaged applications ignore this comparison override. The direct baseline reads the existing secure-store `openai_api_key`; no operator key is embedded or distributed.

## Implementation references

Base checkout: `6dbd432c21321c5a62e2a9de227a610d93bbee09`, with uncommitted candidate changes. No implementation PR/commit has been published. [Machine-readable evidence](photo-managed-comparison.json) pins the measured source files by SHA-256; it omits image contents, recognized text, credentials, and installation identifiers.

- [Provider transports](../src/puripuly_heart/providers/photo.py): dedicated Responses SSE connections for OpenAI, Relay, and OpenRouter. Completion drains the HTTP body for connection reuse; early consumer closure releases the response/pool. Paid Luna requests are not blindly replayed.
- [Photo adapter](../src/puripuly_heart/app/adapters/photo_provider.py), [runtime composition](../src/puripuly_heart/composition/application_runtime.py), and [lifecycle](../src/puripuly_heart/core/photo/lifecycle.py): existing account/network owners, shared Relay token owner, accepted-pull and stale-route fences, local timing, and distinct first-drawn evidence.
- [Relay handler](../agent-relay/src/photo.ts), [admission/settlement](../agent-relay/src/usage.ts), [migration 0006](../agent-relay/migrations/0006_photo_admission.sql): server-selected endpoint/model/prompt/schema; strict photo-only contract; installation-bound, atomic allowance/concurrency admission; owned, durable settlement; no per-photo Broker or telemetry requests.
- [Broker photo access](../broker/src/photo-access.ts), [provider management](../broker/src/openrouter-management.ts), [migration 0031](../broker/migrations/0031_photo_managed_access.sql), and [local credential owner](../src/puripuly_heart/app/adapters/photo_access.py): existing first-binding protection and durable issuance/recovery owners; separate photo policy; disabled, policy-verified keys before delivery; persist before activation ACK; upstream revocation and local pending-revocation fence. Activated/revoked allowance is conservatively reserved for its period rather than recreated after key loss.

Chatbot protocol v6 and its independent accounting are unchanged. Relay deployment defaults `PHOTO_SERVICE_ENABLED` to false. Enabling requires the existing `PURI_TOKEN_SECRET`, server-side `OPENAI_API_KEY`, and migration 0006. `PHOTO_POLICY` is a JSON object with the fields below. The secondary Broker path defaults off and requires `PHOTO_OPENROUTER_ENABLED=true`, `OPENROUTER_MANAGEMENT_API_KEY`, `OPENROUTER_PHOTO_GUARDRAIL_ID`, and migration 0031. Its `PHOTO_OPENROUTER_LIMIT_USD` defaults to 0.25 per UTC day, independently of Relay slots. A direct key holder can use other prompts within the provider's actual permissions; local OCR behavior is not a security boundary.

## Supplied-photo experiment

The owner supplied 10 VRChat originals and 13 difficult images under `C:/tmp/puri_capture_probe`. All 23 were used as inputs. The 23 `runs/mnv4/final` images were visual references only, not translation inputs or authoritative ground truth. Originals were preserved. JPEG inputs were decoded to lossless RGB PNG captures because the existing capture watcher accepts PNG; the native application then performed its normal 1280×720 letterbox/JPEG preparation. Original and capture hashes are recorded separately.

Each image ran through direct OpenAI and Relay, once with a cold client connection and once with its immediately reused connection: **92 paid requests**, 23 per route/state. Route order alternated by image. Both routes used the same authorized API key/project, HTTP Responses SSE, `gpt-6-luna`, Korean target, canonical prompt/schema, no reasoning, and an 8192-token output cap. Every completed response reported `gpt-6-luna`.

Cold means a new client HTTP connection with valid credentials, not necessarily a cold model cache. All 46 cold samples opened one TCP connection; all 46 warm samples reused it. Relay warm reuse establishes app→Relay reuse, not a globally pooled Relay→OpenAI connection. Relay adds its installation-derived safety identifier. Observed Cloudflare edges were ICN for direct calls and HKG/NRT for Relay; these do not identify Worker execution or upstream model regions.

Broker setup took 615.4 ms and one Worker call. The 46 Relay photos each used one Relay invocation and no Broker calls. Direct photos used no PuriPuly Worker. All 46 Relay admissions settled durably with usage, totaling 19,780 output tokens and an estimated $0.0133515 under the repository's configured rates, not an invoice.

### Latency

Milliseconds. Absolute values are route medians across images. Delta is the median of **matched per-image Relay minus API differences**, not subtraction of the two route medians. Different text grouping, cache state, model scheduling, and edge routing prevent interpreting the entire difference as a network-hop cost. These are small, heterogeneous samples, not stable tail estimates.

| State | Metric | API median | Relay median | Paired delta | Pairs |
| --- | --- | ---: | ---: | ---: | ---: |
| Cold | Accepted → first text delta | 1620.3 | 2698.3 | +1095.1 | 23 |
| Cold | Send → first text delta | 1198.4 | 2057.8 | +759.4 | 23 |
| Cold | Accepted → first valid block | 1839.8 | 2932.8 | +1034.8 | 23 |
| Cold | Accepted → schema completion | 4150.4 | 4764.9 | +1027.4 | 23 |
| Warm | Accepted → first text delta | 1619.6 | 2122.8 | +443.8 | 23 |
| Warm | Send → first text delta | 1164.9 | 1675.5 | +441.7 | 23 |
| Warm | Accepted → first valid block | 1897.2 | 2224.0 | +449.5 | 22 |
| Warm | Accepted → schema completion | 3750.0 | 4020.1 | +395.7 | 23 |

Send start is HTTPX's `send_request_headers.started`, after client connection establishment and before request-body transmission. Accepted-to-delta includes unfinished image preparation and connection work. Local body-write traces do not measure remote upload arrival.

All 92 operations reached schema completion. One direct warm `h11_U_en` response was `{"blocks":[]}`: valid JSON but no translated block or first display. It remains in completion/TTFT accounting and is absent only from metrics it cannot produce. Its absence is not evidence of complete text coverage.

Warm responses all reported prompt-cache reads. Cold cache reads occurred in 9/23 API and 7/23 Relay samples; the remaining cold samples reported writes. Cold/warm differences therefore combine transport and cache effects. Preparation ranged from 180.9 to 675.2 ms.

**First actual display was not observed for any sample.** The real native renderer used its offscreen backend with the actual ONNX mask model; native GPU composition is recorded separately and is not a display measurement. Exports waited 250 ms after all accepted blocks were composed. Inspection found readable ordinary regions but also crowding, source remnants, and overlaps, notably in `u07_photo` and dense signs. Neither schema completion nor an export proves full-image usability or headset legibility. Native shutdown acknowledged successfully with exit code 0.

## Parsing-policy investigation

No product parser change was made during this experiment. The 92 new responses contained **922 blocks**, with no malformed structure, canvas/content-boundary violations, degenerate boxes, empty source/translation, or exact duplicate blocks. Same-response comparison across clipping budgets of 0–16 pixels and retained-area thresholds of 80–99% changed none of them. Consequently, this corpus does **not** calibrate an optimal clipping threshold or demonstrate a natural-error recovery rate.

The earlier synthetic dense-photo experiment did produce two real failures. Both returned valid JSON with 48 blocks; the last bbox was invalid for the 1280×720 provider image:

- `[51,722,271,730]`: entirely below the image. Clamping produces zero height.
- `[51,717,271,726]`: clamping leaves only 3 of 9 pixels of height. Treating that as a normal region would hide severe clipping.

### Controlled replay, not additional model calls

A recorded `u02_photo` API response had 21 valid blocks, including a block touching the left image edge at index 15. Only that bbox was perturbed. A provisional policy allowed at most 2 provider pixels of clipping and required at least 95% retained area.

| Injected condition | Existing strict parser | Candidate outcome |
| --- | --- | --- |
| Left edge −1 or −2 px | 15 accepted blocks, then failure | Corrected to the original bbox; all 21 retained |
| Left edge −3 px | 15 accepted blocks, then failure | One excluded; 20 retained, including five later blocks |
| Entire box outside image | 15 accepted blocks, then failure | One excluded; 20 retained, including five later blocks |
| Truncated JSON | Incomplete response | Not repaired into a successful response |

The baseline, corrected −1 px response, and excluded-outside response were replayed through the actual photo owner and native mask/erase/typesetting renderer without provider calls. The corrected export was **pixel-identical** to baseline. Exclusion left the affected original text untranslated; image differences were confined to `(0,659)–(202,843)` in the 1920×1080 export. Other blocks remained rendered. All three native replays completed; shutdown acknowledged with exit code 0. This proves the bounded replay outcome, not a shipped streaming-quarantine implementation. The initial replay harness omitted a required diagnostic callback and failed before rendering; it was corrected before this proof.

### Recommended direction

1. Keep structural/type validation; do not invent missing JSON, swap axes, rescale guessed coordinate conventions, or automatically repeat a paid generation.
2. Validate against both the 1280×720 canvas and the actual source-content rectangle inside its letterbox. The existing source mapper already clips content intersections; canvas-only checks are insufficient.
3. Allow only bounded clipping with both an absolute pixel budget and retained-area guard. **2 px / 95% is a conservative candidate, not an empirically optimized threshold.** The two genuine synthetic failures remain excluded under it.
4. Quarantine irrecoverable geometry before native preparation/erase, continue processing later valid blocks, and retain original block identity and exclusion reasons. Never fabricate a replacement position.
5. Exclude empty source/translation rather than erasing without usable translated content. No empty-text or duplicate cases occurred here; broader fuzzy deduplication or semantic repairs are unsupported by this evidence.
6. Distinguish full completion, completion with correction, partial output with omissions, valid no-text output, and failed/truncated output. If every proposed block is rejected, do not relabel that as a successful no-text result. Keep diagnostics local; no new remote coordination is needed.

This policy is now applied in the worktree through the incremental parser, source mapper, lifecycle, and native preparation boundary. No product UI was added.

### Applied parsing policy

Structural JSON/type validation remains strict. Geometry is validated before native preparation, with a maximum 2-provider-pixel clip and at least 95% retained area against the source-content rectangle. Bad geometry and empty source/translation are excluded without renumbering later blocks. Exclusion tombstones cancel hidden preparations and reject late worker results; rejected text is not erased. Structural/truncation failures remain failures and do not trigger an automatic paid retry.

| Outcome | Recorded phase |
| --- | --- |
| Every block accepted unchanged | `completed` |
| Every block accepted, with bounded correction | `completed_corrected` |
| Valid response with accepted and excluded blocks | `partial` |
| Valid empty block list | `no_text` |
| Every proposed block rejected | `failed`, `photo_no_usable_blocks` |
| Malformed/truncated response | `failed`; accepted prefix retained, unfinished preparation discarded |

Photo IPC is now version **3**, requiring explicit `excluded_blocks` and the new completion phases. Old photo clients are rejected rather than silently accepting incomplete semantics. Chatbot protocol v6 is unchanged; rebuild the native photo executable with the application.

Applied-policy verification passed:

- Focused parser, image-mapping, lifecycle, and bridge tests; Ruff; Black.
- Native integration: 16 cases passed initially; the two failures were obsolete protocol-version assertions. After removing the incidental assertion, both affected cases passed. Two external/device-dependent cases remained ignored.
- The pending-erase quarantine regression passed after correcting its fixture's output dimensions. It proves that rejected preparation is discarded without changing original pixels.
- Release build and two actual-child process tests passed, including restart/reopen without another provider generation and explicit preparation-loss failure without job replay.
- All **92 recorded responses / 922 blocks** replayed through the applied parser and mapper unchanged, with **zero provider calls**.
- Four actual-owner/native ONNX renderings passed: baseline, −1 px clipping, outside-image exclusion, and empty translation. The last case waited for mask preparation before rejecting the block. Both exclusions retained later IDs 16–20 and left no excluded preparation. Native errors were empty; shutdown acknowledged with exit code 0.
- Corrected output was pixel-identical to baseline. Empty-translation and outside-image exclusions were pixel-identical to each other; differences from baseline were confined to `(0,659)–(202,843)` in the 1920×1080 export. Visual inspection confirmed original text remained in the excluded region.

Applied-policy native SHA-256: `f29f5f600301b7bf91a83f95c4ef03cd52c2aceb4fb8b9cfb53d6c8d132a28f8`. Private outputs: `C:/tmp/puri_capture_probe/runs/issue386-bb48cd2fcf/policy-applied/images/{baseline,clipped_1px,excluded_outside,empty_translation}.png`; adjacent `records.json` contains the outcomes. The throwaway replay program was removed. These are offscreen rendering checks, not headset-display measurements or new latency samples. Earlier latency/source fingerprints below and in the evidence JSON remain the historical measured candidate; `policy_application` records the later implementation separately.

## Measured resource policy

| `PHOTO_POLICY` field | Default | Evidence / rationale |
| --- | ---: | --- |
| `maxBodyBytes` | 4194304 | Supplied-photo requests: 98,633–475,850 bytes; image base64: 98,524–473,576 bytes. Separate image allowance, not a copied text-chat cap. |
| `maxOutputTokens` | 8192 | Supplied responses: 11–1328 tokens, 13–4772 JSON bytes; earlier dense synthetic probes reached 2556 tokens. Preserve full output headroom. |
| `concurrency` | 1 | Existing photo owner processes one accepted translation at a time; isolate each installation. |
| `allowance` | 50 | Provisional bounded admissions per period, not a measured daily-use frequency. The supplied run used 46 Relay admissions without changing this limit. |
| `periodMs` | 86400000 | Installation allowance persists across tokens/connections within the period. |
| `deadlineMs` | 240000 | Supplied schema completion: 1.64–11.84 s; earlier dense samples about 18–21 s. Cancellation and lease recovery remain bounded. |

Hard bounds remain 20 MiB body, 32768 output tokens, concurrency 8, 300 s deadline, and a minimum 60 s period. Failed/uncertain admitted work is not a fresh allowance on retry. Chatbot limits were not relaxed.

## Other comparisons and retained failures

- Personal ChatGPT: sign-in refresh succeeded in 1028.5 ms, but all eight attempted photos were quota-rejected with `photo_usage_limit`. No successful matched latency is claimed.
- Managed OpenRouter: a real Responses API compatibility probe using the existing personal inference credential produced six blocks with Luna/OpenAI and fallback disabled. This is **not** restricted-key enforcement evidence. The available key reports `is_management_key=false`; management enumeration returned 401. Live model-denial, spending, revocation, and a matched restricted-key latency comparison remain unverified. Local/Worker contract tests do not replace them.
- Initial synthetic campaign: eight Relay failures came from Cloudflare rejecting `redirect: 'error'` before upstream fetch. The handler now uses `manual` and rejects redirected responses without replay. Of seven recorded direct samples, five completed and two had invalid bboxes; the last planned direct sample hit an application deadline before its sample record was written. HTTP body/consumer ownership was fixed and tested. These attempts are retained, not merged into the corrected candidate's success counts.
- Corrected synthetic campaign: 16 samples; ordinary API and Relay 4/4 each, dense Relay 4/4, dense API 2/4 with the bbox failures above. Dense rendering was not accepted as fully usable.
- Active session experiment: a temporary, unshipped WebSocket prototype reused one app→Relay and one Relay→OpenAI connection for two independent photos, with authentication/admission on each photo. Direct Responses WebSocket was the matched transport control. Warm accepted-to-delta was 1230.5 ms direct versus 1795.7 ms Relay; cold was 2945.3 versus 5186.3 ms. Only one cold/warm pair per route was taken; Relay's cold case included lazy upstream setup. One Worker upgrade for two photos establishes feasibility, not a speed improvement. The first harness attempt failed before sending images because its request-sent event omitted a timestamp. HTTP remains the shipped baseline. Provider reference: [Responses WebSocket mode](https://developers.openai.com/api/docs/guides/websocket-mode).

## Verification, cleanup, and handoff

Focused checks passed: provider/lifecycle/credential behavior tests, two real local-HTTP pooling/cancellation tests, composition tests (one skip), Relay photo tests (8), deployment tests (22), Broker photo tests (6), TypeScript checking, and the existing first-enrollment/managed-recovery compatibility checks. The three credential recovery/revocation tests passed after treating pending revocation as pending rather than success. Ruff passed and Black formatted the changed Python files. No broad simulator or full-suite claim is made.

Historical measured native release binary SHA-256: `1ad07a3354da000e91c4a93cfcac3c5fcaa1e55e5dfa4a3f3ce00ce8f`. Mask model SHA-256: `bfe95d72db1eea72101074293c8d27b77eb87663fa48a23e8258e59249510bcf`.

Measured supplied-photo Worker versions: Broker `a1a7d11b-5857-44ef-8d1c-4c72a77bd080`; Relay `ace9e430-d17b-4e5c-aa93-52c6a3ce5f87`. Earlier HTTP Relay: `f84090c7-ec3d-4e07-b69e-bc6874607801`; temporary session prototype: `4bfa7832-0351-413a-b75d-303c39bee146`. Both isolated deployments' Workers and databases were deleted successfully. Production resources and the supplied originals/model were not changed.

Private raw responses, native exports, and replay exports remain under `C:/tmp/puri_capture_probe/runs/issue386-bb48cd2fcf`; they are not committed. Required next evidence is real first display, resolved personal quota, a management-authorized restricted OpenRouter test, and usable-output assessment. The measured Relay HTTP path is available for #389, but the missing prerequisites must not be represented as completed integration or production readiness.

Protected documentation was intentionally not edited. Drift: `docs/architecture.md` says Luna photo access uses ChatGPT rather than API keys and lacks the new hosted/secondary ownership and bounded clipping/quarantine outcomes; `native/overlay/README.md` lacks explicit preparation cancellation and the distinct photo completion outcomes, and its personal-Luna-only description predates hosted access; `agent-relay/README.md` lacks the photo endpoint/policy/migration; `broker/README.md` lacks the photo-specific disabled-before-ACK and period-reservation lifecycle; `PRIVACY.md` describes audio/text external processing and direct managed OpenRouter translation without the new hosted-photo image path through Relay.
