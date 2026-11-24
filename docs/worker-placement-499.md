# Broker and Agent Relay placement — #499

**Broker: unresolved; leave off. Relay: unresolved; leave off.** Both isolated Workers ended with `INSUFFICIENT_INVOCATIONS`; none of the 28 trial client requests was confirmed remotely placed. Do not interpret the differences below as an improvement or regression from placement. **Photo CPU: no production TypeScript or Wasm change.** The two version-matched completed, app-prepared photo requests used 7 and 19 ms of Cloudflare CPU. The local profile explains where to look, but does not isolate an avoidable cost that justifies changing the handler. Free-plan headroom remains unproven. No user-visible behavior changed. Ordinary speech translation, personal photo routes, established P2P delivery and Broker cron are outside this result.

## What is running

Source under test: [`83b27db2477834bf36b47ea3d89f23999ca35e84`](https://github.com/kapitalismho/PuriPuly-heart/tree/83b27db2477834bf36b47ea3d89f23999ca35e84), equal to remote `dev` at setup. The [Broker](../services/broker/wrangler.jsonc) and [Relay](../services/agent-relay/wrangler.jsonc) sources both omit `placement`. The table describes **deployed** Workers, not deployments of this checkout. Control-plane readback: 2026-11-24, approximately 08:11 UTC.

| Production Worker | Active version (100% of traffic) | Source provenance | Effective settings |
| --- | --- | --- | --- |
| Broker | `325071a2-a935-41f6-9021-432fd8a9c12f` | No Git SHA tag. Deployment at 06:31:57 UTC on Nov 17 falls within the [successful deploy step](https://github.com/kapitalismho/PuriPuly-heart/actions/runs/44031848078) for `508a13809474a6a14948227c18558cfd090cc35c`; this is provenance by correlation, not source identity. | `placement: {}`, no analysis status; compatibility date `2026-04-08`. Fetch and scheduled handlers. Two D1 bindings. |
| Relay | `adda747a-ec95-43f3-802a-29c71049fa74` | Version tagged `461075609fc535c6f9493ef86cbf14209c0ce2dc`; [deployment run](https://github.com/kapitalismho/PuriPuly-heart/actions/runs/44306358896). Relay stream/handler and prompt changes exist since that tag. | `placement: {}`, no analysis status; same compatibility date, `enable_request_signal`. Agent, photo and support flags enabled; usage D1 and support R2 bound. |

The account listed two Worker services, both with only a production environment. The D1 API reported `running_in_region: APAC` for both bound databases. This is control-plane region metadata, not the serving colo or round-trip time of an individual query; it was not inferred from the source hosting hint. Workers account default and both scripts reported usage model `standard`; the subscriptions endpoint returned HTTP 403, so the billing plan, allowance remaining and invoice are not verified. No plan change is proposed.

### Production usage, not photo or token timings

Cloudflare `workersInvocationsAdaptive`, last complete UTC day, **2026-11-23**. Counts and CPU are platform aggregates, not requests sent by this experiment. CPU quantiles are reported in microseconds by the API, converted to milliseconds below. Percentiles across these mixed routes cannot answer the placement or photo-headroom questions.

| Worker / event / invocation status | Requests | Subrequests | Platform errors | CPU P50 (ms) | Platform CPU P99 (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Broker / fetch / success | 126 | 0 | 0 | 7.162 | 26.867 |
| Broker / cron / success — excluded | 1,432 | 723 | 0 | 7.260 | 22.302 |
| Relay / fetch / success | 78 | 37 | 0 | 5.148 | 106.237 |
| Relay / fetch / clientDisconnected | 8 | 8 | 0 | 5.249 | — (sparse) |

Runtime `success` does not mean a token was issued, a reply was complete, or a photo was displayed. `clientDisconnected` does not identify deliberate user cancellation. This dataset exposes no request-path dimension. A dry aggregate-only query to Workers Observability returned HTTP 403; no route-level CPU samples were obtained from logs. Neither the mixed Relay P99 nor [#389’s instrumented photo samples](https://github.com/kapitalismho/PuriPuly-heart/issues/389#issuecomment-6764911074) establishes current production CPU for `/v1/photo` or a CPU-limit failure.

**Repeat the readback** with an already authorized read-only session. Under `GET /client/v4/accounts/{account}` read `workers/services`, `workers/services/{name}` (default-environment script placement fields), `workers/scripts/{name}/settings` (placement), `workers/scripts/{name}/deployments` (active traffic shares), `workers/scripts/{name}/versions/{version}` (source tag/runtime), and `d1/database/{bound-id}` (`running_in_region`). For plan identity use `subscriptions`; do not substitute `workers/account-settings.default_usage_model`. Select only those fields; never publish raw settings/bindings, account IDs, tokens, request headers or bodies.

POST the following read-only query to `/client/v4/graphql` per script, with `account` set privately, `start=2026-11-23T00:00:00Z` and `end=2026-11-24T00:00:00Z`:

```graphql
query($account: string!, $script: string!, $start: string!, $end: string!) {
  viewer { accounts(filter: {accountTag: $account}) {
    workersInvocationsAdaptive(limit: 100, filter: {
      scriptName: $script, datetime_geq: $start, datetime_lt: $end
    }) {
      dimensions { eventType status }
      sum { requests subrequests errors }
      quantiles { cpuTimeP50 cpuTimeP99 }
    }
  } }
}
```


## Isolated Cloudflare measurements

The owner authorized up to three temporary Workers, two temporary D1 databases, 24 Broker and 16 Relay client requests. No production writes, billing changes or external model calls. Public test endpoints required an additional random trial secret. The workload used no customer data or production credentials. Cloudflare observations and published request receipts contain metadata, not raw headers or bodies. Private GUI verification also used synthetic prompt/reply text and owned-window pixels to check presentation.

### Broker: authentication works; placement benefit unresolved

One client location (`request.cf.colo=HKG`); no second approved egress. Same Worker name, code, synthetic signing secrets, fresh matched installations, dedicated D1 (31 ordinary migrations), connection-reuse policy and D1 timing wrapper in both arms; no cron or outbound fetch. Only `placement.mode` changed. The off version was `bbb11345-ac12-40d7-a978-60e95d4406a2`; the smart version was `68c0907f-a023-4746-aa57-ec40d5366101`. Each served 100% of traffic in its arm. Cloud settings read back `{}` and `{"mode":"smart"}` respectively. The client sent a signed body and read the complete response before stopping its latency timer; JWT verification and the expected 401 were checked separately.

| `POST /v1/puri/token` in request order | Off: completion / D1 await (ms) | Smart configured: completion / D1 await (ms) |
| --- | ---: | ---: |
| First enrollment | 653 / 401 | 598 / 366 |
| Renewals, three fresh nonces | 428 / 359; 464 / 352; 428 / 343 | 406 / 345; 412 / 334; 390 / 334 |
| Exact replay, expected 401 | 467 / 406 | 458 / 399 |
| Fresh proof after rejection | 479 / 349 | 419 / 339 |

12/12 expected outcomes, 10 valid tokens, two rejected replays, zero timeouts or unexpected failures. Final isolated D1 readback: two bound installations, ten nonces. D1 await encompasses binding round trips, not only SQL time. Both arms had five D1 awaits/nine statements per successful request. The metadata tail returned 11/12 invocation events. The `cf-placement` request header was absent in **all 12**: zero confirmed remotely placed samples. No P95 or attribution of the smaller second-arm values to placement.

### Relay: client receipt, display and CPU are different observations

Broker configuration was fixed throughout. The separate Relay used real Puri authentication, the eight usage migrations, admission, stream transformation and settlement. The only upstream was a nonpublic service-bound Worker (`26e49a10-1612-4bcc-ba23-367eaf147f88`) with fixed synthetic frames; it checked the outgoing request against the expected model, prompt, history/image and schema before responding. Neither Worker could call a real model. Off version `db337bec-4a8b-4d31-83ae-3b0c5800b1cb`; smart version `266f1baf-73c2-4fd0-ba06-c510440f6a6b`. Equivalent code in both arms, including the gate, service-binding adapter and timing headers. Handler SHA-256 `445f5ebd6767ce82cb2838d2be8c2dffca2dbc478f7a6bcc2baf4f74d6bed484`.

All **16/16** attempted Relay HTTP requests and all 16 CPU traces were retained in the [redacted measurements](worker-placement-499.json). Join each trace by Worker version, route and receipt timestamp; individual requests were spaced apart. One client origin used NRT and HKG ingress; the intended smart arm did **not** produce a remotely placed group. Do not pool those colos, compare mismatched versions or estimate a tail from one pair.

| Controlled request, off / smart-configured | First meaningful client receipt (ms) | First parsed translated block (ms) | Terminal frame (ms) | Cloudflare CPU (ms) |
| --- | ---: | ---: | ---: | ---: |
| Work chat reply following classification | 102 / 171 | — | 161 / 279 | 5 / 7 |
| Work chat aborted after first useful text | 331 / 173 | — | neither read | 7 / 7 |
| Completed prepared photo (one per valid arm) | 167 / 538 | 167 / 539 | 225 / 658 | 7 / 19 |
| Deliberately incomplete prepared photo | 258 / 327 | 258 / 328 | 314 / 432 | 9 / 14 |

Timings above start at client request dispatch; the 304,041-byte photo body already contains the prepared JPEG. Two distinct classifiers completed (228/371 ms); they are not first chat text. For protocol-7 work chat, the fixture used an empty history and “Why are captions missing?”, `gpt-6-luna` classification and `gpt-5.6-terra` response. Both cancelled clients stopped before terminal, both platform traces reported `canceled`, and the next classify/respond pair completed in both arms. This does not prove an external provider stopped. Across the driver and desktop profiles D1 recorded four decisions, six responses, two missing-usage records and zero remaining active slots.

**Photo denominator: six attempts.** Two full completions, two deliberately incomplete with a first block, one 401 before admission, one HTTP 200 from the wrong active version rejected by the client. The latter two used 1 and 20 ms CPU respectively and are **not** successful matched photos. On the 401, the locally issued JWT timestamp was in the next second relative to Worker receipt; preissuing subsequent trial tokens by 5 s avoided that fixture skew without changing auth. The failed slot was not retried. During the smart round the final photo returned the earlier off version even though the control plane had reported 100% smart; its expected/actual versions and `canceled` trace are retained. Five admitted photos, five settled rows, zero missing photo usage. No headset was observed: first parsed block is not actual photo display.

**Actual chat display.** The running Windows Agent view submitted one additional synthetic social chat turn per arm (two HTTP requests each). The app token owner was replaced for this check. A gated loopback adapter supplied its own signed trial JWT, replaced the GUI-built request bodies with a frozen protocol-7 “check the cedar fixture” profile, and streamed the real test Relay response through the app. This exercises Relay verification and the app’s receive/display path, not GUI-to-Broker token acquisition. In the owned-window captures the Cedar temperature sentence appeared first; the settled row added “Amber indicator is steady.” Times from posting Enter:

| Desktop observation (one turn/arm) | Off | Smart configured |
| --- | ---: | ---: |
| First nonblank client text (ms) | 2,313 | 2,317 |
| First useful visible text, conservative capture bracket (ms) | 2,299–2,359 | 2,285–2,341 |
| Terminal frame / settled pixels observed by (ms) | 3,090 / 3,125 | 3,089 / 3,125 |

The brackets overlap. These times include the app, adapter pacing and a deliberately held 700 ms synthetic tail; they do not estimate provider or placement gains. An earlier private-desktop rehearsal completed two turns and cancelled one, but returned only black frames. It is evidence of client/owner behavior only; interactive capture of the owned window supplies the actual display evidence. Visible cancellation itself was not verified.

**CPU decision.** Even the valid completed prepared-photo samples straddle Free’s published 10 ms allowance. Neither invocation failed a CPU limit. CPU includes test gate and binding-adapter overhead; platform wait/wall time and CPU spent in the stub are not counted as Relay computation. Combined with the profile below, this warrants neither a claim of Free headroom nor a rewrite. Investigate a narrowly isolated TS candidate only if representative traces identify avoidable work and a matched end-to-end result preserves validation, atomic admission, image, stream and settlement. There is no demonstrated residual algorithmic hotspot to justify measuring a JS/Wasm boundary now.

**Rollback and teardown.** The initial isolated Relay upload succeeded but its public test endpoint was not enabled (Wrangler 10007). Read back the uploaded version, bindings and disabled endpoint; enabled only that gated test origin with previews still off. No production access path was changed. After recording each service’s analysis status, `off` was redeployed and both test Workers read back `placement: {}`. Deleted both tails, all three test Workers and both test D1s; the D1 reads returned 404 and the service inventory returned to the original two production versions. No production deployment, plan change, or real provider spending.

## Controlled local measurements

**Broker.** Run `node services/broker/scripts/issue-499-broker-probe.mjs` and repeat with `--d1-timing` for attribution. Each run used the actual token handler, ephemeral signing keys, HTTP loopback and migrated local D1. Of 80 signed POSTs per run, 76 issued independently verified tokens; invalid proof and replay returned 401, mismatched key 409, and the ninth request for a dedicated installation 429. A subsequent fresh proof succeeded. No unexpected outcomes. The unwrapped service completed 60/60 renewals in 8.141/9.839 ms (local P50/P95) and four first enrollments in 7.901–8.888 ms. The instrumented renewal distribution was 7.482/8.616 ms; five sequential awaited D1 calls covered nine statements, with await P50/P95 of 5/6 ms. These are two instrumentations of a local baseline, **not** an off/smart comparison or a D1 network estimate.

For the following command, [build and stage the Windows native executable and OpenVR sidecars](../native/overlay/README.md#build-and-verification) and create an empty private output directory first.

**Managed photo profile.** Workerd `1.20261023.1`, local D1, actual `/v1/photo`, fixed synthetic upstream via Miniflare. Generate the fixture with `PYTHONPATH=src python services/agent-relay/operations/issue-499-prepare-photo.py <native-overlay-executable> <new-private-directory>` in the prepared app environment. Run `node services/agent-relay/operations/issue-499-photo-profile.mjs <same-directory> 3 24`. The checkout's Python loader and managed adapter consumed output from the native preparation probe (binary SHA-256 `b92a6863868591cc5228e20cf5d4c71c4a5a1ccdf99c5627d13ae163af76da1d`). A 3840×2160 source with 12 synthetic labels became a 227,930-byte 1280×720 JPEG, SHA-256 `45fc569e6755aa54e38472e60db0eec172cc6b739465c4ee0794eaef528e6867`. The larger transport fixture is not a prepared image. Protocol 2, Korean, `gpt-6-luna`, reasoning `none`, fixed schema/instructions, output budget 8,192. No real provider.

| Profile cohort (72 attempts each) | JSON request bytes | Sampled stacks: start to upstream entry | First-block stream | Later stream and settlement |
| --- | ---: | ---: | ---: | ---: |
| Prepared JPEG, 12-block controlled stream | 304,041 | 277 | 18 | 27 |
| Synthetic near-boundary body, same stream | 4,000,133 | 1,288 | 30 | 52 |
| Same JPEG, 72-block dense stream | 304,041 | 248 | 16 | 165 |

These are V8 **samples, not CPU milliseconds**. Each cell counts attributed non-idle, non-GC, non-`(program)` samples in the indicated phase; the nine saved profiles retain the other samples. Requested interval 101 µs; median observed intervals 535–1,049 µs. The fixed upstream withheld the tail until the client parsed one complete block. All 216 profiled attempts reached that point; each cohort had 60 complete and 12 deliberately incomplete streams. Twelve warmups bring the durable totals to 228 admissions and 228 settlements, no missing usage. Replay returned 409 without upstream work, malformed base64 returned 400, and an over-limit request returned 413 headers before a local reset prevented verification of its error body.

The ordinary pre-output samples implicate the combined validation predicate in `photo.ts`, not specifically its regex. At 4 MB the bounded body reader and upstream serialization become more prominent. With the same prepared image and much more output, incremental SSE parsing and emission add work **after** first block. The three first-block loopback P50/P95s for completed requests were 5.85/7.94, 33.40/52.27 and 5.77/16.93 ms respectively. They exclude native preparation, use a controlled provider, and are not display latency. No isolated avoidable computation has yet demonstrated a comparable end-to-end TypeScript saving; no JS/Wasm prototype or change to validation, image quality, accounting or stream order on this evidence.

**Repeatability checks.** The current tools preserve metadata for interrupted Broker attempts (a read response, including expected rejections, counts as completed) and refuse to mix photo runs: use a fresh fixture directory with no `profile/` subdirectory. The sampler writes that subdirectory exclusively. A post-header reset on the oversized case requires an observed 413; a pre-header reset or wrong status fails the run. A small repaired-tool run settled 30/30 admissions including warmups (15 complete and three deliberately incomplete profiled streams). The original profile observations above predate these evidence-only repairs. Focused unchanged contracts exercised: 43 Vitest cases in `services/agent-relay/tests/{stream,photo}.spec.ts` and 30 Python cases for chat streaming, managed photo transport and partial/truncated photo handling. These checks do not establish headset display or production CPU.

### Follow-up: what changed in code (local, placement off)

Baseline `16fca59f408ec9dff00c580280fe2f2d32d30cbb`. These comparisons change code, **not** placement. They used isolated workerd, migrated local D1, synthetic identities and fixed inputs. No production traffic, external model, desktop display or headset was measured.

**Broker:** On the successful IP path, read the network-identity migration state and the candidate keyed hooks in one batch **after proof verification**. Use the hook result only if that snapshot confirms keyed-only mode. Otherwise evaluate the original policy; a rejected batch goes through one authoritative sequential lookup, not an automatic grant. No-IP requests retain the old path. Legacy and dual modes do one extra speculative SQL read and do not save a binding call. The flood gate and atomic admission writes stay in order.

Run `node services/broker/scripts/issue-499-broker-probe.mjs --d1-timing` on baseline and candidate. Each 80-request arm matched all outcomes, including 76 independently verified tokens and fresh proof after rejection, with identical stored counts. On 60 renewals per arm the successful keyed path went from **five to four sequential D1 binding calls**, still nine SQL statements. Local dispatch-to-complete-body P50/P95 went from 7.813/8.611 to 7.042/8.190 ms. This is a measured round-trip reduction, not a production latency prediction.

**Relay chat:** Pre-serialize the classifier's fixed developer message, questions and cache options once per protocol. The user input and authenticated subject still use ordinary JSON encoding. In the protocol-7 fixture, 8,694 of 8,933 outbound bytes were fixed. The exact-wire checks in `services/agent-relay/tests/contracts.spec.ts` cover both supported protocols.

For the plain comparison, start two loopback Workers on the same locked workerd version with fresh local D1s (all eight Relay migrations). Use the first classify/respond pair in `services/agent-relay/tests/app-requests/protocol-7.json`, fresh turn IDs, signed synthetic subjects, and the same work/high Decisions answer. The local provider emits the first answer sentence immediately, then the second sentence and terminal after an 8 ms timer (400 ms for cancellation). Match the outbound body and the client's ordered answer to their expected bytes. Exclude four warmup turns per Worker; run 48 paired ordinary turns alternating arm order, plus two fresh successful recovery turns per arm. Time each POST from dispatch. For each turn, add classification time through full body receipt to response time through first nonblank sentence or terminal frame. Quantiles below are nearest ranks of those 50 per-turn sums, not elapsed time between GUI events; neither request's preparation time is included.

For the separate amplified comparison, count actual D1 calls in a forwarding wrapper. After admission, call the arm's classifier serializer and `TextEncoder.encodeInto` on the result 4,000 times per request using a reusable 64 KiB buffer. Consume a byte from each encode into a checksum and verify it at the stub so both arms process the same wire text. Alternate arms for two warmup and six measured classifier requests each. Divide client dispatch → provider-stub entry by 4,000 and take the nearest-rank median: **3.705 → 1.327 µs per iteration**. These estimates include amortized transport, auth, parsing and admission. The comparison supports less repeated serialization work, not absolute serializer-only time or per-request CPU. Neither the number of D1 calls nor the number of provider calls changed.

| Client measurement, 50 completed chat turns per arm | Before, P50/P95 (ms) | After, P50/P95 (ms) |
| --- | ---: | ---: |
| Classify completion + response to first meaningful reply | 6.930 / 8.139 | 6.777 / 8.356 |
| Classify completion + response to terminal SSE frame | 30.524 / 31.700 | 29.667 / 31.182 |

Each plain arm sent 115 HTTP requests: 111 HTTP 200 (including the deliberately cancelled reply), one expected 401, one expected 429 and two expected 502s. In the plain arm, abort the HTTP request after receiving the first sentence: neither stub stopped before its held tail. In the instrumented arm, cancel the actual Relay response reader after its first chunk, wait for the slot to clear, verify the missing-usage count and submit a fresh classify/respond turn. This distinguishes Worker-side cancellation from propagation across HTTP. The client timings establish no end-to-end or visible speedup.

**Relay managed photo:** Rejected a candidate that composed the outgoing JSON as a `Blob`. To reproduce the discarded variant, keep admission, validation and fetch options unchanged. Serialize the normal payload with `image_url` set to just `data:image/jpeg;base64,`. Split that JSON immediately after the unique `"image_url":"data:image/jpeg;base64,` field prefix. Serialize `body.image_base64` separately, remove only its outer JSON quotes, and use `new Blob([prefix, encodedImage, suffix])` in place of the original fetch body. Run the same managed-photo profiler above in baseline → candidate → candidate → baseline order, each in a fresh output directory. The table gives client dispatch to first parsed block for completed requests, not display time.

| Input | Baseline first-block P50, two runs (ms) | Blob first-block P50, two runs (ms) |
| --- | ---: | ---: |
| Prepared JPEG, 12 blocks | 5.89–6.35 | 6.17–6.21 |
| Synthetic near-limit body, 12 blocks | 31.90–34.96 | 34.99–35.11 |
| Same JPEG, 72 blocks | 5.86–6.15 | 6.26–6.35 |

Across four runs: **720 complete, 144 deliberately incomplete, 48 warmups, and 12 rejection controls**. All 912 admitted requests settled with no missing usage. Each run also returned replay 409, invalid base64 400 and oversized 413; all four oversized bodies reset after headers, so only those statuses were verified. On the near-limit input the two baseline runs had 1,270 and 1,321 attributed pre-upstream V8 samples versus 1,355 and 1,344 with `Blob` at similar sampling intervals. These are samples, not CPU milliseconds. Moving work out of one serialization call did not improve total pre-upstream processing or first-block time. Keep the current photo handler; this test justifies neither a Wasm change nor a production CPU/headroom claim.

## Reviewable placement control and reversal

No change to either canonical Wrangler source or a production workflow. With Node 24.14.0, pnpm 10.34.9 and locked Wrangler 4.153.0, the following local procedure exercised both repository renderers and both placement modes. Use an empty private directory outside the checkout (`$T`), no cloud credentials or inherited `.dev.vars`/provider environment, and a temporary HOME, Wrangler config and log directory. Disable Wrangler metrics and dotenv loading. From the repository root after `pnpm install --frozen-lockfile`:

```sh
B=22222222-2222-4222-8222-222222222222
A=11111111-1111-4111-8111-111111111111
mkdir -p "$T/broker" "$T/agent-relay"
node services/broker/scripts/render-production-wrangler-config.mjs \
  --database-id "$B" --agent-usage-database-id "$A" --network-identity-hmac-key-version 1 \
  --out "$T/broker/wrangler.json"
env CLOUDFLARE_ACCOUNT_ID=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa CLOUDFLARE_API_TOKEN=unused \
  BROKER_D1_DATABASE_ID_PRODUCTION="$B" AGENT_USAGE_D1_DATABASE_ID_PRODUCTION="$A" \
  AGENT_RELAY_CANONICAL_WORKERS_DEV_URL=https://puripuly-heart-agent-relay.synthetic-account.workers.dev \
  PURI_TOKEN_SECRET_PRODUCTION=synthetic-never-use-for-live-signing AGENT_SERVICE_ENABLED=true \
  PHOTO_SERVICE_ENABLED=true SUPPORT_REPORTS_ENABLED=false \
  node services/agent-relay/operations/prepare-deployment.mjs prepare "$T/agent-relay"
```

Both rendered configs omitted placement. Relay’s preparer forced `observability.enabled=false`; with support off it removed R2 without a storage call. The Broker renderer preserved the minute cron in the generated base. These are actual local renders with dummy IDs, not production bindings.

For each service, derive **one private dry-run config** from its rendered base: name `issue-499-dry-<service>`, dummy account `aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa`, `workers_dev=false`, `preview_urls=false`, no triggers, observability disabled. Resolve `main` and each D1 `migrations_dir` against the original service directory (Broker defaults to `migrations`); rename each database to `issue-499-dry-<binding in lowercase, underscores replaced by hyphens>`, retaining the synthetic IDs. Write two copies differing only in the top-level field `"placement": {"mode":"off"}` or `"placement": {"mode":"smart"}`. For each service/mode, from its workspace run `pnpm exec wrangler deploy --dry-run --config "$T/<service>/<mode>.json" --outdir "$T/<service>/<mode>"`. Never submit these dummy configs. Remove only the owned temporary directory.

Both arms printed `--dry-run: exiting now.` Per Worker the emitted code/modules matched byte-for-byte across modes (excluding source maps and Wrangler’s README). Entry `index.js` SHA-256: `b2f3728cb14ff8818ef9a511bd79bd6dd11ab90c6d94d3b4df1324ed993d2299` (Broker, one module); `560255dbb6be6bf58b6aa191a8482609fd2e0a3642036ec727f9d208bcd259bb` (Relay, seven modules including text). This proves bundling, not placement.

For an authorized isolated deployment, use dedicated resources, synthetic secrets and an access-controlled origin, with verified isolation from production credentials, D1, routes, cron, R2 and webhooks. Retain the same instrumentation in both arms. Do not enable body logging or rely on a source-only observability toggle. Apply only the placement delta to the **same** Worker, not simultaneous code, SQL, prompt, model or provider changes. Reverse with `placement.mode = "off"`, regenerate, compare and read back the active version/setting. Any later production adoption would need the delta in the chosen Worker’s Wrangler source, regeneration through its existing pipeline, separate deployment authorization and readback. No regional hint or Worker split is proposed.

## Repeating the controlled Relay cohort

Use the gated setup above with the same private stub, fixtures and migration set in both arms, no external fallback. Per arm, run six slots in order, with 1,650 ms pacing: **full photo → cancel chat after first nonblank answer → classify a fresh turn → complete that turn → partial photo → full photo**. Generate fresh request/turn IDs, sharing the ID within the classify/respond pair. Use the work/high classify result and the [protocol-7 chat requests](../services/agent-relay/tests/app-requests/protocol-7.json). Use a 30 ms normal stream tail and a 1,800 ms cancellation tail. For photos, use the prepared JPEG above and the `frames` function in the [profile tool](../services/agent-relay/operations/issue-499-photo-profile.mjs) with 12 blocks and 48-character fragments. After the first complete block, send either the remaining blocks and `response.completed` or a truncated second block and `response.incomplete`. Do not retry failed slots. Record the Worker version actually serving each request.

Use one **separate** GUI classify/respond pair per arm as described under actual chat display. The adapter enforces a 1,650 ms interval after classification, pins the social/low fixture and forwards SSE. Send the Cedar sentence, hold the tail for 700 ms, then send the authoritative text with the Amber sentence. Capture the owned answer area from Enter through the first positive and settled frames. Join platform traces on `workerReceivedUtc` (Worker clock), not client dispatch. This reproduces the controlled workload, not the real provider or headset.

## Conditions for a future live comparison (not a production rollout)

1. **Broker first, Relay fixed.** Use an authorized isolated Worker with dedicated D1 and two approved client locations, the same signed `POST /v1/puri/token` profile and rate-limit settings in both arms. Start with one new enrollment, returning-installation renewals and one deliberately replayed nonce per arm/location; record enrollment, renewal, rejection and any unexpected failures separately. Cap the pilot at 24 total requests (six per arm/location); do not manufacture traffic to force placement. Sign before starting the HTTP timer. Stop it after reading the complete body; validate the returned token without recording it. Keep connection reuse, concurrency, schema/state, split by ingress location, and timing boundary constant. Record elapsed D1 awaits separately where instrumented identically; `meta.timings.sql_duration_ms` excludes network time.
2. **Check the placement, not just the setting.** Inspect service analysis status before interpreting samples; use the Cloudflare-supplied `cf-placement` **request** header at the Worker or placement analytics to distinguish remote and local traffic. Do not send a client-supplied header to simulate placement. Record missing/unknown routes as such. `SUCCESS` is analysis, not proof that every request moved; an absent status, `INSUFFICIENT_INVOCATIONS`, or too few comparable local/remote requests is a gap. `UNSUPPORTED_APPLICATION` means the platform reverted placement. The platform’s default 1% control can be too sparse. Finish a bounded round and read back status; elapsed time alone is not a stopping criterion for success.
3. **Relay separately, Broker fixed.** Use the [running-app paths](verification.md#chatbot), frozen protocol-7 synthetic [chat inputs](../services/agent-relay/tests/app-requests/protocol-7.json), conversation state and actual selected model; `/v1/classify` and `/v1/respond` are separate provider calls. Do not pool different response models. Start with one completed turn and one cancel-after-first-sentence followed by one fresh completed turn per arm. Record submit, classify/response request, first nonblank text delta at the client, first useful sentence actually visible in the mounted Agent view, and terminal body/authoritative row. Keep rejected, fallback, late-output and cancellation outcomes. Verify the old turn cannot publish after cancellation and the next can finish; client disconnect alone does not prove the provider stopped.
4. **Managed photos.** Freeze the same synthetic source and app-prepared JPEG (hash, dimensions, byte count), full request bytes, protocol 2, target language, `gpt-6-luna`, instructions, schema, output budget and stream shape where controllable. Use fresh request IDs. Through the [photo path](verification.md#image-translation), record pull and local preparation separately, then send → first text → first displayable accepted translated block → visible headset block for the same attempt → terminal outcome. Record CPU, provider wait, total time and partial/failed/cancelled/missing-display counts separately. Neither native `first_display` nor an offscreen render proves headset presentation. Start with one ordinary prepared image per arm; a separate near-limit synthetic body stresses transport, not image quality. Do not retry incomplete translations to fill the successful group.
5. **Stop or expand by decision value.** No paid calls are authorized by this procedure. A controlled upstream can exercise transport and isolate CPU, not predict real-provider placement. Obtain an explicit provider budget and reachable display surfaces before the real Relay comparison. Count classified/completed/cancelled model calls against that budget, not just user turns. Keep all attempt denominators, raw values for small matched cohorts and missing traces; report P50/P95 only when counts and comparable distributions support them. Recommend `smart` for a Worker only if the affected client outcome improves without a meaningful slow-request, error, output or cancellation regression. Revert/clean up only test-owned resources.
