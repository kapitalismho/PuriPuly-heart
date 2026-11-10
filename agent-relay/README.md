# Agent Relay

Puri's model-request service: a TypeScript Cloudflare Worker with D1 usage accounting and optional R2 support storage. Python owns conversations, app context, and tool execution; Relay owns model requests, instructions, and tool definitions. [Broker](../broker/README.md) supplies installation authentication.

This guide covers service boundaries and development/deployment entry points. Request schemas, policy values, and evaluation details belong in the [owning implementation](#source-map). See the [architecture](../docs/architecture.md#agent-conversation) for application ownership and [Privacy Policy](../PRIVACY.md#6-puri-assistant-and-voluntary-support-reports) for data handling.

## Scope and Boundaries

- Every supported endpoint requires a Broker-issued Puri token, including local development. Model and support admission are enabled separately.
- Classification selects scope, effort, and app-context modules from conversation summaries, not app records. Work responses receive allowlisted app context; brief routes have no tools, and off-topic requests exclude prior work context.
- Relay owns the model, endpoint, trusted instructions, cache configuration, and tool schemas. Clients cannot supply trusted instructions or expand tool permissions.
- Responses stream provisional text. Only a completed response admits tool calls and reasoning continuation; Python fences interrupted text and late results.
- App tools separate reads, immediate actions, and approval proposals, and reuse existing application owners. Batched calls run in order; approval, required user action, or pending work stops later effects. Every settings change requires approval of its exact values. Credentials use secure app controls, staged only by explicit actions; account actions and support uploads remain user actions.
- Cancellation stops future agent actions, not accepted application operations or saved changes. Active model history is bounded separately from the in-memory app-run report archive.

## Authored Content

[Scope](prompts/agent/scope.md) defines classification, [system](prompts/agent/system.md) defines behavior, [guide](prompts/agent/guide.md) supplies product knowledge, and [app context](prompts/agent/app-context.md) describes the current protocol's context format. Work requests include the bundled [cost](prompts/agent/references/keys-and-costs.md) and [benchmark](prompts/agent/references/benchmark.md) references in a stable server-owned prefix; classification has its own. Classification and brief replies exclude those references.

All user-facing language, register, length and formatting guidance lives in the system's `Identity and voice` section. Its style examples are grouped in English, Korean, Chinese and Japanese order, with Chinese kept as one group of the existing Simplified Chinese examples. Work, social and out-of-scope requests share that complete section; the user's requested response language and writing system, or their current message, choose the reply language and script rather than examples, UI settings or earlier conversation.

The relay bundle contains the authoritative English sources; Python does not fetch or duplicate them. Prompt changes require rebuilding and restarting or deploying the Worker. Relay serves every app protocol in its [contract registry](src/contracts.ts); a contract holds tool definitions, context modules, the app-context format, and setup-control guidance. Shared prompt and policy changes apply to all supported protocols. A contract change adds a protocol, deployed to Relay before the app release; older protocols remain until retired.

## Authentication, Usage and Stop Control

- The app obtains an in-memory bearer through [Broker installation proof](../broker/README.md#puri-installation-tokens). Relay validates it on every request and uses the derived installation subject for accounting and OpenAI's safety identifier, not the raw installation ID.
- D1 atomically owns daily prompt-turn admission, per-turn request budgets, and concurrent-request slots. Classification, tool continuations, and eligible app notices share one turn identity. Failure, cancellation, midnight, and token renewal do not renew its budget.
- Usage records, including per-protocol request counts, are separate from optional statistics and contain no conversation bodies. Missing usage stays unknown; estimates are not provider billing. There is no dollar-based or global monthly spending cutoff.
- Authentication recovery is bounded. An uncertain model request or interrupted stream is not automatically replayed. Missing required configuration and failed admission return static errors rather than another model call.
- Deploy `AGENT_SERVICE_ENABLED=false` to stop new model admissions. Already dispatched work is not cancelled. `SUPPORT_REPORTS_ENABLED` controls reports independently, including when model allowance is exhausted.

Normal relay storage and logs do not retain message bodies. OpenAI `store: false` does not establish Zero Data Retention; provider retention and the explicit support-report exception are covered in the Privacy Policy.

## Click-only Support Reports

- A report offer creates no bundle or upload. The user's click captures the app-run Puri archive, including pre-reset conversations, metadata, and the current runtime log. Reasoning and rotated logs are excluded; sensitive prose may remain.
- The app owns upload state independently of conversation cancellation/reset. Explicit retries reuse the click-time bytes and ticket. Uncertain delivery is not automatically replayed.
- R2 storage is create-only. The first stored object fixes the ticket; a same-installation retry retrieves its receipt without replacing it or charging another report slot. D1 and R2 have no shared transaction, so an uncertain write can consume quota.
- Storage success is authoritative. Discord notification failure does not turn it into failure; notifications contain only ticket, app version, compressed size, and diagnostic codes. There is no public report-download endpoint.

### Private Storage and Retention Configuration

Reports require a dedicated, developer-only R2 bucket with the [30-day lifecycle](support-lifecycle.json). From `agent-relay`, using operations-only Cloudflare credentials:

```sh
node operations/support-storage.mjs verify puripuly-heart-support-reports
node operations/support-storage.mjs configure puripuly-heart-support-reports --enabled-config ./wrangler.support-enabled.json
```

`verify` reads remote policy. `configure` changes it and requires separate authorization; it preserves unrelated rules and verifies privacy/retention before emitting an enabled local config. Neither deploys the Worker. Confirm account access separately; local emulation does not verify cloud access or timed deletion.

## Manual Deployment

The manual [deployment workflow](../.github/workflows/deploy-agent-relay.yml) updates code against an **already initialized Relay database**. Dispatch from `dev` with the exact reviewed commit and production confirmation. Leave `apply_migrations` off for ordinary code updates. The workflow checks the database, deploys once, and runs an authentication/protocol rejection smoke.

The GitHub `production` environment needs:

- Secrets: `CLOUDFLARE_ACCOUNT_ID`, `CLOUDFLARE_API_TOKEN`, `AGENT_USAGE_D1_DATABASE_ID_PRODUCTION`, `BROKER_D1_DATABASE_ID_PRODUCTION`, and the Broker-matching `PURI_TOKEN_SECRET_PRODUCTION` for verification only.
- Variables: `AGENT_RELAY_CANONICAL_WORKERS_DEV_URL`, `AGENT_SERVICE_ENABLED`, and `SUPPORT_REPORTS_ENABLED`. Both flags must explicitly be `true` or `false`; missing values stop deployment.
- Before the first simplified deployment, set both variables to the intended current production state. They replace the old dispatch inputs; there is no implicit disabled default.

Existing Worker secrets are preserved, not uploaded again. Keep `OPENAI_API_KEY` server-only and `PURI_TOKEN_SECRET` identical in Broker and Relay. Deployment does not consume GitHub's OpenAI or Discord secret inputs. `--keep-vars` preserves non-owned runtime variables; the two explicit service flags come from the production environment. The checked-in [Wrangler config](wrangler.jsonc) remains a disabled local template with a placeholder database ID.

When reports are enabled, preparation verifies private R2 access and the exact 30-day retention policy without changing either. Disabled reports omit the R2 binding. The rejection smoke makes no model call, D1 write, or report upload; it does not establish provider capability or end-to-end app readiness.

### Ordinary Migration Maintenance

Select `apply_migrations` only for an intended ordinary schema upgrade. Before applying anything, the workflow requires the initial usage migration and its actual base schema. After optional maintenance, it requires every checked-in migration and the current usage, report, prompt-turn, and request-slot columns. A code-only deployment with pending migrations stops rather than installing incompatible code. Read or migration failure prevents code deployment.

**No workflow backup or recovery checkpoint is created or required.** There is no SQL export, encryption certificate, backup/build-receipt artifact, Time Travel gate, or automatic restore. Successfully applied migrations remain committed if later deployment or verification fails. Cloudflare-native history is not an independent recovery copy or an exercised restore guarantee.

### Operator-Only Maintenance

Provision and initialize a dedicated usage D1, separate from Broker, before ordinary deployment. Configure private R2 retention through the separately authorized [storage operation](#private-storage-and-retention-configuration). Register or rotate Worker secrets deliberately through Cloudflare settings or authorized Wrangler secret operations; changing the shared token secret requires coordination with Broker.

For authorized initialization or one-time SQL maintenance, supply the production environment above and choose an external temporary `DEPLOY_DIR`. From `agent-relay`, `node operations/prepare-deployment.mjs prepare "$DEPLOY_DIR"` writes only a deployment config; it does not initialize or deploy anything. After explicit initialization approval, run `pnpm exec wrangler d1 migrations apply AGENT_USAGE_DB --remote --config "$DEPLOY_DIR/wrangler.json"` and remove the temporary directory. Do not force migration journal entries.

Use a reviewed compatible code revert for rollback. Code rollback does not reverse D1 migrations, accounting, or separately changed secrets. Database restoration and full live model/report acceptance remain separately authorized operations.

## Local Development

Use a Linux-native checkout and dependencies, including WSL-internal storage. From the repository root:

```sh
pnpm install --frozen-lockfile
pnpm --filter @puripuly-heart/agent-relay typecheck
pnpm --filter @puripuly-heart/agent-relay test
pnpm --filter @puripuly-heart/agent-relay dev --var AGENT_SERVICE_ENABLED:true
```

Supply `OPENAI_API_KEY` and the Broker-matching `PURI_TOKEN_SECRET` through the Worker environment, never command-line arguments or the app process. Apply all [relay migrations](migrations/) to isolated local D1 before requests. Keep inherited-secret loading enabled in Wrangler; use `WRANGLER_SEND_METRICS=false` for isolated probes.

Source and packaged apps default to the public service. Local development requires an explicit override before starting the app:

```powershell
$env:PURIPULY_AGENT_RELAY_URL = "http://127.0.0.1:8788/v1/respond"
uv run python -m puripuly_heart.main
```

Set `PURIPULY_AGENT_BROKER_URL` for an isolated Broker and match its token secret. Use isolated app data, credentials, and runtime resources; a separate settings file alone is insufficient. Remove overrides to return to the public service. Loopback HTTP is source-only; redirects are not followed. Desktop clients do not use `HTTP_PROXY`/`HTTPS_PROXY` environment variables.

Tool definitions come from the app's current `settings.choices` catalog. Export it from an isolated instance and regenerate with:

```sh
uv run python agent-relay/generate-tools.py path/to/settings-choices.json
```

Keep the export outside the repository; it may contain local device choices. Runtime validation still uses the current catalog and revision. [PR CI](../.github/workflows/pr-ci.yml) owns the automated checks and dry-run bundle build.

## Bounded Scenario Simulator

The [simulator](../src/puripuly_heart/release_evidence/agent_chat_simulator/) replaces the application-control owner beneath production chat adapters with synthetic state. Hidden causes stay outside assistant/user inputs. Synthetic outcomes do not establish actual GUI, device, provider-signup, or remote-display behavior.

Use the checkout's Python environment and write reports outside the repository:

```powershell
uv run --no-sync python -m puripuly_heart.release_evidence.agent_chat_simulator --help
uv run --no-sync python -m puripuly_heart.release_evidence.agent_chat_simulator smoke --output C:\private\puri-smoke.json
uv run --no-sync python -m puripuly_heart.release_evidence.agent_chat_simulator render-html --report C:\private\puri-run.json --output C:\private\puri-run.html
```

`smoke` uses controlled model responses. Live evaluation requires an isolated Relay/Broker and a context manifest from the [exact emitted Worker](operations/export-context.mjs). Freeze source, context, cases, and judge; preserve failed/blocked attempts and historical reports. Paired comparisons apply the same declared criteria to both versions. Recorded latency describes accepted outcomes only and does not replace the usefulness judgment. Benchmark authentication is separate from the app's account. HTML export reads saved results without model calls or rejudgment.

The [#332 closure](https://github.com/kapitalismho/PuriPuly-heart/issues/332#issuecomment-6702068939) and [#333 closure](https://github.com/kapitalismho/PuriPuly-heart/issues/333#issuecomment-6702069184) preserve historical evaluation/UI decisions and discard the remaining work, rather than certify it as passed or defer it. [#365](https://github.com/kapitalismho/PuriPuly-heart/issues/365#issuecomment-6701890465) records calibration completed during #332, with no separate follow-up or reported agreement counts. [#367](https://github.com/kapitalismho/PuriPuly-heart/issues/367#issuecomment-6703215301) records separate bounded Windows/hosted checks, not a full final-GUI acceptance pass. The [#383/#384 record](https://github.com/kapitalismho/PuriPuly-heart/issues/384#issuecomment-6740918645) preserves the protocol-6 tool consolidation and recommendation comparison with its failed and blocked attempts; their owner-requested closures do not certify acceptance.

## Source Map

| Question | Read |
| --- | --- |
| Requests, upstream errors, streaming | [Handler](src/index.ts), [request contract](src/requests.ts), [stream](src/stream.ts) |
| Trusted instructions and tool definitions | [Loader](src/instructions.ts), [contracts](src/contracts.ts), [scope](src/scope.ts), [app tools](src/app-tools.ts), [tool generator](generate-tools.py), [module generator](generate-context.py) |
| Authentication, admission, usage | [Token verification](src/auth.ts), [accounting](src/usage.ts), [migrations](migrations/) |
| Support bundles, storage, retention | [App bundle](../src/puripuly_heart/app/services/agent_chat/report.py), [receiver](src/support.ts), [storage operations](operations/support-storage.mjs) |
| Deployment and schema readiness | [Preparation](operations/prepare-deployment.mjs), [smoke](operations/smoke-deployment.mjs), [deployment checks](tests/deployment.spec.ts) |
| App context, permissions, conversation lifetime | [Chat owner](../src/puripuly_heart/app/services/agent_chat/owner.py), [context](../src/puripuly_heart/app/services/agent_chat/context.py), [modules](../src/puripuly_heart/app/services/agent_chat/modules.py), [tools](../src/puripuly_heart/app/services/agent_chat/tools.py), [execution](../src/puripuly_heart/app/services/agent_chat/execution.py) |
| App transport and presentation | [Transport](../src/puripuly_heart/app/adapters/agent_relay_transport.py), [presentation](../src/puripuly_heart/app/services/agent_chat/presentation.py) |
| Behavior checks | [Relay tests](tests/), [recorded app requests](tests/app-requests/), [app-owner tests](../tests/app/test_agent_chat.py) |
