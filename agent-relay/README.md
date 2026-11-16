# Agent Relay

Puri's model-request service is a TypeScript Cloudflare Worker with D1 usage storage and optional R2 support storage. Relay owns model requests, prompts, and protocol contracts. Python owns conversations, app context, and tool execution. [Broker](../broker/README.md#puri-installation-tokens) supplies installation authentication.

See [application ownership](../docs/architecture.md#agent-conversation) and the [Privacy Policy](../PRIVACY.md#6-puri-assistant-and-voluntary-support-reports).

## Protocol and Authentication

Every supported endpoint requires a Broker-issued Puri token, including local development. Model requests and support reports have separate admission switches.

Prompts and [protocol contracts](src/contracts.ts) ship with the Worker. Rebuild and restart or deploy after changing them. Deploy new protocols to Relay **before the app release**; retain older protocols while supported apps use them.

## Deployment

Run the manual [deployment workflow](../.github/workflows/deploy-agent-relay.yml) from `dev` with the reviewed commit and production confirmation. Configuration lives in the workflow and [preparation script](operations/prepare-deployment.mjs).

- Provision an initialized D1 dedicated to Relay before deployment. Leave `apply_migrations` off for code-only updates; pending migrations block deployment.
- Set `AGENT_SERVICE_ENABLED` and `SUPPORT_REPORTS_ENABLED` explicitly. Disabling model admission leaves already dispatched work running.
- Deployment preserves existing Worker secrets and non-owned runtime variables. Keep `OPENAI_API_KEY` server-only and coordinate shared `PURI_TOKEN_SECRET` rotation with Broker.
- Committed migrations have no automatic rollback. Authorize database recovery separately.

The deployment smoke checks authentication and protocol rejection without model calls or report uploads. Live-cost probes require separate authorization.

## Support Reports

Reports upload only after an explicit user click and may contain sensitive conversation and log content. Normal Relay storage and logs exclude message bodies.

Use a developer-only R2 bucket with the [30-day lifecycle](support-lifecycle.json). From `agent-relay`, with operations-only Cloudflare credentials:

```sh
node operations/support-storage.mjs verify puripuly-heart-support-reports
node operations/support-storage.mjs configure puripuly-heart-support-reports --enabled-config ./wrangler.support-enabled.json
```

`verify` is read-only. `configure` requires authorization and writes a local config; it does not deploy. Deployment checks bucket privacy and retention.

## Local Development

Use a Linux-native checkout and dependencies, including WSL-internal storage. From the repository root:

```sh
pnpm install --frozen-lockfile
pnpm --filter @puripuly-heart/agent-relay typecheck
pnpm --filter @puripuly-heart/agent-relay test
pnpm --filter @puripuly-heart/agent-relay dev --var AGENT_SERVICE_ENABLED:true
```

Apply [migrations](migrations/) to isolated local D1. Supply `OPENAI_API_KEY` and the Broker-matching `PURI_TOKEN_SECRET` through the Worker environment, not CLI arguments or the app.

Apps default to the public service. Set an explicit override before starting a source app:

```powershell
$env:PURIPULY_AGENT_RELAY_URL = "http://127.0.0.1:8788/v1/respond"
uv run python -m puripuly_heart.main
```

Point `PURIPULY_AGENT_BROKER_URL` at an isolated Broker with the same token secret. Isolate app data, secrets, and runtime resources. Loopback HTTP is source-only; remove overrides to use the public service.

## Scenario Simulator

The [simulator](../src/puripuly_heart/release_evidence/agent_chat_simulator/) uses synthetic app state, not a live GUI. Keep output private and outside the repository.

```sh
uv run --no-sync python -m puripuly_heart.release_evidence.agent_chat_simulator --help
```

Pass `--subscription-only` to every live `probe`, `run`, and `calibration-rejudge`. Use ChatGPT sign-in and stop if unavailable; do not fall back to API-key billing.

## Source Map

| Owner | Source |
| --- | --- |
| Requests and streaming | [Handler](src/index.ts), [stream](src/stream.ts) |
| Prompts and tools | [Prompt sources](prompts/agent/), [tool definitions](src/app-tools.ts) |
| Authentication and usage | [Token verification](src/auth.ts), [usage](src/usage.ts) |
| Support uploads | [Receiver](src/support.ts), [storage operations](operations/support-storage.mjs) |
