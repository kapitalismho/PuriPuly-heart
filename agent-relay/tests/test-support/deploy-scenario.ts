import { execFileSync, spawn } from 'node:child_process';
import { once } from 'node:events';
import { mkdtempSync, mkdirSync, readFileSync, readdirSync, writeFileSync } from 'node:fs';
import { createServer } from 'node:http';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { DatabaseSync } from 'node:sqlite';
import { fileURLToPath } from 'node:url';

import { parse } from 'yaml';

import relay from '../../src/index';

const repository = fileURLToPath(new URL('../../../', import.meta.url));
const relayRoot = fileURLToPath(new URL('../../', import.meta.url));
const support = fileURLToPath(new URL('./', import.meta.url));
const canonicalOrigin = 'https://puripuly-heart-agent-relay.synthetic-account.workers.dev';
const tokenSecret = 'synthetic-deployed-token-at-least-32-bytes';
const shellPath = (path: string) => path.replaceAll('\\', '/');
const bash = process.platform === 'win32'
  ? join(dirname(dirname(execFileSync('where.exe', ['git'], { encoding: 'utf8' }).trim().split(/\r?\n/u)[0]!)), 'bin', 'bash.exe')
  : 'bash';

export interface DeployScenario {
  root: string;
  runnerTemp: string;
  databasePath: string;
  workerPath: string;
  env: NodeJS.ProcessEnv;
}

type WorkerState = {
  deployments: number;
  secrets: Record<string, string>;
  vars: Record<string, string>;
  reportsBound: boolean;
};

export function createDeployScenario(options: {
  pendingMigration?: boolean;
  env?: Record<string, string | undefined>;
  mode?: string;
} = {}): DeployScenario {
  const root = mkdtempSync(join(tmpdir(), 'relay-minimal-deploy-'));
  const runnerTemp = join(root, 'runner');
  mkdirSync(runnerTemp);
  const databasePath = join(root, 'synthetic.sqlite');
  const database = new DatabaseSync(databasePath);
  try {
    database.exec('CREATE TABLE d1_migrations (id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE, applied_at TEXT DEFAULT CURRENT_TIMESTAMP)');
    for (const name of readdirSync(join(relayRoot, 'migrations')).filter((name) => name.endsWith('.sql')).sort()) {
      if (options.pendingMigration && name >= '0004_') continue;
      database.exec(readFileSync(join(relayRoot, 'migrations', name), 'utf8'));
      database.prepare('INSERT INTO d1_migrations (name) VALUES (?)').run(name);
    }
    database.exec(`
      INSERT INTO usage_daily (installation_hash, utc_date, decisions_calls, responses_calls, cost_usd)
      VALUES ('synthetic-installation', '2026-11-05', 4, 7, 0.125);
      INSERT INTO support_daily VALUES ('synthetic-installation', '2026-11-05', 2);
      INSERT INTO prompt_turns (installation_hash, turn_id, utc_date)
      VALUES ('synthetic-installation', 'synthetic-turn', '2026-11-05');
      CREATE TABLE deploy_sentinel (value TEXT);
      INSERT INTO deploy_sentinel VALUES ('preserve-me');
    `);
  } finally { database.close(); }
  const workerPath = join(root, 'worker.json');
  writeFileSync(workerPath, JSON.stringify({
    deployments: 0,
    secrets: { PURI_TOKEN_SECRET: tokenSecret, OPENAI_API_KEY: 'deployed-only-provider-key', DISCORD_OPERATIONS_WEBHOOK_URL: 'deployed-only-webhook' },
    vars: { AGENT_SERVICE_ENABLED: 'true', PHOTO_SERVICE_ENABLED: 'true', SUPPORT_REPORTS_ENABLED: 'true', NON_OWNED_RUNTIME_SETTING: 'preserve-me' },
    reportsBound: true,
  } satisfies WorkerState));
  writeFileSync(join(root, 'github-env'), '');
  const commit = execFileSync('git', ['rev-parse', 'HEAD'], { cwd: repository, encoding: 'utf8' }).trim();
  return {
    root, runnerTemp, databasePath, workerPath,
    env: {
      PATH: process.env.PATH,
      SystemRoot: process.env.SystemRoot,
      HOME: shellPath(root),
      TEMP: shellPath(root), TMP: shellPath(root),
      BASH_ENV: shellPath(join(support, 'deploy-env.sh')),
      SCENARIO_NODE: shellPath(process.execPath),
      SCENARIO_PRELOAD: new URL('./deploy-preload.mjs', import.meta.url).href,
      SCENARIO_PORT: shellPath(join(support, 'deploy-port.mjs')),
      SCENARIO_ROOT: shellPath(root), SCENARIO_MODE: options.mode ?? '',
      RUNNER_TEMP: shellPath(runnerTemp), GITHUB_ENV: shellPath(join(root, 'github-env')),
      GITHUB_EVENT_NAME: 'workflow_dispatch', GITHUB_REF: 'refs/heads/dev', GITHUB_SHA: commit,
      REVIEWED_COMMIT: commit, CONFIRM_PRODUCTION_DEPLOY: 'deploy puripuly-heart-agent-relay from dev',
      GITHUB_RUN_ID: '123', GITHUB_RUN_ATTEMPT: '2',
      AGENT_SERVICE_ENABLED: 'true', PHOTO_SERVICE_ENABLED: 'true', SUPPORT_REPORTS_ENABLED: 'true',
      AGENT_RELAY_CANONICAL_WORKERS_DEV_URL: canonicalOrigin,
      CLOUDFLARE_ACCOUNT_ID: 'a'.repeat(32), CLOUDFLARE_API_TOKEN: 'synthetic-cloudflare-token',
      AGENT_USAGE_D1_DATABASE_ID_PRODUCTION: '11111111-1111-4111-8111-111111111111',
      BROKER_D1_DATABASE_ID_PRODUCTION: '22222222-2222-4222-8222-222222222222',
      PURI_TOKEN_SECRET_PRODUCTION: tokenSecret,
      WRANGLER_SEND_METRICS: 'false', CLOUDFLARE_LOAD_DEV_VARS_FROM_DOT_ENV: 'false',
      ...options.env,
    },
  };
}

export function readWorker(scenario: DeployScenario): WorkerState {
  return JSON.parse(readFileSync(scenario.workerPath, 'utf8'));
}

export function readEvents(scenario: DeployScenario): Array<{ event: string; pending?: string[] }> {
  try { return readFileSync(join(scenario.root, 'events.jsonl'), 'utf8').trim().split('\n').filter(Boolean).map((line) => JSON.parse(line)); }
  catch (error) {
    if (error instanceof Error && 'code' in error && error.code === 'ENOENT') return [];
    throw error;
  }
}

export async function runDeployWorkflow(scenario: DeployScenario) {
  const workflow = readFileSync(new URL('../../../.github/workflows/deploy-agent-relay.yml', import.meta.url), 'utf8').replaceAll('\r\n', '\n');
  const query = workflow.match(/^      RELAY_SCHEMA_QUERY: (.+)$/mu);
  if (!query) throw new Error('Missing workflow schema query.');
  let queryValue = query[1].replace(/^(['"])(.*)\1$/u, '$2');
  if (['|', '>', '>-'].includes(query[1])) {
    const lines: string[] = [];
    for (const line of workflow.slice(query.index! + query[0].length).split('\n').slice(1)) {
      if (line && !line.startsWith('        ')) break;
      lines.push(line.slice(8));
    }
    queryValue = lines.join(' ').trim();
  }
  const requests: string[] = [];
  const storageRequests: string[] = [];
  const forbiddenAccess: string[] = [];
  const deny = (name: string) => () => { forbiddenAccess.push(name); throw new Error(`Smoke must not access ${name}.`); };
  const server = createServer(async (incoming, outgoing) => {
    try {
      const path = incoming.url!;
      if (path.startsWith('/cloudflare/')) {
        storageRequests.push(`${incoming.method} ${path}`);
        if (incoming.method !== 'GET') throw new Error('Storage verification must be read-only.');
        const result = path === '/cloudflare/domains/managed' ? { enabled: false }
          : path === '/cloudflare/domains/custom' ? { domains: [] }
            : path === '/cloudflare/lifecycle' ? JSON.parse(readFileSync(join(relayRoot, 'support-lifecycle.json'), 'utf8')) : undefined;
        if (!result) throw new Error('Unexpected storage endpoint.');
        outgoing.writeHead(200, { 'content-type': 'application/json' }).end(JSON.stringify({ success: true, result }));
        return;
      }
      requests.push(`${incoming.method} ${path}`);
      const chunks: Buffer[] = [];
      for await (const chunk of incoming) chunks.push(Buffer.from(chunk));
      const headers = new Headers();
      for (const [name, value] of Object.entries(incoming.headers)) if (typeof value === 'string') headers.set(name, value);
      const request = new Request(`http://127.0.0.1${path}`, { method: incoming.method, headers,
        ...(['GET', 'HEAD'].includes(incoming.method ?? '') ? {} : { body: Buffer.concat(chunks).toString('utf8') }) });
      const worker = readWorker(scenario);
      const actual = await relay.fetch(request, {
        ...worker.vars, ...worker.secrets,
        AGENT_USAGE_DB: { prepare: deny('D1'), batch: deny('D1') } as unknown as D1Database,
        ...(worker.reportsBound ? { SUPPORT_REPORTS: { head: deny('R2'), put: deny('R2') } as unknown as R2Bucket } : {}),
      });
      outgoing.writeHead(worker.deployments ? actual.status : 503, { ...Object.fromEntries(actual.headers),
        ...(scenario.env.SCENARIO_MODE === 'smoke-failure' ? { 'cache-control': 'public' } : {}) });
      outgoing.end(Buffer.from(await actual.arrayBuffer()));
    } catch (error) { outgoing.writeHead(500).end(String(error)); }
  });
  const listening = once(server, 'listening');
  server.listen(0, '127.0.0.1');
  await listening;
  const address = server.address();
  if (!address || typeof address === 'string') throw new Error('Missing loopback server address.');
  const env: NodeJS.ProcessEnv = {
    APPLY_MIGRATIONS: String(parse(workflow).on.workflow_dispatch.inputs.apply_migrations.default ?? false),
    ...scenario.env,
    LOCAL_HTTP_URL: `http://127.0.0.1:${address.port}`,
    RELAY_SCHEMA_QUERY: queryValue,
  };
  const executed: string[] = [];
  let status = 0;
  let failedStep: string | undefined;
  let stdout = '';
  let stderr = '';
  try {
    for (const block of workflow.split(/(?=^      - name: )/mu).slice(1)) {
      const name = block.match(/^      - name: (.+)/u)![1];
      if (block.includes('\n        uses:') || name === 'Install workspace dependencies') continue;
      const condition = block.match(/^        if: (.+)$/mu)?.[1];
      if (condition && !['inputs.apply_migrations', 'always()'].includes(condition)) throw new Error(`Unsupported workflow condition: ${condition}`);
      if ((status !== 0 && condition !== 'always()') || (condition === 'inputs.apply_migrations' && env.APPLY_MIGRATIONS !== 'true')) continue;
      const match = block.match(/^        run: (.*)\n?/mu);
      if (!match) throw new Error(`Missing workflow script: ${name}`);
      let script = match[1];
      if (script === '|' || script === '>-') {
        const body: string[] = [];
        for (const line of block.slice(match.index! + match[0].length).split('\n')) {
          if (line && !line.startsWith('          ')) break;
          body.push(line.slice(10));
        }
        script = body.join(script === '>-' ? ' ' : '\n');
      }
      const scriptPath = join(scenario.root, 'step.sh');
      writeFileSync(scriptPath, script);
      executed.push(name);
      const child = spawn(bash, ['--noprofile', '--norc', '-e', '-o', 'pipefail', shellPath(scriptPath)], {
        cwd: block.includes('working-directory: agent-relay') ? relayRoot : repository, env, stdio: ['ignore', 'pipe', 'pipe'],
      });
      child.stdout.on('data', (data) => { stdout += String(data); });
      child.stderr.on('data', (data) => { stderr += String(data); });
      const [code] = await once(child, 'close');
      if (code !== 0 && status === 0) { status = code ?? 1; failedStep = name; }
      for (const line of readFileSync(join(scenario.root, 'github-env'), 'utf8').split(/\r?\n/u)) {
        const separator = line.indexOf('=');
        if (separator > 0) env[line.slice(0, separator)] = line.slice(separator + 1);
      }
    }
    return { status, failedStep, stdout, stderr, executed, requests, storageRequests, forbiddenAccess };
  } finally {
    const closed = Promise.withResolvers<void>();
    server.close((error) => error ? closed.reject(error) : closed.resolve());
    await closed.promise;
  }
}
