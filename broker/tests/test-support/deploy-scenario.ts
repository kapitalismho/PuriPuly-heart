import { spawn } from 'node:child_process';
import { once } from 'node:events';
import { copyFileSync, mkdirSync, mkdtempSync, readFileSync, readdirSync, writeFileSync } from 'node:fs';
import { createServer } from 'node:http';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { DatabaseSync } from 'node:sqlite';
import { fileURLToPath } from 'node:url';
import { promisify } from 'node:util';

import { parse } from 'yaml';

import { app } from '../../src/app';
import { applyBrokerMigrations, BROKER_MIGRATION_FILENAMES } from './migrations';

const brokerRoot = fileURLToPath(new URL('../../', import.meta.url));
const workflow = readFileSync(new URL('../../../.github/workflows/deploy-broker-direct.yml', import.meta.url), 'utf8');

export interface DeployScenario {
  root: string;
  broker: string;
  databasePath: string;
  workerPath: string;
  env: NodeJS.ProcessEnv;
}

export function createDeployScenario(options: {
  pendingMigration?: boolean;
  env?: Record<string, string | undefined>;
  mode?: string;
} = {}): DeployScenario {
  const root = mkdtempSync(join(tmpdir(), 'broker-minimal-deploy-'));
  const broker = join(root, 'broker');
  for (const dir of [broker, join(broker, 'scripts'), join(broker, 'migrations')]) mkdirSync(dir, { recursive: true });
  mkdirSync(join(broker, 'src'));
  copyFileSync(join(brokerRoot, 'src/trial-policy.ts'), join(broker, 'src/trial-policy.ts'));
  copyFileSync(join(brokerRoot, 'scripts/render-production-wrangler-config.mjs'), join(broker, 'scripts/render-production-wrangler-config.mjs'));
  copyFileSync(join(brokerRoot, 'wrangler.jsonc'), join(broker, 'wrangler.jsonc'));
  for (const name of readdirSync(join(brokerRoot, 'migrations'))) {
    copyFileSync(join(brokerRoot, 'migrations', name), join(broker, 'migrations', name));
  }
  const databasePath = join(root, 'synthetic.sqlite');
  const db = new DatabaseSync(databasePath);
  try {
    applyBrokerMigrations(db, { through: '0024_allow_unattributed_request_events.sql' });
    db.exec(`
      INSERT INTO managed_operations
        (operation_id, issue_source, subject_ref, state, attempt_count, current_attempt_index, resume_token_hash, auth_expires_at, client_action)
      VALUES
        ('ph-mop-v1_synthetic', 'qq', 'synthetic-subject', 'CREATE_UNKNOWN', 1, 1, 'ph-mop-resume-v1_${'a'.repeat(64)}', '2099-01-01', 'wait');
      INSERT INTO managed_operation_attempts (operation_id, attempt_index, provider_key_name, outcome)
      VALUES ('ph-mop-v1_synthetic', 1, 'synthetic-provider-key-name', 'unknown');
    `);
    for (const name of BROKER_MIGRATION_FILENAMES) {
      if (name <= '0024_allow_unattributed_request_events.sql' || name.startsWith('0025_')
        || name.startsWith('0027_') || (options.pendingMigration && name.startsWith('0026_'))) continue;
      db.exec(readFileSync(join(brokerRoot, 'migrations', name), 'utf8'));
    }
    db.exec('CREATE TABLE d1_migrations (id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE, applied_at TEXT DEFAULT CURRENT_TIMESTAMP)');
    for (const name of BROKER_MIGRATION_FILENAMES) {
      if (name.startsWith('0025_') || name.startsWith('0027_') || (options.pendingMigration && name.startsWith('0026_'))) continue;
      db.prepare('INSERT INTO d1_migrations (name) VALUES (?)').run(name);
    }
    db.prepare("UPDATE broker_config SET value = ? WHERE key = 'fingerprint_salt'").run(JSON.stringify({ current: { salt: 'synthetic-initialized-salt', version: 1 }, previous: null, rotated_at: null }));
    db.exec("INSERT INTO installations (installation_id, device_public_key, app_version) VALUES ('synthetic-installation', 'synthetic-public-key', 'synthetic'); CREATE TABLE deploy_sentinel (value TEXT); INSERT INTO deploy_sentinel VALUES ('preserve-me');");
  } finally {
    db.close();
  }
  const workerPath = join(root, 'worker.json');
  writeFileSync(workerPath, JSON.stringify({
    deployments: 0,
    secrets: { NETWORK_IDENTITY_HMAC_SECRET: 'deployed-only-current', NETWORK_IDENTITY_HMAC_SECRET_PREVIOUS: 'deployed-only-previous', OPENROUTER_MANAGEMENT_API_KEY: 'deployed-only-provider' },
    vars: { NETWORK_IDENTITY_HMAC_KEY_VERSION: '7', NETWORK_IDENTITY_HMAC_KEY_VERSION_PREVIOUS: '6', NON_OWNED_RUNTIME_SETTING: 'preserve-me' },
    guardrail: { allowed_models: ['preserve-me'] },
    issuedKeys: [],
  }));
  writeFileSync(join(root, 'ports.mjs'), cloudflarePort);
  writeFileSync(join(root, 'bash-env.sh'), `node() { '${process.execPath}' "$@"; }
pnpm() { node "$SCENARIO_ROOT/ports.mjs" "$@"; }
curl() {
  local args=()
  for arg in "$@"; do
    if [[ "$arg" == https://puripuly-heart-broker.synthetic.workers.dev* ]]; then
      args+=("$LOCAL_HTTP_URL\${arg#https://puripuly-heart-broker.synthetic.workers.dev}")
    elif [[ "$arg" == http://* || "$arg" == https://* ]]; then
      return 97
    else
      args+=("$arg")
    fi
  done
  command curl "\${args[@]}"
}
sleep() { :; }
`);
  return {
    root,
    broker,
    databasePath,
    workerPath,
    env: {
      PATH: process.env.PATH,
      HOME: root,
      BASH_ENV: join(root, 'bash-env.sh'),
      SCENARIO_ROOT: root,
      MODE: options.mode ?? '',
      GITHUB_REF: 'refs/heads/dev',
      CONFIRM_PRODUCTION_DEPLOY: 'deploy puripuly-heart-broker from dev',
      CLOUDFLARE_API_TOKEN: 'synthetic-cloudflare-token',
      CLOUDFLARE_ACCOUNT_ID: 'synthetic-cloudflare-account',
      BROKER_D1_DATABASE_ID_PRODUCTION: 'synthetic-database',
      BROKER_CANONICAL_WORKERS_DEV_URL: 'https://puripuly-heart-broker.synthetic.workers.dev',
      NETWORK_IDENTITY_HMAC_KEY_VERSION_PRODUCTION: '7',
      NETWORK_IDENTITY_HMAC_KEY_VERSION_PREVIOUS_PRODUCTION: '',
      ...options.env,
    } as NodeJS.ProcessEnv,
  };
}

export type PublicResponse = { status?: number; body?: unknown };

export async function runDeployWorkflow(scenario: DeployScenario, responses: Partial<Record<string, PublicResponse>> = {}) {
  const requests: string[] = [];
  const server = createServer(async (request, response) => {
    const path = request.url!;
    requests.push(`${request.method} ${path}`);
    if (!['/healthz', '/v1/foundation'].includes(path) || request.method !== 'GET') {
      response.writeHead(500).end('Unexpected request');
      return;
    }
    const effective = JSON.parse(readFileSync(scenario.workerPath, 'utf8'));
    const actual = await app.request(`http://synthetic.local${path}`);
    const override = responses[path];
    response.writeHead(override?.status ?? (effective.deployments ? actual.status : 503), { 'Content-Type': 'application/json' });
    response.end(override && 'body' in override ? (typeof override.body === 'string' ? override.body : JSON.stringify(override.body)) : await actual.text());
  });
  const listening = once(server, 'listening');
  server.listen(0, '127.0.0.1');
  await listening;
  const address = server.address() as { port: number };
  const env: NodeJS.ProcessEnv = {
    APPLY_MIGRATIONS: String(parse(workflow).on.workflow_dispatch.inputs.apply_migrations.default ?? false),
    ...scenario.env,
    LOCAL_HTTP_URL: `http://127.0.0.1:${address.port}`,
  };
  const executed: string[] = [];
  let stdout = '';
  let stderr = '';
  try {
    for (const block of workflow.split(/(?=^      - name: )/mu).slice(1)) {
      const name = block.match(/^      - name: (.+)/u)![1];
      if (block.includes('\n        uses:') || name === 'Install workspace dependencies') continue;
      const condition = block.match(/^        if: (.+)$/mu)?.[1];
      if (condition && condition !== 'inputs.apply_migrations') throw new Error(`Unsupported workflow condition: ${condition}`);
      if (condition && env.APPLY_MIGRATIONS !== 'true') continue;
      const match = block.match(/^        run: (.*)\n?/mu)!;
      let script = match[1];
      if (script === '|' || script === '>-') {
        const body = block.slice(match.index! + match[0].length).split('\n').map((line) => line.startsWith('          ') ? line.slice(10) : line);
        script = body.join(script === '>-' ? ' ' : '\n');
      }
      const scriptPath = join(scenario.root, 'step.sh');
      writeFileSync(scriptPath, script);
      executed.push(name);
      const child = spawn('bash', ['--noprofile', '--norc', '-e', '-o', 'pipefail', scriptPath], {
        cwd: block.includes('working-directory: broker') ? scenario.broker : scenario.root,
        env,
        stdio: ['ignore', 'pipe', 'pipe'],
      });
      child.stdout.on('data', (data) => { stdout += String(data); });
      child.stderr.on('data', (data) => { stderr += String(data); });
      const [status] = await once(child, 'close');
      if (status !== 0) return { status, stdout, stderr, executed, requests };
    }
    return { status: 0, stdout, stderr, executed, requests };
  } finally {
    await promisify(server.close.bind(server))();
  }
}

export function readWorker(scenario: DeployScenario) {
  return JSON.parse(readFileSync(scenario.workerPath, 'utf8'));
}

export function readEvents(scenario: DeployScenario): Array<{ event: string; [key: string]: unknown }> {
  try {
    return readFileSync(join(scenario.root, 'events.jsonl'), 'utf8').trim().split('\n').map((line) => JSON.parse(line));
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === 'ENOENT') return [];
    throw error;
  }
}

const cloudflarePort = `import assert from 'node:assert/strict';
import { appendFileSync, readFileSync, readdirSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { DatabaseSync } from 'node:sqlite';
const root = process.env.SCENARIO_ROOT;
const args = process.argv.slice(2);
const event = (value) => appendFileSync(join(root, 'events.jsonl'), JSON.stringify(value) + '\\n');
const forbidden = () => { event({ event: 'forbidden', args }); throw new Error('Unexpected mutation or external command'); };
assert.deepEqual(args.slice(0, 2), ['exec', 'wrangler']);
const configPath = args[args.indexOf('--config') + 1];
const config = JSON.parse(readFileSync(configPath, 'utf8').replace(/^\\s*\\/\\/.*$/gm, ''));
assert.equal(config.name, 'puripuly-heart-broker');
assert.equal(config.d1_databases[0].binding, 'BROKER_DB');
assert.equal(config.d1_databases[0].database_name, 'puripuly-heart-broker');
assert.equal(config.d1_databases[0].database_id, 'synthetic-database');
const databasePath = join(root, 'synthetic.sqlite');
if (args[2] === 'types') {
  event({ event: 'validate' });
  if (process.env.MODE === 'config-failure') process.exit(31);
} else if (args[2] === 'd1' && args[3] === 'execute') {
  assert.equal(args[4], 'puripuly-heart-broker');
  assert(args.includes('--remote') && args.includes('--json'));
  const sql = args[args.indexOf('--command') + 1];
  if (!args.includes('--command') || !sql.startsWith('SELECT ') || args.includes('--file')) forbidden();
  event({ event: 'read' });
  if (process.env.MODE === 'read-failure') process.exit(32);
  if (process.env.MODE === 'final-read-failure' && sql.includes("'migration' AS kind")) process.exit(32);
  if (process.env.MODE === 'malformed-read') { process.stdout.write('{}'); process.exit(0); }
  const db = new DatabaseSync(databasePath, { readOnly: true });
  try { process.stdout.write(JSON.stringify([{ success: process.env.MODE !== 'unsuccessful-read', results: db.prepare(sql).all() }])); }
  finally { db.close(); }
} else if (args[2] === 'd1' && args[3] === 'migrations' && args[4] === 'apply') {
  assert.equal(process.env.APPLY_MIGRATIONS, 'true');
  assert.equal(args[5], 'puripuly-heart-broker');
  assert(args.includes('--remote'));
  const dir = resolve(dirname(configPath), config.d1_databases[0].migrations_dir);
  const names = readdirSync(dir).sort();
  const db = new DatabaseSync(databasePath);
  try {
    assert(!names.some((name) => name.startsWith('0025_') || name.startsWith('0027_')));
    const applied = new Set(db.prepare('SELECT name FROM d1_migrations').all().map((row) => row.name));
    const pending = names.filter((name) => !applied.has(name));
    event({ event: 'migrate', pending });
    for (const name of pending) {
      db.exec('BEGIN');
      try {
        db.exec(readFileSync(join(dir, name), 'utf8'));
        if (process.env.MODE === 'migration-failure') db.exec('INSERT INTO nonexistent_failure_probe VALUES (1)');
        db.prepare('INSERT INTO d1_migrations (name) VALUES (?)').run(name);
        db.exec('COMMIT');
      } catch (error) {
        db.exec('ROLLBACK');
        event({ event: 'migration-failed' });
        process.exitCode = 37;
        break;
      }
    }
  } finally { db.close(); }
} else if (args[2] === 'deploy') {
  const db = new DatabaseSync(databasePath, { readOnly: true });
  try { assert(db.prepare(\"SELECT name FROM pragma_table_info('managed_operation_attempts') WHERE name = 'cleanup_token'\").get()); }
  finally { db.close(); }
  event({ event: 'deploy' });
  if (process.env.MODE === 'deploy-failure') process.exit(41);
  const path = join(root, 'worker.json');
  const worker = JSON.parse(readFileSync(path, 'utf8'));
  worker.vars = { ...(args.includes('--keep-vars') ? worker.vars : {}), ...config.vars };
  worker.deployments++;
  writeFileSync(path, JSON.stringify(worker));
} else forbidden();
`;
