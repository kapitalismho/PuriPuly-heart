import { readFileSync, readdirSync, rmSync } from 'node:fs';
import { DatabaseSync } from 'node:sqlite';

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { createDeployScenario, readEvents, readWorker, runDeployWorkflow, type DeployScenario } from './test-support/deploy-scenario';

const scenarios: DeployScenario[] = [];
const codeOnlyInputs = [
  { selection: 'workflow default', env: {} },
  { selection: 'explicit false', env: { APPLY_MIGRATIONS: 'false' } },
];

beforeEach(() => {
  vi.spyOn(globalThis, 'fetch').mockRejectedValue(new Error('External networking is forbidden.'));
});
afterEach(() => {
  vi.restoreAllMocks();
  for (const value of scenarios.splice(0)) rmSync(value.root, { recursive: true, force: true });
});

function scenario(options: Parameters<typeof createDeployScenario>[0] = {}) {
  const value = createDeployScenario(options);
  scenarios.push(value);
  return value;
}

async function deploy(value: DeployScenario) {
  const result = await runDeployWorkflow(value);
  expect(result.executed.at(-1)).toBe('Remove ephemeral deployment files');
  expect(readdirSync(value.runnerTemp)).toEqual([]);
  expect(result.forbiddenAccess).toEqual([]);
  expect(readEvents(value).filter(({ event }) => event === 'forbidden')).toEqual([]);
  return result;
}

function changeDatabase(value: DeployScenario, sql: string) {
  const database = new DatabaseSync(value.databasePath);
  try { database.exec(sql); }
  finally { database.close(); }
}

function records(value: DeployScenario) {
  const database = new DatabaseSync(value.databasePath, { readOnly: true });
  try {
    return {
      usage: database.prepare('SELECT * FROM usage_daily').all(),
      support: database.prepare('SELECT * FROM support_daily').all(),
      turns: database.prepare('SELECT installation_hash, turn_id, utc_date FROM prompt_turns').all(),
      sentinel: database.prepare('SELECT * FROM deploy_sentinel').all(),
    };
  } finally { database.close(); }
}

function expectAdmissionMigration(value: DeployScenario) {
  const database = new DatabaseSync(value.databasePath, { readOnly: true });
  try {
    expect(database.prepare('SELECT name FROM d1_migrations ORDER BY name').all()).toEqual(
      readdirSync(new URL('../migrations/', import.meta.url)).filter((name) => name.endsWith('.sql')).sort().map((name) => ({ name })),
    );
    expect(database.prepare('SELECT expires_at, model_admissions, classifier_admitted FROM prompt_turns').all()).toEqual([
      { expires_at: 0, model_admissions: 15, classifier_admitted: 1 },
    ]);
    expect(database.prepare('SELECT * FROM request_slots').all()).toEqual([]);
    expect(database.prepare('SELECT * FROM protocol_daily').all()).toEqual([]);
  } finally { database.close(); }
}

describe('minimal production Relay deployment', () => {
  it.each(codeOnlyInputs)('deploys code only with $selection while preserving database bytes and deployed secrets', async ({ env }) => {
    const value = scenario({ env });
    const databaseBefore = readFileSync(value.databasePath);
    const workerBefore = readWorker(value);
    const result = await deploy(value);

    expect(result.status, result.stderr).toBe(0);
    expect(readFileSync(value.databasePath)).toEqual(databaseBefore);
    expect(readWorker(value)).toEqual({ ...workerBefore, deployments: 1 });
    expect(readEvents(value).filter(({ event }) => event === 'migrate')).toEqual([]);
    expect(readEvents(value).filter(({ event }) => event === 'deploy')).toHaveLength(1);
    expect(result.stdout).toContain('"status":"rejection_smoke_passed"');
    expect(result.requests).toHaveLength(26);
    expect(result.requests.filter((request) => request.includes('https://puripuly-relay.salee.sarl'))).toHaveLength(13);
    expect(result.requests.filter((request) => request.includes('https://puripuly-heart-agent-relay.kapitalismho.workers.dev'))).toHaveLength(13);
    expect(result.storageRequests.every((request) => request.startsWith('GET '))).toBe(true);
  }, 20000);

  it.each([['true', 'false'], ['false', 'true'], ['false', 'false']])(
    'honors production model=%s reports=%s without resetting non-owned vars or uploading runtime secrets',
    async (AGENT_SERVICE_ENABLED, SUPPORT_REPORTS_ENABLED) => {
      const value = scenario({ env: { AGENT_SERVICE_ENABLED, SUPPORT_REPORTS_ENABLED } });
      const before = readWorker(value);
      const result = await deploy(value);
      expect(result.status, result.stderr).toBe(0);
      expect(readWorker(value)).toEqual({ ...before, deployments: 1, reportsBound: SUPPORT_REPORTS_ENABLED === 'true',
        vars: { ...before.vars, AGENT_SERVICE_ENABLED, SUPPORT_REPORTS_ENABLED } });
      if (SUPPORT_REPORTS_ENABLED === 'false') expect(result.storageRequests).toEqual([]);
    }, 20000,
  );

  it('applies pending migrations and retains all accounting and support data', async () => {
    const value = scenario({ pendingMigration: true, env: { APPLY_MIGRATIONS: 'true' } });
    const before = records(value);
    const workerBefore = readWorker(value);
    const result = await deploy(value);
    expect(result.status, result.stderr).toBe(0);
    expect(records(value)).toEqual(before);
    expectAdmissionMigration(value);
    expect(readWorker(value)).toEqual({ ...workerBefore, deployments: 1 });
  }, 20000);

  it('permits explicitly selected migration maintenance with no pending migration and no database write', async () => {
    const value = scenario({ env: { APPLY_MIGRATIONS: 'true' } });
    const before = readFileSync(value.databasePath);
    const result = await deploy(value);
    expect(result.status, result.stderr).toBe(0);
    expect(readFileSync(value.databasePath)).toEqual(before);
    expect(readEvents(value)).toContainEqual({ event: 'migrate', pending: [] });
    expect(readWorker(value).deployments).toBe(1);
  }, 20000);

  it.each([
    { GITHUB_EVENT_NAME: 'push' }, { GITHUB_REF: 'refs/heads/main' }, { REVIEWED_COMMIT: '2'.repeat(40) },
    { CONFIRM_PRODUCTION_DEPLOY: 'deploy puripuly-heart-agent-relay from dev ' }, { APPLY_MIGRATIONS: 'yes' },
    { AGENT_SERVICE_ENABLED: undefined }, { SUPPORT_REPORTS_ENABLED: '' },
    { CLOUDFLARE_API_TOKEN: '' }, { PURI_TOKEN_SECRET_PRODUCTION: 'short' },
  ])('rejects invalid preparation before remote access or mutation: %j', async (env) => {
    const value = scenario({ env: { APPLY_MIGRATIONS: 'true', ...env } });
    const before = readFileSync(value.databasePath);
    const workerBefore = readWorker(value);
    const result = await deploy(value);
    expect(result.status).not.toBe(0);
    expect(readFileSync(value.databasePath)).toEqual(before);
    expect(readWorker(value)).toEqual(workerBefore);
    expect(readEvents(value)).toEqual([]);
    expect(result.requests).toEqual([]);
    expect(result.storageRequests).toEqual([]);
  }, 20000);

  it('fails domain ownership preflight before D1 and opted-in writes', async () => {
    const value = scenario({ mode: 'foreign-domain', pendingMigration: true, env: { APPLY_MIGRATIONS: 'true' } });
    const before = readFileSync(value.databasePath);
    const workerBefore = readWorker(value);
    const result = await deploy(value);
    expect(result.status).toBe(1);
    expect(result.failedStep).toBe('Verify production Cloudflare target and domain ownership');
    expect(readEvents(value)).toEqual([]);
    expect(readFileSync(value.databasePath)).toEqual(before);
    expect(readWorker(value)).toEqual(workerBefore);
  }, 20000);

  it('allows the same Worker and managed DNS on redeploy but still requires the legacy origin to pass smoke', async () => {
    const redeploy = await deploy(scenario({ mode: 'managed-domain' }));
    expect(redeploy.status, redeploy.stderr).toBe(0);
    const brokenLegacy = await deploy(scenario({ mode: 'legacy-smoke-failure' }));
    expect(brokenLegacy.status).toBe(1);
    expect(brokenLegacy.requests).toHaveLength(14);
    expect(brokenLegacy.stderr).toContain('https://puripuly-heart-agent-relay.kapitalismho.workers.dev: Relay smoke rejection mismatch');
  }, 20000);

  it('retries only custom readiness after TLS propagation delay and completes both smokes without redeploying', async () => {
    const value = scenario({ mode: 'custom-readiness-delay' });
    const result = await deploy(value);
    expect(result.status, result.stderr).toBe(0);
    const custom = result.requests.filter((request) => request.includes('https://puripuly-relay.salee.sarl'));
    const legacy = result.requests.filter((request) => request.includes('https://puripuly-heart-agent-relay.kapitalismho.workers.dev'));
    expect(custom).toHaveLength(14);
    expect(custom.slice(0, 2)).toEqual(Array(2).fill('GET https://puripuly-relay.salee.sarl/deployment-smoke-not-an-endpoint'));
    expect(legacy).toHaveLength(13);
    expect(readEvents(value).filter(({ event }) => event === 'deploy')).toHaveLength(1);
    expect(result.forbiddenAccess).toEqual([]);
  }, 20000);

  it.each(codeOnlyInputs)('blocks pending migrations with $selection before mutation or deployment', async ({ env }) => {
    const value = scenario({ pendingMigration: true, env });
    const before = readFileSync(value.databasePath);
    const workerBefore = readWorker(value);
    const result = await deploy(value);
    expect(result.status, 'Pending ordinary migrations require explicit opt-in.').not.toBe(0);
    expect(result.failedStep).toBe('Require deploy-ready relay schema');
    expect(readFileSync(value.databasePath)).toEqual(before);
    expect(readWorker(value)).toEqual(workerBefore);
    expect(readEvents(value).some(({ event }) => event === 'migrate' || event === 'deploy')).toBe(false);
    expect(result.requests).toEqual([]);
  }, 20000);

  it.each([
    "DELETE FROM d1_migrations WHERE name = '0001_usage_daily.sql'",
    'DROP TABLE usage_daily',
    'ALTER TABLE usage_daily DROP COLUMN missing_usage_count',
    'DROP TABLE d1_migrations',
  ])('rejects uninitialized or fabricated base state before opted-in migrations: %s', async (sql) => {
    const value = scenario({ pendingMigration: true, env: { APPLY_MIGRATIONS: 'true' } });
    changeDatabase(value, sql);
    const before = readFileSync(value.databasePath);
    const result = await deploy(value);
    expect(result.status).not.toBe(0);
    expect(result.failedStep).toBe('Require initialized relay database');
    expect(readFileSync(value.databasePath)).toEqual(before);
    expect(readWorker(value).deployments).toBe(0);
    expect(readEvents(value).some(({ event }) => event === 'migrate' || event === 'deploy')).toBe(false);
    expect(result.requests).toEqual([]);
  }, 20000);

  it('never bootstraps an empty database even with migration opt-in', async () => {
    const value = scenario({ env: { APPLY_MIGRATIONS: 'true' } });
    rmSync(value.databasePath);
    new DatabaseSync(value.databasePath).close();
    const result = await deploy(value);
    expect(result.status).not.toBe(0);
    expect(readFileSync(value.databasePath)).toHaveLength(0);
    expect(readWorker(value).deployments).toBe(0);
    expect(readEvents(value).some(({ event }) => event === 'migrate' || event === 'deploy')).toBe(false);
    expect(result.requests).toEqual([]);
  }, 20000);

  it.each(['false', 'true'])('rejects a completed migration journal without actual admission schema with opt-in=%s', async (APPLY_MIGRATIONS) => {
    const value = scenario({ pendingMigration: true, env: { APPLY_MIGRATIONS } });
    const database = new DatabaseSync(value.databasePath);
    try {
      for (const name of readdirSync(new URL('../migrations/', import.meta.url)).filter((name) => name.endsWith('.sql'))) {
        database.prepare('INSERT OR IGNORE INTO d1_migrations (name) VALUES (?)').run(name);
      }
    } finally { database.close(); }
    const before = readFileSync(value.databasePath);
    const result = await deploy(value);
    expect(result.status).not.toBe(0);
    expect(result.failedStep).toBe('Require deploy-ready relay schema');
    expect(readFileSync(value.databasePath)).toEqual(before);
    expect(readWorker(value).deployments).toBe(0);
    expect(result.requests).toEqual([]);
  }, 20000);

  it.each(['read-failure', 'malformed-read', 'unsuccessful-read', 'malformed-rows', 'migration-failure'])(
    'stops %s without deploying or retaining partial migration effects', async (mode) => {
      const value = scenario({ pendingMigration: true, env: { APPLY_MIGRATIONS: 'true' }, mode });
      const before = readFileSync(value.databasePath);
      const workerBefore = readWorker(value);
      const result = await deploy(value);
      expect(result.status).not.toBe(0);
      expect(result.failedStep).toBe(mode === 'migration-failure' ? 'Apply explicitly selected relay migrations' : 'Require initialized relay database');
      expect(readFileSync(value.databasePath)).toEqual(before);
      expect(readWorker(value)).toEqual(workerBefore);
      expect(readEvents(value).some(({ event }) => event === 'deploy')).toBe(false);
      if (mode !== 'migration-failure') expect(readEvents(value).some(({ event }) => event === 'migrate')).toBe(false);
      expect(result.requests).toEqual([]);
    }, 20000,
  );

  it.each(['final-read-failure', 'deploy-failure', 'smoke-failure'])(
    'leaves successfully committed migration data and journal intact after %s', async (mode) => {
      const value = scenario({ pendingMigration: true, env: { APPLY_MIGRATIONS: 'true' }, mode });
      const before = records(value);
      const workerBefore = readWorker(value);
      const result = await deploy(value);
      expect(result.status).not.toBe(0);
      expect(records(value)).toEqual(before);
      expectAdmissionMigration(value);
      expect(readWorker(value)).toEqual({ ...workerBefore, deployments: mode === 'smoke-failure' ? 1 : 0 });
      if (mode === 'final-read-failure') expect(readEvents(value).some(({ event }) => event === 'deploy')).toBe(false);
      if (mode === 'smoke-failure') {
        expect(result.failedStep).toBe('Smoke endpoint authentication and protocol rejection without model calls');
        expect(result.requests.filter((request) => request.includes('https://puripuly-relay.salee.sarl')))
          .toEqual(['GET https://puripuly-relay.salee.sarl/deployment-smoke-not-an-endpoint']);
        expect(result.requests.filter((request) => request.includes('https://puripuly-heart-agent-relay.kapitalismho.workers.dev'))).toHaveLength(13);
        expect(result.stdout).toContain('"origin":"https://puripuly-heart-agent-relay.kapitalismho.workers.dev","status":"rejection_smoke_passed"');
        expect(result.stdout).not.toContain('"origin":"https://puripuly-relay.salee.sarl","status":"rejection_smoke_passed"');
      } else {
        expect(result.requests).toEqual([]);
        expect(result.stdout).not.toContain('"status":"rejection_smoke_passed"');
      }
    }, 20000,
  );

  it('detects a mismatched verification secret after deploy instead of rotating the existing shared secret', async () => {
    const value = scenario({ env: { PURI_TOKEN_SECRET_PRODUCTION: 'different-synthetic-shared-secret-at-least-32-bytes' } });
    const before = readWorker(value);
    const result = await deploy(value);
    expect(result.status).not.toBe(0);
    expect(result.failedStep).toBe('Smoke endpoint authentication and protocol rejection without model calls');
    expect(readWorker(value)).toEqual({ ...before, deployments: 1 });
    expect(result.stderr).toContain('rejection mismatch');
    expect(result.stdout).not.toContain('"status":"rejection_smoke_passed"');
  }, 20000);

});
