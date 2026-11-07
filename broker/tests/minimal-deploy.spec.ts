import { existsSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { DatabaseSync } from 'node:sqlite';

import { afterEach, describe, expect, it } from 'vitest';

import { FOUNDATION_RESPONSE } from '../src/contract';
import {
  createDeployScenario,
  readEvents,
  readWorker,
  runDeployWorkflow,
  type DeployScenario,
} from './test-support/deploy-scenario';

const scenarios: DeployScenario[] = [];
const codeOnlyInputs = [
  { selection: 'workflow default', env: {} },
  { selection: 'explicit false', env: { APPLY_MIGRATIONS: 'false' } },
];

afterEach(() => {
  for (const scenario of scenarios.splice(0)) rmSync(scenario.root, { recursive: true, force: true });
});

function scenario(options: Parameters<typeof createDeployScenario>[0] = {}) {
  const value = createDeployScenario(options);
  scenarios.push(value);
  return value;
}

function changeDatabase(value: DeployScenario, sql: string) {
  const db = new DatabaseSync(value.databasePath);
  try {
    db.exec(sql);
  } finally {
    db.close();
  }
}

describe('minimal production Broker deployment', () => {
  it.each([...codeOnlyInputs, { selection: 'empty value', env: { APPLY_MIGRATIONS: '' } }])('deploys code once with $selection without database, secret, policy or issuance mutations', async ({ env }) => {
    const value = scenario({ env });
    const databaseBefore = readFileSync(value.databasePath);
    const workerBefore = readWorker(value);
    const result = await runDeployWorkflow(value);

    expect(result.status, result.stderr).toBe(0);
    expect(readFileSync(value.databasePath)).toEqual(databaseBefore);
    expect(readWorker(value)).toEqual({ ...workerBefore, deployments: 1 });
    expect(readEvents(value).map(({ event }) => event)).toEqual(['validate', 'read', 'read', 'read', 'deploy']);
    expect(result.requests).toEqual(['GET /healthz', 'GET /v1/foundation']);
    expect(result.stdout).toContain('Deployed Broker health and public foundation verified.');
    expect(existsSync(join(value.broker, '.deploy-direct'))).toBe(false);
  });

  it('uses explicit current and previous versions without needing provider runtime credentials or replacing deployed secrets', async () => {
    const value = scenario({ env: { NETWORK_IDENTITY_HMAC_KEY_VERSION_PRODUCTION: '9', NETWORK_IDENTITY_HMAC_KEY_VERSION_PREVIOUS_PRODUCTION: '8' } });
    const before = readWorker(value);
    const result = await runDeployWorkflow(value);
    expect(result.status, result.stderr).toBe(0);
    expect(readWorker(value)).toEqual({
      ...before,
      deployments: 1,
      vars: { ...before.vars, NETWORK_IDENTITY_HMAC_KEY_VERSION: '9', NETWORK_IDENTITY_HMAC_KEY_VERSION_PREVIOUS: '8' },
    });
  });

  it('applies only the pending ordinary migration, preserves records and held surfaces, then deploys and verifies public responses', async () => {
    const value = scenario({ pendingMigration: true, env: { APPLY_MIGRATIONS: 'true' } });
    const workerBefore = readWorker(value);
    const result = await runDeployWorkflow(value);
    expect(result.status, result.stderr).toBe(0);
    const db = new DatabaseSync(value.databasePath, { readOnly: true });
    try {
      expect(db.prepare("SELECT name FROM d1_migrations WHERE name = '0026_managed_attempt_cleanup_ownership.sql'").get()).toEqual({
        name: '0026_managed_attempt_cleanup_ownership.sql',
      });
      expect(db.prepare("SELECT name FROM pragma_table_info('managed_operation_attempts') WHERE name IN ('cleanup_token', 'create_in_flight', 'legacy_cleanup_state') ORDER BY name").all()).toEqual([
        { name: 'cleanup_token' }, { name: 'create_in_flight' }, { name: 'legacy_cleanup_state' },
      ]);
      expect(db.prepare('SELECT installation_id, device_public_key, app_version FROM installations').all()).toEqual([
        { installation_id: 'synthetic-installation', device_public_key: 'synthetic-public-key', app_version: 'synthetic' },
      ]);
      expect(db.prepare('SELECT * FROM deploy_sentinel').all()).toEqual([{ value: 'preserve-me' }]);
      expect(db.prepare("SELECT name FROM sqlite_master WHERE type = 'table' AND name IN ('telemetry_active_days', 'telemetry_subjects', 'referral_rewards', 'referral_codes') ORDER BY name").all()).toEqual([
        { name: 'referral_codes' }, { name: 'referral_rewards' }, { name: 'telemetry_active_days' }, { name: 'telemetry_subjects' },
      ]);
      expect(db.prepare('SELECT provider_key_name, outcome, create_in_flight, legacy_cleanup_state FROM managed_operation_attempts').all()).toEqual([
        { provider_key_name: 'synthetic-provider-key-name', outcome: 'unknown', create_in_flight: 1, legacy_cleanup_state: 'resolved' },
      ]);
    } finally {
      db.close();
    }
    expect(readWorker(value)).toEqual({ ...workerBefore, deployments: 1 });
    expect(readEvents(value)).toEqual([
      { event: 'validate' }, { event: 'read' }, { event: 'read' },
      { event: 'migrate', pending: ['0026_managed_attempt_cleanup_ownership.sql'] }, { event: 'read' }, { event: 'deploy' },
    ]);
    expect(result.requests).toEqual(['GET /healthz', 'GET /v1/foundation']);
  });

  it('allows explicitly selected ordinary maintenance when no migration remains pending', async () => {
    const value = scenario({ env: { APPLY_MIGRATIONS: 'true' } });
    const before = readFileSync(value.databasePath);
    const result = await runDeployWorkflow(value);
    expect(result.status, result.stderr).toBe(0);
    expect(readFileSync(value.databasePath)).toEqual(before);
    expect(readEvents(value)).toContainEqual({ event: 'migrate', pending: [] });
    expect(readWorker(value).deployments).toBe(1);
  });

  it.each([
    { GITHUB_REF: 'refs/heads/main' },
    { CONFIRM_PRODUCTION_DEPLOY: 'deploy puripuly-heart-broker from dev ' },
    { APPLY_MIGRATIONS: 'yes' },
    ...['CLOUDFLARE_API_TOKEN', 'CLOUDFLARE_ACCOUNT_ID', 'BROKER_D1_DATABASE_ID_PRODUCTION'].flatMap((key) => [{ [key]: undefined }, { [key]: ' \t ' }]),
    ...['', 'https://other.synthetic.workers.dev', 'https://puripuly-heart-broker.synthetic.workers.dev/extra', 'https://puripuly-heart-broker.synthetic.workers.dev:443', 'http://puripuly-heart-broker.synthetic.workers.dev'].map((url) => ({ BROKER_CANONICAL_WORKERS_DEV_URL: url })),
    ...['', '0', '01', 'v7', '9007199254740992'].map((version) => ({ NETWORK_IDENTITY_HMAC_KEY_VERSION_PRODUCTION: version })),
    ...['0', 'v6', '7', '9007199254740992'].map((version) => ({ NETWORK_IDENTITY_HMAC_KEY_VERSION_PREVIOUS_PRODUCTION: version })),
  ])('rejects invalid deployment configuration %j before D1 or deployment', async (env) => {
    const value = scenario({ env });
    const before = readFileSync(value.databasePath);
    const workerBefore = readWorker(value);
    const result = await runDeployWorkflow(value);
    expect(result.status).not.toBe(0);
    expect(readFileSync(value.databasePath)).toEqual(before);
    expect(readWorker(value)).toEqual(workerBefore);
    expect(readEvents(value)).toEqual([]);
    expect(result.requests).toEqual([]);
  });

  it.each([
    "DELETE FROM d1_migrations WHERE name = '0021_network_identity_purge.sql'",
    "DELETE FROM d1_migrations WHERE name = '0024_allow_unattributed_request_events.sql'",
    "DELETE FROM broker_config WHERE key = 'fingerprint_salt'",
    "UPDATE broker_config SET value = json_set(value, '$.current.salt', '__BOOTSTRAP_REQUIRED__') WHERE key = 'fingerprint_salt'",
    "UPDATE broker_config SET value = json_set(value, '$.current.salt', '   ') WHERE key = 'fingerprint_salt'",
    "UPDATE broker_config SET value = json_set(value, '$.current.version', 0) WHERE key = 'fingerprint_salt'",
    "UPDATE broker_config SET value = json_set(value, '$.phase', 'dual_write') WHERE key = 'network_identity_migration'",
    'ALTER TABLE broker_request_events ADD COLUMN ip TEXT',
    'DROP TABLE broker_config',
    'DROP TABLE d1_migrations',
  ])('rejects unsupported or uninitialized state before opted-in mutation: %s', async (sql) => {
    const value = scenario({ pendingMigration: true, env: { APPLY_MIGRATIONS: 'true' } });
    changeDatabase(value, sql);
    const before = readFileSync(value.databasePath);
    const workerBefore = readWorker(value);
    const result = await runDeployWorkflow(value);
    expect(result.status).not.toBe(0);
    expect(readFileSync(value.databasePath)).toEqual(before);
    expect(readWorker(value)).toEqual(workerBefore);
    expect(readEvents(value).every(({ event }) => ['validate', 'read'].includes(event))).toBe(true);
    expect(result.requests).toEqual([]);
  });

  it('rejects a completely empty database rather than installing or bootstrapping it', async () => {
    const value = scenario({ env: { APPLY_MIGRATIONS: 'true' } });
    rmSync(value.databasePath);
    new DatabaseSync(value.databasePath).close();
    const result = await runDeployWorkflow(value);
    expect(result.status).not.toBe(0);
    expect(readFileSync(value.databasePath)).toHaveLength(0);
    expect(readWorker(value).deployments).toBe(0);
    expect(result.requests).toEqual([]);
  });

  it.each(codeOnlyInputs)('rejects a pending runtime-required migration with $selection before mutation or deployment', async ({ env }) => {
    const value = scenario({ pendingMigration: true, env });
    const before = readFileSync(value.databasePath);
    const workerBefore = readWorker(value);
    const result = await runDeployWorkflow(value);
    expect(result.status, 'Pending ordinary migrations require explicit opt-in.').not.toBe(0);
    expect(readFileSync(value.databasePath)).toEqual(before);
    expect(readWorker(value)).toEqual(workerBefore);
    expect(readEvents(value).some(({ event }) => event === 'migrate' || event === 'deploy')).toBe(false);
    expect(result.requests).toEqual([]);
  });

  it('rejects any missing ordinary history on the code-only path without counting held migrations as pending', async () => {
    const value = scenario();
    changeDatabase(value, "DELETE FROM d1_migrations WHERE name = '0020_network_identity_hmac.sql'");
    const before = readFileSync(value.databasePath);
    const result = await runDeployWorkflow(value);
    expect(result.status).not.toBe(0);
    expect(readFileSync(value.databasePath)).toEqual(before);
    expect(readWorker(value).deployments).toBe(0);
    expect(result.requests).toEqual([]);
  });

  it.each(['false', 'true'])('rejects missing actual cleanup schema despite completed journal with apply_migrations=%s', async (applyMigrations) => {
    const value = scenario({ pendingMigration: true, env: { APPLY_MIGRATIONS: applyMigrations } });
    changeDatabase(value, "INSERT INTO d1_migrations (name) VALUES ('0026_managed_attempt_cleanup_ownership.sql')");
    const before = readFileSync(value.databasePath);
    const result = await runDeployWorkflow(value);
    expect(result.status).not.toBe(0);
    expect(result.stderr).toContain('cleanup schema is missing despite migration history');
    expect(readFileSync(value.databasePath)).toEqual(before);
    expect(readWorker(value).deployments).toBe(0);
    expect(result.requests).toEqual([]);
    if (applyMigrations === 'true') expect(readEvents(value)).toContainEqual({ event: 'migrate', pending: [] });
  });

  it('stops a failed schema read after successful ordinary migration before code deployment', async () => {
    const value = scenario({ pendingMigration: true, env: { APPLY_MIGRATIONS: 'true' }, mode: 'final-read-failure' });
    const result = await runDeployWorkflow(value);
    expect(result.status).not.toBe(0);
    expect(readWorker(value).deployments).toBe(0);
    expect(result.requests).toEqual([]);
    expect(readEvents(value).map(({ event }) => event)).toEqual(['validate', 'read', 'read', 'migrate', 'read']);
    const db = new DatabaseSync(value.databasePath, { readOnly: true });
    try {
      expect(db.prepare("SELECT name FROM d1_migrations WHERE name = '0026_managed_attempt_cleanup_ownership.sql'").get()).toBeDefined();
      expect(db.prepare("SELECT name FROM pragma_table_info('managed_operation_attempts') WHERE name = 'cleanup_token'").get()).toBeDefined();
    } finally {
      db.close();
    }
  });

  it.each(['config-failure', 'read-failure', 'malformed-read', 'unsuccessful-read', 'migration-failure', 'deploy-failure'])('stops %s before later mutation or public success checks', async (mode) => {
    const value = scenario({ pendingMigration: true, env: { APPLY_MIGRATIONS: 'true' }, mode });
    const databaseBefore = readFileSync(value.databasePath);
    const workerBefore = readWorker(value);
    const result = await runDeployWorkflow(value);
    expect(result.status).not.toBe(0);
    expect(readWorker(value)).toEqual(workerBefore);
    expect(result.requests).toEqual([]);
    expect(result.stdout).not.toContain('public foundation verified');
    const events = readEvents(value).map(({ event }) => event);
    expect(events).not.toContain('forbidden');
    if (mode === 'deploy-failure') {
      expect(events).toEqual(['validate', 'read', 'read', 'migrate', 'read', 'deploy']);
      const db = new DatabaseSync(value.databasePath, { readOnly: true });
      try {
        expect(db.prepare("SELECT name FROM d1_migrations WHERE name = '0026_managed_attempt_cleanup_ownership.sql'").get()).toBeDefined();
      } finally {
        db.close();
      }
    } else {
      expect(readFileSync(value.databasePath)).toEqual(databaseBefore);
      expect(events).not.toContain('deploy');
      if (mode === 'migration-failure') expect(events).toEqual(['validate', 'read', 'read', 'migrate', 'migration-failed']);
      else expect(events).not.toContain('migrate');
    }
  });

  it.each([
    { '/healthz': { status: 503 } },
    { '/healthz': { status: 201 } },
    { '/healthz': { body: '<html>not JSON</html>' } },
    { '/healthz': { body: { ok: false, service: 'puripuly-heart-broker' } } },
    { '/healthz': { body: { ok: true, service: 'wrong-service' } } },
    { '/v1/foundation': { status: 302 } },
    { '/v1/foundation': { body: {} } },
    { '/v1/foundation': { body: { ...FOUNDATION_RESPONSE, service: 'wrong-service' } } },
    { '/v1/foundation': { body: { ...FOUNDATION_RESPONSE, serviceBoundary: { ...FOUNDATION_RESPONSE.serviceBoundary, proxiesTranslationText: true } } } },
    { '/v1/foundation': { body: { ...FOUNDATION_RESPONSE, trialProviderPolicy: { managedFreeTrial: { provider: 'Other', models: ['wrong'] } } } } },
    { '/v1/foundation': { body: { ...FOUNDATION_RESPONSE, runtime: { ...FOUNDATION_RESPONSE.runtime, runtime: 'Other' } } } },
    { '/v1/foundation': { body: { ...FOUNDATION_RESPONSE, bindings: { d1: 'WRONG_DB' } } } },
    { '/v1/foundation': { body: { ...FOUNDATION_RESPONSE, trialProviderPolicy: { managedFreeTrial: { provider: 'OpenRouter', models: [] } } } } },
    { '/v1/foundation': { body: { ...FOUNDATION_RESPONSE, trialProviderPolicy: { managedFreeTrial: { provider: 'OpenRouter', models: 'not-a-list' } } } } },
    { '/v1/foundation': { body: { ...FOUNDATION_RESPONSE, trialProviderPolicy: { managedFreeTrial: { provider: 'OpenRouter', models: ['not-an-allowed-model'] } } } } },
  ])('rejects incorrect public status or contract %j instead of accepting arbitrary HTTP 200', async (responses) => {
    const value = scenario();
    const before = readFileSync(value.databasePath);
    const result = await runDeployWorkflow(value, responses);
    expect(result.status).toBe(1);
    expect(result.stdout).not.toContain('public foundation verified');
    expect(result.requests).toHaveLength(24);
    expect(readWorker(value).deployments).toBe(1);
    expect(readFileSync(value.databasePath)).toEqual(before);
    expect(readEvents(value).map(({ event }) => event)).toEqual(['validate', 'read', 'read', 'read', 'deploy']);
  });

  it('fails renderer rejection before remote admission when canonical configuration has drifted', async () => {
    const value = scenario({ env: { APPLY_MIGRATIONS: 'true' } });
    const path = join(value.broker, 'wrangler.jsonc');
    writeFileSync(path, readFileSync(path, 'utf8').replace('"name": "puripuly-heart-broker"', '"name": "other-worker"'));
    const result = await runDeployWorkflow(value);
    expect(result.status).not.toBe(0);
    expect(readEvents(value)).toEqual([]);
    expect(readWorker(value).deployments).toBe(0);
    expect(result.requests).toEqual([]);
  });
});
