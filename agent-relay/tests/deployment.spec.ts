import { mkdtemp, readFile, readdir, rm, stat, writeFile } from 'node:fs/promises';
import { createServer } from 'node:http';
import type { AddressInfo } from 'node:net';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { DatabaseSync } from 'node:sqlite';
import { fileURLToPath } from 'node:url';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { confirmation, deploymentOrigin, guardDeployment, prepareDeployment, verifyDeploymentSchema } from '../operations/prepare-deployment.mjs';

const commit = '1'.repeat(40);
const base = {
  GITHUB_EVENT_NAME: 'workflow_dispatch', GITHUB_REF: 'refs/heads/dev', GITHUB_SHA: commit,
  REVIEWED_COMMIT: commit, CONFIRM_PRODUCTION_DEPLOY: confirmation,
  AGENT_SERVICE_ENABLED: 'false', SUPPORT_REPORTS_ENABLED: 'false',
  AGENT_RELAY_CANONICAL_WORKERS_DEV_URL: 'https://puripuly-heart-agent-relay.synthetic-account.workers.dev',
  CLOUDFLARE_ACCOUNT_ID: 'a'.repeat(32), CLOUDFLARE_API_TOKEN: 'synthetic-cf-token-not-real',
  AGENT_USAGE_D1_DATABASE_ID_PRODUCTION: '11111111-1111-4111-8111-111111111111',
  BROKER_D1_DATABASE_ID_PRODUCTION: '22222222-2222-4222-8222-222222222222',
  PURI_TOKEN_SECRET_PRODUCTION: 'synthetic-deployment-token-at-least-32-bytes',
  GITHUB_SERVER_URL: 'https://github.com', GITHUB_REPOSITORY: 'synthetic/repo', GITHUB_RUN_ID: '123', GITHUB_RUN_ATTEMPT: '2',
};
const temporary: string[] = [];
async function directory() {
  const path = await mkdtemp(join(tmpdir(), 'relay-deployment-test-'));
  temporary.push(path);
  return path;
}
afterEach(async () => {
  await Promise.all(temporary.splice(0).map((path) => rm(path, { recursive: true, force: true })));
  vi.restoreAllMocks();
});

const desired = { id: 'support-report-retention-30d', enabled: true, conditions: { prefix: 'support/' },
  deleteObjectsTransition: { condition: { type: 'Age', maxAge: 2592000 } } };

async function storage<T>(run: (fetcher: typeof fetch, calls: string[]) => Promise<T>, options: {
  publicAccess?: boolean; customDomains?: unknown[]; rules?: unknown[];
} = {}) {
  const calls: string[] = [];
  const server = createServer((request, response) => {
    calls.push(`${request.method} ${request.url}`);
    const result = request.url?.endsWith('/domains/managed') ? { enabled: options.publicAccess ?? false }
      : request.url?.endsWith('/domains/custom') ? { domains: options.customDomains ?? [] } : { rules: options.rules ?? [desired] };
    response.writeHead(200, { 'content-type': 'application/json' });
    response.end(JSON.stringify({ success: true, result }));
  });
  const ready = Promise.withResolvers<void>();
  server.once('error', ready.reject);
  server.listen(0, '127.0.0.1', ready.resolve);
  await ready.promise;
  const port = (server.address() as AddressInfo).port;
  const fetcher: typeof fetch = (input, init) => {
    const url = new URL(String(input));
    expect(url.origin).toBe('https://api.cloudflare.com');
    expect(init?.method).toBe('GET');
    return fetch(`http://127.0.0.1:${port}${url.pathname}`, init);
  };
  try { return await run(fetcher, calls); }
  finally {
    const closed = Promise.withResolvers<void>();
    server.close((error) => error ? closed.reject(error) : closed.resolve());
    await closed.promise;
  }
}

describe('manual relay deployment preparation', () => {
  it('accepts only confirmed manual dev dispatch with identical reviewed, dispatch and checkout SHAs', () => {
    expect(() => guardDeployment(base, commit)).not.toThrow();
    for (const change of [{ GITHUB_EVENT_NAME: 'push' }, { GITHUB_REF: 'refs/heads/release' },
      { CONFIRM_PRODUCTION_DEPLOY: `${confirmation} ` }, { REVIEWED_COMMIT: 'dev' },
      { REVIEWED_COMMIT: 'A'.repeat(40) }, { REVIEWED_COMMIT: '2'.repeat(40) }, { GITHUB_SHA: '2'.repeat(40) },
      { AGENT_SERVICE_ENABLED: '' }, { SUPPORT_REPORTS_ENABLED: 'TRUE' }]) {
      expect(() => guardDeployment({ ...base, ...change }, commit)).toThrow();
    }
    expect(() => guardDeployment(base, '2'.repeat(40))).toThrow();
    for (const APPLY_MIGRATIONS of [undefined, '', 'false', 'true']) {
      expect(() => guardDeployment({ ...base, APPLY_MIGRATIONS }, commit)).not.toThrow();
    }
    for (const APPLY_MIGRATIONS of ['TRUE', 'yes', '0', ' true ']) {
      expect(() => guardDeployment({ ...base, APPLY_MIGRATIONS }, commit)).toThrow();
    }
  });

  it('rejects noncanonical, credential-bearing, preview, path and query smoke origins', () => {
    expect(deploymentOrigin(`${base.AGENT_RELAY_CANONICAL_WORKERS_DEV_URL}/`)).toBe(base.AGENT_RELAY_CANONICAL_WORKERS_DEV_URL);
    for (const origin of ['http://puripuly-heart-agent-relay.account.workers.dev', 'https://other.account.workers.dev',
      'https://puripuly-heart-agent-relay.account.workers.dev.evil.test', 'https://puripuly-heart-agent-relay.preview.account.workers.dev',
      'https://user:secret@puripuly-heart-agent-relay.account.workers.dev', `${base.AGENT_RELAY_CANONICAL_WORKERS_DEV_URL}/v1/respond`,
      `${base.AGENT_RELAY_CANONICAL_WORKERS_DEV_URL}?secret=value`, `${base.AGENT_RELAY_CANONICAL_WORKERS_DEV_URL}#fragment`]) {
      expect(() => deploymentOrigin(origin)).toThrow();
    }
  });

  it('writes only non-secret config for the dedicated database with explicit stopped flags and no R2', async () => {
    const path = await directory();
    const fetcher = vi.fn<typeof fetch>();
    await prepareDeployment({ env: { ...base, AGENT_RELAY_OPENAI_API_KEY_PRODUCTION: ' ', DISCORD_OPERATIONS_WEBHOOK_URL_PRODUCTION: '' }, directory: path, fetcher });
    const configText = await readFile(join(path, 'wrangler.json'), 'utf8');
    const config = JSON.parse(configText);
    expect(config).toMatchObject({ name: 'puripuly-heart-agent-relay', workers_dev: true, preview_urls: false,
      observability: { enabled: false }, vars: { AGENT_SERVICE_ENABLED: 'false', SUPPORT_REPORTS_ENABLED: 'false' },
      secrets: { required: ['PURI_TOKEN_SECRET'] } });
    expect(config.r2_buckets).toBeUndefined();
    expect(config.d1_databases).toHaveLength(1);
    expect(config.d1_databases[0]).toMatchObject({ binding: 'AGENT_USAGE_DB', database_id: base.AGENT_USAGE_D1_DATABASE_ID_PRODUCTION });
    expect(await readdir(path)).toEqual(['wrangler.json']);
    expect(configText).not.toContain(base.PURI_TOKEN_SECRET_PRODUCTION);
    expect(configText).not.toContain(base.CLOUDFLARE_API_TOKEN);
    expect(fetcher).not.toHaveBeenCalled();
    if (process.platform !== 'win32') {
      expect((await stat(path)).mode & 0o777).toBe(0o700);
      expect((await stat(join(path, 'wrangler.json'))).mode & 0o777).toBe(0o600);
    }
  });

  it('verifies real read-only storage boundaries before writing enabled report config, independently of models', async () => {
    await storage(async (fetcher, calls) => {
      const path = await directory();
      await prepareDeployment({ env: { ...base, SUPPORT_REPORTS_ENABLED: 'true' }, directory: path, fetcher });
      const config = JSON.parse(await readFile(join(path, 'wrangler.json'), 'utf8'));
      expect(config.vars).toEqual({ AGENT_SERVICE_ENABLED: 'false', SUPPORT_REPORTS_ENABLED: 'true' });
      expect(config.r2_buckets).toEqual([{ binding: 'SUPPORT_REPORTS', bucket_name: 'puripuly-heart-support-reports' }]);
      expect(calls.every((call) => call.startsWith('GET '))).toBe(true);
      expect(await readdir(path)).toEqual(['wrangler.json']);
    });
  });

  it.each([
    { publicAccess: true }, { customDomains: [{ domain: 'public.synthetic.test' }] },
    { rules: [] }, { rules: [{ ...desired, deleteObjectsTransition: { condition: { type: 'Age', maxAge: 86400 } } }] },
  ])('rejects public or incorrectly retained report storage without writing assets or mutating storage: %j', async (options) => {
    await storage(async (fetcher, calls) => {
      const path = await directory();
      await expect(prepareDeployment({ env: { ...base, SUPPORT_REPORTS_ENABLED: 'true' }, directory: path, fetcher })).rejects.toThrow();
      expect(await readdir(path)).toEqual([]);
      expect(calls.every((call) => call.startsWith('GET '))).toBe(true);
    }, options);
  });

  it('preserves enabled models without reading provider or webhook secret inputs', async () => {
    const path = await directory();
    const env = { ...base, AGENT_SERVICE_ENABLED: 'true' };
    for (const name of ['AGENT_RELAY_OPENAI_API_KEY_PRODUCTION', 'DISCORD_OPERATIONS_WEBHOOK_URL_PRODUCTION']) {
      Object.defineProperty(env, name, { get() { throw new Error('Deployment must not read runtime secrets.'); } });
    }
    await prepareDeployment({ env, directory: path });
    const config = JSON.parse(await readFile(join(path, 'wrangler.json'), 'utf8'));
    expect(config.vars).toMatchObject({ AGENT_SERVICE_ENABLED: 'true', SUPPORT_REPORTS_ENABLED: 'false' });
    expect(config.secrets.required).toEqual(['PURI_TOKEN_SECRET', 'OPENAI_API_KEY']);
    expect(await readdir(path)).toEqual(['wrangler.json']);
  });

  it('measures the verification-only shared token in UTF-8 bytes', async () => {
    const path = await directory();
    const secret = 'é'.repeat(16);
    await prepareDeployment({ env: { ...base, PURI_TOKEN_SECRET_PRODUCTION: secret }, directory: path });
    expect(await readdir(path)).toEqual(['wrangler.json']);
    expect(await readFile(join(path, 'wrangler.json'), 'utf8')).not.toContain(secret);
  });

  it('fails missing credentials, placeholder/shared databases and malformed flags before writing or remote access', async () => {
    const fetcher = vi.fn<typeof fetch>();
    for (const change of [{ CLOUDFLARE_ACCOUNT_ID: '' }, { CLOUDFLARE_ACCOUNT_ID: 'not-an-account' }, { CLOUDFLARE_API_TOKEN: ' ' },
      { AGENT_USAGE_D1_DATABASE_ID_PRODUCTION: 'REQUIRED_AT_DEPLOY_TIME' }, { AGENT_USAGE_D1_DATABASE_ID_PRODUCTION: '00000000-0000-0000-0000-000000000000' },
      { AGENT_USAGE_D1_DATABASE_ID_PRODUCTION: base.BROKER_D1_DATABASE_ID_PRODUCTION }, { BROKER_D1_DATABASE_ID_PRODUCTION: '' },
      { PURI_TOKEN_SECRET_PRODUCTION: '' }, { PURI_TOKEN_SECRET_PRODUCTION: 'short' }, { PURI_TOKEN_SECRET_PRODUCTION: 'é'.repeat(15) },
      { AGENT_SERVICE_ENABLED: 'enabled' }, { SUPPORT_REPORTS_ENABLED: '' },
      { AGENT_RELAY_CANONICAL_WORKERS_DEV_URL: 'https://other.synthetic.workers.dev' }]) {
      const path = await directory();
      await expect(prepareDeployment({ env: { ...base, ...change }, directory: path, fetcher })).rejects.toThrow();
      expect(await readdir(path)).toEqual([]);
    }
    expect(fetcher).not.toHaveBeenCalled();
  });

  it('refuses repository output and never overwrites an existing config', async () => {
    await expect(prepareDeployment({ env: base, directory: fileURLToPath(new URL('../..', import.meta.url)) })).rejects.toThrow();
    const path = await directory();
    const existing = '{"preserve":"existing configuration"}\n';
    await writeFile(join(path, 'wrangler.json'), existing);
    await expect(prepareDeployment({ env: base, directory: path })).rejects.toThrow();
    expect(await readFile(join(path, 'wrangler.json'), 'utf8')).toBe(existing);
    expect(await readdir(path)).toEqual(['wrangler.json']);
  });

  it.each([
    undefined, {}, [], [{ success: false, results: [] }], [{ success: true, results: {} }],
    [{ success: true, results: [] }, { success: true, results: [] }],
    [{ success: true, results: [{ kind: 'migration', name: '0001_usage_daily.sql' }] }],
    [{ success: true, results: [{ kind: 'column', name: 1, table_name: 'usage_daily' }] }],
    [{ success: true, results: [{ kind: 'unknown', name: 'installation_hash', table_name: 'usage_daily' }] }],
  ])('rejects malformed Wrangler schema payloads: %j', async (payload) => {
    await expect(verifyDeploymentSchema({ payload })).rejects.toThrow();
    await expect(verifyDeploymentSchema({ payload, initializedOnly: true })).rejects.toThrow();
  });

  it('validates real checked-in schema and requires every current migration and runtime column', async () => {
    const database = new DatabaseSync(':memory:');
    const rows: Array<{ kind: string; name: string; table_name: string }> = [];
    try {
      for (const name of (await readdir(new URL('../migrations/', import.meta.url))).filter((name) => name.endsWith('.sql')).sort()) {
        database.exec(await readFile(new URL(`../migrations/${name}`, import.meta.url), 'utf8'));
        rows.push({ kind: 'migration', name, table_name: '' });
      }
      for (const table of ['usage_daily', 'support_daily', 'prompt_turns', 'request_slots', 'protocol_daily']) {
        for (const row of database.prepare('SELECT name FROM pragma_table_info(?)').all(table)) {
          rows.push({ kind: 'column', name: String(row.name), table_name: table });
        }
      }
    } finally { database.close(); }
    const payload = (results: typeof rows) => [{ success: true, results }];
    await verifyDeploymentSchema({ payload: payload(rows) });
    const initialized = rows.filter((row) => row.table_name === 'usage_daily' || row.name === '0001_usage_daily.sql');
    await verifyDeploymentSchema({ payload: payload(initialized), initializedOnly: true });
    await expect(verifyDeploymentSchema({ payload: payload(initialized) })).rejects.toThrow();
    for (const row of rows) {
      await expect(verifyDeploymentSchema({ payload: payload(rows.filter((entry) => entry !== row)) })).rejects.toThrow();
    }
    for (const row of initialized) {
      await expect(verifyDeploymentSchema({ payload: payload(initialized.filter((entry) => entry !== row)), initializedOnly: true })).rejects.toThrow();
    }
  });
});
