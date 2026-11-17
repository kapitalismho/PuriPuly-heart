import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import app from '../src/app';
import { createTestBrokerEnv, type TestBrokerEnv } from './test-support/sqlite-d1';
import { signToken, tokenSecret } from '../../agent-relay/tests/fixtures';
import { readBrokerMigrationSql } from './test-support/migrations';

type PhotoCredential = {
  operation_id: string; managed_credential_ref: string; delivery_id: string;
  delivery_ack_token: string; limit_usd: number;
};

const databases: TestBrokerEnv[] = [];
function environment() {
  const env = Object.assign(createTestBrokerEnv(), {
    PURI_TOKEN_SECRET: tokenSecret, PHOTO_OPENROUTER_ENABLED: 'true',
    OPENROUTER_PHOTO_GUARDRAIL_ID: 'photo-only-policy', PHOTO_OPENROUTER_LIMIT_USD: '0.25',
  });
  databases.push(env);
  return env;
}

function post(env: TestBrokerEnv, body: Record<string, unknown>, token = signToken()) {
  return app.request('https://broker.test/v1/photo/access', {
    method: 'POST', headers: { authorization: `Bearer ${token}`, 'content-type': 'application/json' },
    body: JSON.stringify(body),
  }, env);
}

function provider() {
  const keys = new Map<string, Record<string, unknown>>();
  let creates = 0;
  let assignmentVisible = true;
  let uncertainCreation = false;
  const fetcher = vi.fn(async (url: string, init?: RequestInit) => {
    const path = new URL(url).pathname.replace('/api/v1', '');
    const method = init?.method ?? 'GET';
    const body = init?.body ? JSON.parse(String(init.body)) : {};
    if (path === '/guardrails/photo-only-policy' && method === 'GET') {
      return Response.json({ data: { allowed_models: ['openai/gpt-6-luna'], allowed_providers: ['openai'],
        limit_usd: null, include_byok_in_budgets: true, ignored_models: null, ignored_providers: null } });
    }
    if (path === '/keys' && method === 'POST') {
      creates++;
      const hash = `synthetic-photo-key-${creates}`;
      const key = { ...body, hash, disabled: false, usage: 0 };
      keys.set(hash, key);
      if (uncertainCreation) { uncertainCreation = false; throw new Error('Lost creation response'); }
      return Response.json({ key: `synthetic-raw-${hash}`, data: key });
    }
    if (path === '/keys' && method === 'GET') return Response.json({ data: Array.from(keys.values()) });
    if (path === '/guardrails/photo-only-policy/assignments/keys') {
      if (method === 'POST') {
        for (const hash of body.key_hashes) keys.get(hash)!.guardrail_id = 'photo-only-policy';
        return Response.json({ assigned_count: body.key_hashes.length });
      }
      const data = assignmentVisible ? Array.from(keys.values()).filter((key) => key.guardrail_id)
        .map((key) => ({ guardrail_id: key.guardrail_id, key_hash: key.hash })) : [];
      return Response.json({ data, total_count: data.length });
    }
    if (path.startsWith('/keys/')) {
      const hash = path.slice('/keys/'.length);
      const key = keys.get(hash);
      if (!key) return Response.json({ error: { code: 404 } }, { status: 404 });
      if (method === 'PATCH') Object.assign(key, body);
      if (method === 'DELETE') { keys.delete(hash); return Response.json({ deleted: true }); }
      return Response.json({ data: key });
    }
    throw new Error(`Unexpected synthetic provider request ${method} ${path}`);
  });
  vi.stubGlobal('fetch', fetcher);
  return { keys, fetcher, creates: () => creates,
    hideAssignment: () => { assignmentVisible = false; },
    restoreAssignment: () => { assignmentVisible = true; },
    loseCreation: () => { uncertainCreation = true; },
  };
}

beforeEach(() => { vi.useFakeTimers({ toFake: ['Date'] }); vi.setSystemTime(new Date('2026-11-09T12:00:00Z')); });
afterEach(() => { vi.unstubAllGlobals(); vi.useRealTimers(); for (const env of databases.splice(0)) env.__db.close(); });

it('delivers only a disabled, verified photo key; lost acknowledgement retries the same allowance', async () => {
  const env = environment(); const upstream = provider();
  const delivered = await post(env, { action: 'get' });
  expect(delivered.status).toBe(200);
  const credential = await delivered.json() as PhotoCredential;
  expect(upstream.keys.get(credential.managed_credential_ref)).toMatchObject({
    disabled: true, limit: 0.25, limit_reset: null, include_byok_in_limit: true,
    guardrail_id: 'photo-only-policy', expires_at: '2026-11-10T00:00:00.000Z',
  });
  const ack = { action: 'acknowledge', operation_id: credential.operation_id,
    delivery_id: credential.delivery_id, managed_credential_ref: credential.managed_credential_ref,
    delivery_ack_token: credential.delivery_ack_token };
  expect((await post(env, ack)).status).toBe(200);
  expect(upstream.keys.get(credential.managed_credential_ref)?.disabled).toBe(false);
  vi.setSystemTime(new Date(Date.now() + 1000));
  expect((await post(env, ack, signToken())).status).toBe(200);
  expect((await post(env, { action: 'get' })).status).toBe(409);
  expect(upstream.creates()).toBe(1);
  expect(env.__db.prepare('SELECT COUNT(*) AS count FROM openrouter_entitlements').get()?.count).toBe(0);
  const revoked = await post(env, { action: 'revoke', operation_id: credential.operation_id });
  expect(await revoked.json()).toMatchObject({ revoked: true });
  expect(upstream.keys.has(credential.managed_credential_ref)).toBe(false);
  expect((await post(env, ack)).status).toBe(409);
  expect((await post(env, { action: 'get' })).status).toBe(409);
  expect(upstream.creates()).toBe(1);
  expect((await post(env, { action: 'get' }, signToken({ sub: 'b'.repeat(64) }))).status).toBe(200);
  vi.setSystemTime(new Date('2026-11-10T00:01:00Z'));
  expect((await post(env, { action: 'get' })).status).toBe(200);
});

it('recovers a lost disabled delivery without granting usable overlapping credentials', async () => {
  const env = environment(); const upstream = provider();
  const first = await (await post(env, { action: 'get' })).json() as PhotoCredential;
  expect((await post(env, { action: 'get' })).status).toBe(409);
  expect(upstream.creates()).toBe(1);
  vi.setSystemTime(new Date('2026-11-09T12:03:00Z'));
  const replacement = await post(env, { action: 'get' });
  expect(replacement.status).toBe(200);
  const second = await replacement.json() as PhotoCredential;
  expect(second.operation_id).toBe(first.operation_id);
  expect(second.limit_usd).toBe(first.limit_usd);
  expect(upstream.keys.has(first.managed_credential_ref)).toBe(false);
  expect(upstream.keys.get(second.managed_credential_ref)?.disabled).toBe(true);
  expect((await post(env, { action: 'acknowledge', operation_id: first.operation_id,
    delivery_id: first.delivery_id, managed_credential_ref: first.managed_credential_ref,
    delivery_ack_token: first.delivery_ack_token })).status).toBe(409);
});

it('does not release unverified policy and reconciles uncertain creation before another attempt', async () => {
  const env = environment(); const upstream = provider();
  upstream.loseCreation();
  const uncertain = await post(env, { action: 'get' });
  expect(uncertain.status).toBe(503);
  expect(await uncertain.json()).not.toHaveProperty('key');
  upstream.hideAssignment();
  const unverified = await post(env, { action: 'get' });
  expect(unverified.status).toBe(503);
  expect(await unverified.json()).not.toHaveProperty('key');
  expect(upstream.keys.has('synthetic-photo-key-1')).toBe(false);
  expect(upstream.keys.get('synthetic-photo-key-2')?.disabled).toBe(true);
  upstream.restoreAssignment();
  const recovered = await post(env, { action: 'get' });
  expect(recovered.status).toBe(200);
  expect(upstream.keys.has('synthetic-photo-key-2')).toBe(false);
  expect(Array.from(upstream.keys.values()).filter((key) => !key.disabled)).toEqual([]);
});

it('requires valid Puri authorization and denies cross-installation activation/revocation', async () => {
  const env = environment(); const upstream = provider();
  expect((await post(env, { action: 'get' }, 'not-a-token')).status).toBe(401);
  expect((await post(env, { action: 'get' }, signToken({ exp: Math.floor(Date.now() / 1000) }))).status).toBe(401);
  expect(upstream.fetcher).not.toHaveBeenCalled();
  const credential = await (await post(env, { action: 'get' })).json() as PhotoCredential;
  const other = signToken({ sub: 'c'.repeat(64) });
  expect((await post(env, { action: 'revoke', operation_id: credential.operation_id }, other)).status).toBe(404);
  expect((await post(env, { action: 'acknowledge', operation_id: credential.operation_id,
    delivery_id: credential.delivery_id, managed_credential_ref: credential.managed_credential_ref,
    delivery_ack_token: credential.delivery_ack_token }, other)).status).toBe(404);
  expect(upstream.keys.get(credential.managed_credential_ref)?.disabled).toBe(true);
});

it('serializes concurrent provisioning before creating a provider key', async () => {
  const env = environment(); const upstream = provider();
  const { promise: started, resolve: entered } = Promise.withResolvers<void>();
  const { promise: gate, resolve: release } = Promise.withResolvers<void>();
  const implementation = upstream.fetcher.getMockImplementation()!;
  upstream.fetcher.mockImplementation(async (url, init) => {
    if (new URL(url).pathname === '/api/v1/keys' && init?.method === 'POST') {
      entered(); await gate;
    }
    return implementation(url, init);
  });
  const first = post(env, { action: 'get' });
  await started;
  const second = await post(env, { action: 'get' });
  release();
  const delivered = await first;
  expect(second.status).toBe(409);
  expect(delivered.status).toBe(200);
  expect(upstream.creates()).toBe(1);
  expect(env.__db.prepare("SELECT COUNT(*) AS n FROM managed_operations WHERE issue_source = 'photo'").get()?.n).toBe(1);
});

it('migrates pending personal managed deliveries without losing ownership, references or sequence history', () => {
  const env = createTestBrokerEnv({ throughMigration: '0030_puri_first_enrollment_limits.sql' });
  databases.push(env);
  const db = env.__db;
  db.exec('PRAGMA foreign_keys = ON');
  db.prepare(`INSERT INTO managed_operations (operation_id, issue_source, subject_ref, installation_id,
    state, attempt_count, current_attempt_index, resume_token_hash, auth_expires_at, client_action)
    VALUES ('ph-mop-v1_existing', 'discord', 'existing-subject', 'existing-installation',
      'DELIVERY_PENDING', 1, 1, ?, '2026-11-10T00:00:00Z', 'acknowledge_delivery')`)
    .run(`ph-mop-resume-v1_${'a'.repeat(64)}`);
  db.exec(`INSERT INTO managed_operation_attempts (id, operation_id, attempt_index, provider_key_name,
    managed_credential_ref, outcome, legacy_cleanup_state)
    VALUES (41, 'ph-mop-v1_existing', 1, 'existing-provider-key', 'existing-key-hash', 'created', 'resolved');
    UPDATE sqlite_sequence SET seq = 400 WHERE name = 'managed_operation_attempts'`);
  db.prepare(`INSERT INTO managed_key_deliveries (delivery_id, issue_source, installation_id,
    managed_credential_ref, ack_token_hash, status, created_at, expires_at, operation_id, attempt_index)
    VALUES ('existing-delivery', 'discord', 'existing-installation', 'existing-key-hash', ?, 'pending',
      '2026-11-09T12:00:00Z', '2026-11-09T12:05:00Z', 'ph-mop-v1_existing', 1)`)
    .run(`ph-delivery-ack-token-v1_${'b'.repeat(64)}`);
  db.exec(readBrokerMigrationSql('0031_photo_managed_access.sql'));
  expect(db.prepare('SELECT state, current_attempt_index, photo_activated_at FROM managed_operations').get())
    .toMatchObject({ state: 'DELIVERY_PENDING', current_attempt_index: 1, photo_activated_at: null });
  expect(db.prepare('SELECT operation_id, managed_credential_ref, status FROM managed_key_deliveries').get())
    .toMatchObject({ operation_id: 'ph-mop-v1_existing', managed_credential_ref: 'existing-key-hash', status: 'pending' });
  expect(db.prepare('SELECT id, outcome FROM managed_operation_attempts').get()).toMatchObject({ id: 41, outcome: 'created' });
  expect(db.prepare("SELECT seq FROM sqlite_sequence WHERE name = 'managed_operation_attempts'").get()?.seq).toBe(400);
  expect(db.prepare('PRAGMA foreign_key_check').all()).toEqual([]);
});
