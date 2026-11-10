import { createHmac } from 'node:crypto';

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import app from '../src/index';
import { getBrokerAbuseControlsConfig } from '../src/abuse-controls';
import { encodeBase64Url, randomBase64Url } from '../src/crypto';
import {
  deriveStableNetworkIdentityDigest,
  resolveNetworkIdentitySecrets,
  resolveRequestNetworkIdentityCandidates,
} from '../src/network-identity';
import { DEFAULT_BROKER_ABUSE_CONTROLS } from '../src/persistence';
import { deleteExpiredPuriProofNonces } from '../src/puri-token';
import { readAbuseControls } from './test-support/abuse-controls';
import { createDeviceKeyPair, type DeviceKeyPair } from './test-support/ed25519';
import { readBrokerMigrationSql } from './test-support/migrations';
import { createTestBrokerEnv, insertEntitlement, type TestBrokerEnv } from './test-support/sqlite-d1';

const installationId = '019a3544-3380-7000-8000-000000000330';
const otherId = '019a3544-3380-7000-8000-000000000331';
const tokenSecret = 'puri-test-token-secret-0000000000000000';
const identitySecret = 'puri-test-identity-secret-0000000000000';
const databases: TestBrokerEnv[] = [];
const encoder = new TextEncoder();

function environment(options: Parameters<typeof createTestBrokerEnv>[0] = {}) {
  const env = Object.assign(createTestBrokerEnv(options), {
    PURI_TOKEN_SECRET: tokenSecret, PURI_IDENTITY_HMAC_SECRET: identitySecret,
  });
  databases.push(env);
  return env;
}

async function proof(key: DeviceKeyPair, overrides: Record<string, string> = {}, domain = 'puripuly-heart:puri-token:v1') {
  const request = {
    installation_id: installationId, device_public_key: key.devicePublicKey,
    app_version: '1.2.3', signed_at: new Date().toISOString(), nonce: randomBase64Url(32),
    signature_alg: 'ed25519', ...overrides,
  };
  const payload = [domain, 'POST', '/v1/puri/token', 'puripuly-agent-relay',
    request.installation_id, request.device_public_key, request.app_version, request.signed_at, request.nonce].join('\n');
  const signature = await crypto.subtle.sign('Ed25519', key.privateKey, encoder.encode(payload));
  return { ...request, signature: encodeBase64Url(new Uint8Array(signature)) };
}

function post(env: TestBrokerEnv, request: unknown, ip = '203.0.113.33') {
  return app.request('http://broker.test/v1/puri/token', {
    method: 'POST', headers: { 'content-type': 'application/json', 'cf-connecting-ip': ip },
    body: JSON.stringify(request),
  }, env);
}

function registeredInstallation(env: TestBrokerEnv, id: string, key: DeviceKeyPair) {
  env.__db.prepare(`INSERT INTO installations (installation_id, device_public_key, app_version, puri_bound_at)
    VALUES (?, ?, '1.2.3', '2026-10-29T12:00:00.000Z')`).run(id, key.devicePublicKey);
}

function installationIdAt(index: number) {
  return `019a3544-3380-7000-8000-${(4096 + index).toString(16).padStart(12, '0')}`;
}

function preEnroll(env: TestBrokerEnv, id: string, key: DeviceKeyPair, ip = '203.0.113.33') {
  return app.request('http://broker.test/v1/trial/challenge', {
    method: 'POST', headers: { 'content-type': 'application/json', 'cf-connecting-ip': ip },
    body: JSON.stringify({ installation_id: id, device_public_key: key.devicePublicKey, app_version: '1.2.3' }),
  }, env);
}

function enrollments(env: TestBrokerEnv) {
  return env.__db.prepare('SELECT COUNT(*) AS count FROM puri_enrollment_admissions').get();
}

async function networkSubject(env: TestBrokerEnv, ip = '203.0.113.33') {
  const now = new Date();
  const [identity] = await resolveRequestNetworkIdentityCandidates(ip, resolveNetworkIdentitySecrets(env), now, now);
  return identity!.digest;
}

function storedControls(env: TestBrokerEnv) {
  return readAbuseControls(env) as unknown as Record<string, Record<string, unknown>>;
}

describe('Puri installation token boundary', () => {
  beforeEach(() => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-10-30T12:00:00.123Z'));
    vi.stubGlobal('fetch', vi.fn(() => { throw new Error('Provider issuance must not run'); }));
  });
  afterEach(() => {
    for (const env of databases.splice(0)) env.__db.close();
    vi.useRealTimers();
    vi.unstubAllGlobals();
  });

  it('registers only the proven managed identity and mints the exact short-lived HS256 token', async () => {
    const env = environment();
    const key = await createDeviceKeyPair();
    const request = await proof(key);
    const response = await post(env, request);
    expect(response.status).toBe(200);
    expect(response.headers.get('cache-control')).toBe('no-store');
    const body = await response.json() as { access_token: string; token_type: string; expires_in: number; expires_at: string };
    expect(Object.keys(body).sort()).toEqual(['access_token', 'expires_at', 'expires_in', 'token_type']);
    expect(body).toMatchObject({ token_type: 'Bearer', expires_in: 3600, expires_at: '2026-10-30T13:00:00.000Z' });
    const [header, claims, signature] = body.access_token.split('.');
    expect(JSON.parse(Buffer.from(header, 'base64url').toString())).toEqual({ alg: 'HS256', typ: 'JWT' });
    expect(JSON.parse(Buffer.from(claims, 'base64url').toString())).toEqual({
      iss: 'puripuly-heart-broker', aud: 'puripuly-agent-relay',
      sub: createHmac('sha256', identitySecret).update(installationId).digest('hex'),
      iat: 1793361600, exp: 1793365200,
    });
    expect(signature).toBe(createHmac('sha256', tokenSecret).update(`${header}.${claims}`).digest('base64url'));
    expect(env.__db.prepare('SELECT * FROM installations').get()).toMatchObject({
      installation_id: installationId, device_public_key: key.devicePublicKey, app_version: '1.2.3',
      hardware_hash: null, hardware_hash_salt_version: null, challenge: null, challenge_expires_at: null,
      puri_bound_at: '2026-10-30T12:00:00.123Z',
    });
    for (const table of ['openrouter_entitlements', 'managed_operations', 'managed_key_deliveries', 'managed_operation_attempts']) {
      expect(env.__db.prepare(`SELECT COUNT(*) AS count FROM ${table}`).get()).toEqual({ count: 0 });
    }
    expect(fetch).not.toHaveBeenCalled();
  });

  it.each(['active', 'expired', 'revoked', 'pending_release'] as const)('does not gate Puri on %s managed entitlement or alter managed state', async (status) => {
    const env = environment();
    const key = await createDeviceKeyPair();
    env.__db.prepare(`INSERT INTO installations (installation_id, device_public_key, app_version,
      hardware_hash, hardware_hash_salt_version, challenge, challenge_expires_at, challenge_salt_version)
      VALUES (?, ?, 'old-version', 'hardware', 7, 'challenge', '2026-10-30T12:04:00.000Z', 7)`).run(installationId, key.devicePublicKey);
    insertEntitlement(env, { installation_id: installationId, status, budget_usd: 3, managed_credential_ref: 'existing-provider-key' });
    const installation = env.__db.prepare('SELECT * FROM installations').get();
    const entitlement = env.__db.prepare('SELECT * FROM openrouter_entitlements').get();
    expect((await post(env, await proof(key))).status).toBe(200);
    expect(env.__db.prepare('SELECT * FROM openrouter_entitlements').get()).toEqual(entitlement);
    expect(env.__db.prepare('SELECT * FROM installations').get()).toEqual({ ...installation, puri_bound_at: new Date().toISOString() });
    expect(fetch).not.toHaveBeenCalled();
  });

  it('rejects invalid and cross-domain proof before registration and rejects consumed nonces', async () => {
    const env = environment();
    const key = await createDeviceKeyPair();
    const request = await proof(key);
    expect((await post(env, { ...request, app_version: 'tampered' })).status).toBe(401);
    expect((await post(env, await proof(key, {}, 'trial-status'))).status).toBe(401);
    expect(env.__db.prepare('SELECT COUNT(*) AS count FROM installations').get()).toEqual({ count: 0 });
    expect((await post(env, request)).status).toBe(200);
    expect((await post(env, request)).status).toBe(401);
    expect((await post(env, await proof(key))).status).toBe(200);
    expect(env.__db.prepare('SELECT COUNT(*) AS count FROM puri_proof_nonces').get()).toEqual({ count: 2 });
  });

  it.each([-60001, 60001])('rejects signed_at outside the 60-second window (%s ms)', async (skew) => {
    const env = environment();
    const request = await proof(await createDeviceKeyPair(), { signed_at: new Date(Date.now() + skew).toISOString() });
    expect((await post(env, request)).status).toBe(401);
    expect(env.__db.prepare('SELECT COUNT(*) AS count FROM installations').get()).toEqual({ count: 0 });
  });

  it.each([-60000, 60000])('accepts signed_at at the clock window edge (%s ms)', async (skew) => {
    const env = environment();
    expect((await post(env, await proof(await createDeviceKeyPair(), { signed_at: new Date(Date.now() + skew).toISOString() }))).status).toBe(200);
  });

  it.each<Record<string, string>>([
    { installation_id: 'not-a-managed-uuid' }, { installation_id: '019a3544-3380-4000-8000-000000000330' },
    { app_version: 'a\nb' }, { app_version: 'a\u0000b' }, { app_version: 'a\ud800b' }, { app_version: 'x'.repeat(65) },
    { nonce: `${'A'.repeat(42)}B` }, { device_public_key: `${'A'.repeat(42)}B` },
    { signature_alg: 'HS256' }, { signed_at: '2026-10-30T12:00:00Z' },
    { signed_at: '2026-10-30T12:00:00.123+00:00' }, { signed_at: '2026-02-30T12:00:00.123Z' },
  ])('rejects malformed signed input %j', async (overrides) => {
    const env = environment();
    expect((await post(env, await proof(await createDeviceKeyPair(), overrides))).status).toBe(400);
    expect(env.__db.prepare('SELECT COUNT(*) AS count FROM installations').get()).toEqual({ count: 0 });
  });

  it('rejects extra fields, noncanonical signatures, invalid JSON and oversized bodies', async () => {
    const env = environment();
    const request = await proof(await createDeviceKeyPair());
    expect((await post(env, { ...request, scope: 'puri' })).status).toBe(400);
    expect((await post(env, { ...request, signature: request.signature.slice(0, -1) + 'B' })).status).toBe(400);
    expect((await post(env, { ...request, nonce: `${request.nonce}=` })).status).toBe(400);
    expect((await post(env, { ...request, app_version: 'x'.repeat(5000) })).status).toBe(413);
    expect((await app.request('http://broker.test/v1/puri/token', { method: 'POST', body: '{' }, env)).status).toBe(400);
  });

  it('preserves both unique binding directions and rejects registration races', async () => {
    let racingKey: string | null = null;
    const env = environment({ beforeRun: ({ sql }) => {
      if (sql.includes('INSERT INTO installations') && racingKey) {
        env.__db.prepare("INSERT INTO installations (installation_id, device_public_key, app_version) VALUES (?, ?, 'racer')").run(installationId, racingKey);
        racingKey = null;
      }
    } });
    const registered = await createDeviceKeyPair();
    const other = await createDeviceKeyPair();
    racingKey = registered.devicePublicKey;
    expect((await post(env, await proof(other))).status).toBe(409);
    expect((await post(env, await proof(registered, { installation_id: otherId }))).status).toBe(409);
    expect((await post(env, await proof(registered))).status).toBe(200);
    expect((await post(env, await proof(other))).status).toBe(409);
    expect(env.__db.prepare('SELECT installation_id, device_public_key FROM installations').all()).toEqual([
      { installation_id: installationId, device_public_key: registered.devicePublicKey },
    ]);
  });

  it('rolls back new identity and installation admissions when durable nonce consumption fails', async () => {
    let failNonce = true;
    const env = environment({ beforeRun: ({ sql }) => {
      if (sql.includes('INSERT INTO puri_proof_nonces') && failNonce) throw new Error('synthetic D1 failure');
    } });
    env.__db.prepare(`UPDATE broker_config SET value = json_set(value,
      '$.puriEnrollmentShortIp.maxRequests', 1, '$.puriTokenInstallation.maxRequests', 1)
      WHERE key = 'abuse_controls'`).run();
    const request = await proof(await createDeviceKeyPair());
    expect((await post(env, request)).status).toBe(500);
    expect(env.__db.prepare('SELECT COUNT(*) AS count FROM installations').get()).toEqual({ count: 0 });
    expect(env.__db.prepare('SELECT policy, request_count FROM puri_token_rate_limits').all()).toEqual([
      { policy: 'ip_flood', request_count: 1 },
    ]);
    expect(enrollments(env)).toEqual({ count: 0 });
    failNonce = false;
    expect((await post(env, request)).status).toBe(200);
    expect(enrollments(env)).toEqual({ count: 1 });
  });

  it('rolls back a first binding together with its enrollment charge and recovers on retry', async () => {
    let failCharge = true;
    const env = environment({ beforeRun: ({ sql }) => {
      if (sql.includes('INSERT INTO puri_enrollment_admissions') && failCharge) throw new Error('synthetic D1 failure');
    } });
    const key = await createDeviceKeyPair();
    expect((await preEnroll(env, installationId, key)).status).toBe(200);
    const request = await proof(key);
    expect((await post(env, request)).status).toBe(500);
    expect(env.__db.prepare('SELECT puri_bound_at FROM installations').get()).toEqual({ puri_bound_at: null });
    expect(env.__db.prepare('SELECT COUNT(*) AS count FROM puri_proof_nonces').get()).toEqual({ count: 0 });
    expect(env.__db.prepare("SELECT COUNT(*) AS count FROM puri_token_rate_limits WHERE policy = 'installation'").get())
      .toEqual({ count: 0 });
    expect(enrollments(env)).toEqual({ count: 0 });
    failCharge = false;
    expect((await post(env, request)).status).toBe(200);
    expect((await post(env, request)).status).toBe(401);
    expect((await post(env, await proof(key))).status).toBe(200);
    expect(env.__db.prepare('SELECT puri_bound_at FROM installations').get()).toEqual({ puri_bound_at: new Date().toISOString() });
    expect(enrollments(env)).toEqual({ count: 1 });
  });

  it('fails closed for unavailable configuration without disabling existing broker routes', async () => {
    const key = await createDeviceKeyPair();
    for (const secrets of [
      { PURI_TOKEN_SECRET: '', PURI_IDENTITY_HMAC_SECRET: identitySecret },
      { PURI_TOKEN_SECRET: tokenSecret, PURI_IDENTITY_HMAC_SECRET: '' },
      { PURI_TOKEN_SECRET: 'short', PURI_IDENTITY_HMAC_SECRET: identitySecret },
      { PURI_TOKEN_SECRET: tokenSecret, PURI_IDENTITY_HMAC_SECRET: tokenSecret },
      { PURI_TOKEN_SECRET: tokenSecret, PURI_IDENTITY_HMAC_SECRET: identitySecret, NETWORK_IDENTITY_HMAC_SECRET: '' },
      { PURI_TOKEN_SECRET: tokenSecret, PURI_IDENTITY_HMAC_SECRET: identitySecret, NETWORK_IDENTITY_HMAC_KEY_VERSION: 'bad' },
    ]) {
      const env = { ...environment(), ...secrets };
      expect((await post(env, await proof(key))).status).toBe(503);
      expect((await app.request('http://broker.test/healthz', {}, env)).status).toBe(200);
      expect(env.__db.prepare('SELECT COUNT(*) AS count FROM installations').get()).toEqual({ count: 0 });
    }
  });

  it('bounds unauthenticated attempts before proof or identity registration without writing raw request events', async () => {
    const env = environment();
    env.__db.prepare(`UPDATE broker_config SET value = json_set(value, '$.puriTokenIp.maxRequests', 2)
      WHERE key = 'abuse_controls'`).run();
    expect((await post(env, {})).status).toBe(400);
    expect((await post(env, {})).status).toBe(400);
    const rejected = await post(env, {});
    expect(rejected.status).toBe(429);
    expect(await rejected.json()).toMatchObject({
      error: { code: 'rate_limited', class: 'retryable', subcode: 'ip_rate_limited', retry_after_ms: 899877 },
    });
    expect(env.__db.prepare('SELECT policy, request_count FROM puri_token_rate_limits').all()).toEqual([
      { policy: 'ip_flood', request_count: 2 },
    ]);
    expect(env.__db.prepare('SELECT COUNT(*) AS count FROM broker_request_events').get()).toEqual({ count: 0 });
    expect(env.__db.prepare('SELECT COUNT(*) AS count FROM installations').get()).toEqual({ count: 0 });
  });

  it.each(['denylist', 'revocation'] as const)('enforces keyed IP %s hooks using the Cloudflare address without persisting raw IP', async (kind) => {
    const env = environment();
    const secrets = resolveNetworkIdentitySecrets(env)!;
    const [{ digest }] = await deriveStableNetworkIdentityDigest(secrets, '203.0.113.33', 'ip');
    env.__db.prepare(`INSERT INTO broker_abuse_subject_hooks
      (hook_kind, subject_type, subject_value, outcome_code, outcome_class, expires_at)
      VALUES (?, 'ip', ?, 'trial_not_eligible', 'security_fail', '2026-10-30T12:01:00.123Z')`).run(kind, digest);
    const response = await app.request('http://broker.test/v1/puri/token', {
      method: 'POST', headers: {
        'content-type': 'application/json', 'cf-connecting-ip': '203.0.113.33',
        'x-forwarded-for': '198.51.100.91',
      },
      body: JSON.stringify(await proof(await createDeviceKeyPair())),
    }, env);
    expect(response.status).toBe(409);
    expect(await response.json()).toMatchObject({
      error: { code: 'trial_not_eligible', class: 'security_fail', retry_after_ms: 60000 },
    });
    const counters = env.__db.prepare('SELECT * FROM puri_token_rate_limits').all();
    expect(counters).toEqual([expect.objectContaining({
      policy: 'ip_flood', subject_key: expect.stringMatching(/^[0-9a-f]{64}$/u),
      ip_key_version: 1, ip_epoch: '2026-10-30', request_count: 1,
    })]);
    expect(JSON.stringify(counters)).not.toContain('203.0.113.33');
    expect(JSON.stringify(counters)).not.toContain('198.51.100.91');
    expect(env.__db.prepare('SELECT COUNT(*) AS count FROM broker_request_events').get()).toEqual({ count: 0 });
    expect(env.__db.prepare('SELECT COUNT(*) AS count FROM installations').get()).toEqual({ count: 0 });
  });

  it('retains installation revocation hooks and their existing entitlement side effect', async () => {
    const env = environment();
    const key = await createDeviceKeyPair();
    registeredInstallation(env, installationId, key);
    insertEntitlement(env, { installation_id: installationId, status: 'active', budget_usd: 3 });
    env.__db.prepare(`INSERT INTO broker_abuse_subject_hooks
      (hook_kind, subject_type, subject_value, outcome_code, outcome_class)
      VALUES ('revocation', 'installation_id', ?, 'trial_not_eligible', 'security_fail')`).run(installationId);
    expect((await post(env, await proof(key))).status).toBe(409);
    expect(env.__db.prepare('SELECT status FROM openrouter_entitlements').get()).toEqual({ status: 'revoked' });
    expect(env.__db.prepare('SELECT COUNT(*) AS count FROM puri_proof_nonces').get()).toEqual({ count: 0 });
  });

  it('lets dozens of established installations renew behind a NAT with exhausted first-enrollment allowance', async () => {
    const env = environment();
    env.__db.prepare(`UPDATE broker_config SET value = json_set(value, '$.puriEnrollmentShortIp.maxRequests', 1)
      WHERE key = 'abuse_controls'`).run();
    expect((await post(env, await proof(await createDeviceKeyPair()))).status).toBe(200);
    const waitingKey = await createDeviceKeyPair();
    const rejected = await post(env, await proof(waitingKey, { installation_id: otherId }));
    expect(rejected.status).toBe(429);
    expect(await rejected.json()).toMatchObject({
      error: { code: 'rate_limited', class: 'retryable', subcode: 'ip_rate_limited', retry_after_ms: 3600000 },
    });
    for (let index = 0; index < 40; index += 1) {
      const key = await createDeviceKeyPair();
      registeredInstallation(env, installationIdAt(index), key);
      expect((await post(env, await proof(key, { installation_id: installationIdAt(index) }))).status).toBe(200);
    }
    expect(enrollments(env)).toEqual({ count: 1 });
    expect(env.__db.prepare("SELECT request_count FROM puri_token_rate_limits WHERE policy = 'ip_flood'").get())
      .toEqual({ request_count: 42 });
    expect(env.__db.prepare('SELECT COUNT(*) AS count FROM installations').get()).toEqual({ count: 41 });
    vi.setSystemTime(new Date('2026-10-30T13:00:00.122Z'));
    const atBoundary = await post(env, await proof(waitingKey, { installation_id: otherId }));
    expect(atBoundary.status).toBe(429);
    expect(await atBoundary.json()).toMatchObject({ error: { retry_after_ms: 1 } });
    vi.setSystemTime(new Date('2026-10-30T13:00:00.123Z'));
    expect((await post(env, await proof(waitingKey, { installation_id: otherId }))).status).toBe(200);
  });

  it('charges the first Puri binding of trial pre-enrolled installations and keeps unrelated admissions usable', async () => {
    const env = environment();
    const keys = await Promise.all(Array.from({ length: 7 }, () => createDeviceKeyPair()));
    for (let index = 0; index < 6; index += 1) {
      expect((await preEnroll(env, installationIdAt(index), keys[index])).status).toBe(200);
    }
    expect(env.__db.prepare('SELECT COUNT(*) AS count FROM installations WHERE puri_bound_at IS NULL').get())
      .toEqual({ count: 6 });
    const forged = await proof(keys[0], { installation_id: installationIdAt(0) });
    expect((await post(env, { ...forged, app_version: 'tampered' })).status).toBe(401);
    for (let index = 0; index < 5; index += 1) {
      expect((await post(env, await proof(keys[index], { installation_id: installationIdAt(index) }))).status).toBe(200);
    }
    for (const index of [5, 6]) {
      const rejected = await post(env, await proof(keys[index], { installation_id: installationIdAt(index) }));
      expect(rejected.status).toBe(429);
      expect(await rejected.json()).toMatchObject({
        error: { code: 'rate_limited', class: 'retryable', subcode: 'ip_rate_limited', retry_after_ms: 3600000 },
      });
    }
    expect(env.__db.prepare('SELECT puri_bound_at FROM installations WHERE installation_id = ?').get(installationIdAt(5)))
      .toEqual({ puri_bound_at: null });
    expect(env.__db.prepare('SELECT COUNT(*) AS count FROM installations WHERE installation_id = ?').get(installationIdAt(6)))
      .toEqual({ count: 0 });
    expect(env.__db.prepare("SELECT COUNT(*) AS count FROM puri_token_rate_limits WHERE policy = 'installation' AND subject_key IN (?, ?)")
      .get(installationIdAt(5), installationIdAt(6))).toEqual({ count: 0 });
    expect(enrollments(env)).toEqual({ count: 5 });
    expect((await post(env, await proof(keys[0], { installation_id: installationIdAt(0) }))).status).toBe(200);
    expect((await post(env, await proof(keys[6], { installation_id: installationIdAt(6) }), '198.51.100.7')).status).toBe(200);
    expect(enrollments(env)).toEqual({ count: 6 });
  });

  it('enforces both rolling first-enrollment horizons across midnight without boundary resets', async () => {
    vi.setSystemTime(new Date('2026-10-30T23:30:00.000Z'));
    const env = environment();
    let next = 0;
    const enroll = async () => post(env, await proof(await createDeviceKeyPair(), { installation_id: installationIdAt(next++) }));
    const expectRejected = async (retryAfterMs: number) => {
      const response = await enroll();
      expect(response.status).toBe(429);
      expect(await response.json()).toMatchObject({
        error: { code: 'rate_limited', class: 'retryable', retry_after_ms: retryAfterMs },
      });
    };
    for (let index = 0; index < 5; index += 1) expect((await enroll()).status).toBe(200);
    await expectRejected(3_600_000);
    vi.setSystemTime(new Date('2026-10-31T00:00:00.000Z'));
    await expectRejected(1_800_000);
    vi.setSystemTime(new Date('2026-10-31T00:29:59.999Z'));
    await expectRejected(1);
    vi.setSystemTime(new Date('2026-10-31T00:30:00.000Z'));
    for (let index = 0; index < 5; index += 1) expect((await enroll()).status).toBe(200);
    vi.setSystemTime(new Date('2026-10-31T01:30:00.000Z'));
    await expectRejected(22 * 3_600_000);
    vi.setSystemTime(new Date('2026-10-31T23:29:59.999Z'));
    await expectRejected(1);
    vi.setSystemTime(new Date('2026-10-31T23:30:00.000Z'));
    expect((await enroll()).status).toBe(200);
    expect(env.__db.prepare('SELECT ip_epoch, COUNT(*) AS count FROM puri_enrollment_admissions GROUP BY ip_epoch ORDER BY ip_epoch').all())
      .toEqual([{ ip_epoch: '2026-10-30', count: 5 }, { ip_epoch: '2026-10-31', count: 6 }]);
    expect(env.__db.prepare('SELECT COUNT(*) AS count FROM installations').get()).toEqual({ count: 11 });
  });

  it('keeps first-enrollment accounting across a configured network HMAC key rotation', async () => {
    const env = environment();
    env.__db.prepare(`UPDATE broker_config SET value = json_set(value, '$.puriEnrollmentShortIp.maxRequests', 1)
      WHERE key = 'abuse_controls'`).run();
    const established = await createDeviceKeyPair();
    expect((await post(env, await proof(established))).status).toBe(200);
    env.NETWORK_IDENTITY_HMAC_SECRET_PREVIOUS = env.NETWORK_IDENTITY_HMAC_SECRET;
    env.NETWORK_IDENTITY_HMAC_KEY_VERSION_PREVIOUS = '1';
    env.NETWORK_IDENTITY_HMAC_SECRET = 'new-network-identity-hmac-secret';
    env.NETWORK_IDENTITY_HMAC_KEY_VERSION = '2';
    const rejected = await post(env, await proof(await createDeviceKeyPair(), { installation_id: otherId }));
    expect(rejected.status).toBe(429);
    expect(await rejected.json()).toMatchObject({ error: { code: 'rate_limited', retry_after_ms: 3600000 } });
    expect((await post(env, await proof(established))).status).toBe(200);
    vi.setSystemTime(new Date('2026-10-30T13:00:00.123Z'));
    expect((await post(env, await proof(await createDeviceKeyPair(), { installation_id: otherId }))).status).toBe(200);
    expect(env.__db.prepare('SELECT ip_key_version FROM puri_enrollment_admissions ORDER BY admission_id').all())
      .toEqual([{ ip_key_version: 1 }, { ip_key_version: 2 }]);
  });

  it('does not give forged or mismatched installation identities the established-installation allowance', async () => {
    const env = environment();
    env.__db.prepare(`UPDATE broker_config SET value = json_set(value, '$.puriEnrollmentShortIp.maxRequests', 1)
      WHERE key = 'abuse_controls'`).run();
    const registered = await createDeviceKeyPair();
    const attacker = await createDeviceKeyPair();
    registeredInstallation(env, installationId, registered);
    expect((await post(env, await proof(attacker, { installation_id: otherId }))).status).toBe(200);
    expect((await post(env, await proof(attacker, { device_public_key: registered.devicePublicKey }))).status).toBe(401);
    expect((await post(env, await proof(attacker))).status).toBe(409);
    expect((await post(env, await proof(registered, {
      installation_id: '019a3544-3380-7000-8000-000000000332',
    }))).status).toBe(409);
    expect(env.__db.prepare("SELECT COUNT(*) AS count FROM puri_token_rate_limits WHERE policy = 'installation' AND subject_key = ?")
      .get(installationId)).toEqual({ count: 0 });
    expect(enrollments(env)).toEqual({ count: 1 });
    expect((await post(env, await proof(registered))).status).toBe(200);
    expect(enrollments(env)).toEqual({ count: 1 });
  });

  it('caps verified installation issuance across network changes and reopens only at the next window', async () => {
    const env = environment();
    const key = await createDeviceKeyPair();
    for (let index = 0; index < 8; index += 1) {
      expect((await post(env, await proof(key), `203.0.113.${index + 1}`)).status).toBe(200);
    }
    const request = await proof(key);
    const rejected = await post(env, request, '198.51.100.91');
    expect(rejected.status).toBe(429);
    expect(await rejected.json()).toMatchObject({
      error: { code: 'rate_limited', class: 'retryable', subcode: 'installation_rate_limited', retry_after_ms: 899877 },
    });
    expect(env.__db.prepare("SELECT request_count FROM puri_token_rate_limits WHERE policy = 'installation'").get())
      .toEqual({ request_count: 8 });
    expect(env.__db.prepare('SELECT COUNT(*) AS count FROM puri_proof_nonces WHERE nonce = ?').get(request.nonce))
      .toEqual({ count: 0 });
    vi.setSystemTime(new Date('2026-10-30T12:14:59.999Z'));
    const atBoundary = await post(env, await proof(key));
    expect(atBoundary.status).toBe(429);
    expect(await atBoundary.json()).toMatchObject({ error: { retry_after_ms: 1 } });
    vi.setSystemTime(new Date('2026-10-30T12:15:00.000Z'));
    expect((await post(env, await proof(key))).status).toBe(200);
  });

  it('honors another isolate winning the final installation admission before its atomic update', async () => {
    let race = true;
    const env = environment({ beforeRun: ({ sql, params }) => {
      if (race && sql.includes('INSERT INTO puri_token_rate_limits') && params[0] === 'installation') {
        race = false;
        env.__db.prepare(`INSERT INTO puri_token_rate_limits
          (policy, subject_key, window_started_at, expires_at, request_count, ip_key_version, ip_epoch)
          VALUES (?, ?, ?, ?, 1, ?, ?)`).run(...params.slice(0, 6));
      }
    } });
    env.__db.prepare(`UPDATE broker_config SET value = json_set(value, '$.puriTokenInstallation.maxRequests', 1)
      WHERE key = 'abuse_controls'`).run();
    const key = await createDeviceKeyPair();
    registeredInstallation(env, installationId, key);
    const response = await post(env, await proof(key));
    expect(response.status).toBe(429);
    expect(await response.json()).toMatchObject({ error: { subcode: 'installation_rate_limited' } });
    expect(env.__db.prepare("SELECT request_count FROM puri_token_rate_limits WHERE policy = 'installation'").get())
      .toEqual({ request_count: 1 });
    expect(env.__db.prepare('SELECT COUNT(*) AS count FROM puri_proof_nonces').get()).toEqual({ count: 0 });
  });

  it.each(['another installation', 'the same installation'] as const)(
    'charges one first enrollment when %s wins the final allowance before the atomic update', async (winner) => {
      let race = false;
      const env = environment({ beforeRun: ({ sql }) => {
        if (race && sql.includes('INSERT INTO installations')) {
          race = false;
          if (winner === 'the same installation') {
            env.__db.prepare('UPDATE installations SET puri_bound_at = ? WHERE installation_id = ?')
              .run(new Date().toISOString(), installationId);
          }
          env.__db.prepare(`INSERT INTO puri_enrollment_admissions (subject_key, admitted_at, ip_key_version, ip_epoch)
            VALUES (?, ?, 1, '2026-10-30')`).run(subject, new Date().toISOString());
        }
      } });
      env.__db.prepare(`UPDATE broker_config SET value = json_set(value, '$.puriEnrollmentShortIp.maxRequests', 1)
        WHERE key = 'abuse_controls'`).run();
      const subject = await networkSubject(env);
      const key = await createDeviceKeyPair();
      expect((await preEnroll(env, installationId, key)).status).toBe(200);
      race = true;
      const response = await post(env, await proof(key));
      if (winner === 'the same installation') {
        expect(response.status).toBe(200);
      } else {
        expect(response.status).toBe(429);
        expect(await response.json()).toMatchObject({
          error: { code: 'rate_limited', subcode: 'ip_rate_limited', retry_after_ms: 3600000 },
        });
        expect(env.__db.prepare('SELECT puri_bound_at FROM installations').get()).toEqual({ puri_bound_at: null });
        expect(env.__db.prepare('SELECT COUNT(*) AS count FROM puri_proof_nonces').get()).toEqual({ count: 0 });
      }
      expect(enrollments(env)).toEqual({ count: 1 });
    });

  it('keeps IP flood limits and keyed subject hooks effective across configured HMAC key rotation', async () => {
    const env = environment();
    env.__db.prepare(`UPDATE broker_config SET value = json_set(value, '$.puriTokenIp.maxRequests', 1)
      WHERE key = 'abuse_controls'`).run();
    const [{ digest }] = await deriveStableNetworkIdentityDigest(
      resolveNetworkIdentitySecrets(env)!, '198.51.100.91', 'ip',
    );
    const key = await createDeviceKeyPair();
    expect((await post(env, await proof(key))).status).toBe(200);
    env.NETWORK_IDENTITY_HMAC_SECRET_PREVIOUS = env.NETWORK_IDENTITY_HMAC_SECRET;
    env.NETWORK_IDENTITY_HMAC_KEY_VERSION_PREVIOUS = '1';
    env.NETWORK_IDENTITY_HMAC_SECRET = 'new-network-identity-hmac-secret';
    env.NETWORK_IDENTITY_HMAC_KEY_VERSION = '2';
    expect((await post(env, await proof(key))).status).toBe(429);
    env.__db.prepare(`INSERT INTO broker_abuse_subject_hooks
      (hook_kind, subject_type, subject_value, outcome_code, outcome_class)
      VALUES ('denylist', 'ip', ?, 'trial_not_eligible', 'security_fail')`).run(digest);
    expect((await post(env, await proof(key), '198.51.100.91')).status).toBe(409);
  });

  it('uses a bounded unattributed bucket instead of bypassing IP limits when a trusted IP is absent', async () => {
    const env = environment();
    env.__db.prepare(`UPDATE broker_config SET value = json_set(value, '$.puriTokenIp.maxRequests', 1)
      WHERE key = 'abuse_controls'`).run();
    const request = () => app.request('http://broker.test/v1/puri/token', {
      method: 'POST', headers: { 'content-type': 'application/json' }, body: '{}',
    }, env);
    expect((await request()).status).toBe(400);
    expect((await request()).status).toBe(429);
    expect(env.__db.prepare('SELECT subject_key, request_count FROM puri_token_rate_limits').get())
      .toEqual({ subject_key: 'unattributed', request_count: 1 });
  });

  it('shares one bounded unattributed first-enrollment allowance when a trusted IP is absent', async () => {
    const env = environment();
    env.__db.prepare(`UPDATE broker_config SET value = json_set(value, '$.puriEnrollmentShortIp.maxRequests', 1)
      WHERE key = 'abuse_controls'`).run();
    const enroll = async (id: string) => app.request('http://broker.test/v1/puri/token', {
      method: 'POST', headers: { 'content-type': 'application/json' },
      body: JSON.stringify(await proof(await createDeviceKeyPair(), { installation_id: id })),
    }, env);
    expect((await enroll(installationId)).status).toBe(200);
    expect((await enroll(otherId)).status).toBe(429);
    expect(env.__db.prepare('SELECT subject_key, ip_key_version, ip_epoch FROM puri_enrollment_admissions').all())
      .toEqual([{ subject_key: 'unattributed', ip_key_version: null, ip_epoch: null }]);
  });

  it('does not apply Puri admission limits to managed challenge requests sharing the same IP', async () => {
    const env = environment();
    env.__db.prepare(`UPDATE broker_config SET value = json_set(value,
      '$.puriTokenIp.maxRequests', 1, '$.puriEnrollmentShortIp.maxRequests', 1,
      '$.puriEnrollmentLongIp.maxRequests', 1, '$.puriTokenInstallation.maxRequests', 1) WHERE key = 'abuse_controls'`).run();
    expect((await post(env, {})).status).toBe(400);
    expect((await post(env, {})).status).toBe(429);
    for (let index = 0; index <= 10; index += 1) {
      const response = await preEnroll(env, installationIdAt(index), await createDeviceKeyPair());
      expect(response.status).toBe(index < 10 ? 200 : 429);
    }
  });

  it('migrates only Puri defaults and preserves explicit operator and unrelated route limits', async () => {
    for (const override of [null, 47]) {
      const env = environment({ throughMigration: '0028_add_puri_identity_proofs.sql' });
      if (override !== null) {
        env.__db.prepare(`UPDATE broker_config SET value = json_set(value,
          '$.puriTokenIp.maxRequests', ?, '$.trialChallenge.maxRequests', 17) WHERE key = 'abuse_controls'`)
          .run(override);
      }
      const { puriTokenIp: beforeIp, ...beforeOtherLimits } = storedControls(env);
      env.__db.exec(readBrokerMigrationSql('0029_puri_token_admission_limits.sql'));
      const {
        puriTokenIp: afterIp, puriTokenRegistrationIp, puriTokenInstallation, ...afterOtherLimits
      } = storedControls(env);
      expect(beforeIp.maxRequests).toBe(override ?? 30);
      expect(afterIp.maxRequests).toBe(override ?? 600);
      expect(puriTokenRegistrationIp).toEqual({
        endpoint: 'POST /v1/puri/token', scope: 'ip', maxRequests: 20, windowMinutes: 15,
      });
      expect(puriTokenInstallation).toEqual(DEFAULT_BROKER_ABUSE_CONTROLS.puriTokenInstallation);
      expect(afterOtherLimits).toEqual(beforeOtherLimits);
    }
  });

  it('replaces the row-based registration knob with first-enrollment limits while preserving bindings and tuned limits', async () => {
    const operatorShort = { endpoint: 'POST /v1/puri/token', scope: 'ip', maxRequests: 3, windowMinutes: 30 };
    for (const [preset, operatorFlood] of [[null, null], [operatorShort, 47]] as const) {
      const env = environment({ throughMigration: '0029_puri_token_admission_limits.sql' });
      env.__db.prepare(`UPDATE broker_config SET value = json_set(value,
        '$.puriTokenRegistrationIp.maxRequests', 33, '$.puriTokenInstallation.maxRequests', 9,
        '$.trialChallenge.maxRequests', 17) WHERE key = 'abuse_controls'`).run();
      if (preset) {
        env.__db.prepare(`UPDATE broker_config SET value = json_set(value, '$.puriEnrollmentShortIp', json(?),
          '$.puriTokenIp.maxRequests', ?) WHERE key = 'abuse_controls'`).run(JSON.stringify(preset), operatorFlood);
      }
      const established = await createDeviceKeyPair();
      const pending = await createDeviceKeyPair();
      registeredInstallation(env, installationId, established);
      expect((await preEnroll(env, otherId, pending)).status).toBe(200);
      env.__db.prepare(`INSERT INTO puri_proof_nonces (installation_id, nonce, expires_at)
        VALUES (?, 'retained-nonce', '2026-10-30T12:05:00.000Z')`).run(installationId);
      env.__db.prepare(`INSERT INTO puri_token_rate_limits
        (policy, subject_key, window_started_at, expires_at, request_count, ip_key_version, ip_epoch)
        VALUES ('ip_registration', 'retained-subject', '2026-10-30T12:00:00.000Z', '2026-10-30T12:15:00.000Z', 2, 1, '2026-10-30')`).run();
      const tables = ['installations', 'puri_proof_nonces', 'puri_token_rate_limits']
        .map((table) => env.__db.prepare(`SELECT * FROM ${table} ORDER BY 1, 2`).all());
      const {
        puriTokenRegistrationIp, puriEnrollmentShortIp: presetBefore, puriTokenIp: floodBefore, ...beforeOtherLimits
      } = storedControls(env);
      expect(puriTokenRegistrationIp.maxRequests).toBe(33);
      expect(presetBefore).toEqual(preset ?? undefined);
      expect(floodBefore.maxRequests).toBe(operatorFlood ?? 600);
      env.__db.exec(readBrokerMigrationSql('0030_puri_first_enrollment_limits.sql'));
      const { puriEnrollmentShortIp, puriEnrollmentLongIp, puriTokenIp, ...afterOtherLimits } = storedControls(env);
      expect(afterOtherLimits).toEqual(beforeOtherLimits);
      expect(puriTokenIp).toEqual(operatorFlood === null
        ? DEFAULT_BROKER_ABUSE_CONTROLS.puriTokenIp
        : { ...DEFAULT_BROKER_ABUSE_CONTROLS.puriTokenIp, maxRequests: operatorFlood });
      expect(puriEnrollmentShortIp).toEqual(preset ?? DEFAULT_BROKER_ABUSE_CONTROLS.puriEnrollmentShortIp);
      expect(puriEnrollmentLongIp).toEqual(DEFAULT_BROKER_ABUSE_CONTROLS.puriEnrollmentLongIp);
      expect(await getBrokerAbuseControlsConfig(env.BROKER_DB)).toMatchObject({
        puriTokenIp: { maxRequests: operatorFlood ?? 200 }, puriTokenInstallation: { maxRequests: 9 },
        trialChallenge: { maxRequests: 17 },
        puriEnrollmentShortIp: preset ?? DEFAULT_BROKER_ABUSE_CONTROLS.puriEnrollmentShortIp,
      });
      expect(['installations', 'puri_proof_nonces', 'puri_token_rate_limits']
        .map((table) => env.__db.prepare(`SELECT * FROM ${table} ORDER BY 1, 2`).all())).toEqual(tables);
      expect((await post(env, await proof(established))).status).toBe(200);
      expect(enrollments(env)).toEqual({ count: 0 });
      expect((await post(env, await proof(pending, { installation_id: otherId }))).status).toBe(200);
      expect(enrollments(env)).toEqual({ count: 1 });
    }
  });

  it.each([
    ['puriTokenIp', 'maxRequests', 0],
    ['puriEnrollmentShortIp', 'maxRequests', 1.5],
    ['puriTokenInstallation', 'maxRequests', Number.MAX_SAFE_INTEGER + 1],
    ['puriTokenIp', 'windowMinutes', 1441],
    ['puriEnrollmentLongIp', 'windowMinutes', 1441],
    ['puriEnrollmentShortIp', 'windowMinutes', -1],
    ['puriTokenInstallation', 'scope', 'ip'],
    ['puriEnrollmentLongIp', 'endpoint', 'POST /v1/trial/challenge'],
  ])('validates the %s %s knob before using configured admission policies', async (key, field, value) => {
    const env = environment();
    env.__db.prepare(`UPDATE broker_config SET value = json_set(value, ?, json(?)) WHERE key = 'abuse_controls'`)
      .run(`$.${key}.${field}`, JSON.stringify(value));
    expect(await getBrokerAbuseControlsConfig(env.BROKER_DB)).toEqual(DEFAULT_BROKER_ABUSE_CONTROLS);
  });

  it('expires indexed counters and enrollment history without extending their windows or keeping raw IP data', async () => {
    const env = environment();
    const subject = await networkSubject(env);
    const key = await createDeviceKeyPair();
    expect((await post(env, await proof(key))).status).toBe(200);
    vi.setSystemTime(new Date('2026-10-30T12:14:00.000Z'));
    expect((await post(env, await proof(key))).status).toBe(200);
    const rows = env.__db.prepare('SELECT * FROM puri_token_rate_limits ORDER BY policy').all();
    expect(rows).toHaveLength(2);
    expect(rows.every((row) => row.expires_at === '2026-10-30T12:15:00.000Z')).toBe(true);
    const history = env.__db.prepare('SELECT * FROM puri_enrollment_admissions').all();
    expect(history).toEqual([{
      admission_id: 1, subject_key: subject, admitted_at: '2026-10-30T12:00:00.123Z', ip_key_version: 1, ip_epoch: '2026-10-30',
    }]);
    expect(JSON.stringify([rows, history])).not.toContain('203.0.113.33');
    await deleteExpiredPuriProofNonces(env.BROKER_DB, new Date('2026-10-30T12:14:59.999Z'));
    expect(env.__db.prepare('SELECT COUNT(*) AS count FROM puri_token_rate_limits').get()).toEqual({ count: 2 });
    await deleteExpiredPuriProofNonces(env.BROKER_DB, new Date('2026-10-30T12:15:00.000Z'));
    expect(env.__db.prepare('SELECT COUNT(*) AS count FROM puri_token_rate_limits').get()).toEqual({ count: 0 });
    await deleteExpiredPuriProofNonces(env.BROKER_DB, new Date('2026-10-31T12:00:00.122Z'));
    expect(enrollments(env)).toEqual({ count: 1 });
    await deleteExpiredPuriProofNonces(env.BROKER_DB, new Date('2026-10-31T12:00:00.123Z'));
    expect(enrollments(env)).toEqual({ count: 0 });
    expect(env.__db.prepare('SELECT installation_id, puri_bound_at FROM installations').get())
      .toEqual({ installation_id: installationId, puri_bound_at: '2026-10-30T12:00:00.123Z' });
  });

  it('retains established Puri binding in request-time and scheduled preflight cleanup, and expires only old nonces', async () => {
    const env = environment();
    env.__db.prepare(`UPDATE broker_config SET value = json_set(value, '$.dailyReport.enabled', json('false')) WHERE key = 'abuse_controls'`).run();
    const key = await createDeviceKeyPair();
    expect((await post(env, await proof(key))).status).toBe(200);
    env.__db.prepare(`UPDATE installations SET challenge = 'abandoned', challenge_expires_at = '2026-10-20T12:00:00.000Z',
      challenge_salt_version = 7, last_seen_at = '2026-10-20T12:00:00.000Z'`).run();
    const replacement = await createDeviceKeyPair();
    const challenge = await app.request('http://broker.test/v1/trial/challenge', {
      method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({
        installation_id: installationId, device_public_key: replacement.devicePublicKey, app_version: '1.2.3',
      }),
    }, env);
    expect(challenge.status).toBe(409);
    env.__db.prepare(`INSERT INTO installations (installation_id, device_public_key, app_version, challenge, challenge_expires_at,
      challenge_salt_version, last_seen_at) VALUES (?, ?, '1.2.3', 'abandoned', '2026-10-20T12:00:00.000Z', 7, '2026-10-20T12:00:00.000Z')`).run(otherId, replacement.devicePublicKey);
    await deleteExpiredPuriProofNonces(env.BROKER_DB, new Date(Date.now() + 299_999));
    expect(env.__db.prepare('SELECT COUNT(*) AS count FROM puri_proof_nonces').get()).toEqual({ count: 1 });
    await app.scheduled({ scheduledTime: Date.now() + 300_000 }, env, {});
    expect(env.__db.prepare('SELECT installation_id, device_public_key FROM installations').all()).toEqual([
      { installation_id: installationId, device_public_key: key.devicePublicKey },
    ]);
    expect(env.__db.prepare('SELECT COUNT(*) AS count FROM puri_proof_nonces').get()).toEqual({ count: 0 });
    expect(fetch).not.toHaveBeenCalled();
  });
});
