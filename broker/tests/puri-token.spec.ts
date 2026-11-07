import { createHmac } from 'node:crypto';

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import app from '../src/index';
import { getBrokerAbuseControlsConfig } from '../src/abuse-controls';
import { encodeBase64Url, randomBase64Url } from '../src/crypto';
import {
  deriveStableNetworkIdentityDigest,
  resolveNetworkIdentitySecrets,
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
  env.__db.prepare(`INSERT INTO installations (installation_id, device_public_key, app_version)
    VALUES (?, ?, '1.2.3')`).run(id, key.devicePublicKey);
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
      '$.puriTokenRegistrationIp.maxRequests', 1, '$.puriTokenInstallation.maxRequests', 1)
      WHERE key = 'abuse_controls'`).run();
    const request = await proof(await createDeviceKeyPair());
    expect((await post(env, request)).status).toBe(500);
    expect(env.__db.prepare('SELECT COUNT(*) AS count FROM installations').get()).toEqual({ count: 0 });
    expect(env.__db.prepare('SELECT policy, request_count FROM puri_token_rate_limits').all()).toEqual([
      { policy: 'ip_flood', request_count: 1 },
    ]);
    failNonce = false;
    expect((await post(env, request)).status).toBe(200);
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

  it('lets dozens of proven existing installations refresh behind a NAT with exhausted new-registration allowance', async () => {
    const env = environment();
    env.__db.prepare(`UPDATE broker_config SET value = json_set(value, '$.puriTokenRegistrationIp.maxRequests', 1)
      WHERE key = 'abuse_controls'`).run();
    expect((await post(env, await proof(await createDeviceKeyPair()))).status).toBe(200);
    const waitingKey = await createDeviceKeyPair();
    const rejected = await post(env, await proof(waitingKey, { installation_id: otherId }));
    expect(rejected.status).toBe(429);
    expect(await rejected.json()).toMatchObject({
      error: { code: 'rate_limited', subcode: 'ip_rate_limited', retry_after_ms: 899877 },
    });
    for (let index = 0; index < 40; index += 1) {
      const id = `019a3544-3380-7000-8000-${(4096 + index).toString(16).padStart(12, '0')}`;
      const key = await createDeviceKeyPair();
      registeredInstallation(env, id, key);
      expect((await post(env, await proof(key, { installation_id: id }))).status).toBe(200);
    }
    expect(env.__db.prepare("SELECT request_count FROM puri_token_rate_limits WHERE policy = 'ip_registration'").get())
      .toEqual({ request_count: 1 });
    expect(env.__db.prepare("SELECT request_count FROM puri_token_rate_limits WHERE policy = 'ip_flood'").get())
      .toEqual({ request_count: 42 });
    expect(env.__db.prepare('SELECT COUNT(*) AS count FROM installations').get()).toEqual({ count: 41 });
    vi.setSystemTime(new Date('2026-10-30T12:15:00.000Z'));
    expect((await post(env, await proof(waitingKey, { installation_id: otherId }))).status).toBe(200);
  });

  it('does not give forged or mismatched installation identities the registered-installation allowance', async () => {
    const env = environment();
    env.__db.prepare(`UPDATE broker_config SET value = json_set(value, '$.puriTokenRegistrationIp.maxRequests', 1)
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
    expect(env.__db.prepare("SELECT request_count FROM puri_token_rate_limits WHERE policy = 'ip_registration'").get())
      .toEqual({ request_count: 1 });
    expect((await post(env, await proof(registered))).status).toBe(200);
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

  it.each(['ip_registration', 'installation'] as const)('honors another isolate winning the final %s admission before its atomic update', async (policy) => {
    let race = true;
    const env = environment({ beforeRun: ({ sql, params }) => {
      if (race && sql.includes('INSERT INTO puri_token_rate_limits') && params[0] === policy) {
        race = false;
        env.__db.prepare(`INSERT INTO puri_token_rate_limits
          (policy, subject_key, window_started_at, expires_at, request_count, ip_key_version, ip_epoch)
          VALUES (?, ?, ?, ?, 1, ?, ?)`).run(...params.slice(0, 6));
      }
    } });
    env.__db.prepare(`UPDATE broker_config SET value = json_set(value,
      '$.puriTokenRegistrationIp.maxRequests', 1, '$.puriTokenInstallation.maxRequests', 1)
      WHERE key = 'abuse_controls'`).run();
    const key = await createDeviceKeyPair();
    if (policy === 'installation') registeredInstallation(env, installationId, key);
    const response = await post(env, await proof(key));
    expect(response.status).toBe(429);
    expect(await response.json()).toMatchObject({
      error: { subcode: policy === 'installation' ? 'installation_rate_limited' : 'ip_rate_limited' },
    });
    expect(env.__db.prepare('SELECT request_count FROM puri_token_rate_limits WHERE policy = ?').get(policy))
      .toEqual({ request_count: 1 });
    expect(env.__db.prepare('SELECT COUNT(*) AS count FROM puri_proof_nonces').get()).toEqual({ count: 0 });
    expect(env.__db.prepare('SELECT COUNT(*) AS count FROM installations').get())
      .toEqual({ count: policy === 'installation' ? 1 : 0 });
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

  it('does not apply Puri admission limits to managed challenge requests sharing the same IP', async () => {
    const env = environment();
    env.__db.prepare(`UPDATE broker_config SET value = json_set(value,
      '$.puriTokenIp.maxRequests', 1, '$.puriTokenRegistrationIp.maxRequests', 1,
      '$.puriTokenInstallation.maxRequests', 1) WHERE key = 'abuse_controls'`).run();
    expect((await post(env, {})).status).toBe(400);
    expect((await post(env, {})).status).toBe(429);
    for (let index = 0; index <= 10; index += 1) {
      const key = await createDeviceKeyPair();
      const response = await app.request('http://broker.test/v1/trial/challenge', {
        method: 'POST', headers: { 'content-type': 'application/json', 'cf-connecting-ip': '203.0.113.33' },
        body: JSON.stringify({
          installation_id: `019a3544-3380-7000-8000-${(4096 + index).toString(16).padStart(12, '0')}`,
          device_public_key: key.devicePublicKey, app_version: '1.2.3',
        }),
      }, env);
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
      const { puriTokenIp: beforeIp, ...beforeOtherLimits } = readAbuseControls(env);
      env.__db.exec(readBrokerMigrationSql('0029_puri_token_admission_limits.sql'));
      const {
        puriTokenIp: afterIp, puriTokenRegistrationIp, puriTokenInstallation, ...afterOtherLimits
      } = readAbuseControls(env);
      expect(beforeIp.maxRequests).toBe(override ?? 30);
      expect(afterIp.maxRequests).toBe(override ?? 600);
      expect(puriTokenRegistrationIp).toEqual(DEFAULT_BROKER_ABUSE_CONTROLS.puriTokenRegistrationIp);
      expect(puriTokenInstallation).toEqual(DEFAULT_BROKER_ABUSE_CONTROLS.puriTokenInstallation);
      expect(afterOtherLimits).toEqual(beforeOtherLimits);
    }
  });

  it.each([
    ['puriTokenIp', 'maxRequests', 0],
    ['puriTokenRegistrationIp', 'maxRequests', 1.5],
    ['puriTokenInstallation', 'maxRequests', Number.MAX_SAFE_INTEGER + 1],
    ['puriTokenIp', 'windowMinutes', 1441],
    ['puriTokenRegistrationIp', 'windowMinutes', -1],
    ['puriTokenInstallation', 'scope', 'ip'],
    ['puriTokenRegistrationIp', 'endpoint', 'POST /v1/trial/challenge'],
  ])('validates the %s %s knob before using configured admission policies', async (key, field, value) => {
    const env = environment();
    env.__db.prepare(`UPDATE broker_config SET value = json_set(value, ?, json(?)) WHERE key = 'abuse_controls'`)
      .run(`$.${key}.${field}`, JSON.stringify(value));
    expect(await getBrokerAbuseControlsConfig(env.BROKER_DB)).toEqual(DEFAULT_BROKER_ABUSE_CONTROLS);
  });

  it('expires indexed counters without extending their fixed windows or keeping raw IP data', async () => {
    const env = environment();
    const key = await createDeviceKeyPair();
    expect((await post(env, await proof(key))).status).toBe(200);
    vi.setSystemTime(new Date('2026-10-30T12:14:00.000Z'));
    expect((await post(env, await proof(key))).status).toBe(200);
    const rows = env.__db.prepare('SELECT * FROM puri_token_rate_limits ORDER BY policy').all();
    expect(rows).toHaveLength(3);
    expect(rows.every((row) => row.expires_at === '2026-10-30T12:15:00.000Z')).toBe(true);
    expect(JSON.stringify(rows)).not.toContain('203.0.113.33');
    await deleteExpiredPuriProofNonces(env.BROKER_DB, new Date('2026-10-30T12:14:59.999Z'));
    expect(env.__db.prepare('SELECT COUNT(*) AS count FROM puri_token_rate_limits').get()).toEqual({ count: 3 });
    await deleteExpiredPuriProofNonces(env.BROKER_DB, new Date('2026-10-30T12:15:00.000Z'));
    expect(env.__db.prepare('SELECT COUNT(*) AS count FROM puri_token_rate_limits').get()).toEqual({ count: 0 });
    expect(env.__db.prepare('SELECT installation_id FROM installations').get()).toEqual({ installation_id: installationId });
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
