import { afterEach, describe, expect, it, vi } from 'vitest';

import app from '../src/index';
import { updateAbuseControls } from './test-support/abuse-controls';
import { normalizedErrorEnvelope } from './test-support/errors';
import { sha256Base64Url } from './test-support/hash';
import { createTestBrokerEnv, type TestBrokerEnv } from './test-support/sqlite-d1';

const QQ_AUTH_ASSERT_URL = 'http://broker.test/v1/auth/qq/assert';
const QQ_AUTH_ASSERT_ENDPOINT = 'POST /v1/auth/qq/assert';
const encoder = new TextEncoder();

interface QqAuthAssertionRow {
  qq_subject_ref: string;
  credential_hash: string;
  asserted_at: string;
  received_at: string;
  status: string;
}

describe('QQ auth assertion route', () => {
  afterEach(() => {
    vi.useRealTimers();
  });

  it('preserves legacy verification when OpenRouter issuance config is absent and stores only normalized derived evidence', async () => {
    const env = createTestBrokerEnv();
    env.OPENROUTER_MANAGEMENT_API_KEY = '   ';
    const qqIdentity = 'qq-openid-valid-user';
    const credential = await signQqCredential(env.QQ_AUTH_HMAC_PSK, qqIdentity);

    const response = await postQqAssertion(env, {
      qq_identity: qqIdentity,
      credential,
      asserted_at: '2026-06-05T20:03:00+08:00',
    });

    expect(response.status).toBe(200);
    const payload = (await response.json()) as {
      ok: boolean;
      status: string;
      qq_subject_ref: string;
    };
    expect(payload).toEqual({
      ok: true,
      status: 'verified',
      qq_subject_ref: expect.stringMatching(/^ph-qq-subject-v1_[A-Za-z0-9_-]+$/u),
    });

    const rows = listQqAssertions(env);
    expect(rows).toHaveLength(1);
    expect(rows[0]).toEqual({
      qq_subject_ref: payload.qq_subject_ref,
      credential_hash: `sha256-base64url-v1_${await sha256Base64Url(credential)}`,
      asserted_at: '2026-06-05T12:03:00.000Z',
      received_at: expect.any(String),
      status: 'verified',
    });
    const persistedRow = JSON.stringify(rows[0]);
    expect(persistedRow).not.toContain(qqIdentity);
    expect(persistedRow).not.toContain(credential);
    expect(countQqManagedEntitlements(env)).toBe(0);
  });

  it('returns already_verified for a repeated valid assertion only when OpenRouter issuance config is disabled', async () => {
    const env = createTestBrokerEnv();
    env.OPENROUTER_MANAGED_GUARDRAIL_ID = '';
    const qqIdentity = 'qq-openid-duplicate-user';
    const credential = await signQqCredential(env.QQ_AUTH_HMAC_PSK, qqIdentity);

    const firstResponse = await postQqAssertion(env, {
      qq_identity: qqIdentity,
      credential,
      asserted_at: '2026-06-05T12:03:00Z',
    });
    expect(firstResponse.status).toBe(200);
    const firstPayload = (await firstResponse.json()) as { qq_subject_ref: string };
    const originalRow = listQqAssertions(env)[0];

    const duplicateResponse = await postQqAssertion(env, {
      qq_identity: qqIdentity,
      credential,
      asserted_at: '2026-06-06T15:45:00Z',
    });

    expect(duplicateResponse.status).toBe(200);
    await expect(duplicateResponse.json()).resolves.toEqual({
      ok: true,
      status: 'already_verified',
      qq_subject_ref: firstPayload.qq_subject_ref,
    });
    expect(listQqAssertions(env)).toEqual([originalRow]);
    expect(countQqManagedEntitlements(env)).toBe(0);
  });

  it.each([
    [
      'missing management API key',
      (env: TestBrokerEnv) => {
        delete (env as Record<string, unknown>).OPENROUTER_MANAGEMENT_API_KEY;
      },
    ],
    [
      'blank management API key',
      (env: TestBrokerEnv) => {
        env.OPENROUTER_MANAGEMENT_API_KEY = '   ';
      },
    ],
    [
      'missing managed guardrail id',
      (env: TestBrokerEnv) => {
        delete (env as Record<string, unknown>).OPENROUTER_MANAGED_GUARDRAIL_ID;
      },
    ],
    [
      'blank managed guardrail id',
      (env: TestBrokerEnv) => {
        env.OPENROUTER_MANAGED_GUARDRAIL_ID = '   ';
      },
    ],
  ])(
    'preserves verification-only compatibility with %s and does not touch QQ entitlements',
    async (_caseName, mutateEnv) => {
      const env = createTestBrokerEnv();
      mutateEnv(env);
      const qqIdentity = 'qq-openid-runtime-gate-compat-user';
      const credential = await signQqCredential(env.QQ_AUTH_HMAC_PSK, qqIdentity);

      const response = await postQqAssertion(env, {
        qq_identity: qqIdentity,
        credential,
        asserted_at: '2026-06-05T12:03:00Z',
      });

      expect(response.status).toBe(200);
      await expect(response.json()).resolves.toEqual({
        ok: true,
        status: 'verified',
        qq_subject_ref: expect.stringMatching(/^ph-qq-subject-v1_[A-Za-z0-9_-]+$/u),
      });
      expect(listQqAssertions(env)).toHaveLength(1);
      expect(countQqManagedEntitlements(env)).toBe(0);
    },
  );

  it('returns a bounded retryable error instead of legacy verified when issuance config is enabled but issuance is unavailable', async () => {
    const env = createTestBrokerEnv();
    const qqIdentity = 'qq-openid-enabled-issuance-user';
    const credential = await signQqCredential(env.QQ_AUTH_HMAC_PSK, qqIdentity);

    const response = await postQqAssertion(env, {
      qq_identity: qqIdentity,
      credential,
      asserted_at: '2026-06-05T12:03:00Z',
    });

    expect(response.status).toBe(500);
    const responseBody = await response.json();
    expect(responseBody).toEqual(
      normalizedErrorEnvelope({
        code: 'internal_error',
        class: 'retryable',
        message: 'broker encountered an unexpected internal error',
      }),
    );
    const responseText = JSON.stringify(responseBody);
    expect(responseText).not.toContain(qqIdentity);
    expect(responseText).not.toContain(credential);
    const rows = listQqAssertions(env);
    expect(rows).toHaveLength(1);
    expect(rows[0]?.asserted_at).toBe('2026-06-05T12:03:00.000Z');
    expect(countQqManagedEntitlements(env)).toBe(0);
  });

  it('rejects an invalid QQ credential without persisting a row', async () => {
    const env = createTestBrokerEnv();
    const qqIdentity = 'qq-openid-invalid-credential-user';
    const expectedCredential = await signQqCredential(env.QQ_AUTH_HMAC_PSK, qqIdentity);
    const invalidCredential = '0'.repeat(64);

    const response = await postQqAssertion(env, {
      qq_identity: qqIdentity,
      credential: invalidCredential,
      asserted_at: '2026-06-05T12:03:00Z',
    });

    expect(response.status).toBe(401);
    const responseBody = await response.json();
    expect(responseBody).toEqual(
      normalizedErrorEnvelope({
        code: 'invalid_request',
        class: 'security_fail',
        subcode: 'qq_credential_invalid',
        message: 'QQ assertion credential is invalid',
      }),
    );
    const responseText = JSON.stringify(responseBody);
    expect(responseText).not.toContain(qqIdentity);
    expect(responseText).not.toContain(invalidCredential);
    expect(responseText).not.toContain(expectedCredential);
    expect(listQqAssertions(env)).toHaveLength(0);
    expect(countQqManagedEntitlements(env)).toBe(0);
  });

  it('rejects malformed credentials before HMAC comparison without persisting raw request values', async () => {
    const env = createTestBrokerEnv();
    env.OPENROUTER_MANAGEMENT_API_KEY = '';
    const qqIdentity = 'qq-openid-malformed-credential-user';
    const malformedCredential = 'A'.repeat(64);

    const response = await postQqAssertion(env, {
      qq_identity: qqIdentity,
      credential: malformedCredential,
      asserted_at: '2026-06-05T12:03:00Z',
    });

    expect(response.status).toBe(400);
    const responseBody = await response.json();
    expect(responseBody).toEqual(
      normalizedErrorEnvelope({
        code: 'invalid_request',
        class: 'terminal',
        message: 'credential must be exactly 64 lowercase hexadecimal characters',
      }),
    );
    const responseText = JSON.stringify(responseBody);
    expect(responseText).not.toContain(qqIdentity);
    expect(responseText).not.toContain(malformedCredential);
    expect(listQqAssertions(env)).toHaveLength(0);
    expect(countQqManagedEntitlements(env)).toBe(0);
  });

  it('rejects over-broad QQ identity values without persisting raw request values', async () => {
    const env = createTestBrokerEnv();
    env.OPENROUTER_MANAGED_GUARDRAIL_ID = '';
    const qqIdentity = 'q'.repeat(2049);
    const credential = await signQqCredential(env.QQ_AUTH_HMAC_PSK, qqIdentity);

    const response = await postQqAssertion(env, {
      qq_identity: qqIdentity,
      credential,
      asserted_at: '2026-06-05T12:03:00Z',
    });

    expect(response.status).toBe(400);
    const responseBody = await response.json();
    expect(responseBody).toEqual(
      normalizedErrorEnvelope({
        code: 'invalid_request',
        class: 'terminal',
        message: 'qq_identity must be between 1 and 2048 characters',
      }),
    );
    const responseText = JSON.stringify(responseBody);
    expect(responseText).not.toContain(qqIdentity);
    expect(responseText).not.toContain(credential);
    expect(listQqAssertions(env)).toHaveLength(0);
    expect(countQqManagedEntitlements(env)).toBe(0);
  });

  it('rejects asserted_at smuggling attempts without persisting raw timestamp text', async () => {
    const env = createTestBrokerEnv();
    env.OPENROUTER_MANAGEMENT_API_KEY = '';
    const qqIdentity = 'qq-openid-asserted-at-smuggling-user';
    const credential = await signQqCredential(env.QQ_AUTH_HMAC_PSK, qqIdentity);
    const smuggledAssertedAt = '2026-06-05T12:03:00Z\nqq-raw-smuggled-sentinel';

    const response = await postQqAssertion(env, {
      qq_identity: qqIdentity,
      credential,
      asserted_at: smuggledAssertedAt,
    });

    expect(response.status).toBe(400);
    const responseBody = await response.json();
    expect(responseBody).toEqual(
      normalizedErrorEnvelope({
        code: 'invalid_request',
        class: 'terminal',
        message: 'asserted_at must be a valid ISO-8601 timestamp',
      }),
    );
    const responseText = JSON.stringify(responseBody);
    expect(responseText).not.toContain(qqIdentity);
    expect(responseText).not.toContain(credential);
    expect(responseText).not.toContain(smuggledAssertedAt);
    expect(listQqAssertions(env)).toHaveLength(0);
    expect(countQqManagedEntitlements(env)).toBe(0);
  });

  it('counts invalid attempts toward the QQ assert IP rate limit', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-06-05T12:00:00Z'));

    const env = createTestBrokerEnv();
    updateAbuseControls(env, (controls) => {
      controls.qqAuthAssertIp.maxRequests = 1;
    });
    const headers = { 'cf-connecting-ip': '203.0.113.77' };

    const firstResponse = await postQqAssertion(
      env,
      {
        qq_identity: 'qq-openid-rate-limit-one',
        credential: '1'.repeat(64),
        asserted_at: '2026-06-05T12:00:00Z',
      },
      headers,
    );
    expect(firstResponse.status).toBe(401);

    const secondResponse = await postQqAssertion(
      env,
      {
        qq_identity: 'qq-openid-rate-limit-two',
        credential: '2'.repeat(64),
        asserted_at: '2026-06-05T12:00:01Z',
      },
      headers,
    );

    expect(secondResponse.status).toBe(429);
    await expect(secondResponse.json()).resolves.toEqual(
      normalizedErrorEnvelope({
        code: 'rate_limited',
        class: 'retryable',
        subcode: 'ip_rate_limited',
        retryAfterMs: 900000,
        message: `request rate limit exceeded for ${QQ_AUTH_ASSERT_ENDPOINT}`,
      }),
    );
    expect(countQqRequestEvents(env, '203.0.113.77')).toBe(2);
    expect(listQqAssertions(env)).toHaveLength(0);
  });

  it('counts malformed attempts toward the QQ assert IP rate limit', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-06-05T12:00:00Z'));

    const env = createTestBrokerEnv();
    updateAbuseControls(env, (controls) => {
      controls.qqAuthAssertIp.maxRequests = 1;
    });
    const headers = { 'cf-connecting-ip': '203.0.113.78' };

    const firstResponse = await postQqAssertion(
      env,
      {
        qq_identity: 'qq-openid-rate-limit-malformed-one',
        credential: 'short',
        asserted_at: '2026-06-05T12:00:00Z',
      },
      headers,
    );
    expect(firstResponse.status).toBe(400);

    const secondResponse = await postQqAssertion(
      env,
      {
        qq_identity: 'qq-openid-rate-limit-malformed-two',
        credential: '2'.repeat(64),
        asserted_at: '2026-06-05T12:00:01Z',
      },
      headers,
    );

    expect(secondResponse.status).toBe(429);
    await expect(secondResponse.json()).resolves.toEqual(
      normalizedErrorEnvelope({
        code: 'rate_limited',
        class: 'retryable',
        subcode: 'ip_rate_limited',
        retryAfterMs: 900000,
        message: `request rate limit exceeded for ${QQ_AUTH_ASSERT_ENDPOINT}`,
      }),
    );
    expect(countQqRequestEvents(env, '203.0.113.78')).toBe(2);
    expect(listQqAssertions(env)).toHaveLength(0);
    expect(countQqManagedEntitlements(env)).toBe(0);
  });

  it.each([
    [
      'missing',
      (env: TestBrokerEnv) => {
        delete (env as Record<string, unknown>).QQ_AUTH_HMAC_PSK;
      },
    ],
    [
      'blank',
      (env: TestBrokerEnv) => {
        env.QQ_AUTH_HMAC_PSK = '   ';
      },
    ],
  ])('fails closed when QQ_AUTH_HMAC_PSK is %s', async (_caseName, mutateEnv) => {
    const env = createTestBrokerEnv();
    mutateEnv(env);
    const qqIdentity = 'qq-openid-unconfigured-secret-user';
    const credential = '3'.repeat(64);

    const response = await postQqAssertion(env, {
      qq_identity: qqIdentity,
      credential,
      asserted_at: '2026-06-05T12:03:00Z',
    });

    expect(response.status).toBe(500);
    const responseBody = await response.json();
    expect(responseBody).toEqual(
      normalizedErrorEnvelope({
        code: 'internal_error',
        class: 'retryable',
        message: 'broker encountered an unexpected internal error',
      }),
    );
    const responseText = JSON.stringify(responseBody);
    expect(responseText).not.toContain(qqIdentity);
    expect(responseText).not.toContain(credential);
    expect(listQqAssertions(env)).toHaveLength(0);
    expect(countQqManagedEntitlements(env)).toBe(0);
  });

  it('rejects malformed JSON with the public invalid_request envelope', async () => {
    const env = createTestBrokerEnv();

    const response = await app.request(
      QQ_AUTH_ASSERT_URL,
      {
        method: 'POST',
        headers: {
          'content-type': 'application/json',
          'cf-connecting-ip': '203.0.113.88',
        },
        body: '{"qq_identity":"qq-openid-malformed","credential":',
      },
      env,
    );

    expect(response.status).toBe(400);
    const responseBody = await response.json();
    expect(responseBody).toEqual(
      normalizedErrorEnvelope({
        code: 'invalid_request',
        class: 'terminal',
        message: 'request body must be valid JSON',
      }),
    );
    expect(JSON.stringify(responseBody)).not.toContain('qq-openid-malformed');
    expect(listQqAssertions(env)).toHaveLength(0);
    expect(countQqRequestEvents(env, '203.0.113.88')).toBe(1);
  });
});

async function postQqAssertion(
  env: TestBrokerEnv,
  body: Record<string, unknown>,
  headers: Record<string, string> = {},
): Promise<Response> {
  return app.request(
    QQ_AUTH_ASSERT_URL,
    {
      method: 'POST',
      headers: {
        'content-type': 'application/json',
        ...headers,
      },
      body: JSON.stringify(body),
    },
    env,
  );
}

async function signQqCredential(secret: string, qqIdentity: string): Promise<string> {
  const key = await crypto.subtle.importKey(
    'raw',
    encoder.encode(secret),
    { name: 'HMAC', hash: 'SHA-256' },
    false,
    ['sign'],
  );
  const signature = await crypto.subtle.sign('HMAC', key, encoder.encode(qqIdentity));

  return Array.from(new Uint8Array(signature), (value) =>
    value.toString(16).padStart(2, '0'),
  ).join('');
}

function listQqAssertions(env: TestBrokerEnv): QqAuthAssertionRow[] {
  return env.__db
    .prepare(
      `SELECT qq_subject_ref, credential_hash, asserted_at, received_at, status
         FROM qq_auth_assertions
        ORDER BY qq_subject_ref`,
    )
    .all() as unknown as QqAuthAssertionRow[];
}

function countQqRequestEvents(env: TestBrokerEnv, ip: string): number {
  const row = env.__db
    .prepare(
      `SELECT COUNT(*) AS count
         FROM broker_request_events
        WHERE endpoint = ?
          AND ip = ?`,
    )
    .get(QQ_AUTH_ASSERT_ENDPOINT, ip) as { count: number };

  return Number(row.count);
}

function countQqManagedEntitlements(env: TestBrokerEnv): number {
  const row = env.__db
    .prepare('SELECT COUNT(*) AS count FROM qq_managed_entitlements')
    .get() as { count: number };

  return Number(row.count);
}
