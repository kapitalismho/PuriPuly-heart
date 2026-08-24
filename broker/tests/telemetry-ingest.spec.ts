import { describe, expect, it, vi, afterEach } from 'vitest';

import app from '../src/index';
import {
  deriveTelemetrySubjectRef,
  recordTelemetryActiveDay,
  TELEMETRY_SIGNAL_KIND,
} from '../src/telemetry';
import { updateAbuseControls } from './test-support/abuse-controls';
import { normalizedErrorEnvelope } from './test-support/errors';
import { createTestBrokerEnv } from './test-support/sqlite-d1';

const VALID_IDENTIFIER = 'telemetry_identifier_0123456789ABCDEF';
const VALID_PAYLOAD = {
  signal: TELEMETRY_SIGNAL_KIND,
  telemetry_identifier: VALID_IDENTIFIER,
  active_date_utc: '2026-07-02',
};

describe('telemetry active-day ingest', () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    vi.useRealTimers();
  });

  it('creates the active-day schema with uniqueness and no raw identifier column', async () => {
    const env = createTestBrokerEnv();
    const columns = env.__db
      .prepare("SELECT name FROM pragma_table_info('telemetry_active_days') ORDER BY cid")
      .all() as Array<{ name: string }>;

    expect(columns.map(({ name }) => name)).toEqual([
      'subject_ref',
      'active_date_utc',
      'first_received_at',
      'last_received_at',
    ]);
    const subjectColumns = env.__db
      .prepare("SELECT name FROM pragma_table_info('telemetry_subjects') ORDER BY cid")
      .all() as Array<{ name: string }>;
    expect(subjectColumns.map(({ name }) => name)).toEqual([
      'subject_ref',
      'first_active_date_utc',
      'last_active_date_utc',
    ]);

    const subjectRef = await deriveTelemetrySubjectRef(
      env.TELEMETRY_SUBJECT_HMAC_SECRET,
      VALID_IDENTIFIER,
    );

    await recordTelemetryActiveDay(env.BROKER_DB, {
      subjectRef,
      activeDateUtc: '2026-07-02',
      receivedAt: '2026-07-02T00:00:00.000Z',
    });
    await recordTelemetryActiveDay(env.BROKER_DB, {
      subjectRef,
      activeDateUtc: '2026-07-02',
      receivedAt: '2026-07-02T00:01:00.000Z',
    });

    const row = env.__db
      .prepare('SELECT COUNT(*) AS count, MIN(first_received_at) AS first, MAX(last_received_at) AS last FROM telemetry_active_days')
      .get() as { count: number; first: string; last: string };
    expect(row).toEqual({
      count: 1,
      first: '2026-07-02T00:00:00.000Z',
      last: '2026-07-02T00:01:00.000Z',
    });
    expect(
      env.__db
        .prepare(
          'SELECT subject_ref, first_active_date_utc, last_active_date_utc FROM telemetry_subjects',
        )
        .get(),
    ).toEqual({
      subject_ref: subjectRef,
      first_active_date_utc: '2026-07-02',
      last_active_date_utc: '2026-07-02',
    });

    expect(() =>
      env.__db
        .prepare(
          'INSERT INTO telemetry_active_days (subject_ref, active_date_utc, first_received_at, last_received_at) VALUES (?, ?, ?, ?)',
        )
        .run('raw-telemetry-identifier', '2026-07-03', '2026-07-03T00:00:00.000Z', '2026-07-03T00:00:00.000Z'),
    ).toThrow(/constraint/i);

    expect(() =>
      env.__db
        .prepare(
          'INSERT INTO telemetry_active_days (subject_ref, active_date_utc, first_received_at, last_received_at) VALUES (?, ?, ?, ?)',
        )
        .run(
          `ph-telemetry-subject-v1_${'a'.repeat(63)}g`,
          '2026-07-03',
          '2026-07-03T00:00:00.000Z',
          '2026-07-03T00:00:00.000Z',
        ),
    ).toThrow(/constraint/i);
  });

  it('updates durable subject bounds atomically with active-day rows', async () => {
    let env!: ReturnType<typeof createTestBrokerEnv>;
    env = createTestBrokerEnv({
      beforeRun: ({ sql }) => {
        if (sql.includes('INSERT INTO telemetry_active_days')) {
          throw new Error('injected active-day failure');
        }
      },
    });
    const subjectRef = await deriveTelemetrySubjectRef(
      env.TELEMETRY_SUBJECT_HMAC_SECRET,
      VALID_IDENTIFIER,
    );

    await expect(
      recordTelemetryActiveDay(env.BROKER_DB, {
        subjectRef,
        activeDateUtc: '2026-07-02',
        receivedAt: '2026-07-02T00:00:00.000Z',
      }),
    ).rejects.toThrow('injected active-day failure');

    expect(
      env.__db.prepare('SELECT COUNT(*) AS count FROM telemetry_subjects').get(),
    ).toEqual({ count: 0 });
    expect(
      env.__db.prepare('SELECT COUNT(*) AS count FROM telemetry_active_days').get(),
    ).toEqual({ count: 0 });
  });

  it('accepts valid payloads, stores HMAC subject_ref, and collapses duplicates', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-07-02T12:34:56.000Z'));

    const env = createTestBrokerEnv();
    const first = await postTelemetry(env, VALID_PAYLOAD);
    const second = await postTelemetry(env, VALID_PAYLOAD);

    expect(first.status).toBe(200);
    await expect(first.json()).resolves.toEqual({ ok: true });
    expect(second.status).toBe(200);

    const rows = env.__db
      .prepare('SELECT subject_ref, active_date_utc, first_received_at, last_received_at FROM telemetry_active_days')
      .all() as Array<{
      subject_ref: string;
      active_date_utc: string;
      first_received_at: string;
      last_received_at: string;
    }>;
    const expectedSubjectRef = await deriveTelemetrySubjectRef(
      env.TELEMETRY_SUBJECT_HMAC_SECRET,
      VALID_IDENTIFIER,
    );

    expect(rows).toEqual([
      {
        subject_ref: expectedSubjectRef,
        active_date_utc: '2026-07-02',
        first_received_at: '2026-07-02T12:34:56.000Z',
        last_received_at: '2026-07-02T12:34:56.000Z',
      },
    ]);
    expect(JSON.stringify(rows)).not.toContain(VALID_IDENTIFIER);
  });

  it.each([
    ['wrong signal', { ...VALID_PAYLOAD, signal: 'app_launch' }],
    ['missing identifier', { signal: TELEMETRY_SIGNAL_KIND, active_date_utc: '2026-07-02' }],
    ['invalid identifier', { ...VALID_PAYLOAD, telemetry_identifier: 'short' }],
    ['invalid calendar date', { ...VALID_PAYLOAD, active_date_utc: '2026-02-30' }],
    ['timestamp instead of UTC date', { ...VALID_PAYLOAD, active_date_utc: '2026-07-02T00:00:00Z' }],
    ['additional field', { ...VALID_PAYLOAD, model: 'not-allowed' }],
  ])('rejects %s safely', async (_name, payload) => {
    const env = createTestBrokerEnv();
    const response = await postTelemetry(env, payload);

    expect(response.status).toBe(400);
    const body = await response.json();
    expect(body).toMatchObject({
      error: {
        code: 'invalid_request',
        class: 'terminal',
        retry_after_ms: null,
      },
    });
    expect(JSON.stringify(body)).not.toContain(VALID_IDENTIFIER);
    expect(countTelemetryRows(env)).toBe(0);
  });

  it('rejects malformed JSON without persistence', async () => {
    const env = createTestBrokerEnv();
    const response = await app.request(
      'http://broker.test/v1/telemetry/translation-success-day',
      {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: '{not json',
      },
      env,
    );

    expect(response.status).toBe(400);
    await expect(response.json()).resolves.toMatchObject({
      error: { code: 'invalid_request', class: 'terminal' },
    });
    expect(countTelemetryRows(env)).toBe(0);
  });

  it('rate limits telemetry by client IP without requiring app identity', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-07-02T12:00:00.000Z'));

    const env = createTestBrokerEnv();
    updateAbuseControls(env, (controls) => {
      controls.telemetryTranslationSuccessDayIp.maxRequests = 2;
    });

    expect((await postTelemetry(env, { ...VALID_PAYLOAD, active_date_utc: '2026-07-01' }, '203.0.113.99')).status).toBe(200);
    expect((await postTelemetry(env, VALID_PAYLOAD, '203.0.113.99')).status).toBe(200);

    const blocked = await postTelemetry(
      env,
      { ...VALID_PAYLOAD, active_date_utc: '2026-07-03' },
      '203.0.113.99',
    );

    expect(blocked.status).toBe(429);
    await expect(blocked.json()).resolves.toEqual(
      normalizedErrorEnvelope({
        code: 'rate_limited',
        class: 'retryable',
        subcode: 'ip_rate_limited',
        retryAfterMs: 900000,
        message:
          'request rate limit exceeded for POST /v1/telemetry/translation-success-day',
      }),
    );
    expect(countTelemetryRows(env)).toBe(2);
  });
});

async function postTelemetry(
  env: ReturnType<typeof createTestBrokerEnv>,
  payload: unknown,
  ip = '198.51.100.7',
): Promise<Response> {
  return app.request(
    'http://broker.test/v1/telemetry/translation-success-day',
    {
      method: 'POST',
      headers: {
        'content-type': 'application/json',
        'cf-connecting-ip': ip,
      },
      body: JSON.stringify(payload),
    },
    env,
  );
}

function countTelemetryRows(env: ReturnType<typeof createTestBrokerEnv>): number {
  const row = env.__db
    .prepare('SELECT COUNT(*) AS count FROM telemetry_active_days')
    .get() as { count: number };
  return row.count;
}
