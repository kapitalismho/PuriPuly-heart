import { afterEach, describe, expect, it, vi } from 'vitest';

import app from '../src/index';
import {
  deriveAppSubjectRef,
  recordAppActiveDay,
} from '../src/telemetry';
import { createTestBrokerEnv } from './test-support/sqlite-d1';

const VALID_IDENTIFIER = 'telemetry_identifier_0123456789ABCDEF';
const VALID_PAYLOAD = {
  anonymous_id: VALID_IDENTIFIER,
  active_date_utc: '2026-08-28',
};

describe('app active-day ingest', () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    vi.useRealTimers();
  });

  it('stores only a derived subject reference and UTC date', async () => {
    const env = createTestBrokerEnv();
    const columns = env.__db
      .prepare("SELECT name FROM pragma_table_info('app_active_days') ORDER BY cid")
      .all() as Array<{ name: string }>;

    expect(columns.map(({ name }) => name)).toEqual([
      'subject_ref',
      'active_date_utc',
    ]);

    const subjectRef = await deriveAppSubjectRef(
      env.TELEMETRY_SUBJECT_HMAC_SECRET,
      VALID_IDENTIFIER,
    );
    await recordAppActiveDay(env.BROKER_DB, {
      subjectRef,
      activeDateUtc: '2026-08-28',
    });
    await recordAppActiveDay(env.BROKER_DB, {
      subjectRef,
      activeDateUtc: '2026-08-28',
    });

    expect(
      env.__db
        .prepare('SELECT subject_ref, active_date_utc FROM app_active_days')
        .all(),
    ).toEqual([
      {
        subject_ref: subjectRef,
        active_date_utc: '2026-08-28',
      },
    ]);
    expect(JSON.stringify(env.__db.prepare('SELECT * FROM app_active_days').all()))
      .not.toContain(VALID_IDENTIFIER);
  });

  it('accepts today and the previous UTC date and collapses duplicates', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-08-28T12:34:56.000Z'));
    const env = createTestBrokerEnv();

    const today = await postTelemetry(env, VALID_PAYLOAD);
    const duplicate = await postTelemetry(env, VALID_PAYLOAD);
    const previous = await postTelemetry(env, {
      ...VALID_PAYLOAD,
      active_date_utc: '2026-08-27',
    });

    expect(today.status).toBe(200);
    await expect(today.json()).resolves.toEqual({ ok: true });
    expect(duplicate.status).toBe(200);
    expect(previous.status).toBe(200);
    expect(countAppActiveDays(env)).toBe(2);
  });

  it.each([
    ['missing identifier', { active_date_utc: '2026-08-28' }],
    ['invalid identifier', { ...VALID_PAYLOAD, anonymous_id: 'short' }],
    ['invalid calendar date', { ...VALID_PAYLOAD, active_date_utc: '2026-02-30' }],
    ['timestamp instead of UTC date', { ...VALID_PAYLOAD, active_date_utc: '2026-08-28T00:00:00Z' }],
    ['future date', { ...VALID_PAYLOAD, active_date_utc: '2026-08-29' }],
    ['older date', { ...VALID_PAYLOAD, active_date_utc: '2026-08-26' }],
    ['additional metadata', { ...VALID_PAYLOAD, app_version: '2.5.1' }],
    ['legacy signal', { ...VALID_PAYLOAD, signal: 'translation_success_day' }],
  ])('rejects %s without persistence', async (_name, payload) => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-08-28T12:34:56.000Z'));
    const env = createTestBrokerEnv();

    const response = await postTelemetry(env, payload);

    expect(response.status).toBe(400);
    await expect(response.json()).resolves.toMatchObject({
      error: {
        code: 'invalid_request',
        class: 'terminal',
        retry_after_ms: null,
      },
    });
    expect(countAppActiveDays(env)).toBe(0);
  });

  it('rejects malformed JSON without persistence', async () => {
    const env = createTestBrokerEnv();
    const response = await app.request(
      'http://broker.test/v1/telemetry/app-active-day',
      {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: '{not json',
      },
      env,
    );

    expect(response.status).toBe(400);
    expect(countAppActiveDays(env)).toBe(0);
  });

  it('does not rate limit or persist request IP events', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-08-28T12:34:56.000Z'));
    const env = createTestBrokerEnv();

    const responses = await Promise.all(
      Array.from({ length: 65 }, (_, index) =>
        postTelemetry(
          env,
          {
            anonymous_id: `anonymous-installation-${index.toString().padStart(3, '0')}`,
            active_date_utc: '2026-08-28',
          },
          '203.0.113.99',
        ),
      ),
    );

    expect(responses.every((response) => response.status === 200)).toBe(true);
    expect(countAppActiveDays(env)).toBe(65);
    expect(
      env.__db
        .prepare(
          `SELECT COUNT(*) AS count
             FROM broker_request_events
            WHERE endpoint = ?`,
        )
        .get('POST /v1/telemetry/app-active-day'),
    ).toEqual({ count: 0 });
  });

  it('does not mix legacy translation telemetry into app activity', async () => {
    const env = createTestBrokerEnv();
    const response = await app.request(
      'http://broker.test/v1/telemetry/translation-success-day',
      {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({
          signal: 'translation_success_day',
          telemetry_identifier: VALID_IDENTIFIER,
          active_date_utc: '2026-08-28',
        }),
      },
      env,
    );

    expect(response.status).toBe(404);
    expect(countAppActiveDays(env)).toBe(0);
  });
});

async function postTelemetry(
  env: ReturnType<typeof createTestBrokerEnv>,
  payload: unknown,
  ip = '198.51.100.7',
): Promise<Response> {
  return app.request(
    'http://broker.test/v1/telemetry/app-active-day',
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

function countAppActiveDays(env: ReturnType<typeof createTestBrokerEnv>): number {
  const row = env.__db
    .prepare('SELECT COUNT(*) AS count FROM app_active_days')
    .get() as { count: number };
  return row.count;
}
