import { afterEach, describe, expect, it, vi } from 'vitest';

import {
  buildDailySummaryPacket,
  handleScheduled,
  resolveDailyReportWindow,
  runDailyReport,
} from '../src/scheduled';
import {
  applyAppActiveDayRetention,
  recordAppActiveDay,
} from '../src/telemetry';
import {
  updateAbuseControls,
  updateAbuseRuntimeState,
} from './test-support/abuse-controls';
import {
  createTestBrokerEnv,
  seedRequestEvent,
} from './test-support/sqlite-d1';

describe('PuriPuly daily summary v2', () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    vi.useRealTimers();
  });

  it('emits the quiet completed-day contract without legacy security or cohort fields', async () => {
    const env = createTestBrokerEnv();
    const fetchMock = vi.fn(async () => new Response(null, { status: 204 }));
    vi.stubGlobal('fetch', fetchMock);

    const sent = await runDailyReport(env, new Date('2026-04-20T00:05:00.000Z'));

    expect(sent.payload).toEqual({
      schema_version: 'puripuly_daily_summary.v2',
      report_date_utc: '2026-04-19',
      window_start: '2026-04-19T00:00:00.000Z',
      window_end: '2026-04-20T00:00:00.000Z',
      summary: {
        keys_delivered_total: 0,
        keys_delivered_discord: 0,
        keys_delivered_qq: 0,
        app_dau: 0,
        app_wau: 0,
        app_mau: 0,
      },
    });
    expect(sent.sent).toBe(true);
    expect(readDailySummaryDeliveries(env)).toEqual([
      expect.objectContaining({
        report_date_utc: '2026-04-19',
        status: 'delivered',
        attempted_at: '2026-04-20T00:05:00.000Z',
        delivered_at: '2026-04-20T00:05:00.000Z',
      }),
    ]);
    expect(fetchMock).toHaveBeenCalledOnce();

    const request = (
      fetchMock.mock.calls as unknown as Array<[
        string | URL,
        RequestInit | undefined,
      ]>
    )[0]?.[1];
    if (!request) {
      throw new Error('expected daily summary request');
    }
    const body = JSON.parse(String(request.body)) as {
      content: string;
      embeds: Array<{
        title: string;
        description: string;
        fields: Array<{ name: string; value: string }>;
      }>;
    };
    const serialized = JSON.stringify(body);
    expect(body.embeds[0]?.title).toBe('PuriPuly daily summary — 2026-04-19 UTC');
    expect(body.embeds[0]?.description).toBe(
      '2026-04-19T00:00:00.000Z ≤ observed_at < 2026-04-20T00:00:00.000Z',
    );
    expect(body.embeds[0]?.fields).toEqual([
      {
        name: 'Managed key issuance',
        value: 'keys_delivered=0\ndiscord=0\nqq=0',
        inline: true,
      },
      {
        name: 'App usage',
        value: 'dau=0\nwau=0\nmau=0',
        inline: true,
      },
    ]);
    for (const removed of [
      'broker_daily_heartbeat.v1',
      'challenge_24h',
      'verify_24h',
      'highest_alert_level_24h',
      'brake_triggered_24h',
      'manual_revocations_24h',
      'top_asns',
      'cloud_asn_share_24h',
      'dau_mau_stickiness_pct',
      'retention',
    ]) {
      expect(serialized).not.toContain(removed);
    }
  });

  it('waits until 00:05 UTC and delivers each completed report date once', async () => {
    const env = createTestBrokerEnv();
    const fetchMock = vi.fn(async () => new Response(null, { status: 204 }));
    vi.stubGlobal('fetch', fetchMock);
    const executionCtx = {
      waitUntil() {},
      passThroughOnException() {},
    };

    await handleScheduled(
      { scheduledTime: Date.parse('2026-04-20T00:04:00.000Z') },
      env,
      executionCtx,
    );
    await handleScheduled(
      { scheduledTime: Date.parse('2026-04-20T00:05:00.000Z') },
      env,
      executionCtx,
    );
    await handleScheduled(
      { scheduledTime: Date.parse('2026-04-20T23:59:00.000Z') },
      env,
      executionCtx,
    );
    await handleScheduled(
      { scheduledTime: Date.parse('2026-04-21T00:05:00.000Z') },
      env,
      executionCtx,
    );

    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(readDailySummaryDeliveries(env).map((row) => row.report_date_utc)).toEqual([
      '2026-04-19',
      '2026-04-20',
    ]);
  });

  it('ignores a legacy v1 execution-date stamp for the first v2 report', async () => {
    const env = createTestBrokerEnv();
    const fetchMock = vi.fn(async () => new Response(null, { status: 204 }));
    vi.stubGlobal('fetch', fetchMock);
    updateAbuseRuntimeState(env, (state) => {
      state.dailyReport.lastDeliveredAt = '2026-04-19T13:00:00.000Z';
      state.dailyReport.lastDeliveredDateUtc = '2026-04-19';
    });

    await handleScheduled(
      { scheduledTime: Date.parse('2026-04-20T00:05:00.000Z') },
      env,
      {},
    );

    expect(fetchMock).toHaveBeenCalledOnce();
    expect(readDailySummaryDeliveries(env)).toEqual([
      expect.objectContaining({
        report_date_utc: '2026-04-19',
        status: 'delivered',
      }),
    ]);
  });

  it('uses one report-date lease across overlapping cron executions', async () => {
    const env = createTestBrokerEnv();
    const fetchMock = vi.fn(async () => new Response(null, { status: 204 }));
    vi.stubGlobal('fetch', fetchMock);
    const scheduledTime = Date.parse('2026-04-20T00:05:00.000Z');

    await Promise.all([
      handleScheduled({ scheduledTime }, env, {}),
      handleScheduled({ scheduledTime }, env, {}),
    ]);

    expect(fetchMock).toHaveBeenCalledOnce();
    expect(readDailySummaryDeliveries(env)).toHaveLength(1);
    expect(readDailySummaryDeliveries(env)[0]).toMatchObject({
      report_date_utc: '2026-04-19',
      status: 'delivered',
    });
  });

  it('recovers an expired report-date lease after an interrupted attempt', async () => {
    const env = createTestBrokerEnv();
    const fetchMock = vi.fn(async () => new Response(null, { status: 204 }));
    vi.stubGlobal('fetch', fetchMock);
    env.__db
      .prepare(
        `INSERT INTO broker_daily_summary_deliveries (
            report_date_utc,
            status,
            lease_token,
            lease_expires_at,
            attempted_at,
            delivered_at
          ) VALUES (?, 'pending', ?, ?, ?, NULL)`,
      )
      .run(
        '2026-04-19',
        '00000000-0000-0000-0000-000000000000',
        '2026-04-20T00:04:59.999Z',
        '2026-04-20T00:00:00.000Z',
      );

    const result = await runDailyReport(
      env,
      new Date('2026-04-20T00:05:00.000Z'),
    );

    expect(result.sent).toBe(true);
    expect(fetchMock).toHaveBeenCalledOnce();
    expect(readDailySummaryDeliveries(env)[0]).toMatchObject({
      report_date_utc: '2026-04-19',
      status: 'delivered',
      attempted_at: '2026-04-20T00:05:00.000Z',
    });
  });

  it('preserves a failed report across midnight and catches up completed dates in order', async () => {
    const env = createTestBrokerEnv();
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(new Response(null, { status: 503 }))
      .mockResolvedValue(new Response(null, { status: 204 }));
    vi.stubGlobal('fetch', fetchMock);
    const scheduledTime = Date.parse('2026-04-20T23:59:00.000Z');
    insertIssueSuccessEvent(env, {
      source: 'discord',
      label: 'failed-report-midnight-delivery',
      observedAt: '2026-04-19 18:00:00',
    });

    await expect(handleScheduled({ scheduledTime }, env, {})).rejects.toThrow(
      'discord webhook failed: 503',
    );
    expect(readDailySummaryDeliveries(env)).toEqual([
      expect.objectContaining({
        report_date_utc: '2026-04-19',
        status: 'pending',
        attempted_at: '2026-04-20T23:59:00.000Z',
        lease_expires_at: '2026-04-20T23:59:00.000Z',
      }),
    ]);
    await recordActiveDay(env, 'late-received-report-day', '2026-04-19');

    await handleScheduled(
      { scheduledTime: Date.parse('2026-04-21T00:04:00.000Z') },
      env,
      {},
    );
    expect(
      env.__db
        .prepare(
          `SELECT COUNT(*) AS count
             FROM broker_issue_success_events
            WHERE managed_credential_ref = ?`,
        )
        .get('managed-failed-report-midnight-delivery'),
    ).toEqual({ count: 1 });

    await handleScheduled(
      { scheduledTime: Date.parse('2026-04-21T00:05:00.000Z') },
      env,
      {},
    );
    await handleScheduled(
      { scheduledTime: Date.parse('2026-04-21T00:06:00.000Z') },
      env,
      {},
    );

    expect(fetchMock).toHaveBeenCalledTimes(3);
    const sentBodies = fetchMock.mock.calls.map((call) =>
      JSON.parse(String((call[1] as RequestInit).body)) as { content: string },
    );
    const sentPackets = sentBodies.map((body) =>
      JSON.parse(body.content.slice('```json\n'.length, -'\n```'.length)) as {
        report_date_utc: string;
        window_start: string;
        window_end: string;
        summary: { app_dau: number };
      },
    );
    expect(
      sentPackets.map(({ report_date_utc, window_start, window_end }) => ({
        report_date_utc,
        window_start,
        window_end,
      })),
    ).toEqual([
      {
        report_date_utc: '2026-04-19',
        window_start: '2026-04-19T00:00:00.000Z',
        window_end: '2026-04-20T00:00:00.000Z',
      },
      {
        report_date_utc: '2026-04-19',
        window_start: '2026-04-19T00:00:00.000Z',
        window_end: '2026-04-20T00:00:00.000Z',
      },
      {
        report_date_utc: '2026-04-20',
        window_start: '2026-04-20T00:00:00.000Z',
        window_end: '2026-04-21T00:00:00.000Z',
      },
    ]);
    expect(sentPackets.map((packet) => packet.summary.app_dau)).toEqual([0, 1, 0]);
    expect(readDailySummaryDeliveries(env)).toEqual([
      expect.objectContaining({
        report_date_utc: '2026-04-19',
        status: 'delivered',
      }),
      expect.objectContaining({
        report_date_utc: '2026-04-20',
        status: 'delivered',
      }),
    ]);
  });

  it('runs retention during a persistent webhook failure without deleting the pending report window', async () => {
    const env = createTestBrokerEnv();
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => new Response(null, { status: 503 })),
    );
    insertIssueSuccessEvent(env, {
      source: 'discord',
      label: 'retention-during-report-failure',
      observedAt: '2026-04-20 12:00:00',
    });
    await seedRequestEvent(env, {
      endpoint: 'POST /v1/trial/challenge',
      ip: '203.0.113.10',
      installationId: 'retention-during-report-failure',
      observedAt: '2025-01-01 12:00:00',
    });
    env.__db
      .prepare(
        `INSERT INTO broker_abuse_runtime_audit (
            event_kind,
            reason,
            payload_json,
            created_at
          ) VALUES (?, ?, ?, ?)`,
      )
      .run('state_observation', null, '{}', '2025-01-01 12:00:00');
    await recordActiveDay(env, 'retention-during-report-failure', '2025-01-01');

    await expect(
      handleScheduled(
        { scheduledTime: Date.parse('2026-04-21T00:05:00.000Z') },
        env,
        {},
      ),
    ).rejects.toThrow('discord webhook failed: 503');

    expect(
      env.__db.prepare('SELECT COUNT(*) AS count FROM broker_request_events').get(),
    ).toEqual({ count: 0 });
    expect(
      env.__db
        .prepare(
          `SELECT COUNT(*) AS count
             FROM broker_abuse_runtime_audit
            WHERE created_at = ?`,
        )
        .get('2025-01-01 12:00:00'),
    ).toEqual({ count: 0 });
    expect(
      env.__db.prepare('SELECT COUNT(*) AS count FROM app_active_days').get(),
    ).toEqual({ count: 0 });
    expect(
      env.__db
        .prepare(
          `SELECT COUNT(*) AS count
             FROM broker_issue_success_events
            WHERE managed_credential_ref = ?`,
        )
        .get('managed-retention-during-report-failure'),
    ).toEqual({ count: 1 });
    expect(readDailySummaryDeliveries(env)).toEqual([
      expect.objectContaining({
        report_date_utc: '2026-04-20',
        status: 'pending',
      }),
    ]);
  });

  it('retains the whole completed day before a tuned minimum-retention report', async () => {
    const env = createTestBrokerEnv();
    const fetchMock = vi.fn(async () => new Response(null, { status: 204 }));
    vi.stubGlobal('fetch', fetchMock);
    updateAbuseControls(env, (controls) => {
      controls.retention.issueSuccessDays = 1;
    });
    insertIssueSuccessEvent(env, {
      source: 'discord',
      label: 'retained-midnight-delivery',
      observedAt: '2026-04-19T00:00:00.000Z',
    });

    await handleScheduled(
      { scheduledTime: Date.parse('2026-04-20T00:05:00.000Z') },
      env,
      {},
    );

    const request = (
      fetchMock.mock.calls as unknown as Array<[
        string | URL,
        RequestInit | undefined,
      ]>
    )[0]?.[1];
    const body = JSON.parse(String(request?.body)) as { content: string };
    expect(body.content).toContain('"keys_delivered_total":1');
    expect(
      env.__db
        .prepare(
          `SELECT COUNT(*) AS count
             FROM broker_issue_success_events
            WHERE managed_credential_ref = ?`,
        )
        .get('managed-retained-midnight-delivery'),
    ).toEqual({ count: 1 });
  });

  it('anchors retries to one fixed completed-day window', async () => {
    const env = createTestBrokerEnv();

    const early = await buildDailySummaryPacket(
      env.BROKER_DB,
      new Date('2026-04-20T00:05:00.000Z'),
    );
    const late = await buildDailySummaryPacket(
      env.BROKER_DB,
      new Date('2026-04-20T23:59:59.999Z'),
    );

    expect(early).toEqual(late);
    expect(resolveDailyReportWindow(new Date('2026-04-20T12:00:00.000Z'))).toEqual({
      reportDateUtc: '2026-04-19',
      windowStart: '2026-04-19T00:00:00.000Z',
      windowEnd: '2026-04-20T00:00:00.000Z',
    });
  });

  it('uses a half-open key-delivery window and source-aware totals', async () => {
    const env = createTestBrokerEnv();
    insertIssueSuccessEvent(env, {
      source: 'discord',
      label: 'prior-boundary',
      observedAt: '2026-04-18T23:59:59.999Z',
    });
    insertIssueSuccessEvent(env, {
      source: 'discord',
      label: 'discord-start',
      observedAt: '2026-04-19T00:00:00.000Z',
    });
    insertIssueSuccessEvent(env, {
      source: 'discord',
      label: 'discord-end-minus-one',
      observedAt: '2026-04-19T23:59:59.999Z',
    });
    insertIssueSuccessEvent(env, {
      source: 'qq',
      label: 'qq-middle',
      observedAt: '2026-04-19T12:00:00.000Z',
    });
    insertIssueSuccessEvent(env, {
      source: 'qq',
      label: 'qq-legacy-sql-timestamp',
      observedAt: '2026-04-19 18:00:00',
    });
    insertIssueSuccessEvent(env, {
      source: 'qq',
      label: 'qq-middle-duplicate',
      managedCredentialRef: 'managed-qq-middle',
      observedAt: '2026-04-19T12:00:00.001Z',
    });
    insertIssueSuccessEvent(env, {
      source: 'qq',
      label: 'next-boundary',
      observedAt: '2026-04-20T00:00:00.000Z',
    });

    const packet = await buildDailySummaryPacket(
      env.BROKER_DB,
      new Date('2026-04-20T00:05:00.000Z'),
    );

    expect(packet.summary).toMatchObject({
      keys_delivered_total: 4,
      keys_delivered_discord: 2,
      keys_delivered_qq: 2,
    });
    expect(packet.summary.keys_delivered_total).toBe(
      packet.summary.keys_delivered_discord + packet.summary.keys_delivered_qq,
    );
    const serialized = JSON.stringify(packet);
    expect(serialized).not.toContain('discord-start');
    expect(serialized).not.toContain('qq-middle');
    expect(serialized).not.toContain('credential');
  });

  it('calculates completed-day app DAU, WAU, and MAU', async () => {
    const env = createTestBrokerEnv();
    await recordActiveDay(env, 'new-on-report-date', '2026-04-19');
    await recordActiveDay(env, 'new-on-report-date', '2026-04-19');
    await recordActiveDay(env, 'returning-on-report-date', '2026-04-01');
    await recordActiveDay(env, 'returning-on-report-date', '2026-04-19');
    await recordActiveDay(env, 'weekly-only', '2026-04-13');
    await recordActiveDay(env, 'monthly-only', '2026-03-21');
    await recordActiveDay(env, 'outside-month', '2026-03-20');
    await recordActiveDay(env, 'future-date', '2026-04-20');

    const packet = await buildDailySummaryPacket(
      env.BROKER_DB,
      new Date('2026-04-20T00:05:00.000Z'),
    );

    expect(packet.summary).toMatchObject({
      app_dau: 2,
      app_wau: 3,
      app_mau: 4,
    });
    const activeDayCount = env.__db
      .prepare(
        `SELECT COUNT(*) AS count
           FROM app_active_days
          WHERE subject_ref = ?
            AND active_date_utc = ?`,
      )
      .get(validAppSubjectRef('new-on-report-date'), '2026-04-19') as {
      count: number;
    };
    expect(activeDayCount.count).toBe(1);
  });

  it('retains only the 35-day app activity window', async () => {
    const env = createTestBrokerEnv();
    await recordActiveDay(env, 'expired-activity', '2026-03-15');
    await recordActiveDay(env, 'cutoff-activity', '2026-03-16');
    await recordActiveDay(env, 'recent-activity', '2026-04-19');

    const result = await applyAppActiveDayRetention(
      env.BROKER_DB,
      new Date('2026-04-20T00:05:00.000Z'),
    );

    expect(result).toEqual({ deleted: 1, cutoffDateUtc: '2026-03-16' });
    expect(
      env.__db
        .prepare(
          'SELECT active_date_utc FROM app_active_days ORDER BY active_date_utc',
        )
        .all(),
    ).toEqual([
      { active_date_utc: '2026-03-16' },
      { active_date_utc: '2026-04-19' },
    ]);
  });
});

function insertIssueSuccessEvent(
  env: ReturnType<typeof createTestBrokerEnv>,
  input: {
    source: 'discord' | 'qq';
    label: string;
    managedCredentialRef?: string;
    observedAt: string;
  },
): void {
  if (input.source === 'discord') {
    env.__db
      .prepare(
        `INSERT INTO installations (
            installation_id,
            device_public_key,
            app_version,
            created_at,
            last_seen_at
          ) VALUES (?, ?, ?, ?, ?)`,
      )
      .run(
        input.label,
        `device-public-key-${input.label}`,
        '1.2.3',
        input.observedAt,
        input.observedAt,
      );
  }
  const subjectRef =
    input.source === 'discord'
      ? input.label
      : `ph-qq-subject-v1_${input.label}`;
  env.__db
    .prepare(
      `INSERT INTO broker_issue_success_events (
          issue_source,
          installation_id,
          subject_ref,
          managed_credential_ref,
          observed_at
        ) VALUES (?, ?, ?, ?, ?)`,
    )
    .run(
      input.source,
      input.source === 'discord' ? input.label : null,
      subjectRef,
      input.managedCredentialRef ?? `managed-${input.label}`,
      input.observedAt,
    );
}

function readDailySummaryDeliveries(
  env: ReturnType<typeof createTestBrokerEnv>,
): Array<{
  report_date_utc: string;
  status: string;
  lease_token: string;
  lease_expires_at: string;
  attempted_at: string;
  delivered_at: string | null;
}> {
  return env.__db
    .prepare(
      `SELECT report_date_utc, status, lease_token, lease_expires_at,
              attempted_at, delivered_at
         FROM broker_daily_summary_deliveries
        ORDER BY report_date_utc`,
    )
    .all() as Array<{
    report_date_utc: string;
    status: string;
    lease_token: string;
    lease_expires_at: string;
    attempted_at: string;
    delivered_at: string | null;
  }>;
}

async function recordActiveDay(
  env: ReturnType<typeof createTestBrokerEnv>,
  label: string,
  activeDateUtc: string,
): Promise<void> {
  await recordAppActiveDay(env.BROKER_DB, {
    subjectRef: validAppSubjectRef(label),
    activeDateUtc,
  });
}

function validAppSubjectRef(label: string): string {
  let hash = 0;
  for (const character of label) {
    hash = (hash * 31 + character.charCodeAt(0)) >>> 0;
  }
  return `ph-app-subject-v1_${hash.toString(16).padStart(64, '0')}`;
}
