import { afterEach, describe, expect, it, vi } from 'vitest';

import { applyAbuseMonitoringRetention } from '../src/abuse-monitoring';
import {
  buildDailyHeartbeatPacket,
  handleScheduled,
  runDailyReport,
} from '../src/scheduled';
import {
  readAbuseRuntimeState,
  updateAbuseControls,
} from './test-support/abuse-controls';
import { createTestBrokerEnv } from './test-support/sqlite-d1';

describe('broker daily heartbeat', () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    vi.useRealTimers();
  });

  it('emits the daily report even when the last 24 hours were quiet', async () => {
    const env = createTestBrokerEnv();
    updateAbuseControls(env, (controls) => {
      controls.dailyReport.enabled = true;
      controls.dailyReport.hourUtc = 0;
      controls.dailyReport.minuteUtc = 0;
      controls.dailyReport.includeZeroActivity = false;
    });

    const fetchMock = vi.fn(async () => new Response(null, { status: 204 }));
    vi.stubGlobal('fetch', fetchMock);

    const sent = await runDailyReport(env, new Date('2026-04-19T00:00:00.000Z'));

    expect(sent.ok).toBe(true);
    expect(sent.payload.summary.issue_success_24h).toBe(0);
    expect(fetchMock).toHaveBeenCalledOnce();
    expect(fetchMock).toHaveBeenCalledWith(
      env.DISCORD_DAILY_REPORT_WEBHOOK_URL,
      expect.objectContaining({
        method: 'POST',
      }),
    );

    const init = (
      fetchMock.mock.calls as unknown as Array<[
        string | URL,
        RequestInit | undefined,
      ]>
    )[0]?.[1];

    if (!init) {
      throw new Error('expected fetch request init');
    }

    const body = JSON.parse(String(init.body)) as {
      content?: string;
      embeds: Array<{ title: string; fields: Array<{ name: string }> }>;
    };

    expect(body.embeds[0]?.title).toContain('daily heartbeat');
    expect(body.content).toContain('broker_daily_heartbeat.v1');
    expect(body.content).not.toContain('estimated_monthly_exposure_usd');
    expect(body.content).not.toContain('monthly_cap_usd');
    expect(body.content).not.toContain('remaining_budget_usd');
    expect(body.embeds[0]?.fields.map((field) => field.name)).not.toContain(
      'Budget summary',
    );
    expect(readAbuseRuntimeState(env).dailyReport).toEqual({
      lastDeliveredAt: '2026-04-19T00:00:00.000Z',
      lastDeliveredDateUtc: '2026-04-19',
    });
  });

  it('waits until the configured UTC schedule and sends only once per date', async () => {
    const env = createTestBrokerEnv();
    updateAbuseControls(env, (controls) => {
      controls.dailyReport.enabled = true;
      controls.dailyReport.hourUtc = 1;
      controls.dailyReport.minuteUtc = 30;
    });

    const fetchMock = vi.fn(async () => new Response(null, { status: 204 }));
    vi.stubGlobal('fetch', fetchMock);

    const executionCtx = {
      waitUntil() {},
      passThroughOnException() {},
    };

    await handleScheduled(
      {
        cron: '* * * * *',
        scheduledTime: Date.parse('2026-04-19T01:29:00.000Z'),
      },
      env,
      executionCtx,
    );
    expect(fetchMock).not.toHaveBeenCalled();

    await handleScheduled(
      {
        cron: '* * * * *',
        scheduledTime: Date.parse('2026-04-19T01:30:00.000Z'),
      },
      env,
      executionCtx,
    );
    expect(fetchMock).toHaveBeenCalledTimes(1);

    await handleScheduled(
      {
        cron: '* * * * *',
        scheduledTime: Date.parse('2026-04-19T01:45:00.000Z'),
      },
      env,
      executionCtx,
    );
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(readAbuseRuntimeState(env).dailyReport.lastDeliveredDateUtc).toBe(
      '2026-04-19',
    );
  });

  it('exports a scheduled handler from the worker entrypoint', async () => {
    const worker = await import('../src/index');

    expect(worker.default.scheduled).toBeTypeOf('function');
    expect(worker.default.fetch).toBeTypeOf('function');
    expect(worker.default.request).toBeTypeOf('function');
  });

  it('computes cloud_asn_share_24h from the full 24h issue population rather than the top-5 subset', async () => {
    const env = createTestBrokerEnv();
    updateAbuseControls(env, (controls) => {
      controls.dailyReport.enabled = true;
      controls.dailyReport.hourUtc = 0;
      controls.dailyReport.minuteUtc = 0;
      controls.asnClassifications = [
        {
          asn: 65535,
          kind: 'cloud_or_vps',
          displayName: 'Cloud Tail ASN',
        },
      ];
    });

    for (const [index, asn] of [64501, 64502, 64503, 64504, 64505, 65535].entries()) {
      insertIssueSuccessEvent(env, {
        installationId: `daily-cloud-share-${index}`,
        managedCredentialRef: `daily-cloud-share-managed-${index}`,
        asn,
        observedAt: `2026-04-18T0${index}:00:00.000Z`,
      });
    }

    const fetchMock = vi.fn(async () => new Response(null, { status: 204 }));
    vi.stubGlobal('fetch', fetchMock);

    const sent = await runDailyReport(env, new Date('2026-04-19T00:00:00.000Z'));

    expect(sent.payload.summary.top_asns).toHaveLength(5);
    expect(sent.payload.summary.cloud_asn_share_24h).toBe(17);
  });

  it('counts QQ nullable-installation issue successes in the daily heartbeat without raw identity', async () => {
    const env = createTestBrokerEnv();
    updateAbuseControls(env, (controls) => {
      controls.dailyReport.enabled = true;
      controls.dailyReport.hourUtc = 0;
      controls.dailyReport.minuteUtc = 0;
    });

    insertIssueSuccessEvent(env, {
      installationId: 'daily-source-aware-discord',
      managedCredentialRef: 'daily-source-aware-discord-managed',
      asn: 64570,
      observedAt: '2026-06-09T23:00:00.000Z',
    });
    env.__db
      .prepare(
        `INSERT INTO broker_issue_success_events (
            issue_source,
            installation_id,
            subject_ref,
            managed_credential_ref,
            ip_hash,
            ip_prefix_hash,
            asn,
            country,
            http_protocol,
            tls_version,
            tls_cipher,
            risk_label,
            observed_at
          ) VALUES ('qq', NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
      )
      .run(
        'ph-qq-subject-v1_daily-safe-subject',
        'daily-source-aware-qq-managed',
        'ip-hash-daily-qq',
        'ip-prefix-daily-qq',
        64571,
        'CN',
        'HTTP/2',
        'TLSv1.3',
        'TLS_AES_128_GCM_SHA256',
        'low',
        '2026-06-09T23:10:00.000Z',
      );

    const packet = await buildDailyHeartbeatPacket(
      env.BROKER_DB,
      new Date('2026-06-10T00:00:00.000Z'),
    );

    expect(packet.summary.issue_success_24h).toBe(2);
    expect(packet.summary.top_asns).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          asn: 64571,
          count: 1,
        }),
      ]),
    );
    expect(JSON.stringify(packet)).not.toContain('qq_identity');
    expect(JSON.stringify(packet)).not.toContain('credential');
  });

  it('omits monthly budget exposure from daily heartbeat payloads after retention cleanup', async () => {
    const env = createTestBrokerEnv();
    updateAbuseControls(env, (controls) => {
      controls.dailyReport.enabled = true;
      controls.dailyReport.hourUtc = 0;
      controls.dailyReport.minuteUtc = 0;
      controls.retention.issueSuccessDays = 3;
    });

    insertIssueSuccessEvent(env, {
      installationId: 'daily-monthly-exposure-prior-month',
      managedCredentialRef: 'daily-monthly-exposure-managed-prior-month',
      asn: 64501,
      observedAt: '2026-03-31T23:00:00.000Z',
    });
    insertIssueSuccessEvent(env, {
      installationId: 'daily-monthly-exposure-current-month-early',
      managedCredentialRef: 'daily-monthly-exposure-managed-current-month-early',
      asn: 64502,
      observedAt: '2026-04-02T10:00:00.000Z',
    });
    insertIssueSuccessEvent(env, {
      installationId: 'daily-monthly-exposure-current-month-late',
      managedCredentialRef: 'daily-monthly-exposure-managed-current-month-late',
      asn: 64503,
      observedAt: '2026-04-17T12:00:00.000Z',
    });

    const retentionResult = await applyAbuseMonitoringRetention(
      env.BROKER_DB,
      new Date('2026-04-18T00:00:00.000Z'),
    );

    expect(retentionResult.issueSuccessDeleted).toBe(2);

    const fetchMock = vi.fn(async () => new Response(null, { status: 204 }));
    vi.stubGlobal('fetch', fetchMock);

    const sent = await runDailyReport(env, new Date('2026-04-18T12:00:00.000Z'));

    expect(sent.payload.summary.issue_success_24h).toBe(1);
    expect(sent.payload.summary).not.toHaveProperty('estimated_monthly_exposure_usd');
    expect(sent.payload.summary).not.toHaveProperty('monthly_cap_usd');
    expect(sent.payload.summary).not.toHaveProperty('remaining_budget_usd');
  });

  it('derives highest_alert_level_24h from actual rolling threshold state at the report-window boundary', async () => {
    const env = createTestBrokerEnv();
    updateAbuseControls(env, (controls) => {
      controls.immediateAlerts.warn1 = 10;
      controls.immediateAlerts.warn2 = 25;
      controls.immediateAlerts.warn3 = 50;
      controls.immediateAlerts.critical = 70;
    });

    for (let index = 0; index < 11; index += 1) {
      insertIssueSuccessEvent(env, {
        installationId: `daily-highest-alert-prewindow-${index}`,
        managedCredentialRef: `daily-highest-alert-prewindow-managed-${index}`,
        asn: 64510 + index,
        observedAt: `2026-04-17T23:${String(index * 5).padStart(2, '0')}:00.000Z`,
      });
    }

    const packet = await buildDailyHeartbeatPacket(
      env.BROKER_DB,
      new Date('2026-04-19T00:00:00.000Z'),
    );

    expect(packet.summary.issue_success_24h).toBe(0);
    expect(packet.summary.highest_alert_level_24h).toBe('warn1');
  });

  it('does not report a critical fast-path alert when the exact ASN share stays below threshold but rounds to 70%', async () => {
    const env = createTestBrokerEnv();
    updateAbuseControls(env, (controls) => {
      controls.immediateAlerts.warn1 = 100;
      controls.immediateAlerts.warn2 = 200;
      controls.immediateAlerts.warn3 = 300;
      controls.immediateAlerts.critical = 400;
      controls.asnFastPath.enabled = true;
      controls.asnFastPath.minIssueSuccess1h = 20;
      controls.asnFastPath.minTopAsnSharePct = 70;
      controls.asnClassifications = [
        {
          asn: 24940,
          kind: 'cloud_or_vps',
          displayName: 'Hetzner',
        },
      ];
    });

    for (let index = 0; index < 14; index += 1) {
      insertIssueSuccessEvent(env, {
        installationId: `daily-rounded-cloud-${index}`,
        managedCredentialRef: `daily-rounded-cloud-managed-${index}`,
        asn: 24940,
        observedAt: `2026-04-18T23:${String(index).padStart(2, '0')}:00.000Z`,
      });
    }

    for (let index = 0; index < 7; index += 1) {
      insertIssueSuccessEvent(env, {
        installationId: `daily-rounded-other-${index}`,
        managedCredentialRef: `daily-rounded-other-managed-${index}`,
        asn: 64520 + index,
        observedAt: `2026-04-18T23:${String(index + 14).padStart(2, '0')}:00.000Z`,
      });
    }

    for (let index = 14; index < 16; index += 1) {
      insertIssueSuccessEvent(env, {
        installationId: `daily-rounded-cloud-${index}`,
        managedCredentialRef: `daily-rounded-cloud-managed-${index}`,
        asn: 24940,
        observedAt: `2026-04-18T23:${String(index + 7).padStart(2, '0')}:00.000Z`,
      });
    }

    const packet = await buildDailyHeartbeatPacket(
      env.BROKER_DB,
      new Date('2026-04-19T00:00:00.000Z'),
    );

    expect(packet.summary.issue_success_24h).toBe(23);
    expect(packet.summary.highest_alert_level_24h).toBeNull();
  });
});

function insertIssueSuccessEvent(
  env: ReturnType<typeof createTestBrokerEnv>,
  input: {
    installationId: string;
    managedCredentialRef: string;
    asn: number;
    observedAt: string;
  },
): void {
  env.__db
    .prepare(
      `INSERT INTO installations (
          installation_id,
          device_public_key,
          hardware_hash,
          hardware_hash_salt_version,
          app_version,
          challenge,
          challenge_expires_at,
          challenge_salt_version,
          created_at,
          last_seen_at
        ) VALUES (?, ?, NULL, NULL, ?, NULL, NULL, NULL, ?, ?)`,
    )
    .run(
      input.installationId,
      `device-public-key-${input.installationId}`,
      '1.2.3',
      input.observedAt,
      input.observedAt,
    );

  env.__db
    .prepare(
      `INSERT INTO broker_issue_success_events (
          issue_source,
          installation_id,
          subject_ref,
          managed_credential_ref,
          ip_hash,
          ip_prefix_hash,
          asn,
          country,
          http_protocol,
          tls_version,
          tls_cipher,
          risk_label,
          observed_at
        ) VALUES ('discord', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
    )
    .run(
      input.installationId,
      input.installationId,
      input.managedCredentialRef,
      `ip-hash-${input.installationId}`,
      `ip-prefix-${input.installationId}`,
      input.asn,
      'US',
      'HTTP/2',
      'TLSv1.3',
      'TLS_AES_128_GCM_SHA256',
      'low',
      input.observedAt,
    );
}
