import { existsSync, readFileSync } from 'node:fs';

import { describe, expect, it } from 'vitest';

import { applyAbuseMonitoringRetention } from '../src/abuse-monitoring';
import { applyAppActiveDayRetention } from '../src/telemetry';
import { updateAbuseControls } from './test-support/abuse-controls';
import { createTestBrokerEnv } from './test-support/sqlite-d1';

const FIRST_MIGRATION = new URL(
  '../migrations/0000_define_broker_persistent_state.sql',
  import.meta.url,
);

describe('broker persistence retention model', () => {
  it('retains preflight-only none rows long enough to preserve in-flight challenges before cleanup', async () => {
    const contract = await import('../src/contract');

    expect(contract).toHaveProperty('BROKER_RETENTION_POLICY', {
      challengePreflight: {
        statuses: ['none'],
        entitlementRow: 'absent',
        challengeState: 'present',
        inactiveDays: 1,
        reference: 'max(installations.last_seen_at, installations.challenge_expires_at)',
        deleteFrom: 'installations',
        cascadesTo: [],
      },
      pendingRelease: {
        statuses: ['pending_release'],
        inactiveDays: 30,
        reference: 'installations.last_seen_at',
        deleteFrom: 'installations',
        cascadesTo: ['openrouter_entitlements'],
      },
      terminal: {
        statuses: ['expired', 'revoked'],
        inactiveDays: 90,
        reference: 'max(installations.last_seen_at, openrouter_entitlements.expires_at)',
        deleteFrom: 'installations',
        cascadesTo: ['openrouter_entitlements'],
      },
    });
  });

  it('keeps entitlement state as one in-place row per installation instead of append-only history', async () => {
    const contract = await import('../src/contract');

    expect(contract).toHaveProperty(
      'BROKER_PERSISTENCE_MODEL.tables.openrouterEntitlements.updateStrategy',
      'in-place',
    );
    expect(contract).toHaveProperty(
      'BROKER_PERSISTENCE_MODEL.tables.openrouterEntitlements.rowCardinality',
      'zero-or-one-row-per-installation',
    );
    expect(contract).toHaveProperty(
      'BROKER_PERSISTENCE_MODEL.tables.openrouterEntitlements.liveRemainingBudgetSource',
      'OpenRouter metadata',
    );
  });

  it('uses cascading delete from installations so retention cleanup removes entitlement rows too', () => {
    expect(existsSync(FIRST_MIGRATION)).toBe(true);
    if (!existsSync(FIRST_MIGRATION)) {
      return;
    }

    const migration = readFileSync(FIRST_MIGRATION, 'utf8');

    expect(migration).toContain(
      'installation_id TEXT PRIMARY KEY REFERENCES installations(installation_id) ON DELETE CASCADE',
    );
  });

  it('deletes expired request events, issue-success events, and runtime-audit rows using the configured retention windows', async () => {
    const env = createTestBrokerEnv();
    updateAbuseControls(env, (controls) => {
      controls.retention.requestEventSafetyMarginDays = 7;
      controls.retention.issueSuccessDays = 3;
      controls.retention.runtimeAuditDays = 10;
    });

    const insertInstallation = env.__db.prepare(
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
    );
    insertInstallation.run(
      'install-retention-old',
      'device-public-key-retention-old',
      '1.2.3',
      '2026-04-01T00:00:00.000Z',
      '2026-04-01T00:00:00.000Z',
    );
    insertInstallation.run(
      'install-retention-new',
      'device-public-key-retention-new',
      '1.2.3',
      '2026-04-17T00:00:00.000Z',
      '2026-04-17T00:00:00.000Z',
    );

    const insertRequestEvent = env.__db.prepare(
      `INSERT INTO broker_request_events (endpoint, ip, installation_id, observed_at)
        VALUES (?, ?, ?, ?)`,
    );
    insertRequestEvent.run(
      'POST /v1/trial/challenge',
      '203.0.113.1',
      'install-retention-old',
      '2026-04-01 00:00:00',
    );
    insertRequestEvent.run(
      'POST /v1/trial/challenge',
      '203.0.113.2',
      'install-retention-new',
      '2026-04-17 00:00:00',
    );

    const insertIssueSuccess = env.__db.prepare(
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
    );
    insertIssueSuccess.run(
      'install-retention-old',
      'install-retention-old',
      'managed-retention-old',
      'ip-old',
      'prefix-old',
      64512,
      'US',
      'HTTP/2',
      'TLSv1.3',
      'TLS_AES_128_GCM_SHA256',
      'low',
      '2026-03-31 00:00:00',
    );
    insertIssueSuccess.run(
      'install-retention-new',
      'install-retention-new',
      'managed-retention-new',
      'ip-new',
      'prefix-new',
      64513,
      'US',
      'HTTP/2',
      'TLSv1.3',
      'TLS_AES_128_GCM_SHA256',
      'low',
      '2026-04-17 00:00:00',
    );

    const insertRuntimeAudit = env.__db.prepare(
      `INSERT INTO broker_abuse_runtime_audit (event_kind, reason, payload_json, created_at)
        VALUES (?, ?, ?, ?)`,
    );
    insertRuntimeAudit.run(
      'brake_transition',
      'global_threshold',
      '{"old":true}',
      '2026-04-01 00:00:00',
    );
    insertRuntimeAudit.run(
      'brake_transition',
      'global_threshold',
      '{"new":true}',
      '2026-04-17 00:00:00',
    );

    const result = await applyAbuseMonitoringRetention(
      env.BROKER_DB,
      new Date('2026-04-18T00:00:00.000Z'),
    );

    expect(result).toEqual({
      requestEventsDeleted: 1,
      issueSuccessDeleted: 1,
      runtimeAuditDeleted: 1,
    });

    expect(
      env.__db.prepare('SELECT COUNT(*) AS count FROM broker_request_events').get() as {
        count: number;
      },
    ).toEqual({ count: 1 });
    expect(
      env.__db
        .prepare('SELECT COUNT(*) AS count FROM broker_issue_success_events')
        .get() as { count: number },
    ).toEqual({ count: 1 });
    expect(
      env.__db
        .prepare('SELECT COUNT(*) AS count FROM broker_abuse_runtime_audit')
        .get() as { count: number },
    ).toEqual({ count: 1 });
  });

  it('keeps request events for the longest active enforcement window plus the configured safety margin', async () => {
    const env = createTestBrokerEnv();
    updateAbuseControls(env, (controls) => {
      controls.retention.requestEventSafetyMarginDays = 1;
    });
    env.__db
      .prepare(
        `INSERT INTO broker_velocity_cap_hooks (
            subject_type,
            subject_value,
            max_requests,
            window_minutes,
            outcome_code,
            outcome_class,
            active,
            created_at
          ) VALUES ('ip', ?, 2, 4320, 'rate_limited', 'retryable', 1, ?)`,
      )
      .run('203.0.113.44', '2026-04-01T00:00:00.000Z');
    const insert = env.__db.prepare(
      `INSERT INTO broker_request_events (
          endpoint,
          ip,
          installation_id,
          observed_at
        ) VALUES (?, ?, ?, ?)`,
    );
    insert.run(
      'POST /v1/trial/challenge',
      '203.0.113.44',
      'retention-outside-safety',
      '2026-04-15T00:00:00.000Z',
    );
    insert.run(
      'POST /v1/trial/challenge',
      '203.0.113.44',
      'retention-inside-safety',
      '2026-04-16T12:00:00.000Z',
    );

    const result = await applyAbuseMonitoringRetention(
      env.BROKER_DB,
      new Date('2026-04-20T00:00:00.000Z'),
    );

    expect(result.requestEventsDeleted).toBe(1);
    expect(
      env.__db
        .prepare(
          `SELECT installation_id
             FROM broker_request_events
            ORDER BY installation_id`,
        )
        .all(),
    ).toEqual([{ installation_id: 'retention-inside-safety' }]);
  });

  it('includes the pending Discord OAuth IP window in request-event retention', async () => {
    const env = createTestBrokerEnv();
    updateAbuseControls(env, (controls) => {
      controls.pendingDiscordOAuthSessions.windowMinutes = 4320;
      controls.retention.requestEventSafetyMarginDays = 1;
    });
    const insert = env.__db.prepare(
      `INSERT INTO broker_request_events (
          endpoint,
          ip,
          installation_id,
          observed_at
        ) VALUES (?, ?, ?, ?)`,
    );
    insert.run(
      'POST /v1/auth/discord/start',
      '203.0.113.45',
      'pending-oauth-outside-window',
      '2026-04-15T23:59:59.000Z',
    );
    insert.run(
      'POST /v1/auth/discord/start',
      '203.0.113.45',
      'pending-oauth-at-cutoff',
      '2026-04-16T00:00:00.000Z',
    );

    const result = await applyAbuseMonitoringRetention(
      env.BROKER_DB,
      new Date('2026-04-20T00:00:00.000Z'),
    );

    expect(result.requestEventsDeleted).toBe(1);
    expect(
      env.__db
        .prepare(
          `SELECT installation_id
             FROM broker_request_events
            ORDER BY installation_id`,
        )
        .all(),
    ).toEqual([{ installation_id: 'pending-oauth-at-cutoff' }]);
  });

  it('deletes app active-day rows older than 35 days and keeps rows at the cutoff or newer', async () => {
    const env = createTestBrokerEnv();

    insertAppActiveDay(env, 'subject-old', '2026-03-13');
    insertAppActiveDay(env, 'subject-cutoff', '2026-03-14');
    insertAppActiveDay(env, 'subject-new', '2026-04-18');

    const result = await applyAppActiveDayRetention(
      env.BROKER_DB,
      new Date('2026-04-18T22:00:00.000Z'),
    );

    expect(result).toEqual({
      deleted: 1,
      cutoffDateUtc: '2026-03-14',
    });
    expect(readAppActiveDays(env)).toEqual([
      { subject_ref: validAppSubjectRef('subject-cutoff'), active_date_utc: '2026-03-14' },
      { subject_ref: validAppSubjectRef('subject-new'), active_date_utc: '2026-04-18' },
    ]);
    expect(JSON.stringify(result)).not.toContain('subject-old');
  });
});

function insertAppActiveDay(
  env: ReturnType<typeof createTestBrokerEnv>,
  subjectRef: string,
  activeDateUtc: string,
): void {
  env.__db
    .prepare(
      `INSERT INTO app_active_days (
          subject_ref,
          active_date_utc
        ) VALUES (?, ?)`,
    )
    .run(validAppSubjectRef(subjectRef), activeDateUtc);
}

function readAppActiveDays(env: ReturnType<typeof createTestBrokerEnv>): Array<{
  subject_ref: string;
  active_date_utc: string;
}> {
  return env.__db
    .prepare(
      `SELECT subject_ref, active_date_utc
         FROM app_active_days
        ORDER BY active_date_utc ASC, subject_ref ASC`,
    )
    .all() as Array<{ subject_ref: string; active_date_utc: string }>;
}

function validAppSubjectRef(label: string): string {
  let hash = 0;
  for (const character of label) {
    hash = (hash * 31 + character.charCodeAt(0)) >>> 0;
  }

  return `ph-app-subject-v1_${hash.toString(16).padStart(64, '0')}`;
}
