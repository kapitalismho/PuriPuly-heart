import { afterEach, describe, expect, it, vi } from 'vitest';

import {
  deliverImmediateMonitoringSideEffects,
  evaluateImmediateAbuseState,
  recordIssueSuccess,
} from '../src/abuse-monitoring';
import {
  readAbuseRuntimeState,
  updateAbuseControls,
  updateAbuseRuntimeState,
} from './test-support/abuse-controls';
import { createTestBrokerEnv, type TestBrokerEnv } from './test-support/sqlite-d1';

describe('broker immediate incident monitoring', () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('records source-aware Discord and QQ delivery events without joining their identities', async () => {
    const env = createTestBrokerEnv();
    insertInstallation(env, 'discord-source-aware');
    const network = {
      legacyIp: null,
      legacyIpHash: null,
      legacyIpPrefixHash: null,
      ipDigest: null,
      ipPrefixDigest: null,
      ipKeyVersion: null,
      ipEpoch: null,
      asn: null,
      country: null,
      httpProtocol: null,
      tlsVersion: null,
      tlsCipher: null,
      riskLabel: null,
    };

    await recordIssueSuccess(env.BROKER_DB, {
      installationId: 'discord-source-aware',
      managedCredentialRef: 'hash-discord-source-aware',
      observedAt: '2026-04-08T06:00:00.000Z',
      network,
    });
    await recordIssueSuccess(env.BROKER_DB, {
      issueSource: 'qq',
      subjectRef: 'ph-qq-subject-v1_source_aware',
      managedCredentialRef: 'hash-qq-source-aware',
      observedAt: '2026-04-08T06:01:00.000Z',
      network,
    });

    expect(
      env.__db
        .prepare(
          `SELECT issue_source, installation_id, subject_ref
             FROM broker_issue_success_events
            ORDER BY observed_at`,
        )
        .all(),
    ).toEqual([
      {
        issue_source: 'discord',
        installation_id: 'discord-source-aware',
        subject_ref: 'discord-source-aware',
      },
      {
        issue_source: 'qq',
        installation_id: null,
        subject_ref: 'ph-qq-subject-v1_source_aware',
      },
    ]);
  });

  it('keeps healthy operation silent', async () => {
    const env = createTestBrokerEnv();
    const fetchMock = vi.fn(
      async (_input: RequestInfo | URL, _init?: RequestInit) =>
        new Response(null, { status: 204 }),
    );
    vi.stubGlobal('fetch', fetchMock);
    insertIssueSuccessEvent(env, 'healthy-1', '2026-04-08T06:10:00.000Z');

    const result = await evaluateImmediateAbuseState(
      env.BROKER_DB,
      new Date('2026-04-08T06:20:00.000Z'),
    );
    await deliverImmediateMonitoringSideEffects(env, result);

    expect(result).toMatchObject({
      warningTransition: false,
      brakeTransition: null,
      packet: {
        incident_kind: null,
        issue_success_60m: 1,
        brake_active: false,
      },
    });
    expect(fetchMock).not.toHaveBeenCalled();
    expect(
      env.__db.prepare('SELECT COUNT(*) AS count FROM broker_abuse_runtime_audit').get(),
    ).toEqual({ count: 0 });
  });

  it('emits one concise warning when the one-hour warning threshold is crossed', async () => {
    const env = createTestBrokerEnv();
    updateAbuseControls(env, (controls) => {
      controls.immediateAlerts.warning = 1;
      controls.immediateAlerts.brake = 4;
    });
    insertIssueSuccessEvent(env, 'warning-1', '2026-04-08T06:10:00.000Z');
    insertIssueSuccessEvent(env, 'warning-2', '2026-04-08T06:20:00.000Z');
    const fetchMock = vi.fn(
      async (_input: RequestInfo | URL, _init?: RequestInit) =>
        new Response(null, { status: 204 }),
    );
    vi.stubGlobal('fetch', fetchMock);

    const result = await evaluateImmediateAbuseState(
      env.BROKER_DB,
      new Date('2026-04-08T06:20:00.000Z'),
    );
    await deliverImmediateMonitoringSideEffects(env, result);

    expect(result).toMatchObject({
      warningTransition: true,
      brakeTransition: null,
      packet: {
        schema_version: 'puripuly_issuance_incident.v2',
        incident_kind: 'issuance_spike_warning',
        issue_success_60m: 2,
        warning_threshold: 1,
        brake_threshold: 4,
        brake_active: false,
      },
    });
    expect(fetchMock).toHaveBeenCalledOnce();
    const body = String(fetchMock.mock.calls[0]?.[1]?.body);
    expect(body).toContain('Broker issuance-spike warning');
    expect(body).toContain('issuance_spike_warning');
    for (const removedField of [
      'warn1',
      'warn2',
      'warn3',
      'critical',
      'funnel_60m',
      'timeline_5m_buckets',
      'asn_context',
      'protocol_risk_signals',
      'baseline_comparison',
      'derived_flags',
    ]) {
      expect(body).not.toContain(removedField);
    }
  });

  it('emits one brake incident instead of a warning when both thresholds are crossed at once', async () => {
    const env = createTestBrokerEnv();
    updateAbuseControls(env, (controls) => {
      controls.immediateAlerts.warning = 1;
      controls.immediateAlerts.brake = 2;
    });
    for (const suffix of ['1', '2', '3']) {
      insertIssueSuccessEvent(
        env,
        `brake-${suffix}`,
        `2026-04-08T06:1${suffix}:00.000Z`,
      );
    }
    const fetchMock = vi.fn(
      async (_input: RequestInfo | URL, _init?: RequestInit) =>
        new Response(null, { status: 204 }),
    );
    vi.stubGlobal('fetch', fetchMock);

    const result = await evaluateImmediateAbuseState(
      env.BROKER_DB,
      new Date('2026-04-08T06:20:00.000Z'),
    );
    await deliverImmediateMonitoringSideEffects(env, result);

    expect(result.packet.incident_kind).toBe('automatic_issuance_brake');
    expect(result.brakeTransition).toEqual({
      active: true,
      reason: 'global_threshold',
    });
    expect(readAbuseRuntimeState(env)).toMatchObject({
      brake: {
        active: true,
        reason: 'global_threshold',
      },
      alertLatches: {
        warning: true,
      },
    });
    expect(fetchMock).toHaveBeenCalledOnce();
    const body = String(fetchMock.mock.calls[0]?.[1]?.body);
    expect(body).toContain('Broker automatic issuance brake');
    expect(body).not.toContain('Broker issuance-spike warning');
  });

  it('does not duplicate a warning and rearms only after the count returns below threshold', async () => {
    const env = createTestBrokerEnv();
    updateAbuseControls(env, (controls) => {
      controls.immediateAlerts.warning = 1;
      controls.immediateAlerts.brake = 10;
    });
    insertIssueSuccessEvent(env, 'rearm-1', '2026-04-08T06:10:00.000Z');
    insertIssueSuccessEvent(env, 'rearm-2', '2026-04-08T06:20:00.000Z');

    const first = await evaluateImmediateAbuseState(
      env.BROKER_DB,
      new Date('2026-04-08T06:20:00.000Z'),
    );
    const duplicate = await evaluateImmediateAbuseState(
      env.BROKER_DB,
      new Date('2026-04-08T06:21:00.000Z'),
    );
    const below = await evaluateImmediateAbuseState(
      env.BROKER_DB,
      new Date('2026-04-08T08:00:00.000Z'),
    );
    insertIssueSuccessEvent(env, 'rearm-3', '2026-04-08T08:01:00.000Z');
    insertIssueSuccessEvent(env, 'rearm-4', '2026-04-08T08:02:00.000Z');
    const rearmed = await evaluateImmediateAbuseState(
      env.BROKER_DB,
      new Date('2026-04-08T08:02:00.000Z'),
    );

    expect(first.packet.incident_kind).toBe('issuance_spike_warning');
    expect(duplicate.packet.incident_kind).toBeNull();
    expect(below.packet.incident_kind).toBeNull();
    expect(rearmed.packet.incident_kind).toBe('issuance_spike_warning');
  });

  it('allows only one concurrent evaluator to claim a warning transition', async () => {
    const env = createTestBrokerEnv();
    updateAbuseControls(env, (controls) => {
      controls.immediateAlerts.warning = 1;
      controls.immediateAlerts.brake = 10;
    });
    insertIssueSuccessEvent(env, 'concurrent-1', '2026-04-08T06:10:00.000Z');
    insertIssueSuccessEvent(env, 'concurrent-2', '2026-04-08T06:20:00.000Z');

    const results = await Promise.all([
      evaluateImmediateAbuseState(
        env.BROKER_DB,
        new Date('2026-04-08T06:20:00.000Z'),
      ),
      evaluateImmediateAbuseState(
        env.BROKER_DB,
        new Date('2026-04-08T06:20:00.000Z'),
      ),
    ]);

    expect(
      results.filter(
        ({ packet }) => packet.incident_kind === 'issuance_spike_warning',
      ),
    ).toHaveLength(1);
    expect(
      env.__db
        .prepare(
          `SELECT COUNT(*) AS count
             FROM broker_abuse_runtime_audit
            WHERE event_kind = 'issuance_spike_warning'`,
        )
        .get(),
    ).toEqual({ count: 1 });
  });

  it('does not let an older healthy sample clear a newer warning observation', async () => {
    const env = createTestBrokerEnv();
    updateAbuseControls(env, (controls) => {
      controls.immediateAlerts.warning = 1;
      controls.immediateAlerts.brake = 10;
    });
    updateAbuseRuntimeState(env, (state) => {
      state.alertLatches.warning = true;
      state.alertLatches.warningObservedAt = '2026-04-08T08:00:00.000Z';
    });

    const staleResult = await evaluateImmediateAbuseState(
      env.BROKER_DB,
      new Date('2026-04-08T07:00:00.000Z'),
    );

    expect(staleResult.warningTransition).toBe(false);
    expect(readAbuseRuntimeState(env).alertLatches).toMatchObject({
      warning: true,
      warningObservedAt: '2026-04-08T08:00:00.000Z',
    });
  });

  it('records webhook delivery failure without surfacing it as an issuance failure', async () => {
    const env = createTestBrokerEnv();
    updateAbuseControls(env, (controls) => {
      controls.immediateAlerts.warning = 1;
      controls.immediateAlerts.brake = 10;
    });
    insertIssueSuccessEvent(env, 'delivery-fail-1', '2026-04-08T06:10:00.000Z');
    insertIssueSuccessEvent(env, 'delivery-fail-2', '2026-04-08T06:20:00.000Z');
    vi.stubGlobal('fetch', vi.fn(async () => new Response(null, { status: 503 })));
    const result = await evaluateImmediateAbuseState(
      env.BROKER_DB,
      new Date('2026-04-08T06:20:00.000Z'),
    );

    await expect(deliverImmediateMonitoringSideEffects(env, result)).resolves.toBeUndefined();
    expect(
      env.__db
        .prepare(
          `SELECT COUNT(*) AS count
             FROM broker_abuse_runtime_audit
            WHERE event_kind = 'immediate_monitoring_side_effects_failed'`,
        )
        .get(),
    ).toEqual({ count: 1 });
  });
});

function insertIssueSuccessEvent(
  env: TestBrokerEnv,
  suffix: string,
  observedAt: string,
): void {
  const installationId = `install-${suffix}`;
  insertInstallation(env, installationId);
  env.__db
    .prepare(
      `INSERT INTO broker_issue_success_events (
          issue_source,
          installation_id,
          subject_ref,
          managed_credential_ref,
          observed_at
        ) VALUES ('discord', ?, ?, ?, ?)`,
    )
    .run(installationId, installationId, `hash-${suffix}`, observedAt);
}

function insertInstallation(env: TestBrokerEnv, installationId: string): void {
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
      installationId,
      `device-public-key-${installationId}`,
      '1.2.3',
      '2026-04-08T00:00:00.000Z',
      '2026-04-08T00:00:00.000Z',
    );
}
