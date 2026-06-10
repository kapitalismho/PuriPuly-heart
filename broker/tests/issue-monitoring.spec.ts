import { afterEach, describe, expect, it, vi } from 'vitest';

import app from '../src/index';
import {
  deliverImmediateMonitoringSideEffects,
  evaluateImmediateAbuseState,
  recordIssueSuccess,
} from '../src/abuse-monitoring';
import { updateAbuseControls, readAbuseRuntimeState } from './test-support/abuse-controls';
import { createDeviceKeyPair } from './test-support/ed25519';
import { normalizedErrorEnvelope } from './test-support/errors';
import { activatePendingReleaseSession } from './test-support/openrouter-issue';
import { createTestBrokerEnv, type TestBrokerEnv } from './test-support/sqlite-d1';

describe('broker immediate abuse monitoring', () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    vi.useRealTimers();
  });

  it('engages the global brake when the rolling issue-success count exceeds the critical threshold', async () => {
    const env = createTestBrokerEnv();
    updateAbuseControls(env, (controls) => {
      controls.immediateAlerts.warn1 = 10;
      controls.immediateAlerts.warn2 = 25;
      controls.immediateAlerts.warn3 = 50;
      controls.immediateAlerts.critical = 70;
      controls.asnFastPath.enabled = false;
    });

    for (let index = 0; index < 71; index += 1) {
      insertIssueSuccessEvent(env, {
        installationId: `install-global-${index}`,
        managedCredentialRef: `managed-global-${index}`,
        ipHash: `ip-global-${index}`,
        ipPrefixHash: `ip-prefix-global-${index}`,
        asn: 64512 + index,
        country: 'US',
        httpProtocol: 'HTTP/2',
        tlsVersion: 'TLSv1.3',
        tlsCipher: 'TLS_AES_128_GCM_SHA256',
        riskLabel: 'low',
        observedAt: `2026-04-08T06:${String(index % 60).padStart(2, '0')}:00.000Z`,
      });
    }

    const result = await evaluateImmediateAbuseState(
      env.BROKER_DB,
      new Date('2026-04-08T06:59:59.000Z'),
    );

    expect(result.alertsToEmit).toContain('critical');
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
        critical: true,
      },
    });
  });

  it('records QQ issue success with a nullable installation and includes it in immediate alert counts', async () => {
    const env = createTestBrokerEnv();
    updateAbuseControls(env, (controls) => {
      controls.immediateAlerts.warn1 = 1;
      controls.immediateAlerts.warn2 = 10;
      controls.immediateAlerts.warn3 = 20;
      controls.immediateAlerts.critical = 30;
      controls.asnFastPath.enabled = false;
    });

    await recordIssueSuccess(env.BROKER_DB, {
      issueSource: 'qq',
      installationId: null,
      subjectRef: 'ph-qq-subject-v1_monitoring-safe-subject',
      managedCredentialRef: 'managed-qq-monitoring-safe-ref',
      observedAt: '2026-06-10T10:05:00.000Z',
      network: {
        ipHash: 'ip-hash-qq-monitoring',
        ipPrefixHash: 'ip-prefix-qq-monitoring',
        asn: 64588,
        country: 'CN',
        httpProtocol: 'HTTP/2',
        tlsVersion: 'TLSv1.3',
        tlsCipher: 'TLS_AES_128_GCM_SHA256',
        riskLabel: 'low',
      },
    });
    await recordIssueSuccess(env.BROKER_DB, {
      issueSource: 'qq',
      installationId: null,
      subjectRef: 'ph-qq-subject-v1_monitoring-safe-subject',
      managedCredentialRef: 'managed-qq-monitoring-safe-ref-2',
      observedAt: '2026-06-10T10:06:00.000Z',
      network: {
        ipHash: 'ip-hash-qq-monitoring-2',
        ipPrefixHash: 'ip-prefix-qq-monitoring-2',
        asn: 64588,
        country: 'CN',
        httpProtocol: 'HTTP/2',
        tlsVersion: 'TLSv1.3',
        tlsCipher: 'TLS_AES_128_GCM_SHA256',
        riskLabel: 'low',
      },
    });

    const row = env.__db
      .prepare(
        `SELECT issue_source,
                installation_id,
                subject_ref,
                managed_credential_ref,
                asn,
                observed_at
           FROM broker_issue_success_events
          WHERE issue_source = 'qq'
            AND subject_ref = ?`,
      )
      .get('ph-qq-subject-v1_monitoring-safe-subject') as Record<string, unknown>;
    expect(row).toEqual({
      issue_source: 'qq',
      installation_id: null,
      subject_ref: 'ph-qq-subject-v1_monitoring-safe-subject',
      managed_credential_ref: 'managed-qq-monitoring-safe-ref',
      asn: 64588,
      observed_at: '2026-06-10T10:05:00.000Z',
    });

    const result = await evaluateImmediateAbuseState(
      env.BROKER_DB,
      new Date('2026-06-10T10:10:00.000Z'),
    );

    expect(result.packet.rolling_issue_counts.issue_success.last_60m).toBe(2);
    expect(result.alertsToEmit).toEqual(['warn1']);
    expect(JSON.stringify(result.packet)).not.toContain('qq_identity');
    expect(JSON.stringify(result.packet)).not.toContain('credential');
  });

  it('preserves Discord installation IDs exactly when defaulting source-aware subject metadata', async () => {
    const env = createTestBrokerEnv();
    const installationId = ' install-discord-source-aware-exact ';
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
        'device-key-discord-source-aware-exact',
        '1.2.3',
        '2026-06-10T10:15:00.000Z',
        '2026-06-10T10:15:00.000Z',
      );

    await recordIssueSuccess(env.BROKER_DB, {
      installationId,
      managedCredentialRef: 'managed-discord-source-aware-exact',
      observedAt: '2026-06-10T10:15:00.000Z',
      network: {
        ipHash: 'ip-hash-discord-source-aware-exact',
        ipPrefixHash: 'ip-prefix-discord-source-aware-exact',
        asn: 64589,
        country: 'US',
        httpProtocol: 'HTTP/2',
        tlsVersion: 'TLSv1.3',
        tlsCipher: 'TLS_AES_128_GCM_SHA256',
        riskLabel: 'low',
      },
    });

    const row = env.__db
      .prepare(
        `SELECT issue_source, installation_id, subject_ref
           FROM broker_issue_success_events
          WHERE managed_credential_ref = ?`,
      )
      .get('managed-discord-source-aware-exact') as Record<string, unknown>;
    expect(row).toEqual({
      issue_source: 'discord',
      installation_id: installationId,
      subject_ref: installationId,
    });
  });

  it('keeps Discord installation IDs and QQ subject refs distinct when spread metrics collide as strings', async () => {
    const env = createTestBrokerEnv();
    updateAbuseControls(env, (controls) => {
      controls.immediateAlerts.warn1 = 10;
      controls.immediateAlerts.warn2 = 20;
      controls.immediateAlerts.warn3 = 30;
      controls.immediateAlerts.critical = 40;
      controls.asnFastPath.enabled = false;
    });
    const collidingSubjectRef = 'ph-qq-subject-v1_metric-collision';

    insertIssueSuccessEvent(env, {
      installationId: collidingSubjectRef,
      managedCredentialRef: 'managed-discord-metric-collision',
      ipHash: 'ip-discord-metric-collision',
      ipPrefixHash: 'prefix-discord-metric-collision',
      asn: 64590,
      country: 'US',
      httpProtocol: 'HTTP/2',
      tlsVersion: 'TLSv1.3',
      tlsCipher: 'TLS_AES_128_GCM_SHA256',
      riskLabel: 'low',
      observedAt: '2026-06-10T10:20:00.000Z',
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
        collidingSubjectRef,
        'managed-qq-metric-collision',
        'ip-qq-metric-collision',
        'prefix-qq-metric-collision',
        64591,
        'CN',
        'HTTP/2',
        'TLSv1.3',
        'TLS_AES_128_GCM_SHA256',
        'low',
        '2026-06-10T10:21:00.000Z',
      );

    const result = await evaluateImmediateAbuseState(
      env.BROKER_DB,
      new Date('2026-06-10T10:30:00.000Z'),
    );

    expect(result.packet.rolling_issue_counts.issue_success.last_60m).toBe(2);
    expect(result.packet.spread_metrics.unique_installations_60m).toBe(2);
    expect(result.packet.spread_metrics.issues_per_installation_avg).toBe(1);
  });

  it('engages the ASN fast-path brake when cloud concentration crosses the approved threshold', async () => {
    const env = createTestBrokerEnv();
    updateAbuseControls(env, (controls) => {
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

    for (let index = 0; index < 21; index += 1) {
      insertIssueSuccessEvent(env, {
        installationId: `install-fast-path-${index}`,
        managedCredentialRef: `managed-fast-path-${index}`,
        ipHash: `ip-fast-path-${index}`,
        ipPrefixHash: `ip-prefix-fast-path-${index}`,
        asn: 24940,
        country: 'DE',
        httpProtocol: 'HTTP/1.1',
        tlsVersion: 'TLSv1.2',
        tlsCipher: 'TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256',
        riskLabel: 'high',
        observedAt: `2026-04-08T06:${String(index).padStart(2, '0')}:00.000Z`,
      });
    }

    const result = await evaluateImmediateAbuseState(
      env.BROKER_DB,
      new Date('2026-04-08T06:59:59.000Z'),
    );

    expect(result.alertsToEmit).toContain('critical');
    expect(result.brakeTransition).toEqual({ active: true, reason: 'asn_fast_path' });
    expect(result.packet.trigger_context.alert_level).toBe('critical');
    expect(readAbuseRuntimeState(env)).toMatchObject({
      brake: {
        active: true,
        reason: 'asn_fast_path',
      },
      alertLatches: {
        critical: true,
      },
    });
  });

  it('does not engage the ASN fast-path when the exact share stays below threshold but the rounded display share hits 70%', async () => {
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

    for (let index = 0; index < 16; index += 1) {
      insertIssueSuccessEvent(env, {
        installationId: `install-rounded-cloud-${index}`,
        managedCredentialRef: `managed-rounded-cloud-${index}`,
        ipHash: `ip-rounded-cloud-${index}`,
        ipPrefixHash: `ip-prefix-rounded-cloud-${index}`,
        asn: 24940,
        country: 'DE',
        httpProtocol: 'HTTP/1.1',
        tlsVersion: 'TLSv1.2',
        tlsCipher: 'TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256',
        riskLabel: 'high',
        observedAt: `2026-04-08T06:${String(index).padStart(2, '0')}:00.000Z`,
      });
    }

    for (let index = 0; index < 7; index += 1) {
      insertIssueSuccessEvent(env, {
        installationId: `install-rounded-other-${index}`,
        managedCredentialRef: `managed-rounded-other-${index}`,
        ipHash: `ip-rounded-other-${index}`,
        ipPrefixHash: `ip-prefix-rounded-other-${index}`,
        asn: 64500 + index,
        country: 'US',
        httpProtocol: 'HTTP/2',
        tlsVersion: 'TLSv1.3',
        tlsCipher: 'TLS_AES_128_GCM_SHA256',
        riskLabel: 'low',
        observedAt: `2026-04-08T06:${String(index + 16).padStart(2, '0')}:00.000Z`,
      });
    }

    const result = await evaluateImmediateAbuseState(
      env.BROKER_DB,
      new Date('2026-04-08T06:59:59.000Z'),
    );

    expect(result.packet.asn_context.top_asns[0]).toMatchObject({
      asn: 24940,
      count: 16,
      share: 70,
      kind: 'cloud_or_vps',
    });
    expect(result.alertsToEmit).toEqual([]);
    expect(result.brakeTransition).toBeNull();
    expect(result.packet.trigger_context.alert_level).toBeNull();
    expect(readAbuseRuntimeState(env).brake.active).toBe(false);
  });

  it('re-arms warn2 only after the rolling count falls back to the threshold', async () => {
    const env = createTestBrokerEnv();
    updateAbuseControls(env, (controls) => {
      controls.immediateAlerts.warn1 = 1;
      controls.immediateAlerts.warn2 = 2;
      controls.immediateAlerts.warn3 = 100;
      controls.immediateAlerts.critical = 200;
      controls.asnFastPath.enabled = false;
    });

    for (const minute of [5, 10, 15]) {
      insertIssueSuccessEvent(env, {
        installationId: `install-latch-initial-${minute}`,
        managedCredentialRef: `managed-latch-initial-${minute}`,
        ipHash: `ip-latch-initial-${minute}`,
        ipPrefixHash: `ip-prefix-latch-initial-${minute}`,
        asn: 64550,
        country: 'US',
        httpProtocol: 'HTTP/2',
        tlsVersion: 'TLSv1.3',
        tlsCipher: 'TLS_AES_128_GCM_SHA256',
        riskLabel: 'low',
        observedAt: `2026-04-08T06:${String(minute).padStart(2, '0')}:00.000Z`,
      });
    }

    const firstResult = await evaluateImmediateAbuseState(
      env.BROKER_DB,
      new Date('2026-04-08T06:15:59.000Z'),
    );
    expect(firstResult.alertsToEmit).toEqual(['warn1', 'warn2']);
    expect(readAbuseRuntimeState(env).alertLatches).toMatchObject({
      warn1: true,
      warn2: true,
    });

    const secondResult = await evaluateImmediateAbuseState(
      env.BROKER_DB,
      new Date('2026-04-08T06:20:00.000Z'),
    );
    expect(secondResult.alertsToEmit).toEqual([]);

    const resetResult = await evaluateImmediateAbuseState(
      env.BROKER_DB,
      new Date('2026-04-08T07:16:00.000Z'),
    );
    expect(resetResult.alertsToEmit).toEqual([]);
    expect(readAbuseRuntimeState(env).alertLatches).toMatchObject({
      warn1: false,
      warn2: false,
    });

    for (const minute of [17, 18, 19]) {
      insertIssueSuccessEvent(env, {
        installationId: `install-latch-rearm-${minute}`,
        managedCredentialRef: `managed-latch-rearm-${minute}`,
        ipHash: `ip-latch-rearm-${minute}`,
        ipPrefixHash: `ip-prefix-latch-rearm-${minute}`,
        asn: 64550,
        country: 'US',
        httpProtocol: 'HTTP/2',
        tlsVersion: 'TLSv1.3',
        tlsCipher: 'TLS_AES_128_GCM_SHA256',
        riskLabel: 'low',
        observedAt: `2026-04-08T07:${String(minute).padStart(2, '0')}:00.000Z`,
      });
    }

    const rearmResult = await evaluateImmediateAbuseState(
      env.BROKER_DB,
      new Date('2026-04-08T07:19:59.000Z'),
    );
    expect(rearmResult.alertsToEmit).toEqual(['warn1', 'warn2']);
  });

  it('computes cloud_asn_share_60m from the full issue-success population instead of the top-five summary', async () => {
    const env = createTestBrokerEnv();
    updateAbuseControls(env, (controls) => {
      controls.immediateAlerts.warn1 = 100;
      controls.immediateAlerts.warn2 = 200;
      controls.immediateAlerts.warn3 = 300;
      controls.immediateAlerts.critical = 400;
      controls.asnFastPath.enabled = false;
      controls.asnClassifications = [
        {
          asn: 64514,
          kind: 'cloud_or_vps',
          displayName: 'Cloud Included',
        },
        {
          asn: 64515,
          kind: 'cloud_or_vps',
          displayName: 'Cloud Excluded From Top Five',
        },
      ];
    });

    for (const [asn, offsetMinutes] of [
      [64510, 0],
      [64511, 5],
      [64512, 10],
      [64513, 15],
      [64514, 20],
      [64515, 25],
    ] as const) {
      for (let index = 0; index < 2; index += 1) {
        insertIssueSuccessEvent(env, {
          installationId: `install-cloud-share-${asn}-${index}`,
          managedCredentialRef: `managed-cloud-share-${asn}-${index}`,
          ipHash: `ip-cloud-share-${asn}-${index}`,
          ipPrefixHash: `ip-prefix-cloud-share-${asn}-${index}`,
          asn,
          country: 'US',
          httpProtocol: 'HTTP/2',
          tlsVersion: 'TLSv1.3',
          tlsCipher: 'TLS_AES_128_GCM_SHA256',
          riskLabel: 'low',
          observedAt: `2026-04-08T06:${String(offsetMinutes + index).padStart(2, '0')}:00.000Z`,
        });
      }
    }

    const result = await evaluateImmediateAbuseState(
      env.BROKER_DB,
      new Date('2026-04-08T06:59:59.000Z'),
    );

    expect(result.packet.asn_context.top_asns).toHaveLength(5);
    expect(result.packet.asn_context.cloud_asn_share_60m).toBe(33);
  });

  it('builds the richer interpretation packet shape required for immediate alert delivery', async () => {
    const env = createTestBrokerEnv();
    updateAbuseControls(env, (controls) => {
      controls.immediateAlerts.warn1 = 1;
      controls.immediateAlerts.warn2 = 2;
      controls.immediateAlerts.warn3 = 10;
      controls.immediateAlerts.critical = 20;
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

    insertRequestEvent(env, {
      endpoint: 'POST /v1/trial/challenge',
      ip: '203.0.113.1',
      installationId: 'packet-challenge-1',
      observedAt: '2026-04-08T06:05:00.000Z',
    });
    insertRequestEvent(env, {
      endpoint: 'POST /v1/trial/challenge',
      ip: '203.0.113.2',
      installationId: 'packet-challenge-2',
      observedAt: '2026-04-08T06:15:00.000Z',
    });
    insertRequestEvent(env, {
      endpoint: 'POST /v1/trial/challenge',
      ip: '203.0.113.3',
      installationId: 'packet-challenge-3',
      observedAt: '2026-04-08T06:30:00.000Z',
    });
    insertRequestEvent(env, {
      endpoint: 'POST /v1/trial/challenge',
      ip: '203.0.113.4',
      installationId: 'packet-challenge-4',
      observedAt: '2026-04-08T06:50:00.000Z',
    });

    insertRequestEvent(env, {
      endpoint: 'POST /v1/trial/challenge/verify',
      ip: '203.0.113.1',
      installationId: 'packet-verify-1',
      observedAt: '2026-04-08T06:20:00.000Z',
    });
    insertRequestEvent(env, {
      endpoint: 'POST /v1/trial/challenge/verify',
      ip: '203.0.113.2',
      installationId: 'packet-verify-2',
      observedAt: '2026-04-08T06:40:00.000Z',
    });
    insertRequestEvent(env, {
      endpoint: 'POST /v1/trial/challenge/verify',
      ip: '203.0.113.3',
      installationId: 'packet-verify-3',
      observedAt: '2026-04-08T06:55:00.000Z',
    });
    insertRequestEvent(env, {
      endpoint: 'POST /v1/trial/challenge/verify',
      ip: '203.0.113.4',
      installationId: 'packet-verify-4',
      observedAt: '2026-04-08T06:57:00.000Z',
    });
    insertRequestEvent(env, {
      endpoint: 'POST /v1/trial/challenge/verify/success',
      ip: '203.0.113.1',
      installationId: 'packet-verify-1',
      observedAt: '2026-04-08T06:20:00.000Z',
    });
    insertRequestEvent(env, {
      endpoint: 'POST /v1/trial/challenge/verify/success',
      ip: '203.0.113.2',
      installationId: 'packet-verify-2',
      observedAt: '2026-04-08T06:40:00.000Z',
    });
    insertRequestEvent(env, {
      endpoint: 'POST /v1/trial/challenge/verify/success',
      ip: '203.0.113.3',
      installationId: 'packet-verify-3',
      observedAt: '2026-04-08T06:55:00.000Z',
    });
    insertRequestEvent(env, {
      endpoint: 'POST /v1/trial/challenge/verify/fail',
      ip: '203.0.113.4',
      installationId: 'packet-verify-4',
      observedAt: '2026-04-08T06:57:00.000Z',
    });

    insertIssueSuccessEvent(env, {
      installationId: 'packet-issue-1',
      managedCredentialRef: 'packet-managed-1',
      ipHash: 'packet-ip-1',
      ipPrefixHash: 'packet-prefix-shared',
      asn: 24940,
      country: 'DE',
      httpProtocol: 'HTTP/1.1',
      tlsVersion: 'TLSv1.2',
      tlsCipher: 'TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256',
      riskLabel: 'high',
      observedAt: '2026-04-08T06:10:00.000Z',
    });
    insertIssueSuccessEvent(env, {
      installationId: 'packet-issue-2',
      managedCredentialRef: 'packet-managed-2',
      ipHash: 'packet-ip-2',
      ipPrefixHash: 'packet-prefix-shared',
      asn: 24940,
      country: 'DE',
      httpProtocol: 'HTTP/1.1',
      tlsVersion: 'TLSv1.2',
      tlsCipher: 'TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256',
      riskLabel: 'high',
      observedAt: '2026-04-08T06:50:00.000Z',
    });
    insertIssueSuccessEvent(env, {
      installationId: 'packet-issue-3',
      managedCredentialRef: 'packet-managed-3',
      ipHash: 'packet-ip-3',
      ipPrefixHash: 'packet-prefix-unique',
      asn: 64513,
      country: 'US',
      httpProtocol: 'HTTP/2',
      tlsVersion: 'TLSv1.3',
      tlsCipher: 'TLS_AES_128_GCM_SHA256',
      riskLabel: 'low',
      observedAt: '2026-04-08T06:58:00.000Z',
    });
    insertIssueSuccessEvent(env, {
      installationId: 'packet-history-1',
      managedCredentialRef: 'packet-history-managed-1',
      ipHash: 'packet-history-ip-1',
      ipPrefixHash: 'packet-history-prefix-1',
      asn: 64550,
      country: 'US',
      httpProtocol: 'HTTP/2',
      tlsVersion: 'TLSv1.3',
      tlsCipher: 'TLS_AES_128_GCM_SHA256',
      riskLabel: 'low',
      observedAt: '2026-04-06T00:10:00.000Z',
    });
    insertIssueSuccessEvent(env, {
      installationId: 'packet-history-2',
      managedCredentialRef: 'packet-history-managed-2',
      ipHash: 'packet-history-ip-2',
      ipPrefixHash: 'packet-history-prefix-2',
      asn: 64551,
      country: 'US',
      httpProtocol: 'HTTP/2',
      tlsVersion: 'TLSv1.3',
      tlsCipher: 'TLS_AES_128_GCM_SHA256',
      riskLabel: 'low',
      observedAt: '2026-04-07T00:10:00.000Z',
    });
    insertIssueSuccessEvent(env, {
      installationId: 'packet-history-3',
      managedCredentialRef: 'packet-history-managed-3',
      ipHash: 'packet-history-ip-3',
      ipPrefixHash: 'packet-history-prefix-3',
      asn: 64552,
      country: 'US',
      httpProtocol: 'HTTP/2',
      tlsVersion: 'TLSv1.3',
      tlsCipher: 'TLS_AES_128_GCM_SHA256',
      riskLabel: 'low',
      observedAt: '2026-04-07T00:20:00.000Z',
    });
    insertIssueSuccessEvent(env, {
      installationId: 'packet-history-4',
      managedCredentialRef: 'packet-history-managed-4',
      ipHash: 'packet-history-ip-4',
      ipPrefixHash: 'packet-history-prefix-4',
      asn: 64553,
      country: 'US',
      httpProtocol: 'HTTP/2',
      tlsVersion: 'TLSv1.3',
      tlsCipher: 'TLS_AES_128_GCM_SHA256',
      riskLabel: 'low',
      observedAt: '2026-04-07T01:10:00.000Z',
    });
    insertIssueSuccessEvent(env, {
      installationId: 'packet-history-5',
      managedCredentialRef: 'packet-history-managed-5',
      ipHash: 'packet-history-ip-5',
      ipPrefixHash: 'packet-history-prefix-5',
      asn: 64554,
      country: 'US',
      httpProtocol: 'HTTP/2',
      tlsVersion: 'TLSv1.3',
      tlsCipher: 'TLS_AES_128_GCM_SHA256',
      riskLabel: 'low',
      observedAt: '2026-04-07T01:20:00.000Z',
    });
    insertIssueSuccessEvent(env, {
      installationId: 'packet-history-6',
      managedCredentialRef: 'packet-history-managed-6',
      ipHash: 'packet-history-ip-6',
      ipPrefixHash: 'packet-history-prefix-6',
      asn: 64555,
      country: 'US',
      httpProtocol: 'HTTP/2',
      tlsVersion: 'TLSv1.3',
      tlsCipher: 'TLS_AES_128_GCM_SHA256',
      riskLabel: 'low',
      observedAt: '2026-04-07T01:30:00.000Z',
    });

    const result = await evaluateImmediateAbuseState(
      env.BROKER_DB,
      new Date('2026-04-08T06:59:59.000Z'),
    );
    const packet = result.packet as any;

    expect(packet).toMatchObject({
      schema_version: expect.any(String),
      alert_id: expect.any(String),
      generated_at: '2026-04-08T06:59:59.000Z',
      window_start_60m: '2026-04-08T05:59:59.000Z',
      window_end_60m: '2026-04-08T06:59:59.000Z',
      trigger_context: {
        alert_level: 'warn2',
        trigger_reason: 'threshold_crossed',
        triggered_at: '2026-04-08T06:59:59.000Z',
        brake_state: false,
        brake_reason: null,
      },
      rolling_issue_counts: {
        issue_success: {
          last_5m: 1,
          last_15m: 2,
          last_60m: 3,
          timeline_5m_buckets: expect.any(Array),
        },
      },
      funnel_60m: {
        challenge_60m: 4,
        verify_success_60m: 3,
        verify_fail_60m: 1,
        issue_success_60m: 3,
        challenge_to_verify_rate: 0.75,
        verify_to_issue_rate: 1,
      },
      asn_context: {
        top_asns: expect.arrayContaining([
          expect.objectContaining({
            asn: 24940,
            count: 2,
            share: 67,
            kind: 'cloud_or_vps',
          }),
        ]),
        cloud_asn_share_60m: 67,
      },
      spread_metrics: {
        unique_ip_hashes_60m: 3,
        unique_ip_prefixes_60m: 2,
        unique_installations_60m: 3,
        top_ip_prefix_share: 67,
        issues_per_installation_avg: 1,
      },
      protocol_risk_signals: {
        http_protocol_mix: {
          'HTTP/1.1': 2,
          'HTTP/2': 1,
        },
        tls_version_mix: {
          'TLSv1.2': 2,
          'TLSv1.3': 1,
        },
        suspicious_proto_combo_count: 2,
        risk_label_mix: {
          high: 2,
          low: 1,
        },
      },
      baseline_comparison: {
        hourly_issue_median_7d: 2,
        hourly_issue_p95_7d: 3,
        current_vs_median: 1.5,
        current_vs_p95: 1,
      },
      derived_flags: {
        cloud_asn_concentration: false,
        sudden_issue_spike: true,
        high_verify_to_issue_rate: true,
        browser_like_signal_weak: true,
      },
    });
    expect(packet).not.toHaveProperty('budget_context');
    expect(packet.rolling_issue_counts.issue_success.timeline_5m_buckets).toHaveLength(12);
  });

  it('evaluates immediate abuse state after a successful issue and blocks later onboarding once the brake engages', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-04-08T06:00:00Z'));

    const env = createTestBrokerEnv();
    updateAbuseControls(env, (controls) => {
      controls.immediateAlerts.warn1 = 1;
      controls.immediateAlerts.warn2 = 2;
      controls.immediateAlerts.warn3 = 3;
      controls.immediateAlerts.critical = 4;
      controls.asnFastPath.enabled = false;
    });

    for (const suffix of ['1', '2', '3', '4', '5']) {
      const activation = await activatePendingReleaseSession({
        env,
        installationId: `install-route-monitoring-${suffix}`,
        appVersion: '1.2.3',
        hardwareHash: `hardware-hash-route-monitoring-${suffix}`,
      });
      expect(activation.response.status).toBe(200);
    }

    expect(readAbuseRuntimeState(env)).toMatchObject({
      brake: {
        active: true,
        reason: 'global_threshold',
      },
    });

    const blockedKeyPair = await createDeviceKeyPair();
    const blockedResponse = await app.request(
      'http://broker.test/v1/trial/challenge',
      {
        method: 'POST',
        headers: {
          'content-type': 'application/json',
          'cf-connecting-ip': '203.0.113.90',
        },
        body: JSON.stringify({
          installation_id: 'install-route-monitoring-blocked',
          device_public_key: blockedKeyPair.devicePublicKey,
          app_version: '1.2.3',
        }),
      },
      env,
    );

    expect(blockedResponse.status).toBe(503);
    await expect(blockedResponse.json()).resolves.toEqual(
      normalizedErrorEnvelope({
        code: 'issuance_suspended',
        class: 'retryable',
        subcode: 'global_threshold',
        message: 'new entitlement issuance is temporarily suspended',
      }),
    );
  });

  it('delivers immediate alert payloads to the Discord webhook when a threshold crossing occurs', async () => {
    const env = createTestBrokerEnv();
    updateAbuseControls(env, (controls) => {
      controls.immediateAlerts.warn1 = 1;
      controls.immediateAlerts.warn2 = 5;
      controls.immediateAlerts.warn3 = 10;
      controls.immediateAlerts.critical = 20;
      controls.asnFastPath.enabled = false;
      controls.asnClassifications = [
        {
          asn: 64512,
          kind: 'cloud_or_vps',
          displayName: 'Test Cloud',
        },
      ];
    });

    insertIssueSuccessEvent(env, {
      installationId: 'discord-alert-install-1',
      managedCredentialRef: 'discord-alert-managed-1',
      ipHash: 'discord-alert-ip-1',
      ipPrefixHash: 'discord-alert-prefix-1',
      asn: 64512,
      country: 'US',
      httpProtocol: 'HTTP/2',
      tlsVersion: 'TLSv1.3',
      tlsCipher: 'TLS_AES_128_GCM_SHA256',
      riskLabel: 'low',
      observedAt: '2026-04-08T06:10:00.000Z',
    });
    insertIssueSuccessEvent(env, {
      installationId: 'discord-alert-install-2',
      managedCredentialRef: 'discord-alert-managed-2',
      ipHash: 'discord-alert-ip-2',
      ipPrefixHash: 'discord-alert-prefix-2',
      asn: 64513,
      country: 'US',
      httpProtocol: 'HTTP/2',
      tlsVersion: 'TLSv1.3',
      tlsCipher: 'TLS_AES_128_GCM_SHA256',
      riskLabel: 'low',
      observedAt: '2026-04-08T06:20:00.000Z',
    });

    const monitoringResult = await evaluateImmediateAbuseState(
      env.BROKER_DB,
      new Date('2026-04-08T06:20:00.000Z'),
    );
    const fetchMock = vi.fn(async () => new Response(null, { status: 204 }));
    vi.stubGlobal('fetch', fetchMock);

    await deliverImmediateMonitoringSideEffects(env, monitoringResult);

    expect(fetchMock).toHaveBeenCalledOnce();
    expect(fetchMock).toHaveBeenCalledWith(
      env.DISCORD_IMMEDIATE_ALERT_WEBHOOK_URL,
      expect.objectContaining({
        method: 'POST',
      }),
    );

    const init = (
      fetchMock.mock.calls as unknown as Array<[
        string | URL,
        RequestInit | undefined,
      ]>
    )[0]?.[1] as RequestInit;
    const body = JSON.parse(String(init.body)) as {
      content?: string;
      embeds: Array<{
        title: string;
        description?: string;
        fields?: Array<{ name: string; value: string }>;
      }>;
    };

    expect(body.embeds[0]?.title).toContain('immediate abuse alert');
    expect(body.content).toBeUndefined();
    expect(body.embeds[0]?.description).toContain(
      'broker_abuse_interpretation_packet.v1',
    );
    expect(body.embeds[0]?.fields).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          name: 'Cloud/VPS concentration',
          value: expect.stringContaining('cloud_asn_share_60m=50%'),
        }),
      ]),
    );
  });

  it('does not duplicate alert or brake transitions when a concurrent evaluation wins the same runtime-state update', async () => {
    const now = new Date('2026-04-08T06:20:00.000Z');
    let env!: TestBrokerEnv;
    let injectedConcurrentEvaluation = false;
    let competingResult: Awaited<ReturnType<typeof evaluateImmediateAbuseState>> | null =
      null;

    env = createTestBrokerEnv({
      beforeRun: async ({ sql }) => {
        if (!sql.startsWith('UPDATE broker_config') || injectedConcurrentEvaluation) {
          return;
        }

        injectedConcurrentEvaluation = true;
        competingResult = await evaluateImmediateAbuseState(env.BROKER_DB, now);
      },
    });

    updateAbuseControls(env, (controls) => {
      controls.immediateAlerts.warn1 = 1;
      controls.immediateAlerts.warn2 = 25;
      controls.immediateAlerts.warn3 = 50;
      controls.immediateAlerts.critical = 70;
      controls.asnFastPath.enabled = true;
      controls.asnFastPath.minIssueSuccess1h = 1;
      controls.asnFastPath.minTopAsnSharePct = 70;
      controls.asnClassifications = [
        {
          asn: 24940,
          kind: 'cloud_or_vps',
          displayName: 'Hetzner',
        },
      ];
    });

    for (let index = 0; index < 2; index += 1) {
      insertIssueSuccessEvent(env, {
        installationId: `install-race-${index}`,
        managedCredentialRef: `managed-race-${index}`,
        ipHash: `ip-race-${index}`,
        ipPrefixHash: `ip-prefix-race-${index}`,
        asn: 24940,
        country: 'DE',
        httpProtocol: 'HTTP/1.1',
        tlsVersion: 'TLSv1.2',
        tlsCipher: 'TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256',
        riskLabel: 'high',
        observedAt: `2026-04-08T06:1${index}:00.000Z`,
      });
    }

    const result = await evaluateImmediateAbuseState(env.BROKER_DB, now);
    const auditCounts = env.__db
      .prepare(
        `SELECT event_kind, COUNT(*) AS count
           FROM broker_abuse_runtime_audit
          GROUP BY event_kind
          ORDER BY event_kind ASC`,
      )
      .all() as Array<{ event_kind: string; count: number }>;

    expect(competingResult).toMatchObject({
      alertsToEmit: expect.arrayContaining(['critical']),
      brakeTransition: {
        active: true,
        reason: 'asn_fast_path',
      },
    });
    expect(result.alertsToEmit).toEqual([]);
    expect(result.brakeTransition).toBeNull();
    expect(readAbuseRuntimeState(env)).toMatchObject({
      brake: {
        active: true,
        reason: 'asn_fast_path',
      },
      alertLatches: {
        critical: true,
      },
    });
    expect(auditCounts).toEqual([
      {
        event_kind: 'brake_transition',
        count: 1,
      },
      {
        event_kind: 'immediate_alert_levels_emitted',
        count: 1,
      },
    ]);
  });

  it('does not double-count 5-minute timeline boundary events across adjacent buckets', async () => {
    const env = createTestBrokerEnv();
    updateAbuseControls(env, (controls) => {
      controls.immediateAlerts.warn1 = 100;
      controls.immediateAlerts.warn2 = 200;
      controls.immediateAlerts.warn3 = 300;
      controls.immediateAlerts.critical = 400;
      controls.asnFastPath.enabled = false;
    });

    for (const [index, observedAt] of [
      '2026-04-08T06:04:59.000Z',
      '2026-04-08T06:09:59.000Z',
    ].entries()) {
      insertIssueSuccessEvent(env, {
        installationId: `install-boundary-${index}`,
        managedCredentialRef: `managed-boundary-${index}`,
        ipHash: `ip-boundary-${index}`,
        ipPrefixHash: `ip-prefix-boundary-${index}`,
        asn: 64560 + index,
        country: 'US',
        httpProtocol: 'HTTP/2',
        tlsVersion: 'TLSv1.3',
        tlsCipher: 'TLS_AES_128_GCM_SHA256',
        riskLabel: 'low',
        observedAt,
      });
    }

    const result = await evaluateImmediateAbuseState(
      env.BROKER_DB,
      new Date('2026-04-08T06:59:59.000Z'),
    );
    const buckets = result.packet.rolling_issue_counts.issue_success.timeline_5m_buckets;

    expect(result.packet.rolling_issue_counts.issue_success.last_60m).toBe(2);
    expect(buckets.slice(0, 3).map(({ count }) => count)).toEqual([0, 1, 1]);
    expect(buckets.reduce((sum, bucket) => sum + bucket.count, 0)).toBe(2);
  });
});

function insertIssueSuccessEvent(
  env: TestBrokerEnv,
  input: {
    installationId: string;
    managedCredentialRef: string;
    ipHash: string | null;
    ipPrefixHash: string | null;
    asn: number | null;
    country: string | null;
    httpProtocol: string | null;
    tlsVersion: string | null;
    tlsCipher: string | null;
    riskLabel: 'low' | 'medium' | 'high';
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
      input.ipHash,
      input.ipPrefixHash,
      input.asn,
      input.country,
      input.httpProtocol,
      input.tlsVersion,
      input.tlsCipher,
      input.riskLabel,
      input.observedAt,
    );
}

function insertRequestEvent(
  env: TestBrokerEnv,
  input: {
    endpoint: string;
    ip: string;
    installationId: string;
    observedAt: string;
  },
): void {
  env.__db
    .prepare(
      `INSERT INTO broker_request_events (endpoint, ip, installation_id, observed_at)
        VALUES (?, ?, ?, ?)`,
    )
    .run(input.endpoint, input.ip, input.installationId, input.observedAt);
}
