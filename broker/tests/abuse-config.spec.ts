import { afterEach, describe, expect, it, vi } from 'vitest';

import app from '../src/index';
import {
  checkEndpointRateLimit,
  getBrokerAbuseControlsConfig,
  getBrokerAbuseRuntimeState,
} from '../src/abuse-controls';
import {
  TEST_DEFAULT_ABUSE_CONTROLS,
  TEST_DEFAULT_ABUSE_RUNTIME_STATE,
  readAbuseControls,
  readAbuseRuntimeState,
  replaceAbuseControlsValue,
  updateAbuseControls,
  updateAbuseRuntimeState,
} from './test-support/abuse-controls';
import { createDeviceKeyPair } from './test-support/ed25519';
import {
  createTestBrokerEnv,
  seedRequestEvent,
  testNetworkIdentitySecrets,
} from './test-support/sqlite-d1';

describe('broker abuse-controls runtime config validation', () => {
  afterEach(() => {
    vi.useRealTimers();
  });

  it('falls back to default abuse controls when the stored config is still on the previous rollout layout', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-04-08T06:00:00Z'));

    const env = createTestBrokerEnv();
    replaceAbuseControlsValue(env, {
      trialChallenge: {
        endpoint: 'POST /v1/trial/challenge',
        scope: 'ip',
        maxRequests: 1,
        windowMinutes: 15,
      },
      trialChallengeVerify: {
        endpoint: 'POST /v1/trial/challenge/verify',
        scope: 'installation_id',
        maxRequests: 5,
        windowMinutes: 15,
      },
      openrouterIssue: {
        endpoint: 'POST /v1/providers/openrouter/issue',
        scope: 'installation_id',
        maxRequests: 3,
        windowMinutes: 15,
      },
      trialStatus: {
        endpoint: 'GET /v1/trial/status',
        scope: 'installation_id',
        maxRequests: 30,
        windowMinutes: 15,
      },
      newActiveEntitlementsPerDay: {
        endpoint: 'POST /v1/providers/openrouter/issue',
        scope: 'global',
        maxCount: null,
        windowDays: 1,
      },
    });

    for (const suffix of Array.from({ length: 10 }, (_, index) => `${index + 1}`)) {
      const keyPair = await createDeviceKeyPair();
      const response = await app.request(
        'http://broker.test/v1/trial/challenge',
        {
          method: 'POST',
          headers: {
            'content-type': 'application/json',
            'cf-connecting-ip': '203.0.113.71',
          },
          body: JSON.stringify({
            installation_id: `install-malformed-config-${suffix}`,
            device_public_key: keyPair.devicePublicKey,
            app_version: '1.2.3',
          }),
        },
        env,
      );

      expect(response.status).toBe(200);
    }

    const blockedKeyPair = await createDeviceKeyPair();
    const blockedResponse = await app.request(
      'http://broker.test/v1/trial/challenge',
      {
        method: 'POST',
        headers: {
          'content-type': 'application/json',
          'cf-connecting-ip': '203.0.113.71',
        },
        body: JSON.stringify({
          installation_id: 'install-malformed-config-11',
          device_public_key: blockedKeyPair.devicePublicKey,
          app_version: '1.2.3',
        }),
      },
      env,
    );

    expect(blockedResponse.status).toBe(429);
  });

  it('uses runtime overrides only when the full exact abuse-control layout is valid', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-04-08T06:00:00Z'));

    const env = createTestBrokerEnv();
    updateAbuseControls(env, (controls) => {
      controls.trialChallenge.maxRequests = 1;
    });

    const firstKeyPair = await createDeviceKeyPair();
    const firstResponse = await app.request(
      'http://broker.test/v1/trial/challenge',
      {
        method: 'POST',
        headers: {
          'content-type': 'application/json',
          'cf-connecting-ip': '203.0.113.72',
        },
        body: JSON.stringify({
          installation_id: 'install-valid-runtime-config-1',
          device_public_key: firstKeyPair.devicePublicKey,
          app_version: '1.2.3',
        }),
      },
      env,
    );
    expect(firstResponse.status).toBe(200);

    const secondKeyPair = await createDeviceKeyPair();
    const blockedResponse = await app.request(
      'http://broker.test/v1/trial/challenge',
      {
        method: 'POST',
        headers: {
          'content-type': 'application/json',
          'cf-connecting-ip': '203.0.113.72',
        },
        body: JSON.stringify({
          installation_id: 'install-valid-runtime-config-2',
          device_public_key: secondKeyPair.devicePublicKey,
          app_version: '1.2.3',
        }),
      },
      env,
    );

    expect(blockedResponse.status).toBe(429);
  });

  it('seeds the approved warning and automatic-brake defaults', () => {
    const env = createTestBrokerEnv();

    expect(readAbuseControls(env).immediateAlerts).toEqual({
      warning: 10,
      brake: 70,
    });
  });

  it('seeds Discord OAuth endpoint, pending-session, and daily cap defaults', () => {
    const env = createTestBrokerEnv();
    const controls = readAbuseControls(env) as ReturnType<typeof readAbuseControls> &
      Record<string, unknown>;

    expect(controls.discordAuthStartIp).toEqual({
      endpoint: 'POST /v1/auth/discord/start',
      scope: 'ip',
      maxRequests: 20,
      windowMinutes: 15,
    });
    expect(controls.discordAuthStartInstallation).toEqual({
      endpoint: 'POST /v1/auth/discord/start',
      scope: 'installation_id',
      maxRequests: 5,
      windowMinutes: 15,
    });
    expect(controls.discordOpenrouterIssueIp).toEqual({
      endpoint: 'POST /v1/providers/openrouter/discord/issue',
      scope: 'ip',
      maxRequests: 10,
      windowMinutes: 15,
    });
    expect(controls.discordOpenrouterIssueInstallation).toEqual({
      endpoint: 'POST /v1/providers/openrouter/discord/issue',
      scope: 'installation_id',
      maxRequests: 3,
      windowMinutes: 15,
    });
    expect(controls.pendingDiscordOAuthSessions).toEqual({
      maxPerInstallation: 2,
      maxPerIp: 20,
      windowMinutes: 15,
    });
    expect(controls.newActiveEntitlementsPerDay.maxCount).toBe(500);
  });

  it('persists the QQ auth assertion IP endpoint default in abuse controls', () => {
    const env = createTestBrokerEnv();
    const row = env.__db
      .prepare('SELECT value FROM broker_config WHERE key = ?')
      .get('abuse_controls') as { value: string };
    const controls = JSON.parse(row.value) as Record<string, unknown>;

    expect(controls.qqAuthAssertIp).toEqual({
      endpoint: 'POST /v1/auth/qq/assert',
      scope: 'ip',
      maxRequests: 20,
      windowMinutes: 15,
    });
  });
  it('seeds managed-operation and delivery-ACK endpoint rate-limit defaults', () => {
    const env = createTestBrokerEnv();
    const controls = readAbuseControls(env);
    expect(controls.managedOperationStatusIp).toEqual({
      endpoint: 'POST /v1/providers/openrouter/managed-operation/status',
      scope: 'ip',
      maxRequests: 30,
      windowMinutes: 15,
    });
    expect(controls.managedOperationStatusInstallation).toEqual({
      endpoint: 'POST /v1/providers/openrouter/managed-operation/status',
      scope: 'installation_id',
      maxRequests: 30,
      windowMinutes: 15,
    });
    expect(controls.managedOperationResumeIp).toEqual({
      endpoint: 'POST /v1/providers/openrouter/managed-operation/resume',
      scope: 'ip',
      maxRequests: 20,
      windowMinutes: 15,
    });
    expect(controls.managedOperationResumeInstallation).toEqual({
      endpoint: 'POST /v1/providers/openrouter/managed-operation/resume',
      scope: 'installation_id',
      maxRequests: 10,
      windowMinutes: 15,
    });
    expect(controls.managedKeyDeliveryAckIp).toEqual({
      endpoint: 'POST /v1/providers/openrouter/managed-key-delivery/ack',
      scope: 'ip',
      maxRequests: 30,
      windowMinutes: 15,
    });
  });

  it('validates QQ auth assertion overrides and dispatches its IP endpoint rate limit', async () => {
    const env = createTestBrokerEnv();
    updateAbuseControls(env, (controls) => {
      controls.qqAuthAssertIp.maxRequests = 1;
    });

    const controls = await getBrokerAbuseControlsConfig(env.BROKER_DB);
    expect(controls.qqAuthAssertIp).toEqual({
      endpoint: 'POST /v1/auth/qq/assert',
      scope: 'ip',
      maxRequests: 1,
      windowMinutes: 15,
    });

    await seedRequestEvent(env, {
      endpoint: 'POST /v1/auth/qq/assert',
      ip: '203.0.113.98',
      installationId: null,
      observedAt: '2026-06-08T06:00:00.000Z',
    });
    await seedRequestEvent(env, {
      endpoint: 'POST /v1/auth/qq/assert',
      ip: '203.0.113.98',
      installationId: null,
      observedAt: '2026-06-08T06:00:01.000Z',
    });

    await expect(
      checkEndpointRateLimit(env.BROKER_DB, {
        endpoint: 'POST /v1/auth/qq/assert',
        now: new Date('2026-06-08T06:00:02.000Z'),
        ip: '203.0.113.98',
        installationId: null,
        hardwareHash: null,
        networkIdentitySecrets: testNetworkIdentitySecrets(env),
      }),
    ).resolves.toEqual(
      expect.objectContaining({
        status: 429,
        code: 'rate_limited',
        class: 'retryable',
        subcode: 'ip_rate_limited',
      }),
    );
  });

  it('seeds referral attempt, velocity, and retention defaults', async () => {
    const env = createTestBrokerEnv();
    const controls = await getBrokerAbuseControlsConfig(env.BROKER_DB);

    expect(controls.retention).toEqual(
      expect.objectContaining({
        referralSkippedDays: 7,
        referralFailedDays: 30,
      }),
    );
    expect(controls.referralAttempts).toEqual({
      validShaped: {
        maxPerInstallation: 8,
        maxPerIp: 30,
        windowMinutes: 15,
      },
      unknown: {
        maxPerInstallation: 3,
        maxPerIp: 10,
        windowMinutes: 15,
      },
      perReferralIdVelocity: {
        maxAttempts: 25,
        windowMinutes: 60,
      },
      perReferrerRewardVelocity: {
        maxRewards: 5,
        windowMinutes: 1440,
      },
    });
  });

  it('normalizes issue-success retention to the completed-day report minimum', async () => {
    const env = createTestBrokerEnv();
    updateAbuseControls(env, (controls) => {
      controls.trialChallenge.maxRequests = 17;
      controls.retention.issueSuccessDays = 1;
    });

    const controls = await getBrokerAbuseControlsConfig(env.BROKER_DB);

    expect(controls.trialChallenge.maxRequests).toBe(17);
    expect(controls.retention.issueSuccessDays).toBe(2);
  });

  it('falls back to default abuse controls when warning is not below brake', async () => {
    const env = createTestBrokerEnv();
    updateAbuseControls(env, (controls) => {
      controls.immediateAlerts.warning = 70;
      controls.immediateAlerts.brake = 70;
    });

    await expect(getBrokerAbuseControlsConfig(env.BROKER_DB)).resolves.toEqual(
      TEST_DEFAULT_ABUSE_CONTROLS,
    );
  });

  it('seeds exact abuse runtime state defaults and persists runtime-state helper updates', () => {
    const env = createTestBrokerEnv();

    expect(readAbuseRuntimeState(env)).toEqual(TEST_DEFAULT_ABUSE_RUNTIME_STATE);

    updateAbuseRuntimeState(env, (state) => {
      state.brake.active = true;
      state.brake.reason = 'manual';
      state.brake.changedAt = '2026-04-08T06:05:00Z';
      state.brake.changedBy = 'operator';
      state.alertLatches.warning = true;
      state.dailyReport.lastDeliveredAt = '2026-04-08T06:10:00Z';
      state.dailyReport.lastDeliveredDateUtc = '2026-04-08';
    });

    expect(readAbuseRuntimeState(env)).toEqual({
      brake: {
        active: true,
        reason: 'manual',
        changedAt: '2026-04-08T06:05:00Z',
        changedBy: 'operator',
      },
      alertLatches: {
        warning: true,
        warningObservedAt: null,
      },
      dailyReport: {
        lastDeliveredAt: '2026-04-08T06:10:00Z',
        lastDeliveredDateUtc: '2026-04-08',
      },
    });
  });

  it('preserves an active brake when a previous Worker replaces the warning latch object', async () => {
    const env = createTestBrokerEnv();
    env.__db
      .prepare("UPDATE broker_config SET value = ? WHERE key = 'abuse_runtime_state'")
      .run(
        JSON.stringify({
          brake: {
            active: true,
            reason: 'global_threshold',
            changedAt: '2026-04-08T06:05:00.000Z',
            changedBy: 'system',
          },
          alertLatches: {
            warn1: true,
            warn2: false,
            warn3: false,
            critical: false,
          },
          dailyReport: {
            lastDeliveredAt: null,
            lastDeliveredDateUtc: null,
          },
        }),
      );

    await expect(getBrokerAbuseRuntimeState(env.BROKER_DB)).resolves.toMatchObject({
      brake: { active: true, reason: 'global_threshold' },
      alertLatches: { warning: true, warningObservedAt: null },
    });
  });
});
