import { afterEach, describe, expect, it, vi } from 'vitest';

import app from '../src/index';
import {
  checkDailyIssuanceCap,
  getManagedDailyIssuanceCapState,
} from '../src/abuse-controls';
import {
  createDeviceKeyPair,
  signCanonicalIssueRequest,
  signCanonicalStatusRequest,
  signCanonicalVerifyRequest,
} from './test-support/ed25519';
import {
  activatePendingReleaseSession,
  createPendingReleaseSession,
  mockOpenRouterManagementApi,
} from './test-support/openrouter-issue';
import { createTestBrokerEnv } from './test-support/sqlite-d1';
import { getTrialStatus, issueChallenge, postIssue, postVerify } from './test-support/trial-api';
import {
  updateAbuseControls,
  updateAbuseRuntimeState,
} from './test-support/abuse-controls';
import { normalizedErrorEnvelope } from './test-support/errors';

describe('broker daily issuance cap enforcement', () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    vi.useRealTimers();
  });

  it('returns issuance_suspended from challenge once the daily new-active cap is reached while active status remains available', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-04-08T06:00:00Z'));

    const env = createTestBrokerEnv();
    updateAbuseControls(env, (controls) => {
      controls.newActiveEntitlementsPerDay.maxCount = 1;
    });

    const active = await activatePendingReleaseSession({
      env,
      installationId: 'install-cap-active',
      appVersion: '1.2.3',
      hardwareHash: 'hardware-hash-cap-active',
    });
    expect(active.response.status).toBe(200);

    const blockedKeyPair = await createDeviceKeyPair();
    const blockedResponse = await app.request(
      'http://broker.test/v1/trial/challenge',
      {
        method: 'POST',
        headers: {
          'content-type': 'application/json',
          'cf-connecting-ip': '203.0.113.30',
        },
        body: JSON.stringify({
          installation_id: 'install-cap-blocked-challenge',
          device_public_key: blockedKeyPair.devicePublicKey,
          app_version: '1.2.3',
        }),
      },
      env,
    );

    expect(blockedResponse.status).toBe(503);
    await expect(blockedResponse.json()).resolves.toEqual({
      error: {
        code: 'issuance_suspended',
        class: 'retryable',
        subcode: 'global_cap_reached',
        retry_after_ms: 64800000,
        message: 'new entitlement issuance is temporarily suspended',
      },
      managed_state: {
        lifecycle: 'none',
        managed_availability: true,
      },
      current_entitlement: null,
    });

    const signedStatus = await signCanonicalStatusRequest(active.keyPair.privateKey, {
      installation_id: 'install-cap-active',
      timestamp: '2026-04-08T06:00:30.000Z',
    });
    const statusResponse = await getTrialStatus({
      env,
      installationId: 'install-cap-active',
      headers: {
        'X-Puripuly-Timestamp': signedStatus.timestamp,
        'X-Puripuly-Signature': signedStatus.signature,
      },
    });

    expect(statusResponse.status).toBe(200);
    await expect(statusResponse.json()).resolves.toEqual({
      managed_state: {
        lifecycle: 'active',
        managed_availability: true,
      },
      current_entitlement: {
        provider: 'OpenRouter',
        budget_usd: 0.07,
        issued_at: '2026-04-08T06:00:00.000Z',
        expires_at: '2026-07-08T06:00:00.000Z',
      },
      onboarding_eligibility: {
        eligible: false,
        reason: 'active',
        requires_discord_oauth: false,
      },
    });
  });

  it('rechecks the cap at verify time but still allows issue for already-verified pending_release installations', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-04-08T06:00:00Z'));

    const env = createTestBrokerEnv();
    updateAbuseControls(env, (controls) => {
      controls.newActiveEntitlementsPerDay.maxCount = 1;
    });

    const waitingKeyPair = await createDeviceKeyPair();
    const waitingChallenge = await issueChallenge({
      env,
      installationId: 'install-cap-waiting',
      devicePublicKey: waitingKeyPair.devicePublicKey,
      appVersion: '1.2.3',
    });

    const pendingRelease = await createPendingReleaseSession({
      env,
      installationId: 'install-cap-pending',
      appVersion: '1.2.3',
      hardwareHash: 'hardware-hash-cap-pending',
    });
    const active = await activatePendingReleaseSession({
      env,
      installationId: 'install-cap-active-second',
      appVersion: '1.2.3',
      hardwareHash: 'hardware-hash-cap-active-second',
    });
    expect(active.response.status).toBe(200);

    const waitingVerify = await signCanonicalVerifyRequest(waitingKeyPair.privateKey, {
      installation_id: 'install-cap-waiting',
      device_public_key: waitingKeyPair.devicePublicKey,
      challenge: waitingChallenge.challenge,
      challenge_expires_at: waitingChallenge.challenge_expires_at,
      hardware_hash: 'hardware-hash-cap-waiting',
      app_version: '1.2.3',
      signed_at: '2026-04-08T06:00:30.000Z',
    });
    const blockedVerifyResponse = await postVerify(env, waitingVerify);

    expect(blockedVerifyResponse.status).toBe(503);
    await expect(blockedVerifyResponse.json()).resolves.toEqual({
      error: {
        code: 'issuance_suspended',
        class: 'retryable',
        subcode: 'global_cap_reached',
        retry_after_ms: 64800000,
        message: 'new entitlement issuance is temporarily suspended',
      },
      managed_state: {
        lifecycle: 'none',
        managed_availability: true,
      },
      current_entitlement: null,
    });

    const persistedWaitingInstallation = env.__db
      .prepare(
        `SELECT challenge, challenge_expires_at, hardware_hash
           FROM installations
          WHERE installation_id = ?`,
      )
      .get('install-cap-waiting') as Record<string, unknown>;
    expect(persistedWaitingInstallation).toEqual({
      challenge: waitingChallenge.challenge,
      challenge_expires_at: waitingChallenge.challenge_expires_at,
      hardware_hash: null,
    });

    const pendingIssueRequest = await signCanonicalIssueRequest(
      pendingRelease.keyPair.privateKey,
      {
        installation_id: 'install-cap-pending',
        device_public_key: pendingRelease.keyPair.devicePublicKey,
        release_token: pendingRelease.releaseToken,
        hardware_hash: pendingRelease.hardwareHash,
        reason: 'llm_start',
        budget_usd: 0.07,
        model: 'google/gemma-4-26b-a4b-it',
        signed_at: '2026-04-08T06:00:45.000Z',
      },
    );
    const pendingIssueResponse = await postIssue(env, pendingIssueRequest);

    expect(pendingIssueResponse.status).toBe(200);
    await expect(pendingIssueResponse.json()).resolves.toEqual(
      expect.objectContaining({
        managed_state: {
          lifecycle: 'active',
          managed_availability: true,
        },
        expires_at: '2026-07-08T06:00:00.000Z',
        budget_usd: 0.07,
      }),
    );
  });

  it('still counts same-day issuances after the entitlement is later revoked', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-04-08T06:00:00Z'));

    const env = createTestBrokerEnv();
    updateAbuseControls(env, (controls) => {
      controls.newActiveEntitlementsPerDay.maxCount = 1;
    });

    const active = await activatePendingReleaseSession({
      env,
      installationId: 'install-cap-revoked-source',
      appVersion: '1.2.3',
      hardwareHash: 'hardware-hash-cap-revoked-source',
    });
    expect(active.response.status).toBe(200);

    env.__db
      .prepare(
        `UPDATE openrouter_entitlements
            SET status = 'revoked',
                release_session_ref = NULL,
                release_token_hash = NULL,
                release_token_expires_at = NULL
          WHERE installation_id = ?`,
      )
      .run('install-cap-revoked-source');

    const blockedKeyPair = await createDeviceKeyPair();
    const blockedResponse = await app.request(
      'http://broker.test/v1/trial/challenge',
      {
        method: 'POST',
        headers: {
          'content-type': 'application/json',
          'cf-connecting-ip': '203.0.113.32',
        },
        body: JSON.stringify({
          installation_id: 'install-cap-revoked-target',
          device_public_key: blockedKeyPair.devicePublicKey,
          app_version: '1.2.3',
        }),
      },
      env,
    );

    expect(blockedResponse.status).toBe(503);
    await expect(blockedResponse.json()).resolves.toEqual({
      error: {
        code: 'issuance_suspended',
        class: 'retryable',
        subcode: 'global_cap_reached',
        retry_after_ms: 64800000,
        message: 'new entitlement issuance is temporarily suspended',
      },
      managed_state: {
        lifecycle: 'none',
        managed_availability: true,
      },
      current_entitlement: null,
    });
  });

  it('resets the cap at the next UTC day boundary instead of using a rolling 24-hour window', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-04-08T23:55:00Z'));

    const env = createTestBrokerEnv();
    updateAbuseControls(env, (controls) => {
      controls.newActiveEntitlementsPerDay.maxCount = 1;
    });

    const active = await activatePendingReleaseSession({
      env,
      installationId: 'install-cap-utc-reset-source',
      appVersion: '1.2.3',
      hardwareHash: 'hardware-hash-cap-utc-reset-source',
      verifySignedAt: '2026-04-08T23:55:30.000Z',
      issueSignedAt: '2026-04-08T23:55:45.000Z',
    });
    expect(active.response.status).toBe(200);

    vi.setSystemTime(new Date('2026-04-09T00:05:00Z'));

    const nextDayKeyPair = await createDeviceKeyPair();
    const nextDayResponse = await app.request(
      'http://broker.test/v1/trial/challenge',
      {
        method: 'POST',
        headers: {
          'content-type': 'application/json',
          'cf-connecting-ip': '203.0.113.33',
        },
        body: JSON.stringify({
          installation_id: 'install-cap-utc-reset-target',
          device_public_key: nextDayKeyPair.devicePublicKey,
          app_version: '1.2.3',
        }),
      },
      env,
    );

    expect(nextDayResponse.status).toBe(200);
    await expect(nextDayResponse.json()).resolves.toEqual(
      expect.objectContaining({
        challenge: expect.any(String),
        managed_state: {
          lifecycle: 'none',
          managed_availability: true,
        },
        current_entitlement: null,
      }),
    );
  });

  it('returns issuance_suspended from challenge when the automatic brake is active while active status remains available', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-04-08T06:00:00Z'));

    const env = createTestBrokerEnv();
    const active = await activatePendingReleaseSession({
      env,
      installationId: 'install-brake-active',
      appVersion: '1.2.3',
      hardwareHash: 'hardware-hash-brake-active',
    });
    expect(active.response.status).toBe(200);

    updateAbuseRuntimeState(env, (state) => {
      state.brake.active = true;
      state.brake.reason = 'global_threshold';
      state.brake.changedAt = '2026-04-08T06:01:00.000Z';
      state.brake.changedBy = 'system';
    });

    const blockedKeyPair = await createDeviceKeyPair();
    const blockedResponse = await app.request(
      'http://broker.test/v1/trial/challenge',
      {
        method: 'POST',
        headers: {
          'content-type': 'application/json',
          'cf-connecting-ip': '203.0.113.30',
        },
        body: JSON.stringify({
          installation_id: 'install-brake-blocked-challenge',
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

    const signedStatus = await signCanonicalStatusRequest(active.keyPair.privateKey, {
      installation_id: 'install-brake-active',
      timestamp: '2026-04-08T06:00:30.000Z',
    });
    const statusResponse = await getTrialStatus({
      env,
      installationId: 'install-brake-active',
      headers: {
        'X-Puripuly-Timestamp': signedStatus.timestamp,
        'X-Puripuly-Signature': signedStatus.signature,
      },
    });

    expect(statusResponse.status).toBe(200);
    await expect(statusResponse.json()).resolves.toEqual({
      managed_state: {
        lifecycle: 'active',
        managed_availability: true,
      },
      current_entitlement: {
        provider: 'OpenRouter',
        budget_usd: 0.07,
        issued_at: '2026-04-08T06:00:00.000Z',
        expires_at: '2026-07-08T06:00:00.000Z',
      },
      onboarding_eligibility: {
        eligible: false,
        reason: 'active',
        requires_discord_oauth: false,
      },
    });
  });

  it('rechecks the automatic brake at verify time and blocks new issue responses for pending_release installations', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-04-08T06:00:00Z'));

    const env = createTestBrokerEnv();

    const waitingKeyPair = await createDeviceKeyPair();
    const waitingChallenge = await issueChallenge({
      env,
      installationId: 'install-brake-waiting',
      devicePublicKey: waitingKeyPair.devicePublicKey,
      appVersion: '1.2.3',
    });

    const pendingRelease = await createPendingReleaseSession({
      env,
      installationId: 'install-brake-pending',
      appVersion: '1.2.3',
      hardwareHash: 'hardware-hash-brake-pending',
    });

    updateAbuseRuntimeState(env, (state) => {
      state.brake.active = true;
      state.brake.reason = 'asn_fast_path';
      state.brake.changedAt = '2026-04-08T06:01:00.000Z';
      state.brake.changedBy = 'system';
    });
    const managementApi = mockOpenRouterManagementApi();

    const waitingVerify = await signCanonicalVerifyRequest(waitingKeyPair.privateKey, {
      installation_id: 'install-brake-waiting',
      device_public_key: waitingKeyPair.devicePublicKey,
      challenge: waitingChallenge.challenge,
      challenge_expires_at: waitingChallenge.challenge_expires_at,
      hardware_hash: 'hardware-hash-brake-waiting',
      app_version: '1.2.3',
      signed_at: '2026-04-08T06:00:30.000Z',
    });
    const blockedVerifyResponse = await postVerify(env, waitingVerify);

    expect(blockedVerifyResponse.status).toBe(503);
    await expect(blockedVerifyResponse.json()).resolves.toEqual(
      normalizedErrorEnvelope({
        code: 'issuance_suspended',
        class: 'retryable',
        subcode: 'asn_fast_path',
        message: 'new entitlement issuance is temporarily suspended',
      }),
    );

    const pendingIssueRequest = await signCanonicalIssueRequest(
      pendingRelease.keyPair.privateKey,
      {
        installation_id: 'install-brake-pending',
        device_public_key: pendingRelease.keyPair.devicePublicKey,
        release_token: pendingRelease.releaseToken,
        hardware_hash: pendingRelease.hardwareHash,
        reason: 'llm_start',
        budget_usd: 0.07,
        model: 'google/gemma-4-26b-a4b-it',
        signed_at: '2026-04-08T06:00:45.000Z',
      },
    );
    const pendingIssueResponse = await postIssue(env, pendingIssueRequest);

    expect(pendingIssueResponse.status).toBe(503);
    await expect(pendingIssueResponse.json()).resolves.toEqual(
      normalizedErrorEnvelope({
        code: 'issuance_suspended',
        class: 'retryable',
        subcode: 'asn_fast_path',
        message: 'new entitlement issuance is temporarily suspended',
        managedState: {
          lifecycle: 'pending_release',
          managed_availability: true,
        },
        currentEntitlement: {
          provider: 'OpenRouter',
          budget_usd: 0.07,
          issued_at: null,
          expires_at: null,
        },
      }),
    );
    expect(managementApi.fetchMock).not.toHaveBeenCalled();

    const pendingEntitlement = env.__db
      .prepare(
        `SELECT status, managed_credential_ref, issued_at, expires_at,
                release_token_hash, release_token_expires_at
           FROM openrouter_entitlements
          WHERE installation_id = ?`,
      )
      .get('install-brake-pending') as Record<string, unknown>;

    expect(pendingEntitlement).toEqual({
      status: 'pending_release',
      managed_credential_ref: null,
      issued_at: null,
      expires_at: null,
      release_token_hash: expect.any(String),
      release_token_expires_at: pendingRelease.releaseTokenExpiresAt,
    });
  });

  it.each(['issuing', 'active', 'cleanup_required'] as const)(
    'counts QQ %s entitlements toward the global daily managed issuance cap',
    async (status) => {
      vi.useFakeTimers();
      vi.setSystemTime(new Date('2026-04-08T06:00:00Z'));

      const env = createTestBrokerEnv();
      updateAbuseControls(env, (controls) => {
        controls.newActiveEntitlementsPerDay.maxCount = 1;
      });
      insertQqManagedEntitlement(env, {
        status,
        qqSubjectRef: `ph-qq-subject-v1_synthetic-cap-${status}`,
        issueRef: `qq-issue-cap-${status}`,
        reservedAt: '2026-04-08T05:59:00.000Z',
        issuedAt: status === 'active' ? '2026-04-08T06:00:00.000Z' : null,
        expiresAt: status === 'active' ? '2026-07-08T06:00:00.000Z' : null,
        deliveredAt: status === 'active' ? '2026-04-08T06:00:00.000Z' : null,
        managedCredentialRef:
          status === 'issuing' ? null : `hash_qq_managed_cap_${status}`,
      });

      const decision = await checkDailyIssuanceCap(
        env.BROKER_DB,
        new Date('2026-04-08T06:00:00.000Z'),
        null,
      );

      expect(decision).toEqual({
        status: 503,
        code: 'issuance_suspended',
        class: 'retryable',
        message: 'new entitlement issuance is temporarily suspended',
        subcode: 'global_cap_reached',
        retryAfterMs: 64_800_000,
      });
      expect(
        env.__db
          .prepare('SELECT COUNT(*) AS count FROM openrouter_entitlements')
          .get(),
      ).toEqual({ count: 0 });
    },
  );

  it('can exclude the current QQ reservation by source-aware subject and issue refs for post-reservation cap checks', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-04-08T06:00:00Z'));

    const env = createTestBrokerEnv();
    updateAbuseControls(env, (controls) => {
      controls.newActiveEntitlementsPerDay.maxCount = 1;
    });
    const qqSubjectRef = 'ph-qq-subject-v1_synthetic-cap-current';
    const issueRef = 'qq-issue-cap-current';
    insertQqManagedEntitlement(env, {
      status: 'issuing',
      qqSubjectRef,
      issueRef,
      reservedAt: '2026-04-08T05:59:00.000Z',
    });

    const cap = await getManagedDailyIssuanceCapState(
      env.BROKER_DB,
      new Date('2026-04-08T06:00:00.000Z'),
      {
        excludeCurrent: {
          issueSource: 'qq',
          subjectRef: qqSubjectRef,
          issueRef,
        },
      },
    );

    expect(cap).toEqual({
      reached: false,
      retryAfterMs: null,
      count: 0,
      maxCount: 1,
    });
  });
});

function insertQqManagedEntitlement(
  env: ReturnType<typeof createTestBrokerEnv>,
  input: {
    qqSubjectRef: string;
    status: 'issuing' | 'delivery_pending' | 'active' | 'cleanup_required';
    issueRef: string;
    reservedAt: string;
    issuedAt?: string | null;
    expiresAt?: string | null;
    deliveredAt?: string | null;
    managedCredentialRef?: string | null;
  },
): void {
  env.__db
    .prepare(
      `INSERT INTO qq_managed_entitlements (
          qq_subject_ref,
          status,
          issue_ref,
          managed_credential_ref,
          budget_usd,
          reserved_at,
          issued_at,
          expires_at,
          delivered_at,
          created_at,
          updated_at
        ) VALUES (?, ?, ?, ?, 0.07, ?, ?, ?, ?, ?, ?)`,
    )
    .run(
      input.qqSubjectRef,
      input.status,
      input.issueRef,
      input.managedCredentialRef ?? null,
      input.reservedAt,
      input.issuedAt ?? null,
      input.expiresAt ?? null,
      input.deliveredAt ?? null,
      input.reservedAt,
      input.reservedAt,
    );
}
