import { afterEach, describe, expect, it, vi } from 'vitest';

import * as abuseMonitoring from '../src/abuse-monitoring';
import { signCanonicalIssueRequest } from './test-support/ed25519';
import { normalizedErrorEnvelope } from './test-support/errors';
import {
  createPendingReleaseSession,
  mockOpenRouterManagementApi,
} from './test-support/openrouter-issue';
import { sha256Base64Url } from './test-support/hash';
import { createTestBrokerEnv } from './test-support/sqlite-d1';
import {
  postIssue,
  postIssueWithExecutionContext,
} from './test-support/trial-api';

describe('POST /v1/providers/openrouter/issue route contract', () => {
  afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
    vi.useRealTimers();
  });

  it('consumes a pending_release token and activates the entitlement', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-04-08T06:00:00Z'));

    const managementApi = mockOpenRouterManagementApi();
    const env = createTestBrokerEnv();
    const release = await createPendingReleaseSession({
      env,
      installationId: 'install-issue-route',
      appVersion: '1.2.3',
      hardwareHash: 'hardware-hash-issue-route',
    });
    const requestBody = await signCanonicalIssueRequest(release.keyPair.privateKey, {
      installation_id: 'install-issue-route',
      device_public_key: release.keyPair.devicePublicKey,
      release_token: release.releaseToken,
      hardware_hash: release.hardwareHash,
      reason: 'llm_start',
      budget_usd: 0.07,
      model: 'google/gemma-4-26b-a4b-it',
      signed_at: '2026-04-08T06:00:45.000Z',
    });

    const queuedSideEffects: Promise<unknown>[] = [];
    const executionCtx = {
      props: {},
      waitUntil(promise: Promise<unknown>) {
        queuedSideEffects.push(promise);
      },
      passThroughOnException() {},
    };

    const response = await postIssueWithExecutionContext(
      env,
      requestBody,
      executionCtx,
    );
    expect(response.status).toBe(200);

    const payload = (await response.json()) as {
      openrouter_api_key: string;
      managed_credential_ref: string;
      expires_at: string;
      budget_usd: number;
      model: string;
      managed_state: {
        lifecycle: string;
        managed_availability: boolean;
      };
    };
    const entitlement = env.__db
      .prepare(
        `SELECT status, budget_usd, managed_credential_ref, issued_at, expires_at,
                release_session_ref, release_token_hash, release_token_expires_at
           FROM openrouter_entitlements
          WHERE installation_id = ?`,
      )
      .get('install-issue-route') as Record<string, unknown>;

    expect(payload.openrouter_api_key).toBe(managementApi.childKey.rawKey);
    expect(payload.managed_credential_ref).toBe(managementApi.childKey.hash);
    expect(payload.managed_state).toEqual({
      lifecycle: 'active',
      managed_availability: true,
    });
    expect(payload.expires_at).toBe('2026-07-08T06:00:00.000Z');
    expect(payload.budget_usd).toBe(0.07);
    expect(payload.model).toBe('google/gemma-4-26b-a4b-it');
    expect(entitlement).toEqual({
      status: 'active',
      budget_usd: 0.07,
      managed_credential_ref: managementApi.childKey.hash,
      issued_at: '2026-04-08T06:00:00.000Z',
      expires_at: '2026-07-08T06:00:00.000Z',
      release_session_ref: expect.any(String),
      release_token_hash: expect.any(String),
      release_token_expires_at: release.releaseTokenExpiresAt,
    });
    expect(payload.managed_credential_ref).not.toBe(payload.openrouter_api_key);
    expect(payload.openrouter_api_key).not.toBe(env.OPENROUTER_MANAGEMENT_API_KEY);
    await expect(sha256Base64Url(release.releaseToken)).resolves.toBe(
      entitlement.release_token_hash,
    );
    const issueSuccessCount = env.__db
      .prepare('SELECT COUNT(*) AS count FROM broker_issue_success_events')
      .get() as { count: number };
    expect(issueSuccessCount.count).toBe(1);
    expect(queuedSideEffects).toHaveLength(1);
    await Promise.all(queuedSideEffects);
    expect(managementApi.fetchMock).toHaveBeenCalledTimes(2);
  });

  it('fails closed and rolls the activation back out when post-activation monitoring cannot update abuse state', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-04-08T06:05:00Z'));

    const consoleErrorSpy = vi.spyOn(console, 'error').mockImplementation(() => undefined);
    const managementApi = mockOpenRouterManagementApi();
    const env = createTestBrokerEnv();
    const release = await createPendingReleaseSession({
      env,
      installationId: 'install-issue-monitoring-failure',
      appVersion: '1.2.3',
      hardwareHash: 'hardware-hash-issue-monitoring-failure',
      verifySignedAt: '2026-04-08T06:05:30.000Z',
    });
    const requestBody = await signCanonicalIssueRequest(release.keyPair.privateKey, {
      installation_id: 'install-issue-monitoring-failure',
      device_public_key: release.keyPair.devicePublicKey,
      release_token: release.releaseToken,
      hardware_hash: release.hardwareHash,
      reason: 'llm_start',
      budget_usd: 0.07,
      model: 'google/gemma-4-26b-a4b-it',
      signed_at: '2026-04-08T06:05:45.000Z',
    });
    vi.spyOn(abuseMonitoring, 'recordIssueSuccess').mockRejectedValueOnce(
      new Error('issue success event insert failed'),
    );

    const response = await postIssue(env, requestBody);
    expect(response.status).toBe(500);
    await expect(response.json()).resolves.toEqual(
      normalizedErrorEnvelope({
        code: 'internal_error',
        class: 'retryable',
        message: 'broker encountered an unexpected internal error',
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

    const retryResponse = await postIssue(env, requestBody);
    expect(retryResponse.status).toBe(401);
    await expect(retryResponse.json()).resolves.toEqual(
      normalizedErrorEnvelope({
        code: 'challenge_invalid',
        class: 'security_fail',
        subcode: 'release_token_invalid',
        message: 'release_token does not match the active release session for installation_id',
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

    const entitlement = env.__db
      .prepare(
        `SELECT status, managed_credential_ref, issued_at, expires_at,
                release_session_ref, release_token_hash, release_token_expires_at
           FROM openrouter_entitlements
           WHERE installation_id = ?`,
      )
      .get('install-issue-monitoring-failure') as Record<string, unknown>;
    expect(entitlement).toEqual({
      status: 'pending_release',
      managed_credential_ref: null,
      issued_at: null,
      expires_at: null,
      release_session_ref: null,
      release_token_hash: null,
      release_token_expires_at: null,
    });

    const issueSuccessCount = env.__db
      .prepare('SELECT COUNT(*) AS count FROM broker_issue_success_events')
      .get() as { count: number };
    expect(issueSuccessCount.count).toBe(0);
    expect(managementApi.fetchMock).toHaveBeenCalledTimes(4);
    expect(consoleErrorSpy).toHaveBeenCalledWith(
      'post_activation_monitoring_failed',
      expect.objectContaining({
        installation_id: 'install-issue-monitoring-failure',
        managed_credential_ref: managementApi.childKey.hash,
        stage: 'record_or_evaluate',
      }),
    );
  });

  it('rejects issue when the request hardware_hash does not match the verified snapshot', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-04-08T06:00:00Z'));

    const managementApi = mockOpenRouterManagementApi();
    const env = createTestBrokerEnv();
    const release = await createPendingReleaseSession({
      env,
      installationId: 'install-issue-hardware-mismatch',
      appVersion: '1.2.3',
      hardwareHash: 'hardware-hash-verified',
    });
    const requestBody = await signCanonicalIssueRequest(release.keyPair.privateKey, {
      installation_id: 'install-issue-hardware-mismatch',
      device_public_key: release.keyPair.devicePublicKey,
      release_token: release.releaseToken,
      hardware_hash: 'hardware-hash-different',
      reason: 'llm_start',
      budget_usd: 0.07,
      model: 'google/gemma-4-26b-a4b-it',
      signed_at: '2026-04-08T06:00:45.000Z',
    });

    const response = await postIssue(env, requestBody);

    expect(response.status).toBe(409);
    await expect(response.json()).resolves.toEqual(
      normalizedErrorEnvelope({
        code: 'trial_not_eligible',
        class: 'terminal',
        subcode: 'hardware_duplicate',
        message: 'hardware_hash no longer matches the verified release session',
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
  });

  it('rejects expired release tokens without mutating pending_release state', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-04-08T06:00:00Z'));

    const env = createTestBrokerEnv();
    const release = await createPendingReleaseSession({
      env,
      installationId: 'install-issue-expired',
      appVersion: '1.2.3',
      hardwareHash: 'hardware-hash-issue-expired',
    });

    vi.setSystemTime(new Date('2026-04-08T06:15:01Z'));

    const requestBody = await signCanonicalIssueRequest(release.keyPair.privateKey, {
      installation_id: 'install-issue-expired',
      device_public_key: release.keyPair.devicePublicKey,
      release_token: release.releaseToken,
      hardware_hash: release.hardwareHash,
      reason: 'llm_start',
      budget_usd: 0.07,
      model: 'google/gemma-4-26b-a4b-it',
      signed_at: '2026-04-08T06:15:00.000Z',
    });

    const response = await postIssue(env, requestBody);
    expect(response.status).toBe(410);
    await expect(response.json()).resolves.toEqual(
      normalizedErrorEnvelope({
        code: 'challenge_expired',
        class: 'retryable',
        subcode: 'release_token_expired',
        retryAfterMs: 0,
        message: 'release_token has expired and must be reissued',
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

    const entitlement = env.__db
      .prepare(
        `SELECT status, managed_credential_ref, issued_at, expires_at,
                release_token_hash, release_token_expires_at
           FROM openrouter_entitlements
          WHERE installation_id = ?`,
      )
      .get('install-issue-expired') as Record<string, unknown>;

    expect(entitlement).toEqual({
      status: 'pending_release',
      managed_credential_ref: null,
      issued_at: null,
      expires_at: null,
      release_token_hash: expect.any(String),
      release_token_expires_at: release.releaseTokenExpiresAt,
    });
  });

  it.each([
    { discordIssueStatus: 'issuing', managedCredentialRef: null },
    {
      discordIssueStatus: 'cleanup_required',
      managedCredentialRef: 'hash_orphaned_discord_cleanup_required_issue',
    },
  ] satisfies Array<{
    discordIssueStatus: 'issuing' | 'cleanup_required';
    managedCredentialRef: string | null;
  }>)(
    'rejects legacy issue for Discord $discordIssueStatus pending_release without activating or mutating entitlement metadata',
    async ({ discordIssueStatus, managedCredentialRef }) => {
      vi.useFakeTimers();
      vi.setSystemTime(new Date('2026-04-08T06:00:00Z'));

      const managementApi = mockOpenRouterManagementApi();
      const env = createTestBrokerEnv();
      const installationId = `install-discord-${discordIssueStatus}-issue-guard`;
      const release = await createPendingReleaseSession({
        env,
        installationId,
        appVersion: '1.2.3',
        hardwareHash: `hardware-hash-${discordIssueStatus}-issue-guard`,
      });
      env.__db
        .prepare(
          `UPDATE openrouter_entitlements
              SET discord_issue_status = ?,
                  discord_issue_reserved_at = ?,
                  managed_credential_ref = ?
            WHERE installation_id = ?`,
        )
        .run(
          discordIssueStatus,
          '2026-04-08T06:00:00.000Z',
          managedCredentialRef,
          installationId,
        );
      const entitlementBefore = env.__db
        .prepare(
          `SELECT status, managed_credential_ref, issued_at, expires_at,
                  release_session_ref, release_token_hash, release_token_expires_at,
                  verified_hardware_hash, verified_hardware_hash_salt_version,
                  discord_issue_status, discord_issue_reserved_at
             FROM openrouter_entitlements
            WHERE installation_id = ?`,
        )
        .get(installationId) as Record<string, unknown>;
      const requestBody = await signCanonicalIssueRequest(release.keyPair.privateKey, {
        installation_id: installationId,
        device_public_key: release.keyPair.devicePublicKey,
        release_token: release.releaseToken,
        hardware_hash: release.hardwareHash,
        reason: 'llm_start',
        budget_usd: 0.07,
        model: 'google/gemma-4-26b-a4b-it',
        signed_at: '2026-04-08T06:00:45.000Z',
      });

      const response = await postIssue(env, requestBody);

      expect(response.status).toBe(401);
      await expect(response.json()).resolves.toMatchObject({
        error: {
          code: 'challenge_invalid',
          class: 'security_fail',
          subcode: 'release_token_invalid',
        },
      });
      expect(managementApi.fetchMock).not.toHaveBeenCalled();
      const entitlementAfter = env.__db
        .prepare(
          `SELECT status, managed_credential_ref, issued_at, expires_at,
                  release_session_ref, release_token_hash, release_token_expires_at,
                  verified_hardware_hash, verified_hardware_hash_salt_version,
                  discord_issue_status, discord_issue_reserved_at
             FROM openrouter_entitlements
            WHERE installation_id = ?`,
        )
        .get(installationId) as Record<string, unknown>;
      expect(entitlementAfter).toEqual(entitlementBefore);
    },
  );

  it('rejects non-object JSON bodies with invalid_request', async () => {
    const env = createTestBrokerEnv();
    const response = await postIssue(env, 'null');

    expect(response.status).toBe(400);
    await expect(response.json()).resolves.toEqual(
      normalizedErrorEnvelope({
        code: 'invalid_request',
        class: 'terminal',
        message: 'request body must be a JSON object',
      }),
    );
  });
});
