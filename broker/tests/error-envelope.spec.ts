import { afterEach, describe, expect, it, vi } from 'vitest';

import app from '../src/index';
import {
  createDeviceKeyPair,
  signCanonicalIssueRequest,
} from './test-support/ed25519';
import {
  createPendingReleaseSession,
  mockOpenRouterManagementApi,
} from './test-support/openrouter-issue';
import { updateAbuseRuntimeState } from './test-support/abuse-controls';
import { createTestBrokerEnv } from './test-support/sqlite-d1';
import { postIssue } from './test-support/trial-api';
import { normalizedErrorEnvelope } from './test-support/errors';

describe('broker public error envelope', () => {
  afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
    vi.useRealTimers();
  });

  it('normalizes release-token expiry into challenge_expired with bounded code/class/subcode fields', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-04-08T06:00:00Z'));

    const env = createTestBrokerEnv();
    const release = await createPendingReleaseSession({
      env,
      installationId: 'install-error-envelope-expired-release',
      appVersion: '1.2.3',
      hardwareHash: 'hardware-hash-error-envelope-expired-release',
    });

    vi.setSystemTime(new Date('2026-04-08T06:15:01Z'));

    const requestBody = await signCanonicalIssueRequest(release.keyPair.privateKey, {
      installation_id: 'install-error-envelope-expired-release',
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
    await expect(response.json()).resolves.toEqual({
      error: {
        code: 'challenge_expired',
        class: 'retryable',
        subcode: 'release_token_expired',
        retry_after_ms: 0,
        message: 'release_token has expired and must be reissued',
      },
      managed_state: {
        lifecycle: 'pending_release',
        managed_availability: true,
      },
      current_entitlement: {
        provider: 'OpenRouter',
        budget_usd: 0.07,
        issued_at: null,
        expires_at: null,
      },
    });
  });

  it('returns internal_error and leaves a structured orphan-audit trail when activation cleanup fails', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-04-08T06:00:00Z'));

    const consoleErrorSpy = vi.spyOn(console, 'error').mockImplementation(() => undefined);
    const env = createTestBrokerEnv({
      beforeRun({ sql }) {
        if (sql.includes('SET status = ?')) {
          throw new Error('simulated activation failure');
        }
      },
    });
    const managementApi = mockOpenRouterManagementApi({ mode: 'cleanup_failure' });
    const release = await createPendingReleaseSession({
      env,
      installationId: 'install-error-envelope-cleanup-failure',
      appVersion: '1.2.3',
      hardwareHash: 'hardware-hash-error-envelope-cleanup-failure',
    });
    const requestBody = await signCanonicalIssueRequest(release.keyPair.privateKey, {
      installation_id: 'install-error-envelope-cleanup-failure',
      device_public_key: release.keyPair.devicePublicKey,
      release_token: release.releaseToken,
      hardware_hash: release.hardwareHash,
      reason: 'llm_start',
      budget_usd: 0.07,
      model: 'google/gemma-4-26b-a4b-it',
      signed_at: '2026-04-08T06:00:45.000Z',
    });

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
    expect(consoleErrorSpy).toHaveBeenCalledWith(
      'managed_child_key_orphan_audit',
      expect.objectContaining({
        installation_id: 'install-error-envelope-cleanup-failure',
        release_session_ref: expect.any(String),
        managed_credential_ref: managementApi.childKey.hash,
      }),
    );
  });

  it('fails closed and logs orphan-audit detail when child-key creation is ambiguous', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-04-08T06:00:00Z'));

    const consoleErrorSpy = vi.spyOn(console, 'error').mockImplementation(() => undefined);
    const env = createTestBrokerEnv();
    const managementApi = mockOpenRouterManagementApi({ mode: 'malformed_create' });
    const release = await createPendingReleaseSession({
      env,
      installationId: 'install-error-envelope-ambiguous-create',
      appVersion: '1.2.3',
      hardwareHash: 'hardware-hash-error-envelope-ambiguous-create',
    });
    const requestBody = await signCanonicalIssueRequest(release.keyPair.privateKey, {
      installation_id: 'install-error-envelope-ambiguous-create',
      device_public_key: release.keyPair.devicePublicKey,
      release_token: release.releaseToken,
      hardware_hash: release.hardwareHash,
      reason: 'llm_start',
      budget_usd: 0.07,
      model: 'google/gemma-4-26b-a4b-it',
      signed_at: '2026-04-08T06:00:45.000Z',
    });

    const firstResponse = await postIssue(env, requestBody);
    const secondResponse = await postIssue(env, requestBody);

    expect(firstResponse.status).toBe(500);
    expect(secondResponse.status).toBe(401);
    expect(consoleErrorSpy).toHaveBeenCalledWith(
      'managed_child_key_orphan_audit',
      expect.objectContaining({
        installation_id: 'install-error-envelope-ambiguous-create',
        release_session_ref: expect.any(String),
        managed_credential_ref: null,
        creation_failure: expect.objectContaining({
          operation: 'create_key',
          code: 'malformed_upstream',
        }),
      }),
    );
    expect(
      managementApi.fetchMock.mock.calls.filter(
        ([url]) => String(url).includes('/keys/') && String(url) !== 'https://openrouter.ai/api/v1/keys',
      ),
    ).toHaveLength(0);

    const entitlement = env.__db
      .prepare(
        `SELECT status, managed_credential_ref, release_session_ref, release_token_hash, release_token_expires_at
           FROM openrouter_entitlements
          WHERE installation_id = ?`,
      )
      .get('install-error-envelope-ambiguous-create') as Record<string, unknown>;

    expect(entitlement).toEqual({
      status: 'pending_release',
      managed_credential_ref: null,
      release_session_ref: null,
      release_token_hash: null,
      release_token_expires_at: null,
    });
  });

  it('normalizes brake-driven challenge rejection into issuance_suspended with the brake reason subcode', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-04-08T06:00:00Z'));

    const env = createTestBrokerEnv();
    const blockedKeyPair = await createDeviceKeyPair();
    updateAbuseRuntimeState(env, (state) => {
      state.brake.active = true;
      state.brake.reason = 'asn_fast_path';
      state.brake.changedAt = '2026-04-08T06:00:05.000Z';
      state.brake.changedBy = 'system';
    });

    const response = await app.request(
      'http://broker.test/v1/trial/challenge',
      {
        method: 'POST',
        headers: {
          'content-type': 'application/json',
          'cf-connecting-ip': '203.0.113.44',
        },
        body: JSON.stringify({
          installation_id: 'install-error-envelope-brake',
          device_public_key: blockedKeyPair.devicePublicKey,
          app_version: '1.2.3',
        }),
      },
      env,
    );

    expect(response.status).toBe(503);
    await expect(response.json()).resolves.toEqual(
      normalizedErrorEnvelope({
        code: 'issuance_suspended',
        class: 'retryable',
        subcode: 'asn_fast_path',
        message: 'new entitlement issuance is temporarily suspended',
      }),
    );
  });
});
