import { Buffer } from 'node:buffer';

import { afterEach, describe, expect, it, vi } from 'vitest';

import app from '../src/index';
import {
  createDeviceKeyPair,
  signCanonicalIssueRequest,
  signCanonicalStatusRequest,
  signCanonicalVerifyRequest,
} from './test-support/ed25519';
import { normalizedErrorEnvelope } from './test-support/errors';
import {
  createPendingReleaseSession,
  mockOpenRouterManagementApi,
} from './test-support/openrouter-issue';
import { createTestBrokerEnv } from './test-support/sqlite-d1';
import {
  getTrialStatus,
  issueChallenge,
  postIssue,
  postVerify,
} from './test-support/trial-api';

interface ChallengeHardeningCase {
  field: 'installation_id' | 'app_version';
  body: {
    installation_id: string;
    device_public_key: string;
    app_version: string;
  };
  message: string;
}

interface VerifyHardeningCase {
  field: 'installation_id' | 'app_version' | 'hardware_hash';
  overrides: Partial<{
    installation_id: string;
    app_version: string;
    hardware_hash: string;
  }>;
  message: string;
}

function invalidRequestEnvelope(message: string): Record<string, unknown> {
  return normalizedErrorEnvelope({
    code: 'invalid_request',
    class: 'terminal',
    message,
  });
}

describe('broker signed input hardening', () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    vi.useRealTimers();
  });

  describe('POST /v1/trial/challenge', () => {
    it.each([
      {
        field: 'installation_id',
        body: {
          installation_id: 'install-control\nvalue',
          device_public_key: Buffer.alloc(32, 7).toString('base64url'),
          app_version: '1.2.3',
        },
        message: 'installation_id must not contain control characters or newlines',
      },
      {
        field: 'app_version',
        body: {
          installation_id: 'install-control-app-version',
          device_public_key: Buffer.alloc(32, 7).toString('base64url'),
          app_version: '1.2.3\tbeta',
        },
        message: 'app_version must not contain control characters or newlines',
      },
      {
        field: 'installation_id',
        body: {
          installation_id: '   ',
          device_public_key: Buffer.alloc(32, 7).toString('base64url'),
          app_version: '1.2.3',
        },
        message: 'installation_id must not be blank or whitespace-only',
      },
      {
        field: 'app_version',
        body: {
          installation_id: 'install-whitespace-app-version',
          device_public_key: Buffer.alloc(32, 7).toString('base64url'),
          app_version: '   ',
        },
        message: 'app_version must not be blank or whitespace-only',
      },
    ] satisfies ChallengeHardeningCase[])('rejects $field values that contain control characters or newlines', async ({
      body,
      message,
    }: ChallengeHardeningCase) => {
      const env = createTestBrokerEnv();

      const routeResponse = await app.request(
        'http://broker.test/v1/trial/challenge',
        {
          method: 'POST',
          headers: {
            'content-type': 'application/json',
          },
          body: JSON.stringify(body),
        },
        env,
      );

      expect(routeResponse.status).toBe(400);
      await expect(routeResponse.json()).resolves.toEqual(
        invalidRequestEnvelope(message),
      );

      const installationCount = env.__db
        .prepare('SELECT COUNT(*) AS count FROM installations')
        .get() as { count: number };
      expect(installationCount.count).toBe(0);
    });
  });

  describe('POST /v1/trial/challenge/verify', () => {
    it.each([
      {
        field: 'installation_id',
        overrides: {
          installation_id: 'install-verify\ncontrol',
        },
        message: 'installation_id must not contain control characters or newlines',
      },
      {
        field: 'app_version',
        overrides: {
          app_version: '2.0.0\tbeta',
        },
        message: 'app_version must not contain control characters or newlines',
      },
      {
        field: 'hardware_hash',
        overrides: {
          hardware_hash: 'hardware-hash\rvalue',
        },
        message: 'hardware_hash must not contain control characters or newlines',
      },
      {
        field: 'installation_id',
        overrides: {
          installation_id: '   ',
        },
        message: 'installation_id must not be blank or whitespace-only',
      },
      {
        field: 'app_version',
        overrides: {
          app_version: '   ',
        },
        message: 'app_version must not be blank or whitespace-only',
      },
      {
        field: 'hardware_hash',
        overrides: {
          hardware_hash: '   ',
        },
        message: 'hardware_hash must not be blank or whitespace-only',
      },
    ] satisfies VerifyHardeningCase[])(
      'rejects signed $field values that contain control characters or newlines before challenge consumption',
      async ({ overrides, message }: VerifyHardeningCase) => {
        vi.useFakeTimers();
        vi.setSystemTime(new Date('2026-04-08T06:00:00Z'));

        const env = createTestBrokerEnv();
        const keyPair = await createDeviceKeyPair();
        const challenge = await issueChallenge({
          env,
          installationId: 'install-verify-signed-input-hardening',
          devicePublicKey: keyPair.devicePublicKey,
          appVersion: '1.2.3',
        });

        const requestBody = await signCanonicalVerifyRequest(keyPair.privateKey, {
          installation_id: 'install-verify-signed-input-hardening',
          device_public_key: keyPair.devicePublicKey,
          challenge: challenge.challenge,
          challenge_expires_at: challenge.challenge_expires_at,
          hardware_hash: 'hardware-hash-verify-signed-input-hardening',
          app_version: '1.2.3',
          signed_at: '2026-04-08T06:00:30.000Z',
          ...overrides,
        });

        const response = await postVerify(env, requestBody);

        expect(response.status).toBe(400);
        await expect(response.json()).resolves.toEqual(
          invalidRequestEnvelope(message),
        );

        const installation = env.__db
          .prepare(
            `SELECT challenge, challenge_expires_at, hardware_hash, app_version
               FROM installations
              WHERE installation_id = ?`,
          )
          .get('install-verify-signed-input-hardening') as Record<string, unknown>;

        expect(installation).toEqual({
          challenge: challenge.challenge,
          challenge_expires_at: challenge.challenge_expires_at,
          hardware_hash: null,
          app_version: '1.2.3',
        });
      },
    );
  });

  describe('POST /v1/providers/openrouter/issue', () => {
    it('rejects oversized installation_id values before release-token lookup', async () => {
      vi.useFakeTimers();
      vi.setSystemTime(new Date('2026-04-08T06:00:00Z'));

      const env = createTestBrokerEnv();
      const release = await createPendingReleaseSession({
        env,
        installationId: 'install-issue-signed-input-hardening',
        appVersion: '1.2.3',
        hardwareHash: 'hardware-hash-issue-signed-input-hardening',
      });
      const requestBody = await signCanonicalIssueRequest(release.keyPair.privateKey, {
        installation_id: 'i'.repeat(129),
        device_public_key: release.keyPair.devicePublicKey,
        release_token: release.releaseToken,
        hardware_hash: release.hardwareHash,
        reason: 'llm_start',
        budget_usd: 0.07,
        model: 'google/gemma-4-26b-a4b-it',
        signed_at: '2026-04-08T06:00:45.000Z',
      });

      const response = await postIssue(env, requestBody);

      expect(response.status).toBe(400);
      await expect(response.json()).resolves.toEqual(
        invalidRequestEnvelope(
          'installation_id must be between 1 and 128 characters',
        ),
      );

      const entitlement = env.__db
        .prepare(
          `SELECT status, managed_credential_ref, issued_at, expires_at
             FROM openrouter_entitlements
            WHERE installation_id = ?`,
        )
        .get('install-issue-signed-input-hardening') as Record<string, unknown>;

      expect(entitlement).toEqual({
        status: 'pending_release',
        managed_credential_ref: null,
        issued_at: null,
        expires_at: null,
      });
    });

    it('rejects control characters or newlines in installation_id before release-token lookup', async () => {
      vi.useFakeTimers();
      vi.setSystemTime(new Date('2026-04-08T06:00:00Z'));

      const env = createTestBrokerEnv();
      const release = await createPendingReleaseSession({
        env,
        installationId: 'install-issue-signed-input-hardening-control',
        appVersion: '1.2.3',
        hardwareHash: 'hardware-hash-issue-signed-input-hardening-control',
      });
      const requestBody = await signCanonicalIssueRequest(release.keyPair.privateKey, {
        installation_id: 'install-issue\ncontrol',
        device_public_key: release.keyPair.devicePublicKey,
        release_token: release.releaseToken,
        hardware_hash: release.hardwareHash,
        reason: 'llm_start',
        budget_usd: 0.07,
        model: 'google/gemma-4-26b-a4b-it',
        signed_at: '2026-04-08T06:00:45.000Z',
      });

      const response = await postIssue(env, requestBody);

      expect(response.status).toBe(400);
      await expect(response.json()).resolves.toEqual(
        invalidRequestEnvelope(
          'installation_id must not contain control characters or newlines',
        ),
      );

      const entitlement = env.__db
        .prepare(
          `SELECT status, managed_credential_ref, issued_at, expires_at
             FROM openrouter_entitlements
            WHERE installation_id = ?`,
        )
        .get('install-issue-signed-input-hardening-control') as Record<string, unknown>;

      expect(entitlement).toEqual({
        status: 'pending_release',
        managed_credential_ref: null,
        issued_at: null,
        expires_at: null,
      });
    });

    it('rejects whitespace-only installation_id before release-token lookup', async () => {
      vi.useFakeTimers();
      vi.setSystemTime(new Date('2026-04-08T06:00:00Z'));

      const env = createTestBrokerEnv();
      const release = await createPendingReleaseSession({
        env,
        installationId: 'install-issue-signed-input-hardening-whitespace',
        appVersion: '1.2.3',
        hardwareHash: 'hardware-hash-issue-signed-input-hardening-whitespace',
      });
      const requestBody = await signCanonicalIssueRequest(release.keyPair.privateKey, {
        installation_id: '   ',
        device_public_key: release.keyPair.devicePublicKey,
        release_token: release.releaseToken,
        hardware_hash: release.hardwareHash,
        reason: 'llm_start',
        budget_usd: 0.07,
        model: 'google/gemma-4-26b-a4b-it',
        signed_at: '2026-04-08T06:00:45.000Z',
      });

      const response = await postIssue(env, requestBody);

      expect(response.status).toBe(400);
      await expect(response.json()).resolves.toEqual(
        invalidRequestEnvelope(
          'installation_id must not be blank or whitespace-only',
        ),
      );
    });

    it('rejects control characters or newlines in hardware_hash before release-token lookup', async () => {
      vi.useFakeTimers();
      vi.setSystemTime(new Date('2026-04-08T06:00:00Z'));

      const env = createTestBrokerEnv();
      const release = await createPendingReleaseSession({
        env,
        installationId: 'install-issue-signed-input-hardening-hardware-control',
        appVersion: '1.2.3',
        hardwareHash: 'hardware-hash-issue-signed-input-hardening-hardware-control',
      });
      const requestBody = await signCanonicalIssueRequest(release.keyPair.privateKey, {
        installation_id: 'install-issue-signed-input-hardening-hardware-control',
        device_public_key: release.keyPair.devicePublicKey,
        release_token: release.releaseToken,
        hardware_hash: 'hardware-hash\rcontrol',
        reason: 'llm_start',
        budget_usd: 0.07,
        model: 'google/gemma-4-26b-a4b-it',
        signed_at: '2026-04-08T06:00:45.000Z',
      });

      const response = await postIssue(env, requestBody);

      expect(response.status).toBe(400);
      await expect(response.json()).resolves.toEqual(
        invalidRequestEnvelope(
          'hardware_hash must not contain control characters or newlines',
        ),
      );
    });

    it('rejects issue when the installation hardware snapshot becomes stale after verify', async () => {
      vi.useFakeTimers();
      vi.setSystemTime(new Date('2026-04-08T06:00:00Z'));

      const managementApi = mockOpenRouterManagementApi();
      const env = createTestBrokerEnv();
      const release = await createPendingReleaseSession({
        env,
        installationId: 'install-issue-stale-snapshot',
        appVersion: '1.2.3',
        hardwareHash: 'hardware-hash-issue-stale-snapshot',
      });

      env.__db
        .prepare(
          `UPDATE installations
              SET hardware_hash_salt_version = ?
            WHERE installation_id = ?`,
        )
        .run(8, 'install-issue-stale-snapshot');

      const requestBody = await signCanonicalIssueRequest(release.keyPair.privateKey, {
        installation_id: 'install-issue-stale-snapshot',
        device_public_key: release.keyPair.devicePublicKey,
        release_token: release.releaseToken,
        hardware_hash: release.hardwareHash,
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
  });

  describe('GET /v1/trial/status', () => {
    it('rejects control characters or newlines in installation_id before installation lookup', async () => {
      vi.useFakeTimers();
      vi.setSystemTime(new Date('2026-04-08T06:00:00Z'));

      const env = createTestBrokerEnv();
      const keyPair = await createDeviceKeyPair();
      const signedRequest = await signCanonicalStatusRequest(keyPair.privateKey, {
        installation_id: 'install-status\ncontrol',
        timestamp: '2026-04-08T06:00:30.000Z',
      });

      const response = await getTrialStatus({
        env,
        installationId: 'install-status\ncontrol',
        headers: {
          'X-Puripuly-Timestamp': signedRequest.timestamp,
          'X-Puripuly-Signature': signedRequest.signature,
        },
      });

      expect(response.status).toBe(400);
      await expect(response.json()).resolves.toEqual(
        invalidRequestEnvelope(
          'installation_id must not contain control characters or newlines',
        ),
      );
    });

    it('rejects whitespace-only installation_id before installation lookup', async () => {
      vi.useFakeTimers();
      vi.setSystemTime(new Date('2026-04-08T06:00:00Z'));

      const env = createTestBrokerEnv();
      const keyPair = await createDeviceKeyPair();
      const signedRequest = await signCanonicalStatusRequest(keyPair.privateKey, {
        installation_id: '   ',
        timestamp: '2026-04-08T06:00:30.000Z',
      });

      const response = await getTrialStatus({
        env,
        installationId: '   ',
        headers: {
          'X-Puripuly-Timestamp': signedRequest.timestamp,
          'X-Puripuly-Signature': signedRequest.signature,
        },
      });

      expect(response.status).toBe(400);
      await expect(response.json()).resolves.toEqual(
        invalidRequestEnvelope(
          'installation_id must not be blank or whitespace-only',
        ),
      );
  });
  });
});
