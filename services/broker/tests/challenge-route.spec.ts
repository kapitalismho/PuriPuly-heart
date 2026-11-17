import { Buffer } from 'node:buffer';

import { afterEach, describe, expect, it, vi } from 'vitest';

import app from '../src/index';
import {
  createDeviceKeyPair,
  signCanonicalIssueRequest,
  signCanonicalVerifyRequest,
} from './test-support/ed25519';
import { normalizedErrorEnvelope } from './test-support/errors';
import { createPendingReleaseSession } from './test-support/openrouter-issue';
import { createTestBrokerEnv, insertEntitlement } from './test-support/sqlite-d1';
import { issueChallenge, postIssue, postVerify } from './test-support/trial-api';

const DEVICE_PUBLIC_KEY = Buffer.alloc(32, 7).toString('base64url');

const OVERSIZED_INSTALLATION_ID = 'i'.repeat(129);
const OVERSIZED_APP_VERSION = 'v'.repeat(65);

interface ChallengeBoundsCase {
  name: 'installation_id' | 'app_version';
  body: {
    installation_id: string;
    device_public_key: string;
    app_version: string;
  };
  message: string;
}

type DiscordBlockedIssueStatus = 'issuing' | 'cleanup_required';

function createDeferred(): {
  promise: Promise<void>;
  resolve: () => void;
} {
  let resolve!: () => void;
  return {
    promise: new Promise<void>((resolvePromise) => {
      resolve = resolvePromise;
    }),
    resolve,
  };
}

function readEntitlementSnapshot(
  env: ReturnType<typeof createTestBrokerEnv>,
  installationId: string,
): Record<string, unknown> {
  return env.__db
    .prepare(
      `SELECT status, managed_credential_ref, release_session_ref,
              release_token_hash, release_token_expires_at,
              verified_hardware_hash, verified_hardware_hash_salt_version,
              discord_issue_status, discord_issue_reserved_at
         FROM openrouter_entitlements
        WHERE installation_id = ?`,
    )
    .get(installationId) as Record<string, unknown>;
}

describe('POST /v1/trial/challenge', () => {
  afterEach(() => {
    vi.useRealTimers();
  });

  it('accepts installation identity preflight input and returns a non-consuming challenge contract', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-04-08T06:00:00Z'));

    const env = createTestBrokerEnv();
    const response = await app.request(
      'http://broker.test/v1/trial/challenge',
      {
        method: 'POST',
        headers: {
          'content-type': 'application/json',
        },
        body: JSON.stringify({
          installation_id: 'install-123',
          device_public_key: DEVICE_PUBLIC_KEY,
          app_version: '1.2.3',
        }),
      },
      env,
    );

    expect(response.status).toBe(200);

    const payload = (await response.json()) as Record<string, unknown>;

    expect(payload).toEqual({
      challenge: expect.stringMatching(/^[A-Za-z0-9_-]{43}$/),
      challenge_expires_at: '2026-04-08T06:05:00.000Z',
      fingerprint_salt: {
        version: 7,
        salt: 'shared-server-fingerprint-salt',
      },
      managed_state: {
        lifecycle: 'none',
        managed_availability: true,
      },
      current_entitlement: null,
    });
    expect(payload).not.toHaveProperty('release_token');
    expect(payload).not.toHaveProperty('managed_credential_ref');

    const installation = env.__db
      .prepare(
        `SELECT installation_id, device_public_key, hardware_hash, app_version, challenge,
                challenge_expires_at, challenge_salt_version
           FROM installations
          WHERE installation_id = ?`,
      )
      .get('install-123') as Record<string, unknown>;

    expect(installation).toEqual({
      installation_id: 'install-123',
      device_public_key: DEVICE_PUBLIC_KEY,
      hardware_hash: null,
      app_version: '1.2.3',
      challenge: payload.challenge,
      challenge_expires_at: '2026-04-08T06:05:00.000Z',
      challenge_salt_version: 7,
    });
  });

  it('rejects hardware_hash and client signature fields on challenge preflight', async () => {
    const env = createTestBrokerEnv();
    const response = await app.request(
      'http://broker.test/v1/trial/challenge',
      {
        method: 'POST',
        headers: {
          'content-type': 'application/json',
        },
        body: JSON.stringify({
          installation_id: 'install-123',
          device_public_key: DEVICE_PUBLIC_KEY,
          app_version: '1.2.3',
          hardware_hash: 'forbidden-on-challenge',
          signature: 'forbidden-on-challenge',
        }),
      },
      env,
    );

    expect(response.status).toBe(400);
    await expect(response.json()).resolves.toEqual(
      normalizedErrorEnvelope({
        code: 'invalid_request',
        class: 'terminal',
        message:
          'challenge request must not include hardware_hash, signed_at, or signature',
      }),
    );

    const installationCount = env.__db
      .prepare('SELECT COUNT(*) AS count FROM installations')
      .get() as { count: number };

    expect(installationCount.count).toBe(0);
  });

  it('rejects non-object JSON bodies with invalid_request instead of throwing', async () => {
    const env = createTestBrokerEnv();
    const response = await app.request(
      'http://broker.test/v1/trial/challenge',
      {
        method: 'POST',
        headers: {
          'content-type': 'application/json',
        },
        body: 'null',
      },
      env,
    );

    expect(response.status).toBe(400);
    await expect(response.json()).resolves.toEqual(
      normalizedErrorEnvelope({
        code: 'invalid_request',
        class: 'terminal',
        message: 'request body must be a JSON object',
      }),
    );
  });

  it.each([
    {
      name: 'installation_id',
      body: {
        installation_id: OVERSIZED_INSTALLATION_ID,
        device_public_key: DEVICE_PUBLIC_KEY,
        app_version: '1.2.3',
      },
      message: 'installation_id must be between 1 and 128 characters',
    },
    {
      name: 'app_version',
      body: {
        installation_id: 'install-oversized-app-version',
        device_public_key: DEVICE_PUBLIC_KEY,
        app_version: OVERSIZED_APP_VERSION,
      },
      message: 'app_version must be between 1 and 64 characters',
    },
  ] satisfies ChallengeBoundsCase[])(
    'rejects oversized $name values before persistence',
    async (testCase: ChallengeBoundsCase) => {
      const { body, message } = testCase;
    const env = createTestBrokerEnv();
    const response = await app.request(
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

    expect(response.status).toBe(400);
    await expect(response.json()).resolves.toEqual(
      normalizedErrorEnvelope({
        code: 'invalid_request',
        class: 'terminal',
        message,
      }),
    );

    const installationCount = env.__db
      .prepare('SELECT COUNT(*) AS count FROM installations')
      .get() as { count: number };
    expect(installationCount.count).toBe(0);
    },
  );

  it('clears stored hardware hash state when reissuing a challenge', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-04-08T06:00:00Z'));

    const env = createTestBrokerEnv();
    const keyPair = await createDeviceKeyPair();
    const firstChallenge = await issueChallenge({
      env,
      installationId: 'install-reissue-clears-hash',
      devicePublicKey: keyPair.devicePublicKey,
      appVersion: '1.2.3',
    });
    const verifyRequest = await signCanonicalVerifyRequest(keyPair.privateKey, {
      installation_id: 'install-reissue-clears-hash',
      device_public_key: keyPair.devicePublicKey,
      challenge: firstChallenge.challenge,
      challenge_expires_at: firstChallenge.challenge_expires_at,
      hardware_hash: 'hardware-hash-before-reissue',
      app_version: '1.2.3',
      signed_at: '2026-04-08T06:00:30.000Z',
    });

    const verifyResponse = await postVerify(env, verifyRequest);
    expect(verifyResponse.status).toBe(200);

    vi.setSystemTime(new Date('2026-04-08T06:01:00Z'));

    const response = await app.request(
      'http://broker.test/v1/trial/challenge',
      {
        method: 'POST',
        headers: {
          'content-type': 'application/json',
        },
        body: JSON.stringify({
          installation_id: 'install-reissue-clears-hash',
          device_public_key: keyPair.devicePublicKey,
          app_version: '1.2.4',
        }),
      },
      env,
    );

    expect(response.status).toBe(200);

    const installation = env.__db
      .prepare(
        `SELECT hardware_hash, hardware_hash_salt_version, app_version, challenge,
                challenge_expires_at, challenge_salt_version
           FROM installations
          WHERE installation_id = ?`,
      )
      .get('install-reissue-clears-hash') as Record<string, unknown>;

    expect(installation).toEqual({
      hardware_hash: null,
      hardware_hash_salt_version: null,
      app_version: '1.2.4',
      challenge: expect.any(String),
      challenge_expires_at: '2026-04-08T06:06:00.000Z',
      challenge_salt_version: 7,
    });
  });

  it('clears stale pending release session tokens and issue locks while preserving the verified hardware snapshot when reissuing a challenge', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-04-08T06:00:00Z'));

    const env = createTestBrokerEnv();
    const keyPair = await createDeviceKeyPair();
    const firstChallenge = await issueChallenge({
      env,
      installationId: 'install-reissue-clears-release-session',
      devicePublicKey: keyPair.devicePublicKey,
      appVersion: '1.2.3',
    });
    const verifyRequest = await signCanonicalVerifyRequest(keyPair.privateKey, {
      installation_id: 'install-reissue-clears-release-session',
      device_public_key: keyPair.devicePublicKey,
      challenge: firstChallenge.challenge,
      challenge_expires_at: firstChallenge.challenge_expires_at,
      hardware_hash: 'hardware-hash-before-release-session-reissue',
      app_version: '1.2.3',
      signed_at: '2026-04-08T06:00:30.000Z',
    });

    const verifyResponse = await postVerify(env, verifyRequest);
    expect(verifyResponse.status).toBe(200);

    env.__db
      .prepare(
        `UPDATE openrouter_entitlements
            SET managed_credential_ref = ?
          WHERE installation_id = ?`,
      )
      .run('__issue_lock__:stale-release-session-lock', 'install-reissue-clears-release-session');

    vi.setSystemTime(new Date('2026-04-08T06:01:00Z'));

    const response = await app.request(
      'http://broker.test/v1/trial/challenge',
      {
        method: 'POST',
        headers: {
          'content-type': 'application/json',
        },
        body: JSON.stringify({
          installation_id: 'install-reissue-clears-release-session',
          device_public_key: keyPair.devicePublicKey,
          app_version: '1.2.4',
        }),
      },
      env,
    );

    expect(response.status).toBe(200);

    const entitlement = env.__db
      .prepare(
        `SELECT status, release_session_ref, release_token_hash, release_token_expires_at,
                managed_credential_ref, verified_hardware_hash, verified_hardware_hash_salt_version
           FROM openrouter_entitlements
           WHERE installation_id = ?`,
      )
      .get('install-reissue-clears-release-session') as Record<string, unknown>;

    expect(entitlement).toEqual({
      status: 'pending_release',
      release_session_ref: null,
      release_token_hash: null,
      release_token_expires_at: null,
      managed_credential_ref: null,
      verified_hardware_hash: 'hardware-hash-before-release-session-reissue',
      verified_hardware_hash_salt_version: 7,
    });
  });

  it.each([
    { discordIssueStatus: 'issuing', managedCredentialRef: null },
    {
      discordIssueStatus: 'cleanup_required',
      managedCredentialRef: 'hash_orphaned_discord_cleanup_required_challenge',
    },
  ] satisfies Array<{
    discordIssueStatus: DiscordBlockedIssueStatus;
    managedCredentialRef: string | null;
  }>)(
    'rejects challenge reissue for Discord $discordIssueStatus pending_release without mutating entitlement metadata',
    async ({ discordIssueStatus, managedCredentialRef }) => {
      vi.useFakeTimers();
      vi.setSystemTime(new Date('2026-04-08T06:00:00Z'));

      const env = createTestBrokerEnv();
      const keyPair = await createDeviceKeyPair();
      await issueChallenge({
        env,
        installationId: `install-discord-${discordIssueStatus}-challenge-guard`,
        devicePublicKey: keyPair.devicePublicKey,
        appVersion: '1.2.3',
      });
      insertEntitlement(env, {
        installation_id: `install-discord-${discordIssueStatus}-challenge-guard`,
        status: 'pending_release',
        budget_usd: 0.07,
        managed_credential_ref: managedCredentialRef,
        release_session_ref: `release-session-${discordIssueStatus}-challenge-guard`,
        release_token_hash: `release-token-hash-${discordIssueStatus}-challenge-guard`,
        release_token_expires_at: '2026-04-08T06:10:00.000Z',
        verified_hardware_hash: `hardware-hash-${discordIssueStatus}-challenge-guard`,
        verified_hardware_hash_salt_version: 7,
        discord_issue_status: discordIssueStatus,
        discord_issue_reserved_at: '2026-04-08T06:00:00.000Z',
      });
      const before = readEntitlementSnapshot(
        env,
        `install-discord-${discordIssueStatus}-challenge-guard`,
      );

      vi.setSystemTime(new Date('2026-04-08T06:01:00Z'));

      const response = await app.request(
        'http://broker.test/v1/trial/challenge',
        {
          method: 'POST',
          headers: {
            'content-type': 'application/json',
          },
          body: JSON.stringify({
            installation_id: `install-discord-${discordIssueStatus}-challenge-guard`,
            device_public_key: keyPair.devicePublicKey,
            app_version: '1.2.4',
          }),
        },
        env,
      );

      expect(response.status).toBe(409);
      expect(
        readEntitlementSnapshot(
          env,
          `install-discord-${discordIssueStatus}-challenge-guard`,
        ),
      ).toEqual(before);
    },
  );

  it('rejects stale release tokens while challenge rotation is still in flight', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-04-08T06:00:00Z'));

    let armChallengeUpdatePause = false;
    let pausedOnce = false;
    const challengeUpdatePaused = createDeferred();
    const releaseChallengeUpdate = createDeferred();
    const env = createTestBrokerEnv({
      beforeRun: async ({ sql }) => {
        if (
          armChallengeUpdatePause &&
          !pausedOnce &&
          sql.includes('UPDATE installations') &&
          sql.includes('challenge = ?')
        ) {
          pausedOnce = true;
          challengeUpdatePaused.resolve();
          await releaseChallengeUpdate.promise;
        }
      },
    });

    const release = await createPendingReleaseSession({
      env,
      installationId: 'install-reissue-invalidates-stale-token',
      appVersion: '1.2.3',
      hardwareHash: 'hardware-hash-before-reissue-race',
    });

    vi.setSystemTime(new Date('2026-04-08T06:01:00Z'));
    armChallengeUpdatePause = true;

    const reissueResponsePromise = app.request(
      'http://broker.test/v1/trial/challenge',
      {
        method: 'POST',
        headers: {
          'content-type': 'application/json',
        },
        body: JSON.stringify({
          installation_id: 'install-reissue-invalidates-stale-token',
          device_public_key: release.keyPair.devicePublicKey,
          app_version: '1.2.4',
        }),
      },
      env,
    );

    await challengeUpdatePaused.promise;

    const entitlementWhileRotating = env.__db
      .prepare(
        `SELECT release_session_ref, release_token_hash, release_token_expires_at,
                verified_hardware_hash, verified_hardware_hash_salt_version
           FROM openrouter_entitlements
          WHERE installation_id = ?`,
      )
      .get('install-reissue-invalidates-stale-token') as Record<string, unknown>;

    expect(entitlementWhileRotating).toEqual({
      release_session_ref: null,
      release_token_hash: null,
      release_token_expires_at: null,
      verified_hardware_hash: 'hardware-hash-before-reissue-race',
      verified_hardware_hash_salt_version: 7,
    });

    const staleIssueRequest = await signCanonicalIssueRequest(release.keyPair.privateKey, {
      installation_id: 'install-reissue-invalidates-stale-token',
      device_public_key: release.keyPair.devicePublicKey,
      release_token: release.releaseToken,
      hardware_hash: 'hardware-hash-before-reissue-race',
      reason: 'llm_start',
      budget_usd: 0.07,
      model: 'google/gemma-4-26b-a4b-it',
      signed_at: '2026-04-08T06:01:00.000Z',
    });

    const staleIssueResponse = await postIssue(env, staleIssueRequest);

    expect(staleIssueResponse.status).toBe(401);
    await expect(staleIssueResponse.json()).resolves.toMatchObject({
      error: {
        code: 'challenge_invalid',
        class: 'security_fail',
        subcode: 'release_token_invalid',
      },
    });

    releaseChallengeUpdate.resolve();

    const reissueResponse = await reissueResponsePromise;
    expect(reissueResponse.status).toBe(200);
  });

  it('handles first-time same-binding challenge races without surfacing a 500 and keeps one installation row', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-04-08T06:00:00Z'));

    let armInsertPause = false;
    let insertAttemptCount = 0;
    const firstInsertReady = createDeferred();
    const secondInsertReady = createDeferred();
    const releaseBothInserts = createDeferred();
    const env = createTestBrokerEnv({
      beforeRun: async ({ sql }) => {
        if (armInsertPause && sql.includes('INSERT INTO installations')) {
          insertAttemptCount += 1;
          if (insertAttemptCount === 1) {
            firstInsertReady.resolve();
          }
          if (insertAttemptCount === 2) {
            secondInsertReady.resolve();
          }
          await releaseBothInserts.promise;
        }
      },
    });

    armInsertPause = true;
    const firstResponsePromise = app.request(
      'http://broker.test/v1/trial/challenge',
      {
        method: 'POST',
        headers: {
          'content-type': 'application/json',
        },
        body: JSON.stringify({
          installation_id: 'install-race-challenge',
          device_public_key: DEVICE_PUBLIC_KEY,
          app_version: '1.2.3',
        }),
      },
      env,
    );

    await firstInsertReady.promise;

    const secondResponsePromise = app.request(
      'http://broker.test/v1/trial/challenge',
      {
        method: 'POST',
        headers: {
          'content-type': 'application/json',
        },
        body: JSON.stringify({
          installation_id: 'install-race-challenge',
          device_public_key: DEVICE_PUBLIC_KEY,
          app_version: '1.2.3',
        }),
      },
      env,
    );

    await secondInsertReady.promise;
    releaseBothInserts.resolve();

    const [firstResponse, secondResponse] = await Promise.all([
      firstResponsePromise,
      secondResponsePromise,
    ]);

    expect(firstResponse.status).toBe(200);
    expect(secondResponse.status).toBe(200);

    const firstPayload = (await firstResponse.json()) as Record<string, unknown>;
    const secondPayload = (await secondResponse.json()) as Record<string, unknown>;

    expect(firstPayload).toMatchObject({
      challenge: expect.stringMatching(/^[A-Za-z0-9_-]{43}$/),
      challenge_expires_at: '2026-04-08T06:05:00.000Z',
    });
    expect(secondPayload).toMatchObject({
      challenge: expect.stringMatching(/^[A-Za-z0-9_-]{43}$/),
      challenge_expires_at: '2026-04-08T06:05:00.000Z',
    });

    const installationCount = env.__db
      .prepare('SELECT COUNT(*) AS count FROM installations')
      .get() as { count: number };
    expect(installationCount.count).toBe(1);

    const installation = env.__db
      .prepare(
        `SELECT installation_id, device_public_key, app_version, challenge,
                challenge_expires_at
           FROM installations
          WHERE installation_id = ?`,
      )
      .get('install-race-challenge') as Record<string, unknown>;

    expect(installation.installation_id).toBe('install-race-challenge');
    expect(installation.device_public_key).toBe(DEVICE_PUBLIC_KEY);
    expect(installation.app_version).toBe('1.2.3');
    expect(installation.challenge_expires_at).toBe('2026-04-08T06:05:00.000Z');
    expect(firstPayload.challenge).toBe(installation.challenge);
    expect(secondPayload.challenge).toBe(installation.challenge);
  });

  it('reclaims stale preflight-only rows by installation_id after the retention window elapses', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-04-08T06:00:00Z'));

    const env = createTestBrokerEnv();
    env.__db
      .prepare(
        `INSERT INTO installations (
            installation_id,
            device_public_key,
            app_version,
            challenge,
            challenge_expires_at,
            challenge_salt_version,
            created_at,
            last_seen_at
          ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
      )
      .run(
        'install-stale-preflight-id',
        Buffer.alloc(32, 5).toString('base64url'),
        '1.0.0',
        'stale-challenge-token',
        '2026-04-06T06:05:00.000Z',
        7,
        '2026-04-06T06:00:00.000Z',
        '2026-04-06T06:00:00.000Z',
      );

    const response = await app.request(
      'http://broker.test/v1/trial/challenge',
      {
        method: 'POST',
        headers: {
          'content-type': 'application/json',
        },
        body: JSON.stringify({
          installation_id: 'install-stale-preflight-id',
          device_public_key: DEVICE_PUBLIC_KEY,
          app_version: '2.0.0',
        }),
      },
      env,
    );

    expect(response.status).toBe(200);

    const installation = env.__db
      .prepare(
        `SELECT installation_id, device_public_key, app_version, challenge_expires_at
           FROM installations
          WHERE installation_id = ?`,
      )
      .get('install-stale-preflight-id') as Record<string, unknown>;

    expect(installation).toEqual({
      installation_id: 'install-stale-preflight-id',
      device_public_key: DEVICE_PUBLIC_KEY,
      app_version: '2.0.0',
      challenge_expires_at: '2026-04-08T06:05:00.000Z',
    });
  });

  it('reclaims stale preflight-only rows by device_public_key after the retention window elapses', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-04-08T06:00:00Z'));

    const env = createTestBrokerEnv();
    env.__db
      .prepare(
        `INSERT INTO installations (
            installation_id,
            device_public_key,
            app_version,
            challenge,
            challenge_expires_at,
            challenge_salt_version,
            created_at,
            last_seen_at
          ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
      )
      .run(
        'install-stale-preflight-public-key',
        DEVICE_PUBLIC_KEY,
        '1.0.0',
        'stale-challenge-token',
        '2026-04-06T06:05:00.000Z',
        7,
        '2026-04-06T06:00:00.000Z',
        '2026-04-06T06:00:00.000Z',
      );

    const response = await app.request(
      'http://broker.test/v1/trial/challenge',
      {
        method: 'POST',
        headers: {
          'content-type': 'application/json',
        },
        body: JSON.stringify({
          installation_id: 'install-fresh-after-stale-public-key',
          device_public_key: DEVICE_PUBLIC_KEY,
          app_version: '2.0.0',
        }),
      },
      env,
    );

    expect(response.status).toBe(200);

    const installation = env.__db
      .prepare(
        `SELECT installation_id, device_public_key, app_version, challenge_expires_at
           FROM installations
          WHERE installation_id = ?`,
      )
      .get('install-fresh-after-stale-public-key') as Record<string, unknown>;

    expect(installation).toEqual({
      installation_id: 'install-fresh-after-stale-public-key',
      device_public_key: DEVICE_PUBLIC_KEY,
      app_version: '2.0.0',
      challenge_expires_at: '2026-04-08T06:05:00.000Z',
    });
  });

  it('does not reclaim a preflight-only row before its challenge_expires_at boundary', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-04-08T06:00:00Z'));

    const env = createTestBrokerEnv();
    env.__db
      .prepare(
        `INSERT INTO installations (
            installation_id,
            device_public_key,
            app_version,
            challenge,
            challenge_expires_at,
            challenge_salt_version,
            created_at,
            last_seen_at
          ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
      )
      .run(
        'install-preflight-boundary',
        Buffer.alloc(32, 4).toString('base64url'),
        '1.0.0',
        'boundary-challenge-token',
        '2026-04-08T06:00:01.000Z',
        7,
        '2026-04-06T06:00:00.000Z',
        '2026-04-06T06:00:00.000Z',
      );

    const response = await app.request(
      'http://broker.test/v1/trial/challenge',
      {
        method: 'POST',
        headers: {
          'content-type': 'application/json',
        },
        body: JSON.stringify({
          installation_id: 'install-preflight-boundary',
          device_public_key: DEVICE_PUBLIC_KEY,
          app_version: '2.0.0',
        }),
      },
      env,
    );

    expect(response.status).toBe(409);
    await expect(response.json()).resolves.toEqual(
      normalizedErrorEnvelope({
        code: 'trial_not_eligible',
        class: 'security_fail',
        subcode: 'installation_binding_mismatch',
        message: 'installation_id is already bound to a different device_public_key',
      }),
    );
  });

  it('keeps existing-installation reissue races idempotent by returning the final persisted challenge in every 200 response', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-04-08T06:00:00Z'));

    let armUpdatePause = false;
    let updateAttemptCount = 0;
    const firstUpdateReady = createDeferred();
    const secondUpdateReady = createDeferred();
    const releaseBothUpdates = createDeferred();
    const env = createTestBrokerEnv({
      beforeRun: async ({ sql }) => {
        if (
          armUpdatePause &&
          sql.includes('UPDATE installations') &&
          sql.includes('challenge = ?')
        ) {
          updateAttemptCount += 1;
          if (updateAttemptCount === 1) {
            firstUpdateReady.resolve();
          }
          if (updateAttemptCount === 2) {
            secondUpdateReady.resolve();
          }
          await releaseBothUpdates.promise;
        }
      },
    });

    await issueChallenge({
      env,
      installationId: 'install-race-existing',
      devicePublicKey: DEVICE_PUBLIC_KEY,
      appVersion: '1.2.3',
    });

    vi.setSystemTime(new Date('2026-04-08T06:01:00Z'));
    armUpdatePause = true;

    const firstResponsePromise = app.request(
      'http://broker.test/v1/trial/challenge',
      {
        method: 'POST',
        headers: {
          'content-type': 'application/json',
        },
        body: JSON.stringify({
          installation_id: 'install-race-existing',
          device_public_key: DEVICE_PUBLIC_KEY,
          app_version: '1.2.4',
        }),
      },
      env,
    );

    await firstUpdateReady.promise;

    const secondResponsePromise = app.request(
      'http://broker.test/v1/trial/challenge',
      {
        method: 'POST',
        headers: {
          'content-type': 'application/json',
        },
        body: JSON.stringify({
          installation_id: 'install-race-existing',
          device_public_key: DEVICE_PUBLIC_KEY,
          app_version: '1.2.4',
        }),
      },
      env,
    );

    await secondUpdateReady.promise;
    releaseBothUpdates.resolve();

    const [firstResponse, secondResponse] = await Promise.all([
      firstResponsePromise,
      secondResponsePromise,
    ]);

    expect(firstResponse.status).toBe(200);
    expect(secondResponse.status).toBe(200);

    const firstPayload = (await firstResponse.json()) as Record<string, unknown>;
    const secondPayload = (await secondResponse.json()) as Record<string, unknown>;

    const installation = env.__db
      .prepare(
        `SELECT app_version, challenge, challenge_expires_at
           FROM installations
          WHERE installation_id = ?`,
      )
      .get('install-race-existing') as Record<string, unknown>;

    expect(installation.app_version).toBe('1.2.4');
    expect(installation.challenge_expires_at).toBe('2026-04-08T06:06:00.000Z');
    expect(firstPayload.challenge).toBe(installation.challenge);
    expect(secondPayload.challenge).toBe(installation.challenge);
  });

  it('returns a fresh persisted challenge when reissue loses its CAS update to concurrent verify consumption', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-04-08T06:00:00Z'));

    let armUpdatePause = false;
    let pausedOnce = false;
    const updatePaused = createDeferred();
    const releaseUpdate = createDeferred();
    const env = createTestBrokerEnv({
      beforeRun: async ({ sql }) => {
        if (
          armUpdatePause &&
          !pausedOnce &&
          sql.includes('UPDATE installations') &&
          sql.includes('challenge = ?')
        ) {
          pausedOnce = true;
          updatePaused.resolve();
          await releaseUpdate.promise;
        }
      },
    });
    const keyPair = await createDeviceKeyPair();
    const initialChallenge = await issueChallenge({
      env,
      installationId: 'install-race-reissue-vs-verify',
      devicePublicKey: keyPair.devicePublicKey,
      appVersion: '1.2.3',
    });

    vi.setSystemTime(new Date('2026-04-08T06:01:00Z'));
    armUpdatePause = true;

    const reissueResponsePromise = app.request(
      'http://broker.test/v1/trial/challenge',
      {
        method: 'POST',
        headers: {
          'content-type': 'application/json',
        },
        body: JSON.stringify({
          installation_id: 'install-race-reissue-vs-verify',
          device_public_key: keyPair.devicePublicKey,
          app_version: '1.2.4',
        }),
      },
      env,
    );

    await updatePaused.promise;

    const verifyRequest = await signCanonicalVerifyRequest(keyPair.privateKey, {
      installation_id: 'install-race-reissue-vs-verify',
      device_public_key: keyPair.devicePublicKey,
      challenge: initialChallenge.challenge,
      challenge_expires_at: initialChallenge.challenge_expires_at,
      hardware_hash: 'hardware-hash-race-reissue-vs-verify',
      app_version: '1.2.3',
      signed_at: '2026-04-08T06:00:30.000Z',
    });

    const verifyResponse = await postVerify(env, verifyRequest);
    expect(verifyResponse.status).toBe(200);

    releaseUpdate.resolve();

    const reissueResponse = await reissueResponsePromise;
    expect(reissueResponse.status).toBe(200);

    const payload = (await reissueResponse.json()) as Record<string, unknown>;
    const installation = env.__db
      .prepare(
        `SELECT app_version, hardware_hash, hardware_hash_salt_version, challenge,
                challenge_expires_at, challenge_salt_version
           FROM installations
          WHERE installation_id = ?`,
      )
      .get('install-race-reissue-vs-verify') as Record<string, unknown>;

    expect(payload).toMatchObject({
      challenge: expect.stringMatching(/^[A-Za-z0-9_-]{43}$/),
      challenge_expires_at: '2026-04-08T06:06:00.000Z',
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
    expect(payload.challenge).toBe(installation.challenge);
    expect(payload.challenge_expires_at).toBe(installation.challenge_expires_at);
    expect(installation).toEqual({
      app_version: '1.2.4',
      hardware_hash: null,
      hardware_hash_salt_version: null,
      challenge: payload.challenge,
      challenge_expires_at: '2026-04-08T06:06:00.000Z',
      challenge_salt_version: 7,
    });
  });
});
