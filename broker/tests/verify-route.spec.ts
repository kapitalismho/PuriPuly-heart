import { Buffer } from 'node:buffer';

import { afterEach, describe, expect, it, vi } from 'vitest';

import {
  createDeviceKeyPair,
  signCanonicalIssueRequest,
  signCanonicalVerifyRequest,
} from './test-support/ed25519';
import { normalizedErrorEnvelope } from './test-support/errors';
import { sha256Base64Url } from './test-support/hash';
import { createTestBrokerEnv, insertEntitlement } from './test-support/sqlite-d1';
import { issueChallenge, postIssue, postVerify } from './test-support/trial-api';

const OVERSIZED_INSTALLATION_ID = 'i'.repeat(129);
const OVERSIZED_APP_VERSION = 'v'.repeat(65);
const OVERSIZED_HARDWARE_HASH = 'h'.repeat(129);

interface VerifyBoundsCase {
  name: 'installation_id' | 'app_version' | 'hardware_hash';
  overrides: Partial<{
    installation_id: string;
    app_version: string;
    hardware_hash: string;
  }>;
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

function readInstallationChallengeSnapshot(
  env: ReturnType<typeof createTestBrokerEnv>,
  installationId: string,
): Record<string, unknown> {
  return env.__db
    .prepare(
      `SELECT challenge, challenge_expires_at, challenge_salt_version,
              hardware_hash, hardware_hash_salt_version
         FROM installations
        WHERE installation_id = ?`,
    )
    .get(installationId) as Record<string, unknown>;
}

describe('POST /v1/trial/challenge/verify route contract', () => {
  afterEach(() => {
    vi.useRealTimers();
  });

  it('consumes the active challenge only after successful verify and persists hardware_hash with the challenge salt version', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-04-08T06:00:00Z'));

    const env = createTestBrokerEnv();
    const keyPair = await createDeviceKeyPair();
    const challenge = await issueChallenge({
      env,
      installationId: 'install-verify-route',
      devicePublicKey: keyPair.devicePublicKey,
      appVersion: '1.2.3',
    });

    const requestBody = await signCanonicalVerifyRequest(keyPair.privateKey, {
      installation_id: 'install-verify-route',
      device_public_key: keyPair.devicePublicKey,
      challenge: challenge.challenge,
      challenge_expires_at: challenge.challenge_expires_at,
      hardware_hash: 'hardware-hash-route-001',
      app_version: '2.0.0',
      signed_at: '2026-04-08T06:00:30.000Z',
    });

    const response = await postVerify(env, requestBody);
    expect(response.status).toBe(200);

    const installation = env.__db
      .prepare(
        `SELECT hardware_hash, hardware_hash_salt_version, app_version, challenge,
                challenge_expires_at, challenge_salt_version
           FROM installations
          WHERE installation_id = ?`,
      )
      .get('install-verify-route') as Record<string, unknown>;

    expect(installation).toEqual({
      hardware_hash: 'hardware-hash-route-001',
      hardware_hash_salt_version: 7,
      app_version: '2.0.0',
      challenge: null,
      challenge_expires_at: null,
      challenge_salt_version: null,
    });

    const replayResponse = await postVerify(env, requestBody);
    expect(replayResponse.status).toBe(404);
    await expect(replayResponse.json()).resolves.toEqual(
      normalizedErrorEnvelope({
        code: 'challenge_invalid',
        class: 'security_fail',
        subcode: 'challenge_not_found',
        message: 'no active challenge exists for installation_id',
      }),
    );
  });

  it('rejects expired challenges without consuming stored challenge state', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-04-08T06:00:00Z'));

    const env = createTestBrokerEnv();
    const keyPair = await createDeviceKeyPair();
    const challenge = await issueChallenge({
      env,
      installationId: 'install-expired',
      devicePublicKey: keyPair.devicePublicKey,
      appVersion: '1.2.3',
    });

    vi.setSystemTime(new Date('2026-04-08T06:05:01Z'));

    const requestBody = await signCanonicalVerifyRequest(keyPair.privateKey, {
      installation_id: 'install-expired',
      device_public_key: keyPair.devicePublicKey,
      challenge: challenge.challenge,
      challenge_expires_at: challenge.challenge_expires_at,
      hardware_hash: 'hardware-hash-route-002',
      app_version: '1.2.3',
      signed_at: '2026-04-08T06:05:00.000Z',
    });

    const response = await postVerify(env, requestBody);
    expect(response.status).toBe(410);
    await expect(response.json()).resolves.toEqual(
      normalizedErrorEnvelope({
        code: 'challenge_expired',
        class: 'retryable',
        retryAfterMs: 0,
        message: 'challenge has expired and must be reissued',
      }),
    );

    const installation = env.__db
      .prepare(
        `SELECT hardware_hash, challenge, challenge_expires_at, challenge_salt_version
           FROM installations
          WHERE installation_id = ?`,
      )
      .get('install-expired') as Record<string, unknown>;

    expect(installation).toEqual({
      hardware_hash: null,
      challenge: challenge.challenge,
      challenge_expires_at: challenge.challenge_expires_at,
      challenge_salt_version: 7,
    });
  });

  it('stores verified_hardware_hash and salt version on pending_release entitlement', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-04-08T06:00:00Z'));

    const env = createTestBrokerEnv();
    const keyPair = await createDeviceKeyPair();
    const challenge = await issueChallenge({
      env,
      installationId: 'install-verify-snapshot',
      devicePublicKey: keyPair.devicePublicKey,
      appVersion: '2.0.0',
    });

    const requestBody = await signCanonicalVerifyRequest(keyPair.privateKey, {
      installation_id: 'install-verify-snapshot',
      device_public_key: keyPair.devicePublicKey,
      challenge: challenge.challenge,
      challenge_expires_at: challenge.challenge_expires_at,
      hardware_hash: 'hardware-hash-verify-snapshot',
      app_version: '2.0.0',
      signed_at: '2026-04-08T06:00:30.000Z',
    });

    const response = await postVerify(env, requestBody);
    expect(response.status).toBe(200);

    const entitlement = env.__db
      .prepare(
        `SELECT status, verified_hardware_hash, verified_hardware_hash_salt_version
           FROM openrouter_entitlements
          WHERE installation_id = ?`,
      )
      .get('install-verify-snapshot') as Record<string, unknown>;

    expect(entitlement).toEqual({
      status: 'pending_release',
      verified_hardware_hash: 'hardware-hash-verify-snapshot',
      verified_hardware_hash_salt_version: 7,
    });
  });

  it.each([
    { discordIssueStatus: 'issuing', managedCredentialRef: null },
    {
      discordIssueStatus: 'cleanup_required',
      managedCredentialRef: 'hash_orphaned_discord_cleanup_required_verify',
    },
  ] satisfies Array<{
    discordIssueStatus: DiscordBlockedIssueStatus;
    managedCredentialRef: string | null;
  }>)(
    'rejects verify release-token minting for Discord $discordIssueStatus pending_release without mutating entitlement metadata',
    async ({ discordIssueStatus, managedCredentialRef }) => {
      vi.useFakeTimers();
      vi.setSystemTime(new Date('2026-04-08T06:00:00Z'));

      const env = createTestBrokerEnv();
      const keyPair = await createDeviceKeyPair();
      const installationId = `install-discord-${discordIssueStatus}-verify-guard`;
      const challenge = await issueChallenge({
        env,
        installationId,
        devicePublicKey: keyPair.devicePublicKey,
        appVersion: '1.2.3',
      });
      insertEntitlement(env, {
        installation_id: installationId,
        status: 'pending_release',
        budget_usd: 0.07,
        managed_credential_ref: managedCredentialRef,
        release_session_ref: `release-session-${discordIssueStatus}-verify-guard`,
        release_token_hash: `release-token-hash-${discordIssueStatus}-verify-guard`,
        release_token_expires_at: '2026-04-08T06:10:00.000Z',
        verified_hardware_hash: `hardware-hash-${discordIssueStatus}-verify-guard`,
        verified_hardware_hash_salt_version: 7,
        discord_issue_status: discordIssueStatus,
        discord_issue_reserved_at: '2026-04-08T06:00:00.000Z',
      });
      const entitlementBefore = readEntitlementSnapshot(env, installationId);
      const installationBefore = readInstallationChallengeSnapshot(env, installationId);
      const requestBody = await signCanonicalVerifyRequest(keyPair.privateKey, {
        installation_id: installationId,
        device_public_key: keyPair.devicePublicKey,
        challenge: challenge.challenge,
        challenge_expires_at: challenge.challenge_expires_at,
        hardware_hash: `hardware-hash-${discordIssueStatus}-verify-attempt`,
        app_version: '1.2.4',
        signed_at: '2026-04-08T06:00:30.000Z',
      });

      const response = await postVerify(env, requestBody);

      expect(response.status).toBe(409);
      const payload = (await response.json()) as Record<string, unknown>;
      expect(payload).not.toHaveProperty('release_token');
      expect(readEntitlementSnapshot(env, installationId)).toEqual(entitlementBefore);
      expect(readInstallationChallengeSnapshot(env, installationId)).toEqual(
        installationBefore,
      );
    },
  );

  it('records verify outcome markers for both successful and failed verify attempts', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-04-08T06:00:00Z'));

    const env = createTestBrokerEnv();

    const successKeyPair = await createDeviceKeyPair();
    const successChallenge = await issueChallenge({
      env,
      installationId: 'install-verify-outcome-success',
      devicePublicKey: successKeyPair.devicePublicKey,
      appVersion: '1.2.3',
    });
    const successRequest = await signCanonicalVerifyRequest(successKeyPair.privateKey, {
      installation_id: 'install-verify-outcome-success',
      device_public_key: successKeyPair.devicePublicKey,
      challenge: successChallenge.challenge,
      challenge_expires_at: successChallenge.challenge_expires_at,
      hardware_hash: 'hardware-hash-verify-outcome-success',
      app_version: '1.2.3',
      signed_at: '2026-04-08T06:00:30.000Z',
    });

    const successResponse = await postVerify(env, successRequest);
    expect(successResponse.status).toBe(200);

    const failKeyPair = await createDeviceKeyPair();
    const failChallenge = await issueChallenge({
      env,
      installationId: 'install-verify-outcome-fail',
      devicePublicKey: failKeyPair.devicePublicKey,
      appVersion: '1.2.3',
    });
    insertEntitlement(env, {
      installation_id: 'install-verify-outcome-fail',
      status: 'active',
      budget_usd: 0.07,
      managed_credential_ref: 'existing-managed-key',
      issued_at: '2026-04-01T00:00:00.000Z',
      expires_at: '2026-07-01T00:00:00.000Z',
    });
    const failRequest = await signCanonicalVerifyRequest(failKeyPair.privateKey, {
      installation_id: 'install-verify-outcome-fail',
      device_public_key: failKeyPair.devicePublicKey,
      challenge: failChallenge.challenge,
      challenge_expires_at: failChallenge.challenge_expires_at,
      hardware_hash: 'hardware-hash-verify-outcome-fail',
      app_version: '1.2.3',
      signed_at: '2026-04-08T06:00:30.000Z',
    });

    const failResponse = await postVerify(env, failRequest);
    expect(failResponse.status).toBe(409);

    const endpointCounts = Object.fromEntries(
      (
        env.__db
          .prepare(
            `SELECT endpoint, COUNT(*) AS count
               FROM broker_request_events
              GROUP BY endpoint`,
          )
          .all() as Array<{ endpoint: string; count: number }>
      ).map(({ endpoint, count }) => [endpoint, Number(count)]),
    );

    expect(endpointCounts).toMatchObject({
      'POST /v1/trial/challenge': 2,
      'POST /v1/trial/challenge/verify': 2,
      'POST /v1/trial/challenge/verify/success': 1,
      'POST /v1/trial/challenge/verify/fail': 1,
    });
  });

  it('rejects non-object JSON bodies with invalid_request instead of throwing', async () => {
    const env = createTestBrokerEnv();
    const response = await postVerify(env, 'null');

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
      overrides: {
        installation_id: OVERSIZED_INSTALLATION_ID,
      },
      message: 'installation_id must be between 1 and 128 characters',
    },
    {
      name: 'app_version',
      overrides: {
        app_version: OVERSIZED_APP_VERSION,
      },
      message: 'app_version must be between 1 and 64 characters',
    },
    {
      name: 'hardware_hash',
      overrides: {
        hardware_hash: OVERSIZED_HARDWARE_HASH,
      },
      message: 'hardware_hash must be between 1 and 128 characters',
    },
  ] satisfies VerifyBoundsCase[])(
    'rejects oversized $name values before consuming challenge state',
    async (testCase: VerifyBoundsCase) => {
      const { overrides, message } = testCase;
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-04-08T06:00:00Z'));

    const env = createTestBrokerEnv();
    const keyPair = await createDeviceKeyPair();
    const challenge = await issueChallenge({
      env,
      installationId: 'install-verify-bounds',
      devicePublicKey: keyPair.devicePublicKey,
      appVersion: '1.2.3',
    });

    const requestBody = await signCanonicalVerifyRequest(keyPair.privateKey, {
      installation_id: 'install-verify-bounds',
      device_public_key: keyPair.devicePublicKey,
      challenge: challenge.challenge,
      challenge_expires_at: challenge.challenge_expires_at,
      hardware_hash: 'hardware-hash-verify-bounds',
      app_version: '1.2.3',
      signed_at: '2026-04-08T06:00:30.000Z',
      ...overrides,
    });

    const response = await postVerify(env, requestBody);

    expect(response.status).toBe(400);
    await expect(response.json()).resolves.toEqual(
      normalizedErrorEnvelope({
        code: 'invalid_request',
        class: 'terminal',
        message,
      }),
    );

    const installation = env.__db
      .prepare(
        `SELECT challenge, challenge_expires_at, hardware_hash, app_version
           FROM installations
          WHERE installation_id = ?`,
      )
      .get('install-verify-bounds') as Record<string, unknown>;

    expect(installation).toEqual({
      challenge: challenge.challenge,
      challenge_expires_at: challenge.challenge_expires_at,
      hardware_hash: null,
      app_version: '1.2.3',
    });
    },
  );

  it('rejects non-ISO signed_at strings instead of accepting Date.parse-compatible timestamps', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-04-08T06:00:00Z'));

    const env = createTestBrokerEnv();
    const keyPair = await createDeviceKeyPair();
    const challenge = await issueChallenge({
      env,
      installationId: 'install-non-iso-signed-at',
      devicePublicKey: keyPair.devicePublicKey,
      appVersion: '1.2.3',
    });

    const requestBody = await signCanonicalVerifyRequest(keyPair.privateKey, {
      installation_id: 'install-non-iso-signed-at',
      device_public_key: keyPair.devicePublicKey,
      challenge: challenge.challenge,
      challenge_expires_at: challenge.challenge_expires_at,
      hardware_hash: 'hardware-hash-non-iso-signed-at',
      app_version: '1.2.3',
      signed_at: 'Wed, 08 Apr 2026 06:00:30 GMT',
    });

    const response = await postVerify(env, requestBody);

    expect(response.status).toBe(400);
    await expect(response.json()).resolves.toEqual(
      normalizedErrorEnvelope({
        code: 'invalid_request',
        class: 'terminal',
        message: 'challenge_expires_at and signed_at must be valid ISO-8601 timestamps',
      }),
    );
  });

  it('rejects impossible ISO-looking signed_at strings instead of normalizing them', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-04-08T06:00:00Z'));

    const env = createTestBrokerEnv();
    const keyPair = await createDeviceKeyPair();
    const challenge = await issueChallenge({
      env,
      installationId: 'install-impossible-iso-signed-at',
      devicePublicKey: keyPair.devicePublicKey,
      appVersion: '1.2.3',
    });

    const requestBody = await signCanonicalVerifyRequest(keyPair.privateKey, {
      installation_id: 'install-impossible-iso-signed-at',
      device_public_key: keyPair.devicePublicKey,
      challenge: challenge.challenge,
      challenge_expires_at: challenge.challenge_expires_at,
      hardware_hash: 'hardware-hash-impossible-iso-signed-at',
      app_version: '1.2.3',
      signed_at: '2026-04-31T06:00:30Z',
    });

    const response = await postVerify(env, requestBody);

    expect(response.status).toBe(400);
    await expect(response.json()).resolves.toEqual(
      normalizedErrorEnvelope({
        code: 'invalid_request',
        class: 'terminal',
        message: 'challenge_expires_at and signed_at must be valid ISO-8601 timestamps',
      }),
    );
  });

  it('allows only one successful verify when two requests race to consume the same challenge', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-04-08T06:00:00Z'));

    let armRaceHooks = false;
    let pausedConsumeOnce = false;
    let pendingReleaseClearCount = 0;
    const consumePaused = createDeferred();
    const releaseConsume = createDeferred();
    const staleClearPaused = createDeferred();
    const releaseStaleClear = createDeferred();
    const env = createTestBrokerEnv({
      beforeRun: async ({ sql, params }) => {
        if (
          armRaceHooks &&
          sql.includes('UPDATE openrouter_entitlements') &&
          sql.includes('SET release_session_ref = NULL') &&
          params[0] === 'install-race'
        ) {
          pendingReleaseClearCount += 1;
          if (pendingReleaseClearCount === 2) {
            staleClearPaused.resolve();
            await releaseStaleClear.promise;
          }
        }

        if (
          armRaceHooks &&
          !pausedConsumeOnce &&
          sql.includes('UPDATE installations') &&
          sql.includes('challenge = NULL')
        ) {
          pausedConsumeOnce = true;
          consumePaused.resolve();
          await releaseConsume.promise;
        }
      },
    });
    const keyPair = await createDeviceKeyPair();

    const firstChallenge = await issueChallenge({
      env,
      installationId: 'install-race',
      devicePublicKey: keyPair.devicePublicKey,
      appVersion: '1.0.0',
    });
    const firstVerify = await signCanonicalVerifyRequest(keyPair.privateKey, {
      installation_id: 'install-race',
      device_public_key: keyPair.devicePublicKey,
      challenge: firstChallenge.challenge,
      challenge_expires_at: firstChallenge.challenge_expires_at,
      hardware_hash: 'hardware-hash-race-initial',
      app_version: '1.0.0',
      signed_at: '2026-04-08T06:00:30.000Z',
    });
    const initialResponse = await postVerify(env, firstVerify);
    expect(initialResponse.status).toBe(200);

    const secondChallenge = await issueChallenge({
      env,
      installationId: 'install-race',
      devicePublicKey: keyPair.devicePublicKey,
      appVersion: '1.0.1',
    });
    const racingRequest = await signCanonicalVerifyRequest(keyPair.privateKey, {
      installation_id: 'install-race',
      device_public_key: keyPair.devicePublicKey,
      challenge: secondChallenge.challenge,
      challenge_expires_at: secondChallenge.challenge_expires_at,
      hardware_hash: 'hardware-hash-race-final',
      app_version: '1.0.1',
      signed_at: '2026-04-08T06:00:40.000Z',
    });

    armRaceHooks = true;
    pendingReleaseClearCount = 0;
    const firstRacingResponsePromise = postVerify(env, racingRequest);
    await consumePaused.promise;
    const secondRacingResponsePromise = postVerify(env, racingRequest);
    await staleClearPaused.promise;
    releaseConsume.resolve();

    const firstRacingResponse = await firstRacingResponsePromise;
    releaseStaleClear.resolve();
    const secondRacingResponse = await secondRacingResponsePromise;
    const responses = [firstRacingResponse, secondRacingResponse];
    const statuses = responses.map(({ status }) => status).sort();

    expect(statuses).toEqual([200, 409]);

    const successResponse = responses.find(({ status }) => status === 200);
    const conflictResponse = responses.find(({ status }) => status === 409);

    expect(successResponse).toBeDefined();
    expect(conflictResponse).toBeDefined();
    const successPayload = (await successResponse!.json()) as {
      release_token: string;
      release_token_expires_at: string;
    };
    expect(successPayload).toEqual(
      expect.objectContaining({
        release_token: expect.stringMatching(/^[A-Za-z0-9_-]{43}$/),
        release_token_expires_at: '2026-04-08T06:15:00.000Z',
      }),
    );
    await expect(conflictResponse!.json()).resolves.toEqual({
      ...normalizedErrorEnvelope({
        code: 'challenge_invalid',
        class: 'security_fail',
        subcode: 'challenge_consumed',
        message: 'challenge has already been consumed or replaced',
      }),
    });

    const installation = env.__db
      .prepare(
        `SELECT challenge, challenge_expires_at, hardware_hash, app_version
           FROM installations
          WHERE installation_id = ?`,
      )
      .get('install-race') as Record<string, unknown>;

    expect(installation).toEqual({
      challenge: null,
      challenge_expires_at: null,
      hardware_hash: 'hardware-hash-race-final',
      app_version: '1.0.1',
    });

    const entitlement = env.__db
      .prepare(
        `SELECT status, release_session_ref, release_token_hash, release_token_expires_at
           FROM openrouter_entitlements
          WHERE installation_id = ?`,
      )
      .get('install-race') as Record<string, unknown>;

    expect(entitlement.status).toBe('pending_release');
    expect(entitlement.release_session_ref).toBeTypeOf('string');
    await expect(sha256Base64Url(successPayload.release_token)).resolves.toBe(
      entitlement.release_token_hash,
    );
    expect(entitlement.release_token_expires_at).toBe('2026-04-08T06:15:00.000Z');
  });

  it('rejects a stale release token while verify is rotating to a new pending release session', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-04-08T06:00:00Z'));

    let armConsumePause = false;
    let pausedOnce = false;
    const consumePaused = createDeferred();
    const releaseConsume = createDeferred();
    const env = createTestBrokerEnv({
      beforeRun: async ({ sql }) => {
        if (
          armConsumePause &&
          !pausedOnce &&
          sql.includes('UPDATE installations') &&
          sql.includes('challenge = NULL')
        ) {
          pausedOnce = true;
          consumePaused.resolve();
          await releaseConsume.promise;
        }
      },
    });

    const keyPair = await createDeviceKeyPair();
    const challenge = await issueChallenge({
      env,
      installationId: 'install-verify-rotates-release-session',
      devicePublicKey: keyPair.devicePublicKey,
      appVersion: '1.0.0',
    });

    const staleReleaseToken = Buffer.alloc(32, 11).toString('base64url');
    insertEntitlement(env, {
      installation_id: 'install-verify-rotates-release-session',
      status: 'pending_release',
      budget_usd: 0.07,
      release_session_ref: 'old-release-session',
      release_token_hash: await sha256Base64Url(staleReleaseToken),
      release_token_expires_at: '2026-04-08T06:15:00.000Z',
      verified_hardware_hash: 'hardware-hash-verify-old-session',
      verified_hardware_hash_salt_version: 7,
    });

    const nextVerifyRequest = await signCanonicalVerifyRequest(keyPair.privateKey, {
      installation_id: 'install-verify-rotates-release-session',
      device_public_key: keyPair.devicePublicKey,
      challenge: challenge.challenge,
      challenge_expires_at: challenge.challenge_expires_at,
      hardware_hash: 'hardware-hash-verify-new-session',
      app_version: '1.0.1',
      signed_at: '2026-04-08T06:00:30.000Z',
    });

    armConsumePause = true;
    const verifyResponsePromise = postVerify(env, nextVerifyRequest);
    await consumePaused.promise;

    const entitlementWhileRotating = env.__db
      .prepare(
        `SELECT release_session_ref, release_token_hash, release_token_expires_at,
                verified_hardware_hash, verified_hardware_hash_salt_version
           FROM openrouter_entitlements
          WHERE installation_id = ?`,
      )
      .get('install-verify-rotates-release-session') as Record<string, unknown>;

    expect(entitlementWhileRotating).toEqual({
      release_session_ref: null,
      release_token_hash: null,
      release_token_expires_at: null,
      verified_hardware_hash: 'hardware-hash-verify-old-session',
      verified_hardware_hash_salt_version: 7,
    });

    const staleIssueRequest = await signCanonicalIssueRequest(keyPair.privateKey, {
      installation_id: 'install-verify-rotates-release-session',
      device_public_key: keyPair.devicePublicKey,
      release_token: staleReleaseToken,
      hardware_hash: 'hardware-hash-verify-old-session',
      reason: 'llm_start',
      budget_usd: 0.07,
      model: 'google/gemma-4-26b-a4b-it',
      signed_at: '2026-04-08T06:00:30.000Z',
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

    releaseConsume.resolve();

    const verifyResponse = await verifyResponsePromise;
    expect(verifyResponse.status).toBe(200);
  });
});
