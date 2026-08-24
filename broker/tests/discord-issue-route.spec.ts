import { afterEach, describe, expect, it, vi } from 'vitest';

import app from '../src/index';
import {
  createDeviceKeyPair,
  encodeBase64Url,
  signCanonicalDiscordIssueRequest,
  type DeviceKeyPair,
  type SignedDiscordIssueRequestInput,
} from './test-support/ed25519';
import { normalizedErrorEnvelope } from './test-support/errors';
import { sha256Base64Url } from './test-support/hash';
import {
  createTestBrokerEnv,
  insertEntitlement,
  type TestBrokerEnv,
} from './test-support/sqlite-d1';
import { postDiscordIssue, postDiscordStart } from './test-support/trial-api';
import {
  insertSubjectHook,
  updateAbuseControls,
  updateAbuseRuntimeState,
} from './test-support/abuse-controls';
import { expectNoReferralRewardEstimateFields } from './test-support/referral-response-privacy';

const REGISTERED_REDIRECT_URI = 'http://127.0.0.1:62187/discord/callback';
const APP_VERSION = '1.2.3';
const MODEL = 'google/gemma-4-26b-a4b-it';
const NOW_ISO = '2026-04-30T06:00:00.000Z';
const SIGNED_AT_ISO = '2026-04-30T06:00:30.000Z';
const DISCORD_TOKEN_URL = 'https://discord.com/api/oauth2/token';
const DISCORD_USER_URL = 'https://discord.com/api/users/@me';
const OPENROUTER_KEYS_URL = 'https://openrouter.ai/api/v1/keys';
const OPENROUTER_GUARDRAIL_URL =
  'https://openrouter.ai/api/v1/guardrails/test-managed-guardrail-id/assignments/keys';
const DISCORD_EPOCH_MS = 1420070400000n;
const THIRTY_DAYS_MS = 30 * 24 * 60 * 60 * 1000;
const REFERRAL_ID_PATTERN = /^[23456789ABCDEFGHJKMNPQRSTUVWXYZ]{6}$/u;

interface StartedDiscordSession {
  env: TestBrokerEnv;
  keyPair: DeviceKeyPair;
  installationId: string;
  state: string;
  issueNonce: string;
  redirectUri: string;
  appVersion: string;
  fingerprintSaltVersion: number;
}

interface DiscordSessionRow {
  installation_id: string;
  device_public_key: string;
  redirect_uri: string;
  pkce_code_verifier: string | null;
  issue_nonce_hash: string;
  fingerprint_salt_version: number;
  status: string;
  processing_started_at: string | null;
  eligibility_checked_at: string | null;
  consumed_at: string | null;
}

interface DiscordIdentityRow {
  discord_user_ref: string;
  entitlement_installation_id: string | null;
  status: string;
  updated_at: string;
}

interface IssueSuccessEventRow {
  issue_source: string;
  installation_id: string;
  subject_ref: string;
  managed_credential_ref: string;
  ip_hash: string | null;
  ip_prefix_hash: string | null;
  observed_at: string;
}

interface ReferralCodeRow {
  referral_id: string;
  owner_discord_user_ref: string;
  owner_installation_id: string | null;
  status: string;
}

describe('Discord issue gate', () => {
  afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
    vi.useRealTimers();
  });

  it('reservation success exchanges Discord code with PKCE, activates one managed key, monitors success, and consumes the session', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date(NOW_ISO));

    const started = await startDiscordSession('install-discord-issue-gate-valid');
    const sessionBefore = await readSessionByState(started.env, started.state);
    const code = 'discord-oauth-code-valid';
    const rawDiscordUserId = discordSnowflakeForAgeDays(31);
    const rawDiscordEmail = 'verified@example.test';
    const expectedDiscordUserRef = await deriveExpectedDiscordUserRef(
      started.env.DISCORD_USER_REF_SECRET,
      rawDiscordUserId,
    );
    const discordApi = mockDiscordApi({
      user: {
        id: rawDiscordUserId,
        verified: true,
        email: rawDiscordEmail,
      },
    });
    const requestBody = await signedIssueRequest(started, { code });

    const response = await postDiscordIssue(started.env, requestBody);

    expect(response.status).toBe(200);
    const payload = (await response.json()) as Record<string, unknown>;
    expect(payload).toEqual(
      expect.objectContaining({
        openrouter_api_key: 'or-discord-managed-child-key-test-1',
        managed_credential_ref: 'hash_discord_managed_child_test_1',
        managed_state: {
          lifecycle: 'active',
          managed_availability: true,
        },
        expires_at: '2026-07-30T06:00:00.000Z',
        budget_usd: 0.07,
        model: MODEL,
      }),
    );
    expect(payload.referral_id).toMatch(REFERRAL_ID_PATTERN);
    expect(payload.talk_together_pass).toEqual({
      pass_id: payload.referral_id,
      invite_count: 0,
      invite_limit: 5,
      bonus_translations_per_friend: 200,
    });
    const serializedTalkTogetherPass = JSON.stringify(payload.talk_together_pass);
    expect(serializedTalkTogetherPass).not.toContain('discord_user_ref');
    expect(serializedTalkTogetherPass).not.toContain('budget_usd');
    expect(payload).not.toHaveProperty('referral_bonus_applied');
    expectNoReferralRewardEstimateFields(payload);
    expectTokenExchange(discordApi.fetchMock, {
      code,
      redirectUri: started.redirectUri,
      codeVerifier: sessionBefore.pkce_code_verifier,
    });
    expectDiscordUserFetch(discordApi.fetchMock);
    expect(discordApi.openRouterCreateCalls).toHaveLength(1);
    expect(discordApi.openRouterGuardrailCalls).toHaveLength(1);
    expectCallOrder(discordApi.fetchMock, [
      `POST ${OPENROUTER_KEYS_URL}`,
      `POST ${OPENROUTER_GUARDRAIL_URL}`,
    ]);
    await expect(readSessionByState(started.env, started.state)).resolves.toEqual(
      expect.objectContaining({
        status: 'consumed',
        pkce_code_verifier: null,
        processing_started_at: NOW_ISO,
        eligibility_checked_at: NOW_ISO,
        consumed_at: NOW_ISO,
      }),
    );
    await expect(readEntitlement(started.env, started.installationId)).resolves.toEqual(
      expect.objectContaining({
        status: 'active',
        managed_credential_ref: 'hash_discord_managed_child_test_1',
        issued_at: NOW_ISO,
        expires_at: '2026-07-30T06:00:00.000Z',
        verified_hardware_hash: 'hardware-hash-discord-issue',
        discord_user_ref: expectedDiscordUserRef,
        discord_issue_status: 'active',
        discord_issue_reserved_at: NOW_ISO,
        discord_issue_delivered_at: NOW_ISO,
      }),
    );
    await expect(readDiscordIdentity(started.env, expectedDiscordUserRef)).resolves.toEqual(
      expect.objectContaining({
        entitlement_installation_id: started.installationId,
        status: 'active',
        updated_at: NOW_ISO,
      }),
    );
    expect(readReferralCodeForOwner(started.env, expectedDiscordUserRef)).toEqual(
      expect.objectContaining({
        referral_id: payload.referral_id,
        owner_discord_user_ref: expectedDiscordUserRef,
        owner_installation_id: started.installationId,
        status: 'active',
      }),
    );
    const issueSuccessEvents = readIssueSuccessEvents(started.env);
    expect(issueSuccessEvents).toEqual([
      expect.objectContaining({
        issue_source: 'discord',
        installation_id: started.installationId,
        subject_ref: started.installationId,
        managed_credential_ref: 'hash_discord_managed_child_test_1',
        observed_at: NOW_ISO,
      }),
    ]);
    expect(JSON.stringify(issueSuccessEvents)).not.toContain(rawDiscordUserId);
    expect(JSON.stringify(issueSuccessEvents)).not.toContain(rawDiscordEmail);
  });

  it('keeps Referral ID and omits Talk Together Pass status when issue invite count query fails', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date(NOW_ISO));
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => undefined);

    const env = createTestBrokerEnv({
      beforeFirst({ sql }) {
        if (
          sql.includes('FROM referral_rewards counted') &&
          sql.includes('counted.referrer_discord_user_ref = ?')
        ) {
          throw new Error('forced Talk Together Pass count failure');
        }
      },
    });
    const started = await startDiscordSession(
      'install-discord-issue-pass-count-failure',
      env,
    );
    const code = 'discord-oauth-code-pass-count-failure';
    const rawDiscordUserId = discordSnowflakeForAgeDays(31);
    mockDiscordApi({
      user: {
        id: rawDiscordUserId,
        verified: true,
      },
    });
    const requestBody = await signedIssueRequest(started, { code });

    const response = await postDiscordIssue(started.env, requestBody);

    expect(response.status).toBe(200);
    const payload = (await response.json()) as Record<string, unknown>;
    expect(payload.referral_id).toMatch(REFERRAL_ID_PATTERN);
    expect(payload).not.toHaveProperty('talk_together_pass');
    expect(warnSpy).toHaveBeenCalledTimes(1);
    expect(warnSpy).toHaveBeenCalledWith('owned_referral_status_failed', {
      endpoint: 'discord_issue',
      installation_id: expect.any(String),
      reason: 'talk_together_pass_status_failed',
    });
    const warnCalls = JSON.stringify(warnSpy.mock.calls);
    expect(warnCalls).not.toContain('forced Talk Together Pass count failure');
    expect(warnCalls).not.toContain('or-discord-managed-child-key');
    expect(warnCalls).not.toContain('hash_discord_managed_child');
    expect(warnCalls).not.toContain(rawDiscordUserId);
  });

  it('returns a restart boundary when Discord token exchange fails after terminalizing the session', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date(NOW_ISO));

    const started = await startDiscordSession('install-discord-issue-token-failure');
    const sessionBefore = await readSessionByState(started.env, started.state);
    const code = 'discord-oauth-code-token-failure-redact';
    const consoleErrorSpy = vi.spyOn(console, 'error').mockImplementation(() => undefined);
    mockDiscordApi({ tokenStatus: 500 });
    const requestBody = await signedIssueRequest(started, { code });

    const response = await postDiscordIssue(started.env, requestBody);

    expect(response.status).toBe(410);
    const responseText = await response.text();
    expect(JSON.parse(responseText)).toEqual(
      normalizedErrorEnvelope({
        code: 'challenge_expired',
        class: 'retryable',
        subcode: 'discord_oauth_failed',
        retryAfterMs: 0,
        message: 'Discord OAuth verification failed; restart Discord OAuth onboarding',
      }),
    );
    const sensitiveValues = [
      code,
      started.state,
      sessionBefore.pkce_code_verifier,
    ].filter((value): value is string => value !== null);
    expectTextNotToContainSensitiveValues(responseText, sensitiveValues);
    expectTextNotToContainSensitiveValues(
      stringifyConsoleCalls(consoleErrorSpy),
      sensitiveValues,
    );
    await expect(readSessionByState(started.env, started.state)).resolves.toEqual(
      expect.objectContaining({
        status: 'failed',
        pkce_code_verifier: null,
      }),
    );
  });

  it('returns a restart boundary when Discord user fetch fails after terminalizing the session', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date(NOW_ISO));

    const started = await startDiscordSession('install-discord-issue-user-fetch-failure');
    mockDiscordApi({ userStatus: 500 });
    const requestBody = await signedIssueRequest(started);

    const response = await postDiscordIssue(started.env, requestBody);

    expect(response.status).toBe(410);
    await expect(response.json()).resolves.toEqual(
      normalizedErrorEnvelope({
        code: 'challenge_expired',
        class: 'retryable',
        subcode: 'discord_oauth_failed',
        retryAfterMs: 0,
        message: 'Discord OAuth verification failed; restart Discord OAuth onboarding',
      }),
    );
    await expect(readSessionByState(started.env, started.state)).resolves.toEqual(
      expect.objectContaining({
        status: 'failed',
        pkce_code_verifier: null,
      }),
    );
  });

  it('rejects invalid JSON with 400', async () => {
    const env = createTestBrokerEnv();

    const response = await postDiscordIssue(env, '{');

    expect(response.status).toBe(400);
  });

  it('rejects missing required fields with 400', async () => {
    const env = createTestBrokerEnv();

    const response = await postDiscordIssue(env, {});

    expect(response.status).toBe(400);
  });

  it('rejects unknown state without Discord token exchange', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date(NOW_ISO));

    const started = await startDiscordSession('install-discord-issue-unknown-state');
    const discordApi = mockDiscordApi();
    const requestBody = await signedIssueRequest(started, {
      state: 'unknown-discord-oauth-state',
    });

    const response = await postDiscordIssue(started.env, requestBody);

    expect(response.status).toBe(410);
    expect(discordApi.fetchMock).not.toHaveBeenCalled();
  });

  it('rejects state binding mismatch without Discord token exchange', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date(NOW_ISO));

    const started = await startDiscordSession('install-discord-issue-binding');
    const discordApi = mockDiscordApi();
    const requestBody = await signedIssueRequest(started, {
      installation_id: 'install-discord-issue-binding-other',
    });

    const response = await postDiscordIssue(started.env, requestBody);

    expect(response.status).toBe(409);
    expect(discordApi.fetchMock).not.toHaveBeenCalled();
  });

  it('rejects a stale session when installation_id is later bound to a different device_public_key', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date(NOW_ISO));

    const started = await startDiscordSession('install-discord-issue-stale-install-binding');
    const conflictingKeyPair = await createDeviceKeyPair();
    insertInstallation(started.env, {
      installationId: started.installationId,
      devicePublicKey: conflictingKeyPair.devicePublicKey,
      hardwareHash: 'hardware-hash-stale-install-binding',
      hardwareHashSaltVersion: started.fingerprintSaltVersion,
    });
    const discordApi = mockDiscordApi();

    const response = await postDiscordIssue(
      started.env,
      await signedIssueRequest(started, {
        code: 'discord-oauth-code-stale-install-binding',
      }),
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
    expect(discordApi.fetchMock).not.toHaveBeenCalled();
    await expect(readSessionByState(started.env, started.state)).resolves.toEqual(
      expect.objectContaining({
        status: 'pending',
        pkce_code_verifier: expect.any(String),
      }),
    );
  });

  it('rejects a stale session when device_public_key is later registered to another installation_id', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date(NOW_ISO));

    const started = await startDiscordSession('install-discord-issue-stale-key-binding');
    insertInstallation(started.env, {
      installationId: 'install-discord-issue-stale-key-other',
      devicePublicKey: started.keyPair.devicePublicKey,
      hardwareHash: 'hardware-hash-stale-key-binding',
      hardwareHashSaltVersion: started.fingerprintSaltVersion,
    });
    const discordApi = mockDiscordApi();

    const response = await postDiscordIssue(
      started.env,
      await signedIssueRequest(started, {
        code: 'discord-oauth-code-stale-key-binding',
      }),
    );

    expect(response.status).toBe(409);
    await expect(response.json()).resolves.toEqual(
      normalizedErrorEnvelope({
        code: 'trial_not_eligible',
        class: 'security_fail',
        subcode: 'device_public_key_registered',
        message: 'device_public_key is already registered to a different installation_id',
      }),
    );
    expect(discordApi.fetchMock).not.toHaveBeenCalled();
    expect(countDiscordEntitlements(started.env)).toBe(0);
  });

  it('rejects stale signed_at without Discord token exchange', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date(NOW_ISO));

    const started = await startDiscordSession('install-discord-issue-stale');
    const discordApi = mockDiscordApi();
    const requestBody = await signedIssueRequest(started, {
      signed_at: '2026-04-30T06:01:01.000Z',
    });

    const response = await postDiscordIssue(started.env, requestBody);

    expect(response.status).toBe(401);
    expect(discordApi.fetchMock).not.toHaveBeenCalled();
  });

  it('keeps pending session reusable after invalid signature', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date(NOW_ISO));

    const started = await startDiscordSession('install-discord-issue-invalid-signature');
    const code = 'discord-oauth-code-invalid-signature';
    const signedRequest = await signedIssueRequest(started, { code });
    const invalidSignature = encodeBase64Url(new Uint8Array(64).fill(7));

    const invalidResponse = await postDiscordIssue(started.env, {
      ...signedRequest,
      signature: invalidSignature,
    });

    expect(invalidResponse.status).toBe(401);
    await expect(readSessionByState(started.env, started.state)).resolves.toEqual(
      expect.objectContaining({
        status: 'pending',
        pkce_code_verifier: expect.any(String),
      }),
    );

    const sessionBeforeRetry = await readSessionByState(started.env, started.state);
    const discordApi = mockDiscordApi({
      user: {
        id: discordSnowflakeForAgeDays(31),
        verified: true,
      },
    });
    const validResponse = await postDiscordIssue(started.env, signedRequest);

    expect(validResponse.status).toBe(200);
    expectTokenExchange(discordApi.fetchMock, {
      code,
      redirectUri: started.redirectUri,
      codeVerifier: sessionBeforeRetry.pkce_code_verifier,
    });
    await expect(readSessionByState(started.env, started.state)).resolves.toEqual(
      expect.objectContaining({
        status: 'consumed',
        pkce_code_verifier: null,
      }),
    );
  });

  it('rejects hardware salt version mismatch without Discord token exchange', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date(NOW_ISO));

    const started = await startDiscordSession('install-discord-issue-salt-mismatch');
    const discordApi = mockDiscordApi();
    const requestBody = await signedIssueRequest(started, {
      hardware_hash_salt_version: started.fingerprintSaltVersion + 1,
    });

    const response = await postDiscordIssue(started.env, requestBody);

    expect(response.status).toBe(409);
    await expect(response.json()).resolves.toEqual(
      normalizedErrorEnvelope({
        code: 'trial_not_eligible',
        class: 'terminal',
        subcode: 'hardware_salt_mismatch',
        message: 'hardware_hash_salt_version does not match the pending Discord OAuth session',
      }),
    );
    expect(discordApi.fetchMock).not.toHaveBeenCalled();
    await expect(readSessionByState(started.env, started.state)).resolves.toEqual(
      expect.objectContaining({
        status: 'pending',
        pkce_code_verifier: expect.any(String),
      }),
    );
  });

  it.each([
    {
      name: 'installation_id',
      configure: (env: TestBrokerEnv, started: StartedDiscordSession) => {
        updateAbuseControls(env, (controls) => {
          controls.discordOpenrouterIssueInstallation.maxRequests = 1;
        });
        insertRequestEvent(env, {
          endpoint: 'POST /v1/providers/openrouter/discord/issue',
          installationId: started.installationId,
          ip: null,
          observedAt: '2026-04-30T05:59:00.000Z',
        });
      },
      post: async (started: StartedDiscordSession, body: object) =>
        postDiscordIssue(started.env, body),
      expectedSubcode: 'installation_rate_limited',
    },
    {
      name: 'ip',
      configure: (env: TestBrokerEnv) => {
        updateAbuseControls(env, (controls) => {
          controls.discordOpenrouterIssueIp.maxRequests = 1;
        });
        insertRequestEvent(env, {
          endpoint: 'POST /v1/providers/openrouter/discord/issue',
          installationId: null,
          ip: '198.51.100.44',
          observedAt: '2026-04-30T05:59:00.000Z',
        });
      },
      post: async (started: StartedDiscordSession, body: object) =>
        postDiscordIssueWithIp(started.env, body, '198.51.100.44'),
      expectedSubcode: 'ip_rate_limited',
    },
  ])(
    'honors configured Discord issue $name endpoint rate limits before Discord token exchange',
    async ({ configure, post, expectedSubcode }) => {
      vi.useFakeTimers();
      vi.setSystemTime(new Date(NOW_ISO));

      const started = await startDiscordSession(
        `install-discord-rate-limit-${expectedSubcode}`,
      );
      configure(started.env, started);
      const discordApi = mockDiscordApi();
      const response = await post(
        started,
        await signedIssueRequest(started, {
          code: `discord-oauth-code-rate-limit-${expectedSubcode}`,
        }),
      );

      expect(response.status).toBe(429);
      await expect(response.json()).resolves.toEqual(
        normalizedErrorEnvelope({
          code: 'rate_limited',
          class: 'retryable',
          subcode: expectedSubcode,
          retryAfterMs: 840_000,
          message: 'request rate limit exceeded for POST /v1/providers/openrouter/discord/issue',
        }),
      );
      expect(discordApi.fetchMock).not.toHaveBeenCalled();
      expect(countDiscordEntitlements(started.env)).toBe(0);
    },
  );

  it('honors Discord issue subject deny hooks before Discord token exchange', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date(NOW_ISO));

    const started = await startDiscordSession('install-discord-subject-deny');
    insertSubjectHook(started.env, {
      hook_kind: 'denylist',
      subject_type: 'installation_id',
      subject_value: started.installationId,
      outcome_code: 'trial_not_eligible',
      outcome_class: 'terminal',
      reason: 'installation is denied for managed issuance',
    });
    const discordApi = mockDiscordApi();

    const response = await postDiscordIssue(
      started.env,
      await signedIssueRequest(started, {
        code: 'discord-oauth-code-subject-deny',
      }),
    );

    expect(response.status).toBe(409);
    await expect(response.json()).resolves.toEqual(
      normalizedErrorEnvelope({
        code: 'trial_not_eligible',
        class: 'terminal',
        message: 'installation is denied for managed issuance',
      }),
    );
    expect(discordApi.fetchMock).not.toHaveBeenCalled();
    expect(countDiscordEntitlements(started.env)).toBe(0);
  });

  it('honors the active issuance brake before Discord token exchange', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date(NOW_ISO));

    const started = await startDiscordSession('install-discord-active-brake');
    updateAbuseRuntimeState(started.env, (state) => {
      state.brake.active = true;
      state.brake.reason = 'manual';
      state.brake.changedAt = NOW_ISO;
      state.brake.changedBy = 'operator';
    });
    const discordApi = mockDiscordApi();

    const response = await postDiscordIssue(
      started.env,
      await signedIssueRequest(started, {
        code: 'discord-oauth-code-active-brake',
      }),
    );

    expect(response.status).toBe(503);
    await expect(response.json()).resolves.toEqual(
      normalizedErrorEnvelope({
        code: 'issuance_suspended',
        class: 'retryable',
        subcode: 'manual',
        message: 'new entitlement issuance is temporarily suspended',
      }),
    );
    expect(discordApi.fetchMock).not.toHaveBeenCalled();
    expect(countDiscordEntitlements(started.env)).toBe(0);
  });

  it('reservation/lifetime rejects a Discord account that already has a delivered managed entitlement', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date(NOW_ISO));

    const env = createTestBrokerEnv();
    const discordUserId = discordSnowflakeForAgeDays(31);
    const first = await startDiscordSession('install-discord-lifetime-first', env);
    const discordApi = mockDiscordApi({
      user: {
        id: discordUserId,
        verified: true,
      },
    });

    const firstResponse = await postDiscordIssue(
      env,
      await signedIssueRequest(first, {
        code: 'discord-oauth-code-lifetime-first',
        hardware_hash: 'hardware-hash-lifetime-first',
      }),
    );
    expect(firstResponse.status).toBe(200);

    const second = await startDiscordSession('install-discord-lifetime-second', env);
    const secondResponse = await postDiscordIssue(
      env,
      await signedIssueRequest(second, {
        code: 'discord-oauth-code-lifetime-second',
        hardware_hash: 'hardware-hash-lifetime-second',
      }),
    );

    expect(secondResponse.status).toBe(409);
    await expect(secondResponse.json()).resolves.toEqual(
      normalizedErrorEnvelope({
        code: 'trial_not_eligible',
        class: 'terminal',
        subcode: 'discord_lifetime_used',
        message: 'Discord account has already used a managed trial',
      }),
    );
    expect(discordApi.openRouterCreateCalls).toHaveLength(1);
    await expect(readSessionByState(env, second.state)).resolves.toEqual(
      expect.objectContaining({
        status: 'failed',
        pkce_code_verifier: null,
      }),
    );
    expect(countDiscordIdentities(env)).toBe(1);
  });

  it('reservation/lifetime rejects a Discord account that already has an issuing reservation', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date(NOW_ISO));

    const env = createTestBrokerEnv();
    const discordUserId = discordSnowflakeForAgeDays(31);
    const discordUserRef = await deriveExpectedDiscordUserRef(
      env.DISCORD_USER_REF_SECRET,
      discordUserId,
    );
    insertInstallation(env, {
      installationId: 'install-discord-lifetime-reserved-existing',
      devicePublicKey: 'reserved-device-public-key',
      hardwareHash: 'hardware-hash-lifetime-reserved-existing',
      hardwareHashSaltVersion: 7,
    });
    env.__db
      .prepare(
        `INSERT INTO discord_identities (
            discord_user_ref,
            entitlement_installation_id,
            status,
            ref_secret_version,
            created_at,
            updated_at
          ) VALUES (?, ?, 'issuing', 1, ?, ?)`,
      )
      .run(
        discordUserRef,
        'install-discord-lifetime-reserved-existing',
        NOW_ISO,
        NOW_ISO,
      );

    const started = await startDiscordSession('install-discord-lifetime-reserved-new', env);
    const discordApi = mockDiscordApi({
      user: {
        id: discordUserId,
        verified: true,
      },
    });

    const response = await postDiscordIssue(
      env,
      await signedIssueRequest(started, {
        code: 'discord-oauth-code-lifetime-reserved',
        hardware_hash: 'hardware-hash-lifetime-reserved-new',
      }),
    );

    expect(response.status).toBe(409);
    await expect(response.json()).resolves.toEqual(
      normalizedErrorEnvelope({
        code: 'trial_not_eligible',
        class: 'terminal',
        subcode: 'discord_lifetime_used',
        message: 'Discord account has already used a managed trial',
      }),
    );
    expect(discordApi.openRouterCreateCalls).toHaveLength(0);
    expect(countDiscordIdentities(env)).toBe(1);
    expect(countDiscordEntitlements(env)).toBe(0);
  });

  it.each(['active', 'expired', 'revoked'] as const)(
    'hardware duplicate rejects legacy %s installation evidence before creating a child key',
    async (storedStatus) => {
      vi.useFakeTimers();
      vi.setSystemTime(new Date(NOW_ISO));

      const env = createTestBrokerEnv();
      insertInstallation(env, {
        installationId: 'install-discord-hardware-legacy-existing',
        devicePublicKey: 'legacy-device-public-key',
        hardwareHash: 'hardware-hash-duplicate-legacy',
        hardwareHashSaltVersion: 7,
      });
      insertEntitlement(env, {
        installation_id: 'install-discord-hardware-legacy-existing',
        status: storedStatus,
        budget_usd: 0.07,
        managed_credential_ref: 'legacy-managed-key',
        issued_at: NOW_ISO,
        expires_at: '2026-07-30T06:00:00.000Z',
      });

      const started = await startDiscordSession(
        'install-discord-hardware-legacy-new',
        env,
      );
      const discordApi = mockDiscordApi();
      const response = await postDiscordIssue(
        env,
        await signedIssueRequest(started, {
          code: 'discord-oauth-code-hardware-legacy',
          hardware_hash: 'hardware-hash-duplicate-legacy',
        }),
      );

      expect(response.status).toBe(409);
      await expect(response.json()).resolves.toEqual(
        normalizedErrorEnvelope({
          code: 'trial_not_eligible',
          class: 'terminal',
          subcode: 'hardware_duplicate',
          message: 'This device has already used a managed trial',
        }),
      );
      expect(discordApi.openRouterCreateCalls).toHaveLength(0);
      expect(countDiscordIdentities(env)).toBe(0);
      expect(countDiscordEntitlements(env)).toBe(1);
    },
  );

  it.each(['active', 'expired', 'revoked'] as const)(
    'hardware duplicate rejects delivered Discord %s entitlement evidence before creating a child key',
    async (storedStatus) => {
      vi.useFakeTimers();
      vi.setSystemTime(new Date(NOW_ISO));

      const env = createTestBrokerEnv();
      insertInstallation(env, {
        installationId: 'install-discord-hardware-entitlement-existing',
        devicePublicKey: 'entitlement-device-public-key',
        hardwareHash: null,
        hardwareHashSaltVersion: null,
      });
      insertEntitlement(env, {
        installation_id: 'install-discord-hardware-entitlement-existing',
        status: storedStatus,
        budget_usd: 0.07,
        managed_credential_ref: 'discord-managed-key-existing',
        issued_at: NOW_ISO,
        expires_at: '2026-07-30T06:00:00.000Z',
        verified_hardware_hash: 'hardware-hash-duplicate-entitlement',
        verified_hardware_hash_salt_version: 7,
        discord_issue_status: 'active',
        discord_issue_reserved_at: NOW_ISO,
        discord_issue_delivered_at: NOW_ISO,
      });

      const started = await startDiscordSession(
        'install-discord-hardware-entitlement-new',
        env,
      );
      const discordApi = mockDiscordApi({
        user: {
          id: discordSnowflakeForAgeDays(32),
          verified: true,
        },
      });
      const response = await postDiscordIssue(
        env,
        await signedIssueRequest(started, {
          code: 'discord-oauth-code-hardware-entitlement',
          hardware_hash: 'hardware-hash-duplicate-entitlement',
        }),
      );

      expect(response.status).toBe(409);
      await expect(response.json()).resolves.toEqual(
        normalizedErrorEnvelope({
          code: 'trial_not_eligible',
          class: 'terminal',
          subcode: 'hardware_duplicate',
          message: 'This device has already used a managed trial',
        }),
      );
      expect(discordApi.openRouterCreateCalls).toHaveLength(0);
      expect(countDiscordIdentities(env)).toBe(0);
      expect(countDiscordEntitlements(env)).toBe(1);
    },
  );

  it('same installation hardware duplicate rejects a previously delivered entitlement before creating a child key', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date(NOW_ISO));

    const env = createTestBrokerEnv();
    const installationId = 'install-discord-same-installation-delivered-hardware';
    const keyPair = await createDeviceKeyPair();
    insertInstallation(env, {
      installationId,
      devicePublicKey: keyPair.devicePublicKey,
      hardwareHash: 'hardware-hash-same-installation-delivered',
      hardwareHashSaltVersion: 7,
      appVersion: '1.0-existing-active',
      challenge: 'existing-active-challenge',
      challengeExpiresAt: '2026-04-30T06:15:00.000Z',
      challengeSaltVersion: 7,
    });
    insertEntitlement(env, {
      installation_id: installationId,
      status: 'active',
      budget_usd: 0.07,
      managed_credential_ref: 'same-installation-delivered-managed-key',
      issued_at: NOW_ISO,
      expires_at: '2026-07-30T06:00:00.000Z',
      verified_hardware_hash: 'hardware-hash-same-installation-delivered',
      verified_hardware_hash_salt_version: 7,
      discord_issue_status: 'active',
      discord_issue_reserved_at: NOW_ISO,
      discord_issue_delivered_at: NOW_ISO,
    });
    const installationBefore = await readInstallation(env, installationId);

    const started = await startDiscordSession(installationId, env, keyPair);
    const discordApi = mockDiscordApi({
      user: {
        id: discordSnowflakeForAgeDays(32),
        verified: true,
      },
    });
    const response = await postDiscordIssue(
      env,
      await signedIssueRequest(started, {
        code: 'discord-oauth-code-same-installation-delivered-hardware',
        hardware_hash: 'hardware-hash-same-installation-delivered',
      }),
    );

    expect(response.status).toBe(409);
    await expect(response.json()).resolves.toEqual(
      normalizedErrorEnvelope({
        code: 'trial_not_eligible',
        class: 'terminal',
        subcode: 'hardware_duplicate',
        message: 'This device has already used a managed trial',
      }),
    );
    expect(discordApi.openRouterCreateCalls).toHaveLength(0);
    expect(countDiscordIdentities(env)).toBe(0);
    await expect(readEntitlement(env, installationId)).resolves.toEqual(
      expect.objectContaining({
        status: 'active',
        managed_credential_ref: 'same-installation-delivered-managed-key',
        discord_issue_status: 'active',
      }),
    );
    await expect(readInstallation(env, installationId)).resolves.toEqual(
      installationBefore,
    );
  });

  it('same installation issuing conflict does not overwrite another Discord reservation or strand identity', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date(NOW_ISO));

    const env = createTestBrokerEnv();
    const installationId = 'install-discord-same-installation-issuing-conflict';
    const keyPair = await createDeviceKeyPair();
    const existingDiscordUserRef = await deriveExpectedDiscordUserRef(
      env.DISCORD_USER_REF_SECRET,
      discordSnowflakeForAgeDays(35),
    );
    insertInstallation(env, {
      installationId,
      devicePublicKey: keyPair.devicePublicKey,
      hardwareHash: 'hardware-hash-same-installation-issuing-old',
      hardwareHashSaltVersion: 7,
      appVersion: '1.0-existing-issuing',
      challenge: 'existing-issuing-challenge',
      challengeExpiresAt: '2026-04-30T06:15:00.000Z',
      challengeSaltVersion: 7,
    });
    insertDiscordIdentity(env, {
      discordUserRef: existingDiscordUserRef,
      installationId,
      status: 'issuing',
    });
    insertEntitlement(env, {
      installation_id: installationId,
      status: 'pending_release',
      budget_usd: 0.07,
      verified_hardware_hash: 'hardware-hash-same-installation-issuing-old',
      verified_hardware_hash_salt_version: 7,
      discord_user_ref: existingDiscordUserRef,
      discord_issue_status: 'issuing',
      discord_issue_reserved_at: NOW_ISO,
    });
    const installationBefore = await readInstallation(env, installationId);

    const started = await startDiscordSession(installationId, env, keyPair);
    const discordApi = mockDiscordApi({
      user: {
        id: discordSnowflakeForAgeDays(31),
        verified: true,
      },
    });
    const response = await postDiscordIssue(
      env,
      await signedIssueRequest(started, {
        code: 'discord-oauth-code-same-installation-issuing-conflict',
        hardware_hash: 'hardware-hash-same-installation-issuing-new',
      }),
    );

    expect(response.status).toBe(410);
    await expect(response.json()).resolves.toEqual(
      normalizedErrorEnvelope({
        code: 'challenge_expired',
        class: 'retryable',
        subcode: 'discord_installation_already_issuing',
        retryAfterMs: 0,
        message:
          'Discord managed entitlement is already issuing for this installation; restart Discord OAuth onboarding',
      }),
    );
    expect(discordApi.openRouterCreateCalls).toHaveLength(0);
    expect(countDiscordIdentities(env)).toBe(1);
    await expect(readSessionByState(env, started.state)).resolves.toEqual(
      expect.objectContaining({
        status: 'failed',
        pkce_code_verifier: null,
      }),
    );
    await expect(readEntitlement(env, installationId)).resolves.toEqual(
      expect.objectContaining({
        status: 'pending_release',
        discord_user_ref: existingDiscordUserRef,
        discord_issue_status: 'issuing',
        verified_hardware_hash: 'hardware-hash-same-installation-issuing-old',
      }),
    );
    await expect(readInstallation(env, installationId)).resolves.toEqual(
      installationBefore,
    );
  });

  it('cap rejects final issue when the UTC daily managed issuance cap is reached', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date(NOW_ISO));

    const env = createTestBrokerEnv();
    updateAbuseControls(env, (controls) => {
      controls.newActiveEntitlementsPerDay.maxCount = 3;
    });
    env.__db
      .prepare(
        `INSERT INTO qq_managed_entitlements (
            qq_subject_ref, status, issue_ref, managed_credential_ref,
            budget_usd, reserved_at, issued_at, expires_at, delivered_at
          ) VALUES (?, 'delivery_pending', ?, ?, ?, ?, ?, ?, NULL)`,
      )
      .run(
        'ph-qq-subject-v1_discord-cap-existing',
        'qq-issue-discord-cap-existing',
        'hash_qq_discord_cap_existing',
        0.07,
        NOW_ISO,
        NOW_ISO,
        '2026-07-30T06:00:00.000Z',
      );
    insertInstallation(env, {
      installationId: 'install-discord-cap-revoked',
      devicePublicKey: 'cap-revoked-device-public-key',
      hardwareHash: 'hardware-hash-cap-revoked',
      hardwareHashSaltVersion: 7,
    });
    insertEntitlement(env, {
      installation_id: 'install-discord-cap-revoked',
      status: 'revoked',
      budget_usd: 0.07,
      managed_credential_ref: 'hash_discord_cap_revoked',
      issued_at: NOW_ISO,
      expires_at: '2026-07-30T06:00:00.000Z',
      discord_issue_status: 'active',
      discord_issue_reserved_at: NOW_ISO,
      discord_issue_delivered_at: NOW_ISO,
    });
    insertInstallation(env, {
      installationId: 'install-discord-cap-cleanup',
      devicePublicKey: 'cap-cleanup-device-public-key',
      hardwareHash: 'hardware-hash-cap-cleanup',
      hardwareHashSaltVersion: 7,
    });
    insertEntitlement(env, {
      installation_id: 'install-discord-cap-cleanup',
      status: 'pending_release',
      budget_usd: 0.07,
      managed_credential_ref: null,
      issued_at: null,
      expires_at: null,
      discord_issue_status: 'cleanup_required',
      discord_issue_reserved_at: NOW_ISO,
      discord_issue_delivered_at: null,
    });

    const started = await startDiscordSession('install-discord-cap-new', env);
    const discordApi = mockDiscordApi();
    const response = await postDiscordIssue(
      env,
      await signedIssueRequest(started, {
        code: 'discord-oauth-code-cap',
        hardware_hash: 'hardware-hash-cap-new',
      }),
    );

    expect(response.status).toBe(503);
    await expect(response.json()).resolves.toEqual(
      normalizedErrorEnvelope({
        code: 'issuance_suspended',
        class: 'retryable',
        subcode: 'global_cap_reached',
        retryAfterMs: 64_800_000,
        message: 'Daily managed issuance cap reached',
      }),
    );
    expect(discordApi.openRouterCreateCalls).toHaveLength(0);
    await expect(readSessionByState(env, started.state)).resolves.toEqual(
      expect.objectContaining({
        status: 'failed',
        pkce_code_verifier: null,
      }),
    );
    expect(countDiscordIdentities(env)).toBe(0);
  });

  it('replay rejects repeated final issue for the same consumed state without a second key side effect', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date(NOW_ISO));

    const started = await startDiscordSession('install-discord-replay');
    const discordApi = mockDiscordApi();
    const requestBody = await signedIssueRequest(started, {
      code: 'discord-oauth-code-replay',
      hardware_hash: 'hardware-hash-replay',
    });

    const firstResponse = await postDiscordIssue(started.env, requestBody);
    const replayResponse = await postDiscordIssue(started.env, requestBody);

    expect(firstResponse.status).toBe(200);
    expect(replayResponse.status).toBe(409);
    await expect(replayResponse.json()).resolves.toEqual(
      normalizedErrorEnvelope({
        code: 'challenge_invalid',
        class: 'security_fail',
        subcode: 'discord_oauth_session_consumed',
        message: 'Discord OAuth session can no longer issue a managed key',
      }),
    );
    expect(discordApi.openRouterCreateCalls).toHaveLength(1);
    expect(countDiscordEntitlements(started.env, "discord_issue_status = 'active'")).toBe(1);
    expect(countDiscordEntitlements(started.env, "discord_issue_status = 'issuing'")).toBe(0);
  });

  it('PKCE success clears verifier, invalid signature and salt mismatch preserve verifier', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date(NOW_ISO));

    const invalidSignatureSession = await startDiscordSession('install-discord-pkce-invalid-signature');
    const invalidSignatureBody = await signedIssueRequest(invalidSignatureSession, {
      code: 'discord-oauth-code-pkce-invalid-signature',
    });
    const invalidSignature = encodeBase64Url(new Uint8Array(64).fill(9));

    const invalidSignatureResponse = await postDiscordIssue(invalidSignatureSession.env, {
      ...invalidSignatureBody,
      signature: invalidSignature,
    });

    expect(invalidSignatureResponse.status).toBe(401);
    await expect(
      readSessionByState(invalidSignatureSession.env, invalidSignatureSession.state),
    ).resolves.toEqual(
      expect.objectContaining({
        status: 'pending',
        pkce_code_verifier: expect.any(String),
      }),
    );

    const saltMismatchSession = await startDiscordSession('install-discord-pkce-salt-mismatch');
    const saltMismatchResponse = await postDiscordIssue(
      saltMismatchSession.env,
      await signedIssueRequest(saltMismatchSession, {
        code: 'discord-oauth-code-pkce-salt-mismatch',
        hardware_hash_salt_version: saltMismatchSession.fingerprintSaltVersion + 1,
      }),
    );

    expect(saltMismatchResponse.status).toBe(409);
    await expect(readSessionByState(saltMismatchSession.env, saltMismatchSession.state)).resolves.toEqual(
      expect.objectContaining({
        status: 'pending',
        pkce_code_verifier: expect.any(String),
      }),
    );

    const successSession = await startDiscordSession('install-discord-pkce-success');
    mockDiscordApi();
    const successResponse = await postDiscordIssue(
      successSession.env,
      await signedIssueRequest(successSession, {
        code: 'discord-oauth-code-pkce-success',
        hardware_hash: 'hardware-hash-pkce-success',
      }),
    );

    expect(successResponse.status).toBe(200);
    await expect(readSessionByState(successSession.env, successSession.state)).resolves.toEqual(
      expect.objectContaining({
        status: 'consumed',
        pkce_code_verifier: null,
      }),
    );
  });

  it('expiry and cancel reject before Discord token exchange, clear PKCE, and do not burn eligibility', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date(NOW_ISO));

    const expired = await startDiscordSession('install-discord-expiry');
    vi.setSystemTime(new Date('2026-04-30T06:06:00.000Z'));
    const expiredDiscordApi = mockDiscordApi();
    const expiredResponse = await postDiscordIssue(
      expired.env,
      await signedIssueRequest(expired, {
        code: 'discord-oauth-code-expiry',
        signed_at: '2026-04-30T06:06:00.000Z',
      }),
    );

    expect(expiredResponse.status).toBe(410);
    expect(expiredDiscordApi.fetchMock).not.toHaveBeenCalled();
    await expect(readSessionByState(expired.env, expired.state)).resolves.toEqual(
      expect.objectContaining({
        status: 'expired',
        pkce_code_verifier: null,
      }),
    );
    expect(countDiscordIdentities(expired.env)).toBe(0);
    expect(countDiscordEntitlements(expired.env)).toBe(0);

    vi.setSystemTime(new Date(NOW_ISO));
    const canceled = await startDiscordSession('install-discord-cancel');
    await envUpdateSessionStatus(canceled.env, canceled.state, 'canceled');
    const canceledDiscordApi = mockDiscordApi();
    const canceledResponse = await postDiscordIssue(
      canceled.env,
      await signedIssueRequest(canceled, {
        code: 'discord-oauth-code-cancel',
      }),
    );

    expect(canceledResponse.status).toBe(409);
    expect(canceledDiscordApi.fetchMock).not.toHaveBeenCalled();
    await expect(readSessionByState(canceled.env, canceled.state)).resolves.toEqual(
      expect.objectContaining({
        status: 'canceled',
        pkce_code_verifier: null,
      }),
    );
    expect(countDiscordIdentities(canceled.env)).toBe(0);
    expect(countDiscordEntitlements(canceled.env)).toBe(0);
  });

  it('reservation release removes issuing identity and entitlement when child-key creation fails before a key is returned', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date(NOW_ISO));

    const discordUserId = discordSnowflakeForAgeDays(31);
    const env = createTestBrokerEnv();
    const started = await startDiscordSession('install-discord-reservation-release', env);
    const discordApi = mockDiscordApi({
      openRouterMode: 'create_failure',
      user: {
        id: discordUserId,
        verified: true,
      },
    });
    const response = await postDiscordIssue(
      started.env,
      await signedIssueRequest(started, {
        code: 'discord-oauth-code-reservation-release',
        hardware_hash: 'hardware-hash-reservation-release',
      }),
    );

    expect(response.status).toBe(500);
    expect(await response.text()).not.toContain('or-discord-managed-child-key-test-1');
    expect(discordApi.openRouterCreateCalls).toHaveLength(1);
    expect(countDiscordIdentities(started.env)).toBe(0);
    expect(countDiscordEntitlements(started.env)).toBe(0);
    expect(countReferralCodes(started.env)).toBe(0);
    await expect(readSessionByState(started.env, started.state)).resolves.toEqual(
      expect.objectContaining({
        status: 'failed',
        pkce_code_verifier: null,
      }),
    );

    const retry = await startDiscordSession('install-discord-reservation-release-retry', env);
    mockDiscordApi({
      user: {
        id: discordUserId,
        verified: true,
      },
    });
    const retryResponse = await postDiscordIssue(
      env,
      await signedIssueRequest(retry, {
        code: 'discord-oauth-code-reservation-release-retry',
        hardware_hash: 'hardware-hash-reservation-release-retry',
      }),
    );
    expect(retryResponse.status).toBe(200);
  });

  it.each(['create_network_failure', 'create_retryable_failure'] as const)(
    'preserves lifetime blocking and notifies when child-key creation may have succeeded after %s',
    async (createMode) => {
      vi.useFakeTimers();
      vi.setSystemTime(new Date(NOW_ISO));

      const discordUserId = discordSnowflakeForAgeDays(31);
      const env = createTestBrokerEnv();
      const started = await startDiscordSession(
        'install-discord-indeterminate-child-key',
        env,
      );
      const discordApi = mockDiscordApi({
        openRouterMode: createMode,
        user: {
          id: discordUserId,
          verified: true,
        },
      });

      const response = await postDiscordIssue(
        env,
        await signedIssueRequest(started, {
          code: 'discord-oauth-code-indeterminate-child-key',
          hardware_hash: 'hardware-hash-indeterminate-child-key',
        }),
      );

      expect(response.status).toBe(500);
      expect(discordApi.openRouterCreateCalls).toHaveLength(1);
      await expect(readEntitlement(env, started.installationId)).resolves.toEqual(
        expect.objectContaining({
          status: 'pending_release',
          managed_credential_ref: null,
          discord_issue_status: 'cleanup_required',
        }),
      );
      const expectedDiscordUserRef = await deriveExpectedDiscordUserRef(
        env.DISCORD_USER_REF_SECRET,
        discordUserId,
      );
      await expect(readDiscordIdentity(env, expectedDiscordUserRef)).resolves.toEqual(
        expect.objectContaining({ status: 'cleanup_required' }),
      );
      const cleanupIncidentCalls = discordApi.fetchMock.mock.calls.filter(
        ([request]) => String(request) === env.DISCORD_IMMEDIATE_ALERT_WEBHOOK_URL,
      );
      expect(cleanupIncidentCalls).toHaveLength(1);
      expect(String(cleanupIncidentCalls[0]?.[1]?.body)).toContain(
        'Broker managed-key cleanup incident',
      );

      const retry = await startDiscordSession(
        'install-discord-indeterminate-child-key-retry',
        env,
      );
      const retryDiscordApi = mockDiscordApi({
        user: {
          id: discordUserId,
          verified: true,
        },
      });
      const retryResponse = await postDiscordIssue(
        env,
        await signedIssueRequest(retry, {
          code: 'discord-oauth-code-indeterminate-child-key-retry',
          hardware_hash: 'hardware-hash-indeterminate-child-key-retry',
        }),
      );

      expect(retryResponse.status).toBe(409);
      expect(retryDiscordApi.openRouterCreateCalls).toHaveLength(0);
    },
  );

  it('guardrail assignment failure after child-key creation cleans up and releases Discord eligibility for fresh OAuth', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date(NOW_ISO));

    const env = createTestBrokerEnv();
    const discordUserId = discordSnowflakeForAgeDays(31);
    const started = await startDiscordSession('install-discord-guardrail-cleanup-success', env);
    const discordApi = mockDiscordApi({
      openRouterMode: 'guardrail_failure',
      user: {
        id: discordUserId,
        verified: true,
      },
    });

    const response = await postDiscordIssue(
      env,
      await signedIssueRequest(started, {
        code: 'discord-oauth-code-guardrail-cleanup-success',
        hardware_hash: 'hardware-hash-guardrail-cleanup-success',
      }),
    );

    expect(response.status).toBe(500);
    expect(await response.text()).not.toContain('or-discord-managed-child-key-test-1');
    expect(discordApi.openRouterCreateCalls).toHaveLength(1);
    expect(discordApi.openRouterGuardrailCalls).toHaveLength(1);
    expect(discordApi.openRouterCleanupCalls.map(({ init }) => init?.method)).toEqual([
      'PATCH',
      'DELETE',
    ]);
    expect(countDiscordIdentities(env)).toBe(0);
    expect(countDiscordEntitlements(env)).toBe(0);
    await expect(readSessionByState(env, started.state)).resolves.toEqual(
      expect.objectContaining({
        status: 'failed',
        pkce_code_verifier: null,
      }),
    );

    const retry = await startDiscordSession('install-discord-guardrail-cleanup-success-retry', env);
    mockDiscordApi({
      user: {
        id: discordUserId,
        verified: true,
      },
    });
    const retryResponse = await postDiscordIssue(
      env,
      await signedIssueRequest(retry, {
        code: 'discord-oauth-code-guardrail-cleanup-success-retry',
        hardware_hash: 'hardware-hash-guardrail-cleanup-success-retry',
      }),
    );
    expect(retryResponse.status).toBe(200);
  });

  it('keeps Discord release state atomic and notifies when local cleanup persistence fails', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date(NOW_ISO));

    const env = createTestBrokerEnv({
      beforeRun: ({ sql }) => {
        if (
          sql.includes('DELETE FROM openrouter_entitlements') &&
          sql.includes("discord_issue_status IN ('issuing', 'delivery_pending', 'active', 'cleanup_required')")
        ) {
          throw new Error('synthetic Discord release persistence failure');
        }
      },
    });
    const discordUserId = discordSnowflakeForAgeDays(31);
    const expectedDiscordUserRef = await deriveExpectedDiscordUserRef(
      env.DISCORD_USER_REF_SECRET,
      discordUserId,
    );
    const started = await startDiscordSession(
      'install-discord-release-persistence-failure',
      env,
    );
    const discordApi = mockDiscordApi({
      openRouterMode: 'guardrail_failure',
      user: {
        id: discordUserId,
        verified: true,
      },
    });

    const response = await postDiscordIssue(
      env,
      await signedIssueRequest(started, {
        code: 'discord-oauth-code-release-persistence-failure',
        hardware_hash: 'hardware-hash-release-persistence-failure',
      }),
    );

    expect(response.status).toBe(500);
    expect(discordApi.openRouterCleanupCalls.map(({ init }) => init?.method)).toEqual([
      'PATCH',
      'DELETE',
    ]);
    await expect(readEntitlement(env, started.installationId)).resolves.toEqual(
      expect.objectContaining({
        status: 'pending_release',
        discord_issue_status: 'issuing',
      }),
    );
    await expect(readDiscordIdentity(env, expectedDiscordUserRef)).resolves.toEqual(
      expect.objectContaining({ status: 'issuing' }),
    );
    const cleanupIncidentCalls = discordApi.fetchMock.mock.calls.filter(
      ([request]) => String(request) === env.DISCORD_IMMEDIATE_ALERT_WEBHOOK_URL,
    );
    expect(cleanupIncidentCalls).toHaveLength(1);
    expect(String(cleanupIncidentCalls[0]?.[1]?.body)).toContain(
      'cleanup_required state could not be confirmed',
    );
  });

  it('guardrail assignment failure with cleanup failure records cleanup_required without leaking sensitive values', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date(NOW_ISO));

    const rawCode = 'discord-oauth-code-guardrail-cleanup-redact';
    const rawDiscordUserId = discordSnowflakeForAgeDays(31);
    const rawEmail = 'sensitive-redaction@example.test';
    const rawAccessToken = 'discord-access-token-sensitive-redact';
    const rawRefreshToken = 'discord-refresh-token-sensitive-redact';
    const rawOpenRouterChildKey = 'or-discord-managed-child-key-sensitive-redact';
    const childKeyHash = 'hash_discord_managed_child_cleanup_required';
    const env = createTestBrokerEnv();
    const started = await startDiscordSession('install-discord-guardrail-cleanup-required', env);
    const sessionBefore = await readSessionByState(env, started.state);
    const expectedDiscordUserRef = await deriveExpectedDiscordUserRef(
      env.DISCORD_USER_REF_SECRET,
      rawDiscordUserId,
    );
    const sensitiveValues = [
      rawCode,
      started.state,
      sessionBefore.pkce_code_verifier,
      rawAccessToken,
      rawRefreshToken,
      rawDiscordUserId,
      rawEmail,
      rawOpenRouterChildKey,
    ].filter((value): value is string => value !== null);
    const consoleErrorSpy = vi.spyOn(console, 'error').mockImplementation(() => undefined);
    const discordApi = mockDiscordApi({
      openRouterMode: 'guardrail_failure_cleanup_failure',
      rawChildKey: rawOpenRouterChildKey,
      childKeyHash,
      accessToken: rawAccessToken,
      refreshToken: rawRefreshToken,
      user: {
        id: rawDiscordUserId,
        verified: true,
        email: rawEmail,
      },
      guardrailFailureMessage: `guardrail failed ${rawCode} ${started.state} ${sessionBefore.pkce_code_verifier} ${rawAccessToken} ${rawRefreshToken} ${rawDiscordUserId} ${rawEmail} ${rawOpenRouterChildKey}`,
      cleanupFailureMessage: `cleanup failed ${rawCode} ${started.state} ${sessionBefore.pkce_code_verifier} ${rawAccessToken} ${rawRefreshToken} ${rawDiscordUserId} ${rawEmail} ${rawOpenRouterChildKey}`,
    });

    const response = await postDiscordIssue(
      env,
      await signedIssueRequest(started, {
        code: rawCode,
        hardware_hash: 'hardware-hash-guardrail-cleanup-required',
      }),
    );

    expect(response.status).toBe(500);
    const responseText = await response.text();
    expectTextNotToContainSensitiveValues(responseText, sensitiveValues);
    expectTextNotToContainSensitiveValues(
      stringifyConsoleCalls(consoleErrorSpy),
      sensitiveValues,
    );
    expect(discordApi.openRouterCreateCalls).toHaveLength(1);
    expect(discordApi.openRouterGuardrailCalls).toHaveLength(1);
    expect(discordApi.openRouterCleanupCalls.map(({ init }) => init?.method)).toEqual([
      'PATCH',
      'DELETE',
    ]);
    await expect(readEntitlement(env, started.installationId)).resolves.toEqual(
      expect.objectContaining({
        status: 'pending_release',
        managed_credential_ref: childKeyHash,
        issued_at: null,
        expires_at: null,
        discord_user_ref: expectedDiscordUserRef,
        discord_issue_status: 'cleanup_required',
        discord_issue_delivered_at: null,
      }),
    );
    await expect(readDiscordIdentity(env, expectedDiscordUserRef)).resolves.toEqual(
      expect.objectContaining({
        entitlement_installation_id: started.installationId,
        status: 'cleanup_required',
      }),
    );
    const cleanupIncidentCalls = discordApi.fetchMock.mock.calls.filter(
      ([input]) => String(input) === env.DISCORD_IMMEDIATE_ALERT_WEBHOOK_URL,
    );
    expect(cleanupIncidentCalls).toHaveLength(1);
    const cleanupIncidentBody = String(
      (cleanupIncidentCalls[0]?.[1] as RequestInit | undefined)?.body,
    );
    expect(cleanupIncidentBody).toContain('Broker managed-key cleanup incident');
    expect(cleanupIncidentBody).toContain('cleanup_required');
    expectTextNotToContainSensitiveValues(cleanupIncidentBody, sensitiveValues);
    await expect(readSessionByState(env, started.state)).resolves.toEqual(
      expect.objectContaining({
        status: 'failed',
        pkce_code_verifier: null,
      }),
    );
  });

  it('cleanup_required blocks same-installation reservation by a different Discord account without overwriting orphan metadata', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date(NOW_ISO));

    const env = createTestBrokerEnv();
    const installationId = 'install-discord-cleanup-required-same-installation';
    const keyPair = await createDeviceKeyPair();
    const orphanedCredentialRef = 'hash_discord_orphaned_cleanup_required_same_install';
    const existingDiscordUserRef = await deriveExpectedDiscordUserRef(
      env.DISCORD_USER_REF_SECRET,
      discordSnowflakeForAgeDays(45),
    );
    const newDiscordUserId = discordSnowflakeForAgeDays(31);
    const newDiscordUserRef = await deriveExpectedDiscordUserRef(
      env.DISCORD_USER_REF_SECRET,
      newDiscordUserId,
    );
    insertInstallation(env, {
      installationId,
      devicePublicKey: keyPair.devicePublicKey,
      hardwareHash: 'hardware-hash-cleanup-required-existing',
      hardwareHashSaltVersion: 7,
    });
    insertDiscordIdentity(env, {
      discordUserRef: existingDiscordUserRef,
      installationId,
      status: 'cleanup_required',
    });
    insertEntitlement(env, {
      installation_id: installationId,
      status: 'pending_release',
      budget_usd: 0.07,
      managed_credential_ref: orphanedCredentialRef,
      verified_hardware_hash: 'hardware-hash-cleanup-required-existing',
      verified_hardware_hash_salt_version: 7,
      discord_user_ref: existingDiscordUserRef,
      discord_issue_status: 'cleanup_required',
      discord_issue_reserved_at: NOW_ISO,
    });
    const entitlementBefore = await readEntitlement(env, installationId);
    const installationBefore = await readInstallation(env, installationId);

    const started = await startDiscordSession(installationId, env, keyPair);
    const discordApi = mockDiscordApi({
      user: {
        id: newDiscordUserId,
        verified: true,
      },
    });
    const response = await postDiscordIssue(
      env,
      await signedIssueRequest(started, {
        code: 'discord-oauth-code-cleanup-required-same-installation',
        hardware_hash: 'hardware-hash-cleanup-required-new-attempt',
      }),
    );

    expect(response.status).toBe(410);
    expect(discordApi.openRouterCreateCalls).toHaveLength(0);
    await expect(readEntitlement(env, installationId)).resolves.toEqual(entitlementBefore);
    await expect(readInstallation(env, installationId)).resolves.toEqual(
      installationBefore,
    );
    await expect(readDiscordIdentity(env, existingDiscordUserRef)).resolves.toEqual(
      expect.objectContaining({
        status: 'cleanup_required',
        entitlement_installation_id: installationId,
      }),
    );
    await expect(readDiscordIdentity(env, newDiscordUserRef)).resolves.toBeNull();
  });

  it('cleanup_required blocks same-hardware reservation from another installation without creating a new key', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date(NOW_ISO));

    const env = createTestBrokerEnv();
    const existingInstallationId = 'install-discord-cleanup-required-existing-hardware';
    const newInstallationId = 'install-discord-cleanup-required-same-hardware-new';
    const cleanupHardwareHash = 'hardware-hash-cleanup-required-shared';
    const existingDiscordUserRef = await deriveExpectedDiscordUserRef(
      env.DISCORD_USER_REF_SECRET,
      discordSnowflakeForAgeDays(46),
    );
    insertInstallation(env, {
      installationId: existingInstallationId,
      devicePublicKey: 'cleanup-required-existing-hardware-device-key',
      hardwareHash: cleanupHardwareHash,
      hardwareHashSaltVersion: 7,
    });
    insertDiscordIdentity(env, {
      discordUserRef: existingDiscordUserRef,
      installationId: existingInstallationId,
      status: 'cleanup_required',
    });
    insertEntitlement(env, {
      installation_id: existingInstallationId,
      status: 'pending_release',
      budget_usd: 0.07,
      managed_credential_ref: 'hash_discord_orphaned_cleanup_required_same_hardware',
      verified_hardware_hash: cleanupHardwareHash,
      verified_hardware_hash_salt_version: 7,
      discord_user_ref: existingDiscordUserRef,
      discord_issue_status: 'cleanup_required',
      discord_issue_reserved_at: NOW_ISO,
    });
    const cleanupRequiredBefore = await readEntitlement(env, existingInstallationId);

    const started = await startDiscordSession(newInstallationId, env);
    const discordApi = mockDiscordApi({
      user: {
        id: discordSnowflakeForAgeDays(31),
        verified: true,
      },
    });
    const response = await postDiscordIssue(
      env,
      await signedIssueRequest(started, {
        code: 'discord-oauth-code-cleanup-required-same-hardware',
        hardware_hash: cleanupHardwareHash,
      }),
    );

    expect(response.status).toBe(409);
    await expect(response.json()).resolves.toEqual(
      normalizedErrorEnvelope({
        code: 'trial_not_eligible',
        class: 'terminal',
        subcode: 'hardware_duplicate',
        message: 'This device has already used a managed trial',
      }),
    );
    expect(discordApi.openRouterCreateCalls).toHaveLength(0);
    await expect(readEntitlement(env, existingInstallationId)).resolves.toEqual(
      cleanupRequiredBefore,
    );
    await expect(readEntitlement(env, newInstallationId)).resolves.toBeNull();
    expect(countDiscordIdentities(env)).toBe(1);
  });

  it('local session cleanup failure after child-key creation still attempts remote cleanup and does not leak raw key', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date(NOW_ISO));

    const rawCode = 'discord-oauth-code-local-session-cleanup-redact';
    const rawDiscordUserId = discordSnowflakeForAgeDays(31);
    const rawEmail = 'local-session-cleanup@example.test';
    const rawAccessToken = 'discord-access-token-local-session-cleanup-redact';
    const rawRefreshToken = 'discord-refresh-token-local-session-cleanup-redact';
    const rawOpenRouterChildKey = 'or-discord-managed-child-key-local-cleanup-redact';
    const childKeyHash = 'hash_discord_managed_child_local_cleanup';
    let rawState = '';
    let rawPkceVerifier: string | null = null;
    let failSessionCleanup = false;
    const env = createTestBrokerEnv({
      beforeRun: ({ sql }) => {
        if (
          failSessionCleanup &&
          sql.includes('UPDATE discord_oauth_sessions') &&
          sql.includes("status = 'failed'")
        ) {
          throw new Error(
            `session cleanup failed ${rawCode} ${rawState} ${rawPkceVerifier} ${rawAccessToken} ${rawRefreshToken} ${rawDiscordUserId} ${rawEmail} ${rawOpenRouterChildKey}`,
          );
        }
      },
    });
    const started = await startDiscordSession('install-discord-local-session-cleanup', env);
    rawState = started.state;
    rawPkceVerifier = (await readSessionByState(env, started.state)).pkce_code_verifier;
    const sensitiveValues = [
      rawCode,
      rawState,
      rawPkceVerifier,
      rawAccessToken,
      rawRefreshToken,
      rawDiscordUserId,
      rawEmail,
      rawOpenRouterChildKey,
    ].filter((value): value is string => value !== null);
    const consoleErrorSpy = vi.spyOn(console, 'error').mockImplementation(() => undefined);
    const discordApi = mockDiscordApi({
      openRouterMode: 'guardrail_failure',
      rawChildKey: rawOpenRouterChildKey,
      childKeyHash,
      accessToken: rawAccessToken,
      refreshToken: rawRefreshToken,
      user: {
        id: rawDiscordUserId,
        verified: true,
        email: rawEmail,
      },
    });
    failSessionCleanup = true;

    const response = await postDiscordIssue(
      env,
      await signedIssueRequest(started, {
        code: rawCode,
        hardware_hash: 'hardware-hash-local-session-cleanup',
      }),
    );

    expect(response.status).toBe(500);
    const responseText = await response.text();
    expectTextNotToContainSensitiveValues(responseText, sensitiveValues);
    expectTextNotToContainSensitiveValues(
      stringifyConsoleCalls(consoleErrorSpy),
      sensitiveValues,
    );
    expect(discordApi.openRouterCreateCalls).toHaveLength(1);
    expect(discordApi.openRouterGuardrailCalls).toHaveLength(1);
    expect(discordApi.openRouterCleanupCalls.map(({ init }) => init?.method)).toEqual([
      'PATCH',
      'DELETE',
    ]);
    await expect(readEntitlement(env, started.installationId)).resolves.toBeNull();
  });

  it('issue-success recording failure after child-key creation cleans up and releases eligibility without leaking sensitive values', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date(NOW_ISO));

    const rawCode = 'discord-oauth-code-monitoring-record-redact';
    const rawDiscordUserId = discordSnowflakeForAgeDays(31);
    const rawEmail = 'monitoring-record-redaction@example.test';
    const rawAccessToken = 'discord-access-token-monitoring-record-redact';
    const rawRefreshToken = 'discord-refresh-token-monitoring-record-redact';
    const rawOpenRouterChildKey = 'or-discord-managed-child-key-monitoring-record-redact';
    const childKeyHash = 'hash_discord_managed_child_monitoring_record';
    let rawState = '';
    let rawPkceVerifier: string | null = null;
    const env = createTestBrokerEnv({
      beforeRun: ({ sql }) => {
        if (sql.includes('INSERT INTO broker_issue_success_events')) {
          throw new Error(
            `record failed ${rawCode} ${rawState} ${rawPkceVerifier} ${rawAccessToken} ${rawRefreshToken} ${rawDiscordUserId} ${rawEmail} ${rawOpenRouterChildKey}`,
          );
        }
      },
    });
    const started = await startDiscordSession('install-discord-monitoring-record-release', env);
    rawState = started.state;
    rawPkceVerifier = (await readSessionByState(env, started.state)).pkce_code_verifier;
    const sensitiveValues = [
      rawCode,
      rawState,
      rawPkceVerifier,
      rawAccessToken,
      rawRefreshToken,
      rawDiscordUserId,
      rawEmail,
      rawOpenRouterChildKey,
    ].filter((value): value is string => value !== null);
    const consoleErrorSpy = vi.spyOn(console, 'error').mockImplementation(() => undefined);
    const discordApi = mockDiscordApi({
      rawChildKey: rawOpenRouterChildKey,
      childKeyHash,
      accessToken: rawAccessToken,
      refreshToken: rawRefreshToken,
      user: {
        id: rawDiscordUserId,
        verified: true,
        email: rawEmail,
      },
    });

    const response = await postDiscordIssue(
      env,
      await signedIssueRequest(started, {
        code: rawCode,
        hardware_hash: 'hardware-hash-monitoring-record-release',
      }),
    );

    expect(response.status).toBe(500);
    const responseText = await response.text();
    expectTextNotToContainSensitiveValues(responseText, sensitiveValues);
    expectTextNotToContainSensitiveValues(
      stringifyConsoleCalls(consoleErrorSpy),
      sensitiveValues,
    );
    expect(discordApi.openRouterCreateCalls).toHaveLength(1);
    expect(discordApi.openRouterGuardrailCalls).toHaveLength(1);
    expect(discordApi.openRouterCleanupCalls.map(({ init }) => init?.method)).toEqual([
      'PATCH',
      'DELETE',
    ]);
    expect(readIssueSuccessEvents(env)).toEqual([]);
    expect(countDiscordIdentities(env)).toBe(0);
    expect(countDiscordEntitlements(env)).toBe(0);
    await expect(readSessionByState(env, started.state)).resolves.toEqual(
      expect.objectContaining({
        status: 'failed',
        pkce_code_verifier: null,
      }),
    );
  });

  it('issue-success monitoring failure with cleanup failure records cleanup_required without leaking sensitive values', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date(NOW_ISO));

    const rawCode = 'discord-oauth-code-monitoring-evaluate-redact';
    const rawDiscordUserId = discordSnowflakeForAgeDays(31);
    const rawEmail = 'monitoring-evaluate-redaction@example.test';
    const rawAccessToken = 'discord-access-token-monitoring-evaluate-redact';
    const rawRefreshToken = 'discord-refresh-token-monitoring-evaluate-redact';
    const rawOpenRouterChildKey = 'or-discord-managed-child-key-monitoring-evaluate-redact';
    const childKeyHash = 'hash_discord_managed_child_monitoring_cleanup_required';
    let rawState = '';
    let rawPkceVerifier: string | null = null;
    let failRuntimeStateUpdate = false;
    const env = createTestBrokerEnv({
      beforeRun: ({ sql }) => {
        if (failRuntimeStateUpdate && sql.includes('UPDATE broker_config')) {
          throw new Error(
            `evaluate failed ${rawCode} ${rawState} ${rawPkceVerifier} ${rawAccessToken} ${rawRefreshToken} ${rawDiscordUserId} ${rawEmail} ${rawOpenRouterChildKey}`,
          );
        }
      },
    });
    updateAbuseControls(env, (controls) => {
      controls.immediateAlerts.warning = 1;
      controls.immediateAlerts.brake = 4;
    });
    insertInstallation(env, {
      installationId: 'install-discord-monitoring-existing',
      devicePublicKey: 'monitoring-existing-device-public-key',
      hardwareHash: 'hardware-hash-monitoring-existing',
      hardwareHashSaltVersion: 7,
    });
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
      .run(
        'install-discord-monitoring-existing',
        'install-discord-monitoring-existing',
        'existing-monitoring-event',
        NOW_ISO,
      );
    const started = await startDiscordSession('install-discord-monitoring-cleanup-required', env);
    rawState = started.state;
    rawPkceVerifier = (await readSessionByState(env, started.state)).pkce_code_verifier;
    const expectedDiscordUserRef = await deriveExpectedDiscordUserRef(
      env.DISCORD_USER_REF_SECRET,
      rawDiscordUserId,
    );
    const sensitiveValues = [
      rawCode,
      rawState,
      rawPkceVerifier,
      rawAccessToken,
      rawRefreshToken,
      rawDiscordUserId,
      rawEmail,
      rawOpenRouterChildKey,
    ].filter((value): value is string => value !== null);
    const consoleErrorSpy = vi.spyOn(console, 'error').mockImplementation(() => undefined);
    const discordApi = mockDiscordApi({
      openRouterMode: 'cleanup_failure',
      rawChildKey: rawOpenRouterChildKey,
      childKeyHash,
      accessToken: rawAccessToken,
      refreshToken: rawRefreshToken,
      user: {
        id: rawDiscordUserId,
        verified: true,
        email: rawEmail,
      },
      cleanupFailureMessage: `cleanup failed ${rawCode} ${rawState} ${rawPkceVerifier} ${rawAccessToken} ${rawRefreshToken} ${rawDiscordUserId} ${rawEmail} ${rawOpenRouterChildKey}`,
    });
    failRuntimeStateUpdate = true;

    const response = await postDiscordIssue(
      env,
      await signedIssueRequest(started, {
        code: rawCode,
        hardware_hash: 'hardware-hash-monitoring-cleanup-required',
      }),
    );

    expect(response.status).toBe(500);
    const responseText = await response.text();
    expectTextNotToContainSensitiveValues(responseText, sensitiveValues);
    expectTextNotToContainSensitiveValues(
      stringifyConsoleCalls(consoleErrorSpy),
      sensitiveValues,
    );
    expect(discordApi.openRouterCreateCalls).toHaveLength(1);
    expect(discordApi.openRouterGuardrailCalls).toHaveLength(1);
    expect(discordApi.openRouterCleanupCalls.map(({ init }) => init?.method)).toEqual([
      'PATCH',
      'DELETE',
    ]);
    expect(
      readIssueSuccessEvents(env).some(
        (event) => event.managed_credential_ref === childKeyHash,
      ),
    ).toBe(false);
    await expect(readEntitlement(env, started.installationId)).resolves.toEqual(
      expect.objectContaining({
        status: 'pending_release',
        managed_credential_ref: childKeyHash,
        issued_at: null,
        expires_at: null,
        discord_user_ref: expectedDiscordUserRef,
        discord_issue_status: 'cleanup_required',
        discord_issue_delivered_at: null,
      }),
    );
    await expect(readDiscordIdentity(env, expectedDiscordUserRef)).resolves.toEqual(
      expect.objectContaining({
        entitlement_installation_id: started.installationId,
        status: 'cleanup_required',
      }),
    );
    await expect(readSessionByState(env, started.state)).resolves.toEqual(
      expect.objectContaining({
        status: 'failed',
        pkce_code_verifier: null,
      }),
    );
  });
});

describe('Discord eligibility', () => {
  afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
    vi.useRealTimers();
  });

  it('PKCE policy terminal rejection clears verifier for Discord accounts without verified email', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date(NOW_ISO));

    const started = await startDiscordSession('install-discord-eligibility-unverified');
    mockDiscordApi({
      user: {
        id: discordSnowflakeForAgeDays(31),
        verified: false,
      },
    });
    const requestBody = await signedIssueRequest(started);

    const response = await postDiscordIssue(started.env, requestBody);

    expect(response.status).toBe(409);
    await expect(response.json()).resolves.toEqual(
      normalizedErrorEnvelope({
        code: 'trial_not_eligible',
        class: 'terminal',
        subcode: 'discord_email_unverified',
        message: 'Discord email verification is required',
      }),
    );
    await expect(readSessionByState(started.env, started.state)).resolves.toEqual(
      expect.objectContaining({
        status: 'failed',
        pkce_code_verifier: null,
      }),
    );
  });

  it('rejects Discord accounts younger than 30 days', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date(NOW_ISO));

    const started = await startDiscordSession('install-discord-eligibility-too-new');
    mockDiscordApi({
      user: {
        id: discordSnowflakeForAgeMs(THIRTY_DAYS_MS - 1),
        verified: true,
      },
    });
    const requestBody = await signedIssueRequest(started);

    const response = await postDiscordIssue(started.env, requestBody);

    expect(response.status).toBe(409);
    await expect(response.json()).resolves.toEqual(
      normalizedErrorEnvelope({
        code: 'trial_not_eligible',
        class: 'terminal',
        subcode: 'discord_account_too_new',
        message: 'Discord account must be at least 30 days old',
      }),
    );
  });

  it('allows the exact 30-day Discord account age boundary to activate a managed key', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date(NOW_ISO));

    const started = await startDiscordSession('install-discord-eligibility-boundary');
    mockDiscordApi({
      user: {
        id: discordSnowflakeForAgeMs(THIRTY_DAYS_MS),
        verified: true,
      },
    });
    const requestBody = await signedIssueRequest(started);

    const response = await postDiscordIssue(started.env, requestBody);

    expect(response.status).toBe(200);
    await expect(readSessionByState(started.env, started.state)).resolves.toEqual(
      expect.objectContaining({
        status: 'consumed',
        pkce_code_verifier: null,
        eligibility_checked_at: NOW_ISO,
      }),
    );
  });

  it('rejects invalid Discord snowflakes and clears the failed session PKCE verifier', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date(NOW_ISO));

    const started = await startDiscordSession('install-discord-eligibility-invalid-snowflake');
    mockDiscordApi({
      user: {
        id: 'not-a-snowflake',
        verified: true,
      },
    });
    const requestBody = await signedIssueRequest(started);

    const response = await postDiscordIssue(started.env, requestBody);

    expect(response.status).toBe(409);
    await expect(response.json()).resolves.toEqual(
      normalizedErrorEnvelope({
        code: 'trial_not_eligible',
        class: 'terminal',
        subcode: 'discord_invalid_snowflake',
        message: 'Discord account identity is invalid',
      }),
    );
    await expect(readSessionByState(started.env, started.state)).resolves.toEqual(
      expect.objectContaining({
        status: 'failed',
        pkce_code_verifier: null,
      }),
    );
  });
});

async function startDiscordSession(
  installationId: string,
  env: TestBrokerEnv = createTestBrokerEnv(),
  keyPair?: DeviceKeyPair,
): Promise<StartedDiscordSession> {
  const sessionKeyPair = keyPair ?? (await createDeviceKeyPair());
  const response = await postDiscordStart(env, {
    installation_id: installationId,
    device_public_key: sessionKeyPair.devicePublicKey,
    redirect_uri: REGISTERED_REDIRECT_URI,
    app_version: APP_VERSION,
  });

  if (response.status !== 200) {
    throw new Error(`Discord start failed with status ${response.status}`);
  }

  const payload = (await response.json()) as {
    authorization_url: string;
    issue_nonce: string;
    redirect_uri: string;
    fingerprint_salt_version: number;
  };
  const state = new URL(payload.authorization_url).searchParams.get('state');
  if (!state) {
    throw new Error('Discord authorization URL did not include state');
  }

  return {
    env,
    keyPair: sessionKeyPair,
    installationId,
    state,
    issueNonce: payload.issue_nonce,
    redirectUri: payload.redirect_uri,
    appVersion: APP_VERSION,
    fingerprintSaltVersion: payload.fingerprint_salt_version,
  };
}

async function signedIssueRequest(
  started: StartedDiscordSession,
  overrides: Partial<SignedDiscordIssueRequestInput> = {},
): Promise<
  SignedDiscordIssueRequestInput & {
    signature_alg: 'ed25519';
    signature: string;
  }
> {
  return signCanonicalDiscordIssueRequest(started.keyPair.privateKey, {
    installation_id: started.installationId,
    device_public_key: started.keyPair.devicePublicKey,
    state: started.state,
    code: 'discord-oauth-code',
    redirect_uri: started.redirectUri,
    hardware_hash: 'hardware-hash-discord-issue',
    hardware_hash_salt_version: started.fingerprintSaltVersion,
    app_version: started.appVersion,
    reason: 'llm_start',
    budget_usd: 0.07,
    model: MODEL,
    issue_nonce: started.issueNonce,
    signed_at: SIGNED_AT_ISO,
    ...overrides,
  });
}

async function readSessionByState(
  env: TestBrokerEnv,
  state: string,
): Promise<DiscordSessionRow> {
  const stateHash = await sha256Base64Url(state);
  const row = env.__db
    .prepare(
      `SELECT installation_id,
              device_public_key,
              redirect_uri,
              pkce_code_verifier,
              issue_nonce_hash,
              fingerprint_salt_version,
              status,
              processing_started_at,
              eligibility_checked_at,
              consumed_at
         FROM discord_oauth_sessions
        WHERE state_hash = ?`,
    )
    .get(stateHash) as DiscordSessionRow | undefined;

  if (!row) {
    throw new Error('Discord OAuth session row was not found');
  }

  return row;
}

async function readDiscordIdentity(
  env: TestBrokerEnv,
  discordUserRef: string,
): Promise<DiscordIdentityRow | null> {
  const row = env.__db
    .prepare(
      `SELECT discord_user_ref,
              entitlement_installation_id,
              status,
              updated_at
         FROM discord_identities
        WHERE discord_user_ref = ?`,
    )
    .get(discordUserRef) as DiscordIdentityRow | undefined;

  return row ?? null;
}

function readIssueSuccessEvents(env: TestBrokerEnv): IssueSuccessEventRow[] {
  return env.__db
    .prepare(
      `SELECT issue_source,
              installation_id,
              subject_ref,
              managed_credential_ref,
              ip_hash,
              ip_prefix_hash,
              observed_at
         FROM broker_issue_success_events
        ORDER BY observed_at ASC`,
    )
    .all() as unknown as IssueSuccessEventRow[];
}

function readReferralCodeForOwner(
  env: TestBrokerEnv,
  discordUserRef: string,
): ReferralCodeRow | null {
  const row = env.__db
    .prepare(
      `SELECT referral_id,
              owner_discord_user_ref,
              owner_installation_id,
              status
         FROM referral_codes
        WHERE owner_discord_user_ref = ?`,
    )
    .get(discordUserRef) as ReferralCodeRow | undefined;

  return row ?? null;
}

async function readEntitlement(
  env: TestBrokerEnv,
  installationId: string,
): Promise<Record<string, unknown> | null> {
  const row = env.__db
    .prepare(
      `SELECT installation_id,
              status,
              budget_usd,
              managed_credential_ref,
              issued_at,
              expires_at,
              verified_hardware_hash,
              verified_hardware_hash_salt_version,
              discord_user_ref,
              discord_issue_status,
              discord_issue_reserved_at,
              discord_issue_delivered_at
         FROM openrouter_entitlements
        WHERE installation_id = ?`,
    )
    .get(installationId) as Record<string, unknown> | undefined;
  return row ?? null;
}

async function readInstallation(
  env: TestBrokerEnv,
  installationId: string,
): Promise<Record<string, unknown> | null> {
  const row = env.__db
    .prepare(
      `SELECT installation_id,
              device_public_key,
              hardware_hash,
              hardware_hash_salt_version,
              app_version,
              challenge,
              challenge_expires_at,
              challenge_salt_version,
              created_at,
              last_seen_at
         FROM installations
        WHERE installation_id = ?`,
    )
    .get(installationId) as Record<string, unknown> | undefined;
  return row ?? null;
}

function insertInstallation(
  env: TestBrokerEnv,
  input: {
    installationId: string;
    devicePublicKey: string;
    hardwareHash: string | null;
    hardwareHashSaltVersion: number | null;
    appVersion?: string;
    challenge?: string | null;
    challengeExpiresAt?: string | null;
    challengeSaltVersion?: number | null;
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
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
    )
    .run(
      input.installationId,
      input.devicePublicKey,
      input.hardwareHash,
      input.hardwareHashSaltVersion,
      input.appVersion ?? APP_VERSION,
      input.challenge ?? null,
      input.challengeExpiresAt ?? null,
      input.challengeSaltVersion ?? null,
      NOW_ISO,
      NOW_ISO,
    );
}

function insertDiscordIdentity(
  env: TestBrokerEnv,
  input: {
    discordUserRef: string;
    installationId: string;
    status: 'issuing' | 'active' | 'failed' | 'cleanup_required';
  },
): void {
  env.__db
    .prepare(
      `INSERT INTO discord_identities (
          discord_user_ref,
          entitlement_installation_id,
          status,
          ref_secret_version,
          created_at,
          updated_at
        ) VALUES (?, ?, ?, 1, ?, ?)`,
    )
    .run(
      input.discordUserRef,
      input.installationId,
      input.status,
      NOW_ISO,
      NOW_ISO,
    );
}

function insertRequestEvent(
  env: TestBrokerEnv,
  input: {
    endpoint: string;
    ip: string | null;
    installationId: string | null;
    observedAt: string;
  },
): void {
  env.__db
    .prepare(
      `INSERT INTO broker_request_events (
          endpoint,
          ip,
          installation_id,
          observed_at
        ) VALUES (?, ?, ?, ?)`,
    )
    .run(input.endpoint, input.ip, input.installationId, input.observedAt);
}

async function postDiscordIssueWithIp(
  env: TestBrokerEnv,
  body: object | string,
  ip: string,
): Promise<Response> {
  return app.request(
    'http://broker.test/v1/providers/openrouter/discord/issue',
    {
      method: 'POST',
      headers: {
        'content-type': 'application/json',
        'cf-connecting-ip': ip,
      },
      body: typeof body === 'string' ? body : JSON.stringify(body),
    },
    env,
  );
}

function countDiscordIdentities(env: TestBrokerEnv): number {
  const row = env.__db
    .prepare('SELECT COUNT(*) AS count FROM discord_identities')
    .get() as { count: number };
  return Number(row.count);
}

function countDiscordEntitlements(env: TestBrokerEnv, where = '1 = 1'): number {
  const row = env.__db
    .prepare(`SELECT COUNT(*) AS count FROM openrouter_entitlements WHERE ${where}`)
    .get() as { count: number };
  return Number(row.count);
}

function countReferralCodes(env: TestBrokerEnv): number {
  const row = env.__db
    .prepare('SELECT COUNT(*) AS count FROM referral_codes')
    .get() as { count: number };
  return Number(row.count);
}

async function envUpdateSessionStatus(
  env: TestBrokerEnv,
  state: string,
  status: 'canceled' | 'expired' | 'failed' | 'consumed',
): Promise<void> {
  env.__db
    .prepare(
      `UPDATE discord_oauth_sessions
          SET status = ?
        WHERE state_hash = ?`,
    )
    .run(status, await sha256Base64Url(state));
}

async function deriveExpectedDiscordUserRef(
  secret: string,
  discordUserId: string,
): Promise<string> {
  const encoder = new TextEncoder();
  const key = await crypto.subtle.importKey(
    'raw',
    encoder.encode(secret.trim()),
    {
      name: 'HMAC',
      hash: 'SHA-256',
    },
    false,
    ['sign'],
  );
  const signature = await crypto.subtle.sign(
    'HMAC',
    key,
    encoder.encode(`puripuly-heart:discord-user:v1\n${discordUserId.trim()}`),
  );
  return `ph-discord-user-v1_${encodeBase64Url(new Uint8Array(signature))}`;
}

function mockDiscordApi(options: {
  user?: Record<string, unknown>;
  tokenStatus?: number;
  userStatus?: number;
  openRouterMode?:
    | 'success'
    | 'create_failure'
    | 'create_network_failure'
    | 'create_retryable_failure'
    | 'guardrail_failure'
    | 'guardrail_failure_cleanup_failure'
    | 'cleanup_failure';
  rawChildKey?: string;
  childKeyHash?: string;
  accessToken?: string;
  refreshToken?: string;
  guardrailFailureMessage?: string;
  cleanupFailureMessage?: string;
} = {}): {
  fetchMock: ReturnType<typeof vi.fn>;
  openRouterCreateCalls: Array<{ input: string | URL; init?: RequestInit }>;
  openRouterGuardrailCalls: Array<{ input: string | URL; init?: RequestInit }>;
  openRouterCleanupCalls: Array<{ input: string | URL; init?: RequestInit }>;
} {
  const user = options.user ?? {
    id: discordSnowflakeForAgeDays(31),
    verified: true,
  };
  const openRouterCreateCalls: Array<{ input: string | URL; init?: RequestInit }> = [];
  const openRouterGuardrailCalls: Array<{ input: string | URL; init?: RequestInit }> = [];
  const openRouterCleanupCalls: Array<{ input: string | URL; init?: RequestInit }> = [];
  const childKeyHash = options.childKeyHash ?? 'hash_discord_managed_child_test_1';
  const fetchMock = vi.fn(async (input: string | URL, init?: RequestInit) => {
    const url = String(input);
    const method = init?.method ?? 'GET';

    if (url === DISCORD_TOKEN_URL && method === 'POST') {
      if (options.tokenStatus !== undefined && options.tokenStatus >= 400) {
        return jsonResponse({ error: 'token exchange failed' }, options.tokenStatus);
      }

      return jsonResponse({
        access_token: options.accessToken ?? 'discord-access-token',
        token_type: 'Bearer',
        ...(options.refreshToken ? { refresh_token: options.refreshToken } : {}),
      });
    }

    if (url === DISCORD_USER_URL && method === 'GET') {
      if (options.userStatus !== undefined && options.userStatus >= 400) {
        return jsonResponse({ error: 'user fetch failed' }, options.userStatus);
      }

      return jsonResponse(user);
    }

    if (url === OPENROUTER_KEYS_URL && method === 'POST') {
      openRouterCreateCalls.push({ input, init });
      if (options.openRouterMode === 'create_failure') {
        return jsonResponse({ error: { message: 'create failed before key delivery' } }, 400);
      }
      if (options.openRouterMode === 'create_retryable_failure') {
        return jsonResponse({ error: { message: 'create temporarily failed' } }, 503);
      }
      if (options.openRouterMode === 'create_network_failure') {
        throw new TypeError('OpenRouter create response was interrupted');
      }

      const sequence = openRouterCreateCalls.length;
      return jsonResponse(
        {
          key: options.rawChildKey ?? `or-discord-managed-child-key-test-${sequence}`,
          data: {
            hash: options.childKeyHash ?? `hash_discord_managed_child_test_${sequence}`,
          },
        },
        201,
      );
    }

    if (url === OPENROUTER_GUARDRAIL_URL && method === 'POST') {
      openRouterGuardrailCalls.push({ input, init });
      if (
        options.openRouterMode === 'guardrail_failure' ||
        options.openRouterMode === 'guardrail_failure_cleanup_failure'
      ) {
        return jsonResponse(
          {
            error: {
              message: options.guardrailFailureMessage ?? 'guardrail assignment failed',
            },
          },
          500,
        );
      }

      return jsonResponse({ assigned_count: 1 });
    }

    if (url === `${OPENROUTER_KEYS_URL}/${childKeyHash}` && method === 'PATCH') {
      openRouterCleanupCalls.push({ input, init });
      if (
        options.openRouterMode === 'guardrail_failure_cleanup_failure' ||
        options.openRouterMode === 'cleanup_failure'
      ) {
        return jsonResponse(
          {
            error: {
              message: options.cleanupFailureMessage ?? 'disable cleanup failed',
            },
          },
          500,
        );
      }

      return jsonResponse({ data: { hash: childKeyHash, disabled: true } });
    }

    if (url === `${OPENROUTER_KEYS_URL}/${childKeyHash}` && method === 'DELETE') {
      openRouterCleanupCalls.push({ input, init });
      if (
        options.openRouterMode === 'guardrail_failure_cleanup_failure' ||
        options.openRouterMode === 'cleanup_failure'
      ) {
        return jsonResponse(
          {
            error: {
              message: options.cleanupFailureMessage ?? 'delete cleanup failed',
            },
          },
          500,
        );
      }

      return new Response(null, { status: 204 });
    }

    if (url === 'https://discord.test/immediate-alert' && method === 'POST') {
      return new Response(null, { status: 204 });
    }

    throw new Error(`unexpected Discord API request: ${method} ${url}`);
  });

  vi.stubGlobal('fetch', fetchMock as typeof fetch);
  return {
    fetchMock,
    openRouterCreateCalls,
    openRouterGuardrailCalls,
    openRouterCleanupCalls,
  };
}

function expectTokenExchange(
  fetchMock: ReturnType<typeof vi.fn>,
  expected: {
    code: string;
    redirectUri: string;
    codeVerifier: string | null;
  },
): void {
  const [input, init] = fetchMock.mock.calls[0] as [string | URL, RequestInit];
  expect(String(input)).toBe(DISCORD_TOKEN_URL);
  expect(init.method).toBe('POST');
  expect(init.headers).toEqual({
    'content-type': 'application/x-www-form-urlencoded',
  });

  const params = new URLSearchParams(String(init.body));
  expect(params.get('grant_type')).toBe('authorization_code');
  expect(params.get('code')).toBe(expected.code);
  expect(params.get('redirect_uri')).toBe(expected.redirectUri);
  expect(params.get('client_id')).toBe('test-discord-client-id');
  expect(params.get('client_secret')).toBe('test-discord-client-secret');
  expect(params.get('code_verifier')).toBe(expected.codeVerifier);
}

function expectDiscordUserFetch(fetchMock: ReturnType<typeof vi.fn>): void {
  const [input, init] = fetchMock.mock.calls[1] as [string | URL, RequestInit];
  expect(String(input)).toBe(DISCORD_USER_URL);
  expect(init.method).toBe('GET');
  expect(init.headers).toEqual({
    authorization: 'Bearer discord-access-token',
  });
}

function expectCallOrder(
  fetchMock: ReturnType<typeof vi.fn>,
  expectedOrderedCalls: string[],
): void {
  const calls = fetchMock.mock.calls.map(([input, init]) => {
    const method = (init as RequestInit | undefined)?.method ?? 'GET';
    return `${method} ${String(input)}`;
  });
  const indexes = expectedOrderedCalls.map((expected) => calls.indexOf(expected));
  for (const index of indexes) {
    expect(index).toBeGreaterThanOrEqual(0);
  }
  expect(indexes).toEqual([...indexes].sort((left, right) => left - right));
}

function expectTextNotToContainSensitiveValues(text: string, values: string[]): void {
  for (const value of values) {
    expect(text).not.toContain(value);
  }
}

function stringifyConsoleCalls(spy: { mock: { calls: unknown[][] } }): string {
  return JSON.stringify(
    spy.mock.calls.map((call: unknown[]) =>
      call.map((value: unknown) => {
        if (value instanceof Error) {
          return {
            name: value.name,
            message: value.message,
          };
        }

        return value;
      }),
    ),
  );
}

function discordSnowflakeForAgeDays(days: number): string {
  return discordSnowflakeForAgeMs(days * 24 * 60 * 60 * 1000);
}

function discordSnowflakeForAgeMs(ageMs: number): string {
  return discordSnowflakeForDate(new Date(Date.now() - ageMs));
}

function discordSnowflakeForDate(createdAt: Date): string {
  const timestamp = BigInt(createdAt.getTime()) - DISCORD_EPOCH_MS;
  return (timestamp << 22n).toString();
}

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: {
      'content-type': 'application/json',
    },
  });
}
