import { afterEach, describe, expect, it, vi } from 'vitest';

import { createDeviceKeyPair, signCanonicalStatusRequest } from './test-support/ed25519';
import { createTestBrokerEnv, insertEntitlement } from './test-support/sqlite-d1';
import { getTrialStatus, issueChallenge } from './test-support/trial-api';
import { expectNoReferralRewardEstimateFields } from './test-support/referral-response-privacy';

const REFERRAL_ID_PATTERN = /^[23456789ABCDEFGHJKMNPQRSTUVWXYZ]{6}$/u;
const ACTIVE_DISCORD_USER_REF =
  'ph-discord-user-v1_abcdefghijklmnopqrstuvwxyz234567890ABCDEFG';

function collectResponseKeys(value: unknown): string[] {
  if (!value || typeof value !== 'object') {
    return [];
  }

  if (Array.isArray(value)) {
    return value.flatMap((entry) => collectResponseKeys(entry));
  }

  return Object.entries(value).flatMap(([key, nested]) => [
    key,
    ...collectResponseKeys(nested),
  ]);
}

function insertDiscordIdentity(input: {
  env: ReturnType<typeof createTestBrokerEnv>;
  discordUserRef: string;
  installationId: string;
  status: 'issuing' | 'delivery_pending' | 'active' | 'failed' | 'cleanup_required';
}): void {
  input.env.__db
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
      '2026-04-08T06:00:00.000Z',
      '2026-04-08T06:00:00.000Z',
    );
}

function readReferralCode(input: {
  env: ReturnType<typeof createTestBrokerEnv>;
  discordUserRef: string;
}): { referral_id: string; owner_installation_id: string | null; status: string } | null {
  const row = input.env.__db
    .prepare(
      `SELECT referral_id,
              owner_installation_id,
              status
         FROM referral_codes
        WHERE owner_discord_user_ref = ?`,
    )
    .get(input.discordUserRef) as
    | { referral_id: string; owner_installation_id: string | null; status: string }
    | undefined;

  return row ?? null;
}

function insertStatusReferralReward(input: {
  env: ReturnType<typeof createTestBrokerEnv>;
  referralId: string;
  referrerDiscordUserRef: string;
  referrerInstallationId: string;
  index: number;
  referredBonusStatus: 'reserved' | 'credited' | 'skipped' | 'failed';
}): void {
  const referrerBonusStatus =
    input.referredBonusStatus === 'credited'
      ? 'credited'
      : input.referredBonusStatus === 'reserved'
        ? 'pending'
        : input.referredBonusStatus;
  input.env.__db
    .prepare(
      `INSERT INTO referral_rewards (
          referral_id,
          referrer_discord_user_ref,
          referrer_installation_id,
          referred_discord_user_ref,
          referred_installation_id,
          referred_hardware_hash,
          referred_hardware_hash_salt_version,
          referred_bonus_status,
          referrer_bonus_status,
          skip_reason,
          failure_reason,
          referred_managed_credential_ref,
          referrer_managed_credential_ref,
          created_at,
          updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, 7, ?, ?, ?, ?, NULL, NULL, ?, ?)`,
    )
    .run(
      input.referralId,
      input.referrerDiscordUserRef,
      input.referrerInstallationId,
      `ph-discord-user-v1_status-referred-${input.index}`,
      `install-status-referred-${input.index}`,
      `hardware-hash-status-referred-${input.index}`,
      input.referredBonusStatus,
      referrerBonusStatus,
      input.referredBonusStatus === 'skipped' ? 'unknown_referral_id' : null,
      input.referredBonusStatus === 'failed' ? 'issue_delivery_failed' : null,
      '2026-04-08T06:00:00.000Z',
      '2026-04-08T06:00:00.000Z',
    );
}

describe('GET /v1/trial/status response contract', () => {
  afterEach(() => {
    vi.restoreAllMocks();
    vi.useRealTimers();
  });

  it('privacy assertion rejects nested referral reward estimates while allowing normal budget_usd fields', () => {
    expect(() =>
      expectNoReferralRewardEstimateFields({
        current_entitlement: {
          budget_usd: 0.07,
        },
        referral: {
          bonusUsd: '0.02',
          estimate: {
            utteranceCount: 120,
          },
        },
      }),
    ).toThrow(/referral/iu);
  });

  it('returns normalized active managed state with broker-side eligibility metadata', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-04-08T06:00:00Z'));

    const env = createTestBrokerEnv();
    const keyPair = await createDeviceKeyPair();
    await issueChallenge({
      env,
      installationId: 'install-status-active',
      devicePublicKey: keyPair.devicePublicKey,
      appVersion: '1.2.3',
    });
    insertEntitlement(env, {
      installation_id: 'install-status-active',
      status: 'active',
      budget_usd: 0.04,
      managed_credential_ref: 'internal-active-ref',
      issued_at: '2026-04-01T00:00:00Z',
      expires_at: '2026-10-01T00:00:00Z',
    });
    const signedRequest = await signCanonicalStatusRequest(keyPair.privateKey, {
      installation_id: 'install-status-active',
      timestamp: '2026-04-08T06:00:30.000Z',
    });

    const response = await getTrialStatus({
      env,
      installationId: 'install-status-active',
      headers: {
        'X-Puripuly-Timestamp': signedRequest.timestamp,
        'X-Puripuly-Signature': signedRequest.signature,
      },
    });

    expect(response.status).toBe(200);
    await expect(response.json()).resolves.toEqual({
      managed_state: {
        lifecycle: 'active',
        managed_availability: true,
      },
      current_entitlement: {
        provider: 'OpenRouter',
        budget_usd: 0.04,
        issued_at: '2026-04-01T00:00:00Z',
        expires_at: '2026-10-01T00:00:00Z',
      },
      onboarding_eligibility: {
        eligible: false,
        reason: 'active',
        requires_discord_oauth: false,
      },
    });
  });

  it('lazily creates and returns an owned Referral ID for active Discord-managed users', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-04-08T06:00:00Z'));

    const env = createTestBrokerEnv();
    const keyPair = await createDeviceKeyPair();
    await issueChallenge({
      env,
      installationId: 'install-status-active-discord-referral',
      devicePublicKey: keyPair.devicePublicKey,
      appVersion: '1.2.3',
    });
    insertDiscordIdentity({
      env,
      discordUserRef: ACTIVE_DISCORD_USER_REF,
      installationId: 'install-status-active-discord-referral',
      status: 'active',
    });
    insertEntitlement(env, {
      installation_id: 'install-status-active-discord-referral',
      status: 'active',
      budget_usd: 0.04,
      managed_credential_ref: 'internal-active-discord-ref',
      issued_at: '2026-04-01T00:00:00Z',
      expires_at: '2026-10-01T00:00:00Z',
      discord_user_ref: ACTIVE_DISCORD_USER_REF,
      discord_issue_status: 'active',
      discord_issue_reserved_at: '2026-04-01T00:00:00Z',
      discord_issue_delivered_at: '2026-04-01T00:00:01Z',
    });
    const signedRequest = await signCanonicalStatusRequest(keyPair.privateKey, {
      installation_id: 'install-status-active-discord-referral',
      timestamp: '2026-04-08T06:00:30.000Z',
    });

    const response = await getTrialStatus({
      env,
      installationId: 'install-status-active-discord-referral',
      headers: {
        'X-Puripuly-Timestamp': signedRequest.timestamp,
        'X-Puripuly-Signature': signedRequest.signature,
      },
    });

    expect(response.status).toBe(200);
    const payload = (await response.json()) as Record<string, unknown>;
    expect(payload.referral_id).toEqual(expect.stringMatching(REFERRAL_ID_PATTERN));
    expect(payload).not.toHaveProperty('referral_bonus_applied');
    expectNoReferralRewardEstimateFields(payload);
    expect(readReferralCode({ env, discordUserRef: ACTIVE_DISCORD_USER_REF })).toEqual(
      expect.objectContaining({
        referral_id: payload.referral_id,
        owner_installation_id: 'install-status-active-discord-referral',
        status: 'active',
      }),
    );
  });

  it('includes Talk Together Pass status for active Discord-managed users', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-04-08T06:00:00Z'));

    const installationId = 'install-status-active-discord-pass';
    const env = createTestBrokerEnv();
    const keyPair = await createDeviceKeyPair();
    await issueChallenge({
      env,
      installationId,
      devicePublicKey: keyPair.devicePublicKey,
      appVersion: '1.2.3',
    });
    insertDiscordIdentity({
      env,
      discordUserRef: ACTIVE_DISCORD_USER_REF,
      installationId,
      status: 'active',
    });
    insertEntitlement(env, {
      installation_id: installationId,
      status: 'active',
      budget_usd: 0.04,
      managed_credential_ref: 'internal-active-discord-pass-ref',
      issued_at: '2026-04-01T00:00:00Z',
      expires_at: '2026-10-01T00:00:00Z',
      discord_user_ref: ACTIVE_DISCORD_USER_REF,
      discord_issue_status: 'active',
      discord_issue_reserved_at: '2026-04-01T00:00:00Z',
      discord_issue_delivered_at: '2026-04-01T00:00:01Z',
    });
    const signedRequest = await signCanonicalStatusRequest(keyPair.privateKey, {
      installation_id: installationId,
      timestamp: '2026-04-08T06:00:30.000Z',
    });

    const response = await getTrialStatus({
      env,
      installationId,
      headers: {
        'X-Puripuly-Timestamp': signedRequest.timestamp,
        'X-Puripuly-Signature': signedRequest.signature,
      },
    });

    expect(response.status).toBe(200);
    const payload = (await response.json()) as Record<string, unknown>;
    expect(payload.referral_id).toMatch(REFERRAL_ID_PATTERN);
    expect(payload.talk_together_pass).toEqual({
      pass_id: payload.referral_id,
      invite_count: 0,
      invite_limit: 5,
      bonus_translations_per_friend: 200,
    });
    const serialized = JSON.stringify(payload.talk_together_pass);
    expect(serialized).not.toContain('discord_user_ref');
    expect(serialized).not.toContain('hardware');
    expect(serialized).not.toContain('ledger');
    expect(serialized).not.toContain('budget_usd');
  });

  it('counts only reserved and credited Talk Together Pass referral rewards', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-04-08T06:00:00Z'));

    const installationId = 'install-status-pass-counted-rewards';
    const env = createTestBrokerEnv();
    const keyPair = await createDeviceKeyPair();
    await issueChallenge({
      env,
      installationId,
      devicePublicKey: keyPair.devicePublicKey,
      appVersion: '1.2.3',
    });
    insertDiscordIdentity({
      env,
      discordUserRef: ACTIVE_DISCORD_USER_REF,
      installationId,
      status: 'active',
    });
    insertEntitlement(env, {
      installation_id: installationId,
      status: 'active',
      budget_usd: 0.04,
      managed_credential_ref: 'internal-status-pass-counted-ref',
      issued_at: '2026-04-01T00:00:00Z',
      expires_at: '2026-10-01T00:00:00Z',
      discord_user_ref: ACTIVE_DISCORD_USER_REF,
      discord_issue_status: 'active',
      discord_issue_reserved_at: '2026-04-01T00:00:00Z',
      discord_issue_delivered_at: '2026-04-01T00:00:01Z',
    });
    const signedRequest = await signCanonicalStatusRequest(keyPair.privateKey, {
      installation_id: installationId,
      timestamp: '2026-04-08T06:00:30.000Z',
    });

    const initialResponse = await getTrialStatus({
      env,
      installationId,
      headers: {
        'X-Puripuly-Timestamp': signedRequest.timestamp,
        'X-Puripuly-Signature': signedRequest.signature,
      },
    });
    expect(initialResponse.status).toBe(200);
    const initialPayload = (await initialResponse.json()) as Record<string, unknown>;
    expect(initialPayload.referral_id).toMatch(REFERRAL_ID_PATTERN);
    const referralId = String(initialPayload.referral_id);

    (['reserved', 'credited', 'failed', 'skipped'] as const).forEach(
      (referredBonusStatus, index) => {
        insertStatusReferralReward({
          env,
          referralId,
          referrerDiscordUserRef: ACTIVE_DISCORD_USER_REF,
          referrerInstallationId: installationId,
          index,
          referredBonusStatus,
        });
      },
    );

    const response = await getTrialStatus({
      env,
      installationId,
      headers: {
        'X-Puripuly-Timestamp': signedRequest.timestamp,
        'X-Puripuly-Signature': signedRequest.signature,
      },
    });

    expect(response.status).toBe(200);
    const payload = (await response.json()) as Record<string, unknown>;
    expect(payload.referral_id).toBe(referralId);
    expect(payload.talk_together_pass).toMatchObject({
      pass_id: payload.referral_id,
      invite_count: 2,
      invite_limit: 5,
      bonus_translations_per_friend: 200,
    });
  });

  it('keeps Referral ID and omits Talk Together Pass status when invite count query fails', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-04-08T06:00:00Z'));
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => undefined);

    const installationId = 'install-status-pass-count-failure';
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
    const keyPair = await createDeviceKeyPair();
    await issueChallenge({
      env,
      installationId,
      devicePublicKey: keyPair.devicePublicKey,
      appVersion: '1.2.3',
    });
    insertDiscordIdentity({
      env,
      discordUserRef: ACTIVE_DISCORD_USER_REF,
      installationId,
      status: 'active',
    });
    insertEntitlement(env, {
      installation_id: installationId,
      status: 'active',
      budget_usd: 0.04,
      managed_credential_ref: 'internal-status-pass-count-failure-ref',
      issued_at: '2026-04-01T00:00:00Z',
      expires_at: '2026-10-01T00:00:00Z',
      discord_user_ref: ACTIVE_DISCORD_USER_REF,
      discord_issue_status: 'active',
      discord_issue_reserved_at: '2026-04-01T00:00:00Z',
      discord_issue_delivered_at: '2026-04-01T00:00:01Z',
    });
    const signedRequest = await signCanonicalStatusRequest(keyPair.privateKey, {
      installation_id: installationId,
      timestamp: '2026-04-08T06:00:30.000Z',
    });

    const response = await getTrialStatus({
      env,
      installationId,
      headers: {
        'X-Puripuly-Timestamp': signedRequest.timestamp,
        'X-Puripuly-Signature': signedRequest.signature,
      },
    });

    expect(response.status).toBe(200);
    const payload = (await response.json()) as Record<string, unknown>;
    expect(payload.referral_id).toMatch(REFERRAL_ID_PATTERN);
    expect(payload).not.toHaveProperty('talk_together_pass');
    expect(warnSpy).toHaveBeenCalledTimes(1);
    expect(warnSpy).toHaveBeenCalledWith('owned_referral_status_failed', {
      endpoint: 'trial_status',
      installation_id: expect.any(String),
      reason: 'talk_together_pass_status_failed',
    });
    const warnCalls = JSON.stringify(warnSpy.mock.calls);
    expect(warnCalls).not.toContain('forced Talk Together Pass count failure');
    expect(warnCalls).not.toContain('internal-status-pass-count-failure-ref');
    expect(warnCalls).not.toContain(ACTIVE_DISCORD_USER_REF);
  });

  it.each([0, 1, 4, 5, 6])(
    'returns capped Talk Together Pass invite count for %i counted rows',
    async (countedRows) => {
      vi.useFakeTimers();
      vi.setSystemTime(new Date('2026-04-08T06:00:00Z'));

      const installationId = `install-status-pass-boundary-${countedRows}`;
      const env = createTestBrokerEnv();
      const keyPair = await createDeviceKeyPair();
      await issueChallenge({
        env,
        installationId,
        devicePublicKey: keyPair.devicePublicKey,
        appVersion: '1.2.3',
      });
      insertDiscordIdentity({
        env,
        discordUserRef: ACTIVE_DISCORD_USER_REF,
        installationId,
        status: 'active',
      });
      insertEntitlement(env, {
        installation_id: installationId,
        status: 'active',
        budget_usd: 0.04,
        managed_credential_ref: `internal-status-pass-boundary-${countedRows}`,
        issued_at: '2026-04-01T00:00:00Z',
        expires_at: '2026-10-01T00:00:00Z',
        discord_user_ref: ACTIVE_DISCORD_USER_REF,
        discord_issue_status: 'active',
        discord_issue_reserved_at: '2026-04-01T00:00:00Z',
        discord_issue_delivered_at: '2026-04-01T00:00:01Z',
      });
      const signedRequest = await signCanonicalStatusRequest(keyPair.privateKey, {
        installation_id: installationId,
        timestamp: '2026-04-08T06:00:30.000Z',
      });

      const initialResponse = await getTrialStatus({
        env,
        installationId,
        headers: {
          'X-Puripuly-Timestamp': signedRequest.timestamp,
          'X-Puripuly-Signature': signedRequest.signature,
        },
      });
      expect(initialResponse.status).toBe(200);
      const initialPayload = (await initialResponse.json()) as Record<string, unknown>;
      expect(initialPayload.referral_id).toMatch(REFERRAL_ID_PATTERN);
      const referralId = String(initialPayload.referral_id);

      for (let index = 0; index < countedRows; index += 1) {
        insertStatusReferralReward({
          env,
          referralId,
          referrerDiscordUserRef: ACTIVE_DISCORD_USER_REF,
          referrerInstallationId: installationId,
          index,
          referredBonusStatus: index % 2 === 0 ? 'reserved' : 'credited',
        });
      }

      const response = await getTrialStatus({
        env,
        installationId,
        headers: {
          'X-Puripuly-Timestamp': signedRequest.timestamp,
          'X-Puripuly-Signature': signedRequest.signature,
        },
      });
      expect(response.status).toBe(200);
      const payload = (await response.json()) as Record<string, unknown>;
      expect(payload.talk_together_pass).toMatchObject({
        pass_id: referralId,
        invite_count: Math.min(countedRows, 5),
        invite_limit: 5,
        bonus_translations_per_friend: 200,
      });
    },
  );

  it('omits Referral ID and does not create one for non-delivered Discord-managed status', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-04-08T06:00:00Z'));
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => undefined);

    const env = createTestBrokerEnv();
    const keyPair = await createDeviceKeyPair();
    await issueChallenge({
      env,
      installationId: 'install-status-issuing-discord-referral',
      devicePublicKey: keyPair.devicePublicKey,
      appVersion: '1.2.3',
    });
    insertDiscordIdentity({
      env,
      discordUserRef: ACTIVE_DISCORD_USER_REF,
      installationId: 'install-status-issuing-discord-referral',
      status: 'issuing',
    });
    insertEntitlement(env, {
      installation_id: 'install-status-issuing-discord-referral',
      status: 'pending_release',
      budget_usd: 0.04,
      release_session_ref: 'pending-release-ref',
      release_token_hash: 'pending-release-token-hash',
      release_token_expires_at: '2026-04-08T06:15:00Z',
      discord_user_ref: ACTIVE_DISCORD_USER_REF,
      discord_issue_status: 'issuing',
      discord_issue_reserved_at: '2026-04-01T00:00:00Z',
    });
    const signedRequest = await signCanonicalStatusRequest(keyPair.privateKey, {
      installation_id: 'install-status-issuing-discord-referral',
      timestamp: '2026-04-08T06:00:30.000Z',
    });

    const response = await getTrialStatus({
      env,
      installationId: 'install-status-issuing-discord-referral',
      headers: {
        'X-Puripuly-Timestamp': signedRequest.timestamp,
        'X-Puripuly-Signature': signedRequest.signature,
      },
    });

    expect(response.status).toBe(200);
    const payload = (await response.json()) as Record<string, unknown>;
    expect(payload).not.toHaveProperty('referral_id');
    expect(readReferralCode({ env, discordUserRef: ACTIVE_DISCORD_USER_REF })).toBeNull();
    expect(warnSpy).not.toHaveBeenCalled();
  });

  it('returns Discord-required eligibility without silent browser launch fields when no entitlement exists', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-04-08T06:00:00Z'));

    const env = createTestBrokerEnv();
    const keyPair = await createDeviceKeyPair();
    await issueChallenge({
      env,
      installationId: 'install-status-discord-required',
      devicePublicKey: keyPair.devicePublicKey,
      appVersion: '1.2.3',
    });
    const signedRequest = await signCanonicalStatusRequest(keyPair.privateKey, {
      installation_id: 'install-status-discord-required',
      timestamp: '2026-04-08T06:00:30.000Z',
    });

    const response = await getTrialStatus({
      env,
      installationId: 'install-status-discord-required',
      headers: {
        'X-Puripuly-Timestamp': signedRequest.timestamp,
        'X-Puripuly-Signature': signedRequest.signature,
      },
    });

    expect(response.status).toBe(200);
    const payload = (await response.json()) as Record<string, unknown>;
    expect(payload).toEqual({
      managed_state: {
        lifecycle: 'none',
        managed_availability: true,
      },
      current_entitlement: null,
      onboarding_eligibility: {
        eligible: true,
        reason: 'discord_required',
        requires_discord_oauth: true,
      },
    });
    expect(payload).not.toHaveProperty('authorization_url');
    expect(payload).not.toHaveProperty('redirect_uri');
  });

  it('keeps status responses free of challenge, release-session, and credential storage fields', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-04-08T06:00:00Z'));

    const env = createTestBrokerEnv();
    const keyPair = await createDeviceKeyPair();
    await issueChallenge({
      env,
      installationId: 'install-status-hidden-fields',
      devicePublicKey: keyPair.devicePublicKey,
      appVersion: '1.2.3',
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
          ) VALUES (?, ?, ?, ?, ?, ?)`,
      )
      .run(
        'raw-discord-user-123456789012345678:user@example.test',
        'install-status-hidden-fields',
        'issuing',
        1,
        '2026-04-08T06:00:00Z',
        '2026-04-08T06:00:00Z',
      );
    insertEntitlement(env, {
      installation_id: 'install-status-hidden-fields',
      status: 'pending_release',
      budget_usd: 0.07,
      managed_credential_ref: 'internal-pending-ref',
      release_session_ref: 'release-session',
      release_token_hash: 'release-token-hash',
      release_token_expires_at: '2026-04-08T06:15:00Z',
      discord_user_ref: 'raw-discord-user-123456789012345678:user@example.test',
      discord_issue_status: 'issuing',
      discord_issue_reserved_at: '2026-04-08T06:00:00Z',
      discord_issue_delivered_at: '2026-04-08T06:01:00Z',
    });
    const signedRequest = await signCanonicalStatusRequest(keyPair.privateKey, {
      installation_id: 'install-status-hidden-fields',
      timestamp: '2026-04-08T06:00:30.000Z',
    });

    const response = await getTrialStatus({
      env,
      installationId: 'install-status-hidden-fields',
      headers: {
        'X-Puripuly-Timestamp': signedRequest.timestamp,
        'X-Puripuly-Signature': signedRequest.signature,
      },
    });
    expect(response.status).toBe(200);

    const payload = (await response.json()) as Record<string, unknown>;
    expect(payload).not.toHaveProperty('challenge');
    expect(payload).not.toHaveProperty('challenge_expires_at');
    expect(payload).not.toHaveProperty('fingerprint_salt');
    expect(payload).not.toHaveProperty('release_token');
    expect(payload).not.toHaveProperty('release_session_ref');
    expect(payload).not.toHaveProperty('release_token_hash');
    expect(payload).not.toHaveProperty('managed_credential_ref');
    expect(payload).not.toHaveProperty('authorization_url');

    const responseKeys = collectResponseKeys(payload);
    const forbiddenResponseKeys = [
      'discord_user_ref',
      'discord_user_id',
      'discord_id',
      'discord_email',
      'discord_email_verified',
      'discord_account_created_at',
      'discord_issue_status',
      'discord_issue_reserved_at',
      'discord_issue_delivered_at',
      'state_hash',
      'redirect_uri',
      'pkce_code_verifier',
      'issue_nonce_hash',
      'authorization_url',
    ];
    const leakedResponseKeys = forbiddenResponseKeys.filter((key) =>
      responseKeys.includes(key),
    );
    expect(leakedResponseKeys).toEqual([]);
    const serializedPayload = JSON.stringify(payload);
    expect(serializedPayload).not.toContain('raw-discord-user-123456789012345678');
    expect(serializedPayload).not.toContain('user@example.test');
  });

  it('treats pending_release as broker-side onboarding continuation rather than a terminal state', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-04-08T06:00:00Z'));

    const env = createTestBrokerEnv();
    const keyPair = await createDeviceKeyPair();
    await issueChallenge({
      env,
      installationId: 'install-status-pending-release',
      devicePublicKey: keyPair.devicePublicKey,
      appVersion: '1.2.3',
    });
    insertEntitlement(env, {
      installation_id: 'install-status-pending-release',
      status: 'pending_release',
      budget_usd: 0.07,
    });
    const signedRequest = await signCanonicalStatusRequest(keyPair.privateKey, {
      installation_id: 'install-status-pending-release',
      timestamp: '2026-04-08T06:00:30.000Z',
    });

    const response = await getTrialStatus({
      env,
      installationId: 'install-status-pending-release',
      headers: {
        'X-Puripuly-Timestamp': signedRequest.timestamp,
        'X-Puripuly-Signature': signedRequest.signature,
      },
    });

    expect(response.status).toBe(200);
    await expect(response.json()).resolves.toMatchObject({
      managed_state: {
        lifecycle: 'pending_release',
        managed_availability: true,
      },
      onboarding_eligibility: {
        eligible: false,
        reason: 'pending_release',
        requires_discord_oauth: false,
      },
    });
  });
});
