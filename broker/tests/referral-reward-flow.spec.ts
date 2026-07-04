import { afterEach, describe, expect, it, vi } from 'vitest';

import {
  applyReferrerRewardLimitUpdates,
  reserveIssueReferralReward,
  type IssueReferralReservationResult,
} from '../src/referral';
import {
  createDeviceKeyPair,
  encodeBase64Url,
  signCanonicalDiscordIssueRequest,
  type DeviceKeyPair,
  type SignedDiscordIssueRequestInput,
} from './test-support/ed25519';
import { sha256Base64Url } from './test-support/hash';
import {
  createTestBrokerEnv,
  insertEntitlement,
  type TestBrokerEnv,
} from './test-support/sqlite-d1';
import { postDiscordIssue, postDiscordStart } from './test-support/trial-api';
import { expectNoReferralRewardEstimateFields } from './test-support/referral-response-privacy';
import { updateAbuseControls } from './test-support/abuse-controls';

const REGISTERED_REDIRECT_URI = 'http://127.0.0.1:62187/discord/callback';
const APP_VERSION = '1.2.3';
const MODEL = 'google/gemma-4-26b-a4b-it';
const NOW_ISO = '2026-04-30T06:00:00.000Z';
const SIGNED_AT_ISO = '2026-04-30T06:00:30.000Z';
const EXPIRES_AT_ISO = '2026-07-30T06:00:00.000Z';
const DISCORD_TOKEN_URL = 'https://discord.com/api/oauth2/token';
const DISCORD_USER_URL = 'https://discord.com/api/users/@me';
const OPENROUTER_KEYS_URL = 'https://openrouter.ai/api/v1/keys';
const OPENROUTER_GUARDRAIL_URL =
  'https://openrouter.ai/api/v1/guardrails/test-managed-guardrail-id/assignments/keys';
const DISCORD_EPOCH_MS = 1420070400000n;

const REFERRAL_ID = '7KQ9M2';
const UNKNOWN_REFERRAL_ID = 'ABCDEF';
const REFERRER_DISCORD_REF = `ph-discord-user-v1_${'A'.repeat(43)}`;
const REFERRER_INSTALLATION_ID = 'install-referrer-reward-flow';
const REFERRER_HARDWARE_HASH = 'hardware-hash-referrer-reward-flow';
const REFERRER_MANAGED_CREDENTIAL_REF = 'managed-credential-referrer-reward-flow';
const REFERRED_DISCORD_REF = `ph-discord-user-v1_${'B'.repeat(43)}`;
const REFERRED_INSTALLATION_ID = 'install-referred-reward-flow';
const REFERRED_HARDWARE_HASH = 'hardware-hash-referred-reward-flow';

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

interface ReferralRewardRow {
  referral_id: string;
  referrer_discord_user_ref: string | null;
  referrer_installation_id: string | null;
  referred_discord_user_ref: string;
  referred_installation_id: string;
  referred_hardware_hash: string;
  referred_hardware_hash_salt_version: number;
  referred_bonus_status: string;
  referrer_bonus_status: string;
  skip_reason: string | null;
  failure_reason: string | null;
  referred_managed_credential_ref: string | null;
  referrer_managed_credential_ref: string | null;
  attempt_ip_hash: string | null;
  credited_at: string | null;
}

interface EntitlementBudgetRow {
  status: string;
  budget_usd: number;
  managed_credential_ref: string | null;
  discord_issue_status: string | null;
}

describe('issue-time referral reservation primitive', () => {
  it('reserves an eligible issue-time referral as exactly one counted ledger row', async () => {
    const env = createTestBrokerEnv();
    insertActiveReferrer(env);

    const result = await reserveIssueReferralReward(env.BROKER_DB, {
      referralId: REFERRAL_ID,
      referredDiscordUserRef: REFERRED_DISCORD_REF,
      referredInstallationId: REFERRED_INSTALLATION_ID,
      referredHardwareHash: REFERRED_HARDWARE_HASH,
      referredHardwareHashSaltVersion: 7,
      nowIso: NOW_ISO,
    });

    expect(result).toEqual(expect.objectContaining({ outcome: 'reserved' }));
    expect(readReferralRewards(env)).toEqual([
      expect.objectContaining({
        referral_id: REFERRAL_ID,
        referrer_discord_user_ref: REFERRER_DISCORD_REF,
        referrer_installation_id: REFERRER_INSTALLATION_ID,
        referred_discord_user_ref: REFERRED_DISCORD_REF,
        referred_installation_id: REFERRED_INSTALLATION_ID,
        referred_hardware_hash: REFERRED_HARDWARE_HASH,
        referred_hardware_hash_salt_version: 7,
        referred_bonus_status: 'reserved',
        referrer_bonus_status: 'pending',
        skip_reason: null,
        failure_reason: null,
      }),
    ]);
  });

  it('records valid-shaped unknown referral input as a skipped outcome without a referrer', async () => {
    const env = createTestBrokerEnv();

    const result = await reserveIssueReferralReward(env.BROKER_DB, {
      referralId: UNKNOWN_REFERRAL_ID,
      referredDiscordUserRef: REFERRED_DISCORD_REF,
      referredInstallationId: REFERRED_INSTALLATION_ID,
      referredHardwareHash: REFERRED_HARDWARE_HASH,
      referredHardwareHashSaltVersion: 7,
      nowIso: NOW_ISO,
    });

    expect(result).toEqual({ outcome: 'skipped', reason: 'unknown_referral_id' });
    expect(readReferralRewards(env)).toEqual([
      expect.objectContaining({
        referral_id: UNKNOWN_REFERRAL_ID,
        referrer_discord_user_ref: null,
        referrer_installation_id: null,
        referred_bonus_status: 'skipped',
        referrer_bonus_status: 'skipped',
        skip_reason: 'unknown_referral_id',
      }),
    ]);
  });

  it.each([
    {
      name: 'disabled referral ID',
      arrange: (env: TestBrokerEnv) => {
        insertReferralCode(env, {
          referralId: REFERRAL_ID,
          ownerDiscordUserRef: REFERRER_DISCORD_REF,
          ownerInstallationId: REFERRER_INSTALLATION_ID,
          status: 'disabled',
        });
      },
      expectedReason: 'disabled_referral_id',
    },
    {
      name: 'self-referral by Discord identity',
      arrange: (env: TestBrokerEnv) => {
        insertReferralCode(env, {
          referralId: REFERRAL_ID,
          ownerDiscordUserRef: REFERRED_DISCORD_REF,
          ownerInstallationId: REFERRER_INSTALLATION_ID,
        });
      },
      expectedReason: 'self_referral',
    },
    {
      name: 'duplicate referrer hardware',
      arrange: (env: TestBrokerEnv) => {
        insertActiveReferrer(env, {
          hardwareHash: REFERRED_HARDWARE_HASH,
          hardwareHashSaltVersion: 7,
        });
      },
      expectedReason: 'duplicate_hardware',
    },
    {
      name: 'prior counted referred Discord identity',
      arrange: (env: TestBrokerEnv) => {
        insertActiveReferrer(env);
        insertReferralReward(env, {
          referralId: 'BCDEFG',
          referrerDiscordUserRef: `ph-discord-user-v1_${'C'.repeat(43)}`,
          referrerInstallationId: 'install-other-referrer',
          referredDiscordUserRef: REFERRED_DISCORD_REF,
          referredInstallationId: 'install-other-referred',
          referredHardwareHash: 'hardware-hash-other-referred',
          referredBonusStatus: 'credited',
          referrerBonusStatus: 'credited',
        });
      },
      expectedReason: 'referred_already_rewarded',
    },
    {
      name: 'prior counted referred installation',
      arrange: (env: TestBrokerEnv) => {
        insertActiveReferrer(env);
        insertReferralReward(env, {
          referralId: 'CDEFGH',
          referrerDiscordUserRef: `ph-discord-user-v1_${'D'.repeat(43)}`,
          referrerInstallationId: 'install-installation-dupe-referrer',
          referredDiscordUserRef: `ph-discord-user-v1_${'E'.repeat(43)}`,
          referredInstallationId: REFERRED_INSTALLATION_ID,
          referredHardwareHash: 'hardware-hash-installation-dupe',
          referredBonusStatus: 'reserved',
          referrerBonusStatus: 'pending',
        });
      },
      expectedReason: 'referred_already_rewarded',
    },
  ])('records $name as a skipped referral outcome', async ({ arrange, expectedReason }) => {
    const env = createTestBrokerEnv();
    arrange(env);

    const result = await reserveIssueReferralReward(env.BROKER_DB, {
      referralId: REFERRAL_ID,
      referredDiscordUserRef: REFERRED_DISCORD_REF,
      referredInstallationId: REFERRED_INSTALLATION_ID,
      referredHardwareHash: REFERRED_HARDWARE_HASH,
      referredHardwareHashSaltVersion: 7,
      nowIso: NOW_ISO,
    });

    expect(result).toEqual({ outcome: 'skipped', reason: expectedReason });
    expect(readReferralRewards(env).at(-1)).toEqual(
      expect.objectContaining({
        referral_id: REFERRAL_ID,
        referred_bonus_status: 'skipped',
        referrer_bonus_status: 'skipped',
        skip_reason: expectedReason,
      }),
    );
  });

  it('skips the sixth counted referral using ledger-derived cap accounting', async () => {
    const env = createTestBrokerEnv();
    insertActiveReferrer(env);
    seedCountedReferralRewards(env, 5);

    const result = await reserveIssueReferralReward(env.BROKER_DB, {
      referralId: REFERRAL_ID,
      referredDiscordUserRef: REFERRED_DISCORD_REF,
      referredInstallationId: REFERRED_INSTALLATION_ID,
      referredHardwareHash: REFERRED_HARDWARE_HASH,
      referredHardwareHashSaltVersion: 7,
      nowIso: NOW_ISO,
    });

    expect(result).toEqual({ outcome: 'skipped', reason: 'referrer_cap_reached' });
    expect(countCountedRewards(env, REFERRER_DISCORD_REF)).toBe(5);
    expect(readReferralRewards(env).at(-1)).toEqual(
      expect.objectContaining({
        referred_bonus_status: 'skipped',
        referrer_bonus_status: 'skipped',
        skip_reason: 'referrer_cap_reached',
      }),
    );
  });

  it('admits only one reservation when two referred users race for the fifth referrer slot', async () => {
    let gateEnabled = false;
    let waitingInserts = 0;
    let releaseInserts: (() => void) | null = null;
    const bothReservationInsertsReached = new Promise<void>((resolve) => {
      releaseInserts = resolve;
    });
    const env = createTestBrokerEnv({
      beforeRun: async ({ sql }) => {
        if (
          gateEnabled &&
          sql.includes('INSERT') &&
          sql.includes('referral_rewards') &&
          sql.includes('reserved')
        ) {
          waitingInserts += 1;
          if (waitingInserts === 2) {
            releaseInserts?.();
          }
          await bothReservationInsertsReached;
        }
      },
    });
    insertActiveReferrer(env);
    seedCountedReferralRewards(env, 4);
    gateEnabled = true;

    const results = await Promise.all([
      reserveIssueReferralReward(env.BROKER_DB, {
        referralId: REFERRAL_ID,
        referredDiscordUserRef: `ph-discord-user-v1_${'X'.repeat(43)}`,
        referredInstallationId: 'install-race-referred-a',
        referredHardwareHash: 'hardware-hash-race-a',
        referredHardwareHashSaltVersion: 7,
        nowIso: NOW_ISO,
      }),
      reserveIssueReferralReward(env.BROKER_DB, {
        referralId: REFERRAL_ID,
        referredDiscordUserRef: `ph-discord-user-v1_${'Y'.repeat(43)}`,
        referredInstallationId: 'install-race-referred-b',
        referredHardwareHash: 'hardware-hash-race-b',
        referredHardwareHashSaltVersion: 7,
        nowIso: NOW_ISO,
      }),
    ]);

    expect(results.map((result) => result.outcome).sort()).toEqual([
      'reserved',
      'skipped',
    ]);
    const skipped = results.find(
      (result): result is Extract<IssueReferralReservationResult, { outcome: 'skipped' }> =>
        result.outcome === 'skipped',
    );
    expect(skipped?.reason).toBe('referrer_cap_reached');
    expect(countCountedRewards(env, REFERRER_DISCORD_REF)).toBe(5);
  });

  it('drops missing or malformed referral input without storing raw attempts', async () => {
    const env = createTestBrokerEnv();

    await expect(
      reserveIssueReferralReward(env.BROKER_DB, {
        referralId: null,
        referredDiscordUserRef: REFERRED_DISCORD_REF,
        referredInstallationId: REFERRED_INSTALLATION_ID,
        referredHardwareHash: REFERRED_HARDWARE_HASH,
        referredHardwareHashSaltVersion: 7,
        nowIso: NOW_ISO,
      }),
    ).resolves.toEqual({ outcome: 'not_applicable', reason: 'no_referral_input' });
    await expect(
      reserveIssueReferralReward(env.BROKER_DB, {
        referralId: 'bad raw referral',
        referredDiscordUserRef: REFERRED_DISCORD_REF,
        referredInstallationId: REFERRED_INSTALLATION_ID,
        referredHardwareHash: REFERRED_HARDWARE_HASH,
        referredHardwareHashSaltVersion: 7,
        nowIso: NOW_ISO,
      }),
    ).resolves.toEqual({ outcome: 'not_applicable', reason: 'malformed_referral_input' });
    expect(readReferralRewards(env)).toEqual([]);
  });

  it('skips an active-looking Referral ID when the owner no longer has an active Discord identity', async () => {
    const env = createTestBrokerEnv();
    insertReferralCode(env, {
      referralId: REFERRAL_ID,
      ownerDiscordUserRef: REFERRER_DISCORD_REF,
      ownerInstallationId: REFERRER_INSTALLATION_ID,
    });

    const result = await reserveIssueReferralReward(env.BROKER_DB, {
      referralId: REFERRAL_ID,
      referredDiscordUserRef: REFERRED_DISCORD_REF,
      referredInstallationId: REFERRED_INSTALLATION_ID,
      referredHardwareHash: REFERRED_HARDWARE_HASH,
      referredHardwareHashSaltVersion: 7,
      nowIso: NOW_ISO,
    });

    expect(result).toEqual({ outcome: 'skipped', reason: 'referrer_not_eligible' });
    expect(readReferralRewards(env)).toEqual([
      expect.objectContaining({
        referral_id: REFERRAL_ID,
        referrer_discord_user_ref: REFERRER_DISCORD_REF,
        referrer_installation_id: REFERRER_INSTALLATION_ID,
        referred_bonus_status: 'skipped',
        referrer_bonus_status: 'skipped',
        skip_reason: 'referrer_not_eligible',
      }),
    ]);
  });
});

describe('Discord managed issue referral reservation', () => {
  afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
    vi.useRealTimers();
  });

  it('credits a reserved referral after creating a verified referral-increased child key', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date(NOW_ISO));

    const env = createTestBrokerEnv();
    insertActiveReferrer(env);
    const referredDiscordId = discordSnowflakeForAgeDays(31);
    const referredDiscordRef = await deriveExpectedDiscordUserRef(
      env.DISCORD_USER_REF_SECRET,
      referredDiscordId,
    );
    const started = await startDiscordSession({
      env,
      installationId: 'install-issue-referral-reserved',
      referralId: '7kq9m2',
    });
    const discordApi = mockDiscordApi({
      user: {
        id: referredDiscordId,
        verified: true,
      },
    });

    const response = await postDiscordIssue(
      env,
      await signedIssueRequest(started, {
        code: 'discord-oauth-code-referral-reserved',
        hardware_hash: 'hardware-hash-issue-referral-reserved',
      }),
    );

    expect(response.status).toBe(200);
    const payload = (await response.json()) as Record<string, unknown>;
    expect(payload).toEqual(
      expect.objectContaining({
        budget_usd: 0.09,
        managed_credential_ref: 'hash_discord_managed_child_test_1',
        referral_bonus_applied: true,
      }),
    );
    expect(payload.referral_bonus_applied).toBe(true);
    expect(payload.referral_id).not.toBe(REFERRAL_ID);
    expect(payload.talk_together_pass).toMatchObject({
      pass_id: payload.referral_id,
      invite_count: 0,
      invite_limit: 5,
      bonus_translations_per_friend: 200,
    });
    expectNoReferralRewardEstimateFields(payload);
    const createBody = JSON.parse(
      String(discordApi.openRouterCreateCalls[0]?.init?.body),
    ) as Record<string, unknown>;
    expect(createBody).toEqual(
      expect.objectContaining({
        limit: 0.09,
        limit_reset: null,
        include_byok_in_limit: false,
      }),
    );
    expect(discordApi.openRouterReferrerReadCalls).toHaveLength(1);
    expect(discordApi.openRouterReferrerPatchCalls).toHaveLength(1);
    expect(JSON.parse(String(discordApi.openRouterReferrerPatchCalls[0]?.init?.body))).toEqual({
      limit: 0.09,
    });
    await expect(readEntitlementBudget(env, started.installationId)).resolves.toEqual({
      status: 'active',
      budget_usd: 0.09,
      managed_credential_ref: 'hash_discord_managed_child_test_1',
      discord_issue_status: 'active',
    });
    await expect(readEntitlementBudget(env, REFERRER_INSTALLATION_ID)).resolves.toEqual({
      status: 'active',
      budget_usd: 0.09,
      managed_credential_ref: REFERRER_MANAGED_CREDENTIAL_REF,
      discord_issue_status: 'active',
    });
    expect(readReferralRewards(env)).toEqual([
      expect.objectContaining({
        referral_id: REFERRAL_ID,
        referrer_discord_user_ref: REFERRER_DISCORD_REF,
        referred_discord_user_ref: referredDiscordRef,
        referred_installation_id: started.installationId,
        referred_hardware_hash: 'hardware-hash-issue-referral-reserved',
        referred_bonus_status: 'credited',
        referrer_bonus_status: 'credited',
        referred_managed_credential_ref: 'hash_discord_managed_child_test_1',
        referrer_managed_credential_ref: REFERRER_MANAGED_CREDENTIAL_REF,
        skip_reason: null,
        failure_reason: null,
        credited_at: NOW_ISO,
      }),
    ]);
  });

  it('keeps ACK-supported referred issues at base budget until delivery ACK', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date(NOW_ISO));

    const env = createTestBrokerEnv();
    insertActiveReferrer(env);
    const referredDiscordId = discordSnowflakeForAgeDays(31);
    const referredDiscordRef = await deriveExpectedDiscordUserRef(
      env.DISCORD_USER_REF_SECRET,
      referredDiscordId,
    );
    const started = await startDiscordSession({
      env,
      installationId: 'install-issue-referral-delivery-pending',
      referralId: '7kq9m2',
    });
    const discordApi = mockDiscordApi({
      user: {
        id: referredDiscordId,
        verified: true,
      },
    });
    const signed = await signedIssueRequest(started, {
      code: 'discord-oauth-code-referral-delivery-pending',
      hardware_hash: 'hardware-hash-issue-referral-delivery-pending',
    });

    const response = await postDiscordIssue(env, {
      ...signed,
      delivery_ack_supported: true,
    });

    expect(response.status).toBe(200);
    const payload = (await response.json()) as Record<string, unknown>;
    expect(payload).toEqual(
      expect.objectContaining({
        budget_usd: 0.07,
        managed_credential_ref: 'hash_discord_managed_child_test_1',
        delivery_ack_required: true,
      }),
    );
    expect(payload).not.toHaveProperty('referral_bonus_applied');
    expect(JSON.parse(String(discordApi.openRouterCreateCalls[0]?.init?.body))).toEqual(
      expect.objectContaining({ limit: 0.07 }),
    );
    expect(discordApi.openRouterReferrerReadCalls).toHaveLength(0);
    expect(discordApi.openRouterReferrerPatchCalls).toHaveLength(0);
    await expect(readEntitlementBudget(env, started.installationId)).resolves.toEqual({
      status: 'pending_release',
      budget_usd: 0.07,
      managed_credential_ref: 'hash_discord_managed_child_test_1',
      discord_issue_status: 'delivery_pending',
    });
    expect(readReferralRewards(env)).toEqual([
      expect.objectContaining({
        referral_id: REFERRAL_ID,
        referrer_discord_user_ref: REFERRER_DISCORD_REF,
        referred_discord_user_ref: referredDiscordRef,
        referred_installation_id: started.installationId,
        referred_bonus_status: 'reserved',
        referrer_bonus_status: 'pending',
        referred_managed_credential_ref: null,
      }),
    ]);
  });

  it('skips referrer credit without failing issue when the referrer has no active managed key', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date(NOW_ISO));

    const env = createTestBrokerEnv();
    insertReferrerWithoutManagedKey(env);
    const referredDiscordId = discordSnowflakeForAgeDays(31);
    const referredDiscordRef = await deriveExpectedDiscordUserRef(
      env.DISCORD_USER_REF_SECRET,
      referredDiscordId,
    );
    const started = await startDiscordSession({
      env,
      installationId: 'install-issue-referral-no-referrer-key',
      referralId: REFERRAL_ID,
    });
    const discordApi = mockDiscordApi({
      user: {
        id: referredDiscordId,
        verified: true,
      },
    });

    const response = await postDiscordIssue(
      env,
      await signedIssueRequest(started, {
        code: 'discord-oauth-code-referrer-key-missing',
        hardware_hash: 'hardware-hash-issue-referrer-key-missing',
      }),
    );

    expect(response.status).toBe(200);
    expect(((await response.json()) as Record<string, unknown>).referral_bonus_applied).toBe(
      true,
    );
    expect(discordApi.openRouterReferrerReadCalls).toHaveLength(0);
    expect(discordApi.openRouterReferrerPatchCalls).toHaveLength(0);
    expect(readReferralRewards(env)).toEqual([
      expect.objectContaining({
        referral_id: REFERRAL_ID,
        referrer_discord_user_ref: REFERRER_DISCORD_REF,
        referred_discord_user_ref: referredDiscordRef,
        referred_bonus_status: 'credited',
        referrer_bonus_status: 'skipped',
        skip_reason: 'referrer_managed_key_missing',
        failure_reason: null,
        referrer_managed_credential_ref: null,
      }),
    ]);
  });

  it('records referrer provider failures without turning a successful referred issue into an error', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date(NOW_ISO));

    const env = createTestBrokerEnv();
    insertActiveReferrer(env);
    const started = await startDiscordSession({
      env,
      installationId: 'install-issue-referral-referrer-patch-fails',
      referralId: REFERRAL_ID,
    });
    const discordApi = mockDiscordApi({
      referrerOpenRouterMode: 'patch_failure',
      user: {
        id: discordSnowflakeForAgeDays(31),
        verified: true,
      },
    });

    const response = await postDiscordIssue(
      env,
      await signedIssueRequest(started, {
        code: 'discord-oauth-code-referrer-patch-fails',
        hardware_hash: 'hardware-hash-issue-referrer-patch-fails',
      }),
    );

    expect(response.status).toBe(200);
    expect(((await response.json()) as Record<string, unknown>).referral_bonus_applied).toBe(
      true,
    );
    expect(discordApi.openRouterReferrerPatchCalls).toHaveLength(1);
    await expect(readEntitlementBudget(env, REFERRER_INSTALLATION_ID)).resolves.toEqual({
      status: 'active',
      budget_usd: 0.07,
      managed_credential_ref: REFERRER_MANAGED_CREDENTIAL_REF,
      discord_issue_status: 'active',
    });
    expect(readReferralRewards(env)).toEqual([
      expect.objectContaining({
        referred_bonus_status: 'credited',
        referrer_bonus_status: 'failed',
        failure_reason: 'referrer_patch_failed',
        referrer_managed_credential_ref: REFERRER_MANAGED_CREDENTIAL_REF,
      }),
    ]);
  });

  it('uses an already-higher provider limit as a floor for referrer rewards', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date(NOW_ISO));

    const env = createTestBrokerEnv();
    insertActiveReferrer(env);
    const started = await startDiscordSession({
      env,
      installationId: 'install-issue-referral-referrer-provider-floor',
      referralId: REFERRAL_ID,
    });
    const discordApi = mockDiscordApi({
      referrerReadEffectiveLimit: 0.15,
      user: {
        id: discordSnowflakeForAgeDays(31),
        verified: true,
      },
    });

    const response = await postDiscordIssue(
      env,
      await signedIssueRequest(started, {
        code: 'discord-oauth-code-referrer-provider-floor',
        hardware_hash: 'hardware-hash-issue-referrer-provider-floor',
      }),
    );

    expect(response.status).toBe(200);
    expect(discordApi.openRouterReferrerReadCalls).toHaveLength(1);
    expect(discordApi.openRouterReferrerPatchCalls).toHaveLength(0);
    await expect(readEntitlementBudget(env, REFERRER_INSTALLATION_ID)).resolves.toEqual({
      status: 'active',
      budget_usd: 0.15,
      managed_credential_ref: REFERRER_MANAGED_CREDENTIAL_REF,
      discord_issue_status: 'active',
    });
    expect(readReferralRewards(env)).toEqual([
      expect.objectContaining({
        referrer_bonus_status: 'credited',
        referrer_managed_credential_ref: REFERRER_MANAGED_CREDENTIAL_REF,
      }),
    ]);
  });

  it('marks a reserved referral failed and cleans up when the provider effective limit is too low', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date(NOW_ISO));

    const env = createTestBrokerEnv();
    insertActiveReferrer(env);
    const referredDiscordId = discordSnowflakeForAgeDays(31);
    const referredDiscordRef = await deriveExpectedDiscordUserRef(
      env.DISCORD_USER_REF_SECRET,
      referredDiscordId,
    );
    const started = await startDiscordSession({
      env,
      installationId: 'install-issue-referral-low-limit',
      referralId: REFERRAL_ID,
    });
    const discordApi = mockDiscordApi({
      openRouterEffectiveLimit: 0.07,
      user: {
        id: referredDiscordId,
        verified: true,
      },
    });

    const response = await postDiscordIssue(
      env,
      await signedIssueRequest(started, {
        code: 'discord-oauth-code-referral-low-limit',
        hardware_hash: 'hardware-hash-issue-referral-low-limit',
      }),
    );

    expect(response.status).toBe(500);
    expect(await response.text()).not.toContain('or-discord-managed-child-key-test-1');
    expect(discordApi.openRouterCreateCalls).toHaveLength(1);
    expect(discordApi.openRouterCleanupCalls.map(({ init }) => init?.method)).toEqual([
      'PATCH',
      'DELETE',
    ]);
    await expect(readEntitlementBudget(env, started.installationId)).resolves.toBeNull();
    expect(readReferralRewards(env)).toEqual([
      expect.objectContaining({
        referral_id: REFERRAL_ID,
        referrer_discord_user_ref: REFERRER_DISCORD_REF,
        referred_discord_user_ref: referredDiscordRef,
        referred_installation_id: started.installationId,
        referred_bonus_status: 'failed',
        referrer_bonus_status: 'failed',
        failure_reason: 'issue_delivery_failed',
        referred_managed_credential_ref: null,
        credited_at: null,
      }),
    ]);
    expect(countCountedRewards(env, REFERRER_DISCORD_REF)).toBe(0);
  });

  it('marks a reserved referral failed when provider child-key creation fails before delivery', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date(NOW_ISO));

    const env = createTestBrokerEnv();
    insertActiveReferrer(env);
    const referredDiscordId = discordSnowflakeForAgeDays(31);
    const referredDiscordRef = await deriveExpectedDiscordUserRef(
      env.DISCORD_USER_REF_SECRET,
      referredDiscordId,
    );
    const started = await startDiscordSession({
      env,
      installationId: 'install-issue-referral-create-failure',
      referralId: REFERRAL_ID,
    });
    const discordApi = mockDiscordApi({
      openRouterMode: 'create_failure',
      user: {
        id: referredDiscordId,
        verified: true,
      },
    });

    const response = await postDiscordIssue(
      env,
      await signedIssueRequest(started, {
        code: 'discord-oauth-code-referral-create-failure',
        hardware_hash: 'hardware-hash-issue-referral-create-failure',
      }),
    );

    expect(response.status).toBe(500);
    expect(discordApi.openRouterCreateCalls).toHaveLength(1);
    expect(discordApi.openRouterCleanupCalls).toHaveLength(0);
    await expect(readEntitlementBudget(env, started.installationId)).resolves.toBeNull();
    expect(readReferralRewards(env)).toEqual([
      expect.objectContaining({
        referral_id: REFERRAL_ID,
        referred_discord_user_ref: referredDiscordRef,
        referred_installation_id: started.installationId,
        referred_bonus_status: 'failed',
        referrer_bonus_status: 'failed',
        failure_reason: 'issue_delivery_failed',
        referred_managed_credential_ref: null,
        credited_at: null,
      }),
    ]);
    expect(countCountedRewards(env, REFERRER_DISCORD_REF)).toBe(0);
  });

  it('keeps unknown valid-shaped referral input from becoming an auth gate', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date(NOW_ISO));

    const env = createTestBrokerEnv();
    const started = await startDiscordSession({
      env,
      installationId: 'install-issue-referral-unknown',
      referralId: UNKNOWN_REFERRAL_ID,
    });
    mockDiscordApi({
      user: {
        id: discordSnowflakeForAgeDays(31),
        verified: true,
      },
    });

    const response = await postDiscordIssue(
      env,
      await signedIssueRequest(started, {
        code: 'discord-oauth-code-referral-unknown',
        hardware_hash: 'hardware-hash-issue-referral-unknown',
      }),
    );

    expect(response.status).toBe(200);
    const payload = (await response.json()) as Record<string, unknown>;
    expect(payload).not.toHaveProperty('referral_bonus_applied');
    expect(readReferralRewards(env)).toEqual([
      expect.objectContaining({
        referral_id: UNKNOWN_REFERRAL_ID,
        referrer_discord_user_ref: null,
        referrer_installation_id: null,
        referred_installation_id: started.installationId,
        referred_bonus_status: 'skipped',
        referrer_bonus_status: 'skipped',
        skip_reason: 'unknown_referral_id',
      }),
    ]);
  });

  it('hashes the Discord issue route client IP and applies per-IP valid-shaped throttling without gating issue', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date(NOW_ISO));

    const env = createTestBrokerEnv();
    updateAbuseControls(env, (controls) => {
      controls.referralAttempts.validShaped.maxPerInstallation = 50;
      controls.referralAttempts.validShaped.maxPerIp = 1;
      controls.referralAttempts.unknown.maxPerInstallation = 50;
      controls.referralAttempts.unknown.maxPerIp = 50;
    });
    const clientIp = '198.51.100.44';
    const first = await startDiscordSession({
      env,
      installationId: 'install-route-valid-shaped-ip-first',
      referralId: UNKNOWN_REFERRAL_ID,
    });
    const second = await startDiscordSession({
      env,
      installationId: 'install-route-valid-shaped-ip-second',
      referralId: UNKNOWN_REFERRAL_ID,
    });
    mockDiscordApi({
      users: [
        { id: discordSnowflakeForAgeDays(31), verified: true },
        { id: discordSnowflakeForAgeDays(32), verified: true },
      ],
    });

    const firstResponse = await postDiscordIssue(
      env,
      await signedIssueRequest(first, {
        code: 'discord-oauth-code-route-valid-shaped-ip-first',
        hardware_hash: 'hardware-route-valid-shaped-ip-first',
      }),
      { headers: { 'cf-connecting-ip': clientIp } },
    );
    const secondResponse = await postDiscordIssue(
      env,
      await signedIssueRequest(second, {
        code: 'discord-oauth-code-route-valid-shaped-ip-second',
        hardware_hash: 'hardware-route-valid-shaped-ip-second',
      }),
      { headers: { 'cf-connecting-ip': clientIp } },
    );

    expect(firstResponse.status).toBe(200);
    expect(secondResponse.status).toBe(200);
    expect((await firstResponse.json()) as Record<string, unknown>).not.toHaveProperty(
      'referral_bonus_applied',
    );
    expect((await secondResponse.json()) as Record<string, unknown>).not.toHaveProperty(
      'referral_bonus_applied',
    );
    const rewards = readReferralRewards(env);
    expect(rewards).toEqual([
      expect.objectContaining({
        referral_id: UNKNOWN_REFERRAL_ID,
        referred_installation_id: first.installationId,
        referred_bonus_status: 'skipped',
        skip_reason: 'unknown_referral_id',
      }),
      expect.objectContaining({
        referral_id: UNKNOWN_REFERRAL_ID,
        referred_installation_id: second.installationId,
        referred_bonus_status: 'skipped',
        skip_reason: 'referral_attempt_rate_limited',
      }),
    ]);
    expect(rewards[0]?.attempt_ip_hash).toMatch(/^[a-f0-9]{64}$/u);
    expect(rewards[0]?.attempt_ip_hash).toBe(rewards[1]?.attempt_ip_hash);
    expect(rewards[0]?.attempt_ip_hash).not.toBe(clientIp);
  });

  it('applies per-IP unknown referral throttling through the Discord issue route without gating issue', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date(NOW_ISO));

    const env = createTestBrokerEnv();
    updateAbuseControls(env, (controls) => {
      controls.referralAttempts.validShaped.maxPerInstallation = 50;
      controls.referralAttempts.validShaped.maxPerIp = 50;
      controls.referralAttempts.unknown.maxPerInstallation = 50;
      controls.referralAttempts.unknown.maxPerIp = 1;
    });
    const clientIp = '198.51.100.45';
    const first = await startDiscordSession({
      env,
      installationId: 'install-route-unknown-ip-first',
      referralId: UNKNOWN_REFERRAL_ID,
    });
    const second = await startDiscordSession({
      env,
      installationId: 'install-route-unknown-ip-second',
      referralId: UNKNOWN_REFERRAL_ID,
    });
    mockDiscordApi({
      users: [
        { id: discordSnowflakeForAgeDays(31), verified: true },
        { id: discordSnowflakeForAgeDays(32), verified: true },
      ],
    });

    const firstResponse = await postDiscordIssue(
      env,
      await signedIssueRequest(first, {
        code: 'discord-oauth-code-route-unknown-ip-first',
        hardware_hash: 'hardware-route-unknown-ip-first',
      }),
      { headers: { 'cf-connecting-ip': clientIp } },
    );
    const secondResponse = await postDiscordIssue(
      env,
      await signedIssueRequest(second, {
        code: 'discord-oauth-code-route-unknown-ip-second',
        hardware_hash: 'hardware-route-unknown-ip-second',
      }),
      { headers: { 'cf-connecting-ip': clientIp } },
    );

    expect(firstResponse.status).toBe(200);
    expect(secondResponse.status).toBe(200);
    const rewards = readReferralRewards(env);
    expect(rewards.map((reward) => reward.skip_reason)).toEqual([
      'unknown_referral_id',
      'unknown_referral_id_rate_limited',
    ]);
    expect(rewards[0]?.attempt_ip_hash).toMatch(/^[a-f0-9]{64}$/u);
    expect(rewards[0]?.attempt_ip_hash).toBe(rewards[1]?.attempt_ip_hash);
    expect(rewards[0]?.attempt_ip_hash).not.toBe(clientIp);
  });

  it('skips valid referral input on a second Discord-managed issue while preserving lifetime rejection', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date(NOW_ISO));

    const env = createTestBrokerEnv();
    insertActiveReferrer(env);
    const discordUserId = discordSnowflakeForAgeDays(31);
    mockDiscordApi({
      user: {
        id: discordUserId,
        verified: true,
      },
    });

    const first = await startDiscordSession({
      env,
      installationId: 'install-referral-second-issue-first',
    });
    const firstResponse = await postDiscordIssue(
      env,
      await signedIssueRequest(first, {
        code: 'discord-oauth-code-second-issue-first',
        hardware_hash: 'hardware-hash-second-issue-first',
      }),
    );
    expect(firstResponse.status).toBe(200);
    expect(readReferralRewards(env)).toEqual([]);

    const second = await startDiscordSession({
      env,
      installationId: 'install-referral-second-issue-retry',
      referralId: REFERRAL_ID,
    });
    const secondClientIp = '198.51.100.46';
    const secondResponse = await postDiscordIssue(
      env,
      await signedIssueRequest(second, {
        code: 'discord-oauth-code-second-issue-retry',
        hardware_hash: 'hardware-hash-second-issue-retry',
      }),
      { headers: { 'cf-connecting-ip': secondClientIp } },
    );

    expect(secondResponse.status).toBe(409);
    expect(((await secondResponse.json()) as { error?: { subcode?: string } }).error?.subcode).toBe(
      'discord_lifetime_used',
    );
    expect(readCountedReferralRewards(env)).toEqual([]);
    expect(readReferralRewards(env)).toEqual([
      expect.objectContaining({
        referral_id: REFERRAL_ID,
        referrer_discord_user_ref: REFERRER_DISCORD_REF,
        referred_installation_id: second.installationId,
        referred_bonus_status: 'skipped',
        referrer_bonus_status: 'skipped',
        skip_reason: 'referred_not_first_successful',
        attempt_ip_hash: expect.stringMatching(/^[a-f0-9]{64}$/u),
      }),
    ]);
    expect(readReferralRewards(env)[0]?.attempt_ip_hash).not.toBe(secondClientIp);
  });

  it('skips valid referral input for a pre-existing managed user while preserving managed eligibility rejection', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date(NOW_ISO));

    const env = createTestBrokerEnv();
    insertActiveReferrer(env);
    const installationId = 'install-referral-pre-existing-managed-user';
    const keyPair = await createDeviceKeyPair();
    insertInstallation(env, {
      installationId,
      devicePublicKey: keyPair.devicePublicKey,
      hardwareHash: 'hardware-hash-pre-existing-managed-user',
      hardwareHashSaltVersion: 7,
    });
    insertEntitlement(env, {
      installation_id: installationId,
      status: 'active',
      budget_usd: 0.07,
      managed_credential_ref: 'managed-credential-pre-existing-managed-user',
      issued_at: NOW_ISO,
      expires_at: EXPIRES_AT_ISO,
    });
    const started = await startDiscordSession({
      env,
      installationId,
      keyPair,
      referralId: REFERRAL_ID,
    });
    mockDiscordApi({
      user: {
        id: discordSnowflakeForAgeDays(31),
        verified: true,
      },
    });

    const response = await postDiscordIssue(
      env,
      await signedIssueRequest(started, {
        code: 'discord-oauth-code-pre-existing-managed-user',
        hardware_hash: 'hardware-hash-pre-existing-managed-user',
      }),
    );

    expect(response.status).toBe(409);
    expect(((await response.json()) as { error?: { subcode?: string } }).error?.subcode).toBe(
      'hardware_duplicate',
    );
    expect(readCountedReferralRewards(env)).toEqual([]);
    expect(readReferralRewards(env)).toEqual([
      expect.objectContaining({
        referral_id: REFERRAL_ID,
        referrer_discord_user_ref: REFERRER_DISCORD_REF,
        referred_installation_id: started.installationId,
        referred_bonus_status: 'skipped',
        referrer_bonus_status: 'skipped',
        skip_reason: 'pre_existing_managed_user',
      }),
    ]);
  });

  it('does not retroactively reward valid referral input after a pre-feature first Discord auth', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date(NOW_ISO));

    const env = createTestBrokerEnv();
    insertActiveReferrer(env);
    const discordUserId = discordSnowflakeForAgeDays(31);
    const discordUserRef = await deriveExpectedDiscordUserRef(
      env.DISCORD_USER_REF_SECRET,
      discordUserId,
    );
    insertInstallation(env, {
      installationId: 'install-referral-pre-feature-existing-discord',
      devicePublicKey: 'pre-feature-existing-discord-device-key',
      hardwareHash: 'hardware-hash-pre-feature-existing-discord',
      hardwareHashSaltVersion: 7,
    });
    insertDiscordIdentity(env, {
      discordUserRef,
      installationId: 'install-referral-pre-feature-existing-discord',
      status: 'active',
    });
    insertEntitlement(env, {
      installation_id: 'install-referral-pre-feature-existing-discord',
      status: 'active',
      budget_usd: 0.07,
      managed_credential_ref: 'managed-credential-pre-feature-existing-discord',
      issued_at: NOW_ISO,
      expires_at: EXPIRES_AT_ISO,
      verified_hardware_hash: 'hardware-hash-pre-feature-existing-discord',
      verified_hardware_hash_salt_version: 7,
      discord_user_ref: discordUserRef,
      discord_issue_status: 'active',
      discord_issue_reserved_at: NOW_ISO,
      discord_issue_delivered_at: NOW_ISO,
    });
    const started = await startDiscordSession({
      env,
      installationId: 'install-referral-pre-feature-new-attempt',
      referralId: REFERRAL_ID,
    });
    mockDiscordApi({
      user: {
        id: discordUserId,
        verified: true,
      },
    });

    const response = await postDiscordIssue(
      env,
      await signedIssueRequest(started, {
        code: 'discord-oauth-code-pre-feature-new-attempt',
        hardware_hash: 'hardware-hash-pre-feature-new-attempt',
      }),
    );

    expect(response.status).toBe(409);
    expect(((await response.json()) as { error?: { subcode?: string } }).error?.subcode).toBe(
      'discord_lifetime_used',
    );
    expect(readCountedReferralRewards(env)).toEqual([]);
    expect(readReferralRewards(env)).toEqual([
      expect.objectContaining({
        referral_id: REFERRAL_ID,
        referrer_discord_user_ref: REFERRER_DISCORD_REF,
        referred_installation_id: started.installationId,
        referred_bonus_status: 'skipped',
        referrer_bonus_status: 'skipped',
        skip_reason: 'referred_not_first_successful',
      }),
    ]);
  });
});

describe('referrer reward limit update primitive', () => {
  afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
  });

  it('serializes concurrent absolute-limit updates so no referrer rewards are lost', async () => {
    let gateEnabled = false;
    let waitingClaims = 0;
    let releaseClaims: (() => void) | null = null;
    const bothClaimsReached = new Promise<void>((resolve) => {
      releaseClaims = resolve;
    });
    const env = createTestBrokerEnv({
      beforeRun: async ({ sql }) => {
        if (
          gateEnabled &&
          sql.includes('UPDATE referral_rewards') &&
          sql.includes("referrer_bonus_status = 'applying'") &&
          sql.includes('NOT EXISTS')
        ) {
          waitingClaims += 1;
          if (waitingClaims === 2) {
            releaseClaims?.();
          }
          await bothClaimsReached;
        }
      },
    });
    insertActiveReferrer(env);
    insertReferralReward(env, {
      referralId: REFERRAL_ID,
      referrerDiscordUserRef: REFERRER_DISCORD_REF,
      referrerInstallationId: REFERRER_INSTALLATION_ID,
      referredDiscordUserRef: `ph-discord-user-v1_${'M'.repeat(43)}`,
      referredInstallationId: 'install-concurrent-referrer-reward-a',
      referredHardwareHash: 'hardware-hash-concurrent-referrer-reward-a',
      referredBonusStatus: 'credited',
      referrerBonusStatus: 'pending',
    });
    insertReferralReward(env, {
      referralId: REFERRAL_ID,
      referrerDiscordUserRef: REFERRER_DISCORD_REF,
      referrerInstallationId: REFERRER_INSTALLATION_ID,
      referredDiscordUserRef: `ph-discord-user-v1_${'N'.repeat(43)}`,
      referredInstallationId: 'install-concurrent-referrer-reward-b',
      referredHardwareHash: 'hardware-hash-concurrent-referrer-reward-b',
      referredBonusStatus: 'credited',
      referrerBonusStatus: 'pending',
    });
    const openRouter = mockDiscordApi();
    gateEnabled = true;

    const results = await Promise.all([
      applyReferrerRewardLimitUpdates(env.BROKER_DB, {
        referrerDiscordUserRef: REFERRER_DISCORD_REF,
        managementApiKey: env.OPENROUTER_MANAGEMENT_API_KEY,
        nowIso: NOW_ISO,
      }),
      applyReferrerRewardLimitUpdates(env.BROKER_DB, {
        referrerDiscordUserRef: REFERRER_DISCORD_REF,
        managementApiKey: env.OPENROUTER_MANAGEMENT_API_KEY,
        nowIso: NOW_ISO,
      }),
    ]);

    expect(results.map((result) => result.outcome).sort()).toEqual([
      'applying',
      'credited',
    ]);
    expect(openRouter.openRouterReferrerPatchCalls).toHaveLength(1);
    expect(JSON.parse(String(openRouter.openRouterReferrerPatchCalls[0]?.init?.body))).toEqual({
      limit: 0.11,
    });
    await expect(readEntitlementBudget(env, REFERRER_INSTALLATION_ID)).resolves.toEqual({
      status: 'active',
      budget_usd: 0.11,
      managed_credential_ref: REFERRER_MANAGED_CREDENTIAL_REF,
      discord_issue_status: 'active',
    });
    expect(readReferralRewards(env)).toEqual([
      expect.objectContaining({
        referrer_bonus_status: 'credited',
        referrer_managed_credential_ref: REFERRER_MANAGED_CREDENTIAL_REF,
      }),
      expect.objectContaining({
        referrer_bonus_status: 'credited',
        referrer_managed_credential_ref: REFERRER_MANAGED_CREDENTIAL_REF,
      }),
    ]);
  });

  it('drains referrer rewards credited while another worker observes an active lease', async () => {
    const env = createTestBrokerEnv();
    insertActiveReferrer(env);
    insertReferralReward(env, {
      referralId: REFERRAL_ID,
      referrerDiscordUserRef: REFERRER_DISCORD_REF,
      referrerInstallationId: REFERRER_INSTALLATION_ID,
      referredDiscordUserRef: `ph-discord-user-v1_${'Q'.repeat(43)}`,
      referredInstallationId: 'install-drain-referrer-reward-a',
      referredHardwareHash: 'hardware-hash-drain-referrer-reward-a',
      referredBonusStatus: 'credited',
      referrerBonusStatus: 'pending',
    });
    let insertedConcurrentReward = false;
    let concurrentApplyResult: Awaited<
      ReturnType<typeof applyReferrerRewardLimitUpdates>
    > | null = null;
    const openRouter = mockDiscordApi({
      beforeReferrerRead: async () => {
        if (insertedConcurrentReward) {
          return;
        }
        insertedConcurrentReward = true;
        insertReferralReward(env, {
          referralId: REFERRAL_ID,
          referrerDiscordUserRef: REFERRER_DISCORD_REF,
          referrerInstallationId: REFERRER_INSTALLATION_ID,
          referredDiscordUserRef: `ph-discord-user-v1_${'R'.repeat(43)}`,
          referredInstallationId: 'install-drain-referrer-reward-b',
          referredHardwareHash: 'hardware-hash-drain-referrer-reward-b',
          referredBonusStatus: 'credited',
          referrerBonusStatus: 'pending',
        });
        concurrentApplyResult = await applyReferrerRewardLimitUpdates(env.BROKER_DB, {
          referrerDiscordUserRef: REFERRER_DISCORD_REF,
          managementApiKey: env.OPENROUTER_MANAGEMENT_API_KEY,
          nowIso: NOW_ISO,
        });
      },
    });

    await expect(
      applyReferrerRewardLimitUpdates(env.BROKER_DB, {
        referrerDiscordUserRef: REFERRER_DISCORD_REF,
        managementApiKey: env.OPENROUTER_MANAGEMENT_API_KEY,
        nowIso: NOW_ISO,
      }),
    ).resolves.toEqual(
      expect.objectContaining({
        outcome: 'credited',
        targetLimitUsd: 0.11,
      }),
    );

    expect(concurrentApplyResult).toEqual({ outcome: 'applying', reason: 'active_lease' });
    expect(openRouter.openRouterReferrerPatchCalls).toHaveLength(2);
    expect(
      openRouter.openRouterReferrerPatchCalls.map(({ init }) =>
        JSON.parse(String(init?.body ?? '{}')),
      ),
    ).toEqual([{ limit: 0.09 }, { limit: 0.11 }]);
    await expect(readEntitlementBudget(env, REFERRER_INSTALLATION_ID)).resolves.toEqual({
      status: 'active',
      budget_usd: 0.11,
      managed_credential_ref: REFERRER_MANAGED_CREDENTIAL_REF,
      discord_issue_status: 'active',
    });
    expect(readReferralRewards(env)).toEqual([
      expect.objectContaining({
        referred_installation_id: 'install-drain-referrer-reward-a',
        referrer_bonus_status: 'credited',
        referrer_managed_credential_ref: REFERRER_MANAGED_CREDENTIAL_REF,
      }),
      expect.objectContaining({
        referred_installation_id: 'install-drain-referrer-reward-b',
        referrer_bonus_status: 'credited',
        referrer_managed_credential_ref: REFERRER_MANAGED_CREDENTIAL_REF,
      }),
    ]);
  });

  it('recovers stale applying leases before crediting referrer rewards', async () => {
    const env = createTestBrokerEnv();
    insertActiveReferrer(env);
    insertReferralReward(env, {
      referralId: REFERRAL_ID,
      referrerDiscordUserRef: REFERRER_DISCORD_REF,
      referrerInstallationId: REFERRER_INSTALLATION_ID,
      referredDiscordUserRef: `ph-discord-user-v1_${'P'.repeat(43)}`,
      referredInstallationId: 'install-stale-applying-referrer-reward',
      referredHardwareHash: 'hardware-hash-stale-applying-referrer-reward',
      referredBonusStatus: 'credited',
      referrerBonusStatus: 'applying',
      referrerManagedCredentialRef: REFERRER_MANAGED_CREDENTIAL_REF,
      updatedAt: '2026-04-30T05:50:00.000Z',
    });
    const openRouter = mockDiscordApi();

    await expect(
      applyReferrerRewardLimitUpdates(env.BROKER_DB, {
        referrerDiscordUserRef: REFERRER_DISCORD_REF,
        managementApiKey: env.OPENROUTER_MANAGEMENT_API_KEY,
        nowIso: NOW_ISO,
      }),
    ).resolves.toEqual(
      expect.objectContaining({
        outcome: 'credited',
        creditedRows: 1,
        targetLimitUsd: 0.09,
      }),
    );
    expect(openRouter.openRouterReferrerPatchCalls).toHaveLength(1);
    expect(readReferralRewards(env)).toEqual([
      expect.objectContaining({
        referrer_bonus_status: 'credited',
        failure_reason: null,
      }),
    ]);
  });
});

async function startDiscordSession(input: {
  env: TestBrokerEnv;
  installationId: string;
  referralId?: string | null;
  keyPair?: DeviceKeyPair;
}): Promise<StartedDiscordSession> {
  const sessionKeyPair = input.keyPair ?? (await createDeviceKeyPair());
  const response = await postDiscordStart(input.env, {
    installation_id: input.installationId,
    device_public_key: sessionKeyPair.devicePublicKey,
    redirect_uri: REGISTERED_REDIRECT_URI,
    app_version: APP_VERSION,
    ...(input.referralId !== undefined ? { referral_id: input.referralId } : {}),
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
    env: input.env,
    keyPair: sessionKeyPair,
    installationId: input.installationId,
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
    hardware_hash: REFERRED_HARDWARE_HASH,
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

function insertActiveReferrer(
  env: TestBrokerEnv,
  overrides: { hardwareHash?: string; hardwareHashSaltVersion?: number } = {},
): void {
  const hardwareHash = overrides.hardwareHash ?? REFERRER_HARDWARE_HASH;
  const hardwareHashSaltVersion = overrides.hardwareHashSaltVersion ?? 7;
  insertInstallation(env, {
    installationId: REFERRER_INSTALLATION_ID,
    devicePublicKey: 'referrer-device-public-key',
    hardwareHash,
    hardwareHashSaltVersion,
  });
  insertDiscordIdentity(env, {
    discordUserRef: REFERRER_DISCORD_REF,
    installationId: REFERRER_INSTALLATION_ID,
    status: 'active',
  });
  insertEntitlement(env, {
    installation_id: REFERRER_INSTALLATION_ID,
    status: 'active',
    budget_usd: 0.07,
    managed_credential_ref: REFERRER_MANAGED_CREDENTIAL_REF,
    issued_at: NOW_ISO,
    expires_at: EXPIRES_AT_ISO,
    verified_hardware_hash: hardwareHash,
    verified_hardware_hash_salt_version: hardwareHashSaltVersion,
    discord_user_ref: REFERRER_DISCORD_REF,
    discord_issue_status: 'active',
    discord_issue_reserved_at: NOW_ISO,
    discord_issue_delivered_at: NOW_ISO,
  });
  insertReferralCode(env, {
    referralId: REFERRAL_ID,
    ownerDiscordUserRef: REFERRER_DISCORD_REF,
    ownerInstallationId: REFERRER_INSTALLATION_ID,
  });
}

function insertReferrerWithoutManagedKey(env: TestBrokerEnv): void {
  insertInstallation(env, {
    installationId: REFERRER_INSTALLATION_ID,
    devicePublicKey: 'referrer-device-public-key',
    hardwareHash: REFERRER_HARDWARE_HASH,
    hardwareHashSaltVersion: 7,
  });
  insertDiscordIdentity(env, {
    discordUserRef: REFERRER_DISCORD_REF,
    installationId: REFERRER_INSTALLATION_ID,
    status: 'active',
  });
  insertReferralCode(env, {
    referralId: REFERRAL_ID,
    ownerDiscordUserRef: REFERRER_DISCORD_REF,
    ownerInstallationId: REFERRER_INSTALLATION_ID,
  });
}

function insertReferralCode(
  env: TestBrokerEnv,
  input: {
    referralId: string;
    ownerDiscordUserRef: string;
    ownerInstallationId: string | null;
    status?: 'active' | 'disabled';
  },
): void {
  env.__db
    .prepare(
      `INSERT INTO referral_codes (
          referral_id,
          owner_discord_user_ref,
          owner_installation_id,
          status,
          created_at,
          updated_at
        ) VALUES (?, ?, ?, ?, ?, ?)`,
    )
    .run(
      input.referralId,
      input.ownerDiscordUserRef,
      input.ownerInstallationId,
      input.status ?? 'active',
      NOW_ISO,
      NOW_ISO,
    );
}

function insertReferralReward(
  env: TestBrokerEnv,
  input: {
    referralId: string;
    referrerDiscordUserRef: string | null;
    referrerInstallationId: string | null;
    referredDiscordUserRef: string;
    referredInstallationId: string;
    referredHardwareHash: string;
    referredBonusStatus: 'reserved' | 'credited' | 'skipped' | 'failed';
    referrerBonusStatus: 'pending' | 'applying' | 'credited' | 'skipped' | 'failed';
    skipReason?: string | null;
    failureReason?: string | null;
    referredManagedCredentialRef?: string | null;
    referrerManagedCredentialRef?: string | null;
    updatedAt?: string;
  },
): void {
  env.__db
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
        ) VALUES (?, ?, ?, ?, ?, ?, 7, ?, ?, ?, ?, ?, ?, ?, ?)`,
    )
    .run(
      input.referralId,
      input.referrerDiscordUserRef,
      input.referrerInstallationId,
      input.referredDiscordUserRef,
      input.referredInstallationId,
      input.referredHardwareHash,
      input.referredBonusStatus,
      input.referrerBonusStatus,
      input.skipReason ?? null,
      input.failureReason ?? null,
      input.referredManagedCredentialRef ?? null,
      input.referrerManagedCredentialRef ?? null,
      NOW_ISO,
      input.updatedAt ?? NOW_ISO,
    );
}

function seedCountedReferralRewards(env: TestBrokerEnv, count: number): void {
  for (let index = 0; index < count; index += 1) {
    insertReferralReward(env, {
      referralId: REFERRAL_ID,
      referrerDiscordUserRef: REFERRER_DISCORD_REF,
      referrerInstallationId: REFERRER_INSTALLATION_ID,
      referredDiscordUserRef: `ph-discord-user-v1_seeded-referred-${index}`,
      referredInstallationId: `install-seeded-referred-${index}`,
      referredHardwareHash: `hardware-hash-seeded-referred-${index}`,
      referredBonusStatus: index % 2 === 0 ? 'reserved' : 'credited',
      referrerBonusStatus: index % 2 === 0 ? 'pending' : 'credited',
    });
  }
}

function readReferralRewards(env: TestBrokerEnv): ReferralRewardRow[] {
  return env.__db
    .prepare(
      `SELECT referral_id,
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
              attempt_ip_hash,
              credited_at
         FROM referral_rewards
        ORDER BY id ASC`,
    )
    .all() as unknown as ReferralRewardRow[];
}

async function readEntitlementBudget(
  env: TestBrokerEnv,
  installationId: string,
): Promise<EntitlementBudgetRow | null> {
  return (
    (env.__db
      .prepare(
        `SELECT status,
                budget_usd,
                managed_credential_ref,
                discord_issue_status
           FROM openrouter_entitlements
          WHERE installation_id = ?`,
      )
      .get(installationId) as EntitlementBudgetRow | undefined) ?? null
  );
}

function countCountedRewards(env: TestBrokerEnv, referrerDiscordUserRef: string): number {
  const row = env.__db
    .prepare(
      `SELECT COUNT(*) AS count
         FROM referral_rewards
        WHERE referrer_discord_user_ref = ?
          AND referred_bonus_status IN ('reserved', 'credited')`,
    )
    .get(referrerDiscordUserRef) as { count: number };
  return Number(row.count);
}

function readCountedReferralRewards(env: TestBrokerEnv): ReferralRewardRow[] {
  return readReferralRewards(env).filter((reward) =>
    ['reserved', 'credited'].includes(reward.referred_bonus_status),
  );
}

function insertInstallation(
  env: TestBrokerEnv,
  input: {
    installationId: string;
    devicePublicKey: string;
    hardwareHash: string | null;
    hardwareHashSaltVersion: number | null;
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
          created_at,
          last_seen_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?)`,
    )
    .run(
      input.installationId,
      input.devicePublicKey,
      input.hardwareHash,
      input.hardwareHashSaltVersion,
      APP_VERSION,
      NOW_ISO,
      NOW_ISO,
    );
}

function insertDiscordIdentity(
  env: TestBrokerEnv,
  input: {
    discordUserRef: string;
    installationId: string;
    status: 'issuing' | 'delivery_pending' | 'active' | 'failed' | 'cleanup_required';
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
    .run(input.discordUserRef, input.installationId, input.status, NOW_ISO, NOW_ISO);
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
  users?: Array<Record<string, unknown>>;
  openRouterMode?: 'success' | 'create_failure';
  openRouterEffectiveLimit?: number;
  referrerOpenRouterMode?: 'success' | 'patch_failure';
  referrerReadEffectiveLimit?: number;
  referrerPatchEffectiveLimit?: number;
  beforeReferrerRead?: () => Promise<void> | void;
} = {}): {
  fetchMock: ReturnType<typeof vi.fn>;
  openRouterCreateCalls: Array<{ input: string | URL; init?: RequestInit }>;
  openRouterGuardrailCalls: Array<{ input: string | URL; init?: RequestInit }>;
  openRouterCleanupCalls: Array<{ input: string | URL; init?: RequestInit }>;
  openRouterReferrerReadCalls: Array<{ input: string | URL; init?: RequestInit }>;
  openRouterReferrerPatchCalls: Array<{ input: string | URL; init?: RequestInit }>;
} {
  const user = options.user ?? {
    id: discordSnowflakeForAgeDays(31),
    verified: true,
  };
  const users = options.users ?? [user];
  let userReadCount = 0;
  const openRouterCreateCalls: Array<{ input: string | URL; init?: RequestInit }> = [];
  const openRouterGuardrailCalls: Array<{ input: string | URL; init?: RequestInit }> = [];
  const openRouterCleanupCalls: Array<{ input: string | URL; init?: RequestInit }> = [];
  const openRouterReferrerReadCalls: Array<{ input: string | URL; init?: RequestInit }> = [];
  const openRouterReferrerPatchCalls: Array<{ input: string | URL; init?: RequestInit }> = [];
  const fetchMock = vi.fn(async (input: string | URL, init?: RequestInit) => {
    const url = String(input);
    const method = init?.method ?? 'GET';

    if (url === DISCORD_TOKEN_URL && method === 'POST') {
      return jsonResponse({
        access_token: 'discord-access-token',
        token_type: 'Bearer',
      });
    }

    if (url === DISCORD_USER_URL && method === 'GET') {
      const nextUser = users[Math.min(userReadCount, users.length - 1)] ?? user;
      userReadCount += 1;
      return jsonResponse(nextUser);
    }

    if (url === OPENROUTER_KEYS_URL && method === 'POST') {
      openRouterCreateCalls.push({ input, init });
      if (options.openRouterMode === 'create_failure') {
        return jsonResponse({ error: { message: 'create failed before key delivery' } }, 500);
      }

      const sequence = openRouterCreateCalls.length;
      const requestBody = JSON.parse(String(init?.body ?? '{}')) as { limit?: unknown };
      const requestedLimit = typeof requestBody.limit === 'number' ? requestBody.limit : 0.07;
      return jsonResponse(
        {
          key: `or-discord-managed-child-key-test-${sequence}`,
          data: {
            hash: `hash_discord_managed_child_test_${sequence}`,
            limit: options.openRouterEffectiveLimit ?? requestedLimit,
          },
        },
        201,
      );
    }

    if (url === OPENROUTER_GUARDRAIL_URL && method === 'POST') {
      openRouterGuardrailCalls.push({ input, init });
      return jsonResponse({ assigned_count: 1 });
    }

    if (url === `${OPENROUTER_KEYS_URL}/${REFERRER_MANAGED_CREDENTIAL_REF}` && method === 'GET') {
      openRouterReferrerReadCalls.push({ input, init });
      await options.beforeReferrerRead?.();
      return jsonResponse({
        data: {
          hash: REFERRER_MANAGED_CREDENTIAL_REF,
          limit: options.referrerReadEffectiveLimit ?? 0.07,
          limit_reset: null,
        },
      });
    }

    if (url === `${OPENROUTER_KEYS_URL}/${REFERRER_MANAGED_CREDENTIAL_REF}` && method === 'PATCH') {
      openRouterReferrerPatchCalls.push({ input, init });
      if (options.referrerOpenRouterMode === 'patch_failure') {
        return jsonResponse({ error: { message: 'referrer patch failed' } }, 500);
      }
      const requestBody = JSON.parse(String(init?.body ?? '{}')) as { limit?: unknown };
      const requestedLimit = typeof requestBody.limit === 'number' ? requestBody.limit : 0.07;
      return jsonResponse({
        data: {
          hash: REFERRER_MANAGED_CREDENTIAL_REF,
          limit: options.referrerPatchEffectiveLimit ?? requestedLimit,
          limit_reset: null,
        },
      });
    }

    if (url === `${OPENROUTER_KEYS_URL}/hash_discord_managed_child_test_1` && method === 'PATCH') {
      openRouterCleanupCalls.push({ input, init });
      return jsonResponse({ data: { hash: 'hash_discord_managed_child_test_1', disabled: true } });
    }

    if (url === `${OPENROUTER_KEYS_URL}/hash_discord_managed_child_test_1` && method === 'DELETE') {
      openRouterCleanupCalls.push({ input, init });
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
    openRouterReferrerReadCalls,
    openRouterReferrerPatchCalls,
  };
}

function discordSnowflakeForAgeDays(days: number): string {
  return discordSnowflakeForDate(new Date(Date.now() - days * 24 * 60 * 60 * 1000));
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
