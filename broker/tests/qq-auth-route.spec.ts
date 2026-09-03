import { afterEach, describe, expect, it, vi } from 'vitest';

import app from '../src/index';
import {
  updateAbuseControls,
  updateAbuseRuntimeState,
} from './test-support/abuse-controls';
import { normalizedErrorEnvelope } from './test-support/errors';
import { sha256Base64Url } from './test-support/hash';
import {
  createTestBrokerEnv,
  type TestBrokerEnv,
  seedRequestEvent,
} from './test-support/sqlite-d1';

const QQ_AUTH_ASSERT_URL = 'http://broker.test/v1/auth/qq/assert';
const QQ_AUTH_ASSERT_ENDPOINT = 'POST /v1/auth/qq/assert';
const OPENROUTER_KEYS_URL = 'https://openrouter.ai/api/v1/keys';
const OPENROUTER_GUARDRAIL_URL =
  'https://openrouter.ai/api/v1/guardrails/test-managed-guardrail-id/assignments/keys';
const NOW_ISO = '2026-06-05T12:00:00.000Z';
const EXPECTED_QQ_EXPIRES_AT = '2026-09-05T12:00:00.000Z';
const encoder = new TextEncoder();

interface QqAuthAssertionRow {
  qq_subject_ref: string;
  credential_hash: string;
  asserted_at: string;
  received_at: string;
  status: string;
}

interface QqManagedEntitlementRow {
  qq_subject_ref: string;
  status: 'issuing' | 'delivery_pending' | 'active' | 'cleanup_required' | 'revoked';
  issue_ref: string;
  managed_credential_ref: string | null;
  budget_usd: number;
  reserved_at: string;
  issued_at: string | null;
  expires_at: string | null;
  delivered_at: string | null;
  child_key_creation_started_at: string | null;
  created_at: string;
  updated_at: string;
}

interface IssueSuccessEventRow {
  issue_source: string;
  installation_id: string | null;
  subject_ref: string;
  managed_credential_ref: string | null;
  ip_digest: string | null;
  ip_prefix_digest: string | null;
  country: string | null;
  observed_at: string;
}

describe('QQ auth assertion route', () => {
  afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
    vi.useRealTimers();
  });

  it('preserves legacy verification when OpenRouter issuance config is absent and stores only normalized derived evidence', async () => {
    const env = createTestBrokerEnv();
    env.OPENROUTER_MANAGEMENT_API_KEY = '   ';
    const qqIdentity = 'qq-openid-valid-user';
    const credential = await signQqCredential(env.QQ_AUTH_HMAC_PSK, qqIdentity);

    const response = await postQqAssertion(env, {
      qq_identity: qqIdentity,
      credential,
      asserted_at: '2026-06-05T20:03:00+08:00',
    });

    expect(response.status).toBe(200);
    const payload = (await response.json()) as {
      ok: boolean;
      status: string;
      qq_subject_ref: string;
    };
    expect(payload).toEqual({
      ok: true,
      status: 'verified',
      qq_subject_ref: expect.stringMatching(/^ph-qq-subject-v1_[A-Za-z0-9_-]+$/u),
    });

    const rows = listQqAssertions(env);
    expect(rows).toHaveLength(1);
    expect(rows[0]).toEqual({
      qq_subject_ref: payload.qq_subject_ref,
      credential_hash: `sha256-base64url-v1_${await sha256Base64Url(credential)}`,
      asserted_at: '2026-06-05T12:03:00.000Z',
      received_at: expect.any(String),
      status: 'verified',
    });
    const persistedRow = JSON.stringify(rows[0]);
    expect(persistedRow).not.toContain(qqIdentity);
    expect(persistedRow).not.toContain(credential);
    expect(countQqManagedEntitlements(env)).toBe(0);
  });

  it('returns already_verified for a repeated valid assertion only when OpenRouter issuance config is disabled', async () => {
    const env = createTestBrokerEnv();
    env.OPENROUTER_MANAGED_GUARDRAIL_ID = '';
    const qqIdentity = 'qq-openid-duplicate-user';
    const credential = await signQqCredential(env.QQ_AUTH_HMAC_PSK, qqIdentity);

    const firstResponse = await postQqAssertion(env, {
      qq_identity: qqIdentity,
      credential,
      asserted_at: '2026-06-05T12:03:00Z',
    });
    expect(firstResponse.status).toBe(200);
    const firstPayload = (await firstResponse.json()) as { qq_subject_ref: string };
    const originalRow = listQqAssertions(env)[0];

    const duplicateResponse = await postQqAssertion(env, {
      qq_identity: qqIdentity,
      credential,
      asserted_at: '2026-06-06T15:45:00Z',
    });

    expect(duplicateResponse.status).toBe(200);
    await expect(duplicateResponse.json()).resolves.toEqual({
      ok: true,
      status: 'already_verified',
      qq_subject_ref: firstPayload.qq_subject_ref,
    });
    expect(listQqAssertions(env)).toEqual([originalRow]);
    expect(countQqManagedEntitlements(env)).toBe(0);
  });

  it.each([
    [
      'missing management API key',
      (env: TestBrokerEnv) => {
        delete (env as Record<string, unknown>).OPENROUTER_MANAGEMENT_API_KEY;
      },
    ],
    [
      'blank management API key',
      (env: TestBrokerEnv) => {
        env.OPENROUTER_MANAGEMENT_API_KEY = '   ';
      },
    ],
    [
      'missing managed guardrail id',
      (env: TestBrokerEnv) => {
        delete (env as Record<string, unknown>).OPENROUTER_MANAGED_GUARDRAIL_ID;
      },
    ],
    [
      'blank managed guardrail id',
      (env: TestBrokerEnv) => {
        env.OPENROUTER_MANAGED_GUARDRAIL_ID = '   ';
      },
    ],
  ])(
    'preserves verification-only compatibility with %s and does not touch QQ entitlements',
    async (_caseName, mutateEnv) => {
      const env = createTestBrokerEnv();
      mutateEnv(env);
      const qqIdentity = 'qq-openid-runtime-gate-compat-user';
      const credential = await signQqCredential(env.QQ_AUTH_HMAC_PSK, qqIdentity);

      const response = await postQqAssertion(env, {
        qq_identity: qqIdentity,
        credential,
        asserted_at: '2026-06-05T12:03:00Z',
      });

      expect(response.status).toBe(200);
      await expect(response.json()).resolves.toEqual({
        ok: true,
        status: 'verified',
        qq_subject_ref: expect.stringMatching(/^ph-qq-subject-v1_[A-Za-z0-9_-]+$/u),
      });
      expect(listQqAssertions(env)).toHaveLength(1);
      expect(countQqManagedEntitlements(env)).toBe(0);
    },
  );

  it('issues a first eligible QQ assertion with a one-time OpenRouter key and source-aware monitoring', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date(NOW_ISO));

    const env = createTestBrokerEnv();
    const qqIdentity = 'qq-openid-first-production-issue-user';
    const credential = await signQqCredential(env.QQ_AUTH_HMAC_PSK, qqIdentity);
    const openRouter = mockOpenRouterManagementApi();

    const response = await postQqAssertion(
      env,
      {
        qq_identity: qqIdentity,
        credential,
        asserted_at: '2026-06-05T20:03:00+08:00',
      },
      {
        'cf-connecting-ip': '203.0.113.44',
        'cf-ipcountry': 'JP',
      },
    );

    expect(response.status).toBe(200);
    const payload = (await response.json()) as Record<string, unknown>;
    expect(payload).toEqual({
      ok: true,
      status: 'issued',
      qq_subject_ref: expect.stringMatching(/^ph-qq-subject-v1_[A-Za-z0-9_-]+$/u),
      openrouter_api_key: 'or-qq-managed-child-key-test-1',
      managed_credential_ref: 'hash_qq_managed_child_test_1',
      expires_at: EXPECTED_QQ_EXPIRES_AT,
      openrouter_user_id: expect.stringMatching(/^ph-or-user-v1_[A-Za-z0-9_-]+$/u),
    });

    const rows = listQqAssertions(env);
    expect(rows).toHaveLength(1);
    expect(rows[0]?.asserted_at).toBe('2026-06-05T12:03:00.000Z');
    expect(JSON.stringify(rows)).not.toContain(qqIdentity);
    expect(JSON.stringify(rows)).not.toContain(credential);

    const entitlement = readQqManagedEntitlement(
      env,
      payload.qq_subject_ref as string,
    );
    expect(entitlement).toEqual(
      expect.objectContaining({
        qq_subject_ref: payload.qq_subject_ref,
        status: 'active',
        managed_credential_ref: 'hash_qq_managed_child_test_1',
        budget_usd: 0.07,
        reserved_at: NOW_ISO,
        issued_at: NOW_ISO,
        expires_at: EXPECTED_QQ_EXPIRES_AT,
        delivered_at: NOW_ISO,
      }),
    );
    expect(entitlement?.issue_ref).toMatch(/^qq-issue-v1_[A-Za-z0-9_-]+$/u);
    const persistedEntitlement = JSON.stringify(entitlement);
    expect(persistedEntitlement).not.toContain(qqIdentity);
    expect(persistedEntitlement).not.toContain(credential);
    expect(persistedEntitlement).not.toContain('or-qq-managed-child-key-test-1');
    expect(countOpenRouterEntitlements(env)).toBe(0);

    expect(openRouter.openRouterCreateCalls).toHaveLength(1);
    const createBody = JSON.parse(
      String(openRouter.openRouterCreateCalls[0]?.init?.body),
    ) as Record<string, unknown>;
    expect(createBody).toEqual(
      expect.objectContaining({
        name: `puripuly-heart:qq:${entitlement?.issue_ref}`,
        limit: 0.07,
        limit_reset: null,
        include_byok_in_limit: false,
        expires_at: EXPECTED_QQ_EXPIRES_AT,
      }),
    );
    expect(JSON.stringify(createBody)).not.toContain(qqIdentity);
    expect(JSON.stringify(createBody)).not.toContain(credential);
    expect(openRouter.openRouterGuardrailCalls).toHaveLength(1);

    expect(listIssueSuccessEvents(env)).toEqual([
      expect.objectContaining({
        issue_source: 'qq',
        installation_id: null,
        subject_ref: payload.qq_subject_ref,
        managed_credential_ref: 'hash_qq_managed_child_test_1',
        country: 'JP',
        observed_at: NOW_ISO,
      }),
    ]);
  });

  describe('QQ Talk Together Pass runtime behavior', () => {
    const referralId = '7KQ9M2';
    const referrerSubjectRef = `ph-discord-user-v1_${'R'.repeat(43)}`;
    const referrerInstallationId = 'install-qq-pass-discord-owner';

    it('keeps base QQ issuance at 0.07 while the Pass feature is disabled', async () => {
      vi.useFakeTimers();
      vi.setSystemTime(new Date(NOW_ISO));

      const env = createTestBrokerEnv();
      updateQqTalkTogetherPassConfig(env, {
        enabled: false,
        rewards_enabled: false,
        daily_warning_count: 30,
        daily_max_count: 50,
      });
      insertActiveDiscordPassOwner(env, {
        referralId,
        referrerSubjectRef,
        referrerInstallationId,
      });
      const openRouter = mockOpenRouterManagementApi();

      const pendingIdentity = 'qq-openid-pass-disabled-pending';
      const pendingCredential = await signQqCredential(
        env.QQ_AUTH_HMAC_PSK,
        pendingIdentity,
      );
      const pendingResponse = await postQqAssertion(env, {
        qq_identity: pendingIdentity,
        credential: pendingCredential,
        asserted_at: NOW_ISO,
        delivery_ack_supported: true,
        referral_id: referralId,
        installation_id: 'install-qq-pass-disabled-pending',
      });
      const pendingPayload = (await pendingResponse.json()) as Record<string, unknown>;

      const activeIdentity = 'qq-openid-pass-disabled-active';
      const activeCredential = await signQqCredential(
        env.QQ_AUTH_HMAC_PSK,
        activeIdentity,
      );
      const activeResponse = await postQqAssertion(env, {
        qq_identity: activeIdentity,
        credential: activeCredential,
        asserted_at: NOW_ISO,
        referral_id: referralId,
        installation_id: 'install-qq-pass-disabled-active',
      });
      const activePayload = (await activeResponse.json()) as Record<string, unknown>;

      expect(pendingResponse.status).toBe(200);
      expect(pendingPayload.status).toBe('delivery_pending');
      expect(activeResponse.status).toBe(200);
      expect(activePayload).toEqual(
        expect.objectContaining({
          status: 'issued',
          qq_subject_ref: expect.any(String),
        }),
      );
      expect(activePayload).not.toHaveProperty('referral_id');
      expect(activePayload).not.toHaveProperty('talk_together_pass');
      expect(readQqManagedEntitlement(env, pendingPayload.qq_subject_ref as string)).toEqual(
        expect.objectContaining({ budget_usd: 0.07, status: 'delivery_pending' }),
      );
      expect(readQqManagedEntitlement(env, activePayload.qq_subject_ref as string)).toEqual(
        expect.objectContaining({ budget_usd: 0.07, status: 'active' }),
      );
      expect(readQqReferralRewards(env)).toEqual([]);
      expect(readQqOwnedReferralCodes(env)).toEqual([]);
      expect(openRouter.openRouterCreateCalls).toHaveLength(2);
    });

    it('records rewards_disabled without blocking Discord-to-QQ base issuance', async () => {
      vi.useFakeTimers();
      vi.setSystemTime(new Date(NOW_ISO));

      const env = createTestBrokerEnv();
      updateQqTalkTogetherPassConfig(env, {
        enabled: true,
        rewards_enabled: false,
        daily_warning_count: 30,
        daily_max_count: 50,
      });
      insertActiveDiscordPassOwner(env, {
        referralId,
        referrerSubjectRef,
        referrerInstallationId,
      });
      const qqIdentity = 'qq-openid-rewards-disabled-cross-source';
      const credential = await signQqCredential(env.QQ_AUTH_HMAC_PSK, qqIdentity);
      const openRouter = mockOpenRouterManagementApi();

      const response = await postQqAssertion(env, {
        qq_identity: qqIdentity,
        credential,
        asserted_at: NOW_ISO,
        delivery_ack_supported: true,
        referral_id: referralId,
        installation_id: 'install-qq-rewards-disabled',
      });
      const payload = (await response.json()) as Record<string, unknown>;

      expect(response.status).toBe(200);
      expect(payload.status).toBe('delivery_pending');
      expect(readQqManagedEntitlement(env, payload.qq_subject_ref as string)).toEqual(
        expect.objectContaining({ budget_usd: 0.07, status: 'delivery_pending' }),
      );
      expect(readQqReferralRewards(env)).toEqual([
        expect.objectContaining({
          referral_id: referralId,
          referrer_source: 'discord',
          referrer_subject_ref: referrerSubjectRef,
          referred_source: 'qq',
          referred_subject_ref: payload.qq_subject_ref,
          referred_bonus_status: 'skipped',
          referrer_bonus_status: 'skipped',
          skip_reason: 'rewards_disabled',
        }),
      ]);
      expect(openRouter.openRouterCreateCalls).toHaveLength(1);
    });

    it('warns on the 30th counted QQ reward and skips the 51st without blocking issuance', async () => {
      vi.useFakeTimers();
      vi.setSystemTime(new Date(NOW_ISO));

      const warningEnv = createTestBrokerEnv();
      updateQqTalkTogetherPassConfig(warningEnv, {
        enabled: true,
        rewards_enabled: true,
        daily_warning_count: 30,
        daily_max_count: 50,
      });
      insertActiveDiscordPassOwner(warningEnv, {
        referralId,
        referrerSubjectRef,
        referrerInstallationId,
      });
      seedCountedQqReferralRewards(warningEnv, 29);
      const warningSpy = vi.spyOn(console, 'warn').mockImplementation(() => undefined);
      const warningIdentity = 'qq-openid-daily-warning-thirtieth';
      const warningCredential = await signQqCredential(
        warningEnv.QQ_AUTH_HMAC_PSK,
        warningIdentity,
      );
      const warningOpenRouter = mockOpenRouterManagementApi();

      const warningResponse = await postQqAssertion(warningEnv, {
        qq_identity: warningIdentity,
        credential: warningCredential,
        asserted_at: NOW_ISO,
        delivery_ack_supported: true,
        referral_id: referralId,
        installation_id: 'install-qq-daily-warning-thirtieth',
      });
      const warningPayload = (await warningResponse.json()) as Record<string, unknown>;

      expect(warningResponse.status).toBe(200);
      expect(warningPayload.status).toBe('delivery_pending');
      expect(countCountedQqReferralRewards(warningEnv)).toBe(30);
      expect(readQqReferralRewards(warningEnv).at(-1)).toEqual(
        expect.objectContaining({
          referrer_source: 'discord',
          referred_source: 'qq',
          referred_subject_ref: warningPayload.qq_subject_ref,
          referred_bonus_status: 'reserved',
          referrer_bonus_status: 'pending',
          skip_reason: null,
        }),
      );
      expect(warningSpy).toHaveBeenCalledWith(
        'qq_referral_daily_warning_threshold_reached',
        expect.objectContaining({
          counted_rewards: 30,
          warning_threshold: 30,
          daily_max_count: 50,
        }),
      );
      expect(warningOpenRouter.openRouterCreateCalls).toHaveLength(1);

      const capEnv = createTestBrokerEnv();
      updateQqTalkTogetherPassConfig(capEnv, {
        enabled: true,
        rewards_enabled: true,
        daily_warning_count: 30,
        daily_max_count: 50,
      });
      insertActiveDiscordPassOwner(capEnv, {
        referralId,
        referrerSubjectRef,
        referrerInstallationId,
      });
      seedCountedQqReferralRewards(capEnv, 50);
      const capIdentity = 'qq-openid-daily-cap-fifty-first';
      const capCredential = await signQqCredential(capEnv.QQ_AUTH_HMAC_PSK, capIdentity);
      const capOpenRouter = mockOpenRouterManagementApi();

      const capResponse = await postQqAssertion(capEnv, {
        qq_identity: capIdentity,
        credential: capCredential,
        asserted_at: NOW_ISO,
        delivery_ack_supported: true,
        referral_id: referralId,
        installation_id: 'install-qq-daily-cap-fifty-first',
      });
      const capPayload = (await capResponse.json()) as Record<string, unknown>;

      expect(capResponse.status).toBe(200);
      expect(capPayload.status).toBe('delivery_pending');
      expect(readQqManagedEntitlement(capEnv, capPayload.qq_subject_ref as string)).toEqual(
        expect.objectContaining({ budget_usd: 0.07, status: 'delivery_pending' }),
      );
      expect(countCountedQqReferralRewards(capEnv)).toBe(50);
      expect(readQqReferralRewards(capEnv).at(-1)).toEqual(
        expect.objectContaining({
          referrer_source: 'discord',
          referred_source: 'qq',
          referred_subject_ref: capPayload.qq_subject_ref,
          referred_bonus_status: 'skipped',
          referrer_bonus_status: 'skipped',
          skip_reason: 'global_reward_cap_reached',
        }),
      );
      expect(capOpenRouter.openRouterCreateCalls).toHaveLength(1);
    });

    it('lazily creates a QQ-owned Pass from status only while the feature is enabled', async () => {
      vi.useFakeTimers();
      vi.setSystemTime(new Date(NOW_ISO));

      const enabledEnv = createTestBrokerEnv();
      updateQqTalkTogetherPassConfig(enabledEnv, {
        enabled: true,
        rewards_enabled: true,
        daily_warning_count: 30,
        daily_max_count: 50,
      });
      const enabledIdentity = 'qq-openid-lazy-status-enabled';
      const enabledCredential = await signQqCredential(
        enabledEnv.QQ_AUTH_HMAC_PSK,
        enabledIdentity,
      );
      const enabledSubjectRef = await deriveExpectedQqSubjectRef(
        enabledEnv.QQ_AUTH_HMAC_PSK,
        enabledIdentity,
      );
      insertQqManagedEntitlement(enabledEnv, {
        qq_subject_ref: enabledSubjectRef,
        status: 'active',
        issue_ref: 'qq-issue-v1_lazy-status-enabled',
        managed_credential_ref: 'hash_qq_lazy_status_enabled',
        reserved_at: NOW_ISO,
        issued_at: NOW_ISO,
        expires_at: EXPECTED_QQ_EXPIRES_AT,
        delivered_at: NOW_ISO,
      });

      const enabledResponse = await postQqStatus(enabledEnv, {
        qq_identity: enabledIdentity,
        credential: enabledCredential,
        installation_id: 'install-qq-lazy-status-enabled',
      });
      const enabledPayload = (await enabledResponse.json()) as Record<string, unknown>;

      expect(enabledResponse.status).toBe(200);
      expect(enabledPayload).toEqual({
        ok: true,
        status: 'active',
        referral_id: expect.stringMatching(/^[23456789ABCDEFGHJKMNPQRSTUVWXYZ]{6}$/u),
        talk_together_pass: {
          pass_id: enabledPayload.referral_id,
          invite_count: 0,
          invite_limit: 3,
          bonus_translations_per_friend: 200,
        },
      });
      expect(readQqOwnedReferralCodes(enabledEnv)).toEqual([
        expect.objectContaining({
          referral_id: enabledPayload.referral_id,
          owner_source: 'qq',
          owner_subject_ref: enabledSubjectRef,
          owner_installation_id: 'install-qq-lazy-status-enabled',
        }),
      ]);

      const disabledEnv = createTestBrokerEnv();
      updateQqTalkTogetherPassConfig(disabledEnv, {
        enabled: false,
        rewards_enabled: false,
        daily_warning_count: 30,
        daily_max_count: 50,
      });
      const disabledIdentity = 'qq-openid-lazy-status-disabled';
      const disabledCredential = await signQqCredential(
        disabledEnv.QQ_AUTH_HMAC_PSK,
        disabledIdentity,
      );
      const disabledSubjectRef = await deriveExpectedQqSubjectRef(
        disabledEnv.QQ_AUTH_HMAC_PSK,
        disabledIdentity,
      );
      insertQqManagedEntitlement(disabledEnv, {
        qq_subject_ref: disabledSubjectRef,
        status: 'active',
        issue_ref: 'qq-issue-v1_lazy-status-disabled',
        managed_credential_ref: 'hash_qq_lazy_status_disabled',
        reserved_at: NOW_ISO,
        issued_at: NOW_ISO,
        expires_at: EXPECTED_QQ_EXPIRES_AT,
        delivered_at: NOW_ISO,
      });

      const disabledResponse = await postQqStatus(disabledEnv, {
        qq_identity: disabledIdentity,
        credential: disabledCredential,
        installation_id: 'install-qq-lazy-status-disabled',
      });

      expect(disabledResponse.status).toBe(200);
      await expect(disabledResponse.json()).resolves.toEqual({
        ok: true,
        status: 'active',
      });
      expect(readQqOwnedReferralCodes(disabledEnv)).toEqual([]);
    });

    it('isolates referral persistence failure from base QQ key issuance', async () => {
      vi.useFakeTimers();
      vi.setSystemTime(new Date(NOW_ISO));

      const env = createTestBrokerEnv({
        beforeRun: ({ sql }) => {
          if (sql.includes('INSERT OR IGNORE INTO referral_rewards')) {
            throw new Error('synthetic QQ referral reservation failure');
          }
        },
      });
      updateQqTalkTogetherPassConfig(env, {
        enabled: true,
        rewards_enabled: true,
        daily_warning_count: 30,
        daily_max_count: 50,
      });
      insertActiveDiscordPassOwner(env, {
        referralId,
        referrerSubjectRef,
        referrerInstallationId,
      });
      const qqIdentity = 'qq-openid-referral-persistence-isolated';
      const credential = await signQqCredential(env.QQ_AUTH_HMAC_PSK, qqIdentity);
      const openRouter = mockOpenRouterManagementApi();

      const response = await postQqAssertion(env, {
        qq_identity: qqIdentity,
        credential,
        asserted_at: NOW_ISO,
        delivery_ack_supported: true,
        referral_id: referralId,
        installation_id: 'install-qq-referral-persistence-isolated',
      });
      const payload = (await response.json()) as Record<string, unknown>;

      expect(response.status).toBe(200);
      expect(payload.status).toBe('delivery_pending');
      expect(readQqManagedEntitlement(env, payload.qq_subject_ref as string)).toEqual(
        expect.objectContaining({ budget_usd: 0.07, status: 'delivery_pending' }),
      );
      expect(readQqReferralRewards(env)).toEqual([]);
      expect(openRouter.openRouterCreateCalls).toHaveLength(1);
    });

    async function postQqStatus(
      env: TestBrokerEnv,
      body: Record<string, unknown>,
    ): Promise<Response> {
      return app.request(
        'http://broker.test/v1/auth/qq/status',
        {
          method: 'POST',
          headers: { 'content-type': 'application/json' },
          body: JSON.stringify(body),
        },
        env,
      );
    }

    function updateQqTalkTogetherPassConfig(
      env: TestBrokerEnv,
      value: {
        enabled: boolean;
        rewards_enabled: boolean;
        daily_warning_count: number;
        daily_max_count: number;
      },
    ): void {
      env.__db
        .prepare("UPDATE broker_config SET value = ? WHERE key = 'qq_talk_together_pass'")
        .run(JSON.stringify(value));
    }

    function insertActiveDiscordPassOwner(
      env: TestBrokerEnv,
      input: {
        referralId: string;
        referrerSubjectRef: string;
        referrerInstallationId: string;
      },
    ): void {
      env.__db
        .prepare(
          `INSERT INTO installations (
              installation_id, device_public_key, hardware_hash,
              hardware_hash_salt_version, app_version, created_at, last_seen_at
            ) VALUES (?, ?, NULL, NULL, '1.2.3', ?, ?)`,
        )
        .run(
          input.referrerInstallationId,
          `device-key-${input.referrerInstallationId}`,
          NOW_ISO,
          NOW_ISO,
        );
      env.__db
        .prepare(
          `INSERT INTO discord_identities (
              discord_user_ref, entitlement_installation_id, status,
              ref_secret_version, created_at, updated_at
            ) VALUES (?, ?, 'active', 1, ?, ?)`,
        )
        .run(
          input.referrerSubjectRef,
          input.referrerInstallationId,
          NOW_ISO,
          NOW_ISO,
        );
      env.__db
        .prepare(
          `INSERT INTO referral_codes (
              referral_id, owner_source, owner_subject_ref,
              owner_installation_id, status, created_at, updated_at
            ) VALUES (?, 'discord', ?, ?, 'active', ?, ?)`,
        )
        .run(
          input.referralId,
          input.referrerSubjectRef,
          input.referrerInstallationId,
          NOW_ISO,
          NOW_ISO,
        );
    }

    function seedCountedQqReferralRewards(env: TestBrokerEnv, count: number): void {
      const alphabet = '23456789ABCDEFGHJKMNPQRSTUVWXYZ';
      for (let index = 0; index < count; index += 1) {
        let referralSequence = index;
        let seededReferralId = '';
        for (let character = 0; character < 6; character += 1) {
          seededReferralId = `${alphabet[referralSequence % alphabet.length]}${seededReferralId}`;
          referralSequence = Math.floor(referralSequence / alphabet.length);
        }
        const credited = index % 2 === 1;
        env.__db
          .prepare(
            `INSERT INTO referral_rewards (
                referral_id, referrer_source, referrer_subject_ref,
                referrer_installation_id, referred_source, referred_subject_ref,
                referred_installation_id, referred_hardware_hash,
                referred_hardware_hash_salt_version, referred_bonus_status,
                referrer_bonus_status, created_at, updated_at
              ) VALUES (?, 'qq', ?, NULL, 'qq', ?, NULL, NULL, NULL, ?, ?, ?, ?)`,
          )
          .run(
            seededReferralId,
            `ph-qq-subject-v1_seed-referrer-${index}`,
            `ph-qq-subject-v1_seed-referred-${index}`,
            credited ? 'credited' : 'reserved',
            credited ? 'credited' : 'pending',
            NOW_ISO,
            NOW_ISO,
          );
      }
    }

    function readQqReferralRewards(env: TestBrokerEnv): Array<Record<string, unknown>> {
      return env.__db
        .prepare(
          `SELECT referral_id, referrer_source, referrer_subject_ref,
                  referred_source, referred_subject_ref, referred_bonus_status,
                  referrer_bonus_status, skip_reason
             FROM referral_rewards
            ORDER BY id`,
        )
        .all() as unknown as Array<Record<string, unknown>>;
    }

    function countCountedQqReferralRewards(env: TestBrokerEnv): number {
      const row = env.__db
        .prepare(
          `SELECT COUNT(*) AS count
             FROM referral_rewards
            WHERE referred_source = 'qq'
              AND referred_bonus_status IN ('reserved', 'credited')`,
        )
        .get() as { count: number };
      return Number(row.count);
    }

    function readQqOwnedReferralCodes(env: TestBrokerEnv): Array<Record<string, unknown>> {
      return env.__db
        .prepare(
          `SELECT referral_id, owner_source, owner_subject_ref, owner_installation_id
             FROM referral_codes
            WHERE owner_source = 'qq'
            ORDER BY referral_id`,
        )
        .all() as unknown as Array<Record<string, unknown>>;
    }
  });

  it('allows an existing assertion-only subject to issue once when production issuance is enabled', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date(NOW_ISO));

    const env = createTestBrokerEnv();
    const qqIdentity = 'qq-openid-existing-assertion-only-user';
    const credential = await signQqCredential(env.QQ_AUTH_HMAC_PSK, qqIdentity);
    const qqSubjectRef = await deriveExpectedQqSubjectRef(
      env.QQ_AUTH_HMAC_PSK,
      qqIdentity,
    );
    insertQqAssertion(env, {
      qq_subject_ref: qqSubjectRef,
      credential_hash: `sha256-base64url-v1_${await sha256Base64Url(credential)}`,
      asserted_at: '2026-06-04T12:03:00.000Z',
    });
    mockOpenRouterManagementApi();

    const response = await postQqAssertion(env, {
      qq_identity: qqIdentity,
      credential,
      asserted_at: '2026-06-05T12:03:00Z',
    });

    expect(response.status).toBe(200);
    await expect(response.json()).resolves.toEqual(
      expect.objectContaining({
        ok: true,
        status: 'issued',
        qq_subject_ref: qqSubjectRef,
        openrouter_api_key: 'or-qq-managed-child-key-test-1',
      }),
    );
    expect(listQqAssertions(env)).toHaveLength(1);
    expect(readQqManagedEntitlement(env, qqSubjectRef)).toEqual(
      expect.objectContaining({
        status: 'active',
        managed_credential_ref: 'hash_qq_managed_child_test_1',
      }),
    );
  });

  it.each([
    [
      'missing',
      (env: TestBrokerEnv) => {
        delete (env as Record<string, unknown>).OPENROUTER_MANAGED_USER_HMAC_SECRET;
      },
    ],
    [
      'blank',
      (env: TestBrokerEnv) => {
        env.OPENROUTER_MANAGED_USER_HMAC_SECRET = '   ';
      },
    ],
    [
      'non-string',
      (env: TestBrokerEnv) => {
        (env as Record<string, unknown>).OPENROUTER_MANAGED_USER_HMAC_SECRET = 12345;
      },
    ],
  ])(
    'delivers the issued key without optional openrouter_user_id when the user-id secret is %s',
    async (_caseName, mutateEnv) => {
      vi.useFakeTimers();
      vi.setSystemTime(new Date(NOW_ISO));

      const env = createTestBrokerEnv();
      mutateEnv(env);
      const qqIdentity = 'qq-openid-optional-user-id-secret-user';
      const credential = await signQqCredential(env.QQ_AUTH_HMAC_PSK, qqIdentity);
      const openRouter = mockOpenRouterManagementApi();

      const response = await postQqAssertion(env, {
        qq_identity: qqIdentity,
        credential,
        asserted_at: '2026-06-05T12:03:00Z',
      });

      expect(response.status).toBe(200);
      const payload = (await response.json()) as Record<string, unknown>;
      expect(payload).toEqual(
        expect.objectContaining({
          ok: true,
          status: 'issued',
          openrouter_api_key: 'or-qq-managed-child-key-test-1',
          managed_credential_ref: 'hash_qq_managed_child_test_1',
        }),
      );
      expect(payload).not.toHaveProperty('openrouter_user_id');
      expect(readQqManagedEntitlement(env, payload.qq_subject_ref as string)).toEqual(
        expect.objectContaining({
          status: 'active',
          managed_credential_ref: 'hash_qq_managed_child_test_1',
        }),
      );
      expect(openRouter.openRouterCleanupCalls).toHaveLength(0);
    },
  );

  it('does not expose a QQ key when durable issue-success recording fails', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date(NOW_ISO));

    const qqIdentity = 'qq-openid-monitoring-failure-user';
    const env = createTestBrokerEnv({
      beforeRun: ({ sql }) => {
        if (sql.includes('INSERT INTO broker_issue_success_events')) {
          throw new Error(`monitoring failed ${qqIdentity}`);
        }
      },
    });
    const credential = await signQqCredential(env.QQ_AUTH_HMAC_PSK, qqIdentity);
    const qqSubjectRef = await deriveExpectedQqSubjectRef(
      env.QQ_AUTH_HMAC_PSK,
      qqIdentity,
    );
    const consoleErrorSpy = vi.spyOn(console, 'error').mockImplementation(() => undefined);
    const openRouter = mockOpenRouterManagementApi();

    const response = await postQqAssertion(env, {
      qq_identity: qqIdentity,
      credential,
      asserted_at: '2026-06-05T12:03:00Z',
    });

    expect(response.status).toBe(500);
    const payload = (await response.json()) as Record<string, unknown>;
    expect(payload).not.toHaveProperty('openrouter_api_key');
    expect(payload).not.toHaveProperty('managed_credential_ref');
    expect(readQqManagedEntitlement(env, qqSubjectRef)).toBeNull();
    expect(listIssueSuccessEvents(env)).toHaveLength(0);
    expect(
      openRouter.openRouterCleanupCalls.map(({ init }) => init?.method),
    ).toEqual(['PATCH', 'DELETE']);
    expect(stringifyConsoleCalls(consoleErrorSpy)).not.toContain(qqIdentity);
    expect(stringifyConsoleCalls(consoleErrorSpy)).not.toContain(credential);
  });

  it.each(['active', 'cleanup_required', 'revoked'] as const)(
    'returns qq_lifetime_used for %s QQ entitlements without creating a new child key',
    async (status) => {
      vi.useFakeTimers();
      vi.setSystemTime(new Date(NOW_ISO));

      const env = createTestBrokerEnv();
      const qqIdentity = `qq-openid-lifetime-${status}-user`;
      const credential = await signQqCredential(env.QQ_AUTH_HMAC_PSK, qqIdentity);
      const qqSubjectRef = await deriveExpectedQqSubjectRef(
        env.QQ_AUTH_HMAC_PSK,
        qqIdentity,
      );
      insertQqManagedEntitlement(env, {
        qq_subject_ref: qqSubjectRef,
        status,
        issue_ref: `qq-issue-v1_existing-${status}`,
        managed_credential_ref:
          status === 'revoked' ? null : `hash_qq_existing_${status}`,
        reserved_at: '2026-06-04T12:00:00.000Z',
        issued_at: status === 'active' ? '2026-06-04T12:00:00.000Z' : null,
        expires_at: status === 'active' ? '2026-09-04T12:00:00.000Z' : null,
        delivered_at: status === 'active' ? '2026-06-04T12:00:00.000Z' : null,
      });
      const openRouter = mockOpenRouterManagementApi();

      const response = await postQqAssertion(env, {
        qq_identity: qqIdentity,
        credential,
        asserted_at: '2026-06-05T12:03:00Z',
      });

      expect(response.status).toBe(409);
      await expect(response.json()).resolves.toEqual(
        normalizedErrorEnvelope({
          code: 'trial_not_eligible',
          class: 'terminal',
          subcode: 'qq_lifetime_used',
          message: 'QQ subject has already used a managed trial',
        }),
      );
      expect(openRouter.openRouterCreateCalls).toHaveLength(0);
      expect(readQqManagedEntitlement(env, qqSubjectRef)).toEqual(
        expect.objectContaining({ status }),
      );
    },
  );

  it('returns qq_already_issuing for a current issuing reservation without creating a child key', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date(NOW_ISO));

    const env = createTestBrokerEnv();
    const qqIdentity = 'qq-openid-currently-issuing-user';
    const credential = await signQqCredential(env.QQ_AUTH_HMAC_PSK, qqIdentity);
    const qqSubjectRef = await deriveExpectedQqSubjectRef(env.QQ_AUTH_HMAC_PSK, qqIdentity);
    insertQqManagedEntitlement(env, {
      qq_subject_ref: qqSubjectRef,
      status: 'issuing',
      issue_ref: 'qq-issue-v1_currently-issuing',
      managed_credential_ref: null,
      reserved_at: '2026-06-05T11:55:00.000Z',
    });
    const openRouter = mockOpenRouterManagementApi();

    const response = await postQqAssertion(env, {
      qq_identity: qqIdentity,
      credential,
      asserted_at: '2026-06-05T12:03:00Z',
    });

    expect(response.status).toBe(409);
    await expect(response.json()).resolves.toEqual(
      normalizedErrorEnvelope({
        code: 'trial_not_eligible',
        class: 'retryable',
        subcode: 'qq_already_issuing',
        retryAfterMs: 600000,
        message: 'QQ managed issuance is already in progress',
      }),
    );
    expect(openRouter.openRouterCreateCalls).toHaveLength(0);
  });

  it('returns qq_already_issuing for a simultaneous duplicate while first issuance is in progress', async () => {
    const env = createTestBrokerEnv();
    const qqIdentity = 'qq-openid-concurrent-issue-user';
    const credential = await signQqCredential(env.QQ_AUTH_HMAC_PSK, qqIdentity);
    let releaseCreate!: () => void;
    let createGateReleased = false;
    const createGate = new Promise<void>((resolve) => {
      releaseCreate = () => {
        createGateReleased = true;
        resolve();
      };
    });
    const openRouterCreateCalls: Array<{ input: string | URL; init?: RequestInit }> = [];
    const openRouterGuardrailCalls: Array<{ input: string | URL; init?: RequestInit }> = [];
    const fetchMock = vi.fn(async (input: string | URL, init?: RequestInit) => {
      const url = String(input);
      const method = init?.method ?? 'GET';

      if (url === OPENROUTER_KEYS_URL && method === 'POST') {
        openRouterCreateCalls.push({ input, init });
        await createGate;
        return jsonResponse(
          {
            key: 'or-qq-managed-child-key-concurrent-1',
            data: { hash: 'hash_qq_managed_child_concurrent_1' },
          },
          201,
        );
      }

      if (url === OPENROUTER_GUARDRAIL_URL && method === 'POST') {
        openRouterGuardrailCalls.push({ input, init });
        return jsonResponse({ assigned_count: 1 });
      }

      throw new Error(`unexpected OpenRouter API request: ${method} ${url}`);
    });
    vi.stubGlobal('fetch', fetchMock as typeof fetch);

    const assertion = {
      qq_identity: qqIdentity,
      credential,
      asserted_at: '2026-06-05T12:03:00Z',
    };
    const firstResponsePromise = postQqAssertion(env, assertion);
    try {
      await waitForCondition(() => openRouterCreateCalls.length === 1);

      const duplicateResponse = await postQqAssertion(env, assertion);

      expect(duplicateResponse.status).toBe(409);
      const duplicateBody = (await duplicateResponse.json()) as {
        error?: { retry_after_ms?: unknown } & Record<string, unknown>;
      };
      expect(duplicateBody).toEqual(
        expect.objectContaining({
          error: expect.objectContaining({
            code: 'trial_not_eligible',
            class: 'retryable',
            subcode: 'qq_already_issuing',
            message: 'QQ managed issuance is already in progress',
          }),
        }),
      );
      expect(duplicateBody.error?.retry_after_ms).toEqual(expect.any(Number));
      expect(duplicateBody.error?.retry_after_ms as number).toBeGreaterThan(0);
      expect(duplicateBody.error?.retry_after_ms as number).toBeLessThanOrEqual(
        900000,
      );
      expect(openRouterCreateCalls).toHaveLength(1);
      expect(openRouterGuardrailCalls).toHaveLength(0);

      releaseCreate();
      const firstResponse = await firstResponsePromise;
      expect(firstResponse.status).toBe(200);
      await expect(firstResponse.json()).resolves.toEqual(
        expect.objectContaining({
          ok: true,
          status: 'issued',
          openrouter_api_key: 'or-qq-managed-child-key-concurrent-1',
          managed_credential_ref: 'hash_qq_managed_child_concurrent_1',
        }),
      );
      expect(openRouterCreateCalls).toHaveLength(1);
      expect(openRouterGuardrailCalls).toHaveLength(1);
      const qqSubjectRef = await deriveExpectedQqSubjectRef(
        env.QQ_AUTH_HMAC_PSK,
        qqIdentity,
      );
      expect(readQqManagedEntitlement(env, qqSubjectRef)).toEqual(
        expect.objectContaining({
          status: 'active',
          managed_credential_ref: 'hash_qq_managed_child_concurrent_1',
        }),
      );
    } finally {
      if (!createGateReleased) {
        releaseCreate();
      }
      await firstResponsePromise.catch(() => undefined);
    }
  });

  it('reclaims stale no-key issuing reservations but blocks stale key-hash issuing rows as cleanup candidates', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date(NOW_ISO));

    const env = createTestBrokerEnv();
    const noKeyIdentity = 'qq-openid-stale-no-key-user';
    const noKeyCredential = await signQqCredential(
      env.QQ_AUTH_HMAC_PSK,
      noKeyIdentity,
    );
    const noKeySubjectRef = await deriveExpectedQqSubjectRef(
      env.QQ_AUTH_HMAC_PSK,
      noKeyIdentity,
    );
    insertQqManagedEntitlement(env, {
      qq_subject_ref: noKeySubjectRef,
      status: 'issuing',
      issue_ref: 'qq-issue-v1_stale-no-key-old',
      managed_credential_ref: null,
      reserved_at: '2026-06-05T11:40:00.000Z',
    });
    mockOpenRouterManagementApi();

    const reclaimedResponse = await postQqAssertion(env, {
      qq_identity: noKeyIdentity,
      credential: noKeyCredential,
      asserted_at: '2026-06-05T12:03:00Z',
    });

    expect(reclaimedResponse.status).toBe(200);
    const reclaimedEntitlement = readQqManagedEntitlement(env, noKeySubjectRef);
    expect(reclaimedEntitlement).toEqual(
      expect.objectContaining({
        status: 'active',
        managed_credential_ref: 'hash_qq_managed_child_test_1',
      }),
    );
    expect(reclaimedEntitlement?.issue_ref).not.toBe('qq-issue-v1_stale-no-key-old');

    const staleKeyIdentity = 'qq-openid-stale-key-hash-user';
    const staleKeyCredential = await signQqCredential(
      env.QQ_AUTH_HMAC_PSK,
      staleKeyIdentity,
    );
    const staleKeySubjectRef = await deriveExpectedQqSubjectRef(
      env.QQ_AUTH_HMAC_PSK,
      staleKeyIdentity,
    );
    insertQqManagedEntitlement(env, {
      qq_subject_ref: staleKeySubjectRef,
      status: 'issuing',
      issue_ref: 'qq-issue-v1_stale-key-old',
      managed_credential_ref: 'hash_qq_stale_key_candidate',
      reserved_at: '2026-06-05T11:40:00.000Z',
    });

    const blockedResponse = await postQqAssertion(env, {
      qq_identity: staleKeyIdentity,
      credential: staleKeyCredential,
      asserted_at: '2026-06-05T12:03:00Z',
    });

    expect(blockedResponse.status).toBe(409);
    await expect(blockedResponse.json()).resolves.toEqual(
      normalizedErrorEnvelope({
        code: 'trial_not_eligible',
        class: 'terminal',
        subcode: 'qq_lifetime_used',
        message: 'QQ subject has already used a managed trial',
      }),
    );
    expect(readQqManagedEntitlement(env, staleKeySubjectRef)).toEqual(
      expect.objectContaining({
        status: 'cleanup_required',
        issue_ref: 'qq-issue-v1_stale-key-old',
        managed_credential_ref: 'hash_qq_stale_key_candidate',
      }),
    );
  });

  it('notifies when stale QQ cleanup-required persistence fails', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date(NOW_ISO));
    const env = createTestBrokerEnv({
      beforeRun: ({ sql }) => {
        if (
          sql.includes('UPDATE qq_managed_entitlements') &&
          sql.includes("SET status = 'cleanup_required'")
        ) {
          throw new Error('synthetic stale cleanup-required failure');
        }
      },
    });
    const qqIdentity = 'qq-openid-stale-cleanup-state-failure';
    const credential = await signQqCredential(env.QQ_AUTH_HMAC_PSK, qqIdentity);
    const qqSubjectRef = await deriveExpectedQqSubjectRef(
      env.QQ_AUTH_HMAC_PSK,
      qqIdentity,
    );
    insertQqManagedEntitlement(env, {
      qq_subject_ref: qqSubjectRef,
      status: 'issuing',
      issue_ref: 'qq-issue-v1_stale-cleanup-state-failure',
      managed_credential_ref: 'hash_qq_stale_cleanup_state_failure',
      reserved_at: '2026-06-05T11:40:00.000Z',
    });
    const fetchMock = vi.fn(
      async (_input: RequestInfo | URL, _init?: RequestInit) =>
        new Response(null, { status: 204 }),
    );
    vi.stubGlobal('fetch', fetchMock);

    const response = await postQqAssertion(env, {
      qq_identity: qqIdentity,
      credential,
      asserted_at: '2026-06-05T12:03:00Z',
    });

    expect(response.status).toBe(409);
    expect(fetchMock).toHaveBeenCalledOnce();
    expect(String(fetchMock.mock.calls[0]?.[1]?.body)).toContain(
      'cleanup_required state could not be confirmed',
    );
    expect(readQqManagedEntitlement(env, qqSubjectRef)).toMatchObject({
      status: 'issuing',
      managed_credential_ref: 'hash_qq_stale_cleanup_state_failure',
    });
  });

  it('applies the active issuance brake and global cap before OpenRouter side effects', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date(NOW_ISO));

    const brakeEnv = createTestBrokerEnv();
    updateAbuseRuntimeState(brakeEnv, (state) => {
      state.brake.active = true;
      state.brake.reason = 'manual';
      state.brake.changedAt = NOW_ISO;
      state.brake.changedBy = 'operator';
    });
    const brakeIdentity = 'qq-openid-active-brake-user';
    const brakeCredential = await signQqCredential(
      brakeEnv.QQ_AUTH_HMAC_PSK,
      brakeIdentity,
    );
    const brakeOpenRouter = mockOpenRouterManagementApi();

    const brakeResponse = await postQqAssertion(brakeEnv, {
      qq_identity: brakeIdentity,
      credential: brakeCredential,
      asserted_at: '2026-06-05T12:03:00Z',
    });

    expect(brakeResponse.status).toBe(503);
    await expect(brakeResponse.json()).resolves.toEqual(
      normalizedErrorEnvelope({
        code: 'issuance_suspended',
        class: 'retryable',
        subcode: 'manual',
        message: 'new entitlement issuance is temporarily suspended',
      }),
    );
    expect(brakeOpenRouter.openRouterCreateCalls).toHaveLength(0);
    expect(countQqManagedEntitlements(brakeEnv)).toBe(0);

    const capEnv = createTestBrokerEnv();
    updateAbuseControls(capEnv, (controls) => {
      controls.newActiveEntitlementsPerDay.maxCount = 3;
    });
    insertQqManagedEntitlement(capEnv, {
      qq_subject_ref: 'ph-qq-subject-v1_existing-cap-subject',
      status: 'delivery_pending',
      issue_ref: 'qq-issue-v1_existing-cap-issue',
      managed_credential_ref: 'hash_qq_existing_cap_child',
      reserved_at: '2026-06-05T11:00:00.000Z',
      issued_at: '2026-06-05T11:00:00.000Z',
      expires_at: '2026-09-05T11:00:00.000Z',
      delivered_at: null,
    });
    insertQqManagedEntitlement(capEnv, {
      qq_subject_ref: 'ph-qq-subject-v1_delayed-ack-cap-subject',
      status: 'active',
      issue_ref: 'qq-issue-v1_delayed-ack-cap-issue',
      managed_credential_ref: 'hash_qq_delayed_ack_cap_child',
      reserved_at: '2026-06-05T10:00:00.000Z',
      issued_at: '2026-06-05T10:00:00.000Z',
      expires_at: '2026-09-05T10:00:00.000Z',
      delivered_at: '2026-06-06T00:01:00.000Z',
    });
    insertQqManagedEntitlement(capEnv, {
      qq_subject_ref: 'ph-qq-subject-v1_revoked-cap-subject',
      status: 'revoked',
      issue_ref: 'qq-issue-v1_revoked-cap-issue',
      managed_credential_ref: 'hash_qq_revoked_cap_child',
      reserved_at: '2026-06-05T09:00:00.000Z',
      issued_at: '2026-06-05T09:00:00.000Z',
      expires_at: '2026-09-05T09:00:00.000Z',
      delivered_at: '2026-06-05T09:00:00.000Z',
    });
    const capIdentity = 'qq-openid-global-cap-user';
    const capCredential = await signQqCredential(capEnv.QQ_AUTH_HMAC_PSK, capIdentity);
    const capOpenRouter = mockOpenRouterManagementApi();

    const capResponse = await postQqAssertion(capEnv, {
      qq_identity: capIdentity,
      credential: capCredential,
      asserted_at: '2026-06-05T12:03:00Z',
    });

    expect(capResponse.status).toBe(503);
    await expect(capResponse.json()).resolves.toEqual(
      normalizedErrorEnvelope({
        code: 'issuance_suspended',
        class: 'retryable',
        subcode: 'global_cap_reached',
        retryAfterMs: 43200000,
        message: 'Daily managed issuance cap reached',
      }),
    );
    expect(capOpenRouter.openRouterCreateCalls).toHaveLength(0);
    expect(countQqManagedEntitlements(capEnv)).toBe(3);
    expect(readQqManagedEntitlement(capEnv, 'ph-qq-subject-v1_existing-cap-subject')).toEqual(
      expect.objectContaining({
        status: 'delivery_pending',
        managed_credential_ref: 'hash_qq_existing_cap_child',
      }),
    );
  });

  it('releases only the matching reservation on OpenRouter create failure without leaking raw values', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date(NOW_ISO));

    const env = createTestBrokerEnv();
    const qqIdentity = 'qq-openid-openrouter-create-failure-user';
    const credential = await signQqCredential(env.QQ_AUTH_HMAC_PSK, qqIdentity);
    const openRouter = mockOpenRouterManagementApi({
      mode: 'create_failure',
      createFailureMessage: `create failed ${qqIdentity} ${credential}`,
    });

    const response = await postQqAssertion(env, {
      qq_identity: qqIdentity,
      credential,
      asserted_at: '2026-06-05T12:03:00Z',
    });

    expect(response.status).toBe(500);
    const responseText = await response.text();
    expect(responseText).not.toContain(qqIdentity);
    expect(responseText).not.toContain(credential);
    expect(openRouter.openRouterCreateCalls).toHaveLength(1);
    expect(openRouter.openRouterGuardrailCalls).toHaveLength(0);
    expect(countQqManagedEntitlements(env)).toBe(0);
  });

  it.each(['create_network_failure', 'create_retryable_failure'] as const)(
    'lifetime-blocks and notifies when OpenRouter create may have succeeded after %s',
    async (createMode) => {
      vi.useFakeTimers();
      vi.setSystemTime(new Date(NOW_ISO));

      const env = createTestBrokerEnv();
      const qqIdentity = `qq-openid-indeterminate-create-${createMode}-user`;
      const credential = await signQqCredential(env.QQ_AUTH_HMAC_PSK, qqIdentity);
      const qqSubjectRef = await deriveExpectedQqSubjectRef(
        env.QQ_AUTH_HMAC_PSK,
        qqIdentity,
      );
      const openRouter = mockOpenRouterManagementApi({
        mode: createMode,
      });

      const response = await postQqAssertion(env, {
        qq_identity: qqIdentity,
        credential,
        asserted_at: '2026-06-05T12:03:00Z',
      });

      expect(response.status).toBe(500);
      expect(openRouter.openRouterCreateCalls).toHaveLength(1);
      expect(readQqManagedEntitlement(env, qqSubjectRef)).toEqual(
        expect.objectContaining({
          status: 'issuing',
          managed_credential_ref: null,
          child_key_creation_started_at: NOW_ISO,
        }),
      );
      const cleanupIncidentCalls = openRouter.fetchMock.mock.calls.filter(
        ([request]) => String(request) === env.DISCORD_IMMEDIATE_ALERT_WEBHOOK_URL,
      );
      expect(cleanupIncidentCalls).toHaveLength(1);
      expect(String(cleanupIncidentCalls[0]?.[1]?.body)).toContain(
        'cleanup_required state could not be confirmed',
      );

      const retryResponse = await postQqAssertion(env, {
        qq_identity: qqIdentity,
        credential,
        asserted_at: '2026-06-05T12:20:00Z',
      });
      expect(retryResponse.status).toBe(409);
      expect(openRouter.openRouterCreateCalls).toHaveLength(1);
    },
  );

  it('cleans up and releases the matching reservation when guardrail assignment fails after child-key creation', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date(NOW_ISO));

    const env = createTestBrokerEnv();
    const qqIdentity = 'qq-openid-guardrail-cleanup-success-user';
    const credential = await signQqCredential(env.QQ_AUTH_HMAC_PSK, qqIdentity);
    const openRouter = mockOpenRouterManagementApi({ mode: 'guardrail_failure' });

    const response = await postQqAssertion(env, {
      qq_identity: qqIdentity,
      credential,
      asserted_at: '2026-06-05T12:03:00Z',
    });

    expect(response.status).toBe(500);
    expect(await response.text()).not.toContain('or-qq-managed-child-key-test-1');
    expect(openRouter.openRouterCreateCalls).toHaveLength(1);
    expect(openRouter.openRouterGuardrailCalls).toHaveLength(1);
    expect(openRouter.openRouterCleanupCalls.map(({ init }) => init?.method)).toEqual([
      'PATCH',
      'DELETE',
    ]);
    expect(countQqManagedEntitlements(env)).toBe(0);

    mockOpenRouterManagementApi();
    const retryResponse = await postQqAssertion(env, {
      qq_identity: qqIdentity,
      credential,
      asserted_at: '2026-06-05T12:04:00Z',
    });
    expect(retryResponse.status).toBe(200);
  });

  it('cleans up and releases QQ delivery_pending reservation when delivery row creation fails', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date(NOW_ISO));

    const env = createTestBrokerEnv();
    env.__db
      .prepare(
        `CREATE TRIGGER fail_managed_key_delivery_insert
         BEFORE INSERT ON managed_key_deliveries
         BEGIN
           SELECT RAISE(FAIL, 'test delivery insert failure');
         END`,
      )
      .run();
    const qqIdentity = 'qq-openid-delivery-row-cleanup-success-user';
    const credential = await signQqCredential(env.QQ_AUTH_HMAC_PSK, qqIdentity);
    const openRouter = mockOpenRouterManagementApi();

    const response = await postQqAssertion(env, {
      qq_identity: qqIdentity,
      credential,
      asserted_at: '2026-06-05T12:03:00Z',
      delivery_ack_supported: true,
    });

    expect(response.status).toBe(500);
    expect(await response.text()).not.toContain('or-qq-managed-child-key-test-1');
    expect(openRouter.openRouterCreateCalls).toHaveLength(1);
    expect(openRouter.openRouterGuardrailCalls).toHaveLength(1);
    expect(openRouter.openRouterCleanupCalls.map(({ init }) => init?.method)).toEqual([
      'PATCH',
      'DELETE',
    ]);
    expect(countQqManagedEntitlements(env)).toBe(0);
    expect(countManagedKeyDeliveries(env)).toBe(0);
  });

  it('marks QQ delivery_pending reservation cleanup_required when delivery row failure cleanup fails', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date(NOW_ISO));

    const rawOpenRouterChildKey = 'or-qq-managed-child-key-delivery-row-cleanup-redact';
    const childKeyHash = 'hash_qq_managed_child_delivery_row_cleanup_required';
    const env = createTestBrokerEnv();
    env.__db
      .prepare(
        `CREATE TRIGGER fail_managed_key_delivery_insert
         BEFORE INSERT ON managed_key_deliveries
         BEGIN
           SELECT RAISE(FAIL, 'test delivery insert failure');
         END`,
      )
      .run();
    const qqIdentity = 'qq-openid-delivery-row-cleanup-required-user';
    const credential = await signQqCredential(env.QQ_AUTH_HMAC_PSK, qqIdentity);
    const qqSubjectRef = await deriveExpectedQqSubjectRef(env.QQ_AUTH_HMAC_PSK, qqIdentity);
    const consoleErrorSpy = vi.spyOn(console, 'error').mockImplementation(() => undefined);
    const openRouter = mockOpenRouterManagementApi({
      mode: 'cleanup_failure',
      rawChildKey: rawOpenRouterChildKey,
      childKeyHash,
      cleanupFailureMessage: `cleanup failed ${qqIdentity} ${credential} ${rawOpenRouterChildKey}`,
    });

    const response = await postQqAssertion(env, {
      qq_identity: qqIdentity,
      credential,
      asserted_at: '2026-06-05T12:03:00Z',
      delivery_ack_supported: true,
    });

    expect(response.status).toBe(500);
    const responseText = await response.text();
    for (const sensitiveValue of [qqIdentity, credential, rawOpenRouterChildKey]) {
      expect(responseText).not.toContain(sensitiveValue);
      expect(stringifyConsoleCalls(consoleErrorSpy)).not.toContain(sensitiveValue);
    }
    expect(openRouter.openRouterCleanupCalls.map(({ init }) => init?.method)).toEqual([
      'PATCH',
      'DELETE',
    ]);
    expect(readQqManagedEntitlement(env, qqSubjectRef)).toEqual(
      expect.objectContaining({
        status: 'cleanup_required',
        managed_credential_ref: childKeyHash,
        issued_at: null,
        expires_at: null,
        delivered_at: null,
      }),
    );
    const cleanupIncidentCalls = openRouter.fetchMock.mock.calls.filter(
      ([input]) => String(input) === env.DISCORD_IMMEDIATE_ALERT_WEBHOOK_URL,
    );
    expect(cleanupIncidentCalls).toHaveLength(1);
    const cleanupIncidentBody = String(
      (cleanupIncidentCalls[0]?.[1] as RequestInit | undefined)?.body,
    );
    expect(cleanupIncidentBody).toContain('Broker managed-key cleanup incident');
    expect(cleanupIncidentBody).toContain('cleanup_required');
    for (const sensitiveValue of [qqIdentity, credential, rawOpenRouterChildKey]) {
      expect(cleanupIncidentBody).not.toContain(sensitiveValue);
    }
    expect(countManagedKeyDeliveries(env)).toBe(0);
  });

  it('marks the matching reservation cleanup_required when guardrail cleanup fails without leaking sensitive diagnostics', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date(NOW_ISO));

    const rawOpenRouterChildKey = 'or-qq-managed-child-key-sensitive-redact';
    const childKeyHash = 'hash_qq_managed_child_cleanup_required';
    const env = createTestBrokerEnv();
    const qqIdentity = 'qq-openid-guardrail-cleanup-required-user';
    const credential = await signQqCredential(env.QQ_AUTH_HMAC_PSK, qqIdentity);
    const qqSubjectRef = await deriveExpectedQqSubjectRef(env.QQ_AUTH_HMAC_PSK, qqIdentity);
    const consoleErrorSpy = vi.spyOn(console, 'error').mockImplementation(() => undefined);
    const openRouter = mockOpenRouterManagementApi({
      mode: 'guardrail_failure_cleanup_failure',
      rawChildKey: rawOpenRouterChildKey,
      childKeyHash,
      guardrailFailureMessage: `guardrail failed ${qqIdentity} ${credential} ${rawOpenRouterChildKey}`,
      cleanupFailureMessage: `cleanup failed ${qqIdentity} ${credential} ${rawOpenRouterChildKey}`,
    });

    const response = await postQqAssertion(env, {
      qq_identity: qqIdentity,
      credential,
      asserted_at: '2026-06-05T12:03:00Z',
    });

    expect(response.status).toBe(500);
    const responseText = await response.text();
    for (const sensitiveValue of [qqIdentity, credential, rawOpenRouterChildKey]) {
      expect(responseText).not.toContain(sensitiveValue);
      expect(stringifyConsoleCalls(consoleErrorSpy)).not.toContain(sensitiveValue);
    }
    expect(openRouter.openRouterCleanupCalls.map(({ init }) => init?.method)).toEqual([
      'PATCH',
      'DELETE',
    ]);
    expect(readQqManagedEntitlement(env, qqSubjectRef)).toEqual(
      expect.objectContaining({
        status: 'cleanup_required',
        managed_credential_ref: childKeyHash,
        issued_at: null,
        expires_at: null,
        delivered_at: null,
      }),
    );
  });

  it('emits cleanup-required diagnostics when cleanup fails and D1 cleanup-required marking fails', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date(NOW_ISO));

    const rawOpenRouterChildKey = 'or-qq-managed-child-key-d1-mark-redact';
    const childKeyHash = 'hash_qq_managed_child_d1_mark_failure';
    const qqIdentity = 'qq-openid-cleanup-mark-failure-user';
    let credential = '';
    const env = createTestBrokerEnv({
      beforeRun: ({ sql }) => {
        if (
          sql.includes('UPDATE qq_managed_entitlements') &&
          sql.includes("SET status = 'cleanup_required'")
        ) {
          throw new Error(`mark failed ${qqIdentity} ${credential} ${rawOpenRouterChildKey}`);
        }
      },
    });
    credential = await signQqCredential(env.QQ_AUTH_HMAC_PSK, qqIdentity);
    const qqSubjectRef = await deriveExpectedQqSubjectRef(env.QQ_AUTH_HMAC_PSK, qqIdentity);
    const consoleErrorSpy = vi.spyOn(console, 'error').mockImplementation(() => undefined);
    const openRouter = mockOpenRouterManagementApi({
      mode: 'guardrail_failure_cleanup_failure',
      rawChildKey: rawOpenRouterChildKey,
      childKeyHash,
      guardrailFailureMessage: `guardrail failed ${qqIdentity} ${credential} ${rawOpenRouterChildKey}`,
      cleanupFailureMessage: `cleanup failed ${qqIdentity} ${credential} ${rawOpenRouterChildKey}`,
    });

    const response = await postQqAssertion(env, {
      qq_identity: qqIdentity,
      credential,
      asserted_at: '2026-06-05T12:03:00Z',
    });

    expect(response.status).toBe(500);
    expect(openRouter.openRouterCleanupCalls.map(({ init }) => init?.method)).toEqual([
      'PATCH',
      'DELETE',
    ]);
    const consoleText = stringifyConsoleCalls(consoleErrorSpy);
    expect(consoleText).toContain('qq_managed_child_key_cleanup_required');
    expect(consoleText).toContain('qq_managed_child_key_cleanup_state_update_failed');
    expect(consoleText).toContain(childKeyHash);
    for (const sensitiveValue of [qqIdentity, credential, rawOpenRouterChildKey]) {
      expect(consoleText).not.toContain(sensitiveValue);
      expect(await response.clone().text()).not.toContain(sensitiveValue);
    }
    expect(readQqManagedEntitlement(env, qqSubjectRef)).toEqual(
      expect.objectContaining({
        status: 'issuing',
        managed_credential_ref: childKeyHash,
      }),
    );
  });

  it('attempts managed cleanup when activation fails and does not leave a raw-key-bearing row', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date(NOW_ISO));

    const env = createTestBrokerEnv({
      beforeRun: ({ sql }) => {
        if (
          sql.includes('UPDATE qq_managed_entitlements') &&
          sql.includes("SET status = 'active'")
        ) {
          throw new Error('synthetic activation failure');
        }
      },
    });
    const qqIdentity = 'qq-openid-activation-failure-user';
    const credential = await signQqCredential(env.QQ_AUTH_HMAC_PSK, qqIdentity);
    const openRouter = mockOpenRouterManagementApi();

    const response = await postQqAssertion(env, {
      qq_identity: qqIdentity,
      credential,
      asserted_at: '2026-06-05T12:03:00Z',
    });

    expect(response.status).toBe(500);
    expect(openRouter.openRouterCreateCalls).toHaveLength(1);
    expect(openRouter.openRouterGuardrailCalls).toHaveLength(1);
    expect(openRouter.openRouterCleanupCalls.map(({ init }) => init?.method)).toEqual([
      'PATCH',
      'DELETE',
    ]);
    expect(countQqManagedEntitlements(env)).toBe(0);
  });

  it('cleans up and releases the matching no-key reservation when storing the child-key hash fails', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date(NOW_ISO));

    const env = createTestBrokerEnv({
      beforeRun: ({ sql }) => {
        if (
          sql.includes('UPDATE qq_managed_entitlements') &&
          sql.includes('SET managed_credential_ref = ?') &&
          sql.includes('managed_credential_ref IS NULL')
        ) {
          throw new Error('synthetic child-key hash attachment failure');
        }
      },
    });
    const qqIdentity = 'qq-openid-attachment-failure-user';
    const credential = await signQqCredential(env.QQ_AUTH_HMAC_PSK, qqIdentity);
    const openRouter = mockOpenRouterManagementApi();

    const response = await postQqAssertion(env, {
      qq_identity: qqIdentity,
      credential,
      asserted_at: '2026-06-05T12:03:00Z',
    });

    expect(response.status).toBe(500);
    expect(openRouter.openRouterCreateCalls).toHaveLength(1);
    expect(openRouter.openRouterGuardrailCalls).toHaveLength(0);
    expect(openRouter.openRouterCleanupCalls.map(({ init }) => init?.method)).toEqual([
      'PATCH',
      'DELETE',
    ]);
    expect(countQqManagedEntitlements(env)).toBe(0);
  });

  it('emits orphan-key diagnostics when cleanup and D1 marking fail before storing the child-key hash', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date(NOW_ISO));

    const rawOpenRouterChildKey = 'or-qq-managed-child-key-unattached-redact';
    const childKeyHash = 'hash_qq_unattached_cleanup_mark_failure';
    const qqIdentity = 'qq-openid-unattached-mark-failure-user';
    let credential = '';
    const env = createTestBrokerEnv({
      beforeRun: ({ sql }) => {
        if (
          sql.includes('UPDATE qq_managed_entitlements') &&
          sql.includes('SET managed_credential_ref = ?') &&
          sql.includes('managed_credential_ref IS NULL')
        ) {
          throw new Error(`attachment failed ${qqIdentity} ${credential} ${rawOpenRouterChildKey}`);
        }
        if (
          sql.includes('UPDATE qq_managed_entitlements') &&
          sql.includes("SET status = 'cleanup_required'")
        ) {
          throw new Error(`mark failed ${qqIdentity} ${credential} ${rawOpenRouterChildKey}`);
        }
      },
    });
    credential = await signQqCredential(env.QQ_AUTH_HMAC_PSK, qqIdentity);
    const qqSubjectRef = await deriveExpectedQqSubjectRef(env.QQ_AUTH_HMAC_PSK, qqIdentity);
    const consoleErrorSpy = vi.spyOn(console, 'error').mockImplementation(() => undefined);
    const openRouter = mockOpenRouterManagementApi({
      mode: 'guardrail_failure_cleanup_failure',
      rawChildKey: rawOpenRouterChildKey,
      childKeyHash,
      cleanupFailureMessage: `cleanup failed ${qqIdentity} ${credential} ${rawOpenRouterChildKey}`,
    });

    const response = await postQqAssertion(env, {
      qq_identity: qqIdentity,
      credential,
      asserted_at: '2026-06-05T12:03:00Z',
    });

    expect(response.status).toBe(500);
    expect(openRouter.openRouterGuardrailCalls).toHaveLength(0);
    expect(openRouter.openRouterCleanupCalls.map(({ init }) => init?.method)).toEqual([
      'PATCH',
      'DELETE',
    ]);
    const consoleText = stringifyConsoleCalls(consoleErrorSpy);
    expect(consoleText).toContain('qq_managed_child_key_cleanup_required');
    expect(consoleText).toContain('qq_managed_child_key_cleanup_state_update_failed');
    expect(consoleText).toContain(childKeyHash);
    for (const sensitiveValue of [qqIdentity, credential, rawOpenRouterChildKey]) {
      expect(consoleText).not.toContain(sensitiveValue);
      expect(await response.clone().text()).not.toContain(sensitiveValue);
    }
    expect(readQqManagedEntitlement(env, qqSubjectRef)).toEqual(
      expect.objectContaining({
        status: 'issuing',
        managed_credential_ref: null,
        child_key_creation_started_at: NOW_ISO,
      }),
    );
    vi.advanceTimersByTime(16 * 60_000);
    const retryResponse = await postQqAssertion(env, {
      qq_identity: qqIdentity,
      credential,
      asserted_at: new Date().toISOString(),
    });
    expect(retryResponse.status).toBe(409);
    expect(openRouter.openRouterCreateCalls).toHaveLength(1);
  });

  it('logs D1 release failures after successful managed cleanup without leaking sensitive values', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date(NOW_ISO));

    const rawOpenRouterChildKey = 'or-qq-managed-child-key-release-redact';
    const childKeyHash = 'hash_qq_release_failure_child';
    const qqIdentity = 'qq-openid-release-failure-user';
    let credential = '';
    const env = createTestBrokerEnv({
      beforeRun: ({ sql }) => {
        if (sql.includes('DELETE FROM qq_managed_entitlements')) {
          throw new Error(`release failed ${qqIdentity} ${credential} ${rawOpenRouterChildKey}`);
        }
      },
    });
    credential = await signQqCredential(env.QQ_AUTH_HMAC_PSK, qqIdentity);
    const consoleErrorSpy = vi.spyOn(console, 'error').mockImplementation(() => undefined);
    const openRouter = mockOpenRouterManagementApi({
      mode: 'guardrail_failure',
      rawChildKey: rawOpenRouterChildKey,
      childKeyHash,
      guardrailFailureMessage: `guardrail failed ${qqIdentity} ${credential} ${rawOpenRouterChildKey}`,
    });

    const response = await postQqAssertion(env, {
      qq_identity: qqIdentity,
      credential,
      asserted_at: '2026-06-05T12:03:00Z',
    });

    expect(response.status).toBe(500);
    expect(openRouter.openRouterCleanupCalls.map(({ init }) => init?.method)).toEqual([
      'PATCH',
      'DELETE',
    ]);
    const consoleText = stringifyConsoleCalls(consoleErrorSpy);
    expect(consoleText).toContain('qq_managed_child_key_cleanup_release_failed');
    expect(consoleText).toContain(childKeyHash);
    const cleanupIncidentCalls = openRouter.fetchMock.mock.calls.filter(
      ([request]) => String(request) === env.DISCORD_IMMEDIATE_ALERT_WEBHOOK_URL,
    );
    expect(cleanupIncidentCalls).toHaveLength(1);
    expect(String(cleanupIncidentCalls[0]?.[1]?.body)).toContain(
      'cleanup_required state could not be confirmed',
    );
    for (const sensitiveValue of [qqIdentity, credential, rawOpenRouterChildKey]) {
      expect(consoleText).not.toContain(sensitiveValue);
      expect(await response.clone().text()).not.toContain(sensitiveValue);
    }
  });

  it('rejects an invalid QQ credential without persisting a row', async () => {
    const env = createTestBrokerEnv();
    const qqIdentity = 'qq-openid-invalid-credential-user';
    const expectedCredential = await signQqCredential(env.QQ_AUTH_HMAC_PSK, qqIdentity);
    const invalidCredential = '0'.repeat(64);

    const response = await postQqAssertion(env, {
      qq_identity: qqIdentity,
      credential: invalidCredential,
      asserted_at: '2026-06-05T12:03:00Z',
    });

    expect(response.status).toBe(401);
    const responseBody = await response.json();
    expect(responseBody).toEqual(
      normalizedErrorEnvelope({
        code: 'invalid_request',
        class: 'security_fail',
        subcode: 'qq_credential_invalid',
        message: 'QQ assertion credential is invalid',
      }),
    );
    const responseText = JSON.stringify(responseBody);
    expect(responseText).not.toContain(qqIdentity);
    expect(responseText).not.toContain(invalidCredential);
    expect(responseText).not.toContain(expectedCredential);
    expect(listQqAssertions(env)).toHaveLength(0);
    expect(countQqManagedEntitlements(env)).toBe(0);
  });

  it('rejects malformed credentials before HMAC comparison without persisting raw request values', async () => {
    const env = createTestBrokerEnv();
    env.OPENROUTER_MANAGEMENT_API_KEY = '';
    const qqIdentity = 'qq-openid-malformed-credential-user';
    const malformedCredential = 'A'.repeat(64);

    const response = await postQqAssertion(env, {
      qq_identity: qqIdentity,
      credential: malformedCredential,
      asserted_at: '2026-06-05T12:03:00Z',
    });

    expect(response.status).toBe(400);
    const responseBody = await response.json();
    expect(responseBody).toEqual(
      normalizedErrorEnvelope({
        code: 'invalid_request',
        class: 'terminal',
        message: 'credential must be exactly 64 lowercase hexadecimal characters',
      }),
    );
    const responseText = JSON.stringify(responseBody);
    expect(responseText).not.toContain(qqIdentity);
    expect(responseText).not.toContain(malformedCredential);
    expect(listQqAssertions(env)).toHaveLength(0);
    expect(countQqManagedEntitlements(env)).toBe(0);
  });

  it('rejects over-broad QQ identity values without persisting raw request values', async () => {
    const env = createTestBrokerEnv();
    env.OPENROUTER_MANAGED_GUARDRAIL_ID = '';
    const qqIdentity = 'q'.repeat(2049);
    const credential = await signQqCredential(env.QQ_AUTH_HMAC_PSK, qqIdentity);

    const response = await postQqAssertion(env, {
      qq_identity: qqIdentity,
      credential,
      asserted_at: '2026-06-05T12:03:00Z',
    });

    expect(response.status).toBe(400);
    const responseBody = await response.json();
    expect(responseBody).toEqual(
      normalizedErrorEnvelope({
        code: 'invalid_request',
        class: 'terminal',
        message: 'qq_identity must be between 1 and 2048 characters',
      }),
    );
    const responseText = JSON.stringify(responseBody);
    expect(responseText).not.toContain(qqIdentity);
    expect(responseText).not.toContain(credential);
    expect(listQqAssertions(env)).toHaveLength(0);
    expect(countQqManagedEntitlements(env)).toBe(0);
  });

  it('rejects asserted_at smuggling attempts without persisting raw timestamp text', async () => {
    const env = createTestBrokerEnv();
    env.OPENROUTER_MANAGEMENT_API_KEY = '';
    const qqIdentity = 'qq-openid-asserted-at-smuggling-user';
    const credential = await signQqCredential(env.QQ_AUTH_HMAC_PSK, qqIdentity);
    const smuggledAssertedAt = '2026-06-05T12:03:00Z\nqq-raw-smuggled-sentinel';

    const response = await postQqAssertion(env, {
      qq_identity: qqIdentity,
      credential,
      asserted_at: smuggledAssertedAt,
    });

    expect(response.status).toBe(400);
    const responseBody = await response.json();
    expect(responseBody).toEqual(
      normalizedErrorEnvelope({
        code: 'invalid_request',
        class: 'terminal',
        message: 'asserted_at must be a valid ISO-8601 timestamp',
      }),
    );
    const responseText = JSON.stringify(responseBody);
    expect(responseText).not.toContain(qqIdentity);
    expect(responseText).not.toContain(credential);
    expect(responseText).not.toContain(smuggledAssertedAt);
    expect(listQqAssertions(env)).toHaveLength(0);
    expect(countQqManagedEntitlements(env)).toBe(0);
  });

  it('counts invalid attempts toward the QQ assert IP rate limit', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-06-05T12:00:00Z'));

    const env = createTestBrokerEnv();
    updateAbuseControls(env, (controls) => {
      controls.qqAuthAssertIp.maxRequests = 1;
    });
    const headers = { 'cf-connecting-ip': '203.0.113.77' };

    const firstResponse = await postQqAssertion(
      env,
      {
        qq_identity: 'qq-openid-rate-limit-one',
        credential: '1'.repeat(64),
        asserted_at: '2026-06-05T12:00:00Z',
      },
      headers,
    );
    expect(firstResponse.status).toBe(401);

    const secondResponse = await postQqAssertion(
      env,
      {
        qq_identity: 'qq-openid-rate-limit-two',
        credential: '2'.repeat(64),
        asserted_at: '2026-06-05T12:00:01Z',
      },
      headers,
    );

    expect(secondResponse.status).toBe(429);
    await expect(secondResponse.json()).resolves.toEqual(
      normalizedErrorEnvelope({
        code: 'rate_limited',
        class: 'retryable',
        subcode: 'ip_rate_limited',
        retryAfterMs: 900000,
        message: `request rate limit exceeded for ${QQ_AUTH_ASSERT_ENDPOINT}`,
      }),
    );
    expect(await countQqRequestEvents(env, '203.0.113.77')).toBe(2);
    expect(listQqAssertions(env)).toHaveLength(0);
  });

  it('counts malformed attempts toward the QQ assert IP rate limit', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-06-05T12:00:00Z'));

    const env = createTestBrokerEnv();
    updateAbuseControls(env, (controls) => {
      controls.qqAuthAssertIp.maxRequests = 1;
    });
    const headers = { 'cf-connecting-ip': '203.0.113.78' };

    const firstResponse = await postQqAssertion(
      env,
      {
        qq_identity: 'qq-openid-rate-limit-malformed-one',
        credential: 'short',
        asserted_at: '2026-06-05T12:00:00Z',
      },
      headers,
    );
    expect(firstResponse.status).toBe(400);

    const secondResponse = await postQqAssertion(
      env,
      {
        qq_identity: 'qq-openid-rate-limit-malformed-two',
        credential: '2'.repeat(64),
        asserted_at: '2026-06-05T12:00:01Z',
      },
      headers,
    );

    expect(secondResponse.status).toBe(429);
    await expect(secondResponse.json()).resolves.toEqual(
      normalizedErrorEnvelope({
        code: 'rate_limited',
        class: 'retryable',
        subcode: 'ip_rate_limited',
        retryAfterMs: 900000,
        message: `request rate limit exceeded for ${QQ_AUTH_ASSERT_ENDPOINT}`,
      }),
    );
    expect(await countQqRequestEvents(env, '203.0.113.78')).toBe(2);
    expect(listQqAssertions(env)).toHaveLength(0);
    expect(countQqManagedEntitlements(env)).toBe(0);
  });

  it.each([
    [
      'missing',
      (env: TestBrokerEnv) => {
        delete (env as Record<string, unknown>).QQ_AUTH_HMAC_PSK;
      },
    ],
    [
      'blank',
      (env: TestBrokerEnv) => {
        env.QQ_AUTH_HMAC_PSK = '   ';
      },
    ],
  ])('fails closed when QQ_AUTH_HMAC_PSK is %s', async (_caseName, mutateEnv) => {
    const env = createTestBrokerEnv();
    mutateEnv(env);
    const qqIdentity = 'qq-openid-unconfigured-secret-user';
    const credential = '3'.repeat(64);

    const response = await postQqAssertion(env, {
      qq_identity: qqIdentity,
      credential,
      asserted_at: '2026-06-05T12:03:00Z',
    });

    expect(response.status).toBe(500);
    const responseBody = await response.json();
    expect(responseBody).toEqual(
      normalizedErrorEnvelope({
        code: 'internal_error',
        class: 'retryable',
        message: 'broker encountered an unexpected internal error',
      }),
    );
    const responseText = JSON.stringify(responseBody);
    expect(responseText).not.toContain(qqIdentity);
    expect(responseText).not.toContain(credential);
    expect(listQqAssertions(env)).toHaveLength(0);
    expect(countQqManagedEntitlements(env)).toBe(0);
  });

  it('rejects malformed JSON with the public invalid_request envelope', async () => {
    const env = createTestBrokerEnv();

    const response = await app.request(
      QQ_AUTH_ASSERT_URL,
      {
        method: 'POST',
        headers: {
          'content-type': 'application/json',
          'cf-connecting-ip': '203.0.113.88',
        },
        body: '{"qq_identity":"qq-openid-malformed","credential":',
      },
      env,
    );

    expect(response.status).toBe(400);
    const responseBody = await response.json();
    expect(responseBody).toEqual(
      normalizedErrorEnvelope({
        code: 'invalid_request',
        class: 'terminal',
        message: 'request body must be valid JSON',
      }),
    );
    expect(JSON.stringify(responseBody)).not.toContain('qq-openid-malformed');
    expect(listQqAssertions(env)).toHaveLength(0);
    expect(await countQqRequestEvents(env, '203.0.113.88', new Date().toISOString())).toBe(1);
  });
});

async function postQqAssertion(
  env: TestBrokerEnv,
  body: Record<string, unknown>,
  headers: Record<string, string> = {},
): Promise<Response> {
  return app.request(
    QQ_AUTH_ASSERT_URL,
    {
      method: 'POST',
      headers: {
        'content-type': 'application/json',
        ...headers,
      },
      body: JSON.stringify(body),
    },
    env,
  );
}

async function waitForCondition(
  predicate: () => boolean,
  maxAttempts = 50,
): Promise<void> {
  for (let attempt = 0; attempt < maxAttempts; attempt += 1) {
    if (predicate()) {
      return;
    }
    await new Promise<void>((resolve) => {
      setTimeout(resolve, 0);
    });
  }

  throw new Error('timed out waiting for concurrent QQ assertion checkpoint');
}

async function signQqCredential(secret: string, qqIdentity: string): Promise<string> {
  const key = await crypto.subtle.importKey(
    'raw',
    encoder.encode(secret),
    { name: 'HMAC', hash: 'SHA-256' },
    false,
    ['sign'],
  );
  const signature = await crypto.subtle.sign('HMAC', key, encoder.encode(qqIdentity));

  return Array.from(new Uint8Array(signature), (value) =>
    value.toString(16).padStart(2, '0'),
  ).join('');
}

function listQqAssertions(env: TestBrokerEnv): QqAuthAssertionRow[] {
  return env.__db
    .prepare(
      `SELECT qq_subject_ref, credential_hash, asserted_at, received_at, status
         FROM qq_auth_assertions
        ORDER BY qq_subject_ref`,
    )
    .all() as unknown as QqAuthAssertionRow[];
}

async function countQqRequestEvents(env: TestBrokerEnv, ip: string, nowIso = '2026-06-05T12:00:00Z'): Promise<number> {
  const { resolveRequestNetworkIdentity } = await import('../src/network-identity');
  const identity = await resolveRequestNetworkIdentity(
    ip,
    {
      current: env.NETWORK_IDENTITY_HMAC_SECRET,
      previous: null,
      previousVersion: null,
      currentVersion: 1,
    },
    new Date(nowIso),
  );
  const row = env.__db
    .prepare(
      `SELECT COUNT(*) AS count
         FROM broker_request_events
        WHERE endpoint = ?
          AND ip_digest = ?`,
    )
    .get(QQ_AUTH_ASSERT_ENDPOINT, identity?.digest ?? '') as { count: number };

  return Number(row.count);
}

function countQqManagedEntitlements(env: TestBrokerEnv): number {
  const row = env.__db
    .prepare('SELECT COUNT(*) AS count FROM qq_managed_entitlements')
    .get() as { count: number };

  return Number(row.count);
}

function countOpenRouterEntitlements(env: TestBrokerEnv): number {
  const row = env.__db
    .prepare('SELECT COUNT(*) AS count FROM openrouter_entitlements')
    .get() as { count: number };

  return Number(row.count);
}

function countManagedKeyDeliveries(env: TestBrokerEnv): number {
  const row = env.__db
    .prepare('SELECT COUNT(*) AS count FROM managed_key_deliveries')
    .get() as { count: number };

  return Number(row.count);
}

function readQqManagedEntitlement(
  env: TestBrokerEnv,
  qqSubjectRef: string,
): QqManagedEntitlementRow | null {
  return env.__db
    .prepare(
      `SELECT qq_subject_ref, status, issue_ref, managed_credential_ref,
              budget_usd, reserved_at, issued_at, expires_at, delivered_at,
              child_key_creation_started_at, created_at, updated_at
         FROM qq_managed_entitlements
        WHERE qq_subject_ref = ?`,
    )
    .get(qqSubjectRef) as QqManagedEntitlementRow | undefined ?? null;
}

function insertQqAssertion(
  env: TestBrokerEnv,
  input: {
    qq_subject_ref: string;
    credential_hash: string;
    asserted_at: string;
  },
): void {
  env.__db
    .prepare(
      `INSERT INTO qq_auth_assertions (
          qq_subject_ref,
          credential_hash,
          asserted_at,
          status
        ) VALUES (?, ?, ?, 'verified')`,
    )
    .run(input.qq_subject_ref, input.credential_hash, input.asserted_at);
}

function insertQqManagedEntitlement(
  env: TestBrokerEnv,
  input: {
    qq_subject_ref: string;
    status: QqManagedEntitlementRow['status'];
    issue_ref: string;
    managed_credential_ref?: string | null;
    budget_usd?: number;
    reserved_at: string;
    issued_at?: string | null;
    expires_at?: string | null;
    delivered_at?: string | null;
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
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
    )
    .run(
      input.qq_subject_ref,
      input.status,
      input.issue_ref,
      input.managed_credential_ref ?? null,
      input.budget_usd ?? 0.07,
      input.reserved_at,
      input.issued_at ?? null,
      input.expires_at ?? null,
      input.delivered_at ?? null,
      input.reserved_at,
      input.reserved_at,
    );
}

function listIssueSuccessEvents(env: TestBrokerEnv): IssueSuccessEventRow[] {
  return env.__db
    .prepare(
      `SELECT issue_source, installation_id, subject_ref, managed_credential_ref,
              ip_digest, ip_prefix_digest, country, observed_at
         FROM broker_issue_success_events
        ORDER BY id`,
    )
    .all() as unknown as IssueSuccessEventRow[];
}

async function deriveExpectedQqSubjectRef(
  secret: string,
  qqIdentity: string,
): Promise<string> {
  const bytes = await hmacSha256Bytes(
    secret,
    `puripuly-heart:qq-subject:v1\n${qqIdentity}`,
  );

  return `ph-qq-subject-v1_${encodeBase64Url(bytes)}`;
}

async function hmacSha256Bytes(secret: string, value: string): Promise<Uint8Array> {
  const key = await crypto.subtle.importKey(
    'raw',
    encoder.encode(secret),
    { name: 'HMAC', hash: 'SHA-256' },
    false,
    ['sign'],
  );
  const signature = await crypto.subtle.sign('HMAC', key, encoder.encode(value));

  return new Uint8Array(signature);
}

function encodeBase64Url(bytes: Uint8Array): string {
  const binary = Array.from(bytes, (value) => String.fromCharCode(value)).join('');
  return btoa(binary).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/u, '');
}

function mockOpenRouterManagementApi(options: {
  mode?:
    | 'success'
    | 'create_failure'
    | 'create_network_failure'
    | 'create_retryable_failure'
    | 'guardrail_failure'
    | 'guardrail_failure_cleanup_failure'
    | 'cleanup_failure';
  rawChildKey?: string;
  childKeyHash?: string;
  createFailureMessage?: string;
  guardrailFailureMessage?: string;
  cleanupFailureMessage?: string;
} = {}): {
  fetchMock: ReturnType<typeof vi.fn>;
  openRouterCreateCalls: Array<{ input: string | URL; init?: RequestInit }>;
  openRouterGuardrailCalls: Array<{ input: string | URL; init?: RequestInit }>;
  openRouterCleanupCalls: Array<{ input: string | URL; init?: RequestInit }>;
} {
  const openRouterCreateCalls: Array<{ input: string | URL; init?: RequestInit }> = [];
  const openRouterGuardrailCalls: Array<{ input: string | URL; init?: RequestInit }> = [];
  const openRouterCleanupCalls: Array<{ input: string | URL; init?: RequestInit }> = [];
  const childKeyHash = options.childKeyHash ?? 'hash_qq_managed_child_test_1';
  const fetchMock = vi.fn(async (input: string | URL, init?: RequestInit) => {
    const url = String(input);
    const method = init?.method ?? 'GET';

    if (url === OPENROUTER_KEYS_URL && method === 'POST') {
      openRouterCreateCalls.push({ input, init });
      if (options.mode === 'create_failure') {
        return jsonResponse(
          {
            error: {
              message: options.createFailureMessage ?? 'create failed before key delivery',
            },
          },
          400,
        );
      }
      if (options.mode === 'create_retryable_failure') {
        return jsonResponse({ error: { message: 'create temporarily failed' } }, 503);
      }
      if (options.mode === 'create_network_failure') {
        throw new TypeError('OpenRouter create response was interrupted');
      }

      const sequence = openRouterCreateCalls.length;
      return jsonResponse(
        {
          key: options.rawChildKey ?? `or-qq-managed-child-key-test-${sequence}`,
          data: {
            hash: options.childKeyHash ?? `hash_qq_managed_child_test_${sequence}`,
          },
        },
        201,
      );
    }

    if (url === OPENROUTER_GUARDRAIL_URL && method === 'POST') {
      openRouterGuardrailCalls.push({ input, init });
      if (
        options.mode === 'guardrail_failure' ||
        options.mode === 'guardrail_failure_cleanup_failure'
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
        options.mode === 'guardrail_failure_cleanup_failure' ||
        options.mode === 'cleanup_failure'
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
        options.mode === 'guardrail_failure_cleanup_failure' ||
        options.mode === 'cleanup_failure'
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

    throw new Error(`unexpected OpenRouter API request: ${method} ${url}`);
  });

  vi.stubGlobal('fetch', fetchMock as typeof fetch);
  return {
    fetchMock,
    openRouterCreateCalls,
    openRouterGuardrailCalls,
    openRouterCleanupCalls,
  };
}

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'content-type': 'application/json' },
  });
}

function stringifyConsoleCalls(spy: ReturnType<typeof vi.spyOn>): string {
  return JSON.stringify(spy.mock.calls);
}
