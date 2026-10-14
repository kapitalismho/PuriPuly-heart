import { afterEach, describe, expect, it, vi } from 'vitest';

import {
  buildManagedOperationId,
  createManagedOperation,
  expireManagedOperation,
  hashManagedOperationResumeToken,
  sweepStaleManagedOperations,
} from '../src/managed-operation';
import { processManagedReferralSettlementJobs } from '../src/managed-referral-settlement';
import { markReservedIssueReferralFailed } from '../src/referral';
import { createTestBrokerEnv, type TestBrokerEnv } from './test-support/sqlite-d1';

const now = new Date('2026-09-01T12:00:00.000Z');

afterEach(() => vi.restoreAllMocks());

describe('managed lifecycle logging effects', () => {
  it('omits empty sweeps and repeated expiration while retaining a real expiration and its summary', async () => {
    const env = createTestBrokerEnv();
    const info = vi.spyOn(console, 'info').mockImplementation(() => undefined);
    try {
      expect(await sweepStaleManagedOperations(env, now)).toEqual({ expired: 0, reconciled: 0, retryReady: 0 });
      expect(info).not.toHaveBeenCalled();
      const { operation } = await createManagedOperation(env.BROKER_DB, {
        operationId: buildManagedOperationId(),
        resumeTokenHash: await hashManagedOperationResumeToken('synthetic'),
        issueSource: 'qq', subjectRef: 'ph-qq-subject-v1_logging', installationId: null,
        devicePublicKey: null, now: new Date('2026-08-01T00:00:00.000Z'),
      });
      info.mockClear();
      expect(await sweepStaleManagedOperations(env, now)).toEqual({ expired: 1, reconciled: 0, retryReady: 0 });
      expect(info).toHaveBeenCalledWith('managed_operation_expired', expect.objectContaining({ operation_id: operation.operation_id, state: 'FAILED', failure_reason: 'authorization_expired' }));
      expect(info).toHaveBeenCalledWith('managed_operation_sweep_completed', expect.objectContaining({ expired: 1, reconciled: 0, retry_ready: 0 }));
      info.mockClear();
      await expireManagedOperation(env.BROKER_DB, operation, now);
      expect(info).not.toHaveBeenCalled();
      expect(env.__db.prepare('SELECT state, failure_reason FROM managed_operations').get()).toMatchObject({ state: 'FAILED', failure_reason: 'authorization_expired' });
    } finally { env.__db.close(); }
  });

  it('does not claim expiration when the database update fails', async () => {
    const env = createTestBrokerEnv({ beforeRun: ({ sql }) => { if (sql.includes("SET state = 'FAILED'")) throw new Error('synthetic update failure'); } });
    const info = vi.spyOn(console, 'info').mockImplementation(() => undefined);
    try {
      const { operation } = await createManagedOperation(env.BROKER_DB, {
        operationId: buildManagedOperationId(), resumeTokenHash: await hashManagedOperationResumeToken('synthetic'),
        issueSource: 'qq', subjectRef: 'ph-qq-subject-v1_logging', installationId: null, devicePublicKey: null, now,
      });
      info.mockClear();
      await expect(expireManagedOperation(env.BROKER_DB, operation, now)).rejects.toThrow('synthetic update failure');
      expect(info).not.toHaveBeenCalled();
      expect(env.__db.prepare('SELECT state FROM managed_operations').get()).toMatchObject({ state: 'AUTHENTICATED' });
    } finally { env.__db.close(); }
  });

  it('records referral failure only for a reserved row that actually changes', async () => {
    const env = createTestBrokerEnv();
    const info = vi.spyOn(console, 'info').mockImplementation(() => undefined);
    const input = { referralId: 'ABCDEF', referredSource: 'qq' as const, referredSubjectRef: 'ph-qq-subject-v1_logging', referredInstallationId: null, failureReason: 'issue_delivery_failed' as const, nowIso: now.toISOString() };
    try {
      await markReservedIssueReferralFailed(env.BROKER_DB, input);
      expect(info).not.toHaveBeenCalled();
      env.__db.prepare(`INSERT INTO referral_rewards (referral_id, referrer_source, referrer_subject_ref, referred_source, referred_subject_ref, referred_bonus_status, referrer_bonus_status, created_at, updated_at) VALUES ('ABCDEF', 'qq', 'ph-qq-subject-v1_owner', 'qq', ?, 'reserved', 'pending', ?, ?)`).run(input.referredSubjectRef, input.nowIso, input.nowIso);
      await markReservedIssueReferralFailed(env.BROKER_DB, input);
      await markReservedIssueReferralFailed(env.BROKER_DB, input);
      expect(info).toHaveBeenCalledTimes(1);
      expect(info).toHaveBeenCalledWith('referral_reward_outcome', expect.objectContaining({ outcome: 'failed', referral_id: input.referralId, reason: input.failureReason }));
      expect(env.__db.prepare('SELECT referred_bonus_status, referrer_bonus_status, failure_reason FROM referral_rewards').get()).toMatchObject({ referred_bonus_status: 'failed', referrer_bonus_status: 'failed', failure_reason: input.failureReason });
    } finally { env.__db.close(); }
  });

  it.each(['normal', 'fenced', 'throw'] as const)('keeps settlement retry evidence as an attempt under %s release', async (mode) => {
    let env: TestBrokerEnv;
    env = createTestBrokerEnv({ beforeRun: ({ sql }) => {
      if (!sql.includes('SET fencing_token = NULL, lease_expires_at = NULL, next_attempt_at')) return;
      if (mode === 'fenced') env.__db.prepare("UPDATE managed_referral_settlement_jobs SET fencing_token = 'new-owner'").run();
      if (mode === 'throw') throw new Error('synthetic release failure');
    } });
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => undefined);
    const noNetwork = vi.fn(async () => { throw new Error('network forbidden'); });
    try {
      env.__db.prepare(`INSERT INTO managed_referral_settlement_jobs (source, referral_reward_id, delivery_id, phase, attempt_count, next_attempt_at, created_at, updated_at) VALUES ('qq', 999, 'synthetic', 'invitee_pending', 0, ?, ?, ?)`).run(now.toISOString(), now.toISOString(), now.toISOString());
      const run = processManagedReferralSettlementJobs(env, { now, limit: 1, fetchImpl: noNetwork });
      if (mode === 'throw') await expect(run).rejects.toThrow('synthetic release failure');
      else expect(await run).toEqual({ repaired: 0, claimed: 1, advanced: 0, completed: 0, retried: 1 });
      expect(noNetwork).not.toHaveBeenCalled();
      expect(warn).toHaveBeenCalledTimes(mode === 'throw' ? 2 : 1);
      for (const [event, fields] of warn.mock.calls) {
        expect(event).toBe('managed_referral_settlement_release_attempted');
        expect(fields).toMatchObject({ settlement_job_id: 1, referral_reward_id: 999, delivery_id: 'synthetic', phase: 'invitee_pending', attempt_count: 1 });
      }
      expect(env.__db.prepare('SELECT phase, attempt_count, next_attempt_at, last_error_code, fencing_token IS NOT NULL AS leased FROM managed_referral_settlement_jobs').get()).toMatchObject({
        phase: 'invitee_pending', attempt_count: 1,
        next_attempt_at: mode === 'normal' ? '2026-09-01T12:02:00.000Z' : now.toISOString(),
        last_error_code: mode === 'normal' ? 'invitee_context_unavailable' : null,
        leased: mode === 'normal' ? 0 : 1,
      });
    } finally { env.__db.close(); }
  });
});
