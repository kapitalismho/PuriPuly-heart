import { afterEach, describe, expect, it, vi } from 'vitest';
import { buildManagedOperationId } from './test-support/managed-operation';

import {
  createManagedOperation,
  expireManagedOperation,
  hashManagedOperationResumeToken,
  sweepStaleManagedOperations,
} from '../src/managed-operation';
import { createTestBrokerEnv } from './test-support/sqlite-d1';

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


});
