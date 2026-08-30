import { describe, expect, it } from 'vitest';

import {
  getBrokerAbuseRuntimeState,
  persistBrokerAbuseRuntimeState,
} from '../src/abuse-controls';
import { readAbuseRuntimeState } from './test-support/abuse-controls';
import { createTestBrokerEnv } from './test-support/sqlite-d1';

describe('broker abuse runtime-state write path', () => {
  it('preserves delivered daily-report metadata when a stale monitoring writer commits later', async () => {
    const env = createTestBrokerEnv();
    const staleMonitoringBase = await getBrokerAbuseRuntimeState(env.BROKER_DB);
    const monitoringAfter = structuredClone(staleMonitoringBase);
    monitoringAfter.brake.active = true;
    monitoringAfter.brake.reason = 'global_threshold';
    monitoringAfter.brake.changedAt = '2026-04-19T00:05:00.000Z';
    monitoringAfter.brake.changedBy = 'system';
    monitoringAfter.alertLatches.warning = true;

    const reportBefore = await getBrokerAbuseRuntimeState(env.BROKER_DB);
    const reportAfter = structuredClone(reportBefore);
    reportAfter.dailyReport = {
      lastDeliveredAt: '2026-04-19T00:00:00.000Z',
      lastDeliveredDateUtc: '2026-04-19',
    };

    await persistBrokerAbuseRuntimeState(env.BROKER_DB, reportBefore, reportAfter);
    await persistBrokerAbuseRuntimeState(
      env.BROKER_DB,
      staleMonitoringBase,
      monitoringAfter,
    );

    expect(readAbuseRuntimeState(env)).toMatchObject({
      brake: {
        active: true,
        reason: 'global_threshold',
        changedAt: '2026-04-19T00:05:00.000Z',
        changedBy: 'system',
      },
      alertLatches: {
        warning: true,
      },
      dailyReport: {
        lastDeliveredAt: '2026-04-19T00:00:00.000Z',
        lastDeliveredDateUtc: '2026-04-19',
      },
    });
  });

  it('preserves an engaged brake when a stale daily-report writer commits later', async () => {
    const env = createTestBrokerEnv();
    const staleReportBase = await getBrokerAbuseRuntimeState(env.BROKER_DB);
    const reportAfter = structuredClone(staleReportBase);
    reportAfter.dailyReport = {
      lastDeliveredAt: '2026-04-19T00:00:00.000Z',
      lastDeliveredDateUtc: '2026-04-19',
    };

    const monitoringBefore = await getBrokerAbuseRuntimeState(env.BROKER_DB);
    const monitoringAfter = structuredClone(monitoringBefore);
    monitoringAfter.brake.active = true;
    monitoringAfter.brake.reason = 'global_threshold';
    monitoringAfter.brake.changedAt = '2026-04-19T00:05:00.000Z';
    monitoringAfter.brake.changedBy = 'system';
    monitoringAfter.alertLatches.warning = true;

    await persistBrokerAbuseRuntimeState(
      env.BROKER_DB,
      monitoringBefore,
      monitoringAfter,
    );
    await persistBrokerAbuseRuntimeState(env.BROKER_DB, staleReportBase, reportAfter);

    expect(readAbuseRuntimeState(env)).toMatchObject({
      brake: {
        active: true,
        reason: 'global_threshold',
        changedAt: '2026-04-19T00:05:00.000Z',
        changedBy: 'system',
      },
      alertLatches: {
        warning: true,
      },
      dailyReport: {
        lastDeliveredAt: '2026-04-19T00:00:00.000Z',
        lastDeliveredDateUtc: '2026-04-19',
      },
    });
  });

});
