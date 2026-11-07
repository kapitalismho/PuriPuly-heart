import { describe, expect, it } from 'vitest';

import {
  MANAGED_TRIAL_ENTITLEMENT_POLICY,
  MANAGED_TRIAL_LIFECYCLE_VALUES,
  OPENROUTER_ENTITLEMENT_STATUS_VALUES,
  QQ_MANAGED_ENTITLEMENT_STATUS_VALUES,
} from '../src/contract';

describe('managed trial entitlement lifecycle', () => {
  it('freezes the entitlement lifecycle values without inventing extra states', () => {
    expect(MANAGED_TRIAL_LIFECYCLE_VALUES).toEqual([
      'none',
      'pending_release',
      'active',
      'expired',
      'revoked',
    ]);
  });

  it('keeps managed availability as separate reporting from entitlement lifecycle', () => {
    expect(MANAGED_TRIAL_ENTITLEMENT_POLICY.managedAvailability).toEqual({
      field: 'managed_availability',
      reportedSeparatelyFromLifecycle: true,
    });
  });

  it('stores concrete entitlement rows only and treats none as row absence', () => {
    expect(OPENROUTER_ENTITLEMENT_STATUS_VALUES).toEqual([
      'pending_release',
      'active',
      'expired',
      'revoked',
    ]);
  });

  it('includes QQ managed delivery pending as a persisted issuance status', () => {
    expect(QQ_MANAGED_ENTITLEMENT_STATUS_VALUES).toEqual([
      'issuing',
      'delivery_pending',
      'active',
      'cleanup_required',
      'revoked',
    ]);
  });
});
