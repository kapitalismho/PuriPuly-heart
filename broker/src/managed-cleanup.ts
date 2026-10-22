export const MANAGED_CLEANUP_RETRY_MS = 6 * 60 * 60_000;
export const MANAGED_CLEANUP_CLAIM_TTL_MS = 16 * 60_000;
export const SCHEDULED_MANAGED_CLEANUP_LIMIT = 6;
export const SCHEDULED_MANAGED_DELIVERY_LIMIT = 2;

export interface ManagedCleanupBudget {
  remainingCandidates: number;
  remainingProviderPages: number;
}

export function createManagedCleanupBudget(): ManagedCleanupBudget {
  return {
    remainingCandidates: SCHEDULED_MANAGED_CLEANUP_LIMIT,
    remainingProviderPages: SCHEDULED_MANAGED_CLEANUP_LIMIT,
  };
}
