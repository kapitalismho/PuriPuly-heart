import type {
  OpenRouterEntitlementRecord,
  OpenRouterEntitlementStatus,
} from './persistence';
import { TRIAL_PROVIDER_POLICY } from './trial-policy';

export interface ManagedStateResponse {
  managed_state: {
    lifecycle:
      | 'none'
      | 'pending_release'
      | 'active'
      | 'expired'
      | 'revoked';
    managed_availability: boolean;
  };
  current_entitlement:
    | {
        provider: string;
        budget_usd: number;
        issued_at: string | null;
        expires_at: string | null;
      }
    | null;
}


export interface TrialStatusResponse extends ManagedStateResponse {
  onboarding_eligibility: {
    eligible: boolean;
    reason: 'discord_required' | OpenRouterEntitlementStatus;
    requires_discord_oauth: boolean;
  };
}

export function resolveEffectiveEntitlementLifecycle(
  entitlement: OpenRouterEntitlementRecord | null,
  now: Date = new Date(),
): ManagedStateResponse['managed_state']['lifecycle'] {
  if (!entitlement) {
    return 'none';
  }

  if (entitlement.status !== 'active' || !entitlement.expires_at) {
    return entitlement.status;
  }

  const expiresAt = new Date(entitlement.expires_at);
  if (Number.isNaN(expiresAt.getTime())) {
    return entitlement.status;
  }

  return expiresAt.getTime() < now.getTime() ? 'expired' : entitlement.status;
}

export function normalizeManagedState(
  entitlement: OpenRouterEntitlementRecord | null,
): ManagedStateResponse {
  const lifecycle = resolveEffectiveEntitlementLifecycle(entitlement);

  return {
    managed_state: {
      lifecycle,
      managed_availability:
        lifecycle === 'none' ||
        lifecycle === 'pending_release' ||
        lifecycle === 'active',
    },
    current_entitlement: entitlement
      ? {
          provider: TRIAL_PROVIDER_POLICY.managedFreeTrial.provider,
          budget_usd: entitlement.budget_usd,
          issued_at: entitlement.issued_at,
          expires_at: entitlement.expires_at,
        }
      : null,
  };
}

export function normalizeTrialStatusResponse(
  entitlement: OpenRouterEntitlementRecord | null,
): TrialStatusResponse {
  const managedState = normalizeManagedState(entitlement);

  return {
    ...managedState,
    onboarding_eligibility: {
      eligible: entitlement === null,
      reason: entitlement === null ? 'discord_required' : entitlement.status,
      requires_discord_oauth: entitlement === null,
    },
  };
}
