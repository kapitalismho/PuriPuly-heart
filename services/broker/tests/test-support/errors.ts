interface NormalizedErrorEnvelopeInput {
  code:
    | 'invalid_request'
    | 'rate_limited'
    | 'challenge_expired'
    | 'challenge_invalid'
    | 'issuance_suspended'
    | 'trial_unavailable'
    | 'trial_not_eligible'
    | 'internal_error';
  class: 'retryable' | 'terminal' | 'security_fail';
  message: string;
  subcode?: string | null;
  retryAfterMs?: number | null;
  managedState?: {
    lifecycle: 'none' | 'pending_release' | 'active' | 'expired' | 'revoked';
    managed_availability: boolean;
  };
  currentEntitlement?: {
    provider: 'OpenRouter';
    budget_usd: number;
    issued_at: string | null;
    expires_at: string | null;
  } | null;
}

export function normalizedErrorEnvelope(
  input: NormalizedErrorEnvelopeInput,
): Record<string, unknown> {
  return {
    error: {
      code: input.code,
      class: input.class,
      subcode: input.subcode ?? null,
      retry_after_ms: input.retryAfterMs ?? null,
      message: input.message,
    },
    managed_state: input.managedState ?? {
      lifecycle: 'none',
      managed_availability: true,
    },
    current_entitlement: input.currentEntitlement ?? null,
  };
}
