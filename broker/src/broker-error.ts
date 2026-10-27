import type { Context } from 'hono';

import type { BrokerEnv } from './contract';
import type { OpenRouterEntitlementRecord } from './persistence';
import { normalizeManagedState } from './managed-state';

export type PublicErrorCode =
  | 'invalid_request'
  | 'rate_limited'
  | 'challenge_expired'
  | 'challenge_invalid'
  | 'issuance_suspended'
  | 'trial_unavailable'
  | 'trial_not_eligible'
  | 'internal_error';

export type PublicErrorClass = 'retryable' | 'terminal' | 'security_fail';

export interface PublicErrorResponseOptions {
  code: PublicErrorCode;
  class: PublicErrorClass;
  message: string;
  subcode?: string | null;
  retryAfterMs?: number | null;
  entitlement?: OpenRouterEntitlementRecord | null;
}


export function errorResponse(
  c: Context<BrokerEnv>,
  status: 400 | 401 | 404 | 409 | 410 | 429 | 500 | 503,
  options: PublicErrorResponseOptions,
): Response {
  return c.json(
    {
      error: {
        code: options.code,
        class: options.class,
        subcode: options.subcode ?? null,
        retry_after_ms: options.retryAfterMs ?? null,
        message: options.message,
      },
      ...normalizeManagedState(options.entitlement ?? null),
    },
    status,
  );
}

export function internalErrorResponse(c: Context<BrokerEnv>): Response {
  return internalErrorResponseWithEntitlement(c, null);
}

export function internalErrorResponseWithEntitlement(
  c: Context<BrokerEnv>,
  entitlement: OpenRouterEntitlementRecord | null,
): Response {
  return errorResponse(c, 500, {
    code: 'internal_error',
    class: 'retryable',
    message: 'broker encountered an unexpected internal error',
    entitlement,
  });
}

