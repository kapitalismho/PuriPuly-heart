import type {
  ManagedChildKeyCleanupResult,
  ManagedChildKeyCleanupStepResult,
} from './openrouter-management';
import { MANAGED_TRIAL_BUDGET_POLICY } from './trial-policy';

export type ManagedIssueSource = 'discord' | 'qq';

export interface ManagedIssueMetadata {
  issueSource: ManagedIssueSource;
  subjectRef: string;
  issueRef: string;
}

export interface ManagedIssuanceSourcePolicy {
  issue_source: ManagedIssueSource;
  budget_usd: number;
  uses_managed_trial_guardrail: true;
  discord_referral_reservation: boolean;
  owned_referral_id: boolean;
  talk_together_pass: boolean;
  referral_bonus_budget: boolean;
}

export interface ManagedCleanupRequiredAuditPayload {
  event: 'managed_child_key_cleanup_required';
  issue_source: ManagedIssueSource;
  subject_ref: string;
  issue_ref: string;
  managed_credential_ref: string;
  failure: unknown;
  cleanup_outcome: unknown;
  broker_timestamp: string;
}

type ManagedChildKeyCleanupFailureResult = Extract<
  ManagedChildKeyCleanupResult,
  { ok: false }
>;

type ManagedChildKeyCleanupFailureReason =
  ManagedChildKeyCleanupFailureResult['reason'];

export type ManagedCleanupOutcomeForAudit =
  | ManagedChildKeyCleanupFailureResult
  | ManagedChildKeyCleanupFailureReason;

export function getManagedIssuanceSourcePolicy(
  issueSource: ManagedIssueSource,
): ManagedIssuanceSourcePolicy {
  const discordReferralFeatures = issueSource === 'discord';
  return {
    issue_source: issueSource,
    budget_usd: MANAGED_TRIAL_BUDGET_POLICY.hardLimit,
    uses_managed_trial_guardrail: true,
    discord_referral_reservation: discordReferralFeatures,
    owned_referral_id: discordReferralFeatures,
    talk_together_pass: discordReferralFeatures,
    referral_bonus_budget: discordReferralFeatures,
  };
}

export function buildManagedCleanupRequiredAuditPayload(
  input: ManagedIssueMetadata & {
    managedCredentialRef: string;
    failure: unknown;
    cleanupOutcome: ManagedCleanupOutcomeForAudit;
    sensitiveValues?: string[];
    brokerTimestamp?: string;
  },
): ManagedCleanupRequiredAuditPayload {
  return {
    event: 'managed_child_key_cleanup_required',
    issue_source: input.issueSource,
    subject_ref: input.subjectRef,
    issue_ref: input.issueRef,
    managed_credential_ref: input.managedCredentialRef,
    failure: normalizeFailureForAudit(input.failure),
    cleanup_outcome: normalizeCleanupOutcomeForAudit(input.cleanupOutcome),
    broker_timestamp: input.brokerTimestamp ?? new Date().toISOString(),
  };
}

function normalizeFailureForAudit(error: unknown): Record<string, unknown> {
  const managementFailure = normalizeManagementFailureDetails(error);
  if (managementFailure) {
    return error instanceof Error
      ? { name: normalizeErrorName(error.name), ...managementFailure }
      : managementFailure;
  }

  if (error instanceof Error) {
    return {
      name: normalizeErrorName(error.name),
    };
  }

  return {
    name: 'UnknownFailure',
  };
}

function normalizeCleanupOutcomeForAudit(
  value: ManagedCleanupOutcomeForAudit,
): unknown {
  if (isManagedChildKeyCleanupFailureResult(value)) {
    return normalizeCleanupFailureReasonForAudit(value.reason);
  }

  return normalizeCleanupFailureReasonForAudit(value);
}

function normalizeCleanupFailureReasonForAudit(
  value: ManagedChildKeyCleanupFailureReason,
): Record<string, unknown> {
  return {
    disable: normalizeCleanupStepForAudit(value.disable),
    delete: normalizeCleanupStepForAudit(value.delete),
  };
}

function normalizeCleanupStepForAudit(
  value: ManagedChildKeyCleanupStepResult,
): Record<string, unknown> {
  if (value.ok) {
    return { ok: true };
  }

  const error = normalizeManagementFailureDetails(value.error);
  return error ? { ok: false, error } : { ok: false };
}

function isManagedChildKeyCleanupFailureResult(
  value: ManagedCleanupOutcomeForAudit,
): value is ManagedChildKeyCleanupFailureResult {
  return 'ok' in value;
}

function normalizeManagementFailureDetails(value: unknown): Record<string, unknown> | null {
  if (!isRecord(value)) {
    return null;
  }

  const normalized: Record<string, unknown> = {};
  if (isSafeManagementOperation(value.operation)) {
    normalized.operation = value.operation;
  }
  if (isSafeManagementErrorCode(value.code)) {
    normalized.code = value.code;
  }
  if (isNullableFiniteNumber(value.status)) {
    normalized.status = value.status;
  }
  if (isNullableFiniteNumber(value.upstreamCode)) {
    normalized.upstreamCode = value.upstreamCode;
  }
  return Object.keys(normalized).length > 0 ? normalized : null;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null;
}

function normalizeErrorName(name: string): string {
  return ['Error', 'TypeError', 'OpenRouterManagementError'].includes(name)
    ? name
    : 'Error';
}

function isSafeManagementOperation(value: unknown): value is string {
  return [
    'create_key',
    'read_key',
    'update_key_limit',
    'assign_guardrail',
    'disable_key',
    'delete_key',
  ].includes(String(value));
}

function isSafeManagementErrorCode(value: unknown): value is string {
  return ['network_error', 'upstream_http_error', 'malformed_upstream'].includes(
    String(value),
  );
}

function isNullableFiniteNumber(value: unknown): value is number | null {
  return value === null || (typeof value === 'number' && Number.isFinite(value));
}
