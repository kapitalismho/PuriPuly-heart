import type { Context } from 'hono';

import { instrumentPublicPostRoute } from './abuse-controls';
import { errorResponse as publicErrorResponse } from './broker-error';
import type { BrokerEnv } from './contract';
import { finalizeDiscordManagedKeyDeliveryAck } from './discord-managed-issue';
import type { BrokerIssueSuccessSource, ManagedKeyDeliveryRecord } from './persistence';
import { finalizeQqManagedKeyDeliveryAck } from './qq-managed-issue';
import { timingSafeEqualHex } from './network-identity';

const DELIVERY_ID_PREFIX = 'ph-delivery-v1_';
const ACK_TOKEN_HASH_PREFIX = 'ph-delivery-ack-token-v1_';
const DELIVERY_RANDOM_BYTES = 32;
const ACK_TOKEN_RANDOM_BYTES = 32;
export const STALE_DELIVERY_CLEANUP_CLAIM_REASON = 'stale_delivery_cleanup_claimed';
const STALE_DELIVERY_CLEANUP_CLAIM_TTL_MS = 16 * 60_000;

type ManagedKeyDeliveryIssueSource = BrokerIssueSuccessSource;

interface ManagedKeyDeliveryAckRequestBody {
  delivery_id?: unknown;
  managed_credential_ref?: unknown;
  delivery_ack_token?: unknown;
}

export interface CreateManagedKeyDeliveryInput {
  issueSource: ManagedKeyDeliveryIssueSource;
  subjectRef?: string | null;
  installationId?: string | null;
  managedCredentialRef: string;
  createdAt: Date;
  expiresAt: Date;
  operationId?: string | null;
  attemptIndex?: number | null;
}

export interface CreateManagedKeyDeliveryResult {
  deliveryId: string;
  deliveryAckToken: string;
  ackTokenHash: string;
}

export type ManagedKeyDeliveryAckResult =
  | { ok: true; status: 'acknowledged' | 'already_acknowledged' }
  | { ok: false; reason: 'invalid' | 'expired' | 'mismatched' | 'failed' };

type ManagedKeyDeliveryAckValidationResult =
  | { ok: true; status: 'pending' | 'already_acknowledged'; delivery: ManagedKeyDeliveryRecord }
  | { ok: false; reason: 'invalid' | 'expired' | 'mismatched' | 'failed' };

export async function createManagedKeyDelivery(
  db: D1Database,
  input: CreateManagedKeyDeliveryInput,
): Promise<CreateManagedKeyDeliveryResult> {
  const deliveryId = `${DELIVERY_ID_PREFIX}${randomBase64Url(DELIVERY_RANDOM_BYTES)}`;
  const deliveryAckToken = randomBase64Url(ACK_TOKEN_RANDOM_BYTES);
  const ackTokenHash = await hashDeliveryAckToken(deliveryAckToken);

  await db
    .prepare(
      `INSERT INTO managed_key_deliveries (
          delivery_id,
          issue_source,
          subject_ref,
          installation_id,
          managed_credential_ref,
          ack_token_hash,
          status,
          created_at,
          expires_at,
          operation_id,
          attempt_index
        ) VALUES (?, ?, ?, ?, ?, ?, 'pending', ?, ?, ?, ?)`,
    )
    .bind(
      deliveryId,
      input.issueSource,
      input.subjectRef ?? null,
      input.installationId ?? null,
      input.managedCredentialRef,
      ackTokenHash,
      input.createdAt.toISOString(),
      input.expiresAt.toISOString(),
      input.operationId ?? null,
      input.attemptIndex ?? null,
    )
    .run();

  return { deliveryId, deliveryAckToken, ackTokenHash };
}

export async function acknowledgeManagedKeyDelivery(
  db: D1Database,
  input: {
    deliveryId: string;
    managedCredentialRef: string;
    deliveryAckToken: string;
    now: Date;
  },
): Promise<ManagedKeyDeliveryAckResult> {
  const row = await db
    .prepare(
      `SELECT delivery_id, managed_credential_ref, ack_token_hash, status, expires_at
         FROM managed_key_deliveries
        WHERE delivery_id = ?`,
    )
    .bind(input.deliveryId)
    .first<Pick<ManagedKeyDeliveryRecord, 'delivery_id' | 'managed_credential_ref' | 'ack_token_hash' | 'status' | 'expires_at'>>();

  if (!row) {
    return { ok: false, reason: 'invalid' };
  }
  const candidateHash = await hashDeliveryAckToken(input.deliveryAckToken);
  if (!(await timingSafeEqualHex(candidateHash, row.ack_token_hash))) {
    return { ok: false, reason: 'invalid' };
  }

  if (row.managed_credential_ref !== input.managedCredentialRef) {
    return { ok: false, reason: 'mismatched' };
  }

  if (row.status === 'acknowledged') {
    return { ok: true, status: 'already_acknowledged' };
  }

  if (row.status === 'expired') {
    return { ok: false, reason: 'expired' };
  }

  if (row.status === 'cleanup_required') {
    return { ok: false, reason: 'failed' };
  }

  if (input.now.toISOString() > row.expires_at) {
    return { ok: false, reason: 'expired' };
  }

  const acknowledgedAt = input.now.toISOString();
  const result = await db
    .prepare(
      `UPDATE managed_key_deliveries
          SET status = 'acknowledged', acknowledged_at = ?
        WHERE delivery_id = ?
          AND status = 'pending'`,
    )
    .bind(acknowledgedAt, input.deliveryId)
    .run();

  if ((result.meta?.changes ?? 0) === 0) {
    const latest = await getManagedKeyDelivery(db, input.deliveryId);
    if (latest?.status === 'acknowledged') {
      return { ok: true, status: 'already_acknowledged' };
    }
    return { ok: false, reason: 'failed' };
  }

  return { ok: true, status: 'acknowledged' };
}

export async function validateManagedKeyDeliveryAck(
  db: D1Database,
  input: {
    deliveryId: string;
    managedCredentialRef: string;
    deliveryAckToken: string;
    now: Date;
  },
): Promise<ManagedKeyDeliveryAckValidationResult> {
  const row = await getManagedKeyDelivery(db, input.deliveryId);
  if (!row) {
    return { ok: false, reason: 'invalid' };
  }

  const candidateHash = await hashDeliveryAckToken(input.deliveryAckToken);
  if (!(await timingSafeEqualHex(candidateHash, row.ack_token_hash))) {
    return { ok: false, reason: 'invalid' };
  }

  if (row.managed_credential_ref !== input.managedCredentialRef) {
    return { ok: false, reason: 'mismatched' };
  }

  if (row.status === 'acknowledged') {
    return { ok: true, status: 'already_acknowledged', delivery: row };
  }

  if (row.status === 'expired') {
    return { ok: false, reason: 'expired' };
  }

  if (row.status === 'cleanup_required') {
    return { ok: false, reason: 'failed' };
  }

  if (input.now.toISOString() > row.expires_at) {
    return { ok: false, reason: 'expired' };
  }

  return { ok: true, status: 'pending', delivery: row };
}

export async function markManagedKeyDeliveryAcknowledged(
  db: D1Database,
  input: { deliveryId: string; acknowledgedAt: Date },
): Promise<ManagedKeyDeliveryAckResult> {
  const result = await db
    .prepare(
      `UPDATE managed_key_deliveries
          SET status = 'acknowledged', acknowledged_at = ?
        WHERE delivery_id = ?
          AND status = 'pending'`,
    )
    .bind(input.acknowledgedAt.toISOString(), input.deliveryId)
    .run();

  if ((result.meta?.changes ?? 0) === 1) {
    return { ok: true, status: 'acknowledged' };
  }

  const latest = await getManagedKeyDelivery(db, input.deliveryId);
  if (latest?.status === 'acknowledged') {
    return { ok: true, status: 'already_acknowledged' };
  }

  return { ok: false, reason: 'failed' };
}

export async function listStalePendingManagedKeyDeliveries(
  db: D1Database,
  input: { now: Date; limit: number },
): Promise<ManagedKeyDeliveryRecord[]> {
  const result = await db
    .prepare(
      `SELECT delivery_id, issue_source, subject_ref, installation_id, managed_credential_ref,
              ack_token_hash, status, created_at, expires_at, acknowledged_at, failed_at, failure_reason
         FROM managed_key_deliveries
        WHERE (
                status = 'pending'
                AND expires_at <= ?
              )
           OR (
                status = 'expired'
                AND failure_reason = ?
                AND failed_at <= ?
              )
        ORDER BY expires_at ASC
        LIMIT ?`,
    )
    .bind(
      input.now.toISOString(),
      STALE_DELIVERY_CLEANUP_CLAIM_REASON,
      new Date(
        input.now.getTime() - STALE_DELIVERY_CLEANUP_CLAIM_TTL_MS,
      ).toISOString(),
      input.limit,
    )
    .all<ManagedKeyDeliveryRecord>();

  return result.results;
}

export async function claimStaleManagedKeyDeliveryCleanup(
  db: D1Database,
  input: { delivery: ManagedKeyDeliveryRecord; claimedAt: string },
): Promise<boolean> {
  const activeOwnerGuard =
    input.delivery.issue_source === 'discord'
      ? `AND NOT EXISTS (
           SELECT 1
             FROM openrouter_entitlements
            WHERE managed_credential_ref = ?
              AND status = 'active'
              AND discord_issue_status = 'active'
              AND discord_issue_delivered_at IS NOT NULL
         )`
      : `AND NOT EXISTS (
           SELECT 1
             FROM qq_managed_entitlements
            WHERE managed_credential_ref = ?
              AND status = 'active'
              AND delivered_at IS NOT NULL
         )`;
  const statement =
    input.delivery.status === 'pending'
      ? db
          .prepare(
            `UPDATE managed_key_deliveries
                SET status = 'expired', failed_at = ?, failure_reason = ?
              WHERE delivery_id = ?
                AND status = 'pending'
                AND expires_at <= ?
                ${activeOwnerGuard}`,
          )
          .bind(
            input.claimedAt,
            STALE_DELIVERY_CLEANUP_CLAIM_REASON,
            input.delivery.delivery_id,
            input.claimedAt,
            input.delivery.managed_credential_ref,
          )
      : db
          .prepare(
            `UPDATE managed_key_deliveries
                SET failed_at = ?
              WHERE delivery_id = ?
                AND status = 'expired'
                AND failure_reason = ?
                AND failed_at IS ?`,
          )
          .bind(
            input.claimedAt,
            input.delivery.delivery_id,
            STALE_DELIVERY_CLEANUP_CLAIM_REASON,
            input.delivery.failed_at,
          );
  const result = await statement.run();

  return (result.meta?.changes ?? 0) === 1;
}

export async function acknowledgeManagedKeyDeliveryCleanupClaim(
  db: D1Database,
  input: {
    deliveryId: string;
    acknowledgedAt: string;
    expectedClaimedAt: string | null;
  },
): Promise<boolean> {
  const result = await db
    .prepare(
      `UPDATE managed_key_deliveries
          SET status = 'acknowledged',
              acknowledged_at = ?,
              failed_at = NULL,
              failure_reason = NULL
        WHERE delivery_id = ?
          AND status = 'expired'
          AND failure_reason = ?
          AND failed_at IS ?`,
    )
    .bind(
      input.acknowledgedAt,
      input.deliveryId,
      STALE_DELIVERY_CLEANUP_CLAIM_REASON,
      input.expectedClaimedAt,
    )
    .run();
  return (result.meta?.changes ?? 0) === 1;
}

export async function handleManagedKeyDeliveryAck(
  c: Context<BrokerEnv>,
): Promise<Response> {
  const body = await readJsonBody<ManagedKeyDeliveryAckRequestBody>(c);
  if (!body.ok) {
    return ackErrorResponse(c, 400, 'malformed', body.reason);
  }

  const deliveryId = nonEmptyString(body.value.delivery_id);
  const managedCredentialRef = nonEmptyString(body.value.managed_credential_ref);
  const deliveryAckToken = nonEmptyString(body.value.delivery_ack_token);

  if (!deliveryId || !managedCredentialRef || !deliveryAckToken) {
    return ackErrorResponse(
      c,
      400,
      'malformed',
      'delivery_id, managed_credential_ref, and delivery_ack_token are required',
    );
  }
  const ackRateLimit = await instrumentPublicPostRoute(c.env.BROKER_DB, c, {
    endpoint: 'POST /v1/providers/openrouter/managed-key-delivery/ack',
    installationId: null,
  });
  if (ackRateLimit) {
    return ackErrorResponse(c, 429, 'rate_limited', ackRateLimit.message);
  }

  const acknowledgedAt = new Date();
  const validation = await validateManagedKeyDeliveryAck(c.env.BROKER_DB, {
    deliveryId,
    managedCredentialRef,
    deliveryAckToken,
    now: acknowledgedAt,
  });

  if (validation.ok) {
    let ackDetails: Record<string, unknown> = {};
    let status: 'acknowledged' | 'already_acknowledged';
    try {
      if (validation.delivery.issue_source === 'discord') {
        const result = await finalizeDiscordManagedKeyDeliveryAck(c, {
          deliveryId,
          managedCredentialRef,
          acknowledgedAt,
        });
        status = result.acknowledgementStatus;
        if (result.referralBonusApplied) {
          ackDetails = { referral_bonus_applied: true };
        }
      } else {
        const result = await finalizeQqManagedKeyDeliveryAck(c, {
          deliveryId,
          managedCredentialRef,
          acknowledgedAt,
        });
        status = result.acknowledgementStatus;
        ackDetails = {
          ...(result.referralId ? { referral_id: result.referralId } : {}),
          ...(result.talkTogetherPass
            ? { talk_together_pass: result.talkTogetherPass }
            : {}),
        };
      }
    } catch {
      return ackErrorResponse(c, 409, 'failed', 'delivery acknowledgement cannot be applied');
    }
    return c.json({ ok: true, status, ...ackDetails });
  }

  if (validation.reason === 'expired') {
    return ackErrorResponse(c, 410, 'expired', 'delivery acknowledgement expired');
  }

  if (validation.reason === 'mismatched') {
    return ackErrorResponse(c, 409, 'mismatched', 'delivery acknowledgement did not match credential');
  }

  if (validation.reason === 'failed') {
    return ackErrorResponse(c, 409, 'failed', 'delivery acknowledgement cannot be applied');
  }

  return ackErrorResponse(c, 404, 'invalid', 'delivery acknowledgement is invalid');
}

export async function hashDeliveryAckToken(deliveryAckToken: string): Promise<string> {
  const digest = await crypto.subtle.digest(
    'SHA-256',
    new TextEncoder().encode(deliveryAckToken),
  );
  return `${ACK_TOKEN_HASH_PREFIX}${toHex(new Uint8Array(digest))}`;
}

async function getManagedKeyDelivery(
  db: D1Database,
  deliveryId: string,
): Promise<ManagedKeyDeliveryRecord | null> {
  return db
    .prepare(
      `SELECT delivery_id, issue_source, subject_ref, installation_id, managed_credential_ref,
              ack_token_hash, status, created_at, expires_at, acknowledged_at, failed_at, failure_reason
         FROM managed_key_deliveries
        WHERE delivery_id = ?`,
    )
    .bind(deliveryId)
    .first<ManagedKeyDeliveryRecord>();
}

async function readJsonBody<T>(
  c: Context<BrokerEnv>,
): Promise<
  | { ok: true; value: T }
  | { ok: false; reason: 'request body must be valid JSON' | 'request body must be a JSON object' }
> {
  try {
    const value = await c.req.json();
    if (typeof value !== 'object' || value === null || Array.isArray(value)) {
      return { ok: false, reason: 'request body must be a JSON object' };
    }

    return { ok: true, value: value as T };
  } catch {
    return { ok: false, reason: 'request body must be valid JSON' };
  }
}

function ackErrorResponse(
  c: Context<BrokerEnv>,
  status: 400 | 404 | 409 | 410 | 429,
  subcode: 'malformed' | 'invalid' | 'expired' | 'mismatched' | 'failed' | 'rate_limited',
  message: string,
): Response {
  return publicErrorResponse(c, status, {
    code: subcode === 'rate_limited' ? 'rate_limited' : 'invalid_request',
    class: subcode === 'failed' || subcode === 'rate_limited' ? 'retryable' : 'terminal',
    subcode: `delivery_ack_${subcode}`,
    message,
  });
}

function nonEmptyString(value: unknown): string | null {
  return typeof value === 'string' && value.length > 0 ? value : null;
}

function randomBase64Url(byteLength: number): string {
  const bytes = new Uint8Array(byteLength);
  crypto.getRandomValues(bytes);
  let binary = '';
  for (const byte of bytes) {
    binary += String.fromCharCode(byte);
  }
  return btoa(binary).replaceAll('+', '-').replaceAll('/', '_').replaceAll('=', '');
}

function toHex(bytes: Uint8Array): string {
  return Array.from(bytes, (byte) => byte.toString(16).padStart(2, '0')).join('');
}
