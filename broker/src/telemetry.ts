import type { Context } from 'hono';

import {
  checkEndpointRateLimit,
  recordRequestEvent,
  resolveClientIp,
} from './abuse-controls';
import { errorResponse as publicErrorResponse } from './broker-error';
import type { BrokerEnv } from './contract';

export const TELEMETRY_TRANSLATION_SUCCESS_DAY_ENDPOINT =
  'POST /v1/telemetry/translation-success-day';
export const TELEMETRY_SIGNAL_KIND = 'translation_success_day';
export const TELEMETRY_SUBJECT_REF_PREFIX = 'ph-telemetry-subject-v1_';

const TELEMETRY_IDENTIFIER_PATTERN = /^[A-Za-z0-9_-]{16,128}$/u;
const ACTIVE_DATE_PATTERN = /^\d{4}-\d{2}-\d{2}$/u;

interface TelemetryTranslationSuccessDayRequestBody {
  signal?: unknown;
  telemetry_identifier?: unknown;
  active_date_utc?: unknown;
}

interface TelemetryActiveDayRecordInput {
  subjectRef: string;
  activeDateUtc: string;
  receivedAt: string;
}

export async function handleTelemetryTranslationSuccessDay(
  c: Context<BrokerEnv>,
): Promise<Response> {
  const now = new Date();
  const requestContext = {
    endpoint: TELEMETRY_TRANSLATION_SUCCESS_DAY_ENDPOINT,
    ip: resolveClientIp(c),
    installationId: null,
    hardwareHash: null,
    now,
  };
  await recordRequestEvent(c.env.BROKER_DB, requestContext);

  const rateLimitDecision = await checkEndpointRateLimit(
    c.env.BROKER_DB,
    requestContext,
  );
  if (rateLimitDecision) {
    return publicErrorResponse(c, rateLimitDecision.status, {
      code: rateLimitDecision.code,
      class: rateLimitDecision.class,
      subcode: rateLimitDecision.subcode,
      retryAfterMs: rateLimitDecision.retryAfterMs,
      message: rateLimitDecision.message,
    });
  }

  const body = await readJsonBody<TelemetryTranslationSuccessDayRequestBody>(c);
  if (!body.ok) {
    return invalidTelemetryRequest(c, body.reason);
  }

  const validation = validateTelemetryTranslationSuccessDayRequest(body.value);
  if (!validation.ok) {
    return invalidTelemetryRequest(c, validation.reason);
  }

  const subjectRef = await deriveTelemetrySubjectRef(
    c.env.TELEMETRY_SUBJECT_HMAC_SECRET,
    validation.telemetryIdentifier,
  );

  await recordTelemetryActiveDay(c.env.BROKER_DB, {
    subjectRef,
    activeDateUtc: validation.activeDateUtc,
    receivedAt: now.toISOString(),
  });

  return c.json({ ok: true });
}

export async function recordTelemetryActiveDay(
  db: D1Database,
  input: TelemetryActiveDayRecordInput,
): Promise<void> {
  await db
    .prepare(
      `INSERT INTO telemetry_active_days (
          subject_ref,
          active_date_utc,
          first_received_at,
          last_received_at
        ) VALUES (?, ?, ?, ?)
        ON CONFLICT(subject_ref, active_date_utc) DO UPDATE SET
          last_received_at = excluded.last_received_at`,
    )
    .bind(
      input.subjectRef,
      input.activeDateUtc,
      input.receivedAt,
      input.receivedAt,
    )
    .run();
}

export async function deriveTelemetrySubjectRef(
  hmacSecret: string,
  telemetryIdentifier: string,
): Promise<string> {
  const encoder = new TextEncoder();
  const key = await crypto.subtle.importKey(
    'raw',
    encoder.encode(hmacSecret),
    { name: 'HMAC', hash: 'SHA-256' },
    false,
    ['sign'],
  );
  const signature = await crypto.subtle.sign(
    'HMAC',
    key,
    encoder.encode(telemetryIdentifier),
  );
  return `${TELEMETRY_SUBJECT_REF_PREFIX}${toHex(signature)}`;
}

function validateTelemetryTranslationSuccessDayRequest(
  request: TelemetryTranslationSuccessDayRequestBody,
):
  | { ok: true; telemetryIdentifier: string; activeDateUtc: string }
  | { ok: false; reason: string } {
  const keys = Object.keys(request);
  if (
    keys.length !== 3 ||
    !keys.includes('signal') ||
    !keys.includes('telemetry_identifier') ||
    !keys.includes('active_date_utc')
  ) {
    return { ok: false, reason: 'telemetry request has unsupported shape' };
  }

  if (request.signal !== TELEMETRY_SIGNAL_KIND) {
    return { ok: false, reason: 'unsupported telemetry signal' };
  }

  if (
    typeof request.telemetry_identifier !== 'string' ||
    !TELEMETRY_IDENTIFIER_PATTERN.test(request.telemetry_identifier)
  ) {
    return { ok: false, reason: 'invalid telemetry identifier' };
  }

  if (
    typeof request.active_date_utc !== 'string' ||
    !isValidUtcDate(request.active_date_utc)
  ) {
    return { ok: false, reason: 'invalid active date' };
  }

  return {
    ok: true,
    telemetryIdentifier: request.telemetry_identifier,
    activeDateUtc: request.active_date_utc,
  };
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

function invalidTelemetryRequest(c: Context<BrokerEnv>, reason: string): Response {
  return publicErrorResponse(c, 400, {
    code: 'invalid_request',
    class: 'terminal',
    subcode: null,
    message: reason,
  });
}

function isValidUtcDate(value: string): boolean {
  if (!ACTIVE_DATE_PATTERN.test(value)) {
    return false;
  }

  const [yearPart, monthPart, dayPart] = value.split('-');
  const year = Number(yearPart);
  const month = Number(monthPart);
  const day = Number(dayPart);
  const date = new Date(Date.UTC(year, month - 1, day));

  return (
    date.getUTCFullYear() === year &&
    date.getUTCMonth() === month - 1 &&
    date.getUTCDate() === day
  );
}

function toHex(buffer: ArrayBuffer): string {
  return Array.from(new Uint8Array(buffer), (byte) =>
    byte.toString(16).padStart(2, '0'),
  ).join('');
}
