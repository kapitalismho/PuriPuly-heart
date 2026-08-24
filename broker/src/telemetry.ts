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

export interface TelemetryActiveDayRetentionResult {
  deleted: number;
  cutoffDateUtc: string;
}

export interface TelemetryUsageDailyMetrics {
  translated_dau: number;
  translated_wau: number;
  translated_mau: number;
  first_observed_translators: number;
  returning_translators: number;
}

interface CountRow {
  count: number;
}

const TELEMETRY_ACTIVE_DAY_RETENTION_DAYS = 400;

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
  await db.batch([
    db
      .prepare(
        `INSERT INTO telemetry_subjects (
            subject_ref,
            first_active_date_utc,
            last_active_date_utc
          ) VALUES (?, ?, ?)
          ON CONFLICT(subject_ref) DO UPDATE SET
            first_active_date_utc = MIN(first_active_date_utc, excluded.first_active_date_utc),
            last_active_date_utc = MAX(last_active_date_utc, excluded.last_active_date_utc)`,
      )
      .bind(input.subjectRef, input.activeDateUtc, input.activeDateUtc),
    db
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
      ),
  ]);
}

export async function applyTelemetryActiveDayRetention(
  db: D1Database,
  now: Date,
): Promise<TelemetryActiveDayRetentionResult> {
  const cutoffDateUtc = toUtcDateString(
    addUtcDays(startOfUtcDate(now), -TELEMETRY_ACTIVE_DAY_RETENTION_DAYS),
  );
  const result = await db
    .prepare(
      `DELETE FROM telemetry_active_days
        WHERE active_date_utc < ?`,
    )
    .bind(cutoffDateUtc)
    .run();

  return {
    deleted: result.meta?.changes ?? 0,
    cutoffDateUtc,
  };
}

export async function getTelemetryUsageDailyMetrics(
  db: D1Database,
  reportDateUtc: string,
): Promise<TelemetryUsageDailyMetrics> {
  if (!isValidUtcDate(reportDateUtc)) {
    throw new Error('invalid telemetry report date');
  }
  const reportDate = new Date(`${reportDateUtc}T00:00:00.000Z`);
  const sevenDayStartUtc = toUtcDateString(addUtcDays(reportDate, -6));
  const thirtyDayStartUtc = toUtcDateString(addUtcDays(reportDate, -29));

  const [translatedDau, translatedWau, translatedMau, classificationRow] =
    await Promise.all([
      countDistinctActiveSubjects(db, reportDateUtc, reportDateUtc),
      countDistinctActiveSubjects(db, sevenDayStartUtc, reportDateUtc),
      countDistinctActiveSubjects(db, thirtyDayStartUtc, reportDateUtc),
      db
        .prepare(
          `SELECT
             COALESCE(SUM(CASE WHEN subjects.first_active_date_utc = ? THEN 1 ELSE 0 END), 0) AS first_observed,
             COALESCE(SUM(CASE WHEN subjects.first_active_date_utc < ? THEN 1 ELSE 0 END), 0) AS returning_count
           FROM telemetry_active_days AS active_day
           JOIN telemetry_subjects AS subjects
             ON subjects.subject_ref = active_day.subject_ref
          WHERE active_day.active_date_utc = ?`,
        )
        .bind(reportDateUtc, reportDateUtc, reportDateUtc)
        .first<{ first_observed: number; returning_count: number }>(),
    ]);

  return {
    translated_dau: translatedDau,
    translated_wau: translatedWau,
    translated_mau: translatedMau,
    first_observed_translators: Number(classificationRow?.first_observed ?? 0),
    returning_translators: Number(classificationRow?.returning_count ?? 0),
  };
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

async function countDistinctActiveSubjects(
  db: D1Database,
  startDateUtc: string,
  endDateUtc: string,
): Promise<number> {
  const row = await db
    .prepare(
      `SELECT COUNT(DISTINCT subject_ref) AS count
         FROM telemetry_active_days
        WHERE active_date_utc >= ?
          AND active_date_utc <= ?`,
    )
    .bind(startDateUtc, endDateUtc)
    .first<CountRow>();

  return Number(row?.count ?? 0);
}

function startOfUtcDate(date: Date): Date {
  return new Date(
    Date.UTC(date.getUTCFullYear(), date.getUTCMonth(), date.getUTCDate()),
  );
}

function addUtcDays(date: Date, days: number): Date {
  return new Date(date.getTime() + days * 24 * 60 * 60_000);
}

function toUtcDateString(date: Date): string {
  return date.toISOString().slice(0, 10);
}

function toHex(buffer: ArrayBuffer): string {
  return Array.from(new Uint8Array(buffer), (byte) =>
    byte.toString(16).padStart(2, '0'),
  ).join('');
}
