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

export interface TelemetryUsageRetentionSignal {
  cohort_date_utc: string;
  eligible_users: number;
  retained_users: number;
  retention_pct: number | null;
}

export interface TelemetryUsageDailyMetrics {
  active_users_24h: number;
  active_users_7d: number;
  active_users_30d: number;
  dau_mau_stickiness_pct: number | null;
  first_active_users_24h: number;
  returning_active_users_24h: number;
  retention: {
    d1: TelemetryUsageRetentionSignal;
    d7: TelemetryUsageRetentionSignal;
    d30: TelemetryUsageRetentionSignal;
  };
}

interface TelemetryFirstActiveRow {
  subject_ref: string;
  first_active_date_utc: string;
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

export async function applyTelemetryActiveDayRetention(
  db: D1Database,
  now: Date,
): Promise<TelemetryActiveDayRetentionResult> {
  const cutoffDateUtc = toUtcDateString(addUtcDays(startOfUtcDate(now), -TELEMETRY_ACTIVE_DAY_RETENTION_DAYS));
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
  now: Date,
): Promise<TelemetryUsageDailyMetrics> {
  const reportDateUtc = toUtcDateString(now);
  const sevenDayStartUtc = toUtcDateString(addUtcDays(startOfUtcDate(now), -6));
  const thirtyDayStartUtc = toUtcDateString(addUtcDays(startOfUtcDate(now), -29));

  const [active24hRow, active7dRow, active30dRow, firstActiveRowsResult] = await Promise.all([
    countDistinctActiveSubjects(db, reportDateUtc, reportDateUtc),
    countDistinctActiveSubjects(db, sevenDayStartUtc, reportDateUtc),
    countDistinctActiveSubjects(db, thirtyDayStartUtc, reportDateUtc),
    db
      .prepare(
        `SELECT subject_ref, MIN(active_date_utc) AS first_active_date_utc
           FROM telemetry_active_days
          GROUP BY subject_ref`,
      )
      .all<TelemetryFirstActiveRow>(),
  ]);

  const firstActiveBySubject = new Map(
    firstActiveRowsResult.results.map((row) => [row.subject_ref, row.first_active_date_utc]),
  );
  const activeSubjects24h = await getDistinctActiveSubjects(db, reportDateUtc, reportDateUtc);
  const firstActiveUsers24h = activeSubjects24h.filter(
    (subjectRef) => firstActiveBySubject.get(subjectRef) === reportDateUtc,
  ).length;

  return {
    active_users_24h: active24hRow,
    active_users_7d: active7dRow,
    active_users_30d: active30dRow,
    dau_mau_stickiness_pct: active30dRow === 0 ? null : Math.round((active24hRow / active30dRow) * 100),
    first_active_users_24h: firstActiveUsers24h,
    returning_active_users_24h: activeSubjects24h.length - firstActiveUsers24h,
    retention: {
      d1: await getTelemetryRetentionSignal(db, now, 1),
      d7: await getTelemetryRetentionSignal(db, now, 7),
      d30: await getTelemetryRetentionSignal(db, now, 30),
    },
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

async function getDistinctActiveSubjects(
  db: D1Database,
  startDateUtc: string,
  endDateUtc: string,
): Promise<string[]> {
  const result = await db
    .prepare(
      `SELECT DISTINCT subject_ref
         FROM telemetry_active_days
        WHERE active_date_utc >= ?
          AND active_date_utc <= ?`,
    )
    .bind(startDateUtc, endDateUtc)
    .all<{ subject_ref: string }>();

  return result.results.map((row) => row.subject_ref);
}

async function getTelemetryRetentionSignal(
  db: D1Database,
  now: Date,
  intervalDays: 1 | 7 | 30,
): Promise<TelemetryUsageRetentionSignal> {
  const cohortDateUtc = toUtcDateString(addUtcDays(startOfUtcDate(now), -intervalDays));
  const returnDateUtc = toUtcDateString(now);
  const row = await db
    .prepare(
      `WITH first_active AS (
         SELECT subject_ref, MIN(active_date_utc) AS first_active_date_utc
           FROM telemetry_active_days
          GROUP BY subject_ref
       )
       SELECT
         COUNT(first_active.subject_ref) AS eligible_users,
         COUNT(return_day.subject_ref) AS retained_users
       FROM first_active
       LEFT JOIN telemetry_active_days AS return_day
         ON return_day.subject_ref = first_active.subject_ref
        AND return_day.active_date_utc = ?
       WHERE first_active.first_active_date_utc = ?`,
    )
    .bind(returnDateUtc, cohortDateUtc)
    .first<{ eligible_users: number; retained_users: number }>();
  const eligibleUsers = Number(row?.eligible_users ?? 0);
  const retainedUsers = Number(row?.retained_users ?? 0);

  return {
    cohort_date_utc: cohortDateUtc,
    eligible_users: eligibleUsers,
    retained_users: retainedUsers,
    retention_pct: eligibleUsers === 0 ? null : Math.round((retainedUsers / eligibleUsers) * 100),
  };
}

function startOfUtcDate(date: Date): Date {
  return new Date(Date.UTC(date.getUTCFullYear(), date.getUTCMonth(), date.getUTCDate()));
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
