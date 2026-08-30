import type { Context } from 'hono';

import { errorResponse as publicErrorResponse } from './broker-error';
import type { BrokerEnv } from './contract';

export const APP_ACTIVE_DAY_ENDPOINT = 'POST /v1/telemetry/app-active-day';
export const APP_SUBJECT_REF_PREFIX = 'ph-app-subject-v1_';

const ANONYMOUS_ID_PATTERN = /^[A-Za-z0-9_-]{16,128}$/u;
const ACTIVE_DATE_PATTERN = /^\d{4}-\d{2}-\d{2}$/u;
const APP_ACTIVE_DAY_RETENTION_DAYS = 35;

interface AppActiveDayRequestBody {
  anonymous_id?: unknown;
  active_date_utc?: unknown;
}

interface AppActiveDayRecordInput {
  subjectRef: string;
  activeDateUtc: string;
}

export interface AppActiveDayRetentionResult {
  deleted: number;
  cutoffDateUtc: string;
}

export interface AppUsageDailyMetrics {
  app_dau: number;
  app_wau: number;
  app_mau: number;
}

interface CountRow {
  count: number;
}

export async function handleAppActiveDay(c: Context<BrokerEnv>): Promise<Response> {
  const now = new Date();
  const body = await readJsonBody<AppActiveDayRequestBody>(c);
  if (!body.ok) {
    return invalidTelemetryRequest(c, body.reason);
  }

  const validation = validateAppActiveDayRequest(body.value, now);
  if (!validation.ok) {
    return invalidTelemetryRequest(c, validation.reason);
  }

  const subjectRef = await deriveAppSubjectRef(
    c.env.TELEMETRY_SUBJECT_HMAC_SECRET,
    validation.anonymousId,
  );
  await recordAppActiveDay(c.env.BROKER_DB, {
    subjectRef,
    activeDateUtc: validation.activeDateUtc,
  });

  return c.json({ ok: true });
}

export async function recordAppActiveDay(
  db: D1Database,
  input: AppActiveDayRecordInput,
): Promise<void> {
  await db
    .prepare(
      `INSERT INTO app_active_days (subject_ref, active_date_utc)
       VALUES (?, ?)
       ON CONFLICT(subject_ref, active_date_utc) DO NOTHING`,
    )
    .bind(input.subjectRef, input.activeDateUtc)
    .run();
}

export async function applyAppActiveDayRetention(
  db: D1Database,
  now: Date,
): Promise<AppActiveDayRetentionResult> {
  const cutoffDateUtc = toUtcDateString(
    addUtcDays(startOfUtcDate(now), -APP_ACTIVE_DAY_RETENTION_DAYS),
  );
  const result = await db
    .prepare(
      `DELETE FROM app_active_days
        WHERE active_date_utc < ?`,
    )
    .bind(cutoffDateUtc)
    .run();

  return {
    deleted: result.meta?.changes ?? 0,
    cutoffDateUtc,
  };
}

export async function getAppUsageDailyMetrics(
  db: D1Database,
  reportDateUtc: string,
): Promise<AppUsageDailyMetrics> {
  if (!isValidUtcDate(reportDateUtc)) {
    throw new Error('invalid app usage report date');
  }
  const reportDate = new Date(`${reportDateUtc}T00:00:00.000Z`);
  const sevenDayStartUtc = toUtcDateString(addUtcDays(reportDate, -6));
  const thirtyDayStartUtc = toUtcDateString(addUtcDays(reportDate, -29));

  const [appDau, appWau, appMau] = await Promise.all([
    countDistinctActiveSubjects(db, reportDateUtc, reportDateUtc),
    countDistinctActiveSubjects(db, sevenDayStartUtc, reportDateUtc),
    countDistinctActiveSubjects(db, thirtyDayStartUtc, reportDateUtc),
  ]);

  return {
    app_dau: appDau,
    app_wau: appWau,
    app_mau: appMau,
  };
}

export async function deriveAppSubjectRef(
  hmacSecret: string,
  anonymousId: string,
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
    encoder.encode(anonymousId),
  );
  return `${APP_SUBJECT_REF_PREFIX}${toHex(signature)}`;
}

function validateAppActiveDayRequest(
  request: AppActiveDayRequestBody,
  now: Date,
):
  | { ok: true; anonymousId: string; activeDateUtc: string }
  | { ok: false; reason: string } {
  const keys = Object.keys(request);
  if (
    keys.length !== 2 ||
    !keys.includes('anonymous_id') ||
    !keys.includes('active_date_utc')
  ) {
    return { ok: false, reason: 'telemetry request has unsupported shape' };
  }

  if (
    typeof request.anonymous_id !== 'string' ||
    !ANONYMOUS_ID_PATTERN.test(request.anonymous_id)
  ) {
    return { ok: false, reason: 'invalid anonymous identifier' };
  }

  if (
    typeof request.active_date_utc !== 'string' ||
    !isValidUtcDate(request.active_date_utc)
  ) {
    return { ok: false, reason: 'invalid active date' };
  }

  const todayUtc = toUtcDateString(startOfUtcDate(now));
  const previousDateUtc = toUtcDateString(addUtcDays(startOfUtcDate(now), -1));
  if (
    request.active_date_utc !== todayUtc &&
    request.active_date_utc !== previousDateUtc
  ) {
    return { ok: false, reason: 'active date must be today or the previous UTC date' };
  }

  return {
    ok: true,
    anonymousId: request.anonymous_id,
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
         FROM app_active_days
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
