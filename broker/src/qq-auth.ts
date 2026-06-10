import type { Context } from 'hono';

import {
  checkEndpointRateLimit,
  recordRequestEvent,
  resolveClientIp,
} from './abuse-controls';
import {
  errorResponse as publicErrorResponse,
  internalErrorResponse,
} from './broker-error';
import type { BrokerEnv } from './contract';
import { stringValue } from './public-input';
import { issueQqManagedEntitlement } from './qq-managed-issue';

const QQ_AUTH_ASSERT_ENDPOINT = 'POST /v1/auth/qq/assert';
const QQ_SUBJECT_REF_PREFIX = 'ph-qq-subject-v1_';
const QQ_SUBJECT_REF_PAYLOAD_PREFIX = 'puripuly-heart:qq-subject:v1';
const CREDENTIAL_HASH_PREFIX = 'sha256-base64url-v1_';
const QQ_IDENTITY_MAX_LENGTH = 2048;
const ASSERTED_AT_MAX_LENGTH = 64;
const QQ_CREDENTIAL_PATTERN = /^[0-9a-f]{64}$/u;
const STRICT_ISO_8601_TIMESTAMP =
  /^(?<year>\d{4})-(?<month>0[1-9]|1[0-2])-(?<day>0[1-9]|[12]\d|3[01])T(?<hour>[01]\d|2[0-3]):(?<minute>[0-5]\d):(?<second>[0-5]\d)(?:\.(?<millisecond>\d{3}))?(?:(?<utc>Z)|(?<offsetSign>[+-])(?<offsetHour>[01]\d|2[0-3]):(?<offsetMinute>[0-5]\d))$/u;
const CONTROL_OR_NEWLINE_PATTERN = /[\p{Cc}\r\n\u0085\u2028\u2029]/u;
const textEncoder = new TextEncoder();

interface QqAuthAssertRequestBody {
  qq_identity?: unknown;
  credential?: unknown;
  asserted_at?: unknown;
}

interface QqAuthAssertInput {
  qqIdentity: string;
  credential: string;
  assertedAt: string;
}

export async function handleQqAuthAssert(
  c: Context<BrokerEnv>,
): Promise<Response> {
  const now = new Date();
  const requestContext = {
    endpoint: QQ_AUTH_ASSERT_ENDPOINT,
    now,
    ip: resolveClientIp(c),
    installationId: null,
    hardwareHash: null,
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
      entitlement: null,
    });
  }

  const body = await readJsonBody<QqAuthAssertRequestBody>(c);
  if (!body.ok) {
    return invalidRequestBodyResponse(c, body.reason);
  }

  const input = validateQqAuthAssertInput(body.value);
  if (!input.ok) {
    return invalidRequestResponse(c, input.message);
  }

  const hmacPsk = stringValue(c.env.QQ_AUTH_HMAC_PSK);
  if (!hmacPsk || hmacPsk.trim().length === 0) {
    return internalErrorResponse(c);
  }

  const expectedCredential = await hmacSha256Hex(hmacPsk, input.value.qqIdentity);
  if (!constantTimeEqual(input.value.credential, expectedCredential)) {
    return invalidQqCredentialResponse(c);
  }

  const qqSubjectRef = `${QQ_SUBJECT_REF_PREFIX}${encodeBase64Url(
    await hmacSha256Bytes(
      hmacPsk,
      `${QQ_SUBJECT_REF_PAYLOAD_PREFIX}\n${input.value.qqIdentity}`,
    ),
  )}`;
  const credentialHash = `${CREDENTIAL_HASH_PREFIX}${await sha256Base64Url(
    input.value.credential,
  )}`;

  const insertResult = await c.env.BROKER_DB.prepare(
    `INSERT INTO qq_auth_assertions (
        qq_subject_ref,
        credential_hash,
        asserted_at,
        status
      ) VALUES (?, ?, ?, 'verified')
      ON CONFLICT(qq_subject_ref) DO NOTHING`,
  )
    .bind(qqSubjectRef, credentialHash, input.value.assertedAt)
    .run();

  if (!isQqIssuanceRuntimeEnabled(c.env)) {
    return legacyVerificationResponse(c, {
      insertResult,
      qqSubjectRef,
    });
  }

  return issueQqManagedEntitlement(c, {
    qqSubjectRef,
    now,
  });
}

function isQqIssuanceRuntimeEnabled(env: BrokerEnv['Bindings']): boolean {
  return (
    isPresentRuntimeSecret(env.OPENROUTER_MANAGEMENT_API_KEY) &&
    isPresentRuntimeSecret(env.OPENROUTER_MANAGED_GUARDRAIL_ID)
  );
}

function isPresentRuntimeSecret(value: unknown): boolean {
  return stringValue(value)?.trim().length ? true : false;
}

function legacyVerificationResponse(
  c: Context<BrokerEnv>,
  input: { insertResult: D1Result; qqSubjectRef: string },
): Response {
  return c.json({
    ok: true,
    status:
      Number(input.insertResult.meta.changes ?? 0) > 0
        ? 'verified'
        : 'already_verified',
    qq_subject_ref: input.qqSubjectRef,
  });
}

function validateQqAuthAssertInput(
  body: QqAuthAssertRequestBody,
):
  | { ok: true; value: QqAuthAssertInput }
  | { ok: false; message: string } {
  const qqIdentity = stringValue(body.qq_identity);
  const credential = stringValue(body.credential);
  const assertedAt = stringValue(body.asserted_at);

  if (qqIdentity === null || credential === null || assertedAt === null) {
    return {
      ok: false,
      message: 'qq_identity, credential, and asserted_at are required',
    };
  }

  const qqIdentityError = validateQqIdentity(qqIdentity);
  if (qqIdentityError) {
    return { ok: false, message: qqIdentityError };
  }

  if (!QQ_CREDENTIAL_PATTERN.test(credential)) {
    return {
      ok: false,
      message: 'credential must be exactly 64 lowercase hexadecimal characters',
    };
  }

  const assertedAtDate = parseStrictIsoDate(assertedAt);
  if (!assertedAtDate) {
    return {
      ok: false,
      message: 'asserted_at must be a valid ISO-8601 timestamp',
    };
  }

  return {
    ok: true,
    value: {
      qqIdentity,
      credential,
      assertedAt: assertedAtDate.toISOString(),
    },
  };
}

function validateQqIdentity(value: string): string | null {
  const characterCount = Array.from(value).length;
  if (
    value.trim().length === 0 ||
    characterCount < 1 ||
    characterCount > QQ_IDENTITY_MAX_LENGTH
  ) {
    return `qq_identity must be between 1 and ${QQ_IDENTITY_MAX_LENGTH} characters`;
  }

  if (CONTROL_OR_NEWLINE_PATTERN.test(value)) {
    return 'qq_identity must not contain control characters or newlines';
  }

  return null;
}

function parseStrictIsoDate(value: string): Date | null {
  if (
    value.trim().length === 0 ||
    Array.from(value).length > ASSERTED_AT_MAX_LENGTH
  ) {
    return null;
  }

  const match = STRICT_ISO_8601_TIMESTAMP.exec(value);
  if (!match?.groups) {
    return null;
  }

  const year = Number(match.groups.year);
  const month = Number(match.groups.month);
  const day = Number(match.groups.day);
  const hour = Number(match.groups.hour);
  const minute = Number(match.groups.minute);
  const second = Number(match.groups.second);
  const millisecond = Number(match.groups.millisecond ?? '0');
  const offsetMinutes = match.groups.utc
    ? 0
    : (match.groups.offsetSign === '-' ? -1 : 1) *
      (Number(match.groups.offsetHour) * 60 + Number(match.groups.offsetMinute));

  const timestamp =
    Date.UTC(year, month - 1, day, hour, minute, second, millisecond) -
    offsetMinutes * 60_000;
  const reconstructedLocalTime = new Date(timestamp + offsetMinutes * 60_000);

  if (
    reconstructedLocalTime.getUTCFullYear() !== year ||
    reconstructedLocalTime.getUTCMonth() + 1 !== month ||
    reconstructedLocalTime.getUTCDate() !== day ||
    reconstructedLocalTime.getUTCHours() !== hour ||
    reconstructedLocalTime.getUTCMinutes() !== minute ||
    reconstructedLocalTime.getUTCSeconds() !== second ||
    reconstructedLocalTime.getUTCMilliseconds() !== millisecond
  ) {
    return null;
  }

  return new Date(timestamp);
}

async function readJsonBody<T>(
  c: Context<BrokerEnv>,
): Promise<
  | { ok: true; value: T }
  | { ok: false; reason: 'invalid_json' | 'not_object' }
> {
  try {
    const value = await c.req.json();
    if (!isJsonObject(value)) {
      return { ok: false, reason: 'not_object' };
    }

    return { ok: true, value: value as T };
  } catch {
    return { ok: false, reason: 'invalid_json' };
  }
}

function isJsonObject(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function invalidRequestBodyResponse(
  c: Context<BrokerEnv>,
  reason: 'invalid_json' | 'not_object',
): Response {
  return invalidRequestResponse(
    c,
    reason === 'invalid_json'
      ? 'request body must be valid JSON'
      : 'request body must be a JSON object',
  );
}

function invalidRequestResponse(c: Context<BrokerEnv>, message: string): Response {
  return publicErrorResponse(c, 400, {
    code: 'invalid_request',
    class: 'terminal',
    message,
    entitlement: null,
  });
}

function invalidQqCredentialResponse(c: Context<BrokerEnv>): Response {
  return publicErrorResponse(c, 401, {
    code: 'invalid_request',
    class: 'security_fail',
    subcode: 'qq_credential_invalid',
    message: 'QQ assertion credential is invalid',
    entitlement: null,
  });
}

async function hmacSha256Hex(secret: string, value: string): Promise<string> {
  const bytes = await hmacSha256Bytes(secret, value);
  return Array.from(bytes, (byte) => byte.toString(16).padStart(2, '0')).join('');
}

async function hmacSha256Bytes(secret: string, value: string): Promise<Uint8Array> {
  const key = await crypto.subtle.importKey(
    'raw',
    textEncoder.encode(secret),
    { name: 'HMAC', hash: 'SHA-256' },
    false,
    ['sign'],
  );
  const signature = await crypto.subtle.sign('HMAC', key, textEncoder.encode(value));

  return new Uint8Array(signature);
}

async function sha256Base64Url(value: string): Promise<string> {
  const digest = await crypto.subtle.digest('SHA-256', textEncoder.encode(value));
  return encodeBase64Url(new Uint8Array(digest));
}

function encodeBase64Url(bytes: Uint8Array): string {
  const binary = Array.from(bytes, (value) => String.fromCharCode(value)).join('');
  return btoa(binary).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/u, '');
}

function constantTimeEqual(left: string, right: string): boolean {
  const maxLength = Math.max(left.length, right.length);
  let difference = left.length ^ right.length;

  for (let index = 0; index < maxLength; index += 1) {
    difference |= (left.charCodeAt(index) || 0) ^ (right.charCodeAt(index) || 0);
  }

  return difference === 0;
}
