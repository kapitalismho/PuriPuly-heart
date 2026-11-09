import type { Context } from 'hono';

import {
  getBrokerAbuseControlsConfig,
  matchSubjectHook,
  resolveClientIp,
  resolveRequestNetworkIdentitySecrets,
} from './abuse-controls';
import { errorResponse } from './broker-error';
import type { BrokerEnv } from './contract';
import { encodeBase64Url, isBase64Url, verifyEd25519Signature } from './crypto';
import {
  resolveRequestNetworkIdentityCandidates,
  type NetworkIdentitySecrets,
} from './network-identity';
import type { BrokerEndpointRateLimitConfig } from './persistence';
import { parseIsoDate, readJsonBody, validatePublicInput } from './public-input';

const TOKEN_LIFETIME_SECONDS = 3600;
const PROOF_MAX_SKEW_MS = 60_000;
const NONCE_RETENTION_MS = 300_000;
const ENROLLMENT_RETENTION_MS = 1440 * 60_000;
const encoder = new TextEncoder();
const REQUEST_FIELDS = [
  'installation_id', 'device_public_key', 'app_version', 'signed_at',
  'nonce', 'signature_alg', 'signature',
] as const;

type PuriTokenRequest = Record<typeof REQUEST_FIELDS[number], string>;

interface PuriRateLimitBucket {
  policy: 'ip_flood' | 'installation';
  config: BrokerEndpointRateLimitConfig;
  subjectKeys: string[];
  windowStartedAt: string;
  expiresAt: string;
  ipKeyVersion: number | null;
  ipEpoch: string | null;
}

interface PuriEnrollmentSubject {
  subjectKey: string;
  candidateKeys: string[];
  ipKeyVersion: number | null;
  ipEpoch: string | null;
}

type SqlGuard = { sql: string; binds: Array<string | number> };

export async function handlePuriToken(c: Context<BrokerEnv>): Promise<Response> {
  c.header('Cache-Control', 'no-store');
  const tokenSecret = c.env.PURI_TOKEN_SECRET;
  const identitySecret = c.env.PURI_IDENTITY_HMAC_SECRET;
  const networkIdentitySecrets = resolveRequestNetworkIdentitySecrets(c);
  if (
    !tokenSecret || !identitySecret || !networkIdentitySecrets || tokenSecret === identitySecret ||
    encoder.encode(tokenSecret).byteLength < 32 || encoder.encode(identitySecret).byteLength < 32
  ) {
    return errorResponse(c, 503, {
      code: 'internal_error', class: 'retryable', subcode: 'puri_unavailable',
      message: 'Puri authentication is unavailable',
    });
  }

  const now = new Date();
  const ip = resolveClientIp(c);
  const controls = await getBrokerAbuseControlsConfig(c.env.BROKER_DB);
  const floodBucket = await networkRateLimitBucket(
    'ip_flood', controls.puriTokenIp, ip, networkIdentitySecrets, now,
  );
  const floodAdmission = await rateLimitStatement(c.env.BROKER_DB, floodBucket).run();
  if (floodAdmission.meta.changes !== 1) {
    return rateLimited(c, floodBucket, now);
  }

  const body = await readJsonBody(c);
  if (!body.ok || !validRequest(body.value)) {
    return errorResponse(c, 400, {
      code: 'invalid_request', class: 'terminal',
      message: 'Puri authentication requires an exact, valid signed request',
    });
  }
  const request = body.value;
  const signedAt = parseIsoDate(request.signed_at)!;
  if (Math.abs(now.getTime() - signedAt.getTime()) > PROOF_MAX_SKEW_MS) {
    return invalidProof(c);
  }
  const validProof = await verifyEd25519Signature({
    devicePublicKey: request.device_public_key,
    signature: request.signature,
    payload: encoder.encode([
      'puripuly-heart:puri-token:v1', 'POST', '/v1/puri/token',
      'puripuly-agent-relay', request.installation_id, request.device_public_key,
      request.app_version, request.signed_at, request.nonce,
    ].join('\n')),
  });
  if (!validProof) {
    return invalidProof(c);
  }

  const subjectHook = await matchSubjectHook(c.env.BROKER_DB, {
    endpoint: 'POST /v1/puri/token', now, ip,
    installationId: request.installation_id, hardwareHash: null, networkIdentitySecrets,
  });
  if (subjectHook) {
    return errorResponse(c, subjectHook.status, subjectHook);
  }

  const enrollmentLimits = [controls.puriEnrollmentShortIp, controls.puriEnrollmentLongIp];
  const enrollmentSubject = await networkEnrollmentSubject(
    ip, networkIdentitySecrets, now,
    Math.max(...enrollmentLimits.map((config) => config.windowMinutes)),
  );
  const enrollmentRoom = enrollmentRoomGuard(enrollmentSubject, enrollmentLimits, now);
  const installationBucket: PuriRateLimitBucket = {
    policy: 'installation', config: controls.puriTokenInstallation,
    subjectKeys: [request.installation_id],
    ...rateLimitWindow(controls.puriTokenInstallation, now),
    ipKeyVersion: null, ipEpoch: null,
  };
  const nowIso = now.toISOString();
  const results = await c.env.BROKER_DB.batch([
    c.env.BROKER_DB.prepare(
      `INSERT INTO installations (
         installation_id, device_public_key, app_version, created_at, last_seen_at
       ) SELECT ?, ?, ?, ?, ?
          WHERE NOT EXISTS (
            SELECT 1 FROM installations WHERE installation_id = ? OR device_public_key = ?
          ) AND ${enrollmentRoom.sql}
       ON CONFLICT DO NOTHING`,
    ).bind(
      request.installation_id, request.device_public_key, request.app_version, nowIso, nowIso,
      request.installation_id, request.device_public_key, ...enrollmentRoom.binds,
    ),
    rateLimitStatement(c.env.BROKER_DB, installationBucket, {
      sql: `EXISTS (
        SELECT 1 FROM installations WHERE installation_id = ? AND device_public_key = ?
      ) AND (EXISTS (
        SELECT 1 FROM installations
         WHERE installation_id = ? AND device_public_key = ? AND puri_bound_at IS NOT NULL
      ) OR ${enrollmentRoom.sql})`,
      binds: [
        request.installation_id, request.device_public_key,
        request.installation_id, request.device_public_key, ...enrollmentRoom.binds,
      ],
    }),
    c.env.BROKER_DB.prepare(
      `INSERT INTO puri_proof_nonces (installation_id, nonce, expires_at)
       SELECT installation_id, ?, ? FROM installations
        WHERE changes() = 1 AND installation_id = ? AND device_public_key = ?
       ON CONFLICT DO NOTHING`,
    ).bind(
      request.nonce, new Date(now.getTime() + NONCE_RETENTION_MS).toISOString(),
      request.installation_id, request.device_public_key,
    ),
    c.env.BROKER_DB.prepare(
      `UPDATE installations SET puri_bound_at = ?
        WHERE changes() = 1 AND puri_bound_at IS NULL AND installation_id = ? AND device_public_key = ?`,
    ).bind(nowIso, request.installation_id, request.device_public_key),
    c.env.BROKER_DB.prepare(
      `INSERT INTO puri_enrollment_admissions (subject_key, admitted_at, ip_key_version, ip_epoch)
       SELECT ?, ?, ?, ? WHERE changes() = 1`,
    ).bind(
      enrollmentSubject.subjectKey, nowIso, enrollmentSubject.ipKeyVersion, enrollmentSubject.ipEpoch,
    ),
  ]);
  if (results[2].meta.changes !== 1) {
    const installation = await c.env.BROKER_DB.prepare(
      `SELECT installation_id, device_public_key, puri_bound_at FROM installations
        WHERE installation_id = ? OR device_public_key = ?`,
    ).bind(request.installation_id, request.device_public_key)
      .first<{ installation_id: string; device_public_key: string; puri_bound_at: string | null }>();
    if (installation && (installation.installation_id !== request.installation_id ||
      installation.device_public_key !== request.device_public_key)) {
      return errorResponse(c, 409, {
        code: 'invalid_request', class: 'security_fail', subcode: 'installation_binding_mismatch',
        message: 'The installation identity is already bound',
      });
    }
    if (results[1].meta.changes !== 1) {
      const enrollmentRetryAfterMs = installation?.puri_bound_at
        ? null
        : await enrollmentRetryAfter(c.env.BROKER_DB, enrollmentSubject, enrollmentLimits, now);
      if (enrollmentRetryAfterMs !== null || !installation) {
        return errorResponse(c, 429, {
          code: 'rate_limited', class: 'retryable', subcode: 'ip_rate_limited',
          message: 'Puri first-enrollment limit exceeded',
          retryAfterMs: enrollmentRetryAfterMs ?? 1,
        });
      }
      return rateLimited(c, installationBucket, now);
    }
    return invalidProof(c);
  }

  const iat = Math.floor(now.getTime() / 1000);
  const exp = iat + TOKEN_LIFETIME_SECONDS;
  const subject = await hmac(identitySecret, request.installation_id);
  const sub = Array.from(subject, (byte) => byte.toString(16).padStart(2, '0')).join('');
  const header = encodeBase64Url(encoder.encode(JSON.stringify({ alg: 'HS256', typ: 'JWT' })));
  const claims = encodeBase64Url(encoder.encode(JSON.stringify({
    iss: 'puripuly-heart-broker', aud: 'puripuly-agent-relay', sub, iat, exp,
  })));
  const signingInput = `${header}.${claims}`;
  const signature = encodeBase64Url(await hmac(tokenSecret, signingInput));
  return c.json({
    access_token: `${signingInput}.${signature}`, token_type: 'Bearer',
    expires_in: TOKEN_LIFETIME_SECONDS, expires_at: new Date(exp * 1000).toISOString(),
  });
}

function validRequest(value: Record<string, unknown>): value is PuriTokenRequest {
  if (Object.keys(value).length !== REQUEST_FIELDS.length || REQUEST_FIELDS.some(
    (field) => typeof value[field] !== 'string' || /[\p{Cc}\p{Cs}\u2028\u2029]/u.test(value[field] as string),
  )) {
    return false;
  }
  const request = value as PuriTokenRequest;
  const signedAt = parseIsoDate(request.signed_at);
  return /^[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/u.test(request.installation_id) &&
    !validatePublicInput('app_version', request.app_version) &&
    canonicalBase64Url(request.device_public_key, 32) &&
    canonicalBase64Url(request.nonce, 32) &&
    canonicalBase64Url(request.signature, 64) &&
    request.signature_alg === 'ed25519' && signedAt !== null &&
    signedAt.toISOString() === request.signed_at;
}

function canonicalBase64Url(value: string, bytes: number): boolean {
  return value.length === Math.ceil(bytes * 8 / 6) && isBase64Url(value, bytes) &&
    (bytes === 32 ? /[AEIMQUYcgkosw048]$/u : /[AQgw]$/u).test(value);
}

function invalidProof(c: Context<BrokerEnv>): Response {
  return errorResponse(c, 401, {
    code: 'invalid_request', class: 'security_fail', subcode: 'puri_proof_invalid',
    message: 'Puri authentication proof is invalid, expired, or already used',
  });
}

async function hmac(secret: string, value: string): Promise<Uint8Array> {
  const key = await crypto.subtle.importKey(
    'raw', encoder.encode(secret), { name: 'HMAC', hash: 'SHA-256' }, false, ['sign'],
  );
  return new Uint8Array(await crypto.subtle.sign('HMAC', key, encoder.encode(value)));
}

function rateLimitWindow(config: BrokerEndpointRateLimitConfig, now: Date) {
  const durationMs = config.windowMinutes * 60_000;
  const startedAt = Math.floor(now.getTime() / durationMs) * durationMs;
  return {
    windowStartedAt: new Date(startedAt).toISOString(),
    expiresAt: new Date(startedAt + durationMs).toISOString(),
  };
}

async function networkRateLimitBucket(
  policy: 'ip_flood',
  config: BrokerEndpointRateLimitConfig,
  ip: string | null,
  secrets: NetworkIdentitySecrets,
  now: Date,
): Promise<PuriRateLimitBucket> {
  const window = rateLimitWindow(config, now);
  const startedAt = new Date(window.windowStartedAt);
  const identities = await resolveRequestNetworkIdentityCandidates(ip, secrets, startedAt, startedAt);
  return {
    policy, config, ...window,
    subjectKeys: identities.length ? identities.map((identity) => identity.digest) : ['unattributed'],
    ipKeyVersion: identities[0]?.keyVersion ?? null,
    ipEpoch: identities[0]?.epoch ?? null,
  };
}

async function networkEnrollmentSubject(
  ip: string | null,
  secrets: NetworkIdentitySecrets,
  now: Date,
  windowMinutes: number,
): Promise<PuriEnrollmentSubject> {
  const [current] = await resolveRequestNetworkIdentityCandidates(ip, secrets, now, now);
  if (!current) {
    return { subjectKey: 'unattributed', candidateKeys: ['unattributed'], ipKeyVersion: null, ipEpoch: null };
  }
  const candidates = await resolveRequestNetworkIdentityCandidates(
    ip, secrets, now, new Date(now.getTime() - windowMinutes * 60_000),
  );
  return {
    subjectKey: current.digest, candidateKeys: candidates.map((identity) => identity.digest),
    ipKeyVersion: current.keyVersion, ipEpoch: current.epoch,
  };
}

function enrollmentRoomGuard(
  subject: PuriEnrollmentSubject,
  limits: BrokerEndpointRateLimitConfig[],
  now: Date,
): SqlGuard {
  const keys = subject.candidateKeys.map(() => '?').join(', ');
  return {
    sql: `(${limits.map(() => `(SELECT COUNT(*) FROM puri_enrollment_admissions
      WHERE subject_key IN (${keys}) AND admitted_at > ?) < ?`).join(' AND ')})`,
    binds: limits.flatMap((config) => [
      ...subject.candidateKeys,
      new Date(now.getTime() - config.windowMinutes * 60_000).toISOString(),
      config.maxRequests,
    ]),
  };
}

async function enrollmentRetryAfter(
  db: D1Database,
  subject: PuriEnrollmentSubject,
  limits: BrokerEndpointRateLimitConfig[],
  now: Date,
): Promise<number | null> {
  let retryAfterMs: number | null = null;
  for (const config of limits) {
    const windowMs = config.windowMinutes * 60_000;
    const limiting = await db.prepare(
      `SELECT admitted_at FROM puri_enrollment_admissions
        WHERE subject_key IN (${subject.candidateKeys.map(() => '?').join(', ')}) AND admitted_at > ?
        ORDER BY admitted_at DESC LIMIT 1 OFFSET ?`,
    ).bind(
      ...subject.candidateKeys, new Date(now.getTime() - windowMs).toISOString(), config.maxRequests - 1,
    ).first<{ admitted_at: string }>();
    if (limiting) {
      retryAfterMs = Math.max(
        retryAfterMs ?? 0, new Date(limiting.admitted_at).getTime() + windowMs - now.getTime(),
      );
    }
  }
  return retryAfterMs;
}

function rateLimitStatement(
  db: D1Database,
  bucket: PuriRateLimitBucket,
  guard?: SqlGuard,
): D1PreparedStatement {
  return db.prepare(
    `INSERT INTO puri_token_rate_limits (
       policy, subject_key, window_started_at, expires_at, request_count, ip_key_version, ip_epoch
     ) SELECT ?, ?, ?, ?, 1, ?, ?
        WHERE ${guard ? `(${guard.sql}) AND` : ''}
          (SELECT COALESCE(SUM(request_count), 0) FROM puri_token_rate_limits
            WHERE policy = ? AND subject_key IN (${bucket.subjectKeys.map(() => '?').join(', ')})
              AND window_started_at = ?) < ?
     ON CONFLICT (policy, subject_key, window_started_at)
     DO UPDATE SET request_count = puri_token_rate_limits.request_count + 1`,
  ).bind(
    bucket.policy, bucket.subjectKeys[0], bucket.windowStartedAt, bucket.expiresAt,
    bucket.ipKeyVersion, bucket.ipEpoch, ...(guard?.binds ?? []),
    bucket.policy, ...bucket.subjectKeys, bucket.windowStartedAt, bucket.config.maxRequests,
  );
}

function rateLimited(c: Context<BrokerEnv>, bucket: PuriRateLimitBucket, now: Date): Response {
  return errorResponse(c, 429, {
    code: 'rate_limited', class: 'retryable',
    subcode: bucket.policy === 'installation' ? 'installation_rate_limited' : 'ip_rate_limited',
    message: 'Puri token request limit exceeded',
    retryAfterMs: Math.max(new Date(bucket.expiresAt).getTime() - now.getTime(), 0),
  });
}

export async function deleteExpiredPuriProofNonces(db: D1Database, now: Date): Promise<void> {
  await db.batch([
    db.prepare('DELETE FROM puri_proof_nonces WHERE expires_at <= ?').bind(now.toISOString()),
    db.prepare('DELETE FROM puri_token_rate_limits WHERE expires_at <= ?').bind(now.toISOString()),
  ]);
  await db.prepare('DELETE FROM puri_enrollment_admissions WHERE admitted_at <= ?')
    .bind(new Date(now.getTime() - ENROLLMENT_RETENTION_MS).toISOString()).run();
}
