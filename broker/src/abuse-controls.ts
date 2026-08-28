import type { Context } from 'hono';

import type { PublicErrorClass, PublicErrorCode } from './broker-error';
import type { BrokerEnv } from './contract';
import type { ManagedIssueMetadata } from './managed-issuance';
import {
  BROKER_RUNTIME_CONFIG_KEYS,
  DEFAULT_BROKER_ABUSE_CONTROLS,
  DEFAULT_BROKER_ABUSE_RUNTIME_STATE,
  type BrokerAbuseControlsConfigValue,
  type BrokerAbuseRuntimeStateValue,
  type BrokerEndpointRateLimitConfig,
  type OpenRouterEntitlementRecord,
  type QqManagedEntitlementRecord,
} from './persistence';

export interface RequestAbuseContext {
  endpoint: string;
  now: Date;
  ip: string | null;
  installationId: string | null;
  hardwareHash: string | null;
}

export interface AbuseDecision {
  status: 400 | 401 | 404 | 409 | 410 | 429 | 500 | 503;
  code: PublicErrorCode;
  class: PublicErrorClass;
  message: string;
  subcode: string | null;
  retryAfterMs: number | null;
}

type ActiveIssuanceBrakeEntitlement =
  | Pick<OpenRouterEntitlementRecord, 'status'>
  | Pick<QqManagedEntitlementRecord, 'status'>
  | null;

export interface ManagedDailyIssuanceCapState {
  reached: boolean;
  retryAfterMs: number | null;
  count: number;
  maxCount: number | null;
}

export interface ManagedDailyIssuanceCapOptions {
  excludeCurrent?: ManagedIssueMetadata;
}

export interface RequestNetworkMetadata {
  ipHash: string | null;
  ipPrefixHash: string | null;
  asn: number | null;
  country: string | null;
  httpProtocol: string | null;
  tlsVersion: string | null;
  tlsCipher: string | null;
  riskLabel: null;
}

export interface SubjectHookMatch extends AbuseDecision {
  hookKind: 'denylist' | 'reputation' | 'revocation';
}

interface VelocityCapHookRow {
  id: number;
  subject_type: 'ip' | 'installation_id';
  subject_value: string;
  max_requests: number;
  window_minutes: number;
  outcome_code: PublicErrorCode;
  outcome_class: PublicErrorClass;
  outcome_subcode: string | null;
  reason: string | null;
  expires_at: string | null;
}

interface SubjectHookRow {
  id: number;
  hook_kind: 'denylist' | 'reputation' | 'revocation';
  subject_type: 'ip' | 'installation_id' | 'hardware_hash';
  subject_value: string;
  outcome_code: PublicErrorCode;
  outcome_class: PublicErrorClass;
  outcome_subcode: string | null;
  reason: string | null;
  expires_at: string | null;
}

export function resolveClientIp(c: Context<BrokerEnv>): string | null {
  const directIp = nonEmptyString(c.req.header('cf-connecting-ip'));
  if (directIp) {
    return directIp;
  }

  const forwardedFor = nonEmptyString(c.req.header('x-forwarded-for'));
  if (!forwardedFor) {
    return null;
  }

  return nonEmptyString(forwardedFor.split(',')[0] ?? null);
}

export async function getBrokerAbuseControlsConfig(
  db: D1Database,
): Promise<BrokerAbuseControlsConfigValue> {
  const row = await db
    .prepare('SELECT value FROM broker_config WHERE key = ?')
    .bind(BROKER_RUNTIME_CONFIG_KEYS.abuseControls)
    .first<{ value: string }>();

  if (!row) {
    return DEFAULT_BROKER_ABUSE_CONTROLS;
  }

  try {
    const parsed = JSON.parse(row.value) as unknown;
    return validateBrokerAbuseControlsConfig(parsed) ?? DEFAULT_BROKER_ABUSE_CONTROLS;
  } catch {
    return DEFAULT_BROKER_ABUSE_CONTROLS;
  }
}

export async function getBrokerAbuseRuntimeState(
  db: D1Database,
): Promise<BrokerAbuseRuntimeStateValue> {
  const row = await db
    .prepare('SELECT value FROM broker_config WHERE key = ?')
    .bind(BROKER_RUNTIME_CONFIG_KEYS.abuseRuntimeState)
    .first<{ value: string }>();

  if (!row) {
    return DEFAULT_BROKER_ABUSE_RUNTIME_STATE;
  }

  try {
    const parsed = JSON.parse(row.value) as unknown;
    return (
      validateBrokerAbuseRuntimeState(parsed) ?? DEFAULT_BROKER_ABUSE_RUNTIME_STATE
    );
  } catch {
    return DEFAULT_BROKER_ABUSE_RUNTIME_STATE;
  }
}

export async function persistBrokerAbuseRuntimeState(
  db: D1Database,
  before: BrokerAbuseRuntimeStateValue,
  after: BrokerAbuseRuntimeStateValue,
): Promise<boolean> {
  const sectionsToPersist: Array<[
    '$.brake' | '$.alertLatches' | '$.dailyReport',
    | BrokerAbuseRuntimeStateValue['brake']
    | BrokerAbuseRuntimeStateValue['alertLatches']
    | BrokerAbuseRuntimeStateValue['dailyReport'],
  ]> = [];

  if (!jsonEquals(before.brake, after.brake)) {
    sectionsToPersist.push(['$.brake', after.brake]);
  }
  if (!jsonEquals(before.alertLatches, after.alertLatches)) {
    sectionsToPersist.push([
      '$.alertLatches',
      {
        warning: after.alertLatches.warning,
        warningObservedAt: after.alertLatches.warningObservedAt,
        warn1: after.alertLatches.warning,
        warn2: after.alertLatches.warning,
        warn3: after.alertLatches.warning,
        critical: after.alertLatches.warning,
      } as BrokerAbuseRuntimeStateValue['alertLatches'],
    ]);
  }
  if (!jsonEquals(before.dailyReport, after.dailyReport)) {
    sectionsToPersist.push(['$.dailyReport', after.dailyReport]);
  }

  if (sectionsToPersist.length === 0) {
    return true;
  }

  const jsonSetArguments = sectionsToPersist.flatMap(([path, value]) => [
    path,
    JSON.stringify(value),
  ]);
  const guardClauses: string[] = [];
  const guardArguments: Array<string | number | null> = [];

  for (const [path, value] of sectionsToPersist) {
    switch (path) {
      case '$.brake': {
        guardClauses.push(
          `json_extract(value, '$.brake.active') = ?`,
          `json_extract(value, '$.brake.reason') IS ?`,
          `json_extract(value, '$.brake.changedAt') IS ?`,
          `json_extract(value, '$.brake.changedBy') IS ?`,
        );
        guardArguments.push(
          sqliteBoolean(before.brake.active),
          before.brake.reason,
          before.brake.changedAt,
          before.brake.changedBy,
        );
        break;
      }
      case '$.alertLatches': {
        guardClauses.push(
          `COALESCE(
             json_extract(value, '$.alertLatches.warning'),
             CASE
               WHEN COALESCE(json_extract(value, '$.alertLatches.warn1'), 0) = 1
                 OR COALESCE(json_extract(value, '$.alertLatches.warn2'), 0) = 1
                 OR COALESCE(json_extract(value, '$.alertLatches.warn3'), 0) = 1
                 OR COALESCE(json_extract(value, '$.alertLatches.critical'), 0) = 1
               THEN 1
               ELSE 0
             END
           ) = ?`,
          `json_extract(value, '$.alertLatches.warningObservedAt') IS ?`,
        );
        guardArguments.push(
          sqliteBoolean(before.alertLatches.warning),
          before.alertLatches.warningObservedAt,
        );
        break;
      }
      case '$.dailyReport': {
        guardClauses.push(
          `json_extract(value, '$.dailyReport.lastDeliveredAt') IS ?`,
          `json_extract(value, '$.dailyReport.lastDeliveredDateUtc') IS ?`,
        );
        guardArguments.push(
          before.dailyReport.lastDeliveredAt,
          before.dailyReport.lastDeliveredDateUtc,
        );
        break;
      }
    }
  }

  const result = await db
    .prepare(
      `UPDATE broker_config
           SET value = json_set(value, ${sectionsToPersist.map(() => '?, json(?)').join(', ')}),
               updated_at = ?
         WHERE key = ?
           AND ${guardClauses.join('\n           AND ')}`,
    )
    .bind(
      ...jsonSetArguments,
      new Date().toISOString(),
      BROKER_RUNTIME_CONFIG_KEYS.abuseRuntimeState,
      ...guardArguments,
    )
    .run();

  return Number(result.meta.changes ?? 0) > 0;
}

function jsonEquals(left: unknown, right: unknown): boolean {
  return JSON.stringify(left) === JSON.stringify(right);
}

function sqliteBoolean(value: boolean): number {
  return value ? 1 : 0;
}

export async function checkActiveIssuanceBrake(
  db: D1Database,
  currentEntitlement: ActiveIssuanceBrakeEntitlement,
): Promise<AbuseDecision | null> {
  const runtimeState = await getBrokerAbuseRuntimeState(db);
  if (!runtimeState.brake.active) {
    return null;
  }

  if (currentEntitlement?.status === 'active') {
    return null;
  }

  return {
    status: 503,
    code: 'issuance_suspended',
    class: 'retryable',
    message: 'new entitlement issuance is temporarily suspended',
    subcode: runtimeState.brake.reason ?? 'manual',
    retryAfterMs: null,
  };
}

export async function extractRequestNetworkMetadata(
  c: Context<BrokerEnv>,
  db: D1Database,
): Promise<RequestNetworkMetadata> {
  const ip = resolveClientIp(c);
  const cf = getCloudflareMetadata(c);
  const asn = normalizePositiveInteger(cf.asn);

  return {
    ipHash: ip ? await hashNetworkValue(ip) : null,
    ipPrefixHash: ip ? await hashNetworkValue(deriveIpPrefix(ip)) : null,
    asn,
    country: nonEmptyString(cf.country),
    httpProtocol: nonEmptyString(cf.httpProtocol),
    tlsVersion: nonEmptyString(cf.tlsVersion),
    tlsCipher: nonEmptyString(cf.tlsCipher),
    riskLabel: null,
  };
}

export async function recordRequestEvent(
  db: D1Database,
  context: RequestAbuseContext,
): Promise<void> {
  if (!context.ip && !context.installationId) {
    return;
  }

  await db
    .prepare(
      `INSERT INTO broker_request_events (
          endpoint,
          ip,
          installation_id,
          observed_at
        ) VALUES (?, ?, ?, ?)`,
    )
    .bind(
      context.endpoint,
      context.ip,
      context.installationId,
      context.now.toISOString(),
    )
    .run();
}

export async function checkEndpointRateLimit(
  db: D1Database,
  context: RequestAbuseContext,
): Promise<AbuseDecision | null> {
  const controls = await getBrokerAbuseControlsConfig(db);
  const endpointConfigs = getEndpointRateLimitConfigs(controls, context.endpoint);
  if (endpointConfigs.length === 0) {
    return null;
  }

  for (const endpointConfig of endpointConfigs) {
    const scopeValue =
      endpointConfig.scope === 'ip' ? context.ip : context.installationId;
    if (!scopeValue) {
      continue;
    }

    const windowStartIso = new Date(
      context.now.getTime() - endpointConfig.windowMinutes * 60_000,
    ).toISOString();
    const scopeColumn = endpointConfig.scope === 'ip' ? 'ip' : 'installation_id';
    const row = await db
      .prepare(
        `SELECT COUNT(*) AS count, MIN(observed_at) AS oldest
           FROM broker_request_events
          WHERE endpoint = ?
            AND ${scopeColumn} = ?
            AND observed_at >= ?`,
      )
      .bind(context.endpoint, scopeValue, windowStartIso)
      .first<{ count: number; oldest: string | null }>();

    const count = Number(row?.count ?? 0);
    if (count <= endpointConfig.maxRequests) {
      continue;
    }

    return {
      status: 429,
      code: 'rate_limited',
      class: 'retryable',
      message: `request rate limit exceeded for ${context.endpoint}`,
      subcode:
        endpointConfig.scope === 'ip'
          ? 'ip_rate_limited'
          : 'installation_rate_limited',
      retryAfterMs: retryAfterFromIso(
        row?.oldest,
        endpointConfig.windowMinutes * 60_000,
        context.now,
      ),
    };
  }

  return null;
}

export async function checkVelocityCapHook(
  db: D1Database,
  context: RequestAbuseContext,
): Promise<AbuseDecision | null> {
  const matchingHooks = await listMatchingVelocityCapHooks(db, context);
  const excludedEndpoints = [
    'POST /v1/trial/challenge/verify/success',
    'POST /v1/trial/challenge/verify/fail',
  ] as const;

  for (const hook of matchingHooks) {
    const windowStartIso = new Date(
      context.now.getTime() - hook.window_minutes * 60_000,
    ).toISOString();
    const column = hook.subject_type === 'ip' ? 'ip' : 'installation_id';
    const row = await db
      .prepare(
        `SELECT COUNT(*) AS count, MIN(observed_at) AS oldest
           FROM broker_request_events
           WHERE ${column} = ?
             AND observed_at >= ?
             AND endpoint NOT IN (?, ?)`,
      )
      .bind(
        hook.subject_value,
        windowStartIso,
        excludedEndpoints[0],
        excludedEndpoints[1],
      )
      .first<{ count: number; oldest: string | null }>();

    const count = Number(row?.count ?? 0);
    if (count <= hook.max_requests) {
      continue;
    }

    return {
      status: mapPublicErrorCodeToStatus(hook.outcome_code),
      code: hook.outcome_code,
      class: hook.outcome_class,
      message: hook.reason ?? 'velocity cap hook rejected the request',
      subcode: normalizeHookPublicSubcode({
        code: hook.outcome_code,
        subjectType: hook.subject_type,
      }),
      retryAfterMs: retryAfterFromIso(
        row?.oldest,
        hook.window_minutes * 60_000,
        context.now,
      ),
    };
  }

  return null;
}

export async function matchSubjectHook(
  db: D1Database,
  context: RequestAbuseContext,
): Promise<SubjectHookMatch | null> {
  const hooks = await listMatchingSubjectHooks(db, context);
  const hook = hooks[0] ?? null;
  if (!hook) {
    return null;
  }

  if (hook.hook_kind === 'revocation') {
    await applyRevocationHook(db, hook);
  }

  return {
    hookKind: hook.hook_kind,
    status: mapPublicErrorCodeToStatus(hook.outcome_code),
    code: hook.outcome_code,
    class: hook.outcome_class,
    message: hook.reason ?? `${hook.hook_kind} hook rejected the request`,
    subcode: normalizeHookPublicSubcode({
      code: hook.outcome_code,
      subjectType: hook.subject_type,
    }),
    retryAfterMs: retryAfterUntilIso(hook.expires_at, context.now),
  };
}

export async function checkDailyIssuanceCap(
  db: D1Database,
  now: Date,
  currentEntitlement: ActiveIssuanceBrakeEntitlement,
): Promise<AbuseDecision | null> {
  if (
    currentEntitlement?.status === 'pending_release' ||
    currentEntitlement?.status === 'active'
  ) {
    return null;
  }

  const cap = await getManagedDailyIssuanceCapState(db, now);
  if (!cap.reached) {
    return null;
  }

  return {
    status: 503,
    code: 'issuance_suspended',
    class: 'retryable',
    message: 'new entitlement issuance is temporarily suspended',
    subcode: 'global_cap_reached',
    retryAfterMs: cap.retryAfterMs,
  };
}

export async function getManagedDailyIssuanceCapState(
  db: D1Database,
  now: Date,
  options: ManagedDailyIssuanceCapOptions = {},
): Promise<ManagedDailyIssuanceCapState> {
  const controls = await getBrokerAbuseControlsConfig(db);
  const maxCount = controls.newActiveEntitlementsPerDay.maxCount;
  if (maxCount === null) {
    return { reached: false, retryAfterMs: null, count: 0, maxCount };
  }

  const window = getManagedDailyIssuanceCapWindow(
    now,
    controls.newActiveEntitlementsPerDay.windowDays,
  );
  const count = await countManagedDailyIssuances(db, window, options.excludeCurrent);
  const reached = count >= maxCount;
  return {
    reached,
    retryAfterMs: reached ? Math.max(window.end.getTime() - now.getTime(), 0) : null,
    count,
    maxCount,
  };
}

async function countManagedDailyIssuances(
  db: D1Database,
  window: { startIso: string; endIso: string },
  excludeCurrent?: ManagedIssueMetadata,
): Promise<number> {
  const openRouterExcludeClause =
    excludeCurrent?.issueSource === 'discord'
      ? 'AND capped.installation_id <> ?'
      : '';
  const openRouterExcludeParams =
    excludeCurrent?.issueSource === 'discord' ? [excludeCurrent.subjectRef] : [];
  const qqExcludeClause =
    excludeCurrent?.issueSource === 'qq'
      ? 'AND NOT (qq_capped.qq_subject_ref = ? AND qq_capped.issue_ref = ?)'
      : '';
  const qqExcludeParams =
    excludeCurrent?.issueSource === 'qq'
      ? [excludeCurrent.subjectRef, excludeCurrent.issueRef]
      : [];

  const row = await db
    .prepare(
      `SELECT (
          SELECT COUNT(*)
            FROM openrouter_entitlements capped
           WHERE COALESCE(
                   capped.issued_at,
                   capped.discord_issue_reserved_at,
                   capped.discord_issue_delivered_at
                 ) >= ?
             AND COALESCE(
                   capped.issued_at,
                   capped.discord_issue_reserved_at,
                   capped.discord_issue_delivered_at
                 ) < ?
           ${openRouterExcludeClause}
        ) + (
          SELECT COUNT(*)
            FROM qq_managed_entitlements qq_capped
           WHERE COALESCE(
                   qq_capped.issued_at,
                   qq_capped.reserved_at,
                   qq_capped.delivered_at
                 ) >= ?
             AND COALESCE(
                   qq_capped.issued_at,
                   qq_capped.reserved_at,
                   qq_capped.delivered_at
                 ) < ?
           ${qqExcludeClause}
        ) AS count`,
    )
    .bind(
      window.startIso,
      window.endIso,
      ...openRouterExcludeParams,
      window.startIso,
      window.endIso,
      ...qqExcludeParams,
    )
    .first<{ count: number }>();

  return Number(row?.count ?? 0);
}

function getManagedDailyIssuanceCapWindow(
  now: Date,
  windowDays: number,
): { startIso: string; endIso: string; end: Date } {
  const start = startOfUtcDay(now);
  start.setUTCDate(start.getUTCDate() - (windowDays - 1));
  const end = new Date(start.getTime());
  end.setUTCDate(end.getUTCDate() + windowDays);
  return {
    startIso: start.toISOString(),
    endIso: end.toISOString(),
    end,
  };
}

export async function hasConflictingHardwareDuplicate(
  db: D1Database,
  input: {
    installationId: string;
    hardwareHash: string;
    challengeSaltVersion: number | null;
    currentSaltVersion: number;
  },
): Promise<boolean> {
  if (
    input.challengeSaltVersion === null ||
    input.challengeSaltVersion !== input.currentSaltVersion
  ) {
    return false;
  }

  const row = await db
    .prepare(
      `SELECT installation_id
         FROM openrouter_entitlements
        WHERE verified_hardware_hash = ?
          AND verified_hardware_hash_salt_version = ?
          AND status IN ('pending_release', 'active', 'expired', 'revoked')
          AND installation_id <> ?
        LIMIT 1`,
    )
    .bind(
      input.hardwareHash,
      input.challengeSaltVersion,
      input.installationId,
    )
    .first<{ installation_id: string }>();

  if (row !== null) {
    return true;
  }

  const legacyReservedRow = await db
    .prepare(
      `SELECT e.installation_id
         FROM openrouter_entitlements e
         JOIN installations i
            ON i.installation_id = e.installation_id
        WHERE e.status IN ('pending_release', 'active', 'expired', 'revoked')
          AND e.verified_hardware_hash IS NULL
          AND e.verified_hardware_hash_salt_version IS NULL
          AND i.hardware_hash = ?
          AND i.hardware_hash_salt_version = ?
          AND e.installation_id <> ?
        LIMIT 1`,
    )
    .bind(
      input.hardwareHash,
      input.challengeSaltVersion,
      input.installationId,
    )
    .first<{ installation_id: string }>();

  return legacyReservedRow !== null;
}

function getEndpointRateLimitConfigs(
  controls: BrokerAbuseControlsConfigValue,
  endpoint: string,
): BrokerEndpointRateLimitConfig[] {
  switch (endpoint) {
    case 'POST /v1/trial/challenge':
      return [controls.trialChallenge];
    case 'POST /v1/trial/challenge/verify':
      return [controls.trialChallengeVerify];
    case 'POST /v1/providers/openrouter/issue':
      return [controls.openrouterIssue];
    case 'GET /v1/trial/status':
      return [controls.trialStatus];
    case 'POST /v1/auth/discord/start':
      return [controls.discordAuthStartIp, controls.discordAuthStartInstallation];
    case 'POST /v1/providers/openrouter/discord/issue':
      return [
        controls.discordOpenrouterIssueIp,
        controls.discordOpenrouterIssueInstallation,
      ];
    case 'POST /v1/auth/qq/assert':
      return [controls.qqAuthAssertIp];
    default:
      return [];
  }
}

async function listMatchingVelocityCapHooks(
  db: D1Database,
  context: RequestAbuseContext,
): Promise<VelocityCapHookRow[]> {
  const filters: string[] = [];
  const params: Array<string | number | null> = [];
  if (context.ip) {
    filters.push('(subject_type = ? AND subject_value = ?)');
    params.push('ip', context.ip);
  }
  if (context.installationId) {
    filters.push('(subject_type = ? AND subject_value = ?)');
    params.push('installation_id', context.installationId);
  }

  if (filters.length === 0) {
    return [];
  }

  const result = await db
    .prepare(
      `SELECT id, subject_type, subject_value, max_requests, window_minutes,
              outcome_code, outcome_class, outcome_subcode, reason, expires_at
         FROM broker_velocity_cap_hooks
        WHERE active = 1
          AND (expires_at IS NULL OR expires_at > ?)
          AND (${filters.join(' OR ')})
        ORDER BY id ASC`,
    )
    .bind(context.now.toISOString(), ...params)
    .all<VelocityCapHookRow>();

  return result.results;
}

async function listMatchingSubjectHooks(
  db: D1Database,
  context: RequestAbuseContext,
): Promise<SubjectHookRow[]> {
  const filters: string[] = [];
  const params: Array<string | number | null> = [];
  if (context.ip) {
    filters.push('(subject_type = ? AND subject_value = ?)');
    params.push('ip', context.ip);
  }
  if (context.installationId) {
    filters.push('(subject_type = ? AND subject_value = ?)');
    params.push('installation_id', context.installationId);
  }
  if (context.hardwareHash) {
    filters.push('(subject_type = ? AND subject_value = ?)');
    params.push('hardware_hash', context.hardwareHash);
  }

  if (filters.length === 0) {
    return [];
  }

  const result = await db
    .prepare(
      `SELECT id, hook_kind, subject_type, subject_value, outcome_code,
              outcome_class, outcome_subcode, reason, expires_at
         FROM broker_abuse_subject_hooks
        WHERE active = 1
          AND (expires_at IS NULL OR expires_at > ?)
          AND (${filters.join(' OR ')})
        ORDER BY CASE hook_kind
          WHEN 'revocation' THEN 0
          WHEN 'denylist' THEN 1
          ELSE 2
        END,
        id ASC`,
    )
    .bind(context.now.toISOString(), ...params)
    .all<SubjectHookRow>();

  return result.results;
}

async function applyRevocationHook(
  db: D1Database,
  hook: SubjectHookRow,
): Promise<void> {
  const installationIds =
    hook.subject_type === 'installation_id'
      ? [hook.subject_value]
      : hook.subject_type === 'hardware_hash'
        ? await listInstallationIdsByHardwareHash(db, hook.subject_value)
        : [];

  if (installationIds.length === 0) {
    return;
  }

  for (const installationId of installationIds) {
    await db
      .prepare(
        `UPDATE openrouter_entitlements
            SET status = 'revoked',
                release_session_ref = NULL,
                release_token_hash = NULL,
                release_token_expires_at = NULL
          WHERE installation_id = ?
            AND status <> 'revoked'`,
      )
      .bind(installationId)
      .run();
  }
}

async function listInstallationIdsByHardwareHash(
  db: D1Database,
  hardwareHash: string,
): Promise<string[]> {
  const result = await db
    .prepare(
      `SELECT installation_id
         FROM installations
        WHERE hardware_hash = ?`,
    )
    .bind(hardwareHash)
    .all<{ installation_id: string }>();

  return result.results.map(
    ({ installation_id }: { installation_id: string }) => installation_id,
  );
}

function retryAfterFromIso(
  startIso: string | null | undefined,
  durationMs: number,
  now: Date,
): number | null {
  if (!startIso) {
    return null;
  }

  return Math.max(new Date(startIso).getTime() + durationMs - now.getTime(), 0);
}

function retryAfterUntilIso(
  expiresAtIso: string | null,
  now: Date,
): number | null {
  if (!expiresAtIso) {
    return null;
  }

  return Math.max(new Date(expiresAtIso).getTime() - now.getTime(), 0);
}

function startOfUtcDay(value: Date): Date {
  return new Date(
    Date.UTC(value.getUTCFullYear(), value.getUTCMonth(), value.getUTCDate()),
  );
}

function normalizeHookPublicSubcode(input: {
  code: PublicErrorCode;
  subjectType: 'ip' | 'installation_id' | 'hardware_hash';
}): string | null {
  if (input.code !== 'rate_limited') {
    return null;
  }

  if (input.subjectType === 'ip') {
    return 'ip_rate_limited';
  }

  if (input.subjectType === 'installation_id') {
    return 'installation_rate_limited';
  }

  return null;
}

function validateBrokerAbuseControlsConfig(
  value: unknown,
): BrokerAbuseControlsConfigValue | null {
  if (!isJsonObject(value)) {
    return null;
  }

  const trialChallenge = validateEndpointRateLimitConfig(
    value.trialChallenge,
    'POST /v1/trial/challenge',
    'ip',
  );
  const trialChallengeVerify = validateEndpointRateLimitConfig(
    value.trialChallengeVerify,
    'POST /v1/trial/challenge/verify',
    'installation_id',
  );
  const openrouterIssue = validateEndpointRateLimitConfig(
    value.openrouterIssue,
    'POST /v1/providers/openrouter/issue',
    'installation_id',
  );
  const trialStatus = validateEndpointRateLimitConfig(
    value.trialStatus,
    'GET /v1/trial/status',
    'installation_id',
  );
  const discordAuthStartIp = validateEndpointRateLimitConfig(
    value.discordAuthStartIp,
    'POST /v1/auth/discord/start',
    'ip',
  );
  const discordAuthStartInstallation = validateEndpointRateLimitConfig(
    value.discordAuthStartInstallation,
    'POST /v1/auth/discord/start',
    'installation_id',
  );
  const discordOpenrouterIssueIp = validateEndpointRateLimitConfig(
    value.discordOpenrouterIssueIp,
    'POST /v1/providers/openrouter/discord/issue',
    'ip',
  );
  const discordOpenrouterIssueInstallation = validateEndpointRateLimitConfig(
    value.discordOpenrouterIssueInstallation,
    'POST /v1/providers/openrouter/discord/issue',
    'installation_id',
  );
  const qqAuthAssertIp = validateEndpointRateLimitConfig(
    value.qqAuthAssertIp,
    'POST /v1/auth/qq/assert',
    'ip',
  );
  const pendingDiscordOAuthSessions = validatePendingDiscordOAuthSessionsConfig(
    value.pendingDiscordOAuthSessions,
  );
  const newActiveEntitlementsPerDay = validateDailyIssuanceCapConfig(
    value.newActiveEntitlementsPerDay,
  );
  const immediateAlerts = validateImmediateAlertsConfig(value.immediateAlerts);
  const retention = validateRetentionConfig(value.retention);
  const referralAttempts = validateReferralAttemptControlsConfig(value.referralAttempts);
  const dailyReport = validateDailyReportConfig(value.dailyReport);

  if (
    !trialChallenge ||
    !trialChallengeVerify ||
    !openrouterIssue ||
    !trialStatus ||
    !discordAuthStartIp ||
    !discordAuthStartInstallation ||
    !discordOpenrouterIssueIp ||
    !discordOpenrouterIssueInstallation ||
    !qqAuthAssertIp ||
    !pendingDiscordOAuthSessions ||
    !newActiveEntitlementsPerDay ||
    !immediateAlerts ||
    !retention ||
    !referralAttempts ||
    !dailyReport
  ) {
    return null;
  }

  return {
    trialChallenge,
    trialChallengeVerify,
    openrouterIssue,
    trialStatus,
    discordAuthStartIp,
    discordAuthStartInstallation,
    discordOpenrouterIssueIp,
    discordOpenrouterIssueInstallation,
    qqAuthAssertIp,
    pendingDiscordOAuthSessions,
    newActiveEntitlementsPerDay,
    immediateAlerts,
    retention,
    referralAttempts,
    dailyReport,
  };
}

function validateBrokerAbuseRuntimeState(
  value: unknown,
): BrokerAbuseRuntimeStateValue | null {
  if (!isJsonObject(value) || !isJsonObject(value.brake) || !isJsonObject(value.alertLatches) || !isJsonObject(value.dailyReport)) {
    return null;
  }

  const brakeReason = value.brake.reason;
  const brakeChangedBy = value.brake.changedBy;
  const brakeChangedAt = value.brake.changedAt;
  const explicitWarning = value.alertLatches.warning;
  const legacyWarningValues = [
    value.alertLatches.warn1,
    value.alertLatches.warn2,
    value.alertLatches.warn3,
    value.alertLatches.critical,
  ];
  const warning = isBoolean(explicitWarning)
    ? explicitWarning
    : legacyWarningValues.every(isBoolean)
      ? legacyWarningValues.some((entry) => entry === true)
      : null;
  const warningObservedAt = value.alertLatches.warningObservedAt;
  const lastDeliveredAt = value.dailyReport.lastDeliveredAt;
  const lastDeliveredDateUtc = value.dailyReport.lastDeliveredDateUtc;

  if (
    !isBoolean(value.brake.active) ||
    !(
      brakeReason === null ||
      brakeReason === 'global_threshold' ||
      brakeReason === 'asn_fast_path' ||
      brakeReason === 'manual'
    ) ||
    !(brakeChangedBy === null || brakeChangedBy === 'system' || brakeChangedBy === 'operator') ||
    !(brakeChangedAt === null || typeof brakeChangedAt === 'string') ||
    warning === null ||
    !(warningObservedAt === undefined || warningObservedAt === null || typeof warningObservedAt === 'string') ||
    !(lastDeliveredAt === null || typeof lastDeliveredAt === 'string') ||
    !(lastDeliveredDateUtc === null || typeof lastDeliveredDateUtc === 'string')
  ) {
    return null;
  }

  return {
    brake: {
      active: value.brake.active,
      reason: brakeReason,
      changedAt: brakeChangedAt,
      changedBy: brakeChangedBy,
    },
    alertLatches: {
      warning,
      warningObservedAt:
        typeof warningObservedAt === 'string' ? warningObservedAt : null,
    },
    dailyReport: {
      lastDeliveredAt,
      lastDeliveredDateUtc,
    },
  };
}

function validateEndpointRateLimitConfig(
  value: unknown,
  endpoint: BrokerEndpointRateLimitConfig['endpoint'],
  scope: BrokerEndpointRateLimitConfig['scope'],
): BrokerEndpointRateLimitConfig | null {
  if (!isJsonObject(value)) {
    return null;
  }

  if (
    value.endpoint !== endpoint ||
    value.scope !== scope ||
    !isPositiveInteger(value.maxRequests) ||
    !isPositiveInteger(value.windowMinutes)
  ) {
    return null;
  }

  return {
    endpoint,
    scope,
    maxRequests: value.maxRequests,
    windowMinutes: value.windowMinutes,
  };
}

function validateDailyIssuanceCapConfig(
  value: unknown,
): BrokerAbuseControlsConfigValue['newActiveEntitlementsPerDay'] | null {
  if (!isJsonObject(value)) {
    return null;
  }

  const endpoint = value.endpoint;
  if (
    !(
      endpoint === 'POST /v1/providers/openrouter/issue' ||
      endpoint === 'POST /v1/providers/openrouter/discord/issue'
    ) ||
    value.scope !== 'global' ||
    !(value.maxCount === null || isPositiveInteger(value.maxCount)) ||
    !isPositiveInteger(value.windowDays)
  ) {
    return null;
  }

  return {
    endpoint,
    scope: 'global',
    maxCount: value.maxCount,
    windowDays: value.windowDays,
  };
}

function validatePendingDiscordOAuthSessionsConfig(
  value: unknown,
): BrokerAbuseControlsConfigValue['pendingDiscordOAuthSessions'] | null {
  if (!isJsonObject(value)) {
    return null;
  }

  if (
    !isPositiveInteger(value.maxPerInstallation) ||
    !isPositiveInteger(value.maxPerIp) ||
    !isPositiveInteger(value.windowMinutes)
  ) {
    return null;
  }

  return {
    maxPerInstallation: value.maxPerInstallation,
    maxPerIp: value.maxPerIp,
    windowMinutes: value.windowMinutes,
  };
}

function validateImmediateAlertsConfig(
  value: unknown,
): BrokerAbuseControlsConfigValue['immediateAlerts'] | null {
  if (!isJsonObject(value)) {
    return null;
  }

  if (
    !isPositiveInteger(value.warning) ||
    !isPositiveInteger(value.brake) ||
    !(value.warning < value.brake)
  ) {
    return null;
  }

  return {
    warning: value.warning,
    brake: value.brake,
  };
}

function validateRetentionConfig(
  value: unknown,
): BrokerAbuseControlsConfigValue['retention'] | null {
  if (!isJsonObject(value)) {
    return null;
  }

  if (
    !isPositiveInteger(value.requestEventSafetyMarginDays) ||
    !isPositiveInteger(value.issueSuccessDays) ||
    !isPositiveInteger(value.runtimeAuditDays) ||
    !isPositiveInteger(value.referralSkippedDays) ||
    !isPositiveInteger(value.referralFailedDays)
  ) {
    return null;
  }

  return {
    requestEventSafetyMarginDays: value.requestEventSafetyMarginDays,
    issueSuccessDays: Math.max(2, value.issueSuccessDays),
    runtimeAuditDays: value.runtimeAuditDays,
    referralSkippedDays: value.referralSkippedDays,
    referralFailedDays: value.referralFailedDays,
  };
}

function validateReferralAttemptControlsConfig(
  value: unknown,
): BrokerAbuseControlsConfigValue['referralAttempts'] | null {
  if (!isJsonObject(value)) {
    return null;
  }

  const validShaped = validateReferralScopedAttemptLimit(value.validShaped);
  const unknown = validateReferralScopedAttemptLimit(value.unknown);
  const perReferralIdVelocity = validateReferralIdVelocityLimit(
    value.perReferralIdVelocity,
  );
  const perReferrerRewardVelocity = validateReferrerRewardVelocityLimit(
    value.perReferrerRewardVelocity,
  );

  if (
    !validShaped ||
    !unknown ||
    !perReferralIdVelocity ||
    !perReferrerRewardVelocity
  ) {
    return null;
  }

  return {
    validShaped,
    unknown,
    perReferralIdVelocity,
    perReferrerRewardVelocity,
  };
}

function validateReferralScopedAttemptLimit(
  value: unknown,
): BrokerAbuseControlsConfigValue['referralAttempts']['validShaped'] | null {
  if (!isJsonObject(value)) {
    return null;
  }

  if (
    !isPositiveInteger(value.maxPerInstallation) ||
    !isPositiveInteger(value.maxPerIp) ||
    !isPositiveInteger(value.windowMinutes)
  ) {
    return null;
  }

  return {
    maxPerInstallation: value.maxPerInstallation,
    maxPerIp: value.maxPerIp,
    windowMinutes: value.windowMinutes,
  };
}

function validateReferralIdVelocityLimit(
  value: unknown,
): BrokerAbuseControlsConfigValue['referralAttempts']['perReferralIdVelocity'] | null {
  if (!isJsonObject(value)) {
    return null;
  }

  if (!isPositiveInteger(value.maxAttempts) || !isPositiveInteger(value.windowMinutes)) {
    return null;
  }

  return {
    maxAttempts: value.maxAttempts,
    windowMinutes: value.windowMinutes,
  };
}

function validateReferrerRewardVelocityLimit(
  value: unknown,
): BrokerAbuseControlsConfigValue['referralAttempts']['perReferrerRewardVelocity'] | null {
  if (!isJsonObject(value)) {
    return null;
  }

  if (!isPositiveInteger(value.maxRewards) || !isPositiveInteger(value.windowMinutes)) {
    return null;
  }

  return {
    maxRewards: value.maxRewards,
    windowMinutes: value.windowMinutes,
  };
}

function validateDailyReportConfig(
  value: unknown,
): BrokerAbuseControlsConfigValue['dailyReport'] | null {
  if (!isJsonObject(value)) {
    return null;
  }

  if (
    !isBoolean(value.enabled) ||
    !isIntegerInRange(value.hourUtc, 0, 23) ||
    !isIntegerInRange(value.minuteUtc, 0, 59)
  ) {
    return null;
  }

  return {
    enabled: value.enabled,
    hourUtc: value.hourUtc,
    minuteUtc: value.minuteUtc,
  };
}

function getCloudflareMetadata(c: Context<BrokerEnv>): {
  asn: unknown;
  country: unknown;
  httpProtocol: unknown;
  tlsVersion: unknown;
  tlsCipher: unknown;
} {
  const rawRequest = c.req.raw as Request & {
    cf?: Record<string, unknown>;
  };
  const cf = rawRequest.cf ?? {};

  return {
    asn: cf.asn,
    country: cf.country ?? nonEmptyString(c.req.header('cf-ipcountry')),
    httpProtocol: cf.httpProtocol,
    tlsVersion: cf.tlsVersion,
    tlsCipher: cf.tlsCipher,
  };
}

function normalizePositiveInteger(value: unknown): number | null {
  if (typeof value === 'number' && Number.isInteger(value) && value > 0) {
    return value;
  }

  if (typeof value === 'string' && /^\d+$/u.test(value)) {
    const parsed = Number(value);
    return Number.isSafeInteger(parsed) && parsed > 0 ? parsed : null;
  }

  return null;
}

async function hashNetworkValue(value: string): Promise<string> {
  const digest = await crypto.subtle.digest(
    'SHA-256',
    new TextEncoder().encode(value),
  );

  return Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, '0')).join('');
}

function deriveIpPrefix(ip: string): string {
  if (ip.includes(':')) {
    return ip
      .split(':')
      .filter((part) => part.length > 0)
      .slice(0, 4)
      .join(':');
  }

  return ip.split('.').slice(0, 3).join('.');
}

function isJsonObject(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function isPositiveInteger(value: unknown): value is number {
  return typeof value === 'number' && Number.isInteger(value) && value > 0;
}

function isBoolean(value: unknown): value is boolean {
  return typeof value === 'boolean';
}

function isIntegerInRange(value: unknown, min: number, max: number): value is number {
  return typeof value === 'number' && Number.isInteger(value) && value >= min && value <= max;
}

function nonEmptyString(value: unknown): string | null {
  return typeof value === 'string' && value.trim().length > 0 ? value.trim() : null;
}

function mapPublicErrorCodeToStatus(
  code: PublicErrorCode,
): 400 | 401 | 404 | 409 | 410 | 429 | 500 | 503 {
  switch (code) {
    case 'invalid_request':
      return 400;
    case 'rate_limited':
      return 429;
    case 'challenge_expired':
      return 410;
    case 'challenge_invalid':
      return 401;
    case 'issuance_suspended':
    case 'trial_unavailable':
      return 503;
    case 'trial_not_eligible':
      return 409;
    case 'internal_error':
      return 500;
  }
}
