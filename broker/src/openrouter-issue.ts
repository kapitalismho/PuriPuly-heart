import type { Context } from 'hono';

import type {
  InstallationRecord,
  OpenRouterEntitlementRecord,
} from './persistence';
import type { BrokerEnv } from './contract';
import { deleteExpiredChallengePreflightInstallations } from './preflight-retention';
import { nonEmptyString, stringValue, validatePublicInput } from './public-input';
import {
  checkActiveIssuanceBrake,
  extractRequestNetworkMetadata,
  checkEndpointRateLimit,
  checkVelocityCapHook,
  matchSubjectHook,
  recordRequestEvent,
  resolveClientIp,
  resolveRequestNetworkIdentitySecrets,
} from './abuse-controls';
import {
  deliverImmediateMonitoringSideEffects,
  evaluateImmediateAbuseState,
  recordIssueSuccess,
} from './abuse-monitoring';
import {
  errorResponse as publicErrorResponse,
  internalErrorResponseWithEntitlement,
} from './broker-error';
import {
  OpenRouterManagementError,
  assignManagedGuardrail,
  cleanupManagedChildKey,
  createManagedChildKey,
} from './openrouter-management';
import { deriveManagedOpenRouterUserId } from './openrouter-user-id';
import {
  MANAGED_TRIAL_BUDGET_POLICY,
  MANAGED_TRIAL_POLICY,
  TRIAL_PROVIDER_POLICY,
} from './trial-policy';

const ISSUE_MAX_CLOCK_SKEW_SECONDS = 60;
const ISSUE_REQUEST_REASON = 'llm_start';
const ISSUE_SIGNATURE_PAYLOAD_FIELDS = [
  'installation_id',
  'device_public_key',
  'release_token',
  'hardware_hash',
  'reason',
  'budget_usd',
  'model',
  'signed_at',
] as const;
const ISSUE_SINGLE_FLIGHT_LOCK_PREFIX = '__issue_lock__:';
const MANAGED_TRIAL_ALLOWED_MODEL_SET = new Set<string>(
  TRIAL_PROVIDER_POLICY.managedFreeTrial.models,
);
const MANAGED_TRIAL_ALLOWED_MODEL_LIST =
  TRIAL_PROVIDER_POLICY.managedFreeTrial.models.join(', ');

class PostActivationMonitoringStateError extends Error {
  readonly issueSuccessRecorded: boolean;

  constructor(input: { cause: unknown; issueSuccessRecorded: boolean }) {
    super(
      input.cause instanceof Error
        ? input.cause.message
        : 'post-activation monitoring state update failed',
    );
    this.name = 'PostActivationMonitoringStateError';
    this.issueSuccessRecorded = input.issueSuccessRecorded;
  }
}

const STRICT_ISO_8601_TIMESTAMP =
  /^(?<year>\d{4})-(?<month>0[1-9]|1[0-2])-(?<day>0[1-9]|[12]\d|3[01])T(?<hour>[01]\d|2[0-3]):(?<minute>[0-5]\d):(?<second>[0-5]\d)(?:\.(?<millisecond>\d{3}))?(?:(?<utc>Z)|(?<offsetSign>[+-])(?<offsetHour>[01]\d|2[0-3]):(?<offsetMinute>[0-5]\d))$/u;

interface IssueRequestBody {
  installation_id?: unknown;
  device_public_key?: unknown;
  release_token?: unknown;
  hardware_hash?: unknown;
  reason?: unknown;
  budget_usd?: unknown;
  model?: unknown;
  signed_at?: unknown;
  signature?: unknown;
}

export async function handleOpenRouterIssue(
  c: Context<BrokerEnv>,
): Promise<Response> {
  const body = await readJsonBody<IssueRequestBody>(c);
  if (!body.ok) {
    return invalidRequestBodyResponse(c, body.reason);
  }

  const installationId = stringValue(body.value.installation_id);
  const devicePublicKey = nonEmptyString(body.value.device_public_key);
  const releaseToken = nonEmptyString(body.value.release_token);
  const hardwareHash = nonEmptyString(body.value.hardware_hash);
  const reason = nonEmptyString(body.value.reason);
  const model = nonEmptyString(body.value.model);
  const signedAt = nonEmptyString(body.value.signed_at);
  const signature = nonEmptyString(body.value.signature);
  const budgetUsd = typeof body.value.budget_usd === 'number' ? body.value.budget_usd : null;

  if (
    !installationId ||
    !devicePublicKey ||
    !releaseToken ||
    !hardwareHash ||
    !reason ||
    budgetUsd === null ||
    !model ||
    !signedAt ||
    !signature
  ) {
    return errorResponse(
      c,
      400,
      'invalid_request',
      'installation_id, device_public_key, release_token, hardware_hash, reason, budget_usd, model, signed_at, and signature are required',
    );
  }

  const installationIdBoundsError = validatePublicInput(
    'installation_id',
    installationId,
  );
  if (installationIdBoundsError) {
    return errorResponse(c, 400, 'invalid_request', installationIdBoundsError);
  }

  const hardwareHashBoundsError = validatePublicInput('hardware_hash', hardwareHash);
  if (hardwareHashBoundsError) {
    return errorResponse(c, 400, 'invalid_request', hardwareHashBoundsError);
  }

  if (!isBase64Url(devicePublicKey, 32) || !isBase64Url(releaseToken, 32) || !isBase64Url(signature, 64)) {
    return errorResponse(
      c,
      400,
      'invalid_request',
      'device_public_key, release_token, and signature must be base64url-encoded contract values',
    );
  }

  if (reason !== ISSUE_REQUEST_REASON) {
    return errorResponse(c, 400, 'invalid_request', 'reason must be llm_start');
  }

  if (budgetUsd !== MANAGED_TRIAL_BUDGET_POLICY.hardLimit) {
    return errorResponse(
      c,
      400,
      'invalid_request',
      `budget_usd must equal ${MANAGED_TRIAL_BUDGET_POLICY.hardLimit}`,
    );
  }

  if (!MANAGED_TRIAL_ALLOWED_MODEL_SET.has(model)) {
    return errorResponse(
      c,
      400,
      'invalid_request',
      `model must be one of ${MANAGED_TRIAL_ALLOWED_MODEL_LIST}`,
    );
  }

  const now = new Date();
  await deleteExpiredChallengePreflightInstallations(c.env.BROKER_DB, {
    installationId,
    devicePublicKey,
    now,
  });

  const requestContext = {
    endpoint: 'POST /v1/providers/openrouter/issue',
    now,
    ip: resolveClientIp(c),
    networkIdentitySecrets: resolveRequestNetworkIdentitySecrets(c),
    installationId,
    hardwareHash,
  };
  const currentEntitlement = await getEntitlement(c.env.BROKER_DB, installationId);

  const subjectHook = await matchSubjectHook(c.env.BROKER_DB, requestContext);
  if (subjectHook) {
    const hookEntitlement = await getEntitlement(c.env.BROKER_DB, installationId);
    return publicErrorResponse(c, subjectHook.status, {
      code: subjectHook.code,
      class: subjectHook.class,
      subcode: subjectHook.subcode,
      retryAfterMs: subjectHook.retryAfterMs,
      message: subjectHook.message,
      entitlement: hookEntitlement,
    });
  }

  const installation = await getInstallation(c.env.BROKER_DB, installationId);
  if (!installation) {
    return releaseTokenInvalidResponse(c, currentEntitlement);
  }

  if (installation.device_public_key !== devicePublicKey) {
    return errorResponse(
      c,
      409,
      'device_public_key_mismatch',
      'issue must use the registered device_public_key for installation_id',
    );
  }

  const signedAtDate = parseIsoDate(signedAt);
  if (!signedAtDate) {
    return errorResponse(
      c,
      400,
      'invalid_request',
      'signed_at must be a valid ISO-8601 timestamp',
    );
  }

  if (
    Math.abs(signedAtDate.getTime() - now.getTime()) >
    ISSUE_MAX_CLOCK_SKEW_SECONDS * 1000
  ) {
    return errorResponse(
      c,
      401,
      'signature_skew',
      'signed_at must be within ±60 seconds of broker time',
    );
  }

  const signatureIsValid = await verifyEd25519Signature({
    devicePublicKey,
    signature,
    payload: buildCanonicalIssuePayload({
      installation_id: installationId,
      device_public_key: devicePublicKey,
      release_token: releaseToken,
      hardware_hash: hardwareHash,
      reason,
      budget_usd: budgetUsd,
      model,
      signed_at: signedAt,
    }),
  });
  if (!signatureIsValid) {
    return errorResponse(
      c,
      401,
      'signature_invalid',
      'signature verification failed for the registered device_public_key',
    );
  }

  const releaseTokenHash = await sha256Base64Url(releaseToken);
  const entitlement = currentEntitlement;
  if (!entitlement || entitlement.release_token_hash !== releaseTokenHash) {
    return releaseTokenInvalidResponse(c, entitlement);
  }

  if (entitlement.status === 'active') {
    return managedKeyUnrecoverableResponse(c, entitlement);
  }

  if (entitlement.status !== 'pending_release') {
    return releaseTokenInvalidResponse(c, entitlement);
  }

  if (isDiscordManagedPendingRelease(entitlement)) {
    return releaseTokenInvalidResponse(c, entitlement);
  }

  if (!entitlement.release_session_ref || !entitlement.release_token_expires_at) {
    return releaseTokenInvalidResponse(c, entitlement);
  }

  const releaseTokenWindowError = validateReleaseTokenWindow(c, entitlement, now);
  if (releaseTokenWindowError) {
    return releaseTokenWindowError;
  }

  if (
    entitlement.verified_hardware_hash !== hardwareHash ||
    entitlement.verified_hardware_hash !== installation.hardware_hash ||
    entitlement.verified_hardware_hash_salt_version !== installation.hardware_hash_salt_version
  ) {
    return hardwareSnapshotMismatchResponse(c, entitlement);
  }

  const issuanceBrakeDecision = await checkActiveIssuanceBrake(
    c.env.BROKER_DB,
    entitlement,
  );
  if (issuanceBrakeDecision) {
    return publicErrorResponse(c, issuanceBrakeDecision.status, {
      code: issuanceBrakeDecision.code,
      class: issuanceBrakeDecision.class,
      subcode: issuanceBrakeDecision.subcode,
      retryAfterMs: issuanceBrakeDecision.retryAfterMs,
      message: issuanceBrakeDecision.message,
      entitlement,
    });
  }

  await recordRequestEvent(c.env.BROKER_DB, requestContext);

  const rateLimitDecision = await checkEndpointRateLimit(c.env.BROKER_DB, requestContext);
  if (rateLimitDecision) {
    return publicErrorResponse(c, rateLimitDecision.status, {
      code: rateLimitDecision.code,
      class: rateLimitDecision.class,
      subcode: rateLimitDecision.subcode,
      retryAfterMs: rateLimitDecision.retryAfterMs,
      message: rateLimitDecision.message,
      entitlement,
    });
  }

  const velocityCapDecision = await checkVelocityCapHook(c.env.BROKER_DB, requestContext);
  if (velocityCapDecision) {
    return publicErrorResponse(c, velocityCapDecision.status, {
      code: velocityCapDecision.code,
      class: velocityCapDecision.class,
      subcode: velocityCapDecision.subcode,
      retryAfterMs: velocityCapDecision.retryAfterMs,
      message: velocityCapDecision.message,
      entitlement,
    });
  }

  const issueLock = await acquireIssueSingleFlight(c.env.BROKER_DB, {
    installationId,
    releaseSessionRef: entitlement.release_session_ref,
    releaseTokenHash,
    releaseTokenExpiresAt: entitlement.release_token_expires_at,
  });
  if (!issueLock.acquired) {
    if (issueLock.entitlement?.status === 'active') {
      return managedKeyUnrecoverableResponse(c, issueLock.entitlement);
    }

    return issueInProgressResponse(c, issueLock.entitlement ?? entitlement);
  }

  const issuedAt = now.toISOString();
  const expiresAt = addMonthsUtc(
    now,
    MANAGED_TRIAL_POLICY.entitlement.issuance.expiry.durationMonths,
  ).toISOString();

  let childKey: { rawKey: string; hash: string } | null = null;
  let activationCommitted = false;
  try {
    childKey = await createManagedChildKey({
      managementApiKey: c.env.OPENROUTER_MANAGEMENT_API_KEY,
      installationId,
      releaseSessionRef: entitlement.release_session_ref,
      expiresAt,
    });
    await assignManagedGuardrail({
      managementApiKey: c.env.OPENROUTER_MANAGEMENT_API_KEY,
      guardrailId: c.env.OPENROUTER_MANAGED_GUARDRAIL_ID,
      keyHash: childKey.hash,
    });

    const activationSucceeded = await activatePendingEntitlement(c.env.BROKER_DB, {
      installationId,
      releaseTokenHash,
      releaseTokenExpiresAt: entitlement.release_token_expires_at,
      issueLockValue: issueLock.lockValue,
      managedCredentialRef: childKey.hash,
      issuedAt,
      expiresAt,
    });
    if (!activationSucceeded) {
      throw new Error('entitlement activation failed after managed child key creation');
    }

    const activeEntitlement = await getEntitlement(c.env.BROKER_DB, installationId);
    if (!activeEntitlement) {
      throw new Error('active entitlement missing after successful issue activation');
    }

    activationCommitted = true;

    await runPostActivationMonitoring(c, {
      installationId,
      managedCredentialRef: activeEntitlement.managed_credential_ref!,
      issuedAt,
      now,
    });

    return await issueSuccessResponse(
      c,
      activeEntitlement,
      childKey.rawKey,
      model,
      installationId,
    );
  } catch (error) {
    if (childKey) {
      if (activationCommitted && error instanceof PostActivationMonitoringStateError) {
        return handlePostActivationMonitoringFailure(c, {
          installationId,
          releaseSessionRef: entitlement.release_session_ref,
          childKey,
          issuedAt,
          issueSuccessRecorded: error.issueSuccessRecorded,
        });
      }

      return handleManagedChildKeyFailure(c, {
        installationId,
        releaseSessionRef: entitlement.release_session_ref,
        childKey,
      });
    }

    return handleAmbiguousManagedChildKeyCreateFailure(c, {
      installationId,
      releaseSessionRef: entitlement.release_session_ref,
      error,
    });
  }
}

async function activatePendingEntitlement(
  db: D1Database,
  input: {
    installationId: string;
    releaseTokenHash: string;
    releaseTokenExpiresAt: string;
    issueLockValue: string;
    managedCredentialRef: string;
    issuedAt: string;
    expiresAt: string;
  },
): Promise<boolean> {
  const result = await db
    .prepare(
      `UPDATE openrouter_entitlements
          SET status = ?,
              managed_credential_ref = ?,
              issued_at = ?,
              expires_at = ?
         WHERE installation_id = ?
           AND status = ?
           AND release_token_hash = ?
           AND release_token_expires_at = ?
           AND managed_credential_ref = ?
           AND (
             discord_issue_status IS NULL
             OR discord_issue_status NOT IN ('issuing', 'cleanup_required')
           )`,
     )
     .bind(
       'active',
       input.managedCredentialRef,
      input.issuedAt,
      input.expiresAt,
       input.installationId,
       'pending_release',
       input.releaseTokenHash,
       input.releaseTokenExpiresAt,
       input.issueLockValue,
     )
     .run();

  return (result.meta.changes ?? 0) === 1;
}

async function acquireIssueSingleFlight(
  db: D1Database,
  input: {
    installationId: string;
    releaseSessionRef: string;
    releaseTokenHash: string;
    releaseTokenExpiresAt: string;
  },
): Promise<
  | {
      acquired: true;
      lockValue: string;
    }
  | {
      acquired: false;
      entitlement: OpenRouterEntitlementRecord | null;
    }
> {
  const lockValue = `${ISSUE_SINGLE_FLIGHT_LOCK_PREFIX}${input.releaseSessionRef}`;
  const result = await db
    .prepare(
      `UPDATE openrouter_entitlements
          SET managed_credential_ref = ?
        WHERE installation_id = ?
          AND status = 'pending_release'
          AND release_session_ref = ?
          AND release_token_hash = ?
          AND release_token_expires_at = ?
          AND managed_credential_ref IS NULL
          AND (
            discord_issue_status IS NULL
            OR discord_issue_status NOT IN ('issuing', 'cleanup_required')
          )`,
    )
    .bind(
      lockValue,
      input.installationId,
      input.releaseSessionRef,
      input.releaseTokenHash,
      input.releaseTokenExpiresAt,
    )
    .run();

  if ((result.meta.changes ?? 0) === 1) {
    return {
      acquired: true,
      lockValue,
    };
  }

  return {
    acquired: false,
    entitlement: await getEntitlement(db, input.installationId),
  };
}

async function invalidatePendingRelease(
  db: D1Database,
  input: {
    installationId: string;
  },
): Promise<void> {
  await db
    .prepare(
      `UPDATE openrouter_entitlements
          SET managed_credential_ref = NULL,
              release_session_ref = NULL,
              release_token_hash = NULL,
              release_token_expires_at = NULL,
              verified_hardware_hash = COALESCE(
                verified_hardware_hash,
                (
                  SELECT hardware_hash
                    FROM installations
                   WHERE installations.installation_id = openrouter_entitlements.installation_id
                )
              ),
              verified_hardware_hash_salt_version = COALESCE(
                verified_hardware_hash_salt_version,
                (
                  SELECT hardware_hash_salt_version
                    FROM installations
                   WHERE installations.installation_id = openrouter_entitlements.installation_id
                )
              )
        WHERE installation_id = ?
          AND status = 'pending_release'
          AND (
            discord_issue_status IS NULL
            OR discord_issue_status NOT IN ('issuing', 'cleanup_required')
          )`,
    )
    .bind(input.installationId)
    .run();
}

async function runPostActivationMonitoring(
  c: Context<BrokerEnv>,
  input: {
    installationId: string;
    managedCredentialRef: string;
    issuedAt: string;
    now: Date;
  },
): Promise<void> {
  try {
    let monitoringResult: Awaited<ReturnType<typeof evaluateImmediateAbuseState>> | null =
      null;
    let issueSuccessRecorded = false;

    try {
      const network = await extractRequestNetworkMetadata(c, { secrets: resolveRequestNetworkIdentitySecrets(c), now: new Date() });
      await recordIssueSuccess(c.env.BROKER_DB, {
        installationId: input.installationId,
        managedCredentialRef: input.managedCredentialRef,
        observedAt: input.issuedAt,
        network,
      });
      issueSuccessRecorded = true;
      monitoringResult = await evaluateImmediateAbuseState(c.env.BROKER_DB, input.now);
    } catch (error) {
      logPostActivationMonitoringFailure({
        installationId: input.installationId,
        managedCredentialRef: input.managedCredentialRef,
        stage: 'record_or_evaluate',
        error,
      });
      throw new PostActivationMonitoringStateError({
        cause: error,
        issueSuccessRecorded,
      });
    }

    const sideEffectPromise = deliverImmediateMonitoringSideEffects(
      c.env,
      monitoringResult,
    ).catch((error) => {
      logPostActivationMonitoringFailure({
        installationId: input.installationId,
        managedCredentialRef: input.managedCredentialRef,
        stage: 'deliver_side_effects',
        error,
      });
    });

    const waitUntil = resolveExecutionWaitUntil(c);
    if (waitUntil) {
      try {
        waitUntil(sideEffectPromise);
        return;
      } catch {
        // Fall through and await inline when the request context does not support waitUntil.
      }
    }

    await sideEffectPromise;
  } catch (error) {
    if (error instanceof PostActivationMonitoringStateError) {
      throw error;
    }

    logPostActivationMonitoringFailure({
      installationId: input.installationId,
      managedCredentialRef: input.managedCredentialRef,
      stage: 'unexpected',
      error,
    });
  }
}

function logPostActivationMonitoringFailure(input: {
  installationId: string;
  managedCredentialRef: string;
  stage: 'record_or_evaluate' | 'deliver_side_effects' | 'unexpected';
  error: unknown;
}): void {
  console.error('post_activation_monitoring_failed', {
    installation_id: input.installationId,
    managed_credential_ref: input.managedCredentialRef,
    stage: input.stage,
    error_message: input.error instanceof Error ? input.error.message : String(input.error),
    broker_timestamp: new Date().toISOString(),
  });
}

function resolveExecutionWaitUntil(
  c: Context<BrokerEnv>,
): ((promise: Promise<unknown>) => void) | null {
  try {
    if (typeof c.executionCtx?.waitUntil !== 'function') {
      return null;
    }

    return c.executionCtx.waitUntil.bind(c.executionCtx);
  } catch {
    return null;
  }
}

async function getInstallation(
  db: D1Database,
  installationId: string,
): Promise<InstallationRecord | null> {
  return db
    .prepare(
      `SELECT installation_id, device_public_key, hardware_hash, hardware_hash_salt_version,
              app_version, challenge, challenge_expires_at, challenge_salt_version,
              created_at, last_seen_at
         FROM installations
        WHERE installation_id = ?`,
    )
    .bind(installationId)
    .first<InstallationRecord>();
}

async function getEntitlement(
  db: D1Database,
  installationId: string,
): Promise<OpenRouterEntitlementRecord | null> {
  return db
    .prepare(
      `SELECT installation_id, status, budget_usd, managed_credential_ref, issued_at,
              expires_at, release_session_ref, release_token_hash, release_token_expires_at,
              verified_hardware_hash, verified_hardware_hash_salt_version,
              discord_user_ref, discord_issue_status, discord_issue_reserved_at,
              discord_issue_delivered_at
         FROM openrouter_entitlements
         WHERE installation_id = ?`,
    )
    .bind(installationId)
    .first<OpenRouterEntitlementRecord>();
}

async function issueSuccessResponse(
  c: Context<BrokerEnv>,
  entitlement: OpenRouterEntitlementRecord,
  rawKey: string,
  model: string,
  installationId: string,
): Promise<Response> {
  if (!entitlement.managed_credential_ref || !entitlement.expires_at) {
    throw new Error('active entitlement missing managed release metadata');
  }

  const managedUserHmacSecret = nonEmptyString(
    c.env.OPENROUTER_MANAGED_USER_HMAC_SECRET,
  );
  let openRouterUserId: string | null = null;
  if (managedUserHmacSecret) {
    try {
      openRouterUserId = await deriveManagedOpenRouterUserId({
        installationId,
        secret: managedUserHmacSecret,
      });
    } catch {
      openRouterUserId = null;
    }
  }

  return c.json({
    openrouter_api_key: rawKey,
    ...(openRouterUserId ? { openrouter_user_id: openRouterUserId } : {}),
    managed_credential_ref: entitlement.managed_credential_ref,
    managed_state: {
      lifecycle: 'active',
      managed_availability: true,
    },
    expires_at: entitlement.expires_at,
    budget_usd: entitlement.budget_usd,
    model,
  });
}

function managedKeyUnrecoverableResponse(
  c: Context<BrokerEnv>,
  entitlement: OpenRouterEntitlementRecord | null,
): Response {
  return publicErrorResponse(c, 409, {
    code: 'trial_not_eligible',
    class: 'terminal',
    subcode: 'managed_key_unrecoverable',
    message: 'managed key was already issued and cannot be recovered',
    entitlement,
  });
}

function issueInProgressResponse(
  c: Context<BrokerEnv>,
  entitlement: OpenRouterEntitlementRecord | null,
): Response {
  return publicErrorResponse(c, 409, {
    code: 'internal_error',
    class: 'retryable',
    subcode: 'issue_in_progress',
    message: 'managed key issuance is already in progress for this release session',
    entitlement,
  });
}

function hardwareSnapshotMismatchResponse(
  c: Context<BrokerEnv>,
  entitlement: OpenRouterEntitlementRecord | null,
): Response {
  return publicErrorResponse(c, 409, {
    code: 'trial_not_eligible',
    class: 'terminal',
    subcode: 'hardware_duplicate',
    message: 'hardware_hash no longer matches the verified release session',
    entitlement,
  });
}

async function handleManagedChildKeyFailure(
  c: Context<BrokerEnv>,
  input: {
    installationId: string;
    releaseSessionRef: string;
    childKey: {
      rawKey: string;
      hash: string;
    };
  },
): Promise<Response> {
  const cleanup = await cleanupManagedChildKey({
    managementApiKey: c.env.OPENROUTER_MANAGEMENT_API_KEY,
    keyHash: input.childKey.hash,
  });

  if (!cleanup.ok) {
    await invalidatePendingRelease(c.env.BROKER_DB, {
      installationId: input.installationId,
    });
    console.error('managed_child_key_orphan_audit', {
      installation_id: input.installationId,
      release_session_ref: input.releaseSessionRef,
      managed_credential_ref: input.childKey.hash,
      cleanup_outcome: cleanup.reason,
      broker_timestamp: new Date().toISOString(),
    });

    return internalErrorResponseWithEntitlement(
      c,
      await getEntitlement(c.env.BROKER_DB, input.installationId),
    );
  }

  await invalidatePendingRelease(c.env.BROKER_DB, {
    installationId: input.installationId,
  });

  return internalErrorResponseWithEntitlement(
    c,
    await getEntitlement(c.env.BROKER_DB, input.installationId),
  );
}

async function handlePostActivationMonitoringFailure(
  c: Context<BrokerEnv>,
  input: {
    installationId: string;
    releaseSessionRef: string;
    childKey: {
      rawKey: string;
      hash: string;
    };
    issuedAt: string;
    issueSuccessRecorded: boolean;
  },
): Promise<Response> {
  if (input.issueSuccessRecorded) {
    await deleteIssueSuccessRecord(c.env.BROKER_DB, {
      installationId: input.installationId,
      managedCredentialRef: input.childKey.hash,
      observedAt: input.issuedAt,
    });
  }

  await rollbackActivatedEntitlement(c.env.BROKER_DB, {
    installationId: input.installationId,
    managedCredentialRef: input.childKey.hash,
  });

  const cleanup = await cleanupManagedChildKey({
    managementApiKey: c.env.OPENROUTER_MANAGEMENT_API_KEY,
    keyHash: input.childKey.hash,
  });

  if (!cleanup.ok) {
    console.error('managed_child_key_orphan_audit', {
      installation_id: input.installationId,
      release_session_ref: input.releaseSessionRef,
      managed_credential_ref: input.childKey.hash,
      cleanup_outcome: cleanup.reason,
      failure_stage: 'post_activation_monitoring',
      broker_timestamp: new Date().toISOString(),
    });
  }

  return internalErrorResponseWithEntitlement(
    c,
    await getEntitlement(c.env.BROKER_DB, input.installationId),
  );
}

async function handleAmbiguousManagedChildKeyCreateFailure(
  c: Context<BrokerEnv>,
  input: {
    installationId: string;
    releaseSessionRef: string;
    error: unknown;
  },
): Promise<Response> {
  await invalidatePendingRelease(c.env.BROKER_DB, {
    installationId: input.installationId,
  });
  console.error('managed_child_key_orphan_audit', {
    installation_id: input.installationId,
    release_session_ref: input.releaseSessionRef,
    managed_credential_ref: null,
    creation_failure: normalizeCreateFailure(input.error),
    broker_timestamp: new Date().toISOString(),
  });

  return internalErrorResponseWithEntitlement(
    c,
    await getEntitlement(c.env.BROKER_DB, input.installationId),
  );
}

async function deleteIssueSuccessRecord(
  db: D1Database,
  input: {
    installationId: string;
    managedCredentialRef: string;
    observedAt: string;
  },
): Promise<void> {
  await db
    .prepare(
      `DELETE FROM broker_issue_success_events
         WHERE installation_id = ?
           AND managed_credential_ref = ?
           AND observed_at = ?`,
    )
    .bind(input.installationId, input.managedCredentialRef, input.observedAt)
    .run();
}

async function rollbackActivatedEntitlement(
  db: D1Database,
  input: {
    installationId: string;
    managedCredentialRef: string;
  },
): Promise<void> {
  await db
    .prepare(
      `UPDATE openrouter_entitlements
          SET status = 'pending_release',
              managed_credential_ref = NULL,
              issued_at = NULL,
              expires_at = NULL,
              release_session_ref = NULL,
              release_token_hash = NULL,
              release_token_expires_at = NULL
        WHERE installation_id = ?
          AND status = 'active'
          AND managed_credential_ref = ?`,
    )
    .bind(input.installationId, input.managedCredentialRef)
    .run();
}

function normalizeCreateFailure(error: unknown): {
  operation: 'create_key';
  code: 'network_error' | 'upstream_http_error' | 'malformed_upstream';
  status: number | null;
  upstreamCode: number | null;
  message: string;
} {
  if (error instanceof OpenRouterManagementError) {
    return {
      operation: 'create_key',
      code: error.code,
      status: error.status,
      upstreamCode: error.upstreamCode,
      message: error.message,
    };
  }

  return {
    operation: 'create_key',
    code: 'network_error',
    status: null,
    upstreamCode: null,
    message: error instanceof Error ? error.message : 'unknown OpenRouter management error',
  };
}

function releaseTokenInvalidResponse(
  c: Context<BrokerEnv>,
  entitlement: OpenRouterEntitlementRecord | null = null,
): Response {
  return publicErrorResponse(c, 401, {
    code: 'challenge_invalid',
    class: 'security_fail',
    subcode: 'release_token_invalid',
    message: 'release_token does not match the active release session for installation_id',
    entitlement,
  });
}

function validateReleaseTokenWindow(
  c: Context<BrokerEnv>,
  entitlement: OpenRouterEntitlementRecord,
  now: Date,
): Response | null {
  if (!entitlement.release_token_expires_at) {
    return releaseTokenInvalidResponse(c, entitlement);
  }

  const releaseTokenExpiresAt = parseIsoDate(entitlement.release_token_expires_at);
  if (!releaseTokenExpiresAt) {
    return releaseTokenInvalidResponse(c, entitlement);
  }

  if (releaseTokenExpiresAt.getTime() < now.getTime()) {
    return publicErrorResponse(c, 410, {
      code: 'challenge_expired',
      class: 'retryable',
      subcode: 'release_token_expired',
      retryAfterMs: 0,
      message: 'release_token has expired and must be reissued',
      entitlement,
    });
  }

  return null;
}

function isDiscordManagedPendingRelease(
  entitlement: OpenRouterEntitlementRecord | null,
): boolean {
  return (
    entitlement?.status === 'pending_release' &&
    (entitlement.discord_issue_status === 'issuing' ||
      entitlement.discord_issue_status === 'cleanup_required')
  );
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
      return {
        ok: false,
        reason: 'not_object',
      };
    }

    return {
      ok: true,
      value: value as T,
    };
  } catch {
    return {
      ok: false,
      reason: 'invalid_json',
    };
  }
}

function isJsonObject(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function invalidRequestBodyResponse(
  c: Context<BrokerEnv>,
  reason: 'invalid_json' | 'not_object',
): Response {
  return errorResponse(
    c,
    400,
    'invalid_request',
    reason === 'invalid_json'
      ? 'request body must be valid JSON'
      : 'request body must be a JSON object',
  );
}

function errorResponse(
  c: Context<BrokerEnv>,
  status: 400 | 401 | 409 | 410,
  code: string,
  message: string,
): Response {
  const normalized = normalizeLegacyIssueError(code, message);

  return publicErrorResponse(c, status, normalized);
}

function normalizeLegacyIssueError(
  code: string,
  message: string,
): {
  code: 'invalid_request' | 'challenge_invalid' | 'trial_not_eligible';
  class: 'terminal' | 'security_fail';
  subcode?: string | null;
  message: string;
} {
  switch (code) {
    case 'invalid_request':
      return {
        code: 'invalid_request',
        class: 'terminal',
        message,
      };
    case 'signature_skew':
      return {
        code: 'challenge_invalid',
        class: 'security_fail',
        subcode: 'timestamp_skew',
        message,
      };
    case 'signature_invalid':
      return {
        code: 'challenge_invalid',
        class: 'security_fail',
        subcode: 'signature_mismatch',
        message,
      };
    case 'device_public_key_mismatch':
      return {
        code: 'trial_not_eligible',
        class: 'security_fail',
        subcode: 'installation_binding_mismatch',
        message,
      };
    default:
      return {
        code: 'challenge_invalid',
        class: 'security_fail',
        subcode: code,
        message,
      };
  }
}

function isBase64Url(value: string, byteLength?: number): boolean {
  if (!/^[A-Za-z0-9_-]+$/u.test(value)) {
    return false;
  }

  try {
    const decoded = decodeBase64Url(value);
    return byteLength === undefined || decoded.length === byteLength;
  } catch {
    return false;
  }
}

function decodeBase64Url(value: string): Uint8Array {
  const padding = (4 - (value.length % 4 || 4)) % 4;
  const normalized = value.replace(/-/g, '+').replace(/_/g, '/') + '='.repeat(padding);
  const binary = atob(normalized);
  const bytes = new Uint8Array(binary.length);

  for (let index = 0; index < binary.length; index += 1) {
    bytes[index] = binary.charCodeAt(index);
  }

  return bytes;
}

function encodeBase64Url(bytes: Uint8Array): string {
  const binary = Array.from(bytes, (value) => String.fromCharCode(value)).join('');
  return btoa(binary).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/u, '');
}

function parseIsoDate(value: string): Date | null {
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

function buildCanonicalIssuePayload(input: {
  installation_id: string;
  device_public_key: string;
  release_token: string;
  hardware_hash: string;
  reason: string;
  budget_usd: number;
  model: string;
  signed_at: string;
}): Uint8Array {
  return new TextEncoder().encode(
    ISSUE_SIGNATURE_PAYLOAD_FIELDS.map((field) => String(input[field])).join('\n'),
  );
}

async function verifyEd25519Signature(input: {
  devicePublicKey: string;
  signature: string;
  payload: Uint8Array;
}): Promise<boolean> {
  try {
    const publicKey = await crypto.subtle.importKey(
      'raw',
      toArrayBuffer(decodeBase64Url(input.devicePublicKey)),
      { name: 'Ed25519' },
      false,
      ['verify'],
    );

    return crypto.subtle.verify(
      { name: 'Ed25519' },
      publicKey,
      toArrayBuffer(decodeBase64Url(input.signature)),
      toArrayBuffer(input.payload),
    );
  } catch {
    return false;
  }
}

async function sha256Base64Url(value: string): Promise<string> {
  const digest = await crypto.subtle.digest(
    'SHA-256',
    toArrayBuffer(new TextEncoder().encode(value)),
  );

  return encodeBase64Url(new Uint8Array(digest));
}

function toArrayBuffer(bytes: Uint8Array): ArrayBuffer {
  return bytes.buffer.slice(
    bytes.byteOffset,
    bytes.byteOffset + bytes.byteLength,
  ) as ArrayBuffer;
}

function randomBase64Url(byteLength: number): string {
  const bytes = new Uint8Array(byteLength);
  crypto.getRandomValues(bytes);
  return encodeBase64Url(bytes);
}

function addMonthsUtc(value: Date, months: number): Date {
  const next = new Date(value.getTime());
  next.setUTCMonth(next.getUTCMonth() + months);
  return next;
}
