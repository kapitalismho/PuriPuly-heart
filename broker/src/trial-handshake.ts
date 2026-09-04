import type { Context } from 'hono';

import type {
  InstallationRecord,
  OpenRouterEntitlementRecord,
  ReferralCodeRecord,
} from './persistence';
import type { BrokerEnv } from './contract';
import { getFingerprintSaltConfig } from './fingerprint-salt';
import { deleteExpiredChallengePreflightInstallations } from './preflight-retention';
import { nonEmptyString, stringValue, validatePublicInput } from './public-input';
import {
  checkActiveIssuanceBrake,
  checkDailyIssuanceCap,
  checkEndpointRateLimit,
  checkVelocityCapHook,
  hasConflictingHardwareDuplicate,
  matchSubjectHook,
  recordRequestEvent,
  resolveClientIp,
  resolveRequestNetworkIdentitySecrets,
} from './abuse-controls';
import { errorResponse as publicErrorResponse } from './broker-error';
import {
  normalizeManagedState,
  normalizeTrialStatusResponse,
  type ManagedStateResponse,
  type TalkTogetherPassStatusResponse,
  type TrialStatusResponse,
} from './managed-state';
import {
  MANAGED_TRIAL_BUDGET_POLICY,
  TRIAL_PROVIDER_POLICY,
} from './trial-policy';
import {
  ensureOwnedReferralIdForActiveDiscordManagedUser,
  resolveTalkTogetherPassStatusForOwnedReferralCode,
} from './referral';

export const TRIAL_CHALLENGE_TTL_SECONDS = 300;
export const TRIAL_RELEASE_TOKEN_TTL_SECONDS = 900;
export const TRIAL_VERIFY_MAX_CLOCK_SKEW_SECONDS = 60;
export const TRIAL_STATUS_TIMESTAMP_HEADER = 'X-Puripuly-Timestamp';
export const TRIAL_STATUS_SIGNATURE_HEADER = 'X-Puripuly-Signature';
export const TRIAL_STATUS_MAX_CLOCK_SKEW_SECONDS =
  TRIAL_VERIFY_MAX_CLOCK_SKEW_SECONDS;
const VERIFY_OUTCOME_SUCCESS_ENDPOINT = 'POST /v1/trial/challenge/verify/success';
const VERIFY_OUTCOME_FAIL_ENDPOINT = 'POST /v1/trial/challenge/verify/fail';
export const TRIAL_STATUS_SIGNATURE_PAYLOAD_FIELDS = [
  'installation_id',
  'timestamp',
] as const;
export const TRIAL_VERIFY_SIGNATURE_PAYLOAD_FIELDS = [
  'installation_id',
  'device_public_key',
  'challenge',
  'challenge_expires_at',
  'hardware_hash',
  'app_version',
  'signed_at',
] as const;

const STRICT_ISO_8601_TIMESTAMP =
  /^(?<year>\d{4})-(?<month>0[1-9]|1[0-2])-(?<day>0[1-9]|[12]\d|3[01])T(?<hour>[01]\d|2[0-3]):(?<minute>[0-5]\d):(?<second>[0-5]\d)(?:\.(?<millisecond>\d{3}))?(?:(?<utc>Z)|(?<offsetSign>[+-])(?<offsetHour>[01]\d|2[0-3]):(?<offsetMinute>[0-5]\d))$/u;

interface ChallengeRequestBody {
  installation_id?: unknown;
  device_public_key?: unknown;
  app_version?: unknown;
  hardware_hash?: unknown;
  signed_at?: unknown;
  signature?: unknown;
}

interface VerifyRequestBody {
  installation_id?: unknown;
  device_public_key?: unknown;
  challenge?: unknown;
  challenge_expires_at?: unknown;
  hardware_hash?: unknown;
  app_version?: unknown;
  signed_at?: unknown;
  signature?: unknown;
}

export async function handleTrialChallenge(
  c: Context<BrokerEnv>,
): Promise<Response> {
  const body = await readJsonBody<ChallengeRequestBody>(c);
  if (!body.ok) {
    return invalidRequestBodyResponse(c, body.reason);
  }

  const request = body.value;
  if (
    request.hardware_hash !== undefined ||
    request.signed_at !== undefined ||
    request.signature !== undefined
  ) {
    return errorResponse(
      c,
      400,
      'invalid_request',
      'challenge request must not include hardware_hash, signed_at, or signature',
    );
  }

  const installationId = stringValue(request.installation_id);
  const devicePublicKey = nonEmptyString(request.device_public_key);
  const appVersion = stringValue(request.app_version);

  if (!installationId || !devicePublicKey || !appVersion) {
    return errorResponse(
      c,
      400,
      'invalid_request',
      'installation_id, device_public_key, and app_version are required',
    );
  }

  const installationIdBoundsError = validatePublicInput(
    'installation_id',
    installationId,
  );
  if (installationIdBoundsError) {
    return errorResponse(c, 400, 'invalid_request', installationIdBoundsError);
  }

  const appVersionBoundsError = validatePublicInput('app_version', appVersion);
  if (appVersionBoundsError) {
    return errorResponse(c, 400, 'invalid_request', appVersionBoundsError);
  }

  if (!isBase64Url(devicePublicKey, 32)) {
    return errorResponse(
      c,
      400,
      'invalid_request',
      'device_public_key must be base64url-encoded Ed25519 public key bytes',
    );
  }

  const now = new Date();
  await deleteExpiredChallengePreflightInstallations(c.env.BROKER_DB, {
    installationId,
    devicePublicKey,
    now,
  });

  const existingInstallation = await getInstallation(c.env.BROKER_DB, installationId);
  const entitlement = existingInstallation
    ? await getEntitlement(c.env.BROKER_DB, installationId)
    : null;
  const trustedInstallationId =
    existingInstallation && existingInstallation.device_public_key !== devicePublicKey
      ? null
      : installationId;
  const requestContext = {
    endpoint: 'POST /v1/trial/challenge',
    now,
    ip: resolveClientIp(c),
    networkIdentitySecrets: resolveRequestNetworkIdentitySecrets(c),
    installationId: trustedInstallationId,
    hardwareHash: null,
  };

  const subjectHook = await matchSubjectHook(c.env.BROKER_DB, requestContext);
  if (subjectHook) {
    const hookEntitlement = existingInstallation
      ? await getEntitlement(c.env.BROKER_DB, installationId)
      : null;
    return publicErrorResponse(c, subjectHook.status, {
      code: subjectHook.code,
      class: subjectHook.class,
      subcode: subjectHook.subcode,
      retryAfterMs: subjectHook.retryAfterMs,
      message: subjectHook.message,
      entitlement: hookEntitlement,
    });
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

  if (
    existingInstallation &&
    existingInstallation.device_public_key !== devicePublicKey
  ) {
    return publicErrorResponse(c, 409, {
      code: 'trial_not_eligible',
      class: 'security_fail',
      subcode: 'installation_binding_mismatch',
      message: 'installation_id is already bound to a different device_public_key',
      entitlement,
    });
  }

  const installationByPublicKey = await getInstallationByPublicKey(
    c.env.BROKER_DB,
    devicePublicKey,
  );
  if (
    installationByPublicKey &&
    installationByPublicKey.installation_id !== installationId
  ) {
    return publicErrorResponse(c, 409, {
      code: 'trial_not_eligible',
      class: 'security_fail',
      subcode: 'device_public_key_registered',
      message: 'device_public_key is already registered to a different installation_id',
      entitlement,
    });
  }

  if (entitlement && isDiscordManagedPendingRelease(entitlement)) {
    return releaseNotAllowedResponse(c, entitlement);
  }

  const issuanceCapDecision = await checkDailyIssuanceCap(
    c.env.BROKER_DB,
    now,
    entitlement,
  );
  if (issuanceCapDecision) {
    return publicErrorResponse(c, issuanceCapDecision.status, {
      code: issuanceCapDecision.code,
      class: issuanceCapDecision.class,
      subcode: issuanceCapDecision.subcode,
      retryAfterMs: issuanceCapDecision.retryAfterMs,
      message: issuanceCapDecision.message,
      entitlement,
    });
  }

  const challenge = randomBase64Url(32);
  const challengeExpiresAt = new Date(
    now.getTime() + TRIAL_CHALLENGE_TTL_SECONDS * 1000,
  ).toISOString();
  const fingerprintSalt = await getFingerprintSaltConfig(c.env.BROKER_DB);
  const challengeWriteInput = {
    installationId,
    devicePublicKey,
    appVersion,
    challenge,
    challengeExpiresAt,
    challengeSaltVersion: fingerprintSalt.current.version,
    nowIso: now.toISOString(),
  };
  let responseChallenge = challenge;
  let responseChallengeExpiresAt = challengeExpiresAt;

  if (existingInstallation) {
    if (entitlement?.status === 'pending_release') {
      await clearPendingReleaseSessionTokenState(c.env.BROKER_DB, entitlement);
    }

    const updateSucceeded = await updateChallengeForInstallation(c.env.BROKER_DB, {
      ...challengeWriteInput,
      clearHardwareHash:
        entitlement === null || entitlement.status === 'pending_release',
      currentInstallation: existingInstallation,
    });

    if (!updateSucceeded) {
      const persistedChallenge = await resolveExistingInstallationChallengeReissue(
        c.env.BROKER_DB,
        {
          installationId,
          devicePublicKey,
          appVersion,
          challenge,
          challengeExpiresAt,
          challengeSaltVersion: fingerprintSalt.current.version,
          nowIso: now.toISOString(),
        },
      );

      responseChallenge = persistedChallenge.challenge;
      responseChallengeExpiresAt = persistedChallenge.challenge_expires_at;
    }
  } else {
    try {
      await insertChallengeForInstallation(c.env.BROKER_DB, challengeWriteInput);
    } catch (error) {
      if (!isUniqueConstraintError(error)) {
        throw error;
      }

      const conflictingInstallation = await getInstallation(
        c.env.BROKER_DB,
        installationId,
      );
      if (conflictingInstallation) {
        if (conflictingInstallation.device_public_key !== devicePublicKey) {
          return publicErrorResponse(c, 409, {
            code: 'trial_not_eligible',
            class: 'security_fail',
            subcode: 'installation_binding_mismatch',
            message: 'installation_id is already bound to a different device_public_key',
            entitlement,
          });
        }

        if (
          !conflictingInstallation.challenge ||
          !conflictingInstallation.challenge_expires_at
        ) {
          throw error;
        }

        responseChallenge = conflictingInstallation.challenge;
        responseChallengeExpiresAt = conflictingInstallation.challenge_expires_at;
      } else {
        const conflictingPublicKeyInstallation = await getInstallationByPublicKey(
          c.env.BROKER_DB,
          devicePublicKey,
        );

        if (
          conflictingPublicKeyInstallation &&
          conflictingPublicKeyInstallation.installation_id !== installationId
        ) {
          return publicErrorResponse(c, 409, {
            code: 'trial_not_eligible',
            class: 'security_fail',
            subcode: 'device_public_key_registered',
            message: 'device_public_key is already registered to a different installation_id',
            entitlement,
          });
        }

        throw error;
      }
    }
  }

  const responseEntitlement = await getEntitlement(c.env.BROKER_DB, installationId);

  return c.json({
    challenge: responseChallenge,
    challenge_expires_at: responseChallengeExpiresAt,
    fingerprint_salt: {
      version: fingerprintSalt.current.version,
      salt: fingerprintSalt.current.salt,
    },
    ...normalizeManagedState(responseEntitlement),
  });
}

export async function handleTrialChallengeVerify(
  c: Context<BrokerEnv>,
): Promise<Response> {
  const body = await readJsonBody<VerifyRequestBody>(c);
  if (!body.ok) {
    return invalidRequestBodyResponse(c, body.reason);
  }

  const installationId = stringValue(body.value.installation_id);
  const devicePublicKey = nonEmptyString(body.value.device_public_key);
  const challenge = nonEmptyString(body.value.challenge);
  const challengeExpiresAt = nonEmptyString(body.value.challenge_expires_at);
  const hardwareHash = stringValue(body.value.hardware_hash);
  const appVersion = stringValue(body.value.app_version);
  const signedAt = nonEmptyString(body.value.signed_at);
  const signature = nonEmptyString(body.value.signature);

  if (
    !installationId ||
    !devicePublicKey ||
    !challenge ||
    !challengeExpiresAt ||
    !hardwareHash ||
    !appVersion ||
    !signedAt ||
    !signature
  ) {
    return errorResponse(
      c,
      400,
      'invalid_request',
      'installation_id, device_public_key, challenge, challenge_expires_at, hardware_hash, app_version, signed_at, and signature are required',
    );
  }

  const installationIdBoundsError = validatePublicInput(
    'installation_id',
    installationId,
  );
  if (installationIdBoundsError) {
    return errorResponse(c, 400, 'invalid_request', installationIdBoundsError);
  }

  const appVersionBoundsError = validatePublicInput('app_version', appVersion);
  if (appVersionBoundsError) {
    return errorResponse(c, 400, 'invalid_request', appVersionBoundsError);
  }

  const hardwareHashBoundsError = validatePublicInput(
    'hardware_hash',
    hardwareHash,
  );
  if (hardwareHashBoundsError) {
    return errorResponse(c, 400, 'invalid_request', hardwareHashBoundsError);
  }

  if (!isBase64Url(devicePublicKey, 32) || !isBase64Url(signature, 64)) {
    return errorResponse(
      c,
      400,
      'invalid_request',
      'device_public_key and signature must be base64url-encoded Ed25519 values',
    );
  }

  const now = new Date();
  await deleteExpiredChallengePreflightInstallations(c.env.BROKER_DB, {
    installationId,
    devicePublicKey,
    now,
  });

  const requestContext = {
    endpoint: 'POST /v1/trial/challenge/verify',
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

  const issuanceBrakeDecision = await checkActiveIssuanceBrake(
    c.env.BROKER_DB,
    currentEntitlement,
  );
  if (issuanceBrakeDecision) {
    return publicErrorResponse(c, issuanceBrakeDecision.status, {
      code: issuanceBrakeDecision.code,
      class: issuanceBrakeDecision.class,
      subcode: issuanceBrakeDecision.subcode,
      retryAfterMs: issuanceBrakeDecision.retryAfterMs,
      message: issuanceBrakeDecision.message,
      entitlement: currentEntitlement,
    });
  }

  const installation = await getInstallation(c.env.BROKER_DB, installationId);
  if (!installation || !installation.challenge || !installation.challenge_expires_at) {
    return errorResponse(
      c,
      404,
      'challenge_not_found',
      'no active challenge exists for installation_id',
    );
  }

  if (installation.device_public_key !== devicePublicKey) {
    return errorResponse(
      c,
      409,
      'device_public_key_mismatch',
      'verify must use the registered device_public_key for installation_id',
    );
  }

  const signedAtDate = parseIsoDate(signedAt);
  const challengeExpiresDate = parseIsoDate(challengeExpiresAt);
  if (!signedAtDate || !challengeExpiresDate) {
    return errorResponse(
      c,
      400,
      'invalid_request',
      'challenge_expires_at and signed_at must be valid ISO-8601 timestamps',
    );
  }

  if (
    installation.challenge !== challenge ||
    installation.challenge_expires_at !== challengeExpiresAt
  ) {
    return errorResponse(
      c,
      401,
      'challenge_invalid',
      'challenge and challenge_expires_at must match the active challenge',
    );
  }

  if (
    Math.abs(signedAtDate.getTime() - now.getTime()) >
    TRIAL_VERIFY_MAX_CLOCK_SKEW_SECONDS * 1000
  ) {
    return errorResponse(
      c,
      401,
      'signature_skew',
      'signed_at must be within ±60 seconds of broker time',
    );
  }

  if (challengeExpiresDate.getTime() < now.getTime()) {
    return errorResponse(
      c,
      410,
      'challenge_expired',
      'challenge has expired and must be reissued',
    );
  }

  const signatureIsValid = await verifyEd25519Signature({
    devicePublicKey,
    signature,
    payload: buildCanonicalVerifyPayload({
      installation_id: installationId,
      device_public_key: devicePublicKey,
      challenge,
      challenge_expires_at: challengeExpiresAt,
      hardware_hash: hardwareHash,
      app_version: appVersion,
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

  const entitlement = currentEntitlement;

  await recordRequestEvent(c.env.BROKER_DB, requestContext);

  const respondVerifyFailure = async (response: Response): Promise<Response> => {
    await recordVerifyOutcomeEvent(c.env.BROKER_DB, requestContext, 'fail');
    return response;
  };

  const respondVerifySuccess = async (response: Response): Promise<Response> => {
    await recordVerifyOutcomeEvent(c.env.BROKER_DB, requestContext, 'success');
    return response;
  };

  const rateLimitDecision = await checkEndpointRateLimit(c.env.BROKER_DB, requestContext);
  if (rateLimitDecision) {
    return respondVerifyFailure(
      publicErrorResponse(c, rateLimitDecision.status, {
        code: rateLimitDecision.code,
        class: rateLimitDecision.class,
        subcode: rateLimitDecision.subcode,
        retryAfterMs: rateLimitDecision.retryAfterMs,
        message: rateLimitDecision.message,
        entitlement,
      }),
    );
  }

  const velocityCapDecision = await checkVelocityCapHook(c.env.BROKER_DB, requestContext);
  if (velocityCapDecision) {
    return respondVerifyFailure(
      publicErrorResponse(c, velocityCapDecision.status, {
        code: velocityCapDecision.code,
        class: velocityCapDecision.class,
        subcode: velocityCapDecision.subcode,
        retryAfterMs: velocityCapDecision.retryAfterMs,
        message: velocityCapDecision.message,
        entitlement,
      }),
    );
  }

  if (entitlement && entitlement.status !== 'pending_release') {
    return respondVerifyFailure(releaseNotAllowedResponse(c, entitlement));
  }

  if (entitlement && isDiscordManagedPendingRelease(entitlement)) {
    return respondVerifyFailure(releaseNotAllowedResponse(c, entitlement));
  }

  const fingerprintSalt = await getFingerprintSaltConfig(c.env.BROKER_DB);
  const duplicateHardware = await hasConflictingHardwareDuplicate(c.env.BROKER_DB, {
    installationId,
    hardwareHash,
    challengeSaltVersion: installation.challenge_salt_version,
    currentSaltVersion: fingerprintSalt.current.version,
  });
  if (duplicateHardware) {
    return respondVerifyFailure(
      publicErrorResponse(c, 409, {
        code: 'trial_not_eligible',
        class: 'terminal',
        subcode: 'hardware_duplicate',
        message: 'hardware_hash is already reserved by another entitlement',
        entitlement,
      }),
    );
  }

  const issuanceCapDecision = await checkDailyIssuanceCap(
    c.env.BROKER_DB,
    now,
    entitlement,
  );
  if (issuanceCapDecision) {
    return respondVerifyFailure(
      publicErrorResponse(c, issuanceCapDecision.status, {
        code: issuanceCapDecision.code,
        class: issuanceCapDecision.class,
        subcode: issuanceCapDecision.subcode,
        retryAfterMs: issuanceCapDecision.retryAfterMs,
        message: issuanceCapDecision.message,
        entitlement,
      }),
    );
  }

  if (entitlement?.status === 'pending_release') {
    await clearPendingReleaseSessionTokenState(c.env.BROKER_DB, entitlement);
  }

  const challengeConsumed = await consumeChallenge(c.env.BROKER_DB, {
    installationId,
    devicePublicKey,
    challenge,
    challengeExpiresAt,
    challengeSaltVersion: installation.challenge_salt_version,
    hardwareHash,
    appVersion,
    lastSeenAt: now.toISOString(),
  });

  if (!challengeConsumed) {
    return respondVerifyFailure(
      errorResponse(
        c,
        409,
        'challenge_consumed',
        'challenge has already been consumed or replaced',
      ),
    );
  }

  const releaseToken = randomBase64Url(32);
  const releaseTokenExpiresAt = new Date(
    now.getTime() + TRIAL_RELEASE_TOKEN_TTL_SECONDS * 1000,
  ).toISOString();
  const releaseSessionRef = randomBase64Url(16);
  const releaseTokenHash = await sha256Base64Url(releaseToken);

  if (entitlement) {
    const updateResult = await c.env.BROKER_DB.prepare(
      `UPDATE openrouter_entitlements
          SET release_session_ref = ?,
              release_token_hash = ?,
              release_token_expires_at = ?,
              verified_hardware_hash = ?,
              verified_hardware_hash_salt_version = ?
        WHERE installation_id = ?
          AND status = ?
          AND (
            discord_issue_status IS NULL
            OR discord_issue_status NOT IN ('issuing', 'cleanup_required')
          )`,
    )
      .bind(
        releaseSessionRef,
        releaseTokenHash,
        releaseTokenExpiresAt,
        hardwareHash,
        installation.challenge_salt_version,
        installationId,
        'pending_release',
      )
      .run();

    if ((updateResult.meta.changes ?? 0) !== 1) {
      const currentEntitlement = await getEntitlement(c.env.BROKER_DB, installationId);
      if (currentEntitlement) {
        return respondVerifyFailure(
          releaseNotAllowedResponse(c, currentEntitlement),
        );
      }

      throw new Error('pending_release entitlement missing after challenge consumption');
    }
  } else {
    await c.env.BROKER_DB.prepare(
      `INSERT INTO openrouter_entitlements (
          installation_id,
          status,
          budget_usd,
          managed_credential_ref,
          issued_at,
          expires_at,
          release_session_ref,
          release_token_hash,
          release_token_expires_at,
          verified_hardware_hash,
          verified_hardware_hash_salt_version
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
    )
      .bind(
        installationId,
        'pending_release',
        MANAGED_TRIAL_BUDGET_POLICY.hardLimit,
        null,
        null,
        null,
        releaseSessionRef,
        releaseTokenHash,
        releaseTokenExpiresAt,
        hardwareHash,
        installation.challenge_salt_version,
      )
      .run();
  }

  const nextEntitlement = await getEntitlement(c.env.BROKER_DB, installationId);

  return respondVerifySuccess(
    c.json({
      release_token: releaseToken,
      release_token_expires_at: releaseTokenExpiresAt,
      ...normalizeManagedState(nextEntitlement),
    }),
  );
}

export async function handleTrialStatus(c: Context<BrokerEnv>): Promise<Response> {
  const installationId = stringValue(c.req.query('installation_id'));
  if (installationId === null) {
    return errorResponse(
      c,
      400,
      'invalid_request',
      'installation_id query parameter is required',
    );
  }

  const installationIdBoundsError = validatePublicInput(
    'installation_id',
    installationId,
  );
  if (installationIdBoundsError) {
    return errorResponse(c, 400, 'invalid_request', installationIdBoundsError);
  }

  const timestamp = nonEmptyString(c.req.header(TRIAL_STATUS_TIMESTAMP_HEADER));
  if (!timestamp) {
    return errorResponse(
      c,
      400,
      'invalid_request',
      `${TRIAL_STATUS_TIMESTAMP_HEADER} header is required`,
    );
  }

  const signature = nonEmptyString(c.req.header(TRIAL_STATUS_SIGNATURE_HEADER));
  if (!signature) {
    return errorResponse(
      c,
      400,
      'invalid_request',
      `${TRIAL_STATUS_SIGNATURE_HEADER} header is required`,
    );
  }

  const timestampDate = parseIsoDate(timestamp);
  if (!timestampDate) {
    return errorResponse(
      c,
      400,
      'invalid_request',
      `${TRIAL_STATUS_TIMESTAMP_HEADER} must be a valid ISO-8601 timestamp`,
    );
  }

  if (!isBase64Url(signature, 64)) {
    return errorResponse(
      c,
      400,
      'invalid_request',
      `${TRIAL_STATUS_SIGNATURE_HEADER} must be a base64url-encoded Ed25519 signature`,
    );
  }

  const now = new Date();
  await deleteExpiredChallengePreflightInstallations(c.env.BROKER_DB, {
    installationId,
    now,
  });

  const requestContext = {
    endpoint: 'GET /v1/trial/status',
    now,
    ip: resolveClientIp(c),
    networkIdentitySecrets: resolveRequestNetworkIdentitySecrets(c),
    installationId,
    hardwareHash: null,
  };
  const currentEntitlement = await getEntitlement(c.env.BROKER_DB, installationId);

  const subjectHook = await matchSubjectHook(c.env.BROKER_DB, requestContext);
  if (subjectHook && subjectHook.hookKind !== 'revocation') {
    return publicErrorResponse(c, subjectHook.status, {
      code: subjectHook.code,
      class: subjectHook.class,
      subcode: subjectHook.subcode,
      retryAfterMs: subjectHook.retryAfterMs,
      message: subjectHook.message,
      entitlement: currentEntitlement,
    });
  }

  const installation = await getInstallation(c.env.BROKER_DB, installationId);
  if (!installation) {
    return publicErrorResponse(c, 409, {
      code: 'trial_not_eligible',
      class: 'terminal',
      subcode: 'installation_not_found',
      message: 'installation_id is not registered with the broker',
      entitlement: currentEntitlement,
    });
  }

  if (
    Math.abs(timestampDate.getTime() - now.getTime()) >
    TRIAL_STATUS_MAX_CLOCK_SKEW_SECONDS * 1000
  ) {
    return errorResponse(
      c,
      401,
      'signature_skew',
      `${TRIAL_STATUS_TIMESTAMP_HEADER} must be within ±60 seconds of broker time`,
    );
  }

  const signatureIsValid = await verifyEd25519Signature({
    devicePublicKey: installation.device_public_key,
    signature,
    payload: buildCanonicalStatusPayload({
      installation_id: installationId,
      timestamp,
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

  const entitlement = await getEntitlement(c.env.BROKER_DB, installationId);

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

  const ownedReferralStatus = await bestEffortResolveOwnedReferralStatusForStatus(
    c.env.BROKER_DB,
    {
      installationId,
      nowIso: now.toISOString(),
    },
  );

  return c.json(
    normalizeTrialStatusResponse(
      entitlement,
      ownedReferralStatus?.referralCode.referral_id ?? null,
      ownedReferralStatus?.talkTogetherPass ?? null,
    ),
  );
}

type OwnedReferralStatusLookup = {
  referralCode: ReferralCodeRecord;
  talkTogetherPass: TalkTogetherPassStatusResponse | null;
};

async function bestEffortResolveOwnedReferralStatusForStatus(
  db: D1Database,
  input: {
    installationId: string;
    nowIso: string;
  },
): Promise<OwnedReferralStatusLookup | null> {
  try {
    const result = await ensureOwnedReferralIdForActiveDiscordManagedUser(db, input);
    if (!result.ok) {
      logTrialStatusOwnedReferralFailure({
        installationId: input.installationId,
        reason: result.reason,
      });
      return null;
    }

    try {
      return {
        referralCode: result.referralCode,
        talkTogetherPass: await resolveTalkTogetherPassStatusForOwnedReferralCode(
          db,
          result.referralCode,
        ),
      };
    } catch {
      logTrialStatusOwnedReferralFailure({
        installationId: input.installationId,
        reason: 'talk_together_pass_status_failed',
      });
      return {
        referralCode: result.referralCode,
        talkTogetherPass: null,
      };
    }
  } catch {
    logTrialStatusOwnedReferralFailure({
      installationId: input.installationId,
      reason: 'owned_referral_ensure_exception',
    });
    return null;
  }
}

function logTrialStatusOwnedReferralFailure(input: {
  installationId: string;
  reason: string;
}): void {
  if (input.reason === 'not_eligible') {
    return;
  }

  console.warn('owned_referral_status_failed', {
    endpoint: 'trial_status',
    installation_id: input.installationId,
    reason: normalizeTrialStatusOwnedReferralFailureReason(input.reason),
  });
}

function normalizeTrialStatusOwnedReferralFailureReason(reason: string): string {
  return /^[a-z0-9_:-]{1,64}$/u.test(reason)
    ? reason
    : 'owned_referral_ensure_exception';
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
  status: 400 | 401 | 404 | 409 | 410,
  code: string,
  message: string,
): Response {
  const normalized = normalizeLegacyTrialHandshakeError(code, message);

  return publicErrorResponse(c, status, normalized);
}

function releaseNotAllowedResponse(
  c: Context<BrokerEnv>,
  entitlement: OpenRouterEntitlementRecord,
): Response {
  return publicErrorResponse(c, 409, {
    code: 'trial_not_eligible',
    class: 'terminal',
    subcode: 'lifecycle_not_eligible',
    message:
      'verify may only mint release_token for lifecycle none or pending_release',
    entitlement,
  });
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

async function recordVerifyOutcomeEvent(
  db: D1Database,
  context: {
    now: Date;
    ip: string | null;
    installationId: string | null;
    hardwareHash: string | null;
  },
  outcome: 'success' | 'fail',
): Promise<void> {
  try {
    await recordRequestEvent(db, {
      endpoint:
        outcome === 'success'
          ? VERIFY_OUTCOME_SUCCESS_ENDPOINT
          : VERIFY_OUTCOME_FAIL_ENDPOINT,
      now: context.now,
      ip: context.ip,
      installationId: context.installationId,
      hardwareHash: context.hardwareHash,
    });
  } catch (error) {
    console.error('verify_outcome_event_record_failed', {
      installation_id: context.installationId,
      outcome,
      error_message: error instanceof Error ? error.message : String(error),
      broker_timestamp: new Date().toISOString(),
    });
  }
}

function randomBase64Url(byteLength: number): string {
  const bytes = new Uint8Array(byteLength);
  crypto.getRandomValues(bytes);
  return encodeBase64Url(bytes);
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

function buildCanonicalVerifyPayload(input: {
  installation_id: string;
  device_public_key: string;
  challenge: string;
  challenge_expires_at: string;
  hardware_hash: string;
  app_version: string;
  signed_at: string;
}): Uint8Array {
  return new TextEncoder().encode(
    TRIAL_VERIFY_SIGNATURE_PAYLOAD_FIELDS.map((field) => input[field]).join('\n'),
  );
}

function buildCanonicalStatusPayload(input: {
  installation_id: string;
  timestamp: string;
}): Uint8Array {
  return new TextEncoder().encode(
    TRIAL_STATUS_SIGNATURE_PAYLOAD_FIELDS.map((field) => input[field]).join('\n'),
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

async function consumeChallenge(
  db: D1Database,
  input: {
    installationId: string;
    devicePublicKey: string;
    challenge: string;
    challengeExpiresAt: string;
    challengeSaltVersion: number | null;
    hardwareHash: string;
    appVersion: string;
    lastSeenAt: string;
  },
): Promise<boolean> {
  const result = await db
    .prepare(
      `UPDATE installations
          SET hardware_hash = ?,
              hardware_hash_salt_version = ?,
              app_version = ?,
              challenge = NULL,
              challenge_expires_at = NULL,
              challenge_salt_version = NULL,
              last_seen_at = ?
        WHERE installation_id = ?
          AND device_public_key = ?
          AND challenge = ?
          AND challenge_expires_at = ?
          AND challenge_salt_version = ?`,
    )
    .bind(
      input.hardwareHash,
      input.challengeSaltVersion,
      input.appVersion,
      input.lastSeenAt,
      input.installationId,
      input.devicePublicKey,
      input.challenge,
      input.challengeExpiresAt,
      input.challengeSaltVersion,
    )
    .run();

  return (result.meta.changes ?? 0) === 1;
}

async function updateChallengeForInstallation(
  db: D1Database,
  input: {
    installationId: string;
    devicePublicKey: string;
    appVersion: string;
    challenge: string;
    challengeExpiresAt: string;
    challengeSaltVersion: number;
    nowIso: string;
    clearHardwareHash: boolean;
    currentInstallation: InstallationRecord;
  },
): Promise<boolean> {
  const updateResult = await db
    .prepare(
      `UPDATE installations
          SET hardware_hash = ${input.clearHardwareHash ? 'NULL' : 'hardware_hash'},
              hardware_hash_salt_version = ${input.clearHardwareHash ? 'NULL' : 'hardware_hash_salt_version'},
              app_version = ?,
              challenge = ?,
              challenge_expires_at = ?,
              challenge_salt_version = ?,
              last_seen_at = ?
        WHERE installation_id = ?
          AND device_public_key = ?
          AND hardware_hash IS ?
          AND hardware_hash_salt_version IS ?
          AND app_version = ?
          AND challenge IS ?
          AND challenge_expires_at IS ?
          AND challenge_salt_version IS ?
          AND last_seen_at = ?`,
    )
      .bind(
        input.appVersion,
        input.challenge,
        input.challengeExpiresAt,
        input.challengeSaltVersion,
        input.nowIso,
        input.installationId,
        input.devicePublicKey,
        input.currentInstallation.hardware_hash,
        input.currentInstallation.hardware_hash_salt_version,
        input.currentInstallation.app_version,
        input.currentInstallation.challenge,
        input.currentInstallation.challenge_expires_at,
        input.currentInstallation.challenge_salt_version,
        input.currentInstallation.last_seen_at,
      )
      .run();

  return (updateResult.meta.changes ?? 0) === 1;
}

async function clearPendingReleaseSessionTokenState(
  db: D1Database,
  entitlement: Pick<
    OpenRouterEntitlementRecord,
    | 'installation_id'
    | 'managed_credential_ref'
    | 'release_session_ref'
    | 'release_token_hash'
    | 'release_token_expires_at'
    | 'discord_issue_status'
  >,
): Promise<void> {
  if (
    entitlement.discord_issue_status === 'issuing' ||
    entitlement.discord_issue_status === 'cleanup_required'
  ) {
    return;
  }

  await db
    .prepare(
      `UPDATE openrouter_entitlements
          SET release_session_ref = NULL,
              release_token_hash = NULL,
              release_token_expires_at = NULL,
              managed_credential_ref = NULL,
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
          )
          AND release_session_ref IS ?
          AND release_token_hash IS ?
          AND release_token_expires_at IS ?
          AND managed_credential_ref IS ?`,
    )
    .bind(
      entitlement.installation_id,
      entitlement.release_session_ref,
      entitlement.release_token_hash,
      entitlement.release_token_expires_at,
      entitlement.managed_credential_ref,
    )
    .run();
}

async function resolveExistingInstallationChallengeReissue(
  db: D1Database,
  input: {
    installationId: string;
    devicePublicKey: string;
    appVersion: string;
    challenge: string;
    challengeExpiresAt: string;
    challengeSaltVersion: number;
    nowIso: string;
  },
): Promise<{
  challenge: string;
  challenge_expires_at: string;
}> {
  const latestInstallation = await getInstallation(db, input.installationId);

  if (
    latestInstallation &&
    latestInstallation.device_public_key === input.devicePublicKey &&
    latestInstallation.challenge &&
    latestInstallation.challenge_expires_at
  ) {
    return {
      challenge: latestInstallation.challenge,
      challenge_expires_at: latestInstallation.challenge_expires_at,
    };
  }

  if (
    !latestInstallation ||
    latestInstallation.device_public_key !== input.devicePublicKey
  ) {
    throw new Error('existing installation challenge reissue lost installation binding');
  }

  const latestEntitlement = await getEntitlement(db, input.installationId);
  if (latestEntitlement?.status === 'pending_release') {
    await clearPendingReleaseSessionTokenState(db, latestEntitlement);
  }

  const retriedUpdateSucceeded = await updateChallengeForInstallation(db, {
    ...input,
    clearHardwareHash:
      latestEntitlement === null || latestEntitlement.status === 'pending_release',
    currentInstallation: latestInstallation,
  });

  if (retriedUpdateSucceeded) {
    return {
      challenge: input.challenge,
      challenge_expires_at: input.challengeExpiresAt,
    };
  }

  const finalInstallation = await getInstallation(db, input.installationId);

  if (
    finalInstallation &&
    finalInstallation.device_public_key === input.devicePublicKey &&
    finalInstallation.challenge &&
    finalInstallation.challenge_expires_at
  ) {
    return {
      challenge: finalInstallation.challenge,
      challenge_expires_at: finalInstallation.challenge_expires_at,
    };
  }

  throw new Error('existing installation challenge reissue lost persisted challenge state');
}

function isUniqueConstraintError(error: unknown): boolean {
  return error instanceof Error && /unique constraint failed/i.test(error.message);
}

async function insertChallengeForInstallation(
  db: D1Database,
  input: {
    installationId: string;
    devicePublicKey: string;
    appVersion: string;
    challenge: string;
    challengeExpiresAt: string;
    challengeSaltVersion: number;
    nowIso: string;
  },
): Promise<void> {
  await db
    .prepare(
      `INSERT INTO installations (
          installation_id,
          device_public_key,
          app_version,
          challenge,
          challenge_expires_at,
          challenge_salt_version,
          created_at,
          last_seen_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
    )
    .bind(
      input.installationId,
      input.devicePublicKey,
      input.appVersion,
      input.challenge,
      input.challengeExpiresAt,
      input.challengeSaltVersion,
      input.nowIso,
      input.nowIso,
    )
    .run();
}

function toArrayBuffer(bytes: Uint8Array): ArrayBuffer {
  return bytes.buffer.slice(
    bytes.byteOffset,
    bytes.byteOffset + bytes.byteLength,
  ) as ArrayBuffer;
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

async function getInstallationByPublicKey(
  db: D1Database,
  devicePublicKey: string,
): Promise<InstallationRecord | null> {
  return db
    .prepare(
      `SELECT installation_id, device_public_key, hardware_hash, hardware_hash_salt_version,
              app_version, challenge, challenge_expires_at, challenge_salt_version,
              created_at, last_seen_at
         FROM installations
        WHERE device_public_key = ?`,
    )
    .bind(devicePublicKey)
    .first<InstallationRecord>();
}

function normalizeLegacyTrialHandshakeError(
  code: string,
  message: string,
): {
  code:
    | 'invalid_request'
    | 'challenge_expired'
    | 'challenge_invalid'
    | 'trial_not_eligible';
  class: 'retryable' | 'terminal' | 'security_fail';
  subcode?: string | null;
  retryAfterMs?: number | null;
  message: string;
} {
  switch (code) {
    case 'invalid_request':
      return {
        code: 'invalid_request',
        class: 'terminal',
        message,
      };
    case 'challenge_expired':
      return {
        code: 'challenge_expired',
        class: 'retryable',
        retryAfterMs: 0,
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
    case 'device_public_key_already_registered':
      return {
        code: 'trial_not_eligible',
        class: 'security_fail',
        subcode: 'device_public_key_registered',
        message,
      };
    case 'installation_not_found':
      return {
        code: 'trial_not_eligible',
        class: 'terminal',
        subcode: 'installation_not_found',
        message,
      };
    case 'challenge_not_found':
    case 'challenge_invalid':
    case 'challenge_consumed':
      return {
        code: 'challenge_invalid',
        class: 'security_fail',
        subcode: code === 'challenge_invalid' ? null : code,
        message,
      };
    default:
      return {
        code: 'trial_not_eligible',
        class: 'terminal',
        subcode: code,
        message,
      };
  }
}
