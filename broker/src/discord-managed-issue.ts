import type { Context } from 'hono';

import {
  checkActiveIssuanceBrake,
  checkEndpointRateLimit,
  getManagedDailyIssuanceCapState,
  checkVelocityCapHook,
  extractRequestNetworkMetadata,
  getBrokerAbuseControlsConfig,
  matchSubjectHook,
  recordRequestEvent,
  resolveClientIp,
  resolveRequestNetworkIdentitySecrets,
} from './abuse-controls';
import { ensureReferralSettlementJobsForDelivery, hasUnsettledReservedRewardWithoutJob } from './managed-referral-settlement';
import { resolveNetworkIdentityWriteMode, resolveReferralAttemptIdentity, resolveRequestNetworkIdentity, type NetworkIdentitySecrets } from './network-identity';
import { resolveBrokerRequestEventIpMatch } from './abuse-controls';
import { attachReferralToOperation, bindOperationForIssue, buildManagedOperationStatusBodyWithDelivery, failManagedOperationTerminal, getManagedOperationStatusSnapshot, markOperationActiveOnAck, type ManagedOperationRecord as StrictManagedOperationRecord, isManagedOperationId, listManagedOperationAttempts, markAttemptUnknown, operationBindingResponseBody, providerKeyNameForOperationAttempt, reconcileUnknownAttempt, recordAttemptCredential, saveOperationIssuanceContext, startManagedOperationAttempt, transitionManagedOperation, transitionOperationToPostCreateState, hasOtherLiveOperation, findConflictingOperationDelivery } from './managed-operation';
import {
  deliverManagedCleanupIncident,
  deliverImmediateMonitoringSideEffects,
  evaluateImmediateAbuseState,
  prepareIssueSuccessInsert,
  recordIssueSuccess,
} from './abuse-monitoring';
import {
  errorResponse as publicErrorResponse,
  internalErrorResponseWithEntitlement,
} from './broker-error';
import type { BrokerEnv } from './contract';
import {
  assertRedirectAllowed,
  buildDiscordAuthorizationUrl,
  deriveDiscordAccountCreatedAt,
  exchangeDiscordCode,
  fetchDiscordUser,
  generatePkcePair,
  parseDiscordRedirectAllowlist,
  type DiscordUserResponse,
} from './discord-oauth';
import { getFingerprintSaltConfig } from './fingerprint-salt';
import {
  normalizeManagedState,
  type TalkTogetherPassStatusResponse,
} from './managed-state';
import { createManagedKeyDelivery } from './managed-key-delivery';
import { nonEmptyString, stringValue, validatePublicInput } from './public-input';
import type {
  BrokerPendingDiscordOAuthSessionsConfig,
  DiscordOAuthSessionRecord,
  InstallationRecord,
  OpenRouterEntitlementRecord,
  ReferralCodeRecord,
} from './persistence';
import { scheduleManagedReferralSettlement } from './managed-referral-settlement';
import {
  assignManagedGuardrail,
  cleanupManagedChildKey,
  createManagedChildKey,
  isDefinitiveManagedChildKeyCreateRejection,
  OpenRouterManagementError,
} from './openrouter-management';
import { deriveManagedOpenRouterUserId } from './openrouter-user-id';
import {
  MANAGED_TRIAL_BUDGET_POLICY,
  MANAGED_TRIAL_POLICY,
  TRIAL_PROVIDER_POLICY,
} from './trial-policy';
import {
  ensureOwnedReferralIdForActiveDiscordManagedUser,
  getOperationReferralReward,
  markReservedIssueReferralFailed,
  normalizeReferralId,
  recordSkippedIssueReferralReward,
  reserveIssueReferralReward,
  resolveTalkTogetherPassStatusForOwnedReferralCode,
  getReservedReferralRewardIdForOperation,
  type IssueReferralReservationResult,
  type IssueReferralSkipReason,
  type ReferralAttemptIpDigest,
} from './referral';

export const DISCORD_OAUTH_SESSION_TTL_SECONDS = 300;

const DISCORD_AUTH_START_ENDPOINT = 'POST /v1/auth/discord/start';
const DISCORD_OPENROUTER_ISSUE_METHOD = 'POST';
const DISCORD_OPENROUTER_ISSUE_PATH = '/v1/providers/openrouter/discord/issue';
const DISCORD_ISSUE_MAX_CLOCK_SKEW_SECONDS = 60;
const DISCORD_ISSUE_REASON = 'llm_start';
const DISCORD_ACCOUNT_MIN_AGE_MS = 30 * 24 * 60 * 60 * 1000;
const MANAGED_TRIAL_ALLOWED_MODEL_SET = new Set<string>(
  TRIAL_PROVIDER_POLICY.managedFreeTrial.models,
);
const MANAGED_TRIAL_ALLOWED_MODEL_LIST =
  TRIAL_PROVIDER_POLICY.managedFreeTrial.models.join(', ');
const STRICT_ISO_8601_TIMESTAMP =
  /^(?<year>\d{4})-(?<month>0[1-9]|1[0-2])-(?<day>0[1-9]|[12]\d|3[01])T(?<hour>[01]\d|2[0-3]):(?<minute>[0-5]\d):(?<second>[0-5]\d)(?:\.(?<millisecond>\d{3}))?(?:(?<utc>Z)|(?<offsetSign>[+-])(?<offsetHour>[01]\d|2[0-3]):(?<offsetMinute>[0-5]\d))$/u;
const textEncoder = new TextEncoder();
const DISCORD_USER_REF_VERSION = 1;
const DISCORD_USER_REF_PREFIX = `ph-discord-user-v${DISCORD_USER_REF_VERSION}_`;
const DISCORD_USER_REF_PAYLOAD_PREFIX = `puripuly-heart:discord-user:v${DISCORD_USER_REF_VERSION}`;
const MANAGED_KEY_DELIVERY_ACK_TTL_MS = 15 * 60_000;

type DiscordReservationErrorSubcode =
  | 'discord_lifetime_used'
  | 'hardware_duplicate'
  | 'global_cap_reached'
  | 'discord_installation_already_issuing'
  | 'installation_binding_mismatch'
  | 'device_public_key_registered'
  | 'entitlement_reservation_failed';

type DiscordReservationResult =
  | { ok: true }
  | {
      ok: false;
      subcode: DiscordReservationErrorSubcode;
      retryAfterMs?: number | null;
    };

interface DiscordAuthStartRequestBody {
  installation_id?: unknown;
  device_public_key?: unknown;
  redirect_uri?: unknown;
  app_version?: unknown;
  referral_id?: unknown;
}

interface DiscordOpenRouterIssueRequestBody {
  code?: unknown;
  state?: unknown;
  installation_id?: unknown;
  device_public_key?: unknown;
  redirect_uri?: unknown;
  hardware_hash?: unknown;
  hardware_hash_salt_version?: unknown;
  app_version?: unknown;
  reason?: unknown;
  budget_usd?: unknown;
  model?: unknown;
  issue_nonce?: unknown;
  signed_at?: unknown;
  signature_alg?: unknown;
  signature?: unknown;
  delivery_ack_supported?: unknown;
  operation_id?: unknown;
  resume_token?: unknown;
}

interface DiscordOpenRouterIssueInput {
  code: string;
  state: string;
  installationId: string;
  devicePublicKey: string;
  redirectUri: string;
  hardwareHash: string;
  hardwareHashSaltVersion: number;
  appVersion: string;
  reason: string;
  budgetUsd: number;
  model: string;
  issueNonce: string;
  signedAt: string;
  signatureAlg: 'ed25519';
  signature: string;
  deliveryAckSupported: boolean;
  operationId: string | null;
  resumeToken: string | null;
}

interface DiscordEligibilityDecision {
  ok: boolean;
  discordAccountCreatedAt: string | null;
  discordEmailVerified: 0 | 1 | null;
  subcode?: 'discord_email_unverified' | 'discord_account_too_new' | 'discord_invalid_snowflake';
  message?: string;
}

class DiscordIssueSuccessMonitoringStateError extends Error {
  readonly issueSuccessRecorded: boolean;

  constructor(input: { cause: unknown; issueSuccessRecorded: boolean }) {
    super(
      input.cause instanceof Error
        ? input.cause.message
        : 'Discord issue-success monitoring state update failed',
    );
    this.name = 'DiscordIssueSuccessMonitoringStateError';
    this.issueSuccessRecorded = input.issueSuccessRecorded;
  }
}

export async function handleDiscordAuthStart(
  c: Context<BrokerEnv>,
): Promise<Response> {
  const body = await readJsonBody<DiscordAuthStartRequestBody>(c);
  if (!body.ok) {
    return invalidRequestBodyResponse(c, body.reason);
  }

  const installationId = stringValue(body.value.installation_id);
  const devicePublicKey = nonEmptyString(body.value.device_public_key);
  const redirectUri = stringValue(body.value.redirect_uri);
  const appVersion = stringValue(body.value.app_version);
  const referralId = normalizeReferralId(body.value.referral_id);

  if (!installationId || !devicePublicKey || !redirectUri || !appVersion) {
    return invalidRequestResponse(
      c,
      'installation_id, device_public_key, redirect_uri, and app_version are required',
    );
  }

  const installationIdBoundsError = validatePublicInput(
    'installation_id',
    installationId,
  );
  if (installationIdBoundsError) {
    return invalidRequestResponse(c, installationIdBoundsError);
  }

  const appVersionBoundsError = validatePublicInput('app_version', appVersion);
  if (appVersionBoundsError) {
    return invalidRequestResponse(c, appVersionBoundsError);
  }

  if (!isBase64Url(devicePublicKey, 32)) {
    return invalidRequestResponse(
      c,
      'device_public_key must be base64url-encoded Ed25519 public key bytes',
    );
  }

  let redirectAllowlist: string[];
  try {
    redirectAllowlist = parseDiscordRedirectAllowlist(
      c.env.DISCORD_REDIRECT_URI_ALLOWLIST,
    );
    assertRedirectAllowed(redirectUri, redirectAllowlist);
  } catch (error) {
    return invalidRequestResponse(
      c,
      error instanceof Error ? error.message : 'Discord redirect URI is invalid',
    );
  }

  const now = new Date();
  const existingInstallation = await getInstallation(c.env.BROKER_DB, installationId);
  const trustedInstallationId =
    existingInstallation && existingInstallation.device_public_key !== devicePublicKey
      ? null
      : installationId;
  const requestContext = {
    endpoint: DISCORD_AUTH_START_ENDPOINT,
    now,
    ip: resolveClientIp(c),
    networkIdentitySecrets: resolveRequestNetworkIdentitySecrets(c),
    installationId: trustedInstallationId,
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

  const controls = await getBrokerAbuseControlsConfig(c.env.BROKER_DB);
  const pendingControls = controls.pendingDiscordOAuthSessions;
  const pendingLimitDecision = await checkPendingDiscordOAuthIpLimit(
    c.env.BROKER_DB,
    requestContext,
    pendingControls,
  );
  if (pendingLimitDecision) {
    return pendingLimitDecision(c);
  }

  const bindingGateResponse = await discordInstallationBindingGuardResponse(c, {
    installationId,
    devicePublicKey,
    existingInstallation,
    entitlement: null,
  });
  if (bindingGateResponse) {
    return bindingGateResponse;
  }

  const fingerprintSalt = await getFingerprintSaltConfig(c.env.BROKER_DB);
  const pkce = await generatePkcePair();
  const state = randomBase64Url(32);
  const issueNonce = randomBase64Url(32);
  const expiresAt = new Date(
    now.getTime() + DISCORD_OAUTH_SESSION_TTL_SECONDS * 1000,
  ).toISOString();

  const insertSucceeded = await insertPendingDiscordOAuthSession(c.env.BROKER_DB, {
    stateHash: await sha256Base64Url(state),
    installationId,
    devicePublicKey,
    redirectUri,
    pkceCodeVerifier: pkce.codeVerifier,
    issueNonceHash: await sha256Base64Url(issueNonce),
    fingerprintSaltVersion: fingerprintSalt.current.version,
    referralId,
    nowIso: now.toISOString(),
    expiresAt,
    maxPendingPerInstallation: pendingControls.maxPerInstallation,
  });

  if (!insertSucceeded) {
    return pendingInstallationLimitResponse(c, pendingControls);
  }

  return c.json({
    authorization_url: buildDiscordAuthorizationUrl({
      clientId: c.env.DISCORD_CLIENT_ID,
      redirectUri,
      state,
      codeChallenge: pkce.codeChallenge,
    }),
    redirect_uri: redirectUri,
    oauth_session_expires_at: expiresAt,
    issue_nonce: issueNonce,
    fingerprint_salt: {
      version: fingerprintSalt.current.version,
      salt: fingerprintSalt.current.salt,
    },
    fingerprint_salt_version: fingerprintSalt.current.version,
  });
}

export async function handleDiscordOpenRouterIssue(
  c: Context<BrokerEnv>,
): Promise<Response> {
  const body = await readJsonBody<DiscordOpenRouterIssueRequestBody>(c);
  if (!body.ok) {
    return invalidRequestBodyResponse(c, body.reason);
  }

  const input = validateDiscordIssuePublicInput(c, body.value);
  if (!input.ok) {
    return input.response;
  }

  const now = new Date();
  const nowIso = now.toISOString();
  const requestContext = {
    endpoint: `${DISCORD_OPENROUTER_ISSUE_METHOD} ${DISCORD_OPENROUTER_ISSUE_PATH}`,
    now,
    ip: resolveClientIp(c),
    networkIdentitySecrets: resolveRequestNetworkIdentitySecrets(c),
    installationId: input.value.installationId,
    hardwareHash: input.value.hardwareHash,
  };
  const currentEntitlement = await getEntitlement(
    c.env.BROKER_DB,
    input.value.installationId,
  );

  const subjectHook = await matchSubjectHook(c.env.BROKER_DB, requestContext);
  if (subjectHook) {
    const hookEntitlement = await getEntitlement(
      c.env.BROKER_DB,
      input.value.installationId,
    );
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

  const stateHash = await sha256Base64Url(input.value.state);
  const codeHash = await sha256Base64Url(input.value.code);
  const issueNonceHash = await sha256Base64Url(input.value.issueNonce);
  const session = await getDiscordOAuthSession(c.env.BROKER_DB, stateHash);

  if (!session) {
    return discordStateUnknownResponse(c);
  }

  const sessionGateResponse = await validateDiscordSessionGate(c, {
    db: c.env.BROKER_DB,
    stateHash,
    session,
    input: input.value,
    issueNonceHash,
    now,
    nowIso,
  });
  if (sessionGateResponse) {
    return sessionGateResponse;
  }

  const bindingGateResponse = await discordInstallationBindingGuardResponse(c, {
    installationId: input.value.installationId,
    devicePublicKey: input.value.devicePublicKey,
    entitlement: currentEntitlement,
  });
  if (bindingGateResponse) {
    return bindingGateResponse;
  }

  const signedAtDate = parseIsoDate(input.value.signedAt);
  if (!signedAtDate) {
    return invalidRequestResponse(c, 'signed_at must be a valid ISO-8601 timestamp');
  }

  if (
    Math.abs(signedAtDate.getTime() - now.getTime()) >
    DISCORD_ISSUE_MAX_CLOCK_SKEW_SECONDS * 1000
  ) {
    return discordSignatureSkewResponse(c);
  }

  const signatureIsValid = await verifyEd25519Signature({
    devicePublicKey: input.value.devicePublicKey,
    signature: input.value.signature,
    payload: buildCanonicalDiscordIssuePayload({
      input: input.value,
      codeHash,
    }),
  });
  if (!signatureIsValid) {
    return discordSignatureMismatchResponse(c);
  }

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
      entitlement: currentEntitlement,
    });
  }

  const velocityCapDecision = await checkVelocityCapHook(
    c.env.BROKER_DB,
    requestContext,
  );
  if (velocityCapDecision) {
    return publicErrorResponse(c, velocityCapDecision.status, {
      code: velocityCapDecision.code,
      class: velocityCapDecision.class,
      subcode: velocityCapDecision.subcode,
      retryAfterMs: velocityCapDecision.retryAfterMs,
      message: velocityCapDecision.message,
      entitlement: currentEntitlement,
    });
  }

  const claimed = await claimDiscordOAuthSession(c.env.BROKER_DB, {
    stateHash,
    nowIso,
  });
  if (!claimed) {
    return discordSessionAlreadyProcessingResponse(c);
  }

  let discordUser: DiscordUserResponse;
  let discordTokenResponse: Awaited<ReturnType<typeof exchangeDiscordCode>> | null = null;
  try {
    discordTokenResponse = await exchangeDiscordCode({
      clientId: c.env.DISCORD_CLIENT_ID,
      clientSecret: c.env.DISCORD_CLIENT_SECRET,
      code: input.value.code,
      redirectUri: session.redirect_uri,
      codeVerifier: session.pkce_code_verifier!,
    });
    discordUser = await fetchDiscordUser({
      accessToken: discordTokenResponse.access_token,
    });
  } catch {
    await failDiscordOAuthSession(c.env.BROKER_DB, {
      stateHash,
      nowIso,
      discordEmailVerified: null,
      discordAccountCreatedAt: null,
    });
    return discordOAuthFailedResponse(c);
  }

  const eligibility = assertDiscordEligibility(discordUser, now);
  if (!eligibility.ok) {
    await failDiscordOAuthSession(c.env.BROKER_DB, {
      stateHash,
      nowIso,
      discordEmailVerified: eligibility.discordEmailVerified,
      discordAccountCreatedAt: eligibility.discordAccountCreatedAt,
    });
    return discordEligibilityErrorResponse(
      c,
      eligibility.subcode ?? 'discord_invalid_snowflake',
      eligibility.message ?? 'Discord account identity is invalid',
    );
  }

  const discordUserRef = await deriveDiscordUserRef({
    discordUserId: discordUser.id,
    secret: c.env.DISCORD_USER_REF_SECRET,
  });
  if (!discordUserRef) {
    await failDiscordOAuthSession(c.env.BROKER_DB, {
      stateHash,
      nowIso,
      discordEmailVerified: eligibility.discordEmailVerified,
      discordAccountCreatedAt: eligibility.discordAccountCreatedAt,
    });
    return internalErrorResponseWithEntitlement(c, null);
  }

  await markDiscordOAuthSessionEligible(c.env.BROKER_DB, {
    stateHash,
    nowIso,
    discordUserRef,
    discordEmailVerified: eligibility.discordEmailVerified,
    discordAccountCreatedAt: eligibility.discordAccountCreatedAt,
  });

  let referralReservation: IssueReferralReservationResult | null = null;
  let boundOperation: StrictManagedOperationRecord | null = null;
  let boundAttemptIndex: number | null = null;
  const attemptIpDigest = await resolveRequestNetworkIdentity(
    requestContext.ip,
    resolveRequestNetworkIdentitySecrets(c),
    now,
  );
  const attemptIdentity = await resolveReferralAttemptIdentity(
    requestContext.ip,
    resolveRequestNetworkIdentitySecrets(c),
    now,
  );
  const operationBinding = input.value.operationId
    ? await bindOperationForIssue(c.env.BROKER_DB, {
        operationId: input.value.operationId,
        resumeToken: input.value.resumeToken,
        issueSource: 'discord',
        subjectRef: discordUserRef,
        installationId: input.value.installationId,
        devicePublicKey: input.value.devicePublicKey,
        now,
      })
    : null;
  if (operationBinding && operationBinding.status !== 'proceed') {
    if (operationBinding.status === 'invalid') {
      return invalidRequestResponse(c, `managed operation binding ${operationBinding.reason}`);
    }
    const attempts = await listManagedOperationAttempts(c.env.BROKER_DB, operationBinding.operation.operation_id);
    return c.json(operationBindingResponseBody(operationBinding.operation, attempts));
  }
  boundOperation = operationBinding && operationBinding.status === 'proceed' ? operationBinding.operation : null;
  if (boundOperation) {
    const deliveryConflict = await findConflictingOperationDelivery(c.env.BROKER_DB, {
      issueSource: 'discord',
      subjectRef: discordUserRef,
      installationId: input.value.installationId,
      excludeOperationId: boundOperation.operation_id,
    });
    if (deliveryConflict) {
      const blocking = await getManagedOperationStatusSnapshot(c.env.BROKER_DB, deliveryConflict.operationId);
      if (blocking) {
        const blockingAttempts = await listManagedOperationAttempts(c.env.BROKER_DB, deliveryConflict.operationId);
        return c.json(operationBindingResponseBody(blocking, blockingAttempts));
      }
    }
    const started = await startManagedOperationAttempt(c.env.BROKER_DB, boundOperation, now);
    if (!started.ok) {
      const attempts = await listManagedOperationAttempts(c.env.BROKER_DB, boundOperation.operation_id);
      const current = await getManagedOperationStatusSnapshot(c.env.BROKER_DB, boundOperation.operation_id);
      return c.json(operationBindingResponseBody(current ?? boundOperation, attempts));
    }
    boundAttemptIndex = started.attempt.attempt_index;
    await saveOperationIssuanceContext(
      c.env.BROKER_DB,
      boundOperation.operation_id,
      {
        hardwareHash: input.value.hardwareHash,
        hardwareHashSaltVersion: input.value.hardwareHashSaltVersion,
        appVersion: input.value.appVersion,
      },
      now,
    );
  }
  const reservation = await reserveDiscordEntitlement(c.env.BROKER_DB, {
    installationId: input.value.installationId,
    devicePublicKey: input.value.devicePublicKey,
    hardwareHash: input.value.hardwareHash,
    hardwareHashSaltVersion: input.value.hardwareHashSaltVersion,
    appVersion: input.value.appVersion,
    discordUserRef,
    now,
    nowIso,
  });
  if (!reservation.ok) {
    await bestEffortRecordIneligibleIssueReferralSkip(c.env.BROKER_DB, {
      referralId: session.referral_id,
      referredDiscordUserRef: discordUserRef,
      referredInstallationId: input.value.installationId,
      referredHardwareHash: input.value.hardwareHash,
      referredHardwareHashSaltVersion: input.value.hardwareHashSaltVersion,
      skipReason: issueReferralSkipReasonForReservationFailure(
        currentEntitlement,
        reservation.subcode,
      ),
      attemptIpDigest,
      attemptIpLegacyHash: attemptIdentity.legacyHash,
      operationId: boundOperation?.operation_id ?? null,
      nowIso,
    });
    await failDiscordOAuthSession(c.env.BROKER_DB, {
      stateHash,
      nowIso,
      discordEmailVerified: eligibility.discordEmailVerified,
      discordAccountCreatedAt: eligibility.discordAccountCreatedAt,
    });
    return discordReservationErrorResponse(c, reservation);
  }

  referralReservation = await bestEffortReserveIssueReferralReward(c.env.BROKER_DB, {
    referralId: session.referral_id,
    referredDiscordUserRef: discordUserRef,
    referredInstallationId: input.value.installationId,
    referredHardwareHash: input.value.hardwareHash,
    referredHardwareHashSaltVersion: input.value.hardwareHashSaltVersion,
    attemptIpDigest,
    attemptIpLegacyHash: attemptIdentity.legacyHash,
    operationId: boundOperation?.operation_id ?? null,
    nowIso,
  });

  if (boundOperation) {
    const rewardId = await getReservedReferralRewardIdForOperation(c.env.BROKER_DB, boundOperation.operation_id);
    await attachReferralToOperation(
      c.env.BROKER_DB,
      boundOperation.operation_id,
      rewardId,
      referralReservation?.outcome === 'reserved' ? 'reserved' : referralReservation?.outcome === 'skipped' ? 'skipped' : 'none',
      'none',
      now,
    );
  }
  const issuedAt = nowIso;
  const expiresAt = addMonthsUtc(
    now,
    MANAGED_TRIAL_POLICY.entitlement.issuance.expiry.durationMonths,
  ).toISOString();
  const issueLimitUsd = resolveReferredIssueLimitUsd(referralReservation);
  const referralLimitVerificationRequired = referralReservation?.outcome === 'reserved';
  let childKey: { rawKey: string; hash: string } | null = null;
  let childKeyCreationAttempted = false;
  let issueSuccessRecorded = false;
  try {
    childKeyCreationAttempted = true;
    childKey = await createManagedChildKey({
      managementApiKey: c.env.OPENROUTER_MANAGEMENT_API_KEY,
      installationId: input.value.installationId,
      releaseSessionRef: stateHash,
      expiresAt,
      limitUsd: issueLimitUsd,
      requireEffectiveLimitVerification: referralLimitVerificationRequired,
      ...(boundOperation && boundAttemptIndex !== null
        ? {
            keyName: providerKeyNameForOperationAttempt(
              boundOperation.operation_id,
              'discord',
              boundAttemptIndex,
            ),
          }
        : {}),
    });
    if (boundOperation && boundAttemptIndex !== null) {
      await recordAttemptCredential(c.env.BROKER_DB, boundOperation.operation_id, boundAttemptIndex, childKey.hash, now);
    }
    await assignManagedGuardrail({
      managementApiKey: c.env.OPENROUTER_MANAGEMENT_API_KEY,
      guardrailId: c.env.OPENROUTER_MANAGED_GUARDRAIL_ID,
      keyHash: childKey.hash,
    });

    if (input.value.deliveryAckSupported) {
      const pending = await markDiscordReservationDeliveryPending(c.env.BROKER_DB, {
        stateHash,
        installationId: input.value.installationId,
        devicePublicKey: input.value.devicePublicKey,
        discordUserRef,
        managedCredentialRef: childKey.hash,
        issuedAt,
        expiresAt,
        budgetUsd: issueLimitUsd,
        nowIso,
      });
      if (!pending) {
        throw new Error('Discord managed entitlement delivery-pending transition failed');
      }
      const delivery = await createManagedKeyDelivery(c.env.BROKER_DB, {
        issueSource: 'discord',
        subjectRef: discordUserRef,
        installationId: input.value.installationId,
        managedCredentialRef: childKey.hash,
        createdAt: now,
        expiresAt: new Date(now.getTime() + MANAGED_KEY_DELIVERY_ACK_TTL_MS),
        operationId: boundOperation?.operation_id ?? null,
        attemptIndex: boundAttemptIndex,
      });
      if (boundOperation) {
        const settled = await transitionOperationToPostCreateState(c.env.BROKER_DB, c.env.OPENROUTER_MANAGEMENT_API_KEY, boundOperation.operation_id, 'DELIVERY_PENDING', now);
        if (!settled || settled.state !== 'DELIVERY_PENDING') {
          throw new Error('Discord managed operation delivery-pending transition failed');
        }
      }
      return c.json({
        openrouter_api_key: childKey.rawKey,
        managed_credential_ref: childKey.hash,
        expires_at: expiresAt,
        delivery_ack_required: true,
        delivery_id: delivery.deliveryId,
        delivery_ack_token: delivery.deliveryAckToken,
        delivery_ack_expires_at: new Date(
          now.getTime() + MANAGED_KEY_DELIVERY_ACK_TTL_MS,
        ).toISOString(),
      });
    }

    const activationSucceeded = await activateDiscordReservation(c.env.BROKER_DB, {
      stateHash,
      installationId: input.value.installationId,
      devicePublicKey: input.value.devicePublicKey,
      discordUserRef,
      managedCredentialRef: childKey.hash,
      issuedAt,
      expiresAt,
      budgetUsd: issueLimitUsd,
      deliveredAt: nowIso,
    });
    if (boundOperation) {
      const settled = await transitionOperationToPostCreateState(c.env.BROKER_DB, c.env.OPENROUTER_MANAGEMENT_API_KEY, boundOperation.operation_id, 'ACTIVE', now);
      if (!settled || settled.state !== 'ACTIVE') {
        throw new Error('Discord managed operation activation transition failed');
      }
    }

    const activeEntitlement = await getEntitlement(
      c.env.BROKER_DB,
      input.value.installationId,
    );
    if (!activeEntitlement) {
      throw new Error('Discord managed entitlement missing after activation');
    }
    assertDiscordIssueEntitlementDeliverable(activeEntitlement);

    await runDiscordIssueSuccessMonitoring(c, {
      installationId: input.value.installationId,
      managedCredentialRef: activeEntitlement.managed_credential_ref!,
      issuedAt,
      now,
      sensitiveValues: collectDiscordIssueSensitiveValues({
        input: input.value,
        session,
        discordTokenResponse,
        discordUser,
        childKey,
      }),
    });
    issueSuccessRecorded = true;

    const ownedReferralStatus = await bestEffortResolveOwnedReferralStatusForIssueResponse(
      c.env.BROKER_DB,
      {
        installationId: input.value.installationId,
        nowIso,
      },
    );
    await scheduleDirectIssueReferralSettlement(c.env.BROKER_DB, {
      referralReservation,
      referredDiscordUserRef: discordUserRef,
      referredInstallationId: input.value.installationId,
      now,
    });

    return await discordIssueSuccessResponse(c, {
      entitlement: activeEntitlement,
      rawKey: childKey.rawKey,
      model: input.value.model,
      installationId: input.value.installationId,
      referralId: ownedReferralStatus?.referralCode.referral_id ?? null,
      talkTogetherPass: ownedReferralStatus?.talkTogetherPass ?? null,
      referralBonusApplied: false,
    });
  } catch (error) {
    if (
      !childKey &&
      error instanceof OpenRouterManagementError &&
      error.createdChildKey
    ) {
      childKey = error.createdChildKey;
    }
    const sensitiveValues = collectDiscordIssueSensitiveValues({
      input: input.value,
      session,
      discordTokenResponse,
      discordUser,
      childKey,
    });
    await bestEffortFailDiscordOAuthSession(c.env.BROKER_DB, {
      stateHash,
      nowIso,
      discordEmailVerified: eligibility.discordEmailVerified,
      discordAccountCreatedAt: eligibility.discordAccountCreatedAt,
      sensitiveValues,
    });
    if (!boundOperation) {
      await bestEffortMarkIssueReferralReservationFailed(c.env.BROKER_DB, {
        referralReservation,
        referredDiscordUserRef: discordUserRef,
        referredInstallationId: input.value.installationId,
        nowIso,
      });
    }
    if (boundOperation && boundAttemptIndex !== null) {
      if (childKey) {
        await recordAttemptCredential(c.env.BROKER_DB, boundOperation.operation_id, boundAttemptIndex, childKey.hash, now);
      }
      await markAttemptUnknown(c.env.BROKER_DB, boundOperation.operation_id, boundAttemptIndex, now);
    }
    let boundProviderCleanupVerified = false;
    if (boundOperation && isDefinitiveManagedChildKeyCreateRejection(error)) {
      await failManagedOperationTerminal(c.env.BROKER_DB, boundOperation, now, 'terminal_provider_failure');
    } else if (boundOperation && boundAttemptIndex !== null) {
      const reconciled = await reconcileUnknownAttempt(c.env.BROKER_DB, c.env.OPENROUTER_MANAGEMENT_API_KEY, boundOperation, now);
      boundProviderCleanupVerified = reconciled?.state === 'RETRY_READY' || reconciled?.state === 'CLEAN';
    }
    if (!childKey) {
      const definitiveCreateRejection =
        isDefinitiveManagedChildKeyCreateRejection(error);
      if (childKeyCreationAttempted && !definitiveCreateRejection && !boundProviderCleanupVerified) {
        let cleanupRequiredRecorded = false;
        try {
          cleanupRequiredRecorded = await markDiscordIndeterminateChildKeyCreation(
            c.env.BROKER_DB,
            {
              installationId: input.value.installationId,
              discordUserRef,
              operationId: boundOperation?.operation_id ?? null,
              nowIso,
            },
          );
        } catch {
          cleanupRequiredRecorded = false;
        }
        await deliverManagedCleanupIncident(c.env, {
          issueSource: 'discord',
          managedCredentialRef: null,
          phase: 'managed_issue',
          cleanupRequiredRecorded,
          occurredAt: nowIso,
        });
      } else {
        await releaseDiscordReservation(c.env.BROKER_DB, {
          installationId: input.value.installationId,
          discordUserRef,
        });
      }
    } else {
      if (
        (error instanceof DiscordIssueSuccessMonitoringStateError &&
          error.issueSuccessRecorded) ||
        issueSuccessRecorded
      ) {
        await bestEffortDeleteDiscordIssueSuccessRecord(c.env.BROKER_DB, {
          installationId: input.value.installationId,
          managedCredentialRef: childKey.hash,
          observedAt: issuedAt,
          sensitiveValues,
        });
      }
      await handleDiscordManagedChildKeyFailure(c, {
        installationId: input.value.installationId,
        releaseSessionRef: stateHash,
        discordUserRef,
        childKey,
        nowIso,
        error,
        sensitiveValues,
        providerCleanupHandled: boundProviderCleanupVerified,
        operationId: boundOperation?.operation_id ?? null,
      });
    }
    return internalErrorResponseWithEntitlement(
      c,
      await getEntitlement(c.env.BROKER_DB, input.value.installationId),
    );
  }
}

function validateDiscordIssuePublicInput(
  c: Context<BrokerEnv>,
  body: DiscordOpenRouterIssueRequestBody,
):
  | { ok: true; value: DiscordOpenRouterIssueInput }
  | { ok: false; response: Response } {
  const code = nonEmptyString(body.code);
  const state = nonEmptyString(body.state);
  const installationId = stringValue(body.installation_id);
  const devicePublicKey = nonEmptyString(body.device_public_key);
  const redirectUri = nonEmptyString(body.redirect_uri);
  const hardwareHash = nonEmptyString(body.hardware_hash);
  const hardwareHashSaltVersion =
    typeof body.hardware_hash_salt_version === 'number'
      ? body.hardware_hash_salt_version
      : null;
  const appVersion = stringValue(body.app_version);
  const reason = nonEmptyString(body.reason);
  const budgetUsd = typeof body.budget_usd === 'number' ? body.budget_usd : null;
  const model = nonEmptyString(body.model);
  const issueNonce = nonEmptyString(body.issue_nonce);
  const signedAt = nonEmptyString(body.signed_at);
  const signatureAlg = stringValue(body.signature_alg);
  const signature = nonEmptyString(body.signature);
  const deliveryAckSupported = body.delivery_ack_supported === true;
  const operationId = typeof body.operation_id === 'string' ? body.operation_id : null;
  const resumeToken = typeof body.resume_token === 'string' ? body.resume_token : null;
  if (operationId !== null && !isManagedOperationId(operationId)) {
    return {
      ok: false,
      response: invalidRequestResponse(c, 'operation_id must be a ph-mop-v1_ operation identity'),
    };
  }
  if ((operationId === null) !== (resumeToken === null)) {
    return {
      ok: false,
      response: invalidRequestResponse(c, 'operation_id and resume_token must be provided together'),
    };
  }

  if (
    !code ||
    !state ||
    !installationId ||
    !devicePublicKey ||
    !redirectUri ||
    !hardwareHash ||
    hardwareHashSaltVersion === null ||
    !appVersion ||
    !reason ||
    budgetUsd === null ||
    !model ||
    !issueNonce ||
    !signedAt ||
    !signatureAlg ||
    !signature
  ) {
    return {
      ok: false,
      response: invalidRequestResponse(
        c,
        'code, state, installation_id, device_public_key, redirect_uri, hardware_hash, hardware_hash_salt_version, app_version, reason, budget_usd, model, issue_nonce, signed_at, signature_alg, and signature are required',
      ),
    };
  }

  const installationIdBoundsError = validatePublicInput(
    'installation_id',
    installationId,
  );
  if (installationIdBoundsError) {
    return { ok: false, response: invalidRequestResponse(c, installationIdBoundsError) };
  }

  const hardwareHashBoundsError = validatePublicInput('hardware_hash', hardwareHash);
  if (hardwareHashBoundsError) {
    return { ok: false, response: invalidRequestResponse(c, hardwareHashBoundsError) };
  }

  const appVersionBoundsError = validatePublicInput('app_version', appVersion);
  if (appVersionBoundsError) {
    return { ok: false, response: invalidRequestResponse(c, appVersionBoundsError) };
  }

  for (const [field, value] of [
    ['code', code],
    ['state', state],
    ['redirect_uri', redirectUri],
    ['issue_nonce', issueNonce],
    ['signed_at', signedAt],
  ] as const) {
    const fieldError = validateDiscordIssueTextField(field, value);
    if (fieldError) {
      return { ok: false, response: invalidRequestResponse(c, fieldError) };
    }
  }

  if (!isBase64Url(devicePublicKey, 32) || !isBase64Url(signature, 64)) {
    return {
      ok: false,
      response: invalidRequestResponse(
        c,
        'device_public_key and signature must be base64url-encoded Ed25519 contract values',
      ),
    };
  }

  if (!Number.isSafeInteger(hardwareHashSaltVersion) || hardwareHashSaltVersion < 0) {
    return {
      ok: false,
      response: invalidRequestResponse(
        c,
        'hardware_hash_salt_version must be a non-negative integer',
      ),
    };
  }

  if (signatureAlg !== 'ed25519') {
    return {
      ok: false,
      response: invalidRequestResponse(c, 'signature_alg must be ed25519'),
    };
  }

  if (reason !== DISCORD_ISSUE_REASON) {
    return { ok: false, response: invalidRequestResponse(c, 'reason must be llm_start') };
  }

  if (budgetUsd !== MANAGED_TRIAL_BUDGET_POLICY.hardLimit) {
    return {
      ok: false,
      response: invalidRequestResponse(
        c,
        `budget_usd must equal ${MANAGED_TRIAL_BUDGET_POLICY.hardLimit}`,
      ),
    };
  }

  if (!MANAGED_TRIAL_ALLOWED_MODEL_SET.has(model)) {
    return {
      ok: false,
      response: invalidRequestResponse(
        c,
        `model must be one of ${MANAGED_TRIAL_ALLOWED_MODEL_LIST}`,
      ),
    };
  }

  return {
    ok: true,
    value: {
      code,
      state,
      installationId,
      devicePublicKey,
      redirectUri,
      hardwareHash,
      hardwareHashSaltVersion,
      appVersion,
      reason,
      budgetUsd,
      model,
      issueNonce,
      signedAt,
      signatureAlg: 'ed25519',
      signature,
      deliveryAckSupported,
      operationId,
      resumeToken,
    },
  };
}

function validateDiscordIssueTextField(field: string, value: string): string | null {
  if (value.trim().length === 0) {
    return `${field} must not be blank or whitespace-only`;
  }

  if (Array.from(value).length > 2048) {
    return `${field} must be at most 2048 characters`;
  }

  if (/[\p{Cc}\r\n\u0085\u2028\u2029]/u.test(value)) {
    return `${field} must not contain control characters or newlines`;
  }

  return null;
}

async function validateDiscordSessionGate(
  c: Context<BrokerEnv>,
  input: {
    db: D1Database;
    stateHash: string;
    session: DiscordOAuthSessionRecord;
    input: DiscordOpenRouterIssueInput;
    issueNonceHash: string;
    now: Date;
    nowIso: string;
  },
): Promise<Response | null> {
  const expiresAt = parseIsoDate(input.session.expires_at);
  if (
    input.session.status === 'expired' ||
    !expiresAt ||
    expiresAt.getTime() < input.now.getTime()
  ) {
    await expireDiscordOAuthSession(input.db, {
      stateHash: input.stateHash,
      nowIso: input.nowIso,
    });
    return discordSessionExpiredResponse(c);
  }

  if (input.session.status === 'processing') {
    return discordSessionAlreadyProcessingResponse(c);
  }

  if (input.session.status !== 'pending') {
    await clearTerminalDiscordOAuthSessionVerifier(input.db, {
      stateHash: input.stateHash,
    });
    return discordSessionTerminalResponse(c, input.session.status);
  }

  if (!input.session.pkce_code_verifier) {
    return discordSessionTerminalResponse(c, 'failed');
  }

  if (
    input.session.installation_id !== input.input.installationId ||
    input.session.device_public_key !== input.input.devicePublicKey ||
    input.session.redirect_uri !== input.input.redirectUri ||
    input.session.issue_nonce_hash !== input.issueNonceHash
  ) {
    return discordSessionBindingMismatchResponse(c);
  }

  if (
    Number(input.session.fingerprint_salt_version) !==
    input.input.hardwareHashSaltVersion
  ) {
    return discordHardwareSaltMismatchResponse(c);
  }

  return null;
}

async function discordInstallationBindingGuardResponse(
  c: Context<BrokerEnv>,
  input: {
    installationId: string;
    devicePublicKey: string;
    existingInstallation?: InstallationRecord | null;
    entitlement: OpenRouterEntitlementRecord | null;
  },
): Promise<Response | null> {
  const conflict = await getDiscordInstallationBindingConflict(c.env.BROKER_DB, input);
  if (!conflict) {
    return null;
  }

  return discordInstallationBindingErrorResponse(c, conflict, input.entitlement);
}

async function getDiscordInstallationBindingConflict(
  db: D1Database,
  input: {
    installationId: string;
    devicePublicKey: string;
    existingInstallation?: InstallationRecord | null;
  },
): Promise<'installation_binding_mismatch' | 'device_public_key_registered' | null> {
  const installation =
    input.existingInstallation !== undefined
      ? input.existingInstallation
      : await getInstallation(db, input.installationId);
  if (installation && installation.device_public_key !== input.devicePublicKey) {
    return 'installation_binding_mismatch';
  }

  const installationByPublicKey = await getInstallationByPublicKey(
    db,
    input.devicePublicKey,
  );
  if (
    installationByPublicKey &&
    installationByPublicKey.installation_id !== input.installationId
  ) {
    return 'device_public_key_registered';
  }

  return null;
}

async function getDiscordOAuthSession(
  db: D1Database,
  stateHash: string,
): Promise<DiscordOAuthSessionRecord | null> {
  return db
    .prepare(
      `SELECT state_hash,
              installation_id,
              device_public_key,
              redirect_uri,
              pkce_code_verifier,
              issue_nonce_hash,
              fingerprint_salt_version,
              discord_user_ref,
              discord_email_verified,
              discord_account_created_at,
              eligibility_checked_at,
              status,
              created_at,
              expires_at,
              processing_started_at,
              consumed_at,
              referral_id
         FROM discord_oauth_sessions
        WHERE state_hash = ?`,
    )
    .bind(stateHash)
    .first<DiscordOAuthSessionRecord>();
}

async function claimDiscordOAuthSession(
  db: D1Database,
  input: {
    stateHash: string;
    nowIso: string;
  },
): Promise<boolean> {
  const result = await db
    .prepare(
      `UPDATE discord_oauth_sessions
          SET status = 'processing',
              processing_started_at = ?
        WHERE state_hash = ?
          AND status = 'pending'`,
    )
    .bind(input.nowIso, input.stateHash)
    .run();

  return Number(result.meta.changes ?? 0) === 1;
}

async function failDiscordOAuthSession(
  db: D1Database,
  input: {
    stateHash: string;
    nowIso: string;
    discordEmailVerified: 0 | 1 | null;
    discordAccountCreatedAt: string | null;
  },
): Promise<void> {
  await db
    .prepare(
      `UPDATE discord_oauth_sessions
          SET status = 'failed',
              pkce_code_verifier = NULL,
              discord_email_verified = ?,
              discord_account_created_at = ?,
              eligibility_checked_at = ?
        WHERE state_hash = ?`,
    )
    .bind(
      input.discordEmailVerified,
      input.discordAccountCreatedAt,
      input.nowIso,
      input.stateHash,
    )
    .run();
}

async function bestEffortFailDiscordOAuthSession(
  db: D1Database,
  input: {
    stateHash: string;
    nowIso: string;
    discordEmailVerified: 0 | 1 | null;
    discordAccountCreatedAt: string | null;
    sensitiveValues: string[];
  },
): Promise<void> {
  try {
    await failDiscordOAuthSession(db, input);
  } catch (error) {
    console.error('discord_oauth_session_fail_cleanup_failed', {
      state_hash: input.stateHash,
      failure: redactSensitiveDiagnostics(
        normalizeFailureForLog(error),
        input.sensitiveValues,
      ),
      broker_timestamp: new Date().toISOString(),
    });
  }
}

async function expireDiscordOAuthSession(
  db: D1Database,
  input: {
    stateHash: string;
    nowIso: string;
  },
): Promise<void> {
  await db
    .prepare(
      `UPDATE discord_oauth_sessions
          SET status = 'expired',
              pkce_code_verifier = NULL
        WHERE state_hash = ?
          AND status IN ('pending', 'expired')`,
    )
    .bind(input.stateHash)
    .run();
  void input.nowIso;
}

async function clearTerminalDiscordOAuthSessionVerifier(
  db: D1Database,
  input: {
    stateHash: string;
  },
): Promise<void> {
  await db
    .prepare(
      `UPDATE discord_oauth_sessions
          SET pkce_code_verifier = NULL
        WHERE state_hash = ?
          AND status IN ('consumed', 'canceled', 'failed', 'expired')`,
    )
    .bind(input.stateHash)
    .run();
}

async function markDiscordOAuthSessionEligible(
  db: D1Database,
  input: {
    stateHash: string;
    nowIso: string;
    discordUserRef: string;
    discordEmailVerified: 0 | 1 | null;
    discordAccountCreatedAt: string | null;
  },
): Promise<void> {
  await db
    .prepare(
      `UPDATE discord_oauth_sessions
          SET pkce_code_verifier = NULL,
              discord_user_ref = ?,
              discord_email_verified = ?,
              discord_account_created_at = ?,
              eligibility_checked_at = ?
        WHERE state_hash = ?
          AND status = 'processing'`,
    )
    .bind(
      input.discordUserRef,
      input.discordEmailVerified,
      input.discordAccountCreatedAt,
      input.nowIso,
      input.stateHash,
    )
    .run();
}

function assertDiscordEligibility(
  user: DiscordUserResponse,
  now: Date,
): DiscordEligibilityDecision {
  if (user.verified !== true) {
    return {
      ok: false,
      discordAccountCreatedAt: null,
      discordEmailVerified: user.verified === false ? 0 : null,
      subcode: 'discord_email_unverified',
      message: 'Discord email verification is required',
    };
  }

  let discordAccountCreatedAt: string;
  try {
    discordAccountCreatedAt = deriveDiscordAccountCreatedAt(user.id);
  } catch {
    return {
      ok: false,
      discordAccountCreatedAt: null,
      discordEmailVerified: 1,
      subcode: 'discord_invalid_snowflake',
      message: 'Discord account identity is invalid',
    };
  }

  const createdAt = parseIsoDate(discordAccountCreatedAt);
  if (!createdAt) {
    return {
      ok: false,
      discordAccountCreatedAt: null,
      discordEmailVerified: 1,
      subcode: 'discord_invalid_snowflake',
      message: 'Discord account identity is invalid',
    };
  }

  if (now.getTime() - createdAt.getTime() < DISCORD_ACCOUNT_MIN_AGE_MS) {
    return {
      ok: false,
      discordAccountCreatedAt,
      discordEmailVerified: 1,
      subcode: 'discord_account_too_new',
      message: 'Discord account must be at least 30 days old',
    };
  }

  return {
    ok: true,
    discordAccountCreatedAt,
    discordEmailVerified: 1,
  };
}

async function deriveDiscordUserRef(input: {
  discordUserId: string;
  secret: string;
}): Promise<string | null> {
  const discordUserId = input.discordUserId.trim();
  const secret = input.secret.trim();
  if (!discordUserId || !secret) {
    return null;
  }

  const key = await crypto.subtle.importKey(
    'raw',
    textEncoder.encode(secret),
    {
      name: 'HMAC',
      hash: 'SHA-256',
    },
    false,
    ['sign'],
  );
  const signature = await crypto.subtle.sign(
    'HMAC',
    key,
    textEncoder.encode(`${DISCORD_USER_REF_PAYLOAD_PREFIX}\n${discordUserId}`),
  );

  return `${DISCORD_USER_REF_PREFIX}${encodeBase64Url(new Uint8Array(signature))}`;
}

async function reserveDiscordEntitlement(
  db: D1Database,
  input: {
    installationId: string;
    devicePublicKey: string;
    hardwareHash: string;
    hardwareHashSaltVersion: number;
    appVersion: string;
    discordUserRef: string;
    now: Date;
    nowIso: string;
  },
): Promise<DiscordReservationResult> {
  let identityInserted = false;
  try {
    const bindingConflict = await getDiscordInstallationBindingConflict(db, input);
    if (bindingConflict) {
      return { ok: false, subcode: bindingConflict };
    }

    await insertDiscordInstallationIfAbsent(db, input);

    const postInsertBindingConflict = await getDiscordInstallationBindingConflict(
      db,
      input,
    );
    if (postInsertBindingConflict) {
      return { ok: false, subcode: postInsertBindingConflict };
    }

    identityInserted = await insertDiscordIdentityReservation(db, input);
    if (!identityInserted) {
      return { ok: false, subcode: 'discord_lifetime_used' };
    }

    const reserved = await insertOrUpdateIssuingDiscordEntitlement(db, input);
    if (reserved) {
      const bindingUpdated = await updateDiscordInstallationBinding(db, input);
      if (!bindingUpdated) {
        await releaseDiscordReservation(db, input);
        const reserveBindingConflict = await getDiscordInstallationBindingConflict(
          db,
          input,
        );
        return {
          ok: false,
          subcode: reserveBindingConflict ?? 'entitlement_reservation_failed',
        };
      }
      return { ok: true };
    }

    const duplicateHardware = await hasDeliveredHardwareDuplicate(db, input);
    if (duplicateHardware) {
      await releaseDiscordReservation(db, input);
      return { ok: false, subcode: 'hardware_duplicate' };
    }

    const issuingConflict = await hasSameInstallationIssuingConflict(db, input);
    if (issuingConflict) {
      await releaseDiscordReservation(db, input);
      return { ok: false, subcode: 'discord_installation_already_issuing' };
    }

    const cap = await getDiscordDailyIssuanceCapState(db, input.now);
    if (cap.reached) {
      await releaseDiscordReservation(db, input);
      return {
        ok: false,
        subcode: 'global_cap_reached',
        retryAfterMs: cap.retryAfterMs,
      };
    }

    await releaseDiscordReservation(db, input);
    return { ok: false, subcode: 'entitlement_reservation_failed' };
  } catch {
    if (identityInserted) {
      await releaseDiscordReservation(db, input);
    }
    return { ok: false, subcode: 'entitlement_reservation_failed' };
  }
}

async function insertDiscordInstallationIfAbsent(
  db: D1Database,
  input: {
    installationId: string;
    devicePublicKey: string;
    hardwareHash: string;
    hardwareHashSaltVersion: number;
    appVersion: string;
    nowIso: string;
  },
): Promise<void> {
  await db
    .prepare(
      `INSERT OR IGNORE INTO installations (
          installation_id,
          device_public_key,
          hardware_hash,
          hardware_hash_salt_version,
          app_version,
          challenge,
          challenge_expires_at,
          challenge_salt_version,
          created_at,
          last_seen_at
        ) VALUES (?, ?, ?, ?, ?, NULL, NULL, NULL, ?, ?)`,
    )
    .bind(
      input.installationId,
      input.devicePublicKey,
      input.hardwareHash,
      input.hardwareHashSaltVersion,
      input.appVersion,
      input.nowIso,
      input.nowIso,
    )
    .run();
}

async function updateDiscordInstallationBinding(
  db: D1Database,
  input: {
    installationId: string;
    devicePublicKey: string;
    hardwareHash: string;
    hardwareHashSaltVersion: number;
    appVersion: string;
    nowIso: string;
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
          AND device_public_key = ?`,
    )
    .bind(
      input.hardwareHash,
      input.hardwareHashSaltVersion,
      input.appVersion,
      input.nowIso,
      input.installationId,
      input.devicePublicKey,
    )
    .run();

  return Number(result.meta.changes ?? 0) === 1;
}

async function insertDiscordIdentityReservation(
  db: D1Database,
  input: {
    installationId: string;
    discordUserRef: string;
    nowIso: string;
  },
): Promise<boolean> {
  const result = await db
    .prepare(
      `INSERT INTO discord_identities (
          discord_user_ref,
          entitlement_installation_id,
          status,
          ref_secret_version,
          created_at,
          updated_at
        ) VALUES (?, ?, 'issuing', ?, ?, ?)
        ON CONFLICT(discord_user_ref) DO NOTHING`,
    )
    .bind(
      input.discordUserRef,
      input.installationId,
      DISCORD_USER_REF_VERSION,
      input.nowIso,
      input.nowIso,
    )
    .run();

  return Number(result.meta.changes ?? 0) === 1;
}

async function insertOrUpdateIssuingDiscordEntitlement(
  db: D1Database,
  input: {
    installationId: string;
    hardwareHash: string;
    hardwareHashSaltVersion: number;
    discordUserRef: string;
    now: Date;
    nowIso: string;
  },
): Promise<boolean> {
  const controls = await getBrokerAbuseControlsConfig(db);
  const maxCount = controls.newActiveEntitlementsPerDay.maxCount;
  const capWindow = getDailyCapWindow(
    input.now,
    controls.newActiveEntitlementsPerDay.windowDays,
  );
  const capPredicate =
    maxCount === null
      ? '1 = 1'
      : `(
          (
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
          )
        ) < ?`;
  const result = await db
    .prepare(
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
          verified_hardware_hash_salt_version,
          discord_user_ref,
          discord_issue_status,
          discord_issue_reserved_at,
          discord_issue_delivered_at
        )
        SELECT ?, 'pending_release', ?, NULL, NULL, NULL, NULL, NULL, NULL,
               ?, ?, ?, 'issuing', ?, NULL
         WHERE ${deliveredHardwareNotExistsPredicate()}
           AND ${capPredicate}
        ON CONFLICT(installation_id) DO UPDATE SET
          status = 'pending_release',
          budget_usd = excluded.budget_usd,
          managed_credential_ref = NULL,
          issued_at = NULL,
          expires_at = NULL,
          release_session_ref = NULL,
          release_token_hash = NULL,
          release_token_expires_at = NULL,
          verified_hardware_hash = excluded.verified_hardware_hash,
          verified_hardware_hash_salt_version = excluded.verified_hardware_hash_salt_version,
          discord_user_ref = excluded.discord_user_ref,
          discord_issue_status = excluded.discord_issue_status,
          discord_issue_reserved_at = excluded.discord_issue_reserved_at,
          discord_issue_delivered_at = NULL
        WHERE openrouter_entitlements.status NOT IN ('active', 'expired', 'revoked')
          AND (
            openrouter_entitlements.discord_issue_status IS NULL
             OR openrouter_entitlements.discord_issue_status NOT IN ('issuing', 'delivery_pending', 'cleanup_required')
            OR (
              openrouter_entitlements.discord_issue_status = 'issuing'
              AND (
                openrouter_entitlements.discord_user_ref IS NULL
                OR openrouter_entitlements.discord_user_ref = excluded.discord_user_ref
              )
            )
          )`,
    )
    .bind(
      input.installationId,
      MANAGED_TRIAL_BUDGET_POLICY.hardLimit,
      input.hardwareHash,
      input.hardwareHashSaltVersion,
      input.discordUserRef,
      input.nowIso,
      input.hardwareHash,
      input.hardwareHashSaltVersion,
      input.installationId,
      input.discordUserRef,
      input.hardwareHash,
      input.hardwareHashSaltVersion,
      ...(maxCount === null
        ? []
        : [
            capWindow.startIso,
            capWindow.endIso,
            capWindow.startIso,
            capWindow.endIso,
            maxCount,
          ]),
    )
    .run();

  return Number(result.meta.changes ?? 0) === 1;
}

function deliveredHardwareNotExistsPredicate(): string {
  return `NOT EXISTS (
            SELECT 1
              FROM openrouter_entitlements reserved
             WHERE reserved.verified_hardware_hash = ?
               AND reserved.verified_hardware_hash_salt_version = ?
               AND (
                 reserved.status IN ('active', 'expired', 'revoked')
                 OR (
                   reserved.discord_issue_status IN ('issuing', 'delivery_pending', 'cleanup_required')
                  AND NOT (
                    reserved.installation_id = ?
                    AND reserved.discord_user_ref = ?
                   )
                 )
               )
          )
          AND NOT EXISTS (
            SELECT 1
              FROM installations legacy
              JOIN openrouter_entitlements legacy_entitlement
                ON legacy_entitlement.installation_id = legacy.installation_id
             WHERE legacy.hardware_hash = ?
               AND legacy.hardware_hash_salt_version = ?
               AND legacy_entitlement.status IN ('active', 'expired', 'revoked')
          )`;
}

async function hasDeliveredHardwareDuplicate(
  db: D1Database,
  input: {
    installationId: string;
    hardwareHash: string;
    hardwareHashSaltVersion: number;
    discordUserRef: string;
  },
): Promise<boolean> {
  const row = await db
    .prepare(
      `SELECT EXISTS(
          SELECT 1
            FROM openrouter_entitlements reserved
           WHERE reserved.verified_hardware_hash = ?
             AND reserved.verified_hardware_hash_salt_version = ?
             AND (
               reserved.status IN ('active', 'expired', 'revoked')
               OR (
                   reserved.discord_issue_status IN ('issuing', 'delivery_pending', 'cleanup_required')
                  AND NOT (
                    reserved.installation_id = ?
                    AND reserved.discord_user_ref = ?
                 )
               )
             )
        )
        OR EXISTS(
          SELECT 1
            FROM installations legacy
            JOIN openrouter_entitlements legacy_entitlement
              ON legacy_entitlement.installation_id = legacy.installation_id
           WHERE legacy.hardware_hash = ?
             AND legacy.hardware_hash_salt_version = ?
             AND legacy_entitlement.status IN ('active', 'expired', 'revoked')
        ) AS duplicate_found`,
    )
    .bind(
      input.hardwareHash,
      input.hardwareHashSaltVersion,
      input.installationId,
      input.discordUserRef,
      input.hardwareHash,
      input.hardwareHashSaltVersion,
    )
    .first<{ duplicate_found: number }>();

  return Number(row?.duplicate_found ?? 0) === 1;
}

async function hasSameInstallationIssuingConflict(
  db: D1Database,
  input: {
    installationId: string;
    discordUserRef: string;
  },
): Promise<boolean> {
  const row = await db
    .prepare(
      `SELECT EXISTS(
          SELECT 1
            FROM openrouter_entitlements existing
           WHERE existing.installation_id = ?
             AND existing.status = 'pending_release'
               AND existing.discord_issue_status IN ('issuing', 'delivery_pending', 'cleanup_required')
              AND existing.discord_user_ref IS NOT NULL
              AND existing.discord_user_ref <> ?
        ) AS conflict_found`,
    )
    .bind(input.installationId, input.discordUserRef)
    .first<{ conflict_found: number }>();

  return Number(row?.conflict_found ?? 0) === 1;
}

async function getDiscordDailyIssuanceCapState(
  db: D1Database,
  now: Date,
): Promise<{ reached: boolean; retryAfterMs: number | null }> {
  const cap = await getManagedDailyIssuanceCapState(db, now);
  return {
    reached: cap.reached,
    retryAfterMs: cap.retryAfterMs,
  };
}

function getDailyCapWindow(
  now: Date,
  windowDays: number,
): { startIso: string; endIso: string; end: Date } {
  const start = new Date(
    Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), now.getUTCDate()),
  );
  start.setUTCDate(start.getUTCDate() - (windowDays - 1));
  const end = new Date(start.getTime());
  end.setUTCDate(end.getUTCDate() + windowDays);
  return {
    startIso: start.toISOString(),
    endIso: end.toISOString(),
    end,
  };
}

async function releaseDiscordReservation(
  db: D1Database,
  input: {
    installationId: string;
    discordUserRef: string;
  },
): Promise<void> {
  await db.batch([
    db.prepare(
      `DELETE FROM openrouter_entitlements
        WHERE installation_id = ?
          AND discord_user_ref = ?
          AND status = 'pending_release'
          AND discord_issue_status = 'issuing'
          AND managed_credential_ref IS NULL`,
    ).bind(input.installationId, input.discordUserRef),
    db.prepare(
      `DELETE FROM discord_identities
        WHERE discord_user_ref = ?
          AND entitlement_installation_id = ?
          AND status = 'issuing'`,
    ).bind(input.discordUserRef, input.installationId),
  ]);
}
async function handleDiscordManagedChildKeyFailure(
  c: Context<BrokerEnv>,
  input: {
    installationId: string;
    releaseSessionRef: string;
    discordUserRef: string;
    childKey: {
      rawKey: string;
      hash: string;
    };
    nowIso: string;
    error: unknown;
    sensitiveValues: string[];
    providerCleanupHandled?: boolean;
    operationId?: string | null;
  },
): Promise<void> {
  const cleanup = input.providerCleanupHandled
    ? { ok: true as const }
    : await cleanupManagedChildKey({
        managementApiKey: c.env.OPENROUTER_MANAGEMENT_API_KEY,
        keyHash: input.childKey.hash,
      });

  if (cleanup.ok) {
    try {
      await releaseDiscordReservationAfterManagedCleanup(c.env.BROKER_DB, {
        installationId: input.installationId,
        discordUserRef: input.discordUserRef,
        managedCredentialRef: input.childKey.hash,
        operationId: input.operationId ?? null,
      });
    } catch (error) {
      await deliverManagedCleanupIncident(c.env, {
        issueSource: 'discord',
        managedCredentialRef: input.childKey.hash,
        phase: 'managed_issue',
        cleanupRequiredRecorded: false,
        occurredAt: input.nowIso,
      });
      throw error;
    }
    return;
  }

  let cleanupTransition = {
    incidentRecorded: false,
    cleanupRequiredRecorded: false,
  };
  try {
    cleanupTransition = await markDiscordCleanupRequired(c.env.BROKER_DB, {
      installationId: input.installationId,
      discordUserRef: input.discordUserRef,
      managedCredentialRef: input.childKey.hash,
      operationId: input.operationId ?? null,
      nowIso: input.nowIso,
    });
  } catch (error) {
    await deliverManagedCleanupIncident(c.env, {
      issueSource: 'discord',
      managedCredentialRef: input.childKey.hash,
      phase: 'managed_issue',
      cleanupRequiredRecorded: false,
      occurredAt: input.nowIso,
    });
    throw error;
  }
  if (cleanupTransition.incidentRecorded) {
    await deliverManagedCleanupIncident(c.env, {
      issueSource: 'discord',
      managedCredentialRef: input.childKey.hash,
      phase: 'managed_issue',
      cleanupRequiredRecorded: cleanupTransition.cleanupRequiredRecorded,
      occurredAt: input.nowIso,
    });
  }
  console.error('discord_managed_child_key_cleanup_required', {
    installation_id: input.installationId,
    release_session_ref: input.releaseSessionRef,
    managed_credential_ref: input.childKey.hash,
    failure: redactSensitiveDiagnostics(
      normalizeFailureForLog(input.error),
      input.sensitiveValues,
    ),
    cleanup_outcome: redactSensitiveDiagnostics(cleanup.reason, input.sensitiveValues),
    broker_timestamp: new Date().toISOString(),
  });
}

async function releaseDiscordReservationAfterManagedCleanup(
  db: D1Database,
  input: {
    installationId: string;
    discordUserRef: string;
    managedCredentialRef: string;
    operationId: string | null;
  },
): Promise<void> {
  const sharedRows = await hasOtherLiveOperation(db, {
    issueSource: 'discord',
    subjectRef: input.discordUserRef,
    installationId: input.installationId,
    excludeOperationId: input.operationId,
  });
  const statements = [
    db.prepare(
      `DELETE FROM openrouter_entitlements
        WHERE installation_id = ?
          AND discord_user_ref = ?
          AND managed_credential_ref = ?
          AND discord_issue_status IN ('issuing', 'delivery_pending', 'active', 'cleanup_required')`,
    ).bind(
      input.installationId,
      input.discordUserRef,
      input.managedCredentialRef,
    ),
  ];
  if (!sharedRows) {
    statements.push(
      db.prepare(
        `DELETE FROM openrouter_entitlements
          WHERE installation_id = ?
            AND discord_user_ref = ?
            AND status = 'pending_release'
            AND discord_issue_status = 'issuing'
            AND managed_credential_ref IS NULL`,
      ).bind(input.installationId, input.discordUserRef),
      db.prepare(
        `DELETE FROM discord_identities
          WHERE discord_user_ref = ?
            AND entitlement_installation_id = ?
             AND status IN ('issuing', 'active', 'cleanup_required')`,
      ).bind(input.discordUserRef, input.installationId),
    );
  }
  await db.batch(statements);
}

async function markDiscordCleanupRequired(
  db: D1Database,
  input: {
    installationId: string;
    discordUserRef: string;
    managedCredentialRef: string;
    operationId: string | null;
    nowIso: string;
  },
): Promise<{ incidentRecorded: boolean; cleanupRequiredRecorded: boolean; sharedRowsPreserved: boolean }> {
  const sharedRows = await hasOtherLiveOperation(db, {
    issueSource: 'discord',
    subjectRef: input.discordUserRef,
    installationId: input.installationId,
    excludeOperationId: input.operationId,
  });
  if (sharedRows) {
    return { incidentRecorded: true, cleanupRequiredRecorded: false, sharedRowsPreserved: true };
  }
  const [entitlementResult, identityResult] = await db.batch([
    db.prepare(
      `UPDATE openrouter_entitlements
          SET status = 'pending_release',
              managed_credential_ref = ?,
              issued_at = NULL,
              expires_at = NULL,
              release_session_ref = NULL,
              release_token_hash = NULL,
              release_token_expires_at = NULL,
              discord_issue_status = 'cleanup_required',
              discord_issue_delivered_at = NULL
        WHERE installation_id = ?
          AND discord_user_ref = ?
          AND discord_issue_status IN ('issuing', 'delivery_pending', 'active')
          AND (managed_credential_ref IS NULL OR managed_credential_ref = ?)`,
    )
    .bind(
      input.managedCredentialRef,
      input.installationId,
      input.discordUserRef,
      input.managedCredentialRef,
    ),
    db.prepare(
      `UPDATE discord_identities
          SET status = 'cleanup_required',
              updated_at = ?
        WHERE discord_user_ref = ?
          AND entitlement_installation_id = ?
          AND status IN ('issuing', 'active')`,
    ).bind(input.nowIso, input.discordUserRef, input.installationId),
  ]);
  const entitlementRecorded = Number(entitlementResult.meta.changes ?? 0) === 1;
  const identityRecorded = Number(identityResult.meta.changes ?? 0) === 1;
  return {
    incidentRecorded: entitlementRecorded || identityRecorded,
    cleanupRequiredRecorded: entitlementRecorded && identityRecorded,
    sharedRowsPreserved: false,
  };
}

async function markDiscordIndeterminateChildKeyCreation(
  db: D1Database,
  input: {
    installationId: string;
    discordUserRef: string;
    operationId?: string | null;
    nowIso: string;
  },
): Promise<boolean> {
  const sharedRows = await hasOtherLiveOperation(db, {
    issueSource: 'discord',
    subjectRef: input.discordUserRef,
    installationId: input.installationId,
    excludeOperationId: input.operationId ?? null,
  });
  if (sharedRows) {
    return false;
  }
  const [entitlementResult, identityResult] = await db.batch([
    db.prepare(
      `UPDATE openrouter_entitlements
          SET discord_issue_status = 'cleanup_required',
              discord_issue_delivered_at = NULL
        WHERE installation_id = ?
          AND discord_user_ref = ?
          AND status = 'pending_release'
          AND discord_issue_status = 'issuing'
          AND managed_credential_ref IS NULL`,
    ).bind(input.installationId, input.discordUserRef),
    db.prepare(
      `UPDATE discord_identities
          SET status = 'cleanup_required', updated_at = ?
        WHERE discord_user_ref = ?
          AND entitlement_installation_id = ?
          AND status = 'issuing'`,
    ).bind(input.nowIso, input.discordUserRef, input.installationId),
  ]);
  return (
    Number(entitlementResult.meta.changes ?? 0) === 1 &&
    Number(identityResult.meta.changes ?? 0) === 1
  );
}

async function activateDiscordReservation(
  db: D1Database,
  input: {
    stateHash: string;
    installationId: string;
    devicePublicKey: string;
    discordUserRef: string;
    managedCredentialRef: string;
    issuedAt: string;
    expiresAt: string;
    budgetUsd: number;
    deliveredAt: string;
  },
): Promise<boolean> {
  const entitlementResult = await db
    .prepare(
      `UPDATE openrouter_entitlements
          SET status = 'active',
              budget_usd = ?,
              managed_credential_ref = ?,
              issued_at = ?,
              expires_at = ?,
              release_session_ref = NULL,
              release_token_hash = NULL,
              release_token_expires_at = NULL,
              discord_issue_status = 'active',
              discord_issue_delivered_at = ?
        WHERE installation_id = ?
          AND discord_user_ref = ?
          AND status = 'pending_release'
          AND discord_issue_status = 'issuing'
          AND managed_credential_ref IS NULL
          AND EXISTS (
            SELECT 1
              FROM installations activation_installation
             WHERE activation_installation.installation_id = openrouter_entitlements.installation_id
               AND activation_installation.device_public_key = ?
          )`,
    )
    .bind(
      input.budgetUsd,
      input.managedCredentialRef,
      input.issuedAt,
      input.expiresAt,
      input.deliveredAt,
      input.installationId,
      input.discordUserRef,
      input.devicePublicKey,
    )
    .run();
  if (Number(entitlementResult.meta.changes ?? 0) !== 1) {
    return false;
  }

  const identityResult = await db
    .prepare(
      `UPDATE discord_identities
          SET status = 'active',
              updated_at = ?
        WHERE discord_user_ref = ?
          AND entitlement_installation_id = ?
          AND status = 'issuing'`,
    )
    .bind(input.deliveredAt, input.discordUserRef, input.installationId)
    .run();
  if (Number(identityResult.meta.changes ?? 0) !== 1) {
    return false;
  }

  const sessionResult = await db
    .prepare(
      `UPDATE discord_oauth_sessions
          SET status = 'consumed',
              pkce_code_verifier = NULL,
              consumed_at = ?
        WHERE state_hash = ?
          AND status = 'processing'`,
    )
    .bind(input.deliveredAt, input.stateHash)
    .run();

  return Number(sessionResult.meta.changes ?? 0) === 1;
}

async function markDiscordReservationDeliveryPending(
  db: D1Database,
  input: {
    stateHash: string;
    installationId: string;
    devicePublicKey: string;
    discordUserRef: string;
    managedCredentialRef: string;
    issuedAt: string;
    expiresAt: string;
    budgetUsd: number;
    nowIso: string;
  },
): Promise<boolean> {
  const entitlementResult = await db
    .prepare(
      `UPDATE openrouter_entitlements
          SET status = 'pending_release',
              budget_usd = ?,
              managed_credential_ref = ?,
              issued_at = ?,
              expires_at = ?,
              release_session_ref = NULL,
              release_token_hash = NULL,
              release_token_expires_at = NULL,
              discord_issue_status = 'delivery_pending',
              discord_issue_delivered_at = NULL
        WHERE installation_id = ?
          AND discord_user_ref = ?
          AND status = 'pending_release'
          AND discord_issue_status = 'issuing'
          AND managed_credential_ref IS NULL
          AND EXISTS (
            SELECT 1
              FROM installations delivery_installation
             WHERE delivery_installation.installation_id = openrouter_entitlements.installation_id
               AND delivery_installation.device_public_key = ?
          )`,
    )
    .bind(
      input.budgetUsd,
      input.managedCredentialRef,
      input.issuedAt,
      input.expiresAt,
      input.installationId,
      input.discordUserRef,
      input.devicePublicKey,
    )
    .run();
  if (Number(entitlementResult.meta.changes ?? 0) !== 1) {
    return false;
  }

  const sessionResult = await db
    .prepare(
      `UPDATE discord_oauth_sessions
          SET status = 'consumed',
              pkce_code_verifier = NULL,
              consumed_at = ?
        WHERE state_hash = ?
          AND status = 'processing'`,
    )
    .bind(input.nowIso, input.stateHash)
    .run();

  return Number(sessionResult.meta.changes ?? 0) === 1;
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

async function isDiscordDeliveryAckFinalized(db: D1Database, deliveryId: string): Promise<boolean> {
  const finalized = await db.prepare(
    `SELECT 1 AS finalized
       FROM managed_key_deliveries AS delivery
       JOIN openrouter_entitlements AS entitlement
         ON entitlement.managed_credential_ref = delivery.managed_credential_ref
       JOIN discord_identities AS identity
         ON identity.discord_user_ref = entitlement.discord_user_ref
        AND identity.entitlement_installation_id = entitlement.installation_id
      WHERE delivery.delivery_id = ?
        AND delivery.status = 'acknowledged'
        AND entitlement.status = 'active'
        AND entitlement.discord_issue_status = 'active'
        AND entitlement.discord_issue_delivered_at IS NOT NULL
        AND identity.status = 'active'
        AND EXISTS (
          SELECT 1 FROM broker_issue_success_events
           WHERE issue_source = 'discord'
             AND managed_credential_ref = delivery.managed_credential_ref
        )
        AND NOT EXISTS (
          SELECT 1
            FROM referral_rewards reward
           WHERE reward.referred_source = 'discord'
             AND reward.referred_subject_ref = entitlement.discord_user_ref
             AND reward.referred_installation_id IS delivery.installation_id
             AND reward.referred_bonus_status = 'reserved'
             AND NOT EXISTS (
               SELECT 1
                 FROM managed_referral_settlement_jobs job
                WHERE job.referral_reward_id = reward.id
                  AND job.delivery_id = delivery.delivery_id
             )
        )`,
  )
    .bind(deliveryId)
    .first<{ finalized: number }>()
    .catch(() => null);
  return Number(finalized?.finalized ?? 0) === 1;
}

export async function finalizeDiscordManagedKeyDeliveryAck(
  c: Context<BrokerEnv>,
  input: {
    deliveryId: string;
    managedCredentialRef: string;
    acknowledgedAt: Date;
  },
): Promise<{
  acknowledgementStatus: 'acknowledged' | 'already_acknowledged';
  referralBonusApplied?: boolean;
}> {
  const entitlement = await c.env.BROKER_DB.prepare(
    `SELECT installation_id, status, budget_usd, managed_credential_ref, issued_at,
            expires_at, release_session_ref, release_token_hash, release_token_expires_at,
            verified_hardware_hash, verified_hardware_hash_salt_version,
            discord_user_ref, discord_issue_status, discord_issue_reserved_at,
            discord_issue_delivered_at
       FROM openrouter_entitlements
      WHERE managed_credential_ref = ?`,
  )
    .bind(input.managedCredentialRef)
    .first<OpenRouterEntitlementRecord>();
  if (!entitlement) {
    throw new Error('Discord delivery ACK target is missing');
  }
  const alreadyActive =
    entitlement.status === 'active' && entitlement.discord_issue_status === 'active';
  if (!alreadyActive && (
    entitlement.status !== 'pending_release' ||
    entitlement.discord_issue_status !== 'delivery_pending'
  )) {
    throw new Error('Discord delivery ACK target is not pending');
  }
  if (!entitlement.discord_user_ref || !entitlement.issued_at || !entitlement.expires_at) {
    throw new Error('Discord delivery ACK target is missing entitlement metadata');
  }

  const deliveredAt = entitlement.discord_issue_delivered_at ?? input.acknowledgedAt.toISOString();
  const network = await extractRequestNetworkMetadata(c, { secrets: resolveRequestNetworkIdentitySecrets(c), now: new Date() });
  const netMode = await resolveNetworkIdentityWriteMode(c.env.BROKER_DB);
  const results = await c.env.BROKER_DB.batch([
    c.env.BROKER_DB.prepare(
      `UPDATE openrouter_entitlements
          SET status = 'active', discord_issue_status = 'active', discord_issue_delivered_at = ?
         WHERE installation_id = ?
           AND managed_credential_ref = ?
           AND status = 'pending_release'
           AND discord_issue_status = 'delivery_pending'
           AND EXISTS (
             SELECT 1 FROM managed_key_deliveries
              WHERE delivery_id = ?
                AND managed_credential_ref = ?
                AND status = 'pending'
           )`,
    ).bind(
      deliveredAt,
      entitlement.installation_id,
      input.managedCredentialRef,
      input.deliveryId,
      input.managedCredentialRef,
    ),
    c.env.BROKER_DB.prepare(
      `UPDATE discord_identities
        SET status = 'active', updated_at = ?
      WHERE discord_user_ref = ?
        AND entitlement_installation_id = ?
        AND status = 'issuing'
        AND EXISTS (
          SELECT 1 FROM managed_key_deliveries
           WHERE delivery_id = ?
             AND managed_credential_ref = ?
             AND status = 'pending'
        )`,
    ).bind(
      deliveredAt,
      entitlement.discord_user_ref,
      entitlement.installation_id,
      input.deliveryId,
      input.managedCredentialRef,
    ),
    prepareIssueSuccessInsert(c.env.BROKER_DB, {
      installationId: entitlement.installation_id,
      managedCredentialRef: input.managedCredentialRef,
      observedAt: deliveredAt,
      network,
      deliveryId: input.deliveryId,
    }, netMode),
    c.env.BROKER_DB.prepare(
      `UPDATE managed_key_deliveries
          SET status = 'acknowledged', acknowledged_at = ?
        WHERE delivery_id = ?
          AND managed_credential_ref = ?
          AND status = 'pending'
          AND EXISTS (
            SELECT 1 FROM openrouter_entitlements
             WHERE installation_id = ?
               AND managed_credential_ref = ?
               AND status = 'active'
               AND discord_issue_status = 'active'
               AND discord_issue_delivered_at IS NOT NULL
          )
          AND EXISTS (
            SELECT 1 FROM discord_identities
             WHERE discord_user_ref = ?
               AND entitlement_installation_id = ?
               AND status = 'active'
          )
          AND EXISTS (
            SELECT 1 FROM broker_issue_success_events
             WHERE issue_source = 'discord'
               AND managed_credential_ref = ?
          )`,
    ).bind(
      input.acknowledgedAt.toISOString(),
      input.deliveryId,
      input.managedCredentialRef,
      entitlement.installation_id,
      input.managedCredentialRef,
      entitlement.discord_user_ref,
      entitlement.installation_id,
      input.managedCredentialRef,
    ),
    c.env.BROKER_DB.prepare(
      `INSERT INTO managed_referral_settlement_jobs (
          source,
          referral_reward_id,
          delivery_id,
          operation_id,
          phase,
          attempt_count,
          last_attempt_at,
          next_attempt_at,
          fencing_token,
          lease_expires_at,
          last_error_code,
          created_at,
          updated_at,
          completed_at
        )
        SELECT 'discord',
               reward.id,
               delivery.delivery_id,
               reward.operation_id,
               'invitee_pending',
               0,
               NULL,
               ?,
               NULL,
               NULL,
               NULL,
               ?,
               ?,
               NULL
          FROM referral_rewards reward
          JOIN managed_key_deliveries delivery
            ON delivery.delivery_id = ?
           AND delivery.issue_source = 'discord'
           AND delivery.subject_ref = reward.referred_subject_ref
           AND delivery.installation_id IS reward.referred_installation_id
           AND delivery.managed_credential_ref = ?
           AND delivery.status = 'acknowledged'
         WHERE reward.id = (
           SELECT candidate.id
             FROM referral_rewards candidate
            WHERE candidate.referred_source = 'discord'
              AND candidate.referred_subject_ref = ?
              AND candidate.referred_installation_id IS ?
              AND candidate.referred_bonus_status = 'reserved'
            ORDER BY candidate.created_at DESC, candidate.id DESC
            LIMIT 1
         )
        ON CONFLICT(referral_reward_id) DO NOTHING`,
    ).bind(
      input.acknowledgedAt.toISOString(),
      input.acknowledgedAt.toISOString(),
      input.acknowledgedAt.toISOString(),
      input.deliveryId,
      input.managedCredentialRef,
      entitlement.discord_user_ref,
      entitlement.installation_id,
    ),
  ]);
  if (!(await isDiscordDeliveryAckFinalized(c.env.BROKER_DB, input.deliveryId))) {
    await ensureReferralSettlementJobsForDelivery(c.env.BROKER_DB, {
      source: 'discord',
      deliveryId: input.deliveryId,
      now: input.acknowledgedAt,
    });
  }
  if (
    !(await isDiscordDeliveryAckFinalized(c.env.BROKER_DB, input.deliveryId)) &&
    (await hasUnsettledReservedRewardWithoutJob(c.env.BROKER_DB, {
      source: 'discord',
      subjectRef: entitlement.discord_user_ref,
      installationId: entitlement.installation_id,
      deliveryId: input.deliveryId,
    }))
  ) {
    throw new Error('Discord delivery ACK finalization failed');
  }
  const operationActivated = await markOperationActiveOnAck(c.env.BROKER_DB, input.deliveryId, input.acknowledgedAt);
  if (!operationActivated) {
    throw new Error('Managed operation ACK activation failed');
  }
  await runDiscordIssueSuccessMonitoring(c, {
    installationId: entitlement.installation_id,
    managedCredentialRef: input.managedCredentialRef,
    issuedAt: deliveredAt,
    now: input.acknowledgedAt,
    sensitiveValues: [],
  });
  return {
    acknowledgementStatus:
      Number(results[3]?.meta.changes ?? 0) === 1
        ? 'acknowledged'
        : 'already_acknowledged',
  };
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

function assertDiscordIssueEntitlementDeliverable(
  entitlement: OpenRouterEntitlementRecord,
): void {
  if (!entitlement.managed_credential_ref || !entitlement.expires_at) {
    throw new Error('active Discord entitlement missing managed release metadata');
  }
}

async function discordIssueSuccessResponse(
  c: Context<BrokerEnv>,
  input: {
    entitlement: OpenRouterEntitlementRecord;
    rawKey: string;
    model: string;
    installationId: string;
    referralId: string | null;
    talkTogetherPass: TalkTogetherPassStatusResponse | null;
    referralBonusApplied: boolean;
  },
): Promise<Response> {
  assertDiscordIssueEntitlementDeliverable(input.entitlement);

  const managedUserHmacSecret = nonEmptyString(
    c.env.OPENROUTER_MANAGED_USER_HMAC_SECRET,
  );
  let openRouterUserId: string | null = null;
  if (managedUserHmacSecret) {
    try {
      openRouterUserId = await deriveManagedOpenRouterUserId({
        installationId: input.installationId,
        secret: managedUserHmacSecret,
      });
    } catch {
      openRouterUserId = null;
    }
  }

  return c.json({
    openrouter_api_key: input.rawKey,
    ...(openRouterUserId ? { openrouter_user_id: openRouterUserId } : {}),
    managed_credential_ref: input.entitlement.managed_credential_ref,
    managed_state: {
      lifecycle: 'active',
      managed_availability: true,
    },
    expires_at: input.entitlement.expires_at,
    budget_usd: input.entitlement.budget_usd,
    model: input.model,
    ...(input.referralId ? { referral_id: input.referralId } : {}),
    ...(input.talkTogetherPass ? { talk_together_pass: input.talkTogetherPass } : {}),
    ...(input.referralBonusApplied ? { referral_bonus_applied: true } : {}),
  });
}

type OwnedReferralIssueLookup = {
  referralCode: ReferralCodeRecord;
  talkTogetherPass: TalkTogetherPassStatusResponse | null;
};

async function bestEffortResolveOwnedReferralStatusForIssueResponse(
  db: D1Database,
  input: {
    installationId: string;
    nowIso: string;
  },
): Promise<OwnedReferralIssueLookup | null> {
  try {
    const result = await ensureOwnedReferralIdForActiveDiscordManagedUser(db, input);
    if (!result.ok) {
      logOwnedReferralIssueFailure({
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
      logOwnedReferralIssueFailure({
        installationId: input.installationId,
        reason: 'talk_together_pass_status_failed',
      });
      return {
        referralCode: result.referralCode,
        talkTogetherPass: null,
      };
    }
  } catch {
    logOwnedReferralIssueFailure({
      installationId: input.installationId,
      reason: 'owned_referral_ensure_exception',
    });
    return null;
  }
}

function logOwnedReferralIssueFailure(input: {
  installationId: string;
  reason: string;
}): void {
  console.warn('owned_referral_status_failed', {
    endpoint: 'discord_issue',
    installation_id: input.installationId,
    reason: normalizeOwnedReferralFailureReason(input.reason),
  });
}

function normalizeOwnedReferralFailureReason(reason: string): string {
  return /^[a-z0-9_:-]{1,64}$/u.test(reason)
    ? reason
    : 'owned_referral_ensure_exception';
}

async function bestEffortReserveIssueReferralReward(
  db: D1Database,
  input: {
    referralId: string | null;
    referredDiscordUserRef: string;
    referredInstallationId: string;
    referredHardwareHash: string;
    referredHardwareHashSaltVersion: number;
    attemptIpDigest?: ReferralAttemptIpDigest | null;
    attemptIpLegacyHash?: string | null;
    operationId?: string | null;
    nowIso: string;
  },
): Promise<IssueReferralReservationResult | null> {
  try {
    return await reserveIssueReferralReward(db, {
      referralId: input.referralId,
      referredSource: 'discord',
      referredSubjectRef: input.referredDiscordUserRef,
      referredInstallationId: input.referredInstallationId,
      referredHardwareHash: input.referredHardwareHash,
      referredHardwareHashSaltVersion: input.referredHardwareHashSaltVersion,
      attemptIpDigest: input.attemptIpDigest ?? null,
      attemptIpLegacyHash: input.attemptIpLegacyHash ?? null,
      operationId: input.operationId ?? null,
      nowIso: input.nowIso,
    });
  } catch {
    return null;
  }
}

function resolveReferredIssueLimitUsd(
  referralReservation: IssueReferralReservationResult | null,
): number {
  void referralReservation;
  return MANAGED_TRIAL_BUDGET_POLICY.hardLimit;
}

async function scheduleDirectIssueReferralSettlement(
  db: D1Database,
  input: {
    referralReservation: IssueReferralReservationResult | null;
    referredDiscordUserRef: string;
    referredInstallationId: string;
    now: Date;
  },
): Promise<void> {
  if (input.referralReservation?.outcome !== 'reserved') {
    return;
  }
  const reward = await db
    .prepare(
      `SELECT id, operation_id FROM referral_rewards
        WHERE referral_id = ?
          AND referred_source = 'discord'
          AND referred_subject_ref = ?
          AND referred_installation_id IS ?
          AND referred_bonus_status IN ('reserved', 'credited')
        ORDER BY created_at DESC, id DESC
        LIMIT 1`,
    )
    .bind(input.referralReservation.referralId, input.referredDiscordUserRef, input.referredInstallationId)
    .first<{ id: number; operation_id: string | null }>();
  if (!reward) {
    return;
  }
  await scheduleManagedReferralSettlement(db, {
    source: 'discord',
    referralRewardId: reward.id,
    deliveryId: null,
    operationId: reward.operation_id,
    now: input.now,
  });
}
export type DiscordRetryReservation =
  | { ok: true }
  | { ok: false; subcode: DiscordReservationErrorSubcode; retryAfterMs?: number | null }
  | { ok: false; subcode: 'retry_conflict' };

export async function ensureDiscordRetryReservation(
  db: D1Database,
  input: {
    operationId: string;
    attemptIndex: number;
    installationId: string;
    devicePublicKey: string;
    hardwareHash: string;
    hardwareHashSaltVersion: number;
    appVersion: string;
    discordUserRef: string;
    managementApiKey: string;
    hasLiveDelivery: boolean;
    now: Date;
    nowIso: string;
  },
): Promise<DiscordRetryReservation> {
  const entitlement = await getEntitlement(db, input.installationId);
  const installation = await getInstallation(db, input.installationId);
  if (installation && installation.device_public_key !== input.devicePublicKey) {
    return { ok: false, subcode: 'installation_binding_mismatch' };
  }
  if (!entitlement || entitlement.discord_user_ref !== input.discordUserRef) {
    if (entitlement && entitlement.discord_user_ref && entitlement.discord_user_ref !== input.discordUserRef) {
      return { ok: false, subcode: 'installation_binding_mismatch' };
    }
    return reserveDiscordEntitlement(db, input);
  }
  if (
    entitlement.status === 'active' ||
    entitlement.status === 'expired' ||
    entitlement.status === 'revoked'
  ) {
    return { ok: false, subcode: 'discord_lifetime_used' };
  }
  if (entitlement.discord_issue_status === 'delivery_pending' && entitlement.managed_credential_ref) {
    if (input.hasLiveDelivery) {
      return { ok: false, subcode: 'retry_conflict' };
    }
  }
  if (entitlement.managed_credential_ref) {
    const issuedAttempts = await listManagedOperationAttempts(db, input.operationId);
    const ownsCredential = issuedAttempts.some((attempt) => attempt.managed_credential_ref === entitlement.managed_credential_ref);
    if (!ownsCredential) {
      const blocked = await hasOtherLiveOperation(db, {
        issueSource: 'discord',
        subjectRef: input.discordUserRef,
        installationId: input.installationId,
        excludeOperationId: input.operationId,
      });
      if (blocked) {
        await markAttemptUnknown(db, input.operationId, input.attemptIndex, input.now);
        return { ok: false, subcode: 'retry_conflict' };
      }
    }
    await recordAttemptCredential(db, input.operationId, input.attemptIndex, entitlement.managed_credential_ref, input.now);
    const cleanup = await cleanupManagedChildKey({
      managementApiKey: input.managementApiKey,
      keyHash: entitlement.managed_credential_ref,
    });
    if (!cleanup.ok) {
      await markAttemptUnknown(db, input.operationId, input.attemptIndex, input.now);
      return { ok: false, subcode: 'retry_conflict' };
    }
  }
  await db
    .prepare(
      `UPDATE openrouter_entitlements
          SET managed_credential_ref = NULL,
              issued_at = NULL,
              expires_at = NULL,
              discord_issue_status = 'issuing',
              discord_issue_delivered_at = NULL
        WHERE installation_id = ?
          AND discord_user_ref = ?
          AND status = 'pending_release'
          AND discord_issue_status IN ('issuing', 'delivery_pending', 'cleanup_required')`,
    )
    .bind(input.installationId, input.discordUserRef)
    .run();
  const identity = await db
    .prepare(
      `SELECT status, entitlement_installation_id FROM discord_identities WHERE discord_user_ref = ?`,
    )
    .bind(input.discordUserRef)
    .first<{ status: string; entitlement_installation_id: string | null }>();
  if (!identity) {
    const inserted = await insertDiscordIdentityReservation(db, input);
    if (!inserted) {
      return { ok: false, subcode: 'entitlement_reservation_failed' };
    }
  } else if (identity.entitlement_installation_id !== input.installationId) {
    return { ok: false, subcode: 'installation_binding_mismatch' };
  } else if (identity.status === 'cleanup_required') {
    await db
      .prepare(
        `UPDATE discord_identities
            SET status = 'issuing', updated_at = ?
          WHERE discord_user_ref = ?
            AND entitlement_installation_id = ?
            AND status = 'cleanup_required'`,
      )
      .bind(input.nowIso, input.discordUserRef, input.installationId)
      .run();
  } else if (identity.status !== 'issuing') {
    return { ok: false, subcode: 'discord_lifetime_used' };
  }
  if (
    await hasDeliveredHardwareDuplicate(db, {
      installationId: input.installationId,
      hardwareHash: input.hardwareHash,
      hardwareHashSaltVersion: input.hardwareHashSaltVersion,
      discordUserRef: input.discordUserRef,
    })
  ) {
    return { ok: false, subcode: 'hardware_duplicate' };
  }
  const cap = await getDiscordDailyIssuanceCapState(db, input.now);
  if (cap.reached) {
    return { ok: false, subcode: 'global_cap_reached', retryAfterMs: cap.retryAfterMs };
  }
  if (!installation) {
    await insertDiscordInstallationIfAbsent(db, input);
  }
  const rebound = await updateDiscordInstallationBinding(db, input);
  if (!rebound) {
    return { ok: false, subcode: 'entitlement_reservation_failed' };
  }
  return { ok: true };
}

export async function markDiscordRetryDeliveryPending(
  db: D1Database,
  input: {
    installationId: string;
    devicePublicKey: string;
    discordUserRef: string;
    managedCredentialRef: string;
    issuedAt: string;
    expiresAt: string;
    budgetUsd: number;
    nowIso: string;
  },
): Promise<boolean> {
  const result = await db
    .prepare(
      `UPDATE openrouter_entitlements
          SET status = 'pending_release',
              budget_usd = ?,
              managed_credential_ref = ?,
              issued_at = ?,
              expires_at = ?,
              release_session_ref = NULL,
              release_token_hash = NULL,
              release_token_expires_at = NULL,
              discord_issue_status = 'delivery_pending',
              discord_issue_delivered_at = NULL
        WHERE installation_id = ?
          AND discord_user_ref = ?
          AND status = 'pending_release'
          AND discord_issue_status = 'issuing'
          AND managed_credential_ref IS NULL
          AND EXISTS (
            SELECT 1
              FROM installations delivery_installation
             WHERE delivery_installation.installation_id = openrouter_entitlements.installation_id
               AND delivery_installation.device_public_key = ?
          )`,
    )
    .bind(
      input.budgetUsd,
      input.managedCredentialRef,
      input.issuedAt,
      input.expiresAt,
      input.installationId,
      input.discordUserRef,
      input.devicePublicKey,
    )
    .run();
  return Number(result.meta.changes ?? 0) === 1;
}

export async function executeDiscordResumeIssuance(
  c: Context<BrokerEnv>,
  input: {
    operation: StrictManagedOperationRecord;
    attemptIndex: number;
    hasLiveDelivery: boolean;
    now: Date;
    nowIso: string;
  },
): Promise<Response> {
  const db = c.env.BROKER_DB;
  const operation = input.operation;
  const installationId = operation.installation_id;
  const devicePublicKey = operation.device_public_key;
  const hardwareHash = operation.hardware_hash;
  const hardwareHashSaltVersion = operation.hardware_hash_salt_version;
  const appVersion = operation.app_version;
  if (
    !installationId ||
    !devicePublicKey ||
    !hardwareHash ||
    hardwareHashSaltVersion === null ||
    !appVersion
  ) {
    throw new Error('discord resume issuance context is missing after eligibility check');
  }
  const entitlement = await getEntitlement(db, installationId);
  const brakeDecision = await checkActiveIssuanceBrake(db, entitlement);
  if (brakeDecision) {
    return publicErrorResponse(c, brakeDecision.status, {
      code: brakeDecision.code,
      class: brakeDecision.class,
      subcode: brakeDecision.subcode,
      retryAfterMs: brakeDecision.retryAfterMs,
      message: brakeDecision.message,
      entitlement,
    });
  }
  const ensured = await ensureDiscordRetryReservation(db, {
    operationId: operation.operation_id,
    attemptIndex: input.attemptIndex,
    installationId,
    devicePublicKey,
    hardwareHash,
    hardwareHashSaltVersion,
    appVersion,
    discordUserRef: operation.subject_ref,
    managementApiKey: c.env.OPENROUTER_MANAGEMENT_API_KEY,
    hasLiveDelivery: input.hasLiveDelivery,
    now: input.now,
    nowIso: input.nowIso,
  });
  if (!ensured.ok) {
    if (ensured.subcode === 'retry_conflict' || ensured.subcode === 'discord_installation_already_issuing') {
      const attempts = await listManagedOperationAttempts(db, operation.operation_id);
      const current = await getManagedOperationStatusSnapshot(db, operation.operation_id);
      return c.json(await buildManagedOperationStatusBodyWithDelivery(db, current ?? operation, attempts));
    }
    if (ensured.subcode === 'global_cap_reached') {
      await markAttemptUnknown(db, operation.operation_id, input.attemptIndex, input.now);
      return discordReservationErrorResponse(c, ensured);
    }
    await markAttemptUnknown(db, operation.operation_id, input.attemptIndex, input.now);
    await failManagedOperationTerminal(db, operation, input.now, 'terminal_provider_failure');
    const attempts = await listManagedOperationAttempts(db, operation.operation_id);
    const terminal = await getManagedOperationStatusSnapshot(db, operation.operation_id);
    return c.json(await buildManagedOperationStatusBodyWithDelivery(db, terminal ?? operation, attempts));
  }
  const referralReservation = await getOperationReferralReward(db, operation.operation_id);
  if (operation.referral_status === 'reserved') {
    const rewardId = await getReservedReferralRewardIdForOperation(db, operation.operation_id);
    await attachReferralToOperation(db, operation.operation_id, rewardId, 'reserved', 'none', input.now);
  }
  const issuedAt = input.nowIso;
  const expiresAt = addMonthsUtc(
    input.now,
    MANAGED_TRIAL_POLICY.entitlement.issuance.expiry.durationMonths,
  ).toISOString();
  const issueLimitUsd = MANAGED_TRIAL_BUDGET_POLICY.hardLimit;
  const referralLimitVerificationRequired = referralReservation?.outcome === 'reserved';
  let childKey: { rawKey: string; hash: string } | null = null;
  try {
    childKey = await createManagedChildKey({
      managementApiKey: c.env.OPENROUTER_MANAGEMENT_API_KEY,
      installationId,
      releaseSessionRef: operation.operation_id,
      expiresAt,
      limitUsd: issueLimitUsd,
      requireEffectiveLimitVerification: referralLimitVerificationRequired,
      keyName: providerKeyNameForOperationAttempt(operation.operation_id, 'discord', input.attemptIndex),
    });
    await recordAttemptCredential(db, operation.operation_id, input.attemptIndex, childKey.hash, input.now);
    await assignManagedGuardrail({
      managementApiKey: c.env.OPENROUTER_MANAGEMENT_API_KEY,
      guardrailId: c.env.OPENROUTER_MANAGED_GUARDRAIL_ID,
      keyHash: childKey.hash,
    });
    const pending = await markDiscordRetryDeliveryPending(db, {
      installationId,
      devicePublicKey,
      discordUserRef: operation.subject_ref,
      managedCredentialRef: childKey.hash,
      issuedAt,
      expiresAt,
      budgetUsd: issueLimitUsd,
      nowIso: input.nowIso,
    });
    if (!pending) {
      throw new Error('Discord managed entitlement delivery-pending transition failed');
    }
    const delivery = await createManagedKeyDelivery(db, {
      issueSource: 'discord',
      subjectRef: operation.subject_ref,
      installationId,
      managedCredentialRef: childKey.hash,
      createdAt: input.now,
      expiresAt: new Date(input.now.getTime() + MANAGED_KEY_DELIVERY_ACK_TTL_MS),
      operationId: operation.operation_id,
      attemptIndex: input.attemptIndex,
    });
    const settled = await transitionOperationToPostCreateState(db, c.env.OPENROUTER_MANAGEMENT_API_KEY, operation.operation_id, 'DELIVERY_PENDING', input.now);
    if (!settled || settled.state !== 'DELIVERY_PENDING') {
      const attempts = await listManagedOperationAttempts(db, operation.operation_id);
      const current = await getManagedOperationStatusSnapshot(db, operation.operation_id);
      return c.json(await buildManagedOperationStatusBodyWithDelivery(db, current ?? operation, attempts));
    }
    return c.json({
      openrouter_api_key: childKey.rawKey,
      managed_credential_ref: childKey.hash,
      expires_at: expiresAt,
      delivery_ack_required: true,
      delivery_id: delivery.deliveryId,
      delivery_ack_token: delivery.deliveryAckToken,
      delivery_ack_expires_at: new Date(
        input.now.getTime() + MANAGED_KEY_DELIVERY_ACK_TTL_MS,
      ).toISOString(),
    });
  } catch (error) {
    if (
      !childKey &&
      error instanceof OpenRouterManagementError &&
      error.createdChildKey
    ) {
      childKey = error.createdChildKey;
    }
    if (childKey) {
      await recordAttemptCredential(db, operation.operation_id, input.attemptIndex, childKey.hash, input.now);
    }
    await markAttemptUnknown(db, operation.operation_id, input.attemptIndex, input.now);
    if (isDefinitiveManagedChildKeyCreateRejection(error)) {
      await failManagedOperationTerminal(db, operation, input.now, 'terminal_provider_failure');
      const attempts = await listManagedOperationAttempts(db, operation.operation_id);
      const terminal = await getManagedOperationStatusSnapshot(db, operation.operation_id);
      return c.json(await buildManagedOperationStatusBodyWithDelivery(db, terminal ?? operation, attempts));
    }
    await markDiscordIndeterminateChildKeyCreation(db, {
      installationId,
      discordUserRef: operation.subject_ref,
      operationId: operation.operation_id,
      nowIso: input.nowIso,
    }).catch((): boolean => false);
    const attempts = await listManagedOperationAttempts(db, operation.operation_id);
    const current = await getManagedOperationStatusSnapshot(db, operation.operation_id);
    return c.json(await buildManagedOperationStatusBodyWithDelivery(db, current ?? operation, attempts));
  }
}

async function bestEffortRecordIneligibleIssueReferralSkip(
  db: D1Database,
  input: {
    referralId: string | null;
    referredDiscordUserRef: string;
    referredInstallationId: string;
    referredHardwareHash: string;
    referredHardwareHashSaltVersion: number;
    skipReason: IssueReferralSkipReason | null;
    attemptIpDigest?: ReferralAttemptIpDigest | null;
    attemptIpLegacyHash?: string | null;
    operationId?: string | null;
    nowIso: string;
  },
): Promise<void> {
  if (!input.skipReason) {
    return;
  }

  try {
    await recordSkippedIssueReferralReward(db, {
      referralId: input.referralId,
      referredSource: 'discord',
      referredSubjectRef: input.referredDiscordUserRef,
      referredInstallationId: input.referredInstallationId,
      referredHardwareHash: input.referredHardwareHash,
      referredHardwareHashSaltVersion: input.referredHardwareHashSaltVersion,
      skipReason: input.skipReason,
      attemptIpDigest: input.attemptIpDigest ?? null,
      attemptIpLegacyHash: input.attemptIpLegacyHash ?? null,
      operationId: input.operationId ?? null,
      nowIso: input.nowIso,
    });
  } catch {
    // Referral skip accounting is best-effort and must not replace issue errors.
  }
}

function issueReferralSkipReasonForReservationFailure(
  currentEntitlement: OpenRouterEntitlementRecord | null,
  subcode: DiscordReservationErrorSubcode,
): IssueReferralSkipReason | null {
  if (currentEntitlement?.status === 'active') {
    return 'pre_existing_managed_user';
  }

  switch (subcode) {
    case 'discord_lifetime_used':
      return 'referred_not_first_successful';
    case 'hardware_duplicate':
      return 'duplicate_hardware';
    case 'global_cap_reached':
    case 'discord_installation_already_issuing':
    case 'installation_binding_mismatch':
    case 'device_public_key_registered':
    case 'entitlement_reservation_failed':
      return null;
  }
}

async function bestEffortMarkIssueReferralReservationFailed(
  db: D1Database,
  input: {
    referralReservation: IssueReferralReservationResult | null;
    referredDiscordUserRef: string;
    referredInstallationId: string;
    nowIso: string;
  },
): Promise<void> {
  if (input.referralReservation?.outcome !== 'reserved') {
    return;
  }

  try {
    await markReservedIssueReferralFailed(db, {
      referralId: input.referralReservation.referralId,
      referredSource: 'discord',
      referredSubjectRef: input.referredDiscordUserRef,
      referredInstallationId: input.referredInstallationId,
      failureReason: 'issue_delivery_failed',
      nowIso: input.nowIso,
    });
  } catch {
    // Referral accounting is best-effort during managed issue cleanup.
  }
}

async function runDiscordIssueSuccessMonitoring(
  c: Context<BrokerEnv>,
  input: {
    installationId: string;
    managedCredentialRef: string;
    issuedAt: string;
    now: Date;
    sensitiveValues: string[];
  },
): Promise<void> {
  try {
    let issueSuccessRecorded = false;
    let monitoringResult: Awaited<ReturnType<typeof evaluateImmediateAbuseState>> | null =
      null;

    try {
      const network = await extractRequestNetworkMetadata(c, { secrets: resolveRequestNetworkIdentitySecrets(c), now: new Date() });
      await recordIssueSuccess(c.env.BROKER_DB, {
        installationId: input.installationId,
        managedCredentialRef: input.managedCredentialRef,
        observedAt: input.issuedAt,
        network,
      });
      issueSuccessRecorded = true;
      monitoringResult = await evaluateImmediateAbuseState(
        c.env.BROKER_DB,
        input.now,
      );
    } catch (error) {
      logDiscordIssueMonitoringFailure({
        installationId: input.installationId,
        managedCredentialRef: input.managedCredentialRef,
        stage: 'record_or_evaluate',
        error,
        sensitiveValues: input.sensitiveValues,
      });
      throw new DiscordIssueSuccessMonitoringStateError({
        cause: error,
        issueSuccessRecorded,
      });
    }

    const sideEffectPromise = deliverImmediateMonitoringSideEffects(
      c.env,
      monitoringResult,
    ).catch((error) => {
      logDiscordIssueMonitoringFailure({
        installationId: input.installationId,
        managedCredentialRef: input.managedCredentialRef,
        stage: 'deliver_side_effects',
        error,
        sensitiveValues: input.sensitiveValues,
      });
    });

    const waitUntil = resolveExecutionWaitUntil(c);
    if (waitUntil) {
      try {
        waitUntil(sideEffectPromise);
        return;
      } catch {
        // Fall through and await inline when waitUntil is not usable in tests.
      }
    }

    await sideEffectPromise;
  } catch (error) {
    if (error instanceof DiscordIssueSuccessMonitoringStateError) {
      throw error;
    }

    logDiscordIssueMonitoringFailure({
      installationId: input.installationId,
      managedCredentialRef: input.managedCredentialRef,
      stage: 'unexpected',
      error,
      sensitiveValues: input.sensitiveValues,
    });
  }
}

function logDiscordIssueMonitoringFailure(input: {
  installationId: string;
  managedCredentialRef: string;
  stage: 'record_or_evaluate' | 'deliver_side_effects' | 'unexpected';
  error: unknown;
  sensitiveValues: string[];
}): void {
  console.error('discord_issue_success_monitoring_failed', {
    installation_id: input.installationId,
    managed_credential_ref: input.managedCredentialRef,
    stage: input.stage,
    error_message: redactSensitiveString(
      input.error instanceof Error ? input.error.message : String(input.error),
      input.sensitiveValues,
    ),
    broker_timestamp: new Date().toISOString(),
  });
}

async function deleteDiscordIssueSuccessRecord(
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

async function bestEffortDeleteDiscordIssueSuccessRecord(
  db: D1Database,
  input: {
    installationId: string;
    managedCredentialRef: string;
    observedAt: string;
    sensitiveValues: string[];
  },
): Promise<void> {
  try {
    await deleteDiscordIssueSuccessRecord(db, input);
  } catch (error) {
    console.error('discord_issue_success_cleanup_failed', {
      installation_id: input.installationId,
      managed_credential_ref: input.managedCredentialRef,
      observed_at: input.observedAt,
      failure: redactSensitiveDiagnostics(
        normalizeFailureForLog(error),
        input.sensitiveValues,
      ),
      broker_timestamp: new Date().toISOString(),
    });
  }
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

function collectDiscordIssueSensitiveValues(input: {
  input: DiscordOpenRouterIssueInput;
  session: DiscordOAuthSessionRecord;
  discordTokenResponse: Awaited<ReturnType<typeof exchangeDiscordCode>> | null;
  discordUser: DiscordUserResponse;
  childKey: { rawKey: string; hash: string } | null;
}): string[] {
  return [
    input.input.code,
    input.input.state,
    input.session.pkce_code_verifier,
    input.discordTokenResponse?.access_token ?? null,
    input.discordTokenResponse?.refresh_token ?? null,
    input.discordUser.id,
    typeof input.discordUser.email === 'string' ? input.discordUser.email : null,
    input.childKey?.rawKey ?? null,
  ].filter((value): value is string => Boolean(value));
}

function redactSensitiveDiagnostics(value: unknown, sensitiveValues: string[]): unknown {
  if (typeof value === 'string') {
    return redactSensitiveString(value, sensitiveValues);
  }

  if (Array.isArray(value)) {
    return value.map((entry) => redactSensitiveDiagnostics(entry, sensitiveValues));
  }

  if (value && typeof value === 'object') {
    return Object.fromEntries(
      Object.entries(value).map(([key, entry]) => [
        key,
        redactSensitiveDiagnostics(entry, sensitiveValues),
      ]),
    );
  }

  return value;
}

function redactSensitiveString(value: string, sensitiveValues: string[]): string {
  let redacted = value;
  for (const sensitiveValue of [...sensitiveValues].sort(
    (left, right) => right.length - left.length,
  )) {
    redacted = redacted.split(sensitiveValue).join('[REDACTED]');
  }
  return redacted;
}

function normalizeFailureForLog(error: unknown): Record<string, unknown> {
  if (error instanceof Error) {
    return {
      name: error.name,
      message: error.message,
    };
  }

  return {
    name: 'UnknownFailure',
    message: String(error),
  };
}

function discordReservationErrorResponse(
  c: Context<BrokerEnv>,
  reservation: Exclude<DiscordReservationResult, { ok: true }>,
): Response {
  switch (reservation.subcode) {
    case 'discord_lifetime_used':
      return publicErrorResponse(c, 409, {
        code: 'trial_not_eligible',
        class: 'terminal',
        subcode: reservation.subcode,
        message: 'Discord account has already used a managed trial',
        entitlement: null,
      });
    case 'hardware_duplicate':
      return publicErrorResponse(c, 409, {
        code: 'trial_not_eligible',
        class: 'terminal',
        subcode: reservation.subcode,
        message: 'This device has already used a managed trial',
        entitlement: null,
      });
    case 'global_cap_reached':
      return publicErrorResponse(c, 503, {
        code: 'issuance_suspended',
        class: 'retryable',
        subcode: reservation.subcode,
        retryAfterMs: reservation.retryAfterMs ?? null,
        message: 'Daily managed issuance cap reached',
        entitlement: null,
      });
    case 'discord_installation_already_issuing':
      return publicErrorResponse(c, 410, {
        code: 'challenge_expired',
        class: 'retryable',
        subcode: reservation.subcode,
        retryAfterMs: 0,
        message:
          'Discord managed entitlement is already issuing for this installation; restart Discord OAuth onboarding',
        entitlement: null,
      });
    case 'installation_binding_mismatch':
    case 'device_public_key_registered':
      return discordInstallationBindingErrorResponse(c, reservation.subcode, null);
    case 'entitlement_reservation_failed':
      return publicErrorResponse(c, 500, {
        code: 'internal_error',
        class: 'retryable',
        subcode: reservation.subcode,
        message: 'Managed entitlement reservation failed',
        entitlement: null,
      });
  }
}

function discordInstallationBindingErrorResponse(
  c: Context<BrokerEnv>,
  subcode: 'installation_binding_mismatch' | 'device_public_key_registered',
  entitlement: OpenRouterEntitlementRecord | null,
): Response {
  return publicErrorResponse(c, 409, {
    code: 'trial_not_eligible',
    class: 'security_fail',
    subcode,
    message:
      subcode === 'installation_binding_mismatch'
        ? 'installation_id is already bound to a different device_public_key'
        : 'device_public_key is already registered to a different installation_id',
    entitlement,
  });
}

function addMonthsUtc(value: Date, months: number): Date {
  const next = new Date(value.getTime());
  next.setUTCMonth(next.getUTCMonth() + months);
  return next;
}

function buildCanonicalDiscordIssuePayload(input: {
  input: DiscordOpenRouterIssueInput;
  codeHash: string;
}): Uint8Array {
  return textEncoder.encode(
    [
      DISCORD_OPENROUTER_ISSUE_METHOD,
      DISCORD_OPENROUTER_ISSUE_PATH,
      input.input.installationId,
      input.input.devicePublicKey,
      input.input.state,
      input.codeHash,
      input.input.redirectUri,
      input.input.hardwareHash,
      String(input.input.hardwareHashSaltVersion),
      input.input.appVersion,
      input.input.reason,
      String(input.input.budgetUsd),
      input.input.model,
      input.input.issueNonce,
      input.input.signedAt,
    ].join('\n'),
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

function discordStateUnknownResponse(c: Context<BrokerEnv>): Response {
  return publicErrorResponse(c, 410, {
    code: 'challenge_expired',
    class: 'retryable',
    subcode: 'discord_oauth_state_unknown',
    retryAfterMs: 0,
    message: 'Discord OAuth session was not found or expired',
    entitlement: null,
  });
}

function discordSessionExpiredResponse(c: Context<BrokerEnv>): Response {
  return publicErrorResponse(c, 410, {
    code: 'challenge_expired',
    class: 'retryable',
    subcode: 'discord_oauth_session_expired',
    retryAfterMs: 0,
    message: 'Discord OAuth session has expired and must be restarted',
    entitlement: null,
  });
}

function discordOAuthFailedResponse(c: Context<BrokerEnv>): Response {
  return publicErrorResponse(c, 410, {
    code: 'challenge_expired',
    class: 'retryable',
    subcode: 'discord_oauth_failed',
    retryAfterMs: 0,
    message: 'Discord OAuth verification failed; restart Discord OAuth onboarding',
    entitlement: null,
  });
}

function discordSessionTerminalResponse(
  c: Context<BrokerEnv>,
  status: string,
): Response {
  return publicErrorResponse(c, 409, {
    code: 'challenge_invalid',
    class: 'security_fail',
    subcode: `discord_oauth_session_${status}`,
    message: 'Discord OAuth session can no longer issue a managed key',
    entitlement: null,
  });
}

function discordSessionAlreadyProcessingResponse(c: Context<BrokerEnv>): Response {
  return publicErrorResponse(c, 409, {
    code: 'trial_unavailable',
    class: 'retryable',
    subcode: 'discord_oauth_session_processing',
    message: 'Discord OAuth session is already processing',
    entitlement: null,
  });
}

function discordSessionBindingMismatchResponse(c: Context<BrokerEnv>): Response {
  return publicErrorResponse(c, 409, {
    code: 'trial_not_eligible',
    class: 'security_fail',
    subcode: 'discord_session_binding_mismatch',
    message: 'Discord OAuth session binding does not match the issue request',
    entitlement: null,
  });
}

function discordHardwareSaltMismatchResponse(c: Context<BrokerEnv>): Response {
  return publicErrorResponse(c, 409, {
    code: 'trial_not_eligible',
    class: 'terminal',
    subcode: 'hardware_salt_mismatch',
    message: 'hardware_hash_salt_version does not match the pending Discord OAuth session',
    entitlement: null,
  });
}

function discordSignatureSkewResponse(c: Context<BrokerEnv>): Response {
  return publicErrorResponse(c, 401, {
    code: 'challenge_invalid',
    class: 'security_fail',
    subcode: 'timestamp_skew',
    message: 'signed_at must be within ±60 seconds of broker time',
    entitlement: null,
  });
}

function discordSignatureMismatchResponse(c: Context<BrokerEnv>): Response {
  return publicErrorResponse(c, 401, {
    code: 'challenge_invalid',
    class: 'security_fail',
    subcode: 'signature_mismatch',
    message: 'signature verification failed for the registered device_public_key',
    entitlement: null,
  });
}

function discordEligibilityErrorResponse(
  c: Context<BrokerEnv>,
  subcode:
    | 'discord_email_unverified'
    | 'discord_account_too_new'
    | 'discord_invalid_snowflake',
  message: string,
): Response {
  return publicErrorResponse(c, 409, {
    code: 'trial_not_eligible',
    class: 'terminal',
    subcode,
    message,
    entitlement: null,
  });
}

function discordActivationPlaceholderResponse(c: Context<BrokerEnv>): Response {
  return c.json(
    {
      error: {
        code: 'trial_unavailable',
        class: 'retryable',
        subcode: 'not_implemented',
        retry_after_ms: null,
        message: 'Discord OpenRouter activation is not implemented yet',
      },
      ...normalizeManagedState(null),
    },
    501,
  );
}

async function resolveRequestEventIpMatchForPendingLimit(
  db: D1Database,
  context: {
    endpoint: string;
    now: Date;
    ip: string | null;
    installationId: string | null;
    networkIdentitySecrets?: NetworkIdentitySecrets | null;
  },
  windowStart: Date,
): Promise<{ predicate: string; binds: string[] } | null> {
  return resolveBrokerRequestEventIpMatch(db, {
    endpoint: context.endpoint,
    now: context.now,
    ip: context.ip,
    installationId: context.installationId,
    hardwareHash: null,
    networkIdentitySecrets: context.networkIdentitySecrets ?? null,
  }, windowStart);
}

async function checkPendingDiscordOAuthIpLimit(
  db: D1Database,
  context: {
    endpoint: string;
    now: Date;
    ip: string | null;
    installationId: string | null;
    networkIdentitySecrets?: NetworkIdentitySecrets | null;
  },
  pendingControls: BrokerPendingDiscordOAuthSessionsConfig,
): Promise<((c: Context<BrokerEnv>) => Response) | null> {
  if (!context.ip) {
    return null;
  }

  const windowStart = new Date(
    context.now.getTime() - pendingControls.windowMinutes * 60_000,
  );
  const match = await resolveRequestEventIpMatchForPendingLimit(
    db,
    context,
    windowStart,
  );
  if (!match) {
    return null;
  }
  const ipStartCount = await db
    .prepare(
      `SELECT COUNT(*) AS count
         FROM broker_request_events
        WHERE endpoint = ?
          AND (${match.predicate})
          AND observed_at >= ?`,
    )
    .bind(context.endpoint, ...match.binds, windowStart.toISOString())
    .first<{ count: number }>();

  if (Number(ipStartCount?.count ?? 0) > pendingControls.maxPerIp) {
    return (c: Context<BrokerEnv>) =>
      publicErrorResponse(c, 429, {
        code: 'rate_limited',
        class: 'retryable',
        subcode: 'pending_discord_oauth_ip_limit',
        retryAfterMs: pendingControls.windowMinutes * 60_000,
        message: 'pending Discord OAuth session limit exceeded for client IP',
        entitlement: null,
      });
  }

  return null;
}

async function insertPendingDiscordOAuthSession(
  db: D1Database,
  input: {
    stateHash: string;
    installationId: string;
    devicePublicKey: string;
    redirectUri: string;
    pkceCodeVerifier: string;
    issueNonceHash: string;
    fingerprintSaltVersion: number;
    referralId: string | null;
    nowIso: string;
    expiresAt: string;
    maxPendingPerInstallation: number;
  },
): Promise<boolean> {
  const result = await db
    .prepare(
      `INSERT INTO discord_oauth_sessions (
          state_hash,
          installation_id,
          device_public_key,
          redirect_uri,
          pkce_code_verifier,
          issue_nonce_hash,
          fingerprint_salt_version,
          referral_id,
          status,
          created_at,
          expires_at
        )
        SELECT ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?
         WHERE (
           SELECT COUNT(*)
             FROM discord_oauth_sessions
            WHERE installation_id = ?
              AND status = 'pending'
              AND expires_at > ?
         ) < ?`,
    )
    .bind(
      input.stateHash,
      input.installationId,
      input.devicePublicKey,
      input.redirectUri,
      input.pkceCodeVerifier,
      input.issueNonceHash,
      input.fingerprintSaltVersion,
      input.referralId,
      input.nowIso,
      input.expiresAt,
      input.installationId,
      input.nowIso,
      input.maxPendingPerInstallation,
    )
    .run();

  return Number(result.meta.changes ?? 0) === 1;
}

function pendingInstallationLimitResponse(
  c: Context<BrokerEnv>,
  pendingControls: BrokerPendingDiscordOAuthSessionsConfig,
): Response {
  return publicErrorResponse(c, 429, {
    code: 'rate_limited',
    class: 'retryable',
    subcode: 'pending_discord_oauth_installation_limit',
    retryAfterMs: pendingControls.windowMinutes * 60_000,
    message: 'pending Discord OAuth session limit exceeded for installation_id',
    entitlement: null,
  });
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
  return invalidRequestResponse(
    c,
    reason === 'invalid_json'
      ? 'request body must be valid JSON'
      : 'request body must be a JSON object',
  );
}

function invalidRequestResponse(
  c: Context<BrokerEnv>,
  message: string,
): Response {
  return publicErrorResponse(c, 400, {
    code: 'invalid_request',
    class: 'terminal',
    message,
    entitlement: null,
  });
}

async function sha256Base64Url(value: string): Promise<string> {
  const digest = await crypto.subtle.digest('SHA-256', textEncoder.encode(value));
  return encodeBase64Url(new Uint8Array(digest));
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

function toArrayBuffer(bytes: Uint8Array): ArrayBuffer {
  return bytes.buffer.slice(
    bytes.byteOffset,
    bytes.byteOffset + bytes.byteLength,
  ) as ArrayBuffer;
}
