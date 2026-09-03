import {
  MANAGED_TRIAL_POLICY,
  TRIAL_PROVIDER_POLICY,
} from './trial-policy';

export {
  BROKER_PUBLIC_INPUT_BOUNDS,
  BROKER_PERSISTENCE_MODEL,
  BROKER_RETENTION_POLICY,
  BROKER_RUNTIME_CONFIG_KEYS,
  BROKER_RUNTIME_CONFIG_SCHEMA,
  DEFAULT_BROKER_ABUSE_CONTROLS,
  DEFAULT_BROKER_ABUSE_RUNTIME_STATE,
  FINGERPRINT_SALT_POLICY,
  MANAGED_KEY_DELIVERY_STATUS_VALUES,
  OPENROUTER_ENTITLEMENT_STATUS_VALUES,
  QQ_MANAGED_ENTITLEMENT_AUTOMATIC_REISSUE_BLOCKING_STATUS_VALUES,
  QQ_MANAGED_ENTITLEMENT_STALE_ISSUING_POLICY,
  QQ_MANAGED_ENTITLEMENT_STATUS_VALUES,
  MANAGED_REFERRAL_SETTLEMENT_PHASE_VALUES,
  QQ_SUBJECT_REF_POLICY,
  REFERRAL_CODE_STATUS_VALUES,
  REFERRAL_ID_FORMAT_DESCRIPTION,
  REFERRAL_REFERRED_BONUS_STATUS_VALUES,
  REFERRAL_REFERRER_BONUS_STATUS_VALUES,
} from './persistence';
export {
  applyReferralRewardRetention,
  disableReferralId,
  ensureOwnedReferralIdForActiveDiscordManagedUser,
  generateReferralId,
  normalizeReferralId,
  reconcileStaleReferralRewards,
  REFERRAL_ID_ALPHABET,
  REFERRAL_ID_LENGTH,
} from './referral';
export {
  MANAGED_TRIAL_BUDGET_POLICY,
  MANAGED_TRIAL_COST_ACCOUNTING_POLICY,
  MANAGED_TRIAL_ENTITLEMENT_POLICY,
  MANAGED_TRIAL_LIFECYCLE_VALUES,
  MANAGED_TRIAL_LIVE_USAGE_POLICY,
  MANAGED_TRIAL_POLICY,
  TRIAL_PROVIDER_POLICY,
} from './trial-policy';
export type {
  BrokerAbuseRuntimeAuditRecord,
  BrokerAbuseRuntimeStateValue,
  BrokerDailyIssuanceCapConfig,
  BrokerDailyIssuanceCapEndpoint,
  BrokerEndpointRateLimitConfig,
  BrokerPendingDiscordOAuthSessionsConfig,
  QqAuthAssertionRecord,
  BrokerReferralAttemptControlsConfig,
  BrokerAbuseControlsConfigValue,
  BrokerAbuseSubjectHookRecord,
  BrokerConfigRow,
  BrokerIssueSuccessSource,
  BrokerIssueSuccessEventRecord,
  BrokerRequestEventRecord,
  BrokerVelocityCapHookRecord,
  DiscordOAuthSessionRecord,
  DiscordOAuthSessionStatus,
  FingerprintSaltConfigValue,
  FingerprintSaltVersion,
  InstallationRecord,
  OpenRouterEntitlementRecord,
  OpenRouterEntitlementStatus,
  ManagedKeyDeliveryRecord,
  ManagedKeyDeliveryStatus,
  QqManagedEntitlementRecord,
  QqManagedEntitlementStatus,
  ManagedReferralSettlementPhase,
  ManagedReferralSettlementJobRecord,
  ReferralCodeRecord,
  ReferralCodeStatus,
  ReferralReferredBonusStatus,
  ReferralReferrerBonusStatus,
  ReferralRewardRecord,
} from './persistence';
export type {
  OwnedReferralIdEnsureFailureReason,
  OwnedReferralIdEnsureResult,
  DisableReferralIdResult,
  IssueReferralFailureReason,
  IssueReferralSkipReason,
  ReferralDisableReason,
  ReferralIdGenerator,
  ReferralIdRandomBytes,
  ReferralRewardRetentionResult,
  StaleReferralRewardReconciliationResult,
} from './referral';
export type { ManagedTrialLifecycle } from './trial-policy';

export const BROKER_SERVICE_NAME = 'puripuly-heart-broker';

export const BROKER_RUNTIME_STACK = {
  language: 'TypeScript',
  framework: 'Hono',
  runtime: 'Cloudflare Workers',
  database: 'Cloudflare D1',
  secretStorage: 'Worker secrets',
} as const;

export const REQUIRED_BINDINGS = {
  d1: 'BROKER_DB',
  // Transitional: the shared-key binding remains required until the runtime
  // issuance path switches away from OPENROUTER_MANAGED_API_KEY in a later task.
  secrets: [
    'OPENROUTER_MANAGED_API_KEY',
    'OPENROUTER_MANAGEMENT_API_KEY',
    'OPENROUTER_MANAGED_GUARDRAIL_ID',
    'OPENROUTER_MANAGED_USER_HMAC_SECRET',
    'DISCORD_CLIENT_ID',
    'DISCORD_CLIENT_SECRET',
    'DISCORD_REDIRECT_URI_ALLOWLIST',
    'DISCORD_USER_REF_SECRET',
    'DISCORD_IMMEDIATE_ALERT_WEBHOOK_URL',
    'DISCORD_DAILY_REPORT_WEBHOOK_URL',
    'QQ_AUTH_HMAC_PSK',
    'TELEMETRY_SUBJECT_HMAC_SECRET',
    'NETWORK_IDENTITY_HMAC_SECRET',
    'NETWORK_IDENTITY_HMAC_SECRET_PREVIOUS',
  ],
} as const;

export interface BrokerBindings {
  BROKER_DB: D1Database;
  OPENROUTER_MANAGEMENT_API_KEY: string;
  OPENROUTER_MANAGED_GUARDRAIL_ID: string;
  OPENROUTER_MANAGED_API_KEY: string;
  OPENROUTER_MANAGED_USER_HMAC_SECRET: string;
  DISCORD_CLIENT_ID: string;
  DISCORD_CLIENT_SECRET: string;
  DISCORD_REDIRECT_URI_ALLOWLIST: string;
  DISCORD_USER_REF_SECRET: string;
  DISCORD_IMMEDIATE_ALERT_WEBHOOK_URL: string;
  DISCORD_DAILY_REPORT_WEBHOOK_URL: string;
  QQ_AUTH_HMAC_PSK: string;
  TELEMETRY_SUBJECT_HMAC_SECRET: string;
  NETWORK_IDENTITY_HMAC_SECRET: string;
  NETWORK_IDENTITY_HMAC_SECRET_PREVIOUS?: string;
  NETWORK_IDENTITY_HMAC_KEY_VERSION: string;
  NETWORK_IDENTITY_HMAC_KEY_VERSION_PREVIOUS?: string;
}

export type BrokerEnv = {
  Bindings: BrokerBindings;
};

export const HOSTING_ASSUMPTIONS = {
  regionMode: 'single-region-rollout-assumption',
  d1LocationHint: 'apac',
  infrastructure: ['worker-service', 'd1-database', 'worker-secrets'],
  exclusions: [
    'translation-proxying',
    'multi-region-deployment',
    'kv',
    'r2',
    'admin-dashboard',
  ],
} as const;

export const SERVICE_BOUNDARY = {
  role: 'trial-credential-broker',
  proxiesTranslationText: false,
  inferencePath: 'app-direct-to-openrouter',
} as const;

export const FOUNDATION_RESPONSE = {
  service: BROKER_SERVICE_NAME,
  runtime: BROKER_RUNTIME_STACK,
  bindings: REQUIRED_BINDINGS,
  hosting: HOSTING_ASSUMPTIONS,
  serviceBoundary: SERVICE_BOUNDARY,
  trialProviderPolicy: TRIAL_PROVIDER_POLICY,
  managedTrialPolicy: MANAGED_TRIAL_POLICY,
} as const;
