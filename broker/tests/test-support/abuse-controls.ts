import * as networkIdentity from '../../src/network-identity';
import type { TestBrokerEnv } from './sqlite-d1';

export interface StoredAbuseControls {
  trialChallenge: {
    endpoint: string;
    scope: 'ip';
    maxRequests: number;
    windowMinutes: number;
  };
  trialChallengeVerify: {
    endpoint: string;
    scope: 'installation_id';
    maxRequests: number;
    windowMinutes: number;
  };
  openrouterIssue: {
    endpoint: string;
    scope: 'installation_id';
    maxRequests: number;
    windowMinutes: number;
  };
  trialStatus: {
    endpoint: string;
    scope: 'installation_id';
    maxRequests: number;
    windowMinutes: number;
  };
  discordAuthStartIp: {
    endpoint: string;
    scope: 'ip';
    maxRequests: number;
    windowMinutes: number;
  };
  discordAuthStartInstallation: {
    endpoint: string;
    scope: 'installation_id';
    maxRequests: number;
    windowMinutes: number;
  };
  discordOpenrouterIssueIp: {
    endpoint: string;
    scope: 'ip';
    maxRequests: number;
    windowMinutes: number;
  };
  discordOpenrouterIssueInstallation: {
    endpoint: string;
    scope: 'installation_id';
    maxRequests: number;
    windowMinutes: number;
  };
  qqAuthAssertIp: {
    endpoint: string;
    scope: 'ip';
    maxRequests: number;
    windowMinutes: number;
  };
  managedOperationStatusIp: {
    endpoint: string;
    scope: 'ip';
    maxRequests: number;
    windowMinutes: number;
  };
  managedOperationStatusInstallation: {
    endpoint: string;
    scope: 'installation_id';
    maxRequests: number;
    windowMinutes: number;
  };
  managedOperationResumeIp: {
    endpoint: string;
    scope: 'ip';
    maxRequests: number;
    windowMinutes: number;
  };
  managedOperationResumeInstallation: {
    endpoint: string;
    scope: 'installation_id';
    maxRequests: number;
    windowMinutes: number;
  };
  managedKeyDeliveryAckIp: {
    endpoint: string;
    scope: 'ip';
    maxRequests: number;
    windowMinutes: number;
  };
  pendingDiscordOAuthSessions: {
    maxPerInstallation: number;
    maxPerIp: number;
    windowMinutes: number;
  };
  newActiveEntitlementsPerDay: {
    endpoint: string;
    scope: 'global';
    maxCount: number | null;
    windowDays: number;
  };
  immediateAlerts: {
    warning: number;
    brake: number;
  };
  retention: {
    requestEventSafetyMarginDays: number;
    issueSuccessDays: number;
    runtimeAuditDays: number;
  };
  dailyReport: {
    enabled: boolean;
    hourUtc: number;
    minuteUtc: number;
  };
}

export interface StoredAbuseRuntimeState {
  brake: {
    active: boolean;
    reason: 'global_threshold' | 'asn_fast_path' | 'manual' | null;
    changedAt: string | null;
    changedBy: 'system' | 'operator' | null;
  };
  alertLatches: {
    warning: boolean;
    warningObservedAt: string | null;
  };
  dailyReport: {
    lastDeliveredAt: string | null;
    lastDeliveredDateUtc: string | null;
  };
}

export const TEST_DEFAULT_ABUSE_CONTROLS: StoredAbuseControls = {
  trialChallenge: {
    endpoint: 'POST /v1/trial/challenge',
    scope: 'ip',
    maxRequests: 10,
    windowMinutes: 15,
  },
  trialChallengeVerify: {
    endpoint: 'POST /v1/trial/challenge/verify',
    scope: 'installation_id',
    maxRequests: 5,
    windowMinutes: 15,
  },
  openrouterIssue: {
    endpoint: 'POST /v1/providers/openrouter/issue',
    scope: 'installation_id',
    maxRequests: 3,
    windowMinutes: 15,
  },
  trialStatus: {
    endpoint: 'GET /v1/trial/status',
    scope: 'installation_id',
    maxRequests: 30,
    windowMinutes: 15,
  },
  discordAuthStartIp: {
    endpoint: 'POST /v1/auth/discord/start',
    scope: 'ip',
    maxRequests: 20,
    windowMinutes: 15,
  },
  discordAuthStartInstallation: {
    endpoint: 'POST /v1/auth/discord/start',
    scope: 'installation_id',
    maxRequests: 5,
    windowMinutes: 15,
  },
  discordOpenrouterIssueIp: {
    endpoint: 'POST /v1/providers/openrouter/discord/issue',
    scope: 'ip',
    maxRequests: 10,
    windowMinutes: 15,
  },
  discordOpenrouterIssueInstallation: {
    endpoint: 'POST /v1/providers/openrouter/discord/issue',
    scope: 'installation_id',
    maxRequests: 3,
    windowMinutes: 15,
  },
  qqAuthAssertIp: {
    endpoint: 'POST /v1/auth/qq/assert',
    scope: 'ip',
    maxRequests: 20,
    windowMinutes: 15,
  },
  managedOperationStatusIp: {
    endpoint: 'POST /v1/providers/openrouter/managed-operation/status',
    scope: 'ip',
    maxRequests: 30,
    windowMinutes: 15,
  },
  managedOperationStatusInstallation: {
    endpoint: 'POST /v1/providers/openrouter/managed-operation/status',
    scope: 'installation_id',
    maxRequests: 30,
    windowMinutes: 15,
  },
  managedOperationResumeIp: {
    endpoint: 'POST /v1/providers/openrouter/managed-operation/resume',
    scope: 'ip',
    maxRequests: 20,
    windowMinutes: 15,
  },
  managedOperationResumeInstallation: {
    endpoint: 'POST /v1/providers/openrouter/managed-operation/resume',
    scope: 'installation_id',
    maxRequests: 10,
    windowMinutes: 15,
  },
  managedKeyDeliveryAckIp: {
    endpoint: 'POST /v1/providers/openrouter/managed-key-delivery/ack',
    scope: 'ip',
    maxRequests: 30,
    windowMinutes: 15,
  },
  pendingDiscordOAuthSessions: {
    maxPerInstallation: 2,
    maxPerIp: 20,
    windowMinutes: 15,
  },
  newActiveEntitlementsPerDay: {
    endpoint: 'POST /v1/providers/openrouter/discord/issue',
    scope: 'global',
    maxCount: 500,
    windowDays: 1,
  },
  immediateAlerts: {
    warning: 10,
    brake: 70,
  },
  retention: {
    requestEventSafetyMarginDays: 1,
    issueSuccessDays: 30,
    runtimeAuditDays: 90,
  },
  dailyReport: {
    enabled: true,
    hourUtc: 0,
    minuteUtc: 5,
  },
};

export const TEST_DEFAULT_ABUSE_RUNTIME_STATE: StoredAbuseRuntimeState = {
  brake: {
    active: false,
    reason: null,
    changedAt: null,
    changedBy: null,
  },
  alertLatches: {
    warning: false,
    warningObservedAt: null,
  },
  dailyReport: {
    lastDeliveredAt: null,
    lastDeliveredDateUtc: null,
  },
};

export function readAbuseControls(env: TestBrokerEnv): StoredAbuseControls {
  const row = env.__db
    .prepare('SELECT value FROM broker_config WHERE key = ?')
    .get('abuse_controls') as { value: string } | undefined;

  if (!row) {
    throw new Error('missing broker_config row: abuse_controls');
  }

  return JSON.parse(row.value) as StoredAbuseControls;
}

export function updateAbuseControls(
  env: TestBrokerEnv,
  mutate: (controls: StoredAbuseControls) => void,
): void {
  const controls = readAbuseControls(env);
  mutate(controls);
  env.__db
    .prepare('UPDATE broker_config SET value = ?, updated_at = ? WHERE key = ?')
    .run(JSON.stringify(controls), new Date().toISOString(), 'abuse_controls');
}

export function replaceAbuseControlsValue(env: TestBrokerEnv, value: unknown): void {
  env.__db
    .prepare('UPDATE broker_config SET value = ?, updated_at = ? WHERE key = ?')
    .run(JSON.stringify(value), new Date().toISOString(), 'abuse_controls');
}

export function readAbuseRuntimeState(env: TestBrokerEnv): StoredAbuseRuntimeState {
  const row = env.__db
    .prepare('SELECT value FROM broker_config WHERE key = ?')
    .get('abuse_runtime_state') as { value: string } | undefined;

  if (!row) {
    throw new Error('missing broker_config row: abuse_runtime_state');
  }

  return JSON.parse(row.value) as StoredAbuseRuntimeState;
}

export function updateAbuseRuntimeState(
  env: TestBrokerEnv,
  mutate: (state: StoredAbuseRuntimeState) => void,
): void {
  const state = readAbuseRuntimeState(env);
  mutate(state);
  env.__db
    .prepare('UPDATE broker_config SET value = ?, updated_at = ? WHERE key = ?')
    .run(JSON.stringify(state), new Date().toISOString(), 'abuse_runtime_state');
}

export function replaceAbuseRuntimeStateValue(env: TestBrokerEnv, value: unknown): void {
  env.__db
    .prepare('UPDATE broker_config SET value = ?, updated_at = ? WHERE key = ?')
    .run(JSON.stringify(value), new Date().toISOString(), 'abuse_runtime_state');
}

export function insertVelocityCapHook(
  env: TestBrokerEnv,
  input: {
    subject_type: 'ip' | 'installation_id';
    subject_value: string;
    max_requests: number;
    window_minutes: number;
    outcome_code: string;
    outcome_class: string;
    outcome_subcode?: string | null;
    reason?: string | null;
    expires_at?: string | null;
  },
): void {
  env.__db
    .prepare(
      `INSERT INTO broker_velocity_cap_hooks (
          subject_type,
          subject_value,
          max_requests,
          window_minutes,
          outcome_code,
          outcome_class,
          outcome_subcode,
          reason,
          expires_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`,
    )
    .run(
      input.subject_type,
      input.subject_value,
      input.max_requests,
      input.window_minutes,
      input.outcome_code,
      input.outcome_class,
      input.outcome_subcode ?? null,
      input.reason ?? null,
      input.expires_at ?? null,
    );
}

export function insertSubjectHook(
  env: TestBrokerEnv,
  input: {
    hook_kind: 'denylist' | 'reputation' | 'revocation';
    subject_type: 'ip' | 'installation_id' | 'hardware_hash';
    subject_value: string;
    outcome_code: string;
    outcome_class: string;
    outcome_subcode?: string | null;
    reason?: string | null;
    expires_at?: string | null;
  },
): void {
  env.__db
    .prepare(
      `INSERT INTO broker_abuse_subject_hooks (
          hook_kind,
          subject_type,
          subject_value,
          outcome_code,
          outcome_class,
          outcome_subcode,
          reason,
          expires_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
    )
    .run(
      input.hook_kind,
      input.subject_type,
      input.subject_value,
      input.outcome_code,
      input.outcome_class,
      input.outcome_subcode ?? null,
      input.reason ?? null,
      input.expires_at ?? null,
    );
}

export async function stableIpHookValue(
  env: TestBrokerEnv,
  ip: string,
): Promise<string> {
  const { deriveStableNetworkIdentityDigest, normalizeNetworkIdentityIp } = networkIdentity;
  const normalized = normalizeNetworkIdentityIp(ip);
  if (!normalized) {
    throw new Error('test IP normalization failed');
  }
  const digests = await deriveStableNetworkIdentityDigest(
    {
      current: env.NETWORK_IDENTITY_HMAC_SECRET,
      previous: null,
      previousVersion: null,
      currentVersion: 1,
    },
    normalized,
    'ip',
  );
  return digests[0]?.digest ?? '';
}
