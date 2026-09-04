PRAGMA defer_foreign_keys = on;

DROP TABLE IF EXISTS managed_key_deliveries_current_ack_hash_gate;
DROP TABLE IF EXISTS managed_key_deliveries_operation_link_migration;
DROP TABLE IF EXISTS referral_rewards_operation_link_migration;
DROP TABLE IF EXISTS managed_operation_attempts;
DROP TABLE IF EXISTS managed_operations;
DROP TABLE IF EXISTS referral_rewards_sequence_backup_0018;

CREATE TABLE referral_rewards_sequence_backup_0018 (
  seq INTEGER NOT NULL
) STRICT;

INSERT INTO referral_rewards_sequence_backup_0018 (seq)
SELECT COALESCE((SELECT seq FROM sqlite_sequence WHERE name = 'referral_rewards'), 0);

CREATE TABLE managed_operations (
  operation_id TEXT PRIMARY KEY CHECK (
    length(operation_id) > length('ph-mop-v1_')
    AND substr(operation_id, 1, length('ph-mop-v1_')) = 'ph-mop-v1_'
  ),
  issue_source TEXT NOT NULL CHECK (issue_source IN ('discord', 'qq')),
  subject_ref TEXT NOT NULL CHECK (length(subject_ref) > 0),
  installation_id TEXT CHECK (installation_id IS NULL OR length(installation_id) BETWEEN 1 AND 128),
  device_public_key TEXT CHECK (device_public_key IS NULL OR length(device_public_key) BETWEEN 1 AND 256),
  state TEXT NOT NULL CHECK (
    state IN (
      'AUTHENTICATED', 'ISSUE_READY', 'CREATING', 'CREATE_UNKNOWN', 'RECONCILING',
      'CLEANUP_REQUIRED', 'CLEAN', 'RETRY_READY', 'DELIVERY_PENDING', 'ACTIVE', 'FAILED'
    )
  ),
  attempt_count INTEGER NOT NULL DEFAULT 0 CHECK (attempt_count >= 0),
  current_attempt_index INTEGER NOT NULL DEFAULT 0 CHECK (current_attempt_index >= 0),
  resume_token_hash TEXT NOT NULL UNIQUE CHECK (
    length(resume_token_hash) = length('ph-mop-resume-v1_') + 64
    AND substr(resume_token_hash, 1, length('ph-mop-resume-v1_')) = 'ph-mop-resume-v1_'
    AND substr(resume_token_hash, length('ph-mop-resume-v1_') + 1) NOT GLOB '*[^0-9a-f]*'
  ),
  auth_expires_at TEXT NOT NULL CHECK (length(auth_expires_at) > 0),
  failure_reason TEXT CHECK (
    failure_reason IS NULL
    OR failure_reason IN ('authorization_expired', 'terminal_provider_failure', 'cleanup_failed_terminal')
  ),
  client_action TEXT NOT NULL CHECK (client_action IN ('wait', 'retry_authorized', 'acknowledge_delivery', 'action_required')),
  referral_reward_id INTEGER UNIQUE,
  referral_status TEXT NOT NULL DEFAULT 'none' CHECK (referral_status IN ('none', 'reserved', 'credited', 'skipped', 'failed')),
  settlement_status TEXT NOT NULL DEFAULT 'none' CHECK (settlement_status IN ('none', 'invitee_pending', 'referrer_pending', 'completed')),
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  last_reconciled_at TEXT,
  cleanup_attempts INTEGER NOT NULL DEFAULT 0 CHECK (cleanup_attempts >= 0),
  CHECK (
    (state <> 'FAILED' AND failure_reason IS NULL)
    OR (state = 'FAILED' AND failure_reason IS NOT NULL)
    OR (state <> 'FAILED')
  ),
  CHECK (attempt_count >= current_attempt_index)
) STRICT;

CREATE INDEX idx_managed_operations_state_updated_at
  ON managed_operations(state, updated_at);
CREATE INDEX idx_managed_operations_auth_expires_at
  ON managed_operations(auth_expires_at)
  WHERE state NOT IN ('ACTIVE', 'FAILED');
CREATE INDEX idx_managed_operations_subject_source
  ON managed_operations(issue_source, subject_ref, created_at);

CREATE TABLE managed_operation_attempts (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  operation_id TEXT NOT NULL REFERENCES managed_operations(operation_id) ON DELETE CASCADE,
  attempt_index INTEGER NOT NULL CHECK (attempt_index > 0),
  provider_key_name TEXT NOT NULL CHECK (length(provider_key_name) BETWEEN 1 AND 200),
  managed_credential_ref TEXT CHECK (managed_credential_ref IS NULL OR length(managed_credential_ref) > 0),
  outcome TEXT NOT NULL CHECK (outcome IN ('created', 'unknown', 'cleaned')),
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
) STRICT;

CREATE UNIQUE INDEX idx_managed_operation_attempts_operation_index
  ON managed_operation_attempts(operation_id, attempt_index);
CREATE UNIQUE INDEX idx_managed_operation_attempts_provider_key_name
  ON managed_operation_attempts(provider_key_name);

CREATE TABLE managed_key_deliveries_operation_link_migration (
  delivery_id TEXT PRIMARY KEY,
  issue_source TEXT NOT NULL CHECK (issue_source IN ('discord', 'qq')),
  subject_ref TEXT,
  installation_id TEXT,
  managed_credential_ref TEXT NOT NULL,
  ack_token_hash TEXT NOT NULL UNIQUE,
  status TEXT NOT NULL CHECK (status IN ('pending', 'acknowledged', 'expired', 'cleanup_required')),
  created_at TEXT NOT NULL,
  expires_at TEXT NOT NULL,
  acknowledged_at TEXT,
  failed_at TEXT,
  failure_reason TEXT,
  operation_id TEXT REFERENCES managed_operations(operation_id) ON DELETE SET NULL,
  attempt_index INTEGER CHECK (attempt_index IS NULL OR attempt_index > 0),
  CHECK (length(delivery_id) > 0),
  CHECK (length(managed_credential_ref) > 0),
  CHECK (
    length(ack_token_hash) = length('ph-delivery-ack-token-v1_') + 64
    AND substr(ack_token_hash, 1, length('ph-delivery-ack-token-v1_')) = 'ph-delivery-ack-token-v1_'
    AND substr(ack_token_hash, length('ph-delivery-ack-token-v1_') + 1) NOT GLOB '*[^0-9a-f]*'
  ),
  CHECK (status <> 'acknowledged' OR acknowledged_at IS NOT NULL),
  CHECK (status = 'acknowledged' OR acknowledged_at IS NULL),
  CHECK (status <> 'cleanup_required' OR failed_at IS NOT NULL),
  CHECK (status <> 'pending' OR (acknowledged_at IS NULL AND failed_at IS NULL AND failure_reason IS NULL))
) STRICT;

CREATE TABLE managed_key_deliveries_current_ack_hash_gate (
  leftover_pending INTEGER NOT NULL CHECK (leftover_pending = 0)
) STRICT;

INSERT INTO managed_key_deliveries_current_ack_hash_gate (leftover_pending)
SELECT COUNT(*)
FROM managed_key_deliveries
WHERE status = 'pending'
  AND NOT (
    length(ack_token_hash) = length('ph-delivery-ack-token-v1_') + 64
    AND substr(ack_token_hash, 1, length('ph-delivery-ack-token-v1_')) = 'ph-delivery-ack-token-v1_'
    AND substr(ack_token_hash, length('ph-delivery-ack-token-v1_') + 1) NOT GLOB '*[^0-9a-f]*'
  );

DROP TABLE managed_key_deliveries_current_ack_hash_gate;

INSERT INTO managed_key_deliveries_operation_link_migration (
  delivery_id, issue_source, subject_ref, installation_id, managed_credential_ref,
  ack_token_hash, status, created_at, expires_at, acknowledged_at, failed_at, failure_reason,
  operation_id, attempt_index
)
SELECT
  delivery_id, issue_source, subject_ref, installation_id, managed_credential_ref,
  ack_token_hash, status, created_at, expires_at, acknowledged_at, failed_at, failure_reason,
  NULL, NULL
FROM managed_key_deliveries
WHERE length(ack_token_hash) = length('ph-delivery-ack-token-v1_') + 64
  AND substr(ack_token_hash, 1, length('ph-delivery-ack-token-v1_')) = 'ph-delivery-ack-token-v1_'
  AND substr(ack_token_hash, length('ph-delivery-ack-token-v1_') + 1) NOT GLOB '*[^0-9a-f]*';

DROP TABLE managed_key_deliveries;
ALTER TABLE managed_key_deliveries_operation_link_migration RENAME TO managed_key_deliveries;

CREATE INDEX idx_managed_key_deliveries_status_expires_at
  ON managed_key_deliveries(status, expires_at);
CREATE INDEX idx_managed_key_deliveries_managed_credential_ref
  ON managed_key_deliveries(managed_credential_ref);
CREATE INDEX idx_managed_key_deliveries_issue_source_created_at
  ON managed_key_deliveries(issue_source, created_at);
CREATE INDEX idx_managed_key_deliveries_operation_id
  ON managed_key_deliveries(operation_id)
  WHERE operation_id IS NOT NULL;

CREATE TABLE referral_rewards_operation_link_migration (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  referral_id TEXT NOT NULL CHECK (
    length(referral_id) = 6
    AND referral_id NOT GLOB '*[^23456789ABCDEFGHJKMNPQRSTUVWXYZ]*'
  ),
  referrer_source TEXT,
  referrer_subject_ref TEXT,
  referrer_discord_user_ref TEXT,
  referrer_installation_id TEXT,
  referred_source TEXT,
  referred_subject_ref TEXT,
  referred_discord_user_ref TEXT,
  referred_installation_id TEXT,
  referred_hardware_hash TEXT CHECK (
    referred_hardware_hash IS NULL OR length(referred_hardware_hash) BETWEEN 1 AND 128
  ),
  referred_hardware_hash_salt_version INTEGER,
  referred_bonus_status TEXT NOT NULL CHECK (referred_bonus_status IN ('reserved', 'credited', 'skipped', 'failed')),
  referrer_bonus_status TEXT NOT NULL CHECK (referrer_bonus_status IN ('pending', 'applying', 'credited', 'skipped', 'failed')),
  skip_reason TEXT CHECK (skip_reason IS NULL OR length(skip_reason) BETWEEN 1 AND 64),
  failure_reason TEXT CHECK (failure_reason IS NULL OR length(failure_reason) BETWEEN 1 AND 64),
  referred_managed_credential_ref TEXT,
  referrer_managed_credential_ref TEXT,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  credited_at TEXT,
  attempt_ip_hash TEXT,
  operation_id TEXT UNIQUE REFERENCES managed_operations(operation_id) ON DELETE SET NULL,
  CHECK (
    (referrer_source IS NULL AND referrer_subject_ref IS NULL)
    OR (referrer_source IS NOT NULL AND referrer_subject_ref IS NOT NULL)
  ),
  CHECK (
    referrer_discord_user_ref IS NULL
    OR referrer_source IS NULL
    OR (
      referrer_source = 'discord'
      AND referrer_subject_ref = referrer_discord_user_ref
    )
  ),
  CHECK (referrer_source <> 'qq' OR referrer_discord_user_ref IS NULL),
  CHECK (
    referrer_source IS NULL
    OR (referrer_source = 'discord' AND referrer_subject_ref GLOB 'ph-discord-user-v*_*')
    OR (referrer_source = 'qq' AND referrer_subject_ref GLOB 'ph-qq-subject-v1_*')
  ),
  CHECK (
    (referred_source IS NULL AND referred_subject_ref IS NULL AND referred_discord_user_ref IS NOT NULL)
    OR (referred_source IS NOT NULL AND referred_subject_ref IS NOT NULL)
  ),
  CHECK (
    referred_discord_user_ref IS NULL
    OR referred_source IS NULL
    OR (
      referred_source = 'discord'
      AND referred_subject_ref = referred_discord_user_ref
    )
  ),
  CHECK (referred_source <> 'qq' OR referred_discord_user_ref IS NULL),
  CHECK (
    (
      COALESCE(
        referred_source,
        CASE WHEN referred_discord_user_ref IS NOT NULL THEN 'discord' END
      ) = 'discord'
      AND COALESCE(referred_subject_ref, referred_discord_user_ref) GLOB 'ph-discord-user-v*_*'
      AND referred_installation_id IS NOT NULL
      AND referred_hardware_hash IS NOT NULL
      AND referred_hardware_hash_salt_version IS NOT NULL
    )
    OR (
      referred_source = 'qq'
      AND referred_subject_ref GLOB 'ph-qq-subject-v1_*'
      AND referred_discord_user_ref IS NULL
      AND referred_hardware_hash IS NULL
      AND referred_hardware_hash_salt_version IS NULL
    )
  ),
  CHECK (
    COALESCE(
      referrer_source,
      CASE WHEN referrer_discord_user_ref IS NOT NULL THEN 'discord' END
    ) IS NOT NULL
    OR (
      referred_bonus_status = 'skipped'
      AND referrer_bonus_status = 'skipped'
      AND skip_reason IS NOT NULL
    )
  )
) STRICT;

INSERT INTO referral_rewards_operation_link_migration (
  id, referral_id, referrer_source, referrer_subject_ref, referrer_discord_user_ref,
  referrer_installation_id, referred_source, referred_subject_ref, referred_discord_user_ref,
  referred_installation_id, referred_hardware_hash, referred_hardware_hash_salt_version,
  referred_bonus_status, referrer_bonus_status, skip_reason, failure_reason,
  referred_managed_credential_ref, referrer_managed_credential_ref,
  created_at, updated_at, credited_at, attempt_ip_hash, operation_id
)
SELECT
  id, referral_id, referrer_source, referrer_subject_ref, referrer_discord_user_ref,
  referrer_installation_id, referred_source, referred_subject_ref, referred_discord_user_ref,
  referred_installation_id, referred_hardware_hash, referred_hardware_hash_salt_version,
  referred_bonus_status, referrer_bonus_status, skip_reason, failure_reason,
  referred_managed_credential_ref, referrer_managed_credential_ref,
  created_at, updated_at, credited_at, attempt_ip_hash, NULL
FROM referral_rewards;

DROP TABLE referral_rewards;
ALTER TABLE referral_rewards_operation_link_migration RENAME TO referral_rewards;

CREATE INDEX idx_referral_rewards_referral_id_created_at
  ON referral_rewards(referral_id, created_at);
CREATE INDEX idx_referral_rewards_referrer_subject_status
  ON referral_rewards(referrer_source, referrer_subject_ref, referred_bonus_status);
CREATE INDEX idx_referral_rewards_referred_subject_created_at
  ON referral_rewards(referred_source, referred_subject_ref, created_at);
CREATE INDEX idx_referral_rewards_referred_installation_created_at
  ON referral_rewards(referred_installation_id, created_at)
  WHERE referred_installation_id IS NOT NULL;
CREATE INDEX idx_referral_rewards_attempt_ip_created_at
  ON referral_rewards(attempt_ip_hash, created_at)
  WHERE attempt_ip_hash IS NOT NULL;
CREATE INDEX idx_referral_rewards_referrer_subject_created_at
  ON referral_rewards(referrer_source, referrer_subject_ref, created_at);
CREATE UNIQUE INDEX idx_referral_rewards_counted_referred_subject
  ON referral_rewards(referred_source, referred_subject_ref)
  WHERE referred_bonus_status IN ('reserved', 'credited');
CREATE UNIQUE INDEX idx_referral_rewards_counted_referred_installation
  ON referral_rewards(referred_installation_id)
  WHERE referred_installation_id IS NOT NULL AND referred_bonus_status IN ('reserved', 'credited');

WITH sequence_repair(seq) AS (
  SELECT max(
    COALESCE((SELECT seq FROM referral_rewards_sequence_backup_0018), 0),
    COALESCE((SELECT MAX(id) FROM referral_rewards), 0)
  )
)
UPDATE sqlite_sequence
   SET seq = (SELECT seq FROM sequence_repair)
 WHERE name = 'referral_rewards'
   AND seq < (SELECT seq FROM sequence_repair);

INSERT INTO sqlite_sequence (name, seq)
SELECT 'referral_rewards', (SELECT max(
    COALESCE((SELECT seq FROM referral_rewards_sequence_backup_0018), 0),
    COALESCE((SELECT MAX(id) FROM referral_rewards), 0)
  ))
 WHERE NOT EXISTS (SELECT 1 FROM sqlite_sequence WHERE name = 'referral_rewards');

DROP TABLE referral_rewards_sequence_backup_0018;
