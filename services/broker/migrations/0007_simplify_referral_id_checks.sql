PRAGMA defer_foreign_keys = on;

CREATE TABLE discord_oauth_sessions_referral_id_checks_v2 (
  state_hash TEXT PRIMARY KEY,
  installation_id TEXT NOT NULL,
  device_public_key TEXT NOT NULL,
  redirect_uri TEXT NOT NULL,
  pkce_code_verifier TEXT,
  issue_nonce_hash TEXT NOT NULL,
  fingerprint_salt_version INTEGER NOT NULL,
  discord_user_ref TEXT,
  discord_email_verified INTEGER CHECK (discord_email_verified IS NULL OR discord_email_verified IN (0, 1)),
  discord_account_created_at TEXT,
  eligibility_checked_at TEXT,
  status TEXT NOT NULL CHECK(status IN ('pending', 'processing', 'consumed', 'canceled', 'failed', 'expired')),
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  expires_at TEXT NOT NULL,
  processing_started_at TEXT,
  consumed_at TEXT,
  referral_id TEXT CHECK (
    referral_id IS NULL
    OR (
      length(referral_id) = 6
      AND referral_id NOT GLOB '*[^23456789ABCDEFGHJKMNPQRSTUVWXYZ]*'
    )
  ),
  CHECK (length(installation_id) BETWEEN 1 AND 128),
  CHECK (length(device_public_key) > 0),
  CHECK (length(redirect_uri) > 0)
) STRICT;

INSERT INTO discord_oauth_sessions_referral_id_checks_v2 (
  state_hash,
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
)
SELECT
  state_hash,
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
FROM discord_oauth_sessions;

DROP TABLE discord_oauth_sessions;
ALTER TABLE discord_oauth_sessions_referral_id_checks_v2 RENAME TO discord_oauth_sessions;

CREATE INDEX idx_discord_oauth_sessions_installation_status
  ON discord_oauth_sessions(installation_id, status, created_at);
CREATE INDEX idx_discord_oauth_sessions_expires_at
  ON discord_oauth_sessions(expires_at);
CREATE INDEX idx_discord_oauth_sessions_referral_id
  ON discord_oauth_sessions(referral_id)
  WHERE referral_id IS NOT NULL;

CREATE TABLE referral_codes_referral_id_checks_v2 (
  referral_id TEXT PRIMARY KEY CHECK (
    length(referral_id) = 6
    AND referral_id NOT GLOB '*[^23456789ABCDEFGHJKMNPQRSTUVWXYZ]*'
  ),
  owner_discord_user_ref TEXT NOT NULL CHECK (length(owner_discord_user_ref) > 0),
  owner_installation_id TEXT,
  status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'disabled')),
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  disabled_reason TEXT CHECK (
    disabled_reason IS NULL OR length(disabled_reason) BETWEEN 1 AND 64
  ),
  disabled_by TEXT CHECK (
    disabled_by IS NULL OR length(disabled_by) BETWEEN 1 AND 64
  ),
  disabled_at TEXT
) STRICT;

INSERT INTO referral_codes_referral_id_checks_v2 (
  referral_id,
  owner_discord_user_ref,
  owner_installation_id,
  status,
  created_at,
  updated_at,
  disabled_reason,
  disabled_by,
  disabled_at
)
SELECT
  referral_id,
  owner_discord_user_ref,
  owner_installation_id,
  status,
  created_at,
  updated_at,
  disabled_reason,
  disabled_by,
  disabled_at
FROM referral_codes;

DROP TABLE referral_codes;
ALTER TABLE referral_codes_referral_id_checks_v2 RENAME TO referral_codes;

CREATE UNIQUE INDEX idx_referral_codes_owner_discord_user_ref
  ON referral_codes(owner_discord_user_ref);
CREATE INDEX idx_referral_codes_owner_installation_id
  ON referral_codes(owner_installation_id)
  WHERE owner_installation_id IS NOT NULL;
CREATE INDEX idx_referral_codes_status
  ON referral_codes(status, referral_id);

CREATE TABLE referral_rewards_sequence_0007 (
  seq INTEGER NOT NULL
) STRICT;

INSERT INTO referral_rewards_sequence_0007 (seq)
SELECT COALESCE(MAX(seq), 0)
  FROM sqlite_sequence
 WHERE name = 'referral_rewards';

CREATE TABLE referral_rewards_referral_id_checks_v2 (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  referral_id TEXT NOT NULL CHECK (
    length(referral_id) = 6
    AND referral_id NOT GLOB '*[^23456789ABCDEFGHJKMNPQRSTUVWXYZ]*'
  ),
  referrer_discord_user_ref TEXT CHECK (
    referrer_discord_user_ref IS NULL OR length(referrer_discord_user_ref) > 0
  ),
  referrer_installation_id TEXT CHECK (
    referrer_installation_id IS NULL OR length(referrer_installation_id) > 0
  ),
  referred_discord_user_ref TEXT NOT NULL CHECK (length(referred_discord_user_ref) > 0),
  referred_installation_id TEXT NOT NULL CHECK (length(referred_installation_id) > 0),
  referred_hardware_hash TEXT NOT NULL CHECK (length(referred_hardware_hash) BETWEEN 1 AND 128),
  referred_hardware_hash_salt_version INTEGER NOT NULL CHECK (referred_hardware_hash_salt_version > 0),
  referred_bonus_status TEXT NOT NULL CHECK (referred_bonus_status IN ('reserved', 'credited', 'skipped', 'failed')),
  referrer_bonus_status TEXT NOT NULL CHECK (referrer_bonus_status IN ('pending', 'applying', 'credited', 'skipped', 'failed')),
  skip_reason TEXT CHECK (skip_reason IS NULL OR length(skip_reason) BETWEEN 1 AND 64),
  failure_reason TEXT CHECK (failure_reason IS NULL OR length(failure_reason) BETWEEN 1 AND 64),
  referred_managed_credential_ref TEXT,
  referrer_managed_credential_ref TEXT,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  credited_at TEXT,
  attempt_ip_hash TEXT CHECK (
    attempt_ip_hash IS NULL OR length(attempt_ip_hash) = 64
  ),
  CHECK (
    (referrer_discord_user_ref IS NULL AND referrer_installation_id IS NULL)
    OR (referrer_discord_user_ref IS NOT NULL AND referrer_installation_id IS NOT NULL)
  ),
  CHECK (
    referrer_discord_user_ref IS NOT NULL
    OR (
      referred_bonus_status = 'skipped'
      AND referrer_bonus_status = 'skipped'
      AND skip_reason IS NOT NULL
    )
  )
) STRICT;

INSERT INTO referral_rewards_referral_id_checks_v2 (
  id,
  referral_id,
  referrer_discord_user_ref,
  referrer_installation_id,
  referred_discord_user_ref,
  referred_installation_id,
  referred_hardware_hash,
  referred_hardware_hash_salt_version,
  referred_bonus_status,
  referrer_bonus_status,
  skip_reason,
  failure_reason,
  referred_managed_credential_ref,
  referrer_managed_credential_ref,
  created_at,
  updated_at,
  credited_at,
  attempt_ip_hash
)
SELECT
  id,
  referral_id,
  referrer_discord_user_ref,
  referrer_installation_id,
  referred_discord_user_ref,
  referred_installation_id,
  referred_hardware_hash,
  referred_hardware_hash_salt_version,
  referred_bonus_status,
  referrer_bonus_status,
  skip_reason,
  failure_reason,
  referred_managed_credential_ref,
  referrer_managed_credential_ref,
  created_at,
  updated_at,
  credited_at,
  attempt_ip_hash
FROM referral_rewards;

DROP TABLE referral_rewards;
ALTER TABLE referral_rewards_referral_id_checks_v2 RENAME TO referral_rewards;

WITH referral_rewards_sequence_repair(seq) AS (
  SELECT max(
    COALESCE((SELECT MAX(seq) FROM referral_rewards_sequence_0007), 0),
    COALESCE((SELECT MAX(id) FROM referral_rewards), 0)
  )
)
UPDATE sqlite_sequence
   SET seq = (SELECT seq FROM referral_rewards_sequence_repair)
 WHERE name = 'referral_rewards'
   AND seq < (SELECT seq FROM referral_rewards_sequence_repair);

WITH referral_rewards_sequence_repair(seq) AS (
  SELECT max(
    COALESCE((SELECT MAX(seq) FROM referral_rewards_sequence_0007), 0),
    COALESCE((SELECT MAX(id) FROM referral_rewards), 0)
  )
)
INSERT INTO sqlite_sequence (name, seq)
SELECT 'referral_rewards', seq
  FROM referral_rewards_sequence_repair
 WHERE seq > 0
   AND NOT EXISTS (
     SELECT 1
       FROM sqlite_sequence
      WHERE name = 'referral_rewards'
   );

DROP TABLE referral_rewards_sequence_0007;

CREATE INDEX idx_referral_rewards_referral_id
  ON referral_rewards(referral_id);
CREATE INDEX idx_referral_rewards_referrer_cap
  ON referral_rewards(referrer_discord_user_ref, referred_bonus_status)
  WHERE referrer_discord_user_ref IS NOT NULL
    AND referred_bonus_status IN ('reserved', 'credited');
CREATE UNIQUE INDEX idx_referral_rewards_counted_referred_discord_user
  ON referral_rewards(referred_discord_user_ref)
  WHERE referred_bonus_status IN ('reserved', 'credited');
CREATE UNIQUE INDEX idx_referral_rewards_counted_referred_installation
  ON referral_rewards(referred_installation_id)
  WHERE referred_bonus_status IN ('reserved', 'credited');
CREATE INDEX idx_referral_rewards_attempt_installation_time
  ON referral_rewards(referred_installation_id, created_at);
CREATE INDEX idx_referral_rewards_attempt_ip_hash_time
  ON referral_rewards(attempt_ip_hash, created_at)
  WHERE attempt_ip_hash IS NOT NULL;
CREATE INDEX idx_referral_rewards_referral_velocity
  ON referral_rewards(referral_id, created_at);
CREATE INDEX idx_referral_rewards_referrer_velocity
  ON referral_rewards(referrer_discord_user_ref, created_at)
  WHERE referrer_discord_user_ref IS NOT NULL;

PRAGMA foreign_key_check;
