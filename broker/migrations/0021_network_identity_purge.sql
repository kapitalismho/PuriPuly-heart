PRAGMA defer_foreign_keys = on;

DROP TABLE IF EXISTS network_identity_sequence_backup_0021;
DROP TABLE IF EXISTS network_identity_finalize_gate;
CREATE TABLE network_identity_sequence_backup_0021 (
  name TEXT PRIMARY KEY,
  seq INTEGER NOT NULL
) STRICT;

INSERT OR REPLACE INTO network_identity_sequence_backup_0021 (name, seq)
SELECT name, seq FROM sqlite_sequence
 WHERE name IN ('broker_request_events', 'broker_issue_success_events', 'referral_rewards');

CREATE TABLE network_identity_finalize_gate (
  ok INTEGER NOT NULL CHECK (ok = 1)
) STRICT;

INSERT INTO network_identity_finalize_gate (ok)
SELECT 0
 WHERE (SELECT COALESCE(json_extract(value, '$.phase'), 'dual_write') FROM broker_config WHERE key = 'network_identity_migration') <> 'keyed_only'
    OR EXISTS (
      SELECT 1 FROM broker_request_events
       WHERE ip IS NOT NULL
         AND ip_digest IS NULL
        AND (ip_epoch IS NULL OR ip_epoch != '0000-00-00')
        AND observed_at >= datetime('now', '-' || (
          SELECT max(1440,
            COALESCE((SELECT json_extract(value, '$.trialChallenge.windowMinutes') FROM broker_config WHERE key = 'abuse_controls'), 0),
            COALESCE((SELECT json_extract(value, '$.trialChallengeVerify.windowMinutes') FROM broker_config WHERE key = 'abuse_controls'), 0),
            COALESCE((SELECT json_extract(value, '$.openrouterIssue.windowMinutes') FROM broker_config WHERE key = 'abuse_controls'), 0),
            COALESCE((SELECT json_extract(value, '$.trialStatus.windowMinutes') FROM broker_config WHERE key = 'abuse_controls'), 0),
            COALESCE((SELECT json_extract(value, '$.qqAuthAssertIp.windowMinutes') FROM broker_config WHERE key = 'abuse_controls'), 0),
            COALESCE((SELECT json_extract(value, '$.qqAuthStatusIp.windowMinutes') FROM broker_config WHERE key = 'abuse_controls'), 0),
            COALESCE((SELECT json_extract(value, '$.pendingDiscordOAuthSessions.windowMinutes') FROM broker_config WHERE key = 'abuse_controls'), 0),
            COALESCE((SELECT json_extract(value, '$.referralAttempts.validShaped.windowMinutes') FROM broker_config WHERE key = 'abuse_controls'), 0),
            COALESCE((SELECT json_extract(value, '$.referralAttempts.unknown.windowMinutes') FROM broker_config WHERE key = 'abuse_controls'), 0),
            COALESCE((SELECT json_extract(value, '$.referralAttempts.perReferralIdVelocity.windowMinutes') FROM broker_config WHERE key = 'abuse_controls'), 0),
            COALESCE((SELECT json_extract(value, '$.referralAttempts.perReferrerRewardVelocity.windowMinutes') FROM broker_config WHERE key = 'abuse_controls'), 0),
            COALESCE((SELECT json_extract(value, '$.managedOperationStatusIp.windowMinutes') FROM broker_config WHERE key = 'abuse_controls'), 0),
            COALESCE((SELECT json_extract(value, '$.managedOperationStatusInstallation.windowMinutes') FROM broker_config WHERE key = 'abuse_controls'), 0),
            COALESCE((SELECT json_extract(value, '$.managedOperationResumeIp.windowMinutes') FROM broker_config WHERE key = 'abuse_controls'), 0),
            COALESCE((SELECT json_extract(value, '$.managedOperationResumeInstallation.windowMinutes') FROM broker_config WHERE key = 'abuse_controls'), 0),
            COALESCE((SELECT json_extract(value, '$.managedKeyDeliveryAckIp.windowMinutes') FROM broker_config WHERE key = 'abuse_controls'), 0),
            COALESCE((SELECT MAX(window_minutes) FROM broker_velocity_cap_hooks WHERE active = 1), 0))
        ) || ' minutes')
    )
    OR EXISTS (
      SELECT 1 FROM referral_rewards
       WHERE attempt_ip_hash IS NOT NULL
         AND attempt_ip_digest IS NULL
        AND created_at >= datetime('now', '-' || (
          SELECT max(1440,
            COALESCE((SELECT json_extract(value, '$.trialChallenge.windowMinutes') FROM broker_config WHERE key = 'abuse_controls'), 0),
            COALESCE((SELECT json_extract(value, '$.trialChallengeVerify.windowMinutes') FROM broker_config WHERE key = 'abuse_controls'), 0),
            COALESCE((SELECT json_extract(value, '$.openrouterIssue.windowMinutes') FROM broker_config WHERE key = 'abuse_controls'), 0),
            COALESCE((SELECT json_extract(value, '$.trialStatus.windowMinutes') FROM broker_config WHERE key = 'abuse_controls'), 0),
            COALESCE((SELECT json_extract(value, '$.qqAuthAssertIp.windowMinutes') FROM broker_config WHERE key = 'abuse_controls'), 0),
            COALESCE((SELECT json_extract(value, '$.qqAuthStatusIp.windowMinutes') FROM broker_config WHERE key = 'abuse_controls'), 0),
            COALESCE((SELECT json_extract(value, '$.pendingDiscordOAuthSessions.windowMinutes') FROM broker_config WHERE key = 'abuse_controls'), 0),
            COALESCE((SELECT json_extract(value, '$.referralAttempts.validShaped.windowMinutes') FROM broker_config WHERE key = 'abuse_controls'), 0),
            COALESCE((SELECT json_extract(value, '$.referralAttempts.unknown.windowMinutes') FROM broker_config WHERE key = 'abuse_controls'), 0),
            COALESCE((SELECT json_extract(value, '$.referralAttempts.perReferralIdVelocity.windowMinutes') FROM broker_config WHERE key = 'abuse_controls'), 0),
            COALESCE((SELECT json_extract(value, '$.referralAttempts.perReferrerRewardVelocity.windowMinutes') FROM broker_config WHERE key = 'abuse_controls'), 0),
            COALESCE((SELECT json_extract(value, '$.managedOperationStatusIp.windowMinutes') FROM broker_config WHERE key = 'abuse_controls'), 0),
            COALESCE((SELECT json_extract(value, '$.managedOperationStatusInstallation.windowMinutes') FROM broker_config WHERE key = 'abuse_controls'), 0),
            COALESCE((SELECT json_extract(value, '$.managedOperationResumeIp.windowMinutes') FROM broker_config WHERE key = 'abuse_controls'), 0),
            COALESCE((SELECT json_extract(value, '$.managedOperationResumeInstallation.windowMinutes') FROM broker_config WHERE key = 'abuse_controls'), 0),
            COALESCE((SELECT json_extract(value, '$.managedKeyDeliveryAckIp.windowMinutes') FROM broker_config WHERE key = 'abuse_controls'), 0),
            COALESCE((SELECT MAX(window_minutes) FROM broker_velocity_cap_hooks WHERE active = 1), 0))
        ) || ' minutes')
    )
    OR EXISTS (
      SELECT 1 FROM broker_velocity_cap_hooks
       WHERE subject_type = 'ip' AND active = 1
         AND (length(subject_value) != 64 OR subject_value GLOB '*[^0-9a-f]*')
    )
    OR EXISTS (
      SELECT 1 FROM broker_abuse_subject_hooks
       WHERE subject_type = 'ip' AND active = 1
         AND (length(subject_value) != 64 OR subject_value GLOB '*[^0-9a-f]*')
    );

DROP TABLE network_identity_finalize_gate;

CREATE TABLE broker_request_events_keyed (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  endpoint TEXT NOT NULL,
  ip_digest TEXT CHECK (ip_digest IS NULL OR length(ip_digest) = 64),
  ip_key_version INTEGER CHECK (ip_key_version IS NULL OR ip_key_version > 0),
  ip_epoch TEXT CHECK (ip_epoch IS NULL OR length(ip_epoch) = 10),
  installation_id TEXT,
  observed_at TEXT NOT NULL,
  CHECK (ip_digest IS NOT NULL OR installation_id IS NOT NULL),
  CHECK (
    (ip_digest IS NULL AND ip_key_version IS NULL AND ip_epoch IS NULL)
    OR (ip_digest IS NOT NULL AND ip_key_version IS NOT NULL AND ip_epoch IS NOT NULL)
  )
) STRICT;

INSERT INTO broker_request_events_keyed (id, endpoint, ip_digest, ip_key_version, ip_epoch, installation_id, observed_at)
SELECT id, endpoint, ip_digest, ip_key_version, ip_epoch, installation_id, observed_at
  FROM broker_request_events;

DROP TABLE broker_request_events;
ALTER TABLE broker_request_events_keyed RENAME TO broker_request_events;

CREATE INDEX idx_broker_request_events_endpoint_digest_time
  ON broker_request_events(endpoint, ip_digest, observed_at);
CREATE INDEX idx_broker_request_events_endpoint_installation_time
  ON broker_request_events(endpoint, installation_id, observed_at);
CREATE INDEX idx_broker_request_events_digest_time
  ON broker_request_events(ip_digest, observed_at)
  WHERE ip_digest IS NOT NULL;
CREATE INDEX idx_broker_request_events_installation_time
  ON broker_request_events(installation_id, observed_at);

CREATE TABLE broker_issue_success_events_keyed (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  issue_source TEXT NOT NULL CHECK(issue_source IN ('discord', 'qq')),
  installation_id TEXT REFERENCES installations(installation_id) ON DELETE CASCADE,
  subject_ref TEXT NOT NULL CHECK (length(subject_ref) > 0),
  managed_credential_ref TEXT,
  ip_digest TEXT CHECK (ip_digest IS NULL OR length(ip_digest) = 64),
  ip_prefix_digest TEXT CHECK (ip_prefix_digest IS NULL OR length(ip_prefix_digest) = 64),
  ip_key_version INTEGER CHECK (ip_key_version IS NULL OR ip_key_version > 0),
  ip_epoch TEXT CHECK (ip_epoch IS NULL OR length(ip_epoch) = 10),
  asn INTEGER CHECK (asn IS NULL OR asn > 0),
  country TEXT,
  http_protocol TEXT,
  tls_version TEXT,
  tls_cipher TEXT,
  risk_label TEXT,
  observed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  CHECK (
    (
      issue_source = 'discord'
      AND installation_id IS NOT NULL
      AND subject_ref = installation_id
    )
    OR (
      issue_source = 'qq'
      AND installation_id IS NULL
      AND length(subject_ref) > length('ph-qq-subject-v1_')
      AND subject_ref GLOB 'ph-qq-subject-v1_*'
    )
  ),
  CHECK (
    (ip_digest IS NULL AND ip_prefix_digest IS NULL AND ip_key_version IS NULL AND ip_epoch IS NULL)
    OR (ip_digest IS NOT NULL AND ip_prefix_digest IS NOT NULL AND ip_key_version IS NOT NULL AND ip_epoch IS NOT NULL)
  )
) STRICT;

INSERT INTO broker_issue_success_events_keyed (
  id, issue_source, installation_id, subject_ref, managed_credential_ref,
  ip_digest, ip_prefix_digest, ip_key_version, ip_epoch,
  asn, country, http_protocol, tls_version, tls_cipher, risk_label, observed_at
)
SELECT
  id, issue_source, installation_id, subject_ref, managed_credential_ref,
  ip_digest, ip_prefix_digest, ip_key_version, ip_epoch,
  asn, country, http_protocol, tls_version, tls_cipher, risk_label, observed_at
FROM broker_issue_success_events;

DROP TABLE broker_issue_success_events;
ALTER TABLE broker_issue_success_events_keyed RENAME TO broker_issue_success_events;

CREATE INDEX idx_broker_issue_success_events_installation_time
  ON broker_issue_success_events(installation_id, observed_at);
CREATE INDEX idx_broker_issue_success_events_source_subject_time
  ON broker_issue_success_events(issue_source, subject_ref, observed_at);
CREATE INDEX idx_broker_issue_success_events_credential_time
  ON broker_issue_success_events(managed_credential_ref, observed_at);
CREATE INDEX idx_broker_issue_success_events_digest_time
  ON broker_issue_success_events(ip_digest, observed_at)
  WHERE ip_digest IS NOT NULL;
CREATE INDEX idx_broker_issue_success_events_asn_time
  ON broker_issue_success_events(asn, observed_at);
CREATE INDEX idx_broker_issue_success_events_time
  ON broker_issue_success_events(observed_at);

CREATE TABLE referral_rewards_keyed (
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
  attempt_ip_digest TEXT CHECK (attempt_ip_digest IS NULL OR length(attempt_ip_digest) = 64),
  attempt_ip_key_version INTEGER CHECK (attempt_ip_key_version IS NULL OR attempt_ip_key_version > 0),
  attempt_ip_epoch TEXT CHECK (attempt_ip_epoch IS NULL OR length(attempt_ip_epoch) = 10),
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
  CHECK (
    (attempt_ip_digest IS NULL AND attempt_ip_key_version IS NULL AND attempt_ip_epoch IS NULL)
    OR (attempt_ip_digest IS NOT NULL AND attempt_ip_key_version IS NOT NULL AND attempt_ip_epoch IS NOT NULL)
  )
) STRICT;

INSERT INTO referral_rewards_keyed (
  id, referral_id, referrer_source, referrer_subject_ref, referrer_discord_user_ref,
  referrer_installation_id, referred_source, referred_subject_ref, referred_discord_user_ref,
  referred_installation_id, referred_hardware_hash, referred_hardware_hash_salt_version,
  referred_bonus_status, referrer_bonus_status, skip_reason, failure_reason,
  referred_managed_credential_ref, referrer_managed_credential_ref,
  created_at, updated_at, credited_at,
  attempt_ip_digest, attempt_ip_key_version, attempt_ip_epoch, operation_id
)
SELECT
  id, referral_id, referrer_source, referrer_subject_ref, referrer_discord_user_ref,
  referrer_installation_id, referred_source, referred_subject_ref, referred_discord_user_ref,
  referred_installation_id, referred_hardware_hash, referred_hardware_hash_salt_version,
  referred_bonus_status, referrer_bonus_status, skip_reason, failure_reason,
  referred_managed_credential_ref, referrer_managed_credential_ref,
  created_at, updated_at, credited_at,
  attempt_ip_digest, attempt_ip_key_version, attempt_ip_epoch, operation_id
FROM referral_rewards;

DROP TABLE referral_rewards;
ALTER TABLE referral_rewards_keyed RENAME TO referral_rewards;

CREATE INDEX idx_referral_rewards_referral_id_created_at
  ON referral_rewards(referral_id, created_at);
CREATE INDEX idx_referral_rewards_referrer_subject_status
  ON referral_rewards(referrer_source, referrer_subject_ref, referred_bonus_status);
CREATE INDEX idx_referral_rewards_referred_subject_created_at
  ON referral_rewards(referred_source, referred_subject_ref, created_at);
CREATE INDEX idx_referral_rewards_referred_installation_created_at
  ON referral_rewards(referred_installation_id, created_at)
  WHERE referred_installation_id IS NOT NULL;
CREATE INDEX idx_referral_rewards_attempt_digest_created_at
  ON referral_rewards(attempt_ip_digest, created_at)
  WHERE attempt_ip_digest IS NOT NULL;
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
    COALESCE((SELECT seq FROM network_identity_sequence_backup_0021 WHERE name = 'broker_request_events'), 0),
    COALESCE((SELECT MAX(id) FROM broker_request_events), 0)
  )
)
UPDATE sqlite_sequence
   SET seq = (SELECT seq FROM sequence_repair)
 WHERE name = 'broker_request_events'
   AND seq < (SELECT seq FROM sequence_repair);

INSERT INTO sqlite_sequence (name, seq)
SELECT 'broker_request_events', (SELECT max(
    COALESCE((SELECT seq FROM network_identity_sequence_backup_0021 WHERE name = 'broker_request_events'), 0),
    COALESCE((SELECT MAX(id) FROM broker_request_events), 0)
  ))
 WHERE NOT EXISTS (SELECT 1 FROM sqlite_sequence WHERE name = 'broker_request_events');

WITH sequence_repair(seq) AS (
  SELECT max(
    COALESCE((SELECT seq FROM network_identity_sequence_backup_0021 WHERE name = 'broker_issue_success_events'), 0),
    COALESCE((SELECT MAX(id) FROM broker_issue_success_events), 0)
  )
)
UPDATE sqlite_sequence
   SET seq = (SELECT seq FROM sequence_repair)
 WHERE name = 'broker_issue_success_events'
   AND seq < (SELECT seq FROM sequence_repair);

INSERT INTO sqlite_sequence (name, seq)
SELECT 'broker_issue_success_events', (SELECT max(
    COALESCE((SELECT seq FROM network_identity_sequence_backup_0021 WHERE name = 'broker_issue_success_events'), 0),
    COALESCE((SELECT MAX(id) FROM broker_issue_success_events), 0)
  ))
 WHERE NOT EXISTS (SELECT 1 FROM sqlite_sequence WHERE name = 'broker_issue_success_events');

WITH sequence_repair(seq) AS (
  SELECT max(
    COALESCE((SELECT seq FROM network_identity_sequence_backup_0021 WHERE name = 'referral_rewards'), 0),
    COALESCE((SELECT MAX(id) FROM referral_rewards), 0)
  )
)
UPDATE sqlite_sequence
   SET seq = (SELECT seq FROM sequence_repair)
 WHERE name = 'referral_rewards'
   AND seq < (SELECT seq FROM sequence_repair);

INSERT INTO sqlite_sequence (name, seq)
SELECT 'referral_rewards', (SELECT max(
    COALESCE((SELECT seq FROM network_identity_sequence_backup_0021 WHERE name = 'referral_rewards'), 0),
    COALESCE((SELECT MAX(id) FROM referral_rewards), 0)
  ))
 WHERE NOT EXISTS (SELECT 1 FROM sqlite_sequence WHERE name = 'referral_rewards');

DROP TABLE network_identity_sequence_backup_0021;

UPDATE broker_config
   SET value = '{"phase":"keyed_only","purged_at":"finalized"}',
       updated_at = CURRENT_TIMESTAMP
 WHERE key = 'network_identity_migration';

PRAGMA foreign_key_check;
