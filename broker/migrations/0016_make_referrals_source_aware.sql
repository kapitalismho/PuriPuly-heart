PRAGMA defer_foreign_keys = on;

CREATE TABLE referral_codes_source_aware (
  referral_id TEXT PRIMARY KEY CHECK (
    length(referral_id) = 6
    AND referral_id NOT GLOB '*[^23456789ABCDEFGHJKMNPQRSTUVWXYZ]*'
  ),
  owner_source TEXT CHECK (
    owner_source IS NULL OR owner_source IN ('discord', 'qq')
  ),
  owner_subject_ref TEXT CHECK (
    owner_subject_ref IS NULL OR length(owner_subject_ref) > 0
  ),
  owner_discord_user_ref TEXT CHECK (
    owner_discord_user_ref IS NULL OR length(owner_discord_user_ref) > 0
  ),
  owner_installation_id TEXT CHECK (
    owner_installation_id IS NULL OR length(owner_installation_id) > 0
  ),
  status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'disabled')),
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  disabled_reason TEXT CHECK (
    disabled_reason IS NULL OR length(disabled_reason) BETWEEN 1 AND 64
  ),
  disabled_by TEXT CHECK (
    disabled_by IS NULL OR length(disabled_by) BETWEEN 1 AND 64
  ),
  disabled_at TEXT,
  CHECK (
    (owner_source IS NULL AND owner_subject_ref IS NULL AND owner_discord_user_ref IS NOT NULL)
    OR (owner_source IS NOT NULL AND owner_subject_ref IS NOT NULL)
  ),
  CHECK (
    owner_source IS NULL
    OR (owner_source = 'discord' AND owner_subject_ref GLOB 'ph-discord-user-v*_*')
    OR (owner_source = 'qq' AND owner_subject_ref GLOB 'ph-qq-subject-v1_*')
  ),
  CHECK (
    owner_discord_user_ref IS NULL
    OR owner_source IS NULL
    OR (
      owner_source = 'discord'
      AND owner_subject_ref = owner_discord_user_ref
    )
  ),
  CHECK (owner_source <> 'qq' OR owner_discord_user_ref IS NULL)
) STRICT;

INSERT INTO referral_codes_source_aware (
  referral_id,
  owner_source,
  owner_subject_ref,
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
  'discord',
  owner_discord_user_ref,
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
ALTER TABLE referral_codes_source_aware RENAME TO referral_codes;

CREATE UNIQUE INDEX idx_referral_codes_owner_subject
  ON referral_codes(owner_source, owner_subject_ref)
  WHERE owner_source IS NOT NULL AND owner_subject_ref IS NOT NULL;
CREATE UNIQUE INDEX idx_referral_codes_owner_discord_user_ref
  ON referral_codes(owner_discord_user_ref)
  WHERE owner_discord_user_ref IS NOT NULL;
CREATE INDEX idx_referral_codes_owner_installation_id
  ON referral_codes(owner_installation_id)
  WHERE owner_installation_id IS NOT NULL;
CREATE INDEX idx_referral_codes_status
  ON referral_codes(status, referral_id);

CREATE TRIGGER sync_referral_codes_compatibility_insert
AFTER INSERT ON referral_codes
FOR EACH ROW
WHEN
  NEW.owner_source IS NULL
  OR NEW.owner_subject_ref IS NULL
  OR (NEW.owner_source = 'discord' AND NEW.owner_discord_user_ref IS NULL)
BEGIN
  UPDATE referral_codes
     SET owner_source = COALESCE(NEW.owner_source, 'discord'),
         owner_subject_ref = COALESCE(
           NEW.owner_subject_ref,
           NEW.owner_discord_user_ref
         ),
         owner_discord_user_ref = CASE
           WHEN COALESCE(NEW.owner_source, 'discord') = 'discord'
             THEN COALESCE(
               NEW.owner_discord_user_ref,
               NEW.owner_subject_ref
             )
           ELSE NULL
         END
   WHERE referral_id = NEW.referral_id;
END;

CREATE TABLE referral_rewards_sequence_0016 (
  seq INTEGER NOT NULL
) STRICT;

INSERT INTO referral_rewards_sequence_0016 (seq)
SELECT COALESCE(MAX(seq), 0)
  FROM sqlite_sequence
 WHERE name = 'referral_rewards';

CREATE TABLE referral_rewards_source_aware (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  referral_id TEXT NOT NULL CHECK (
    length(referral_id) = 6
    AND referral_id NOT GLOB '*[^23456789ABCDEFGHJKMNPQRSTUVWXYZ]*'
  ),
  referrer_source TEXT CHECK (
    referrer_source IS NULL OR referrer_source IN ('discord', 'qq')
  ),
  referrer_subject_ref TEXT CHECK (
    referrer_subject_ref IS NULL OR length(referrer_subject_ref) > 0
  ),
  referrer_discord_user_ref TEXT CHECK (
    referrer_discord_user_ref IS NULL OR length(referrer_discord_user_ref) > 0
  ),
  referrer_installation_id TEXT CHECK (
    referrer_installation_id IS NULL OR length(referrer_installation_id) > 0
  ),
  referred_source TEXT CHECK (
    referred_source IS NULL OR referred_source IN ('discord', 'qq')
  ),
  referred_subject_ref TEXT CHECK (
    referred_subject_ref IS NULL OR length(referred_subject_ref) > 0
  ),
  referred_discord_user_ref TEXT CHECK (
    referred_discord_user_ref IS NULL OR length(referred_discord_user_ref) > 0
  ),
  referred_installation_id TEXT CHECK (
    referred_installation_id IS NULL OR length(referred_installation_id) > 0
  ),
  referred_hardware_hash TEXT CHECK (
    referred_hardware_hash IS NULL OR length(referred_hardware_hash) BETWEEN 1 AND 128
  ),
  referred_hardware_hash_salt_version INTEGER CHECK (
    referred_hardware_hash_salt_version IS NULL OR referred_hardware_hash_salt_version > 0
  ),
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

INSERT INTO referral_rewards_source_aware (
  id,
  referral_id,
  referrer_source,
  referrer_subject_ref,
  referrer_discord_user_ref,
  referrer_installation_id,
  referred_source,
  referred_subject_ref,
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
  CASE WHEN referrer_discord_user_ref IS NULL THEN NULL ELSE 'discord' END,
  referrer_discord_user_ref,
  referrer_discord_user_ref,
  referrer_installation_id,
  'discord',
  referred_discord_user_ref,
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
ALTER TABLE referral_rewards_source_aware RENAME TO referral_rewards;

WITH referral_rewards_sequence_repair(seq) AS (
  SELECT max(
    COALESCE((SELECT MAX(seq) FROM referral_rewards_sequence_0016), 0),
    COALESCE((SELECT MAX(id) FROM referral_rewards), 0)
  )
)
UPDATE sqlite_sequence
   SET seq = (SELECT seq FROM referral_rewards_sequence_repair)
 WHERE name = 'referral_rewards'
   AND seq < (SELECT seq FROM referral_rewards_sequence_repair);

WITH referral_rewards_sequence_repair(seq) AS (
  SELECT max(
    COALESCE((SELECT MAX(seq) FROM referral_rewards_sequence_0016), 0),
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

DROP TABLE referral_rewards_sequence_0016;

CREATE INDEX idx_referral_rewards_referral_id
  ON referral_rewards(referral_id);
CREATE INDEX idx_referral_rewards_referrer_cap
  ON referral_rewards(referrer_source, referrer_subject_ref, referred_bonus_status)
  WHERE referrer_source IS NOT NULL
    AND referred_bonus_status IN ('reserved', 'credited');
CREATE INDEX idx_referral_rewards_referrer_cap_legacy
  ON referral_rewards(referrer_discord_user_ref, referred_bonus_status)
  WHERE referrer_discord_user_ref IS NOT NULL
    AND referred_bonus_status IN ('reserved', 'credited');
CREATE UNIQUE INDEX idx_referral_rewards_counted_referred_subject
  ON referral_rewards(referred_source, referred_subject_ref)
  WHERE referred_source IS NOT NULL
    AND referred_bonus_status IN ('reserved', 'credited');
CREATE UNIQUE INDEX idx_referral_rewards_counted_referred_discord_user
  ON referral_rewards(referred_discord_user_ref)
  WHERE referred_discord_user_ref IS NOT NULL
    AND referred_bonus_status IN ('reserved', 'credited');
CREATE UNIQUE INDEX idx_referral_rewards_counted_referred_installation
  ON referral_rewards(referred_installation_id)
  WHERE referred_installation_id IS NOT NULL
    AND referred_bonus_status IN ('reserved', 'credited');
CREATE INDEX idx_referral_rewards_attempt_subject_time
  ON referral_rewards(referred_source, referred_subject_ref, created_at);
CREATE INDEX idx_referral_rewards_attempt_installation_time
  ON referral_rewards(referred_installation_id, created_at)
  WHERE referred_installation_id IS NOT NULL;
CREATE INDEX idx_referral_rewards_attempt_ip_hash_time
  ON referral_rewards(attempt_ip_hash, created_at)
  WHERE attempt_ip_hash IS NOT NULL;
CREATE INDEX idx_referral_rewards_referral_velocity
  ON referral_rewards(referral_id, created_at);
CREATE INDEX idx_referral_rewards_referrer_velocity
  ON referral_rewards(referrer_source, referrer_subject_ref, created_at)
  WHERE referrer_source IS NOT NULL;
CREATE INDEX idx_referral_rewards_referrer_velocity_legacy
  ON referral_rewards(referrer_discord_user_ref, created_at)
  WHERE referrer_discord_user_ref IS NOT NULL;

CREATE TRIGGER sync_referral_rewards_compatibility_insert
AFTER INSERT ON referral_rewards
FOR EACH ROW
WHEN
  NEW.referred_source IS NULL
  OR NEW.referred_subject_ref IS NULL
  OR (NEW.referred_source = 'discord' AND NEW.referred_discord_user_ref IS NULL)
  OR (
    NEW.referrer_discord_user_ref IS NOT NULL
    AND (NEW.referrer_source IS NULL OR NEW.referrer_subject_ref IS NULL)
  )
  OR (
    NEW.referrer_source = 'discord'
    AND NEW.referrer_subject_ref IS NOT NULL
    AND NEW.referrer_discord_user_ref IS NULL
  )
BEGIN
  UPDATE referral_rewards
     SET referrer_source = CASE
           WHEN NEW.referrer_source IS NOT NULL THEN NEW.referrer_source
           WHEN NEW.referrer_discord_user_ref IS NOT NULL THEN 'discord'
           ELSE NULL
         END,
         referrer_subject_ref = COALESCE(
           NEW.referrer_subject_ref,
           NEW.referrer_discord_user_ref
         ),
         referrer_discord_user_ref = CASE
           WHEN COALESCE(
             NEW.referrer_source,
             CASE WHEN NEW.referrer_discord_user_ref IS NOT NULL THEN 'discord' END
           ) = 'discord'
             THEN COALESCE(
               NEW.referrer_discord_user_ref,
               NEW.referrer_subject_ref
             )
           ELSE NULL
         END,
         referred_source = COALESCE(NEW.referred_source, 'discord'),
         referred_subject_ref = COALESCE(
           NEW.referred_subject_ref,
           NEW.referred_discord_user_ref
         ),
         referred_discord_user_ref = CASE
           WHEN COALESCE(NEW.referred_source, 'discord') = 'discord'
             THEN COALESCE(
               NEW.referred_discord_user_ref,
               NEW.referred_subject_ref
             )
           ELSE NULL
         END
   WHERE id = NEW.id;
END;

CREATE TABLE broker_config_qq_pass (
  key TEXT PRIMARY KEY CHECK (
    key IN (
      'fingerprint_salt',
      'abuse_controls',
      'abuse_runtime_state',
      'qq_talk_together_pass'
    )
  ),
  value TEXT NOT NULL CHECK (json_valid(value)),
  updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
) STRICT;

INSERT INTO broker_config_qq_pass (key, value, updated_at)
SELECT key, value, updated_at
  FROM broker_config;

DROP TABLE broker_config;
ALTER TABLE broker_config_qq_pass RENAME TO broker_config;

INSERT INTO broker_config (key, value, updated_at)
VALUES (
  'qq_talk_together_pass',
  '{"enabled":false,"rewards_enabled":false,"daily_warning_count":30,"daily_max_count":50}',
  CURRENT_TIMESTAMP
);

UPDATE broker_config
   SET value = json_insert(
     value,
     '$.qqAuthStatusIp',
     json('{"endpoint":"POST /v1/auth/qq/status","scope":"ip","maxRequests":30,"windowMinutes":15}')
   ),
       updated_at = CURRENT_TIMESTAMP
 WHERE key = 'abuse_controls';

PRAGMA foreign_key_check;
