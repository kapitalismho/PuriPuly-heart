ALTER TABLE discord_oauth_sessions
  ADD COLUMN referral_id TEXT CHECK (
    referral_id IS NULL
    OR (
      length(referral_id) = 6
      AND referral_id NOT GLOB '*[^23456789ABCDEFGHJKMNPQRSTUVWXYZ]*'
    )
  );

CREATE INDEX idx_discord_oauth_sessions_referral_id
  ON discord_oauth_sessions(referral_id)
  WHERE referral_id IS NOT NULL;

CREATE TABLE referral_codes (
  referral_id TEXT PRIMARY KEY CHECK (
    length(referral_id) = 6
    AND referral_id NOT GLOB '*[^23456789ABCDEFGHJKMNPQRSTUVWXYZ]*'
  ),
  owner_discord_user_ref TEXT NOT NULL CHECK (length(owner_discord_user_ref) > 0),
  owner_installation_id TEXT,
  status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'disabled')),
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
) STRICT;

CREATE UNIQUE INDEX idx_referral_codes_owner_discord_user_ref
  ON referral_codes(owner_discord_user_ref);
CREATE INDEX idx_referral_codes_owner_installation_id
  ON referral_codes(owner_installation_id)
  WHERE owner_installation_id IS NOT NULL;
CREATE INDEX idx_referral_codes_status
  ON referral_codes(status, referral_id);

CREATE TABLE referral_rewards (
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
