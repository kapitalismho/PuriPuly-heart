PRAGMA defer_foreign_keys = on;

CREATE TABLE openrouter_entitlements_pre_0012_backup AS
SELECT * FROM openrouter_entitlements;

CREATE TABLE qq_managed_entitlements_pre_0012_backup AS
SELECT * FROM qq_managed_entitlements;

CREATE TABLE openrouter_entitlements_delivery_ack_v2 (
  installation_id TEXT PRIMARY KEY REFERENCES installations(installation_id) ON DELETE CASCADE,
  status TEXT NOT NULL CHECK(status IN ('pending_release', 'active', 'expired', 'revoked')),
  budget_usd REAL NOT NULL CHECK (budget_usd >= 0),
  managed_credential_ref TEXT UNIQUE,
  issued_at TEXT,
  expires_at TEXT,
  release_session_ref TEXT,
  release_token_hash TEXT,
  release_token_expires_at TEXT,
  verified_hardware_hash TEXT,
  verified_hardware_hash_salt_version INTEGER CHECK (
    (verified_hardware_hash IS NULL AND verified_hardware_hash_salt_version IS NULL)
    OR (
      verified_hardware_hash IS NOT NULL
      AND verified_hardware_hash_salt_version IS NOT NULL
    )
  ),
  discord_user_ref TEXT REFERENCES discord_identities(discord_user_ref),
  discord_issue_status TEXT CHECK(discord_issue_status IS NULL OR discord_issue_status IN ('issuing', 'delivery_pending', 'active', 'failed', 'cleanup_required')),
  discord_issue_reserved_at TEXT,
  discord_issue_delivered_at TEXT,
  CHECK (
    (release_session_ref IS NULL AND release_token_hash IS NULL AND release_token_expires_at IS NULL)
    OR (
      release_session_ref IS NOT NULL
      AND release_token_hash IS NOT NULL
      AND release_token_expires_at IS NOT NULL
    )
  )
) STRICT;

INSERT INTO openrouter_entitlements_delivery_ack_v2 (
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
SELECT installation_id,
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
  FROM openrouter_entitlements;

DROP TABLE openrouter_entitlements;

ALTER TABLE openrouter_entitlements_delivery_ack_v2 RENAME TO openrouter_entitlements;

CREATE INDEX idx_openrouter_entitlements_status
  ON openrouter_entitlements(status);
CREATE INDEX idx_openrouter_entitlements_expires_at
  ON openrouter_entitlements(expires_at);
CREATE UNIQUE INDEX idx_openrouter_entitlements_release_token_hash
  ON openrouter_entitlements(release_token_hash)
  WHERE release_token_hash IS NOT NULL;
CREATE UNIQUE INDEX idx_openrouter_entitlements_discord_user_ref
  ON openrouter_entitlements(discord_user_ref)
  WHERE discord_user_ref IS NOT NULL;
CREATE INDEX idx_openrouter_entitlements_discord_issue_reserved_at
  ON openrouter_entitlements(discord_issue_reserved_at)
  WHERE discord_issue_status = 'issuing';

CREATE TABLE qq_managed_entitlements_delivery_ack_v2 (
  qq_subject_ref TEXT PRIMARY KEY
    CHECK (length(qq_subject_ref) > length('ph-qq-subject-v1_'))
    CHECK (qq_subject_ref GLOB 'ph-qq-subject-v1_*'),
  status TEXT NOT NULL CHECK(status IN ('issuing', 'delivery_pending', 'active', 'cleanup_required', 'revoked')),
  issue_ref TEXT NOT NULL CHECK (length(issue_ref) > 0),
  managed_credential_ref TEXT,
  budget_usd REAL NOT NULL CHECK (budget_usd >= 0),
  reserved_at TEXT NOT NULL CHECK (length(reserved_at) > 0),
  issued_at TEXT,
  expires_at TEXT,
  delivered_at TEXT,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  CHECK (
    status <> 'active'
    OR (
      managed_credential_ref IS NOT NULL
      AND issued_at IS NOT NULL
      AND expires_at IS NOT NULL
      AND delivered_at IS NOT NULL
    )
  ),
  CHECK (
    status <> 'delivery_pending'
    OR (
      managed_credential_ref IS NOT NULL
      AND issued_at IS NOT NULL
      AND expires_at IS NOT NULL
      AND delivered_at IS NULL
    )
  ),
  CHECK (
    status <> 'cleanup_required'
    OR managed_credential_ref IS NOT NULL
  ),
  CHECK (
    status <> 'issuing'
    OR (
      issued_at IS NULL
      AND expires_at IS NULL
      AND delivered_at IS NULL
    )
  )
) STRICT;

INSERT INTO qq_managed_entitlements_delivery_ack_v2 (
  qq_subject_ref,
  status,
  issue_ref,
  managed_credential_ref,
  budget_usd,
  reserved_at,
  issued_at,
  expires_at,
  delivered_at,
  created_at,
  updated_at
)
SELECT qq_subject_ref,
       status,
       issue_ref,
       managed_credential_ref,
       budget_usd,
       reserved_at,
       issued_at,
       expires_at,
       delivered_at,
       created_at,
       updated_at
  FROM qq_managed_entitlements;

DROP TABLE qq_managed_entitlements;

ALTER TABLE qq_managed_entitlements_delivery_ack_v2 RENAME TO qq_managed_entitlements;

CREATE UNIQUE INDEX idx_qq_managed_entitlements_issue_ref
  ON qq_managed_entitlements(issue_ref);
CREATE UNIQUE INDEX idx_qq_managed_entitlements_managed_credential_ref
  ON qq_managed_entitlements(managed_credential_ref)
  WHERE managed_credential_ref IS NOT NULL;
CREATE INDEX idx_qq_managed_entitlements_status_updated_at
  ON qq_managed_entitlements(status, updated_at);
CREATE INDEX idx_qq_managed_entitlements_expires_at
  ON qq_managed_entitlements(expires_at);

CREATE TABLE managed_key_deliveries (
  delivery_id TEXT PRIMARY KEY,
  issue_source TEXT NOT NULL CHECK (issue_source IN ('discord', 'qq')),
  subject_ref TEXT,
  installation_id TEXT,
  managed_credential_ref TEXT NOT NULL,
  ack_token_hash TEXT NOT NULL,
  status TEXT NOT NULL CHECK (status IN ('pending', 'acknowledged', 'expired', 'cleanup_required')),
  created_at TEXT NOT NULL,
  expires_at TEXT NOT NULL,
  acknowledged_at TEXT,
  failed_at TEXT,
  failure_reason TEXT
) STRICT;

CREATE INDEX idx_managed_key_deliveries_status_expires_at
  ON managed_key_deliveries(status, expires_at);
CREATE INDEX idx_managed_key_deliveries_managed_credential_ref
  ON managed_key_deliveries(managed_credential_ref);
CREATE INDEX idx_managed_key_deliveries_issue_source_created_at
  ON managed_key_deliveries(issue_source, created_at);

PRAGMA foreign_key_check;

DROP TABLE openrouter_entitlements_pre_0012_backup;
DROP TABLE qq_managed_entitlements_pre_0012_backup;
