PRAGMA foreign_keys = ON;

CREATE TABLE broker_config (
  key TEXT PRIMARY KEY CHECK (key IN ('fingerprint_salt', 'abuse_controls')),
  value TEXT NOT NULL CHECK (json_valid(value)),
  updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
) STRICT;

-- Bootstrap config rows live in broker_config so runtime-tunable controls stay out
-- of code constants. Deployment bootstrap must replace the fingerprint salt
-- placeholder before challenge or verify traffic is enabled.
INSERT INTO broker_config (key, value)
VALUES
  (
    'fingerprint_salt',
    '{"current":{"version":1,"salt":"__BOOTSTRAP_REQUIRED__"},"previous":null,"rotated_at":null}'
  ),
  (
    'abuse_controls',
    '{"trialChallenge":{"endpoint":"POST /v1/trial/challenge","scope":"ip","maxRequests":10,"windowMinutes":15},"trialChallengeVerify":{"endpoint":"POST /v1/trial/challenge/verify","scope":"installation_id","maxRequests":5,"windowMinutes":15},"openrouterIssue":{"endpoint":"POST /v1/providers/openrouter/issue","scope":"installation_id","maxRequests":3,"windowMinutes":15},"trialStatus":{"endpoint":"GET /v1/trial/status","scope":"installation_id","maxRequests":30,"windowMinutes":15},"newActiveEntitlementsPerDay":{"endpoint":"POST /v1/providers/openrouter/issue","scope":"global","maxCount":null,"windowDays":1}}'
  );

-- The abuse_controls row fixes the per-endpoint dimensions while keeping the
-- thresholds runtime-configurable through broker_config updates.
CREATE TABLE installations (
  installation_id TEXT PRIMARY KEY,
  device_public_key TEXT NOT NULL UNIQUE,
  hardware_hash TEXT,
  hardware_hash_salt_version INTEGER,
  app_version TEXT NOT NULL,
  challenge TEXT,
  challenge_expires_at TEXT,
  challenge_salt_version INTEGER,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  last_seen_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  CHECK (
    (hardware_hash IS NULL AND hardware_hash_salt_version IS NULL)
    OR (hardware_hash IS NOT NULL AND hardware_hash_salt_version IS NOT NULL)
  ),
  CHECK (length(installation_id) BETWEEN 1 AND 128),
  CHECK (length(app_version) BETWEEN 1 AND 64),
  CHECK (hardware_hash IS NULL OR length(hardware_hash) BETWEEN 1 AND 128),
  CHECK (
    (challenge IS NULL AND challenge_expires_at IS NULL AND challenge_salt_version IS NULL)
    OR (challenge IS NOT NULL AND challenge_expires_at IS NOT NULL AND challenge_salt_version IS NOT NULL)
  )
) STRICT;

CREATE INDEX idx_installations_hardware_hash
  ON installations(hardware_hash);
CREATE INDEX idx_installations_hardware_hash_salt_version
  ON installations(hardware_hash_salt_version);
CREATE INDEX idx_installations_challenge_expires_at
  ON installations(challenge_expires_at);
CREATE INDEX idx_installations_last_seen_at
  ON installations(last_seen_at);

-- One entitlement row is updated in place per installation. Remaining live budget
-- stays upstream in OpenRouter metadata; the broker persists only bounded release
-- state needed for installation lookup, issuance, and expiry.
CREATE TABLE openrouter_entitlements (
  installation_id TEXT PRIMARY KEY REFERENCES installations(installation_id) ON DELETE CASCADE,
  status TEXT NOT NULL CHECK(status IN ('pending_release', 'active', 'expired', 'revoked')),
  budget_usd REAL NOT NULL CHECK (budget_usd >= 0),
  managed_credential_ref TEXT UNIQUE,
  issued_at TEXT,
  expires_at TEXT,
  release_session_ref TEXT,
  release_token_hash TEXT,
  release_token_expires_at TEXT,
  CHECK (
    (release_session_ref IS NULL AND release_token_hash IS NULL AND release_token_expires_at IS NULL)
    OR (
      release_session_ref IS NOT NULL
      AND release_token_hash IS NOT NULL
      AND release_token_expires_at IS NOT NULL
    )
  )
) STRICT;

CREATE INDEX idx_openrouter_entitlements_status
  ON openrouter_entitlements(status);
CREATE INDEX idx_openrouter_entitlements_expires_at
  ON openrouter_entitlements(expires_at);
CREATE UNIQUE INDEX idx_openrouter_entitlements_release_token_hash
  ON openrouter_entitlements(release_token_hash)
  WHERE release_token_hash IS NOT NULL;

-- Retention rules are enforced by deleting from installations only:
--   * pending_release rows inactive for more than 30 days from last_seen_at
--   * expired or revoked rows older than 90 days from max(last_seen_at, expires_at)
-- ON DELETE CASCADE removes the entitlement row with the installation.
-- No entitlement row means lifecycle = none; concrete rows store only
-- pending_release, active, expired, or revoked.
-- Salt rotation keeps one current and one previous fingerprint salt version.
-- New challenges always issue the current version. Duplicate matching only uses
-- hardware_hash rows tagged with the current salt version; in-flight challenges
-- may finish on the previous version until challenge_expires_at, after which
-- stale hashes are refreshed in place on successful verify or cleared on reissue.
