CREATE TABLE discord_oauth_sessions (
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
  CHECK (length(installation_id) BETWEEN 1 AND 128),
  CHECK (length(device_public_key) > 0),
  CHECK (length(redirect_uri) > 0)
) STRICT;

CREATE INDEX idx_discord_oauth_sessions_installation_status
  ON discord_oauth_sessions(installation_id, status, created_at);
CREATE INDEX idx_discord_oauth_sessions_expires_at
  ON discord_oauth_sessions(expires_at);

CREATE TABLE discord_identities (
  discord_user_ref TEXT PRIMARY KEY,
  entitlement_installation_id TEXT REFERENCES installations(installation_id) ON DELETE SET NULL,
  status TEXT NOT NULL CHECK(status IN ('issuing', 'active', 'failed', 'cleanup_required')),
  ref_secret_version INTEGER NOT NULL DEFAULT 1,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
) STRICT;

ALTER TABLE openrouter_entitlements
  ADD COLUMN discord_user_ref TEXT REFERENCES discord_identities(discord_user_ref);
ALTER TABLE openrouter_entitlements
  ADD COLUMN discord_issue_status TEXT CHECK(discord_issue_status IS NULL OR discord_issue_status IN ('issuing', 'active', 'failed', 'cleanup_required'));
ALTER TABLE openrouter_entitlements
  ADD COLUMN discord_issue_reserved_at TEXT;
ALTER TABLE openrouter_entitlements
  ADD COLUMN discord_issue_delivered_at TEXT;

CREATE UNIQUE INDEX idx_openrouter_entitlements_discord_user_ref
  ON openrouter_entitlements(discord_user_ref)
  WHERE discord_user_ref IS NOT NULL;
CREATE INDEX idx_openrouter_entitlements_discord_issue_reserved_at
  ON openrouter_entitlements(discord_issue_reserved_at)
  WHERE discord_issue_status = 'issuing';

UPDATE broker_config
   SET value = json_set(
         json_insert(
           value,
           '$.discordAuthStartIp', json('{"endpoint":"POST /v1/auth/discord/start","scope":"ip","maxRequests":20,"windowMinutes":15}'),
           '$.discordAuthStartInstallation', json('{"endpoint":"POST /v1/auth/discord/start","scope":"installation_id","maxRequests":5,"windowMinutes":15}'),
           '$.discordOpenrouterIssueIp', json('{"endpoint":"POST /v1/providers/openrouter/discord/issue","scope":"ip","maxRequests":10,"windowMinutes":15}'),
           '$.discordOpenrouterIssueInstallation', json('{"endpoint":"POST /v1/providers/openrouter/discord/issue","scope":"installation_id","maxRequests":3,"windowMinutes":15}'),
           '$.pendingDiscordOAuthSessions', json('{"maxPerInstallation":2,"maxPerIp":20,"windowMinutes":15}')
         ),
         '$.newActiveEntitlementsPerDay.endpoint', 'POST /v1/providers/openrouter/discord/issue',
         '$.newActiveEntitlementsPerDay.maxCount', CASE
           WHEN json_type(value, '$.newActiveEntitlementsPerDay.maxCount') = 'integer'
                AND json_extract(value, '$.newActiveEntitlementsPerDay.maxCount') > 0
             THEN json_extract(value, '$.newActiveEntitlementsPerDay.maxCount')
           ELSE 500
         END
       ),
       updated_at = CURRENT_TIMESTAMP
 WHERE key = 'abuse_controls';
