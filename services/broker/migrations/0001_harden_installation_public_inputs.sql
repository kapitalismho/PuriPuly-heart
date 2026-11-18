PRAGMA defer_foreign_keys = on;

CREATE TABLE installations_hardened (
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
  CHECK (
    length(
      trim(
        installation_id,
        char(9) || char(10) || char(11) || char(12) || char(13) || char(32) || char(160)
        || char(5760) || char(8192) || char(8193) || char(8194) || char(8195) || char(8196)
        || char(8197) || char(8198) || char(8199) || char(8200) || char(8201) || char(8202)
        || char(8232) || char(8233) || char(8239) || char(8287) || char(12288) || char(65279)
      )
    ) > 0
  ),
  CHECK (
    instr(installation_id, char(0)) = 0
    AND instr(installation_id, char(1)) = 0
    AND instr(installation_id, char(2)) = 0
    AND instr(installation_id, char(3)) = 0
    AND instr(installation_id, char(4)) = 0
    AND instr(installation_id, char(5)) = 0
    AND instr(installation_id, char(6)) = 0
    AND instr(installation_id, char(7)) = 0
    AND instr(installation_id, char(8)) = 0
    AND instr(installation_id, char(9)) = 0
    AND instr(installation_id, char(10)) = 0
    AND instr(installation_id, char(11)) = 0
    AND instr(installation_id, char(12)) = 0
    AND instr(installation_id, char(13)) = 0
    AND instr(installation_id, char(14)) = 0
    AND instr(installation_id, char(15)) = 0
    AND instr(installation_id, char(16)) = 0
    AND instr(installation_id, char(17)) = 0
    AND instr(installation_id, char(18)) = 0
    AND instr(installation_id, char(19)) = 0
    AND instr(installation_id, char(20)) = 0
    AND instr(installation_id, char(21)) = 0
    AND instr(installation_id, char(22)) = 0
    AND instr(installation_id, char(23)) = 0
    AND instr(installation_id, char(24)) = 0
    AND instr(installation_id, char(25)) = 0
    AND instr(installation_id, char(26)) = 0
    AND instr(installation_id, char(27)) = 0
    AND instr(installation_id, char(28)) = 0
    AND instr(installation_id, char(29)) = 0
    AND instr(installation_id, char(30)) = 0
    AND instr(installation_id, char(31)) = 0
    AND instr(installation_id, char(127)) = 0
    AND instr(installation_id, char(128)) = 0
    AND instr(installation_id, char(129)) = 0
    AND instr(installation_id, char(130)) = 0
    AND instr(installation_id, char(131)) = 0
    AND instr(installation_id, char(132)) = 0
    AND instr(installation_id, char(133)) = 0
    AND instr(installation_id, char(134)) = 0
    AND instr(installation_id, char(135)) = 0
    AND instr(installation_id, char(136)) = 0
    AND instr(installation_id, char(137)) = 0
    AND instr(installation_id, char(138)) = 0
    AND instr(installation_id, char(139)) = 0
    AND instr(installation_id, char(140)) = 0
    AND instr(installation_id, char(141)) = 0
    AND instr(installation_id, char(142)) = 0
    AND instr(installation_id, char(143)) = 0
    AND instr(installation_id, char(144)) = 0
    AND instr(installation_id, char(145)) = 0
    AND instr(installation_id, char(146)) = 0
    AND instr(installation_id, char(147)) = 0
    AND instr(installation_id, char(148)) = 0
    AND instr(installation_id, char(149)) = 0
    AND instr(installation_id, char(150)) = 0
    AND instr(installation_id, char(151)) = 0
    AND instr(installation_id, char(152)) = 0
    AND instr(installation_id, char(153)) = 0
    AND instr(installation_id, char(154)) = 0
    AND instr(installation_id, char(155)) = 0
    AND instr(installation_id, char(156)) = 0
    AND instr(installation_id, char(157)) = 0
    AND instr(installation_id, char(158)) = 0
    AND instr(installation_id, char(159)) = 0
    AND instr(installation_id, char(8232)) = 0
    AND instr(installation_id, char(8233)) = 0
  ),
  CHECK (length(app_version) BETWEEN 1 AND 64),
  CHECK (
    length(
      trim(
        app_version,
        char(9) || char(10) || char(11) || char(12) || char(13) || char(32) || char(160)
        || char(5760) || char(8192) || char(8193) || char(8194) || char(8195) || char(8196)
        || char(8197) || char(8198) || char(8199) || char(8200) || char(8201) || char(8202)
        || char(8232) || char(8233) || char(8239) || char(8287) || char(12288) || char(65279)
      )
    ) > 0
  ),
  CHECK (
    instr(app_version, char(0)) = 0
    AND instr(app_version, char(1)) = 0
    AND instr(app_version, char(2)) = 0
    AND instr(app_version, char(3)) = 0
    AND instr(app_version, char(4)) = 0
    AND instr(app_version, char(5)) = 0
    AND instr(app_version, char(6)) = 0
    AND instr(app_version, char(7)) = 0
    AND instr(app_version, char(8)) = 0
    AND instr(app_version, char(9)) = 0
    AND instr(app_version, char(10)) = 0
    AND instr(app_version, char(11)) = 0
    AND instr(app_version, char(12)) = 0
    AND instr(app_version, char(13)) = 0
    AND instr(app_version, char(14)) = 0
    AND instr(app_version, char(15)) = 0
    AND instr(app_version, char(16)) = 0
    AND instr(app_version, char(17)) = 0
    AND instr(app_version, char(18)) = 0
    AND instr(app_version, char(19)) = 0
    AND instr(app_version, char(20)) = 0
    AND instr(app_version, char(21)) = 0
    AND instr(app_version, char(22)) = 0
    AND instr(app_version, char(23)) = 0
    AND instr(app_version, char(24)) = 0
    AND instr(app_version, char(25)) = 0
    AND instr(app_version, char(26)) = 0
    AND instr(app_version, char(27)) = 0
    AND instr(app_version, char(28)) = 0
    AND instr(app_version, char(29)) = 0
    AND instr(app_version, char(30)) = 0
    AND instr(app_version, char(31)) = 0
    AND instr(app_version, char(127)) = 0
    AND instr(app_version, char(128)) = 0
    AND instr(app_version, char(129)) = 0
    AND instr(app_version, char(130)) = 0
    AND instr(app_version, char(131)) = 0
    AND instr(app_version, char(132)) = 0
    AND instr(app_version, char(133)) = 0
    AND instr(app_version, char(134)) = 0
    AND instr(app_version, char(135)) = 0
    AND instr(app_version, char(136)) = 0
    AND instr(app_version, char(137)) = 0
    AND instr(app_version, char(138)) = 0
    AND instr(app_version, char(139)) = 0
    AND instr(app_version, char(140)) = 0
    AND instr(app_version, char(141)) = 0
    AND instr(app_version, char(142)) = 0
    AND instr(app_version, char(143)) = 0
    AND instr(app_version, char(144)) = 0
    AND instr(app_version, char(145)) = 0
    AND instr(app_version, char(146)) = 0
    AND instr(app_version, char(147)) = 0
    AND instr(app_version, char(148)) = 0
    AND instr(app_version, char(149)) = 0
    AND instr(app_version, char(150)) = 0
    AND instr(app_version, char(151)) = 0
    AND instr(app_version, char(152)) = 0
    AND instr(app_version, char(153)) = 0
    AND instr(app_version, char(154)) = 0
    AND instr(app_version, char(155)) = 0
    AND instr(app_version, char(156)) = 0
    AND instr(app_version, char(157)) = 0
    AND instr(app_version, char(158)) = 0
    AND instr(app_version, char(159)) = 0
    AND instr(app_version, char(8232)) = 0
    AND instr(app_version, char(8233)) = 0
  ),
  CHECK (hardware_hash IS NULL OR length(hardware_hash) BETWEEN 1 AND 128),
  CHECK (
    hardware_hash IS NULL
    OR length(
      trim(
        hardware_hash,
        char(9) || char(10) || char(11) || char(12) || char(13) || char(32) || char(160)
        || char(5760) || char(8192) || char(8193) || char(8194) || char(8195) || char(8196)
        || char(8197) || char(8198) || char(8199) || char(8200) || char(8201) || char(8202)
        || char(8232) || char(8233) || char(8239) || char(8287) || char(12288) || char(65279)
      )
    ) > 0
  ),
  CHECK (
    hardware_hash IS NULL
    OR (
      instr(hardware_hash, char(0)) = 0
      AND instr(hardware_hash, char(1)) = 0
      AND instr(hardware_hash, char(2)) = 0
      AND instr(hardware_hash, char(3)) = 0
      AND instr(hardware_hash, char(4)) = 0
      AND instr(hardware_hash, char(5)) = 0
      AND instr(hardware_hash, char(6)) = 0
      AND instr(hardware_hash, char(7)) = 0
      AND instr(hardware_hash, char(8)) = 0
      AND instr(hardware_hash, char(9)) = 0
      AND instr(hardware_hash, char(10)) = 0
      AND instr(hardware_hash, char(11)) = 0
      AND instr(hardware_hash, char(12)) = 0
      AND instr(hardware_hash, char(13)) = 0
      AND instr(hardware_hash, char(14)) = 0
      AND instr(hardware_hash, char(15)) = 0
      AND instr(hardware_hash, char(16)) = 0
      AND instr(hardware_hash, char(17)) = 0
      AND instr(hardware_hash, char(18)) = 0
      AND instr(hardware_hash, char(19)) = 0
      AND instr(hardware_hash, char(20)) = 0
      AND instr(hardware_hash, char(21)) = 0
      AND instr(hardware_hash, char(22)) = 0
      AND instr(hardware_hash, char(23)) = 0
      AND instr(hardware_hash, char(24)) = 0
      AND instr(hardware_hash, char(25)) = 0
      AND instr(hardware_hash, char(26)) = 0
      AND instr(hardware_hash, char(27)) = 0
      AND instr(hardware_hash, char(28)) = 0
      AND instr(hardware_hash, char(29)) = 0
      AND instr(hardware_hash, char(30)) = 0
      AND instr(hardware_hash, char(31)) = 0
      AND instr(hardware_hash, char(127)) = 0
      AND instr(hardware_hash, char(128)) = 0
      AND instr(hardware_hash, char(129)) = 0
      AND instr(hardware_hash, char(130)) = 0
      AND instr(hardware_hash, char(131)) = 0
      AND instr(hardware_hash, char(132)) = 0
      AND instr(hardware_hash, char(133)) = 0
      AND instr(hardware_hash, char(134)) = 0
      AND instr(hardware_hash, char(135)) = 0
      AND instr(hardware_hash, char(136)) = 0
      AND instr(hardware_hash, char(137)) = 0
      AND instr(hardware_hash, char(138)) = 0
      AND instr(hardware_hash, char(139)) = 0
      AND instr(hardware_hash, char(140)) = 0
      AND instr(hardware_hash, char(141)) = 0
      AND instr(hardware_hash, char(142)) = 0
      AND instr(hardware_hash, char(143)) = 0
      AND instr(hardware_hash, char(144)) = 0
      AND instr(hardware_hash, char(145)) = 0
      AND instr(hardware_hash, char(146)) = 0
      AND instr(hardware_hash, char(147)) = 0
      AND instr(hardware_hash, char(148)) = 0
      AND instr(hardware_hash, char(149)) = 0
      AND instr(hardware_hash, char(150)) = 0
      AND instr(hardware_hash, char(151)) = 0
      AND instr(hardware_hash, char(152)) = 0
      AND instr(hardware_hash, char(153)) = 0
      AND instr(hardware_hash, char(154)) = 0
      AND instr(hardware_hash, char(155)) = 0
      AND instr(hardware_hash, char(156)) = 0
      AND instr(hardware_hash, char(157)) = 0
      AND instr(hardware_hash, char(158)) = 0
      AND instr(hardware_hash, char(159)) = 0
      AND instr(hardware_hash, char(8232)) = 0
      AND instr(hardware_hash, char(8233)) = 0
    )
  ),
  CHECK (
    (challenge IS NULL AND challenge_expires_at IS NULL AND challenge_salt_version IS NULL)
    OR (challenge IS NOT NULL AND challenge_expires_at IS NOT NULL AND challenge_salt_version IS NOT NULL)
  )
) STRICT;

CREATE TABLE openrouter_entitlements_hardened (
  installation_id TEXT PRIMARY KEY REFERENCES installations_hardened(installation_id) ON DELETE CASCADE,
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

-- Follow-up schema hardening assumes existing rows already satisfy the tightened
-- public-input contract. Invalid historical rows are unsupported and cause the
-- migration to fail rather than being rewritten.
INSERT INTO installations_hardened (
  installation_id,
  device_public_key,
  hardware_hash,
  hardware_hash_salt_version,
  app_version,
  challenge,
  challenge_expires_at,
  challenge_salt_version,
  created_at,
  last_seen_at
)
SELECT installation_id,
       device_public_key,
       hardware_hash,
       hardware_hash_salt_version,
       app_version,
       challenge,
       challenge_expires_at,
       challenge_salt_version,
       created_at,
       last_seen_at
  FROM installations;

INSERT INTO openrouter_entitlements_hardened (
  installation_id,
  status,
  budget_usd,
  managed_credential_ref,
  issued_at,
  expires_at,
  release_session_ref,
  release_token_hash,
  release_token_expires_at
)
SELECT installation_id,
       status,
       budget_usd,
       managed_credential_ref,
       issued_at,
       expires_at,
       release_session_ref,
       release_token_hash,
       release_token_expires_at
  FROM openrouter_entitlements;

DROP TABLE openrouter_entitlements;
DROP TABLE installations;

ALTER TABLE installations_hardened RENAME TO installations;
ALTER TABLE openrouter_entitlements_hardened RENAME TO openrouter_entitlements;

CREATE INDEX idx_installations_hardware_hash
  ON installations(hardware_hash);
CREATE INDEX idx_installations_hardware_hash_salt_version
  ON installations(hardware_hash_salt_version);
CREATE INDEX idx_installations_challenge_expires_at
  ON installations(challenge_expires_at);
CREATE INDEX idx_installations_last_seen_at
  ON installations(last_seen_at);

CREATE INDEX idx_openrouter_entitlements_status
  ON openrouter_entitlements(status);
CREATE INDEX idx_openrouter_entitlements_expires_at
  ON openrouter_entitlements(expires_at);
CREATE UNIQUE INDEX idx_openrouter_entitlements_release_token_hash
  ON openrouter_entitlements(release_token_hash)
  WHERE release_token_hash IS NOT NULL;

PRAGMA foreign_key_check;
