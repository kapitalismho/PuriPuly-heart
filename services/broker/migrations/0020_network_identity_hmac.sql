PRAGMA defer_foreign_keys = on;

CREATE TABLE broker_request_events_sequence_backup_0020 (
  seq INTEGER NOT NULL
) STRICT;

INSERT INTO broker_request_events_sequence_backup_0020 (seq)
SELECT COALESCE((SELECT seq FROM sqlite_sequence WHERE name = 'broker_request_events'), 0);

CREATE TABLE broker_request_events_dual_write (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  endpoint TEXT NOT NULL,
  ip TEXT,
  ip_digest TEXT CHECK (ip_digest IS NULL OR length(ip_digest) = 64),
  ip_key_version INTEGER CHECK (ip_key_version IS NULL OR ip_key_version > 0),
  ip_epoch TEXT CHECK (ip_epoch IS NULL OR length(ip_epoch) = 10),
  installation_id TEXT,
  observed_at TEXT NOT NULL,
  CHECK (ip IS NOT NULL OR ip_digest IS NOT NULL OR installation_id IS NOT NULL),
  CHECK (
    ip_digest IS NULL
    OR (ip_key_version IS NOT NULL AND ip_epoch IS NOT NULL)
  )
) STRICT;

INSERT INTO broker_request_events_dual_write (id, endpoint, ip, ip_digest, ip_key_version, ip_epoch, installation_id, observed_at)
SELECT id, endpoint, ip, NULL, NULL, NULL, installation_id, observed_at
  FROM broker_request_events;

DROP TABLE broker_request_events;
ALTER TABLE broker_request_events_dual_write RENAME TO broker_request_events;

WITH sequence_repair(seq) AS (
  SELECT max(
    COALESCE((SELECT seq FROM broker_request_events_sequence_backup_0020), 0),
    COALESCE((SELECT MAX(id) FROM broker_request_events), 0)
  )
)
UPDATE sqlite_sequence
   SET seq = (SELECT seq FROM sequence_repair)
 WHERE name = 'broker_request_events'
   AND seq < (SELECT seq FROM sequence_repair);

INSERT INTO sqlite_sequence (name, seq)
SELECT 'broker_request_events', (SELECT max(
    COALESCE((SELECT seq FROM broker_request_events_sequence_backup_0020), 0),
    COALESCE((SELECT MAX(id) FROM broker_request_events), 0)
  ))
 WHERE NOT EXISTS (SELECT 1 FROM sqlite_sequence WHERE name = 'broker_request_events');

DROP TABLE broker_request_events_sequence_backup_0020;

CREATE INDEX idx_broker_request_events_endpoint_ip_time
  ON broker_request_events(endpoint, ip, observed_at);
CREATE INDEX idx_broker_request_events_endpoint_installation_time
  ON broker_request_events(endpoint, installation_id, observed_at);
CREATE INDEX idx_broker_request_events_ip_time
  ON broker_request_events(ip, observed_at);
CREATE INDEX idx_broker_request_events_installation_time
  ON broker_request_events(installation_id, observed_at);
CREATE INDEX idx_broker_request_events_endpoint_digest_time
  ON broker_request_events(endpoint, ip_digest, observed_at);
CREATE INDEX idx_broker_request_events_digest_time
  ON broker_request_events(ip_digest, observed_at)
  WHERE ip_digest IS NOT NULL;

ALTER TABLE broker_issue_success_events
  ADD COLUMN ip_digest TEXT CHECK (ip_digest IS NULL OR length(ip_digest) = 64);
ALTER TABLE broker_issue_success_events
  ADD COLUMN ip_prefix_digest TEXT CHECK (ip_prefix_digest IS NULL OR length(ip_prefix_digest) = 64);
ALTER TABLE broker_issue_success_events
  ADD COLUMN ip_key_version INTEGER CHECK (ip_key_version IS NULL OR ip_key_version > 0);
ALTER TABLE broker_issue_success_events
  ADD COLUMN ip_epoch TEXT CHECK (ip_epoch IS NULL OR length(ip_epoch) = 10);

CREATE INDEX idx_broker_issue_success_events_digest_time
  ON broker_issue_success_events(ip_digest, observed_at)
  WHERE ip_digest IS NOT NULL;

ALTER TABLE referral_rewards
  ADD COLUMN attempt_ip_digest TEXT CHECK (attempt_ip_digest IS NULL OR length(attempt_ip_digest) = 64);
ALTER TABLE referral_rewards
  ADD COLUMN attempt_ip_key_version INTEGER CHECK (attempt_ip_key_version IS NULL OR attempt_ip_key_version > 0);
ALTER TABLE referral_rewards
  ADD COLUMN attempt_ip_epoch TEXT CHECK (attempt_ip_epoch IS NULL OR length(attempt_ip_epoch) = 10);

CREATE INDEX idx_referral_rewards_attempt_digest_created_at
  ON referral_rewards(attempt_ip_digest, created_at)
  WHERE attempt_ip_digest IS NOT NULL;

CREATE TABLE broker_config_network_identity (
  key TEXT PRIMARY KEY CHECK (
    key IN (
      'fingerprint_salt',
      'abuse_controls',
      'abuse_runtime_state',
      'qq_talk_together_pass',
      'network_identity_migration'
    )
  ),
  value TEXT NOT NULL CHECK (json_valid(value)),
  updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
) STRICT;

INSERT INTO broker_config_network_identity (key, value, updated_at)
SELECT key, value, updated_at
  FROM broker_config;

DROP TABLE broker_config;
ALTER TABLE broker_config_network_identity RENAME TO broker_config;

INSERT INTO broker_config (key, value, updated_at)
SELECT 'network_identity_migration',
       CASE
         WHEN EXISTS (SELECT 1 FROM broker_request_events WHERE ip IS NOT NULL)
           OR EXISTS (SELECT 1 FROM referral_rewards WHERE attempt_ip_hash IS NOT NULL)
           OR EXISTS (
             SELECT 1 FROM broker_velocity_cap_hooks
              WHERE subject_type = 'ip' AND active = 1
                AND (length(subject_value) != 64 OR subject_value GLOB '*[^0-9a-f]*')
           )
           OR EXISTS (
             SELECT 1 FROM broker_abuse_subject_hooks
              WHERE subject_type = 'ip' AND active = 1
                AND (length(subject_value) != 64 OR subject_value GLOB '*[^0-9a-f]*')
           )
           THEN '{"phase":"dual_write","purge_after":null}'
         ELSE '{"phase":"keyed_only","purge_after":null}'
       END,
       CURRENT_TIMESTAMP
 WHERE NOT EXISTS (SELECT 1 FROM broker_config WHERE key = 'network_identity_migration');

PRAGMA foreign_key_check;
