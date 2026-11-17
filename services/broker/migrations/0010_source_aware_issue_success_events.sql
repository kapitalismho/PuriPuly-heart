PRAGMA defer_foreign_keys = on;

CREATE TABLE broker_issue_success_events_pre_0010_backup AS
SELECT * FROM broker_issue_success_events;

CREATE TABLE broker_issue_success_events_sequence_0010_backup (
  seq INTEGER NOT NULL
) STRICT;

INSERT INTO broker_issue_success_events_sequence_0010_backup (seq)
SELECT COALESCE(
         (
           SELECT seq
             FROM sqlite_sequence
            WHERE name = 'broker_issue_success_events'
         ),
         0
       );

CREATE TABLE broker_issue_success_events_source_v2 (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  issue_source TEXT NOT NULL CHECK(issue_source IN ('discord', 'qq')),
  installation_id TEXT REFERENCES installations(installation_id) ON DELETE CASCADE,
  subject_ref TEXT NOT NULL CHECK (length(subject_ref) > 0),
  managed_credential_ref TEXT,
  ip_hash TEXT,
  ip_prefix_hash TEXT,
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
  )
) STRICT;

INSERT INTO broker_issue_success_events_source_v2 (
  id,
  issue_source,
  installation_id,
  subject_ref,
  managed_credential_ref,
  ip_hash,
  ip_prefix_hash,
  asn,
  country,
  http_protocol,
  tls_version,
  tls_cipher,
  risk_label,
  observed_at
)
SELECT id,
       'discord',
       installation_id,
       installation_id,
       managed_credential_ref,
       ip_hash,
       ip_prefix_hash,
       asn,
       country,
       http_protocol,
       tls_version,
       tls_cipher,
       risk_label,
       observed_at
  FROM broker_issue_success_events;

DROP TABLE broker_issue_success_events;

ALTER TABLE broker_issue_success_events_source_v2 RENAME TO broker_issue_success_events;

DELETE FROM sqlite_sequence
 WHERE name = 'broker_issue_success_events_source_v2';

UPDATE sqlite_sequence
   SET seq = (
         SELECT MAX(seq_value)
           FROM (
             SELECT seq AS seq_value
               FROM broker_issue_success_events_sequence_0010_backup
             UNION ALL
             SELECT COALESCE(MAX(id), 0) AS seq_value
               FROM broker_issue_success_events
           )
       )
 WHERE name = 'broker_issue_success_events';

INSERT INTO sqlite_sequence (name, seq)
SELECT 'broker_issue_success_events',
       (
         SELECT MAX(seq_value)
           FROM (
             SELECT seq AS seq_value
               FROM broker_issue_success_events_sequence_0010_backup
             UNION ALL
             SELECT COALESCE(MAX(id), 0) AS seq_value
               FROM broker_issue_success_events
           )
       )
 WHERE NOT EXISTS (
       SELECT 1
         FROM sqlite_sequence
        WHERE name = 'broker_issue_success_events'
       );

CREATE INDEX idx_broker_issue_success_events_installation_time
  ON broker_issue_success_events(installation_id, observed_at);
CREATE INDEX idx_broker_issue_success_events_source_subject_time
  ON broker_issue_success_events(issue_source, subject_ref, observed_at);
CREATE INDEX idx_broker_issue_success_events_credential_time
  ON broker_issue_success_events(managed_credential_ref, observed_at);
CREATE INDEX idx_broker_issue_success_events_ip_hash_time
  ON broker_issue_success_events(ip_hash, observed_at);
CREATE INDEX idx_broker_issue_success_events_ip_prefix_hash_time
  ON broker_issue_success_events(ip_prefix_hash, observed_at);
CREATE INDEX idx_broker_issue_success_events_asn_time
  ON broker_issue_success_events(asn, observed_at);
CREATE INDEX idx_broker_issue_success_events_time
  ON broker_issue_success_events(observed_at);

PRAGMA foreign_key_check;

DROP TABLE broker_issue_success_events_sequence_0010_backup;
DROP TABLE broker_issue_success_events_pre_0010_backup;
