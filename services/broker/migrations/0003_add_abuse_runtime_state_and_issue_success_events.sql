CREATE TABLE broker_config_v2 (
  key TEXT PRIMARY KEY CHECK (key IN ('fingerprint_salt', 'abuse_controls', 'abuse_runtime_state')),
  value TEXT NOT NULL CHECK (json_valid(value)),
  updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
) STRICT;

INSERT INTO broker_config_v2 (key, value, updated_at)
SELECT key, value, updated_at
  FROM broker_config;

DROP TABLE broker_config;

ALTER TABLE broker_config_v2 RENAME TO broker_config;

UPDATE broker_config
   SET value = json_insert(
         value,
         '$.immediateAlerts', json('{"warn1":10,"warn2":25,"warn3":50,"critical":70}'),
         '$.asnFastPath', json('{"enabled":true,"minIssueSuccess1h":20,"minTopAsnSharePct":70}'),
         '$.asnClassifications', json('[]'),
         '$.retention', json('{"requestEventsDays":30,"issueSuccessDays":30,"runtimeAuditDays":90}'),
         '$.dailyReport', json('{"enabled":true,"hourUtc":13,"minuteUtc":0,"includeZeroActivity":false}')
       ),
        updated_at = CURRENT_TIMESTAMP
 WHERE key = 'abuse_controls';

INSERT INTO broker_config (key, value)
VALUES (
  'abuse_runtime_state',
  '{"brake":{"active":false,"reason":null,"changedAt":null,"changedBy":null},"alertLatches":{"warn1":false,"warn2":false,"warn3":false,"critical":false},"dailyReport":{"lastDeliveredAt":null,"lastDeliveredDateUtc":null}}'
);

CREATE TABLE broker_issue_success_events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  installation_id TEXT NOT NULL REFERENCES installations(installation_id) ON DELETE CASCADE,
  managed_credential_ref TEXT,
  ip_hash TEXT,
  ip_prefix_hash TEXT,
  asn INTEGER CHECK (asn IS NULL OR asn > 0),
  country TEXT,
  http_protocol TEXT,
  tls_version TEXT,
  tls_cipher TEXT,
  risk_label TEXT,
  observed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
) STRICT;

CREATE INDEX idx_broker_issue_success_events_installation_time
  ON broker_issue_success_events(installation_id, observed_at);
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

CREATE TABLE broker_abuse_runtime_audit (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  event_kind TEXT NOT NULL,
  reason TEXT,
  payload_json TEXT NOT NULL CHECK (json_valid(payload_json)),
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
) STRICT;

CREATE INDEX idx_broker_abuse_runtime_audit_kind_time
  ON broker_abuse_runtime_audit(event_kind, created_at);
CREATE INDEX idx_broker_abuse_runtime_audit_time
  ON broker_abuse_runtime_audit(created_at);
