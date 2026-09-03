DROP TABLE IF EXISTS network_identity_unattributed_gate;
CREATE TABLE network_identity_unattributed_gate (
  ok INTEGER NOT NULL CHECK (ok = 1)
) STRICT;
INSERT INTO network_identity_unattributed_gate (ok)
SELECT 0 WHERE EXISTS (
  SELECT 1 FROM pragma_table_info('broker_request_events') WHERE name = 'ip'
);
DROP TABLE network_identity_unattributed_gate;

CREATE TABLE broker_request_events_unattributed (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  endpoint TEXT NOT NULL,
  ip_digest TEXT CHECK (ip_digest IS NULL OR length(ip_digest) = 64),
  ip_key_version INTEGER CHECK (ip_key_version IS NULL OR ip_key_version > 0),
  ip_epoch TEXT CHECK (ip_epoch IS NULL OR length(ip_epoch) = 10),
  installation_id TEXT,
  observed_at TEXT NOT NULL,
  CHECK (
    (ip_digest IS NULL AND ip_key_version IS NULL AND ip_epoch IS NULL)
    OR (ip_digest IS NOT NULL AND ip_key_version IS NOT NULL AND ip_epoch IS NOT NULL)
  )
) STRICT;

INSERT INTO broker_request_events_unattributed (id, endpoint, ip_digest, ip_key_version, ip_epoch, installation_id, observed_at)
SELECT id, endpoint, ip_digest, ip_key_version, ip_epoch, installation_id, observed_at
  FROM broker_request_events;

DROP TABLE broker_request_events;
ALTER TABLE broker_request_events_unattributed RENAME TO broker_request_events;

CREATE INDEX idx_broker_request_events_endpoint_digest_time
  ON broker_request_events(endpoint, ip_digest, observed_at);
CREATE INDEX idx_broker_request_events_endpoint_installation_time
  ON broker_request_events(endpoint, installation_id, observed_at);
CREATE INDEX idx_broker_request_events_digest_time
  ON broker_request_events(ip_digest, observed_at)
  WHERE ip_digest IS NOT NULL;
CREATE INDEX idx_broker_request_events_installation_time
  ON broker_request_events(installation_id, observed_at);
CREATE INDEX idx_broker_request_events_endpoint_time
  ON broker_request_events(endpoint, observed_at);
