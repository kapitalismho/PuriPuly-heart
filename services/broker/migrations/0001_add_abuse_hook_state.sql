CREATE TABLE broker_request_events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  endpoint TEXT NOT NULL,
  ip TEXT,
  installation_id TEXT,
  observed_at TEXT NOT NULL,
  CHECK (ip IS NOT NULL OR installation_id IS NOT NULL)
) STRICT;

CREATE INDEX idx_broker_request_events_endpoint_ip_time
  ON broker_request_events(endpoint, ip, observed_at);
CREATE INDEX idx_broker_request_events_endpoint_installation_time
  ON broker_request_events(endpoint, installation_id, observed_at);
CREATE INDEX idx_broker_request_events_ip_time
  ON broker_request_events(ip, observed_at);
CREATE INDEX idx_broker_request_events_installation_time
  ON broker_request_events(installation_id, observed_at);

CREATE TABLE broker_velocity_cap_hooks (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  subject_type TEXT NOT NULL CHECK (subject_type IN ('ip', 'installation_id')),
  subject_value TEXT NOT NULL,
  max_requests INTEGER NOT NULL CHECK (max_requests > 0),
  window_minutes INTEGER NOT NULL CHECK (window_minutes > 0),
  outcome_code TEXT NOT NULL CHECK (outcome_code IN ('rate_limited', 'issuance_suspended', 'trial_unavailable', 'trial_not_eligible')),
  outcome_class TEXT NOT NULL CHECK (outcome_class IN ('retryable', 'terminal', 'security_fail')),
  outcome_subcode TEXT,
  reason TEXT,
  active INTEGER NOT NULL DEFAULT 1 CHECK (active IN (0, 1)),
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  expires_at TEXT
) STRICT;

CREATE INDEX idx_broker_velocity_cap_hooks_lookup
  ON broker_velocity_cap_hooks(subject_type, subject_value, active, expires_at);

CREATE TABLE broker_abuse_subject_hooks (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  hook_kind TEXT NOT NULL CHECK (hook_kind IN ('denylist', 'reputation', 'revocation')),
  subject_type TEXT NOT NULL CHECK (subject_type IN ('ip', 'installation_id', 'hardware_hash')),
  subject_value TEXT NOT NULL,
  outcome_code TEXT NOT NULL CHECK (outcome_code IN ('issuance_suspended', 'trial_unavailable', 'trial_not_eligible')),
  outcome_class TEXT NOT NULL CHECK (outcome_class IN ('retryable', 'terminal', 'security_fail')),
  outcome_subcode TEXT,
  reason TEXT,
  active INTEGER NOT NULL DEFAULT 1 CHECK (active IN (0, 1)),
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  expires_at TEXT
) STRICT;

CREATE INDEX idx_broker_abuse_subject_hooks_lookup
  ON broker_abuse_subject_hooks(subject_type, subject_value, hook_kind, active, expires_at);
