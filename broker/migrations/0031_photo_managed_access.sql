PRAGMA defer_foreign_keys = on;

CREATE TABLE managed_operations_backup_0031 AS
SELECT operation_id, issue_source, subject_ref, installation_id, device_public_key,
       state, attempt_count, current_attempt_index, resume_token_hash, auth_expires_at,
       failure_reason, client_action, created_at, updated_at, last_reconciled_at, cleanup_attempts,
       hardware_hash, hardware_hash_salt_version, app_version
FROM managed_operations;
CREATE TABLE managed_attempts_backup_0031 AS SELECT * FROM managed_operation_attempts;
CREATE TABLE managed_deliveries_backup_0031 AS SELECT * FROM managed_key_deliveries;
CREATE TABLE managed_attempt_sequence_0031 AS
SELECT COALESCE((SELECT seq FROM sqlite_sequence WHERE name = 'managed_operation_attempts'), 0) AS seq;

DROP TABLE managed_key_deliveries;
DROP TABLE managed_operation_attempts;
DROP TABLE managed_operations;

CREATE TABLE managed_operations (
  operation_id TEXT PRIMARY KEY CHECK (length(operation_id) > length('ph-mop-v1_') AND substr(operation_id, 1, length('ph-mop-v1_')) = 'ph-mop-v1_'),
  issue_source TEXT NOT NULL CHECK (issue_source IN ('discord', 'qq', 'photo')),
  subject_ref TEXT NOT NULL CHECK (length(subject_ref) > 0),
  installation_id TEXT CHECK (installation_id IS NULL OR length(installation_id) BETWEEN 1 AND 128),
  device_public_key TEXT CHECK (device_public_key IS NULL OR length(device_public_key) BETWEEN 1 AND 256),
  state TEXT NOT NULL CHECK (state IN ('AUTHENTICATED', 'ISSUE_READY', 'CREATING', 'CREATE_UNKNOWN', 'RECONCILING', 'CLEANUP_REQUIRED', 'CLEAN', 'RETRY_READY', 'DELIVERY_PENDING', 'ACTIVE', 'FAILED')),
  attempt_count INTEGER NOT NULL DEFAULT 0 CHECK (attempt_count >= 0),
  current_attempt_index INTEGER NOT NULL DEFAULT 0 CHECK (current_attempt_index >= 0),
  resume_token_hash TEXT NOT NULL UNIQUE CHECK (length(resume_token_hash) = length('ph-mop-resume-v1_') + 64 AND substr(resume_token_hash, 1, length('ph-mop-resume-v1_')) = 'ph-mop-resume-v1_' AND substr(resume_token_hash, length('ph-mop-resume-v1_') + 1) NOT GLOB '*[^0-9a-f]*'),
  auth_expires_at TEXT NOT NULL CHECK (length(auth_expires_at) > 0),
  failure_reason TEXT CHECK (failure_reason IS NULL OR failure_reason IN ('authorization_expired', 'terminal_provider_failure', 'cleanup_failed_terminal')),
  client_action TEXT NOT NULL CHECK (client_action IN ('wait', 'retry_authorized', 'acknowledge_delivery', 'action_required')),
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  last_reconciled_at TEXT,
  cleanup_attempts INTEGER NOT NULL DEFAULT 0 CHECK (cleanup_attempts >= 0),
  hardware_hash TEXT CHECK (hardware_hash IS NULL OR length(hardware_hash) BETWEEN 1 AND 128),
  hardware_hash_salt_version INTEGER CHECK (hardware_hash_salt_version IS NULL OR hardware_hash_salt_version >= 0),
  app_version TEXT CHECK (app_version IS NULL OR length(app_version) BETWEEN 1 AND 64),
  CHECK ((state <> 'FAILED' AND failure_reason IS NULL) OR (state = 'FAILED' AND failure_reason IS NOT NULL) OR state <> 'FAILED'),
  CHECK (attempt_count >= current_attempt_index)
) STRICT;
INSERT INTO managed_operations SELECT * FROM managed_operations_backup_0031;
CREATE INDEX idx_managed_operations_state_updated_at ON managed_operations(state, updated_at);
CREATE INDEX idx_managed_operations_auth_expires_at ON managed_operations(auth_expires_at) WHERE state NOT IN ('ACTIVE', 'FAILED');
CREATE INDEX idx_managed_operations_subject_source ON managed_operations(issue_source, subject_ref, created_at);

CREATE TABLE managed_operation_attempts (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  operation_id TEXT NOT NULL REFERENCES managed_operations(operation_id) ON DELETE CASCADE,
  attempt_index INTEGER NOT NULL CHECK (attempt_index > 0),
  provider_key_name TEXT NOT NULL CHECK (length(provider_key_name) BETWEEN 1 AND 200),
  managed_credential_ref TEXT CHECK (managed_credential_ref IS NULL OR length(managed_credential_ref) > 0),
  outcome TEXT NOT NULL CHECK (outcome IN ('created', 'unknown', 'cleaned')),
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  cleanup_token TEXT,
  cleanup_started_at TEXT,
  scan_offset INTEGER NOT NULL DEFAULT 0 CHECK (scan_offset >= 0),
  create_in_flight INTEGER NOT NULL DEFAULT 0 CHECK (create_in_flight IN (0, 1)),
  legacy_cleanup_state TEXT NOT NULL DEFAULT 'unclassified' CHECK (legacy_cleanup_state IN ('unclassified', 'pending', 'resolved')),
  legacy_cleanup_checked_at TEXT
) STRICT;
INSERT INTO managed_operation_attempts SELECT * FROM managed_attempts_backup_0031;
UPDATE sqlite_sequence SET seq = max(seq, (SELECT seq FROM managed_attempt_sequence_0031)) WHERE name = 'managed_operation_attempts';
INSERT INTO sqlite_sequence (name, seq) SELECT 'managed_operation_attempts', seq FROM managed_attempt_sequence_0031 WHERE NOT EXISTS (SELECT 1 FROM sqlite_sequence WHERE name = 'managed_operation_attempts');
CREATE UNIQUE INDEX idx_managed_operation_attempts_operation_index ON managed_operation_attempts(operation_id, attempt_index);
CREATE UNIQUE INDEX idx_managed_operation_attempts_provider_key_name ON managed_operation_attempts(provider_key_name);
CREATE INDEX idx_managed_attempt_cleanup_pending ON managed_operation_attempts(operation_id, outcome, cleanup_started_at);
CREATE INDEX idx_managed_attempt_legacy_cleanup ON managed_operation_attempts(legacy_cleanup_state, legacy_cleanup_checked_at, id);
CREATE INDEX idx_managed_attempt_create_pending ON managed_operation_attempts(operation_id, attempt_index) WHERE create_in_flight = 1;

CREATE TABLE managed_key_deliveries (
  delivery_id TEXT PRIMARY KEY,
  issue_source TEXT NOT NULL CHECK (issue_source IN ('discord', 'qq', 'photo')),
  subject_ref TEXT,
  installation_id TEXT,
  managed_credential_ref TEXT NOT NULL,
  ack_token_hash TEXT NOT NULL UNIQUE,
  status TEXT NOT NULL CHECK (status IN ('pending', 'acknowledged', 'expired', 'cleanup_required')),
  created_at TEXT NOT NULL,
  expires_at TEXT NOT NULL,
  acknowledged_at TEXT,
  failed_at TEXT,
  failure_reason TEXT,
  operation_id TEXT REFERENCES managed_operations(operation_id) ON DELETE SET NULL,
  attempt_index INTEGER CHECK (attempt_index IS NULL OR attempt_index > 0),
  CHECK (length(delivery_id) > 0),
  CHECK (length(managed_credential_ref) > 0),
  CHECK (length(ack_token_hash) = length('ph-delivery-ack-token-v1_') + 64 AND substr(ack_token_hash, 1, length('ph-delivery-ack-token-v1_')) = 'ph-delivery-ack-token-v1_' AND substr(ack_token_hash, length('ph-delivery-ack-token-v1_') + 1) NOT GLOB '*[^0-9a-f]*'),
  CHECK (status <> 'acknowledged' OR acknowledged_at IS NOT NULL),
  CHECK (status = 'acknowledged' OR acknowledged_at IS NULL),
  CHECK (status <> 'cleanup_required' OR failed_at IS NOT NULL),
  CHECK (status <> 'pending' OR (acknowledged_at IS NULL AND failed_at IS NULL AND failure_reason IS NULL))
) STRICT;
INSERT INTO managed_key_deliveries SELECT * FROM managed_deliveries_backup_0031;
CREATE INDEX idx_managed_key_deliveries_status_expires_at ON managed_key_deliveries(status, expires_at);
CREATE INDEX idx_managed_key_deliveries_managed_credential_ref ON managed_key_deliveries(managed_credential_ref);
CREATE INDEX idx_managed_key_deliveries_issue_source_created_at ON managed_key_deliveries(issue_source, created_at);
CREATE INDEX idx_managed_key_deliveries_operation_id ON managed_key_deliveries(operation_id) WHERE operation_id IS NOT NULL;
CREATE TRIGGER managed_delivery_ack_cleanup_guard
BEFORE UPDATE OF status ON managed_key_deliveries
WHEN NEW.status = 'acknowledged' AND OLD.status <> 'acknowledged'
 AND NEW.operation_id IS NOT NULL
 AND NOT EXISTS (
   SELECT 1 FROM managed_operations operation
   JOIN managed_operation_attempts attempt ON attempt.operation_id = operation.operation_id
    AND attempt.attempt_index = NEW.attempt_index
   WHERE operation.operation_id = NEW.operation_id
    AND operation.current_attempt_index = NEW.attempt_index
    AND operation.state IN ('CREATING', 'DELIVERY_PENDING', 'FAILED')
    AND attempt.outcome = 'created' AND attempt.cleanup_token IS NULL
    AND attempt.managed_credential_ref = NEW.managed_credential_ref
 )
BEGIN
 SELECT RAISE(ABORT, 'managed delivery cleanup owns attempt');
END;

DROP TABLE managed_operations_backup_0031;
DROP TABLE managed_attempts_backup_0031;
DROP TABLE managed_deliveries_backup_0031;
DROP TABLE managed_attempt_sequence_0031;

ALTER TABLE managed_operations ADD COLUMN photo_period_start INTEGER;
ALTER TABLE managed_operations ADD COLUMN photo_limit_usd REAL;
ALTER TABLE managed_operations ADD COLUMN photo_activated_at TEXT;
ALTER TABLE managed_operations ADD COLUMN photo_revoked_at TEXT;
CREATE UNIQUE INDEX idx_managed_photo_period ON managed_operations(subject_ref, photo_period_start) WHERE issue_source = 'photo';
PRAGMA foreign_key_check;

CREATE TRIGGER managed_photo_reissue_guard
BEFORE UPDATE OF state ON managed_operations
WHEN NEW.issue_source = 'photo' AND NEW.state = 'CREATING'
 AND (NEW.photo_activated_at IS NOT NULL OR NEW.photo_revoked_at IS NOT NULL)
BEGIN
 SELECT RAISE(ABORT, 'photo allowance already activated or revoked');
END;
