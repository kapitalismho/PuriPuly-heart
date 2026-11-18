ALTER TABLE managed_operation_attempts ADD COLUMN cleanup_token TEXT;
ALTER TABLE managed_operation_attempts ADD COLUMN cleanup_started_at TEXT;
ALTER TABLE managed_operation_attempts ADD COLUMN scan_offset INTEGER NOT NULL DEFAULT 0 CHECK (scan_offset >= 0);
ALTER TABLE managed_operation_attempts ADD COLUMN create_in_flight INTEGER NOT NULL DEFAULT 0 CHECK (create_in_flight IN (0, 1));
ALTER TABLE managed_operation_attempts ADD COLUMN legacy_cleanup_state TEXT NOT NULL DEFAULT 'unclassified' CHECK (legacy_cleanup_state IN ('unclassified', 'pending', 'resolved'));
ALTER TABLE managed_operation_attempts ADD COLUMN legacy_cleanup_checked_at TEXT;
UPDATE managed_operation_attempts
SET create_in_flight = CASE WHEN outcome = 'unknown' AND managed_credential_ref IS NULL THEN 1 ELSE create_in_flight END,
    legacy_cleanup_state = CASE
      WHEN (outcome = 'cleaned' AND managed_credential_ref IS NULL)
        OR (outcome <> 'cleaned' AND EXISTS (
          SELECT 1 FROM managed_operations operation
          WHERE operation.operation_id = managed_operation_attempts.operation_id
            AND operation.current_attempt_index > managed_operation_attempts.attempt_index
        )) THEN 'pending'
      ELSE 'resolved' END
WHERE legacy_cleanup_state = 'unclassified';
CREATE INDEX idx_managed_attempt_cleanup_pending ON managed_operation_attempts(operation_id, outcome, cleanup_started_at);
CREATE INDEX idx_managed_attempt_legacy_cleanup ON managed_operation_attempts(legacy_cleanup_state, legacy_cleanup_checked_at, id);
CREATE INDEX idx_managed_attempt_create_pending ON managed_operation_attempts(operation_id, attempt_index) WHERE create_in_flight = 1;
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
CREATE TRIGGER managed_discord_entitlement_cleanup_guard
BEFORE UPDATE OF status, discord_issue_status, managed_credential_ref ON openrouter_entitlements
WHEN NEW.managed_credential_ref IS NOT NULL
 AND (NEW.status = 'active' OR NEW.discord_issue_status = 'delivery_pending')
 AND EXISTS (SELECT 1 FROM managed_operation_attempts WHERE managed_credential_ref = NEW.managed_credential_ref)
 AND NOT EXISTS (
   SELECT 1 FROM managed_operation_attempts attempt
   JOIN managed_operations operation ON operation.operation_id = attempt.operation_id
   WHERE attempt.managed_credential_ref = NEW.managed_credential_ref
    AND operation.current_attempt_index = attempt.attempt_index
    AND operation.state IN ('CREATING', 'DELIVERY_PENDING', 'ACTIVE', 'FAILED')
    AND attempt.outcome = 'created' AND attempt.cleanup_token IS NULL
 )
BEGIN
 SELECT RAISE(ABORT, 'managed delivery cleanup owns attempt');
END;
CREATE TRIGGER managed_qq_entitlement_cleanup_guard
BEFORE UPDATE OF status, managed_credential_ref ON qq_managed_entitlements
WHEN NEW.managed_credential_ref IS NOT NULL
 AND NEW.status IN ('active', 'delivery_pending')
 AND EXISTS (SELECT 1 FROM managed_operation_attempts WHERE managed_credential_ref = NEW.managed_credential_ref)
 AND NOT EXISTS (
   SELECT 1 FROM managed_operation_attempts attempt
   JOIN managed_operations operation ON operation.operation_id = attempt.operation_id
   WHERE attempt.managed_credential_ref = NEW.managed_credential_ref
    AND operation.current_attempt_index = attempt.attempt_index
    AND operation.state IN ('CREATING', 'DELIVERY_PENDING', 'ACTIVE', 'FAILED')
    AND attempt.outcome = 'created' AND attempt.cleanup_token IS NULL
 )
BEGIN
 SELECT RAISE(ABORT, 'managed delivery cleanup owns attempt');
END;
