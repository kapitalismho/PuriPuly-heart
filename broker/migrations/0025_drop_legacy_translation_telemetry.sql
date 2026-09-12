DROP TRIGGER IF EXISTS telemetry_active_days_sync_subject_after_insert;
DROP INDEX IF EXISTS idx_telemetry_active_days_date;
DROP INDEX IF EXISTS idx_telemetry_active_days_received;
DROP INDEX IF EXISTS idx_telemetry_subjects_last_active_date;
DROP TABLE IF EXISTS telemetry_active_days;
DROP TABLE IF EXISTS telemetry_subjects;
