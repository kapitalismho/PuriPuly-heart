-- Forward migration: remove retired translation-success telemetry storage
-- (issue #74 Phase 2 approved deletion direction).
-- Drops telemetry_active_days, telemetry_subjects, and their obsolete
-- dependent trigger and indexes. Preserves app_active_days and all
-- unrelated production tables. Applying this migration to production D1
-- deletes the retired rows and requires separate explicit approval; this
-- file alone does not authorize production deletion or deployment.
DROP TRIGGER IF EXISTS telemetry_active_days_sync_subject_after_insert;
DROP INDEX IF EXISTS idx_telemetry_active_days_date;
DROP INDEX IF EXISTS idx_telemetry_active_days_received;
DROP INDEX IF EXISTS idx_telemetry_subjects_last_active_date;
DROP TABLE IF EXISTS telemetry_active_days;
DROP TABLE IF EXISTS telemetry_subjects;
