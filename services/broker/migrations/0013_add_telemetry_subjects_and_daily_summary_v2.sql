CREATE TABLE telemetry_subjects (
  subject_ref TEXT PRIMARY KEY CHECK (
    length(subject_ref) = length('ph-telemetry-subject-v1_') + 64
    AND substr(subject_ref, 1, length('ph-telemetry-subject-v1_')) = 'ph-telemetry-subject-v1_'
    AND substr(subject_ref, length('ph-telemetry-subject-v1_') + 1) NOT GLOB '*[^0-9a-f]*'
  ),
  first_active_date_utc TEXT NOT NULL CHECK (
    first_active_date_utc GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]'
  ),
  last_active_date_utc TEXT NOT NULL CHECK (
    last_active_date_utc GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]'
  ),
  CHECK (last_active_date_utc >= first_active_date_utc)
) STRICT;

INSERT INTO telemetry_subjects (
  subject_ref,
  first_active_date_utc,
  last_active_date_utc
)
SELECT subject_ref,
       MIN(active_date_utc),
       MAX(active_date_utc)
  FROM telemetry_active_days
 GROUP BY subject_ref;

CREATE TRIGGER telemetry_active_days_sync_subject_after_insert
AFTER INSERT ON telemetry_active_days
BEGIN
  INSERT INTO telemetry_subjects (
    subject_ref,
    first_active_date_utc,
    last_active_date_utc
  ) VALUES (
    NEW.subject_ref,
    NEW.active_date_utc,
    NEW.active_date_utc
  )
  ON CONFLICT(subject_ref) DO UPDATE SET
    first_active_date_utc = MIN(
      telemetry_subjects.first_active_date_utc,
      excluded.first_active_date_utc
    ),
    last_active_date_utc = MAX(
      telemetry_subjects.last_active_date_utc,
      excluded.last_active_date_utc
    );
END;

CREATE INDEX idx_telemetry_subjects_last_active_date
  ON telemetry_subjects(last_active_date_utc);

CREATE TABLE broker_daily_summary_deliveries (
  report_date_utc TEXT PRIMARY KEY CHECK (
    report_date_utc GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]'
  ),
  status TEXT NOT NULL CHECK (status IN ('pending', 'delivered')),
  lease_token TEXT NOT NULL CHECK (
    length(lease_token) = 36
    AND substr(lease_token, 9, 1) = '-'
    AND substr(lease_token, 14, 1) = '-'
    AND substr(lease_token, 19, 1) = '-'
    AND substr(lease_token, 24, 1) = '-'
    AND replace(lease_token, '-', '') NOT GLOB '*[^0-9a-f]*'
  ),
  lease_expires_at TEXT NOT NULL CHECK (julianday(lease_expires_at) IS NOT NULL),
  attempted_at TEXT NOT NULL CHECK (julianday(attempted_at) IS NOT NULL),
  delivered_at TEXT,
  CHECK (
    julianday(lease_expires_at) >= julianday(attempted_at)
    AND julianday(lease_expires_at) <= julianday(attempted_at, '+15 minutes')
  ),
  CHECK (
    (status = 'pending' AND delivered_at IS NULL)
    OR (
      status = 'delivered'
      AND julianday(delivered_at) IS NOT NULL
      AND julianday(delivered_at) >= julianday(attempted_at)
    )
  )
) STRICT;

CREATE INDEX idx_broker_daily_summary_deliveries_status_lease
  ON broker_daily_summary_deliveries(status, report_date_utc, lease_expires_at);

UPDATE broker_config
   SET value = json_set(
         value,
         '$.dailyReport.hourUtc',
         0,
         '$.dailyReport.minuteUtc',
         5,
         '$.retention.issueSuccessDays',
         CASE
           WHEN json_type(value, '$.retention.issueSuccessDays') = 'integer'
            AND json_extract(value, '$.retention.issueSuccessDays') >= 2
           THEN json_extract(value, '$.retention.issueSuccessDays')
           ELSE 2
         END
       ),
       updated_at = CURRENT_TIMESTAMP
 WHERE key = 'abuse_controls';
