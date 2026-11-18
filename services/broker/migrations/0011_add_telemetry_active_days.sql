CREATE TABLE telemetry_active_days (
  subject_ref TEXT NOT NULL CHECK (
    length(subject_ref) = length('ph-telemetry-subject-v1_') + 64
    AND substr(subject_ref, 1, length('ph-telemetry-subject-v1_')) = 'ph-telemetry-subject-v1_'
    AND substr(subject_ref, length('ph-telemetry-subject-v1_') + 1) NOT GLOB '*[^0-9a-f]*'
  ),
  active_date_utc TEXT NOT NULL CHECK (
    active_date_utc GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]'
  ),
  first_received_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  last_received_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (subject_ref, active_date_utc),
  CHECK (last_received_at >= first_received_at)
) STRICT;

CREATE INDEX idx_telemetry_active_days_date
  ON telemetry_active_days(active_date_utc);
CREATE INDEX idx_telemetry_active_days_received
  ON telemetry_active_days(last_received_at);

UPDATE broker_config
   SET value = json_set(
         value,
         '$.telemetryTranslationSuccessDayIp',
         json('{"endpoint":"POST /v1/telemetry/translation-success-day","scope":"ip","maxRequests":60,"windowMinutes":15}')
       ),
       updated_at = CURRENT_TIMESTAMP
 WHERE key = 'abuse_controls'
   AND json_type(value, '$.telemetryTranslationSuccessDayIp') IS NULL;
