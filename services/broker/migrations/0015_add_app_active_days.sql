CREATE TABLE app_active_days (
  subject_ref TEXT NOT NULL CHECK (
    length(subject_ref) = length('ph-app-subject-v1_') + 64
    AND substr(subject_ref, 1, length('ph-app-subject-v1_')) = 'ph-app-subject-v1_'
    AND substr(subject_ref, length('ph-app-subject-v1_') + 1) NOT GLOB '*[^0-9a-f]*'
  ),
  active_date_utc TEXT NOT NULL CHECK (
    active_date_utc GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]'
  ),
  PRIMARY KEY (subject_ref, active_date_utc)
) STRICT;

CREATE INDEX idx_app_active_days_date
  ON app_active_days(active_date_utc);
