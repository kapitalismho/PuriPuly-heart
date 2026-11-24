CREATE TABLE puri_token_baseline (
  day_utc TEXT NOT NULL CHECK (
    day_utc GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]' AND date(day_utc, '+0 days') IS day_utc
  ),
  version_id TEXT NOT NULL CHECK (length(version_id) = 36),
  ingress_colo TEXT NOT NULL CHECK (ingress_colo = 'unknown' OR ingress_colo GLOB '[A-Z][A-Z][A-Z]'),
  cohort TEXT NOT NULL CHECK (cohort IN ('enrollment', 'renewal', 'unknown')),
  status INTEGER NOT NULL CHECK (status BETWEEN 100 AND 599),
  upper_ms INTEGER NOT NULL CHECK (upper_ms IN (-1, 25, 50, 100, 200, 400, 800, 1600, 3200, 6400)),
  requests INTEGER NOT NULL CHECK (requests > 0),
  PRIMARY KEY (day_utc, version_id, ingress_colo, cohort, status, upper_ms)
) STRICT, WITHOUT ROWID;
