CREATE TABLE photo_admissions (
  installation_hash TEXT NOT NULL,
  request_id TEXT NOT NULL,
  period_start INTEGER NOT NULL,
  owner TEXT NOT NULL,
  expires_at INTEGER NOT NULL,
  settled INTEGER NOT NULL DEFAULT 0 CHECK (settled IN (0, 1)),
  uncached_input_tokens INTEGER NOT NULL DEFAULT 0,
  cache_read_tokens INTEGER NOT NULL DEFAULT 0,
  cache_write_tokens INTEGER NOT NULL DEFAULT 0,
  output_tokens INTEGER NOT NULL DEFAULT 0,
  missing_usage INTEGER NOT NULL DEFAULT 1,
  cost_usd REAL NOT NULL DEFAULT 0,
  PRIMARY KEY (installation_hash, request_id)
);
CREATE INDEX photo_admissions_period ON photo_admissions (installation_hash, period_start);
CREATE INDEX photo_admissions_active ON photo_admissions (installation_hash, settled, expires_at);
