CREATE TABLE usage_daily (
  installation_hash TEXT NOT NULL,
  utc_date TEXT NOT NULL,
  uncached_input_tokens INTEGER NOT NULL DEFAULT 0,
  cache_read_tokens INTEGER NOT NULL DEFAULT 0,
  cache_write_tokens INTEGER NOT NULL DEFAULT 0,
  decisions_output_tokens INTEGER NOT NULL DEFAULT 0,
  responses_output_tokens INTEGER NOT NULL DEFAULT 0,
  decisions_calls INTEGER NOT NULL DEFAULT 0,
  responses_calls INTEGER NOT NULL DEFAULT 0,
  missing_usage_count INTEGER NOT NULL DEFAULT 0,
  cost_usd REAL NOT NULL DEFAULT 0,
  PRIMARY KEY (installation_hash, utc_date)
);

CREATE INDEX usage_daily_utc_date ON usage_daily (utc_date);
