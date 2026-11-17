CREATE TABLE protocol_daily (
  installation_hash TEXT NOT NULL,
  utc_date TEXT NOT NULL,
  agent_protocol INTEGER NOT NULL,
  requests INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY (installation_hash, utc_date, agent_protocol)
);

CREATE INDEX protocol_daily_utc_date ON protocol_daily (utc_date);
