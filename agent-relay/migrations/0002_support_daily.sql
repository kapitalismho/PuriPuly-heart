CREATE TABLE support_daily (
  installation_hash TEXT NOT NULL,
  utc_date TEXT NOT NULL,
  reports INTEGER NOT NULL CHECK (reports BETWEEN 0 AND 3),
  PRIMARY KEY (installation_hash, utc_date)
);
