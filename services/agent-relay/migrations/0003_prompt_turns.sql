CREATE TABLE prompt_turns (
  installation_hash TEXT NOT NULL,
  turn_id TEXT NOT NULL,
  utc_date TEXT NOT NULL,
  PRIMARY KEY (installation_hash, turn_id)
);

CREATE INDEX prompt_turns_installation_utc_date ON prompt_turns (installation_hash, utc_date);
