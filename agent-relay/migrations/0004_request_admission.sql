ALTER TABLE prompt_turns ADD COLUMN expires_at INTEGER NOT NULL DEFAULT 0;
ALTER TABLE prompt_turns ADD COLUMN model_admissions INTEGER NOT NULL DEFAULT 15 CHECK (model_admissions BETWEEN 1 AND 15);
ALTER TABLE prompt_turns ADD COLUMN classifier_admitted INTEGER NOT NULL DEFAULT 1 CHECK (classifier_admitted IN (0, 1));

CREATE TABLE request_slots (
  installation_hash TEXT NOT NULL,
  slot INTEGER NOT NULL CHECK (slot BETWEEN 0 AND 2),
  owner TEXT NOT NULL,
  turn_id TEXT,
  classification INTEGER NOT NULL CHECK (classification IN (0, 1)),
  expires_at INTEGER NOT NULL,
  PRIMARY KEY (installation_hash, slot)
);
