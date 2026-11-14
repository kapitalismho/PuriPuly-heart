ALTER TABLE prompt_turns ADD COLUMN response_model TEXT CHECK (response_model IN ('gpt-5.6-terra', 'gpt-6-luna'));

UPDATE prompt_turns SET response_model = 'gpt-6-luna' WHERE model_admissions > classifier_admitted;

CREATE TABLE terra_daily (
  utc_date TEXT PRIMARY KEY,
  tokens INTEGER NOT NULL DEFAULT 0 CHECK (tokens >= 0)
);
