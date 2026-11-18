CREATE TABLE qq_auth_assertions (
  qq_subject_ref TEXT PRIMARY KEY,
  credential_hash TEXT NOT NULL,
  asserted_at TEXT NOT NULL,
  received_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  status TEXT NOT NULL CHECK(status IN ('verified'))
) STRICT;

UPDATE broker_config
   SET value = json_insert(
         value,
         '$.qqAuthAssertIp', json('{"endpoint":"POST /v1/auth/qq/assert","scope":"ip","maxRequests":20,"windowMinutes":15}')
       ),
       updated_at = CURRENT_TIMESTAMP
 WHERE key = 'abuse_controls';
