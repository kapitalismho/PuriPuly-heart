ALTER TABLE installations ADD COLUMN puri_bound_at TEXT;

CREATE TABLE puri_proof_nonces (
  installation_id TEXT NOT NULL REFERENCES installations(installation_id) ON DELETE CASCADE,
  nonce TEXT NOT NULL,
  expires_at TEXT NOT NULL,
  PRIMARY KEY (installation_id, nonce)
) STRICT;

CREATE INDEX puri_proof_nonces_expiry_idx ON puri_proof_nonces(expires_at);

UPDATE broker_config
   SET value = json_set(
         value,
         '$.puriTokenIp', json('{"endpoint":"POST /v1/puri/token","scope":"ip","maxRequests":30,"windowMinutes":15}')
       ),
       updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
 WHERE key = 'abuse_controls';
