CREATE TABLE puri_token_rate_limits (
  policy TEXT NOT NULL CHECK (policy IN ('ip_flood', 'ip_registration', 'installation')),
  subject_key TEXT NOT NULL,
  window_started_at TEXT NOT NULL,
  expires_at TEXT NOT NULL,
  request_count INTEGER NOT NULL CHECK (request_count > 0),
  ip_key_version INTEGER,
  ip_epoch TEXT,
  PRIMARY KEY (policy, subject_key, window_started_at)
) STRICT;

CREATE INDEX puri_token_rate_limits_expiry_idx ON puri_token_rate_limits(expires_at);

UPDATE broker_config
   SET value = json_set(
         value,
         '$.puriTokenIp', json(CASE
           WHEN json_extract(value, '$.puriTokenIp.maxRequests') = 30
            AND json_extract(value, '$.puriTokenIp.windowMinutes') = 15
            AND json_extract(value, '$.puriTokenIp.endpoint') = 'POST /v1/puri/token'
            AND json_extract(value, '$.puriTokenIp.scope') = 'ip'
           THEN json('{"endpoint":"POST /v1/puri/token","scope":"ip","maxRequests":600,"windowMinutes":15}')
           ELSE COALESCE(
             json_extract(value, '$.puriTokenIp'),
             json('{"endpoint":"POST /v1/puri/token","scope":"ip","maxRequests":600,"windowMinutes":15}')
           )
         END),
         '$.puriTokenRegistrationIp', json(COALESCE(
           json_extract(value, '$.puriTokenRegistrationIp'),
           json('{"endpoint":"POST /v1/puri/token","scope":"ip","maxRequests":20,"windowMinutes":15}')
         )),
         '$.puriTokenInstallation', json(COALESCE(
           json_extract(value, '$.puriTokenInstallation'),
           json('{"endpoint":"POST /v1/puri/token","scope":"installation_id","maxRequests":8,"windowMinutes":15}')
         ))
       ),
       updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
 WHERE key = 'abuse_controls';
