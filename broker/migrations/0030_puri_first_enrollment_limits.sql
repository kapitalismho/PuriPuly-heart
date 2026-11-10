CREATE TABLE puri_enrollment_admissions (
  admission_id INTEGER PRIMARY KEY,
  subject_key TEXT NOT NULL,
  admitted_at TEXT NOT NULL,
  ip_key_version INTEGER,
  ip_epoch TEXT
) STRICT;

CREATE INDEX puri_enrollment_admissions_subject_idx
  ON puri_enrollment_admissions(subject_key, admitted_at);
CREATE INDEX puri_enrollment_admissions_admitted_idx
  ON puri_enrollment_admissions(admitted_at);

UPDATE broker_config
   SET value = json_remove(
         json_set(
           value,
           '$.puriTokenIp', json(CASE
             WHEN json_extract(value, '$.puriTokenIp.maxRequests') = 600
              AND json_extract(value, '$.puriTokenIp.windowMinutes') = 15
              AND json_extract(value, '$.puriTokenIp.endpoint') = 'POST /v1/puri/token'
              AND json_extract(value, '$.puriTokenIp.scope') = 'ip'
             THEN json('{"endpoint":"POST /v1/puri/token","scope":"ip","maxRequests":200,"windowMinutes":15}')
             ELSE COALESCE(
               json_extract(value, '$.puriTokenIp'),
               json('{"endpoint":"POST /v1/puri/token","scope":"ip","maxRequests":200,"windowMinutes":15}')
             )
           END),
           '$.puriEnrollmentShortIp', json(COALESCE(
             json_extract(value, '$.puriEnrollmentShortIp'),
             json('{"endpoint":"POST /v1/puri/token","scope":"ip","maxRequests":5,"windowMinutes":60}')
           )),
           '$.puriEnrollmentLongIp', json(COALESCE(
             json_extract(value, '$.puriEnrollmentLongIp'),
             json('{"endpoint":"POST /v1/puri/token","scope":"ip","maxRequests":10,"windowMinutes":1440}')
           ))
         ),
         '$.puriTokenRegistrationIp'
       ),
       updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
 WHERE key = 'abuse_controls';
