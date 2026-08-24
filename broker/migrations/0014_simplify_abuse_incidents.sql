UPDATE broker_config
   SET value = json_set(
         value,
         '$.immediateAlerts.warning', COALESCE(json_extract(value, '$.immediateAlerts.warning'), json_extract(value, '$.immediateAlerts.warn1'), 10),
         '$.immediateAlerts.brake', COALESCE(json_extract(value, '$.immediateAlerts.brake'), json_extract(value, '$.immediateAlerts.critical'), 70),
         '$.retention.requestEventSafetyMarginDays', COALESCE(json_extract(value, '$.retention.requestEventSafetyMarginDays'), 1)
       ),
       updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
 WHERE key = 'abuse_controls';

UPDATE broker_config
   SET value = json_set(
         value,
         '$.alertLatches.warning',
         json(
           CASE
             WHEN COALESCE(json_extract(value, '$.alertLatches.warning'), 0) = 1
               OR COALESCE(json_extract(value, '$.alertLatches.warn1'), 0) = 1
               OR COALESCE(json_extract(value, '$.alertLatches.warn2'), 0) = 1
               OR COALESCE(json_extract(value, '$.alertLatches.warn3'), 0) = 1
               OR COALESCE(json_extract(value, '$.alertLatches.critical'), 0) = 1
             THEN 'true'
             ELSE 'false'
           END
         ),
         '$.alertLatches.warningObservedAt', COALESCE(json_extract(value, '$.alertLatches.warningObservedAt'), NULL)
       ),
       updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
 WHERE key = 'abuse_runtime_state';

ALTER TABLE qq_managed_entitlements
  ADD COLUMN child_key_creation_started_at TEXT;
