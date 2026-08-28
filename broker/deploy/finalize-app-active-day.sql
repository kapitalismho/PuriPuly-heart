UPDATE broker_config
   SET value = json_remove(value, '$.telemetryTranslationSuccessDayIp'),
       updated_at = CURRENT_TIMESTAMP
 WHERE key = 'abuse_controls'
   AND json_type(value, '$.telemetryTranslationSuccessDayIp') IS NOT NULL;
