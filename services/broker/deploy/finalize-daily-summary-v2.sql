UPDATE broker_config
   SET value = json_remove(value, '$.dailyReport.includeZeroActivity'),
       updated_at = CURRENT_TIMESTAMP
 WHERE key = 'abuse_controls';
