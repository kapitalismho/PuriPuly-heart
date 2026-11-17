UPDATE broker_config
   SET value = json_insert(
         value,
         '$.managedOperationStatusIp', json('{"endpoint":"POST /v1/providers/openrouter/managed-operation/status","scope":"ip","maxRequests":30,"windowMinutes":15}'),
         '$.managedOperationStatusInstallation', json('{"endpoint":"POST /v1/providers/openrouter/managed-operation/status","scope":"installation_id","maxRequests":30,"windowMinutes":15}'),
         '$.managedOperationResumeIp', json('{"endpoint":"POST /v1/providers/openrouter/managed-operation/resume","scope":"ip","maxRequests":20,"windowMinutes":15}'),
         '$.managedOperationResumeInstallation', json('{"endpoint":"POST /v1/providers/openrouter/managed-operation/resume","scope":"installation_id","maxRequests":10,"windowMinutes":15}'),
         '$.managedKeyDeliveryAckIp', json('{"endpoint":"POST /v1/providers/openrouter/managed-key-delivery/ack","scope":"ip","maxRequests":30,"windowMinutes":15}')
       ),
       updated_at = CURRENT_TIMESTAMP
 WHERE key = 'abuse_controls';
