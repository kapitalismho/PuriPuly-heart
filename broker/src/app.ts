import { Hono, type Context } from 'hono';

import { internalErrorResponse } from './broker-error';
import {
  BROKER_SERVICE_NAME,
  FOUNDATION_RESPONSE,
  type BrokerEnv,
} from './contract';
import {
  handleTrialChallenge,
  handleTrialChallengeVerify,
  handleTrialStatus,
} from './trial-handshake';
import { handleOpenRouterIssue } from './openrouter-issue';
import {
  handleDiscordAuthStart,
  handleDiscordOpenRouterIssue,
} from './discord-managed-issue';
import { handleQqAuthAssert, handleQqAuthStatus } from './qq-auth';
import { handleAppActiveDay } from './telemetry';
import { handleManagedKeyDeliveryAck } from './managed-key-delivery';

export const app = new Hono<BrokerEnv>();

app.onError((_error: Error, c: Context<BrokerEnv>) => {
  return internalErrorResponse(c);
});

app.get('/healthz', (c: Context<BrokerEnv>) => {
  return c.json({
    ok: true,
    service: BROKER_SERVICE_NAME,
  });
});

app.get('/v1/foundation', (c: Context<BrokerEnv>) => {
  return c.json(FOUNDATION_RESPONSE);
});

app.post('/v1/trial/challenge', handleTrialChallenge);
app.post('/v1/trial/challenge/verify', handleTrialChallengeVerify);
app.get('/v1/trial/status', handleTrialStatus);
app.post('/v1/auth/discord/start', handleDiscordAuthStart);
app.post('/v1/auth/qq/assert', handleQqAuthAssert);
app.post('/v1/auth/qq/status', handleQqAuthStatus);
app.post('/v1/telemetry/app-active-day', handleAppActiveDay);
app.post('/v1/providers/openrouter/issue', handleOpenRouterIssue);
app.post('/v1/providers/openrouter/discord/issue', handleDiscordOpenRouterIssue);
app.post('/v1/providers/openrouter/managed-key-delivery/ack', handleManagedKeyDeliveryAck);

export default app;
