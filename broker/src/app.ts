import { Hono, type Context } from 'hono';

import { internalErrorResponse } from './broker-error';
import {
  BROKER_SERVICE_NAME,
  BROKER_SOURCE_OFFER,
  BROKER_SOURCE_OFFER_LINK_HEADER,
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
import { handleQqAuthAssert } from './qq-auth';
import { handleTelemetryTranslationSuccessDay } from './telemetry';

export const app = new Hono<BrokerEnv>();

app.use('*', async (c: Context<BrokerEnv>, next) => {
  await next();
  c.header('Link', BROKER_SOURCE_OFFER_LINK_HEADER);
});

app.onError((_error: Error, c: Context<BrokerEnv>) => {
  c.header('Link', BROKER_SOURCE_OFFER_LINK_HEADER);
  return internalErrorResponse(c);
});

app.get('/healthz', (c: Context<BrokerEnv>) => {
  return c.json({
    ok: true,
    service: BROKER_SERVICE_NAME,
  });
});

app.get('/source', (c: Context<BrokerEnv>) => {
  return c.json(BROKER_SOURCE_OFFER);
});

app.get('/v1/foundation', (c: Context<BrokerEnv>) => {
  return c.json(FOUNDATION_RESPONSE);
});

app.post('/v1/trial/challenge', handleTrialChallenge);
app.post('/v1/trial/challenge/verify', handleTrialChallengeVerify);
app.get('/v1/trial/status', handleTrialStatus);
app.post('/v1/auth/discord/start', handleDiscordAuthStart);
app.post('/v1/auth/qq/assert', handleQqAuthAssert);
app.post('/v1/telemetry/translation-success-day', handleTelemetryTranslationSuccessDay);
app.post('/v1/providers/openrouter/issue', handleOpenRouterIssue);
app.post('/v1/providers/openrouter/discord/issue', handleDiscordOpenRouterIssue);

export default app;
