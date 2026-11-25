import { Hono, type Context } from 'hono';
import { bodyLimit } from 'hono/body-limit';

import { createServicePage } from '../../service-page';
import { errorResponse, internalErrorResponse } from './broker-error';
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
import { handleQqAuthAssert } from './qq-auth';
import { handleAppActiveDay } from './telemetry';
import { handleManagedKeyDeliveryAck } from './managed-key-delivery';
import { handleManagedOperationResume } from './managed-operation-resume';
import { handleManagedOperationStatus } from './managed-operation';
import { handlePuriToken } from './puri-token';
import { handlePhotoAccess } from './photo-access';

export const app = new Hono<BrokerEnv>();

const rootPage = createServicePage(
  'Broker',
  'Broker authenticates PuriPuly Heart installations, checks access to managed features, and issues the credentials the app needs to use them. Speech recognition and translation requests do not pass through Broker.',
);

app.onError((_error: Error, c: Context<BrokerEnv>) => {
  return internalErrorResponse(c);
});

app.get('/', (c) => rootPage(new URL(c.req.url), c.req.method) ?? c.notFound());

app.get('/healthz', (c: Context<BrokerEnv>) => {
  return c.json({
    ok: true,
    service: BROKER_SERVICE_NAME,
  });
});

app.get('/v1/foundation', (c: Context<BrokerEnv>) => {
  return c.json(FOUNDATION_RESPONSE);
});

app.post('/v1/puri/token', bodyLimit({
  maxSize: 4096,
  onError: (c) => {
    c.header('Cache-Control', 'no-store');
    return errorResponse(c, 413, {
      code: 'invalid_request', class: 'terminal', message: 'Puri authentication request is too large',
    });
  },
}), handlePuriToken);
app.post('/v1/photo/access', bodyLimit({ maxSize: 4096 }), handlePhotoAccess);
app.post('/v1/trial/challenge', handleTrialChallenge);
app.post('/v1/trial/challenge/verify', handleTrialChallengeVerify);
app.get('/v1/trial/status', handleTrialStatus);
app.post('/v1/auth/discord/start', handleDiscordAuthStart);
app.post('/v1/auth/qq/assert', handleQqAuthAssert);
app.post('/v1/telemetry/app-active-day', handleAppActiveDay);
app.post('/v1/providers/openrouter/issue', handleOpenRouterIssue);
app.post('/v1/providers/openrouter/discord/issue', handleDiscordOpenRouterIssue);
app.post('/v1/providers/openrouter/managed-key-delivery/ack', handleManagedKeyDeliveryAck);
app.post('/v1/providers/openrouter/managed-operation/status', handleManagedOperationStatus);
app.post('/v1/providers/openrouter/managed-operation/resume', handleManagedOperationResume);

export default app;
