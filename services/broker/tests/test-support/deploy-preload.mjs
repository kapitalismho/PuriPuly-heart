import { preflightResponse } from '../../../test-support/worker-deployment-port.mjs';

globalThis.fetch = async (input, init) => {
  const response = preflightResponse(input, init, {
    worker: 'puripuly-heart-broker', hostname: 'puripuly-broker.salee.sarl', mode: process.env.MODE,
  });
  if (response) return response;
  throw new Error('Unexpected external request.');
};
