import { app } from './app';
import { handleScheduled } from './scheduled';

const worker = {
  fetch: app.fetch.bind(app),
  request: app.request.bind(app),
  scheduled: handleScheduled,
};

export default worker;
