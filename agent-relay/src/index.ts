import { authenticate, hasTokenSecret } from './auth';
import { contracts, contractFor, supportedProtocols } from './contracts';
import { classifierInstructions, outOfScopeInstructions, socialInstructions, workInstructions } from './instructions';
import { beforeAbort, requestLifetime } from './lifetime';
import { ClassifyRequest, isClassifyRequest, isRespondRequest, maxClassifyBytes, maxRespondBytes, readRequestJson, RespondRequest } from './requests';
import { classifierQuestions } from './scope';
import { receiveSupport, SupportEnv } from './support';
import { responseStream } from './stream';
import { isRecord } from './type-guards';
import { Admission, checkAllowance, usageRecorder, UsageEnv } from './usage';

type Env = UsageEnv & SupportEnv & {
  OPENAI_API_KEY?: string;
  PURI_TOKEN_SECRET?: string;
  AGENT_SERVICE_ENABLED?: string;
};

const developerMessage = (text: string, breakpoint: Record<string, string> = { mode: 'explicit' }) => ({
  role: 'developer',
  content: [{ type: 'input_text', text, prompt_cache_breakpoint: breakpoint }],
});
const prepared = new Map([...contracts].map(([protocol, contract]) => [protocol, {
  classifierDeveloperMessage: developerMessage(classifierInstructions(contract)),
  questions: classifierQuestions(contract),
  workDeveloperMessage: developerMessage(workInstructions(contract), { mode: 'explicit', ttl: '12h' }),
  workOptions: {
    tools: contract.tools,
    reasoning: { effort: 'high' },
    prompt_cache_options: { mode: 'implicit', ttl: '30m' },
  },
}]));
const supported = supportedProtocols.join(', ');
const socialOptions = { instructions: socialInstructions, tools: [], reasoning: { effort: 'low' } };
const outOfScopeOptions = { instructions: outOfScopeInstructions, tools: [], reasoning: { effort: 'low' } };

function error(status: number, code: string, message: string): Response {
  const busy = code === 'service_busy';
  return Response.json({ error: { code, message, ...(busy ? { retry_after_ms: 1000 } : {}) } }, {
    status,
    headers: { 'cache-control': 'no-store', ...(busy ? { 'retry-after': '1' } : {}) },
  });
}

function capabilityError(body: unknown, status: number): boolean {
  if (status === 404) return true;
  if (!isRecord(body) || !isRecord(body.error)) return false;
  return ['model_not_found', 'unsupported_model', 'unsupported_parameter', 'unsupported_value']
    .includes(body.error.code as string);
}

export default {
  async fetch(request: Request, env: Env, context?: Pick<ExecutionContext, 'waitUntil'>): Promise<Response> {
    const url = new URL(request.url);
    const classification = url.pathname === '/v1/classify';
    const support = url.pathname === '/v1/support';
    if (!classification && !support && url.pathname !== '/v1/respond') {
      return error(404, 'not_found', 'Unknown relay endpoint.');
    }
    if (request.method !== 'POST') {
      return error(405, 'method_not_allowed', 'Use POST /v1/classify, /v1/respond or /v1/support.');
    }
    if (!hasTokenSecret(env.PURI_TOKEN_SECRET)) {
      return error(503, 'service_shutdown', 'The assistant service is not configured. Try again later.');
    }
    const installation = await authenticate(request, env.PURI_TOKEN_SECRET);
    if (!installation) {
      return error(401, 'service_authentication_failed', 'The assistant service could not authenticate. Reconnect through PuriPuly.');
    }
    if (support) return receiveSupport(request, env, installation);
    if (env.AGENT_SERVICE_ENABLED !== 'true') {
      return error(503, 'service_shutdown', 'The assistant service is stopped. Settings, Cancel and the static guide remain available.');
    }
    let body: unknown;
    try {
      body = await readRequestJson(request, classification ? maxClassifyBytes : maxRespondBytes);
    } catch (caught) {
      if (caught instanceof Error && caught.message === 'request_too_large') {
        return error(413, 'request_too_large', 'The agent request exceeds the supported byte limit.');
      }
      return error(400, 'invalid_request', 'Send a valid agent response request.');
    }
    if (!isRecord(body) || !contractFor(body.agent_protocol) || body.turn_id === undefined) {
      return error(400, 'unsupported_agent_protocol', `Update the app to use a supported agent protocol (${supported}).`);
    }
    if (!(classification ? isClassifyRequest(body) : isRespondRequest(body))) {
      return error(400, 'invalid_request', `Use a supported agent_protocol (${supported}) and its classification or response fields only.`);
    }
    if (!env.OPENAI_API_KEY) {
      return error(503, 'service_shutdown', 'The assistant service is not configured. Try again later.');
    }
    let admission: Admission;
    try {
      admission = await checkAllowance(env, installation, (body as ClassifyRequest | RespondRequest).turn_id, classification);
    } catch {
      return error(503, 'service_shutdown', 'Usage accounting is unavailable. No model request was started. Try again later.');
    }
    if ('error' in admission) {
      if (admission.error === 'service_turn_exhausted') {
        return error(429, admission.error, 'This prompt turn is expired or has reached its model request limit. Submit a new prompt.');
      }
      if (admission.error === 'service_busy') {
        return error(429, admission.error, 'An assistant request is already running. Wait for it to finish before trying again.');
      }
      return error(429, admission.error, 'The daily limit of 30 submitted prompts is reached for this UTC day. Settings, Cancel and the static guide remain available.');
    }
    const accounting = { ...admission.accounting, protocol: (body as ClassifyRequest | RespondRequest).agent_protocol };
    const input = (body as ClassifyRequest | RespondRequest).input;
    const contract = prepared.get((body as ClassifyRequest | RespondRequest).agent_protocol)!;
    const responseBody = body as RespondRequest;
    const social = !classification && responseBody.route === 'social';
    const outOfScope = !classification && responseBody.route === 'out_of_scope';
    const noTools = social || outOfScope;
    const upstreamBody = classification ? {
      model: 'gpt-6-luna',
      input: [contract.classifierDeveloperMessage, ...input],
      safety_identifier: installation,
      questions: contract.questions,
      prompt_cache_options: { mode: 'explicit', ttl: '30m' },
    } : {
      model: 'gpt-6-luna',
      safety_identifier: installation,
      ...(outOfScope ? outOfScopeOptions : social ? socialOptions : contract.workOptions),
      input: noTools ? input : [contract.workDeveloperMessage, ...input],
      store: false,
      stream: true,
      parallel_tool_calls: !noTools,
      tool_choice: responseBody.allow_tools && !noTools && !accounting.finalAdmission ? 'auto' : 'none',
      max_output_tokens: 25000,
      include: ['reasoning.encrypted_content'],
    };
    let text = '';
    let result: unknown;
    let upstream: Response | undefined;
    const settle = usageRecorder(accounting, classification, context);
    const { abort, release } = requestLifetime(accounting.deadline, request.signal);
    try {
      upstream = await beforeAbort(abort.signal, () => fetch(`https://api.openai.com/v1/${classification ? 'decisions' : 'responses'}`, {
        method: 'POST',
        headers: {
          authorization: `Bearer ${env.OPENAI_API_KEY}`,
          'content-type': 'application/json',
        },
        body: JSON.stringify(upstreamBody),
        signal: abort.signal,
      }));
      if (!classification && upstream.ok) {
        if (!upstream.body || upstream.headers.get('content-type')?.split(';')[0] !== 'text/event-stream') {
          const invalidBody = upstream.body;
          if (invalidBody) await beforeAbort(abort.signal, () => invalidBody.cancel());
          result = undefined;
        } else {
          return responseStream(upstream, abort, request.signal, settle, release, context);
        }
      } else {
        const reader = upstream.body?.getReader();
        if (!reader) throw new Error('upstream_unavailable');
        const stop = () => { void reader.cancel().catch(() => {}); };
        abort.signal.addEventListener('abort', stop, { once: true });
        if (abort.signal.aborted) stop();
        const decoder = new TextDecoder();
        try {
          while (true) {
            const { done, value } = await beforeAbort(abort.signal, () => reader.read());
            if (done) break;
            text += decoder.decode(value, { stream: true });
          }
          text += decoder.decode();
        } finally {
          abort.signal.removeEventListener('abort', stop);
          reader.releaseLock();
        }
        result = JSON.parse(text);
      }
    } catch {
      result = undefined;
    }
    abort.abort();
    release();
    try {
      await settle(result);
    } catch {
      return error(503, 'service_shutdown', 'Usage accounting is unavailable. This request may have incurred unrecorded usage. Try again later.');
    }
    if (upstream?.status === 401 || upstream?.status === 403) {
      return error(503, 'service_authentication_failed', 'The assistant service could not authenticate. Check the service connection.');
    }
    if (!upstream || result === undefined) {
      return error(502, 'upstream_unavailable', 'The model service could not complete this request.');
    }
    if (!upstream.ok) {
      const upstreamCode = isRecord(result) && isRecord(result.error) ? result.error.code : undefined;
      if (['insufficient_quota', 'billing_hard_limit_reached', 'usage_limit_reached'].includes(upstreamCode as string)) {
        return error(503, 'service_allowance_exhausted', 'The assistant service allowance is exhausted. Try again after it resets.');
      }
      if (capabilityError(result, upstream.status)) {
        return error(502, 'upstream_capability_unavailable', `The configured gpt-6-luna ${classification ? 'Decisions' : 'Responses'} capability is unavailable. No alternate model was used.`);
      }
      return error(502, 'upstream_unavailable', 'The model service could not complete this request.');
    }
    if (!isRecord(result) || (classification
      ? !Array.isArray(result.answers)
      : result.object !== 'response' || !Array.isArray(result.output))) {
      return error(502, 'upstream_invalid_response', 'The model service returned an invalid result.');
    }
    return new Response(text, {
      status: 200,
      headers: { 'content-type': 'application/json', 'cache-control': 'no-store' },
    });
  },
};
