import contract from './photo-contract.json';
import { beforeAbort, requestLifetime } from './lifetime';
import { readRequestJson } from './requests';
import { responseStream } from './stream';
import { isRecord } from './type-guards';
import { checkPhotoAllowance, photoPolicy, UsageEnv } from './usage';

export type PhotoEnv = UsageEnv & {
  OPENAI_API_KEY?: string;
  PHOTO_SERVICE_ENABLED?: string;
  PHOTO_POLICY?: string;
};

function error(status: number, code: string): Response {
  return Response.json({ error: { code } }, {
    status, headers: { 'cache-control': 'no-store', ...(code === 'service_busy' ? { 'retry-after': '1' } : {}) },
  });
}

export async function receivePhoto(
  request: Request, env: PhotoEnv, installation: string, context?: Pick<ExecutionContext, 'waitUntil'>,
): Promise<Response> {
  if (env.PHOTO_SERVICE_ENABLED !== 'true' || !env.OPENAI_API_KEY) return error(503, 'service_shutdown');
  let policy;
  try { policy = photoPolicy(env.PHOTO_POLICY); } catch { return error(503, 'service_shutdown'); }
  let body: unknown;
  try { body = await readRequestJson(request, policy.maxBodyBytes); } catch (caught) {
    return error(caught instanceof Error && caught.message === 'request_too_large' ? 413 : 400, 'invalid_photo_request');
  }
  const sized = isRecord(body) && body.photo_protocol === 2;
  const size: unknown = isRecord(body) && sized ? body.image_size : [1280, 720];
  if (!isRecord(body) || Object.keys(body).length !== (sized ? 5 : 4)
    || !Object.keys(body).every((key) => ['photo_protocol', 'request_id', 'image_base64', 'target_language', ...(sized ? ['image_size'] : [])].includes(key))
    || (body.photo_protocol !== 1 && !sized)
    || !Array.isArray(size) || size.length !== 2
    || !((size[0] === 1280 && size[1] === 720) || (size[0] === 720 && size[1] === 1280))
    || typeof body.request_id !== 'string' || !/^[a-f0-9]{32}$/.test(body.request_id)
    || typeof body.image_base64 !== 'string' || body.image_base64.length < 8
    || body.image_base64.length % 4 !== 0 || !body.image_base64.startsWith('/9j/')
    || !/^[A-Za-z0-9+/]*={0,2}$/.test(body.image_base64)
    || typeof body.target_language !== 'string' || !Object.hasOwn(contract.languages, body.target_language)) {
    return error(400, 'invalid_photo_request');
  }
  const language = contract.languages[body.target_language as keyof typeof contract.languages];
  let admission;
  try { admission = await checkPhotoAllowance(env, installation, body.request_id, policy); }
  catch { return error(503, 'service_shutdown'); }
  if ('error' in admission) return error(admission.error === 'photo_request_already_admitted' ? 409 : 429, admission.error);
  const { abort, release } = requestLifetime(admission.deadline, request.signal);
  const settle = (result: unknown) => {
    const pending = admission.settle(result);
    context?.waitUntil(pending);
    return pending;
  };
  try {
    const upstream = await beforeAbort(abort.signal, () => fetch('https://api.openai.com/v1/responses', {
      method: 'POST',
      headers: { authorization: `Bearer ${env.OPENAI_API_KEY}`, 'content-type': 'application/json' },
      redirect: 'manual',
      signal: abort.signal,
      body: JSON.stringify({
        model: 'gpt-6-luna', store: false, stream: true, reasoning: { effort: 'none' },
        safety_identifier: installation,
        instructions: contract.instructions.replaceAll('${targetName}', language),
        input: [{ type: 'message', role: 'user', content: [
          { type: 'input_image', image_url: `data:image/jpeg;base64,${body.image_base64}` },
          { type: 'input_text', text: `Target language: ${language}.\nThe image is ${size[0]}x${size[1]} pixels.` },
        ] }],
        text: { format: { type: 'json_schema', name: 'blocks', schema: contract.schema, strict: true } },
        max_output_tokens: policy.maxOutputTokens,
      }),
    }));
    if (upstream.ok && upstream.body && upstream.headers.get('content-type')?.split(';')[0] === 'text/event-stream') {
      return responseStream(upstream, abort, request.signal, settle, release, context);
    }
    abort.abort();
    await upstream.body?.cancel();
    await settle(undefined);
    return error(upstream.status === 429 ? 429 : upstream.status === 400 ? 400 : 502,
      upstream.status === 429 ? 'photo_quota_exhausted' : upstream.status === 400 ? 'photo_request_rejected' : 'photo_provider_unavailable');
  } catch {
    abort.abort();
    try { await settle(undefined); } catch { return error(503, 'service_shutdown'); }
    return error(502, 'photo_provider_unavailable');
  } finally {
    if (abort.signal.aborted) release();
  }
}
