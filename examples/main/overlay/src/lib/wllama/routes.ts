// the llama-server HTTP API (router mode), answered by WllamaServer
// only the routes that Llama Chat calls are here

import { errorMessage, toastError } from './errors';
import { getWllamaServer, type ModelEntry, type ModelsSseEvent } from './wllama-server';
import type { ChatCompletionParams } from '@wllama/wllama';
import { WLLAMA_VERSION } from 'virtual:wllama-build';

type Handler = (req: Request, url: URL) => Promise<Response> | Response;

const json = (data: unknown, status = 200) =>
	new Response(JSON.stringify(data), {
		headers: { 'Content-Type': 'application/json' },
		status
	});

const error = (message: string, status = 400, type = 'invalid_request_error') =>
	json({ error: { code: status, message, type } }, status);

const encoder = new TextEncoder();
const sse = (data: unknown) => encoder.encode(`data: ${JSON.stringify(data)}\n\n`);

// llama.cpp defaults, shown by the app as the "server default" of each setting
const DEFAULT_SAMPLING = {
	dry_allowed_length: 2,
	dry_base: 1.75,
	dry_multiplier: 0,
	dry_penalty_last_n: -1,
	dynatemp_exponent: 1,
	dynatemp_range: 0,
	frequency_penalty: 0,
	max_tokens: -1,
	min_keep: 0,
	min_p: 0.05,
	mirostat: 0,
	mirostat_eta: 0.1,
	mirostat_tau: 5,
	n_predict: -1,
	presence_penalty: 0,
	repeat_last_n: 64,
	repeat_penalty: 1,
	samplers: [
		'penalties',
		'dry',
		'top_n_sigma',
		'top_k',
		'typ_p',
		'top_p',
		'min_p',
		'xtc',
		'temperature'
	],
	stop: [],
	temperature: 0.8,
	top_k: 40,
	top_n_sigma: -1,
	top_p: 0.95,
	typ_p: 1,
	xtc_probability: 0,
	xtc_threshold: 0.1
};

function toApiModel(m: ModelEntry, created: number) {
	return {
		architecture: { input_modalities: ['text', ...m.modalities] },
		can_remove: true,
		created,
		id: m.id,
		in_cache: true,
		object: 'model',
		owned_by: 'wllama',
		path: m.url,
		sidecars: m.mmprojUrl ? ['mmproj'] : [],
		source: 'cache',
		status: { value: m.status }
	};
}

const getProps: Handler = async (_req, url) => {
	const server = getWllamaServer();
	await server.getModels();
	const modelId = url.searchParams.get('model');
	const entry = modelId ? server.getModel(modelId) : undefined;
	if (modelId && !entry) return error(`model not found: ${modelId}`, 404, 'not_found_error');
	const info = entry && server.getLoadedId() === entry.id ? server.getRuntimeInfo() : null;
	const wllama = info ? server.getWllama() : null;
	return json({
		build_info: `wllama ${WLLAMA_VERSION}`,
		chat_template: wllama?.getChatTemplate() ?? '',
		default_generation_settings: {
			n_ctx: info?.nCtx ?? server.getLoadParams().nContext,
			params: DEFAULT_SAMPLING
		},
		modalities: {
			audio: entry ? entry.modalities.includes('audio') : false,
			video: false,
			vision: entry ? entry.modalities.includes('image') : false
		},
		model_alias: entry?.id,
		model_path: entry?.url ?? '',
		role: entry ? 'model' : 'router',
		total_slots: 1,
		// defaults for a new user: the Models page is where the models are downloaded
		// MCP runs in the browser (no llama-server needed), only the servers that allow CORS can be used
		ui_settings: { showMcp: true, showModelsManager: true }
	});
};

const listModels: Handler = async () => {
	const server = getWllamaServer();
	// only the downloaded models can be used, the others are on the Models page
	const models = (await server.getModels()).filter((m) => m.cached);
	return json({ data: models.map((m) => toApiModel(m, server.startedAt)), object: 'list' });
};

const modelEvents: Handler = (req) => {
	const server = getWllamaServer();
	let unsubscribe = () => {};
	const body = new ReadableStream<Uint8Array>({
		cancel: () => unsubscribe(),
		start(controller) {
			const send = (ev: ModelsSseEvent) => {
				try {
					controller.enqueue(sse(ev));
				} catch {
					unsubscribe();
				}
			};
			unsubscribe = server.onSseEvent(send);
			req.signal.addEventListener('abort', () => {
				unsubscribe();
				controller.close();
			});
			for (const m of server.getModelsNow()) {
				if (m.status !== 'unloaded') {
					send({ data: { status: m.status }, event: 'model_status', model: m.id });
				}
			}
		}
	});
	return new Response(body, { headers: { 'Content-Type': 'text/event-stream' } });
};

const loadModel: Handler = async (req) => {
	const { model } = await req.json();
	try {
		await getWllamaServer().load(model);
		return json({ success: true });
	} catch (e) {
		return json({ error: (e as Error).message, success: false }, 500);
	}
};

const unloadModel: Handler = async (req) => {
	const { model } = await req.json();
	const server = getWllamaServer();
	if (server.getLoadedId() === model) await server.unload();
	return json({ success: true });
};

// DELETE /models: cancel a download or remove the model files
const deleteModel: Handler = async (req) => {
	const { model } = await req.json();
	const server = getWllamaServer();
	server.cancelDownload(model);
	await server.removeFromCache(model);
	return json({ success: true });
};

// turns the OpenAI media parts into the parts that wllama takes (raw bytes)
async function convertContent(content: unknown): Promise<unknown> {
	if (!Array.isArray(content)) return content;
	return Promise.all(
		content.map(async (part) => {
			if (part?.type === 'image_url') {
				const res = await fetch(part.image_url.url);
				return { data: await res.arrayBuffer(), type: 'image' };
			}
			if (part?.type === 'input_audio') {
				const res = await fetch(`data:application/octet-stream;base64,${part.input_audio.data}`);
				return { data: await res.arrayBuffer(), type: 'audio' };
			}
			if (part?.type === 'input_video') {
				throw new Error('Video input is not supported yet');
			}
			return part;
		})
	);
}

const chatCompletions: Handler = async (req) => {
	const server = getWllamaServer();
	const body = await req.json();
	const modelId: string | undefined = body.model ?? server.getLoadedId() ?? undefined;
	if (!modelId) return error('No model is loaded, load one on the Models page', 400);

	try {
		// like the router, load the model asked by the request (the server shows a load error)
		await server.load(modelId);
	} catch (e) {
		return error(errorMessage(e), 400);
	}
	try {
		for (const msg of body.messages ?? []) {
			msg.content = await convertContent(msg.content);
		}
	} catch (e) {
		toastError('Cannot read the attachment', e);
		return error(errorMessage(e), 400);
	}

	const wllama = server.getWllama();
	if (!wllama) return error('Model is not loaded', 503, 'unavailable_error');

	const abort = new AbortController();
	req.signal.addEventListener('abort', () => abort.abort());

	if (!body.stream) {
		try {
			const result = await wllama.createChatCompletion({
				...body,
				abortSignal: abort.signal,
				stream: false
			});
			return json({ ...result, model: modelId });
		} catch (e) {
			toastError('Chat completion failed', e);
			return error(errorMessage(e), 500, 'server_error');
		}
	}

	const params: ChatCompletionParams & { stream: true } = {
		...body,
		abortSignal: abort.signal,
		stream: true
	};
	const chunks = (await wllama.createChatCompletion(params))[Symbol.asyncIterator]();
	// wait for the first chunk: an error at the start (e.g. a bad grammar) becomes an HTTP error, like llama-server
	// an error in the stream is shown by the app as a lost connection, without the message
	let first: IteratorResult<unknown>;
	try {
		first = await chunks.next();
	} catch (e) {
		if (abort.signal.aborted) return error('Request aborted', 400);
		toastError('Chat completion failed', e);
		return error(errorMessage(e), 500, 'server_error');
	}
	const stream = new ReadableStream<Uint8Array>({
		cancel: () => abort.abort(),
		async start(controller) {
			try {
				for (let r = first; !r.done; r = await chunks.next()) {
					controller.enqueue(sse({ ...(r.value as object), model: modelId }));
				}
				controller.enqueue(encoder.encode('data: [DONE]\n\n'));
				controller.close();
			} catch (e) {
				if (abort.signal.aborted) {
					controller.close();
				} else {
					toastError('Chat completion failed', e);
					controller.error(e);
				}
			}
		}
	});
	return new Response(stream, { headers: { 'Content-Type': 'text/event-stream' } });
};

// key: "METHOD /path"
const ROUTES: Record<string, Handler> = {
	'DELETE /models': deleteModel,
	'GET /models': listModels,
	'GET /models/sse': modelEvents,
	'GET /props': getProps,
	'GET /slots': () => json([]),
	'GET /tools': () => json([]),
	'GET /v1/models': listModels,
	'POST /models': () => error('Download the models on the Models page', 400),
	'POST /models/load': loadModel,
	'POST /models/unload': unloadModel,
	'POST /v1/chat/completions': chatCompletions,
	'POST /v1/streams/lookup': () => json([])
};

// API paths with no handler get a 404, the others go to the network (static files)
const API_PREFIXES = ['/v1', '/models', '/props', '/slots', '/tools', '/cors-proxy'];

export function isApiPath(path: string): boolean {
	return API_PREFIXES.some((p) => path === p || path.startsWith(p + '/'));
}

export async function handleApiRequest(req: Request, path: string): Promise<Response> {
	const handler = ROUTES[`${req.method} ${path}`];
	if (!handler)
		return error(`${req.method} ${path} is not supported by wllama`, 404, 'not_found_error');
	try {
		return await handler(req, new URL(req.url));
	} catch (e) {
		toastError(`${req.method} ${path} failed`, e);
		return error(errorMessage(e), 500, 'server_error');
	}
}
