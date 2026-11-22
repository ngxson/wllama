// in-page "llama-server": owns the wllama instance and the model cache
// routes.ts maps the llama-server HTTP API onto this class; the Models page uses it directly

import { DEFAULT_LOAD_PARAMS, type LoadParams, PRESET_MODELS } from './config';
import { toastError } from './errors';
import { type Model, ModelManager, Wllama } from '@wllama/wllama';
import { detectThinkingSupport, detectToolUseSupport } from '#lib/utils/index.ts';
import { WLLAMA_COMPAT, WLLAMA_WASM } from 'virtual:wllama-build';

export type ModelStatus = 'unloaded' | 'loading' | 'loaded';

export interface ModelEntry {
	id: string;
	url: string;
	mmprojUrl?: string;
	size: number;
	// input modalities besides text, and tool call support: from the preset list, or read once the model is loaded
	modalities: string[];
	// null when not known yet (a custom model that was never loaded)
	tools: boolean | null;
	thinking: boolean | null;
	isUserAdded: boolean;
	cached?: Model;
	status: ModelStatus;
	// 0 to 1, -1 when not downloading
	downloadProgress: number;
}

type Caps = Pick<ModelEntry, 'modalities' | 'tools' | 'thinking'>;

export interface RuntimeInfo {
	isMultithread: boolean;
	nThreads: number;
	nCtx: number;
	hasChatTemplate: boolean;
	supportsImage: boolean;
	supportsAudio: boolean;
}

// one record of /models/sse, same shape as llama-server
export interface ModelsSseEvent {
	model: string;
	event: string;
	data?: Record<string, unknown>;
}

interface UserAddedModel {
	url: string;
	size: number;
	mmprojUrl?: string;
}

// same key and format as the old React example, so its custom models are kept
const CUSTOM_MODELS_KEY = 'custom_models';
const LOAD_PARAMS_KEY = 'wllama_load_params';
const MODEL_CAPS_KEY = 'wllama_model_caps';

function loadJson<T>(key: string, fallback: T): T {
	try {
		const raw = localStorage.getItem(key);
		return raw ? (JSON.parse(raw) as T) : fallback;
	} catch {
		return fallback;
	}
}

function saveJson(key: string, value: unknown) {
	localStorage.setItem(key, JSON.stringify(value));
}

// "org/file" for a HF URL (without .gguf and shard suffix), else the file name
export function modelIdFromUrl(url: string): string {
	const clean = url.split('?')[0];
	const file = (clean.split('/').pop() ?? clean)
		.replace(/\.gguf$/i, '')
		.replace(/-\d{5}-of-\d{5}$/, '');
	const org = clean.match(/^https?:\/\/(?:huggingface\.co|hf\.co)\/+([^/]+)\//);
	return org ? `${org[1]}/${file}` : file;
}

export class WllamaServer {
	readonly startedAt = Math.floor(Date.now() / 1000);
	private modelManager = new ModelManager();
	private wllama: Wllama | null = null;
	private entries: ModelEntry[] = [];
	private loadedId: string | null = null;
	private loadingId: string | null = null;
	private loadPromise: Promise<void> | null = null;
	private runtimeInfo: RuntimeInfo | null = null;
	private downloads = new Map<string, AbortController>();
	private loadParams: LoadParams = { ...DEFAULT_LOAD_PARAMS, ...loadJson(LOAD_PARAMS_KEY, {}) };
	private sseListeners = new Set<(ev: ModelsSseEvent) => void>();
	private changeListeners = new Set<() => void>();
	// capabilities read from the loaded models, they matter for the custom models
	private detectedCaps = new Map<string, Caps>(Object.entries(loadJson(MODEL_CAPS_KEY, {})));
	private ready: Promise<void>;

	constructor() {
		this.ready = this.refresh();
	}

	/////////////////////////////////////////////
	// state

	async getModels(): Promise<ModelEntry[]> {
		await this.ready;
		return this.entries;
	}

	getModelsNow(): ModelEntry[] {
		return this.entries;
	}

	getModel(id: string): ModelEntry | undefined {
		return this.entries.find((m) => m.id === id);
	}

	getLoadedId(): string | null {
		return this.loadedId;
	}

	getWllama(): Wllama | null {
		return this.loadedId ? this.wllama : null;
	}

	getRuntimeInfo(): RuntimeInfo | null {
		return this.runtimeInfo;
	}

	getLoadParams(): LoadParams {
		return { ...this.loadParams };
	}

	setLoadParams(params: LoadParams) {
		this.loadParams = { ...params };
		saveJson(LOAD_PARAMS_KEY, this.loadParams);
		this.notifyChange();
	}

	isBusy(): boolean {
		return this.loadingId !== null;
	}

	onChange(cb: () => void): () => void {
		this.changeListeners.add(cb);
		return () => this.changeListeners.delete(cb);
	}

	onSseEvent(cb: (ev: ModelsSseEvent) => void): () => void {
		this.sseListeners.add(cb);
		return () => this.sseListeners.delete(cb);
	}

	private notifyChange() {
		for (const cb of this.changeListeners) cb();
	}

	private emit(ev: ModelsSseEvent) {
		for (const cb of this.sseListeners) cb(ev);
	}

	private setStatus(id: string, status: ModelStatus, extra: Record<string, unknown> = {}) {
		const entry = this.getModel(id);
		if (entry) entry.status = status;
		this.emit({ data: { status, ...extra }, event: 'status_change', model: id });
		this.notifyChange();
	}

	/////////////////////////////////////////////
	// model list and cache

	// rebuild the list from the presets, the custom models and the cache
	async refresh(): Promise<void> {
		const cachedModels = await this.modelManager.getModels();
		const progress = new Map(this.entries.map((m) => [m.id, m.downloadProgress]));
		const entries: ModelEntry[] = [];
		const add = (m: UserAddedModel & Partial<Caps>, isUserAdded: boolean) => {
			const id = modelIdFromUrl(m.url);
			if (entries.some((e) => e.id === id)) return;
			const cached = cachedModels.find((c) => c.url === m.url);
			const detected = this.detectedCaps.get(id);
			entries.push({
				cached,
				downloadProgress: progress.get(id) ?? -1,
				id,
				isUserAdded,
				mmprojUrl: m.mmprojUrl ?? cached?.mmprojUrl,
				modalities: detected?.modalities ?? m.modalities ?? [],
				size: cached?.size ?? m.size,
				status: id === this.loadedId ? 'loaded' : id === this.loadingId ? 'loading' : 'unloaded',
				thinking: detected?.thinking ?? m.thinking ?? null,
				tools: detected?.tools ?? m.tools ?? null,
				url: m.url
			});
		};
		for (const m of loadJson<UserAddedModel[]>(CUSTOM_MODELS_KEY, [])) add(m, true);
		// sorted by size in models.generated.json
		for (const m of PRESET_MODELS) add(m, false);
		// cached models that are in no list (e.g. a custom model removed from the list)
		// getModels() also returns the mmproj files as models, skip them
		for (const c of cachedModels) {
			if (c.url !== c.mmprojUrl) add({ mmprojUrl: c.mmprojUrl, size: c.size, url: c.url }, true);
		}
		this.entries = entries;
		this.emit({ event: 'models_reload', model: '' });
		this.notifyChange();
	}

	async download(id: string): Promise<void> {
		const entry = this.getModel(id);
		if (!entry || entry.cached || this.downloads.has(id)) return;
		const abort = new AbortController();
		// refresh() replaces the entries, so always look up the current one
		const setProgress = (value: number) => {
			const current = this.getModel(id);
			if (current) current.downloadProgress = value;
			this.notifyChange();
		};
		this.downloads.set(id, abort);
		setProgress(0);
		try {
			await this.modelManager.downloadModel(
				{ mmprojUrl: entry.mmprojUrl, url: entry.url },
				{
					progressCallback: ({ loaded, total }) => setProgress(total > 0 ? loaded / total : 0),
					signal: abort.signal
				}
			);
		} finally {
			this.downloads.delete(id);
			setProgress(-1);
			await this.refresh();
		}
	}

	cancelDownload(id: string) {
		this.downloads.get(id)?.abort();
	}

	async removeFromCache(id: string): Promise<void> {
		const entry = this.getModel(id);
		if (!entry?.cached) return;
		if (this.loadedId === id) await this.unload();
		await entry.cached.remove();
		this.emit({ event: 'model_remove', model: id });
		await this.refresh();
	}

	async clearCache(): Promise<void> {
		await this.unload();
		await this.modelManager.clear();
		await this.refresh();
	}

	// checks that the URL is a GGUF file, then adds it to the custom models
	async addCustomModel(url: string, mmprojUrl?: string): Promise<void> {
		const cleanUrl = url.replace(/\?.*/, '');
		const res = await fetch(cleanUrl, { headers: { Range: 'bytes=0-3' } });
		if (!res.ok) throw new Error(`Fetch error with status code = ${res.status}`);
		const magic = new Uint8Array(await res.arrayBuffer()).slice(0, 4);
		if (String.fromCharCode(...magic) !== 'GGUF') {
			throw new Error('Not a valid GGUF file: it does not start with the GGUF magic number');
		}
		const size = await getRemoteSize(cleanUrl);
		const list = loadJson<UserAddedModel[]>(CUSTOM_MODELS_KEY, []);
		if (!list.some((m) => m.url === cleanUrl)) {
			list.unshift({ mmprojUrl, size, url: cleanUrl });
			saveJson(CUSTOM_MODELS_KEY, list);
		}
		await this.refresh();
	}

	async removeCustomModel(id: string): Promise<void> {
		const entry = this.getModel(id);
		if (!entry) return;
		const list = loadJson<UserAddedModel[]>(CUSTOM_MODELS_KEY, []);
		saveJson(
			CUSTOM_MODELS_KEY,
			list.filter((m) => m.url !== entry.url)
		);
		await this.refresh();
	}

	/////////////////////////////////////////////
	// load and unload

	// loads a cached model, the model loaded before is unloaded first (one model at a time)
	async load(id: string): Promise<void> {
		if (this.loadedId === id) return;
		if (this.loadPromise) {
			await this.loadPromise.catch(() => {});
			if (this.loadedId === id) return;
		}
		const entry = this.getModel(id);
		if (!entry) throw new Error(`Unknown model: ${id}`);
		if (!entry.cached) throw new Error(`Model is not downloaded: ${id}`);
		this.loadPromise = this.doLoad(entry);
		try {
			await this.loadPromise;
		} finally {
			this.loadPromise = null;
		}
	}

	private async doLoad(entry: ModelEntry) {
		await this.unload();
		this.loadingId = entry.id;
		this.setStatus(entry.id, 'loading');
		const wllama = new Wllama({ default: WLLAMA_WASM });
		wllama.setCompat(WLLAMA_COMPAT);
		try {
			const p = this.loadParams;
			await wllama.loadModel(entry.cached!, {
				n_batch: p.nBatch,
				n_ctx: p.nContext,
				n_threads: p.nThreads > 0 ? p.nThreads : undefined
			});
		} catch (e) {
			this.loadingId = null;
			await wllama.exit().catch(() => {});
			this.setStatus(entry.id, 'unloaded', { exit_code: 1 });
			// shown here: a load is started by the Models page or by a chat request
			toastError(`Cannot load ${entry.id}`, e);
			throw e;
		}
		this.wllama = wllama;
		this.loadedId = entry.id;
		this.loadingId = null;
		this.runtimeInfo = {
			hasChatTemplate: !!wllama.getChatTemplate(),
			isMultithread: wllama.isMultithread(),
			nCtx: wllama.getLoadedContextInfo().n_ctx,
			nThreads: wllama.getNumThreads(),
			supportsAudio: wllama.supportInputModality('audio'),
			supportsImage: wllama.supportInputModality('image')
		};
		const template = wllama.getChatTemplate() ?? '';
		const caps: Caps = {
			modalities: [
				...(this.runtimeInfo.supportsImage ? ['image'] : []),
				...(this.runtimeInfo.supportsAudio ? ['audio'] : [])
			],
			thinking: detectThinkingSupport(template),
			tools: detectToolUseSupport(template)
		};
		this.detectedCaps.set(entry.id, caps);
		saveJson(MODEL_CAPS_KEY, Object.fromEntries(this.detectedCaps));
		Object.assign(entry, caps);
		this.setStatus(entry.id, 'loaded');
	}

	async unload(): Promise<void> {
		const id = this.loadedId;
		if (!id || !this.wllama) return;
		const wllama = this.wllama;
		this.wllama = null;
		this.loadedId = null;
		this.runtimeInfo = null;
		await wllama.exit();
		this.setStatus(id, 'unloaded', { exit_code: 0 });
	}
}

async function getRemoteSize(url: string): Promise<number> {
	const shard = url.match(/^(.*)-(\d{5})-of-(\d{5})\.gguf$/);
	const urls = shard
		? Array.from(
				{ length: Number(shard[3]) },
				(_, i) => `${shard[1]}-${String(i + 1).padStart(5, '0')}-of-${shard[3]}.gguf`
			)
		: [url];
	const sizes = await Promise.all(
		urls.map(async (u) => {
			const res = await fetch(u, { method: 'HEAD' });
			if (!res.ok) throw new Error(`Fetch error with status code = ${res.status}`);
			return Number(res.headers.get('Content-Length') ?? 0);
		})
	);
	return sizes.reduce((a, b) => a + b, 0);
}

let server: WllamaServer | null = null;

// created on first use: the module is also imported by the server-side build
export function getWllamaServer(): WllamaServer {
	server ??= new WllamaServer();
	return server;
}
