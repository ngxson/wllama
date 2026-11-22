// reactive view of WllamaServer, for the Models page

import type { LoadParams } from './config';
import { getWllamaServer, type ModelEntry, type RuntimeInfo } from './wllama-server';
import { DEFAULT_LOAD_PARAMS } from './config';

class WllamaStore {
	models = $state.raw<ModelEntry[]>([]);
	loadedId = $state<string | null>(null);
	busy = $state(false);
	runtimeInfo = $state.raw<RuntimeInfo | null>(null);
	loadParams = $state.raw<LoadParams>(DEFAULT_LOAD_PARAMS);
	private started = false;

	get isDownloading(): boolean {
		return this.models.some((m) => m.downloadProgress >= 0);
	}

	start() {
		if (this.started) return;
		this.started = true;
		getWllamaServer().onChange(() => this.sync());
		this.sync();
	}

	private sync() {
		const server = getWllamaServer();
		// copy the entries: the server changes them in place
		this.models = server.getModelsNow().map((m) => ({ ...m }));
		this.loadedId = server.getLoadedId();
		this.busy = server.isBusy();
		this.runtimeInfo = server.getRuntimeInfo();
		this.loadParams = server.getLoadParams();
	}
}

export const wllamaStore = new WllamaStore();
