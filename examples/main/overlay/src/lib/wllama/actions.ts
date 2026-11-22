// what the Models page does: calls WllamaServer, then keeps the app stores in sync
// load and unload call the server directly: it is the source of truth, and its /models/sse events update the app store

import { toastError } from './errors';
import { getWllamaServer } from './wllama-server';
import { goto } from '$app/navigation';
import { ROUTES } from '#lib/constants/index.ts';
import { modelsStore } from '#lib/stores/index.ts';
import { toast } from 'svelte-sonner';

// the model list of the app only has the downloaded models
const refreshAppModels = () => modelsStore.fetch(true).catch(() => {});

export async function downloadModel(id: string) {
	try {
		await getWllamaServer().download(id);
		toast.success(`Download finished: ${id}`);
		await refreshAppModels();
	} catch (e) {
		if ((e as Error)?.name === 'AbortError') {
			toast.info(`Download cancelled: ${id}`);
		} else {
			toastError(`Download failed: ${id}`, e);
		}
	}
}

export function cancelDownload(id: string) {
	getWllamaServer().cancelDownload(id);
}

export async function loadModel(id: string) {
	try {
		await getWllamaServer().load(id);
	} catch {
		return; // the server shows the error
	}
	toast.success(`Model loaded: ${id}`);
	// the model selector of the chat follows
	await modelsStore.selectModelById(id).catch(() => {});
}

export async function unloadModel(id: string) {
	try {
		const server = getWllamaServer();
		if (server.getLoadedId() === id) await server.unload();
	} catch (e) {
		toastError(`Cannot unload ${id}`, e);
	}
}

export async function startChat(id: string) {
	await modelsStore.selectModelById(id).catch(() => {});
	await goto(ROUTES.START);
}

export async function removeFromCache(id: string) {
	try {
		await getWllamaServer().removeFromCache(id);
		await refreshAppModels();
	} catch (e) {
		toastError(`Cannot remove ${id}`, e);
	}
}

export async function clearCache() {
	try {
		await getWllamaServer().clearCache();
		await refreshAppModels();
		toast.success('All downloaded models are removed');
	} catch (e) {
		toastError('Cannot clear the cache', e);
	}
}

export async function removeCustomModel(id: string) {
	await getWllamaServer().removeCustomModel(id);
}

export async function addCustomModel(url: string, mmprojUrl?: string) {
	await getWllamaServer().addCustomModel(url, mmprojUrl);
}
