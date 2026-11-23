import generatedModels from './models.generated.json';

// a model shown on the Models page
export interface PresetModel {
	repo: string;
	url: string;
	mmprojUrl?: string;
	// in bytes, model and mmproj together
	size: number;
	// input modalities of the mmproj, besides text
	modalities: ('image' | 'audio')[];
	// read from the chat template, with the detectors of the upstream models manager
	tools: boolean;
	thinking: boolean;
}

export interface LoadParams {
	nThreads: number; // -1 means auto
	nContext: number;
	nBatch: number;
}

export const DEFAULT_LOAD_PARAMS: LoadParams = {
	nBatch: 128,
	nContext: 16384,
	nThreads: -1
};

// edit models.json and run update-models.mjs to change this list (sizes are baked in to avoid HF API calls)
export const PRESET_MODELS = generatedModels as PresetModel[];
